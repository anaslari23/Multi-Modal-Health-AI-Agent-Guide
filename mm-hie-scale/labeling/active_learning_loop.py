import argparse
import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests


logger = logging.getLogger(__name__)


@dataclass
class EpisodePrediction:
    episode_id: str
    probs: List[float]  # posterior over classes
    metadata: Dict[str, Any]


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def entropy(probs: List[float]) -> float:
    eps = 1e-12
    return -sum(p * math.log(p + eps) for p in probs if p > 0)


def load_fusion_predictions(path: Path) -> List[EpisodePrediction]:
    """Load fusion model predictions from JSONL or JSON.

    Expected format per record:
      {"episode_id": str, "probs": [float, ...], "metadata": {...}}
    """
    if not path.exists():
        raise FileNotFoundError(f"Fusion predictions file not found: {path}")

    preds: List[EpisodePrediction] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                preds.append(
                    EpisodePrediction(
                        episode_id=str(obj["episode_id"]),
                        probs=list(obj["probs"]),
                        metadata=obj.get("metadata", {}),
                    )
                )
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for obj in data:
            preds.append(
                EpisodePrediction(
                    episode_id=str(obj["episode_id"]),
                    probs=list(obj["probs"]),
                    metadata=obj.get("metadata", {}),
                )
            )

    return preds


def sample_high_uncertainty(
    preds: List[EpisodePrediction],
    k: int,
) -> List[EpisodePrediction]:
    scored = [
        (entropy(p.probs), p) for p in preds
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


# ---------------- Label Studio API helpers ----------------


def get_labelstudio_client() -> Dict[str, Any]:
    """Return Label Studio connection info from env.

    Requires LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY.
    """
    url = os.getenv("LABEL_STUDIO_URL")
    api_key = os.getenv("LABEL_STUDIO_API_KEY")
    if not url or not api_key:
        logger.warning("LABEL_STUDIO_URL or LABEL_STUDIO_API_KEY not set; API calls will be skipped")
    return {"url": url, "api_key": api_key}


def push_tasks_to_labelstudio(
    project_id: int,
    episodes: List[EpisodePrediction],
    ls_client: Optional[Dict[str, Any]] = None,
) -> None:
    if ls_client is None:
        ls_client = get_labelstudio_client()

    url = ls_client.get("url")
    api_key = ls_client.get("api_key")
    if not url or not api_key:
        logger.warning("Label Studio client not configured; skipping push")
        return

    tasks = []
    for ep in episodes:
        data = ep.metadata.copy()
        data["episode_id"] = ep.episode_id
        tasks.append({"data": data})

    endpoint = f"{url.rstrip('/')}/api/projects/{project_id}/import"
    headers = {"Authorization": f"Token {api_key}"}

    try:
        resp = requests.post(endpoint, headers=headers, json=tasks, timeout=30)
        resp.raise_for_status()
        logger.info("Pushed %d tasks to Label Studio project %s", len(tasks), project_id)
    except Exception:
        logger.exception("Failed to push tasks to Label Studio")


def pull_annotations_from_labelstudio(
    project_id: int,
    ls_client: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if ls_client is None:
        ls_client = get_labelstudio_client()

    url = ls_client.get("url")
    api_key = ls_client.get("api_key")
    if not url or not api_key:
        logger.warning("Label Studio client not configured; skipping pull")
        return []

    endpoint = f"{url.rstrip('/')}/api/projects/{project_id}/export?exportType=JSON"
    headers = {"Authorization": f"Token {api_key}"}

    try:
        resp = requests.get(endpoint, headers=headers, timeout=60)
        resp.raise_for_status()
        annotations = resp.json()
        logger.info("Pulled %d annotations from Label Studio project %s", len(annotations), project_id)
        return annotations
    except Exception:
        logger.exception("Failed to pull annotations from Label Studio")
        return []


# ---------------- Merge / manifest helpers ----------------


def merge_annotations(
    gold_labels_path: Path,
    new_annotations: List[Dict[str, Any]],
) -> None:
    """Merge Label Studio annotations into a gold label JSONL file.

    Very simple union-by-episode_id; extend as needed.
    """
    existing: Dict[str, Dict[str, Any]] = {}
    if gold_labels_path.exists():
        with gold_labels_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                eid = str(obj.get("episode_id"))
                existing[eid] = obj

    for ann in new_annotations:
        data = ann.get("data", {})
        eid = str(data.get("episode_id"))
        if not eid:
            continue
        existing[eid] = data

    gold_labels_path.parent.mkdir(parents=True, exist_ok=True)
    with gold_labels_path.open("w", encoding="utf-8") as f:
        for eid, obj in existing.items():
            record = obj.copy()
            record["episode_id"] = eid
            f.write(json.dumps(record) + "\n")

    logger.info("Merged %d annotations into %s", len(existing), gold_labels_path)


def update_training_manifest(
    manifest_path: Path,
    gold_labels_path: Path,
) -> None:
    """Update a simple training manifest pointing to gold label file.

    Format is a JSON file, e.g. {"gold_labels": "path/to/gold.jsonl"}.
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"gold_labels": str(gold_labels_path)}
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Updated training manifest at %s", manifest_path)


# ---------------- CLI subcommands ----------------


def cmd_sample(args: argparse.Namespace) -> None:
    preds = load_fusion_predictions(args.fusion_predictions)
    selected = sample_high_uncertainty(preds, args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for ep in selected:
            f.write(
                json.dumps(
                    {
                        "episode_id": ep.episode_id,
                        "probs": ep.probs,
                        "metadata": ep.metadata,
                    }
                )
                + "\n"
            )
    logger.info("Sampled %d high-uncertainty episodes to %s", len(selected), args.output)


def cmd_push(args: argparse.Namespace) -> None:
    # Load sampled episodes
    eps: List[EpisodePrediction] = []
    with args.sampled_episodes.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            eps.append(
                EpisodePrediction(
                    episode_id=str(obj["episode_id"]),
                    probs=list(obj.get("probs", [])),
                    metadata=obj.get("metadata", {}),
                )
            )
    push_tasks_to_labelstudio(args.project_id, eps)


def cmd_pull(args: argparse.Namespace) -> None:
    anns = pull_annotations_from_labelstudio(args.project_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(anns, f, indent=2)
    logger.info("Saved pulled annotations to %s", args.output)


def cmd_merge(args: argparse.Namespace) -> None:
    with args.annotations.open("r", encoding="utf-8") as f:
        anns = json.load(f)
    merge_annotations(args.gold_labels, anns)
    update_training_manifest(args.manifest, args.gold_labels)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Active learning loop with Label Studio")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # sample
    p_sample = subparsers.add_parser("sample", help="Sample high-uncertainty episodes")
    p_sample.add_argument("fusion_predictions", type=Path, help="Fusion model predictions (JSON or JSONL)")
    p_sample.add_argument("--top-k", type=int, default=100, help="Number of episodes to sample")
    p_sample.add_argument("--output", type=Path, default=Path("mm-hie-scale/labeling/sampled_episodes.jsonl"))
    p_sample.set_defaults(func=cmd_sample)

    # push
    p_push = subparsers.add_parser("push", help="Push sampled episodes as tasks to Label Studio")
    p_push.add_argument("project_id", type=int, help="Label Studio project ID")
    p_push.add_argument(
        "--sampled-episodes",
        type=Path,
        default=Path("mm-hie-scale/labeling/sampled_episodes.jsonl"),
        help="Sampled episodes JSONL from the sample step",
    )
    p_push.set_defaults(func=cmd_push)

    # pull
    p_pull = subparsers.add_parser("pull", help="Pull completed annotations from Label Studio")
    p_pull.add_argument("project_id", type=int, help="Label Studio project ID")
    p_pull.add_argument(
        "--output",
        type=Path,
        default=Path("mm-hie-scale/labeling/annotations.json"),
        help="Where to save pulled annotations",
    )
    p_pull.set_defaults(func=cmd_pull)

    # merge
    p_merge = subparsers.add_parser("merge", help="Merge annotations into gold labels and update manifest")
    p_merge.add_argument(
        "annotations",
        type=Path,
        default=Path("mm-hie-scale/labeling/annotations.json"),
        help="Annotations JSON from pull step",
    )
    p_merge.add_argument(
        "--gold-labels",
        type=Path,
        default=Path("mm-hie-scale/labeling/gold_labels.jsonl"),
        help="Gold label JSONL file to write",
    )
    p_merge.add_argument(
        "--manifest",
        type=Path,
        default=Path("mm-hie-scale/labeling/training_manifest.json"),
        help="Training manifest JSON to update",
    )
    p_merge.set_defaults(func=cmd_merge)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    try:
        args.func(args)
    except Exception:
        logger.exception("Active learning command failed")
        raise


if __name__ == "__main__":
    main()

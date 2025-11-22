from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FUSION_DIR = DATA_DIR / "fusion"


def _load_or_random(path: Path, key: str, shape: int, n: int) -> np.ndarray:
    if path.exists():
        arr = np.load(path)
        if key in arr:
            data = arr[key]
            if data.shape[0] >= n:
                return data[:n]
    # fallback: random embeddings
    return np.random.normal(size=(n, shape)).astype("float32")


def make_fusion_from_embeddings(
    out_path: str,
    n_samples: int,
    num_labels: int,
    nlp_path: str | None = None,
    img_path: str | None = None,
    labs_path: str | None = None,
    vitals_path: str | None = None,
) -> None:
    # Default locations for modality embeddings
    nlp_file = Path(nlp_path) if nlp_path else DATA_DIR / "nlp" / "embeddings.npz"
    img_file = Path(img_path) if img_path else DATA_DIR / "imaging" / "img_embeddings.npz"
    labs_file = Path(labs_path) if labs_path else DATA_DIR / "labs" / "labs_embeddings.npz"
    vitals_file = Path(vitals_path) if vitals_path else DATA_DIR / "vitals" / "vitals_embeddings.npz"

    # Load or synthesize each modality
    nlp = _load_or_random(nlp_file, "emb", 768, n_samples)
    img = _load_or_random(img_file, "emb", 512, n_samples)
    labs = _load_or_random(labs_file, "emb", 50, n_samples)
    vitals = _load_or_random(vitals_file, "emb", 128, n_samples)

    # Multi-label indicator vectors and a simple scalar risk
    labels = (np.random.rand(n_samples, num_labels) < 0.1).astype("float32")
    risk = labels.mean(axis=1) + 0.1 * np.random.randn(n_samples).astype("float32")

    records: List[Dict[str, Any]] = []
    for i in range(n_samples):
        records.append(
            {
                "nlp": nlp[i],
                "img": img[i],
                "labs": labs[i],
                "vitals": vitals[i],
                "labels": labels[i],
                "risk": float(risk[i]),
            }
        )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_file, records=np.array(records, dtype=object))
    print(f"Wrote fusion embeddings dataset to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(FUSION_DIR / "train.npz"))
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--num_labels", type=int, default=50)
    parser.add_argument("--nlp_emb")
    parser.add_argument("--img_emb")
    parser.add_argument("--labs_emb")
    parser.add_argument("--vitals_emb")
    args = parser.parse_args()

    make_fusion_from_embeddings(
        out_path=args.out,
        n_samples=args.n_samples,
        num_labels=args.num_labels,
        nlp_path=args.nlp_emb,
        img_path=args.img_emb,
        labs_path=args.labs_emb,
        vitals_path=args.vitals_emb,
    )


if __name__ == "__main__":
    main()

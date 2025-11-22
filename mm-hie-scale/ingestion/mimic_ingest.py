import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from common_normalizer import (
    build_episode,
    deidentify_record,
    write_episode_json,
    ensure_directory,
)


logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def run_mimic_ingestion(
    raw_root: Path,
    processed_root: Path,
    max_episodes: Optional[int] = None,
    diagnoses_path_override: Optional[Path] = None,
) -> None:
    """Ingest MIMIC-IV into canonical episodes.

    This is a schema-agnostic stub: wire it to real MIMIC-IV tables
    (e.g. admissions, icustays, chartevents, labevents, noteevents) and
    construct episodes per hospital stay or ICU stay.
    """
    raw_mimic_root = raw_root / "mimic"
    output_dir = processed_root / "mimic"
    ensure_directory(output_dir)

    if diagnoses_path_override is not None:
        diagnoses_path = diagnoses_path_override
    else:
        if not raw_mimic_root.exists():
            logger.warning("Raw MIMIC root does not exist: %s", raw_mimic_root)
            return
        diagnoses_path = raw_mimic_root / "hosp" / "diagnoses_icd.csv"
    if not diagnoses_path.exists():
        logger.warning("MIMIC diagnoses file not found at %s", diagnoses_path)
        return

    logger.info("Starting MIMIC-IV ingestion from %s to %s", diagnoses_path, output_dir)

    try:
        df = pd.read_csv(diagnoses_path)
    except Exception:
        logger.exception("Failed to read diagnoses file: %s", diagnoses_path)
        return

    required_cols = {"hadm_id", "icd_code", "icd_version"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("Diagnoses file missing expected columns %s", ", ".join(sorted(missing)))
        return

    df = df[df["icd_version"].astype(str) == "10"]
    if df.empty:
        logger.info("No ICD-10 diagnoses found in %s", diagnoses_path)
        return

    episodes_processed = 0

    for hadm_id, group in df.groupby("hadm_id"):
        if max_episodes is not None and episodes_processed >= max_episodes:
            break

        try:
            episode_id = str(hadm_id)
            labels = []
            for _, row in group.iterrows():
                code = str(row["icd_code"]).strip()
                if not code:
                    continue
                labels.append(
                    {
                        "system": "ICD-10",
                        "code": code,
                        "description": "",
                    }
                )

            if not labels:
                continue

            episode = build_episode(
                episode_id=episode_id,
                notes=[],
                labs=[],
                vitals=[],
                imaging=[],
                labels=labels,
                timestamps={},
            )
            write_episode_json(episode, output_dir)
            episodes_processed += 1
        except Exception:
            logger.exception("Failed to process MIMIC episode for hadm_id=%s", hadm_id)

    logger.info("Finished MIMIC-IV ingestion, episodes=%d", episodes_processed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest MIMIC-IV into canonical episodes")
    parser.add_argument("--raw-root", type=Path, default=Path("raw"), help="Root of raw data lake (expects raw/mimic)")
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("processed"),
        help="Root of processed data lake (writes processed/mimic)",
    )
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional cap on number of episodes")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    parser.add_argument(
        "--diagnoses-path",
        type=Path,
        default=None,
        help="Optional explicit path to diagnoses_icd.csv (overrides raw-root)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    try:
        run_mimic_ingestion(
            raw_root=args.raw_root,
            processed_root=args.processed_root,
            max_episodes=args.max_episodes,
            diagnoses_path_override=args.diagnoses_path,
        )
    except Exception:
        logger.exception("Unhandled exception in MIMIC ingestion")
        raise


if __name__ == "__main__":
    main()

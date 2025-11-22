import argparse
import logging
from pathlib import Path
from typing import Dict, Type

from base_ingestor import BaseIngestor
from chexpert_ingestor import CheXpertIngestor
from nih_chestxray14_ingestor import NIHChestXray14Ingestor
from medquad_ingestor import MedQuadIngestor
from meddialog_ingestor import MedDialogIngestor
from pubmedqa_ingestor import PubMedQAIngestor
from pubmed_ingestor import PubMedAbstractsIngestor
from fda_drugs_ingestor import FDADrugLabelsIngestor
from mimic_cxr_ingestor import MIMICCXRIngestor
from ptbxl_ingestor import PTBXLIngestor
from ham10000_ingestor import HAM10000Ingestor


logger = logging.getLogger(__name__)


INGESTORS: Dict[str, Type[BaseIngestor]] = {
    "chexpert": CheXpertIngestor,
    "nih_chestxray14": NIHChestXray14Ingestor,
    "medquad": MedQuadIngestor,
    "meddialog": MedDialogIngestor,
    "pubmedqa": PubMedQAIngestor,
    "pubmed_abstracts": PubMedAbstractsIngestor,
    "fda_drug_labels": FDADrugLabelsIngestor,
    "mimic_cxr": MIMICCXRIngestor,
    "ptbxl": PTBXLIngestor,
    "ham10000": HAM10000Ingestor,
}


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )


def run_single(dataset_key: str, args: argparse.Namespace) -> None:
    ingestor_cls = INGESTORS.get(dataset_key)
    if ingestor_cls is None:
        raise ValueError(f"Unknown dataset key: {dataset_key}. Valid keys: {sorted(INGESTORS.keys())}")

    ingestor = ingestor_cls(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        split=args.split,
        max_episodes=args.max_episodes,
    )
    logger.info("Running ingestion for dataset=%s", dataset_key)
    ingestor.download()
    samples = ingestor.transform()
    ingestor.canonicalize(samples)
    ingestor.write_output()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified HF ingestion manager for MM-HIE")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset key to ingest (or use --all)")
    parser.add_argument("--all", action="store_true", help="Run ingestion for all supported datasets")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", type=str, default=None, help="Optional HF split to use (e.g. train, validation)")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if args.all:
        for key in sorted(INGESTORS.keys()):
            run_single(key, args)
    else:
        if not args.dataset:
            raise SystemExit("Must provide --dataset or --all")
        run_single(args.dataset, args)


if __name__ == "__main__":
    main()

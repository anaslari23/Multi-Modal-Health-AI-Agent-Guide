import abc
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset

from .canonicalizer import Canonicalizer


logger = logging.getLogger(__name__)


@dataclass
class EpisodeRecord:
    episode_id: str
    source: str
    text: Dict[str, Any]
    imaging: Dict[str, Any]
    labs: Dict[str, Any]
    vitals: Dict[str, Any]
    diagnosis_labels: List[Dict[str, Any]] = field(default_factory=list)
    drug_labels: List[Dict[str, Any]] = field(default_factory=list)
    ontology_codes: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embeddings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseIngestor(abc.ABC):
    """Abstract base class for HuggingFace ingestion.

    Typical lifecycle:
      1) download()   -> populate self.hf_dataset
      2) transform()  -> lightweight restructuring / field selection
      3) canonicalize() -> convert to EpisodeRecord instances
      4) write_output() -> write JSON episodes + manifest
    """

    dataset_name: str

    def __init__(
        self,
        raw_root: Path,
        processed_root: Path,
        split: Optional[str] = None,
        max_episodes: Optional[int] = None,
    ) -> None:
        self.raw_root = raw_root
        self.processed_root = processed_root
        self.split = split
        self.max_episodes = max_episodes

        self.hf_dataset: Optional[DatasetDict | Dataset] = None
        self.canonicalizer = Canonicalizer()

        self.episodes: List[EpisodeRecord] = []
        self.manifest: Dict[str, Any] = {}

    # -------------------------- Core stages --------------------------

    @abc.abstractmethod
    def dataset_name(self) -> str:  # type: ignore[override]
        """Return the HuggingFace dataset identifier (e.g. "StanfordAIMI/chexpert")."""

    def download(self) -> None:
        """Download dataset from HuggingFace using `datasets.load_dataset`.

        Subclasses may override to customize config or streaming.
        """
        logger.info("Loading dataset %s from HuggingFace", self.dataset_name())
        if self.split is None:
            self.hf_dataset = load_dataset(self.dataset_name())
        else:
            self.hf_dataset = load_dataset(self.dataset_name(), split=self.split)

    @abc.abstractmethod
    def transform(self) -> Iterable[Dict[str, Any]]:
        """Yield intermediate sample dicts.

        Each sample should be a simple dict that `canonicalize_sample` can
        convert into an EpisodeRecord. This is where you select columns
        and basic fields from the raw HF dataset.
        """

    def canonicalize(self, samples: Iterable[Dict[str, Any]]) -> None:
        """Convert intermediate samples into canonical EpisodeRecord objects."""
        episodes: List[EpisodeRecord] = []
        for i, sample in enumerate(samples):
            if self.max_episodes is not None and i >= self.max_episodes:
                break
            try:
                episode = self.canonicalize_sample(sample)
                episodes.append(episode)
            except Exception:
                logger.exception("Failed to canonicalize sample index=%d", i)
        self.episodes = episodes

    @abc.abstractmethod
    def canonicalize_sample(self, sample: Dict[str, Any]) -> EpisodeRecord:
        """Convert a single transformed sample into an EpisodeRecord.

        Use Canonicalizer helpers for text, imaging, vitals, labels, etc.
        """

    def write_output(self) -> None:
        """Write canonical episodes and manifest to the data lake."""
        if not self.episodes:
            logger.warning("No episodes to write for dataset %s", self.dataset_name())
            return

        dataset_key = self.dataset_name().split("/")[-1]
        episodes_dir = self.processed_root / dataset_key / "episodes"
        manifest_path = self.processed_root / dataset_key / "manifest.json"

        episodes_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        for ep in self.episodes:
            out_path = episodes_dir / f"{ep.episode_id}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(ep.to_dict(), f, ensure_ascii=False)

        self.manifest = self._build_manifest(self.episodes)

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(self.manifest, f, ensure_ascii=False, indent=2)

        logger.info(
            "Wrote %d episodes and manifest to %s",
            len(self.episodes),
            manifest_path.parent,
        )

    # -------------------------- Helpers --------------------------

    def _build_manifest(self, episodes: List[EpisodeRecord]) -> Dict[str, Any]:
        """Compute dataset-level statistics for manifest.json."""
        label_counter: Counter[str] = Counter()
        icd_counter: Counter[str] = Counter()
        snomed_counter: Counter[str] = Counter()
        text_count = 0
        image_count = 0

        for ep in episodes:
            if ep.text:
                text_count += 1
            if ep.imaging.get("path") or ep.imaging.get("paths"):
                image_count += 1

            for lab in ep.diagnosis_labels:
                key = f"{lab.get('system','')}:" f"{lab.get('code','')}".strip(":")
                if key:
                    label_counter[key] += 1

            codes = ep.ontology_codes or {}
            for code in codes.get("ICD10", []) or []:
                icd_counter[code] += 1
            for code in codes.get("SNOMED", []) or []:
                snomed_counter[code] += 1

        manifest: Dict[str, Any] = {
            "num_episodes": len(episodes),
            "text_episode_count": text_count,
            "image_episode_count": image_count,
            "label_distribution": dict(label_counter),
            "icd10_distribution": dict(icd_counter),
            "snomed_distribution": dict(snomed_counter),
            "missing_values": self._compute_missing_stats(episodes),
        }
        return manifest

    def _compute_missing_stats(self, episodes: List[EpisodeRecord]) -> Dict[str, Any]:
        """Simple missing-value analysis over top-level fields."""
        fields = [
            "text",
            "imaging",
            "labs",
            "vitals",
            "embeddings",
            "diagnosis_labels",
            "drug_labels",
            "ontology_codes",
        ]
        counts: Dict[str, int] = defaultdict(int)
        total = len(episodes)

        for ep in episodes:
            d = ep.to_dict()
            for f in fields:
                if not d.get(f):
                    counts[f] += 1

        return {f: counts[f] / total if total else 0.0 for f in fields}

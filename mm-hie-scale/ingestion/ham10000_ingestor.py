import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

from datasets import Dataset, DatasetDict

from base_ingestor import BaseIngestor, EpisodeRecord


logger = logging.getLogger(__name__)


class HAM10000Ingestor(BaseIngestor):
    """Ingestor for NeilDeshmukh/ham10000 (skin lesions).

    Imaging dataset: save images to data/raw and reference paths.
    """

    def dataset_name(self) -> str:  # type: ignore[override]
        return "NeilDeshmukh/ham10000"

    def _get_split_dataset(self) -> Dataset:
        assert self.hf_dataset is not None
        if isinstance(self.hf_dataset, DatasetDict):
            split = self.split or "train"
            return self.hf_dataset[split]
        return self.hf_dataset  # type: ignore[return-value]

    def transform(self) -> Iterable[Dict[str, Any]]:
        ds = self._get_split_dataset()

        images_root = self.raw_root / "ham10000" / "images"
        images_root.mkdir(parents=True, exist_ok=True)

        for idx, row in enumerate(ds):
            img = row.get("image")
            rel_img_path = None
            if img is not None:
                rel_img_path = f"images/{idx}.png"
                abs_img_path = images_root / f"{idx}.png"
                try:
                    img.save(abs_img_path)
                except Exception:
                    logger.exception("Failed to save HAM10000 image index=%d", idx)
                    rel_img_path = None

            label = row.get("label") or row.get("dx")  # disease label
            labels: List[str] = [label] if label else []

            yield {
                "episode_id": str(row.get("id") or idx),
                "image_path": rel_img_path,
                "labels": labels,
                "raw": row,
            }

    def canonicalize_sample(self, sample: Dict[str, Any]) -> EpisodeRecord:
        episode_id = str(sample["episode_id"])
        rel_img = sample.get("image_path")
        dataset_key = "ham10000"
        imaging = {}
        if rel_img:
            imaging = self.canonicalizer.resolve_image_paths(
                root=self.raw_root / dataset_key,
                relative_paths=[rel_img],
            )

        label_texts: List[str] = sample.get("labels", [])
        label_info = self.canonicalizer.map_diagnoses(label_texts) if label_texts else {
            "diagnosis_labels": [],
            "ontology_codes": {"ICD10": [], "SNOMED": []},
        }

        text = self.canonicalizer.normalize_text_fields(
            notes=str(sample.get("raw", {})),
        )

        return EpisodeRecord(
            episode_id=episode_id,
            source=self.dataset_name(),
            text=text,
            imaging={**imaging, "labels": label_info["diagnosis_labels"]},
            labs={},
            vitals={},
            diagnosis_labels=label_info["diagnosis_labels"],
            drug_labels=[],
            ontology_codes=label_info["ontology_codes"],
            metadata={"hf_row_keys": list(sample.get("raw", {}).keys())},
        )

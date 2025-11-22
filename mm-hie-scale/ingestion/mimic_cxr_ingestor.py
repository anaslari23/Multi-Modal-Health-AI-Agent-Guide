import logging
from typing import Any, Dict, Iterable

from datasets import Dataset, DatasetDict

from base_ingestor import BaseIngestor, EpisodeRecord


logger = logging.getLogger(__name__)


class MIMICCXRIngestor(BaseIngestor):
    """Metadata-only ingestor for MIT/mimic-cxr.

    SAFETY NOTICE:
      - Access to MIMIC-CXR requires credentialing and IRB-style approvals.
      - This ingestor operates only on metadata from the HuggingFace dataset
        and does NOT attempt to download or store image pixels.
      - Ensure you comply with the PhysioNet/MIMIC data use agreement.
    """

    def dataset_name(self) -> str:  # type: ignore[override]
        return "MIT/mimic-cxr"

    def _get_split_dataset(self) -> Dataset:
        assert self.hf_dataset is not None
        if isinstance(self.hf_dataset, DatasetDict):
            split = self.split or "train"
            return self.hf_dataset[split]
        return self.hf_dataset  # type: ignore[return-value]

    def transform(self) -> Iterable[Dict[str, Any]]:
        ds = self._get_split_dataset()
        for idx, row in enumerate(ds):
            # Do NOT touch image pixels here; we only keep metadata such as study ID,
            # subject ID, and label metadata.
            yield {
                "episode_id": str(row.get("study_id") or row.get("id") or idx),
                "labels": row.get("labels") or [],
                "metadata": {
                    k: v
                    for k, v in row.items()
                    if k not in {"image", "pixels", "pixel_array"}
                },
            }

    def canonicalize_sample(self, sample: Dict[str, Any]) -> EpisodeRecord:
        episode_id = str(sample["episode_id"])

        label_texts = [str(l) for l in sample.get("labels", [])]
        label_info = self.canonicalizer.map_diagnoses(label_texts) if label_texts else {
            "diagnosis_labels": [],
            "ontology_codes": {"ICD10": [], "SNOMED": []},
        }

        text = self.canonicalizer.normalize_text_fields(
            notes=str(sample.get("metadata", {})),
        )

        # No imaging paths or pixels are stored; only metadata.
        imaging: Dict[str, Any] = {"labels": label_info["diagnosis_labels"], "metadata_only": True}

        return EpisodeRecord(
            episode_id=episode_id,
            source=self.dataset_name(),
            text=text,
            imaging=imaging,
            labs={},
            vitals={},
            diagnosis_labels=label_info["diagnosis_labels"],
            drug_labels=[],
            ontology_codes=label_info["ontology_codes"],
            metadata={"safety_notice": "MIMIC-CXR metadata-only ingestion; images not stored."},
        )

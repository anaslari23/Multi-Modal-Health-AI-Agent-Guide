import logging
from typing import Any, Dict, Iterable

from datasets import Dataset, DatasetDict

from base_ingestor import BaseIngestor, EpisodeRecord


logger = logging.getLogger(__name__)


class MedDialogIngestor(BaseIngestor):
    """Ingestor for UCSD-AI4H/MedDialog.

    Treat each doctor-patient dialog as one text episode.
    TODO: Refine dialog flattening (speaker turns, structure).
    """

    def dataset_name(self) -> str:  # type: ignore[override]
        return "UCSD-AI4H/MedDialog"

    def _get_split_dataset(self) -> Dataset:
        assert self.hf_dataset is not None
        if isinstance(self.hf_dataset, DatasetDict):
            split = self.split or "train"
            return self.hf_dataset[split]
        return self.hf_dataset  # type: ignore[return-value]

    def transform(self) -> Iterable[Dict[str, Any]]:
        ds = self._get_split_dataset()
        for idx, row in enumerate(ds):
            # Approximate schema: "dialog" or concatenated utterances
            dialog = row.get("dialog") or row.get("conversation") or ""
            yield {
                "episode_id": str(row.get("id") or idx),
                "dialog": dialog,
                "raw": row,
            }

    def canonicalize_sample(self, sample: Dict[str, Any]) -> EpisodeRecord:
        episode_id = str(sample["episode_id"])

        text = self.canonicalizer.normalize_text_fields(
            notes=sample.get("dialog"),
        )

        return EpisodeRecord(
            episode_id=episode_id,
            source=self.dataset_name(),
            text=text,
            imaging={},
            labs={},
            vitals={},
            diagnosis_labels=[],
            drug_labels=[],
            ontology_codes={"ICD10": [], "SNOMED": [], "RxNorm": []},
            metadata={"hf_row_keys": list(sample.get("raw", {}).keys())},
        )

import logging
from typing import Any, Dict, Iterable

from datasets import Dataset, DatasetDict

from base_ingestor import BaseIngestor, EpisodeRecord


logger = logging.getLogger(__name__)


class PubMedQAIngestor(BaseIngestor):
    """Ingestor for pubmed_qa.

    Treat each QA item as one episode with rich QA fields.
    """

    def dataset_name(self) -> str:  # type: ignore[override]
        return "pubmed_qa"

    def _get_split_dataset(self) -> Dataset:
        assert self.hf_dataset is not None
        if isinstance(self.hf_dataset, DatasetDict):
            split = self.split or "train"
            return self.hf_dataset[split]
        return self.hf_dataset  # type: ignore[return-value]

    def transform(self) -> Iterable[Dict[str, Any]]:
        ds = self._get_split_dataset()
        for idx, row in enumerate(ds):
            yield {
                "episode_id": str(row.get("id") or idx),
                "question": row.get("question") or row.get("query") or "",
                "context": row.get("context") or row.get("context_text") or "",
                "answer": row.get("answer") or row.get("long_answer") or "",
                "raw": row,
            }

    def canonicalize_sample(self, sample: Dict[str, Any]) -> EpisodeRecord:
        episode_id = str(sample["episode_id"])

        text = self.canonicalizer.normalize_text_fields(
            qa_context=sample.get("context"),
            question=sample.get("question"),
            answer=sample.get("answer"),
        )

        # Disease mentions may appear in the question
        disease_strings = []
        if sample.get("question"):
            disease_strings.append(str(sample["question"]))
        label_info = self.canonicalizer.map_diagnoses(disease_strings) if disease_strings else {
            "diagnosis_labels": [],
            "ontology_codes": {"ICD10": [], "SNOMED": []},
        }

        return EpisodeRecord(
            episode_id=episode_id,
            source=self.dataset_name(),
            text=text,
            imaging={},
            labs={},
            vitals={},
            diagnosis_labels=label_info["diagnosis_labels"],
            drug_labels=[],
            ontology_codes=label_info["ontology_codes"],
            metadata={"hf_row_keys": list(sample.get("raw", {}).keys())},
        )

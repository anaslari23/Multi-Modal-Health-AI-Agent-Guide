import logging
from typing import Any, Dict, Iterable

from datasets import Dataset, DatasetDict

from base_ingestor import BaseIngestor, EpisodeRecord


logger = logging.getLogger(__name__)


class PubMedAbstractsIngestor(BaseIngestor):
    """Ingestor for pubmed_abstracts (large-scale text corpus).

    TODO: Confirm exact HF dataset name and schema (title, abstract, MeSH terms).
    """

    def dataset_name(self) -> str:  # type: ignore[override]
        return "pubmed_abstracts"

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
                "episode_id": str(row.get("pmid") or row.get("id") or idx),
                "title": row.get("title") or "",
                "abstract": row.get("abstract") or row.get("text") or "",
                "mesh_terms": row.get("mesh_terms") or [],
                "raw": row,
            }

    def canonicalize_sample(self, sample: Dict[str, Any]) -> EpisodeRecord:
        episode_id = str(sample["episode_id"])

        text = self.canonicalizer.normalize_text_fields(
            summary=sample.get("title"),
            notes=sample.get("abstract"),
            extra={"mesh_terms": sample.get("mesh_terms")},
        )

        disease_strings = [str(t) for t in sample.get("mesh_terms") or []]
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

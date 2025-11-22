import logging
from typing import Any, Dict, Iterable, List

from datasets import Dataset, DatasetDict

from base_ingestor import BaseIngestor, EpisodeRecord


logger = logging.getLogger(__name__)


class FDADrugLabelsIngestor(BaseIngestor):
    """Ingestor for fda_museum/fda_drug_labels.

    Treat each drug label document as one episode with drug-centric text.
    """

    def dataset_name(self) -> str:  # type: ignore[override]
        return "fda_museum/fda_drug_labels"

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
                "episode_id": str(row.get("id") or row.get("set_id") or idx),
                "drug_name": row.get("drug_name") or row.get("brand_name") or "",
                "text": row.get("label_text") or row.get("text") or "",
                "indications": row.get("indications") or [],
                "raw": row,
            }

    def canonicalize_sample(self, sample: Dict[str, Any]) -> EpisodeRecord:
        episode_id = str(sample["episode_id"])

        text = self.canonicalizer.normalize_text_fields(
            notes=sample.get("text"),
            summary=sample.get("drug_name"),
            extra={"indications": sample.get("indications")},
        )

        drug_names: List[str] = []
        if sample.get("drug_name"):
            drug_names.append(str(sample["drug_name"]))

        drug_info = self.canonicalizer.map_drugs(drug_names)

        # Disease indications (may map to ICD/SNOMED later)
        disease_strings = [str(i) for i in sample.get("indications") or []]
        label_info = self.canonicalizer.map_diagnoses(disease_strings) if disease_strings else {
            "diagnosis_labels": [],
            "ontology_codes": {"ICD10": [], "SNOMED": []},
        }

        ontology_codes = {
            "ICD10": label_info["ontology_codes"].get("ICD10", []),
            "SNOMED": label_info["ontology_codes"].get("SNOMED", []),
            "RxNorm": drug_info["ontology_codes"].get("RxNorm", []),
        }

        return EpisodeRecord(
            episode_id=episode_id,
            source=self.dataset_name(),
            text=text,
            imaging={},
            labs={},
            vitals={},
            diagnosis_labels=label_info["diagnosis_labels"],
            drug_labels=drug_info["drug_labels"],
            ontology_codes=ontology_codes,
            metadata={"hf_row_keys": list(sample.get("raw", {}).keys())},
        )

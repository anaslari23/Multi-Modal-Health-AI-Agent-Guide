import logging
from typing import Any, Dict, Iterable, List

from datasets import Dataset, DatasetDict

from base_ingestor import BaseIngestor, EpisodeRecord


logger = logging.getLogger(__name__)


class PTBXLIngestor(BaseIngestor):
    """Ingestor for physionet_org/ptb-xl (ECG/vitals).

    Represent ECG waveforms as vitals time series.
    """

    def dataset_name(self) -> str:  # type: ignore[override]
        return "physionet_org/ptb-xl"

    def _get_split_dataset(self) -> Dataset:
        assert self.hf_dataset is not None
        if isinstance(self.hf_dataset, DatasetDict):
            split = self.split or "train"
            return self.hf_dataset[split]
        return self.hf_dataset  # type: ignore[return-value]

    def transform(self) -> Iterable[Dict[str, Any]]:
        ds = self._get_split_dataset()
        for idx, row in enumerate(ds):
            # TODO: Confirm schema for signal and labels in HF version
            signal = row.get("signal") or []
            timestamps = list(range(len(signal)))  # placeholder uniform sampling
            labels: List[str] = []  # e.g. diagnostic labels

            yield {
                "episode_id": str(row.get("ecg_id") or row.get("id") or idx),
                "signal": signal,
                "timestamps": timestamps,
                "labels": labels,
                "raw": row,
            }

    def canonicalize_sample(self, sample: Dict[str, Any]) -> EpisodeRecord:
        episode_id = str(sample["episode_id"])
        vitals = self.canonicalizer.normalize_vitals_timeseries(
            signal=sample.get("signal", []),
            timestamps=sample.get("timestamps", []),
            kind="ECG",
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
            imaging={},
            labs={},
            vitals=vitals,
            diagnosis_labels=label_info["diagnosis_labels"],
            drug_labels=[],
            ontology_codes=label_info["ontology_codes"],
            metadata={"hf_row_keys": list(sample.get("raw", {}).keys())},
        )

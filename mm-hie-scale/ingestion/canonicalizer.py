import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .ontology_mapper import (
    map_disease_string_to_codes,
    map_icd10_to_snomed,
    map_drugname_to_rxnorm,
)


logger = logging.getLogger(__name__)


class Canonicalizer:
    """Utility for normalizing fields into the MM-HIE episode schema.

    Expected final episode layout:
      {
        "episode_id": "...",
        "source": "...",
        "text": { "symptoms": "...", "notes": "...", "summary": "...", "qa_context": "..." },
        "imaging": { "path": "..." | "paths": [...], "labels": [...] },
        "labs": { ... },
        "vitals": { ... },
        "diagnosis_labels": [ ... ],
        "drug_labels": [ ... ],
        "ontology_codes": { "ICD10": [...], "SNOMED": [...], "RxNorm": [...] },
        "metadata": { ... }
      }
    """

    # ------------------------- Text helpers -------------------------

    def normalize_text_fields(
        self,
        *,
        symptoms: Optional[str] = None,
        notes: Optional[str] = None,
        summary: Optional[str] = None,
        qa_context: Optional[str] = None,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        text: Dict[str, Any] = {}
        if symptoms:
            text["symptoms"] = symptoms
        if notes:
            text["notes"] = notes
        if summary:
            text["summary"] = summary
        if qa_context:
            text["qa_context"] = qa_context
        if question:
            text["question"] = question
        if answer:
            text["answer"] = answer
        if extra:
            text.update(extra)
        return text

    # ------------------------- Imaging helpers -------------------------

    def resolve_image_paths(
        self,
        root: Path,
        relative_paths: List[str],
    ) -> Dict[str, Any]:
        paths = [str(root / p) for p in relative_paths]
        return {"paths": paths}

    # ------------------------- Time helpers -------------------------

    def normalize_timestamp(self, ts: Any) -> Optional[str]:
        """Normalize timestamps to ISO 8601 strings.

        Accepts datetime, int (epoch), or str; returns ISO8601 or None.
        """
        if ts is None:
            return None
        if isinstance(ts, datetime):
            return ts.isoformat()
        if isinstance(ts, (int, float)):
            try:
                return datetime.utcfromtimestamp(ts).isoformat() + "Z"
            except Exception:
                logger.exception("Failed to convert epoch timestamp %s", ts)
                return None
        if isinstance(ts, str):
            try:
                # best-effort parse; downstream can tighten this
                return datetime.fromisoformat(ts.replace("Z", "")).isoformat()
            except Exception:
                return ts
        return str(ts)

    # ------------------------- Vitals helpers -------------------------

    def normalize_vitals_timeseries(
        self,
        signal: List[float],
        timestamps: List[Any],
        kind: str = "ECG",
    ) -> Dict[str, Any]:
        if len(signal) != len(timestamps):
            logger.warning("Vitals signal/timestamp length mismatch: %d vs %d", len(signal), len(timestamps))
        norm_ts = [self.normalize_timestamp(t) for t in timestamps]
        return {
            "vitals": {kind: signal},
            "timestamp": norm_ts,
        }

    # ------------------------- Label helpers -------------------------

    def map_diagnoses(
        self,
        disease_strings: List[str],
    ) -> Dict[str, Any]:
        icd10_codes: List[str] = []
        snomed_codes: List[str] = []
        label_objs: List[Dict[str, Any]] = []

        for text in disease_strings:
            codes = map_disease_string_to_codes(text)
            for key in codes:
                try:
                    system, code = key.split(":", 1)
                except ValueError:
                    continue
                if system == "ICD-10":
                    icd10_codes.append(code)
                elif system == "SNOMED-CT":
                    snomed_codes.append(code)
                label_objs.append({"system": system, "code": code, "description": text})

        # deduplicate
        icd10_codes = sorted(set(icd10_codes))
        snomed_codes = sorted(set(snomed_codes))

        return {
            "diagnosis_labels": label_objs,
            "ontology_codes": {
                "ICD10": icd10_codes,
                "SNOMED": snomed_codes,
            },
        }

    def map_icd10_and_snomed(
        self,
        icd_codes: List[str],
    ) -> Dict[str, Any]:
        snomed: List[str] = []
        for icd in icd_codes:
            mapped = map_icd10_to_snomed(icd)
            if mapped:
                snomed.extend(mapped)
        return {
            "ICD10": sorted(set(icd_codes)),
            "SNOMED": sorted(set(snomed)),
        }

    def map_drugs(self, drug_names: List[str]) -> Dict[str, Any]:
        rxnorm_codes: List[str] = []
        labels: List[Dict[str, Any]] = []
        for name in drug_names:
            codes = map_drugname_to_rxnorm(name)
            for c in codes:
                rxnorm_codes.append(c)
                labels.append({"system": "RxNorm", "code": c, "name": name})
        return {
            "drug_labels": labels,
            "ontology_codes": {"RxNorm": sorted(set(rxnorm_codes))},
        }

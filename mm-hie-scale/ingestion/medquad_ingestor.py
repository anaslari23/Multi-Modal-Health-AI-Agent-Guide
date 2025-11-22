import logging
from typing import Any, Dict, Iterable

from datasets import Dataset, DatasetDict

from .base_ingestor import BaseIngestor, EpisodeRecord

try:
    from shared.embeddings.llm_embedder import GGUFEmbedder
except Exception:  # shared package may not be on PYTHONPATH in some environments
    GGUFEmbedder = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)

if GGUFEmbedder is not None:
    try:
        _EMBEDDER = GGUFEmbedder()
        logger.info("Initialized GGUFEmbedder for MedQuAD ingestion")
    except Exception:
        _EMBEDDER = None
        logger.warning(
            "Failed to initialize GGUFEmbedder; MedQuAD embeddings will be empty.",
            exc_info=True,
        )
else:
    _EMBEDDER = None


class MedQuadIngestor(BaseIngestor):
    """Ingestor for medical_ai/medquad (medical QA pairs).

    Expected HF schema (approximate, TODO verify):
      - question: str
      - answer: str
      - category / url / source: optional metadata
    """

    def dataset_name(self) -> str:  # type: ignore[override]
        # NOTE: Original 'medical_ai/medquad' is no longer available.
        # The maintained version is at RAKAI/MedQuAD.
        return "RAKAI/MedQuAD"

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
                "question": row.get("question"),
                "answer": row.get("answer"),
                "category": row.get("category"),
                "raw": row,
            }

    def canonicalize_sample(self, sample: Dict[str, Any]) -> EpisodeRecord:
        episode_id = str(sample["episode_id"])

        text = self.canonicalizer.normalize_text_fields(
            qa_context=sample.get("question"),
            question=sample.get("question"),
            answer=sample.get("answer"),
            extra={"category": sample.get("category")},
        )

        embeddings: Dict[str, Any] = {}
        try:
            if _EMBEDDER is not None:
                query = text.get("query") or ""
                answer = text.get("answer") or ""
                combined = f"{query}\n{answer}".strip()
                if combined:
                    emb = _EMBEDDER.embed(combined)
                else:
                    emb = []
                embeddings["text"] = emb
            else:
                embeddings["text"] = []
        except Exception:
            logger.warning(
                "Failed to compute MedQuAD text embedding for episode_id=%s", episode_id,
                exc_info=True,
            )
            embeddings["text"] = []

        # Map potential disease mentions from question/answer text
        disease_strings = []
        if sample.get("category"):
            disease_strings.append(str(sample["category"]))
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
            embeddings=embeddings,
            diagnosis_labels=label_info["diagnosis_labels"],
            drug_labels=[],
            ontology_codes=label_info["ontology_codes"],
            metadata={"hf_row_keys": list(sample.get("raw", {}).keys())},
        )

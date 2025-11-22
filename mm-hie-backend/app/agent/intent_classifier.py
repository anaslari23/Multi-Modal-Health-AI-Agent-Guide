from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


IntentLabel = Literal[
    "symptoms",
    "upload",
    "vitals",
    "meds",
    "question",
]


@dataclass
class IntentResult:
    label: IntentLabel
    confidence: float


class IntentClassifier:
    """Lightweight rule-based intent classifier.

    This is intentionally simple and deterministic; it can be
    augmented with RAG/LLM in a later iteration.
    """

    def classify(self, text: str) -> IntentResult:
        t = text.lower()

        if any(k in t for k in ["pain", "cough", "fever", "symptom", "nausea", "vomit", "shortness of breath"]):
            return IntentResult("symptoms", 0.8)

        if any(k in t for k in ["upload", "attach", "pdf", "report", "scan", "x-ray", "xray", "ct", "mri"]):
            return IntentResult("upload", 0.75)

        if any(k in t for k in ["heart rate", "pulse", "spo2", "oxygen", "temperature", "fever", "bp", "blood pressure", "respiratory rate"]):
            return IntentResult("vitals", 0.7)

        if any(k in t for k in ["medication", "medicine", "drug", "tablet", "dose"]):
            return IntentResult("meds", 0.8)

        return IntentResult("question", 0.5)


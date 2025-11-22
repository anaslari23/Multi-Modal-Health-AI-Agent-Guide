from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import json


@dataclass
class DrugSuggestion:
    name: str
    indication: str
    safe: bool
    rationale: str


class DrugChecker:
    """Very small, local safety checker for suggested medications.

    This is NOT a real clinical decision support system. It exists to
    demonstrate how the agent could reason about allergies and basic
    interactions before surfacing medications to the user.
    """

    def __init__(self, db_path: Union[Path, str ]| None = None) -> None:
        if db_path is None:
            db_path = Path("app/agent/data/drugs_minimal.json")
        self.db_path = Path(db_path)
        self._db: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.db_path.exists():
            # Minimal fallback dataset.
            self._db = [
                {
                    "name": "paracetamol",
                    "indication": "pain or fever",
                    "contra_allergy": ["paracetamol"],
                    "notes": "Avoid overdose; max daily dose depends on weight and liver function.",
                },
                {
                    "name": "ibuprofen",
                    "indication": "pain, inflammation, or fever",
                    "contra_allergy": ["ibuprofen", "nsaid", "aspirin"],
                    "notes": "Take with food. Avoid if you have stomach ulcers or asthma.",
                },
                {
                    "name": "amoxicillin",
                    "indication": "bacterial infection",
                    "contra_allergy": ["amoxicillin", "penicillin"],
                    "notes": "Complete the full course. Watch for rash.",
                },
                {
                    "name": "cetirizine",
                    "indication": "allergies",
                    "contra_allergy": ["cetirizine"],
                    "notes": "May cause drowsiness in some people.",
                },
                {
                    "name": "aspirin",
                    "indication": "pain, inflammation, or heart protection",
                    "contra_allergy": ["aspirin", "nsaid", "ibuprofen"],
                    "notes": "Do not give to children/teenagers (Reye's syndrome risk). Watch for bleeding.",
                }
            ]
            return

        with self.db_path.open("r", encoding="utf-8") as f:
            self._db = json.load(f)

    def suggest(self, symptoms: str, allergies: Optional[str] = None) -> List[DrugSuggestion]:
        allergies_lower = (allergies or "").lower()
        out: List[DrugSuggestion] = []

        for entry in self._db:
            name = str(entry.get("name", "")).lower()
            indication = str(entry.get("indication", ""))
            notes = str(entry.get("notes", ""))
            contra_allergy = [str(a).lower() for a in entry.get("contra_allergy", [])]

            safe = True
            rationale_parts = []

            if allergies_lower and any(a in allergies_lower for a in contra_allergy):
                safe = False
                rationale_parts.append("Possible allergy or intolerance mentioned.")

            if not symptoms:
                rationale_parts.append("General symptomatic option; final decision must be made by a clinician.")

            if not rationale_parts:
                rationale_parts.append(notes or "Use with usual clinical precautions.")

            out.append(
                DrugSuggestion(
                    name=name,
                    indication=indication,
                    safe=safe,
                    rationale=" ".join(rationale_parts),
                )
            )

        return out

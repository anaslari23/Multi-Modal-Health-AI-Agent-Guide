from __future__ import annotations

from typing import Dict, Iterable, List


def generate_followups(slots: Dict[str, str], missing_slots: Iterable[str]) -> List[str]:
    """Generate up to ~3 focused clinical follow-up questions.

    This is intentionally template-based and deterministic so it is
    cheap and predictable. RAG/LLM augmentation can be layered on top
    later if needed.
    """

    questions: List[str] = []

    missing = list(missing_slots)

    if "duration" in missing:
        questions.append("How long have you had these symptoms?")
    if "severity" in missing:
        questions.append("On a scale from mild to severe, how bad are the symptoms?")
    if "onset" in missing:
        questions.append("Did the symptoms start suddenly or gradually?")
    if "risk_factors" in missing:
        questions.append("Do you have any chronic conditions (for example asthma, COPD, heart or kidney disease)?")
    if "allergies" in missing:
        questions.append("Do you have any medication or food allergies?")
    if "medications" in missing:
        questions.append("Are you currently taking any regular medications?")

    # Cap to 3 follow-ups to avoid overwhelming the user.
    return questions[:3]

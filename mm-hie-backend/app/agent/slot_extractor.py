from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List


DURATION_PATTERN = re.compile(r"(\d+\s*(day|days|week|weeks|month|months|year|years))", re.I)
SEVERITY_PATTERN = re.compile(r"(mild|moderate|severe|very severe|worst)", re.I)


@dataclass
class SlotResult:
    slots: Dict[str, str] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)


class SlotExtractor:
    """Simple regex + keyword-based slot extractor for clinical intake."""

    REQUIRED_SLOTS = [
        "symptoms",
        "duration",
        "severity",
        "onset",
        "risk_factors",
        "allergies",
        "medications",
    ]

    def extract(self, text: str) -> SlotResult:
        t = text.strip()
        slots: Dict[str, str] = {}

        if t:
            slots["symptoms"] = t

        dur_match = DURATION_PATTERN.search(t)
        if dur_match:
            slots["duration"] = dur_match.group(0)

        sev_match = SEVERITY_PATTERN.search(t)
        if sev_match:
            slots["severity"] = sev_match.group(0)

        # Very lightweight heuristics for some risk factors / allergies / meds
        if "asthma" in t or "copd" in t or "smoke" in t:
            slots["risk_factors"] = "respiratory risk factors mentioned"

        if "allerg" in t:
            slots["allergies"] = "allergy history mentioned"

        if "mg" in t or "tablet" in t or "dose" in t:
            slots["medications"] = "medication details mentioned"

        missing = [s for s in self.REQUIRED_SLOTS if s not in slots]
        return SlotResult(slots=slots, missing=missing)


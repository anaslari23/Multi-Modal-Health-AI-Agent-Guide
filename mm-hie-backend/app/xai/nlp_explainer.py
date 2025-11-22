from __future__ import annotations

import math
import re
from typing import Dict


class NLPExplainer:
    """Heuristic token-level explainer for symptom text.

    This component operates on raw symptom text and produces a mapping
    from word -> importance weight in [0, 1]. It does not depend on
    model internals, which keeps it lightweight and easy to test, but
    the scoring is inspired by attention-style weighting:

    - Frequent terms in the sentence get higher weights.
    - Clinically salient keywords (e.g. fever, cough, dyspnea) receive
      an additional boost.
    """

    _TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")

    # Very small, illustrative set of symptom keywords.
    _KEYWORDS = {
        "fever",
        "cough",
        "dyspnea",
        "shortness",
        "breath",
        "chest",
        "pain",
        "fatigue",
        "headache",
        "nausea",
        "vomiting",
    }

    def _tokenise(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for match in self._TOKEN_RE.finditer(text.lower()):
            tok = match.group(0)
            counts[tok] = counts.get(tok, 0) + 1
        return counts

    def explain(self, text: str) -> Dict[str, float]:
        """Return a dict of word -> weight in [0, 1]."""

        counts = self._tokenise(text)
        if not counts:
            return {}

        scores: Dict[str, float] = {}
        for tok, c in counts.items():
            # Base score: log-scaled frequency.
            score = 1.0 + math.log(1.0 + float(c))
            # Keyword boost.
            if tok in self._KEYWORDS:
                score *= 1.5
            scores[tok] = score

        max_score = max(scores.values()) if scores else 1.0
        if max_score <= 0:
            return {tok: 0.0 for tok in scores}

        # Normalise to [0, 1].
        return {tok: float(score / max_score) for tok, score in scores.items()}

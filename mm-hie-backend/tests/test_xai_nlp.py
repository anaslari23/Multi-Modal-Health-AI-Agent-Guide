from __future__ import annotations

from app.xai.nlp_explainer import NLPExplainer
from app.xai.xai_router import XAIAggregator
from app.schemas import FusionOutput


def test_nlp_explainer_returns_normalised_weights():
    explainer = NLPExplainer()
    text = "Patient with fever and cough and chest pain"

    weights = explainer.explain(text)

    assert weights, "Expected at least one token weight"
    # All weights should lie in [0, 1].
    for w in weights.values():
        assert 0.0 <= w <= 1.0
    # Keyword tokens should be present.
    keys = set(weights.keys())
    assert {"fever", "cough"}.issubset(keys)


def test_xai_aggregator_populates_nlp_highlights_from_meta_notes():
    agg = XAIAggregator()
    symptom_text = "Fever, cough and shortness of breath"
    case_data = {
        "meta": {"notes": symptom_text},
        "nlp": object(),
        "labs": None,
        "imaging": None,
        "vitals": None,
    }
    fusion = FusionOutput(final_risk_score=10.0, triage="Low", conditions=[])

    xai = agg.explain("case_nlp", case_data, fusion)

    assert xai.nlp_highlights is not None
    # Expect highlights for at least one key symptom token.
    keys = set(xai.nlp_highlights.keys())
    assert "fever" in keys or "cough" in keys or "breath" in keys

from __future__ import annotations

import os

from app.schemas import LabResults, LabValue, FusionOutput
from app.xai.lab_explainer import LabExplainer
from app.xai.xai_router import XAIAggregator


def _make_dummy_labs() -> LabResults:
    return LabResults(
        values={
            "CRP": LabValue(value=100.0, flag="HIGH"),
            "Hb": LabValue(value=7.5, flag="LOW"),
            "Na": LabValue(value=140.0, flag="NORMAL"),
        }
    )


def test_lab_explainer_creates_png_and_summary(tmp_path, monkeypatch):
    # Redirect base dir to a temp folder so tests don't write to the real models directory.
    base = tmp_path / "xai_labs"
    explainer = LabExplainer(base_dir=str(base))

    labs = _make_dummy_labs()
    summary, png_path = explainer.explain(case_id="testcase", labs=labs)

    assert "CRP" in summary or "Hb" in summary
    assert png_path.endswith(".png")
    assert os.path.exists(png_path)


def test_xai_aggregator_includes_labs_shap_path(tmp_path, monkeypatch):
    # Monkeypatch LabExplainer inside XAIAggregator to control output and path location.
    class DummyLabExplainer:
        def __init__(self) -> None:
            self.called_with = None

        def explain(self, case_id, labs):  # pragma: no cover - simple passthrough
            self.called_with = (case_id, labs)
            # Create a dummy file in tmp_path
            png = tmp_path / f"labs_shap_{case_id}.png"
            png.write_bytes(b"dummy")
            return "Dummy lab summary.", str(png)

    dummy = DummyLabExplainer()

    def _dummy_init(self):  # pragma: no cover - simple init override
        self.lab_explainer = dummy

    monkeypatch.setattr("app.xai.xai_router.XAIAggregator.__init__", _dummy_init)

    agg = XAIAggregator()
    labs = _make_dummy_labs()
    fusion = FusionOutput(final_risk_score=50.0, triage="Medium", conditions=[])
    case_data = {"labs": labs, "nlp": None, "imaging": None, "vitals": None}

    xai = agg.explain("case123", case_data, fusion)

    assert xai.labs_shap_path is not None
    assert xai.labs_shap_path.endswith(".png")
    assert os.path.exists(xai.labs_shap_path)
    assert "Dummy lab summary" in xai.summary

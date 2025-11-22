from __future__ import annotations

from typing import Dict, Any

from ..schemas import XAIOutput, FusionOutput
from .lab_explainer import LabExplainer
from .nlp_explainer import NLPExplainer


class XAIAggregator:
    def __init__(self) -> None:
        self.lab_explainer = LabExplainer()
        self.nlp_explainer = NLPExplainer()

    def explain(self, case_id: str, case_data: Dict[str, Any], fusion: FusionOutput) -> XAIOutput:
        summary_parts = []

        nlp_highlights = None
        if case_data.get("nlp") is not None:
            summary_parts.append("Text symptoms contributed to risk via NLP model.")

            # If raw symptom text is available, derive token-level highlights.
            symptom_text = None
            meta = case_data.get("meta") or {}
            # Prefer a dedicated field if present, fall back to notes.
            symptom_text = meta.get("symptom_text") or meta.get("notes")
            if isinstance(symptom_text, str) and symptom_text.strip():
                nlp_highlights = self.nlp_explainer.explain(symptom_text)
        if case_data.get("imaging") is not None:
            summary_parts.append("Imaging probabilities influenced the final score.")
        if case_data.get("vitals") is not None:
            summary_parts.append("Vital sign anomalies informed the triage.")

        labs_shap_path = None
        labs = case_data.get("labs")
        if labs is not None:
            lab_summary, labs_shap_path = self.lab_explainer.explain(case_id, labs)
            summary_parts.append(lab_summary)

        summary = " ".join(summary_parts) or "No data available for explanation."

        gradcam_path = None
        imaging = case_data.get("imaging")
        if imaging is not None:
            gradcam_path = imaging.gradcam_path

        return XAIOutput(
            summary=summary,
            gradcam_path=gradcam_path,
            labs_shap_path=labs_shap_path,
            nlp_highlights=nlp_highlights,
        )

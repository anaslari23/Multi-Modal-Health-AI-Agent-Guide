from __future__ import annotations

from typing import Any, Dict, List


class AgentBrain:
    """Autonomous reasoning agent for the clinical workflow.

    This is a lightweight, deterministic agent that reasons over the
    in-memory case state maintained by the Orchestrator. It does not
    perform any external API calls, which keeps tests fast and stable.
    """

    def _present_modalities(self, case: Dict[str, Any]) -> Dict[str, bool]:
        return {
            "nlp": case.get("nlp") is not None,
            "labs": case.get("labs") is not None,
            "imaging": case.get("imaging") is not None,
            "vitals": case.get("vitals") is not None,
        }

    def _suggest_next_steps(self, present: Dict[str, bool]) -> List[str]:
        steps: List[str] = []

        if not present["nlp"]:
            steps.append("Collect symptom history via NLP intake form.")
        if not present["labs"]:
            steps.append("Request upload of lab report (CBC, CMP, inflammatory markers).")
        if not present["imaging"]:
            steps.append("Request upload of relevant imaging (CXR/CT) for review.")
        if not present["vitals"]:
            steps.append("Capture vital signs time series (HR, SpO2, temp, RR).")

        steps.append("Review multimodal analysis and generate clinical report.")
        return steps

    def _estimate_confidence_gain(self, case: Dict[str, Any]) -> float:
        """Estimate incremental confidence based on available evidence.

        Simple heuristic:
        - Base gain from having symptom NLP.
        - Additional gain for each extra modality.
        - If Bayesian posterior exists, modulate by top posterior prob.
        """

        present = self._present_modalities(case)
        gain = 0.0

        if present["nlp"]:
            gain += 0.15
        if present["labs"]:
            gain += 0.15
        if present["imaging"]:
            gain += 0.25
        if present["vitals"]:
            gain += 0.15

        posterior = case.get("posterior") or []
        if posterior:
            top_prob = float(posterior[0].prob)
            gain += 0.3 * max(0.0, min(top_prob, 1.0))

        return float(max(0.0, min(gain, 1.0)))

    def _build_reasoning_chain(self, case: Dict[str, Any]) -> str:
        present = self._present_modalities(case)
        posterior = case.get("posterior") or []

        lines: List[str] = []
        lines.append("Step 1: Review current evidence across modalities.")

        desc_parts = []
        if present["nlp"]:
            desc_parts.append("symptom NLP")
        if present["labs"]:
            desc_parts.append("labs")
        if present["imaging"]:
            desc_parts.append("imaging")
        if present["vitals"]:
            desc_parts.append("vitals")

        if desc_parts:
            lines.append(" - Available modalities: " + ", ".join(desc_parts) + ".")
        else:
            lines.append(" - No structured modalities available yet.")

        if posterior:
            top = posterior[0]
            lines.append(
                f"Step 2: Bayesian update prioritises {top.condition} (posterior {top.prob:.2f})."
            )
        else:
            lines.append("Step 2: Posterior not available yet; rely on prior symptom model.")

        lines.append("Step 3: Decide which additional data would most reduce uncertainty.")

        return "\n".join(lines)

    def reason(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a reasoning step over the current case state.

        Returns a dict with keys:
        - next_steps: list of recommended actions.
        - agent_reasoning: human-readable reasoning chain.
        - confidence_gain: float in [0, 1] representing incremental confidence.
        """

        present = self._present_modalities(case)
        next_steps = self._suggest_next_steps(present)
        reasoning = self._build_reasoning_chain(case)
        confidence_gain = self._estimate_confidence_gain(case)

        return {
            "next_steps": next_steps,
            "agent_reasoning": reasoning,
            "confidence_gain": confidence_gain,
        }

from __future__ import annotations

from typing import Optional

import numpy as np

from ..schemas import FusionOutput, SymptomOutput, LabResults, ImagingOutput, VitalsOutput, ConditionProb


class MetaLearner:
    def __init__(self) -> None:
        self.weights = {
            "nlp": 0.35,
            "labs": 0.25,
            "imaging": 0.25,
            "vitals": 0.15,
        }

    def fuse(
        self,
        nlp: Optional[SymptomOutput],
        labs: Optional[LabResults],
        imaging: Optional[ImagingOutput],
        vitals: Optional[VitalsOutput],
    ) -> FusionOutput:
        risk_nlp = max((c.prob for c in nlp.conditions), default=0.0) if nlp else 0.0

        lab_abnormal_count = 0
        if labs:
            for v in labs.values.values():
                if v.flag.upper() in {"HIGH", "LOW"}:
                    lab_abnormal_count += 1
        risk_labs = min(lab_abnormal_count / 5.0, 1.0)

        risk_imaging = max(imaging.probabilities.values()) if imaging else 0.0

        risk_vitals = vitals.vitals_risk if vitals else 0.0

        final_risk_score = (
            risk_nlp * self.weights["nlp"]
            + risk_labs * self.weights["labs"]
            + risk_imaging * self.weights["imaging"]
            + risk_vitals * self.weights["vitals"]
        ) * 100.0

        if final_risk_score >= 70:
            triage = "High"
        elif final_risk_score >= 40:
            triage = "Medium"
        else:
            triage = "Low"

        conditions = nlp.conditions if nlp else []
        conditions = sorted(conditions, key=lambda c: c.prob, reverse=True)

        return FusionOutput(
            final_risk_score=float(np.round(final_risk_score, 2)),
            triage=triage,
            conditions=conditions,
        )

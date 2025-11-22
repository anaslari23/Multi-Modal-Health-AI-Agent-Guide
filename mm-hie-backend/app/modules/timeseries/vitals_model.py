from __future__ import annotations

from typing import List

import numpy as np

from ...schemas import VitalsOutput


class VitalsModel:
    def infer(
        self,
        heart_rate: List[float],
        spo2: List[float],
        temperature: List[float],
        resp_rate: List[float],
    ) -> VitalsOutput:
        hr = np.array(heart_rate, dtype=float) if heart_rate else np.array([80.0])
        s2 = np.array(spo2, dtype=float) if spo2 else np.array([97.0])
        temp = np.array(temperature, dtype=float) if temperature else np.array([37.0])
        rr = np.array(resp_rate, dtype=float) if resp_rate else np.array([16.0])

        anomalies: List[str] = []

        if hr.mean() > 100:
            anomalies.append("tachycardia")
        if s2.min() < 92:
            anomalies.append("oxygen_desaturation")
        if temp.max() > 38.0:
            anomalies.append("fever")
        if rr.mean() > 20:
            anomalies.append("tachypnea")

        risk_components = [
            min((hr.mean() - 60) / 60, 1.0),
            max((97 - s2.min()) / 10, 0.0),
            max((temp.max() - 37) / 3, 0.0),
            max((rr.mean() - 16) / 10, 0.0),
        ]
        vitals_risk = float(np.clip(sum(risk_components) / len(risk_components), 0, 1))

        return VitalsOutput(vitals_risk=vitals_risk, anomalies=anomalies)

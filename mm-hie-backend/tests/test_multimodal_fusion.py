from __future__ import annotations

from app.fusion.multimodal_transformer import MultimodalTransformerFusion
from app.schemas import SymptomOutput, ConditionProb, LabResults, LabValue, ImagingOutput, VitalsOutput


def _make_dummy_inputs():
    nlp = SymptomOutput(
        conditions=[
            ConditionProb(condition="Pneumonia", prob=0.8),
            ConditionProb(condition="Sepsis", prob=0.2),
        ],
        embedding=[0.1] * 768,
    )

    labs = LabResults(
        values={
            f"lab_{i}": LabValue(value=float(i), flag="NORMAL") for i in range(10)
        }
    )

    imaging = ImagingOutput(
        probabilities={"Pneumonia": 0.7, "Normal": 0.3},
        gradcam_path=None,
        embedding=[0.2] * 512,
    )

    vitals = VitalsOutput(
        vitals_risk=0.5,
        anomalies=["tachycardia"],
        embedding=[0.3] * 128,
        heart_rate=[80.0, 90.0, 85.0],
        spo2=[97.0, 96.0, 98.0],
    )

    return nlp, labs, imaging, vitals


def test_multimodal_fusion_shapes_and_risk_range():
    model = MultimodalTransformerFusion(num_diseases=20)
    nlp, labs, imaging, vitals = _make_dummy_inputs()

    out = model.fuse(nlp=nlp, labs=labs, imaging=imaging, vitals=vitals)

    fused = out["fused_vector"]
    disease_probs = out["disease_probs"]
    risk = out["risk_score"]

    assert fused.shape == (256,)
    assert disease_probs.shape == (20,)
    assert 0.0 <= risk <= 100.0

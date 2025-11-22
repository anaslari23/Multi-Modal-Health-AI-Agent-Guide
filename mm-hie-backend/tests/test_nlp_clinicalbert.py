from __future__ import annotations

from app.modules.nlp.model_clinical_bert import ClinicalBERTSymptomModel


def test_clinical_bert_symptom_infer_basic():
    model = ClinicalBERTSymptomModel()
    out = model.infer("fever and cough", top_n=5)

    assert len(out.conditions) == 5
    # ClinicalBERT hidden size is 768
    assert len(out.embedding) == 768

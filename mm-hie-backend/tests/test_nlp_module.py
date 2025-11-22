from app.modules.nlp.symptom_model import SymptomNLPModel


def test_symptom_nlp_infer_basic():
    model = SymptomNLPModel()
    out = model.infer("fever and cough", top_n=3)
    assert len(out.conditions) == 3
    assert len(out.embedding) > 0

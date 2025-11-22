from app.modules.timeseries.vitals_transformer import VitalsTransformerModel


def test_vitals_transformer_basic_shapes():
    model = VitalsTransformerModel()
    out = model.infer(
        heart_rate=[80, 85, 120, 110, 100],
        spo2=[98, 97, 90, 92, 93],
        temperature=[36.8, 37.2, 38.5],
        resp_rate=[16, 18, 24],
    )

    assert 0.0 <= out.vitals_risk <= 1.0
    assert isinstance(out.anomalies, list)
    assert len(out.embedding) == model.config.d_model

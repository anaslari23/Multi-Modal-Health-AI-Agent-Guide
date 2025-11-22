import io

from PIL import Image

from app.modules.imaging.imaging_model import ImagingModel


def test_imaging_model_efficientnet_outputs():
    model = ImagingModel()

    img = Image.new("RGB", (320, 320), color="gray")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    content = buf.getvalue()

    out = model.infer_bytes(content, case_id="test-case")

    assert set(out.probabilities.keys()) == {
        "Normal",
        "Pneumonia",
        "Edema/Effusion",
    }
    assert len(out.embedding) == model.embedding_dim
    assert 0.0 <= sum(out.probabilities.values()) <= 1.01

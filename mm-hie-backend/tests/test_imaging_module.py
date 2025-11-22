import io
from PIL import Image

from app.modules.imaging.imaging_model import ImagingModel


def test_imaging_model_infer_loads():
    model = ImagingModel()
    img = Image.new("RGB", (256, 256), color="white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out = model.infer_bytes(buf.getvalue())
    assert set(out.probabilities.keys()) == {"Normal", "Pneumonia", "Other"}

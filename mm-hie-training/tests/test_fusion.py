import sys
from pathlib import Path

import torch


# Make sure the project src/ is on sys.path when running pytest
ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fusion.model import CrossModalTransformer


def test_fusion_forward_shapes():
    batch = 4
    num_labels = 10
    model = CrossModalTransformer(num_labels=num_labels)

    nlp = torch.randn(batch, 768)
    img = torch.randn(batch, 512)
    labs = torch.randn(batch, 50)
    vitals = torch.randn(batch, 128)

    logits, risk, pooled = model(nlp, img, labs, vitals)

    assert logits.shape == (batch, num_labels)
    assert risk.shape == (batch,)
    assert pooled.shape[0] == batch

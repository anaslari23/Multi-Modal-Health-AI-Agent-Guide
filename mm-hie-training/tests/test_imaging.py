import sys
from pathlib import Path

import torch


# Make sure the project src/ is on sys.path when running pytest
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.imaging.model import load_efficientnet_b0


def test_imaging_forward_shape():
    num_labels = 3
    model = load_efficientnet_b0(num_labels=num_labels)
    dummy = torch.randn(1, 3, 224, 224)
    logits = model(dummy)

    assert logits.shape == (1, num_labels)

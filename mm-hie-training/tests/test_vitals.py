import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure src is on the path when running pytest from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vitals.model import build_vitals_model  # noqa: E402


def test_vitals_model_forward():
    """Smoke test: vitals transformer forward pass and shapes."""

    batch = 4
    seq_len = 48
    num_features = 2

    model = build_vitals_model(num_features=num_features, d_model=32)
    x = torch.randn(batch, seq_len, num_features)

    risk_logits, anomaly_logits, emb = model(x)

    assert risk_logits.shape == (batch,)
    assert anomaly_logits.shape == (batch,)
    assert emb.shape[0] == batch

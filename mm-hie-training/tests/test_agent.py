import os
import sys
from pathlib import Path

import torch

# Ensure src is on the path when running pytest from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from agent.model import load_agent_model  # noqa: E402


def test_agent_model_forward():
    """Smoke test: agent seq2seq model forward pass produces a loss."""

    # Keep model small for tests; default is t5-small
    os.environ.setdefault("AGENT_MODEL", "t5-small")
    tokenizer, model = load_agent_model()

    inputs = tokenizer(["Symptoms: cough, fever."], return_tensors="pt")
    # Use input_ids as pseudo-target to ensure labels are present
    outputs = model(**inputs, labels=inputs["input_ids"])

    assert "loss" in outputs
    assert outputs.loss is not None

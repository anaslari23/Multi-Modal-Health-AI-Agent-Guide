from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path


# Ensure project root (containing the `app` package) is on sys.path when
# tests are executed directly from the backend folder.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm.reasoning_model import MedicalReasoningLLM
from app.llm.config import LLMConfig


@contextmanager
def _env_override(key: str, value: str):
    old = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old


def test_generate_with_dummy_model_smoke():
    """Smoke test that generate() returns a non-empty string using a dummy model.

    This avoids loading any real HF model and is device-agnostic (CPU/MPS/CUDA).
    """

    import torch

    class DummyModel:
        def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Always return a fixed token sequence (shape-compatible with input_ids).
            # In our wrapper, generate is called with keyword arguments only.
            input_ids = kwargs.get("input_ids")
            return input_ids

    class DummyTokenizer:
        def __call__(self, text, return_tensors="pt"):  # type: ignore[no-untyped-def]
            # Return an object with an input_ids tensor.
            return type("obj", (), {"input_ids": torch.ones((1, 3), dtype=torch.long)})

        def decode(self, ids, skip_special_tokens=True):  # type: ignore[no-untyped-def]
            # Return a simple string regardless of ids.
            return "Hello from dummy model"

    cfg = LLMConfig.from_env()
    llm = MedicalReasoningLLM(config=cfg)

    # Monkeypatch internal artifacts so we don't load a real model or touch GPUs.
    from app.llm import reasoning_model

    llm._artifacts = reasoning_model._ModelArtifacts(  # type: ignore[attr-defined]
        tokenizer=DummyTokenizer(),
        model=DummyModel(),
        device=torch.device("cpu"),
    )

    out = llm.generate("Hello, world", max_tokens=8)
    assert isinstance(out, str)
    assert out.strip() != ""


def test_timeout_behavior(monkeypatch):
    """Ensure that a slow generate path raises TimeoutError when configured."""

    class DummyModel:
        def generate(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            import time

            time.sleep(0.2)
            return args[0]

    class DummyTokenizer:
        def __call__(self, text, return_tensors="pt"):  # type: ignore[no-untyped-def]
            import torch

            return type("obj", (), {"input_ids": torch.zeros((1, 1), dtype=torch.long)})

        def decode(self, ids, skip_special_tokens=True):  # type: ignore[no-untyped-def]
            return "dummy"

    cfg = LLMConfig.from_env()
    cfg.timeout_seconds = 0  # force immediate timeout

    llm = MedicalReasoningLLM(config=cfg)

    # Monkeypatch internal artifacts so we don't load a real model.
    from app.llm import reasoning_model

    reasoning_model._ModelArtifacts  # touch for coverage

    llm._artifacts = reasoning_model._ModelArtifacts(  # type: ignore[attr-defined]
        tokenizer=DummyTokenizer(),
        model=DummyModel(),
        device=None,  # type: ignore[arg-type]
    )

    try:
        llm.generate("slow")
    except TimeoutError:
        # Expected path
        return
    assert False, "Expected TimeoutError to be raised"

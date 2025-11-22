from __future__ import annotations

import os
from typing import Optional, Tuple

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = os.environ.get("AGENT_MODEL", "t5-small")


def load_agent_model(model_name: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:  # type: ignore[name-defined]
    """Load tokenizer and seq2seq model for the clinical reasoning agent.

    Defaults to t5-small but can be switched via env AGENT_MODEL or parameter.
    """

    name = model_name or MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tokenizer, model

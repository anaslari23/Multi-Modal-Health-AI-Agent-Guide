"""Configuration helpers for the medical reasoning LLM.

All configuration is driven by environment variables so operators can tune
runtime behavior without code changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """Configuration for the medical reasoning LLM runtime."""

    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    device: str = "auto"  # auto|cpu|cuda
    device_map: Optional[str] = "auto"  # auto|balanced|sequential|null
    max_tokens: int = 1024
    streaming: bool = False
    quantize: str = "none"  # none|bitsandbytes|gptq|gguf
    timeout_seconds: int = 120
    max_input_length: int = 4096

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build config from environment variables."""

        # model_name = os.getenv("MMHIE_REASONER_MODEL", cls.model_name)
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Much faster on CPU
        print(f"DEBUG: FORCED model_name: {model_name}")

        # Optional test override to avoid loading very large models in CI.
        test_override = os.getenv("MMHIE_REASONER_MODEL_TEST_OVERRIDE")
        if test_override:
            model_name = test_override

        device = os.getenv("MMHIE_DEVICE", cls.device)

        raw_device_map = os.getenv("MMHIE_DEVICE_MAP", "auto")
        if raw_device_map.lower() == "null":
            device_map: Optional[str] = None
        else:
            device_map = raw_device_map

        max_tokens = int(os.getenv("MMHIE_MAX_TOKENS", str(cls.max_tokens)))

        streaming_env = os.getenv("MMHIE_STREAMING", str(cls.streaming)).lower()
        streaming = streaming_env in {"1", "true", "yes", "on"}

        quantize = os.getenv("MMHIE_QUANTIZE", cls.quantize)

        timeout_seconds = int(os.getenv("MMHIE_TIMEOUT_SECONDS", str(cls.timeout_seconds)))

        max_input_length = int(os.getenv("MMHIE_MAX_INPUT_LENGTH", str(cls.max_input_length)))

        return cls(
            model_name=model_name,
            device=device,
            device_map=device_map,
            max_tokens=max_tokens,
            streaming=streaming,
            quantize=quantize,
            timeout_seconds=timeout_seconds,
            max_input_length=max_input_length,
        )

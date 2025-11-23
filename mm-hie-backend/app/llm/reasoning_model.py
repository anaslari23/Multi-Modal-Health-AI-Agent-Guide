from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Generator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import LLMConfig


logger = logging.getLogger(__name__)


@dataclass
class _ModelArtifacts:
    # No longer needed for Ollama but keeping for compatibility if needed
    pass


class MedicalReasoningLLM:
    """Singleton-style wrapper around Ollama LLM service.
    
    Replaces the previous local Transformers implementation to improve performance
    and stability by offloading inference to the Ollama service.
    """

    _instance_lock = threading.Lock()

    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig.from_env()
        # Default to tinyllama for speed, or use what's in config
        self.model_name = self.config.model_name or "tinyllama:latest"
        # Use the standard generate endpoint
        self.ollama_url = "http://127.0.0.1:11434/api/generate"

    def _truncate_prompt(self, prompt: str) -> str:
        if len(prompt) <= self.config.max_input_length:
            return prompt
        logger.warning(
            "Prompt length %d exceeds max_input_length=%d; truncating.",
            len(prompt),
            self.config.max_input_length,
        )
        return prompt[-self.config.max_input_length :]

    def _response_postprocess(self, text: str) -> str:
        for marker in ["<NAME>", "<PATIENT>", "<PHONE>", "<EMAIL>"]:
            text = text.replace(marker, "[REDACTED]")
        return text

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> str:
        """Generate text using Ollama API (Legacy completion)."""
        
        prompt = self._truncate_prompt(prompt)
        effective_max_tokens = min(max_tokens or self.config.max_tokens, self.config.max_tokens)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": effective_max_tokens,
                "top_p": 0.95,
            }
        }

        try:
            import httpx
            logger.info(f"Calling Ollama generate ({self.model_name})")
            
            response = httpx.post(
                self.ollama_url, 
                json=payload, 
                timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
            
            result = response.json()
            text = result.get("response", "")
            
            return self._response_postprocess(text)
            
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate chat response using Ollama /api/chat."""
        
        effective_max_tokens = min(max_tokens or self.config.max_tokens, self.config.max_tokens)
        
        # Use the chat endpoint
        chat_url = self.ollama_url.replace("/api/generate", "/api/chat")

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": effective_max_tokens,
            }
        }

        try:
            import httpx
            logger.info(f"Calling Ollama chat ({self.model_name})")
            
            response = httpx.post(
                chat_url, 
                json=payload, 
                timeout=self.config.timeout_seconds
            )
            response.raise_for_status()
            
            result = response.json()
            # Ollama chat response format: {"message": {"role": "assistant", "content": "..."}}
            text = result.get("message", {}).get("content", "")
            
            return self._response_postprocess(text)
            
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise RuntimeError(f"Ollama chat failed: {e}")

    def stream_generate(
        self,
        prompt: str,
        chunk_size: int = 64,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
    ) -> Generator[str, None, None]:
        """Yield generated text in chunks using Ollama streaming."""
        
        prompt = self._truncate_prompt(prompt)
        effective_max_tokens = min(max_tokens or self.config.max_tokens, self.config.max_tokens)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": effective_max_tokens,
            }
        }

        try:
            import httpx
            import json
            
            with httpx.stream("POST", self.ollama_url, json=payload, timeout=self.config.timeout_seconds) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            chunk = data.get("response", "")
                            if chunk:
                                yield chunk
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Ollama streaming failed: {e}")
            yield f"[Error: {e}]"

    def embed(self, text: str):
        raise NotImplementedError("Use dedicated embedding service")


# Module-level singleton for convenient import.
medical_reasoner = MedicalReasoningLLM()

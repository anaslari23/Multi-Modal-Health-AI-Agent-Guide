from __future__ import annotations

import logging
from typing import Any, AsyncGenerator, Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .reasoning_model import medical_reasoner


logger = logging.getLogger(__name__)

app = FastAPI(title="MM-HIE Medical Reasoning LLM Service", version="0.1.0")


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    stream: bool = False
    temperature: float = 0.0


@app.get("/healthz")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Dict[str, str]:
    # Placeholder for Prometheus or other metrics integration.
    return {"metrics": "not_implemented"}


async def _stream_chunks(prompt: str, max_tokens: Optional[int], temperature: float):
    for chunk in medical_reasoner.stream_generate(prompt, max_tokens=max_tokens, temperature=temperature):
        yield chunk


@app.post("/llm/generate")
async def llm_generate(payload: GenerateRequest):
    """Generate text from the medical reasoning LLM.

    WARNING: Outputs are not clinical advice and must be reviewed by a
    qualified clinician before any medical use.
    """

    try:
        if payload.stream:
            generator = _stream_chunks(payload.prompt, payload.max_tokens, payload.temperature)
            return StreamingResponse(generator, media_type="text/plain")

        text = medical_reasoner.generate(
            prompt=payload.prompt,
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
        )
        return JSONResponse({"text": text})
    except TimeoutError as exc:
        logger.warning("LLM generation timed out: %s", exc)
        raise HTTPException(status_code=504, detail="LLM generation timed out") from exc
    except Exception as exc:  # pragma: no cover - generic error path
        logger.exception("LLM generation failed: %s", exc)
        raise HTTPException(status_code=500, detail="LLM generation failed") from exc

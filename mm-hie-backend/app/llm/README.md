# Medical Reasoning LLM Integration

This package integrates a large medical reasoning language model (default:
`dousery/medical-reasoning-gpt-oss-20b`) into the MM-HIE backend.

It provides:

- **`MedicalReasoningLLM`**: singleton-style wrapper around a Hugging Face
  `transformers` causal LM with device-aware loading.
- **`medical_reasoner`**: module-level singleton instance for convenient use.
- **FastAPI microservice** (`reasoning_service.py`) for remote deployment.
- **Env-driven configuration** (`config.py`).

> **Safety notice**: Model outputs are _not_ clinical advice. They must not be
> used for diagnosis or treatment decisions without review by a licensed
> clinician.

## Configuration

Environment variables:

- `MMHIE_REASONER_MODEL` (default: `dousery/medical-reasoning-gpt-oss-20b`)
- `MMHIE_DEVICE` (default: `auto`) — one of `auto|cpu|cuda`.
- `MMHIE_DEVICE_MAP` (default: `auto`) — `auto|balanced|sequential|null`.
- `MMHIE_MAX_TOKENS` (default: `512`).
- `MMHIE_STREAMING` (default: `false`) — currently informational; streaming
  is simulated by chunking full outputs.
- `MMHIE_QUANTIZE` (default: `none`) — one of `none|bitsandbytes|gptq|gguf`.
  This wrapper does not enable quantization by itself but documents the
  intended mode for operators.
- `MMHIE_TIMEOUT_SECONDS` (default: `30`).
- `MMHIE_MAX_INPUT_LENGTH` (default: `4096`).
- `MMHIE_REASONER_MODEL_TEST_OVERRIDE` — optional; used by tests to force a
  small model (e.g. `sshleifer/tiny-gpt2`).

## Hardware guidance

For `dousery/medical-reasoning-gpt-oss-20b` (20B parameters), you should
expect roughly:

- **FP16**: 40–50 GB GPU VRAM.
- **8-bit / 4-bit quantization**: 16–24 GB VRAM, depending on config.
- **CPU-only**: not recommended for production; may require >64 GB RAM and
  will be slow.

**Recommendations:**

- Run large models on a dedicated GPU node or as a separate service.
- Prefer quantized checkpoints (bitsandbytes, GPTQ) or GGUF + `llama-cpp`
  for lower-resource environments.
- For constrained hosts, set `MMHIE_REASONER_MODEL` to a smaller model
  (e.g. `facebook/opt-1.3b`) or rely on a remote LLM service.

## Usage in backend code

Example (local singleton):

```python
from app.llm.reasoning_model import medical_reasoner

prompt = "Summarise the multimodal findings for this patient and suggest next steps."
summary = medical_reasoner.generate(prompt)
``

> **Note**: initial model load can be slow and memory intensive. Consider
> deferring LLM construction to a dedicated worker or microservice.

## FastAPI microservice

`reasoning_service.py` exposes:

- `GET /healthz` — simple health check.
- `GET /metrics` — stub for future metrics integration.
- `POST /llm/generate` — body `{ "prompt": "...", "max_tokens": 256, "stream": false }`.

When `stream=true`, the endpoint returns a chunked plain-text response with the
model output split into reasonable-sized chunks. This is sufficient for
progressive rendering in UIs.

### Running via Docker

A minimal Dockerfile can be found at `docker/llm.Dockerfile`. Example build
and run:

```bash
docker build -f docker/llm.Dockerfile -t mmhie-llm .

# CPU example
docker run --rm -p 8000:8000 \
  -e MMHIE_DEVICE=cpu \
  -e MMHIE_REASONER_MODEL=dousery/medical-reasoning-gpt-oss-20b \
  mmhie-llm

# GPU example (NVIDIA), assuming nvidia-container-runtime is configured
# and a compatible torch/transformers stack is installed in the image.
docker run --rm -p 8000:8000 --gpus all \
  -e MMHIE_DEVICE=cuda \
  -e MMHIE_DEVICE_MAP=auto \
  -e MMHIE_QUANTIZE=bitsandbytes \
  mmhie-llm
```

## Operational notes

- Prefer running the LLM microservice on a separate GPU node.
- Use quantized or GGUF-based variants when VRAM is limited.
- For embeddings, use a dedicated embedder (e.g. GGUF embedder in
  `mm-hie-core/shared/embeddings`) rather than this causal LM wrapper.

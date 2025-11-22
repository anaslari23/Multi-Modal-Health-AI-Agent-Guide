FROM python:3.11-slim

# Basic OS deps (adjust as needed for GPU builds).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only backend and minimal requirements.
COPY mm-hie-backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn[standard] transformers torch

COPY mm-hie-backend /app/mm-hie-backend

ENV PYTHONPATH=/app/mm-hie-backend
ENV MMHIE_DEVICE=auto
ENV MMHIE_DEVICE_MAP=auto
ENV MMHIE_REASONER_MODEL=dousery/medical-reasoning-gpt-oss-20b

EXPOSE 8000

CMD ["uvicorn", "app.llm.reasoning_service:app", "--host", "0.0.0.0", "--port", "8000"]

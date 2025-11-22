# MM-HIE Agent Backend

FastAPI backend for the MM-HIE multimodal agent. Exposes REST endpoints for case creation, symptom NLP, lab OCR, imaging classification, vitals analysis, fusion, XAI, and PDF report generation.

## Structure

- `app/main.py` – FastAPI app and routes
- `app/orchestrator.py` – case management and fusion orchestration
- `app/modules/` – NLP, OCR, imaging, timeseries modules
- `app/fusion/meta_learner.py` – weighted fusion meta-learner
- `app/xai/` – XAI aggregators
- `app/utils/pdf_report.py` – PDF report generation
- `models/` – saved models, Grad-CAMs, reports
- `tests/` – unit and integration tests

## Running locally

```bash
cd mm-hie-backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then visit `http://localhost:8000/docs` for the OpenAPI UI.

## Docker

Build and run via Docker:

```bash
cd mm-hie-backend
docker-compose up --build
```

The API will be exposed at `http://localhost:8000`.

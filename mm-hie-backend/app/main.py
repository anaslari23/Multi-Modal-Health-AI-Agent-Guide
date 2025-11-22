from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sqlalchemy.orm import Session

from .schemas import (
    CaseCreate,
    SymptomInput,
    VitalsInput,
    AnalysisResponse,
    ReportResponse,
    AgentPosteriorTimelineResponse,
    LocalTrainRequest,
    LocalTrainResponse,
)
from .orchestrator_instance import orchestrator
from .database import get_db
from .routers.cases_router import router as cases_router
from .routers.timeline_router import router as timeline_router
from .routers.reports_router import router as reports_router
from .routers.medicines_router import router as medicines_router
from .agent.agent_router import router as agent_router
from .rag.rag_router import router as rag_router
from .routers.medical_llm_router import router as medical_llm_router
from .federated.fed_trainer import get_federated_trainer

app = FastAPI(title="MM-HIE Agent Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (e.g. Grad-CAM PNGs) from the project root.
app.mount("/static", StaticFiles(directory="."), name="static")

orchestrator = orchestrator


@app.post("/cases/", response_model=str)
async def create_case(case: CaseCreate, db: Session = Depends(get_db)) -> str:
    case_id = orchestrator.create_case(case, db)
    return case_id


@app.post("/cases/{case_id}/symptoms")
async def add_symptoms(case_id: str, payload: SymptomInput, db: Session = Depends(get_db)):
    try:
        return orchestrator.add_symptoms(case_id, payload, db)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")


@app.post("/train/local", response_model=LocalTrainResponse)
async def train_local(payload: LocalTrainRequest):
    """Prototype endpoint to run a local federated training round.

    This endpoint does *not* touch patient data. It performs a small synthetic
    local update on a toy model and applies FedAvg with the current global
    state, returning the new logical version number.
    """

    trainer = get_federated_trainer()
    entry = trainer.run_local_round(
        model_name=payload.model_name,
        epochs=payload.epochs,
        lr=payload.learning_rate,
    )
    return LocalTrainResponse(model_name=entry.name, version=entry.version)


@app.get("/cases/{case_id}/agent-timeline", response_model=AgentPosteriorTimelineResponse)
async def get_agent_timeline(case_id: str):
    try:
        return orchestrator.get_agent_timeline(case_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")


@app.post("/cases/{case_id}/upload-report")
async def upload_lab_report(case_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        return await orchestrator.add_lab_report(case_id, file, db)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")


@app.post("/cases/{case_id}/upload-image")
async def upload_image(case_id: str, file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        return await orchestrator.add_image(case_id, file, db)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")


@app.post("/cases/{case_id}/vitals")
async def add_vitals(case_id: str, payload: VitalsInput, db: Session = Depends(get_db)):
    try:
        return orchestrator.add_vitals(case_id, payload, db)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")


@app.get("/cases/{case_id}/analysis", response_model=AnalysisResponse)
async def get_analysis(case_id: str, db: Session = Depends(get_db)):
    try:
        return orchestrator.run_analysis(case_id, db)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")


@app.get("/cases/{case_id}/report", response_model=ReportResponse)
async def get_report(case_id: str, db: Session = Depends(get_db)):
    try:
        return orchestrator.generate_report(case_id, db)
    except KeyError:
        raise HTTPException(status_code=404, detail="Case not found")


app.include_router(cases_router)
app.include_router(timeline_router)
app.include_router(reports_router)
app.include_router(medicines_router)
app.include_router(agent_router)
app.include_router(rag_router)
app.include_router(medical_llm_router)

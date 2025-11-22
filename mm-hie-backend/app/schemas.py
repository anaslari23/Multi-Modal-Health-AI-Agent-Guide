from __future__ import annotations

from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field


class CaseCreate(BaseModel):
    patient_id: str
    notes: Optional[str] = None


class SymptomInput(BaseModel):
    text: str = Field(..., description="Free text symptom description")
    top_n: int = 5


class ConditionProb(BaseModel):
    condition: str
    prob: float


class SymptomOutput(BaseModel):
    conditions: List[ConditionProb]
    embedding: List[float]


class LabValue(BaseModel):
    value: float
    flag: str


class LabResults(BaseModel):
    values: Dict[str, LabValue]


class ImagingOutput(BaseModel):
    probabilities: Dict[str, float]
    gradcam_path: Optional[str]
    embedding: List[float]
    # Optional signed URL for the Grad-CAM overlay when stored in S3.
    gradcam_url: Optional[str] = None


class VitalsInput(BaseModel):
    heart_rate: List[float]
    spo2: List[float]
    temperature: List[float]
    resp_rate: List[float]


class VitalsOutput(BaseModel):
    vitals_risk: float
    anomalies: List[str]
    embedding: List[float]
    heart_rate: List[float]
    spo2: List[float]


class FusionOutput(BaseModel):
    final_risk_score: float
    triage: str
    conditions: List[ConditionProb]

    # Optional per-modality contribution scores in [0, 1] for UI visualisation.
    modality_scores: Optional[Dict[str, float]] = None


class XAIOutput(BaseModel):
    summary: str
    gradcam_path: Optional[str]
    labs_shap_path: Optional[str] = None
    nlp_highlights: Optional[Dict[str, float]] = None
    # Optional signed URLs for S3-hosted artefacts.
    gradcam_url: Optional[str] = None
    labs_shap_url: Optional[str] = None


class AnalysisResponse(BaseModel):
    nlp: Optional[SymptomOutput]
    labs: Optional[LabResults]
    imaging: Optional[ImagingOutput]
    vitals: Optional[VitalsOutput]
    fusion: Optional[FusionOutput]
    xai: Optional[XAIOutput]
    posterior_probabilities: Optional[List[ConditionProb]] = None
    agent_summary: Optional[str] = None


class PosteriorHistoryEntry(BaseModel):
    step: int
    modalities: Dict[str, bool]
    posterior: List[ConditionProb]


class AgentActionState(BaseModel):
    trigger: str
    next_steps: List[str]
    agent_reasoning: str
    confidence_gain: Optional[float] = None


class AgentPosteriorTimelineResponse(BaseModel):
    posterior_history: List[PosteriorHistoryEntry]
    agent_actions: List[AgentActionState]


class ReportResponse(BaseModel):
    case_id: str
    pdf_path: str
    # Optional signed URL for downloading the PDF from S3.
    pdf_url: Optional[str] = None


class LocalTrainRequest(BaseModel):
    model_name: str = Field("multimodal_fusion", description="Logical name of the model to train")
    epochs: int = Field(1, ge=1, le=10, description="Number of local epochs")
    learning_rate: float = Field(1e-2, gt=0, description="Local learning rate")


class LocalTrainResponse(BaseModel):
    model_name: str
    version: int

"""
Router for fine-tuned medical LLM endpoints.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/medical-llm", tags=["medical-llm"])


class MedicineSuggestionRequest(BaseModel):
    disease: str
    patient_info: Optional[str] = None


class MedicineInfoRequest(BaseModel):
    medicine_name: str


class LLMResponse(BaseModel):
    response: str
    model: str = "fine-tuned-tinyllama"


@router.post("/suggest", response_model=LLMResponse)
async def suggest_medicine(request: MedicineSuggestionRequest):
    """Suggest medicines for a disease using fine-tuned model."""
    try:
        from app.rag.medical_llm import get_medical_llm
        
        llm = get_medical_llm()
        response = llm.medicine_suggestion(
            disease=request.disease,
            patient_info=request.patient_info
        )
        
        return LLMResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")


@router.post("/info", response_model=LLMResponse)
async def medicine_info(request: MedicineInfoRequest):
    """Get information about a specific medicine."""
    try:
        from app.rag.medical_llm import get_medical_llm
        
        llm = get_medical_llm()
        response = llm.medicine_info(medicine_name=request.medicine_name)
        
        return LLMResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")


@router.post("/contraindications", response_model=LLMResponse)
async def get_contraindications(request: MedicineInfoRequest):
    """Get contraindications for a medicine."""
    try:
        from app.rag.medical_llm import get_medical_llm
        
        llm = get_medical_llm()
        response = llm.contraindications(medicine_name=request.medicine_name)
        
        return LLMResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")


@router.get("/health")
async def health_check():
    """Check if the fine-tuned model is available."""
    try:
        from pathlib import Path
        model_path = Path("./models/medical-medicine-lora")
        
        if not model_path.exists():
            return {
                "status": "unavailable",
                "message": "Fine-tuned model not found"
            }
        
        return {
            "status": "available",
            "model_path": str(model_path),
            "message": "Fine-tuned medical model is ready"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

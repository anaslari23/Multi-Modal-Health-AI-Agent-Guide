from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from .rag_engine import rag


router = APIRouter(prefix="/rag", tags=["rag"])


class RAGQueryRequest(BaseModel):
    symptoms: Optional[str] = None
    question: Optional[str] = None
    patient_info: Optional[str] = None
    top_k: int = 5


class RAGResponse(BaseModel):
    answer: str
    context: List[dict]


@router.post("/query", response_model=RAGResponse)
async def rag_query(payload: RAGQueryRequest) -> RAGResponse:
    if not payload.question and not payload.symptoms:
        raise HTTPException(status_code=400, detail="Either question or symptoms must be provided")
    result = rag.query(
        question=payload.question,
        symptoms=payload.symptoms,
        patient_info=payload.patient_info,
        top_k=payload.top_k,
    )
    return RAGResponse(answer=result["answer"], context=result["context"])


@router.post("/diagnose", response_model=RAGResponse)
async def rag_diagnose(payload: RAGQueryRequest) -> RAGResponse:
    result = rag.diagnose(
        symptoms=payload.symptoms or payload.question or "",
        patient_info=payload.patient_info,
        top_k=payload.top_k,
    )
    return RAGResponse(answer=result["answer"], context=result["context"])


@router.post("/explain", response_model=RAGResponse)
async def rag_explain(payload: RAGQueryRequest) -> RAGResponse:
    result = rag.explain(
        question=payload.question or "Explain the current diagnosis.",
        symptoms=payload.symptoms,
        patient_info=payload.patient_info,
        top_k=payload.top_k,
    )
    return RAGResponse(answer=result["answer"], context=result["context"])


@router.post("/treatment", response_model=RAGResponse)
async def rag_treatment(payload: RAGQueryRequest) -> RAGResponse:
    result = rag.treatment(
        question=payload.question or "Treatment options?",
        symptoms=payload.symptoms,
        patient_info=payload.patient_info,
        top_k=payload.top_k,
    )
    return RAGResponse(answer=result["answer"], context=result["context"])


@router.post("/drug", response_model=RAGResponse)
async def rag_drug(payload: RAGQueryRequest) -> RAGResponse:
    result = rag.drug_info(
        question=payload.question or payload.symptoms or "",
        patient_info=payload.patient_info,
        top_k=payload.top_k,
    )
    return RAGResponse(answer=result["answer"], context=result["context"])

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import crud
from app.db_models import Case


router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("/", response_model=list[dict])
def list_cases(db: Session = Depends(get_db)):
    cases = crud.get_all_cases(db)
    return [
        {
            "id": str(c.id),
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat(),
            "patient_name": c.patient_name,
            "patient_age": c.patient_age,
            "patient_gender": c.patient_gender,
            "status": c.status,
        }
        for c in cases
    ]


@router.get("/{case_id}", response_model=dict)
def get_case_detail(case_id: str, db: Session = Depends(get_db)):
    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case id")

    case: Optional[Case] = crud.get_case(db, case_uuid)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    return {
        "id": str(case.id),
        "created_at": case.created_at.isoformat(),
        "updated_at": case.updated_at.isoformat(),
        "patient_name": case.patient_name,
        "patient_age": case.patient_age,
        "patient_gender": case.patient_gender,
        "status": case.status,
    }


@router.delete("/{case_id}", status_code=204)
def delete_case(case_id: str, db: Session = Depends(get_db)):
    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case id")

    success = crud.delete_case(db, case_uuid)
    if not success:
        raise HTTPException(status_code=404, detail="Case not found")
    return

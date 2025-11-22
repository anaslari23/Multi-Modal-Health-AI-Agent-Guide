from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app import crud


router = APIRouter(prefix="/cases", tags=["timeline"])


@router.get("/{case_id}/timeline", response_model=list[dict])
def get_case_timeline(case_id: str, db: Session = Depends(get_db)):
    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case id")

    timeline = crud.get_timeline(db, case_uuid)
    return timeline

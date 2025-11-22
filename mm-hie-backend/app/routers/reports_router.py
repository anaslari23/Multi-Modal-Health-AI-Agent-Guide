from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.db_models import Report


router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/{case_id}", response_model=dict)
def get_latest_report(case_id: str, db: Session = Depends(get_db)):
    try:
        case_uuid = uuid.UUID(case_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid case id")

    stmt = (
        select(Report)
        .where(Report.case_id == case_uuid)
        .order_by(Report.created_at.desc())
        .limit(1)
    )
    report = db.scalar(stmt)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")

    return {"case_id": str(report.case_id), "pdf_path": report.pdf_path, "created_at": report.created_at.isoformat()}

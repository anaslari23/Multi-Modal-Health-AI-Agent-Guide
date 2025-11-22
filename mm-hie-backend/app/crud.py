from __future__ import annotations

import uuid
from typing import Any, List, Dict, Optional, Union

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import schemas
from .db_models import (
    Case,
    Modality,
    AnalysisResult,
    XAIOutput,
    AgentAction,
    Report,
    AuditLog,
)


def create_case(
    db: Session,
    patient_name: str,
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
    status: str = "in_progress",
) -> Case:
    case = Case(
        patient_name=patient_name,
        patient_age=patient_age,
        patient_gender=patient_gender,
        status=status,
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def get_case(db: Session, case_id: Union[uuid.UUID, str]) -> Optional[Case]:
    """Get case by ID, accepting either UUID object or string."""
    # Don't convert to UUID - database stores IDs as VARCHAR strings
    stmt = select(Case).where(Case.id == str(case_id))
    return db.scalar(stmt)


def get_all_cases(db: Session) -> List[Case]:
    stmt = select(Case).order_by(Case.created_at.desc())
    return list(db.scalars(stmt))


def add_modality(
    db: Session,
    case_id: uuid.UUID,
    modality_type: str,
    payload: Optional[Dict[str, Any]],
    processed: Optional[Dict[str, Any]],
) -> Modality:
    modality = Modality(
        case_id=case_id,
        type=modality_type,
        payload=payload,
        processed=processed,
    )
    db.add(modality)
    db.commit()
    db.refresh(modality)
    return modality


def save_analysis_result(
    db: Session,
    case_id: uuid.UUID,
    risk_score: float,
    triage: str,
    conditions: Optional[Dict[str, Any]],
    fusion_vector: Optional[Dict[str, Any]],
    agent_summary: Optional[str] = None,
) -> AnalysisResult:
    result = AnalysisResult(
        case_id=case_id,
        risk_score=risk_score,
        triage=triage,
        conditions=conditions,
        fusion_vector=fusion_vector,
        agent_summary=agent_summary,
    )
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


def save_xai_output(
    db: Session,
    case_id: uuid.UUID,
    labs_shap_path: Optional[str],
    nlp_highlights: Optional[Dict[str, Any]],
    gradcam_path: Optional[str],
    explanation_text: Optional[str],
) -> XAIOutput:
    xai = XAIOutput(
        case_id=case_id,
        labs_shap_path=labs_shap_path,
        nlp_highlights=nlp_highlights,
        gradcam_path=gradcam_path,
        explanation_text=explanation_text,
    )
    db.add(xai)
    db.commit()
    db.refresh(xai)
    return xai


def save_agent_action(
    db: Session,
    case_id: uuid.UUID,
    step_number: int,
    action_type: str,
    action_detail: Optional[Dict[str, Any]],
    confidence_gain: Optional[float],
) -> AgentAction:
    action = AgentAction(
        case_id=case_id,
        step_number=step_number,
        action_type=action_type,
        action_detail=action_detail,
        confidence_gain=confidence_gain,
    )
    db.add(action)
    db.commit()
    db.refresh(action)
    return action


def save_pdf_report(
    db: Session,
    case_id: uuid.UUID,
    pdf_path: str,
) -> Report:
    report = Report(case_id=case_id, pdf_path=pdf_path)
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def log_event(
    db: Session,
    case_id: uuid.UUID,
    event: str,
    description: Optional[str] = None,
) -> AuditLog:
    log = AuditLog(case_id=case_id, event=event, description=description)
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def get_timeline(db: Session, case_id: uuid.UUID) -> List[Dict[str, Any]]:
    stmt_modalities = select(Modality).where(Modality.case_id == case_id)
    stmt_actions = select(AgentAction).where(AgentAction.case_id == case_id)
    stmt_analysis = select(AnalysisResult).where(AnalysisResult.case_id == case_id)
    stmt_xai = select(XAIOutput).where(XAIOutput.case_id == case_id)
    stmt_reports = select(Report).where(Report.case_id == case_id)
    stmt_audit = select(AuditLog).where(AuditLog.case_id == case_id)

    modalities = list(db.scalars(stmt_modalities))
    actions = list(db.scalars(stmt_actions))
    analyses = list(db.scalars(stmt_analysis))
    xais = list(db.scalars(stmt_xai))
    reports = list(db.scalars(stmt_reports))
    audits = list(db.scalars(stmt_audit))

    items: List[Dict[str, Any]] = []

    for m in modalities:
        items.append(
            {
                "type": f"modality_{m.type}",
                "created_at": m.created_at.isoformat(),
                "payload": m.payload,
                "processed": m.processed,
            }
        )

    for a in actions:
        items.append(
            {
                "type": "agent_action",
                "created_at": a.created_at.isoformat(),
                "step_number": a.step_number,
                "action_type": a.action_type,
                "action_detail": a.action_detail,
                "confidence_gain": a.confidence_gain,
            }
        )

    for r in analyses:
        items.append(
            {
                "type": "analysis_result",
                "created_at": r.created_at.isoformat(),
                "risk_score": r.risk_score,
                "triage": r.triage,
                "conditions": r.conditions,
                "fusion_vector": r.fusion_vector,
                "agent_summary": r.agent_summary,
            }
        )

    for x in xais:
        items.append(
            {
                "type": "xai_output",
                "created_at": x.created_at.isoformat(),
                "labs_shap_path": x.labs_shap_path,
                "nlp_highlights": x.nlp_highlights,
                "gradcam_path": x.gradcam_path,
                "explanation_text": x.explanation_text,
            }
        )

    for r in reports:
        items.append(
            {
                "type": "report",
                "created_at": r.created_at.isoformat(),
                "pdf_path": r.pdf_path,
            }
        )

    for l in audits:
        items.append(
            {
                "type": "audit_log",
                "created_at": l.created_at.isoformat(),
                "event": l.event,
                "description": l.description,
            }
        )

    items.sort(key=lambda x: x["created_at"])
    return items


def delete_case(db: Session, case_id: uuid.UUID) -> bool:
    case = get_case(db, case_id)
    if case:
        db.delete(case)
        db.commit()
        return True
    return False

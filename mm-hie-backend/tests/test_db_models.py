from __future__ import annotations

import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.db_models import Case, Modality, AnalysisResult, XAIOutput, AgentAction, Report, AuditLog


def _make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return TestingSessionLocal()


def test_case_and_relationships_basic():
    db = _make_session()

    case = Case(patient_name="John Doe", patient_age=42, patient_gender="M")
    db.add(case)
    db.commit()
    db.refresh(case)

    mod = Modality(case_id=case.id, type="symptoms", payload={"a": 1}, processed={"b": 2})
    db.add(mod)

    analysis = AnalysisResult(
        case_id=case.id,
        risk_score=0.9,
        triage="high",
        conditions={"x": 1},
        fusion_vector={"vec": [0.1, 0.2]},
    )
    db.add(analysis)

    xai = XAIOutput(case_id=case.id, labs_shap_path=None, nlp_highlights=None, gradcam_path=None, explanation_text="test")
    db.add(xai)

    action = AgentAction(case_id=case.id, step_number=1, action_type="test", action_detail={}, confidence_gain=0.1)
    db.add(action)

    report = Report(case_id=case.id, pdf_path="/tmp/test.pdf")
    db.add(report)

    log = AuditLog(case_id=case.id, event="evt", description="desc")
    db.add(log)

    db.commit()

    fetched = db.get(Case, case.id)
    assert fetched is not None
    assert len(fetched.modalities) == 1
    assert len(fetched.analysis_results) == 1
    assert len(fetched.xai_outputs) == 1
    assert len(fetched.agent_actions) == 1
    assert len(fetched.reports) == 1
    assert len(fetched.audit_logs) == 1

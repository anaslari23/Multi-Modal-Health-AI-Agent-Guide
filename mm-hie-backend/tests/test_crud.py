from __future__ import annotations

import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app import crud
from app.db_models import Case


def _make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return TestingSessionLocal()


def test_create_and_get_case():
    db = _make_session()

    created = crud.create_case(db, patient_name="Alice", patient_age=30, patient_gender="F")
    assert created.id is not None

    fetched = crud.get_case(db, created.id)
    assert fetched is not None
    assert fetched.patient_name == "Alice"


def test_get_all_cases():
    db = _make_session()
    crud.create_case(db, patient_name="A", patient_age=None, patient_gender=None)
    crud.create_case(db, patient_name="B", patient_age=None, patient_gender=None)

    all_cases = crud.get_all_cases(db)
    assert len(all_cases) == 2


def test_timeline_basic():
    db = _make_session()
    case = crud.create_case(db, patient_name="T", patient_age=None, patient_gender=None)

    crud.add_modality(db, case_id=case.id, modality_type="symptoms", payload={"x": 1}, processed={"y": 2})
    crud.save_agent_action(db, case_id=case.id, step_number=1, action_type="step", action_detail={"a": 1}, confidence_gain=0.5)
    crud.save_analysis_result(db, case_id=case.id, risk_score=0.7, triage="med", conditions=None, fusion_vector=None)
    crud.save_xai_output(db, case_id=case.id, labs_shap_path=None, nlp_highlights=None, gradcam_path=None, explanation_text="e")
    crud.save_pdf_report(db, case_id=case.id, pdf_path="/tmp/r.pdf")
    crud.log_event(db, case_id=case.id, event="evt", description="desc")

    timeline = crud.get_timeline(db, case.id)
    assert len(timeline) == 6
    types = {item["type"] for item in timeline}
    assert "modality_symptoms" in types
    assert "agent_action" in types
    assert "analysis_result" in types
    assert "xai_output" in types
    assert "report" in types
    assert "audit_log" in types

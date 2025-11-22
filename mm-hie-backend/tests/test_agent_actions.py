from __future__ import annotations

import uuid

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app import crud


def _make_session():
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return TestingSessionLocal()


def test_save_and_load_agent_action():
    db = _make_session()

    case = crud.create_case(db, patient_name="T", patient_age=None, patient_gender=None)

    action = crud.save_agent_action(
        db,
        case_id=case.id,
        step_number=1,
        action_type="add_symptoms",
        action_detail={"foo": "bar"},
        confidence_gain=0.42,
    )

    assert action.id is not None
    assert action.case_id == case.id

    # Ensure it appears in the timeline as an agent_action.
    timeline = crud.get_timeline(db, case.id)
    types = {item["type"] for item in timeline}
    assert "agent_action" in types
    agent_items = [item for item in timeline if item["type"] == "agent_action"]
    assert len(agent_items) == 1
    assert agent_items[0]["step_number"] == 1
    assert agent_items[0]["action_type"] == "add_symptoms"
    assert agent_items[0]["confidence_gain"] == 0.42

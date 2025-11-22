from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_timeline_includes_agent_actions(monkeypatch):
    # Prepare a fake timeline with an agent_action entry
    events = [
        {
            "type": "modality_symptoms",
            "created_at": "2020-01-01T00:00:00",
            "payload": {},
            "processed": {},
        },
        {
            "type": "agent_action",
            "created_at": "2020-01-01T00:00:01",
            "step_number": 1,
            "action_type": "add_symptoms",
            "action_detail": {"foo": "bar"},
            "confidence_gain": 0.5,
        },
    ]

    class DummyCrud:
        @staticmethod
        def get_timeline(db, case_id):  # pragma: no cover - simple passthrough
            return events

    from app import crud as real_crud

    monkeypatch.setattr("app.crud.get_timeline", DummyCrud.get_timeline)

    res = client.get(f"/cases/{uuid.uuid4()}/timeline")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    assert len(body) == 2

    types = {item["type"] for item in body}
    assert "modality_symptoms" in types
    assert "agent_action" in types

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_timeline_endpoint_exists_and_returns_list(monkeypatch):
    events = [
        {"type": "modality_symptoms", "created_at": "2020-01-01T00:00:00", "payload": {}, "processed": {}},
        {"type": "agent_action", "created_at": "2020-01-01T00:00:01", "step_number": 1},
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
    assert body[0]["type"] == "modality_symptoms"

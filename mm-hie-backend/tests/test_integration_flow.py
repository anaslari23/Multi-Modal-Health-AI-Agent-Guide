from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_full_case_flow():
    res = client.post("/cases/", json={"patient_id": "p1", "notes": "test"})
    assert res.status_code == 200
    case_id = res.json()

    res = client.post(f"/cases/{case_id}/symptoms", json={"text": "cough and fever", "top_n": 3})
    assert res.status_code == 200

    res = client.post(
        f"/cases/{case_id}/vitals",
        json={
            "heart_rate": [120, 110],
            "spo2": [90, 92],
            "temperature": [38.5],
            "resp_rate": [24],
        },
    )
    assert res.status_code == 200

    res = client.get(f"/cases/{case_id}/analysis")
    assert res.status_code == 200
    body = res.json()
    assert body["fusion"] is not None

    res = client.get(f"/cases/{case_id}/report")
    assert res.status_code == 200
    assert "pdf_path" in res.json()

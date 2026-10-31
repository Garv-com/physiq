import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from tests.helpers import seed_nutrition


@pytest.fixture
def client(temp_db):
    import main

    return TestClient(main.app)


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_nutrition_log_validates_input(client):
    good = client.post("/nutrition/log", json={"date": "2026-03-01", "calories": 2400, "weight": 75.0})
    assert good.status_code == 200
    assert client.post("/nutrition/log", json={"date": "2026-03-02", "calories": -5, "weight": 75.0}).status_code == 422
    assert client.post("/nutrition/log", json={"date": "banana", "calories": 2400, "weight": 75.0}).status_code == 422
    assert client.post("/nutrition/log", json={"date": "2026-03-02", "calories": 2400, "weight": 5}).status_code == 422


def test_training_set_logging_assigns_sessions(client):
    body = {"date": "2026-03-01", "exercise": "Squat", "weight_kg": 100, "reps": 5, "rpe": 8}
    first = client.post("/training/sets", json=body)
    second = client.post("/training/sets", json=body)
    assert first.status_code == 201
    assert second.json()["set_number"] == 2
    assert second.json()["session_id"] == first.json()["session_id"]


def test_training_set_validation(client):
    body = {"date": "2026-03-01", "exercise": "Squat", "weight_kg": 100, "reps": 0}
    assert client.post("/training/sets", json=body).status_code == 422


def test_unknown_exercise_is_a_404(client):
    assert client.get("/training/summary/Nope").status_code == 404


def test_nutrition_endpoints_refuse_when_there_is_too_little_data(client):
    for path in ["/nutrition/maintenance", "/nutrition/trend-status", "/nutrition/forecast", "/nutrition/anomalies"]:
        assert client.get(path).status_code == 400, path


def test_nutrition_endpoints_work_with_enough_data(client):
    seed_nutrition(days=60)

    maintenance = client.get("/nutrition/maintenance").json()
    assert {"estimated_maintenance_kcal", "days_used", "coverage"} <= set(maintenance)

    path = client.get("/nutrition/forecast/path?days=14").json()
    assert len(path) == 15
    assert client.get("/nutrition/trend-status").json()["status"] in {"stable", "gaining", "losing", "uncertain"}
    assert isinstance(client.get("/nutrition/anomalies").json(), list)
    assert len(client.get("/nutrition/history?days=30").json()) == 30
    assert client.get("/nutrition/maintenance/rolling").status_code == 200


def test_query_parameters_are_range_checked(client):
    assert client.get("/nutrition/forecast?days=0").status_code == 422
    assert client.get("/nutrition/forecast?days=500").status_code == 422
    assert client.get("/nutrition/anomalies?threshold=0.5").status_code == 422


def test_chat_rejects_malformed_conversations(client):
    assert client.post("/assistant/chat", json={"messages": []}).status_code == 422
    bad_order = {"messages": [{"role": "assistant", "content": "hi"}]}
    assert client.post("/assistant/chat", json=bad_order).status_code == 422
    too_long = {"messages": [{"role": "user", "content": "x" * 2001}]}
    assert client.post("/assistant/chat", json=too_long).status_code == 422


def test_chat_without_an_api_key_returns_503(client, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/assistant/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 503


def test_chat_passes_the_assistant_reply_through(client, monkeypatch):
    import main

    monkeypatch.setattr(main, "assistant_chat", lambda messages: {"reply": "ok", "tools_used": []})
    response = client.post("/assistant/chat", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert response.json()["reply"] == "ok"
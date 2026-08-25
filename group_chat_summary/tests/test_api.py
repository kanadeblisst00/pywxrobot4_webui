from fastapi.testclient import TestClient

from app.config import AppSettings
from app.main import create_app
from app.repository import Repository


def build_client(tmp_path):
    repository = Repository(tmp_path / "api.sqlite3")
    settings = AppSettings(data_dir=tmp_path)
    return TestClient(create_app(settings=settings, repository=repository))


def test_health_and_builtin_models(tmp_path):
    client = build_client(tmp_path)

    health = client.get("/api/v1/health")
    models = client.get("/api/v1/model-profiles")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert models.status_code == 200
    assert any(item["id"] == "llamacpp-qwen35-9b" for item in models.json())


def test_update_pipeline_settings(tmp_path):
    client = build_client(tmp_path)
    payload = client.get("/api/v1/settings").json()
    payload["chunk_max_chars"] = 9000

    response = client.put("/api/v1/settings", json=payload)

    assert response.status_code == 200
    assert response.json()["chunk_max_chars"] == 9000


def test_summary_rejects_unknown_model(tmp_path):
    client = build_client(tmp_path)
    response = client.post(
        "/api/v1/summaries",
        json={
            "room_name": "测试群",
            "model_profile_id": "missing",
            "messages": [{"id": "m1", "sender_name": "A", "content": "hello"}],
        },
    )

    assert response.status_code == 422

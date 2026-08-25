from __future__ import annotations

from pathlib import Path
from time import sleep

from fastapi.testclient import TestClient

from app.config import AppConfig
from app.main import create_app


def make_config(tmp_path: Path, *, token: str = "") -> AppConfig:
    data_dir = tmp_path / "data"
    return AppConfig(
        host="127.0.0.1",
        port=28110,
        database_path=data_dir / "test.sqlite3",
        api_token=token,
        max_context_messages=8,
        data_dir=data_dir,
        artifacts_dir=data_dir / "artifacts",
        datasets_dir=data_dir / "datasets",
        web_dir=Path(__file__).resolve().parents[1] / "static",
    )


def test_health_topics_and_dashboard(tmp_path: Path) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    assert client.get("/api/health").status_code == 200
    topics = client.get("/api/v1/topics").json()
    assert {item["name"] for item in topics} >= {"资金借贷请求", "色情", "赌博", "广告"}
    dashboard = client.get("/api/v1/stats").json()
    assert dashboard["enabled_topics"] == 4
    assert dashboard["samples"] >= 12


def test_analyze_detects_paraphrase_and_respects_negative_context(tmp_path: Path) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    positive = client.post("/api/v1/messages/analyze", json={
        "message_id": "m-1",
        "room_id": "room-1",
        "sender_name": "小王",
        "text": "能不能先支援两千，下周还你",
        "persist": False,
    })
    assert positive.status_code == 200, positive.text
    result = positive.json()
    assert result["matched"] is True
    assert result["matches"][0]["topic_name"] == "资金借贷请求"

    negative = client.post("/api/v1/messages/analyze", json={
        "room_id": "room-1",
        "text": "大家千万别在群里借钱，小心被骗",
        "context": ["管理员：这是反诈提醒"],
        "persist": False,
    })
    assert negative.status_code == 200
    assert not any(item["matched"] and item["topic_name"] == "资金借贷请求" for item in negative.json()["matches"])


def test_persist_event_and_feedback(tmp_path: Path) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.post("/api/v1/analyze", json={
        "message_id": "persist-1",
        "room_id": "risk-room",
        "sender_name": "测试用户",
        "text": "成人资源合集，想看的私聊",
    })
    assert response.status_code == 200, response.text
    event_id = response.json()["event_ids"][0]
    events = client.get("/api/v1/events").json()
    assert events["total"] >= 1
    feedback = client.post(f"/api/v1/events/{event_id}/feedback", json={"verdict": "correct", "note": "人工确认"})
    assert feedback.status_code == 200
    assert feedback.json()["feedback"] == "correct"


def test_model_settings_masks_secret_and_auth(tmp_path: Path) -> None:
    client = TestClient(create_app(make_config(tmp_path, token="secret-token")))
    assert client.get("/api/v1/topics").status_code == 401
    headers = {"X-API-Token": "secret-token"}
    settings = client.get("/api/v1/models/settings", headers=headers).json()
    settings.update({
        "profile": "custom",
        "embedding_provider": "openai_compatible",
        "embedding_model": "demo-embedding",
        "embedding_api_base": "http://localhost:9999",
        "embedding_api_key": "private-key",
    })
    saved = client.put("/api/v1/models/settings", headers=headers, json=settings)
    assert saved.status_code == 200, saved.text
    assert saved.json()["embedding_api_key"] == "••••••••"
    reread = client.get("/api/v1/models/settings", headers=headers).json()
    assert reread["embedding_api_key"] == "••••••••"


def test_topic_rejects_unknown_fields(tmp_path: Path) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.post("/api/v1/topics", json={
        "name": "测试主题",
        "examples": [{"text": "测试正例", "polarity": "positive"}],
        "unexpected": True,
    })
    assert response.status_code == 422


def test_builtin_dataset_training_and_activation(tmp_path: Path) -> None:
    with TestClient(create_app(make_config(tmp_path))) as client:
        datasets = client.get("/api/v1/datasets").json()
        builtin = next(item for item in datasets if item["slug"] == "builtin-wechat-moderation-seed")
        created = client.post("/api/v1/training/runs", json={
            "name": "测试轻量分类器",
            "algorithm": "char_ngram_nb",
            "dataset_ids": [builtin["id"]],
            "test_ratio": 0.2,
            "min_ngram": 1,
            "max_ngram": 3,
            "min_df": 1,
            "alpha": 1,
            "threshold": 0.52,
            "seed": 42,
        })
        assert created.status_code == 202, created.text
        run_id = created.json()["run_id"]
        run = None
        for _ in range(60):
            run = client.get(f"/api/v1/training/runs/{run_id}").json()
            if run["status"] in {"completed", "failed"}:
                break
            sleep(0.05)
        assert run and run["status"] == "completed", run
        assert run["metrics"]["micro_f1"] >= 0
        activated = client.post(f"/api/v1/models/activate/{run_id}")
        assert activated.status_code == 200, activated.text
        assert activated.json()["classifier_provider"] == "builtin_nb"

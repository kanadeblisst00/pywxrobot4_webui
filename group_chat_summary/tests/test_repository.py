from app.repository import Repository


def test_repository_seeds_builtin_profiles_and_settings(tmp_path):
    repository = Repository(tmp_path / "test.sqlite3")

    profiles = repository.list_model_profiles()
    settings = repository.get_settings()

    assert len(profiles) >= 8
    assert profiles[0]["id"] == "llamacpp-qwen35-9b"
    assert any(item["id"] == "llamacpp-qwen35-9b" for item in profiles)
    assert all(item["api_key"] == "" for item in profiles)
    assert settings["default_model_profile_id"] == "llamacpp-qwen35-9b"


def test_custom_model_profile_lifecycle(tmp_path):
    repository = Repository(tmp_path / "test.sqlite3")
    created = repository.create_model_profile(
        {
            "name": "测试模型",
            "provider": "openai-compatible",
            "base_url": "http://localhost:9000/v1/",
            "model": "local-model",
            "api_key": "secret",
            "enabled": True,
            "supports_json_schema": False,
            "description": "test",
        }
    )

    assert created["base_url"] == "http://localhost:9000/v1"
    assert created["api_key_configured"] is True
    assert created["api_key"] == ""

    updated = repository.update_model_profile(created["id"], {"name": "新名称", "enabled": False})
    assert updated["name"] == "新名称"
    assert updated["enabled"] is False
    assert repository.delete_model_profile(created["id"]) is True
    assert repository.get_model_profile(created["id"]) is None

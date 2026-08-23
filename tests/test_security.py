import pytest
from core.config import PluginServiceSettings

from server.security import extract_bearer_token, is_public_request_path


def test_extract_bearer_token() -> None:
    assert extract_bearer_token("Bearer secret-token") == "secret-token"
    assert extract_bearer_token("Basic abc") == ""
    assert extract_bearer_token("") == ""


def test_public_request_paths() -> None:
    assert is_public_request_path("/")
    assert is_public_request_path("/health")
    assert is_public_request_path("/static/js/app.js")
    assert not is_public_request_path("/api/overview")


@pytest.mark.parametrize("callback_path", ["/api/settings", "/static/upload", "/health", "/docs", "/openapi.json", "/redoc", "/plugins"])
def test_callback_path_rejects_reserved_routes(callback_path: str) -> None:
    with pytest.raises(ValueError):
        PluginServiceSettings(callback_path=callback_path)


def test_callback_path_normalizes_regular_route() -> None:
    settings = PluginServiceSettings(callback_path="custom/messages/")
    assert settings.callback_path == "/custom/messages"

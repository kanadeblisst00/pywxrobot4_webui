import asyncio

import httpx
import pytest

from core.config import PluginServiceSettings
from server import create_app
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


def test_security_middleware_adds_headers_without_breaking_docs() -> None:
    async def fetch_responses():
        app = create_app(PluginServiceSettings())
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            root_response = await client.get("/")
            docs_response = await client.get("/docs")
            return root_response, docs_response

    root_response, docs_response = asyncio.run(fetch_responses())
    assert root_response.headers["X-Content-Type-Options"] == "nosniff"
    assert root_response.headers["X-Frame-Options"] == "DENY"
    assert root_response.headers["Referrer-Policy"] == "no-referrer"
    assert "camera=()" in root_response.headers["Permissions-Policy"]
    content_security_policy = root_response.headers["Content-Security-Policy"]
    assert "default-src 'self'" in content_security_policy
    assert "frame-ancestors 'none'" in content_security_policy
    assert docs_response.headers["X-Content-Type-Options"] == "nosniff"
    assert "Content-Security-Policy" not in docs_response.headers

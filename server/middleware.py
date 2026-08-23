"""HTTP 安全中间件。"""

from __future__ import annotations

from fastapi import FastAPI, Request, Response

from runtime.engine import PluginRuntime
from server.security import is_public_request_path, verify_api_token, verify_callback_secret

DEFAULT_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}
CONTENT_SECURITY_POLICY = "default-src 'self'; base-uri 'self'; frame-ancestors 'none'; object-src 'none'; form-action 'self'; img-src 'self' data: blob: http: https:; connect-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; font-src 'self' data:"
CONTENT_SECURITY_POLICY_EXEMPT_PATHS = {"/docs", "/redoc"}


def apply_security_headers(response: Response, request_path: str) -> Response:
    for header_name, header_value in DEFAULT_SECURITY_HEADERS.items():
        response.headers.setdefault(header_name, header_value)
    normalized_path = str(request_path or "").rstrip("/") or "/"
    if normalized_path not in CONTENT_SECURITY_POLICY_EXEMPT_PATHS:
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
    return response


def register_security_middleware(app: FastAPI, runtime: PluginRuntime) -> None:
    @app.middleware("http")
    async def enforce_security_middleware(request: Request, call_next):
        request_path = str(request.url.path or "")
        if not is_public_request_path(request_path):
            current_settings = getattr(app.state, "plugin_runtime", runtime).settings
            callback_path = str(current_settings.callback_path or "/messages").rstrip("/") or "/messages"
            normalized_request_path = request_path.rstrip("/") or "/"
            if request.method.upper() == "POST" and normalized_request_path == callback_path:
                verify_callback_secret(request, current_settings)
            else:
                verify_api_token(request, current_settings)

        response = await call_next(request)
        return apply_security_headers(response, request_path)

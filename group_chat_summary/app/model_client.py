from __future__ import annotations

import json
import re
from typing import Any

import httpx


class ModelResponseError(RuntimeError):
    pass


def build_response_format(provider: str, schema: dict[str, Any]) -> dict[str, Any]:
    if str(provider or "").lower() == "llama.cpp":
        return {"type": "json_schema", "schema": schema}
    return {
        "type": "json_schema",
        "json_schema": {"name": "group_chat_summary", "strict": True, "schema": schema},
    }


def _extract_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelResponseError(f"模型响应缺少 choices: {payload}")
    choice = choices[0] if isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [str(item.get("text") or "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(texts).strip()
    raise ModelResponseError("模型没有返回文本内容")


def parse_json_content(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.IGNORECASE).strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ModelResponseError(f"模型未返回有效 JSON: {content[:500]}")
        try:
            parsed = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"模型返回的 JSON 无法解析: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ModelResponseError("模型返回的 JSON 顶层必须是对象")
    return parsed


class OpenAICompatibleClient:
    def __init__(self, profile: dict[str, Any], timeout_seconds: float = 180):
        self.profile = profile
        self.timeout_seconds = timeout_seconds

    @property
    def headers(self) -> dict[str, str]:
        api_key = str(self.profile.get("api_key") or "sk-no-key-required")
        return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    @property
    def base_url(self) -> str:
        return str(self.profile["base_url"]).rstrip("/")

    async def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
        temperature: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.profile["model"],
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        supports_schema = bool(self.profile.get("supports_json_schema", True))
        if supports_schema:
            payload["response_format"] = build_response_format(str(self.profile.get("provider") or ""), schema)

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload)
            if response.status_code in {400, 404, 422} and "response_format" in payload:
                payload.pop("response_format", None)
                response = await client.post(f"{self.base_url}/chat/completions", headers=self.headers, json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text[:1000]
                raise ModelResponseError(f"模型接口返回 HTTP {response.status_code}: {detail}") from exc
            try:
                response_payload = response.json()
            except ValueError as exc:
                raise ModelResponseError(f"模型接口返回了非 JSON 内容: {response.text[:500]}") from exc
        return parse_json_content(_extract_content(response_payload))

    async def check_connection(self) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(timeout=min(30, self.timeout_seconds)) as client:
                response = await client.get(f"{self.base_url}/models", headers=self.headers)
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            return {"ok": False, "installed": False, "detail": f"连接失败：{exc}", "models": []}

        raw_models = payload.get("data") if isinstance(payload, dict) else []
        models = []
        for item in raw_models if isinstance(raw_models, list) else []:
            model_name = item.get("id") if isinstance(item, dict) else item
            if model_name:
                models.append(str(model_name))
        expected = str(self.profile.get("model") or "")
        installed = expected in models
        detail = "连接正常，模型已安装" if installed else f"连接正常，但未在模型列表中找到 {expected}"
        return {"ok": True, "installed": installed, "detail": detail, "models": models}

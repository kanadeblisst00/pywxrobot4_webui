from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppSettings:
    host: str = os.getenv("SUMMARY_HOST", "127.0.0.1")
    port: int = int(os.getenv("SUMMARY_PORT", "28120"))
    data_dir: Path = Path(os.getenv("SUMMARY_DATA_DIR", str(PROJECT_DIR / "data"))).resolve()
    request_timeout_seconds: float = float(os.getenv("SUMMARY_REQUEST_TIMEOUT", "180"))
    cors_origins: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv("SUMMARY_CORS_ORIGINS", "").split(",")
        if item.strip()
    )

    @property
    def database_path(self) -> Path:
        return self.data_dir / "summary.sqlite3"


DEFAULT_SETTINGS = {
    "default_model_profile_id": "llamacpp-qwen35-9b",
    "chunk_max_chars": 12000,
    "chunk_overlap_messages": 2,
    "max_output_tokens": 2400,
    "temperature": 0.1,
    "keep_raw_messages": True,
    "ignored_message_types": ["emoji", "notice", "system", "revoke", "red_packet"],
    "custom_instruction": "重点提取明确结论、待办、负责人、截止时间、风险和未决问题。",
}


BUILTIN_MODEL_PROFILES = [
    {
        "id": "llamacpp-qwen35-9b",
        "name": "Qwen3.5 9B · llama.cpp（推荐）",
        "provider": "llama.cpp",
        "base_url": os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:18080/v1"),
        "model": "qwen3.5-9b",
        "api_key": "",
        "enabled": True,
        "supports_json_schema": True,
        "description": "默认推理后端；建议使用 Q4_K_M GGUF，约 5.7GB。",
    },
    {
        "id": "llamacpp-qwen35-4b",
        "name": "Qwen3.5 4B · llama.cpp（轻量）",
        "provider": "llama.cpp",
        "base_url": os.getenv("LLAMA_CPP_4B_BASE_URL", "http://127.0.0.1:18081/v1"),
        "model": "qwen3.5-4b",
        "api_key": "",
        "enabled": False,
        "supports_json_schema": True,
        "description": "低资源备选；启动 4B 服务后再启用。",
    },
    {
        "id": "llamacpp-qwen35-27b",
        "name": "Qwen3.5 27B · llama.cpp（高质量）",
        "provider": "llama.cpp",
        "base_url": os.getenv("LLAMA_CPP_27B_BASE_URL", "http://127.0.0.1:18082/v1"),
        "model": "qwen3.5-27b",
        "api_key": "",
        "enabled": False,
        "supports_json_schema": True,
        "description": "高质量备选；需要更大的显存或系统内存。",
    },
    {
        "id": "ollama-qwen35-4b",
        "name": "Qwen3.5 4B（轻量）",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3.5:4b",
        "api_key": "ollama",
        "enabled": False,
        "supports_json_schema": True,
        "description": "兼容保留：适合已有 Ollama 环境。",
    },
    {
        "id": "ollama-qwen35-9b",
        "name": "Qwen3.5 9B（推荐）",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3.5:9b",
        "api_key": "ollama",
        "enabled": False,
        "supports_json_schema": True,
        "description": "兼容保留：适合已有 Ollama 环境。",
    },
    {
        "id": "ollama-qwen35-27b",
        "name": "Qwen3.5 27B（高质量）",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3.5:27b",
        "api_key": "ollama",
        "enabled": False,
        "supports_json_schema": True,
        "description": "兼容保留：适合已有 Ollama 环境。",
    },
    {
        "id": "ollama-qwen3-8b",
        "name": "Qwen3 8B（兼容备选）",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3:8b",
        "api_key": "ollama",
        "enabled": False,
        "supports_json_schema": True,
        "description": "纯文本工具链成熟，适合作为 Qwen3.5 的兼容备选。",
    },
    {
        "id": "ollama-minicpm41-8b",
        "name": "MiniCPM4.1 8B（端侧备选）",
        "provider": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "minicpm4.1:8b",
        "api_key": "ollama",
        "enabled": False,
        "supports_json_schema": True,
        "description": "端侧效率优先；模型标签可按本机实际安装名称调整。",
    },
]

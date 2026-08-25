from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    database_path: Path
    api_token: str
    max_context_messages: int
    data_dir: Path
    artifacts_dir: Path
    datasets_dir: Path
    web_dir: Path

    @classmethod
    def from_env(cls) -> "AppConfig":
        database_value = os.getenv("SEMANTIC_MONITOR_DB", "").strip()
        database_path = Path(database_value) if database_value else PROJECT_DIR / "data" / "semantic_monitor.sqlite3"
        data_dir = PROJECT_DIR / "data"
        return cls(
            host=os.getenv("SEMANTIC_MONITOR_HOST", "127.0.0.1"),
            port=int(os.getenv("SEMANTIC_MONITOR_PORT", "28110")),
            database_path=database_path.resolve(),
            api_token=os.getenv("SEMANTIC_MONITOR_API_TOKEN", "").strip(),
            max_context_messages=max(1, min(50, int(os.getenv("SEMANTIC_MONITOR_CONTEXT_SIZE", "8")))),
            data_dir=data_dir,
            artifacts_dir=data_dir / "artifacts",
            datasets_dir=data_dir / "datasets",
            web_dir=PROJECT_DIR / "static",
        )

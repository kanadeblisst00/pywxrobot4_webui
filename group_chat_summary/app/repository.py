from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import BUILTIN_MODEL_PROFILES, DEFAULT_SETTINGS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Repository:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_profiles (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    base_url TEXT NOT NULL,
                    model TEXT NOT NULL,
                    api_key TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    is_builtin INTEGER NOT NULL DEFAULT 0,
                    supports_json_schema INTEGER NOT NULL DEFAULT 1,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    id TEXT PRIMARY KEY,
                    room_id TEXT NOT NULL DEFAULT '',
                    room_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    model_profile_id TEXT NOT NULL,
                    request_json TEXT,
                    result_json TEXT,
                    error TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_summaries_created_at ON summaries(created_at DESC);
                """
            )
            now = utc_now()
            for profile in BUILTIN_MODEL_PROFILES:
                connection.execute(
                    """
                    INSERT INTO model_profiles (
                        id, name, provider, base_url, model, api_key, enabled,
                        is_builtin, supports_json_schema, description, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        name=excluded.name,
                        provider=excluded.provider,
                        description=excluded.description,
                        is_builtin=1
                    """,
                    (
                        profile["id"], profile["name"], profile["provider"], profile["base_url"],
                        profile["model"], profile["api_key"], int(profile["enabled"]),
                        int(profile["supports_json_schema"]), profile["description"], now, now,
                    ),
                )
            for key, value in DEFAULT_SETTINGS.items():
                connection.execute(
                    "INSERT OR IGNORE INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)",
                    (key, json.dumps(value, ensure_ascii=False), now),
                )
            for old_port, new_port in ((8080, 18080), (8081, 18081), (8082, 18082)):
                connection.execute(
                    """
                    UPDATE model_profiles SET base_url=?, updated_at=?
                    WHERE provider='llama.cpp' AND is_builtin=1 AND base_url=?
                    """,
                    (f"http://127.0.0.1:{new_port}/v1", now, f"http://127.0.0.1:{old_port}/v1"),
                )
            old_default = connection.execute(
                "SELECT value_json FROM settings WHERE key='default_model_profile_id'"
            ).fetchone()
            if old_default and old_default["value_json"] == json.dumps("ollama-qwen35-9b"):
                connection.execute(
                    "UPDATE settings SET value_json=?, updated_at=? WHERE key='default_model_profile_id'",
                    (json.dumps("llamacpp-qwen35-9b"), now),
                )
                connection.execute(
                    "UPDATE model_profiles SET enabled=0, updated_at=? WHERE provider='ollama' AND is_builtin=1",
                    (now,),
                )

    @staticmethod
    def _profile_dict(row: sqlite3.Row, reveal_secret: bool = False) -> dict[str, Any]:
        item = dict(row)
        raw_api_key = str(item.get("api_key") or "")
        item["api_key_configured"] = bool(raw_api_key)
        item["api_key"] = raw_api_key if reveal_secret else ""
        item["enabled"] = bool(item["enabled"])
        item["is_builtin"] = bool(item["is_builtin"])
        item["supports_json_schema"] = bool(item["supports_json_schema"])
        return item

    def list_model_profiles(self, reveal_secret: bool = False) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM model_profiles ORDER BY enabled DESC, is_builtin DESC, name"
            ).fetchall()
        return [self._profile_dict(row, reveal_secret) for row in rows]

    def get_model_profile(self, profile_id: str, reveal_secret: bool = True) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM model_profiles WHERE id=?", (profile_id,)).fetchone()
        return self._profile_dict(row, reveal_secret) if row else None

    def create_model_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        profile_id = f"custom-{uuid.uuid4().hex[:12]}"
        now = utc_now()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO model_profiles (
                    id, name, provider, base_url, model, api_key, enabled,
                    is_builtin, supports_json_schema, description, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (
                    profile_id, payload["name"], payload["provider"], payload["base_url"].rstrip("/"),
                    payload["model"], payload.get("api_key", ""), int(payload.get("enabled", True)),
                    int(payload.get("supports_json_schema", True)), payload.get("description", ""), now, now,
                ),
            )
        return self.get_model_profile(profile_id, reveal_secret=False) or {}

    def update_model_profile(self, profile_id: str, changes: dict[str, Any]) -> dict[str, Any] | None:
        existing = self.get_model_profile(profile_id, reveal_secret=True)
        if not existing:
            return None
        allowed = {
            "name", "provider", "base_url", "model", "api_key", "enabled",
            "supports_json_schema", "description",
        }
        updates = {key: value for key, value in changes.items() if key in allowed and value is not None}
        if not updates:
            return self.get_model_profile(profile_id, reveal_secret=False)
        if "base_url" in updates:
            updates["base_url"] = str(updates["base_url"]).rstrip("/")
        for key in ("enabled", "supports_json_schema"):
            if key in updates:
                updates[key] = int(bool(updates[key]))
        updates["updated_at"] = utc_now()
        assignments = ", ".join(f"{key}=?" for key in updates)
        values = [*updates.values(), profile_id]
        with self._lock, self._connect() as connection:
            connection.execute(f"UPDATE model_profiles SET {assignments} WHERE id=?", values)
        return self.get_model_profile(profile_id, reveal_secret=False)

    def delete_model_profile(self, profile_id: str) -> bool:
        existing = self.get_model_profile(profile_id)
        if not existing or existing["is_builtin"]:
            return False
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM model_profiles WHERE id=?", (profile_id,))
        return cursor.rowcount > 0

    def get_settings(self) -> dict[str, Any]:
        result = dict(DEFAULT_SETTINGS)
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value_json FROM settings").fetchall()
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value_json"])
            except json.JSONDecodeError:
                continue
        return result

    def update_settings(self, values: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        with self._lock, self._connect() as connection:
            for key, value in values.items():
                connection.execute(
                    """
                    INSERT INTO settings(key, value_json, updated_at) VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at
                    """,
                    (key, json.dumps(value, ensure_ascii=False), now),
                )
        return self.get_settings()

    def create_summary(self, payload: dict[str, Any], model_profile_id: str, keep_raw: bool) -> dict[str, Any]:
        summary_id = uuid.uuid4().hex
        now = utc_now()
        request_json = json.dumps(payload, ensure_ascii=False) if keep_raw else None
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO summaries (
                    id, room_id, room_name, status, model_profile_id,
                    request_json, result_json, error, created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, ?, NULL, '', ?, ?)
                """,
                (summary_id, payload.get("room_id", ""), payload.get("room_name", "未命名群聊"), model_profile_id, request_json, now, now),
            )
        return self.get_summary(summary_id) or {}

    def update_summary(self, summary_id: str, *, status: str, result: dict[str, Any] | None = None, error: str = "") -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "UPDATE summaries SET status=?, result_json=?, error=?, updated_at=? WHERE id=?",
                (status, json.dumps(result, ensure_ascii=False) if result is not None else None, error, utc_now(), summary_id),
            )

    @staticmethod
    def _summary_dict(row: sqlite3.Row, include_request: bool = False) -> dict[str, Any]:
        item = dict(row)
        raw_result = item.pop("result_json", None)
        raw_request = item.pop("request_json", None)
        item["result"] = json.loads(raw_result) if raw_result else None
        if include_request:
            item["request"] = json.loads(raw_request) if raw_request else None
        return item

    def get_summary(self, summary_id: str, include_request: bool = False) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM summaries WHERE id=?", (summary_id,)).fetchone()
        return self._summary_dict(row, include_request) if row else None

    def list_summaries(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM summaries ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 200)),)
            ).fetchall()
        return [self._summary_dict(row) for row in rows]

    def delete_summary(self, summary_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM summaries WHERE id=?", (summary_id,))
        return cursor.rowcount > 0

    def dashboard_stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) AS completed,
                    SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS failed,
                    SUM(CASE WHEN status IN ('pending','running') THEN 1 ELSE 0 END) AS running
                FROM summaries
                """
            ).fetchone()
            enabled_models = connection.execute("SELECT COUNT(*) FROM model_profiles WHERE enabled=1").fetchone()[0]
        stats = dict(row)
        stats["enabled_models"] = enabled_models
        return {key: int(value or 0) for key, value in stats.items()}

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .db import Database
from .schemas import (
    DatasetRecord,
    Event,
    FeedbackInput,
    ModelSettings,
    SampleRecord,
    Topic,
    TopicInput,
    TrainingRequest,
    TrainingRun,
)


SECRET_PLACEHOLDER = "••••••••"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _parse(value: str, fallback: Any) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return fallback


class Repository:
    def __init__(self, db: Database):
        self.db = db

    @staticmethod
    def _topic_from_row(row: Any) -> Topic:
        return Topic(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            enabled=bool(row["enabled"]),
            severity=row["severity"],
            keywords=_parse(row["keywords_json"], []),
            regex_patterns=_parse(row["regex_patterns_json"], []),
            exclude_patterns=_parse(row["exclude_patterns_json"], []),
            examples=_parse(row["examples_json"], []),
            semantic_threshold=row["semantic_threshold"],
            review_threshold=row["review_threshold"],
            context_enabled=bool(row["context_enabled"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def list_topics(self, *, enabled_only: bool = False) -> list[Topic]:
        sql = "SELECT * FROM topics"
        params: tuple[Any, ...] = ()
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY id"
        rows = self.db.connect().execute(sql, params).fetchall()
        return [self._topic_from_row(row) for row in rows]

    def get_topic(self, topic_id: int) -> Topic | None:
        row = self.db.connect().execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
        return self._topic_from_row(row) if row else None

    def create_topic(self, value: TopicInput) -> Topic:
        data = value.model_dump(mode="json")
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO topics(
                    name, description, enabled, severity, keywords_json, regex_patterns_json,
                    exclude_patterns_json, examples_json, semantic_threshold, review_threshold, context_enabled
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["name"], data["description"], int(data["enabled"]), data["severity"],
                    _json(data["keywords"]), _json(data["regex_patterns"]), _json(data["exclude_patterns"]),
                    _json(data["examples"]), data["semantic_threshold"], data["review_threshold"],
                    int(data["context_enabled"]),
                ),
            )
            topic_id = int(cursor.lastrowid)
        return self.get_topic(topic_id)  # type: ignore[return-value]

    def update_topic(self, topic_id: int, value: TopicInput) -> Topic | None:
        data = value.model_dump(mode="json")
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE topics SET
                    name=?, description=?, enabled=?, severity=?, keywords_json=?, regex_patterns_json=?,
                    exclude_patterns_json=?, examples_json=?, semantic_threshold=?, review_threshold=?,
                    context_enabled=?, updated_at=CURRENT_TIMESTAMP
                WHERE id=?
                """,
                (
                    data["name"], data["description"], int(data["enabled"]), data["severity"],
                    _json(data["keywords"]), _json(data["regex_patterns"]), _json(data["exclude_patterns"]),
                    _json(data["examples"]), data["semantic_threshold"], data["review_threshold"],
                    int(data["context_enabled"]), topic_id,
                ),
            )
        return self.get_topic(topic_id) if cursor.rowcount else None

    def delete_topic(self, topic_id: int) -> bool:
        with self.db.transaction() as connection:
            cursor = connection.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
        return bool(cursor.rowcount)

    def get_model_settings(self, *, mask_secrets: bool = False) -> ModelSettings:
        row = self.db.connect().execute("SELECT value_json FROM settings WHERE key='models'").fetchone()
        value = _parse(row["value_json"], {}) if row else {}
        settings = ModelSettings.model_validate(value)
        if not mask_secrets:
            return settings
        payload = settings.model_dump()
        for key in ("embedding_api_key", "classifier_api_key"):
            payload[key] = SECRET_PLACEHOLDER if payload[key] else ""
        return ModelSettings.model_validate(payload)

    def save_model_settings(self, value: ModelSettings) -> ModelSettings:
        existing = self.get_model_settings()
        payload = value.model_dump()
        for key in ("embedding_api_key", "classifier_api_key"):
            if payload[key] == SECRET_PLACEHOLDER:
                payload[key] = getattr(existing, key)
        settings = ModelSettings.model_validate(payload)
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO settings(key, value_json, updated_at) VALUES('models', ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_at=CURRENT_TIMESTAMP
                """,
                (_json(settings.model_dump()),),
            )
        return self.get_model_settings(mask_secrets=True)

    def add_context(self, room_id: str, sender_name: str, text: str, limit: int) -> None:
        with self.db.transaction() as connection:
            connection.execute(
                "INSERT INTO room_context(room_id, sender_name, text) VALUES(?, ?, ?)",
                (room_id, sender_name, text),
            )
            connection.execute(
                """
                DELETE FROM room_context WHERE room_id=? AND id NOT IN (
                    SELECT id FROM room_context WHERE room_id=? ORDER BY id DESC LIMIT ?
                )
                """,
                (room_id, room_id, max(limit, 1)),
            )

    def get_context(self, room_id: str, limit: int) -> list[str]:
        rows = self.db.connect().execute(
            "SELECT sender_name, text FROM room_context WHERE room_id=? ORDER BY id DESC LIMIT ?",
            (room_id, max(limit, 1)),
        ).fetchall()
        return [f"{row['sender_name'] or '成员'}：{row['text']}" for row in reversed(rows)]

    def add_event(self, payload: dict[str, Any]) -> int:
        with self.db.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT INTO events(
                    message_id, room_id, sender_id, sender_name, text, normalized_text,
                    topic_id, topic_name, severity, confidence, semantic_score, rule_score,
                    classifier_score, stage, evidence, needs_review
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["message_id"], payload["room_id"], payload["sender_id"], payload["sender_name"],
                    payload["text"], payload["normalized_text"], payload["topic_id"], payload["topic_name"],
                    payload["severity"], payload["confidence"], payload.get("semantic_score"),
                    payload["rule_score"], payload.get("classifier_score"), payload["stage"],
                    payload["evidence"], int(payload["needs_review"]),
                ),
            )
        return int(cursor.lastrowid)

    @staticmethod
    def _event_from_row(row: Any) -> Event:
        payload = dict(row)
        payload["needs_review"] = bool(row["needs_review"])
        payload["created_at"] = datetime.fromisoformat(row["created_at"])
        return Event.model_validate(payload)

    def list_events(
        self,
        *,
        page: int = 1,
        page_size: int = 30,
        topic_id: int | None = None,
        feedback: str | None = None,
        room_id: str | None = None,
    ) -> tuple[list[Event], int]:
        clauses: list[str] = []
        params: list[Any] = []
        if topic_id is not None:
            clauses.append("topic_id=?")
            params.append(topic_id)
        if feedback:
            clauses.append("feedback=?")
            params.append(feedback)
        if room_id:
            clauses.append("room_id=?")
            params.append(room_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        total = int(self.db.connect().execute(f"SELECT COUNT(*) FROM events{where}", params).fetchone()[0])
        rows = self.db.connect().execute(
            f"SELECT * FROM events{where} ORDER BY id DESC LIMIT ? OFFSET ?",
            [*params, page_size, (page - 1) * page_size],
        ).fetchall()
        return [self._event_from_row(row) for row in rows], total

    def save_feedback(self, event_id: int, value: FeedbackInput) -> Event | None:
        with self.db.transaction() as connection:
            cursor = connection.execute(
                "UPDATE events SET feedback=?, feedback_note=? WHERE id=?",
                (value.verdict, value.note, event_id),
            )
        if not cursor.rowcount:
            return None
        row = self.db.connect().execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return self._event_from_row(row)

    def stats(self) -> dict[str, Any]:
        connection = self.db.connect()
        total = int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
        today = int(connection.execute("SELECT COUNT(*) FROM events WHERE date(created_at)=date('now','localtime')").fetchone()[0])
        pending = int(connection.execute("SELECT COUNT(*) FROM events WHERE needs_review=1 AND feedback IS NULL").fetchone()[0])
        false_positive = int(connection.execute("SELECT COUNT(*) FROM events WHERE feedback='false_positive'").fetchone()[0])
        feedback_total = int(connection.execute("SELECT COUNT(*) FROM events WHERE feedback IS NOT NULL").fetchone()[0])
        enabled_topics = int(connection.execute("SELECT COUNT(*) FROM topics WHERE enabled=1").fetchone()[0])
        rows = connection.execute(
            "SELECT topic_name, COUNT(*) count FROM events GROUP BY topic_name ORDER BY count DESC LIMIT 6"
        ).fetchall()
        return {
            "events_total": total,
            "events_today": today,
            "pending_review": pending,
            "false_positive_rate": round(false_positive / feedback_total, 4) if feedback_total else 0,
            "enabled_topics": enabled_topics,
            "top_topics": [dict(row) for row in rows],
            "datasets": int(connection.execute("SELECT COUNT(*) FROM datasets").fetchone()[0]),
            "samples": int(connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]),
            "trained_models": int(connection.execute("SELECT COUNT(*) FROM training_runs WHERE status='completed'").fetchone()[0]),
        }

    @staticmethod
    def _dataset_from_row(row: Any) -> DatasetRecord:
        return DatasetRecord(
            **{key: row[key] for key in ("id", "slug", "name", "version", "license", "source_url", "status", "sample_count")},
            labels=_parse(row["labels_json"], []),
            metadata=_parse(row["metadata_json"], {}),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def list_datasets(self) -> list[DatasetRecord]:
        rows = self.db.connect().execute("SELECT * FROM datasets ORDER BY id DESC").fetchall()
        return [self._dataset_from_row(row) for row in rows]

    def get_dataset(self, dataset_id: int) -> DatasetRecord | None:
        row = self.db.connect().execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
        return self._dataset_from_row(row) if row else None

    def upsert_dataset(
        self,
        *,
        slug: str,
        name: str,
        version: str,
        license_name: str,
        source_url: str,
        status: str = "ready",
        labels: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DatasetRecord:
        with self.db.transaction() as connection:
            connection.execute(
                """
                INSERT INTO datasets(slug,name,version,license,source_url,status,labels_json,metadata_json)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(slug,version) DO UPDATE SET
                    name=excluded.name, license=excluded.license, source_url=excluded.source_url,
                    status=excluded.status, labels_json=excluded.labels_json,
                    metadata_json=excluded.metadata_json, updated_at=CURRENT_TIMESTAMP
                """,
                (slug, name, version, license_name, source_url, status, _json(labels or []), _json(metadata or {})),
            )
        row = self.db.connect().execute(
            "SELECT * FROM datasets WHERE slug=? AND version=?", (slug, version)
        ).fetchone()
        return self._dataset_from_row(row)

    def add_samples(self, dataset_id: int, samples: list[dict[str, Any]]) -> tuple[int, int]:
        inserted = 0
        skipped = 0
        with self.db.transaction() as connection:
            for sample in samples:
                cursor = connection.execute(
                    """
                    INSERT OR IGNORE INTO samples(
                        dataset_id,text,normalized_text,labels_json,split,source_ref,metadata_json,fingerprint
                    ) VALUES(?,?,?,?,?,?,?,?)
                    """,
                    (
                        dataset_id,
                        sample["text"],
                        sample["normalized_text"],
                        _json(sample["labels"]),
                        sample.get("split", "train"),
                        sample.get("source_ref", ""),
                        _json(sample.get("metadata", {})),
                        sample["fingerprint"],
                    ),
                )
                if cursor.rowcount:
                    inserted += 1
                else:
                    skipped += 1
            row = connection.execute(
                "SELECT COUNT(DISTINCT samples.id), COALESCE(json_group_array(DISTINCT value),'[]') FROM samples, json_each(samples.labels_json) WHERE dataset_id=?",
                (dataset_id,),
            ).fetchone()
            connection.execute(
                "UPDATE datasets SET sample_count=?, labels_json=?, status='ready', updated_at=CURRENT_TIMESTAMP WHERE id=?",
                (int(row[0]), row[1], dataset_id),
            )
        return inserted, skipped

    def delete_dataset(self, dataset_id: int) -> bool:
        with self.db.transaction() as connection:
            cursor = connection.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
        return bool(cursor.rowcount)

    def list_samples(
        self, dataset_ids: list[int] | None = None, *, split: str | None = None, limit: int = 200, offset: int = 0
    ) -> tuple[list[SampleRecord], int]:
        clauses: list[str] = []
        params: list[Any] = []
        if dataset_ids:
            placeholders = ",".join("?" for _ in dataset_ids)
            clauses.append(f"dataset_id IN ({placeholders})")
            params.extend(dataset_ids)
        if split:
            clauses.append("split=?")
            params.append(split)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        connection = self.db.connect()
        total = int(connection.execute(f"SELECT COUNT(*) FROM samples{where}", params).fetchone()[0])
        rows = connection.execute(
            f"SELECT id,dataset_id,text,labels_json,split,source_ref FROM samples{where} ORDER BY id LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
        items = [
            SampleRecord(
                id=row["id"], dataset_id=row["dataset_id"], text=row["text"],
                labels=_parse(row["labels_json"], []), split=row["split"], source_ref=row["source_ref"]
            )
            for row in rows
        ]
        return items, total

    def all_training_samples(self, dataset_ids: list[int]) -> list[dict[str, Any]]:
        if not dataset_ids:
            return []
        placeholders = ",".join("?" for _ in dataset_ids)
        rows = self.db.connect().execute(
            f"SELECT id,text,normalized_text,labels_json,split,dataset_id FROM samples WHERE dataset_id IN ({placeholders}) ORDER BY id",
            dataset_ids,
        ).fetchall()
        return [
            {**dict(row), "labels": _parse(row["labels_json"], [])}
            for row in rows
        ]

    @staticmethod
    def _training_run_from_row(row: Any) -> TrainingRun:
        return TrainingRun(
            id=row["id"], name=row["name"], algorithm=row["algorithm"], status=row["status"],
            dataset_ids=_parse(row["dataset_ids_json"], []), config=_parse(row["config_json"], {}),
            metrics=_parse(row["metrics_json"], {}), artifact_path=row["artifact_path"],
            sample_count=row["sample_count"], error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=datetime.fromisoformat(row["started_at"]) if row["started_at"] else None,
            completed_at=datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None,
        )

    def create_training_run(self, request: TrainingRequest) -> TrainingRun:
        config = request.model_dump()
        with self.db.transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO training_runs(name,algorithm,dataset_ids_json,config_json) VALUES(?,?,?,?)",
                (request.name, request.algorithm, _json(request.dataset_ids), _json(config)),
            )
        return self.get_training_run(int(cursor.lastrowid))  # type: ignore[return-value]

    def get_training_run(self, run_id: int) -> TrainingRun | None:
        row = self.db.connect().execute("SELECT * FROM training_runs WHERE id=?", (run_id,)).fetchone()
        return self._training_run_from_row(row) if row else None

    def list_training_runs(self) -> list[TrainingRun]:
        rows = self.db.connect().execute("SELECT * FROM training_runs ORDER BY id DESC LIMIT 100").fetchall()
        return [self._training_run_from_row(row) for row in rows]

    def update_training_run(self, run_id: int, **changes: Any) -> None:
        allowed = {"status", "metrics_json", "artifact_path", "sample_count", "error", "started_at", "completed_at"}
        values = {key: value for key, value in changes.items() if key in allowed}
        if not values:
            return
        assignments = ",".join(f"{key}=?" for key in values)
        with self.db.transaction() as connection:
            connection.execute(
                f"UPDATE training_runs SET {assignments} WHERE id=?",
                [*values.values(), run_id],
            )

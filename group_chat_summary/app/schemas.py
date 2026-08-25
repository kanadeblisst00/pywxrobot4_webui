from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1, max_length=200)
    sender_id: str | None = Field(default=None, max_length=200)
    sender_name: str = Field(default="未知成员", max_length=200)
    timestamp: datetime | str | None = None
    content: str = Field(min_length=1, max_length=100_000)
    message_type: str = Field(default="text", max_length=50)

    @field_validator("id", "sender_name", "content", "message_type", mode="before")
    @classmethod
    def strip_required_text(cls, value: Any) -> str:
        return str(value or "").strip()


class Topic(BaseModel):
    title: str
    summary: str
    participants: list[str] = Field(default_factory=list)
    evidence_message_ids: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    content: str
    owner: str | None = None
    evidence_message_ids: list[str] = Field(default_factory=list)


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    deadline: str | None = None
    status: Literal["pending", "in_progress", "done", "unknown"] = "unknown"
    evidence_message_ids: list[str] = Field(default_factory=list)


class OpenQuestion(BaseModel):
    question: str
    owner: str | None = None
    evidence_message_ids: list[str] = Field(default_factory=list)


class Risk(BaseModel):
    content: str
    level: Literal["low", "medium", "high", "unknown"] = "unknown"
    evidence_message_ids: list[str] = Field(default_factory=list)


class ChunkSummary(BaseModel):
    overview: str = ""
    topics: list[Topic] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    open_questions: list[OpenQuestion] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)


class SummaryStats(BaseModel):
    source_message_count: int
    included_message_count: int
    chunk_count: int
    participant_count: int
    started_at: str | None = None
    ended_at: str | None = None


class SummaryResult(ChunkSummary):
    title: str = "群聊摘要"
    stats: SummaryStats


class SummaryCreate(BaseModel):
    room_id: str = Field(default="", max_length=200)
    room_name: str = Field(default="未命名群聊", max_length=200)
    model_profile_id: str | None = None
    messages: list[ChatMessage] = Field(min_length=1, max_length=20_000)
    custom_instruction: str | None = Field(default=None, max_length=4000)


class ModelProfileInput(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    provider: str = Field(default="openai-compatible", max_length=50)
    base_url: str = Field(min_length=1, max_length=500)
    model: str = Field(min_length=1, max_length=200)
    api_key: str = Field(default="", max_length=1000)
    enabled: bool = True
    supports_json_schema: bool = True
    description: str = Field(default="", max_length=500)


class ModelProfile(ModelProfileInput):
    id: str
    is_builtin: bool = False
    created_at: str
    updated_at: str
    api_key_configured: bool = False


class ModelProfileUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    provider: str | None = Field(default=None, max_length=50)
    base_url: str | None = Field(default=None, min_length=1, max_length=500)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    api_key: str | None = Field(default=None, max_length=1000)
    enabled: bool | None = None
    supports_json_schema: bool | None = None
    description: str | None = Field(default=None, max_length=500)


class PipelineSettings(BaseModel):
    default_model_profile_id: str
    chunk_max_chars: int = Field(ge=2000, le=100_000)
    chunk_overlap_messages: int = Field(ge=0, le=20)
    max_output_tokens: int = Field(ge=512, le=16_000)
    temperature: float = Field(ge=0, le=1.5)
    keep_raw_messages: bool = True
    ignored_message_types: list[str] = Field(default_factory=list)
    custom_instruction: str = Field(default="", max_length=4000)


class SummaryRecord(BaseModel):
    id: str
    room_id: str
    room_name: str
    status: Literal["pending", "running", "completed", "failed"]
    model_profile_id: str
    result: dict[str, Any] | None = None
    error: str = ""
    created_at: str
    updated_at: str

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExampleInput(StrictModel):
    text: str = Field(min_length=1, max_length=2000)
    polarity: Literal["positive", "negative"] = "positive"

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()


class TopicInput(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=1000)
    enabled: bool = True
    severity: Literal["low", "medium", "high", "critical"] = "medium"
    keywords: list[str] = Field(default_factory=list, max_length=100)
    regex_patterns: list[str] = Field(default_factory=list, max_length=50)
    exclude_patterns: list[str] = Field(default_factory=list, max_length=50)
    examples: list[ExampleInput] = Field(default_factory=list, max_length=200)
    semantic_threshold: float = Field(default=0.66, ge=-1, le=1)
    review_threshold: float = Field(default=0.46, ge=-1, le=1)
    context_enabled: bool = True

    @field_validator("name", "description")
    @classmethod
    def strip_string(cls, value: str) -> str:
        return value.strip()

    @field_validator("keywords", "regex_patterns", "exclude_patterns")
    @classmethod
    def normalize_list(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(item.strip() for item in value if item.strip()))


class Topic(TopicInput):
    id: int
    created_at: datetime
    updated_at: datetime


class ModelSettings(StrictModel):
    profile: Literal["eco", "balanced", "quality", "custom"] = "eco"
    embedding_provider: Literal["hashing", "sentence_transformers", "openai_compatible"] = "hashing"
    embedding_model: str = "builtin/char-ngram-zh-v1"
    embedding_api_base: str = ""
    embedding_api_key: str = ""
    classifier_provider: Literal["disabled", "heuristic", "builtin_nb", "transformers", "openai_compatible"] = "builtin_nb"
    classifier_model: str = "builtin/char-ngram-nb"
    classifier_api_base: str = ""
    classifier_api_key: str = ""
    request_timeout_seconds: float = Field(default=20, ge=1, le=120)
    classifier_artifact: str = ""
    review_lower_bound: float = Field(default=0.45, ge=0, le=1)
    block_threshold: float = Field(default=0.86, ge=0, le=1)
    max_new_tokens: int = Field(default=48, ge=8, le=256)
    qrcode_enabled: bool = True
    ocr_provider: Literal["disabled", "rapidocr"] = "rapidocr"
    nsfw_provider: Literal["disabled", "nudenet", "opennsfw2"] = "nudenet"
    nsfw_threshold: float = Field(default=0.72, ge=0.1, le=0.99)
    max_image_megapixels: float = Field(default=25, ge=1, le=100)

    @field_validator("embedding_model", "classifier_model")
    @classmethod
    def require_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("模型名称不能为空")
        return value

    @field_validator("classifier_artifact")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return value.strip()

    @field_validator("embedding_api_base", "classifier_api_base")
    @classmethod
    def normalize_url(cls, value: str) -> str:
        return value.strip().rstrip("/")


class MessageInput(StrictModel):
    message_id: str = Field(default="", max_length=200)
    room_id: str = Field(min_length=1, max_length=200)
    sender_id: str = Field(default="", max_length=200)
    sender_name: str = Field(default="", max_length=200)
    text: str = Field(min_length=1, max_length=10000)
    message_type: str = Field(default="text", max_length=50)
    timestamp: datetime | None = None
    context: list[str] = Field(default_factory=list, max_length=50)
    metadata: dict[str, Any] = Field(default_factory=dict)
    persist: bool = True

    @field_validator("message_id", "room_id", "sender_id", "sender_name", "text", "message_type")
    @classmethod
    def strip_fields(cls, value: str) -> str:
        return value.strip()


class MatchResult(BaseModel):
    topic_id: int
    topic_name: str
    matched: bool
    confidence: float
    semantic_score: float | None = None
    rule_score: float = 0
    classifier_score: float | None = None
    stage: Literal["rule", "semantic", "classifier", "none"]
    severity: str
    evidence: str
    needs_review: bool = False
    matched_terms: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    message_id: str
    normalized_text: str
    matched: bool
    risk_level: Literal["none", "low", "medium", "high", "critical"]
    processing_ms: float
    matches: list[MatchResult]
    event_ids: list[int] = Field(default_factory=list)
    model_trace: dict[str, str]


class FeedbackInput(StrictModel):
    verdict: Literal["correct", "false_positive", "missed"]
    note: str = Field(default="", max_length=1000)


class Event(BaseModel):
    id: int
    message_id: str
    room_id: str
    sender_id: str
    sender_name: str
    text: str
    normalized_text: str
    topic_id: int
    topic_name: str
    severity: str
    confidence: float
    semantic_score: float | None
    rule_score: float
    classifier_score: float | None
    stage: str
    evidence: str
    needs_review: bool
    feedback: str | None
    feedback_note: str
    created_at: datetime


class PaginatedEvents(BaseModel):
    items: list[Event]
    total: int
    page: int
    page_size: int


class ModelProbeInput(StrictModel):
    kind: Literal["embedding", "classifier"]


class DatasetImportOptions(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(min_length=1, max_length=120)
    version: str = Field(default="local", max_length=60)
    license: str = Field(default="user-provided", max_length=120)
    text_column: str = Field(default="text", max_length=120)
    label_column: str = Field(default="label", max_length=120)
    split_column: str = Field(default="split", max_length=120)
    default_label: str = Field(default="normal", max_length=80)
    label_mapping: dict[str, list[str]] = Field(default_factory=dict)


class DatasetRecord(BaseModel):
    id: int
    slug: str
    name: str
    version: str
    license: str
    source_url: str
    status: str
    sample_count: int
    labels: list[str]
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class SampleRecord(BaseModel):
    id: int
    dataset_id: int
    text: str
    labels: list[str]
    split: str
    source_ref: str


class TrainingRequest(StrictModel):
    name: str = Field(default="字符 n-gram 审核模型", min_length=1, max_length=120)
    algorithm: Literal["char_ngram_nb"] = "char_ngram_nb"
    dataset_ids: list[int] = Field(min_length=1)
    test_ratio: float = Field(default=0.2, ge=0.1, le=0.5)
    min_ngram: int = Field(default=1, ge=1, le=4)
    max_ngram: int = Field(default=3, ge=1, le=5)
    min_df: int = Field(default=2, ge=1, le=100)
    alpha: float = Field(default=1.0, gt=0, le=20)
    threshold: float = Field(default=0.52, ge=0.1, le=0.95)
    seed: int = 42


class TrainingRun(BaseModel):
    id: int
    name: str
    algorithm: str
    status: str
    dataset_ids: list[int]
    config: dict[str, Any]
    metrics: dict[str, Any]
    artifact_path: str
    sample_count: int
    error: str
    created_at: datetime
    started_at: datetime | None
    completed_at: datetime | None


class BatchTestItem(StrictModel):
    text: str = Field(min_length=1, max_length=10000)
    expected: list[str] = Field(default_factory=list)


class BatchTestInput(StrictModel):
    items: list[BatchTestItem] = Field(min_length=1, max_length=500)


class VisionComponent(BaseModel):
    enabled: bool
    available: bool
    provider: str
    status: Literal["ok", "disabled", "unavailable", "failed"]
    error: str = ""


class QRCodeResult(VisionComponent):
    detected: bool = False
    count: int = 0
    formats: list[str] = Field(default_factory=list)


class OCRLine(BaseModel):
    text: str
    confidence: float


class OCRResult(VisionComponent):
    text: str = ""
    lines: list[OCRLine] = Field(default_factory=list)
    average_confidence: float | None = None


class NSFWFinding(BaseModel):
    label: str
    score: float


class NSFWResult(VisionComponent):
    score: float | None = None
    matched: bool = False
    threshold: float
    findings: list[NSFWFinding] = Field(default_factory=list)


class ImageAnalyzeResponse(BaseModel):
    message_id: str
    filename: str
    media_type: str
    width: int
    height: int
    size_bytes: int
    matched: bool
    risk_level: Literal["none", "low", "medium", "high", "critical"]
    processing_ms: float
    qrcode: QRCodeResult
    ocr: OCRResult
    nsfw: NSFWResult
    text_analysis: AnalyzeResponse | None = None
    event_ids: list[int] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

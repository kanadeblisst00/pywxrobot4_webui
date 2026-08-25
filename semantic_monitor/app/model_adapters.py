from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
import numpy as np

from .schemas import ModelSettings, Topic


def cosine_similarity(left: list[float], right: list[float]) -> float:
    a = np.asarray(left, dtype=np.float32)
    b = np.asarray(right, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


class EmbeddingBackend(ABC):
    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbedding(EmbeddingBackend):
    """用于零依赖启动的字符 n-gram 特征模型，不替代真正的语义模型。"""

    def __init__(self, dimensions: int = 1536):
        self.dimensions = dimensions

    def _one(self, text: str) -> list[float]:
        compact = re.sub(r"\s+", "", text.lower())
        tokens: list[tuple[str, float]] = []
        for size, weight in ((1, 0.45), (2, 1.0), (3, 1.25), (4, 0.75)):
            tokens.extend((compact[index:index + size], weight) for index in range(max(0, len(compact) - size + 1)))
        words = re.findall(r"[a-z0-9]+", text.lower())
        tokens.extend((word, 1.5) for word in words)
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token, weight in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            number = int.from_bytes(digest, "little")
            index = number % self.dimensions
            sign = 1.0 if (number >> 63) == 0 else -1.0
            vector[index] += weight * sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector.tolist()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]


class SentenceTransformersEmbedding(EmbeddingBackend):
    _models: dict[str, Any] = {}
    _load_lock = asyncio.Lock()

    def __init__(self, model_name: str):
        self.model_name = model_name

    async def _model(self) -> Any:
        if self.model_name in self._models:
            return self._models[self.model_name]
        async with self._load_lock:
            if self.model_name in self._models:
                return self._models[self.model_name]
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("本地语义模型需要安装 sentence-transformers，可执行 pip install -r requirements-models.txt") from exc
            model = await asyncio.to_thread(SentenceTransformer, self.model_name, trust_remote_code=True)
            self._models[self.model_name] = model
            return model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        model = await self._model()
        vectors = await asyncio.to_thread(
            model.encode,
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32).tolist()


def _api_url(base: str, suffix: str) -> str:
    normalized = base.rstrip("/")
    if normalized.endswith(suffix):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}{suffix}"
    return f"{normalized}/v1{suffix}"


class OpenAICompatibleEmbedding(EmbeddingBackend):
    def __init__(self, settings: ModelSettings):
        if not settings.embedding_api_base:
            raise RuntimeError("请先配置 Embedding API 地址")
        self.url = _api_url(settings.embedding_api_base, "/embeddings")
        self.model = settings.embedding_model
        self.api_key = settings.embedding_api_key
        self.timeout = settings.request_timeout_seconds

    async def embed(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json={"model": self.model, "input": texts})
            response.raise_for_status()
            payload = response.json()
        data = sorted(payload.get("data", []), key=lambda item: item.get("index", 0))
        if len(data) != len(texts):
            raise RuntimeError("Embedding 接口返回的向量数量与输入不一致")
        return [item["embedding"] for item in data]


class ClassifierBackend(ABC):
    @abstractmethod
    async def classify(self, text: str, context: list[str], topic: Topic, base_score: float) -> tuple[float, str]:
        raise NotImplementedError


class DisabledClassifier(ClassifierBackend):
    async def classify(self, text: str, context: list[str], topic: Topic, base_score: float) -> tuple[float, str]:
        return base_score, "未启用上下文复核"


class HeuristicClassifier(ClassifierBackend):
    NEGATION_PATTERNS = (
        r"(不是|并非|没有|别|不要|禁止|谨防|小心|防止|避免).{0,10}(借|贷款|投资|转账|加群|联系)",
        r"(借|贷款|投资|转账|加群|联系).{0,10}(是假的|是诈骗|不可信|已拒绝|不要|别)",
        r"(提醒|曝光|新闻|案例|辟谣|反诈).{0,12}(借钱|贷款|投资|转账|诈骗)",
    )

    async def classify(self, text: str, context: list[str], topic: Topic, base_score: float) -> tuple[float, str]:
        combined = "\n".join([*context[-4:], text])
        for pattern in self.NEGATION_PATTERNS:
            if re.search(pattern, combined, flags=re.IGNORECASE):
                return min(0.28, base_score * 0.35), "检测到否定、劝阻或风险提醒语境"
        negative_examples = [item.text for item in topic.examples if item.polarity == "negative"]
        if any(example and example in combined for example in negative_examples):
            return min(0.3, base_score * 0.4), "与主题反例直接重合"
        context_bonus = 0.05 if context and any(word in combined for word in topic.keywords) else 0
        score = min(0.97, max(base_score, base_score * 0.9 + 0.08 + context_bonus))
        return score, "规则和语义信号在上下文中未发现明显冲突"


class OpenAICompatibleClassifier(ClassifierBackend):
    def __init__(self, settings: ModelSettings):
        if not settings.classifier_api_base:
            raise RuntimeError("请先配置分类模型 API 地址")
        self.url = _api_url(settings.classifier_api_base, "/chat/completions")
        self.model = settings.classifier_model
        self.api_key = settings.classifier_api_key
        self.timeout = settings.request_timeout_seconds

    async def classify(self, text: str, context: list[str], topic: Topic, base_score: float) -> tuple[float, str]:
        positives = [item.text for item in topic.examples if item.polarity == "positive"][:12]
        negatives = [item.text for item in topic.examples if item.polarity == "negative"][:12]
        prompt = {
            "监测主题": topic.name,
            "定义": topic.description,
            "正例": positives,
            "反例": negatives,
            "前文": context[-8:],
            "当前消息": text,
            "召回分": round(base_score, 4),
        }
        system = (
            "你是中文群聊内容风控分类器。只判断当前消息在给定上下文中是否真正符合监测主题；"
            "正确处理否定、反讽、引用、劝阻、新闻讨论和指代。只输出 JSON："
            '{"matched":true,"confidence":0.0,"reason":"简短依据"}。confidence 范围 0 到 1。'
        )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(self.url, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
        content = payload["choices"][0]["message"]["content"]
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            raise RuntimeError("分类模型未返回有效 JSON")
        result = json.loads(match.group(0))
        confidence = float(result.get("confidence", 0.5))
        matched = bool(result.get("matched"))
        score = confidence if matched else 1 - confidence
        if math.isnan(score):
            score = 0
        return min(1.0, max(0.0, score)), str(result.get("reason") or "模型已完成上下文复核")[:300]


def create_embedding_backend(settings: ModelSettings) -> EmbeddingBackend:
    if settings.embedding_provider == "hashing":
        return HashingEmbedding()
    if settings.embedding_provider == "sentence_transformers":
        return SentenceTransformersEmbedding(settings.embedding_model)
    return OpenAICompatibleEmbedding(settings)


def create_classifier_backend(settings: ModelSettings) -> ClassifierBackend:
    if settings.classifier_provider == "disabled":
        return DisabledClassifier()
    if settings.classifier_provider == "heuristic":
        return HeuristicClassifier()
    return OpenAICompatibleClassifier(settings)

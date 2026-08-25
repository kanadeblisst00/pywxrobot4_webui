from __future__ import annotations

import re
from time import perf_counter
from typing import Any

from .classifier import ClassifierRuntime
from .config import AppConfig
from .model_adapters import EmbeddingBackend, cosine_similarity, create_embedding_backend
from .normalizer import compact_for_rule, normalize_text
from .repository import Repository
from .schemas import AnalyzeResponse, MatchResult, MessageInput, ModelSettings, Topic


CATEGORY_BY_NAME = {
    "色情": "porn", "淫秽色情": "porn", "porn": "porn",
    "赌博": "gambling", "涉赌": "gambling", "gambling": "gambling",
    "广告": "advertising", "垃圾广告": "advertising", "advertising": "advertising",
}
SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
NEGATION_PATTERNS = (
    r"(不是|并非|没有|别|不要|禁止|谨防|小心|防止|避免).{0,12}(借|贷款|投资|转账|加群|联系|赌博|下注)",
    r"(借|贷款|投资|转账|加群|联系|赌博|下注).{0,12}(是假的|是诈骗|不可信|已拒绝|不要|别)",
    r"(提醒|曝光|新闻|案例|辟谣|反诈).{0,16}(借钱|贷款|投资|转账|诈骗|赌博)",
)


def category_for_topic(topic: Topic) -> str:
    name = topic.name.strip().lower()
    return CATEGORY_BY_NAME.get(name, name)


def rule_score(text: str, topic: Topic) -> tuple[float, list[str], str, bool]:
    compact = compact_for_rule(text)
    for pattern in topic.exclude_patterns:
        try:
            if re.search(pattern, text, flags=re.I):
                return 0.0, [], f"命中例外规则：{pattern}", True
        except re.error:
            continue
    matched_terms = [keyword for keyword in topic.keywords if compact_for_rule(keyword) in compact]
    score = 0.0
    evidence: list[str] = []
    if matched_terms:
        score = min(0.96, 0.76 + 0.07 * len(matched_terms))
        evidence.append("关键词：" + "、".join(matched_terms[:6]))
    for pattern in topic.regex_patterns:
        try:
            if re.search(pattern, text, flags=re.I):
                score = max(score, 0.96)
                evidence.append(f"正则：{pattern}")
        except re.error:
            continue
    return score, matched_terms, "；".join(evidence), False


def context_adjustment(text: str, context: list[str], topic: Topic, base_score: float) -> tuple[float, str]:
    combined = "\n".join([*context[-6:], text])
    if any(re.search(pattern, combined, flags=re.I) for pattern in NEGATION_PATTERNS):
        return min(0.30, base_score * 0.38), "上下文存在否定、劝阻或风险提醒"
    negative_examples = [item.text for item in topic.examples if item.polarity == "negative"]
    if any(example and example in combined for example in negative_examples):
        return min(0.32, base_score * 0.42), "上下文与主题反例重合"
    return min(0.98, base_score + (0.04 if context else 0.02)), "上下文未发现明显冲突"


class MonitorEngine:
    def __init__(self, repository: Repository, config: AppConfig):
        self.repository = repository
        self.config = config
        self.classifier = ClassifierRuntime()
        self._embedding: EmbeddingBackend | None = None
        self._embedding_key = ""
        self._embedding_cache: dict[tuple[str, str], list[float]] = {}

    def invalidate_models(self) -> None:
        self._embedding = None
        self._embedding_key = ""
        self._embedding_cache.clear()
        self.classifier = ClassifierRuntime()

    def _embedding_backend(self, settings: ModelSettings) -> tuple[EmbeddingBackend, str]:
        key = f"{settings.embedding_provider}:{settings.embedding_model}:{settings.embedding_api_base}"
        if self._embedding is None or key != self._embedding_key:
            self._embedding = create_embedding_backend(settings)
            self._embedding_key = key
            self._embedding_cache.clear()
        return self._embedding, key

    async def _semantic_score(
        self, backend: EmbeddingBackend, key: str, text: str, topic: Topic
    ) -> tuple[float | None, str]:
        positive = [item.text for item in topic.examples if item.polarity == "positive"]
        negative = [item.text for item in topic.examples if item.polarity == "negative"]
        if not positive:
            return None, ""
        query = (await backend.embed([text]))[0]
        examples = [*positive, *negative]
        missing = [item for item in examples if (key, item) not in self._embedding_cache]
        if missing:
            vectors = await backend.embed(missing)
            self._embedding_cache.update({(key, item): vector for item, vector in zip(missing, vectors)})
        positive_score = max(cosine_similarity(query, self._embedding_cache[(key, item)]) for item in positive)
        negative_score = max(
            (cosine_similarity(query, self._embedding_cache[(key, item)]) for item in negative),
            default=-1.0,
        )
        adjusted = positive_score
        if negative_score >= positive_score - 0.04:
            adjusted -= max(0.0, negative_score) * 0.45
        return max(-1.0, min(1.0, adjusted)), f"正例相似 {positive_score:.2f}，反例相似 {negative_score:.2f}"

    async def analyze(self, message: MessageInput) -> AnalyzeResponse:
        started = perf_counter()
        normalized = normalize_text(message.text)
        topics = self.repository.list_topics(enabled_only=True)
        settings = self.repository.get_model_settings()
        embedding, embedding_trace = self._embedding_backend(settings)
        stored_context = self.repository.get_context(message.room_id, self.config.max_context_messages)
        context = [*stored_context, *[normalize_text(item) for item in message.context]][-self.config.max_context_messages:]
        guard_scores, classifier_trace = await self.classifier.classify(normalized, settings)

        matches: list[MatchResult] = []
        event_ids: list[int] = []
        highest_risk = "none"
        for topic in topics:
            current_rule, terms, rule_evidence, excluded = rule_score(normalized, topic)
            if excluded:
                continue
            semantic, semantic_evidence = await self._semantic_score(embedding, embedding_trace, normalized, topic)
            candidates = [current_rule, semantic if semantic is not None else -1.0]
            guard_score = guard_scores.get(category_for_topic(topic))
            if guard_score is not None:
                candidates.append(guard_score)
            base_score = max(candidates)
            if base_score < topic.review_threshold and current_rule <= 0:
                continue

            contextual_score: float | None = None
            context_evidence = ""
            if topic.context_enabled:
                contextual_score, context_evidence = context_adjustment(normalized, context, topic, base_score)
                confidence = base_score * 0.62 + contextual_score * 0.38
            else:
                confidence = base_score
            confidence = max(0.0, min(1.0, confidence))
            is_match = confidence >= topic.semantic_threshold
            in_review_band = confidence >= topic.review_threshold
            if not (is_match or in_review_band):
                continue

            if guard_score is not None and guard_score >= max(current_rule, semantic or -1):
                stage = "classifier"
            elif current_rule >= (semantic or -1):
                stage = "rule"
            else:
                stage = "semantic"
            needs_review = bool(
                (not is_match and in_review_band)
                or topic.severity in {"high", "critical"}
                or confidence < settings.block_threshold
            )
            evidence = [item for item in (rule_evidence, semantic_evidence, context_evidence) if item]
            if guard_score is not None:
                evidence.append(f"安全分类模型 {guard_score:.2f}")
            classifier_score = guard_score if guard_score is not None else contextual_score
            result = MatchResult(
                topic_id=topic.id,
                topic_name=topic.name,
                matched=is_match,
                confidence=round(confidence, 4),
                semantic_score=round(semantic, 4) if semantic is not None else None,
                rule_score=round(current_rule, 4),
                classifier_score=round(classifier_score, 4) if classifier_score is not None else None,
                stage=stage,
                severity=topic.severity,
                evidence="；".join(evidence) or "进入人工复核区间",
                needs_review=needs_review,
                matched_terms=terms,
            )
            matches.append(result)
            if is_match and SEVERITY_RANK.get(topic.severity, 0) > SEVERITY_RANK.get(highest_risk, 0):
                highest_risk = topic.severity
            if message.persist:
                event_ids.append(self.repository.add_event({
                    "message_id": message.message_id,
                    "room_id": message.room_id,
                    "sender_id": message.sender_id,
                    "sender_name": message.sender_name,
                    "text": message.text,
                    "normalized_text": normalized,
                    "topic_id": topic.id,
                    "topic_name": topic.name,
                    "severity": topic.severity,
                    "confidence": confidence,
                    "semantic_score": semantic,
                    "rule_score": current_rule,
                    "classifier_score": result.classifier_score,
                    "stage": stage,
                    "evidence": result.evidence,
                    "needs_review": needs_review,
                }))

        if message.persist:
            self.repository.add_context(message.room_id, message.sender_name, normalized, self.config.max_context_messages)
        matches.sort(key=lambda item: (SEVERITY_RANK.get(item.severity, 0), item.confidence), reverse=True)
        return AnalyzeResponse(
            message_id=message.message_id,
            normalized_text=normalized,
            matched=any(item.matched for item in matches),
            risk_level=highest_risk,
            processing_ms=round((perf_counter() - started) * 1000, 2),
            matches=matches,
            event_ids=event_ids,
            model_trace={
                "embedding": embedding_trace,
                "classifier": classifier_trace,
                "profile": settings.profile,
            },
        )

    async def probe(self, kind: str) -> dict[str, Any]:
        settings = self.repository.get_model_settings()
        started = perf_counter()
        if kind == "embedding":
            backend, _key = self._embedding_backend(settings)
            vectors = await backend.embed(["中文语义模型连通性测试"])
            detail = f"已返回 {len(vectors[0])} 维向量"
        else:
            _scores, trace = await self.classifier.classify("群消息分类模型连通性测试", settings)
            detail = f"判定器可用：{trace}"
        return {"ok": True, "kind": kind, "detail": detail, "latency_ms": round((perf_counter() - started) * 1000, 2)}

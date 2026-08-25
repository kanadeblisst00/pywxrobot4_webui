from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .model_client import OpenAICompatibleClient
from .prompts import SYSTEM_PROMPT, build_chunk_prompt, build_merge_prompt
from .schemas import ChatMessage, ChunkSummary, SummaryResult, SummaryStats


def _timestamp_text(value: datetime | str | None) -> str:
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    return str(value or "").strip()


def render_message(message: ChatMessage) -> str:
    timestamp = _timestamp_text(message.timestamp) or "时间未知"
    return f"[{message.id}] [{timestamp}] {message.sender_name}: {message.content}"


def _expand_oversized_messages(messages: list[ChatMessage], max_chars: int) -> list[ChatMessage]:
    expanded: list[ChatMessage] = []
    content_limit = max(500, max_chars - 500)
    for message in messages:
        if len(render_message(message)) <= max_chars:
            expanded.append(message)
            continue
        parts = [message.content[index : index + content_limit] for index in range(0, len(message.content), content_limit)]
        for index, part in enumerate(parts, start=1):
            expanded.append(message.model_copy(update={"content": f"（第 {index}/{len(parts)} 段）{part}"}))
    return expanded


def chunk_messages(messages: list[ChatMessage], max_chars: int, overlap_messages: int) -> list[list[ChatMessage]]:
    if not messages:
        return []
    messages = _expand_oversized_messages(messages, max_chars)
    chunks: list[list[ChatMessage]] = []
    current: list[ChatMessage] = []
    current_size = 0
    for message in messages:
        message_size = len(render_message(message)) + 1
        if current and current_size + message_size > max_chars:
            chunks.append(current)
            carry = current[-overlap_messages:] if overlap_messages else []
            carry_size = sum(len(render_message(item)) + 1 for item in carry)
            if carry and carry_size + message_size <= max_chars:
                current = list(carry)
                current_size = carry_size
            else:
                current = []
                current_size = 0
        current.append(message)
        current_size += message_size
    if current:
        chunks.append(current)
    return chunks


def _normalize_evidence(summary: ChunkSummary, valid_ids: set[str]) -> ChunkSummary:
    payload = summary.model_dump()
    for collection_name in ("topics", "decisions", "action_items", "open_questions", "risks"):
        for item in payload[collection_name]:
            evidence = item.get("evidence_message_ids") or []
            item["evidence_message_ids"] = list(dict.fromkeys(str(value) for value in evidence if str(value) in valid_ids))
        payload[collection_name] = [item for item in payload[collection_name] if item["evidence_message_ids"]]
    return ChunkSummary.model_validate(payload)


class SummaryPipeline:
    def __init__(self, client: OpenAICompatibleClient):
        self.client = client

    async def _extract_chunk(
        self,
        chunk: list[ChatMessage],
        custom_instruction: str,
        max_tokens: int,
        temperature: float,
    ) -> ChunkSummary:
        schema = ChunkSummary.model_json_schema()
        transcript = "\n".join(render_message(message) for message in chunk)
        payload = await self.client.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_chunk_prompt(transcript, custom_instruction, schema),
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return ChunkSummary.model_validate(payload)

    async def _merge_group(
        self,
        summaries: list[ChunkSummary],
        custom_instruction: str,
        max_tokens: int,
        temperature: float,
    ) -> ChunkSummary:
        if len(summaries) == 1:
            return summaries[0]
        schema = ChunkSummary.model_json_schema()
        payload = await self.client.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_merge_prompt([item.model_dump() for item in summaries], custom_instruction, schema),
            schema=schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return ChunkSummary.model_validate(payload)

    async def _hierarchical_merge(
        self,
        summaries: list[ChunkSummary],
        custom_instruction: str,
        max_tokens: int,
        temperature: float,
        merge_max_chars: int,
    ) -> ChunkSummary:
        current = summaries
        while len(current) > 1:
            groups: list[list[ChunkSummary]] = []
            group: list[ChunkSummary] = []
            size = 0
            for summary in current:
                item_size = len(json.dumps(summary.model_dump(), ensure_ascii=False))
                if group and size + item_size > merge_max_chars:
                    groups.append(group)
                    group = []
                    size = 0
                group.append(summary)
                size += item_size
            if group:
                groups.append(group)
            if len(groups) == len(current):
                groups = [current[index : index + 2] for index in range(0, len(current), 2)]
            current = [
                await self._merge_group(group_items, custom_instruction, max_tokens, temperature)
                for group_items in groups
            ]
        return current[0]

    async def run(
        self,
        *,
        room_name: str,
        messages: list[ChatMessage],
        settings: dict[str, Any],
        custom_instruction: str = "",
    ) -> SummaryResult:
        ignored_types = {str(item).lower() for item in settings.get("ignored_message_types", [])}
        included = [message for message in messages if message.message_type.lower() not in ignored_types and message.content.strip()]
        if not included:
            raise ValueError("过滤后没有可用于总结的消息")

        max_chars = int(settings.get("chunk_max_chars", 12000))
        overlap = int(settings.get("chunk_overlap_messages", 2))
        max_tokens = int(settings.get("max_output_tokens", 2400))
        temperature = float(settings.get("temperature", 0.1))
        instruction = custom_instruction.strip() or str(settings.get("custom_instruction") or "").strip()
        chunks = chunk_messages(included, max_chars, overlap)
        chunk_summaries = [
            await self._extract_chunk(chunk, instruction, max_tokens, temperature)
            for chunk in chunks
        ]
        merged = await self._hierarchical_merge(
            chunk_summaries,
            instruction,
            max_tokens,
            temperature,
            merge_max_chars=max(8000, max_chars),
        )
        valid_ids = {message.id for message in included}
        merged = _normalize_evidence(merged, valid_ids)
        participants = {message.sender_name for message in included if message.sender_name}
        timestamps = [_timestamp_text(message.timestamp) for message in included if _timestamp_text(message.timestamp)]
        stats = SummaryStats(
            source_message_count=len(messages),
            included_message_count=len(included),
            chunk_count=len(chunks),
            participant_count=len(participants),
            started_at=timestamps[0] if timestamps else None,
            ended_at=timestamps[-1] if timestamps else None,
        )
        return SummaryResult(
            title=f"{room_name or '群聊'}摘要",
            stats=stats,
            **merged.model_dump(),
        )

import asyncio

from app.pipeline import SummaryPipeline, chunk_messages
from app.schemas import ChatMessage


class FakeModelClient:
    async def complete_json(self, **kwargs):
        return {
            "overview": "团队确定今天提交修复。",
            "topics": [
                {
                    "title": "支付修复",
                    "summary": "修复重复入账问题。",
                    "participants": ["小周"],
                    "evidence_message_ids": ["m1", "not-exists"],
                }
            ],
            "decisions": [],
            "action_items": [
                {
                    "task": "提交幂等修复",
                    "owner": "小周",
                    "deadline": "今天 18:00",
                    "status": "in_progress",
                    "evidence_message_ids": ["m1"],
                }
            ],
            "open_questions": [],
            "risks": [],
        }


def test_chunk_messages_keeps_overlap():
    messages = [
        ChatMessage(id=f"m{index}", sender_name="成员", content="x" * 80)
        for index in range(5)
    ]
    chunks = chunk_messages(messages, max_chars=300, overlap_messages=1)

    assert len(chunks) >= 2
    assert chunks[0][-1].id == chunks[1][0].id


def test_pipeline_filters_messages_and_drops_unknown_evidence():
    pipeline = SummaryPipeline(FakeModelClient())
    messages = [
        ChatMessage(id="m1", sender_name="小周", content="我今天 18 点前提交幂等修复。"),
        ChatMessage(id="m2", sender_name="系统", content="小王加入群聊", message_type="notice"),
    ]

    result = asyncio.run(
        pipeline.run(
            room_name="研发群",
            messages=messages,
            settings={
                "ignored_message_types": ["notice"],
                "chunk_max_chars": 12000,
                "chunk_overlap_messages": 2,
                "max_output_tokens": 2048,
                "temperature": 0.1,
            },
        )
    )

    assert result.title == "研发群摘要"
    assert result.stats.source_message_count == 2
    assert result.stats.included_message_count == 1
    assert result.topics[0].evidence_message_ids == ["m1"]

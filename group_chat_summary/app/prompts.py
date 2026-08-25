from __future__ import annotations

import json
from typing import Any


SYSTEM_PROMPT = """你是严谨的中文群聊信息分析员。你的任务是从提供的聊天记录中提取事实，而不是续写或猜测。

必须遵守：
1. 只能使用聊天记录里明确出现的信息，不得补充常识性推测。
2. 负责人、截止时间或状态不明确时使用 null 或 unknown。
3. 每个话题、决定、待办、问题和风险都必须附带原始消息 ID；没有证据就不要输出。
4. 区分讨论建议与最终决定，不要把疑问当成结论。
5. 合并重复表达，忽略寒暄、刷屏和无信息量内容。
6. 只返回符合指定 JSON Schema 的 JSON，不要 Markdown，不要解释。"""


def build_chunk_prompt(transcript: str, custom_instruction: str, schema: dict[str, Any]) -> str:
    return f"""请提取下面这段群聊中的关键信息。

补充要求：{custom_instruction or '无'}

输出 JSON Schema：
{json.dumps(schema, ensure_ascii=False)}

聊天记录：
{transcript}
"""


def build_merge_prompt(chunk_summaries: list[dict[str, Any]], custom_instruction: str, schema: dict[str, Any]) -> str:
    return f"""请合并以下分段摘要，去重并解决前后状态变化。较晚消息明确推翻较早结论时，以较晚消息为准，同时保留相关证据 ID。

补充要求：{custom_instruction or '无'}

输出 JSON Schema：
{json.dumps(schema, ensure_ascii=False)}

分段摘要：
{json.dumps(chunk_summaries, ensure_ascii=False)}
"""

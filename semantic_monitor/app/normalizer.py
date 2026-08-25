from __future__ import annotations

import html
import re
import unicodedata


_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_SPACE = re.compile(r"\s+")
_REPEATED_PUNCT = re.compile(r"([!?！？。,.，])\1+")


def normalize_text(text: str) -> str:
    """保留中文语义信息，同时消除常见绕过字符。"""
    value = html.unescape(str(text or ""))
    value = unicodedata.normalize("NFKC", value)
    value = _ZERO_WIDTH.sub("", value)
    value = value.replace("\u3000", " ")
    value = _REPEATED_PUNCT.sub(r"\1", value)
    value = _SPACE.sub(" ", value).strip()
    return value


def compact_for_rule(text: str) -> str:
    value = normalize_text(text).lower()
    return re.sub(r"[\s\-_.·•,，。!！?？:：;；/\\|]+", "", value)

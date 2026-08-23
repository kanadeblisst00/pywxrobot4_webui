"""插件资源上传路径校验。"""

from __future__ import annotations

from io import BytesIO
import re
from os import PathLike
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from core.config import PROJECT_ROOT

PLUGIN_ASSET_IMAGE_FORMATS = {
    ".png": frozenset({"PNG"}),
    ".jpg": frozenset({"JPEG"}),
    ".jpeg": frozenset({"JPEG"}),
    ".gif": frozenset({"GIF"}),
    ".webp": frozenset({"WEBP"}),
    ".bmp": frozenset({"BMP"}),
}
PLUGIN_ASSET_IMAGE_EXTENSIONS = frozenset(PLUGIN_ASSET_IMAGE_FORMATS)
PLUGIN_ASSET_MAX_PIXELS = 40_000_000
PLUGIN_ASSET_UPLOAD_DIR_NAME = "uploads"


def sanitize_upload_path_segment(value: str | PathLike[str] | None, fallback: str = "file") -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._")
    return sanitized or fallback


def resolve_project_relative_dir(value: str | None, default: str = PLUGIN_ASSET_UPLOAD_DIR_NAME) -> Path:
    raw_value = str(value or default).strip().replace("\\", "/").strip("/")
    relative_dir = Path(raw_value or default)
    if relative_dir.is_absolute() or any(part == ".." for part in relative_dir.parts):
        raise ValueError("上传目录必须是项目根目录下的相对路径")
    upload_root = (PROJECT_ROOT / PLUGIN_ASSET_UPLOAD_DIR_NAME).resolve()
    resolved_dir = (PROJECT_ROOT / relative_dir).resolve()
    if resolved_dir != upload_root and upload_root not in resolved_dir.parents:
        raise ValueError("上传目录必须位于项目 uploads 目录内")
    return resolved_dir


async def read_upload_with_limit(upload: Any, max_bytes: int) -> bytes:
    limit = max(1, int(max_bytes))
    try:
        content = bytes(await upload.read(limit + 1))
    finally:
        await upload.close()
    return content


def validate_plugin_asset_image(content: bytes, suffix: str) -> None:
    normalized_suffix = str(suffix or "").strip().lower()
    expected_formats = PLUGIN_ASSET_IMAGE_FORMATS.get(normalized_suffix)
    if not expected_formats:
        raise ValueError("不支持的图片扩展名")
    try:
        with Image.open(BytesIO(content)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > PLUGIN_ASSET_MAX_PIXELS:
                raise ValueError("图片像素尺寸过大或无效")
            detected_format = str(image.format or "").strip().upper()
            image.verify()
    except ValueError:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("文件内容不是有效图片") from exc
    if detected_format not in expected_formats:
        raise ValueError("图片内容与文件扩展名不匹配")

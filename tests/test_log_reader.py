import asyncio
from datetime import datetime
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from server.log_reader import build_log_entries, filter_log_entries, parse_log_line
from server.upload_paths import (
    read_upload_with_limit,
    resolve_project_relative_dir,
    sanitize_upload_path_segment,
    validate_plugin_asset_image,
)


def test_parse_log_line() -> None:
    line = "2026-07-06 12:00:00.123 | INFO     | server:create_app:100 - started"
    parsed = parse_log_line(line)
    assert parsed is not None
    assert parsed["level"] == "INFO"
    assert parsed["message"] == "started"


def test_filter_log_entries_by_level() -> None:
    entries = build_log_entries(
        [
            "2026-07-06 12:00:00.123 | INFO     | server:create_app:1 - ok",
            "2026-07-06 12:00:01.123 | ERROR    | server:create_app:2 - bad",
        ]
    )
    filtered = filter_log_entries(entries, "all", "ERROR", "", "")
    assert len(filtered) == 1
    assert filtered[0]["level"] == "ERROR"


def test_sanitize_upload_path_segment() -> None:
    assert sanitize_upload_path_segment("hello world") == "hello_world"


def test_resolve_project_relative_dir(tmp_path, monkeypatch) -> None:
    import server.upload_paths as upload_paths

    monkeypatch.setattr(upload_paths, "PROJECT_ROOT", tmp_path)
    resolved = resolve_project_relative_dir("uploads/assets")
    assert resolved == (tmp_path / "uploads" / "assets").resolve()
    with pytest.raises(ValueError):
        resolve_project_relative_dir("static/assets")


def _render_image(image_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    image = Image.new("RGB", (2, 2), color="red")
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def test_validate_plugin_asset_image_accepts_matching_content() -> None:
    content = _render_image("PNG")
    validate_plugin_asset_image(content, ".png")


def test_validate_plugin_asset_image_rejects_mismatched_extension() -> None:
    content = _render_image("PNG")
    with pytest.raises(ValueError, match="扩展名"):
        validate_plugin_asset_image(content, ".jpg")


def test_validate_plugin_asset_image_rejects_invalid_content() -> None:
    with pytest.raises(ValueError, match="有效图片"):
        validate_plugin_asset_image(b"not-an-image", ".png")


def test_read_upload_with_limit_reads_only_one_byte_past_limit_and_closes() -> None:
    class FakeUpload:
        def __init__(self) -> None:
            self.requested = 0
            self.closed = False

        async def read(self, size: int) -> bytes:
            self.requested = size
            return b"x" * size

        async def close(self) -> None:
            self.closed = True

    upload = FakeUpload()
    content = asyncio.run(read_upload_with_limit(upload, 4))
    assert len(content) == 5
    assert upload.requested == 5
    assert upload.closed is True

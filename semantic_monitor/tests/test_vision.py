from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import AppConfig
from app.main import create_app
from app.schemas import NSFWResult, OCRLine, OCRResult
from app.vision import VisionRuntime


def make_config(tmp_path) -> AppConfig:
    data_dir = tmp_path / "data"
    return AppConfig(
        host="127.0.0.1",
        port=28110,
        database_path=data_dir / "test.sqlite3",
        api_token="",
        max_context_messages=8,
        data_dir=data_dir,
        artifacts_dir=data_dir / "artifacts",
        datasets_dir=data_dir / "datasets",
        web_dir=tmp_path,
    )


def png_bytes(color: str = "white") -> bytes:
    image = Image.new("RGB", (320, 180), color)
    handle = io.BytesIO()
    image.save(handle, "PNG")
    return handle.getvalue()


def test_qrcode_detection_does_not_return_content() -> None:
    zxingcpp = pytest.importorskip("zxingcpp")
    barcode = zxingcpp.create_barcode("private payload must not be exposed", zxingcpp.BarcodeFormat.QRCode)
    pixels = np.asarray(zxingcpp.write_barcode_to_image(barcode, 240))
    result = VisionRuntime._detect_qrcode(pixels)
    payload = result.model_dump()
    assert result.detected is True
    assert result.count == 1
    assert not ({"text", "content", "bytes"} & payload.keys())


def test_image_endpoint_runs_ocr_text_through_moderation(tmp_path, monkeypatch) -> None:
    app = create_app(make_config(tmp_path))
    monkeypatch.setattr(
        app.state.vision,
        "_detect_ocr",
        lambda _pixels, provider: OCRResult(
            enabled=True,
            available=True,
            provider=provider,
            status="ok",
            text="兼职日结，加微信 abc123 了解详情",
            lines=[OCRLine(text="兼职日结，加微信 abc123 了解详情", confidence=0.98)],
            average_confidence=0.98,
        ),
    )
    monkeypatch.setattr(
        app.state.vision,
        "_detect_nsfw",
        lambda _image, provider, threshold: NSFWResult(
            enabled=True,
            available=True,
            provider=provider,
            status="ok",
            score=0.02,
            matched=False,
            threshold=threshold,
        ),
    )
    client = TestClient(app)
    response = client.post(
        "/api/v1/images/analyze",
        files={"file": ("poster.png", png_bytes(), "image/png")},
        data={"room_id": "test@chatroom", "persist": "false"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["width"] == 320
    assert payload["ocr"]["text"].startswith("兼职日结")
    assert payload["text_analysis"]["matched"] is True
    assert any(item["topic_name"] == "广告" for item in payload["text_analysis"]["matches"])


def test_image_endpoint_rejects_non_image(tmp_path) -> None:
    client = TestClient(create_app(make_config(tmp_path)))
    response = client.post(
        "/api/v1/images/analyze",
        files={"file": ("fake.png", b"not an image", "image/png")},
        data={"room_id": "test@chatroom", "persist": "false"},
    )
    assert response.status_code == 400

from __future__ import annotations

import asyncio
import importlib.util
import io
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from .engine import MonitorEngine, SEVERITY_RANK
from .repository import Repository
from .schemas import (
    ImageAnalyzeResponse,
    MessageInput,
    NSFWFinding,
    NSFWResult,
    OCRLine,
    OCRResult,
    QRCodeResult,
)


EXPLICIT_NUDENET_CLASSES = {
    "ANUS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


class VisionRuntime:
    def __init__(self, repository: Repository, text_engine: MonitorEngine):
        self.repository = repository
        self.text_engine = text_engine
        self._rapidocr: Any = None
        self._nudenet: Any = None

    def status(self) -> dict[str, Any]:
        settings = self.repository.get_model_settings()
        return {
            "qrcode": {
                "enabled": settings.qrcode_enabled,
                "provider": "zxingcpp",
                "available": _module_available("zxingcpp"),
                "privacy": "二维码内容不会被读取或返回",
            },
            "ocr": {
                "enabled": settings.ocr_provider != "disabled",
                "provider": settings.ocr_provider,
                "available": settings.ocr_provider == "disabled" or _module_available("rapidocr_onnxruntime") or _module_available("rapidocr"),
            },
            "nsfw": {
                "enabled": settings.nsfw_provider != "disabled",
                "provider": settings.nsfw_provider,
                "available": settings.nsfw_provider == "disabled" or _module_available(settings.nsfw_provider),
                "threshold": settings.nsfw_threshold,
            },
        }

    @staticmethod
    def _open_image(content: bytes, max_megapixels: float) -> tuple[Image.Image, str]:
        if not content:
            raise ValueError("图片内容为空")
        try:
            probe = Image.open(io.BytesIO(content))
            probe.verify()
            image = Image.open(io.BytesIO(content)).convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
            raise ValueError("无法识别图片格式或图片已经损坏") from exc
        pixels = image.width * image.height
        if pixels > max_megapixels * 1_000_000:
            raise ValueError(f"图片像素超过限制：最多 {max_megapixels:g} MP")
        media_type = Image.MIME.get(probe.format or "", "application/octet-stream")
        return image, media_type

    @staticmethod
    def _disabled_qrcode() -> QRCodeResult:
        return QRCodeResult(enabled=False, available=True, provider="zxingcpp", status="disabled")

    @staticmethod
    def _detect_qrcode(pixel_array: np.ndarray) -> QRCodeResult:
        try:
            import zxingcpp
        except ImportError:
            return QRCodeResult(enabled=True, available=False, provider="zxingcpp", status="unavailable", error="缺少 zxing-cpp")
        try:
            barcodes = zxingcpp.read_barcodes(pixel_array)
            formats: list[str] = []
            for barcode in barcodes:
                format_value = getattr(getattr(barcode, "format", None), "name", "") or str(getattr(barcode, "format", ""))
                if "qr" in format_value.lower():
                    # 故意不读取 barcode.text / bytes：这里只判断二维码是否存在。
                    formats.append(format_value or "QRCode")
            return QRCodeResult(
                enabled=True,
                available=True,
                provider="zxingcpp",
                status="ok",
                detected=bool(formats),
                count=len(formats),
                formats=formats,
            )
        except Exception as exc:
            return QRCodeResult(enabled=True, available=True, provider="zxingcpp", status="failed", error=str(exc))

    def _ocr_engine(self) -> Any:
        if self._rapidocr is not None:
            return self._rapidocr
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise RuntimeError("缺少 RapidOCR，请安装 requirements-vision.txt") from exc
        self._rapidocr = RapidOCR()
        return self._rapidocr

    def _detect_ocr(self, pixel_array: np.ndarray, provider: str) -> OCRResult:
        if provider == "disabled":
            return OCRResult(enabled=False, available=True, provider=provider, status="disabled")
        try:
            engine = self._ocr_engine()
            output = engine(pixel_array)
            result = output[0] if isinstance(output, tuple) else output
            lines: list[OCRLine] = []
            if isinstance(result, list):
                for item in result:
                    if not isinstance(item, (list, tuple)) or len(item) < 3:
                        continue
                    text = str(item[1] or "").strip()
                    if text:
                        lines.append(OCRLine(text=text, confidence=max(0.0, min(1.0, float(item[2])))))
            else:
                texts = getattr(result, "txts", None) or getattr(result, "texts", None) or []
                scores = getattr(result, "scores", None) or [1.0] * len(texts)
                lines = [
                    OCRLine(text=str(text).strip(), confidence=max(0.0, min(1.0, float(score))))
                    for text, score in zip(texts, scores)
                    if str(text).strip()
                ]
            average = sum(item.confidence for item in lines) / len(lines) if lines else None
            return OCRResult(
                enabled=True,
                available=True,
                provider=provider,
                status="ok",
                text="\n".join(item.text for item in lines),
                lines=lines,
                average_confidence=round(average, 4) if average is not None else None,
            )
        except RuntimeError as exc:
            return OCRResult(enabled=True, available=False, provider=provider, status="unavailable", error=str(exc))
        except Exception as exc:
            return OCRResult(enabled=True, available=True, provider=provider, status="failed", error=str(exc))

    def _nudenet_engine(self) -> Any:
        if self._nudenet is None:
            try:
                from nudenet import NudeDetector
            except ImportError as exc:
                raise RuntimeError("缺少 NudeNet，请安装 requirements-vision.txt") from exc
            self._nudenet = NudeDetector()
        return self._nudenet

    def _detect_nsfw(self, image: Image.Image, provider: str, threshold: float) -> NSFWResult:
        if provider == "disabled":
            return NSFWResult(enabled=False, available=True, provider=provider, status="disabled", threshold=threshold)
        try:
            findings: list[NSFWFinding] = []
            if provider == "nudenet":
                detector = self._nudenet_engine()
                with tempfile.TemporaryDirectory(prefix="semantic-monitor-") as directory:
                    path = Path(directory) / "image.jpg"
                    image.save(path, format="JPEG", quality=92)
                    detections = detector.detect(str(path))
                for item in detections if isinstance(detections, list) else []:
                    label = str(item.get("class") or item.get("label") or "").upper()
                    score = float(item.get("score") or 0)
                    if label in EXPLICIT_NUDENET_CLASSES:
                        findings.append(NSFWFinding(label=label, score=round(score, 4)))
                score = max((item.score for item in findings), default=0.0)
            elif provider == "opennsfw2":
                try:
                    import opennsfw2
                except ImportError as exc:
                    raise RuntimeError("缺少 OpenNSFW2 及其 Keras 后端") from exc
                score = float(opennsfw2.predict_image(image))
                findings = [NSFWFinding(label="NSFW", score=round(score, 4))]
            else:
                raise RuntimeError(f"不支持的 NSFW 模型：{provider}")
            return NSFWResult(
                enabled=True,
                available=True,
                provider=provider,
                status="ok",
                score=round(score, 4),
                matched=score >= threshold,
                threshold=threshold,
                findings=sorted(findings, key=lambda item: item.score, reverse=True),
            )
        except RuntimeError as exc:
            return NSFWResult(enabled=True, available=False, provider=provider, status="unavailable", threshold=threshold, error=str(exc))
        except Exception as exc:
            return NSFWResult(enabled=True, available=True, provider=provider, status="failed", threshold=threshold, error=str(exc))

    async def analyze(
        self,
        *,
        content: bytes,
        filename: str,
        message_id: str,
        room_id: str,
        sender_id: str,
        sender_name: str,
        context: list[str],
        persist: bool,
    ) -> ImageAnalyzeResponse:
        started = perf_counter()
        settings = self.repository.get_model_settings()
        image, media_type = self._open_image(content, settings.max_image_megapixels)
        pixels = np.asarray(image)
        qrcode_task = asyncio.to_thread(self._detect_qrcode, pixels) if settings.qrcode_enabled else asyncio.to_thread(self._disabled_qrcode)
        ocr_task = asyncio.to_thread(self._detect_ocr, pixels, settings.ocr_provider)
        nsfw_task = asyncio.to_thread(self._detect_nsfw, image.copy(), settings.nsfw_provider, settings.nsfw_threshold)
        qrcode, ocr, nsfw = await asyncio.gather(qrcode_task, ocr_task, nsfw_task)

        warnings = [item.error for item in (qrcode, ocr, nsfw) if item.error]
        text_analysis = None
        if ocr.text.strip():
            text_analysis = await self.text_engine.analyze(
                MessageInput(
                    message_id=f"{message_id}:ocr" if message_id else "",
                    room_id=room_id,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    text=ocr.text[:10000],
                    message_type="image_ocr",
                    context=context,
                    metadata={"source": "image_ocr", "filename": filename},
                    persist=persist,
                )
            )
        event_ids: list[int] = list(text_analysis.event_ids) if text_analysis else []
        if nsfw.matched and persist:
            topic = next((item for item in self.repository.list_topics(enabled_only=True) if item.name in {"色情", "淫秽色情"}), None)
            if topic:
                event_ids.append(
                    self.repository.add_event(
                        {
                            "message_id": message_id,
                            "room_id": room_id,
                            "sender_id": sender_id,
                            "sender_name": sender_name,
                            "text": f"[图片色情风险] {filename}",
                            "normalized_text": "[图片色情风险]",
                            "topic_id": topic.id,
                            "topic_name": topic.name,
                            "severity": topic.severity,
                            "confidence": nsfw.score or 0,
                            "semantic_score": None,
                            "rule_score": 0,
                            "classifier_score": nsfw.score,
                            "stage": "classifier",
                            "evidence": f"{nsfw.provider} 图片检测分数 {nsfw.score:.3f}",
                            "needs_review": True,
                        }
                    )
                )
        matched = nsfw.matched or bool(text_analysis and text_analysis.matched)
        risk_level = "critical" if nsfw.matched else (text_analysis.risk_level if text_analysis else "none")
        if text_analysis and SEVERITY_RANK.get(text_analysis.risk_level, 0) > SEVERITY_RANK.get(risk_level, 0):
            risk_level = text_analysis.risk_level
        if event_ids:
            warnings.append(f"已记录 {len(event_ids)} 条图片审核事件")
        return ImageAnalyzeResponse(
            message_id=message_id,
            filename=filename,
            media_type=media_type,
            width=image.width,
            height=image.height,
            size_bytes=len(content),
            matched=matched,
            risk_level=risk_level,
            processing_ms=round((perf_counter() - started) * 1000, 2),
            qrcode=qrcode,
            ocr=ocr,
            nsfw=nsfw,
            text_analysis=text_analysis,
            event_ids=event_ids,
            warnings=warnings,
        )

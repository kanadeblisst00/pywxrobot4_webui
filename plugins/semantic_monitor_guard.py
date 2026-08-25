import json
import mimetypes
from pathlib import Path
from typing import Any

from core.config import PROJECT_ROOT
from ._plugin_sdk import MESSAGE_TYPES, async_http_post, get_message_type, normalize_text, post_json_request, unique_strings


name = "semantic_monitor_guard"
description = "调用本地内容审核中心，审核指定群聊的文本消息并记录或提醒"
event_filters = ["text", "image"]
scope_targets = ["rooms"]

DEFAULT_SERVICE_URL = "http://127.0.0.1:28110/api/v1/messages/analyze"
DEFAULT_IMAGE_SERVICE_URL = "http://127.0.0.1:28110/api/v1/images/analyze"
DEFAULT_WARNING = "@{user_name} 这条消息可能涉及{topics}，请遵守群规。"

config_schema = [
    {
        "key": "service_url",
        "label": "审核接口地址",
        "type": "url",
        "default": DEFAULT_SERVICE_URL,
        "description": "本地 semantic_monitor 服务的消息审核接口。",
    },
    {
        "key": "api_token",
        "label": "API Token",
        "type": "password",
        "default": "",
        "description": "仅在审核服务配置了 SEMANTIC_MONITOR_API_TOKEN 时填写。",
    },
    {
        "key": "image_service_url",
        "label": "图片审核接口地址",
        "type": "url",
        "default": DEFAULT_IMAGE_SERVICE_URL,
        "description": "图片消息上传到这里执行色情、OCR 和二维码存在性检测。",
    },
    {
        "key": "room_ids",
        "label": "生效群聊",
        "type": "searchable-multi-checkbox",
        "options_source": "room_options",
        "default": [],
        "search_placeholder": "输入群名称或 wxid 搜索",
        "show_selected_label": "仅显示已勾选群聊",
        "empty_text": "没有匹配到群聊。",
        "empty_no_options_text": "当前还没有可选群聊。",
        "description": "为空时不审核任何群聊。",
    },
    {
        "key": "send_group_warning",
        "label": "高置信度命中时群内提醒",
        "type": "checkbox",
        "default": False,
        "description": "默认仅写入审核中心和插件日志；建议完成阈值校准后再开启。",
    },
    {
        "key": "notify_wxid",
        "label": "管理员通知 wxid",
        "type": "text",
        "default": "",
        "description": "可选。命中后私聊该微信账号；留空则不私聊。",
    },
    {
        "key": "notify_on_qrcode",
        "label": "检测到二维码时通知管理员",
        "type": "checkbox",
        "default": False,
        "description": "只通知“图片包含二维码”，不会读取或审核二维码中的文本或网址。",
    },
    {
        "key": "warning_template",
        "label": "提醒文本",
        "type": "text",
        "default": DEFAULT_WARNING,
        "description": "支持 @{user_name}、{topics}、{confidence}、{room_id}。",
    },
    {
        "key": "timeout_seconds",
        "label": "接口超时（秒）",
        "type": "number",
        "default": 30,
        "min": 1,
        "max": 60,
    },
]


def normalize_room_ids(config: Any) -> list[str]:
    value = config.get("room_ids") if isinstance(config, dict) else []
    return [item for item in unique_strings(value) if item.endswith("@chatroom")]


def strip_sender_prefix(content: Any, sender_wxid: str) -> str:
    value = str(content or "").strip()
    for prefix in (f"{sender_wxid}:\n", f"{sender_wxid}:\r\n", f"{sender_wxid}：\n"):
        if sender_wxid and value.startswith(prefix):
            return value[len(prefix):].strip()
    return value


def sender_name_from_event(event: Any, sender_wxid: str) -> str:
    first_non_empty = getattr(event, "first_non_empty", None)
    if callable(first_non_empty):
        value = normalize_text(first_non_empty("room_sender_display_name", "sender_display_name"))
        if value:
            return value
    return sender_wxid or "群成员"


def render_notice(template: Any, *, sender_name: str, topics: list[str], confidence: float, room_id: str) -> str:
    value = str(template or DEFAULT_WARNING).strip() or DEFAULT_WARNING
    mention = f"@{sender_name}\u2005"
    value = value.replace("@{user_name}", mention).replace("{user_name}", sender_name)
    value = value.replace("{topics}", "、".join(topics) or "违规内容")
    value = value.replace("{confidence}", f"{confidence:.0%}").replace("{room_id}", room_id)
    return value


def resolve_downloaded_image_path(response: Any) -> Path | None:
    if not isinstance(response, dict):
        return None
    candidates: list[Any] = [response.get(key) for key in ("path", "save_path", "file_path", "download_path")]
    data = response.get("data")
    if isinstance(data, dict):
        candidates.extend(data.get(key) for key in ("path", "save_path", "file_path", "download_path"))
    for value in candidates:
        path_text = normalize_text(value)
        if not path_text:
            continue
        path = Path(path_text)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if path.is_file():
            return path
    return None


async def handle_image_message(event: Any, context: Any, room_id: str, sender_wxid: str, sender_name: str):
    msgid = normalize_text(event.normalized_msgid)
    try:
        download_response = await context.api.download_cdn_image(
            msgid=msgid,
            wxid=room_id,
            wxpid=event.normalized_wxpid,
            flag=1,
            wait=True,
            timeout=int(context.settings.image_download_timeout or 15),
        )
        image_path = resolve_downloaded_image_path(download_response)
        if image_path is None:
            raise RuntimeError("未找到图片下载结果路径")
        content = image_path.read_bytes()
        media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        image_url = normalize_text(context.config.get("image_service_url")) or DEFAULT_IMAGE_SERVICE_URL
        token = normalize_text(context.config.get("api_token"))
        headers = {"X-API-Token": token} if token else None
        status, response_text = await async_http_post(
            image_url,
            headers=headers,
            files={"file": (image_path.name, content, media_type)},
            data={
                "message_id": msgid,
                "room_id": room_id,
                "sender_id": sender_wxid,
                "sender_name": sender_name,
                "persist": "true",
            },
            timeout=float(context.config.get("timeout_seconds") or 30),
        )
        result = json.loads(response_text)
    except Exception as exc:
        context.logger.warning(
            "本地图片审核服务调用失败，本条图片已放行",
            {"roomid": room_id, "sender_wxid": sender_wxid, "msgid": msgid, "error": str(exc)},
        )
        return {"handled": False, "detail": "图片审核服务不可用，已放行"}

    topics: list[str] = []
    confidences: list[float] = []
    nsfw = result.get("nsfw") or {}
    if nsfw.get("matched"):
        topics.append("图片色情风险")
        confidences.append(float(nsfw.get("score") or 0))
    text_analysis = result.get("text_analysis") or {}
    for item in text_analysis.get("matches") or []:
        if item.get("matched"):
            topics.append(f"海报文字-{item.get('topic_name')}")
            confidences.append(float(item.get("confidence") or 0))
    qrcode_detected = bool((result.get("qrcode") or {}).get("detected"))
    if qrcode_detected:
        topics.append("包含二维码")
    topics = unique_strings(topics)
    if not topics:
        return {"handled": False, "detail": ""}

    confidence = max(confidences, default=1.0 if qrcode_detected else 0.0)
    notice = render_notice(
        context.config.get("warning_template"),
        sender_name=sender_name,
        topics=topics,
        confidence=confidence,
        room_id=room_id,
    )
    actions: list[str] = ["recorded"]
    actual_violation = bool(nsfw.get("matched") or text_analysis.get("matched"))
    if actual_violation and bool(context.config.get("send_group_warning")):
        await context.api.send_text(wxid=room_id, content=notice, atlist=sender_wxid, wxpid=event.normalized_wxpid)
        actions.append("warned")
    notify_wxid = normalize_text(context.config.get("notify_wxid"))
    should_notify = actual_violation or (qrcode_detected and bool(context.config.get("notify_on_qrcode")))
    if notify_wxid and should_notify:
        await context.api.send_text(
            wxid=notify_wxid,
            content=f"群聊 {room_id} 图片审核：{notice}",
            wxpid=event.normalized_wxpid,
        )
        actions.append("notified")
    context.logger.warning(
        "群图片命中本地内容审核",
        {
            "roomid": room_id,
            "sender_wxid": sender_wxid,
            "topics": topics,
            "qrcode_detected": qrcode_detected,
            "qrcode_content_read": False,
            "nsfw_score": nsfw.get("score"),
            "actions": actions,
            "status": status,
        },
    )
    return {"handled": True, "detail": f"图片审核命中：{'、'.join(topics)}", "data": {"topics": topics, "actions": actions}}


async def handle_message(event: Any, context: Any):
    message_type = get_message_type(event)
    if message_type not in {MESSAGE_TYPES.TEXT, MESSAGE_TYPES.IMAGE}:
        return {"handled": False, "detail": ""}
    room_id = normalize_text(event.conversation_wxid)
    if room_id not in set(normalize_room_ids(context.config)):
        return {"handled": False, "detail": ""}
    sender_wxid = normalize_text(event.sender_wxid)
    sender_name = sender_name_from_event(event, sender_wxid)
    if message_type == MESSAGE_TYPES.IMAGE:
        return await handle_image_message(event, context, room_id, sender_wxid, sender_name)
    text = strip_sender_prefix(getattr(event, "content", "") or event.normalized_content, sender_wxid)
    if not text:
        return {"handled": False, "detail": ""}

    service_url = normalize_text(context.config.get("service_url")) or DEFAULT_SERVICE_URL
    token = normalize_text(context.config.get("api_token"))
    headers = {"X-API-Token": token} if token else None
    payload = {
        "message_id": normalize_text(event.normalized_msgid),
        "room_id": room_id,
        "sender_id": sender_wxid,
        "sender_name": sender_name,
        "text": text,
        "message_type": "text",
        "metadata": {"wxpid": event.normalized_wxpid},
        "persist": True,
    }
    try:
        status, response_text = await post_json_request(
            service_url,
            payload,
            headers=headers,
            timeout=float(context.config.get("timeout_seconds") or 8),
        )
        result = json.loads(response_text)
    except Exception as exc:
        context.logger.warning(
            "本地内容审核服务调用失败，本条消息已放行",
            {"roomid": room_id, "sender_wxid": sender_wxid, "service_url": service_url, "error": str(exc)},
        )
        return {"handled": False, "detail": "审核服务不可用，已放行"}

    matched = [item for item in result.get("matches", []) if item.get("matched")]
    if not matched:
        return {"handled": False, "detail": ""}
    topics = unique_strings([item.get("topic_name") for item in matched])
    confidence = max(float(item.get("confidence") or 0) for item in matched)
    notice = render_notice(
        context.config.get("warning_template"),
        sender_name=sender_name,
        topics=topics,
        confidence=confidence,
        room_id=room_id,
    )
    actions: list[str] = ["recorded"]
    if bool(context.config.get("send_group_warning")):
        await context.api.send_text(
            wxid=room_id,
            content=notice,
            atlist=sender_wxid,
            wxpid=event.normalized_wxpid,
        )
        actions.append("warned")
    notify_wxid = normalize_text(context.config.get("notify_wxid"))
    if notify_wxid:
        await context.api.send_text(
            wxid=notify_wxid,
            content=f"群聊 {room_id} 命中内容审核：{notice}\n原消息：{text[:500]}",
            wxpid=event.normalized_wxpid,
        )
        actions.append("notified")
    context.logger.warning(
        "群消息命中本地内容审核",
        {
            "roomid": room_id,
            "sender_wxid": sender_wxid,
            "topics": topics,
            "confidence": confidence,
            "risk_level": result.get("risk_level"),
            "actions": actions,
            "status": status,
        },
    )
    return {
        "handled": True,
        "detail": f"命中内容审核：{'、'.join(topics)}",
        "data": {"topics": topics, "confidence": confidence, "risk_level": result.get("risk_level"), "actions": actions},
    }

import asyncio
import json
from types import SimpleNamespace

from plugins import semantic_monitor_guard as plugin


def test_strip_sender_prefix_and_notice_rendering():
    assert plugin.strip_sender_prefix("wxid_a:\n测试消息", "wxid_a") == "测试消息"
    notice = plugin.render_notice(
        "@{user_name} 命中{topics}（{confidence}）",
        sender_name="小王",
        topics=["赌博", "广告"],
        confidence=0.91,
        room_id="1@chatroom",
    )
    assert "@小王" in notice
    assert "赌博、广告" in notice
    assert "91%" in notice


def test_normalize_room_ids_filters_non_rooms():
    assert plugin.normalize_room_ids({"room_ids": ["a@chatroom", "wxid_user", "a@chatroom"]}) == ["a@chatroom"]


def test_image_message_only_reports_qrcode_presence(tmp_path, monkeypatch):
    image_path = tmp_path / "qr.png"
    image_path.write_bytes(b"fake-image-upload")

    class Api:
        def __init__(self):
            self.sent = []

        async def download_cdn_image(self, **_kwargs):
            return {"path": str(image_path)}

        async def send_text(self, **kwargs):
            self.sent.append(kwargs)

    async def fake_post(*_args, **_kwargs):
        return 200, json.dumps({
            "matched": False,
            "risk_level": "none",
            "qrcode": {"detected": True, "count": 1, "formats": ["QRCode"]},
            "ocr": {"text": "", "lines": []},
            "nsfw": {"matched": False, "score": 0.01},
            "text_analysis": None,
        })

    monkeypatch.setattr(plugin, "async_http_post", fake_post)
    api = Api()
    logger = SimpleNamespace(warning=lambda *_args, **_kwargs: None)
    context = SimpleNamespace(
        api=api,
        logger=logger,
        settings=SimpleNamespace(image_download_timeout=15),
        config={
            "room_ids": ["a@chatroom"],
            "notify_wxid": "wxid_admin",
            "notify_on_qrcode": True,
            "send_group_warning": True,
        },
    )
    event = SimpleNamespace(
        normalized_local_type=int(plugin.MESSAGE_TYPES.IMAGE),
        normalized_msg_type=int(plugin.MESSAGE_TYPES.IMAGE),
        local_type=int(plugin.MESSAGE_TYPES.IMAGE),
        msg_type=int(plugin.MESSAGE_TYPES.IMAGE),
        conversation_wxid="a@chatroom",
        sender_wxid="wxid_sender",
        normalized_msgid="m-1",
        normalized_wxpid=1,
        first_non_empty=lambda *_args: "小王",
    )
    result = asyncio.run(plugin.handle_message(event, context))
    assert result["handled"] is True
    assert result["data"]["topics"] == ["包含二维码"]
    assert len(api.sent) == 1
    assert api.sent[0]["wxid"] == "wxid_admin"
    assert "二维码" in api.sent[0]["content"]

import asyncio

from routes.observability_routes import (
    SSE_RESPONSE_HEADERS,
    generate_runtime_event_stream,
)
from runtime.events import RuntimeEventHub


def test_runtime_event_stream_emits_events_heartbeats_and_unsubscribes() -> None:
    async def collect_stream() -> tuple[str, str, str, set]:
        event_hub = RuntimeEventHub()
        stream = generate_runtime_event_stream(event_hub, heartbeat_interval_seconds=0.01)
        connected = await anext(stream)
        await event_hub.publish("message_processed", {"internal_id": 1})
        event = await anext(stream)
        heartbeat = await anext(stream)
        await stream.aclose()
        return connected, event, heartbeat, event_hub._subscribers

    connected, event, heartbeat, subscribers = asyncio.run(collect_stream())
    assert '"type": "connected"' in connected
    assert '"type": "message_processed"' in event
    assert heartbeat == ": heartbeat\n\n"
    assert subscribers == set()


def test_sse_response_headers_disable_proxy_buffering() -> None:
    assert SSE_RESPONSE_HEADERS["Cache-Control"] == "no-cache, no-transform"
    assert SSE_RESPONSE_HEADERS["X-Accel-Buffering"] == "no"

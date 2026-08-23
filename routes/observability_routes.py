"""健康检查、指标与 SSE 事件流路由。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from server.context import AppContext


SSE_HEARTBEAT_INTERVAL_SECONDS = 15.0
SSE_RESPONSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
}


async def generate_runtime_event_stream(event_hub, heartbeat_interval_seconds: float = SSE_HEARTBEAT_INTERVAL_SECONDS) -> AsyncIterator[str]:
    queue = await event_hub.subscribe()
    try:
        connected_event = json.dumps({"type": "connected", "payload": {}}, ensure_ascii=False)
        yield f"data: {connected_event}\n\n"
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=max(0.1, heartbeat_interval_seconds))
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except asyncio.CancelledError:
        raise
    finally:
        await event_hub.unsubscribe(queue)


def register_observability_routes(app: FastAPI, ctx: AppContext) -> None:
    runtime = ctx.runtime

    @app.get("/api/events/stream")
    async def stream_runtime_events() -> StreamingResponse:
        return StreamingResponse(
            generate_runtime_event_stream(runtime.event_hub),
            media_type="text/event-stream",
            headers=SSE_RESPONSE_HEADERS,
        )

    @app.get("/health")
    async def health() -> dict:
        uptime_seconds = int((datetime.now().astimezone() - runtime.started_at).total_seconds()) if runtime.started_at else 0
        return {
            "status": "ok",
            "queued_messages": runtime.queue.qsize(),
            "queue_size": runtime.settings.queue_size,
            "worker_count": runtime.settings.worker_count,
            "loaded_plugin_count": len(runtime.manager.plugins),
            "enabled_plugin_count": len(runtime.settings.plugins),
            "uptime_seconds": uptime_seconds,
            "wxrobot_api_reachable": runtime.wxrobot_api_reachable,
            "heartbeat_healthy": runtime.heartbeat_healthy,
            "plugins": [plugin.name for plugin in runtime.manager.plugins],
        }

    @app.get("/api/metrics")
    async def metrics() -> dict:
        uptime_seconds = int((datetime.now().astimezone() - runtime.started_at).total_seconds()) if runtime.started_at else 0
        active_workers = sum(1 for task in runtime._workers if not task.done())
        rejected_count = runtime.message_store.count_queue_rejections()
        return {
            "uptime_seconds": uptime_seconds,
            "queue": {
                "size": runtime.queue.qsize(),
                "capacity": runtime.settings.queue_size,
                "enqueue_wait_seconds": runtime.settings.queue_enqueue_wait_seconds,
            },
            "workers": {
                "configured": runtime.settings.worker_count,
                "active": active_workers,
            },
            "messages": {
                "recent": len(runtime.recent_messages),
                "queue_rejections": rejected_count,
            },
            "plugins": {
                "loaded": len(runtime.manager.plugins),
                "enabled": len(runtime.settings.plugins),
            },
            "plugin_logs": {
                "recent": len(runtime.recent_plugin_logs),
            },
            "connectivity": {
                "wxrobot_api_reachable": runtime.wxrobot_api_reachable,
                "heartbeat_healthy": runtime.heartbeat_healthy,
            },
        }

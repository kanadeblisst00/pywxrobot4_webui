from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import AppSettings, PROJECT_DIR
from .model_client import OpenAICompatibleClient
from .pipeline import SummaryPipeline
from .repository import Repository
from .schemas import (
    ModelProfile,
    ModelProfileInput,
    ModelProfileUpdate,
    PipelineSettings,
    SummaryCreate,
    SummaryRecord,
)


async def run_summary_job(
    repository: Repository,
    summary_id: str,
    request: SummaryCreate,
    profile: dict[str, Any],
    pipeline_settings: dict[str, Any],
    timeout_seconds: float,
) -> None:
    repository.update_summary(summary_id, status="running")
    try:
        client = OpenAICompatibleClient(profile, timeout_seconds=timeout_seconds)
        pipeline = SummaryPipeline(client)
        result = await pipeline.run(
            room_name=request.room_name,
            messages=request.messages,
            settings=pipeline_settings,
            custom_instruction=request.custom_instruction or "",
        )
        repository.update_summary(summary_id, status="completed", result=result.model_dump(mode="json"))
    except Exception as exc:
        repository.update_summary(summary_id, status="failed", error=str(exc))


def create_app(
    *,
    settings: AppSettings | None = None,
    repository: Repository | None = None,
) -> FastAPI:
    app_settings = settings or AppSettings()
    app_repository = repository or Repository(app_settings.database_path)
    static_dir = PROJECT_DIR / "static"

    app = FastAPI(
        title="群聊消息总结服务",
        description="独立部署的群聊分段总结 API 与模型管理控制台",
        version=__version__,
    )
    app.state.repository = app_repository
    app.state.settings = app_settings

    if app_settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(app_settings.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "group-chat-summary", "version": __version__}

    @app.get("/api/v1/dashboard")
    async def dashboard() -> dict[str, Any]:
        return app_repository.dashboard_stats()

    @app.get("/api/v1/model-profiles", response_model=list[ModelProfile])
    async def list_model_profiles() -> list[dict[str, Any]]:
        return app_repository.list_model_profiles()

    @app.post("/api/v1/model-profiles", response_model=ModelProfile, status_code=201)
    async def create_model_profile(payload: ModelProfileInput) -> dict[str, Any]:
        return app_repository.create_model_profile(payload.model_dump())

    @app.put("/api/v1/model-profiles/{profile_id}", response_model=ModelProfile)
    async def update_model_profile(profile_id: str, payload: ModelProfileUpdate) -> dict[str, Any]:
        profile = app_repository.update_model_profile(profile_id, payload.model_dump(exclude_unset=True))
        if not profile:
            raise HTTPException(status_code=404, detail="模型配置不存在")
        return profile

    @app.delete("/api/v1/model-profiles/{profile_id}", status_code=204)
    async def delete_model_profile(profile_id: str) -> None:
        profile = app_repository.get_model_profile(profile_id)
        if not profile:
            raise HTTPException(status_code=404, detail="模型配置不存在")
        if profile["is_builtin"]:
            raise HTTPException(status_code=409, detail="内置模型不能删除，可以将其停用")
        if not app_repository.delete_model_profile(profile_id):
            raise HTTPException(status_code=404, detail="模型配置不存在")

    @app.post("/api/v1/model-profiles/{profile_id}/test")
    async def test_model_profile(profile_id: str) -> dict[str, Any]:
        profile = app_repository.get_model_profile(profile_id, reveal_secret=True)
        if not profile:
            raise HTTPException(status_code=404, detail="模型配置不存在")
        client = OpenAICompatibleClient(profile, timeout_seconds=app_settings.request_timeout_seconds)
        return await client.check_connection()

    @app.get("/api/v1/settings", response_model=PipelineSettings)
    async def get_pipeline_settings() -> dict[str, Any]:
        return app_repository.get_settings()

    @app.put("/api/v1/settings", response_model=PipelineSettings)
    async def update_pipeline_settings(payload: PipelineSettings) -> dict[str, Any]:
        profile = app_repository.get_model_profile(payload.default_model_profile_id)
        if not profile:
            raise HTTPException(status_code=422, detail="默认模型配置不存在")
        return app_repository.update_settings(payload.model_dump())

    @app.post("/api/v1/summaries", response_model=SummaryRecord, status_code=202)
    async def create_summary(payload: SummaryCreate, background_tasks: BackgroundTasks) -> dict[str, Any]:
        pipeline_settings = app_repository.get_settings()
        profile_id = payload.model_profile_id or pipeline_settings["default_model_profile_id"]
        profile = app_repository.get_model_profile(profile_id, reveal_secret=True)
        if not profile:
            raise HTTPException(status_code=422, detail="指定的模型配置不存在")
        if not profile["enabled"]:
            raise HTTPException(status_code=422, detail="指定的模型配置已停用")
        record = app_repository.create_summary(
            payload.model_dump(mode="json"),
            profile_id,
            keep_raw=bool(pipeline_settings.get("keep_raw_messages", True)),
        )
        background_tasks.add_task(
            run_summary_job,
            app_repository,
            record["id"],
            payload,
            profile,
            pipeline_settings,
            app_settings.request_timeout_seconds,
        )
        return record

    @app.get("/api/v1/summaries", response_model=list[SummaryRecord])
    async def list_summaries(limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
        return app_repository.list_summaries(limit)

    @app.get("/api/v1/summaries/{summary_id}")
    async def get_summary(summary_id: str, include_request: bool = False) -> dict[str, Any]:
        record = app_repository.get_summary(summary_id, include_request=include_request)
        if not record:
            raise HTTPException(status_code=404, detail="摘要任务不存在")
        return record

    @app.delete("/api/v1/summaries/{summary_id}", status_code=204)
    async def delete_summary(summary_id: str) -> None:
        if not app_repository.delete_summary(summary_id):
            raise HTTPException(status_code=404, detail="摘要任务不存在")

    return app


app = create_app()

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import AppConfig, PROJECT_DIR
from .dataset_catalog import DATASET_CATALOG
from .datasets import DatasetManager
from .db import Database
from .engine import MonitorEngine
from .model_catalog import MODEL_CATALOG
from .repository import Repository
from .schemas import (
    BatchTestInput,
    DatasetImportOptions,
    FeedbackInput,
    MessageInput,
    ModelProbeInput,
    ModelSettings,
    PaginatedEvents,
    TopicInput,
    TrainingRequest,
)
from .training import TrainingService
from .vision import VisionRuntime


DEFAULT_TOPICS = [
    TopicInput(
        name="资金借贷请求",
        description="主动借钱、寻求贷款、垫付或短期资金周转；排除反诈提醒、新闻讨论和还款陈述。",
        severity="high",
        keywords=["借钱", "贷款", "资金周转", "先垫一下"],
        regex_patterns=[r"(?:支援|借|垫)(?:我)?\s*[0-9一二两三四五六七八九十百千万]+(?:元|块|千|万)?"],
        exclude_patterns=[r"(?:反诈|谨防|小心).{0,8}(?:借钱|贷款|转账)"],
        examples=[
            {"text": "最近手头有点紧，谁方便先帮我垫一下", "polarity": "positive"},
            {"text": "有没有能短期周转的渠道", "polarity": "positive"},
            {"text": "能不能先支援两千，下周还你", "polarity": "positive"},
            {"text": "千万别找群里的人借钱", "polarity": "negative"},
            {"text": "银行贷款利率又降了", "polarity": "negative"},
            {"text": "我已经把之前借的钱还了", "polarity": "negative"},
        ],
        semantic_threshold=0.66,
        review_threshold=0.46,
    ),
    TopicInput(
        name="色情",
        description="色情资源、招嫖、裸聊及成人内容推广。医学和科普语境应作为负样本处理。",
        severity="critical",
        keywords=["成人视频", "黄色视频", "裸聊", "上门服务", "特殊服务", "成人网站", "福利视频"],
        regex_patterns=[r"(?:色情|成人|裸聊).{0,12}(?:资源|视频|网站|服务|直播)"],
        exclude_patterns=[r"(?:医院|医学|科普|治理|举报).{0,10}(?:色情|性教育)"],
        examples=[
            {"text": "成人资源合集，想看的私聊", "polarity": "positive"},
            {"text": "学校开展青春期健康教育", "polarity": "negative"},
        ],
        semantic_threshold=0.72,
        review_threshold=0.46,
    ),
    TopicInput(
        name="赌博",
        description="现金或可提现资产相关的下注、博彩、上分、盘口和代理招募。",
        severity="critical",
        keywords=["在线博彩", "下注赔率", "棋牌上分", "充值送彩金", "百家乐", "彩票计划", "网赌平台", "盘口开放"],
        regex_patterns=[r"(?:下注|博彩|赌博|上分|盘口|百家乐).{0,14}(?:提现|充值|彩金|返佣|代理|赔率)"],
        exclude_patterns=[r"(?:新闻|治理|举报|禁止|远离).{0,10}(?:赌博|博彩|网赌)"],
        examples=[
            {"text": "平台稳定上分，充值送彩金", "polarity": "positive"},
            {"text": "新闻报道打击网络赌博", "polarity": "negative"},
        ],
        semantic_threshold=0.72,
        review_threshold=0.45,
    ),
    TopicInput(
        name="广告",
        description="群内商业推广、引流、刷单兼职及重复联系方式；单一联系方式默认只进入复核。",
        severity="high",
        keywords=["加微信", "扫码进群", "联系QQ", "私聊了解", "厂家直销", "代理招募", "日结兼职", "免费领取"],
        regex_patterns=[r"(?:加|添加|联系).{0,3}(?:微信|QQ|V信).{0,10}[a-zA-Z0-9_-]{5,}", r"https?://[^\s]+"],
        exclude_patterns=[r"(?:管理员通知|禁止|不要发送|谨防).{0,10}(?:广告|链接|二维码)"],
        examples=[
            {"text": "兼职日结，加微信了解详情", "polarity": "positive"},
            {"text": "这是活动组织人的联系方式", "polarity": "negative"},
        ],
        semantic_threshold=0.82,
        review_threshold=0.48,
    ),
]


def _seed(repository: Repository, datasets: DatasetManager) -> None:
    existing_topic_names = {topic.name for topic in repository.list_topics()}
    for topic in DEFAULT_TOPICS:
        if topic.name not in existing_topic_names:
            repository.create_topic(topic)
    if not any(item.slug == "builtin-wechat-moderation-seed" for item in repository.list_datasets()):
        datasets.load_builtin()


def create_app(config: AppConfig | None = None) -> FastAPI:
    settings = config or AppConfig.from_env()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifacts_dir.mkdir(parents=True, exist_ok=True)
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    database = Database(settings.database_path)
    repository = Repository(database)
    datasets = DatasetManager(repository, PROJECT_DIR)
    engine = MonitorEngine(repository, settings)
    trainer = TrainingService(repository, settings.artifacts_dir)
    vision = VisionRuntime(repository, engine)
    _seed(repository, datasets)

    app = FastAPI(
        title="群聊内容审核中心",
        description="本地优先的中文群聊内容审核、数据集管理与小模型训练服务。",
        version="0.2.0",
    )
    app.state.config = settings
    app.state.repository = repository
    app.state.datasets = datasets
    app.state.engine = engine
    app.state.trainer = trainer
    app.state.vision = vision

    async def authorize(x_api_token: str = Header(default="")) -> None:
        if settings.api_token and x_api_token != settings.api_token:
            raise HTTPException(status_code=401, detail="API Token 无效")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": app.version, "database": str(settings.database_path)}

    @app.get("/api/v1/dashboard", dependencies=[])
    @app.get("/api/v1/stats", include_in_schema=False)
    async def dashboard(_: None = Header(default=None, alias="X-Ignored"), x_api_token: str = Header(default="")) -> dict[str, Any]:
        await authorize(x_api_token)
        return repository.stats()

    @app.post("/api/v1/analyze")
    @app.post("/api/v1/messages/analyze", include_in_schema=False)
    @app.post("/api/v1/messages", include_in_schema=False)
    async def analyze(value: MessageInput, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        return await engine.analyze(value)

    @app.get("/api/v1/vision/status")
    async def vision_status(x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        return vision.status()

    @app.post("/api/v1/images/analyze")
    async def analyze_image(
        file: UploadFile = File(...),
        message_id: str = Form(default=""),
        room_id: str = Form(default="image-test"),
        sender_id: str = Form(default=""),
        sender_name: str = Form(default=""),
        context_json: str = Form(default="[]"),
        persist: bool = Form(default=True),
        x_api_token: str = Header(default=""),
    ):
        await authorize(x_api_token)
        content = await file.read()
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="图片文件不能超过 20 MB")
        try:
            context = json.loads(context_json)
            if not isinstance(context, list) or not all(isinstance(item, str) for item in context):
                raise ValueError("context_json 必须是字符串数组")
            return await vision.analyze(
                content=content,
                filename=file.filename or "image",
                message_id=message_id,
                room_id=room_id,
                sender_id=sender_id,
                sender_name=sender_name,
                context=context[:50],
                persist=persist,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/test/batch")
    async def batch_test(value: BatchTestInput, x_api_token: str = Header(default="")) -> dict[str, Any]:
        await authorize(x_api_token)
        results = []
        exact = 0
        for index, item in enumerate(value.items):
            response = await engine.analyze(
                MessageInput(room_id="__batch_test__", message_id=str(index), text=item.text, persist=False)
            )
            predicted = sorted({match.topic_name for match in response.matches if match.matched})
            expected = sorted(item.expected)
            exact += int(predicted == expected)
            results.append({"text": item.text, "expected": expected, "predicted": predicted, "response": response})
        return {"total": len(results), "exact_match": round(exact / len(results), 4), "items": results}

    @app.get("/api/v1/topics")
    async def list_topics(x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        return repository.list_topics()

    @app.post("/api/v1/topics", status_code=201)
    async def create_topic(value: TopicInput, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        try:
            return repository.create_topic(value)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.put("/api/v1/topics/{topic_id}")
    async def update_topic(topic_id: int, value: TopicInput, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        topic = repository.update_topic(topic_id, value)
        if not topic:
            raise HTTPException(status_code=404, detail="审核主题不存在")
        engine._embedding_cache.clear()
        return topic

    @app.delete("/api/v1/topics/{topic_id}")
    async def delete_topic(topic_id: int, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        if not repository.delete_topic(topic_id):
            raise HTTPException(status_code=404, detail="审核主题不存在")
        return {"deleted": True}

    @app.get("/api/v1/models/catalog")
    async def model_catalog(x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        return MODEL_CATALOG

    @app.get("/api/v1/models/settings")
    async def get_model_settings(x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        return repository.get_model_settings(mask_secrets=True)

    @app.put("/api/v1/models/settings")
    async def save_model_settings(value: ModelSettings, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        result = repository.save_model_settings(value)
        engine.invalidate_models()
        return result

    @app.post("/api/v1/models/activate/{run_id}")
    async def activate_model(run_id: int, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        run = repository.get_training_run(run_id)
        if not run or run.status != "completed" or not run.artifact_path:
            raise HTTPException(status_code=409, detail="训练任务尚未生成可用模型")
        current = repository.get_model_settings()
        payload = current.model_dump()
        payload.update({"profile": "eco", "classifier_provider": "builtin_nb", "classifier_model": "builtin/char-ngram-nb", "classifier_artifact": run.artifact_path})
        result = repository.save_model_settings(ModelSettings.model_validate(payload))
        engine.invalidate_models()
        return result

    @app.post("/api/v1/models/probe")
    async def probe_model(value: ModelProbeInput, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        return await engine.probe(value.kind)

    @app.get("/api/v1/datasets/catalog")
    async def dataset_catalog(x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        return datasets.catalog()

    @app.get("/api/v1/datasets")
    async def list_datasets(x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        return repository.list_datasets()

    @app.post("/api/v1/datasets/{slug}/load")
    async def load_dataset(slug: str, accepted: bool = Query(default=False), x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        try:
            if slug == "builtin-wechat-moderation-seed":
                return datasets.load_builtin()
            if slug == "fbs-sms-dataset":
                return await datasets.load_fbs(accepted)
            entry = next((item for item in DATASET_CATALOG if item["slug"] == slug), None)
            if not entry:
                raise HTTPException(status_code=404, detail="数据集不存在")
            raise HTTPException(status_code=409, detail="该数据集需要先按来源页面要求下载，再使用导入功能加载")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/datasets/import")
    async def import_dataset(
        file: UploadFile = File(...),
        options_json: str = Form(...),
        x_api_token: str = Header(default=""),
    ):
        await authorize(x_api_token)
        content = await file.read()
        if len(content) > 100 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="数据文件不能超过 100 MB")
        try:
            options = DatasetImportOptions.model_validate_json(options_json)
            return datasets.import_bytes(file.filename or "dataset.jsonl", content, options)
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/api/v1/datasets/{dataset_id}")
    async def delete_dataset(dataset_id: int, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        if not repository.delete_dataset(dataset_id):
            raise HTTPException(status_code=404, detail="数据集不存在")
        return {"deleted": True}

    @app.get("/api/v1/samples")
    async def list_samples(
        dataset_id: int | None = None,
        split: str | None = None,
        limit: int = Query(default=100, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
        x_api_token: str = Header(default=""),
    ):
        await authorize(x_api_token)
        items, total = repository.list_samples([dataset_id] if dataset_id else None, split=split, limit=limit, offset=offset)
        return {"items": items, "total": total, "limit": limit, "offset": offset}

    @app.post("/api/v1/training/runs", status_code=202)
    async def create_training_run(value: TrainingRequest, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        missing = [dataset_id for dataset_id in value.dataset_ids if not repository.get_dataset(dataset_id)]
        if missing:
            raise HTTPException(status_code=400, detail=f"数据集不存在：{missing}")
        run_id = trainer.schedule(value)
        return {"run_id": run_id, "status": "queued"}

    @app.get("/api/v1/training/runs")
    async def list_training_runs(x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        return repository.list_training_runs()

    @app.get("/api/v1/training/runs/{run_id}")
    async def get_training_run(run_id: int, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        run = repository.get_training_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        return run

    @app.get("/api/v1/events", response_model=PaginatedEvents)
    async def list_events(
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=30, ge=1, le=200),
        topic_id: int | None = None,
        feedback: str | None = None,
        room_id: str | None = None,
        x_api_token: str = Header(default=""),
    ):
        await authorize(x_api_token)
        items, total = repository.list_events(page=page, page_size=page_size, topic_id=topic_id, feedback=feedback, room_id=room_id)
        return PaginatedEvents(items=items, total=total, page=page, page_size=page_size)

    @app.post("/api/v1/events/{event_id}/feedback")
    async def save_feedback(event_id: int, value: FeedbackInput, x_api_token: str = Header(default="")):
        await authorize(x_api_token)
        event = repository.save_feedback(event_id, value)
        if not event:
            raise HTTPException(status_code=404, detail="审核事件不存在")
        return event

    if settings.web_dir.exists():
        app.mount("/assets", StaticFiles(directory=settings.web_dir), name="assets")

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(settings.web_dir / "index.html")

    return app


app = create_app()

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .classifier import evaluate_classifier, split_samples, train_char_ngram_nb
from .repository import Repository
from .schemas import TrainingRequest


class TrainingService:
    def __init__(self, repository: Repository, artifacts_dir: Path):
        self.repository = repository
        self.artifacts_dir = artifacts_dir
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._tasks: set[asyncio.Task[Any]] = set()

    def schedule(self, request: TrainingRequest) -> int:
        run = self.repository.create_training_run(request)
        task = asyncio.create_task(asyncio.to_thread(self._run, run.id, request))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return run.id

    def _run(self, run_id: int, request: TrainingRequest) -> None:
        self.repository.update_training_run(run_id, status="running", started_at=datetime.now().isoformat(timespec="seconds"))
        try:
            samples = self.repository.all_training_samples(request.dataset_ids)
            if len(samples) < 12:
                raise ValueError("训练样本不足 12 条，请先加载或导入数据集")
            train_samples, test_samples = split_samples(samples, request.test_ratio, request.seed)
            config = request.model_dump()
            model = train_char_ngram_nb(train_samples, config)
            metrics = evaluate_classifier(model, test_samples)
            metrics.update(
                {
                    "train_samples": len(train_samples),
                    "labels": model.labels,
                    "vocabulary_size": model.artifact["vocabulary_size"],
                    "threshold": model.threshold,
                }
            )
            artifact_path = self.artifacts_dir / f"char_ngram_nb_run_{run_id}.json.gz"
            model.save(artifact_path)
            self.repository.update_training_run(
                run_id,
                status="completed",
                metrics_json=json.dumps(metrics, ensure_ascii=False, separators=(",", ":")),
                artifact_path=str(artifact_path.resolve()),
                sample_count=len(samples),
                completed_at=datetime.now().isoformat(timespec="seconds"),
            )
        except Exception as exc:
            self.repository.update_training_run(
                run_id,
                status="failed",
                error=str(exc),
                completed_at=datetime.now().isoformat(timespec="seconds"),
            )

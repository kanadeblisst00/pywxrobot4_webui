from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import httpx

from .classifier import normalize_label
from .dataset_catalog import DATASET_CATALOG
from .normalizer import normalize_text
from .repository import Repository
from .schemas import DatasetImportOptions


FBS_LABELS = {
    "AD:Loan": ["advertising"],
    "AD:Network_service": ["advertising"],
    "AD:Other": ["advertising"],
    "AD:Real_estate": ["advertising"],
    "AD:Retail": ["advertising"],
    "FR:Financial": ["advertising"],
    "FR:Other": ["advertising"],
    "FR:Phishing(Bank)": ["advertising"],
    "FR:Phishing(Other)": ["advertising"],
    "IL:Escort_service": ["porn", "advertising"],
    "IL:Fake_ID_and_invoice": ["advertising"],
    "IL:Gambling": ["gambling", "advertising"],
    "IL:Political_propaganda": ["advertising"],
}


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _labels(value: Any, mapping: dict[str, list[str]], default: str) -> list[str]:
    if isinstance(value, list):
        raw = [str(item).strip() for item in value]
    else:
        raw = [item.strip() for item in re.split(r"[|,，;/]", str(value or "")) if item.strip()]
    resolved: list[str] = []
    for item in raw or [default]:
        candidates = mapping.get(item, [normalize_label(item)])
        for candidate in candidates:
            normalized = normalize_label(candidate)
            if normalized and normalized not in resolved:
                resolved.append(normalized)
    return resolved or [normalize_label(default)]


def make_sample(text: str, labels: list[str], *, split: str = "train", source_ref: str = "", metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    clean = normalize_text(text)
    if len(clean) < 2:
        return None
    fingerprint = hashlib.sha256((clean + "\0" + "|".join(sorted(labels))).encode("utf-8")).hexdigest()
    return {
        "text": clean,
        "normalized_text": clean,
        "labels": labels,
        "split": split if split in {"train", "test", "validation", "dev"} else "train",
        "source_ref": source_ref,
        "metadata": metadata or {},
        "fingerprint": fingerprint,
    }


class DatasetManager:
    def __init__(self, repository: Repository, project_dir: Path):
        self.repository = repository
        self.project_dir = project_dir

    def catalog(self) -> list[dict[str, Any]]:
        loaded = {(item.slug, item.version): item for item in self.repository.list_datasets()}
        result: list[dict[str, Any]] = []
        for item in DATASET_CATALOG:
            current = loaded.get((item["slug"], item["version"]))
            result.append({**item, "loaded": bool(current), "sample_count": current.sample_count if current else 0, "dataset_id": current.id if current else None})
        return result

    def load_builtin(self) -> dict[str, Any]:
        item = next(value for value in DATASET_CATALOG if value["slug"] == "builtin-wechat-moderation-seed")
        path = self.project_dir / "datasets" / "builtin_seed.jsonl"
        return self.import_bytes(
            path.name,
            path.read_bytes(),
            DatasetImportOptions(
                name=item["name"], slug=item["slug"], version=item["version"], license=item["license"],
                text_column="text", label_column="labels", split_column="split",
            ),
            source_url=item["homepage"],
        )

    async def load_fbs(self, accepted: bool) -> dict[str, Any]:
        if not accepted:
            raise ValueError("必须先确认已阅读数据集作者声明和引用要求")
        item = next(value for value in DATASET_CATALOG if value["slug"] == "fbs-sms-dataset")
        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            response = await client.get(item["download_url"])
            response.raise_for_status()
        samples: list[dict[str, Any]] = []
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            for name in archive.namelist():
                if name.endswith("/") or Path(name).name.lower().startswith("readme"):
                    continue
                category = next((key for key in FBS_LABELS if f"/{key}/" in f"/{name}" or f"\\{key}\\" in name), "")
                if not category:
                    continue
                content = _decode(archive.read(name))
                candidates = [line.strip() for line in content.splitlines() if line.strip()]
                if len(candidates) <= 1 and content.strip():
                    candidates = [content.strip()]
                for index, text in enumerate(candidates):
                    sample = make_sample(text, FBS_LABELS[category], source_ref=f"{name}:{index + 1}", metadata={"category": category})
                    if sample:
                        samples.append(sample)
        dataset = self.repository.upsert_dataset(
            slug=item["slug"], name=item["name"], version=item["version"], license_name=item["license"],
            source_url=item["homepage"], labels=item["labels"], metadata={"accepted_terms": True},
        )
        inserted, skipped = self.repository.add_samples(dataset.id, samples)
        return {"dataset": self.repository.get_dataset(dataset.id), "inserted": inserted, "skipped": skipped}

    def import_bytes(self, filename: str, content: bytes, options: DatasetImportOptions, *, source_url: str = "") -> dict[str, Any]:
        suffix = Path(filename).suffix.lower()
        records: list[dict[str, Any]] = []
        text = _decode(content)
        if suffix == ".csv":
            records = [dict(row) for row in csv.DictReader(io.StringIO(text))]
        elif suffix in {".jsonl", ".ndjson"}:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        elif suffix == ".json":
            payload = json.loads(text)
            records = payload if isinstance(payload, list) else payload.get("data", payload.get("items", []))
        elif suffix == ".txt":
            records = [{options.text_column: line} for line in text.splitlines() if line.strip()]
        else:
            raise ValueError("仅支持 CSV、JSON、JSONL、NDJSON 和 TXT")
        if not isinstance(records, list):
            raise ValueError("数据文件顶层必须是记录数组")
        samples: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                continue
            raw_text = record.get(options.text_column, "")
            labels = _labels(record.get(options.label_column), options.label_mapping, options.default_label)
            split = str(record.get(options.split_column, "train")).strip().lower()
            sample = make_sample(str(raw_text), labels, split=split, source_ref=f"{filename}:{index + 1}")
            if sample:
                samples.append(sample)
        dataset = self.repository.upsert_dataset(
            slug=options.slug, name=options.name, version=options.version, license_name=options.license,
            source_url=source_url, labels=sorted({label for sample in samples for label in sample["labels"]}),
            metadata={"filename": filename, "import_options": options.model_dump()},
        )
        inserted, skipped = self.repository.add_samples(dataset.id, samples)
        return {"dataset": self.repository.get_dataset(dataset.id), "inserted": inserted, "skipped": skipped}

from __future__ import annotations

import gzip
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import httpx

from .normalizer import compact_for_rule, normalize_text


LABEL_ALIASES = {
    "porn": "porn",
    "色情": "porn",
    "淫秽色情": "porn",
    "sexual": "porn",
    "gambling": "gambling",
    "赌博": "gambling",
    "advertising": "advertising",
    "广告": "advertising",
    "spam": "advertising",
    "垃圾信息": "advertising",
    "normal": "normal",
    "正常": "normal",
    "不违规": "normal",
}


def normalize_label(value: str) -> str:
    text = str(value or "").strip().lower()
    return LABEL_ALIASES.get(text, text)


def char_ngrams(text: str, minimum: int = 1, maximum: int = 3) -> list[str]:
    value = compact_for_rule(text)
    if not value:
        return []
    tokens: list[str] = []
    for size in range(minimum, maximum + 1):
        if len(value) < size:
            continue
        tokens.extend(value[index : index + size] for index in range(len(value) - size + 1))
    return tokens


class CharNGramNB:
    """无第三方 ML 依赖的多标签字符 n-gram 朴素贝叶斯分类器。"""

    def __init__(self, artifact: dict[str, Any]):
        self.artifact = artifact
        self.labels = artifact.get("labels", [])
        self.minimum = int(artifact.get("min_ngram", 1))
        self.maximum = int(artifact.get("max_ngram", 3))
        self.threshold = float(artifact.get("threshold", 0.52))

    @classmethod
    def load(cls, path: Path) -> "CharNGramNB":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(self.artifact, handle, ensure_ascii=False, separators=(",", ":"))

    def predict_scores(self, text: str) -> dict[str, float]:
        counts = Counter(char_ngrams(text, self.minimum, self.maximum))
        scores: dict[str, float] = {}
        for label in self.labels:
            model = self.artifact["models"][label]
            log_odds = float(model["prior_log_odds"])
            weights = model["weights"]
            for token, count in counts.items():
                # 训练词表之外的字符不携带类别信息，不能使用类别分母惩罚。
                # 否则较短类别语料会让任意新文本获得极端正分。
                if token in weights:
                    log_odds += count * float(weights[token])
            log_odds = max(-30.0, min(30.0, log_odds))
            scores[label] = round(1.0 / (1.0 + math.exp(-log_odds)), 6)
        return scores

    def predict(self, text: str) -> list[str]:
        return [label for label, score in self.predict_scores(text).items() if score >= self.threshold]


def train_char_ngram_nb(samples: list[dict[str, Any]], config: dict[str, Any]) -> CharNGramNB:
    minimum = int(config.get("min_ngram", 1))
    maximum = int(config.get("max_ngram", 3))
    min_df = int(config.get("min_df", 2))
    alpha = float(config.get("alpha", 1.0))
    labels = sorted(
        {
            normalize_label(label)
            for sample in samples
            for label in sample.get("labels", [])
            if normalize_label(label) not in {"", "normal"}
        }
    )
    document_frequency: Counter[str] = Counter()
    tokenized: list[tuple[Counter[str], set[str]]] = []
    for sample in samples:
        counts = Counter(char_ngrams(sample["text"], minimum, maximum))
        document_frequency.update(counts.keys())
        tokenized.append((counts, {normalize_label(item) for item in sample.get("labels", [])}))
    vocabulary = {token for token, count in document_frequency.items() if count >= min_df}
    if not vocabulary:
        raise ValueError("有效词表为空，请增加样本或降低 min_df")

    models: dict[str, Any] = {}
    total_documents = len(samples)
    for label in labels:
        positive_counts: Counter[str] = Counter()
        negative_counts: Counter[str] = Counter()
        positive_documents = 0
        for counts, sample_labels in tokenized:
            target = positive_counts if label in sample_labels else negative_counts
            if label in sample_labels:
                positive_documents += 1
            target.update({token: count for token, count in counts.items() if token in vocabulary})
        negative_documents = total_documents - positive_documents
        positive_total = sum(positive_counts.values())
        negative_total = sum(negative_counts.values())
        vocabulary_size = len(vocabulary)
        positive_denominator = positive_total + alpha * vocabulary_size
        negative_denominator = negative_total + alpha * vocabulary_size
        weights = {
            token: round(
                math.log((positive_counts[token] + alpha) / positive_denominator)
                - math.log((negative_counts[token] + alpha) / negative_denominator),
                7,
            )
            for token in vocabulary
            if positive_counts[token] or negative_counts[token]
        }
        prior_log_odds = math.log((positive_documents + alpha) / (negative_documents + alpha))
        models[label] = {
            "prior_log_odds": round(prior_log_odds, 7),
            "unknown_weight": 0.0,
            "positive_documents": positive_documents,
            "negative_documents": negative_documents,
            "weights": weights,
        }
    artifact = {
        "format": "semantic-monitor-char-ngram-nb-v1",
        "labels": labels,
        "min_ngram": minimum,
        "max_ngram": maximum,
        "min_df": min_df,
        "alpha": alpha,
        "threshold": float(config.get("threshold", 0.52)),
        "vocabulary_size": len(vocabulary),
        "sample_count": total_documents,
        "models": models,
    }
    return CharNGramNB(artifact)


def split_samples(samples: list[dict[str, Any]], test_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    explicit_test = [item for item in samples if item.get("split") in {"test", "validation", "dev"}]
    train_pool = [item for item in samples if item not in explicit_test]
    if explicit_test and train_pool:
        return train_pool, explicit_test
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    test_size = max(1, min(len(shuffled) - 1, round(len(shuffled) * test_ratio)))
    return shuffled[test_size:], shuffled[:test_size]


def evaluate_classifier(model: CharNGramNB, samples: list[dict[str, Any]]) -> dict[str, Any]:
    labels = model.labels
    counts = {label: defaultdict(int) for label in labels}
    exact = 0
    for sample in samples:
        expected = {normalize_label(item) for item in sample.get("labels", [])} - {"normal", ""}
        predicted = set(model.predict(sample["text"]))
        exact += int(expected == predicted)
        for label in labels:
            if label in expected and label in predicted:
                counts[label]["tp"] += 1
            elif label not in expected and label in predicted:
                counts[label]["fp"] += 1
            elif label in expected and label not in predicted:
                counts[label]["fn"] += 1
            else:
                counts[label]["tn"] += 1
    per_label: dict[str, Any] = {}
    macro_f1 = 0.0
    totals = defaultdict(int)
    for label, value in counts.items():
        precision = value["tp"] / max(1, value["tp"] + value["fp"])
        recall = value["tp"] / max(1, value["tp"] + value["fn"])
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        macro_f1 += f1
        for key in ("tp", "fp", "fn", "tn"):
            totals[key] += value[key]
        per_label[label] = {
            **dict(value), "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)
        }
    micro_precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    micro_recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    micro_f1 = 2 * micro_precision * micro_recall / max(1e-12, micro_precision + micro_recall)
    return {
        "test_samples": len(samples),
        "exact_match": round(exact / max(1, len(samples)), 4),
        "micro_f1": round(micro_f1, 4),
        "macro_f1": round(macro_f1 / max(1, len(labels)), 4),
        "per_label": per_label,
    }


class ClassifierRuntime:
    def __init__(self):
        self._artifact_path = ""
        self._builtin_model: CharNGramNB | None = None
        self._transformers_key = ""
        self._transformers: tuple[Any, Any] | None = None

    def _load_artifact(self, path: str) -> CharNGramNB | None:
        if not path:
            return None
        resolved = str(Path(path).resolve())
        if resolved != self._artifact_path:
            self._builtin_model = CharNGramNB.load(Path(resolved))
            self._artifact_path = resolved
        return self._builtin_model

    async def classify(self, text: str, settings: Any) -> tuple[dict[str, float], str]:
        provider = settings.classifier_provider
        if provider in {"disabled", "heuristic"}:
            return {}, provider
        if provider == "builtin_nb":
            model = self._load_artifact(settings.classifier_artifact)
            return (model.predict_scores(text), "builtin_nb") if model else ({}, "builtin_nb:no-artifact")
        if provider == "openai_compatible":
            return await self._classify_openai_compatible(text, settings)
        if provider == "transformers":
            return self._classify_transformers(text, settings)
        return {}, f"unsupported:{provider}"

    async def _classify_openai_compatible(self, text: str, settings: Any) -> tuple[dict[str, float], str]:
        if not settings.classifier_api_base:
            return {}, "openai_compatible:no-base"
        headers = {"Content-Type": "application/json"}
        if settings.classifier_api_key:
            headers["Authorization"] = f"Bearer {settings.classifier_api_key}"
        payload = {
            "model": settings.classifier_model,
            "temperature": 0,
            "max_tokens": settings.max_new_tokens,
            "messages": [{"role": "user", "content": text}],
        }
        async with httpx.AsyncClient(timeout=settings.request_timeout_seconds) as client:
            response = await client.post(f"{settings.classifier_api_base}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        return self._parse_guard_output(content, text), "openai_compatible"

    def _classify_transformers(self, text: str, settings: Any) -> tuple[dict[str, float], str]:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            return {}, "transformers:not-installed"
        key = settings.classifier_model
        if self._transformers is None or self._transformers_key != key:
            tokenizer = AutoTokenizer.from_pretrained(key)
            model = AutoModelForCausalLM.from_pretrained(key, device_map="auto", torch_dtype="auto")
            self._transformers = (tokenizer, model)
            self._transformers_key = key
        tokenizer, model = self._transformers
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=settings.max_new_tokens, do_sample=False)
        content = tokenizer.decode(output[0][len(inputs.input_ids[0]) :], skip_special_tokens=True)
        return self._parse_guard_output(content, text), "transformers"

    @staticmethod
    def _parse_guard_output(content: str, original_text: str) -> dict[str, float]:
        lowered = normalize_text(content).lower()
        if "unsafe" not in lowered and "controversial" not in lowered:
            return {}
        score = 0.88 if "unsafe" in lowered else 0.62
        scores: dict[str, float] = {}
        if "sexual content" in lowered or "sexual acts" in lowered:
            scores["porn"] = score
        gambling_cues = ("赌博", "下注", "博彩", "上分", "赔率", "彩票", "娱乐城")
        if "non-violent illegal" in lowered and any(item in original_text for item in gambling_cues):
            scores["gambling"] = score
        return scores

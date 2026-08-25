from __future__ import annotations

from app.classifier import evaluate_classifier, train_char_ngram_nb


def test_char_ngram_classifier_training_roundtrip() -> None:
    samples = []
    for index in range(10):
        samples.append({"text": f"博彩下注充值送彩金{index}", "labels": ["gambling"]})
        samples.append({"text": f"今天正常讨论项目进度{index}", "labels": ["normal"]})
    model = train_char_ngram_nb(samples, {"min_ngram": 1, "max_ngram": 3, "min_df": 1, "alpha": 1, "threshold": 0.5})
    assert model.predict_scores("博彩平台充值下注")["gambling"] > 0.5
    metrics = evaluate_classifier(model, samples)
    assert metrics["micro_f1"] >= 0.8

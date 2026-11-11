"""Пороги multilabel и метрики без обучения модели."""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def select_labels(probabilities, thresholds):
    """Порог включается в результат; пустая строка остаётся без категорий."""
    probabilities = np.asarray(probabilities)
    thresholds = np.asarray(thresholds)
    if probabilities.ndim != 2 or thresholds.shape != (probabilities.shape[1],):
        raise ValueError("Размерности оценок и порогов не совпадают")
    if not np.isfinite(probabilities).all() or not np.isfinite(thresholds).all():
        raise ValueError("Оценки и пороги должны быть конечными")
    if np.any((probabilities < 0) | (probabilities > 1)) or np.any((thresholds < 0) | (thresholds > 1)):
        raise ValueError("Оценки и пороги должны быть от 0 до 1")
    return (probabilities >= thresholds).astype(int)


def metrics(targets, predictions, labels):
    """Метрики по категориям, агрегаты и точное совпадение наборов."""
    if not len(targets):
        return {"rows": 0, "per_category": None, "micro": None, "macro": None, "exact_match": None}
    p, r, f, support = precision_recall_fscore_support(targets, predictions, average=None, zero_division=0)
    result = {"rows": len(targets), "per_category": {
        label: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i])}
        for i, label in enumerate(labels)}, "exact_match": float(accuracy_score(targets, predictions))}
    for average in ("micro", "macro"):
        p, r, f, _ = precision_recall_fscore_support(targets, predictions, average=average, zero_division=0)
        result[average] = {"precision": float(p), "recall": float(r), "f1": float(f)}
    return result


def quality_report(targets, probabilities, thresholds, labels):
    predictions = select_labels(probabilities, thresholds)
    counts = targets.sum(axis=1)
    return {"all": metrics(targets, predictions, labels),
            "singlelabel": metrics(targets[counts == 1], predictions[counts == 1], labels),
            "multilabel": metrics(targets[counts > 1], predictions[counts > 1], labels),
            "empty_predictions": int((predictions.sum(axis=1) == 0).sum())}


def tune_thresholds(targets, probabilities, labels, *, objective="f1", min_precision=0.95, min_support=10):
    """Настроить общий порог и затем частные пороги только на validation."""
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities)
    if targets.ndim != 2 or targets.shape != probabilities.shape or not len(targets) or targets.shape[1] != len(labels):
        raise ValueError("Нужны непустые совместимые цели и оценки validation")
    if not np.isin(targets, [0, 1]).all():
        raise ValueError("Цели должны быть multi-hot")
    select_labels(probabilities, np.full(len(labels), 0.5))
    if objective not in {"f1", "precision"} or not 0 < min_precision <= 1 or min_support < 1:
        raise ValueError("Неверная цель, precision или минимальная поддержка")

    def choose(y, scores):
        candidates = []
        for threshold in GRID:
            predicted = scores >= threshold
            tp = int((predicted & (y == 1)).sum())
            fp = int((predicted & (y == 0)).sum())
            fn = int((~predicted & (y == 1)).sum())
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
            if objective == "precision" and (not tp + fp or precision < min_precision):
                continue
            score = f1 if objective == "f1" else recall
            candidates.append(((score, -abs(float(threshold) - 0.5), float(threshold)), float(threshold)))
        return max(candidates)[1] if candidates else None

    common = choose(targets, probabilities)
    if common is None:
        raise ValueError("На сетке нет общего порога с требуемой precision; test не оценивается")
    thresholds, decisions = [], []
    for index, label in enumerate(labels):
        positive = int(targets[:, index].sum())
        negative = len(targets) - positive
        threshold = None
        reason = "low_support"
        if min(positive, negative) >= min_support:
            threshold = choose(targets[:, index], probabilities[:, index])
            reason = "per_category" if threshold is not None else "precision_unreachable"
        thresholds.append(common if threshold is None else threshold)
        decisions.append({"label": label, "positive": positive, "negative": negative, "rule": reason})
    return {"objective": objective, "min_precision": min_precision, "min_support": min_support,
            "comparison": ">=", "grid": GRID.tolist(), "common_threshold": common,
            "thresholds": thresholds, "decisions": decisions,
            "precision_constraint_guaranteed": False if objective == "precision" else None}

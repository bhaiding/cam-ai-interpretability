from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def safe_auroc(y, scores) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, scores))
    except Exception:
        return float("nan")


def select_validation_threshold(y, scores, metric: str = "accuracy") -> float:
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return 0.0
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201)))
    best_value, best_threshold = -np.inf, float(np.median(scores))
    for threshold in candidates:
        pred = (scores >= threshold).astype(np.int64)
        if metric == "accuracy":
            value = accuracy_score(y, pred)
        elif metric == "f1":
            _, _, value, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        else:
            raise ValueError(metric)
        if value > best_value:
            best_value, best_threshold = value, float(threshold)
    return best_threshold


def binary_metrics(y, scores, threshold: float):
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    pred = (scores >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": safe_auroc(y, scores),
        "threshold": float(threshold),
        "n": int(len(y)),
    }

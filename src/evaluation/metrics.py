from __future__ import annotations

from typing import Iterable


def sensitivity(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    true = list(y_true)
    pred = list(y_pred)
    tp = sum(1 for t, p in zip(true, pred) if t == 1 and p == 1)
    fn = sum(1 for t, p in zip(true, pred) if t == 1 and p == 0)
    return tp / (tp + fn) if (tp + fn) else 0.0


def specificity(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    true = list(y_true)
    pred = list(y_pred)
    tn = sum(1 for t, p in zip(true, pred) if t == 0 and p == 0)
    fp = sum(1 for t, p in zip(true, pred) if t == 0 and p == 1)
    return tn / (tn + fp) if (tn + fp) else 0.0


def auc_roc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    pairs = sorted(zip(y_score, y_true), key=lambda item: item[0])
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return 0.0

    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)

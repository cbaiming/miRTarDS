"""Metrics for the released P/U benchmark and fixed-threshold classifications."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def ranking_metrics(y, scores):
    """Compute class-balanced AP, ordinary AP, and ROC-AUC on binary labels."""
    y = np.asarray(y, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("Aligned, nonempty one-dimensional labels and scores are required")
    if not np.isin(y, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores must be finite")
    counts = np.bincount(y, minlength=2)
    if counts.min() <= 0:
        raise ValueError("Both P and U classes are required")
    weights = np.ones(len(y)) / (2 * counts[y])
    return {
        "balanced_aps": float(average_precision_score(y, scores, sample_weight=weights)),
        "raw_aps": float(average_precision_score(y, scores)),
        "auroc": float(roc_auc_score(y, scores)),
    }


def evaluate_scores(y, scores, threshold=0.5):
    """Evaluate continuous scores and classifications with P as the positive class.

    U is the operational negative label in this benchmark. The classification
    threshold is supplied by the caller and is never selected using these labels.
    """
    if not np.isfinite(threshold):
        raise ValueError("The classification threshold must be finite")
    result = ranking_metrics(y, scores)
    y = np.asarray(y, dtype=int)
    predicted = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    result.update({
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "threshold": float(threshold),
        "rows": int(len(y)),
        "positives": int(y.sum()),
    })
    return result


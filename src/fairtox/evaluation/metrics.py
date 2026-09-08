"""Global classification metrics.

Accuracy is computed and reported but never drives a decision: roughly nine in
ten comments are non-toxic, so a majority-class predictor scores about 90% while
detecting nothing at all. Early stopping and model selection key off macro-F1,
and the headline table leads with macro-F1, ROC-AUC and PR-AUC.
"""

from __future__ import annotations

from typing import Any

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


def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Threshold-dependent and threshold-free metrics for one split."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_true) != len(y_prob):
        raise ValueError(f"{len(y_true)} labels but {len(y_prob)} probabilities")
    y_pred = (y_prob >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "positive_rate_true": float(y_true.mean()) if len(y_true) else 0.0,
        "positive_rate_pred": float(y_pred.mean()) if len(y_pred) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # AUCs are undefined on a single-class split. Report None rather than crash:
    # it matters for tiny smoke runs and for small subgroup slices.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["roc_auc"] = None
        metrics["pr_auc"] = None

    metrics.update(confusion_counts(y_true, y_pred))
    metrics["degenerate"] = is_degenerate(y_pred)
    return metrics


def is_degenerate(y_pred: np.ndarray) -> bool:
    """True when the model predicts a single class for every input.

    A classifier that flags nothing has zero FPR in *every* subgroup, so a perfect
    gap, variance and EOD: read without this check, a dead model is the fairest
    model in the study. Heavy upweighting is exactly the pressure that produces
    one, and the alpha sweep pushes hard on purpose.
    """
    y_pred = np.asarray(y_pred)
    return bool(len(y_pred) > 0 and len(np.unique(y_pred)) == 1)


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """TN/FP/FN/TP plus the two error rates the fairness audit is built on."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        # None, not 0.0: "no non-toxic rows here" and "no false positives here"
        # are different facts and the audit must not confuse them.
        "fpr": fp / (fp + tn) if (fp + tn) else None,
        "fnr": fn / (fn + tp) if (fn + tp) else None,
    }


def summarise(metrics: dict[str, Any]) -> str:
    """One-line log summary."""

    def fmt(key: str) -> str:
        value = metrics.get(key)
        usable = isinstance(value, (int, float)) and not isinstance(value, bool)
        return f"{value:.4f}" if usable else "n/a"

    return (
        f"macro_f1={fmt('macro_f1')} f1={fmt('f1')} roc_auc={fmt('roc_auc')} "
        f"pr_auc={fmt('pr_auc')} fpr={fmt('fpr')} fnr={fmt('fnr')}"
    )

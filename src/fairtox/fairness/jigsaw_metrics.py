"""The benchmark's own bias metrics (Borkan et al., 2019).

Threshold-free, so they complement the FPR analysis rather than restating it, and
they make our numbers comparable to published work. Each isolates a different
failure: **subgroup AUC** is separability within the group; **BPSN** puts
background positives against subgroup negatives, the threshold-free analogue of
our FPR gap; **BNSP** does the reverse and measures under-protection, which is
why a good BPSN alone does not make a model fair.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """AUC, or None when the slice carries only one class."""
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def subgroup_aucs(
    predictions: pd.DataFrame,
    identity_columns: list[str],
    min_support: int = 100,
) -> pd.DataFrame:
    """Per-subgroup, BPSN and BNSP AUCs for one prediction file."""
    labels = predictions["label"].to_numpy().astype(int)
    scores = predictions["prob"].to_numpy(dtype=float)

    rows = []
    for column in identity_columns:
        if column not in predictions.columns:
            continue
        member = predictions[column].to_numpy().astype(int) == 1
        if not member.any():
            continue
        background = ~member

        # BPSN: background toxic (label 1) against subgroup benign (label 0).
        bpsn_mask = (background & (labels == 1)) | (member & (labels == 0))
        # BNSP: background benign against subgroup toxic.
        bnsp_mask = (background & (labels == 0)) | (member & (labels == 1))

        rows.append(
            {
                "subgroup": column,
                "n": int(member.sum()),
                "n_nontoxic": int((member & (labels == 0)).sum()),
                "n_toxic": int((member & (labels == 1)).sum()),
                "subgroup_auc": _safe_auc(labels[member], scores[member]),
                "bpsn_auc": _safe_auc(labels[bpsn_mask], scores[bpsn_mask]),
                "bnsp_auc": _safe_auc(labels[bnsp_mask], scores[bnsp_mask]),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # An AUC needs both classes present, so a subgroup only carries a claim when
    # it clears the floor on both sides -- not just on non-toxic rows.
    table["reportable"] = (table["n_nontoxic"] >= min_support) & (table["n_toxic"] >= min_support)
    return table.sort_values("n", ascending=False).reset_index(drop=True)


def generalised_mean(values: Any, power: float = -5.0) -> float | None:
    """Power mean, as the competition used it.

    A strongly negative power makes the aggregate dominated by the *worst*
    subgroup rather than the average one -- the point being that a model is not
    fair merely because it treats most groups well.
    """
    finite = [
        float(v) for v in np.asarray(list(values), dtype=object)
        if v is not None and isinstance(v, (int, float, np.floating)) and np.isfinite(float(v))
    ]
    if not finite:
        return None
    clipped = np.clip(np.asarray(finite, dtype=float), 1e-6, None)
    return float(np.power(np.mean(np.power(clipped, power)), 1.0 / power))


def summarise(table: pd.DataFrame, overall_auc: float | None) -> dict[str, Any]:
    """Worst-case aggregates plus the competition's headline score."""
    if table.empty:
        return {"overall_auc": overall_auc, "n_reportable": 0, "final_bias_score": None}

    reportable = table[table["reportable"]] if "reportable" in table.columns else table
    summary: dict[str, Any] = {
        "overall_auc": overall_auc,
        "n_reportable": int(len(reportable)),
    }

    for name in ("subgroup_auc", "bpsn_auc", "bnsp_auc"):
        values = reportable[name].tolist()
        summary[f"{name}_worst_mean"] = generalised_mean(values)
        finite = [float(v) for v in values if v is not None and np.isfinite(float(v))]
        summary[f"{name}_min"] = float(np.min(finite)) if finite else None
        summary[f"{name}_mean"] = float(np.mean(finite)) if finite else None

    # The competition's headline: overall AUC averaged with the three worst-case
    # bias AUCs, so a model cannot buy a good score with aggregate accuracy.
    components = [
        summary.get("subgroup_auc_worst_mean"),
        summary.get("bpsn_auc_worst_mean"),
        summary.get("bnsp_auc_worst_mean"),
    ]
    usable = [c for c in components if c is not None]
    summary["final_bias_score"] = (
        float((overall_auc + sum(usable)) / (1 + len(usable)))
        if overall_auc is not None and usable
        else None
    )
    return summary

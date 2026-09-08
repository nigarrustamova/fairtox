"""Global metrics, the collapse detector, and the Jigsaw benchmark metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairtox.evaluation.metrics import classification_metrics, confusion_counts, is_degenerate
from fairtox.fairness import jigsaw_metrics


def test_metrics_on_a_hand_computable_case():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.2])  # one FP, one FN

    metrics = classification_metrics(y_true, y_prob, 0.5)
    assert metrics["tn"] == 1 and metrics["fp"] == 1
    assert metrics["fn"] == 1 and metrics["tp"] == 1
    assert metrics["fpr"] == pytest.approx(0.5)
    assert metrics["fnr"] == pytest.approx(0.5)
    assert metrics["accuracy"] == pytest.approx(0.5)


def test_auc_is_none_on_a_single_class_slice():
    metrics = classification_metrics(np.zeros(5), np.linspace(0, 1, 5), 0.5)
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None


def test_rates_are_none_when_undefined_not_zero():
    """No non-toxic rows means FPR is undefined; 0.0 would be a different claim."""
    counts = confusion_counts(np.array([1, 1]), np.array([1, 1]))
    assert counts["fpr"] is None
    assert counts["fnr"] == pytest.approx(0.0)


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="probabilities"):
        classification_metrics(np.zeros(4), np.zeros(3))


def test_is_degenerate_detects_single_class_predictions():
    assert is_degenerate(np.zeros(10, dtype=int)) is True
    assert is_degenerate(np.ones(10, dtype=int)) is True
    assert is_degenerate(np.array([0, 1, 0, 1])) is False
    assert is_degenerate(np.array([], dtype=int)) is False


def test_degenerate_flag_travels_with_the_metrics():
    metrics = classification_metrics(np.array([0, 1, 0, 1]), np.array([0.1] * 4), 0.5)
    assert metrics["degenerate"] is True
    assert metrics["accuracy"] == pytest.approx(0.5)  # and yet it looks unremarkable


# -- the benchmark's own metrics ---------------------------------------------


def test_bpsn_auc_falls_when_benign_subgroup_text_scores_high():
    """BPSN is the metric designed to detect exactly our failure mode."""
    frame = pd.DataFrame(
        {
            "id": range(8),
            "label": [1, 1, 1, 1, 0, 0, 0, 0],
            # Background toxic scores low; subgroup benign scores high.
            "prob": [0.2, 0.3, 0.1, 0.2, 0.9, 0.8, 0.95, 0.85],
            "grp": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    table = jigsaw_metrics.subgroup_aucs(frame, ["grp"], min_support=1)
    assert table.loc[0, "bpsn_auc"] == pytest.approx(0.0)


def test_bnsp_auc_is_the_safety_counterpart():
    """Low BNSP means abuse aimed at the group is scored no higher than benign text."""
    frame = pd.DataFrame(
        {
            "id": range(8),
            "label": [0, 0, 0, 0, 1, 1, 1, 1],
            # Background benign scores high; subgroup toxic scores low.
            "prob": [0.9, 0.8, 0.95, 0.85, 0.2, 0.3, 0.1, 0.2],
            "grp": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    table = jigsaw_metrics.subgroup_aucs(frame, ["grp"], min_support=1)
    assert table.loc[0, "bnsp_auc"] == pytest.approx(0.0)


def test_generalised_mean_is_dominated_by_the_worst_value():
    balanced = jigsaw_metrics.generalised_mean([0.8, 0.8, 0.8])
    with_one_bad = jigsaw_metrics.generalised_mean([0.8, 0.8, 0.2])
    assert with_one_bad < balanced
    assert with_one_bad < 0.5  # the worst group pulls it down hard


def test_generalised_mean_tolerates_missing_values():
    assert jigsaw_metrics.generalised_mean([None, None]) is None
    assert jigsaw_metrics.generalised_mean([0.5, None]) == pytest.approx(0.5, abs=1e-6)


def test_jigsaw_reportable_needs_support_on_both_sides():
    """An AUC needs both classes, so one-sided support is not enough."""
    frame = pd.DataFrame(
        {
            "id": range(12),
            "label": [0] * 10 + [1] * 2,
            "prob": [0.1] * 10 + [0.9] * 2,
            "grp": [1] * 12,
        }
    )
    table = jigsaw_metrics.subgroup_aucs(frame, ["grp"], min_support=5)
    assert bool(table.loc[0, "reportable"]) is False


def test_final_bias_score_is_none_without_an_overall_auc():
    frame = pd.DataFrame(
        {"id": range(4), "label": [0, 1, 0, 1], "prob": [0.1, 0.9, 0.2, 0.8], "grp": [1] * 4}
    )
    table = jigsaw_metrics.subgroup_aucs(frame, ["grp"], min_support=1)
    assert jigsaw_metrics.summarise(table, None)["final_bias_score"] is None

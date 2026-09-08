"""Fairness maths, checked against hand-computed values."""

from __future__ import annotations

import pandas as pd
import pytest

from fairtox.fairness import auditor


def test_subgroup_rates_match_hand_computation(predictions, config):
    table, _ = auditor.audit(predictions, ["group_a", "group_b"], config)
    by_group = table.set_index("subgroup")

    assert by_group.loc["group_a", "fpr"] == pytest.approx(0.5)
    assert by_group.loc["group_a", "fnr"] == pytest.approx(0.5)
    assert by_group.loc["group_b", "fpr"] == pytest.approx(0.0)
    assert by_group.loc["group_b", "fnr"] == pytest.approx(0.0)


def test_confusion_counts_are_internally_consistent(predictions, config):
    table, _ = auditor.audit(predictions, ["group_a", "group_b"], config)
    for _, row in table.iterrows():
        assert row["tp"] + row["fp"] + row["tn"] + row["fn"] == row["n"]
        assert row["fp"] + row["tn"] == row["n_nontoxic"]
        assert row["tp"] + row["fn"] == row["n_toxic"]


def test_fpr_gap_is_max_minus_min(predictions, config):
    _, summary = auditor.audit(predictions, ["group_a", "group_b"], config)
    assert summary["fpr_gap"] == pytest.approx(0.5)
    assert summary["worst_subgroup"]["name"] == "group_a"
    assert summary["best_subgroup"]["name"] == "group_b"


def test_support_floor_excludes_small_groups_from_the_gap(predictions, config):
    """A noisy small group must not be able to drive the headline number."""
    config.set("evaluation.min_support", 100)  # nothing here clears it
    table, summary = auditor.audit(predictions, ["group_a", "group_b"], config)

    assert not table["reportable_fpr"].any()
    assert summary["n_reportable_fpr"] == 0
    assert summary["fpr_gap"] is None  # undefined, not silently zero


def test_small_groups_are_still_measured_just_not_aggregated(predictions, config):
    config.set("evaluation.min_support", 100)
    table, _ = auditor.audit(predictions, ["group_a", "group_b"], config)
    assert table["fpr"].notna().all()
    assert len(table) == 2


def test_fpr_and_fnr_floors_are_independent(config):
    """FPR is estimated on non-toxic rows and FNR on toxic ones.

    A group with plenty of benign comments and three toxic ones may report an
    FPR and must not report an FNR, so the two flags have to move separately.
    """
    frame = pd.DataFrame(
        {
            "id": range(13),
            "label": [0] * 10 + [1] * 3,
            "prob": [0.1] * 10 + [0.9] * 3,
            "grp": [1] * 13,
        }
    )
    config.set("evaluation.min_support", 5)
    table, summary = auditor.audit(frame, ["grp"], config)

    assert bool(table.loc[0, "reportable_fpr"]) is True   # 10 non-toxic >= 5
    assert bool(table.loc[0, "reportable_fnr"]) is False  # 3 toxic < 5
    assert summary["fpr_gap"] is not None
    assert summary["macro_fnr"] is None


def test_confidence_interval_brackets_the_estimate(predictions, config):
    table, _ = auditor.audit(predictions, ["group_a", "group_b"], config)
    for _, row in table.iterrows():
        assert row["fpr_ci_low"] <= row["fpr"] <= row["fpr_ci_high"]
        assert row["fnr_ci_low"] <= row["fnr"] <= row["fnr_ci_high"]


def test_wilson_interval_stays_wide_at_zero_successes():
    """The reason Wilson is the default rather than the percentile bootstrap.

    At zero observed false positives the bootstrap reports [0, 0] whatever the
    sample size. Wilson still widens as the sample shrinks, which is the honest
    behaviour for a subgroup with only a hundred benign comments.
    """
    small_low, small_high = auditor.wilson_interval(0, 100)
    large_low, large_high = auditor.wilson_interval(0, 10_000)

    assert small_low == 0.0 and large_low == 0.0
    assert small_high > large_high > 0.0
    assert small_high > 0.01  # not the zero-width interval a bootstrap would give


def test_bootstrap_interval_is_available_but_collapses_at_zero(predictions, config):
    import numpy as np

    rng = np.random.default_rng(0)
    low, high = auditor.bootstrap_interval(0, 100, rng, 500, 0.95)
    assert (low, high) == (0.0, 0.0)

    config.set("evaluation.ci_method", "bootstrap")
    table, _ = auditor.audit(predictions, ["group_a", "group_b"], config)
    assert table["fpr_ci_low"].notna().all()


def test_unknown_ci_method_is_rejected(predictions, config):
    config.set("evaluation.ci_method", "jackknife")
    with pytest.raises(ValueError, match="ci_method"):
        auditor.audit(predictions, ["group_a"], config)


def test_threshold_changes_the_verdict(predictions, config):
    _, strict = auditor.audit(predictions, ["group_a", "group_b"], config, threshold=0.99)
    # At tau=0.99 nothing is flagged, so there are no false positives anywhere.
    assert strict["macro_fpr"] == pytest.approx(0.0)


def test_perfect_model_reports_zero_not_missing(config):
    """Regression: a true FPR of 0.0 is falsy and must not become None or nan."""
    frame = pd.DataFrame(
        {
            "id": range(6),
            "label": [0, 0, 0, 0, 1, 1],
            "prob": [0.1, 0.1, 0.2, 0.2, 0.9, 0.9],  # flawless
            "grp": [1, 1, 1, 1, 1, 1],
        }
    )
    table, summary = auditor.audit(frame, ["grp"], config)

    assert summary["global"]["fpr"] == 0.0
    assert summary["global"]["fpr"] is not None
    assert table.loc[0, "fpr"] == 0.0
    assert summary["macro_fpr"] == pytest.approx(0.0)
    assert summary["fpr_gap"] == pytest.approx(0.0)  # zero gap, not undefined


def test_collapsed_model_is_flagged_not_celebrated(config):
    """A model predicting one class has perfect parity. That is collapse."""
    frame = pd.DataFrame(
        {
            "id": range(8),
            "label": [0, 0, 1, 1, 0, 0, 1, 1],
            "prob": [0.1] * 8,  # never flags anything
            "grp_a": [1, 1, 1, 1, 0, 0, 0, 0],
            "grp_b": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    _, summary = auditor.audit(frame, ["grp_a", "grp_b"], config)

    # The parity numbers look flawless...
    assert summary["fpr_gap"] == pytest.approx(0.0)
    assert summary["macro_fpr"] == pytest.approx(0.0)
    # ...and the flag is what stops them being reported as a result.
    assert summary["degenerate"] is True
    assert summary["macro_fnr"] == pytest.approx(1.0)  # it misses every toxic comment


def test_healthy_model_is_not_flagged(predictions, config):
    _, summary = auditor.audit(predictions, ["group_a", "group_b"], config)
    assert summary["degenerate"] is False


def test_empty_subgroup_list_returns_an_empty_audit(predictions, config):
    table, summary = auditor.audit(predictions, [], config)
    assert table.empty
    assert summary["n_reportable_fpr"] == 0


def test_eod_is_measured_against_the_global_rate(predictions, config):
    _, summary = auditor.audit(predictions, ["group_a", "group_b"], config)
    global_fpr = summary["global"]["fpr"]
    assert summary["eod_fpr"] == pytest.approx(max(abs(0.5 - global_fpr), abs(0.0 - global_fpr)))

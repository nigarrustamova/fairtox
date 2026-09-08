"""Per-subgroup decision thresholds.

The fixture is built so every expected number can be worked out on paper:
``group_a`` has ten non-toxic scores spaced 0.05 apart, five of them at or above
the global threshold, and ``group_b`` has ten that are all well below it. That
makes the global false-positive rate 5/20 = 0.25, which is the target every
subgroup is then moved onto.

The two policies are tested against each other on purpose. ``equalize`` reaches
parity partly by flagging MORE benign comments about the group that was already
well served -- the levelling-down failure this project named in experiment A --
and ``cap`` is the variant that cannot do that. A test that only checked the gap
would call both of them a success.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairtox.stages.postproc import (
    fit_group_thresholds,
    global_fpr,
    row_thresholds,
    shift_to_global,
)

GLOBAL = 0.5
COLUMNS = ["group_a", "group_b"]


@pytest.fixture
def val() -> pd.DataFrame:
    """20 non-toxic rows and 4 toxic ones.

    group_a non-toxic scores: 0.05 .. 0.95, so 5 of 10 clear 0.5 -> FPR 0.5
    group_b non-toxic scores: 0.01 .. 0.10, so 0 of 10 clear 0.5 -> FPR 0.0
    global FPR over the twenty non-toxic rows                    -> 0.25
    """
    rows = []
    for index, prob in enumerate(np.arange(0.05, 1.0, 0.10)):
        rows.append({"id": index, "label": 0, "prob": round(float(prob), 2),
                     "group_a": 1, "group_b": 0})
    for index, prob in enumerate(np.arange(0.01, 0.11, 0.01)):
        rows.append({"id": 100 + index, "label": 0, "prob": round(float(prob), 2),
                     "group_a": 0, "group_b": 1})
    # Toxic rows exist so the fixture is not degenerate; they never enter an FPR.
    for index in range(4):
        rows.append({"id": 200 + index, "label": 1, "prob": 0.9,
                     "group_a": index % 2, "group_b": (index + 1) % 2})
    frame = pd.DataFrame(rows)
    frame["any_identity"] = 1
    return frame


# -- the target --------------------------------------------------------------


def test_global_fpr_ignores_toxic_rows(val):
    assert global_fpr(val, GLOBAL) == pytest.approx(0.25)


def test_global_fpr_is_none_without_negatives():
    toxic_only = pd.DataFrame({"label": [1, 1], "prob": [0.9, 0.8]})
    assert global_fpr(toxic_only, GLOBAL) is None


# -- fitting -----------------------------------------------------------------


def test_threshold_lands_on_or_under_the_target(val):
    """floor(0.25 * 10) = 2 rows may be flagged, so the threshold is the 2nd largest."""
    thresholds, rows = fit_group_thresholds(val, COLUMNS, 0.25, 1, GLOBAL, "equalize")

    assert thresholds["group_a"] == pytest.approx(0.85)
    achieved = next(r["val_fpr_after"] for r in rows if r["subgroup"] == "group_a")
    assert achieved == pytest.approx(0.2)
    assert achieved <= 0.25


def test_equalize_lowers_a_group_that_was_already_below_target(val):
    """Parity, bought by flagging more benign comments about group_b."""
    thresholds, rows = fit_group_thresholds(val, COLUMNS, 0.25, 1, GLOBAL, "equalize")
    after = next(r["val_fpr_after"] for r in rows if r["subgroup"] == "group_b")

    assert thresholds["group_b"] < GLOBAL
    assert after > 0.0


def test_cap_never_lowers_a_threshold(val):
    """The same fit, with the levelling-down move forbidden."""
    thresholds, rows = fit_group_thresholds(val, COLUMNS, 0.25, 1, GLOBAL, "cap")
    after = next(r["val_fpr_after"] for r in rows if r["subgroup"] == "group_b")

    assert thresholds["group_b"] == pytest.approx(GLOBAL)
    assert thresholds["group_a"] == pytest.approx(0.85)
    assert after == pytest.approx(0.0)


def test_a_zero_target_flags_nothing(val):
    """floor(0 * n) = 0, so the threshold has to sit strictly above the top score."""
    thresholds, _ = fit_group_thresholds(val, COLUMNS, 0.0, 1, GLOBAL, "equalize")
    scores = val.loc[(val["group_a"] == 1) & (val["label"] == 0), "prob"].to_numpy()

    assert (scores >= thresholds["group_a"]).sum() == 0


def test_a_subgroup_below_the_floor_gets_no_threshold(val):
    """Eleven required, ten available: no threshold, and the support is recorded."""
    thresholds, rows = fit_group_thresholds(val, COLUMNS, 0.25, 11, GLOBAL, "equalize")

    assert thresholds == {}
    assert all(row["fitted"] is False for row in rows)
    assert {row["n_nontoxic_val"] for row in rows} == {10}


def test_unknown_policy_is_refused(val):
    with pytest.raises(ValueError, match="unknown policy"):
        fit_group_thresholds(val, COLUMNS, 0.25, 1, GLOBAL, "whatever")


# -- applying ----------------------------------------------------------------


def test_row_threshold_is_the_mean_over_memberships():
    """Subgroups overlap, so a row can belong to two groups with two thresholds."""
    frame = pd.DataFrame({"prob": [0.5], "group_a": [1], "group_b": [1]})
    per_row = row_thresholds(frame, {"group_a": 0.8, "group_b": 0.4}, GLOBAL)

    assert per_row[0] == pytest.approx(0.6)


def test_a_row_in_no_fitted_group_keeps_the_global_threshold():
    frame = pd.DataFrame({"prob": [0.5, 0.5], "group_a": [1, 0], "group_b": [0, 0]})
    per_row = row_thresholds(frame, {"group_a": 0.8}, GLOBAL)

    assert per_row.tolist() == pytest.approx([0.8, GLOBAL])


def test_no_fitted_thresholds_is_a_no_op(val):
    per_row = row_thresholds(val, {}, GLOBAL)

    assert per_row.tolist() == pytest.approx([GLOBAL] * len(val))


def test_the_shift_reproduces_the_per_row_rule(val):
    """`prob - t + 0.5 >= 0.5` must decide exactly what `prob >= t` decides."""
    per_row = row_thresholds(val, {"group_a": 0.85, "group_b": 0.09}, GLOBAL)
    shifted = shift_to_global(val, per_row)

    direct = val["prob"].to_numpy() >= per_row
    through_shift = shifted["prob"].to_numpy() >= 0.5
    assert np.array_equal(direct, through_shift)


def test_the_shift_leaves_the_caller_alone(val):
    before = val["prob"].tolist()
    shift_to_global(val, row_thresholds(val, {"group_a": 0.85}, GLOBAL))

    assert val["prob"].tolist() == before

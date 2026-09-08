"""The mechanism analysis: the shortcut fit, matched operating points, levelling.

These three numbers carry the paper's central argument, so each is checked
against a case whose answer can be worked out by hand rather than against the
function's own output.
"""

from __future__ import annotations

import pytest

from fairtox.stages.mechanism import (
    gap_spread,
    levelling_direction,
    matched_operating_points,
    toxic_rate_fit,
)


def _subgroup(name: str, toxic_rate: float, fpr: float, n: int = 1000, reportable: bool = True):
    return {
        "subgroup": name,
        "n": n,
        "n_toxic": int(round(toxic_rate * n)),
        "n_nontoxic": n - int(round(toxic_rate * n)),
        "fpr": fpr,
        "reportable_fpr": reportable,
    }


# -- the shortcut fit --------------------------------------------------------


def test_fit_recovers_an_exact_line():
    """FPR = 0.5 * toxic_rate + 0.01, planted exactly."""
    rows = [_subgroup(f"g{i}", r, 0.5 * r + 0.01) for i, r in enumerate((0.1, 0.2, 0.3, 0.4))]
    fit = toxic_rate_fit(rows)

    assert fit["slope"] == pytest.approx(0.5)
    assert fit["intercept"] == pytest.approx(0.01)
    assert fit["pearson_r"] == pytest.approx(1.0)
    assert fit["n_subgroups"] == 4


def test_the_toxic_rate_comes_from_the_audited_rows():
    """n_toxic / n, not a corpus-wide figure, so x and y describe the same rows."""
    rows = [
        _subgroup("a", 0.10, 0.02, n=500),
        _subgroup("b", 0.20, 0.04, n=2000),
        _subgroup("c", 0.30, 0.06, n=800),
    ]
    fit = toxic_rate_fit(rows)
    assert fit["toxic_rate_min"] == pytest.approx(0.10, abs=1e-3)
    assert fit["toxic_rate_max"] == pytest.approx(0.30, abs=1e-3)


def test_subgroups_below_the_floor_are_excluded():
    """A rate from a handful of comments must not move the slope the paper quotes."""
    solid = [_subgroup(f"g{i}", r, 0.5 * r) for i, r in enumerate((0.1, 0.2, 0.3))]
    noisy = _subgroup("tiny", 0.35, 0.9, n=40, reportable=False)

    assert toxic_rate_fit(solid + [noisy])["slope"] == pytest.approx(0.5)
    assert "tiny" not in toxic_rate_fit(solid + [noisy])["subgroups"]


def test_too_few_points_returns_none_rather_than_a_number():
    rows = [_subgroup("a", 0.1, 0.02), _subgroup("b", 0.2, 0.04)]
    assert toxic_rate_fit(rows) is None


def test_missing_rates_are_skipped():
    rows = [_subgroup(f"g{i}", r, 0.5 * r) for i, r in enumerate((0.1, 0.2, 0.3))]
    rows.append({**_subgroup("undefined", 0.4, 0.0), "fpr": None})
    assert toxic_rate_fit(rows)["n_subgroups"] == 3


def test_a_flat_relationship_reports_a_flat_slope():
    """The fit has to be able to refute the hypothesis, not only support it."""
    rows = [_subgroup(f"g{i}", r, 0.05) for i, r in enumerate((0.1, 0.2, 0.3, 0.4))]
    fit = toxic_rate_fit(rows)
    assert fit["slope"] == pytest.approx(0.0, abs=1e-9)
    assert fit["pearson_r"] is None  # no variance in y, so the correlation is undefined


# -- matched operating points ------------------------------------------------


def _row(model: str, threshold: float, global_fpr: float, gap: float):
    return {"model": model, "threshold": threshold, "global_fpr": global_fpr, "fpr_gap": gap}


def test_interpolation_at_a_matched_global_fpr():
    """Baseline gap is 0.10 at FPR 0.02 and 0.20 at 0.04, so 0.03 interpolates to 0.15."""
    rows = [
        _row("baseline", 0.6, 0.02, 0.10),
        _row("baseline", 0.4, 0.04, 0.20),
        _row("mitigated", 0.5, 0.03, 0.12),
    ]
    matched = matched_operating_points(rows)

    assert len(matched) == 1
    assert matched[0]["baseline_gap_at_same_fpr"] == pytest.approx(0.15)
    assert matched[0]["gap_reduction"] == pytest.approx(0.2)  # 0.12 is 20% below 0.15
    assert matched[0]["comparable"] is True


def test_points_outside_the_baseline_range_are_flagged_not_clamped():
    """Regression guard: np.interp holds the endpoint value and would present an
    extrapolation as a measurement -- at exactly the end where a conservative
    mitigated model lands."""
    rows = [
        _row("baseline", 0.6, 0.02, 0.10),
        _row("baseline", 0.4, 0.04, 0.20),
        _row("mitigated", 0.9, 0.005, 0.05),
    ]
    matched = matched_operating_points(rows)

    assert matched[0]["comparable"] is False
    assert matched[0]["gap_reduction"] is None


def test_a_mitigated_model_that_is_worse_reports_a_negative_reduction():
    rows = [
        _row("baseline", 0.6, 0.02, 0.10),
        _row("baseline", 0.4, 0.04, 0.20),
        _row("mitigated", 0.5, 0.03, 0.18),
    ]
    assert matched_operating_points(rows)[0]["gap_reduction"] == pytest.approx(-0.2)


def test_one_baseline_point_cannot_be_interpolated():
    rows = [_row("baseline", 0.5, 0.02, 0.10), _row("mitigated", 0.5, 0.02, 0.08)]
    assert matched_operating_points(rows) == []


def test_gap_spread_measures_threshold_sensitivity():
    rows = [
        _row("baseline", 0.3, 0.05, 0.14),
        _row("baseline", 0.5, 0.03, 0.08),
        _row("baseline", 0.7, 0.01, 0.03),
        _row("mitigated", 0.3, 0.04, 0.07),
        _row("mitigated", 0.5, 0.03, 0.05),
        _row("mitigated", 0.7, 0.02, 0.04),
    ]
    assert gap_spread(rows, "baseline") == pytest.approx(0.11)
    assert gap_spread(rows, "mitigated") == pytest.approx(0.03)


# -- levelling direction -----------------------------------------------------


def _summary(worst: float, best: float) -> dict:
    return {"worst_subgroup": {"fpr": worst}, "best_subgroup": {"fpr": best}}


def test_levelling_up_is_the_worst_group_improving():
    result = levelling_direction(_summary(0.087, 0.010), _summary(0.064, 0.015))
    assert result["direction"] == "mixed"  # best got worse too, so it is not clean


def test_a_clean_improvement_is_levelling_up():
    result = levelling_direction(_summary(0.110, 0.026), _summary(0.058, 0.010))
    assert result["direction"] == "levelling up"


def test_levelling_down_is_named_and_not_counted_as_a_gain():
    """The wtox pattern: the gap narrows while the worst-off group gets worse.

    max - min falls in both cases, so without this check a run that made every
    group worse and the best-off group worst of all would be written up as a
    fairness improvement.
    """
    result = levelling_direction(_summary(0.087, 0.010), _summary(0.089, 0.024))
    assert result["direction"] == "levelling down"
    assert "best-off" in result["reason"]


def test_a_missing_extreme_is_indeterminate():
    result = levelling_direction({"worst_subgroup": None, "best_subgroup": None}, _summary(1, 1))
    assert result["direction"] == "indeterminate"

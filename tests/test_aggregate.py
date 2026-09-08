"""Cross-family aggregation: the statistics the report's headline numbers are.

Every number in the paper's comparison table comes out of these functions, so
each is checked against a case whose answer is known in advance rather than
against the function's own output. The sign test in particular has to disagree
with the interval on the family where they genuinely disagree -- that
disagreement is what the write-up is required to report.
"""

from __future__ import annotations

import gzip
import json

import pytest

from fairtox.evaluation.aggregate import (
    exchange_rate,
    family_deltas,
    load_study,
    paired_summary,
    subgroup_deltas,
)

# -- paired_summary ----------------------------------------------------------


def test_mean_and_spread_of_a_hand_computed_sample():
    stats = paired_summary([-1.0, -2.0, -3.0])

    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(-2.0)
    assert stats["sd"] == pytest.approx(1.0)      # ddof=1
    assert stats["n_negative"] == 3


def test_interval_is_the_student_t_one():
    """t(0.975, 2) = 4.303, so the half-width is 4.303 * 1 / sqrt(3)."""
    stats = paired_summary([-1.0, -2.0, -3.0])
    half = 4.302652 * 1.0 / 3 ** 0.5

    assert stats["ci_low"] == pytest.approx(-2.0 - half, rel=1e-4)
    assert stats["ci_high"] == pytest.approx(-2.0 + half, rel=1e-4)
    assert stats["excludes_zero"] is False


def test_eight_agreeing_seeds_reach_the_sign_test():
    """The headline claim: 8/8 in one direction is p = 0.0078, not p = 0.5."""
    stats = paired_summary([-0.05] * 8)
    assert stats["sign_p"] == pytest.approx(2 * 0.5 ** 8)


def test_three_seeds_can_never_reach_significance():
    """Reported anyway, because 0.25 next to a 3-seed result is the honest label."""
    assert paired_summary([-0.05] * 3)["sign_p"] == pytest.approx(0.25)


def test_the_interval_and_the_sign_test_are_allowed_to_disagree():
    """6/8 in one direction: the interval can exclude zero while the test does not.

    This is the held-out result's shape, and the write-up must quote both.
    """
    stats = paired_summary([-0.03, -0.03, -0.03, -0.03, -0.03, -0.03, 0.001, 0.001])

    assert stats["excludes_zero"] is True
    assert stats["sign_p"] > 0.05


def test_exact_zeros_are_dropped_from_the_sign_test():
    """A difference of exactly zero has no sign, so it cannot vote."""
    stats = paired_summary([-1.0, -1.0, 0.0])

    assert stats["n_negative"] == 2
    assert stats["n_positive"] == 0
    assert stats["sign_p"] == pytest.approx(0.5)   # 2 of 2, not 2 of 3


def test_a_single_value_has_no_interval():
    stats = paired_summary([-0.05])
    assert stats["ci_low"] == stats["ci_high"] == pytest.approx(-0.05)
    assert stats["excludes_zero"] is False


def test_nothing_to_summarise():
    assert paired_summary([])["n"] == 0
    assert paired_summary([None, float("nan")])["n"] == 0


# -- exchange_rate -----------------------------------------------------------


def test_tax_is_the_ratio_of_the_means():
    """0.10 FNR paid for 0.05 of gap closed is a tax of 2.0."""
    gap = [-0.045, -0.050, -0.055, -0.048, -0.052]           # mean exactly -0.05
    assert exchange_rate(gap, [0.10] * 5) == pytest.approx(2.0)


def test_the_tax_is_computed_from_means_not_from_per_seed_ratios():
    """One seed whose gap barely moved would otherwise dominate the average."""
    gap = [-0.005, -0.05, -0.05, -0.05, -0.05]
    fnr = [0.08] * 5
    mean_of_ratios = sum(f / -g for f, g in zip(fnr, gap, strict=True)) / 5
    tax = exchange_rate(gap, fnr)

    assert tax == pytest.approx(0.08 / 0.041)                # the honest rate, ~1.95
    assert mean_of_ratios > 2 * tax                          # ~4.48, inflated by seed 1


def test_no_tax_when_the_gap_did_not_close():
    """A negative or infinite 'rate' is not something anyone should quote."""
    assert exchange_rate([+0.02, +0.03], [0.05, 0.05]) is None
    assert exchange_rate([0.0, 0.0], [0.05, 0.05]) is None


def test_no_tax_when_safety_also_improved():
    """Both axes moving the right way is a free win, not a rate."""
    assert exchange_rate([-0.04, -0.06], [-0.01, -0.02]) is None


def test_no_tax_when_the_gap_change_does_not_exclude_zero():
    """The held-out family's shape: 0.083 / 0.0065 = 12.7 describes only the noise."""
    noisy_gap = [-0.03, +0.02, -0.01, +0.01, -0.02]
    assert paired_summary(noisy_gap)["excludes_zero"] is False
    assert exchange_rate(noisy_gap, [0.08] * 5) is None


# -- loading -----------------------------------------------------------------


def _study() -> dict:
    def audit(name, fpr, reportable=True):
        return {"subgroup": name, "fpr": fpr, "reportable_fpr": reportable}

    return {
        "main": {
            "compare": {"axes": {
                "parity": {"fpr_gap": {"delta": -0.05}, "macro_fpr": {"delta": -0.03},
                           "eod_fpr": {"delta": -0.04}},
                "utility": {"macro_f1": {"delta": -0.01}},
                "safety": {"macro_fnr": {"delta": 0.10}},
            }},
            "audit": {
                "baseline": [audit("male", 0.10), audit("black", 0.20), audit("hindu", 0.5, False)],
                "mitigated": [audit("male", 0.08), audit("black", 0.15), audit("hindu", 0.4, False)],
            },
        }
    }


def test_a_digest_round_trips(tmp_path):
    path = tmp_path / "digest.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(_study(), handle)

    assert load_study(path) == _study()


def test_a_results_tree_is_read(tmp_path):
    for name, payload in (("compare", _study()["main"]["compare"]),
                          ("audit_baseline", {"subgroups": _study()["main"]["audit"]["baseline"]})):
        target = tmp_path / "main" / "metrics"
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    study = load_study(tmp_path)
    assert set(study) == {"main"}
    assert study["main"]["audit"]["baseline"][0]["subgroup"] == "male"


def test_smoke_runs_are_never_aggregated(tmp_path):
    """A synthetic-data run in the results tree must not reach a published table."""
    for run in ("main", "smoke", "smoke_gpu"):
        target = tmp_path / run / "metrics"
        target.mkdir(parents=True)
        (target / "compare.json").write_text(
            json.dumps(_study()["main"]["compare"]), encoding="utf-8")

    assert set(load_study(tmp_path)) == {"main"}


def test_a_missing_source_is_named(tmp_path):
    with pytest.raises(FileNotFoundError, match="no study data"):
        load_study(tmp_path / "nowhere")


# -- selection ---------------------------------------------------------------


def test_a_missing_run_is_excluded_not_guessed():
    assert family_deltas(_study(), ["main", "main_s99"], "fpr_gap") == [-0.05]


def test_an_unknown_metric_is_refused():
    with pytest.raises(KeyError, match="unknown metric"):
        family_deltas(_study(), ["main"], "accuracy")


def test_subgroups_need_both_arms_reportable():
    """Comparing a measured rate against an unmeasurable one is not a comparison."""
    deltas = subgroup_deltas(_study(), "main")

    assert set(deltas) == {"male", "black"}
    assert deltas["male"] == pytest.approx(-0.02)
    assert deltas["black"] == pytest.approx(-0.05)


def test_an_unknown_run_yields_nothing():
    assert subgroup_deltas(_study(), "nope") == {}

"""The ablation sweep table.

Two things it must never do: print a blank safety column, and let an arm that
stopped flagging read as the fairest model in the study. Both were observed on a
real run before these guards existed -- alpha=10 posted the best FPR gap and the
lowest macro FPR in the sweep, while missing 76% of genuine abuse, and the safety
column showed n/a for every row.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from fairtox.stages.ablate import _plan, _warn_on_inactive_arms


def _sweep(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# -- the flag-rate diagnostic ------------------------------------------------


def test_an_arm_that_stopped_flagging_is_called_out(caplog):
    """The alpha=10 pattern: near-perfect parity bought by predicting nothing."""
    caplog.set_level(logging.WARNING)
    _warn_on_inactive_arms(
        _sweep(
            [
                {"arm": "abl_class", "scheme": "class", "flag_rate": 0.182},
                {"arm": "abl_subgroup_a1", "scheme": "subgroup", "flag_rate": 0.053},
                {"arm": "abl_subgroup_a10", "scheme": "subgroup", "flag_rate": 0.031},
            ]
        )
    )
    assert "abl_subgroup_a10" in caplog.text
    assert "stopped flagging" in caplog.text


def test_a_collapsed_arm_is_not_warned_about_twice(caplog):
    """It already has its own warning, a COLLAPSED note and an INVALID verdict.

    This warning exists for the arms that stop *short* of full collapse -- the
    ones nothing else catches.
    """
    caplog.set_level(logging.WARNING)
    _warn_on_inactive_arms(
        _sweep(
            [
                {"arm": "abl_class", "scheme": "class", "flag_rate": 0.355,
                 "degenerate": False},
                {"arm": "abl_subgroup_a10", "scheme": "subgroup", "flag_rate": 0.0,
                 "degenerate": True},
            ]
        )
    )
    assert "stopped flagging" not in caplog.text


def test_a_near_collapsed_arm_is_still_warned_about(caplog):
    """Not degenerate, so no other check names it -- this one must."""
    caplog.set_level(logging.WARNING)
    _warn_on_inactive_arms(
        _sweep(
            [
                {"arm": "abl_class", "scheme": "class", "flag_rate": 0.182,
                 "degenerate": False},
                {"arm": "abl_subgroup_a10", "scheme": "subgroup", "flag_rate": 0.031,
                 "degenerate": False},
            ]
        )
    )
    assert "abl_subgroup_a10" in caplog.text
    assert "stopped flagging" in caplog.text


def test_comparable_arms_are_not_flagged(caplog):
    caplog.set_level(logging.WARNING)
    _warn_on_inactive_arms(
        _sweep(
            [
                {"arm": "abl_subgroup_a1", "scheme": "subgroup", "flag_rate": 0.100},
                {"arm": "abl_subgroup_a2", "scheme": "subgroup", "flag_rate": 0.085},
            ]
        )
    )
    assert "stopped flagging" not in caplog.text


def test_an_empty_sweep_is_not_an_error():
    _warn_on_inactive_arms(pd.DataFrame())


def test_a_sweep_without_flag_rates_is_not_an_error():
    _warn_on_inactive_arms(_sweep([{"arm": "a", "scheme": "subgroup"}]))


def test_all_zero_flag_rates_do_not_divide_by_zero(caplog):
    caplog.set_level(logging.WARNING)
    _warn_on_inactive_arms(
        _sweep([{"arm": "a", "scheme": "subgroup", "flag_rate": 0.0}])
    )


def test_missing_flag_rates_are_skipped(caplog):
    caplog.set_level(logging.WARNING)
    _warn_on_inactive_arms(
        _sweep(
            [
                {"arm": "a", "scheme": "subgroup", "flag_rate": 0.20},
                {"arm": "b", "scheme": "subgroup", "flag_rate": None},
            ]
        )
    )
    assert "'b'" not in caplog.text


# -- arm planning ------------------------------------------------------------


def test_only_the_subgroup_scheme_varies_with_alpha():
    """`none` and `class` ignore alpha, so sweeping them would queue clones."""
    arms = _plan(["class", "subgroup"], [1.0, 4.0, 10.0])
    names = [name for name, _, _ in arms]

    assert names == [
        "abl_class",
        "abl_subgroup_a1",
        "abl_subgroup_a4",
        "abl_subgroup_a10",
    ]
    assert [alpha for _, _, alpha in arms] == [None, 1.0, 4.0, 10.0]


def test_alphas_are_swept_in_order_whatever_the_config_says():
    """The trade-off curve is read left to right, so the arms are sorted."""
    arms = _plan(["subgroup"], [10.0, 1.0, 4.0])
    assert [alpha for _, _, alpha in arms] == [1.0, 4.0, 10.0]


@pytest.mark.parametrize("alpha, expected", [(1.0, "a1"), (2.5, "a2.5"), (10.0, "a10")])
def test_arm_names_carry_their_alpha(alpha, expected):
    """Names must differ per alpha, or the arms would share artefact paths."""
    (name, _, _), = _plan(["subgroup"], [alpha])
    assert name == f"abl_subgroup_{expected}"

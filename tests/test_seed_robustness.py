"""Aggregation across seeds.

The previous project was criticised for reporting single-seed numbers, so this
stage is not decoration. What it has to get right: find the runs without being
told, treat the per-seed *difference* as the unit of analysis, and say plainly
when the effect is smaller than its own spread.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

from fairtox.stages.optional_analyses import _aggregate, _discover_seed_runs, seed_robustness
from fairtox.utils import io


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(io, "EXPERIMENTS_DIR", tmp_path / "experiments")
    return tmp_path / "results"


def _write_compare(run: str, d_gap: float, d_f1: float, d_fnr: float, verdict: str = "A") -> None:
    io.save_json(
        {
            "stage": "compare",
            "verdict": f"{verdict}: something",
            "degenerate": False,
            "axes": {
                "parity": {
                    "fpr_gap": {"delta": d_gap},
                    "macro_fpr": {"delta": d_gap / 2},
                    "eod_fpr": {"delta": d_gap / 3},
                },
                "utility": {"macro_f1": {"delta": d_f1}},
                "safety": {"macro_fnr": {"delta": d_fnr}},
            },
        },
        io.metrics_path(run, "compare"),
    )


def _ctx(config, run_name="main"):
    config.set("run.name", run_name)
    return SimpleNamespace(
        config=config,
        run_name=run_name,
        metrics_path=lambda stage: io.metrics_path(run_name, stage),
        table_path=lambda name, suffix=".md": io.table_path(run_name, name, suffix),
        save_metrics=lambda stage, payload: io.save_json(payload, io.metrics_path(run_name, stage)),
    )


# -- discovery ---------------------------------------------------------------


def test_runs_are_discovered_without_being_listed(results_root, config):
    for run in ("main", "main_s43", "main_s44"):
        _write_compare(run, -0.04, -0.002, 0.005)
    config.set("analysis.seed_run_glob", "main*")

    assert _discover_seed_runs(_ctx(config)) == ["main", "main_s43", "main_s44"]


def test_the_glob_excludes_unrelated_runs(results_root, config):
    """A throwaway timing run must not be aggregated into the headline claim."""
    _write_compare("main", -0.04, -0.002, 0.005)
    _write_compare("timing", -0.99, -0.5, 0.5)
    config.set("analysis.seed_run_glob", "main*")

    assert _discover_seed_runs(_ctx(config)) == ["main"]


def test_an_explicit_list_wins_over_discovery(results_root, config):
    for run in ("main", "main_s43", "main_s44"):
        _write_compare(run, -0.04, -0.002, 0.005)
    config.set("analysis.seed_runs", ["main", "main_s44"])

    assert _discover_seed_runs(_ctx(config)) == ["main", "main_s44"]


def test_a_listed_run_with_no_results_is_dropped(results_root, config):
    _write_compare("main", -0.04, -0.002, 0.005)
    config.set("analysis.seed_runs", ["main", "main_s99"])

    assert _discover_seed_runs(_ctx(config)) == ["main"]


def test_no_runs_is_not_an_error(results_root, config):
    assert _discover_seed_runs(_ctx(config)) == []


# -- the statistics ----------------------------------------------------------


def test_aggregate_reports_mean_spread_and_sign_agreement():
    table = pd.DataFrame({"d": [-0.05, -0.04, -0.06]})
    stats = _aggregate(table, "d")

    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(-0.05)
    assert stats["sd"] == pytest.approx(0.01)
    assert stats["n_negative"] == 3
    assert stats["sign_agreement"] == pytest.approx(1.0)


def test_sign_agreement_falls_when_seeds_disagree():
    table = pd.DataFrame({"d": [-0.05, 0.04, -0.06]})
    assert _aggregate(table, "d")["sign_agreement"] == pytest.approx(2 / 3)


def test_sample_standard_deviation_is_used():
    """The population form understates the spread, which is the wrong direction.

    For [0, 2] the sample sd (ddof=1) is sqrt(2); the population sd is 1.0. With
    three seeds that difference is the margin between quoting an effect and
    calling it inconclusive.
    """
    stats = _aggregate(pd.DataFrame({"d": [0.0, 2.0]}), "d")
    assert stats["sd"] == pytest.approx(2.0 ** 0.5)
    assert stats["sd"] > 1.0


def test_a_single_run_has_no_spread():
    stats = _aggregate(pd.DataFrame({"d": [-0.05]}), "d")
    assert stats["n"] == 1 and stats["sd"] is None


def test_missing_values_are_skipped_not_counted():
    stats = _aggregate(pd.DataFrame({"d": [-0.05, None, -0.03]}), "d")
    assert stats["n"] == 2


# -- the stage ---------------------------------------------------------------


def test_stage_writes_a_summary_over_discovered_runs(results_root, config):
    _write_compare("main", -0.050, -0.002, 0.004, verdict="A")
    _write_compare("main_s43", -0.040, -0.003, 0.006, verdict="A")
    _write_compare("main_s44", -0.060, -0.001, 0.005, verdict="A")
    config.set("analysis.seed_run_glob", "main*")

    seed_robustness(_ctx(config))
    payload = io.load_json(io.metrics_path("main", "seed_robustness"))

    assert payload["n_runs"] == 3
    assert payload["consistency"]["verdict_is_consistent"] is True
    assert payload["summary"]["d_fpr_gap"]["mean"] == pytest.approx(-0.05)
    assert payload["summary"]["d_fpr_gap"]["sign_agreement"] == pytest.approx(1.0)
    assert len(payload["per_run"]) == 3


def test_disagreeing_verdicts_are_recorded_not_hidden(results_root, config):
    """Picking the seed you prefer is the failure this makes impossible to miss."""
    _write_compare("main", -0.05, -0.002, 0.004, verdict="A")
    _write_compare("main_s43", 0.01, -0.003, 0.006, verdict="D")
    config.set("analysis.seed_run_glob", "main*")

    seed_robustness(_ctx(config))
    payload = io.load_json(io.metrics_path("main", "seed_robustness"))

    assert payload["consistency"]["verdict_is_consistent"] is False
    assert payload["consistency"]["verdicts"] == ["A", "D"]


def test_an_effect_smaller_than_its_spread_is_called_out(results_root, config, caplog):
    _write_compare("main", -0.05, -0.002, 0.004)
    _write_compare("main_s43", 0.04, -0.003, 0.006)
    _write_compare("main_s44", 0.02, -0.001, 0.005)
    config.set("analysis.seed_run_glob", "main*")

    seed_robustness(_ctx(config))
    assert "inconclusive" in caplog.text


def test_a_clear_effect_is_not_called_inconclusive(results_root, config, caplog):
    _write_compare("main", -0.050, -0.002, 0.004)
    _write_compare("main_s43", -0.048, -0.003, 0.006)
    _write_compare("main_s44", -0.052, -0.001, 0.005)
    config.set("analysis.seed_run_glob", "main*")

    seed_robustness(_ctx(config))
    assert "inconclusive" not in caplog.text


def test_a_single_run_warns_that_it_says_nothing(results_root, config, caplog):
    _write_compare("main", -0.05, -0.002, 0.004)
    config.set("analysis.seed_run_glob", "main*")

    seed_robustness(_ctx(config))
    assert "at least 2 are needed" in caplog.text


def test_nothing_to_aggregate_points_at_the_script(results_root, config, caplog):
    caplog.set_level(logging.INFO)
    seed_robustness(_ctx(config))
    assert "run_seeds.sh" in caplog.text
    assert io.load_json(io.metrics_path("main", "seed_robustness"))["n_runs"] == 0

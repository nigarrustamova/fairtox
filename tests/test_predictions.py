"""The prediction file format, which every analysis stage depends on."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairtox.evaluation import predictions as pred_io
from fairtox.stages.compare import decide_verdict, safety_signal


@pytest.fixture(autouse=True)
def isolated_results(tmp_path, monkeypatch):
    """Point the artefact tree at a temporary directory for these tests."""
    from fairtox.utils import io

    monkeypatch.setattr(io, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(io, "EXPERIMENTS_DIR", tmp_path / "experiments")
    return tmp_path


def _frame(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "label": np.array([0, 1] * (n // 2)),
            "any_identity": np.array([1, 0] * (n // 2)),
            "group_a": np.array([1, 0] * (n // 2)),
            "comment_text": [f"row {i}" for i in range(n)],
        }
    )


def test_round_trip_preserves_ids_labels_and_probabilities():
    frame = _frame()
    probs = np.linspace(0.05, 0.95, len(frame))
    pred_io.save_predictions("run", "baseline", "test", frame, probs, ["group_a"])

    loaded = pred_io.load_predictions("run", "baseline", "test")
    assert list(loaded["id"]) == list(frame["id"])
    assert list(loaded["label"]) == list(frame["label"])
    assert np.allclose(loaded["prob"], probs)


def test_length_mismatch_is_rejected():
    """A shuffling or dropping evaluation loader must fail loudly, not silently."""
    with pytest.raises(ValueError, match="must not shuffle or drop"):
        pred_io.save_predictions("run", "baseline", "test", _frame(6), np.zeros(5), [])


def test_identity_columns_exclude_the_reserved_names():
    frame = _frame()
    pred_io.save_predictions("run", "baseline", "test", frame, np.zeros(len(frame)), ["group_a"])
    loaded = pred_io.load_predictions("run", "baseline", "test")

    assert pred_io.identity_columns_in(loaded) == ["group_a"]
    for reserved in ("id", "label", "prob", "any_identity"):
        assert reserved not in pred_io.identity_columns_in(loaded)


def test_has_predictions_reports_absence():
    assert pred_io.has_predictions("run", "never_trained", "test") is False


def test_missing_file_says_how_to_produce_it():
    with pytest.raises(FileNotFoundError, match="run_all.py --only train_"):
        pred_io.load_predictions("run", "mitigated", "test")


def test_the_same_format_serves_every_model():
    """The classical baseline is audited by identical code, which needs this."""
    frame = _frame()
    for model in ("baseline", "mitigated", "tfidf"):
        pred_io.save_predictions("run", model, "test", frame, np.zeros(len(frame)), ["group_a"])

    columns = {
        model: list(pred_io.load_predictions("run", model, "test").columns)
        for model in ("baseline", "mitigated", "tfidf")
    }
    assert columns["baseline"] == columns["mitigated"] == columns["tfidf"]


# -- the verdict -------------------------------------------------------------


def _axes(
    d_gap: float | None,
    d_f1: float | None,
    d_fnr: float | None,
    d_global_fnr: float | None = None,
) -> dict:
    return {
        "parity": {"fpr_gap": {"delta": d_gap}},
        "utility": {"macro_f1": {"delta": d_f1}},
        "safety": {
            "macro_fnr": {"delta": d_fnr},
            "global_fnr": {"delta": d_global_fnr},
        },
    }


def test_verdict_a_near_pareto(config):
    verdict = decide_verdict(_axes(-0.05, -0.002, 0.001), config)
    assert verdict.startswith("A:")


def test_verdict_b_utility_traded_away(config):
    verdict = decide_verdict(_axes(-0.05, -0.04, 0.001), config)
    assert verdict.startswith("B:")


def test_verdict_c_safety_regression_outranks_utility(config):
    """Missing more genuine abuse is the finding that must not be buried."""
    verdict = decide_verdict(_axes(-0.05, -0.001, 0.06), config)
    assert verdict.startswith("C:")


def test_verdict_d_no_improvement(config):
    verdict = decide_verdict(_axes(0.01, 0.0, 0.0), config)
    assert verdict.startswith("D:")


def test_collapse_outranks_every_other_verdict(config):
    """A collapsed model has perfect parity; it must never get an outcome letter."""
    verdict = decide_verdict(_axes(-0.99, 0.0, 0.0), config, degenerate=True)
    assert verdict.startswith("INVALID")


def test_verdict_thresholds_come_from_the_config(config):
    axes = _axes(-0.05, -0.001, 0.03)
    assert decide_verdict(axes, config).startswith("C:")  # safety_tolerance 0.02

    config.set("verdict.safety_tolerance", 0.10)
    assert decide_verdict(axes, config).startswith("A:")


def test_missing_metrics_give_an_honest_non_answer(config):
    assert decide_verdict(_axes(None, None, None), config).startswith("indeterminate")


# -- the safety axis must never go missing -----------------------------------


def test_subgroup_fnr_is_preferred_when_available():
    delta, source = safety_signal(_axes(-0.05, -0.001, 0.03, d_global_fnr=0.09))
    assert (delta, source) == (0.03, "subgroup")


def test_global_fnr_stands_in_when_no_subgroup_clears_the_floor():
    delta, source = safety_signal(_axes(-0.05, -0.001, None, d_global_fnr=0.0735))
    assert (delta, source) == (0.0735, "global")


def test_a_safety_regression_is_caught_through_global_fnr(config):
    """Regression, observed on a real 20k-row run.

    No subgroup cleared the FNR floor, so macro_fnr was None, the safety branch
    was skipped, and a model missing 7.4 points more genuine abuse was written up
    as a mere utility trade-off.
    """
    verdict = decide_verdict(_axes(-0.0112, -0.0360, None, d_global_fnr=0.0735), config)
    assert verdict.startswith("C:")
    assert "global FNR" in verdict


def test_no_measurable_fnr_at_all_is_indeterminate_not_a_pass(config):
    """Unknown safety cost is not the same as no safety cost."""
    verdict = decide_verdict(_axes(-0.05, -0.001, None, d_global_fnr=None), config)
    assert verdict.startswith("indeterminate")
    assert "safety" in verdict


def test_global_fallback_does_not_override_a_healthy_run(config):
    verdict = decide_verdict(_axes(-0.05, -0.002, None, d_global_fnr=0.001), config)
    assert verdict.startswith("A:")

"""Experiment 4 -- the controlled comparison, and the stage the paper rests on.

Baseline against mitigated on three axes kept deliberately separate, because a
single "is it better?" number would hide the trade-off the study exists to
measure: **utility** (macro-F1, AUCs), **parity** (FPR gap, macro FPR, variance,
EOD) and **safety** (FNR on genuine abuse).

Three checks run before any number is written: same split, same training
settings, neither arm collapsed. Each of those failures produces output that
looks entirely normal and means nothing.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..data.split import assert_same_split
from ..evaluation.metrics import classification_metrics
from ..evaluation.predictions import load_predictions
from ..provenance import assert_comparable, warn_on_hardware_mismatch
from ..registry import Tier, stage
from ..utils import io, plotting
from ..utils.logging_utils import get_logger
from .audit import audit_model
from .train import load_result, metrics_key

logger = get_logger(__name__)

UTILITY_KEYS = ("macro_f1", "roc_auc", "pr_auc")
PARITY_KEYS = ("fpr_gap", "macro_fpr", "fpr_variance", "eod_fpr")
SAFETY_KEYS = ("macro_fnr", "fnr_gap")


def _model_result(ctx, model_name: str) -> dict[str, Any] | None:
    cached = ctx.take(f"{model_name}_metrics")
    if cached:
        return cached
    if ctx.metrics_path(metrics_key(model_name)).exists():
        return load_result(ctx, model_name)
    return None


def _global_metrics(ctx, model_name: str) -> dict[str, Any]:
    """Global test metrics for a model, from its saved record or recomputed."""
    result = _model_result(ctx, model_name)
    if result and "metrics" in result and "test" in result["metrics"]:
        return result["metrics"]["test"]

    predictions = load_predictions(ctx.run_name, model_name, "test")
    return classification_metrics(
        predictions["label"].to_numpy(),
        predictions["prob"].to_numpy(),
        float(ctx.config.get("evaluation.classification_threshold", 0.5)),
    )


def _delta(new: Any, old: Any) -> float | None:
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)):
        return None
    if isinstance(new, bool) or isinstance(old, bool):
        return None
    return float(new - old)


def _axis(keys, baseline: dict, mitigated: dict) -> dict[str, Any]:
    return {
        key: {
            "baseline": baseline.get(key),
            "mitigated": mitigated.get(key),
            "delta": _delta(mitigated.get(key), baseline.get(key)),
        }
        for key in keys
    }


@stage(
    "compare",
    tier=Tier.MUST,
    summary="Experiment 4: baseline vs mitigated on utility, parity and safety.",
    depends_on=("audit_baseline", "train_mitigated"),
)
def compare(ctx) -> None:
    # Guard first, on three axes. A comparison across two different partitions,
    # two different hyperparameter sets, or two different machines is not a
    # result -- and nothing in the numbers themselves would reveal any of them.
    baseline_result = _model_result(ctx, "baseline") or {}
    mitigated_result = _model_result(ctx, "mitigated") or {}
    what = "compare(baseline, mitigated)"

    assert_same_split(
        baseline_result.get("split_fingerprint"),
        mitigated_result.get("split_fingerprint"),
        what,
    )
    assert_comparable(baseline_result, mitigated_result, what)
    warn_on_hardware_mismatch(baseline_result, mitigated_result, what)

    # Both arms are audited by the same function at the same threshold.
    baseline_summary = ctx.take("audit_baseline_summary") or audit_model(ctx, "baseline")
    mitigated_summary = audit_model(ctx, "mitigated")

    baseline_global = _global_metrics(ctx, "baseline")
    mitigated_global = _global_metrics(ctx, "mitigated")

    axes = {
        "utility": _axis(UTILITY_KEYS, baseline_global, mitigated_global),
        "parity": _axis(PARITY_KEYS, baseline_summary, mitigated_summary),
        "safety": _axis(SAFETY_KEYS, baseline_summary, mitigated_summary),
    }
    axes["safety"]["global_fnr"] = {
        "baseline": baseline_global.get("fnr"),
        "mitigated": mitigated_global.get("fnr"),
        "delta": _delta(mitigated_global.get("fnr"), baseline_global.get("fnr")),
    }

    degenerate = bool(mitigated_global.get("degenerate") or baseline_global.get("degenerate"))
    verdict = decide_verdict(axes, ctx.config, degenerate)

    table = _headline_table(axes)
    io.write_table(table, ctx.table_path("tab02_headline_comparison"))
    ctx.save_metrics(
        "compare",
        {
            "stage": "compare",
            "axes": axes,
            "verdict": verdict,
            "degenerate": degenerate,
            "split_fingerprint": baseline_result.get("split_fingerprint"),
            "training_fingerprint": baseline_result.get("training_fingerprint"),
            "baseline_summary": baseline_summary,
            "mitigated_summary": mitigated_summary,
        },
    )

    _make_figures(ctx, baseline_global, mitigated_global)
    _log_verdict(axes, verdict)


def _headline_table(axes: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "axis": axis,
            "metric": name,
            "baseline": values["baseline"],
            "mitigated": values["mitigated"],
            "delta": values["delta"],
        }
        for axis, metrics in axes.items()
        for name, values in metrics.items()
    ]
    frame = pd.DataFrame(rows)
    for column in ("baseline", "mitigated", "delta"):
        frame[column] = frame[column].map(
            lambda v: round(v, 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
        )
    return frame


def safety_signal(axes: dict[str, Any]) -> tuple[float | None, str]:
    """The change in missed abuse, and which measurement it came from.

    Subgroup FNR is preferred, but it is undefined when no subgroup clears the
    FNR floor. Falling through on None was a real failure: on a 20k-row run the
    mitigated model missed 7.4 points more abuse and the verdict still read
    "B: utility trade-off" because the safety branch had been skipped. Global FNR
    now stands in, and the verdict names which was used.
    """
    macro = axes["safety"]["macro_fnr"]["delta"]
    if macro is not None:
        return macro, "subgroup"
    return axes["safety"]["global_fnr"]["delta"], "global"


def decide_verdict(axes: dict[str, Any], config: Any, degenerate: bool = False) -> str:
    """Name which of the pre-registered outcomes the run landed on.

    The two tolerances live in the config, not in this function, so they are
    fixed in a file that predates the results. Choosing them after seeing the
    numbers would be tuning the conclusion.
    """
    utility_tolerance = float(config.get("verdict.utility_tolerance", 0.01))
    safety_tolerance = float(config.get("verdict.safety_tolerance", 0.02))

    # A model predicting one class has a perfect FPR gap in every subgroup. That
    # is collapse, not fairness, and it must never be written up as an outcome.
    if degenerate:
        return (
            "INVALID: a model collapsed to a single class, so its parity numbers are an "
            "artefact. Lower mitigation.alpha, or train longer, and re-run that arm."
        )

    fpr_gap = axes["parity"]["fpr_gap"]["delta"]
    macro_f1 = axes["utility"]["macro_f1"]["delta"]
    fnr_delta, fnr_source = safety_signal(axes)

    if fpr_gap is None or macro_f1 is None:
        return "indeterminate: the metrics needed for a verdict are unavailable"
    if fpr_gap >= 0:
        return "D: null or adverse -- the weighting did not reduce the FPR gap"
    if fnr_delta is None:
        return (
            "indeterminate: parity improved, but no false-negative rate could be measured, "
            "so the safety cost of the intervention is unknown"
        )
    if fnr_delta > safety_tolerance:
        return (
            f"C: safety trade-off -- parity improved, but {fnr_source} FNR rose by "
            f"{fnr_delta:+.4f} (> {safety_tolerance}), so more genuine abuse is missed"
        )
    if macro_f1 > -utility_tolerance:
        return (
            f"A: near-Pareto -- parity improved at a macro-F1 cost of {macro_f1:+.4f} "
            f"(within {utility_tolerance})"
        )
    return f"B: utility trade-off -- parity improved, macro-F1 fell by {macro_f1:+.4f}"


def _make_figures(ctx, baseline_global: dict, mitigated_global: dict) -> None:
    baseline_table = ctx.take("audit_baseline_table")
    mitigated_table = ctx.take("audit_mitigated_table")
    if baseline_table is None or baseline_table.empty:
        logger.warning("no audit table in context; skipping figures")
        return

    min_support = int(ctx.config.get("evaluation.min_support", 100))
    plotting.fig_subgroup_support(
        baseline_table, ctx.figure_path("fig02_subgroup_support"), min_support
    )
    plotting.fig_fpr_by_subgroup(
        baseline_table, mitigated_table, ctx.figure_path("fig05_fpr_by_subgroup")
    )
    plotting.fig_fnr_by_subgroup(
        baseline_table, mitigated_table, ctx.figure_path("fig06_fnr_by_subgroup")
    )
    plotting.fig_confusion(
        baseline_global, ctx.figure_path("fig03_confusion_baseline"), "Baseline"
    )
    plotting.fig_confusion(
        mitigated_global, ctx.figure_path("fig04_confusion_mitigated"), "Mitigated"
    )


def _log_verdict(axes: dict[str, Any], verdict: str) -> None:
    def show(axis: str, key: str) -> str:
        values = axes[axis][key]
        parts = []
        for name in ("baseline", "mitigated", "delta"):
            value = values[name]
            usable = isinstance(value, (int, float)) and not isinstance(value, bool)
            parts.append(f"{name}={value:.4f}" if usable else f"{name}=n/a")
        return f"  {key:<14} " + "  ".join(parts)

    logger.info("=" * 74)
    logger.info("HEADLINE COMPARISON")
    for axis in ("utility", "parity", "safety"):
        logger.info("%s", axis.upper())
        for key in axes[axis]:
            logger.info("%s", show(axis, key))
    logger.info("verdict -> %s", verdict)
    logger.info("=" * 74)

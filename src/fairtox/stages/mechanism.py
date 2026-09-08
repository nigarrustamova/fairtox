"""Experiment 8 -- the mechanism, and the two objections it has to survive.

The audit says *that* the gap moved. This says *why*, from saved predictions and
with no GPU: it fits subgroup FPR against the subgroup's toxic rate (the shortcut,
measured), compares the arms at matched global false-positive rates (so a model
that simply flags less cannot look fairer), and names whether the gap closed from
the top or the bottom.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..registry import Tier, stage
from ..utils import io, plotting
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Two points make a line and say nothing; the correlation is undefined below three.
MIN_POINTS_FOR_FIT = 3


def toxic_rate_fit(subgroups: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Least-squares fit of subgroup FPR against subgroup toxic rate.

    Only rows clearing the FPR floor are used, and the toxic rate is measured on
    the same rows the FPR was -- the test split. Mixing in a corpus-wide rate
    would correlate two different populations and overstate the fit.
    """
    usable = [
        row for row in subgroups
        if row.get("reportable_fpr") and row.get("fpr") is not None and row.get("n")
    ]
    if len(usable) < MIN_POINTS_FOR_FIT:
        logger.warning(
            "only %d subgroup(s) clear the FPR floor; the toxic-rate fit needs %d and is skipped",
            len(usable), MIN_POINTS_FOR_FIT,
        )
        return None

    toxic_rate = np.array([row["n_toxic"] / row["n"] for row in usable], dtype=float)
    fpr = np.array([row["fpr"] for row in usable], dtype=float)

    slope, intercept = (float(v) for v in np.polyfit(toxic_rate, fpr, 1))
    correlation = (
        float(np.corrcoef(toxic_rate, fpr)[0, 1])
        if toxic_rate.std() > 0 and fpr.std() > 0
        else None
    )
    return {
        "n_subgroups": len(usable),
        "slope": slope,
        "intercept": intercept,
        "pearson_r": correlation,
        "toxic_rate_min": float(toxic_rate.min()),
        "toxic_rate_max": float(toxic_rate.max()),
        "subgroups": [row["subgroup"] for row in usable],
    }


def matched_operating_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The mitigated gap against the baseline gap at the same global FPR.

    Points outside the baseline's measured range are marked ``comparable: false``
    and excluded, never clamped: ``np.interp`` holds its endpoint value beyond
    either end, which would present an extrapolation as a measurement -- at
    exactly the end where a conservative mitigated model lands.
    """
    baseline = sorted(
        (r for r in rows if r.get("model") == "baseline" and r.get("fpr_gap") is not None),
        key=lambda r: r["global_fpr"],
    )
    mitigated = sorted(
        (r for r in rows if r.get("model") == "mitigated" and r.get("fpr_gap") is not None),
        key=lambda r: r["global_fpr"],
    )
    if len(baseline) < 2 or not mitigated:
        logger.warning("not enough threshold points to match operating points; skipping")
        return []

    xs = np.array([r["global_fpr"] for r in baseline], dtype=float)
    ys = np.array([r["fpr_gap"] for r in baseline], dtype=float)
    low, high = float(xs.min()), float(xs.max())

    matched = []
    for row in mitigated:
        x = float(row["global_fpr"])
        comparable = low <= x <= high
        baseline_gap = float(np.interp(x, xs, ys))
        reduction = (
            float(1.0 - row["fpr_gap"] / baseline_gap)
            if comparable and baseline_gap > 0
            else None
        )
        matched.append(
            {
                "threshold": row.get("threshold"),
                "global_fpr": x,
                "mitigated_gap": float(row["fpr_gap"]),
                "baseline_gap_at_same_fpr": baseline_gap,
                "gap_reduction": reduction,
                "comparable": comparable,
            }
        )
        if not comparable:
            logger.warning(
                "threshold %.2f puts the mitigated model at a global FPR of %.4f, outside the "
                "baseline's measured range [%.4f, %.4f]; reported but excluded from the range",
                row.get("threshold", float("nan")), x, low, high,
            )
    return matched


def gap_spread(rows: list[dict[str, Any]], model: str) -> float | None:
    """How much a model's FPR gap moves across the threshold sweep.

    A deployed moderation system retunes its threshold constantly, so a disparity
    that barely moves with it is a more useful claim than one measured at 0.5.
    """
    gaps = [r["fpr_gap"] for r in rows if r.get("model") == model and r.get("fpr_gap") is not None]
    return float(max(gaps) - min(gaps)) if len(gaps) > 1 else None


def levelling_direction(baseline: dict[str, Any], mitigated: dict[str, Any]) -> dict[str, Any]:
    """Did the gap close from the top, or from the bottom?

    ``max - min`` falls both when the worst-off subgroup improves and when the
    best-off one deteriorates. Only the first is a fairness gain, so the direction
    is named rather than left for the reader to reconstruct.
    """
    worst_before = (baseline.get("worst_subgroup") or {}).get("fpr")
    best_before = (baseline.get("best_subgroup") or {}).get("fpr")
    worst_after = (mitigated.get("worst_subgroup") or {}).get("fpr")
    best_after = (mitigated.get("best_subgroup") or {}).get("fpr")

    if None in (worst_before, best_before, worst_after, best_after):
        return {"direction": "indeterminate", "reason": "a worst or best subgroup is missing"}

    worst_improved = worst_after < worst_before
    best_worsened = best_after > best_before

    if worst_improved and not best_worsened:
        direction, reason = "levelling up", "the worst-off subgroup improved"
    elif worst_improved and best_worsened:
        direction, reason = "mixed", "the worst-off subgroup improved and the best-off got worse"
    elif not worst_improved and best_worsened:
        direction, reason = (
            "levelling down",
            "the worst-off subgroup did not improve; the gap closed because the best-off "
            "subgroup got worse",
        )
    else:
        direction, reason = "no movement", "neither end moved in the expected direction"

    return {
        "direction": direction,
        "reason": reason,
        "worst_fpr_before": float(worst_before),
        "worst_fpr_after": float(worst_after),
        "best_fpr_before": float(best_before),
        "best_fpr_after": float(best_after),
    }


@stage(
    "mechanism_analysis",
    tier=Tier.OPTIONAL,
    summary="Experiment 8: shortcut regression, matched operating points, levelling direction.",
    depends_on=("compare", "threshold_sensitivity"),
)
def mechanism_analysis(ctx) -> None:
    audits = {}
    for model in ("baseline", "mitigated"):
        path = ctx.metrics_path(f"audit_{model}")
        if not path.exists():
            raise FileNotFoundError(
                f"mechanism_analysis needs {path.name}. Run the comparison first:\n"
                "  python scripts/run_all.py --only compare"
            )
        audits[model] = io.load_json(path)

    fits = {model: toxic_rate_fit(payload["subgroups"]) for model, payload in audits.items()}

    fit_rows = [
        {
            "model": model,
            "n_subgroups": fit["n_subgroups"],
            "slope": fit["slope"],
            "intercept": fit["intercept"],
            "pearson_r": fit["pearson_r"],
        }
        for model, fit in fits.items()
        if fit is not None
    ]
    io.write_table(pd.DataFrame(fit_rows), ctx.table_path("tab10_mechanism_fit"))

    slope_change = None
    if fits["baseline"] and fits["mitigated"] and fits["baseline"]["slope"] != 0:
        slope_change = float(fits["mitigated"]["slope"] / fits["baseline"]["slope"] - 1.0)

    sensitivity_path = ctx.metrics_path("threshold_sensitivity")
    matched: list[dict[str, Any]] = []
    spreads: dict[str, float | None] = {}
    if sensitivity_path.exists():
        rows = io.load_json(sensitivity_path).get("rows", [])
        matched = matched_operating_points(rows)
        spreads = {model: gap_spread(rows, model) for model in ("baseline", "mitigated")}
        io.write_table(pd.DataFrame(matched), ctx.table_path("tab11_matched_operating_points"))
    else:
        logger.warning(
            "no threshold_sensitivity metrics on disk, so the matched-operating-point "
            "comparison is skipped. Produce it with:\n"
            "  python scripts/run_all.py --only threshold_sensitivity --no-deps"
        )

    levelling = levelling_direction(
        audits["baseline"]["summary"], audits["mitigated"]["summary"]
    )

    usable = [m["gap_reduction"] for m in matched if m["gap_reduction"] is not None]
    ctx.save_metrics(
        "mechanism_analysis",
        {
            "stage": "mechanism_analysis",
            "toxic_rate_fit": fits,
            "slope_change": slope_change,
            "matched_operating_points": matched,
            "matched_reduction_min": min(usable) if usable else None,
            "matched_reduction_max": max(usable) if usable else None,
            "n_comparable_points": len(usable),
            "gap_spread": spreads,
            "levelling": levelling,
            "note": (
                "The toxic rate is measured on the test split, the same rows the FPR is "
                "measured on. Matched operating points are interpolated within the baseline's "
                "observed range only. A least-squares slope describes the association; it is "
                "not a causal estimate."
            ),
        },
    )

    if fits["baseline"] and fits["mitigated"]:
        plotting.fig_mechanism(
            audits["baseline"]["subgroups"],
            audits["mitigated"]["subgroups"],
            fits["baseline"],
            fits["mitigated"],
            ctx.figure_path("fig10_mechanism"),
        )

    _log_summary(fits, slope_change, usable, spreads, levelling)


def _log_summary(
    fits: dict[str, Any],
    slope_change: float | None,
    reductions: list[float],
    spreads: dict[str, float | None],
    levelling: dict[str, Any],
) -> None:
    logger.info("=" * 74)
    logger.info("MECHANISM ANALYSIS")
    for model, fit in fits.items():
        if fit is None:
            logger.info("  %-10s no fit (too few reportable subgroups)", model)
            continue
        logger.info(
            "  %-10s FPR = %.3f * toxic_rate %+.4f | r=%s | %d subgroups",
            model, fit["slope"], fit["intercept"],
            f"{fit['pearson_r']:.3f}" if fit["pearson_r"] is not None else "n/a",
            fit["n_subgroups"],
        )
    if slope_change is not None:
        logger.info("  slope change %+.1f%% -- this number is the mechanism claim",
                    100 * slope_change)
    if reductions:
        logger.info(
            "  matched operating points: gap %.1f%%-%.1f%% smaller at the same global FPR "
            "(%d comparable point(s))",
            100 * min(reductions), 100 * max(reductions), len(reductions),
        )
    if spreads.get("baseline") and spreads.get("mitigated"):
        logger.info(
            "  gap spread across thresholds: baseline %.4f vs mitigated %.4f (%.1fx narrower)",
            spreads["baseline"], spreads["mitigated"],
            spreads["baseline"] / spreads["mitigated"],
        )
    logger.info("  gap closed by %s -- %s", levelling["direction"], levelling["reason"])
    logger.info("=" * 74)

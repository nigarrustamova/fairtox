"""Optional-tier analyses.

These re-read saved probabilities: no checkpoint, no GPU, no booked window.
That makes them the cheapest quality-per-minute in the project and the safest
thing to run on a laptop the night before the deadline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..evaluation.metrics import classification_metrics
from ..evaluation.predictions import has_predictions, identity_columns_in, load_predictions
from ..fairness import auditor, jigsaw_metrics
from ..registry import Tier, stage
from ..utils import io
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

MODELS = ("baseline", "mitigated", "tfidf")


@stage(
    "threshold_sensitivity",
    tier=Tier.OPTIONAL,
    summary="Re-run the audit across thresholds; tests whether the finding is an artefact.",
    depends_on=("compare",),
)
def threshold_sensitivity(ctx) -> None:
    """Does the disparity finding survive a different decision threshold?

    Binarising toxicity at 0.5 is a choice, not a fact. If the FPR gap exists
    only at that one threshold, the claim is an artefact of the choice; if it
    holds across the range, the finding is robust. Either answer is worth
    reporting and neither costs a GPU-second.
    """
    thresholds = [
        float(t) for t in ctx.config.get("sensitivity.thresholds", [0.3, 0.4, 0.5, 0.6, 0.7])
    ]
    models = [m for m in ("baseline", "mitigated") if has_predictions(ctx.run_name, m, "test")]
    if not models:
        raise FileNotFoundError("threshold_sensitivity needs at least one model's predictions")

    rows = []
    for model_name in models:
        predictions = load_predictions(ctx.run_name, model_name, "test")
        identity_columns = identity_columns_in(predictions)
        labels = predictions["label"].to_numpy()
        probs = predictions["prob"].to_numpy()

        for threshold in thresholds:
            _, summary = auditor.audit(predictions, identity_columns, ctx.config, threshold)
            global_metrics = classification_metrics(labels, probs, threshold)
            rows.append(
                {
                    "model": model_name,
                    "threshold": threshold,
                    "macro_f1": global_metrics.get("macro_f1"),
                    "global_fpr": global_metrics.get("fpr"),
                    "global_fnr": global_metrics.get("fnr"),
                    "fpr_gap": summary.get("fpr_gap"),
                    "macro_fpr": summary.get("macro_fpr"),
                    "macro_fnr": summary.get("macro_fnr"),
                    "degenerate": bool(summary.get("degenerate")),
                }
            )

    table = pd.DataFrame(rows)
    io.write_table(table, ctx.table_path("tab06_threshold_sensitivity"))
    ctx.save_metrics(
        "threshold_sensitivity",
        {"stage": "threshold_sensitivity", "thresholds": thresholds,
         "rows": table.to_dict(orient="records")},
    )

    for model_name in models:
        gaps = pd.to_numeric(
            table[table["model"] == model_name]["fpr_gap"], errors="coerce"
        ).dropna()
        if len(gaps):
            logger.info(
                "%s | FPR gap across tau %.1f-%.1f: min=%.4f max=%.4f (spread %.4f)",
                model_name, min(thresholds), max(thresholds),
                gaps.min(), gaps.max(), gaps.max() - gaps.min(),
            )


@stage(
    "jigsaw_bias_metrics",
    tier=Tier.OPTIONAL,
    summary="Subgroup AUC, BPSN-AUC and BNSP-AUC: the benchmark's own fairness metrics.",
    depends_on=("audit_baseline",),
)
def jigsaw_bias_metrics(ctx) -> None:
    """Report the metrics this benchmark was designed around.

    Threshold-free, so they complement the FPR analysis rather than restating
    it, and they make our numbers directly comparable to published work.
    """
    min_support = int(ctx.config.get("evaluation.min_support", 100))
    models = [m for m in MODELS if has_predictions(ctx.run_name, m, "test")]
    if not models:
        raise FileNotFoundError("jigsaw_bias_metrics needs at least one model's predictions")

    summaries = {}
    for model_name in models:
        predictions = load_predictions(ctx.run_name, model_name, "test")
        identity_columns = identity_columns_in(predictions)
        table = jigsaw_metrics.subgroup_aucs(predictions, identity_columns, min_support)

        overall = classification_metrics(
            predictions["label"].to_numpy(), predictions["prob"].to_numpy()
        ).get("roc_auc")
        summary = jigsaw_metrics.summarise(table, overall)
        summaries[model_name] = summary

        io.write_table(table, ctx.table_path(f"tab07_jigsaw_aucs_{model_name}"))
        logger.info(
            "%-9s | overall AUC=%s | worst-case: subgroup=%s BPSN=%s BNSP=%s | final=%s",
            model_name,
            _fmt(summary.get("overall_auc")),
            _fmt(summary.get("subgroup_auc_worst_mean")),
            _fmt(summary.get("bpsn_auc_worst_mean")),
            _fmt(summary.get("bnsp_auc_worst_mean")),
            _fmt(summary.get("final_bias_score")),
        )

    ctx.save_metrics("jigsaw_bias_metrics", {"stage": "jigsaw_bias_metrics", "models": summaries})


@stage(
    "seed_robustness",
    tier=Tier.OPTIONAL,
    summary="Aggregate the headline result across repeated seeds; needs no GPU.",
)
def seed_robustness(ctx) -> None:
    """Is the headline effect larger than the run-to-run noise?

    ``run.seed`` drives the split, the shuffling and the head initialisation, so
    each seed is an independent replication of the whole experiment, and within a
    seed the two arms still differ only in the weighting. The unit of analysis is
    therefore the **per-seed paired difference**, not the raw metric.

    No significance test: with three or four seeds a p-value would claim more
    than the sample supports. Mean, spread and sign agreement say what the
    evidence is.
    """
    runs = _discover_seed_runs(ctx)
    if not runs:
        logger.info(
            "no completed runs found to aggregate. Produce some with:\n"
            "  bash scripts/run_seeds.sh 42 43 44\n"
            "then re-run this stage. (Or list run names in analysis.seed_runs.)"
        )
        ctx.save_metrics("seed_robustness", {"stage": "seed_robustness", "n_runs": 0})
        return

    records = []
    for name in runs:
        payload = io.load_json(io.metrics_path(name, "compare"))
        axes = payload.get("axes", {})
        records.append(
            {
                "run": name,
                "verdict": str(payload.get("verdict", ""))[:1],
                "degenerate": bool(payload.get("degenerate")),
                "d_fpr_gap": _dig(axes, "parity", "fpr_gap"),
                "d_macro_fpr": _dig(axes, "parity", "macro_fpr"),
                "d_eod_fpr": _dig(axes, "parity", "eod_fpr"),
                "d_macro_f1": _dig(axes, "utility", "macro_f1"),
                "d_macro_fnr": _dig(axes, "safety", "macro_fnr"),
            }
        )

    table = pd.DataFrame(records).sort_values("run").reset_index(drop=True)
    io.write_table(table, ctx.table_path("tab09_seed_robustness"))

    metrics = ("d_fpr_gap", "d_macro_fpr", "d_eod_fpr", "d_macro_f1", "d_macro_fnr")
    summary = {name: _aggregate(table, name) for name in metrics}

    verdicts = sorted({v for v in table["verdict"].tolist() if v})
    consistency = {
        "n_runs": int(len(table)),
        "verdicts": verdicts,
        # The most readable robustness statement available: either every seed
        # landed on the same pre-registered outcome, or they did not.
        "verdict_is_consistent": len(verdicts) == 1,
        "any_degenerate": bool(table["degenerate"].any()),
    }

    ctx.save_metrics(
        "seed_robustness",
        {
            "stage": "seed_robustness",
            "n_runs": int(len(table)),
            "runs": runs,
            "consistency": consistency,
            "summary": summary,
            "per_run": table.to_dict(orient="records"),
            "note": (
                "Each seed re-partitions the corpus and re-initialises the head, so the runs "
                "are independent replications. The unit of analysis is the per-seed paired "
                "difference. No significance test is reported: the number of seeds cannot "
                "support one."
            ),
        },
    )
    _log_seed_summary(table, summary, consistency)


def _discover_seed_runs(ctx) -> list[str]:
    """Run names to aggregate: an explicit list, else a glob over ``results/``.

    Auto-discovery is the default because the manual list is the step that gets
    forgotten -- and a forgotten list looks exactly like "we only ran one seed",
    which is the criticism this stage exists to answer.
    """
    explicit = [str(n) for n in (ctx.config.get("analysis.seed_runs", []) or [])]
    if explicit:
        found = [n for n in explicit if io.metrics_path(n, "compare").exists()]
        for missing in sorted(set(explicit) - set(found)):
            logger.warning("no compare.json for run '%s'; excluding it", missing)
        return found

    pattern = str(ctx.config.get("analysis.seed_run_glob", "*"))
    candidates = sorted(
        path.parent.parent.name
        for path in io.RESULTS_DIR.glob(f"{pattern}/metrics/compare.json")
    )
    if candidates:
        logger.info("discovered %d completed run(s): %s", len(candidates), ", ".join(candidates))
    return candidates


def _aggregate(table: pd.DataFrame, column: str) -> dict[str, Any]:
    """Mean, spread and sign agreement for one per-seed delta."""
    values = pd.to_numeric(table[column], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None,
                "n_negative": 0, "sign_agreement": None}

    n_negative = int((values < 0).sum())
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        # Sample standard deviation. With three seeds the population form
        # understates the spread, and understating it is the failure that matters.
        "sd": float(values.std(ddof=1)) if values.size > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
        "n_negative": n_negative,
        # How many seeds moved the metric the same way. For a handful of
        # replications this is more informative than any p-value would be.
        "sign_agreement": float(max(n_negative, values.size - n_negative) / values.size),
    }


def _log_seed_summary(table: pd.DataFrame, summary: dict, consistency: dict) -> None:
    n = consistency["n_runs"]
    logger.info("=" * 78)
    logger.info("SEED ROBUSTNESS across %d run(s): %s", n, ", ".join(table["run"].tolist()))
    logger.info("  %-14s %9s %9s %9s %9s  %s", "delta", "mean", "sd", "min", "max", "same sign")

    for name, stats in summary.items():
        if not stats["n"]:
            continue
        logger.info(
            "  %-14s %9.4f %9s %9.4f %9.4f  %d/%d",
            name, stats["mean"],
            f"{stats['sd']:.4f}" if stats["sd"] is not None else "n/a",
            stats["min"], stats["max"],
            max(stats["n_negative"], stats["n"] - stats["n_negative"]), stats["n"],
        )

    if n < 2:
        logger.warning(
            "only %d run(s): at least 2 are needed to say anything about seed variation, and "
            "3 is the usual minimum for a spread worth quoting.", n,
        )
        return

    if consistency["any_degenerate"]:
        logger.warning("at least one run contains a COLLAPSED model; fix it before aggregating")

    if consistency["verdict_is_consistent"]:
        logger.info("every seed landed on outcome '%s'", consistency["verdicts"][0])
    else:
        logger.warning(
            "seeds disagree on the outcome (%s). Report the disagreement -- it is the finding "
            "-- rather than picking the seed you prefer.",
            ", ".join(consistency["verdicts"]),
        )

    gap = summary["d_fpr_gap"]
    if gap["mean"] is not None and gap["sd"] is not None:
        agreed = max(gap["n_negative"], gap["n"] - gap["n_negative"])
        if abs(gap["mean"]) < gap["sd"]:
            logger.warning(
                "the mean FPR-gap change (%+.4f) is SMALLER than its spread across seeds "
                "(%.4f). Report it as inconclusive, not as an effect.",
                gap["mean"], gap["sd"],
            )
        else:
            logger.info(
                "FPR-gap change %+.4f exceeds its spread (%.4f) and agreed in sign by %d/%d seeds",
                gap["mean"], gap["sd"], agreed, gap["n"],
            )
    logger.info("=" * 78)


def _dig(axes: dict, axis: str, key: str) -> float | None:
    value = axes.get(axis, {}).get(key, {}).get("delta")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _fmt(value) -> str:
    usable = isinstance(value, (int, float)) and not isinstance(value, bool)
    return f"{value:.4f}" if usable else "n/a"

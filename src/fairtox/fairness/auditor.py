"""Subgroup error-rate audit.

The central quantity is the per-subgroup false-positive rate: of the comments
that mention group *k* and are genuinely non-toxic, what fraction did the
classifier flag?

Three guards keep it honest. **Separate support floors** for FPR and FNR, because
FPR is estimated on a subgroup's non-toxic rows and FNR on its toxic ones, and
here those counts differ by an order of magnitude. **Wilson intervals**, because
at zero observed false positives a percentile bootstrap returns [0, 0] whatever
the sample size. **Collapse detection**, because a model that flags nothing has
perfect parity in every subgroup.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.metrics import is_degenerate
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Chosen over the normal approximation and over the percentile bootstrap
    because it stays sensible when the estimate sits on a boundary: at zero
    observed false positives it still widens as the sample shrinks, which is the
    behaviour a subgroup with 120 non-toxic rows deserves.
    """
    if trials <= 0:
        return (float("nan"), float("nan"))
    z = NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (p_hat + z * z / (2 * trials)) / denominator
    half = (z / denominator) * np.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials))
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def bootstrap_interval(
    successes: int, trials: int, rng: np.random.Generator, n_samples: int, confidence: float
) -> tuple[float, float]:
    """Percentile bootstrap for a rate, kept for comparison with Wilson.

    Resampling a 0/1 vector with replacement and taking the mean is exactly a
    Binomial(n, p-hat)/n draw, so we sample that directly: identical
    distribution, O(n_samples) instead of O(n_samples x n).
    """
    if trials <= 0 or n_samples <= 0:
        return (float("nan"), float("nan"))
    p_hat = successes / trials
    draws = rng.binomial(trials, p_hat, size=n_samples) / trials
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return (float(low), float(high))


def _interval(
    successes: int, trials: int, config_method: str, rng: np.random.Generator,
    n_samples: int, confidence: float,
) -> tuple[float | None, float | None]:
    if trials <= 0:
        return (None, None)
    if config_method == "bootstrap":
        return bootstrap_interval(successes, trials, rng, n_samples, confidence)
    return wilson_interval(successes, trials, confidence)


def _group_rates(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    ci_method: str,
    rng: np.random.Generator,
    n_samples: int,
    confidence: float,
) -> dict[str, Any]:
    """Confusion counts, error rates and intervals for one slice."""
    is_negative = labels == 0
    is_positive = labels == 1
    fp = int((is_negative & (predictions == 1)).sum())
    tn = int((is_negative & (predictions == 0)).sum())
    fn = int((is_positive & (predictions == 0)).sum())
    tp = int((is_positive & (predictions == 1)).sum())

    n_negative, n_positive = fp + tn, fn + tp
    fpr = fp / n_negative if n_negative else None
    fnr = fn / n_positive if n_positive else None

    fpr_low, fpr_high = _interval(fp, n_negative, ci_method, rng, n_samples, confidence)
    fnr_low, fnr_high = _interval(fn, n_positive, ci_method, rng, n_samples, confidence)

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / n_positive if n_positive else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )

    return {
        "n": int(len(labels)),
        "n_nontoxic": n_negative,
        "n_toxic": n_positive,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "fpr": fpr, "fpr_ci_low": fpr_low, "fpr_ci_high": fpr_high,
        "fnr": fnr, "fnr_ci_low": fnr_low, "fnr_ci_high": fnr_high,
        "precision": precision, "recall": recall, "f1": f1,
    }


def audit(
    predictions: pd.DataFrame,
    identity_columns: list[str],
    config: Any,
    threshold: float | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Audit one prediction file.

    Returns the per-subgroup table and the aggregate disparity summary.
    """
    threshold = float(
        config.get("evaluation.classification_threshold", 0.5) if threshold is None else threshold
    )
    min_support = int(config.get("evaluation.min_support", 100))
    ci_method = str(config.get("evaluation.ci_method", "wilson")).lower()
    n_samples = int(config.get("evaluation.bootstrap_samples", 2000))
    confidence = float(config.get("evaluation.confidence_level", 0.95))
    rng = np.random.default_rng(int(config.get("run.seed", 42)))

    if ci_method not in ("wilson", "bootstrap"):
        raise ValueError(f"evaluation.ci_method must be 'wilson' or 'bootstrap', got '{ci_method}'")

    labels = predictions["label"].to_numpy().astype(int)
    flagged = (predictions["prob"].to_numpy() >= threshold).astype(int)

    rates_kwargs = dict(
        ci_method=ci_method, rng=rng, n_samples=n_samples, confidence=confidence
    )
    global_rates = _group_rates(labels, flagged, **rates_kwargs)
    degenerate = is_degenerate(flagged)
    if degenerate:
        only = int(flagged[0]) if len(flagged) else -1
        logger.warning(
            "DEGENERATE MODEL: every prediction is '%s'. Its parity numbers are meaningless "
            "-- a classifier that flags nothing has a perfect FPR gap in every subgroup. "
            "Do not report these as a fairness result.",
            "toxic" if only == 1 else "non-toxic",
        )

    rows = []
    for column in identity_columns:
        if column not in predictions.columns:
            continue
        member = predictions[column].to_numpy().astype(int) == 1
        if not member.any():
            continue
        entry: dict[str, Any] = {"subgroup": column}
        entry.update(_group_rates(labels[member], flagged[member], **rates_kwargs))
        # FPR is estimated on non-toxic rows and FNR on toxic rows, so each gets
        # its own floor. Using n_nontoxic for both would let an FNR computed from
        # eleven toxic comments into the safety aggregate.
        entry["reportable_fpr"] = entry["n_nontoxic"] >= min_support
        entry["reportable_fnr"] = entry["n_toxic"] >= min_support
        rows.append(entry)

    table = pd.DataFrame(rows)
    if table.empty:
        logger.warning("no subgroups present in the prediction file; the audit is empty")
        return table, {
            "threshold": threshold, "global": global_rates, "min_support": min_support,
            "n_subgroups": 0, "n_reportable_fpr": 0, "n_reportable_fnr": 0,
            "degenerate": degenerate,
        }

    table = table.sort_values("n", ascending=False).reset_index(drop=True)
    summary = _summarise(table, global_rates, threshold, min_support, ci_method)
    summary["degenerate"] = degenerate
    _log_summary(table, summary, min_support)
    return table, summary


def _summarise(
    table: pd.DataFrame,
    global_rates: dict[str, Any],
    threshold: float,
    min_support: int,
    ci_method: str,
) -> dict[str, Any]:
    """Disparity aggregates, each computed over the subgroups entitled to it."""
    fpr_rows = table[table["reportable_fpr"] & table["fpr"].notna()]
    fnr_rows = table[table["reportable_fnr"] & table["fnr"].notna()]

    summary: dict[str, Any] = {
        "threshold": threshold,
        "min_support": min_support,
        "ci_method": ci_method,
        "n_subgroups": int(len(table)),
        "n_reportable_fpr": int(len(fpr_rows)),
        "n_reportable_fnr": int(len(fnr_rows)),
        "reportable_fpr_subgroups": fpr_rows["subgroup"].tolist(),
        "reportable_fnr_subgroups": fnr_rows["subgroup"].tolist(),
        "global": global_rates,
    }

    fpr_keys = ("fpr_gap", "fpr_ratio", "eod_fpr", "macro_fpr", "fpr_variance",
                "worst_subgroup", "best_subgroup")
    if fpr_rows.empty:
        logger.warning(
            "no subgroup clears the FPR support floor of %d non-toxic rows; gaps are undefined",
            min_support,
        )
        summary.update(dict.fromkeys(fpr_keys))
    else:
        fprs = fpr_rows["fpr"].to_numpy(dtype=float)
        worst_idx, best_idx = int(np.argmax(fprs)), int(np.argmin(fprs))
        global_fpr = global_rates.get("fpr")
        summary.update(
            {
                "fpr_gap": float(fprs.max() - fprs.min()),
                "fpr_ratio": float(fprs.max() / fprs.min()) if fprs.min() > 0 else None,
                "eod_fpr": (
                    float(np.max(np.abs(fprs - global_fpr))) if global_fpr is not None else None
                ),
                "macro_fpr": float(fprs.mean()),
                "fpr_variance": float(fprs.var(ddof=0)),
                "worst_subgroup": {
                    "name": str(fpr_rows.iloc[worst_idx]["subgroup"]),
                    "fpr": float(fprs[worst_idx]),
                    "ci": [
                        float(fpr_rows.iloc[worst_idx]["fpr_ci_low"]),
                        float(fpr_rows.iloc[worst_idx]["fpr_ci_high"]),
                    ],
                    "n_nontoxic": int(fpr_rows.iloc[worst_idx]["n_nontoxic"]),
                },
                "best_subgroup": {
                    "name": str(fpr_rows.iloc[best_idx]["subgroup"]),
                    "fpr": float(fprs[best_idx]),
                    "ci": [
                        float(fpr_rows.iloc[best_idx]["fpr_ci_low"]),
                        float(fpr_rows.iloc[best_idx]["fpr_ci_high"]),
                    ],
                    "n_nontoxic": int(fpr_rows.iloc[best_idx]["n_nontoxic"]),
                },
            }
        )

    # The safety side of the ledger. Mitigation that closes the FPR gap by
    # missing more genuine abuse is not a win, so FNR sits in the same summary
    # rather than in an appendix nobody reads.
    if fnr_rows.empty:
        summary.update({"fnr_gap": None, "macro_fnr": None, "worst_fnr_subgroup": None})
    else:
        fnrs = fnr_rows["fnr"].to_numpy(dtype=float)
        worst = int(np.argmax(fnrs))
        summary.update(
            {
                "fnr_gap": float(fnrs.max() - fnrs.min()),
                "macro_fnr": float(fnrs.mean()),
                "worst_fnr_subgroup": {
                    "name": str(fnr_rows.iloc[worst]["subgroup"]),
                    "fnr": float(fnrs[worst]),
                    "n_toxic": int(fnr_rows.iloc[worst]["n_toxic"]),
                },
            }
        )
    return summary


def _log_summary(table: pd.DataFrame, summary: dict[str, Any], min_support: int) -> None:
    global_fpr = summary["global"].get("fpr")
    logger.info(
        "audit | %d/%d subgroups clear the FPR floor, %d clear the FNR floor (N >= %d) | "
        "global FPR=%s",
        summary["n_reportable_fpr"], summary["n_subgroups"], summary["n_reportable_fnr"],
        min_support,
        # A genuine FPR of 0.0 is falsy, so this tests for None explicitly --
        # otherwise a flawless model would be logged as "n/a".
        f"{global_fpr:.4f}" if global_fpr is not None else "n/a",
    )
    if summary.get("fpr_gap") is not None:
        worst, best = summary["worst_subgroup"], summary["best_subgroup"]
        logger.info(
            "  FPR gap=%.4f | worst=%s %.4f [%.4f, %.4f] n=%s | best=%s %.4f | macro=%.4f",
            summary["fpr_gap"],
            worst["name"], worst["fpr"], worst["ci"][0], worst["ci"][1],
            f"{worst['n_nontoxic']:,}",
            best["name"], best["fpr"], summary["macro_fpr"],
        )
    excluded = table[~table["reportable_fpr"]]["subgroup"].tolist()
    if excluded:
        logger.info("  below the FPR floor (measured, but excluded from gaps): %s",
                    ", ".join(excluded))

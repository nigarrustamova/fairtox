"""Experiment 10 -- duplicate comments across the split, and what they cost.

The partition is made on rows, not on text, and the corpus contains the same
comment more than once: 4,289 of the 448k annotated rows share their text with
another row, and 771 of those texts appear in more than one split. So a fraction
of the test set was seen verbatim during training, and every absolute number the
paper quotes is measured partly on memorised rows.

Two things follow, and they must not be confused.

The CONTROLLED comparison is untouched. Both arms share one partition, so both
are inflated by exactly the same rows, and the paired difference -- which is what
the study reports -- carries no contamination at all.

The ABSOLUTE numbers are not untouched. "macro-F1 0.80" and "the baseline misses
42% of genuine abuse" are claims about performance, and they are measured on a
test set that is not fully unseen.

This stage measures the second effect instead of arguing about it: it finds the
test rows whose text also occurs in train, recomputes both arms without them, and
reports the difference. It needs no GPU and no retraining, because dropping rows
from a saved prediction file is all it takes.

Matching is on whitespace-collapsed lowercase text. Exact matching would miss the
copy-paste reposts that make up most of the duplication, and fuzzy matching would
put a similarity threshold -- another arbitrary knob -- inside a robustness check.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..evaluation.metrics import classification_metrics
from ..evaluation.predictions import identity_columns_in, load_predictions
from ..fairness import auditor
from ..registry import Tier, stage
from ..utils import io
from ..utils.logging_utils import get_logger
from ._common import get_corpus, get_splits

logger = get_logger(__name__)

MODELS = ("baseline", "mitigated")


def normalise(series: pd.Series) -> pd.Series:
    """Whitespace-collapsed lowercase text, for duplicate matching."""
    return (
        series.astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .str.lower()
    )


def contaminated_ids(splits: dict[str, pd.DataFrame]) -> tuple[set, dict[str, Any]]:
    """Test ids whose text also occurs in the training split, plus a summary.

    Label disagreement among duplicates is counted at the same time. It is not
    leakage -- it is the annotator noise this corpus is known for -- but it is the
    same scan, and the paper's error analysis already rests on the claim that a
    large share of the false positives are disputable labels.
    """
    train_text = normalise(splits["train"]["comment_text"])
    test = splits["test"].copy()
    test["_norm"] = normalise(test["comment_text"])

    train_set = set(train_text)
    overlap = test["_norm"].isin(train_set)
    ids = set(test.loc[overlap, "id"].tolist())

    train_frame = pd.DataFrame(
        {"_norm": train_text, "label": splits["train"]["label"].to_numpy()}
    )
    duplicated = train_frame[train_frame["_norm"].duplicated(keep=False)]
    inconsistent = int(
        (duplicated.groupby("_norm")["label"].nunique() > 1).sum()
    ) if len(duplicated) else 0

    summary = {
        "n_test": int(len(test)),
        "n_test_seen_in_train": int(overlap.sum()),
        "share_of_test": float(overlap.mean()) if len(test) else 0.0,
        "n_train_duplicate_rows": int(len(duplicated)),
        "n_duplicate_texts_with_conflicting_labels": inconsistent,
    }
    return ids, summary


def _row(model: str, subset: str, frame: pd.DataFrame, config: Any) -> dict[str, Any]:
    table, audit_summary = auditor.audit(frame, identity_columns_in(frame), config)
    del table
    metrics = classification_metrics(
        frame["label"].to_numpy(),
        frame["prob"].to_numpy(),
        float(config.get("evaluation.classification_threshold", 0.5)),
    )
    macro_fnr = audit_summary.get("macro_fnr")
    return {
        "model": model,
        "test_set": subset,
        "n_rows": int(len(frame)),
        "macro_f1": metrics.get("macro_f1"),
        "roc_auc": metrics.get("roc_auc"),
        "global_fnr": metrics.get("fnr"),
        "fpr_gap": audit_summary.get("fpr_gap"),
        "macro_fpr": audit_summary.get("macro_fpr"),
        "macro_fnr": macro_fnr,
        "n_reportable_fpr": audit_summary.get("n_reportable_fpr"),
    }


@stage(
    "leakage_check",
    tier=Tier.OPTIONAL,
    summary="Experiment 10: re-measure both arms with the test rows seen in training removed.",
    depends_on=("compare",),
)
def leakage_check(ctx) -> None:
    config = ctx.config
    splits = get_splits(ctx)
    get_corpus(ctx)  # ensures the corpus is loaded and cached for the split build

    ids, summary = contaminated_ids(splits)
    logger.info(
        "duplicate scan | %s of %s test rows (%.2f%%) also occur verbatim in train | "
        "%s duplicate training rows, %s texts carry conflicting labels",
        f"{summary['n_test_seen_in_train']:,}", f"{summary['n_test']:,}",
        100 * summary["share_of_test"], f"{summary['n_train_duplicate_rows']:,}",
        f"{summary['n_duplicate_texts_with_conflicting_labels']:,}",
    )

    rows: list[dict[str, Any]] = []
    for model in MODELS:
        full = load_predictions(ctx.run_name, model, "test")
        clean = full[~full["id"].isin(ids)].reset_index(drop=True)
        if clean.empty:
            raise ValueError(
                f"removing the contaminated rows left '{model}' with no test rows at all"
            )
        rows.append(_row(model, "full", full, config))
        rows.append(_row(model, "deduplicated", clean, config))

    table = pd.DataFrame(rows)
    io.write_table(table, ctx.table_path("tab16_leakage_check"))

    # The number the paper needs: does the CONTROLLED effect move when the
    # memorised rows are dropped? If it does not, the contamination is a caveat
    # on the absolute figures only, which is exactly what it should be.
    effects = {}
    for subset in ("full", "deduplicated"):
        part = table[table["test_set"] == subset].set_index("model")
        effects[subset] = {
            metric: _difference(part, metric)
            for metric in ("fpr_gap", "macro_fpr", "macro_f1", "macro_fnr")
        }

    ctx.save_metrics(
        "leakage_check",
        {
            "stage": "leakage_check",
            "duplicates": summary,
            "per_model": rows,
            "mitigation_effect": effects,
            "note": (
                "Matching is on whitespace-collapsed lowercase text. Both arms share one "
                "partition, so the paired mitigation effect is contaminated identically on "
                "both sides and the 'deduplicated' effect is the one to compare it against. "
                "Duplicate texts carrying conflicting labels are annotator disagreement, not "
                "leakage, and are counted here only because it is the same scan."
            ),
        },
    )
    _log_summary(effects, summary)


def _difference(part: pd.DataFrame, metric: str) -> float | None:
    try:
        baseline = part.loc["baseline", metric]
        mitigated = part.loc["mitigated", metric]
    except KeyError:
        return None
    if pd.isna(baseline) or pd.isna(mitigated):
        return None
    return float(mitigated - baseline)


def _log_summary(effects: dict[str, Any], summary: dict[str, Any]) -> None:
    logger.info("=" * 74)
    logger.info("LEAKAGE CHECK -- %s of %s test rows seen verbatim in training (%.2f%%)",
                f"{summary['n_test_seen_in_train']:,}", f"{summary['n_test']:,}",
                100 * summary["share_of_test"])
    logger.info("  %-14s %14s %14s %10s", "effect", "full test", "deduplicated", "shift")
    for metric in ("fpr_gap", "macro_fpr", "macro_f1", "macro_fnr"):
        full, clean = effects["full"].get(metric), effects["deduplicated"].get(metric)
        if full is None or clean is None:
            logger.info("  %-14s %14s %14s %10s", metric, "n/a", "n/a", "n/a")
            continue
        logger.info("  %-14s %+14.4f %+14.4f %+10.4f", metric, full, clean, clean - full)
    logger.info(
        "  The paired effect is what the paper reports; the absolute figures are the ones "
        "the duplication inflates."
    )
    logger.info("=" * 74)

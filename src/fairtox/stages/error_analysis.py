"""Experiment 6 -- the linguistic error taxonomy.

The regular expressions below are a triage aid, not a claim: a regex cannot tell
counter-speech from the thing it is speaking against. So the stage exports two
samples that are **not** interchangeable.

``tab04_errors_<model>``  the most confident mistakes -- for *reading*. Their
                          category mix is the mix of the confident tail and is
                          not a proportion of anything.
``tab12_coding_sample``   a uniform random draw with an empty ``manual_category``
                          column -- the only one whose counts are proportions.

The filled-in file is committed back at ``error_analysis.coding_file``, and the
stage then reports the coded counts beside the regex ones. Absent, it says so
rather than letting the paper inherit a promise the work did not keep.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..config import REPO_ROOT
from ..evaluation.predictions import has_predictions, identity_columns_in, load_predictions
from ..registry import Tier, stage
from ..utils import io
from ..utils.logging_utils import get_logger
from ._common import text_lookup

logger = get_logger(__name__)

# Deliberately conservative patterns: a candidate flagged here still needs a
# human to confirm it. Over-matching would inflate a category and mislead.
TAXONOMY: dict[str, re.Pattern] = {
    "counter_speech": re.compile(
        r"\b(?:against|oppose|opposing|fight(?:ing)?|stand up to|condemn|denounce|"
        r"call out|no place for|stop the)\b",
        re.IGNORECASE,
    ),
    "negation": re.compile(
        r"\b(?:not|never|no one|nobody|isn't|aren't|wasn't|weren't|don't|doesn't|"
        r"didn't|won't|can't|cannot|shouldn't)\b",
        re.IGNORECASE,
    ),
    "self_identification": re.compile(
        r"\b(?:as an?|i am|i'm|my|we are|we're|our)\s+\w+", re.IGNORECASE
    ),
    "quotation_or_report": re.compile(
        r"[\"“”]|\b(?:said|says|quoted|reported)\b", re.IGNORECASE
    ),
    "question": re.compile(r"\?\s*$"),
}


@stage(
    "error_analysis",
    tier=Tier.MUST,
    summary="Experiment 6: linguistic taxonomy of false positives and negatives.",
    depends_on=("compare",),
)
def error_analysis(ctx) -> None:
    config = ctx.config
    threshold = float(config.get("evaluation.classification_threshold", 0.5))
    n_examples = int(config.get("error_analysis.examples_per_category", 15))

    texts = text_lookup(ctx)
    frames = {
        name: load_predictions(ctx.run_name, name, "test")
        for name in ("baseline", "mitigated")
        if has_predictions(ctx.run_name, name, "test")
    }
    if not frames:
        raise FileNotFoundError("error_analysis needs at least one model's predictions")

    counts_rows = []
    for model_name, predictions in frames.items():
        errors = _classify_errors(predictions, texts, threshold)
        if errors.empty:
            logger.info("%s: no misclassifications on the test split", model_name)
            continue

        io.write_table(_sample(errors, n_examples), ctx.table_path(f"tab04_errors_{model_name}"))
        if model_name == "baseline":
            io.write_table(
                _coding_sample(
                    errors,
                    int(config.get("error_analysis.coding_sample_size", 60)),
                    int(config.get("error_analysis.coding_seed", 7)),
                ),
                ctx.table_path("tab12_coding_sample"),
            )
        for error_type in ("false_positive", "false_negative"):
            subset = errors[errors["error_type"] == error_type]
            row = {
                "model": model_name,
                "error_type": error_type,
                "n": int(len(subset)),
                "identity_bearing": int(subset["any_identity"].sum()) if len(subset) else 0,
            }
            for category in TAXONOMY:
                row[category] = int(subset[category].sum()) if len(subset) else 0
            counts_rows.append(row)

    counts = pd.DataFrame(counts_rows)
    io.write_table(counts, ctx.table_path("tab05_error_taxonomy_counts"))

    coded = _coded_counts(ctx)
    if coded is not None:
        io.write_table(coded, ctx.table_path("tab13_manual_coding"))

    ctx.save_metrics(
        "error_analysis",
        {
            "stage": "error_analysis",
            "threshold": threshold,
            "categories": list(TAXONOMY),
            "counts": counts.to_dict(orient="records"),
            "manual_coding": coded.to_dict(orient="records") if coded is not None else None,
            "note": (
                "The regex columns are a heuristic pre-sort and are reported as such. Any "
                "category count the paper states as a finding comes from the manually coded "
                "sample in tab13, which is drawn at random rather than from the confident "
                "tail; where that file is absent, only the heuristic counts exist and the "
                "paper must say so."
            ),
        },
    )
    _log_counts(counts, coded)


def _coded_counts(ctx) -> pd.DataFrame | None:
    """Read the committed manual coding, if there is one, and count it.

    A repository artefact, not a run artefact: it records human judgement, does
    not change when the pipeline re-runs, and is reviewable in a pull request.
    The ``regex_*`` columns say how often the pre-sort flagged a row a person put
    in each category -- a stated error rate for the heuristic.
    """
    relative = str(ctx.config.get("error_analysis.coding_file", "analysis/error_coding.csv"))
    path = Path(relative)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        logger.info(
            "no manual coding at %s; reporting the heuristic pre-sort only. Fill in "
            "tab12_coding_sample.csv and commit it there to add coded counts.",
            relative,
        )
        return None

    coded = pd.read_csv(path)
    if "manual_category" not in coded.columns or coded["manual_category"].isna().all():
        logger.warning("%s has no filled-in manual_category column; ignoring it", relative)
        return None

    coded = coded[coded["manual_category"].notna() & (coded["manual_category"] != "")]
    total = len(coded)
    rows = []
    for category, group in coded.groupby("manual_category"):
        row = {
            "manual_category": str(category),
            "n": int(len(group)),
            "share": round(len(group) / total, 4) if total else None,
        }
        # The second coded dimension, and the one the limitations section needs:
        # how many of these the dataset's own "non-toxic" label is arguable for.
        # A false positive on a comment a reasonable annotator would have called
        # toxic is annotator disagreement, not a model error, and the paper cannot
        # claim the whole FPR as the latter.
        if "label_arguable" in group.columns:
            row["label_arguable"] = int(
                pd.to_numeric(group["label_arguable"], errors="coerce").fillna(0).sum()
            )
        # How often the regex triage flagged a row a human put in this category.
        for heuristic in TAXONOMY:
            if heuristic in group.columns:
                row[f"regex_{heuristic}"] = int(pd.to_numeric(group[heuristic], errors="coerce")
                                                .fillna(0).sum())
        rows.append(row)

    table = pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)
    logger.info("manual coding: %d comment(s) across %d categories", total, len(table))
    if "label_arguable" in coded.columns:
        arguable = int(pd.to_numeric(coded["label_arguable"], errors="coerce").fillna(0).sum())
        logger.info(
            "  of those, %d (%.1f%%) are comments whose non-toxic label is arguable -- "
            "annotator disagreement rather than model error",
            arguable, 100 * arguable / total,
        )
    return table


def _coding_sample(errors: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    """A **random** sample of identity-bearing false positives, ready to hand-code.

    Deliberately not ``tab04``'s selection: counting categories over the confident
    tail would estimate the composition of the tail, not of the errors. Seeded
    from its own key, so changing ``run.seed`` cannot redraw a sample somebody has
    already coded.
    """
    candidates = errors[(errors["error_type"] == "false_positive") & (errors["any_identity"] == 1)]
    if candidates.empty:
        logger.warning("no identity-bearing false positives to sample for coding")
        return pd.DataFrame()

    take = min(int(size), len(candidates))
    sample = candidates.sample(n=take, random_state=int(seed)).sort_values("id")
    columns = ["id", "prob", "any_identity", "comment_text", *TAXONOMY]
    out = sample[[c for c in columns if c in sample.columns]].copy()
    out["prob"] = out["prob"].round(4)
    # Full text, not the 220-character preview tab04 uses: a coder has to be able
    # to read the whole comment before assigning it a category.
    out["manual_category"] = ""
    logger.info(
        "wrote a coding sample of %d identity-bearing false positive(s) drawn at random "
        "(seed %d) from %d candidates",
        take, seed, len(candidates),
    )
    return out


def _classify_errors(
    predictions: pd.DataFrame, texts: pd.Series, threshold: float
) -> pd.DataFrame:
    flagged = (predictions["prob"].to_numpy() >= threshold).astype(int)
    labels = predictions["label"].to_numpy().astype(int)
    mistaken = flagged != labels

    errors = predictions.loc[mistaken].copy()
    if errors.empty:
        return errors

    errors["predicted"] = flagged[mistaken]
    errors["error_type"] = ["false_positive" if lab == 0 else "false_negative"
                            for lab in errors["label"].to_numpy()]
    errors["comment_text"] = errors["id"].map(texts).fillna("")

    if "any_identity" not in errors.columns:
        identity_columns = identity_columns_in(errors)
        errors["any_identity"] = (
            errors[identity_columns].sum(axis=1).gt(0).astype(int) if identity_columns else 0
        )

    for category, pattern in TAXONOMY.items():
        errors[category] = errors["comment_text"].str.contains(pattern, na=False).astype(int)

    # Surface the most confident mistakes first: those are the ones that reveal
    # what the model has actually learned.
    errors["confidence"] = (errors["prob"] - threshold).abs()
    return errors.sort_values("confidence", ascending=False).reset_index(drop=True)


def _sample(errors: pd.DataFrame, per_category: int) -> pd.DataFrame:
    """A readable slice for manual coding."""
    picked = []
    for error_type in ("false_positive", "false_negative"):
        subset = errors[errors["error_type"] == error_type]
        if subset.empty:
            continue
        # Prioritise identity-bearing errors -- they are what the study is about.
        ordered = subset.sort_values(["any_identity", "confidence"], ascending=[False, False])
        picked.append(ordered.head(per_category))

    if not picked:
        return pd.DataFrame()
    sample = pd.concat(picked, ignore_index=True)
    columns = ["id", "error_type", "label", "prob", "any_identity", "comment_text", *TAXONOMY]
    sample = sample[[c for c in columns if c in sample.columns]].copy()
    sample["prob"] = sample["prob"].round(4)
    sample["comment_text"] = sample["comment_text"].str.slice(0, 220)
    return sample


def _log_counts(counts: pd.DataFrame, coded: pd.DataFrame | None = None) -> None:
    if not counts.empty:
        logger.info("error taxonomy (heuristic pre-sort):")
        for _, row in counts.iterrows():
            share = 100 * row["identity_bearing"] / row["n"] if row["n"] else 0.0
            logger.info(
                "  %-10s %-15s n=%-7s identity-bearing=%s (%.1f%%)",
                row["model"], row["error_type"], f"{row['n']:,}",
                f"{row['identity_bearing']:,}", share,
            )
    if coded is None or coded.empty:
        return
    logger.info("manual coding of a random identity-bearing false-positive sample:")
    for _, row in coded.iterrows():
        logger.info(
            "  %-24s n=%-4d %5.1f%%",
            row["manual_category"], int(row["n"]), 100 * float(row["share"]),
        )

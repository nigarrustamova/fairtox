"""Experiment 0 -- dataset profiling.

Every statistic the report quotes is computed here and written to ``results/``,
so no number in the paper is hand-copied or invented. Run this and read the
output *before* the training window opens: it is the only cheap chance to find
out that the corpus cannot support the study you planned.
"""

from __future__ import annotations

import pandas as pd

from ..data import split as data_split
from ..registry import Tier, stage
from ..utils import io
from ..utils.logging_utils import get_logger
from ._common import get_corpus, get_splits

logger = get_logger(__name__)


@stage(
    "eda",
    tier=Tier.MUST,
    summary="Profile the corpus: class balance, subgroup support, split sizes.",
)
def run_eda(ctx) -> None:
    config = ctx.config
    frame, identity_columns = get_corpus(ctx)
    splits = get_splits(ctx)
    min_support = int(config.get("evaluation.min_support", 100))

    # Support is counted on the *test* split, not the whole corpus. The audit
    # slices the test split, so that is the only count which decides whether a
    # subgroup's error rate means anything. Reporting the corpus-wide figure here
    # would promise several times the subgroups the paper can actually deliver.
    subgroup_rows = _subgroup_table(frame, splits["test"], identity_columns)
    reportable_fpr = _clearing(subgroup_rows, "test_n_nontoxic", min_support)
    reportable_fnr = _clearing(subgroup_rows, "test_n_toxic", min_support)
    ctx.put("reportable_subgroups", reportable_fpr)

    payload = {
        "stage": "eda",
        "n_total": int(len(frame)),
        "split_sizes": data_split.split_sizes(splits),
        "split_fingerprint": data_split.fingerprint(splits),
        "toxic_rate": float(frame["label"].mean()),
        "n_toxic": int(frame["label"].sum()),
        "n_nontoxic": int((frame["label"] == 0).sum()),
        "any_identity_rate": float(frame["any_identity"].mean()),
        "toxicity_threshold": float(config.get("data.toxicity_threshold", 0.5)),
        "identity_threshold": float(config.get("data.identity_threshold", 0.5)),
        "annotated_only": bool(config.get("data.annotated_only", True)),
        "subsample": config.get("data.subsample"),
        "min_support": min_support,
        "support_counted_on": "test split",
        "n_subgroups_total": len(identity_columns),
        "n_reportable_fpr": len(reportable_fpr),
        "n_reportable_fnr": len(reportable_fnr),
        "reportable_fpr_subgroups": reportable_fpr,
        "reportable_fnr_subgroups": reportable_fnr,
        "subgroups": subgroup_rows.to_dict(orient="records"),
    }
    ctx.save_metrics("eda", payload)
    io.write_table(subgroup_rows, ctx.table_path("tab01_subgroup_support"))

    logger.info(
        "%s rows | %.2f%% toxic | %d/%d subgroups clear the FPR floor and %d clear the FNR "
        "floor on the TEST split (N >= %d)",
        f"{len(frame):,}", 100 * frame["label"].mean(),
        len(reportable_fpr), len(identity_columns), len(reportable_fnr), min_support,
    )

    dropped = sorted(set(subgroup_rows["subgroup"]) - set(reportable_fpr))
    if dropped:
        logger.info("below the FPR floor, reported without claims: %s", ", ".join(dropped))

    # A fairness study that can only speak about its largest groups has lost its
    # argument, so say so loudly here rather than letting it surface as a thin
    # results section after the window has been spent.
    floor_warning = int(config.get("evaluation.min_reportable_subgroups", 6))
    if len(reportable_fpr) < floor_warning:
        logger.warning(
            "ONLY %d SUBGROUP(S) CAN CARRY AN FPR CLAIM. That is a thin basis for a fairness "
            "paper, and the groups that drop out are usually the ones the study is about. "
            "Before spending the training window: use the full corpus (remove data.subsample), "
            "raise data.splits.test, or lower evaluation.min_support and justify it in the "
            "paper -- but decide NOW, not after seeing results.",
            len(reportable_fpr),
        )


def _clearing(table: pd.DataFrame, column: str, min_support: int) -> list[str]:
    if table.empty:
        return []
    return table[table[column] >= min_support]["subgroup"].tolist()


def _subgroup_table(
    frame: pd.DataFrame, test_split: pd.DataFrame, identity_columns: list[str]
) -> pd.DataFrame:
    """Support counts per subgroup, corpus-wide and on the test split.

    The two test-split columns are what decide everything: FPR is estimated on
    non-toxic test rows and FNR on toxic ones, so those counts -- not the
    corpus-wide totals -- govern whether either rate means anything for a group.
    Both scales are reported because the gap between them is itself worth seeing
    when judging whether the corpus is large enough.
    """
    rows = []
    for column in identity_columns:
        member = frame[frame[column] == 1]
        test_member = test_split[test_split[column] == 1]
        rows.append(
            {
                "subgroup": column,
                "n": int(len(member)),
                "n_toxic": int(member["label"].sum()) if len(member) else 0,
                "n_nontoxic": int((member["label"] == 0).sum()) if len(member) else 0,
                "toxic_rate": round(float(member["label"].mean()), 4) if len(member) else 0.0,
                "test_n": int(len(test_member)),
                "test_n_nontoxic": (
                    int((test_member["label"] == 0).sum()) if len(test_member) else 0
                ),
                "test_n_toxic": int(test_member["label"].sum()) if len(test_member) else 0,
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("n", ascending=False).reset_index(drop=True)

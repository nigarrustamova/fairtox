"""Experiment 2 -- the subgroup fairness audit.

Reads a prediction file and reports per-group error rates. It never loads a
model, so it re-runs in seconds on any machine, at any threshold, long after the
training window has closed.
"""

from __future__ import annotations

from typing import Any

from ..evaluation.predictions import identity_columns_in, load_predictions
from ..fairness import auditor
from ..registry import Tier, stage
from ..utils import io
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def audit_model(
    ctx,
    model_name: str,
    split: str = "test",
    threshold: float | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    """Audit one model's predictions; returns the summary and caches the table.

    Shared by ``audit_baseline``, ``compare`` and the ablation so that every
    model in the study is measured by exactly the same code at exactly the same
    threshold. A comparison that straddled two measurement paths would not be
    one.
    """
    artifact = label or f"audit_{model_name}"
    predictions = load_predictions(ctx.run_name, model_name, split)
    identity_columns = identity_columns_in(predictions)

    table, summary = auditor.audit(predictions, identity_columns, ctx.config, threshold)
    summary.update({"model": model_name, "split": split, "n_rows": int(len(predictions))})

    io.write_table(table, ctx.table_path(artifact))
    ctx.save_metrics(
        artifact,
        {"stage": artifact, "summary": summary, "subgroups": table.to_dict(orient="records")},
    )
    ctx.put(f"{artifact}_table", table)
    ctx.put(f"{artifact}_summary", summary)
    return summary


@stage(
    "audit_baseline",
    tier=Tier.MUST,
    summary="Experiment 2: subgroup FPR/FNR with confidence intervals.",
    depends_on=("train_baseline",),
)
def audit_baseline(ctx) -> None:
    audit_model(ctx, "baseline")

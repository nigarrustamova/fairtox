"""Experiment 9 -- post-processing, the third place a mitigation can live.

Reweighting changes the training objective. Counterfactual swapping changes the
training data. This changes neither: it leaves a trained model exactly as it is
and moves the DECISION THRESHOLD per subgroup -- the false-positive side of the
classical equalized-odds post-processing of Hardt et al. (2016).

It earns its place three times over.

1. It costs no GPU. It reads the prediction files the training runs already
   wrote, so it applies to every model in every run, including the nine that are
   already finished and whose GPU window has closed.
2. It turns the reviewer's first objection into a measurement. ``mechanism``
   already answers the GLOBAL version -- "you only moved the threshold" -- by
   matching operating points. This answers the per-group version, which is the
   stronger one, because a per-group threshold is a real method and not a
   confound.
3. It fails in a way the other two do not, and the failure is the finding: a
   per-group threshold has to know which group a comment belongs to AT INFERENCE
   TIME. Reweighting and swapping consume identity annotations during training
   and never again. No table in this repository shows that difference, so it is
   written into the metrics payload and the log instead of left to the reader.

Two policies are reported side by side, because the difference between them is
exactly the levelling question the mechanism stage asks:

    equalize   move every subgroup onto the target, including the ones already
               below it -- textbook parity, achieved partly by flagging MORE
               benign comments about the best-served groups.
    cap        only raise the threshold of subgroups above the target, never
               lower one. Parity improves without any subgroup being made worse.

Thresholds are fitted on VALIDATION and applied to TEST. Fitting them on the
split they are then scored on would report the best achievable parity rather than
the parity a deployed system gets, and the distance between those two is largest
on exactly the small subgroups this study is about.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..evaluation.metrics import classification_metrics
from ..evaluation.predictions import identity_columns_in, load_predictions
from ..fairness import auditor
from ..registry import Tier, stage
from ..utils import io
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

POLICIES = ("equalize", "cap")


def global_fpr(frame: pd.DataFrame, threshold: float) -> float | None:
    """False-positive rate over every row, at one threshold."""
    labels = frame["label"].to_numpy().astype(int)
    nontoxic = labels == 0
    if not nontoxic.any():
        return None
    return float((frame["prob"].to_numpy()[nontoxic] >= threshold).mean())


def fit_group_thresholds(
    val: pd.DataFrame,
    identity_columns: list[str],
    target_fpr: float,
    min_support: int,
    global_threshold: float,
    policy: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """One threshold per subgroup, each putting that subgroup at ``target_fpr``.

    ``FPR_k(t)`` is the share of group *k*'s non-toxic scores at or above ``t``,
    so putting the group on the target means flagging ``m = floor(target * n)`` of
    them: the threshold is the m-th largest score. Rounding DOWN is deliberate --
    the achieved rate lands at or just under the target rather than over it, and
    overshooting on a subgroup with 120 non-toxic rows would be a parity claim the
    support cannot carry. Tied scores cannot be split, so a group whose scores are
    heavily tied lands above the target; that shows up in ``val_fpr_after``.

    Subgroups below the support floor get no threshold of their own. Fitting one
    from forty rows and applying it to a different split is not a mitigation, it
    is noise with a name.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy '{policy}' (expected one of {', '.join(POLICIES)})")

    labels = val["label"].to_numpy().astype(int)
    probs = val["prob"].to_numpy(dtype=float)

    thresholds: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for column in identity_columns:
        member = val[column].to_numpy().astype(int) == 1
        nontoxic = member & (labels == 0)
        support = int(nontoxic.sum())
        if support < min_support:
            rows.append(
                {
                    "subgroup": column, "policy": policy, "n_nontoxic_val": support,
                    "threshold": None, "val_fpr_before": None, "val_fpr_after": None,
                    "fitted": False,
                }
            )
            continue

        scores = probs[nontoxic]
        before = float((scores >= global_threshold).mean())
        descending = np.sort(scores)[::-1]
        allowed = int(np.floor(target_fpr * support))
        fitted = (
            # Nothing may be flagged, so the threshold has to sit strictly above
            # the highest score. `nextafter` is the smallest value that does;
            # 1.0 would still flag a score of exactly 1.0.
            float(np.nextafter(float(descending[0]), np.inf))
            if allowed == 0
            else float(descending[allowed - 1])
        )
        if policy == "cap":
            # Never move a subgroup's threshold DOWN. Lowering it flags more
            # benign comments about a group that was already well served, which
            # closes the gap by levelling down -- the failure this project named
            # in experiment A and must not commit itself.
            fitted = max(fitted, global_threshold)

        thresholds[column] = fitted
        rows.append(
            {
                "subgroup": column, "policy": policy, "n_nontoxic_val": support,
                "threshold": fitted, "val_fpr_before": before,
                "val_fpr_after": float((scores >= fitted).mean()), "fitted": True,
            }
        )

    if not thresholds:
        logger.warning(
            "no subgroup clears the support floor of %d non-toxic validation rows, so every "
            "row keeps the global threshold and post-processing is a no-op",
            min_support,
        )
    return thresholds, rows


def row_thresholds(
    frame: pd.DataFrame, thresholds: dict[str, float], global_threshold: float
) -> np.ndarray:
    """The threshold each row is judged at.

    Jigsaw's subgroups OVERLAP -- a comment can be flagged ``black`` and
    ``female`` at once -- so Hardt et al.'s disjoint-group construction does not
    apply unchanged and a row with two memberships has two candidate thresholds.
    The mean of them is used. That is a choice, not a derivation, and it is why
    the achieved per-group FPR does not land exactly on the target: under
    overlapping membership no per-row rule can put every group on the same FPR at
    once. That impossibility is a property of the setting, so it is reported
    rather than tuned away.
    """
    per_row = np.full(len(frame), float(global_threshold), dtype=float)
    if not thresholds:
        return per_row

    total = np.zeros(len(frame), dtype=float)
    count = np.zeros(len(frame), dtype=float)
    for column, threshold in thresholds.items():
        if column not in frame.columns:
            continue
        member = frame[column].to_numpy().astype(int) == 1
        total[member] += threshold
        count[member] += 1.0

    has_group = count > 0
    per_row[has_group] = total[has_group] / count[has_group]
    return per_row


def shift_to_global(frame: pd.DataFrame, per_row: np.ndarray) -> pd.DataFrame:
    """Recentre the scores so a single global 0.5 reproduces the per-row rule.

    ``prob - t_row + 0.5 >= 0.5`` is exactly ``prob >= t_row``, which lets the
    audit, the confidence intervals and the utility metrics run unchanged instead
    of being reimplemented for this stage. The shifted column is a decision
    variable and no longer a probability: it is never written to disk, and every
    threshold-free number in this stage is read off the ORIGINAL scores.
    """
    shifted = frame.copy()
    shifted["prob"] = frame["prob"].to_numpy(dtype=float) - per_row + 0.5
    return shifted


def evaluate(
    test: pd.DataFrame,
    identity_columns: list[str],
    config: Any,
    per_row: np.ndarray,
    threshold_free: dict[str, Any],
) -> dict[str, Any]:
    """Audit and score one threshold policy on the test split."""
    shifted = shift_to_global(test, per_row)
    table, summary = auditor.audit(shifted, identity_columns, config, threshold=0.5)
    labels = test["label"].to_numpy().astype(int)
    metrics = classification_metrics(labels, shifted["prob"].to_numpy(dtype=float), 0.5)

    # Post-processing moves the threshold and never the score, so it cannot
    # change a ranking metric. Carrying the model's own AUCs over says that in
    # the table; recomputing them on the shifted column would produce a number
    # that looks like a result and is an artefact of the shift.
    metrics["roc_auc"] = threshold_free.get("roc_auc")
    metrics["pr_auc"] = threshold_free.get("pr_auc")
    return {"table": table, "summary": summary, "metrics": metrics}


def _row(
    arm: str, model: str, policy: str, result: dict[str, Any], n_fitted: int
) -> dict[str, Any]:
    summary, metrics = result["summary"], result["metrics"]
    macro_fnr = summary.get("macro_fnr")
    return {
        "arm": arm,
        "model": model,
        "policy": policy,
        "n_thresholds_fitted": n_fitted,
        "fpr_gap": summary.get("fpr_gap"),
        "macro_fpr": summary.get("macro_fpr"),
        "eod_fpr": summary.get("eod_fpr"),
        "macro_f1": metrics.get("macro_f1"),
        "roc_auc": metrics.get("roc_auc"),
        "macro_fnr": macro_fnr,
        "global_fnr": metrics.get("fnr"),
        "safety_fnr": macro_fnr if macro_fnr is not None else metrics.get("fnr"),
        "fnr_source": "subgroup" if macro_fnr is not None else "global",
        # An arm can post a flawless gap by flagging almost nothing. The flag
        # rate beside it makes that visible, exactly as in the ablation table.
        "flag_rate": metrics.get("positive_rate_pred"),
        "degenerate": metrics.get("degenerate"),
    }


@stage(
    "postproc_thresholds",
    tier=Tier.OPTIONAL,
    summary="Experiment 9: per-subgroup decision thresholds, fitted on val, scored on test.",
    depends_on=("compare",),
)
def postproc_thresholds(ctx) -> None:
    config = ctx.config
    global_threshold = float(config.get("evaluation.classification_threshold", 0.5))
    min_support = int(config.get("evaluation.min_support", 100))

    rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    payload_models: dict[str, Any] = {}

    for model in ("baseline", "mitigated"):
        val = load_predictions(ctx.run_name, model, "val")
        test = load_predictions(ctx.run_name, model, "test")
        identity_columns = identity_columns_in(test)

        target = global_fpr(val, global_threshold)
        if target is None:
            raise ValueError(
                f"'{model}' has no non-toxic validation rows, so there is no target "
                "false-positive rate to move the subgroups onto"
            )

        labels = test["label"].to_numpy().astype(int)
        unmitigated = classification_metrics(
            labels, test["prob"].to_numpy(dtype=float), global_threshold
        )
        _, base_summary = auditor.audit(test, identity_columns, config, threshold=global_threshold)
        rows.append(
            _row(
                model, model, "global",
                {"summary": base_summary, "metrics": unmitigated}, 0,
            )
        )

        per_model: dict[str, Any] = {"target_val_fpr": target, "policies": {}}
        for policy in POLICIES:
            thresholds, detail = fit_group_thresholds(
                val, identity_columns, target, min_support, global_threshold, policy
            )
            threshold_rows.extend({"model": model, **entry} for entry in detail)
            per_row = row_thresholds(test, thresholds, global_threshold)
            result = evaluate(test, identity_columns, config, per_row, unmitigated)
            rows.append(_row(f"{model} + {policy}", model, policy, result, len(thresholds)))

            achieved = {
                str(entry["subgroup"]): float(entry["fpr"])
                for _, entry in result["table"].iterrows()
                if entry.get("reportable_fpr") and entry.get("fpr") is not None
            }
            per_model["policies"][policy] = {
                "thresholds": thresholds,
                "n_fitted": len(thresholds),
                "test_fpr_by_subgroup": achieved,
                "summary": result["summary"],
            }
        payload_models[model] = per_model

    comparison = pd.DataFrame(rows)
    io.write_table(comparison, ctx.table_path("tab14_postprocessing"))
    io.write_table(pd.DataFrame(threshold_rows), ctx.table_path("tab15_group_thresholds"))

    ctx.save_metrics(
        "postproc_thresholds",
        {
            "stage": "postproc_thresholds",
            "global_threshold": global_threshold,
            "min_support": min_support,
            "policies": list(POLICIES),
            "comparison": rows,
            "models": payload_models,
            "note": (
                "Thresholds are fitted on the validation split and scored on test. The target "
                "is each model's own global validation FPR, so the policy preserves how much "
                "that model flags overall and the comparison is not a flag-rate comparison in "
                "disguise. Subgroups overlap, so a per-row rule cannot put every subgroup on "
                "the target at once; the per-row threshold is the mean over the groups a row "
                "belongs to and the residual is reported rather than tuned away. "
                "DEPLOYMENT: this method needs the identity annotation of a comment at "
                "INFERENCE time. Loss reweighting and counterfactual augmentation use identity "
                "annotations during training only. That difference is not visible in any "
                "metric here and has to be stated wherever these numbers are quoted."
            ),
        },
    )
    _log_summary(rows)


def _log_summary(rows: list[dict[str, Any]]) -> None:
    logger.info("=" * 74)
    logger.info("POST-PROCESSING: per-subgroup thresholds (fitted on val, scored on test)")
    logger.info(
        "  %-22s %9s %9s %9s %9s", "arm", "fpr_gap", "macro_f1", "safe_fnr", "flag_rate"
    )
    for row in rows:
        logger.info(
            "  %-22s %9s %9s %9s %9s",
            row["arm"],
            f"{row['fpr_gap']:.4f}" if row["fpr_gap"] is not None else "n/a",
            f"{row['macro_f1']:.4f}" if row["macro_f1"] is not None else "n/a",
            f"{row['safety_fnr']:.4f}" if row["safety_fnr"] is not None else "n/a",
            f"{row['flag_rate']:.4f}" if row["flag_rate"] is not None else "n/a",
        )
    logger.info(
        "  Per-group thresholds need the identity annotation AT INFERENCE TIME; the "
        "training-time methods do not. Do not quote these rows without that sentence."
    )
    logger.info("=" * 74)

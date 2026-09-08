"""Experiment 5 -- the ablation.

Answers two defense questions by construction. *Is identity-awareness necessary?*
The ``class`` arm is the control that separates "we handled imbalance" from "we
handled identity". *How hard can we push?* The alpha sweep maps where the gain
stops being free and starts costing safety.

Arms whose weights come out uniform are not trained: they are arithmetically the
baseline, so the sweep reuses its predictions and keeps the GPU hour.
``ablation.alphas`` must contain ``mitigation.alpha``, or the trade-off figure
would omit the point the paper's claim sits on.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..evaluation.predictions import identity_columns_in, load_predictions
from ..fairness import auditor
from ..provenance import model_fingerprint
from ..registry import Tier, stage
from ..training.losses import compute_sample_weights, is_uniform
from ..utils import io, plotting
from ..utils.logging_utils import get_logger
from ._common import get_splits
from .train import (
    augmentation_enabled,
    find_equivalent_model,
    is_complete,
    load_result,
    train_model,
    training_objective,
)

logger = get_logger(__name__)


@stage(
    "ablate",
    tier=Tier.MUST,
    summary="Experiment 5: none vs class vs subgroup weighting, and an alpha sweep.",
    depends_on=("compare",),
    trains=True,
)
def ablate(ctx) -> None:
    config = ctx.config
    schemes = list(config.get("ablation.schemes", ["class", "subgroup"]))
    alphas = [float(a) for a in config.get("ablation.alphas", [1.0, 2.0, 4.0, 6.0, 10.0])]
    headline_alpha = float(config.get("mitigation.alpha", 4.0))

    if "subgroup" in schemes and headline_alpha not in alphas:
        logger.warning(
            "mitigation.alpha=%.2f is not in ablation.alphas=%s, so the headline result will "
            "not appear on the trade-off curve. Adding it.",
            headline_alpha, alphas,
        )
        alphas = sorted({*alphas, headline_alpha})

    # The sweep varies the WEIGHTING. Whatever else the config's mitigation block
    # declares is held fixed across the arms, exactly as batch size and seed are,
    # so an augmented run's ablation sweeps alpha on top of augmentation rather
    # than silently dropping back to unaugmented text for the arms alone.
    augment = augmentation_enabled(config)
    objective = training_objective(config)
    train_split = get_splits(ctx)["train"]
    arms = _plan(schemes, alphas)
    logger.info("ablation: %d arm(s) -> %s", len(arms), ", ".join(name for name, _, _ in arms))

    rows = []
    for model_name, scheme, alpha in arms:
        logger.info("--- ablation arm: %s (scheme=%s, alpha=%s) ---", model_name, scheme, alpha)
        weights = compute_sample_weights(train_split, scheme, config, alpha)

        arm_fingerprint = model_fingerprint(
            config, scheme, alpha, augment=augment, objective=objective
        )
        baseline_is_current = is_complete(
            ctx, "baseline", model_fingerprint(config, "none", None)
        )
        twin = find_equivalent_model(ctx, arm_fingerprint, exclude=model_name)

        if is_complete(ctx, model_name, arm_fingerprint):
            # This arm's own artefacts are on disk and match these settings.
            source, note = model_name, ""
            payload = load_result(ctx, model_name)
            logger.info("arm '%s' is already complete -- reading its artefacts", model_name)
        elif (
            is_uniform(weights) and not augment
            and objective == "bce" and baseline_is_current
        ):
            # Every weight is 1.0, so this is arithmetically the unweighted
            # baseline whatever the scheme is called -- but only while the arm
            # and the baseline also read the same training text. Under
            # augmentation they do not, and reusing the baseline here would file
            # an unaugmented model under an augmented arm's name.
            source, note = "baseline", "reused baseline (weights are uniform)"
            payload = load_result(ctx, "baseline")
            logger.info("arm '%s' has uniform weights -- reusing the baseline run", model_name)
        elif twin is not None:
            # Same settings, same weighting: the arm and the twin are the same
            # model. The headline mitigated arm always lands here.
            source, note = twin, f"reused {twin} (identical settings)"
            payload = load_result(ctx, twin)
            logger.info(
                "arm '%s' has the same settings as '%s' -- reusing it instead of retraining",
                model_name, twin,
            )
        else:
            source, note = model_name, ""
            payload = train_model(ctx, model_name, scheme=scheme, alpha=alpha,
                                  augment=augment, objective=objective)

        predictions = load_predictions(ctx.run_name, source, "test")
        table, summary = auditor.audit(predictions, identity_columns_in(predictions), config)
        io.write_table(table, ctx.table_path(f"audit_{model_name}"))

        test_metrics = payload["metrics"]["test"]
        # Global FNR stands in when no subgroup clears the FNR floor: a blank
        # safety column would let the arm that flags least read as the best.
        macro_fnr = summary.get("macro_fnr")
        safety_fnr = macro_fnr if macro_fnr is not None else test_metrics.get("fnr")

        rows.append(
            {
                "arm": model_name,
                "scheme": scheme,
                "alpha": alpha,
                "macro_f1": test_metrics.get("macro_f1"),
                "roc_auc": test_metrics.get("roc_auc"),
                "fpr_gap": summary.get("fpr_gap"),
                "macro_fpr": summary.get("macro_fpr"),
                "fpr_variance": summary.get("fpr_variance"),
                "macro_fnr": macro_fnr,
                "global_fnr": test_metrics.get("fnr"),
                "safety_fnr": safety_fnr,
                "fnr_source": "subgroup" if macro_fnr is not None else "global",
                # An arm can post a near-perfect FPR gap simply by flagging
                # almost nothing. The flag rate beside it makes that visible.
                "flag_rate": test_metrics.get("positive_rate_pred"),
                # A collapsed arm shows a perfect FPR gap; the flag stops that
                # row being read as the best result in the study.
                "degenerate": bool(summary.get("degenerate")),
                "note": note,
            }
        )
        if summary.get("degenerate"):
            logger.warning(
                "arm '%s' collapsed to a single class -- its parity numbers are void", model_name
            )

    sweep = pd.DataFrame(rows)
    _warn_on_inactive_arms(sweep)
    io.write_table(sweep, ctx.table_path("tab03_ablation"))
    ctx.save_metrics("ablate", {"stage": "ablate", "arms": sweep.to_dict(orient="records")})
    ctx.put("ablation_sweep", sweep)

    # The alpha curve only means anything within the subgroup arm; the other
    # schemes have no alpha.
    subgroup_arm = sweep[sweep["scheme"] == "subgroup"].sort_values("alpha")
    if len(subgroup_arm) > 1:
        plotting.fig_alpha_tradeoff(subgroup_arm, ctx.figure_path("fig09_alpha_tradeoff"))

    _log_sweep(sweep)


def _plan(schemes: list[str], alphas: list[float]) -> list[tuple[str, str, float | None]]:
    """Expand schemes x alphas into named arms.

    Only ``subgroup`` varies with alpha; ``none`` and ``class`` ignore it, so
    sweeping them would queue several runs of an identical model.
    """
    arms: list[tuple[str, str, float | None]] = []
    for scheme in schemes:
        if scheme == "subgroup":
            arms.extend((f"abl_subgroup_a{alpha:g}", scheme, alpha) for alpha in sorted(alphas))
        else:
            arms.append((f"abl_{scheme}", scheme, None))
    return arms


def _warn_on_inactive_arms(sweep: pd.DataFrame) -> None:
    """Say so when an arm bought its parity by flagging less, not by being fairer.

    Every subgroup false-positive rate falls together when a model stops
    predicting the positive class, so the FPR gap shrinks toward zero for a
    reason that has nothing to do with fairness. ``is_degenerate`` catches only
    the extreme -- one class for every row -- and the interesting cases stop
    short of it, which is exactly why they are easy to report by accident.
    """
    if sweep.empty or "flag_rate" not in sweep.columns:
        return

    rates = pd.to_numeric(sweep["flag_rate"], errors="coerce")
    reference = rates.max()
    if pd.isna(reference) or reference <= 0:
        return

    for _, row in sweep.iterrows():
        # A COLLAPSED arm is already named three times over. This warning is for
        # the arms that stop *short* of collapse, which nothing else catches.
        if bool(row.get("degenerate")):
            continue
        rate = pd.to_numeric(row.get("flag_rate"), errors="coerce")
        if pd.notna(rate) and rate < 0.5 * reference:
            logger.warning(
                "arm '%s' flags only %.2f%% of rows, %.0f%% of the most active arm. Its FPR "
                "gap improves partly because it stopped flagging, not because it became "
                "fairer -- read that row against its FNR, never on its own.",
                row["arm"], 100 * rate, 100 * rate / reference,
            )


def _log_sweep(sweep: pd.DataFrame) -> None:
    def fmt(row: Any, key: str) -> str:
        value = row[key]
        if isinstance(value, (int, float)) and not isinstance(value, bool) and pd.notna(value):
            return f"{value:9.4f}"
        return "      n/a"

    sources = sorted(set(sweep.get("fnr_source", pd.Series(dtype=str)).dropna()))
    logger.info("=" * 104)
    logger.info("ABLATION SWEEP  (FNR column: %s rate)", " / ".join(sources) or "unavailable")
    logger.info(
        "  %-22s %9s %9s %9s %9s %9s  %s",
        "arm", "macro_f1", "fpr_gap", "macro_fpr", "FNR", "flag_rate", "note",
    )
    for _, row in sweep.iterrows():
        note = "COLLAPSED - parity void" if row.get("degenerate") else str(row.get("note") or "")
        logger.info(
            "  %-22s %s %s %s %s %s  %s",
            row["arm"], fmt(row, "macro_f1"), fmt(row, "fpr_gap"), fmt(row, "macro_fpr"),
            fmt(row, "safety_fnr"), fmt(row, "flag_rate"), note,
        )
    logger.info("=" * 104)

"""Experiment 7 -- token attribution on false positives.

Runs both checkpoints over the *same* benign identity-bearing comments that the
baseline got wrong, and compares how much attribution mass lands on the identity
token. Same sentences, two models: that pairing is what makes the figure an
argument rather than an illustration.

Optional tier. It needs a checkpoint but no training, so it can be run on a
laptop after the window closes -- and it is reported as supporting analysis, not
as evidence about how the model reasons.
"""

from __future__ import annotations

import pandas as pd
import torch

from ..data import dataset as data_module
from ..evaluation.predictions import identity_columns_in, load_predictions
from ..explain.attributions import AttributionExplainer, identity_attribution_mass
from ..models.transformer import build_model
from ..registry import Tier, stage
from ..utils import device as device_utils
from ..utils import io
from ..utils.logging_utils import get_logger
from ._common import text_lookup

logger = get_logger(__name__)

# Surface forms the column names do not carry. Kept short and explicit rather
# than inferred, so the paper can state exactly which tokens were counted.
EXTRA_IDENTITY_TERMS = {
    "gay", "lesbian", "queer", "trans", "muslim", "jewish", "christian", "islam",
    "jew", "jews", "blacks", "whites", "women", "woman", "men", "man",
}


@stage(
    "attributions",
    tier=Tier.OPTIONAL,
    summary="Experiment 7: Integrated Gradients attribution on identity false positives.",
    depends_on=("compare",),
)
def attributions(ctx) -> None:
    config = ctx.config
    n_examples = int(config.get("attributions.n_examples", 20))
    n_steps = int(config.get("attributions.n_steps", 32))
    threshold = float(config.get("evaluation.classification_threshold", 0.5))

    cases = _select_cases(ctx, threshold, n_examples)
    if cases.empty:
        logger.warning("no identity-bearing false positives in the baseline; nothing to explain")
        ctx.save_metrics("attributions", {"stage": "attributions", "n_cases": 0})
        return

    identity_terms = _identity_terms(ctx)
    tokenizer = data_module.build_tokenizer(config)
    device = device_utils.resolve_device(str(config.get("run.device", "auto")))
    max_length = int(config.get("model.max_seq_length", 128))

    rows, detailed = [], []
    for model_name in ("baseline", "mitigated"):
        checkpoint = ctx.checkpoint_dir(model_name) / "best.pt"
        if not checkpoint.exists():
            logger.warning("no checkpoint for '%s' at %s; skipping", model_name, checkpoint)
            continue

        model = build_model(config)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        explainer = AttributionExplainer(model, tokenizer, device)
        logger.info("explaining %d case(s) with the %s model", len(cases), model_name)

        for _, case in cases.iterrows():
            explanation = explainer.explain(str(case["comment_text"]), max_length, n_steps)
            mass = identity_attribution_mass(explanation, identity_terms)
            rows.append(
                {
                    "model": model_name,
                    "id": case["id"],
                    "prob": round(explanation["prob"], 4),
                    "identity_share": round(mass["identity_share"], 4),
                    "matched_tokens": " ".join(mass["matched_tokens"]),
                    "comment_text": str(case["comment_text"])[:160],
                }
            )
            detailed.append({"model": model_name, "id": int(case["id"]), **explanation, **mass})

        del model, explainer

    table = pd.DataFrame(rows)
    io.write_table(table, ctx.table_path("tab08_attribution_summary"))
    ctx.save_metrics(
        "attributions",
        {
            "stage": "attributions",
            "n_cases": int(len(cases)),
            "n_steps": n_steps,
            "identity_terms": sorted(identity_terms),
            "summary": table.to_dict(orient="records"),
            "explanations": detailed,
            "caveat": (
                "Integrated Gradients measures local input sensitivity along a path from a "
                "baseline input. It is not counterfactual proof, not a statement about the "
                "model's reasoning in general, and is reported as supporting analysis only."
            ),
        },
    )
    _log_shift(table)


def _select_cases(ctx, threshold: float, n_examples: int) -> pd.DataFrame:
    """The baseline's most confident false positives on identity-bearing text."""
    predictions = load_predictions(ctx.run_name, "baseline", "test")
    identity_columns = identity_columns_in(predictions)
    if "any_identity" not in predictions.columns and identity_columns:
        predictions["any_identity"] = predictions[identity_columns].sum(axis=1).gt(0).astype(int)

    false_positives = predictions[
        (predictions["label"] == 0)
        & (predictions["prob"] >= threshold)
        & (predictions.get("any_identity", 0) == 1)
    ]
    if false_positives.empty:
        return false_positives

    texts = text_lookup(ctx)
    chosen = false_positives.nlargest(n_examples, "prob").copy()
    chosen["comment_text"] = chosen["id"].map(texts)
    return chosen.dropna(subset=["comment_text"]).reset_index(drop=True)


def _identity_terms(ctx) -> set[str]:
    """Token surface forms of the audited identity columns."""
    terms: set[str] = set()
    for column in ctx.config.get("data.identity_columns", []) or []:
        for part in str(column).split("_"):
            if len(part) > 2 and part not in {"other", "and", "the", "learning"}:
                terms.add(part.lower())
    terms.update(EXTRA_IDENTITY_TERMS)
    return terms


def _log_shift(table: pd.DataFrame) -> None:
    if table.empty or table["model"].nunique() < 2:
        return
    means = table.groupby("model")["identity_share"].mean()
    if "baseline" in means.index and "mitigated" in means.index:
        logger.info(
            "mean identity attribution share | baseline=%.4f -> mitigated=%.4f (delta %+.4f)",
            means["baseline"], means["mitigated"], means["mitigated"] - means["baseline"],
        )

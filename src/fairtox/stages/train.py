"""Experiments 1 and 3 -- the baseline and the mitigated model.

``train_baseline`` and ``train_mitigated`` are **the same function** reading the
same config; the only difference between them is the per-sample weight vector
handed to the loss. That makes the control structural: a reviewer checks it by
reading one function instead of diffing two config files.

Runs are idempotent -- a model whose checkpoint, predictions and metrics are all
on disk is not retrained, so re-running a downstream stage costs seconds instead
of hours. Override with ``--set run.force_retrain=True``.
"""

from __future__ import annotations

from typing import Any

from ..data import augment as augment_module
from ..data import dataset as data_module
from ..evaluation.metrics import classification_metrics, summarise
from ..evaluation.predictions import has_predictions, save_predictions
from ..models.transformer import build_model
from ..provenance import comparable_settings, model_fingerprint, training_fingerprint
from ..registry import Tier, stage
from ..training.losses import compute_group_ids, compute_sample_weights, is_uniform
from ..training.trainer import Trainer
from ..utils import device as device_utils
from ..utils import io
from ..utils.logging_utils import get_logger
from ._common import get_split_fingerprint, get_splits

logger = get_logger(__name__)


def metrics_key(model_name: str) -> str:
    return f"model_{model_name}"


def artefacts_exist(ctx, model_name: str) -> bool:
    """True when this model's checkpoint, predictions and metrics are all on disk."""
    return (
        ctx.metrics_path(metrics_key(model_name)).exists()
        and has_predictions(ctx.run_name, model_name, "test")
        and has_predictions(ctx.run_name, model_name, "val")
        and (ctx.checkpoint_dir(model_name) / "best.pt").exists()
    )


def is_complete(ctx, model_name: str, expected_fingerprint: str) -> bool:
    """True when finished artefacts exist *and* they came from these settings.

    Existence alone is not enough. Artefact paths are keyed on the model name, and
    the headline pair's names are fixed, so files at that path may have been
    produced by any settings at all. Re-running with a different alpha, batch size
    or seed would otherwise find them, skip, and quietly file the old model under
    the new label -- and the run would report success.
    """
    if not artefacts_exist(ctx, model_name):
        return False

    stored = load_result(ctx, model_name).get("model_fingerprint")
    if stored is None:
        logger.warning(
            "'%s' has artefacts but no model fingerprint, so the settings that produced "
            "them cannot be verified -- retraining rather than trusting them.",
            model_name,
        )
        return False
    if stored != expected_fingerprint:
        logger.info(
            "'%s' exists but was trained under different settings (%s != %s) -- retraining.",
            model_name, stored, expected_fingerprint,
        )
        return False
    return True


def load_result(ctx, model_name: str) -> dict[str, Any]:
    return io.load_json(ctx.metrics_path(metrics_key(model_name)))


def find_equivalent_model(ctx, fingerprint: str, exclude: str | None = None) -> str | None:
    """Name of an already-trained model with exactly these settings, if any.

    The ablation's alpha=4 arm *is* the headline mitigated model, so training it
    under the arm's own name would spend a booked-window run recomputing a result
    already on disk. Matching on the fingerprint rather than the name is what
    makes the reuse general.
    """
    metrics_dir = ctx.metrics_path("probe").parent
    if not metrics_dir.exists():
        return None

    for path in sorted(metrics_dir.glob("model_*.json")):
        name = path.stem.removeprefix("model_")
        if name == exclude:
            continue
        try:
            payload = io.load_json(path)
        except (OSError, ValueError):
            continue
        if payload.get("model_fingerprint") == fingerprint and artefacts_exist(ctx, name):
            return name
    return None


def augmentation_enabled(config) -> bool:
    """Whether this config's mitigated arm rewrites its training text."""
    return bool(config.get("mitigation.augmentation.enabled", False))


def training_objective(config) -> str:
    """The loss this config's mitigated arm optimises: ``bce`` or ``group_dro``."""
    return str(config.get("mitigation.objective", "bce")).lower()


def train_model(
    ctx,
    model_name: str,
    scheme: str,
    alpha: float | None = None,
    augment: bool = False,
    objective: str = "bce",
    allow_skip: bool = True,
) -> dict[str, Any]:
    """Fine-tune one model, evaluate it, and write its artefacts."""
    config = ctx.config
    force = bool(config.get("run.force_retrain", False))
    expected = model_fingerprint(config, scheme, alpha, augment=augment, objective=objective)

    if allow_skip and not force and is_complete(ctx, model_name, expected):
        payload = load_result(ctx, model_name)
        logger.info(
            "'%s' is already complete (checkpoint, predictions and metrics on disk) -- skipping. "
            "Use --set run.force_retrain=True to train it again.",
            model_name,
        )
        ctx.put(f"{model_name}_metrics", payload)
        return payload

    splits = get_splits(ctx)
    identity_columns = ctx.take("identity_columns") or []
    fingerprint = get_split_fingerprint(ctx)

    # The intervention, when it is a data one. Only the training split is ever
    # rewritten: augmenting val or test would change what the model is measured
    # on, and the split fingerprint -- the thing that makes the nine runs
    # comparable -- is computed from the partition, not from the text.
    train_frame = splits["train"]
    augmentation: dict[str, Any] | None = None
    if augment:
        train_frame, augmentation = augment_module.swap_identity_terms(
            train_frame,
            probability=float(config.get("mitigation.augmentation.probability", 0.5)),
            seed=int(config.get("mitigation.augmentation.seed", 13)),
            scope=str(config.get("mitigation.augmentation.scope", "all")),
        )

    weights = compute_sample_weights(train_frame, scheme, config, alpha)
    # Cell ids only for the objective that needs them. The baseline and every
    # BCE arm get None, so their dataset is byte-for-byte what it always was.
    groups = compute_group_ids(train_frame) if objective == "group_dro" else None
    tokenizer = data_module.build_tokenizer(config)
    max_length = int(config.get("model.max_seq_length", 128))
    num_workers = int(config.get("training.num_workers", 0))

    train_set = data_module.ToxicityDataset(train_frame, tokenizer, max_length, weights, groups)
    val_set = data_module.ToxicityDataset(splits["val"], tokenizer, max_length)
    test_set = data_module.ToxicityDataset(splits["test"], tokenizer, max_length)

    batch_size = int(config.get("training.batch_size", 32))
    eval_batch_size = int(config.get("training.eval_batch_size", batch_size * 2))
    train_loader = data_module.build_loader(
        train_set, batch_size, shuffle=True, num_workers=num_workers, seed=config.seed
    )
    # shuffle=False on the evaluation loaders is load-bearing: the audit rejoins
    # probabilities to identity flags by position.
    val_loader = data_module.build_loader(val_set, eval_batch_size, shuffle=False,
                                          num_workers=num_workers)
    test_loader = data_module.build_loader(test_set, eval_batch_size, shuffle=False,
                                           num_workers=num_workers)

    model = build_model(config)
    # The objective is handed over per model, never re-read from the config:
    # the baseline is plain BCE in every run, whatever the mitigated arm is.
    trainer = Trainer(model, config, objective=objective)
    history = trainer.fit(train_loader, val_loader, ctx.checkpoint_dir(model_name))

    threshold = float(config.get("evaluation.classification_threshold", 0.5))
    split_metrics: dict[str, Any] = {}
    for split_name, loader in (("val", val_loader), ("test", test_loader)):
        probs, labels = trainer.predict(loader)
        metrics = classification_metrics(labels, probs, threshold)
        split_metrics[split_name] = metrics
        save_predictions(
            ctx.run_name, model_name, split_name, splits[split_name], probs, identity_columns
        )
        logger.info("%s %s | %s", model_name, split_name, summarise(metrics))

    payload = {
        "stage": metrics_key(model_name),
        "model_name": model_name,
        "weighting_scheme": scheme,
        "alpha": alpha if scheme == "subgroup" else None,
        "uniform_weights": is_uniform(weights),
        # None when the arm was not augmented, so a reader can tell "no
        # augmentation" from "augmentation that rewrote nothing" -- the second
        # would be a broken run reported as a null result.
        "augmentation": augmentation,
        "objective": objective,
        # What group DRO ended up deciding, so the paper can print the LEARNED
        # cell weights beside the ones we assumed.
        "learned_group_weights": (
            trainer.criterion.learned_weights()
            if hasattr(trainer.criterion, "learned_weights") else None
        ),
        "split_fingerprint": fingerprint,
        # Recorded so `compare` can refuse two arms that were not configured
        # identically -- a per-arm --set override is otherwise invisible.
        "training_fingerprint": training_fingerprint(config),
        # Everything that determined these weights, so a later invocation can tell
        # whether the artefacts on disk match the settings it was asked for.
        "model_fingerprint": expected,
        "comparable_settings": comparable_settings(config),
        "split_sizes": {name: len(part) for name, part in splits.items()},
        "backbone": str(config.get("model.backbone", "distilbert-base-uncased")),
        "hyperparameters": {
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": float(config.get("training.learning_rate", 2e-5)),
            "num_epochs": int(config.get("training.num_epochs", 3)),
            "gradient_accumulation_steps": int(
                config.get("training.gradient_accumulation_steps", 1)
            ),
            "max_seq_length": max_length,
            "mixed_precision": bool(config.get("training.mixed_precision", True)),
            "weight_decay": float(config.get("training.weight_decay", 0.01)),
            "warmup_ratio": float(config.get("training.warmup_ratio", 0.1)),
        },
        "training": history.as_dict(),
        "device": device_utils.describe_device(trainer.device),
        "metrics": split_metrics,
    }
    ctx.save_metrics(metrics_key(model_name), payload)
    ctx.put(f"{model_name}_metrics", payload)

    if split_metrics["test"].get("degenerate"):
        logger.warning(
            "'%s' predicts a SINGLE CLASS on the test split. Its subgroup parity numbers are "
            "an artefact of collapse, not a fairness result.",
            model_name,
        )
    return payload


@stage(
    "train_baseline",
    tier=Tier.MUST,
    summary="Experiment 1: fine-tune the backbone with unweighted BCE.",
    depends_on=("eda",),
    trains=True,
)
def train_baseline(ctx) -> None:
    train_model(ctx, "baseline", scheme="none")


@stage(
    "train_mitigated",
    tier=Tier.MUST,
    summary="Experiment 3: the same run under whatever mitigation the config declares.",
    depends_on=("train_baseline",),
    trains=True,
)
def train_mitigated(ctx) -> None:
    config = ctx.config
    scheme = str(config.get("mitigation.scheme", "subgroup"))
    alpha = float(config.get("mitigation.alpha", 4.0))
    # The mitigated arm is whatever the `mitigation:` block says it is: loss
    # weighting, counterfactual augmentation of the training text, or both. The
    # baseline above is fixed at neither, so the pair stays a controlled one
    # whichever mitigation is being measured.
    train_model(ctx, "mitigated", scheme=scheme, alpha=alpha,
                augment=augmentation_enabled(config), objective=training_objective(config))

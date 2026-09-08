"""Comparability guards for the controlled comparison.

Driving both arms from one config makes them identical by construction -- but the
runbook launches them as two commands, and a command line takes ``--set``. A
``--set training.batch_size=128`` on one arm only would make the comparison
measure batch size as much as the intervention, and nothing in the output would
look wrong. So every model records a fingerprint of the settings that must match,
and ``compare`` refuses a pair whose fingerprints differ.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .utils.logging_utils import get_logger

logger = get_logger(__name__)

# Anything absent from this list is either the variable under test (mitigation.*)
# or cannot affect the result (paths, log level, dataloader workers).
COMPARABLE_KEYS: tuple[str, ...] = (
    "model.backbone",
    "model.max_seq_length",
    "model.dropout",
    "model.freeze_encoder",
    "training.batch_size",
    "training.learning_rate",
    "training.num_epochs",
    "training.gradient_accumulation_steps",
    "training.weight_decay",
    "training.warmup_ratio",
    "training.max_grad_norm",
    "training.mixed_precision",
    "training.early_stopping_patience",
    "training.early_stopping_metric",
    "evaluation.classification_threshold",
    "data.toxicity_threshold",
    "data.identity_threshold",
    "data.annotated_only",
    "data.subsample",
    "run.seed",
)

# Everything that determines the trained weights: the comparable keys plus the
# data-shaping settings, with the weighting supplied separately.
MODEL_KEYS: tuple[str, ...] = COMPARABLE_KEYS + (
    "data.splits",
    "data.identity_columns",
    "data.raw_file",
    "data.text_column",
    "data.target_column",
)


def comparable_settings(config: Any) -> dict[str, Any]:
    """The subset of the config that two comparable runs must agree on."""
    return {key: config.get(key) for key in COMPARABLE_KEYS}


def training_fingerprint(config: Any) -> str:
    """Stable hash of the settings that must match across arms."""
    payload = json.dumps(comparable_settings(config), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def model_fingerprint(
    config: Any,
    scheme: str,
    alpha: float | None = None,
    augment: bool = False,
    objective: str = "bce",
) -> str:
    """Stable hash of everything that determines *this* model's weights.

    Deliberately not ``Config.fingerprint()``, which covers the whole file: that
    changes when ``data.raw_dir`` changes -- the Kaggle notebook overrides it on
    every cell -- and would retrain the study on a path change.

    ``augment`` is an argument rather than another config read because both arms
    read the same config: an augmented arm and its unaugmented baseline would
    otherwise hash identically, and the reuse-by-fingerprint path would hand the
    baseline's predictions back as the augmented model's.
    """
    payload = {
        "settings": {key: config.get(key) for key in MODEL_KEYS},
        "weighting": {
            "scheme": scheme,
            "alpha": alpha,
            "weights": config.get("mitigation.weights"),
        },
    }
    if augment:
        # Added only when augmentation is on, so every fingerprint recorded
        # before this existed still hashes to the same value. The nine finished
        # runs stay idempotent and are not retrained by a re-run of an analysis.
        payload["augmentation"] = config.get("mitigation.augmentation")
    narrowed = config.get("mitigation.weighting_columns")
    if narrowed:
        # Which identity columns the weighting was allowed to see determines the
        # weight vector, so it belongs in the hash. Conditional, like the fields
        # below, so a run that does not narrow keeps the fingerprint it had.
        payload["weighting_columns"] = sorted(narrowed)
    select = str(config.get("training.select_checkpoint", "best")).lower()
    if select != "best":
        # Which epoch is kept determines the weights, so it belongs in the hash.
        # Conditional for the same reason as the fields around it: the default
        # must leave every fingerprint recorded before this existed unchanged.
        payload["select_checkpoint"] = select
    if objective != "bce":
        # Same reason: the default objective adds nothing to the payload, so the
        # fingerprints of every model trained before this feature are unchanged.
        payload["objective"] = {"name": objective, "settings": config.get("mitigation.dro")}
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _differences(first: dict[str, Any], second: dict[str, Any]) -> list[str]:
    return [
        f"  {key}: {first.get(key)!r} vs {second.get(key)!r}"
        for key in COMPARABLE_KEYS
        if first.get(key) != second.get(key)
    ]


def assert_comparable(
    first: dict[str, Any],
    second: dict[str, Any],
    what: str,
    names: tuple[str, str] = ("baseline", "mitigated"),
) -> None:
    """Fail when two runs being compared were not configured identically.

    Uses the recorded settings as well as the hash, so the error names the
    offending key instead of only reporting that two hashes differ.
    """
    left, right = first.get("training_fingerprint"), second.get("training_fingerprint")
    if left is None or right is None:
        logger.warning(
            "%s: a training fingerprint is missing, so the settings could not be verified. "
            "Re-run both arms if the comparison is going into the paper.",
            what,
        )
        return
    if left == right:
        return

    detail = _differences(
        first.get("comparable_settings", {}), second.get("comparable_settings", {})
    )
    listing = "\n".join(detail) if detail else "  (settings were not recorded in full)"
    raise ValueError(
        f"{what}: '{names[0]}' and '{names[1]}' were trained under DIFFERENT settings "
        f"({left} vs {right}):\n{listing}\n"
        "The comparison assumes only the loss weighting changed, so this result is not "
        "interpretable. Re-run both arms from the same config, without per-arm --set "
        "overrides."
    )


def warn_on_hardware_mismatch(
    first: dict[str, Any], second: dict[str, Any], what: str
) -> None:
    """Warn when two arms were trained on different hardware.

    A warning, not an error: moving machines is legitimate. But a headline pair
    split across a T4 and an A100 has a hardware confound inside exactly the
    comparison the study rests on, and that must be visible in the log.
    """
    left = (first.get("device") or {}).get("gpu_name")
    right = (second.get("device") or {}).get("gpu_name")
    if left != right:
        logger.warning(
            "%s: the two arms were trained on DIFFERENT hardware (%s vs %s). Re-run the pair "
            "on one machine before reporting it, or state the split explicitly in the paper.",
            what, left, right,
        )

"""Reuse of an existing checkpoint must depend on the settings, not just the path.

Artefact paths are keyed on the model name, and the headline pair's names are
fixed ("baseline", "mitigated"). So the files at a path say nothing about the
config that produced them. Skipping on existence alone means a rerun with a
different alpha, batch size or seed silently keeps the old model and reports
success -- and `compare` then writes it up under the new label.

The failure that motivated these tests: train at alpha=4, watch it collapse,
rerun at alpha=2, and have the collapsed alpha=4 model reported as the alpha=2
result.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fairtox.evaluation import predictions as pred_io
from fairtox.provenance import model_fingerprint
from fairtox.stages.train import (
    artefacts_exist,
    find_equivalent_model,
    is_complete,
    metrics_key,
)


@pytest.fixture
def ctx(tmp_path, monkeypatch, config):
    """A minimal run context whose artefact tree lives in a temp directory."""
    from fairtox.utils import io

    monkeypatch.setattr(io, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(io, "EXPERIMENTS_DIR", tmp_path / "experiments")

    return SimpleNamespace(
        config=config,
        run_name=config.run_name,
        metrics_path=lambda stage: io.metrics_path(config.run_name, stage),
        checkpoint_dir=lambda model: io.checkpoint_dir(config.run_name, model),
    )


def _write_artefacts(ctx, model_name: str, fingerprint: str | None) -> None:
    """Lay down the files a finished run leaves behind."""
    frame = pd.DataFrame(
        {
            "id": np.arange(4),
            "label": np.array([0, 1, 0, 1]),
            "any_identity": np.array([1, 0, 1, 0]),
            "group_a": np.array([1, 0, 1, 0]),
        }
    )
    for split in ("val", "test"):
        pred_io.save_predictions(
            ctx.run_name, model_name, split, frame, np.array([0.1, 0.9, 0.2, 0.8]), ["group_a"]
        )

    checkpoint = ctx.checkpoint_dir(model_name)
    checkpoint.mkdir(parents=True, exist_ok=True)
    (checkpoint / "best.pt").write_bytes(b"not really a checkpoint")

    payload = {"stage": metrics_key(model_name), "model_name": model_name}
    if fingerprint is not None:
        payload["model_fingerprint"] = fingerprint
    path = ctx.metrics_path(metrics_key(model_name))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_same_settings_are_reused(ctx, config):
    """The whole point of idempotency: do not spend the window twice."""
    fingerprint = model_fingerprint(config, "subgroup", 4.0)
    _write_artefacts(ctx, "mitigated", fingerprint)
    assert is_complete(ctx, "mitigated", fingerprint) is True


def test_a_different_alpha_forces_a_retrain(ctx, config):
    """The dangerous case: a collapsed alpha=4 model reported as alpha=2."""
    _write_artefacts(ctx, "mitigated", model_fingerprint(config, "subgroup", 4.0))
    assert is_complete(ctx, "mitigated", model_fingerprint(config, "subgroup", 2.0)) is False


def test_a_different_batch_size_forces_a_retrain(ctx, config):
    """The throughput-tuning case: you keep batch 64 believing you ran 128."""
    _write_artefacts(ctx, "baseline", model_fingerprint(config, "none", None))
    config.set("training.batch_size", 128)
    assert is_complete(ctx, "baseline", model_fingerprint(config, "none", None)) is False


@pytest.mark.parametrize(
    "key, value",
    [
        ("run.seed", 43),
        ("model.backbone", "roberta-base"),
        ("model.max_seq_length", 256),
        ("training.learning_rate", 5e-5),
        ("training.num_epochs", 5),
        ("data.subsample", 50_000),
        ("data.splits", {"train": 0.8, "val": 0.1, "test": 0.1}),
        ("data.identity_columns", ["group_a"]),
        ("data.toxicity_threshold", 0.7),
    ],
)
def test_every_model_changing_setting_forces_a_retrain(ctx, config, key, value):
    _write_artefacts(ctx, "baseline", model_fingerprint(config, "none", None))
    config.set(key, value)
    assert is_complete(ctx, "baseline", model_fingerprint(config, "none", None)) is False


@pytest.mark.parametrize(
    "key, value",
    [
        # The Kaggle notebook passes this on every single cell. Keying reuse on
        # the whole-config hash would retrain the entire study because of it.
        ("data.raw_dir", "/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification"),
        ("run.log_level", "DEBUG"),
        ("training.num_workers", 8),
        ("training.eval_batch_size", 512),
        ("analysis.seed_runs", ["main", "main_s43"]),
        ("sensitivity.thresholds", [0.5]),
        ("attributions.n_examples", 50),
        ("evaluation.min_support", 50),
        ("evaluation.ci_method", "bootstrap"),
    ],
)
def test_settings_that_cannot_change_a_weight_do_not_force_a_retrain(ctx, config, key, value):
    """Idempotency has to survive the overrides people actually pass."""
    _write_artefacts(ctx, "baseline", model_fingerprint(config, "none", None))
    config.set(key, value)
    assert is_complete(ctx, "baseline", model_fingerprint(config, "none", None)) is True


def test_a_different_scheme_forces_a_retrain(ctx, config):
    _write_artefacts(ctx, "abl_class", model_fingerprint(config, "class", None))
    assert is_complete(ctx, "abl_class", model_fingerprint(config, "subgroup", 4.0)) is False


def test_explicit_mitigation_weights_are_part_of_the_identity(ctx, config):
    _write_artefacts(ctx, "mitigated", model_fingerprint(config, "subgroup", 4.0))
    config.set("mitigation.weights", {"background": 0.5, "identity_toxic": 2.0})
    assert is_complete(ctx, "mitigated", model_fingerprint(config, "subgroup", 4.0)) is False


def test_missing_fingerprint_is_not_trusted(ctx, config):
    """Artefacts without provenance cannot be verified, so they are not reused."""
    _write_artefacts(ctx, "baseline", fingerprint=None)
    assert artefacts_exist(ctx, "baseline") is True
    assert is_complete(ctx, "baseline", model_fingerprint(config, "none", None)) is False


def test_incomplete_artefacts_are_not_reused(ctx, config):
    """A run killed between the checkpoint and the metrics file must not count."""
    fingerprint = model_fingerprint(config, "none", None)
    _write_artefacts(ctx, "baseline", fingerprint)
    ctx.metrics_path(metrics_key("baseline")).unlink()
    assert is_complete(ctx, "baseline", fingerprint) is False


def test_nothing_on_disk_is_not_complete(ctx, config):
    assert is_complete(ctx, "never_trained", model_fingerprint(config, "none", None)) is False


# -- reusing a model already trained under another name -----------------------


def test_an_identically_configured_model_is_found(ctx, config):
    """The ablation's alpha=4 arm is the headline mitigated model.

    `mitigation.alpha` must appear in `ablation.alphas` or the trade-off figure
    would omit the point the paper's claim sits on. Retraining it under the arm's
    own name spends a booked-window run reproducing a model already on disk.
    """
    fingerprint = model_fingerprint(config, "subgroup", 4.0)
    _write_artefacts(ctx, "mitigated", fingerprint)

    assert find_equivalent_model(ctx, fingerprint, exclude="abl_subgroup_a4") == "mitigated"


def test_a_differently_configured_model_is_not_offered(ctx, config):
    _write_artefacts(ctx, "mitigated", model_fingerprint(config, "subgroup", 4.0))
    assert find_equivalent_model(ctx, model_fingerprint(config, "subgroup", 6.0)) is None


def test_the_baseline_is_not_mistaken_for_a_mitigated_arm(ctx, config):
    _write_artefacts(ctx, "baseline", model_fingerprint(config, "none", None))
    assert find_equivalent_model(ctx, model_fingerprint(config, "subgroup", 4.0)) is None


def test_a_model_excludes_itself(ctx, config):
    """Otherwise an arm would 'reuse' its own artefacts and never retrain."""
    fingerprint = model_fingerprint(config, "subgroup", 4.0)
    _write_artefacts(ctx, "abl_subgroup_a4", fingerprint)
    assert find_equivalent_model(ctx, fingerprint, exclude="abl_subgroup_a4") is None


def test_a_match_with_missing_artefacts_is_not_offered(ctx, config):
    """A metrics file alone is not a usable model: predictions are what get read."""
    fingerprint = model_fingerprint(config, "subgroup", 4.0)
    _write_artefacts(ctx, "mitigated", fingerprint)
    (ctx.checkpoint_dir("mitigated") / "best.pt").unlink()

    assert find_equivalent_model(ctx, fingerprint) is None


def test_no_models_at_all_is_not_an_error(ctx, config):
    assert find_equivalent_model(ctx, model_fingerprint(config, "none", None)) is None

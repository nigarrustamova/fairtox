"""The comparability guard.

`compare` refuses two arms that were not trained under identical settings. The
failure it exists to catch is quiet: a `--set` override applied to one of the two
launch commands changes what the study measures, and nothing in the resulting
numbers looks wrong.
"""

from __future__ import annotations

import pytest

from fairtox.config import Config
from fairtox.provenance import (
    COMPARABLE_KEYS,
    assert_comparable,
    comparable_settings,
    training_fingerprint,
    warn_on_hardware_mismatch,
)


def _payload(config: Config, gpu: str = "NVIDIA A100-SXM4-40GB") -> dict:
    return {
        "training_fingerprint": training_fingerprint(config),
        "comparable_settings": comparable_settings(config),
        "device": {"gpu_name": gpu},
    }


def test_identical_configs_are_comparable(config):
    payload = _payload(config)
    assert_comparable(payload, dict(payload), "test")


def test_a_different_batch_size_is_rejected(config):
    """The exact scenario: --set training.batch_size=128 on one arm only."""
    baseline = _payload(config)
    config.set("training.batch_size", 128)
    mitigated = _payload(config)

    with pytest.raises(ValueError, match="DIFFERENT settings"):
        assert_comparable(baseline, mitigated, "compare(baseline, mitigated)")


def test_the_error_names_the_offending_key(config):
    """A hash mismatch alone would send someone hunting; name the key instead."""
    baseline = _payload(config)
    config.set("training.learning_rate", 4.5e-5)
    mitigated = _payload(config)

    with pytest.raises(ValueError) as excinfo:
        assert_comparable(baseline, mitigated, "test")
    message = str(excinfo.value)
    assert "training.learning_rate" in message
    assert "4.5e-05" in message or "4.5e-5" in message


@pytest.mark.parametrize(
    "key, value",
    [
        ("model.backbone", "roberta-base"),
        ("model.max_seq_length", 256),
        ("training.num_epochs", 5),
        ("training.gradient_accumulation_steps", 4),
        ("training.mixed_precision", False),
        ("run.seed", 43),
        ("data.subsample", 50_000),
        ("evaluation.classification_threshold", 0.7),
    ],
)
def test_every_confounding_setting_is_caught(config, key, value):
    baseline = _payload(config)
    config.set(key, value)
    with pytest.raises(ValueError, match="DIFFERENT settings"):
        assert_comparable(baseline, _payload(config), "test")


def test_the_variable_under_test_is_not_a_confound(config):
    """The arms MUST differ in the mitigation settings -- that is the experiment."""
    baseline = _payload(config)
    config.set("mitigation.alpha", 8.0)
    config.set("mitigation.scheme", "class")
    assert_comparable(baseline, _payload(config), "test")


def test_mitigation_keys_are_excluded_by_construction():
    assert not any(key.startswith("mitigation.") for key in COMPARABLE_KEYS)


def test_bookkeeping_settings_are_not_confounds(config):
    """Paths, workers and log verbosity cannot change a prediction."""
    baseline = _payload(config)
    for key, value in (
        ("run.name", "something_else"),
        ("run.log_level", "DEBUG"),
        ("training.num_workers", 8),
        ("training.eval_batch_size", 1024),
        ("data.raw_dir", "/kaggle/input/whatever"),
        ("training.memory_budget_gb", 11.0),
    ):
        config.set(key, value)
    assert_comparable(baseline, _payload(config), "test")


def test_a_missing_fingerprint_warns_rather_than_crashing(config, caplog):
    """Results predating the guard should be usable, but not silently."""
    payload = _payload(config)
    assert_comparable(payload, {"device": {}}, "test")
    assert "could not be verified" in caplog.text


def test_hardware_mismatch_warns_but_does_not_block(config, caplog):
    a100 = _payload(config, gpu="NVIDIA A100-SXM4-40GB")
    t4 = _payload(config, gpu="Tesla T4")

    warn_on_hardware_mismatch(a100, t4, "test")
    assert "DIFFERENT hardware" in caplog.text


def test_matching_hardware_is_silent(config, caplog):
    payload = _payload(config)
    warn_on_hardware_mismatch(payload, dict(payload), "test")
    assert "DIFFERENT hardware" not in caplog.text


def test_fingerprint_is_stable_across_equal_configs():
    first = Config({"training": {"batch_size": 64}, "run": {"seed": 42}})
    second = Config({"run": {"seed": 42}, "training": {"batch_size": 64}})
    assert training_fingerprint(first) == training_fingerprint(second)


def test_shipped_configs_agree_where_it_matters():
    """default and a100 differ in batch size, so their fingerprints must differ.

    A guard that could not tell those two apart would not catch the real mistake
    either.
    """
    assert training_fingerprint(Config.load("configs/default.yaml")) != training_fingerprint(
        Config.load("configs/a100.yaml")
    )

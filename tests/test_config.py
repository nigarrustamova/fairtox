"""Config loading, the `extends` chain, and the invariants the study relies on."""

from __future__ import annotations

import pytest

from fairtox.config import Config


def test_dotted_get_and_default(config):
    assert config.get("run.seed") == 42
    assert config.get("run.nothing_here", "fallback") == "fallback"
    assert config.get("totally.absent.key") is None


def test_dotted_set_creates_intermediate_levels(config):
    config.set("a.b.c", 5)
    assert config.get("a.b.c") == 5


def test_set_overwrites_a_scalar_with_a_mapping(config):
    config.set("run.device", "cpu")
    config.set("run.device.sub", 1)
    assert config.get("run.device.sub") == 1


def test_require_names_the_missing_key(config):
    with pytest.raises(KeyError, match="run.absent"):
        config.require("run.absent")


def test_fingerprint_changes_with_content(config):
    before = config.fingerprint()
    config.set("training.batch_size", 999)
    assert config.fingerprint() != before


def test_fingerprint_is_order_independent():
    first = Config({"a": 1, "b": 2})
    second = Config({"b": 2, "a": 1})
    assert first.fingerprint() == second.fingerprint()


def test_as_dict_is_a_copy(config):
    snapshot = config.as_dict()
    snapshot["run"]["seed"] = 999
    assert config.get("run.seed") == 42


def test_missing_file_lists_what_is_available():
    with pytest.raises(FileNotFoundError, match="Available in configs/"):
        Config.load("configs/does_not_exist.yaml")


# -- the shipped configs -----------------------------------------------------


def test_default_config_loads():
    config = Config.load("configs/default.yaml")
    assert config.get("model.backbone")
    assert config.get("data.identity_columns")


def test_smoke_config_extends_default():
    smoke = Config.load("configs/smoke.yaml")
    # Overridden in smoke.yaml...
    assert smoke.get("training.num_epochs") == 1
    assert smoke.get("run.device") == "cpu"
    # ...and inherited from default.yaml.
    assert smoke.get("data.identity_columns") == Config.load(
        "configs/default.yaml"
    ).get("data.identity_columns")


def test_a100_config_extends_default():
    a100 = Config.load("configs/a100.yaml")
    assert a100.get("training.batch_size") == 64
    assert a100.get("mitigation.scheme") == "subgroup"


@pytest.mark.parametrize("name", ["default", "a100"])
def test_headline_alpha_is_in_the_ablation_sweep(name):
    """Otherwise the trade-off figure omits the point the paper's claim sits on.

    The ablation stage repairs this at run time, but a config that needs
    repairing is a config that will be misread by whoever edits it next.
    """
    config = Config.load(f"configs/{name}.yaml")
    assert float(config.get("mitigation.alpha")) in [
        float(a) for a in config.get("ablation.alphas")
    ]


@pytest.mark.parametrize("name", ["default", "a100"])
def test_memory_budget_matches_the_announced_allocation(name):
    """Exceeding the allocation is an automatic deduction, so it must be set.

    The brief says ~10-12 GB; the teaching assistants raised it to 20 GB. This
    pins the figure the code actually enforces, so a change to the allocation has
    to be a deliberate edit rather than a config that quietly drifted.
    """
    budget = Config.load(f"configs/{name}.yaml").get("training.memory_budget_gb")
    assert budget is not None, "an unset budget disables the VRAM guard entirely"
    assert float(budget) == 20.0


def test_the_a100_batch_leaves_room_under_the_allocation():
    """Sanity floor on the headline settings.

    DistilBERT at seq 128 costs roughly 0.1 GB per unit of batch. This is not a
    precise model -- it is a guard against someone raising the batch to a value
    that cannot fit, which is only discovered as an OOM inside the booked window.
    """
    config = Config.load("configs/a100.yaml")
    batch = int(config.get("training.batch_size"))
    budget = float(config.get("training.memory_budget_gb"))
    assert batch * 0.1 < budget


@pytest.mark.parametrize("name", ["default", "a100"])
def test_split_fractions_sum_to_one(name):
    splits = Config.load(f"configs/{name}.yaml").get("data.splits")
    assert sum(float(v) for v in splits.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["default", "a100"])
def test_subsampling_is_off_for_reportable_runs(name):
    """Subsampling shrinks the test split, and with it the reportable subgroups."""
    assert Config.load(f"configs/{name}.yaml").get("data.subsample") is None


@pytest.mark.parametrize("name", ["default", "a100"])
def test_model_selection_does_not_key_off_accuracy(name):
    """A majority-class predictor scores ~90% accuracy on this corpus."""
    metric = Config.load(f"configs/{name}.yaml").get("training.early_stopping_metric")
    assert "accuracy" not in metric

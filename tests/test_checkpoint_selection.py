"""Checkpoint selection: the study's rule, and the control that removes it.

The headline runs keep the epoch with the best validation macro-F1. Both arms
follow that rule, so the protocol is fair -- but they do not peak on the same
epoch, so the KEPT models differ in more than the weight vector. ``last`` is the
control that removes the difference, and these tests pin the two behaviours apart
so a future edit cannot quietly merge them.
"""

from __future__ import annotations

import pytest

from fairtox.config import Config
from fairtox.provenance import model_fingerprint


def _config(**training) -> Config:
    return Config(
        {
            "run": {"seed": 42},
            "model": {"backbone": "distilbert-base-uncased"},
            "training": training,
            "mitigation": {"weights": {"background": 1.0}},
        }
    )


def test_default_is_best():
    assert _config().get("training.select_checkpoint", "best") == "best"


def test_the_default_leaves_old_fingerprints_untouched():
    """Every model trained before this option existed must keep its hash.

    Nine runs are already on disk. If the default changed their fingerprint, the
    next analysis re-run would decide they were trained under other settings and
    retrain them -- spending a closed GPU window to reproduce what it deleted.
    """
    without = model_fingerprint(_config(), "none", None)
    with_default = model_fingerprint(_config(select_checkpoint="best"), "none", None)

    assert without == with_default


def test_the_control_changes_the_fingerprint():
    """Different kept weights must not be able to reuse each other's artefacts."""
    study = model_fingerprint(_config(select_checkpoint="best"), "none", None)
    control = model_fingerprint(_config(select_checkpoint="last"), "none", None)

    assert study != control


def test_an_unknown_selection_is_refused():
    import torch

    from fairtox.training.trainer import Trainer

    model = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="select_checkpoint"):
        Trainer(model, _config(select_checkpoint="middle"), device="cpu")

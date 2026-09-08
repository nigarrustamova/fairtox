"""Group DRO: the learned counterpart of the static weight vector.

Two things have to hold for the comparison in the paper to mean anything. The
cell index must be the same partition the static scheme weights, or the two arms
are not weighting the same thing. And the learned weight must move TOWARDS the
cell with the higher loss, or the arm is a plain mean wearing another name -- the
failure that would look like "the method does nothing".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from fairtox.config import Config
from fairtox.training.losses import (
    GROUP_NAMES,
    GroupDROLoss,
    WeightedBCEWithLogitsLoss,
    build_criterion,
    compute_group_ids,
)

# -- the partition -----------------------------------------------------------


def test_cell_ids_match_the_static_scheme(cells):
    """2 * any_identity + label, in the order the weight logger prints."""
    ids = compute_group_ids(cells)
    expected = 2 * cells["any_identity"].to_numpy() + cells["label"].to_numpy()

    assert np.array_equal(ids, expected)
    assert set(np.unique(ids)) <= set(range(len(GROUP_NAMES)))


def test_every_cell_is_named():
    assert GROUP_NAMES == (
        "background_nontoxic", "background_toxic", "identity_nontoxic", "identity_toxic",
    )


# -- the learned weights -----------------------------------------------------


def _batch(losses_by_group: dict[int, float], n_groups: int = 4):
    """Logits whose BCE loss per group is roughly what the caller asked for.

    Targets are all 0, so the loss of a sample is softplus(logit) and rises
    monotonically with it. Exact values do not matter; the ordering does.
    """
    logits, groups = [], []
    for group, level in losses_by_group.items():
        logits.append(level)
        groups.append(group)
    return (
        torch.tensor(logits, dtype=torch.float32),
        torch.zeros(len(logits)),
        torch.tensor(groups, dtype=torch.long),
    )


def test_weight_rises_for_the_worse_cell():
    loss_fn = GroupDROLoss(n_groups=4, step_size=1.0)
    logits, targets, groups = _batch({0: -4.0, 1: -4.0, 2: 4.0, 3: -4.0})

    loss_fn(logits, targets, None, groups)
    learned = loss_fn.learned_weights()

    assert learned["identity_nontoxic"] > learned["background_nontoxic"]
    assert learned["identity_nontoxic"] > 0.25


def test_weights_stay_on_the_simplex():
    loss_fn = GroupDROLoss(n_groups=4, step_size=1.0)
    logits, targets, groups = _batch({0: -3.0, 1: 0.0, 2: 5.0, 3: 1.0})

    for _ in range(20):
        loss_fn(logits, targets, None, groups)

    values = list(loss_fn.learned_weights().values())
    assert sum(values) == pytest.approx(1.0)
    assert all(v >= 0 for v in values)


def test_a_cell_absent_from_the_batch_is_only_renormalised():
    """Its raw weight is untouched; it moves only because the others grew.

    The guard this pins is the division by a zero count. Left unguarded it
    produces a NaN, the NaN spreads through the simplex on the next step, and
    every subsequent batch trains on a loss of nan -- a run that looks like it is
    progressing and is not.
    """
    loss_fn = GroupDROLoss(n_groups=4, step_size=1.0)
    logits, targets, groups = _batch({0: 2.0, 1: 2.0, 2: 2.0})
    loss_fn(logits, targets, None, groups)
    after = loss_fn.learned_weights()

    # Three cells share one loss, softplus(2); the fourth keeps its raw 0.25.
    grown = float(np.exp(np.logaddexp(0.0, 2.0)))
    assert after["identity_toxic"] == pytest.approx(1.0 / (3 * grown + 1), rel=1e-5)
    assert after["identity_toxic"] > 0
    assert not any(np.isnan(list(after.values())))


def test_uniform_start():
    loss_fn = GroupDROLoss(n_groups=4)
    assert list(loss_fn.learned_weights().values()) == pytest.approx([0.25] * 4)


def test_sample_weights_are_honoured_within_a_cell():
    """Nothing is silently dropped when both interventions are switched on."""
    loss_fn = GroupDROLoss(n_groups=4, step_size=0.0)
    logits = torch.tensor([4.0, -4.0], dtype=torch.float32)
    targets = torch.zeros(2)
    groups = torch.tensor([0, 0], dtype=torch.long)

    heavy_on_the_bad_one = loss_fn(logits, targets, torch.tensor([9.0, 1.0]), groups)
    heavy_on_the_good_one = loss_fn(logits, targets, torch.tensor([1.0, 9.0]), groups)

    assert heavy_on_the_bad_one > heavy_on_the_good_one


# -- refusals and wiring -----------------------------------------------------


def test_missing_groups_is_refused():
    """A silent fallback to plain BCE would be reported as a null result."""
    loss_fn = GroupDROLoss()
    with pytest.raises(ValueError, match="group id per sample"):
        loss_fn(torch.zeros(3), torch.zeros(3), None, None)


def test_build_criterion_defaults_to_weighted_bce(config):
    assert isinstance(build_criterion(config), WeightedBCEWithLogitsLoss)


def test_build_criterion_selects_group_dro():
    cfg = Config({"mitigation": {"objective": "group_dro", "dro": {"step_size": 0.05}}})
    criterion = build_criterion(cfg)

    assert isinstance(criterion, GroupDROLoss)
    assert criterion.step_size == pytest.approx(0.05)


def test_an_unknown_objective_is_refused():
    cfg = Config({"mitigation": {"objective": "magic"}})
    with pytest.raises(ValueError, match="unknown mitigation.objective"):
        build_criterion(cfg)


def test_weighted_bce_ignores_groups(config):
    """Both objectives share one signature, so the training loop needs no branch."""
    criterion = WeightedBCEWithLogitsLoss()
    logits, targets = torch.tensor([1.0, -1.0]), torch.tensor([1.0, 0.0])
    groups = torch.tensor([0, 3], dtype=torch.long)

    assert criterion(logits, targets, None, groups) == pytest.approx(
        float(criterion(logits, targets, None, None))
    )


def test_the_dataset_carries_the_cell_id():
    frame = pd.DataFrame(
        {"id": [0, 1], "comment_text": ["a", "b"], "label": [0, 1], "any_identity": [1, 1]}
    )
    ids = compute_group_ids(frame)

    assert ids.tolist() == [2, 3]

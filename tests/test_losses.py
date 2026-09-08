"""The weighting scheme is the intervention, so it gets the closest tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from fairtox.training.losses import (
    SCHEMES,
    WeightedBCEWithLogitsLoss,
    compute_sample_weights,
    is_uniform,
)


def test_none_scheme_is_uniform(cells, config):
    weights = compute_sample_weights(cells, "none", config)
    assert np.allclose(weights, 1.0)


def test_subgroup_scheme_targets_the_right_cell(cells, config):
    """Only identity-bearing, non-toxic rows may be upweighted."""
    weights = compute_sample_weights(cells, "subgroup", config, alpha=4.0)
    identity = cells["any_identity"].to_numpy() == 1
    labels = cells["label"].to_numpy()

    assert np.allclose(weights[identity & (labels == 0)], 4.0)
    assert np.allclose(weights[identity & (labels == 1)], 1.0)
    assert np.allclose(weights[~identity], 1.0)


@pytest.mark.parametrize("alpha", [1.0, 2.0, 4.0, 6.0, 10.0])
def test_alpha_flows_through(cells, config, alpha):
    weights = compute_sample_weights(cells, "subgroup", config, alpha=alpha)
    target_cell = ((cells["any_identity"] == 1) & (cells["label"] == 0)).to_numpy()
    assert np.allclose(weights[target_cell], alpha)


def test_class_scheme_ignores_identity(cells, config):
    """Class weighting must depend on the label alone -- it is the control arm."""
    weights = compute_sample_weights(cells, "class", config)
    labels = cells["label"].to_numpy()
    for label in (0, 1):
        assert len(np.unique(np.round(weights[labels == label], 8))) == 1


def test_unknown_scheme_rejected(cells, config):
    with pytest.raises(ValueError, match="unknown weighting scheme"):
        compute_sample_weights(cells, "not_a_scheme", config)


def test_all_schemes_produce_finite_positive_weights(cells, config):
    for scheme in SCHEMES:
        weights = compute_sample_weights(cells, scheme, config)
        assert len(weights) == len(cells)
        assert np.all(np.isfinite(weights)) and np.all(weights > 0)


# -- the ablation's compute saving ------------------------------------------


def test_subgroup_at_alpha_one_is_uniform(cells, config):
    """Why the sweep does not train alpha=1.0.

    Every weight is 1.0, so the arm is arithmetically the unweighted baseline.
    Training it would spend a GPU run reproducing a model already on disk.
    """
    weights = compute_sample_weights(cells, "subgroup", config, alpha=1.0)
    assert is_uniform(weights)


def test_none_scheme_is_detected_as_uniform(cells, config):
    assert is_uniform(compute_sample_weights(cells, "none", config))


def test_real_mitigation_is_not_uniform(cells, config):
    assert not is_uniform(compute_sample_weights(cells, "subgroup", config, alpha=4.0))


def test_class_weighting_on_imbalanced_data_is_not_uniform(cells, config):
    assert not is_uniform(compute_sample_weights(cells, "class", config))


# -- the loss ---------------------------------------------------------------


def test_uniform_weights_reproduce_plain_bce():
    """Normalisation must not change the loss when weights carry no information."""
    criterion = WeightedBCEWithLogitsLoss()
    logits = torch.tensor([0.5, -1.2, 2.0, -0.3])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

    weighted = criterion(logits.unsqueeze(-1), targets, torch.ones(4))
    reference = F.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(weighted, reference)


def test_loss_is_invariant_to_weight_scale():
    """The property that stops alpha acting as a hidden learning-rate multiplier.

    Normalising by the sum of weights rather than the batch size means doubling
    every weight leaves the loss -- and so the gradient magnitude -- unchanged.
    Without it, a larger alpha would partly be a larger learning rate and the
    ablation would be measuring that confound.
    """
    criterion = WeightedBCEWithLogitsLoss()
    logits = torch.tensor([0.5, -1.2, 2.0, -0.3]).unsqueeze(-1)
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    weights = torch.tensor([1.0, 4.0, 1.0, 4.0])

    assert torch.allclose(
        criterion(logits, targets, weights), criterion(logits, targets, weights * 7.5)
    )


def test_gradient_magnitude_does_not_scale_with_alpha():
    """The same property, measured where it actually matters: on the gradient."""
    criterion = WeightedBCEWithLogitsLoss()
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    weights = torch.tensor([1.0, 4.0, 1.0, 4.0])

    grads = []
    for scale in (1.0, 10.0):
        logits = torch.tensor([0.5, -1.2, 2.0, -0.3], requires_grad=True)
        criterion(logits.unsqueeze(-1), targets, weights * scale).backward()
        grads.append(logits.grad.clone())

    assert torch.allclose(grads[0], grads[1])


def test_upweighting_moves_the_loss_toward_the_upweighted_rows():
    criterion = WeightedBCEWithLogitsLoss()
    # Row 0 is predicted badly, row 1 well.
    logits = torch.tensor([-3.0, 3.0]).unsqueeze(-1)
    targets = torch.tensor([1.0, 1.0])

    uniform = criterion(logits, targets, torch.ones(2))
    emphasise_bad = criterion(logits, targets, torch.tensor([9.0, 1.0]))
    assert emphasise_bad > uniform


def test_zero_weight_is_honoured_not_treated_as_unset(cells, config):
    """A configured weight of 0.0 means 'drop this cell', not 'use the default'.

    Regression: `cfg.get(k) or 1.0` silently turns a deliberate 0.0 into 1.0 and
    runs a different experiment than the config describes.
    """
    config.set(
        "mitigation.weights",
        {"background": 0.0, "identity_nontoxic": 3.0, "identity_toxic": 0.0},
    )
    weights = compute_sample_weights(cells, "subgroup", config)
    identity = cells["any_identity"].to_numpy() == 1
    labels = cells["label"].to_numpy()

    assert np.allclose(weights[~identity], 0.0)
    assert np.allclose(weights[identity & (labels == 1)], 0.0)
    assert np.allclose(weights[identity & (labels == 0)], 3.0)


def test_explicit_identity_nontoxic_weight_overrides_alpha(cells, config):
    config.set(
        "mitigation.weights",
        {"background": 1.0, "identity_nontoxic": 2.5, "identity_toxic": 1.0},
    )
    weights = compute_sample_weights(cells, "subgroup", config, alpha=99.0)
    target_cell = ((cells["any_identity"] == 1) & (cells["label"] == 0)).to_numpy()
    assert np.allclose(weights[target_cell], 2.5)

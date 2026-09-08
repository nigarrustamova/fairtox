"""Narrowing the weighting to a subset of identity axes.

The held-out-group experiment rests on one asymmetry: the LOSS sees some axes,
the AUDIT sees all of them. If that ever collapses into "train and score on the
same seven groups", the experiment measures nothing and still produces a
plausible-looking number, so both halves are pinned here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairtox.config import Config
from fairtox.provenance import model_fingerprint
from fairtox.training.losses import compute_sample_weights, weighting_columns, weighting_mask

GENDER = ["male", "female"]


@pytest.fixture
def frame() -> pd.DataFrame:
    """Four rows: one gender-only, one race-only, one both, one neither."""
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "label": [0, 0, 0, 0],
            "male": [1, 0, 1, 0],
            "female": [0, 0, 0, 0],
            "black": [0, 1, 1, 0],
            "any_identity": [1, 1, 1, 0],
        }
    )


def _config(**mitigation) -> Config:
    base = {"scheme": "subgroup", "alpha": 4.0, "weights": {}}
    base.update(mitigation)
    return Config({"run": {"seed": 42}, "mitigation": base})


def test_none_means_every_column(frame):
    mask = weighting_mask(frame, _config())
    assert mask.tolist() == [True, True, True, False]


def test_a_subset_hides_the_other_axes(frame):
    """Row 1 mentions only a held-out group, so the loss must not see it."""
    mask = weighting_mask(frame, _config(weighting_columns=GENDER))
    assert mask.tolist() == [True, False, True, False]


def test_a_row_in_both_still_counts(frame):
    mask = weighting_mask(frame, _config(weighting_columns=GENDER))
    assert bool(mask[2]) is True


def test_the_audit_column_is_left_alone(frame):
    """`any_identity` drives the audit and must survive the narrowing untouched."""
    before = frame["any_identity"].tolist()
    weighting_mask(frame, _config(weighting_columns=GENDER))
    assert frame["any_identity"].tolist() == before


def test_weights_follow_the_narrowed_mask(frame):
    weights = compute_sample_weights(frame, "subgroup", _config(weighting_columns=GENDER))
    # rows 0 and 2 are visible to the loss and non-toxic -> alpha; the rest are 1.0
    assert weights.tolist() == pytest.approx([4.0, 1.0, 4.0, 1.0])


def test_full_weighting_upweights_every_identity_row(frame):
    weights = compute_sample_weights(frame, "subgroup", _config())
    assert weights.tolist() == pytest.approx([4.0, 4.0, 4.0, 1.0])


def test_columns_absent_from_the_corpus_are_dropped(frame):
    mask = weighting_mask(frame, _config(weighting_columns=["male", "klingon"]))
    assert mask.tolist() == [True, False, True, False]


def test_a_subset_matching_nothing_is_refused(frame):
    """Otherwise the arm is the baseline under another name."""
    with pytest.raises(ValueError, match="matched no column"):
        weighting_mask(frame, _config(weighting_columns=["klingon"]))


def test_helper_reports_none_for_the_default():
    assert weighting_columns(_config()) is None
    assert weighting_columns(_config(weighting_columns=[])) is None
    assert weighting_columns(_config(weighting_columns=GENDER)) == GENDER


# -- provenance --------------------------------------------------------------


def _fp_config(**mitigation) -> Config:
    base = {"weights": {"background": 1.0}}
    base.update(mitigation)
    return Config({"run": {"seed": 42}, "training": {}, "mitigation": base})


def test_the_default_leaves_old_fingerprints_untouched():
    assert model_fingerprint(_fp_config(), "subgroup", 4.0) == model_fingerprint(
        _fp_config(weighting_columns=None), "subgroup", 4.0
    )


def test_narrowing_changes_the_fingerprint():
    """A narrowed arm must not be able to reuse the full arm's artefacts."""
    full = model_fingerprint(_fp_config(), "subgroup", 4.0)
    narrow = model_fingerprint(_fp_config(weighting_columns=GENDER), "subgroup", 4.0)
    assert full != narrow


def test_the_fingerprint_ignores_the_order_of_the_columns():
    a = model_fingerprint(_fp_config(weighting_columns=["male", "female"]), "subgroup", 4.0)
    b = model_fingerprint(_fp_config(weighting_columns=["female", "male"]), "subgroup", 4.0)
    assert a == b


def test_the_shipped_config_splits_the_reportable_groups_evenly():
    """The seven/seven split is the experiment; a typo in the list would break it."""
    cfg = Config.load("configs/a100_heldout.yaml")
    weighted = set(cfg.get("mitigation.weighting_columns"))
    reportable = {
        "male", "female", "transgender", "heterosexual", "homosexual_gay_or_lesbian",
        "christian", "jewish", "muslim", "atheist", "black", "white", "asian",
        "latino", "psychiatric_or_mental_illness",
    }
    seen = reportable & weighted
    held_out = reportable - weighted

    assert len(seen) == 7, sorted(seen)
    assert len(held_out) == 7, sorted(held_out)
    assert np.isclose(cfg.get("mitigation.alpha"), 4.0)
    assert cfg.get("mitigation.scheme") == "subgroup"

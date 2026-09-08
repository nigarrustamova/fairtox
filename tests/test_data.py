"""Split integrity. Test-set leakage is an automatic deduction, so it gets tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fairtox.data import load as data_load
from fairtox.data import split as data_split


def test_splits_are_disjoint(corpus, config, tmp_path):
    config.set("data.processed_dir", str(tmp_path))
    splits = data_split.make_splits(corpus, config)
    ids = {name: set(part["id"]) for name, part in splits.items()}

    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_splits_partition_the_whole_corpus(corpus, config, tmp_path):
    config.set("data.processed_dir", str(tmp_path))
    splits = data_split.make_splits(corpus, config)

    assert sum(len(part) for part in splits.values()) == len(corpus)
    covered = set().union(*(set(part["id"]) for part in splits.values()))
    assert covered == set(corpus["id"])


def test_split_is_deterministic_under_a_fixed_seed(corpus, config, tmp_path):
    config.set("data.processed_dir", str(tmp_path))
    first = data_split.make_splits(corpus, config)
    second = data_split.make_splits(corpus, config)
    for name in ("train", "val", "test"):
        assert list(first[name]["id"]) == list(second[name]["id"])


def test_split_fingerprint_is_stable_and_seed_sensitive(corpus, config, tmp_path):
    """The guard that stops a comparison across two different partitions."""
    config.set("data.processed_dir", str(tmp_path))
    first = data_split.fingerprint(data_split.make_splits(corpus, config))
    again = data_split.fingerprint(data_split.make_splits(corpus, config))
    assert first == again

    config.set("run.seed", 1234)
    different = data_split.fingerprint(data_split.make_splits(corpus, config))
    assert different != first


def test_compare_refuses_mismatched_splits():
    with pytest.raises(ValueError, match="DIFFERENT splits"):
        data_split.assert_same_split("aaaa", "bbbb", "compare(baseline, mitigated)")


def test_matching_splits_pass_the_guard():
    data_split.assert_same_split("aaaa", "aaaa", "compare(baseline, mitigated)")


def test_split_fractions_are_respected(corpus, config, tmp_path):
    config.set("data.processed_dir", str(tmp_path))
    splits = data_split.make_splits(corpus, config)
    assert len(splits["train"]) == pytest.approx(0.75 * len(corpus), abs=2)
    assert len(splits["val"]) == pytest.approx(0.10 * len(corpus), abs=2)
    assert len(splits["test"]) == pytest.approx(0.15 * len(corpus), abs=2)


def test_fractions_must_sum_to_one(corpus, config):
    config.set("data.splits", {"train": 0.8, "val": 0.3, "test": 0.1})
    with pytest.raises(ValueError, match="must sum to 1.0"):
        data_split.make_splits(corpus, config)


def test_stratification_preserves_class_and_identity_balance(corpus, config, tmp_path):
    """Subgroup metrics are undefined if a split has no identity-bearing rows."""
    config.set("data.processed_dir", str(tmp_path))
    splits = data_split.make_splits(corpus, config)
    overall_toxic = corpus["label"].mean()
    overall_identity = corpus["any_identity"].mean()

    for part in splits.values():
        assert part["label"].mean() == pytest.approx(overall_toxic, abs=0.10)
        assert part["any_identity"].mean() == pytest.approx(overall_identity, abs=0.10)


# -- preparation -------------------------------------------------------------


def _raw(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "comment_text": ["a", "b", "c", "d"],
            "target": [0.0, 0.49, 0.5, 1.0],
            "identity_annotator_count": [4, 4, 4, 4],
        }
    )
    for key, value in overrides.items():
        frame[key] = value
    return frame


def test_binarisation_uses_the_configured_threshold(config):
    config.set("data.identity_columns", [])
    prepared = data_load.prepare(_raw(), config)
    assert list(prepared["label"]) == [0, 0, 1, 1]

    config.set("data.toxicity_threshold", 0.4)
    assert list(data_load.prepare(_raw(), config)["label"]) == [0, 1, 1, 1]


def test_annotated_only_filters_unrated_rows(config):
    """An unannotated row is not an identity-free row -- it must be excluded."""
    raw = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "comment_text": ["a", "b", "c"],
            "target": [0.1, 0.2, 0.9],
            "identity_annotator_count": [0, 5, 8],
        }
    )
    config.set("data.identity_columns", [])
    prepared = data_load.prepare(raw, config)
    assert list(prepared["id"]) == [1, 2]

    config.set("data.annotated_only", False)
    assert len(data_load.prepare(raw, config)) == 3


def test_empty_comments_are_dropped(config):
    raw = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "comment_text": ["real comment", "   ", ""],
            "target": [0.1, 0.2, 0.9],
            "identity_annotator_count": [4, 4, 4],
        }
    )
    config.set("data.identity_columns", [])
    assert len(data_load.prepare(raw, config)) == 1


def test_duplicate_ids_are_rejected(config):
    """Predictions are joined back to text by id, so duplicates are unusable."""
    raw = pd.DataFrame(
        {
            "id": [7, 7],
            "comment_text": ["a", "b"],
            "target": [0.1, 0.9],
            "identity_annotator_count": [4, 4],
        }
    )
    config.set("data.identity_columns", [])
    with pytest.raises(ValueError, match="duplicates"):
        data_load.prepare(raw, config)


def test_identity_flags_are_binarised_and_any_identity_derived(config):
    raw = pd.DataFrame(
        {
            "id": [0, 1, 2],
            "comment_text": ["a", "b", "c"],
            "target": [0.1, 0.1, 0.9],
            "identity_annotator_count": [4, 4, 4],
            "group_a": [0.9, 0.1, 0.0],
            "group_b": [0.0, 0.0, 0.6],
        }
    )
    prepared = data_load.prepare(raw, config)
    assert list(prepared["group_a"]) == [1, 0, 0]
    assert list(prepared["group_b"]) == [0, 0, 1]
    assert list(prepared["any_identity"]) == [1, 0, 1]


def test_subsample_preserves_composition(corpus, config):
    """Subsampling must not quietly remove the identity-bearing non-toxic cell."""
    config.set("data.subsample", 100)
    config.set("data.annotated_only", False)
    corpus = corpus.drop(columns=["label", "any_identity"])

    prepared = data_load.prepare(corpus, config)
    assert 90 <= len(prepared) <= 110
    assert prepared["any_identity"].sum() > 0
    assert ((prepared["any_identity"] == 1) & (prepared["label"] == 0)).sum() > 0


def test_subsample_is_a_no_op_when_larger_than_the_corpus(corpus, config):
    config.set("data.subsample", 10_000)
    config.set("data.annotated_only", False)
    prepared = data_load.prepare(corpus.drop(columns=["label", "any_identity"]), config)
    assert len(prepared) == len(corpus)


# -- where the corpus is read from -------------------------------------------


def test_a_relative_raw_dir_is_taken_from_the_repository_root(config):
    from fairtox.config import REPO_ROOT
    from fairtox.data.load import raw_csv_path

    config.set("data.raw_dir", "data/raw")
    config.set("data.raw_file", "all_data.csv")
    assert raw_csv_path(config) == REPO_ROOT / "data/raw/all_data.csv"


def test_an_absolute_raw_dir_is_used_as_given(config):
    """Read-only mounts such as Kaggle's /kaggle/input are read in place."""
    from fairtox.data.load import raw_csv_path

    config.set("data.raw_dir", "/kaggle/input/jigsaw")
    config.set("data.raw_file", "all_data.csv")
    assert str(raw_csv_path(config)).replace("\\", "/").endswith(
        "/kaggle/input/jigsaw/all_data.csv"
    )


def test_a_tilde_path_is_expanded_not_joined_onto_the_repo(config):
    """Regression: `~/corpus` is not absolute, so it was joined onto the repo root.

    The loader then reported a missing file at a path containing a literal `~`
    directory inside the checkout -- a location nobody had typed.
    """
    from fairtox.config import REPO_ROOT
    from fairtox.data.load import raw_csv_path

    config.set("data.raw_dir", "~/fairtox_data")
    config.set("data.raw_file", "all_data.csv")
    resolved = raw_csv_path(config)

    assert "~" not in str(resolved)
    assert REPO_ROOT not in resolved.parents
    assert resolved == Path.home() / "fairtox_data" / "all_data.csv"

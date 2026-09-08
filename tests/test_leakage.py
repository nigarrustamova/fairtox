"""The duplicate-comment check.

The corpus is partitioned by row, not by text, and it contains reposts: 771
distinct comments occur in more than one split. That does not touch the paired
mitigation effect -- both arms share the partition -- but it does inflate every
absolute figure, and the difference between those two statements is the whole
point of this stage.
"""

from __future__ import annotations

import pandas as pd

from fairtox.stages.leakage import contaminated_ids, normalise


def _splits(train_texts, test_texts):
    def frame(texts, start):
        return pd.DataFrame(
            {
                "id": range(start, start + len(texts)),
                "comment_text": texts,
                "label": [0] * len(texts),
                "any_identity": [1] * len(texts),
            }
        )

    return {"train": frame(train_texts, 0), "test": frame(test_texts, 100)}


def test_normalisation_collapses_case_and_whitespace():
    out = normalise(pd.Series(["  Hello   WORLD ", "hello world"]))
    assert out.tolist() == ["hello world", "hello world"]


def test_an_exact_repost_is_caught():
    ids, summary = contaminated_ids(_splits(["a repeated comment"], ["a repeated comment"]))

    assert ids == {100}
    assert summary["n_test_seen_in_train"] == 1
    assert summary["share_of_test"] == 1.0


def test_a_repost_with_different_spacing_is_caught():
    """Exact matching would miss most reposts, which is why matching is normalised."""
    ids, _ = contaminated_ids(_splits(["A Repeated  Comment"], ["a repeated comment"]))

    assert ids == {100}


def test_unseen_test_rows_are_left_alone():
    ids, summary = contaminated_ids(_splits(["something"], ["something else entirely"]))

    assert ids == set()
    assert summary["n_test_seen_in_train"] == 0


def test_conflicting_labels_among_duplicates_are_counted():
    """Annotator disagreement, not leakage -- but it is the same scan."""
    splits = _splits(["same text", "same text", "other"], ["unrelated"])
    splits["train"]["label"] = [0, 1, 0]
    _, summary = contaminated_ids(splits)

    assert summary["n_train_duplicate_rows"] == 2
    assert summary["n_duplicate_texts_with_conflicting_labels"] == 1


def test_agreeing_duplicates_are_not_counted_as_conflicting():
    splits = _splits(["same text", "same text"], ["unrelated"])
    splits["train"]["label"] = [1, 1]
    _, summary = contaminated_ids(splits)

    assert summary["n_train_duplicate_rows"] == 2
    assert summary["n_duplicate_texts_with_conflicting_labels"] == 0


def test_a_corpus_without_duplicates_reports_none():
    _, summary = contaminated_ids(_splits(["one", "two"], ["three"]))

    assert summary["n_train_duplicate_rows"] == 0
    assert summary["n_duplicate_texts_with_conflicting_labels"] == 0

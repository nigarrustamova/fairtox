"""Counterfactual identity-term swapping.

Every assertion here is about a property the METHOD needs, not about a string the
current term table happens to produce: swaps stay on one axis, keep grammatical
number, stay consistent inside a comment, and never touch the caller's frame. A
table entry can be added tomorrow without rewriting these tests.

The loudest test is the last one. An augmentation that silently rewrites nothing
trains a model identical to the baseline, and the run would be written up as
"the method does nothing" -- the same shape of failure as the tokenizer bug,
where an artefact read as a result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairtox.data.augment import _index, rewrite, swap_identity_terms


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _groups_in(text: str) -> list[tuple[str, str, str]]:
    """Every (axis, column, slot) the term index finds in ``text``.

    Group 2 is the surface; group 1 is the indefinite article the pattern swallows
    along with it so the swap can put the right one back.
    """
    pattern, lookup, _, _ = _index()
    return [lookup[m.group(2).lower()] for m in pattern.finditer(text)]


# -- what a swap is allowed to change ----------------------------------------


def test_swap_stays_on_the_same_axis():
    """A religion becomes another religion, never a race or a gender."""
    out, swapped = rewrite("i talked to a muslim yesterday", _rng())

    assert swapped == 1
    axes = {axis for axis, _, _ in _groups_in(out)}
    assert axes == {"religion"}


def test_swap_changes_the_group():
    out, _ = rewrite("a muslim wrote this", _rng())
    columns = {column for _, column, _ in _groups_in(out)}

    assert "muslim" not in columns
    assert len(columns) == 1


def test_number_agreement_is_kept():
    """A plural swaps to a plural: `muslims` must not become `christian`."""
    out, _ = rewrite("muslims commented on the article", _rng())
    slots = {slot for _, _, slot in _groups_in(out)}

    assert slots == {"pl"}


def test_one_comment_gets_one_target():
    """Both mentions move together, or the sentence is about two groups at once."""
    out, swapped = rewrite("muslims say this because a muslim believes it", _rng())
    columns = {column for _, column, _ in _groups_in(out)}

    assert swapped == 2
    assert len(columns) == 1


def test_capitalisation_is_carried_over():
    out, _ = rewrite("Muslims disagree", _rng())

    assert out[0].isupper()


def test_the_indefinite_article_is_corrected():
    """`a muslim` -> `a christian` or `an atheist`, never `a atheist`."""
    for seed in range(12):
        out, swapped = rewrite("i met a muslim today", np.random.default_rng(seed))
        assert swapped == 1
        term = out.split("met ")[1].split(" today")[0]
        article, noun = term.split(" ", 1)
        assert article == ("an" if noun[0] in "aeiou" else "a")


def test_the_article_keeps_its_capitalisation():
    out, _ = rewrite("A muslim replied", np.random.default_rng(0))

    assert out[0].isupper()
    assert out.split(" ")[0] in {"A", "An"}


def test_longer_surfaces_match_first():
    """`black people` is one term, not `black` with `people` left behind."""
    out, swapped = rewrite("black people are discussed here", _rng())

    assert swapped == 1
    assert {slot for _, _, slot in _groups_in(out)} == {"pl"}


def test_word_boundaries_are_respected():
    """`man` inside `chairman` is not an identity mention."""
    out, swapped = rewrite("the chairman opened the meeting", _rng())

    assert swapped == 0
    assert out == "the chairman opened the meeting"


def test_column_surfaces_are_recognised():
    """The synthetic corpus writes the column name, so the smoke test must match it."""
    out, swapped = rewrite("as a homosexual gay or lesbian person, hello", _rng())

    assert swapped == 1
    assert "homosexual gay or lesbian" not in out


# -- what a swap must not touch ----------------------------------------------


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": range(6),
            "comment_text": [
                "a muslim wrote this",
                "christians disagree with it",
                "black people are discussed",
                "you are an idiot and a muslim",
                "women replied to the thread",
                "nothing identifying here at all",
            ],
            "label": [0, 0, 0, 1, 0, 1],
            "any_identity": [1, 1, 1, 1, 1, 0],
        }
    )


def test_the_callers_frame_is_not_mutated():
    """The split is cached and reused, so an in-place edit would leak into every arm."""
    frame = _frame()
    before = frame["comment_text"].tolist()

    swap_identity_terms(frame, probability=1.0, seed=1)

    assert frame["comment_text"].tolist() == before


def test_labels_and_identity_flags_survive():
    frame = _frame()
    out, _ = swap_identity_terms(frame, probability=1.0, seed=1)

    pd.testing.assert_series_equal(out["label"], frame["label"])
    pd.testing.assert_series_equal(out["any_identity"], frame["any_identity"])
    assert len(out) == len(frame)


def test_nontoxic_scope_leaves_toxic_rows_alone():
    frame = _frame()
    out, stats = swap_identity_terms(frame, probability=1.0, seed=1, scope="nontoxic")

    toxic = frame["label"] == 1
    assert out.loc[toxic, "comment_text"].tolist() == frame.loc[toxic, "comment_text"].tolist()
    assert stats["n_eligible"] == int((~toxic).sum())


def test_zero_probability_is_a_clean_no_op():
    frame = _frame()
    out, stats = swap_identity_terms(frame, probability=0.0, seed=1)

    assert out["comment_text"].tolist() == frame["comment_text"].tolist()
    assert stats["n_rows_rewritten"] == 0


# -- reproducibility ---------------------------------------------------------


def test_same_seed_gives_the_same_corpus():
    first, _ = swap_identity_terms(_frame(), probability=1.0, seed=5)
    second, _ = swap_identity_terms(_frame(), probability=1.0, seed=5)

    assert first["comment_text"].tolist() == second["comment_text"].tolist()


def test_a_different_seed_gives_a_different_corpus():
    first, _ = swap_identity_terms(_frame(), probability=1.0, seed=5)
    second, _ = swap_identity_terms(_frame(), probability=1.0, seed=6)

    assert first["comment_text"].tolist() != second["comment_text"].tolist()


# -- refusals ----------------------------------------------------------------


def test_a_silent_no_op_is_refused():
    """The failure this guard exists for: an augmented arm that is the baseline."""
    frame = pd.DataFrame(
        {
            "id": [0, 1],
            "comment_text": ["nothing here", "nor here either"],
            "label": [0, 1],
            "any_identity": [0, 0],
        }
    )
    with pytest.raises(RuntimeError, match="rewrote NO rows"):
        swap_identity_terms(frame, probability=1.0, seed=1)


def test_probability_outside_the_unit_interval_is_refused():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        swap_identity_terms(_frame(), probability=1.5, seed=1)


def test_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="unknown augmentation scope"):
        swap_identity_terms(_frame(), probability=1.0, seed=1, scope="toxic-only")


def test_a_missing_text_column_is_named():
    frame = _frame().drop(columns=["comment_text"])
    with pytest.raises(KeyError, match="comment_text"):
        swap_identity_terms(frame, probability=1.0, seed=1)

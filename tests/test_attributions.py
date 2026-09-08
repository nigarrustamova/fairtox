"""Identity-token matching in the attribution summary.

The share of attribution mass sitting on identity tokens is the number the
paired comparison turns on, and it is computed by matching token surface forms
against a term list. Every tokenizer marks word boundaries differently, so the
normalisation that strips those markers is the whole of the measurement -- and
when it silently fails, the stage still produces a plausible-looking number.
"""

from __future__ import annotations

import pytest

from fairtox.explain.attributions import identity_attribution_mass

TERMS = {"muslim", "black", "woman", "gay"}


def _explanation(tokens: list[str], attributions: list[float]) -> dict:
    return {"tokens": tokens, "attributions": attributions, "prob": 0.9}


def test_wordpiece_continuations_are_stripped():
    """BERT-family: "##im" is part of a word, not a token in its own right."""
    result = identity_attribution_mass(
        _explanation(["[CLS]", "mus", "##lim", "woman", "[SEP]"], [0.0, 0.1, 0.1, 0.8, 0.0]),
        TERMS,
    )
    assert result["matched_tokens"] == ["woman"]
    assert result["identity_share"] == pytest.approx(0.8)


def test_byte_level_boundary_markers_are_stripped():
    """Regression: RoBERTa's U+0120 marker survived normalisation.

    The prefix was removed *after* lowercasing, and lowercasing U+0120 yields
    U+0121, which ``removeprefix("Ġ")`` never matches. Measured on
    distilroberta-base, two identity tokens matched across twenty explanations
    instead of eighty, and the reported identity share -- 0.0040 against 0.1164
    for DistilBERT -- was an artefact of the tokenizer, not a property of the
    model. Nothing in the output looked wrong.
    """
    result = identity_attribution_mass(
        _explanation(["<s>", "ĠMuslim", "Ġwoman", "</s>"], [0.0, 0.6, 0.4, 0.0]),
        TERMS,
    )
    assert result["matched_tokens"] == ["ĠMuslim", "Ġwoman"]
    assert result["identity_share"] == pytest.approx(1.0)


def test_matching_is_case_insensitive():
    result = identity_attribution_mass(
        _explanation(["Black", "people"], [0.5, 0.5]), TERMS
    )
    assert result["matched_tokens"] == ["Black"]


def test_no_identity_token_reports_zero_not_an_error():
    result = identity_attribution_mass(_explanation(["hello", "there"], [0.5, 0.5]), TERMS)
    assert result["identity_share"] == 0.0
    assert result["matched_tokens"] == []


def test_zero_total_mass_does_not_divide_by_zero():
    result = identity_attribution_mass(_explanation(["muslim"], [0.0]), TERMS)
    assert result["identity_share"] == 0.0


def test_the_share_is_taken_over_absolute_mass():
    """Attribution is signed; a term pushing the logit down still carries mass."""
    result = identity_attribution_mass(
        _explanation(["muslim", "cake"], [-0.6, 0.4]), TERMS
    )
    assert result["identity_share"] == pytest.approx(0.6)

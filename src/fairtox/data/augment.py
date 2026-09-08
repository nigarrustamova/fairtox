"""Counterfactual identity-term swapping -- the pre-processing arm.

Loss reweighting attacks the shortcut from the *loss*: it makes the
identity-bearing non-toxic cell expensive to get wrong. This attacks the same
shortcut from the *data*: if ``muslim`` and ``christian`` occur in the same
sentences with the same labels, neither token carries information about toxicity
and there is no shortcut left to learn. It is the method the Jigsaw bias
benchmark was introduced with (Dixon et al., 2018), which is why a study of one
mitigation family has to be measured against it rather than only cite it.

The two interventions rest on different inputs, and that is the point of running
both:

    reweighting  needs identity ANNOTATIONS on the training rows; changes the loss.
    swapping     needs an identity TERM LIST; changes the text.

The load-bearing assumption is Dixon et al.'s -- that toxicity is invariant under
substituting one identity term for another. It is close to true for abuse and
plainly false for statements of fact ("muslims are a minority in France"), so it
is reported as an assumption of the method, not hidden inside it.

Swaps are made IN PLACE, never appended. Doubling the training set would change
the number of optimizer steps, the wall-clock cost and the effective epoch count
all at once, and the comparison against the weighted arm would stop being a
comparison of mitigations.
"""

from __future__ import annotations

import re
from copy import deepcopy
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

SCOPES = ("all", "nontoxic")

# Surface forms, grouped by identity column and by the grammatical slot they
# fill. A swap replaces a matched surface with the SAME slot from a different
# column on the SAME axis, so `muslims` becomes `christians` and not `christian`
# -- number agreement is what keeps the rewritten sentence readable, and an
# unreadable sentence would train the model on a distribution no user produces.
#
# The first surface in each slot is the one substituted IN; the rest are only
# recognised on the way out.
#
# Two exclusions are deliberate. The five `other_*` columns have no surface term
# ("other race or ethnicity" is a codebook label, not something anyone writes).
# And a polysemous surface is admitted only when its column is one the audit
# actually reports on: `black` and `white` carry non-identity senses and are kept
# because they are the study's headline subgroups, while `straight` is dropped
# because `heterosexual` never clears the reporting floor, so admitting it would
# buy noise and nothing else.
IDENTITY_AXES: dict[str, dict[str, dict[str, list[str]]]] = {
    "gender": {
        "male": {"sg": ["man"], "pl": ["men"], "adj": ["male"]},
        "female": {"sg": ["woman"], "pl": ["women"], "adj": ["female"]},
        "transgender": {
            "sg": ["transgender person"],
            "pl": ["transgender people"],
            "adj": ["transgender"],
        },
    },
    "orientation": {
        "heterosexual": {"adj": ["heterosexual"], "pl": ["heterosexuals"]},
        "homosexual_gay_or_lesbian": {
            "adj": ["gay", "homosexual", "lesbian"],
            "pl": ["gay people", "homosexuals", "lesbians"],
        },
        "bisexual": {"adj": ["bisexual"], "pl": ["bisexuals"]},
    },
    "religion": {
        "christian": {
            "sg": ["christian"], "pl": ["christians"],
            "adj": ["christian"], "abs": ["christianity"],
        },
        "jewish": {
            "sg": ["jew"], "pl": ["jews"], "adj": ["jewish"], "abs": ["judaism"],
        },
        "muslim": {
            "sg": ["muslim"], "pl": ["muslims"], "adj": ["muslim"], "abs": ["islam"],
        },
        "hindu": {
            "sg": ["hindu"], "pl": ["hindus"], "adj": ["hindu"], "abs": ["hinduism"],
        },
        "buddhist": {
            "sg": ["buddhist"], "pl": ["buddhists"],
            "adj": ["buddhist"], "abs": ["buddhism"],
        },
        "atheist": {
            "sg": ["atheist"], "pl": ["atheists"], "adj": ["atheist"], "abs": ["atheism"],
        },
    },
    "race": {
        "black": {"adj": ["black"], "pl": ["black people", "blacks"]},
        "white": {"adj": ["white"], "pl": ["white people", "whites"]},
        "asian": {"adj": ["asian"], "pl": ["asian people", "asians"]},
        "latino": {"adj": ["latino", "latina", "hispanic"], "pl": ["latinos", "hispanics"]},
    },
    "disability": {
        "physical_disability": {
            "adj": ["physically disabled"], "pl": ["disabled people"],
        },
        "intellectual_or_learning_disability": {
            "adj": ["learning disabled"], "pl": ["people with learning disabilities"],
        },
        "psychiatric_or_mental_illness": {
            "adj": ["mentally ill"], "pl": ["people with mental illness"],
        },
    },
}

# Registration order for the surface -> slot index. A surface that fills two
# slots (`christian` is both the noun and the adjective) is registered under the
# first one listed, so `a christian man` swaps to `a jewish man` rather than to
# the ungrammatical `a jew man`. `col` sits last because it exists only to catch
# the synthetic corpus and would otherwise outrank a real surface form.
#
# No slot table resolves every case without a part-of-speech tagger: `i am a
# muslim` uses the noun where this index sees the adjective, so it rewrites to
# `i am a jewish`. Five of the six religions have identical noun and adjective
# forms, and every other axis is clean, so the residue is small -- but it is a
# limitation of the method as implemented, and it is stated rather than tuned.
FORM_ORDER: tuple[str, ...] = ("abs", "pl", "adj", "sg", "col")


def _with_column_surfaces() -> dict[str, dict[str, dict[str, list[str]]]]:
    """The axes with a ``col`` slot derived from each column name.

    Typing those surfaces out by hand would let the table and the column list
    drift apart. Deriving them also means the synthetic corpus -- whose generator
    writes ``as a <column with spaces> person`` -- exercises this code path in the
    smoke test instead of silently matching nothing.
    """
    axes = deepcopy(IDENTITY_AXES)
    for groups in axes.values():
        for column, forms in groups.items():
            forms.setdefault("col", [column.replace("_", " ")])
    return axes


@lru_cache(maxsize=1)
def _index() -> tuple[
    re.Pattern[str],
    dict[str, tuple[str, str, str]],
    dict[str, list[str]],
    dict[str, dict[str, dict[str, list[str]]]],
]:
    """Compile the matcher once: pattern, surface -> slot, axis members, axes."""
    axes = _with_column_surfaces()

    lookup: dict[str, tuple[str, str, str]] = {}
    members: dict[str, list[str]] = {}
    for axis, groups in axes.items():
        members[axis] = sorted(groups)
        for column, forms in groups.items():
            for form in FORM_ORDER:
                for surface in forms.get(form, []):
                    lookup.setdefault(surface.lower(), (axis, column, form))

    # Longest first, so `black people` is consumed before the bare `black` can
    # match its first word and leave `people` stranded next to a new group name.
    #
    # An indefinite article in front of the term is captured with it. `a muslim`
    # swapping to `a atheist` reads as broken English, and text no human wrote is
    # exactly what this method must not put into the training set, so the article
    # is re-emitted to agree with whatever went in.
    surfaces = sorted(lookup, key=len, reverse=True)
    pattern = re.compile(
        r"\b(?:(an?)\s+)?(" + "|".join(re.escape(s) for s in surfaces) + r")\b",
        re.IGNORECASE,
    )
    return pattern, lookup, members, axes


def _substitute(source: str, replacement: str) -> str:
    """Carry the matched token's capitalisation over to the replacement."""
    if source.isupper() and len(source) > 1:
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _article_for(replacement: str, original: str) -> str:
    """``a`` or ``an`` to agree with the replacement, capitalised like the original.

    Every surface in the table starts with a plain consonant or a plain vowel, so
    the first letter decides it. A term like ``hour`` or ``unicorn`` would need a
    pronunciation dictionary; none is in the table, and adding one means revisiting
    this function rather than discovering the problem in the augmented corpus.
    """
    article = "an" if replacement[:1].lower() in "aeiou" else "a"
    return article.capitalize() if original[:1].isupper() else article


def _target_surface(axes: dict[str, Any], axis: str, target: str, form: str) -> str:
    """The target column's surface for ``form``, falling back down the slots."""
    forms = axes[axis][target]
    if forms.get(form):
        return str(forms[form][0])
    for candidate in FORM_ORDER:
        if forms.get(candidate):
            return str(forms[candidate][0])
    return target.replace("_", " ")


def rewrite(text: str, rng: np.random.Generator) -> tuple[str, int]:
    """Swap every identity term in one comment, consistently.

    Both mentions in "muslims say X because a muslim believes Y" move to the same
    target column. Choosing independently per mention would produce a sentence
    whose two halves are about different groups, which is not a counterfactual of
    anything.
    """
    pattern, lookup, members, axes = _index()
    mapping: dict[tuple[str, str], str] = {}
    swapped = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal swapped
        article, source = match.group(1), match.group(2)
        axis, column, form = lookup[source.lower()]
        key = (axis, column)
        if key not in mapping:
            alternatives = [c for c in members[axis] if c != column]
            if not alternatives:
                return match.group(0)
            mapping[key] = alternatives[int(rng.integers(len(alternatives)))]
        swapped += 1
        replacement = _substitute(source, _target_surface(axes, axis, mapping[key], form))
        if article is None:
            return replacement
        return f"{_article_for(replacement, article)} {replacement}"

    return pattern.sub(replace, text), swapped


def swap_identity_terms(
    frame: pd.DataFrame,
    *,
    probability: float = 0.5,
    seed: int = 13,
    scope: str = "all",
    text_column: str = "comment_text",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return a copy of ``frame`` with identity terms counterfactually swapped.

    A COPY, never an edit in place: the split is cached on the run context and
    handed to every model in the run, so mutating it here would quietly train the
    next arm -- and the ablation's arms after it -- on augmented text.

    The identity flag columns are deliberately left alone. A row that mentioned
    some identity still mentions some identity after the swap, so ``any_identity``
    is still correct and the weighting scheme keeps working if the two mitigations
    are ever combined. The per-column flags no longer describe the rewritten text,
    but nothing reads them on the training split except the weighting, and that
    only looks at ``any_identity``.
    """
    if not 0.0 <= float(probability) <= 1.0:
        raise ValueError(f"augmentation probability must be in [0, 1], got {probability}")
    if scope not in SCOPES:
        raise ValueError(
            f"unknown augmentation scope '{scope}' (expected one of {', '.join(SCOPES)})"
        )
    if text_column not in frame.columns:
        raise KeyError(f"no '{text_column}' column to augment; found {list(frame.columns)[:8]}")

    rng = np.random.default_rng(int(seed))
    texts = frame[text_column].astype(str).to_numpy()

    eligible = np.ones(len(frame), dtype=bool)
    if scope == "nontoxic":
        eligible = frame["label"].to_numpy().astype(int) == 0
    drawn = eligible & (rng.random(len(frame)) < float(probability))

    rewritten = texts.copy()
    n_rows_changed, n_terms = 0, 0
    for position in np.flatnonzero(drawn):
        new_text, swapped = rewrite(str(texts[position]), rng)
        if swapped:
            rewritten[position] = new_text
            n_rows_changed += 1
            n_terms += swapped

    if n_rows_changed == 0 and probability > 0 and len(frame):
        # A no-op augmentation trains a model identical to the baseline, and it
        # would be written up as "the method does nothing" -- the same failure
        # mode as the tokenizer bug, where an artefact looked like a result.
        raise RuntimeError(
            "counterfactual augmentation rewrote NO rows. The term list matched nothing "
            "in the training text, so the augmented arm would be the baseline under "
            "another name. Check that the corpus is the real one and that "
            f"'{text_column}' holds comment text."
        )

    out = frame.copy()
    out[text_column] = rewritten
    stats = {
        "probability": float(probability),
        "seed": int(seed),
        "scope": scope,
        "n_rows": int(len(frame)),
        "n_eligible": int(eligible.sum()),
        "n_drawn": int(drawn.sum()),
        "n_rows_rewritten": int(n_rows_changed),
        "n_terms_swapped": int(n_terms),
        "share_rewritten": float(n_rows_changed / max(len(frame), 1)),
    }
    logger.info(
        "counterfactual augmentation | scope=%s p=%.2f seed=%d | rewrote %s of %s rows "
        "(%.1f%%), %s terms swapped",
        scope, probability, seed, f"{n_rows_changed:,}", f"{len(frame):,}",
        100 * stats["share_rewritten"], f"{n_terms:,}",
    )
    return out, stats

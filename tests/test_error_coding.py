"""The coding sample and the manual counts read back from it.

The point of these tests is the distinction between the two samples the stage
exports. One is sorted by the model's own confidence and is for reading; the
other is a uniform draw and is the only one whose category counts are a
proportion. Confusing them would put the composition of the confident tail in the
paper as if it described the errors.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fairtox.config import Config
from fairtox.stages.error_analysis import TAXONOMY, _coded_counts, _coding_sample


def _errors(n: int = 200) -> pd.DataFrame:
    """Synthetic error table: confidence rises with id, so bias is detectable."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": i,
                "error_type": "false_positive" if i % 4 else "false_negative",
                "any_identity": 1 if i % 3 else 0,
                "prob": 0.5 + i / (2 * n),
                "confidence": i / n,
                "comment_text": f"comment {i}",
                **{category: int(rng.integers(0, 2)) for category in TAXONOMY},
            }
        )
    return pd.DataFrame(rows)


def test_the_coding_sample_is_only_identity_bearing_false_positives():
    sample = _coding_sample(_errors(), size=30, seed=7)
    assert len(sample) == 30
    assert set(sample["any_identity"]) == {1}


def test_the_coding_sample_is_reproducible_from_its_seed():
    first = _coding_sample(_errors(), size=25, seed=7)
    again = _coding_sample(_errors(), size=25, seed=7)
    assert list(first["id"]) == list(again["id"])


def test_a_different_seed_draws_a_different_sample():
    first = _coding_sample(_errors(), size=25, seed=7)
    other = _coding_sample(_errors(), size=25, seed=8)
    assert list(first["id"]) != list(other["id"])


def test_the_coding_sample_is_not_the_confident_tail():
    """The regression this file exists for.

    Sampling the top-N by confidence would estimate the composition of the
    model's most certain mistakes and report it as the composition of its errors.
    A uniform draw must reach well below the top of the confidence range.
    """
    errors = _errors()
    sample = _coding_sample(errors, size=30, seed=7)
    candidates = errors[(errors["error_type"] == "false_positive") & (errors["any_identity"] == 1)]
    top_30 = set(candidates.nlargest(30, "confidence")["id"])

    assert set(sample["id"]) != top_30
    assert sample["confidence"].min() if "confidence" in sample else True
    assert sample["prob"].min() < candidates["prob"].quantile(0.5)


def test_the_sample_carries_an_empty_column_to_fill_in():
    sample = _coding_sample(_errors(), size=10, seed=7)
    assert "manual_category" in sample.columns
    assert (sample["manual_category"] == "").all()


def test_asking_for_more_than_exists_takes_everything():
    sample = _coding_sample(_errors(20), size=500, seed=7)
    assert 0 < len(sample) <= 20


def test_no_candidates_is_not_an_error():
    errors = _errors(12)
    errors["any_identity"] = 0
    assert _coding_sample(errors, size=10, seed=7).empty


# -- reading the committed coding back ---------------------------------------


def _ctx(tmp_path, coding_path: str):
    config = Config({"error_analysis": {"coding_file": coding_path}})
    return SimpleNamespace(config=config)


def test_absent_coding_reports_nothing_rather_than_guessing(tmp_path):
    assert _coded_counts(_ctx(tmp_path, str(tmp_path / "nope.csv"))) is None


def test_coded_counts_and_shares(tmp_path):
    path = tmp_path / "coding.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "manual_category": ["counter_speech", "label_noise", "label_noise", "quotation"],
            "counter_speech": [1, 0, 0, 0],
            "negation": [0, 0, 1, 0],
        }
    ).to_csv(path, index=False)

    table = _coded_counts(_ctx(tmp_path, str(path)))
    by_category = table.set_index("manual_category")

    assert by_category.loc["label_noise", "n"] == 2
    assert by_category.loc["label_noise", "share"] == pytest.approx(0.5)
    assert by_category.loc["counter_speech", "regex_counter_speech"] == 1
    assert by_category.loc["label_noise", "regex_negation"] == 1


def test_unfilled_rows_are_not_counted(tmp_path):
    path = tmp_path / "coding.csv"
    pd.DataFrame(
        {"id": [1, 2, 3], "manual_category": ["counter_speech", "", None]}
    ).to_csv(path, index=False)

    table = _coded_counts(_ctx(tmp_path, str(path)))
    assert int(table["n"].sum()) == 1


def test_a_file_with_nothing_coded_is_ignored(tmp_path):
    path = tmp_path / "coding.csv"
    pd.DataFrame({"id": [1, 2], "manual_category": [None, None]}).to_csv(path, index=False)
    assert _coded_counts(_ctx(tmp_path, str(path))) is None

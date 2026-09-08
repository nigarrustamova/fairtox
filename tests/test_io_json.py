"""JSON artefacts have to be readable by something other than Python.

Every metrics file this project writes is a research artefact: it goes in the
repository, and someone will eventually open it in a browser, in R, or with jq.
``json.dump`` emits the bare token ``NaN`` for a non-finite float -- a Python
extension that every strict parser rejects -- and it does so *silently*, because
Python reads its own extension back without complaint. That is how 91 of the
study's 539 metrics files ended up unparseable outside Python before anyone
looked at them with a strict parser.

``default=`` does not prevent it: json consults ``default`` only for types it
cannot serialise, and it believes it can serialise a NaN.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from fairtox.utils import io


def _reads_strictly(path) -> dict:
    """Parse with the non-standard constants disabled, as other languages do."""
    def refuse(token):
        raise ValueError(f"not JSON: {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def test_a_plain_payload_round_trips(tmp_path):
    path = io.save_json({"a": 1, "b": "two", "c": [1.5, None], "d": True},
                        tmp_path / "x.json")
    assert _reads_strictly(path) == {"a": 1, "b": "two", "c": [1.5, None], "d": True}


def test_nan_becomes_null_not_the_nan_token(tmp_path):
    path = io.save_json({"fpr": float("nan")}, tmp_path / "x.json")

    assert "NaN" not in path.read_text(encoding="utf-8")
    assert _reads_strictly(path) == {"fpr": None}


def test_infinities_become_null_too(tmp_path):
    path = io.save_json({"a": math.inf, "b": -math.inf}, tmp_path / "x.json")
    assert _reads_strictly(path) == {"a": None, "b": None}


def test_nested_and_listed_nans_are_reached(tmp_path):
    payload = {"subgroups": [{"subgroup": "hindu", "fpr": float("nan"), "n": 12},
                             {"subgroup": "male", "fpr": 0.05, "n": 900}],
               "summary": {"worst": {"fpr": float("nan")}}}
    restored = _reads_strictly(io.save_json(payload, tmp_path / "x.json"))

    assert restored["subgroups"][0]["fpr"] is None
    assert restored["subgroups"][1]["fpr"] == 0.05
    assert restored["summary"]["worst"]["fpr"] is None


def test_the_real_shape_that_broke_it(tmp_path):
    """A pandas table with an unmeasurable rate, exactly as the auditor writes it."""
    table = pd.DataFrame({"subgroup": ["male", "hindu"], "fpr": [0.05, None],
                          "n_nontoxic": [900, 0]})
    path = io.save_json({"subgroups": table.to_dict(orient="records")},
                        tmp_path / "audit.json")

    assert _reads_strictly(path)["subgroups"][1]["fpr"] is None


def test_numpy_scalars_still_survive(tmp_path):
    payload = {"i": np.int64(3), "f": np.float64(0.25), "b": np.bool_(True),
               "a": np.array([1.0, 2.0])}
    assert _reads_strictly(io.save_json(payload, tmp_path / "x.json")) == {
        "i": 3, "f": 0.25, "b": True, "a": [1.0, 2.0]}


def test_a_numpy_nan_is_null_as_well(tmp_path):
    path = io.save_json({"f": np.float64("nan")}, tmp_path / "x.json")
    assert _reads_strictly(path) == {"f": None}


def test_load_json_reads_what_save_json_wrote(tmp_path):
    path = io.save_json({"fpr": float("nan"), "n": 5}, tmp_path / "x.json")
    assert io.load_json(path) == {"fpr": None, "n": 5}


def test_every_committed_analysis_json_is_strict_json():
    """The regression this file exists for, checked on the shipped artefacts."""
    from fairtox.config import REPO_ROOT

    analysis = REPO_ROOT / "analysis"
    files = sorted(analysis.glob("*.json"))
    assert files, "no analysis JSON to check; run scripts/summarise_study.py"
    for path in files:
        try:
            _reads_strictly(path)
        except ValueError as exc:  # pragma: no cover - only fires on a regression
            pytest.fail(f"{path.name} is not valid JSON: {exc}")

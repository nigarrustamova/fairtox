"""Shared fixtures.

Everything here is small and hand-built so the assertions can be checked by
reading them. The corpus fixtures deliberately carry the pathology the study is
about -- identity mentions that co-occur with toxicity -- so a test that passes
on them is exercising the code path the real run takes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fairtox.config import Config


@pytest.fixture
def config() -> Config:
    """A minimal in-memory config, independent of the YAML files on disk."""
    return Config(
        {
            "run": {"name": "pytest", "seed": 42, "device": "cpu"},
            "data": {
                "toxicity_threshold": 0.5,
                "identity_threshold": 0.5,
                "annotated_only": True,
                "subsample": None,
                "splits": {"train": 0.75, "val": 0.10, "test": 0.15},
                "identity_columns": ["group_a", "group_b"],
            },
            "evaluation": {
                "classification_threshold": 0.5,
                "min_support": 1,
                "ci_method": "wilson",
                "confidence_level": 0.95,
                "bootstrap_samples": 200,
            },
            "mitigation": {"scheme": "subgroup", "alpha": 4.0, "weights": {}},
            "verdict": {"utility_tolerance": 0.01, "safety_tolerance": 0.02},
        }
    )


@pytest.fixture
def corpus() -> pd.DataFrame:
    """A prepared corpus: 200 rows, both classes, both subgroups, no missing ids."""
    rng = np.random.default_rng(7)
    n = 200
    group_a = (rng.random(n) < 0.30).astype(np.int8)
    group_b = (rng.random(n) < 0.20).astype(np.int8)
    any_identity = ((group_a + group_b) > 0).astype(np.int8)
    # Identity-bearing rows are likelier to be labelled toxic: the correlation
    # the mitigation is meant to break.
    toxic_prob = np.where(any_identity == 1, 0.40, 0.15)
    label = (rng.random(n) < toxic_prob).astype(np.int8)

    return pd.DataFrame(
        {
            "id": np.arange(n),
            "comment_text": [f"comment number {i} about something" for i in range(n)],
            "target": np.where(label == 1, 0.8, 0.2),
            "label": label,
            "group_a": group_a,
            "group_b": group_b,
            "any_identity": any_identity,
        }
    )


@pytest.fixture
def cells() -> pd.DataFrame:
    """Twelve rows covering all four (identity x label) cells, unevenly."""
    rows = [
        # (any_identity, label, how many)
        (0, 0, 4),
        (0, 1, 3),
        (1, 0, 3),
        (1, 1, 2),
    ]
    records = []
    row_id = 0
    for any_identity, label, count in rows:
        for _ in range(count):
            records.append(
                {
                    "id": row_id,
                    "comment_text": f"row {row_id}",
                    "label": label,
                    "any_identity": any_identity,
                    "group_a": any_identity,
                    "group_b": 0,
                }
            )
            row_id += 1
    return pd.DataFrame(records)


@pytest.fixture
def predictions() -> pd.DataFrame:
    """A hand-computable prediction file.

    group_a: 4 non-toxic of which 2 are flagged  -> FPR 0.5
             2 toxic     of which 1 is missed    -> FNR 0.5
    group_b: 4 non-toxic, none flagged           -> FPR 0.0
             2 toxic, none missed                -> FNR 0.0
    """
    rows = [
        # group_a, non-toxic: two flagged
        {"id": 0, "label": 0, "prob": 0.90, "group_a": 1, "group_b": 0},
        {"id": 1, "label": 0, "prob": 0.70, "group_a": 1, "group_b": 0},
        {"id": 2, "label": 0, "prob": 0.10, "group_a": 1, "group_b": 0},
        {"id": 3, "label": 0, "prob": 0.20, "group_a": 1, "group_b": 0},
        # group_a, toxic: one missed
        {"id": 4, "label": 1, "prob": 0.95, "group_a": 1, "group_b": 0},
        {"id": 5, "label": 1, "prob": 0.15, "group_a": 1, "group_b": 0},
        # group_b, non-toxic: none flagged
        {"id": 6, "label": 0, "prob": 0.05, "group_a": 0, "group_b": 1},
        {"id": 7, "label": 0, "prob": 0.11, "group_a": 0, "group_b": 1},
        {"id": 8, "label": 0, "prob": 0.22, "group_a": 0, "group_b": 1},
        {"id": 9, "label": 0, "prob": 0.33, "group_a": 0, "group_b": 1},
        # group_b, toxic: none missed
        {"id": 10, "label": 1, "prob": 0.88, "group_a": 0, "group_b": 1},
        {"id": 11, "label": 1, "prob": 0.77, "group_a": 0, "group_b": 1},
    ]
    frame = pd.DataFrame(rows)
    frame["any_identity"] = 1
    return frame

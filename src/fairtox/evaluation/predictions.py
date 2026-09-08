"""The prediction file: the interface between training and every analysis.

A run writes one CSV per split:

    id, label, prob, any_identity, <one column per identity subgroup>

Every analysis reads *these files*, not a model. So the TF-IDF baseline is
measured by exactly the same auditor code as the transformer, the analysis half
of the project re-runs on a laptop days after the window closed, and re-auditing
at another threshold costs no GPU time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..utils import io
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Column names that are never a demographic subgroup. Anything else in a
# prediction file is one, which keeps the audit from silently missing a subgroup
# because a hard-coded list somewhere fell out of date.
RESERVED_COLUMNS = frozenset({"id", "label", "prob", "any_identity", "split", "comment_text"})


def predictions_path(run_name: str, model_name: str, split: str) -> Path:
    return io.predictions_path(run_name, model_name, split)


def save_predictions(
    run_name: str,
    model_name: str,
    split: str,
    frame: pd.DataFrame,
    probs: np.ndarray,
    identity_columns: list[str],
) -> Path:
    """Write one run's probabilities for one split."""
    if len(frame) != len(probs):
        raise ValueError(
            f"{len(frame)} rows but {len(probs)} probabilities for "
            f"{model_name}/{split} -- the evaluation loader must not shuffle or drop"
        )

    columns = {
        "id": frame["id"].to_numpy(),
        "label": frame["label"].to_numpy().astype(int),
        "prob": np.asarray(probs, dtype=float),
        "any_identity": frame["any_identity"].to_numpy().astype(int),
    }
    for column in identity_columns:
        if column in frame.columns:
            columns[column] = frame[column].to_numpy().astype(int)

    path = predictions_path(run_name, model_name, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns).to_csv(path, index=False)
    logger.info("wrote predictions %s (%s rows)", path.name, f"{len(frame):,}")
    return path


def load_predictions(run_name: str, model_name: str, split: str = "test") -> pd.DataFrame:
    path = predictions_path(run_name, model_name, split)
    if not path.exists():
        raise FileNotFoundError(
            f"no predictions at {path}.\n"
            f"Run the model first:  python scripts/run_all.py --only train_{model_name}"
        )
    return pd.read_csv(path)


def has_predictions(run_name: str, model_name: str, split: str = "test") -> bool:
    return predictions_path(run_name, model_name, split).exists()


def identity_columns_in(frame: pd.DataFrame) -> list[str]:
    """Subgroup columns present in a prediction file, in file order."""
    return [c for c in frame.columns if c not in RESERVED_COLUMNS]

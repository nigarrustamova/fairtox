"""Where artefacts live, and how they are written.

One study writes into one directory tree, keyed by ``run.name``:

    results/<run>/metrics/<stage>.json        every number the paper quotes
    results/<run>/tables/<name>.{md,csv}      tables, human- and machine-readable
    results/<run>/figures/<name>.png          figures
    results/<run>/predictions/<model>_<split>.csv
    results/<run>/run.log
    experiments/<run>/<model>/best.pt         weights (last.pt while training)

Tables are written twice on purpose: the Markdown copy is what a person reads in
a pull request, and the CSV is what the report build reads back, so no number is
ever retyped by hand into the paper.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from .logging_utils import get_logger

logger = get_logger(__name__)

RESULTS_DIR = REPO_ROOT / "results"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


def run_dir(run_name: str) -> Path:
    return RESULTS_DIR / run_name


def ensure_dirs(run_name: str) -> None:
    for sub in ("metrics", "tables", "figures", "predictions"):
        (run_dir(run_name) / sub).mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def metrics_path(run_name: str, stage_name: str) -> Path:
    return run_dir(run_name) / "metrics" / f"{stage_name}.json"


def figure_path(run_name: str, name: str, suffix: str = ".png") -> Path:
    return run_dir(run_name) / "figures" / f"{name}{suffix}"


def table_path(run_name: str, name: str, suffix: str = ".md") -> Path:
    return run_dir(run_name) / "tables" / f"{name}{suffix}"


def predictions_path(run_name: str, model_name: str, split: str) -> Path:
    return run_dir(run_name) / "predictions" / f"{model_name}_{split}.csv"


def checkpoint_dir(run_name: str, model_name: str) -> Path:
    return EXPERIMENTS_DIR / run_name / model_name


def _jsonable(value: Any) -> Any:
    """Coerce numpy and pandas scalars that ``json`` refuses to serialise."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def _finite(payload: Any) -> Any:
    """Replace non-finite floats with ``None``, recursively, before dumping.

    ``json.dump(default=...)`` does NOT solve this. ``default`` is consulted only
    for objects json cannot serialise, and a float NaN is one it CAN: it emits the
    bare token ``NaN``, which is a Python extension and not JSON. Every strict
    parser -- JavaScript, Go, R, jq -- rejects the file. That went unnoticed
    because Python reads its own extension back without complaint.

    ``None`` is also the honest value. These NaNs arrive from pandas, which turns
    the auditor's ``None`` -- "this subgroup has no rows on the relevant side, so
    the rate is undefined" -- into NaN on the way through ``to_dict``. Writing
    ``null`` puts the original meaning back.
    """
    if isinstance(payload, dict):
        return {k: _finite(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_finite(v) for v in payload]
    if isinstance(payload, float) and not math.isfinite(payload):
        return None
    return payload


def save_json(payload: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        # allow_nan=False turns any survivor into an exception instead of an
        # invalid file: a guard, not a formatting choice.
        json.dump(_finite(payload), handle, indent=2, default=_jsonable,
                  ensure_ascii=False, allow_nan=False)
    logger.info("wrote %s", _relative(path))
    return path


def load_json(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_table(frame: pd.DataFrame, path: Path, float_format: str = "%.4f") -> Path:
    """Write a table as Markdown (for humans) and CSV (for the report build)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if frame is None or frame.empty:
        path.write_text("_(empty table)_\n", encoding="utf-8")
        logger.warning("table %s is empty", path.name)
        return path

    csv_path = path.with_suffix(".csv")
    frame.to_csv(csv_path, index=False)

    try:
        markdown = frame.to_markdown(index=False, floatfmt=".4f")
    except ImportError:  # tabulate absent -- degrade rather than lose the table
        markdown = frame.to_string(index=False, float_format=lambda v: float_format % v)
    path.write_text(markdown + "\n", encoding="utf-8")
    logger.info("wrote %s (+ .csv)", _relative(path))
    return path


def _relative(path: Path) -> str:
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(path)

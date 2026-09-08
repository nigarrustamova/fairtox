"""Loading and preparing the Jigsaw Unintended Bias corpus.

Three decisions here shape every downstream number. **Binarisation** at
``data.toxicity_threshold`` is a reported assumption, not a fact --
``threshold_sensitivity`` re-runs the audit at other values. **The annotated
subset**: a row with no identity rating is *unlabelled*, not *identity-free*, and
conflating the two would push identity-bearing comments into the background
weight. **Subsampling stays off**: the reporting floor is counted on the test
split, so shrinking the corpus removes the small subgroups the study is about.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import REPO_ROOT
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

TEXT_FALLBACKS = ("comment_text", "text")
# The competition train.csv calls the toxicity fraction `target`; the
# post-competition all_data.csv release calls the same column `toxicity`.
# Accept either, rather than making the config depend on which file was fetched.
TARGET_FALLBACKS = ("target", "toxicity")
ANNOTATOR_COUNT_COLUMN = "identity_annotator_count"


def raw_csv_path(config: Any) -> Path:
    """Resolve the raw CSV, honouring an absolute ``data.raw_dir``.

    Absolute paths matter for read-only mounts such as Kaggle's /kaggle/input.
    ``~`` is expanded *before* the absolute test: without that, ``~/corpus`` is
    not absolute and would be joined onto the repository root, reporting a
    missing file at a path nobody typed.
    """
    raw_dir = Path(str(config.get("data.raw_dir", "data/raw"))).expanduser()
    if not raw_dir.is_absolute():
        raw_dir = REPO_ROOT / raw_dir

    explicit = config.get("data.raw_file")
    if explicit:
        return raw_dir / str(explicit)
    for candidate in ("train.csv", "all_data.csv", "civil_comments.csv"):
        if (raw_dir / candidate).exists():
            return raw_dir / candidate
    return raw_dir / "train.csv"


def load_raw(config: Any) -> pd.DataFrame:
    """Read the raw corpus from disk, reading only the columns we use."""
    path = raw_csv_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"raw data not found at {path}\n"
            "Fetch it first:   bash scripts/download_data.sh\n"
            "Or work offline:  python scripts/make_synthetic_data.py --rows 3000\n"
            "(see README 'Getting the data' for the licence and source)"
        )

    text_col = str(config.get("data.text_column", "comment_text"))
    target_col = str(config.get("data.target_column", "target"))
    identity_cols = list(config.get("data.identity_columns", []) or [])

    header = pd.read_csv(path, nrows=0).columns.tolist()

    if text_col not in header:
        text_col = next((c for c in TEXT_FALLBACKS if c in header), text_col)
    if text_col not in header:
        raise KeyError(f"no comment-text column in {path.name}; found: {header[:12]}")

    if target_col not in header:
        resolved = next((c for c in TARGET_FALLBACKS if c in header), None)
        if resolved is None:
            raise KeyError(
                f"no toxicity column in {path.name}. Looked for '{target_col}' and "
                f"{list(TARGET_FALLBACKS)}; found: {header[:12]}"
            )
        logger.info("target column '%s' not present; using '%s'", target_col, resolved)
        target_col = resolved

    missing = sorted(set(identity_cols) - set(header))
    if missing:
        logger.warning(
            "%d identity column(s) absent from this release and dropped from the audit: %s",
            len(missing), ", ".join(missing),
        )

    wanted = [
        c for c in ["id", target_col, text_col, ANNOTATOR_COUNT_COLUMN, *identity_cols]
        if c in header
    ]

    logger.info("reading %s", path.name)
    frame = pd.read_csv(path, usecols=wanted, low_memory=False)
    frame = frame.rename(columns={text_col: "comment_text", target_col: "target"})
    logger.info("read %s rows x %d columns", f"{len(frame):,}", frame.shape[1])
    return frame


def prepare(frame: pd.DataFrame, config: Any) -> pd.DataFrame:
    """Binarise labels and identity flags, filter, and optionally subsample."""
    tau = float(config.get("data.toxicity_threshold", 0.5))
    tau_g = float(config.get("data.identity_threshold", 0.5))
    identity_cols = [
        c for c in (config.get("data.identity_columns", []) or []) if c in frame.columns
    ]

    frame = frame.dropna(subset=["comment_text", "target"]).copy()
    frame["comment_text"] = frame["comment_text"].astype(str).str.strip()
    frame = frame[frame["comment_text"].str.len() > 0]

    if bool(config.get("data.annotated_only", True)):
        before = len(frame)
        if ANNOTATOR_COUNT_COLUMN in frame.columns:
            frame = frame[frame[ANNOTATOR_COUNT_COLUMN].fillna(0) > 0]
        elif identity_cols:
            # Older mirrors omit the counter; fall back to "any rating present".
            frame = frame[frame[identity_cols].notna().any(axis=1)]
        else:
            logger.warning("annotated_only requested but no way to tell -- keeping every row")
        logger.info(
            "identity-annotated subset: %s of %s rows (%.1f%%)",
            f"{len(frame):,}", f"{before:,}", 100 * len(frame) / max(before, 1),
        )

    frame["label"] = (frame["target"].astype(float) >= tau).astype(np.int8)

    for column in identity_cols:
        frame[column] = (frame[column].fillna(0.0).astype(float) >= tau_g).astype(np.int8)
    frame["any_identity"] = (
        frame[identity_cols].sum(axis=1).gt(0).astype(np.int8)
        if identity_cols
        else np.zeros(len(frame), dtype=np.int8)
    )

    # Predictions are joined back to text by id for the error analysis, and the
    # split map is keyed on it. A mirror that ships without one gets a stable
    # surrogate here rather than an obscure KeyError six stages later.
    if "id" not in frame.columns:
        logger.warning("corpus has no 'id' column; assigning positional ids")
        frame = frame.reset_index(drop=True)
        frame.insert(0, "id", np.arange(len(frame), dtype=np.int64))
    elif frame["id"].duplicated().any():
        raise ValueError(
            "the 'id' column contains duplicates, so predictions could not be joined "
            "back to comment text unambiguously"
        )

    frame = frame.reset_index(drop=True)
    subsample = config.get("data.subsample")
    if subsample and int(subsample) < len(frame):
        frame = _stratified_subsample(frame, int(subsample), int(config.get("run.seed", 42)))

    frame = frame.reset_index(drop=True)
    logger.info(
        "prepared %s rows | toxic %.2f%% | any identity %.2f%%",
        f"{len(frame):,}", 100 * frame["label"].mean(), 100 * frame["any_identity"].mean(),
    )
    return frame


def _stratified_subsample(frame: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Down-sample while preserving the label x identity composition.

    Stratifying on the interaction, not the label alone, is what keeps a small
    run from containing no identity-bearing non-toxic rows -- which would leave
    every subgroup false-positive rate undefined.
    """
    rng = np.random.default_rng(seed)
    strata = (frame["label"].astype(str) + "_" + frame["any_identity"].astype(str)).to_numpy()

    keep: list[np.ndarray] = []
    for value in np.unique(strata):
        positions = np.flatnonzero(strata == value)
        take = min(len(positions), max(1, round(n * len(positions) / len(frame))))
        keep.append(rng.choice(positions, size=take, replace=False))

    selected = np.sort(np.concatenate(keep))
    out = frame.iloc[selected].reset_index(drop=True)
    logger.warning(
        "SUBSAMPLED to %s rows (stratified by label x identity). Subgroup support on the "
        "test split shrinks with it -- read the eda warning before trusting the audit.",
        f"{len(out):,}",
    )
    return out


def load_prepared(config: Any) -> pd.DataFrame:
    return prepare(load_raw(config), config)


def active_identity_columns(frame: pd.DataFrame, config: Any) -> list[str]:
    """Configured identity columns that actually survived the load."""
    return [c for c in (config.get("data.identity_columns", []) or []) if c in frame.columns]

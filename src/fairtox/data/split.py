"""Leakage-free train/validation/test splitting.

The split is made once, up front, before any statistic is fitted, and written to
disk as an id -> split map. Stratification is on ``label x any_identity`` rather
than the label alone: subgroup metrics are undefined if a split happens to
contain no identity-bearing non-toxic rows.

Every split carries a hash of its id -> split assignment, and ``compare`` refuses
a pair whose hashes differ. Without it a stray ``--set data.subsample=...`` on
one arm would produce two different test sets and a comparison that means
nothing, while the output looked entirely normal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import REPO_ROOT
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def split_map_path(config: Any) -> Path:
    processed = Path(str(config.get("data.processed_dir", "data/processed")))
    if not processed.is_absolute():
        processed = REPO_ROOT / processed
    processed.mkdir(parents=True, exist_ok=True)
    return processed / f"splits_{config.run_name}.csv"


def fingerprint(splits: dict[str, pd.DataFrame]) -> str:
    """Stable hash of the id -> split assignment.

    Order-independent by construction (ids are sorted within each split), so two
    runs that partitioned the same rows the same way agree even if the rows
    arrived in a different order.
    """
    digest = hashlib.sha256()
    for name in SPLIT_NAMES:
        part = splits.get(name)
        if part is None:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(b"|")
        digest.update(",".join(map(str, sorted(part["id"].tolist()))).encode("utf-8"))
        digest.update(b"||")
    return digest.hexdigest()[:16]


def make_splits(frame: pd.DataFrame, config: Any) -> dict[str, pd.DataFrame]:
    """Partition ``frame`` into train/val/test, stratified and seeded."""
    fractions = config.get("data.splits", {}) or {}
    train_frac = float(fractions.get("train", 0.75))
    val_frac = float(fractions.get("val", 0.10))
    test_frac = float(fractions.get("test", 0.15))
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total:.4f}")

    seed = int(config.get("run.seed", 42))
    strata = frame["label"].astype(str) + "_" + frame["any_identity"].astype(str)
    # Stratification needs at least two rows per cell; fall back rather than
    # crash when a tiny smoke subset has a rare cell.
    stratify = strata if strata.value_counts().min() >= 2 else None
    if stratify is None:
        logger.warning("a stratum has fewer than 2 rows; falling back to an unstratified split")

    train_df, holdout = train_test_split(
        frame, train_size=train_frac, random_state=seed, shuffle=True, stratify=stratify
    )

    holdout_strata = None
    if stratify is not None:
        holdout_strata = holdout["label"].astype(str) + "_" + holdout["any_identity"].astype(str)
        if holdout_strata.value_counts().min() < 2:
            holdout_strata = None

    val_share = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        holdout, train_size=val_share, random_state=seed, shuffle=True, stratify=holdout_strata
    )

    splits = {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }
    for name, part in splits.items():
        logger.info(
            "%-5s %9s rows | toxic %5.2f%% | identity %5.2f%%",
            name, f"{len(part):,}", 100 * part["label"].mean(), 100 * part["any_identity"].mean(),
        )

    _persist(splits, config)
    logger.info("split fingerprint %s", fingerprint(splits))
    return splits


def _persist(splits: dict[str, pd.DataFrame], config: Any) -> Path:
    """Record which row landed in which split, for auditability."""
    records = [
        pd.DataFrame({"id": part["id"].to_numpy(), "split": name})
        for name, part in splits.items()
    ]
    path = split_map_path(config)
    pd.concat(records, ignore_index=True).to_csv(path, index=False)
    logger.info("split map written to %s", path.name)
    return path


def split_sizes(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {name: len(part) for name, part in splits.items()}


def assert_same_split(first: str | None, second: str | None, what: str) -> None:
    """Fail loudly when two runs being compared did not share a partition."""
    if first is None or second is None:
        logger.warning("%s: a split fingerprint is missing; cannot verify the partition", what)
        return
    if first != second:
        raise ValueError(
            f"{what}: the two runs used DIFFERENT splits ({first} vs {second}).\n"
            "The comparison assumes only the loss weighting changed, so this result "
            "is not interpretable. Re-run both arms from the same config and seed."
        )

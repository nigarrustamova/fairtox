"""Helpers shared by stages.

The corpus and the split are built once per driver invocation and handed on
through the run context. Every stage that needs them calls the same accessor, so
a stage run on its own (``--only audit_baseline --no-deps``) rebuilds exactly the
partition the training run used -- same config, same seed, same fingerprint --
rather than quietly producing a second, different one.
"""

from __future__ import annotations

import pandas as pd

from ..data import load as data_load
from ..data import split as data_split
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def get_corpus(ctx) -> tuple[pd.DataFrame, list[str]]:
    """The prepared corpus and its surviving identity columns."""
    frame = ctx.take("frame")
    identity_columns = ctx.take("identity_columns")
    if frame is None:
        frame = data_load.load_prepared(ctx.config)
        identity_columns = data_load.active_identity_columns(frame, ctx.config)
        ctx.put("frame", frame)
        ctx.put("identity_columns", identity_columns)
    return frame, list(identity_columns or [])


def get_splits(ctx) -> dict[str, pd.DataFrame]:
    """The train/val/test partition, built once and reused."""
    splits = ctx.take("splits")
    if splits is None:
        frame, _ = get_corpus(ctx)
        splits = data_split.make_splits(frame, ctx.config)
        ctx.put("splits", splits)
        ctx.put("split_fingerprint", data_split.fingerprint(splits))
    return splits


def get_split_fingerprint(ctx) -> str:
    fingerprint = ctx.take("split_fingerprint")
    if fingerprint is None:
        fingerprint = data_split.fingerprint(get_splits(ctx))
        ctx.put("split_fingerprint", fingerprint)
    return fingerprint


def text_lookup(ctx) -> pd.Series:
    """Map row id -> comment text, so errors can be read rather than counted."""
    frame, _ = get_corpus(ctx)
    if "id" not in frame.columns:
        raise KeyError("the corpus has no 'id' column; cannot join text to predictions")
    return frame.set_index("id")["comment_text"]

"""Tokenisation and the PyTorch dataset.

Text is tokenised once, up front, into int32 tensors: at 448k rows x 128 tokens
that is ~460 MB, it removes the tokenizer from the hot loop, and it lets the
loaders run without pickling a tokenizer into worker processes.

Each row carries its per-sample loss weight, so the mitigated arm stays on the
same code path as the baseline -- one column differs, not a branch in the loop.
"""

from __future__ import annotations

import os
from typing import Any

# The corpus is tokenised once, up front, before any DataLoader worker is forked.
# The fast tokenizer notices the fork and prints a four-line warning about
# disabling its own parallelism -- which is harmless here, because nothing is
# tokenised inside the workers. Setting this makes the log readable on training
# day; it changes no behaviour we rely on.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def build_tokenizer(config: Any):
    """Load the backbone's tokenizer.

    Some checkpoints publish weights without tokenizer files, and the resulting
    error names three conversion strategies without saying that the repository
    simply has no tokenizer. Re-raise with the backbone named, so the fix -- pick
    a different one -- is obvious from the message.
    """
    from transformers import AutoTokenizer

    backbone = str(config.get("model.backbone", "distilbert-base-uncased"))
    try:
        return AutoTokenizer.from_pretrained(backbone, use_fast=True)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"could not load a tokenizer for model.backbone='{backbone}'. "
            "Check that the checkpoint publishes tokenizer files (vocab.txt or "
            "tokenizer.json), not just weights. "
            f"Underlying error: {exc}"
        ) from exc


class ToxicityDataset(Dataset):
    """Pre-tokenised comments with labels and per-sample loss weights."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: Any,
        max_length: int = 128,
        weights: np.ndarray | None = None,
        groups: np.ndarray | None = None,
    ) -> None:
        texts = frame["comment_text"].astype(str).tolist()
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=int(max_length),
            return_tensors="np",
        )
        # int32 halves the memory of int64 and covers every vocabulary in use
        # (DistilBERT is ~30k, RoBERTa ~50k), so nothing is lost by narrowing.
        self.input_ids = torch.from_numpy(encoded["input_ids"].astype(np.int32))
        self.attention_mask = torch.from_numpy(encoded["attention_mask"].astype(np.int8))
        self.labels = torch.from_numpy(frame["label"].to_numpy(dtype=np.float32))

        if weights is None:
            weights = np.ones(len(frame), dtype=np.float32)
        if len(weights) != len(frame):
            raise ValueError(
                f"weight vector has {len(weights)} entries for {len(frame)} rows"
            )
        self.weights = torch.from_numpy(np.asarray(weights, dtype=np.float32))

        # A cell index per row, for objectives that reweight cells during
        # training rather than before it. Zeros when nothing asked for one, so
        # the tensor is always present and the training loop needs no branch.
        if groups is None:
            groups = np.zeros(len(frame), dtype=np.int64)
        if len(groups) != len(frame):
            raise ValueError(f"group vector has {len(groups)} entries for {len(frame)} rows")
        self.groups = torch.from_numpy(np.asarray(groups, dtype=np.int64))
        self.ids = frame["id"].to_numpy()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            # Cast back to int64 here: embedding lookups require long indices,
            # but only one batch is ever in that form at a time.
            "input_ids": self.input_ids[index].long(),
            "attention_mask": self.attention_mask[index].long(),
            "label": self.labels[index],
            "weight": self.weights[index],
            "group": self.groups[index],
        }


def build_loader(
    dataset: ToxicityDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Wrap a dataset in a loader.

    Evaluation loaders are built with ``shuffle=False`` and never drop a batch:
    the audit rejoins probabilities to identity flags by position, so any
    reordering or silent truncation there would mislabel every subgroup.
    """
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
        persistent_workers=bool(num_workers),
    )

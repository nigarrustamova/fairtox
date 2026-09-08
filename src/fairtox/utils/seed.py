"""Deterministic seeding.

The brief makes reproducibility an explicit requirement and a non-reproducible
headline result an automatic deduction, so every source of randomness the
project touches is pinned from one place: Python, NumPy and Torch (CPU and CUDA).
"""

from __future__ import annotations

import os
import random

import numpy as np

from .logging_utils import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42, deterministic: bool = False) -> int:
    """Seed every RNG in play. Returns the seed, for logging."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # the analysis-only stages do not need torch
        return seed

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Bit-exact at a real throughput cost. Off by default: seeding gives
        # run-to-run stability, and the honest answer to residual GPU
        # non-determinism is to report across several seeds, not to pretend a
        # single run is exact.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, RuntimeError) as exc:
            logger.warning("could not enable deterministic algorithms: %s", exc)
    return seed

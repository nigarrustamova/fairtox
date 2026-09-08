"""Logging that goes to the console and to a file at the same time.

The booked window can end mid-run and the terminal scrollback goes with it, so
everything worth reading later is written to ``results/<run>/run.log`` as it
happens rather than being reconstructed afterwards.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Attach console and file handlers to the root logger, exactly once."""
    global _CONFIGURED

    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if not _CONFIGURED:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        root.addHandler(console)
        # Transformers is chatty about weights it did not use; the classification
        # head is new by construction, so that warning is noise here. The hub
        # clients log a line per HTTP request, which buries the training log
        # under download chatter the first time a machine fetches the backbone.
        for noisy in ("transformers", "httpx", "huggingface_hub", "filelock", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.ERROR)
        _CONFIGURED = True

    if log_file is not None:
        existing = {
            Path(h.baseFilename).resolve()
            for h in root.handlers
            if isinstance(h, logging.FileHandler)
        }
        target = Path(log_file).resolve()
        if target not in existing:
            target.parent.mkdir(parents=True, exist_ok=True)
            handler = logging.FileHandler(target, encoding="utf-8")
            handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
            root.addHandler(handler)

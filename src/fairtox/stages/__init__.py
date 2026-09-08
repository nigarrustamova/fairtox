"""Importing this package registers every pipeline stage.

Adding an experiment means dropping a module in here, decorating its entry point
with ``@stage(...)`` and listing it below -- nothing else in the pipeline
changes. That is what keeps the optional analyses genuinely optional and lets
scope be cut with a flag instead of an edit.

Import order is irrelevant to the driver, which resolves dependencies by name,
but is kept alphabetical here so merge conflicts in this file resolve by keeping
both lines.
"""

from . import (
    _planned,
    attributions,
    audit,
    eda,
    error_analysis,
    tfidf_baseline,
    train,
)

__all__ = [
    "_planned",
    "attributions",
    "audit",
    "eda",
    "error_analysis",
    "tfidf_baseline",
    "train",
]

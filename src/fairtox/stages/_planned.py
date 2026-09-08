"""Stages that are designed but deliberately not built.

Both were stretch goals, and both were ultimately answered ELSEWHERE rather than
here: the second backbone as a config file (`a100_drob.yaml`), the per-group
thresholds as a stage of their own (`postproc_thresholds`). They stay registered
because ``run_all.py --list`` should show the full design space including the
routes not taken, and because each one's message now records where the question
actually got answered -- which is more useful than deleting the entry and leaving
a reader to wonder whether it was ever considered.

Each raises on execution rather than silently doing nothing, so a half-built
pipeline can never be mistaken for a finished result. They sit at EXTRA, which
no default invocation reaches -- ``--tier must`` and ``--tier optional`` both
exclude them.
"""

from __future__ import annotations

from ..registry import Tier, stage


def _not_yet(name: str, why: str):
    def run(ctx):
        raise NotImplementedError(
            f"stage '{name}' is a stretch goal and is not implemented. {why} "
            "Run with --tier optional or lower to exclude it."
        )

    return run


stage(
    "group_threshold_calibration",
    tier=Tier.EXTRA,
    summary="Post-hoc per-group thresholds as a cheap alternative to retraining.",
    depends_on=("compare",),
)(
    _not_yet(
        "group_threshold_calibration",
        "It was superseded rather than dropped: per-group thresholds ARE measured, by "
        "the `postproc_thresholds` stage, which fits them on validation and scores them "
        "on test across every run. It stays here at EXTRA because the name describes a "
        "slightly different thing -- calibration per group rather than a threshold per "
        "group -- and because the policy question it raises, deploying a different "
        "decision rule per demographic group, is answered in the write-up rather than "
        "in code.",
    )
)

stage(
    "roberta_backbone",
    tier=Tier.EXTRA,
    summary="Superseded: the second backbone is a config file, not a stage.",
    depends_on=("compare",),
)(
    _not_yet(
        "roberta_backbone",
        "It was superseded rather than dropped: the headline comparison HAS been "
        "repeated on a second backbone, by running configs/a100_drob.yaml through the "
        "same stages. Doing it as a config rather than as a stage is the point -- a "
        "stage would be a second code path, and a second code path is a second thing "
        "that can quietly differ between the backbones being compared.",
    )
)

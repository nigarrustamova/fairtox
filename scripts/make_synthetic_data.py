#!/usr/bin/env python
"""Generate a synthetic corpus in the Jigsaw schema.

Purpose: verify the pipeline end to end without the download, and give everyone
on the team a way to run the whole MUST tier on a laptop in minutes.

The generator deliberately plants the phenomenon the project studies -- identity
terms co-occur with toxicity more often than chance -- so the audit has something
real to find and the mitigation has something to fix. Subgroup support is
deliberately uneven, so the reporting floor and the confidence intervals are
exercised rather than assumed.

**This is not data.** Results from it are a pipeline check, never a finding.
Every row carries a `synthetic_marker` column so a generated file cannot be
mistaken for the real corpus, and download_data.sh refuses to leave one in place.

    python scripts/make_synthetic_data.py --rows 3000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

IDENTITY_COLUMNS = [
    "male", "female", "transgender", "homosexual_gay_or_lesbian", "christian",
    "jewish", "muslim", "hindu", "buddhist", "atheist", "black", "white",
    "asian", "latino", "physical_disability",
    "intellectual_or_learning_disability", "psychiatric_or_mental_illness",
]

# Uneven support, so some groups land below the reporting floor: the realistic
# and awkward case the auditor has to handle without pretending otherwise.
GROUP_WEIGHTS = np.array([14, 14, 2, 5, 12, 4, 6, 2, 1, 4, 8, 12, 5, 4, 2, 1, 3], dtype=float)

BENIGN = [
    "i really appreciate this perspective on the issue",
    "thanks for taking the time to write this up",
    "the article makes a fair point about the policy",
    "i disagree with the conclusion but it is well argued",
    "does anyone have a source for this claim",
    "we should be talking about this more often",
    "this was a thoughtful and measured response",
    "good reporting, i learned something today",
    "my community has been discussing this for years",
    "there is a lot of nuance being lost in this debate",
]

TOXIC = [
    "you are completely worthless and should shut up",
    "this is absolute garbage written by an idiot",
    "nobody cares what a moron like you thinks",
    "get lost you pathetic waste of space",
    "what a stupid and disgusting thing to say",
]

# Benign sentences that mention hostility. These are the hard negatives the whole
# study is about: counter-speech that a shortcut-learning model flags.
COUNTER_SPEECH = [
    "we must stand against hate directed at this community",
    "attacking people for who they are is never acceptable",
    "i am reporting the abusive comments in this thread",
    "calling out bigotry is not the same as being toxic",
    "no one should have to read that kind of harassment",
]


def build(n_rows: int, seed: int, label_noise: float = 0.18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probabilities = GROUP_WEIGHTS / GROUP_WEIGHTS.sum()
    rows = []

    for row_id in range(n_rows):
        mentions_identity = rng.random() < 0.45
        group = (
            IDENTITY_COLUMNS[rng.choice(len(IDENTITY_COLUMNS), p=probabilities)]
            if mentions_identity
            else None
        )

        # The planted shortcut: identity mentions are likelier to be toxic.
        is_toxic = rng.random() < (0.34 if mentions_identity else 0.13)

        if is_toxic:
            text = str(rng.choice(TOXIC))
        elif mentions_identity and rng.random() < 0.35:
            text = str(rng.choice(COUNTER_SPEECH))
        else:
            text = str(rng.choice(BENIGN))

        # Annotator disagreement, which the real corpus has in abundance. Without
        # it the task is linearly separable, every model scores 1.0, and the
        # audit has nothing to measure -- so the smoke test would pass while
        # proving nothing at all about the fairness code.
        if rng.random() < label_noise:
            is_toxic = not is_toxic

        if group:
            surface = group.replace("_", " ")
            text = (
                f"as a {surface} person, {text}"
                if rng.random() < 0.5
                else f"{text} for the {surface} community"
            )

        row = {
            "id": row_id,
            "comment_text": text,
            # Fractional target, like the real release, so binarisation is exercised.
            "target": float(rng.uniform(0.55, 1.0) if is_toxic else rng.uniform(0.0, 0.45)),
            "identity_annotator_count": int(rng.integers(4, 20)),
            "toxicity_annotator_count": int(rng.integers(10, 60)),
            "synthetic_marker": 1,
        }
        for column in IDENTITY_COLUMNS:
            row[column] = 0.0
        if group:
            row[group] = float(rng.uniform(0.6, 1.0))
        rows.append(row)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--label-noise", type=float, default=0.18,
        help="fraction of labels flipped, standing in for annotator disagreement",
    )
    parser.add_argument("--out", type=Path, default=Path("data/raw/train.csv"))
    args = parser.parse_args(argv)

    frame = build(args.rows, args.seed, args.label_noise)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)

    toxic_rate = (frame["target"] >= 0.5).mean()
    identity_rate = (frame[IDENTITY_COLUMNS].max(axis=1) >= 0.5).mean()
    print(f"wrote {len(frame):,} SYNTHETIC rows to {args.out}")
    print(f"  toxic: {toxic_rate:.1%} | mentions identity: {identity_rate:.1%}")
    print("  NOTE: synthetic data. Results are a pipeline check, not a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

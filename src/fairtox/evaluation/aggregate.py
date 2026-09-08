"""Cross-family aggregation: the statistics the paper's headline numbers are.

``seed_robustness`` aggregates the seeds of ONE run family and deliberately
reports no p-value, because three or four seeds cannot support one. This module
is the other half: it aggregates ACROSS families, and at eight seeds a test is
honest, so it computes one.

The unit of analysis is always the **per-seed paired difference**. Within a seed
the two arms share the partition, the initialisation and the recipe, so the
difference isolates the intervention; across seeds the differences are
independent replications. Nothing here ever pools raw metrics across seeds --
that would mix between-split variation into an estimate of a within-split effect.

Two tests are reported side by side because they disagree in an informative way.
The paired t interval is the more powerful when the effect is real and one or two
seeds go the other way; the sign test assumes almost nothing and is the one to
quote when they conflict. A family where the interval excludes zero and the sign
test does not is *suggestive*, and the write-up has to say so rather than pick
the friendlier number.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

# Metrics read off `compare.json`, as (axis, key) pairs.
AXES = {
    "fpr_gap": ("parity", "fpr_gap"),
    "macro_fpr": ("parity", "macro_fpr"),
    "eod_fpr": ("parity", "eod_fpr"),
    "macro_f1": ("utility", "macro_f1"),
    "macro_fnr": ("safety", "macro_fnr"),
}


def paired_summary(values: list[float], confidence: float = 0.95) -> dict[str, Any]:
    """Mean, spread, Student-t interval and an exact sign test for one family.

    ``sign_p`` is a two-sided binomial test on how many differences are negative,
    with exact zeros dropped as the sign test requires. It is reported for every
    family, including the three-seed ones where it can never reach significance:
    printing ``0.25`` next to a three-seed result is a more honest reminder of
    what three seeds buy than omitting the column.
    """
    from scipy import stats

    clean = [float(v) for v in values if v is not None and np.isfinite(v)]
    n = len(clean)
    if n == 0:
        return {"n": 0}

    array = np.asarray(clean, dtype=float)
    mean = float(array.mean())
    sd = float(array.std(ddof=1)) if n > 1 else 0.0

    if n > 1 and sd > 0:
        half = float(stats.t.ppf(0.5 + confidence / 2.0, n - 1) * sd / np.sqrt(n))
    else:
        half = 0.0

    negative = int((array < 0).sum())
    positive = int((array > 0).sum())
    nonzero = negative + positive
    sign_p = (
        float(stats.binomtest(negative, nonzero, 0.5).pvalue) if nonzero else None
    )

    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half,
        "ci_high": mean + half,
        "n_negative": negative,
        "n_positive": positive,
        "sign_p": sign_p,
        "excludes_zero": bool(n > 1 and sd > 0 and (mean - half) * (mean + half) > 0),
        "values": clean,
    }


def exchange_rate(gap_deltas: list[float], fnr_deltas: list[float]) -> float | None:
    """FNR points paid per point of FPR gap closed -- the study's "safety tax".

    Computed from the two MEANS, not as the mean of per-seed ratios: a seed whose
    gap barely moved would otherwise contribute an enormous ratio and dominate an
    average that is supposed to describe the family.

    Returned as None -- not as a number -- in three cases, because in each of them
    a "rate" would be an artefact rather than a measurement:

    * the family did not close the gap (a negative tax is not a tax);
    * it closed the gap at no safety cost (there is nothing to divide);
    * its gap change does not exclude zero, so the denominator is not
      distinguishable from zero and the ratio is whatever the noise makes it.

    That last guard is the one that matters here. The held-out family closes the
    gap by 0.0065 and pays 0.083, which arithmetically is a tax of 12.7 -- a number
    that describes a near-zero denominator and nothing else.
    """
    gap_stats = paired_summary(gap_deltas)
    fnr = paired_summary(fnr_deltas).get("mean")
    gap = gap_stats.get("mean")
    if not gap or fnr is None or gap >= 0 or fnr <= 0:
        return None
    if not gap_stats.get("excludes_zero"):
        return None
    return float(-fnr / gap)


def load_study(source: str | Path) -> dict[str, dict[str, Any]]:
    """Every run's metrics, from a ``results/`` tree or an archived digest.

    The digest exists because the raw tree is ~2,000 files and the analysis has
    to stay runnable long after the machine that produced them is gone. Both
    paths return the same shape, so nothing downstream knows which was used.
    """
    source = Path(source)
    if source.is_file():
        opener = gzip.open if source.suffix == ".gz" else open
        with opener(source, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        logger.info("loaded %d run(s) from digest %s", len(payload), source.name)
        return payload

    if not source.is_dir():
        raise FileNotFoundError(
            f"no study data at {source}. Pass a results/ directory or a digest "
            "written by scripts/summarise_study.py --write-digest."
        )

    names = (
        "compare", "audit_baseline", "audit_mitigated", "model_baseline",
        "model_mitigated", "postproc_thresholds", "leakage_check",
        "mechanism_analysis", "jigsaw_bias_metrics", "ablate", "eda",
    )
    keys = {
        "audit_baseline": ("audit", "baseline"), "audit_mitigated": ("audit", "mitigated"),
        "model_baseline": ("model", "baseline"), "model_mitigated": ("model", "mitigated"),
        "postproc_thresholds": ("postproc",), "leakage_check": ("leakage",),
        "mechanism_analysis": ("mechanism",), "jigsaw_bias_metrics": ("jigsaw",),
    }

    study: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        if run_dir.name.startswith("smoke"):
            continue
        entry: dict[str, Any] = {}
        for name in names:
            path = run_dir / "metrics" / f"{name}.json"
            if not path.exists():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                logger.warning("could not read %s; skipped", path)
                continue
            target = keys.get(name, (name,))
            if len(target) == 2:
                entry.setdefault(target[0], {})[target[1]] = (
                    payload.get("subgroups", payload) if name.startswith("audit") else payload
                )
            else:
                entry[target[0]] = payload
        if entry:
            study[run_dir.name] = entry
    logger.info("loaded %d run(s) from %s", len(study), source)
    return study


def family_deltas(study: dict[str, Any], runs: list[str], metric: str) -> list[float]:
    """One paired difference per run, for the named metric."""
    if metric not in AXES:
        raise KeyError(f"unknown metric '{metric}' (expected one of {', '.join(AXES)})")
    axis, key = AXES[metric]
    out = []
    for run in runs:
        compare = (study.get(run) or {}).get("compare")
        if not compare:
            logger.warning("run '%s' has no compare.json; excluded from the aggregate", run)
            continue
        value = compare["axes"][axis][key]["delta"]
        if value is not None:
            out.append(float(value))
    return out


def subgroup_deltas(study: dict[str, Any], run: str) -> dict[str, float]:
    """Per-subgroup FPR change for one run, over the subgroups entitled to it.

    A subgroup enters only when BOTH arms cleared the support floor. Letting one
    arm in alone would compare a measured rate against an unmeasurable one.
    """
    audit = (study.get(run) or {}).get("audit") or {}
    baseline = {r["subgroup"]: r for r in audit.get("baseline", [])}
    mitigated = {r["subgroup"]: r for r in audit.get("mitigated", [])}
    out = {}
    for name, before in baseline.items():
        after = mitigated.get(name)
        if not after:
            continue
        if not (before.get("reportable_fpr") and after.get("reportable_fpr")):
            continue
        if before.get("fpr") is None or after.get("fpr") is None:
            continue
        out[name] = float(after["fpr"]) - float(before["fpr"])
    return out

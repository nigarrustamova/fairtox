#!/usr/bin/env python
"""Re-render every figure from the metrics already on disk. No GPU, no model.

    python scripts/rebuild_figures.py                  # every run under results/
    python scripts/rebuild_figures.py --runs main drob
    python scripts/rebuild_figures.py --dry-run

Figures are the one artefact that gets stale silently. A plotting fix, a change
of column width for the paper, a relabelled axis -- none of them touch a number,
so nothing in the pipeline notices that the PNGs on disk were drawn by older
code. Re-running the stages that drew them is not an option either: those stages
need the prediction files, and by then the booked window has closed.

So this reads what the stages already wrote -- `audit_*.json` for the subgroup
rates, `model_*.json` for the confusion counts, `tab03_ablation.csv` for the
sweep, `mechanism_analysis.json` for the fitted lines -- and redraws from that.
Every figure in the repository can therefore be regenerated at any time, from
committed inputs, in seconds.

It draws only what a run actually has. A run without an ablation gets no
trade-off curve rather than an empty one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fairtox.utils import plotting  # noqa: E402
from fairtox.utils.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)


def _load(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        logger.warning("could not parse %s; skipped", path)
        return None


def rebuild_run(run_dir: Path, dry_run: bool = False) -> list[str]:
    """Redraw every figure this run has the inputs for. Returns their names."""
    metrics, figures = run_dir / "metrics", run_dir / "figures"
    audits = {m: _load(metrics / f"audit_{m}.json") for m in ("baseline", "mitigated")}
    models = {m: _load(metrics / f"model_{m}.json") for m in ("baseline", "mitigated")}

    if not audits["baseline"]:
        logger.info("%s has no baseline audit; nothing to draw", run_dir.name)
        return []

    tables = {
        m: pd.DataFrame(payload["subgroups"])
        for m, payload in audits.items() if payload
    }
    # The floor is read back from the audit that used it, never assumed: a run
    # made under a different `evaluation.min_support` must not be redrawn with
    # a line at 100 that its own numbers never respected.
    min_support = int(audits["baseline"]["summary"].get("min_support", 100))

    drawn = []
    plan = [
        ("fig02_subgroup_support", lambda p: plotting.fig_subgroup_support(
            tables["baseline"], p, min_support)),
        ("fig05_fpr_by_subgroup", lambda p: plotting.fig_fpr_by_subgroup(
            tables["baseline"], tables.get("mitigated"), p)),
        ("fig06_fnr_by_subgroup", lambda p: plotting.fig_fnr_by_subgroup(
            tables["baseline"], tables.get("mitigated"), p)),
    ]
    for arm, title in (("baseline", "Baseline"), ("mitigated", "Mitigated")):
        payload = models[arm]
        if payload and payload.get("metrics", {}).get("test"):
            name = "fig03_confusion_baseline" if arm == "baseline" else "fig04_confusion_mitigated"
            plan.append((name, lambda p, d=payload["metrics"]["test"], t=title:
                         plotting.fig_confusion(d, p, t)))

    sweep_path = run_dir / "tables" / "tab03_ablation.csv"
    if sweep_path.exists():
        sweep = pd.read_csv(sweep_path)
        subgroup_arm = sweep[sweep["scheme"] == "subgroup"].sort_values("alpha")
        if len(subgroup_arm) > 1:
            plan.append(("fig09_alpha_tradeoff",
                         lambda p, s=subgroup_arm: plotting.fig_alpha_tradeoff(s, p)))

    mechanism = _load(metrics / "mechanism_analysis.json")
    if mechanism:
        fits = mechanism.get("toxic_rate_fit") or {}
        if fits.get("baseline") and fits.get("mitigated") and audits["mitigated"]:
            plan.append(("fig10_mechanism", lambda p, f=fits: plotting.fig_mechanism(
                audits["baseline"]["subgroups"], audits["mitigated"]["subgroups"],
                f["baseline"], f["mitigated"], p)))

    for name, draw in plan:
        if dry_run:
            drawn.append(name)
            continue
        result = draw(figures / f"{name}.png")
        if Path(result).exists():
            drawn.append(name)
    return drawn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(REPO_ROOT / "results"))
    parser.add_argument("--runs", nargs="*", help="run names; default is every run found")
    parser.add_argument("--dry-run", action="store_true", help="list what would be drawn")
    args = parser.parse_args(argv)

    setup_logging("INFO")
    root = Path(args.results)
    if not root.is_dir():
        raise SystemExit(f"no results directory at {root}")

    names = args.runs or sorted(p.name for p in root.iterdir() if p.is_dir())
    total, per_run = 0, {}
    for name in names:
        drawn = rebuild_run(root / name, args.dry_run)
        per_run[name] = len(drawn)
        total += len(drawn)

    logger.info("=" * 74)
    verb = "would redraw" if args.dry_run else "redrew"
    logger.info("%s %d figure(s) across %d run(s)", verb, total, len(names))
    for count in sorted(set(per_run.values()), reverse=True):
        runs = [n for n, c in per_run.items() if c == count]
        logger.info("  %d figure(s): %d run(s) -- %s", count, len(runs),
                    ", ".join(runs[:6]) + (" ..." if len(runs) > 6 else ""))
    logger.info("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

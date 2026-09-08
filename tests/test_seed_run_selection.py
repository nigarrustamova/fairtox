"""A seed aggregate must contain its own family and nothing else.

`run_seeds.sh` used to hand `seed_robustness` the glob `<base>*`. That is fine
for `main` and wrong for `dro`, because `dro*` also matches `drob*`: the Group
DRO aggregate silently averaged three Group DRO runs (FPR gap +0.19) together
with three DistilRoBERTa runs (-0.06) and reported the mean of two different
experiments.

The pipeline was not silent about it -- `seed_robustness` warned that the seeds
had landed on different outcome letters -- but a warning in a log is not a guard,
and the file sat in the results tree for two days. The script now passes an
explicit `analysis.seed_runs` list built from the seeds it just ran.

Two tests, because the bug has two halves: the glob is still available and still
over-matches (so anyone reaching for it should know), and the artefacts actually
committed are family-pure.
"""

from __future__ import annotations

import json
import re

import pytest

from fairtox.config import REPO_ROOT

RESULTS = REPO_ROOT / "results"


def test_a_prefix_glob_matches_a_sibling_family():
    """The trap itself, pinned: this is why the script no longer uses a glob."""
    import fnmatch

    families = ["dro", "dro_s43", "dro_s44", "drob", "drob_s43", "drob_s44"]
    caught = [name for name in families if fnmatch.fnmatch(name, "dro*")]

    assert "drob" in caught, "if this ever stops being true, the guard below can relax"
    assert len(caught) == 6

    explicit = ["dro", "dro_s43", "dro_s44"]
    assert [n for n in families if n in explicit] == explicit


def test_every_committed_seed_aggregate_holds_one_family_only():
    """The artefacts on disk, checked directly rather than through the code."""
    files = sorted(RESULTS.glob("*/metrics/seed_robustness.json"))
    if not files:
        pytest.skip("no results tree in this checkout")

    foreign = []
    for path in files:
        base = path.parents[1].name
        payload = json.loads(path.read_text(encoding="utf-8"))
        pattern = re.compile(rf"^{re.escape(base)}(_s\d+)?$")
        for run in payload.get("runs") or []:
            if not pattern.match(run):
                foreign.append(f"{base}: aggregated '{run}'")

    assert not foreign, (
        "a seed aggregate contains a run from another family:\n  "
        + "\n  ".join(foreign)
        + "\nRe-run it with an explicit list:\n"
        "  python scripts/run_all.py -c <config> --only seed_robustness --no-deps \\\n"
        "    --set run.name=<base> --set \"analysis.seed_runs=['<base>','<base>_s43',...]\""
    )


def test_the_dro_aggregate_is_the_corrected_one():
    """The specific file the bug produced, pinned to its corrected values."""
    path = RESULTS / "dro" / "metrics" / "seed_robustness.json"
    if not path.exists():
        pytest.skip("no dro run in this checkout")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["runs"] == ["dro", "dro_s43", "dro_s44"]
    assert payload["consistency"]["verdicts"] == ["D"], "all three seeds are outcome D"
    assert payload["summary"]["d_fpr_gap"]["mean"] == pytest.approx(0.1904, abs=5e-4)

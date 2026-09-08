"""The committed tables must still be what the committed inputs produce.

`analysis/study0*.md` and their CSVs are the report's tables. They are generated,
but they are also checked in, which means they can drift: edit the summariser, or
add a run to a family, and the files on disk keep saying what they said before.
Nothing would fail, and the paper would quote a stale number.

So this regenerates them from the committed digest and compares. It is the one
check that ties the three committed things -- the digest, the code, and the tables
-- into a single claim, and it runs everywhere pytest runs, including CI.

It is skipped rather than failed when the digest is absent, so a clone that has
not fetched the archive still gets a green suite.
"""

from __future__ import annotations

import subprocess
import sys

import pandas as pd
import pytest

from fairtox.config import REPO_ROOT

DIGEST = REPO_ROOT / "analysis" / "study_digest.json.gz"
SUMMARISE = REPO_ROOT / "scripts" / "summarise_study.py"
EXPECTED_TABLES = {
    "study01_families", "study02_dose_response", "study03_generalisation",
    "study03b_generalisation_per_seed", "study04_mechanism", "study05_postprocessing",
    "study06_integrity", "study07_threshold_free", "study08_ablation",
}


@pytest.fixture(scope="module")
def regenerated(tmp_path_factory):
    if not DIGEST.exists():
        pytest.skip(f"no digest at {DIGEST}; nothing to compare against")
    out = tmp_path_factory.mktemp("analysis")
    result = subprocess.run(
        [sys.executable, str(SUMMARISE), "--source", str(DIGEST), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return out


def test_every_expected_table_is_produced(regenerated):
    produced = {p.stem for p in regenerated.glob("*.csv")}
    assert produced == EXPECTED_TABLES


def test_the_committed_tables_match_what_the_code_produces(regenerated):
    """If this fails, run scripts/summarise_study.py and commit the result."""
    stale = []
    for produced in sorted(regenerated.glob("*.csv")):
        committed = REPO_ROOT / "analysis" / produced.name
        if not committed.exists():
            stale.append(f"{produced.name}: not committed")
            continue
        left, right = pd.read_csv(produced), pd.read_csv(committed)
        if left.shape != right.shape:
            stale.append(f"{produced.name}: {left.shape} vs committed {right.shape}")
            continue
        cells = int((left.fillna("~") != right.fillna("~")).sum().sum())
        if cells:
            stale.append(f"{produced.name}: {cells} cell(s) differ")
    assert not stale, "committed analysis tables are stale:\n  " + "\n  ".join(stale)


def test_the_markdown_tables_are_markdown():
    """`write_table` falls back to plain text when tabulate is missing.

    That fallback is silent and produces a `.md` file with no pipes in it, which
    is how the first version of these tables shipped as non-Markdown.
    """
    tables = sorted((REPO_ROOT / "analysis").glob("study*.md"))
    assert tables, "no analysis tables committed"
    for path in tables:
        assert "|" in path.read_text(encoding="utf-8"), f"{path.name} is not a Markdown table"


def test_the_headline_number_is_the_one_the_paper_quotes(regenerated):
    """A tripwire on the single number every other claim is measured against."""
    families = pd.read_csv(regenerated / "study01_families.csv").set_index("family")
    main = families.loc["main"]

    assert main["n_seeds"] == 8
    assert main["d_fpr_gap"] == pytest.approx(-0.0520, abs=5e-4)
    assert main["ci_high"] < 0, "the headline interval must exclude zero"
    assert main["n_improved"] == 8
    assert main["sign_p"] == pytest.approx(0.0078, abs=5e-4)

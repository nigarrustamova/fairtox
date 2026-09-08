"""Numbers written in the report's prose must match the numbers in the artefacts.

The tables and figures in `report/report.tex` are generated, so they cannot drift.
The prose is typed, and it drifted twice before this file existed: a spread was
described as falling "more than fivefold" when the two numbers actually quoted
differ by a factor of 3.7 (the 5.9 belonged to a like-for-like three-seed
comparison), and "five of the eight configurations have three seeds" was written
where four do.

Both are the same failure -- a sentence about numbers that was not re-derived from
the numbers -- and neither is visible to a LaTeX compiler. So the headline claims
are pinned here, against the same committed artefacts the tables are built from.

This does not check every sentence; it checks the ones a reader would quote.
"""

from __future__ import annotations

import re

import pytest

from fairtox.config import REPO_ROOT
from fairtox.evaluation.aggregate import family_deltas, load_study, paired_summary

REPORT = REPO_ROOT / "report" / "report.tex"
DECK = REPO_ROOT / "presentation" / "presentation.tex"
MAIN_SEEDS = ["main"] + [f"main_s{s}" for s in range(43, 50)]


@pytest.fixture(scope="module")
def tex() -> str:
    if not REPORT.exists():
        pytest.skip("no report/report.tex in this checkout")
    return REPORT.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def deck() -> str:
    if not DECK.exists():
        pytest.skip("no presentation/presentation.tex in this checkout")
    return DECK.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def study() -> dict:
    results = REPO_ROOT / "results"
    if not results.is_dir():
        pytest.skip("no results tree in this checkout")
    return load_study(results)


def test_the_headline_number_and_its_interval_are_the_measured_ones(tex, study):
    stats = paired_summary(family_deltas(study, MAIN_SEEDS, "fpr_gap"))

    assert f"{abs(stats['mean']):.4f}" == "0.0520"
    assert "0.0520" in tex
    assert f"{stats['ci_low']:.4f}" == "-0.0742"
    assert f"{stats['ci_high']:.4f}" == "-0.0297"
    assert "-0.0742, -0.0297" in tex.replace("$", "").replace("\\,", " ").replace("  ", " ")
    assert f"{stats['sign_p']:.4f}" == "0.0078"
    assert "0.0078" in tex


def test_the_safety_tax_is_the_ratio_the_paper_prints(tex, study):
    gap = paired_summary(family_deltas(study, MAIN_SEEDS, "fpr_gap"))["mean"]
    fnr = paired_summary(family_deltas(study, MAIN_SEEDS, "macro_fnr"))["mean"]
    tax = -fnr / gap

    assert f"{tax:.2f}" == "2.19", "the reported exchange rate moved"
    assert "2.19" in tex


def test_the_fixed_epoch_spread_claim_is_like_for_like(tex, study):
    """The claim that broke once: which two standard deviations are compared."""
    three = paired_summary(family_deltas(
        study, ["main", "main_s43", "main_s44"], "fpr_gap"))
    last = paired_summary(family_deltas(
        study, ["lastep", "lastep_s43", "lastep_s44"], "fpr_gap"))

    assert f"{three['sd']:.4f}" == "0.0427"
    assert f"{last['sd']:.4f}" == "0.0072"
    ratio = three["sd"] / last["sd"]
    assert f"{ratio:.1f}" == "5.9"

    # Both endpoints must appear, so the factor cannot be quoted against a
    # standard deviation the sentence does not name.
    assert "0.0427" in tex and "0.0072" in tex and "5.9" in tex


def test_the_seed_counts_in_the_prose_match_the_families(tex, study):
    """The other claim that broke: how many configurations have how many seeds."""
    from importlib import import_module

    summarise = import_module("summarise_study")
    counts: dict[int, int] = {}
    for spec in summarise.FAMILIES.values():
        n = len(family_deltas(study, spec["runs"], "fpr_gap"))
        counts[n] = counts.get(n, 0) + 1

    assert counts == {8: 2, 5: 2, 3: 4}, f"family seed counts changed: {counts}"
    assert "Four have" in tex and "two have five" in tex


def test_every_number_the_abstract_quotes_appears_in_a_generated_table(tex):
    """An abstract figure with no table behind it is a figure nobody can check."""
    abstract = tex.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    quoted = set(re.findall(r"\d\.\d{4}", abstract))

    tables = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (REPO_ROOT / "report" / "tables").glob("*.tex")
    )
    missing = sorted(v for v in quoted if v not in tables)
    assert not missing, f"abstract quotes figures no table contains: {missing}"


def test_no_template_boilerplate_survived(tex):
    """The brief makes leftover template text an automatic deduction."""
    for marker in ("Lorem ipsum", "XXX", "TODO", "\\todo", "First Author",
                   "Abstract--This", "Insert ", "PLACEHOLDER"):
        assert marker not in tex, f"template leftover in report.tex: {marker!r}"


def test_the_per_subgroup_counts_are_the_ones_the_audit_produces(tex, deck):
    """Both documents said thirteen of fourteen subgroups improved. Twelve do.

    The arithmetic gave it away -- thirteen improving, one unchanged and one worse
    is fifteen subgroups out of fourteen -- and a reader can count the bars in the
    figure. The counts are pinned here because they are the easiest claim in the
    study to check by eye and the most expensive one to get wrong.
    """
    import json

    def audit(arm: str) -> dict[str, dict]:
        path = REPO_ROOT / "results" / "main" / "metrics" / f"audit_{arm}.json"
        if not path.exists():
            pytest.skip("no results tree in this checkout")
        rows = json.loads(path.read_text(encoding="utf-8"))["subgroups"]
        return {r["subgroup"]: r for r in rows}

    base, mitigated = audit("baseline"), audit("mitigated")
    reportable = [k for k, r in base.items()
                  if r.get("reportable_fpr") and r.get("fpr") is not None]
    deltas = {k: mitigated[k]["fpr"] - base[k]["fpr"] for k in reportable}

    improved = sum(1 for d in deltas.values() if d < 0)
    unchanged = sum(1 for d in deltas.values() if d == 0)
    worse = sum(1 for d in deltas.values() if d > 0)

    assert (len(reportable), improved, unchanged, worse) == (14, 12, 1, 1)
    assert [k for k, d in deltas.items() if d > 0] == ["atheist"]
    assert [k for k, d in deltas.items() if d == 0] == ["asian"]

    for document in (tex, deck):
        assert "Thirteen" not in document and "13 of" not in document


def test_the_peak_memory_figure_is_the_measured_maximum(tex, deck):
    """The reported peak must be the largest peak, not a middling run's."""
    import json

    models = sorted((REPO_ROOT / "results").glob("*/metrics/model_*.json"))
    if not models:
        pytest.skip("no results tree in this checkout")

    peaks = [json.loads(p.read_text(encoding="utf-8")).get("training", {})
             .get("peak_reserved_gb") for p in models]
    peak = max(v for v in peaks if v)

    assert f"{peak:.2f}" == "3.99"
    assert "3.99" in tex and "3.99" in deck


def test_the_documented_test_count_is_the_real_one(tex, request):
    """The paper and the README both quote a test count; the README's was stale.

    Skipped on a partial run, because then the collected count is not the
    suite's. CI runs the whole suite, which is where this has to hold.
    """
    collected = len(request.session.items)
    if collected < 100:
        pytest.skip("partial run: the collected count is not the suite's")

    assert f"${collected}$ tests" in tex, f"report.tex does not say {collected} tests"

    readme = REPO_ROOT / "README.md"
    if readme.exists():
        assert f"# {collected} tests" in readme.read_text(encoding="utf-8"), \
            f"README.md does not say {collected} tests"


def test_the_abstract_ends_with_the_repository_link(tex):
    """Required by the brief, and easy to lose when the abstract is edited."""
    abstract = tex.split("\\begin{abstract}")[1].split("\\end{abstract}")[0]
    tail = abstract.strip().rstrip(".").rstrip()
    assert "\\repourl" in tail[-120:], "the repository link must close the abstract"
    assert "\\repotag" in tail[-120:], "the tag must be named with the link"

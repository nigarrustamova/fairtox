"""Stage resolution: the machinery that makes a scope cut a flag, not an edit."""

from __future__ import annotations

import pytest

import fairtox.stages  # noqa: F401  (importing registers every stage)
from fairtox import registry
from fairtox.registry import Tier

MUST_STAGES = {
    "eda", "tfidf_baseline", "train_baseline", "audit_baseline",
    "train_mitigated", "compare", "ablate", "error_analysis",
}


def test_every_stage_declares_a_summary():
    for name, stage in registry.all_stages().items():
        assert stage.summary, f"stage '{name}' has no summary"


def test_dependencies_all_exist():
    known = set(registry.all_stages())
    for name, stage in registry.all_stages().items():
        for dependency in stage.depends_on:
            assert dependency in known, f"'{name}' depends on unknown stage '{dependency}'"


def test_must_tier_is_the_expected_set():
    assert {s.name for s in registry.resolve(max_tier=Tier.MUST)} == MUST_STAGES


def test_must_tier_resolves_in_dependency_order():
    resolved = registry.resolve(max_tier=Tier.MUST)
    positions = {stage.name: i for i, stage in enumerate(resolved)}
    for stage in resolved:
        for dependency in stage.depends_on:
            assert positions[dependency] < positions[stage.name], (
                f"'{dependency}' must run before '{stage.name}'"
            )


def test_cheap_stages_run_before_expensive_ones():
    """A data problem should surface in the seconds eda takes, not an hour in."""
    order = [s.name for s in registry.resolve(max_tier=Tier.MUST)]
    assert order.index("eda") < order.index("train_baseline")
    assert order.index("tfidf_baseline") < order.index("train_mitigated")


def test_tiers_are_nested():
    must = {s.name for s in registry.resolve(max_tier=Tier.MUST)}
    optional = {s.name for s in registry.resolve(max_tier=Tier.OPTIONAL)}
    everything = {s.name for s in registry.resolve(max_tier=Tier.EXTRA)}
    assert must < optional < everything


def test_only_pulls_in_upstream_dependencies():
    resolved = {s.name for s in registry.resolve(["compare"])}
    assert resolved == {
        "eda", "train_baseline", "audit_baseline", "train_mitigated", "compare"
    }


def test_no_deps_runs_the_stage_alone():
    resolved = registry.resolve(["compare"], with_dependencies=False)
    assert [s.name for s in resolved] == ["compare"]


def test_no_deps_is_what_keeps_analysis_off_the_gpu():
    """The reason the flag exists: re-running analysis must not retrain."""
    with_deps = registry.resolve(["threshold_sensitivity"])
    without = registry.resolve(["threshold_sensitivity"], with_dependencies=False)

    assert any(s.trains for s in with_deps)
    assert not any(s.trains for s in without)


def test_dependencies_survive_a_tier_filter():
    """An OPTIONAL stage must still pull in its MUST dependencies."""
    resolved = {s.name for s in registry.resolve(["jigsaw_bias_metrics"])}
    assert "audit_baseline" in resolved and "train_baseline" in resolved


def test_training_stages_are_marked():
    """The plan warns before it spends the window, which needs this flag set."""
    trains = {name for name, s in registry.all_stages().items() if s.trains}
    assert trains == {"train_baseline", "train_mitigated", "ablate"}


def test_analysis_stages_are_not_marked_as_training():
    for name in ("eda", "audit_baseline", "compare", "error_analysis",
                 "threshold_sensitivity", "jigsaw_bias_metrics"):
        assert registry.get(name).trains is False


def test_unknown_stage_names_are_rejected_helpfully():
    with pytest.raises(KeyError, match="unknown stage"):
        registry.resolve(["not_a_stage"])


def test_tier_parsing():
    assert Tier.parse("must") is Tier.MUST
    assert Tier.parse("ALL") is Tier.EXTRA
    with pytest.raises(ValueError, match="unknown tier"):
        Tier.parse("nonsense")


def test_resolution_is_stable_between_calls():
    first = [s.name for s in registry.resolve(max_tier=Tier.EXTRA)]
    second = [s.name for s in registry.resolve(max_tier=Tier.EXTRA)]
    assert first == second


def test_planned_stages_are_extra_and_raise_rather_than_no_op():
    """A half-built stage must never look like a finished one."""
    for name in ("group_threshold_calibration", "roberta_backbone"):
        stage = registry.get(name)
        assert stage.tier is Tier.EXTRA
        with pytest.raises(NotImplementedError, match="stretch goal"):
            stage(None)

"""Driver CLI behaviour.

A silently dropped flag is worse than a crash: the run completes, the numbers
look plausible, and the thing you believed you had asked for never happened.
Both regressions pinned here are of exactly that kind.
"""

from __future__ import annotations

import pytest

import run_all
from fairtox.registry import Tier
from run_all import _coerce, build_parser


def test_repeated_set_flags_accumulate():
    """Regression: nargs='+' alone keeps only the last flag's values."""
    args = build_parser().parse_args(
        ["--set", "training.batch_size=16", "--set", "training.gradient_accumulation_steps=2"]
    )
    assert args.set == ["training.batch_size=16", "training.gradient_accumulation_steps=2"]


def test_multiple_values_after_one_set_flag():
    args = build_parser().parse_args(["--set", "a=1", "b=2"])
    assert args.set == ["a=1", "b=2"]


def test_repeated_only_flags_accumulate():
    args = build_parser().parse_args(["--only", "eda", "--only", "compare"])
    assert args.only == ["eda", "compare"]


def test_repeated_skip_flags_accumulate():
    args = build_parser().parse_args(["--skip", "ablate", "--skip", "attributions"])
    assert args.skip == ["ablate", "attributions"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1", 1),
        ("0.5", 0.5),
        ("True", True),
        ("None", None),
        ("[1, 2]", [1, 2]),
        ("jigsaw_hf.csv", "jigsaw_hf.csv"),  # bare strings survive literal_eval failing
        ("distilbert-base-uncased", "distilbert-base-uncased"),
        ("/kaggle/input/jigsaw", "/kaggle/input/jigsaw"),
    ],
)
def test_coerce_parses_values(text, expected):
    assert _coerce(text) == expected


def test_defaults_are_sane():
    args = build_parser().parse_args([])
    assert args.tier == "must"
    assert args.only is None
    assert args.skip == []
    assert args.set == []
    assert args.no_deps is False
    assert args.force is False


# -- the flag has to actually reach the driver -------------------------------


@pytest.fixture
def captured(monkeypatch):
    """Intercept pipeline.run and record the keyword arguments it received."""
    calls: dict = {}

    def fake_run(config, **kwargs):
        calls["config"] = config
        calls.update(kwargs)
        return []

    monkeypatch.setattr(run_all.pipeline, "run", fake_run)
    return calls


def test_no_deps_is_forwarded_to_the_driver(captured):
    """Regression: the flag was parsed and then never passed on.

    With it dropped, `--only compare --no-deps` silently resolved compare's
    dependencies and re-ran both training stages -- hours of a booked window,
    spent producing models that were already on disk.
    """
    run_all.main(["--config", "configs/smoke.yaml", "--only", "compare", "--no-deps", "--dry-run"])
    assert captured["with_dependencies"] is False


def test_dependencies_are_pulled_in_by_default(captured):
    run_all.main(["--config", "configs/smoke.yaml", "--only", "compare", "--dry-run"])
    assert captured["with_dependencies"] is True


def test_force_sets_the_retrain_flag(captured):
    run_all.main(["--config", "configs/smoke.yaml", "--force", "--dry-run"])
    assert captured["config"].get("run.force_retrain") is True


def test_force_is_off_by_default(captured):
    run_all.main(["--config", "configs/smoke.yaml", "--dry-run"])
    assert captured["config"].get("run.force_retrain") is False


def test_set_overrides_reach_the_config(captured):
    run_all.main(
        ["--config", "configs/smoke.yaml", "--set", "training.num_epochs=7", "--dry-run"]
    )
    assert captured["config"].get("training.num_epochs") == 7


def test_tier_is_parsed_and_forwarded(captured):
    run_all.main(["--config", "configs/smoke.yaml", "--tier", "optional", "--dry-run"])
    assert captured["max_tier"] is Tier.OPTIONAL


def test_malformed_set_is_rejected():
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        run_all.main(["--config", "configs/smoke.yaml", "--set", "no_equals_sign", "--dry-run"])

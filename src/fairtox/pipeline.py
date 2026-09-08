"""Run orchestration: the context handed to each stage, and the driver loop."""

from __future__ import annotations

import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import registry
from .config import REPO_ROOT, Config
from .registry import Stage, Tier
from .utils import io
from .utils.logging_utils import get_logger, setup_logging
from .utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class RunContext:
    """Everything a stage needs, and where it leaves things for later stages.

    ``artifacts`` is an in-process handoff (dataframes, checkpoint paths) for a
    single driver invocation. Anything a *later* invocation needs must also be
    written to disk, which is what makes ``--only audit_baseline --no-deps``
    work days after the training run that produced the predictions.
    """

    config: Config
    artifacts: dict[str, Any] = field(default_factory=dict)
    completed: list[str] = field(default_factory=list)

    @property
    def run_name(self) -> str:
        return self.config.run_name

    @property
    def seed(self) -> int:
        return self.config.seed

    def put(self, key: str, value: Any) -> None:
        self.artifacts[key] = value

    def take(self, key: str, default: Any = None) -> Any:
        return self.artifacts.get(key, default)

    def metrics_path(self, stage_name: str) -> Path:
        return io.metrics_path(self.run_name, stage_name)

    def figure_path(self, name: str, suffix: str = ".png") -> Path:
        return io.figure_path(self.run_name, name, suffix)

    def table_path(self, name: str, suffix: str = ".md") -> Path:
        return io.table_path(self.run_name, name, suffix)

    def checkpoint_dir(self, model_name: str) -> Path:
        return io.checkpoint_dir(self.run_name, model_name)

    def save_metrics(self, stage_name: str, payload: dict[str, Any]) -> Path:
        enriched = {"config_sha": self.config.fingerprint(), "seed": self.seed, **payload}
        return io.save_json(enriched, self.metrics_path(stage_name))


@dataclass
class StageResult:
    name: str
    status: str  # "ok" | "failed" | "skipped"
    seconds: float
    error: str | None = None


def environment_manifest(config: Config) -> dict[str, Any]:
    """Versions, hardware and commit -- everything the Setup section must state.

    Captured at run time rather than written from memory a week later, because
    reconstructing "which torch, which GPU, which commit" after the fact is how
    honest reporting quietly turns into fiction.
    """
    from .utils import device as device_utils

    manifest: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config_sha": config.fingerprint(),
        "config_source": config.source.name if config.source else None,
        "seed": config.seed,
    }
    for name in ("torch", "transformers", "numpy", "pandas", "sklearn"):
        try:
            module = __import__(name)
            manifest[name] = getattr(module, "__version__", "unknown")
        except ImportError:
            manifest[name] = "not installed"
    manifest["device"] = device_utils.describe_device(
        device_utils.resolve_device(str(config.get("run.device", "auto")))
    )
    manifest["git_commit"] = _git_commit()
    return manifest


def _git_commit() -> str | None:
    """Short commit hash, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def run(
    config: Config,
    *,
    only: list[str] | None = None,
    max_tier: Tier = Tier.MUST,
    skip: set[str] | None = None,
    dry_run: bool = False,
    keep_going: bool = False,
    with_dependencies: bool = True,
) -> list[StageResult]:
    """Resolve, then execute, the requested stages in dependency order."""
    io.ensure_dirs(config.run_name)
    setup_logging(
        level=str(config.get("run.log_level", "INFO")),
        log_file=io.run_dir(config.run_name) / "run.log",
    )

    stages = registry.resolve(only, max_tier=max_tier, with_dependencies=with_dependencies)
    if skip:
        unknown = skip - set(registry.all_stages())
        if unknown:
            raise KeyError(f"cannot skip unknown stage(s): {', '.join(sorted(unknown))}")
        stages = [st for st in stages if st.name not in skip]

    set_seed(config.seed)
    logger.info("run '%s' | seed %d | config %s", config.run_name, config.seed, config)
    _log_plan(stages, with_dependencies)

    if dry_run:
        return [StageResult(st.name, "skipped", 0.0) for st in stages]

    io.save_json(environment_manifest(config), io.metrics_path(config.run_name, "environment"))

    context = RunContext(config=config)
    results: list[StageResult] = []

    for st in stages:
        started = time.perf_counter()
        logger.info("--- stage: %s (%s) ---", st.name, st.tier.name.lower())
        try:
            # Reseed per stage so a stage's result does not depend on which
            # stages happened to run before it in this invocation.
            set_seed(config.seed)
            st(context)
        except Exception as exc:
            elapsed = time.perf_counter() - started
            logger.error("stage '%s' failed after %.1fs: %s", st.name, elapsed, exc)
            logger.debug(traceback.format_exc())
            results.append(StageResult(st.name, "failed", elapsed, str(exc)))
            if not keep_going:
                _log_summary(results)
                raise
            continue

        elapsed = time.perf_counter() - started
        context.completed.append(st.name)
        results.append(StageResult(st.name, "ok", elapsed))
        logger.info("stage '%s' finished in %.1fs", st.name, elapsed)

    _log_summary(results)
    return results


def _log_plan(stages: list[Stage], with_dependencies: bool) -> None:
    training = [st.name for st in stages if st.trains]
    logger.info("plan: %d stage(s)%s", len(stages), "" if with_dependencies else " (--no-deps)")
    for index, st in enumerate(stages, start=1):
        deps = f" <- {', '.join(st.depends_on)}" if st.depends_on else ""
        mark = " [GPU]" if st.trains else ""
        logger.info("  %d. %-22s [%s]%s%s", index, st.name, st.tier.name.lower(), mark, deps)
    if training:
        # The booked window is the one resource that cannot be recovered, so a
        # plan that will train says so before it starts rather than after.
        logger.warning(
            "this plan TRAINS (%s). Pass --no-deps if you only meant to re-run analysis.",
            ", ".join(training),
        )


def _log_summary(results: list[StageResult]) -> None:
    total = sum(r.seconds for r in results)
    ok = sum(1 for r in results if r.status == "ok")
    failed = [r for r in results if r.status == "failed"]
    logger.info("=" * 68)
    logger.info("%d/%d stage(s) ok in %.1fs", ok, len(results), total)
    for r in failed:
        logger.info("  FAILED %-22s %s", r.name, r.error)

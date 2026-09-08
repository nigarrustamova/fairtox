"""Stage registry.

Each experiment is a *stage*: a named function that reads the config, does one
unit of work, and writes its artefacts under ``results/``. Stages declare a tier
and their upstream dependencies; the driver topologically sorts whatever subset
you ask for.

The tiers let scope be cut with a flag rather than an edit:

``MUST``      the experiments the report is built on.
``OPTIONAL``  analyses that strengthen the paper and cost no training.
``EXTRA``     stretch goals; never required, and not all of them are built.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Tier(IntEnum):
    MUST = 0
    OPTIONAL = 1
    EXTRA = 2

    @classmethod
    def parse(cls, name: str) -> Tier:
        key = str(name).strip().upper()
        if key == "ALL":
            return cls.EXTRA
        try:
            return cls[key]
        except KeyError as exc:
            valid = ", ".join(t.name.lower() for t in cls)
            raise ValueError(f"unknown tier '{name}' (expected one of {valid}, all)") from exc


@dataclass(frozen=True)
class Stage:
    name: str
    fn: Callable[..., Any]
    tier: Tier
    summary: str
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    trains: bool = False  # consumes GPU time; used to warn before a long plan

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


_REGISTRY: dict[str, Stage] = {}


def _first_docstring_line(fn: Callable[..., Any]) -> str:
    """First non-empty line of ``fn``'s docstring, or an empty string."""
    doc = (fn.__doc__ or "").strip()
    if not doc:
        return ""
    # A docstring of pure whitespace strips to "" and splitlines() to [], so the
    # emptiness check above has to come first -- indexing [0] would raise.
    return doc.splitlines()[0].strip()


def stage(
    name: str,
    *,
    tier: Tier | str = Tier.MUST,
    summary: str = "",
    depends_on: Sequence[str] = (),
    trains: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a pipeline stage under ``name``."""
    resolved_tier = tier if isinstance(tier, Tier) else Tier.parse(str(tier))

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(f"stage '{name}' is already registered")
        _REGISTRY[name] = Stage(
            name=name,
            fn=fn,
            tier=resolved_tier,
            summary=summary or _first_docstring_line(fn),
            depends_on=tuple(depends_on),
            trains=trains,
        )
        return fn

    return decorator


def get(name: str) -> Stage:
    if name not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise KeyError(f"unknown stage '{name}'. Registered: {known}")
    return _REGISTRY[name]


def all_stages() -> dict[str, Stage]:
    return dict(_REGISTRY)


def clear() -> None:
    """Empty the registry. Tests only."""
    _REGISTRY.clear()


def resolve(
    requested: Iterable[str] | None = None,
    *,
    max_tier: Tier = Tier.MUST,
    with_dependencies: bool = True,
) -> list[Stage]:
    """Return the stages to run, in dependency order.

    ``requested`` selects stages by name; when it is ``None`` every stage at or
    below ``max_tier`` is selected. Dependencies are pulled in regardless of
    their own tier -- a MUST dependency can never be starved by a tier filter --
    unless ``with_dependencies`` is False, which runs exactly what was asked for
    and nothing else.
    """
    if requested is None:
        selected = {name for name, st in _REGISTRY.items() if st.tier <= max_tier}
    else:
        selected = set()
        for name in requested:
            get(name)  # raises with a helpful message on typos
            selected.add(name)

    if with_dependencies:
        frontier = list(selected)
        while frontier:
            current = _REGISTRY[frontier.pop()]
            for dep in current.depends_on:
                if dep not in selected:
                    get(dep)
                    selected.add(dep)
                    frontier.append(dep)

    return _topological_sort(selected)


def _depth(name: str, cache: dict[str, int]) -> int:
    """Length of the longest dependency chain ending at ``name``."""
    if name in cache:
        return cache[name]
    deps = [d for d in _REGISTRY[name].depends_on if d in _REGISTRY]
    # Provisional value guards against unbounded recursion on a cyclic graph;
    # `visit` below is what actually reports the cycle, with the trail.
    cache[name] = 0
    cache[name] = 0 if not deps else 1 + max(_depth(d, cache) for d in deps)
    return cache[name]


def _topological_sort(selected: set[str]) -> list[Stage]:
    ordered: list[Stage] = []
    permanent: set[str] = set()
    temporary: set[str] = set()

    def visit(name: str, trail: tuple[str, ...]) -> None:
        if name in permanent:
            return
        if name in temporary:
            raise ValueError(f"circular stage dependency: {' -> '.join((*trail, name))}")
        temporary.add(name)
        for dep in sorted(_REGISTRY[name].depends_on):
            if dep in selected:
                visit(dep, (*trail, name))
        temporary.discard(name)
        permanent.add(name)
        ordered.append(_REGISTRY[name])

    # Visit shallow stages first, so cheap foundational work runs before the
    # expensive training stages: a data problem should surface in the seconds
    # `eda` takes, not an hour into fine-tuning. Ties break on name, so the
    # order stays stable and reproducible between runs.
    depth_cache: dict[str, int] = {}
    for name in sorted(selected, key=lambda n: (_depth(n, depth_cache), n)):
        visit(name, ())
    return ordered

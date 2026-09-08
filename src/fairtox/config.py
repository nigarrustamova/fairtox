"""Configuration loading.

One study, one config file: stages read it by dotted key, and ``extends:`` lets a
machine-specific file adjust a few values without restating the study.

One file drives both arms deliberately. Configured separately, a divergent batch
size, learning rate, seed or split could creep into exactly the comparison the
paper rests on, and nothing in the output would reveal it.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

_MISSING = object()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


class Config:
    """A nested dict with dotted access and an ``extends`` chain."""

    def __init__(self, data: dict[str, Any] | None = None, source: Path | None = None) -> None:
        self._data: dict[str, Any] = copy.deepcopy(data or {})
        self.source = source

    # -- construction --------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path, _seen: set[Path] | None = None) -> Config:
        """Load ``path``, resolving any ``extends:`` parent first."""
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = (REPO_ROOT / resolved).resolve()
        if not resolved.exists():
            available = sorted(p.name for p in (REPO_ROOT / "configs").glob("*.yaml"))
            raise FileNotFoundError(
                f"config not found: {resolved}\nAvailable in configs/: {', '.join(available)}"
            )

        seen = _seen or set()
        if resolved in seen:
            chain = " -> ".join(p.name for p in seen)
            raise ValueError(f"circular 'extends' in config chain: {chain} -> {resolved.name}")
        seen.add(resolved)

        with resolved.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise TypeError(f"{resolved.name} must contain a YAML mapping at the top level")

        parent_name = data.pop("extends", None)
        if parent_name:
            parent = cls.load(resolved.parent / str(parent_name), _seen=seen)
            data = _deep_merge(parent.as_dict(), data)

        return cls(data, source=resolved)

    # -- access --------------------------------------------------------------

    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        """Read a key that must be present, with a message naming the file."""
        value = self.get(dotted_key, _MISSING)
        if value is _MISSING:
            where = self.source.name if self.source else "<in-memory config>"
            raise KeyError(f"required config key '{dotted_key}' is missing from {where}")
        return value

    def set(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        node = self._data
        for part in parts[:-1]:
            existing = node.get(part)
            if not isinstance(existing, dict):
                existing = {}
                node[part] = existing
            node = existing
        node[parts[-1]] = value

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- identity ------------------------------------------------------------

    @property
    def run_name(self) -> str:
        """The study name. Model runs live underneath it, named by their stage."""
        return str(self.get("run.name", "main"))

    @property
    def seed(self) -> int:
        return int(self.get("run.seed", 42))

    def fingerprint(self) -> str:
        """Stable hash of the resolved config.

        Recorded with every run so a results file can be traced back to the exact
        settings that produced it, including any ``--set`` overrides applied on
        the command line -- which are otherwise invisible after the fact.
        """
        payload = json.dumps(self._data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def __repr__(self) -> str:
        where = self.source.name if self.source else "in-memory"
        return f"Config({where}, run={self.run_name!r}, seed={self.seed}, sha={self.fingerprint()})"

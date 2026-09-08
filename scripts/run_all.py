#!/usr/bin/env python
"""FairTox experiment driver -- the single entry point.

    python scripts/run_all.py --config configs/a100.yaml --tier must

``--tier must``      the experiments the paper is built on (this trains).
``--tier optional``  adds the analyses that need no training.
``--tier all``       adds the EXTRA tier, which contains deliberately
                     unimplemented stretch goals and will raise. It exists so
                     ``--list`` can show the full design space; it is not a
                     "run everything" switch.
``--only <stage>``   run specific stages. Dependencies are pulled in
                     automatically -- add ``--no-deps`` to run exactly what you
                     named and nothing else.

WATCH THE PLAN LINE. ``--only`` without ``--no-deps`` will happily re-run the
training stages a stage depends on. The driver prints the resolved plan, marks
every stage that trains with [GPU], and warns before it starts.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fairtox import (  # noqa: E402
    pipeline,
    registry,
    stages,  # noqa: F401  (importing registers every stage)
)
from fairtox.config import Config  # noqa: E402
from fairtox.registry import Tier  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_all",
        description="FairTox experiment driver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", "-c", default="configs/smoke.yaml", help="path to a run config YAML"
    )
    parser.add_argument(
        "--tier", "-t", default="must",
        help="highest tier to run: must | optional | extra (alias: all)",
    )
    # action="extend" so repeated flags accumulate. With a plain nargs="+",
    # `--only a --only b` silently keeps only b -- which on training day means a
    # stage you thought you had selected was never selected.
    parser.add_argument(
        "--only", nargs="+", action="extend", metavar="STAGE",
        help="run only these stages (dependencies are added unless --no-deps)",
    )
    parser.add_argument(
        "--skip", nargs="+", action="extend", metavar="STAGE", default=[],
        help="stages to exclude",
    )
    parser.add_argument(
        "--no-deps", action="store_true",
        help="with --only, run exactly the named stages and pull in nothing else",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="retrain models whose checkpoint and predictions already exist",
    )
    parser.add_argument(
        "--keep-going", action="store_true",
        help="continue after a stage fails instead of aborting",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan and exit")
    parser.add_argument("--list", action="store_true", help="list registered stages and exit")
    parser.add_argument(
        "--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
        help="override config entries, e.g. --set training.num_epochs=1 (repeatable)",
    )
    return parser


def _list_stages() -> int:
    known = registry.all_stages()
    if not known:
        print("No stages registered.")
        return 1
    width = max(len(name) for name in known)
    for tier in Tier:
        in_tier = {n: s for n, s in known.items() if s.tier is tier}
        if not in_tier:
            continue
        print(f"\n{tier.name}")
        print("-" * (width + 62))
        for name in sorted(in_tier):
            st = in_tier[name]
            deps = f"   <- {', '.join(st.depends_on)}" if st.depends_on else ""
            mark = " [GPU]" if st.trains else ""
            print(f"  {name:<{width}}{mark}  {st.summary}{deps}")
    print("\n[GPU] marks stages that train a model.\n")
    return 0


def _coerce(text: str) -> object:
    """Best-effort literal parsing for --set overrides."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list:
        return _list_stages()

    config = Config.load(args.config)
    for override in args.set:
        if "=" not in override:
            raise SystemExit(f"--set expects KEY=VALUE, got '{override}'")
        key, _, value = override.partition("=")
        config.set(key.strip(), _coerce(value.strip()))
    if args.force:
        config.set("run.force_retrain", True)

    results = pipeline.run(
        config,
        only=args.only,
        max_tier=Tier.parse(args.tier),
        skip=set(args.skip),
        dry_run=args.dry_run,
        keep_going=args.keep_going,
        with_dependencies=not args.no_deps,
    )
    return 1 if any(r.status == "failed" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())

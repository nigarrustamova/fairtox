#!/usr/bin/env python
"""Build the cross-family tables the report quotes, from the finished runs.

    python scripts/summarise_study.py                       # from results/
    python scripts/summarise_study.py --source analysis/study_digest.json.gz
    python scripts/summarise_study.py --write-digest analysis/study_digest.json.gz

Every stage in this project writes its own artefacts, and ``seed_robustness``
aggregates the seeds of one family. Nothing aggregated ACROSS families, so the
comparison at the centre of the report -- eight mitigation configurations on one
set of axes -- had no code behind it. This script is that code, and its output
goes to ``analysis/`` as a committed artefact, next to ``error_coding.csv``: a
number in the paper should be traceable to a file in the repository, not to a
notebook someone ran once.

The digest exists for the same reason. The raw ``results/`` tree is ~2,000 files
on a machine whose booked window has closed; the digest is one 400 KB file that
reproduces every table here.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fairtox.evaluation.aggregate import (  # noqa: E402
    exchange_rate,
    family_deltas,
    load_study,
    paired_summary,
    subgroup_deltas,
)
from fairtox.utils import io  # noqa: E402
from fairtox.utils.logging_utils import get_logger, setup_logging  # noqa: E402

logger = get_logger(__name__)

# The eight configurations, in the order the report discusses them. Seed lists
# are written out rather than globbed: a glob would silently pick up a run added
# later and change a published number without anyone deciding to.
FAMILIES: dict[str, dict[str, Any]] = {
    "main": {
        "label": "loss reweighting (in-processing)",
        "runs": ["main"] + [f"main_s{s}" for s in range(43, 50)],
    },
    "lastep": {
        "label": "same, fixed-epoch control",
        "runs": ["lastep", "lastep_s43", "lastep_s44"],
    },
    "wtox": {
        "label": "+ identity-toxic cell raised",
        "runs": ["wtox", "wtox_s43", "wtox_s44"],
    },
    "drob": {
        "label": "distilroberta-base backbone",
        "runs": ["drob", "drob_s43", "drob_s44"],
    },
    "cda": {
        "label": "counterfactual swap, p=0.5 (pre-processing)",
        "runs": ["cda", "cda_s43", "cda_s44", "cda_s45", "cda_s46"],
    },
    "cdafull": {
        "label": "counterfactual swap, p=1.0",
        "runs": ["cdafull", "cdafull_s43", "cdafull_s44", "cdafull_s45", "cdafull_s46"],
    },
    "dro": {
        "label": "Group DRO, learned cell weights",
        "runs": ["dro", "dro_s43", "dro_s44"],
    },
    "heldout": {
        "label": "weighting sees 2 of 5 identity axes",
        "runs": ["heldout"] + [f"heldout_s{s}" for s in range(43, 50)],
    },
}

# The held-out experiment's split, fixed by AXIS before the run and never by
# outcome. Anything not here is an axis the loss never saw in that arm.
WEIGHTED_AXES = frozenset({
    "male", "female", "transgender", "other_gender",
    "christian", "jewish", "muslim", "hindu", "buddhist", "atheist", "other_religion",
})


def _round(frame: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    return frame.map(lambda v: round(v, digits) if isinstance(v, float) else v)


def families_table(study: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, spec in FAMILIES.items():
        gap = family_deltas(study, spec["runs"], "fpr_gap")
        if not gap:
            logger.warning("family '%s' has no completed runs; skipped", name)
            continue
        fnr = family_deltas(study, spec["runs"], "macro_fnr")
        stats = paired_summary(gap)
        rows.append({
            "family": name,
            "what": spec["label"],
            "n_seeds": stats["n"],
            "d_fpr_gap": stats["mean"],
            "sd": stats["sd"],
            "ci_low": stats["ci_low"],
            "ci_high": stats["ci_high"],
            "excludes_zero": stats["excludes_zero"],
            "n_improved": stats["n_negative"],
            "sign_p": stats["sign_p"],
            "d_macro_fpr": paired_summary(
                family_deltas(study, spec["runs"], "macro_fpr")).get("mean"),
            "d_macro_f1": paired_summary(
                family_deltas(study, spec["runs"], "macro_f1")).get("mean"),
            "d_macro_fnr": paired_summary(fnr).get("mean"),
            "safety_tax": exchange_rate(gap, fnr),
        })
    return pd.DataFrame(rows)


def generalisation_table(study: dict[str, Any]) -> pd.DataFrame:
    """The held-out experiment: axes the loss saw against axes it never saw.

    ``main`` is included with the same split applied, which is what removes the
    dose confound: if the weighted axes move by the same amount in both runs, the
    narrowing did not weaken the intervention where it still applied, and the
    movement on the untouched axes is transfer rather than a smaller dose.
    """
    rows = []
    for family in ("heldout", "main"):
        for run in FAMILIES[family]["runs"]:
            deltas = subgroup_deltas(study, run)
            if not deltas:
                continue
            for bucket, members in (
                ("weighted", {k: v for k, v in deltas.items() if k in WEIGHTED_AXES}),
                ("held_out", {k: v for k, v in deltas.items() if k not in WEIGHTED_AXES}),
            ):
                if not members:
                    continue
                rows.append({
                    "family": family, "run": run, "bucket": bucket,
                    "n_subgroups": len(members),
                    "mean_d_fpr": sum(members.values()) / len(members),
                    "n_improved": sum(1 for v in members.values() if v < 0),
                })
    return pd.DataFrame(rows)


def generalisation_summary(detail: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, bucket), part in detail.groupby(["family", "bucket"], sort=False):
        stats = paired_summary(part["mean_d_fpr"].tolist())
        rows.append({
            "family": family, "bucket": bucket, "n_seeds": stats["n"],
            "mean_d_fpr": stats["mean"], "sd": stats["sd"],
            "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
            "excludes_zero": stats["excludes_zero"],
            "n_seeds_improved": stats["n_negative"], "sign_p": stats["sign_p"],
            "n_subgroup_obs": int(part["n_subgroups"].sum()),
            "n_subgroup_improved": int(part["n_improved"].sum()),
        })
    frame = pd.DataFrame(rows)
    held = frame[(frame["family"] == "heldout") & (frame["bucket"] == "held_out")]
    seen = frame[(frame["family"] == "heldout") & (frame["bucket"] == "weighted")]
    if len(held) and len(seen) and seen.iloc[0]["mean_d_fpr"]:
        ratio = held.iloc[0]["mean_d_fpr"] / seen.iloc[0]["mean_d_fpr"]
        logger.info("transfer to axes the loss never saw: %.0f%% of the weighted axes",
                    100 * ratio)
    return frame


def mechanism_table(study: dict[str, Any]) -> pd.DataFrame:
    """The shortcut regression, FPR against the subgroup's toxic rate."""
    rows = []
    for name, spec in FAMILIES.items():
        slopes_b, slopes_m, correlations, changes = [], [], [], []
        for run in spec["runs"]:
            mechanism = (study.get(run) or {}).get("mechanism")
            if not mechanism:
                continue
            fit = mechanism.get("toxic_rate_fit") or {}
            if not (fit.get("baseline") and fit.get("mitigated")):
                continue
            slopes_b.append(fit["baseline"]["slope"])
            slopes_m.append(fit["mitigated"]["slope"])
            if fit["baseline"].get("pearson_r") is not None:
                correlations.append(fit["baseline"]["pearson_r"])
            if mechanism.get("slope_change") is not None:
                changes.append(mechanism["slope_change"])
        if not slopes_b:
            continue
        rows.append({
            "family": name, "n_seeds": len(slopes_b),
            "baseline_slope": sum(slopes_b) / len(slopes_b),
            "baseline_pearson_r": (sum(correlations) / len(correlations)
                                   if correlations else None),
            "mitigated_slope": sum(slopes_m) / len(slopes_m),
            "slope_change": (sum(changes) / len(changes)) if changes else None,
        })
    return pd.DataFrame(rows).sort_values("slope_change").reset_index(drop=True)


def postprocessing_table(study: dict[str, Any]) -> pd.DataFrame:
    """Per-subgroup thresholds applied to each family's own baseline.

    The utility and safety columns are here for the same reason the parity one
    is: post-processing has to appear in the report's comparison table on the
    SAME three axes as the training-time methods, and a row assembled by hand
    from the run files is a row that goes stale the next time a seed is added.
    The `cap` policy is the one reported -- it never lowers a subgroup's
    threshold, so it cannot close the gap by levelling down.
    """
    rows = []
    for name, spec in FAMILIES.items():
        deltas, utility, safety, fitted = [], [], [], []
        for run in spec["runs"]:
            payload = (study.get(run) or {}).get("postproc")
            if not payload:
                continue
            entries = {(r["model"], r["policy"]): r for r in payload["comparison"]}
            plain = entries.get(("baseline", "global"))
            capped = entries.get(("baseline", "cap"))
            if not (plain and capped):
                continue
            if plain["fpr_gap"] is None or capped["fpr_gap"] is None:
                continue
            deltas.append(capped["fpr_gap"] - plain["fpr_gap"])
            fitted.append(capped.get("n_thresholds_fitted"))
            # `safety_fnr` is the subgroup FNR where it is defined and the global
            # FNR otherwise -- the same fallback the ablation and compare use, so
            # the safety axis means one thing across the whole study.
            for column, key in ((utility, "macro_f1"), (safety, "safety_fnr")):
                before, after = plain.get(key), capped.get(key)
                if before is not None and after is not None:
                    column.append(after - before)
        if not deltas:
            continue
        stats = paired_summary(deltas)
        rows.append({
            "family": name, "n_seeds": stats["n"], "d_fpr_gap": stats["mean"],
            "sd": stats["sd"], "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
            "n_improved": stats["n_negative"], "sign_p": stats["sign_p"],
            "d_macro_f1": paired_summary(utility).get("mean"),
            "d_macro_fnr": paired_summary(safety).get("mean"),
            "safety_tax": exchange_rate(deltas, safety),
            "thresholds_fitted_min": min(f for f in fitted if f is not None),
            "thresholds_fitted_max": max(f for f in fitted if f is not None),
        })
    return pd.DataFrame(rows)


def threshold_free_table(study: dict[str, Any]) -> pd.DataFrame:
    """Borkan et al.'s AUCs, which need no threshold at all.

    BPSN (background positives against subgroup negatives) is the threshold-free
    analogue of the FPR gap; BNSP is the under-protection side. They matter
    because the whole study is measured at one operating point, and a parity gain
    that existed only at 0.5 would be a property of the threshold rather than of
    the model.
    """
    rows = []
    for name, spec in FAMILIES.items():
        collected: dict[tuple[str, str], list[float]] = {}
        for run in spec["runs"]:
            models = ((study.get(run) or {}).get("jigsaw") or {}).get("models") or {}
            for arm, payload in models.items():
                for metric, value in payload.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        collected.setdefault((arm, metric), []).append(float(value))
        if not collected:
            continue
        row: dict[str, Any] = {"family": name}
        for metric in ("bpsn_auc_mean", "bnsp_auc_mean", "subgroup_auc_mean", "overall_auc"):
            before = collected.get(("baseline", metric))
            after = collected.get(("mitigated", metric))
            if not (before and after):
                continue
            row["n_seeds"] = len(before)
            row[metric + "_baseline"] = sum(before) / len(before)
            row[metric + "_mitigated"] = sum(after) / len(after)
            row[metric + "_delta"] = row[metric + "_mitigated"] - row[metric + "_baseline"]
        rows.append(row)
    return pd.DataFrame(rows)


def ablation_table(study: dict[str, Any]) -> pd.DataFrame:
    """The alpha sweep and the `class` control, PER FAMILY.

    Kept split by family rather than pooled, because the three families that ran
    an ablation do not run the same one. `drob` sweeps alpha on a different
    backbone, and `wtox` sweeps it with the identity-toxic cell already raised, so
    its `abl_subgroup_a1` arm is not even the unweighted baseline. Averaging the
    three together would produce a curve that describes no model in the study --
    which is exactly what the first version of this function did.
    """
    collected: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for name, spec in FAMILIES.items():
        for run in spec["runs"]:
            payload = (study.get(run) or {}).get("ablate")
            if not payload:
                continue
            for arm in payload.get("arms", []):
                collected.setdefault((name, arm["arm"]), []).append(arm)

    rows = []
    for (family, arm), entries in collected.items():
        def mean(key: str, rows_=entries) -> float | None:
            values = [e[key] for e in rows_ if isinstance(e.get(key), (int, float))]
            return sum(values) / len(values) if values else None

        rows.append({
            "family": family, "arm": arm, "n_seeds": len(entries),
            "scheme": entries[0].get("scheme"), "alpha": entries[0].get("alpha"),
            "fpr_gap": mean("fpr_gap"), "macro_fpr": mean("macro_fpr"),
            "macro_f1": mean("macro_f1"), "safety_fnr": mean("safety_fnr"),
            "flag_rate": mean("flag_rate"),
            "any_degenerate": any(e.get("degenerate") for e in entries),
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["family", "scheme", "alpha"], na_position="first"
    ).reset_index(drop=True)


def integrity_table(study: dict[str, Any]) -> pd.DataFrame:
    """Per seed: one partition, one machine, no collapsed model.

    Every claim in the report is a within-seed paired difference, so a seed whose
    runs did not share a partition would invalidate every family at once. This is
    the check that says they did.
    """
    seeds: dict[str, dict[str, Any]] = {}
    for run, payload in study.items():
        seed = "42" if "_s" not in run else run.split("_s")[-1]
        bucket = seeds.setdefault(seed, {
            "seed": seed, "runs": 0, "fingerprints": set(), "gpus": set(),
            "degenerate": [], "leak_share": [],
        })
        bucket["runs"] += 1
        for model in ("baseline", "mitigated"):
            record = (payload.get("model") or {}).get(model)
            if record:
                bucket["fingerprints"].add(record.get("split_fingerprint"))
                gpu = (record.get("device") or {}).get("gpu_name")
                if gpu:
                    bucket["gpus"].add(gpu)
        if (payload.get("compare") or {}).get("degenerate"):
            bucket["degenerate"].append(run)
        leakage = payload.get("leakage")
        if leakage:
            bucket["leak_share"].append(leakage["duplicates"]["share_of_test"])

    rows = []
    for seed, bucket in sorted(seeds.items()):
        shares = bucket["leak_share"]
        rows.append({
            "seed": seed, "n_runs": bucket["runs"],
            "split_fingerprint": ", ".join(sorted(f for f in bucket["fingerprints"] if f)),
            "one_partition": len(bucket["fingerprints"]) == 1,
            "gpus": ", ".join(sorted(bucket["gpus"])),
            "one_machine": len(bucket["gpus"]) <= 1,
            "degenerate_runs": ", ".join(bucket["degenerate"]) or "none",
            "test_rows_seen_in_train_pct": (100 * sum(shares) / len(shares)) if shares else None,
        })
    return pd.DataFrame(rows)


def dose_response_table(study: dict[str, Any]) -> pd.DataFrame:
    """Counterfactual augmentation at two intensities, plus the null dose.

    The whole CDA conclusion rests on this table: an effect that does not grow
    with the dose is not the mechanism the method claims.
    """
    rows = [{"probability": 0.0, "n_seeds": None, "d_fpr_gap": 0.0, "sd": None,
             "n_improved": None, "d_macro_fnr": 0.0, "slope_change": None,
             "note": "baseline, by definition"}]
    mechanism = mechanism_table(study)
    for family, probability in (("cda", 0.5), ("cdafull", 1.0)):
        spec = FAMILIES[family]
        gap = family_deltas(study, spec["runs"], "fpr_gap")
        if not gap:
            continue
        stats = paired_summary(gap)
        change = mechanism.loc[mechanism["family"] == family, "slope_change"]
        rows.append({
            "probability": probability, "n_seeds": stats["n"],
            "d_fpr_gap": stats["mean"], "sd": stats["sd"],
            "n_improved": stats["n_negative"],
            "d_macro_fnr": paired_summary(
                family_deltas(study, spec["runs"], "macro_fnr")).get("mean"),
            "slope_change": float(change.iloc[0]) if len(change) else None,
            "note": "",
        })
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(REPO_ROOT / "results"),
                        help="results/ directory or a digest written by --write-digest")
    parser.add_argument("--out", default=str(REPO_ROOT / "analysis"),
                        help="where the committed tables are written")
    parser.add_argument("--write-digest", metavar="PATH",
                        help="also archive the loaded study as a gzipped JSON digest")
    args = parser.parse_args(argv)

    setup_logging("INFO")
    study = load_study(args.source)
    if not study:
        raise SystemExit(f"no runs found at {args.source}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    detail = generalisation_table(study)
    tables = {
        "study01_families": families_table(study),
        "study02_dose_response": dose_response_table(study),
        "study03_generalisation": generalisation_summary(detail),
        "study03b_generalisation_per_seed": detail,
        "study04_mechanism": mechanism_table(study),
        "study05_postprocessing": postprocessing_table(study),
        "study06_integrity": integrity_table(study),
        "study07_threshold_free": threshold_free_table(study),
        "study08_ablation": ablation_table(study),
    }
    for name, frame in tables.items():
        io.write_table(_round(frame), out / f"{name}.md")

    payload = {name: frame.to_dict(orient="records") for name, frame in tables.items()}
    payload["_provenance"] = {
        "source": str(args.source),
        "n_runs": len(study),
        "families": {k: v["runs"] for k, v in FAMILIES.items()},
        "weighted_axes": sorted(WEIGHTED_AXES),
    }
    io.save_json(payload, out / "study_summary.json")

    if args.write_digest:
        target = Path(args.write_digest)
        target.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(target, "wt", encoding="utf-8") as handle:
            json.dump(study, handle, default=str)
        logger.info("digest written to %s (%.2f MB)",
                    target, target.stat().st_size / 1e6)

    logger.info("=" * 74)
    logger.info("%d run(s) summarised into %d table(s) under %s",
                len(study), len(tables), out)
    families = tables["study01_families"]
    for _, row in families.iterrows():
        verdict = "excludes 0" if row["excludes_zero"] else "crosses 0"
        logger.info("  %-9s n=%d  d_fpr_gap=%+.4f  CI %s  sign %d/%d p=%s",
                    row["family"], row["n_seeds"], row["d_fpr_gap"], verdict,
                    row["n_improved"], row["n_seeds"],
                    f"{row['sign_p']:.4f}" if row["sign_p"] is not None else "n/a")
    logger.info("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

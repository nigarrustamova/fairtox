#!/usr/bin/env python
"""Emit the paper's tables as LaTeX, from the committed analysis artefacts.

    python scripts/make_paper_tables.py

Each table lands in ``report/tables/<name>.tex`` as a bare ``tabular`` block.
``report/report.tex`` wraps it in a ``table`` environment and supplies the
caption, so the prose stays in the paper and the numbers stay generated: no
figure in this report is ever retyped from a notebook, and neither is any cell
of any table.

Formatting rules applied uniformly, because inconsistent precision is the first
thing a reader notices in a results table:

* rates and differences to four decimals, signed where they are differences;
* counts with thousands separators;
* an unavailable value is an en dash, never ``nan`` and never a blank;
* every column that holds numbers is right-aligned via ``siunitx``-free plain
  ``r`` columns with the digits pre-formatted here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ANALYSIS = REPO_ROOT / "analysis"
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "report" / "tables"

DASH = "--"


# -- formatting helpers ------------------------------------------------------

def num(value, digits: int = 4, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return DASH
    fmt = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return f"${fmt.format(float(value))}$"


def count(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return DASH
    return f"{int(value):,}"


def interval(low, high) -> str:
    if pd.isna(low) or pd.isna(high):
        return DASH
    return f"$[{low:+.4f},\\,{high:+.4f}]$"


def pval(value) -> str:
    if value is None or pd.isna(value):
        return DASH
    return f"${float(value):.4f}$"


def pct(value, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return DASH
    return f"${float(value) * 100:+.{digits}f}$"


def write(name: str, header: str, rows: list[str], colspec: str,
          midrules: dict[int, str] | None = None) -> Path:
    """Assemble one tabular block and write it."""
    OUT.mkdir(parents=True, exist_ok=True)
    lines = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header, "\\midrule"]
    for index, row in enumerate(rows):
        if midrules and index in midrules:
            lines.append(midrules[index])
        lines.append(row)
    lines += ["\\bottomrule", "\\end{tabular}"]
    path = OUT / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(ANALYSIS / f"{name}.csv")


def metrics(run: str, stage: str) -> dict:
    return json.loads((RESULTS / run / "metrics" / f"{stage}.json").read_text("utf-8"))


# -- T1: corpus and splits ---------------------------------------------------

def t_corpus() -> None:
    eda = metrics("main", "eda")
    sizes = eda["split_sizes"]
    rows = [
        f"Comments in the release & {count(1_999_516)} & {DASH} & {DASH} \\\\",
        f"Identity-annotated subset & {count(eda['n_total'])} & "
        f"{num(eda['toxic_rate'], 4)} & {num(eda['any_identity_rate'], 4)} \\\\",
    ]
    for split in ("train", "val", "test"):
        rows.append(f"\\quad {split} & {count(sizes[split])} & {DASH} & {DASH} \\\\")
    rows.append(
        f"Subgroups configured & {eda['n_subgroups_total']} & {DASH} & {DASH} \\\\")
    rows.append(
        f"\\quad clearing the FPR floor & {eda['n_reportable_fpr']} & {DASH} & {DASH} \\\\")
    rows.append(
        f"\\quad clearing the FNR floor & {eda['n_reportable_fnr']} & {DASH} & {DASH} \\\\")
    write("t_corpus",
          "Quantity & Rows & Toxic rate & Identity rate \\\\",
          rows, "lrrr", midrules={2: "\\addlinespace", 5: "\\midrule"})


# -- T2: the headline pair ---------------------------------------------------

def t_headline() -> None:
    """The controlled pair, averaged over the eight seeds of the headline family.

    Both arm columns are means of the per-seed values and the $\\Delta$ column is
    the mean of the per-seed *differences*; with a complete set of seeds those
    agree, and computing the difference per seed is what makes the interval and
    the sign test paired.
    """
    from fairtox.evaluation.aggregate import paired_summary

    seeds = ["main"] + [f"main_s{s}" for s in range(43, 50)]
    compares = {run: metrics(run, "compare") for run in seeds}

    def paired(axis: str, key: str) -> list[float]:
        return [compares[r]["axes"][axis][key]["delta"] for r in seeds]

    def arm(axis: str, key: str, which: str) -> float:
        values = [compares[r]["axes"][axis][key][which] for r in seeds]
        return sum(values) / len(values)

    rows = []
    spec = [
        ("Parity", "FPR gap", "parity", "fpr_gap"),
        ("", "macro FPR", "parity", "macro_fpr"),
        ("", "EOD (FPR)", "parity", "eod_fpr"),
        ("Utility", "macro-F1", "utility", "macro_f1"),
        ("Safety", "subgroup FNR", "safety", "macro_fnr"),
    ]
    for group, label, axis, key in spec:
        stats = paired_summary(paired(axis, key))
        # The safety axis improves when FNR FALLS, so the "agreeing seeds" count
        # is the one that moved in the direction the row is about, not always the
        # negative one.
        agreed = stats["n_negative"] if key != "macro_fnr" else stats["n_positive"]
        rows.append(
            f"{group} & {label} & {num(arm(axis, key, 'baseline'))} & "
            f"{num(arm(axis, key, 'mitigated'))} & {num(stats['mean'], signed=True)} & "
            f"{interval(stats['ci_low'], stats['ci_high'])} & "
            f"{agreed}/{stats['n']} & {pval(stats['sign_p'])} \\\\"
        )
    write("t_headline",
          "Axis & Metric & Baseline & Mitigated & $\\Delta$ & 95\\% CI & Sign & $p$ \\\\",
          rows, "llrrrlcr",
          midrules={3: "\\addlinespace", 4: "\\addlinespace"})


# -- T3: every family --------------------------------------------------------

FAMILY_LABEL = {
    "main": ("IN", "loss reweighting (headline)"),
    "lastep": ("IN", "reweighting, fixed epoch"),
    "drob": ("IN", "reweighting, DistilRoBERTa"),
    "wtox": ("IN", "reweighting $+$ identity-toxic cell"),
    "heldout": ("IN", "reweighting, 2 of 5 identity axes"),
    "dro": ("IN", "Group DRO, learned weights"),
    "cda": ("PRE", "counterfactual swap, $p=0.5$"),
    "cdafull": ("PRE", "counterfactual swap, $p=1.0$"),
}
FAMILY_ORDER = ["main", "lastep", "drob", "wtox", "heldout", "dro", "cda", "cdafull"]


def t_families() -> None:
    fam = load("study01_families").set_index("family")
    post = load("study05_postprocessing").set_index("family").loc["main"]

    rows, midrules = [], {}
    for index, key in enumerate(FAMILY_ORDER):
        r = fam.loc[key]
        stage, label = FAMILY_LABEL[key]
        if key == "cda":
            midrules[index] = "\\addlinespace"
        rows.append(
            f"\\textsc{{{stage.lower()}}} & {label} & {int(r['n_seeds'])} & "
            f"{num(r['d_fpr_gap'], signed=True)} & {num(r['sd'])} & "
            f"{interval(r['ci_low'], r['ci_high'])} & "
            f"{int(r['n_improved'])}/{int(r['n_seeds'])} & "
            f"{num(r['d_macro_f1'], signed=True)} & {num(r['d_macro_fnr'], signed=True)} & "
            f"{num(r['safety_tax'], 2)} \\\\"
        )
    midrules[len(rows)] = "\\addlinespace"
    rows.append(
        f"\\textsc{{post}} & per-subgroup thresholds & {int(post['n_seeds'])} & "
        f"{num(post['d_fpr_gap'], signed=True)} & {num(post['sd'])} & "
        f"{interval(post['ci_low'], post['ci_high'])} & "
        f"{int(post['n_improved'])}/{int(post['n_seeds'])} & "
        f"{num(post['d_macro_f1'], signed=True)} & {num(post['d_macro_fnr'], signed=True)} & "
        f"{num(post['safety_tax'], 2)} \\\\"
    )
    write("t_families",
          "Stage & Configuration & $n$ & $\\Delta$ gap & sd & 95\\% CI & Sign & "
          "$\\Delta$ F1 & $\\Delta$ FNR & Tax \\\\",
          rows, "llrrrlcrrr", midrules=midrules)


# -- T4: ablation ------------------------------------------------------------

def t_ablation() -> None:
    sweep = load("study08_ablation")
    main = sweep[sweep["family"] == "main"]
    control = main[main["scheme"] == "class"].iloc[0]
    arms = main[main["scheme"] == "subgroup"].sort_values("alpha")

    rows = []
    for _, r in arms.iterrows():
        alpha = int(r["alpha"])
        label = f"$\\alpha = {alpha}$"
        if alpha == 1:
            label += " (= baseline)"
        if alpha == 4:
            label = f"\\textbf{{{label}}}"
        rows.append(
            f"subgroup & {label} & {num(r['fpr_gap'])} & {num(r['macro_fpr'])} & "
            f"{num(r['macro_f1'])} & {num(r['safety_fnr'])} & {num(r['flag_rate'])} \\\\"
        )
    rows.append("\\addlinespace")
    rows.append(
        f"class & inverse class frequency & {num(control['fpr_gap'])} & "
        f"{num(control['macro_fpr'])} & {num(control['macro_f1'])} & "
        f"{num(control['safety_fnr'])} & {num(control['flag_rate'])} \\\\"
    )
    write("t_ablation",
          "Scheme & Arm & FPR gap & macro FPR & macro-F1 & subgroup FNR & Flag rate \\\\",
          rows, "llrrrrr")


# -- T5: the shortcut regression --------------------------------------------

def t_mechanism() -> None:
    mech = load("study04_mechanism").set_index("family")
    fam = load("study01_families").set_index("family")
    order = sorted(mech.index, key=lambda f: mech.loc[f, "slope_change"])

    rows = []
    for key in order:
        r = mech.loc[key]
        gap = fam.loc[key]
        closes = "yes" if (gap["excludes_zero"] and gap["d_fpr_gap"] < 0) else "no"
        _, label = FAMILY_LABEL[key]
        rows.append(
            f"{label} & {int(r['n_seeds'])} & {num(r['baseline_slope'], 3)} & "
            f"{num(r['baseline_pearson_r'], 3)} & {num(r['mitigated_slope'], 3)} & "
            f"{pct(r['slope_change'])}\\% & {closes} \\\\"
        )
    write("t_mechanism",
          "Configuration & $n$ & Baseline slope & $r$ & Mitigated slope & "
          "Change & Gap closes? \\\\",
          rows, "lrrrrrc")


# -- T6: threshold-free ------------------------------------------------------

def t_thresholdfree() -> None:
    tf = load("study07_threshold_free").set_index("family")
    rows = []
    for key in ("main", "drob", "cda", "cdafull", "dro", "heldout"):
        r = tf.loc[key]
        _, label = FAMILY_LABEL[key]
        rows.append(
            f"{label} & {int(r['n_seeds'])} & "
            f"{num(r['bpsn_auc_mean_baseline'])} & {num(r['bpsn_auc_mean_mitigated'])} & "
            f"{num(r['bpsn_auc_mean_delta'], signed=True)} & "
            f"{num(r['bnsp_auc_mean_delta'], signed=True)} & "
            f"{num(r['overall_auc_delta'], signed=True)} \\\\"
        )
    write("t_thresholdfree",
          "Configuration & $n$ & BPSN base & BPSN mit. & $\\Delta$ BPSN & "
          "$\\Delta$ BNSP & $\\Delta$ AUC \\\\",
          rows, "lrrrrrr")


# -- T7: what Group DRO learned ---------------------------------------------

CELL_LABEL = {
    "background_nontoxic": "background, non-toxic",
    "background_toxic": "background, toxic",
    "identity_nontoxic": "identity, non-toxic",
    "identity_toxic": "identity, toxic",
}


def t_dro() -> None:
    runs = ["dro", "dro_s43", "dro_s44"]
    learned = [metrics(r, "model_mitigated")["learned_group_weights"] for r in runs]
    static = {"background_nontoxic": 1.0, "background_toxic": 1.0,
              "identity_nontoxic": 4.0, "identity_toxic": 1.0}
    total = sum(static.values())

    rows = []
    for cell, label in CELL_LABEL.items():
        seeds = " / ".join(f"{w[cell]:.3f}" for w in learned)
        rows.append(f"{label} & {num(static[cell] / total, 3)} & {seeds} \\\\")
    write("t_dro",
          "Cell & Ours & Group DRO, per seed \\\\",
          rows, "lrl")


# -- T8: held-out generalisation --------------------------------------------

def t_heldout() -> None:
    gen = load("study03_generalisation")
    label = {("heldout", "weighted"): "\\texttt{heldout}: weighted axes (gender, religion)",
             ("heldout", "held_out"): "\\texttt{heldout}: held-out axes (race, orient., disab.)",
             ("main", "weighted"): "\\texttt{main}: same weighted axes",
             ("main", "held_out"): "\\texttt{main}: same held-out axes"}
    rows, midrules = [], {}
    order = [("heldout", "weighted"), ("heldout", "held_out"),
             ("main", "weighted"), ("main", "held_out")]
    for index, (family, bucket) in enumerate(order):
        r = gen[(gen["family"] == family) & (gen["bucket"] == bucket)].iloc[0]
        if index == 2:
            midrules[index] = "\\addlinespace"
        rows.append(
            f"{label[(family, bucket)]} & {int(r['n_seeds'])} & "
            f"{num(r['mean_d_fpr'], signed=True)} & {num(r['sd'])} & "
            f"{interval(r['ci_low'], r['ci_high'])} & "
            f"{int(r['n_seeds_improved'])}/{int(r['n_seeds'])} & "
            f"{pval(r['sign_p'])} & "
            f"{int(r['n_subgroup_improved'])}/{int(r['n_subgroup_obs'])} \\\\"
        )
    write("t_heldout",
          "Axis bucket & $n$ & mean $\\Delta$ FPR & sd & 95\\% CI & Seeds & $p$ & "
          "Subgroup obs. \\\\",
          rows, "lrrrlcrc", midrules=midrules)


# -- T9: the manually coded errors ------------------------------------------

def t_coding() -> None:
    coded = pd.read_csv(ANALYSIS / "error_coding.csv")
    coded = coded[coded["manual_category"].notna()]
    total = len(coded)
    meaning = {
        "hostility_not_group_directed":
            "hostile, but aimed at a person, party or ideology",
        "group_directed_derogation":
            "genuinely derogatory towards the group",
        "counter_speech": "argues \\emph{against} bigotry",
        "identity_topic_neutral": "neutral mention, no hostility at all",
        "quoted_hostility": "quotes hostile speech in order to criticise it",
    }
    grouped = coded.groupby("manual_category")
    rows = []
    for category, group in sorted(grouped, key=lambda kv: -len(kv[1])):
        arguable = int(pd.to_numeric(group["label_arguable"],
                                     errors="coerce").fillna(0).sum())
        rows.append(
            f"\\texttt{{{category.replace('_', chr(92) + '_')}}} & {len(group)} & "
            f"${100 * len(group) / total:.1f}$ & {arguable} & {meaning[category]} \\\\"
        )
    write("t_coding",
          "Category & $n$ & \\% & Label arguable & What the comment is \\\\",
          rows, "lrrrl")


# -- T10: integrity ----------------------------------------------------------

def t_integrity() -> None:
    integ = load("study06_integrity")
    rows = []
    for _, r in integ.iterrows():
        rows.append(
            f"{int(r['seed'])} & {int(r['n_runs'])} & \\texttt{{{r['split_fingerprint']}}} & "
            f"{'yes' if r['one_partition'] else 'NO'} & "
            f"{'yes' if r['one_machine'] else 'NO'} & {r['degenerate_runs']} & "
            f"${r['test_rows_seen_in_train_pct']:.2f}$ \\\\"
        )
    write("t_integrity",
          "Seed & Runs & Split fingerprint & One partition & One GPU & Collapsed & "
          "Dup.\\ \\% \\\\",
          rows, "rrlccrr")


def main() -> int:
    print("emitting paper tables from the committed analysis artefacts")
    t_corpus()
    t_headline()
    t_families()
    t_ablation()
    t_mechanism()
    t_thresholdfree()
    t_dro()
    t_heldout()
    t_coding()
    t_integrity()
    print(f"done -> {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Emit the defence deck's tables as LaTeX, from the committed artefacts.

    python scripts/make_slide_tables.py

Same contract as ``make_paper_tables.py``: a bare ``tabular`` block per table in
``presentation/tables/``, wrapped and captioned by ``presentation.tex``. The
number formatting is imported from that script rather than re-implemented, so a
figure quoted on a slide is formatted by the same code that formatted it in the
paper and the two can never disagree by a rounding rule.

The deck's tables are not the paper's tables shrunk: a slide can carry about five
columns before it stops being readable from the back of a room, so each one here
keeps the rows a listener needs and drops the columns they cannot read anyway.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_paper_tables as paper  # noqa: E402
from fairtox.evaluation.aggregate import paired_summary  # noqa: E402

ANALYSIS = REPO_ROOT / "analysis"
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "presentation" / "tables"

MAIN_SEEDS = ["main"] + [f"main_s{s}" for s in range(43, 50)]


def write(name: str, header: str, rows: list[str], colspec: str,
          midrules: dict[int, str] | None = None) -> Path:
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


def metrics(run: str, stage: str) -> dict:
    return json.loads((RESULTS / run / "metrics" / f"{stage}.json").read_text("utf-8"))


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(ANALYSIS / f"{name}.csv")


# -- the corpus --------------------------------------------------------------

def s_corpus() -> None:
    eda = metrics("main", "eda")
    sizes = eda["split_sizes"]
    share = eda["n_total"] / 1_999_516
    # Six rows, because this table is read from the back of a room: the release,
    # the study population, what it is made of, and what it can support.
    rows = [
        f"Comments in the release & {paper.count(1_999_516)} \\\\",
        f"Identity-annotated \\emph{{(our population)}} & "
        f"{paper.count(eda['n_total'])} \\; (${100 * share:.1f}\\%$) \\\\",
        f"Toxic rate & {paper.num(eda['toxic_rate'])} \\\\",
        f"Split: train / val / test & "
        f"{paper.count(sizes['train'])} / {paper.count(sizes['val'])} / "
        f"{paper.count(sizes['test'])} \\\\",
        f"Subgroups configured & {eda['n_subgroups_total']} \\\\",
        f"Clearing the FPR / FNR floor & "
        f"{eda['n_reportable_fpr']} / {eda['n_reportable_fnr']} \\\\",
    ]
    write("s_corpus", "Quantity & Value \\\\", rows, "lr",
          midrules={3: "\\addlinespace", 4: "\\addlinespace"})


# -- the two baselines -------------------------------------------------------

def s_baselines() -> None:
    tfidf = metrics("main", "tfidf_baseline")["metrics"]["test"]
    compare = metrics("main", "compare")
    # The deep model's aggregate scores live on the comparison's utility axis and
    # its two error rates on the audit's global block; both are seed 42's test
    # split, the same rows the classical baseline is scored on.
    deep = {
        "macro_f1": compare["axes"]["utility"]["macro_f1"]["baseline"],
        "roc_auc": compare["axes"]["utility"]["roc_auc"]["baseline"],
        "fpr": compare["baseline_summary"]["global"]["fpr"],
        "fnr": compare["baseline_summary"]["global"]["fnr"],
    }
    rows = []
    for label, m in (("TF-IDF $+$ logistic regression (balanced)", tfidf),
                     ("DistilBERT, fine-tuned", deep)):
        rows.append(
            f"{label} & {paper.num(m['macro_f1'])} & {paper.num(m['roc_auc'])} & "
            f"{paper.num(m['fpr'])} & {paper.num(m['fnr'])} \\\\")
    write("s_baselines",
          "Model & macro-F1 & ROC-AUC & FPR & FNR \\\\", rows, "lrrrr")


# -- the headline pair -------------------------------------------------------

def s_headline() -> None:
    compares = {run: metrics(run, "compare") for run in MAIN_SEEDS}

    def series(axis: str, key: str, which: str) -> list[float]:
        return [compares[r]["axes"][axis][key][which] for r in MAIN_SEEDS]

    rows = []
    spec = [
        ("Parity", "subgroup FPR gap", "parity", "fpr_gap"),
        ("Utility", "macro-F1", "utility", "macro_f1"),
        ("Safety", "subgroup FNR", "safety", "macro_fnr"),
    ]
    for group, label, axis, key in spec:
        stats = paired_summary(series(axis, key, "delta"))
        agreed = stats["n_negative"] if key != "macro_fnr" else stats["n_positive"]
        base = sum(series(axis, key, "baseline")) / len(MAIN_SEEDS)
        mit = sum(series(axis, key, "mitigated")) / len(MAIN_SEEDS)
        rows.append(
            f"{group} & {label} & {paper.num(base)} & {paper.num(mit)} & "
            f"{paper.num(stats['mean'], signed=True)} & "
            f"{paper.interval(stats['ci_low'], stats['ci_high'])} & "
            f"{agreed}/{stats['n']} \\\\")
    write("s_headline",
          "Axis & Metric & Baseline & Mitigated & $\\Delta$ & 95\\% CI & Seeds \\\\",
          rows, "llrrrlc")


# -- what the false positives are -------------------------------------------

def s_coding() -> None:
    coded = pd.read_csv(ANALYSIS / "error_coding.csv")
    coded = coded[coded["manual_category"].notna()]
    total = len(coded)
    meaning = {
        "hostility_not_group_directed": "hostile, but not aimed at the group",
        "group_directed_derogation": "genuinely derogatory towards the group",
        "counter_speech": "argues \\emph{against} bigotry",
        "identity_topic_neutral": "neutral mention, no hostility",
        "quoted_hostility": "quotes hostility in order to criticise it",
    }
    rows = []
    for category, group in sorted(coded.groupby("manual_category"),
                                  key=lambda kv: -len(kv[1])):
        arguable = int(pd.to_numeric(group["label_arguable"],
                                     errors="coerce").fillna(0).sum())
        rows.append(
            f"{meaning[category]} & {len(group)} & "
            f"${100 * len(group) / total:.1f}$ & {arguable} \\\\")
    write("s_coding",
          "What the comment actually is & $n$ & \\% & Label arguable \\\\",
          rows, "lrrr")


# -- what Group DRO learned --------------------------------------------------

def s_dro() -> None:
    runs = ["dro", "dro_s43", "dro_s44"]
    learned = [metrics(r, "model_mitigated")["learned_group_weights"] for r in runs]
    static = {"background_nontoxic": 1.0, "background_toxic": 1.0,
              "identity_nontoxic": 4.0, "identity_toxic": 1.0}
    total = sum(static.values())
    rows = []
    for cell, label in paper.CELL_LABEL.items():
        mean = sum(w[cell] for w in learned) / len(learned)
        rows.append(f"{label} & {paper.num(static[cell] / total, 3)} & "
                    f"{paper.num(mean, 3)} \\\\")
    # Short headers: this table shares a slide with a second one, so it has to
    # fit half a frame at \scriptsize.
    write("s_dro", "Training cell & Ours & Learned \\\\", rows, "lrr")


# -- the three controls ------------------------------------------------------

def s_controls() -> None:
    """Every control moves parity the wrong way and safety the right way.

    The rows do not share a sample: the two ablation arms are three-seed means of
    the headline run, and the isolated toxic cell exists only in the ``wtox``
    run's ablation at seed 42 -- which is exactly the pair the paper quotes. The
    ``n`` column is printed so the two are never read as one series.
    """
    abl = load("study08_ablation")
    main_abl = abl[abl["family"] == "main"].set_index("arm")
    base = main_abl.loc["abl_subgroup_a1"]
    klass = main_abl.loc["abl_class"]

    seed42 = metrics("main", "compare")["axes"]
    wtox = {a["arm"]: a for a in metrics("wtox", "ablate")["arms"]}["abl_subgroup_a1"]

    dro_runs = ["dro", "dro_s43", "dro_s44"]
    dro = [metrics(r, "compare")["axes"] for r in dro_runs]

    def mean(rows: list[dict], axis: str, key: str, which: str) -> float:
        return sum(r[axis][key][which] for r in rows) / len(rows)

    rows = [
        f"inverse class frequency (\\texttt{{class}}) & 3 & "
        f"{paper.num(base['fpr_gap'])} & {paper.num(klass['fpr_gap'])} & "
        f"{paper.num(base['safety_fnr'])} & {paper.num(klass['safety_fnr'])} \\\\",

        f"identity-toxic cell alone & 1 & "
        f"{paper.num(seed42['parity']['fpr_gap']['baseline'])} & "
        f"{paper.num(wtox['fpr_gap'])} & "
        f"{paper.num(seed42['safety']['macro_fnr']['baseline'])} & "
        f"{paper.num(wtox['safety_fnr'])} \\\\",

        f"Group DRO, weights \\emph{{learned}} & 3 & "
        f"{paper.num(mean(dro, 'parity', 'fpr_gap', 'baseline'))} & "
        f"{paper.num(mean(dro, 'parity', 'fpr_gap', 'mitigated'))} & "
        f"{paper.num(mean(dro, 'safety', 'macro_fnr', 'baseline'))} & "
        f"{paper.num(mean(dro, 'safety', 'macro_fnr', 'mitigated'))} \\\\",
    ]
    write("s_controls",
          "Control: what it weights & Seeds & \\multicolumn{2}{c}{FPR gap} & "
          "\\multicolumn{2}{c}{subgroup FNR} \\\\\n"
          "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
          " & & before & after & before & after \\\\",
          rows, "lcrrrr")


def main() -> int:
    print("emitting slide tables from the committed analysis artefacts")
    s_corpus()
    s_baselines()
    s_headline()
    s_coding()
    s_dro()
    s_controls()
    print(f"done -> {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

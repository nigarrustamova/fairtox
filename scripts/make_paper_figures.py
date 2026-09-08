#!/usr/bin/env python
"""Render the figures the paper uses, at IEEE column width.

    python scripts/make_paper_figures.py

The figures under ``results/`` are diagnostics: they carry an in-image title, a
legend sized for a screen, and a shape chosen to be readable on its own. A
two-column paper wants the opposite -- no title (the caption carries it), 7 pt
type, and exactly ``\\columnwidth`` so nothing is scaled after the fact, because
scaling a figure in LaTeX scales its fonts out of agreement with the body text.

So these are drawn separately rather than reused, from the same committed
artefacts: ``analysis/study0*.csv`` for the cross-family numbers and
``results/<run>/metrics/*.json`` for the per-subgroup ones. Nothing here reads a
model, a prediction file or a notebook, so every figure in the paper regenerates
from the repository in seconds.

Output goes to ``report/figures/`` as PDF -- vector, so the type stays sharp at
any zoom and matches the document's own rendering.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ANALYSIS = REPO_ROOT / "analysis"
RESULTS = REPO_ROOT / "results"
OUT = REPO_ROOT / "report" / "figures"

# IEEEtran: a single column is 3.5 in, the full text block 7.16 in.
COL = 3.45
WIDE = 7.16

BASE = "#3B5BA5"      # baseline
MIT = "#2F8F5B"       # mitigated
WARN = "#C0562A"      # safety / adverse
GREY = "#6B7280"

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "pdf.fonttype": 42,
})


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path


def _audit(run: str, arm: str) -> pd.DataFrame:
    payload = json.loads((RESULTS / run / "metrics" / f"audit_{arm}.json").read_text("utf-8"))
    return pd.DataFrame(payload["subgroups"])


def _pretty(name: str) -> str:
    """Subgroup column name as a label a reader can scan."""
    return name.replace("_or_", "/").replace("_", " ")


# -- Fig: the eight families, side by side -----------------------------------

def fig_families() -> None:
    """Every mitigation, on one axis, with the interval that decides it."""
    fam = pd.read_csv(ANALYSIS / "study01_families.csv").set_index("family")
    post = pd.read_csv(ANALYSIS / "study05_postprocessing.csv").set_index("family")

    order = [
        ("drob", "DistilRoBERTa backbone"),
        ("lastep", "fixed-epoch control"),
        ("main", "headline"),
        ("wtox", "+ identity-toxic cell"),
        ("heldout", "weighting sees 2 of 5 axes"),
        ("dro", "Group DRO (learned weights)"),
    ]
    rows = []
    for key, label in order:
        r = fam.loc[key]
        rows.append(("IN", label, r["d_fpr_gap"], r["ci_low"], r["ci_high"],
                     int(r["n_seeds"]), bool(r["excludes_zero"])))
    for key, label in (("cda", "counterfactual swap, p = 0.5"),
                       ("cdafull", "counterfactual swap, p = 1.0")):
        r = fam.loc[key]
        rows.append(("PRE", label, r["d_fpr_gap"], r["ci_low"], r["ci_high"],
                     int(r["n_seeds"]), bool(r["excludes_zero"])))
    p = post.loc["main"]
    rows.append(("POST", "per-subgroup thresholds", p["d_fpr_gap"], p["ci_low"],
                 p["ci_high"], int(p["n_seeds"]), bool(p["ci_high"] < 0)))

    fig, ax = plt.subplots(figsize=(COL, 2.75))
    ys = np.arange(len(rows))[::-1]
    for y, (_stage, _label, mean, lo, hi, _n, excl) in zip(ys, rows, strict=True):
        colour = MIT if (mean < 0 and excl) else (WARN if (mean > 0 and excl) else GREY)
        ax.plot([lo, hi], [y, y], color=colour, lw=1.1, solid_capstyle="butt")
        for end_x in (lo, hi):
            ax.plot([end_x, end_x], [y - 0.17, y + 0.17], color=colour, lw=1.1)
        # A filled marker means the interval clears zero; hollow means it does not.
        ax.plot([mean], [y], "o", color=colour, ms=3.6,
                mfc=colour if excl else "white", mew=1.0)

    ax.axvline(0, color="black", lw=0.7, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{st}  {lab}  ($n{{=}}{n}$)"
                        for st, lab, _m, _l, _h, n, _e in rows])
    ax.set_xlabel(r"$\Delta$ subgroup FPR gap (mitigated $-$ baseline)")
    ax.set_xlim(-0.16, 0.38)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _save(fig, "fig_families")


# -- Fig: per-subgroup rates -------------------------------------------------

def _paired(metric: str, flag: str, xlabel: str, name: str, height: float) -> None:
    base, mit = _audit("main", "baseline"), _audit("main", "mitigated")
    data = base[base[flag] & base[metric].notna()].sort_values(metric)
    names = data["subgroup"].tolist()
    aligned = mit.set_index("subgroup").reindex(names).reset_index()

    def bars(frame):
        centre = pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)
        lo = np.nan_to_num(centre - pd.to_numeric(frame[f"{metric}_ci_low"],
                                                  errors="coerce").to_numpy(float))
        hi = np.nan_to_num(pd.to_numeric(frame[f"{metric}_ci_high"],
                                         errors="coerce").to_numpy(float) - centre)
        return centre, np.vstack([np.clip(lo, 0, None), np.clip(hi, 0, None)])

    y = np.arange(len(names))
    h = 0.38
    fig, ax = plt.subplots(figsize=(COL, height))
    ekw = {"elinewidth": 0.6, "ecolor": "#333333", "capsize": 1.2}
    cb, eb = bars(data)
    cm, em = bars(aligned)
    ax.barh(y + h / 2, cb, height=h, color=BASE, label="baseline", xerr=eb, error_kw=ekw)
    ax.barh(y - h / 2, cm, height=h, color=MIT, label="mitigated", xerr=em, error_kw=ekw)

    ax.set_yticks(y)
    ax.set_yticklabels([_pretty(n) for n in names])
    ax.set_xlabel(xlabel)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _save(fig, name)


def fig_fpr_subgroup() -> None:
    _paired("fpr", "reportable_fpr",
            "false-positive rate (benign comments flagged)", "fig_fpr_subgroup", 3.15)


def fig_fnr_subgroup() -> None:
    _paired("fnr", "reportable_fnr",
            "false-negative rate (genuine abuse missed)", "fig_fnr_subgroup", 2.25)


# -- Fig: the shortcut ------------------------------------------------------

def fig_mechanism() -> None:
    """Subgroup FPR against the subgroup's toxic rate, both arms, seed 42."""
    mech = json.loads((RESULTS / "main" / "metrics" / "mechanism_analysis.json")
                      .read_text("utf-8"))
    fits = mech["toxic_rate_fit"]

    def pts(arm):
        rows = [r for r in _audit("main", arm).to_dict("records")
                if r.get("reportable_fpr") and pd.notna(r.get("fpr"))]
        x = np.array([r["n_toxic"] / r["n"] for r in rows], float)
        y = np.array([r["fpr"] for r in rows], float)
        return x, y, [r["subgroup"] for r in rows]

    bx, by, names = pts("baseline")
    mx, my, _ = pts("mitigated")

    fig, ax = plt.subplots(figsize=(COL, 2.5))
    for x, fit, colour, style in ((bx, fits["baseline"], BASE, "-"),
                                  (mx, fits["mitigated"], MIT, "--")):
        span = np.array([x.min(), x.max()])
        ax.plot(span, fit["slope"] * span + fit["intercept"], style, color=colour, lw=1.2)
    ax.scatter(bx, by, s=13, color=BASE, zorder=3,
               label=f"baseline (slope {fits['baseline']['slope']:.3f})")
    ax.scatter(mx, my, s=13, facecolors="none", edgecolors=MIT, linewidths=0.9, zorder=3,
               label=f"mitigated (slope {fits['mitigated']['slope']:.3f})")

    mid = (bx.min() + bx.max()) / 2
    for i in {int(np.argmin(bx)), int(np.argmax(bx)), int(np.argmax(by))}:
        left = bx[i] > mid
        ax.annotate(_pretty(names[i]), (bx[i], by[i]), textcoords="offset points",
                    xytext=(-4 if left else 4, 3), fontsize=6, color="#444444",
                    ha="right" if left else "left")

    ax.set_xlabel("subgroup toxic rate on the test split")
    ax.set_ylabel("subgroup false-positive rate")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    _save(fig, "fig_mechanism")


# -- Fig: the alpha sweep ---------------------------------------------------

def fig_alpha() -> None:
    """Parity, utility and safety as the weighting strength rises (3 seeds)."""
    sweep = pd.read_csv(ANALYSIS / "study08_ablation.csv")
    arm = sweep[(sweep["family"] == "main") & (sweep["scheme"] == "subgroup")]
    arm = arm.sort_values("alpha")
    a = arm["alpha"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(COL, 2.35))
    ax.plot(a, arm["fpr_gap"], "o-", color=MIT, ms=3.2, label="FPR gap")
    ax.plot(a, arm["macro_fpr"], "s--", color=MIT, ms=3.0, alpha=0.55, label="macro FPR")
    ax.set_xlabel(r"weighting strength $\alpha$")
    ax.set_ylabel("parity (lower is fairer)", color=MIT)
    ax.tick_params(axis="y", labelcolor=MIT)
    ax.set_xticks(a)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)

    tw = ax.twinx()
    tw.plot(a, arm["safety_fnr"], "^:", color=WARN, ms=3.4, label="subgroup FNR")
    tw.plot(a, arm["macro_f1"], "v-.", color=BASE, ms=3.0, label="macro-F1")
    tw.set_ylabel("safety and utility (rates)", color="#444444")
    tw.spines["top"].set_visible(False)
    ax.spines["top"].set_visible(False)

    # The pre-registered default is marked by the rule alone and named in the
    # caption. A text label inside the axes collides with one of the four series
    # wherever it is placed, and the caption is where IEEE readers look anyway.
    ax.axvline(4.0, color=GREY, lw=0.7, ls=(0, (1, 2)), zorder=0)

    # Legend below the axes: with four series on two y-scales there is no
    # interior region that stays clear across the whole sweep.
    handles = ax.get_legend_handles_labels()[0] + tw.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + tw.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.28),
              frameon=False, ncol=2, columnspacing=1.2, handlelength=1.8)
    _save(fig, "fig_alpha")


# -- Fig: the shortcut slope across every family ----------------------------

def fig_slope() -> None:
    """How far each mitigation moves the shortcut, and whether the gap follows."""
    mech = pd.read_csv(ANALYSIS / "study04_mechanism.csv")
    fam = pd.read_csv(ANALYSIS / "study01_families.csv").set_index("family")

    labels = {"drob": "DistilRoBERTa", "lastep": "fixed epoch", "main": "reweighting",
              "cdafull": "swap $p{=}1.0$", "wtox": "+ toxic cell", "cda": "swap $p{=}0.5$",
              "heldout": "2 of 5 axes", "dro": "Group DRO"}
    mech = mech[mech["family"].isin(labels)].sort_values("slope_change")

    closes = {f: bool(fam.loc[f, "excludes_zero"] and fam.loc[f, "d_fpr_gap"] < 0)
              for f in mech["family"]}

    fig, ax = plt.subplots(figsize=(COL, 2.2))
    y = np.arange(len(mech))[::-1]
    values = 100 * mech["slope_change"].to_numpy(float)
    colours = [MIT if closes[f] else WARN if v > 0 else GREY
               for f, v in zip(mech["family"], values, strict=True)]
    ax.barh(y, values, color=colours, height=0.62)

    # Values are written just past the end of their own bar, never on it: the
    # bars are dark enough that dark text on them is unreadable and white text
    # would disappear on the short ones.
    for yi, v in zip(y, values, strict=True):
        ax.text(v + (6 if v >= 0 else -6), yi, f"{v:+.0f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=6, color="#333333")

    ax.axvline(0, color="black", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([labels[f] for f in mech["family"]])
    ax.set_xlabel("change in the shortcut slope (%)")
    ax.set_xlim(-118, 290)
    ax.grid(axis="x", alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _save(fig, "fig_slope")


def main() -> int:
    print("rendering paper figures at IEEE column width")
    fig_families()
    fig_fpr_subgroup()
    fig_fnr_subgroup()
    fig_mechanism()
    fig_alpha()
    fig_slope()
    print(f"done -> {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Render the two figures the defence deck needs and the repository does not have.

    python scripts/make_slide_figures.py

Almost every figure in the deck comes straight out of ``results/<run>/figures/``:
those are already screen-sized with a title baked in, which is exactly what a
slide wants and exactly what a two-column paper does not, so the deck reads them
in place rather than re-rendering them. ``presentation.tex`` puts that directory
on its ``\\graphicspath``.

Three figures have no usable equivalent there. Two are built across runs rather
than inside one -- the cross-family comparison of every mitigation, and the
counterfactual-augmentation dose response -- and the third, the alpha sweep, is
drawn here over the three seeds the sweep actually ran rather than over seed 42
alone. All three come from the committed ``analysis/study0*.csv`` aggregates, so
they regenerate in seconds with no GPU, no model and no prediction file.

Output goes to ``presentation/figures/`` as PDF -- vector, so a projector cannot
soften it.
"""

from __future__ import annotations

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
OUT = REPO_ROOT / "presentation" / "figures"

# The body of a 16:9 beamer frame, once the title and the footline are taken
# out, is about 5.9 in by 2.9 in. Nothing here is drawn wider than that, so no
# figure is ever scaled up in LaTeX.
FULL_W = 5.75

BASE = "#3B5BA5"      # baseline arm
MIT = "#2F8F5B"       # mitigated arm
WARN = "#C0562A"      # safety cost / adverse movement
GREY = "#6B7280"
INK = "#222222"

plt.rcParams.update({
    "font.size": 9.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.6,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
})


def _save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")
    return path


def _despine(ax, sides=("top", "right")) -> None:
    for side in sides:
        ax.spines[side].set_visible(False)


# -- every configuration on one axis ----------------------------------------

def s_families() -> None:
    """The whole study in one picture: nine arms, one axis, one interval each."""
    fam = pd.read_csv(ANALYSIS / "study01_families.csv").set_index("family")
    post = pd.read_csv(ANALYSIS / "study05_postprocessing.csv").set_index("family")

    order = [
        ("IN", "drob", "reweighting, DistilRoBERTa"),
        ("IN", "lastep", "reweighting, fixed epoch"),
        ("IN", "main", "reweighting  (headline)"),
        ("IN", "wtox", "reweighting + identity-toxic cell"),
        ("IN", "heldout", "reweighting, 2 of 5 axes"),
        ("IN", "dro", "Group DRO  (learned weights)"),
        ("PRE", "cda", "counterfactual swap, $p = 0.5$"),
        ("PRE", "cdafull", "counterfactual swap, $p = 1.0$"),
    ]
    rows = []
    for stage, key, label in order:
        r = fam.loc[key]
        rows.append((stage, label, r["d_fpr_gap"], r["ci_low"], r["ci_high"],
                     int(r["n_seeds"]), bool(r["excludes_zero"])))
    p = post.loc["main"]
    rows.append(("POST", "per-subgroup thresholds", p["d_fpr_gap"], p["ci_low"],
                 p["ci_high"], int(p["n_seeds"]), bool(p["ci_high"] < 0)))

    fig, ax = plt.subplots(figsize=(FULL_W, 3.0))
    ys = np.arange(len(rows))[::-1]
    for y, (_stage, _label, mean, lo, hi, _n, excl) in zip(ys, rows, strict=True):
        colour = MIT if (mean < 0 and excl) else (WARN if (mean > 0 and excl) else GREY)
        ax.plot([lo, hi], [y, y], color=colour, lw=1.6, solid_capstyle="butt")
        for end_x in (lo, hi):
            ax.plot([end_x, end_x], [y - 0.18, y + 0.18], color=colour, lw=1.6)
        # A filled marker means the interval clears zero; hollow means it does not.
        ax.plot([mean], [y], "o", color=colour, ms=5.5,
                mfc=colour if excl else "white", mew=1.4)

    ax.axvline(0, color="black", lw=0.9, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{st}   {lab}   ($n{{=}}{n}$)"
                        for st, lab, _m, _l, _h, n, _e in rows])
    ax.set_xlabel(r"$\Delta$ subgroup FPR gap   (mitigated $-$ baseline)")
    ax.set_xlim(-0.17, 0.39)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    _despine(ax, ("top", "right", "left"))
    ax.tick_params(axis="y", length=0)
    ax.annotate("fairer", xy=(-0.115, -0.55), fontsize=9, color=MIT, ha="center")
    ax.annotate("less fair", xy=(0.30, -0.55), fontsize=9, color=WARN, ha="center")
    _save(fig, "s_families")


# -- counterfactual augmentation, two doses ---------------------------------

def s_dose() -> None:
    """The dose experiment: the shortcut moves, the gap does not."""
    dose = pd.read_csv(ANALYSIS / "study02_dose_response.csv").sort_values("probability")
    x = np.arange(len(dose))
    gap = dose["d_fpr_gap"].to_numpy(float)
    slope = 100 * dose["slope_change"].fillna(0.0).to_numpy(float)

    fig, ax = plt.subplots(figsize=(FULL_W, 2.75))
    colours = [GREY if abs(v) < 1e-9 else (WARN if v > 0 else MIT) for v in gap]
    ax.bar(x, gap, width=0.42, color=colours, label=r"$\Delta$ FPR gap")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"$p = {p:g}$" for p in dose["probability"]])
    ax.set_xlabel("share of identity mentions randomised in training")
    ax.set_ylabel(r"$\Delta$ FPR gap", color=INK)
    ax.set_ylim(min(gap.min(), 0) * 1.9 - 0.004, max(gap.max(), 0) * 1.9 + 0.004)
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    for xi, v in zip(x, gap, strict=True):
        if abs(v) < 1e-9:
            continue
        ax.annotate(f"{v:+.4f}", (xi, v), textcoords="offset points",
                    xytext=(0, 6 if v > 0 else -13), ha="center", fontsize=9)

    tw = ax.twinx()
    tw.plot(x, slope, "o--", color=BASE, ms=5, label="shortcut slope")
    tw.set_ylabel("change in shortcut slope (%)", color=BASE)
    tw.tick_params(axis="y", labelcolor=BASE)
    tw.set_ylim(-46, 14)
    # A slope label goes on the side of its marker the bar is not on, so it can
    # never land inside a bar: the p=1 marker sits exactly on top of the tallest
    # one.
    for xi, v, bar in zip(x, slope, gap, strict=True):
        if abs(v) < 1e-9:
            continue
        above = bar <= 0
        tw.annotate(f"{v:.1f}%", (xi, v), textcoords="offset points",
                    xytext=(0, 8 if above else -15), ha="center", fontsize=9,
                    color=BASE)
    _despine(ax, ("top",))
    _despine(tw, ("top",))

    handles = ax.get_legend_handles_labels()[0] + tw.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + tw.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="lower left", frameon=False, ncol=1)
    _save(fig, "s_dose")


# -- the weighting strength --------------------------------------------------

def s_alpha() -> None:
    """The alpha sweep, over the three seeds that ran it.

    ``results/main/figures/fig09_alpha_tradeoff.png`` shows the same sweep, but
    for seed 42 alone and with its legend sitting on top of one of the series.
    The deck quotes the three-seed means that the paper's ablation table quotes,
    so this is drawn from ``analysis/study08_ablation.csv`` instead.
    """
    sweep = pd.read_csv(ANALYSIS / "study08_ablation.csv")
    arm = sweep[(sweep["family"] == "main") & (sweep["scheme"] == "subgroup")]
    arm = arm.sort_values("alpha")
    a = arm["alpha"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(FULL_W, 2.7))
    ax.plot(a, arm["fpr_gap"], "o-", color=MIT, ms=5, label="FPR gap")
    ax.plot(a, arm["macro_fpr"], "s--", color=MIT, ms=4.5, alpha=0.55,
            label="macro FPR")
    ax.set_xlabel(r"weighting strength $\alpha$")
    ax.set_ylabel("parity (lower is fairer)", color=MIT)
    ax.tick_params(axis="y", labelcolor=MIT)
    ax.set_xticks(a)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)

    tw = ax.twinx()
    tw.plot(a, arm["safety_fnr"], "^:", color=WARN, ms=5, label="subgroup FNR")
    tw.plot(a, arm["macro_f1"], "v-.", color=BASE, ms=4.5, label="macro-F1")
    tw.set_ylabel("safety and utility (rates)")
    ax.axvline(4.0, color=GREY, lw=0.9, ls=(0, (1, 2)), zorder=0)
    _despine(ax, ("top",))
    _despine(tw, ("top",))

    # Four series on two scales leave no interior region clear across the sweep,
    # so the legend goes under the axes rather than on top of a line.
    handles = ax.get_legend_handles_labels()[0] + tw.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + tw.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.30),
              frameon=False, ncol=4, columnspacing=1.4, handlelength=1.9)
    _save(fig, "s_alpha")


def main() -> int:
    print("rendering the cross-run figures the deck needs")
    s_families()
    s_dose()
    s_alpha()
    print(f"done -> {OUT.relative_to(REPO_ROOT)}")
    print("every other figure in the deck is read from results/<run>/figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

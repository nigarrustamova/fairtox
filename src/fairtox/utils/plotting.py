"""Figure generation -- only the figures that carry an argument.

The FPR and FNR panels filter on **different** reportable flags: FPR is estimated
on a subgroup's non-toxic rows and FNR on its toxic ones, and those counts differ
by an order of magnitude here. Filtering both on the FPR flag would put an FNR
computed from a dozen comments on the safety figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Non-interactive backend: figures are written on a headless workstation.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .logging_utils import get_logger  # noqa: E402

logger = get_logger(__name__)

BASELINE_COLOR = "#4C6EF5"
MITIGATED_COLOR = "#2F9E44"
WARN_COLOR = "#E8590C"
GRID_KW = {"alpha": 0.25, "linewidth": 0.6}


def _finish(fig: plt.Figure, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path.name)
    return path


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Column as float with None/NA as NaN.

    Rates are None when a slice has no rows on the relevant side of the
    confusion matrix, which makes the column object-dtype; matplotlib cannot
    plot that, so it is coerced once here rather than guarded at each call site.
    """
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def fig_subgroup_support(table: pd.DataFrame, path: Path, min_support: int = 100) -> Path:
    """Non-toxic and toxic support per subgroup, with the reporting floor marked."""
    data = table.sort_values("n", ascending=True)
    positions = np.arange(len(data))
    height = 0.38

    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.34 * len(data))))
    ax.barh(positions + height / 2, _numeric(data, "n_nontoxic"), height=height,
            color=BASELINE_COLOR, label="non-toxic (drives FPR)")
    ax.barh(positions - height / 2, _numeric(data, "n_toxic"), height=height,
            color=WARN_COLOR, label="toxic (drives FNR)")
    ax.axvline(min_support, color="black", linestyle="--", linewidth=1)
    # Above the axes, not inside them: at the top of the data the annotation sat
    # on the longest bar, and the largest subgroup is the one a reader checks first.
    ax.annotate(f"reporting floor (N={min_support})", xy=(min_support, 1.0),
                xycoords=("data", "axes fraction"), xytext=(4, 4),
                textcoords="offset points", fontsize=8, va="bottom")

    ax.set_yticks(positions)
    ax.set_yticklabels(data["subgroup"].tolist(), fontsize=8)
    ax.set_xscale("symlog")
    ax.set_xlabel("comments in the test split")
    # `pad` lifts the title clear of the floor annotation, which sits just above
    # the top spine. Without it the two overlap on every run -- the annotation is
    # anchored at x = min_support, which is near the middle of a log axis.
    ax.set_title("Subgroup support", pad=16)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.grid(axis="x", **GRID_KW)
    ax.set_axisbelow(True)
    return _finish(fig, path)


def _paired_rate_plot(
    baseline: pd.DataFrame,
    mitigated: pd.DataFrame | None,
    metric: str,
    reportable_flag: str,
    title: str,
    xlabel: str,
    path: Path,
) -> Path:
    """Shared renderer for the FPR and FNR subgroup comparisons."""
    data = baseline[baseline[reportable_flag]].copy()
    data = data[pd.to_numeric(data[metric], errors="coerce").notna()]
    if data.empty:
        logger.warning("no subgroup clears the floor for %s; skipping %s", metric, Path(path).name)
        return Path(path)

    data = data.sort_values(metric, ascending=True)
    names = data["subgroup"].tolist()
    positions = np.arange(len(names))
    height = 0.38 if mitigated is not None else 0.62
    low_col, high_col = f"{metric}_ci_low", f"{metric}_ci_high"

    def error_bars(frame: pd.DataFrame) -> np.ndarray | None:
        if low_col not in frame.columns or high_col not in frame.columns:
            return None
        centre = _numeric(frame, metric)
        low = np.nan_to_num(centre - _numeric(frame, low_col), nan=0.0)
        high = np.nan_to_num(_numeric(frame, high_col) - centre, nan=0.0)
        return np.vstack([np.clip(low, 0, None), np.clip(high, 0, None)])

    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.34 * len(names))))
    error_kw = {"elinewidth": 0.8, "ecolor": "#333333", "capsize": 2}

    offset = height / 2 if mitigated is not None else 0.0
    ax.barh(positions + offset, _numeric(data, metric), height=height,
            color=BASELINE_COLOR, label="baseline", xerr=error_bars(data), error_kw=error_kw)

    if mitigated is not None:
        aligned = mitigated.set_index("subgroup").reindex(names).reset_index()
        ax.barh(positions - height / 2, _numeric(aligned, metric), height=height,
                color=MITIGATED_COLOR, label="mitigated", xerr=error_bars(aligned),
                error_kw=error_kw)
        ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax.set_yticks(positions)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", **GRID_KW)
    ax.set_axisbelow(True)
    return _finish(fig, path)


def fig_fpr_by_subgroup(baseline: pd.DataFrame, mitigated: pd.DataFrame | None, path: Path) -> Path:
    return _paired_rate_plot(
        baseline, mitigated, "fpr", "reportable_fpr",
        "False-positive rate by subgroup (95% CI)",
        "FPR - benign comments wrongly flagged", path,
    )


def fig_fnr_by_subgroup(baseline: pd.DataFrame, mitigated: pd.DataFrame | None, path: Path) -> Path:
    return _paired_rate_plot(
        baseline, mitigated, "fnr", "reportable_fnr",
        "False-negative rate by subgroup (safety check, 95% CI)",
        "FNR - genuine abuse missed", path,
    )


def fig_alpha_tradeoff(sweep: pd.DataFrame, path: Path) -> Path:
    """Utility, parity and safety together as the weighting strength rises."""
    data = sweep.sort_values("alpha")
    alphas = _numeric(data, "alpha")

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(alphas, _numeric(data, "fpr_gap"), marker="o", color=MITIGATED_COLOR, label="FPR gap")
    ax.set_xlabel(r"weighting strength $\alpha$")
    ax.set_ylabel("FPR gap (lower is fairer)", color=MITIGATED_COLOR)
    ax.tick_params(axis="y", labelcolor=MITIGATED_COLOR)
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)

    twin = ax.twinx()
    twin.plot(alphas, _numeric(data, "macro_f1"), marker="s", linestyle="--",
              color=BASELINE_COLOR, label="macro-F1")
    # Both series on this axis are rates in [0, 1], which is why they can share
    # it; the label says so, because "macro-F1 / macro FNR" alone reads like a
    # ratio of the two.
    twin.set_ylabel("macro-F1 and FNR (rates)", color="#444444")
    # `safety_fnr` is the subgroup FNR where it is defined and the global FNR
    # otherwise. Plotting `macro_fnr` alone left the safety line off the figure
    # entirely on any run where no subgroup cleared the FNR floor -- which is
    # exactly the run where the largest alpha looks best and is not.
    safety_column = "safety_fnr" if "safety_fnr" in data.columns else "macro_fnr"
    if safety_column in data.columns:
        twin.plot(alphas, _numeric(data, safety_column), marker="^", linestyle=":",
                  color=WARN_COLOR, label="FNR (safety)")

    # Collapsed arms are marked on the figure, not silently plotted as the best
    # result: their parity numbers are artefacts of a dead model.
    if "degenerate" in data.columns:
        for alpha, flag in zip(alphas, data["degenerate"].tolist(), strict=False):
            if bool(flag):
                ax.axvline(alpha, color=WARN_COLOR, linestyle="-.", linewidth=1, alpha=0.7)
                ax.annotate("collapsed", xy=(alpha, ax.get_ylim()[1]), rotation=90,
                            fontsize=7, color=WARN_COLOR, va="top", ha="right")

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, loc="best", frameon=False, fontsize=8)
    ax.set_title("Fairness-utility-safety trade-off")
    return _finish(fig, path)


def fig_mechanism(
    baseline_rows: list[dict],
    mitigated_rows: list[dict],
    baseline_fit: dict,
    mitigated_fit: dict,
    path: Path,
) -> Path:
    """Subgroup FPR against subgroup toxic rate, with both fitted lines.

    This figure carries the paper's mechanism claim, so each fitted line is drawn
    only across the range its own data covers. A line extended past the leftmost
    and rightmost subgroup would invite the reader to take a prediction for a
    measurement, at exactly the end of the axis where we have no subgroups.
    """

    def points(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        usable = [r for r in rows if r.get("reportable_fpr") and r.get("fpr") is not None]
        x = np.array([r["n_toxic"] / r["n"] for r in usable], dtype=float)
        y = np.array([r["fpr"] for r in usable], dtype=float)
        return x, y, [r["subgroup"] for r in usable]

    bx, by, names = points(baseline_rows)
    mx, my, _ = points(mitigated_rows)
    if len(bx) == 0:
        logger.warning("no reportable subgroups; skipping %s", Path(path).name)
        return Path(path)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    def fitted_line(x: np.ndarray, fit: dict, **style) -> None:
        # Each line spans its own data, not the other arm's. The two arms share a
        # test split so today the spans coincide, but that is a property of the
        # data, not something this figure should assume: if a subgroup ever drops
        # below the support floor in one arm only, borrowing the other arm's span
        # would draw an extrapolation as if it were measured.
        if len(x) == 0:
            return
        span = np.array([x.min(), x.max()])
        ax.plot(span, fit["slope"] * span + fit["intercept"], **style)

    fitted_line(bx, baseline_fit, color=BASELINE_COLOR, linewidth=1.8)
    fitted_line(mx, mitigated_fit, color=MITIGATED_COLOR, linewidth=1.8, linestyle="--")
    ax.scatter(bx, by, s=34, color=BASELINE_COLOR, zorder=3,
               label=f"baseline (slope {baseline_fit['slope']:.3f})")
    ax.scatter(mx, my, s=34, facecolors="none", edgecolors=MITIGATED_COLOR, linewidths=1.6,
               zorder=3, label=f"mitigated (slope {mitigated_fit['slope']:.3f})")

    # Label the extremes only. Fourteen labels on a two-column figure is a mess,
    # and the ends are the ones the text quotes.
    #
    # The label is placed on whichever side keeps it inside the axes. A fixed
    # rightward offset pushes the rightmost point's name past the spine, where a
    # PDF crops it -- and the rightmost point is `black`, which the text quotes.
    midpoint = (bx.min() + bx.max()) / 2
    for index in {int(np.argmin(bx)), int(np.argmax(bx)), int(np.argmax(by))}:
        to_the_left = bx[index] > midpoint
        ax.annotate(
            names[index], (bx[index], by[index]), textcoords="offset points",
            xytext=(-6 if to_the_left else 6, 4), fontsize=7.5, color="#444444",
            ha="right" if to_the_left else "left",
        )

    ax.set_xlabel("subgroup toxic rate in the corpus (test split)")
    ax.set_ylabel("subgroup false-positive rate")
    ax.set_title("The disparity tracks the training distribution")
    ax.grid(**GRID_KW)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)
    return _finish(fig, path)


def fig_confusion(metrics: dict, path: Path, title: str) -> Path:
    matrix = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], ["pred non-toxic", "pred toxic"])
    ax.set_yticks([0, 1], ["true non-toxic", "true toxic"])
    midpoint = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > midpoint else "black", fontsize=11)
    ax.set_title(title)
    return _finish(fig, path)

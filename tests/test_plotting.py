"""Figure legibility, for the two places where a label leaves the axes.

A clipped label is not a cosmetic complaint when the clipped label is the one the
text quotes. Both defects here were found by opening the shipped PNGs rather than
by any test, and both survived because a figure that renders without raising
looks fine to a test suite. These two pin the fixes.

Nothing here checks that a figure is *pretty*. They check the two properties that
decide whether a reader can read the thing: that the extreme annotation is
anchored on the side that keeps it inside the frame, and that the reporting-floor
caption sits above the plot rather than on top of the longest bar.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fairtox.utils import plotting  # noqa: E402


@pytest.fixture
def kept_figure(monkeypatch):
    """Keep the figure alive so its contents can be inspected.

    ``_finish`` saves and then closes, which is right for a pipeline that renders
    hundreds of figures and wrong for a test that wants to look at one. Only the
    close is suppressed; the figure is still written to disk exactly as usual.
    """
    captured = {}

    real_close = plt.close

    def remember(fig=None):
        if fig is not None and not isinstance(fig, str):
            captured["fig"] = fig
            return
        real_close(fig)

    monkeypatch.setattr(plotting.plt, "close", remember)
    yield captured
    real_close("all")


def _subgroup_rows(names, toxic_rates, fprs):
    return [
        {"subgroup": n, "n": 1000, "n_toxic": int(1000 * t), "n_nontoxic": 1000 - int(1000 * t),
         "fpr": f, "reportable_fpr": True}
        for n, t, f in zip(names, toxic_rates, fprs, strict=True)
    ]


def _fit(slope, intercept=0.0):
    return {"slope": slope, "intercept": intercept, "pearson_r": 0.9, "n_subgroups": 4}


# -- fig_mechanism: the extreme labels ---------------------------------------


def test_the_rightmost_label_is_anchored_on_its_left(tmp_path, kept_figure):
    """A fixed rightward offset pushed `black` past the spine, where a PDF crops it."""
    names = ["christian", "muslim", "white", "black"]
    rows = _subgroup_rows(names, [0.10, 0.18, 0.28, 0.33], [0.02, 0.05, 0.09, 0.08])
    plotting.fig_mechanism(rows, rows, _fit(0.3), _fit(0.2), tmp_path / "m.png")

    labels = {t.get_text(): t for ax in kept_figure["fig"].axes for t in ax.texts}
    assert labels["black"].get_ha() == "right"      # rightmost point, label goes left
    assert labels["christian"].get_ha() == "left"   # leftmost point, label goes right


def test_every_label_is_pushed_towards_the_middle(tmp_path, kept_figure):
    """Whichever half a point sits in, its label is written towards the centre.

    ``.xy`` is the anchor in data coordinates; ``get_position()`` would return the
    offset in points, which is not what decides whether the text clears the spine.
    """
    names = ["a", "b", "c", "d"]
    rows = _subgroup_rows(names, [0.10, 0.18, 0.28, 0.33], [0.02, 0.05, 0.09, 0.08])
    plotting.fig_mechanism(rows, rows, _fit(0.3), _fit(0.2), tmp_path / "m.png")

    ax = kept_figure["fig"].axes[0]
    midpoint = (0.10 + 0.33) / 2
    labelled = [t for t in ax.texts if t.get_text() in names]
    assert labelled, "the extremes were not labelled at all"
    for text in labelled:
        expected = "right" if text.xy[0] > midpoint else "left"
        assert text.get_ha() == expected, f"{text.get_text()} is anchored outwards"


def test_the_figure_is_written(tmp_path):
    rows = _subgroup_rows(["a", "b", "c"], [0.1, 0.2, 0.3], [0.02, 0.05, 0.08])
    path = plotting.fig_mechanism(rows, rows, _fit(0.3), _fit(0.2), tmp_path / "m.png")
    assert path.exists() and path.stat().st_size > 5000


def test_too_few_points_is_skipped_not_crashed(tmp_path):
    """An empty subgroup set must not take the pipeline down."""
    path = plotting.fig_mechanism([], [], _fit(0.3), _fit(0.2), tmp_path / "m.png")
    assert not path.exists()


# -- fig_subgroup_support: the reporting-floor caption ------------------------


def _support_table():
    return pd.DataFrame({
        "subgroup": ["male", "female", "hindu"],
        "n": [5000, 4000, 90],
        "n_nontoxic": [4400, 3500, 80],
        "n_toxic": [600, 500, 10],
    })


def test_the_floor_caption_sits_above_the_axes(tmp_path, kept_figure):
    """Inside the axes it landed on the longest bar -- the first row a reader checks."""
    plotting.fig_subgroup_support(_support_table(), tmp_path / "s.png", min_support=100)

    ax = kept_figure["fig"].axes[0]
    caption = next(t for t in ax.texts if "reporting floor" in t.get_text())
    assert caption.xycoords == ("data", "axes fraction")
    assert caption.xy[1] == pytest.approx(1.0)
    assert caption.get_va() == "bottom"


def test_the_floor_caption_names_the_configured_value(tmp_path, kept_figure):
    plotting.fig_subgroup_support(_support_table(), tmp_path / "s.png", min_support=250)

    ax = kept_figure["fig"].axes[0]
    assert any("N=250" in t.get_text() for t in ax.texts)


def test_the_floor_line_is_drawn_at_the_floor(tmp_path, kept_figure):
    plotting.fig_subgroup_support(_support_table(), tmp_path / "s.png", min_support=100)

    ax = kept_figure["fig"].axes[0]
    verticals = [ln for ln in ax.lines if np.allclose(ln.get_xdata(), 100)]
    assert verticals, "the reporting floor is not marked"

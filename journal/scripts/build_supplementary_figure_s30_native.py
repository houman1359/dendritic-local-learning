"""Supplementary Figure S30 -- the between-by-within credit factorial.

    A additive accuracy | B shunting accuracy | C, D paired seed contrasts

The main-text feedback ladder varies only the within-neuron distribution of
a per-neuron error supplied by exact readout backpropagation. This figure
crosses that ladder with an approximate between-neuron source: a fixed
random soma-level feedback matrix (direct feedback alignment) inside the
LocalCA rule, plus the soma-DFA trainer as the DFA-by-exact-autograd
anchor. Panels read the frozen tables written by
``collect_mnist_between_within_factorial.py``; the exact-readout row is the
frozen ladder release, not a rerun.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figure_canvas import (
    COLORS,
    LW_DATA,
    LW_REF,
    MARKER_MS,
    PT_LEGEND,
    PT_SMALL,
    SEED_ALPHA,
    SEED_MS,
    Margins,
    NativeCanvas,
    audit_native_pdf,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "mnist_between_within_factorial"
OUT = ROOT / "figures" / "supplementary" / "figure_S30_panels_A-D.pdf"

HEIGHT_IN = 340.0 / 72.0
MARGINS = Margins(left=51.0, right=8.0, top=23.0, bottom=40.0)
ROW_WEIGHTS = (135.0, 130.0)
LABEL_COLUMN_PT = 34.0   # shared left lock for the contrast row names
RIGHT_RAIL_PT = 10.0     # shared right lock (last x tick label overhang)

WITHIN = ("scalar broadcast", "neuron specific", "exact path",
          "exact autograd")
WITHIN_SHORT = ("scalar", "neuron", "exact\npath", "exact\nautograd")
# Palette semantics: red-brown is backpropagation, the low-rank rose is the
# fixed random feedback family; additive blue and shunting green stay the
# dynamics colours as in Fig. 2.
C_READOUT = COLORS["bp"]
C_DFA = COLORS["low_rank"]

# Row names are two-line y tick labels: a right-aligned label column
# outside the left spine, centred on the pair it names, so the dashed zero
# line never strikes through a name (2026-09-11 visual review).
LARGE_ROWS = (
    ("dfa within: neuron - scalar", "DFA within:\nneuron $-$ scalar"),
    ("between at scalar: readout - dfa", "between at scalar:\nBP $-$ DFA"),
)
SMALL_ROWS = (
    ("dfa within: exact path - neuron", "DFA within:\npath $-$ neuron"),
    ("dfa within: autograd - exact path", "DFA within:\nautograd $-$ path"),
    ("between at neuron: readout - dfa", "between at neuron:\nBP $-$ DFA"),
    ("between at exact path: readout - dfa", "between at path:\nBP $-$ DFA"),
)
# Each contrast is the paired per-seed difference of two factorial cells,
# (minuend, subtrahend) as (between, within); the seed rows are drawn behind
# the frozen bootstrap interval and their mean is asserted against the
# frozen paired_contrasts.csv mean before anything is drawn.
CONTRAST_CELLS = {
    "dfa within: neuron - scalar":
        (("dfa", "neuron specific"), ("dfa", "scalar broadcast")),
    "dfa within: exact path - neuron":
        (("dfa", "exact path"), ("dfa", "neuron specific")),
    "dfa within: autograd - exact path":
        (("dfa", "exact autograd"), ("dfa", "exact path")),
    "between at scalar: readout - dfa":
        (("readout backprop", "scalar broadcast"), ("dfa", "scalar broadcast")),
    "between at neuron: readout - dfa":
        (("readout backprop", "neuron specific"), ("dfa", "neuron specific")),
    "between at exact path: readout - dfa":
        (("readout backprop", "exact path"), ("dfa", "exact path")),
}


def seed_differences(outcomes, architecture, key):
    """Per-seed paired difference (pp) for one contrast, seed-aligned."""
    (b1, w1), (b0, w0) = CONTRAST_CELLS[key]
    part = outcomes[outcomes.architecture.eq(architecture)]

    def cell(between, within):
        rows = part[part.between.eq(between) & part.within.eq(within)]
        if rows.duplicated("seed").any():
            raise RuntimeError(f"duplicate seeds in {architecture}/{between}/{within}")
        return rows.set_index("seed")["test_accuracy"].sort_index()

    first, second = cell(b1, w1), cell(b0, w0)
    if not first.index.equals(second.index):
        raise RuntimeError(f"{architecture}/{key}: the two cells are not seed-paired")
    return 100.0 * (first.to_numpy(float) - second.to_numpy(float))


def panel_accuracy(ax, summary, architecture):
    part = summary[summary.architecture.eq(architecture)]
    for between, color in (("readout backprop", C_READOUT), ("dfa", C_DFA)):
        rows = part[part.between.eq(between)]
        xs, means, lows, highs = [], [], [], []
        for i, within in enumerate(WITHIN):
            row = rows[rows.within.eq(within)]
            if row.empty:
                continue
            xs.append(i)
            means.append(float(row.mean_test_accuracy.iloc[0]))
            lows.append(float(row.ci95_low_test_accuracy.iloc[0]))
            highs.append(float(row.ci95_high_test_accuracy.iloc[0]))
        xs = np.asarray(xs, float)
        # The autograd anchor is a different trainer, so the line stops at
        # the exact-path rung and the anchor stands alone.
        joined = xs <= 2
        ax.plot(xs[joined], np.asarray(means)[joined], color=color,
                lw=LW_DATA, zorder=2)
        ax.errorbar(xs, means, yerr=[np.asarray(means) - np.asarray(lows),
                                     np.asarray(highs) - np.asarray(means)],
                    fmt="o", color=color, ms=MARKER_MS, lw=0,
                    elinewidth=LW_REF, capsize=1.6, zorder=3)
    ax.set_xticks(range(len(WITHIN)))
    ax.set_xticklabels(WITHIN_SHORT, fontsize=PT_SMALL)
    ax.set_xlim(-0.4, len(WITHIN) - 0.6)
    ax.set_ylabel("held-out accuracy")
    ax.set_xlabel("within-neuron distribution")


def panel_contrasts(ax, contrasts, outcomes, rows, *, xlabel):
    """A forest of paired seed contrasts: one row per contrast type, both
    dynamics drawn as vertically offset points inside the row.  The fifteen
    paired seed differences sit as small open marks behind each interval;
    the filled mark is the mean and the bar the frozen 95% bootstrap
    interval.  Row names are y tick labels outside the left spine; the
    additive/shunting key is two coloured words in the top-right corner,
    above every data row, so no key ink is drawn on a data row."""
    ys, labels = [], []
    rng = np.random.default_rng(20260911)
    for index, (key, label) in enumerate(rows):
        y = -index
        for architecture, color, offset in (
            ("additive", COLORS["additive"], 0.13),
            ("shunting", COLORS["shunting"], -0.13),
        ):
            part = contrasts[contrasts.architecture.eq(architecture)
                             & contrasts.contrast.eq(key)]
            if part.empty:
                continue
            mean = 100.0 * float(part.mean_difference.iloc[0])
            low = 100.0 * float(part.ci95_low.iloc[0])
            high = 100.0 * float(part.ci95_high.iloc[0])
            seeds = seed_differences(outcomes, architecture, key)
            if seeds.size != int(part.n_seeds.iloc[0]) or not np.isclose(
                    seeds.mean(), mean, rtol=0.0, atol=1e-9):
                raise RuntimeError(
                    f"{architecture}/{key}: seed differences do not reproduce "
                    f"the frozen mean ({seeds.mean():.6f} vs {mean:.6f})")
            print(f"  {key} [{architecture}]: mean {mean:.4f} pp, "
                  f"CI {low:.4f}..{high:.4f}, seeds {seeds.min():.3f}..{seeds.max():.3f}")
            jitter = rng.uniform(-0.055, 0.055, seeds.size)
            ax.plot(seeds, y + offset + jitter, ls="none", marker="o",
                    ms=SEED_MS, mfc="white", mec=color, mew=0.45,
                    alpha=SEED_ALPHA, zorder=2)
            ax.plot([low, high], [y + offset] * 2, color=color, lw=LW_DATA,
                    solid_capstyle="butt", zorder=3)
            ax.plot([mean], [y + offset], marker="o", ms=MARKER_MS - 0.6,
                    mfc=color, mec="white", mew=0.45, ls="none", zorder=4)
        ys.append(y)
        labels.append(label)
    ax.axvline(0.0, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=PT_SMALL, ha="right",
                       linespacing=1.05)
    ax.tick_params(axis="y", length=0, pad=3.0)
    ax.set_ylim(min(ys) - 0.42, max(ys) + 0.78)
    ax.set_xlabel(xlabel)
    # Word key, top-right, right-to-left: measure the first word so the
    # second sits one gap to its left at any panel width.
    second = ax.text(0.985, 0.975, "shunting", transform=ax.transAxes,
                     ha="right", va="top", fontsize=PT_LEGEND,
                     color=COLORS["shunting"])
    ax.figure.canvas.draw()
    extent = second.get_window_extent(ax.figure.canvas.get_renderer())
    x_left = ax.transAxes.inverted().transform((extent.x0, extent.y0))[0]
    ax.text(x_left - 0.04, 0.975, "additive", transform=ax.transAxes,
            ha="right", va="top", fontsize=PT_LEGEND,
            color=COLORS["additive"])


def main():
    summary = pd.read_csv(DATA / "condition_summary.csv")
    contrasts = pd.read_csv(DATA / "paired_contrasts.csv")
    outcomes = pd.read_csv(DATA / "seed_outcomes.csv")

    canvas = NativeCanvas(
        HEIGHT_IN, nrows=2, row_weights=ROW_WEIGHTS, hgutter_pt=40.0,
        vgutter_pt=56.0, margins=MARGINS)
    # The contrast row-name columns reach 28 pt into the gutter; one
    # figure-wide 30 pt letter offset keeps every letter the leftmost mark
    # of its own panel, which the supplement paste step relies on to
    # partition the row at each letter.
    canvas.letter_dx = 30.0
    ax_a = canvas.panel("A", 0, 0, 6, title="Additive")
    ax_b = canvas.panel("B", 0, 6, 6, title="Shunting")
    ax_c = canvas.panel("C", 1, 0, 6, title="MNIST, DFA: large contrasts")
    ax_d = canvas.panel("D", 1, 6, 6,
                        title="MNIST, DFA: contrasts below 0.5 pp")
    panel_accuracy(ax_a, summary, "additive")
    panel_accuracy(ax_b, summary, "shunting")
    handles = [Line2D([], [], color=C_READOUT, marker="o", lw=LW_DATA,
                      markersize=MARKER_MS, label="readout backprop"),
               Line2D([], [], color=C_DFA, marker="o", lw=LW_DATA,
                      markersize=MARKER_MS, label="soma-level DFA")]
    ax_a.legend(handles=handles, loc="lower right", frameon=False,
                fontsize=PT_LEGEND, handlelength=1.2, borderaxespad=0.2)
    panel_contrasts(ax_c, contrasts, outcomes, LARGE_ROWS,
                    xlabel="paired accuracy difference (pp)")
    panel_contrasts(ax_d, contrasts, outcomes, SMALL_ROWS,
                    xlabel="paired accuracy difference (pp; expanded scale)")
    # The two row-name columns are nearly the same width but one sits on
    # the outer margin and the other on a gutter; one declared reserve for
    # both start columns keeps every 6-module panel at one width.
    canvas.declare_reserve("C", left=LABEL_COLUMN_PT, right=RIGHT_RAIL_PT)
    canvas.declare_reserve("D", left=LABEL_COLUMN_PT, right=RIGHT_RAIL_PT)
    problems = canvas.save(OUT, name="figure_S30_panels_A-D")
    for violation in audit_native_pdf(OUT):
        print(f"    {violation}")
    return problems


if __name__ == "__main__":
    main()

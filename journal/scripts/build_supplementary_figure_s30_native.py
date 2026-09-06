"""Supplementary Figure S30 -- the between-by-within credit factorial.

    A additive accuracy | B shunting accuracy | C paired seed contrasts

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
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    Margins,
    NativeCanvas,
    audit_native_pdf,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "mnist_between_within_factorial"
OUT = ROOT / "figures" / "supplementary" / "figure_S30_panels_A-D.pdf"

HEIGHT_IN = 340.0 / 72.0
MARGINS = Margins(left=51.0, right=13.5, top=23.0, bottom=40.0)
ROW_WEIGHTS = (135.0, 130.0)

WITHIN = ("scalar broadcast", "neuron specific", "exact path",
          "exact autograd")
WITHIN_SHORT = ("scalar", "neuron", "exact\npath", "exact\nautograd")
# Palette semantics: red-brown is backpropagation, the low-rank rose is the
# fixed random feedback family; additive blue and shunting green stay the
# dynamics colours as in Fig. 2.
C_READOUT = COLORS["bp"]
C_DFA = COLORS["low_rank"]

LARGE_ROWS = (
    ("dfa within: neuron - scalar", "DFA within: neuron $-$ scalar"),
    ("between at scalar: readout - dfa", "between at scalar: BP $-$ DFA"),
)
SMALL_ROWS = (
    ("dfa within: exact path - neuron", "DFA within: path $-$ neuron"),
    ("dfa within: autograd - exact path", "DFA within: autograd $-$ path"),
    ("between at neuron: readout - dfa", "between at neuron: BP $-$ DFA"),
    ("between at exact path: readout - dfa", "between at path: BP $-$ DFA"),
)


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


def panel_contrasts(ax, contrasts, rows, *, legend=False):
    """A forest of paired seed contrasts: one row per contrast type, both
    dynamics drawn as vertically offset points inside the row. Row names are
    drawn inside the axes so they never inflate the shared column reserve."""
    from matplotlib.transforms import blended_transform_factory

    label_transform = blended_transform_factory(ax.transAxes, ax.transData)
    ys = []
    for index, (key, label) in enumerate(rows):
        y = -index
        for architecture, color, offset in (
            ("additive", COLORS["additive"], 0.10),
            ("shunting", COLORS["shunting"], -0.10),
        ):
            part = contrasts[contrasts.architecture.eq(architecture)
                             & contrasts.contrast.eq(key)]
            if part.empty:
                continue
            mean = 100.0 * float(part.mean_difference.iloc[0])
            low = 100.0 * float(part.ci95_low.iloc[0])
            high = 100.0 * float(part.ci95_high.iloc[0])
            ax.plot([low, high], [y + offset] * 2, color=color, lw=LW_DATA,
                    solid_capstyle="butt", zorder=2)
            ax.plot([mean], [y + offset], marker="o", ms=MARKER_MS - 0.6,
                    mfc=color, mec=color, ls="none", zorder=3)
        ax.text(0.01, y + 0.36, label, transform=label_transform,
                ha="left", va="center", fontsize=PT_SMALL,
                color=COLORS["ink"])
        ys.append(y)
    ax.axvline(0.0, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=1)
    ax.set_yticks([])
    ax.set_ylim(min(ys) - 0.5, max(ys) + 0.62)
    ax.set_xlabel("paired accuracy difference (pp)")
    if legend:
        handles = [Line2D([], [], color=COLORS["additive"], lw=LW_DATA,
                          label="additive"),
                   Line2D([], [], color=COLORS["shunting"], lw=LW_DATA,
                          label="shunting")]
        ax.legend(handles=handles, loc="lower left", frameon=False,
                  fontsize=PT_LEGEND, handlelength=1.2, borderaxespad=0.2)


def main():
    summary = pd.read_csv(DATA / "condition_summary.csv")
    contrasts = pd.read_csv(DATA / "paired_contrasts.csv")

    canvas = NativeCanvas(
        HEIGHT_IN, nrows=2, row_weights=ROW_WEIGHTS, hgutter_pt=40.0,
        vgutter_pt=56.0, margins=MARGINS)
    ax_a = canvas.panel("A", 0, 0, 6, title="Additive")
    ax_b = canvas.panel("B", 0, 6, 6, title="Shunting")
    ax_c = canvas.panel("C", 1, 0, 5, title="Dominant contrasts")
    ax_d = canvas.panel("D", 1, 5, 7, title="Sub-point contrasts")
    panel_accuracy(ax_a, summary, "additive")
    panel_accuracy(ax_b, summary, "shunting")
    handles = [Line2D([], [], color=C_READOUT, marker="o", lw=LW_DATA,
                      markersize=MARKER_MS, label="readout backprop"),
               Line2D([], [], color=C_DFA, marker="o", lw=LW_DATA,
                      markersize=MARKER_MS, label="soma-level DFA")]
    ax_a.legend(handles=handles, loc="lower right", frameon=False,
                fontsize=PT_LEGEND, handlelength=1.2, borderaxespad=0.2)
    panel_contrasts(ax_c, contrasts, LARGE_ROWS)
    panel_contrasts(ax_d, contrasts, SMALL_ROWS, legend=True)
    problems = canvas.save(OUT, name="figure_S30_panels_A-D")
    for violation in audit_native_pdf(OUT):
        print(f"    {violation}")
    return problems


if __name__ == "__main__":
    main()

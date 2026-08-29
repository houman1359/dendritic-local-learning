#!/usr/bin/env python3
"""Build vector assets for the workshop credit-conflict experiment.

The task schematic and plotting logic are shared with Supplementary Fig. S29,
but the aspect ratios and type sizes are rebuilt for a 16:9 talk.  All labels
use ``chi`` for the conflict probability.  The generated PDFs contain vector
paths and text; no Canvas or raster export is embedded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
JOURNAL_SCRIPTS = REPO / "journal" / "scripts"
SOURCE = REPO / "journal" / "source_data" / "path_necessity_fashion"
OUTPUT = HERE / "pdf_assets"

sys.path.insert(0, str(JOURNAL_SCRIPTS))
import build_path_necessity_fashion_figure as source_figure  # noqa: E402


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "DejaVu Sans"],
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 8.8,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # The source component was typeset for a journal page.  These globals are
    # read at call time, so increasing them here preserves the same geometry
    # while making the derivative legible at projection distance.
    source_figure.PT_SMALL = 9.6
    source_figure.PT_ANNOT = 10.5
    source_figure.PT_LABEL = 10.5
    source_figure.PT_LEGEND = 8.8
    source_figure.LW_DATA = 1.75
    source_figure.LW_EDGE = 0.95
    source_figure.LW_ERR = 0.85
    source_figure.LW_HAIR = 0.65
    source_figure.LW_REF = 0.85
    source_figure.MARKER_MS = 6.0


def _save(fig: mpl.figure.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / name
    fig.savefig(
        target,
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={
            "Creator": "dendritic-local-learning",
            "Subject": "Workshop vector derivative of Supplementary Fig. S29",
        },
    )
    plt.close(fig)
    print(target)


def _task_card(frame, rect, *, conflict: bool) -> None:
    """Draw one clean B=4 realization without journal-scale micro-labels."""
    x0, y0, width, height = rect
    green = source_figure.GREEN
    mute = source_figure.MUTE
    grid = source_figure.GRID
    bp = source_figure.COLORS["bp"]
    ink = source_figure.INK
    tint = source_figure.mix("bp", 7) if conflict else source_figure.mix("shunting", 7)
    edge = source_figure.mix("bp", 38) if conflict else source_figure.mix("shunting", 38)
    frame.group(rect, tint=tint, edge=edge)
    frame.text(
        (x0 + 0.035 * width, y0 + 0.90 * height),
        r"conflicting, $\chi=1$" if conflict else r"compatible, $\chi=0$",
        size=10.8,
        color=bp if conflict else green,
        ha="left",
    )

    xs = x0 + width * np.asarray([0.11, 0.35, 0.65, 0.89])
    selected = 1
    y_view = y0 + 0.68 * height
    y_branch = y0 + 0.43 * height
    y_gate = y0 + 0.16 * height
    gate_x = x0 + 0.50 * width

    for branch, xpos in enumerate(xs):
        is_selected = branch == selected
        disagrees = conflict and not is_selected
        face = (
            source_figure.mix("shunting", 12)
            if is_selected
            else source_figure.mix("bp", 12)
            if disagrees
            else "white"
        )
        box_edge = green if is_selected else bp if disagrees else grid
        label = r"$x_c:y$" if is_selected else (r"$x_b:1-y$" if disagrees else r"$x_b:y$")
        source_figure._round_box(
            frame.ax,
            (xpos, y_view),
            0.19 * width,
            0.15 * height,
            face=face,
            edge=box_edge,
            radius=0.012,
        )
        frame.text(
            (xpos, y_view),
            label,
            size=9.6,
            color=green if is_selected else bp if disagrees else mute,
        )
        frame.leader(
            (xpos, y_view - 0.082 * height),
            (xpos, y_branch + 0.045 * height),
            color=green if is_selected else mute,
            lw=1.2 if is_selected else 0.7,
        )
        source_figure._round_box(
            frame.ax,
            (xpos, y_branch),
            0.105 * width,
            0.09 * height,
            face="white",
            edge=green if is_selected else grid,
            radius=0.009,
        )
        frame.text(
            (xpos, y_branch),
            rf"$b={branch + 1}$",
            size=7.8,
            color=green if is_selected else mute,
        )
        line, = frame.ax.plot(
            [xpos, gate_x],
            [y_branch - 0.05 * height, y_gate + 0.055 * height],
            color=green if is_selected else mute,
            lw=2.0 if is_selected else 0.65,
            alpha=1.0 if is_selected else 0.45,
            solid_capstyle="round",
            zorder=2,
        )
        if not is_selected:
            line.set_dashes((2.0, 1.8))

    source_figure._round_box(
        frame.ax,
        (gate_x, y_gate),
        0.27 * width,
        0.12 * height,
        face="white",
        edge=green,
    )
    frame.text((gate_x, y_gate), r"downstream gate $c$", size=8.9, color=green)
    z_x = x0 + 0.83 * width
    frame.arrow(
        (gate_x + 0.15 * width, y_gate),
        (z_x - 0.045 * width, y_gate),
        color=green,
        lw=1.1,
        head=4.5,
    )
    frame.disc(
        (z_x, y_gate),
        5.0,
        fill=source_figure.mix("soma", 24),
        edge=source_figure.COLORS["soma"],
        lw=0.9,
    )
    frame.text((z_x, y_gate), r"$z$", size=8.8, color=ink)


def build_task() -> None:
    fig, ax = plt.subplots(figsize=(5.15, 3.30))
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    frame = source_figure.Frame(ax, labels=True, scale=0.96)
    _task_card(frame, (0.02, 0.54, 0.96, 0.43), conflict=False)
    _task_card(frame, (0.02, 0.03, 0.96, 0.43), conflict=True)
    _save(fig, "path_necessity_task.pdf")


def build_results() -> None:
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    crossings = pd.read_csv(SOURCE / "plotted_crossings.csv")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(9.35, 3.55),
        gridspec_kw={"width_ratios": [1.40, 1.0]},
    )
    source_figure._accuracy_panel(axes[0], summary)
    source_figure._boundary_panel(axes[1], crossings)
    axes[0].set_title("Learning crosses the predicted boundary", pad=7.0)
    axes[1].set_title("Observed collapse follows theory", pad=7.0)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.88, wspace=0.36)
    _save(fig, "path_necessity_results.pdf")


def main() -> None:
    _style()
    build_task()
    build_results()


if __name__ == "__main__":
    main()

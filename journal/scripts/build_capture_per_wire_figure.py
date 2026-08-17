#!/usr/bin/env python3
"""Render the wiring-normalized capture supplementary figure.

Reads the frozen tables written by ``scripts/analyze_capture_per_wire.py``
(``source_data/capture_per_wire/``) and draws two panels: the 8-channel
wiring-capture plane with iso-efficiency references, and capture per unit
wiring across channel counts with paired cell-bootstrap bands.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_REF,
    MARKER_MS,
    MARKERS,
    PT_SMALL,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "capture_per_wire"
FIGURES = ROOT / "figures" / "generated"

# Method order, colours and markers match the Figure 3 conventions.
METHODS = [
    ("dense PCA oracle", "dense", COLORS["oracle"]),
    ("morphology-aware paths", "ances.", COLORS["shunting"]),
    ("random paths", "random", COLORS["point_mlp"]),
    ("depth-only bins", "depth", COLORS["additive"]),
    ("shuffled ancestry", "shuffle", COLORS["highlight"]),
]


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, name)
    audit_text_over_data(fig, name)
    fig.savefig(
        FIGURES / f"{name}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / f"{name}.png", dpi=600)
    plt.close(fig)


def iso_efficiency(ax: plt.Axes, ratios, x_low: float, y_top: float) -> None:
    """Dashed capture/wiring = const references on the log-x plane."""
    xs = np.geomspace(x_low, ax.get_xlim()[1], 256)
    for ratio in ratios:
        ys = ratio * xs
        keep = ys <= y_top
        ax.plot(xs[keep], ys[keep], ls="--", lw=LW_REF, color=COLORS["mute"],
                zorder=1)
        exit_x = min(y_top / ratio, ax.get_xlim()[1])
        label = f"{ratio:g}" if ratio >= 1 else f"{ratio:.1f}"
        ax.text(exit_x * 1.1, min(y_top, ratio * exit_x) - 0.012, label,
                ha="left", va="top", fontsize=PT_SMALL, style="italic",
                color=COLORS["mute"])


def main() -> None:
    apply_neurips_style()
    cell = pd.read_csv(SOURCE / "cell_method_channel.csv")
    summary = pd.read_csv(SOURCE / "summary.csv")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(FIG_W, 2.55),
        gridspec_kw={"left": 0.075, "right": 0.985, "bottom": 0.155,
                     "top": 0.855, "wspace": 0.42},
    )

    # ── A: 8-channel wiring-capture plane with iso-efficiency lines ───────
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.008, 1.6)
    ax_a.set_ylim(0.0, 1.34)
    ax_a.set_yticks([0.0, 0.5, 1.0])
    iso_efficiency(ax_a, (0.5, 2.0, 8.0), x_low=0.009, y_top=1.0)
    eight = cell[cell.channels.eq(8)]
    for index, (method, _, color) in enumerate(METHODS):
        part = eight[eight.method.eq(method)]
        ax_a.scatter(part.wiring_density, part.credit_capture, s=12,
                     color=color, marker=MARKERS[index], alpha=0.35,
                     edgecolors="none")
        ax_a.scatter(part.wiring_density.mean(), part.credit_capture.mean(),
                     marker=MARKERS[index], s=30, color=color,
                     edgecolor="white", linewidth=0.4, zorder=4)
    ax_a.set_xlabel("wiring density")
    ax_a.set_ylabel("field capture")
    panel_title(ax_a, "K", "Iso-efficiency at eight channels")
    style_axis(ax_a)
    clean_legend(ax_a, handles=[
        Line2D([0], [0], marker=MARKERS[index], color="none",
               markerfacecolor=color, markeredgecolor="none",
               markersize=4.2, label=label)
        for index, (_, label, color) in enumerate(METHODS)
    ], loc="upper left", ncol=2, fontsize=PT_SMALL, columnspacing=0.6,
        handlelength=0.9, handletextpad=0.25)
    # The note lives in the dedicated headroom band (panel-E convention of
    # Figure 3), right of the two-column key and clear of the point cloud
    # and the iso-line end labels.
    ax_a.text(0.98, 0.97, "dashed: constant\ncapture per wire",
              transform=ax_a.transAxes, ha="right", va="top",
              fontsize=PT_SMALL, style="italic", color=COLORS["mute"],
              linespacing=1.25)

    # ── B: capture per unit wiring across channel counts ──────────────────
    for index, (method, label, color) in enumerate(METHODS):
        part = summary[summary.method.eq(method)].sort_values("channels")
        ax_b.plot(part.channels, part.mean_capture_per_wire, color=color,
                  marker=MARKERS[index], ms=MARKER_MS, lw=LW_DATA,
                  markeredgecolor="white", markeredgewidth=0.3, label=label)
        ax_b.fill_between(part.channels, part.ci95_low_capture_per_wire,
                          part.ci95_high_capture_per_wire, color=color,
                          alpha=0.10, linewidth=0)
    ax_b.set_xscale("log", base=2)
    ax_b.set_yscale("log")
    ax_b.set_xticks([1, 2, 4, 8])
    ax_b.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_b.set_xlabel("feedback channels")
    ax_b.set_ylabel("capture per unit wiring")
    panel_title(ax_b, "L", "Wiring-normalized capture")
    style_axis(ax_b)
    clean_legend(ax_b, loc="lower right", ncol=2, fontsize=PT_SMALL,
                 columnspacing=0.6, handlelength=1.1, handletextpad=0.3)
    ax_b.text(0.02, 0.97, "bands: 95% cell bootstrap",
              transform=ax_b.transAxes, ha="left", va="top",
              fontsize=PT_SMALL, style="italic", color=COLORS["mute"])

    save(fig, "fig_capture_per_wire")


if __name__ == "__main__":
    main()

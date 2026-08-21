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

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_REF,
    MARKER_MS,
    MARKERS,
    PT_ANNOT,
    PT_LEGEND,
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
    """Dashed capture/wiring = const references on the log-x plane.

    Each line carries a short ratio label at its exit point, inside the
    axes; the caption decodes the family (constant capture per wire), so
    the panel needs no separate note.
    """
    x_high = ax.get_xlim()[1]
    xs = np.geomspace(x_low, x_high, 256)
    for ratio in sorted(ratios, reverse=True):
        ys = ratio * xs
        keep = ys <= y_top
        ax.plot(xs[keep], ys[keep], ls="--", lw=LW_REF, color=COLORS["mute"],
                zorder=1)
        label = f"{ratio:g}"
        if y_top / ratio < x_high:
            # Exits through the top: label just right of the exit point,
            # tucked under the axis ceiling.
            ax.text(y_top / ratio * 1.14, y_top, label, ha="left", va="top",
                    fontsize=PT_ANNOT, style="italic", color=COLORS["mute"])
        else:
            # Exits through the right edge: label above the line end,
            # inside the axes.
            ax.text(x_high * 0.96, ratio * x_high * 0.96 + 0.015, label,
                    ha="right", va="bottom", fontsize=PT_ANNOT,
                    style="italic", color=COLORS["mute"])


def main() -> None:
    apply_neurips_style()
    cell = pd.read_csv(SOURCE / "cell_method_channel.csv")
    summary = pd.read_csv(SOURCE / "summary.csv")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(FIG_W, 2.55),
        # top = 0.825 gives the K/L letters the same ~65 px (600 dpi) head
        # margin as the figure-6 A-J block above them.
        gridspec_kw={"left": 0.075, "right": 0.985, "bottom": 0.155,
                     "top": 0.825, "wspace": 0.42},
    )

    # ── K: 8-channel wiring-capture plane with iso-efficiency lines ───────
    # Axis ranges match panel E of this figure (the same data on the same
    # plane); the iso lines are labeled in-line, so the panel keeps no
    # headroom band for a key or a decoder note — the five-method key is
    # stated once in panel L (panels D/I of the A-J strip state it too).
    ax_a.set_xscale("log")
    ax_a.set_xlim(0.008, 1.35)
    ax_a.set_ylim(0.0, 0.88)
    ax_a.set_yticks([0.0, 0.4, 0.8])
    iso_efficiency(ax_a, (0.5, 2.0, 8.0), x_low=0.009, y_top=0.84)
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
    ax_a.text(0.03, 0.95, "colors as in F", transform=ax_a.transAxes,
              ha="left", va="top", fontsize=PT_ANNOT, style="italic",
              color=COLORS["mute"])
    panel_title(ax_a, "K", "Iso-efficiency at eight channels")
    style_axis(ax_a)

    # ── L: capture per unit wiring across channel counts ──────────────────
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
    # Single key for the strip (K shares its colors and markers); the
    # bootstrap-band provenance lives in the caption, not the panel.
    # Slim white backing keeps the key legible against the CI bands it
    # borders (and on reruns with jittered data).
    clean_legend(ax_b, loc="lower right", ncol=2, fontsize=PT_LEGEND,
                 columnspacing=0.7, handlelength=1.1, handletextpad=0.3,
                 frameon=True, facecolor="white", edgecolor="none",
                 framealpha=0.85, borderpad=0.25)

    save(fig, "fig_capture_per_wire")


if __name__ == "__main__":
    main()

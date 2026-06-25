#!/usr/bin/env python3
"""Generate the submission Figure 1 schematic.

Saves:
  figures/fig1_model_and_credit.{pdf,png}

Layout (single row, 7.0 x 2.6 inches):
  A. Single dendritic E/I unit  (3-level branched tree, E/I synapses,
     gold-ringed soma; nonnegative E and I input streams on the left).
  B. Network layer  (E pool / I pool feed N=4 dendritic units; one unit
     gold-ringed to mark it as "the unit shown in panel A"; per-unit
     somas project to a task readout; a delta_0 callout exits the readout
     on the right and seeds the broadcast in panel C).
  C. Credit assignment and broadcast modes  (left: small dendritic tree
     receives exact alpha_n delta_0 errors per branch and a rank-1 shared
     broadcast bar; right: vertical stack of broadcast modes; bottom:
     local/non-local color-split equation strip).

Panels A and B follow the polished schematics in
  drafts/dendritic-information-processing/.../scripts/generate_neurips_figures.py
  (function `fig1_framework_local`, polished_panel_a branch),
adapted to the local-CA color palette and with the input column rewired
to nonnegative E and I streams instead of the original gain/g population.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import COLORS, apply_neurips_style, panel_label

apply_neurips_style()

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"

# Local-CA palette mapping (single source of truth used by all panels)
EXC = COLORS["exc"]            # excitatory blue
INH = COLORS["inh"]            # inhibitory red
DENDRITE = "#8a7a6a"           # earthy dendrite shaft (anatomical)
LEAF_FACE = "white"
LEAF_EDGE = "#555"
SOMA_FACE = "#e8e8e8"
SOMA_EDGE = "#333"
GOLD_FACE = "#fff3d8"
GOLD_EDGE = "#e3a635"
INK = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]
DELTA_COLOR = COLORS["bp"]     # delta_0 / exact-error red-brown
BROADCAST_COLOR = COLORS["local"]  # rank-1 shared broadcast amber

# Broadcast-mode colors (kept consistent with rest of paper)
MODE_COLORS = {
    "rank1":   COLORS["scalar"],
    "persoma": COLORS["local"],
    "rankk":   COLORS["low_rank"],
    "path":    COLORS["pathway"],
    "oracle":  COLORS["oracle"],
}


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------

def _draw_compartment(ax, nx, ny, nr, syn_r):
    """White compartment circle with E (left) and I (right) synapses on top."""
    ax.add_patch(Circle((nx, ny), nr, fc=LEAF_FACE, ec=LEAF_EDGE,
                        lw=0.95, zorder=5))
    ax.add_patch(Circle((nx - 0.55, ny + nr + syn_r + 0.08), syn_r,
                        fc=EXC, ec="white", lw=0.3, zorder=8))
    ax.add_patch(Circle((nx + 0.55, ny + nr + syn_r + 0.08), syn_r,
                        fc=INH, ec="white", lw=0.3, zorder=8))


def _draw_mini_tree(ax, cx, cy, *, scale=1.0, highlight=False):
    """4-leaf -> 2-mid -> 1-soma minimalist dendritic icon for panel B."""
    s = scale
    leaf_x = cx - 0.052 * s
    mid_x = cx - 0.018 * s
    soma_x = cx + 0.026 * s
    leaf_dy = 0.030 * s
    leaf_ys = np.array([cy + 1.5 * leaf_dy, cy + 0.5 * leaf_dy,
                        cy - 0.5 * leaf_dy, cy - 1.5 * leaf_dy])
    mid_ys = np.array([cy + leaf_dy, cy - leaf_dy])
    for i, my in enumerate(mid_ys):
        for ly in leaf_ys[i * 2: i * 2 + 2]:
            ax.plot([leaf_x, mid_x], [ly, my],
                    color=DENDRITE, lw=0.55,
                    solid_capstyle="round", zorder=3, alpha=0.9)
    for my in mid_ys:
        ax.plot([mid_x, soma_x], [my, cy],
                color=DENDRITE, lw=0.85,
                solid_capstyle="round", zorder=3, alpha=0.95)
    leaf_r = 0.0070 * s
    for ly in leaf_ys:
        ax.add_patch(Circle((leaf_x, ly), leaf_r,
                            fc=LEAF_FACE, ec="#6e5d4f", lw=0.5, zorder=5))
    for my in mid_ys:
        ax.add_patch(Circle((mid_x, my), leaf_r,
                            fc=LEAF_FACE, ec="#6e5d4f", lw=0.5, zorder=5))
    soma_r = 0.0125 * s
    if highlight:
        ax.add_patch(Circle((soma_x, cy), soma_r + 0.005,
                            fc=GOLD_FACE, ec=GOLD_EDGE,
                            lw=1.0, zorder=6))
    ax.add_patch(Circle((soma_x, cy), soma_r,
                        fc="#efefef", ec=SOMA_EDGE, lw=0.7, zorder=7))


def _round_box(ax, xy, w, h, *, fc, ec, text=None, color="#333",
               fontsize=6.7, lw=0.9, zorder=4, fontweight="bold"):
    patch = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        fc=fc, ec=ec, lw=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    if text is not None:
        ax.text(xy[0] + w / 2, xy[1] + h / 2, text,
                ha="center", va="center",
                fontsize=fontsize, color=color,
                fontweight=fontweight, zorder=zorder + 1)
    return patch


# ---------------------------------------------------------------------------
# Panel A: Single dendritic E/I unit
# ---------------------------------------------------------------------------

def panel_a(ax):
    """Single dendritic E/I unit with explicit nonnegative E and I input
    streams on the left.  Adapted from the polished schematic in
    dendritic-information-processing/scripts/generate_neurips_figures.py.
    """
    ax.set_title("Single dendritic E/I unit",
                 fontsize=9.0, pad=4.0, fontweight="bold", loc="center")
    ax.axis("off")
    ax.set_xlim(-2.5, 22.0)
    ax.set_ylim(-2.5, 12.4)
    ax.set_aspect("equal", adjustable="datalim")

    syn_r = 0.22
    nr_a = 0.40
    sr_a = 0.60

    x_d3 = 5.0
    x_d2 = 10.0
    x_d1 = 14.5
    x_soma = 18.5
    ys_d3 = np.array([10.5, 9.0, 7.0, 5.5, 4.5, 3.0, 1.0, -0.5])
    ys_d2 = np.array([9.75, 6.25, 3.75, 0.25])
    ys_d1 = np.array([8.0, 2.0])
    y_soma = 5.0

    # ---------------- E and I input streams (replaces the source's
    # "gain g / input population" block) ----------------
    # Two compact vertical stacks of small circles, each with a single
    # arrow pointing into the distal (level-3) leaf cluster.
    e_in_x = -0.5
    i_in_x = -0.5
    e_in_ys = np.linspace(7.6, 9.6, 4)
    i_in_ys = np.linspace(0.4, 2.4, 4)
    for ey in e_in_ys:
        ax.add_patch(Circle((e_in_x, ey), 0.20,
                            fc=EXC, ec="white", lw=0.25, zorder=4))
    for iy in i_in_ys:
        ax.add_patch(Circle((i_in_x, iy), 0.20,
                            fc=INH, ec="white", lw=0.25, zorder=4))
    # Arrows from each input column into the upper / lower halves of the
    # distal tree.
    ax.annotate("", xy=(x_d3 - nr_a - 0.4, np.mean(ys_d3[:4])),
                xytext=(e_in_x + 0.30, np.mean(e_in_ys)),
                arrowprops=dict(arrowstyle="-|>", color=EXC, lw=1.1,
                                shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(x_d3 - nr_a - 0.4, np.mean(ys_d3[4:])),
                xytext=(i_in_x + 0.30, np.mean(i_in_ys)),
                arrowprops=dict(arrowstyle="-|>", color=INH, lw=1.1,
                                shrinkA=0, shrinkB=0))
    # Stream labels
    ax.text(e_in_x, e_in_ys[-1] + 0.95, r"$x^E\!\geq\!0$",
            ha="center", va="center", fontsize=6.6,
            color=EXC, fontweight="bold")
    ax.text(i_in_x, i_in_ys[0] - 0.95, r"$x^I\!\geq\!0$",
            ha="center", va="center", fontsize=6.6,
            color=INH, fontweight="bold")

    # ---------------- Dendritic tree edges -----------------
    for i, d2y in enumerate(ys_d2):
        for d3y in ys_d3[i * 2: i * 2 + 2]:
            ax.plot([x_d3 + nr_a, x_d2 - nr_a], [d3y, d2y],
                    color=DENDRITE, lw=1.0,
                    solid_capstyle="round", zorder=2)
    for i, d1y in enumerate(ys_d1):
        for d2y in ys_d2[i * 2: i * 2 + 2]:
            ax.plot([x_d2 + nr_a, x_d1 - nr_a], [d2y, d1y],
                    color=DENDRITE, lw=1.4,
                    solid_capstyle="round", zorder=2)
    for d1y in ys_d1:
        ax.plot([x_d1 + nr_a, x_soma - sr_a], [d1y, y_soma],
                color=DENDRITE, lw=1.8,
                solid_capstyle="round", zorder=2)

    # ---------------- Compartments + synapse markers -----------------
    for d3y in ys_d3:
        _draw_compartment(ax, x_d3, d3y, nr_a, syn_r)
    for d2y in ys_d2:
        _draw_compartment(ax, x_d2, d2y, nr_a, syn_r)
    for d1y in ys_d1:
        _draw_compartment(ax, x_d1, d1y, nr_a, syn_r)

    # ---------------- Soma + output -----------------
    # Gold ring matches the highlighted unit in panel B
    ax.add_patch(Circle((x_soma, y_soma), sr_a + 0.18,
                        fc=GOLD_FACE, ec=GOLD_EDGE,
                        lw=1.8, zorder=9))
    ax.add_patch(Circle((x_soma, y_soma), sr_a,
                        fc=SOMA_FACE, ec=SOMA_EDGE, lw=1.6, zorder=10))
    ax.text(x_soma, y_soma + sr_a + 1.6, r"$V_0$", ha="center", va="bottom",
            fontsize=7.0, color="#111", fontweight="bold", zorder=11)
    ax.annotate("", xy=(x_soma + sr_a + 1.5, y_soma),
                xytext=(x_soma + sr_a + 0.1, y_soma),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))
    ax.text(x_soma + sr_a + 1.7, y_soma, r"$\hat{y}$",
            ha="left", va="center", fontsize=7.4,
            color="#333", fontweight="bold")

    # Depth tags
    for xp, lbl in [(x_d3, r"$\ell\!=\!3$"), (x_d2, r"$\ell\!=\!2$"),
                    (x_d1, r"$\ell\!=\!1$")]:
        ax.text(xp, -2.1, lbl, ha="center", fontsize=5.6, color="#777")
    ax.text(x_soma + 0.7, -2.1, "soma", ha="center", fontsize=5.6, color="#555")

    # Compact axes-coords legend (top right): E and I synapse markers
    ax.scatter([0.60, 0.78], [0.94, 0.94], s=22, c=[EXC, INH],
               transform=ax.transAxes, clip_on=False, zorder=20)
    ax.text(0.63, 0.94, "E", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.0, color=EXC,
            fontweight="bold")
    ax.text(0.81, 0.94, "I", transform=ax.transAxes,
            ha="left", va="center", fontsize=7.0, color=INH,
            fontweight="bold")


# ---------------------------------------------------------------------------
# Panel B: Network layer
# ---------------------------------------------------------------------------

def panel_b(ax):
    """E and I input pools feed N=4 dendritic units; per-unit somas project
    to a task readout; delta_0 exits on the right as the broadcast source."""
    ax.set_title("Network layer", fontsize=9.0, pad=4.0,
                 fontweight="bold", loc="center")
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    # E / I pools
    _round_box(ax, (0.02, 0.62), 0.16, 0.11,
               fc="#eef7ef", ec=EXC, text="E pool",
               color=EXC, fontsize=6.4)
    _round_box(ax, (0.02, 0.27), 0.16, 0.11,
               fc="#fdecec", ec=INH, text="I pool",
               color=INH, fontsize=6.4)

    e_bus_x, i_bus_x = 0.245, 0.275
    unit_x = 0.450
    out_x = 0.620
    out_bus_x = 0.730
    readout_x0 = 0.745
    readout_w = 0.140
    delta_x = 0.952
    ys = np.array([0.82, 0.62, 0.42, 0.22])
    highlight_idx = 1  # second unit zooms into Panel A

    # Pool -> bus arrows
    ax.annotate("", xy=(e_bus_x, 0.675), xytext=(0.18, 0.675),
                arrowprops=dict(arrowstyle="-|>", color=EXC, lw=0.95,
                                shrinkA=1, shrinkB=1))
    ax.annotate("", xy=(i_bus_x, 0.325), xytext=(0.18, 0.325),
                arrowprops=dict(arrowstyle="-|>", color=INH, lw=0.95,
                                shrinkA=1, shrinkB=1))
    # Vertical buses
    ax.plot([e_bus_x, e_bus_x], [0.20, 0.84],
            color=EXC, lw=0.85, alpha=0.70, zorder=2)
    ax.plot([i_bus_x, i_bus_x], [0.20, 0.84],
            color=INH, lw=0.85, alpha=0.70, zorder=2)

    # Per-unit fan-out and mini-tree
    for k, y in enumerate(ys, start=1):
        ax.annotate("", xy=(unit_x - 0.066, y + 0.018),
                    xytext=(e_bus_x, y + 0.032),
                    arrowprops=dict(arrowstyle="-|>", color=EXC, lw=0.75,
                                    alpha=0.85, shrinkA=1, shrinkB=1))
        ax.annotate("", xy=(unit_x - 0.066, y - 0.018),
                    xytext=(i_bus_x, y - 0.032),
                    arrowprops=dict(arrowstyle="-|>", color=INH, lw=0.75,
                                    alpha=0.85, shrinkA=1, shrinkB=1))
        is_highlight = (k - 1) == highlight_idx
        _draw_mini_tree(ax, unit_x, y, scale=1.20, highlight=is_highlight)
        # unit -> per-unit output -> readout bus
        ax.annotate("", xy=(out_x - 0.030, y), xytext=(unit_x + 0.060, y),
                    arrowprops=dict(arrowstyle="-|>", color=MUTE,
                                    lw=0.8, shrinkA=1, shrinkB=1))
        ax.add_patch(Circle((out_x, y), 0.030,
                            fc="#f1f1f1", ec="#666", lw=0.8, zorder=6))
        ax.text(out_x, y, rf"$y_{k}$",
                ha="center", va="center", fontsize=5.6,
                color="#444", zorder=7)
        ax.plot([out_x + 0.030, out_bus_x], [y, y],
                color=MUTE, lw=0.65, alpha=0.80, zorder=2)

    ax.plot([out_bus_x, out_bus_x], [ys[-1], ys[0]],
            color=MUTE, lw=0.75, alpha=0.85, zorder=2)

    # Task readout block
    _round_box(ax, (readout_x0, 0.38), readout_w, 0.24,
               fc="#f6f6f6", ec="#666",
               text="task\nreadout", color="#444",
               fontsize=5.0, lw=0.85)
    ax.annotate("", xy=(readout_x0, 0.50), xytext=(out_bus_x, 0.50),
                arrowprops=dict(arrowstyle="-|>", color=MUTE,
                                lw=0.85, alpha=0.9, shrinkA=1, shrinkB=1))

    # delta_0 callout: short red arrow exits the readout, ends in a filled
    # circle.  This is the broadcast source picked up by panel C.  The
    # label is right-aligned at the dot so it cannot clip the panel edge.
    ax.annotate("", xy=(delta_x - 0.014, 0.50),
                xytext=(readout_x0 + readout_w, 0.50),
                arrowprops=dict(arrowstyle="-|>", color=DELTA_COLOR,
                                lw=1.1, shrinkA=1, shrinkB=1))
    ax.add_patch(Circle((delta_x, 0.50), 0.018,
                        fc=DELTA_COLOR, ec="white", lw=0.5, zorder=10))
    ax.text(delta_x, 0.50 + 0.05, r"$\delta_0$",
            ha="center", va="bottom",
            fontsize=7.5, color=DELTA_COLOR, fontweight="bold", zorder=11)

    ax.text(unit_x, 0.10, r"$N$ dendritic E/I units",
            ha="center", va="center", fontsize=6.0, color="#555")


# ---------------------------------------------------------------------------
# Panel C: Credit assignment + broadcast modes
# ---------------------------------------------------------------------------

def panel_c(ax):
    """Mechanism panel: path-specific compartment errors, low-bandwidth
    broadcast estimators, and the exact-vs-LocalCA factorization."""
    ax.set_title("Credit: eligibility $\\times$ error",
                 fontsize=9.0, pad=4.0, fontweight="bold", loc="center")
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    # =========  TOP LEFT: exact path errors vs. rank-1 broadcast  =========
    leaf_x = 0.10
    mid_x = 0.25
    soma_x = 0.45
    branch_ys = np.array([0.82, 0.68, 0.54])
    leaf_offsets = np.array([0.035, -0.035])
    leaf_r = 0.008
    mid_r = 0.016
    soma_r = 0.027
    soma_y = 0.68

    branch_cols = ["#D95F4B", "#58A66E", "#4F79B8"]

    for bi, by in enumerate(branch_ys):
        for off in leaf_offsets:
            ax.plot([leaf_x + leaf_r, mid_x - mid_r],
                    [by + off, by],
                    color=branch_cols[bi], lw=0.9, alpha=0.70,
                    solid_capstyle="round", zorder=2)
        ax.plot([mid_x + mid_r, soma_x - soma_r],
                [by, soma_y + 0.15 * (by - soma_y)],
                color=branch_cols[bi], lw=1.5, alpha=0.90,
                solid_capstyle="round", zorder=2)

    for bi, by in enumerate(branch_ys):
        for off in leaf_offsets:
            ax.add_patch(Circle((leaf_x, by + off), leaf_r,
                                fc=LEAF_FACE, ec=LEAF_EDGE,
                                lw=0.5, zorder=4))
        ax.add_patch(Circle((mid_x, by), mid_r,
                            fc=branch_cols[bi], ec=EDGE,
                            lw=0.55, zorder=5))
    ax.add_patch(Circle((soma_x, soma_y), soma_r + 0.005,
                        fc=GOLD_FACE, ec=GOLD_EDGE, lw=1.0, zorder=6))
    ax.add_patch(Circle((soma_x, soma_y), soma_r,
                        fc=SOMA_FACE, ec=SOMA_EDGE, lw=0.9, zorder=7))

    # Somatic error source.
    delta_x = 0.55
    ax.add_patch(Circle((delta_x, soma_y), 0.018,
                        fc=DELTA_COLOR, ec="white", lw=0.4, zorder=10))
    ax.text(delta_x + 0.022, soma_y, r"$\delta_0$",
            ha="left", va="center",
            fontsize=6.2, color=DELTA_COLOR, fontweight="bold")
    ax.annotate("", xy=(soma_x + soma_r, soma_y),
                xytext=(delta_x - 0.018, soma_y),
                arrowprops=dict(arrowstyle="-|>", color=DELTA_COLOR,
                                lw=1.0, shrinkA=0, shrinkB=0))

    # Exact compartment errors: dashed path-specific transport.
    for bi, by in enumerate(branch_ys):
        rad = 0.18 if by > soma_y else (-0.18 if by < soma_y else 0)
        ax.add_patch(FancyArrowPatch(
            (soma_x - 0.005, soma_y), (mid_x + mid_r + 0.005, by),
            arrowstyle="-|>", mutation_scale=6.0,
            linewidth=1.0, color=DELTA_COLOR,
            linestyle=(0, (3.5, 2.2)), alpha=0.78,
            shrinkA=2, shrinkB=2, zorder=5,
            connectionstyle=f"arc3,rad={rad}",
        ))
        ax.text(mid_x + 0.032, by + (0.035 if bi == 0 else -0.038 if bi == 2 else 0.033),
                rf"$\delta_{bi+1}=\alpha_{bi+1}\delta_0$",
                ha="left", va="center", fontsize=4.8,
                color=branch_cols[bi], fontweight="bold", zorder=11)

    # Uniform rank-1 broadcast: same field to every branch.
    bar_x = leaf_x - 0.045
    bar_top = branch_ys[0] + leaf_offsets[0] + 0.020
    bar_bot = branch_ys[-1] - leaf_offsets[0] - 0.020
    ax.add_patch(FancyBboxPatch(
        (bar_x, bar_bot), 0.016, bar_top - bar_bot,
        boxstyle="round,pad=0.0,rounding_size=0.008",
        fc=BROADCAST_COLOR, ec="none", alpha=0.75, zorder=3,
    ))
    for by in branch_ys:
        ax.annotate("", xy=(leaf_x - leaf_r - 0.003, by),
                    xytext=(bar_x + 0.016, by),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=BROADCAST_COLOR,
                                    lw=0.9, alpha=0.85,
                                    shrinkA=0, shrinkB=0))
    ax.text(bar_x + 0.008, bar_top + 0.016,
            "shared $e_n$",
            ha="center", va="bottom",
            fontsize=5.0, color=BROADCAST_COLOR,
            fontweight="bold")

    ax.text(0.36, 0.93, "path-specific exact errors",
            ha="center", va="center", fontsize=5.6,
            color=DELTA_COLOR, fontweight="bold")
    # =========  TOP RIGHT: broadcast-mode cards  =========
    ax.text(0.78, 0.93, "broadcast estimator",
            ha="center", va="center",
            fontsize=6.4, color=INK, fontweight="bold")

    modes = [
        ("Scalar",   "global shared",       MODE_COLORS["rank1"],  "rank1"),
        ("Per-soma", "one per neuron",       MODE_COLORS["persoma"], "persoma"),
        ("Rank-$K$", "$K$ channels",         MODE_COLORS["rankk"],  "rankk"),
        ("Path",     "branch roles",         MODE_COLORS["path"],   "path"),
        ("Oracle", r"$\tilde{\alpha}_n\delta_0$", MODE_COLORS["oracle"], "oracle"),
    ]
    row_x = 0.62
    row_w = 0.34
    row_h = 0.070
    row_ys = np.linspace(0.84, 0.54, len(modes))
    for (name, note, col, kind), ry in zip(modes, row_ys):
        _round_box(ax, (row_x, ry - row_h / 2), row_w, row_h,
                   fc="white", ec="#cfd4dc", lw=0.7, fontsize=5.6,
                   zorder=2)
        ax.add_patch(Circle((row_x + 0.030, ry), 0.014,
                            fc=col, ec="white", lw=0.4, zorder=8))
        ax.text(row_x + 0.055, ry + 0.015, name,
                ha="left", va="center",
                fontsize=5.6, color=col, fontweight="bold", zorder=10)
        ax.text(row_x + 0.055, ry - 0.019, note,
                ha="left", va="center",
                fontsize=4.7, color=MUTE, zorder=10)
        ic_x = row_x + row_w - 0.105
        ic_w = 0.075
        ap = dict(shrinkA=0, shrinkB=0)
        if kind == "rank1":
            ax.annotate("", xy=(ic_x + ic_w, ry), xytext=(ic_x, ry),
                        arrowprops=dict(arrowstyle="-|>", color=col,
                                        lw=1.0, **ap),
                        zorder=10)
        elif kind == "rankk":
            for dy in (-0.018, 0.0, 0.018):
                ax.annotate("", xy=(ic_x + ic_w, ry + dy),
                            xytext=(ic_x, ry + dy),
                            arrowprops=dict(arrowstyle="-|>", color=col,
                                            lw=0.75, alpha=0.85, **ap),
                            zorder=10)
        elif kind == "path":
            ax.plot([ic_x, ic_x + ic_w * 0.55], [ry, ry],
                    color=col, lw=1.0, zorder=10)
            for dy in (-0.018, 0.018):
                ax.annotate("", xy=(ic_x + ic_w, ry + dy),
                            xytext=(ic_x + ic_w * 0.55, ry),
                            arrowprops=dict(arrowstyle="-|>", color=col,
                                            lw=0.85, **ap),
                            zorder=10)
        else:
            for dy in (-0.018, 0.0, 0.018):
                ax.annotate("", xy=(ic_x + ic_w, ry + dy),
                            xytext=(ic_x, ry + dy),
                            arrowprops=dict(arrowstyle="-|>", color=col,
                                            lw=0.75, alpha=0.9,
                                            linestyle="--", **ap),
                            zorder=10)

    # =========  BOTTOM: two-row factorization band  =========
    # Name the two factors once, above the boxes, so both rules read as
    # gradient = local eligibility x compartment error; only the second
    # factor (exact delta_n vs broadcast e_n) changes between the rows.
    ax.text(0.3875, 0.400, "local eligibility (shared)",
            ha="center", va="center", fontsize=5.4,
            color=COLORS["soma"], fontweight="bold")
    ax.text(0.725, 0.400, "compartment error",
            ha="center", va="center", fontsize=5.4,
            color=INK, fontweight="bold")

    def equation_row(y, label, rhs, rhs_color):
        _round_box(ax, (0.035, y), 0.93, 0.125,
                   fc="#F8FAFC", ec="#CBD5E1", lw=0.65,
                   text=None, zorder=2)
        ax.text(0.055, y + 0.062, label,
                ha="left", va="center", fontsize=5.3,
                color=INK, fontweight="bold", zorder=10,
                linespacing=0.95)
        _round_box(ax, (0.245, y + 0.025), 0.285, 0.075,
                   fc="#FFF7ED", ec="#FDBA74", lw=0.65,
                   text=r"$x_iR_n^{\mathrm{tot}}(E_i\!-\!V_n)$",
                   color=COLORS["soma"], fontsize=5.5,
                   fontweight="bold", zorder=5)
        ax.text(0.565, y + 0.062, r"$\times$",
                ha="center", va="center", fontsize=6.4,
                color=INK, zorder=10)
        _round_box(ax, (0.600, y + 0.025), 0.250, 0.075,
                   fc="#FEF2F2" if rhs_color == DELTA_COLOR else "#FFFBEB",
                   ec=rhs_color, lw=0.65,
                   text=rhs, color=rhs_color, fontsize=5.5,
                   fontweight="bold", zorder=5)

    equation_row(0.235, "Exact\n$\\partial L/\\partial g_i=$",
                 r"$\delta_n=\tilde{\alpha}_n\delta_0$", DELTA_COLOR)
    equation_row(0.075, "LocalCA\n$\\Delta g_i\\propto$",
                 r"$e_n\approx\delta_n$", BROADCAST_COLOR)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.0, 2.6))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.50, 2.05, 3.25],
        wspace=0.14,
        left=0.025, right=0.99, top=0.86, bottom=0.06,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    for ax, lbl, x_off, y_off in [(ax_a, "A", -0.06, 1.22),
                                  (ax_b, "B", -0.04, 1.22),
                                  (ax_c, "C", -0.03, 1.22)]:
        panel_label(ax, lbl, x=x_off, y=y_off, fontsize=12.0)

    out = OUTPUT_DIR / "fig1_model_and_credit"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()

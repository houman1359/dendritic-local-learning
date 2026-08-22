#!/usr/bin/env python3
"""Main Figure 5 as ONE native full-width canvas.

Physical depth and the point/dendrite controls used to live in four
pre-rendered sub-blocks (``fig_nonlinear_physical_depth``,
``fig_point_dendrite_credit_controls``, ``fig_physical_alignment_dose`` and
``fig_remaining_physical_crossovers``) that the compositor scaled into grid
slots.  Every block was scaled by a different factor, so one figure carried
eight different type sizes and a dozen different stroke weights.  This module
rebuilds the figure natively at scale 1.0 on a single 12-column module grid,
so a 7.6 pt tick label is 7.6 pt and an ``LW_EDGE`` spine is 0.7 pt.

Nothing here recomputes an estimate.  Every mean, 95 % interval and paired
contrast is read from the frozen source-data tables that the confirmatory
analyzers wrote (``source_data/nonlinear_physical_depth_confirmatory`` and
``source_data/point_dendrite_credit_controls``); this file only decides where
the numbers sit on the page and how they are inked.

Layout (12 modules, three rows, three column edges)::

    A  matched-resource depth   C  backprop depth test  D  local credit
    B  nested divisive task     E  divisive control      F  serial composition
                                G  paired effect sizes   H  architecture

Every panel spans four modules, so every panel of a row owns an identical
axes box and the figure starts at one of three column edges: the schematic
column at module 0, the shared-axis block at modules 4 and 8.  C, D, E and F
are one shared-axis small-multiple block -- identical y range, ticks and
units, with the tick column and y label drawn once per row.  G's contrast
names are not a slice of its panel: they are drawn into the free left cell of
the bottom row, so the forest keeps the same axes box as every other panel
instead of printing two thirds as wide, and its group rule stands on the
module-0 edge that A and B start from.

Marker dodges, series coincidences and the contrast definitions are stated in
the caption rather than inside the panels: the geometry is the report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from journal_style import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
)
from figure_canvas import Margins, NativeCanvas  # noqa: E402
from credit_tree_schematics import MS_JUNCTION, mix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEPTH_SOURCE = ROOT / "source_data" / "nonlinear_physical_depth_confirmatory"
POINT_SOURCE = ROOT / "source_data" / "point_dendrite_credit_controls"
OUT = ROOT / "figures" / "components" / "main_figure_05_native.pdf"

# ── canvas geometry, in points ───────────────────────────────────────────
# 410 pt is the tallest this figure may print: its caption is long, and a
# taller canvas pushes the float off the page ("Float too large") in main.tex.
CANVAS_H_PT = 396.0                      # 518.4 / 396 = 1.31 aspect
ROW_H_PT = (89.0, 89.0, 96.0)
# The horizontal gutter carries the shared-axis block's y label and tick
# column AND the panel letter that sits left of it, and the vertical gutter
# carries a row's x label band plus the next row's letter and title band, so
# no panel has to carve a reserve out of its own slot and every panel of a
# grid column keeps one x0 and one axes width.
HGUTTER = 38.0
VGUTTER = 38.0
MARGINS = Margins(left=44.0, right=13.0, top=18.0, bottom=30.0)

# ── the shared accuracy axis (identical in C, D, E and F) ────────────────
ACC_YLIM = (0.44, 1.06)
ACC_YTICKS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ACC_XLIM = (0.66, 3.34)
ACC_XTICKS = (1, 2, 3)
ACC_XLABEL = "physical stage count Dₚ"
ACC_YLABEL = "test accuracy"

MINUS = "−"

# ── colour semantics (route of interest coloured, controls neutral) ──────
C_ROUTE = COLORS["shunting"]             # aligned / shunting / serial tree
C_ADDITIVE = COLORS["additive"]          # raw additive divisive control
C_SHUFFLE = COLORS["highlight"]          # sensor shuffled owns the rose slot
C_CONTROL = COLORS["point_mlp"]          # control gray
C_CONTROL_LIGHT = mix(COLORS["point_mlp"], 72)   # second control lightness
C_SHARED = COLORS["local"]               # shared-soma (scalar/local) amber
C_PATH = COLORS["oracle"]                # exact path transport violet
INK = COLORS["ink"]
MUTE = COLORS["mute"]
DEND = COLORS["dend"]

# Library stroke taper (credit_tree_schematics): terminal branches print at
# 0.70 TikZ pt, thickening toward the trunk; normalized so 1.60 pt == LW_DATA.
_TAPER_FROM_LEAF = (0.70, 0.90, 1.20, 1.60)
_PT2LW = LW_DATA / 1.60


# ── data access (frozen tables only; nothing is recomputed here) ─────────
def load_tables():
    depth_summary = pd.read_csv(DEPTH_SOURCE / "condition_summary.csv")
    depth_contrasts = pd.read_csv(DEPTH_SOURCE / "paired_contrasts.csv")
    point_summary = pd.read_csv(POINT_SOURCE / "condition_summary.csv")
    point_contrasts = pd.read_csv(POINT_SOURCE / "paired_contrasts.csv")
    return depth_summary, depth_contrasts, point_summary, point_contrasts


def _rows(frame, **filters):
    part = frame
    for column, value in filters.items():
        part = part[part[column].eq(value)]
    return part.sort_values("depth")


def series(ax, frame, *, color, marker, filled=True, dashes=None, dx=0.0,
           label=None, handles=None, **filters):
    """Seed-mean line at true x with 95 % seed-bootstrap interval bars.

    ``dx`` moves only the marker/error-bar column so honestly coincident
    series stay separable, and ``dashes`` interleaves the dash phase of two
    curves that genuinely lie on top of one another; both are declared inside
    the panel that uses them.  Values, intervals and n are the frozen table's.
    """
    part = _rows(frame, **filters)
    x = part.depth.to_numpy(float)
    mean = part.mean_test_accuracy.to_numpy(float)
    low = part.ci95_low_test_accuracy.to_numpy(float)
    high = part.ci95_high_test_accuracy.to_numpy(float)
    face = color if filled else "white"
    style = "-" if dashes is None else dashes
    # Trim the polyline to the dodged marker centre instead of letting it run
    # past: every drawn vertex stays at the table's own x, so no segment of
    # the curve is given a distorted run, and no bare stub hangs off the end.
    xs_line, ys_line = x, mean
    if dx:
        edge = (x[0] if dx > 0 else x[-1]) + dx
        keep = x > edge if dx > 0 else x < edge
        y_edge = float(np.interp(edge, x, mean))
        xs_line = (np.append(edge, x[keep]) if dx > 0
                   else np.append(x[keep], edge))
        ys_line = (np.append(y_edge, mean[keep]) if dx > 0
                   else np.append(mean[keep], y_edge))
    ax.plot(xs_line, ys_line, color=color, lw=LW_DATA, ls=style, zorder=2,
            solid_capstyle="round")
    ax.errorbar(
        x + dx, mean, yerr=np.vstack([mean - low, high - mean]),
        fmt=marker, ms=MARKER_MS, color=color, markerfacecolor=face,
        markeredgecolor=color if not filled else "white",
        markeredgewidth=LW_EDGE if not filled else 0.55,
        ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=3,
    )
    if handles is not None and label:
        handles.append(Line2D(
            [], [], color=color, lw=LW_DATA, ls=style, marker=marker,
            ms=MARKER_MS, markerfacecolor=face,
            markeredgecolor=color if not filled else "white",
            markeredgewidth=LW_EDGE if not filled else 0.55, label=label))
    return part


def accuracy_axis(ax, *, left=True, bottom=True):
    ax.set_xlim(*ACC_XLIM)
    ax.set_xticks(list(ACC_XTICKS))
    ax.set_ylim(*ACC_YLIM)
    ax.set_yticks(list(ACC_YTICKS))
    if left:
        ax.set_ylabel(ACC_YLABEL)
    else:
        ax.tick_params(axis="y", labelleft=False)
    if bottom:
        ax.set_xlabel(ACC_XLABEL)
    return ax


def key(ax, handles, *, loc="upper left", bbox=(0.0, 1.0)):
    legend = ax.legend(handles=handles, loc=loc, bbox_to_anchor=bbox,
                       frameon=False, fontsize=PT_LEGEND, handlelength=2.0,
                       handletextpad=0.5, labelspacing=0.28, borderpad=0.1,
                       borderaxespad=0.0)
    for text in legend.get_texts():
        text.set_color(INK)
    return legend


# ── schematic vocabulary, drawn in the panel's own point frame ───────────
def point_frame(ax):
    """Give ``ax`` a 1:1 point coordinate frame filling its own box."""
    fig = ax.get_figure()
    fw, fh = fig.get_size_inches()
    box = ax.get_position()
    w_pt = box.width * fw * 72.0
    h_pt = box.height * fh * 72.0
    ax.set_xlim(0.0, w_pt)
    ax.set_ylim(0.0, h_pt)
    ax.set_axis_off()
    ax.set_facecolor("none")
    return w_pt, h_pt


def draw_tree(ax, factors, *, x0, y0, radius, sector_deg, color=DEND,
              soma_r=3.0):
    """Radial dendritic tree in the credit-tree library vocabulary.

    ``factors`` is ordered soma-to-distal, as in the production DendriNet
    planner.  Leaves spread evenly on an arc so an 8-way fan stays readable;
    stroke taper, white junction rings on internal nodes and the soma disc
    with its ink-mix rim come from ``credit_tree_schematics``.
    """
    n_leaves = int(np.prod(factors))
    leaf_angles = (
        np.deg2rad(np.linspace(-sector_deg / 2.0, sector_deg / 2.0, n_leaves))
        if n_leaves > 1 else np.zeros(1)
    )
    depth = len(factors)
    step = radius / depth
    # A trunk, as in the shared library: the arbor leaves the soma as one
    # LW_DATA stroke and tapers outward, instead of fanning straight off the
    # soma disc.  Without it these trees print lighter than figure 4's.
    trunk = 0.22 * step
    ax.plot([x0, x0], [y0, y0 + trunk], color=color,
            lw=_TAPER_FROM_LEAF[-1] * _PT2LW, solid_capstyle="round",
            zorder=3)
    current = [(x0, y0 + trunk, 0, n_leaves)]
    coords = [[(x0, y0 + trunk)]]
    for level, factor in enumerate(factors, start=1):
        lw = _TAPER_FROM_LEAF[min(depth - level,
                                  len(_TAPER_FROM_LEAF) - 1)] * _PT2LW
        r = level * step
        nxt = []
        for px, py, lo, hi in current:
            span = (hi - lo) // factor
            for child in range(factor):
                clo, chi = lo + child * span, lo + (child + 1) * span
                angle = float(leaf_angles[clo:chi].mean())
                cx = x0 + r * np.sin(angle)
                cy = y0 + r * np.cos(angle)
                ax.plot([px, cx], [py, cy], color=color, lw=lw,
                        solid_capstyle="round", zorder=3)
                nxt.append((cx, cy, clo, chi))
        coords.append([(nx, ny) for nx, ny, _, _ in nxt])
        current = nxt
    for level_nodes in coords[1:-1]:
        for nx, ny in level_nodes:
            ax.plot([nx], [ny], marker="o", ms=MS_JUNCTION, mfc="white",
                    mec=color, mew=LW_EDGE, ls="none", zorder=4)
    ax.add_patch(Circle((x0, y0), soma_r, facecolor=COLORS["soma"],
                        edgecolor=mix("ink", 30), lw=LW_EDGE, zorder=5))
    return coords


def bracket(ax, x0, x1, y, *, drop=3.0, color=MUTE):
    """Scaffolding brace naming a shared budget across the motifs beneath."""
    ax.plot([x0, x1], [y, y], color=color, lw=LW_HAIR, zorder=1,
            solid_capstyle="round")
    for x in (x0, x1):
        ax.plot([x, x], [y, y - drop], color=color, lw=LW_HAIR, zorder=1,
                solid_capstyle="round")


def scope_box(ax, points, pad, edge, *, fill="#20509E", alpha=0.05,
              round_pt=3.0):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs) - pad, max(xs) + pad
    y_lo, y_hi = min(ys) - pad, max(ys) + pad
    for face, edge_color, z in ((fill, "none", 1), ("none", edge, 2)):
        ax.add_patch(FancyBboxPatch(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            boxstyle=f"round,pad=1.0,rounding_size={round_pt}",
            facecolor=face, alpha=alpha if face != "none" else 1.0,
            edgecolor=edge_color, lw=LW_EDGE, zorder=z))
    return x_lo - 1.0, x_hi + 1.0, y_lo - 1.0, y_hi + 1.0


# ── panels ───────────────────────────────────────────────────────────────
def panel_depth_ladder(ax):
    """A: the three physical depth stages at one matched forward budget."""
    w, h = point_frame(ax)
    specs = ((8,), (2, 3), (2, 1, 2))
    sectors = (42.0, 38.0, 34.0)
    centres = [w * f for f in (0.165, 0.50, 0.835)]
    base_y = 0.225 * h
    radius = 0.495 * h
    for x, factors, sector in zip(centres, specs, sectors, strict=True):
        draw_tree(ax, list(factors), x0=x, y0=base_y, radius=radius,
                  sector_deg=sector)
    for x, factors, name in zip(centres, specs, ("1", "2", "3"), strict=True):
        ax.text(x, 0.120 * h, f"Dₚ = {name}", ha="center",
                va="center", fontsize=PT_SMALL, color=INK)
        ax.text(x, 0.030 * h, "[" + ",".join(map(str, factors)) + "]",
                ha="center", va="center", fontsize=PT_SMALL, color=MUTE)
    bracket(ax, w * 0.02, w * 0.98, 0.800 * h, drop=0.030 * h)
    # The brace names the matched budget in one line, exactly as panel H does;
    # what the three stages hold constant beyond the branch count (contacts and
    # parameters) is stated in the caption, not wrapped across the panel.
    ax.text(w * 0.5, 0.905 * h, "same 8 branch units",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)


STAGE_COLORS = ("#20509E", "#3F6BB0", "#5580BE")   # fine -> coarse -> global


def panel_task(ax):
    """B: the serial divisive chain, one gain sensor per physical stage."""
    w, h = point_frame(ax)
    box_x0, box_x1 = 0.033 * w, 0.545 * w
    centres = (0.785 * h, 0.585 * h, 0.385 * h)
    box_h = 0.135 * h
    stages = (
        ("G₁   fine", STAGE_COLORS[0]),
        ("G₂   coarse", STAGE_COLORS[1]),
        ("G₃   global", STAGE_COLORS[2]),
    )
    ax.text(0.5 * (box_x0 + box_x1), 0.945 * h, "class signal  s",
            ha="center", va="center", fontsize=PT_SMALL, color=INK)
    ax.annotate("", xy=(0.5 * (box_x0 + box_x1), centres[0] + box_h / 2.0),
                xytext=(0.5 * (box_x0 + box_x1), 0.900 * h),
                arrowprops=dict(arrowstyle="-|>,head_length=3.2,head_width=2.0",
                                mutation_scale=1.0, color=MUTE,
                                lw=LW_EDGE, shrinkA=0, shrinkB=0))
    sensor_x = 0.685 * w
    for index, ((label, color), y) in enumerate(zip(stages, centres,
                                                    strict=True)):
        ax.add_patch(FancyBboxPatch(
            (box_x0, y - box_h / 2.0), box_x1 - box_x0, box_h,
            boxstyle="round,pad=0,rounding_size=2.6",
            facecolor=mix(color, 9), edgecolor=color, lw=LW_EDGE, zorder=2))
        ax.text(0.5 * (box_x0 + box_x1), y, label, ha="center", va="center",
                fontsize=PT_SMALL, color=color, zorder=4)
        ax.plot([box_x1 + 1.5, sensor_x - 3.4], [y, y], color=MUTE,
                lw=LW_HAIR, zorder=1, solid_capstyle="round")
        ax.plot([sensor_x], [y], marker="o", ms=MS_JUNCTION + 0.6,
                mfc="white", mec=color, mew=LW_EDGE, ls="none", zorder=4)
        if index < 2:
            ax.annotate(
                "", xy=(0.5 * (box_x0 + box_x1), centres[index + 1]
                        + box_h / 2.0),
                xytext=(0.5 * (box_x0 + box_x1), y - box_h / 2.0),
                arrowprops=dict(
                    arrowstyle="-|>,head_length=3.2,head_width=2.0",
                    mutation_scale=1.0, color=MUTE, lw=LW_EDGE,
                    shrinkA=0, shrinkB=0))
    bracket_x = sensor_x + 0.055 * w
    ax.plot([bracket_x, bracket_x], [centres[2], centres[0]], color=MUTE,
            lw=LW_HAIR, zorder=1, solid_capstyle="round")
    for y in (centres[0], centres[2]):
        ax.plot([bracket_x - 0.022 * w, bracket_x], [y, y], color=MUTE,
                lw=LW_HAIR, zorder=1, solid_capstyle="round")
    # A one-word rail label, not a sentence: the drawing already shows one
    # sensor opposite each stage, and the caption says so in words.
    ax.text(bracket_x + 0.030 * w, centres[1], "sensors",
            ha="left", va="center", fontsize=PT_SMALL, color=MUTE)
    ax.annotate("", xy=(0.5 * (box_x0 + box_x1), 0.230 * h),
                xytext=(0.5 * (box_x0 + box_x1), centres[2] - box_h / 2.0),
                arrowprops=dict(arrowstyle="-|>,head_length=3.2,head_width=2.0",
                                mutation_scale=1.0, color=MUTE,
                                lw=LW_EDGE, shrinkA=0, shrinkB=0))
    ax.add_patch(Circle((0.5 * (box_x0 + box_x1), 0.180 * h), 0.045 * h,
                        facecolor=COLORS["soma"], edgecolor=mix("ink", 30),
                        lw=LW_EDGE, zorder=5))
    ax.text(0.5 * (box_x0 + box_x1) + 0.075 * w, 0.180 * h, "soma",
            ha="left", va="center", fontsize=PT_SMALL, color=MUTE)
    ax.text(0.5 * w, 0.045 * h, "distal drive = s · G₁ · G₂ · G₃",
            ha="center", va="center", fontsize=PT_ANNOT, color=INK)


def panel_architecture(ax):
    """H: the resource-identical architectures compared in F."""
    w, h = point_frame(ax)
    centres = [w * f for f in (0.16, 0.50, 0.84)]
    base_y = 0.235 * h
    radius = 0.52 * h
    # Point MLP: crossed dense layers, control gray.
    layer_y = (base_y, base_y + 0.26 * h, base_y + 0.52 * h)
    node_dx = 0.052 * w
    layers = []
    for y, count in zip(layer_y, (3, 4, 3), strict=True):
        layers.append([(centres[0] + off, y)
                       for off in np.linspace(-node_dx, node_dx, count)])
    for lower, upper in zip(layers[:-1], layers[1:], strict=True):
        for px, py in lower:
            for qx, qy in upper:
                ax.plot([px, qx], [py, qy], color=MUTE, lw=LW_HAIR,
                        zorder=2, solid_capstyle="round")
    for level in layers:
        for px, py in level:
            ax.plot([px], [py], marker="o", ms=MS_JUNCTION, mfc="white",
                    mec=C_CONTROL_LIGHT, mew=LW_EDGE, ls="none", zorder=4)
    draw_tree(ax, [4, 1], x0=centres[1], y0=base_y, radius=radius,
              sector_deg=34.0, color=C_CONTROL)
    draw_tree(ax, [2, 1, 2], x0=centres[2], y0=base_y, radius=radius,
              sector_deg=30.0, color=C_ROUTE)
    for x, label, color in (
        (centres[0], "point\nMLP", MUTE),
        (centres[1], "grouped\nstar", C_CONTROL),
        (centres[2], "serial\ntree", C_ROUTE),
    ):
        ax.text(x, 0.075 * h, label, ha="center", va="center",
                linespacing=1.15, fontsize=PT_SMALL, color=color)
    bracket(ax, centres[1] - 0.14 * w, centres[2] + 0.14 * w, 0.905 * h,
            drop=0.03 * h)
    ax.text((centres[1] + centres[2]) / 2.0, 0.962 * h,
            "same 8 branch modules", ha="center", va="center",
            fontsize=PT_SMALL, color=MUTE)


def panel_backprop_depth(ax, summary):
    """C: does physical depth help, and only when the hierarchy matches?"""
    handles = []
    # Zero alignment and sensor shuffled lie on top of one another to within
    # 0.01 pp, so their dash phases interleave and their marker columns are
    # dodged; the dodge is declared in the panel.
    spec = (
        ("aligned", "aligned serial tree", C_ROUTE, "o", True, None, -0.055),
        ("zero_alignment", "zero alignment", C_CONTROL_LIGHT, "v", False,
         (0.0, (1.4, 1.6)), -0.055),
        ("sensor_shuffled", "sensor shuffled", C_SHUFFLE, "^", False,
         (1.5, (1.4, 1.6)), 0.055),
        ("rewired_tree", "tree reversed", C_CONTROL, "D", False,
         (0.0, (1.4, 1.6)), 0.055),
    )
    for regime, label, color, marker, filled, dashes, dx in spec:
        series(ax, summary, color=color, marker=marker, filled=filled,
               dashes=dashes, dx=dx, label=label, handles=handles,
               regime=regime, mechanism="shunting", method="bp",
               transport="backpropagation")
    accuracy_axis(ax, left=True, bottom=True)
    key(ax, handles)


def panel_local_transport(ax, summary):
    """D: the same depth test under two local credit coordinates."""
    handles = []
    # All four series meet at one stage and the two reversed curves coincide
    # throughout, so the four marker columns are dodged and the reversed dash
    # phases interleave.
    spec = (
        ("per_soma_shared", "aligned", "shared soma, aligned", C_SHARED, "o",
         True, None, -0.150),
        ("path_transport", "aligned", "exact path, aligned", C_PATH, "^",
         True, None, -0.050),
        ("per_soma_shared", "rewired_tree", "shared soma, reversed", C_SHARED,
         "s", False, (0.0, (3.2, 3.2)), 0.050),
        ("path_transport", "rewired_tree", "exact path, reversed", C_PATH,
         "D", False, (3.2, (3.2, 3.2)), 0.150),
    )
    for transport, regime, label, color, marker, filled, dashes, dx in spec:
        series(ax, summary, color=color, marker=marker, filled=filled,
               dashes=dashes, dx=dx, label=label, handles=handles,
               regime=regime, mechanism="shunting", method="local3f",
               transport=transport)
    accuracy_axis(ax, left=False, bottom=True)
    key(ax, handles)


def panel_divisive(ax, summary):
    """E: the depth gain needs the divisive (shunting) mechanism."""
    series(ax, summary, color=C_ROUTE, marker="o", regime="aligned",
           mechanism="shunting", method="bp", transport="backpropagation")
    series(ax, summary, color=C_ADDITIVE, marker="s", regime="aligned",
           mechanism="additive", method="bp", transport="backpropagation")
    accuracy_axis(ax, left=True, bottom=True)
    ax.text(2.06, 0.800, "aligned serial tree", ha="right", va="bottom",
            fontsize=PT_LEGEND, color=C_ROUTE)
    ax.text(3.02, 0.492, "raw additive", ha="right", va="top",
            fontsize=PT_LEGEND, color=C_ADDITIVE)


def panel_serial(ax, summary):
    """F: serial composition, against the resource-identical grouped star."""
    series(ax, summary, color=C_ROUTE, marker="o", regime="aligned",
           architecture="serial_tree", credit="full_bp")
    series(ax, summary, color=C_CONTROL, marker="s", filled=False,
           dashes=(0.0, (3.2, 2.0)), regime="aligned",
           architecture="all_active_star", credit="full_bp")
    accuracy_axis(ax, left=False, bottom=True)
    ax.text(3.02, 0.952, "aligned serial tree", ha="right", va="bottom",
            fontsize=PT_LEGEND, color=C_ROUTE)
    ax.text(3.02, 0.575, "grouped star", ha="right", va="top",
            fontsize=PT_LEGEND, color=C_CONTROL)


# A row keeps its own condition's colour AND its own marker in every panel
# of the figure, and a control keeps the control's colour in both the level
# block and the interaction block, so one row label never carries two hues.
FOREST = (
    ("BP depth", (
        ("aligned", "bp_depth_d3_minus_d1__aligned", "depth", C_ROUTE, "o"),
        ("zero align", "bp_depth_d3_minus_d1__zero_alignment", "depth",
         C_CONTROL_LIGHT, "v"),
        ("shuffled", "bp_depth_d3_minus_d1__sensor_shuffled", "depth",
         C_SHUFFLE, "^"),
        ("reversed", "bp_depth_d3_minus_d1__rewired_tree", "depth",
         C_CONTROL, "D"),
    )),
    ("vs control", (
        ("zero align",
         "bp_depth_interaction__aligned_minus_zero_alignment", "depth",
         C_CONTROL_LIGHT, "v"),
        ("shuffled",
         "bp_depth_interaction__aligned_minus_sensor_shuffled", "depth",
         C_SHUFFLE, "^"),
        ("reversed",
         "bp_depth_interaction__aligned_minus_rewired_tree", "depth",
         C_CONTROL, "D"),
    )),
    ("LocalCA depth", (
        ("shared soma",
         "local_depth_d3_minus_d1__aligned__per_soma_shared", "depth",
         C_SHARED, "o"),
        ("exact path",
         "local_depth_d3_minus_d1__aligned__path_transport", "depth",
         C_PATH, "^"),
    )),
    ("architecture", (
        ("serial − star", "serial_minus_star__aligned__d3", "point", C_ROUTE,
         "o"),
    )),
)


def panel_forest(ax, depth_contrasts, point_contrasts, *, label_x, group_x,
                 rule_x):
    """G: every prespecified paired contrast on one effect-size axis.

    The contrast names are drawn as artists in the free left cell of the
    bottom row rather than as y tick labels, so they cost the panel no width:
    the forest keeps the same four-module axes box every other panel of the
    figure owns, and the group rule stands on the module-0 edge that the
    schematic column starts from.  ``label_x``, ``group_x`` and ``rule_x`` are
    axes fractions the caller derives from that module grid.
    """
    depth = depth_contrasts.set_index("contrast")
    point = point_contrasts.set_index("contrast")
    rows = [(name, key_, source, color, marker)
            for _, entries in FOREST
            for name, key_, source, color, marker in entries]
    n = len(rows)
    gap = 0.55                       # extra spacing between contrast families
    offsets = []
    step = 0.0
    for group_index, (_, entries) in enumerate(FOREST):
        for _ in entries:
            offsets.append(step)
        step += gap
    y_of = {i: n - 1 - i + (offsets[-1] - offsets[i]) for i in range(n)}
    for index, (name, key_, source, color, marker) in enumerate(rows):
        record = (depth if source == "depth" else point).loc[key_]
        mean = 100.0 * float(record.mean_difference)
        low = 100.0 * float(record.ci95_low)
        high = 100.0 * float(record.ci95_high)
        y = y_of[index]
        ax.errorbar(
            [mean], [y], xerr=np.array([[mean - low], [high - mean]]),
            fmt=marker, ms=MARKER_MS, color=color, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=0.55, ecolor=color,
            elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=3,
        )
        ax.text(label_x, y, name, ha="right", va="center", fontsize=PT_SMALL,
                color=INK, transform=ax.get_yaxis_transform(), clip_on=False)
    ax.axvline(0.0, color=MUTE, lw=LW_REF, ls=(0, (3.0, 2.2)), zorder=1)
    ax.set_yticks([])
    ax.set_ylim(-0.75, n - 1 + offsets[-1] + 0.75)
    ax.set_xlim(-3.0, 34.5)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_xlabel("paired difference (pp)")
    ax.tick_params(axis="y", length=0.0, pad=2.0)
    ax.spines["left"].set_visible(False)

    # Group scaffolding: a hairline rule on the module-0 edge and a mute name
    # beside it, so the rows read as four contrast families on one axis.
    start = 0
    for group, entries in FOREST:
        stop = start + len(entries)
        y_hi = y_of[start] + 0.42
        y_lo = y_of[stop - 1] - 0.42
        ax.plot([rule_x, rule_x], [y_lo, y_hi], color=MUTE, lw=LW_HAIR,
                clip_on=False, transform=ax.get_yaxis_transform(),
                solid_capstyle="round", zorder=1)
        ax.text(group_x, 0.5 * (y_lo + y_hi), group, ha="left", va="center",
                fontsize=PT_SMALL, color=MUTE, linespacing=1.2,
                transform=ax.get_yaxis_transform(), clip_on=False)
        start = stop


def build():
    depth_summary, depth_contrasts, point_summary, point_contrasts = \
        load_tables()

    canvas = NativeCanvas(
        CANVAS_H_PT / 72.0, 3, row_weights=list(ROW_H_PT),
        hgutter_pt=HGUTTER, vgutter_pt=VGUTTER, margins=MARGINS,
        letters=False,
    )

    # Column 0, rows 0 and 1: the two design schematics, immediately left of
    # the shared-axis accuracy block they explain.  No inset: the horizontal
    # gutter -- not a slice of these panels -- carries the block's y label,
    # its tick column and the panel letter that sits left of them.
    ax_a = canvas.panel("A", 0, 0, 4, schematic=True,
                        title="Matched-resource depth")
    ax_b = canvas.panel("B", 1, 0, 4, schematic=True,
                        title="Nested divisive task")

    # Rows 0-1, columns 4-11: one shared-axis small-multiple block.  Same y
    # range, ticks and units throughout; the tick column and label are drawn
    # once per row and the right-hand siblings drop them entirely.
    ax_c = canvas.panel("C", 0, 4, 4, grid="y", title="Backprop depth test")
    ax_d = canvas.panel("D", 0, 8, 4, grid="y", sharey=ax_c,
                        title="Local credit transport")
    ax_e = canvas.panel("E", 1, 4, 4, grid="y", title="Divisive control")
    ax_f = canvas.panel("F", 1, 8, 4, grid="y", sharey=ax_e,
                        title="Serial composition")

    # Row 2: the forest of paired effect sizes under the block it summarises,
    # and the architecture schematic directly under the serial-composition
    # result it explains.  The forest's contrast names occupy the free left
    # cell of this row instead of a reserve carved out of the panel.
    ax_g = canvas.panel("G", 2, 4, 4, grid="x", title="Paired effect sizes")
    ax_h = canvas.panel("H", 2, 8, 4, schematic=True,
                        title="Architecture controls")

    panel_depth_ladder(ax_a)
    panel_task(ax_b)
    panel_backprop_depth(ax_c, depth_summary)
    panel_local_transport(ax_d, depth_summary)
    panel_divisive(ax_e, depth_summary)
    panel_serial(ax_f, point_summary)
    panel_architecture(ax_h)

    # The name column of the forest is measured off the module grid: its rule
    # stands on the module-0 edge (the edge A and B start from) and its names
    # end just left of the axes, so the block spans the empty cell exactly.
    box_g = ax_g.get_position()
    g_x0_pt = box_g.x0 * canvas.width_pt
    g_w_pt = box_g.width * canvas.width_pt
    col0_pt = canvas.slot_pt(2, 0, 4)[0]
    panel_forest(ax_g, depth_contrasts, point_contrasts,
                 label_x=-4.0 / g_w_pt,
                 group_x=(col0_pt + 5.0 - g_x0_pt) / g_w_pt,
                 rule_x=(col0_pt - g_x0_pt) / g_w_pt)

    # Panel letters sit in a fixed gutter: wide where a y label and tick
    # column occupy it, narrow where a shared axis leaves it empty.  The
    # forest's letter stands on the same canvas x as the schematic column's,
    # at the left edge of the name block it belongs to.
    for name, dx in (("A", 30.0), ("B", 30.0), ("C", 34.0), ("D", 14.0),
                     ("E", 34.0), ("F", 14.0), ("H", 14.0)):
        canvas.add_letter(name, canvas.axes[name], dx_pt=dx)
    canvas.add_letter("G", ax_g, dx_pt=g_x0_pt - (col0_pt - 30.0))

    problems = canvas.save(OUT, name="main_figure_05_native")
    return problems


if __name__ == "__main__":
    issues = build()
    if issues:
        raise SystemExit(f"main_figure_05_native: {len(issues)} layout issues")

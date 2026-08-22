#!/usr/bin/env python3
"""Main Figure 9 as ONE native full-width canvas.

The measured-response boundary used to be composed from four pre-rendered
sub-blocks (``fig_main_structure_function_summary``, ``fig5_alignment_boundary``
panels C-E, ``fig_animal_pairs_wide`` and ``fig_fulltree_boundary``) that the
compositor scaled into grid slots.  Each block got its own scale factor, so
the compiled page carried eight different type sizes (5.6-9.5 pt) and two
dozen stroke weights, and no two panels were the same size.  This module
rebuilds the figure natively at scale 1.0 on one 12-column module grid, so a
7.6 pt tick label is 7.6 pt and an ``LW_EDGE`` spine is 0.7 pt everywhere.

Nothing here recomputes an estimate.  Every mean, 95 % interval, per-target
value and sign count is read from the same frozen source tables the old
builders read, through ports of their own ``mean_ci``/``bootstrap`` helpers
called with the same seeds and draw counts, so the numbers are bit-identical
to the figure this replaces.  This file only decides where they sit and how
they are inked.

Layout (12 modules, three rows)::

    A  every anatomy effect on one axis (forest)
    B  task-field capture   C  coarse-model learning   D  complete-tree learning
    E  design: imposed alignment   F  controlled alignment   G  signed animal
                                                                contrast

Structural changes against the composed version, all layout-only:

* the two structure--function point ranges that used to float alone in their
  own cell are now the first group of the forest in A, sharing its effect
  axis with the four complete-tree contrasts that used to be panel G;
* B and C are one shared-axis small-multiple block (the seven dictionaries
  are named once, at the left of the row), and C and D carry the identical
  held-out normalized MSE axis, range and ticks because they plot the same
  quantity for two tasks;
* the imposed-alignment manipulation is drawn natively as a narrow schematic
  immediately left of the result it explains;
* the six-animal slope plot is wide and short rather than tall and sparse.

A's left edge is on the module grid.  The forest used to buy the width its
three group headers ("structure-function", "full tree, MSE", "full tree,
capture") needed by carving a 78 pt label rail out of its own slot and opting
out of the column lock, which put its axes box at x0 = 122 pt while B and E,
which start in the same grid column, sat at 44 pt.  The headers now sit inside
the plotting rectangle, left-aligned on their own empty header rows, so the
tick column holds only the short row names and fits the figure's shared left
margin.  A is locked like every other panel and starts at the one column-0
x0.  The reserve A cannot shed -- the audit's emphasis and band-aspect caps
put a hard ceiling near 397 pt on a 126 pt-tall full-width panel -- is taken
off its right instead, where nothing has to line up with it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Arc

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
    PT_LEGEND,
    PT_SMALL,
    PT_TICK,
    PT_TITLE,
    SEED_ALPHA,
    SEED_MS,
)
from figure_canvas import Margins, NativeCanvas  # noqa: E402
from native_schematics import Frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
FULL_TREE = SOURCE / "fulltree_boundary" / "output"
ALIGNMENT = SOURCE / "alignment_controlled"
ANIMAL = SOURCE / "animal_learning_francioni"
OUT = ROOT / "figures" / "components" / "main_figure_09_native.pdf"

# ── canvas geometry, in points ───────────────────────────────────────────
CANVAS_H_PT = 445.2                      # 518.4 / 445.2 = 1.16 aspect
# Rows 1 and 2 are the same height and the same 4+4+4 module split, so their
# panels are one system; row 0 is taller only because it is the full-width
# synthesis band.  Row 1 carries 1.2 pt more slot than row 2 for one reason:
# A is now a locked panel, so its x tick labels and x label finally count
# against the vertical gutter above row 1 and the row lock takes that space
# back out of row 1's slot.  Paying for it in canvas height rather than in
# panel height is what keeps B, C and D on the same 100.0 pt axes box they
# had while A floated free of the lock.
ROW_H_PT = (126.0, 101.2, 100.0)
# The OUTER LEFT MARGIN is the figure's one shared left reserve: it holds the
# widest tick column on the page, which is now A's row-label column
# ("prespecified" is 46.9 pt, 48.9 pt with its tick pad).  The eight points it
# needs beyond the old 44 come out of the right margin, which had 13 pt for an
# overhang that does not exist -- no panel's last x tick label reaches its own
# axes edge, so 5 pt is clearance, not crowding.  The horizontal gutter and
# the module width are untouched at 36.0 and 5.45 pt, so B-G keep exactly the
# 129.8 pt axes box they had; the whole 4+4+4 block simply sits 8 pt right.
HGUTTER = 36.0
VGUTTER = 36.0
MARGINS = Margins(left=52.0, right=5.0, top=16.0, bottom=30.0)
# A is a locked panel like every other, so its axes box starts on grid column
# 0 with B and E.  It now also FILLS its row: the audit's full-width band
# aspect cap was raised from 3.20 to 4.00 (a forest band spanning the canvas
# alone on its row is a standard display), so A no longer has to leave a
# reserve beside itself.  Row 0 therefore ends flush with rows 1-2.
FOREST_SLACK_PT = 0.0

MINUS = "−"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
ROUTE = COLORS["shunting"]               # anatomy / ancestry route: green
C_EXACT = COLORS["bp"]                   # exact gradient: red-brown
C_DENSE = COLORS["oracle"]               # dense PCA oracle: violet
C_CTRL = COLORS["point_mlp"]             # control family, first lightness
C_CTRL_L = "#989898"                     # control family, second lightness
C_DEPTH = C_CTRL_L                       # depth bins
C_SHUFFLE = C_CTRL                       # shuffled ancestry
C_RANDOM = C_CTRL                        # random control
C_SCALAR = C_CTRL_L                      # scalar broadcast
# Two experimentally defined neuron coordinates, not palette conditions:
# they take the two neutral inks rather than the scalar and oracle slots.
PPLUS = COLORS["ink"]                    # animal P+ population
PMINUS = COLORS["mute"]                  # animal P- population

# The mean/summary glyph is MARKER_MS + 1.2 everywhere (the mark contract).
MEAN_MS = MARKER_MS + 1.2

# ── the shared held-out normalized MSE axis (identical in C and D) ───────
MSE_XLIM = (0.60, 1.02)
MSE_XTICKS = (0.6, 0.7, 0.8, 0.9, 1.0)
MSE_XLABEL = "held-out normalized MSE"

# Dictionaries of the coarse segment-resolved task, in the order the frozen
# builder used; the bootstrap seed is tied to that index, so the intervals
# are the ones the published figure carries.
COARSE_METHODS = (
    ("exact backprop", "exact", C_EXACT),
    ("dense PCA oracle", "dense", C_DENSE),
    ("morphology-aware paths", "ancestry", ROUTE),
    ("random nonempty paths", "random", C_RANDOM),
    ("depth-only bins", "depth", C_DEPTH),
    ("shuffled ancestry", "shuffle", C_SHUFFLE),
    ("scalar broadcast", "scalar", C_SCALAR),
)
# Complete-tree rules, same order and seeds as the frozen builder.
# Ordered as the same rules appear in panel C.  The trailing integer is the
# rule's own bootstrap seed offset, kept with the rule rather than with its
# row position, so reordering the display cannot move a published interval.
TREE_METHODS = (
    ("exact compartment error", "exact", C_EXACT, 0),
    ("topology-matched routes", "ancestry", ROUTE, 1),
    ("random anatomical routes", "random", C_RANDOM, 3),
    ("site-shuffled routes", "shuffle", C_SHUFFLE, 2),
)
# Imposed-alignment dictionaries: colour and marker are the ones the sibling
# alignment figure uses, so a condition keeps one identity across figures.
# S5, as figure 7 applies it: the morphology route keeps the anatomy green
# and the three controls are one neutral family at two lightnesses, separated
# by marker AND dash so they stay apart between marker positions -- they run
# within 0.01 of one another below alignment 0.6.
ALIGN_METHODS = (
    ("morphology-selected paths", "morphology", ROUTE, "o", None),
    ("random paths", "random", C_CTRL, "s", None),
    ("depth bins", "depth", C_CTRL, "^", (3.0, 1.8)),
    ("ancestry-shuffled paths", "shuffle", C_CTRL_L, "D", None),
)


# ── frozen-estimate helpers (ports; same seeds, same draw counts) ────────
def mean_ci(values, seed, n_boot=20_000):
    """``build_journal_figures.mean_ci``: target-bootstrap mean and 95 % CI."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size),
                       replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def bootstrap(values, seed, draws=20_000):
    """``build_new_confirmatory_figures.bootstrap``: identical estimator."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)),
                       replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def load_tables():
    coarse = pd.read_csv(SOURCE / "figure5" / "task_target_method_means_ch4.csv")
    tree = pd.read_csv(FULL_TREE / "cell_method_means.csv")
    curves = pd.read_csv(ALIGNMENT / "alignment_controlled_curves.csv")
    animal = pd.read_csv(ANIMAL / "animal_signed_contrasts.csv")
    prespecified = pd.read_csv(SOURCE / "figure5" /
                               "functional_target_metrics.csv")
    all_scans = pd.read_csv(SOURCE / "functional_topology_all_scans" /
                            "cell_metrics.csv")
    original = json.loads(
        (SOURCE / "figure5" / "functional_summary.json").read_text(
            encoding="utf-8"))["tests"]["partial_shared_path_r"]
    expanded = json.loads(
        (SOURCE / "functional_topology_all_scans" / "summary.json").read_text(
            encoding="utf-8"))["metrics"]["shared_path_partial_r"]
    return coarse, tree, curves, animal, prespecified, all_scans, original, \
        expanded


def _spread(count, half=0.17):
    """Deterministic within-row dodge for per-target dots (never random)."""
    if count == 1:
        return np.zeros(1)
    return np.linspace(-half, half, count)


# ── A: every anatomy effect on one axis ──────────────────────────────────
def panel_forest(ax, prespecified, all_scans, tree, original, expanded, *,
                 header_x):
    """A: six paired contrasts, one effect axis, positive favours anatomy.

    The right-hand ``mean [95 % CI]`` column this panel used to print beside
    every row is gone: the marker and its whisker already ARE those three
    numbers, and the exact values live in Source Data and the running text.
    What the deleted column frees goes back into the plotting rectangle: the
    forest is 383 pt wide here against 301 pt when it carried the numbers.

    Only the short row names ("prespecified", "vs shuffled") are set in the
    tick column.  The three group headers are wider than any of them, and
    setting them in the same column is what used to force a 78 pt label rail
    and push this panel's axes box 78 pt right of the column-0 edge that B and
    E start from.  They are drawn instead as left-aligned mute text INSIDE the
    plotting rectangle, on the empty header row above the first row of their
    group, so they still read as group titles over the rows they name while
    the tick column stays inside the figure's shared left margin.
    """
    partial_a = prespecified.partial_shared_path_r.to_numpy(float)
    partial_b = all_scans.shared_path_partial_r.to_numpy(float)
    wide = tree.pivot(index="target_root_id", columns="method",
                      values=["heldout_normalized_mse",
                              "common_checkpoint_update_capture"])
    mse = "heldout_normalized_mse"
    capture = "common_checkpoint_update_capture"
    # Sign convention throughout: positive favours the anatomy route.  Lower
    # MSE is better, so the MSE contrast is control minus ancestry; higher
    # capture is better, so the capture contrast is ancestry minus control.
    contrasts = [
        (wide[(mse, "site-shuffled routes")]
         - wide[(mse, "topology-matched routes")]).to_numpy(float),
        (wide[(mse, "random anatomical routes")]
         - wide[(mse, "topology-matched routes")]).to_numpy(float),
        (wide[(capture, "topology-matched routes")]
         - wide[(capture, "site-shuffled routes")]).to_numpy(float),
        (wide[(capture, "topology-matched routes")]
         - wide[(capture, "random anatomical routes")]).to_numpy(float),
    ]
    stats = [bootstrap(values, 91_000 + index)
             for index, values in enumerate(contrasts)]

    groups = (
        ("structure–function",
         (("prespecified", partial_a, float(original["mean"]),
           *map(float, original["bootstrap_95_ci_mean"]), ROUTE),
          ("all scans", partial_b, float(expanded["mean"]),
           *map(float, expanded["target_bootstrap_ci95"]), ROUTE))),
        ("full tree, MSE",
         (("vs shuffled", contrasts[0], *stats[0], C_SHUFFLE),
          ("vs random", contrasts[1], *stats[1], C_RANDOM))),
        ("full tree, capture",
         (("vs shuffled", contrasts[2], *stats[2], C_SHUFFLE),
          ("vs random", contrasts[3], *stats[3], C_RANDOM))),
    )

    ticks, labels = [], []
    unit = 0
    for header, rows in groups:
        # ``header_x`` is an axes fraction on the y-axis transform, so a small
        # positive value sits just inside the panel's own left spine edge.
        ax.text(header_x, unit, header, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=PT_SMALL, color=MUTE,
                zorder=6)
        unit += 1
        for name, values, mean, low, high, color in rows:
            ax.scatter(values, unit + _spread(len(values)), s=SEED_MS ** 2,
                       color=color, alpha=SEED_ALPHA, edgecolors="none",
                       zorder=2)
            ax.errorbar([mean], [unit],
                        xerr=np.array([[mean - low], [high - mean]]),
                        marker="D", ms=MEAN_MS, color=color,
                        markerfacecolor="white", markeredgecolor=color,
                        markeredgewidth=LW_ERR, ecolor=color,
                        elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=4)
            ticks.append(unit)
            labels.append(name)
            unit += 1
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=PT_TICK)
    ax.set_ylim(unit - 0.4, -1.35)
    # Drawn as an explicit segment rather than ``axvline`` so the reference
    # line owns no vertex on a label row.
    ax.plot([0.0, 0.0], [unit - 0.4, -1.35], color=MUTE, lw=LW_REF,
            ls=(0, (3.0, 2.2)), zorder=1, solid_capstyle="butt")
    ax.set_xlim(-0.55, 0.36)
    ax.set_xticks([-0.4, -0.2, 0.0, 0.2])
    ax.set_xlabel("effect (positive favors anatomy)")
    ax.tick_params(axis="y", length=0.0, pad=2.0)
    ax.spines["left"].set_visible(False)


# ── B/C/D: horizontal point-range strips ─────────────────────────────────
def _strip(ax, rows, *, xlim, xticks, xlabel, tick_labels=True,
           inside_labels=False):
    """One dictionary per row: per-target dots, open-diamond mean, 95 % CI.

    ``inside_labels`` writes the row names in the panel's own empty left
    field instead of in a tick column outside it, so the panel can keep the
    full module width its row-mates have.
    """
    for index, (label, values, mean, low, high, color) in enumerate(rows):
        ax.scatter(values, index + _spread(len(values)), s=SEED_MS ** 2,
                   color=color, alpha=SEED_ALPHA, edgecolors="none", zorder=2)
        ax.errorbar([mean], [index],
                    xerr=np.array([[mean - low], [high - mean]]),
                    marker="D", ms=MEAN_MS, color=color,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=LW_ERR, ecolor=color, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE, zorder=4)
        if inside_labels:
            ax.text(0.022, index, label, transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=PT_SMALL, color=INK,
                    zorder=6)
    ax.set_yticks(range(len(rows)))
    if tick_labels:
        ax.set_yticklabels([row[0] for row in rows], fontsize=PT_TICK)
    else:
        ax.tick_params(axis="y", labelleft=False)   # never set_yticklabels:
        # with sharey that would strip the labels off the row's axis owner
    ax.set_ylim(len(rows) - 0.45, -0.55)
    ax.set_xlim(*xlim)
    ax.set_xticks(list(xticks))
    ax.set_xlabel(xlabel)
    ax.tick_params(axis="y", length=0.0, pad=2.0)
    ax.spines["left"].set_visible(False)


def coarse_rows(coarse, metric):
    rows = []
    for index, (method, label, color) in enumerate(COARSE_METHODS):
        values = coarse[coarse.method.eq(method)][metric].to_numpy(float)
        rows.append((label, values, *mean_ci(values, seed=1830 + index),
                     color))
    return rows


def tree_rows(tree):
    rows = []
    for method, label, color, seed_index in TREE_METHODS:
        values = tree[tree.method.eq(method)].heldout_normalized_mse \
            .to_numpy(float)
        rows.append((label, values,
                     *bootstrap(values, 90_000 + seed_index), color))
    return rows


def panel_capture(ax, coarse):
    """B: held-out task-field capture; oracle ceilings sit at the limit."""
    _strip(ax, coarse_rows(coarse, "heldout_credit_capture"),
           xlim=(-0.02, 1.08), xticks=(0.0, 0.5, 1.0),
           xlabel="held-out field capture")
    # The exact and dense rows sit on the ceiling for every target, and the
    # per-target dots are fanned within their own row: both are coincidence
    # and dodging disclosures, so the caption carries them, not the panel.


def panel_coarse_learning(ax, coarse):
    """C: held-out normalized MSE on the same cohort, same rows as B."""
    _strip(ax, coarse_rows(coarse, "heldout_normalized_mse"),
           xlim=MSE_XLIM, xticks=MSE_XTICKS, xlabel=MSE_XLABEL,
           tick_labels=False)


def panel_tree_learning(ax, tree):
    """D: the same quantity, same axis, on the complete reconstructed tree."""
    _strip(ax, tree_rows(tree), xlim=MSE_XLIM, xticks=MSE_XTICKS,
           xlabel="", tick_labels=False, inside_labels=True)


# ── E: the imposed-alignment manipulation, drawn natively ────────────────
def panel_alignment_design(ax):
    """E: one fixed dictionary, the task field rotated into its span."""
    f = Frame(ax)
    px, py = f.fx, f.fy                      # points -> frame fractions

    # Every landmark below is a fraction of the panel's own cell, not a fixed
    # number of points, so the drawing fills whatever module width the row
    # gives it instead of shrinking into a corner of a wider slot.
    ox, oy = 0.035, 0.300
    rx, ry = 0.620, 0.545                    # the rotated field's reach

    f.text((0.185, 0.955), "task credit field", size=PT_SMALL,
           color=MUTE, ha="left")

    # The dictionary span: the anatomy the rules are read off, drawn as the
    # green route it is, with a stage tree standing at its far end.
    ax.plot([ox, 0.955], [oy, oy], color=ROUTE, lw=LW_DATA,
            solid_capstyle="round", zorder=3)
    f.stage_tree((0.865, oy + 0.020), 0.300, 2, color=ROUTE, root_r_pt=2.6)
    f.text((0.300, 0.185), "ancestry span", size=PT_SMALL, color=ROUTE)

    # One field at three rotations: the family is neutral, so no condition
    # colour of the data panels is spent on the design.
    for angle, label, dx, dy in ((90.0, "α = 0", 0.042, -0.020),
                                 (50.8, "α = 0.4", 0.025, 0.060),
                                 (0.0, "α = 1", -0.090, 0.080)):
        theta = np.deg2rad(angle)
        tip = (ox + rx * np.cos(theta), oy + ry * np.sin(theta))
        f.arrow((ox, oy), tip, color=INK, lw=LW_EDGE, head=3.4, zorder=4)
        f.text((tip[0] + dx, tip[1] + dy), label, size=PT_SMALL,
               color=INK, ha="left", va="center")
    arc = Arc((ox, oy), 2 * 0.478 * rx, 2 * 0.478 * ry, theta1=0.0,
              theta2=90.0, color=MUTE, lw=LW_REF, zorder=2)
    arc.set_linestyle((0, (2.2, 1.8)))
    ax.add_patch(arc)
    # Short mute leader from the field label into the arrow it names, so the
    # line reads as an axis label and not as a second panel title.
    f.leader((0.165, 0.945), (ox + 0.012, oy + ry - 0.060))

    f.text((0.5, 0.060), "one dictionary, fixed gradient energy",
           size=PT_SMALL, color=MUTE)
    _ = px, py
    return ax


# ── F: capture against imposed alignment ─────────────────────────────────
def panel_controlled_alignment(ax, curves):
    """F: the same fixed dictionaries as alignment is imposed on the field."""
    for method, label, color, marker, dashes in ALIGN_METHODS:
        part = curves[curves.method.eq(method)].sort_values("alignment")
        line, = ax.plot(part.alignment, part.credit_capture, color=color,
                        marker=marker, ms=MARKER_MS, lw=LW_DATA,
                        markeredgecolor="white", markeredgewidth=LW_HAIR,
                        label=label, zorder=3)
        if dashes is not None:
            line.set_dashes(dashes)
        ax.fill_between(part.alignment, part.credit_capture_ci_low,
                        part.credit_capture_ci_high, color=color, alpha=0.12,
                        linewidth=0, zorder=2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.04, 1.06)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_xlabel("imposed morphology alignment")
    ax.set_ylabel("held-out field capture")
    handles = []
    for _, label, color, marker, dashes in ALIGN_METHODS:
        handle = Line2D([0], [0], color=color, marker=marker, lw=LW_DATA,
                        ms=MARKER_MS, markeredgecolor="white",
                        markeredgewidth=LW_HAIR, label=label)
        if dashes is not None:
            handle.set_dashes(dashes)
        handles.append(handle)
    legend = ax.legend(handles=handles, loc="upper left", frameon=False,
                       fontsize=PT_LEGEND, handlelength=1.3,
                       handletextpad=0.45, borderpad=0.0,
                       labelspacing=0.32, borderaxespad=0.25)
    for text in legend.get_texts():          # never pure #000000
        text.set_color(INK)
    legend.set_zorder(6)
    # The depth and shuffle curves coincide at alignment 0; that is a
    # coincidence disclosure and is reported in the caption.


# ── G: the signed animal coordinate ──────────────────────────────────────
def panel_animal_pairs(ax, animal):
    """G: one line per animal between its P+ and P- dendritic contrast."""
    for row in animal.itertuples(index=False):
        ax.plot([0, 1], [row.pplus_contrast, row.pminus_contrast],
                color=MUTE, lw=LW_HAIR, alpha=0.55, zorder=1)
        ax.scatter([0], [row.pplus_contrast], s=SEED_MS ** 2 * 2.1,
                   color=PPLUS, edgecolor="white", linewidth=LW_HAIR,
                   zorder=3)
        ax.scatter([1], [row.pminus_contrast], s=SEED_MS ** 2 * 2.1,
                   color=PMINUS, edgecolor="white", linewidth=LW_HAIR,
                   zorder=3)
    ax.axhline(0.0, color=MUTE, lw=LW_REF, ls=(0, (3.0, 2.2)), zorder=0)
    ax.set_xlim(-0.16, 1.16)
    ax.set_xticks([0, 1], ["P+", f"P{MINUS}"])
    for label, color in zip(ax.get_xticklabels(), (PPLUS, PMINUS)):
        label.set_color(color)
    ax.set_ylim(-0.28, 0.21)
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    ax.set_ylabel("dendritic contrast")
    # The sign convention, the six animals and the 6/6 sign count are all
    # reported in the caption; the panel carries only the paired geometry.


# ── the canvas ───────────────────────────────────────────────────────────
def build():
    coarse, tree, curves, animal, prespecified, all_scans, original, \
        expanded = load_tables()

    canvas = NativeCanvas(
        CANVAS_H_PT / 72.0, 3, row_weights=list(ROW_H_PT),
        hgutter_pt=HGUTTER, vgutter_pt=VGUTTER, margins=MARGINS,
        letters=False,
    )

    # Row 0: the full-width synthesis band, locked to the module grid like
    # every other panel.  Its left side is flush with column 0, so its axes
    # box starts at the same x0 as B and E; the aspect/emphasis reserve it
    # cannot avoid is taken off its RIGHT side instead.
    ax_a = canvas.panel("A", 0, 0, 12, grid="x",
                        inset_pt=(0, FOREST_SLACK_PT, 0, 0))
    ax_a.set_title("Measured-response effects on one axis", fontsize=PT_TITLE,
                   color=INK, pad=3.0, fontweight="normal")

    # Rows 1 and 2: one 4+4+4 module split shared by both rows, so every
    # panel below the band has the same width and the same height and the
    # three column starts are the same on both rows.
    ax_b = canvas.panel("B", 1, 0, 4, grid="x", title="Task-field capture")
    ax_c = canvas.panel("C", 1, 4, 4, grid="x", sharey=ax_b,
                        title="Coarse-model learning")
    ax_d = canvas.panel("D", 1, 8, 4, grid="x",
                        title="Complete-tree learning")
    ax_c.tick_params(axis="y", labelleft=False)

    ax_e = canvas.panel("E", 2, 0, 4, schematic=True,
                        title="Imposed alignment")
    ax_f = canvas.panel("F", 2, 4, 4, grid="y", title="Controlled alignment")
    ax_g = canvas.panel("G", 2, 8, 4, grid="y",
                        title="Signed animal contrast")

    panel_forest(ax_a, prespecified, all_scans, tree, original, expanded,
                 header_x=0.010)
    panel_capture(ax_b, coarse)
    panel_coarse_learning(ax_c, coarse)
    panel_tree_learning(ax_d, tree)
    # C and D plot the same quantity over the same range: the axis is named
    # once, centred under the pair, and D drops the duplicate label.
    ax_c.xaxis.set_label_coords(1.139, -0.190)
    panel_alignment_design(ax_e)
    panel_controlled_alignment(ax_f, curves)
    panel_animal_pairs(ax_g, animal)

    # One letter offset per grid column, so every letter sits the same
    # distance left of the column its panel starts in.
    for name, dx in (("A", 24.0), ("B", 24.0), ("C", 24.0),
                     ("D", 24.0), ("E", 24.0), ("F", 24.0), ("G", 24.0)):
        canvas.add_letter(name, canvas.axes[name], dx_pt=dx)

    problems = canvas.save(OUT, name="main_figure_09_native")
    return problems


if __name__ == "__main__":
    issues = build()
    if issues:
        raise SystemExit(f"main_figure_09_native: {len(issues)} layout issues")

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

    A  every anatomy effect on one axis (forest, 7)   B  task-field capture (5)
    C  coarse-model learning (6)              D  complete-tree learning (6)
    E  design: imposed alignment (5)   F  controlled alignment (4)
                                       G  signed animal contrast (3)

Structural changes against the composed version, all layout-only:

* the two structure--function point ranges that used to float alone in their
  own cell are now the first group of the forest in A, sharing its effect
  axis with the four complete-tree contrasts that used to be panel G;
* C and D are an explicit shared-axis pair: one ``sharex`` axis, the identical
  held-out normalized MSE range and ticks because they plot the same quantity
  for two tasks, and one x label centred under the pair rather than one under
  each panel;
* the imposed-alignment manipulation is drawn natively as a three-stage tree
  strip immediately left of the result it explains.  It takes five modules and
  not four because the three stage trees have to survive being cut into
  thirds: at four modules a stage is 40 pt wide, which is below the width the
  arbor needs before its branches stop resolving;
* the six-animal slope plot is short and narrow: it plots two x positions, so
  the three modules it keeps are all the paired geometry needs.

Why A is 7 modules and not 12.  The forest is six short rows; run across all
twelve modules it had ~460 pt of axes width for a 0.9-wide effect axis and
read as an empty band, and the last of that width was bought only to hold two
outlying seed dots.  A now spans 7 modules beside B, and its effect axis is
cut to the interval structure: four of the 42 per-target points fall outside
it and are clipped, which the caption states.  Its left edge is still on the
module grid -- A, C and E share the one column-0 ``x0`` -- and the three group
headers ("structure-function", "full tree, MSE", "full tree, capture") still
sit inside the plotting rectangle on their own empty header rows, so the tick
column holds only the short row names and fits the figure's shared left
margin.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

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
from credit_tree_schematics import GHOST, RIM, mix  # noqa: E402
from figure_canvas import Margins, NativeCanvas  # noqa: E402
from native_schematics import Frame  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
FULL_TREE = SOURCE / "fulltree_boundary" / "output"
ALIGNMENT = SOURCE / "alignment_controlled"
ANIMAL = SOURCE / "animal_learning_francioni"
OUT = ROOT / "figures" / "components" / "main_figure_09_native.pdf"

# ── canvas geometry, in points ───────────────────────────────────────────
CANVAS_H_PT = 426.0                      # 518.4 / 426 = 1.22 aspect
# Row 0 is the tallest row because A, at 7 modules, needs 116 pt of axes
# height to stay inside the audit's 2.20 aspect cap for a panel that shares
# its row; row 2 is next because the three-stage schematic in E has to carry
# a field strip, a tree and a stage label in one column of its own cell.  The
# spread is what the emphasis rule measures (slot-fill varies with row
# height alone here), so the three numbers are kept inside 1.35x of one
# another rather than sized panel by panel.
ROW_H_PT = (120.0, 94.0, 97.0)
# The OUTER LEFT MARGIN is the figure's one shared left reserve: it holds the
# widest tick column on the page, which is A's row-label column
# ("prespecified" is 46.9 pt, 48.9 pt with its tick pad).  The right margin is
# 5 pt because no panel's last x tick label reaches its own axes edge, so 5 pt
# is clearance rather than crowding; the one thing that did reach the canvas
# edge was G's title, which is why the narrowest panel on the page carries the
# shortest of the seven titles.  One horizontal gutter (36 pt) and one module
# width (5.45 pt) serve every row.
HGUTTER = 36.0
VGUTTER = 36.0
MARGINS = Margins(left=52.0, right=5.0, top=16.0, bottom=30.0)
# Grid column 0 carries A, C and E, and the lock it takes is shared by all
# three.  C's row-mate D starts in column 6 and needs no left reserve, so any
# reserve column 0 took would make the C/D pair two different widths -- which
# is why the left margin has to stay wider than A's row-label column
# ("prespecified", 46.9 pt, 48.9 pt with its tick pad) plus the 1.5 pt
# reserve pad.  52 pt clears it and column 0 locks nothing.

MINUS = "−"

# Statements the drawing must not carry inside the panel but the reader needs:
# printed at build time so they can be pasted into the figure caption.
CAPTION_NOTES = (
    "CAPTION: FIG09 A - Four of the 42 per-target points fall outside the "
    "effect axis and are clipped (two at +0.30 in the structure–function "
    "rows, two near −0.49 in full tree, capture vs shuffled); every mean and "
    "95 % interval is drawn in full.",
    "CAPTION: FIG09 E - One dictionary, fixed gradient energy: the same two "
    "ancestry routes are held fixed across the three stages and only the "
    "direction of the imposed task credit field changes.",
    "CAPTION: FIG09 E - The field is f(α) ∝ (1 − α) f⊥ + α f∥, where f∥ is "
    "constant within each ancestry route (so the routes span it) and f⊥ "
    "alternates within each route (so they cannot), renormalized at every α "
    "so that ‖f‖ is the same at every stage.",
    "CAPTION: FIG09 E - Stems above each tree are the imposed field on the "
    "four terminal branches, up positive and down negative; the capsules "
    "beneath are the two ancestry routes, unemphasised where they cannot "
    "carry the field and in shunting green where they carry it in full.",
    "CAPTION: FIG09 E - The three stages are α = 0, 0.4 and 1, the same "
    "imposed alignment values swept along the x axis of F.",
)

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
    # The effect axis is cut to the interval structure, not to the seed
    # cloud: every mean and every 95 % interval is inside (-0.376, +0.118),
    # and holding the last four per-target dots (two at +0.30 in the
    # structure-function rows, two near -0.49 in full tree, capture vs
    # shuffled) cost 0.25 of range -- a quarter of the panel -- and flattened
    # the four near-zero contrasts into the reference line.  The four dots
    # are clipped and the caption says so.
    ax.set_xlim(-0.45, 0.22)
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
    """C: held-out normalized MSE on the same cohort, same rows as B.

    C opens its own row, so it names its seven dictionaries in its own tick
    column; it carries the x label for the C/D pair, which ``build`` then
    centres on the pair rather than on C.
    """
    _strip(ax, coarse_rows(coarse, "heldout_normalized_mse"),
           xlim=MSE_XLIM, xticks=MSE_XTICKS, xlabel=MSE_XLABEL)


def panel_tree_learning(ax, tree):
    """D: the same quantity, same axis, on the complete reconstructed tree."""
    _strip(ax, tree_rows(tree), xlim=MSE_XLIM, xticks=MSE_XTICKS,
           xlabel="", tick_labels=False, inside_labels=True)


# ── E: the imposed-alignment manipulation, drawn natively ────────────────
#
# The manipulation used to be drawn as vector geometry -- two bare axes, three
# arrows at 90 / 51 / 0 degrees and a dashed quarter arc for the fixed norm --
# which states the algebra and shows nothing a reader can decode.  It is drawn
# here in the paper's own tree vocabulary instead, as the three stages panel F
# sweeps: one fixed dictionary of ancestry routes, and a task credit field
# whose direction turns into their span.
#
# The stage tree is the credit-tree library's arbor truncated one level: at a
# ~40 pt stage width the eight-terminal tree resolves into a thicket, so the
# four subtree heads are the terminals here.  Every coordinate below is the
# library's own, so the object is the same object as in Figures 2-4.
E_P = {
    "S": (0.00, 0.00), "J1": (0.06, 0.85), "JL": (-0.92, 1.52),
    "JR": (0.88, 1.44), "T_LL": (-1.72, 2.02), "T_LR": (-0.58, 2.32),
    "T_RL": (0.56, 2.28), "T_RR": (1.62, 1.92),
}
E_TRUNK = (("S", "J1"),)
E_LIMBS = (("J1", "JL"), ("J1", "JR"))
E_TWIGS = (("JL", "T_LL"), ("JL", "T_LR"), ("JR", "T_RL"), ("JR", "T_RR"))
# The dictionary: two ancestry routes, one per first-order subtree.  Identical
# in all three stages -- only which of them can carry the field changes.
E_ROUTES = (
    (("T_LL", "JL", "T_LR"), ("J1", "JL")),
    (("T_RL", "JR", "T_RR"), ("J1", "JR")),
)
E_TERMINALS = ("T_LL", "T_LR", "T_RL", "T_RR")
E_SPAN_X = (-2.00, 1.90)
E_SPAN_Y = (-0.34, 2.60)
E_STAGES = ((0.0, "α = 0"), (0.4, "α = 0.4"), (1.0, "α = 1"))
E_CAPSULE_PT = 5.2                       # an area mark, not a line weight
E_ROUTE_TINT = (mix("mute", 26), mix("shunting", 42))
E_ROUTE_INK = (GHOST, ROUTE)
# ``COLORS['dend']`` is itself a green two units from the shunting green, so
# an arbor inked in it cannot show the emphasis this panel turns on and off.
# The whole stage tree therefore takes the emphasis ink: unemphasised gray
# where the routes cannot carry the field, shunting green where they can.


def _blend(color_a, color_b, t):
    """Straight RGB interpolation between two palette colours."""
    a = np.asarray(to_rgb(color_a), dtype=float)
    b = np.asarray(to_rgb(color_b), dtype=float)
    return tuple(a + float(t) * (b - a))


def _imposed_field(alpha):
    """The four terminal loads of the imposed field at one alignment.

    ``perp`` alternates inside each route, so no combination of the two route
    indicators can express it; ``par`` is constant inside each route, so it is
    exactly a combination of them.  The mixture is renormalized, which is the
    fixed gradient energy the caption states.
    """
    perp = np.array([1.0, -1.0, 1.0, -1.0])
    par = np.array([1.0, 1.0, -1.0, -1.0])
    field = (1.0 - alpha) * perp + alpha * par
    return field / np.linalg.norm(field)


def _stage_frame(f, cell):
    """Point-per-tree-unit scale and a mapper for one stage's tree."""
    span_x = E_SPAN_X[1] - E_SPAN_X[0]
    span_y = E_SPAN_Y[1] - E_SPAN_Y[0]
    scale = min(cell[2] * f.w_pt / span_x, cell[3] * f.h_pt / span_y)
    cx = cell[0] + cell[2] / 2.0
    cy = cell[1] + cell[3] / 2.0
    mid_x = 0.5 * (E_SPAN_X[0] + E_SPAN_X[1])
    mid_y = 0.5 * (E_SPAN_Y[0] + E_SPAN_Y[1])

    def place(point):
        x, y = E_P[point] if isinstance(point, str) else point
        return (cx + f.fx((x - mid_x) * scale),
                cy + f.fy((y - mid_y) * scale))

    return place, scale


def _draw_stage(f, cell, alpha, label):
    """One stage: the imposed field over the fixed pair of ancestry routes.

    The stage reads top to bottom -- its alignment value, the field it
    imposes, then the anatomy that has to carry it -- so the stage label
    heads its own column and the tree stands on the cell floor.  That also
    puts ink at both ends of the cell, which is what the canvas audit's
    schematic cell-fill check asks of a drawing that owns a whole slot.
    """
    ax = f.ax
    label_pt = 11.5
    tree_aspect = ((E_SPAN_X[1] - E_SPAN_X[0])
                   / (E_SPAN_Y[1] - E_SPAN_Y[0]))
    tree_h_pt = min(cell[2] * f.w_pt / tree_aspect,
                    (cell[3] * f.h_pt - label_pt) * 0.52)
    tree_cell = (cell[0], cell[1], cell[2], f.fy(tree_h_pt))
    place, _ = _stage_frame(f, tree_cell)

    tint = _blend(*E_ROUTE_TINT, alpha)
    route_ink = _blend(*E_ROUTE_INK, alpha)

    # 1. the dictionary: one pale capsule per ancestry route, fixed geometry
    for chains in E_ROUTES:
        for chain in chains:
            xy = np.array([place(point) for point in chain], dtype=float)
            ax.plot(xy[:, 0], xy[:, 1], color=tint, lw=E_CAPSULE_PT,
                    solid_capstyle="round", solid_joinstyle="round",
                    zorder=1.3)
    # 2. the anatomy, tapered as the tree library tapers it, in the emphasis
    #    the field's alignment earns the routes
    for edges, width in ((E_TRUNK, LW_DATA), (E_LIMBS, LW_ERR),
                         (E_TWIGS, LW_EDGE)):
        for a, b in edges:
            ax.plot(*zip(place(a), place(b)), color=route_ink, lw=width,
                    solid_capstyle="round", zorder=2.4)
    for point in ("J1", "JL", "JR"):
        f.disc(place(point), 1.25, fill="white", edge=route_ink,
               lw=LW_EDGE, zorder=3.2)
    for point in E_TERMINALS:
        f.disc(place(point), 1.15, fill=route_ink, zorder=3.2)
    f.disc(place("S"), 2.1, fill=COLORS["soma"], edge=RIM, lw=LW_EDGE,
           zorder=3.4)

    # 3. the imposed field, one stem per terminal on a shared baseline
    strip_top = cell[1] + cell[3] - f.fy(label_pt)
    strip_bot = tree_cell[1] + tree_cell[3]
    base = 0.5 * (strip_top + strip_bot)
    strip_pt = (strip_top - strip_bot) * f.h_pt
    reach_pt = max(4.0, min(17.0, 0.42 * strip_pt))
    values = _imposed_field(alpha)
    xs = [place(point)[0] for point in E_TERMINALS]
    # The field's baseline is not one rule but one capsule per route, in the
    # route's own tint: what a route can carry is the level it holds over the
    # terminals it owns, so a reader compares the two stems standing on one
    # capsule rather than four stems on a common line.
    for lo, hi in ((0, 1), (2, 3)):
        ax.plot([xs[lo] - f.fx(2.6), xs[hi] + f.fx(2.6)], [base, base],
                color=tint, lw=E_CAPSULE_PT, solid_capstyle="round",
                zorder=1.2)
    for x, value, point in zip(xs, values, E_TERMINALS, strict=True):
        tip = base + f.fy(value * reach_pt / 0.7071)
        ax.plot([x, x], [base, tip], color=INK, lw=LW_DATA,
                solid_capstyle="round", zorder=4)
        f.disc((x, tip), 1.5, fill=INK, zorder=4.2)
        # a hairline back to the branch the load belongs to, so the strip
        # reads as a field ON the tree and not as a second little chart
        foot = base - f.fy(1.8) if value >= 0 else tip - f.fy(2.6)
        leader, = ax.plot([x, x], [foot, place(point)[1] + f.fy(1.8)],
                          color=mix("mute", 34), lw=LW_HAIR, zorder=1.5,
                          solid_capstyle="butt")
        leader.set_dashes((1.5, 1.4))       # a connector, not a branch

    f.text((cell[0] + cell[2] / 2.0, cell[1] + cell[3] - f.fy(5.2)), label,
           size=PT_SMALL, color=INK, va="center")


def panel_alignment_design(ax):
    """E: one fixed dictionary, the task field turned into its span."""
    f = Frame(ax)
    for cell, (alpha, label) in zip(f.split(3, axis="x", gap_pt=4.0),
                                    E_STAGES, strict=True):
        _draw_stage(f, cell, alpha, label)
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

    # Row 0: the forest beside the capture strip.  Seven modules is what the
    # forest's six rows and its 0.67-wide effect axis need; the five it gives
    # up go to B, which shares the row instead of leaving the band alone on
    # it.  A, C and E all start in grid column 0 and share one locked x0.
    ax_a = canvas.panel("A", 0, 0, 7, grid="x")
    ax_a.set_title("Measured-response effects on one axis", fontsize=PT_TITLE,
                   color=INK, pad=3.0, fontweight="normal")
    ax_b = canvas.panel("B", 0, 7, 5, grid="x", title="Task-field capture")

    # Row 1: the two held-out normalized MSE panels as ONE shared-axis pair --
    # a real ``sharex`` link, so the range and the ticks cannot drift apart,
    # and a single x label centred under both instead of one under each.
    ax_c = canvas.panel("C", 1, 0, 6, grid="x", title="Coarse-model learning")
    ax_d = canvas.panel("D", 1, 6, 6, grid="x", sharex=ax_c,
                        title="Complete-tree learning")

    # Row 2: the manipulation, its result and the animal contrast, in that
    # order, so the schematic sits immediately left of the panel it explains.
    ax_e = canvas.panel("E", 2, 0, 5, schematic=True,
                        title="Imposed alignment")
    ax_f = canvas.panel("F", 2, 5, 4, grid="y", title="Controlled alignment")
    ax_g = canvas.panel("G", 2, 9, 3, grid="y",
                        title="Signed contrast")

    panel_forest(ax_a, prespecified, all_scans, tree, original, expanded,
                 header_x=0.014)
    panel_capture(ax_b, coarse)
    panel_coarse_learning(ax_c, coarse)
    panel_tree_learning(ax_d, tree)
    panel_alignment_design(ax_e)
    panel_controlled_alignment(ax_f, curves)
    panel_animal_pairs(ax_g, animal)

    # One letter offset per grid column, so every letter sits the same
    # distance left of the column its panel starts in.
    for name in ("A", "B", "C", "D", "E", "F", "G"):
        canvas.add_letter(name, canvas.axes[name], dx_pt=24.0)

    # Lock first, then centre the pair's one x label on the pair's true
    # midpoint: the lock is idempotent, so the save below re-runs it without
    # moving anything, and an x label's horizontal overhang is invisible to
    # the reserve measurement, which reads the x axis vertically only.
    canvas.lock_reserves()
    box_c, box_d = ax_c.get_position(), ax_d.get_position()
    mid = 0.5 * (box_c.x0 + box_d.x1)
    ax_c.xaxis.set_label_coords((mid - box_c.x0) / box_c.width, -0.190)

    problems = canvas.save(OUT, name="main_figure_09_native")
    for note in CAPTION_NOTES:
        print(note)
    return problems


if __name__ == "__main__":
    issues = build()
    if issues:
        raise SystemExit(f"main_figure_09_native: {len(issues)} layout issues")

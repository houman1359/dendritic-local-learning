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

The figure answers one question -- what biological evidence supports the
theory, and where does that evidence stop -- so it is ordered as an argument
rather than as a tour of the experiments.

Layout (12 modules, three rows, each row internally uniform)::

    A  complete-tree learning (6)     B  anatomy effect summary (6)
    C  design: imposed alignment (6)  D  controlled alignment (6)
    E  signed contrast (4)   F  signed-mode energy (4)   G  boundary (4)

Two widths for seven panels, and both are whole module counts: the two
argument rows are halves of the module and the closing row is thirds of it.
Every panel of a row therefore has the same axes-box width and the same
axes-box height, which is the regularity the previous seven-panel arrangement
(six different widths across three rows, no row internally uniform) did not
have.

Narrative order, and what it moved:

* A is the measured-response null itself -- held-out learning on the complete
  reconstructed tree for the exact, ancestry, random and shuffled rules.  It
  opens the figure because it is the result the rest of the page qualifies;
  it was panel D;
* B is the compact effect summary: the same six paired contrasts on one
  effect axis (topology minus shuffled and minus random, on both the MSE and
  the capture side, under the two structure--function rows).  It was the
  seven-module panel A and keeps every row, mean and interval; it is compact
  because it now spans half the module beside A rather than the whole of it;
* C and D are the manipulation and its result, unchanged and still adjacent,
  so the schematic sits immediately left of the curve it explains (they were
  E and F).  C takes half the module rather than the five it took before, and
  the three stage trees are scaled up to fill the extra width instead of
  sitting in it: a stage is 62 pt wide here against 54 pt before, and the
  arbor is drawn to the full stage width, so it resolves further, not less;
* E is the six-animal P+/P- comparison (it was G): it plots two x positions,
  so a third of the module is all the paired geometry needs, and it opens the
  closing row rather than crowding the end of the middle one;
* F is the signed-mode energy decomposition, promoted from Supplementary
  Fig. S17c and ported verbatim -- same frozen ``mode_decomposition`` block,
  same two bars, same animal-bootstrap interval;
* G is the evidence boundary: three labelled tiers -- supported, conditional,
  not established -- set in the paper's own hedged words.  It carries no
  statistic of its own;
* the coarse-surrogate capture and learning strips that were B and C are
  demoted to Supplementary Fig. S22 (panels I and J), which already holds the
  coarse-surrogate sensitivity analyses; they are appended there with their
  rendering intact rather than deleted.

How B keeps its row-label column without breaking the row.  B's effect axis is
cut to the interval structure: four of the 42 per-target points fall outside
it and are clipped, which the caption states, and the three group headers
("structure-function", "full tree, MSE", "full tree, capture") sit inside the
plotting rectangle on their own empty header rows, so the tick column holds
only the short row names.  That column still needs 48.9 pt, which is more than
the 36 pt gutter it shares with A, and a panel that carves the shortfall out
of its own slot alone ends up narrower than its row-mate.  So the shortfall is
declared on BOTH sides of that one grid boundary
(``LABEL_RESERVE_PT``): A and C yield it on their right, B and D yield it on
their left, the canvas locks one reserve per column as it does for a colorbar
rail, and the two panels of the row come out the same width.  Row 2 needs no
declaration at all -- the 36 pt gutter already holds every label on it.
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
    PT_ANNOT,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_LEGEND,
    PT_SMALL,
    PT_TICK,
    SEED_ALPHA,
    SEED_MS,
)
from credit_tree_schematics import AMBER_TEXT, GHOST, RIM, mix  # noqa: E402
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
# One height per row, and the three are within 1.13x of one another, which is
# what the emphasis rule measures once every row is width-uniform (slot-fill
# then varies with row height alone).  Row 2 is the tallest because G's three
# tiers of type are a drawing with a floor -- three text bands that cannot be
# flattened -- and row 1 is the shortest because C's stage trees and D's
# rescue curve both gained width in the regrid and neither needs height to
# spend it.  The canvas total is 2 pt taller than the ragged seven-panel
# figure this replaces, which still keeps the float and its caption on one
# page.
ROW_H_PT = (104.0, 96.0, 108.0)
# The OUTER LEFT MARGIN is the figure's one shared left reserve.  It is kept
# at 52 pt, the width the forest's row-label column used to need in column 0,
# even though the three panels that now start in column 0 (A's labels are
# drawn inside its own plotting rectangle, C is a schematic, E carries a
# five-glyph tick column and one y label, 33.9 pt) need less: the margin has
# to clear the WIDEST of them with room to spare, because any reserve grid
# column 0 took would shorten A and C without shortening their row-mates and
# so break the row.  The right margin is 8 pt for the same reason at the
# other edge: it clears the last x tick label of B and D, so grid column 12
# locks nothing either.  One horizontal gutter (36 pt) and one module width
# (5.2 pt) serve every row.
HGUTTER = 36.0
VGUTTER = 36.0
MARGINS = Margins(left=52.0, right=8.0, top=16.0, bottom=30.0)
# The one reserve on the page is B's row-label column ("prespecified",
# 46.9 pt, 48.9 pt with its tick pad), which does not fit inside the 36 pt
# gutter it shares with A.  It is DECLARED on both sides of that boundary
# rather than measured on one: A and C give it up on their right, B and D on
# their left, so the four panels of the two argument rows keep one width
# instead of two.  16 pt clears the 14.4 pt the measurement asks for.
LABEL_RESERVE_PT = 16.0

MINUS = "−"

# Statements the drawing must not carry inside the panel but the reader needs:
# printed at build time so they can be pasted into the figure caption.
CAPTION_NOTES = (
    "CAPTION: FIG09 B - Four of the 42 per-target points fall outside the "
    "effect axis and are clipped (two at +0.30 in the structure–function "
    "rows, two near −0.49 in full tree, capture vs shuffled); every mean and "
    "95 % interval is drawn in full.",
    "CAPTION: FIG09 C - One dictionary, fixed gradient energy: the same two "
    "ancestry routes are held fixed across the three stages and only the "
    "direction of the imposed task credit field changes.",
    "CAPTION: FIG09 C - The field is f(α) ∝ (1 − α) f⊥ + α f∥, where f∥ is "
    "constant within each ancestry route (so the routes span it) and f⊥ "
    "alternates within each route (so they cannot), renormalized at every α "
    "so that ‖f‖ is the same at every stage.",
    "CAPTION: FIG09 C - Stems above each tree are the imposed field on the "
    "four terminal branches, up positive and down negative; the capsules "
    "beneath are the two ancestry routes, unemphasised where they cannot "
    "carry the field and in shunting green where they carry it in full.",
    "CAPTION: FIG09 C - The three stages are α = 0, 0.4 and 1, the same "
    "imposed alignment values swept along the x axis of D.",
    "CAPTION: FIG09 F - Promoted from Supplementary Fig. S17c and unchanged: "
    "83.7 % of pooled squared projection energy across the six animal "
    "contrast vectors lies in the signed P+/P− mode against 16.3 % in a "
    "common scalar mode; the whiskers are the animal-bootstrap 95 % interval "
    "for the signed fraction (60.6–95.7 %) and its complement.",
    "CAPTION: FIG09 G - The evidence tiers restate claims established "
    "elsewhere in the paper in the paper's own hedged words; the panel "
    "carries no estimate, interval or test of its own.",
    "CAPTION: FIG09 G - In full, the six tier statements are: supported -- "
    "availability of signed neuron-specific coordinates in six animals, and "
    "modeled ancestry capacity in the same direction in two MICrONS "
    "animals; conditional -- benefit only when task credit rotates into the "
    "anatomical span, and shunting regulates route gain only in permissive "
    "electrotonic regimes; not established -- measured visual responses show "
    "no morphology-specific alignment, and no evidence that these routes "
    "carry endogenous task credit in vivo.",
)

# Where each panel of the previous figure went, printed at build time so the
# caption and every in-text reference can be resequenced from the build log.
LETTER_MOVES = (
    "CAPTION: FIG09 D -> A (complete-tree held-out learning: exact, "
    "topology-matched, random and site-shuffled routes)",
    "CAPTION: FIG09 A -> B (compact effect summary: structure–function and "
    "full-tree topology-minus-random / minus-shuffled contrasts on one axis)",
    "CAPTION: FIG09 E -> C (imposed-alignment design, three stages)",
    "CAPTION: FIG09 F -> D (controlled-alignment rescue curve)",
    "CAPTION: FIG09 G -> E (signed P+/P− animal comparison, six animals)",
    "CAPTION: FIG09 new F (signed-mode energy, promoted from Supplementary "
    "Fig. S17c)",
    "CAPTION: FIG09 new G (evidence boundary: supported, conditional, not "
    "established)",
    "CAPTION: FIG09 B -> Supplementary Fig. S22 I (coarse-surrogate "
    "task-field capture, seven dictionaries)",
    "CAPTION: FIG09 C -> Supplementary Fig. S22 J (coarse-surrogate held-out "
    "learning, seven dictionaries)",
)

INK = COLORS["ink"]
MUTE = COLORS["mute"]
ROUTE = COLORS["shunting"]               # anatomy / ancestry route: green
C_EXACT = COLORS["bp"]                   # exact gradient: red-brown
C_CTRL = COLORS["point_mlp"]             # control family, first lightness
C_CTRL_L = "#989898"                     # control family, second lightness
C_SHUFFLE = C_CTRL                       # shuffled ancestry
C_RANDOM = C_CTRL                        # random control
# Two experimentally defined neuron coordinates, not palette conditions:
# they take the two neutral inks rather than the scalar and oracle slots.
PPLUS = COLORS["ink"]                    # animal P+ population
PMINUS = COLORS["mute"]                  # animal P- population

# The mean/summary glyph is MARKER_MS + 1.2 everywhere (the mark contract).
MEAN_MS = MARKER_MS + 1.2

# ── the held-out normalized MSE axis of panel A ─────────────────────────
MSE_XLIM = (0.60, 1.02)
MSE_XTICKS = (0.6, 0.7, 0.8, 0.9, 1.0)
MSE_XLABEL = "held-out normalized MSE"

# Complete-tree rules, same order and seeds as the frozen builder: exact
# first, then the topology-matched routes and their two matched controls.
# The trailing integer is the rule's own bootstrap seed offset, kept with the
# rule rather than with its row position, so reordering the display cannot
# move a published interval.
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


# ── frozen-estimate helper (a port; same seeds, same draw counts) ────────
#
# The ``mean_ci`` port that stood here served only the two coarse-surrogate
# strips, and it went with them to ``build_journal_figures`` (which is where
# it was ported from, and which still calls it at the same seeds for the
# demoted panels).  What the panels on this page need is the target
# bootstrap below.
def bootstrap(values, seed, draws=20_000):
    """``build_new_confirmatory_figures.bootstrap``: identical estimator."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)),
                       replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def load_tables():
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
    # The frozen mode decomposition behind the promoted signed-mode panel.
    mode = json.loads(
        (ANIMAL / "summary.json").read_text(
            encoding="utf-8"))["mode_decomposition"]
    return tree, curves, animal, prespecified, all_scans, original, \
        expanded, mode


def _spread(count, half=0.17):
    """Deterministic within-row dodge for per-target dots (never random)."""
    if count == 1:
        return np.zeros(1)
    return np.linspace(-half, half, count)


# ── B: the compact effect summary, every anatomy effect on one axis ──────
def panel_forest(ax, prespecified, all_scans, tree, original, expanded, *,
                 header_x):
    """B: six paired contrasts, one effect axis, positive favours anatomy.

    The right-hand ``mean [95 % CI]`` column this panel used to print beside
    every row is gone: the marker and its whisker already ARE those three
    numbers, and the exact values live in Source Data and the running text.
    What the deleted column frees goes back into the plotting rectangle: the
    forest is 383 pt wide here against 301 pt when it carried the numbers.

    Only the short row names ("prespecified", "vs shuffled") are set in the
    tick column.  The three group headers are wider than any of them, and
    setting them in the same column is what used to force a 78 pt label rail
    and push this panel's axes box 78 pt right of the column edge its
    row-mate starts from.  They are drawn instead as left-aligned mute text
    INSIDE the
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


# ── A: the horizontal point-range strip ──────────────────────────────────
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


def tree_rows(tree):
    rows = []
    for method, label, color, seed_index in TREE_METHODS:
        values = tree[tree.method.eq(method)].heldout_normalized_mse \
            .to_numpy(float)
        rows.append((label, values,
                     *bootstrap(values, 90_000 + seed_index), color))
    return rows


def panel_tree_learning(ax, tree):
    """A: held-out learning on the complete reconstructed tree.

    The measured-response null, and the result the rest of the page
    qualifies, so it opens the figure.  Nothing about the drawing changed
    when it moved from the second row: the same four rules in the same
    order, the same estimator and seeds, the same normalized-MSE range and
    ticks, and the rule names still set inside the panel's own empty left
    field rather than in a tick column.  It carries the x label itself now,
    because the coarse-surrogate strip it used to share an axis with has
    moved to Supplementary Fig. S22.
    """
    _strip(ax, tree_rows(tree), xlim=MSE_XLIM, xticks=MSE_XTICKS,
           xlabel=MSE_XLABEL, tick_labels=False, inside_labels=True)


# ── C: the imposed-alignment manipulation, drawn natively ────────────────
#
# The manipulation used to be drawn as vector geometry -- two bare axes, three
# arrows at 90 / 51 / 0 degrees and a dashed quarter arc for the fixed norm --
# which states the algebra and shows nothing a reader can decode.  It is drawn
# here in the paper's own tree vocabulary instead, as the three stages panel D
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
    # The tree takes the WIDTH of its stage cell whenever the cell is tall
    # enough to hold it -- the first term -- and falls back to a share of the
    # cell height only when it is not.  That share is 0.60 and not the 0.52 it
    # was at five modules: half the module gives each stage 62 pt instead of
    # 54, and at 0.52 the arbor would have stopped growing at 57.5 pt wide and
    # left 5 pt of white space either side of it.  What is left of the cell is
    # the field strip, which still gets its full reach.
    tree_h_pt = min(cell[2] * f.w_pt / tree_aspect,
                    (cell[3] * f.h_pt - label_pt) * 0.60)
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
    """C: one fixed dictionary, the task field turned into its span."""
    f = Frame(ax)
    for cell, (alpha, label) in zip(f.split(3, axis="x", gap_pt=4.0),
                                    E_STAGES, strict=True):
        _draw_stage(f, cell, alpha, label)
    return ax


# ── D: capture against imposed alignment ─────────────────────────────────
def panel_controlled_alignment(ax, curves):
    """D: the same fixed dictionaries as alignment is imposed on the field."""
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


# ── E: the signed animal coordinate ──────────────────────────────────────
def panel_animal_pairs(ax, animal):
    """E: one line per animal between its P+ and P- dendritic contrast."""
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


# ── F: signed-mode energy, promoted from Supplementary Fig. S17c ─────────
def panel_mode_energy(ax, mode):
    """F: where the six animal contrast vectors put their energy.

    A port of ``build_alignment_animal_figure.mode_energy`` -- the panel that
    was Supplementary Fig. S17c -- and nothing here recomputes it: the two
    fractions and the animal-bootstrap interval are read from the same frozen
    ``mode_decomposition`` block that panel read, the bars keep their colours,
    width and edge, and the two printed percentages keep their format.  Only
    the type and stroke tokens are the canvas's own, which is what the panel
    was already set in.
    """
    fractions = [mode["common_energy_fraction"],
                 mode["signed_energy_fraction"]]
    ci_lo, ci_hi = mode["animal_bootstrap_95_ci"]
    # The bootstrap is over the signed fraction; the common fraction is its
    # complement, so its interval is the reflected one.
    intervals = [(1.0 - ci_hi, 1.0 - ci_lo), (ci_lo, ci_hi)]
    # The signed bar is a descriptive energy fraction, not a backprop series:
    # keep the reserved BP red out of it (CVD colour grammar).
    ax.bar([0, 1], fractions, color=[C_CTRL, COLORS["dend"]],
           edgecolor=COLORS["edge"], linewidth=LW_EDGE, width=0.64)
    yerr = np.array(
        [[value - low
          for value, (low, _) in zip(fractions, intervals, strict=True)],
         [high - value
          for value, (_, high) in zip(fractions, intervals, strict=True)]])
    ax.errorbar([0, 1], fractions, yerr=yerr, fmt="none",
                ecolor=COLORS["edge"], elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=4)
    for x, value, (_, high) in zip((0, 1), fractions, intervals,
                                   strict=True):
        ax.text(x, high + 0.035, f"{100 * value:.1f}%", ha="center",
                va="bottom", fontsize=PT_ANNOT, color=INK)
    ax.text(0.03, 0.97, "95% CI,\nanimal\nbootstrap", transform=ax.transAxes,
            ha="left", va="top", fontsize=PT_SMALL, color=MUTE)
    ax.set_xticks([0, 1], ["common", "signed"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("contrast energy")


# ── G: what the biological evidence carries, and where it stops ──────────
#
# Three tiers, one band each, set in the paper's own hedged words: every line
# below is a claim the manuscript already makes (the abstract for the two
# conditional lines and the measured-response null, the six-animal reanalysis
# for the signed coordinate, the cross-animal structural replication for the
# route-capacity line, and the in-vivo caveat that closes both).  The panel
# states no estimate, no interval and no test -- it is the figure's summary of
# what the evidence supports, what it supports only conditionally, and what it
# does not establish.
# Each line is a phrase the manuscript itself uses, cut to the width a
# third-of-the-module panel gives it at the 6.8 pt type floor (about 117 pt of
# measured line): the statement is the short form, and the sentence it
# abbreviates is printed into the caption at build time rather than set here
# at a size below the type scale.  The two conditional lines are deliberately
# parallel ("only if"), because the condition IS the claim.
EVIDENCE_TIERS = (
    ("supported", ROUTE, ROUTE, (
        "signed coordinates in six animals",
        "modeled capacity, two animals",
    )),
    ("conditional", COLORS["local"], AMBER_TEXT, (
        "benefit only if credit rotates in",
        "shunting gain only if permissive",
    )),
    ("not established", C_CTRL, C_CTRL, (
        "no morphology-specific alignment",
        "no endogenous task credit in vivo",
    )),
)
# Type bands inside one tier, in points from the band's own top edge: the
# heading line, then the two statements.  They are points and not fractions
# because type is points: at any panel height the three lines keep the same
# 10 pt and 9 pt separations, which is what keeps them off one another.
TIER_HEAD_PT = 9.0
TIER_LINE_PT = (19.0, 28.0)
# The tier's own left rail, in points from the band edge: the bullet, then the
# statement.  Both are tighter than they were at seven modules, because at
# four modules the line itself is what the width has to be spent on.
TIER_BULLET_X_PT = 5.0
TIER_TEXT_X_PT = 9.0


def panel_evidence_boundary(ax):
    """G: three tiers -- supported, conditional, not established."""
    f = Frame(ax)
    for cell, (name, color, text_color, lines) in zip(
            f.split(3, axis="y", gap_pt=4.0), EVIDENCE_TIERS, strict=True):
        # The band is what makes a tier an object rather than three loose
        # lines, and it is drawn full-cell-width in the tier's own tint so
        # the three tiers read as one ordered scale.
        f.group(cell, tint=mix(color, 9), edge=mix(color, 45), lw=LW_HAIR,
                radius_pt=2.5)
        top = cell[1] + cell[3]
        f.text((cell[0] + f.fx(TIER_TEXT_X_PT), top - f.fy(TIER_HEAD_PT)),
               name, size=PT_ANNOT, color=text_color, ha="left")
        for offset, line in zip(TIER_LINE_PT, lines, strict=True):
            y = top - f.fy(offset)
            f.disc((cell[0] + f.fx(TIER_BULLET_X_PT), y), 0.9, fill=color,
                   zorder=5)
            f.text((cell[0] + f.fx(TIER_TEXT_X_PT), y), line, size=PT_SMALL,
                   color=INK, ha="left")
    return ax


# ── the canvas ───────────────────────────────────────────────────────────
def build():
    tree, curves, animal, prespecified, all_scans, original, expanded, mode \
        = load_tables()

    canvas = NativeCanvas(
        CANVAS_H_PT / 72.0, 3, row_weights=list(ROW_H_PT),
        hgutter_pt=HGUTTER, vgutter_pt=VGUTTER, margins=MARGINS,
        letters=False,
    )

    # Row 0: the measured-response null, then the effect summary that
    # generalises it -- half the module each.
    ax_a = canvas.panel("A", 0, 0, 6, grid="x",
                        title="Complete-tree learning")
    ax_b = canvas.panel("B", 0, 6, 6, grid="x", title="Anatomy effects")

    # Row 1: the manipulation and its result, adjacent and equal, so the
    # schematic sits immediately left of the curve it explains.
    ax_c = canvas.panel("C", 1, 0, 6, schematic=True,
                        title="Imposed alignment")
    ax_d = canvas.panel("D", 1, 6, 6, grid="y", title="Controlled alignment")

    # Row 2: the animal coordinate, the decomposition behind it, and the
    # boundary the whole page argues for -- a third of the module each.
    ax_e = canvas.panel("E", 2, 0, 4, grid="y", title="Signed contrast")
    ax_f = canvas.panel("F", 2, 4, 4, grid="y", title="Signed mode")
    ax_g = canvas.panel("G", 2, 8, 4, schematic=True,
                        title="Evidence boundary")

    # The one declared reserve, taken symmetrically on the single grid
    # boundary that cannot hold its labels in the gutter: B's row-label
    # column.  Declaring it on both sides is what keeps the two panels of
    # row 0 -- and of row 1, which shares the boundary -- one width.
    for panel_name in ("A", "C"):
        canvas.declare_reserve(panel_name, right=LABEL_RESERVE_PT)
    for panel_name in ("B", "D"):
        canvas.declare_reserve(panel_name, left=LABEL_RESERVE_PT)

    panel_tree_learning(ax_a, tree)
    panel_forest(ax_b, prespecified, all_scans, tree, original, expanded,
                 header_x=0.014)
    panel_alignment_design(ax_c)
    panel_controlled_alignment(ax_d, curves)
    panel_animal_pairs(ax_e, animal)
    panel_mode_energy(ax_f, mode)
    panel_evidence_boundary(ax_g)

    # One letter offset per grid column, so every letter sits the same
    # distance left of the column its panel starts in.
    for name in ("A", "B", "C", "D", "E", "F", "G"):
        canvas.add_letter(name, canvas.axes[name], dx_pt=24.0)

    problems = canvas.save(OUT, name="main_figure_09_native")
    for note in CAPTION_NOTES:
        print(note)
    for note in LETTER_MOVES:
        print(note)
    return problems


if __name__ == "__main__":
    issues = build()
    if issues:
        raise SystemExit(f"main_figure_09_native: {len(issues)} layout issues")

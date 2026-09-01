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

Layout (12 modules, three rows)::

    A measured-response pipeline (3)  B complete-tree learning (4)
    C anatomy effect summary (5)
    D design: imposed alignment (6)   E controlled alignment (6)
    F signed contrast (4)   G signed-mode energy (4)   H boundary (4)

The first row begins with the measured-response pipeline, followed by the
complete-tree learning result and a compact matched-control effect summary.
The second row keeps the imposed-alignment manipulation adjacent to its
controlled rescue.  The final row shows the external signed neuronal contrast,
its mode decomposition and the evidence boundary.  The coarse major-branch and
channel-sensitivity analyses remain in Supplementary Fig. S22.

The label-heavy effect forest receives five modules and a symmetric reserve on
its boundary with panel B; all other panels use the shared gutters and margins.
No panel is rasterized or rescaled after drawing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.transforms import offset_copy

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
from native_schematics import Frame, _text_w_pt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
FULL_TREE = SOURCE / "fulltree_boundary" / "output"
ALIGNMENT = SOURCE / "alignment_controlled"
ANIMAL = SOURCE / "animal_learning_francioni"
OUT = ROOT / "figures" / "components" / "main_figure_09_native.pdf"

# ── canvas geometry, in points ───────────────────────────────────────────
CANVAS_H_PT = 468.0
# One height per row.  Row 2 is the tallest because G's ladder is a drawing
# with a floor -- four rungs, each carrying a tree glyph that has to stay
# legible and up to two bands of type that cannot be flattened, plus the
# status key -- and it was raised from 118 pt to 128 pt so the rung glyph
# and the air between rungs both grow; E and F centre their content in the
# taller box rather than stretching it.
#
# Row 2 cannot be grown on its own.  The emphasis rule compares each panel's
# axes area against its own slot, and for a row-2 panel the slot width IS the
# panel width (four modules and three gutters, no reserve), so its slot fill
# is exactly h2 / mean-row-height, while the reserved half-module panels fill
# 0.924 * h / mean-row-height.  The mean cancels: the spread is
# (h2 - 1.14) / (0.924 * min(h0, h1 - 1.14)), independent of the canvas
# height, so the ONLY way to buy row 2 height inside the 1.35x cap is to
# raise the shortest of the other rows with it.  Row 1 therefore goes 96 ->
# 106 pt, which lifts the floor from C (94.86 pt of axes) to A (104 pt) and
# leaves the spread at 1.32x; row 0 is untouched.  C and D keep their
# designs and gain 10 pt of height with the row.  The canvas grows by the
# same 20 pt to 456 pt, aspect 1.14, still above the 1.10 floor.
# Row 2 carries the evidence ladder, whose four rungs each need a title
# line, a qualifier line and a badge; at 128 pt its lines overlapped in
# eleven places.  The extra 26 pt goes to that row alone.
ROW_H_PT = (98.0, 98.0, 166.0)
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
# Compact the formerly empty inter-row bands rather than shrinking panel
# contents.  This preserves nearly the same usable panel heights while
# keeping the complete figure and caption within one manuscript page.
VGUTTER = 48.0
MARGINS = Margins(left=52.0, right=8.0, top=22.0, bottom=30.0)
# The one reserve on the page is B's row-label column ("prespecified",
# 46.9 pt, 48.9 pt with its tick pad), which does not fit inside the 36 pt
# gutter it shares with A.  It is DECLARED on both sides of that boundary
# rather than measured on one: A and C give it up on their right, B and D on
# their left, so the four panels of the two argument rows keep one width
# instead of two.  16 pt clears the 14.4 pt the measurement asks for.
LABEL_RESERVE_PT = 16.0
C_LABEL_RESERVE_PT = 30.0   # C row labels -> clear of the letter column

MINUS = "−"

# Statements the drawing must not carry inside the panel but the reader needs:
# printed at build time so they can be pasted into the figure caption.
CAPTION_NOTES = (
    "CAPTION: FIG09 C - Effects with different native units are displayed as "
    "standardized target-level effects; intervals bootstrap the complete "
    "target-level estimand.",
    "CAPTION: FIG09 D - The fixed-energy imposed field is "
    "φ(a)=√a u∥+√(1−a)u⊥, where u∥ lies in the subtree-route span and u⊥ "
    "lies in its orthogonal complement.",
    "CAPTION: FIG09 F - The signed P+/P− fraction and common scalar fraction "
    "partition the same pooled projection energy; one interval therefore "
    "bounds both complementary fractions.",
    "CAPTION: FIG09 G - The evidence ladder is biological rather than "
    "statistical: green, amber and open dashed steps denote supported, "
    "conditional and not established, respectively.",
    "CAPTION: FIG09 G - The imposed-alignment rescue in D,E is a controlled "
    "sufficiency test and is not evidence that the measured cells use these "
    "routes for endogenous task credit.",
)

# Current publication panel inventory, printed with the build log so captions
# and text references can be audited against the generated PDF.
LETTER_MOVES = (
    "CAPTION: FIG09 A response-prediction pipeline",
    "CAPTION: FIG09 B complete-tree held-out learning",
    "CAPTION: FIG09 C matched topology-effect summary",
    "CAPTION: FIG09 D imposed-alignment design",
    "CAPTION: FIG09 E controlled-alignment rescue",
    "CAPTION: FIG09 F signed P+/P− animal contrast",
    "CAPTION: FIG09 G evidence boundary",
    "CAPTION: coarse-surrogate and channel-sensitivity analyses remain in S22",
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

# ── the standardized target-level-effect axis of panel C ────────────────
# The right limit holds every upper bound on the page: the widest is the
# full-tree MSE "vs shuffled" row at +1.76, which the previous +1.7 limit cut
# by six hundredths of a unit and drew as though it ended on the spine.
# The left limit does NOT hold every lower bound.  The full-tree capture
# "vs random" row runs to -10.97 -- a standardized effect that wide is a
# small-denominator excursion of the bootstrap, and an axis stretched to
# contain it would squeeze the other five rows and all their dots into a
# fifth of the panel.  That one whisker is therefore drawn to the panel edge
# and given an explicit arrowhead there (``_truncation_head``), and its true
# bound is printed into the caption at build time.  A clipped whisker is
# never left to look bounded.
FOREST_XLIM = (-2.2, 1.9)
FOREST_XTICKS = (-2.0, -1.0, 0.0, 1.0)
FOREST_XLABEL = "standardized target-level effect"

# ── the held-out normalized MSE axis of panel B ─────────────────────────
MSE_XLIM = (0.60, 1.02)
MSE_XTICKS = (0.6, 0.7, 0.8, 0.9, 1.0)
MSE_XLABEL = "held-out normalized MSE  (lower is better)"

# Complete-tree rules, same order and seeds as the frozen builder: exact
# first, then the topology-matched routes and their two matched controls.
# The trailing integer is the rule's own bootstrap seed offset, kept with the
# rule rather than with its row position, so reordering the display cannot
# move a published interval.
TREE_METHODS = (
    ("exact compartment error", "exact", C_EXACT, 0),
    ("topology-matched routes", "subtree", ROUTE, 1),
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
    ("morphology-selected paths", "subtree", ROUTE, "o", None),
    ("random anatomical routes", "random", C_CTRL, "s", None),
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


def standardized_bootstrap(values, seed, draws=20_000):
    """Paired standardized mean and target-bootstrap interval.

    Panel B combines partial rank correlations, normalized-MSE differences
    and common-checkpoint update-match differences. Dividing each paired target contrast by its
    across-target sample standard deviation puts all rows on one effect-size
    scale. The bootstrap recomputes the complete standardized estimand.
    """
    values = np.asarray(values, dtype=float)
    scale = values.std(ddof=1)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("standardized effect requires nonzero target variance")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True)
    sample_scale = samples.std(axis=1, ddof=1)
    keep = np.isfinite(sample_scale) & (sample_scale > 0)
    effects = samples[keep].mean(axis=1) / sample_scale[keep]
    low, high = np.quantile(effects, [0.025, 0.975])
    return values / scale, float(values.mean() / scale), float(low), float(high)


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


# ── B: a whisker that leaves the panel says so ──────────────────────────
def _truncation_head(ax, y, value, color, *, xlim=None):
    """Draw an arrowhead where an interval bound falls outside the axes.

    Matplotlib clips a whisker at the axes edge, so an interval that runs
    off the panel and an interval that happens to end on the spine ink the
    same pixels: the reader sees a bounded interval that is not bounded.
    The head is set in the row's own colour with its tip on the limit and
    its body inside the panel, so the truncation is visible at the whisker
    itself, and the true bound is stated in the caption.

    Returns ``True`` when the bound is out of range (nothing about the
    estimate changes; this only reports that the interval continues).
    """
    low, high = FOREST_XLIM if xlim is None else xlim
    if low <= value <= high:
        return False
    edge, marker, inward = ((low, "<", 1.0) if value < low
                            else (high, ">", -1.0))
    # Offset in POINTS, not in data: the head is a fixed-size glyph, so its
    # tip lands on the limit whatever the row's data scale is.
    # MARKER_MS, not the larger MEAN_MS: the head is a note on the whisker,
    # not a second summary glyph competing with the mean diamond, and the
    # smaller head also keeps its tip clear of the row-label column.
    inside = offset_copy(ax.transData, fig=ax.figure,
                         x=inward * MARKER_MS / 2.0, y=0.0, units="points")
    ax.plot([edge], [y], marker=marker, ms=MARKER_MS, linestyle="none",
            color=color, markerfacecolor=color, markeredgecolor=color,
            markeredgewidth=LW_ERR, transform=inside, clip_on=False,
            zorder=5)
    return True


def _truncation_note(truncated):
    """The build-time caption sentence for every whisker clipped in B."""
    spans = "; ".join(
        f"{header} {name} = [{MINUS if low < 0 else ''}{abs(low):.2f}, "
        f"{MINUS if high < 0 else ''}{abs(high):.2f}]"
        for header, name, low, high in truncated)
    low, high = FOREST_XLIM
    return ("CAPTION: FIG09 C - The effect axis is clipped at "
            f"{MINUS if low < 0 else ''}{abs(low):.1f} to "
            f"{high:.1f}; an arrowhead on a whisker marks an interval that "
            "continues past the panel edge rather than ending there. "
            f"Clipped row{'s' if len(truncated) > 1 else ''} and "
            f"{'their' if len(truncated) > 1 else 'its'} full 95 % "
            f"interval{'s' if len(truncated) > 1 else ''}: {spans}. "
            "Every other interval is shown whole.")


# ── B: the compact effect summary, every anatomy effect on one axis ──────
def panel_forest(ax, prespecified, all_scans, tree, original, expanded, *,
                 header_x):
    """B: six paired standardized contrasts; positive favours anatomy.

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

    Returns the rows whose interval leaves ``FOREST_XLIM``, each with its
    true bounds, so the caption can name them: such a row is drawn with an
    arrowhead at the panel edge instead of a silently clipped whisker.
    """
    partial_a = prespecified.partial_shared_path_r.to_numpy(float)
    partial_b = all_scans.shared_path_partial_r.to_numpy(float)
    wide = tree.pivot(index="target_root_id", columns="method",
                      values=["heldout_normalized_mse",
                              "common_checkpoint_update_capture"])
    mse = "heldout_normalized_mse"
    update_match = "common_checkpoint_update_capture"
    # Sign convention throughout: positive favours the anatomy route.  Lower
    # MSE is better, so the MSE contrast is control minus ancestry; higher
    # Higher update match is better, so its contrast is ancestry minus control.
    contrasts = [
        (wide[(mse, "site-shuffled routes")]
         - wide[(mse, "topology-matched routes")]).to_numpy(float),
        (wide[(mse, "random anatomical routes")]
         - wide[(mse, "topology-matched routes")]).to_numpy(float),
        (wide[(update_match, "topology-matched routes")]
         - wide[(update_match, "site-shuffled routes")]).to_numpy(float),
        (wide[(update_match, "topology-matched routes")]
         - wide[(update_match, "random anatomical routes")]).to_numpy(float),
    ]
    raw_groups = (
        ("structure–function",
        (("single scan", partial_a, ROUTE),
          ("all eligible scans", partial_b, ROUTE))),
        ("full tree, MSE",
         (("vs shuffled", contrasts[0], C_SHUFFLE),
          ("vs random", contrasts[1], C_RANDOM))),
        ("full tree, update match",
         (("vs shuffled", contrasts[2], C_SHUFFLE),
          ("vs random", contrasts[3], C_RANDOM))),
    )
    groups = []
    seed_offset = 0
    for header, rows in raw_groups:
        standardized_rows = []
        for name, values, color in rows:
            shown, mean, low, high = standardized_bootstrap(
                values, 91_000 + seed_offset)
            standardized_rows.append((name, shown, mean, low, high, color))
            seed_offset += 1
        groups.append((header, tuple(standardized_rows)))

    ticks, labels = [], []
    truncated = []
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
            # Any bound outside the axis gets a head in the row's colour,
            # and the row is handed to the caption with its true bounds.
            heads = [_truncation_head(ax, unit, bound, color)
                     for bound in (low, high)]
            if any(heads):
                truncated.append((header, name, low, high))
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
    ax.set_xlim(*FOREST_XLIM)
    ax.set_xticks(list(FOREST_XTICKS))
    ax.set_xlabel(FOREST_XLABEL)
    ax.tick_params(axis="y", length=0.0, pad=2.0)
    ax.spines["left"].set_visible(False)
    return truncated


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


# ── A: what B and C actually compare ─────────────────────────────────────
#
# This panel used to be a four-stage flowchart -- measured inputs, mapped
# conductances, predicted response, held-out error.  That is the generic
# setup of any regression, and it is not what the text sends a reader here
# to see: the sentence citing panel A describes the FEEDBACK ROUTE
# DICTIONARY, and B's four categories ("exact", "subtree", "random",
# "shuffle") were otherwise bare words on an axis.  The panel now draws that
# dictionary, in B's order and B's colours, so A reads as a pictorial legend
# for the strip beside it.
#
# The arbor is a row of eight terminals under four tinted bands, one band per
# nested subtree.  A route is a chip laid over the terminals it feeds, so the
# comparison is one question of shape: does the chip lie inside a band?
# Matched routes do, random anatomical routes are contiguous but cut across
# them, and shuffled routes are not contiguous at all.
N_TERM = 8
TERM_X = np.linspace(0.375, 0.965, N_TERM)
BAND_OF = (0, 0, 1, 1, 2, 2, 3, 3)          # which nested subtree each feeds
ROW_Y = (0.815, 0.635, 0.455, 0.275)        # one row per rule, B's order
SITE_Y = 0.145                              # the eight sites, drawn once
CHIP_LW = 4.0                               # route chip thickness
TERM_R_PT = 1.35
LABEL_X = 0.315

# Exact learning is the only rule with a trial-specific signal at every
# compartment, so its chips are eight singletons. Each restricted dictionary
# contains four routes and supplies its own once-calibrated, frozen spatial
# profile.
ROUTE_SETS = (
    tuple((i,) for i in range(N_TERM)),                 # exact: per compartment
    ((0, 1), (2, 3), (4, 5), (6, 7)),                   # matched to the bands
    ((0,), (2, 3), (4, 5), (6, 7)),                     # four valid toy subtrees
    ((0, 5), (1, 3), (2, 7), (4, 6)),                   # not contiguous at all
)


def panel_route_dictionary(ax):
    """A: the feedback route dictionary the rest of the row evaluates.

    One row per rule, in the order and colour B uses, so the reader can carry
    a shape from here straight to its estimate.  Nothing is recomputed: the
    panel is a definition, and the four rules are the four in TREE_METHODS.
    """
    f = Frame(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    # The anatomy, drawn once behind every rule: four nested subtrees as
    # tinted columns, so "inside a band" is legible without four little trees
    # (at this width an eight-terminal tree resolves into a thicket).
    half = 0.5 * (TERM_X[1] - TERM_X[0])
    for band in range(4):
        members = [i for i in range(N_TERM) if BAND_OF[i] == band]
        x0 = TERM_X[members[0]] - half * 0.86
        x1 = TERM_X[members[-1]] + half * 0.86
        f.group((x0, 0.105, x1 - x0, 0.82), tint=mix("point_mlp", 9),
                edge="none", radius_pt=2.0, zorder=0.3)
    f.text((0.5 * (TERM_X[0] + TERM_X[-1]), 0.955),
           "illustrative sites; complete-arbor routes may overlap",
           size=PT_SMALL, color=MUTE)

    # The eight sites, drawn ONCE beneath the rules rather than repeated under
    # every one of them: repeated, they read as a second kind of block.
    for i in range(N_TERM):
        f.disc((TERM_X[i], SITE_Y), TERM_R_PT, fill=MUTE, zorder=3)
    f.text((TERM_X[0] - half * 1.5, SITE_Y), "sites", size=PT_SMALL,
           color=MUTE, ha="right")

    # The three restricted rules are marked as one group: they do not get
    # four independently varying teaching signals, and the figure should say
    # so.  A rule beside them, not a box around them -- a box wide enough to
    # hold the labels ran off both edges of the panel.
    ax.plot([0.055, 0.055], [0.205, 0.700], color=mix("point_mlp", 40),
            lw=LW_HAIR, solid_capstyle="butt", zorder=2)
    for y_end in (0.205, 0.700):
        ax.plot([0.055, 0.085], [y_end, y_end], color=mix("point_mlp", 40),
                lw=LW_HAIR, solid_capstyle="butt", zorder=2)

    for row, ((_, label, color, _), routes) in enumerate(
            zip(TREE_METHODS, ROUTE_SETS, strict=True)):
        y = ROW_Y[row]
        f.text((LABEL_X, y), label, size=PT_ANNOT, color=color, ha="right")
        for route in routes:
            xs = [TERM_X[i] for i in route]
            if len(route) == 1:
                ax.plot([xs[0] - half * 0.30, xs[0] + half * 0.30],
                        [y + 0.028, y + 0.028], color=color, lw=CHIP_LW,
                        solid_capstyle="round", zorder=4)
            elif route[-1] - route[0] == len(route) - 1:
                ax.plot([xs[0], xs[-1]], [y + 0.028, y + 0.028], color=color,
                        lw=CHIP_LW, solid_capstyle="round", zorder=4)
            else:
                # A route that is not a subtree cannot be one bar: its sites
                # are drawn where they are and tied by an arc, which is the
                # whole point of the shuffled control.
                for x in xs:
                    ax.plot([x - half * 0.30, x + half * 0.30],
                            [y + 0.028, y + 0.028], color=color, lw=CHIP_LW,
                            solid_capstyle="round", zorder=4)
                ax.add_patch(FancyArrowPatch(
                    (xs[0], y + 0.028), (xs[-1], y + 0.028),
                    arrowstyle="-", connectionstyle="arc3,rad=-0.16",
                    lw=LW_HAIR, color=color, zorder=3.5, clip_on=False))

    # Short enough to sit inside the panel: the full statement -- that the
    # error is a scalar and only the routes differ -- is in the caption.
    f.text((0.5, 0.042), "one fixed field in a four-route span per rule",
           size=PT_SMALL, color=MUTE)
    return ax


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
E_STAGES = ((0.0, "a = 0"), (0.4, "a = 0.4"), (1.0, "a = 1"))
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


def _imposed_field(a):
    """The four terminal loads of the imposed field at one alignment.

    ``perp`` alternates inside each route, so no combination of the two route
    indicators can express it; ``par`` is constant inside each route, so it is
    exactly a combination of them.  The square-root parameterization preserves
    field energy and makes ``a`` the exact fraction of energy in the
    ancestry-route span, matching the analyzed experiment and the Methods.
    """
    perp = np.array([1.0, -1.0, 1.0, -1.0])
    par = np.array([1.0, 1.0, -1.0, -1.0])
    perp = perp / np.linalg.norm(perp)
    par = par / np.linalg.norm(par)
    return np.sqrt(1.0 - a) * perp + np.sqrt(a) * par


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


def _draw_stage(f, cell, alignment_a, label):
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

    tint = _blend(*E_ROUTE_TINT, alignment_a)
    route_ink = _blend(*E_ROUTE_INK, alignment_a)

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
    values = _imposed_field(alignment_a)
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


E_KEY_PT = 12.0        # the key band under the three stages


def _alignment_key(f):
    """Name D's three marks: the route, the field on it, and the alignment.

    The panel draws a capsule, a stem and a grey-to-green ramp, and none of
    them is a convention a reader arrives with.  A key costs about an eighth
    of the panel height and removes the need to carry three definitions from
    the caption into the drawing.
    """
    ax = f.ax
    y = f.fy(0.46 * E_KEY_PT)
    glyph, gap, pad = f.fx(9.0), f.fx(3.0), f.fx(7.5)
    tint = _blend(*E_ROUTE_TINT, 0.55)

    def capsule(x0):
        ax.plot([x0, x0 + glyph], [y, y], color=tint, lw=E_CAPSULE_PT,
                solid_capstyle="round", zorder=3)

    def stem(x0):
        capsule(x0)
        for at, reach in ((0.30, 4.2), (0.70, -4.2)):
            sx = x0 + at * glyph
            ax.plot([sx, sx], [y, y + f.fy(reach)], color=INK, lw=LW_DATA,
                    solid_capstyle="round", zorder=4)
            f.disc((sx, y + f.fy(reach)), 1.15, fill=INK, zorder=4)

    def ramp(x0):
        # Two halves, not a smooth sweep: at 9 pt a continuous grey-to-green
        # gradient reads as one green line.
        for k, t in enumerate((0.0, 1.0)):
            ax.plot([x0 + 0.5 * glyph * k, x0 + 0.5 * glyph * (k + 1)],
                    [y, y], color=_blend(*E_ROUTE_INK, t), lw=E_CAPSULE_PT,
                    solid_capstyle="butt", zorder=3)

    x = f.fx(1.5)
    for draw, text in ((capsule, "route"), (stem, "signed field"),
                       (ramp, "alignment a")):
        draw(x)
        f.text((x + glyph + gap, y), text, size=PT_SMALL, color=MUTE,
               ha="left")
        x += glyph + gap + f.fx(_text_w_pt(ax, text, PT_SMALL)) + pad


def panel_alignment_design(ax):
    """C: one fixed dictionary, the task field turned into its span."""
    f = Frame(ax)
    for cell, (a, label) in zip(f.split(3, axis="x", gap_pt=4.0),
                                E_STAGES, strict=True):
        _draw_stage(f, Frame.inset(cell, bottom=f.fy(E_KEY_PT) / cell[3]),
                    a, label)
    _alignment_key(f)
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
    ax.set_xlabel("imposed subtree alignment  a")
    ax.set_ylabel("field-energy capture")
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
def panel_animal_pairs(ax, animal, mode=None):
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
    # The mode partition, folded in from what used to be its own panel: the
    # six animals' contrast vectors put most of their projection energy in
    # the signed mode.  It rides the empty upper-right corner as a thin
    # stacked bar, so the panel carries the per-animal contrast and the
    # decomposition of those same vectors together.
    if mode is not None:
        signed = float(mode["signed_energy_fraction"])
        bar_x0, bar_x1, bar_y = 0.40, 1.12, 0.185
        split = bar_x0 + signed * (bar_x1 - bar_x0)
        ax.plot([bar_x0, bar_x1], [bar_y, bar_y], color=mix("point_mlp", 26),
                lw=5.4, solid_capstyle="butt", zorder=2)
        ax.plot([bar_x0, split], [bar_y, bar_y], color=ROUTE, lw=5.4,
                solid_capstyle="butt", zorder=3)
        # The green segment is wide enough to carry its own value; the 16 %
        # remainder is not, so a hairline leader drops from its middle to a
        # label parked in the empty right shoulder of the panel.
        ax.text(0.5 * (bar_x0 + split), bar_y, f"signed {100 * signed:.0f}%",
                fontsize=PT_SMALL, color="white", ha="center", va="center",
                zorder=4)
        ax.text(bar_x1, bar_y - 0.046,
                f"common {100 * (1.0 - signed):.0f}%", fontsize=PT_SMALL,
                color=mix("point_mlp", 62), ha="right", va="center")

    ax.set_xlim(-0.16, 1.16)
    ax.set_xticks([0, 1], ["P+", f"P{MINUS}"])
    for label, color in zip(ax.get_xticklabels(), (PPLUS, PMINUS)):
        label.set_color(color)
    # The limits widen with row 2 rather than the data stretching into it:
    # the panel's information is the SLOPE of each animal's pair, so the
    # units-per-point of the y axis is held at what the 116.9 pt box had and
    # the extra 10 pt becomes air above and below, symmetric about the same
    # centre (−0.035).  The tick set does not move.
    ax.set_ylim(-0.30, 0.23)
    ax.set_yticks([-0.2, -0.1, 0.0, 0.1, 0.2])
    # Name the quantity exactly as the caption does; "dendritic
    # contrast" appeared nowhere in the caption or main text.
    ax.set_ylabel("source residual contrast")
    # The sign convention, the six animals and the 6/6 sign count are all
    # reported in the caption; the panel carries only the paired geometry.


# ── F: one partition of the contrast energy, drawn as one measured bar ───
#
# The promoted Supplementary Fig. S17c panel drew the common and the signed
# fraction as two separate vertical bars, each with its own animal-bootstrap
# whisker, and printed both percentages above them.  That states one quantity
# three times: the two fractions are one partition and sum to 1 by
# construction, so the second bar carries no information the first does not,
# the second whisker is the reflection of the first -- one uncertainty drawn
# twice -- and the printed numbers repeat what the axis already encodes.
#
# The first single-bar redraw kept that honesty but lost the uncertainty.  It
# hung the capped whisker in the WHITE ABOVE a 50 pt stacked bar and reached
# the 83.7 % boundary through a thin vertical connector, so the interval read
# as a detached element rather than as the uncertainty ON that boundary, and
# its right cap sat close enough to the 100 % end to look like a statement
# about the whole bar.  This is the conventional drawing instead: ONE
# measured bar for the signed fraction on a light 0-100 % track that keeps
# the partition visible, and a capped error bar at the END of that bar drawn
# on the bar's OWN centreline, where it can only be read as the uncertainty
# of that bar's length -- no floating whisker and no connector.  The bar is
# less than half its former thickness, so the bar and its error bar read as
# one object rather than as a slab with an afterthought above it.  Each mode
# is still named beside its own segment rather than in a legend, and the two
# percentages and the interval stay in the caption.
SIGNED_LABEL_PT = 9.2        # name band above the measured bar
SIGNED_LABEL_GAP_PT = 6.5    # that band -> the bar it names
BAR_H_PT = 20.0              # thin enough that bar and error bar are one mark
# The remainder is only 16 % of the axis, so its name cannot sit over its own
# segment; a longer leader makes the pointer, not the proximity, carry the
# association -- at 6 pt the name read as if it labelled the green bar above.
COMMON_LEADER_PT = 13.0      # bar bottom -> the remainder's own name
COMMON_LABEL_PT = 9.2        # direct label for the too-small remainder
TRACK_FILL_PCT = 12          # the 0-100 % track: the control gray, lightened
TRACK_EDGE_PCT = 34
ENERGY_TICKS = (0, 25, 50, 75, 100)


def panel_mode_energy(ax, mode):
    """F: where the six animal contrast vectors put their energy.

    One measured horizontal bar on a 0-100 % axis -- the signed P+/P− mode in
    the anatomy green -- standing on a light track that runs the full extent,
    so the remainder that is the common scalar mode stays visible as part of
    one partition.  Nothing is recomputed: the two fractions and the
    animal-bootstrap interval are read from the same ``mode_decomposition``
    block the supplementary panel read.
    """
    signed = 100.0 * float(mode["signed_energy_fraction"])
    common = 100.0 * float(mode["common_energy_fraction"])
    ci_lo, ci_hi = (100.0 * float(v)
                    for v in mode["animal_bootstrap_95_ci"])

    box = ax.get_position()
    h_pt = box.height * ax.get_figure().get_size_inches()[1] * 72.0

    def fy(pt):
        """Points measured down from the axes top -> axes y fraction."""
        return 1.0 - pt / h_pt

    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 1.0)

    # The bar band is a fixed stack of points -- name, bar, leader, name --
    # CENTRED in whatever height row 2 gives the panel, so a taller row buys
    # the panel air above and below its one mark instead of a thicker slab.
    block_pt = (SIGNED_LABEL_PT + SIGNED_LABEL_GAP_PT + BAR_H_PT
                + COMMON_LEADER_PT + COMMON_LABEL_PT)
    top_pt = max(0.5 * (h_pt - block_pt), 0.0)

    bar_top_pt = top_pt + SIGNED_LABEL_PT + SIGNED_LABEL_GAP_PT
    y_top, y_bot = fy(bar_top_pt), fy(bar_top_pt + BAR_H_PT)
    y_mid = 0.5 * (y_top + y_bot)

    # 1. the full extent, so the partition is still one quantity ...
    ax.barh(y_mid, 100.0, left=0.0, height=y_top - y_bot,
            color=mix("point_mlp", TRACK_FILL_PCT),
            edgecolor=mix("point_mlp", TRACK_EDGE_PCT), linewidth=LW_HAIR,
            zorder=2)
    # 2. ... and the measured part of it: one bar, one length.
    ax.barh(y_mid, signed, left=0.0, height=y_top - y_bot,
            # Every other data mark on this page uses the
            # manuscript selective-routing green; the schematic
            # dend green put two near-identical greens side by
            # side with different meanings.
            color=COLORS["shunting"], edgecolor=COLORS["edge"],
            linewidth=LW_EDGE, zorder=3)

    # The signed bar is named directly above its own run of green (the name
    # is 46 % of the axis wide and starts at 0, so it never overhangs the
    # 83.7 % boundary); the remainder is 16 % of the axis, about 21 pt, so it
    # is named just beneath itself on a hairline leader.  No legend.
    ax.text(0.0, fy(top_pt + SIGNED_LABEL_PT / 2.0),
            f"signed P+/P{MINUS} mode", ha="left", va="center",
            fontsize=PT_ANNOT, color=INK, zorder=5)
    x_common = signed + 0.5 * common
    label_pt = bar_top_pt + BAR_H_PT + COMMON_LEADER_PT + COMMON_LABEL_PT / 2.0
    ax.plot([x_common, x_common],
            [y_bot, fy(bar_top_pt + BAR_H_PT + COMMON_LEADER_PT)],
            color=MUTE, lw=LW_HAIR, solid_capstyle="round", zorder=2)
    ax.text(100.0, fy(label_pt), "common scalar mode", ha="right",
            va="center", fontsize=PT_ANNOT, color=INK, zorder=5)

    # The animal bootstrap, once, and conventionally: a capped error bar on
    # the END of the measured bar, on the bar's own centreline.
    _, caps, _ = ax.errorbar(
        [signed], [y_mid], xerr=[[signed - ci_lo], [ci_hi - signed]],
        fmt="none", ecolor=INK, elinewidth=LW_ERR,
        capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=5)
    for cap in caps:                     # a cap is a stroke, not a fill
        cap.set_markerfacecolor("none")

    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xticks(list(ENERGY_TICKS))
    # The caption and main text both call this projection energy.
    ax.set_xlabel("share of projection energy (%)")
    # The two percentages, the interval and the aggregation are in the
    # caption; the panel carries the proportion and its one uncertainty.


# ── G: the evidence ladder ───────────────────────────────────────────────
#
# The closing panel used to be three tinted boxes of bullets, which is a
# slide, not a figure.  It is drawn here as the paper's OWN credit hierarchy
# (Fig. 1c) turned into a ladder: four stacked rungs read from the bottom up
# -- neuron coordinate δᵤ, subtree address Aᵤ,ₖ, route gain, and the use of
# those routes for endogenous task credit in vivo -- so the closing panel
# answers "where does the evidence stop" spatially rather than in prose.
#
# Each rung carries the credit tree in the manipulation mode that DEFINES
# that level, in the vocabulary of Figures 1-4 and of panel C of this figure:
# the coordinate arrives at the soma and stops there (dashed barrier, one
# arrow in the adjoint blue); the address splits the arbor into two tinted
# subtree routes; the gain keeps those routes and thickens one of them; and
# the top rung is the same arbor with nothing lit at all.  The tree is the
# library's arbor truncated one level, exactly as panel C truncates it, so
# four terminals resolve at a 4-module rung height where eight would be a
# thicket; the library's own labels are never set at this size.
#
# EVIDENCE STATUS IS THE FILL OF THE STEP, not a word list: each rung stands
# on a full-width step inked shunting green where measurement supports the
# level, amber where the level holds only under a stated condition, and left
# open -- white, dashed outline, a broken step -- where nothing is
# established.  Read upward the steps go green, green, amber, broken, so the
# eye sees the evidence fade exactly where the paper says it stops before it
# reads a single word.  One three-swatch key at the foot names the three
# fills; there are no tier headings, no tinted tier boxes and no bullets.
#
# Beside each rung sits the short form of its condition or gap and one mute
# chip naming the evidence that carries it (``E,F`` is a panel pair of this
# figure, ``7g`` is Fig. 7g and ``8f`` is Fig. 8f).  Every shortened statement
# is printed in full, with its attribution, into the caption at build time.
LADDER_SUPPORTED = ROUTE                 # measurement supports the level
LADDER_CONDITIONAL = COLORS["local"]     # holds only under a stated condition
LADDER_OPEN = None                       # nothing established: an open step

# (glyph mode, rung name, symbol, symbol tone, short condition, chip, status)
# Index 0 is the BOTTOM rung: the ladder is read upward, in the hierarchy's
# own order, so the drawing order below is the reading order.
EVIDENCE_RUNGS = (
    ("coordinate", "coordinate", r"$\delta_u$", COLORS["additive"],
     "retrospective (n=6)", None, LADDER_CONDITIONAL),
    ("address", "subtree route", r"$A_{u,\cdot k}$", INK,
     "capacity, not use", None, LADDER_SUPPORTED),
    ("gain", "route gain", r"$\Lambda_k$", INK,
     "high conductance only", None, LADDER_CONDITIONAL),
    ("open", "endogenous task use", None, MUTE,
     "no anatomy alignment", None, LADDER_OPEN),
)
LADDER_KEY = (
    (LADDER_SUPPORTED, "supported"),
    (LADDER_CONDITIONAL, "conditional"),
    (LADDER_OPEN, "not established"),
)

# The ladder band, in points: type is points, so every reserve here is one.
RUNG_STEP_PT = 2.4           # the step a rung stands on (an area mark)
RUNG_STEP_GAP_PT = 0.9       # a rung -> the step it stands on
RUNG_GAP_PT = 4.0            # a step -> the NEXT rung up.  The compact
                             # gap above, so a step reads as belonging to
                             # the rung standing on it rather than as a
                             # divider floating between two of them.  The
                             # 10 pt row 2 gained is split between this gap
                             # and the glyph: the ladder is not four rungs
                             # closer together at a larger size, it is four
                             # larger rungs further apart.
# 6.8 pt type has a line box near 9 pt, so a 7.2 pt lead made consecutive
# condition lines overlap; the name band was tight against its type as well.
RUNG_NAME_PT = 9.6           # name band (7.2 pt type)
RUNG_LINE_PT = 9.4           # condition leading (6.8 pt type)
RUNG_SYMBOL_GAP_PT = 3.2     # rung name -> its symbol
GLYPH_GAP_PT = 3.5           # tree glyph -> text column
GLYPH_MIN_PT = 12.0          # below this the four-terminal arbor stops reading
RUNG_TEXT_R_PT = 1.5         # text column -> panel right edge
KEY_GAP_PT = 2.8             # last step -> the status key
KEY_LINE_PT = 9.4            # key row
KEY_SWATCH_PT = 4.6
KEY_LABEL_GAP_PT = 2.4       # swatch -> its label
KEY_ITEM_GAP_PT = 7.0        # between two key items
CHIP_PAD_X_PT = 2.2
CHIP_H_PT = 8.2
CHIP_GAP_PT = 2.4            # statement text -> chip
CHIP_CLEAR_PT = 1.5          # a chip never lands this close to the text
# Words a condition is not broken after when any other break will do.
HANGING_WORDS = frozenset(
    ("a", "an", "and", "in", "into", "of", "on", "only", "the", "to",
     "two", "six", "no", "not", "if", "when"))

# The rung glyph is panel C's stage tree -- the credit-tree library's arbor
# truncated one level, on the library's own coordinates -- so a rung here and
# a stage there are the same object, and both are the Figure 1-4 arbor with
# its two first-order subtrees kept and its eight terminals folded into four.
G_SPAN_X = E_SPAN_X
G_SPAN_Y = (-0.30, 2.60)                 # clears the soma disc and the tips
G_ASPECT = ((G_SPAN_X[1] - G_SPAN_X[0])
            / (G_SPAN_Y[1] - G_SPAN_Y[0]))
# The two subtree-route capsules, in the tints the library gives K = 2.
G_ROUTE_TINT = (mix("shunting", 20), mix("additive", 18))
G_CAPSULE_PT = 3.2                       # an area mark, not a line weight
# Gain mode: the same taper the library keys per edge, folded onto the
# truncated arbor -- one first-order subtree driven hard, the other weakly.
G_GAIN_LW = {("J1", "JL"): LW_ERR, ("J1", "JR"): LW_HAIR,
             ("JL", "T_LL"): LW_EDGE, ("JL", "T_LR"): LW_EDGE,
             ("JR", "T_RL"): LW_HAIR, ("JR", "T_RR"): LW_HAIR}


def _glyph_place(f, cell):
    """Map the truncated stage tree's coordinates into one rung cell."""
    x0, y0, w, h = cell

    def place(point):
        x, y = E_P[point] if isinstance(point, str) else point
        return (x0 + w * (x - G_SPAN_X[0]) / (G_SPAN_X[1] - G_SPAN_X[0]),
                y0 + h * (y - G_SPAN_Y[0]) / (G_SPAN_Y[1] - G_SPAN_Y[0]))

    return place


def _rung_glyph(f, cell, mode):
    """One rung's tree, decorated in the mode that defines its level."""
    ax = f.ax
    place = _glyph_place(f, cell)
    # The arbor is lit only where the level itself is defined on it: the
    # coordinate stops at the soma, so its arbor is the library's ghost tree
    # with one blue arrow; the address and the gain live ON the arbor, so
    # theirs is inked; and the top rung is the address arbor with nothing lit
    # at all -- the same two routes, in the de-emphasis tint, and no soma
    # arrival, which is exactly the claim that has no evidence behind it.
    lit = mode in ("address", "gain")
    ink = COLORS["dend"] if lit else GHOST
    if mode in ("address", "gain", "open"):
        tints = (G_ROUTE_TINT if lit
                 else (mix("mute", 13), mix("mute", 13)))
        for tint, chains in zip(tints, E_ROUTES, strict=True):
            for chain in chains:
                xy = np.array([place(point) for point in chain], dtype=float)
                ax.plot(xy[:, 0], xy[:, 1], color=tint, lw=G_CAPSULE_PT,
                        solid_capstyle="round", solid_joinstyle="round",
                        zorder=1.3)
    for a, b in E_TRUNK:
        ax.plot(*zip(place(a), place(b)), color=ink,
                lw=LW_DATA if mode == "gain" else LW_ERR,
                solid_capstyle="round", zorder=2.4)
    for edges, width in ((E_LIMBS, LW_EDGE), (E_TWIGS, LW_HAIR)):
        for a, b in edges:
            ax.plot(*zip(place(a), place(b)), color=ink,
                    lw=G_GAIN_LW[(a, b)] if mode == "gain" else width,
                    solid_capstyle="round", zorder=2.4)
    for point in ("J1", "JL", "JR"):
        f.disc(place(point), 1.05, fill="white", edge=ink, lw=LW_HAIR,
               zorder=3.2)
    for point in E_TERMINALS:
        f.disc(place(point), 0.95, fill=ink, zorder=3.2)
    soma_lit = mode != "open"
    f.disc(place("S"), 1.7,
           fill=COLORS["soma"] if soma_lit else mix("soma", 34),
           edge=RIM if soma_lit else GHOST, lw=LW_HAIR, zorder=3.4)
    if mode == "coordinate":
        # the coordinate arrives at the soma and stops there: the library's
        # dashed barrier across the trunk, and one arrow in the adjoint blue
        mid = place((0.03, 0.40))
        half = f.fx(3.2)
        barrier, = ax.plot([mid[0] - half, mid[0] + half],
                           [mid[1] - f.fy(0.9), mid[1] + f.fy(0.9)],
                           color=MUTE, lw=LW_HAIR, zorder=3.6,
                           solid_capstyle="butt")
        barrier.set_dashes((1.3, 1.2))
        f.arrow(place((1.62, 0.66)), place((0.34, 0.12)),
                color=COLORS["additive"], lw=LW_EDGE, head=3.2, rad=0.12,
                zorder=4.0)
    if mode == "gain":
        # the library's gain ring on the junction whose path conductance is
        # the one being read off
        f.disc(place("JL"), 3.0, fill="none", edge=INK, lw=LW_HAIR,
               zorder=4.2)


def _wrap_condition(ax, text, *, width_pt, reserve_pt, max_lines=2):
    """Break one condition into at most ``max_lines`` lines, or ``None``.

    ``reserve_pt`` is what the chip takes on the LAST line, so a condition is
    never set under its own tag.
    """
    def w(s):
        return _text_w_pt(ax, s, PT_SMALL)

    if w(text) + reserve_pt <= width_pt - CHIP_CLEAR_PT:
        return [text]
    if max_lines < 2:
        return None
    words = text.split()
    # Among the breaks that fit, take the one that BALANCES the two lines
    # rather than the greedy one: a 22 pt column sets a greedy break as a
    # full line over a one-word widow, and two even lines read better beside
    # a glyph the same height as the pair.  Two passes: the first refuses to
    # leave a function word hanging at the end of line one, the second takes
    # any break that fits.
    for avoid_hanging in (True, False):
        best = None
        for split in range(1, len(words)):
            if avoid_hanging and words[split - 1].lower() in HANGING_WORDS:
                continue
            first = " ".join(words[:split])
            second = " ".join(words[split:])
            w_first, w_second = w(first), w(second)
            if w_first > width_pt:
                continue
            if w_second + reserve_pt > width_pt - CHIP_CLEAR_PT:
                continue
            span = max(w_first, w_second + reserve_pt)
            if best is None or span < best[0]:
                best = (span, [first, second])
        if best is not None:
            return best[1]
    return None


def _chip(f, x_right, y, label, *, width_pt):
    """One small mute pill naming the evidence a rung rests on."""
    height = f.fy(CHIP_H_PT)
    x0 = x_right - f.fx(width_pt)
    f.ax.add_patch(FancyBboxPatch(
        (x0, y - height / 2.0), f.fx(width_pt), height,
        boxstyle=f"round,pad=0,rounding_size={height / 2.0}",
        facecolor=mix("mute", 10), edgecolor=mix("mute", 42),
        linewidth=LW_HAIR, zorder=5, transform=f.ax.transData, clip_on=False,
    ))
    f.text((x0 + f.fx(width_pt) / 2.0, y), label, size=PT_SMALL, color=MUTE,
           zorder=6)


def _step(f, y_pt, status, *, x0=0.0, width=1.0, height_pt=RUNG_STEP_PT,
          radius_pt=1.2):
    """One rung of the ladder, filled with the status of its evidence.

    A supported or conditional level stands on a solid step; a level with no
    evidence behind it stands on an OPEN step -- white, dashed outline -- so
    the ladder is visibly broken where the paper says the evidence stops.
    """
    height = f.fy(height_pt)
    patch = FancyBboxPatch(
        (x0, 1.0 - f.fy(y_pt) - height), width, height,
        boxstyle=f"round,pad=0,rounding_size={f.fy(radius_pt)}",
        facecolor="white" if status is None else status,
        edgecolor=mix("mute", 50) if status is None else "none",
        linewidth=LW_HAIR if status is None else 0.0,
        zorder=2.0, transform=f.ax.transData, clip_on=False,
    )
    if status is None:
        patch.set_linestyle((0, (2.4, 1.9)))
    f.ax.add_patch(patch)
    return patch


def _key_rows(ax, f):
    """Pack the three status swatches into as few full-width rows as fit."""
    items = [(status, label,
              KEY_SWATCH_PT + KEY_LABEL_GAP_PT
              + _text_w_pt(ax, label, PT_SMALL))
             for status, label in LADDER_KEY]
    rows, row, used = [], [], 0.0
    for item in items:
        step = item[2] + (KEY_ITEM_GAP_PT if row else 0.0)
        if row and used + step > f.w_pt:
            rows.append(row)
            row, used = [item], item[2]
            continue
        row.append(item)
        used += step
    if row:
        rows.append(row)
    return rows


def panel_evidence_boundary(ax):
    """H: the credit hierarchy as a ladder, each rung filled with its status.

    Nothing here is a statistic: the panel restates, in the manuscript's own
    hedged words, what the biological evidence carries at each level of the
    hierarchy the paper tests, and where it stops.
    """
    f = Frame(ax)
    key_rows = _key_rows(ax, f)
    key_h = KEY_LINE_PT * len(key_rows)
    ladder_h = f.h_pt - key_h - KEY_GAP_PT
    n = len(EVIDENCE_RUNGS)
    chips = [(0.0 if not rung[5]
              else _text_w_pt(ax, rung[5], PT_SMALL) + 2 * CHIP_PAD_X_PT)
             for rung in EVIDENCE_RUNGS]

    # 1. The glyph is as tall as the ladder can afford it: start from the
    #    uniform rung pitch and shrink until every condition has been set in
    #    at most two lines AND the four rungs still fit the band.  Shrinking
    #    the glyph widens the text column, so a rung never loses a word to
    #    make room for its own picture.
    plan = None
    glyph_h = (ladder_h - (n - 1) * RUNG_GAP_PT) / n \
        - RUNG_STEP_PT - RUNG_STEP_GAP_PT
    while glyph_h >= GLYPH_MIN_PT:
        glyph_w = glyph_h * G_ASPECT
        text_w = f.w_pt - glyph_w - GLYPH_GAP_PT - RUNG_TEXT_R_PT
        rows, total, ok = [], (n - 1) * RUNG_GAP_PT, True
        for rung, chip_w in zip(EVIDENCE_RUNGS, chips, strict=True):
            _, name, symbol, _, condition, chip, _ = rung
            head_w = _text_w_pt(ax, name, PT_ANNOT)
            if symbol:
                head_w += (RUNG_SYMBOL_GAP_PT
                           + _text_w_pt(ax, symbol, PT_ANNOT))
            on_head = bool(chip) and (head_w + CHIP_GAP_PT + chip_w
                                      <= text_w - CHIP_CLEAR_PT)
            lines = _wrap_condition(
                ax, condition, width_pt=text_w,
                reserve_pt=(0.0 if on_head or not chip
                            else chip_w + CHIP_GAP_PT))
            if lines is None:
                ok = False
                break
            # The chip rides the EARLIEST line that has room for it -- the
            # name row when the name and its symbol leave the space, else the
            # first condition line -- so a tag never floats a line below the
            # statement it tags.
            chip_line = (-1 if on_head else
                         (len(lines) - 1 if chip else None))
            if chip and not on_head:
                for line_no, line in enumerate(lines):
                    if (_text_w_pt(ax, line, PT_SMALL) + CHIP_GAP_PT + chip_w
                            <= text_w - CHIP_CLEAR_PT):
                        chip_line = line_no
                        break
            text_h = RUNG_NAME_PT + RUNG_LINE_PT * len(lines)
            rows.append((head_w, chip_line, lines, max(glyph_h, text_h)))
            total += rows[-1][3] + RUNG_STEP_PT + RUNG_STEP_GAP_PT
        if ok and total <= ladder_h + 1e-6:
            plan = (glyph_w, text_w, rows, ladder_h - total)
            break
        glyph_h -= 0.25
    if plan is None:
        raise ValueError("evidence ladder does not fit panel G at "
                         f"{f.w_pt:.1f} x {f.h_pt:.1f} pt")
    glyph_w, text_w, rows, slack = plan
    air = slack / n                      # spent as air inside every rung
    text_x = f.fx(glyph_w + GLYPH_GAP_PT)
    right = 1.0 - f.fx(RUNG_TEXT_R_PT)

    # 2. Draw from the TOP of the panel down, which is the top of the ladder
    #    down: ``EVIDENCE_RUNGS`` is in reading order, bottom first.
    y_pt = 0.0
    order = list(zip(EVIDENCE_RUNGS, chips, rows, strict=True))[::-1]
    for index, (rung, chip_w, row) in enumerate(order):
        mode, name, symbol, tone, _, chip, status = rung
        head_w, chip_line, lines, content_h = row
        content_h += air
        # the tree glyph literally STANDS on the rung's own step: it is
        # bottom-aligned in the band, not centred in it, so the soma sits on
        # the step whatever the text beside it does.
        gh = min(content_h, glyph_w / G_ASPECT)
        _rung_glyph(f, (0.0, 1.0 - f.fy(y_pt + content_h),
                        f.fx(glyph_w), f.fy(gh)), mode)
        # the rung name, its symbol and (where it fits) its evidence chip
        text_h = RUNG_NAME_PT + RUNG_LINE_PT * len(lines)
        ty = y_pt + (content_h - text_h) / 2.0
        head_y = 1.0 - f.fy(ty + RUNG_NAME_PT / 2.0)
        f.text((text_x, head_y), name, size=PT_ANNOT,
               color=MUTE if status is None else INK, ha="left")
        if symbol:
            f.text((text_x + f.fx(_text_w_pt(ax, name, PT_ANNOT)
                                  + RUNG_SYMBOL_GAP_PT), head_y), symbol,
                   size=PT_ANNOT, color=tone, ha="left")
        if chip_line is not None and chip_line < 0:
            _chip(f, right, head_y, chip, width_pt=chip_w)
        ty += RUNG_NAME_PT
        for line_no, line in enumerate(lines):
            y = 1.0 - f.fy(ty + RUNG_LINE_PT / 2.0)
            f.text((text_x, y), line, size=PT_SMALL, color=MUTE, ha="left")
            if chip_line is not None and line_no == chip_line:
                _chip(f, right, y, chip, width_pt=chip_w)
            ty += RUNG_LINE_PT
        y_pt += content_h + RUNG_STEP_GAP_PT
        _step(f, y_pt, status)
        y_pt += RUNG_STEP_PT + (RUNG_GAP_PT if index < n - 1 else 0.0)

    # 3. One key, three swatches, no tier headings.
    y_pt += KEY_GAP_PT
    for row in key_rows:
        y = 1.0 - f.fy(y_pt + KEY_LINE_PT / 2.0)
        x = 0.0
        for status, label, width in row:
            _step(f, y_pt + (KEY_LINE_PT - KEY_SWATCH_PT) / 2.0, status,
                  x0=x, width=f.fx(KEY_SWATCH_PT), height_pt=KEY_SWATCH_PT,
                  radius_pt=1.1)
            f.text((x + f.fx(KEY_SWATCH_PT + KEY_LABEL_GAP_PT), y), label,
                   size=PT_SMALL, color=MUTE, ha="left")
            x += f.fx(width + KEY_ITEM_GAP_PT)
        y_pt += KEY_LINE_PT
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

    # Row 0 begins with the experiment itself, then shows its primary null
    # and the matched-control effect summary.  The procedural schematic is a
    # compact three-module entry point; the label-heavy effect forest receives
    # the widest slot.
    # Panels sharing a module span must share an axes-box width, which a
    # schematic and a data panel cannot, so every row mixes spans.
    ax_a = canvas.panel("A", 0, 0, 6, schematic=True,
                        title="Feedback route dictionary")
    ax_b = canvas.panel("B", 0, 6, 6, grid="x",
                        title="Measured-response prediction")

    # Row 1: the label-heavy effect forest gets a row of its own rather than
    # the tail of row 0, beside the manipulation it is later contrasted with.
    ax_c = canvas.panel("C", 1, 0, 7, grid="x",
                        title="Morphology-specific effects")
    ax_d = canvas.panel("D", 1, 7, 5, schematic=True,
                        title="Controlled alignment")

    # Row 2: the alignment result, the animal coordinate WITH its mode
    # partition folded in, and the boundary the whole page argues for.
    ax_e = canvas.panel("E", 2, 0, 4, grid="y",
                        title="Alignment-controlled capture")
    ax_f = canvas.panel("F", 2, 4, 4, grid="y",
                        title="P+ − P− signed separation")
    ax_g = canvas.panel("G", 2, 8, 4, schematic=True,
                        title="Evidence ladder")

    # E and G reserve 16 pt for their y-axis furniture.  F has a comparable
    # four-module slot but no natural left reserve, so declare the same amount
    # explicitly and keep the closing row geometrically aligned.
    canvas.declare_reserve("F", left=LABEL_RESERVE_PT)
    # C's category labels are drawn as text, not tick labels, so measuring
    # cannot see them: undeclared they hang past the left margin and into the
    # column the row-leading panel letters own.  Declaring them insets the
    # whole first column -- A, C and E together -- which keeps the letters
    # leftmost without widening the page margin, and so without squeezing D.
    canvas.declare_reserve("C", left=C_LABEL_RESERVE_PT)

    panel_route_dictionary(ax_a)
    panel_tree_learning(ax_b, tree)
    truncated = panel_forest(ax_c, prespecified, all_scans, tree, original,
                             expanded, header_x=0.014)
    panel_alignment_design(ax_d)
    panel_controlled_alignment(ax_e, curves)
    panel_animal_pairs(ax_f, animal, mode)

    # One letter offset per grid column, so every letter sits the same
    # distance left of the column its panel starts in.
    for name in ("A", "B", "C", "D", "E", "F", "G"):
        canvas.add_letter(name, canvas.axes[name], dx_pt=24.0)

    # G is set in points inside its own axes box, and the box is only final
    # once the reserves are locked (row 2 yields whatever F's new percentage
    # axis cannot hang in the bottom margin).  Lock here, with every other
    # panel and every letter already on the canvas, so the tier bands are
    # laid out against the height the panel actually gets; ``save`` locks
    # again and finds nothing to move, because G declares no decoration.
    canvas.lock_reserves()
    panel_evidence_boundary(ax_g)

    problems = canvas.save(OUT, name="main_figure_09_native")
    # B's clipping note is generated, not typed: the bounds it prints are the
    # ones the panel just drew, so the caption cannot drift from the figure.
    notes = list(CAPTION_NOTES)
    if truncated:
        notes.insert(1, _truncation_note(truncated))
    for note in notes:
        print(note)
    for note in LETTER_MOVES:
        print(note)
    return problems


if __name__ == "__main__":
    issues = build()
    if issues:
        raise SystemExit(f"main_figure_09_native: {len(issues)} layout issues")

#!/usr/bin/env python3
"""Supplementary sheet S46 -- credit-operator phase theory -- as ONE native canvas.

The figure is authored at exactly ``journal_style.FIG_W`` (518.4 pt) and
emitted at scale 1.0, so every tick label is a type token on the compiled
page and every panel in a row is the same physical size.  Nothing here
recomputes a result: the panels read the same frozen source tables and apply
the same pivots and the same summary statistics as
``scripts/build_credit_phase_figure.py``.

Layout (12-column module, one horizontal and one vertical gutter)::

    row 0   A operator schematic  | B credit hierarchy
    row 1   C spectral advantage  | D hierarchy x resolution
    row 2   E projection boundary | F reliability | G utility

The letters run in reading order and A-E run as one uninterrupted
theoretical argument -- operator utility, the task hierarchy, spectral
alignment, hierarchy x resolution, the signal-noise boundary, the
reliability gain -- before G checks the moment utility against observed
one-step progress.  Supplementary Fig. S3 (``supplement_consolidation``)
pastes A, C, D and E of this sheet as its rows 1 and 2, so those four panels
are set at six modules and the rows they form fill the supplement text
width; F and G are carried here for provenance only, at three modules each.

Every explanatory or methodological note that used to be printed inside a
panel -- the marker dodges in D and F, the identity tag at the full-rank
endpoint of D, the colour keys of the two heatmaps -- lives in the caption,
which is where the reader looks for how a panel was drawn.  The two 5x5
signed heatmaps share one diverging treatment centred on zero and one
cell-annotation format; every cell prints its value, so neither carries a
colour key.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from credit_tree_schematics import draw_credit_tree, mix
from figure_canvas import (
    COLORS,
    DIV_CMAP,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    MARKERS,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    SEQ_CMAP,
    Margins,
    NativeCanvas,
    audit_native_pdf,
    token_subscript,
)
from journal_style import annotate_heatmap, strengthen, tint_patch


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "source_data" / "credit_phase_theory"
OUT = ROOT / "figures" / "supplementary" / "figure_S46_utility.pdf"

HEIGHT_IN = 490.0 / 72.0
# Every reserve this figure needs is paid for by the outer margins and by the
# two uniform gutters, never by a slice of one panel: the horizontal gutter
# carries the next panel's y label and tick column (34.8 pt for the widest
# panel) and the panel letter that sits left of it, and the vertical gutter
# carries a row's x label band plus the next row's letter and title band.  So
# no panel is ever carved on its own and every panel that starts in one grid
# column keeps one x0 and one axes width.
HGUTTER = 38.0
VGUTTER = 55.0
MARGINS = Margins(left=51.0, right=12.0, top=23.0, bottom=27.0)
# 2026-09-11: two six-module panels in rows 0 and 1; row 2 is one six-module
# heatmap and two three-module panels.  Supplementary Fig. S3 pastes A, C, D
# and E of this sheet as its rows 1 and 2, and at four modules those crops
# filled 64-70% of the supplement text width beneath a full-bleed third row;
# at six modules they fill it.  F and G are carried for provenance and take
# the three modules each that remain beside E.
ROW_WEIGHTS = (96.0, 116.0, 118.0)
LETTER_DX_PT = 34.0


# One colour per family, fixed for the whole figure.
C_ROUTE = COLORS["shunting"]      # the anatomy/route-of-interest series
C_CTRL = COLORS["point_mlp"]      # neutral control gray
C_SHUFFLE = COLORS["highlight"]   # shuffle rose (palette semantics)
C_ANTI = COLORS["additive"]       # anti-aligned route
C_ORACLE = COLORS["oracle"]
C_BP = COLORS["bp"]
# Task hierarchy H is an ordered factor, so it gets the sequential ramp
# rather than four unrelated hues (S5).
# 2026-09-11: stops spread so the two dark ends differ by 26 L* (they were
# 17 L* apart at 0.78/1.00 and the 0.78 stop was within 6 dE of the additive
# navy that panel F of the supplement sheet uses for depth bins).
H_COLORS = [SEQ_CMAP(v) for v in (0.30, 0.50, 0.68, 1.00)]


class _MinusFmt:
    """Heatmap cell values with a true minus sign (and never a bare "-0").

    ``decimals`` rounds half away from zero, so an exact 0.125 prints as
    0.13 (the reading a reader gets by hand) rather than Python's
    round-half-even 0.12.
    """

    def __init__(self, decimals: int = 2):
        self.decimals = int(decimals)

    def format(self, value: float) -> str:
        v = float(value)
        scale = 10.0 ** self.decimals
        v = math.copysign(math.floor(abs(v) * scale + 0.5) / scale, v)
        s = f"{v:.{self.decimals}f}"
        if s.lstrip("-").strip("0").strip(".") == "":
            s = s.lstrip("-")
        return s.replace("-", "−")


def signed_heatmap(ax, matrix, xlabels, ylabels):
    """The figure's ONE signed-heatmap treatment: DIV, zero-centred, annotated.

    Every cell prints its value, so the colour is a reading aid for sign and
    magnitude rather than a scale to be looked up: red is positive, blue is
    negative and the tint saturates with |value| towards the matrix's own
    maximum.  2026-09-11: the key rail that used to sit above the matrix is
    gone.  It cost 41% of the panel footprint, duplicated the printed cells,
    and in the spectral panel half of it (-55 to -1) keyed values that never
    occur; the matrix now owns the whole axes box.
    """
    limit = float(np.max(np.abs(matrix)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=DIV_CMAP,
                      norm=norm)
    ax.set_xticks(np.arange(len(xlabels)), xlabels)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    annotate_heatmap(ax, image, matrix, fmt=_MinusFmt(2),
                     fontsize=PT_SMALL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0.0)
    rows, cols = matrix.shape
    # A hairline frame around the matrix: the spines are hidden, so without it
    # zero-valued edge cells at the near-background centre colour dissolve
    # into the page and their annotations read as orphaned digits.  A line
    # loop, not a Rectangle: the overlap audit treats every patch as an
    # obstacle, and a matrix-sized patch would flag each cell annotation.
    ax.plot([-0.5, cols - 0.5, cols - 0.5, -0.5, -0.5],
            [-0.5, -0.5, rows - 0.5, rows - 0.5, -0.5],
            color=COLORS["grid"], lw=LW_HAIR, solid_joinstyle="miter",
            zorder=4)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    return image


# ── A: the credit operator ───────────────────────────────────────────────
def task_generator_schematic(ax) -> None:
    """B: the synthetic task as patterns over one set of terminals.

    An earlier draft put signal and noise at different heights of the arbor,
    which read as different dendritic LOCATIONS; they differ in pattern
    SCALE across the same terminals.  One tree, and above its eight leaves
    three aligned strips: the feedback routes (one value per subtree at the
    resolved depth), the task signal (piecewise-constant across whole
    subtrees -- the tree's coarse Haar patterns, H_c scales), and minibatch
    noise (leaf-by-leaf jitter).  Coarse-versus-fine becomes visible
    smoothness, not height.
    """
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    green = COLORS["dend"]
    blue = COLORS["additive"]

    leaf_x = np.linspace(0.355, 0.925, 8)
    lvl3 = leaf_x.reshape(4, 2).mean(axis=1)
    lvl2 = lvl3.reshape(2, 2).mean(axis=1)
    root_x = float(lvl2.mean())
    y_soma, y2, y3, y_leaf = 0.035, 0.140, 0.245, 0.350

    def edge(xa, ya, xb, yb, lw):
        ax.plot([xa, xb], [ya, yb], color=green, lw=lw,
                solid_capstyle="round", zorder=2)
    for x in lvl2:
        edge(root_x, y_soma, x, y2, LW_EDGE)
    for i, x in enumerate(lvl3):
        edge(lvl2[i // 2], y2, x, y3, LW_EDGE)
    for i, x in enumerate(leaf_x):
        edge(lvl3[i // 2], y3, x, y_leaf, LW_HAIR)
    ax.scatter([root_x], [y_soma], s=26.0, color=COLORS["soma"],
               edgecolor="white", linewidths=LW_HAIR, zorder=4)

    # strip 1, on the leaves themselves: the feedback routes
    y_route = y_leaf + 0.055
    half = (leaf_x[1] - leaf_x[0]) / 2.0
    # 2026-09-11: a tint patch, not a 4.4 pt stroke -- the stroke rule snaps
    # every open line above 1.35 pt down to LW_DATA, which turned the four
    # route bars into hairlines.
    for x in lvl3:
        tint_patch(ax, ("rect", x - half - 0.014, y_route - 0.022,
                        2.0 * half + 0.028, 0.044), color=blue, pct=45,
                   edge=False, zorder=3, radius_pt=2.0)

    # strips 2 and 3: the same eight terminals, two pattern scales
    signal = np.array([0.9, 0.9, 0.9, 0.9, -0.9, -0.9, -0.9, -0.9])
    noise = np.array([0.55, -0.65, 0.4, -0.5, 0.6, -0.4, -0.55, 0.5])
    for base, values, tone, reach in ((0.60, signal, green, 0.085),
                                      (0.88, noise, "#5F6B7E", 0.060)):
        ax.plot([leaf_x[0] - 0.03, leaf_x[-1] + 0.03], [base, base],
                color=COLORS["grid"], lw=LW_HAIR, zorder=2)
        for x, v in zip(leaf_x, values):
            tip = base + v * reach
            ax.plot([x, x], [base, tip], color=tone, lw=LW_EDGE,
                    solid_capstyle="round", zorder=3)
            ax.scatter([x], [tip], s=4.5, color=tone, zorder=4)

    # One line per label: the two-line forms stacked into each other and
    # into the strips.  The qualifiers each label dropped live in the
    # caption, which already states them in full.
    # Token-size subscripts (mathtext would shrink them to 4.9 pt).
    token_subscript(ax, 0.01, y_route, "routes (D", "r", ")", size=PT_SMALL,
                    color=blue)
    token_subscript(ax, 0.01, 0.60, "signal μ (H", "c", ")", size=PT_SMALL,
                    color=green)
    ax.text(0.01, 0.88, "noise ξ", fontsize=PT_SMALL, color="#5F6B7E",
            ha="left", va="center")


def operator_schematic(ax) -> None:
    """Restricted-route credit tree feeding the guaranteed-utility ratio.

    The drawing is the shared credit-tree vocabulary in *address* mode (the
    K = 4 nested route capsules this figure manipulates): the task gradient
    enters at the soma, the route capsules carry it, and the routed update
    leaves at the canopy.  Underneath, on a recessive card that spans the
    cell, sits the guarantee the phase panels sweep.  The world frame is
    sized in points from the cell itself, so the schematic fills its cell
    instead of floating inside it: the tree-plus-card height sets the point
    size of one tree unit and the drawing is centred in whatever width the
    six-module slot gives it, with the mean-gradient label on the left of
    the soma and the update / routes labels on the right of the canopy.
    """
    fig_w, fig_h = ax.figure.get_size_inches()
    box = ax.get_position()
    w_pt = box.width * fig_w * 72.0
    h_pt = box.height * fig_h * 72.0
    y_hi, y_lo = 3.50, -1.72                      # canopy top .. card bottom
    unit = h_pt / (y_hi - y_lo)                   # points per tree unit
    span = w_pt / unit
    x_lo = -0.5 * span
    x_hi = 0.5 * span
    draw_credit_tree(ax, mode="address", K=4, labels=False, scale=0.85,
                     xlim=(x_lo, x_hi), ylim=(y_lo, y_hi))
    # The four capsules ARE the route subspaces of M, so they take one light
    # hue -- the figure-wide route green -- instead of the shared vocabulary's
    # four family tints, which collided with the family palette of F and H
    # (green=trained, purple=imposed).  Two alternating tints of that one hue
    # keep adjacent capsules separable where they meet near a junction.
    # 2026-09-11: the vocabulary still emits the capsules as 9 pt open
    # strokes, which the stroke rule now snaps to LW_DATA hairlines; they are
    # re-drawn here as the filled ribbons the rule asks for (tint_patch), on
    # the same chains.
    ax.apply_aspect()
    capsules = [ln for ln in ax.lines if abs(ln.get_zorder() - 1.4) < 1e-9]
    for i, line in enumerate(capsules):
        tone = mix("shunting", 26 if (i // 2) % 2 == 0 else 14)
        chain = list(zip(line.get_xdata(), line.get_ydata()))
        tint_patch(ax, ("ribbon", [chain], float(line.get_linewidth())),
                   color=tone, face=tone, edge_color=strengthen(tone, 1.6),
                   lw=LW_HAIR, zorder=1.4)
        line.remove()

    # Right of the canopy, top to bottom: the operator flow.
    lab_x = 4.35
    ax.add_patch(FancyArrowPatch(
        (2.32, 2.30), (3.32, 2.70), arrowstyle="-|>", mutation_scale=7,
        connectionstyle="arc3,rad=0.12", lw=LW_REF, color=COLORS["ink"],
        capstyle="round", zorder=4.5))
    ax.text(3.46, 2.66, "update\n−η M(μ + ξ)", ha="left",
            va="center", fontsize=PT_SMALL, color=COLORS["ink"],
            linespacing=1.25)
    # An arrow, like this panel's two other annotations, and stopped short of
    # the label: as a bare line starting at 3.55 it ran 2.4 pt into the "routes
    # M" text and pointed at nothing in particular.  The head now lands on the
    # route capsules the label names.
    ax.add_patch(FancyArrowPatch(
        (3.20, 1.47), (2.34, 1.79), arrowstyle="-|>", mutation_scale=7,
        connectionstyle="arc3,rad=-0.10", lw=LW_REF, color=C_ROUTE,
        capstyle="round", zorder=4.5))
    ax.text(lab_x, 1.45, "routes M", ha="center", va="center",
            fontsize=PT_SMALL, color=C_ROUTE)
    # Left of the soma: the mean gradient enters there.
    ax.add_patch(FancyArrowPatch(
        (-2.55, 0.34), (-0.36, 0.02), arrowstyle="-|>", mutation_scale=7,
        connectionstyle="arc3,rad=0.10", lw=LW_REF, color=C_ORACLE,
        capstyle="round", zorder=4.5))
    ax.text(-3.55, 0.62, "mean\ngradient μ", ha="center", va="center",
            fontsize=PT_SMALL, color=C_ORACLE, linespacing=1.25)

    # The guarantee, on a recessive card that spans the cell.
    card_x0, card_x1 = x_lo + 0.06, x_hi - 0.06
    card_y1 = -0.30
    card_y0 = y_lo
    ax.add_patch(FancyBboxPatch(
        (card_x0, card_y0), card_x1 - card_x0, card_y1 - card_y0,
        boxstyle="round,pad=0.10", facecolor=COLORS["panel_bg"],
        edgecolor=COLORS["grid"], lw=LW_HAIR, zorder=0.5))
    # Spell out the two moments instead of printing a nested fraction whose
    # subscripts become illegible at manuscript placement; the positive part
    # is written max(., 0) because a subscript "+" has no token-size glyph.
    # The caption retains the exact mathematical definition.
    frac_x = 0.5 * (card_x0 + card_x1)
    ax.text(
        frac_x, 0.5 * (card_y0 + card_y1),
        "U(M) = max(mean alignment, 0)² /\n(2L × mean squared update)",
        ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["ink"],
        linespacing=1.3,
    )


def main() -> None:
    spectral = pd.read_csv(PHASE / "spectral_phase_summary.csv")
    depth = pd.read_csv(PHASE / "depth_training_summary.csv")
    projection = pd.read_csv(PHASE / "projection_phase_summary.csv")
    reliability = pd.read_csv(PHASE / "reliability_phase_summary.csv")

    canvas = NativeCanvas(
        HEIGHT_IN, 3, row_weights=list(ROW_WEIGHTS),
        hgutter_pt=HGUTTER, vgutter_pt=VGUTTER, margins=MARGINS,
        letters=False,
    )
    # Titles name the quantity a panel shows where its axes do not: the two
    # annotated heatmaps carry no colour key, so their titles name the
    # printed cell value, and D's title names the factor its curves vary
    # (its x axis already says "route resolution").
    ax_a = canvas.panel("A", 0, 0, 6, schematic=True,
                        title="Fixed-operator special case")
    ax_b = canvas.panel("B", 0, 6, 6, schematic=True,
                        title="Credit hierarchy")
    ax_c = canvas.panel("C", 1, 0, 6, title="Spectral capture advantage")
    ax_d = canvas.panel("D", 1, 6, 6, title="Hierarchy × resolution")
    ax_e = canvas.panel("E", 2, 0, 6, title="Projection boundary: Δ loss")
    ax_f = canvas.panel("F", 2, 6, 3, title="Reliability gains")
    ax_g = canvas.panel("G", 2, 9, 3, title="Utility reanalysis")
    # Letters 34 pt left of their module column rather than the default 16:
    # D's y axis ("final loss" beside 0.03-3 tick labels) reserves 28 pt, so
    # at 16 pt the letter sat INSIDE that reserve and the supplement crop
    # rule (a panel's clip starts 3 pt left of its letter) handed D's y label
    # to panel C.  One offset for every letter keeps each column's letters
    # on one x.
    for name, ax in canvas.axes.items():
        canvas.add_letter(name, ax, dx_pt=LETTER_DX_PT)

    # ── A ───────────────────────────────────────────────────────────────
    operator_schematic(ax_a)

    # ── B: the synthetic task every phase panel sweeps ───────────────────
    task_generator_schematic(ax_b)

    # ── B: ancestry-minus-random spectral capture (signed) ──────────────
    wide = spectral.pivot_table(
        index=["alignment", "budget_k"], columns="method",
        values="mean_spectral_capture")
    advantage = (wide.ancestry - wide.random_rank).unstack("budget_k")
    # 2026-09-11: cells print the fraction itself (0.55, not 55 x 10^-2),
    # so no unit tag is needed and no superscript has to be set below the
    # type floor.
    signed_heatmap(
        ax_c, advantage.to_numpy(),
        [str(v) for v in advantage.columns],
        [f"{v:.2f}" for v in advantage.index])
    ax_c.set_xlabel("route budget K   (16 = full rank)", labelpad=2.0)
    ax_c.set_ylabel("covariance mixture ρ", labelpad=1.5)

    # ── C: final loss versus route resolution, one series per task depth ─
    # Small enough that a fanned marker still covers its own curve (the caption
    # states the fan); +/-0.20 planted markers at impossible x-values.
    depth_marker_dx = {3: -0.07, 4: 0.07}
    aligned = depth[depth.method.eq("aligned_tree")].pivot(
        index="task_depth", columns="model_depth",
        values="mean_final_population_loss")
    for task_depth, color, marker in zip(sorted(aligned.index), H_COLORS,
                                         MARKERS):
        part = depth[depth.method.eq("aligned_tree")
                     & depth.task_depth.eq(task_depth)].sort_values(
                         "model_depth")
        xs = part.model_depth.to_numpy(dtype=float)
        ys = part.mean_final_population_loss.to_numpy(dtype=float)
        # Only the MARKER column is dodged, exactly as panel E does it: the
        # line, and the 95% ribbon under it, stay at the true x, so a ribbon
        # can never sit beside the mean it belongs to and no segment of a
        # log-loss curve is given a distorted run.
        fanned = (np.isin(xs, (1.0, 2.0)) if task_depth in depth_marker_dx
                  else np.zeros(xs.shape, dtype=bool))
        # "hierarchy n", not "$H_c = n$": mathtext sets the subscript at
        # 4.9 pt, below the type floor; the caption ties the word to H_c.
        ax_d.plot(xs, ys, color=color, marker=marker, ms=MARKER_MS,
                  lw=LW_DATA, mec="white", mew=LW_HAIR,
                  markevery=list(np.flatnonzero(~fanned)),
                  label=f"hierarchy {task_depth}")
        if fanned.any():
            ax_d.scatter(xs[fanned] + depth_marker_dx[task_depth], ys[fanned],
                         color=color, marker=marker, s=MARKER_MS ** 2,
                         zorder=3, edgecolors="white", linewidths=LW_HAIR)
        # 2026-09-11: 0.22, not 0.10 -- at 0.10 the band vanished under the
        # two dark series and only the pale ones showed a halo.
        ax_d.fill_between(part.model_depth,
                          part.ci95_low_final_population_loss,
                          part.ci95_high_final_population_loss, color=color,
                          alpha=0.22, linewidth=0)
    ax_d.set_yscale("log")
    ax_d.set_yticks([0.03, 0.1, 0.3, 1.0, 3.0])
    ax_d.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
    ax_d.set_xticks([1, 2, 3, 4])
    ax_d.set_xlim(0.70, 4.30)
    ax_d.set_xlabel("route resolution (4 = full rank)")
    ax_d.set_ylabel("final loss")
    # 2026-09-11: the "identity M = I" tag that sat at the foot of the
    # D_r = 4 column is gone -- the caption states that the full-rank
    # endpoint is the identity operator -- and the panel title no longer
    # repeats the x-axis label.
    ax_d.legend(loc="upper left", ncol=2, frameon=False, fontsize=PT_LEGEND,
                handlelength=1.3, handletextpad=0.4, labelspacing=0.25,
                columnspacing=0.8, borderaxespad=0.2)

    # ── D: signed loss change of the common route projection ────────────
    projection_wide = projection.pivot_table(
        index=["signal_retention", "noise_retention"], columns="method",
        values="mean_expected_population_loss")
    delta = (projection_wide.bp_plus_route_projection
             - projection_wide.full_stochastic_bp).unstack("signal_retention")
    signed_heatmap(
        ax_e, delta.to_numpy(),
        [f"{v:.2f}" for v in delta.columns],
        [f"{v:.2f}" for v in delta.index])
    ax_e.plot([-0.5, 0.5, 0.5, 1.5, 1.5, 3.5, 3.5],
              [1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5],
              ls="--", color=COLORS["mute"], lw=LW_REF, zorder=3)
    # The sampled fractions are unequally spaced (0.25, 0.50, 0.75, 0.90,
    # 1.00) and drawn as equal cells, so the dashed line is the cell-wise
    # sign boundary, not the diagonal f_noise = f_sig; the caption says so.
    # f_sig / f_noise are named in the caption: a mathtext subscript would
    # be set at 5.6 pt, below the type floor.
    ax_e.set_xlabel("retained signal fraction", labelpad=2.0)
    ax_e.set_ylabel("retained noise fraction", labelpad=1.5)

    # ── E: one-step reliability gains for the four gain policies ────────
    reliability_styles = [
        ("reliability_aligned", "SNR-aligned", C_ROUTE, MARKERS[0]),
        ("best_global_gain", "best global", C_CTRL, MARKERS[1]),
        ("shuffled_shunting", "shuffled", C_SHUFFLE, MARKERS[2]),
        ("anti_aligned_shunting", "anti-aligned", C_ANTI, MARKERS[3]),
    ]
    # All four series coincide exactly at heterogeneity 0; the fan is small
    # enough that every marker still covers x=0 (the caption states the fan),
    # where +/-0.165 put the outer markers at visibly impossible x-values.
    reliability_marker_dx = {
        "reliability_aligned": -0.06, "best_global_gain": -0.02,
        "shuffled_shunting": 0.02, "anti_aligned_shunting": 0.06,
    }
    for method, label, color, marker in reliability_styles:
        part = reliability[reliability.method.eq(method)].sort_values(
            "reliability_heterogeneity")
        xs = part.reliability_heterogeneity.to_numpy(dtype=float)
        ys = part.mean_population_loss_decrease.to_numpy(dtype=float)
        fanned = xs == 0.0
        ax_f.plot(xs, ys, color=color, marker=marker, ms=MARKER_MS,
                  lw=LW_DATA, label=label,
                  markevery=list(np.flatnonzero(~fanned)),
                  mec="white", mew=LW_HAIR)
        if fanned.any():
            ax_f.scatter(xs[fanned] + reliability_marker_dx[method],
                         ys[fanned], color=color, marker=marker,
                         s=MARKER_MS ** 2, zorder=3, edgecolors="white",
                         linewidths=LW_HAIR)
        ax_f.fill_between(part.reliability_heterogeneity,
                          part.ci95_low_population_loss_decrease,
                          part.ci95_high_population_loss_decrease,
                          color=color, alpha=0.08, linewidth=0)
    ax_f.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_f.margins(x=0.115)
    # Pinned, not auto-located: one tick per swept heterogeneity level, so the
    # axis reads the same whatever width the module grid gives this panel.
    ax_f.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_f.set_ylim(-4.0, 3.9)
    ax_f.set_xlabel("branch-SNR heterogeneity")
    ax_f.set_ylabel("one-step loss decrease")
    ax_f.legend(loc="lower left", ncol=1, frameon=False, fontsize=PT_LEGEND,
                handlelength=1.3, handletextpad=0.4, labelspacing=0.28,
                borderaxespad=0.2)

    # ── F: phase utility predicts observed one-step progress ────────────
    merged = pd.read_csv(ROOT / "source_data/review_evidence_reanalysis/moment_scores.csv")
    merged = merged[merged.architecture.eq("dendritic_tree")]
    utility = "deterministic_maximum_bound_decrease"
    # U(M) is defined as 0 wherever the routed update is not positively
    # aligned (mu^T M mu <= 0, optimal step 0): 214 of the 540 fits.  Those
    # sit on the x = 0 wall by definition, not by measurement, so they are
    # drawn as a rug of mute horizontal ticks at their true (0, P1)
    # coordinates -- a strip on the wall rather than a column of the same
    # dots as the measured spread -- and the count is read off the frozen
    # table so the note can never drift from the drawing.
    clamped = merged[utility].eq(0.0)
    assert bool((merged.loc[clamped, "retained_signal_inner_product"]
                 <= 0.0).all())
    spread = merged[~clamped]
    wall = merged[clamped]
    ax_g.scatter(spread[utility],
                 spread.norm_matched_one_step_progress,
                 s=7, alpha=0.30, edgecolors="none", color=COLORS["additive"])
    ax_g.scatter(wall[utility],
                 wall.norm_matched_one_step_progress,
                 s=22, alpha=0.40, marker="_", linewidths=LW_EDGE,
                 color=COLORS["mute"], zorder=2.5)
    ax_g.text(0.14, -0.74,
              f"U = 0: not\npositively aligned\n({len(wall)}/{len(merged)})",
              ha="left", va="center", fontsize=PT_SMALL,
              color=COLORS["mute"], linespacing=1.15)
    ax_g.set_ylim(-1.42, 1.22)
    ax_g.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    # The Spearman coefficient, its seed-block interval and n are reported in
    # the caption; printing them on the panel duplicated the caption text.
    ax_g.set_xlabel("deterministic moment utility")
    ax_g.set_ylabel("observed one-step progress")

    from journal_style import style_direct_color_labels
    style_direct_color_labels(canvas.fig)
    problems = canvas.save(OUT, name="supplementary_figure_S46_utility")
    for violation in audit_native_pdf(OUT):
        print(f"    {violation}")
    return problems


if __name__ == "__main__":
    main()

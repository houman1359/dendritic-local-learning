#!/usr/bin/env python3
"""Main Figure 1 as ONE native full-width canvas (no sub-block rescaling).

Figure 1 is entirely schematic, so the whole page is drawn here from the
shared credit-tree vocabulary (``credit_tree_schematics``) on a single
:class:`figure_canvas.NativeCanvas`.  Nothing is pre-rendered and nothing is
scaled afterwards, so a 7.2 pt annotation is 7.2 pt in the compiled PDF and
an ``LW_EDGE`` stroke is 0.7 pt everywhere on the page.

Layout (one 12-module grid, two half-width columns and a closing band)::

    +--------------------------+--------------------------+
    | A  point unit -> tree    | B  network layer         |
    +--------------------------+--------------------------+
    | C  coordinate -> address | D  eligibility x error   |
    |    -> gain               |                          |
    +--------------------------+--------------------------+
    | E  roadmap of the Results: six stages, full width    |
    +------------------------------------------------------+

Every panel of a row is the same axes-box height and every panel of a grid
column the same width: A, B, C and D each claim six of the twelve modules
and E claims all twelve, so the only size difference on the page is a whole
number of modules.  A and B are the feeders, C is the coordinate ladder, D
is the equation anchor and E is the closing band whose six cards run in the
order of the Results subsections: exact factorization, the coordinate and
address tests, the credit-operator boundary, task-aligned physical depth,
anatomical routes with conductance gain, and the functional boundary.

Emits ``figures/components/main_figure_01_native.pdf`` (+ 600 dpi PNG), the
path ``assemble_compact_main_figures.emit_native(1)`` copies verbatim into
``figures/main/figure_01.pdf``.

Panels A, C and D keep the content of ``build_journal_figures.figure1``
(``_panel_point_vs_dendritic``, ``_panel_credit_hierarchy`` and
``_panel_general_adjoint``): same schematic vocabulary, same symbols, same
equations.  Panel B adds the neuron-specific feedback return the caption
describes, and panel E is the six-stage roadmap above.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import credit_tree_schematics as CT  # noqa: E402
from credit_tree_schematics import GHOST, RIM, draw_credit_tree, mix  # noqa: E402
from figure_canvas import (  # noqa: E402
    COLORS,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    PT_ANNOT,
    PT_LABEL,
    PT_SMALL,
    PT_TITLE,
    Margins,
    NativeCanvas,
)
from native_schematics import Frame, _wrap_to_width  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ROOT / "figures" / "components"
OUT = COMPONENTS / "main_figure_01_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
BLUE = COLORS["additive"]
GREEN = COLORS["shunting"]
DEND = COLORS["dend"]
SOMA = COLORS["soma"]
EXC = COLORS["exc"]
INH = COLORS["inh"]
EDGE = COLORS["edge"]

# ── canvas geometry, in points (one module grid, one h- and one v-gutter) ──
CANVAS_H_PT = 453.0            # 518.4 / 453.0 = 1.14 aspect
MARGINS = Margins(left=32.0, right=8.0, top=26.0, bottom=9.0)
HGUTTER = 22.0
VGUTTER = 26.0
# Two 122 pt rows of half-width panels and one 130 pt closing band.  With six
# roadmap cards across 478.4 pt each card is ~71 pt wide, so its width-limited
# tree glyph stands ~55 pt tall; 130 pt is that glyph plus the card's two text
# bands and paddings, which keeps the closing band free of dead vertical space
# while its row-mates stay identical.
ROW_PT = [122.0, 122.0, 130.0]
LETTER_DX = 20.0
LETTER_DY = 5.0
TITLE_PAD = 4.0
LW_ERR_ARROW = 0.95        # LW_ERR: the coordinate arrow weight

# Credit-tree frames: the library's own limits, quoted so a tree inset can be
# given a rectangle of exactly the right aspect instead of floating in one.
TREE_XL = (-2.55, 2.55)
# The frame clears the K=4 address capsules, whose fat round caps stand ~0.33
# data units above the terminal tips; the library's own 3.32 top clipped them.
TREE_YL = (-0.72, 3.60)
TREE_ASPECT = (TREE_XL[1] - TREE_XL[0]) / (TREE_YL[1] - TREE_YL[0])
# Panel C stacks three trees, so its insets use a frame cropped to the ink the
# tree actually carries: with the library's own 0.53-unit dead strip under the
# soma the three stages would be separated by a 17 pt blank strip, i.e. a
# third internal spacing value on a page that declares only 22 and 20 pt.
TREE_YL_TIGHT = (-0.32, 3.45)

SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"


# ── shared helpers ────────────────────────────────────────────────────────
def tree_inset(frame, rect, *, mode, K=4, shunted=True, labels=False,
               scale=1.0, arrow_scale=None, hide_arrows=False,
               xlim=TREE_XL, ylim=TREE_YL):
    """Credit tree filling ``rect`` (frame fractions) at library proportions."""
    aspect = (xlim[1] - xlim[0]) / (ylim[1] - ylim[0])
    x0, y0, w, h = rect
    w_pt, h_pt = w * frame.w_pt, h * frame.h_pt
    if w_pt / h_pt > aspect:                    # height-limited
        fit_h, fit_w = h_pt, h_pt * aspect
    else:                                        # width-limited
        fit_w, fit_h = w_pt, w_pt / aspect
    box = (x0 + (w - frame.fx(fit_w)) / 2.0,
           y0 + (h - frame.fy(fit_h)) / 2.0,
           frame.fx(fit_w), frame.fy(fit_h))
    sub = frame.ax.inset_axes(box, transform=frame.ax.transData, zorder=3)
    sub.set_facecolor("none")
    draw_credit_tree(sub, mode=mode, K=K, shunted=shunted, scale=scale,
                     labels=labels, xlim=xlim, ylim=ylim)
    if hide_arrows or arrow_scale is not None:
        for patch in sub.patches:
            if isinstance(patch, FancyArrowPatch):
                if hide_arrows:
                    patch.set_visible(False)
                else:
                    patch.set_mutation_scale(arrow_scale)
    return sub, box


def label_tone(color, *, max_luma=0.30):
    """Darken a palette hue with ink until label text is print-legible.

    Every stream name uses the same rule, so the ribbon keeps one typographic
    convention while the pale hues (rose, salmon) stop reading as gray.
    """
    from matplotlib.colors import to_rgb

    for pct in range(100, 39, -4):
        rgb = mix(color, pct, "ink")
        luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        if luma <= max_luma:
            return rgb
    return to_rgb(mix(color, 40, "ink"))


def mini_tree(frame, cx, cy, half_h, *, color=DEND):
    """Four-tip -> two-junction -> soma icon in the anatomy vocabulary.

    The same object as the full credit tree, drawn small: tips taper to
    ``LW_HAIR``, the two junctions are open circles and the soma is the
    filled ``COLORS['soma']`` disc, so a unit here and the tree in A read as
    the same cell.
    """
    dy = half_h / 1.5
    tip_x = cx - frame.fx(18.0)
    mid_x = cx - frame.fx(9.5)
    tip_ys = [cy + 1.55 * dy, cy + 0.50 * dy, cy - 0.50 * dy, cy - 1.55 * dy]
    mid_ys = [cy + 0.98 * dy, cy - 0.98 * dy]
    for i, my in enumerate(mid_ys):
        for ly in tip_ys[2 * i:2 * i + 2]:
            frame.ax.plot([tip_x, mid_x], [ly, my], color=color,
                          lw=LW_HAIR, solid_capstyle="round", zorder=3)
    for my in mid_ys:
        frame.ax.plot([mid_x, cx], [my, cy], color=color, lw=LW_EDGE,
                      solid_capstyle="round", zorder=3)
    for my in mid_ys:
        frame.disc((mid_x, my), 0.9, fill="white", edge=color, lw=LW_HAIR,
                   zorder=4)
    frame.disc((cx, cy), 2.6, fill=SOMA, edge=RIM, lw=LW_HAIR, zorder=5)
    return tip_x, cx


# ── A: point unit -> dendritic tree ───────────────────────────────────────
def panel_a(ax):
    f = Frame(ax)
    label_y = f.fy(11.0)

    soma = (0.235, 0.585)
    fan_x = 0.020
    for y in (0.930, 0.770, 0.455, 0.295):
        ax.plot([fan_x, soma[0] - f.fx(6.6)], [y, soma[1]], color=MUTE,
                lw=LW_EDGE, solid_capstyle="round", zorder=2)
        ax.plot([fan_x], [y], marker="o", ms=3.0, mfc="white", mec=MUTE,
                mew=LW_EDGE, ls="none", zorder=3)
    ax.text(fan_x + f.fx(5.2), 0.968, "xᵢ", ha="left", va="center",
            fontsize=PT_SMALL, color=INK)
    f.disc(soma, 5.8, fill=SOMA, edge=RIM, lw=LW_EDGE, zorder=4)
    # This is the title's "point unit -> dendritic tree" mapping arrow, so it
    # takes the mute feedforward ink.  In blue it read as the neuron EMITTING
    # a coordinate, since a blue arrow means delivered error everywhere else.
    f.arrow((soma[0] + f.fx(7.2), soma[1]), (0.430, soma[1]), color=MUTE,
            lw=LW_ERR_ARROW, head=3.6, zorder=5)
    ax.text(0.215, label_y, "one neuronal coordinate  δᵤ", ha="center",
            va="center", fontsize=PT_ANNOT, color=BLUE)

    # The tree fills its sub-cell edge to edge and starts exactly one
    # h-gutter (22 pt) right of the point unit, so the page carries no third
    # internal spacing value.
    gap = f.fx(HGUTTER)
    tree_x0 = 0.430 + gap
    tree_ax, tree_box = tree_inset(
        f, (tree_x0, f.fy(13.0), 1.0 - tree_x0, 1.0 - f.fy(13.0)),
        mode="address", K=4, scale=0.95, arrow_scale=0.65)
    ax.text(tree_box[0] + tree_box[2] / 2.0, label_y, "route signals  cᵤ,ₖ",
            ha="center", va="center", fontsize=PT_ANNOT, color=INK)



# ── B: network layer ──────────────────────────────────────────────────────
def panel_b(ax):
    f = Frame(ax)
    unit_ys = (0.910, 0.692, 0.474, 0.256)
    bus_e, bus_i = 0.215, 0.245
    unit_x = 0.435
    out_x = 0.615
    out_bus = 0.700
    box_x0, box_x1 = 0.735, 0.905
    delta_x = 0.975

    for y0, y1, key, text in ((0.850, 0.980, "exc", "E pool"),
                              (0.135, 0.265, "inh", "I pool")):
        f.group((0.0, y0, 0.150, y1 - y0), tint=mix(key, 10),
                edge=COLORS[key], lw=LW_EDGE, radius_pt=2.5, zorder=1.0)
        ax.text(0.075, (y0 + y1) / 2.0, text, ha="center", va="center",
                fontsize=PT_SMALL, color=COLORS[key])

    # The two rails stop short of each other's feed height, so the pool
    # arrows reach their own rail without ever crossing the other one.
    f.arrow((0.155, 0.915), (bus_e, 0.915), color=EXC, lw=LW_EDGE, head=3.4)
    f.arrow((0.155, 0.200), (bus_i, 0.200), color=INH, lw=LW_EDGE, head=3.4)
    ax.plot([bus_e, bus_e], [0.236, 0.915], color=EXC, lw=LW_EDGE,
            alpha=0.80, solid_capstyle="round", zorder=2)
    ax.plot([bus_i, bus_i], [0.200, 0.880], color=INH, lw=LW_EDGE,
            alpha=0.80, solid_capstyle="round", zorder=2)

    half_h = 0.078
    for index, y in enumerate(unit_ys, start=1):
        f.arrow((bus_e, y + f.fy(5.0)), (unit_x - f.fx(19.0), y + f.fy(2.4)),
                color=EXC, lw=LW_HAIR, head=3.4)
        f.arrow((bus_i, y - f.fy(5.0)), (unit_x - f.fx(19.0), y - f.fy(2.4)),
                color=INH, lw=LW_HAIR, head=3.4)
        mini_tree(f, unit_x, y, half_h)
        f.arrow((unit_x + f.fx(4.6), y), (out_x - f.fx(5.6), y), color=MUTE,
                lw=LW_HAIR, head=3.6)
        f.disc((out_x, y), 4.4, fill="white", edge=EDGE, lw=LW_EDGE, zorder=6)
        ax.text(out_x, y, f"y{SUBSCRIPT_DIGITS[index]}", ha="center",
                va="center", fontsize=PT_SMALL, color=INK, zorder=7)
        ax.plot([out_x + f.fx(4.6), out_bus], [y, y], color=MUTE,
                lw=LW_HAIR, zorder=2)
    ax.plot([out_bus, out_bus], [unit_ys[-1], unit_ys[0]], color=MUTE,
            lw=LW_HAIR, zorder=2)

    f.group((box_x0, 0.480, box_x1 - box_x0, 0.250), tint=COLORS["panel_bg"],
            edge=EDGE, lw=LW_EDGE, radius_pt=2.5, zorder=1.0)
    ax.text((box_x0 + box_x1) / 2.0, 0.605, "task\nreadout", ha="center",
            va="center", fontsize=PT_SMALL, color=INK, linespacing=1.2)
    f.arrow((out_bus, 0.605), (box_x0, 0.605), color=MUTE, lw=LW_HAIR,
            head=3.6)
    f.arrow((box_x1, 0.605), (delta_x - f.fx(3.4), 0.605), color=COLORS["bp"],
            lw=LW_EDGE, head=3.6)
    f.disc((delta_x, 0.605), 2.6, fill=COLORS["bp"], edge="white",
           lw=LW_HAIR, zorder=8)
    ax.text(delta_x, 0.605 + f.fy(6.5), "δₒᵤₜ", ha="center",
            va="bottom", fontsize=PT_ANNOT, color=COLORS["bp"])
    ax.text(unit_x, 0.075, "N dendritic E/I units", ha="center",
            va="center", fontsize=PT_SMALL, color=MUTE)

    # -- feedback return: the loss assigns one error coordinate per neuron,
    #    so a thin additive-blue bus leaves the readout, runs under the
    #    output column and rises on one spine; a short arrow then delivers
    #    δᵤ to each soma from below, echoing panel A's coordinate arrow.
    fb_x = 0.664
    fb_y = 0.150
    rd_cx = (box_x0 + box_x1) / 2.0
    ax.plot([rd_cx, rd_cx], [0.480, fb_y], color=BLUE, lw=LW_HAIR,
            solid_capstyle="round", zorder=2)
    ax.plot([rd_cx, fb_x], [fb_y, fb_y], color=BLUE, lw=LW_HAIR,
            solid_capstyle="round", zorder=2)
    ax.plot([fb_x, fb_x], [fb_y, unit_ys[0] - f.fy(9.4)], color=BLUE,
            lw=LW_HAIR, solid_capstyle="round", zorder=2)
    # Orthogonal delivery, not an arc.  A curved arrow takes its head angle
    # from the tangent where the arc happens to end, so these heads pointed
    # off in directions that named nothing; and running beside the forward
    # stroke, the pair read as one two-headed connector.  The return now
    # drops to its own line, runs back under the output column and turns UP
    # into the soma, so the head is vertical and aimed at what it delivers to.
    drop = f.fy(9.4)
    for y in unit_ys:
        ax.plot([fb_x, unit_x], [y - drop, y - drop], color=BLUE,
                lw=LW_HAIR, solid_capstyle="round", zorder=2.4)
        f.arrow((unit_x, y - drop), (unit_x, y - f.fy(3.4)), color=BLUE,
                lw=LW_HAIR, head=3.2, zorder=2.4)
    ax.text(fb_x - f.fx(4.0), 0.365, "δᵤ", ha="right", va="center",
            fontsize=PT_ANNOT, color=BLUE)


# ── C: coordinate -> address -> gain (the headline ladder) ────────────────
# Each rung carries its tree, its name, its symbol and the one question the
# stage answers.  The sentence that used to sit under every rung ("one error
# value per neuron...", "K coordinates per neuron...", "path conductance
# sets...") is prose, not graphic content, and has moved to the caption.
STAGES = (("coordinate", "coordinate", "which neuron"),
          ("address", "address", "which subtree"),
          ("gain", "gain", "how strongly"))

TREE_COL = 0.30                   # the rung's tree column, in frame fractions
TEXT_COL = 0.345

# The address rung's ONE selected subtree, and the route that reaches it.
# The K=4 library mode drew all four subtree capsules over one tree, and at
# rung size the overlapping tints read as noise while nothing was actually
# selected; "which subtree" needs exactly one answer on display.  The gain
# rung then re-uses panel D's idiom -- the transported route drawn edge by
# edge in additive blue -- so "which" (a lit subtree) and "how strongly" (the
# blue delivery with its gain) stop being the same picture with a ring on it.
_ADDR_SUBTREE_C = [("JL", "JLL")]
_ADDR_SUBTREE_D = [("JLL", "T1"), ("JLL", "T2")]
_ADDR_CAPSULE = [(CT._lerp("JL", "JLL", 0.3), "JLL", "T1"), ("JLL", "T2")]
# Panel D's route strokes, refitted: at a third of D's tree height its head
# geometry renders as blobs that swallow the edges they decorate.
_GAIN_ROUTE = ((CT.ROOT_PT, CT.P["J1"]),
               (CT.P["J1"], CT.P["JL"]),
               (CT.P["JL"], CT.P["JLL"]))


RUNG_XL = (-3.05, 2.55)          # room for the halo left of T1; no dead right


def _rung_tree(f, rect, rung):
    """One rung glyph, drawn natively in panel D's vocabulary."""
    if rung == "coordinate":
        sub, _ = tree_inset(f, rect, scale=1.0, arrow_scale=0.75,
                            xlim=RUNG_XL, ylim=TREE_YL_TIGHT,
                            mode="coordinate")
        sub.text(1.42, 0.72, "δᵤ", ha="left", va="center",
                 fontsize=PT_SMALL, color=BLUE)
        return
    aspect = (RUNG_XL[1] - RUNG_XL[0]) / (TREE_YL_TIGHT[1] - TREE_YL_TIGHT[0])
    x0, y0, w, h = rect
    w_pt, h_pt = w * f.w_pt, h * f.h_pt
    if w_pt / h_pt > aspect:
        fit_h, fit_w = h_pt, h_pt * aspect
    else:
        fit_w, fit_h = w_pt, w_pt / aspect
    box = (x0 + (w - f.fx(fit_w)) / 2.0, y0 + (h - f.fy(fit_h)) / 2.0,
           f.fx(fit_w), f.fy(fit_h))
    sub = f.ax.inset_axes(box, transform=f.ax.transData, zorder=3)
    sub.set_facecolor("none")
    CT._setup_axes(sub, RUNG_XL, TREE_YL_TIGHT)
    t = CT._Tree(sub, 1.0, False)
    t.capsule(mix("shunting", 18), 11, _ADDR_CAPSULE)
    t.tree(GHOST)
    t.edges(_ADDR_SUBTREE_C, DEND, CT._TAPER_PT["C"], zorder=2.2)
    t.edges(_ADDR_SUBTREE_D, DEND, CT._TAPER_PT["D"], zorder=2.2)
    t.junctions(edge=GHOST)
    t.soma(SOMA, RIM)
    if rung == "address":
        # One digit inside each quarter's fork -- the address is an index
        # into K discrete subtrees, and the selected index is the one the
        # halo answers.  Set above the canopy the digits crowded the divider
        # to the row above; a white backing lets each sit in its own fork
        # over the ghost strokes instead.
        for k, (tx, ty) in enumerate(((-1.85, 2.42), (-0.57, 2.78),
                                      (0.59, 2.74), (1.77, 2.32)), 1):
            selected = k == 1
            sub.text(tx, ty, str(k), ha="center", va="center",
                     fontsize=PT_SMALL, color=INK if selected else MUTE,
                     fontweight="bold" if selected else "normal",
                     bbox=None if selected else dict(
                         facecolor="white", edgecolor="none", pad=0.5),
                     zorder=5)
        sub.text(-1.62, 1.10, "cᵤ,ₖ₌₁", ha="center", va="center",
                 fontsize=PT_SMALL, color=INK)
        return
    for a, b in _GAIN_ROUTE:
        sub.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=LW_HAIR,
                 solid_capstyle="round", zorder=2.6)
        sub.add_patch(FancyArrowPatch(
            CT._lerp(a, b, 0.30), CT._lerp(a, b, 0.62),
            arrowstyle="-|>,head_length=1.7,head_width=1.1",
            mutation_scale=1.0, color=BLUE, lw=LW_HAIR, capstyle="round",
            zorder=4.5))
    sub.text(-0.52, 0.55, "α̃ₙ", ha="right", va="center",
             fontsize=PT_SMALL, color=BLUE)


def panel_c(ax):
    f = Frame(ax)
    band = 1.0 / 3.0
    for index, (rung, name, question) in enumerate(STAGES):
        top = 1.0 - index * band
        core = (0.0, top - band * 0.96, TREE_COL, band * 0.92)
        _rung_tree(f, core, rung)
        ax.text(TEXT_COL, top - band * 0.36, name, ha="left", va="center",
                fontsize=PT_LABEL, color=INK)
        ax.text(TEXT_COL, top - band * 0.68, question, ha="left",
                va="center", fontsize=PT_ANNOT, color=MUTE)
        if index:
            y = top - band * 0.005
            ax.plot([0.0, 1.0], [y, y], color=COLORS["grid"], lw=LW_HAIR,
                    zorder=1.0)
            ax.plot([TREE_COL / 2.0], [y], marker="v", ms=3.0, mfc=MUTE,
                    mec="none", ls="none", zorder=2)


# ── D: local eligibility x transported error ──────────────────────────────
SYN_F = 0.55                     # tagged synapse along the JRL -> T6 branch


def panel_d(ax):
    f = Frame(ax)
    eq_band = f.fy(28.0)
    band_h = 1.0 - eq_band

    # -- the tree: the eligibility decoration (ghost arbor, one lit branch,
    #    one tagged synapse) with the transported-error path overlaid, exactly
    #    as ``build_journal_figures._panel_general_adjoint`` composes it; the
    #    library's zoom bubble becomes the keyed column at the right, so the
    #    two factors of the theorem sit side by side.
    core = (0.0, eq_band, 0.545, band_h)
    w_pt, h_pt = core[2] * f.w_pt, core[3] * f.h_pt
    if w_pt / h_pt > TREE_ASPECT:
        fit_h, fit_w = h_pt, h_pt * TREE_ASPECT
    else:
        fit_w, fit_h = w_pt, w_pt / TREE_ASPECT
    # The tree is height-limited in this cell, so it is set against the cell's
    # left edge rather than centred in it: centring would leave a dead strip
    # on the left of the panel while the key column already holds the right.
    box = (core[0],
           core[1] + (core[3] - f.fy(fit_h)) / 2.0,
           f.fx(fit_w), f.fy(fit_h))
    sub = ax.inset_axes(box, transform=ax.transData, zorder=3)
    sub.set_facecolor("none")
    CT._setup_axes(sub, TREE_XL, TREE_YL)
    t = CT._Tree(sub, 1.0, False)
    t.tree(GHOST)
    t.seg("JR", "JRL", DEND, CT._TAPER_PT["C"], zorder=2.2)
    t.seg("JRL", "T6", DEND, CT._TAPER_PT["D"], zorder=2.2)
    t.junctions(edge=GHOST)
    t.junctions(names=("JRL",))
    t.soma(SOMA, RIM)
    syn = CT._lerp("JRL", "T6", SYN_F)
    t.dot(syn, 4.6, EXC, RIM, LW_HAIR)

    path = [(CT.ROOT_PT, CT.P["J1"], LW_DATA),
            (CT.P["J1"], CT.P["JR"], LW_ERR_ARROW),
            (CT.P["JR"], CT.P["JRL"], LW_EDGE)]
    for a, b, lw in path:
        sub.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=lw,
                 solid_capstyle="round", zorder=2.6)
        sub.add_patch(FancyArrowPatch(
            CT._lerp(a, b, 0.32), CT._lerp(a, b, 0.60),
            arrowstyle="-|>,head_length=3.2,head_width=2.0",
            mutation_scale=1.0, color=BLUE, lw=lw, capstyle="round",
            zorder=4.5))
    sub.plot([CT.P["JRL"][0]], [CT.P["JRL"][1]], marker="o", ms=8.0,
             mfc="none", mec=INK, mew=LW_EDGE, ls="none", zorder=4.4)
    sub.text(-0.30, -0.02, "δᵤ", ha="right", va="center",
             fontsize=PT_SMALL, color=BLUE)
    sub.text(0.24, 0.44, "α₁", ha="left", va="center",
             fontsize=PT_SMALL, color=BLUE)
    sub.text(0.74, 0.92, "α₂", ha="left", va="center",
             fontsize=PT_SMALL, color=BLUE)
    sub.text(0.22, 1.98, "α₃", ha="right", va="center",
             fontsize=PT_SMALL, color=BLUE)
    sub.text(0.16, 2.66, "qₙ", ha="right", va="center",
             fontsize=PT_SMALL, color=INK)

    # -- the local-factor key: three factors, direct-labelled beside their
    #    own glyphs (S6), with no box and no leader crossing the canopy.
    key_h = f.fy(63.0)
    key = (0.600, eq_band + (band_h - key_h) / 2.0, 0.400, key_h)
    kx = key[0] + f.fx(4.0)
    lx = key[0] + f.fx(11.0)
    head_y = key[1] + key[3] - f.fy(8.0)
    ax.text(key[0], head_y, "directed-tree eligibility  eᵢ", ha="left", va="center",
            fontsize=PT_SMALL, color=MUTE, zorder=6)
    # mute scaffolding rule instead of a box: it groups the three rows and
    # carries the column out to the cell edge without a second key convention
    ax.plot([key[0], 1.0], [head_y - f.fy(6.5)] * 2, color=MUTE, lw=LW_HAIR,
            solid_capstyle="butt", zorder=1.4)
    rows = [key[1] + key[3] - f.fy(v) for v in (23.5, 38.5, 53.5)]
    ax.plot([kx], [rows[0]], marker="o", ms=3.4, mfc=EXC, mec="none",
            ls="none", zorder=6)
    ax.text(lx, rows[0], "xᵢ", ha="left", va="center", fontsize=PT_SMALL,
            color=INK, zorder=6)
    ax.text(lx + f.fx(11.5), rows[0], "presyn.", ha="left", va="center",
            fontsize=PT_SMALL, color=MUTE, zorder=6)
    ax.add_patch(Arc((kx, rows[1]), 2 * f.fx(3.6), 2 * f.fy(3.6),
                     theta1=-20, theta2=200, color=MUTE, lw=LW_HAIR,
                     zorder=6))
    ang = np.deg2rad(52.0)
    ax.plot([kx, kx + f.fx(3.6) * np.cos(ang)],
            [rows[1], rows[1] + f.fy(3.6) * np.sin(ang)], color=INK,
            lw=LW_HAIR, solid_capstyle="round", zorder=6)
    ax.text(lx, rows[1], "Eᵢ − Vₙ", ha="left", va="center",
            fontsize=PT_SMALL, color=INK, zorder=6)
    zx, zy = kx - f.fx(4.6), rows[2]
    xs, ys = [zx], [zy]
    for dx, dy in ((1.4, 1.6), (2.2, -3.2), (2.2, 3.2), (2.2, -3.2),
                   (1.4, 1.6)):
        xs.append(xs[-1] + f.fx(dx))
        ys.append(ys[-1] + f.fy(dy))
    ax.plot(xs, ys, color=INK, lw=LW_HAIR, solid_joinstyle="round", zorder=6)
    ax.text(lx, rows[2], "Rₙᵗᵒᵗ", ha="left", va="center",
            fontsize=PT_SMALL, color=INK, zorder=6)

    ax.text(0.5, f.fy(23.0), "directed tree:  ∂ℒ / ∂gᵢ = [xᵢ Rₙᵗᵒᵗ (Eᵢ − Vₙ)] [∂ℒ / ∂Vₙ]",
            ha="center", va="center", fontsize=PT_ANNOT, color=INK)
    ax.text(0.5, f.fy(5.0), "general adjoint:  ∂ℒ / ∂gᵢ = xᵢ (Eᵢ − Vₙ) qₙ,   qₙ = Rₙᵗᵒᵗ ∂ℒ / ∂Vₙ",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)


# ── E: roadmap of the Results (full-width ribbon) ─────────────────────────
# Six cards, one per stage of the Results, in the order the subsections run:
# exact factorization; neuron selection; the credit-operator boundary;
# controlled branch-address tests; task-aligned physical depth; and the
# anatomical, conductance and measured-response boundary. Every glyph is drawn from the shared
# credit-tree vocabulary so the roadmap previews the figures that follow.


def _glyph_factorization(f, rect):
    """Transported error over the ghost arbor: the blue path product."""
    tree_inset(f, rect, mode="transport", scale=0.85, arrow_scale=0.7)


def _glyph_coordinate(f, rect):
    """A neuron-specific signal delivered to one soma."""
    sub, _ = tree_inset(f, rect, mode="plain", scale=0.85,
                        hide_arrows=True)
    sub.add_patch(FancyArrowPatch(
        (1.30, 0.60), (0.27, 0.09),
        arrowstyle="-|>,head_length=3.2,head_width=2.0", mutation_scale=1.0,
        connectionstyle="arc3,rad=0.12", color=BLUE, lw=LW_EDGE,
        capstyle="round", zorder=4.5))


def _glyph_coord_address(f, rect):
    """Subtree-address capsules after the neuron has been selected."""
    sub, _ = tree_inset(f, rect, mode="address", K=4, scale=0.85)
    sub.add_patch(FancyArrowPatch(
        (1.30, 0.60), (0.27, 0.09),
        arrowstyle="-|>,head_length=3.2,head_width=2.0", mutation_scale=1.0,
        connectionstyle="arc3,rad=0.12", color=BLUE, lw=LW_EDGE,
        capstyle="round", zorder=4.5))


def _glyph_operator(f, rect):
    """Stochastic credit ĝ through the route operator M: kept directions.

    The compact form of ``native_schematics.draw_credit_operator``: the
    operator box in the route green, the exiting dot row in the kept /
    admitted-noise / discarded tones (green / amber / gray).
    """
    x0, y0, w, h = rect
    cx = x0 + w / 2.0
    cy = y0 + h / 2.0
    f.text((cx, cy + f.fy(25.0)), "ĝ", size=PT_ANNOT, color=INK)
    f.arrow((cx, cy + f.fy(19.5)), (cx, cy + f.fy(10.5)), color=MUTE,
            lw=LW_EDGE, head=3.4)
    box = (cx - f.fx(11.0), cy - f.fy(5.0), f.fx(22.0), f.fy(14.0))
    f.group(box, tint=mix("shunting", 10), edge=mix("shunting", 45),
            lw=LW_EDGE, radius_pt=2.0, zorder=2)
    f.text((cx, cy + f.fy(2.0)), "M", size=PT_ANNOT, color=GREEN, zorder=6)
    f.arrow((cx, cy - f.fy(6.5)), (cx, cy - f.fy(15.5)), color=MUTE,
            lw=LW_EDGE, head=3.4)
    tones = (GREEN, GREEN, GREEN, COLORS["local"], MUTE)
    for i, tone in enumerate(tones):
        f.disc((cx + f.fx(5.2 * (i - 2)), cy - f.fy(21.0)), 1.9, fill=tone,
               zorder=5)


def _glyph_depth(f, rect):
    """Two-to-three physical stages of increasing depth (Fig. 5 geometry)."""
    x0, y0, w, h = rect
    base = y0 + h / 2.0 - f.fy(18.0)   # soma line of the tree-glyph cards
    height = f.fy(22.0)
    for dx, depth in zip((-22.0, 0.0, 22.0), (1, 2, 3)):
        f.stage_tree((x0 + w / 2.0 + f.fx(dx), base), height, depth)


def _glyph_anatomy_gain(f, rect):
    """Reconstructed arbor with contacts, plus the route-gain ring."""
    sub, _ = tree_inset(f, rect, mode="plain", scale=0.85, hide_arrows=True)
    sub.plot([CT.P["JL"][0]], [CT.P["JL"][1]], marker="o", ms=8.0,
             mfc="none", mec=INK, mew=LW_EDGE, ls="none", zorder=4.4)


def _glyph_boundary(f, rect):
    """The de-emphasised (shunted) arbor of the measured-response test."""
    tree_inset(f, rect, mode="shunt", shunted=True, scale=0.85)


STREAMS = (
    (_glyph_factorization, "Exact factorization",
     "eligibility × error"),
    (_glyph_coordinate, "Neuron selection",
     "one signal per neuron"),
    (_glyph_operator, "Credit operator",
     "when restricted routes help"),
    (_glyph_coord_address, "Branch addresses",
     "conflict + hierarchy"),
    (_glyph_depth, "Physical depth",
     "task-aligned stage count"),
    (_glyph_anatomy_gain, "Biological boundary",
     "capacity, gain, alignment"),
)


# Each stage is a card filling the band's full height: the glyph above one
# name line (wrapping to two on the narrow six-across cards) and a mute-free
# one-line gloss that follows the name down, so mixed name depths never
# overprint their glosses.
CARD_PAD_PT = 3.5
GLYPH_BOT_PT = 52.0
NAME_TOP_PT = 48.5
NAME_LINE_PT = 9.6
NAME_GLOSS_GAP_PT = 2.4


def panel_e(ax):
    f = Frame(ax)
    n = len(STREAMS)
    gap = f.fx(10.0)
    cell_w = (1.0 - (n - 1) * gap) / n
    pad = f.fx(CARD_PAD_PT)
    glyph_bot = f.fy(GLYPH_BOT_PT)
    text_w_pt = (cell_w - 2 * pad) * f.w_pt
    for index, (glyph, name, phrase) in enumerate(STREAMS):
        x0 = index * (cell_w + gap)
        cx = x0 + cell_w / 2.0
        f.group((x0, 0.0, cell_w, 1.0), tint=None, edge=COLORS["grid"],
                lw=LW_HAIR, radius_pt=3.0, zorder=0.5)
        glyph(f, (x0 + pad, glyph_bot, cell_w - 2 * pad,
                  1.0 - glyph_bot - f.fy(CARD_PAD_PT)))
        wrapped_name = _wrap_to_width(ax, name, PT_ANNOT, text_w_pt,
                                      max_lines=2) or name
        name_lines = wrapped_name.count("\n") + 1
        ax.text(cx, f.fy(NAME_TOP_PT), wrapped_name, ha="center", va="top",
                fontsize=PT_ANNOT, color=INK, linespacing=1.25)
        gloss_top = (NAME_TOP_PT - NAME_LINE_PT * name_lines
                     - NAME_GLOSS_GAP_PT)
        wrapped = _wrap_to_width(ax, phrase, PT_SMALL, text_w_pt,
                                 max_lines=4 - name_lines) or phrase
        ax.text(cx, f.fy(gloss_top), wrapped, ha="center", va="top",
                fontsize=PT_SMALL, color=MUTE, linespacing=1.35)
        if index + 1 < n:
            mid = x0 + cell_w + gap / 2.0
            y = (1.0 + glyph_bot) / 2.0
            f.arrow((mid - f.fx(4.5), y), (mid + f.fx(4.5), y), color=MUTE,
                    lw=LW_HAIR, head=4.6, zorder=2)


# ── build ─────────────────────────────────────────────────────────────────
def build():
    canvas = NativeCanvas(
        CANVAS_H_PT / 72.0, nrows=3, row_weights=ROW_PT,
        hgutter_pt=HGUTTER, vgutter_pt=VGUTTER, margins=MARGINS,
        letters=False,
    )
    spec = [
        ("A", 0, 0, 6, 1, "Point unit → dendritic tree", panel_a),
        ("B", 0, 6, 6, 1, "Network layer", panel_b),
        ("C", 1, 0, 6, 1, "Coordinate → address → gain", panel_c),
        ("D", 1, 6, 6, 1, "Local eligibility × transported error", panel_d),
        ("E", 2, 0, 12, 1, "Roadmap of the Results", panel_e),
    ]
    for letter, row, col, span, rowspan, title, draw in spec:
        ax = canvas.panel(letter, row, col, span, rowspan=rowspan,
                          schematic=True)
        # Centre panel titles as every other main sheet does; Figure 1 was
        # the only builder still left-aligning them.
        ax.set_title(title, fontsize=PT_TITLE, color=INK, pad=TITLE_PAD,
                     loc="center", fontweight="normal")
        canvas.add_letter(letter, ax, dx_pt=LETTER_DX, dy_pt=LETTER_DY)
        draw(ax)
    problems = canvas.save(OUT, name="main_figure_01_native")
    return problems


def main():
    problems = build()
    if problems:
        print(f"  {len(problems)} layout problems reported")
    return 0


if __name__ == "__main__":
    sys.exit(main())

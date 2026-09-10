#!/usr/bin/env python3
"""Main Figure 2 -- branch selection versus shared credit (``fig:branchconflict``).

Component ``figures/components/main_figure_04_native.pdf`` (the historical
component name is kept so ``rebuild_final_publication_figures.py`` and the
provenance map stay valid), copied to ``figures/main/figure_02.pdf``.
Eight panels on one :class:`figure_canvas.NativeCanvas`, in reading order:

  row 0  A context selects one of B branches   B three ways to deliver δ0
  row 1  C predicted and measured              D trained order as predicted
         E only shared credit forgets
  row 2  F/G/H held-out accuracy vs dose, one facet per branch count

Built to ``analysis/figure_overhaul_20260908/v2/fig2/PLAN.md`` as amended by
``v2/AMENDMENTS.md`` and ruled by ``v2/DECISIONS.md``.

Cross-figure rules carried by this builder (AMENDMENTS §3)
----------------------------------------------------------
CF-1 canvas 518.4 x 490.0 pt, on the 340/415/490 ladder, aspect 1.058.
CF-2 exactly three type sizes 7.0 / 8.0 / 9.0-bold; nothing below 7.0; no
     DejaVu; subscripts via ``Frame.subscript`` / ``token_subscript``.
CF-3 strokes only 0.55 / 0.70 / 0.85 / 0.95 / 1.25 pt; every area mark is a
     16 % ``tint_patch`` with a 0.55 pt edge.
CF-4 the glyph family of ``native_schematics``; four delivery modes only
     (``subtree``, ``neuron``, ``subtree`` here); no ``DELTA0_EXEMPTIONS``
     entry for this figure.
CF-5 zero legend boxes and zero figure-level legends.  Panel A's footer glyph
     key is an in-panel schematic key, not a legend box, and sits in no data
     axes; the set's one sanctioned in-axes key is Fig 5C.
CF-6 the forest idiom for E.
CF-7 every reference line dashed ``mute`` at ``LW_REF`` with its label on the
     line; zero drawn once.
CF-8 caption contract (main.tex, applied by the integrator).
CF-9 titles sentence case, no terminal period.
CF-10 schematic area on the B12 formula (below).
CF-11 no raster below 300 dpi (this figure places no raster at all).
CF-12 letters 9 pt bold via ``align_letters()``.

Schematic area (AMENDMENTS B12 / CF-10), the one common formula
---------------------------------------------------------------
``schematic_fraction = Σ(schematic panel slot w_pt × h_pt)
                       / (live_w_pt × live_h_pt)``
with ``live_w_pt = 518.4 − 51.0 − 13.0 = 454.4`` and
``live_h_pt = 490 − 16 − 26 = 448.0``::

    A slot 250.9 × 116.0 = 29,104.4 pt²
    B slot 169.5 × 116.0 = 19,662.0 pt²
    Σ schematic slots     = 48,766.4 pt²
    live canvas           = 454.4 × 448.0 = 203,571.2 pt²
    schematic_fraction    = 48,766.4 / 203,571.2 = 0.2396 → 24.0 %

24.0 % (≤ 30 %); Figure 2 claims no DECISIONS G4 schematic-area waiver.  The
plan's 24.6 % was computed with margins 22/32; the row-separation audit needs
a 46 pt vertical gutter (see below), which moves 12 pt of outer margin into
the two gutters and moves the fraction by 0.6 points.  The superseded value
26.6 % must not reappear.

Waiver (DESIGN_SPEC D3), the figure's only waiver
-------------------------------------------------
Row 1 places three 4-module data panels that do not share one axis: C (x = χ)
and D (y = χ, x = B) are locked on the same χ scale 0-1.1; E is the
SI-promoted forgetting forest and is the row's one recorded exception.
Rows 1 and 2 share one axes width via ``_equalize_row_widths``.

Private helpers (DECISIONS G5; report them for later promotion)
---------------------------------------------------------------
``fan_tree``        a B-branch parallel fan: the library's ``balanced_tree``
                    is binary and this task has B parallel branches.  Built
                    from ``Frame.dendrite / junction / contact / gate / fade /
                    soma / error_in`` at the library's own sizes, and
                    registered on the frame so ``require_soma_lowest()``
                    actually inspects it.  It splays the terminals wider than
                    the junctions, keeps an open junction ring under every
                    gate (panel A's glyph key names the junction, so one has
                    to be visible) and can lift the gate onto the distal
                    segment above its junction.
``_error_in_compact``  ``Frame.error_in`` re-seated on a shorter diagonal.
                    The library reaches 11 pt out and 7 pt down from the soma
                    rim and hangs the tag at the tail, i.e. about 13 pt of
                    clear space under every soma; panel B has under 10 pt per
                    card.  The library still draws and registers the arrow,
                    so ``require_delta0()`` is unaffected.
``_minus`` / ``_signed``  numbers with the typographic minus U+2212 the tick
                    labels already use (Python's format gives a hyphen).
``_corner_lines`` / ``_lines`` / ``_rug`` / ``_exponent``  point-anchored
                    annotation stacks, the sampled-dose rug, and a 7 pt
                    raised exponent (Nimbus Sans has no U+207B and CF-2 bans
                    mathtext).
``_recolour_last_capsule``  ``Frame.credit_delivery(mode='subtree')`` paints
                    its capsule from the frozen K-cycle, whose first entry is
                    ``shunting``; in this figure ``shunting`` is reserved for
                    the branch-specific arm (AMENDMENTS §5), so the deranged
                    row's capsule is repainted as a 16 % ``point_mlp`` tint
                    after the library call.
``_glyph_key``      panel A's footer key: the library has no mini-glyph strip
                    that is not a rule key (a legend), and CF-5 bans keys.
``_equalize_row_widths``  one axes width for every 4-module panel of a row.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from credit_tree_schematics import AMBER_TEXT, mix
from figure_canvas import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    ORDINAL_RAMP,
    PT_BASE,
    PT_EMPH,
    SEED_MS,
    Margins,
    NativeCanvas,
    tint_patch,
    token_subscript,
)
from journal_style import label_color, style_direct_color_labels
from native_schematics import (
    CONTACT_DIA_PT,
    FADED_RING,
    Frame,
    Nodes,
    _text_w_pt,
)
from native_schematics import reference_line
from routing_figure_panels import PATH_NECESSITY


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "figures" / "components" / "main_figure_04_native.pdf"
PUBLISHED = ROOT / "figures" / "main" / "figure_02.pdf"
SUBTREE = ROOT / "source_data" / "trained_subtree_address"
CURATED = ROOT / "source_data" / "curated_publication" / "figure_02_plotted.csv"

# ── canvas (CF-1) ─────────────────────────────────────────────────────────
# 16 + 116 + 46 + 116 + 46 + 124 + 26 = 490.  The plan's 22/40/32 split gives
# a measured row-1|row-2 ink separation of 6.7 pt, under the 8.5 pt (3 mm)
# floor of ``audit_row_separation.py``; 12 pt of outer margin moved into the
# two vertical gutters buys the clearance without touching the row heights,
# the canvas height or the module grid.
CANVAS_H_PT = 490.0
HEIGHT_IN = CANVAS_H_PT / 72.0
ROW_PT = [116.0, 116.0, 124.0]
HGUTTER_PT = 34.0
VGUTTER_PT = 46.0
MARGINS = Margins(left=51.0, right=13.0, top=16.0, bottom=26.0)
# One in-slot left reserve for every data column: the forest's row-label
# gutter in E sets it, and the row-alignment contract makes every 4-module
# panel of a row share it.  It is also what keeps column 0's y label out of
# the 9 pt panel-letter column (audit_letter_alignment.py).
DATA_LEFT_PT = 33.0

INK = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]
GREEN = COLORS["shunting"]          # branch-specific (AMENDMENTS §5)
AMBER = COLORS["scalar"]            # neuron-shared / broadcast
GRAY = COLORS["point_mlp"]          # the neutral control series
GREEN_TEXT = label_color(GREEN)
GRAY_TEXT = label_color(GRAY)

BRANCHES = (2, 4, 8)
# ORDINAL_RAMP entries 0, 2, 3: an ordinal position within THIS figure's own
# list (branch count), never a fixed task identity (AMENDMENTS B14.2).
RAMP = {2: ORDINAL_RAMP[0], 4: ORDINAL_RAMP[2], 8: ORDINAL_RAMP[3]}
RAMP_TEXT = {b: label_color(to_hex(c)) for b, c in RAMP.items()}
B_MARKER = {2: "o", 4: "s", 8: "^"}
CHI_C = {b: b / (2.0 * (b - 1)) for b in BRANCHES}
DOSES = (0.0, 0.25, 0.5, 4.0 / 7.0, 0.6, 2.0 / 3.0, 0.75, 1.0)
DASHES = (2.2, 1.8)

SERIES = (
    ("correct_path", GREEN, "o", "branch-specific"),
    ("neuron_shared_k1", AMBER, "s", "neuron-shared"),
    ("within_neuron_deranged", GRAY, "^", "deranged route"),
)


# ── private helpers ───────────────────────────────────────────────────────
MINUS = "\u2212"


def _minus(value, digits=1):
    """A number with the typographic minus the tick labels already use."""
    return f"{value:.{digits}f}".replace("-", MINUS)


def _signed(value, digits=1):
    """A signed number with a real plus and a typographic minus."""
    return f"{value:+.{digits}f}".replace("-", MINUS)


def _rug(ax, doses, y0=None, length_pt=2.6):
    """Sampled-dose rug on the bottom spine, at LW_HAIR."""
    lo, hi = ax.get_ylim()
    fig = ax.get_figure()
    h_pt = ax.get_position().height * fig.get_size_inches()[1] * 72.0
    dy = length_pt * (hi - lo) / max(h_pt, 1e-6)
    y0 = lo if y0 is None else y0
    for d in doses:
        ax.plot([d, d], [y0, y0 + dy], color=EDGE, lw=LW_HAIR,
                solid_capstyle="butt", zorder=2, clip_on=False)


def _lines(ax, xy, rows, *, ha="left", va="top", step_pt=8.8, color=None,
           size=PT_BASE, colors=None):
    """A short stack of 7 pt annotation lines anchored in data coordinates."""
    fig = ax.get_figure()
    h_pt = ax.get_position().height * fig.get_size_inches()[1] * 72.0
    lo, hi = ax.get_ylim()
    step = step_pt * abs(hi - lo) / max(h_pt, 1e-6)
    if ax.yaxis_inverted():
        step = -step
    x, y = xy
    out = []
    for i, text in enumerate(rows):
        col = (colors[i] if colors else None) or color or MUTE
        out.append(ax.text(x, y - i * step, text, fontsize=size, color=col,
                           ha=ha, va=va, zorder=6))
    return out


def fan_tree(frame: Frame, *, origin, xs_pt, y_soma_pt, y_junc_pt,
             y_tip_pt, tip_xs_pt=None, contact_pt=None, ghost=False,
             selected=0, output=None, delta="δ0", delta_side="right",
             delta_compact=None, soma_r_pt=3.0, gate_badge="c", labels=None,
             label_lift_pt=3.0, gate_unselected=True, gate_lift_pt=0.0,
             gate_badge_offset=(-4.8, -3.4)) -> Nodes:
    """PRIVATE (G5): the B-branch parallel fan, soma lowest.

    ``origin`` is the (x, y) of the drawing core in points inside the frame;
    every other coordinate is in points relative to it.  One proximal segment
    soma -> junction (taper level 1) and one distal segment junction -> tip
    (level 3) per branch, an open white junction ring at every junction, an
    ``exc`` contact on each distal segment, the open context gate on
    ``selected`` and closed gates (with the branch attenuated) on the rest.
    ``tip_xs_pt`` splays the tips wider than the junctions so the fan reads
    as a fan; ``gate_lift_pt`` puts the gate that far up the distal segment
    so the junction ring it would otherwise cover stays visible (panel A's
    glyph key names the junction, so one must be on the artwork).  Returns a
    :class:`Nodes` and registers it on the frame so
    :meth:`Frame.require_soma_lowest` inspects a real tree.
    """
    ox, oy = origin
    tip_xs_pt = list(xs_pt) if tip_xs_pt is None else list(tip_xs_pt)

    def P(x_pt, y_pt):
        return (ox + frame.fx(x_pt), oy + frame.fy(y_pt))

    B = len(xs_pt)
    nodes = Nodes()
    soma = (ox + frame.fx(0.5 * (xs_pt[0] + xs_pt[-1])), oy + frame.fy(y_soma_pt))
    nodes["S"] = soma
    nodes.children["S"] = []
    for i, x in enumerate(xs_pt):
        j, t = f"J{i + 1}", f"T{i + 1}"
        nodes[j] = P(x, y_junc_pt)
        nodes[t] = P(tip_xs_pt[i], y_tip_pt)
        nodes.parent[j] = "S"
        nodes.parent[t] = j
        nodes.children["S"].append(j)
        nodes.children[j] = [t]
        nodes.children[t] = []
        nodes.level[j] = 1
        nodes.level[t] = 3
    nodes.terminals = [f"T{i + 1}" for i in range(B)]
    nodes.soma = soma
    nodes.soma_r_pt = soma_r_pt
    nodes.orient = "up"
    nodes.pitch_pt = float(xs_pt[1] - xs_pt[0]) if B > 1 else 20.0
    for name, par in nodes.parent.items():
        p0 = nodes[par]
        nodes.edges[(par, name)] = frame.dendrite(
            p0, nodes[name], level=nodes.level[name], ghost=ghost)
    for i in range(B):
        nodes.rings[f"J{i + 1}"] = frame.junction(nodes[f"J{i + 1}"],
                                                  ghost=ghost)
    span = max(y_tip_pt - y_junc_pt, 1e-6)
    for i in range(B):
        j, t = f"J{i + 1}", f"T{i + 1}"
        f_up = min(max(gate_lift_pt / span, 0.0), 0.9)
        gxy = (nodes[j][0] + (nodes[t][0] - nodes[j][0]) * f_up,
               nodes[j][1] + (nodes[t][1] - nodes[j][1]) * f_up)
        if i == selected:
            frame.gate(gxy, closed=False, badge=gate_badge,
                       badge_offset=gate_badge_offset)
        elif gate_unselected:
            frame.gate(gxy, closed=True, badge=None)
            frame.fade([nodes.edges[("S", j)]])
            nodes.rings[j].set_edgecolor(FADED_RING)
        contact_at = y_tip_pt - 6.0 if contact_pt is None else contact_pt
        f_c = min(max((contact_at - y_junc_pt) / span, 0.0), 1.0)
        frame.contact((nodes[j][0] + (nodes[t][0] - nodes[j][0]) * f_c,
                       nodes[j][1] + (nodes[t][1] - nodes[j][1]) * f_c),
                      kind="exc")
    if labels:
        for i, (base, sub) in enumerate(labels):
            frame.subscript(P(tip_xs_pt[i], y_tip_pt + label_lift_pt),
                            base, sub, size=PT_BASE, color=INK, ha="center",
                            va="bottom")
    frame.soma(soma, r_pt=soma_r_pt, output=output, label=output and "z")
    if delta:
        if delta_compact is None:
            frame.error_in(soma, label=delta, side=delta_side, r_pt=soma_r_pt)
        else:
            _error_in_compact(frame, soma, label=delta, r_pt=soma_r_pt,
                              reach_pt=delta_compact[0],
                              drop_pt=delta_compact[1], side=delta_side)
    frame._trees.append(nodes)
    return nodes


def _error_in_compact(frame: Frame, soma_xy, *, label="δ0", r_pt=2.4,
                      reach_pt=9.5, drop_pt=5.5, side="right"):
    """PRIVATE (G5): ``Frame.error_in`` on a short leash.

    The library's somatic-error arrow reaches 11 pt out and 7 pt down from
    the soma rim and hangs its 7 pt tag at the tail, so it needs about 13 pt
    of clear space under every soma.  Panel B stacks three cards in 85 pt,
    which leaves under 10 pt.  The arrow is drawn by the library (so
    ``require_delta0`` still sees a registered arrival, and the ink, weight
    and 4.5 pt head are the library's) and then re-seated on a shorter
    diagonal, tag and all.
    """
    n_before = len(frame.ax.texts)
    arr = frame.error_in(soma_xy, label=label, side=side, r_pt=r_pt)
    x, y = soma_xy
    r = r_pt * frame.scale
    sgn = -1.0 if side == "left" else 1.0
    tail0 = (x + sgn * frame.fx(r + 11.0), y - frame.fy(r + 7.0))
    tip = (x + sgn * frame.fx(0.72 * r + 0.9), y - frame.fy(0.72 * r + 0.9))
    tail = (x + sgn * frame.fx(reach_pt), y - frame.fy(drop_pt))
    arr.set_positions(tail, tip)
    # the tag is carried by the same offset the library gave it, so a
    # right-anchored ('left' side) tag stays right-anchored on the new tail
    for text in frame.ax.texts[n_before:n_before + 1]:
        tx, ty = text.get_position()
        text.set_position((tail[0] + (tx - tail0[0]), tail[1]))
    return arr


def _recolour_last_capsule(frame: Frame, before: int, cname: str,
                           pct: int = 16) -> None:
    """PRIVATE (G5): repaint the capsule ``credit_delivery`` just drew.

    ``Frame.credit_delivery(mode='subtree')`` takes its capsule tint from the
    frozen K-cycle, whose first entry is ``shunting``.  ``shunting`` is the
    branch-specific arm in this figure and may carry no second role
    (AMENDMENTS §5), so the deranged row's capsule is repainted here rather
    than by editing the library.
    """
    from journal_style import strengthen, tint_pct
    face = tint_pct(COLORS[cname], pct)
    for patch in frame.ax.patches[before:]:
        patch.set_facecolor(face)
        patch.set_edgecolor(strengthen(face, 2.4))


def _glyph_key(frame: Frame, *, y_pt, x0_pt, width_pt):
    """PRIVATE (G5): panel A's footer glyph key.

    Five mini glyphs at their true sizes with a 7 pt tag each, laid out left
    to right across ``width_pt``.  This is an in-panel schematic key, not a
    legend box (CF-5): it carries no data series, sits in no data axes and has
    no frame.  Returns the number of rows it used.
    """
    ax = frame.ax
    items = [("soma", "soma"), ("junction", "junction"),
             ("contact", "excitatory contact"), ("gate", "context gate c"),
             ("error", "somatic error δ0")]
    # QA 2026-09-09: the key's own token is set with Frame.subscript so it
    # reads δ₀ exactly like the tag on the tree beside it (CF-2).
    subs = {"somatic error δ0": ("somatic error δ", "0")}
    glyph_pt = {"soma": 8.0, "junction": 7.0, "contact": 7.0, "gate": 10.0,
                "error": 13.0}
    widths = [glyph_pt[k] + 2.6 + _text_w_pt(ax, s, PT_BASE) for k, s in items]
    gap = (width_pt - sum(widths)) / (len(items) - 1)
    rows = 1
    if gap < 3.0:                       # two rows rather than sub-7 pt type
        rows = 2
        split = 3
        groups = [items[:split], items[split:]]
        gws = [widths[:split], widths[split:]]
        ys = [y_pt + 4.3, y_pt - 4.3]
    else:
        groups, gws, ys = [items], [widths], [y_pt]
    for group, gw, yy in zip(groups, gws, ys):
        g = ((width_pt - sum(gw)) / (len(group) - 1)) if len(group) > 1 else 0.0
        g = min(g, 26.0)
        x = x0_pt
        for (kind, text), w in zip(group, gw):
            cx = x + glyph_pt[kind] * 0.5
            p = (frame.fx(cx), frame.fy(yy))
            if kind == "soma":
                frame.disc(p, 3.0, fill=COLORS["soma"],
                           edge=mix("ink", 62), lw=LW_EDGE, zorder=4)
            elif kind == "junction":
                frame.junction(p)
            elif kind == "contact":
                frame.contact(p, kind="exc")
            elif kind == "gate":
                frame.disc(p, 1.9, fill="white", edge=COLORS["gate"],
                           lw=LW_EDGE, zorder=4.2)
                frame.disc(p, 3.6, fill="none", edge=COLORS["gate"],
                           lw=LW_EDGE, zorder=4.2)
            else:
                frame.arrow((frame.fx(x), frame.fy(yy - 2.0)),
                            (frame.fx(x + 11.0), frame.fy(yy + 2.0)),
                            color=INK, lw=LW_EDGE, head=4.5)
            anchor = (frame.fx(x + glyph_pt[kind] + 2.6), frame.fy(yy))
            if text in subs:
                base, sub = subs[text]
                frame.subscript(anchor, base, sub, size=PT_BASE,
                                color=MUTE, ha="left", va="center")
            else:
                frame.text(anchor, text, size=PT_BASE, color=MUTE,
                           ha="left", va="center")
            x += w + g
    return rows


# ── A: the task ───────────────────────────────────────────────────────────
A_TITLE = "Context selects one of B branches"
A_FOOT_PT = 22.0


def branch_conflict_task(ax) -> Frame:
    """One tree with a context gate, and the two conflict states as tiles."""
    frame = Frame(ax)
    core = frame.cell_text((0.0, 0.0, 1.0, 1.0), title=A_TITLE)
    core = (core[0], core[1] + frame.fy(A_FOOT_PT), core[2],
            core[3] - frame.fy(A_FOOT_PT))
    ox, oy = core[0], core[1]
    core_w = core[2] * frame.w_pt

    def P(x_pt, y_pt):
        return (ox + frame.fx(x_pt), oy + frame.fy(y_pt))

    # -- the tree -----------------------------------------------------------
    xs = [46.0, 68.0, 90.0, 112.0]
    tips = [41.0, 66.0, 92.0, 117.0]
    nodes = fan_tree(frame, origin=(ox, oy), xs_pt=xs, tip_xs_pt=tips,
                     y_soma_pt=13.0, y_junc_pt=36.0, y_tip_pt=66.0,
                     contact_pt=56.0, gate_lift_pt=7.0,
                     gate_badge_offset=(5.2, 0.6),
                     selected=0, output=11.0,
                     delta_side="left", delta_compact=(12.0, 7.0),
                     labels=[("x", "1"), ("x", "2"), ("x", "3"), ("x", "4")])
    # context input: its own carmine arrow and tag, in the free lane left of
    # the selected (leftmost) branch -- the review's required separate glyph.
    frame.arrow(P(34.0, 52.0), P(41.6, 45.4), color=COLORS["inh"],
                lw=LW_EDGE, head=4.5, zorder=5)
    frame.text(P(1.0, 56.5), "context c", size=PT_BASE,
               color=label_color(COLORS["inh"]), ha="left", va="center")

    # -- the conflict-state tiles -------------------------------------------
    tx0, tw, tgap = 130.0, 27.0, 4.0
    rows = ((76.0, 53.0, "χ = 0  compatible", False),
            (49.0, 26.0, "χ = 1  conflicting", True))
    for y_label, y_tile, label, conflict in rows:
        frame.text(P(tx0, y_label), label, size=PT_BASE, color=INK,
                   ha="left", va="top")
        for i in range(4):
            x = tx0 + i * (tw + tgap)
            selected = (i == 0)
            if conflict and not selected:
                tint_patch(ax, ("rect", ox + frame.fx(x),
                                oy + frame.fy(y_tile),
                                frame.fx(tw), frame.fy(13.0)),
                           color="mute", pct=16, edge=True, lw=LW_HAIR,
                           radius_pt=1.5, zorder=1.0)
            else:
                tint_patch(ax, ("rect", ox + frame.fx(x),
                                oy + frame.fy(y_tile),
                                frame.fx(tw), frame.fy(13.0)),
                           color="mute", pct=0, face="white",
                           edge_color=INK if selected else COLORS["grid"],
                           edge=True, lw=LW_EDGE if selected else LW_HAIR,
                           radius_pt=1.5, zorder=1.0)
            sub = "1−y" if (conflict and not selected) else "y"
            frame.subscript(P(x + tw * 0.5, y_tile + 6.5), "x", sub,
                            size=PT_BASE, color=INK, ha="center", va="center")

    # -- footer: the intermediate-dose rule, then the glyph key -------------
    # QA 2026-09-10: the somatic-error tag now hangs on the soma's LEFT (the
    # forward 'z' tag owns the right at soma height) and on a 7 pt leash, so
    # it clears the footer band; the three footer baselines are then set on a
    # uniform 8.6 pt pitch instead of the 7.0 pt slab the previous build had.
    frame.text((0.5, frame.fy(A_FOOT_PT - 6.0)),
               "0 < χ < 1: each nonselected view is replaced independently",
               size=PT_BASE, color=MUTE, ha="center", va="center")
    _glyph_key(frame, y_pt=3.1, x0_pt=1.0, width_pt=frame.w_pt - 2.0)
    return frame


# ── B: three deliveries of the same error ─────────────────────────────────
# The three cards are stacked in 85 pt of core, which is 28.3 pt each on an
# even split.  A card has to hold, bottom to top, the somatic-error tag, the
# soma, the junction row, the terminal row and -- for the neuron-shared card
# only -- the amber bus the library seats 8 pt above the terminals.  An even
# split leaves the tree about 7 pt tall, which is not a tree; the split below
# gives the bus card its 6 pt of extra head-room and keeps the FAN GEOMETRY
# IDENTICAL in all three, which is what the comparison-card rule asks for.
B_TITLE = "Three ways to deliver the same error"
B_FOOT_PT = 12.0
B_CARDS = ((56.0, 29.0), (25.0, 31.0), (0.0, 25.0))     # top card first
# QA 2026-09-10: the fan sits 4 pt further right and splays 5-6 pt wider than
# it did, and the contacts step back off the tips (19.4 of the 15.5 -> 20.0
# distal run).  Both moves are for the neuron-shared card: the amber bus drops
# vertically onto the JUNCTIONS, so the drop lines pass the contacts, and at
# the old +/-2 pt splay the 2.6 pt heads landed on the blue discs and the
# glyph read as one drop per excitatory contact.  The wider splay buys every
# drop >= 4.3 pt of lateral clearance; the extra 4 pt of left margin is where
# the bus riser now stands (``B_RISER_X``), clear of the 'c' badge and of the
# carmine gate ring the library's own s_lo ran through.
B_JUNC_X = (16.0, 26.0, 36.0, 46.0)
B_TIP_X = (10.0, 21.0, 41.0, 52.0)
B_CONTACT_PT = 19.4
B_RISER_X = 2.0
B_RIM_PT = 1.9          # drop head on the junction ring rim (r 1.6 pt)
B_TEXT_X = 58.0


def _seat_neuron_bus(frame: Frame, nodes, *, n_lines, n_patches,
                     riser_x_pt=B_RISER_X, rim_pt=B_RIM_PT):
    """PRIVATE (G5): re-seat the artists ``credit_delivery('neuron')`` drew.

    Two things the library fixes by construction do not survive a four-branch
    fan this narrow, and both are placement, not vocabulary, so they are
    repaired here rather than by editing the library (the same treatment
    ``_error_in_compact`` and ``_recolour_last_capsule`` get):

    * each drop stops 2.8 pt short of its target so a *terminal* dot is not
      overprinted; the targets here are junction RINGS (r 1.6 pt), so the
      head is brought in to the rim and the drop visibly lands on the ring;
    * the riser stands 2 pt outside the leftmost target, which on this tree
      is the selected branch -- i.e. straight through its open gate ring and
      its 'c' badge.  It is moved out to ``riser_x_pt``, and the bus's low
      end travels with it so the two still meet.

    Nothing else moves: rail height, drop count, fan and source dot are the
    library's, so the three ghost fans stay identical.
    """
    ax = frame.ax
    bus, riser = ax.lines[n_lines], ax.lines[n_lines + 1]
    drops = ax.patches[n_patches:n_patches + len(nodes.terminals)]
    x_riser = frame.fx(riser_x_pt)
    bx, by = [list(v) for v in (bus.get_xdata(), bus.get_ydata())]
    bus.set_data([x_riser, max(bx)], [by[0], by[0]])
    rx, ry = [list(v) for v in (riser.get_xdata(), riser.get_ydata())]
    riser.set_data([rx[0], x_riser, x_riser], ry)
    for i, drop in enumerate(drops):
        jx, jy = nodes[f"J{i + 1}"]
        drop.set_positions((jx, by[0]), (jx, jy + frame.fy(rim_pt)))
    return drops


def backward_credit_schematic(ax) -> Frame:
    """The same δ0 delivered three ways on three identical ghost fans."""
    frame = Frame(ax)
    core = frame.cell_text((0.0, 0.0, 1.0, 1.0), title=B_TITLE)
    core = (core[0], core[1] + frame.fy(B_FOOT_PT), core[2],
            core[3] - frame.fy(B_FOOT_PT))
    ox0, oy0 = core[0], core[1]

    specs = (
        ("branch-specific", GREEN_TEXT, "exact", "subtree", "J1",
         ("δ", "b", " = δ 1[b = b*]"), "= BP = gated point"),
        ("neuron-shared", AMBER_TEXT, "local rule", "neuron", None,
         ("δ", "b", " = δ / B"), None),
        ("deranged route", GRAY_TEXT, "control", "subtree", "J2",
         ("δ", "b", " = δ 1[b = b* + 1]"), "b → b + 1"),
    )
    for (y0, h), (name, text_col, badge, mode, target, formula, tail) \
            in zip(B_CARDS, specs):
        oy = oy0 + frame.fy(y0)
        nodes = fan_tree(frame, origin=(ox0, oy), xs_pt=B_JUNC_X,
                         tip_xs_pt=B_TIP_X, y_soma_pt=9.5, y_junc_pt=15.5,
                         y_tip_pt=20.0, contact_pt=B_CONTACT_PT, ghost=True,
                         selected=0, soma_r_pt=2.4, delta_compact=(10.5, 6.0),
                         gate_badge_offset=((6.6, -3.2) if mode == "neuron"
                                            else (-6.8, -3.2)),
                         gate_unselected=False)
        before = len(ax.patches)
        n_lines = len(ax.lines)
        if mode == "neuron":
            # QA 2026-09-09: the bus is seated on the JUNCTIONS (the plan's
            # "one drop per junction"), which drops its rail 5.5 pt into
            # row 2's own cell instead of leaving it in the inter-row gap
            # where it read as the hairline §3 deletes, and moves the riser
            # 5 pt right of the row's 'c' badge.
            frame.credit_delivery(nodes, mode="neuron", rule_color=AMBER,
                                  targets=[f"J{i + 1}" for i in range(4)])
            _seat_neuron_bus(frame, nodes, n_lines=n_lines, n_patches=before)
        else:
            frame.credit_delivery(nodes, mode="subtree", targets=[target],
                                  rule_color=GREEN if target == "J1" else GRAY)
            if target != "J1":
                _recolour_last_capsule(frame, before, "point_mlp")
                j1, j2 = nodes["J1"], nodes["J2"]
                # QA 2026-09-10: the arc bows UP, over the contacts, where
                # the old downward bow passed through the b3 contact disc
                # QA 2026-09-10: a flatter bow from ring rim to ring rim
                # passes above the b1 contact and lands on the b2 ring
                frame.arrow((j1[0], j1[1] + frame.fy(3.0)),
                            (j2[0], j2[1] + frame.fy(3.0)),
                            color=GRAY, lw=LW_HAIR, head=3.2, rad=0.28,
                            zorder=4.6)
        tx = ox0 + frame.fx(B_TEXT_X)
        ytop = oy + frame.fy(24.0)
        frame.text((tx, ytop), name, size=PT_BASE, color=text_col,
                   ha="left", va="top")
        frame.subscript((tx, ytop - frame.fy(9.6)), formula[0], formula[1],
                        formula[2], size=PT_BASE, color=INK, ha="left",
                        va="top")
        if tail:
            frame.text((tx, ytop - frame.fy(19.2)), tail, size=PT_BASE,
                       color=MUTE, ha="left", va="top")
        # every badge uses the neutral 'control' face: BADGE_STYLE's 'exact'
        # is drawn in ``bp`` and its 'local rule' in ``shunting``, and both
        # hues are barred / reserved in this figure (AMENDMENTS §5)
        frame.badge((core[0] + core[2], oy + frame.fy(25.0)), "control",
                    text=badge, ha="right", va="top")
    # Γ footer token (AMENDMENTS B10): a printed symbol, not a new glyph.
    frame.text((frame.fx(1.0), frame.fy(5.0)),
               "Γ ∈ {0,1}:  1[b = b*]   1/B   permuted",
               size=PT_BASE, color=MUTE, ha="left", va="center")
    return frame


# ── C: predicted and measured ─────────────────────────────────────────────
# The panel is 97 pt wide (the plan's own accepted forest-gutter cost), so the
# statements the plan puts on this panel are set in the two wedges the fan of
# analytic lines leaves empty: above the B = 2 line at the top right, and
# below the B = 8 line at the bottom left.  Every annotation is anchored in
# POINTS off an axes corner, so it keeps its clearance after the reserve lock
# moves the axes box.  xlim reaches 1.30 so the three direct labels sit past
# the data at their own line ends instead of on top of each other at chi = 0.
C_XLIM = (-0.03, 1.30)
C_YLIM = (-1.35, 1.05)
C_LABEL_X = 1.06


def _corner_lines(ax, texts, *, corner=(0.0, 0.0), x_pt=1.0, y_pt=0.0,
                  step_pt=9.0, ha="left", va="top", color=None,
                  colors=None, size=PT_BASE):
    """A stack of 7 pt lines anchored in points off an axes corner."""
    out = []
    for i, text in enumerate(texts):
        col = (colors[i] if colors else None) or color or MUTE
        out.append(ax.annotate(
            text, xy=corner, xycoords="axes fraction",
            xytext=(x_pt, y_pt - i * step_pt), textcoords="offset points",
            fontsize=size, color=col, ha=ha, va=va, zorder=6,
            annotation_clip=False))
    return out


def initial_utility(ax, summary: pd.DataFrame) -> dict:
    """Analytic s(χ) against the neuron-shared rule's measured utility."""
    shared = summary[summary.condition.eq("neuron_shared_k1")]
    chi = np.linspace(0.0, 1.0, 301)
    excluded, max_dev = 0, 0.0
    ax.set_xlim(*C_XLIM)
    ax.set_ylim(*C_YLIM)
    for b in BRANCHES:
        colour = RAMP[b]
        ax.plot(chi, 1.0 - 2.0 * (b - 1) * chi / b, color=colour, lw=LW_DATA,
                zorder=2, solid_capstyle="round")
        part = shared[shared.branches.eq(b)].sort_values("conflict_probability")
        x = part.conflict_probability.to_numpy(float)
        m = part.mean_initial_signed_utility.to_numpy(float)
        lo = part.ci95_low_initial_signed_utility.to_numpy(float)
        hi = part.ci95_high_initial_signed_utility.to_numpy(float)
        pred = 1.0 - 2.0 * (b - 1) * x / b
        max_dev = max(max_dev, float(np.max(np.abs(m - pred))))
        excluded += int(np.sum((hi < pred - 1e-12) | (lo > pred + 1e-12)))
        ax.errorbar(x, m, yerr=[m - lo, hi - m], linestyle="none",
                    marker=B_MARKER[b], ms=MARKER_MS, markerfacecolor="white",
                    markeredgecolor=colour, markeredgewidth=LW_EDGE,
                    ecolor=colour, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                    capthick=LW_ERR, zorder=4)
        ax.text(C_LABEL_X, 1.0 - 2.0 * (b - 1) / b, f"B = {b}",
                fontsize=PT_BASE, color=RAMP_TEXT[b], ha="left", va="center",
                zorder=6)
    reference_line(ax, 0.0, axis="y", label=None, color=MUTE, zorder=0,
                   span=(C_XLIM[0], 1.0))
    # CF-7 exception (recorded in the deviation list): the right end of the
    # zero rule carries the three chi_c diamonds and the 'B = 2' direct
    # label, so the reference label sits right-aligned at chi = 0.52 below
    # the rule instead of at its right end.
    ax.annotate("zero utility", xy=(0.52, 0.0), xytext=(0.0, -1.8),
                textcoords="offset points", fontsize=PT_BASE, color=MUTE,
                ha="right", va="top", zorder=6, annotation_clip=False)
    for b in BRANCHES:
        ax.plot([CHI_C[b]], [0.0], linestyle="none", marker="D",
                ms=MARKER_MS, markerfacecolor="white", markeredgecolor=INK,
                markeredgewidth=LW_EDGE, zorder=5)
    _corner_lines(ax, ("s(χ) = 1 − 2χ(B − 1)/B",), corner=(1.0, 1.0),
                  x_pt=-1.0, y_pt=-1.0, ha="right", color=INK)
    token_subscript(ax, 0.50, 0.86, "χ", "c", " = B/[2(B − 1)]",
                    size=PT_BASE, sub_size=PT_BASE, color=MUTE, ha="left",
                    va="top")
    # four lines in the wedge the B = 8 line leaves empty; the top line is
    # the short one because that is where the wedge is narrowest
    _corner_lines(ax, (
        f"max deviation {max_dev:.3f}",
        f"{excluded} of {len(shared)} intervals exclude",
        "the line. n = 20 seeds; mean",
        "and 95 % bootstrap, epoch 0",
    ), corner=(0.0, 0.0), x_pt=1.0, y_pt=36.0, step_pt=8.6)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0],
                  ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_yticks([-0.75, -0.5, -0.25, 0.0, 0.5, 1.0],
                  ["−0.75", "−0.5", "−0.25", "0", "0.5", "1"])
    ax.set_xlabel("conflict dose χ")
    ax.set_ylabel("initial signed utility s(χ)")
    _rug(ax, DOSES, y0=C_YLIM[0])
    return {"max_dev": max_dev, "excluded": excluded}


# ── D: trained order as predicted ─────────────────────────────────────────
D_XLIM = (-0.66, 2.56)
D_YLIM = (0.18, 1.31)
D_BREAK = 1.09
D_STRIP = (1.11, 1.31)
# QA 2026-09-10: the per-seed dose values are sweep LEVELS, so at three of
# the five levels the whole seed row sits on the same y as the analytic
# diamond and the trained circle.  A plain +/-0.30 jitter therefore threaded
# the string through both summary markers.  The string keeps its width and
# its honest y, and opens a clear lane of +/-D_SEED_GAP around the category
# centre -- 4.9 pt, i.e. half a 4.6 pt summary marker plus half a 2.9 pt seed
# dot plus 0.75 pt -- so the two marks stand beside the cloud, not inside it.
D_SEED_GAP = 0.17
D_SEED_STEP = 0.032


def _seed_lane(n):
    """PRIVATE (G5): symmetric x-jitter with a clear lane at the centre."""
    hi = (n + 1) // 2
    left = [-(D_SEED_GAP + k * D_SEED_STEP) for k in range(hi)]
    right = [D_SEED_GAP + k * D_SEED_STEP for k in range(n - hi)]
    return np.array(sorted(left + right)), D_SEED_GAP + (hi - 1) * D_SEED_STEP


def _exponent(ax, corner, x_pt, y_pt, base, exp, tail="", *, color):
    """PRIVATE (G5): ``10⁻⁶`` as base + raised exponent, both at 7 pt.

    Nimbus Sans has no U+207B and CF-2 bans mathtext, so the exponent is a
    second 7 pt token raised off the base line, exactly as
    ``token_subscript`` drops a subscript.
    """
    art = ax.annotate(base, xy=corner, xycoords="axes fraction",
                      xytext=(x_pt, y_pt), textcoords="offset points",
                      fontsize=PT_BASE, color=color, ha="left", va="top",
                      zorder=6, annotation_clip=False)
    sup = ax.annotate(exp, xy=(1.0, 1.0), xycoords=art, xytext=(0.3, 2.0),
                      textcoords="offset points", fontsize=PT_BASE,
                      color=color, ha="left", va="top", zorder=6,
                      annotation_clip=False)
    if tail:
        ax.annotate(tail, xy=(1.0, 0.0), xycoords=(sup, art),
                    xytext=(0.6, 0.0), textcoords="offset points",
                    fontsize=PT_BASE, color=color, ha="left", va="bottom",
                    zorder=6, annotation_clip=False)
    return art


def boundary_order(ax, crossings: pd.DataFrame, seeds: pd.DataFrame,
                   order: dict) -> None:
    """Analytic χ_c against the trained crossing and each seed's own dose."""
    crossings = crossings.sort_values("branches")
    xs = np.arange(3, dtype=float)
    predicted = crossings.predicted_boundary.to_numpy(float)
    trained = crossings.trained_mean_curve_chance_crossing.to_numpy(float)

    ax.set_xlim(*D_XLIM)
    ax.set_ylim(*D_YLIM)
    tint_patch(ax, ("rect", D_XLIM[0], D_STRIP[0], D_XLIM[1] - D_XLIM[0],
                    D_STRIP[1] - D_STRIP[0]),
               color="mute", pct=6, edge=False, radius_pt=1.5, zorder=0.2,
               clip_on=True)
    for dy in (-0.013, 0.013):
        ax.plot([D_XLIM[0] - 0.05, D_XLIM[0] + 0.07],
                [D_BREAK + dy - 0.012, D_BREAK + dy + 0.012],
                color=EDGE, lw=LW_HAIR, clip_on=False, zorder=6,
                solid_capstyle="butt")
    # QA 2026-09-09: the plan's verbatim strip label is 131 pt at 7 pt and
    # the axes are 93.2 pt, so it is set on two right-aligned lines inside a
    # strip deepened from 0.14 to 0.20 data units (20.5 pt) to hold them.
    _corner_lines(ax, ("no crossing within", "the sweep (χ ≤ 1)"),
                  corner=(1.0, 1.0), x_pt=-2.0, y_pt=-2.0, step_pt=8.0,
                  ha="right")

    for i, b in enumerate(BRANCHES):
        column = seeds[seeds.branches.eq(b)].first_at_or_below_chance_accuracy_dose
        values = column.dropna().to_numpy(float)
        levels = sorted(set(values))
        for k, level in enumerate(levels):
            n = int(np.sum(values == level))
            jitter, reach = _seed_lane(n)
            ax.plot(xs[i] + jitter, np.full(n, level), linestyle="none",
                    marker="o", ms=SEED_MS, markerfacecolor="none",
                    markeredgecolor=EDGE, markeredgewidth=LW_HAIR,
                    alpha=0.85, zorder=2.2)
            # the count sits above the top cluster and below the bottom one:
            # beside it, a 15-seed row at 0.60 module-widths of jitter runs
            # into the axis break on the left and the direct labels on the
            # right
            top = (k == len(levels) - 1)
            if top:
                ax.annotate(f"×{n}", xy=(xs[i], level), xytext=(0.0, 5.0),
                            textcoords="offset points", fontsize=PT_BASE,
                            color=MUTE, ha="center", va="bottom", zorder=6)
            else:
                ax.annotate(f"×{n}", xy=(xs[i] - reach - 0.13, level),
                            xytext=(-1.0, 0.0), textcoords="offset points",
                            fontsize=PT_BASE, color=MUTE, ha="right",
                            va="center", zorder=6)
        missing = int(column.isna().sum())
        if missing:
            ax.plot(xs[i] + np.linspace(-0.16, 0.16, missing),
                    np.full(missing, 1.250), linestyle="none", marker="^",
                    ms=MARKER_MS - 0.8, markerfacecolor="white",
                    markeredgecolor=AMBER, markeredgewidth=LW_EDGE, zorder=5)
            # QA 2026-09-09: right of the triangles, inside the axes (the
            # left-of-cluster placement was struck through by the spine)
            ax.text(xs[i], 1.165, f"{missing}/20", fontsize=PT_BASE,
                    color=AMBER_TEXT, ha="center", va="center", zorder=6)

    for i in range(3):
        ax.plot([xs[i], xs[i]], [predicted[i], trained[i]], color=INK,
                lw=LW_REF, dashes=DASHES, zorder=3.0, solid_capstyle="butt")
    ax.plot(xs, predicted, linestyle="none", marker="D", ms=MARKER_MS,
            markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=LW_EDGE, zorder=4)
    ax.plot(xs, trained, linestyle="none", marker="o", ms=MARKER_MS,
            markerfacecolor=AMBER, markeredgecolor="white",
            markeredgewidth=LW_HAIR, zorder=5)
    # the printed pair per branch count, direct-labelled once at B = 2 and
    # then given as bare values, so the two series keep their identity
    # without a legend and without a footer table
    token_subscript(ax, 0.44, predicted[0] + 0.010, "analytic χ", "c", "",
                    size=PT_BASE, sub_size=PT_BASE, color=INK, ha="left",
                    va="bottom")
    ax.text(0.44, trained[0] - 0.010, "trained crossing", fontsize=PT_BASE,
            color=AMBER_TEXT, ha="left", va="top", zorder=6)
    # the three printed pairs, colour-keyed to the two direct labels above;
    # per-cluster tags do not fit -- the x axis is 3.2 categorical units on
    # 93 pt, so a 46 pt value pair is half the panel wide
    analytic = ", ".join(f"{v:.3f}" for v in predicted)
    measured = ", ".join(f"{v:.3f}" for v in trained)
    _corner_lines(ax, (f"analytic {analytic}",), corner=(0.0, 0.0),
                  x_pt=1.0, y_pt=34.0, color=INK)
    _corner_lines(ax, (f"trained {measured}",), corner=(0.0, 0.0),
                  x_pt=1.0, y_pt=25.4, color=AMBER_TEXT)
    _corner_lines(ax, (
        f"B = 8 < 4 < 2 in "
        f"{order['accuracy_strict_order_pairs']}/"
        f"{order['accuracy_total_pairs']} seeds",
    ), corner=(0.0, 0.0), x_pt=1.0, y_pt=16.8)
    _exponent(ax, (0.0, 0.0), 1.0, 8.2, "P = 1.9 × 10", "−6",
              "; n = 20 per B", color=MUTE)
    ax.set_xticks(xs, [str(b) for b in BRANCHES])
    ax.set_yticks([0.6, 0.8, 1.0], ["0.6", "0.8", "1.0"])
    ax.set_xlabel("branches B")
    ax.set_ylabel("chance-crossing dose χ")


# ── E: only shared credit forgets (the forest idiom) ──────────────────────
E_ROWS = (
    ("neuron-\nshared", "neuron_shared_k1", "scalar", "s"),
    ("branch-\nspecific", "correct_subtree_k2", "shunting", "o"),
    ("deranged", "within_neuron_deranged_k2", "point_mlp", "^"),
    ("random\nrank-2", "random_dense_rank2", "point_mlp", "^"),
)
E_XLIM = (-8.0, 86.0)


def forgetting_forest(canvas: NativeCanvas, ax, summary: pd.DataFrame,
                      seeds: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    """Context-0 accuracy lost after a context switch, four credit routes."""
    rows = []
    for label, condition, colour, marker in E_ROWS:
        s = summary[summary.condition.eq(condition)].iloc[0]
        rows.append({
            "label": label,
            "mean": 100.0 * float(s.mean_context_switch_forgetting),
            "lo": 100.0 * float(s.ci95_low_context_switch_forgetting),
            "hi": 100.0 * float(s.ci95_high_context_switch_forgetting),
            "seeds": list(100.0 * seeds[seeds.condition.eq(condition)]
                          .context_switch_forgetting.to_numpy(float)),
            "color": colour, "marker": marker, "n": int(s.n_seeds),
        })
    out = canvas.forest(
        ax, rows, value_label="context-0 accuracy lost (pp)",
        reference=0.0, reference_label=None, xlim=E_XLIM, tag="",
        label_size=PT_BASE, band=False)
    # The four rows keep the top 4.04 row-units; the strip below them carries
    # the reference label and the n / interval / endpoint tag, because the
    # forest's own right-aligned tag above the top spine lands on this
    # panel's title at 93 pt of width.
    ax.set_ylim(5.30, -0.62)
    for line in list(ax.lines):
        xd = list(line.get_xdata())
        if len(xd) == 2 and xd[0] == xd[1] == 0.0 and line.get_linestyle() != "-":
            line.remove()
    ax.plot([0.0, 0.0], [-0.55, 3.30], color=MUTE, lw=LW_REF, zorder=1.0,
            dashes=(2.6, 2.0), solid_capstyle="butt")
    # a right-aligned value column: a per-row note set inside the axes lands
    # on its own interval, and outside the right spine it would leave the
    # page (E is the last module column)
    # QA 2026-09-09: every row now carries its interval.  The headline row's
    # `55.7 [53.8, 57.6]` is 61.9 pt at 7 pt and its own seed fan already
    # reaches 68.9 pt of the 93.2 pt axis, so the interval is set on a second
    # right-aligned line half a row below the value, clear of the fan and of
    # the branch-specific row's own tag.
    values = [f'{_minus(r["mean"])} [{_minus(r["lo"])}, {_minus(r["hi"])}]'
              for r in rows]
    ties = int((np.asarray(rows[1]["seeds"]) == 0.0).sum())
    values[1] = f'{_minus(rows[1]["mean"])} ({ties}/{len(rows[1]["seeds"])})'
    head = values[0].split(" [")
    values[0] = head[0]
    for y, text in zip(out["ypos"], values):
        ax.annotate(text, xy=(1.0, y), xycoords=("axes fraction", "data"),
                    xytext=(7.0, 0.0), textcoords="offset points",
                    ha="right", va="center", fontsize=PT_BASE, color=MUTE,
                    zorder=6, annotation_clip=False)
    ax.annotate("[" + head[1], xy=(1.0, out["ypos"][0] + 0.42),
                xycoords=("axes fraction", "data"), xytext=(7.0, 0.0),
                textcoords="offset points", ha="right", va="center",
                fontsize=PT_BASE, color=MUTE, zorder=6, annotation_clip=False)
    # CF-7: right-aligned at the TOP of the rule (the forest's own placement
    # is above the top spine, which is this panel's title band)
    # CF-7 asks for the label right-aligned at the top of the rule; the rule
    # is 8 pp from the left spine here, which is narrower than the label, so
    # it is set immediately to the RIGHT of the rule instead of running out
    # of the axes and over the top row's tick
    ax.annotate("no forgetting", xy=(0.0, -0.42), xycoords=("data", "data"),
                xytext=(2.5, 0.0), textcoords="offset points",
                fontsize=PT_BASE, color=MUTE, ha="left", va="center",
                zorder=6, annotation_clip=False)
    c = contrasts[contrasts.contrast.eq("correct - neuron shared")
                  & contrasts.endpoint.eq("context_switch_forgetting")].iloc[0]
    # anchored at the panel's own left edge, not the axes' -- below the rows
    # the row-label gutter is free, so the footer gets the full 126 pt
    _corner_lines(ax, (
        "branch-specific − neuron-shared:",
        f"{_signed(100.0 * float(c.mean_difference))} pp "
        f"[{_minus(100.0 * float(c.ci95_low))}, "
        f"{_minus(100.0 * float(c.ci95_high))}], "
        f"P = {float(c.wilcoxon_p_two_sided):.3f}",
        "n = 10 paired seeds; mean and 95 %",
        "bootstrap, after 20 switch epochs",
    # QA 2026-09-09: lifted 3 pt so the last baseline clears the x-axis
    # spine by 3.2 pt (it stood 0.2 pt off it), while the zero rule is
    # shortened to y = 3.30 so the first footer line still clears its foot.
    ), corner=(0.0, 0.0), x_pt=-DATA_LEFT_PT, y_pt=36.0, step_pt=8.6)
    ax.set_xticks([0, 20, 40, 60])
    return out


# ── F/G/H: dose facets ────────────────────────────────────────────────────
F_XLIM = (-0.03, 1.06)
F_YLIM = (0.0, 102.0)
BAND_ALPHA = 0.22
# per facet: the height of the chi = 1 shared-deficit tag, chosen so the tag
# sits in the band its own facet's steep amber segment leaves empty
# per facet: (x, y) of the chi = 1 shared-deficit tag and of the chi = 0
# deranged-deficit tag, each chosen to sit in the wedge its own facet's
# curves leave empty (the amber step moves left as B grows)
FULL_TAG = {2: (0.80, 66.0, "right"), 4: (0.72, 55.0, "left"),
            8: (0.98, 41.0, "right")}
ZERO_TAG = {2: (0.03, 64.0), 4: (0.03, 65.0), 8: (0.03, 73.5)}


def accuracy_facet(ax, summary: pd.DataFrame, seeds: pd.DataFrame,
                   contrasts: pd.DataFrame, branches: int, *, first: bool,
                   notes) -> None:
    ax.set_xlim(*F_XLIM)
    ax.set_ylim(*F_YLIM)
    reference_line(ax, 50.0, axis="y", label=None, color=MUTE, zorder=0)
    boundary = CHI_C[branches]
    ax.plot([boundary, boundary], [22.0, 80.5], color=MUTE, lw=LW_REF,
            dashes=DASHES, zorder=0.5, solid_capstyle="butt")
    rng = np.random.default_rng(2026)
    for condition, colour, marker, _name in SERIES:
        part = summary[summary.condition.eq(condition)
                       & summary.branches.eq(branches)] \
            .sort_values("conflict_probability")
        x = part.conflict_probability.to_numpy(float)
        m = 100.0 * part.mean_test_accuracy.to_numpy(float)
        ax.fill_between(x, 100.0 * part.ci95_low_test_accuracy,
                        100.0 * part.ci95_high_test_accuracy, color=colour,
                        alpha=BAND_ALPHA, linewidth=0, zorder=1)
        ax.plot(x, m, color=colour, lw=LW_DATA, marker=marker,
                ms=MARKER_MS - 0.8, markerfacecolor="white",
                markeredgecolor=colour, markeredgewidth=LW_EDGE,
                zorder=3 if condition == "within_neuron_deranged"
                else (3.5 if condition == "neuron_shared_k1" else 4.0))
        swarm = 100.0 * seeds[seeds.condition.eq(condition)
                              & seeds.branches.eq(branches)
                              & seeds.conflict_probability.eq(1.0)] \
            .test_accuracy.to_numpy(float)
        ax.plot(1.0 + rng.uniform(-0.018, 0.018, swarm.size), swarm,
                linestyle="none", marker="o", ms=SEED_MS,
                markerfacecolor=colour, markeredgecolor="none", alpha=0.55,
                zorder=2.5)
    _rug(ax, DOSES, y0=F_YLIM[0])
    # token_subscript grows to the RIGHT of its base, so the chain is placed
    # by its measured width immediately left of the rule; anchored on the
    # rule it would run past the right spine at B = 2 and be clipped
    chain = _text_w_pt(ax, f"χc = {boundary:.2f}", PT_BASE) + 1.2
    per_unit = 93.16 / (F_XLIM[1] - F_XLIM[0])
    token_subscript(ax, boundary - 0.035 - chain / per_unit, 100.0, "χ", "c",
                    f" = {boundary:.2f}", size=PT_BASE, sub_size=PT_BASE,
                    color=MUTE, ha="left", va="top")
    zero = contrasts[contrasts.contrast.eq("correct - deranged")
                     & contrasts.branches.eq(branches)
                     & contrasts.endpoint.eq("test_accuracy")
                     & contrasts.conflict_probability.eq(0.0)].iloc[0]
    full = contrasts[contrasts.contrast.eq("correct - shared")
                     & contrasts.branches.eq(branches)
                     & contrasts.endpoint.eq("test_accuracy")
                     & contrasts.conflict_probability.eq(1.0)].iloc[0]
    y_grey0 = 100.0 * float(summary[summary.condition.eq("within_neuron_deranged")
                                    & summary.branches.eq(branches)
                                    & summary.conflict_probability.eq(0.0)]
                            .mean_test_accuracy.iloc[0])
    zx, zy = ZERO_TAG[branches]
    ax.text(zx, zy, f"{_signed(-100.0 * float(zero.mean_difference))} pp",
            fontsize=PT_BASE, color=GRAY_TEXT, ha="left", va="center",
            zorder=6)
    ax.plot([zx + 0.02, 0.012], [zy - 3.8, y_grey0 + 1.6], color=GRAY,
            lw=LW_HAIR, zorder=1.5, solid_capstyle="round")
    fx, fy, fha = FULL_TAG[branches]
    ax.text(fx, fy, f"{_signed(100.0 * float(full.mean_difference))} pp",
            fontsize=PT_BASE, color=AMBER_TEXT, ha=fha, va="center",
            zorder=6)
    if first:
        ax.text(0.03, 84.0, "branch-specific", fontsize=PT_BASE,
                color=GREEN_TEXT, ha="left", va="bottom", zorder=6)
        ax.text(0.03, 76.0, "neuron-shared", fontsize=PT_BASE,
                color=AMBER_TEXT, ha="left", va="top", zorder=6)
        ax.plot([0.20, 0.20], [76.4, 79.2], color=AMBER, lw=LW_HAIR,
                zorder=1.5, solid_capstyle="round")
        ax.text(0.03, 31.0, "deranged route", fontsize=PT_BASE,
                color=GRAY_TEXT, ha="left", va="center", zorder=6)
        # CF-7: right-aligned at the right end of the rule, just left of the
        # chi = 1 swarm; below the rule, because the amber curve's steep
        # segment occupies the band immediately above it from chi = 0.91.
        ax.annotate("chance", xy=(0.930, 50.0), xytext=(0.0, -1.8),
                    textcoords="offset points", fontsize=PT_BASE, color=MUTE,
                    ha="right", va="top", zorder=6)
        ax.set_ylabel("held-out accuracy (%)")
    # QA 2026-09-09: a note may be given as (text, x_pt).  F's block is
    # re-split so its short top line starts 12 pt inside the axes and sits
    # 5 pt below the '20' y tick, which it previously shared a baseline with.
    for i, note in enumerate(notes):
        text, x_pt = note if isinstance(note, tuple) else (note, 1.0)
        ax.annotate(text, xy=(0.0, 0.0), xycoords="axes fraction",
                    xytext=(x_pt, 3.0 + 8.0 * (len(notes) - 1 - i)),
                    textcoords="offset points", fontsize=PT_BASE, color=MUTE,
                    ha="left", va="bottom", zorder=6, annotation_clip=False)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_yticks([20, 50, 80])
    if not first:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_xlabel("conflict dose χ")


# ── checks, curated table, layout ─────────────────────────────────────────
def _check_identities(contrasts, interactions, order) -> dict:
    if not (order["accuracy_strict_order_pairs"]
            == order["accuracy_total_pairs"] == 20):
        raise ValueError("Fig. 2D claims the predicted branch order in 20/20 "
                         f"seeds; boundary_order.json reports {order}")
    acc = contrasts[contrasts.endpoint.eq("test_accuracy")]
    ties = acc[acc.contrast.isin(["correct - BP", "correct - gated point"])]
    if not (ties.ties.eq(20).all() and np.allclose(ties.mean_difference, 0.0)):
        raise ValueError("branch-specific, BP and gated point no longer tie "
                         "in all 20 pairs")
    positive = interactions.set_index("branches").loc[list(BRANCHES)] \
        .positive_pairs
    if not np.all(positive.to_numpy(int) == 20):
        raise ValueError("Fig. 2G claims 20/20 positive interaction slopes at "
                         f"each B; source reports {positive.to_dict()}")
    return {"ties": int(ties.ties.min()),
            "positive_pairs": int(positive.min())}


CURATED_COLUMNS = [
    "panel", "record", "branches", "conflict_probability", "condition",
    "n_seeds", "mean_test_accuracy", "ci95_low_test_accuracy",
    "ci95_high_test_accuracy", "mean_initial_signed_utility",
    "ci95_low_initial_signed_utility", "ci95_high_initial_signed_utility",
    "predicted_boundary", "trained_mean_curve_chance_crossing",
    "first_at_or_below_chance_accuracy_dose", "mean_context_switch_forgetting",
    "ci95_low_context_switch_forgetting", "ci95_high_context_switch_forgetting",
    "contrast", "mean_difference", "ci95_low", "ci95_high", "positive_pairs",
    "ties", "wilcoxon_p_two_sided",
]


def write_curated(summary, crossings, seed_boundaries, contrasts,
                  sub_summary, sub_contrasts) -> Path:
    """Emit ``source_data/curated_publication/figure_02_plotted.csv``."""
    rows = []
    shared = summary[summary.condition.eq("neuron_shared_k1")]
    for _, r in shared.sort_values(["branches", "conflict_probability"]).iterrows():
        rows.append({
            "panel": "C", "record": "measured initial signed utility",
            "branches": r.branches, "conflict_probability": r.conflict_probability,
            "condition": r.condition, "n_seeds": r.n_seeds,
            "mean_initial_signed_utility": r.mean_initial_signed_utility,
            "ci95_low_initial_signed_utility": r.ci95_low_initial_signed_utility,
            "ci95_high_initial_signed_utility": r.ci95_high_initial_signed_utility,
            "predicted_boundary": 1.0 - 2.0 * (r.branches - 1)
            * r.conflict_probability / r.branches,
        })
    for _, r in crossings.sort_values("branches").iterrows():
        rows.append({
            "panel": "D", "record": "analytic and trained crossing",
            "branches": r.branches, "predicted_boundary": r.predicted_boundary,
            "trained_mean_curve_chance_crossing":
                r.trained_mean_curve_chance_crossing, "n_seeds": 20,
        })
    for _, r in seed_boundaries.sort_values(["branches", "seed"]).iterrows():
        rows.append({
            "panel": "D", "record": f"seed {int(r.seed)} crossing",
            "branches": r.branches,
            "first_at_or_below_chance_accuracy_dose":
                r.first_at_or_below_chance_accuracy_dose,
        })
    for _, r in sub_summary.iterrows():
        rows.append({
            "panel": "E", "record": "context-switch forgetting",
            "condition": r.condition, "n_seeds": r.n_seeds,
            "mean_context_switch_forgetting": r.mean_context_switch_forgetting,
            "ci95_low_context_switch_forgetting":
                r.ci95_low_context_switch_forgetting,
            "ci95_high_context_switch_forgetting":
                r.ci95_high_context_switch_forgetting,
        })
    sc = sub_contrasts[sub_contrasts.endpoint.eq("context_switch_forgetting")]
    for _, r in sc.iterrows():
        rows.append({
            "panel": "E", "record": "paired contrast", "contrast": r.contrast,
            "n_seeds": r.n_pairs, "mean_difference": r.mean_difference,
            "ci95_low": r.ci95_low, "ci95_high": r.ci95_high,
            "positive_pairs": r.wins, "ties": r.ties,
            "wilcoxon_p_two_sided": r.wilcoxon_p_two_sided,
        })
    facet = {2: "F", 4: "G", 8: "H"}
    keep = [c for c, _, _, _ in SERIES]
    part = summary[summary.condition.isin(keep)]
    for _, r in part.sort_values(["branches", "condition",
                                  "conflict_probability"]).iterrows():
        rows.append({
            "panel": facet[int(r.branches)], "record": "held-out accuracy",
            "branches": r.branches, "conflict_probability": r.conflict_probability,
            "condition": r.condition, "n_seeds": r.n_seeds,
            "mean_test_accuracy": r.mean_test_accuracy,
            "ci95_low_test_accuracy": r.ci95_low_test_accuracy,
            "ci95_high_test_accuracy": r.ci95_high_test_accuracy,
        })
    ac = contrasts[contrasts.endpoint.eq("test_accuracy")
                   & contrasts.conflict_probability.isin([0.0, 1.0])]
    for _, r in ac.sort_values(["branches", "contrast",
                               "conflict_probability"]).iterrows():
        rows.append({
            "panel": facet[int(r.branches)], "record": "paired contrast",
            "branches": r.branches, "conflict_probability": r.conflict_probability,
            "contrast": r.contrast, "n_seeds": r.n_pairs,
            "mean_difference": r.mean_difference, "ci95_low": r.ci95_low,
            "ci95_high": r.ci95_high, "positive_pairs": r.positive_pairs,
            "ties": r.ties, "wilcoxon_p_two_sided": r.wilcoxon_p_two_sided,
        })
    frame = pd.DataFrame(rows, columns=CURATED_COLUMNS)
    CURATED.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CURATED, index=False)
    return CURATED


def register_curated(path: Path) -> None:
    """Add / refresh the figure-2 record in ``curated_publication``."""
    manifest = path.parent / "manifest.json"
    data = json.loads(manifest.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    record = {
        "figure": 2,
        "source": "scripts/build_main_figure_04.py",
        "path": "source_data/curated_publication/figure_02_plotted.csv",
        "sha256": digest,
        "scope": ("Display summaries from existing inputs; panel column "
                  "identifies the selected data. Not an additional "
                  "experiment."),
    }
    records = [r for r in data["records"] if r.get("figure") != 2]
    records.append(record)
    # 2026-09-10: the manifest now also carries the supplement's "S1".."S36"
    # records, so sort main figures first by number and SI figures after.
    def _order(record):
        value = str(record.get("figure", ""))
        digits = re.sub(r"\D", "", value)
        return (value.upper().startswith("S"), int(digits) if digits else 0, value)

    data["records"] = sorted(records, key=_order)
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    readme = path.parent / "README.md"
    line = ("Figure 2 panels C-H are emitted by `scripts/build_main_figure_04.py` "
            "from `source_data/path_necessity_fashion/` and "
            "`source_data/trained_subtree_address/`.")
    text = readme.read_text()
    if line not in text:
        readme.write_text(text.rstrip("\n") + "\n\n" + line + "\n")


def _equalize_row_widths(canvas: NativeCanvas, rows) -> None:
    """One axes width for every 4-module panel of a row (audit L2)."""
    locks = canvas.lock_reserves()
    for names in rows:
        totals = {n: locks[n][0] + locks[n][1] for n in names}
        target = max(totals.values())
        for n in names:
            short = target - totals[n]
            if short > 0.05:
                canvas.declare_reserve(n, right=locks[n][1] + short)
    canvas.lock_reserves()


LAST_BUILD: dict = {}


def build() -> list:
    mpl.rcParams["lines.markeredgewidth"] = LW_EDGE
    summary = pd.read_csv(PATH_NECESSITY / "condition_summary.csv")
    seed_outcomes = pd.read_csv(PATH_NECESSITY / "seed_outcomes.csv")
    crossings = pd.read_csv(PATH_NECESSITY / "plotted_crossings.csv")
    seed_boundaries = pd.read_csv(PATH_NECESSITY / "boundary_by_seed.csv")
    contrasts = pd.read_csv(PATH_NECESSITY / "paired_contrasts.csv")
    interactions = pd.read_csv(PATH_NECESSITY / "interaction_summary.csv")
    order = json.loads((PATH_NECESSITY / "boundary_order.json").read_text())
    sub_summary = pd.read_csv(SUBTREE / "condition_summary.csv")
    sub_seeds = pd.read_csv(SUBTREE / "seed_outcomes.csv")
    sub_contrasts = pd.read_csv(SUBTREE / "paired_contrasts.csv")
    facts = _check_identities(contrasts, interactions, order)

    canvas = NativeCanvas(
        HEIGHT_IN, 3, row_weights=ROW_PT,
        hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT, margins=MARGINS,
    )
    # the 5 pt bottom inset keeps the module-normalised slot fill of the two
    # schematics within EMPHASIS_MAX_RATIO of the data panels, whose own slot
    # fill is capped by the shared 33 pt forest gutter.
    ax_a = canvas.panel("A", 0, 0, 7, schematic=True, lock=False,
                        inset_pt=(0.0, 0.0, 0.0, 6.0))
    ax_b = canvas.panel("B", 0, 7, 5, schematic=True, lock=False,
                        inset_pt=(0.0, 0.0, 0.0, 6.0))
    ax_c = canvas.panel("C", 1, 0, 4, title="Predicted and measured")
    ax_d = canvas.panel("D", 1, 4, 4, title="Trained order as predicted")
    ax_e = canvas.panel("E", 1, 8, 4, title="Only shared credit forgets")
    ax_f = canvas.panel("F", 2, 0, 4, title="B = 2: shared still learns")
    ax_g = canvas.panel("G", 2, 4, 4, title="B = 4: boundary moves left",
                        sharey=ax_f)
    ax_h = canvas.panel("H", 2, 8, 4, title="B = 8: earliest collapse",
                        sharey=ax_f)

    frame_a = branch_conflict_task(ax_a)
    frame_b = backward_credit_schematic(ax_b)
    for frame in (frame_a, frame_b):
        frame.require_soma_lowest()
        frame.require_delta0()

    stats_c = initial_utility(ax_c, summary)
    boundary_order(ax_d, crossings, seed_boundaries, order)
    for name in ("C", "D", "E", "F", "G", "H"):
        canvas.declare_reserve(name, left=DATA_LEFT_PT)
    forgetting_forest(canvas, ax_e, sub_summary, sub_seeds, sub_contrasts)
    canvas.declare_reserve("E", left=DATA_LEFT_PT)

    sub = summary[summary.branches.eq(8)
                  & summary.conflict_probability.eq(1.0)]
    shared8 = 100.0 * float(
        sub[sub.condition.eq("neuron_shared_k1")].mean_test_accuracy.iloc[0])
    der8 = 100.0 * float(
        sub[sub.condition.eq("within_neuron_deranged")].mean_test_accuracy.iloc[0])
    accuracy_facet(ax_f, summary, seed_outcomes, contrasts, 2, first=True,
                   notes=(("n = 20 seeds;", 12.0),
                          ("mean and 95 % bootstrap", 1.0),
                          (f"BP = gated point tie {facts['ties']}/20", 1.0)))
    accuracy_facet(ax_g, summary, seed_outcomes, contrasts, 4, first=False,
                   notes=("selection × conflict slope",
                          f"> 0 in {facts['positive_pairs']}/20 seeds (each B)"))
    accuracy_facet(ax_h, summary, seed_outcomes, contrasts, 8, first=False,
                   notes=("shared and deranged fall",
                          f"below chance ({shared8:.1f} %, {der8:.1f} %)"))

    style_direct_color_labels(canvas.fig)
    _equalize_row_widths(canvas, (("C", "D", "E"), ("F", "G", "H")))
    findings = canvas.align_letters()
    LAST_BUILD.update({"canvas": canvas, "stats_c": stats_c, "facts": facts})
    COMPONENT.parent.mkdir(parents=True, exist_ok=True)
    problems = canvas.save(COMPONENT, name="main_figure_04_native")
    PUBLISHED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(COMPONENT, PUBLISHED)
    path = write_curated(summary, crossings, seed_boundaries, contrasts,
                         sub_summary, sub_contrasts)
    register_curated(path)
    return list(findings) + list(problems)


def main() -> None:
    problems = build()
    for problem in problems:
        print(f"  {problem}")
    live_h = CANVAS_H_PT - MARGINS.top - MARGINS.bottom
    live_w = 518.4 - MARGINS.left - MARGINS.right
    frac = (250.9 * ROW_PT[0] + 169.5 * ROW_PT[0]) / (live_w * live_h)
    print(f"  canvas {FIG_W * 72:.1f} x {CANVAS_H_PT:.1f} pt, "
          f"schematic fraction {frac * 100:.1f} %")


if __name__ == "__main__":
    main()

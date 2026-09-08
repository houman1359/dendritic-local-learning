#!/usr/bin/env python3
"""Gallery of the shared schematic glyph library on one native canvas.

Every helper family in :mod:`native_schematics` is drawn once, in the slot
sizes the main figures actually use (3-, 4- and 5-module cells), so a
per-figure implementer can see each glyph at print size and the strict
audit can prove the library itself emits only journal tokens::

    python3 scripts/native_schematics_gallery.py
    python3 scripts/figure_canvas.py --audit \\
        figures/components/native_schematics_gallery.pdf --strict

Rows: A anatomy primitives, B the four credit-delivery modes, C the
dictionary triplet with a row-aligned site tree; D task cards and badges,
E teacher / student / chance furniture with the rule key, F a partition
with the focal shunt, G the local-gate tree; H the stage pair, I the
operator tree pair, J the measured-response boundary card.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from figure_canvas import (  # noqa: E402
    COLORS,
    LW_EDGE,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    PT_TICK,
    Margins,
    NativeCanvas,
)
from journal_style import label_color  # noqa: E402
from native_schematics import (  # noqa: E402
    CONTACT_DIA_PT,
    Frame,
    LINE_BAND_PT,
    draw_local_gate_tree,
    draw_measured_boundary,
    draw_operator_tree_pair,
    draw_stage_pair,
    reference_line,
)

OUT = HERE.parent / "figures" / "components" / "native_schematics_gallery.pdf"
INK = COLORS["ink"]


# ── A: anatomy primitives ─────────────────────────────────────────────────
def panel_primitives(ax):
    f = Frame(ax)
    core = f.task_card((0.0, 0.0, 1.0, 1.0))
    x0, y0, w, h = core
    ncol, nrow = 3, 4
    cw, ch = w / ncol, h / nrow
    items = [
        ("soma", lambda c: f.soma(c, output=True, label="z")),
        ("junction", lambda c: f.junction(c)),
        ("site disc", lambda c: f.junction(c, r_pt=2.1,
                                           site_color=COLORS["shunting"])),
        ("taper 0–3", "taper"),
        ("exc", lambda c: f.contact(c, kind="exc")),
        ("inh", lambda c: f.contact(c, kind="inh")),
        ("inh off", lambda c: f.contact(c, kind="inh", active=False)),
        ("shunt", "shunt"),
        ("gate", lambda c: f.gate(c)),
        ("closed", "closed"),
        ("error in", "error"),
        ("ghost", "ghost"),
    ]
    for i, (caption, draw) in enumerate(items):
        r, c = divmod(i, ncol)
        cx = x0 + (c + 0.5) * cw
        cy = y0 + h - (r + 0.5) * ch
        gx, gy = cx, cy + f.fy(3.0)
        if draw == "taper":
            seg, gap = 5.0, 2.0
            left = cx - f.fx((4 * seg + 3 * gap) / 2.0)
            for level in range(4):
                xa = left + f.fx(level * (seg + gap))
                f.dendrite((xa, gy), (xa + f.fx(seg), gy), level=level)
        elif draw == "shunt":
            f.shunt((cx - f.fx(11.0), gy))
        elif draw == "closed":
            base = (cx - f.fx(2.0), gy - f.fy(2.0))
            f.dendrite((cx - f.fx(6.0), gy - f.fy(9.0)), base, level=1)
            kids = [f.dendrite(base, (cx + f.fx(6.0), gy + f.fy(7.0)), level=3)]
            f.gate(base, closed=True, descendants=kids,
                   incident=[(8.0 / f.w_pt, 9.0 / f.h_pt),
                             (-4.0 / f.w_pt, -7.0 / f.h_pt)])
        elif draw == "error":
            soma = (cx - f.fx(11.0), gy + f.fy(5.0))
            f.soma(soma)
            f.error_in(soma, side="right")
        elif draw == "ghost":
            f.dendrite((cx - f.fx(8.0), gy - f.fy(5.0)), (cx, gy), level=1,
                       ghost=True)
            f.dendrite((cx, gy), (cx + f.fx(8.0), gy + f.fy(5.0)), level=3,
                       ghost=True)
            f.junction((cx, gy), ghost=True)
        else:
            draw((gx, gy))
        # captions need a 26 pt cell under the glyph; shorter cells keep
        # the glyph and drop the word rather than piling the two together
        if ch * f.h_pt >= 26.0:
            f.text((cx, y0 + h - (r + 1) * ch + f.fy(2.0)), caption,
                   size=PT_SMALL, color=COLORS["mute"], va="bottom")
    return ax


# ── B: the four credit-delivery modes ─────────────────────────────────────
def panel_delivery(ax):
    f = Frame(ax)
    core = f.task_card((0.0, 0.0, 1.0, 1.0))
    cells = [Frame.inset(c, left=0.02, right=0.02, top=0.02, bottom=0.02)
             for c in _grid(f, core, 2, 2, gap_pt=4.0)]
    modes = (("scalar", dict(mode="scalar"), "scalar", 11.0, 6.0),
             ("neuron", dict(mode="neuron"), "neuron", 0.0, 0.0),
             ("subtree", dict(mode="subtree", K=4, alpha_tags=True),
              "K = 4", 9.0, 0.0),
             ("exact", dict(mode="exact", alpha_tags=True), "exact", 0.0, 0.0))
    for cell, (name, kw, tag, top_pt, right_pt) in zip(cells, modes):
        body = (cell[0], cell[1], cell[2] - f.fx(right_pt),
                cell[3] - f.fy(top_pt))
        nodes = f.balanced_tree(body, mode="plain")
        f.credit_delivery(nodes, **kw)
        f.text((cell[0] + cell[2] - f.fx(2.0), cell[1] + f.fy(1.0)), tag,
               size=PT_SMALL, color=COLORS["mute"], ha="right", va="bottom")
    return ax


def _grid(f, rect, ncol, nrow, *, gap_pt=4.0):
    x0, y0, w, h = rect
    gx, gy = f.fx(gap_pt), f.fy(gap_pt)
    cw = (w - (ncol - 1) * gx) / ncol
    ch = (h - (nrow - 1) * gy) / nrow
    cells = []
    for r in range(nrow):
        for c in range(ncol):
            cells.append((x0 + c * (cw + gx), y0 + h - (r + 1) * ch - r * gy,
                          cw, ch))
    return cells


# ── C: dictionary triplet with a row-aligned site tree ────────────────────
def panel_dictionary(ax):
    f = Frame(ax)
    core = f.task_card((0.0, 0.0, 1.0, 1.0))
    x0, y0, w, h = core
    subtree_colors = ("shunting", "additive", "oracle")
    A = np.zeros((12, 3))
    for k in range(3):
        A[[k, 3 + 3 * k, 4 + 3 * k, 5 + 3 * k], k] = 1.0
    order = [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11]
    c = np.array([1.0, -1.0, 0.0])
    # the tree owns the row pitch: 12 rows at <= 8 pt, product rows snap to it
    pitch = min(8.0, (h * f.h_pt - LINE_BAND_PT - 8.0) / 12.0)
    tree_h = f.fy(12 * pitch)
    tree_y = y0 + f.fy(LINE_BAND_PT) + (h - f.fy(LINE_BAND_PT) - tree_h) / 2.0
    tree_rect = (x0 + f.fx(6.0), tree_y, f.fx(50.0), tree_h)
    nodes = f.site_tree(tree_rect, subtree_colors=subtree_colors,
                        numbered=True)
    prod_rect = (x0 + f.fx(60.0), y0, f.fx(66.0), h)
    f.dictionary_product(prod_rect, A[order], c, col_colors=subtree_colors,
                         row_groups=[4, 4, 4], align_to=nodes)
    # realized route supports (measured=True) beside the product -- only
    # when the slot leaves the 38 pt they need (a 5-module slot does)
    if w * f.w_pt < 60.0 + 66.0 + 12.0 + 26.0 + 6.0:
        return ax
    R = np.zeros((12, 4))
    R[:, 0] = 1.0
    R[0:4, 1] = 1.0
    R[4:8, 2] = 1.0
    R[8:12, 3] = 1.0
    mx = x0 + f.fx(60.0 + 66.0 + 12.0)
    my = nodes[nodes.order[-1]][1] - f.fy(pitch / 2.0)
    f.dictionary_matrix((mx, my, f.fx(4 * 6.5), f.fy(12 * pitch)), R,
                        measured=True, row_groups=[4, 4, 4], label="R")
    return ax


# ── D: task cards and badges ──────────────────────────────────────────────
def panel_cards(ax):
    f = Frame(ax)
    top, bottom = f.split(2, axis="y", gap_pt=6.0)
    core = f.task_card(top, title="compatible", emphasis=True)
    f.badge((top[0] + top[2] - f.fx(4.0), top[1] + top[3] - f.fy(3.5)),
            "local rule")
    body = Frame.inset(core, left=0.06, right=0.06, top=0.04, bottom=0.04)
    nodes = f.balanced_tree(body, depth=2, trunk=False, mode="inputs")
    f.credit_delivery(nodes, mode="subtree", targets=["JL"])
    core = f.task_card(bottom, title="conflicting", tone="control")
    f.badge((bottom[0] + bottom[2] - f.fx(4.0),
             bottom[1] + bottom[3] - f.fy(3.5)), "control")
    body = Frame.inset(core, left=0.06, right=0.06, top=0.04, bottom=0.04)
    f.balanced_tree(body, depth=2, trunk=False, mode="inputs", ghost=True)
    return ax


# ── E: teacher / student / chance furniture and the rule key ──────────────
def panel_furniture(ax):
    f = Frame(ax)
    core = f.task_card((0.0, 0.0, 1.0, 1.0))
    x0, y0, w, h = core
    key_pt, badge_pt, tick_pt = 22.0, 13.0, 12.0
    inset = (x0 + f.fx(14.0), y0 + f.fy(key_pt + badge_pt + tick_pt),
             w - f.fx(18.0), h - f.fy(key_pt + badge_pt + tick_pt + 5.0))
    inner = f.axes_inset(inset)
    x = np.linspace(0.0, 1.0, 80)
    teacher = 0.35 + 0.30 * np.sin(2 * np.pi * x)
    student = 0.35 + 0.30 * np.sin(2 * np.pi * x) * (1.0 - 0.18 * x)
    f.teacher_student(inner, x, teacher, student, rule_color="shunting",
                      label_student="student")
    inner.set_xlim(0.0, 1.0)
    inner.set_ylim(0.0, 1.05)
    inner.set_xticks([0.0, 1.0], ["0", "1"])
    inner.set_yticks([0.0, 1.0], ["0", "1"])
    inner.tick_params(labelsize=PT_TICK)
    reference_line(inner, 0.35, label="chance")
    inner.legend(loc="upper right", frameon=False, fontsize=PT_LEGEND,
                 handlelength=1.5, borderaxespad=0.1, labelspacing=0.25)
    by = y0 + f.fy(key_pt + badge_pt * 0.5)
    bx = x0 + f.fx(6.0)
    for kind in ("oracle", "exact"):
        art = f.badge((bx, by), kind, ha="left", va="center")
        bx += f.fx(_badge_w(f, kind) + 6.0)
    f.rule_key((x0, y0, w, f.fy(key_pt)),
               [("scalar", "scalar", "scalar"), ("exact", "bp", "exact"),
                ("subtree", "shunting", "subtree"),
                ("neuron", "credit_ink", "neuron")])
    return ax


def _badge_w(f, kind):
    from native_schematics import _text_w_pt
    return _text_w_pt(f.ax, kind, PT_SMALL) + 4.5


# ── F: partition (Fig. 8A idiom) with the focal shunt ──────────────────────
def panel_partition(ax):
    f = Frame(ax)
    core = f.task_card((0.0, 0.0, 1.0, 1.0), title="focal shunt")
    body = Frame.inset(core, left=0.05, right=0.05, top=0.02)
    body = (body[0], body[1] + f.fy(15.0), body[2], body[3] - f.fy(15.0))
    nodes = f.balanced_tree(body, mode="plain")
    f.partition(nodes,
                [nodes.subtree("JL"), nodes.subtree("JR"), ["S", "J1"]],
                colors=("shunting", "oracle", "mute"),
                labels=("descendants", "sister", "soma side"))
    site = _lerp(nodes["J1"], nodes["JL"], 0.86)
    f.contact(site, kind="inh")
    f.subscript((site[0] - f.fx(3.0), site[1] - f.fy(3.0)), "g", "shunt",
                size=PT_SMALL, color=COLORS["inh"], ha="right", va="top")
    f.error_in(nodes.soma, side="right")
    return ax


def _lerp(a, b, t):
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


# ── canvas ────────────────────────────────────────────────────────────────
SPEC = (
    ("A", 0, 0, 3, "Anatomy glyphs", panel_primitives),
    ("B", 0, 3, 4, "Four delivery modes", panel_delivery),
    ("C", 0, 7, 5, "Dictionary triplet, aligned rows", panel_dictionary),
    ("D", 1, 0, 3, "Cards and badges", panel_cards),
    ("E", 1, 3, 3, "Teacher and chance", panel_furniture),
    ("F", 1, 6, 3, "Partition, shunt", panel_partition),
    ("G", 1, 9, 3, "Local gate tree", draw_local_gate_tree),
    ("H", 2, 0, 4, "Stage pair", draw_stage_pair),
    ("I", 2, 4, 5, "Operator tree pair", draw_operator_tree_pair),
    ("J", 2, 9, 3, "Measured boundary", draw_measured_boundary),
)


def build(out=OUT, *, png=True, dpi=180, quiet=False):
    """Draw the gallery; returns the inherited layout problems (a list)."""
    canvas = NativeCanvas(
        490.0 / 72.0, 3, row_weights=[130, 128, 140], hgutter_pt=26.0,
        vgutter_pt=30.0,
        margins=Margins(left=34.0, right=12.0, top=22.0, bottom=14.0),
        lock_reserves=False,
    )
    for letter, row, col, span, title, draw in SPEC:
        ax = canvas.panel(letter, row, col, span, schematic=True, lock=False,
                          title=title)
        draw(ax)
    findings = canvas.align_letters()
    problems = canvas.save(Path(out), name="native_schematics_gallery",
                           png=png, dpi=dpi, quiet=quiet)
    return list(findings) + list(problems)


if __name__ == "__main__":
    problems = build()
    if problems:
        print("layout notes:", *problems, sep="\n  ")

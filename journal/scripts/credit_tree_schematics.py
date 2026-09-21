"""Credit-tree schematic library — matplotlib port of the talk's TikZ vocabulary.

Mirrors ``drafts/dendritic-local-learning/presentations/credit_tree_lib.tex``: one
parameterized dendritic tree with identical geometry in every mode; only the
decoration changes to show the quantity being defined or manipulated.  All
coordinates are the TikZ ones verbatim, so cross-panel changes read as the
same object and side-by-side trees align on their somas.

Public interface
    draw_credit_tree(ax, mode=..., scale=1.0, **kw)
        modes: plain | forward | eligibility | transport | scalar |
               coordinate (soma-only blue delta) | address (uses K=2|4|8) |
               gain | shunt (uses shunted=True/False)
    draw_deranged_pair(ax, scale=1.0, **kw)
        correct vs deranged coordinate assignment (two mini-tree pairs)

Style: every color, line width and font size comes from journal_style tokens.
The deck's stroke taper (1.60/1.20/0.90/0.70 pt) is normalized so the trunk
prints at LW_DATA; deck bold labels are demoted to regular weight (in-panel
bold is forbidden by the journal contract).  ``labels=False`` strips every
text artist so builders can reuse the anatomy as a tiny inset.  ``scale``
tempers stroke widths and marker sizes exactly like the TikZ ``\\ctlwf``
factor (fonts never scale; keep scale in 0.5-1.4).
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Arc, Circle, FancyArrowPatch

from journal_style import (
    COLORS,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    PT_ANNOT,
    PT_SMALL,
)

# ── color helpers (TikZ ``X!nn`` and ``A!nn!B`` mixes) ────────────────────


def mix(color, pct, base="white"):
    """TikZ ``color!pct!base``: pct% of color blended with (100-pct)% base."""
    a = np.asarray(to_rgb(COLORS.get(color, color)), dtype=float)
    b = np.asarray(to_rgb(COLORS.get(base, base)), dtype=float)
    f = float(pct) / 100.0
    return tuple(f * a + (1.0 - f) * b)


# ── fixed geometry (identical across ALL modes; TikZ coordinates) ─────────

P = {
    "S":   (0.00, 0.00),
    "J1":  (0.06, 0.85),
    "JL":  (-0.92, 1.52),
    "JR":  (0.88, 1.44),
    "JLL": (-1.72, 2.02),
    "JLR": (-0.58, 2.32),
    "JRL": (0.56, 2.28),
    "JRR": (1.62, 1.92),
    "T1":  (-2.30, 2.42),
    "T2":  (-1.52, 2.82),
    "T3":  (-0.92, 3.02),
    "T4":  (-0.22, 3.06),
    "T5":  (0.28, 3.04),
    "T6":  (0.94, 2.94),
    "T7":  (1.54, 2.72),
    "T8":  (2.24, 2.28),
}
ROOT_PT = (0.0, 0.17)          # trunk starts above the soma disc
SOMA_R = 0.185

# edge lists by taper level (trunk / level-2 / level-3 / terminal)
EDGES_A = [(ROOT_PT, "J1")]
EDGES_B = [("J1", "JL"), ("J1", "JR")]
EDGES_C = [("JL", "JLL"), ("JL", "JLR"), ("JR", "JRL"), ("JR", "JRR")]
EDGES_D = [("JLL", "T1"), ("JLL", "T2"), ("JLR", "T3"), ("JLR", "T4"),
           ("JRL", "T5"), ("JRL", "T6"), ("JRR", "T7"), ("JRR", "T8")]
JUNCTIONS = ("J1", "JL", "JR", "JLL", "JLR", "JRL", "JRR")

# Deck stroke taper in TikZ pt; normalized so the trunk prints at LW_DATA.
_TAPER_PT = {"A": 1.60, "B": 1.20, "C": 0.90, "D": 0.70}
_PT2LW = LW_DATA / _TAPER_PT["A"]

# marker diameters (pt, deck values)
MS_JUNCTION = 3.2
MS_SYN = 3.6

_BASE_XLIM = (-2.55, 2.55)
_BASE_YLIM = (-0.72, 3.32)

# Exactly two neutral tones are spent on de-emphasis, both mixes of the one
# palette gray: GHOST for a back-grounded arbor (strokes and its junction
# rings) and RIM for every soma / synapse rim on the page.  Before this the
# same de-emphasised tree carried nine different grays across one figure.
GHOST = mix("mute", 45)        # the single de-emphasis tint
RIM = mix("ink", 30)           # the single soma / synapse rim tone
# Amber #E2A23F carries only 2.2:1 against white, below the 4.5:1 floor for
# 6.8-7.4 pt type, so the amber SERIES keeps the palette token for fills,
# strokes and markers while amber TEXT is set in this one darkened tone
# (4.5:1).  It is the only text-only colour on the page.
AMBER_TEXT = mix("local", 60, "ink")


def _pt(name):
    return P[name] if isinstance(name, str) else name


def _lerp(a, b, f):
    ax_, ay = _pt(a)
    bx, by = _pt(b)
    return (ax_ + f * (bx - ax_), ay + f * (by - ay))


def _lwf(scale):
    """TikZ ``\\ctlwf``: tempers widths and capsules at small scales."""
    return min(1.15, max(0.62, float(scale)))


class _Tree:
    """Per-call drawing context: one axes, one scale, one label switch."""

    def __init__(self, ax, scale, labels):
        self.ax = ax
        self.f = _lwf(scale)
        self.labels = bool(labels)

    def lw(self, tikz_pt):
        """Deck stroke width (TikZ pt) -> journal line width."""
        return tikz_pt * _PT2LW * self.f

    def seg(self, a, b, color, tikz_pt, zorder=2):
        (x0, y0), (x1, y1) = _pt(a), _pt(b)
        self.ax.plot([x0, x1], [y0, y1], color=color, lw=self.lw(tikz_pt),
                     solid_capstyle="round", zorder=zorder)

    def edges(self, edge_list, color, tikz_pt, zorder=2):
        for a, b in edge_list:
            self.seg(a, b, color, tikz_pt, zorder=zorder)

    def tree(self, color, zorder=2):
        """Whole tapered tree in one color (TikZ ``\\CTplaintree``)."""
        self.edges(EDGES_A, color, _TAPER_PT["A"], zorder)
        self.edges(EDGES_B, color, _TAPER_PT["B"], zorder)
        self.edges(EDGES_C, color, _TAPER_PT["C"], zorder)
        self.edges(EDGES_D, color, _TAPER_PT["D"], zorder)

    def dot(self, at, size, fill, edge="none", edge_lw=0.0, zorder=4):
        x, y = _pt(at)
        self.ax.plot([x], [y], marker="o", ms=size * self.f, mfc=fill,
                     mec=edge, mew=edge_lw, ls="none", zorder=zorder)

    def junctions(self, names=JUNCTIONS, edge=None, zorder=3):
        edge = COLORS["dend"] if edge is None else edge
        for name in names:
            self.dot(name, MS_JUNCTION, "white", edge, LW_EDGE, zorder=zorder)

    def soma(self, fill, edge, edge_lw=LW_EDGE, zorder=3.5):
        self.ax.add_patch(Circle((0, 0), SOMA_R, facecolor=fill,
                                 edgecolor=edge, lw=edge_lw, zorder=zorder))

    def text(self, xy, s, color, *, size=PT_ANNOT, ha="center", va="center",
             zorder=6):
        if self.labels:
            self.ax.text(xy[0], xy[1], s, color=color, fontsize=size,
                         ha=ha, va=va, zorder=zorder)

    def arrow(self, a, b, color, tikz_pt=0.8, head=4.5, rad=0.0, zorder=4.5):
        style = f"-|>,head_length={head},head_width={0.62 * head}"
        arr = FancyArrowPatch(
            _pt(a), _pt(b), arrowstyle=style, mutation_scale=1.0,
            connectionstyle=f"arc3,rad={rad}", color=color,
            lw=self.lw(tikz_pt), capstyle="round", zorder=zorder,
        )
        self.ax.add_patch(arr)

    def capsule(self, tint, width_pt, chains, zorder=1.4):
        """Route-field capsule: pale solid tint, fat round stroke."""
        for chain in chains:
            xy = np.array([_pt(q) for q in chain], dtype=float)
            self.ax.plot(xy[:, 0], xy[:, 1], color=tint,
                         lw=width_pt * self.f, solid_capstyle="round",
                         solid_joinstyle="round", zorder=zorder)

    def stage(self, label):
        if label:
            self.text((0.0, -0.30), label, COLORS["mute"], ha="center",
                      va="top")


def _setup_axes(ax, xlim, ylim):
    # Register the frame as data limits instead of fixing xlim/ylim: with
    # ``adjustable="datalim"`` the equal-aspect draw then widens whichever
    # range is short without matplotlib warning about ignored fixed limits.
    ax.set_aspect("equal", adjustable="datalim")
    ax.update_datalim([(xlim[0], ylim[0]), (xlim[1], ylim[1])])
    ax.margins(0)
    ax.autoscale_view()
    ax.axis("off")


# ── mode decorations ──────────────────────────────────────────────────────


def _mode_plain(t):
    t.tree(COLORS["dend"])
    t.junctions()
    t.soma(COLORS["soma"], RIM)
    for a, b, f in (("JLL", "T1", 0.55), ("JLL", "T2", 0.62),
                    ("JLR", "T3", 0.55), ("JLR", "T4", 0.62),
                    ("JRL", "T5", 0.55), ("JRL", "T6", 0.62),
                    ("JRR", "T7", 0.55), ("JRR", "T8", 0.58)):
        t.dot(_lerp(a, b, f), MS_SYN, COLORS["exc"])
    t.dot(_lerp("J1", "JL", 0.55), MS_SYN, COLORS["inh"])
    t.dot(_lerp("JR", "JRL", 0.45), MS_SYN, COLORS["inh"])
    t.arrow((1.30, 0.60), (0.27, 0.09), COLORS["additive"], tikz_pt=0.9,
            rad=0.12)


def _mode_forward(t):
    t.tree(COLORS["dend"])
    t.junctions()
    t.soma(COLORS["soma"], RIM)
    syn_e = _lerp("JRR", "T7", 0.55)
    t.dot(syn_e, MS_SYN, COLORS["exc"])
    t.text((syn_e[0] + 0.12, syn_e[1] + 0.05), "gᵢ, Eᵢ",
           COLORS["exc"], size=PT_SMALL, ha="left")
    syn_i = _lerp("J1", "JL", 0.55)
    t.dot(syn_i, MS_SYN, COLORS["inh"])
    t.text((syn_i[0] - 0.10, syn_i[1] - 0.08), "gⱼ, Eⱼ",
           COLORS["inh"], size=PT_SMALL, ha="right")
    t.text((P["JRR"][0] + 0.20, P["JRR"][1] - 0.04), "Vₙ",
           COLORS["ink"], ha="left")
    t.text((0.26, -0.16), "V₀", COLORS["ink"], ha="left")
    qx, qy = P["JRR"][0] + 0.30, P["JRR"][1] - 0.62
    t.ax.plot([P["JRR"][0], qx], [P["JRR"][1], qy], color=COLORS["mute"],
              lw=LW_HAIR, dashes=(1.4, 1.3), zorder=1.8)
    t.text((qx, qy - 0.05), "Rₙᵗᵒᵗ", COLORS["mute"],
           size=PT_SMALL, va="top")


def _mode_eligibility(t):
    t.tree(GHOST)
    t.seg("JR", "JRL", COLORS["dend"], _TAPER_PT["C"], zorder=2.2)
    t.seg("JRL", "T6", COLORS["dend"], _TAPER_PT["D"], zorder=2.2)
    t.junctions(edge=GHOST)
    t.junctions(names=("JRL",))
    t.soma(COLORS["soma"], RIM)
    syn = _lerp("JRL", "T6", 0.55)
    t.dot(syn, 4.6, COLORS["exc"], RIM, 0.5 * _PT2LW)
    # zoom bubble to the side
    zc = (2.62, 1.30)
    for tip in ((zc[0] - 0.30, zc[1] + 0.86), (zc[0] - 0.88, zc[1] - 0.18)):
        t.ax.plot([syn[0], tip[0]], [syn[1], tip[1]], color=COLORS["grid"],
                  lw=LW_EDGE, zorder=1.2, solid_capstyle="round")
    # White halo pad under the keyed circle: masks ghost-tree strokes and
    # connector ends that would otherwise graze the ring (key/data ink mix).
    t.ax.add_patch(Circle(zc, 1.04, facecolor="white",
                          edgecolor="none", zorder=4.55))
    t.ax.add_patch(Circle(zc, 0.92, facecolor="white",
                          edgecolor=COLORS["mute"], lw=LW_EDGE, zorder=4.6))
    # row 1: presynaptic activity
    t.dot((zc[0] - 0.52, zc[1] + 0.44), MS_SYN, COLORS["exc"], zorder=5)
    t.text((zc[0] - 0.36, zc[1] + 0.44), "xᵢ", COLORS["ink"], ha="left")
    # "presyn." hugs the x_i label so the row stays inside the ring even on
    # small host axes, where fixed-point type covers more data units.
    t.text((zc[0] - 0.04, zc[1] + 0.435), "presyn.", COLORS["mute"],
           size=PT_SMALL, ha="left")
    # row 2: driving-force gauge
    gc = (zc[0] - 0.52, zc[1] - 0.06)
    t.ax.add_patch(Arc(gc, 0.30, 0.30, theta1=-20, theta2=200,
                       color=COLORS["mute"], lw=LW_EDGE, zorder=5))
    ang = np.deg2rad(52.0)
    t.ax.plot([gc[0], gc[0] + 0.15 * np.cos(ang)],
              [gc[1], gc[1] + 0.15 * np.sin(ang)], color=COLORS["ink"],
              lw=LW_EDGE, zorder=5, solid_capstyle="round")
    t.text((zc[0] - 0.30, zc[1] + 0.00), "Eᵢ − Vₙ", COLORS["ink"],
           ha="left")
    # row 3: input resistance (zigzag)
    zx, zy = zc[0] - 0.66, zc[1] - 0.50
    steps = [(0.05, 0.08), (0.08, -0.16), (0.08, 0.16), (0.08, -0.16),
             (0.05, 0.08)]
    xs, ys = [zx], [zy]
    for dx, dy in steps:
        xs.append(xs[-1] + dx)
        ys.append(ys[-1] + dy)
    t.ax.plot(xs, ys, color=COLORS["ink"], lw=LW_EDGE, zorder=5,
              solid_joinstyle="round")
    t.text((zc[0] - 0.28, zc[1] - 0.48), "Rₙᵗᵒᵗ",
           COLORS["ink"], ha="left")
    t.text((zc[0], zc[1] - 0.98), "local factors of eᵢ", COLORS["mute"],
           size=PT_SMALL, va="top")


_TRANSPORT_PATH = [(ROOT_PT, "J1", "A"), ("J1", "JL", "B"),
                   ("JL", "JLR", "C"), ("JLR", "T4", "D")]


def _mode_transport(t):
    t.tree(GHOST)
    t.junctions(edge=GHOST)
    add = COLORS["additive"]
    for a, b, lvl in _TRANSPORT_PATH:
        t.seg(a, b, add, _TAPER_PT[lvl], zorder=2.4)
        t.arrow(_lerp(a, b, 0.34), _lerp(a, b, 0.60), add, tikz_pt=0.8)
    t.junctions(names=("J1", "JL", "JLR"), edge=add)
    t.text((0.34, 0.66), "α₁", add, ha="left")
    t.text((-1.16, 1.42), "α₂", add, ha="right")
    t.text((-0.42, 2.10), "α₃", add, ha="left")
    t.soma(COLORS["soma"], RIM)
    t.text((-0.30, -0.04), "δ₀", add, ha="right")
    t.dot("T4", MS_JUNCTION, add)


def _mode_scalar(t):
    amber = COLORS["local"]
    t.tree(amber)
    t.junctions(edge=amber)
    t.soma(COLORS["soma"], RIM)
    t.text((-0.30, -0.04), "mᵤ", AMBER_TEXT, ha="right")
    for r in (0.36, 0.50, 0.64):
        t.ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=122, theta2=158,
                           color=mix("local", 80), lw=LW_HAIR, zorder=4))


def _mode_coordinate(t):
    add = COLORS["additive"]
    t.tree(GHOST)
    t.junctions(edge=GHOST)
    # the coordinate stops at the soma: dashed barrier across the trunk
    # (TikZ arc from (-0.42,0.44), 150 deg -> 30 deg, r = 0.485)
    cx = -0.42 - 0.485 * np.cos(np.deg2rad(150.0))
    cy = 0.44 - 0.485 * np.sin(np.deg2rad(150.0))
    arc = Arc((cx, cy), 0.97, 0.97, theta1=30, theta2=150,
              color=COLORS["mute"], lw=LW_HAIR, zorder=4)
    arc.set_linestyle((0, (1.5, 1.4)))
    t.ax.add_patch(arc)
    t.soma(COLORS["soma"], RIM, zorder=4.2)
    t.arrow((1.30, 0.60), (0.27, 0.09), add, tikz_pt=0.9, rad=0.12)
    t.text((1.36, 0.64), "δ₀,ᵤ", add, ha="left")


_CAPSULES_K2 = [
    (("shunting", 16), 13, [((0.02, 0.62), "JL", "JLL", "T1"),
                            ("JLL", "T2"), ("JL", "JLR", "T3"),
                            ("JLR", "T4")]),
    (("additive", 15), 13, [((0.10, 0.62), "JR", "JRL", "T5"),
                            ("JRL", "T6"), ("JR", "JRR", "T7"),
                            ("JRR", "T8")]),
]
_CAPSULES_K4 = [
    (("shunting", 16), 11, [(_lerp("JL", "JLL", 0.3), "JLL", "T1"),
                            ("JLL", "T2")]),
    (("additive", 15), 11, [(_lerp("JL", "JLR", 0.3), "JLR", "T3"),
                            ("JLR", "T4")]),
    (("local", 24), 11, [(_lerp("JR", "JRL", 0.3), "JRL", "T5"),
                         ("JRL", "T6")]),
    (("oracle", 16), 11, [(_lerp("JR", "JRR", 0.3), "JRR", "T7"),
                          ("JRR", "T8")]),
]
_CAPSULES_K8 = [
    (("shunting", 18), 8, [(_lerp("JLL", "T1", 0.25), "T1")]),
    (("additive", 16), 8, [(_lerp("JLL", "T2", 0.25), "T2")]),
    (("local", 26), 8, [(_lerp("JLR", "T3", 0.25), "T3")]),
    (("oracle", 18), 8, [(_lerp("JLR", "T4", 0.25), "T4")]),
    (("highlight", 28), 8, [(_lerp("JRL", "T5", 0.25), "T5")]),
    (("soma", 26), 8, [(_lerp("JRL", "T6", 0.25), "T6")]),
    (("exc", 16), 8, [(_lerp("JRR", "T7", 0.25), "T7")]),
    (("bp", 14), 8, [(_lerp("JRR", "T8", 0.25), "T8")]),
]
_ADDRESS_TAGS = {
    2: [((-1.62, 3.32), "δᵤ,₁", ("shunting", 70)),
        ((1.66, 3.22), "δᵤ,₂", ("additive", 70))],
    4: [((-2.12, 3.02), "δᵤ,₁", ("shunting", 70)),
        ((-0.60, 3.42), "δᵤ,₂", ("additive", 70)),
        ((0.64, 3.40), "δᵤ,₃", ("local", 62)),
        ((2.12, 2.86), "δᵤ,₄", ("oracle", 70))],
    8: [],
}


def _draw_capsules(t, spec):
    for (cname, pct), width_pt, chains in spec:
        t.capsule(mix(cname, pct), width_pt, chains)


def _mode_address(t, K):
    if K not in (2, 4, 8):
        raise ValueError(f"address mode needs K in {{2, 4, 8}}, got {K!r}")
    spec = {2: _CAPSULES_K2, 4: _CAPSULES_K4, 8: _CAPSULES_K8}[K]
    _draw_capsules(t, spec)
    t.tree(COLORS["dend"])
    t.junctions()
    t.soma(COLORS["soma"], RIM)
    for xy, tag, (cname, pct) in _ADDRESS_TAGS[K]:
        t.text(xy, tag, AMBER_TEXT if cname == "local" else COLORS[cname])


# gain-mode stroke widths (TikZ pt), keyed by edge
_GAIN_PT = {
    (ROOT_PT, "J1"): 1.70, ("J1", "JL"): 1.50, ("J1", "JR"): 0.80,
    ("JL", "JLL"): 1.45, ("JL", "JLR"): 0.90,
    ("JR", "JRL"): 0.70, ("JR", "JRR"): 0.45,
    ("JLL", "T1"): 1.15, ("JLL", "T2"): 1.15,
    ("JLR", "T3"): 0.70, ("JLR", "T4"): 0.70,
    ("JRL", "T5"): 0.55, ("JRL", "T6"): 0.55,
    ("JRR", "T7"): 0.35, ("JRR", "T8"): 0.35,
}


def _mode_gain(t):
    _draw_capsules(t, _CAPSULES_K4)
    for (a, b), w in _GAIN_PT.items():
        t.seg(a, b, COLORS["dend"], w)
    t.junctions()
    t.soma(COLORS["soma"], RIM)
    t.dot("JL", 9.5, "none", COLORS["ink"], LW_EDGE, zorder=4.4)
    t.text((-1.24, 1.32), "ᾶₙ = 1.6", COLORS["ink"],
           ha="right")


def _mode_shunt(t, shunted):
    dend = COLORS["dend"]
    syn = _lerp("J1", "JL", 0.86)
    if shunted:
        # ancestors + sibling subtree unchanged
        t.edges(EDGES_A, dend, _TAPER_PT["A"])
        t.edges(EDGES_B, dend, _TAPER_PT["B"])
        t.edges([("JR", "JRL"), ("JR", "JRR")], dend, _TAPER_PT["C"])
        t.edges([("JRL", "T5"), ("JRL", "T6"), ("JRR", "T7"),
                 ("JRR", "T8")], dend, _TAPER_PT["D"])
        # descendant subtree: thinned + faded (attenuated credit)
        faded = mix("dend", 38)
        t.edges([("JL", "JLL"), ("JL", "JLR")], faded, 0.50)
        t.edges([("JLL", "T1"), ("JLL", "T2"), ("JLR", "T3"),
                 ("JLR", "T4")], faded, 0.40)
        t.junctions(names=("J1", "JR", "JRL", "JRR", "JL"))
        t.junctions(names=("JLL", "JLR"), edge=mix("dend", 40))
        t.dot(syn, 4.2, COLORS["inh"])
    else:
        t.tree(dend)
        t.junctions()
        t.dot(syn, 4.2, "white", COLORS["inh"], LW_EDGE)
    t.soma(COLORS["soma"], RIM)
    t.text((syn[0] - 0.12, syn[1] - 0.13), "gₛₕᵤₙₜ",
           COLORS["inh"], size=PT_SMALL, ha="right", va="top")


_MODES = ("plain", "forward", "eligibility", "transport", "scalar",
          "coordinate", "address", "gain", "shunt")


def draw_credit_tree(ax, mode="forward", scale=1.0, *, labels=True, K=4,
                     shunted=True, stage_label=None, xlim=None, ylim=None,
                     autoscale=True):
    """Draw the shared credit tree on ``ax`` in one decoration ``mode``.

    Parameters mirror the TikZ ``\\CreditTree`` keys.  ``labels=False``
    strips every text artist (for tiny insets); ``scale`` tempers stroke
    widths and marker sizes only (geometry and fonts are fixed).
    ``xlim``/``ylim`` override the mode's default frame; ``autoscale=False``
    leaves the axes limits and appearance completely untouched.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {_MODES}")
    if autoscale:
        x = _BASE_XLIM if xlim is None else xlim
        y = _BASE_YLIM if ylim is None else ylim
        if xlim is None and mode == "eligibility":
            x = (_BASE_XLIM[0], 3.75)
        if ylim is None and mode == "address" and K in (2, 4):
            y = (_BASE_YLIM[0], 3.72)
        _setup_axes(ax, x, y)
    t = _Tree(ax, scale, labels)
    if mode == "plain":
        _mode_plain(t)
    elif mode == "forward":
        _mode_forward(t)
    elif mode == "eligibility":
        _mode_eligibility(t)
    elif mode == "transport":
        _mode_transport(t)
    elif mode == "scalar":
        _mode_scalar(t)
    elif mode == "coordinate":
        _mode_coordinate(t)
    elif mode == "address":
        _mode_address(t, int(K))
    elif mode == "gain":
        _mode_gain(t)
    elif mode == "shunt":
        _mode_shunt(t, bool(shunted))
    t.stage(stage_label)
    return ax


# ── DerangedPair: correct vs crossed coordinate assignment ────────────────

_MINI_EDGES = [((0.00, 0.11), (0.02, 0.52), 1.10),
               ((0.02, 0.52), (-0.40, 0.94), 0.80),
               ((0.02, 0.52), (0.44, 0.90), 0.80),
               ((-0.40, 0.94), (-0.72, 1.24), 0.55),
               ((-0.40, 0.94), (-0.16, 1.34), 0.55),
               ((0.44, 0.90), (0.22, 1.30), 0.55),
               ((0.44, 0.90), (0.76, 1.18), 0.55)]
_MINI_JUNCTIONS = [(0.02, 0.52), (-0.40, 0.94), (0.44, 0.90)]


def _mini_tree(t, sx, sy):
    for (x0, y0), (x1, y1), w in _MINI_EDGES:
        t.seg((sx + x0, sy + y0), (sx + x1, sy + y1), COLORS["dend"], w)
    for (jx, jy) in _MINI_JUNCTIONS:
        t.dot((sx + jx, sy + jy), 2.4, "white", COLORS["dend"], LW_EDGE)
    t.ax.add_patch(Circle((sx, sy), 0.13, facecolor=COLORS["soma"],
                          edgecolor=RIM, lw=LW_HAIR, zorder=3.5))


def draw_deranged_pair(ax, scale=1.0, *, labels=True, xlim=None, ylim=None,
                       autoscale=True):
    """Correct assignment (top, blue) vs deranged transport (bottom, gray)."""
    if autoscale:
        x = (-0.95, 4.35) if xlim is None else xlim
        y = (-1.25, 4.05) if ylim is None else ylim
        _setup_axes(ax, x, y)
    t = _Tree(ax, scale, labels)
    # The deranged map is a CONTROL, so it is drawn in the control
    # gray.  It previously used COLORS["bp"], the hue this figure
    # simultaneously declares to mean the exact backpropagated field.
    add, bp = COLORS["additive"], COLORS["mute"]
    # correct assignment (top)
    _mini_tree(t, 0.0, 2.55)
    _mini_tree(t, 2.30, 2.55)
    for x0 in (0.0, 2.30):
        t.dot((x0, 1.70), 4.0, add)
        t.arrow((x0, 1.78), (x0, 2.36), add, tikz_pt=0.8, head=4.0)
    # Set beside the feed dot, not under it: at va="top" from y=1.60 the
    # glyphs fell inside the 4 pt dot and were overprinted by the arrow.
    t.text((-0.22, 1.70), "δᵤ", add, size=PT_SMALL, ha="right", va="center")
    t.text((2.08, 1.70), "δᵥ", add, size=PT_SMALL, ha="right", va="center")
    t.text((3.20, 2.85), "correct", COLORS["additive"], ha="left")
    # deranged assignment (bottom)
    _mini_tree(t, 0.0, 0.0)
    _mini_tree(t, 2.30, 0.0)
    t.dot((0.0, -0.85), 4.0, bp)
    t.dot((2.30, -0.85), 4.0, bp)
    t.text((0.0, -0.95), "δᵤ", bp, size=PT_SMALL, va="top")
    t.text((2.30, -0.95), "δᵥ", bp, size=PT_SMALL, va="top")
    t.arrow((0.08, -0.78), (2.24, -0.18), bp, tikz_pt=0.8, head=4.0)
    t.arrow((2.22, -0.78), (0.06, -0.18), bp, tikz_pt=0.8, head=4.0)
    t.text((3.20, 0.30), "deranged", bp, ha="left")
    return ax


# ── regression sheet: every mode on one canvas ────────────────────────────

if __name__ == "__main__":
    from pathlib import Path

    from journal_style import (
        apply_neurips_style,
        audit_layout,
        audit_text_over_data,
        grid_figure,
    )

    apply_neurips_style()
    fig, axes = grid_figure(
        3, nrows=5, panel_h=1.72, gap_w=0.42, gap_h=0.40,
        margin_l=0.24, margin_r=0.24, margin_t=0.34, margin_b=0.22,
    )
    panels = [
        ("plain", dict(mode="plain", stage_label="hook variant")),
        ("forward", dict(mode="forward", stage_label="forward biophysics")),
        ("eligibility",
         dict(mode="eligibility", stage_label="eligibility eᵢ")),
        ("transport",
         dict(mode="transport", stage_label="adjoint transport")),
        ("scalar", dict(mode="scalar", stage_label="global scalar")),
        ("coordinate",
         dict(mode="coordinate", stage_label="neuron coordinate")),
        ("address K=2", dict(mode="address", K=2, stage_label="K = 2")),
        ("address K=4", dict(mode="address", K=4, stage_label="K = 4")),
        ("address K=8", dict(mode="address", K=8, stage_label="K = 8")),
        ("gain", dict(mode="gain", stage_label="route gain")),
        ("shunt off",
         dict(mode="shunt", shunted=False,
              stage_label="gₛₕᵤₙₜ off")),
        ("shunt on",
         dict(mode="shunt", shunted=True,
              stage_label="gₛₕᵤₙₜ on")),
        ("deranged pair", None),
        ("plain, scale 0.62 no labels",
         dict(mode="plain", scale=0.62, labels=False)),
        ("forward, no labels", dict(mode="forward", labels=False)),
    ]
    for ax, (title, kw) in zip(axes.ravel(), panels):
        if kw is None:
            draw_deranged_pair(ax)
        else:
            draw_credit_tree(ax, **kw)
        ax.set_title(title, fontsize=PT_ANNOT, color=COLORS["ink"], pad=3.0)

    out = Path(__file__).resolve().parents[1] / "figures" / "generated"
    out.mkdir(parents=True, exist_ok=True)
    name = "credit_tree_schematics_test"
    fig.canvas.draw()
    audit_layout(fig, name)
    audit_text_over_data(fig, name)
    fig.savefig(out / f"{name}.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(out / f"{name}.png", dpi=600)
    print(f"wrote {out / name}.png")

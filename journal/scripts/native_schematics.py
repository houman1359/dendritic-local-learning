#!/usr/bin/env python3
"""Main-figure schematics drawn onto a caller-supplied Axes.

Every schematic used by main Figures 2-9 lives here as
``draw_<name>(ax, *, labels=True, scale=1.0)``.  The drawing is laid out in
the axes' own 0-1 frame and always fills its cell, so a figure builder can
place a schematic in any slot of a native full-width canvas and get the same
stroke weights, type sizes and anatomy vocabulary as its data panels -- no
pre-rendered sub-block, no rescaling.

Vocabulary (shared with ``credit_tree_schematics``, the matplotlib port of
the talk's tree library, which draws the trees used here):

* soma       filled ``COLORS['soma']`` disc with a thin ink edge;
* dendrite   ``COLORS['dend']`` strokes tapering ``LW_DATA`` -> ``LW_EDGE``
             with round caps; junctions are small open circles;
* synapses   excitatory ``COLORS['exc']`` / inhibitory ``COLORS['inh']`` dots;
* arrows     ``-|>`` heads scaled to the line weight;
* scaffolding group frames, brackets and leaders in ``COLORS['mute']`` at
             ``LW_HAIR``..``LW_REF``, never competing with the anatomy;
* labels     sentence case, mathtext with real Greek and a true minus, placed
             in clear whitespace and tied back with a short mute leader.

``labels=False`` strips the prose (for a thumbnail-sized inset); ``scale``
tempers marker sizes and stroke weights only -- geometry always fills the
cell -- and every resulting weight is snapped back onto the journal token
set, so a schematic can never introduce a foreign line width.

The shared vocabulary is reused by the native numbered builders: feedback
resolution (Fig. 2), the credit operator (Fig. 3), routing tasks (Figs. 4--5),
physical depth (Fig. 6), anatomy (Fig. 7), focal shunting (Fig. 8) and the
measured-response boundary (Fig. 9).

Glyph library (journal figure overhaul)
---------------------------------------
``Frame`` also carries the shared glyph vocabulary every main-figure
schematic is drawn with, each sized in POINTS so it is identical in a 4-,
5-, 6- or 7-module slot and each dropping a label it cannot fit rather
than shrinking it: ``soma`` / ``junction`` / ``terminal`` / ``dendrite`` /
``contact`` / ``shunt`` primitives, the ``gate`` ring (closed gates fade
their descendants), the rule-agnostic ``error_in`` arrow, the four
``credit_delivery`` modes (scalar bus, per-neuron coordinate, subtree
capsules, exact path with α tags), ``balanced_tree`` / ``site_tree`` /
``partition`` trees returning :class:`Nodes`, ``dictionary_matrix`` /
``dictionary_product`` insets, ``task_card`` frames, ``badge`` tags,
``teacher_student`` line styles, ``rule_key`` and ``subscript`` (token
subscripts, never mathtext).  Module-level ``reference_line`` draws the
chance / zero reference, and ``draw_stage_pair``,
``draw_operator_tree_pair``, ``draw_local_gate_tree`` and
``draw_measured_boundary`` are the figure-specific compositions
(``COMPOSITIONS``).  ``scripts/native_schematics_gallery.py`` draws every
family once and must pass the strict audit.

2026-09-08 spec upgrade
-----------------------
``Frame._draw_chains`` -- the one place every capsule, partition band and
route ribbon is drawn -- now emits a FILLED tint patch
(:func:`journal_style.tint_patch`) instead of a 2.6-13 pt round-capped
stroke.  Same geometry, same colours, same call signature; what changes is
that an area is a fill, which is what the tightened stroke rule in
``figure_canvas`` (nothing above 1.35 pt unless it is a closed filled path)
now requires.  Glyph sizes that were tuned against the old 6.8-8.8 pt type
tokens inherit the new three-value scale (7.0 / 8.0 / 9.0) through the same
``PT_*`` names.

The same date, the SCHEMATIC LANGUAGE itself became enforceable.  One neuron
is drawn the same way in every figure, and the library now asserts it rather
than trusting the builder:

1. ORIENTATION.  Every tree helper fans UPWARD from a soma that is the lowest
   node of its own drawing.  ``site_tree`` keeps its ``orient`` argument but
   now defaults to ``'up'`` (it was ``'right'``), and :func:`require_soma_lowest`
   raises when a caller places a soma above a node of the same tree; the tree
   helpers call it on themselves.  A matrix whose rows must line up with sites
   no longer justifies a soma-at-the-left tree: draw the soma-lowest tree and
   put :meth:`Frame.site_strip` (``orient='vertical'``) -- the site order as a
   strip, with no soma to mis-orient -- beside the matrix.  ``orient='right'``
   still draws, for the Fig. 1 builder that has not migrated, but is recorded
   in ``ORIENTATION_LEGACY`` and is not soma-lowest.
2. DELTA-0.  Credit arrival is rule-agnostic, so every schematic states it.
   :meth:`Frame.soma` registers each soma it draws and :meth:`Frame.error_in`
   registers each arrow; :meth:`Frame.require_delta0` asserts that every
   non-ghost soma in the panel carries exactly one delta-0 arrow.  A card that
   genuinely has no soma (a stimulus or dictionary card) declares
   ``require_delta0(allow_no_delta0=True, reason=...)``; the reason is recorded
   in ``DELTA0_EXEMPTIONS`` and in the canvas manifest (``schematic_notes``),
   never assumed.  All four ``COMPOSITIONS`` assert it.
3. GATES AND SHUNTS are the inhibitory-contact family and nothing else: a
   CLOSED gate is a filled inh contact inside the gate ring, an OPEN gate is
   the same contact drawn white with an inh rim, and each carries a 7 pt badge
   ('c', ``('a', 'p', ' = 10')``, ``('g', 'shunt')``).  Attenuation is never a
   new mark -- ``closed=True`` fades the descendant subtree through
   :meth:`Frame.fade` (the same drawing at 38 % tint, hairline).  A red bar or
   a free red segment now RAISES from :meth:`Frame.dendrite` and
   :meth:`Frame.rule`.
4. DELIVERY.  Exactly four glyphs, aliased not extended: ``'scalar'`` (amber
   bus with its source dot OUTSIDE the trees, tagged s), ``'neuron'`` (the
   identical amber bus, confined to one tree, source dot AT that soma -- the
   arrow-plus-barrier-arc drawing is retired), ``'subtree'`` (16 % tint patch
   over the addressed subtree plus one arrow into its root) and ``'exact'``
   (alpha-tagged chain along the route).  ``DELIVERY_ALIASES`` maps the older
   names ('broadcast', 'layer', 'per_neuron', 'ancestry', 'address', 'path')
   onto them; anything else still raises.
5. TINTS.  Every area mark -- capsule, partition band, route ribbon, bar --
   is a :func:`journal_style.tint_patch` fill at 16 % with a 0.55 pt edge.
6. MATRICES.  :meth:`Frame.dictionary_matrix` and
   :meth:`Frame.dictionary_product` ASSERT at least ``MIN_CELL_PT`` (6.0) per
   row and per column and a column header no wider than 1.5 x its column, and
   offer the collapsed-band mode (``collapse``, with ``row_groups``) that
   prints "rows collapsed: N per band" instead of drawing sub-6-pt rows.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import (LinearSegmentedColormap, ListedColormap,
                                to_rgb, to_rgba)
from matplotlib.lines import Line2D
from matplotlib.patches import (Arc, Circle, Ellipse, FancyArrowPatch,
                                FancyBboxPatch)

import credit_tree_schematics as _ct
from credit_tree_schematics import AMBER_TEXT, GHOST, RIM, draw_credit_tree, mix
from figure_canvas import (enforce_tokens, snap_stroke_pt, style_panel,
                           tint_patch, token_subscript)
from journal_style import (
    COLORS,
    DIV_CMAP,
    K_CYCLE as ADDRESS_CYCLE,
    SERIES_COLORS,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    PT_ANNOT,
    PT_LABEL,
    PT_LEGEND,
    PT_SMALL,
    PT_TITLE,
    label_color,
    strengthen,
    tint_pct,
)

INK = COLORS["ink"]
MUTE = COLORS["mute"]
# 2026-09-08: the soma is a pale yellow in the new anatomy register, so its
# rim stepped from a 30 % grey (invisible against it) to a 62 % ink.
_SOMA_RIM = mix("ink", 62)
_SOMA_RIM_LEGACY = mix("ink", 30)
GRID = COLORS["grid"]
GREEN = COLORS["shunting"]
BLUE = COLORS["additive"]
PURPLE = COLORS["oracle"]
AMBER = COLORS["local"]
ROSE = COLORS["highlight"]
RED = COLORS["bp"]

# Aspect (w/h) of the shared credit-tree frame, so a tree inset can be given
# a rectangle it fills exactly instead of floating inside one.
_XL = getattr(_ct, "_BASE_XLIM", (-2.55, 2.55))
_YL = getattr(_ct, "_BASE_YLIM", (-0.72, 3.32))
TREE_ASPECT = (_XL[1] - _XL[0]) / (_YL[1] - _YL[0])

# Label bands are reserved in points, never as a fraction of the cell: a
# 8.4 pt title needs the same 13 pt strip whether its cell is 40 or 150 pt
# tall.  ``MIN_CORE_PT`` is the least room a tree or box diagram can occupy
# and still read; below it prose is dropped rather than overlaid.
TITLE_BAND_PT = 13.0
LINE_BAND_PT = 10.5
MIN_CORE_PT = 30.0

# Narrowest credit-tree inset that can still carry the tree library's own
# fixed-size per-subtree labels without them colliding (measured: they are
# clean at the 112 pt design width, collide by 47 pt).
TREE_LABEL_MIN_W_PT = 95.0


# ── shared glyph constants (the journal vocabulary, in points) ───────────
# Every glyph below is sized in POINTS so it is identical in a 4-, 5-, 6- or
# 7-module slot; a builder only chooses where (frame fractions) it goes.
FADED = mix("dend", 38)          # attenuated strokes below a closed gate/shunt
FADED_RING = mix("dend", 40)     # junction rims inside an attenuated subtree
TAPER_LW = (LW_DATA, LW_ERR, LW_REF, LW_EDGE)   # taper level 0 (trunk) .. 3
SOMA_R_PT = 3.0                  # soma disc radius
JUNCTION_R_PT = 1.6              # MS_JUNCTION 3.2 pt diameter
CONTACT_DIA_PT = 3.6             # MS_SYN
SITE_R_PT = (2.1, 1.65)          # proximal / distal site discs
GATE_R_PT = (1.9, 3.6)           # inner white ring / outer open ring
CAPSULE_PCT = 16                 # addressed-subtree capsule tint
CAPSULE_W_PT = (2.6, 13.0)       # capsule stroke width floor / ceiling
ALPHA_TAG_MIN_PITCH_PT = 8.5     # terminal pitch below which route tags drop
INPUT_LABEL_MIN_PITCH_PT = 11.0  # terminal pitch below which x-labels drop

# The addressed-subtree tint cycle is the one credit_tree_schematics already
# uses for its K = 2 / 4 / 8 capsules, so an address drawn here and one drawn
# by draw_credit_tree read as the same object.
K_CYCLE = {
    2: [spec[0] for spec in _ct._CAPSULES_K2],
    4: [spec[0] for spec in _ct._CAPSULES_K4],
    8: [spec[0] for spec in _ct._CAPSULES_K8],
}

# Full 8-terminal topology, read off the library's edge lists so the two
# drawings can never disagree about which junction owns which terminal.
_FULL_PARENT = {"J1": "S"}
for _a, _b in _ct.EDGES_B + _ct.EDGES_C + _ct.EDGES_D:
    _FULL_PARENT[_b] = _a
_FULL_AT_DEPTH = {1: ["JL", "JR"], 2: ["JLL", "JLR", "JRL", "JRR"],
                  3: [f"T{i}" for i in range(1, 9)]}

BADGE_STYLE = {
    # kind: (text colour key, face, edge)
    "oracle": ("oracle", mix("oracle", 8), mix("oracle", 45)),
    "local rule": ("shunting", mix("shunting", 8), mix("shunting", 45)),
    "exact": ("bp", "white", mix("bp", 45)),
    "control": ("mute", COLORS["panel_bg"], COLORS["grid"]),
    "teacher": ("ink", "white", mix("ink", 45)),
}


# ── 2026-09-08 glyph-language rules (assertions, not conventions) ────────
#: The four delivery glyphs, and only these four.
DELIVERY_MODES = ("scalar", "neuron", "subtree", "exact")

#: Older / prose names for the same four glyphs.  A builder that says
#: 'broadcast' means the layer scalar; saying it a fifth way does not make it
#: a fifth glyph.  Anything outside these two tables still raises.
DELIVERY_ALIASES = {
    "layer": "scalar", "broadcast": "scalar", "layer_scalar": "scalar",
    "shared": "scalar",
    "per_neuron": "neuron", "neuron_specific": "neuron",
    "per_soma": "neuron", "coordinate": "neuron",
    "ancestry": "subtree", "address": "subtree", "capsule": "subtree",
    "path": "exact", "route": "exact", "exact_path": "exact",
}

BADGE_PT = PT_SMALL          # every glyph badge is one 7 pt token
GHOST_PCT = 45               # background / neighbouring cell tint
FADE_PCT = 38                # attenuated (post-gate, post-shunt) subtree tint
MIN_CELL_PT = 6.0            # matrix rule: points per row and per column
HEADER_MAX_RATIO = 1.5       # matrix rule: header width / column width
SOMA_LOWEST_TOL_PT = 0.75    # a node this far below the soma is a mistake
DELTA0_TOL_PT = 6.0          # an error_in this far from a soma misses it
SOURCE_OUT_PT = 9.0          # how far a layer-scalar source sits outside

#: The sanctioned four-hue address cycle, by NAME (``journal_style.K_CYCLE``).
#: Kept as one object so an address drawn here and one drawn by the palette
#: module cannot disagree.  ``ADDRESS_TINT_STRICT`` turns the spec's third
#: glyph rule ("no subtree tint in a series hue") from a note into a raise; it
#: is off because the frozen cycle in ``journal_style`` is itself built from
#: series names, so switching it on today would fail the paper's own cycle.
ADDRESS_TINT_STRICT = False
ADDRESS_TINT_NOTES = []

#: Every declared delta-0 exemption, and every legacy soma-at-the-left tree,
#: recorded for the manifest and for the figure report.
DELTA0_EXEMPTIONS = []
ORIENTATION_LEGACY = []


class SchematicRuleError(ValueError):
    """A drawing broke one of the glyph rules (orientation, delta-0, family)."""


class MatrixTooDense(SchematicRuleError):
    """A matrix cell, or a column header, is below the legibility floor."""


def _frame_h_pt(ref):
    """Height in points of a Frame / Axes / (w_pt, h_pt) reference."""
    if ref is None:
        return None
    h = getattr(ref, "h_pt", None)
    if h:
        return float(h)
    ax = getattr(ref, "ax", ref)
    try:
        fig = ax.get_figure()
        return ax.get_position().height * fig.get_size_inches()[1] * 72.0
    except Exception:
        return None


def require_soma_lowest(nodes, rect=None, *, tol_pt=SOMA_LOWEST_TOL_PT,
                        name=None):
    """GLYPH RULE (a): the soma is the LOWEST node of its own tree.

    ``nodes`` is a :class:`Nodes` (or any mapping of name -> frame xy with a
    ``soma`` attribute); ``rect`` is the Frame, Axes or panel the tree was
    drawn in, used only to read the panel height so ``tol_pt`` means points.
    Raises :class:`SchematicRuleError` naming the offending nodes -- the five
    root-at-top trees the 2026-09-08 review found were all of this shape.
    Returns ``nodes`` so it can wrap a call.
    """
    soma = getattr(nodes, "soma", None)
    if soma is None:
        soma = nodes.get("S") if hasattr(nodes, "get") else None
    if soma is None:
        return nodes
    h_pt = _frame_h_pt(rect)
    tol = (tol_pt / h_pt) if h_pt else 1e-4
    below = [n for n, xy in nodes.items()
             if isinstance(xy, (tuple, list)) and len(xy) == 2
             and float(xy[1]) < float(soma[1]) - tol]
    if below:
        raise SchematicRuleError(
            f"{name or 'tree'}: the soma must be the lowest node in the panel "
            f"(2026-09-08 spec §5, glyph rules); "
            f"{len(below)} node(s) sit below it: {sorted(map(str, below))[:6]}. "
            "Draw the tree fanning upward (orient='up').")
    return nodes


def require_address_tint(cname, *, where="capsule", strict=None):
    """GLYPH RULE (c): a subtree tint is an ADDRESS, never a data series.

    Returns the colour name.  Records a note (or raises, when
    ``ADDRESS_TINT_STRICT``) if the tint is one of the frozen series hues and
    is not part of the sanctioned :data:`ADDRESS_CYCLE`.
    """
    strict = ADDRESS_TINT_STRICT if strict is None else bool(strict)
    if cname in ADDRESS_CYCLE:
        return cname
    if cname in SERIES_COLORS:
        note = (f"{where}: subtree tint {cname!r} is a data-series hue; the "
                f"address cycle is {tuple(ADDRESS_CYCLE)}")
        if strict:
            raise SchematicRuleError(note)
        if note not in ADDRESS_TINT_NOTES:
            ADDRESS_TINT_NOTES.append(note)
    return cname


def resolve_delivery_mode(mode):
    """One of :data:`DELIVERY_MODES`, resolving :data:`DELIVERY_ALIASES`."""
    key = str(mode).strip().lower().replace(" ", "_").replace("-", "_")
    if key in DELIVERY_MODES:
        return key
    if key in DELIVERY_ALIASES:
        return DELIVERY_ALIASES[key]
    raise ValueError(
        f"unknown delivery mode {mode!r}: the schematic language has exactly "
        f"four delivery glyphs {DELIVERY_MODES} "
        f"(aliases {sorted(DELIVERY_ALIASES)})")


def check_matrix_cells(w_pt, h_pt, n, k, *, where="dictionary_matrix",
                       min_cell_pt=MIN_CELL_PT, headers=None, ax=None,
                       header_pt=PT_SMALL):
    """MATRIX RULE: >= 6 pt per row and column, header <= 1.5 x its column.

    Raises :class:`MatrixTooDense` with the remedy in the message (collapse
    the rows into bands, group the headers, or move the matrix to a wider
    module span).  Returns ``(col_pt, row_pt)``.
    """
    col_pt = float(w_pt) / max(int(k), 1)
    row_pt = float(h_pt) / max(int(n), 1)
    if row_pt < min_cell_pt - 1e-6:
        raise MatrixTooDense(
            f"{where}: {n} rows in {h_pt:.1f} pt is {row_pt:.2f} pt per row, "
            f"below the {min_cell_pt:.0f} pt floor -- pass row_groups=[...] "
            "(collapse='auto' then prints 'rows collapsed: N per band') or "
            "give the matrix a taller rect")
    if col_pt < min_cell_pt - 1e-6:
        raise MatrixTooDense(
            f"{where}: {k} columns in {w_pt:.1f} pt is {col_pt:.2f} pt per "
            f"column, below the {min_cell_pt:.0f} pt floor -- group the "
            "columns under one header or move the matrix to a wider module "
            "span")
    if headers and ax is not None:
        for j, head in enumerate(headers):
            if not head:
                continue
            wide = _text_w_pt(ax, str(head), header_pt)
            if wide > HEADER_MAX_RATIO * col_pt + 1e-6:
                raise MatrixTooDense(
                    f"{where}: column header {head!r} is {wide:.1f} pt over a "
                    f"{col_pt:.1f} pt column (limit "
                    f"{HEADER_MAX_RATIO:g}x = {HEADER_MAX_RATIO * col_pt:.1f} "
                    "pt) -- shorten it, group the columns, or widen the slot")
    return col_pt, row_pt


def _collapse_rows(A, row_groups):
    """Band means of ``A`` (and the band sizes), for the collapsed matrix."""
    A = np.asarray(A, dtype=float)
    sizes = [int(g) for g in row_groups]
    out, start = [], 0
    for g in sizes:
        out.append(A[start:start + g].mean(axis=0))
        start += g
    if start < len(A):                       # trailing rows form one band
        sizes.append(len(A) - start)
        out.append(A[start:].mean(axis=0))
    return np.asarray(out), sizes


def collapsed_row_note(sizes):
    """The printed line for a collapsed matrix: 'rows collapsed: N per band'."""
    sizes = list(sizes)
    if len(set(sizes)) == 1:
        return f"rows collapsed: {sizes[0]} per band"
    return "rows collapsed: " + "/".join(str(g) for g in sizes) + " per band"


def _same_color(a, b):
    if a is None or b is None:
        return False
    try:
        return all(abs(u - v) < 1e-4 for u, v in zip(to_rgb(a), to_rgb(b)))
    except Exception:
        return False


def _forbid_inhibitory_bar(color, where):
    """GLYPH RULE (3): inhibition is a CONTACT, never a bar or a segment."""
    if _same_color(color, COLORS["inh"]):
        raise SchematicRuleError(
            f"{where}: a red bar / free red segment is not a glyph. "
            "Inhibition, gating and shunting are one contact family -- use "
            "Frame.contact(kind='inh'), Frame.shunt() or Frame.gate(), and "
            "show attenuation by fading the real subtree (Frame.fade).")


def _lerp(a, b, f):
    return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))


def _signed(value, m=1.0):
    """Printed cell value with a true minus and no trailing zeros."""
    v = float(value)
    if abs(v) < 1e-9:
        return "0"
    s = f"{v:+.2g}" if abs(v) < 10 else f"{v:+.0f}"
    return s.replace("-", "−")


def _split_label(label):
    """'δ0' -> ('δ', '0'); tuples pass through; longer strings stay plain."""
    if isinstance(label, (tuple, list)):
        return tuple(label)
    s = str(label)
    if len(s) == 2 and s[1].isalnum():
        return (s[0], s[1])
    return (s,)


class Nodes(dict):
    """Node name -> (x, y) of one drawn tree, with its topology and artists.

    Returned by :meth:`Frame.balanced_tree` (string keys ``'S'``, ``'J1'``,
    ``'JL'``, ..., ``'T1'``..``'T8'``) and :meth:`Frame.site_tree` (integer
    site indices).  The topology (``parent``, ``children``, ``level``) lets
    the delivery, gate and partition helpers find routes and subtrees, and
    the artist registries (``edges``, ``rings``) let a closed gate restyle
    the strokes it attenuates instead of painting over them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent = {}
        self.children = {}
        self.level = {}
        self.edges = {}
        self.rings = {}
        self.soma = None
        self.soma_r_pt = SOMA_R_PT
        self.terminals = []
        self.orient = "up"
        self.pitch_pt = 0.0
        self.rect = None
        self.rows = {}
        self.order = []
        # 2026-09-08: which screen axis the dictionary rows run along.  'y'
        # (a site strip, or the legacy soma-at-the-left tree) is the only one
        # a matrix can be aligned to; a soma-lowest tree spreads its sites
        # across 'x' and must be paired with a Frame.site_strip instead.
        self.row_axis = None
        self.kind = "tree"

    # -- topology ---------------------------------------------------------
    def root(self):
        for name, par in self.parent.items():
            if par not in self.parent:
                return par
        return "S" if "S" in self else None

    def subtree(self, node):
        """``node`` and every descendant, parents before children."""
        out, stack = [], [node]
        while stack:
            n = stack.pop(0)
            out.append(n)
            stack = list(self.children.get(n, [])) + stack
        return out

    def route(self, node):
        """Path of node names from the root to ``node``."""
        path = [node]
        while path[-1] in self.parent:
            path.append(self.parent[path[-1]])
        return path[::-1]

    def terminals_under(self, node):
        return [n for n in self.subtree(node) if not self.children.get(n)]

    def at_depth(self, K):
        """The K subtree roots that partition the terminals evenly."""
        if K <= 1:
            root = self.root()
            kids = self.children.get(root, [])
            return kids[:1] if len(kids) == 1 else [root]
        frontier = [self.root()]
        while frontier and len(frontier) < K:
            nxt = []
            for n in frontier:
                nxt += list(self.children.get(n, [])) or [n]
            if nxt == frontier:
                break
            frontier = nxt
        return frontier

    # -- orientation ------------------------------------------------------
    @property
    def along(self):
        """Unit vector (pt space) pointing from the soma into the canopy."""
        return np.array([0.0, 1.0]) if self.orient == "up" else np.array([1.0, 0.0])

    @property
    def side(self):
        """Unit vector (pt space) toward the side credit enters from."""
        return np.array([1.0, 0.0]) if self.orient == "up" else np.array([0.0, -1.0])


class Frame:
    """A 0-1 drawing frame over one Axes that knows its physical aspect.

    Fractions are used for layout (so the drawing always fills the cell) and
    the physical aspect is used for anything that must stay isometric: discs,
    tree insets and square nodes.
    """

    def __init__(self, ax, *, labels=True, scale=1.0):
        self.ax = ax
        self.labels = bool(labels)
        self.scale = float(np.clip(scale, 0.6, 1.4))
        fig = ax.get_figure()
        fw, fh = fig.get_size_inches()
        box = ax.get_position()
        self.w_pt = box.width * fw * 72.0
        self.h_pt = box.height * fh * 72.0
        self.aspect = self.w_pt / self.h_pt
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_axis_off()
        ax.set_facecolor("none")
        # 2026-09-08 glyph-rule registries: what this panel drew, so the rules
        # can be asserted instead of eyeballed.  ``_somata`` is every soma
        # glyph (ghosts flagged), ``_delta0`` every somatic-error arrow,
        # ``_trees`` every Nodes drawn here, ``_notes`` the manifest record.
        self._somata = []
        self._delta0 = []
        self._trees = []
        self._notes = getattr(ax, "_journal_schematic_notes", None) or []
        ax._journal_schematic_notes = self._notes

    # -- unit helpers -----------------------------------------------------
    def fx(self, pt):
        """Points -> x fraction."""
        return pt / self.w_pt

    def fy(self, pt):
        """Points -> y fraction."""
        return pt / self.h_pt

    def lw(self, width):
        return snap_stroke_pt(width * self.scale)

    def ms(self, size):
        return size * self.scale

    # -- glyph-rule bookkeeping (2026-09-08) ------------------------------
    def note(self, kind, **payload):
        """Record a schematic note on the Axes (picked up by the manifest)."""
        record = {"kind": str(kind), **payload}
        self._notes.append(record)
        return record

    def require_soma_lowest(self, *, name=None):
        """GLYPH RULE (a) for every tree drawn in this panel.

        Each registered tree is checked against its own soma (a panel may
        stack two cards, so trees are never compared with each other).
        """
        for i, nodes in enumerate(self._trees):
            require_soma_lowest(nodes, self, name=name or f"tree {i}")
        return self

    def require_delta0(self, *, allow_no_delta0=False, reason=None):
        """GLYPH RULE (b): exactly one delta-0 arrow enters every soma.

        The somatic error is the one glyph that is identical in every
        schematic, so the reader learns that credit ARRIVAL is rule-agnostic
        before learning that its spatial spread is not.  Called by every
        composition; the gallery test calls it for all of ``COMPOSITIONS``.

        A card that is a stimulus / dictionary / matrix card and has no soma
        of its own passes ``allow_no_delta0=True`` with a ``reason``; the
        reason is recorded (``DELTA0_EXEMPTIONS`` and the canvas manifest),
        which is what makes an exemption a decision rather than an omission.
        Returns the tally; raises :class:`SchematicRuleError` otherwise.
        """
        live = [rec for rec in self._somata
                if not rec["ghost"] and rec["delta0"]]
        tally = {"somata": len(live), "arrows": len(self._delta0)}
        if allow_no_delta0:
            text = "" if reason is None else str(reason).strip()
            if not text:
                raise SchematicRuleError(
                    "require_delta0(allow_no_delta0=True) needs a reason "
                    "string: an exemption is recorded in the manifest, never "
                    "assumed")
            record = {"reason": text, **tally}
            DELTA0_EXEMPTIONS.append(record)
            self.note("delta0-exemption", **record)
            return record
        if not self._delta0:
            raise SchematicRuleError(
                "no δ0 arrow in this schematic: every schematic shows the "
                "somatic error entering the soma (Frame.error_in). A card "
                "with no soma declares require_delta0(allow_no_delta0=True, "
                "reason='...').")
        for arrow in self._delta0:
            if arrow["soma"] is None:
                raise SchematicRuleError(
                    "a δ0 arrow does not land on a soma: error_in() must be "
                    "given the soma's own xy (nodes.soma)")
        counts = {}
        for arrow in self._delta0:
            counts[arrow["soma"]] = counts.get(arrow["soma"], 0) + 1
        for idx, rec in enumerate(self._somata):
            n = counts.get(idx, 0)
            if rec["ghost"] or not rec["delta0"]:
                continue
            if n != 1:
                raise SchematicRuleError(
                    f"soma {idx} carries {n} δ0 arrows, not exactly one "
                    "(ghost / background somata are exempt: draw them with "
                    "ghost=True, or soma(..., delta0=False) for a swatch)")
        tally["exempt"] = len(self._somata) - len(live)
        return tally

    def _register_soma(self, xy, *, ghost=False, delta0=True):
        self._somata.append({"xy": (float(xy[0]), float(xy[1])),
                             "ghost": bool(ghost), "delta0": bool(delta0)})
        return len(self._somata) - 1

    def _nearest_soma(self, xy, *, tol_pt=DELTA0_TOL_PT):
        best, best_d = None, tol_pt
        target = self._to_pt(xy)
        for i, rec in enumerate(self._somata):
            d = float(np.linalg.norm(self._to_pt(rec["xy"]) - target))
            if d <= best_d:
                best, best_d = i, d
        return best

    # -- layout -----------------------------------------------------------
    def split(self, n, *, axis="auto", gap_pt=10.0, pad_pt=(0, 0, 0, 0)):
        """``n`` equal cells filling the frame, as (x0, y0, w, h) fractions.

        ``axis='auto'`` lays the cells in a row while each stays wider than
        it is tall, and stacks them otherwise, so one schematic definition
        survives both a full-width band and a one-third slot.
        """
        left, right, top, bottom = [float(v) for v in pad_pt]
        x0 = self.fx(left)
        y0 = self.fy(bottom)
        w = 1.0 - self.fx(left + right)
        h = 1.0 - self.fy(top + bottom)
        if axis == "auto":
            axis = "x" if (self.aspect * w / max(n, 1)) / h >= 0.62 else "y"
        cells = []
        if axis == "x":
            gap = self.fx(gap_pt)
            cw = (w - (n - 1) * gap) / n
            for i in range(n):
                cells.append((x0 + i * (cw + gap), y0, cw, h))
        else:
            gap = self.fy(gap_pt)
            ch = (h - (n - 1) * gap) / n
            for i in range(n):
                cells.append((x0, y0 + (n - 1 - i) * (ch + gap), w, ch))
        return cells

    @staticmethod
    def inset(rect, *, left=0.0, right=0.0, top=0.0, bottom=0.0):
        x0, y0, w, h = rect
        return (x0 + w * left, y0 + h * bottom,
                w * (1.0 - left - right), h * (1.0 - top - bottom))

    def cell_text(self, rect, *, title=None, title_color=None, subtitle=None,
                  foot=None, foot_color=None, min_core_pt=MIN_CORE_PT,
                  title_size=PT_LABEL, sub_size=PT_ANNOT, foot_size=PT_SMALL):
        """Set a cell's prose in point-high bands; return the drawing core.

        A label band is a fixed number of *points* because type is a fixed
        number of points, but the drawing inside a cell is naturally sized as
        a *fraction* of it.  Mixing the two is what makes a schematic that is
        perfect at its design height pile its own captions onto its own
        strokes when a builder drops it into a shorter slot.  Reserving the
        bands here in points keeps the drawing clear at every cell shape, and
        when the cell is genuinely too short the least load-bearing line is
        dropped -- subtitle, then caption, then title -- so a small slot
        loses words instead of becoming unreadable.
        """
        if not self.labels:
            title = subtitle = foot = None
        h_pt = rect[3] * self.h_pt
        w_pt = rect[2] * self.w_pt - 4.0        # keep off the group frame
        # Fit each line to the cell width first: a label that cannot be made
        # to fit is dropped here, so it can never run out of its own cell.
        title = _wrap_to_width(self.ax, title, title_size, w_pt) if title \
            else None
        subtitle = _wrap_to_width(self.ax, subtitle, sub_size, w_pt) \
            if subtitle else None
        foot = _wrap_to_width(self.ax, foot, foot_size, w_pt) if foot else None
        items = []
        if title:
            items.append(["title", title,
                          TITLE_BAND_PT + LINE_BAND_PT * title.count("\n")])
        if subtitle:
            items.append(["sub", subtitle,
                          LINE_BAND_PT * (1 + subtitle.count("\n"))])
        if foot:
            items.append(["foot", foot,
                          LINE_BAND_PT * (1 + foot.count("\n"))])
        while items and h_pt - sum(i[2] for i in items) < min_core_pt:
            for kind in ("sub", "foot", "title"):
                hit = [n for n, item in enumerate(items) if item[0] == kind]
                if hit:
                    items.pop(hit[0])
                    break
            else:
                break
        keep = {item[0]: item[2] for item in items}
        cx = rect[0] + rect[2] / 2.0
        y = rect[1] + rect[3]
        if "title" in keep:
            self.text((cx, y - self.fy(keep["title"] * 0.5)), title,
                      size=title_size, color=title_color,
                      linespacing=1.15)
            y -= self.fy(keep["title"])
        if "sub" in keep:
            self.text((cx, y - self.fy(keep["sub"] * 0.5)), subtitle,
                      size=sub_size, linespacing=1.15)
        if "foot" in keep:
            self.text((cx, rect[1] + self.fy(keep["foot"] * 0.5)), foot,
                      size=foot_size, linespacing=1.15,
                      color=MUTE if foot_color is None else foot_color)
        top_pt = keep.get("title", 0.0) + keep.get("sub", 0.0)
        bot_pt = keep.get("foot", 0.0)
        return (rect[0], rect[1] + self.fy(bot_pt), rect[2],
                rect[3] - self.fy(top_pt + bot_pt))

    def room(self, rect, pt):
        """Whether ``rect`` is at least ``pt`` points tall."""
        return rect[3] * self.h_pt >= pt

    def footer(self, s, *, size=PT_SMALL, band_pt=12.0, min_frame_pt=95.0,
               color=None):
        """One frame-wide summary line under everything; returns its points.

        Returns 0.0 (and draws nothing) when the frame is too short to give
        the line its own strip, so the caller's cells simply keep the space.
        """
        if not self.labels or not s or self.h_pt < min_frame_pt:
            return 0.0
        s = _wrap_to_width(self.ax, s, size, self.w_pt - 4.0)
        if s is None:
            return 0.0
        band_pt *= 1 + s.count("\n")
        self.text((0.5, self.fy(band_pt * 0.45)), s, size=size,
                  linespacing=1.15, color=INK if color is None else color)
        return band_pt

    # -- marks ------------------------------------------------------------
    def group(self, rect, *, tint=None, edge=None, lw=None, radius_pt=3.0,
              zorder=0.4):
        """Scaffolding frame around one conceptual group."""
        x0, y0, w, h = rect
        patch = FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle=f"round,pad=0,rounding_size={self.fy(radius_pt)}",
            facecolor="none" if tint is None else tint,
            edgecolor=MUTE if edge is None else edge,
            linewidth=LW_HAIR if lw is None else lw,
            zorder=zorder, transform=self.ax.transData, clip_on=False,
        )
        self.ax.add_patch(patch)
        return patch

    def text(self, xy, s, *, size=PT_SMALL, color=None, ha="center",
             va="center", zorder=6, force=False, **kw):
        if not (self.labels or force):
            return None
        return self.ax.text(xy[0], xy[1], s, fontsize=size,
                            color=INK if color is None else color,
                            ha=ha, va=va, zorder=zorder, **kw)

    def disc(self, xy, r_pt, *, fill, edge="none", lw=LW_EDGE, zorder=4):
        patch = Ellipse(xy, 2 * self.fx(r_pt * self.scale),
                        2 * self.fy(r_pt * self.scale), facecolor=fill,
                        edgecolor=edge, linewidth=self.lw(lw) if edge != "none"
                        else 0.0, zorder=zorder)
        self.ax.add_patch(patch)
        return patch

    def arrow(self, p0, p1, *, color=None, lw=LW_EDGE, head=5.0, rad=0.0,
              zorder=5):
        width = self.lw(lw)
        arr = FancyArrowPatch(
            p0, p1,
            arrowstyle=f"-|>,head_length={head * width / LW_EDGE * 0.8:.2f},"
                       f"head_width={0.6 * head * width / LW_EDGE * 0.8:.2f}",
            mutation_scale=1.0, connectionstyle=f"arc3,rad={rad}",
            color=MUTE if color is None else color, linewidth=width,
            shrinkA=0, shrinkB=0, capstyle="round", zorder=zorder,
        )
        self.ax.add_patch(arr)
        return arr

    def leader(self, p0, p1, *, color=None, lw=LW_HAIR):
        self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]],
                     color=MUTE if color is None else color, lw=self.lw(lw),
                     solid_capstyle="round", zorder=1.5)

    def rule(self, y, x0, x1, *, color=None, lw=LW_HAIR, dashed=False):
        _forbid_inhibitory_bar(color, "Frame.rule")
        line, = self.ax.plot([x0, x1], [y, y], color=MUTE if color is None
                             else color, lw=self.lw(lw), zorder=1.2,
                             solid_capstyle="round")
        if dashed:
            line.set_dashes((2.2, 1.8))
        return line

    # -- anatomy ----------------------------------------------------------
    def tree(self, rect, *, mode="plain", K=4, shunted=True, labels=None,
             tree_scale=1.0):
        """Credit tree filling ``rect`` at the library's own proportions."""
        x0, y0, w, h = rect
        w_pt, h_pt = w * self.w_pt, h * self.h_pt
        if w_pt / h_pt > TREE_ASPECT:            # height-limited
            fit_h_pt, fit_w_pt = h_pt, h_pt * TREE_ASPECT
        else:                                     # width-limited
            fit_w_pt, fit_h_pt = w_pt, w_pt / TREE_ASPECT
        fit = (x0 + (w - self.fx(fit_w_pt)) / 2.0,
               y0 + (h - self.fy(fit_h_pt)) / 2.0,
               self.fx(fit_w_pt), self.fy(fit_h_pt))
        sub = self.ax.inset_axes(fit, transform=self.ax.transData, zorder=3)
        sub.set_facecolor("none")
        want = self.labels if labels is None else labels
        # The tree library's own labels are a fixed type size on a tree whose
        # geometry shrinks with the inset, so below this width its per-subtree
        # annotations land on each other.  Draw the tree unlabelled instead.
        if want and fit_w_pt < TREE_LABEL_MIN_W_PT:
            want = False
        draw_credit_tree(
            sub, mode=mode, K=K, shunted=shunted, scale=tree_scale,
            labels=want,
        )
        enforce_tokens(sub)
        return sub

    def stage_height(self, avail, slot_w):
        """Tallest a stage tree may be before it collides with its neighbour.

        ``stage_tree`` spreads horizontally in proportion to its height, so a
        tree given all the room a tall cell offers grows wider than the share
        of the row it owns.  Cap it against the slot width instead.
        """
        cap = self.fy(0.336 * slot_w * self.w_pt)
        return max(0.0, min(avail, cap))

    def stage_tree(self, xy, height, depth, *, color=None, root_r_pt=2.4,
                   branching=None, width=None, rings=False):
        """Small balanced serial tree: one physical processing stage.

        The default (``branching=None``) is the original binary stage tree.
        ``branching`` -- a list of fan-outs per stage such as ``[8]`` or
        ``[2, 1, 2]`` -- draws the journal's physical-stage tree instead:
        the shared soma glyph at ``xy``, ``depth`` ignored in favour of
        ``len(branching)``, compartments spread over ``width`` (frame
        fraction; default 60 % of the tree height in points) and drawn as
        open junction rings when ``rings`` is true.  In that mode the
        per-stage node coordinates are returned as a list of lists so
        contacts and tags can be attached to them.
        """
        if branching is not None:
            return self._stage_tree_branching(xy, height, list(branching),
                                              color=color, width=width,
                                              rings=rings)
        color = GREEN if color is None else color
        cx, base = xy
        # Pale shared rim, as every other soma in the figure system uses;
        # the near-black INK rim made stage trees read as a different object
        # from their sibling cards.
        self.disc((cx, base), root_r_pt, fill=COLORS["soma"],
                  edge=_SOMA_RIM, lw=LW_EDGE, zorder=4)
        current = [(cx, base + height * 0.06)]
        span = self.fx(0.30 * self.h_pt * height)
        step = height * 0.94 / max(depth, 1)
        for level in range(depth):
            nxt = []
            span_l = span / (level + 1.15)
            width = max(LW_EDGE, LW_DATA - 0.18 * level)
            for px, py in current:
                for sign in (-1, 1):
                    qx, qy = px + sign * span_l, py + step
                    self.ax.plot([px, qx], [py, qy], color=color,
                                 lw=self.lw(width), solid_capstyle="round",
                                 zorder=2)
                    nxt.append((qx, qy))
            current = nxt
            span *= 0.62
        for px, py in current:
            self.disc((px, py), 1.15, fill=color, zorder=3)

    def _stage_tree_branching(self, xy, height, stages, *, color=None,
                              width=None, rings=True):
        color = COLORS["dend"] if color is None else color
        cx, base = xy
        self.soma((cx, base), r_pt=SOMA_R_PT)
        n_stage = max(len(stages), 1)
        step = height / n_stage
        spread = (self.fx(0.60 * height * self.h_pt) if width is None
                  else float(width))
        current = [cx]
        y = base
        out = []
        for s, b in enumerate(stages):
            b = max(int(b), 1)
            y_next = y + step
            level = 3 if s == n_stage - 1 else min(s, 2)
            per = spread / max(len(current), 1)
            nxt = []
            for px in current:
                offs = (np.linspace(-per / 2.0 + per / (2.0 * b),
                                    per / 2.0 - per / (2.0 * b), b)
                        if b > 1 else [0.0])
                for o in offs:
                    self.dendrite((px, y), (px + o, y_next), level=level,
                                  color=color)
                    nxt.append((px + o, y_next))
            out.append(nxt)
            current = [p[0] for p in nxt]
            y = y_next
        for stage in out:
            for px, py in stage:
                if rings:
                    self.junction((px, py))
                else:
                    self.disc((px, py), 1.15, fill=color, zorder=3)
        return out

    # ══ shared glyph vocabulary (journal figure system) ═══════════════════
    # Every helper below sizes itself in points and drops a label that does
    # not fit rather than shrinking it, so one definition survives 4-, 5-,
    # 6- and 7-module slots.  Colours are COLORS keys only; strokes LW
    # tokens only; type PT tokens only, subscripts through token_subscript.

    # -- unit helpers -----------------------------------------------------
    def _to_pt(self, xy):
        return np.array([xy[0] * self.w_pt, xy[1] * self.h_pt], dtype=float)

    def _from_pt(self, XY):
        return (float(XY[0]) / self.w_pt, float(XY[1]) / self.h_pt)

    def _off(self, xy, dx_pt=0.0, dy_pt=0.0):
        return (xy[0] + self.fx(dx_pt), xy[1] + self.fy(dy_pt))

    def _fits(self, s, size, width_pt):
        return _text_w_pt(self.ax, s, size) <= width_pt

    def _label(self, xy, label, *, color, size=PT_SMALL, ha="left",
               va="center", zorder=6):
        """Plain or subscripted tag: str, 'δ0'-style pair, or (base, sub)."""
        if not self.labels or label is None:
            return None
        parts = _split_label(label)
        if len(parts) >= 2:
            return self.subscript(xy, parts[0], parts[1],
                                  parts[2] if len(parts) > 2 else "",
                                  size=size, color=color, ha=ha, va=va,
                                  zorder=zorder)
        return self.text(xy, parts[0], size=size, color=color, ha=ha, va=va,
                         zorder=zorder)

    # -- type -------------------------------------------------------------
    def subscript(self, xy, base, sub, tail="", *, size=PT_ANNOT, color=None,
                  ha="center", va="center", zorder=6, drop_pt=1.6):
        """Token-sized ``base``+subscript+``tail`` anchored at ``xy``.

        Delegates the base/subscript pair to :func:`figure_canvas.token_subscript`
        (base at ``size``, subscript at PT_SMALL) and adds centred / right
        anchoring by measuring the chain, plus a tail set back on the base
        line rather than on the subscript's.  PT_SMALL is the type floor, so
        a PT_SMALL base steps up to PT_ANNOT to keep the subscript smaller
        than its base -- the same treatment the Fig. 2 builder uses.
        """
        if not self.labels:
            return None
        color = INK if color is None else color
        if size <= PT_SMALL:
            size = PT_ANNOT
        sub_size = PT_SMALL
        w_base = _text_w_pt(self.ax, base, size)
        w_sub = _text_w_pt(self.ax, sub, sub_size)
        w_tail = _text_w_pt(self.ax, tail, size) if tail else 0.0
        total = w_base + 0.4 + w_sub + ((0.6 + w_tail) if tail else 0.0)
        x, y = xy
        if ha == "center":
            x -= self.fx(total / 2.0)
        elif ha == "right":
            x -= self.fx(total)
        base_text = token_subscript(self.ax, x, y, base, sub, size=size,
                                    sub_size=sub_size, color=color, ha="left",
                                    va=va, drop_pt=drop_pt, zorder=zorder,
                                    clip_on=False)
        if tail:
            # x from the subscript's box, y from the BASE box: the tail sits
            # on the base line instead of inheriting the subscript's drop.
            sub_text = [t for t in self.ax.texts
                        if getattr(t, "xycoords", None) is base_text]
            anchor = sub_text[-1] if sub_text else base_text
            self.ax.annotate(tail, xy=(1.0, 0.0), xycoords=(anchor, base_text),
                             xytext=(0.6, 0.0), textcoords="offset points",
                             fontsize=size, color=color, ha="left",
                             va="bottom", zorder=zorder, annotation_clip=False)
        return base_text

    # -- anatomy primitives -------------------------------------------------
    def soma(self, xy, *, r_pt=SOMA_R_PT, output=None, label=None, zorder=4,
             ghost=False, delta0=True):
        """The one soma glyph: soma fill, ink rim at LW_EDGE.

        ``output`` (True or a length in points) adds the mute forward arrow
        leaving to the right, with ``label`` (e.g. 'z') at PT_SMALL.

        2026-09-08: the disc is registered with the frame, so
        :meth:`require_delta0` can assert that exactly one somatic-error
        arrow enters it.  ``ghost=True`` draws a background / neighbouring
        cell's soma at the 45 % tint the rest of a ghost tree uses (it was
        drawn at full strength, which made a ghost neighbour compete with the
        cell in front of it) and exempts it from the δ0 rule;
        ``delta0=False`` exempts a soma drawn as a glyph SWATCH.  The rim
        moved from ``mix('ink', 30)`` to ``_SOMA_RIM``: the anatomy register
        moved the soma to a pale yellow, and a 30 % grey rim disappeared
        around it.
        """
        fill = mix("soma", GHOST_PCT) if ghost else COLORS["soma"]
        patch = self.disc(xy, r_pt, fill=fill,
                          edge=GHOST if ghost else _SOMA_RIM, lw=LW_EDGE,
                          zorder=zorder)
        self._register_soma(xy, ghost=ghost, delta0=delta0)
        if output:
            length = 11.0 if output is True else float(output)
            x0 = xy[0] + self.fx(r_pt * self.scale + 1.5)
            self.arrow((x0, xy[1]), (x0 + self.fx(length), xy[1]), color=MUTE,
                       lw=LW_EDGE, head=3.4, zorder=zorder + 0.5)
            if label:
                self.text((x0 + self.fx(length + 2.5), xy[1]), label,
                          size=PT_SMALL, color=INK, ha="left")
        return patch

    def junction(self, xy, *, r_pt=JUNCTION_R_PT, site_color=None, ghost=False,
                 zorder=3):
        """Open white ring with dend edge; a filled site disc when coloured."""
        if site_color is not None:
            return self.disc(xy, r_pt, fill=site_color, zorder=zorder)
        return self.disc(xy, r_pt, fill="white",
                         edge=GHOST if ghost else COLORS["dend"], lw=LW_EDGE,
                         zorder=zorder)

    def terminal(self, xy, *, site_color=None, r_pt=SITE_R_PT[1], zorder=3):
        """A terminal is the bare stroke end; only a site tree marks it."""
        if site_color is None:
            return None
        return self.disc(xy, r_pt, fill=site_color, zorder=zorder)

    def dendrite(self, p0, p1, *, level=0, color=None, ghost=False,
                 faded=False, zorder=2):
        """One tapered branch segment, level 0 (trunk) .. 3 (terminal)."""
        level = int(min(max(level, 0), 3))
        width = TAPER_LW[level]
        _forbid_inhibitory_bar(color, "Frame.dendrite")
        if faded:
            color, width = FADED, LW_HAIR
        elif ghost:
            color = GHOST
        elif color is None:
            color = COLORS["dend"]
        line, = self.ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=color,
                             lw=self.lw(width), solid_capstyle="round",
                             zorder=zorder)
        return line

    def contact(self, xy, *, kind="exc", active=True, label=None,
                dia_pt=CONTACT_DIA_PT, zorder=4.5):
        """Excitatory / inhibitory contact dot (inactive inh = white, rim).

        ``label`` may be a string or a (base, sub) pair such as
        ``('g', 'shunt')`` for the focal-shunt tag.
        """
        color = COLORS["exc"] if kind == "exc" else COLORS["inh"]
        if active:
            mfc, mec, mew = color, "none", 0.0
        else:
            mfc, mec, mew = "white", color, self.lw(LW_EDGE)
        line, = self.ax.plot([xy[0]], [xy[1]], marker="o", ms=self.ms(dia_pt),
                             mfc=mfc, mec=mec, mew=mew, ls="none",
                             zorder=zorder)
        if label:
            self._label(self._off(xy, dia_pt * 0.5 + 2.2, 0.0), label,
                        color=color, size=PT_SMALL, ha="left")
        return line

    def shunt(self, xy, *, active=True, label=("g", "shunt"), badge=None):
        """Focal shunt: the inhibitory contact with its 7 pt g_shunt badge.

        Same family as :meth:`contact` and :meth:`gate` (glyph rule 3): a
        filled inh contact when active, the white-with-inh-rim contact when
        not.  ``badge`` is an alias for ``label`` so gate and shunt read the
        same way at the call site.
        """
        return self.contact(xy, kind="inh", active=active,
                            label=badge if badge is not None else label)

    def gate(self, xy, *, closed=False, badge="c", descendants=None,
             nodes=None, node=None, incident=None, badge_offset=None,
             zorder=4.2, badge_size=BADGE_PT):
        """Context / gate: the inhibitory CONTACT inside a ring, plus a badge.

        2026-09-08 glyph rule (3).  A gate is a member of the
        inhibitory-contact family and is drawn exactly like one: a CLOSED
        gate (inhibition on) is the filled inh contact inside the gate ring,
        an OPEN gate is the same contact drawn white with an inh rim -- the
        contact's own inactive form.  Before this both states drew the same
        white disc, so an open and a closed gate were indistinguishable
        except by whatever the caller did to the subtree.  It is never a bar
        and never a free red segment.

        ``badge`` is the 7 pt tag: a string ('c'), or a (base, sub[, tail])
        pair such as ``('a', 'p', ' = 10')`` or ``('g', 'shunt')``.  It sits
        in the free corner of the junction: upper-left by default, or -- when
        ``node`` (with ``nodes``) or ``incident`` stroke directions are given
        -- the diagonal farthest from every stroke meeting there, so it never
        lands on a child branch.  ``badge_offset`` (dx, dy in points) forces a
        position.

        ``closed=True`` attenuates ``descendants``: node names (with
        ``nodes``), ``(p0, p1)`` segments, or Line2D artists -- restyled to
        FADED at LW_HAIR so the partition is visible on the tree itself,
        which is the ONLY way attenuation is ever shown.  With ``nodes`` and
        ``node`` given, ``descendants`` now defaults to that node's own
        subtree instead of silently drawing a closed gate over a live tree.
        """
        col = COLORS["gate"]
        self.disc(xy, GATE_R_PT[0], fill=col if closed else "white",
                  edge="none" if closed else col, lw=LW_EDGE, zorder=zorder)
        self.disc(xy, GATE_R_PT[1], fill="none", edge=col, lw=LW_EDGE,
                  zorder=zorder)
        if closed and descendants is None and nodes is not None \
                and node is not None:
            descendants = [node]
        if badge:
            dirs = []
            if node is not None and nodes is not None:
                here = self._to_pt(nodes[node])
                for other in list(nodes.children.get(node, [])) + \
                        [nodes.parent.get(node)]:
                    if other is None:
                        continue
                    there = nodes.soma if other == "S" and other not in nodes \
                        else nodes[other]
                    dirs.append(self._to_pt(there) - here)
            elif incident:
                dirs = [np.array([dx * self.w_pt, dy * self.h_pt])
                        for dx, dy in incident]
            if badge_offset is not None:
                ox, oy = badge_offset
                ha, va = ("right" if ox < 0 else "left",
                          "bottom" if oy >= 0 else "top")
            else:
                (ox, oy), ha, va = self._free_corner(dirs)
            self._label(self._off(xy, ox, oy), badge, size=badge_size,
                        color=col, ha=ha, va=va, zorder=6)
        if closed and descendants:
            self.fade(descendants, nodes=nodes)

    @staticmethod
    def _free_corner(dirs, *, dx=4.6, dy=3.4):
        """The diagonal (dx, dy) farthest from the given stroke directions."""
        cands = (((-dx, dy), "right", "bottom"), ((dx, dy), "left", "bottom"),
                 ((-dx, -dy), "right", "top"), ((dx, -dy), "left", "top"))
        angles = [np.arctan2(d[1], d[0]) for d in dirs
                  if np.hypot(d[0], d[1]) > 1e-9]
        if not angles:
            return cands[0]
        best = None
        for (ox, oy), ha, va in cands:
            a = np.arctan2(oy, ox)
            gap = min(abs((a - b + np.pi) % (2 * np.pi) - np.pi)
                      for b in angles)
            if best is None or gap > best[0] + 1e-9:
                best = (gap, (ox, oy), ha, va)
        return best[1], best[2], best[3]

    def fade(self, items, *, nodes=None):
        """Attenuate strokes: node names (needs ``nodes``), segments, lines."""
        from matplotlib.lines import Line2D as _L
        lines, rings = [], []
        for item in items:
            if isinstance(item, _L):
                lines.append(item)
            elif isinstance(item, str) and nodes is not None:
                for n in nodes.subtree(item):
                    for kid in nodes.children.get(n, []):
                        art = nodes.edges.get((n, kid))
                        if art is not None:
                            lines.append(art)
                        if kid in nodes.rings:
                            rings.append(nodes.rings[kid])
            else:
                p0, p1 = item
                hit = [ln for ln in self.ax.lines
                       if len(ln.get_xdata()) == 2
                       and np.allclose(ln.get_xydata(), [p0, p1], atol=1e-9)]
                if hit:
                    lines += hit
                else:
                    self.dendrite(p0, p1, faded=True, zorder=2.6)
        for ln in lines:
            ln.set_color(FADED)
            ln.set_linewidth(self.lw(LW_HAIR))
        for ring in rings:
            ring.set_edgecolor(FADED_RING)

    def error_in(self, soma_xy, *, label="δ0", side="right", color=None,
                 r_pt=SOMA_R_PT, dashed=False, zorder=5):
        """The rule-agnostic ink arrow bringing the somatic error into the soma.

        Identical in every figure: LW_EDGE, head 4.5, PT_ANNOT tag with a
        token subscript.  ``side`` 'right' enters from the lower right (the
        forward output arrow keeps the right side at soma height), 'below'
        enters straight up, 'left' mirrors 'right'.  ``dashed`` marks an
        imposed field that is not observed.
        """
        color = COLORS["credit_ink"] if color is None else color
        x, y = soma_xy
        r = r_pt * self.scale
        if side == "below":
            tip = (x, y - self.fy(r + 1.2))
            tail = (x, y - self.fy(r + 12.0))
            tag_xy, ha, va, rad = self._off(tail, 2.2, 0.0), "left", "center", 0.0
        else:
            sgn = -1.0 if side == "left" else 1.0
            tip = (x + sgn * self.fx(0.72 * r + 0.9), y - self.fy(0.72 * r + 0.9))
            tail = (x + sgn * self.fx(r + 11.0), y - self.fy(r + 7.0))
            tag_xy = self._off(tail, sgn * 1.8, 0.0)
            ha, va, rad = ("left" if sgn > 0 else "right"), "center", 0.0
        arr = self.arrow(tail, tip, color=color, lw=LW_EDGE, head=4.5, rad=rad,
                         zorder=zorder)
        if dashed:
            arr.set_linestyle((0, (2.2, 1.8)))
        if label:
            self._label(tag_xy, label, color=color, size=PT_ANNOT, ha=ha, va=va)
        # 2026-09-08: register the arrival so require_delta0() can assert the
        # rule instead of the builder remembering it.
        self._delta0.append({"xy": (float(x), float(y)),
                             "soma": self._nearest_soma((x, y)),
                             "dashed": bool(dashed), "label": label})
        return arr

    # -- trees ------------------------------------------------------------
    def _fit_tree(self, rect, pos, *, pad_pt=(4.0, 4.0, 4.0, 4.0)):
        """Map unit coordinates into ``rect`` (points, aspect-true, soma at
        the bottom, centred horizontally); returns the frame coordinates,
        the pt-per-unit scale and the fitted rect."""
        left, right, top, bottom = pad_pt
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        w_pt = rect[2] * self.w_pt - left - right
        h_pt = rect[3] * self.h_pt - top - bottom
        s = max(1e-6, min(w_pt / max(xmax - xmin, 1e-6),
                          h_pt / max(ymax - ymin, 1e-6)))
        x_off = left + (w_pt - (xmax - xmin) * s) / 2.0
        out = {}
        for name, (px, py) in pos.items():
            out[name] = (rect[0] + self.fx(x_off + (px - xmin) * s),
                         rect[1] + self.fy(bottom + (py - ymin) * s))
        fitted = (rect[0] + self.fx(x_off), rect[1] + self.fy(bottom),
                  self.fx((xmax - xmin) * s), self.fy((ymax - ymin) * s))
        return out, s, fitted

    def balanced_tree(self, rect, *, depth=3, mode="plain", ghost=False,
                      badges=None, capsules=None, labels=True, trunk=True,
                      input_labels=None, output="z", soma_r_pt=SOMA_R_PT,
                      site_colors=None):
        """Soma-at-bottom balanced binary tree filling ``rect``.

        Wraps the credit_tree_schematics geometry (8 terminals at depth 3;
        depth 2 and 1 use the same junction positions) so every figure
        shares one drawing.  ``mode``: 'plain' (bare tree), 'inputs' (exc
        contact on every terminal), 'forward' (inputs plus the soma output
        arrow labelled ``output``).  ``trunk=False`` omits the J1 junction so
        the first branches leave the soma directly (Fig. 5's seven
        compartments).  ``badges`` maps junction names to operator /
        coefficient strings; ``capsules`` is a list of (node, colour key)
        addressed subtrees; ``input_labels`` (list of str or (base, sub))
        sit above the terminals and are dropped when they cannot fit.
        Returns a :class:`Nodes` dict (name -> frame xy).
        """
        depth = int(min(max(depth, 1), 3))
        pos = {"S": _ct.P["S"]}
        nodes = Nodes()
        nodes.soma_r_pt = soma_r_pt
        nodes.children["S"] = []
        if trunk:
            pos["J1"] = _ct.P["J1"]
            nodes.parent["J1"] = "S"
            nodes.children["S"] = ["J1"]
            nodes.children["J1"] = []
            nodes.level["J1"] = 0
        for d in range(1, depth + 1):
            terminal = d == depth
            for i, full in enumerate(_FULL_AT_DEPTH[d]):
                if d == 1:
                    par = "J1" if trunk else "S"
                else:
                    par = _FULL_PARENT[full]
                name = f"T{i + 1}" if terminal else full
                pos[name] = _ct.P[full]
                nodes.parent[name] = par
                nodes.children.setdefault(par, []).append(name)
                nodes.children.setdefault(name, [])
                nodes.level[name] = 3 if terminal else (0 if par == "S"
                                                        else min(d, 2))
        nodes.terminals = [n for n in pos if n.startswith("T")]
        # room above the canopy for contacts and input labels, below for the
        # soma disc; both fixed in points
        want_inputs = bool(labels and self.labels and input_labels)
        top_pad = 4.0 + (LINE_BAND_PT if want_inputs else 0.0)
        coords, s, fitted = self._fit_tree(
            rect, pos, pad_pt=(4.0, 4.0, top_pad, soma_r_pt + 1.5))
        nodes.update(coords)
        nodes.soma = coords["S"]
        nodes.rect = fitted
        nodes.orient = "up"
        term_x = sorted(coords[t][0] for t in nodes.terminals)
        nodes.pitch_pt = ((term_x[-1] - term_x[0]) * self.w_pt
                          / max(len(term_x) - 1, 1))
        if want_inputs and (term_x[-1] - term_x[0]) * self.w_pt < 36.0:
            # not even the ends-only labels fit: re-fit without the band
            want_inputs = False
            coords, s, fitted = self._fit_tree(
                rect, pos, pad_pt=(4.0, 4.0, 4.0, soma_r_pt + 1.5))
            nodes.update(coords)
            nodes.soma = coords["S"]
            nodes.rect = fitted
        # capsules go under everything
        for entry in (capsules or []):
            node, cname = entry[0], entry[1]
            pct = entry[2] if len(entry) > 2 else CAPSULE_PCT
            self._capsule(nodes, node, cname, pct)
        for name, par in nodes.parent.items():
            nodes.edges[(par, name)] = self.dendrite(
                nodes[par], nodes[name], level=nodes.level[name], ghost=ghost)
        badges = badges or {}
        for name in nodes.parent:
            if name in nodes.terminals:
                if site_colors is not None:
                    self.terminal(nodes[name], site_color=site_colors[
                        nodes.terminals.index(name) % len(site_colors)])
                continue
            if name in badges and self._badge_room(nodes):
                self._junction_badge(nodes[name], badges[name])
            else:
                nodes.rings[name] = self.junction(nodes[name], ghost=ghost)
        if mode in ("inputs", "forward"):
            for t in nodes.terminals:
                self.contact(nodes[t], kind="exc")
        self.soma(nodes.soma, r_pt=soma_r_pt, ghost=ghost,
                  output=(mode == "forward"),
                  label=output if (labels and self.labels) else None)
        if want_inputs:
            lift = CONTACT_DIA_PT * 0.5 + 2.0
            span_pt = (term_x[-1] - term_x[0]) * self.w_pt
            if nodes.pitch_pt >= INPUT_LABEL_MIN_PITCH_PT:
                for t, lab in zip(nodes.terminals, input_labels):
                    self._label(self._off(nodes[t], 0.0, lift), lab,
                                color=INK, size=PT_SMALL, ha="center",
                                va="bottom")
            elif span_pt >= 36.0 and len(input_labels) >= 2:
                # too dense for one label per terminal: name the two ends
                # and mark the run between them, never shrink the type
                first, last = nodes.terminals[0], nodes.terminals[-1]
                top = max(nodes[t][1] for t in nodes.terminals)
                self._label(self._off((nodes[first][0], top), 0.0, lift),
                            input_labels[0], color=INK, size=PT_SMALL,
                            ha="left", va="bottom")
                self._label(self._off((nodes[last][0], top), 0.0, lift),
                            input_labels[-1], color=INK, size=PT_SMALL,
                            ha="right", va="bottom")
                self.text(self._off(((nodes[first][0] + nodes[last][0]) / 2.0,
                                     top), 0.0, lift), "…", size=PT_SMALL,
                          color=MUTE, va="bottom")
        # 2026-09-08 glyph rule (a): assert what the drawing promised.
        nodes.row_axis = "x"
        self._trees.append(nodes)
        require_soma_lowest(nodes, self, name="balanced_tree")
        return nodes

    def _badge_room(self, nodes):
        """Whether junction badges (r 4.2 pt) fit between sibling junctions."""
        xs = sorted(self._to_pt(nodes[n])[0] for n in nodes.parent
                    if n not in nodes.terminals)
        gaps = [b - a for a, b in zip(xs, xs[1:]) if b - a > 0.5]
        return (min(gaps) if gaps else 1e9) >= 9.5

    def _junction_badge(self, xy, text, *, r_pt=4.2):
        self.disc(xy, r_pt, fill="white", edge=COLORS["dend"], lw=LW_EDGE,
                  zorder=3.6)
        self.text(xy, str(text), size=PT_ANNOT if len(str(text)) == 1
                  else PT_SMALL, color=INK, zorder=6, force=True)

    def site_tree(self, rect, *, branching=(3, 3), subtree_colors=None,
                  numbered=False, orient="up", soma_r_pt=SOMA_R_PT):
        """The [3,3] 12-site arbor, soma at the BOTTOM, sites fanning upward.

        2026-09-08: ``orient`` now defaults to ``'up'`` (it was ``'right'``).
        A tree is drawn one way in this paper -- soma lowest -- and a matrix
        that has to line up with its sites is no longer a reason to lay the
        neuron on its side: draw the tree upright and put a
        :meth:`site_strip` (the site order as a vertical strip, with no soma
        to mis-orient) between the tree and the matrix.

        ``orient='right'`` still draws the legacy soma-at-the-left arbor for
        the one builder that has not migrated; it is recorded in
        :data:`ORIENTATION_LEGACY`, is NOT soma-lowest, and is the only
        orientation :meth:`dictionary_product`'s ``align_to`` accepts from a
        tree.  Site discs are filled with their subtree colour (default
        shunting).  Returns a :class:`Nodes` dict (native site index -> frame
        xy) with ``rows`` (site -> display row) and ``order`` (native indices
        in display order) for :meth:`dictionary_matrix` alignment.
        """
        b1, b2 = int(branching[0]), int(branching[1])
        n = b1 * (1 + b2)
        colors = ([COLORS["shunting"]] * b1 if subtree_colors is None
                  else [COLORS.get(c, c) for c in subtree_colors])
        nodes = Nodes()
        nodes.soma_r_pt = soma_r_pt
        nodes.orient = orient
        x0, y0, w, h = rect
        w_pt, h_pt = w * self.w_pt, h * self.h_pt
        # local frame: u along soma->leaf, v across (display rows)
        length_pt = w_pt if orient == "right" else h_pt
        span_pt = h_pt if orient == "right" else w_pt
        number_pt = (10.0 if numbered else 0.0)
        u_soma = soma_r_pt + 1.5
        u_leaf = length_pt - 3.0 - number_pt
        u_prox = u_soma + 0.48 * (u_leaf - u_soma)
        pitch = span_pt / n
        rows = {}
        for k in range(b1):
            rows[k] = k * (b2 + 1)
            for j in range(b2):
                rows[b1 + b2 * k + j] = k * (b2 + 1) + 1 + j

        def place(u, r):
            v = span_pt - (r + 0.5) * pitch
            if orient == "right":
                return (x0 + self.fx(u), y0 + self.fy(v))
            return (x0 + self.fx(v), y0 + self.fy(u))

        soma_xy = place(u_soma, (n - 1) / 2.0)
        nodes.soma = soma_xy
        nodes.pitch_pt = pitch
        nodes.rows = rows
        nodes.order = sorted(rows, key=rows.get)
        for k in range(b1):
            nodes[k] = place(u_prox, rows[k])
            nodes.parent[k] = "S"
            nodes.children.setdefault("S", []).append(k)
            nodes.children[k] = []
            nodes.level[k] = 0
            for j in range(b2):
                site = b1 + b2 * k + j
                nodes[site] = place(u_leaf, rows[site])
                nodes.parent[site] = k
                nodes.children[k].append(site)
                nodes.children[site] = []
                nodes.level[site] = 3
        nodes.terminals = [s for s in nodes if nodes.level[s] == 3]
        nodes["S"] = soma_xy
        for site, par in nodes.parent.items():
            p0 = soma_xy if par == "S" else nodes[par]
            nodes.edges[(par, site)] = self.dendrite(p0, nodes[site],
                                                     level=nodes.level[site])
        for k in range(b1):
            nodes.rings[k] = self.junction(nodes[k], r_pt=SITE_R_PT[0],
                                           site_color=colors[k % len(colors)])
            for j in range(b2):
                site = b1 + b2 * k + j
                nodes.rings[site] = self.terminal(
                    nodes[site], site_color=colors[k % len(colors)])
        self.soma(soma_xy, r_pt=soma_r_pt)
        if numbered and self.labels and pitch >= 7.5:
            for site in nodes.terminals:
                xy = (self._off(nodes[site], 3.2, 0.0) if orient == "right"
                      else self._off(nodes[site], 0.0, 3.2))
                self.text(xy, str(site), size=PT_SMALL, color=INK,
                          ha="left" if orient == "right" else "center",
                          va="center" if orient == "right" else "bottom")
            for k in range(b1):
                # the number steps away from the soma stroke, which reaches
                # the proximal disc from the soma's side
                if orient == "right":
                    away = np.sign(nodes[k][1] - soma_xy[1]) or 1.0
                    xy = self._off(nodes[k], -2.6, 2.4 * away)
                    self.text(xy, str(k), size=PT_SMALL, color=INK,
                              ha="right", va="bottom" if away > 0 else "top")
                else:
                    away = np.sign(nodes[k][0] - soma_xy[0]) or -1.0
                    xy = self._off(nodes[k], 2.6 * away, -2.4)
                    self.text(xy, str(k), size=PT_SMALL, color=INK,
                              ha="left" if away > 0 else "right", va="top")
        nodes.row_axis = "y" if orient == "right" else "x"
        self._trees.append(nodes)
        if orient == "right":
            ORIENTATION_LEGACY.append(
                {"helper": "site_tree", "orient": "right",
                 "reason": "legacy soma-at-the-left arbor kept for a builder "
                           "that aligns a matrix to its rows; migrate to "
                           "orient='up' + Frame.site_strip"})
            self.note("orientation-legacy", helper="site_tree", orient="right")
        else:
            require_soma_lowest(nodes, self, name="site_tree")
        return nodes

    def site_strip(self, rect, *, branching=(3, 3), subtree_colors=None,
                   numbered=False, orient="vertical", band=True,
                   pct=CAPSULE_PCT, min_cell_pt=MIN_CELL_PT, indent_pt=5.0):
        """The site ORDER as a strip, for a matrix to align its rows to.

        New on 2026-09-08.  A dictionary's rows have to sit in the arbor's
        site order, and the only way the library could do that was to draw
        the arbor on its side (``site_tree(orient='right')``), which broke
        the one-orientation rule for every figure that carried a dictionary.
        The strip carries the same information the sideways tree carried --
        the display order, the subtree blocks and the proximal / distal
        distinction -- with no soma in it, so nothing can be mis-oriented:
        the neuron itself is drawn upright next to it.

        Rows run top-to-bottom (``orient='vertical'``, the default) in the
        same order :meth:`site_tree` uses: each proximal site, then its
        distal children.  Each block gets a 16 % tint band and a hairline
        ancestry spine; proximal sites are indented left of their children.
        Returns a :class:`Nodes` with ``rows``, ``order``, ``pitch_pt`` and
        ``row_axis='y'``, which is what :meth:`dictionary_product`'s
        ``align_to`` and :meth:`dictionary_matrix` want.  Raises
        :class:`MatrixTooDense` below ``min_cell_pt`` per row -- a strip that
        cannot be read is not an alignment.
        """
        if orient not in ("vertical", "horizontal"):
            raise ValueError("site_strip orient is 'vertical' or 'horizontal'")
        b1, b2 = int(branching[0]), int(branching[1])
        n = b1 * (1 + b2)
        colors = ([COLORS["shunting"]] * b1 if subtree_colors is None
                  else [COLORS.get(c, c) for c in subtree_colors])
        x0, y0, w, h = rect
        w_pt, h_pt = w * self.w_pt, h * self.h_pt
        span_pt = h_pt if orient == "vertical" else w_pt
        across_pt = w_pt if orient == "vertical" else h_pt
        pitch = span_pt / n
        if pitch < min_cell_pt - 1e-6:
            raise MatrixTooDense(
                f"site_strip: {n} rows in {span_pt:.1f} pt is {pitch:.2f} pt "
                f"per row, below the {min_cell_pt:.0f} pt floor -- give the "
                "strip a taller rect or collapse the dictionary to bands")
        nodes = Nodes()
        nodes.kind = "site_strip"
        nodes.orient = "up"
        nodes.row_axis = "y" if orient == "vertical" else "x"
        nodes.pitch_pt = pitch
        rows = {}
        for k in range(b1):
            rows[k] = k * (b2 + 1)
            for j in range(b2):
                rows[b1 + b2 * k + j] = k * (b2 + 1) + 1 + j
        nodes.rows = rows
        nodes.order = sorted(rows, key=rows.get)
        prox_pt = min(indent_pt, max(2.0, across_pt * 0.28))
        dist_pt = min(across_pt - 2.5, prox_pt + indent_pt)

        def place(across, r):
            v = span_pt - (r + 0.5) * pitch
            if orient == "vertical":
                return (x0 + self.fx(across), y0 + self.fy(v))
            return (x0 + self.fx(v), y0 + self.fy(across))

        for k in range(b1):
            colour = colors[k % len(colors)]
            block = [k] + [b1 + b2 * k + j for j in range(b2)]
            r_lo = min(rows[i] for i in block)
            r_hi = max(rows[i] for i in block)
            if band:
                top = span_pt - r_lo * pitch
                bot = span_pt - (r_hi + 1) * pitch
                if orient == "vertical":
                    tint_patch(self.ax,
                               ("rect", x0, y0 + self.fy(bot), w,
                                self.fy(top - bot)),
                               color=colour, pct=pct, radius_pt=1.5,
                               zorder=0.6, clip_on=False)
                else:
                    tint_patch(self.ax,
                               ("rect", x0 + self.fx(bot), y0,
                                self.fx(top - bot), h),
                               color=colour, pct=pct, radius_pt=1.5,
                               zorder=0.6, clip_on=False)
            nodes[k] = place(prox_pt, rows[k])
            nodes.parent[k] = "S"
            nodes.children.setdefault("S", []).append(k)
            nodes.children[k] = []
            nodes.level[k] = 0
            for j in range(b2):
                site = b1 + b2 * k + j
                nodes[site] = place(dist_pt, rows[site])
                nodes.parent[site] = k
                nodes.children[k].append(site)
                nodes.children[site] = []
                nodes.level[site] = 3
                # the ancestry bracket: a hairline elbow, never a new mark
                elbow = (nodes[k][0], nodes[site][1]) if orient == "vertical" \
                    else (nodes[site][0], nodes[k][1])
                self.dendrite(nodes[k], elbow, level=2)
                self.dendrite(elbow, nodes[site], level=3)
            self.junction(nodes[k], r_pt=SITE_R_PT[0], site_color=colour)
            for j in range(b2):
                self.terminal(nodes[b1 + b2 * k + j], site_color=colour)
        nodes.terminals = [s for s in nodes if nodes.level.get(s) == 3]
        if numbered and self.labels and pitch >= 6.0:
            for site in nodes.order:
                lead = nodes.level[site] == 0
                xy = (self._off(nodes[site], dist_pt - prox_pt + 4.0
                                if lead else 4.0, 0.0)
                      if orient == "vertical"
                      else self._off(nodes[site], 0.0, 4.0))
                self.text(xy, str(site), size=PT_SMALL, color=INK,
                          ha="left" if orient == "vertical" else "center",
                          va="center" if orient == "vertical" else "bottom")
        return nodes

    # -- capsules / partitions ----------------------------------------------
    def _capsule_width(self, nodes, n_terminals):
        factor = {1: 0.56, 2: 0.77, 4: 0.91}.get(int(n_terminals), 1.0)
        lo, hi = CAPSULE_W_PT
        return float(min(hi, max(lo, factor * nodes.pitch_pt)))

    def _draw_chains(self, chains, color, width_pt, zorder=1.4,
                     clip_on=True):
        """One capsule/ribbon as a FILLED tint patch (2026-09-08 spec §5).

        These used to be round-capped strokes 2.6-13 pt wide: legal only
        because the audit exempted anything above 2.5 pt as an "area mark", and
        in print heavier than every data line on the page.  The geometry is
        unchanged -- the chains are buffered to the same ``width_pt`` with
        round caps and joins -- but the result is one closed filled path with a
        ``LW_HAIR`` boundary in the same hue, which is what the tightened
        stroke rule (nothing above 1.35 pt unless it is a closed filled path)
        requires.  ``color`` is still the finished tint, so every call site is
        unchanged.
        """
        if width_pt <= LW_DATA:      # already a legal line weight: keep it
            for chain in chains:
                xy = np.array(chain, dtype=float)
                self.ax.plot(xy[:, 0], xy[:, 1], color=color,
                             lw=self.lw(width_pt), solid_capstyle="round",
                             solid_joinstyle="round", zorder=zorder,
                             clip_on=clip_on)
            return None
        return tint_patch(self.ax, ("ribbon", chains, width_pt),
                          color=color, face=color,
                          edge_color=strengthen(color, 2.4),
                          lw=LW_HAIR, zorder=zorder, clip_on=clip_on)

    def _capsule(self, nodes, root, cname, pct=CAPSULE_PCT, *, entry=True):
        """Pale capsule hugging the subtree rooted at ``root``."""
        terms = nodes.terminals_under(root) or [root]
        chains = []
        for i, t in enumerate(terms):
            path = [n for n in nodes.route(t) if n in nodes.subtree(root)]
            pts = [nodes[n] for n in path]
            if i == 0 and entry and root in nodes.parent:
                par = nodes.parent[root]
                p0 = nodes.soma if par == "S" and par not in nodes else nodes[par]
                pts = [_lerp(p0, nodes[root], 0.3)] + pts
            if len(pts) == 1:
                pts = [pts[0], pts[0]]
            chains.append(pts)
        require_address_tint(cname, where="Frame._capsule")
        self._draw_chains(chains, mix(cname, pct),
                          self._capsule_width(nodes, len(terms)))

    def partition(self, nodes, blocks, *, colors=None, labels=None,
                  pct=CAPSULE_PCT):
        """Pale capsules and PT_SMALL block labels over node-name blocks.

        ``blocks`` is a list of node-name lists (a descendant subtree, a
        sister block, the soma side).  ``colors`` are COLORS keys (default:
        the K-cycle order); ``labels`` entries are strings or (string, xy)
        pairs; a label that does not fit over its block is dropped.
        """
        cycle = list(colors) if colors else [c for c, _ in K_CYCLE[4]]
        labels = list(labels) if labels else [None] * len(blocks)
        for i, block in enumerate(blocks):
            block = list(block)
            cname = cycle[i % len(cycle)]
            members = set(block)
            chains = []
            for n in block:
                par = nodes.parent.get(n)
                if par in members:
                    chains.append([nodes[par], nodes[n]])
            if not chains:
                p = nodes[block[0]]
                chains.append([p, p])
            n_terms = sum(1 for n in block if n in nodes.terminals)
            require_address_tint(cname, where="Frame.partition")
            self._draw_chains(chains, mix(cname, pct),
                              self._capsule_width(nodes, max(n_terms, 1)))
            label = labels[i] if i < len(labels) else None
            if not label or not self.labels:
                continue
            text_color = AMBER_TEXT if cname in ("local", "scalar") \
                else label_color(COLORS.get(cname, cname))
            if isinstance(label, (tuple, list)):
                self.text(label[1], label[0], size=PT_SMALL, color=text_color)
                continue
            terms = [n for n in block if n in nodes.terminals]
            if terms:
                xs = [nodes[t][0] for t in terms]
                top = max(nodes[t][1] for t in terms)
                span = (max(xs) - min(xs)) * self.w_pt + nodes.pitch_pt
                if self._fits(label, PT_SMALL, span):
                    self.text(((max(xs) + min(xs)) / 2.0,
                               top + self.fy(CONTACT_DIA_PT * 0.5 + 2.5)),
                              label, size=PT_SMALL, color=text_color,
                              va="bottom")
            else:
                cx = np.mean([nodes[n][0] for n in block])
                cy = np.mean([nodes[n][1] for n in block])
                self.text((cx - self.fx(nodes.soma_r_pt + 3.0), cy), label,
                          size=PT_SMALL, color=text_color, ha="right")

    # -- credit delivery ----------------------------------------------------
    def credit_delivery(self, nodes, *, mode, rule_color=None, targets=None,
                        K=None, alpha_tags=False, label=None, source=None):
        """One of the FOUR delivery glyphs on a drawn tree -- and only four.

        2026-09-08 glyph rule (4).  Spread is the only thing that separates
        one credit rule from another, so the library draws it four ways and
        no other way; ``mode`` is a member of :data:`DELIVERY_MODES` or one
        of :data:`DELIVERY_ALIASES` ('broadcast', 'per_neuron', 'ancestry',
        'path', ...), and anything else raises.

        scalar   the layer scalar: one amber bus over the targets (default:
                 every terminal) with a hairline drop into each, its SOURCE
                 DOT sitting OUTSIDE the trees and tagged 's' -- the delta
                 comes from somewhere else and is the same everywhere;
        neuron   the per-neuron broadcast: the IDENTICAL amber bus, confined
                 to this tree, with the source dot AT this soma and a
                 hairline riser from the soma to the bus.  (Before this it
                 was an arrow into the trunk plus a dashed barrier arc -- a
                 second vocabulary for the same idea, and unreadable next to
                 the scalar card.  The arc is retired.)
        subtree  one 16 % tint patch over each addressed subtree (``targets``
                 or the K-cycle for ``K`` in {1, 2, 4, 8}) and ONE arrow in
                 the rule colour into that subtree's root; ``alpha_tags``
                 adds δ_k tags;
        exact    the arrow chain soma -> junction -> junction -> site along
                 the route to each target (default the library's T4) with
                 α_k tags at the junctions.

        ``rule_color`` defaults to the rule family's colour (amber for both
        bus glyphs, shunting for the address, bp for the exact path).
        ``label`` tags the bus ('s' by default for the layer scalar);
        ``source`` overrides where the bus is sourced ('outside' / 'soma').
        """
        mode = resolve_delivery_mode(mode)
        default = {"scalar": COLORS["scalar"], "neuron": COLORS["scalar"],
                   "subtree": COLORS["shunting"], "exact": COLORS["bp"]}[mode]
        color = default if rule_color is None else COLORS.get(rule_color,
                                                              rule_color)
        along, sidev = nodes.along, nodes.side
        if mode in ("scalar", "neuron"):
            source = source or ("outside" if mode == "scalar" else "soma")
            targets = list(targets) if targets else list(nodes.terminals)
            P = [self._to_pt(nodes[t]) for t in targets]
            a_vals = [float(p @ along) for p in P]
            s_vals = [float(p @ sidev) for p in P]
            S_pt = self._to_pt(nodes.soma)
            bus_a = max(a_vals) + 8.0
            s_lo, s_hi = min(s_vals) - 2.0, max(s_vals) + 6.0
            if source == "outside":
                # the source sits OUTSIDE the tree; clamped to the frame so a
                # narrow slot moves the dot in rather than off the page
                head_room = float((np.array([self.w_pt, self.h_pt])
                                   @ np.abs(sidev)) - 3.0)
                s_hi = min(s_hi + SOURCE_OUT_PT, max(s_hi + 2.0, head_room))
                if label is None:
                    label = "s"
            else:
                # the riser leaves the soma on the bus's NEAR end, so the far
                # end stays free for the card's own tag
                s_lo = min(s_lo, float(S_pt @ sidev) - 6.0)
            p_lo = self._from_pt(bus_a * along + s_lo * sidev)
            p_hi = self._from_pt(bus_a * along + s_hi * sidev)
            self.ax.plot([p_lo[0], p_hi[0]], [p_lo[1], p_hi[1]], color=color,
                         lw=self.lw(LW_HAIR), solid_capstyle="round", zorder=5)
            for a, s in zip(a_vals, s_vals):
                self.arrow(self._from_pt(bus_a * along + s * sidev),
                           self._from_pt((a + 2.8) * along + s * sidev),
                           color=color, lw=LW_HAIR, head=2.6, zorder=5)
            if source == "soma":
                # the riser: this soma is where the broadcast comes from
                r = nodes.soma_r_pt * self.scale
                foot = S_pt - sidev * (r + 2.2)
                corner = float(s_lo) * sidev + float(foot @ along) * along
                chain = [self._from_pt(foot), self._from_pt(corner),
                         self._from_pt(bus_a * along + s_lo * sidev)]
                self.ax.plot([p[0] for p in chain], [p[1] for p in chain],
                             color=color, lw=self.lw(LW_HAIR),
                             solid_capstyle="round", solid_joinstyle="round",
                             zorder=5)
                self.disc(self._from_pt(foot), 1.6, fill=color, zorder=6)
            else:
                self.disc(p_hi, 1.6, fill=color, zorder=6)
            if label and self.labels:
                text_color = AMBER_TEXT if _same_color(color, COLORS["scalar"]) \
                    else label_color(color)
                anchor = self._from_pt(bus_a * along + s_hi * sidev) \
                    if source == "outside" else p_lo
                tag = (self._off(anchor, 0.0, 3.0) if nodes.orient == "up"
                       else self._off(anchor, 3.0, 0.0))
                # a tag that would leave the cell is dropped, never shrunk
                if 0.01 <= tag[0] <= 0.99 and 0.01 <= tag[1] <= 0.97:
                    self.text(tag, label, size=PT_SMALL, color=text_color,
                              ha="center" if nodes.orient == "up" else "left",
                              va="bottom" if nodes.orient == "up" else "center")
            return None
        S = nodes.soma
        S_pt = self._to_pt(S)
        r = nodes.soma_r_pt * self.scale
        if mode == "subtree":
            if targets is None:
                K = 4 if K is None else int(K)
                targets = nodes.at_depth(K)
            targets = list(targets)
            cycle = K_CYCLE.get(len(targets), K_CYCLE[4])
            for i, t in enumerate(targets):
                cname, pct = cycle[i % len(cycle)]
                self._capsule(nodes, t, cname, pct)
                par = nodes.parent.get(t)
                if par is not None:
                    p0 = nodes[par] if par in nodes else nodes.soma
                    self.arrow(_lerp(p0, nodes[t], 0.40), _lerp(p0, nodes[t], 0.80),
                               color=color, lw=LW_EDGE, head=4.0, zorder=4.6)
                if alpha_tags and self.labels:
                    terms = nodes.terminals_under(t) or [t]
                    xs = [nodes[u][0] for u in terms]
                    top = max(nodes[u][1] for u in terms)
                    span = (max(xs) - min(xs)) * self.w_pt + nodes.pitch_pt
                    if span >= 12.0:
                        self.subscript(((max(xs) + min(xs)) / 2.0,
                                        top + self.fy(CONTACT_DIA_PT * 0.5 + 2.5)),
                                       "δ", str(i + 1), size=PT_ANNOT,
                                       color=AMBER_TEXT if cname == "local"
                                       else COLORS[cname], ha="center",
                                       va="bottom")
            return None
        # exact path
        if targets is None:
            targets = [nodes.terminals[min(3, len(nodes.terminals) - 1)]]
        for t in targets:
            route = nodes.route(t)
            for a, b in zip(route, route[1:]):
                pa = nodes[a] if a in nodes else nodes.soma
                self.dendrite(pa, nodes[b], level=nodes.level[b], color=color,
                              zorder=2.4)
                self.arrow(_lerp(pa, nodes[b], 0.34), _lerp(pa, nodes[b], 0.60),
                           color=color, lw=LW_EDGE, head=4.5, zorder=4.6)
            for j, (n, nxt) in enumerate(zip(route[1:-1], route[2:]), 1):
                self.disc(nodes[n], JUNCTION_R_PT, fill="white", edge=color,
                          lw=LW_EDGE, zorder=4.4)
                if alpha_tags and nodes.pitch_pt >= ALPHA_TAG_MIN_PITCH_PT:
                    outward = 1.0 if nodes[nxt][0] <= nodes[n][0] else -1.0
                    self.subscript(self._off(nodes[n], outward * 4.2, -2.0),
                                   "α", str(j), size=PT_ANNOT, color=color,
                                   ha="left" if outward > 0 else "right",
                                   va="center")
            self.disc(nodes[t], JUNCTION_R_PT, fill=color, zorder=4.6)
        return None

    # -- dictionaries -------------------------------------------------------
    def axes_inset(self, rect, *, grid="none"):
        """A data inset (spines, PT_TICK labels) sized by the caller in points.

        Use it for tuning curves or matrices inside a schematic cell; the
        journal panel look is applied so the inset audits like a panel.
        """
        inner = self.ax.inset_axes(rect, transform=self.ax.transData)
        inner.set_facecolor("none")
        style_panel(inner, grid=grid)
        return inner

    def dictionary_matrix(self, rect, A, *, color=None, row_groups=None,
                          col_colors=None, label="A", measured=False,
                          yticks=None, zorder=3, col_labels=None,
                          collapse="auto", collapse_blocks=False,
                          min_cell_pt=MIN_CELL_PT):
        """Site x profile matrix inset: rule-colour cells on panel_bg zeros.

        Hairline separators between ``row_groups`` (block sizes), LW_HAIR
        edge spines, no ticks unless ``yticks`` labels are given, and the
        PT_ANNOT caption ``'A  (N × K)'`` beneath (dropped when the rect
        leaves no room).  ``col_colors`` colours each profile column (the
        K = 3 subtree dictionary); ``measured=True`` draws realized route
        supports on ListedColormap([panel_bg, shunting]).

        2026-09-08 MATRIX RULE, asserted rather than advised: at least
        ``min_cell_pt`` (6.0) points per row AND per column, and a
        ``col_labels`` header no wider than 1.5 x its own column -- the
        review found eight columns in 30 pt under 6.8 pt headers.  A matrix
        that is too TALL has a remedy and takes it: with ``row_groups`` and
        ``collapse`` ('auto', the default) the bands are averaged into one
        row each and the inset prints "rows collapsed: N per band"; a matrix
        that is too WIDE has none, and raises :class:`MatrixTooDense` telling
        the caller to group the columns or use a wider module span.
        ``collapse_blocks=True`` collapses the bands whether or not the rows
        would have fitted -- the call ``Frame.dictionary_matrix(...,
        collapse_blocks=True)`` that the Fig. 7 builder's private
        ``_block_matrix`` was standing in for.
        """
        A = np.asarray(A, dtype=float)
        if A.ndim == 1:
            A = A[:, None]
        n, k = A.shape
        w_pt, h_pt = rect[2] * self.w_pt, rect[3] * self.h_pt
        collapsed = None
        if collapse_blocks and row_groups:
            collapse = True
        if collapse and row_groups and (collapse_blocks
                                        or h_pt / max(n, 1)
                                        < min_cell_pt - 1e-6):
            A, sizes = _collapse_rows(A, row_groups)
            n = A.shape[0]
            row_groups = None
            collapsed = collapsed_row_note(sizes)
            if yticks is not None and len(list(yticks)) != n:
                yticks = None
            self.note("matrix-collapsed", text=collapsed, rows=int(n))
        check_matrix_cells(w_pt, h_pt, n, k, where="dictionary_matrix",
                           min_cell_pt=min_cell_pt, headers=col_labels,
                           ax=self.ax)
        rule = COLORS["shunting"] if color is None else COLORS.get(color, color)
        inner = self.ax.inset_axes(rect, transform=self.ax.transData,
                                   zorder=zorder)
        inner.set_facecolor("white")
        if measured:
            inner.imshow((A != 0).astype(int), aspect="auto",
                         interpolation="nearest", vmin=0, vmax=1,
                         cmap=ListedColormap([COLORS["panel_bg"], rule]))
        elif col_colors is not None:
            rgba = np.ones((n, k, 4))
            zero = to_rgba(COLORS["panel_bg"])
            top = float(np.abs(A).max()) or 1.0
            for j in range(k):
                cj = np.asarray(to_rgba(COLORS.get(col_colors[j % len(col_colors)],
                                                   col_colors[j % len(col_colors)])))
                for i in range(n):
                    f = abs(A[i, j]) / top
                    rgba[i, j] = f * cj + (1.0 - f) * np.asarray(zero)
            inner.imshow(rgba, aspect="auto", interpolation="nearest")
        elif A.min() < 0:
            m = float(np.abs(A).max()) or 1.0
            inner.imshow(A, aspect="auto", interpolation="nearest",
                         cmap=DIV_CMAP, vmin=-m, vmax=m)
        else:
            top = float(A.max()) or 1.0
            inner.imshow(A, aspect="auto", interpolation="nearest",
                         cmap=LinearSegmentedColormap.from_list(
                             "dictionary", [COLORS["panel_bg"], rule]),
                         vmin=0, vmax=top)
        inner.set_xticks([])
        inner.set_yticks([])
        for spine in inner.spines.values():
            spine.set_visible(True)
            spine.set_color(COLORS["edge"])
            spine.set_linewidth(LW_HAIR)
        if row_groups:
            edge_at = np.cumsum([int(g) for g in row_groups])[:-1]
            for b in edge_at:
                inner.axhline(b - 0.5, color=COLORS["edge"], lw=LW_HAIR,
                              zorder=4)
        if yticks is not None:
            inner.set_yticks(range(n))
            inner.set_yticklabels([str(t) for t in yticks], fontsize=PT_SMALL,
                                  color=INK)
            inner.tick_params(axis="y", length=0, pad=1.5, labelsize=PT_SMALL)
        if col_labels is not None and self.labels:
            col_pt = w_pt / max(k, 1)
            for j, head in enumerate(col_labels[:k]):
                if not head:
                    continue
                self.text((rect[0] + self.fx((j + 0.5) * col_pt),
                           rect[1] + rect[3] + self.fy(2.0)), str(head),
                          size=PT_SMALL, color=INK, va="bottom")
        drop_pt = 6.5
        if label and self.labels and rect[1] * self.h_pt >= 11.0:
            caption = f"{label}  ({n} × {k})"
            if self._fits(caption, PT_ANNOT, rect[2] * self.w_pt + 14.0):
                self.text((rect[0] + rect[2] / 2.0, rect[1] - self.fy(drop_pt)),
                          caption, size=PT_ANNOT, color=INK)
                drop_pt += LINE_BAND_PT
        if collapsed and self.labels:
            # the note is a caption, so it may be wider than the matrix; it is
            # wrapped to the CELL and only dropped if even that cannot hold it
            collapsed = _wrap_to_width(self.ax, collapsed, PT_SMALL,
                                       self.w_pt - 4.0)
        if collapsed and self.labels:
            # the note goes under the matrix, or over it when the rect sits on
            # the floor of the cell -- it is never dropped: a collapsed matrix
            # that does not say so is a matrix with the wrong number of rows
            if rect[1] * self.h_pt >= drop_pt + 4.0:
                self.text((rect[0] + rect[2] / 2.0, rect[1] - self.fy(drop_pt)),
                          collapsed, size=PT_SMALL, color=MUTE)
            else:
                self.text((rect[0] + rect[2] / 2.0,
                           rect[1] + rect[3] + self.fy(2.5)), collapsed,
                          size=PT_SMALL, color=MUTE, va="bottom")
        return inner

    def dictionary_product(self, rect, A, c, *, color=None, numbers=True,
                           cell_pt=8.0, col_colors=None, row_groups=None,
                           captions=("A", "c", "A c"), align_to=None,
                           collapse="auto", min_cell_pt=MIN_CELL_PT):
        """``A × c = A c`` as three row-aligned insets inside ``rect``.

        Cells are laid out in points (``cell_pt`` for the two columns,
        6-8 pt for the matrix columns) and the whole group is centred in the
        rect, so the triplet reads identically in a 5- or 7-module slot.
        The coefficient and field columns sit on DIV_CMAP with printed
        PT_ANNOT values; operators are PT_TITLE.  ``align_to`` takes the
        :class:`Nodes` of a :meth:`site_strip` (or the legacy
        ``site_tree(orient='right')``) and snaps the matrix rows onto its
        site rows (same pitch, same top edge) so row *i* of ``A`` sits beside
        site row *i*; a soma-lowest tree spreads its sites across x and is
        refused, with the strip named as the remedy.

        2026-09-08: the same MATRIX RULE as :meth:`dictionary_matrix` -- at
        least ``min_cell_pt`` per row and per column, the triplet must fit
        the rect, and ``collapse`` with ``row_groups`` averages the bands and
        prints "rows collapsed: N per band" rather than drawing rows nobody
        can resolve.
        Returns (A, c, Ac) axes.
        """
        A = np.asarray(A, dtype=float)
        c = np.asarray(c, dtype=float).reshape(-1)
        n, k = A.shape
        x0, y0, w, h = rect
        w_pt, h_pt = w * self.w_pt, h * self.h_pt
        cap_pt = LINE_BAND_PT if (captions and self.labels) else 0.0
        if align_to is not None and getattr(align_to, "row_axis", "y") == "x":
            raise SchematicRuleError(
                "dictionary_product(align_to=...) needs rows that run down "
                "the page: a soma-lowest tree spreads its sites across x. "
                "Draw the tree upright and align the matrix to a "
                "Frame.site_strip(orient='vertical') instead.")
        collapsed = None
        row = min(cell_pt, (h_pt - cap_pt) / max(n, 1))
        if collapse and row_groups and row < min_cell_pt - 1e-6:
            A, sizes = _collapse_rows(A, row_groups)
            n = A.shape[0]
            row_groups = None
            collapsed = collapsed_row_note(sizes)
            row = min(cell_pt, (h_pt - cap_pt) / max(n, 1))
            self.note("matrix-collapsed", text=collapsed, rows=int(n))
        field = A @ c
        if align_to is not None and getattr(align_to, "rows", None):
            row = float(align_to.pitch_pt)
            first = align_to.order[0]
            top_pt = (align_to[first][1] - y0) * self.h_pt + row / 2.0
        else:
            top_pt = None
        col = max(6.0, min(cell_pt, row))
        op_pt = 10.0
        total = k * col + op_pt + cell_pt + op_pt + cell_pt
        if total > w_pt:
            col = max(6.0, col - (total - w_pt) / max(k, 1))
            total = k * col + op_pt + cell_pt + op_pt + cell_pt
        if row < min_cell_pt - 1e-6:
            raise MatrixTooDense(
                f"dictionary_product: {n} rows at {row:.2f} pt, below the "
                f"{min_cell_pt:.0f} pt floor -- pass row_groups=[...] so the "
                "bands can be collapsed, or give the triplet a taller rect")
        if col < min_cell_pt - 1e-6 or total > w_pt + 0.5:
            raise MatrixTooDense(
                f"dictionary_product: the A × c = Ac triplet needs "
                f"{total:.1f} pt ({k} columns at {col:.2f} pt) and has "
                f"{w_pt:.1f} pt -- group the profile columns or move the "
                "triplet to a wider module span")
        left = (w_pt - total) / 2.0
        top = (h_pt - (h_pt - cap_pt - n * row) / 2.0 if top_pt is None
               else top_pt)
        yA = y0 + self.fy(top - n * row)

        def box(x_pt, width_pt, rows):
            return (x0 + self.fx(left + x_pt), y0 + self.fy(top - rows * row)
                    if rows == n else yA + self.fy((n - rows) * row / 2.0),
                    self.fx(width_pt), self.fy(rows * row))

        rect_A = box(0.0, k * col, n)
        rect_c = box(k * col + op_pt, cell_pt, k)
        rect_f = box(k * col + op_pt + cell_pt + op_pt, cell_pt, n)
        ax_A = self.dictionary_matrix(rect_A, A, color=color, label=None,
                                      col_colors=col_colors,
                                      row_groups=row_groups)
        m = max(float(np.abs(c).max()), float(np.abs(field).max()), 1e-9)

        def column(box_rect, values):
            inner = self.ax.inset_axes(box_rect, transform=self.ax.transData,
                                       zorder=3)
            inner.imshow(values.reshape(-1, 1), aspect="auto",
                         interpolation="nearest", cmap=DIV_CMAP, vmin=-m,
                         vmax=m)
            inner.set_xticks([])
            inner.set_yticks([])
            for spine in inner.spines.values():
                spine.set_color(COLORS["edge"])
                spine.set_linewidth(LW_HAIR)
            if numbers and row >= 7.5:
                for i, v in enumerate(values):
                    inner.text(0, i, _signed(v), ha="center", va="center",
                               fontsize=PT_ANNOT,
                               color="white" if abs(v) / m > 0.55 else INK)
            return inner

        ax_c = column(rect_c, c)
        ax_f = column(rect_f, field)
        if row_groups and n == len(field):
            edge_at = np.cumsum([int(g) for g in row_groups])[:-1]
            for b in edge_at:
                ax_f.axhline(b - 0.5, color=COLORS["edge"], lw=LW_HAIR)
        mid_y = yA + self.fy(n * row / 2.0)
        for x_pt, op in ((k * col + op_pt / 2.0, "×"),
                         (k * col + op_pt + cell_pt + op_pt / 2.0, "=")):
            self.text((x0 + self.fx(left + x_pt), mid_y), op, size=PT_TITLE,
                      color=INK, force=True)
        if cap_pt:
            for r_, text in zip((rect_A, rect_c, rect_f), captions):
                if text:
                    self.text((r_[0] + r_[2] / 2.0, yA - self.fy(cap_pt * 0.55)),
                              text, size=PT_ANNOT, color=INK)
        if collapsed and self.labels:
            y_note = yA - self.fy(cap_pt + 4.0)
            if y_note > rect[1] - self.fy(2.0) - self.fy(LINE_BAND_PT):
                self.text((rect_A[0] + rect_A[2] / 2.0, y_note), collapsed,
                          size=PT_SMALL, color=MUTE, va="top")
        return ax_A, ax_c, ax_f

    # -- cards, badges, keys ------------------------------------------------
    def task_card(self, rect, *, title=None, footer=None, emphasis=False,
                  tone=None, title_color=None, radius_pt=3.0):
        """Rounded white card with a hairline grid edge; returns the core rect.

        ``emphasis`` tints the hero card mix('shunting', 8) with a
        mix('shunting', 45) edge; ``tone`` is a COLORS key for another
        tonal card ('control' gives the panel_bg control card).  The header
        is PT_ANNOT top-left, the footer PT_SMALL mute centred; each is
        wrapped to the card width or dropped, and the core rect excludes
        their bands so the glyph never collides with them.
        """
        if emphasis:
            tint, edge = mix("shunting", 8), mix("shunting", 45)
        elif tone == "control":
            tint, edge = COLORS["panel_bg"], GRID
        elif tone:
            tint, edge = mix(tone, 8), mix(tone, 45)
        else:
            tint, edge = "white", GRID
        self.group(rect, tint=tint, edge=edge, lw=LW_HAIR, radius_pt=radius_pt)
        x0, y0, w, h = rect
        w_pt, h_pt = w * self.w_pt, h * self.h_pt
        top_pt = bot_pt = 0.0
        title_text = footer_text = None
        if title and self.labels:
            title_text = _wrap_to_width(self.ax, title, PT_ANNOT, w_pt - 10.0)
            if title_text:
                top_pt = LINE_BAND_PT * (1 + title_text.count("\n")) + 3.0
        if footer and self.labels:
            footer_text = _wrap_to_width(self.ax, footer, PT_SMALL, w_pt - 8.0)
            if footer_text:
                bot_pt = LINE_BAND_PT * (1 + footer_text.count("\n")) + 1.0
        if h_pt - top_pt - bot_pt < MIN_CORE_PT and footer_text:
            footer_text, bot_pt = None, 0.0
        if h_pt - top_pt - bot_pt < MIN_CORE_PT and title_text:
            title_text, top_pt = None, 0.0
        if title_text:
            self.text((x0 + self.fx(5.0), y0 + h - self.fy(4.0)), title_text,
                      size=PT_ANNOT, color=INK if title_color is None
                      else title_color, ha="left", va="top", linespacing=1.15)
        if footer_text:
            self.text((x0 + w / 2.0, y0 + self.fy(bot_pt * 0.5)), footer_text,
                      size=PT_SMALL, color=MUTE, linespacing=1.15)
        return (x0, y0 + self.fy(bot_pt), w, h - self.fy(top_pt + bot_pt))

    def badge(self, xy, kind, *, ha="right", va="top", text=None, zorder=7):
        """Rounded PT_SMALL tag: oracle / local rule / exact / control / teacher."""
        if kind not in BADGE_STYLE:
            raise ValueError(f"unknown badge kind {kind!r}; "
                             f"expected one of {tuple(BADGE_STYLE)}")
        key, face, edge = BADGE_STYLE[kind]
        colour = COLORS[key]
        try:
            colour = label_color(colour, background=face)
        except ValueError:
            pass
        size = PT_SMALL
        art = self.ax.text(
            xy[0], xy[1], kind if text is None else text, fontsize=size,
            color=colour, ha=ha, va=va, zorder=zorder,
            bbox=dict(boxstyle=f"round,pad=0.28,rounding_size={2.0 / size:.3f}",
                      facecolor=face, edgecolor=edge, linewidth=LW_HAIR))
        return art

    def teacher_student(self, ax, x, y_teacher, y_student, *, rule_color,
                        label_teacher="teacher", label_student=None):
        """Teacher = ink dashed LW_REF; student = rule-coloured solid LW_DATA.

        Draws on ``ax`` (a data axes, e.g. from :meth:`axes_inset`) and sets
        the legend labels so every figure's key reads 'teacher' the same
        way.  Returns the two lines.
        """
        rule_color = COLORS.get(rule_color, rule_color)
        teacher, = ax.plot(x, y_teacher, color=INK, lw=LW_REF, zorder=3,
                           label=label_teacher)
        teacher.set_dashes((2.2, 1.8))
        student, = ax.plot(x, y_student, color=rule_color, lw=LW_DATA,
                           zorder=4, label=label_student,
                           solid_capstyle="round")
        return teacher, student

    def _mini_glyph(self, xy, mode, color, *, w_pt=12.0):
        """Delivery mini-glyph for a rule key entry, ``w_pt`` wide.

        2026-09-08: the key draws the same four glyphs the schematics draw,
        so 'scalar' and 'neuron' are one bus differing only in where its
        source dot sits (outside the tree / at the soma) -- which is the only
        difference between the two rules on the tree as well.
        """
        mode = None if mode is None else (
            resolve_delivery_mode(mode) if str(mode) not in ("teacher",)
            else "teacher")
        x, y = xy
        x1 = x + self.fx(w_pt)
        if mode in ("scalar", "neuron"):
            self.ax.plot([x, x1], [y + self.fy(2.2)] * 2, color=color,
                         lw=self.lw(LW_HAIR), solid_capstyle="round", zorder=5)
            self.disc((x1 if mode == "scalar" else x, y + self.fy(2.2)), 1.1,
                      fill=color, zorder=6)
            for f in (0.15, 0.5, 0.85):
                xx = x + self.fx(w_pt * f)
                self.arrow((xx, y + self.fy(2.2)), (xx, y - self.fy(2.4)),
                           color=color, lw=LW_HAIR, head=2.2, zorder=5)
            if mode == "neuron":       # the source dot is AT the soma
                self.disc((x, y - self.fy(2.4)), 1.7, fill=color, zorder=6)
        elif mode == "subtree":
            self._draw_chains([[(x + self.fx(1.5), y), (x1 - self.fx(1.5), y)]],
                              mix(color, CAPSULE_PCT), 3.2, zorder=1.4)
            self.arrow((x, y), (x + self.fx(w_pt * 0.55), y), color=color,
                       lw=LW_EDGE, head=3.0, zorder=5)
        elif mode == "exact":
            self.ax.plot([x, x1], [y, y], color=color, lw=self.lw(LW_EDGE),
                         solid_capstyle="round", zorder=4)
            for f in (0.35, 0.75):
                self.arrow((x + self.fx(w_pt * (f - 0.2)), y),
                           (x + self.fx(w_pt * f), y), color=color,
                           lw=LW_EDGE, head=3.4, zorder=5)
        elif mode == "teacher":
            line, = self.ax.plot([x, x1], [y, y], color=INK,
                                 lw=self.lw(LW_REF), zorder=4)
            line.set_dashes((2.2, 1.8))
        else:
            self.ax.plot([x, x1], [y, y], color=color, lw=self.lw(LW_DATA),
                         solid_capstyle="round", zorder=4)

    def rule_key(self, rect, rules, *, ncol=None, size=PT_LEGEND):
        """One-line key of rule swatches: mini delivery glyph + name + badge.

        ``rules`` entries are ``(name, colour key, mode[, badge kind])`` or
        dicts with those keys; ``mode`` is a delivery mode, 'teacher', or
        None for a plain swatch.  Entries wrap onto further lines while
        the rect is tall enough and are dropped otherwise.

        THE FOUR DELIVERY GLYPHS, and the only four the paper draws
        (:data:`DELIVERY_MODES`; :data:`DELIVERY_ALIASES` maps the older
        names onto them):

        ``'scalar'``   layer scalar -- an amber bus spanning the tree and its
                       ghost neighbours, source dot OUTSIDE the trees, tagged
                       's'.  Same delta everywhere.
        ``'neuron'``   per-neuron broadcast -- the identical amber bus
                       confined to ONE tree, source dot AT that soma.  Which
                       tree, not where in it.
        ``'subtree'``  subtree / ancestry address -- a 16 % tint patch hugging
                       the addressed subtree and ONE arrow in the rule colour
                       into its root; tints come from the four-hue address
                       cycle, never from a data series.
        ``'exact'``    exact path -- the arrow chain soma -> junction ->
                       junction -> site with α tags.

        This key is the one place a figure may state a series before a plot
        does (spec §5, "legend idiom"): the reader meets the glyph, the name
        and the badge here, and no data panel carries a key at all.
        """
        if not self.labels:
            return None
        entries = []
        for r in rules:
            if isinstance(r, dict):
                name, cname = r["name"], r.get("color", "mute")
                mode, kind = r.get("mode"), r.get("badge")
            else:
                name, cname = r[0], r[1]
                mode = r[2] if len(r) > 2 else None
                kind = r[3] if len(r) > 3 else None
            entries.append((name, COLORS.get(cname, cname), mode, kind))
        glyph_pt, gap_pt, entry_gap = 12.0, 3.5, 9.0
        widths = []
        for name, _, _, kind in entries:
            wpt = glyph_pt + gap_pt + _text_w_pt(self.ax, name, size)
            if kind:
                wpt += 4.0 + _text_w_pt(self.ax, kind, PT_SMALL) + 4.5
            widths.append(wpt)
        x0, y0, w, h = rect
        w_pt, h_pt = w * self.w_pt, h * self.h_pt
        rows_fit = max(1, int(h_pt // LINE_BAND_PT))
        lines, cur, cur_w = [], [], 0.0
        for i, wpt in enumerate(widths):
            over = (ncol and len(cur) >= ncol) or (cur and cur_w + entry_gap
                                                   + wpt > w_pt)
            if over:
                lines.append(cur)
                cur, cur_w = [], 0.0
            cur.append(i)
            cur_w += (entry_gap if len(cur) > 1 else 0.0) + wpt
        if cur:
            lines.append(cur)
        lines = lines[:rows_fit]
        y = y0 + h - self.fy(LINE_BAND_PT * 0.5)
        for line in lines:
            total = sum(widths[i] for i in line) + entry_gap * (len(line) - 1)
            x = x0 + self.fx((w_pt - total) / 2.0)
            for i in line:
                name, colour, mode, kind = entries[i]
                self._mini_glyph((x, y), mode, colour)
                x += self.fx(glyph_pt + gap_pt)
                self.text((x, y), name, size=size, color=INK, ha="left")
                x += self.fx(_text_w_pt(self.ax, name, size))
                if kind:
                    self.badge((x + self.fx(4.0), y), kind, ha="left",
                               va="center")
                    x += self.fx(4.0 + _text_w_pt(self.ax, kind, PT_SMALL) + 4.5)
                x += self.fx(entry_gap)
            y -= self.fy(LINE_BAND_PT)
        return None


def _text_w_pt(ax, s, size):
    """Rendered width of ``s`` in points, measured rather than guessed."""
    fig = ax.get_figure()
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:                      # no renderer yet: estimate
        return 0.52 * size * len(s)
    art = ax.text(0.0, 0.0, s, fontsize=size)
    try:
        return art.get_window_extent(renderer=renderer).width / fig.dpi * 72.0
    except Exception:
        return 0.52 * size * len(s)
    finally:
        art.remove()


def _wrap_to_width(ax, s, size, max_w_pt, max_lines=2):
    """Wrap ``s`` to fit ``max_w_pt``; ``None`` when it cannot.

    Mathtext is never broken -- there is no safe place to split ``$...$`` --
    so a formula that does not fit is reported as unfittable and the caller
    drops it instead of letting it run out of its cell.
    """
    import textwrap

    if _text_w_pt(ax, s, size) <= max_w_pt:
        return s
    if "$" in s or " " not in s:
        return None
    words = s.split()
    approx = max(1, int(len(s) * max_w_pt / max(_text_w_pt(ax, s, size), 1e-6)))
    for width in range(approx + 4, 3, -1):
        try:
            lines = textwrap.wrap(s, width=width, max_lines=max_lines)
        except ValueError:                  # width below textwrap's floor
            break
        if not lines or len(lines) > max_lines:
            continue
        if any(_text_w_pt(ax, line, size) > max_w_pt for line in lines):
            continue
        if sum(len(l.split()) for l in lines) != len(words):
            continue                        # textwrap dropped a word
        return "\n".join(lines)
    return None


def _tint(color, pct=8):
    return mix(color, pct)


# ── Figure 2 ─────────────────────────────────────────────────────────────
def draw_ownership_address(ax, *, labels=True, scale=1.0):
    """Neuron ownership (which tree) versus within-tree address (where)."""
    f = Frame(ax, labels=labels, scale=scale)
    left, right = f.split(2, gap_pt=12.0)
    f.group(left, tint=_tint("shunting"), edge=mix("shunting", 45))
    f.group(right, tint=_tint("oracle"), edge=mix("oracle", 45))

    core_l = f.cell_text(
        left, title="ownership", title_color=GREEN,
        subtitle=r"which neuronal tree receives $\delta_u$?",
        foot="same bandwidth; different neuron-to-tree map")
    # The pair of trees is one comparison: each takes half the core, and the
    # two tags share a band below them.
    pair = Frame.inset(core_l, left=0.04, right=0.04)
    tag_pt = LINE_BAND_PT if f.room(pair, MIN_CORE_PT + LINE_BAND_PT) else 0.0
    body = (pair[0], pair[1] + f.fy(tag_pt), pair[2], pair[3] - f.fy(tag_pt))
    wa = body[2] / 2.0
    tree_a = (body[0], body[1], wa * 0.92, body[3])
    tree_b = (body[0] + wa * 1.08, body[1], wa * 0.92, body[3])
    f.tree(tree_a, mode="coordinate", labels=False)
    f.tree(tree_b, mode="coordinate", labels=False)
    if tag_pt:
        for rect, tag, color in ((tree_a, "correct tree", GREEN),
                                 (tree_b, "deranged tree", MUTE)):
            f.text((rect[0] + rect[2] / 2.0,
                    pair[1] + f.fy(tag_pt * 0.45)), tag,
                   size=PT_SMALL, color=color)
    cx = tree_b[0] + tree_b[2] / 2.0
    cy = tree_b[1] + tree_b[3] * 0.56
    rx, ry = f.fx(6.5), f.fy(6.5)
    for sign in (1, -1):
        ax.plot([cx - rx, cx + rx], [cy - sign * ry, cy + sign * ry],
                color=RED, lw=f.lw(LW_DATA), solid_capstyle="round", zorder=7)

    core_r = f.cell_text(
        right, title="within-tree address", title_color=PURPLE,
        subtitle="once the neuron is known, where should credit go?",
        foot=r"subtree routes refine $\delta_u \rightarrow \delta_{u,k}$")
    f.tree(Frame.inset(core_r, left=0.12, right=0.12, top=0.03, bottom=0.03),
           mode="address", K=4)
    return ax


# ── Figure 3 ─────────────────────────────────────────────────────────────
ROUTE_KEY = (
    (GREEN, "o", "ancestry"),
    (ROSE, "s", "deranged"),
    (AMBER, "^", "depth-interleaved"),
    (MUTE, "D", "random sparse"),
    (BLUE, "v", r"random rank-$K$"),
    (PURPLE, "P", r"learned rank-$K$"),
)


def draw_route_resolution(ax, *, labels=True, scale=1.0, key=True):
    """Coarse-to-fine ancestry addresses, with the route-family key."""
    f = Frame(ax, labels=labels, scale=scale)
    # The six-entry key needs a wide frame.  In a narrow slot it is dropped
    # rather than wrapped into an illegible stack: the style contract allows
    # a key to be stated once in the figure and cross-referenced.
    key = bool(key and labels and f.h_pt >= 95.0 and f.w_pt >= 300.0)
    key_pt = 13.0 if key else 0.0
    cells = f.split(4, gap_pt=8.0, pad_pt=(0, 0, 0, key_pt))
    modes = (("coordinate", 1), ("address", 2), ("address", 4), ("address", 8))
    tags = (
        (r"$K=1$", "one shared coordinate"),
        (r"$K=2$", "two coarse subtrees"),
        (r"$K=4$", "intermediate routes"),
        (r"$K=8$", "full route resolution"),
    )
    for index, (cell, (mode, budget), (tag, desc)) in enumerate(
            zip(cells, modes, tags, strict=True)):
        emphasis = index == 2
        f.group(cell, tint=_tint("shunting") if emphasis else COLORS["panel_bg"],
                edge=mix("shunting", 45) if emphasis else GRID)
        # The budget tag and its gloss are two point-high bands under the
        # tree, not a fraction of the cell: at a short cell height a
        # fractional reserve puts the tag straight through the branches.
        core = f.cell_text(cell, foot=desc,
                           foot_color=GREEN if emphasis else MUTE)
        core = f.cell_text(core, foot=tag, foot_size=PT_ANNOT,
                           foot_color=INK)
        f.tree(Frame.inset(core, left=0.04, right=0.04, top=0.05,
                           bottom=0.02),
               mode=mode, K=budget, labels=False)
    if key:
        handles = [
            Line2D([0], [0], color=color, marker=marker, lw=LW_DATA,
                   markersize=4.0, markeredgecolor="white",
                   markeredgewidth=LW_HAIR, label=label)
            for color, marker, label in ROUTE_KEY
        ]
        ax.legend(handles=handles, loc="lower center",
                  bbox_to_anchor=(0.5, -0.02), ncol=len(ROUTE_KEY),
                  frameon=False, fontsize=PT_LEGEND, handlelength=1.2,
                  handletextpad=0.35, columnspacing=0.9, borderaxespad=0.0)
    return ax


# ── Figure 4 ─────────────────────────────────────────────────────────────
def draw_credit_operator(ax, *, labels=True, scale=1.0, formula=True):
    """Exact stochastic credit enters a route operator and exits filtered."""
    f = Frame(ax, labels=labels, scale=scale)
    foot_pt = f.footer(
        r"$U(M)=\dfrac{[g^{\mathsf{T}}Mg]^{2}}"
        r"{2L\{\|Mg\|^{2}+\mathrm{tr}(M\Sigma M^{\mathsf{T}})\}}$",
        size=PT_ANNOT, band_pt=20.0, min_frame_pt=115.0) if formula else 0.0
    body = (0.0, f.fy(foot_pt), 1.0, 1.0 - f.fy(foot_pt))
    # The operator row takes its share of the body, but never drops below the
    # height its two stacked lines of type need.
    row_pt = max(34.0, body[3] * f.h_pt * 0.34)
    box_top = body[1] + body[3]
    box_bot = box_top - f.fy(row_pt)
    gap = f.fx(16.0)
    bw = (1.0 - 2 * gap) / 3.0
    boxes = [(i * (bw + gap), box_bot, bw, box_top - box_bot)
             for i in range(3)]
    fills = (COLORS["panel_bg"], _tint("shunting"), _tint("oracle"))
    edges = (MUTE, mix("shunting", 45), mix("oracle", 45))
    inks = (MUTE, GREEN, PURPLE)
    titles = ("stochastic credit", "route operator", "routed update")
    symbols = (r"$\widehat{g}$", r"$M$", r"$M\widehat{g}$")
    for box, fill, edge, ink, title, symbol in zip(
            boxes, fills, edges, inks, titles, symbols, strict=True):
        f.group(box, tint=fill, edge=edge)
        cx = box[0] + box[2] / 2.0
        f.text((cx, box[1] + box[3] - f.fy(9.0)), title, size=PT_SMALL)
        f.text((cx, box[1] + f.fy(11.0)), symbol, size=PT_LABEL, color=ink)
    mid = box_bot + (box_top - box_bot) / 2.0
    for a, b in ((boxes[0], boxes[1]), (boxes[1], boxes[2])):
        f.arrow((a[0] + a[2] + f.fx(2.0), mid), (b[0] - f.fx(2.0), mid))

    # What leaves the operator, in the band under the row: the dots sit
    # beneath the third box and the gloss stacks under the first two, each
    # line on its own point-high row so the three never merge.
    under = (0.0, body[1], 1.0, box_bot - body[1])
    dot_y = under[1] + under[3] * 0.44
    xs = np.linspace(boxes[2][0] + f.fx(6.0),
                     boxes[2][0] + boxes[2][2] - f.fx(6.0), 6)
    for x, color in zip(xs, (GREEN, GREEN, GREEN, AMBER, MUTE, MUTE),
                        strict=True):
        f.disc((x, dot_y), 2.1, fill=color, zorder=6)
    cx3 = boxes[2][0] + boxes[2][2] / 2.0
    f.leader((cx3, box_bot - f.fy(2.0)), (cx3, dot_y + f.fy(5.0)))
    if f.room(under, 26.0):
        # centred under the third box, so it may only claim that box's width
        kept = _wrap_to_width(ax, "credit directions kept", PT_SMALL,
                              boxes[2][2] * f.w_pt)
        if kept:
            f.text((cx3, dot_y - f.fy(LINE_BAND_PT * 0.85)), kept,
                   size=PT_SMALL, color=MUTE, linespacing=1.15)

    cx12 = (boxes[0][0] + boxes[1][0] + boxes[1][2]) / 2.0
    gloss_w = (boxes[1][0] + boxes[1][2]) * f.w_pt
    lines = (("retained task signal", GREEN), ("versus", MUTE),
             ("representation error + admitted noise", AMBER))
    fitted = [(_wrap_to_width(ax, text, PT_SMALL, gloss_w), color)
              for text, color in lines]
    if all(text for text, _ in fitted) and f.room(under, 3 * LINE_BAND_PT):
        # centre the three-line gloss on the dot row so the band reads as one
        # object instead of two blocks separated by a blank stripe
        top_y = dot_y + f.fy(LINE_BAND_PT)
        for row, (text, color) in enumerate(fitted):
            f.text((cx12, top_y - f.fy(row * LINE_BAND_PT)), text,
                   size=PT_SMALL, color=color, linespacing=1.15)
    elif fitted[0][0] and f.room(under, LINE_BAND_PT):
        f.text((cx12, under[1] + under[3] / 2.0), fitted[0][0],
               size=PT_SMALL, color=GREEN, linespacing=1.15)
    return ax


# ── Figure 5 ─────────────────────────────────────────────────────────────
def draw_physical_depth(ax, *, labels=True, scale=1.0):
    """Matched resources, ordered task factors and the point control."""
    f = Frame(ax, labels=labels, scale=scale)
    a, b, c = f.split(3, gap_pt=10.0)
    for cell, color in ((a, "shunting"), (b, "oracle"), (c, "point_mlp")):
        f.group(cell, tint=_tint(color), edge=mix(color, 45))

    core_a = f.cell_text(a, title="physical stage count", title_color=GREEN,
                         foot="same forward budget")
    tag_pt = LINE_BAND_PT if f.room(core_a, MIN_CORE_PT + LINE_BAND_PT) else 0.0
    stage_h = f.stage_height(core_a[3] - f.fy(tag_pt), core_a[2] * 0.28)
    for index, depth in enumerate((1, 2, 3)):
        x = core_a[0] + core_a[2] * (0.22 + 0.28 * index)
        f.stage_tree((x, core_a[1] + f.fy(tag_pt)), stage_h, depth)
        if tag_pt:
            f.text((x, core_a[1] + f.fy(tag_pt * 0.42)),
                   rf"$D_{{\mathrm{{p}}}}={depth}$", size=PT_SMALL)

    core_b = f.cell_text(b, title="multiplicative task hierarchy",
                         title_color=PURPLE,
                         foot=r"$y^{*}=s\,G_1G_2G_3$", foot_size=PT_ANNOT,
                         foot_color=INK, min_core_pt=34.0)
    ys = [core_b[1] + core_b[3] * frac for frac in (0.84, 0.50, 0.16)]
    for idx, (y, color, label) in enumerate(
            zip(ys, (GREEN, BLUE, PURPLE),
                ("coarse factor", "intermediate factor", "fine factor"),
                strict=True), 1):
        box = (core_b[0] + core_b[2] * 0.08, y - f.fy(5.5),
               core_b[2] * 0.84, f.fy(11.0))
        f.group(box, tint="white", edge=color, lw=LW_EDGE, radius_pt=2.0,
                zorder=1.0)
        f.text((core_b[0] + core_b[2] / 2.0, y), rf"$G_{idx}$  {label}",
               size=PT_SMALL, color=color)
        if idx < 3:
            f.arrow((core_b[0] + core_b[2] / 2.0, y - f.fy(6.0)),
                    (core_b[0] + core_b[2] / 2.0, ys[idx] + f.fy(6.0)),
                    head=4.5)

    core_c = f.cell_text(c, title="representation-matched control",
                         title_color=MUTE, foot="same routed fields")
    hub = (core_c[0] + core_c[2] / 2.0, core_c[1] + core_c[3] * 0.22)
    centres = [(core_c[0] + core_c[2] * frac, core_c[1] + core_c[3] * 0.78)
               for frac in (0.24, 0.50, 0.76)]
    for x, y in centres:
        f.leader((x, y - f.fy(5.0)), (hub[0], hub[1] + f.fy(5.0)))
    for index, (x, y) in enumerate(centres):
        f.disc((x, y), 5.4, fill="white", edge=MUTE, lw=LW_EDGE, zorder=4)
        f.text((x, y), rf"$z_{index + 1}$", size=PT_SMALL)
    f.disc(hub, 6.0, fill=COLORS["panel_bg"], edge=MUTE, lw=LW_EDGE, zorder=4)
    f.text(hub, r"$\sigma$", size=PT_ANNOT)
    return ax


# ── Figure 6 ─────────────────────────────────────────────────────────────
def draw_physical_generalization(ax, *, labels=True, scale=1.0):
    """Second-hierarchy depth saturation and the task-family boundary."""
    f = Frame(ax, labels=labels, scale=scale)
    left, right = f.split(2, gap_pt=12.0)
    f.group(left, tint=_tint("shunting"), edge=mix("shunting", 45))
    f.group(right, tint=_tint("oracle"), edge=mix("oracle", 45))

    core_l = f.cell_text(
        left, title="depth saturation", title_color=GREEN,
        subtitle=r"second hierarchy: $H=4$",
        foot="does the benefit saturate beyond the matched order?")
    tag_pt = LINE_BAND_PT if f.room(core_l, MIN_CORE_PT + LINE_BAND_PT) else 0.0
    stage_h = f.stage_height(core_l[3] - f.fy(tag_pt), core_l[2] * 0.28)
    for index, depth in enumerate((2, 3, 4)):
        x = core_l[0] + core_l[2] * (0.22 + 0.28 * index)
        f.stage_tree((x, core_l[1] + f.fy(tag_pt)), stage_h, depth)
        if tag_pt:
            f.text((x, core_l[1] + f.fy(tag_pt * 0.42)),
                   rf"$D_{{\mathrm{{p}}}}={depth}$", size=PT_SMALL)

    core_r = f.cell_text(
        right, title=r"task family $\times$ alignment", title_color=PURPLE,
        foot=r"sensor alignment $\alpha:\ 0 \rightarrow 1$")
    xs = [core_r[0] + core_r[2] * frac for frac in (0.22, 0.50, 0.78)]
    profiles = ((0.20, 0.27, 0.42), (0.25, 0.31, 0.32), (0.37, 0.31, 0.25))
    names = ("nested factors", "flat factors", "local ratios")
    colors = (GREEN, BLUE, MUTE)
    # Each name may only claim its own third of the cell, or the three
    # direct labels collide with one another as the cell narrows.
    fitted = [_wrap_to_width(ax, name, PT_SMALL, core_r[2] * f.w_pt / 3.2)
              for name in names]
    name_pt = 0.0
    if all(fitted) and f.room(core_r, MIN_CORE_PT + LINE_BAND_PT):
        name_pt = LINE_BAND_PT * (1 + max(n.count("\n") for n in fitted))
    plot_h = core_r[3] - f.fy(name_pt)
    y0 = core_r[1] + plot_h * 0.16
    span = plot_h * 0.62
    for x, color, profile, name in zip(xs, colors, profiles, fitted,
                                       strict=True):
        ys = [y0 + span * (p / 0.42) for p in profile]
        px = [x - core_r[2] * 0.09, x, x + core_r[2] * 0.09]
        ax.plot(px, ys, color=color, lw=f.lw(LW_DATA), marker="o",
                ms=f.ms(2.6), mfc=color, mec="none", zorder=4,
                solid_capstyle="round")
        if name_pt:
            f.text((x, core_r[1] + core_r[3] - f.fy(name_pt * 0.5)), name,
                   size=PT_SMALL, color=color, linespacing=1.15)
    f.arrow((xs[0], core_r[1] + f.fy(2.0)), (xs[2], core_r[1] + f.fy(2.0)),
            head=5.0)
    return ax


# ── Figure 7 ─────────────────────────────────────────────────────────────
def draw_anatomy_pipeline(ax, *, labels=True, scale=1.0):
    """Reconstruction -> ancestry route dictionary -> wiring economics."""
    f = Frame(ax, labels=labels, scale=scale)
    a, b, c = f.split(3, gap_pt=14.0)
    for cell in (a, b, c):
        f.group(cell, tint=COLORS["panel_bg"], edge=GRID)

    core_a = f.cell_text(a, title="reconstructed arbor",
                         foot="mapped E/I contacts")
    f.tree(Frame.inset(core_a, left=0.06, right=0.06, top=0.04, bottom=0.04),
           mode="plain", labels=False)

    core_b = f.cell_text(b, title="ancestry route dictionary",
                         title_color=GREEN,
                         foot="nested subtrees define addresses")
    f.tree(Frame.inset(core_b, left=0.06, right=0.06, top=0.04, bottom=0.04),
           mode="address", K=4, labels=False)

    core_c = f.cell_text(c, title="capacity per wire", title_color=PURPLE,
                         foot="model- and density-matched gains",
                         min_core_pt=26.0)
    bars = (("dense", 0.94, PURPLE), ("ancestry", 0.64, GREEN),
            ("control", 0.37, MUTE))
    x_bar = core_c[0] + core_c[2] * 0.42
    span = core_c[2] * 0.52
    for i, (label, frac, color) in enumerate(bars):
        y = core_c[1] + core_c[3] * (0.78 - 0.28 * i)
        f.text((x_bar - f.fx(3.0), y), label, size=PT_SMALL, color=color,
               ha="right")
        # 2026-09-08: a bar is an AREA mark, so it is a filled patch, not a
        # 3 pt stroke (nothing strokes above 1.35 pt any more).
        tint_patch(ax, ("rect", x_bar, y - f.fy(1.5), span * frac, f.fy(3.0)),
                   color=color, face=color, edge=False, radius_pt=1.5,
                   zorder=3, clip_on=False)
    mid = a[1] + a[3] * 0.52
    f.arrow((a[0] + a[2] + f.fx(2.0), mid), (b[0] - f.fx(2.0), mid))
    f.arrow((b[0] + b[2] + f.fx(2.0), mid), (c[0] - f.fx(2.0), mid))
    return ax


# ── Figure 8 ─────────────────────────────────────────────────────────────
def draw_focal_shunt(ax, *, labels=True, scale=1.0, formula=True):
    """A local conductance edit changes only the descendant adjoint field."""
    f = Frame(ax, labels=labels, scale=scale)
    foot_pt = f.footer(
        r"conductance edits $G$, and therefore "
        r"$q=G^{-\mathsf{T}}\nabla_V\mathcal{L}$",
        band_pt=13.0) if formula else 0.0
    left, right = f.split(2, gap_pt=14.0, pad_pt=(0, 0, 0, foot_pt))
    f.group(left, tint=COLORS["panel_bg"], edge=GRID)
    f.group(right, tint=_tint("shunting"), edge=mix("shunting", 45))
    core_l = f.cell_text(left, title="matched additive control",
                         title_color=BLUE, foot="same local operating point")
    core_r = f.cell_text(right, title="focal shunting conductance",
                         title_color=GREEN,
                         foot="descendant transport changes selectively",
                         foot_color=GREEN)
    f.tree(Frame.inset(core_l, left=0.08, right=0.08, top=0.03, bottom=0.03),
           mode="shunt", shunted=False)
    f.tree(Frame.inset(core_r, left=0.08, right=0.08, top=0.03, bottom=0.03),
           mode="shunt", shunted=True)
    mid = core_l[1] + core_l[3] * 0.52
    f.arrow((left[0] + left[2] + f.fx(2.0), mid), (right[0] - f.fx(2.0), mid))
    return ax


# ── Figure 9 ─────────────────────────────────────────────────────────────
def draw_alignment_boundary(ax, *, labels=True, scale=1.0):
    """Availability, imposed alignment, then learning and animal evidence."""
    f = Frame(ax, labels=labels, scale=scale)
    foot_pt = f.footer("availability  →  controlled sufficiency  →  "
                       "evidence for endogenous use")
    cells = f.split(3, gap_pt=14.0, pad_pt=(0, 0, 0, foot_pt))
    entries = (
        ("measured responses", "no morphology-specific alignment", "plain",
         MUTE, "point_mlp"),
        ("imposed alignment", "same dictionary, task field rotated",
         "address", GREEN, "shunting"),
        ("learning and animal tests", "sufficiency versus endogenous use",
         "gain", PURPLE, "oracle"),
    )
    for cell, (title, subtitle, mode, color, key) in zip(cells, entries,
                                                         strict=True):
        f.group(cell, tint=_tint(key), edge=mix(key, 45))
        core = f.cell_text(cell, title=title, title_color=color, foot=subtitle)
        f.tree(Frame.inset(core, left=0.08, right=0.08, top=0.04,
                           bottom=0.04),
               mode=mode, K=4, labels=False)
    mid = cells[0][1] + cells[0][3] * 0.52
    for a, b in zip(cells, cells[1:]):
        f.arrow((a[0] + a[2] + f.fx(2.0), mid), (b[0] - f.fx(2.0), mid),
                head=4.5)
    return ax


# ── data-panel furniture ─────────────────────────────────────────────────
def reference_line(ax, value, *, axis="y", label="chance", color=None,
                   span=None, zorder=1):
    """Dashed mute LW_REF chance / zero reference with a right-aligned label.

    ``axis='y'`` draws a horizontal line at ``value`` (label sitting on the
    line at its right end); ``axis='x'`` draws a vertical line (label
    rotated along its top).  ``span`` limits the line to a data range so
    its dashes never strike through a label beyond the data.  ``ax`` may
    be an Axes or a Frame.
    """
    ax = getattr(ax, "ax", ax)
    color = MUTE if color is None else COLORS.get(color, color)
    if axis == "y":
        lo, hi = ax.get_xlim() if span is None else span
        line, = ax.plot([lo, hi], [value, value], color=color, lw=LW_REF,
                        zorder=zorder, solid_capstyle="butt")
        line.set_dashes((2.2, 1.8))
        if label:
            ax.annotate(label, xy=(hi, value), xytext=(0.0, 1.4),
                        textcoords="offset points", fontsize=PT_SMALL,
                        color=color, ha="right", va="bottom", zorder=5,
                        annotation_clip=False)
    else:
        lo, hi = ax.get_ylim() if span is None else span
        line, = ax.plot([value, value], [lo, hi], color=color, lw=LW_REF,
                        zorder=zorder, solid_capstyle="butt")
        line.set_dashes((2.2, 1.8))
        if label:
            ax.annotate(label, xy=(value, hi), xytext=(-1.4, 0.0),
                        textcoords="offset points", fontsize=PT_SMALL,
                        color=color, ha="right", va="top", rotation=90,
                        zorder=5, annotation_clip=False)
    return line


# ── figure-specific compositions ─────────────────────────────────────────
_STAGE_BRANCHING = {1: [8], 2: [2, 4], 3: [2, 1, 2]}
_STAGE_TIERS_D1 = (("fine", 4), ("coarse", 2), ("global", 2))


def draw_stage_pair(ax, *, depths=(1, 3), sensors=("fine", "coarse", "global"),
                    labels=True):
    """Fig. 6A: the same eight compartments in one versus three serial stages.

    Two task cards (hero = the deeper arrangement) drawn with
    :meth:`Frame.stage_tree` branching [8] and [2, 1, 2], the shared soma
    glyph, inh sensor contacts tagged by tier (distal tier first), the δ0
    arrow at the soma and a shared-versus-exact rule key beneath.
    """
    f = Frame(ax, labels=labels)
    rules = [("shared", "scalar", "scalar"), ("exact", "bp", "exact")]
    key_pt = 12.0 if (labels and f.w_pt >= 120.0 and f.h_pt >= 95.0) else 0.0
    cells = f.split(len(depths), axis="x", gap_pt=10.0,
                    pad_pt=(0, 0, 0, key_pt))
    tiers = list(sensors)
    for cell, depth in zip(cells, depths):
        hero = depth == max(depths)
        stages = _STAGE_BRANCHING.get(depth, [2] * depth)
        core = f.task_card(cell, title=f"D{depth}  [{', '.join(map(str, stages))}]",
                           footer=f"{'one' if depth == 1 else 'three' if depth == 3 else depth} "
                                  f"physical stage{'s' if depth > 1 else ''}",
                           emphasis=hero)
        core = Frame.inset(core, left=0.04, right=0.04)
        # the soma keeps 14 pt below it for the δ0 arrow and tag
        base = (core[0] + core[2] * (0.34 if depth > 1 else 0.5),
                core[1] + f.fy(15.0))
        tag_band = 2 * LINE_BAND_PT if (depth == 1 and labels) else 4.0
        height = max(0.0, core[3] - f.fy(15.0 + tag_band + 2.0))
        width = core[2] * (0.46 if depth > 1 else 0.86)
        tree = f.stage_tree(base, height, depth, branching=stages,
                            width=width, rings=True)
        # sensors: distal tier on the last stage, proximal on the first
        if depth == 1:
            comps = tree[0]
            xs = sorted(comps, key=lambda p: p[0])
            k = 0
            rows_right = [-1e9, -1e9]      # two staggered tag rows
            for tier, count in _STAGE_TIERS_D1:
                group = xs[k:k + count]
                k += count
                for (px, py) in group:
                    f.contact(_lerp(base, (px, py), 0.62), kind="inh")
                if tier in tiers and labels:
                    cx = (group[0][0] + group[-1][0]) / 2.0
                    wpt = _text_w_pt(ax, tier, PT_SMALL)
                    left = cx * f.w_pt - wpt / 2.0
                    for row_i, right in enumerate(rows_right):
                        if left > right + 2.0:
                            f.text((cx, group[0][1] + f.fy(JUNCTION_R_PT + 2.5
                                                            + row_i * LINE_BAND_PT)),
                                   tier, size=PT_SMALL, color=COLORS["inh"],
                                   va="bottom")
                            rows_right[row_i] = left + wpt
                            break
        else:
            per_stage = list(reversed(tiers))[:len(tree)]   # global first
            for s, comps in enumerate(tree):
                tier = per_stage[s] if s < len(per_stage) else None
                # contacts sit on the incoming segment of every compartment
                for i, (px, py) in enumerate(comps):
                    par = base if s == 0 else tree[s - 1][i // max(1, stages[s])]
                    f.contact(_lerp(par, (px, py), 0.62), kind="inh")
                if tier and labels:
                    right = max(p[0] for p in comps)
                    if f._fits(tier, PT_SMALL,
                               (core[0] + core[2] - right) * f.w_pt - 6.0):
                        f.text((right + f.fx(5.0), comps[0][1]), tier,
                               size=PT_SMALL, color=COLORS["inh"], ha="left")
        f.error_in(base, side="right")
    if key_pt:
        f.rule_key((0.0, 0.0, 1.0, f.fy(key_pt)), rules)
    f.require_soma_lowest()
    f.require_delta0()          # one δ0 per soma, in every composition
    return ax


_OPERATOR_BADGES = {
    "pairwise": {"JLL": "×", "JLR": "×", "JRL": "×", "JRR": "×",
                 "JL": "+", "JR": "+", "J1": "+"},
    "quartic": {"JLL": "×", "JLR": "×", "JRL": "×", "JRR": "×",
                "JL": "×", "JR": "×", "J1": "+"},
}


def draw_operator_tree_pair(ax, *, kinds=("pairwise", "quartic"), labels=True):
    """Fig. 4A: two targets on one shared tree, as operator badges.

    Each card is a :meth:`Frame.balanced_tree` with + / × badges at the
    junctions, x1..x8 inputs entering the terminals from above, the soma
    output, the δ0 entry arrow and a 'teacher' badge.
    """
    f = Frame(ax, labels=labels)
    cells = f.split(len(kinds), axis="x", gap_pt=10.0)
    for cell, kind in zip(cells, kinds):
        core = f.task_card(cell, title=kind)
        badge_w = _text_w_pt(ax, "teacher", PT_SMALL) + 9.0
        title_w = _text_w_pt(ax, kind, PT_ANNOT) + 10.0
        if labels and cell[2] * f.w_pt >= badge_w + title_w:
            f.badge((cell[0] + cell[2] - f.fx(4.0), cell[1] + cell[3] - f.fy(3.5)),
                    "teacher")
        body = Frame.inset(core, left=0.05, right=0.05)
        body = (body[0], body[1] + f.fy(15.0), body[2], body[3] - f.fy(16.0))
        nodes = f.balanced_tree(
            body, mode="forward", badges=_OPERATOR_BADGES.get(kind, {}),
            input_labels=[("x", str(i)) for i in range(1, 9)], output="y")
        f.error_in(nodes.soma, side="right")
    f.require_soma_lowest()
    f.require_delta0()
    return ax


def draw_local_gate_tree(ax, *, labels=True):
    """Fig. 5A: the seven-compartment tree with one inhibited proximal gate.

    Four terminals with excitatory inputs, two proximal compartments each
    wearing the context gate ring (the right one closed: its terminals
    faded), the amber 'unit proximal credit' drops into both proximal
    compartments, the soma output and the δ0 entry arrow.
    """
    f = Frame(ax, labels=labels)
    core = f.task_card((0.0, 0.0, 1.0, 1.0), title="local distal gate",
                       footer="unit proximal credit; terminal credit gated by its parent")
    body = Frame.inset(core, left=0.06, right=0.06)
    body = (body[0], body[1] + f.fy(15.0), body[2],
            body[3] - f.fy(15.0 + LINE_BAND_PT))
    nodes = f.balanced_tree(body, depth=2, trunk=False, mode="forward")
    f.gate(nodes["JL"], closed=False, nodes=nodes, node="JL")
    f.gate(nodes["JR"], closed=True, descendants=["JR"], nodes=nodes,
           node="JR")
    # the bus is named in the card's own footer ("unit proximal credit"), so
    # the source tag is suppressed here: two tags in one 21 pt band collide
    f.credit_delivery(nodes, mode="scalar", targets=["JL", "JR"], label=False)
    f.error_in(nodes.soma, side="right")
    if labels:
        half = nodes.pitch_pt * 2.0
        top = max(nodes[t][1] for t in nodes.terminals)
        for names, tag, colour in ((("T1", "T2"), "uninhibited",
                                    label_color(COLORS["shunting"])),
                                   (("T3", "T4"), "inhibited", COLORS["gate"])):
            if f._fits(tag, PT_SMALL, half + 6.0):
                cx = (nodes[names[0]][0] + nodes[names[1]][0]) / 2.0
                f.text((cx, top + f.fy(CONTACT_DIA_PT * 0.5 + 2.5)), tag,
                       size=PT_SMALL, color=colour, va="bottom")
    f.require_soma_lowest()
    f.require_delta0()
    return ax


def _default_route_matrix():
    """Broadcast plus three nested ancestry routes over the 8 terminals."""
    A = np.zeros((8, 4))
    A[:, 0] = 1.0
    A[0:4, 1] = 1.0          # JL subtree
    A[0:2, 2] = 1.0          # JLL subtree
    A[4:6, 3] = 1.0          # JRL subtree
    return A


def draw_measured_boundary(ax, *, route_matrix=None, labels=True):
    """Fig. 9 schematic card: reconstructed arbor, realized routes, δ imposed.

    The tree icon carries the mapped inputs, the realized route support
    matrix sits beside it (:meth:`Frame.dictionary_matrix` measured=True)
    and a dashed δ arrow at the soma marks the field that is imposed, not
    observed.
    """
    f = Frame(ax, labels=labels)
    A = _default_route_matrix() if route_matrix is None else np.asarray(route_matrix)
    core = f.task_card((0.0, 0.0, 1.0, 1.0), title="measured responses",
                       footer="δ imposed, not observed")
    left = Frame.inset(core, left=0.03, right=0.46)
    right = Frame.inset(core, left=0.60, right=0.05)
    body = (left[0], left[1] + f.fy(15.0), left[2], left[3] - f.fy(16.0))
    nodes = f.balanced_tree(body, mode="inputs")
    f.error_in(nodes.soma, side="right", label="δ", dashed=True)
    n, k = A.shape
    avail_w = right[2] * f.w_pt
    avail_h = right[3] * f.h_pt - 14.0
    cell = max(6.0, min(8.0, avail_w / k, avail_h / n))
    mw, mh = cell * k, cell * n
    rx = right[0] + (right[2] - f.fx(mw)) / 2.0
    ry = right[1] + f.fy(12.0) + (right[3] - f.fy(mh + 12.0)) / 2.0
    f.dictionary_matrix((rx, ry, f.fx(mw), f.fy(mh)), A, measured=True,
                        label="A")
    f.require_soma_lowest()
    f.require_delta0()
    return ax


COMPOSITIONS = {
    "stage_pair": draw_stage_pair,
    "operator_tree_pair": draw_operator_tree_pair,
    "local_gate_tree": draw_local_gate_tree,
    "measured_boundary": draw_measured_boundary,
}


SCHEMATICS = {
    "ownership_address": draw_ownership_address,
    "route_resolution": draw_route_resolution,
    "credit_operator": draw_credit_operator,
    "physical_depth": draw_physical_depth,
    "physical_generalization": draw_physical_generalization,
    "anatomy_pipeline": draw_anatomy_pipeline,
    "focal_shunt": draw_focal_shunt,
    "alignment_boundary": draw_alignment_boundary,
}


def _regression_sheets():
    """Draw all eight schematics on native full-width canvases."""
    from pathlib import Path

    from figure_canvas import Margins, native_figure, save_native

    sheets = {
        "native_schematics_regression_a": ["ownership_address",
                                           "route_resolution",
                                           "physical_depth",
                                           "anatomy_pipeline"],
        "native_schematics_regression_b": ["physical_generalization",
                                           "focal_shunt",
                                           "alignment_boundary",
                                           "credit_operator"],
    }
    out = Path(__file__).resolve().parents[1] / "figures" / "generated"
    problems = []
    for stem, names in sheets.items():
        fig, axes = native_figure(
            6.6, [[(name, 12)] for name in names],
            vgutter_pt=14.0,
            margins=Margins(left=8.0, right=8.0, top=8.0, bottom=8.0),
            letters=False,
            panel_kw={name: {"schematic": True} for name in names},
        )
        for name, ax in axes.items():
            SCHEMATICS[name](ax)
        problems += save_native(fig, out / f"{stem}.pdf")
    return problems


if __name__ == "__main__":
    _regression_sheets()

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

The eight builders keep the ``build_main_figure_schematics`` component names:
ownership/address (Fig 2), route resolution (Fig 3), credit operator (Fig 4),
physical depth (Fig 5), physical generalization (Fig 6), anatomy pipeline
(Fig 7), focal shunt (Fig 8) and alignment boundary (Fig 9).
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch

import credit_tree_schematics as _ct
from credit_tree_schematics import draw_credit_tree, mix
from figure_canvas import enforce_tokens, snap_stroke_pt
from journal_style import (
    COLORS,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    PT_ANNOT,
    PT_LABEL,
    PT_LEGEND,
    PT_SMALL,
)

INK = COLORS["ink"]
MUTE = COLORS["mute"]
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

    def stage_tree(self, xy, height, depth, *, color=None, root_r_pt=2.4):
        """Small balanced serial tree: one physical processing stage."""
        color = GREEN if color is None else color
        cx, base = xy
        self.disc((cx, base), root_r_pt, fill=COLORS["soma"], edge=INK,
                  lw=LW_EDGE, zorder=4)
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
        ax.plot([x_bar, x_bar + span * frac], [y, y], color=color,
                lw=3.0, solid_capstyle="round", zorder=3)
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

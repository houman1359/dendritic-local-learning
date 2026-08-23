#!/usr/bin/env python3
"""Native full-width canvas framework for the journal main figures.

Every main figure is authored as ONE matplotlib canvas at exactly the
canonical width (``journal_style.FIG_W`` inches = 518.4 pt) and emitted at
scale 1.0.  Nothing is ever rescaled afterwards, so a nominal 7.6 pt tick
label is 7.6 pt in the compiled PDF of every figure, an ``LW_EDGE`` spine is
0.7 pt everywhere, and two panels in one row are the same physical size.
This module owns that geometry, hard-enforces the journal token set on the
artists it produces, and audits the emitted PDF.

Three entry points
------------------
``native_figure(height_in, layout, ...)``
    Row-list convenience wrapper: ``layout=[["A", "B"], [("C", 7), ("D", 5)]]``
    where a bare name splits the 12-column module evenly and ``(name, span)``
    claims ``span`` modules.  Returns ``(fig, {name: Axes})``.

``NativeCanvas``
    The engine: explicit ``panel(name, row, col, colspan, rowspan)`` placement
    on one 12-column module grid with a single horizontal gutter and a single
    vertical gutter, panel letters in a fixed left gutter, and
    ``save(path)`` which locks the reserves, enforces the tokens, runs the
    inherited layout and text-over-data audits, embeds a geometry manifest
    and writes PDF + PNG.

Column-locked reserves (the default)
------------------------------------
A panel does not carve the space for its own y label and tick column out of
its own slot -- that is exactly what makes two panels of one grid column
start at different ``x0`` and end up different widths.  Instead
``NativeCanvas.lock_reserves()`` measures what every panel's decorations
need, takes the MAXIMUM per grid column (and per grid row, vertically),
subtracts what the uniform gutter or the outer margin already provides, and
applies that single reserve to every panel of the column.  So every panel
starting in grid column *c* shares one ``x0`` and one axes width, and every
panel of grid row *r* shares one ``y0`` and one axes height, with no
per-figure hand tuning.  It runs automatically inside ``save()``; pass
``NativeCanvas(..., lock_reserves=False)`` to disable it figure-wide or
``panel(..., lock=False)`` for the one panel that genuinely must differ.

``audit_native_pdf(path)``
    Post-hoc audit of a compiled PDF: type census, stroke census, content
    fill ratios, internal blank bands, ink at the canvas edge, and -- read
    off the embedded geometry manifest -- the layout contract: per-row panel
    heights, column-locked ``x0`` and widths, row-locked ``y0`` and heights,
    module-normalised panel emphasis and the per-panel aspect band.  Also
    runnable::

        python scripts/figure_canvas.py --audit figures/main/figure_02.pdf

Two deliberate exemptions in the audit, both reported in the informational
buckets rather than silently ignored:

* strokes at least ``DECORATIVE_LW_PT`` (2.5 pt) wide are area marks drawn as
  fat round-capped strokes (route capsules, bar-like rules), not line weights;
* math glyphs shrink by 0.7 per sub/superscript level, and stretchy accents
  and delimiters (STIXSize* fonts) are scaled to the expression they cover,
  so ``token * 0.7**k`` sizes and STIXSize spans are legal type, not a second
  type scale.  ``strict=True`` removes the mathtext exemption.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import journal_style as J
from journal_style import (  # noqa: F401 - re-exported for figure builders
    COLORS,
    DIV_CMAP,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    MARKERS,
    PANEL_LABEL_PT,
    PT_ANNOT,
    PT_LABEL,
    PT_LEGEND,
    PT_SMALL,
    PT_TICK,
    PT_TITLE,
    SEED_ALPHA,
    SEED_MS,
    SEQ_CMAP,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    wrap_ticklabels,
)


# ── the token set (the single source of truth for the audit) ─────────────
CANVAS_W_PT = 72.0 * FIG_W                       # 518.4 pt, exactly
PT_TOKENS = (PT_SMALL, PT_ANNOT, PT_LEGEND, PT_TICK, PT_LABEL, PT_TITLE,
             PANEL_LABEL_PT)                     # 6.8 7.2 7.4 7.6 8.4 8.8 10.5
LW_TOKENS = (LW_HAIR, LW_EDGE, LW_REF, LW_ERR, LW_DATA)  # .55 .7 .85 .95 1.25
MODULE_COLS = 12
DECORATIVE_LW_PT = 2.5      # at or above this a stroke is an area mark
PT_TOL = 0.05               # pt
LW_TOL = 0.02               # pt
MATH_SHRINK = 0.7           # matplotlib mathtext sub/superscript factor

ASPECT_MIN, ASPECT_MAX = 1.05, 1.55
FILL_W_MIN, FILL_H_MIN = 0.92, 0.90

# ── the layout contract (column-locked reserves) ─────────────────────────
ALIGN_TOL_PT = 0.5          # a column lock / row lock is exact to this
EMPHASIS_MAX_RATIO = 1.35   # module-normalised panel area spread
PANEL_ASPECT_MIN = 0.70     # axes-box w/h band for a panel sharing its row
PANEL_ASPECT_MAX = 2.40    # a wide categorical dot plot (fig 09 C,D at 2.26)
                           # is a normal shape; 2.20 flagged those as letterboxes
                           # while still catching genuine strips (fig 04 G was 3.84)
PANEL_BAND_ASPECT_MAX = 4.00  # a full-width synthesis band alone on its row
# A forest/synthesis band that spans the whole canvas alone on its row is a
# standard display: capping it at 3.20 forced figure 09 panel A to leave ~78 pt
# of white beside it, trading a ragged left edge for a ragged right one.  4.00
# admits a full-width band at the row heights these figures use.
RESERVE_PAD_PT = 1.5        # breathing room added to a measured reserve
RESERVE_MAX_FRAC = 0.42     # a lock may never eat more of a slot than this
LETTER_BAND_PT = 13.0       # the panel letter's own band in the top gutter
BLANK_BAND_PT = 14.0
EDGE_CLEAR_PT = 2.0
CELL_FILL_MIN = 0.85        # schematic drawing vs its own cell

MANIFEST_KEY = "keywords"
MANIFEST_SCHEMA = "native-canvas/1"


# ── default geometry, in points ──────────────────────────────────────────
@dataclass(frozen=True)
class Margins:
    """Outer margins in points: minimal, but never clipping."""

    left: float = 34.0      # y label + tick labels + the letter gutter
    right: float = 12.0     # last x tick label overhang
    top: float = 16.0       # panel letter / panel title band
    bottom: float = 26.0    # x tick labels + x label

    def as_dict(self) -> dict:
        return {"left": self.left, "right": self.right,
                "top": self.top, "bottom": self.bottom}


HGUTTER_PT = 30.0           # one horizontal gutter, figure-wide
VGUTTER_PT = 32.0           # one vertical gutter, figure-wide
LETTER_DX_PT = 28.0         # letter sits in the gutter, left of the tick band
LETTER_DY_PT = 2.5          # baseline above the axes top


def snap_font_pt(value: float) -> float:
    """Snap a font size to the journal type scale (letters included)."""
    return min(PT_TOKENS, key=lambda t: abs(t - float(value)))


def snap_stroke_pt(value: float) -> float:
    """Snap a line width to the journal weights; leave area marks alone."""
    v = float(value)
    if v <= 0 or v >= DECORATIVE_LW_PT:
        return v
    return min(LW_TOKENS, key=lambda t: abs(t - v))


def is_token_pt(value: float) -> bool:
    return any(abs(value - t) <= PT_TOL for t in PT_TOKENS)


def is_token_lw(value: float) -> bool:
    return any(abs(value - t) <= LW_TOL for t in LW_TOKENS)


# ── style enforcement ────────────────────────────────────────────────────
def style_panel(ax, *, grid="none", spines=("left", "bottom"),
                schematic=False):
    """Hard-enforce the journal panel look: spines, ticks, grid, type."""
    if schematic:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor("none")
        return ax
    for name, spine in ax.spines.items():
        spine.set_visible(name in spines)
        spine.set_linewidth(LW_EDGE)
        spine.set_color(COLORS["edge"])
    ax.tick_params(axis="both", which="major", direction="out", length=2.8,
                   width=LW_EDGE, pad=1.8, labelsize=PT_TICK,
                   color=COLORS["edge"], labelcolor=COLORS["ink"])
    ax.tick_params(axis="both", which="minor", direction="out", length=1.6,
                   width=LW_HAIR, color=COLORS["edge"])
    if grid in {"x", "y", "both"}:
        ax.grid(True, axis=grid, zorder=0, linewidth=LW_HAIR, alpha=0.9,
                color=COLORS["grid"])
    else:
        ax.grid(False)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_fontsize(PT_LABEL)
    ax.yaxis.label.set_fontsize(PT_LABEL)
    ax.xaxis.label.set_color(COLORS["ink"])
    ax.yaxis.label.set_color(COLORS["ink"])
    return ax


def enforce_tokens(fig, *, fonts=True, strokes=True):
    """Snap every text size and every line weight in ``fig`` to the tokens.

    Runs immediately before saving, so a builder that inherits a stray
    ``markeredgewidth=0.35`` from a helper library still emits a compliant
    figure.  Area marks (>= ``DECORATIVE_LW_PT``) keep their width.
    """
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.text import Text

    for artist in _walk(fig):
        if fonts and isinstance(artist, Text):
            size = artist.get_fontsize()
            if size and not is_token_pt(size):
                artist.set_fontsize(snap_font_pt(size))
        if not strokes:
            continue
        if isinstance(artist, Line2D):
            artist.set_linewidth(snap_stroke_pt(artist.get_linewidth()))
            mew = artist.get_markeredgewidth()
            if mew:
                artist.set_markeredgewidth(snap_stroke_pt(mew))
        elif isinstance(artist, Patch):
            lw = artist.get_linewidth()
            if lw:
                artist.set_linewidth(snap_stroke_pt(lw))
        elif isinstance(artist, Collection):
            try:
                widths = artist.get_linewidths()
            except Exception:
                continue
            if widths is None or len(np.atleast_1d(widths)) == 0:
                continue
            artist.set_linewidth([snap_stroke_pt(w)
                                  for w in np.atleast_1d(widths)])
    return fig


def _walk(artist, _seen=None):
    """Every artist in the tree, once."""
    if _seen is None:
        _seen = set()
    if id(artist) in _seen:
        return
    _seen.add(id(artist))
    yield artist
    try:
        children = artist.get_children()
    except Exception:
        return
    for child in children:
        yield from _walk(child, _seen)


def _reserve(need_pt, avail_pt, pad_pt, cap_pt):
    """The extra space a column/row must carve out of its own slots.

    ``need_pt`` is what the widest label in that column hangs outside its
    axes box; ``avail_pt`` is what the grid already offers there (the outer
    margin at the canvas edge, the uniform gutter inside).  Only the
    shortfall is reserved, and never more than ``cap_pt`` of the slot.
    """
    return float(min(max(0.0, need_pt + pad_pt - avail_pt), max(cap_pt, 0.0)))


# ── the canvas ───────────────────────────────────────────────────────────
class NativeCanvas:
    """One full-width figure on one 12-column module grid.

    Panels in the same grid row share an identical axes-box height by
    construction; every column width is an integer number of modules; there
    is exactly one horizontal gutter value and one vertical gutter value.
    """

    def __init__(self, height_in, nrows=1, *, row_weights=None,
                 hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
                 margins=None, module_cols=MODULE_COLS, letters=True,
                 style=True, lock_reserves=True):
        if style:
            apply_neurips_style()
        height_in = float(height_in)
        self.fig = plt.figure(figsize=(FIG_W, height_in))
        self.fig.patch.set_facecolor("white")
        self.width_pt = CANVAS_W_PT
        self.height_pt = 72.0 * height_in
        self.nrows = int(nrows)
        self.module_cols = int(module_cols)
        self.hgutter = float(hgutter_pt)
        self.vgutter = float(vgutter_pt)
        self.margins = margins or Margins()
        weights = ([1.0] * self.nrows if row_weights is None
                   else [float(w) for w in row_weights])
        if len(weights) != self.nrows:
            raise ValueError("row_weights must name one weight per row")
        self._weights = weights
        self.auto_letters = bool(letters)
        self.lock_enabled = bool(lock_reserves)
        self.axes: dict[str, plt.Axes] = {}
        self._records: list[dict] = []
        self._letters: list[dict] = []
        self._satellites: list[dict] = []
        self._letter_i = 0
        self._locks: dict[str, tuple] = {}
        self._locked_once = False
        self.fig.native_canvas = self

        usable_w = self.width_pt - self.margins.left - self.margins.right
        self._module_w = (usable_w - (self.module_cols - 1) * self.hgutter) \
            / self.module_cols
        if self._module_w <= 0:
            raise ValueError("margins and gutter leave no room for modules")
        usable_h = (self.height_pt - self.margins.top - self.margins.bottom
                    - (self.nrows - 1) * self.vgutter)
        if usable_h <= 0:
            raise ValueError("margins and gutters leave no room for rows")
        total = sum(weights)
        self._row_h = [usable_h * w / total for w in weights]
        self._row_top = []
        y = self.margins.top
        for h in self._row_h:
            self._row_top.append(y)
            y += h + self.vgutter

    # -- geometry ---------------------------------------------------------
    def slot_pt(self, row, col, colspan=None, rowspan=1):
        """Slot rectangle (x0, y_top, w, h) in points from the top-left."""
        colspan = self.module_cols if colspan is None else int(colspan)
        if col < 0 or col + colspan > self.module_cols:
            raise ValueError(f"columns {col}..{col + colspan} leave the module")
        if row < 0 or row + rowspan > self.nrows:
            raise ValueError(f"rows {row}..{row + rowspan} leave the grid")
        x0 = self.margins.left + col * (self._module_w + self.hgutter)
        w = colspan * self._module_w + (colspan - 1) * self.hgutter
        y0 = self._row_top[row]
        h = sum(self._row_h[row:row + rowspan]) + (rowspan - 1) * self.vgutter
        return x0, y0, w, h

    def _rect_fig(self, row, col, colspan, rowspan):
        x0, y_top, w, h = self.slot_pt(row, col, colspan, rowspan)
        return [x0 / self.width_pt,
                (self.height_pt - y_top - h) / self.height_pt,
                w / self.width_pt,
                h / self.height_pt]

    def _panel_rect(self, rec):
        """Final axes rectangle: slot, minus the column/row lock, minus the
        panel's own explicit ``inset_pt``, minus whatever the builder later
        nudged by hand (``ax.set_position``, ``slim_colorbar``).

        The lock term is the only one shared across a column, and it is the
        term that makes ``x0`` and the axes width identical for every panel
        that starts in the same grid column.
        """
        x0, y_top, w, h = self.slot_pt(rec["row"], rec["col"],
                                       rec["colspan"], rec["rowspan"])
        y0 = self.height_pt - y_top - h
        inset = rec.get("inset_pt") or (0.0, 0.0, 0.0, 0.0)
        manual = rec.get("manual_pt") or (0.0, 0.0, 0.0, 0.0)
        own = [inset[i] + manual[i] for i in range(4)]
        lock = (self._locks.get(rec["name"], (0.0, 0.0, 0.0, 0.0))
                if rec.get("locked", True) else (0.0, 0.0, 0.0, 0.0))
        # The builder's own claim on a side and the column/row lock are two
        # bids for the SAME reserve, so the panel yields the larger of the
        # two -- never their sum, which would carve a hand-tuned figure
        # twice and leave the panel a sliver.
        left, right, top, bottom = (max(lock[i], own[i]) for i in range(4))
        x0 += left
        w -= (left + right)
        y0 += bottom
        h -= (top + bottom)
        w = max(w, 4.0)
        h = max(h, 4.0)
        return [x0 / self.width_pt, y0 / self.height_pt,
                w / self.width_pt, h / self.height_pt]

    # -- panels -----------------------------------------------------------
    def panel(self, name, row, col, colspan=None, rowspan=1, *, letter=None,
              title=None, grid="none", schematic=False, style=True,
              inset_pt=None, lock=True, **axes_kw):
        """Add one panel and return its Axes.

        ``inset_pt`` shrinks the axes box inside its slot by
        ``(left, right, top, bottom)`` points -- a manual override that the
        column lock does *not* share with the panel's column neighbours, so
        the audit reports it whenever it breaks the column lock.  Prefer
        letting :meth:`lock_reserves` measure the reserve instead.

        ``lock=False`` opts the panel out of the column/row lock entirely,
        for the rare panel that genuinely must differ (a schematic with no
        axes that has to fill its whole slot, a full-bleed synthesis band).
        """
        colspan = self.module_cols if colspan is None else int(colspan)
        rec = {
            "name": str(name), "row": int(row), "col": int(col),
            "colspan": colspan, "rowspan": int(rowspan),
            "schematic": bool(schematic),
            "locked": bool(lock),
            "inset_pt": tuple(float(v) for v in inset_pt) if inset_pt
            else (0.0, 0.0, 0.0, 0.0),
            "manual_pt": (0.0, 0.0, 0.0, 0.0),
        }
        rect = self._panel_rect(rec)
        rec["_placed"] = list(rect)
        ax = self.fig.add_axes(rect, **axes_kw)
        if style:
            style_panel(ax, grid=grid, schematic=schematic)
        if title:
            ax.set_title(title, fontsize=PT_TITLE, color=COLORS["ink"],
                         pad=3.0, fontweight="normal")
        self.axes[name] = ax
        box = ax.get_position()
        rec.update({
            "x0_pt": round(box.x0 * self.width_pt, 3),
            "y0_pt": round(box.y0 * self.height_pt, 3),
            "w_pt": round(box.width * self.width_pt, 3),
            "h_pt": round(box.height * self.height_pt, 3),
        })
        self._records.append(rec)
        if letter is None and self.auto_letters:
            letter = chr(ord("A") + self._letter_i)
        if letter:
            self._letter_i += 1
            self.add_letter(letter, ax)
        return ax

    def add_letter(self, letter, ax, *, dx_pt=LETTER_DX_PT,
                   dy_pt=LETTER_DY_PT):
        """Panel letter in the fixed gutter: the only bold text on the page."""
        box = ax.get_position()
        x = box.x0 * self.width_pt - dx_pt
        y = box.y1 * self.height_pt + dy_pt
        art = self.fig.text(
            max(x, 2.5) / self.width_pt, y / self.height_pt, str(letter),
            fontsize=PANEL_LABEL_PT, fontweight="bold", color=COLORS["ink"],
            ha="left", va="baseline",
        )
        self._letters.append({"letter": str(letter), "art": art, "ax": ax,
                              "dx_pt": float(dx_pt), "dy_pt": float(dy_pt)})
        for rec in self._records:
            if self.axes.get(rec["name"]) is ax:
                rec["_letter"] = True
        return art

    def _sync_letters(self):
        """Keep every panel letter glued to its panel after a lock pass."""
        for item in self._letters:
            box = item["ax"].get_position()
            x = box.x0 * self.width_pt - item["dx_pt"]
            y = box.y1 * self.height_pt + item["dy_pt"]
            item["art"].set_position((max(x, 2.5) / self.width_pt,
                                      y / self.height_pt))

    # -- column-locked reserves -------------------------------------------
    def _record_for(self, panel):
        """The record behind a panel name or a panel Axes."""
        for rec in self._records:
            if rec["name"] == panel or self.axes.get(rec["name"]) is panel:
                return rec
        raise KeyError(f"no panel {panel!r} on this canvas")

    def declare_reserve(self, panel, *, left=0.0, right=0.0, top=0.0,
                        bottom=0.0):
        """Declare space a panel needs that measuring cannot see.

        A colorbar rail, a category-label column drawn as text rather than as
        tick labels, a legend rail: declare it and it joins the per-column /
        per-row maximum like a measured reserve, so the whole column keeps
        one ``x0`` and one width instead of that one panel going narrow.
        """
        rec = self._record_for(panel)
        current = rec.get("declared_pt", (0.0, 0.0, 0.0, 0.0))
        rec["declared_pt"] = tuple(
            max(current[i], float(v))
            for i, v in enumerate((left, right, top, bottom)))
        return self

    def bind_satellite(self, sat_ax, host_ax):
        """Make a non-panel axes (a colorbar rail, an inset) follow its host.

        The lock pass moves panel boxes, so anything a builder placed in
        coordinates derived from a panel box has to be carried along or it
        silently drifts off its panel.
        """
        box = host_ax.get_position()
        self._satellites.append({
            "ax": sat_ax, "host": host_ax,
            "ref": (box.x0, box.y0, box.width, box.height),
        })
        return sat_ax

    def _capture_manual(self):
        """Record whatever the builder nudged by hand since we placed a panel.

        Anything a builder did with ``ax.set_position`` (``slim_colorbar``
        carving a rail, a hand-tuned shared-axis pair) is preserved as a
        per-panel manual term so the lock pass re-derives the box from the
        slot without throwing that work away.
        """
        for rec in self._records:
            base = rec.get("_placed")
            if not base:
                continue
            cur = self.axes[rec["name"]].get_position()
            rec["manual_pt"] = (
                (cur.x0 - base[0]) * self.width_pt,
                ((base[0] + base[2]) - (cur.x0 + cur.width)) * self.width_pt,
                ((base[1] + base[3]) - (cur.y0 + cur.height)) * self.height_pt,
                (cur.y0 - base[1]) * self.height_pt,
            )

    def _measure_needs(self):
        """Points of decoration each panel hangs outside its own axes box.

        ``(left, right, top, bottom)`` per panel: the y tick column, the y
        label and the y offset text on the left; the x tick labels and x
        label below; the title above.
        """
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        scale = 72.0 / self.fig.dpi
        needs = {}
        for rec in self._records:
            ax = self.axes[rec["name"]]
            box = ax.get_position()
            ax_x0 = box.x0 * self.width_pt
            ax_x1 = box.x1 * self.width_pt
            ax_y0 = box.y0 * self.height_pt
            ax_y1 = box.y1 * self.height_pt
            left = right = top = bottom = 0.0
            for axis, horizontal in ((ax.yaxis, True), (ax.xaxis, False)):
                try:
                    bbox = axis.get_tightbbox(renderer)
                except Exception:
                    bbox = None
                if bbox is None or bbox.width <= 0 or bbox.height <= 0:
                    continue
                if horizontal:
                    left = max(left, ax_x0 - bbox.x0 * scale)
                    right = max(right, bbox.x1 * scale - ax_x1)
                else:
                    bottom = max(bottom, ax_y0 - bbox.y0 * scale)
                    top = max(top, bbox.y1 * scale - ax_y1)
            if ax.get_title():
                try:
                    tb = ax.title.get_window_extent(renderer)
                    top = max(top, tb.y1 * scale - ax_y1)
                except Exception:
                    pass
            if rec.get("_letter"):
                top = max(top, LETTER_BAND_PT)
            needs[rec["name"]] = (max(left, 0.0), max(right, 0.0),
                                  max(top, 0.0), max(bottom, 0.0))
        return needs

    def _column_locks(self, needs, pad_pt=RESERVE_PAD_PT):
        """Turn per-panel needs into ONE reserve per grid column and row.

        The reserve is what a panel needs *beyond* the space the grid already
        gives it -- the outer margin at the edge of the canvas, the uniform
        gutter everywhere else -- so a figure whose gutters already hold its
        labels locks nothing and keeps its full module widths.
        """
        left_need: dict[int, float] = {}
        right_need: dict[int, float] = {}
        top_need: dict[int, float] = {}
        bottom_need: dict[int, float] = {}
        # A declared reserve is space INSIDE the slot (a colorbar rail, a
        # label column drawn as artists), so unlike a measured overhang it
        # never spends the gutter or the outer margin.
        declared: dict[str, dict[int, float]] = {
            "left": {}, "right": {}, "top": {}, "bottom": {}}
        for rec in self._records:
            if not rec.get("locked", True):
                continue
            left, right, top, bottom = needs[rec["name"]]
            c0 = rec["col"]
            c1 = rec["col"] + rec["colspan"]
            r0 = rec["row"]
            r1 = rec["row"] + rec["rowspan"] - 1
            left_need[c0] = max(left_need.get(c0, 0.0), left)
            right_need[c1] = max(right_need.get(c1, 0.0), right)
            top_need[r0] = max(top_need.get(r0, 0.0), top)
            bottom_need[r1] = max(bottom_need.get(r1, 0.0), bottom)
            dl, dr, dt, db = rec.get("declared_pt", (0.0, 0.0, 0.0, 0.0))
            for side, key, value in (("left", c0, dl), ("right", c1, dr),
                                     ("top", r0, dt), ("bottom", r1, db)):
                declared[side][key] = max(declared[side].get(key, 0.0), value)

        locks = {}
        for rec in self._records:
            name = rec["name"]
            if not rec.get("locked", True):
                locks[name] = (0.0, 0.0, 0.0, 0.0)
                continue
            c0 = rec["col"]
            c1 = rec["col"] + rec["colspan"]
            r0 = rec["row"]
            r1 = rec["row"] + rec["rowspan"] - 1
            # One gutter is shared by the panel on its left and the panel on
            # its right: the y label / tick column of the panel that STARTS
            # at a boundary claims it first, and the last x tick label of the
            # panel that ENDS there gets what is left.  The same rule runs
            # vertically, with the panel letter's band taken off the top.
            avail_left = self.margins.left if c0 == 0 else self.hgutter
            avail_right = (self.margins.right if c1 == self.module_cols
                           else max(0.0, self.hgutter
                                    - left_need.get(c1, 0.0)))
            avail_bottom = (self.margins.bottom if r1 == self.nrows - 1
                            else self.vgutter)
            if r0 == 0:
                avail_top = self.margins.top
            else:
                avail_top = max(0.0, self.vgutter
                                - bottom_need.get(r0 - 1, 0.0))
            _, _, slot_w, slot_h = self.slot_pt(r0, c0, rec["colspan"],
                                                rec["rowspan"])
            cap_w = RESERVE_MAX_FRAC * slot_w
            cap_h = RESERVE_MAX_FRAC * slot_h
            locks[name] = (
                max(_reserve(left_need.get(c0, 0.0), avail_left, pad_pt,
                             cap_w), min(declared["left"].get(c0, 0.0),
                                         cap_w)),
                max(_reserve(right_need.get(c1, 0.0), avail_right, pad_pt,
                             cap_w), min(declared["right"].get(c1, 0.0),
                                         cap_w)),
                max(_reserve(top_need.get(r0, 0.0), avail_top, pad_pt,
                             cap_h), min(declared["top"].get(r0, 0.0),
                                         cap_h)),
                max(_reserve(bottom_need.get(r1, 0.0), avail_bottom, pad_pt,
                             cap_h), min(declared["bottom"].get(r1, 0.0),
                                         cap_h)),
            )
        return locks

    def _apply_locks(self):
        """Re-place every panel from its slot under the current locks."""
        for rec in self._records:
            ax = self.axes[rec["name"]]
            before = ax.get_position()
            rect = self._panel_rect(rec)
            ax.set_position(rect)
            after = ax.get_position()
            for sat in self._satellites:
                if sat["host"] is not ax:
                    continue
                self._remap_satellite(sat, before, after)
        self._sync_letters()

    @staticmethod
    def _remap_satellite(sat, before, after):
        box = sat["ax"].get_position()
        if before.width <= 0 or before.height <= 0:
            return
        fx0 = (box.x0 - before.x0) / before.width
        fw = box.width / before.width
        fy0 = (box.y0 - before.y0) / before.height
        fh = box.height / before.height
        sat["ax"].set_position([after.x0 + fx0 * after.width,
                                after.y0 + fy0 * after.height,
                                fw * after.width, fh * after.height])
        sat["ref"] = (after.x0, after.y0, after.width, after.height)

    def lock_reserves(self, *, pad_pt=RESERVE_PAD_PT, passes=3):
        """Measure, lock per column and per row, and re-place every panel.

        This is the mechanism that makes alignment structural: after it runs
        every panel starting in grid column *c* has the same ``x0`` and (for
        the same colspan) the same axes width, and every panel of grid row
        *r* has the same ``y0`` and the same axes height.  Idempotent, and
        run automatically by :meth:`save` unless the canvas was built with
        ``lock_reserves=False``.
        """
        if not self._records:
            return {}
        if not self._locked_once:
            self._capture_manual()
            self._locked_once = True
        previous = None
        for _ in range(max(1, int(passes))):
            locks = self._column_locks(self._measure_needs(), pad_pt=pad_pt)
            self._locks = locks
            self._apply_locks()
            if previous is not None and all(
                    max(abs(a - b) for a, b in zip(locks[k], previous[k]))
                    <= 0.05 for k in locks):
                break
            previous = locks
        self.fig.canvas.draw()
        return self._locks

    # -- row helpers ------------------------------------------------------
    def row_axes(self, row):
        return [self.axes[r["name"]] for r in self._records
                if r["row"] == row]

    def match_row_heights(self, row=None):
        """Force every panel of a row onto the row's shared axes-box height.

        Panels created through :meth:`panel` already share it; this repairs a
        row whose members were nudged by a caller.
        """
        rows = range(self.nrows) if row is None else [row]
        for r in rows:
            records = [rec for rec in self._records
                       if rec["row"] == r and rec["rowspan"] == 1]
            if len(records) < 2:
                continue
            target = min(rec["h_pt"] for rec in records)
            for rec in records:
                ax = self.axes[rec["name"]]
                box = ax.get_position()
                h = target / self.height_pt
                ax.set_position([box.x0, box.y1 - h, box.width, h])
                rec["h_pt"] = round(h * self.height_pt, 3)
                rec["y0_pt"] = round((box.y1 - h) * self.height_pt, 3)

    # -- output -----------------------------------------------------------
    PUBLIC_RECORD_KEYS = ("name", "row", "col", "colspan", "rowspan",
                          "x0_pt", "y0_pt", "w_pt", "h_pt", "schematic",
                          "locked")

    def manifest(self):
        panels = []
        for rec in self._records:                 # honour post-hoc nudges
            box = self.axes[rec["name"]].get_position()
            rec["x0_pt"] = round(box.x0 * self.width_pt, 3)
            rec["y0_pt"] = round(box.y0 * self.height_pt, 3)
            rec["w_pt"] = round(box.width * self.width_pt, 3)
            rec["h_pt"] = round(box.height * self.height_pt, 3)
            panels.append({k: rec[k] for k in self.PUBLIC_RECORD_KEYS
                           if k in rec})
        return {
            "schema": MANIFEST_SCHEMA,
            "width_pt": round(self.width_pt, 3),
            "height_pt": round(self.height_pt, 3),
            "hgutter_pt": self.hgutter,
            "vgutter_pt": self.vgutter,
            "module_cols": self.module_cols,
            "margins_pt": self.margins.as_dict(),
            "reserves_locked": bool(self.lock_enabled),
            "panels": panels,
        }

    def save(self, path, *, name=None, png=True, dpi=600, quiet=False,
             lock=None):
        """Lock the reserves, then write the PDF (+PNG).

        ``lock`` overrides the canvas-wide setting for this one save.
        """
        if self.lock_enabled if lock is None else lock:
            self.lock_reserves()
        return save_native(self.fig, path, manifest=self.manifest(),
                           name=name, png=png, dpi=dpi, quiet=quiet)


def native_figure(height_in, layout, **kwargs):
    """Row-list wrapper around :class:`NativeCanvas`.

    ``layout`` is a list of rows; each row is a list of panel specs, each a
    bare ``name`` (the 12 modules split evenly across the row) or a
    ``(name, colspan)`` pair.  ``None`` reserves an empty slot.  Returns
    ``(fig, {name: Axes})``; the canvas itself is ``fig.native_canvas``.
    """
    row_weights = kwargs.pop("row_weights", None)
    panel_kw = kwargs.pop("panel_kw", {})
    canvas = NativeCanvas(height_in, len(layout), row_weights=row_weights,
                          **kwargs)
    for r, row in enumerate(layout):
        specs = []
        bare = [s for s in row if not isinstance(s, (tuple, list))]
        if bare and len(bare) != len(row):
            raise ValueError("mix names and (name, span) pairs in one row "
                             "only by giving every entry a span")
        if bare:
            if canvas.module_cols % len(row):
                raise ValueError(
                    f"{len(row)} equal panels do not divide "
                    f"{canvas.module_cols} modules; give explicit spans")
            span = canvas.module_cols // len(row)
            specs = [(s, span) for s in row]
        else:
            specs = [(s[0], int(s[1])) for s in row]
        if sum(span for _, span in specs) != canvas.module_cols:
            raise ValueError(f"row {r} spans do not sum to "
                             f"{canvas.module_cols} modules")
        col = 0
        for nm, span in specs:
            if nm is not None:
                canvas.panel(nm, r, col, span, **panel_kw.get(nm, {}))
            col += span
    canvas.fig.native_canvas = canvas
    return canvas.fig, canvas.axes


def save_native(fig, path, *, manifest=None, name=None, png=True, dpi=600,
                quiet=False):
    """Enforce the tokens, audit the layout, and write PDF (+600 dpi PNG).

    The canvas geometry is exact, so nothing here re-crops the page: what the
    builder laid out in points is what the PDF contains.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    label = name or path.stem
    canvas = getattr(fig, "native_canvas", None)
    if manifest is None and canvas is not None:
        if getattr(canvas, "lock_enabled", False):
            canvas.lock_reserves()
        manifest = canvas.manifest()
    width_in, height_in = fig.get_size_inches()
    if abs(width_in - FIG_W) > 1e-6:
        raise ValueError(
            f"{label}: canvas is {width_in:.3f} in wide; every main figure "
            f"must be authored at exactly {FIG_W} in")
    fig.canvas.draw()
    enforce_tokens(fig)
    fig.canvas.draw()
    problems = list(audit_layout(fig, label))
    problems += list(audit_text_over_data(fig, label))
    meta = {"Creator": "dendritic-local-learning", "CreationDate": None}
    if manifest is not None:
        meta["Keywords"] = json.dumps(manifest, separators=(",", ":"))
    fig.savefig(path, format="pdf", facecolor="white", metadata=meta)
    if png:
        fig.savefig(path.with_suffix(".png"), format="png", dpi=dpi,
                    facecolor="white")
    if not quiet:
        print(f"  wrote {path.name} "
              f"({width_in * 72:.1f} x {height_in * 72:.1f} pt, "
              f"aspect {width_in / height_in:.2f})")
    return problems


def slim_colorbar(fig, ax, mappable, *, label="", width_pt=4.5, pad_pt=3.0,
                  tick_allow_pt=15.0, **kwargs):
    """Labelled colorbar in a rail carved out of the host panel's own slot.

    Every heatmap needs a key, and a colorbar bolted onto the right of a panel
    silently widens it into the gutter and collides with the next panel's y
    label.  Taking the rail out of the panel keeps the grid intact.

    ``tick_allow_pt`` is only the opening bid for the rail width: after the
    bar is drawn the rail is measured and widened until the tick labels *and*
    the colorbar label sit inside the host panel's original slot.  Without
    that correction a long label on the right-most panel of a row silently
    runs off the canvas edge -- exactly the class of defect the audit's
    ``edge-clearance`` check reports.
    """
    fw_pt = fig.get_size_inches()[0] * 72.0
    box = ax.get_position()               # the slot the panel owns right now
    slot_x1_pt = box.x1 * fw_pt

    def place(reserve_pt):
        reserve = reserve_pt / fw_pt
        ax.set_position([box.x0, box.y0, box.width - reserve, box.height])
        rail_x0 = box.x0 + box.width - reserve + pad_pt / fw_pt
        return rail_x0, width_pt / fw_pt

    reserve_pt = pad_pt + width_pt + tick_allow_pt
    rail_x0, rail_w = place(reserve_pt)
    cax = fig.add_axes([rail_x0, box.y0, rail_w, box.height])
    canvas = getattr(fig, "native_canvas", None)
    if canvas is not None:
        canvas.bind_satellite(cax, ax)
    cbar = fig.colorbar(mappable, cax=cax, **kwargs)
    cbar.outline.set_linewidth(LW_EDGE)
    cbar.outline.set_edgecolor(COLORS["edge"])
    cbar.ax.tick_params(labelsize=PT_TICK, width=LW_EDGE, length=2.2, pad=1.5,
                        color=COLORS["edge"], labelcolor=COLORS["ink"])
    if label:
        cbar.set_label(label, fontsize=PT_LABEL, labelpad=2.5,
                       color=COLORS["ink"])

    for _ in range(4):                    # measure, widen the rail, repeat
        try:
            fig.canvas.draw()
            bbox = cax.get_tightbbox(fig.canvas.get_renderer())
        except Exception:
            break
        overflow_pt = bbox.x1 / fig.dpi * 72.0 - slot_x1_pt
        if overflow_pt <= 0.25:
            break
        reserve_pt += overflow_pt + 0.5
        rail_x0, rail_w = place(reserve_pt)
        cax.set_position([rail_x0, box.y0, rail_w, box.height])
    if canvas is not None:
        # The rail is a right-hand reserve like any other: declare it so the
        # column lock gives every panel of that column the same width instead
        # of leaving this one narrower than its neighbours.
        try:
            canvas.declare_reserve(ax, right=reserve_pt)
        except KeyError:
            pass
    return cbar


# ── audit ────────────────────────────────────────────────────────────────
@dataclass
class Violation:
    kind: str
    detail: str
    value: float | None = None

    def __str__(self) -> str:
        return f"[{self.kind}] {self.detail}"


@dataclass
class AuditReport:
    path: Path
    violations: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    def __iter__(self):
        return iter(self.violations)

    def __len__(self):
        return len(self.violations)


def _page_ink(page, dpi=150.0):
    """Boolean ink mask of a rendered page plus its pt-per-pixel scale."""
    import fitz

    zoom = dpi / 72.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom),
                          colorspace=fitz.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.height, len(img) // pix.height)[:, :pix.width]
    return img < 250, zoom


def _blank_runs(has_ink, lo, hi):
    """(start, stop) index pairs of blank runs strictly inside [lo, hi]."""
    runs = []
    start = None
    for i in range(lo, hi + 1):
        if not has_ink[i]:
            start = i if start is None else start
        elif start is not None:
            runs.append((start, i))
            start = None
    return runs


def _is_math_glyph(span) -> bool:
    """Stretchy accent / delimiter drawn from a STIXSize extension font."""
    if "STIXSize" in str(span.get("font", "")):
        return True
    text = span.get("text", "").strip()
    return bool(text) and all(unicodedata.category(c) == "Mn" for c in text)


TEXT_OVERLAP_FRAC = 0.30    # of the smaller span's area
TEXT_SHRINK = 0.14          # trim each bbox by this fraction of its height


def _overlapping_spans(spans):
    """Pairs of text spans that genuinely sit on top of one another.

    Each bbox is trimmed vertically before the test because PDF span boxes
    carry the font's full ascent/descent, so two comfortably-set lines of
    prose touch without a reader ever seeing a collision.  Spans of one
    rendered line are skipped (kerned neighbours legitimately abut) and so
    are sub/superscripts, which sit inside their base span's box by design.
    """
    boxes = []
    for span in spans:
        x0, y0, x1, y1 = span["bbox"]
        pad = TEXT_SHRINK * (y1 - y0)
        boxes.append((x0, y0 + pad, x1, y1 - pad))
    out = []
    for i in range(len(spans)):
        for j in range(i + 1, len(spans)):
            if spans[i]["_line"] == spans[j]["_line"]:
                continue
            si, sj = float(spans[i]["size"]), float(spans[j]["size"])
            if abs(si - sj) > 0.05 and min(si, sj) <= MATH_SHRINK * max(si, sj) \
                    + 0.15:
                continue                      # sub/superscript inside its base
            ax0, ay0, ax1, ay1 = boxes[i]
            bx0, by0, bx1, by1 = boxes[j]
            iw = min(ax1, bx1) - max(ax0, bx0)
            ih = min(ay1, by1) - max(ay0, by0)
            if iw <= 0 or ih <= 0:
                continue
            area_a = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
            area_b = max((bx1 - bx0) * (by1 - by0), 1e-6)
            if iw * ih >= TEXT_OVERLAP_FRAC * min(area_a, area_b):
                out.append((spans[i], spans[j]))
    return out


def _mathtext_size_ok(size, strict=False):
    if strict:
        return False
    for token in PT_TOKENS:
        for level in (1, 2, 3):
            if abs(size - token * MATH_SHRINK ** level) <= max(PT_TOL,
                                                               0.02 * size):
                return True
    return False


def audit_native_pdf(path, *, strict=False, page_index=0, report=False,
                     dpi=150.0):
    """Audit one compiled figure PDF against the journal contract.

    Returns the list of :class:`Violation` (``report=True`` returns the full
    :class:`AuditReport` with the informational notes attached).
    """
    import fitz

    path = Path(path)
    doc = fitz.open(path)
    page = doc[page_index]
    out = AuditReport(path=path)
    V = out.violations.append
    N = out.notes.append

    # 1. canvas
    w, h = page.rect.width, page.rect.height
    if abs(w - CANVAS_W_PT) > 0.5:
        V(Violation("canvas-width",
                    f"page is {w:.1f} pt wide, must be {CANVAS_W_PT:.1f} pt",
                    w))
    aspect = w / h if h else 0.0
    if not (ASPECT_MIN - 1e-9 <= aspect <= ASPECT_MAX + 1e-9):
        V(Violation("canvas-aspect",
                    f"aspect {aspect:.2f} outside "
                    f"{ASPECT_MIN:.2f}-{ASPECT_MAX:.2f} "
                    f"({w:.1f} x {h:.1f} pt)", aspect))

    # 2. type census
    sizes: dict[float, list] = {}
    spans: list[dict] = []
    for bi, block in enumerate(page.get_text("dict")["blocks"]):
        for li, line in enumerate(block.get("lines", [])):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text.strip():
                    continue
                if _is_math_glyph(span):
                    N(f"math extension glyph at {span['size']:.2f} pt "
                      f"({span.get('font', '?')})")
                    continue
                spans.append({**span, "_line": (bi, li)})
                sizes.setdefault(round(float(span["size"]), 2), []).append(text)
    for size in sorted(sizes):
        samples = sizes[size]
        if is_token_pt(size):
            continue
        if _mathtext_size_ok(size, strict):
            N(f"mathtext-shrunk type at {size:.2f} pt "
              f"({len(samples)} spans, e.g. {samples[0].strip()!r})")
            continue
        sample = max(samples, key=lambda s: len(s.strip())).strip()
        V(Violation("text-size",
                    f"{size:.2f} pt is not a type token "
                    f"({len(samples)} spans, e.g. {sample[:40]!r})", size))

    # 2b. text piled on text
    #
    # The inherited ``audit_layout`` compares an artist against *other panels*
    # and ``audit_text_over_data`` compares text against data marks, so
    # neither sees two labels of one schematic landing on each other -- the
    # exact failure a schematic shows when its cell is shorter than the label
    # bands it reserves.  Catch it on the compiled page, where the collision
    # is a fact rather than a prediction.
    for a, b in _overlapping_spans(spans):
        V(Violation("text-collision",
                    f"{a['text'].strip()[:34]!r} overlaps "
                    f"{b['text'].strip()[:34]!r}"))

    # 3. stroke census
    widths: dict[float, int] = {}
    for drawing in page.get_drawings():
        if drawing.get("type") not in ("s", "fs"):
            continue
        width = drawing.get("width")
        if width is None:
            continue
        widths[round(float(width), 3)] = widths.get(round(float(width), 3),
                                                    0) + 1
    for width in sorted(widths):
        count = widths[width]
        if width <= 0 or is_token_lw(width):
            continue
        if width >= DECORATIVE_LW_PT:
            N(f"area mark stroked at {width:.2f} pt ({count} paths)")
            continue
        V(Violation("stroke-width",
                    f"{width:.3f} pt is not a line-weight token "
                    f"({count} paths)", width))

    # 4-6. ink geometry
    ink, zoom = _page_ink(page, dpi=dpi)
    rows = ink.any(axis=1)
    cols = ink.any(axis=0)
    if not rows.any():
        V(Violation("empty-page", "no ink on the page"))
        return out if report else out.violations
    r0, r1 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0, c1 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    fill_w = (c1 - c0 + 1) / zoom / w
    fill_h = (r1 - r0 + 1) / zoom / h
    if fill_w < FILL_W_MIN:
        V(Violation("fill-width",
                    f"content fills {fill_w * 100:.1f}% of the canvas width "
                    f"(need {FILL_W_MIN * 100:.0f}%)", fill_w))
    if fill_h < FILL_H_MIN:
        V(Violation("fill-height",
                    f"content fills {fill_h * 100:.1f}% of the canvas height "
                    f"(need {FILL_H_MIN * 100:.0f}%)", fill_h))
    left_pt, right_pt = c0 / zoom, w - (c1 + 1) / zoom
    top_pt, bottom_pt = r0 / zoom, h - (r1 + 1) / zoom
    for side, clear in (("left", left_pt), ("right", right_pt),
                        ("top", top_pt), ("bottom", bottom_pt)):
        if clear < EDGE_CLEAR_PT:
            V(Violation("edge-clearance",
                        f"ink is {clear:.1f} pt from the {side} canvas edge "
                        f"(need {EDGE_CLEAR_PT:.0f} pt)", clear))

    manifest = _read_manifest(doc)
    hgut = manifest.get("hgutter_pt", HGUTTER_PT) if manifest else None
    vgut = manifest.get("vgutter_pt", VGUTTER_PT) if manifest else None
    for axis, has_ink, lo, hi, gutter in (
        ("horizontal", rows, r0, r1, vgut),
        ("vertical", cols, c0, c1, hgut),
    ):
        bands = [(a, b) for a, b in _blank_runs(has_ink, lo, hi)
                 if (b - a) / zoom > BLANK_BAND_PT]
        if not bands:
            continue
        widths_pt = sorted((b - a) / zoom for a, b in bands)
        allow = (gutter + 4.0 if gutter is not None
                 else max(BLANK_BAND_PT, float(np.median(widths_pt)) + 3.0))
        for (a, b) in bands:
            band = (b - a) / zoom
            if band <= allow:
                N(f"{axis} gutter band of {band:.0f} pt "
                  f"at {a / zoom:.0f} pt (within the uniform gutter)")
                continue
            V(Violation("blank-band",
                        f"{band:.0f} pt {axis} blank band at "
                        f"{a / zoom:.0f}-{b / zoom:.0f} pt "
                        f"(gutter allowance {allow:.0f} pt)", band))

    # 7. per-row panel heights (needs the native geometry manifest)
    if not manifest:
        V(Violation("manifest-missing",
                    "no native-canvas manifest: the figure was not built as "
                    "one native full-width canvas, so per-row panel geometry "
                    "cannot be verified"))
    else:
        by_row: dict[int, list] = {}
        for rec in manifest.get("panels", []):
            if rec.get("rowspan", 1) == 1:
                by_row.setdefault(rec.get("row", 0), []).append(rec)
        for row, records in sorted(by_row.items()):
            heights = [rec["h_pt"] for rec in records]
            if max(heights) - min(heights) > 0.75:
                names = ", ".join(f"{rec['name']}={rec['h_pt']:.1f}"
                                  for rec in records)
                V(Violation("row-height",
                            f"row {row} panels differ in axes-box height "
                            f"({names})", max(heights) - min(heights)))
        cells = _cell_fill(page, manifest, ink, zoom)
        for rec, frac in cells:
            if rec.get("schematic") and frac < CELL_FILL_MIN:
                V(Violation("cell-fill",
                            f"schematic panel {rec['name']!r} fills "
                            f"{frac * 100:.0f}% of its cell "
                            f"(need {CELL_FILL_MIN * 100:.0f}%)", frac))
        _audit_layout_contract(manifest, V, N)
    return out if report else out.violations


def _audit_layout_contract(manifest, V, N):
    """The four layout-contract checks, all read off the geometry manifest.

    ``column-alignment``  every panel starting in one grid column shares an
                          ``x0``, and panels of one (column, colspan) share a
                          width -- i.e. the left reserve was locked per
                          column instead of carved per panel;
    ``row-alignment``     panels of one grid row share ``y0`` and height, and
                          a row-spanning panel's top edge sits on the top
                          edge of the row it starts in;
    ``panel-emphasis``    axes area normalised by module span varies by no
                          more than ``EMPHASIS_MAX_RATIO`` across the figure,
                          so a panel is bigger only by spanning more modules;
    ``panel-aspect``      every panel that shares its row keeps an axes box
                          inside the ``PANEL_ASPECT_MIN``-``PANEL_ASPECT_MAX``
                          band -- no letterbox strips, no thin towers.
    """
    panels = list(manifest.get("panels", []))
    if not panels:
        return
    locked = [p for p in panels if p.get("locked", True)]
    opted_out = [p["name"] for p in panels if not p.get("locked", True)]
    if opted_out:
        N(f"panels opted out of the column/row lock: {', '.join(opted_out)}")

    # -- L1: one x0 per grid column, one width per (column, colspan) -------
    by_col: dict[int, list] = {}
    for rec in locked:
        by_col.setdefault(int(rec["col"]), []).append(rec)
    for col, group in sorted(by_col.items()):
        lock_x0 = max(rec["x0_pt"] for rec in group)
        stray = [rec for rec in group
                 if abs(rec["x0_pt"] - lock_x0) > ALIGN_TOL_PT]
        if stray:
            detail = ", ".join(f"{rec['name']}={rec['x0_pt']:.1f}"
                               for rec in stray)
            V(Violation("column-alignment",
                        f"grid column {col} locks x0 at {lock_x0:.1f} pt but "
                        f"{detail} start elsewhere (each panel carved its "
                        f"own left reserve)",
                        max(abs(rec["x0_pt"] - lock_x0) for rec in stray)))
    by_span: dict[tuple, list] = {}
    for rec in locked:
        by_span.setdefault((int(rec["col"]), int(rec["colspan"])),
                           []).append(rec)
    for (col, span), group in sorted(by_span.items()):
        widths = [rec["w_pt"] for rec in group]
        spread = max(widths) - min(widths)
        if spread > ALIGN_TOL_PT:
            detail = ", ".join(f"{rec['name']}={rec['w_pt']:.1f}"
                               for rec in group)
            V(Violation("column-alignment",
                        f"grid column {col} span {span}: panels differ in "
                        f"axes-box width by {spread:.1f} pt ({detail})",
                        spread))

    # -- L2: one y0 and one height per grid row ---------------------------
    by_row: dict[int, list] = {}
    for rec in locked:
        if int(rec.get("rowspan", 1)) == 1:
            by_row.setdefault(int(rec["row"]), []).append(rec)
    for row, group in sorted(by_row.items()):
        for key, label in (("y0_pt", "axes-box bottom"),
                           ("h_pt", "axes-box height")):
            values = [rec[key] for rec in group]
            spread = max(values) - min(values)
            if spread <= ALIGN_TOL_PT:
                continue
            detail = ", ".join(f"{rec['name']}={rec[key]:.1f}"
                               for rec in group)
            V(Violation("row-alignment",
                        f"row {row} panels differ in {label} by "
                        f"{spread:.1f} pt ({detail})", spread))
    for row, group in sorted(by_row.items()):
        by_row_span: dict[int, list] = {}
        for rec in group:
            by_row_span.setdefault(int(rec["colspan"]), []).append(rec)
        for span, mates in sorted(by_row_span.items()):
            widths = [rec["w_pt"] for rec in mates]
            spread = max(widths) - min(widths)
            if spread <= ALIGN_TOL_PT:
                continue
            detail = ", ".join(f"{rec['name']}={rec['w_pt']:.1f}"
                               for rec in mates)
            V(Violation("row-alignment",
                        f"row {row}: panels spanning {span} modules differ "
                        f"in axes-box width by {spread:.1f} pt ({detail}); a "
                        f"width difference must be a whole module span",
                        spread))
    for rec in locked:
        if int(rec.get("rowspan", 1)) <= 1:
            continue
        siblings = by_row.get(int(rec["row"]), [])
        if not siblings:
            continue
        top = rec["y0_pt"] + rec["h_pt"]
        row_top = max(s["y0_pt"] + s["h_pt"] for s in siblings)
        if abs(top - row_top) > ALIGN_TOL_PT:
            V(Violation("row-alignment",
                        f"row-spanning panel {rec['name']!r} tops out at "
                        f"{top:.1f} pt, {abs(top - row_top):.1f} pt off the "
                        f"row {rec['row']} top edge ({row_top:.1f} pt)",
                        abs(top - row_top)))

    # -- L3: modest emphasis, measured per module -------------------------
    # Normalise by the SLOT a panel is allocated, not by a raw module count:
    # a span-n panel also absorbs n-1 gutters, so dividing by n alone credits
    # wide panels with area they never had and made a legitimate full-width
    # forest band (figure 09 A) read as 1.59x "too big".  Comparing each panel
    # against its own allocation measures what the rule actually wants: is any
    # panel disproportionately large for the space it was given.
    _cols = max(int(manifest.get("module_cols", 12) or 12), 1)
    _marg = manifest.get("margins_pt", {}) or {}
    _hg = float(manifest.get("hgutter_pt", 0.0) or 0.0)
    _vg = float(manifest.get("vgutter_pt", 0.0) or 0.0)
    _avail_w = (float(manifest.get("width_pt", 0.0) or 0.0)
                - float(_marg.get("left", 0.0)) - float(_marg.get("right", 0.0))
                - (_cols - 1) * _hg)
    _mod_w = _avail_w / _cols if _cols else 0.0
    _nrows = max((int(r.get("row", 0)) + int(r.get("rowspan", 1))) for r in panels) if panels else 1
    _avail_h = (float(manifest.get("height_pt", 0.0) or 0.0)
                - float(_marg.get("top", 0.0)) - float(_marg.get("bottom", 0.0))
                - (_nrows - 1) * _vg)
    _row_h = _avail_h / _nrows if _nrows else 0.0

    def _slot_area(rec):
        cs = max(int(rec["colspan"]), 1)
        rs = max(int(rec.get("rowspan", 1)), 1)
        w = cs * _mod_w + (cs - 1) * _hg
        h = rs * _row_h + (rs - 1) * _vg
        return max(w * h, 1e-6)

    density = [(rec, rec["w_pt"] * rec["h_pt"] / _slot_area(rec))
               for rec in panels]
    if len(density) > 1:
        big_rec, big = max(density, key=lambda item: item[1])
        small_rec, small = min(density, key=lambda item: item[1])
        ratio = big / small if small > 0 else float("inf")
        if ratio > EMPHASIS_MAX_RATIO + 1e-9:
            V(Violation("panel-emphasis",
                        f"slot-fill varies {ratio:.2f}x "
                        f"({big_rec['name']} fills {big:.2f} vs "
                        f"{small_rec['name']} {small:.2f} of its slot; "
                        f"allowed {EMPHASIS_MAX_RATIO:.2f}x) -- a panel may "
                        f"be bigger only by spanning more modules",
                        ratio))

    # -- L4: a sane aspect band for every panel that shares its row -------
    row_population: dict[int, int] = {}
    for rec in panels:
        row_population[int(rec["row"])] = \
            row_population.get(int(rec["row"]), 0) + 1
    module_cols = int(manifest.get("module_cols", MODULE_COLS))
    for rec in panels:
        aspect = rec["w_pt"] / max(rec["h_pt"], 1e-6)
        alone = (row_population[int(rec["row"])] <= 1
                 and int(rec["colspan"]) >= module_cols)
        hi = PANEL_BAND_ASPECT_MAX if alone else PANEL_ASPECT_MAX
        if PANEL_ASPECT_MIN - 1e-9 <= aspect <= hi + 1e-9:
            if alone:
                N(f"panel {rec['name']!r} is a full-width band alone on row "
                  f"{rec['row']}: aspect {aspect:.2f} allowed up to "
                  f"{PANEL_BAND_ASPECT_MAX:.2f}")
            continue
        shape = "letterbox strip" if aspect > hi else "thin tower"
        band = (" even for a full-width band alone on its row" if alone
                else "")
        V(Violation("panel-aspect",
                    f"panel {rec['name']!r} is a {shape}{band}: "
                    f"{rec['w_pt']:.0f} x {rec['h_pt']:.0f} pt, aspect "
                    f"{aspect:.2f} outside "
                    f"{PANEL_ASPECT_MIN:.2f}-{hi:.2f}", aspect))


def _cell_fill(page, manifest, ink, zoom):
    """Ink bbox area fraction of each manifest panel's own slot."""
    height_pt = manifest.get("height_pt", page.rect.height)
    out = []
    for rec in manifest.get("panels", []):
        x0 = rec["x0_pt"] * zoom
        x1 = (rec["x0_pt"] + rec["w_pt"]) * zoom
        y1 = (height_pt - rec["y0_pt"]) * zoom
        y0 = (height_pt - rec["y0_pt"] - rec["h_pt"]) * zoom
        sub = ink[max(int(y0), 0):int(y1), max(int(x0), 0):int(x1)]
        if sub.size == 0 or not sub.any():
            out.append((rec, 0.0))
            continue
        rows = sub.any(axis=1)
        cols = sub.any(axis=0)
        r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
        c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
        frac = ((r1 - r0) * (c1 - c0)) / float(sub.shape[0] * sub.shape[1])
        out.append((rec, frac))
    return out


def _read_manifest(doc):
    raw = (doc.metadata or {}).get(MANIFEST_KEY) or ""
    if not raw.strip().startswith("{"):
        return None
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    return data if data.get("schema") == MANIFEST_SCHEMA else None


def _cli(argv=None):
    parser = argparse.ArgumentParser(
        description="Audit compiled figure PDFs against the journal contract")
    parser.add_argument("--audit", nargs="+", required=True, metavar="PDF")
    parser.add_argument("--strict", action="store_true",
                        help="forbid mathtext-shrunk sizes as well")
    parser.add_argument("--notes", action="store_true",
                        help="print the informational buckets too")
    parser.add_argument("--json", action="store_true",
                        help="machine-readable per-file violation counts")
    args = parser.parse_args(argv)

    summary = {}
    counts = {}
    for target in args.audit:
        report = audit_native_pdf(target, strict=args.strict, report=True)
        kinds: dict[str, int] = {}
        for violation in report.violations:
            kinds[violation.kind] = kinds.get(violation.kind, 0) + 1
        counts[str(target)] = len(report.violations)
        summary[str(target)] = {
            "violations": len(report.violations),
            "by_kind": dict(sorted(kinds.items())),
            "details": [str(v) for v in report.violations],
        }
        if not args.json:
            print(f"{Path(target).name}: {len(report.violations)} violations")
            for violation in report.violations:
                print(f"    {violation}")
            if args.notes:
                for note in report.notes:
                    print(f"    (note) {note}")
    if args.json:
        print(json.dumps({"total": sum(counts.values()),
                          "files": summary}, indent=2))
    return 1 if any(counts.values()) else 0


if __name__ == "__main__":
    sys.exit(_cli())

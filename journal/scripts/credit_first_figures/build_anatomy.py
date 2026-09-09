#!/usr/bin/env python3
"""Main Figure 7 -- ancestry routes on reconstructed MICrONS arbors.

Eight lettered panels on one native 12-module canvas (488 pt, three rows,
row weights 124/112/112)::

    row 0   A reconstructed arbor      B seven routes + A_8   C shunt -> field
    row 1   D capture vs K             E K = 8 partition      F cell scatter
    row 2   G residual, three cohorts        H paired advantage + wiring

Row 0 is one left-to-right chain: the same reconstructed cell in A (mapped
E/I balance) and B (ghost skeleton with the seven K = 8 ancestry routes and
the collapsed dictionary), then the paper tree in C which defines the focal
shunt, its response field and the capture functional.  Rows 1 and 2 are the
data: every mean, interval and n comes from the frozen
``source_data/anatomy_commonmode`` tables (and ``source_data/figure3`` for
the skeleton), never from a typed number.

Waivers and local decisions, in one place:

* D3 waiver -- row 1 is a row of three 4-module data panels.  D, E and F all
  plot the same quantity (captured fraction of the weighted response energy)
  against the same six route families, so they are one comparison family
  sharing a y meaning; D carries the y label and the family key for the row.
* D7 -- ``native_schematics.K_CYCLE`` still holds the retired
  shunting/additive/local/oracle capsule hues, so this builder defines the
  D7 cycle (dend, soma, exc, mute, then their 62 % ink mixes) privately as
  ``K_CYCLE_D7`` and uses full strength for route strokes and 16 % for
  capsules.  No architecture, bp, scalar, oracle or highlight hue is used
  for a route.
* ``Frame.arbor`` (D10) does not exist in the shared library, so
  ``_arbor_geometry``/``_draw_arbor`` below are private helpers that lift
  ``build_main_figure_07.morphology_geometry`` and ``panel_arbor`` with the
  Ellipse soma replaced by ``Frame.soma``.  Same for the collapsed-block
  dictionary (``_block_matrix``), because ``Frame.dictionary_matrix`` has no
  ``collapse_blocks``/``min_cell_pt`` argument yet.
* C's field ``t`` is the real operator column for i-site 4396, reduced to the
  eight tree-ordered site blocks of the paper tree by an E-area-weighted
  mean and normalised by max |t| with the sign kept; the blocks are drawn as
  DIV_CMAP squares above the eight terminals rather than as a ninth column,
  which a 4-module cell cannot hold at >= 6 pt.
* B's broadcast column and its soma source dot are tagged ``s`` (amber), not
  the specification's ``1``: the seven routes are numbered 1-7 on the arbor,
  so a ``1`` at the soma would be read as route 1.  C keeps the paper's
  ``A = [1 | r1 r2 r3]`` notation, where the 1 is the column of ones.
* B's spec footer ("rows: tree-ordered site blocks, not to scale") is 146 pt
  at PT_SMALL and a 4-module panel is 124 pt, so it is set in two lines with
  the matrix caption beside it.
* G's oracle badge sits inside the axes beside the violet marker it names
  rather than outside it: a 62 pt category reserve plus a badge in the
  6-module gutter would push the panel past its column lock.
* H's x limit runs to 60 pp so the two PT_ANNOT report columns (wiring,
  cells +) live inside the axes; the ticks still stop at 30 pp, and a mute
  hairline separates the columns from the data.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
sys.path.insert(0, str(JOURNAL / "code/reconstructed_tree"))

import build_main_figure_07 as anatomy                        # noqa: E402
from analyze_microns_morphology_credit import (                # noqa: E402
    ancestry_matrix, parent_map)
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR,  # noqa
                           LW_REF, MARKER_MS, Margins, NativeCanvas,
                           PT_ANNOT, PT_LEGEND, PT_SMALL,
                           PT_TITLE as PT_TITLE_, style_panel)
from journal_style import DIV_CMAP, style_direct_color_labels    # noqa: E402
from credit_tree_schematics import mix                           # noqa: E402
from native_schematics import Frame, reference_line              # noqa: E402

SOURCE = JOURNAL / "source_data"
COMMON = SOURCE / "anatomy_commonmode"
RECORDS = SOURCE / "credit_first_figures"
OUT = JOURNAL / "figures/components/credit_first_figure_06.pdf"
MAIN = JOURNAL / "figures/main/figure_07.pdf"

METHODS = ["common + ancestry", "common-constrained SVD",
           "common + surrogate ancestry", "common + depth bins",
           "common + shuffled routes", "common + random routes"]
LABELS = ["Ancestry", "SVD oracle", "Surrogate", "Depth bins",
          "Shuffled", "Random"]
COHORTS = ["original8", "v661", "pinky"]
COHORT_ROWS = [("original8", "Original\n8 cells"),
               ("v661", "Disjoint\n47 cells"),
               ("pinky", "Pinky\n8 cells")]
BOOT_SEED = 202609061
N_BOOT = 20_000
CAPTURE_LABEL = "captured energy fraction"

# One encoding for a route family, shared by D, E, G (and named in the
# caption).  ancestry = the implementable morphology-matched rule (shunting);
# SVD = the oracle ceiling (violet); surrogate tree = a rewired-tree control
# (point_mlp); depth / shuffled / random = randomized or scaffolding controls
# (mute), separated by marker and dash pattern, never by a second hue.
FAMILY = {
    "common + ancestry": ("shunting", "o", True, "-"),
    "common-constrained SVD": ("oracle", "D", True, "-"),
    "common + surrogate ancestry": ("point_mlp", "D", False, (0, (3.2, 2.0))),
    "common + depth bins": ("mute", "s", False, (0, (1.1, 1.5))),
    "common + shuffled routes": ("mute", "v", False, (0, (3.2, 2.0))),
    "common + random routes": ("mute", "^", False, (0, (1.1, 1.5))),
}
# D7 K-cycle: four hues that are never a data series, then their 62 % ink
# mixes for k = 5..8.  Full strength for a real-arbor route stroke, 16 % for
# an addressed-subtree capsule.
K_CYCLE_D7 = ["dend", "soma", "exc", "mute"]
ROOT_B = 864691135409937097
ROUTE_SITE = 4396          # the route whose response field C illustrates


CAPTION = r"""\caption{\textbf{Ancestry routes on reconstructed arbors compress the tree's own focal-shunt response fields.}
\textbf{A}, Reconstructed MICrONS arbor (root 864691135409937097, median-sized of the original eight cells; 78 of 616 segments), principal-plane projection. Hue, mapped $(E-I)/(E+I)$ contact-area balance; stroke width, log mapped area; orange disc, soma; bar, 50~$\mu$m.
\textbf{B}, Same arbor in grey with its seven $K=8$ ancestry routes, numbered in tree order and colored by route; red dots, inhibitory origins; pale capsules, the two largest supports. Supports nest ($4458\subset4396\subset4209$) and hold 29, 5 and five single sites, covering 32 of the cell's 70 excitatory sites. Matrix rows, the eight tree-ordered site blocks with counts at right, not to scale; column $s$, the broadcast.
\textbf{C}, Paper tree: a focal shunt ($g_{\mathrm{shunt}}$) attenuates its descendants (pale strokes), making the field $\bm t$ (squares above the sites, diverging scale; route 4396's measured field, eight blocks). Capsules, the three addressed subtrees, the columns of $A=[\bm 1\,|\,\bm r_1\bm r_2\bm r_3]$; capture is the $W$-weighted energy of the projection $A\bm c$ of $\bm t$. $\delta_0$, somatic error; $z$, output.
\textbf{D}, Total capture against column budget, 47 disjoint v661 cells (46 at $K=16$); cohort means; family by marker and dash; bands, 95\% cell-bootstrap intervals (20,000 draws) for ancestry and surrogates.
\textbf{E}, Energy partition at $K=8$: broadcast, spatial, remainder; 47-cell means; families as in \textbf{D}.
\textbf{F}, Per-cell ancestry capture against its own 200 degree--depth surrogate trees; dashed, equality; open pink, the 11 cells where at least half the surrogates reach it; white diamond, cohort mean with 95\% cell-bootstrap intervals.
\textbf{G}, Residual capture after the broadcast at $K=8$, three cohorts ($n=8$, 47, 8 cells); means with 95\% cell-bootstrap intervals.
\textbf{H}, Paired ancestry-minus-control residual capture, 47 cells; means with retained 95\% cell-bootstrap intervals. The mute SVD-gap row is descriptive, outside the Holm family; right columns, each dictionary's wiring density and cells favoring ancestry. Randomized controls average 200 draws/cell; the SVD is an oracle ceiling; fields are modeled, not observed.}"""


def k_hue(k):
    """Full-strength D7 K-cycle hue for route ``k`` (0-based)."""
    key = K_CYCLE_D7[k % 4]
    return COLORS[key] if k < 4 else mix(key, 62, "ink")


def mean_ci(values, seed):
    """Descriptive cell-bootstrap mean and 95 % interval (20,000 draws)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(N_BOOT, values.size),
                       replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


# ── data loaders (frozen tables only) ────────────────────────────────────
def cohort_tables():
    return {c: pd.read_csv(COMMON / c / "cell_method_summary.csv")
            for c in COHORTS}


def cohort_summaries():
    return {c: pd.read_csv(COMMON / c / "cohort_method_summary.csv")
            for c in COHORTS}


def reports():
    return {c: json.loads((COMMON / c / "summary.json").read_text())
            for c in COHORTS}


def surrogate_pairs():
    """Per v661 cell: ancestry total capture, the 200-surrogate mean, and the
    fraction of surrogate trees that reach or beat the real tree at K = 8."""
    rows = []
    for path in sorted((COMMON / "v661" / "cells").glob("rows_*.csv.gz")):
        table = pd.read_csv(path)
        eight = table[table.channels.eq(8)]
        tree = float(eight[eight.method.eq(METHODS[0])].total_capture.iloc[0])
        draws = eight[eight.method.eq(METHODS[2])].total_capture.to_numpy(float)
        rows.append(dict(root_id=int(table.root_id.iloc[0]), tree=tree,
                         surrogate_mean=float(draws.mean()),
                         n_replicates=int(draws.size),
                         fraction_ge=float((draws >= tree).mean())))
    return pd.DataFrame(rows)


def arbor_routes(n_routes=7):
    """The seven K = 8 ancestry routes of ROOT_B, reproduced from
    ``scripts/anatomy_commonmode/run.py`` lines 111-112."""
    segments = pd.read_csv(SOURCE / "figure3" / "segment_metrics.csv")
    cell = segments[segments.root_id.eq(ROOT_B)].copy()
    _, parent, children = parent_map(cell)
    npz = np.load(COMMON / "original8" / "cells" / f"operator_{ROOT_B}.npz")
    e_sites = [int(v) for v in npz["e_sites"]]
    i_sites = [int(v) for v in npz["i_sites"]]
    beta = np.asarray(npz["beta"], float)
    weights = np.asarray(npz["weights"], float)
    ancestry = ancestry_matrix(e_sites, i_sites, parent)
    order = np.argsort(-(beta * (ancestry.T @ weights) / weights.sum()))
    chosen = [int(j) for j in order[:n_routes]]
    origins = [i_sites[j] for j in chosen]
    assert origins == [4784, 4621, 4396, 4458, 4975, 4209, 4516], origins
    supports = [ancestry[:, j] > 0 for j in chosen]
    assert [int(s.sum()) for s in supports] == [1, 1, 5, 1, 1, 29, 1]
    # depth-first tree order: the row order of every collapsed matrix here
    sequence, stack = [], [min(cell.segment_id)]
    root_seg = int(cell.loc[cell.parent_segment_id < 0, "segment_id"].iloc[0])
    stack = [root_seg]
    while stack:
        node = stack.pop()
        sequence.append(node)
        stack.extend(sorted(children.get(node, []), reverse=True))
    rank = {seg: k for k, seg in enumerate(sequence)}
    return dict(cell=cell, parent=parent, children=children, npz=npz,
                e_sites=e_sites, i_sites=i_sites, weights=weights,
                origins=origins, supports=supports, rank=rank,
                site_rank=np.asarray([rank[s] for s in e_sites]))


def _arbor_geometry(cell):
    """Isotropic principal-plane projection of one reconstruction.

    Lifted verbatim from ``build_main_figure_07.morphology_geometry`` so the
    three real-arbor drawings of the paper share one projection (D10).
    """
    xyz = cell[["x_um", "y_um", "z_um"]].to_numpy(float)
    centred = xyz - xyz.mean(axis=0, keepdims=True)
    _, _, basis = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ basis[:2].T
    span_um = max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
    projected /= span_um
    positions = {int(seg): point for seg, point
                 in zip(cell.segment_id.to_numpy(int), projected, strict=True)}
    rows = {int(row.segment_id): row for row in cell.itertuples(index=False)}
    parent = {int(row.segment_id): int(row.parent_segment_id)
              for row in cell.itertuples(index=False)}
    return positions, rows, parent, span_um


# ── private glyph helpers (missing from the shared library) ──────────────
def _fit_iso(f, rect, xy, pad_pt=2.0):
    """Isotropic map of projected coordinates into ``rect`` (frame fractions)."""
    xy = np.asarray(xy, float)
    x0, y0, w, h = rect
    avail_x = w * f.w_pt - 2 * pad_pt
    avail_y = h * f.h_pt - 2 * pad_pt
    span_x = max(np.ptp(xy[:, 0]), 1e-9)
    span_y = max(np.ptp(xy[:, 1]), 1e-9)
    scale = min(avail_x / span_x, avail_y / span_y)
    ox = x0 + (w - f.fx(span_x * scale)) / 2.0
    oy = y0 + (h - f.fy(span_y * scale)) / 2.0
    xmin, ymin = xy[:, 0].min(), xy[:, 1].min()

    def place(point):
        return (ox + f.fx((point[0] - xmin) * scale),
                oy + f.fy((point[1] - ymin) * scale))

    return place, scale


def _draw_arbor(f, rect, cell, *, mode="balance", routes=None,
                capsules=(), scale_bar_um=50.0, hues=None):
    """The one real-arbor drawing (D10) as a private helper.

    ``mode='balance'`` paints every segment on the anatomy E/I ramp with the
    five stroke tokens as a mapped-area ladder (panel A); ``mode='ghost'``
    draws the whole skeleton in GHOST and then the ``routes`` -- lists of
    segment ids -- in full-strength D7 K-cycle strokes, with 16 % capsules
    over the ``capsules`` (segment ids, tint, stroke width) blocks.  Returns
    the placement map and the soma point.
    """
    positions, rows, parent, span_um = _arbor_geometry(cell)
    place, _ = _fit_iso(f, rect, np.asarray(list(positions.values()), float))
    xy = {seg: place(p) for seg, p in positions.items()}
    e_area = cell.E_size.to_numpy(float)
    i_area = cell.I_size.to_numpy(float)
    total = e_area + i_area
    balance = np.divide(e_area - i_area, total, out=np.zeros_like(total),
                        where=total > 0)
    burden = np.log1p(total)
    burden /= max(burden.max(), 1e-12)
    ids = cell.segment_id.to_numpy(int)
    balance_by = dict(zip(ids, balance, strict=True))
    burden_by = dict(zip(ids, burden, strict=True))
    ramp = LinearSegmentedColormap.from_list(
        "contact_balance", [COLORS["inh"], "#DCE0E5", COLORS["exc"]])
    ladder = (LW_HAIR, LW_EDGE, LW_REF, LW_ERR, LW_DATA)
    edges = [(s, parent[s]) for s in rows if parent[s] in rows]
    for seg, par in edges:
        f.ax.plot([xy[seg][0], xy[par][0]], [xy[seg][1], xy[par][1]],
                  color=mix("mute", 45), lw=LW_HAIR, solid_capstyle="round",
                  zorder=1)
    if mode == "balance":
        for seg, par in edges:
            weight = burden_by[seg]
            if weight <= 0:
                continue
            # I-rich proximal trunks sit above the E-rich distal branches so
            # the panel's one red story stays visible where strokes cross.
            f.ax.plot([xy[seg][0], xy[par][0]], [xy[seg][1], xy[par][1]],
                      color=ramp((balance_by[seg] + 1.0) / 2.0),
                      lw=ladder[int(round(weight * (len(ladder) - 1)))],
                      alpha=0.45 + 0.55 * weight, solid_capstyle="round",
                      zorder=3.0 if balance_by[seg] >= 0 else 3.6)
    else:
        # a capsule hugs an addressed SUPPORT (the sites a route owns), the
        # stroke traces the route itself (origin up to the soma)
        for members, tint, size_pt in capsules:
            block = set(members)
            for seg in members:
                if parent.get(seg) not in block:
                    continue
                f.ax.plot([xy[seg][0], xy[parent[seg]][0]],
                          [xy[seg][1], xy[parent[seg]][1]], color=tint,
                          lw=size_pt, solid_capstyle="round",
                          solid_joinstyle="round", zorder=1.4)
        for index, route in enumerate(routes or []):
            colour = hues[index] if hues else k_hue(index)
            for seg in route:
                par = parent.get(seg)
                if par not in rows:
                    continue
                f.ax.plot([xy[seg][0], xy[par][0]], [xy[seg][1], xy[par][1]],
                          color=colour, lw=LW_EDGE, solid_capstyle="round",
                          zorder=3.2 + 0.01 * index)
    soma_id = min(rows, key=lambda key: rows[key].topological_depth)
    f.soma(xy[soma_id], zorder=6)
    if scale_bar_um:  # B reuses A's bar: same cell, same projection
        raw = np.asarray(list(positions.values()), float)
        anchor = np.asarray([raw[:, 0].min(), raw[:, 1].min()])
        p0 = place(anchor)
        p1 = place(anchor + [scale_bar_um / span_um, 0.0])
        bar = p1[0] - p0[0]
        bx, by = rect[0] + f.fx(1.5), rect[1] + f.fy(3.5)
        f.ax.plot([bx, bx + bar], [by, by], color=COLORS["ink"], lw=LW_DATA,
                  solid_capstyle="butt", zorder=7)
        f.text((bx + bar / 2.0, by + f.fy(2.0)), f"{scale_bar_um:.0f} µm",
               size=PT_SMALL, color=COLORS["ink"], va="bottom")
    return xy, xy[soma_id]


def _block_matrix(f, rect, A, *, counts=None, col_colors=None,
                  col_labels=None, row_groups=None, label=None):
    """Collapsed site-block dictionary: one row per block, >= 6 pt tall.

    Stands in for ``Frame.dictionary_matrix(collapse_blocks=True)``, which
    the shared library does not implement.  ``counts`` are printed at the
    right of each row, ``col_labels`` under each column with a
    ``col_colors`` swatch, so the reader can name a route in both drawings.
    """
    A = np.asarray(A, float)
    n, k = A.shape
    inner = f.ax.inset_axes(rect, transform=f.ax.transData, zorder=3)
    inner.imshow((A != 0).astype(float), aspect="auto",
                 interpolation="nearest", vmin=0, vmax=1,
                 cmap=LinearSegmentedColormap.from_list(
                     "support", [COLORS["panel_bg"], COLORS["shunting"]]))
    inner.set_xticks([])
    inner.set_yticks([])
    for spine in inner.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS["edge"])
        spine.set_linewidth(LW_HAIR)
    if row_groups:
        for edge in np.cumsum([int(g) for g in row_groups])[:-1]:
            inner.axhline(edge - 0.5, color=COLORS["edge"], lw=LW_HAIR,
                          zorder=4)
    x0, y0, w, h = rect
    row_h = h / n
    if counts is not None:
        for i, value in enumerate(counts):
            f.text((x0 + w + f.fx(1.6), y0 + h - (i + 0.5) * row_h),
                   str(value), size=PT_SMALL, color=COLORS["ink"], ha="left")
    if col_labels is not None:
        col_w = w / k
        for j, name in enumerate(col_labels):
            cx = x0 + (j + 0.5) * col_w
            if col_colors is not None:
                f.ax.add_patch(Rectangle(
                    (cx - f.fx(1.6), y0 + h + f.fy(1.4)), f.fx(3.2),
                    f.fy(2.2), facecolor=col_colors[j], edgecolor="none",
                    zorder=5))
            f.text((cx, y0 + h + f.fy(4.4)), name, size=PT_SMALL,
                   color=COLORS["ink"], va="bottom")
    if label:
        f.text((x0 + w + f.fx(2.0), y0 + h + f.fy(4.4)), label,
               size=PT_ANNOT, color=COLORS["ink"], ha="left", va="bottom")
    return inner


# ── row 0: the schematics ────────────────────────────────────────────────
def panel_arbor(ax, cell):
    """A: the reconstruction, hue = mapped E/I balance, width = mapped area."""
    f = Frame(ax)
    band = 34.0                                    # key strip + footer bands
    _draw_arbor(f, (0.0, f.fy(band), 1.0, 1.0 - f.fy(band)), cell,
                mode="balance")
    # key strip: the anatomy ramp with its two poles and a signed scale
    bar_w, bar_h = 62.0, 5.0
    bx = (1.0 - f.fx(bar_w)) / 2.0
    by = f.fy(21.0)
    ramp = LinearSegmentedColormap.from_list(
        "contact_balance", [COLORS["inh"], "#DCE0E5", COLORS["exc"]])
    f.ax.imshow(np.linspace(0, 1, 256)[None, :], cmap=ramp, aspect="auto",
                origin="lower", zorder=4,
                extent=(bx, bx + f.fx(bar_w), by, by + f.fy(bar_h)))
    f.ax.add_patch(Rectangle((bx, by), f.fx(bar_w), f.fy(bar_h),
                             facecolor="none", edgecolor=COLORS["edge"],
                             lw=LW_HAIR, zorder=5))
    f.text((bx - f.fx(2.5), by + f.fy(bar_h / 2.0)), "I", size=PT_SMALL,
           color=COLORS["inh"], ha="right")
    f.text((bx + f.fx(bar_w + 2.5), by + f.fy(bar_h / 2.0)), "E",
           size=PT_SMALL, color=COLORS["exc"], ha="left")
    f.text((0.5, f.fy(10.5)), "hue: mapped E/I balance", size=PT_SMALL,
           color=COLORS["mute"], va="bottom")
    f.text((0.5, f.fy(1.5)), "stroke width: mapped contact area",
           size=PT_SMALL, color=COLORS["mute"], va="bottom")
    return ax


def panel_routes(ax, routes):
    """B: the same arbor in ghost with seven ancestry routes and A_8."""
    f = Frame(ax)
    cell, parent = routes["cell"], routes["parent"]
    e_sites, supports = routes["e_sites"], routes["supports"]
    origins, rank = routes["origins"], routes["rank"]
    n_routes = len(origins)
    ids = set(cell.segment_id.to_numpy(int))
    # routes are numbered 1..7 in tree order, so the arbor dots, the matrix
    # columns and the caption all name a route by the same digit
    order = sorted(range(n_routes), key=lambda k: rank[origins[k]])
    slot = {k: i for i, k in enumerate(order)}

    def chain(origin):
        out, cursor = [], int(origin)
        while cursor in parent and parent[cursor] in ids:
            out.append(cursor)
            cursor = parent[cursor]
        out.append(cursor)
        return out

    def support_block(k):
        """Every segment the route's support spans: its sites and the
        ancestors joining them, up to the route origin."""
        origin = int(origins[k])
        block = {origin}
        for site, on in zip(e_sites, supports[k]):
            if not on:
                continue
            cursor = int(site)
            while cursor in ids:
                block.add(cursor)
                if cursor == origin:
                    break
                cursor = parent.get(cursor, -1)
        return sorted(block)

    route_segments = [chain(o) for o in origins]
    big = sorted(range(n_routes), key=lambda k: -int(supports[k].sum()))[:2]
    matrix_w = 6.0 * (n_routes + 1)
    matrix_h = 6.0 * (n_routes + 1)
    arbor_rect = (0.0, f.fy(71.0), 1.0, 1.0 - f.fy(73.0))
    xy, soma = _draw_arbor(f, arbor_rect, cell, mode="ghost",
                           routes=route_segments,
                           capsules=[(support_block(k),
                                      mix(k_hue(slot[k]), 16, "white"), 6.0)
                                     for k in big],
                           hues={k: k_hue(slot[k]) for k in range(n_routes)},
                           scale_bar_um=None)
    # the broadcast column is the soma's own scalar: amber tag at the soma
    tag = (soma[0] + f.fx(4.5), soma[1] - f.fy(2.0))
    f.text(tag, "s", size=PT_SMALL, color=mix("local", 60, "ink"), ha="left")
    # the digits keep clear of the panel title and of the amber soma tag
    ceiling_pt = f.h_pt - 1.0
    floor_pt = 74.0
    taken = [(tag[0] * f.w_pt, tag[1] * f.h_pt)]
    # every route is numbered: the digit takes the freest of four diagonal
    # slots around its origin dot rather than being dropped
    for index, origin in enumerate(origins):
        point = xy[int(origin)]
        f.contact(point, kind="inh", dia_pt=2.9)
        best = fallback = None
        for dx, dy, ha, va in ((3.0, 2.4, "left", "bottom"),
                               (3.0, -2.4, "left", "top"),
                               (-3.0, 2.4, "right", "bottom"),
                               (-3.0, -2.4, "right", "top")):
            px = point[0] * f.w_pt + dx
            py = point[1] * f.h_pt + dy
            room = min(((px - qx) ** 2 + (py - qy) ** 2
                        for qx, qy in taken), default=1e9)
            item = (room, px, py, ha, va)
            if fallback is None or room > fallback[0]:
                fallback = item
            # the digit must stay inside the arbor band: its own box clear
            # of the title above and of the matrix column labels below
            top = py + (9.0 if va == "bottom" else 0.0)
            bottom = py - (9.0 if va == "top" else 0.0)
            if top > ceiling_pt or bottom < floor_pt:
                continue
            if best is None or room > best[0]:
                best = item
        room, px, py, ha, va = best or fallback
        taken.append((px, py))
        f.text((px / f.w_pt, py / f.h_pt), str(slot[index] + 1),
               size=PT_SMALL, color=k_hue(slot[index]), ha=ha, va=va)
    # collapsed dictionary: one row per route support plus the off-route row
    union = np.zeros(len(e_sites), bool)
    for support in supports:
        union |= support
    A = np.zeros((n_routes + 1, n_routes + 1))
    A[:, 0] = 1.0
    counts = []
    for row, k in enumerate(order):
        for j, other in enumerate(order):
            if supports[other][supports[k]].all():
                A[row, j + 1] = 1.0
        counts.append(int(supports[k].sum()))
    counts.append(int((~union).sum()))
    _block_matrix(f, (f.fx(2.0), f.fy(21.0), f.fx(matrix_w), f.fy(matrix_h)),
                  A, counts=counts,
                  col_colors=[mix("local", 55)]
                  + [k_hue(slot[k]) for k in order],
                  col_labels=["s"] + [str(slot[k] + 1) for k in order],
                  row_groups=[1] * n_routes,
                  label=None)
    f.text((1.0, f.fy(21.0 + matrix_h)),
           f"{int(union.sum())} of {len(e_sites)}\nsites lie\non a route",
           size=PT_SMALL, color=COLORS["mute"], ha="right", va="top")
    # the spec footer does not fit a 4-module line at PT_SMALL (146 pt), so
    # it is set left in two lines with the matrix caption on its right
    f.text((0.0, f.fy(2.0)), "rows: tree-ordered site\nblocks, not to scale",
           size=PT_SMALL, color=COLORS["mute"], ha="left", va="bottom",
           linespacing=1.15)
    f.text((1.0, f.fy(2.0)),
           f"A  ({len(e_sites)} \u00d7 {n_routes + 1})", size=PT_ANNOT,
           color=COLORS["ink"], ha="right", va="bottom")
    return ax


def panel_field(ax, field):
    """C: a focal shunt makes a field; capture is its energy in span(A)."""
    f = Frame(ax)
    t, A, coefficients, capture = (field["t"], field["A"], field["c"],
                                   field["capture"])
    tree_rect = (0.0, f.fy(74.0), f.fx(56.0), f.fy(46.0))
    nodes = f.balanced_tree(tree_rect, depth=3, mode="forward", output="z")
    # the focal shunt sits on the trunk of the left subtree; its descendants
    # are the sites the field lives on, so they are the faded partition
    site = ((nodes["J1"][0] + nodes["JL"][0]) / 2.0,
            (nodes["J1"][1] + nodes["JL"][1]) / 2.0)
    f.fade(["JL"], nodes=nodes)
    f.dendrite(nodes["J1"], nodes["JL"], level=nodes.level["JL"], faded=True)
    f.shunt(site)
    f.error_in(nodes.soma, side="right")
    # the three ancestry addresses of A, as nested K-cycle capsules on the
    # same tree, so the reader can read r1, r2, r3 off the drawing
    f.partition(nodes, [nodes.subtree("JL"), nodes.subtree("JLL"),
                        nodes.subtree("JRL")],
                colors=[K_CYCLE_D7[0], K_CYCLE_D7[1], K_CYCLE_D7[2]],
                labels=None)
    for name in nodes.terminals:
        f.contact(nodes[name], kind="exc", dia_pt=2.6)
    # the real field, one DIV_CMAP square per site, above its own terminal
    top = max(nodes[n][1] for n in nodes.terminals)
    scale = mpl.cm.ScalarMappable(mpl.colors.Normalize(-1.0, 1.0), DIV_CMAP)
    for index, name in enumerate(sorted(nodes.terminals,
                                        key=lambda n: nodes[n][0])):
        x = nodes[name][0]
        f.ax.add_patch(Rectangle((x - f.fx(2.2), top + f.fy(3.0)),
                                 f.fx(4.4), f.fy(4.4), lw=LW_HAIR,
                                 facecolor=scale.to_rgba(t[index]),
                                 edgecolor=COLORS["edge"], zorder=5))
    f.text((nodes[min(nodes.terminals, key=lambda n: nodes[n][0])][0]
            - f.fx(4.0), top + f.fy(5.2)), "t", size=PT_ANNOT,
           color=COLORS["ink"], ha="right")
    product = (f.fx(59.0), f.fy(49.0), f.fx(65.0), f.fy(71.0))
    f.dictionary_product(product, A, coefficients, cell_pt=7.5, numbers=False,
                         captions=("A", "c", "A c"),
                         col_colors=["local"] + K_CYCLE_D7[:3])
    f.text((0.0, f.fy(12.5)),
           f"capture = ||A c||² / ||t||² = {capture:.2f}", size=PT_ANNOT,
           color=COLORS["ink"], ha="left", va="bottom")
    f.text((0.0, f.fy(23.0)), "A = [1 | r1 r2 r3]", size=PT_SMALL,
           color=COLORS["ink"], ha="left", va="bottom")
    f.text((0.0, f.fy(2.0)), "W = excitatory contact area", size=PT_SMALL,
           color=COLORS["mute"], ha="left", va="bottom")
    return ax


# ── row 1: the budget, the partition and the per-cell test ───────────────
def panel_budget(ax, summaries, tables):
    """D: total capture against the column budget K, six families."""
    table = summaries["v661"]
    cells = tables["v661"]
    rows = []
    for index, method in enumerate(METHODS):
        colour, marker, filled, dashes = FAMILY[method]
        part = table[table.method.eq(method)].sort_values("channels")
        x = part.channels.to_numpy(float)
        mean = part.total_capture_mean.to_numpy(float)
        if method in (METHODS[0], METHODS[2]):
            lo, hi = [], []
            for k in x:
                subset = cells[cells.channels.eq(int(k))
                               & cells.method.eq(method)]
                _, a, b = mean_ci(subset.total_capture.to_numpy(float),
                                  BOOT_SEED + 10 * index + int(k))
                lo.append(a)
                hi.append(b)
            ax.fill_between(x, lo, hi, color=COLORS[colour], alpha=0.12,
                            linewidth=0, zorder=1.2)
        else:
            lo = hi = [np.nan] * len(x)
        line, = ax.plot(x, mean, color=COLORS[colour], lw=LW_DATA,
                        marker=marker, ms=MARKER_MS,
                        mfc=COLORS[colour] if filled else "white",
                        mec=COLORS[colour], mew=LW_EDGE, zorder=2 + index * .01,
                        label=LABELS[index], solid_capstyle="round")
        if dashes != "-":
            line.set_dashes(dashes[1])
        for k, m, a, b in zip(x, mean, lo, hi):
            rows.append(dict(panel="D", method=method, channels=int(k),
                             total_capture=m, low=a, high=b))
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.86, 18.6)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(0.0, 1.38)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("dictionary columns K")
    ax.set_ylabel(CAPTURE_LABEL)
    ax.plot([8, 8], [0.0, 1.0], color=COLORS["mute"], lw=LW_REF, zorder=0.5,
            dashes=(2.2, 1.8), solid_capstyle="butt")
    ax.text(8.35, 0.035, "K = 8", fontsize=PT_SMALL, color=COLORS["mute"],
            ha="left", va="bottom")
    ax.text(0.015, 0.020, "47 cells; 46 at K = 16", fontsize=PT_SMALL,
            color=COLORS["mute"], ha="left", va="bottom",
            transform=ax.transAxes)
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.002), ncol=2,
              frameon=False, fontsize=PT_LEGEND, handlelength=1.5,
              handletextpad=0.4, columnspacing=0.8, labelspacing=0.28,
              borderaxespad=0.0)
    style_panel(ax)
    return pd.DataFrame(rows)


def panel_partition(ax, summaries):
    """E: at K = 8 the broadcast is shared; the routes add the rest."""
    table = summaries["v661"]
    eight = table[table.channels.eq(8)].set_index("method")
    rows = []
    for index, method in enumerate(METHODS):
        colour, marker, filled, _ = FAMILY[method]
        spatial = float(eight.loc[method, "incremental_total_capture_mean"])
        total = float(eight.loc[method, "total_capture_mean"])
        common = total - spatial
        ax.bar(index, common, width=0.62, color=COLORS["scalar"],
               edgecolor="white", lw=LW_HAIR, zorder=2)
        ax.bar(index, spatial, bottom=common, width=0.62,
               color=COLORS[colour], edgecolor="white", lw=LW_HAIR, zorder=2)
        ax.bar(index, 1.0 - total, bottom=total, width=0.62,
               color=COLORS["grid"], edgecolor="white", lw=LW_HAIR, zorder=2)
        ax.plot([index], [-0.055], marker=marker, ms=MARKER_MS,
                mfc=COLORS[colour] if filled else "white",
                mec=COLORS[colour], mew=LW_EDGE, ls="none", clip_on=False,
                zorder=3)
        rows.append(dict(panel="E", method=method, common=common,
                         spatial=spatial, uncaptured=1.0 - total))
    # three-swatch key in the band above the bars (no rotated type, no
    # second hue: the family colour is the middle segment of its own bar)
    keys = [(-0.34, "scalar", "common"), (1.62, "shunting", "spatial"),
            (3.50, "grid", "rest")]
    for x, key, name in keys:
        ax.add_patch(Rectangle((x, 1.19), 0.30, 0.085,
                               facecolor=COLORS[key], edgecolor="white",
                               lw=LW_HAIR, clip_on=False, zorder=3))
        ax.text(x + 0.40, 1.232, name, fontsize=PT_SMALL,
                color=COLORS["ink"], ha="left", va="center", zorder=3)
    ax.set_xlim(-0.62, 5.62)
    ax.set_ylim(0.0, 1.38)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticks([])
    ax.tick_params(axis="y", labelleft=False)
    ax.text(2.5, 1.055, "families as in D", fontsize=PT_SMALL,
            color=COLORS["mute"], ha="center", va="bottom", zorder=3)
    style_panel(ax, spines=("left", "bottom"))
    return pd.DataFrame(rows)


def panel_cells(ax, pairs):
    """F: per-cell ancestry capture against its own surrogate trees."""
    hard = pairs.fraction_ge >= 0.5
    ax.plot([0.25, 1.02], [0.25, 1.02], color=COLORS["mute"], lw=LW_REF,
            dashes=(2.2, 1.8), zorder=1, solid_capstyle="butt")
    ax.plot(pairs.surrogate_mean[~hard], pairs.tree[~hard], marker="o",
            ls="none", ms=MARKER_MS - 1.0, mfc=COLORS["shunting"],
            mec="white", mew=LW_HAIR, alpha=0.85, zorder=2)
    ax.plot(pairs.surrogate_mean[hard], pairs.tree[hard], marker="o",
            ls="none", ms=MARKER_MS, mfc="white", mec=COLORS["highlight"],
            mew=LW_ERR, zorder=3)
    mx, mlo, mhi = mean_ci(pairs.surrogate_mean.to_numpy(float), BOOT_SEED + 1)
    my, ylo, yhi = mean_ci(pairs.tree.to_numpy(float), BOOT_SEED + 2)
    ax.errorbar([mx], [my], xerr=[[mx - mlo], [mhi - mx]],
                yerr=[[my - ylo], [yhi - my]], fmt="D",
                color=COLORS["shunting"], mfc="white", mec=COLORS["shunting"],
                mew=LW_ERR, ms=MARKER_MS, elinewidth=LW_ERR, capsize=2,
                zorder=4)
    above = int((pairs.tree > pairs.surrogate_mean).sum())
    ax.text(0.985, 0.030, f"{above} of {len(pairs)} above equality",
            fontsize=PT_SMALL, color=COLORS["mute"], ha="right", va="bottom",
            transform=ax.transAxes)
    ax.text(0.985, 0.125,
            f"{int(hard.sum())} open: \u2265 half of surrogates tie",
            fontsize=PT_SMALL, color=COLORS["highlight"], ha="right",
            va="bottom", transform=ax.transAxes)
    ax.set_xlim(0.25, 1.02)
    ax.set_ylim(0.25, 1.02)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("surrogate-tree mean")
    ax.set_ylabel("ancestry capture", labelpad=0.5)
    style_panel(ax)
    return pd.DataFrame(dict(root_id=pairs.root_id, tree=pairs.tree,
                             surrogate_mean=pairs.surrogate_mean,
                             fraction_surrogates_ge=pairs.fraction_ge,
                             n_replicates=pairs.n_replicates))


# ── row 2: cohorts and the paired advantage ──────────────────────────────
def panel_cohorts(ax, tables):
    """G: residual capture at K = 8 in three cohorts, four families."""
    shown = [METHODS[0], METHODS[2], METHODS[3], METHODS[1]]
    offsets = [0.255, 0.085, -0.085, -0.255]
    rows = []
    for row, (cohort, _) in enumerate(COHORT_ROWS):
        y = len(COHORT_ROWS) - 1 - row
        table = tables[cohort]
        eight = table[table.channels.eq(8)]
        for index, method in enumerate(shown):
            colour, marker, filled, _ = FAMILY[method]
            values = eight.loc[eight.method.eq(method),
                               "residual_capture"].to_numpy(float)
            mean, lo, hi = mean_ci(values, BOOT_SEED + 100 * row + index)
            ax.errorbar([mean], [y + offsets[index]],
                        xerr=[[mean - lo], [hi - mean]], fmt=marker,
                        color=COLORS[colour],
                        mfc=COLORS[colour] if filled else "white",
                        mec=COLORS[colour], mew=LW_ERR, ms=MARKER_MS,
                        elinewidth=LW_ERR, capsize=2, zorder=3)
            rows.append(dict(panel="G", cohort=cohort, method=method,
                             residual_capture=mean, low=lo, high=hi,
                             n_cells=int(values.size)))
    ax.set_yticks(range(len(COHORT_ROWS)),
                  [label for _, label in reversed(COHORT_ROWS)])
    ax.set_ylim(-0.55, len(COHORT_ROWS) - 0.35)
    ax.set_xlim(0.0, 1.10)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("residual capture after the broadcast")
    style_panel(ax, grid="x")
    ax.tick_params(axis="y", length=0, labelsize=PT_SMALL)
    ax.spines["left"].set_visible(False)
    ax.text(0.700, 1.745, "oracle", fontsize=PT_SMALL,
            color=COLORS["oracle"], ha="left", va="center",
            bbox=dict(boxstyle="round,pad=0.28,rounding_size=0.294",
                      facecolor=mix("oracle", 8), edgecolor=mix("oracle", 45),
                      linewidth=LW_HAIR))
    ax.text(0.02, 0.045, "Pinky: 9–13 excitatory sites per cell",
            fontsize=PT_SMALL, color=COLORS["mute"], ha="left", va="bottom",
            transform=ax.transAxes)
    ax.text(0.0, -0.235, "families as in D; 20,000-draw cell bootstrap",
            fontsize=PT_SMALL, color=COLORS["mute"], ha="left", va="top",
            transform=ax.transAxes)
    return pd.DataFrame(rows)


def panel_advantage(ax, report, tables):
    """H: paired ancestry advantage, its wiring cost and the oracle gap."""
    controls = [("common + surrogate ancestry", "Surrogate"),
                ("common + depth bins", "Depth bins"),
                ("common + shuffled routes", "Shuffled"),
                ("common + random routes", "Random")]
    eight = tables["v661"][tables["v661"].channels.eq(8)]
    wiring = eight.groupby("method").wiring_density.mean()
    rows, labels = [], []
    for index, (method, label) in enumerate(controls):
        y = len(controls) - index
        item = next(v for v in report["comparisons"]
                    if v["metric"] == "residual_capture"
                    and v["control"] == method)
        mean = 100.0 * item["mean_difference"]
        lo, hi = 100.0 * np.asarray(item["ci95"], float)
        ax.errorbar([mean], [y], xerr=[[mean - lo], [hi - mean]], fmt="D",
                    color=COLORS["shunting"], mfc=COLORS["shunting"],
                    mec=COLORS["shunting"], mew=LW_EDGE, ms=MARKER_MS,
                    elinewidth=LW_ERR, capsize=2, zorder=3)
        labels.append((y, label, COLORS["ink"]))
        rows.append(dict(panel="H", control=method, mean_pp=mean, low_pp=lo,
                         high_pp=hi, n_cells=item["n_cells"],
                         positive_cells=item["cells_positive"],
                         wiring_pct=100.0 * float(wiring.loc[method]),
                         family="Holm"))
    # the oracle gap is descriptive: hairline interval, mute ink, its own tag
    gap = eight.pivot_table(index="root_id", columns="method",
                            values="residual_capture")
    delta = 100.0 * (gap[METHODS[0]] - gap[METHODS[1]]).to_numpy(float)
    mean, lo, hi = mean_ci(delta, BOOT_SEED + 7)
    ax.errorbar([mean], [0], xerr=[[mean - lo], [hi - mean]], fmt="D",
                color=COLORS["mute"], mfc="white", mec=COLORS["mute"],
                mew=LW_HAIR, ms=MARKER_MS, elinewidth=LW_HAIR, capsize=2,
                zorder=3)
    labels.append((0, "SVD gap", COLORS["mute"]))
    rows.append(dict(panel="H", control=METHODS[1], mean_pp=mean, low_pp=lo,
                     high_pp=hi, n_cells=int(delta.size),
                     positive_cells=int((delta > 0).sum()),
                     wiring_pct=100.0 * float(wiring.loc[METHODS[1]]),
                     family="descriptive"))
    ax.set_xlim(-21.0, 60.5)
    ax.set_ylim(-1.40, 5.05)
    ax.set_xticks([-20, -10, 0, 10, 20, 30])
    ax.set_yticks([])
    ax.set_xlabel("ancestry − control, residual capture (pp)")
    style_panel(ax, spines=("bottom",), grid="x")
    reference_line(ax, 0.0, axis="x", label=None, span=(-0.22, 4.35))
    ax.text(0.8, 4.78, "no advantage", fontsize=PT_SMALL,
            color=COLORS["mute"], ha="left", va="center")
    for y, label, colour in labels:
        ax.text(-20.2, y + 0.30, label, fontsize=PT_SMALL, color=colour,
                ha="left", va="bottom")
    ax.text(-20.2, -0.75, "descriptive, not in the Holm family",
            fontsize=PT_SMALL, color=COLORS["mute"], ha="left", va="bottom")
    # the two right-hand columns, inside the axes behind a mute rule
    ax.plot([36.0, 36.0], [-0.22, 4.95], color=COLORS["grid"], lw=LW_HAIR,
            zorder=0.4)
    ax.text(48.0, 4.62, "wiring", fontsize=PT_ANNOT, color=COLORS["ink"],
            ha="right", va="center")
    ax.text(59.5, 4.62, "cells", fontsize=PT_ANNOT, color=COLORS["ink"],
            ha="right", va="center")
    for row, (y, _, colour) in zip(rows, labels):
        ax.text(48.0, y, f"{row['wiring_pct']:.1f} %", fontsize=PT_ANNOT,
                color=colour, ha="right", va="center")
        cells = ("—" if row["family"] == "descriptive"
                 else f"{row['positive_cells']}/{row['n_cells']}")
        ax.text(59.5, y, cells, fontsize=PT_ANNOT, color=colour,
                ha="right", va="center")
    ancestry = eight[eight.method.eq(METHODS[0])]
    ax.text(-20.2, -1.36, f"Ancestry: {100 * ancestry.wiring_density.mean():.1f} "
            f"% of dense wiring, rank {ancestry.dictionary_rank.mean():.2f}",
            fontsize=PT_ANNOT, color=COLORS["mute"], ha="left", va="bottom")
    return pd.DataFrame(rows)


# ── panel C's illustrative field, derived from the real operator ─────────
def field_from_operator(routes):
    """The paper tree's eight-site field, A and c, from the real operator.

    ``t`` is the weighted response to the focal shunt at i-site
    ``ROUTE_SITE``, reduced to the eight tree-ordered site blocks of the
    depth-3 balanced tree by an E-area-weighted mean and normalised by
    max |t| (sign kept).  ``A`` is that tree's broadcast plus three ancestry
    indicators; ``c`` is the W-weighted least-squares coefficient vector, so
    ``A c`` is the projection of ``t`` onto span(A) and the capture is its
    weighted energy share.
    """
    npz = routes["npz"]
    column = routes["i_sites"].index(ROUTE_SITE)
    response = np.asarray(npz["weighted_response"], float)[:, column]
    weights = routes["weights"]
    order = np.argsort(routes["site_rank"])
    blocks = np.array_split(order, 8)
    t = np.array([float((response[b] * weights[b]).sum() / weights[b].sum())
                  for b in blocks])
    w = np.array([float(weights[b].sum()) for b in blocks])
    t = t / max(float(np.abs(t).max()), 1e-30)
    w = w / w.max()
    A = np.zeros((8, 4))
    A[:, 0] = 1.0                 # broadcast
    A[0:4, 1] = 1.0               # r1: the shunted subtree
    A[0:2, 2] = 1.0               # r2: its proximal half
    A[4:6, 3] = 1.0               # r3: a sister subtree
    root = np.sqrt(w)
    coefficients, *_ = np.linalg.lstsq(A * root[:, None], t * root, rcond=None)
    fit = A @ coefficients
    capture = float((w * fit ** 2).sum() / (w * t ** 2).sum())
    return dict(t=t, w=w, A=A, c=coefficients, capture=capture)


# ── the canvas ───────────────────────────────────────────────────────────
def build(out=OUT, *, png=True, dpi=180, quiet=False):
    RECORDS.mkdir(parents=True, exist_ok=True)
    tables = cohort_tables()
    summaries = cohort_summaries()
    report = reports()
    routes = arbor_routes()
    field = field_from_operator(routes)
    pairs = surrogate_pairs()
    assert int(pairs.n_replicates.min()) == 200
    assert len(pairs) == 47

    canvas = NativeCanvas(
        488.0 / 72.0, 3, row_weights=[124, 112, 112], hgutter_pt=38.0,
        vgutter_pt=44.0,
        margins=Margins(left=56.0, right=14.0, top=24.0, bottom=36.0))
    a = canvas.panel("A", 0, 0, 4, schematic=True, lock=False,
                     title="Mapped E/I on one arbor")
    b = canvas.panel("B", 0, 4, 4, schematic=True, lock=False,
                     title="Seven ancestry routes")
    c = canvas.panel("C", 0, 8, 4, schematic=True, lock=False,
                     title="A shunt makes a field")
    d = canvas.panel("D", 1, 0, 4, title="Ancestry tracks the oracle")
    e = canvas.panel("E", 1, 4, 4, title="K = 8: broadcast is shared")
    g = canvas.panel("F", 1, 8, 4, title="Ancestry beats surrogates")
    h = canvas.panel("G", 2, 0, 6,
                     title="Residual capture in three cohorts")
    i = canvas.panel("H", 2, 6, 6,
                     title="Paired advantage and wiring cost")

    for ax_ in (h, i):
        ax_.set_title(ax_.get_title(), fontsize=PT_TITLE_, pad=1.0,
                      color=COLORS["ink"], fontweight="normal")
    panel_arbor(a, routes["cell"])
    panel_routes(b, routes)
    panel_field(c, field)
    budget = panel_budget(d, summaries, tables)
    partition = panel_partition(e, summaries)
    cells = panel_cells(g, pairs)
    cohorts = panel_cohorts(h, tables)
    advantage = panel_advantage(i, report["v661"], tables)
    for ax_ in (d, e, g):
        ax_.xaxis.labelpad = 0.8
        ax_.tick_params(axis="x", pad=1.0)

    style_direct_color_labels(canvas.fig)
    # lock first: ``lock_reserves`` re-syncs every letter, which would undo
    # the canvas-level column alignment if it ran afterwards.
    canvas.lock_reserves()
    findings = canvas.align_letters()
    problems = canvas.save(Path(out), name="credit_first_figure_06", png=png,
                           dpi=dpi, quiet=quiet, lock=False)
    layout = list(findings) + list(problems)

    # ── render-time source tables and the provenance record ─────────────
    budget.to_csv(RECORDS / "figure_06_primary_means.csv", index=False)
    partition.to_csv(RECORDS / "figure_06_energy_partition.csv", index=False)
    cells.to_csv(RECORDS / "figure_06_cell_surrogates.csv", index=False)
    cohorts.to_csv(RECORDS / "figure_06_cohort_points.csv", index=False)
    advantage.to_csv(RECORDS / "figure_06_residual_contrasts.csv", index=False)
    pd.DataFrame(dict(site_block=np.arange(1, 9), field_t=field["t"],
                      weight_W=field["w"],
                      projection=field["A"] @ field["c"])).to_csv(
        RECORDS / "figure_06_illustrative_field.csv", index=False)

    files = [Path(__file__), Path(anatomy.__file__),
             JOURNAL / "scripts/figure_canvas.py",
             JOURNAL / "scripts/journal_style.py",
             JOURNAL / "scripts/native_schematics.py",
             JOURNAL / "scripts/anatomy_commonmode/run.py",
             SOURCE / "figure3/segment_metrics.csv",
             COMMON / "protocol_freeze.json",
             COMMON / "original8/cells" / f"operator_{ROOT_B}.npz"]
    files += [COMMON / cohort / name for cohort in COHORTS
              for name in ("cell_method_summary.csv",
                           "cohort_method_summary.csv", "summary.json")]
    mapping = {
        "A": (f"Reconstructed arbor of root {ROOT_B} (78 segments), the "
              "median-sized cell of the original eight, from "
              "source_data/figure3/segment_metrics.csv; hue = mapped "
              "(E − I)/(E + I) contact area, stroke width = log mapped area "
              "in the five journal weights; isotropic principal-plane "
              "projection, 50 µm bar"),
        "B": ("Same arbor in ghost with the seven K = 8 ancestry routes "
              "reproduced from scripts/anatomy_commonmode/run.py:111-112 on "
              f"source_data/anatomy_commonmode/original8/cells/operator_{ROOT_B}.npz "
              "(origins 4784, 4621, 4396, 4458, 4975, 4209, 4516; supports "
              "1, 1, 5, 1, 1, 29, 1); the matrix collapses the 70 sites to "
              "eight tree-ordered blocks"),
        "C": ("Paper tree; the illustrative field is the weighted_response "
              f"column of i-site {ROUTE_SITE} in the same operator npz, "
              "reduced to eight tree-ordered site blocks by an E-area "
              "weighted mean and normalised by max |t|; A, c and the "
              "capture value are computed in the builder"),
        "D": ("v661 cohort_method_summary.csv total_capture_mean at K = "
              "1, 2, 4, 8, 16 for the six families; ancestry and surrogate "
              "bands are 20,000-draw cell bootstraps of "
              "cell_method_summary.csv (seed 202609061 + 10i + K); 47 cells, "
              "46 at K = 16"),
        "E": ("v661 cohort_method_summary.csv at K = 8: common = "
              "total_capture_mean − incremental_total_capture_mean, spatial "
              "= incremental_total_capture_mean, rest = 1 − "
              "total_capture_mean"),
        "F": ("Per-cell K = 8 total capture of common + ancestry against "
              "the mean of the 200 degree-depth surrogate trees, from "
              "source_data/anatomy_commonmode/v661/cells/rows_<root>.csv.gz; "
              "open points are the cells where at least half the surrogates "
              "reach the real tree; cohort mean with 20,000-draw cell "
              "bootstrap intervals"),
        "G": ("residual_capture at K = 8 per cohort from each cohort's "
              "cell_method_summary.csv (original8 8 cells, v661 47, Pinky 8) "
              "for ancestry, surrogate tree, depth bins and the "
              "common-constrained SVD; 20,000-draw cell bootstrap, seed "
              "202609061 + 100r + i"),
        "H": ("v661 summary.json residual_capture comparisons (paired means, "
              "retained 95 % cell-bootstrap intervals, Holm-adjusted "
              "Wilcoxon cells-positive counts); the SVD row is the "
              "descriptive per-cell ancestry − oracle difference from "
              "cell_method_summary.csv; wiring column = wiring_density mean "
              "of each control at K = 8; sub-line = ancestry wiring_density "
              "and dictionary_rank means"),
    }
    payload = dict(
        panel_sources=mapping,
        source_sha256={str(p.relative_to(JOURNAL)):
                       hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in files},
        layout_findings=layout,
        scope=("Modeled response capacity on measured anatomy; no evidence of "
               "endogenous biological route usage. Original8/v661 are the "
               "same animal; Pinky is one second animal."))
    (RECORDS / "figure_06_sources.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    # the caption stays in the record set beside the numbers it quotes; the
    # authoritative copy for main.tex lives in
    # analysis/figure_overhaul_20260908/fig7/TEXT.md
    (RECORDS / "figure_06_caption.md").write_text(CAPTION + "\n")
    MAIN.parent.mkdir(parents=True, exist_ok=True)
    MAIN.write_bytes(Path(out).read_bytes())
    if not quiet:
        print(json.dumps({"capture_C": field["capture"],
                          "layout": layout}, indent=2))
    return layout


if __name__ == "__main__":
    build()

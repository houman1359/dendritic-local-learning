#!/usr/bin/env python3
"""Main Figure 7 as ONE native full-width canvas.

MICrONS route capacity and wiring economy used to arrive as five pre-rendered
sub-blocks (``fig_main_mapped_reconstruction``, ``fig_main_ancestry_addresses``,
two panels lifted out of ``fig3_microns_topology``, ``fig_main_wire_efficiency``,
one panel of ``fig_capture_per_wire`` and ``fig_main_cross_animal``) that the
compositor scaled into grid slots.  Each block took its own scale factor, so
the compiled figure carried eight different type sizes and dozens of stroke
weights.  This module rebuilds the figure natively at scale 1.0 on a single
12-column module grid: a 7.6 pt tick label is 7.6 pt and an ``LW_EDGE`` spine
is 0.7 pt everywhere on the page.

Nothing here recomputes an estimate.  Every mean, every 95 % cell-bootstrap
interval and every n is taken from the same frozen source tables, through the
same ``mean_ci`` resampler at the same seeds, as the blocks it replaces
(``source_data/figure3``, ``source_data/reciprocal_routing``,
``source_data/capture_per_wire`` and
``source_data/pinky_v185_replication/routing``).  This file only decides where
the numbers sit on the page and how they are inked.

Layout (12 modules, three rows, four column starts)::

    A reconstructed arbor  B ancestry addresses  C cable field  D model field
    E capture and wiring at eight channels    F wiring-normalized capture
    G independent-animal direction              (F spans rows 1-2)

Row 0 is four equal three-module panels at columns 0, 3, 6 and 9; rows 1 and
2 are six-module panels at columns 0 and 6, so the whole figure uses those
same four column starts and every panel of a column shares one x0 and one
axes width.  C and D are one shared-axis small-multiple pair: identical
``field capture`` range, ticks and units, drawn once at the left of the pair.
F, the wiring-economy headline, is bigger only because it spans two rows --
its module-normalised area is within 1.3x of the smallest panel, not 9x as in
the composed version.  Colour
carries meaning only where it has to: the ancestry route keeps the shunting
green and the dense oracle the oracle violet, while every structural control
dictionary is rendered in the neutral control gray at two lightnesses and is
separated by marker and dash pattern instead of hue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from journal_style import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    PT_TICK,
    SEED_ALPHA,
    SEED_MS,
)
from figure_canvas import Margins, NativeCanvas, token_subscript  # noqa: E402
from credit_tree_schematics import MS_JUNCTION, mix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
OUT = ROOT / "figures" / "components" / "main_figure_07_native.pdf"

# ── canvas geometry, in points ───────────────────────────────────────────
CANVAS_H_PT = 482.0                      # 518.4 / 440 = 1.18 aspect
# Three equal rows and one wide horizontal gutter.  The gutter is the
# figure's single shared left reserve: it is set wider than the widest y
# label and tick column on the page, and the left margin wider than the
# widest category rail, so no panel carves that space out of its own module
# slot.  Every panel of a grid column therefore keeps one x0 and one axes
# width, and every row-mate one height.
HGUTTER = 34.0
VGUTTER = 46.0
MARGINS = Margins(left=61.0, right=13.0, top=22.0, bottom=28.0)

INK = COLORS["ink"]
MUTE = COLORS["mute"]
GRID = COLORS["grid"]

# ── colour semantics (S5: colour only where it carries meaning) ──────────
C_ORACLE = COLORS["oracle"]              # dense oracle keeps the violet slot
C_ROUTE = COLORS["shunting"]             # ancestry routes keep the green slot
C_CTRL = COLORS["point_mlp"]             # control gray, first lightness
C_CTRL_L = mix("point_mlp", 62)          # control gray, second lightness
C_MINNIE = COLORS["additive"]            # volume, not condition: blue
C_PINKY = COLORS["local"]                # volume, not condition: amber

MINUS = "−"

# One encoding for a routing dictionary, shared by C, D, E and F.
# (colour, marker, filled, dashed)
ROLE = {
    "dense": (C_ORACLE, "o", True, False),
    "ancestry": (C_ROUTE, "s", True, False),
    "random": (C_CTRL, "^", True, False),
    "depth": (C_CTRL_L, "D", True, False),
    "shuffled": (C_CTRL, "v", False, True),
    "surrogate": (C_CTRL_L, "P", False, True),
}
ROLE_NAME = {
    "dense": "dense oracle",
    "ancestry": "nested-subtree routes",
    "random": "random routes",
    "depth": "depth bins",
    "shuffled": "shuffled subtree",
    "surrogate": "matched tree",
}
# Short forms for the category rail of E, where the name is a tick label and
# not a key entry: F's key carries the full name, the caption defines both.
ROLE_TICK = {
    "dense": "dense",
    "ancestry": "subtree",
    "random": "random",
    "depth": "depth",
    "shuffled": "shuffled",
    "surrogate": "surrogate",
}

# Frozen table method names -> role, per generator.
MODEL_METHODS = [
    ("dense PCA oracle", "dense"),
    ("morphology-aware paths", "ancestry"),
    ("random paths", "random"),
    ("depth-only bins", "depth"),
    ("shuffled ancestry", "shuffled"),
]
CABLE_METHODS = [
    ("dense SVD oracle", "dense"),
    ("morphology paths", "ancestry"),
    ("random real paths", "random"),
    ("depth bins", "depth"),
    ("row-shuffled paths", "shuffled"),
    ("degree-depth surrogate tree", "surrogate"),
]
WIRE_METHODS = [
    ("dense PCA oracle", "dense"),
    ("morphology-aware paths", "ancestry"),
    ("random paths", "random"),
    ("depth-only bins", "depth"),
    ("shuffled ancestry", "shuffled"),
]

CAPTURE_YLIM = (0.0, 0.95)
CAPTURE_YTICKS = (0.0, 0.2, 0.4, 0.6, 0.8)
CHANNEL_LABEL = "feedback channels"
CAPTURE_LABEL = "field capture"


# ── data access (frozen tables, original seeds, nothing recomputed) ──────
def mean_ci(values, seed: int, n_boot: int = 20_000):
    """Descriptive cell-bootstrap mean and 95 % interval.

    Byte-for-byte the resampler used by the blocks this figure replaces
    (``build_journal_figures.mean_ci`` and
    ``build_main_panel_redesigns.mean_ci``); the seeds below are the ones
    those builders used, so every interval on the page is unchanged.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size),
                       replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def morphology_geometry():
    """The illustrative reconstruction: the median-sized cell of the eight."""
    segments = pd.read_csv(DATA / "figure3" / "segment_metrics.csv")
    sizes = segments.groupby("root_id").size()
    root = int((sizes - sizes.median()).abs().sort_values(kind="stable")
               .index[0])
    cell = segments[segments.root_id.eq(root)].copy()
    xyz = cell[["x_um", "y_um", "z_um"]].to_numpy(float)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ basis[:2].T
    span_um = max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
    projected /= span_um
    positions = {int(seg): point for seg, point
                 in zip(cell.segment_id.to_numpy(int), projected, strict=True)}
    rows = {int(row.segment_id): row for row in cell.itertuples(index=False)}
    parent = {int(row.segment_id): int(row.parent_segment_id)
              for row in cell.itertuples(index=False)}
    return cell, positions, rows, parent, span_um


def model_field_curves():
    """Per-cell model-field capture, then the original per-method seeds."""
    curves = pd.read_csv(DATA / "figure3" / "routing_capacity_curves.csv.gz")
    by_cell = curves.groupby(["root_id", "channels", "method"],
                             as_index=False).agg(
        credit_capture=("credit_capture", "mean"),
        wiring_density=("wiring_density", "mean"))
    out = {}
    for index, (method, role) in enumerate(MODEL_METHODS):
        part = by_cell[by_cell.method.eq(method)]
        x, mean, low, high = [], [], [], []
        for channels, group in part.groupby("channels"):
            m, lo, hi = mean_ci(group.credit_capture.to_numpy(float),
                                seed=1200 + 10 * index + int(channels))
            x.append(float(channels))
            mean.append(m)
            low.append(lo)
            high.append(hi)
        order = np.argsort(x)
        out[role] = tuple(np.asarray(v)[order]
                          for v in (x, mean, low, high))
    return out


def cable_field_curves():
    """Per-cell reciprocal-cable capture, at the original per-channel seeds."""
    table = pd.read_csv(DATA / "reciprocal_routing" / "cell_method_capture.csv")
    out = {}
    for method, role in CABLE_METHODS:
        part = table[table.method.eq(method)]
        grouped = part.groupby("channels").capture
        x = np.asarray(sorted(grouped.groups), dtype=float)
        mean, low, high = [], [], []
        for channels in x:
            m, lo, hi = mean_ci(grouped.get_group(channels).to_numpy(float),
                                seed=1400 + int(channels))
            mean.append(m)
            low.append(lo)
            high.append(hi)
        out[role] = (x, np.asarray(mean), np.asarray(low), np.asarray(high))
    return out


def wire_efficiency_table():
    """Eight-channel capture retained and wiring required, in % of oracle."""
    cell = pd.read_csv(DATA / "capture_per_wire" / "cell_method_channel.csv")
    eight = cell[cell.channels.eq(8)]
    out = {}
    for metric, seed in (("oracle_fraction", 20260851),
                         ("wiring_density", 20260871)):
        rows = []
        for index, (method, role) in enumerate(WIRE_METHODS):
            values = 100.0 * eight.loc[eight.method.eq(method),
                                       metric].to_numpy(float)
            rows.append((role,) + mean_ci(values, seed + index))
        out[metric] = rows
    return out


def per_wire_summary():
    return pd.read_csv(DATA / "capture_per_wire" / "summary.csv")


def cross_animal_rows():
    """Ancestry-route advantage per cell, per control, per volume."""
    contrasts = pd.read_csv(DATA / "pinky_v185_replication" / "routing"
                            / "k4_cross_animal_contrasts.csv")
    controls = [("random paths", "vs random"),
                ("depth-only bins", "vs depth"),
                ("shuffled ancestry", "vs shuffled")]
    animals = [("minnie65 v661", "MICrONS mouse 1 (n=47)",
                C_MINNIE, "s", -0.13, 0),
               ("Pinky v185", "MICrONS mouse 2 (n=10)",
                C_PINKY, "o", 0.13, 20)]
    out = []
    for animal, label, color, marker, offset, seed_shift in animals:
        subset = contrasts[contrasts.animal.eq(animal)]
        series = []
        for index, (control, _) in enumerate(controls):
            values = subset[subset.control.eq(control)] \
                .morphology_capture_advantage.to_numpy(float)
            stats = mean_ci(values, 20260830 + index + seed_shift)
            series.append((values, stats))
        out.append((label, color, marker, offset, series))
    return [name for _, name in controls], out


# ── drawing helpers ──────────────────────────────────────────────────────
def box_pt(ax):
    fig = ax.get_figure()
    fw, fh = fig.get_size_inches()
    box = ax.get_position()
    return box.width * fw * 72.0, box.height * fh * 72.0


def fit_isotropic(xy, rect, w_pt, h_pt, pad_pt=1.5):
    """Map data points into ``rect`` (axes fractions) without distortion."""
    xy = np.asarray(xy, dtype=float)
    x0, y0, w, h = rect
    avail_x = w * w_pt - 2 * pad_pt
    avail_y = h * h_pt - 2 * pad_pt
    span_x = max(np.ptp(xy[:, 0]), 1e-9)
    span_y = max(np.ptp(xy[:, 1]), 1e-9)
    scale = min(avail_x / span_x, avail_y / span_y)      # pt per data unit
    used_x, used_y = span_x * scale / w_pt, span_y * scale / h_pt
    ox = x0 + (w - used_x) / 2.0
    oy = y0 + (h - used_y) / 2.0
    xmin, ymin = xy[:, 0].min(), xy[:, 1].min()

    def to_axes(points):
        points = np.asarray(points, dtype=float).reshape(-1, 2)
        return np.column_stack([
            ox + (points[:, 0] - xmin) * scale / w_pt,
            oy + (points[:, 1] - ymin) * scale / h_pt,
        ])

    return to_axes


def blank_axes(ax):
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_axis_off()
    ax.set_facecolor("none")
    return ax


def role_line(ax, role, x, mean, low, high, *, zorder=2):
    color, marker, filled, dashed = ROLE[role]
    style = (0, (3.2, 2.0)) if dashed else "-"
    ax.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0,
                    zorder=zorder - 0.5)
    ax.plot(x, mean, color=color, lw=LW_DATA, ls=style, marker=marker,
            ms=MARKER_MS, mfc=color if filled else "white", mec=color,
            mew=LW_EDGE, solid_capstyle="round", zorder=zorder,
            clip_on=False)


def capture_axis(ax, *, left: bool):
    ax.set_xscale("log", base=2)
    ax.set_xlim(0.86, 18.6)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylim(*CAPTURE_YLIM)
    ax.set_yticks(list(CAPTURE_YTICKS))
    ax.set_xlabel(CHANNEL_LABEL)
    if left:
        ax.set_ylabel(CAPTURE_LABEL)
    else:
        ax.tick_params(axis="y", labelleft=False)


# ── A: the reconstruction, hue = E/I balance, width = mapped burden ──────
def panel_arbor(ax):
    cell, positions, rows, parent, span_um = morphology_geometry()
    blank_axes(ax)
    w_pt, h_pt = box_pt(ax)

    # Both visual channels encode mapped contact area.  Earlier drafts used
    # contact counts in the renderer while the panel and caption said area.
    # Using E_size/I_size here makes the graphic agree with the reported
    # anatomical quantity without changing any downstream numerical result.
    e_area = cell.E_size.to_numpy(float)
    i_area = cell.I_size.to_numpy(float)
    total = e_area + i_area
    balance = np.divide(e_area - i_area, total,
                        out=np.zeros_like(total), where=total > 0)
    burden = np.log1p(total)
    burden /= max(burden.max(), 1e-12)
    balance_by = dict(zip(cell.segment_id.to_numpy(int), balance, strict=True))
    burden_by = dict(zip(cell.segment_id.to_numpy(int), burden, strict=True))
    cmap = LinearSegmentedColormap.from_list(
        "contact_balance", [COLORS["inh"], "#DCE0E5", COLORS["exc"]])

    xy = np.asarray(list(positions.values()), dtype=float)
    to_axes = fit_isotropic(xy, (0.0, 0.330, 1.0, 0.670), w_pt, h_pt)
    place = {key: to_axes(point)[0] for key, point in positions.items()}

    # The pale complete skeleton establishes the morphology; the bivariate
    # overlay then carries the mapped burden on top of it.  Stroke widths are
    # the five journal weights, so the burden encoding is an explicit
    # five-step ladder rather than a continuous width the audit would snap.
    ladder = (LW_HAIR, LW_EDGE, LW_REF, LW_ERR, LW_DATA)
    for segment in rows:
        p = parent[segment]
        if p not in rows:
            continue
        start, end = place[segment], place[p]
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=mix("mute", 24), lw=LW_HAIR, solid_capstyle="round",
                zorder=1)
    for segment in rows:
        p = parent[segment]
        if p not in rows:
            continue
        weight = burden_by[segment]
        if weight <= 0:
            continue
        start, end = place[segment], place[p]
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=cmap((balance_by[segment] + 1.0) / 2.0),
                lw=ladder[int(round(weight * (len(ladder) - 1)))],
                alpha=0.45 + 0.55 * weight, solid_capstyle="round", zorder=3)

    soma_id = min(rows, key=lambda key: rows[key].topological_depth)
    sx, sy = place[soma_id]
    # Axes fractions are not isotropic in a portrait cell, so a node drawn as
    # a Circle would print as an ellipse: give it the cell's own aspect.
    ax.add_patch(Ellipse((sx, sy), 2 * 0.030, 2 * 0.030 * w_pt / h_pt,
                         facecolor=COLORS["soma"], edgecolor="white",
                         lw=LW_EDGE, zorder=6))

    # The PCA projection is isotropically scaled, so a horizontal 50-µm bar
    # remains metric after fitting the reconstruction into the panel.
    xy_raw = np.asarray(list(positions.values()), dtype=float)
    anchor = np.asarray([xy_raw[:, 0].min(), xy_raw[:, 1].min()])
    scale_points = to_axes(np.vstack([anchor,
                                      anchor + [50.0 / span_um, 0.0]]))
    scale_width = float(scale_points[1, 0] - scale_points[0, 0])
    scale_x0, scale_y = 0.055, 0.405
    ax.plot([scale_x0, scale_x0 + scale_width], [scale_y, scale_y],
            color=INK, lw=LW_DATA, solid_capstyle="butt", zorder=7)
    ax.text(scale_x0 + scale_width / 2.0, scale_y - 0.032, "50 µm",
            ha="center", va="top", fontsize=PT_SMALL, color=INK, zorder=7)

    # Slim labelled key for the signed hue, in its own band under the arbor:
    # the ramp with its two poles named at the ends, then the width encoding.
    gradient = np.linspace(0.0, 1.0, 256)[None, :]
    bar = (0.245, 0.755, 0.186, 0.228)
    ax.imshow(gradient, cmap=cmap, aspect="auto", origin="lower",
              extent=bar, zorder=4)
    ax.add_patch(Rectangle((bar[0], bar[2]), bar[1] - bar[0], bar[3] - bar[2],
                           facecolor="none", edgecolor=COLORS["edge"],
                           lw=LW_EDGE, zorder=5))
    ax.text(bar[0] - 0.030, 0.5 * (bar[2] + bar[3]), "I-rich", ha="right",
            va="center", fontsize=PT_SMALL, color=INK)
    ax.text(bar[1] + 0.030, 0.5 * (bar[2] + bar[3]), "E-rich", ha="left",
            va="center", fontsize=PT_SMALL, color=INK)
    # The key carries a scale, not only two words: ticks at -1, 0 and +1 and
    # the quantity the ramp encodes.  The ramp itself borrows the anatomy
    # exc/inh hues rather than DIV_CMAP, and says so.
    for frac, tick in ((0.0, MINUS + "1"), (0.5, "0"), (1.0, "+1")):
        x = bar[0] + frac * (bar[1] - bar[0])
        ax.plot([x, x], [bar[2] - 0.014, bar[2]], color=COLORS["edge"],
                lw=LW_HAIR, zorder=5, clip_on=False)
        ax.text(x, bar[2] - 0.030, tick, ha="center", va="top",
                fontsize=PT_SMALL, color=INK)
    ax.text(0.5 * (bar[0] + bar[1]), 0.290, "(E " + MINUS + " I) / (E + I)",
            ha="center", va="center", fontsize=PT_SMALL, color=INK)
    ax.text(0.5, 0.038, "width = log mapped area (E + I), 5 levels",
            ha="center", va="center",
            fontsize=PT_SMALL, color=MUTE)


# ── B: subtree routes and the field-capture operation ────────────────────
def panel_addresses(ax):
    blank_axes(ax)
    w_pt, h_pt = box_pt(ax)

    soma = (0.50, 0.055)
    j1 = (0.50, 0.235)
    jl, jr = (0.255, 0.405), (0.755, 0.405)
    jll, jlr = (0.135, 0.560), (0.375, 0.560)
    jrl, jrr = (0.630, 0.560), (0.885, 0.560)
    tips = {
        jll: [(0.065, 0.700), (0.215, 0.700)],
        jlr: [(0.315, 0.700), (0.455, 0.700)],
        jrl: [(0.565, 0.700), (0.700, 0.700)],
        jrr: [(0.820, 0.700), (0.955, 0.700)],
    }
    edges = [(soma, j1, LW_DATA), (j1, jl, LW_ERR), (j1, jr, LW_ERR),
             (jl, jll, LW_REF), (jl, jlr, LW_REF),
             (jr, jrl, LW_REF), (jr, jrr, LW_REF)]
    for node, ends in tips.items():
        for end in ends:
            edges.append((node, end, LW_EDGE))

    # Route capsules first: pale fat round strokes, exactly the vocabulary the
    # credit-tree library uses for an address.  B's capsule lies inside A's,
    # which is the whole point of the panel.
    chain_a = [[jr, jrl, tips[jrl][0]], [jrl, tips[jrl][1]],
               [jr, jrr, tips[jrr][0]], [jrr, tips[jrr][1]]]
    chain_b = [[jrr, tips[jrr][0]], [jrr, tips[jrr][1]]]
    for chain in chain_a:
        pts = np.asarray(chain, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color=mix("shunting", 24), lw=7.6,
                solid_capstyle="round", solid_joinstyle="round", zorder=1)
    for chain in chain_b:
        pts = np.asarray(chain, dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color=mix("shunting", 52), lw=4.4,
                solid_capstyle="round", solid_joinstyle="round", zorder=2)

    for start, end, width in edges:
        ax.plot([start[0], end[0]], [start[1], end[1]], color=COLORS["dend"],
                lw=width, solid_capstyle="round", zorder=3)
    for node in (j1, jl, jr, jll, jlr, jrl, jrr):
        if node in (jr, jrr):
            continue
        ax.plot([node[0]], [node[1]], marker="o", ms=MS_JUNCTION, mfc="white",
                mec=COLORS["dend"], mew=LW_EDGE, ls="none", zorder=4)
    ax.add_patch(Ellipse((soma[0], soma[1]), 2 * 0.034,
                         2 * 0.034 * w_pt / h_pt, facecolor=COLORS["soma"],
                         edgecolor=mix("ink", 30), lw=LW_EDGE, zorder=5))

    # The two addressed junctions are named in place: a white node carrying
    # its address letter, so no label has to cross a stroke.
    for node, tag in ((jr, "A"), (jrr, "B")):
        ax.add_patch(Ellipse(node, 2 * 5.2 / w_pt, 2 * 5.2 / h_pt,
                             facecolor="white", edgecolor=C_ROUTE,
                             lw=LW_EDGE, zorder=5, transform=ax.transData))
        ax.text(node[0], node[1], tag, ha="center", va="center",
                fontsize=PT_SMALL, color=C_ROUTE, zorder=6)

    # Define the quantity used by every data panel before showing its values.
    # P_A is the orthogonal projection onto the span of the K route columns;
    # the caption expands the notation, while the panel makes the operation
    # and its normalized energy measure visible at first encounter.
    token_subscript(ax, 0.395, 0.935, "q → P", "A", " q",
                    size=PT_ANNOT, sub_size=PT_SMALL, color=INK,
                    ha="left", va="center")
    token_subscript(ax, 0.285, 0.842, "C", "A", " = ‖Pq‖² ⁄ ‖q‖²",
                    size=PT_SMALL, sub_size=PT_SMALL, color=MUTE,
                    ha="left", va="center")
    ax.text(0.50, 0.755, "B ⊂ A", ha="center", va="center",
            fontsize=PT_SMALL, color=INK)
    # "one channel per address" is the panel's definition, not its geometry;
    # the caption carries it.
    ax.text(0.585, 0.055, "soma", ha="left", va="center", fontsize=PT_SMALL,
            color=MUTE)


# ── C, D: the two capture generators on one shared axis ──────────────────
def panel_cable(ax, curves):
    for role in ("dense", "ancestry", "random", "depth", "shuffled",
                 "surrogate"):
        role_line(ax, role, *curves[role])
    capture_axis(ax, left=True)


def panel_model(ax, curves):
    for role in ("dense", "ancestry", "random", "depth", "shuffled"):
        role_line(ax, role, *curves[role])
    capture_axis(ax, left=False)


# ── E: eight-channel capture retained against wiring required ────────────
def panel_wire_efficiency(ax, table):
    order = [role for _, role in WIRE_METHODS]
    y = {role: len(order) - 1 - index for index, role in enumerate(order)}
    dy, height = 0.21, 0.34
    for metric, offset, filled in (("oracle_fraction", +dy, True),
                                   ("wiring_density", -dy, False)):
        for role, mean, low, high in table[metric]:
            color = ROLE[role][0]
            ypos = y[role] + offset
            ax.barh(ypos, mean, height=height,
                    facecolor=mix(color, 22) if filled else "white",
                    edgecolor=color, lw=LW_EDGE, zorder=2)
            ax.errorbar(mean, ypos, xerr=[[mean - low], [high - mean]],
                        fmt="none", ecolor=color, elinewidth=LW_ERR,
                        capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=4)
    # No value callouts beside the bars: the bar length and the % axis below
    # already are those numbers, and the exact values live in Source Data.
    ax.set_xlim(0.0, 112.0)
    ax.set_xticks([0, 50, 100])
    ax.set_xlabel("% of dense oracle")
    ax.set_yticks([y[role] for role in order])
    ax.set_yticklabels([ROLE_TICK[role] for role in order], fontsize=PT_TICK)
    ax.set_ylim(-0.62, len(order) - 0.38)
    ax.tick_params(axis="y", length=0.0, pad=2.5)
    ax.spines["left"].set_visible(False)

    # Two-entry key for the metric, in the whitespace the short wiring bars
    # leave under the dense and ancestry rows.
    for index, (label, filled) in enumerate((("capture retained", True),
                                             ("wiring required", False))):
        ypos = 1.72 - 0.52 * index
        ax.add_patch(Rectangle((52.0, ypos - 0.145), 7.5, 0.29,
                               facecolor=mix(MUTE, 22) if filled else "white",
                               edgecolor=MUTE, lw=LW_EDGE, zorder=5,
                               clip_on=False))
        ax.text(62.0, ypos, label, ha="left", va="center", fontsize=PT_SMALL,
                color=MUTE, zorder=5)


# ── F: the headline, capture per unit wiring across budgets ──────────────
def panel_per_wire(ax, summary):
    for method, role in WIRE_METHODS:
        part = summary[summary.method.eq(method)].sort_values("channels")
        x = part.channels.to_numpy(float)
        role_line(ax, role, x,
                  part.mean_capture_per_wire.to_numpy(float),
                  part.ci95_low_capture_per_wire.to_numpy(float),
                  part.ci95_high_capture_per_wire.to_numpy(float))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(0.90, 9.6)
    ax.set_ylim(0.032, 11.0)
    ax.set_xticks([1, 2, 4, 8])
    ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_yticks([0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(
        ["0.05", "0.1", "0.5", "1", "5", "10"]))
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlabel(CHANNEL_LABEL)
    ax.set_ylabel("capture per unit wiring")

    # The eight-channel ancestry advantage over the dense oracle, and the
    # part of it that survives at matched wiring density, are ratios of two
    # plotted means: the geometry is the report, so both numbers are stated
    # in the caption instead of being printed on the panel.

    handles = [
        Line2D([0], [0], color=ROLE[role][0], lw=LW_DATA,
               ls=(0, (3.2, 2.0)) if ROLE[role][3] else "-",
               marker=ROLE[role][1], ms=MARKER_MS,
               mfc=ROLE[role][0] if ROLE[role][2] else "white",
               mec=ROLE[role][0], mew=LW_EDGE,
               label=ROLE_NAME[role])
        for _, role in WIRE_METHODS
    ]
    legend = ax.legend(handles=handles, loc="lower right", fontsize=PT_LEGEND,
                       frameon=True, facecolor="white", edgecolor="none",
                       framealpha=0.90, handlelength=1.6, handletextpad=0.4,
                       borderpad=0.35, labelspacing=0.30,
                       borderaxespad=0.45,
                       title="route dictionary (C–F)",
                       title_fontsize=PT_SMALL)
    legend.get_title().set_color(MUTE)
    for text in legend.get_texts():          # never pure #000000
        text.set_color(INK)
    legend.set_zorder(6)


# ── G: the same direction in an independent volume ───────────────────────
def panel_cross_animal(ax, labels, animals):
    n = len(labels)
    y = {index: n - 1 - index for index in range(n)}
    for label, color, marker, offset, series in animals:
        for index, (values, (mean, low, high)) in enumerate(series):
            ypos = y[index] + offset
            ax.scatter(values, np.full(values.size, ypos), s=SEED_MS ** 2,
                       color=color, alpha=SEED_ALPHA,
                       edgecolors="none", zorder=1)
            ax.errorbar(mean, ypos, xerr=[[mean - low], [high - mean]],
                        fmt=marker, ms=MARKER_MS + 1.2, color=color,
                        mfc=color, mec="white", mew=LW_EDGE, ecolor=color,
                        elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                        capthick=LW_ERR, zorder=4)
    ax.axvline(0.0, color=MUTE, lw=LW_REF, ls=(0, (3.0, 2.2)), zorder=1)
    ax.set_yticks([y[index] for index in range(n)])
    ax.set_yticklabels(labels, fontsize=PT_TICK)
    ax.set_ylim(-0.60, n - 1 + 0.92)
    ax.set_xlim(-0.06, 0.99)
    ax.set_xticks([0.0, 0.4, 0.8])
    ax.set_xlabel("subtree-route capture advantage")
    ax.tick_params(axis="y", length=0.0, pad=2.5)
    ax.spines["left"].set_visible(False)

    # Direct labels replace the key: two series, named over the marker they
    # belong to, staggered so the two names never share a line.
    top = y[0]
    for label, color, offset, x_at, y_at in (
            ("MICrONS mouse 1 (n=47)", C_MINNIE, -0.13, 0.268, 0.30),
            ("MICrONS mouse 2 (n=10)", C_PINKY, 0.13, 0.527, 0.70)):
        ax.plot([x_at, x_at], [top + offset + 0.10, top + y_at - 0.13],
                color=MUTE, lw=LW_HAIR, solid_capstyle="round", zorder=2)
        ax.text(x_at, top + y_at, label, ha="center", va="center",
                fontsize=PT_ANNOT, color=color)


# ── the canvas ───────────────────────────────────────────────────────────
def build():
    cable = cable_field_curves()
    model = model_field_curves()
    wire = wire_efficiency_table()
    summary = per_wire_summary()
    labels, animals = cross_animal_rows()

    canvas = NativeCanvas(
        CANVAS_H_PT / 72.0, 3,
        hgutter_pt=HGUTTER, vgutter_pt=VGUTTER, margins=MARGINS,
        letters=False,
    )

    # Row 0: the anatomy, then the schematic that defines a route, then the
    # two capture generators as one shared-axis pair.  No panel carries a
    # hand inset: the category rails of E and G live in the left margin and
    # the y labels of C and F in the gutter, so the column lock finds nothing
    # left to carve and every panel keeps its whole module slot.
    ax_a = canvas.panel("A", 0, 0, 3, schematic=True,
                        title="Reconstructed arbor")
    ax_b = canvas.panel("B", 0, 3, 3, schematic=True,
                        title="Subtree projection")
    ax_c = canvas.panel("C", 0, 6, 3, grid="y", title="Reciprocal cable field")
    ax_d = canvas.panel("D", 0, 9, 3, grid="y", sharey=ax_c,
                        title="Model-derived field")

    # Rows 1-2: the eight-channel economy and the independent volume on the
    # left, the wiring-normalized headline on the right across both rows.
    ax_e = canvas.panel("E", 1, 0, 6, grid="x",
                        title="Eight-channel capture and wiring")
    ax_f = canvas.panel("F", 1, 6, 6, rowspan=2, grid="y",
                        title="Wiring-normalized capture")
    ax_g = canvas.panel("G", 2, 0, 6, grid="x",
                        title="Independent-animal direction")

    panel_arbor(ax_a)
    panel_addresses(ax_b)
    panel_cable(ax_c, cable)
    panel_model(ax_d, model)
    panel_wire_efficiency(ax_e, wire)
    panel_per_wire(ax_f, summary)
    panel_cross_animal(ax_g, labels, animals)

    # One letter offset for the whole figure, so every letter sits the same
    # distance left of the column its panel starts in.
    for name in ("A", "B", "C", "D", "E", "F", "G"):
        canvas.add_letter(name, canvas.axes[name], dx_pt=24.0)

    problems = canvas.save(OUT, name="main_figure_07_native")
    return problems


if __name__ == "__main__":
    issues = build()
    for issue in issues:
        print(f"  {issue}")
    print(f"  {len(issues)} layout/overlap warnings")

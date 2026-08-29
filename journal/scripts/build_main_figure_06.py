#!/usr/bin/env python3
"""Build the unified physical-depth boundary figure.

This figure combines the clearest design material from the former physical-
depth figure with the hierarchy-depth and task-family boundary experiments.
It is deliberately a *single* native 7.2-inch canvas: all panel titles,
labels, line weights, gutters and colours therefore use the same journal
tokens as the other main figures and are never rescaled by a compositor.

No result is recomputed. Panels C--G read the frozen seed summaries produced
by the confirmatory analyzers. Panels A and B only explain the manipulations:

* H is the number of latent gain levels in the task;
* D_p is the number of ordered nonlinear stages in the model;
* nested factors require ordered cancellation, flat factors preserve the
  product without a nested grouping, and local ratios expose the divisive
  information within a stage;
* the serial dendritic model, resource-identical grouped-point emulation and
  flexible point-network ceiling are visually and verbally distinct.

Layout (12-column module grid)::

    A  three task families (7)        B  architecture controls (5)
    C  H=3 quantitative boundary (6) D  H=4 aligned/reversed boundary (6)
    E  depth x hierarchy (4)         F  alignment dose, BP (4)
                                      G  alignment dose, local rule (4)

Output: ``figures/components/main_figure_06_native.pdf`` and its audit PNG.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from credit_tree_schematics import AMBER_TEXT, mix  # noqa: E402
from figure_canvas import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    Margins,
    NativeCanvas,
    PT_ANNOT,
    PT_SMALL,
    SEQ_CMAP,
    style_panel,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
OUTPUT = ROOT / "figures" / "components" / "main_figure_06_native.pdf"

# ── canonical geometry and palette ──────────────────────────────────────
CANVAS_H_PT = 446.0
ROW_H_PT = (105.0, 92.0, 92.0)
HGUTTER = 38.0
VGUTTER = 48.0
MARGINS = Margins(left=48.0, right=35.0, top=24.0, bottom=28.0)

INK = COLORS["ink"]
MUTE = COLORS["mute"]
C_SERIAL = COLORS["shunting"]
# "serial BP" is exact backpropagation, so it takes the reserved
# backprop hue.  C_SERIAL stays green where it marks the serial
# ARCHITECTURE (the panel-B schematic and the aligned-sensor tag).
C_SERIAL_BP = COLORS["bp"]
C_GROUPED = COLORS["point_mlp"]
C_POINT = COLORS["ink"]   # neutral ceiling reference, not an architecture
C_PATH = COLORS["oracle"]
C_SHARED = COLORS["local"]
C_ADDITIVE = COLORS["additive"]

C_NESTED = COLORS["shunting"]
C_FLAT = COLORS["additive"]
C_LOCAL_RATIO = COLORS["highlight"]
FAMILY_SPECS = (
    ("nested_factor", "nested factors", C_NESTED, "o"),
    ("flat_factor", "flat factors", C_FLAT, "s"),
    ("local_ratio", "local ratios", C_LOCAL_RATIO, "^"),
)

ACC_NORM = Normalize(vmin=0.52, vmax=0.99)
H4_ROWS = (
    ("BP", "serial_tree", "shunting", "full_bp"),
    ("LocalCA", "serial_tree", "shunting", "local_path"),
    ("shared", "serial_tree", "shunting", "local_shared"),
    ("grouped", "grouped_point", "shunting", "full_bp"),
    ("additive", "serial_tree", "raw_additive", "full_bp"),
)


# ── general drawing helpers ─────────────────────────────────────────────
def _advance_pt(text: str, size: float) -> float:
    """Advance width of ``text`` at the figure's own sans face, in points."""
    if not text:
        return 0.0
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath
    path = TextPath((0.0, 0.0), text, size=size, prop=FontProperties(size=size))
    return float(path.get_extents().x1)


def formula(ax, x, y, parts, *, size=PT_SMALL, color=None, ha="center"):
    """Draw an expression as baseline-shifted runs, all at one token size.

    ``parts`` is a sequence of ``(text, dy_pt)``; a positive ``dy`` raises a
    superscript and a negative one drops a subscript.  Mathtext is banned by
    the figure contract because its 0.7x sub-glyphs fall below the 6.8 pt
    floor, and precomposed Unicode sub/superscripts render at whatever size
    and weight the face happens to carry for them -- the Latin capital
    superscripts especially.  Chaining real spans keeps every glyph on the
    token scale and in the same face.
    """
    color = INK if color is None else color
    widths = [_advance_pt(text, size) for text, _ in parts]
    total = sum(widths)
    if ha == "center":
        cursor = x - total / 2.0
    elif ha == "right":
        cursor = x - total
    else:
        cursor = x
    for (text, dy), width in zip(parts, widths):
        ax.text(cursor, y + dy, text, ha="left", va="center",
                fontsize=size, color=color)
        cursor += width
    return total



def point_frame(ax):
    """Put a schematic axes into its own physical point coordinate frame."""
    fig = ax.get_figure()
    fw, fh = fig.get_size_inches()
    box = ax.get_position()
    width = box.width * fw * 72.0
    height = box.height * fh * 72.0
    ax.set_xlim(0.0, width)
    ax.set_ylim(0.0, height)
    ax.set_axis_off()
    ax.set_facecolor("none")
    return width, height


def round_box(ax, xy, width, height, *, edge, fill=None, radius=2.4,
              lw=LW_EDGE, zorder=2):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fill if fill is not None else mix(edge, 10),
        edgecolor=edge, lw=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(ax, start, end, *, color=MUTE, lw=LW_EDGE, zorder=3):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(
            arrowstyle="-|>,head_length=3.0,head_width=1.9",
            mutation_scale=1.0, color=color, lw=lw,
            shrinkA=0, shrinkB=0,
        ),
        zorder=zorder,
    )


def _series(frame: pd.DataFrame, **filters) -> pd.DataFrame:
    out = frame
    for key, value in filters.items():
        out = out[out[key].eq(value)]
    return out.sort_values("depth")


def accuracy_curve(ax, frame, *, color, marker, label, y_columns,
                   filled=True, dashes="-", dx=0.0, **filters):
    rows = _series(frame, **filters)
    x = rows.depth.to_numpy(float) + dx
    mean = rows[y_columns[0]].to_numpy(float)
    low = rows[y_columns[1]].to_numpy(float)
    high = rows[y_columns[2]].to_numpy(float)
    ax.plot(x, mean, color=color, lw=LW_DATA, ls=dashes,
            solid_capstyle="round", zorder=2)
    ax.errorbar(
        x, mean, yerr=np.vstack([mean - low, high - mean]),
        fmt=marker, ms=MARKER_MS, mfc=color if filled else "white",
        mec="white" if filled else color,
        mew=LW_EDGE, ecolor=color, elinewidth=LW_ERR,
        capsize=ERR_CAPSIZE, zorder=3,
    )
    return float(x[-1]), float(mean[-1]), label


def _luminance(rgba) -> float:
    return 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]


def heatmap(ax, matrix, row_labels, col_labels, *, best_by_row=True,
            fmt="{:.2f}"):
    """Sequential heatmap with values as the primary quantitative channel."""
    cmap = SEQ_CMAP.copy()
    cmap.set_bad("#F2F3F5")
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=cmap, norm=ACC_NORM, aspect="auto",
                      interpolation="nearest")
    for row in range(matrix.shape[0]):
        finite = np.flatnonzero(np.isfinite(matrix[row]))
        best = finite[np.argmax(matrix[row, finite])] if len(finite) else None
        if best is not None and len(finite) > 1:
            spread = matrix[row, finite].max() - matrix[row, finite].min()
            if spread < 0.01:
                best = None
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                continue
            rgba = cmap(ACC_NORM(value))
            ax.text(col, row, fmt.format(value), ha="center", va="center",
                    fontsize=PT_SMALL,
                    color="white" if _luminance(rgba) < 0.48 else INK)
            if best_by_row and col == best:
                x0, x1 = col - 0.46, col + 0.46
                y0, y1 = row - 0.43, row + 0.43
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                        color=INK, lw=LW_ERR, solid_joinstyle="miter",
                        solid_capstyle="butt", zorder=5)
    ax.set_xticks(range(len(col_labels)), list(col_labels))
    ax.set_yticks(range(len(row_labels)), list(row_labels))
    ax.tick_params(axis="both", length=0, pad=2.0, labelsize=PT_SMALL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def color_rail(canvas, image, ax, label="test accuracy"):
    """Put one shared accuracy key in the reserved right outer margin."""
    box = ax.get_position()
    x0 = box.x1 + 4.0 / canvas.width_pt
    cax = canvas.fig.add_axes([
        x0, box.y0, 6.0 / canvas.width_pt, box.height,
    ])
    cbar = canvas.fig.colorbar(image, cax=cax, ticks=[0.55, 0.70, 0.85, 0.99])
    cbar.outline.set_linewidth(LW_EDGE)
    cbar.outline.set_edgecolor(COLORS["edge"])
    cbar.ax.tick_params(labelsize=PT_SMALL, width=LW_EDGE, length=2.2,
                        pad=1.3, color=COLORS["edge"], labelcolor=INK)
    # At pad=2 the title sat on the 0.99 tick label (3.3 pt of overlap); the
    # rail is narrow, so the title needs the tick row's own height cleared.
    cbar.ax.set_title("accuracy", fontsize=PT_SMALL, pad=7.5, color=INK)
    canvas.bind_satellite(cax, ax)


# ── A: task-family schematics ───────────────────────────────────────────
def panel_task_families(ax):
    w, h = point_frame(ax)
    gap = 8.0
    card_w = (w - 2 * gap) / 3.0
    card_h = h - 17.0
    card_y = 10.0

    cards = (
        (0.0, "nested factors", C_NESTED),
        (card_w + gap, "flat factors", C_FLAT),
        (2 * (card_w + gap), "local ratios", C_LOCAL_RATIO),
    )
    for x0, label, color in cards:
        round_box(ax, (x0, card_y), card_w, card_h, edge=mix(color, 50),
                  fill=mix(color, 5), radius=3.0, lw=LW_HAIR, zorder=0)
        ax.text(x0 + card_w / 2, h - 5.0, label, ha="center", va="top",
                fontsize=7.2, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5))

    # Nested: one signal is successively modulated by three ordered levels.
    x0 = cards[0][0]
    cy = card_y + 0.57 * card_h
    # Lay the chain out from explicit widths.  The five linspace anchors gave
    # 12.65 pt of spacing for 16 pt boxes, so the boxes overlapped by 3.35 pt
    # and every arrow ran from +9 to +3.65 -- a negative length whose head
    # landed on the next box's border and on its label.
    # Caps differ because "s" is one glyph and the terminal is the two-run
    # x_E; a wider gap gives each arrow a visible shaft instead of a head
    # jammed against the next box.
    box_w, gap = 12.5, 5.8
    cap_left, cap_right = 4.5, 9.5
    chain_w = 3 * box_w + 4 * gap + cap_left + cap_right
    left = x0 + (card_w - chain_w) / 2.0
    ax.text(left + cap_left / 2.0, cy, "sᵧ", ha="center", va="center",
            fontsize=PT_SMALL, color=INK)
    edges = [left + cap_left + gap + index * (box_w + gap) for index in range(3)]
    for index, (bx, color_pct) in enumerate(zip(edges, (24, 40, 56)), start=1):
        arrow(ax, (bx - gap + 1.3, cy), (bx - 1.3, cy),
              color=mix(C_NESTED, 65))
        round_box(ax, (bx, cy - 7.0), box_w, 14.0,
                  edge=C_NESTED, fill=mix(C_NESTED, color_pct), radius=2.0)
        ax.text(bx + box_w / 2.0, cy, f"h{index}", ha="center", va="center",
                fontsize=PT_SMALL, color=INK)
    tail = edges[-1] + box_w
    arrow(ax, (tail + 1.3, cy), (tail + gap - 1.3, cy), color=C_NESTED)
    # The chain ends at the OBSERVED excitatory coordinate.  Labelling it "y"
    # collided with the Methods, where y is the class label that enters the
    # signal, not the quantity that leaves the gain chain.
    formula(ax, tail + gap + cap_right / 2.0, cy,
            (("x", 0.0), ("E", -1.6)))
    # The generative parameterisation (b_E, class label y, contrast delta) is
    # Methods material: none of those symbols is defined in this figure, and
    # its y is not the y that used to end the chain above.  The drawing already
    # states the structure -- one signal, three ordered gains.
    formula(ax, x0 + card_w / 2, card_y + 0.20 * card_h,
            (("s", 0.0), ("y", -1.6), (" \u00d7 h", 0.0), ("1", -1.6),
             ("h", 0.0), ("2", -1.6), ("h", 0.0), ("3", -1.6),
             (" in order", 0.0)))
    ax.text(x0 + card_w / 2, card_y + 0.06 * card_h,
            # 90.4 pt of text in a 74.6 pt card overflowed 8 pt each side;
            # the full phrasing is in the caption.
            "ordered cancellation", ha="center", va="bottom",
            fontsize=PT_SMALL, color=MUTE)

    # Flat: the same product is present, but factors belong to unrelated
    # feature groups rather than one nested hierarchy.
    x0 = cards[1][0]
    centres = np.linspace(x0 + 0.22 * card_w, x0 + 0.78 * card_w, 3)
    top_y = card_y + 0.64 * card_h
    comb = (x0 + card_w / 2, card_y + 0.39 * card_h)
    for index, cx in enumerate(centres, start=1):
        round_box(ax, (cx - 10.0, top_y - 7.0), 20.0, 14.0,
                  edge=C_FLAT, fill=mix(C_FLAT, 16 + 13 * index), radius=2.0)
        ax.text(cx, top_y, f"h{index}", ha="center", va="center",
                fontsize=PT_SMALL, color=INK)
        # Stop on the product node's top edge, not its centre: ending at
        # ``comb`` drove every connector through the box border and under
        # the product glyph.
        start = (cx, top_y - 9.0)
        stop_y = comb[1] + 6.5 + 2.0
        span = start[1] - comb[1]
        t = (start[1] - stop_y) / span if span else 0.0
        arrow(ax, start, (cx + t * (comb[0] - cx), stop_y),
              color=mix(C_FLAT, 65))
    round_box(ax, (comb[0] - 10.0, comb[1] - 6.5), 20.0, 13.0,
              edge=C_FLAT, fill="white", radius=6.0)
    ax.text(*comb, "∏", ha="center", va="center",
            fontsize=7.2, color=C_FLAT)
    # Parallel to the nested card's "s x G1G2G3 in order", so the product
    # node is grounded by the expression directly beneath it: the same three
    # gains, combined without an order.  The bare product glyph appears
    # nowhere in the manuscript, so it cannot stand unexplained.
    formula(ax, x0 + card_w / 2, card_y + 0.17 * card_h,
            (("h", 0.0), ("1", -1.6), ("h", 0.0), ("2", -1.6),
             ("h", 0.0), ("3", -1.6), (" in any order", 0.0)))

    # Local ratio: excitatory and inhibitory observations already meet in
    # each module; no across-stage cancellation is required.
    x0 = cards[2][0]
    centres = np.linspace(x0 + 0.20 * card_w, x0 + 0.80 * card_w, 3)
    pair_y = card_y + 0.62 * card_h
    ratio_y = card_y + 0.38 * card_h
    for index, cx in enumerate(centres, start=1):
        ax.plot([cx - 5.0], [pair_y], marker="o", ms=4.2, mfc=mix("exc", 24),
                mec=COLORS["exc"], mew=LW_EDGE, ls="none", zorder=4)
        ax.plot([cx + 5.0], [pair_y], marker="o", ms=4.2, mfc=mix("inh", 24),
                mec=COLORS["inh"], mew=LW_EDGE, ls="none", zorder=4)
        ax.text(cx - 5.0, pair_y + 8.0, "E", ha="center", va="center",
                fontsize=PT_SMALL, color=COLORS["exc"])
        ax.text(cx + 5.0, pair_y + 8.0, "I", ha="center", va="center",
                fontsize=PT_SMALL, color=COLORS["inh"])
        arrow(ax, (cx, pair_y - 3.0), (cx, ratio_y + 7.0),
              color=mix(C_LOCAL_RATIO, 65))
        round_box(ax, (cx - 9.0, ratio_y - 6.0), 18.0, 12.0,
                  edge=C_LOCAL_RATIO, fill=mix(C_LOCAL_RATIO, 18), radius=2.0)
        ax.text(cx, ratio_y, f"r{index}", ha="center", va="center",
                fontsize=PT_SMALL, color=INK)
    # "is local" is the card's own title, so the expression alone is enough
    # and stays inside the 74.6 pt card once its runs are at token size.
    formula(ax, x0 + card_w / 2, card_y + 0.13 * card_h,
            (("r", 0.0), ("l", -1.6), (" = x", 0.0), ("l", -1.6),
             ("E", 1.9), (" / x", 0.0), ("l", -1.6), ("I", 1.9)))

    # The H and alpha definitions are caption material, not panel furniture.


# ── B: architecture schematics ──────────────────────────────────────────
def _soma(ax, x, y):
    ax.add_patch(Circle((x, y), 3.2, facecolor=COLORS["soma"],
                        edgecolor=mix("ink", 30), lw=LW_EDGE, zorder=5))


def panel_architectures(ax):
    w, h = point_frame(ax)
    centres = (0.16 * w, 0.50 * w, 0.84 * w)
    y0 = 0.24 * h
    module_y = 0.66 * h
    bracket_y = 0.87 * h

    x = centres[0]
    _soma(ax, x, y0)
    prev = (x, y0 + 3.3)
    for level in range(3):
        cy = y0 + 14.0 + level * 13.0
        ax.plot([prev[0], x], [prev[1], cy - 4.5], color=C_SERIAL,
                lw=LW_DATA, solid_capstyle="round")
        # 28 pt, not 18: the 25.4 pt "stage N" label overran the capsule by
        # 3.8 pt on each side and the border stroke cut through the glyphs.
        round_box(ax, (x - 14.0, cy - 4.5), 28.0, 9.0,
                  edge=C_SERIAL, fill=mix(C_SERIAL, 20 + 14 * level),
                  radius=2.0)
        ax.text(x, cy, f"stage {level + 1}", ha="center", va="center",
                fontsize=PT_SMALL, color=INK)
        prev = (x, cy + 4.5)
    ax.text(x, 0.14 * h, "serial tree", ha="center", va="center",
            fontsize=7.2, color=C_SERIAL)
    ax.text(x, 0.045 * h, "Dₚ = 3\nserial",
            ha="center", va="center", linespacing=1.05,
            fontsize=PT_SMALL, color=MUTE)

    x = centres[1]
    _soma(ax, x, y0)
    # 16 pt boxes on a 15 pt pitch overlapped by 1 pt, so the three modules
    # touched.  13 pt boxes on a 16.5 pt pitch leave a 3.5 pt gap and still
    # hold the 9.6 pt "mN" label.
    offsets = (-16.5, 0.0, 16.5)
    for index, dx in enumerate(offsets, start=1):
        cy = module_y
        round_box(ax, (x + dx - 6.5, cy - 4.5), 13.0, 9.0,
                  edge=C_GROUPED, fill=mix(C_GROUPED, 12 + 11 * index),
                  radius=2.0)
        ax.text(x + dx, cy, f"m{index}", ha="center", va="center",
                fontsize=PT_SMALL, color=INK)
        ax.plot([x + dx, x], [cy - 4.5, y0 + 3.3], color=C_GROUPED,
                lw=LW_EDGE, solid_capstyle="round")
    ax.text(x, 0.14 * h, "grouped point", ha="center", va="center",
            fontsize=7.2, color=C_GROUPED)
    ax.text(x, 0.045 * h, "same modules\nparallel", ha="center", va="center",
            linespacing=1.05, fontsize=PT_SMALL, color=MUTE)

    x = centres[2]
    layer_y = (y0, y0 + 21.0, y0 + 42.0)
    counts = (3, 4, 3)
    layers = []
    for cy, count in zip(layer_y, counts, strict=True):
        nodes = [(x + dx, cy) for dx in np.linspace(-12.0, 12.0, count)]
        layers.append(nodes)
    for lower, upper in zip(layers[:-1], layers[1:], strict=True):
        for px, py in lower:
            for qx, qy in upper:
                ax.plot([px, qx], [py, qy], color=mix(C_POINT, 48),
                        lw=LW_HAIR, solid_capstyle="round", zorder=1)
    for nodes in layers:
        for px, py in nodes:
            ax.plot([px], [py], marker="o", ms=3.5, mfc="white",
                    mec=C_POINT, mew=LW_EDGE, ls="none", zorder=4)
    ax.text(x, 0.14 * h, "point MLP", ha="center", va="center",
            fontsize=7.2, color=C_POINT)
    ax.text(x, 0.045 * h, "flexible\nceiling", ha="center",
            va="center", linespacing=1.05, fontsize=PT_SMALL, color=MUTE)

    ax.plot([centres[0] - 20.0, centres[1] + 20.0],
            [bracket_y, bracket_y],
            color=MUTE, lw=LW_HAIR)
    for bx in (centres[0] - 20.0, centres[1] + 20.0):
        ax.plot([bx, bx], [bracket_y, bracket_y - 3.0],
                color=MUTE, lw=LW_HAIR)
    ax.text((centres[0] + centres[1]) / 2, bracket_y + 3.0,
            "identical contacts and trainable resources", ha="center",
            va="bottom", fontsize=PT_SMALL, color=MUTE)


# ── C: H=3 boundary ─────────────────────────────────────────────────────
def panel_h3(ax, depth_summary, point_summary):
    labels = []
    ycols = ("mean_test_accuracy", "ci95_low_test_accuracy",
             "ci95_high_test_accuracy")
    labels.append(accuracy_curve(
        ax, depth_summary, color=C_SERIAL_BP, marker="o", label="serial BP",
        y_columns=ycols, regime="aligned", mechanism="shunting",
        method="bp", transport="backpropagation"))
    labels.append(accuracy_curve(
        ax, depth_summary, color=C_PATH, marker="^", label="exact-path LocalCA",
        y_columns=ycols, regime="aligned", mechanism="shunting",
        method="local3f", transport="path_transport", dx=-0.025))
    labels.append(accuracy_curve(
        ax, depth_summary, color=C_SHARED, marker="s", label="shared signal",
        y_columns=ycols, regime="aligned", mechanism="shunting",
        method="local3f", transport="per_soma_shared", dx=0.025))
    labels.append(accuracy_curve(
        ax, point_summary, color=C_GROUPED, marker="D", label="grouped point",
        y_columns=ycols, regime="aligned", architecture="all_active_star",
        credit="full_bp", filled=False, dashes=(0, (3.0, 2.0))))
    labels.append(accuracy_curve(
        ax, depth_summary, color=C_ADDITIVE, marker="v", label="raw additive",
        y_columns=ycols, regime="aligned", mechanism="additive",
        method="bp", transport="backpropagation", filled=False,
        dashes=(0, (1.5, 1.5))))

    point = point_summary[
        point_summary.regime.eq("aligned")
        & point_summary.architecture.eq("point_mlp_total")
        & point_summary.credit.eq("full_bp")
    ].iloc[0]
    ceiling = float(point.mean_test_accuracy)
    ax.axhspan(float(point.ci95_low_test_accuracy),
               float(point.ci95_high_test_accuracy), color=mix(C_POINT, 8),
               zorder=0)
    ax.axhline(ceiling, color=C_POINT, lw=LW_REF,
               ls=(0, (5.0, 2.2)), zorder=1)

    ax.set_xlim(0.78, 4.03)
    ax.set_xticks([1, 2, 3], ["D1", "D2", "D3"])
    ax.set_ylim(0.48, 1.025)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("serial physical depth Dₚ")
    ax.set_ylabel("test accuracy")

    direct = [
        (ceiling, "point-network ceiling", C_POINT),
        (labels[0][1], labels[0][2], C_SERIAL_BP),
        (labels[1][1], labels[1][2], C_PATH),
        (labels[2][1], labels[2][2], AMBER_TEXT),
        (labels[3][1], labels[3][2], C_GROUPED),
        (labels[4][1], labels[4][2], C_ADDITIVE),
    ]
    offsets = (0.000, 0.006, -0.006, 0.000, 0.000, 0.000)
    for index, ((value, label, color), dy) in enumerate(
            zip(direct, offsets, strict=True)):
        # The ceiling is an axhline spanning the whole axes, so its own label
        # sat on the rule and was struck through; the curve labels end at
        # D3 and need no backing.
        backing = (dict(facecolor="white", edgecolor="none", pad=0.8)
                   if index == 0 else None)
        ax.text(3.12, value + dy, label, ha="left", va="center",
                fontsize=PT_SMALL, color=color, bbox=backing)


# ── D: H=4 aligned/reversed boundary ────────────────────────────────────
def h4_matrix(summary, regime):
    """Row-by-depth means and their 95% paired-seed bootstrap bounds."""
    matrix = np.full((len(H4_ROWS), 4), np.nan)
    low = np.full((len(H4_ROWS), 4), np.nan)
    high = np.full((len(H4_ROWS), 4), np.nan)
    for row_index, (_, architecture, mechanism, credit) in enumerate(H4_ROWS):
        rows = summary[
            summary.regime.eq(regime)
            & summary.architecture.eq(architecture)
            & summary.mechanism.eq(mechanism)
            & summary.credit.eq(credit)
        ]
        for row in rows.itertuples(index=False):
            col = int(row.depth) - 1
            matrix[row_index, col] = float(row.mean_test_accuracy)
            low[row_index, col] = float(row.ci_low)
            high[row_index, col] = float(row.ci_high)
    return matrix, low, high


def panel_h4(ax, summary):
    aligned, aligned_lo, aligned_hi = h4_matrix(summary, "aligned")
    reversed_, _, _ = h4_matrix(summary, "rewired_tree")
    combined = np.concatenate(
        [aligned, np.full((len(H4_ROWS), 1), np.nan), reversed_], axis=1)
    col_labels = ("D1", "D2", "D3", "D4", "", "D1", "D2", "D3", "D4")
    image = heatmap(ax, combined, [row[0] for row in H4_ROWS], col_labels,
                    best_by_row=False)

    # An outline is a claim that this depth is the best of its row, so draw it
    # only where the bootstrap intervals actually support one: the best cell's
    # lower bound must clear the runner-up's upper bound.  The old rule marked
    # any non-tied maximum whose row spread exceeded 0.01, which outlined
    # differences whose intervals overlap -- and, in the reversed block, a
    # 0.8 pp blip on an otherwise flat row, contradicting the result that
    # reversing sensor order removes the depth benefit.  Reversed placement is
    # therefore left unmarked.
    for row in range(aligned.shape[0]):
        finite = np.flatnonzero(np.isfinite(aligned[row]))
        if len(finite) < 2:
            continue
        order = finite[np.argsort(aligned[row, finite])[::-1]]
        best, runner_up = int(order[0]), int(order[1])
        if aligned_lo[row, best] <= aligned_hi[row, runner_up]:
            continue
        x0, x1 = best - 0.46, best + 0.46
        y0, y1 = row - 0.43, row + 0.43
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                color=INK, lw=LW_ERR, solid_joinstyle="miter",
                solid_capstyle="butt", zorder=5)

    ax.axvline(4.0, color="white", lw=LW_DATA, zorder=4)
    ax.text(1.5, 1.01, "aligned sensors", ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=7.2,
            color=C_SERIAL, clip_on=False)
    ax.text(6.5, 1.01, "reversed sensors", ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=7.2,
            color=MUTE, clip_on=False)
    ax.set_xlabel("serial physical depth Dₚ")
    return image


# ── E: hierarchy x physical depth ───────────────────────────────────────
def hierarchy_matrix(h4_seed):
    h23 = pd.read_csv(
        SOURCE / "physical_depth_clean_source_replication" / "seed_outcomes.csv"
    )
    common = pd.concat([h23, h4_seed], ignore_index=True, sort=False)
    common = common[
        common.hierarchy.isin([2, 3, 4])
        & common.regime.eq("aligned")
        & common.architecture.eq("serial_tree")
        & common.mechanism.eq("shunting")
        & common.credit.eq("full_bp")
    ]
    matrix = np.full((3, 4), np.nan)
    for (hierarchy, depth), rows in common.groupby(["hierarchy", "depth"]):
        matrix[int(hierarchy) - 2, int(depth) - 1] = rows.test_accuracy.mean()
    return matrix


# ── F,G: task-family alignment dose response ────────────────────────────
def panel_alignment(ax, effects, credit, *, left):
    for family, label, color, marker in FAMILY_SPECS:
        rows = effects[
            effects.family.eq(family) & effects.credit.eq(credit)
        ].sort_values("alignment_alpha")
        x = rows.alignment_alpha.to_numpy(float)
        mean = 100.0 * rows.mean_difference.to_numpy(float)
        low = 100.0 * rows.ci95_low.to_numpy(float)
        high = 100.0 * rows.ci95_high.to_numpy(float)
        ax.plot(x, mean, color=color, lw=LW_DATA, marker=marker,
                ms=MARKER_MS, mfc=color, mec="white", mew=LW_EDGE,
                solid_capstyle="round", zorder=2)
        ax.errorbar(x, mean, yerr=np.vstack([mean - low, high - mean]),
                    fmt="none", ecolor=color, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE, zorder=3)
        # The zero rule runs the full axis, so a family that ends near zero
        # ("local ratios" ends at -0.7 pp) had its label struck through.
        ax.text(1.035, mean[-1], label, ha="left", va="center",
                fontsize=PT_SMALL, color=color,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

    ax.axhline(0, color=MUTE, lw=LW_REF, ls=(0, (3.0, 2.2)), zorder=1)
    ax.set_xlim(-0.06, 1.48)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ax.set_ylim(-10.5, 33.5)
    ax.set_yticks([-10, 0, 10, 20, 30])
    ax.set_xlabel("task–sensor alignment α")
    if left:
        ax.set_ylabel("serial − grouped point (pp)")
    else:
        ax.tick_params(axis="y", labelleft=False)


def build() -> list[str]:
    depth_summary = pd.read_csv(
        SOURCE / "nonlinear_physical_depth_confirmatory" / "condition_summary.csv"
    )
    point_summary = pd.read_csv(
        SOURCE / "point_dendrite_credit_controls" / "condition_summary.csv"
    )
    h4_summary = pd.read_csv(
        SOURCE / "physical_depth_h4_factorial" / "condition_summary.csv"
    )
    h4_seed = pd.read_csv(
        SOURCE / "physical_depth_h4_factorial" / "seed_outcomes.csv"
    )
    task_effects = pd.read_csv(
        SOURCE / "task_family_alignment" / "architecture_effects.csv"
    )

    canvas = NativeCanvas(
        CANVAS_H_PT / 72.0, 3, row_weights=list(ROW_H_PT),
        hgutter_pt=HGUTTER, vgutter_pt=VGUTTER, margins=MARGINS,
        letters=False,
    )

    ax_a = canvas.panel("A", 0, 0, 7, schematic=True,
                        title="Task structure, held apart from model depth")
    ax_b = canvas.panel("B", 0, 7, 5, schematic=True,
                        title="Architectures compared")
    ax_c = canvas.panel("C", 1, 0, 6, grid="y",
                        title="Hₚ=3: matched serial computation")
    ax_d = canvas.panel("D", 1, 6, 6,
                        title="Hₚ=4: alignment exposes the boundary")
    ax_e = canvas.panel("E", 2, 0, 4,
                        title="Serial BP depth optimum")
    ax_f = canvas.panel("F", 2, 4, 4, grid="y",
                        title="Alignment dose, BP")
    ax_g = canvas.panel("G", 2, 8, 4, grid="y", sharey=ax_f,
                        title="Alignment dose, path LocalCA")

    panel_task_families(ax_a)
    panel_architectures(ax_b)
    panel_h3(ax_c, depth_summary, point_summary)
    image = panel_h4(ax_d, h4_summary)
    ax_d.set_title("Hₚ=4: alignment exposes the boundary", pad=14.0)

    hierarchy = hierarchy_matrix(h4_seed)
    heatmap(ax_e, hierarchy, ("Hₚ=2", "Hₚ=3", "Hₚ=4"),
            ("D1", "D2", "D3", "D4"), best_by_row=True)
    for row, col in np.argwhere(~np.isfinite(hierarchy)):
        ax_e.text(col, row, "—", ha="center", va="center",
                  fontsize=PT_SMALL, color=MUTE, zorder=5)
    ax_e.set_xlabel("serial physical depth Dₚ")
    ax_e.set_ylabel("task gain tiers Hₚ")

    panel_alignment(ax_f, task_effects, "bp", left=True)
    panel_alignment(ax_g, task_effects, "local3f", left=False)

    style_panel(ax_c, grid="y")
    style_panel(ax_f, grid="y")
    style_panel(ax_g, grid="y")

    for name, dx in (
        ("A", 30.0), ("B", 14.0), ("C", 34.0), ("D", 30.0),
        ("E", 32.0), ("F", 34.0), ("G", 14.0),
    ):
        canvas.add_letter(name, canvas.axes[name], dx_pt=dx)

    color_rail(canvas, image, ax_d)
    return canvas.save(OUTPUT, name="main_figure_06_native")


def main() -> None:
    problems = build()
    if problems:
        raise SystemExit(
            f"main_figure_06_native: {len(problems)} layout/overlap problems"
        )
    print("main_figure_06_native: clean")


if __name__ == "__main__":
    main()

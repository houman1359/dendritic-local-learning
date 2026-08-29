#!/usr/bin/env python3
"""Final Main Figure 5 -- partial addresses in a hierarchical task.

The eight-context subtree-address factorial is shown as its own experiment.
The canvas makes the task hierarchy and every load-bearing control explicit
before showing the bandwidth sweep: correct ancestry routes, incorrect route
assignment, a degree/depth-matched rewired tree and matched non-anatomical
rank/sparsity controls.  All numerical panels read the unchanged frozen
source-data tables used by the former combined Figure 3.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.textpath import TextPath

from routing_figure_panels import (
    SUBTREE,
    address_ladder_compact,
    bandwidth_sweep_compact,
    route_contrasts_compact,
    topology_alignment_compact,
)
from credit_tree_schematics import mix
from figure_canvas import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    PT_ANNOT,
    PT_LABEL,
    PT_SMALL,
    Margins,
    NativeCanvas,
    token_subscript,
)
from native_schematics import Frame


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "figures" / "components" / "main_figure_05_native.pdf"
CANONICAL = ROOT / "figures" / "main" / "figure_05.pdf"

CANVAS_H_PT = 440.0
HEIGHT_IN = CANVAS_H_PT / 72.0
ROW_PT = [139.0, 101.0, 105.0]
HGUTTER_PT = 34.0
VGUTTER_PT = 37.0

INK = COLORS["ink"]
MUTE = COLORS["mute"]
GREEN = COLORS["shunting"]
PURPLE = COLORS["oracle"]
GRAY = COLORS["point_mlp"]
# Route index is an ordinal series, so the four routes use the manuscript
# slate ramp rather than reserved condition hues (green = correct routing,
# amber = neuron-shared, blue = Fig 2 additive, violet = oracle/best control
# in E and F).  The ramp's three published steps are #9AA5B4 / #5F6B7E /
# #2E3947; the second entry is their light-to-mid midpoint, added here only
# because this schematic draws four routes.
ROUTE_COLORS = ("#9AA5B4", "#7C8899", "#5F6B7E", "#2E3947")


def _span_pt(text: str, size: float) -> float:
    """Advance width of ``text`` in points at the figure's own sans face."""
    if not text:
        return 0.0
    path = TextPath((0.0, 0.0), text, size=size,
                    prop=FontProperties(size=size))
    return float(path.get_extents().x1)


def _centred_subscript(f: Frame, xy, base, sub, tail="", *, color,
                       size=PT_SMALL, va="center") -> None:
    """Centred base+subscript+tail chain drawn with token type sizes only.

    ``token_subscript`` anchors its chain on the left, so the group is
    measured first and the base span is shifted by half the total advance.
    A long tail is set as its own span sharing the base's baseline: the
    helper's chained tail re-anchors on the subscript's box and would leave
    a running sentence sitting a descender below the words before it.
    """
    head_pt = _span_pt(base, size) + 0.4 + _span_pt(sub, size)
    width_pt = head_pt + ((0.6 + _span_pt(tail, size)) if tail else 0.0)
    left = xy[0] - f.fx(width_pt / 2.0)
    token_subscript(f.ax, left, xy[1], base, sub,
                    size=size, sub_size=size, color=color, ha="left", va=va,
                    zorder=6, clip_on=False)
    if tail:
        f.text((left + f.fx(head_pt + 0.6), xy[1]), tail, size=size,
               color=color, ha="left", va=va, zorder=6)


def hierarchical_task(ax) -> None:
    """Eight active streams whose distractors follow binary-tree distance."""
    f = Frame(ax, labels=True, scale=0.95)
    f.text((0.50, 0.955),
           "context c = 3 selects stream 3; selected stream carries +y; all streams active",
           size=PT_SMALL, color=INK, ha="center", va="top")

    leaf_x = np.linspace(0.055, 0.945, 8)
    levels = (
        (np.asarray([0.50]), 0.82),
        (np.asarray([0.275, 0.725]), 0.66),
        (np.asarray([0.105, 0.335, 0.665, 0.895]), 0.49),
        (leaf_x, 0.31),
    )
    selected = 2
    # Tree edges, with the selected ancestry path highlighted.
    for level in range(3):
        parents, py = levels[level]
        children, cy = levels[level + 1]
        for child_idx, cx in enumerate(children):
            parent_idx = child_idx // 2
            on_path = (
                (level == 0 and parent_idx == 0 and child_idx == 0)
                or (level == 1 and parent_idx == 0 and child_idx == 1)
                or (level == 2 and parent_idx == 1 and child_idx == selected)
            )
            f.leader((parents[parent_idx], py), (cx, cy),
                     color=GREEN if on_path else COLORS["edge"],
                     lw=LW_DATA if on_path else LW_EDGE)
    for level, (nodes, y) in enumerate(levels):
        for idx, x in enumerate(nodes):
            on_path = ((level == 0) or
                       (level == 1 and idx == 0) or
                       (level == 2 and idx == 1) or
                       (level == 3 and idx == selected))
            f.disc((x, y), 3.3 if level < 3 else 3.7,
                   fill="white" if level < 3 else
                   (GREEN if idx == selected else COLORS["panel_bg"]),
                   edge=GREEN if on_path else COLORS["edge"], lw=LW_EDGE)

    evidence = (
        "−0.45y", "−0.45y", "+y", "−0.15y",
        "−0.75y", "−0.75y", "−0.75y", "−0.75y",
    )
    for idx, (x, val) in enumerate(zip(leaf_x, evidence, strict=True)):
        color = GREEN if idx == selected else MUTE
        _centred_subscript(f, (x, 0.215), "c", str(idx + 1), color=color)
        f.text((x, 0.115), val, size=PT_SMALL, color=color)

    # The three brackets name the increasing distractor strength without
    # repeating prose under every leaf.
    bracket_y = 0.035
    groups = ((0, 1, "same half"), (3, 3, "sibling"), (4, 7, "other half"))
    for start, end, label in groups:
        left = leaf_x[start] - 0.025
        right = leaf_x[end] + 0.025
        f.rule(bracket_y + 0.018, left, right, color=COLORS["grid"], lw=LW_HAIR)
        f.text(((left + right) / 2.0, bracket_y), label,
               size=PT_SMALL, color=MUTE, va="top")


def _card(f: Frame, rect, title, subtitle, *, accent):
    x0, y0, width, height = rect
    f.group(rect, tint=mix("shunting", 6) if accent == GREEN else COLORS["panel_bg"],
            edge=mix("shunting", 45) if accent == GREEN else COLORS["grid"])
    f.text((x0 + frame_dx(f, 5.0), y0 + height - f.fy(6.0)), title,
           size=PT_ANNOT, color=accent, ha="left", va="top")
    f.text((x0 + frame_dx(f, 5.0), y0 + height - f.fy(17.0)), subtitle,
           size=PT_SMALL, color=MUTE, ha="left", va="top")


def frame_dx(f: Frame, pt: float) -> float:
    return f.fx(pt)


# The eight task contexts fall into four groups of two; a matched route is
# exactly one of those groups.  Colouring the leaves by group and the route
# bars by the signal they carry lets one encoding separate all three
# tree-structured controls: matched = bar colour equals its leaves' colour;
# deranged = right leaves, wrong colour; rewired = right colour, wrong leaves.
LEAF_GROUP = (0, 0, 1, 1, 2, 2, 3, 3)
ROUTE_SUPPORTS = {
    "matched": ((0, 1), (2, 3), (4, 5), (6, 7)),
    "deranged": ((0, 1), (2, 3), (4, 5), (6, 7)),
    # same four routes, same two leaves each (degree and depth unchanged),
    # but the pairs no longer follow the task's own grouping
    "rewired": ((0, 4), (1, 6), (2, 5), (3, 7)),
}
ROUTE_SIGNAL = {
    "matched": (0, 1, 2, 3),
    "deranged": (2, 3, 0, 1),   # the permutation the control applies
    "rewired": (0, 1, 2, 3),
}


def _bar_levels(supports):
    """Greedy vertical levels so overlapping route spans never collide."""
    levels, ends = [], []
    for lo, hi in supports:
        for level, last in enumerate(ends):
            if lo > last:
                levels.append(level)
                ends[level] = hi
                break
        else:
            levels.append(len(ends))
            ends.append(hi)
    return levels


def _route_bars(f: Frame, rect, mode: str, *, name_rows: bool = False) -> None:
    """Four rank-matched routes over the eight terminal dendritic branches."""
    x0, y0, width, height = rect
    xs = np.linspace(x0 + 0.09 * width, x0 + 0.91 * width, 8)
    leaf_y = y0 + 0.13 * height
    supports = ROUTE_SUPPORTS[mode]
    signals = ROUTE_SIGNAL[mode]
    levels = _bar_levels(supports)
    span = 0.66 * height
    step = span / max(len(set(levels)), 1)

    # Each leaf is drawn as a terminal dendritic branch -- a short stub
    # continuing toward the soma, capped by the addressable terminal itself --
    # so the row reads as branches rather than as anonymous circles.  The cap
    # is tinted by the task group whose stream that branch carries, which is
    # what makes a group-mixing route visible.
    for idx, xpos in enumerate(xs):
        tone = ROUTE_COLORS[LEAF_GROUP[idx]]
        f.ax.plot([xpos, xpos], [leaf_y - f.fy(5.0), leaf_y - f.fy(1.4)],
                  color=mix("dend", 45), lw=LW_HAIR, solid_capstyle="round",
                  zorder=3)
        f.disc((xpos, leaf_y), 2.6, fill=mix(tone, 26), edge=tone,
               lw=LW_EDGE, zorder=4)

    for route, (lo, hi) in enumerate(supports):
        color = ROUTE_COLORS[signals[route]]
        bar_y = leaf_y + 0.22 * height + levels[route] * step
        left, right = xs[lo], xs[hi]
        f.ax.plot([left, right], [bar_y, bar_y], color=color, lw=2.6,
                  solid_capstyle="round", zorder=3)
        # one leg per covered leaf: the route's support, drawn explicitly
        for leaf in (lo, hi):
            f.ax.plot([xs[leaf], xs[leaf]],
                      [bar_y - f.fy(1.0), leaf_y + f.fy(3.2)],
                      color=color, lw=LW_HAIR, solid_capstyle="round",
                      zorder=2)

    if name_rows:
        top = leaf_y + 0.22 * height + max(levels) * step
        f.text((x0 + 0.50 * width, top + f.fy(7.5)), "K = 4 feedback routes",
               size=PT_SMALL, color=MUTE, va="bottom")
        f.text((x0 + 0.50 * width, leaf_y - f.fy(8.0)),
               "8 terminal branches", size=PT_SMALL, color=MUTE, va="top")


def _nonanatomical_basis(f: Frame, rect) -> None:
    """Dense signed rank-four basis over the SAME eight addresses.

    Drawn on the other cards' geometry -- the eight leaves sit where they sit
    everywhere else and each matrix column is one of them -- so the reader can
    see directly that this control keeps the feedback bandwidth (four rows)
    while abandoning contiguous subtree support (every row touches addresses
    from several task groups).
    """
    x0, y0, width, height = rect
    xs = np.linspace(x0 + 0.09 * width, x0 + 0.91 * width, 8)
    leaf_y = y0 + 0.13 * height
    matrix = np.asarray([
        [1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1],
        [1, 0, -1, 1, 0, -1, 1, -1],
        [0, 1, 1, -1, -1, 1, -1, 0],
    ])
    for idx, xpos in enumerate(xs):
        tone = ROUTE_COLORS[LEAF_GROUP[idx]]
        f.ax.plot([xpos, xpos], [leaf_y - f.fy(5.0), leaf_y - f.fy(1.4)],
                  color=mix("dend", 45), lw=LW_HAIR, solid_capstyle="round",
                  zorder=3)
        f.disc((xpos, leaf_y), 2.6, fill=mix(tone, 26), edge=tone,
               lw=LW_EDGE, zorder=4)

    cell_w = (xs[1] - xs[0]) * 0.86
    cell_h = 0.13 * height
    base_y = leaf_y + 0.20 * height
    for row in range(4):
        y = base_y + (3 - row) * cell_h
        # a chip in the route ramp names the row as one feedback channel
        # inset so the four chips read as four rows, not one continuous rail
        f.ax.plot([x0 + 0.042 * width, x0 + 0.042 * width],
                  [y + 0.16 * cell_h, y + 0.84 * cell_h],
                  color=ROUTE_COLORS[row], lw=2.6, solid_capstyle="butt",
                  zorder=4)
        for col in range(8):
            value = matrix[row, col]
            face = (mix("oracle", 30) if value > 0 else
                    mix("point_mlp", 34) if value < 0 else "white")
            f.ax.add_patch(Rectangle(
                (xs[col] - cell_w / 2.0, y), cell_w, cell_h,
                facecolor=face, edgecolor="white", linewidth=LW_HAIR,
                transform=f.ax.transData, zorder=3,
            ))

    # Compact signed-weight key.  Without it, the purple/gray/white cells can
    # be mistaken for four categorical route identities rather than entries
    # of a dense signed basis.
    key_y = y0 + 0.955 * height
    key_x = x0 + 0.48 * width
    key_specs = ((mix("oracle", 30), "+"),
                 (mix("point_mlp", 34), "−"),
                 ("white", "0"))
    f.text((key_x - f.fx(4.0), key_y), "weight", size=PT_SMALL,
           color=MUTE, ha="right", va="center")
    for index, (face, symbol) in enumerate(key_specs):
        xpos = key_x + index * f.fx(13.0)
        f.ax.add_patch(Rectangle(
            (xpos, key_y - f.fy(3.0)), f.fx(6.0), f.fy(6.0),
            facecolor=face, edgecolor=COLORS["grid"], linewidth=LW_HAIR,
            transform=f.ax.transData, zorder=4,
        ))
        f.text((xpos + f.fx(8.0), key_y), symbol, size=PT_SMALL,
               color=INK, ha="left", va="center")

    # One cell is one route's signed weight on one branch: the column sits
    # over the branch it weights and the row carries its route chip.  A label
    # above the matrix collided with the card subtitle, so the caption names
    # the cell and the row chips carry the route identity here.
    f.text((x0 + 0.50 * width, leaf_y - f.fy(8.0)),
           "8 terminal branches", size=PT_SMALL, color=MUTE, va="top")


def route_controls(ax, indices) -> None:
    """Visual dictionary for selected control questions at K=4."""
    f = Frame(ax, labels=True, scale=0.94)
    all_specs = (
        ("matched subtrees", "task groups = subtrees", GREEN),
        ("deranged assignment", "same routes, wrong signals", GRAY),
        ("rewired tree", "same degree and depth", GRAY),
        ("non-anatomical basis", "same feedback bandwidth", PURPLE),
    )
    chosen = tuple((index, all_specs[index]) for index in indices)
    cells = f.split(len(chosen), axis="x", gap_pt=8.0,
                    pad_pt=(0, 0, 0, 0))
    for rect, (_, (title, subtitle, accent)) in zip(cells, chosen, strict=True):
        _card(f, rect, title, subtitle, accent=accent)
    # Lower drawing rooms deliberately use identical geometry.
    for (index, _), rect in zip(chosen, cells, strict=True):
        x0, y0, width, height = rect
        room = (x0, y0 + 0.07 * height, width, 0.67 * height)
        if index == 0:
            _route_bars(f, room, "matched", name_rows=True)
        elif index == 1:
            _route_bars(f, room, "deranged")
        elif index == 2:
            _route_bars(f, room, "rewired")
        else:
            _nonanatomical_basis(f, room)



def _move_label(ax, text: str, xy) -> None:
    """Re-anchor one in-panel direct label without changing its wording."""
    for artist in ax.texts:
        if artist.get_text() == text:
            artist.set_position(xy)
            return
    raise LookupError(f"direct label {text!r} not found")


def _rename_label(ax, old: str, new: str) -> None:
    """Rename one direct series label without touching plotted values."""
    for artist in ax.texts:
        if artist.get_text() == old:
            artist.set_text(new)
            return
    raise LookupError(f"direct label {old!r} not found")


def _annotate_k4_advantage(ax, contrasts: pd.DataFrame) -> None:
    """Expose the small but prespecified K=4 interior contrast on its scale."""
    row = contrasts[
        contrasts.architecture.eq("dendritic_tree")
        & contrasts.endpoint.eq("heldout_accuracy")
        & contrasts.contrast.eq(
            "correct - best_matched_nonanatomical_oracle")
        & contrasts.budget_k.eq(4)
    ]
    if len(row) != 1:
        raise ValueError(f"expected one K=4 matched-control contrast, got {len(row)}")
    value = 100.0 * float(row.iloc[0].mean_difference)
    ax.annotate(f"K=4: +{value:.2f} pp", xy=(2.0, value), xytext=(2.74, 14.0),
                textcoords="data", ha="right", va="center",
                fontsize=PT_SMALL, color=PURPLE,
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                          edgecolor=COLORS["grid"], linewidth=LW_HAIR),
                arrowprops=dict(arrowstyle="-", color=PURPLE, lw=LW_HAIR,
                                shrinkA=2.0, shrinkB=3.0), zorder=7)


def build() -> list:
    mpl.rcParams["lines.markeredgewidth"] = LW_EDGE
    outcomes = pd.read_csv(SUBTREE / "seed_outcomes.csv")
    summary = pd.read_csv(SUBTREE / "condition_summary.csv")
    contrasts = pd.read_csv(SUBTREE / "paired_contrasts.csv")
    dendritic = summary[summary.architecture.eq("dendritic_tree")]

    canvas = NativeCanvas(
        HEIGHT_IN, 3, row_weights=ROW_PT,
        hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
        margins=Margins(left=43.0, right=12.0, top=22.0, bottom=26.0),
    )
    ax_a = canvas.panel("A", 0, 0, 7, schematic=True,
                        title="Eight-context hierarchical task")
    ax_b = canvas.panel("B", 0, 7, 5, schematic=True,
                        title="Feedback bandwidth within one neuron")
    ax_c = canvas.panel("C", 1, 0, 6, schematic=True,
                        title="Route-assignment controls at K = 4")
    ax_d = canvas.panel("D", 1, 6, 6, schematic=True,
                        title="Topology and basis controls at K = 4")
    ax_e = canvas.panel("E", 2, 0, 4,
                        title="Learning across bandwidth")
    ax_f = canvas.panel("F", 2, 4, 4,
                        title="Route assignment")
    ax_g = canvas.panel("G", 2, 8, 4,
                        title="Matched versus rewired topology")

    hierarchical_task(ax_a)
    address_ladder_compact(ax_b)
    route_controls(ax_c, (0, 1))
    route_controls(ax_d, (2, 3))
    bandwidth_sweep_compact(ax_e, outcomes, dendritic)
    _rename_label(ax_e, "best control", "best non-anatomical")
    route_contrasts_compact(ax_f, contrasts)
    # "vs best" sat on the K = 1 diamond and its lower whisker; move it into
    # the empty well under the rising violet segment.
    _move_label(ax_f, "vs best", (0.52, -46.0))
    _rename_label(ax_f, "vs best", "vs best non-anatomical")
    _annotate_k4_advantage(ax_f, contrasts)
    topology_alignment_compact(ax_g, outcomes)

    COMPONENT.parent.mkdir(parents=True, exist_ok=True)
    problems = canvas.save(COMPONENT, name="main_figure_05_native")
    CANONICAL.parent.mkdir(parents=True, exist_ok=True)
    CANONICAL.write_bytes(COMPONENT.read_bytes())
    return problems


def main() -> None:
    problems = build()
    for problem in problems:
        print(f"  {problem}")
    print(f"  canvas width {FIG_W * 72:.1f} pt, height {CANVAS_H_PT:.1f} pt")


if __name__ == "__main__":
    main()

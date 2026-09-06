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
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
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
# The published figures/main copy is emitted ONLY by
# assemble_compact_main_figures.py, whose FIGURE_SOURCES map assigns
# this component its publication number; a builder-side copy would
# bypass the 2026-09-04 dictionary-forward renumbering.

CANVAS_H_PT = 550.0
HEIGHT_IN = CANVAS_H_PT / 72.0
ROW_PT = [139.0, 101.0, 105.0, 90.0]
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
# Leaf-cap fills reuse the same ordinal ramp, but a flat 26% wash of each tone
# printed as four near-identical pale grays (#E5E8EC..#C9CCCF), so "cap = task
# group" was unreadable at print size.  Widening the mix strength along the
# ramp keeps the slate ordinal semantics (light = group 1, dark = group 4)
# while separating adjacent caps by ~40 L* points instead of ~4.
CAP_FILL_PCT = (14, 42, 70, 100)


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
    leaf_x = np.linspace(0.055, 0.945, 8)
    levels = (
        (np.asarray([0.50]), 0.82),
        (np.asarray([0.275, 0.725]), 0.66),
        (np.asarray([0.105, 0.335, 0.665, 0.895]), 0.49),
        (leaf_x, 0.31),
    )
    selected = 2
    # One mute line anchors the scoring, which the ink alone never states:
    # the cued context routes its +s_i stream to the readout, and s_i is the
    # trial's binary label sign.  Without it the leaf coefficients read as
    # abstract tree annotations and the classification target is only
    # inferable from panel E's chance line.
    f.text((0.5, 0.965),
           r"context picks the $+s_i$ stream;  $s_i$ = label sign",
           size=PT_SMALL, color=MUTE)
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
        r"$-0.45s_i$", r"$-0.45s_i$", r"$+s_i$", r"$-0.15s_i$",
        r"$-0.75s_i$", r"$-0.75s_i$", r"$-0.75s_i$", r"$-0.75s_i$",
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
    # Destination-indexed inverse of assigned=np.roll(arange(K), 1) in the
    # runner: destinations 0,1,2,3 receive source groups 1,2,3,0.
    "deranged": (1, 2, 3, 0),
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


def _route_bars(f: Frame, rect, mode: str, *, name_rows: bool = False,
                digits: bool = False) -> None:
    """Four rank-matched routes over the eight terminal dendritic branches.

    ``digits`` prints the delivered signal group over each bar and the task
    group under each leaf pair: the four slate tones alone are too close to
    carry the matched-versus-deranged distinction, so the index itself is
    printed and the panel's shared key names both digit rows.
    """
    x0, y0, width, height = rect
    xs = np.linspace(x0 + 0.09 * width, x0 + 0.91 * width, 8)
    stacked = mode == "rewired"
    leaf_y = y0 + (0.24 if digits else 0.13) * height
    supports = ROUTE_SUPPORTS[mode]
    signals = ROUTE_SIGNAL[mode]
    levels = _bar_levels(supports)
    span = (0.52 if stacked else 0.66) * height
    step = span / max(len(set(levels)), 1)
    bar_rise = (0.30 if digits else 0.22) * height

    # Each leaf is drawn as a terminal dendritic branch -- a short stub
    # continuing toward the soma, capped by the addressable terminal itself --
    # so the row reads as branches rather than as anonymous circles.  The cap
    # is tinted by the task group whose stream that branch carries, which is
    # what makes a group-mixing route visible.
    for idx, xpos in enumerate(xs):
        group = LEAF_GROUP[idx]
        tone = ROUTE_COLORS[group]
        f.ax.plot([xpos, xpos], [leaf_y - f.fy(5.0), leaf_y - f.fy(1.4)],
                  color=mix("dend", 45), lw=LW_HAIR, solid_capstyle="round",
                  zorder=3)
        f.disc((xpos, leaf_y), 2.6, fill=mix(tone, CAP_FILL_PCT[group]),
               edge=tone, lw=LW_EDGE, zorder=4)

    for route, (lo, hi) in enumerate(supports):
        color = ROUTE_COLORS[signals[route]]
        bar_y = leaf_y + bar_rise + levels[route] * step
        left, right = xs[lo], xs[hi]
        # A stacked (rewired) span drawn fat reads as covering every leaf it
        # crosses; membership is its two endpoints, so the span thins and the
        # endpoints carry filled dots over the supported leaves only.
        f.ax.plot([left, right], [bar_y, bar_y], color=color,
                  lw=LW_DATA if stacked else 2.6,
                  solid_capstyle="round", zorder=3)
        if stacked:
            for leaf in (lo, hi):
                f.disc((xs[leaf], bar_y), 1.9, fill=color, zorder=4)
            f.text((xs[lo] - f.fx(3.4), bar_y), str(signals[route] + 1),
                   size=PT_SMALL, color=INK, ha="right", va="center")
        # one leg per covered leaf: the route's support, drawn explicitly
        for leaf in (lo, hi):
            f.ax.plot([xs[leaf], xs[leaf]],
                      [bar_y - f.fy(1.0), leaf_y + f.fy(3.2)],
                      color=color, lw=LW_HAIR, solid_capstyle="round",
                      zorder=2)
        if digits:
            f.text(((left + right) / 2.0, bar_y + f.fy(2.6)),
                   str(signals[route] + 1), size=PT_SMALL, color=INK,
                   va="bottom")

    if digits:
        for group in range(4):
            members = [idx for idx, g in enumerate(LEAF_GROUP) if g == group]
            f.text((float(np.mean(xs[members])), leaf_y - f.fy(6.6)),
                   str(group + 1), size=PT_SMALL, color=MUTE, va="top")

    if name_rows:
        top = leaf_y + bar_rise + max(levels) * step
        f.text((x0 + 0.50 * width, top + f.fy(12.0 if digits else 7.5)),
               "K = 4 feedback routes",
               size=PT_SMALL, color=MUTE, va="bottom")
        if not digits:
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
        group = LEAF_GROUP[idx]
        tone = ROUTE_COLORS[group]
        f.ax.plot([xpos, xpos], [leaf_y - f.fy(5.0), leaf_y - f.fy(1.4)],
                  color=mix("dend", 45), lw=LW_HAIR, solid_capstyle="round",
                  zorder=3)
        f.disc((xpos, leaf_y), 2.6, fill=mix(tone, CAP_FILL_PCT[group]),
               edge=tone, lw=LW_EDGE, zorder=4)

    cell_w = (xs[1] - xs[0]) * 0.86
    cell_h = 0.115 * height
    base_y = leaf_y + 0.20 * height
    chip_x = xs[0] - cell_w / 2.0 - f.fx(3.0)
    for row in range(4):
        y = base_y + (3 - row) * cell_h
        # a chip in the route ramp names the row as one feedback channel:
        # clear of the matrix edge and nearly row-high so it reads as a row
        # marker (small gaps keep the four chips from fusing into one rail)
        f.ax.plot([chip_x, chip_x],
                  [y + 0.08 * cell_h, y + 0.92 * cell_h],
                  color=ROUTE_COLORS[row], lw=4.0, solid_capstyle="butt",
                  zorder=4)
        for col in range(8):
            value = matrix[row, col]
            face = (COLORS["oracle"] if value > 0 else
                    "#2E3947" if value < 0 else "white")
            f.ax.add_patch(Rectangle(
                (xs[col] - cell_w / 2.0, y), cell_w, cell_h,
                facecolor=face, edgecolor=mix("point_mlp", 22),
                linewidth=LW_HAIR,
                transform=f.ax.transData, zorder=3,
            ))

    # Compact signed-weight key.  Without it, the purple/gray/white cells can
    # be mistaken for four categorical route identities rather than entries
    # of a dense signed basis.
    key_y = y0 + 0.89 * height
    key_x = x0 + 0.44 * width
    key_specs = ((COLORS["oracle"], "+"),
                 ("#2E3947", "−"),
                 ("white", "0"))
    f.text((key_x - f.fx(4.0), key_y), "weight", size=PT_SMALL,
           color=MUTE, ha="right", va="center")
    for index, (face, symbol) in enumerate(key_specs):
        xpos = key_x + index * f.fx(17.0)
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


def route_controls(ax, indices, *, key: bool = False) -> None:
    """Visual dictionary for selected control questions at K=4.

    ``key`` reserves a strip under the cards for one shared legend naming the
    two digit rows the assignment cards print (delivered signal group on the
    bars, task group under the leaf pairs); the cards then draw digits.
    """
    f = Frame(ax, labels=True, scale=0.94)
    all_specs = (
        ("matched subtrees", "task groups = subtrees", GREEN),
        ("deranged assignment", "same routes, wrong signals", GRAY),
        ("example leaf permutation", "degree + depth preserved", GRAY),
        ("non-ancestry basis", "same feedback bandwidth", PURPLE),
    )
    chosen = tuple((index, all_specs[index]) for index in indices)
    cells = f.split(len(chosen), axis="x", gap_pt=8.0,
                    pad_pt=(0, 0, 0, 0))
    if key:
        lift = f.fy(10.0)
        cells = [(cx, cy + lift, cw, ch - lift)
                 for (cx, cy, cw, ch) in cells]
        f.text((0.5, f.fy(3.2)),
               "8 terminal branches · bar = delivered signal · "
               "cap = task group",
               size=PT_SMALL, color=MUTE, va="center")
    for rect, (_, (title, subtitle, accent)) in zip(cells, chosen, strict=True):
        _card(f, rect, title, subtitle, accent=accent)
    # Lower drawing rooms deliberately use identical geometry.
    for (index, _), rect in zip(chosen, cells, strict=True):
        x0, y0, width, height = rect
        if key:
            room = (x0, y0 + 0.06 * height, width, 0.60 * height)
        else:
            room = (x0, y0 + 0.07 * height, width, 0.67 * height)
        if index == 0:
            _route_bars(f, room, "matched", name_rows=True, digits=key)
        elif index == 1:
            _route_bars(f, room, "deranged", name_rows=key, digits=key)
        elif index == 2:
            _route_bars(f, room, "rewired", name_rows=True)
        else:
            _nonanatomical_basis(f, room)



def _terminal_stream_note(ax) -> None:
    """One mute line under B's cards: the terminals ARE the task's streams.

    Nothing inside the bandwidth ladder links the morphology's eight
    terminals to panel A's eight streams (the identity map pi lives only in
    the caption).  The four cards leave no interior band that clears both the
    K labels and the tallest capsules at this size, so the note hangs just
    below the frame the way panel C's key line sits under its cards; the
    37 pt row gutter keeps it well clear of panel D.  Spans are chained by
    measured advance exactly as ``_centred_subscript`` chains them --
    mathtext subscripts would shrink to 0.7x and fail the type-token audit.
    """
    f = Frame(ax, labels=True, scale=0.86)
    spans = (("terminals carry streams c", 0.0, 0.4),
             ("1", -1.6, 0.6),
             ("–c", 0.0, 0.4),
             ("8", -1.6, 0.0))
    width_pt = sum(_span_pt(text, PT_SMALL) + gap for text, _, gap in spans)
    x = 0.5 - f.fx(width_pt) / 2.0
    base_y = -0.055
    for text, dy_pt, gap in spans:
        f.text((x, base_y + f.fy(dy_pt)), text, size=PT_SMALL, color=MUTE,
               ha="left", va="baseline", clip_on=False)
        x += f.fx(_span_pt(text, PT_SMALL) + gap)


def _structural_tie_budgets(contrasts: pd.DataFrame, name: str) -> list:
    """Budgets at which a paired contrast ties in every one of the 20 seeds.

    ``ties == 20`` in the frozen table means all 20 paired seed differences
    are exactly zero -- a tie by construction (identity derangement at K=1,
    one-hot transport through pi at K=8), not an estimated null.
    """
    rows = contrasts[
        contrasts.architecture.eq("dendritic_tree")
        & contrasts.endpoint.eq("heldout_accuracy")
        & contrasts.contrast.eq(name)
    ]
    return sorted(int(b) for b in rows[rows.ties.eq(20)].budget_k)


def _topology_tie_budgets(outcomes: pd.DataFrame) -> list:
    """Budgets at which matched and rewired trees tie in every seed."""
    correct = outcomes[
        outcomes.feedback_family.eq("correct_ancestry_subtrees")
    ]
    budgets = []
    for budget in (1, 2, 4, 8):
        part = correct[correct.budget_k.eq(budget)]
        left = part[part.architecture.eq("dendritic_tree")].set_index(
            "seed").heldout_accuracy
        right = part[part.architecture.eq(
            "degree_depth_matched_rewired_tree")].set_index(
            "seed").heldout_accuracy
        diff = (left - right).dropna()
        if len(diff) == 20 and bool(diff.eq(0.0).all()):
            budgets.append(budget)
    return budgets


def _tie_note(ax, xy, *, ha, text="exact tie", va="center") -> None:
    """Mute microtext marking a zero that is exact by construction."""
    ax.text(xy[0], xy[1], text, fontsize=PT_SMALL, color=MUTE,
            ha=ha, va=va, zorder=6)


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


def k4_contrasts(ax_h, ax_i):
    directory = ROOT / "source_data/review_evidence_reanalysis"
    summary = pd.read_csv(directory / "ancestry_k4_control_contrasts.csv")
    pairs = pd.read_csv(directory / "ancestry_control_paired_differences.csv")
    pairs = pairs[pairs.budget_k.eq(4)]
    specifications = (
        (ax_h, [("best_matched_nonanatomical_oracle", "maximum of four controls")]),
        (ax_i, [("random_rank_k", "dense rank-4"),
                ("random_sparse_matched", "random sparse"),
                ("depth_interleaved_bins", "depth-interleaved"),
                ("within_neuron_route_derangement", "deranged")]),
    )
    for ax, controls in specifications:
        for index, (control, label) in enumerate(controls):
            row = summary[summary.control.eq(control)].iloc[0]
            values = pairs[pairs.control.eq(control)].accuracy_difference_pp.to_numpy(float)
            y = len(controls) - 1 - index
            ax.scatter(values, y + np.linspace(-0.16, 0.16, len(values)),
                       color=PURPLE, s=8, alpha=0.35, edgecolors="none", zorder=2)
            ax.errorbar(row.mean_difference_pp, y,
                        xerr=[[row.mean_difference_pp - row.ci95_low_pp],
                              [row.ci95_high_pp - row.mean_difference_pp]],
                        fmt="D", ms=MARKER_MS, color=PURPLE, mfc="white",
                        lw=LW_DATA, capsize=ERR_CAPSIZE, zorder=3)
            if ax is ax_i:
                p = row.p_holm_four_individual_controls_at_k4
                ax.text(-0.035, y, label,
                        transform=ax.get_yaxis_transform(), ha="right",
                        va="center", fontsize=PT_SMALL, color=MUTE)
                ax.text(0.99, y + 0.23, f"Holm P={p:.2g}",
                        transform=ax.get_yaxis_transform(), ha="right",
                        va="bottom", fontsize=PT_SMALL, color=MUTE)
        ax.axvline(0, color=MUTE, ls="--", lw=LW_REF, zorder=0)
        ax.set_yticks([])
        ax.tick_params(axis="y", labelsize=PT_SMALL, length=0)
        ax.set_ylim(-0.45, len(controls) - 0.25)
        ax.set_xlabel("ancestry − control accuracy (pp)")
    ax_h.set_yticks([])
    ax_h.set_xlim(-5, 7)
    primary = summary[summary.control.eq("best_matched_nonanatomical_oracle")].iloc[0]
    ax_h.text(0.03, 0.93,
              f"{primary.mean_difference_pp:+.2f} pp "
              f"[{primary.ci95_low_pp:.2f}, {primary.ci95_high_pp:.2f}]\n"
              f"{int(primary.positive_seeds)}/20 seeds positive; Holm P="
              f"{primary.p_holm_four_budgets:.4f}",
              transform=ax_h.transAxes, va="top", ha="left",
              fontsize=PT_SMALL, color=PURPLE,
              bbox=dict(facecolor="white", edgecolor="none", pad=0.2))
    ax_i.set_xlim(-12, 78)


def build() -> list:
    mpl.rcParams["lines.markeredgewidth"] = LW_EDGE
    outcomes = pd.read_csv(SUBTREE / "seed_outcomes.csv")
    summary = pd.read_csv(SUBTREE / "condition_summary.csv")
    contrasts = pd.read_csv(SUBTREE / "paired_contrasts.csv")
    dendritic = summary[summary.architecture.eq("dendritic_tree")]

    canvas = NativeCanvas(
        HEIGHT_IN, 4, row_weights=ROW_PT,
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
                        title="Bandwidth and learning")
    ax_f = canvas.panel("F", 2, 4, 4,
                        title="Route assignment")
    ax_g = canvas.panel("G", 2, 8, 4,
                        title="Matched versus rewired")
    ax_h = canvas.panel("H", 3, 0, 6,
                        title="K = 4: ancestry versus best control")
    # The forest has its own explicit label column, not a column-wide reserve
    # that would also narrow the schematic above it.
    ax_i = canvas.panel("I", 3, 6, 6, inset_pt=(48.0, 0.0, 0.0, 0.0),
                        lock=False, title="K = 4: individual controls")

    hierarchical_task(ax_a)
    address_ladder_compact(ax_b)
    _terminal_stream_note(ax_b)
    route_controls(ax_c, (0, 1), key=True)
    route_controls(ax_d, (2, 3))
    bandwidth_sweep_compact(ax_e, outcomes, dendritic)
    # The exactly balanced binary label makes 0.5 the chance level; without
    # the reference a reader cannot see that deranged routing is actively
    # harmful rather than merely unhelpful.
    ax_e.axhline(0.5, color=MUTE, lw=LW_REF, dashes=(2.4, 2.0), zorder=0)
    ax_e.text(3.10, 0.505, "chance", color=MUTE, fontsize=PT_SMALL,
              ha="right", va="bottom")
    ax_e.set_ylabel("held-out accuracy")
    # One comparator, one name: E's series and F's contrast against it both
    # say "best matched control" (the caption defines it once).
    _rename_label(ax_e, "best control", "best matched\ncontrol")
    _move_label(ax_e, "best matched\ncontrol", (0.12, 0.69))
    route_contrasts_compact(ax_f, contrasts)
    # "vs best" sat on the K = 1 diamond and its lower whisker; move it into
    # the empty well under the rising violet segment.
    _move_label(ax_f, "vs best", (0.52, -46.0))
    _rename_label(ax_f, "vs best", "vs best matched control")
    _annotate_k4_advantage(ax_f, contrasts)
    topology_alignment_compact(ax_g, outcomes)
    k4_contrasts(ax_h, ax_i)

    # Structural exact ties.  At these budgets every one of the 20 paired
    # seed differences is exactly zero by construction, but the points were
    # drawn as ordinary data with zero-length whiskers, so a reader could not
    # tell a tie-by-identity from an estimated no-effect.  The marked set is
    # derived from the frozen tables here, never hard-coded.
    derange_ties = _structural_tie_budgets(
        contrasts, "correct - within_neuron_route_derangement")
    oracle_ties = _structural_tie_budgets(
        contrasts, "correct - best_matched_nonanatomical_oracle")
    topology_ties = _topology_tie_budgets(outcomes)
    if (derange_ties, oracle_ties, topology_ties) != ([1], [8], [1, 8]):
        raise ValueError(
            "structural-tie set changed: deranged %r, oracle %r, topology %r"
            % (derange_ties, oracle_ties, topology_ties))
    # E: the concentric K=1 pair (correct == deranged in all 20 seeds).
    _tie_note(ax_e, (0.14, 0.142), ha="left")
    # F: the deranged contrast at K=1 and the oracle contrast at K=8.
    _tie_note(ax_f, (0.10, -9.0), ha="left")
    _tie_note(ax_f, (2.92, -9.0), ha="right")
    # G: both zero endpoints; per-point notes would sit on the rising and
    # falling segments, so one note names both from the empty upper right.
    _tie_note(ax_g, (3.10, 27.5), ha="right", text="K = 1, 8: exact ties")

    from build_main_figure_02 import _equalise_row
    canvas.lock_reserves()
    _equalise_row(canvas, {"E": 0, "F": 4, "G": 8})
    canvas.lock_reserves()
    # I's private label column must not exempt its plot from H's row height.
    for record in canvas._records:
        if record["name"] == "I":
            record["inset_pt"] = (48.0, 0.0, *canvas._locks["H"][2:])
    COMPONENT.parent.mkdir(parents=True, exist_ok=True)
    from journal_style import style_direct_color_labels
    style_direct_color_labels(canvas.fig)
    problems = canvas.save(COMPONENT, name="main_figure_05_native")
    return problems


def main() -> None:
    problems = build()
    for problem in problems:
        print(f"  {problem}")
    print(f"  canvas width {FIG_W * 72:.1f} pt, height {CANVAS_H_PT:.1f} pt")


if __name__ == "__main__":
    main()

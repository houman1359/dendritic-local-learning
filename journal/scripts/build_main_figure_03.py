#!/usr/bin/env python3
"""Main Figure 3 -- trained subtree-address factorial -- as ONE native canvas.

The figure is authored at exactly the canonical full width
(``journal_style.FIG_W`` = 7.2 in = 518.4 pt) on a single 12-column module
grid and emitted at scale 1.0, so every type size and stroke weight in the
compiled PDF is the token the builder asked for.  Nothing here is a
pre-rendered sub-block scaled into a slot.

Structure: one regular 3 x 2 grid of four-module panels.

* the K = 1, 2, 4, 8 address ladder is drawn natively from the shared credit
  tree vocabulary and sits immediately LEFT of the bandwidth sweep it
  explains (S2);
* the bandwidth sweep, the ladder and the contrast forest are the same size:
  a panel here is bigger than another only by spanning more modules, and
  none of them does, so every panel of a row shares one axes-box height and
  every panel of a column one x0 and one width;
* the five paired ``correct - control`` contrasts, previously one composite
  curve plus four numbers in the prose, are consolidated into one forest
  panel on a shared effect-size axis (S4);
* correct ancestry keeps the anatomy green and the learned upper bound keeps
  the oracle violet; the four matched non-anatomical controls are drawn as
  one neutral-gray family at two lightnesses, separated by marker (S5);
* the representation-match and capture panels share one held-out-accuracy
  axis, the label and tick column appearing once (S1).

The methodological notes the panels used to print inside their own axes --
that overlapping points are fanned (B), that the seed cloud is spread in x
(D), that the four architectures coincide and are drawn concentrically (E),
and the cross-references in F -- are disclosure rather than graphic content
and are carried by the caption instead.  Only the marker-size key, which is
the sole encoding of K in F, stays on the plot.

Every plotted value, n, interval and test is read from the same source-data
CSVs the previous builder used and is passed through unchanged.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import transforms as mtransforms
from matplotlib.lines import Line2D

from credit_tree_schematics import (
    _BASE_XLIM,
    _Tree,
    _setup_axes,
    draw_credit_tree,
    mix,
)
from figure_canvas import (
    COLORS,
    enforce_tokens,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_ANNOT,
    PT_SMALL,
    SEED_ALPHA,
    SEED_MS,
    Margins,
    NativeCanvas,
)
from native_schematics import Frame

# Every type size and line weight below is a journal_style token, re-exported
# through figure_canvas together with the canvas that enforces them.


ROOT = Path(__file__).resolve().parents[1]
SUBTREE = ROOT / "source_data" / "trained_subtree_address_full_factorial"
COMPONENT = ROOT / "figures" / "components" / "main_figure_03_native.pdf"

# ── canvas geometry ──────────────────────────────────────────────────────
CANVAS_H_PT = 396.0                     # aspect 518.4 / 396.0 = 1.31
HEIGHT_IN = CANVAS_H_PT / 72.0
HGUTTER_PT = 38.0
VGUTTER_PT = 38.0
# Two rows of three four-module panels.  The top row is taller because the
# forest in C stacks twenty intervals under five block headers; the ratio is
# held under the 1.35x module-normalised emphasis band so no panel of the
# page dominates the others.
ROW_PT = [176.0, 140.0]

# ── one accuracy axis for panels B, E and F ──────────────────────────────
ACC_LIM = (0.10, 0.88)
ACC_TICKS = (0.2, 0.4, 0.6, 0.8)
ACC_LABEL = "held-out accuracy"
K_TICKS = (1, 2, 4, 8)
K_LABEL = "feedback channels $K$"

GREEN = COLORS["shunting"]
PURPLE = COLORS["oracle"]
MUTE = COLORS["mute"]
INK = COLORS["ink"]
GRAY_D = COLORS["point_mlp"]             # the control gray, #686868
GRAY_L = mix("point_mlp", 68)            # its lighter second lightness

# One palette slot + one marker per feedback family, shared by panels B and F
# (and cross-referenced by the forest markers in panel C).  Anatomy keeps the
# green, the unrestricted learned bound keeps the oracle violet, and the four
# matched non-anatomical controls read as one neutral family.
SOLID = (None, None)
DASH = (3.0, 1.8)

# One short name per route family, used by the key in B and by the block
# headers in C so a reader meets each control under one label.  The full
# definition of every family is a caption sentence.
ROUTE_STYLE = {
    "correct_ancestry_subtrees": (GREEN, "o", "correct", SOLID),
    "within_neuron_route_derangement": (GRAY_D, "s", "deranged", SOLID),
    "depth_interleaved_bins": (GRAY_L, "^", "depth bins", SOLID),
    "random_sparse_matched": (GRAY_D, "D", "sparse", DASH),
    "random_rank_k": (GRAY_L, "v", "random rank", DASH),
    "learned_rank_k_upper_bound": (PURPLE, "P", "learned rank", SOLID),
}

# Forest rows: one block per paired contrast, four budgets per block.  The
# marker repeats the control's marker in panels B and F.
# Colour repeats the control family's own colour from panel B, so a row label
# carries the same hue in both panels; the block whose control is a composite
# of all four keeps the anatomy green.
CONTRAST_BLOCKS = (
    ("correct - best_matched_nonanatomical_oracle", "best of four",
     "X", GREEN),
    ("correct - within_neuron_route_derangement", "deranged", "s", GRAY_D),
    ("correct - depth_interleaved_bins", "depth bins", "^", GRAY_L),
    ("correct - random_sparse_matched", "sparse", "D", GRAY_D),
    ("correct - random_rank_k", "random rank", "v", GRAY_L),
)

# Panel E: the same routed field carried by four implementations.  The
# dendritic tree keeps the anatomy green; the three point/flat emulations are
# the neutral control family at two lightnesses, drawn concentrically.
# Sizes fall fast enough that every ring of the concentric stack stays
# visible, and the lightness alternates so no ring is lost against the one
# drawn on top of it.
ARCH_SPECS = (
    ("dendritic_tree", "dendritic", GREEN, "o", 9.6),
    ("point_neuron_explicit_gating", "gated point", GRAY_L, "s", 7.2),
    ("flat_compartment", "flat", GRAY_D, "^", 4.9),
    ("grouped_point_subunits", "grouped point", GRAY_L, "D", 2.7),
)


def bootstrap(values: np.ndarray, seed: int, draws: int = 20_000):
    """Seed-bootstrap mean and 95% interval (verbatim from the old builder)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


# ── A: the address ladder ────────────────────────────────────────────────
# One frame for all four trees, so the four somas sit on one baseline and the
# canopies are drawn at one scale: the ladder must read as the same tree under
# four route budgets, not as four differently sized trees.
ADDRESS_YLIM = (-0.55, 3.62)
ADDRESS_ASPECT = ((_BASE_XLIM[1] - _BASE_XLIM[0])
                  / (ADDRESS_YLIM[1] - ADDRESS_YLIM[0]))
TREE_SCALE = 0.62

# K = 1 is the same tree under a single shared route field covering every
# branch, drawn with the library's own capsule primitive (gray = the shared /
# control slot).  The library's address mode starts at K = 2.
K1_CHAINS = (
    ((0.02, 0.34), "J1", "JL", "JLL", "T1"), ("JLL", "T2"),
    ("JL", "JLR", "T3"), ("JLR", "T4"),
    ("J1", "JR", "JRL", "T5"), ("JRL", "T6"),
    ("JR", "JRR", "T7"), ("JRR", "T8"),
)

LADDER = (
    (1, "$K = 1$", "one shared\nfield"),
    (2, "$K = 2$", "two coarse\nsubtrees"),
    (4, "$K = 4$", "four nested\nroutes"),
    (8, "$K = 8$", "eight leaf\nroutes"),
)


def _tree_inset(f: Frame, rect):
    """Axes filling ``rect`` at the shared address frame's exact aspect."""
    x0, y0, w, h = rect
    w_pt, h_pt = w * f.w_pt, h * f.h_pt
    if w_pt / h_pt > ADDRESS_ASPECT:
        fit_h_pt, fit_w_pt = h_pt, h_pt * ADDRESS_ASPECT
    else:
        fit_w_pt, fit_h_pt = w_pt, w_pt / ADDRESS_ASPECT
    fit = (x0 + (w - f.fx(fit_w_pt)) / 2.0,
           y0 + (h - f.fy(fit_h_pt)) / 2.0,
           f.fx(fit_w_pt), f.fy(fit_h_pt))
    sub = f.ax.inset_axes(fit, transform=f.ax.transData, zorder=3)
    sub.set_facecolor("none")
    return sub


def _address_tree(ax, budget_k: int) -> None:
    """One address-mode credit tree; K = 1 adds the whole-tree capsule."""
    if budget_k == 1:
        _setup_axes(ax, _BASE_XLIM, ADDRESS_YLIM)
        t = _Tree(ax, TREE_SCALE, False)
        t.capsule(mix("point_mlp", 15), 13, list(K1_CHAINS))
        t.tree(COLORS["dend"])
        t.junctions()
        t.soma(COLORS["soma"], mix("ink", 30))
    else:
        draw_credit_tree(ax, mode="address", K=budget_k, scale=TREE_SCALE,
                         labels=False, xlim=_BASE_XLIM, ylim=ADDRESS_YLIM)
    enforce_tokens(ax)


def address_ladder(ax) -> None:
    """K = 1, 2, 4, 8 nested route fields on the shared credit tree."""
    f = Frame(ax, labels=True, scale=0.92)
    cells = f.split(4, axis="y", gap_pt=5.0)
    subs = []
    for (x0, y0, w, h), (budget, tag, gloss) in zip(cells, LADDER, strict=True):
        emphasis = budget == 4
        f.group((x0, y0, w, h),
                tint=mix("shunting", 9) if emphasis else COLORS["panel_bg"],
                edge=mix("shunting", 45) if emphasis else COLORS["grid"])
        pad_x, pad_y = f.fx(2.0), f.fy(2.0)
        sub = _tree_inset(f, (x0 + pad_x, y0 + pad_y, w * 0.50 - pad_x,
                              h - 2 * pad_y))
        _address_tree(sub, budget)
        subs.append(sub)
        text_x = x0 + w * 0.53
        cy = y0 + h / 2.0
        f.text((text_x, cy + f.fy(8.5)), tag, size=PT_ANNOT,
               color=GREEN if emphasis else INK, ha="left")
        f.text((text_x, cy - f.fy(6.0)), gloss, size=PT_SMALL, color=MUTE,
               ha="left", linespacing=1.2)
    _share_tree_frame(subs)


def _share_tree_frame(subs) -> None:
    """One frame for every rung of the ladder.

    ``draw_credit_tree`` autoscales each tree to its own ink, and the K = 8
    route capsules reach further out than the K = 2 ones, so the rungs would
    otherwise be drawn at four slightly different sizes -- exactly the
    inconsistency the ladder exists to rule out.  Take the union frame,
    centred on the soma, and give every rung the same one.
    """
    cy = 0.5 * (ADDRESS_YLIM[0] + ADDRESS_YLIM[1])
    half_w = max(max(abs(v) for v in sub.get_xlim()) for sub in subs)
    half_h = max(max(abs(v - cy) for v in sub.get_ylim()) for sub in subs)
    half_w = max(half_w, half_h * ADDRESS_ASPECT)
    half_h = half_w / ADDRESS_ASPECT
    for sub in subs:
        sub.set_aspect("auto")
        sub.set_xlim(-half_w, half_w)
        sub.set_ylim(cy - half_h, cy + half_h)


# ── B: the bandwidth sweep (headline) ────────────────────────────────────
def bandwidth_sweep(fig, ax, dendritic: pd.DataFrame) -> None:
    ax.set_xlim(0.45, 8.55)
    ax.set_ylim(*ACC_LIM)
    rows_by_family = {
        family: dendritic[dendritic.feedback_family.eq(family)].sort_values("budget_k")
        for family in ROUTE_STYLE
    }
    # Families coincide exactly at K = 1 (four at 0.187) and K = 8 (five at
    # 0.810).  Draw the lines and error bars at the true values, then fan the
    # exact pile-ups on a small ring and dodge near-overlaps sideways in point
    # space, with one mute note naming the convention.
    members: dict[tuple[float, float], list[str]] = {}
    for family, part in rows_by_family.items():
        for _, row in part.iterrows():
            key = (float(row.budget_k), round(float(row.mean_heldout_accuracy), 3))
            members.setdefault(key, []).append(family)
    shift: dict[tuple[str, float], tuple[float, float]] = {}
    for (budget, _), families in members.items():
        if len(families) < 2:
            continue
        radius = 2.4 + 0.28 * len(families)
        for index, family in enumerate(families):
            angle = np.pi / 2 + 2 * np.pi * index / len(families)
            shift[(family, budget)] = (radius * np.cos(angle),
                                       radius * np.sin(angle))
    axes_h_pt = ax.get_position().height * fig.get_figheight() * 72.0
    near = 3.6 * (ACC_LIM[1] - ACC_LIM[0]) / axes_h_pt
    for budget in K_TICKS:
        stack = sorted(
            (float(part[part.budget_k.eq(budget)].mean_heldout_accuracy.iloc[0]),
             family)
            for family, part in rows_by_family.items()
        )
        for (y0, fam0), (y1, fam1) in zip(stack, stack[1:]):
            if 0 < y1 - y0 < near and (fam0, float(budget)) not in shift \
                    and (fam1, float(budget)) not in shift:
                shift[(fam0, float(budget))] = (-2.4, 0.0)
                shift[(fam1, float(budget))] = (2.4, 0.0)
    handles = []
    for family, (color, marker, label, dashes) in ROUTE_STYLE.items():
        part = rows_by_family[family]
        mean = part.mean_heldout_accuracy.to_numpy(float)
        low = part.ci95_low_heldout_accuracy.to_numpy(float)
        high = part.ci95_high_heldout_accuracy.to_numpy(float)
        budgets = part.budget_k.to_numpy(float)
        line, = ax.plot(budgets, mean, color=color, lw=LW_DATA, zorder=2)
        if dashes[0] is not None:
            line.set_dashes(dashes)
        for x, y, lo, hi in zip(budgets, mean, low, high):
            dx, dy = shift.get((family, x), (0.0, 0.0))
            offset = mtransforms.offset_copy(ax.transData, fig=fig, x=dx, y=dy,
                                             units="points")
            ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], color=color,
                        marker=marker, ms=3.8, lw=0, elinewidth=LW_ERR,
                        capsize=ERR_CAPSIZE, markeredgecolor="white",
                        markeredgewidth=LW_EDGE, transform=offset, zorder=3)
        handle = Line2D([0], [0], color=color, marker=marker,
                        lw=LW_DATA, ms=3.8, markeredgecolor="white",
                        markeredgewidth=LW_EDGE, label=label)
        if dashes[0] is not None:
            handle.set_dashes(dashes)
        handles.append(handle)
    ax.set_xticks(K_TICKS)
    ax.set_yticks(ACC_TICKS)
    ax.set_xlabel(K_LABEL)
    ax.set_ylabel(ACC_LABEL)
    # One column: the panel is four modules wide like every other panel of
    # the page, and the six route families are the figure's colour and marker
    # key, so they are set as one readable stack in the corner the curves
    # leave empty rather than as two cramped columns.
    legend = ax.legend(handles=handles, loc="center right",
                       bbox_to_anchor=(1.005, 0.455), ncol=1, frameon=False,
                       fontsize=PT_SMALL, handlelength=0.95,
                       handletextpad=0.30, labelspacing=0.34,
                       borderaxespad=0.0)
    legend.set_zorder(6)
    for text in legend.get_texts():          # never pure #000000
        text.set_color(INK)


# ── C: the contrast forest ───────────────────────────────────────────────
def contrast_forest(fig, ax, contrasts: pd.DataFrame) -> None:
    """One shared effect-size axis for all five paired route contrasts.

    Rows are grouped by contrast and labelled horizontally; each block header
    owns 1.4 row units so the label never lands on the interval below it.
    """
    frame = contrasts[contrasts.architecture.eq("dendritic_tree")
                      & contrasts.endpoint.eq("heldout_accuracy")]
    header_units, block_gap = 1.4, 1.4
    cursor = 0.0
    yticks, ylabels = [], []
    tie_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=4.4, y=0.0,
                                         units="points")
    for name, header, marker, color in CONTRAST_BLOCKS:
        ax.text(0.015, -cursor, header, transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=PT_SMALL, color=INK,
                zorder=6,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        cursor += header_units
        block = frame[frame.contrast.eq(name)].sort_values("budget_k")
        for offset, (_, row) in enumerate(block.iterrows()):
            y = -(cursor + offset)
            mean = 100.0 * float(row.mean_difference)
            low = 100.0 * float(row.ci95_low)
            high = 100.0 * float(row.ci95_high)
            ax.errorbar([mean], [y], xerr=[[mean - low], [high - mean]],
                        color=color, marker=marker, ms=MARKER_MS, lw=0,
                        elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                        markeredgecolor="white", markeredgewidth=LW_EDGE,
                        zorder=3)
            if int(row.ties) == int(row.n_pairs):
                ax.text(mean, y, "tie", transform=tie_offset, ha="left",
                        va="center", fontsize=PT_SMALL, color=MUTE)
            yticks.append(y)
            ylabels.append(f"{int(row.budget_k)}")
        cursor += len(block) - 1 + block_gap
    low_y, high_y = -(cursor - block_gap + 0.85), 0.85
    ax.set_ylim(low_y, high_y)
    # Drawn as an explicit segment rather than ``axvline`` so its vertices are
    # real data coordinates: an axvline reports a vertex at (0, 0), which the
    # inherited text-over-data audit reads as a datum sitting under the first
    # block header.
    zero = Line2D([0.0, 0.0], [low_y, high_y], color=MUTE, lw=LW_REF,
                  zorder=1, solid_capstyle="butt")
    zero.set_dashes((3.2, 2.2))
    ax.add_line(zero)
    ax.set_yticks(yticks, ylabels)
    ax.set_xlim(-50.0, 70.0)
    ax.set_xticks([-40, 0, 40])
    ax.set_xlabel("correct − control (pp)")
    ax.set_ylabel(K_LABEL)


# ── D: task-topology alignment ───────────────────────────────────────────
def topology_alignment(fig, ax, outcomes: pd.DataFrame) -> None:
    """Paired matched-versus-rewired differences, one row of seeds per K.

    Exact ties (every seed identical) are named rather than left as a bare
    marker on the zero line, the same convention the forest in C uses.
    """
    # Above the marker, not beside it: the zero reference is horizontal here,
    # so a label on the same row would read as struck through.
    tie_offset = mtransforms.offset_copy(ax.transData, fig=fig, x=0.0, y=6.5,
                                         units="points")
    correct = outcomes[outcomes.feedback_family.eq("correct_ancestry_subtrees")]
    for budget_index, budget in enumerate(K_TICKS):
        left = correct[correct.architecture.eq("dendritic_tree")
                       & correct.budget_k.eq(budget)].set_index("seed").heldout_accuracy
        right = correct[correct.architecture.eq("degree_depth_matched_rewired_tree")
                        & correct.budget_k.eq(budget)].set_index("seed").heldout_accuracy
        values = (left - right).to_numpy(float)
        mean, low, high = bootstrap(values, 70_000 + budget_index)
        ax.scatter(np.full(len(values), budget) + np.linspace(-0.13, 0.13, len(values)),
                   100 * values, s=SEED_MS ** 2, color=GREEN, alpha=SEED_ALPHA,
                   edgecolors="none")
        ax.errorbar(budget, 100 * mean,
                    yerr=[[100 * (mean - low)], [100 * (high - mean)]],
                    color=GREEN, marker="D", markerfacecolor="white",
                    ms=MARKER_MS + 1.2, lw=LW_ERR, capsize=ERR_CAPSIZE,
                    zorder=4)
        if np.allclose(values, 0.0):
            ax.text(budget, 0.0, "tie", transform=tie_offset, ha="center",
                    va="bottom", fontsize=PT_SMALL, color=MUTE)
    ax.axhline(0, color=MUTE, ls="--", lw=LW_REF)
    ax.set_xlim(0.45, 8.55)
    ax.set_ylim(-3.4, 35.8)
    ax.set_xticks(K_TICKS)
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel(K_LABEL)
    ax.set_ylabel("matched − rewired tree (pp)")


# ── E: representation match ──────────────────────────────────────────────
def representation_match(ax, summary: pd.DataFrame) -> None:
    base = summary[summary.architecture.eq("dendritic_tree")
                   & summary.feedback_family.eq("correct_ancestry_subtrees")
                   ].sort_values("budget_k")
    ax.plot(base.budget_k, base.mean_heldout_accuracy, color=mix("point_mlp", 38),
            lw=LW_DATA, zorder=1)
    handles = []
    for depth, (architecture, label, color, marker, size) in enumerate(ARCH_SPECS):
        part = summary[summary.architecture.eq(architecture)
                       & summary.feedback_family.eq("correct_ancestry_subtrees")
                       ].sort_values("budget_k")
        ax.plot(part.budget_k, part.mean_heldout_accuracy, marker=marker,
                color=color, ms=size, lw=0, markeredgecolor="white",
                markeredgewidth=LW_EDGE, zorder=3 + depth)
        handles.append(Line2D([0], [0], marker=marker, color=color, lw=0,
                              ms=MARKER_MS, markeredgecolor="white",
                              markeredgewidth=LW_EDGE, label=label))
    ax.set_xlim(0.45, 8.55)
    ax.set_ylim(*ACC_LIM)
    ax.set_xticks(K_TICKS)
    ax.set_yticks(ACC_TICKS)
    ax.set_xlabel(K_LABEL)
    ax.set_ylabel(ACC_LABEL)
    legend = ax.legend(handles=handles, loc="lower right",
                       bbox_to_anchor=(1.0, 0.02), frameon=False,
                       fontsize=PT_SMALL, handlelength=1.0,
                       handletextpad=0.35, labelspacing=0.3,
                       borderaxespad=0.0)
    legend.set_zorder(6)
    for text in legend.get_texts():          # never pure #000000
        text.set_color(INK)


# ── F: capture and learning ──────────────────────────────────────────────
def capture_and_learning(fig, ax, dendritic: pd.DataFrame) -> None:
    selected = dendritic[dendritic.feedback_family.isin(ROUTE_STYLE)].copy()
    ax.set_xlim(-0.05, 1.10)
    ax.set_ylim(*ACC_LIM)

    def key(x: float, y: float) -> tuple[float, float]:
        return (round(float(x), 4), round(float(y), 4))

    members: dict[tuple[float, float], list[str]] = {}
    for family in ROUTE_STYLE:
        part = selected[selected.feedback_family.eq(family)]
        for _, row in part.iterrows():
            members.setdefault(key(row.mean_initial_gradient_capture,
                                   row.mean_heldout_accuracy), []).append(family)
    fan: dict[tuple[str, tuple[float, float]], tuple[float, float]] = {}
    for spot, families in members.items():
        if len(families) < 2:
            continue
        radius = 3.4 + 0.35 * len(families)
        for index, family in enumerate(families):
            angle = np.pi / 2 + 2 * np.pi * index / len(families)
            fan[(family, spot)] = (radius * np.cos(angle), radius * np.sin(angle))
    box = ax.get_position()
    x_pt = box.width * fig.get_figwidth() * 72.0 / (1.10 + 0.05)
    y_pt = box.height * fig.get_figheight() * 72.0 / (ACC_LIM[1] - ACC_LIM[0])
    loose = [
        (family, key(row.mean_initial_gradient_capture, row.mean_heldout_accuracy),
         float(np.sqrt(13.0 + 3.2 * float(row.budget_k))))
        for family in ROUTE_STYLE
        for _, row in selected[selected.feedback_family.eq(family)].iterrows()
        if (family, key(row.mean_initial_gradient_capture,
                        row.mean_heldout_accuracy)) not in fan
    ]
    for i, (fam0, key0, ms0) in enumerate(loose):
        for fam1, key1, ms1 in loose[i + 1:]:
            if (fam0, key0) in fan or (fam1, key1) in fan:
                continue
            du = (key1[0] - key0[0]) * x_pt
            dv = (key1[1] - key0[1]) * y_pt
            gap = float(np.hypot(du, dv))
            need = (ms0 + ms1) / 2.0 + 1.1
            if gap >= need:
                continue
            ux, uy = (du / gap, dv / gap) if gap > 0 else (0.0, 1.0)
            push = (need - gap) / 2.0
            fan[(fam0, key0)] = (-ux * push, -uy * push)
            fan[(fam1, key1)] = (ux * push, uy * push)
    for family, (color, marker, _, dashes) in ROUTE_STYLE.items():
        part = selected[selected.feedback_family.eq(family)].sort_values("budget_k")
        trace, = ax.plot(part.mean_initial_gradient_capture,
                         part.mean_heldout_accuracy, color=color, lw=LW_HAIR,
                         alpha=0.35, zorder=1)
        if dashes[0] is not None:
            trace.set_dashes(dashes)
        for _, row in part.sort_values("budget_k", ascending=False).iterrows():
            x = float(row.mean_initial_gradient_capture)
            y = float(row.mean_heldout_accuracy)
            dx, dy = fan.get((family, key(x, y)), (0.0, 0.0))
            offset = mtransforms.offset_copy(ax.transData, fig=fig, x=dx, y=dy,
                                             units="points")
            ax.plot([x], [y], marker=marker, color=color,
                    ms=float(np.sqrt(13.0 + 3.2 * float(row.budget_k))), lw=0,
                    alpha=0.9, markeredgecolor="white", markeredgewidth=LW_EDGE,
                    transform=offset, zorder=3.0 - 0.05 * float(row.budget_k))
    # The only encoding of K in this panel, so it stays on the plot; the
    # cross-references it used to carry ("route families as in B; y axis as
    # in E") are caption sentences.
    ax.text(0.99, 0.015, "marker size $\\propto K$", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=PT_ANNOT, color=MUTE)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("initial gradient capture")


def build() -> list:
    mpl.rcParams["lines.markeredgewidth"] = LW_EDGE
    outcomes = pd.read_csv(SUBTREE / "seed_outcomes.csv")
    summary = pd.read_csv(SUBTREE / "condition_summary.csv")
    contrasts = pd.read_csv(SUBTREE / "paired_contrasts.csv")
    dendritic = summary[summary.architecture.eq("dendritic_tree")]

    canvas = NativeCanvas(HEIGHT_IN, 2, row_weights=ROW_PT,
                          hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
                          margins=Margins(left=34.0, right=12.0, top=16.0,
                                          bottom=26.0))
    fig = canvas.fig
    ax_a = canvas.panel("A", 0, 0, 4, schematic=True, title="Address bandwidth")
    ax_b = canvas.panel("B", 0, 4, 4, title="Learning across bandwidth")
    ax_c = canvas.panel("C", 0, 8, 4, grid="x", title="Route contrasts")
    ax_d = canvas.panel("D", 1, 0, 4, title="Task–topology alignment")
    ax_e = canvas.panel("E", 1, 4, 4, title="Representation match")
    ax_f = canvas.panel("F", 1, 8, 4, title="Capture and learning", sharey=ax_e)
    ax_f.tick_params(axis="y", labelleft=False)

    address_ladder(ax_a)
    bandwidth_sweep(fig, ax_b, dendritic)
    contrast_forest(fig, ax_c, contrasts)
    topology_alignment(fig, ax_d, outcomes)
    representation_match(ax_e, summary)
    capture_and_learning(fig, ax_f, dendritic)

    COMPONENT.parent.mkdir(parents=True, exist_ok=True)
    return canvas.save(COMPONENT, name="main_figure_03_native")


def main() -> None:
    problems = build()
    for problem in problems:
        print(f"  {problem}")
    print(f"  canvas width {FIG_W * 72:.1f} pt, height {CANVAS_H_PT:.1f} pt")


if __name__ == "__main__":
    main()

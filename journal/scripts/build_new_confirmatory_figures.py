#!/usr/bin/env python3
"""Render journal figures for the August 2026 confirmatory extensions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import transforms as mtransforms
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from journal_style import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    SEED_ALPHA,
    SEED_MS,
    add_headroom,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
    wrap_ticklabels,
)


ROOT = Path(__file__).resolve().parents[1]
SUBTREE = ROOT / "source_data" / "trained_subtree_address_full_factorial"
ACTIVE = ROOT / "source_data" / "focal_selectivity_active_ensemble"
FULL_TREE = ROOT / "source_data" / "fulltree_boundary" / "output"
FIGURES = ROOT / "figures" / "generated"

# One palette slot + one marker per feedback family, shared by panels B and F
# and by the figure-wide key.  Hues follow the manuscript-wide routing
# taxonomy (figs 3/5/8 and the full-tree extension): green = anatomy-correct
# routing, rose = broken permutation control (the 'shuffle' slot — a route
# derangement is a within-neuron shuffle), gray = random control, violet =
# learned oracle.  The bp red-brown is reserved manuscript-wide for the
# exact/backprop ceiling and never marks an adversarial control.  Sibling
# hues are CVD-checked; the rose/amber pair (tritan-close, like fig5's
# shuffle/scalar pair) is disambiguated by its distinct markers (s vs ^).
ROUTE_STYLE = {
    "correct_ancestry_subtrees": (COLORS["shunting"], "o", "correct ancestry"),
    "within_neuron_route_derangement": (COLORS["highlight"], "s", "route derangement"),
    "depth_interleaved_bins": (COLORS["local"], "^", "depth-interleaved"),
    "random_sparse_matched": (COLORS["point_mlp"], "D", "random sparse"),
    "random_rank_k": (COLORS["additive"], "v", "random rank-$K$"),
    "learned_rank_k_upper_bound": (COLORS["oracle"], "P", "learned rank-$K$ oracle"),
}


def bootstrap(values: np.ndarray, seed: int, draws: int = 20_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, name)
    audit_text_over_data(fig, name)
    fig.savefig(
        FIGURES / f"{name}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / f"{name}.png", dpi=600)
    plt.close(fig)


def hierarchy_schematic(ax: plt.Axes) -> None:
    """Binary tree over nested route-field brackets, one row per budget K."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "A", "Address bandwidth")

    # ── binary tree: soma (root) on top, 8 leaf compartments below.  The
    # tree spans the same width as the bracket rows so leaves and route
    # fields sit on one alignment grid.
    leaf_x = np.linspace(0.360, 0.970, 8)
    levels = [leaf_x, (leaf_x[::2] + leaf_x[1::2]) / 2]
    levels.append((levels[1][::2] + levels[1][1::2]) / 2)
    levels.append(np.array([levels[2].mean()]))
    level_y = [0.530, 0.625, 0.720, 0.815]
    for level in range(3):
        for index, child_x in enumerate(levels[level]):
            parent_x = levels[level + 1][index // 2]
            ax.plot([child_x, parent_x], [level_y[level], level_y[level + 1]],
                    color="#A9B1B9", lw=LW_EDGE, zorder=1,
                    solid_capstyle="round")
    for value in leaf_x:
        ax.add_patch(Circle((value, level_y[0]), 0.013, facecolor=COLORS["exc"],
                            edgecolor="white", lw=0.4, zorder=2))
    for level in (1, 2):
        for value in levels[level]:
            ax.add_patch(Circle((value, level_y[level]), 0.015, facecolor=COLORS["dend"],
                                edgecolor="white", lw=0.4, zorder=2))
    ax.add_patch(Circle((levels[3][0], level_y[3]), 0.021, facecolor=COLORS["soma"],
                        edgecolor="white", lw=0.4, zorder=2))
    ax.text(levels[3][0] + 0.055, level_y[3] + 0.030, "soma", ha="left",
            va="center", fontsize=PT_SMALL, color=COLORS["mute"])
    ax.text(0.02, 1.00, "one somatic error\n→ nested route fields",
            ha="left", va="top", fontsize=PT_SMALL, color=COLORS["mute"],
            linespacing=1.35)

    # ── nested route-field brackets: K groups of consecutive leaves; plain
    # right-aligned colour-coded labels form one column in the left gutter.
    rows = [
        (0.440, 1, "$K{=}1$ shared", COLORS["point_mlp"]),
        (0.330, 2, "$K{=}2$", COLORS["additive"]),
        (0.220, 4, "$K{=}4$", COLORS["shunting"]),
        (0.110, 8, "$K{=}8$ exact", COLORS["oracle"]),
    ]
    margin, tick = 0.018, 0.022
    for y, budget, label, color in rows:
        group = len(leaf_x) // budget
        for g in range(budget):
            x0 = leaf_x[g * group] - margin
            x1 = leaf_x[(g + 1) * group - 1] + margin
            ax.plot([x0, x0, x1, x1], [y + tick, y, y, y + tick], color=color,
                    lw=LW_EDGE, solid_capstyle="round", zorder=2)
        ax.text(0.295, y + tick / 2, label, ha="right", va="center",
                fontsize=PT_SMALL, color=color)


def subtree_figure() -> None:
    apply_neurips_style()
    outcomes = pd.read_csv(SUBTREE / "seed_outcomes.csv")
    summary = pd.read_csv(SUBTREE / "condition_summary.csv")
    contrasts = pd.read_csv(SUBTREE / "paired_contrasts.csv")
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_W, 5.75),
        gridspec_kw={"left": 0.085, "right": 0.985, "bottom": 0.095, "top": 0.91, "wspace": 0.55, "hspace": 0.60},
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    hierarchy_schematic(ax_a)

    dendritic = summary[summary.architecture.eq("dendritic_tree")]
    ax_b.set_xlim(0.45, 8.55)
    ax_b.set_ylim(0.12, 0.86)
    # Families coincide exactly at K = 1 (four at 0.187) and K = 8 (five at
    # 0.810); the rank-K pair nearly coincides at K ≤ 2.  Draw the lines and
    # error bars at the true values, then fan exact pile-ups on a small ring
    # and dodge near-overlaps horizontally in point space — the panel-F
    # treatment — with one mute note naming the convention.
    rows_by_family = {
        family: dendritic[dendritic.feedback_family.eq(family)].sort_values("budget_k")
        for family in ROUTE_STYLE
    }
    members: dict[tuple[float, float], list[str]] = {}
    for family, part in rows_by_family.items():
        for _, row in part.iterrows():
            key = (float(row.budget_k), round(float(row.mean_heldout_accuracy), 3))
            members.setdefault(key, []).append(family)
    marker_shift: dict[tuple[str, float], tuple[float, float]] = {}
    for (budget, _), families in members.items():
        if len(families) < 2:
            continue
        radius = 2.6 + 0.3 * len(families)
        for index, family in enumerate(families):
            angle = np.pi / 2 + 2 * np.pi * index / len(families)
            marker_shift[(family, budget)] = (radius * np.cos(angle),
                                              radius * np.sin(angle))
    # Near-coincidences (distinct values closer than one marker diameter):
    # walk value-sorted neighbours at each K and dodge the pair sideways.
    axes_h_pt = ax_b.get_position().height * fig.get_figheight() * 72.0
    near = 3.4 * (ax_b.get_ylim()[1] - ax_b.get_ylim()[0]) / axes_h_pt
    for budget in (1.0, 2.0, 4.0, 8.0):
        stack = sorted(
            (float(part[part.budget_k.eq(budget)].mean_heldout_accuracy.iloc[0]), family)
            for family, part in rows_by_family.items()
        )
        for (y0, fam0), (y1, fam1) in zip(stack, stack[1:]):
            if y1 - y0 < near and 0 < y1 - y0 and (fam0, budget) not in marker_shift \
                    and (fam1, budget) not in marker_shift:
                marker_shift[(fam0, budget)] = (-3.0, 0.0)
                marker_shift[(fam1, budget)] = (3.0, 0.0)
    for family, (color, marker, _) in ROUTE_STYLE.items():
        part = rows_by_family[family]
        mean = part.mean_heldout_accuracy.to_numpy(float)
        low = part.ci95_low_heldout_accuracy.to_numpy(float)
        high = part.ci95_high_heldout_accuracy.to_numpy(float)
        budgets = part.budget_k.to_numpy(float)
        ax_b.plot(budgets, mean, color=color, lw=LW_DATA, zorder=2)
        for x, y, lo, hi in zip(budgets, mean, low, high):
            dx, dy = marker_shift.get((family, x), (0.0, 0.0))
            offset = mtransforms.offset_copy(
                ax_b.transData, fig=fig, x=dx, y=dy, units="points"
            )
            ax_b.errorbar(
                [x], [y], yerr=[[y - lo], [hi - y]], color=color,
                marker=marker, ms=3.6, lw=0, elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE, markeredgecolor="white",
                markeredgewidth=0.35, transform=offset, zorder=3,
            )
    ax_b.text(0.97, 0.24, "overlaps fanned\n(lines meet at values)",
              transform=ax_b.transAxes, ha="right", va="bottom",
              fontsize=PT_ANNOT, color=COLORS["mute"], linespacing=1.25)
    ax_b.set_xticks([1, 2, 4, 8])
    ax_b.set_xlabel("feedback channels $K$")
    ax_b.set_ylabel("held-out accuracy")
    panel_title(ax_b, "B", "Learning across $K$")
    style_axis(ax_b)
    # Figure-wide key for the six feedback families (panels B and F): one
    # centred row in the deliberate inter-row gutter, clear of every panel.
    handles = [
        Line2D([0], [0], color=color, marker=marker, ms=MARKER_MS - 0.6,
               lw=LW_DATA, markeredgecolor="white", markeredgewidth=0.35,
               label=label)
        for color, marker, label in ROUTE_STYLE.values()
    ]
    clean_legend(
        ax_b,
        handles=handles,
        fontsize=PT_SMALL,
        loc="center",
        bbox_to_anchor=(0.535, 0.496),
        bbox_transform=fig.transFigure,
        ncol=6,
        columnspacing=1.0,
        handlelength=1.2,
        handletextpad=0.45,
    )

    primary = contrasts[
        contrasts.architecture.eq("dendritic_tree")
        & contrasts.endpoint.eq("heldout_accuracy")
        & contrasts.contrast.eq("correct - best_matched_nonanatomical_oracle")
    ].sort_values("budget_k")
    ax_c.errorbar(primary.budget_k, 100 * primary.mean_difference, yerr=[100 * (primary.mean_difference - primary.ci95_low), 100 * (primary.ci95_high - primary.mean_difference)], color=COLORS["shunting"], marker="D", ms=MARKER_MS, lw=LW_ERR, capsize=ERR_CAPSIZE)
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_c.set_xticks([1, 2, 4, 8])
    ax_c.set_xlabel("feedback channels $K$")
    ax_c.set_ylabel("correct − best control (pp)")
    panel_title(ax_c, "C", "Best-control contrast")
    style_axis(ax_c)

    correct = outcomes[outcomes.feedback_family.eq("correct_ancestry_subtrees")]
    for budget_index, budget in enumerate([1, 2, 4, 8]):
        left = correct[correct.architecture.eq("dendritic_tree") & correct.budget_k.eq(budget)].set_index("seed").heldout_accuracy
        right = correct[correct.architecture.eq("degree_depth_matched_rewired_tree") & correct.budget_k.eq(budget)].set_index("seed").heldout_accuracy
        values = (left - right).to_numpy(float)
        mean, low, high = bootstrap(values, 70_000 + budget_index)
        ax_d.scatter(np.full(len(values), budget) + np.linspace(-0.13, 0.13, len(values)), 100 * values, s=SEED_MS**2, color=COLORS["shunting"], alpha=SEED_ALPHA, edgecolors="none")
        ax_d.errorbar(budget, 100 * mean, yerr=[[100 * (mean - low)], [100 * (high - mean)]], color=COLORS["shunting"], marker="D", markerfacecolor="white", ms=MARKER_MS + 1.2, lw=LW_ERR, capsize=ERR_CAPSIZE)
    ax_d.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    # Padding so the zero-valued mean diamonds at K = 1 and K = 8 clear the
    # left/bottom spines and the K = 2 seed cloud clears the top.
    ax_d.set_xlim(0.35, 8.65)
    ax_d.set_ylim(-3.4, 35.8)
    ax_d.set_xticks([1, 2, 4, 8])
    ax_d.set_xlabel("feedback channels $K$")
    ax_d.set_ylabel("matched − rewired tree (pp)")
    panel_title(ax_d, "D", "Task–topology alignment")
    style_axis(ax_d)

    # The four architectures produce *identical* curves under correct-ancestry
    # feedback.  Show that honestly: one shared line, and at *every* K the four
    # architecture markers stacked concentrically (largest behind), so each
    # budget visibly carries all four identities at once.
    equivalent_specs = [
        ("dendritic_tree", "dendritic", COLORS["shunting"], "o", 8.2),
        ("point_neuron_explicit_gating", "gated point", COLORS["point_mlp"], "s", 6.3),
        ("flat_compartment", "flat", COLORS["additive"], "^", 4.5),
        ("grouped_point_subunits", "grouped point", COLORS["per_soma"], "D", 2.7),
    ]
    base = summary[
        summary.architecture.eq("dendritic_tree")
        & summary.feedback_family.eq("correct_ancestry_subtrees")
    ].sort_values("budget_k")
    ax_e.plot(base.budget_k, base.mean_heldout_accuracy, color="#B9C0C8", lw=LW_DATA, zorder=1)
    for depth, (architecture, label, color, marker, size) in enumerate(equivalent_specs):
        part = summary[
            summary.architecture.eq(architecture)
            & summary.feedback_family.eq("correct_ancestry_subtrees")
        ].sort_values("budget_k")
        ax_e.plot(part.budget_k, part.mean_heldout_accuracy, marker=marker,
                  color=color, ms=size, lw=0, markeredgecolor="white",
                  markeredgewidth=0.4, zorder=3 + depth)
    ax_e.text(0.97, 0.54, "all four coincide\n(markers nested)",
              transform=ax_e.transAxes, ha="right", va="bottom",
              fontsize=PT_ANNOT, color=COLORS["mute"], linespacing=1.25)
    ax_e.set_xticks([1, 2, 4, 8])
    ax_e.set_xlabel("feedback channels $K$")
    ax_e.set_ylabel("held-out accuracy")
    ax_e.set_ylim(0.12, 0.86)
    panel_title(ax_e, "E", "Representation match")
    style_axis(ax_e)
    arch_handles = [
        Line2D([0], [0], marker=marker, color=color, lw=0, ms=MARKER_MS,
               markeredgecolor="white", markeredgewidth=0.4, label=label)
        for _, label, color, marker, _ in equivalent_specs
    ]
    clean_legend(ax_e, handles=arch_handles, fontsize=PT_SMALL, loc="lower right")

    selected = dendritic[dendritic.feedback_family.isin(ROUTE_STYLE)].copy()
    # Different families land on *exactly* the same (capture, accuracy) point
    # at K = 1 and K = 8.  Fan those markers on a small ring in point space —
    # the true value stays at the ring centre, where the family lines
    # converge — and dodge *near*-coincident pairs (measured in point space
    # against the size-coded marker diameters) apart along their own
    # separation axis, so e.g. the derangement K = 2/K = 4 squares one point
    # apart at capture = 0 both stay readable.  Limits are fixed here so the
    # data→point conversion below is exact.
    ax_f.set_xlim(-0.05, 1.10)
    ax_f.set_ylim(0.12, 0.86)

    def _key(x: float, y: float) -> tuple[float, float]:
        return (round(float(x), 4), round(float(y), 4))

    members: dict[tuple[float, float], list[str]] = {}
    for family in ROUTE_STYLE:
        part = selected[selected.feedback_family.eq(family)]
        for _, row in part.iterrows():
            key = _key(row.mean_initial_gradient_capture, row.mean_heldout_accuracy)
            members.setdefault(key, []).append(family)
    fan: dict[tuple[str, tuple[float, float]], tuple[float, float]] = {}
    for key, families in members.items():
        if len(families) < 2:
            continue
        radius = 3.4 + 0.35 * len(families)
        for index, family in enumerate(families):
            angle = np.pi / 2 + 2 * np.pi * index / len(families)
            fan[(family, key)] = (radius * np.cos(angle), radius * np.sin(angle))
    # Near-coincidences the exact-key fan misses: any two unfanned markers
    # whose centres sit closer than their mean diameter get pushed apart along
    # their existing separation axis (the panel-B nearest-neighbour dodge),
    # keeping their value order; the family lines stay at the true values.
    box_f = ax_f.get_position()
    x_pt = box_f.width * fig.get_figwidth() * 72.0 / np.diff(ax_f.get_xlim())[0]
    y_pt = box_f.height * fig.get_figheight() * 72.0 / np.diff(ax_f.get_ylim())[0]
    loose = [
        (family,
         _key(row.mean_initial_gradient_capture, row.mean_heldout_accuracy),
         float(np.sqrt(13.0 + 3.2 * float(row.budget_k))))
        for family in ROUTE_STYLE
        for _, row in selected[selected.feedback_family.eq(family)].iterrows()
        if (family, _key(row.mean_initial_gradient_capture,
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
    for family, (color, marker, _) in ROUTE_STYLE.items():
        part = selected[selected.feedback_family.eq(family)].sort_values("budget_k")
        ax_f.plot(
            part.mean_initial_gradient_capture,
            part.mean_heldout_accuracy,
            color=color,
            lw=LW_HAIR,
            alpha=0.30,
            zorder=1,
        )
        for _, row in part.sort_values("budget_k", ascending=False).iterrows():
            x = float(row.mean_initial_gradient_capture)
            y = float(row.mean_heldout_accuracy)
            dx, dy = fan.get((family, _key(x, y)), (0.0, 0.0))
            offset = mtransforms.offset_copy(
                ax_f.transData, fig=fig, x=dx, y=dy, units="points"
            )
            ax_f.plot(
                [x], [y],
                marker=marker,
                color=color,
                ms=float(np.sqrt(13.0 + 3.2 * float(row.budget_k))),
                lw=0,
                alpha=0.9,
                markeredgecolor="white",
                markeredgewidth=0.35,
                transform=offset,
                zorder=3.0 - 0.05 * float(row.budget_k),
            )
    ax_f.text(0.97, 0.03, "marker size $\\propto$ $K$\noverlaps fanned",
              transform=ax_f.transAxes, ha="right", va="bottom",
              fontsize=PT_ANNOT, color=COLORS["mute"], linespacing=1.25)
    ax_f.set_xlabel("initial gradient capture")
    ax_f.set_ylabel("held-out accuracy")
    panel_title(ax_f, "F", "Capture and learning")
    style_axis(ax_f)
    save(fig, "fig_trained_subtree_full_factorial")


def active_schematic(ax: plt.Axes) -> None:
    """Mini dendrite with active-channel swatches and a focal shunt callout."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "J", "Active channels")

    ax.text(0.50, 0.925, "Na · K · Ca · HCN · NMDA",
            ha="center", va="center", fontsize=PT_SMALL,
            color=COLORS["mute"])

    # ── mini dendrite: soma, tapering trunk → branches → twigs, with the
    # channel conductances drawn as beads on the membrane they live in.
    dend = COLORS["dend"]
    ax.plot([0.47, 0.47], [0.325, 0.47], color=dend, lw=3.0, solid_capstyle="round")
    ax.add_patch(Circle((0.47, 0.285), 0.036, facecolor=COLORS["soma"],
                        edgecolor="white", lw=0.5))
    ax.text(0.535, 0.285, "soma", ha="left", va="center", fontsize=PT_SMALL,
            color=COLORS["mute"])
    ax.plot([0.47, 0.25], [0.47, 0.63], color=dend, lw=2.0, solid_capstyle="round")
    ax.plot([0.47, 0.68], [0.47, 0.615], color=dend, lw=2.0, solid_capstyle="round")
    ax.plot([0.25, 0.13], [0.63, 0.735], color=dend, lw=1.4, solid_capstyle="round")
    ax.plot([0.25, 0.34], [0.63, 0.755], color=dend, lw=1.4, solid_capstyle="round")
    ax.plot([0.68, 0.59], [0.615, 0.750], color=dend, lw=1.4, solid_capstyle="round")
    ax.plot([0.68, 0.84], [0.615, 0.700], color=dend, lw=1.4, solid_capstyle="round")
    beads = [
        (0.470, 0.385, COLORS["exc"]),        # Na on the trunk
        (0.360, 0.550, COLORS["local"]),      # K on the left branch
        (0.190, 0.682, COLORS["highlight"]),  # Ca on the left twig
        (0.565, 0.535, COLORS["oracle"]),     # HCN on the right branch
        (0.768, 0.662, COLORS["per_soma"]),   # NMDA on the right twig
    ]
    for x, y, color in beads:
        ax.add_patch(Circle((x, y), 0.015, facecolor=color, edgecolor="white",
                            lw=0.4, zorder=3))

    # ── focal shunt on the right branch, label in clear whitespace ──
    ax.add_patch(Circle((0.638, 0.586), 0.024, facecolor=COLORS["inh"],
                        edgecolor="white", lw=0.5, zorder=3))
    ax.annotate(
        "focal shunt",
        xy=(0.660, 0.573), xytext=(0.97, 0.44),
        ha="right", va="center", fontsize=PT_ANNOT, color=COLORS["inh"],
        arrowprops={"arrowstyle": "-|>", "lw": LW_REF, "color": COLORS["inh"],
                    "mutation_scale": 7.0, "shrinkA": 3.0, "shrinkB": 1.5},
    )

    ax.text(0.47, 0.135, "512 accepted cell–draws · exact Jacobian",
            ha="center", va="center", fontsize=PT_SMALL, color=COLORS["mute"])


def active_extension_figure() -> None:
    """Keep the active-channel test with the focal-shunting mechanism."""
    apply_neurips_style()
    active_summary = pd.read_csv(ACTIVE / "condition_summary.csv")
    active_contrasts = pd.read_csv(ACTIVE / "paired_contrasts.csv")
    active_cells = pd.read_csv(ACTIVE / "cell_condition_metrics.csv")
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(FIG_W, 2.62),
        gridspec_kw={"left": 0.075, "right": 0.985, "bottom": 0.165,
                     "top": 0.855, "wspace": 0.72},
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    active_schematic(ax_a)

    for perturbation, color, marker in [("focal shunt", COLORS["shunting"], "o"), ("matched additive", COLORS["additive"], "s")]:
        part = active_summary[active_summary.perturbation.eq(perturbation)].sort_values("dose_relative_to_local_input_conductance")
        x = part.dose_relative_to_local_input_conductance.to_numpy(float)
        ax_b.plot(x, part.mean_localization_index, color=color, marker=marker, ms=MARKER_MS, lw=LW_DATA, markeredgecolor="white", markeredgewidth=0.35, label=perturbation)
        ax_b.fill_between(x, part.ci95_low_localization_index, part.ci95_high_localization_index, color=color, alpha=0.10, linewidth=0)
    ax_b.set_xscale("log"); ax_b.set_xticks([0.25, 1, 4], ["0.25", "1", "4"])
    ax_b.set_xlabel("normalized shunt dose")
    ax_b.set_ylabel("descendant localization")
    panel_title(ax_b, "K", "Dose response")
    style_axis(ax_b)
    clean_legend(ax_b, fontsize=PT_LEGEND, loc="upper left")

    localization = active_contrasts[active_contrasts.metric.eq("localization_index")].sort_values("dose_relative_to_local_input_conductance")
    ax_c.errorbar(localization.dose_relative_to_local_input_conductance, localization.mean_shunt_minus_additive, yerr=[localization.mean_shunt_minus_additive - localization.ci95_low, localization.ci95_high - localization.mean_shunt_minus_additive], color=COLORS["shunting"], marker="D", ms=MARKER_MS, lw=LW_ERR, capsize=ERR_CAPSIZE)
    for dose in [0.25, 1, 4]:
        values = active_cells[np.isclose(active_cells.dose_relative_to_local_input_conductance, dose)].pivot(index="root_id", columns="perturbation", values="localization_index")
        diff = values["focal shunt"] - values["matched additive"]
        ax_c.scatter(np.full(len(diff), dose) * np.exp(np.linspace(-0.045, 0.045, len(diff))), diff, s=SEED_MS**2, color=COLORS["shunting"], alpha=SEED_ALPHA, edgecolors="none")
    ax_c.set_xscale("log"); ax_c.set_xticks([0.25, 1, 4], ["0.25", "1", "4"])
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_c.set_xlabel("normalized shunt dose")
    ax_c.set_ylabel("shunt − additive localization")
    panel_title(ax_c, "L", "Cellwise contrast")
    style_axis(ax_c)

    # Horizontal bars keep every category label horizontal, give zero-valued
    # bars a visible "0.00" annotation, and let a dashed rule separate the
    # fraction bars from the energy-ratio bar (different units).
    dose_one = active_summary[active_summary.perturbation.eq("focal shunt") & np.isclose(active_summary.dose_relative_to_local_input_conductance, 1)].iloc[0]
    names = ["attenuated", "enhanced", "sign flip", "energy ratio"]
    values = [dose_one.mean_descendant_attenuated_fraction, dose_one.mean_descendant_enhanced_fraction, dose_one.mean_descendant_sign_flip_fraction, dose_one.mean_descendant_gradient_energy_ratio]
    bar_colors = [COLORS["shunting"], COLORS["additive"], COLORS["mute"], COLORS["oracle"]]
    positions = [3, 2, 1, 0]
    ax_d.barh(positions, values, color=bar_colors, height=0.62, edgecolor="white", linewidth=0.5)
    for y, value in zip(positions, values):
        ax_d.text(value + 0.04, y, f"{value:.2f}", ha="left", va="center",
                  fontsize=PT_ANNOT, color=COLORS["ink"])
    ax_d.axhline(0.5, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_d.set_yticks(positions, wrap_ticklabels(names, width=9))
    ax_d.set_ylim(-0.55, 3.55)
    ax_d.set_xlim(0, 1.30)
    ax_d.set_xticks([0, 0.5, 1.0])
    ax_d.set_xlabel("value at unit dose")
    panel_title(ax_d, "M", "Signed outcome")
    style_axis(ax_d, grid="x")

    save(fig, "fig_active_focal_extension")


def fulltree_boundary_figure() -> None:
    """Place the complete-tree response task with the other alignment nulls."""
    apply_neurips_style()
    tree_cells = pd.read_csv(FULL_TREE / "cell_method_means.csv")
    fig, (ax_e, ax_f) = plt.subplots(
        1,
        2,
        figsize=(FIG_W, 2.62),
        gridspec_kw={"left": 0.075, "right": 0.985, "bottom": 0.165,
                     "top": 0.855, "wspace": 0.42},
    )

    # "exact" gets the chromatic backprop slot (bp red-brown) so it can never
    # be confused with the neutral-gray "random" control, in-panel or when
    # cross-read against panel F where gray also means "random".
    methods = ["exact compartment error", "topology-matched routes", "site-shuffled routes", "random anatomical routes"]
    labels = ["exact", "topology", "shuffle", "random"]
    colors = [COLORS["bp"], COLORS["shunting"], COLORS["highlight"], COLORS["mute"]]
    for index, (method, color) in enumerate(zip(methods, colors)):
        values = tree_cells[tree_cells.method.eq(method)].heldout_normalized_mse.to_numpy(float)
        ax_e.scatter(index + np.linspace(-0.06, 0.06, len(values)), values, s=SEED_MS**2, color=color, alpha=SEED_ALPHA, edgecolors="none")
        mean, low, high = bootstrap(values, 90_000 + index)
        ax_e.errorbar(index, mean, yerr=[[mean - low], [high - mean]], color=color, marker="D", markerfacecolor="white", ms=MARKER_MS + 1.2, lw=LW_ERR, capsize=ERR_CAPSIZE)
    ax_e.set_xticks(range(4), labels)
    ax_e.set_ylabel("normalized test MSE")
    panel_title(ax_e, "I", "All-scan full-tree task")
    style_axis(ax_e, grid="y")
    add_headroom(ax_e, 0.20, bottom=True)
    ax_e.text(0.03, 0.03, "13 scans / 7 cells", transform=ax_e.transAxes,
              ha="left", va="bottom", fontsize=PT_ANNOT, color=COLORS["mute"])

    tree_wide = tree_cells.pivot(index="target_root_id", columns="method", values=["heldout_normalized_mse", "common_checkpoint_update_capture"])
    contrasts = [
        ("shuffle", tree_wide[("heldout_normalized_mse", "site-shuffled routes")] - tree_wide[("heldout_normalized_mse", "topology-matched routes")], COLORS["highlight"]),
        ("random", tree_wide[("heldout_normalized_mse", "random anatomical routes")] - tree_wide[("heldout_normalized_mse", "topology-matched routes")], COLORS["mute"]),
        ("shuffle", tree_wide[("common_checkpoint_update_capture", "topology-matched routes")] - tree_wide[("common_checkpoint_update_capture", "site-shuffled routes")], COLORS["highlight"]),
        ("random", tree_wide[("common_checkpoint_update_capture", "topology-matched routes")] - tree_wide[("common_checkpoint_update_capture", "random anatomical routes")], COLORS["mute"]),
    ]
    # The four contrasts pair up by endpoint: a gap plus a hairline divider
    # separates the MSE pair from the capture pair, so the single-line control
    # labels can no longer run together, and a mute group header names each
    # endpoint beneath its pair.
    positions = [0.0, 1.2, 2.7, 3.9]
    for position, (label, values, color) in zip(positions, contrasts):
        array = values.to_numpy(float)
        ax_f.scatter(position + np.linspace(-0.06, 0.06, len(array)), array, s=SEED_MS**2, color=color, alpha=SEED_ALPHA, edgecolors="none")
        mean, low, high = bootstrap(array, 91_000 + positions.index(position))
        ax_f.errorbar(position, mean, yerr=[[mean - low], [high - mean]], color=color, marker="D", markerfacecolor="white", ms=MARKER_MS + 1.2, lw=LW_ERR, capsize=ERR_CAPSIZE)
    ax_f.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_f.axvline(1.95, color=COLORS["grid"], lw=LW_HAIR, zorder=0)
    ax_f.set_xticks(positions, [label for label, _, _ in contrasts])
    group_transform = mtransforms.blended_transform_factory(
        ax_f.transData, ax_f.transAxes
    )
    for centre, header in [(0.6, "test MSE"), (3.3, "capture")]:
        ax_f.text(centre, -0.155, header, transform=group_transform,
                  ha="center", va="top", fontsize=PT_ANNOT,
                  color=COLORS["mute"])
    ax_f.set_ylabel("topology advantage")
    panel_title(ax_f, "J", "Anatomy boundary")
    style_axis(ax_f, grid="y")
    add_headroom(ax_f, 0.18)
    ax_f.text(0.50, 0.97, "positive favors topology", transform=ax_f.transAxes,
              ha="center", va="top", fontsize=PT_ANNOT, color=COLORS["mute"],
              style="italic")
    save(fig, "fig_fulltree_boundary")


def main() -> None:
    subtree_figure()
    active_extension_figure()
    fulltree_boundary_figure()


if __name__ == "__main__":
    main()

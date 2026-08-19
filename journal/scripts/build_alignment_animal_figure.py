#!/usr/bin/env python3
"""Build the integrated Figure 8 from frozen alignment and animal source data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

from journal_style import (
    COLORS as STYLE_COLORS,
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
    PT_TICK,
    SEED_MS,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    panel_title,
    style_axis,
)
from credit_tree_schematics import (
    EDGES_A,
    EDGES_B,
    EDGES_C,
    EDGES_D,
    JUNCTIONS,
    P as TREE_P,
    ROOT_PT,
    SOMA_R,
    mix,
)


ROOT = Path(__file__).resolve().parents[1]
ALIGNMENT = ROOT / "source_data" / "alignment_controlled"
ANIMAL = ROOT / "source_data" / "animal_learning_francioni"
OUTPUT_ALIGNMENT = ROOT / "figures" / "generated" / "fig8_alignment_controlled"
OUTPUT_ANIMAL = ROOT / "figures" / "generated" / "fig_animal_credit_supplement"

METHODS = (
    "morphology-selected paths",
    "random paths",
    "depth bins",
    "ancestry-shuffled paths",
)
METHOD_COLORS = {
    "morphology-selected paths": STYLE_COLORS["shunting"],
    "random paths": STYLE_COLORS["point_mlp"],
    "depth bins": STYLE_COLORS["additive"],
    "ancestry-shuffled paths": STYLE_COLORS["highlight"],
}
METHOD_MARKERS = {
    "morphology-selected paths": "o",
    "random paths": "s",
    "depth bins": "^",
    "ancestry-shuffled paths": "D",
}
METHOD_LABELS = {
    "morphology-selected paths": "ancestry",
    "random paths": "random",
    "depth bins": "depth",
    "ancestry-shuffled paths": "shuffle",
}
# Route highlight in the design schematic shares the morphology-series green.
ROUTE = STYLE_COLORS["shunting"]
# Animal BCI populations get hues of their own (amber / violet) so the bottom
# row can never be misread through the morphology/depth key of the top row.
PPLUS = STYLE_COLORS["local"]
PMINUS = STYLE_COLORS["oracle"]


def _schematic_frame(ax: plt.Axes) -> float:
    """Axis-off frame with equal aspect filling the panel box; returns ymax."""
    ax.set_axis_off()
    fig = ax.figure
    box = ax.get_position()
    width_in = box.width * fig.get_figwidth()
    height_in = box.height * fig.get_figheight()
    ymax = height_in / width_in
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_aspect("equal")
    return ymax


# Credit-tree schematic vocabulary (shared library geometry): deck taper in
# TikZ pt, normalized so the trunk prints at LW_DATA, tempered exactly like
# the library's scale-0.62 inset rendering.
_TAPER_PT = {"A": 1.60, "B": 1.20, "C": 0.90, "D": 0.70}
_TREE_LW = LW_DATA / _TAPER_PT["A"] * 0.62
# Morphology-selected (ancestry) route: one root-to-leaf path of the tree.
_ROUTE_PATH = ((ROOT_PT, "J1", "A"), ("J1", "JL", "B"),
               ("JL", "JLR", "C"), ("JLR", "T4", "D"))
_ROUTE_JUNCTIONS = ("J1", "JL", "JLR")


def _tree_pt(q):
    return TREE_P[q] if isinstance(q, str) else q


def _route_tree(ax: plt.Axes) -> None:
    """Library credit tree, muted, with one ancestry route selected in green."""
    ax.set_aspect("equal", adjustable="datalim")
    ax.update_datalim([(-2.55, -0.72), (2.55, 3.60)])
    ax.margins(0)
    ax.autoscale_view()
    ax.axis("off")
    faded = mix("mute", 30)
    for edges, level in ((EDGES_A, "A"), (EDGES_B, "B"),
                         (EDGES_C, "C"), (EDGES_D, "D")):
        for a, b in edges:
            (x0, y0), (x1, y1) = _tree_pt(a), _tree_pt(b)
            ax.plot([x0, x1], [y0, y1], color=faded,
                    lw=_TAPER_PT[level] * _TREE_LW, solid_capstyle="round",
                    zorder=2)
    for a, b, level in _ROUTE_PATH:
        (x0, y0), (x1, y1) = _tree_pt(a), _tree_pt(b)
        ax.plot([x0, x1], [y0, y1], color=ROUTE,
                lw=_TAPER_PT[level] * _TREE_LW, solid_capstyle="round",
                zorder=2.4)
    # Credit flows outward along the selected route (transport-mode arrows).
    for a, b in ((ROOT_PT, "J1"), ("JL", "JLR")):
        (x0, y0), (x1, y1) = _tree_pt(a), _tree_pt(b)
        start = (x0 + 0.38 * (x1 - x0), y0 + 0.38 * (y1 - y0))
        end = (x0 + 0.72 * (x1 - x0), y0 + 0.72 * (y1 - y0))
        ax.add_patch(FancyArrowPatch(start, end,
                                     arrowstyle="-|>,head_length=3.4,head_width=2.1",
                                     mutation_scale=1.0, lw=0.8 * _TREE_LW,
                                     color=ROUTE, shrinkA=0, shrinkB=0,
                                     zorder=4.5))
    for name in JUNCTIONS:
        on_route = name in _ROUTE_JUNCTIONS
        ax.plot(*_tree_pt(name), marker="o", ms=2.0, mfc="white",
                mec=ROUTE if on_route else mix("mute", 40), mew=LW_EDGE * 0.62,
                ls="none", zorder=3)
    ax.plot(*_tree_pt("T4"), marker="o", ms=2.0, mfc=ROUTE, mec="none",
            ls="none", zorder=4)
    ax.add_patch(Circle((0.0, 0.0), SOMA_R, fc=STYLE_COLORS["soma"],
                        ec=mix("ink", 30), lw=LW_EDGE, zorder=3.5))
    ax.text(-0.22, 3.32, "route", ha="center", va="bottom",
            fontsize=PT_SMALL, color=ROUTE)


def alignment_schematic(ax: plt.Axes) -> None:
    _schematic_frame(ax)
    panel_title(ax, "K", "Alignment design")
    mute = STYLE_COLORS["mute"]
    ink = STYLE_COLORS["ink"]

    # Top: the shared credit tree with one morphology-selected route.
    tree_ax = ax.inset_axes([0.0, 0.50, 1.0, 0.50])
    _route_tree(tree_ax)

    # Bottom: credit vignette — target gradient vs routed credit, equal norm
    # (the dashed arc states the matched norm; theta is the dose in L-M).
    origin = (0.50, 0.29)
    radius = 0.56
    ang_target, ang_routed = np.deg2rad(100.0), np.deg2rad(62.0)
    tip_target = (origin[0] + radius * np.cos(ang_target),
                  origin[1] + radius * np.sin(ang_target))
    tip_routed = (origin[0] + radius * np.cos(ang_routed),
                  origin[1] + radius * np.sin(ang_routed))
    ax.add_patch(Arc(origin, 2 * radius, 2 * radius, theta1=56, theta2=106,
                     ls=(0, (2.4, 2.0)), lw=LW_HAIR, color=mute, zorder=1))
    ax.add_patch(Arc(origin, 0.46, 0.46, theta1=62, theta2=100,
                     lw=LW_HAIR, color=mute, zorder=1))
    ax.add_patch(FancyArrowPatch(origin, tip_target, arrowstyle="-|>",
                                 mutation_scale=9, lw=LW_DATA, color=ink,
                                 shrinkA=0, shrinkB=0, zorder=3))
    ax.add_patch(FancyArrowPatch(origin, tip_routed, arrowstyle="-|>",
                                 mutation_scale=9, lw=LW_DATA, color=ROUTE,
                                 shrinkA=0, shrinkB=0, zorder=3))
    ax.text(origin[0] + 0.065, origin[1] + 0.37, r"$\theta$", ha="center",
            va="center", fontsize=PT_SMALL, color=ink)
    ax.text(tip_target[0] - 0.10, tip_target[1] - 0.07, "target", ha="right",
            va="center", fontsize=PT_SMALL, color=ink)
    ax.text(tip_routed[0] + 0.03, tip_routed[1] - 0.13, "routed", ha="left",
            va="center", fontsize=PT_SMALL, color=ROUTE)
    ax.text(0.50, 0.07, r"$\theta$ = route alignment", ha="center",
            va="center", fontsize=PT_SMALL, color=mute)


def alignment_curve(ax: plt.Axes, curve: pd.DataFrame, metric: str, letter: str,
                    title: str, ylabel: str) -> None:
    for method in METHODS:
        part = curve[curve.method.eq(method)].sort_values("alignment")
        x = 100 * part.alignment.to_numpy(float)
        y = part[metric].to_numpy(float)
        low = part[f"{metric}_ci_low"].to_numpy(float)
        high = part[f"{metric}_ci_high"].to_numpy(float)
        ax.fill_between(x, low, high, color=METHOD_COLORS[method], alpha=0.12, linewidth=0)
        ax.plot(x, y, color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                ms=MARKER_MS, mec="white", mew=0.5, lw=LW_DATA,
                label=METHOD_LABELS[method])
    ax.set_xlim(-2, 102)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xticks([0, 50, 100])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("route alignment (%)")
    ax.set_ylabel(ylabel)
    panel_title(ax, letter, title)
    style_axis(ax, grid="y")


def alignment_relation(ax: plt.Axes, cell: pd.DataFrame) -> None:
    # Reversed draw order puts the sparse morphology series on top; the mute
    # note declares the honest coincidence of the four strategies.
    for method in reversed(METHODS):
        part = cell[cell.method.eq(method)]
        ax.scatter(part.credit_capture, part.iterative_progress, s=SEED_MS**2,
                   color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
                   alpha=0.42, linewidths=0)
    ax.plot([0, 1], [0, 1], color=STYLE_COLORS["mute"], lw=LW_REF, ls="--")
    controls = cell[cell.method.ne(METHODS[0])]
    relations = [
        stats.spearmanr(group.credit_capture, group.iterative_progress).statistic
        for _, group in controls.groupby("root_id")
    ]
    ax.text(0.05, 0.95, rf"median $\rho_s={np.median(relations):.2f}$",
            transform=ax.transAxes, va="top", fontsize=PT_ANNOT,
            color=STYLE_COLORS["ink"])
    ax.text(0.97, 0.04, "all methods\ncoincide",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=PT_SMALL, color=STYLE_COLORS["mute"])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.04)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_xlabel("field capture")
    ax.set_ylabel("20-step progress")
    panel_title(ax, "N", "Capture–progress")
    style_axis(ax)


def animal_schematic(ax: plt.Axes) -> None:
    _schematic_frame(ax)
    panel_title(ax, "A", "Causal sign")
    mute = STYLE_COLORS["mute"]
    ink = STYLE_COLORS["ink"]

    ax.add_patch(FancyBboxPatch((0.275, 1.035), 0.45, 0.13,
                                boxstyle="round,pad=0.02,rounding_size=0.035",
                                fc=STYLE_COLORS["panel_bg"], ec=mute,
                                lw=LW_EDGE, zorder=2))
    ax.text(0.50, 1.10, "BCI error", ha="center", va="center",
            fontsize=PT_ANNOT, color=ink, zorder=3)

    for x, label, color in ((0.27, "P+", PPLUS), (0.73, "P−", PMINUS)):
        start = (0.43, 1.005) if x < 0.5 else (0.57, 1.005)
        end = (x + (0.045 if x < 0.5 else -0.045), 0.78)
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                     mutation_scale=8, lw=LW_ERR, color=color,
                                     shrinkA=0, shrinkB=0, zorder=2))
        ax.add_patch(Circle((x, 0.63), 0.125, fc="white", ec=color,
                            lw=LW_DATA, zorder=3))
        ax.text(x, 0.63, label, ha="center", va="center", color=ink,
                fontsize=PT_ANNOT, zorder=4)
        ax.text(x, 0.42, "sign +" if x < 0.5 else "sign −", ha="center",
                va="center", color=ink, fontsize=PT_SMALL)

    # Predicted contrast glyphs: opposite signs about a zero baseline.
    ax.plot([0.10, 0.90], [0.20, 0.20], ls=(0, (2.4, 2.0)), lw=LW_HAIR,
            color=mute, zorder=1)
    ax.text(0.055, 0.20, "0", ha="center", va="center", fontsize=PT_SMALL,
            color=mute)
    ax.add_patch(FancyArrowPatch((0.27, 0.205), (0.27, 0.335),
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=LW_ERR, color=PPLUS, shrinkA=0, shrinkB=0,
                                 zorder=2))
    ax.add_patch(FancyArrowPatch((0.73, 0.195), (0.73, 0.10),
                                 arrowstyle="-|>", mutation_scale=7,
                                 lw=LW_ERR, color=PMINUS, shrinkA=0, shrinkB=0,
                                 zorder=2))
    ax.text(0.47, 0.03, "predicted contrast", ha="center", va="center",
            fontsize=PT_SMALL, color=mute)


def animal_pairs(ax: plt.Axes, animal: pd.DataFrame) -> None:
    for row in animal.itertuples(index=False):
        ax.plot([0, 1], [row.pplus_contrast, row.pminus_contrast],
                color=STYLE_COLORS["mute"], lw=LW_HAIR, alpha=0.55, zorder=1)
        ax.scatter(0, row.pplus_contrast, s=18, color=PPLUS, edgecolor="white",
                   linewidth=0.4, zorder=3)
        ax.scatter(1, row.pminus_contrast, s=18, color=PMINUS, edgecolor="white",
                   linewidth=0.4, zorder=3)
    ax.axhline(0, color=STYLE_COLORS["mute"], lw=LW_REF, ls="--", zorder=0)
    ax.set_xticks([0, 1], ["P+", "P−"])
    ax.set_ylabel("dendritic contrast\n(reduction − increase)")
    panel_title(ax, "B", "6/6 signed pairs")
    style_axis(ax, grid="y")


def mode_energy(ax: plt.Axes, summary: dict) -> None:
    mode = summary["mode_decomposition"]
    fractions = [mode["common_energy_fraction"], mode["signed_energy_fraction"]]
    ci_lo, ci_hi = mode["animal_bootstrap_95_ci"]
    intervals = [(1.0 - ci_hi, 1.0 - ci_lo), (ci_lo, ci_hi)]
    ax.bar([0, 1], fractions,
           color=[STYLE_COLORS["point_mlp"], STYLE_COLORS["bp"]],
           edgecolor=STYLE_COLORS["edge"], linewidth=LW_EDGE, width=0.64)
    yerr = np.array([[value - lo for value, (lo, _) in zip(fractions, intervals)],
                     [hi - value for value, (_, hi) in zip(fractions, intervals)]])
    ax.errorbar([0, 1], fractions, yerr=yerr, fmt="none",
                ecolor=STYLE_COLORS["edge"], elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=4)
    for x, value, (_, hi) in zip((0, 1), fractions, intervals):
        ax.text(x, hi + 0.035, f"{100 * value:.1f}%", ha="center", va="bottom",
                fontsize=PT_ANNOT, color=STYLE_COLORS["ink"])
    ax.text(0.03, 0.97, "95% CI,\nanimal\nbootstrap", transform=ax.transAxes,
            ha="left", va="top", fontsize=PT_SMALL, color=STYLE_COLORS["mute"])
    ax.set_xticks([0, 1], ["common", "signed"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("contrast energy")
    panel_title(ax, "C", "Signed mode")
    style_axis(ax, grid="y")


def neuron_distributions(ax: plt.Axes, neuron: pd.DataFrame) -> None:
    order = [
        ("P+", "error increase"),
        ("P+", "error reduction"),
        ("P-", "error increase"),
        ("P-", "error reduction"),
    ]
    labels = ["inc.", "red.", "inc.", "red."]
    positions = [0.0, 1.0, 2.5, 3.5]
    colors = [PPLUS, PPLUS, PMINUS, PMINUS]
    for (population, epoch), color, pos in zip(order, colors, positions):
        values = neuron.loc[
            (neuron.population == population) & (neuron.epoch == epoch),
            "sd_residual_z",
        ].to_numpy(float)
        parts = ax.violinplot(values, positions=[pos], widths=0.72,
                              showmeans=False, showmedians=False, showextrema=False)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.28)
        mean = float(np.mean(values))
        sem = float(stats.sem(values))
        ax.errorbar(pos, mean, yerr=sem, fmt="o", ms=MARKER_MS + 1.2, color=color,
                    mec="white", mew=0.4, lw=LW_ERR, capsize=ERR_CAPSIZE, zorder=4)
    ax.axhline(0, color=STYLE_COLORS["mute"], lw=LW_REF, ls="--", zorder=0)
    # Keep the first line clear of the P− 'inc.' violin tail (pos 2.5).
    ax.text(0.03, 0.97, "s.e.m. <\nmarker size", transform=ax.transAxes,
            ha="left", va="top", fontsize=PT_SMALL, color=STYLE_COLORS["mute"])
    ax.set_xticks(positions, labels)
    ax.set_xlim(-0.65, 4.15)
    for center, group in ((0.5, "P+"), (3.0, "P−")):
        ax.text(center, -0.20, group, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=PT_TICK,
                color=STYLE_COLORS["ink"])
    ax.set_ylabel("residual (z-score)")
    panel_title(ax, "D", "Neuron residuals")
    style_axis(ax, grid="y")


def _save(fig: plt.Figure, output: Path) -> None:
    fig.canvas.draw()
    layout = audit_layout(fig, output.name)
    overlap = audit_text_over_data(fig, output.name)
    if layout or overlap:
        print(f"layout audit: {len(layout)} layout and {len(overlap)} text/data warnings")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(output.with_suffix(".png"), dpi=600)
    plt.close(fig)
    print(f"Wrote {output.with_suffix('.pdf')}")


def main() -> None:
    curve = pd.read_csv(ALIGNMENT / "alignment_controlled_curves.csv")
    cell = pd.read_csv(ALIGNMENT / "cell_alignment_metrics.csv")
    animal = pd.read_csv(ANIMAL / "animal_signed_contrasts.csv")
    neuron = pd.read_csv(ANIMAL / "neuron_sd_residual_distributions.csv")
    summary = json.loads((ANIMAL / "summary.json").read_text())

    apply_neurips_style()
    fig = plt.figure(figsize=(FIG_W, 2.54))
    # left = 0.088 puts the K letter in the same gutter column as the A-H
    # block; the wider top/right margins match the sibling fig-8 blocks
    # (letters and panel N's last tick no longer graze the canvas edges).
    grid = fig.add_gridspec(
        1, 4, left=0.088, right=0.985, bottom=0.205, top=0.842,
        wspace=0.62,
    )
    axes = [fig.add_subplot(grid[0, col]) for col in range(4)]

    # The schematic column has no x label or legend row: let it use the
    # bottom margin so its content fills the cell like the data panels.
    box = axes[0].get_position()
    axes[0].set_position([box.x0, 0.055, box.width, box.y1 - 0.055])

    alignment_schematic(axes[0])
    alignment_curve(axes[1], curve, "credit_capture", "L", "Field capture",
                    "field capture\n(gradient energy)")
    alignment_curve(axes[2], curve, "iterative_progress", "M", "20-step learning",
                    "20-step progress")
    axes[2].axhline(1, color=STYLE_COLORS["mute"], lw=LW_REF, ls="--", zorder=0)
    alignment_relation(axes[3], cell)

    handles = [
        Line2D([0], [0], color=METHOD_COLORS[method], marker=METHOD_MARKERS[method],
               lw=LW_DATA, ms=MARKER_MS, mec="white", mew=0.5,
               label=METHOD_LABELS[method])
        for method in METHODS
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.010),
               ncol=4, frameon=False, fontsize=PT_LEGEND,
               handlelength=1.4, columnspacing=1.6, handletextpad=0.45)
    _save(fig, OUTPUT_ALIGNMENT)

    animal_fig = plt.figure(figsize=(FIG_W, 2.62))
    animal_grid = animal_fig.add_gridspec(
        1, 4, left=0.075, right=0.99, bottom=0.165, top=0.855,
        wspace=0.78,
    )
    animal_axes = [animal_fig.add_subplot(animal_grid[0, col]) for col in range(4)]
    animal_schematic(animal_axes[0])
    animal_pairs(animal_axes[1], animal)
    mode_energy(animal_axes[2], summary)
    neuron_distributions(animal_axes[3], neuron)
    _save(animal_fig, OUTPUT_ANIMAL)


if __name__ == "__main__":
    main()

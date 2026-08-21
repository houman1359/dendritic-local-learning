#!/usr/bin/env python3
"""Build the explanatory vector schematics used across main Figures 2--9.

The workshop PNGs are compositional references only.  Every asset produced
here is editable vector art, uses the manuscript's exact notation, and avoids
placing illustrative numbers beside the empirical panels.  The same credit
tree geometry is reused throughout so coordinate, address, gain and shunting
changes read as transformations of one object rather than unrelated cartoons.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from credit_tree_schematics import draw_credit_tree
from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    PT_ANNOT,
    PT_LABEL,
    PT_LEGEND,
    PT_SMALL,
    apply_neurips_style,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

INK = COLORS["ink"]
MUTE = COLORS["mute"]
GRID = COLORS["grid"]
GREEN = COLORS["shunting"]
BLUE = COLORS["additive"]
PURPLE = COLORS["oracle"]
AMBER = COLORS["local"]
ROSE = COLORS["highlight"]
RED = COLORS["bp"]


def _blank(width: float, height: float):
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.01, 0.02, 0.98, 0.94])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _save(fig, stem: str) -> None:
    path = OUT / f"{stem}.pdf"
    fig.savefig(
        path,
        format="pdf",
        facecolor="white",
        metadata={"Creator": "dendritic-local-learning", "CreationDate": None},
    )
    plt.close(fig)


def _box(ax, xy, width, height, *, edge=GRID, fill="white", radius=0.025,
         lw=LW_EDGE, zorder=0.5):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fill,
        edgecolor=edge,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _arrow(ax, start, end, *, color=MUTE, lw=LW_ERR, head=7.0,
           style="-|>", zorder=4):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=head,
        color=color,
        linewidth=lw,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def _tree_inset(ax, rect, *, mode="plain", K=4, shunted=False, scale=0.68):
    sub = ax.inset_axes(rect)
    draw_credit_tree(
        sub,
        mode=mode,
        K=K,
        shunted=shunted,
        labels=False,
        scale=scale,
    )
    return sub


def ownership_address() -> None:
    """Figure 2: neuron identity/ownership versus within-tree address."""
    # A wide banner survives the journal compositor at nearly native scale;
    # the earlier portrait asset was reduced to illegible type in a one-third
    # slot even though its conceptual distinction is central to the paper.
    fig, ax = _blank(7.2, 1.9)
    left_fill = "#F2F8F6"
    right_fill = "#F6F2FB"
    _box(ax, (0.015, 0.08), 0.465, 0.82, edge=GREEN, fill=left_fill)
    _box(ax, (0.52, 0.08), 0.465, 0.82, edge=PURPLE, fill=right_fill)

    ax.text(0.247, 0.84, "ownership", ha="center", va="center",
            fontsize=PT_LABEL, color=GREEN)
    ax.text(0.247, 0.74, r"which neuronal tree receives $\delta_u$?",
            ha="center", va="center", fontsize=PT_SMALL, color=INK)
    _tree_inset(ax, [0.080, 0.23, 0.12, 0.46], mode="coordinate", scale=0.55)
    _tree_inset(ax, [0.297, 0.23, 0.12, 0.46], mode="coordinate", scale=0.55)
    ax.text(0.140, 0.20, "correct tree", ha="center", va="center",
            fontsize=PT_SMALL, color=GREEN)
    ax.text(0.357, 0.20, "deranged tree", ha="center", va="center",
            fontsize=PT_SMALL, color=MUTE)
    ax.plot([0.322, 0.395], [0.32, 0.57], color=RED, lw=LW_DATA,
            solid_capstyle="round", zorder=7)
    ax.plot([0.322, 0.395], [0.57, 0.32], color=RED, lw=LW_DATA,
            solid_capstyle="round", zorder=7)
    ax.text(0.247, 0.115, "same bandwidth; different neuron-to-tree map",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)

    ax.text(0.752, 0.84, "within-tree address", ha="center", va="center",
            fontsize=PT_LABEL, color=PURPLE)
    ax.text(0.752, 0.74, "once the neuron is known, where should credit go?",
            ha="center", va="center", fontsize=PT_SMALL, color=INK)
    _tree_inset(ax, [0.682, 0.20, 0.14, 0.52], mode="address", K=4, scale=0.66)
    ax.text(0.752, 0.115, r"subtree routes refine $\delta_u\rightarrow\delta_{u,k}$",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)
    _save(fig, "schematic_fig2_ownership_address")


def route_resolution() -> None:
    """Figure 3: coarse-to-fine ancestry addresses and comparison routes."""
    fig, ax = _blank(7.2, 2.05)
    positions = [0.02, 0.265, 0.51, 0.755]
    modes = [("coordinate", 1), ("address", 2), ("address", 4), ("address", 8)]
    labels = [
        (r"$K=1$", "one shared coordinate"),
        (r"$K=2$", "two coarse subtrees"),
        (r"$K=4$", "intermediate routes"),
        (r"$K=8$", "full route resolution"),
    ]
    for index, (x0, (mode, budget), (tag, desc)) in enumerate(
        zip(positions, modes, labels, strict=True)
    ):
        tint = "#F7F8FA" if index != 2 else "#F0F8F3"
        edge = GRID if index != 2 else GREEN
        _box(ax, (x0, 0.29), 0.225, 0.61, edge=edge, fill=tint)
        _tree_inset(ax, [x0 + 0.023, 0.43, 0.18, 0.43], mode=mode, K=budget,
                    scale=0.62)
        ax.text(x0 + 0.1125, 0.39, tag, ha="center", va="center",
                fontsize=PT_ANNOT, color=INK)
        ax.text(x0 + 0.1125, 0.315, desc, ha="center", va="center",
                fontsize=PT_SMALL, color=GREEN if index == 2 else MUTE)
    route_key = [
        (GREEN, "o", "ancestry"),
        (ROSE, "s", "deranged"),
        (AMBER, "^", "depth-interleaved"),
        (MUTE, "D", "random sparse"),
        (BLUE, "v", r"random rank-$K$"),
        (PURPLE, "P", r"learned rank-$K$"),
    ]
    handles = [
        Line2D([0], [0], color=color, marker=marker, lw=LW_DATA,
               markersize=4.0, markeredgecolor="white", markeredgewidth=0.35,
               label=label)
        for color, marker, label in route_key
    ]
    ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.025),
        ncol=6,
        frameon=False,
        fontsize=PT_SMALL,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.85,
        borderaxespad=0,
    )
    _save(fig, "schematic_fig3_route_resolution")


def credit_operator() -> None:
    """Figure 4: exact credit enters a route operator and exits filtered."""
    fig, ax = _blank(3.45, 2.65)
    box_y, box_h, box_w = 0.47, 0.28, 0.22
    xs = (0.03, 0.39, 0.75)
    fills = ("#F7F8FA", "#F2F8F6", "#F6F2FB")
    edges = (MUTE, GREEN, PURPLE)
    titles = ("stochastic credit", "route operator", "routed update")
    symbols = (r"$\widehat{\mathbf{g}}$", r"$M$", r"$M\widehat{\mathbf{g}}$")
    for x, fill, edge, title, symbol in zip(xs, fills, edges, titles, symbols,
                                             strict=True):
        _box(ax, (x, box_y), box_w, box_h, edge=edge, fill=fill)
        ax.text(x + box_w / 2, box_y + 0.205, title, ha="center", va="center",
                fontsize=PT_SMALL, color=INK)
        ax.text(x + box_w / 2, box_y + 0.095, symbol, ha="center", va="center",
                fontsize=PT_LABEL, color=edge)
    _arrow(ax, (0.26, 0.61), (0.37, 0.61), color=MUTE)
    _arrow(ax, (0.62, 0.61), (0.73, 0.61), color=MUTE)

    dot_x = np.linspace(0.785, 0.935, 6)
    for x, color in zip(dot_x, (GREEN, GREEN, GREEN, AMBER, MUTE, MUTE), strict=True):
        ax.plot([x], [0.51], marker="o", ms=4.2, mfc=color, mec="none", zorder=6)
    ax.text(0.50, 0.38, "retained task signal", ha="center", va="center",
            fontsize=PT_SMALL, color=GREEN)
    ax.text(0.50, 0.29, "versus", ha="center", va="center",
            fontsize=PT_SMALL, color=MUTE)
    ax.text(0.50, 0.20, "representation error + admitted noise",
            ha="center", va="center", fontsize=PT_SMALL, color=AMBER)
    ax.text(
        0.50,
        0.075,
        r"$U(M)=\dfrac{[\mathbf{g}^{\mathsf{T}}M\mathbf{g}]^2}"
        r"{2L\{\|M\mathbf{g}\|^2+\mathrm{tr}(M\Sigma M^{\mathsf{T}})\}}$",
        ha="center",
        va="center",
        fontsize=PT_SMALL,
        color=INK,
    )
    _save(fig, "schematic_fig4_credit_operator")


def _stage_tree(ax, x0, y0, depth, *, color=GREEN, scale=1.0):
    """Small balanced serial tree used only as a physical-stage icon."""
    ax.add_patch(Circle((x0, y0), 0.018 * scale, fc=COLORS["soma"], ec=INK,
                        lw=LW_EDGE, zorder=4))
    current = [(x0, y0 + 0.02 * scale)]
    span = 0.115 * scale
    step = 0.105 * scale
    for level in range(depth):
        nxt = []
        new_span = span / (level + 1.15)
        for px, py in current:
            for sign in (-1, 1):
                qx, qy = px + sign * new_span, py + step
                ax.plot([px, qx], [py, qy], color=color,
                        lw=max(LW_EDGE, LW_DATA - 0.12 * level),
                        solid_capstyle="round", zorder=2)
                nxt.append((qx, qy))
        current = nxt
        span *= 0.55
    for px, py in current:
        ax.plot([px], [py], marker="o", ms=2.2, mfc=color, mec="none", zorder=3)


def physical_depth() -> None:
    """Figure 5: matched resources, ordered task factors and point control."""
    fig, ax = _blank(7.2, 2.05)
    _box(ax, (0.015, 0.12), 0.31, 0.78, edge=GREEN, fill="#F2F8F6")
    _box(ax, (0.345, 0.12), 0.31, 0.78, edge=PURPLE, fill="#F6F2FB")
    _box(ax, (0.675, 0.12), 0.31, 0.78, edge=MUTE, fill="#F7F8FA")

    ax.text(0.17, 0.83, "physical stage count", ha="center", va="center",
            fontsize=PT_LABEL, color=GREEN)
    for index, depth in enumerate((1, 2, 3)):
        x = 0.075 + 0.095 * index
        _stage_tree(ax, x, 0.30, depth, color=GREEN, scale=0.55)
        ax.text(x, 0.20, rf"$D_{{\rm p}}={depth}$", ha="center", va="center",
                fontsize=PT_SMALL, color=INK)
    ax.text(0.17, 0.135, "same forward budget",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)

    ax.text(0.50, 0.83, "multiplicative task hierarchy", ha="center",
            va="center", fontsize=PT_LABEL, color=PURPLE)
    colors = (GREEN, BLUE, PURPLE)
    ys = (0.63, 0.47, 0.31)
    labels = ("coarse factor", "intermediate factor", "fine factor")
    for idx, (y, color, label) in enumerate(zip(ys, colors, labels, strict=True), 1):
        _box(ax, (0.405, y - 0.05), 0.19, 0.09, edge=color, fill="white",
             radius=0.012)
        ax.text(0.50, y, rf"$G_{idx}$  {label}", ha="center", va="center",
                fontsize=PT_SMALL, color=color)
        if idx < 3:
            _arrow(ax, (0.50, y - 0.055), (0.50, ys[idx] + 0.055),
                   color=MUTE, head=5.5)
    ax.text(0.50, 0.18, r"$y^*=s\,G_1G_2G_3$", ha="center", va="center",
            fontsize=PT_ANNOT, color=INK)

    ax.text(0.83, 0.83, "representation-matched control", ha="center",
            va="center", fontsize=PT_LABEL, color=MUTE)
    centers = [(0.75, 0.55), (0.83, 0.55), (0.91, 0.55)]
    for index, (x, y) in enumerate(centers):
        ax.add_patch(Circle((x, y), 0.040, fc="white", ec=MUTE, lw=LW_EDGE))
        ax.text(x, y, rf"$z_{index+1}$", ha="center", va="center",
                fontsize=PT_SMALL, color=INK)
    for x, y in centers:
        ax.plot([x, 0.83], [y - 0.04, 0.32], color=MUTE, lw=LW_EDGE)
    ax.add_patch(Circle((0.83, 0.29), 0.045, fc="#F7F8FA", ec=MUTE, lw=LW_EDGE))
    ax.text(0.83, 0.29, r"$\sigma$", ha="center", va="center",
            fontsize=PT_ANNOT, color=INK)
    ax.text(0.83, 0.17, "same routed fields",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)
    _save(fig, "schematic_fig5_physical_depth")


def physical_generalization() -> None:
    """Figure 6: second-depth and task-family boundary tests."""
    fig, ax = _blank(7.2, 1.8)
    _box(ax, (0.015, 0.12), 0.46, 0.77, edge=GREEN, fill="#F2F8F6")
    _box(ax, (0.525, 0.12), 0.46, 0.77, edge=PURPLE, fill="#F6F2FB")
    ax.text(0.245, 0.82, "depth saturation", ha="center", va="center",
            fontsize=PT_LABEL, color=GREEN)
    for index, depth in enumerate((2, 3, 4)):
        x = 0.105 + 0.14 * index
        _stage_tree(ax, x, 0.29, min(depth, 3), color=GREEN, scale=0.54)
        ax.text(x, 0.20, rf"$D_{{\rm p}}={depth}$", ha="center", va="center",
                fontsize=PT_SMALL, color=INK)
    ax.text(0.245, 0.68, r"second hierarchy: $H=4$", ha="center", va="center",
            fontsize=PT_ANNOT, color=INK)
    ax.text(0.245, 0.135, "test whether benefit saturates beyond the matched order",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)

    ax.text(0.755, 0.82, "task-family × alignment boundary", ha="center",
            va="center", fontsize=PT_LABEL, color=PURPLE)
    ax.text(0.60, 0.65, "nested factors", ha="center", va="center",
            fontsize=PT_SMALL, color=GREEN)
    ax.text(0.755, 0.65, "flat factors", ha="center", va="center",
            fontsize=PT_SMALL, color=BLUE)
    ax.text(0.91, 0.65, "local ratios", ha="center", va="center",
            fontsize=PT_SMALL, color=MUTE)
    for x, color, profile in ((0.60, GREEN, (0.20, 0.27, 0.42)),
                              (0.755, BLUE, (0.25, 0.31, 0.32)),
                              (0.91, MUTE, (0.37, 0.31, 0.25))):
        ax.plot([x - 0.045, x, x + 0.045], profile, color=color, lw=LW_DATA,
                marker="o", ms=2.6, mfc=color, mec="none")
    ax.text(0.755, 0.19, r"sensor alignment $\alpha: 0\rightarrow1$",
            ha="center", va="center", fontsize=PT_ANNOT, color=INK)
    _arrow(ax, (0.60, 0.13), (0.91, 0.13), color=MUTE, head=6.0)
    _save(fig, "schematic_fig6_generalization")


def anatomy_pipeline() -> None:
    """Figure 7: reconstruction to route dictionary to wiring economics."""
    fig, ax = _blank(7.2, 1.75)
    xs = (0.02, 0.35, 0.68)
    widths = (0.28, 0.28, 0.30)
    for x, w in zip(xs, widths, strict=True):
        _box(ax, (x, 0.14), w, 0.74, edge=GRID, fill="#FAFAFB")
    ax.text(0.16, 0.81, "reconstructed arbor", ha="center", va="center",
            fontsize=PT_LABEL, color=INK)
    _tree_inset(ax, [0.06, 0.25, 0.20, 0.50], mode="plain", scale=0.64)
    ax.text(0.16, 0.19, "mapped E/I contacts",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)

    ax.text(0.49, 0.81, "ancestry route dictionary", ha="center", va="center",
            fontsize=PT_LABEL, color=GREEN)
    _tree_inset(ax, [0.39, 0.25, 0.20, 0.50], mode="address", K=4, scale=0.64)
    ax.text(0.49, 0.19, "nested subtrees define addresses",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)

    ax.text(0.83, 0.81, "capacity per wire", ha="center", va="center",
            fontsize=PT_LABEL, color=PURPLE)
    for i, (label, frac, color) in enumerate(
        (("dense", 0.94, PURPLE), ("ancestry", 0.64, GREEN),
         ("control", 0.37, MUTE))
    ):
        y = 0.62 - 0.16 * i
        ax.text(0.715, y, label, ha="left", va="center",
                fontsize=PT_SMALL, color=color)
        ax.plot([0.79, 0.79 + 0.15 * frac], [y, y], color=color, lw=3.0,
                solid_capstyle="round")
    ax.text(0.83, 0.19, "model- and density-matched gains",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)
    _arrow(ax, (0.305, 0.51), (0.34, 0.51), color=MUTE)
    _arrow(ax, (0.635, 0.51), (0.67, 0.51), color=MUTE)
    _save(fig, "schematic_fig7_anatomy_pipeline")


def focal_shunt() -> None:
    """Figure 8: a local conductance edit changes a descendant adjoint field."""
    fig, ax = _blank(5.2, 2.65)
    _box(ax, (0.015, 0.08), 0.43, 0.82, edge=GRID, fill="#FAFAFB")
    _box(ax, (0.555, 0.08), 0.43, 0.82, edge=GREEN, fill="#F2F8F6")
    ax.text(0.23, 0.83, "matched additive control", ha="center", va="center",
            fontsize=PT_LABEL, color=BLUE)
    ax.text(0.77, 0.83, "focal shunting conductance", ha="center", va="center",
            fontsize=PT_LABEL, color=GREEN)
    _tree_inset(ax, [0.075, 0.22, 0.30, 0.52], mode="shunt", shunted=False,
                scale=0.70)
    _tree_inset(ax, [0.615, 0.22, 0.30, 0.52], mode="shunt", shunted=True,
                scale=0.70)
    _arrow(ax, (0.455, 0.51), (0.545, 0.51), color=MUTE)
    ax.text(0.23, 0.17, "same local operating point", ha="center", va="center",
            fontsize=PT_SMALL, color=MUTE)
    ax.text(0.77, 0.17, "descendant transport changes selectively",
            ha="center", va="center", fontsize=PT_SMALL, color=GREEN)
    ax.text(0.50, 0.035,
            r"conductance edits $G$ and therefore $q=G^{-T}\nabla_V\mathcal{L}$",
            ha="center", va="center", fontsize=PT_SMALL, color=INK)
    _save(fig, "schematic_fig8_focal_shunt")


def alignment_boundary() -> None:
    """Figure 9: availability, imposed alignment and external evidence."""
    fig, ax = _blank(7.2, 1.8)
    entries = [
        (0.02, "measured responses", "no morphology-specific alignment", "plain", MUTE),
        (0.35, "imposed alignment", "same dictionary, task field rotated", "address", GREEN),
        (0.68, "learning and animal tests", "sufficiency versus endogenous use", "gain", PURPLE),
    ]
    for x, title, subtitle, mode, color in entries:
        _box(ax, (x, 0.13), 0.30, 0.75, edge=color, fill="#FAFAFB")
        ax.text(x + 0.15, 0.81, title, ha="center", va="center",
                fontsize=PT_LABEL, color=color)
        _tree_inset(ax, [x + 0.065, 0.29, 0.17, 0.43], mode=mode, K=4, scale=0.58)
        ax.text(x + 0.15, 0.18, subtitle, ha="center", va="center",
                fontsize=PT_SMALL, color=MUTE)
    _arrow(ax, (0.325, 0.51), (0.345, 0.51), color=MUTE, head=5.5)
    _arrow(ax, (0.655, 0.51), (0.675, 0.51), color=MUTE, head=5.5)
    ax.text(0.50, 0.045,
            "availability  →  controlled sufficiency  →  evidence for endogenous use",
            ha="center", va="center", fontsize=PT_SMALL, color=INK)
    _save(fig, "schematic_fig9_alignment_boundary")


def main() -> None:
    apply_neurips_style()
    ownership_address()
    route_resolution()
    credit_operator()
    physical_depth()
    physical_generalization()
    anatomy_pipeline()
    focal_shunt()
    alignment_boundary()
    print("Built eight main-figure schematic assets.")


if __name__ == "__main__":
    main()

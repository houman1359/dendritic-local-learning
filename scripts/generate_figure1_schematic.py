#!/usr/bin/env python3
"""Generate Figure 1: model and credit-assignment schematic.

The figure is intentionally schematic. It emphasizes four ideas:
network units are dendritic neurons, conductance trees carry E/I banks,
exact errors are path-specific transports from the soma, and LocalCA
replaces only that non-local error field.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from neurips_style import COLORS, apply_neurips_style, panel_label


ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "figures"


def ax_setup(ax, title: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.115,
        0.955,
        title,
        ha="left",
        va="top",
        fontsize=9.0,
        fontweight="bold",
        color=COLORS["ink"],
    )


def arrow(
    ax,
    start,
    end,
    *,
    color=COLORS["edge"],
    lw=1.5,
    alpha=1.0,
    style="-|>",
    mutation_scale=9,
    ls="-",
    zorder=3,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            alpha=alpha,
            linestyle=ls,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=zorder,
        )
    )


def box(
    ax,
    xy,
    w,
    h,
    text,
    *,
    fc="white",
    ec=COLORS["edge"],
    color=COLORS["ink"],
    fontsize=7.1,
    weight="normal",
    radius=0.035,
    lw=1.0,
    zorder=4,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        zorder=zorder + 1,
    )
    return patch


def draw_tree_icon(
    ax,
    cx,
    cy,
    *,
    scale=1.0,
    soma_label=None,
    leaf_labels=False,
    credit_colors=False,
    trunk_lw=2.0,
    zorder=3,
):
    """Draw a compact soma-rooted binary tree and return key node positions."""
    soma = (cx + 0.20 * scale, cy)
    prox_top = (cx + 0.05 * scale, cy + 0.10 * scale)
    prox_bot = (cx + 0.05 * scale, cy - 0.10 * scale)
    leaves = [
        (cx - 0.15 * scale, cy + 0.18 * scale),
        (cx - 0.15 * scale, cy + 0.06 * scale),
        (cx - 0.15 * scale, cy - 0.06 * scale),
        (cx - 0.15 * scale, cy - 0.18 * scale),
    ]
    edges = [
        (leaves[0], prox_top),
        (leaves[1], prox_top),
        (leaves[2], prox_bot),
        (leaves[3], prox_bot),
        (prox_top, soma),
        (prox_bot, soma),
    ]
    edge_cols = (
        [COLORS["bp"], COLORS["bp"], COLORS["oracle"], COLORS["local"], COLORS["bp"], COLORS["oracle"]]
        if credit_colors
        else [COLORS["dend"]] * len(edges)
    )
    for (a, b), col in zip(edges, edge_cols):
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=col,
            lw=trunk_lw,
            solid_capstyle="round",
            zorder=zorder,
        )
    branch_nodes = [prox_top, prox_bot, *leaves]
    ax.scatter(
        [p[0] for p in branch_nodes],
        [p[1] for p in branch_nodes],
        s=16 * scale,
        color="white",
        edgecolor=COLORS["dend"],
        linewidth=1.0,
        zorder=zorder + 1,
    )
    ax.scatter(
        [soma[0]],
        [soma[1]],
        s=62 * scale,
        color=COLORS["soma"],
        edgecolor="white",
        linewidth=0.8,
        zorder=zorder + 2,
    )
    if soma_label:
        ax.text(
            soma[0] + 0.035 * scale,
            soma[1],
            soma_label,
            ha="left",
            va="center",
            fontsize=6.8,
            color=COLORS["ink"],
            zorder=zorder + 3,
        )
    if leaf_labels:
        for idx, p in enumerate(leaves[:3], start=1):
            ax.text(
                p[0] - 0.028 * scale,
                p[1],
                rf"$n_{idx}$",
                ha="right",
                va="center",
                fontsize=6.5,
                color=COLORS["mute"],
                zorder=zorder + 3,
            )
    return {"soma": soma, "prox": [prox_top, prox_bot], "leaves": leaves}


def panel_network(ax) -> None:
    ax_setup(ax, "Network layer: dendritic E units")
    panel_label(ax, "A", x=0.018, y=0.985, fontsize=11.8)

    box(
        ax,
        (0.04, 0.40),
        0.14,
        0.20,
        "$x\\geq0$\ninput",
        fc="#EDF4FB",
        ec=COLORS["exc"],
        fontsize=7.5,
        weight="bold",
    )
    box(
        ax,
        (0.82, 0.40),
        0.13,
        0.20,
        "$\\hat y$\nreadout",
        fc="#F5F0F8",
        ec=COLORS["oracle"],
        fontsize=7.5,
        weight="bold",
    )

    unit_ys = [0.75, 0.58, 0.41, 0.24]
    soma_positions = []
    for i, y in enumerate(unit_ys, start=1):
        key = draw_tree_icon(ax, 0.46, y, scale=0.62, soma_label=rf"$V_{i}$", trunk_lw=1.45)
        soma_positions.append(key["soma"])
        arrow(ax, (0.18, 0.50), (0.31, y), color=COLORS["exc"], lw=0.9, alpha=0.58, mutation_scale=6)
        arrow(ax, key["soma"], (0.82, 0.50), color=COLORS["edge"], lw=0.9, alpha=0.42, mutation_scale=6)

    box(
        ax,
        (0.31, 0.035),
        0.40,
        0.10,
        "each hidden unit contains\nbranch compartments + E/I banks",
        fc=COLORS["panel_bg"],
        ec="#BFC7D1",
        fontsize=6.7,
    )


def panel_conductance_tree(ax) -> None:
    ax_setup(ax, "Single conductance tree")
    panel_label(ax, "B", x=0.018, y=0.985, fontsize=11.8)

    key = draw_tree_icon(ax, 0.43, 0.49, scale=1.36, soma_label=None, leaf_labels=True, trunk_lw=2.4)
    leaves = key["leaves"]
    prox_top, prox_bot = key["prox"]
    soma = key["soma"]

    for idx, p in enumerate([leaves[0], leaves[2], prox_top]):
        ax.scatter(
            [p[0] - 0.035],
            [p[1] + 0.022],
            s=44,
            marker="o",
            color=COLORS["exc"],
            edgecolor="white",
            linewidth=0.7,
            zorder=8,
        )
        ax.text(
            p[0] - 0.067,
            p[1] + 0.052,
            "E" if idx == 0 else "",
            fontsize=6.5,
            color=COLORS["exc"],
            fontweight="bold",
            ha="center",
            va="center",
        )

    for idx, p in enumerate([leaves[1], prox_bot]):
        ax.scatter(
            [p[0] - 0.035],
            [p[1] - 0.022],
            s=54,
            marker="v",
            color=COLORS["inh"],
            edgecolor="white",
            linewidth=0.7,
            zorder=8,
        )
        ax.text(
            p[0] - 0.067,
            p[1] - 0.058,
            "I" if idx == 0 else "",
            fontsize=6.5,
            color=COLORS["inh"],
            fontweight="bold",
            ha="center",
            va="center",
        )

    arrow(ax, soma, (0.86, 0.49), color=COLORS["edge"], lw=1.2, mutation_scale=8)
    ax.text(soma[0] + 0.030, soma[1] + 0.082, "soma", fontsize=7.2, va="bottom", ha="left", color=COLORS["ink"])
    ax.text(0.875, 0.49, "$V_0$", fontsize=7.4, va="center", ha="left", color=COLORS["ink"])

    box(ax, (0.72, 0.69), 0.22, 0.09, "E pulls\n$V\\to1$", fc="#EDF4FB", ec=COLORS["exc"], fontsize=6.6)
    box(ax, (0.72, 0.56), 0.22, 0.09, "I shunts\n$V\\to0$", fc="#FBEDEE", ec=COLORS["inh"], fontsize=6.6)
    box(
        ax,
        (0.70, 0.20),
        0.26,
        0.12,
        "$R_n^{\\mathrm{tot}}=1/g_n^{\\mathrm{tot}}$\nsets local gain",
        fc=COLORS["panel_bg"],
        ec="#BFC7D1",
        fontsize=6.45,
    )


def panel_credit(ax) -> None:
    ax_setup(ax, "Path-specific vs. shared credit")
    panel_label(ax, "C", x=0.018, y=0.985, fontsize=11.8)

    key = draw_tree_icon(ax, 0.35, 0.49, scale=1.35, leaf_labels=True, credit_colors=True, trunk_lw=2.4)
    leaves = key["leaves"]
    soma = key["soma"]

    box(
        ax,
        (0.77, 0.47),
        0.13,
        0.10,
        "$\\delta_0$",
        fc="#F6ECE8",
        ec=COLORS["bp"],
        color=COLORS["bp"],
        fontsize=8.4,
        weight="bold",
    )
    arrow(ax, soma, (0.77, 0.52), color=COLORS["bp"], lw=1.3, mutation_scale=8)

    exact_targets = [leaves[0], leaves[1], leaves[2]]
    exact_labels = [r"$\alpha_1\delta_0$", r"$\alpha_2\delta_0$", r"$\alpha_3\delta_0$"]
    y_offsets = [0.08, 0.015, -0.05]
    label_offsets = [0.025, -0.030, 0.030]
    for target, label, dy, label_dy in zip(exact_targets, exact_labels, y_offsets, label_offsets):
        start = (0.77, 0.52 + dy)
        end = (target[0] + 0.012, target[1])
        arrow(ax, start, end, color=COLORS["bp"], lw=1.05, ls="--", alpha=0.82, mutation_scale=7)
        mx = start[0] * 0.47 + end[0] * 0.53
        my = start[1] * 0.47 + end[1] * 0.53
        ax.text(mx, my + label_dy, label, fontsize=6.35, color=COLORS["bp"], ha="center", va="center")

    # Rank-1 LocalCA comes from the same somatic side but carries one shared field.
    bus_x = 0.66
    ax.plot([bus_x, bus_x], [0.25, 0.72], color=COLORS["local"], lw=2.0, alpha=0.95, zorder=4)
    ax.text(bus_x + 0.025, 0.74, "shared $e$", fontsize=6.7, color=COLORS["local"], ha="left", va="bottom")
    for target in [leaves[0], leaves[1], leaves[2], leaves[3]]:
        arrow(
            ax,
            (bus_x, target[1]),
            (target[0] + 0.02, target[1]),
            color=COLORS["local"],
            lw=1.05,
            alpha=0.88,
            mutation_scale=7,
        )

    box(
        ax,
        (0.06, 0.058),
        0.47,
        0.10,
        "exact: path-specific errors\nLocalCA: shared low-bandwidth field",
        fc=COLORS["panel_bg"],
        ec="#BFC7D1",
        fontsize=6.45,
    )


def feedback_card(ax, x, y, w, h, title, formula, color, subtitle=None) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.035",
        facecolor="white",
        edgecolor=color,
        linewidth=1.15,
        zorder=5,
    )
    ax.add_patch(patch)
    ax.text(x + 0.03 * w, y + h - 0.18 * h, title, ha="left", va="center", fontsize=6.6, fontweight="bold", color=color, zorder=6)
    ax.text(x + w / 2, y + 0.47 * h, formula, ha="center", va="center", fontsize=6.2, color=COLORS["ink"], zorder=6)
    if subtitle:
        ax.text(x + w / 2, y + 0.17 * h, subtitle, ha="center", va="center", fontsize=5.55, color=COLORS["mute"], zorder=6)


def panel_broadcast_modes(ax) -> None:
    ax_setup(ax, "LocalCA substitutes only error")
    panel_label(ax, "D", x=0.018, y=0.985, fontsize=11.8)

    box(
        ax,
        (0.06, 0.28),
        0.48,
        0.43,
        "exact local eligibility\n\n"
        "$\\frac{\\partial L}{\\partial g_i}=x_iR_n^{\\mathrm{tot}}(E_i-V_n)\\,\\delta_n$"
        "\n\n"
        "LocalCA update\n"
        "$\\Delta g_i\\propto x_iR_n^{\\mathrm{tot}}(E_i-V_n)\\,e_n$",
        fc="#FBFCFD",
        ec="#BFC7D1",
        fontsize=6.65,
        weight="bold",
        radius=0.03,
    )

    ax.text(0.61, 0.735, "feedback object", ha="left", va="bottom", fontsize=6.9, fontweight="bold", color=COLORS["mute"])
    rows = [
        ("rank-1", r"$e_n=\bar\delta\,\mathbf{1}$", COLORS["scalar"]),
        ("neuron", r"$e_n=\delta_0$", COLORS["per_soma"]),
        ("rank-$K$", r"$e_n=Q_nP_K\delta_0$", COLORS["low_rank"]),
        ("path", r"$e_n=\Gamma_n(\delta_0)$", COLORS["pathway"]),
        ("oracle", r"$e_n=\tilde\alpha_n\delta_0$", COLORS["oracle"]),
    ]
    y0 = 0.65
    for i, (name, formula, color) in enumerate(rows):
        y = y0 - i * 0.088
        box(ax, (0.60, y), 0.34, 0.062, f"{name}   {formula}", fc="white", ec=color, color=color, fontsize=6.35, weight="bold", radius=0.025, lw=1.0)

    ax.text(
        0.50,
        0.115,
        "same synapse-local eligibility; different non-local feedback bandwidth",
        ha="center",
        va="center",
        fontsize=6.7,
        color=COLORS["mute"],
    )


def main() -> None:
    apply_neurips_style()
    fig = plt.figure(figsize=(7.0, 4.35))
    gs = fig.add_gridspec(
        2,
        2,
        left=0.02,
        right=0.99,
        top=0.98,
        bottom=0.04,
        wspace=0.07,
        hspace=0.17,
    )

    panel_network(fig.add_subplot(gs[0, 0]))
    panel_conductance_tree(fig.add_subplot(gs[0, 1]))
    panel_credit(fig.add_subplot(gs[1, 0]))
    panel_broadcast_modes(fig.add_subplot(gs[1, 1]))

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"fig1_model_and_credit.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()

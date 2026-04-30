#!/usr/bin/env python3
"""Generate Figure 1 for the LocalCA manuscript.

The figure is deliberately conceptual: it defines the conductance tree, the
path-gain object, the exact credit factorization, and the feedback taxonomy
used throughout the paper.

Outputs:
  figures/fig1_model_and_credit.{pdf,png}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.patheffects import withStroke
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import COLORS, apply_neurips_style, panel_label

apply_neurips_style()


OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"

EXC = COLORS["exc"]
INH = COLORS["inh"]
DEND = COLORS["dend"]
SOMA = COLORS["soma"]
INK = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]
EXACT = COLORS["bp"]
LOCAL = COLORS["local"]
ORACLE = COLORS["oracle"]
LOW_RANK = COLORS["low_rank"]
PATHWAY = COLORS["pathway"]


def setup_panel(ax, width: float = 1.0, title: str | None = None) -> None:
    ax.set_xlim(0, width)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", x=0.10, fontsize=10.2, pad=5, fontweight="bold")


def box(ax, xy, w, h, fc="white", ec="#D8DEE8", lw=0.8, radius=0.02, z=2):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    xy1,
    xy2,
    color=INK,
    lw=1.2,
    ms=9,
    alpha=1.0,
    ls="-",
    rad=0.0,
    z=6,
):
    ax.add_patch(
        FancyArrowPatch(
            xy1,
            xy2,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            alpha=alpha,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=0,
            shrinkB=0,
            zorder=z,
        )
    )


def text_with_halo(ax, x, y, text, **kwargs):
    kwargs.setdefault("fontsize", 7)
    kwargs.setdefault("ha", "center")
    kwargs.setdefault("va", "center")
    kwargs.setdefault("color", "white")
    kwargs.setdefault("fontweight", "bold")
    t = ax.text(x, y, text, zorder=12, **kwargs)
    t.set_path_effects([withStroke(linewidth=1.2, foreground="black", alpha=0.25)])
    return t


def draw_synapse(ax, x, y, kind="E", scale=1.0):
    if kind == "E":
        ax.add_patch(
            Circle((x, y), 0.012 * scale, fc=EXC, ec="white", linewidth=0.55, zorder=9)
        )
    else:
        ax.add_patch(
            mpatches.RegularPolygon(
                (x, y),
                numVertices=3,
                radius=0.018 * scale,
                orientation=np.pi,
                fc=INH,
                ec="white",
                linewidth=0.55,
                zorder=9,
            )
        )


def draw_tree(
    ax,
    *,
    width: float,
    x_leaf: float,
    x_branch: float,
    x_soma: float,
    branch_ys=(0.78, 0.50, 0.22),
    leaf_offsets=(0.075, 0.0, -0.075),
    leaf_color=DEND,
    branch_colors=None,
    show_synapses=True,
    labels=True,
):
    branch_ys = np.asarray(branch_ys)
    leaf_offsets = np.asarray(leaf_offsets)
    branch_colors = branch_colors or [DEND] * len(branch_ys)
    leaves = []
    for bi, by in enumerate(branch_ys):
        for off in leaf_offsets:
            ly = float(by + off)
            leaves.append((x_leaf, ly, bi))
            ax.plot(
                [x_leaf + 0.018, x_branch - 0.025],
                [ly, by + 0.25 * off],
                color=branch_colors[bi],
                linewidth=1.35,
                alpha=0.85,
                solid_capstyle="round",
                zorder=2,
            )
        ax.plot(
            [x_branch + 0.025, x_soma - 0.04],
            [by, 0.50 + 0.16 * (by - 0.50)],
            color=branch_colors[bi],
            linewidth=2.15,
            alpha=0.90,
            solid_capstyle="round",
            zorder=2,
        )

    for x, y, _bi in leaves:
        ax.add_patch(
            Circle((x, y), 0.020, fc=leaf_color, ec=EDGE, linewidth=0.65, zorder=4)
        )
        if show_synapses:
            draw_synapse(ax, x - 0.038, y - 0.008, "E", 1.08)
            draw_synapse(ax, x - 0.038, y + 0.012, "E", 1.08)
            draw_synapse(ax, x + 0.008, y + 0.038, "I", 1.08)

    for bi, by in enumerate(branch_ys):
        ax.add_patch(
            Circle((x_branch, by), 0.028, fc=branch_colors[bi], ec=EDGE, linewidth=0.7, zorder=5)
        )

    ax.add_patch(Circle((x_soma, 0.50), 0.040, fc=SOMA, ec=EDGE, linewidth=0.9, zorder=6))
    if labels:
        text_with_halo(ax, x_leaf, branch_ys[1], r"$V_n$", fontsize=7.3)
        text_with_halo(ax, x_branch, branch_ys[1], r"$V_p$", fontsize=7.3)
        text_with_halo(ax, x_soma, 0.50, "soma", fontsize=6.5)
    return {"branches": list(zip([x_branch] * len(branch_ys), branch_ys)), "leaves": leaves, "soma": (x_soma, 0.50)}


def panel_a(ax):
    width = 1.32
    setup_panel(ax, width, "Conductance tree")
    draw_tree(
        ax,
        width=width,
        x_leaf=0.43,
        x_branch=0.71,
        x_soma=1.03,
        branch_ys=(0.73, 0.52, 0.31),
        labels=False,
    )

    arrow(ax, (1.07, 0.50), (1.22, 0.50), lw=1.2, ms=9)
    ax.text(1.24, 0.50, "out", ha="left", va="center", fontsize=7.6, color=INK, fontweight="bold")

    ax.text(0.320, 0.220, r"$V_n$", fontsize=8.8, color=INK, ha="center", fontweight="bold")
    ax.plot([0.355, 0.43], [0.245, 0.300], color=MUTE, lw=0.7)
    ax.text(0.665, 0.205, r"$V_p$", fontsize=8.8, color=INK, ha="center", fontweight="bold")
    ax.plot([0.685, 0.71], [0.235, 0.310], color=MUTE, lw=0.7)
    ax.text(1.055, 0.385, "soma", fontsize=7.8, color=INK, ha="center", fontweight="bold")
    ax.plot([1.045, 1.03], [0.412, 0.455], color=MUTE, lw=0.7)

    component_boxes = [
        (0.14, 0.100, 0.23, "E drive", EXC),
        (0.43, 0.100, 0.23, "I shunt", INH),
        (0.73, 0.100, 0.23, "branch", DEND),
        (1.04, 0.100, 0.20, "soma", SOMA),
    ]
    for x, y, w, label, color in component_boxes:
        box(ax, (x - w / 2, y - 0.038), w, 0.076, fc="white", ec="#D8DEE8", lw=0.55)
        ax.add_patch(Circle((x - w / 2 + 0.028, y), 0.014, fc=color, ec="white", linewidth=0.35, zorder=8))
        ax.text(
            x - w / 2 + 0.055,
            y,
            label,
            fontsize=6.6,
            ha="left",
            va="center",
            color=INK,
            linespacing=0.9,
        )


def panel_b(ax):
    width = 1.58
    setup_panel(ax, width, "Exact factorization")

    ax.text(0.79, 0.81, r"$\frac{\partial L}{\partial g_i^{\rm syn}}=$", ha="center", va="center", fontsize=15, color=INK)
    box(ax, (0.13, 0.48), 0.72, 0.20, fc="#EAF3EF", ec=COLORS["shunting"])
    box(ax, (0.92, 0.48), 0.44, 0.20, fc="#F4EEF7", ec=ORACLE)
    ax.text(0.49, 0.59, r"$x_i\,R_n^{\rm tot}\,(E_i-V_n)$", ha="center", va="center", fontsize=12, color=INK)
    ax.text(1.14, 0.59, r"$\delta_n$", ha="center", va="center", fontsize=15, color=INK)
    ax.text(0.49, 0.49, "local eligibility", ha="center", va="top", fontsize=6.2, color=COLORS["shunting"], fontweight="bold")
    ax.text(1.14, 0.49, "error signal", ha="center", va="top", fontsize=6.2, color=ORACLE, fontweight="bold")

    box(ax, (0.10, 0.22), 0.59, 0.12, fc="#FFF7ED", ec="#E7C8A5")
    box(ax, (0.89, 0.22), 0.59, 0.12, fc="#F8FAFC", ec="#CBD5E1")
    ax.text(0.395, 0.28, r"Exact: $\delta_n=\partial L/\partial V_n$", fontsize=8.2, ha="center", va="center", color=EXACT, fontweight="bold")
    ax.text(1.185, 0.28, r"LocalCA: $\delta_n\approx e_n$", fontsize=8.2, ha="center", va="center", color=LOCAL, fontweight="bold")

    ax.text(
        0.79,
        0.075,
        "The synapse-local term is exact; the broadcast approximation is entirely in $e_n$.",
        ha="center",
        va="center",
        fontsize=7.0,
        color=MUTE,
        style="italic",
    )


def panel_rules(ax):
    width = 1.25
    setup_panel(ax, width, "LocalCA rule family")
    cards = [
        ("3F", COLORS["rule_3f"], r"$x_iR_n^{\rm tot}(E_i-V_n)e_n$", "exact eligibility"),
        ("4F", COLORS["rule_4f"], r"$\mathrm{3F}\cdot \rho_n$", "morphology scale"),
        ("5F", COLORS["rule_5f"], r"$\mathrm{4F}\cdot \phi_n$", "confidence scale"),
    ]
    yvals = [0.74, 0.50, 0.26]
    for (name, color, eq, note), y in zip(cards, yvals):
        box(ax, (0.06, y - 0.085), 1.08, 0.17, fc="white", ec="#D8DEE8", lw=0.7)
        ax.add_patch(Circle((0.17, y), 0.052, fc=color, ec="white", linewidth=0.5, zorder=7))
        ax.text(0.17, y, name, ha="center", va="center", fontsize=8.2, color="white", fontweight="bold", zorder=8)
        ax.text(0.27, y + 0.030, eq, ha="left", va="center", fontsize=7.2, color=INK)
        ax.text(0.27, y - 0.038, note, ha="left", va="center", fontsize=5.8, color=MUTE, style="italic")
    ax.text(
        0.60,
        0.065,
        r"$e_n$ is the only non-local signal; all other factors are local.",
        ha="center",
        va="center",
        fontsize=5.9,
        color=MUTE,
        style="italic",
    )


def panel_design(ax):
    width = 2.05
    setup_panel(ax, width, "Rules and feedback")

    ax.text(0.08, 0.87, "Local update", fontsize=6.8, color=MUTE, fontweight="bold")
    cards = [
        ("3F", COLORS["rule_3f"], r"$x_iR_n^{\rm tot}(E_i-V_n)e_n$", "exact eligibility"),
        ("4F", COLORS["rule_4f"], r"$\mathrm{3F}\cdot\rho_n$", "morphology scale"),
        ("5F", COLORS["rule_5f"], r"$\mathrm{4F}\cdot\phi_n$", "confidence scale"),
    ]
    yvals = [0.72, 0.52, 0.32]
    for (name, color, eq, note), y in zip(cards, yvals):
        box(ax, (0.08, y - 0.060), 0.82, 0.12, fc="white", ec="#D8DEE8", lw=0.65)
        ax.add_patch(Circle((0.16, y), 0.038, fc=color, ec="white", linewidth=0.45, zorder=7))
        ax.text(0.16, y, name, ha="center", va="center", fontsize=6.8, color="white", fontweight="bold", zorder=8)
        ax.text(0.24, y + 0.022, eq, ha="left", va="center", fontsize=6.4, color=INK)
        ax.text(0.24, y - 0.032, note, ha="left", va="center", fontsize=5.3, color=MUTE, style="italic")

    ax.text(1.08, 0.87, "Error broadcast $e_n$", fontsize=6.8, color=MUTE, fontweight="bold")
    modes = [
        ("Rank-1", "1 shared field", COLORS["scalar"]),
        ("Neuron-wise", "soma-aligned", COLORS["per_soma"]),
        ("Rank-K", "few random channels", LOW_RANK),
        ("Path", "branch roles", PATHWAY),
        ("Oracle", r"$\alpha_n\delta_0$", ORACLE),
    ]
    yvals = [0.76, 0.62, 0.48, 0.34, 0.20]
    for (name, note, color), y in zip(modes, yvals):
        box(ax, (1.08, y - 0.048), 0.82, 0.096, fc="white", ec="#D8DEE8", lw=0.6)
        ax.plot([1.12, 1.26], [y, y], color=color, lw=1.9, solid_capstyle="round")
        ax.add_patch(Circle((1.12, y), 0.017, fc=color, ec="white", linewidth=0.3, zorder=8))
        ax.text(1.31, y + 0.018, name, ha="left", va="center", fontsize=6.1, color=color, fontweight="bold")
        ax.text(1.31, y - 0.024, note, ha="left", va="center", fontsize=5.1, color=MUTE, style="italic")

    arrow(ax, (0.94, 0.52), (1.03, 0.52), color=MUTE, lw=0.8, ms=6)
    ax.text(
        1.01,
        0.075,
        r"3F gives the exact local eligibility; 5F is the practical stabilizer used in headline runs.",
        ha="center",
        va="center",
        fontsize=5.4,
        color=MUTE,
        style="italic",
    )


def panel_c(ax):
    width = 1.38
    setup_panel(ax, width, "Path gains")
    colors = ["#D95F4B", "#58A66E", "#4F79B8"]
    tree = draw_tree(
        ax,
        width=width,
        x_leaf=0.27,
        x_branch=0.59,
        x_soma=1.02,
        branch_ys=(0.80, 0.56, 0.32),
        leaf_offsets=(0.055, 0.0, -0.055),
        branch_colors=colors,
        show_synapses=False,
        labels=False,
    )
    labels = [r"$\alpha_1$ high", r"$\alpha_2$ mid", r"$\alpha_3$ low"]
    for (x, y), col, label in zip(tree["branches"], colors, labels):
        ax.text(x - 0.10, y + 0.07, label, fontsize=7.8, color=col, ha="center", fontweight="bold")
        arrow(ax, (1.12, 0.50), (x + 0.035, y), color=EXACT, lw=1.15, ms=7.5, alpha=0.75, ls="--", rad=0.20 if y > 0.5 else (-0.20 if y < 0.5 else 0.0))

    ax.text(1.18, 0.59, "exact\nerrors", ha="center", va="bottom", fontsize=8.1, color=EXACT, fontweight="bold")
    ax.add_patch(Circle((1.12, 0.50), 0.022, fc=EXACT, ec="white", linewidth=0.5, zorder=10))

    ax.add_patch(Circle((1.12, 0.17), 0.022, fc=LOCAL, ec="white", linewidth=0.5, zorder=10))
    for _, y in tree["branches"]:
        arrow(ax, (1.10, 0.17), (0.63, y - 0.02), color=LOCAL, lw=0.95, ms=6.5, alpha=0.55, rad=0.18)
    ax.text(1.18, 0.17, "rank-1\nbroadcast", ha="left", va="center", fontsize=7.7, color=LOCAL, fontweight="bold", linespacing=0.85)

    ax.text(
        0.60,
        0.085,
        "compressible gains make shared broadcast work",
        ha="center",
        va="center",
        fontsize=6.5,
        color=MUTE,
        style="italic",
    )


def draw_tiny_tree(ax, x, y, scale=0.10, branch_colors=None):
    branch_colors = branch_colors or [DEND, DEND]
    soma = (x + scale * 0.55, y)
    hubs = [(x, y + scale * 0.28), (x, y - scale * 0.28)]
    leaves = [
        (x - scale * 0.55, y + scale * 0.48),
        (x - scale * 0.55, y + scale * 0.12),
        (x - scale * 0.55, y - scale * 0.12),
        (x - scale * 0.55, y - scale * 0.48),
    ]
    for i, hub in enumerate(hubs):
        ax.plot([hub[0], soma[0] - scale * 0.20], [hub[1], soma[1]], color=branch_colors[i], lw=1.0, alpha=0.85)
        for leaf in leaves[2 * i : 2 * i + 2]:
            ax.plot([leaf[0], hub[0]], [leaf[1], hub[1]], color=branch_colors[i], lw=0.8, alpha=0.78)
    for leaf in leaves:
        ax.add_patch(Circle(leaf, scale * 0.075, fc=DEND, ec=EDGE, linewidth=0.35, zorder=4))
    for i, hub in enumerate(hubs):
        ax.add_patch(Circle(hub, scale * 0.09, fc=branch_colors[i], ec=EDGE, linewidth=0.35, zorder=5))
    ax.add_patch(Circle(soma, scale * 0.13, fc=SOMA, ec=EDGE, linewidth=0.45, zorder=6))
    return soma, hubs, leaves


def panel_d(ax):
    width = 2.05
    setup_panel(ax, width, "Error broadcast modes")
    modes = [
        ("Rank-1", "shared", COLORS["scalar"], "local"),
        ("Neuron-\nwise", "soma aligned", COLORS["per_soma"], "local"),
        ("Rank-K", "K channels", LOW_RANK, "low-rank"),
        ("Path", "branches", PATHWAY, "path"),
        ("Oracle", r"$\alpha_n\delta_0$", ORACLE, "oracle"),
    ]
    xs = np.linspace(0.19, 1.86, len(modes))
    for x, (name, note, col, kind) in zip(xs, modes):
        box(ax, (x - 0.168, 0.235), 0.336, 0.59, fc="white", ec="#D8DEE8", lw=0.72, radius=0.025)
        is_multiline = "\n" in name
        ax.text(
            x,
            0.776 if is_multiline else 0.765,
            name,
            ha="center",
            va="center",
            fontsize=7.15 if is_multiline else 7.7,
            color=col,
            fontweight="bold",
            linespacing=0.78,
        )
        ax.text(x, 0.692 if is_multiline else 0.715, note, ha="center", va="center", fontsize=5.45, color=MUTE)
        soma, hubs, leaves = draw_tiny_tree(ax, x - 0.02, 0.47, scale=0.17, branch_colors=[DEND, DEND])

        if kind == "local":
            srcs = [(x - 0.13, 0.60)] if name == "Rank-1" else [(x - 0.13, 0.55), (x - 0.13, 0.39)]
            for si, src in enumerate(srcs):
                ax.add_patch(Circle(src, 0.014, fc=col, ec="white", linewidth=0.4, zorder=9))
                target = soma if len(srcs) == 1 else (soma[0], soma[1] + (0.035 if si == 0 else -0.035))
                arrow(ax, (src[0] + 0.014, src[1]), (target[0] - 0.024, target[1]), color=col, lw=0.82, ms=5.0, alpha=0.85, rad=-0.08 if si == 0 else 0.08)
        elif kind == "low-rank":
            src = (x - 0.135, 0.50)
            chans = [(x - 0.055, 0.56), (x - 0.055, 0.44)]
            ax.add_patch(Circle(src, 0.013, fc=col, ec="white", linewidth=0.4, zorder=9))
            for ch in chans:
                ax.add_patch(Circle(ch, 0.010, fc=col, ec="white", linewidth=0.3, zorder=9))
                arrow(ax, (src[0] + 0.013, src[1]), (ch[0] - 0.010, ch[1]), color=col, lw=0.72, ms=4.4, alpha=0.75)
                arrow(ax, (ch[0] + 0.010, ch[1]), (soma[0] - 0.024, soma[1]), color=col, lw=0.72, ms=4.4, alpha=0.55)
        elif kind == "path":
            for i, hub in enumerate(hubs):
                src = (x - 0.13, 0.56 if i == 0 else 0.38)
                ax.add_patch(Circle(src, 0.011, fc=col, ec="white", linewidth=0.35, zorder=9))
                arrow(ax, (src[0] + 0.011, src[1]), (hub[0] - 0.017, hub[1]), color=col, lw=0.82, ms=5.0, alpha=0.85, rad=-0.06 if i == 0 else 0.06)
        else:
            for leaf in leaves:
                arrow(ax, (soma[0] + 0.020, soma[1]), (leaf[0] + 0.014, leaf[1]), color=col, lw=0.72, ms=4.4, alpha=0.72, ls="--", rad=0.12)

    box(ax, (0.20, 0.065), 1.65, 0.115, fc="#F8FAFC", ec="#CBD5E1", lw=0.65)
    ax.text(
        1.025,
        0.125,
        r"$\partial L/\partial g_i=x_iR_n^{\rm tot}(E_i-V_n)\delta_n;\quad \Delta g_i\propto x_iR_n^{\rm tot}(E_i-V_n)e_n,\ e_n\approx\delta_n$",
        ha="center",
        va="center",
        fontsize=6.55,
        color=INK,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.05, 2.82))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.25, 1.30, 2.30],
        left=0.040,
        right=0.988,
        top=0.82,
        bottom=0.075,
        wspace=0.17,
    )
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
    ]
    panel_a(axes[0])
    panel_c(axes[1])
    panel_d(axes[2])
    for ax, label in zip(axes, "ABC"):
        panel_label(ax, label, x=-0.08, y=1.13, fontsize=13.0)

    out = OUTPUT_DIR / "fig1_model_and_credit"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=350)
    print(f"Saved {out}.pdf and {out}.png")
    plt.close(fig)


if __name__ == "__main__":
    main()

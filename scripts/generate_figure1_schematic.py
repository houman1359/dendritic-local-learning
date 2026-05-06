#!/usr/bin/env python3
"""Generate the submission Figure 1 schematic.

Saves:
  figures/fig1_model_and_credit.{pdf,png}

Layout:
  * One full-width row: A conductance tree, B path gains, C feedback modes.
  * Caption is handled in LaTeX below the figure, not as a side minipage.
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


def setup_panel(ax, width=1.0, height=1.0, title=None, title_x=0.04, title_size=8.5):
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", x=title_x, fontsize=title_size,
                     pad=3, fontweight="bold")


def rounded_box(ax, xy, w, h, fc="white", ec="#D8DEE8", lw=0.7, radius=0.018, z=2):
    p = FancyBboxPatch(xy, w, h,
                       boxstyle=f"round,pad=0.010,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, xy1, xy2, color=INK, lw=1.0, ms=7, alpha=1.0,
          ls="-", rad=0.0, z=6):
    ax.add_patch(FancyArrowPatch(
        xy1, xy2, arrowstyle="-|>", mutation_scale=ms,
        linewidth=lw, color=color, alpha=alpha,
        linestyle=ls, connectionstyle=f"arc3,rad={rad}",
        shrinkA=0, shrinkB=0, zorder=z,
    ))


def draw_synapse(ax, x, y, kind="E"):
    if kind == "E":
        ax.add_patch(Circle((x, y), 0.010, fc=EXC, ec="white",
                            linewidth=0.4, zorder=9))
    else:
        ax.add_patch(mpatches.RegularPolygon(
            (x, y), numVertices=3, radius=0.014,
            orientation=np.pi, fc=INH, ec="white",
            linewidth=0.4, zorder=9,
        ))


def draw_tree(ax, *, x_leaf, x_branch, x_soma, branch_ys, leaf_offsets,
              branch_colors=None, show_synapses=True, leaf_r=0.017,
              branch_r=0.024, soma_r=0.034):
    branch_ys = np.asarray(branch_ys)
    leaf_offsets = np.asarray(leaf_offsets)
    branch_colors = branch_colors or [DEND] * len(branch_ys)
    leaves = []
    for bi, by in enumerate(branch_ys):
        for off in leaf_offsets:
            ly = float(by + off)
            leaves.append((x_leaf, ly, bi))
            ax.plot([x_leaf + leaf_r, x_branch - branch_r],
                    [ly, by + 0.25 * off],
                    color=branch_colors[bi], linewidth=1.2, alpha=0.85,
                    solid_capstyle="round", zorder=2)
        ax.plot([x_branch + branch_r, x_soma - soma_r],
                [by, 0.50 + 0.16 * (by - 0.50)],
                color=branch_colors[bi], linewidth=1.9, alpha=0.92,
                solid_capstyle="round", zorder=2)
    for x, y, _ in leaves:
        ax.add_patch(Circle((x, y), leaf_r, fc=DEND, ec=EDGE,
                            linewidth=0.55, zorder=4))
        if show_synapses:
            draw_synapse(ax, x - 0.030, y - 0.005, "E")
            draw_synapse(ax, x - 0.030, y + 0.011, "E")
            draw_synapse(ax, x + 0.005, y + 0.030, "I")
    for bi, by in enumerate(branch_ys):
        ax.add_patch(Circle((x_branch, by), branch_r, fc=branch_colors[bi],
                            ec=EDGE, linewidth=0.6, zorder=5))
    ax.add_patch(Circle((x_soma, 0.50), soma_r, fc=SOMA, ec=EDGE,
                        linewidth=0.7, zorder=6))
    return {"branches": list(zip([x_branch] * len(branch_ys), branch_ys)),
            "leaves": leaves, "soma": (x_soma, 0.50)}


def panel_a(ax):
    width = 1.20
    setup_panel(ax, width, title="Conductance tree", title_size=9.0)
    draw_tree(ax,
              x_leaf=0.30, x_branch=0.56, x_soma=0.82,
              branch_ys=(0.74, 0.52, 0.30), leaf_offsets=(0.075, 0.0, -0.075))
    arrow(ax, (0.86, 0.50), (1.02, 0.50), lw=1.1, ms=8)
    ax.text(1.04, 0.50, "out", ha="left", va="center",
            fontsize=7.4, color=INK, fontweight="bold")
    ax.text(0.22, 0.22, r"$V_n$", fontsize=8.4, color=INK, fontweight="bold")
    ax.text(0.53, 0.20, r"$V_p$", fontsize=8.4, color=INK, fontweight="bold")
    ax.text(0.82, 0.40, "soma", fontsize=7.0, color=INK, ha="center", fontweight="bold")
    legend = [(0.06, "E", EXC), (0.26, "I", INH),
              (0.46, "branch", DEND), (0.82, "soma", SOMA)]
    for x, label, color in legend:
        ax.add_patch(Circle((x, 0.08), 0.012, fc=color, ec="white", linewidth=0.35, zorder=8))
        ax.text(x + 0.020, 0.08, label, fontsize=6.2, ha="left", va="center", color=INK)


def panel_b(ax):
    width = 1.30
    setup_panel(ax, width, title="Path gains", title_size=9.0)
    cols = ["#D95F4B", "#58A66E", "#4F79B8"]
    tree = draw_tree(ax,
                     x_leaf=0.20, x_branch=0.46, x_soma=0.78,
                     branch_ys=(0.78, 0.54, 0.30),
                     leaf_offsets=(0.055, 0.0, -0.055),
                     branch_colors=cols, show_synapses=False,
                     leaf_r=0.014, branch_r=0.020, soma_r=0.030)
    labels = [r"$\alpha_1$ hi", r"$\alpha_2$ mid", r"$\alpha_3$ lo"]
    for (x, y), col, lab in zip(tree["branches"], cols, labels):
        ax.text(x - 0.09, y + 0.060, lab,
                fontsize=7.2, color=col, ha="center", fontweight="bold")
    # Exact-errors source on the right (well inside the panel width)
    ax.add_patch(Circle((0.97, 0.50), 0.020, fc=EXACT, ec="white", linewidth=0.45, zorder=10))
    ax.text(1.00, 0.66, "exact\nerrors", ha="left", va="bottom",
            fontsize=6.6, color=EXACT, fontweight="bold", linespacing=0.85)
    for (x, y), col in zip(tree["branches"], cols):
        arrow(ax, (0.95, 0.50), (x + 0.025, y),
              color=EXACT, lw=1.0, ms=6, alpha=0.85, ls="--",
              rad=0.18 if y > 0.5 else (-0.18 if y < 0.5 else 0))
    # Rank-1 broadcast source on the left
    ax.add_patch(Circle((0.05, 0.50), 0.020, fc=LOCAL, ec="white", linewidth=0.45, zorder=10))
    ax.text(0.05, 0.30, "rank-1\nbroadcast", ha="center", va="top",
            fontsize=6.4, color=LOCAL, fontweight="bold", linespacing=0.85)
    for (x, y), col in zip(tree["branches"], cols):
        arrow(ax, (0.075, 0.50), (x - 0.025, y),
              color=LOCAL, lw=0.9, ms=6, alpha=0.62,
              rad=-0.16 if y > 0.5 else (0.16 if y < 0.5 else 0))
    ax.text(0.60, 0.05, "compressible gains $\\Rightarrow$ broadcast works",
            ha="center", va="center", fontsize=6.0, color=MUTE, style="italic")


def panel_c(ax):
    """Five broadcast modes in one row, each as a tiny labeled card.
    The equation strip sits BELOW the cards in its own gutter."""
    width = 2.30
    height = 1.0
    setup_panel(ax, width, height, title="Error broadcast modes",
                title_x=0.02, title_size=9.0)

    modes = [
        ("Rank-1",     "shared",    COLORS["scalar"],   "local"),
        ("Neuron",     "per soma",  COLORS["per_soma"], "neuron"),
        ("Rank-K",     "$K$ chans", LOW_RANK,           "low-rank"),
        ("Path",       "branches",  PATHWAY,            "path"),
        ("Oracle",     r"$\alpha_n\delta_0$", ORACLE,   "oracle"),
    ]
    xs = np.linspace(0.25, 2.05, len(modes))
    card_h = 0.50
    card_w = 0.36
    card_top = 0.86
    for x, (name, note, col, kind) in zip(xs, modes):
        rounded_box(ax, (x - card_w / 2, card_top - card_h),
                    card_w, card_h,
                    fc="white", ec="#D8DEE8", lw=0.7, radius=0.020)
        ax.text(x, card_top - 0.05, name, ha="center", va="center",
                fontsize=7.3, color=col, fontweight="bold")
        ax.text(x, card_top - 0.10, note, ha="center", va="center",
                fontsize=5.6, color=MUTE)
        # mini-tree
        soma = (x + 0.08, card_top - 0.30)
        hubs = [(x - 0.05, card_top - 0.22), (x - 0.05, card_top - 0.38)]
        leaves = [(x - 0.15, card_top - 0.18), (x - 0.15, card_top - 0.26),
                  (x - 0.15, card_top - 0.34), (x - 0.15, card_top - 0.42)]
        for hi, hub in enumerate(hubs):
            ax.plot([hub[0], soma[0] - 0.03], [hub[1], soma[1]],
                    color=DEND, lw=0.85, alpha=0.85)
            for leaf in leaves[2 * hi:2 * hi + 2]:
                ax.plot([leaf[0], hub[0]], [leaf[1], hub[1]],
                        color=DEND, lw=0.65, alpha=0.78)
        for leaf in leaves:
            ax.add_patch(Circle(leaf, 0.010, fc=DEND, ec=EDGE, linewidth=0.3, zorder=4))
        for hub in hubs:
            ax.add_patch(Circle(hub, 0.012, fc=DEND, ec=EDGE, linewidth=0.3, zorder=5))
        ax.add_patch(Circle(soma, 0.018, fc=SOMA, ec=EDGE, linewidth=0.4, zorder=6))
        # Source(s) on the left
        if kind == "local":
            srcs = [(x - 0.16, card_top - 0.30)]
        elif kind == "neuron":
            srcs = [(x - 0.16, card_top - 0.22), (x - 0.16, card_top - 0.38)]
        elif kind == "low-rank":
            srcs = [(x - 0.16, card_top - 0.26), (x - 0.16, card_top - 0.34)]
        elif kind == "path":
            srcs = [(x - 0.16, card_top - 0.22), (x - 0.16, card_top - 0.38)]
        else:  # oracle
            srcs = [(x - 0.16, card_top - 0.30)]
        for si, src in enumerate(srcs):
            ax.add_patch(Circle(src, 0.010, fc=col, ec="white", linewidth=0.35, zorder=9))
            tgt = soma if kind in ("local", "oracle") else (
                hubs[si] if kind in ("path", "neuron") else soma
            )
            arrow(ax, (src[0] + 0.010, src[1]), (tgt[0] - 0.018, tgt[1]),
                  color=col, lw=0.75, ms=4.5, alpha=0.85,
                  rad=0 if kind == "local" else (0.06 if si == 0 else -0.06))
        if kind == "oracle":
            for leaf in leaves:
                arrow(ax, (soma[0] + 0.018, soma[1]), (leaf[0] + 0.010, leaf[1]),
                      color=col, lw=0.6, ms=4.0, alpha=0.66, ls="--", rad=0.08)

    # Equation strip BELOW the cards.
    rounded_box(ax, (0.01, 0.04), width - 0.02, 0.16,
                fc="#F8FAFC", ec="#CBD5E1", lw=0.6, radius=0.020)
    ax.text(width / 2, 0.12,
            r"$\partial L/\partial g_i = x_i\,R_n^{\rm tot}(E_i - V_n)\,\delta_n,\quad"
            r"\Delta g_i \propto x_i\,R_n^{\rm tot}(E_i - V_n)\,e_n,\ e_n \approx \delta_n$",
            ha="center", va="center", fontsize=6.7, color=INK)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Full-width NeurIPS row. The aspect is intentionally shallow so the
    # caption can sit below the figure without pushing the paper over length.
    fig = plt.figure(figsize=(6.85, 2.15))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.18, 1.28, 2.32],
        wspace=0.24,
        left=0.030, right=0.995, top=0.86, bottom=0.10,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)
    for ax, lbl, x_off in [(ax_a, "A", -0.08), (ax_b, "B", -0.06), (ax_c, "C", -0.035)]:
        panel_label(ax, lbl, x=x_off, y=1.08, fontsize=11.0)

    out = OUTPUT_DIR / "fig1_model_and_credit"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()

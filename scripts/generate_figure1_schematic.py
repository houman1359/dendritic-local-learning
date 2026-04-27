#!/usr/bin/env python3
"""Generate Figure 1: dendritic neuron, exact credit vs LocalCA broadcast,
local-rule family, and feedback-channel structure.

Outputs:
  - figures/fig1_model_and_credit.{pdf,png}

Design principles:
  - Use aspect='equal' on every schematic so circles stay circular.
  - Use one coordinate frame per panel that matches the panel's physical
    aspect ratio, so content fills the panel without horizontal waste.
  - Show error feedback reaching BOTH proximal and distal compartments.
  - Panel D uses somas (orange) as broadcast targets, each with a
    small dendritic tree, so channel structure is unambiguous.
  - Keep Fig. 1 conceptual. Quantitative evidence is handled by the
    main result figures, where axes and legends can be large enough.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import (  # noqa: E402
    COLORS,
    apply_neurips_style,
    clean_schematic_axis,
    panel_label,
    style_axis,
)

apply_neurips_style()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as mpatches  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ── Palette shortcuts ────────────────────────────────────────────────────
EXC  = COLORS["exc"]
INH  = COLORS["inh"]
DEND = COLORS["dend"]
SOMA = COLORS["soma"]
INK  = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]
EXACT  = "#A03E36"    # backprop / exact credit
APPROX = "#D97D51"    # local-broadcast credit

# ── Paths ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"
CIFAR_DFX_CSV = (
    OUTPUT_DIR.parent
    / "analysis"
    / "cifar10_compactei_depth4_decoderfix_mechanism_5seed"
    / "cifar10_compactei_depth4_decoderfix_mechanism_summary.csv"
)
CIFAR_BP_CSV = (
    OUTPUT_DIR.parent
    / "analysis"
    / "cifar10_compactei_depth4"
    / "cifar10_compactei_depth4_grouped_summary.csv"
)


# ── Drawing primitives ──────────────────────────────────────────────────
def draw_arrow(ax, x1, y1, x2, y2, color=INK, lw=1.1, style="-|>",
               mutation_scale=10, zorder=5, alpha=1.0, linestyle="-",
               connectionstyle="arc3"):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=mutation_scale,
        color=color, lw=lw, alpha=alpha, zorder=zorder,
        shrinkA=0, shrinkB=0, linestyle=linestyle,
        connectionstyle=connectionstyle,
    ))


def rounded_box(ax, cx, cy, w, h, fc="white", ec=EDGE, lw=0.8, radius=0.025,
                alpha=1.0, zorder=2):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        fc=fc, ec=ec, linewidth=lw, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_exc(ax, x, y, r=0.010):
    ax.add_patch(Circle((x, y), r, fc=EXC, ec="white", linewidth=0.4, zorder=7))


def draw_inh(ax, x, y, r=0.011):
    ax.add_patch(mpatches.RegularPolygon(
        (x, y), numVertices=3, radius=r * 1.45, orientation=np.pi,
        fc=INH, ec="white", linewidth=0.4, zorder=7,
    ))


def soma_circle(ax, xy, r=0.045, label=None, color=SOMA, label_color="white",
                ec=EDGE, lw=0.9, zorder=5, fontsize=7.4):
    ax.add_patch(Circle(xy, r, fc=color, ec=ec, linewidth=lw, zorder=zorder))
    if label is not None:
        ax.text(xy[0], xy[1], label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=label_color,
                zorder=zorder + 1)


def draw_wire(ax, x1, y1, x2, y2, color=DEND, lw=1.3, alpha=0.82, zorder=1):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, alpha=alpha, zorder=zorder,
            solid_capstyle="round")


def setup_panel(ax, W=1.0, H=1.0, title=None, title_x=0.02):
    """Configure a schematic panel with aspect='equal' and the given coordinate width/height."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    if title is not None:
        ax.set_title(title, fontsize=9.4, pad=5, loc="left", x=title_x)


# ── Tree drawing helper ─────────────────────────────────────────────────
def draw_tree(ax, *, x_leaf, x_branch, x_soma, branch_ys=None, leaf_offsets=None,
              leaf_r=0.020, branch_r=0.028, soma_r=0.035,
              draw_synapses=True, label_only_center=True,
              distal_label=r"$V_n$", proximal_label=r"$V_{p(n)}$",
              soma_label=r"$V_{\mathrm{out}}$"):
    """Draw a [3,3] dendritic tree using *circular* nodes so they stay round
    under aspect='equal'.
    """
    if branch_ys is None:
        branch_ys = np.array([0.78, 0.50, 0.22])
    if leaf_offsets is None:
        leaf_offsets = np.array([+0.10, 0.0, -0.10])

    leaf_ys = []
    for by in branch_ys:
        for off in leaf_offsets:
            leaf_ys.append(by + off)
    leaf_ys = np.array(leaf_ys)

    # Wires: leaf → branch
    for bi, by in enumerate(branch_ys):
        for off in leaf_offsets:
            ly = by + off
            draw_wire(ax, x_leaf + leaf_r + 0.002, ly,
                      x_branch - branch_r - 0.002, by + off * 0.22,
                      color=DEND, lw=1.05, alpha=0.70)

    # Wires: branch → soma
    for by in branch_ys:
        draw_wire(ax, x_branch + branch_r + 0.002, by,
                  x_soma - soma_r - 0.002, 0.5 + (by - 0.5) * 0.18,
                  color=DEND, lw=1.7, alpha=0.82)

    # Leaves (distal compartments) — CIRCLES
    for ly in leaf_ys:
        ax.add_patch(Circle((x_leaf, ly), leaf_r, fc=DEND, ec=EDGE,
                            linewidth=0.5, alpha=0.92, zorder=3))

    # Proximal branches — CIRCLES (slightly larger)
    for by in branch_ys:
        ax.add_patch(Circle((x_branch, by), branch_r, fc=DEND, ec=EDGE,
                            linewidth=0.6, alpha=1.0, zorder=3))

    # Soma
    soma_circle(ax, (x_soma, 0.5), r=soma_r, label=None)

    # Synapses on distal leaves
    if draw_synapses:
        for ly in leaf_ys:
            draw_exc(ax, x_leaf - leaf_r - 0.012, ly + 0.006, r=0.0075)
            draw_exc(ax, x_leaf - leaf_r - 0.012, ly - 0.006, r=0.0075)
            draw_inh(ax, x_leaf,                  ly + leaf_r + 0.012, r=0.009)

    # Center-compartment labels
    if label_only_center:
        ax.text(x_leaf,   leaf_ys[4], distal_label,    ha="center", va="center",
                fontsize=6.8, color="white", fontweight="bold", zorder=9)
        ax.text(x_branch, branch_ys[1], proximal_label, ha="center", va="center",
                fontsize=6.8, color="white", fontweight="bold", zorder=9)
        ax.text(x_soma, 0.5, soma_label, ha="center", va="center",
                fontsize=6.8, color="white", fontweight="bold", zorder=9)

    return dict(leaf_x=x_leaf, branch_x=x_branch, soma_x=x_soma,
                branch_ys=branch_ys, leaf_ys=leaf_ys,
                leaf_r=leaf_r, branch_r=branch_r, soma_r=soma_r)


# ── Panel A: dendritic neuron (anatomy + shunting) ──────────────────────
def panel_A(ax, W=1.20):
    setup_panel(ax, W=W, H=1.0,
                title="Shunting dendritic neuron")

    # Tree: use generous horizontal spacing
    tree = draw_tree(
        ax,
        x_leaf=0.34, x_branch=0.58, x_soma=0.82,
        branch_ys=np.array([0.78, 0.50, 0.22]),
        leaf_offsets=np.array([+0.09, 0.0, -0.09]),
        leaf_r=0.020, branch_r=0.028, soma_r=0.035,
    )

    # Legend — top-left, not overlapping the tree
    lx, ly0 = 0.02, 0.92
    ax.text(lx, ly0, "external input", fontsize=7.5,
            color=INK, fontweight="bold", ha="left", va="center")
    draw_exc(ax, lx + 0.010, ly0 - 0.07, r=0.0085)
    ax.text(lx + 0.028, ly0 - 0.07, r"exc. ($E_j^E > 0$)",
            fontsize=7.0, color=EXC, va="center", fontweight="bold")
    draw_inh(ax, lx + 0.010, ly0 - 0.14, r=0.0095)
    ax.text(lx + 0.028, ly0 - 0.14, r"inh. ($E_j^I = 0$)",
            fontsize=7.0, color=INH, va="center", fontweight="bold")

    # Anatomy labels below the tree
    ax.text(tree["leaf_x"],   0.05, "distal",   ha="center", va="center",
            fontsize=7.2, color=MUTE, style="italic")
    ax.text(tree["branch_x"], 0.05, "proximal", ha="center", va="center",
            fontsize=7.2, color=MUTE, style="italic")
    ax.text(tree["soma_x"],   0.05, "soma",     ha="center", va="center",
            fontsize=7.2, color=MUTE, style="italic")
    ax.text(W * 0.47, 0.01, r"tree $[3,3]$: 3 proximal $\times$ 3 distal",
            ha="center", va="center", fontsize=6.7, color=MUTE, style="italic")

    # Output arrow from soma
    draw_arrow(ax, tree["soma_x"] + tree["soma_r"], 0.5, W - 0.14, 0.5,
               color=INK, lw=1.3, mutation_scale=11)
    ax.text(W - 0.12, 0.5, "output", fontsize=7.8, color=INK,
            va="center", ha="left", fontweight="bold")

    # Equation callout — placed above the output arrow in the top-right corner.
    # Tall enough to give the annotation clear vertical space from the fraction bar.
    eq_cx, eq_cy = W - 0.15, 0.84
    rounded_box(ax, eq_cx, eq_cy, 0.28, 0.18,
                fc="#FAFBFC", ec="#D7DCE2", lw=0.8, radius=0.02)
    ax.text(eq_cx, eq_cy + 0.032,
            r"$V_n = \frac{\sum_j g_j\,x_j\,E_j}{g_n^{\mathrm{tot}}}$",
            ha="center", va="center", fontsize=10.0, color=INK)
    ax.text(eq_cx, eq_cy - 0.063,
            r"shunting: $I$ in denominator",
            ha="center", va="center", fontsize=6.6, color=INH,
            fontweight="bold", style="italic")


# ── Panel B: credit-assignment contrast (BP vs LocalCA) ─────────────────
def panel_B(ax, W=2.60):
    setup_panel(ax, W=W, H=1.0,
                title="Exact credit vs. LocalCA",
                title_x=0.01)

    # Left subpanel (BP) — centred around x = W*0.25
    bp_cx = W * 0.25
    tree_L = draw_tree(
        ax,
        x_leaf=bp_cx - 0.30, x_branch=bp_cx - 0.10, x_soma=bp_cx + 0.12,
        branch_ys=np.array([0.70, 0.50, 0.30]),
        leaf_offsets=np.array([+0.075, 0.0, -0.075]),
        leaf_r=0.015, branch_r=0.022, soma_r=0.026,
        draw_synapses=False,
        label_only_center=False,
    )

    ax.text(bp_cx, 0.92, "Backprop (exact)",
            ha="center", va="center",
            fontsize=9.0, color=EXACT, fontweight="bold")

    # BP arrows: to each PROXIMAL branch (solid dashed, thicker)
    for by in tree_L["branch_ys"]:
        draw_arrow(
            ax,
            tree_L["branch_x"], by + 0.12,
            tree_L["branch_x"], by + tree_L["branch_r"] + 0.004,
            color=EXACT, lw=1.3, mutation_scale=9, linestyle="--",
            alpha=0.95,
        )
    # BP arrows: to each DISTAL leaf (thinner, also dashed) — shows compartment-specific error
    for ly in tree_L["leaf_ys"]:
        draw_arrow(
            ax,
            tree_L["leaf_x"] - 0.085, ly + 0.050,
            tree_L["leaf_x"] - tree_L["leaf_r"] - 0.004, ly + 0.008,
            color=EXACT, lw=0.85, mutation_scale=6, linestyle="--",
            alpha=0.75, connectionstyle="arc3,rad=-0.10",
        )

    ax.text(bp_cx, 0.865,
            r"$\partial L/\partial V_n$ per compartment",
            ha="center", va="center", fontsize=7.0,
            color=EXACT, fontweight="bold")
    ax.text(bp_cx, 0.12,
            "compartment-specific error\nat EVERY dendritic level",
            ha="center", va="center", fontsize=6.8, color=EXACT,
            fontweight="bold", style="italic")

    # Right subpanel (LocalCA) — centred around x = W*0.72
    lc_cx = W * 0.72
    tree_R = draw_tree(
        ax,
        x_leaf=lc_cx - 0.30, x_branch=lc_cx - 0.10, x_soma=lc_cx + 0.12,
        branch_ys=np.array([0.70, 0.50, 0.30]),
        leaf_offsets=np.array([+0.075, 0.0, -0.075]),
        leaf_r=0.015, branch_r=0.022, soma_r=0.026,
        draw_synapses=False,
        label_only_center=False,
    )

    ax.text(lc_cx, 0.92, "LocalCA (scalar broadcast)",
            ha="center", va="center",
            fontsize=9.0, color=APPROX, fontweight="bold")

    # Single source on the far right
    src_R = (W - 0.12, 0.50)
    ax.add_patch(Circle(src_R, 0.022, fc=APPROX, ec="white",
                        linewidth=0.6, zorder=7))
    ax.text(src_R[0], src_R[1], r"$e$",
            ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold", zorder=8)
    ax.text(src_R[0], src_R[1] + 0.09, "broadcast",
            ha="center", va="bottom", fontsize=7.0,
            color=APPROX, fontweight="bold")

    # e → soma
    draw_arrow(ax, src_R[0] - 0.022, src_R[1],
               tree_R["soma_x"] + tree_R["soma_r"] + 0.003, 0.50,
               color=APPROX, lw=1.4, mutation_scale=10, alpha=0.95)

    # Soma → each proximal branch (curved, thicker), then → distal leaves (thinner)
    for by in tree_R["branch_ys"]:
        # soma → proximal
        draw_arrow(
            ax, tree_R["soma_x"] - tree_R["soma_r"] * 0.25,
            0.50,
            tree_R["branch_x"] + tree_R["branch_r"] + 0.003, by,
            color=APPROX, lw=1.1, mutation_scale=7, alpha=0.85,
            connectionstyle=f"arc3,rad={0.22 if by > 0.5 else (-0.22 if by < 0.5 else 0)}",
        )
        # proximal → each of its 3 distal leaves
        for off in [+0.075, 0.0, -0.075]:
            draw_arrow(
                ax,
                tree_R["branch_x"] - tree_R["branch_r"] * 0.25, by,
                tree_R["leaf_x"] + tree_R["leaf_r"] + 0.003, by + off,
                color=APPROX, lw=0.75, mutation_scale=5, alpha=0.65,
                connectionstyle=f"arc3,rad={0.12 if off > 0 else (-0.12 if off < 0 else 0)}",
            )

    ax.text(lc_cx, 0.12,
            "single $e$ flows soma $\\rightarrow$ proximal $\\rightarrow$ distal",
            ha="center", va="center", fontsize=6.8, color=APPROX,
            fontweight="bold", style="italic")

    # Center separator
    ax.plot([W * 0.485, W * 0.485], [0.17, 0.82],
            color="#D0D0D0", lw=0.6, ls=":")

    # Bottom summary line
    ax.text(W * 0.5, 0.015,
            r"Shunting concentrates per-synapse path gains $\Rightarrow$ one scalar $e$ is enough for local credit",
            ha="center", va="center", fontsize=7.2, color=INK, fontweight="bold")


# ── Panel C: rule family ────────────────────────────────────────────────
def panel_C(ax, W=1.20):
    setup_panel(ax, W=W, H=1.0, title="Local rule family")

    cards = [
        dict(name="3F", color=COLORS["rule_3f"], y=0.80,
             badge="exact", badge_fc="#E4F0EA", badge_ec=COLORS["shunting"],
             eq=r"$\Delta g_j \propto x_j\,R_n^{\mathrm{tot}}\,(E_j-V_n)\,e_n$",
             note="theorem-facing 3-factor"),
        dict(name="4F", color=COLORS["rule_4f"], y=0.50,
             badge="heuristic", badge_fc="#FBEFE8", badge_ec=COLORS["rule_4f"],
             eq=r"$\Delta g_j \propto \mathrm{3F} \cdot \rho_n$",
             note=r"adds $\rho_n$: morphology / variance"),
        dict(name="5F", color=COLORS["rule_5f"], y=0.20,
             badge="heuristic", badge_fc="#ECEEF7", badge_ec=COLORS["rule_5f"],
             eq=r"$\Delta g_j \propto \mathrm{4F} \cdot \phi_n$",
             note=r"adds $\phi_n$: confidence"),
    ]
    for c in cards:
        rounded_box(ax, W * 0.52, c["y"], W * 0.94, 0.24,
                    fc="white", ec="#D7DCE2", lw=0.6, radius=0.02)
        # Tag — circle so it stays round
        ax.add_patch(Circle((W * 0.10, c["y"]), 0.075,
                            fc=c["color"], ec=c["color"], lw=0, zorder=4))
        ax.text(W * 0.10, c["y"], c["name"], ha="center", va="center",
                fontsize=11.0, color="white", fontweight="bold", zorder=5)
        # Badge
        rounded_box(ax, W * 0.87, c["y"] + 0.075, 0.14, 0.05,
                    fc=c["badge_fc"], ec=c["badge_ec"], lw=0.7, radius=0.012)
        ax.text(W * 0.87, c["y"] + 0.075, c["badge"],
                ha="center", va="center",
                fontsize=6.2, color=c["badge_ec"], fontweight="bold")
        # Equation & note
        ax.text(W * 0.22, c["y"] + 0.030, c["eq"],
                ha="left", va="center", fontsize=8.6, color=INK)
        ax.text(W * 0.22, c["y"] - 0.055, c["note"],
                ha="left", va="center", fontsize=6.7, color=MUTE, style="italic")

    for y_top, y_bot in [(0.68, 0.62), (0.38, 0.32)]:
        draw_arrow(ax, W * 0.10, y_top, W * 0.10, y_bot,
                   color=MUTE, lw=0.8, mutation_scale=7, style="-|>")


# ── Panel D: broadcast modes ────────────────────────────────────────────
def _mini_tree(ax, cx, cy, scale=0.06, color_soma=SOMA):
    """Draw a mini dendritic tree at (cx, cy): soma on right, two leaves on left."""
    soma = (cx + 0.0, cy)
    hub  = (cx - scale * 0.9, cy)
    leaves = [(cx - scale * 1.8, cy + scale * 0.75),
              (cx - scale * 1.8, cy - scale * 0.75)]

    # Wires
    draw_wire(ax, hub[0], hub[1], soma[0] - scale * 0.4, soma[1],
              color=DEND, lw=1.1, alpha=0.82)
    for leaf in leaves:
        draw_wire(ax, leaf[0], leaf[1], hub[0], hub[1],
                  color=DEND, lw=0.9, alpha=0.78)

    # Nodes — all CIRCLES so they stay round
    for leaf in leaves:
        ax.add_patch(Circle(leaf, scale * 0.22, fc=DEND, ec=EDGE,
                            linewidth=0.4, alpha=0.92, zorder=3))
    ax.add_patch(Circle(hub, scale * 0.25, fc=DEND, ec=EDGE,
                        linewidth=0.4, alpha=1.0, zorder=3))
    ax.add_patch(Circle(soma, scale * 0.40, fc=color_soma, ec=EDGE,
                        linewidth=0.5, alpha=0.95, zorder=4))
    return dict(soma=soma, hub=hub, leaves=leaves, scale=scale)


def panel_D(ax, W=1.20):
    setup_panel(ax, W=W, H=1.0, title="Feedback channels for $e$")

    # Header strip indicating the structure
    hdr_y = 0.965
    ax.text(W * 0.085, hdr_y, "source",       fontsize=7.0, color=MUTE,
            ha="center", va="center", fontweight="bold")
    ax.text(W * 0.40,  hdr_y, "channels",     fontsize=7.0, color=MUTE,
            ha="center", va="center", fontweight="bold")
    ax.text(W * 0.82,  hdr_y, "target neurons", fontsize=7.0, color=MUTE,
            ha="center", va="center", fontweight="bold")

    rows = [
        dict(label="scalar",   sub="1 shared field",            color=COLORS["scalar"],   y=0.80, kind="scalar"),
        dict(label="per-soma", sub="one $e_n$ per soma",         color=COLORS["per_soma"], y=0.58, kind="per_soma"),
        dict(label="low-rank", sub="$K{=}2$ mixed channels",     color=COLORS["low_rank"], y=0.36, kind="low_rank"),
        dict(label="pathway",  sub="branch-specific $e^{(p)}$",  color=COLORS["pathway"],  y=0.14, kind="pathway"),
    ]

    SOURCE_X = W * 0.085
    CHAN_X   = W * 0.40
    SOMA_X   = W * 0.82            # single x — two somas STACKED vertically here
    # Vertical offset of the two somas within each row
    SOMA_DY  = 0.045

    for r in rows:
        y = r["y"]
        # Row label on far-left, vertically centred on the row
        ax.text(0.005, y + 0.020, r["label"], ha="left", va="center",
                fontsize=8.2, color=r["color"], fontweight="bold")
        ax.text(0.005, y - 0.035, r["sub"], ha="left", va="center",
                fontsize=6.3, color=MUTE, style="italic")

        # Two target somas stacked VERTICALLY at the same x
        t_top = _mini_tree(ax, SOMA_X, y + SOMA_DY, scale=0.022)
        t_bot = _mini_tree(ax, SOMA_X, y - SOMA_DY, scale=0.022)

        tgt_top = (t_top["soma"][0] - 0.011, t_top["soma"][1])
        tgt_bot = (t_bot["soma"][0] - 0.011, t_bot["soma"][1])

        if r["kind"] == "scalar":
            ax.add_patch(Circle((SOURCE_X, y), 0.015,
                                fc=r["color"], ec="white", linewidth=0.4, zorder=7))
            ax.text(SOURCE_X, y, r"$e$", ha="center", va="center",
                    fontsize=6.2, color="white", fontweight="bold", zorder=8)
            # Source → both somas
            for tx, ty in [tgt_top, tgt_bot]:
                draw_arrow(ax, SOURCE_X + 0.015, y, tx, ty,
                           color=r["color"], lw=0.95, mutation_scale=6,
                           alpha=0.88,
                           connectionstyle=f"arc3,rad={0.08 if ty > y else -0.08}")

        elif r["kind"] == "per_soma":
            # Two separate sources stacked vertically, each → its soma
            for (ty, (tx, _ty)) in zip([y + SOMA_DY, y - SOMA_DY], [tgt_top, tgt_bot]):
                ax.add_patch(Circle((SOURCE_X, ty), 0.012,
                                    fc=r["color"], ec="white",
                                    linewidth=0.4, zorder=7))
                draw_arrow(ax, SOURCE_X + 0.012, ty, tx, ty,
                           color=r["color"], lw=0.9, mutation_scale=6,
                           alpha=0.9)
            ax.text(SOURCE_X, y + SOMA_DY + 0.035, r"$e_1$",
                    ha="center", fontsize=6.0, color=r["color"], fontweight="bold")
            ax.text(SOURCE_X, y - SOMA_DY - 0.035, r"$e_2$",
                    ha="center", fontsize=6.0, color=r["color"], fontweight="bold")

        elif r["kind"] == "low_rank":
            # single source → K=2 stacked channels → all-to-all to stacked somas
            ax.add_patch(Circle((SOURCE_X, y), 0.014, fc=r["color"],
                                ec="white", linewidth=0.4, zorder=7))
            channels = [(CHAN_X, y + 0.035), (CHAN_X, y - 0.035)]
            for i, (cx, cy) in enumerate(channels):
                ax.add_patch(Circle((cx, cy), 0.011, fc=r["color"],
                                    ec="white", linewidth=0.4, zorder=7))
                draw_arrow(ax, SOURCE_X + 0.014, y, cx - 0.011, cy,
                           color=r["color"], lw=0.8, mutation_scale=5, alpha=0.85)
                for (tx, ty) in [tgt_top, tgt_bot]:
                    rad = 0.10 if ty > cy else (-0.10 if ty < cy else 0)
                    draw_arrow(ax, cx + 0.011, cy, tx, ty,
                               color=r["color"], lw=0.7, mutation_scale=5,
                               alpha=0.65,
                               connectionstyle=f"arc3,rad={rad}")
            # Channel labels to the LEFT of the channel dots (less overlap)
            ax.text(CHAN_X - 0.030, channels[0][1], r"$c_1$",
                    ha="right", va="center",
                    fontsize=6.2, color=r["color"], fontweight="bold")
            ax.text(CHAN_X - 0.030, channels[1][1], r"$c_2$",
                    ha="right", va="center",
                    fontsize=6.2, color=r["color"], fontweight="bold")

        else:  # pathway
            # 4 stacked channels — each → a specific distal compartment across the 2 trees
            channel_colors = ["#7C5AA6", "#C15A8A", "#D08C2F", "#4EAE91"]
            ax.add_patch(Circle((SOURCE_X, y), 0.014,
                                fc=r["color"], ec="white",
                                linewidth=0.4, zorder=7))
            # Order leaves top-to-bottom across the two stacked trees
            leaves_ordered = [
                t_top["leaves"][0],   # top tree, upper leaf
                t_top["leaves"][1],   # top tree, lower leaf
                t_bot["leaves"][0],   # bottom tree, upper leaf
                t_bot["leaves"][1],   # bottom tree, lower leaf
            ]
            ch_ys = np.linspace(y + 0.070, y - 0.070, 4)
            for i, (cy_, col) in enumerate(zip(ch_ys, channel_colors)):
                ax.add_patch(Circle((CHAN_X, cy_), 0.008, fc=col, ec="white",
                                    linewidth=0.3, zorder=7))
                draw_arrow(ax, SOURCE_X + 0.014, y, CHAN_X - 0.008, cy_,
                           color=col, lw=0.65, mutation_scale=4, alpha=0.78)
                (lx, ly) = leaves_ordered[i]
                draw_arrow(ax, CHAN_X + 0.008, cy_, lx - 0.004, ly,
                           color=col, lw=0.85, mutation_scale=4, alpha=0.9,
                           connectionstyle="arc3,rad=0.08")
            ax.text(CHAN_X, y + 0.110, r"$e^{(p)}$ per branch",
                    ha="center", va="bottom", fontsize=6.0,
                    color=r["color"], fontweight="bold")

    # Key below the panel (circle legend) — add a third entry for the source blob
    key_y = 0.005
    ax.add_patch(Circle((0.04, key_y), 0.011, fc=SOMA, ec=EDGE,
                        linewidth=0.4, zorder=3))
    ax.text(0.06, key_y, "= soma", fontsize=6.2, va="center", color=INK)
    ax.add_patch(Circle((0.22, key_y), 0.011, fc=DEND, ec=EDGE,
                        linewidth=0.4, zorder=3))
    ax.text(0.24, key_y, "= dendritic compartment",
            fontsize=6.2, va="center", color=INK)


# ── Panel E: CIFAR-10 per-epoch test-accuracy curves ────────────────────
CIFAR_SWEEP_ROOT = Path(
    "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/"
    "dendritic-local-learning/local_sweep_runs"
)
CIFAR_LOCAL_SWEEP = (
    CIFAR_SWEEP_ROOT
    / "cifar10_compactei_depth4_decoderfix_mechanism_5seed_20260409110234"
    / "results"
)
CIFAR_BP_SWEEP = (
    CIFAR_SWEEP_ROOT
    / "cifar10_compactei_depth4_standard_ceiling_5seed_20260408131458"
    / "results"
)
# The BP sweep sits under the "Lab" mirror in the CIFAR sweep_runs; if the local
# path doesn't exist, fall back to the shared scratch mirror.
_BP_SWEEP_ALT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "sweep_runs/cifar10_compactei_depth4_standard_ceiling_5seed_20260408131458/"
    "results"
)


def _cifar_run_dirs_by_prefix(sweep_root: Path, prefix: str) -> list[Path]:
    """Return config dirs whose run_name starts with `prefix`."""
    import json as _json
    out: list[Path] = []
    if not sweep_root.exists():
        return out
    for cfg in sorted(sweep_root.glob("config_*")):
        cj = cfg / "config.json"
        if not cj.exists():
            continue
        try:
            with open(cj) as fh:
                payload = _json.load(fh)
        except Exception:
            continue
        rn = payload.get("outputs", {}).get("run_name", "")
        if rn.startswith(prefix):
            out.append(cfg)
    return out


def _cifar_average_curve(run_dirs: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-epoch test-accuracy curves from a list of run dirs, align on
    epoch, and return (epochs, mean_acc%, std_acc%).
    """
    import json as _json
    per_run: list[pd.DataFrame] = []
    for run in run_dirs:
        ep_dir = run / "performance" / "epochs"
        if not ep_dir.exists():
            continue
        rows = []
        for ep_file in sorted(ep_dir.glob("epoch*.json"),
                              key=lambda p: int(p.stem.replace("epoch", ""))):
            try:
                with open(ep_file) as fh:
                    payload = _json.load(fh)
                ep_num = int(ep_file.stem.replace("epoch", ""))
                acc = float(payload["accuracy"]["test"])
                rows.append((ep_num, acc))
            except Exception:
                continue
        if rows:
            per_run.append(pd.DataFrame(rows, columns=["epoch", "test_acc"]).sort_values("epoch"))
    if not per_run:
        return np.array([]), np.array([]), np.array([])
    min_n = min(len(df) for df in per_run)
    stacked = np.stack([df.iloc[:min_n]["test_acc"].to_numpy() for df in per_run])
    epochs = per_run[0].iloc[:min_n]["epoch"].to_numpy()
    return epochs, 100.0 * stacked.mean(axis=0), 100.0 * stacked.std(axis=0)


def panel_E(ax):
    """CIFAR-10 per-epoch test-accuracy curves across broadcast modes (5 seeds)."""
    # Resolve BP sweep (prefer local path, fall back to shared mirror).
    bp_sweep = CIFAR_BP_SWEEP if CIFAR_BP_SWEEP.exists() else _BP_SWEEP_ALT

    # Build series specs: (label, prefix, sweep_root, color, linestyle)
    series = [
        ("Shunt. BP (ceiling)",
         "cifar10_shunting_standard",      bp_sweep, "#185A33",              "-"),
        ("Shunt. 5F path-trans.",
         "cifar10_shunting_5f_path_transport_bpdec_wd0", CIFAR_LOCAL_SWEEP,
         COLORS["pathway"],  "-"),
        ("Shunt. 5F low-rank 4",
         "cifar10_shunting_5f_low_rank4_bpdec_wd0",     CIFAR_LOCAL_SWEEP,
         COLORS["low_rank"], "--"),
        ("Shunt. 5F per-soma",
         "cifar10_shunting_5f_per_soma_bpdec_wd0",      CIFAR_LOCAL_SWEEP,
         COLORS["per_soma"], ":"),
        ("Add. BP (ceiling)",
         "cifar10_additive_standard",      bp_sweep, "#9A503B",              "-"),
        ("Add. 5F path-trans.",
         "cifar10_additive_5f_path_transport_bpdec_wd0", CIFAR_LOCAL_SWEEP,
         COLORS["additive"], "-."),
    ]

    for label, prefix, sweep_root, color, ls in series:
        run_dirs = _cifar_run_dirs_by_prefix(sweep_root, prefix)
        if not run_dirs:
            continue
        epochs, mean, std = _cifar_average_curve(run_dirs)
        if epochs.size == 0:
            continue
        ax.plot(epochs, mean, color=color, lw=1.4, ls=ls,
                label=label, alpha=0.95, zorder=4)
        ax.fill_between(epochs, mean - std, mean + std,
                        color=color, alpha=0.12, linewidth=0, zorder=2)

    ax.axhline(10, color=MUTE, ls=":", lw=0.7, alpha=0.55, zorder=0)
    ax.text(ax.get_xlim()[1] * 0.98 if ax.has_data() else 100, 11.5,
            "chance", color=MUTE, fontsize=6.6, ha="right", va="bottom")

    ax.set_xlabel("epoch")
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_title("CIFAR-10 dynamics (5 seeds, 5F rule)",
                 fontsize=9.5, pad=6, loc="left", x=0.02)
    ax.set_ylim(0, 58)
    style_axis(ax, grid="y")
    ax.legend(loc="lower right", fontsize=6.3, ncol=1,
              handlelength=1.8, handletextpad=0.5, labelspacing=0.22,
              borderaxespad=0.5, frameon=True, framealpha=0.92,
              facecolor="white", edgecolor="#DDDDDD")

    # Small footer note — 3F/4F/5F comparison is elsewhere (MNIST appendix);
    # CIFAR only has 5F runs at non-chance accuracy in our sweeps.
    ax.text(0.015, 0.02,
            "3F/4F/5F comparison: see MNIST appendix (CIFAR has only 5F at $>$ chance)",
            transform=ax.transAxes, fontsize=5.8, color=MUTE, style="italic",
            ha="left", va="bottom")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Four conceptual panels in a single row. This is easier to scan in the
    # main text and avoids mixing schematic definitions with headline results.
    fig = plt.figure(figsize=(15.2, 3.55))
    gs = fig.add_gridspec(
        1, 4,
        width_ratios=[1.18, 2.35, 1.24, 1.35],
        wspace=0.13,
        left=0.025, right=0.992, top=0.84, bottom=0.12,
    )

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[0, 3])

    panel_A(ax_A, W=1.20)
    panel_B(ax_B, W=2.55)
    panel_C(ax_C, W=1.20)
    panel_D(ax_D, W=1.20)

    for ax, lbl, x_off in [
        (ax_A, "A", -0.10),
        (ax_B, "B", -0.075),
        (ax_C, "C", -0.10),
        (ax_D, "D", -0.10),
    ]:
        panel_label(ax, lbl, x=x_off, y=1.10, fontsize=12)

    out = OUTPUT_DIR / "fig1_model_and_credit"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved: {out}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Figure 1: dendritic neuron, exact credit vs LocalCA broadcast,
local-rule family, broadcast channels, and CIFAR-10 mechanism evidence.

Outputs:
  - figures/fig1_model_and_credit.{pdf,png}

Design principles:
  - Use aspect='equal' on every schematic so circles stay circular.
  - Use one coordinate frame per panel that matches the panel's physical
    aspect ratio, so content fills the panel without horizontal waste.
  - Show error feedback reaching BOTH proximal and distal compartments.
  - Panel D uses somas (orange) as broadcast targets, each with a
    small dendritic tree, so channel structure is unambiguous.
  - Panel E shows CIFAR-10 bars (harder task separates broadcast modes
    more cleanly than MNIST, which all variants solve).
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
        ax.set_title(title, fontsize=10.0, pad=6, loc="left", x=title_x)


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
                title="Dendritic neuron with shunting E/I integration")

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
                title="Credit assignment: backprop vs. LocalCA",
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
    setup_panel(ax, W=W, H=1.0, title="Broadcast channels for $e$")

    # Header strip indicating the structure (pushed up so rows don't collide)
    hdr_y = 0.965
    ax.text(W * 0.08, hdr_y, "source",     fontsize=7.0, color=MUTE,
            ha="center", va="center", fontweight="bold")
    ax.text(W * 0.36, hdr_y, "channels",   fontsize=7.0, color=MUTE,
            ha="center", va="center", fontweight="bold")
    ax.text(W * 0.75, hdr_y, "target somas", fontsize=7.0, color=MUTE,
            ha="center", va="center", fontweight="bold")

    rows = [
        dict(label="scalar",   sub="1 shared field",         color=COLORS["scalar"],   y=0.78, kind="scalar"),
        dict(label="per-soma", sub="one $e_n$ per soma",     color=COLORS["per_soma"], y=0.55, kind="per_soma"),
        dict(label="low-rank", sub="$K{=}2$ mixed channels", color=COLORS["low_rank"], y=0.32, kind="low_rank"),
        dict(label="pathway",  sub="branch-specific $e^{(p)}$", color=COLORS["pathway"], y=0.09, kind="pathway"),
    ]

    SOURCE_X = W * 0.08
    CHAN_X   = W * 0.36
    SOMA_X1  = W * 0.72
    SOMA_X2  = W * 0.92

    for r in rows:
        y = r["y"]
        # Label bundle on far-left (below header row, offset so nothing overlaps)
        ax.text(0.005, y + 0.035, r["label"], ha="left", va="center",
                fontsize=8.2, color=r["color"], fontweight="bold")
        ax.text(0.005, y - 0.040, r["sub"], ha="left", va="center",
                fontsize=6.4, color=MUTE, style="italic")

        # Two target somas (with mini trees) per row
        t1 = _mini_tree(ax, SOMA_X1, y, scale=0.035)
        t2 = _mini_tree(ax, SOMA_X2, y, scale=0.035)

        # Arrow-target = soma left edge
        tgt1 = (t1["soma"][0] - 0.014, t1["soma"][1])
        tgt2 = (t2["soma"][0] - 0.014, t2["soma"][1])

        if r["kind"] == "scalar":
            ax.add_patch(Circle((SOURCE_X, y), 0.016,
                                fc=r["color"], ec="white", linewidth=0.4, zorder=7))
            for tx, ty in [tgt1, tgt2]:
                draw_arrow(ax, SOURCE_X + 0.016, y, tx, ty,
                           color=r["color"], lw=0.95, mutation_scale=7,
                           alpha=0.85,
                           connectionstyle=f"arc3,rad={0.08 if ty > y else -0.08}")

        elif r["kind"] == "per_soma":
            # Two separate sources, one per target soma
            for (sx_off, (tx, ty)) in zip([+0.025, -0.025], [tgt1, tgt2]):
                src = (SOURCE_X, ty + sx_off * 0.2)
                ax.add_patch(Circle(src, 0.012, fc=r["color"], ec="white",
                                    linewidth=0.4, zorder=7))
                draw_arrow(ax, src[0] + 0.012, src[1], tx, ty,
                           color=r["color"], lw=0.9, mutation_scale=6,
                           alpha=0.9,
                           connectionstyle="arc3,rad=0.04")

        elif r["kind"] == "low_rank":
            # single source → K=2 channels → all-to-all to somas
            ax.add_patch(Circle((SOURCE_X, y), 0.014, fc=r["color"],
                                ec="white", linewidth=0.4, zorder=7))
            channels = [(CHAN_X, y + 0.040), (CHAN_X, y - 0.040)]
            for (cx, cy) in channels:
                ax.add_patch(Circle((cx, cy), 0.012, fc=r["color"],
                                    ec="white", linewidth=0.4, zorder=7))
                draw_arrow(ax, SOURCE_X + 0.014, y, cx - 0.012, cy,
                           color=r["color"], lw=0.8, mutation_scale=6,
                           alpha=0.85)
                for (tx, ty) in [tgt1, tgt2]:
                    draw_arrow(ax, cx + 0.012, cy, tx, ty,
                               color=r["color"], lw=0.7, mutation_scale=5,
                               alpha=0.65,
                               connectionstyle=f"arc3,rad={0.08 if ty > cy else -0.08}")
            ax.text(CHAN_X, y + 0.090, "$c_1, c_2$",
                    ha="center", va="bottom", fontsize=6.2,
                    color=r["color"], fontweight="bold")

        else:  # pathway
            # Each soma's dendritic branches each get their own channel
            # We connect channels directly into the distal leaves of each mini-tree
            channel_colors = ["#7C5AA6", "#C15A8A", "#D08C2F", "#4EAE91"]
            ax.add_patch(Circle((SOURCE_X, y), 0.014,
                                fc=r["color"], ec="white",
                                linewidth=0.4, zorder=7))
            leaves_all = t1["leaves"] + t2["leaves"]
            # Place channels at CHAN_X
            ch_ys = np.linspace(y + 0.055, y - 0.055, 4)
            for i, ((cy_), col) in enumerate(zip(ch_ys, channel_colors)):
                ax.add_patch(Circle((CHAN_X, cy_), 0.009, fc=col, ec="white",
                                    linewidth=0.3, zorder=7))
                # source → channel
                draw_arrow(ax, SOURCE_X + 0.014, y, CHAN_X - 0.010, cy_,
                           color=col, lw=0.65, mutation_scale=5, alpha=0.78)
                # channel → specific leaf
                (lx, ly) = leaves_all[i]
                draw_arrow(ax, CHAN_X + 0.010, cy_, lx - 0.004, ly,
                           color=col, lw=0.85, mutation_scale=5, alpha=0.88,
                           connectionstyle="arc3,rad=0.05")
            ax.text(CHAN_X, y + 0.095, r"$e^{(p)}$ per branch",
                    ha="center", va="bottom", fontsize=6.0,
                    color=r["color"], fontweight="bold")

    # Key below the panel (circle legend)
    key_y = -0.02
    ax.add_patch(Circle((0.05, key_y + 0.03), 0.011, fc=SOMA, ec=EDGE,
                        linewidth=0.4, zorder=3))
    ax.text(0.075, key_y + 0.03, "= soma", fontsize=6.2, va="center",
            color=INK)
    ax.add_patch(Circle((0.24, key_y + 0.03), 0.011, fc=DEND, ec=EDGE,
                        linewidth=0.4, zorder=3))
    ax.text(0.265, key_y + 0.03, "= dendritic compartment",
            fontsize=6.2, va="center", color=INK)


# ── Panel E: CIFAR-10 mechanism bars ─────────────────────────────────────
def panel_E(ax):
    """Swap MNIST curves for CIFAR-10 bars that expose the broadcast-mode
    separation. 5-seed means ± std (from the decoderfix mechanism sweep).
    """
    # Load data
    try:
        dfx = pd.read_csv(CIFAR_DFX_CSV)
        bp = pd.read_csv(CIFAR_BP_CSV)
    except Exception as exc:
        ax.text(0.5, 0.5, f"Data missing:\n{exc}", transform=ax.transAxes,
                ha="center", va="center", fontsize=7, color="red")
        return

    def _local(cond):
        row = dfx[dfx["condition"] == cond].iloc[0]
        return float(row["acc_test_mean"]) * 100.0, float(row["acc_test_std"]) * 100.0

    def _bp(model_type):
        row = bp[(bp["strategy"] == "standard") &
                 (bp["model_type"] == model_type)].iloc[0]
        return float(row["mean_test_accuracy"]) * 100.0, float(row["std_test_accuracy"]) * 100.0

    # Rows: (label, (mean, std), color, group)
    bars = [
        ("Shunt.\nBP",        _bp("dendritic_shunting"),                         "#185A33",            "shunt"),
        ("Shunt.\npath-trans.",_local("cifar10_shunting_5f_path_transport_bpdec_wd0"), COLORS["pathway"],  "shunt"),
        ("Shunt.\nlow-rank 4", _local("cifar10_shunting_5f_low_rank4_bpdec_wd0"),      COLORS["low_rank"], "shunt"),
        ("Shunt.\nper-soma",   _local("cifar10_shunting_5f_per_soma_bpdec_wd0"),       COLORS["per_soma"], "shunt"),
        ("Add.\nBP",          _bp("dendritic_additive"),                         "#9A503B",            "add"),
        ("Add.\npath-trans.",  _local("cifar10_additive_5f_path_transport_bpdec_wd0"), COLORS["pathway"],  "add"),
        ("Add.\nper-soma",     _local("cifar10_additive_5f_per_soma_bpdec_wd0"),       COLORS["per_soma"], "add"),
    ]

    xs = np.arange(len(bars))
    means = np.array([b[1][0] for b in bars])
    stds  = np.array([b[1][1] for b in bars])
    colors = [b[2] for b in bars]
    labels = [b[0] for b in bars]

    # BP bars hatched; local bars solid
    for x, m, s, c, lbl in zip(xs, means, stds, colors, labels):
        if "BP" in lbl:
            ax.bar(x, m, 0.78, yerr=s, color=c, alpha=0.95, edgecolor="white",
                   lw=0.4, capsize=1.8, error_kw={"lw": 0.6}, hatch="//")
        else:
            ax.bar(x, m, 0.78, yerr=s, color=c, alpha=0.92, edgecolor="white",
                   lw=0.4, capsize=1.8, error_kw={"lw": 0.6})
        ax.text(x, m + s + 0.9, f"{m:.1f}",
                ha="center", va="bottom", fontsize=6.6, color=INK)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.4)
    ax.set_ylabel("CIFAR-10 test accuracy (%)")
    ax.set_ylim(0, 58)
    ax.set_title("CIFAR-10 mechanism ($5$ seeds)", fontsize=9.5, pad=6,
                 loc="left", x=0.02)
    style_axis(ax, grid="y")

    # Vertical divider between shunting and additive groups
    ax.axvline(3.5, color="#C0C0C0", ls=":", lw=0.7)
    ax.text(1.5, 54.5, "shunting", ha="center", va="center",
            fontsize=7.0, color=COLORS["shunting"], fontweight="bold")
    ax.text(5.5, 54.5, "additive", ha="center", va="center",
            fontsize=7.0, color=COLORS["additive"], fontweight="bold")

    # Chance line
    ax.axhline(10, color=MUTE, ls=":", lw=0.7, alpha=0.6, zorder=0)
    ax.text(6.45, 10.6, "chance", color=MUTE, fontsize=6.4,
            ha="right", va="bottom")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Wider figure to give Panel B (spans 2 cols) enough horizontal room.
    fig = plt.figure(figsize=(14.4, 8.2))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[1.0, 1.02],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.32, wspace=0.16,
        left=0.035, right=0.985, top=0.935, bottom=0.06,
    )

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1:])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])
    ax_E = fig.add_subplot(gs[1, 2])

    panel_A(ax_A, W=1.20)
    panel_B(ax_B, W=2.60)
    panel_C(ax_C, W=1.20)
    panel_D(ax_D, W=1.20)
    panel_E(ax_E)

    for ax, lbl, x_off in [
        (ax_A, "A", -0.04),
        (ax_B, "B", -0.02),
        (ax_C, "C", -0.05),
        (ax_D, "D", -0.05),
        (ax_E, "E", -0.16),
    ]:
        panel_label(ax, lbl, x=x_off, y=1.12, fontsize=13)

    out = OUTPUT_DIR / "fig1_model_and_credit"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved: {out}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()

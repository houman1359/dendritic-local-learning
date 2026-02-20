#!/usr/bin/env python3
"""Generate Figure 1: Model schematic for the local credit assignment paper.

Creates a two-panel figure:
  (A) Dendritic neuron architecture: soma, dendritic branches, synapses,
      excitatory/inhibitory inputs, and the shunting voltage equation.
  (B) Local learning rule hierarchy: 3F → 4F → 5F with the signals used.

Output: figures/fig_model_schematic.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Style ────────────────────────────────────────────────────────────
EXC_COLOR = "#2166AC"      # Blue - excitatory
INH_COLOR = "#B2182B"      # Red - inhibitory
DEN_COLOR = "#4DAF4A"      # Green - dendritic
SOMA_COLOR = "#FF7F00"     # Orange - soma
BG_COLOR = "#F7F7F7"       # Light gray background
RULE3_COLOR = "#66C2A5"    # Teal - 3F
RULE4_COLOR = "#FC8D62"    # Salmon - 4F
RULE5_COLOR = "#8DA0CB"    # Lavender - 5F

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"


def draw_synapse(ax, x, y, color, label=None, size=0.08):
    """Draw a small filled circle representing a synapse."""
    circle = plt.Circle((x, y), size, fc=color, ec="k", linewidth=0.5, zorder=5)
    ax.add_patch(circle)
    if label:
        ax.text(x, y - size - 0.06, label, ha="center", va="top", fontsize=6, color=color)


def draw_compartment(ax, x, y, w, h, label, color, fontsize=7):
    """Draw a rounded rectangle representing a dendritic compartment."""
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02",
        fc=color, ec="k", linewidth=0.8, alpha=0.3, zorder=3,
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", zorder=6)


def draw_arrow(ax, x1, y1, x2, y2, color="k", style="-|>", lw=1.0):
    """Draw an arrow from (x1,y1) to (x2,y2)."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw),
        zorder=4,
    )


def panel_a(ax):
    """Panel A: Dendritic neuron architecture."""
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.3, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(A) Compartmental dendritic neuron", fontsize=10, fontweight="bold", pad=8)

    # ── Soma ──
    soma_x, soma_y = 3.8, 1.5
    soma = plt.Circle((soma_x, soma_y), 0.28, fc=SOMA_COLOR, ec="k", linewidth=1.2,
                       alpha=0.5, zorder=5)
    ax.add_patch(soma)
    ax.text(soma_x, soma_y, "Soma\n$V_{\\mathrm{out}}$", ha="center", va="center",
            fontsize=7, fontweight="bold", zorder=6)

    # ── Branch layers ──
    # Layer 2 (closer to soma)
    branch2_x, branch2_y = 2.6, 1.5
    draw_compartment(ax, branch2_x, branch2_y, 0.7, 0.5, "$V_{b_2}$", DEN_COLOR, fontsize=8)

    # Layer 1 (distal)
    branch1_positions = [(1.2, 2.5), (1.2, 0.5)]
    for i, (bx, by) in enumerate(branch1_positions):
        draw_compartment(ax, bx, by, 0.7, 0.5, f"$V_{{b_1}}^{{({i+1})}}$", DEN_COLOR, fontsize=8)

    # ── Connections: branches → soma, branches → branches ──
    # Branch2 → soma
    draw_arrow(ax, branch2_x + 0.35, branch2_y, soma_x - 0.28, soma_y,
               color=DEN_COLOR, lw=1.5)
    ax.text(3.15, 1.65, "$g^{\\mathrm{den}}$", fontsize=6, color=DEN_COLOR)

    # Branch1 → Branch2
    for bx, by in branch1_positions:
        draw_arrow(ax, bx + 0.35, by, branch2_x - 0.35,
                   branch2_y + 0.15 * (1 if by > 1.5 else -1),
                   color=DEN_COLOR, lw=1.2)

    # ── Excitatory synapses ──
    exc_positions = [(0.2, 2.8), (0.2, 2.2), (0.2, 0.8), (0.2, 0.2)]
    for i, (sx, sy) in enumerate(exc_positions):
        draw_synapse(ax, sx, sy, EXC_COLOR)
        target = branch1_positions[0] if sy > 1.5 else branch1_positions[1]
        draw_arrow(ax, sx + 0.08, sy, target[0] - 0.35,
                   target[1] + 0.1 * (1 if sy > target[1] else -1),
                   color=EXC_COLOR, lw=0.8)

    ax.text(0.2, 3.15, "Excitatory\ninputs $x_j$", ha="center", va="bottom",
            fontsize=7, color=EXC_COLOR, fontweight="bold")

    # ── Inhibitory synapses ──
    inh_x = 2.6
    inh_positions = [(inh_x, 2.6), (inh_x, 0.4)]
    for sx, sy in inh_positions:
        draw_synapse(ax, sx, sy, INH_COLOR, size=0.07)
        # Inhibitory inputs to branch2 compartment
        draw_arrow(ax, sx, sy + (-0.07 if sy < 1.5 else 0.07),
                   branch2_x, branch2_y + 0.25 * (1 if sy > 1.5 else -1),
                   color=INH_COLOR, lw=0.8)

    ax.text(inh_x + 0.5, 2.7, "Inh $g_i^{\\mathrm{syn}}$", fontsize=6,
            color=INH_COLOR, fontweight="bold")

    # ── Voltage equation (use simpler notation matplotlib can render) ──
    eq_text = (
        r"$V_n = \frac{\sum_j E_j x_j g_j^{\mathrm{syn}} + \sum_j V_j g_j^{\mathrm{den}}}"
        r"{\sum_j x_j g_j^{\mathrm{syn}} + \sum_j g_j^{\mathrm{den}} + 1}$"
    )
    ax.text(2.0, -0.15, eq_text, ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # ── Output arrow ──
    draw_arrow(ax, soma_x + 0.28, soma_y, 4.3, soma_y, color="k", lw=1.5)
    ax.text(4.35, soma_y, "output", fontsize=7, va="center")

    # ── Legend ──
    legend_items = [
        mpatches.Patch(fc=EXC_COLOR, ec="k", label="Excitatory", alpha=0.7),
        mpatches.Patch(fc=INH_COLOR, ec="k", label="Inhibitory", alpha=0.7),
        mpatches.Patch(fc=DEN_COLOR, ec="k", label="Dendritic", alpha=0.7),
        mpatches.Patch(fc=SOMA_COLOR, ec="k", label="Soma", alpha=0.5),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=6, framealpha=0.9)


def panel_b(ax):
    """Panel B: Local learning rule hierarchy (3F → 4F → 5F)."""
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.3, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(B) Local learning rule hierarchy", fontsize=10, fontweight="bold", pad=8)

    # Vertical layout: 3F at top, 5F at bottom (increasing complexity)
    rules = [
        ("3F", 2.8, RULE3_COLOR,
         r"$\Delta g \propto x_j \cdot (E_j - V_n) \cdot \delta$",
         "pre-synaptic  ×  driving force  ×  broadcast error"),
        ("4F", 1.7, RULE4_COLOR,
         r"$\Delta g \propto x_j \cdot (E_j - V_n) \cdot \delta \cdot \rho$",
         "+  variance / EMA modulator  ρ"),
        ("5F", 0.6, RULE5_COLOR,
         r"$\Delta g \propto x_j \cdot (E_j - V_n) \cdot \delta \cdot \rho \cdot \phi$",
         "+  information-theoretic factor  φ"),
    ]

    for name, y_center, color, equation, description in rules:
        # Rule box
        box = FancyBboxPatch(
            (-0.2, y_center - 0.35), 4.8, 0.7,
            boxstyle="round,pad=0.05",
            fc=color, ec="k", linewidth=0.8, alpha=0.15, zorder=2,
        )
        ax.add_patch(box)

        # Rule name
        ax.text(-0.05, y_center, name, ha="left", va="center",
                fontsize=12, fontweight="bold", color=color, zorder=5)

        # Equation
        ax.text(0.55, y_center + 0.05, equation, ha="left", va="center",
                fontsize=8, zorder=5)

        # Description
        ax.text(0.55, y_center - 0.22, description, ha="left", va="center",
                fontsize=6, color="gray", style="italic", zorder=5)

    # Arrows showing hierarchy (3F → 4F → 5F)
    for y_top, y_bot in [(2.45, 2.05), (1.35, 0.95)]:
        ax.annotate(
            "", xy=(0.1, y_bot), xytext=(0.1, y_top),
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.0, ls="--"),
        )

    # Broadcast error box
    broadcast_y = -0.05
    box = FancyBboxPatch(
        (0.5, broadcast_y - 0.15), 3.5, 0.3,
        boxstyle="round,pad=0.03",
        fc=SOMA_COLOR, ec="k", linewidth=0.6, alpha=0.15, zorder=2,
    )
    ax.add_patch(box)
    ax.text(2.25, broadcast_y, "Broadcast error $\\delta$: scalar / per-soma / local mismatch",
            ha="center", va="center", fontsize=6.5, zorder=5)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(12, 4.5),
        gridspec_kw={"width_ratios": [1.1, 1]},
    )

    panel_a(ax_a)
    panel_b(ax_b)

    fig.tight_layout(pad=1.5)

    out_path = OUTPUT_DIR / "fig_model_schematic"
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Saved: {out_path}.{{png,pdf}}")
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate Figure 1: Model schematic for the local credit assignment paper.

Creates a two-panel figure:
  (A) Dendritic neuron architecture: soma, dendritic branches, synapses,
      excitatory/inhibitory inputs, and the shunting voltage equation.
      EVERY branch receives BOTH excitatory AND inhibitory inputs.
  (B) Local learning rule hierarchy: 3F -> 4F -> 5F with the signals used.

Output: figures/fig_model_schematic.{pdf,png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Style ────────────────────────────────────────────────────────────
EXC_COLOR = "#2166AC"      # Blue - excitatory
INH_COLOR = "#B2182B"      # Red - inhibitory
DEN_COLOR = "#4DAF4A"      # Green - dendritic
SOMA_COLOR = "#FF7F00"     # Orange - soma
RULE3_COLOR = "#66C2A5"    # Teal - 3F
RULE4_COLOR = "#FC8D62"    # Salmon - 4F
RULE5_COLOR = "#8DA0CB"    # Lavender - 5F

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"


def draw_synapse(ax, x, y, color, size=0.07):
    """Draw a small filled circle representing a synapse."""
    circle = plt.Circle((x, y), size, fc=color, ec="k", linewidth=0.5, zorder=5)
    ax.add_patch(circle)


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
    """Panel A: Dendritic neuron architecture.

    Architecture (matching the code):
      - Each branch has BOTH excitatory AND inhibitory synapses
      - E synapses: TopKLinear with reversal E_exc > 0, contribute to both
        numerator (E_j * x_j * g_j) and denominator (x_j * g_j)
      - I synapses: TopKLinear with E_inh = 0, contribute only to
        denominator (shunting / divisive normalization)
      - Dendritic conductances (BlockLinear) connect branches to parent
    """
    ax.set_xlim(-0.8, 4.7)
    ax.set_ylim(-0.5, 3.7)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(A) Compartmental dendritic neuron", fontsize=11,
                 fontweight="bold", pad=8)

    # ── Soma ──
    soma_x, soma_y = 3.9, 1.5
    soma = plt.Circle((soma_x, soma_y), 0.28, fc=SOMA_COLOR, ec="k",
                       linewidth=1.2, alpha=0.5, zorder=5)
    ax.add_patch(soma)
    ax.text(soma_x, soma_y, "Soma\n$V_{\\mathrm{out}}$", ha="center",
            va="center", fontsize=7, fontweight="bold", zorder=6)

    # ── Branch compartments ──
    # Layer 2 (proximal, closer to soma)
    b2_x, b2_y = 2.6, 1.5
    draw_compartment(ax, b2_x, b2_y, 0.7, 0.5, "$V_{b_2}$", DEN_COLOR,
                     fontsize=8)

    # Layer 1 (distal, two branches)
    b1_pos = [(1.0, 2.6), (1.0, 0.4)]
    for i, (bx, by) in enumerate(b1_pos):
        draw_compartment(ax, bx, by, 0.7, 0.5,
                         f"$V_{{b_1}}^{{({i+1})}}$", DEN_COLOR, fontsize=8)

    # ── Dendritic conductance arrows: branches → parent ──
    # Branch2 → Soma
    draw_arrow(ax, b2_x + 0.35, b2_y, soma_x - 0.28, soma_y,
               color=DEN_COLOR, lw=1.5)
    ax.text(3.2, 1.72, "$g^{\\mathrm{den}}$", fontsize=6, color=DEN_COLOR)

    # Branch1 → Branch2
    for bx, by in b1_pos:
        dy = 0.15 if by > 1.5 else -0.15
        draw_arrow(ax, bx + 0.35, by, b2_x - 0.35, b2_y + dy,
                   color=DEN_COLOR, lw=1.2)

    # ── External input labels ──
    ax.text(-0.55, 3.45, "Excitatory\ninputs $x_j^E$", ha="center",
            va="bottom", fontsize=7, color=EXC_COLOR, fontweight="bold")
    ax.text(-0.55, -0.35, "Inhibitory\ninputs $x_j^I$", ha="center",
            va="top", fontsize=7, color=INH_COLOR, fontweight="bold")

    # ── Draw E and I synapses on EVERY branch ──
    # Each branch receives both E (blue) and I (red) synapses
    all_branches = b1_pos + [(b2_x, b2_y)]

    for bx, by in all_branches:
        # Excitatory synapses (2 per branch, on left/top side)
        e_offsets = [(-0.55, 0.12), (-0.55, -0.12)]
        for dx, dy in e_offsets:
            sx, sy = bx + dx, by + dy
            draw_synapse(ax, sx, sy, EXC_COLOR, size=0.06)
            draw_arrow(ax, sx + 0.06, sy, bx - 0.35, by + dy * 0.3,
                       color=EXC_COLOR, lw=0.7)

        # Inhibitory synapses (1 per branch, on right/bottom side)
        # Drawn slightly offset to distinguish from E
        i_offsets = [(0.0, 0.38)]
        for dx, dy in i_offsets:
            sx, sy = bx + dx, by + dy
            draw_synapse(ax, sx, sy, INH_COLOR, size=0.055)
            draw_arrow(ax, sx, sy - 0.055, bx, by + 0.25,
                       color=INH_COLOR, lw=0.7)

    # (input wiring lines omitted for clarity; labels indicate shared input)

    # ── Annotation: shunting mechanism ──
    # Small annotation near proximal branch
    ax.annotate(
        "shunting:\nI enters\ndenominator",
        xy=(b2_x + 0.05, b2_y + 0.38), xytext=(b2_x + 0.65, b2_y + 0.95),
        fontsize=5, color=INH_COLOR, ha="center",
        arrowprops=dict(arrowstyle="->", color=INH_COLOR, lw=0.6),
    )

    # ── Voltage equation ──
    eq_text = (
        r"$V_n = \frac{\sum_j E_j x_j g_j^{\mathrm{syn}} + \sum_j V_j g_j^{\mathrm{den}}}"
        r"{\sum_j x_j g_j^{\mathrm{syn}} + \sum_j g_j^{\mathrm{den}} + 1}$"
    )
    ax.text(2.0, -0.35, eq_text, ha="center", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                      alpha=0.9))
    # Note: sums include both E (E_j>0) and I (E_j=0) synapses
    ax.text(2.0, -0.75, "(sums include both E and I synapses per branch)",
            ha="center", va="top", fontsize=6.0, color="gray",
            style="italic")

    # ── Output arrow ──
    draw_arrow(ax, soma_x + 0.28, soma_y, 4.5, soma_y, color="k", lw=1.5)
    ax.text(4.55, soma_y, "output", fontsize=7, va="center")

    # ── Legend ──
    legend_items = [
        mpatches.Patch(fc=EXC_COLOR, ec="k", label="Excitatory ($E_j > 0$)",
                       alpha=0.7),
        mpatches.Patch(fc=INH_COLOR, ec="k", label="Inhibitory ($E_j = 0$)",
                       alpha=0.7),
        mpatches.Patch(fc=DEN_COLOR, ec="k", label="Dendritic cond.",
                       alpha=0.7),
        mpatches.Patch(fc=SOMA_COLOR, ec="k", label="Soma", alpha=0.5),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=6.2,
              framealpha=0.9)


def panel_b(ax):
    """Panel B: Local learning rule hierarchy (3F -> 4F -> 5F)."""
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-0.3, 3.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(B) Local learning rule hierarchy", fontsize=11,
                 fontweight="bold", pad=8)

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
        box = FancyBboxPatch(
            (-0.2, y_center - 0.35), 4.8, 0.7,
            boxstyle="round,pad=0.05",
            fc=color, ec="k", linewidth=0.8, alpha=0.15, zorder=2,
        )
        ax.add_patch(box)
        ax.text(-0.05, y_center, name, ha="left", va="center",
                fontsize=12, fontweight="bold", color=color, zorder=5)
        ax.text(0.55, y_center + 0.05, equation, ha="left", va="center",
                fontsize=8, zorder=5)
        ax.text(0.55, y_center - 0.22, description, ha="left", va="center",
                fontsize=6, color="gray", style="italic", zorder=5)

    # Arrows: 3F → 4F → 5F
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
    ax.text(2.25, broadcast_y,
            "Broadcast error $\\delta$: scalar / per-soma / local mismatch",
            ha="center", va="center", fontsize=6.5, zorder=5)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13.4, 5.0),
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

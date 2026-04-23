#!/usr/bin/env python3
"""Generate a supplementary figure that visually describes each synthetic task.

Outputs:
  - figures/fig_s_task_schematics.{pdf,png}

Tasks:
  A. Context gating   — right half is the digit; left half is low-amplitude noise
  B. Noise resilience — structured Gaussian channel noise added to MNIST
  C. Cue integration  — a "context / cue" vector + a "conflict" vector, 2-way decision
  D. Fashion-MNIST    — category label set (apparel items) vs. MNIST digits

Each panel shows one representative example image + a short one-line description.
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
)

apply_neurips_style()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "figures"

INK   = COLORS["ink"]
MUTE  = COLORS["mute"]
EDGE  = COLORS["edge"]
EXC   = COLORS["exc"]
INH   = COLORS["inh"]
SHUNT = COLORS["shunting"]
ADD   = COLORS["additive"]
CUE_A = "#3A7CA5"
CUE_B = "#C15A5A"


def _digit_3(canvas, y0, x0, h=10, w=8, color=1.0):
    """Paint a stylised digit "3" onto canvas using a small bitmap."""
    pattern = np.array([
        [0,1,1,1,1,1,0,0],
        [1,1,0,0,0,1,1,0],
        [0,0,0,0,0,1,1,0],
        [0,0,0,0,1,1,0,0],
        [0,0,0,1,1,1,0,0],
        [0,0,0,0,1,1,0,0],
        [0,0,0,0,0,1,1,0],
        [0,0,0,0,0,1,1,0],
        [1,1,0,0,0,1,1,0],
        [0,1,1,1,1,1,0,0],
    ], dtype=float)
    canvas[y0:y0 + h, x0:x0 + w] = np.maximum(canvas[y0:y0 + h, x0:x0 + w],
                                              pattern * color)


def _digit_7(canvas, y0, x0, h=10, w=8, color=1.0):
    pattern = np.array([
        [1,1,1,1,1,1,1,0],
        [0,0,0,0,0,1,1,0],
        [0,0,0,0,1,1,0,0],
        [0,0,0,1,1,0,0,0],
        [0,0,1,1,0,0,0,0],
        [0,1,1,0,0,0,0,0],
        [0,1,1,0,0,0,0,0],
        [1,1,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0],
    ], dtype=float)
    canvas[y0:y0 + h, x0:x0 + w] = np.maximum(canvas[y0:y0 + h, x0:x0 + w],
                                              pattern * color)


def panel_A_context_gating(ax):
    """Context gating: right half carries digit, left half is uniform low-amplitude noise."""
    ax.set_title("Context gating", fontsize=9.5, pad=6, loc="left", x=0.00)

    # Build 28×28 canvas: left half noise, right half digit
    rng = np.random.default_rng(42)
    canvas = np.zeros((28, 28), dtype=float)
    # Left half: uniform noise in [0, 0.25]
    canvas[:, :14] = rng.uniform(0.0, 0.25, size=(28, 14))
    # Right half: draw a "3" centered
    _digit_3(canvas, y0=9, x0=17, h=10, w=8, color=0.95)

    ax.imshow(canvas, cmap="Greys", vmin=0, vmax=1.0,
              interpolation="nearest", aspect="equal")
    # Outline left/right halves
    ax.axvline(13.5, color=CUE_A, linestyle="--", lw=1.0, alpha=0.8)
    ax.add_patch(Rectangle((-0.5, -0.5), 14, 28, fill=False,
                           ec=CUE_A, lw=1.0, alpha=0.9))
    ax.add_patch(Rectangle((13.5, -0.5), 14, 28, fill=False,
                           ec=CUE_B, lw=1.0, alpha=0.9))

    ax.text(6.5, 30.5, "context (noise)", ha="center", va="top",
            fontsize=7.4, color=CUE_A, fontweight="bold")
    ax.text(20.5, 30.5, "digit signal", ha="center", va="top",
            fontsize=7.4, color=CUE_B, fontweight="bold")

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.text(13.5, 36, "right half carries the label; left is low-amplitude noise",
            ha="center", va="top", fontsize=7.0, color=MUTE, style="italic")


def panel_B_noise_resilience(ax):
    """Structured Gaussian channel noise added to MNIST."""
    ax.set_title("Noise resilience", fontsize=9.5, pad=6, loc="left", x=0.00)

    rng = np.random.default_rng(7)
    # Base MNIST-like image
    clean = np.zeros((28, 28))
    _digit_7(clean, y0=9, x0=10, h=10, w=8, color=0.95)

    # Structured channel noise: a 50-D latent projected, clipped
    latent = rng.normal(0, 1, 50)
    proj = rng.normal(0, 0.04, size=(28 * 28, 50))
    noisy = clean + (proj @ latent).reshape(28, 28)
    noisy = np.clip(noisy, 0, 1)

    ax.imshow(noisy, cmap="Greys", vmin=0, vmax=1.0,
              interpolation="nearest", aspect="equal")
    ax.add_patch(Rectangle((-0.5, -0.5), 28, 28, fill=False,
                           ec=EDGE, lw=0.8))

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.text(13.5, 30.5, "digit + structured channel noise",
            ha="center", va="top", fontsize=7.4, color=INK, fontweight="bold")
    ax.text(13.5, 36, "fixed 50-D latent projection, $\\sigma_{\\mathrm{task}}=1.5$, clipped",
            ha="center", va="top", fontsize=7.0, color=MUTE, style="italic")


def panel_C_cue_integration(ax):
    """Cue integration: context + cue + conflict → binary decision."""
    clean_schematic_axis(ax)
    ax.set_aspect("auto")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Cue integration", fontsize=9.5, pad=6, loc="left", x=0.00)

    # Layout: 3 inputs on left, → mixture → soma → 2-way decision on right
    # Input bundles
    def input_bundle(y, color, label, vals):
        # bundle box
        ax.add_patch(FancyBboxPatch((0.04, y - 0.10), 0.22, 0.20,
                                    boxstyle="round,pad=0.008,rounding_size=0.02",
                                    fc="white", ec=color, lw=1.0))
        ax.text(0.15, y + 0.11, label, ha="center", va="bottom",
                fontsize=7.4, color=color, fontweight="bold")
        # stacked mini-bars representing the vector
        x0 = 0.06
        bar_w = 0.16 / len(vals)
        for i, v in enumerate(vals):
            bx = x0 + i * bar_w
            ax.add_patch(Rectangle((bx, y - 0.06), bar_w * 0.85, 0.12 * v,
                                   fc=color, ec=color, alpha=0.8, lw=0))

    rng = np.random.default_rng(2)
    input_bundle(0.80, CUE_A, "context cue", rng.uniform(0.2, 0.9, 8))
    input_bundle(0.50, "#6D8B3D", "reliable cue", rng.uniform(0.3, 1.0, 8))
    input_bundle(0.20, "#9B6FB0", "conflict cue", rng.uniform(0.1, 0.7, 8))

    # Mixer (dendrite) box — narrower so the soma sits clear
    ax.add_patch(FancyBboxPatch((0.34, 0.38), 0.18, 0.24,
                                boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc="#F4F8F5", ec=COLORS["dend"], lw=1.0))
    ax.text(0.43, 0.50, "dendritic\nintegration",
            ha="center", va="center", fontsize=7.6, color=INK, fontweight="bold")

    # Arrows from inputs to mixer
    for (y, col) in [(0.80, CUE_A), (0.50, "#6D8B3D"), (0.20, "#9B6FB0")]:
        ax.add_patch(FancyArrowPatch(
            (0.26, y), (0.34, 0.50 + (y - 0.50) * 0.35),
            arrowstyle="-|>", mutation_scale=9, color=col, lw=1.1, alpha=0.9,
            shrinkA=0, shrinkB=0,
        ))

    # Soma (readout) — placed clear of mixer box; arrow stops short of circle
    soma_xy = (0.65, 0.50)
    soma_r = 0.048
    # Arrow from mixer to soma (stop before the soma so the label stays clear)
    ax.add_patch(FancyArrowPatch(
        (0.52, 0.50), (soma_xy[0] - soma_r - 0.010, 0.50),
        arrowstyle="-|>", mutation_scale=9, color=INK, lw=1.1,
        shrinkA=0, shrinkB=0, zorder=2,
    ))
    ax.add_patch(Circle(soma_xy, soma_r, fc=COLORS["soma"], ec=EDGE,
                        lw=0.9, zorder=4))
    ax.text(soma_xy[0], soma_xy[1], "soma", ha="center", va="center",
            fontsize=7.4, color="white", fontweight="bold", zorder=5)

    # Binary decision — placed further right
    for i, (tx, ty, label, col) in enumerate([
        (0.90, 0.72, "class 0", CUE_A),
        (0.90, 0.28, "class 1", CUE_B),
    ]):
        ax.add_patch(FancyBboxPatch((tx - 0.08, ty - 0.045), 0.16, 0.09,
                                    boxstyle="round,pad=0.005,rounding_size=0.015",
                                    fc="white", ec=col, lw=1.0))
        ax.text(tx, ty, label, ha="center", va="center",
                fontsize=7.4, color=col, fontweight="bold")
        ax.add_patch(FancyArrowPatch(
            (soma_xy[0] + soma_r, 0.50), (tx - 0.08, ty),
            arrowstyle="-|>", mutation_scale=8, color=MUTE, lw=0.9, alpha=0.8,
            shrinkA=0, shrinkB=0,
        ))

    ax.text(0.5, 0.05,
            "route reliable cue to target despite context-dependent conflict",
            ha="center", va="bottom", fontsize=7.0, color=MUTE, style="italic")


def panel_D_fashion_mnist(ax):
    """Fashion-MNIST: category label examples."""
    ax.set_title("Fashion-MNIST", fontsize=9.5, pad=6, loc="left", x=0.00)

    # Build a small composite: 2×3 grid of synthetic apparel-like icons
    rng = np.random.default_rng(11)
    canvas = np.ones((28, 28)) * 0.0

    # A simplified T-shirt silhouette
    # shoulders
    for row in range(6, 9):
        canvas[row, 6:22] = 0.85
    # sleeves
    canvas[9:12, 4:8] = 0.85
    canvas[9:12, 20:24] = 0.85
    # body
    canvas[9:22, 8:20] = 0.85
    # slight noise texture
    canvas += rng.uniform(0, 0.1, size=(28, 28))
    canvas = np.clip(canvas, 0, 1)

    ax.imshow(canvas, cmap="Greys", vmin=0, vmax=1.0,
              interpolation="nearest", aspect="equal")
    ax.add_patch(Rectangle((-0.5, -0.5), 28, 28, fill=False, ec=EDGE, lw=0.8))

    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    ax.text(13.5, 30.5, "10-class apparel classification",
            ha="center", va="top", fontsize=7.4, color=INK, fontweight="bold")
    ax.text(13.5, 36, "same input shape as MNIST; harder decision boundary",
            ha="center", va="top", fontsize=7.0, color=MUTE, style="italic")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1, 4, figsize=(12.5, 3.8),
        gridspec_kw={"wspace": 0.22},
    )

    panel_A_context_gating(axes[0])
    panel_B_noise_resilience(axes[1])
    panel_C_cue_integration(axes[2])
    panel_D_fashion_mnist(axes[3])

    for ax, lbl in zip(axes, ["A", "B", "C", "D"]):
        panel_label(ax, lbl, x=-0.05, y=1.15, fontsize=12)

    plt.subplots_adjust(left=0.03, right=0.985, top=0.87, bottom=0.14)
    out = OUTPUT_DIR / "fig_s_task_schematics"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=300)
    print(f"Saved: {out}.{{pdf,png}}")
    plt.close(fig)


if __name__ == "__main__":
    main()

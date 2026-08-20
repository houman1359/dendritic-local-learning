#!/usr/bin/env python3
"""Render the irregular-tree multiscale route-field analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_REF,
    PANEL_LABEL_PT,
    PT_ANNOT,
    PT_LEGEND,
    SEED_ALPHA,
    SEED_MS,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "irregular_tree_wavelets"
FIGURE = ROOT / "figures" / "generated" / "fig_irregular_tree_wavelets.pdf"

SCALES = ["coarse", "intermediate", "fine"]
SCALE_LABELS = ["coarse\n$>1/4$", "intermediate\n$1/16$--$1/4$", "fine\n$\leq1/16$"]
SCALE_COLORS = [COLORS["shunting"], COLORS["additive"], COLORS["highlight"]]


def compact_panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    """Place the panel letter beside, not across, a vertical axis label."""

    artist = panel_title(ax, letter, title)
    artist.set_visible(False)
    ax.set_title(title, loc="left", x=0.09, pad=8)
    ax.text(0.0, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=PANEL_LABEL_PT, fontweight="bold", color=COLORS["ink"],
            clip_on=False)


def draw_tree_basis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    compact_panel_title(ax, "A", "Irregular-tree scale basis")
    points = {
        0: (0.50, 0.10), 1: (0.48, 0.26), 2: (0.28, 0.45), 3: (0.68, 0.43),
        4: (0.16, 0.68), 5: (0.38, 0.67), 6: (0.59, 0.65), 7: (0.80, 0.65),
        8: (0.10, 0.88), 9: (0.22, 0.86), 10: (0.34, 0.89), 11: (0.44, 0.84),
        12: (0.55, 0.87), 13: (0.64, 0.84), 14: (0.77, 0.88), 15: (0.88, 0.83),
    }
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7),
             (4, 8), (4, 9), (5, 10), (5, 11), (6, 12), (6, 13), (7, 14), (7, 15)]
    for parent, child in edges:
        x0, y0 = points[parent]
        x1, y1 = points[child]
        color = COLORS["mute"]
        width = LW_REF + 0.45
        if child in {3, 6, 7, 12, 13, 14, 15}:
            color, width = SCALE_COLORS[0], LW_DATA + 0.5
        if child in {5, 10, 11}:
            color, width = SCALE_COLORS[1], LW_DATA + 0.3
        if child in {8, 9}:
            color, width = SCALE_COLORS[2], LW_DATA + 0.2
        ax.plot([x0, x1], [y0, y1], color=color, lw=width,
                solid_capstyle="round", transform=ax.transAxes)
    ax.add_patch(Circle(points[0], 0.038, transform=ax.transAxes,
                        fc=COLORS["soma"], ec=COLORS["edge"], lw=0.6))
    ax.text(0.50, 0.01, "nested weighted contrasts", ha="center", va="bottom",
            fontsize=PT_ANNOT, color=COLORS["ink"], transform=ax.transAxes)


def main() -> None:
    apply_neurips_style()
    cell = pd.read_csv(SOURCE / "cell_scale_summary.csv")
    cohort = pd.read_csv(SOURCE / "cohort_scale_summary.csv")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_W, 4.45),
        gridspec_kw={"left": 0.095, "right": 0.985, "bottom": 0.12,
                     "top": 0.90, "wspace": 0.42, "hspace": 0.72},
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    draw_tree_basis(ax_a)

    primary = cohort[cohort.cohort.eq("public_v661_47")].set_index("scale").loc[SCALES]
    x = np.arange(len(SCALES), dtype=float)
    actual = primary.mean_actual_energy_fraction.to_numpy(float)
    low = primary.actual_energy_fraction_ci_low.to_numpy(float)
    high = primary.actual_energy_fraction_ci_high.to_numpy(float)
    shuffled = primary.mean_shuffled_energy_fraction.to_numpy(float)
    isotropic = primary.mean_isotropic_energy_fraction.to_numpy(float)
    ax_b.errorbar(x - 0.10, actual, yerr=[actual - low, high - actual], fmt="o-",
                  color=COLORS["shunting"], lw=LW_DATA, ms=4, capsize=2,
                  label="actual routes")
    ax_b.plot(x + 0.10, shuffled, "s--", color=COLORS["point_mlp"], lw=LW_REF,
              ms=3.6, label="permuted ancestry")
    ax_b.plot(x, isotropic, "_", color=COLORS["oracle"], ms=11,
              markeredgewidth=1.6, label="isotropic noise")
    ax_b.set_xticks(x, SCALE_LABELS)
    ax_b.set_ylim(0, 0.56)
    ax_b.set_ylabel("non-scalar route-energy fraction")
    compact_panel_title(ax_b, "B", "Independent 47-cell cohort")
    style_axis(ax_b, grid="y")
    clean_legend(ax_b, fontsize=PT_LEGEND - 0.7, loc="upper right")

    primary_cell = cell[cell.cohort.eq("public_v661_47")]
    for index, (scale, color) in enumerate(zip(SCALES, SCALE_COLORS, strict=True)):
        values = primary_cell[primary_cell.scale.eq(scale)].isotropic_enrichment.to_numpy(float)
        jitter = np.linspace(-0.10, 0.10, len(values))
        ax_c.scatter(index + jitter, values, s=SEED_MS**2, color=color,
                     alpha=SEED_ALPHA * 0.75, edgecolors="none", rasterized=True)
        row = primary.loc[scale]
        mean = float(row.mean_isotropic_enrichment)
        lo = float(row.isotropic_enrichment_ci_low)
        hi = float(row.isotropic_enrichment_ci_high)
        ax_c.errorbar(index, mean, yerr=[[mean - lo], [hi - mean]], fmt="D",
                      color=color, markeredgecolor="white", markeredgewidth=0.45,
                      ms=4.3, capsize=2, lw=0.8, zorder=3)
    ax_c.axhline(1, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_c.set_yscale("log", base=2)
    ax_c.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16], ["0.25", "0.5", "1", "2", "4", "8", "16"])
    ax_c.set_xticks(x, ["coarse", "intermediate", "fine"])
    ax_c.set_ylabel("signal / isotropic-noise power")
    compact_panel_title(ax_c, "C", "Coarse modes carry excess power")
    style_axis(ax_c, grid="y")

    cohorts = [("original_8", "original 8", COLORS["additive"]),
               ("public_v661_47", "disjoint 47", COLORS["shunting"])]
    for index, (name, _, color) in enumerate(cohorts):
        values = cell[cell.cohort.eq(name) & cell.scale.eq("coarse")].actual_minus_shuffled.to_numpy(float)
        jitter = np.linspace(-0.10, 0.10, len(values))
        ax_d.scatter(index + jitter, values, s=SEED_MS**2, color=color,
                     alpha=SEED_ALPHA * 0.78, edgecolors="none", rasterized=True)
        row = cohort[cohort.cohort.eq(name) & cohort.scale.eq("coarse")].iloc[0]
        mean = float(row.mean_actual_minus_shuffled)
        lo = float(row.actual_minus_shuffled_ci_low)
        hi = float(row.actual_minus_shuffled_ci_high)
        ax_d.errorbar(index, mean, yerr=[[mean - lo], [hi - mean]], fmt="D",
                      color=color, markeredgecolor="white", markeredgewidth=0.45,
                      ms=4.3, capsize=2, lw=0.8, zorder=3)
    ax_d.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_d.set_xticks([0, 1], [item[1] for item in cohorts])
    ax_d.set_ylabel("actual $-$ permuted coarse energy")
    compact_panel_title(ax_d, "D", "Ancestry excess is not replicated")
    style_axis(ax_d, grid="y")

    fig.canvas.draw()
    audit_layout(fig, "fig_irregular_tree_wavelets")
    audit_text_over_data(fig, "fig_irregular_tree_wavelets")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the adaptive local conductance-reliability experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_REF,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
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
SOURCE = ROOT / "source_data" / "adaptive_conductance_reliability"
FIGURE = ROOT / "figures" / "generated" / "fig_adaptive_conductance_reliability.pdf"


def estimator_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "A", "Local reliability estimate")
    boxes = [
        (0.03, 0.55, 0.25, 0.20, "paired noisy\ncredit $g_1,g_2$", COLORS["local"]),
        (0.38, 0.55, 0.25, 0.20, "$\\widehat S_b,\\widehat N_b$\nlocal moments", COLORS["oracle"]),
        (0.73, 0.55, 0.24, 0.20, "adaptive shunt\n$\\kappa_b\\geq0$", COLORS["shunting"]),
    ]
    for x, y, width, height, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.018",
                facecolor="white", edgecolor=color, lw=LW_DATA,
            )
        )
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
                fontsize=PT_ANNOT - 0.5, color=color, linespacing=1.25)
    for left, right in ((0.28, 0.38), (0.63, 0.73)):
        ax.add_patch(
            FancyArrowPatch(
                (left, 0.65), (right, 0.65), arrowstyle="-|>", mutation_scale=8,
                lw=LW_REF, color=COLORS["mute"],
            )
        )
    ax.text(
        0.5,
        0.35,
        r"$\widehat S_b=\langle g_1,g_2\rangle$"
        "   "
        r"$\widehat N_b=\|g_1-g_2\|^2/2$",
        ha="center",
        va="center",
        fontsize=PT_ANNOT,
        color=COLORS["ink"],
    )
    ax.text(
        0.5,
        0.18,
        r"$a_b=\min\{1,\widehat S_b/[c(\widehat S_b+\widehat N_b)]\}$",
        ha="center",
        va="center",
        fontsize=PT_ANNOT,
        color=COLORS["shunting"],
    )


def main() -> None:
    apply_neurips_style()
    branch = pd.read_csv(SOURCE / "branch_estimates.csv")
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    outcomes = pd.read_csv(SOURCE / "seed_outcomes.csv")
    contrasts = pd.read_csv(SOURCE / "paired_contrasts.csv")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_W, 4.45),
        gridspec_kw={
            "left": 0.10,
            "right": 0.985,
            "bottom": 0.12,
            "top": 0.90,
            "wspace": 0.40,
            "hspace": 0.72,
        },
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    estimator_schematic(ax_a)

    high = branch[np.isclose(branch.reliability_heterogeneity, 2.0)]
    gain_summary = high.groupby("branch", as_index=False).agg(
        oracle=("oracle_initial_gain", "mean"),
        adaptive=("adaptive_final_gain", "mean"),
        adaptive_sd=("adaptive_final_gain", "std"),
    )
    x = gain_summary.branch.to_numpy(int) + 1
    ax_b.plot(x, gain_summary.oracle, color=COLORS["oracle"], marker="D", ms=3.5,
              lw=LW_DATA, label="fixed initial oracle")
    ax_b.errorbar(x, gain_summary.adaptive, yerr=gain_summary.adaptive_sd,
                  color=COLORS["shunting"], marker="o", ms=3.5, lw=LW_DATA,
                  elinewidth=0.7, capsize=1.8, label="adaptive local")
    ax_b.set_xticks(x)
    ax_b.set_xlabel("branch index")
    ax_b.set_ylabel("attenuation gain")
    panel_title(ax_b, "B", "Estimated branch ordering")
    style_axis(ax_b, grid="y")
    clean_legend(ax_b, fontsize=PT_LEGEND - 0.4, loc="lower right")

    methods = [
        ("noisy_no_shunt", "no shunt", COLORS["ink"], "o"),
        # Both are controls: grays, not the salmon feedback slot and
        # not Figure 2's additive-architecture blue.
        ("adaptive_global_shunt", "adaptive global", COLORS["point_mlp"], "s"),
        ("adaptive_shuffled_shunt", "adaptive shuffled", COLORS["mute"], "^"),
        ("adaptive_local_shunt", "adaptive local", COLORS["shunting"], "D"),
        ("initial_oracle_shunt", "initial oracle", COLORS["oracle"], "P"),
    ]
    for method, label, color, marker in methods:
        part = summary[summary.method.eq(method)].sort_values("reliability_heterogeneity")
        mean = part.mean_final_test_loss.to_numpy(float)
        low = part.ci95_low_final_test_loss.to_numpy(float)
        high_ci = part.ci95_high_final_test_loss.to_numpy(float)
        h = part.reliability_heterogeneity.to_numpy(float)
        ax_c.plot(h, mean, color=color, marker=marker, ms=3.5, lw=LW_DATA, label=label)
        ax_c.fill_between(h, low, high_ci, color=color, alpha=0.08, linewidth=0)
    ax_c.set_xticks([0, 1, 2])
    ax_c.set_xlabel("credit-reliability heterogeneity")
    ax_c.set_ylabel("test loss after 40 updates")
    panel_title(ax_c, "C", "Adaptive placement is conditional")
    style_axis(ax_c, grid="both")
    handles, labels = ax_c.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.50, 0.505),
        ncol=5,
        frameon=False,
        fontsize=PT_LEGEND - 0.8,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.75,
    )

    controls = [
        ("adaptive_global_shunt", "global", COLORS["per_soma"]),
        ("adaptive_shuffled_shunt", "shuffled", COLORS["additive"]),
        ("noisy_no_shunt", "no shunt", COLORS["ink"]),
        ("initial_oracle_shunt", "oracle", COLORS["oracle"]),
        ("adaptive_point_gate", "point gain", COLORS["dend"]),
    ]
    high_outcomes = outcomes[np.isclose(outcomes.reliability_heterogeneity, 2.0)]
    wide = high_outcomes.pivot(index="seed", columns="method", values="final_test_loss")
    positions = np.arange(len(controls))
    for index, (control, _, color) in enumerate(controls):
        values = (wide[control] - wide["adaptive_local_shunt"]).to_numpy(float)
        jitter = np.linspace(-0.11, 0.11, len(values))
        ax_d.scatter(index + jitter, values, s=SEED_MS**2, color=color,
                     alpha=SEED_ALPHA * 0.72, edgecolors="none", zorder=1)
        row = contrasts[
            contrasts.left_minus_right.eq(f"adaptive_local_shunt - {control}")
            & contrasts.metric.eq("final_test_loss")
        ].iloc[0]
        mean = -float(row.mean_difference)
        low = -float(row.ci95_high)
        high_ci = -float(row.ci95_low)
        ax_d.errorbar(index, mean, yerr=[[mean - low], [high_ci - mean]], fmt="D",
                      ms=4.0, color=color, markeredgecolor="white",
                      markeredgewidth=0.45, elinewidth=0.8, capsize=2.0, zorder=3)
    ax_d.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    # Horizontal category labels (style contract: no rotated ticks); the two
    # two-word names wrap instead of rotating.
    horizontal = ["global", "shuffled", "no\nshunt", "oracle", "point\ngain"]
    ax_d.set_xticks(positions, horizontal)
    ax_d.tick_params(axis="x", labelsize=PT_SMALL - 0.8, pad=1.5)
    ax_d.set_ylabel("control loss $-$ adaptive-local loss")
    panel_title(ax_d, "D", "Final-loss boundary at high heterogeneity")
    style_axis(ax_d, grid="y")

    fig.canvas.draw()
    audit_layout(fig, "fig_adaptive_conductance_reliability")
    audit_text_over_data(fig, "fig_adaptive_conductance_reliability")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()

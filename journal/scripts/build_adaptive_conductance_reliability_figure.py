#!/usr/bin/env python3
"""Render the adaptive local conductance-reliability experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import style_direct_color_labels
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
        figsize=(FIG_W, 4.65),
        gridspec_kw={
            "left": 0.10,
            "right": 0.985,
            "bottom": 0.175,
            "top": 0.905,
            "wspace": 0.40,
            "hspace": 0.62,
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
              lw=LW_DATA, label="fixed oracle")
    ax_b.errorbar(x, gain_summary.adaptive, yerr=gain_summary.adaptive_sd,
                  color=COLORS["shunting"], marker="o", ms=3.5, lw=LW_DATA,
                  elinewidth=0.7, capsize=1.8, label="adaptive local")
    ax_b.set_xticks(x)
    ax_b.set_xlabel("branch index")
    ax_b.set_ylabel("attenuation gain")
    panel_title(ax_b, "B", "Estimated branch ordering")
    ax_b.text(.03,.96,"mean ± SD",transform=ax_b.transAxes,va="top",
              fontsize=PT_LEGEND-0.4,color=COLORS["mute"])
    style_axis(ax_b, grid="y")
    clean_legend(ax_b, fontsize=PT_LEGEND - 0.4, loc="lower right")

    # One colour per condition, shared with the fixed-profile study (the
    # consolidated supplement pastes both contrast panels on one sheet): the
    # global control takes the same salmon there, which also separates it from
    # the shuffled control where the two cross at low heterogeneity.
    methods = [
        ("noisy_no_shunt", "no shunt", COLORS["ink"], "o"),
        ("adaptive_global_shunt", "adaptive global", COLORS["per_soma"], "s"),
        ("adaptive_shuffled_shunt", "adaptive shuffled", COLORS["mute"], "^"),
        ("adaptive_local_shunt", "adaptive local", COLORS["shunting"], "D"),
        ("initial_oracle_shunt", "fixed oracle", COLORS["oracle"], "P"),
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
    panel_title(ax_c, "C", "No endpoint gain over no shunt")
    ax_c.text(.03,.95,"mean / 95% CI",transform=ax_c.transAxes,va="top",
              fontsize=PT_LEGEND,color=COLORS["mute"])
    style_axis(ax_c, grid="both")
    # The key belongs to C alone (B has its own), so it sits under C's axis
    # label, spanning C's width, rather than between the rows.
    handles, labels = ax_c.get_legend_handles_labels()
    ax_c.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=False,
        fontsize=PT_LEGEND,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    # The point-gate control given the adaptive gains ties the conductance
    # rule exactly in every seed (final loss difference 0, 50/50 ties); it is
    # reported as a number rather than a zero-spread column.
    controls = [
        ("adaptive_global_shunt", "global", COLORS["per_soma"]),
        ("adaptive_shuffled_shunt", "shuffled", COLORS["mute"]),
        ("noisy_no_shunt", "no shunt", COLORS["ink"]),
        ("initial_oracle_shunt", "fixed oracle", COLORS["oracle"]),
    ]
    high_outcomes = outcomes[np.isclose(outcomes.reliability_heterogeneity, 2.0)]
    wide = high_outcomes.pivot(index="seed", columns="method", values="final_test_loss")
    point_gate = (wide["adaptive_point_gate"] - wide["adaptive_local_shunt"]).dropna()
    assert len(point_gate) == 50 and float(np.abs(point_gate).max()) == 0.0
    positions = np.arange(len(controls))
    for index, (control, _, color) in enumerate(controls):
        values = (wide[control] - wide["adaptive_local_shunt"]).to_numpy(float)
        jitter = np.linspace(-0.13, 0.13, len(values))
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
    # Name the zero reference in a short right-hand margin, clear of the last
    # column's seed dots.
    ax_d.set_xlim(-0.5, len(controls) - 0.5 + 0.72)
    ax_d.text(len(controls) - 0.5 + 0.68, 0, "adaptive\nlocal", ha="right",
              va="bottom", fontsize=PT_SMALL, color=COLORS["mute"],
              linespacing=1.1)
    ax_d.set_xticks(positions, [entry[1] for entry in controls])
    ax_d.tick_params(axis="x", labelsize=PT_SMALL, pad=2.0)
    ax_d.set_ylabel("control loss $-$ adaptive-local loss")
    panel_title(ax_d, "D", "Final-loss boundary at high heterogeneity")
    style_axis(ax_d, grid="y")

    style_direct_color_labels(fig)
    fig.canvas.draw()
    audit_layout(fig, "fig_adaptive_conductance_reliability")
    audit_text_over_data(fig, "fig_adaptive_conductance_reliability")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()

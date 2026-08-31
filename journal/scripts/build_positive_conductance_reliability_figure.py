#!/usr/bin/env python3
"""Render the state-matched positive-conductance reliability experiment."""

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
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "positive_conductance_reliability_step_consistent"
FIGURES = ROOT / "figures" / "generated"


METHODS = [
    ("noisy_no_shunt", "no shunt", COLORS["ink"], "o"),
    ("best_global_shunt", "best global", COLORS["per_soma"], "s"),
    ("reliability_aligned_shunt", "SNR-aligned", COLORS["shunting"], "D"),
    ("shuffled_shunt", "shuffled", COLORS["mute"], "^"),
    ("anti_aligned_shunt", "anti-aligned", COLORS["additive"], "v"),
]


def schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "A", "Physical isolation")
    boxes = [
        (0.03, 0.59, 0.25, 0.19, "positive input\nrates $x\\geq0$", COLORS["dend"]),
        (0.38, 0.59, 0.25, 0.19, "branch\nvoltage $V$", COLORS["oracle"]),
        (0.73, 0.59, 0.24, 0.19, "local\neligibility", COLORS["shunting"]),
    ]
    for x, y, width, height, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.018",
                facecolor="white", edgecolor=color, lw=LW_DATA,
            )
        )
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center",
                fontsize=5.6, color=color, linespacing=1.25)
    for left, right in ((0.28, 0.38), (0.63, 0.73)):
        ax.add_patch(FancyArrowPatch((left, 0.685), (right, 0.685), arrowstyle="-|>",
                                     mutation_scale=8, lw=LW_REF, color=COLORS["mute"]))
    ax.text(0.50, 0.42, r"shunt $\kappa_b\geq0$  +  clamp current $\kappa_bV$",
            ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["ink"])
    ax.text(0.50, 0.28, r"$V'=V$   but   eligibility $\times\;G/(G+\kappa_b)$",
            ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["shunting"])
    ax.text(0.50, 0.10, "isolates physical input-resistance attenuation",
            ha="center", va="center", fontsize=PT_SMALL, color=COLORS["mute"])


def line_with_interval(
    ax: plt.Axes,
    frame: pd.DataFrame,
    method: str,
    metric: str,
    label: str,
    color: str,
    marker: str,
) -> None:
    part = frame[frame.method.eq(method)].sort_values("reliability_heterogeneity")
    x = part.reliability_heterogeneity.to_numpy(float)
    mean = part[f"mean_{metric}"].to_numpy(float)
    low = part[f"ci95_low_{metric}"].to_numpy(float)
    high = part[f"ci95_high_{metric}"].to_numpy(float)
    ax.plot(x, mean, color=color, marker=marker, ms=3.4, lw=LW_DATA, label=label)
    ax.fill_between(x, low, high, color=color, alpha=0.09, linewidth=0)


def main() -> None:
    apply_neurips_style()
    branch = pd.read_csv(SOURCE / "branch_reliability.csv")
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    contrasts = pd.read_csv(SOURCE / "extended_contrasts.csv")

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_W, 4.95),
        gridspec_kw={
            "left": 0.105,
            "right": 0.985,
            "bottom": 0.105,
            "top": 0.90,
            "wspace": 0.68,
            "hspace": 0.82,
        },
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    schematic(ax_a)

    high = branch[np.isclose(branch.reliability_heterogeneity, 2.0)]
    branch_summary = high.groupby("branch", as_index=False).agg(
        reliability=("optimal_reliability_gain", "mean"),
        reliability_sd=("optimal_reliability_gain", "std"),
        snr=("signal_to_noise", "mean"),
    )
    ax_b.plot(
        branch_summary.branch + 1,
        branch_summary.reliability,
        color=COLORS["shunting"], marker="o", ms=3.4, lw=LW_DATA,
    )
    ax_b.set_xticks(np.arange(1, 9))
    ax_b.set_xlabel("branch index")
    ax_b.set_ylabel(r"fixed-step gain $a_b^*$")
    panel_title(ax_b, "B", "Step-consistent profile")
    style_axis(ax_b)

    for method, label, color, marker in METHODS:
        line_with_interval(
            ax_c, summary, method, "one_step_test_loss_decrease",
            label, color, marker,
        )
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_c.set_xlabel("SNR heterogeneity")
    ax_c.set_ylabel("one-step test-loss decrease")
    panel_title(ax_c, "C", "Immediate step")
    style_axis(ax_c)

    comparison_styles = [
        ("reliability_aligned_shunt - best_global_shunt", "aligned $-$ global",
         COLORS["shunting"], "o"),
        ("reliability_aligned_shunt - noisy_no_shunt", "aligned $-$ no shunt",
         COLORS["oracle"], "s"),
    ]
    for comparison, label, color, marker in comparison_styles:
        part = contrasts[
            contrasts.left_minus_right.eq(comparison)
            & contrasts.metric.eq("one_step_test_loss_decrease")
        ].sort_values("heterogeneity")
        ax_d.plot(part.heterogeneity, part.mean_difference, color=color, marker=marker,
                  ms=3.4, lw=LW_DATA, label=label)
        ax_d.fill_between(part.heterogeneity, part.ci95_low, part.ci95_high,
                          color=color, alpha=0.10, linewidth=0)
    ax_d.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_d.set_xlabel("SNR heterogeneity")
    ax_d.set_ylabel("paired loss-decrease difference")
    panel_title(ax_d, "D", "Alignment contrast")
    style_axis(ax_d)
    clean_legend(ax_d, fontsize=PT_LEGEND - 0.4, loc="upper left")

    for method, label, color, marker in METHODS:
        line_with_interval(
            ax_e, summary, method, "final_test_loss", label, color, marker,
        )
    line_with_interval(
        ax_e, summary, "exact_clean_bp", "final_test_loss", "exact clean BP",
        COLORS["oracle"], "*",
    )
    ax_e.set_xlabel("SNR heterogeneity")
    ax_e.set_ylabel("test loss after 40 updates")
    panel_title(ax_e, "E", "Training horizon")
    style_axis(ax_e)
    handles, labels = ax_e.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.54, 0.505),
        ncol=6,
        frameon=False,
        fontsize=PT_LEGEND - 1.0,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.75,
    )

    controls = [
        ("best_global_shunt", "global", COLORS["per_soma"]),
        ("shuffled_shunt", "shuffled", COLORS["mute"]),
        ("anti_aligned_shunt", "anti", COLORS["additive"]),
        ("noisy_no_shunt", "no shunt", COLORS["ink"]),
        ("explicit_point_gate", "point gain", COLORS["dend"]),
    ]
    values, lows, highs = [], [], []
    for control, _, _ in controls:
        row = contrasts[
            np.isclose(contrasts.heterogeneity, 2.0)
            & contrasts.left_minus_right.eq(
                f"reliability_aligned_shunt - {control}"
            )
            & contrasts.metric.eq("final_test_loss")
        ].iloc[0]
        # Convert aligned-minus-control loss into control-minus-aligned so that
        # positive bars favor aligned shunting.
        values.append(-float(row.mean_difference))
        lows.append(-float(row.ci95_high))
        highs.append(-float(row.ci95_low))
    x = np.arange(len(controls))
    values = np.asarray(values)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    ax_f.bar(x, values, color=[entry[2] for entry in controls], width=0.68)
    ax_f.errorbar(x, values, yerr=np.vstack([values - lows, highs - values]),
                  color=COLORS["ink"], fmt="none", elinewidth=0.7, capsize=1.8)
    ax_f.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    # Five multi-word labels in a third-width panel overprinted at
    # 24 deg ("global" ran into "shuffled", "no shunt" into "point
    # gate"); a steeper angle separates them.
    ax_f.set_xticks(x, [entry[1] for entry in controls],
                    rotation=45, ha="right", rotation_mode="anchor")
    ax_f.set_ylabel("control loss $-$ aligned loss")
    panel_title(ax_f, "F", "Final boundary")
    style_axis(ax_f, grid="y")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_positive_conductance_reliability")
    audit_text_over_data(fig, "fig_positive_conductance_reliability")
    fig.savefig(
        FIGURES / "fig_positive_conductance_reliability.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_positive_conductance_reliability.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()

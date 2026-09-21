#!/usr/bin/env python3
"""Render the trained ownership/partition-residual diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_ERR,
    LW_REF,
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
SOURCE = ROOT / "source_data" / "trained_partition_residual"
FIGURE = ROOT / "figures" / "generated" / "fig_trained_partition_residual.pdf"

RESTRICTED = [
    "correct_ancestry_subtrees",
    "within_neuron_route_derangement",
    "depth_interleaved_bins",
    "random_sparse_matched",
    "random_rank_k",
    "learned_rank_k_upper_bound",
]


# Panel geometry in points.  Panel A is the one this figure contributes to
# the consolidated supplement (a two-row sheet whose second row holds two
# 226-pt panels), so it is authored ~300 pt wide with an ordinal budget axis
# and a difference inset; B and C share the second row.
FIG_H = 5.3
AX_H = 96.0
AX_H_A = 120.0
ROW_Y_TOP = (36.0, 224.0)
PANEL_A = (42.0, 340.0)      # x0, width
PANEL_B = (42.0, 160.0)
PANEL_C = (262.0, 160.0)
BUDGETS = [1, 2, 4, 8]
# Matched-bandwidth partition families drawn in the difference inset: one
# neutral lightness ladder and three marker shapes, so the blue/teal series
# hues stay reserved for the coefficient-encoder panels of the same sheet.
PARTITIONS = [
    ("correct_ancestry_subtrees", "ancestry", "#232323", "o"),
    ("depth_interleaved_bins", "depth bins", "#78818F", "s"),
    ("random_sparse_matched", "random sparse", "#A9AFB8", "^"),
]


def _panel_a(ax, dendritic: pd.DataFrame) -> None:
    from matplotlib.lines import Line2D

    position = {k: i for i, k in enumerate(BUDGETS)}

    def series(family: str) -> pd.DataFrame:
        part = dendritic[dendritic.feedback_family.eq(family)].sort_values("budget_k")
        if list(part.budget_k) != BUDGETS:
            raise ValueError(f"{family}: expected budgets {BUDGETS}")
        return part

    families = {family: series(family) for family, *_ in PARTITIONS}
    ancestry = families["correct_ancestry_subtrees"]
    # The three matched-bandwidth partitions differ by < 0.005 capture at
    # every budget (checked here, resolved in the inset), so one stroke
    # carries them on the main axes.
    for family, part in families.items():
        gap = np.abs(part.mean_address_capture.to_numpy() - ancestry.mean_address_capture.to_numpy())
        if gap.max() >= 0.006:
            raise ValueError(f"{family} departs from ancestry by {gap.max():.4f}; draw it separately")
    x = [position[k] for k in ancestry.budget_k]
    ax.fill_between(
        x,
        ancestry.ci95_low_address_capture,
        ancestry.ci95_high_address_capture,
        color=COLORS["ink"],
        alpha=0.12,
        linewidth=0,
    )
    ax.plot(
        x,
        ancestry.mean_address_capture,
        color=COLORS["ink"],
        marker="o",
        ms=3.8,
        lw=LW_DATA,
        label="matched-bandwidth partitions\n(ancestry, depth bins, random sparse)",
        zorder=3,
    )
    deranged = series("within_neuron_route_derangement")
    ax.fill_between(
        x,
        deranged.ci95_low_address_capture,
        deranged.ci95_high_address_capture,
        color=COLORS["mute"],
        alpha=0.12,
        linewidth=0,
    )
    ax.plot(
        x,
        deranged.mean_address_capture,
        color=COLORS["mute"],
        marker="D",
        ms=3.4,
        ls="--",
        lw=LW_DATA,
        label="deranged ownership",
        zorder=3,
    )
    ax.set_xlim(-0.35, 3.35)
    ax.set_xticks(list(position.values()), [str(k) for k in BUDGETS])
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("teaching-route budget $K$ (ordinal axis)")
    ax.set_ylabel("exact-field capture")
    panel_title(ax, "A", "Capture by route budget")
    style_axis(ax, grid="y")
    clean_legend(ax, fontsize=PT_LEGEND, loc="lower right",
                 bbox_to_anchor=(1.0, 0.07), handlelength=1.6)

    # Inset: the between-partition differences, re-centred on the ancestry
    # mean, at the two budgets where the families are not tied by
    # construction.  Bars are each family's own 95% interval.
    inset = ax.inset_axes([0.14, 0.50, 0.36, 0.46])
    offsets = np.linspace(-0.22, 0.22, len(PARTITIONS))
    handles = []
    for offset, (family, label, color, marker) in zip(offsets, PARTITIONS):
        part = families[family].set_index("budget_k")
        centre = ancestry.set_index("budget_k").mean_address_capture
        for slot, budget in enumerate((2, 4)):
            mean = part.loc[budget, "mean_address_capture"] - centre.loc[budget]
            low = part.loc[budget, "ci95_low_address_capture"] - centre.loc[budget]
            high = part.loc[budget, "ci95_high_address_capture"] - centre.loc[budget]
            inset.errorbar(
                slot + offset,
                100 * mean,
                yerr=[[100 * (mean - low)], [100 * (high - mean)]],
                fmt=marker,
                ms=3.2,
                color=color,
                lw=LW_ERR,
                capsize=1.5,
                capthick=LW_ERR,
                zorder=3,
            )
        handles.append(Line2D([], [], marker=marker, color=color, ls="none", ms=3.2, label=label))
    inset.axhline(0, color=COLORS["grid"], lw=LW_REF, zorder=1)
    inset.set_xlim(-0.6, 1.6)
    inset.set_xticks([0, 1], ["$K=2$", "$K=4$"])
    inset.set_ylim(-0.65, 0.65)
    inset.set_yticks([-0.5, 0, 0.5])
    inset.set_ylabel("capture \u2212 ancestry (pp)", fontsize=PT_LEGEND, labelpad=1.5)
    inset.tick_params(labelsize=PT_LEGEND, length=2.4, pad=1.5)
    for spine in ("top", "right"):
        inset.spines[spine].set_visible(False)
    # Family key to the right of the inset, in the empty upper-middle of the
    # main axes (the partition stroke passes well below it there).
    inset.legend(handles=handles, loc="upper left", fontsize=PT_LEGEND, frameon=False,
                 handlelength=0.8, handletextpad=0.3, borderaxespad=0.0, labelspacing=0.15,
                 bbox_to_anchor=(1.03, 1.0))


def main() -> None:
    apply_neurips_style()
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    outcomes = pd.read_csv(SOURCE / "seed_state_residuals.csv")
    correlations = pd.read_csv(SOURCE / "seed_level_associations.csv")

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fw, fh = FIG_W * 72.0, FIG_H * 72.0

    def axes_pt(x0, y_top, width, height):
        return fig.add_axes([x0 / fw, (fh - y_top - height) / fh, width / fw, height / fh])

    ax_a = axes_pt(PANEL_A[0], ROW_Y_TOP[0], PANEL_A[1], AX_H_A)
    ax_b = axes_pt(PANEL_B[0], ROW_Y_TOP[1], PANEL_B[1], AX_H)
    ax_c = axes_pt(PANEL_C[0], ROW_Y_TOP[1], PANEL_C[1], AX_H)

    dendritic = summary[
        summary.architecture.eq("dendritic_tree") & summary.state.eq("trained")
    ]
    _panel_a(ax_a, dendritic)

    trained = outcomes[
        outcomes.state.eq("trained")
        & outcomes.feedback_family.isin(RESTRICTED)
        & outcomes.budget_k.lt(8)
    ]
    seed_condition = trained.groupby(
        ["seed", "feedback_family", "budget_k"], as_index=False
    ).agg(
        address_capture=("address_capture", "mean"),
        heldout_accuracy=("heldout_accuracy", "mean"),
    )
    ax_b.scatter(
        seed_condition.address_capture,
        100 * seed_condition.heldout_accuracy,
        s=SEED_MS**2,
        color=COLORS["mute"],
        alpha=0.16,
        edgecolors="none",
        rasterized=True,
    )
    condition_means = seed_condition.groupby(
        ["feedback_family", "budget_k"], as_index=False
    ).agg(address_capture=("address_capture", "mean"), heldout_accuracy=("heldout_accuracy", "mean"))
    # K is an ordinal budget, so it takes the manuscript slate ramp rather
    # than three unrelated semantic hues.
    budget_style = {
        1: ("o", "#9AA5B4"),
        2: ("s", "#5F6B7E"),
        4: ("D", "#2E3947"),
    }
    for budget, (marker, color) in budget_style.items():
        part = condition_means[condition_means.budget_k.eq(budget)]
        ax_b.scatter(
            part.address_capture,
            100 * part.heldout_accuracy,
            s=24,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            label=f"$K={budget}$ mean",
            zorder=3,
        )
    ax_b.set_xlabel("trained address capture")
    ax_b.set_ylabel("held-out accuracy (%)")
    panel_title(ax_b, "B", "Capture tracks trained utility")
    style_axis(ax_b, grid="both")
    clean_legend(
        ax_b,
        fontsize=PT_LEGEND,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
    )

    subset_order = ["all_conditions", "restricted_k_lt_8"]
    labels = ["all", "$K<8$"]
    wide = correlations.pivot(index="seed", columns="subset", values="spearman_rho")
    for _, row in wide.iterrows():
        ax_c.plot([0, 1], [row[subset_order[0]], row[subset_order[1]]],
                  color=COLORS["grid"], lw=LW_REF, zorder=1)
    for index, subset in enumerate(subset_order):
        values = wide[subset].to_numpy(float)
        jitter = np.linspace(-0.055, 0.055, len(values))
        ax_c.scatter(
            index + jitter,
            values,
            s=SEED_MS**2,
            color=COLORS["shunting" if index == 0 else "additive"],
            alpha=SEED_ALPHA,
            edgecolors="none",
            zorder=2,
        )
        mean = float(values.mean())
        ax_c.plot([index - 0.15, index + 0.15], [mean, mean],
                  color=COLORS["ink"], lw=LW_DATA + 0.4, zorder=3)
    ax_c.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax_c.set_xticks([0, 1], labels)
    ax_c.set_ylim(-0.05, 1.0)
    ax_c.set_ylabel(r"within-seed Spearman $\rho$")
    panel_title(ax_c, "C", "Seed-level association")
    style_axis(ax_c, grid="y")

    fig.canvas.draw()
    audit_layout(fig, "figure_S23_partition_residual")
    audit_text_over_data(fig, "figure_S23_partition_residual")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()

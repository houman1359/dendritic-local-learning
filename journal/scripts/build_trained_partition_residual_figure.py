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

FAMILIES = [
    ("correct_ancestry_subtrees", "ancestry routes", COLORS["dend"], "o"),
    ("depth_interleaved_bins", "depth bins", COLORS["additive"], "s"),
    ("random_sparse_matched", "random sparse", COLORS["oracle"], "^"),
    ("within_neuron_route_derangement", "deranged ownership", COLORS["bp"], "D"),
]
RESTRICTED = [
    "correct_ancestry_subtrees",
    "within_neuron_route_derangement",
    "depth_interleaved_bins",
    "random_sparse_matched",
    "random_rank_k",
    "learned_rank_k_upper_bound",
]


def main() -> None:
    apply_neurips_style()
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    outcomes = pd.read_csv(SOURCE / "seed_state_residuals.csv")
    correlations = pd.read_csv(SOURCE / "seed_level_associations.csv")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(FIG_W, 2.48),
        gridspec_kw={
            "left": 0.095,
            "right": 0.985,
            "bottom": 0.23,
            "top": 0.82,
            "wspace": 0.62,
        },
    )
    ax_a, ax_b, ax_c = axes

    dendritic = summary[
        summary.architecture.eq("dendritic_tree") & summary.state.eq("trained")
    ]
    for family, label, color, marker in FAMILIES:
        part = dendritic[dendritic.feedback_family.eq(family)].sort_values("budget_k")
        ax_a.plot(
            part.budget_k,
            part.mean_address_capture,
            color=color,
            marker=marker,
            ms=3.5,
            lw=LW_DATA,
            label=label,
        )
        ax_a.fill_between(
            part.budget_k,
            part.ci95_low_address_capture,
            part.ci95_high_address_capture,
            color=color,
            alpha=0.09,
            linewidth=0,
        )
    ax_a.set_xticks([1, 2, 4, 8])
    ax_a.set_ylim(-0.04, 1.04)
    ax_a.set_xlabel("teaching-route budget $K$")
    ax_a.set_ylabel("capture of exact coefficient field")
    panel_title(ax_a, "A", "Irreducible address residual")
    style_axis(ax_a, grid="y")
    clean_legend(ax_a, fontsize=PT_LEGEND - 1.0, loc="upper left")

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
    budget_style = {
        1: ("o", COLORS["low_rank"]),
        2: ("s", COLORS["additive"]),
        4: ("D", COLORS["dend"]),
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
    ax_b.set_ylabel(r"held-out accuracy (\%)")
    panel_title(ax_b, "B", "Capture tracks trained utility")
    style_axis(ax_b, grid="both")
    clean_legend(
        ax_b,
        fontsize=PT_LEGEND - 0.9,
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

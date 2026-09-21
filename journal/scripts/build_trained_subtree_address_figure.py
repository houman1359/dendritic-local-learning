#!/usr/bin/env python3
"""Render the completed phase-1 subtree-address experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from journal_style import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_ERR,
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
SOURCE = ROOT / "source_data" / "trained_subtree_address"
FIGURES = ROOT / "figures" / "generated"

COLORS_BY_CONDITION = {
    "neuron_shared_k1": COLORS["per_soma"],
    "correct_subtree_k2": COLORS["shunting"],
    "within_neuron_deranged_k2": COLORS["mute"],
    "random_dense_rank2": COLORS["additive"],
    "exact_transport": COLORS["oracle"],
    "gated_point_emulation": COLORS["inh"],
}


def interval(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def points_and_interval(ax, x: int, values: np.ndarray, color: str, seed: int) -> None:
    ax.scatter(
        x + np.linspace(-0.055, 0.055, len(values)),
        values,
        s=11,
        color=color,
        alpha=0.55,
        edgecolors="none",
    )
    mean, low, high = interval(values, seed)
    ax.errorbar(
        x,
        mean,
        yerr=[[mean - low], [high - mean]],
        marker="D",
        markerfacecolor="white",
        markeredgecolor=color,
        color=color,
        ms=4.2,
        lw=LW_ERR,
        capsize=ERR_CAPSIZE,
        zorder=5,
    )


def main() -> None:
    apply_neurips_style()
    outcomes = pd.read_csv(SOURCE / "seed_outcomes.csv")
    gradients = pd.read_csv(SOURCE / "gradient_audit.csv")
    frame = outcomes.merge(gradients, on=["seed", "condition"], validate="one_to_one")

    displayed = [
        "neuron_shared_k1",
        "correct_subtree_k2",
        "within_neuron_deranged_k2",
        "random_dense_rank2",
        "exact_transport",
        "gated_point_emulation",
    ]
    labels = ["neuron-\nshared", "correct", "deranged", "random", "exact", "gated\npoint"]
    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1,
        3,
        figsize=(FIG_W, 3.00),
        gridspec_kw={
            "left": 0.083,
            "right": 0.982,
            "bottom": 0.205,
            "top": 0.84,
            "wspace": 0.52,
        },
    )
    for index, condition in enumerate(displayed):
        values = frame[frame.condition.eq(condition)].test_accuracy.to_numpy(float)
        points_and_interval(
            ax_a, index, values, COLORS_BY_CONDITION[condition], 50_000 + index
        )
    ax_a.set_xticks(range(len(displayed)), labels, rotation=27, ha="right")
    ax_a.tick_params(axis="x", labelsize=PT_SMALL)
    ax_a.set_ylabel("held-out accuracy")
    ax_a.set_ylim(0.0, 1.03)
    panel_title(ax_a, "A", "Within-neuron routing")
    style_axis(ax_a, grid="y")

    geometry_conditions = [
        "neuron_shared_k1",
        "correct_subtree_k2",
        "within_neuron_deranged_k2",
        "random_dense_rank2",
    ]
    geometry_labels = ["neuron-shared", "correct", "deranged", "random rank-2"]
    for condition, label in zip(geometry_conditions, geometry_labels):
        part = frame[frame.condition.eq(condition)]
        ax_b.scatter(
            part.initial_gradient_scaled_capture,
            part.initial_norm_matched_one_step_progress,
            s=16,
            color=COLORS_BY_CONDITION[condition],
            alpha=0.68,
            edgecolors="none",
            label=label,
        )
    ax_b.axhline(0, color=COLORS["mute"], ls="--", lw=0.8)
    ax_b.set_xlabel("exact-gradient capture")
    ax_b.set_ylabel("norm-matched one-step progress")
    panel_title(ax_b, "B", "Gradient geometry")
    style_axis(ax_b)
    clean_legend(
        ax_b,
        fontsize=PT_LEGEND - 0.8,
        loc="center",
        bbox_to_anchor=(0.74, 0.43),
    )

    switch_conditions = [
        "correct_subtree_k2",
        "neuron_shared_k1",
        "within_neuron_deranged_k2",
        "random_dense_rank2",
    ]
    switch_labels = ["correct", "neuron-shared", "deranged", "random"]
    for index, condition in enumerate(switch_conditions):
        values = frame[
            frame.condition.eq(condition)
        ].context_switch_forgetting.to_numpy(float)
        points_and_interval(
            ax_c, index, values, COLORS_BY_CONDITION[condition], 51_000 + index
        )
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=0.8)
    ax_c.set_xticks(range(len(switch_conditions)), switch_labels, rotation=20, ha="right")
    ax_c.tick_params(axis="x", labelsize=PT_SMALL)
    ax_c.set_ylabel("context-0 forgetting")
    panel_title(ax_c, "C", "Switch interference")
    style_axis(ax_c, grid="y")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_trained_subtree_address")
    audit_text_over_data(fig, "fig_trained_subtree_address")
    fig.savefig(
        FIGURES / "fig_trained_subtree_address.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_trained_subtree_address.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()

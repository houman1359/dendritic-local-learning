#!/usr/bin/env python3
"""Generate appendix figure for inhibitory causality and error-rank diagnostics."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import COLORS, apply_neurips_style, style_axis

apply_neurips_style()

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"
CAUSAL_CSV = ROOT / "analysis" / "inhibition_causality_selected_20260427" / "inhibition_causality_runs.csv"
RANK_CSV = ROOT / "analysis" / "error_rank_selected_20260427" / "error_rank_diagnostics.csv"


INTERVENTION_ORDER = [
    "original",
    "zero_i",
    "shuffle_i",
    "mean_clamp_i",
    "uniform_matched_i",
]
INTERVENTION_LABELS = {
    "original": "learned",
    "zero_i": "zero",
    "shuffle_i": "shuffle",
    "mean_clamp_i": "mean",
    "uniform_matched_i": "uniform",
}
INTERVENTION_COLORS = {
    "original": COLORS["shunting"],
    "zero_i": COLORS["mute"],
    "shuffle_i": COLORS["low_rank"],
    "mean_clamp_i": COLORS["per_soma"],
    "uniform_matched_i": COLORS["oracle"],
}


def _mean_std(frame: pd.DataFrame, value: str, group: str) -> pd.DataFrame:
    return frame.groupby(group, dropna=False)[value].agg(["mean", "std", "count"]).reset_index()


def _bar_panel(ax, data: pd.DataFrame, dataset: str, title: str) -> None:
    sub = data[data["dataset"] == dataset].copy()
    stats = _mean_std(sub, "accuracy", "intervention").set_index("intervention")
    xs = np.arange(len(INTERVENTION_ORDER))
    means = [stats.loc[k, "mean"] if k in stats.index else np.nan for k in INTERVENTION_ORDER]
    stds = [stats.loc[k, "std"] if k in stats.index else 0.0 for k in INTERVENTION_ORDER]
    colors = [INTERVENTION_COLORS[k] for k in INTERVENTION_ORDER]
    ax.bar(xs, means, yerr=stds, color=colors, edgecolor="white", linewidth=0.6, capsize=2.5)
    for x, key in zip(xs, INTERVENTION_ORDER):
        vals = sub[sub["intervention"] == key]["accuracy"].to_numpy(dtype=float)
        if vals.size:
            jitter = np.linspace(-0.10, 0.10, vals.size)
            ax.scatter(
                np.full(vals.size, x) + jitter,
                vals,
                s=8,
                color="white",
                edgecolor="#333333",
                linewidth=0.35,
                zorder=5,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [INTERVENTION_LABELS[k] for k in INTERVENTION_ORDER],
        rotation=35,
        ha="right",
        fontsize=6.7,
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Test accuracy", fontsize=8.2)
    ax.set_title(title, fontsize=8.8, pad=7)
    style_axis(ax, grid="y")


def _rank_panel(ax, data: pd.DataFrame, metric: str, ylabel: str, title: str) -> None:
    sub = data[(data["dataset"] == "mnist") & (data["ie_value"] == 5) & (data["scope"] == "all_layers")]
    order = ["dendritic_additive", "dendritic_shunting"]
    labels = ["Additive", "Shunting"]
    colors = [COLORS["additive"], COLORS["shunting"]]
    xs = np.arange(len(order))
    means = []
    stds = []
    for net in order:
        vals = sub[sub["network_type"] == net][metric].to_numpy(dtype=float)
        means.append(float(np.nanmean(vals)))
        stds.append(float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0)
    ax.bar(xs, means, yerr=stds, color=colors, edgecolor="white", linewidth=0.6, capsize=2.5)
    for x, net in zip(xs, order):
        vals = sub[sub["network_type"] == net][metric].to_numpy(dtype=float)
        if vals.size:
            jitter = np.linspace(-0.07, 0.07, vals.size)
            ax.scatter(
                np.full(vals.size, x) + jitter,
                vals,
                s=10,
                color="white",
                edgecolor="#333333",
                linewidth=0.35,
                zorder=5,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=8.2)
    ax.set_title(title, fontsize=8.8, pad=7)
    style_axis(ax, grid="y")


def main() -> None:
    causal = pd.read_csv(CAUSAL_CSV)
    rank = pd.read_csv(RANK_CSV)

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.35))
    _bar_panel(axes[0], causal, "mnist", "MNIST")
    _bar_panel(axes[1], causal, "noise_resilience", "Noise resilience")
    _rank_panel(
        axes[2],
        rank,
        "rank1_residual",
        "SVD residual",
        "Exact-error residual",
    )
    _rank_panel(
        axes[3],
        rank,
        "effective_rank_participation",
        "Participation rank",
        "Exact-error rank",
    )
    for ax, label in zip(axes, "ABCD"):
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            fontsize=7.8,
            fontweight="bold",
            va="top",
            ha="left",
            color=COLORS["ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
        )
    fig.subplots_adjust(left=0.065, right=0.995, top=0.79, bottom=0.30, wspace=0.64)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_s_inhibition_causality_error_rank"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=350)
    print(f"Saved {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()

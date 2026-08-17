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
from neurips_style import (  # noqa: E402
    # noqa: E402,
    COLORS,
    FIG_W,
    LW_HAIR,
    PT_LEGEND,
    PT_SMALL,
    REF_LW,
    apply_neurips_style,
    clean_legend,
    grid_figure,
    panel_title,
    style_axis,
)

apply_neurips_style()

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"


def _tracked(name: str, fallback: Path) -> Path:
    """Prefer the git-tracked figures/data/ copy; fall back to local analysis/."""
    t = ROOT / "figures" / "data" / name
    return t if t.exists() else fallback


CAUSAL_CSV = _tracked(
    "inhibition_causality_runs.csv",
    ROOT / "analysis" / "inhibition_causality_selected_20260427" / "inhibition_causality_runs.csv",
)
RANK_CSV = _tracked(
    "error_rank_diagnostics.csv",
    ROOT / "analysis" / "error_rank_selected_20260427" / "error_rank_diagnostics.csv",
)
ERROR_FIELD_DECOMPOSITION_RUNS_CSV = _tracked(
    "error_field_decomposition_runs.csv",
    ROOT
    / "analysis"
    / "scope_corrected_decomposition_20260725"
    / "error_field_decomposition_runs.csv",
)


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


def _bar_panel(ax, data: pd.DataFrame, dataset: str, title: str, letter="") -> None:
    sub = data[data["dataset"] == dataset].copy()
    stats = _mean_std(sub, "accuracy", "intervention").set_index("intervention")
    xs = np.arange(len(INTERVENTION_ORDER))
    means = [stats.loc[k, "mean"] if k in stats.index else np.nan for k in INTERVENTION_ORDER]
    stds = [stats.loc[k, "std"] if k in stats.index else 0.0 for k in INTERVENTION_ORDER]
    colors = [INTERVENTION_COLORS[k] for k in INTERVENTION_ORDER]
    ax.bar(xs, means, yerr=stds, color=colors, edgecolor="white", linewidth=LW_HAIR, capsize=2.5)
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
                linewidth=LW_HAIR,
                zorder=5,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [INTERVENTION_LABELS[k] for k in INTERVENTION_ORDER],
        rotation=35,
        ha="right",
        fontsize=PT_SMALL,
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Test accuracy", fontsize=PT_LEGEND)
    panel_title(ax, letter, title)
    style_axis(ax, grid="y")


def _rank_panel(ax, data: pd.DataFrame, metric: str, ylabel: str, title: str, letter: str = "") -> None:
    sub = data[(data["dataset"] == "mnist") & (data["ie_value"] == 5) & (data["scope"] == "all_layers")]
    order = ["dendritic_additive", "dendritic_shunting"]
    labels = ["Add.", "Shunt."]
    colors = [COLORS["additive"], COLORS["shunting"]]
    xs = np.arange(len(order))
    means = []
    stds = []
    for net in order:
        vals = sub[sub["network_type"] == net][metric].to_numpy(dtype=float)
        means.append(float(np.nanmean(vals)))
        stds.append(float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0)
    ax.bar(xs, means, yerr=stds, color=colors, edgecolor="white", linewidth=LW_HAIR, capsize=2.5)
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
                linewidth=LW_HAIR,
                zorder=5,
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel, fontsize=PT_LEGEND)
    panel_title(ax, letter, title)
    style_axis(ax, grid="y")


def _dendritic_fidelity_panel(ax, data: pd.DataFrame) -> None:
    """Stage-resolved fidelity, excluding the exact-by-construction soma."""
    sub = data[
        (data["dataset"] == "mnist")
        & (data["strategy"] == "local_ca")
        & (data["rule_variant"] == "5f")
        & (data["error_broadcast_mode"] == "per_soma")
        & (data["scope_type"] == "stage")
        & (data["population"] == 0)
        & (data["stage_role"].isin(["distal", "proximal"]))
        & (data["family"] == "submitted_mw")
    ].copy()
    if sub.empty:
        raise ValueError("Scope-corrected stage-resolved submitted-field data are unavailable")
    per_checkpoint = (
        sub.groupby(
            ["run_name", "seed", "network_type", "stage_role"],
            as_index=False,
        )["cosine"]
        .mean()
    )
    stages = ["distal", "proximal"]
    stage_labels = ["Distal", "Prox."]
    cores = [
        ("dendritic_additive", "Add.", COLORS["additive"]),
        ("dendritic_shunting", "Shunt.", COLORS["shunting"]),
    ]
    xs = np.arange(len(stages))
    width = 0.34
    for offset, (net, label, color) in zip([-width / 2, width / 2], cores):
        means = []
        stds = []
        for stage in stages:
            values = per_checkpoint[
                (per_checkpoint["network_type"] == net)
                & (per_checkpoint["stage_role"] == stage)
            ]["cosine"]
            if len(values) != 5:
                raise ValueError(
                    f"Expected five checkpoint values for {net}/{stage}, got {len(values)}"
                )
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)))
        ax.bar(
            xs + offset,
            means,
            width,
            yerr=stds,
            color=color,
            edgecolor="white",
            linewidth=LW_HAIR,
            capsize=2.5,
            label=label,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(stage_labels, rotation=20, ha="right", fontsize=PT_SMALL)
    ax.set_ylim(0, 0.30)
    ax.set_ylabel("MW-field cosine", fontsize=PT_LEGEND)
    panel_title(ax, "E", "Dendritic fidelity")
    clean_legend(ax, fontsize=PT_SMALL,
        frameon=False,
        loc="upper left",
        handlelength=0.9,
        handletextpad=0.25,
    )
    style_axis(ax, grid="y")


def main() -> None:
    causal = pd.read_csv(CAUSAL_CSV)
    rank = pd.read_csv(RANK_CSV)
    decomposition_runs = pd.read_csv(ERROR_FIELD_DECOMPOSITION_RUNS_CSV)

    # Five panels on one row leaves ~0.9 in each, too narrow for the y labels
    # and titles; a 3x2 grid gives every panel the standard panel box.
    fig, axgrid = grid_figure(3, 2)
    axes = list(axgrid.ravel())
    axes[5].set_visible(False)
    _bar_panel(axes[0], causal, "mnist", "MNIST", letter="A")
    _bar_panel(axes[1], causal, "noise_resilience", "Noise", letter="B")
    _rank_panel(
        axes[2],
        rank,
        "rank1_residual",
        "SVD residual",
        "SVD residual",
        letter="C",
    )
    _rank_panel(
        axes[3],
        rank,
        "effective_rank_participation",
        "Participation rank",
        "Participation rank",
        letter="D",
    )
    _dendritic_fidelity_panel(axes[4], decomposition_runs)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_s_inhibition_causality_error_rank"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=350)
    print(f"Saved {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Plot LocalCA rule components across training by dendritic depth."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


SYNAPTIC_COMPONENT_ROWS = [
    ("driving_force", "Driving force $E_{rev}-V_b$"),
    ("presynaptic_drive", "Presynaptic drive $x$"),
    ("rule_eligibility", "Rule eligibility"),
    ("local_update_factor", "Local update factor"),
]

DEPTH_COLORS = {
    "distal": "#355C7D",
    "intermediate": "#6C5B7B",
    "proximal": "#C06C84",
    "soma": "#F67280",
    "unknown": "#555555",
}

SYNAPSE_LABELS = {
    "excitatory": "Excitatory synapses",
    "inhibitory": "Inhibitory synapses",
}


def _epoch_from_filename(value: str) -> int:
    match = re.search(r"epoch(\d+)", str(value))
    if match is None:
        return -1
    return int(match.group(1))


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["epoch"] = out["filename"].map(_epoch_from_filename)
    out = out[out["epoch"] >= 0]
    out = out[out["population"] == "excitatory"]
    out["depth_order"] = out["dendritic_depth"].astype(int)
    return out


def _aggregate(sub: pd.DataFrame) -> pd.DataFrame:
    if sub.empty:
        return sub
    group_cols = ["epoch", "dendritic_depth", "depth_label"]
    return (
        sub.groupby(group_cols, as_index=False)
        .agg(mean=("mean", "mean"), mean_abs=("mean_abs", "mean"))
        .sort_values(["dendritic_depth", "epoch"])
    )


def _plot_depth_lines(ax, agg: pd.DataFrame) -> None:
    for (_, depth_label), depth_df in agg.groupby(
        ["dendritic_depth", "depth_label"], sort=True
    ):
        color = DEPTH_COLORS.get(str(depth_label), DEPTH_COLORS["unknown"])
        ax.plot(
            depth_df["epoch"],
            depth_df["mean"],
            label=str(depth_label),
            color=color,
            linewidth=2.0,
        )
    ax.axhline(0, color="0.75", linewidth=0.8, linestyle="--")


def plot_components(trajectory_csv: Path, output: Path) -> None:
    df = _prepare(pd.read_csv(trajectory_csv))

    fig, axes = plt.subplots(
        len(SYNAPTIC_COMPONENT_ROWS),
        2,
        figsize=(9.5, 8.0),
        sharex=True,
    )

    for row_idx, (component, ylabel) in enumerate(SYNAPTIC_COMPONENT_ROWS):
        for col_idx, synapse_type in enumerate(("excitatory", "inhibitory")):
            ax = axes[row_idx, col_idx]
            sub = df[
                (df["synapse_type"] == synapse_type)
                & (df["component"] == component)
            ]
            agg = _aggregate(sub)

            _plot_depth_lines(ax, agg)
            ax.set_ylabel(ylabel)
            if row_idx == 0:
                ax.set_title(SYNAPSE_LABELS[synapse_type])
            if row_idx == len(SYNAPTIC_COMPONENT_ROWS) - 1:
                ax.set_xlabel("Epoch")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=max(1, len(by_label)),
        frameon=False,
        title="Dendritic depth",
    )
    fig.suptitle(
        "Synapse-specific LocalCA component trajectories by dendritic depth",
        y=0.99,
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    pdf_output = output.with_suffix(".pdf")
    fig.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)

    rtot_output = output.with_name(output.stem.replace("components", "input_resistance") + output.suffix)
    plot_input_resistance(df, rtot_output)


def plot_input_resistance(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    sub = df[(df["synapse_type"] == "branch") & (df["component"] == "input_resistance")]
    agg = _aggregate(sub)
    _plot_depth_lines(ax, agg)
    ax.set_title("Branch input resistance $R^{tot}$ by dendritic depth")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean $R^{tot}$")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, title="Dendritic depth", ncol=3)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_components(args.trajectory_csv, args.output)


if __name__ == "__main__":
    main()

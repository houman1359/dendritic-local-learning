#!/usr/bin/env python3
"""Generate gradient-alignment heatmaps from component summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save_dual(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path(
            "drafts/dendritic-local-learning/figures/fig_gradient_alignment_components"
        ),
    )
    args = parser.parse_args()

    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise RuntimeError(f"No data found in {args.summary_csv}")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )

    comp_order = [
        "excitatory_synapse",
        "inhibitory_synapse",
        "dendritic_conductance",
        "reactivation",
    ]
    core_order = ["dendritic_shunting", "dendritic_additive"]
    rule_order = ["3f", "4f", "5f"]

    cond_labels: list[str] = []
    for core in core_order:
        core_short = "shunt" if core == "dendritic_shunting" else "add"
        for rule in rule_order:
            cond_labels.append(f"{core_short}-{rule}")

    cosine_grid = np.full((len(comp_order), len(cond_labels)), np.nan, dtype=float)
    norm_err_grid = np.full((len(comp_order), len(cond_labels)), np.nan, dtype=float)

    for ci, comp in enumerate(comp_order):
        for core in core_order:
            for rule in rule_order:
                col = (
                    core_order.index(core) * len(rule_order)
                    + rule_order.index(rule)
                )
                row = df[
                    (df["component"] == comp)
                    & (df["core_type"] == core)
                    & (df["rule_variant"] == rule)
                ]
                if row.empty:
                    continue
                cosine_grid[ci, col] = float(row["weighted_mean_cosine"].iloc[0])
                ratio = float(row["mean_norm_ratio"].iloc[0])
                norm_err_grid[ci, col] = abs(np.log10(max(ratio, 1e-12)))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.5), constrained_layout=True)

    im0 = axes[0].imshow(cosine_grid, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0].set_title("A. Weighted Cosine(Local, Backprop)")
    axes[0].set_xticks(np.arange(len(cond_labels)))
    axes[0].set_xticklabels(cond_labels, rotation=35, ha="right")
    axes[0].set_yticks(np.arange(len(comp_order)))
    axes[0].set_yticklabels(comp_order)
    cbar0 = fig.colorbar(im0, ax=axes[0], shrink=0.9)
    cbar0.set_label("cosine")

    im1 = axes[1].imshow(norm_err_grid, aspect="auto", cmap="viridis", vmin=0.0)
    axes[1].set_title("B. Scale Mismatch |log10(||g_local||/||g_bp||)|")
    axes[1].set_xticks(np.arange(len(cond_labels)))
    axes[1].set_xticklabels(cond_labels, rotation=35, ha="right")
    axes[1].set_yticks(np.arange(len(comp_order)))
    axes[1].set_yticklabels(comp_order)
    cbar1 = fig.colorbar(im1, ax=axes[1], shrink=0.9)
    cbar1.set_label("scale mismatch")

    _save_dual(fig, args.output_base)
    print(f"Saved figure to {args.output_base.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


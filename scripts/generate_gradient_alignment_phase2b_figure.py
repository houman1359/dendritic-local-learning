#!/usr/bin/env python3
"""Plot gradient-fidelity summary for best Phase2b configurations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MNIST_CSV = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/diagnostics/neurips_alignment_phase2b_mnist_best/component_alignment_summary.csv"
)
DEFAULT_CONTEXT_CSV = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/diagnostics/neurips_alignment_phase2b_context_best/component_alignment_summary.csv"
)
DEFAULT_OUTPUT_BASE = Path(
    "drafts/dendritic-local-learning/figures/fig_gradient_alignment_phase2b_best"
)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for core_type, group in df.groupby("core_type"):
        weights = group["total_numel"].to_numpy(dtype=float)
        cosine = group["weighted_mean_cosine"].to_numpy(dtype=float)
        norm_ratio = group["mean_norm_ratio"].to_numpy(dtype=float)
        rel_l2 = group["mean_relative_l2_error"].to_numpy(dtype=float)

        wsum = np.sum(weights)
        if wsum <= 0:
            continue
        rows.append(
            {
                "core_type": str(core_type),
                "weighted_cosine": float(np.sum(cosine * weights) / wsum),
                "scale_mismatch": float(
                    np.sum(np.abs(np.log10(np.maximum(norm_ratio, 1e-12))) * weights)
                    / wsum
                ),
                "relative_l2": float(np.sum(rel_l2 * weights) / wsum),
            }
        )
    return pd.DataFrame(rows)


def _save_dual(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mnist-csv", type=Path, default=DEFAULT_MNIST_CSV)
    parser.add_argument("--context-csv", type=Path, default=DEFAULT_CONTEXT_CSV)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()

    mnist = pd.read_csv(args.mnist_csv)
    context = pd.read_csv(args.context_csv)

    if mnist.empty or context.empty:
        raise RuntimeError("Missing data in one or both summary CSV files.")

    mnist_a = _aggregate(mnist)
    mnist_a["dataset"] = "MNIST"
    context_a = _aggregate(context)
    context_a["dataset"] = "Context Gating"
    merged = pd.concat([mnist_a, context_a], ignore_index=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
        }
    )

    datasets = ["MNIST", "Context Gating"]
    core_order = ["dendritic_shunting", "dendritic_additive"]
    labels = {"dendritic_shunting": "Shunting", "dendritic_additive": "Additive"}
    colors = {"dendritic_shunting": "#0B6E4F", "dendritic_additive": "#B02E0C"}
    x = np.arange(len(datasets))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), constrained_layout=True)

    # Panel A: weighted cosine (higher is better)
    for idx, core in enumerate(core_order):
        vals = []
        for ds in datasets:
            row = merged[(merged["dataset"] == ds) & (merged["core_type"] == core)]
            vals.append(float(row["weighted_cosine"].iloc[0]))
        axes[0].bar(
            x + (idx - 0.5) * width,
            vals,
            width=width,
            color=colors[core],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label=labels[core],
        )
    axes[0].set_title("A. Directional Fidelity (Local vs Backprop)")
    axes[0].set_ylabel("Weighted cosine similarity")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(datasets)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].legend(frameon=False, loc="upper right")

    # Panel B: scale mismatch (lower is better)
    for idx, core in enumerate(core_order):
        vals = []
        for ds in datasets:
            row = merged[(merged["dataset"] == ds) & (merged["core_type"] == core)]
            vals.append(float(row["scale_mismatch"].iloc[0]))
        axes[1].bar(
            x + (idx - 0.5) * width,
            vals,
            width=width,
            color=colors[core],
            alpha=0.9,
            edgecolor="black",
            linewidth=0.5,
            label=labels[core],
        )
    axes[1].set_title("B. Gradient Scale Fidelity")
    axes[1].set_ylabel(r"$|\log_{10}(\|g_{local}\|/\|g_{bp}\|)|$ (weighted)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(datasets)

    _save_dual(fig, args.output_base)
    print(f"Saved figure: {args.output_base.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


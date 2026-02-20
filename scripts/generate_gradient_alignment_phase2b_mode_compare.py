#!/usr/bin/env python3
"""Plot per_soma vs scalar gradient-fidelity comparison for Phase2b best configs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_MNIST_CSV = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/diagnostics/neurips_alignment_phase2b_mnist_ckpt_modes/component_alignment_summary.csv"
)
DEFAULT_CONTEXT_CSV = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/diagnostics/neurips_alignment_phase2b_context_ckpt_modes/component_alignment_summary.csv"
)
DEFAULT_OUTPUT_BASE = Path(
    "drafts/dendritic-local-learning/figures/fig_gradient_alignment_phase2b_mode_compare"
)


def _aggregate(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for (core_type, mode), group in df.groupby(
        ["core_type", "error_broadcast_mode"], dropna=False
    ):
        w = group["total_numel"].to_numpy(dtype=float)
        wsum = np.sum(w)
        if wsum <= 0:
            continue
        cosine = group["weighted_mean_cosine"].to_numpy(dtype=float)
        ratio = group["mean_norm_ratio"].to_numpy(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "core_type": str(core_type),
                "mode": str(mode),
                "weighted_cosine": float(np.sum(cosine * w) / wsum),
                "scale_mismatch": float(
                    np.sum(np.abs(np.log10(np.maximum(ratio, 1e-12))) * w) / wsum
                ),
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
    df = pd.concat(
        [
            _aggregate(mnist, "MNIST"),
            _aggregate(context, "Context Gating"),
        ],
        ignore_index=True,
    )
    if df.empty:
        raise RuntimeError("No rows to plot.")

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
    cores = ["dendritic_shunting", "dendritic_additive"]
    modes = ["per_soma", "scalar"]

    x = np.arange(len(datasets) * len(cores))
    width = 0.36
    tick_labels = []
    for ds in datasets:
        for core in cores:
            tick_labels.append(f"{ds}\n{'Shunting' if core=='dendritic_shunting' else 'Additive'}")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), constrained_layout=True)

    for mi, mode in enumerate(modes):
        vals_cos = []
        vals_scale = []
        for ds in datasets:
            for core in cores:
                row = df[
                    (df["dataset"] == ds)
                    & (df["core_type"] == core)
                    & (df["mode"] == mode)
                ]
                vals_cos.append(float(row["weighted_cosine"].iloc[0]))
                vals_scale.append(float(row["scale_mismatch"].iloc[0]))

        offset = (mi - 0.5) * width
        color = "#1D4E89" if mode == "per_soma" else "#A05A2C"
        label = "per_soma" if mode == "per_soma" else "scalar"
        axes[0].bar(
            x + offset,
            vals_cos,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )
        axes[1].bar(
            x + offset,
            vals_scale,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=label,
        )

    axes[0].set_title("A. Weighted Cosine(Local, Backprop)")
    axes[0].set_ylabel("higher is better")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tick_labels)
    axes[0].axhline(0.0, color="black", linewidth=0.8)

    axes[1].set_title(r"B. Scale Mismatch $|\log_{10}(\|g_l\|/\|g_{bp}\|)|$")
    axes[1].set_ylabel("lower is better")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tick_labels)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)

    _save_dual(fig, args.output_base)
    print(f"Saved figure: {args.output_base.with_suffix('.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


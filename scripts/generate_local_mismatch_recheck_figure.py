#!/usr/bin/env python3
"""Generate a compact figure for local_mismatch recheck sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/analysis/local_mismatch_recheck_20260224_summary.csv"
)
DEFAULT_OUTPUT = Path(
    "drafts/dendritic-local-learning/figures/fig_local_mismatch_recheck"
)


def _save_dual(fig: plt.Figure, out_base: Path) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".pdf"), dpi=350, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=350, bbox_inches="tight")


def _human_core(name: str) -> str:
    return "Shunting" if name == "dendritic_shunting" else "Additive"


def _human_mode(name: str) -> str:
    if name == "per_soma":
        return "Per-soma"
    if name == "local_mismatch":
        return "Local mismatch"
    return name


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise RuntimeError(f"No rows in {args.input_csv}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
        }
    )

    cores = ["dendritic_additive", "dendritic_shunting"]
    modes = ["per_soma", "local_mismatch"]
    decoders = ["local", "backprop"]
    colors = {"per_soma": "#0B6E4F", "local_mismatch": "#A03A2C"}
    marker = {"local": "o", "backprop": "D"}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax_idx, decoder in enumerate(decoders):
        ax = axes[ax_idx]
        subset = df[df["decoder_update_mode"] == decoder]
        x = np.arange(len(cores))
        width = 0.35
        for i, mode in enumerate(modes):
            vals = []
            errs = []
            for core in cores:
                group = subset[(subset["core_type"] == core) & (subset["error_broadcast_mode"] == mode)]
                vals.append(float(group["test_acc"].mean()) if not group.empty else np.nan)
                errs.append(float(group["test_acc"].std(ddof=1)) if len(group) > 1 else 0.0)
            ax.bar(
                x + (i - 0.5) * width,
                vals,
                width=width,
                color=colors[mode],
                edgecolor="black",
                linewidth=0.7,
                alpha=0.9,
                yerr=errs,
                capsize=3.0,
                label=_human_mode(mode),
            )

            # Overlay seed points.
            for xi, core in enumerate(cores):
                points = subset[
                    (subset["core_type"] == core)
                    & (subset["error_broadcast_mode"] == mode)
                ]["test_acc"].to_numpy(dtype=float)
                if points.size == 0:
                    continue
                jitter = np.linspace(-0.04, 0.04, points.size)
                ax.scatter(
                    np.full(points.size, x[xi] + (i - 0.5) * width) + jitter,
                    points,
                    color="#111111",
                    s=14,
                    marker=marker[decoder],
                    zorder=5,
                )

        ax.set_title(f"Decoder update: {decoder}")
        ax.set_xticks(x)
        ax.set_xticklabels([_human_core(core) for core in cores])
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(0.0, 1.0)

    axes[0].set_ylabel("Test accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Local mismatch recheck (MNIST, 5F)", y=1.10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_dual(fig, args.output_base)
    print(f"Saved {args.output_base}.pdf/.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

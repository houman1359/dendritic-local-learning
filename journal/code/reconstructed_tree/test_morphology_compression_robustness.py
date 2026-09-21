#!/usr/bin/env python3
"""Repeat the real-tree feedback-compression test over independent RNG seeds.

Random seeds are Monte Carlo sensitivity checks, not biological replicates.  All
tests and hierarchical intervals therefore retain the eight MICrONS cells as
the top-level replication unit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze_microns_morphology_credit import (
    PROJECT,
    ancestry_matrix,
    cell_rng,
    parent_map,
    reconstruction_curves,
    write_json,
)


DEFAULT_INPUT = PROJECT / "results" / "microns_morphology_credit"
DEFAULT_OUTPUT = PROJECT / "results" / "microns_morphology_compression_robustness"
METHODS = [
    "dense PCA oracle",
    "morphology-aware paths",
    "random paths",
    "depth-only bins",
    "shuffled ancestry",
]
COLORS = {
    "dense PCA oracle": "#e69f00",
    "morphology-aware paths": "#26828e",
    "random paths": "#999999",
    "depth-only bins": "#7a5195",
    "shuffled ancestry": "#cc79a7",
}


def reconstruct_inputs(segments: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Recover the modeled ancestry kernel from saved per-segment quantities."""

    _, parent, _ = parent_map(segments)
    e_segments = [int(x) for x in segments.loc[segments["E_size"] > 0, "segment_id"]]
    i_segments = [int(x) for x in segments.loc[segments["I_size"] > 0, "segment_id"]]
    indexed = segments.set_index("segment_id")
    ancestry = ancestry_matrix(e_segments, i_segments, parent)
    beta = (
        indexed.loc[i_segments, "g_i"].to_numpy(dtype=float)
        / indexed.loc[i_segments, "g_total"].to_numpy(dtype=float)
    )
    kernel = ancestry * beta[None, :]
    e_weight = indexed.loc[e_segments, "E_size"].to_numpy(dtype=float)
    e_depth = indexed.loc[e_segments, "path_length_um"].to_numpy(dtype=float)
    domain_fraction = (ancestry.T @ e_weight) / max(float(e_weight.sum()), 1e-12)
    leverage = beta * domain_fraction
    return kernel, e_weight, e_depth, leverage


def hierarchical_ci(
    frame: pd.DataFrame,
    value: str,
    rng: np.random.Generator,
    n_boot: int = 20_000,
) -> list[float]:
    """Bootstrap cells, then draw one Monte Carlo seed within each sampled cell."""

    groups = {int(root): group[value].to_numpy(dtype=float) for root, group in frame.groupby("root_id")}
    roots = np.asarray(sorted(groups), dtype=np.int64)
    draws = np.empty(int(n_boot), dtype=float)
    for index in range(int(n_boot)):
        selected = rng.choice(roots, size=len(roots), replace=True)
        draws[index] = np.mean([rng.choice(groups[int(root)]) for root in selected])
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def paired_summary(wide: pd.DataFrame, control: str, rng: np.random.Generator) -> dict[str, Any]:
    paired = wide[["root_id", "seed", "morphology-aware paths", control]].dropna().copy()
    paired["difference"] = paired["morphology-aware paths"] - paired[control]
    cell_means = paired.groupby("root_id", as_index=False)["difference"].mean()
    differences = cell_means["difference"].to_numpy(dtype=float)
    test = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    per_seed = paired.groupby("seed", as_index=False)["difference"].mean()
    return {
        "control": control,
        "n_cells": int(cell_means["root_id"].nunique()),
        "n_seeds": int(paired["seed"].nunique()),
        "mean_cell_difference": float(differences.mean()),
        "hierarchical_bootstrap_ci95": hierarchical_ci(paired, "difference", rng),
        "wilcoxon_cell_mean_p": float(test.pvalue),
        "cells_with_lower_morphology_residual": int((differences < 0).sum()),
        "cell_seed_fraction_lower": float((paired["difference"] < 0).mean()),
        "seed_grand_mean_difference_range": [
            float(per_seed["difference"].min()),
            float(per_seed["difference"].max()),
        ],
    }


def make_figure(curves: pd.DataFrame, outdir: Path) -> None:
    seed_means = curves.groupby(["seed", "channels", "method"], as_index=False)["residual"].mean()
    summary = (
        seed_means.groupby(["channels", "method"])["residual"]
        .agg(mean="mean", q025=lambda x: np.quantile(x, 0.025), q975=lambda x: np.quantile(x, 0.975))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    ax = axes[0]
    for method in METHODS:
        part = summary[summary["method"] == method].sort_values("channels")
        if part.empty:
            continue
        x = part["channels"].to_numpy(dtype=float)
        ax.plot(x, part["mean"], "-o", ms=4, color=COLORS[method], label=method)
        ax.fill_between(
            x,
            part["q025"].to_numpy(dtype=float),
            part["q975"].to_numpy(dtype=float),
            color=COLORS[method],
            alpha=0.13,
            linewidth=0,
        )
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("feedback channels")
    ax.set_ylabel("weighted reconstruction residual")
    ax.set_title("A  Compression across channel budgets")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    controls = ["random paths", "depth-only bins", "shuffled ancestry"]
    selected = curves[curves["channels"] == 8]
    wide = selected.pivot_table(
        index=["root_id", "seed"], columns="method", values="residual"
    ).reset_index()
    interval_rng = np.random.default_rng(20260721 + 2_000_003)
    for index, control in enumerate(controls):
        paired = wide[["root_id", "seed", "morphology-aware paths", control]].dropna().copy()
        paired["difference"] = paired["morphology-aware paths"] - paired[control]
        cell_means = paired.groupby("root_id", as_index=False)["difference"].mean()
        jitter = np.linspace(-0.11, 0.11, len(cell_means))
        ax.scatter(
            np.full(len(cell_means), index, dtype=float) + jitter,
            cell_means["difference"],
            s=24,
            color="#4d4d4d",
            alpha=0.72,
            zorder=2,
        )
        mean = float(cell_means["difference"].mean())
        low, high = hierarchical_ci(paired, "difference", interval_rng)
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            ms=6,
            color="#d95f5f",
            capsize=4,
            lw=1.7,
            zorder=3,
        )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(range(len(controls)), ["random\npaths", "depth-only\nbins", "shuffled\nancestry"])
    ax.set_ylabel("morphology minus control residual")
    ax.set_title("B  Eight-channel paired effects")
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.02,
        0.02,
        "gray: cell means; red: mean and hierarchical 95% CI",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(outdir / "compression_seed_robustness.png", dpi=260)
    fig.savefig(outdir / "compression_seed_robustness.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-seed", type=int, default=20260721)
    parser.add_argument("--n-seeds", type=int, default=50)
    parser.add_argument("--n-samples", type=int, default=384)
    parser.add_argument("--n-random", type=int, default=64)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    segments = pd.read_csv(args.input_dir / "segment_metrics.csv")
    prepared = {
        int(root): reconstruct_inputs(group.copy())
        for root, group in segments.groupby("root_id", sort=True)
    }
    rows: list[dict[str, Any]] = []
    for offset in range(int(args.n_seeds)):
        seed = int(args.base_seed) + offset
        print(f"compression robustness seed {offset + 1}/{args.n_seeds}", flush=True)
        for root_id, (kernel, e_weight, e_depth, leverage) in prepared.items():
            curves = reconstruction_curves(
                kernel,
                e_weight,
                e_depth,
                leverage,
                cell_rng(seed, root_id, stream=2),
                n_samples=args.n_samples,
                n_random=args.n_random,
            )
            for row in curves:
                row.update({"root_id": root_id, "seed": seed})
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(args.outdir / "compression_seed_curves.csv.gz", index=False, compression="gzip")
    cell_means = frame.groupby(["root_id", "channels", "method"], as_index=False)["residual"].mean()
    cell_means.to_csv(args.outdir / "cell_method_means.csv", index=False)
    make_figure(frame, args.outdir)

    channel = 8
    selected = frame[frame["channels"] == channel]
    wide = selected.pivot_table(index=["root_id", "seed"], columns="method", values="residual").reset_index()
    rng = np.random.default_rng(args.base_seed + 1_000_003)
    method_summaries: dict[str, Any] = {}
    for method in METHODS:
        values = selected[selected["method"] == method]
        per_seed = values.groupby("seed", as_index=False)["residual"].mean()
        method_summaries[method] = {
            "grand_mean": float(values["residual"].mean()),
            "hierarchical_bootstrap_ci95": hierarchical_ci(values, "residual", rng),
            "seed_grand_mean_range": [float(per_seed["residual"].min()), float(per_seed["residual"].max())],
        }
    controls = ["random paths", "depth-only bins", "shuffled ancestry"]
    comparisons = {control: paired_summary(wide, control, rng) for control in controls}
    summary = {
        "status": "complete",
        "interpretation_unit": "MICrONS cell; random seeds are sensitivity checks, not replicates",
        "n_cells": int(frame["root_id"].nunique()),
        "n_seeds": int(frame["seed"].nunique()),
        "n_samples_per_cell_seed": int(args.n_samples),
        "random_dictionaries_per_count": int(args.n_random),
        "focus_channels": channel,
        "method_residuals": method_summaries,
        "morphology_comparisons": comparisons,
    }
    write_json(args.outdir / "summary.json", summary)

    lines = [
        "# Morphology-compression Monte Carlo robustness",
        "",
        f"The analysis repeated the perturbation-field reconstruction over {args.n_seeds} independent seeds for each of {summary['n_cells']} real MICrONS cells. Seeds are sensitivity checks; the cell remains the replication unit.",
        "",
        "## Eight-channel result",
        "",
    ]
    for method in METHODS:
        item = method_summaries[method]
        lines.append(
            f"- {method}: residual {item['grand_mean']:.3f}; hierarchical 95% CI [{item['hierarchical_bootstrap_ci95'][0]:.3f}, {item['hierarchical_bootstrap_ci95'][1]:.3f}]."
        )
    lines.extend(["", "## Cell-level comparisons", ""])
    for control in controls:
        item = comparisons[control]
        lines.append(
            f"- Morphology minus {control}: {item['mean_cell_difference']:.3f} "
            f"(95% CI [{item['hierarchical_bootstrap_ci95'][0]:.3f}, {item['hierarchical_bootstrap_ci95'][1]:.3f}]); "
            f"lower in {item['cells_with_lower_morphology_residual']}/{item['n_cells']} cell means; "
            f"cell-level Wilcoxon p={item['wilcoxon_cell_mean_p']:.4f}."
        )
    lines.extend(
        [
            "",
            "These intervals include both between-cell and Monte Carlo variability. They do not address uncertainty from cell selection, connectome reconstruction, E/I proxy labels, or the modeled electrical calibration.",
            "",
        ]
    )
    (args.outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

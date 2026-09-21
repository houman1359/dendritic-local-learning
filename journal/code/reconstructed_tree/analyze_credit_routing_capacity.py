#!/usr/bin/env python3
"""Translate feedback reconstruction into a constrained-learning quantity.

For an exact weighted gradient g and its orthogonal projection P_D g onto a
feedback dictionary D, 1 - residual**2 is the fraction of gradient energy
captured.  For an L-smooth loss and step size 1/L, this is also the ratio of
the standard one-step descent guarantee for projected versus full-gradient
descent.  Random streams remain sensitivity checks; MICrONS cells are the
replication units.
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


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_CURVES = (
    PROJECT
    / "results"
    / "microns_morphology_compression_robustness"
    / "compression_seed_curves.csv.gz"
)
DEFAULT_CELLS = PROJECT / "results" / "microns_morphology_credit" / "cell_metrics.csv"
DEFAULT_OUTDIR = PROJECT / "results" / "microns_credit_routing_capacity"
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
    "random paths": "#8c8c8c",
    "depth-only bins": "#7a5195",
    "shuffled ancestry": "#cc79a7",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hierarchical_ci(
    frame: pd.DataFrame,
    value: str,
    rng: np.random.Generator,
    n_boot: int = 20_000,
) -> list[float]:
    groups = {
        int(root): group[value].dropna().to_numpy(dtype=float)
        for root, group in frame.groupby("root_id")
    }
    groups = {root: values for root, values in groups.items() if len(values)}
    roots = np.asarray(sorted(groups), dtype=np.int64)
    draws = np.empty(int(n_boot), dtype=float)
    for index in range(int(n_boot)):
        selected = rng.choice(roots, size=len(roots), replace=True)
        draws[index] = np.mean([rng.choice(groups[int(root)]) for root in selected])
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def paired_capture_summary(
    wide: pd.DataFrame,
    control: str,
    rng: np.random.Generator,
) -> dict[str, Any]:
    paired = wide[["root_id", "seed", "morphology-aware paths", control]].dropna().copy()
    paired["capture_advantage"] = paired["morphology-aware paths"] - paired[control]
    cell_means = paired.groupby("root_id", as_index=False)["capture_advantage"].mean()
    difference = cell_means["capture_advantage"].to_numpy(dtype=float)
    test = stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
    return {
        "control": control,
        "mean_capture_advantage": float(difference.mean()),
        "hierarchical_ci95": hierarchical_ci(paired, "capture_advantage", rng),
        "cells_positive": int((difference > 0).sum()),
        "n_cells": int(len(difference)),
        "cell_level_wilcoxon_p": float(test.pvalue),
    }


def make_figure(frame: pd.DataFrame, summary: dict[str, Any], outdir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))

    ax = axes[0]
    seed_means = frame.groupby(["seed", "channels", "method"], as_index=False)["credit_capture"].mean()
    curve = (
        seed_means.groupby(["channels", "method"])["credit_capture"]
        .agg(mean="mean", q025=lambda x: np.quantile(x, 0.025), q975=lambda x: np.quantile(x, 0.975))
        .reset_index()
    )
    for method in METHODS:
        part = curve[curve["method"] == method].sort_values("channels")
        x = part["channels"].to_numpy(dtype=float)
        ax.plot(x, part["mean"], "-o", ms=3.5, color=COLORS[method], label=method)
        ax.fill_between(
            x,
            part["q025"].to_numpy(dtype=float),
            part["q975"].to_numpy(dtype=float),
            color=COLORS[method],
            alpha=0.12,
            linewidth=0,
        )
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("feedback channels")
    ax.set_ylabel("captured credit energy  $1-r^2$")
    ax.set_title("A  Predicted descent capacity")
    ax.legend(frameon=False, fontsize=6.5)

    ax = axes[1]
    eight = frame[frame["channels"] == 8]
    cell = eight.groupby(["root_id", "method"], as_index=False).agg(
        credit_capture=("credit_capture", "mean"),
        wiring_density=("wiring_density", "mean"),
    )
    for method in METHODS:
        part = cell[cell["method"] == method]
        ax.scatter(
            part["wiring_density"],
            part["credit_capture"],
            s=17,
            alpha=0.28,
            color=COLORS[method],
        )
        ax.scatter(
            part["wiring_density"].mean(),
            part["credit_capture"].mean(),
            s=58,
            marker="D",
            edgecolor="white",
            linewidth=0.7,
            color=COLORS[method],
            label=method,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.025, 1.25)
    ax.set_ylim(0, 0.72)
    ax.set_xlabel("feedback wiring density")
    ax.set_ylabel("captured credit energy")
    ax.set_title("B  Eight-channel efficiency frontier")

    ax = axes[2]
    controls = ["random paths", "depth-only bins", "shuffled ancestry"]
    capture_wide = eight.pivot_table(
        index=["root_id", "seed"], columns="method", values="credit_capture"
    ).reset_index()
    rng = np.random.default_rng(20260721 + 5_000_003)
    for index, control in enumerate(controls):
        paired = capture_wide[["root_id", "seed", "morphology-aware paths", control]].dropna().copy()
        paired["advantage"] = paired["morphology-aware paths"] - paired[control]
        cell_mean = paired.groupby("root_id", as_index=False)["advantage"].mean()
        jitter = np.linspace(-0.11, 0.11, len(cell_mean))
        ax.scatter(
            np.full(len(cell_mean), index) + jitter,
            cell_mean["advantage"],
            s=22,
            color="#555555",
            alpha=0.7,
        )
        mean = float(cell_mean["advantage"].mean())
        low, high = hierarchical_ci(paired, "advantage", rng)
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            color="#d95f5f",
            ms=6,
            capsize=4,
            lw=1.6,
        )
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xticks(range(3), ["random\npaths", "depth-only\nbins", "shuffled\nancestry"])
    ax.set_ylabel("morphology capture advantage")
    ax.set_title("C  Cell-level topology advantage")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Real dendritic topology converts sparse feedback into captured credit",
        fontsize=12,
        weight="bold",
    )
    fig.tight_layout()
    fig.savefig(outdir / "microns_credit_routing_capacity.png", dpi=280, bbox_inches="tight")
    fig.savefig(outdir / "microns_credit_routing_capacity.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=Path, default=DEFAULT_CURVES)
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.curves)
    cells = pd.read_csv(args.cells)[["root_id", "n_e_segments"]]
    frame = frame.merge(cells, on="root_id", how="left", validate="many_to_one")
    if frame["wiring_nonzeros"].isna().any():
        raise ValueError("wiring_nonzeros is missing; rerun the robustness analysis with current code")
    frame["credit_capture"] = np.clip(1.0 - frame["residual"] ** 2, 0.0, 1.0)
    frame["wiring_density"] = frame["wiring_nonzeros"] / (
        frame["n_e_segments"] * frame["channels"]
    )
    oracle = frame[frame["method"] == "dense PCA oracle"][
        ["root_id", "seed", "channels", "credit_capture"]
    ].rename(columns={"credit_capture": "oracle_capture"})
    frame = frame.merge(oracle, on=["root_id", "seed", "channels"], how="left", validate="many_to_one")
    frame["oracle_fraction"] = frame["credit_capture"] / frame["oracle_capture"].clip(lower=1e-12)
    frame.to_csv(args.outdir / "routing_capacity_curves.csv.gz", index=False, compression="gzip")

    focus = frame[frame["channels"] == 8]
    wide = focus.pivot_table(
        index=["root_id", "seed"], columns="method", values="credit_capture"
    ).reset_index()
    rng = np.random.default_rng(20260721 + 4_000_003)
    methods: dict[str, Any] = {}
    for method in METHODS:
        part = focus[focus["method"] == method]
        cell_mean = part.groupby("root_id", as_index=False).agg(
            credit_capture=("credit_capture", "mean"),
            oracle_fraction=("oracle_fraction", "mean"),
            wiring_density=("wiring_density", "mean"),
        )
        methods[method] = {
            "mean_credit_capture": float(cell_mean["credit_capture"].mean()),
            "hierarchical_ci95": hierarchical_ci(part, "credit_capture", rng),
            "mean_oracle_fraction": float(cell_mean["oracle_fraction"].mean()),
            "mean_wiring_density": float(cell_mean["wiring_density"].mean()),
        }
    controls = ["random paths", "depth-only bins", "shuffled ancestry"]
    comparisons = {
        control: paired_capture_summary(wide, control, rng) for control in controls
    }
    summary = {
        "status": "complete",
        "hypothesis": (
            "A dendritic tree is a sparse physical decoder for low-bandwidth teaching signals; "
            "learning efficiency depends on alignment between its ancestry dictionary and task credit covariance."
        ),
        "theory_quantity": (
            "credit_capture = 1 - weighted_projection_residual^2; for an L-smooth loss and step 1/L, "
            "this is the ratio of projected-gradient to full-gradient one-step descent guarantees"
        ),
        "evidence_scope": "real MICrONS anatomy with modeled conductance and perturbation fields",
        "n_cells": int(frame["root_id"].nunique()),
        "n_monte_carlo_streams": int(frame["seed"].nunique()),
        "focus_channels": 8,
        "methods": methods,
        "morphology_comparisons": comparisons,
        "boundaries": [
            "The perturbation covariance is generated from the same conductance-tree model; this tests anatomical capacity, not endogenous learning in the mouse.",
            "Random streams quantify Monte Carlo sensitivity and are not biological replicates.",
            "Task utility requires independent task-derived credit fields and learning experiments.",
        ],
    }
    write_json(args.outdir / "summary.json", summary)
    make_figure(frame, summary, args.outdir)

    morphology = methods["morphology-aware paths"]
    lines = [
        "# MICrONS topology-matched credit capacity",
        "",
        summary["hypothesis"],
        "",
        "For an orthogonally projected exact gradient, `1 - residual^2` is the captured gradient-energy fraction and the relative one-step descent guarantee at step size `1/L` for an `L`-smooth loss.",
        "",
        "## Eight-channel result",
        "",
        f"Morphology-aware paths capture {morphology['mean_credit_capture']:.3f} of modeled credit energy at {100*morphology['mean_wiring_density']:.2f}% wiring density, or {morphology['mean_oracle_fraction']:.3f} of the dense eight-channel oracle's captured energy.",
        "",
    ]
    for control in controls:
        item = comparisons[control]
        lines.append(
            f"- Capture advantage over {control}: {item['mean_capture_advantage']:.3f} "
            f"(hierarchical 95% CI [{item['hierarchical_ci95'][0]:.3f}, {item['hierarchical_ci95'][1]:.3f}]); "
            f"positive in {item['cells_positive']}/{item['n_cells']} cell means; "
            f"cell-level Wilcoxon p={item['cell_level_wilcoxon_p']:.4f}."
        )
    lines.extend(["", "## Boundary", "", *[f"- {x}" for x in summary["boundaries"]], ""])
    (args.outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tier-2 analyses: depth scaling and noise robustness.

Run after sweeps complete:
    python analyze_tier2_results.py \
        --depth-sweep sweep_neurips_depth_scaling_20260221184328 \
        --noise-sweep sweep_neurips_noise_robustness_20260221184442 \
        --output-dir analysis/mechanistic
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TEXTWIDTH = 5.5
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SHUNTING_COLOR = "#2ca02c"
ADDITIVE_COLOR = "#1f77b4"
SWEEP_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)


def _dig(d: dict, keys: list[str], default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def extract_sweep_results(sweep_dir: Path) -> pd.DataFrame:
    """Extract final test accuracy + config metadata from all configs in a sweep."""
    results = []
    results_dir = sweep_dir / "results"
    if not results_dir.exists():
        print(f"  WARNING: {results_dir} does not exist")
        return pd.DataFrame()

    for cfg_dir in sorted(results_dir.iterdir()):
        if not cfg_dir.is_dir() or not cfg_dir.name.startswith("config_"):
            continue
        cfg_json = cfg_dir / "config.json"
        if not cfg_json.exists():
            continue
        with open(cfg_json) as f:
            cfg = json.load(f)

        # Extract metadata
        core_type = _dig(cfg, ["model", "core", "type"], "unknown")
        strategy = _dig(cfg, ["training", "main", "strategy"], "unknown")
        seed = _dig(cfg, ["experiment", "seed"], -1)
        branch_factors = _dig(
            cfg, ["model", "core", "architecture", "excitatory_branch_factors"], []
        )
        depth = len(branch_factors)

        # Rule variant
        rule = _dig(
            cfg,
            ["training", "main", "learning_strategy_config", "rule_variant"],
            "",
        )
        # Noise sigma
        noise_sigma = _dig(
            cfg,
            ["training", "main", "learning_strategy_config", "error_noise_sigma"],
            0.0,
        )

        # Extract test accuracy
        perf_dir = cfg_dir / "performance"
        test_acc = None

        # Try final.json (primary format)
        final_json = perf_dir / "final.json"
        if final_json.exists():
            with open(final_json) as f:
                pdata = json.load(f)
            acc_data = pdata.get("accuracy", {})
            if isinstance(acc_data, dict):
                test_acc = acc_data.get("test")
            elif isinstance(acc_data, (int, float)):
                test_acc = acc_data

        # Try other formats
        if test_acc is None:
            for perf_file in ["test_accuracy.json", "final_metrics.json"]:
                p = perf_dir / perf_file
                if p.exists():
                    with open(p) as f:
                        pdata = json.load(f)
                    test_acc = pdata.get("test_accuracy", pdata.get("accuracy"))
                    break

        # Try train.log as fallback
        if test_acc is None:
            log_path = cfg_dir / "train.log"
            if log_path.exists():
                with open(log_path) as f:
                    for line in f:
                        if "test_accuracy" in line or "Test accuracy" in line:
                            # Try to parse
                            parts = line.strip().split()
                            for i, p in enumerate(parts):
                                if "accuracy" in p.lower() and i + 1 < len(parts):
                                    try:
                                        test_acc = float(
                                            parts[i + 1].strip(",").strip(":")
                                        )
                                    except ValueError:
                                        pass

        # Try performance CSV
        if test_acc is None:
            for csv_name in ["test_results.csv", "metrics.csv"]:
                csv_path = perf_dir / csv_name
                if csv_path.exists():
                    try:
                        df_perf = pd.read_csv(csv_path)
                        if "test_accuracy" in df_perf.columns:
                            test_acc = df_perf["test_accuracy"].iloc[-1]
                        elif "accuracy" in df_perf.columns:
                            test_acc = df_perf["accuracy"].iloc[-1]
                    except Exception:
                        pass

        # Try mnist/test_accuracy dir
        if test_acc is None:
            mnist_test = cfg_dir / "mnist" / "test_accuracy"
            if mnist_test.exists():
                try:
                    files = sorted(mnist_test.iterdir())
                    if files:
                        with open(files[-1]) as f:
                            test_acc = float(f.read().strip())
                except Exception:
                    pass

        results.append({
            "config": cfg_dir.name,
            "core_type": core_type,
            "strategy": strategy,
            "seed": seed,
            "depth": depth,
            "rule": rule,
            "noise_sigma": noise_sigma,
            "test_accuracy": test_acc,
            "branch_factors": str(branch_factors),
        })

    return pd.DataFrame(results)


# ════════════════════════════════════════════════════════════════
# Depth scaling analysis (Bartunov-style)
# ════════════════════════════════════════════════════════════════
def plot_depth_scaling(sweep_dir: Path, output_dir: Path):
    print(f"  Extracting results from {sweep_dir.name}...")
    df = extract_sweep_results(sweep_dir)
    if df.empty:
        print("  No results found, skipping depth scaling plot")
        return

    # Filter out configs with no test accuracy
    df_valid = df[df["test_accuracy"].notna()].copy()
    print(f"  {len(df_valid)}/{len(df)} configs have test accuracy")

    if df_valid.empty:
        print("  No valid results, skipping")
        return

    # Simplify core type
    df_valid["core"] = df_valid["core_type"].apply(
        lambda x: "shunting" if "shunting" in x else "additive"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))

    # Panel A: Absolute accuracy by depth
    for strategy, ls in [("standard", "--"), ("local_ca", "-")]:
        for core, color in [("shunting", SHUNTING_COLOR), ("additive", ADDITIVE_COLOR)]:
            sub = df_valid[
                (df_valid["core"] == core) & (df_valid["strategy"] == strategy)
            ]
            if sub.empty:
                continue
            agg = sub.groupby("depth")["test_accuracy"].agg(["mean", "sem"]).reset_index()
            label = f"{core.capitalize()} ({'BP' if strategy == 'standard' else '5F'})"
            ax1.errorbar(
                agg["depth"],
                agg["mean"],
                yerr=agg["sem"],
                color=color,
                linestyle=ls,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=label,
                capsize=2,
            )

    ax1.set_xlabel("Dendritic depth (# branch layers)")
    ax1.set_ylabel("Test accuracy")
    ax1.set_title("Accuracy vs depth")
    ax1.legend(frameon=False, fontsize=6)

    # Panel B: Gap (BP - local) by depth
    for core, color in [("shunting", SHUNTING_COLOR), ("additive", ADDITIVE_COLOR)]:
        bp = (
            df_valid[(df_valid["core"] == core) & (df_valid["strategy"] == "standard")]
            .groupby("depth")["test_accuracy"]
            .mean()
        )
        local = (
            df_valid[(df_valid["core"] == core) & (df_valid["strategy"] == "local_ca")]
            .groupby("depth")["test_accuracy"]
            .mean()
        )
        common_depths = sorted(set(bp.index) & set(local.index))
        if common_depths:
            gaps = [bp[d] - local[d] for d in common_depths]
            ax2.plot(
                common_depths,
                gaps,
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=f"{core.capitalize()}",
            )

    ax2.set_xlabel("Dendritic depth (# branch layers)")
    ax2.set_ylabel("BP − Local gap")
    ax2.set_title("Backprop gap vs depth")
    ax2.legend(frameon=False)
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle("Depth scaling (Bartunov-style)", fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_depth_scaling.{ext}")
    plt.close(fig)
    print(f"  Saved fig_depth_scaling.pdf/png")

    # Print summary
    print("\n  Depth scaling summary:")
    summary = df_valid.groupby(["depth", "core", "strategy"])["test_accuracy"].agg(
        ["mean", "std", "count"]
    )
    print(summary.to_string())


# ════════════════════════════════════════════════════════════════
# Noise robustness analysis
# ════════════════════════════════════════════════════════════════
def plot_noise_robustness(sweep_dir: Path, output_dir: Path):
    print(f"  Extracting results from {sweep_dir.name}...")
    df = extract_sweep_results(sweep_dir)
    if df.empty:
        print("  No results found, skipping noise robustness plot")
        return

    df_valid = df[df["test_accuracy"].notna()].copy()
    print(f"  {len(df_valid)}/{len(df)} configs have test accuracy")

    if df_valid.empty:
        print("  No valid results, skipping")
        return

    df_valid["core"] = df_valid["core_type"].apply(
        lambda x: "shunting" if "shunting" in x else "additive"
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.5))

    # Panel A: Accuracy vs noise level
    for core, color in [("shunting", SHUNTING_COLOR), ("additive", ADDITIVE_COLOR)]:
        sub = df_valid[df_valid["core"] == core]
        agg = sub.groupby("noise_sigma")["test_accuracy"].agg(["mean", "sem"]).reset_index()
        agg = agg.sort_values("noise_sigma")
        ax1.errorbar(
            agg["noise_sigma"],
            agg["mean"],
            yerr=agg["sem"],
            color=color,
            marker="o",
            markersize=4,
            linewidth=1.5,
            label=core.capitalize(),
            capsize=2,
        )

    ax1.set_xlabel("Error noise σ")
    ax1.set_ylabel("Test accuracy")
    ax1.set_title("Accuracy under noisy credit")
    ax1.set_xscale("symlog", linthresh=0.01)
    ax1.legend(frameon=False)

    # Panel B: Relative degradation (normalized to σ=0)
    for core, color in [("shunting", SHUNTING_COLOR), ("additive", ADDITIVE_COLOR)]:
        sub = df_valid[df_valid["core"] == core]
        agg = sub.groupby("noise_sigma")["test_accuracy"].mean().reset_index()
        agg = agg.sort_values("noise_sigma")
        baseline = agg[agg["noise_sigma"] == 0.0]["test_accuracy"].values
        if len(baseline) > 0:
            agg["relative"] = agg["test_accuracy"] / baseline[0]
            ax2.plot(
                agg["noise_sigma"],
                agg["relative"],
                color=color,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=core.capitalize(),
            )

    ax2.set_xlabel("Error noise σ")
    ax2.set_ylabel("Relative accuracy (vs σ=0)")
    ax2.set_title("Robustness to noisy feedback")
    ax2.set_xscale("symlog", linthresh=0.01)
    ax2.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax2.legend(frameon=False)

    fig.suptitle("Noise robustness of credit signals", fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_noise_robustness.{ext}")
    plt.close(fig)
    print(f"  Saved fig_noise_robustness.pdf/png")

    # Print summary
    print("\n  Noise robustness summary:")
    summary = df_valid.groupby(["noise_sigma", "core"])["test_accuracy"].agg(
        ["mean", "std", "count"]
    )
    print(summary.to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--depth-sweep",
        type=str,
        default="sweep_neurips_depth_scaling_20260221184328",
    )
    parser.add_argument(
        "--noise-sweep",
        type=str,
        default="sweep_neurips_noise_robustness_20260221184442",
    )
    parser.add_argument("--sweep-root", type=Path, default=SWEEP_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/mechanistic"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Depth scaling analysis ===")
    depth_dir = args.sweep_root / args.depth_sweep
    if depth_dir.exists():
        plot_depth_scaling(depth_dir, args.output_dir)
    else:
        print(f"  Sweep not found: {depth_dir}")

    print("\n=== Noise robustness analysis ===")
    noise_dir = args.sweep_root / args.noise_sweep
    if noise_dir.exists():
        plot_noise_robustness(noise_dir, args.output_dir)
    else:
        print(f"  Sweep not found: {noise_dir}")


if __name__ == "__main__":
    main()

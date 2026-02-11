#!/usr/bin/env python3
"""Generate NeurIPS-ready figures for the local credit assignment draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SWEEP_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)
DEFAULT_OUTPUT_DIR = Path(
    "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/figures"
)
DEFAULT_ANALYSIS_DIR = Path(
    "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/analysis"
)


def _latest_match(root: Path, pattern: str) -> Path:
    matches = sorted(root.glob(pattern), key=lambda path: path.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No sweep directory matches pattern: {pattern}")
    return matches[-1]


def _load_processed(sweep_dir: Path) -> pd.DataFrame:
    csv_path = sweep_dir / "plots" / "locallearning_processed_data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing processed data: {csv_path}")
    return pd.read_csv(csv_path)


def _save_dual(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def _configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
        }
    )


def _plot_decoder_locality(
    claim2_df: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, Path]:
    grouped = (
        claim2_df.groupby(["dataset", "decoder_update_mode"])["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    mode_order = ["backprop", "local", "none"]
    mode_colors = {"backprop": "#1D4E89", "local": "#0B6E4F", "none": "#A62639"}
    dataset_order = ["mnist", "cifar10"]
    dataset_titles = {"mnist": "MNIST", "cifar10": "CIFAR-10"}

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=False)

    for axis, dataset in zip(axes, dataset_order):
        subset = grouped[grouped["dataset"] == dataset].copy()
        subset["decoder_update_mode"] = pd.Categorical(
            subset["decoder_update_mode"], categories=mode_order, ordered=True
        )
        subset = subset.sort_values("decoder_update_mode")

        x_values = np.arange(len(subset))
        bars = axis.bar(
            x_values,
            subset["mean"],
            yerr=subset["std"].fillna(0.0),
            capsize=4,
            color=[mode_colors[mode] for mode in subset["decoder_update_mode"]],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.92,
        )
        axis.set_xticks(x_values)
        axis.set_xticklabels([mode.upper() for mode in subset["decoder_update_mode"]])
        axis.set_title(dataset_titles[dataset])
        axis.set_ylabel("Test Accuracy")
        axis.set_ylim(0.08, max(0.42, subset["mean"].max() + 0.06))
        axis.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, subset["mean"]):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("Decoder Locality: Local Matches Backprop, None Degrades", y=1.04)
    fig.tight_layout()

    out_base = output_dir / "fig_decoder_locality"
    _save_dual(fig, out_base)
    plt.close(fig)

    return grouped, out_base.with_suffix(".pdf")


def _plot_shunting_regime(
    claim3_df: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, Path]:
    grouped = (
        claim3_df.groupby(
            ["dataset", "network_type", "error_broadcast_mode", "ie_value"]
        )["test_accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    dataset_order = ["mnist", "cifar10"]
    dataset_titles = {"mnist": "MNIST", "cifar10": "CIFAR-10"}
    style_map = {
        ("dendritic_shunting", "per_soma"): ("#0B6E4F", "-", "o", "Shunting + per-soma"),
        ("dendritic_shunting", "scalar"): ("#1D4E89", "-", "s", "Shunting + scalar"),
        ("dendritic_additive", "per_soma"): ("#BF5B17", "--", "o", "Additive + per-soma"),
        ("dendritic_additive", "scalar"): ("#7F3C8D", "--", "s", "Additive + scalar"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=False)

    for axis, dataset in zip(axes, dataset_order):
        subset = grouped[grouped["dataset"] == dataset].copy()
        subset = subset.sort_values("ie_value")

        for (network_type, mode), (color, linestyle, marker, label) in style_map.items():
            line_data = subset[
                (subset["network_type"] == network_type)
                & (subset["error_broadcast_mode"] == mode)
            ]
            if line_data.empty:
                continue
            axis.errorbar(
                line_data["ie_value"],
                line_data["mean"],
                yerr=line_data["std"].fillna(0.0),
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.8,
                markersize=5,
                capsize=3,
                label=label,
                alpha=0.95,
            )

        axis.set_title(dataset_titles[dataset])
        axis.set_xlabel("Inhibitory Synapses per Branch (IE)")
        axis.set_ylabel("Test Accuracy")
        axis.set_xticks(sorted(subset["ie_value"].unique()))
        axis.grid(axis="y", alpha=0.25)
        if dataset == "mnist":
            axis.set_ylim(0.08, 0.40)
        else:
            axis.set_ylim(0.08, 0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.07),
    )
    fig.suptitle(
        "Regime Dependence: Shunting Dominates Additive Across Inhibition Levels", y=1.05
    )
    fig.tight_layout()

    out_base = output_dir / "fig_shunting_regime"
    _save_dual(fig, out_base)
    plt.close(fig)

    return grouped, out_base.with_suffix(".pdf")


def _plot_source_interaction(
    claim4_df: pd.DataFrame, output_dir: Path
) -> tuple[pd.DataFrame, Path]:
    metrics = ["test_accuracy", "mi_E_I_C", "mi_V_C"]
    available_metrics = [metric for metric in metrics if metric in claim4_df.columns]
    grouped = (
        claim4_df.groupby(["error_broadcast_mode", "use_path_propagation"])[
            available_metrics
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(column).rstrip("_") if isinstance(column, tuple) else str(column)
        for column in grouped.columns
    ]

    mode_order = ["scalar", "per_soma"]
    mode_colors = {"scalar": "#1D4E89", "per_soma": "#0B6E4F"}
    path_offsets = {False: -0.16, True: 0.16}
    path_markers = {False: "o", True: "D"}

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # Left panel: test accuracy grouped bars
    x_base = np.arange(len(mode_order))
    bar_width = 0.3
    for path_flag in [False, True]:
        subset = grouped[grouped["use_path_propagation"] == path_flag]
        subset = subset.set_index("error_broadcast_mode").reindex(mode_order).reset_index()
        x_values = x_base + path_offsets[path_flag]
        axes[0].bar(
            x_values,
            subset["test_accuracy_mean"],
            width=bar_width,
            yerr=subset["test_accuracy_std"].fillna(0.0),
            capsize=3,
            color=[
                mode_colors[str(mode)] if mode in mode_colors else "#999999"
                for mode in subset["error_broadcast_mode"]
            ],
            alpha=0.45 if not path_flag else 0.88,
            edgecolor="black",
            linewidth=0.6,
            label=f"path={str(path_flag).lower()}",
        )

    axes[0].set_xticks(x_base)
    axes[0].set_xticklabels(["SCALAR", "PER-SOMA"])
    axes[0].set_ylabel("Test Accuracy")
    axes[0].set_ylim(0.33, 0.40)
    axes[0].set_title("Accuracy by Broadcast and Path")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, loc="lower right")

    # Right panel: MI(E,I;C) vs test accuracy
    if "mi_E_I_C_mean" in grouped.columns:
        x_metric = "mi_E_I_C_mean"
        x_label = "MI(E,I;C)"
    elif "mi_V_C_mean" in grouped.columns:
        x_metric = "mi_V_C_mean"
        x_label = "MI(V;C)"
    else:
        x_metric = "test_accuracy_mean"
        x_label = "Test Accuracy"

    for _, row in grouped.iterrows():
        mode = row["error_broadcast_mode"]
        path_flag = bool(row["use_path_propagation"])
        axes[1].scatter(
            row[x_metric],
            row["test_accuracy_mean"],
            s=75,
            marker=path_markers[path_flag],
            color=mode_colors[str(mode)],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.92,
        )
        label = f"{mode}, path={str(path_flag).lower()}"
        axes[1].annotate(
            label,
            (row[x_metric], row["test_accuracy_mean"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=8,
        )

    for mode in mode_order:
        mode_rows = grouped[grouped["error_broadcast_mode"] == mode].set_index(
            "use_path_propagation"
        )
        if False in mode_rows.index and True in mode_rows.index:
            axes[1].annotate(
                "",
                xy=(mode_rows.loc[True, x_metric], mode_rows.loc[True, "test_accuracy_mean"]),
                xytext=(
                    mode_rows.loc[False, x_metric],
                    mode_rows.loc[False, "test_accuracy_mean"],
                ),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "color": mode_colors[mode]},
            )

    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Test Accuracy")
    axes[1].set_title("Path Interaction in Representation-Performance Space")
    axes[1].grid(alpha=0.25)

    fig.suptitle("Source Analysis: Broadcast/Path Interaction is Metric-Dependent", y=1.04)
    fig.tight_layout()

    out_base = output_dir / "fig_source_interaction"
    _save_dual(fig, out_base)
    plt.close(fig)

    return grouped, out_base.with_suffix(".pdf")


def _build_effect_summary(
    claim2_df: pd.DataFrame, claim3_df: pd.DataFrame, claim4_df: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []

    # Claim2: local vs backprop/none
    claim2_pivot = claim2_df.pivot_table(
        index=["dataset", "seed"], columns="decoder_update_mode", values="test_accuracy"
    ).reset_index()
    for dataset in sorted(claim2_pivot["dataset"].unique()):
        subset = claim2_pivot[claim2_pivot["dataset"] == dataset]
        rows.append(
            {
                "effect": f"{dataset}: local - backprop",
                "value": float((subset["local"] - subset["backprop"]).mean()),
            }
        )
        rows.append(
            {
                "effect": f"{dataset}: local - none",
                "value": float((subset["local"] - subset["none"]).mean()),
            }
        )

    # Claim3: shunting vs additive at matched IE and broadcast
    grouped3 = (
        claim3_df.groupby(
            ["dataset", "network_type", "error_broadcast_mode", "ie_value"]
        )["test_accuracy"]
        .mean()
        .reset_index()
    )
    shunting = grouped3[grouped3["network_type"] == "dendritic_shunting"].rename(
        columns={"test_accuracy": "test_sh"}
    )
    additive = grouped3[grouped3["network_type"] == "dendritic_additive"].rename(
        columns={"test_accuracy": "test_ad"}
    )
    merged = shunting.merge(
        additive, on=["dataset", "error_broadcast_mode", "ie_value"], how="inner"
    )
    merged["delta"] = merged["test_sh"] - merged["test_ad"]
    for dataset in sorted(merged["dataset"].unique()):
        subset = merged[merged["dataset"] == dataset]
        rows.append(
            {
                "effect": f"{dataset}: shunting - additive (avg matched)",
                "value": float(subset["delta"].mean()),
            }
        )

    # Claim4: path effect by broadcast
    grouped4 = (
        claim4_df.groupby(["error_broadcast_mode", "use_path_propagation"])[
            ["test_accuracy", "mi_E_I_C"]
        ]
        .mean()
        .reset_index()
    )
    for mode in sorted(grouped4["error_broadcast_mode"].unique()):
        subset = grouped4[grouped4["error_broadcast_mode"] == mode].set_index(
            "use_path_propagation"
        )
        if False in subset.index and True in subset.index:
            rows.append(
                {
                    "effect": f"{mode}: path(true-false) test",
                    "value": float(
                        subset.loc[True, "test_accuracy"] - subset.loc[False, "test_accuracy"]
                    ),
                }
            )
            rows.append(
                {
                    "effect": f"{mode}: path(true-false) MI(E,I;C)",
                    "value": float(
                        subset.loc[True, "mi_E_I_C"] - subset.loc[False, "mi_E_I_C"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-root", type=Path, default=DEFAULT_SWEEP_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--claim2-dir", type=Path, default=None)
    parser.add_argument("--claim3-dir", type=Path, default=None)
    parser.add_argument("--claim4-dir", type=Path, default=None)
    args = parser.parse_args()

    _configure_style()

    claim2_dir = args.claim2_dir or _latest_match(
        args.sweep_root, "sweep_neurips_claim2_decoder_locality_multidataset_robust_*"
    )
    claim3_dir = args.claim3_dir or _latest_match(
        args.sweep_root, "sweep_neurips_claim3_shunting_regime_robust_*"
    )
    claim4_dir = args.claim4_dir or _latest_match(
        args.sweep_root, "sweep_neurips_claim4_source_analysis_robust_*"
    )

    claim2_df = _load_processed(claim2_dir)
    claim3_df = _load_processed(claim3_dir)
    claim4_df = _load_processed(claim4_dir)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    claim2_grouped, fig1_path = _plot_decoder_locality(claim2_df, output_dir)
    claim3_grouped, fig2_path = _plot_shunting_regime(claim3_df, output_dir)
    claim4_grouped, fig3_path = _plot_source_interaction(claim4_df, output_dir)

    claim2_grouped.to_csv(output_dir / "data" / "fig_decoder_locality_data.csv", index=False)
    claim3_grouped.to_csv(output_dir / "data" / "fig_shunting_regime_data.csv", index=False)
    claim4_grouped.to_csv(output_dir / "data" / "fig_source_interaction_data.csv", index=False)

    summary = _build_effect_summary(claim2_df, claim3_df, claim4_df)
    summary.to_csv(output_dir / "data" / "robust_effect_sizes.csv", index=False)

    analysis_dir = args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_md = analysis_dir / "robust_effect_sizes.md"
    with summary_md.open("w", encoding="utf-8") as handle:
        handle.write("# Robust Local-Learning Effect Sizes\n\n")
        handle.write(f"- claim2 dir: `{claim2_dir}`\n")
        handle.write(f"- claim3 dir: `{claim3_dir}`\n")
        handle.write(f"- claim4 dir: `{claim4_dir}`\n\n")
        handle.write("## Effects\n\n")
        for row in summary.itertuples(index=False):
            handle.write(f"- `{row.effect}`: `{row.value:.6f}`\n")
        handle.write("\n## Generated Figures\n\n")
        handle.write(f"- `{fig1_path}`\n")
        handle.write(f"- `{fig2_path}`\n")
        handle.write(f"- `{fig3_path}`\n")

    print(f"Generated figures in: {output_dir}")
    print(f"Wrote effect summary: {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

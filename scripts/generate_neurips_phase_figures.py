#!/usr/bin/env python3
"""Generate phase-program paper figures from summarized sweep outputs."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_DIR = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/analysis/neurips_phase_program"
)
DEFAULT_OUTPUT_DIR = Path(
    "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/figures"
)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _save_dual(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=300, bbox_inches="tight")


def _setup_style() -> None:
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


def _parse_list_string(value: str) -> list[int]:
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    if isinstance(parsed, list):
        out: list[int] = []
        for item in parsed:
            try:
                out.append(int(item))
            except Exception:
                continue
        return out
    return []


def _plot_phase1_capacity(phase1_best: pd.DataFrame, output_dir: Path) -> Path | None:
    if phase1_best.empty:
        return None

    datasets = sorted(phase1_best["dataset"].dropna().unique().tolist())
    cores = [
        "point_mlp",
        "dendritic_mlp",
        "dendritic_additive",
        "dendritic_shunting",
    ]
    color_map = {
        "point_mlp": "#3B4CC0",
        "dendritic_mlp": "#1FA187",
        "dendritic_additive": "#F98E09",
        "dendritic_shunting": "#D7263D",
    }

    width = 0.18
    x_base = np.arange(len(datasets))

    fig, ax = plt.subplots(figsize=(10.2, 4.2))
    for idx, core in enumerate(cores):
        subset = (
            phase1_best[phase1_best["network_type"] == core]
            .set_index("dataset")
            .reindex(datasets)
            .reset_index()
        )
        values = subset["test_accuracy"].fillna(0.0).to_numpy()
        ax.bar(
            x_base + (idx - 1.5) * width,
            values,
            width=width,
            color=color_map[core],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            label=core,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([dataset.replace("_", " ").title() for dataset in datasets])
    ax.set_ylabel("Best Test Accuracy")
    ax.set_title("Phase 1 Capacity Calibration: Best Standard Backprop per Core")
    ax.legend(ncol=2, frameon=False, loc="best")
    ax.set_ylim(bottom=0.0)

    fig.tight_layout()
    out_base = output_dir / "fig_phase1_capacity_best_core"
    _save_dual(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def _plot_claimA_shunting(claimA: pd.DataFrame, output_dir: Path) -> Path | None:
    if claimA.empty:
        return None

    required = {
        "dataset",
        "error_broadcast_mode",
        "ie_value",
        "network_type",
        "test_accuracy_mean",
    }
    if not required.issubset(set(claimA.columns)):
        return None

    pivot = claimA.pivot_table(
        index=["dataset", "error_broadcast_mode", "ie_value"],
        columns="network_type",
        values="test_accuracy_mean",
    ).reset_index()

    if not {
        "dendritic_shunting",
        "dendritic_additive",
    }.issubset(set(pivot.columns)):
        return None

    pivot["delta"] = pivot["dendritic_shunting"] - pivot["dendritic_additive"]

    datasets = sorted(pivot["dataset"].dropna().unique().tolist())
    modes = sorted(pivot["error_broadcast_mode"].dropna().unique().tolist())

    nrows = max(1, len(datasets))
    ncols = max(1, len(modes))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.2 * ncols, 2.3 * nrows),
        squeeze=False,
    )

    all_delta = pivot["delta"].to_numpy()
    vmax = float(np.nanmax(np.abs(all_delta))) if len(all_delta) else 0.1
    vmax = max(vmax, 0.01)

    for row_idx, dataset in enumerate(datasets):
        for col_idx, mode in enumerate(modes):
            axis = axes[row_idx][col_idx]
            subset = pivot[
                (pivot["dataset"] == dataset)
                & (pivot["error_broadcast_mode"] == mode)
            ].sort_values("ie_value")
            if subset.empty:
                axis.set_axis_off()
                continue

            ie_values = subset["ie_value"].to_numpy(dtype=float)
            deltas = subset["delta"].to_numpy(dtype=float)
            heat = axis.imshow(
                deltas.reshape(1, -1),
                aspect="auto",
                cmap="coolwarm",
                vmin=-vmax,
                vmax=vmax,
            )
            axis.set_xticks(np.arange(len(ie_values)))
            axis.set_xticklabels([str(int(v)) if float(v).is_integer() else f"{v:.1f}" for v in ie_values])
            axis.set_yticks([])
            axis.set_title(f"{dataset} | {mode}")
            axis.set_xlabel("IE synapses/branch")

            for idx, value in enumerate(deltas):
                axis.text(
                    idx,
                    0,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    fig.suptitle("Claim A: Shunting Advantage (Shunting - Additive)", y=1.02)
    cbar = fig.colorbar(heat, ax=axes.ravel().tolist(), shrink=0.82)
    cbar.set_label("Delta test accuracy")
    fig.tight_layout()

    out_base = output_dir / "fig_phase3_claimA_shunting_heatmap"
    _save_dual(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def _plot_phase2b_gap_closing(phase2b: pd.DataFrame, output_dir: Path) -> Path | None:
    if phase2b.empty:
        return None

    required = {
        "dataset",
        "error_broadcast_mode",
        "hsic_enabled",
        "hsic_weight",
        "test_accuracy_mean",
        "test_accuracy_std",
    }
    if not required.issubset(set(phase2b.columns)):
        return None

    frame = phase2b.copy()
    frame = frame[frame["test_accuracy_mean"].notna()].copy()
    if frame.empty:
        return None

    # Normalize hsic_enabled to bool-ish and hsic_weight to float when possible.
    frame["hsic_enabled"] = frame["hsic_enabled"].map(
        lambda value: bool(value)
        if isinstance(value, (bool, int, float))
        else str(value).lower() in {"true", "1", "yes"}
    )
    frame["hsic_weight"] = pd.to_numeric(frame["hsic_weight"], errors="coerce")

    datasets = sorted(frame["dataset"].dropna().unique().tolist())
    modes = sorted(frame["error_broadcast_mode"].dropna().unique().tolist())
    if not datasets or not modes:
        return None

    # Prefer plotting enabled rows; if none exist, fall back to all rows.
    enabled = frame[frame["hsic_enabled"] == True]
    plot_df = enabled if not enabled.empty else frame

    weights = (
        plot_df["hsic_weight"]
        .dropna()
        .unique()
        .tolist()
    )
    weights = sorted(float(w) for w in weights)
    if not weights:
        return None

    # Use categorical x positions so we can include a "0.0" baseline.
    x_positions = np.arange(len(weights))

    nrows = 1
    ncols = max(1, len(datasets))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.6 * ncols, 3.0),
        squeeze=False,
    )

    color_map = {
        "scalar": "#3B4CC0",
        "per_soma": "#D7263D",
        "local_mismatch": "#1FA187",
    }

    for col_idx, dataset in enumerate(datasets):
        ax = axes[0][col_idx]
        subset = plot_df[plot_df["dataset"] == dataset].copy()
        if subset.empty:
            ax.set_axis_off()
            continue

        for mode in modes:
            group = subset[subset["error_broadcast_mode"] == mode].copy()
            if group.empty:
                continue

            group = group.set_index("hsic_weight").reindex(weights).reset_index()
            y = group["test_accuracy_mean"].to_numpy(dtype=float)
            yerr = group["test_accuracy_std"].to_numpy(dtype=float)

            ax.errorbar(
                x_positions,
                y,
                yerr=yerr,
                marker="o",
                markersize=4.5,
                linewidth=1.6,
                capsize=2.5,
                color=color_map.get(mode, "#4B5563"),
                label=mode,
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{w:g}" for w in weights])
        ax.set_xlabel("HSIC weight")
        ax.set_ylabel("Test accuracy (mean ± std)")
        ax.set_title(dataset.replace("_", " ").title())
        ax.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(3, len(labels)), frameon=False)
    fig.suptitle("Phase 2b Gap Closing: HSIC Weight x Error Broadcast Mode", y=1.10)

    fig.tight_layout()
    out_base = output_dir / "fig_phase2b_gap_closing_hsic_weight"
    _save_dual(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def _plot_claimB_morphology(claimB: pd.DataFrame, output_dir: Path) -> Path | None:
    if claimB.empty:
        return None

    needed = {
        "dataset",
        "branch_factors",
        "use_path_propagation",
        "morphology_modulator_mode",
        "test_accuracy_mean",
        "test_accuracy_std",
    }
    if not needed.issubset(set(claimB.columns)):
        return None

    frame = claimB.copy()
    frame["branch_first"] = frame["branch_factors"].map(
        lambda text: (_parse_list_string(text) or [0])[0]
    )

    datasets = sorted(frame["dataset"].dropna().unique().tolist())
    if not datasets:
        return None

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.8), squeeze=False)

    line_styles = {
        (False, "none"): ("#6B7280", "--", "o"),
        (True, "none"): ("#111827", "-", "o"),
        (False, "depth"): ("#A855F7", "--", "s"),
        (True, "depth"): ("#7C3AED", "-", "s"),
        (False, "centrality"): ("#F59E0B", "--", "D"),
        (True, "centrality"): ("#D97706", "-", "D"),
    }

    for idx, dataset in enumerate(datasets):
        axis = axes[0][idx]
        subset = frame[frame["dataset"] == dataset]
        for (path_flag, mode), (color, linestyle, marker) in line_styles.items():
            line = subset[
                (subset["use_path_propagation"] == path_flag)
                & (subset["morphology_modulator_mode"] == mode)
            ].sort_values("branch_first")
            if line.empty:
                continue
            axis.errorbar(
                line["branch_first"],
                line["test_accuracy_mean"],
                yerr=line["test_accuracy_std"].fillna(0.0),
                color=color,
                linestyle=linestyle,
                marker=marker,
                linewidth=1.7,
                markersize=5,
                capsize=3,
                alpha=0.95,
                label=f"path={str(path_flag).lower()}, mode={mode}",
            )

        axis.set_title(dataset.replace("_", " ").title())
        axis.set_xlabel("Branch factor (first level)")
        axis.set_ylabel("Test accuracy")
        axis.grid(axis="y", alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Claim B: Morphology Scaling and Path-Modulation Effects", y=1.03)
    fig.tight_layout()
    out_base = output_dir / "fig_phase3_claimB_morphology_scaling"
    _save_dual(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def _plot_claimC_error_shaping(claimC: pd.DataFrame, output_dir: Path) -> Path | None:
    if claimC.empty:
        return None

    needed = {
        "dataset",
        "error_broadcast_mode",
        "decoder_update_mode",
        "use_path_propagation",
        "test_accuracy_mean",
    }
    if not needed.issubset(set(claimC.columns)):
        return None

    # Keep best path setting per condition for a cleaner panel
    frame = (
        claimC.sort_values("test_accuracy_mean", ascending=False)
        .groupby(["dataset", "error_broadcast_mode", "decoder_update_mode"], as_index=False)
        .first()
    )

    datasets = sorted(frame["dataset"].dropna().unique().tolist())
    modes = ["scalar", "per_soma", "local_mismatch"]
    decoders = ["backprop", "local"]
    decoder_colors = {"backprop": "#1D4E89", "local": "#0B6E4F"}

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.8), squeeze=False)

    for idx, dataset in enumerate(datasets):
        axis = axes[0][idx]
        subset = frame[frame["dataset"] == dataset]
        x = np.arange(len(modes))
        width = 0.35

        for offset, decoder in [(-width / 2, "backprop"), (width / 2, "local")]:
            y_values = []
            for mode in modes:
                row = subset[
                    (subset["error_broadcast_mode"] == mode)
                    & (subset["decoder_update_mode"] == decoder)
                ]
                y_values.append(float(row["test_accuracy_mean"].iloc[0]) if not row.empty else 0.0)

            axis.bar(
                x + offset,
                y_values,
                width=width,
                color=decoder_colors[decoder],
                edgecolor="black",
                linewidth=0.6,
                alpha=0.9,
                label=decoder,
            )

        axis.set_xticks(x)
        axis.set_xticklabels([mode.replace("_", "\n") for mode in modes])
        axis.set_title(dataset.replace("_", " ").title())
        axis.set_ylabel("Test accuracy")
        axis.set_ylim(bottom=0.0)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Claim C: Local Error-Shaping Comparison", y=1.03)
    fig.tight_layout()

    out_base = output_dir / "fig_phase3_claimC_error_shaping"
    _save_dual(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def _plot_learning_curves(combined: pd.DataFrame, curves: pd.DataFrame, output_dir: Path) -> Path | None:
    if combined.empty or curves.empty:
        return None

    candidate = combined[combined["strategy"] == "local_ca"].copy()
    if candidate.empty:
        return None

    best = (
        candidate.sort_values("valid_accuracy", ascending=False)
        .groupby("dataset", as_index=False)
        .first()[["dataset", "config_id", "sweep_dir", "valid_accuracy", "test_accuracy"]]
    )

    curve_rows = []
    for row in best.itertuples(index=False):
        subset = curves[
            (curves["config_id"] == row.config_id)
            & (curves["sweep_dir"] == row.sweep_dir)
        ].copy()
        if subset.empty:
            continue
        subset["dataset"] = row.dataset
        curve_rows.append(subset)

    if not curve_rows:
        return None

    frame = pd.concat(curve_rows, ignore_index=True)
    frame = frame.dropna(subset=["epoch", "valid_accuracy"])
    if frame.empty:
        return None

    datasets = sorted(frame["dataset"].dropna().unique().tolist())
    fig, ax = plt.subplots(figsize=(8.8, 4.2))

    palette = ["#0B6E4F", "#1D4E89", "#D97706", "#A62639", "#7C3AED", "#2563EB"]
    for idx, dataset in enumerate(datasets):
        subset = frame[frame["dataset"] == dataset].sort_values("epoch")
        ax.plot(
            subset["epoch"],
            subset["valid_accuracy"],
            linewidth=2.0,
            color=palette[idx % len(palette)],
            label=dataset,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Best Local-CA Learning Curves by Dataset")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3)

    fig.tight_layout()
    out_base = output_dir / "fig_phase_learning_curves"
    _save_dual(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def _plot_info_panel(info_panel: pd.DataFrame, output_dir: Path) -> Path | None:
    if info_panel.empty:
        return None

    if not {"test_accuracy_mean", "network_type"}.issubset(set(info_panel.columns)):
        return None

    x_metric = "mi_E_I_C_mean" if "mi_E_I_C_mean" in info_panel.columns else None
    if x_metric is None:
        return None

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    markers = {False: "o", True: "D", "False": "o", "True": "D"}
    colors = {
        "dendritic_additive": "#1D4E89",
        "dendritic_shunting": "#0B6E4F",
    }

    for row in info_panel.itertuples(index=False):
        marker = markers.get(getattr(row, "use_path_propagation"), "o")
        network_type = getattr(row, "network_type", "")
        color = colors.get(network_type, "#4B5563")
        ax.scatter(
            getattr(row, x_metric),
            getattr(row, "test_accuracy_mean"),
            marker=marker,
            s=72,
            color=color,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("MI(E,I;C) mean")
    ax.set_ylabel("Test accuracy mean")
    ax.set_title("Information Panel: Accuracy vs MI(E,I;C)")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    out_base = output_dir / "fig_phase_information_panel"
    _save_dual(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    _setup_style()

    analysis_dir = args.analysis_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_best = _safe_read_csv(analysis_dir / "phase1_best_standard.csv")
    phase2b = _safe_read_csv(analysis_dir / "phase2b_gap_closing.csv")
    claimA = _safe_read_csv(analysis_dir / "claimA_shunting_regime.csv")
    claimB = _safe_read_csv(analysis_dir / "claimB_morphology_scaling.csv")
    claimC = _safe_read_csv(analysis_dir / "claimC_error_shaping.csv")
    combined = _safe_read_csv(analysis_dir / "combined_results.csv")
    curves = _safe_read_csv(analysis_dir / "learning_curves.csv")
    info_panel = _safe_read_csv(analysis_dir / "info_panel_metrics.csv")

    generated: list[Path] = []

    for fig_path in [
        _plot_phase1_capacity(phase1_best, output_dir),
        _plot_phase2b_gap_closing(phase2b, output_dir),
        _plot_claimA_shunting(claimA, output_dir),
        _plot_claimB_morphology(claimB, output_dir),
        _plot_claimC_error_shaping(claimC, output_dir),
        _plot_learning_curves(combined, curves, output_dir),
        _plot_info_panel(info_panel, output_dir),
    ]:
        if fig_path is not None:
            generated.append(fig_path)

    summary_path = output_dir / "phase_figure_manifest.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        for path in generated:
            handle.write(f"{path}\n")

    print(f"Generated {len(generated)} phase figures into: {output_dir}")
    print(f"Manifest: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

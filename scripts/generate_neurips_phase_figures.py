#!/usr/bin/env python3
"""Generate publication-ready phase figures from summarized sweep outputs."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_DIR = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/analysis/publication_bundle_faircheck_20260223"
)
DEFAULT_OUTPUT_DIR = Path(
    "/n/holylabs/kempner_dev/Users/hsafaai/Code/dendritic-modeling/drafts/dendritic-local-learning/figures"
)

# Consistent paper palette:
# - shunting: green
# - additive: blue
# - controls: neutral grays
COLOR_SHUNT = "#18864B"
COLOR_ADD = "#2D5DA8"
COLOR_POINT = "#6C757D"
COLOR_DEND_MLP = "#A0A4AA"
COLOR_SCALAR = "#495057"
COLOR_PER_SOMA = "#0B6E4F"
COLOR_LOCAL_MISMATCH = "#A03A2C"
COLOR_DEPTH = "#18864B"
COLOR_CENTRALITY = "#D08300"
COLOR_NONE = "#4C4C4C"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _save_dual(fig: plt.Figure, path_base: Path) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".pdf"), dpi=350, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".png"), dpi=350, bbox_inches="tight")


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.9,
            "grid.color": "#D8DCE3",
            "grid.linestyle": "-",
            "grid.linewidth": 0.8,
            "legend.fontsize": 9,
            "figure.titlesize": 12,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _order_datasets(values: list[str]) -> list[str]:
    priority = [
        "mnist",
        "context_gating",
        "noise_resilience",
        "info_shunting",
        "hierarchical_processing",
        "orthonet",
        "cifar10",
    ]
    rank = {name: i for i, name in enumerate(priority)}
    return sorted(values, key=lambda value: (rank.get(value, 10_000), value))


def _humanize_dataset(name: str) -> str:
    mapping = {
        "mnist": "MNIST",
        "cifar10": "CIFAR-10",
        "context_gating": "Context Gating",
        "noise_resilience": "Noise Resilience",
        "info_shunting": "Info Shunting",
        "hierarchical_processing": "Hierarchical Processing",
        "orthonet": "Orthonet",
    }
    return mapping.get(name, str(name).replace("_", " ").title())


def _humanize_core(name: str) -> str:
    mapping = {
        "point_mlp": "MLP (point)",
        "dendritic_mlp": "Dendritic MLP",
        "dendritic_additive": "Additive",
        "dendritic_shunting": "Shunting",
    }
    return mapping.get(name, name)


def _humanize_mode(name: str) -> str:
    mapping = {
        "scalar": "Scalar",
        "per_soma": "Per-soma",
        "local_mismatch": "Local mismatch",
        "local": "Local",
        "backprop": "Backprop",
    }
    return mapping.get(name, str(name).replace("_", " ").title())


def _parse_list_string(value: Any) -> list[int]:
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    out: list[int] = []
    for item in parsed:
        try:
            out.append(int(item))
        except Exception:
            continue
    return out


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _plot_phase1_capacity(phase1_best: pd.DataFrame, output_dir: Path) -> Path | None:
    if phase1_best.empty:
        return None
    required = {"dataset", "network_type", "test_accuracy"}
    if not required.issubset(set(phase1_best.columns)):
        return None

    datasets = _order_datasets(phase1_best["dataset"].dropna().astype(str).unique().tolist())
    cores = ["point_mlp", "dendritic_mlp", "dendritic_additive", "dendritic_shunting"]
    colors = {
        "point_mlp": COLOR_POINT,
        "dendritic_mlp": COLOR_DEND_MLP,
        "dendritic_additive": COLOR_ADD,
        "dendritic_shunting": COLOR_SHUNT,
    }

    width = 0.19
    x_base = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(10.4, 4.6))

    for idx, core in enumerate(cores):
        subset = (
            phase1_best[phase1_best["network_type"] == core]
            .set_index("dataset")
            .reindex(datasets)
            .reset_index()
        )
        values = pd.to_numeric(subset["test_accuracy"], errors="coerce").to_numpy(dtype=float)
        ax.bar(
            x_base + (idx - 1.5) * width,
            values,
            width=width,
            color=colors[core],
            edgecolor="#1E1E1E",
            linewidth=0.7,
            alpha=0.92,
            label=_humanize_core(core),
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([_humanize_dataset(dataset) for dataset in datasets], rotation=15, ha="right")
    ax.set_ylabel("Best Test Accuracy")
    ax.set_xlabel("Dataset")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Backprop Capacity Ceiling by Architecture", pad=10)
    ax.grid(axis="y", alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))

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
    if not {"dendritic_shunting", "dendritic_additive"}.issubset(set(pivot.columns)):
        return None
    pivot["delta"] = pivot["dendritic_shunting"] - pivot["dendritic_additive"]

    datasets = _order_datasets(pivot["dataset"].dropna().astype(str).unique().tolist())
    modes = sorted(pivot["error_broadcast_mode"].dropna().astype(str).unique().tolist())
    if not datasets or not modes:
        return None

    fig, axes = plt.subplots(
        len(datasets),
        len(modes),
        figsize=(3.3 * len(modes) + 1.0, 1.9 * len(datasets) + 1.0),
        squeeze=False,
    )
    vmax = float(np.nanmax(np.abs(pivot["delta"].to_numpy(dtype=float))))
    vmax = max(vmax, 0.01)
    heat = None

    for row_idx, dataset in enumerate(datasets):
        for col_idx, mode in enumerate(modes):
            axis = axes[row_idx][col_idx]
            subset = pivot[
                (pivot["dataset"] == dataset)
                & (pivot["error_broadcast_mode"] == mode)
            ].copy()
            subset["ie_value"] = pd.to_numeric(subset["ie_value"], errors="coerce")
            subset = subset.dropna(subset=["ie_value"]).sort_values("ie_value")
            if subset.empty:
                axis.axis("off")
                continue

            ie_values = subset["ie_value"].to_numpy(dtype=float)
            deltas = subset["delta"].to_numpy(dtype=float)
            heat = axis.imshow(
                deltas.reshape(1, -1),
                aspect="auto",
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            axis.set_xticks(np.arange(len(ie_values)))
            axis.set_xticklabels(
                [str(int(v)) if float(v).is_integer() else f"{v:.1f}" for v in ie_values],
                fontsize=9,
            )
            axis.set_yticks([])
            axis.set_xlabel("I synapses / branch", fontsize=9)
            if col_idx == 0:
                axis.set_ylabel(_humanize_dataset(dataset), fontsize=10)
            if row_idx == 0:
                axis.set_title(_humanize_mode(mode), fontsize=10)

            for idx, value in enumerate(deltas):
                axis.text(
                    idx,
                    0,
                    f"{value:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#111111",
                )

    if heat is not None:
        cbar = fig.colorbar(
            heat,
            ax=axes.ravel().tolist(),
            fraction=0.022,
            pad=0.02,
        )
        cbar.set_label("Shunting - Additive (test acc)")

    fig.suptitle("Claim A: Shunting Advantage Across Inhibitory Regimes", y=1.02)
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

    frame["hsic_weight"] = pd.to_numeric(frame["hsic_weight"], errors="coerce")
    frame["hsic_weight"] = frame["hsic_weight"].fillna(0.0)
    datasets = _order_datasets(frame["dataset"].dropna().astype(str).unique().tolist())
    modes = sorted(frame["error_broadcast_mode"].dropna().astype(str).unique().tolist())
    if not datasets or not modes:
        return None

    color_map = {
        "scalar": COLOR_SCALAR,
        "per_soma": COLOR_PER_SOMA,
        "local_mismatch": COLOR_LOCAL_MISMATCH,
    }
    fig, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(4.2 * len(datasets), 3.4),
        squeeze=False,
    )

    for col_idx, dataset in enumerate(datasets):
        ax = axes[0][col_idx]
        subset = frame[frame["dataset"] == dataset].copy()
        if subset.empty:
            ax.axis("off")
            continue

        weights = sorted(subset["hsic_weight"].dropna().unique().tolist())
        x_positions = np.arange(len(weights))

        for mode in modes:
            group = subset[subset["error_broadcast_mode"] == mode].copy()
            if group.empty:
                continue
            # Multiple rows can share the same HSIC weight (different cores/aux
            # toggles). Aggregate first to avoid duplicate-index reindex errors.
            group = (
                group.groupby("hsic_weight", as_index=False)
                .agg(
                    {
                        "test_accuracy_mean": "mean",
                        "test_accuracy_std": "mean",
                    }
                )
            )
            group = group.set_index("hsic_weight").reindex(weights).reset_index()
            y = pd.to_numeric(group["test_accuracy_mean"], errors="coerce").to_numpy(dtype=float)
            yerr = pd.to_numeric(group["test_accuracy_std"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(
                x_positions,
                y,
                yerr=yerr,
                marker="o",
                markersize=4.7,
                linewidth=1.9,
                capsize=2.8,
                color=color_map.get(mode, "#444444"),
                label=_humanize_mode(mode),
            )

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"{w:g}" for w in weights])
        ax.set_xlabel("HSIC weight")
        ax.set_ylabel("Test accuracy (mean +- std)")
        ax.set_title(_humanize_dataset(dataset))
        ax.grid(axis="y", alpha=0.65)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(3, len(labels)),
            frameon=False,
            bbox_to_anchor=(0.5, 1.06),
        )
    fig.suptitle("Phase-2b: Broadcast Mode x HSIC", y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
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
    frame["branch_first"] = frame["branch_factors"].map(lambda text: (_parse_list_string(text) or [0])[0])
    frame["use_path_propagation"] = frame["use_path_propagation"].map(_to_bool)
    frame = frame.dropna(subset=["test_accuracy_mean"])
    if frame.empty:
        return None

    dataset = _humanize_dataset(str(frame["dataset"].dropna().iloc[0]))
    modes = ["none", "depth", "centrality"]
    mode_colors = {"none": COLOR_NONE, "depth": COLOR_DEPTH, "centrality": COLOR_CENTRALITY}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for idx, path_flag in enumerate([False, True]):
        axis = axes[idx]
        sub_path = frame[frame["use_path_propagation"] == path_flag]
        for mode in modes:
            sub = sub_path[sub_path["morphology_modulator_mode"] == mode].copy()
            sub = sub.sort_values("branch_first")
            if sub.empty:
                continue
            axis.errorbar(
                sub["branch_first"],
                sub["test_accuracy_mean"],
                yerr=sub["test_accuracy_std"].fillna(0.0),
                color=mode_colors[mode],
                marker="o",
                linewidth=2.0,
                markersize=5.0,
                capsize=3.0,
                label=_humanize_mode(mode),
            )
        axis.set_title(f"Path propagation: {'On' if path_flag else 'Off'}")
        axis.set_xlabel("Branch factor (level 1)")
        axis.grid(axis="y", alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0].set_ylabel("Test accuracy (mean +- std)")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            ncol=3,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.06),
        )
    fig.suptitle(f"Claim B: Morphology Modulator Effects ({dataset})", y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

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
        "test_accuracy_mean",
    }
    if not needed.issubset(set(claimC.columns)):
        return None

    frame = claimC.copy()
    frame = frame.dropna(subset=["test_accuracy_mean"])
    if frame.empty:
        return None

    datasets = _order_datasets(frame["dataset"].dropna().astype(str).unique().tolist())
    modes = ["scalar", "per_soma", "local_mismatch"]
    decoders = ["backprop", "local"]
    decoder_colors = {"backprop": "#444444", "local": COLOR_PER_SOMA}

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.1 * len(datasets), 3.7), squeeze=False)
    for idx, dataset in enumerate(datasets):
        axis = axes[0][idx]
        subset = frame[frame["dataset"] == dataset]
        x = np.arange(len(modes))
        width = 0.36

        for offset, decoder in [(-width / 2, "backprop"), (width / 2, "local")]:
            y_values: list[float] = []
            y_err: list[float] = []
            for mode in modes:
                row = subset[
                    (subset["error_broadcast_mode"] == mode)
                    & (subset["decoder_update_mode"] == decoder)
                ]
                if row.empty:
                    y_values.append(np.nan)
                    y_err.append(0.0)
                else:
                    y_values.append(float(row["test_accuracy_mean"].iloc[0]))
                    y_err.append(float(row["test_accuracy_std"].iloc[0]) if "test_accuracy_std" in row.columns else 0.0)

            axis.bar(
                x + offset,
                y_values,
                width=width,
                color=decoder_colors[decoder],
                edgecolor="#111111",
                linewidth=0.7,
                alpha=0.9,
                label=_humanize_mode(decoder),
                yerr=y_err,
                capsize=2.6,
            )

        axis.set_xticks(x)
        axis.set_xticklabels([_humanize_mode(mode).replace("-", "\n") for mode in modes], fontsize=9)
        axis.set_title(_humanize_dataset(dataset))
        axis.set_ylim(bottom=0.0)
        axis.grid(axis="y", alpha=0.65)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        if idx == 0:
            axis.set_ylabel("Test accuracy (mean +- std)")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Claim C: Error Broadcast x Decoder Update", y=1.12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_base = output_dir / "fig_phase3_claimC_error_shaping"
    _save_dual(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".pdf")


def _plot_info_panel(info_panel: pd.DataFrame, output_dir: Path) -> Path | None:
    if info_panel.empty:
        return None
    required = {"test_accuracy_mean", "network_type"}
    if not required.issubset(set(info_panel.columns)):
        return None
    x_metric = "mi_E_I_C_mean" if "mi_E_I_C_mean" in info_panel.columns else None
    if x_metric is None:
        return None

    frame = info_panel.copy().dropna(subset=[x_metric, "test_accuracy_mean"])
    if frame.empty:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    marker_map = {False: "o", True: "D"}
    color_map = {"dendritic_additive": COLOR_ADD, "dendritic_shunting": COLOR_SHUNT}

    for row in frame.itertuples(index=False):
        path_flag = _to_bool(getattr(row, "use_path_propagation"))
        marker = marker_map[path_flag]
        network_type = str(getattr(row, "network_type", ""))
        color = color_map.get(network_type, "#555555")
        ax.scatter(
            getattr(row, x_metric),
            getattr(row, "test_accuracy_mean"),
            marker=marker,
            s=86,
            color=color,
            alpha=0.9,
            edgecolor="#111111",
            linewidth=0.55,
        )

    ax.set_xlabel("MI(E, I; C) (mean)")
    ax.set_ylabel("Test accuracy (mean)")
    ax.set_title("Information Panel: Accuracy vs Dendritic MI")
    ax.grid(alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    import matplotlib.lines as mlines

    legend_items = [
        mlines.Line2D([], [], color=COLOR_ADD, marker="o", linestyle="None", markersize=8, label="Additive"),
        mlines.Line2D([], [], color=COLOR_SHUNT, marker="o", linestyle="None", markersize=8, label="Shunting"),
        mlines.Line2D([], [], color="#666666", marker="o", linestyle="None", markersize=8, label="Path off"),
        mlines.Line2D([], [], color="#666666", marker="D", linestyle="None", markersize=8, label="Path on"),
    ]
    ax.legend(handles=legend_items, loc="best", frameon=False)

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
    info_panel = _safe_read_csv(analysis_dir / "info_panel_metrics.csv")

    generated: list[Path] = []
    for fig_path in [
        _plot_phase1_capacity(phase1_best, output_dir),
        _plot_phase2b_gap_closing(phase2b, output_dir),
        _plot_claimA_shunting(claimA, output_dir),
        _plot_claimB_morphology(claimB, output_dir),
        _plot_claimC_error_shaping(claimC, output_dir),
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

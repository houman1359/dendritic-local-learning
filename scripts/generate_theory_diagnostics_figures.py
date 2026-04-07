#!/usr/bin/env python3
"""Generate publication-ready mechanistic figures from theory diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = DRAFT_DIR / "figures"
DATA_DIR = DRAFT_DIR / "data"
ANALYSIS_DIR = DRAFT_DIR / "analysis"
SUMMARY_DIR = ANALYSIS_DIR / "theory_diag_gradient_fidelity_vs_ie_summary"
SUMMARY_CSV = SUMMARY_DIR / "theory_diag_by_condition.csv"
MERGED_CSV = SUMMARY_DIR / "theory_diag_merged_runs.csv"
ORACLE_SUMMARY_CSV = (
    ANALYSIS_DIR / "path_transport_upper_bound" / "path_transport_upper_bound_summary.csv"
)
LOW_BW_CSV = DATA_DIR / "low_bandwidth_results.csv"

COLOR_SHUNTING = "#18864B"
COLOR_ADDITIVE = "#2D5DA8"
COLOR_TRANSPORT = "#9A4D1E"
DOUBLE_COL_W = 11.0
DPI = 300


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.titlepad": 7,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )


def _panel(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.12) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=DPI)
    print(f"Saved {name}.{{pdf,png}}")


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _plot_path_gain_dispersion(ax: plt.Axes, summary: pd.DataFrame) -> None:
    _panel(ax, "A")
    mnist = summary[summary["dataset"] == "mnist"].copy()
    for network_type, color, label in [
        ("dendritic_shunting", COLOR_SHUNTING, "Shunting"),
        ("dendritic_additive", COLOR_ADDITIVE, "Additive"),
    ]:
        sub = mnist[mnist["network_type"] == network_type].sort_values("ie_value")
        x = pd.to_numeric(sub["ie_value"], errors="coerce").to_numpy(dtype=float)
        y = sub["path_gain_cv_mean_mean"].to_numpy(dtype=float)
        err = sub["path_gain_cv_mean_std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", markersize=3.5, lw=1.4, color=color, label=label)
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel("$N_I$ (inhibitory synapses / branch)")
    ax.set_ylabel("Path-gain CV")
    ax.set_title("Shunting narrows the conductance-stage path-gain distribution")
    ax.set_xticks([0, 5, 10, 20, 40])
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left")


def _plot_compartment_error_fidelity(ax: plt.Axes, summary: pd.DataFrame) -> None:
    _panel(ax, "B")
    noise = summary[summary["dataset"] == "noise_resilience"].copy()
    style_map = {
        ("dendritic_shunting", "per_soma"): (COLOR_SHUNTING, "-", "Shunting, per-soma"),
        ("dendritic_shunting", "path_transport"): (COLOR_SHUNTING, "--", "Shunting, transported"),
        ("dendritic_additive", "per_soma"): (COLOR_ADDITIVE, "-", "Additive, per-soma"),
        ("dendritic_additive", "path_transport"): (COLOR_ADDITIVE, "--", "Additive, transported"),
    }
    series = [
        ("dendritic_shunting", "per_soma_weighted_cosine_mean"),
        ("dendritic_shunting", "path_transport_weighted_cosine_mean"),
        ("dendritic_additive", "per_soma_weighted_cosine_mean"),
        ("dendritic_additive", "path_transport_weighted_cosine_mean"),
    ]
    std_map = {
        "per_soma_weighted_cosine_mean": "per_soma_weighted_cosine_std",
        "path_transport_weighted_cosine_mean": "path_transport_weighted_cosine_std",
    }
    for network_type, metric in series:
        sub = noise[noise["network_type"] == network_type].sort_values("ie_value")
        x = pd.to_numeric(sub["ie_value"], errors="coerce").to_numpy(dtype=float)
        y = sub[metric].to_numpy(dtype=float)
        err = sub[std_map[metric]].fillna(0.0).to_numpy(dtype=float)
        label_key = (
            network_type,
            "path_transport" if "path_transport" in metric else "per_soma",
        )
        color, linestyle, label = style_map[label_key]
        ax.plot(
            x,
            y,
            marker="o" if linestyle == "-" else "s",
            markersize=3.2,
            lw=1.3,
            linestyle=linestyle,
            color=color,
            label=label,
        )
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.10, linewidth=0)

    ax.set_xlabel("$N_I$ (inhibitory synapses / branch)")
    ax.set_ylabel(r"Cosine$(e_n,\partial L/\partial V_n)$")
    ax.set_title("Per-soma broadcast tracks compartment error")
    ax.set_xticks([0, 5, 10, 20, 40])
    ax.set_ylim(-0.35, 1.05)
    ax.legend(loc="upper left", ncol=1, handlelength=1.2, handletextpad=0.4,
              columnspacing=0.8)


def _plot_low_bandwidth(ax: plt.Axes, low_bw: pd.DataFrame) -> None:
    _panel(ax, "D")

    def bw_label(row: pd.Series) -> str:
        bw = row["broadcast_bandwidth"]
        bits = row.get("broadcast_bits", 8)
        if bw == "full":
            return "Full"
        if bw == "quantized":
            return f"Q{int(bits)}b"
        if bw == "sign_only":
            return "Sign"
        if bw == "sparse_topk":
            return "Top-30%"
        return str(bw)

    def bw_bits_equiv(row: pd.Series) -> float:
        bw = row["broadcast_bandwidth"]
        bits = row.get("broadcast_bits", 8)
        if bw == "full":
            return 32
        if bw == "quantized":
            return float(bits)
        if bw == "sign_only":
            return 1.0
        if bw == "sparse_topk":
            return 0.3
        return 16.0

    df = low_bw.copy()
    df["bw_label"] = df.apply(bw_label, axis=1)
    df["bw_bits"] = df.apply(bw_bits_equiv, axis=1)

    quant_modes = df[df["broadcast_bandwidth"].isin(["full", "quantized", "sign_only"])]
    grp = (
        quant_modes.groupby(["bw_label", "bw_bits"])
        .agg(mean=("test_accuracy", "mean"), std=("test_accuracy", "std"))
        .reset_index()
        .sort_values("bw_bits")
    )
    x = grp["bw_bits"].to_numpy(dtype=float)
    y = 100.0 * grp["mean"].to_numpy(dtype=float)
    err = 100.0 * grp["std"].fillna(0.0).to_numpy(dtype=float)
    ax.errorbar(
        x,
        y,
        yerr=err,
        marker="o",
        markersize=4,
        lw=1.4,
        capsize=2,
        color=COLOR_SHUNTING,
        capthick=0.5,
        zorder=5,
    )
    for _, row in grp.iterrows():
        y_pt = 100.0 * float(row["mean"])
        offset = 2.5 if row["bw_bits"] < 16 else -2.5
        va = "bottom" if row["bw_bits"] < 16 else "top"
        ax.annotate(
            row["bw_label"],
            xy=(row["bw_bits"], y_pt),
            xytext=(0, offset),
            textcoords="offset points",
            fontsize=7,
            ha="center",
            va=va,
            color=COLOR_SHUNTING,
        )

    sparse = df[df["broadcast_bandwidth"] == "sparse_topk"]
    if len(sparse):
        sp_mean = 100.0 * sparse["test_accuracy"].mean()
        sp_std = 100.0 * sparse["test_accuracy"].std()
        ax.errorbar(
            [0.5],
            [sp_mean],
            yerr=[sp_std],
            marker="^",
            markersize=5,
            color=COLOR_TRANSPORT,
            capsize=2,
            capthick=0.5,
            zorder=5,
        )
        ax.annotate(
            "Top-30%",
            xy=(0.5, sp_mean),
            xytext=(12, -6),
            textcoords="offset points",
            fontsize=7,
            color=COLOR_TRANSPORT,
        )

    ax.set_xlabel("Effective bits per neuron")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Coarse broadcast remains useful when sensitivities are stable")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([1, 2, 4, 8, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "32"])
    ax.set_ylim(25, 72)


def _plot_oracle_learning(
    ax: plt.Axes,
    summary: pd.DataFrame,
    oracle_summary: pd.DataFrame,
) -> None:
    _panel(ax, "C")
    baseline = summary[summary["dataset"] == "noise_resilience"].copy()
    oracle = oracle_summary[oracle_summary["dataset"] == "noise_resilience"].copy()
    for network_type, color, label in [
        ("dendritic_shunting", COLOR_SHUNTING, "Shunting"),
        ("dendritic_additive", COLOR_ADDITIVE, "Additive"),
    ]:
        sub = baseline[baseline["network_type"] == network_type].sort_values("ie_value")
        x = pd.to_numeric(sub["ie_value"], errors="coerce").to_numpy(dtype=float)
        y = 100.0 * sub["test_accuracy_mean"].to_numpy(dtype=float)
        err = 100.0 * sub["test_accuracy_std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=3.4,
            lw=1.3,
            color=color,
            linestyle="-",
            label=f"{label}, per-soma",
        )
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.10, linewidth=0)

        sub_oracle = oracle[oracle["network_type"] == network_type].sort_values("ie_value")
        x2 = pd.to_numeric(sub_oracle["ie_value"], errors="coerce").to_numpy(dtype=float)
        y2 = 100.0 * sub_oracle["test_accuracy_mean"].to_numpy(dtype=float)
        err2 = 100.0 * sub_oracle["test_accuracy_std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            x2,
            y2,
            marker="s",
            markersize=3.0,
            lw=1.3,
            color=color,
            linestyle="--",
            label=f"{label}, transported",
        )
        ax.fill_between(x2, y2 - err2, y2 + err2, color=color, alpha=0.08, linewidth=0)

    ax.set_xlabel("$N_I$ (inhibitory synapses / branch)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Better transport closes the local-learning gap")
    ax.set_xticks([0, 5, 10, 20, 40])
    ax.set_ylim(20, 101)
    ax.legend(loc="upper left", ncol=1, handlelength=1.2, handletextpad=0.4,
              columnspacing=0.8)


def build_figure(
    summary_csv: Path = SUMMARY_CSV,
    low_bw_csv: Path = LOW_BW_CSV,
    oracle_summary_csv: Path = ORACLE_SUMMARY_CSV,
) -> plt.Figure:
    _setup_style()
    summary = _safe_csv(summary_csv)
    low_bw = _safe_csv(low_bw_csv)
    oracle_summary = _safe_csv(oracle_summary_csv)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(DOUBLE_COL_W, 7.0),
        gridspec_kw={"wspace": 0.38, "hspace": 0.52},
    )

    _plot_path_gain_dispersion(axes[0, 0], summary)
    _plot_compartment_error_fidelity(axes[0, 1], summary)
    _plot_oracle_learning(axes[1, 0], summary, oracle_summary)
    _plot_low_bandwidth(axes[1, 1], low_bw)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    parser.add_argument("--low-bandwidth-csv", type=Path, default=LOW_BW_CSV)
    parser.add_argument("--oracle-summary-csv", type=Path, default=ORACLE_SUMMARY_CSV)
    parser.add_argument("--name", type=str, default="fig5_mechanistic_evidence")
    args = parser.parse_args()

    fig = build_figure(
        summary_csv=args.summary_csv,
        low_bw_csv=args.low_bandwidth_csv,
        oracle_summary_csv=args.oracle_summary_csv,
    )
    _save(fig, args.name)
    plt.close(fig)


if __name__ == "__main__":
    main()

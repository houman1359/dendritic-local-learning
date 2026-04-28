#!/usr/bin/env python3
"""Generate publication-ready mechanistic figures from theory diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from neurips_style import apply_neurips_style, COLORS, panel_label, style_axis


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = DRAFT_DIR / "figures"
DATA_DIR = DRAFT_DIR / "data"
ANALYSIS_DIR = DRAFT_DIR / "analysis"
SUMMARY_DIR = ANALYSIS_DIR / "theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix_summary"
SUMMARY_CSV = SUMMARY_DIR / "theory_diag_by_condition.csv"
MERGED_CSV = SUMMARY_DIR / "theory_diag_merged_runs.csv"
ORACLE_SUMMARY_CSV = (
    ANALYSIS_DIR
    / "path_transport_upper_bound_nonnegativeinput_fix_5seed"
    / "path_transport_upper_bound_summary.csv"
)
ERROR_RANK_SUMMARY_CSV = (
    ANALYSIS_DIR / "error_rank_selected_20260427" / "error_rank_summary.csv"
)
LOW_BW_CSV = DATA_DIR / "low_bandwidth_results.csv"
CIFAR10_BP_SUMMARY_CSV = (
    ANALYSIS_DIR / "cifar10_compactei_depth4" / "cifar10_compactei_depth4_grouped_summary.csv"
)
CIFAR10_LOCALCA_SUMMARY_CSV = (
    ANALYSIS_DIR
    / "cifar10_compactei_depth4_decoderfix_mechanism_5seed"
    / "cifar10_compactei_depth4_decoderfix_mechanism_summary.csv"
)

COLOR_SHUNTING = COLORS["shunting"]
COLOR_ADDITIVE = COLORS["additive"]
COLOR_TRANSPORT = COLORS["oracle"]
COLOR_LOW_BW = COLORS["local"]
DOUBLE_COL_W = 7.0
DPI = 300


def _setup_style() -> None:
    apply_neurips_style()
    plt.rcParams.update(
        {
            "font.size": 10.2,
            "axes.labelsize": 10.6,
            "axes.titlesize": 10.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.4,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "#D0D5DD",
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


def _panel(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.20) -> None:
    panel_label(ax, label, x=x, y=y)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=DPI)
    print(f"Saved {name}.{{pdf,png}}")


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _draw_gain_tree(
    ax: plt.Axes,
    *,
    x0: float,
    title: str,
    gains: list[float],
    color: str,
    norm: mcolors.Normalize,
    cmap,
) -> None:
    """Draw a compact three-path tree colored by log path gain."""
    soma = (x0 + 0.38, 0.52)
    branch_x = x0 + 0.18
    leaf_x = x0 + 0.02
    ys = [0.78, 0.52, 0.26]
    for idx, y in enumerate(ys):
        leaf = (leaf_x, y)
        branch = (branch_x, y)
        log_gain = np.log10(max(gains[idx], 1e-5))
        line_color = cmap(norm(log_gain))
        ax.plot(
            [leaf[0], branch[0], soma[0]],
            [leaf[1], branch[1], soma[1]],
            color=line_color,
            linewidth=3.0,
            solid_capstyle="round",
            zorder=2,
        )
        ax.add_patch(
            mpatches.Circle(
                leaf,
                0.018,
                facecolor=line_color,
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
        )
        ax.text(
            leaf[0] - 0.020,
            y,
            rf"$\alpha_{idx + 1}$",
            ha="right",
            va="center",
            fontsize=7.4,
            color=COLORS["ink"],
        )
    ax.add_patch(
        mpatches.Circle(
            soma,
            0.035,
            facecolor=COLORS["soma"],
            edgecolor=COLORS["edge"],
            linewidth=0.6,
            zorder=5,
        )
    )
    ax.text(soma[0], soma[1], r"$\delta_0$", ha="center", va="center", fontsize=7.0)
    ax.text(
        x0 + 0.20,
        0.94,
        title,
        ha="center",
        va="center",
        fontsize=8.5,
        color=color,
        fontweight="bold",
    )


def _plot_path_gain_map(ax: plt.Axes, summary: pd.DataFrame) -> None:
    _panel(ax, "A")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)

    mnist = summary[summary["dataset"] == "mnist"].copy()
    add_cv = float(
        mnist[
            (mnist["network_type"] == "dendritic_additive") & (mnist["ie_value"] == 5)
        ]["path_gain_cv_mean_mean"].iloc[0]
    )
    shunt_cv = float(
        mnist[
            (mnist["network_type"] == "dendritic_shunting") & (mnist["ie_value"] == 5)
        ]["path_gain_cv_mean_mean"].iloc[0]
    )

    # The path colors are data-scaled summaries: larger CV produces a broader
    # deterministic spread of representative path gains.
    add_gains = [1.0 - add_cv / 3.0, 1.0, 1.0 + add_cv]
    shunt_gains = [1.0 - shunt_cv / 3.0, 1.0, 1.0 + shunt_cv]
    all_logs = np.log10(np.clip(add_gains + shunt_gains, 1e-5, None))
    norm = mcolors.Normalize(vmin=float(all_logs.min()), vmax=float(all_logs.max()))
    cmap = plt.get_cmap("viridis")

    _draw_gain_tree(
        ax,
        x0=0.06,
        title=f"Additive\nCV={add_cv:.2f}",
        gains=add_gains,
        color=COLOR_ADDITIVE,
        norm=norm,
        cmap=cmap,
    )
    _draw_gain_tree(
        ax,
        x0=0.52,
        title=f"Shunting\nCV={shunt_cv:.2f}",
        gains=shunt_gains,
        color=COLOR_SHUNTING,
        norm=norm,
        cmap=cmap,
    )
    ax.text(
        0.50,
        0.08,
        r"Path colors show $\log_{10}\alpha_n$ at matched $N_I{=}5$",
        ha="center",
        va="center",
        fontsize=7.4,
        color=COLORS["mute"],
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, fraction=0.040, pad=0.01)
    cbar.set_label(r"$\log_{10}\alpha_n$", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7.0, width=0.4, length=2)
    ax.set_title("Path-gain field")


def _plot_error_compressibility(ax: plt.Axes, rank_summary: pd.DataFrame) -> None:
    _panel(ax, "B")
    style_axis(ax, grid="y")
    sub = rank_summary[
        (rank_summary["dataset"] == "mnist")
        & (rank_summary["strategy"] == "local_ca")
        & (rank_summary["rule_variant"] == "5f")
        & (rank_summary["error_broadcast_mode"] == "per_soma")
        & (rank_summary["scope"] == "all_layers")
    ].copy()
    order = ["dendritic_additive", "dendritic_shunting"]
    labels = ["Additive", "Shunting"]
    colors = [COLOR_ADDITIVE, COLOR_SHUNTING]
    x = np.arange(len(order))
    means = []
    stds = []
    pranks = []
    prank_stds = []
    for core in order:
        row = sub[sub["network_type"] == core].iloc[0]
        means.append(float(row["rank1_residual_mean"]))
        stds.append(float(row["rank1_residual_std"]))
        pranks.append(float(row["effective_rank_participation_mean"]))
        prank_stds.append(float(row["effective_rank_participation_std"]))
    bars = ax.bar(
        x,
        means,
        yerr=stds,
        color=colors,
        edgecolor="white",
        linewidth=0.4,
        width=0.58,
        capsize=2,
        error_kw={"lw": 0.7},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rank-1 residual")
    ax.set_ylim(0, 1.02)
    ax.set_title("Error compressibility")
    for rect, mean, prank, prank_std in zip(bars, means, pranks, prank_stds):
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            mean + 0.055,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.8,
        )
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            0.08,
            f"rank\n{prank:.1f}$\\pm${prank_std:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="white",
            fontweight="bold",
        )


def _plot_compartment_error_fidelity(ax: plt.Axes, summary: pd.DataFrame) -> None:
    _panel(ax, "C")
    style_axis(ax, grid="y")
    noise = summary[summary["dataset"] == "noise_resilience"].copy()
    style_map = {
        ("dendritic_shunting", "per_soma"): (COLOR_SHUNTING, "-", "Shunt. rank-1"),
        ("dendritic_shunting", "path_transport"): (COLOR_SHUNTING, "--", "Shunt. transported"),
        ("dendritic_additive", "per_soma"): (COLOR_ADDITIVE, "-", "Add. rank-1"),
        ("dendritic_additive", "path_transport"): (COLOR_ADDITIVE, "--", "Add. transported"),
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

    ax.set_xlabel(r"$N_I$ per branch")
    ax.set_ylabel(r"Cosine$(e_n,\partial L/\partial V_n)$")
    ax.set_title("Broadcast fidelity")
    ax.set_xticks([0, 5, 10, 20, 40])
    ax.set_ylim(-0.35, 1.05)
    ax.legend(
        loc="lower right",
        ncol=1,
        fontsize=7.0,
        handlelength=1.1,
        handletextpad=0.3,
        columnspacing=0.6,
        frameon=False,
    )


def _plot_mechanism_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Panel D: scatter showing the mechanistic chain at a glance.

    x-axis: per-soma cosine alignment to the exact compartment error
    y-axis: test accuracy
    color: dataset (MNIST vs. noise resilience)
    marker: core (additive vs. shunting)
    Points are (core, ie, dataset) cells, 5 seeds each.
    """
    _panel(ax, "D", x=-0.24)
    style_axis(ax, grid="y")

    ds_colors = {"mnist": "#4A7CB5", "noise_resilience": "#E67E22"}
    ds_titles = {"mnist": "MNIST", "noise_resilience": "Noise resil."}
    core_marker = {"dendritic_additive": "o", "dendritic_shunting": "s"}
    core_label = {"dendritic_additive": "Add.", "dendritic_shunting": "Shunt."}

    for ds, ds_df in summary.groupby("dataset"):
        if ds not in ds_colors:
            continue
        for core, sub in ds_df.groupby("network_type"):
            if core not in core_marker:
                continue
            x = sub["per_soma_weighted_cosine_mean"].to_numpy(dtype=float)
            y = 100.0 * sub["test_accuracy_mean"].to_numpy(dtype=float)
            xerr = sub["per_soma_weighted_cosine_std"].fillna(0.0).to_numpy(dtype=float)
            yerr = 100.0 * sub["test_accuracy_std"].fillna(0.0).to_numpy(dtype=float)
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt=core_marker[core],
                color=ds_colors[ds],
                markersize=5.4,
                alpha=0.85,
                capsize=2,
                lw=0.7,
                elinewidth=0.7,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=f"{ds_titles[ds]} {core_label[core]}",
            )

    ax.set_xlabel("Per-soma cosine alignment")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Alignment predicts accuracy")
    ax.legend(
        fontsize=6.2,
        handlelength=1.0,
        handletextpad=0.3,
        loc="lower right",
        framealpha=0.9,
        ncol=2,
    )


def _plot_low_bandwidth(ax: plt.Axes, low_bw: pd.DataFrame, panel_letter: str | None = None) -> None:
    """Broadcast-bandwidth sweep. When used standalone (appendix figure),
    pass panel_letter=None to suppress the panel label."""
    if panel_letter is not None:
        _panel(ax, panel_letter)
    style_axis(ax, grid="y")

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
        color=COLOR_LOW_BW,
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
            color=COLOR_LOW_BW,
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
    ax.set_title("Coarse shared broadcast still works")
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([1, 2, 4, 8, 32])
    ax.set_xticklabels(["1", "2", "4", "8", "32"])
    ax.set_ylim(25, 72)


def _plot_oracle_learning(
    ax: plt.Axes,
    summary: pd.DataFrame,
    oracle_summary: pd.DataFrame,
) -> None:
    _panel(ax, "D")
    style_axis(ax, grid="y")
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
            label=f"{label} rank-1",
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
            label=f"{label} transported",
        )
        ax.fill_between(x2, y2 - err2, y2 + err2, color=color, alpha=0.08, linewidth=0)

    ax.set_xlabel(r"$N_I$ per branch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Transported learning")
    ax.set_xticks([0, 5, 10, 20, 40])
    ax.set_ylim(20, 101)
    ax.legend(
        loc="lower right",
        ncol=1,
        fontsize=7.0,
        handlelength=1.1,
        handletextpad=0.25,
        borderpad=0.22,
        labelspacing=0.22,
        frameon=False,
    )


def build_figure(
    summary_csv: Path = SUMMARY_CSV,
    low_bw_csv: Path = LOW_BW_CSV,  # retained for backward compatibility; not used
    oracle_summary_csv: Path = ORACLE_SUMMARY_CSV,
) -> plt.Figure:
    """Main mechanistic figure. Panel D is the global mechanism-summary scatter
    (was previously the low-bandwidth quantization; that panel is now exported
    as a standalone appendix figure via build_low_bandwidth_figure)."""
    _setup_style()
    summary = _safe_csv(summary_csv)
    oracle_summary = _safe_csv(oracle_summary_csv)
    rank_summary = _safe_csv(ERROR_RANK_SUMMARY_CSV)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(10.2, 3.15),
        gridspec_kw={"wspace": 0.56, "width_ratios": [1.08, 0.86, 1.05, 1.12]},
    )

    _plot_path_gain_map(axes[0], summary)
    _plot_error_compressibility(axes[1], rank_summary)
    _plot_compartment_error_fidelity(axes[2], summary)
    _plot_oracle_learning(axes[3], summary, oracle_summary)
    fig.subplots_adjust(left=0.060, right=0.992, top=0.84, bottom=0.25, wspace=0.56)
    return fig


def build_low_bandwidth_figure(
    low_bw_csv: Path = LOW_BW_CSV,
) -> plt.Figure:
    """Standalone appendix figure: broadcast-bandwidth (quantization) sweep.

    This was previously panel D of the main mechanistic figure; it was moved
    to the appendix so the main figure can carry the full causal chain
    (path-gain CV -> alignment -> oracle -> global summary scatter).
    """
    _setup_style()
    low_bw = _safe_csv(low_bw_csv)
    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.5))
    _plot_low_bandwidth(ax, low_bw, panel_letter=None)
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

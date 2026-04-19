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
            "font.size": 8.2,
            "axes.labelsize": 8.8,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.8,
            "legend.frameon": True,
            "legend.framealpha": 0.96,
            "legend.edgecolor": "#D0D5DD",
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


def _panel(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.12) -> None:
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


def _plot_path_gain_dispersion(ax: plt.Axes, summary: pd.DataFrame) -> None:
    _panel(ax, "A")
    style_axis(ax, grid="y")
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
    ax.set_title("Path-gain concentration")
    ax.set_xticks([0, 5, 10, 20, 40])
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left")


def _plot_compartment_error_fidelity(ax: plt.Axes, summary: pd.DataFrame) -> None:
    _panel(ax, "B")
    style_axis(ax, grid="y")
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
    ax.set_title("Compartment-error fidelity")
    ax.set_xticks([0, 5, 10, 20, 40])
    ax.set_ylim(-0.35, 1.05)
    ax.legend(loc="upper left", ncol=1, handlelength=1.2, handletextpad=0.4,
              columnspacing=0.8)


def _plot_mechanism_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Panel D: scatter showing the mechanistic chain at a glance.

    x-axis: per-soma cosine alignment to the exact compartment error
    y-axis: test accuracy
    color: dataset (MNIST vs. noise resilience)
    marker: core (additive vs. shunting)
    Points are (core, ie, dataset) cells, 5 seeds each.
    """
    _panel(ax, "D")
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
    ax.set_title("Mechanism summary (alignment $\\to$ accuracy)")
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
    _panel(ax, "C")
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
    ax.set_title("Transported error improves learning")
    ax.set_xticks([0, 5, 10, 20, 40])
    ax.set_ylim(20, 101)
    ax.legend(loc="upper left", ncol=1, handlelength=1.2, handletextpad=0.4,
              columnspacing=0.8)

    # Harder-data inset: corrected CIFAR-10 strong family (shunting only).
    try:
        cifar_bp = _safe_csv(CIFAR10_BP_SUMMARY_CSV)
        cifar_local = _safe_csv(CIFAR10_LOCALCA_SUMMARY_CSV)
        bp_row = cifar_bp[
            (cifar_bp["strategy"] == "standard")
            & (cifar_bp["model_type"] == "dendritic_shunting")
        ].iloc[0]
        ps_row = cifar_local[
            cifar_local["condition"] == "cifar10_shunting_5f_per_soma_bpdec_wd0"
        ].iloc[0]
        pt_row = cifar_local[
            cifar_local["condition"] == "cifar10_shunting_5f_path_transport_bpdec_wd0"
        ].iloc[0]

        # Only create the inset AFTER data validation succeeds.
        inset = ax.inset_axes([0.56, 0.08, 0.39, 0.34])
        style_axis(inset, grid="y")
        vals = np.array([
            100.0 * float(bp_row["mean_test_accuracy"]),
            100.0 * float(ps_row["acc_test_mean"]),
            100.0 * float(pt_row["acc_test_mean"]),
        ])
        errs = np.array([
            100.0 * float(bp_row["std_test_accuracy"]),
            100.0 * float(ps_row["acc_test_std"]),
            100.0 * float(pt_row["acc_test_std"]),
        ])
        colors = [COLORS["bp"], COLOR_SHUNTING, COLOR_TRANSPORT]
        bars = inset.bar(
            np.arange(3),
            vals,
            yerr=errs,
            color=colors,
            edgecolor="white",
            lw=0.3,
            width=0.55,
            capsize=1.5,
            error_kw={"lw": 0.5},
            zorder=3,
        )
        for rect, v in zip(bars, vals):
            inset.text(
                rect.get_x() + rect.get_width() / 2,
                v + 0.9,
                f"{v:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.2,
            )
        inset.set_xticks(np.arange(3))
        inset.set_xticklabels(["BP", "Per", "Trans"], fontsize=5.6)
        inset.set_ylim(20, 55)
        inset.set_title("CIFAR-10 shunt.", fontsize=6.0, pad=1.5)
        inset.tick_params(axis="y", labelsize=5.4)
    except Exception as exc:
        # Surface silent failures instead of leaving an empty inset behind.
        print(f"  [warn] CIFAR inset skipped: {type(exc).__name__}: {exc}")
        # Remove any empty inset so it doesn't leave a ghost subplot.
        try:
            inset.remove()
        except Exception:
            pass


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

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 8.2),
        gridspec_kw={"wspace": 0.28, "hspace": 0.38,
                     "width_ratios": [1.0, 1.0],
                     "height_ratios": [1.0, 1.05]},
    )
    axes = axes.flatten()

    _plot_path_gain_dispersion(axes[0], summary)
    _plot_compartment_error_fidelity(axes[1], summary)
    _plot_oracle_learning(axes[2], summary, oracle_summary)
    _plot_mechanism_summary(axes[3], summary)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.94, bottom=0.07)
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

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

from neurips_style import (  # noqa: E402
    # noqa: E402,
    COLORS,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MAIN_W,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    PT_TICK,
    REF_LW,
    add_headroom,
    apply_neurips_style,
    clean_legend,
    grid_figure,
    panel_label,
    panel_title,
    style_axis,
    tidy_ticks,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = DRAFT_DIR / "figures"
DATA_DIR = DRAFT_DIR / "data"
ANALYSIS_DIR = DRAFT_DIR / "analysis"
FIGURE_DATA_DIR = FIGURES_DIR / "data"


def _tracked_csv(tracked_name: str, fallback: Path) -> Path:
    """Prefer the git-tracked figures/data/ copy; fall back to local analysis/."""
    tracked = FIGURE_DATA_DIR / tracked_name
    return tracked if tracked.exists() else fallback


SUMMARY_DIR = ANALYSIS_DIR / "theory_diag_gradient_fidelity_vs_ie_nonnegativeinput_fix_summary"
SUMMARY_CSV = _tracked_csv("theory_diag_by_condition.csv", SUMMARY_DIR / "theory_diag_by_condition.csv")
MERGED_CSV = SUMMARY_DIR / "theory_diag_merged_runs.csv"
PATH_GAIN_SEED_CSV = _tracked_csv(
    "path_gain_cv_mnist_ni5_seed.csv",
    MERGED_CSV,
)
ORACLE_SUMMARY_CSV = _tracked_csv(
    "path_transport_upper_bound_summary.csv",
    ANALYSIS_DIR
    / "path_transport_upper_bound_nonnegativeinput_fix_5seed"
    / "path_transport_upper_bound_summary.csv",
)
ERROR_FIELD_DECOMPOSITION_RUNS_CSV = _tracked_csv(
    "error_field_decomposition_runs.csv",
    ANALYSIS_DIR
    / "scope_corrected_decomposition_20260725"
    / "error_field_decomposition_runs.csv",
)
INPUT_MODE_SUMMARY_CSV = (
    ANALYSIS_DIR
    / "input_mode_onelayer_probe_summary_20260425"
    / "input_mode_onelayer_grouped.csv"
)
INHIBITION_CAUSALITY_CSV = _tracked_csv(
    "inhibition_causality_runs.csv",
    ANALYSIS_DIR
    / "inhibition_causality_selected_20260427"
    / "inhibition_causality_runs.csv",
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
    """Apply the shared paper style.

    This module previously re-declared font sizes and legend framing on top of
    apply_neurips_style(), which silently diverged this figure from every other
    one (most visibly: framed legends here, frameless everywhere else).  Only
    output resolution is overridden now; typography comes from the single
    shared source of truth.
    """
    apply_neurips_style()
    plt.rcParams.update(
        {
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
        }
    )


def _panel(ax: plt.Axes, label: str) -> None:
    # Points-offset placement (see neurips_style.panel_label): identical on
    # every panel regardless of width, and cannot collide with a wrapped title.
    panel_label(ax, label)


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
    label_soma: bool = True,
) -> None:
    """Draw a compact three-path tree colored by log path gain."""
    soma = (x0 + 0.32, 0.52)
    branch_x = x0 + 0.15
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
            linewidth=LW_DATA,
            solid_capstyle="round",
            zorder=2,
        )
        ax.add_patch(
            mpatches.Circle(
                leaf,
                0.018,
                facecolor=line_color,
                edgecolor="white",
                linewidth=LW_HAIR,
                zorder=4,
            )
        )
        ax.text(
            leaf[0] - 0.020,
            y,
            rf"$\alpha_{idx + 1}$",
            ha="right",
            va="center",
            fontsize=PT_TICK,
            color=COLORS["ink"],
            zorder=6,
            bbox={"facecolor": "white", "edgecolor": "none",
                  "pad": 0.6, "alpha": 0.88},
        )
    ax.add_patch(
        mpatches.Circle(
            soma,
            0.035,
            facecolor=COLORS["soma"],
            edgecolor=COLORS["edge"],
                linewidth=LW_EDGE,
            zorder=5,
        )
    )
    if label_soma:
        ax.text(soma[0], soma[1], r"$\delta_0$", ha="center", va="center",
                fontsize=PT_LEGEND, zorder=6,
                bbox={"facecolor": "white", "edgecolor": "none",
                      "pad": 0.5, "alpha": 0.85})
    ax.text(
        x0 + 0.17,
        0.88,
        title,
        ha="center",
        va="center",
        fontsize=PT_SMALL,
        color=color,
        fontweight="bold",
        linespacing=0.86,
    )


def _plot_path_gain_map(
    ax: plt.Axes,
    summary: pd.DataFrame,
    path_gain_seed: pd.DataFrame,
) -> None:
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
        x0=0.02,
        title=f"Add.\n{add_cv:.2f}",
        gains=add_gains,
        color=COLOR_ADDITIVE,
        norm=norm,
        cmap=cmap,
        label_soma=False,
    )
    _draw_gain_tree(
        ax,
        x0=0.62,
        title=f"Shunt.\n{shunt_cv:.2f}",
        gains=shunt_gains,
        color=COLOR_SHUNTING,
        norm=norm,
        cmap=cmap,
    )
    # Show the actual paired seed-level result, rather than only the two
    # aggregate CV labels above the schematic.
    paired = path_gain_seed.copy()
    if "dataset" in paired:
        paired = paired[paired["dataset"] == "mnist"]
    if "ie_value" in paired:
        paired = paired[paired["ie_value"] == 5]
    pivot = paired.pivot_table(
        index="seed",
        columns="network_type",
        values="path_gain_cv_mean",
        aggfunc="mean",
    ).dropna()
    inset = ax.inset_axes([0.12, 0.004, 0.76, 0.185])
    add_vals = pivot["dendritic_additive"].to_numpy(dtype=float)
    shunt_vals = pivot["dendritic_shunting"].to_numpy(dtype=float)
    means = [float(np.mean(add_vals)), float(np.mean(shunt_vals))]
    sds = [float(np.std(add_vals, ddof=1)) if len(add_vals) > 1 else 0.0,
           float(np.std(shunt_vals, ddof=1)) if len(shunt_vals) > 1 else 0.0]
    inset.bar(
        [0, 1], means, 0.56, yerr=sds,
        color=[COLOR_ADDITIVE, COLOR_SHUNTING],
        edgecolor="white", linewidth=LW_HAIR,
        error_kw={"lw": 0.7, "capthick": 0.7}, capsize=2.0, zorder=2,
    )
    for x, vals in ((0, add_vals), (1, shunt_vals)):
        if len(vals):
            jitter = np.linspace(-0.13, 0.13, len(vals))
            inset.scatter(
                np.full(len(vals), x) + jitter, vals,
                s=5, facecolors="white", edgecolors=COLORS["edge"],
                linewidths=0.4, zorder=3,
            )
    inset.set_xlim(-0.55, 1.55)
    inset.set_ylim(0.0, 1.36)
    inset.set_xticks([0, 1])
    inset.set_xticklabels(["Add.", "Shunt."], fontsize=PT_SMALL - 0.6)
    inset.set_yticks([0, 1])
    inset.set_yticklabels(["0", "1"], fontsize=PT_SMALL)
    inset.tick_params(length=1.8, width=0.55, pad=0.6)
    inset.spines["left"].set_linewidth(0.55)
    inset.spines["bottom"].set_linewidth(0.55)
    inset.set_ylabel("CV", fontsize=PT_SMALL, labelpad=1.0)
    panel_title(ax, "A", "Path gains")


def _plot_dendritic_feedback_fidelity(
    ax: plt.Axes,
    decomposition_runs: pd.DataFrame,
) -> None:
    """Plot submitted-field fidelity only where feedback is actually restricted.

    The former panel pooled distal, proximal, and width-matched somatic stages.
    At the somatic stage the submitted mode reuses the soma vector verbatim,
    giving cosine one by construction. Unequal somatic error energy therefore
    made the pooled cross-core contrast an energy-allocation diagnostic rather
    than a dendritic-feedback comparison.
    """
    style_axis(ax, grid="y")
    sub = decomposition_runs[
        (decomposition_runs["dataset"] == "mnist")
        & (decomposition_runs["strategy"] == "local_ca")
        & (decomposition_runs["rule_variant"] == "5f")
        & (decomposition_runs["error_broadcast_mode"] == "per_soma")
        & (decomposition_runs["scope_type"] == "stage")
        & (decomposition_runs["population"] == 0)
        & (decomposition_runs["stage_role"].isin(["distal", "proximal"]))
        & (decomposition_runs["family"] == "submitted_mw")
    ].copy()
    if sub.empty:
        raise ValueError("Scope-corrected stage-resolved submitted-field data are unavailable")

    # Average train/validation/test batches within each checkpoint first, so
    # checkpoints—not diagnostic batches—remain the independent units.
    per_checkpoint = (
        sub.groupby(
            ["run_name", "seed", "network_type", "stage_role"],
            as_index=False,
        )["cosine"]
        .mean()
    )

    stages = ["distal", "proximal"]
    stage_labels = ["Distal", "Proximal"]
    cores = [
        ("dendritic_additive", "Add.", COLOR_ADDITIVE),
        ("dendritic_shunting", "Shunt.", COLOR_SHUNTING),
    ]
    x = np.arange(len(stages))
    width = 0.34
    for offset, (core, label, color) in zip([-width / 2, width / 2], cores):
        means = []
        stds = []
        for stage in stages:
            values = per_checkpoint[
                (per_checkpoint["network_type"] == core)
                & (per_checkpoint["stage_role"] == stage)
            ]["cosine"]
            if len(values) != 5:
                raise ValueError(
                    f"Expected five checkpoint values for {core}/{stage}, got {len(values)}"
                )
            means.append(float(values.mean()))
            stds.append(float(values.std(ddof=1)))
        ax.bar(
            x + offset,
            means,
            yerr=stds,
            color=color,
            edgecolor="white",
            linewidth=LW_EDGE,
            width=width,
            capsize=2.4,
            error_kw={"lw": 1.1},
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=18, ha="right")
    ax.set_ylabel("MW-field cosine")
    ax.set_ylim(0, 0.30)
    tidy_ticks(ax, ny=4)
    panel_title(ax, "B", "Field cosine")
    clean_legend(ax, loc="upper center",
        ncol=1,
        frameon=False,
        handlelength=1.0,
        handletextpad=0.35,
        borderpad=0.1,
    )


def _plot_compartment_error_fidelity(ax: plt.Axes, summary: pd.DataFrame) -> None:
    style_axis(ax, grid="y")
    noise = summary[summary["dataset"] == "noise_resilience"].copy()
    style_map = {
        ("dendritic_shunting", "per_soma"): (COLOR_SHUNTING, "-", "Shunt. MW"),
        ("dendritic_shunting", "path_transport"): (COLOR_SHUNTING, "--", "Shunt. oracle"),
        ("dendritic_additive", "per_soma"): (COLOR_ADDITIVE, "-", "Add. MW"),
        ("dendritic_additive", "path_transport"): (COLOR_ADDITIVE, "--", "Add. oracle"),
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
            markersize=5.0,
            lw=LW_DATA,
            linestyle=linestyle,
            color=color,
            label=label,
        )
        ax.fill_between(x, y - err, y + err, color=color, alpha=0.10, linewidth=0)

    ax.set_xlabel(r"$N_I$ per branch")
    ax.set_ylabel("Cosine")
    tidy_ticks(ax, ny=4)
    panel_title(ax, "D", "Fidelity")
    # Label 0/10/20/40 only: the axis is linear in N_I, so 0-5-10 fall within
    # the first ~18% of the span and their labels collided. The N_I=5 point is
    # still plotted; dropping only its tick label keeps the axis honestly linear
    # (an equal-spaced categorical axis would misrepresent the sweep spacing).
    ax.set_xticks([0, 20, 40])
    ax.set_ylim(-0.35, 1.05)
    ax.set_xlim(-1.5, 52.5)
    ax.text(24.0, 0.34, "MW", color=COLORS["ink"], fontsize=PT_ANNOT,
            fontweight="bold", ha="center", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.9})
    ax.text(24.0, 0.84, "oracle", color=COLORS["ink"], fontsize=PT_ANNOT,
            fontweight="bold", ha="center", va="top", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.9})


def _plot_mechanism_summary(ax: plt.Axes, summary: pd.DataFrame) -> None:
    """Panel D: scatter showing the mechanistic chain at a glance.

    x-axis: submitted matched-width/scalar-fallback cosine alignment
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
                markersize=6.2,
                alpha=0.85,
                capsize=2,
                lw=LW_ERR,
                elinewidth=1.0,
                markeredgecolor="white",
                markeredgewidth=0.5,
                label=f"{ds_titles[ds]} {core_label[core]}",
            )

    ax.set_xlabel("Submitted-field cosine")
    ax.set_ylabel("Test (%)")
    ax.set_title("Alignment predicts accuracy")
    clean_legend(ax, fontsize=PT_ANNOT,
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
        lw=LW_REF,
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
            fontsize=PT_SMALL,
            ha="center",
            va=va,
            color=COLOR_LOW_BW,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85},
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
            fontsize=PT_SMALL,
            color=COLOR_TRANSPORT,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85},
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
        markersize=4.9,
        lw=LW_DATA,
            color=color,
            linestyle="-",
            label=f"{label} MW",
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
            markersize=4.5,
            lw=LW_DATA,
            color=color,
            linestyle="--",
            label=f"{label} oracle",
        )
        ax.fill_between(x2, y2 - err2, y2 + err2, color=color, alpha=0.08, linewidth=0)

    ax.set_xlabel(r"$N_I$ per branch")
    ax.set_ylabel("Test accuracy (%)")
    tidy_ticks(ax, ny=4)
    panel_title(ax, "E", "Learning")
    # Label 0/10/20/40 only: the axis is linear in N_I, so 0-5-10 fall within
    # the first ~18% of the span and their labels collided. The N_I=5 point is
    # still plotted; dropping only its tick label keeps the axis honestly linear
    # (an equal-spaced categorical axis would misrepresent the sweep spacing).
    ax.set_xticks([0, 20, 40])
    ax.set_ylim(20, 101)
    ax.set_xlim(-1.5, 52.5)
    ax.text(24.0, 77.0, "MW", color=COLORS["ink"], fontsize=PT_ANNOT,
            fontweight="bold", ha="center", va="top", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.9})
    ax.text(24.0, 96.5, "oracle", color=COLORS["ink"], fontsize=PT_ANNOT,
            fontweight="bold", ha="center", va="bottom", bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.9})


def _plot_causal_inhibition(ax: plt.Axes, causal: pd.DataFrame) -> None:
    style_axis(ax, grid="y")

    order = ["original", "zero_i", "shuffle_i", "mean_clamp_i", "uniform_matched_i"]
    labels = {
        "original": "Learned",
        "zero_i": "Zero",
        "shuffle_i": "Shuf.",
        "mean_clamp_i": "Mean",
        "uniform_matched_i": "Unif.",
    }
    colors = {
        "original": COLOR_SHUNTING,
        "zero_i": COLORS["mute"],
        "shuffle_i": COLORS["low_rank"],
        "mean_clamp_i": COLORS["per_soma"],
        "uniform_matched_i": COLOR_TRANSPORT,
    }
    dataset_order = [("mnist", "MNIST"), ("noise_resilience", "Noise")]
    group_gap = 0.70
    width = 0.205
    xs_all = []
    means_all = []
    stds_all = []
    colors_all = []

    for gi, (dataset, _label) in enumerate(dataset_order):
        center = gi * (len(order) * width + group_gap)
        sub = causal[causal["dataset"] == dataset].copy()
        stats = sub.groupby("intervention")["accuracy"].agg(["mean", "std"]).reindex(order)
        for ji, intervention in enumerate(order):
            x = center + (ji - (len(order) - 1) / 2) * width
            xs_all.append(x)
            means_all.append(100.0 * float(stats.loc[intervention, "mean"]))
            stds_all.append(100.0 * float(stats.loc[intervention, "std"]))
            colors_all.append(colors[intervention])

            vals = 100.0 * sub[sub["intervention"] == intervention]["accuracy"].to_numpy(dtype=float)
            if vals.size:
                jitter = np.linspace(-0.062, 0.062, vals.size)
                ax.scatter(
                    np.full(vals.size, x) + jitter,
                    vals,
                    s=4.5,
                    color="white",
                    edgecolor=COLORS["ink"],
                    linewidth=LW_HAIR,
                    zorder=5,
                )

    ax.bar(
        xs_all,
        means_all,
        width * 0.88,
        yerr=stds_all,
        color=colors_all,
        edgecolor="white",
        linewidth=LW_HAIR,
        capsize=1.8,
        error_kw={"lw": 0.85},
        zorder=3,
    )
    centers = [gi * (len(order) * width + group_gap) for gi, _ in enumerate(dataset_order)]
    ax.set_xticks(centers)
    ax.set_xticklabels([label for _dataset, label in dataset_order], fontsize=PT_LEGEND)
    if len(centers) == 2:
        ax.axvline((centers[0] + centers[1]) / 2, color=COLORS["edge"], lw=LW_HAIR, alpha=0.85)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Accuracy (%)")
    add_headroom(ax, 0.34)
    tidy_ticks(ax, ny=4)
    panel_title(ax, "C", "Inhibition")
    legend_handles = [mpatches.Patch(color=colors[k], label=labels[k]) for k in order]
    clean_legend(ax, handles=legend_handles,
        fontsize=PT_SMALL,
        loc="upper center",
        ncol=3,
        handlelength=0.85,
        handletextpad=0.3,
        columnspacing=0.6,
        borderpad=0.2,
        labelspacing=0.25,
        frameon=False,
    )


def _plot_inhibitory_path_probe(ax: plt.Axes, input_mode: pd.DataFrame) -> None:
    _panel(ax, "E")
    style_axis(ax, grid="x")

    def _row(condition: str) -> pd.Series:
        rows = input_mode[input_mode["condition"] == condition]
        if rows.empty:
            raise KeyError(condition)
        return rows.iloc[0]

    direct_rank1 = _row("direct_i_stream__localca_per_soma")
    direct_path = _row("direct_i_stream__localca_path_transport")
    explicit_path = _row("explicit_i_cells__localca_path_transport__i_updates_True")
    explicit_bp = _row("explicit_i_cells__standard_bp")
    rows = [
        ("MW/scalar", direct_rank1, COLORS["per_soma"], ""),
        ("Path transp.", direct_path, COLOR_TRANSPORT, ""),
        ("I-cell path", explicit_path, COLORS["shunting"], "//"),
    ]
    bp = 100.0 * float(explicit_bp["test_accuracy_mean"])
    y = np.arange(len(rows))[::-1]
    means = np.asarray([100.0 * float(row["test_accuracy_mean"]) for _, row, _, _ in rows])
    errs = np.asarray([100.0 * float(row["test_accuracy_std"]) for _, row, _, _ in rows])
    colors = [color for _, _, color, _ in rows]
    hatches = [hatch for _, _, _, hatch in rows]

    for yi, mean, err, color, hatch in zip(y, means, errs, colors, hatches):
        ax.barh(
            yi,
            mean - 80.0,
            left=80.0,
            height=0.48,
            color=color,
            edgecolor="white",
            linewidth=LW_EDGE,
            hatch=hatch,
            alpha=0.92,
            zorder=2,
        )
        ax.errorbar(
            mean,
            yi,
            xerr=err,
            fmt="o",
            markersize=5.0,
            color=COLORS["ink"],
            ecolor=COLORS["ink"],
            elinewidth=1.1,
            capsize=2.2,
            capthick=0.75,
            zorder=4,
        )
        ax.text(
            mean + 0.42,
            yi,
            f"{mean:.1f}",
            ha="left",
            va="center",
            fontsize=PT_TICK,
            color=COLORS["ink"],
        )

    ax.axvline(bp, color=COLORS["bp"], linestyle="--", linewidth=LW_REF, alpha=0.90, zorder=3)
    ax.text(
        bp - 0.15,
        2.64,
        "BP",
        ha="right",
        va="bottom",
        fontsize=PT_TICK,
        color=COLORS["bp"],
        fontweight="bold",
    )
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _, _ in rows], fontsize=PT_TICK)
    ax.set_xlim(80.0, 98.0)
    ax.set_xticks([80, 85, 90, 95])
    ax.set_xlabel("Probe accuracy (%)")
    ax.set_title("Pathway\nprobe", pad=5, linespacing=0.9)
    ax.set_ylim(-0.55, 2.72)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


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
    path_gain_seed = _safe_csv(PATH_GAIN_SEED_CSV)
    oracle_summary = _safe_csv(oracle_summary_csv)
    decomposition_runs = _safe_csv(ERROR_FIELD_DECOMPOSITION_RUNS_CSV)
    causal = _safe_csv(INHIBITION_CAUSALITY_CSV)

    # Sized to NeurIPS \textwidth (≈7 in) so the printed figure does not
    # need to be down-scaled from the matplotlib render — the new larger
    # global font sizes therefore render at intended size in the PDF.
    # Authored at MAIN_W so this figure takes the same LaTeX rescaling as every
    # other main figure (previously 7.0 in here vs 7.35 / 6.95 / 5.5 elsewhere,
    # which made identical nominal type print at a different size per figure).
    # Equal width_ratios: the panels carry comparable content, and the old
    # [1.18, 0.80, 1.52, 0.96, 0.96] both looked ragged and squeezed panel B so
    # hard that its legend had to be shrunk to 5.2 pt (~4 pt printed) to fit.
    fig, axes = grid_figure(5, width_ratios=[1.0, 1.0, 1.12, 1.0, 1.0])

    _plot_path_gain_map(axes[0], summary, path_gain_seed)
    _plot_dendritic_feedback_fidelity(axes[1], decomposition_runs)
    _plot_causal_inhibition(axes[2], causal)
    _plot_compartment_error_fidelity(axes[3], summary)
    _plot_oracle_learning(axes[4], summary, oracle_summary)

    # Typography comes from the shared style; only the "(n=3)" annotation is
    # stripped from titles. The former per-panel font shrinking (titles 9.6,
    # ticks 7.6, legends 5.2/6.6) is gone: it defeated the global style and drove
    # legend text below legibility once LaTeX rescaled the figure.
    for ax in axes:
        title = ax.get_title()
        if "(n=3)" in title:
            ax.set_title(title.replace("(n=3)", "").strip())

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
    parser.add_argument("--name", type=str, default="fig3_mechanistic_evidence")
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

#!/usr/bin/env python3
"""Generate a publication-ready morphology x inhibition regime-map figure."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from neurips_style import apply_neurips_style, COLORS, panel_label, style_axis


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = DRAFT_DIR / "figures"
ANALYSIS_DIR = DRAFT_DIR / "analysis" / "morphology_ie_regime"
DIAG_DIR = DRAFT_DIR / "analysis" / "morphology_ie_diag_subset_diagnostics_20260427"
SELECTED_DIAG_CSV = DRAFT_DIR / "analysis" / "morphology_ie_diag_subset" / "selected_runs.csv"
DEFAULT_SWEEP_DIR = (
    DRAFT_DIR
    / "local_sweep_runs"
    / "noise_resilience_morphology_ie_regime_nonnegativeinput_fix_20260409164042"
)

COLOR_SHUNTING = COLORS["shunting"]
COLOR_ADDITIVE = COLORS["additive"]


def _load_results(sweep_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cfg_path in sorted((sweep_dir / "configs").glob("unified_config_*.yaml")):
        idx = cfg_path.stem.split("_")[-1]
        perf_path = sweep_dir / "results" / f"config_{idx}" / "performance" / "final.json"
        if not perf_path.exists():
            continue
        cfg = yaml.safe_load(cfg_path.read_text())
        perf = json.loads(perf_path.read_text())
        branch_factors = cfg["model"]["core"]["architecture"]["excitatory_branch_factors"]
        rows.append(
            {
                "config_idx": int(idx),
                "network_type": cfg["model"]["core"]["type"],
                "branch_factors": str(branch_factors),
                "depth": len(branch_factors),
                "branch_product": math.prod(branch_factors),
                "ie": cfg["model"]["core"]["connectivity"]["ie_synapses_per_branch_per_layer"][0],
                "seed": cfg["experiment"]["seed"],
                "test_acc": perf["accuracy"]["test"],
                "valid_acc": perf["accuracy"]["valid"],
                "train_acc": perf["accuracy"]["train"],
            }
        )
    if not rows:
        raise RuntimeError(f"No completed results found under {sweep_dir}")
    return pd.DataFrame(rows)


def _ordered_branch_factors(values: list[str]) -> list[str]:
    def _key(text: str) -> tuple[int, int, str]:
        nums = [int(x.strip()) for x in text.strip("[]").split(",")]
        return (len(nums), math.prod(nums), text)

    return sorted(values, key=_key)


def _heatmap(ax, frame: pd.DataFrame, title: str, cmap: str, center: float | None = None) -> None:
    sns.heatmap(
        frame,
        ax=ax,
        cmap=cmap,
        center=center,
        annot=True,
        fmt=".2f",
        cbar=True,
        linewidths=0.5,
        linecolor="white",
        square=False,
        annot_kws={"fontsize": 8},
    )
    ax.collections[0].colorbar.outline.set_linewidth(0.4)
    ax.collections[0].colorbar.ax.tick_params(labelsize=8, width=0.4, length=2)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlabel(r"$N_I$ per branch", fontsize=10)
    ax.set_ylabel("Morphology", fontsize=10)
    ax.tick_params(labelsize=9)


def _load_selected_diagnostics() -> pd.DataFrame | None:
    summary_csv = DIAG_DIR / "run_summary.csv"
    if not summary_csv.exists() or not SELECTED_DIAG_CSV.exists():
        return None
    diag = pd.read_csv(summary_csv)
    selected = pd.read_csv(SELECTED_DIAG_CSV)
    selected["run_dir_short"] = selected["run_dir"].astype(str).str.extract(r"(results/config_\d+)$")[0]
    diag["run_dir_short"] = diag["run_dir"].astype(str).str.extract(r"(results/config_\d+)$")[0]
    merged = selected.merge(diag, on="run_dir_short", how="inner", suffixes=("", "_diag"))
    if not merged.empty:
        return merged
    if len(selected) != len(diag):
        return None
    selected = selected.reset_index(drop=True)
    diag = diag.reset_index(drop=True)
    return pd.concat([selected, diag.drop(columns=["run_dir"], errors="ignore")], axis=1)


def build_figure(sweep_dir: Path = DEFAULT_SWEEP_DIR) -> tuple[plt.Figure, pd.DataFrame]:
    apply_neurips_style()
    sns.set_style("white")
    df = _load_results(sweep_dir)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    grouped = (
        df.groupby(["network_type", "branch_factors", "depth", "branch_product", "ie"])
        .agg(test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"))
        .reset_index()
    )

    branch_order = _ordered_branch_factors(grouped["branch_factors"].unique().tolist())
    ie_order = sorted(grouped["ie"].unique().tolist())

    shunting = grouped[grouped["network_type"] == "dendritic_shunting"].pivot(
        index="branch_factors", columns="ie", values="test_acc_mean"
    ).reindex(index=branch_order, columns=ie_order)
    additive = grouped[grouped["network_type"] == "dendritic_additive"].pivot(
        index="branch_factors", columns="ie", values="test_acc_mean"
    ).reindex(index=branch_order, columns=ie_order)
    gap = shunting - additive

    gap_long = (
        grouped.pivot_table(
            index=["branch_factors", "depth", "branch_product", "ie"],
            columns="network_type",
            values="test_acc_mean",
        )
        .reset_index()
        .assign(gap=lambda x: x["dendritic_shunting"] - x["dendritic_additive"])
    )
    best_ie = (
        gap_long.sort_values(["branch_factors", "gap"], ascending=[True, False])
        .groupby("branch_factors")
        .head(1)
        .copy()
    )
    depth_gap = (
        gap_long.groupby(["depth", "ie"])["gap"]
        .mean()
        .reset_index()
    )
    depth_peak = (
        depth_gap.sort_values(["depth", "gap"], ascending=[True, False])
        .groupby("depth")
        .head(1)
        .copy()
    )
    diag = _load_selected_diagnostics()

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14.0, 3.4),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.12, 0.92, 1.0, 1.05]},
    )

    _heatmap(axes[0], gap, "Shunting advantage", "vlag", center=0.0)
    axes[0].set_title("Gap: shunting - additive", fontsize=11, pad=8)

    ax = axes[1]
    style_axis(ax, grid="x")
    best_ie = best_ie.set_index("branch_factors").reindex(branch_order).reset_index()
    y = np.arange(len(best_ie))
    colors = [COLORS["shunting"] if v > 0 else COLORS["additive"] for v in best_ie["gap"]]
    ax.barh(y, best_ie["ie"], color=colors, edgecolor="white", linewidth=0.5, height=0.56)
    ax.set_yticks(y)
    ax.set_yticklabels(best_ie["branch_factors"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"Best $N_I$")
    ax.set_title("Best inhibition by tree", fontsize=11, pad=8)
    for yi, (_, row) in enumerate(best_ie.iterrows()):
        ax.text(
            row["ie"] + 0.7,
            yi,
            f"{100 * row['gap']:+.1f} pp",
            va="center",
            ha="left",
            fontsize=7,
            color=COLORS["ink"],
        )
    ax.set_xlim(0, max(ie_order) + 11)

    ax = axes[2]
    style_axis(ax, grid="y")
    if diag is not None:
        diag = diag.sort_values(["branch_factors", "ie", "network_type"])
        diag["label"] = diag["branch_factors"] + "\n" + diag["ie"].astype(str)
        keep = diag[
            diag["tag"].isin(
                [
                    "best_depth2_lowI_shunting",
                    "best_depth2_lowI_additive",
                    "best_depth3_midI_shunting",
                    "best_depth3_midI_additive",
                    "highI_collapse_shunting",
                    "highI_match_additive",
                ]
            )
        ].copy()
        labels = []
        vals = []
        colors = []
        for tag in [
            "best_depth2_lowI_additive",
            "best_depth2_lowI_shunting",
            "best_depth3_midI_additive",
            "best_depth3_midI_shunting",
            "highI_match_additive",
            "highI_collapse_shunting",
        ]:
            row = keep[keep["tag"] == tag]
            if row.empty:
                continue
            row = row.iloc[0]
            labels.append(f"{row['branch_factors']}\n$N_I$={int(row['ie'])}")
            vals.append(float(row["path_gain_cv_mean"]))
            colors.append(COLOR_SHUNTING if row["network_type"] == "dendritic_shunting" else COLOR_ADDITIVE)
        x = np.arange(len(vals))
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.45, width=0.70)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.8)
        ax.set_ylabel("Path-gain CV")
        ax.set_title("Representative credit geometry", fontsize=11, pad=8)
        handles = [
            plt.Line2D([0], [0], color=COLOR_ADDITIVE, lw=5, label="Additive"),
            plt.Line2D([0], [0], color=COLOR_SHUNTING, lw=5, label="Shunting"),
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=6.8, frameon=False)
    else:
        ax.text(
            0.5,
            0.58,
            "Selected morphology\npath-gain diagnostics\npending",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8.2,
            color=COLORS["mute"],
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Representative credit geometry", fontsize=11, pad=8)

    ax = axes[3]
    style_axis(ax, grid="y")
    depth_palette = {2: "#7B5EA7", 3: "#D95F02"}
    for depth, sub in depth_gap.groupby("depth"):
        ax.plot(
            sub["ie"],
            sub["gap"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            color=depth_palette.get(depth, "#444444"),
            label=f"depth {depth}",
        )
    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_title("Average gain by depth", fontsize=11, pad=8)
    ax.set_xlabel(r"$N_I$ per branch", fontsize=10)
    ax.set_ylabel("Shunting - additive accuracy", fontsize=10)
    ax.set_xticks(ie_order)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="best")
    for _, row in depth_peak.iterrows():
        ax.scatter(
            [row["ie"]],
            [row["gap"]],
            s=52,
            facecolor="white",
            edgecolor=depth_palette.get(int(row["depth"]), "#444444"),
            linewidth=1.5,
            zorder=6,
        )

    for label, ax in zip(["A", "B", "C", "D"], axes.flat):
        panel_label(ax, label, x=-0.16, y=1.05, fontsize=11)

    grouped.to_csv(ANALYSIS_DIR / "morphology_ie_regime_grouped.csv", index=False)
    df.to_csv(ANALYSIS_DIR / "morphology_ie_regime_runs.csv", index=False)
    fig.savefig(FIGURES_DIR / "fig_morphology_ie_regime.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_morphology_ie_regime.png", dpi=300, bbox_inches="tight")
    # Legacy aliases are kept so older drafts and slides do not break.
    fig.savefig(FIGURES_DIR / "fig_s8_morphology_ie_regime.pdf", bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig_s8_morphology_ie_regime.png", dpi=300, bbox_inches="tight")
    return fig, grouped


if __name__ == "__main__":
    build_figure()

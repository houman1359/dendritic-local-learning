#!/usr/bin/env python3
"""Generate a publication-ready morphology x inhibition regime-map figure."""

from __future__ import annotations

import argparse
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

from neurips_style import (  # noqa: E402
    COLORS,
    FIG_W,
    REF_LW,
    apply_neurips_style,
    clean_legend,
    grid_figure,
    panel_label,
    panel_title,
    style_axis,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = DRAFT_DIR / "figures"
ANALYSIS_DIR = DRAFT_DIR / "analysis" / "morphology_ie_regime"
DIAG_DIR = DRAFT_DIR / "analysis" / "morphology_ie_diag_subset_diagnostics_20260427"
_FDATA = FIGURES_DIR / "data"


def _tracked(name: str, fallback: Path) -> Path:
    """Prefer the git-tracked figures/data/ copy; fall back to local analysis/."""
    t = _FDATA / name
    return t if t.exists() else fallback


SELECTED_DIAG_CSV = _tracked(
    "morphology_ie_selected_runs.csv",
    DRAFT_DIR / "analysis" / "morphology_ie_diag_subset" / "selected_runs.csv",
)
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
        linewidths=0.65,
        linecolor="white",
        square=False,
        annot_kws={"fontsize": 6.4},
    )
    ax.collections[0].colorbar.outline.set_linewidth(0.65)
    ax.collections[0].colorbar.ax.tick_params(labelsize=8.3, width=0.4, length=2)
    ax.set_xlabel(r"$N_I$ per branch", fontsize=10.5)
    ax.set_ylabel("Tree", fontsize=10.5, labelpad=12)
    ax.tick_params(labelsize=8.6)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)


def _load_selected_diagnostics(sweep_dir: Path) -> pd.DataFrame | None:
    summary_csv = _tracked("morphology_ie_run_summary.csv", DIAG_DIR / "run_summary.csv")
    if not summary_csv.exists() or not SELECTED_DIAG_CSV.exists():
        return None
    diag = pd.read_csv(summary_csv)
    selected = pd.read_csv(SELECTED_DIAG_CSV)
    expected_root = str((sweep_dir / "results").resolve())
    if not selected["run_dir"].astype(str).map(
        lambda value: str(Path(value).resolve()).startswith(expected_root)
    ).all():
        return None
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
    _runs_csv = _FDATA / "morphology_ie_regime_runs.csv"
    df = pd.read_csv(_runs_csv) if _runs_csv.exists() else _load_results(sweep_dir)
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
    shunting_runs = df[df["network_type"] == "dendritic_shunting"]
    additive_runs = df[df["network_type"] == "dendritic_additive"]
    seed_gap = shunting_runs.merge(
        additive_runs,
        on=["branch_factors", "depth", "branch_product", "ie", "seed"],
        suffixes=("_s", "_a"),
    )
    seed_gap["gap"] = seed_gap["test_acc_s"] - seed_gap["test_acc_a"]
    depth_gap = (
        seed_gap.groupby(["depth", "ie"])["gap"]
        .agg(gap="mean", gap_std="std")
        .reset_index()
    )
    best_ie = (
        gap_long.sort_values(["branch_factors", "gap"], ascending=[True, False])
        .groupby("branch_factors")
        .head(1)
        .copy()
    )
    depth_peak = (
        depth_gap.sort_values(["depth", "gap"], ascending=[True, False])
        .groupby("depth")
        .head(1)
        .copy()
    )
    diag = _load_selected_diagnostics(sweep_dir)

    fig, axes = grid_figure(4, width_ratios=[1.12, 0.92, 1.02, 1.02])

    _heatmap(axes[0], gap, "Shunting advantage", "vlag", center=0.0)
    panel_title(axes[0], "A", "Accuracy gap")

    ax = axes[1]
    style_axis(ax, grid="x")
    best_ie = best_ie.set_index("branch_factors").reindex(branch_order).reset_index()
    y = np.arange(len(best_ie))
    colors = [COLORS["shunting"] if v > 0 else COLORS["additive"] for v in best_ie["gap"]]
    ax.barh(y, 100 * best_ie["gap"], color=colors, edgecolor="white",
            linewidth=0.65, height=0.56)
    ax.set_yticks(y)
    ax.set_yticklabels(best_ie["branch_factors"], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Peak gap (pp)")
    panel_title(ax, "B", "Peak gap")
    for yi, (_, row) in enumerate(best_ie.iterrows()):
        ax.text(
            100 * row["gap"] + 0.5,
            yi,
            f"$N_I$={int(row['ie'])}",
            va="center",
            ha="left",
            fontsize=8.6,
            color=COLORS["ink"],
        )
    ax.set_xlim(0, 100 * best_ie["gap"].max() + 9)

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
        tag_groups = [
            (
                "[4,4]\n0",
                "best_depth2_lowI_additive",
                "best_depth2_lowI_shunting",
            ),
            (
                "[3,3,3]\n5",
                "best_depth3_midI_additive",
                "best_depth3_midI_shunting",
            ),
            (
                "[3,3,3]\n20",
                "highI_match_additive",
                "highI_collapse_shunting",
            ),
        ]
        group_labels = []
        add_vals = []
        shunt_vals = []
        for label, add_tag, shunt_tag in tag_groups:
            add_row = keep[keep["tag"] == add_tag]
            shunt_row = keep[keep["tag"] == shunt_tag]
            if add_row.empty or shunt_row.empty:
                continue
            group_labels.append(label)
            add_vals.append(float(add_row.iloc[0]["path_gain_cv_mean"]))
            shunt_vals.append(float(shunt_row.iloc[0]["path_gain_cv_mean"]))
        x = np.arange(len(group_labels))
        width = 0.32
        ax.bar(
            x - width / 2,
            add_vals,
            width,
            color=COLOR_ADDITIVE,
            edgecolor="white",
            linewidth=0.65,
            label="Additive",
        )
        ax.bar(
            x + width / 2,
            shunt_vals,
            width,
            color=COLOR_SHUNTING,
            edgecolor="white",
            linewidth=0.65,
            label="Shunting",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=8.2)
        ax.set_xlabel(r"Tree / $N_I$")
        ax.set_ylabel("Path-gain CV")
        panel_title(ax, "C", "Credit geometry")
        ax.legend(loc="upper right", fontsize=8.4, frameon=False)
        ax.margins(x=0.08)
    else:
        architecture_summary = (
            df.groupby(["network_type", "ie"])["test_acc"]
            .agg(["mean", "std"])
            .reset_index()
        )
        for network_type, color, label, marker in (
            ("dendritic_additive", COLOR_ADDITIVE, "Additive", "s"),
            ("dendritic_shunting", COLOR_SHUNTING, "Shunting", "o"),
        ):
            subset = architecture_summary[
                architecture_summary["network_type"] == network_type
            ].sort_values("ie")
            ax.errorbar(
                subset["ie"],
                subset["mean"],
                yerr=subset["std"],
                color=color,
                marker=marker,
                linewidth=2.2,
                markersize=5.5,
                capsize=2.2,
                label=label,
            )
        ax.set_xlabel(r"$N_I$ per branch")
        ax.set_ylabel("Test accuracy")
        panel_title(ax, "C", "Mean perf.")
        ax.legend(fontsize=8.4, frameon=False, loc="best")

    ax = axes[3]
    style_axis(ax, grid="y")
    depth_palette = {2: "#7B5EA7", 3: "#D95F02"}
    for depth, sub in depth_gap.groupby("depth"):
        ax.errorbar(
            sub["ie"],
            sub["gap"],
            yerr=sub["gap_std"].fillna(0),
            marker="o",
            linewidth=2.4,
            markersize=6.5,
            color=depth_palette.get(depth, "#444444"),
            label=f"depth {depth}",
            capsize=2.4,
            capthick=0.8,
        )
    ax.axhline(0.0, color="black", linewidth=1.1, linestyle="--", alpha=0.6)
    panel_title(ax, "D", "Depth summary")
    ax.set_xlabel(r"$N_I$ per branch", fontsize=10.5)
    ax.set_ylabel("Accuracy gap", fontsize=10.5)
    ax.set_xticks(ie_order)
    ax.tick_params(labelsize=9.4)
    ax.legend(fontsize=9.2, loc="best")
    for _, row in depth_peak.iterrows():
        ax.scatter(
            [row["ie"]],
            [row["gap"]],
            s=52,
            facecolor="white",
            edgecolor=depth_palette.get(int(row["depth"]), "#444444"),
            linewidth=1.7,
            zorder=6,
        )

    grouped.to_csv(ANALYSIS_DIR / "morphology_ie_regime_grouped.csv", index=False)
    df.to_csv(ANALYSIS_DIR / "morphology_ie_regime_runs.csv", index=False)
    fig.subplots_adjust(left=0.055, right=0.988, bottom=0.31, top=0.80, wspace=0.58)
    fig.savefig(FIGURES_DIR / "fig_morphology_ie_regime.pdf")
    fig.savefig(FIGURES_DIR / "fig_morphology_ie_regime.png", dpi=300)
    return fig, grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    parser.add_argument(
        "--refresh-tracked-data",
        action="store_true",
        help="Rebuild figures/data/morphology_ie_regime_runs.csv from the sweep.",
    )
    args = parser.parse_args()
    if args.refresh_tracked_data:
        refreshed = _load_results(args.sweep_dir)
        _FDATA.mkdir(parents=True, exist_ok=True)
        refreshed.to_csv(_FDATA / "morphology_ie_regime_runs.csv", index=False)
    build_figure(args.sweep_dir)

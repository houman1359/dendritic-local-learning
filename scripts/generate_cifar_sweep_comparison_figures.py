#!/usr/bin/env python
"""Generate compact comparison figures for the new CIFAR sweeps.

These figures are intentionally endpoint-focused: seed dots plus mean/SD
intervals, not epoch traces. They are meant to support paper decisions about
which CIFAR and routed-task results deserve main-text emphasis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurips_style import COLORS, apply_neurips_style, panel_label, style_axis


REPO_ROOT = Path(__file__).resolve().parents[3]
DRAFT_ROOT = REPO_ROOT / "drafts" / "dendritic-local-learning"
ANALYSIS = DRAFT_ROOT / "analysis"
FIGURES = DRAFT_ROOT / "figures"


CIFAR_LABELS = {
    "cifar10_additive_standard": "Additive\nBP",
    "cifar10_shunting_standard_learned_i": "Shunting\nBP",
    "cifar10_shunting_standard_no_i": "Shunting BP\nno I->E",
    "cifar10_additive_5f_per_soma": "Add.\nPS",
    "cifar10_shunting_5f_per_soma_learned_i": "Shunt.\nPS",
    "cifar10_shunting_5f_per_soma_no_i": "Shunting per-soma\nno I->E",
    "cifar10_additive_5f_path_transport": "Add.\nPT",
    "cifar10_shunting_5f_low_rank4_learned_i": "Shunt.\nK4",
    "cifar10_shunting_5f_path_transport_learned_i": "Shunt.\nPT",
    "cifar10_shunting_5f_path_transport_no_i": "Shunting transport\nno I->E",
    "cifar10_additive_5f_per_soma_gain_input_dependent": "Input\n gain",
    "cifar10_additive_5f_per_soma_gain_running_stats": "Run-stat\n gain",
    "cifar10_additive_5f_per_soma_gain_learned": "Learned\n gain",
    "cifar10_additive_5f_per_soma_dendritic_norm": "Dend.\n norm",
}

DOUBLE_LABELS = {
    "double_cifar_additive_standard": "Additive\nBP",
    "double_cifar_shunting_standard": "Shunting\nBP",
    "double_cifar_additive_5f_per_soma": "Additive\nper-soma",
    "double_cifar_shunting_5f_per_soma": "Shunting\nper-soma",
    "double_cifar_additive_5f_low_rank2": "Additive\nK2",
    "double_cifar_shunting_5f_low_rank2": "Shunting\nK2",
    "double_cifar_additive_5f_path_transport": "Additive\ntransport",
    "double_cifar_shunting_5f_path_transport": "Shunting\ntransport",
    "double_cifar_shunting_5f_pathway_vector": "Shunting\npathway",
}


def _latest_summary(prefix: str) -> Path:
    matches = sorted(ANALYSIS.glob(f"{prefix}_*_summary_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No analysis summary matching {prefix!r} under {ANALYSIS}")
    return matches[-1]


def _load_detailed(summary_dir: Path) -> pd.DataFrame:
    path = summary_dir / "detailed_results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "complete"].copy()
    if "test_accuracy" in df.columns:
        df = df.dropna(subset=["test_accuracy"])
    return df


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    print(f"Saved {name}.{{pdf,png}}")


def _condition_color(condition: str) -> str:
    if "path_transport" in condition:
        return COLORS["oracle"]
    if "low_rank" in condition:
        return COLORS["low_rank"]
    if "pathway" in condition:
        return COLORS["pathway"]
    if "shunting" in condition:
        return COLORS["shunting"]
    if "additive" in condition:
        return COLORS["additive"]
    return COLORS["edge"]


def _dot_interval_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    conditions: list[str],
    labels: dict[str, str],
    title: str,
    ylabel: str = "Test accuracy (%)",
) -> None:
    xs = np.arange(len(conditions))
    rng = np.random.default_rng(7)
    missing_xs: list[int] = []
    for x, condition in zip(xs, conditions):
        values = 100.0 * df.loc[df["condition"] == condition, "test_accuracy"].dropna().to_numpy()
        color = _condition_color(condition)
        if values.size == 0:
            missing_xs.append(int(x))
            continue
        jitter = rng.normal(0, 0.035, size=values.size)
        ax.scatter(
            np.full(values.size, x) + jitter,
            values,
            s=14,
            color=color,
            edgecolor="white",
            linewidth=0.4,
            alpha=0.8,
            zorder=3,
        )
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        ax.errorbar(
            [x],
            [mean],
            yerr=[[sd], [sd]],
            fmt="o",
            ms=4.0,
            color=COLORS["ink"],
            ecolor=COLORS["ink"],
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [labels.get(c, c.replace("_", "\n")) for c in conditions],
        fontsize=7.2,
        rotation=28,
        ha="right",
        rotation_mode="anchor",
        linespacing=0.9,
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    style_axis(ax, grid="y")
    ax.set_ylim(bottom=max(0, ax.get_ylim()[0]))
    if missing_xs:
        y0, y1 = ax.get_ylim()
        y = y0 + 0.5 * (y1 - y0)
        for x in missing_xs:
            ax.text(
                x,
                y,
                "no result",
                ha="center",
                va="center",
                rotation=90,
                fontsize=6,
                color=COLORS["mute"],
            )


def generate_cifar_figure(summary_dir: Path) -> bool:
    df = _load_detailed(summary_dir)
    if df.empty:
        print(f"No completed CIFAR rows in {summary_dir}; skipping figure.")
        return False

    apply_neurips_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15))

    _dot_interval_panel(
        axes[0],
        df,
        [
            "cifar10_additive_standard",
            "cifar10_shunting_standard_learned_i",
            "cifar10_shunting_standard_no_i",
        ],
        CIFAR_LABELS,
        "Backprop references",
    )
    _dot_interval_panel(
        axes[1],
        df,
        [
            "cifar10_additive_5f_per_soma",
            "cifar10_shunting_5f_per_soma_learned_i",
            "cifar10_shunting_5f_low_rank4_learned_i",
            "cifar10_additive_5f_path_transport",
            "cifar10_shunting_5f_path_transport_learned_i",
        ],
        CIFAR_LABELS,
        "Broadcast ladder",
        ylabel="",
    )
    _dot_interval_panel(
        axes[2],
        df,
        [
            "cifar10_additive_5f_per_soma",
            "cifar10_additive_5f_per_soma_gain_input_dependent",
            "cifar10_additive_5f_per_soma_gain_running_stats",
            "cifar10_additive_5f_per_soma_gain_learned",
            "cifar10_additive_5f_per_soma_dendritic_norm",
        ],
        CIFAR_LABELS,
        "Additive controls",
        ylabel="",
    )

    for label, ax in zip("ABC", axes):
        panel_label(ax, label, x=0.01, y=0.98, fontsize=10)

    fig.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.42, wspace=0.40)
    _save(fig, "fig_s_cifar10_control_ladder_20260427")
    plt.close(fig)
    return True


def generate_double_cifar_figure(summary_dir: Path) -> bool:
    df = _load_detailed(summary_dir)
    if df.empty:
        print(f"No completed Double-CIFAR rows in {summary_dir}; skipping figure.")
        return False

    apply_neurips_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15))

    _dot_interval_panel(
        axes[0],
        df,
        ["double_cifar_additive_standard", "double_cifar_shunting_standard"],
        DOUBLE_LABELS,
        "Routed CIFAR ceiling",
    )
    _dot_interval_panel(
        axes[1],
        df,
        [
            "double_cifar_additive_5f_per_soma",
            "double_cifar_shunting_5f_per_soma",
            "double_cifar_additive_5f_low_rank2",
            "double_cifar_shunting_5f_low_rank2",
        ],
        DOUBLE_LABELS,
        "Per-soma vs K2 broadcast",
        ylabel="",
    )
    _dot_interval_panel(
        axes[2],
        df,
        [
            "double_cifar_additive_5f_path_transport",
            "double_cifar_shunting_5f_path_transport",
            "double_cifar_shunting_5f_pathway_vector",
        ],
        DOUBLE_LABELS,
        "Transport and pathway feedback",
        ylabel="",
    )

    for label, ax in zip("ABC", axes):
        panel_label(ax, label, x=-0.14, y=1.08, fontsize=13)

    fig.subplots_adjust(left=0.07, right=0.995, top=0.80, bottom=0.33, wspace=0.36)
    _save(fig, "fig_s_double_cifar_routed_screen_20260427")
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cifar-summary", type=Path, default=None)
    parser.add_argument("--double-cifar-summary", type=Path, default=None)
    args = parser.parse_args()

    cifar_summary = args.cifar_summary or _latest_summary("cifar10_control_ladder_20260427")
    double_summary = args.double_cifar_summary or _latest_summary("double_cifar_routed_screen_20260427")
    generate_cifar_figure(cifar_summary)
    generate_double_cifar_figure(double_summary)


if __name__ == "__main__":
    main()

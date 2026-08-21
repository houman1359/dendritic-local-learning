#!/usr/bin/env python3
"""Generate the pathway-routing and inhibition-control appendix figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurips_style import COLORS, apply_neurips_style, panel_label, style_axis


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_ROOT = SCRIPT_DIR.parent
ANALYSIS = DRAFT_ROOT / "analysis"
FIGURES = DRAFT_ROOT / "figures"

CUE_SUMMARY = (
    ANALYSIS
    / "cue_routing_pv_inhibition_controls_20260428_20260428010311_summary_20260428"
    / "detailed_results.csv"
)
CUE_PATHWAY = (
    ANALYSIS
    / "cue_routing_pv_inhibition_controls_20260428_20260428010311_pathway_learning_20260428"
    / "pathway_learning_grouped.csv"
)
DOUBLE_SUMMARY = (
    ANALYSIS
    / "double_cifar_routed_nonnegative_pv_20260428_20260428010311_summary_20260428"
    / "detailed_results.csv"
)

INHIBITION_DIAG_DIRS = {
    "rank-1": ANALYSIS / "cue_positive_rank1_learned_i_inhibitory_path_gain_20260428",
    "rank-2": ANALYSIS / "cue_positive_low_rank2_learned_i_inhibitory_path_gain_20260428",
    "pathway": ANALYSIS / "cue_positive_pathway_vector_learned_i_inhibitory_path_gain_20260428",
    "transport": ANALYSIS / "cue_positive_path_transport_learned_i_inhibitory_path_gain_20260428",
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _condition_values(df: pd.DataFrame, condition: str, metric: str = "test_accuracy") -> np.ndarray:
    values = df.loc[df["condition"] == condition, metric].dropna().to_numpy(dtype=float)
    if values.size == 0:
        raise KeyError(f"No values for {condition!r} in {metric!r}")
    return values


def _dot_interval(
    ax: plt.Axes,
    x: float,
    values: np.ndarray,
    color: str,
    label: str | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    if rng is None:
        rng = np.random.default_rng(1)
    jitter = rng.normal(0.0, 0.025, size=values.size)
    ax.scatter(
        np.full(values.size, x) + jitter,
        values,
        s=16,
        color=color,
        edgecolor="white",
        linewidth=0.45,
        alpha=0.88,
        zorder=3,
        label=label,
    )
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if values.size > 1 else 0.0
    ax.errorbar(
        [x],
        [mean],
        yerr=[[sd], [sd]],
        fmt="o",
        color=COLORS["ink"],
        ecolor=COLORS["ink"],
        ms=4.3,
        elinewidth=1.05,
        capsize=2.5,
        zorder=4,
    )


def _load_inhibition_diagnostics() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for label, directory in INHIBITION_DIAG_DIRS.items():
        path = directory / "inhibitory_path_gain_grouped_diagnostics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        data = json.loads(path.read_text())
        rows.append(
            {
                "mode": label,
                "context_decode": 100.0
                * float(data["context_from_inhibition_best_polarity_accuracy_mean"]),
                "context_decode_sd": 100.0
                * float(data["context_from_inhibition_best_polarity_accuracy_std"]),
                "mi": float(data["context_balance_mi_bits_mean"]),
                "mi_sd": float(data["context_balance_mi_bits_std"]),
                "resistance_coupling": -float(data["inhibition_vs_local_resistance_r_mean"]),
                "resistance_coupling_sd": float(data["inhibition_vs_local_resistance_r_std"]),
            }
        )
    return pd.DataFrame(rows)


def plot_cue_accuracy(ax: plt.Axes, cue: pd.DataFrame) -> None:
    modes = [
        ("standard", "BP", "cue_positive_shunting_standard"),
        ("rank1", "rank-1", "cue_positive_shunting_rank1"),
        ("low_rank2", "rank-2", "cue_positive_shunting_low_rank2"),
        ("pathway_vector", "PV", "cue_positive_shunting_pathway_vector"),
        ("path_transport", "trans.", "cue_positive_shunting_path_transport"),
    ]
    rng = np.random.default_rng(3)
    xs = np.arange(len(modes), dtype=float)
    for idx, (_key, _label, stem) in enumerate(modes):
        learned = 100.0 * _condition_values(cue, f"{stem}_learned_i")
        no_i = 100.0 * _condition_values(cue, f"{stem}_no_i")
        _dot_interval(
            ax,
            xs[idx] - 0.13,
            learned,
            COLORS["shunting"],
            "learned I" if idx == 0 else None,
            rng,
        )
        _dot_interval(
            ax,
            xs[idx] + 0.13,
            no_i,
            COLORS["mute"],
            "no I-to-E" if idx == 0 else None,
            rng,
        )
    ax.axhline(99.0, color=COLORS["grid"], linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([item[1] for item in modes])
    ax.set_ylim(98.3, 100.0)
    ax.set_ylabel("Cue accuracy (%)", fontsize=8.8)
    ax.set_title("Cue control saturates", fontsize=9.2)
    ax.legend(loc="lower right", frameon=False, ncol=1, fontsize=7.5)
    ax.tick_params(labelsize=8.0)
    style_axis(ax, grid="y")


def plot_pathway_purity(ax: plt.Axes, pathway: pd.DataFrame) -> None:
    modes = [
        ("cue_positive_shunting_rank1_learned_i", "rank-1"),
        ("cue_positive_shunting_low_rank2_learned_i", "rank-2"),
        ("cue_positive_shunting_pathway_vector_learned_i", "pathway"),
        ("cue_positive_shunting_path_transport_learned_i", "transport"),
    ]
    xs = np.arange(len(modes), dtype=float)
    width = 0.32
    e_vals = []
    i_vals = []
    for condition, _label in modes:
        row = pathway.loc[pathway["condition"] == condition].iloc[0]
        e_vals.append(100.0 * float(row["e_pathway_branch_purity_mean"]))
        i_vals.append(100.0 * float(row["i_pathway_branch_purity_mean"]))
    ax.bar(xs - width / 2, e_vals, width, color=COLORS["exc"], label="E branch purity")
    ax.bar(xs + width / 2, i_vals, width, color=COLORS["inh"], label="I branch purity")
    ax.axhline(50.0, color=COLORS["grid"], linewidth=1.0, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _condition, label in modes])
    ax.set_ylabel("Branch purity (%)", fontsize=8.8)
    ax.set_ylim(50, 76)
    ax.set_title("Branch pathway purity", fontsize=9.2)
    ax.legend(loc="upper left", frameon=False, fontsize=7.5)
    ax.tick_params(labelsize=8.0)
    style_axis(ax, grid="y")


def plot_inhibition_diagnostics(ax: plt.Axes, diag: pd.DataFrame) -> None:
    labels = diag["mode"].tolist()
    xs = np.arange(len(labels), dtype=float)
    width = 0.25
    ax.bar(
        xs - width,
        diag["context_decode"],
        width,
        yerr=diag["context_decode_sd"],
        color=COLORS["pathway"],
        capsize=2.5,
        label="context decode (%)",
    )
    ax.bar(
        xs,
        100.0 * diag["mi"],
        width,
        yerr=100.0 * diag["mi_sd"],
        color=COLORS["low_rank"],
        capsize=2.5,
        label="MI (bits x 100)",
    )
    ax.bar(
        xs + width,
        100.0 * diag["resistance_coupling"],
        width,
        yerr=100.0 * diag["resistance_coupling_sd"],
        color=COLORS["inh"],
        capsize=2.5,
        label="-corr(I, R)",
    )
    ax.axhline(50.0, color=COLORS["grid"], linewidth=1.0, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Diagnostic value", fontsize=8.8)
    ax.set_title("Inhibitory context signal", fontsize=9.2)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=3,
        frameon=False,
        fontsize=7.3,
        handlelength=1.2,
        columnspacing=0.8,
    )
    ax.tick_params(labelsize=8.0)
    style_axis(ax, grid="y")


def plot_double_cifar_screen(ax: plt.Axes, double: pd.DataFrame) -> None:
    conditions = [
        ("double_cifar_positive_additive_standard", "Add.\nBP"),
        ("double_cifar_positive_shunting_standard_learned_i", "Sh.\nBP"),
        ("double_cifar_positive_shunting_5f_rank1_learned_i", "rank-1"),
        ("double_cifar_positive_shunting_5f_low_rank2_learned_i", "rank-2"),
        ("double_cifar_positive_shunting_5f_pathway_vector_learned_i", "PV"),
        ("double_cifar_positive_shunting_5f_path_transport_learned_i", "trans."),
    ]
    rng = np.random.default_rng(5)
    xs = np.arange(len(conditions), dtype=float)
    for idx, (condition, _label) in enumerate(conditions):
        values = 100.0 * _condition_values(double, condition)
        color = COLORS["additive"] if "additive" in condition else COLORS["shunting"]
        if "path_transport" in condition:
            color = COLORS["oracle"]
        if "pathway" in condition:
            color = COLORS["pathway"]
        if "low_rank" in condition:
            color = COLORS["low_rank"]
        _dot_interval(ax, xs[idx], values, color, rng=rng)
    ax.axhline(10.0, color=COLORS["edge"], linewidth=1.0, linestyle="--", label="chance")
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _condition, label in conditions])
    ax.set_ylim(8.0, 13.2)
    ax.set_ylabel("Test accuracy (%)", fontsize=8.8)
    ax.set_title("Routed CIFAR screen", fontsize=9.2)
    ax.legend(loc="upper right", frameon=False, fontsize=7.5)
    ax.tick_params(labelsize=8.0)
    style_axis(ax, grid="y")


def main() -> None:
    apply_neurips_style()
    cue = _read_csv(CUE_SUMMARY)
    pathway = _read_csv(CUE_PATHWAY)
    double = _read_csv(DOUBLE_SUMMARY)
    diag = _load_inhibition_diagnostics()

    fig, axes = plt.subplots(1, 4, figsize=(14.8, 3.25))
    plot_cue_accuracy(axes[0], cue)
    plot_pathway_purity(axes[1], pathway)
    plot_inhibition_diagnostics(axes[2], diag)
    plot_double_cifar_screen(axes[3], double)

    for label, ax in zip("ABCD", axes):
        panel_label(ax, label, x=-0.15, y=1.08, fontsize=11.5)

    fig.subplots_adjust(left=0.055, right=0.995, top=0.82, bottom=0.33, wspace=0.48)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_s_pathway_inhibition_controls_20260428.{ext}")
    print("Saved fig_s_pathway_inhibition_controls_20260428.{pdf,png}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the expanded regular-tree regime figure.

The panels recover the validated task, inhibition, stress, rule, feedback, and
CIFAR-10 controls used in the arXiv/NeurIPS manuscript.  Plotting uses the
byte-identical ``neurips_style.py`` visual system shared by every journal
figure.  No panel is used to claim a universal shunting advantage.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from neurips_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    PT_SMALL,
    PT_TICK,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "regular_tree_regimes"
FIGURES = ROOT / "figures" / "generated"

apply_neurips_style()

SHUNT = COLORS["shunting"]
ADD = COLORS["additive"]
BP = COLORS["bp"]
ORACLE = COLORS["oracle"]
RULE3 = COLORS["rule_3f"]
RULE4 = COLORS["rule_4f"]
RULE5 = COLORS["rule_5f"]


def save(fig: plt.Figure) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    layout = audit_layout(fig, "fig3_regular_tree_regimes")
    overlap = audit_text_over_data(fig, "fig3_regular_tree_regimes")
    if layout or overlap:
        print(
            "  review fig3_regular_tree_regimes: "
            f"{len(layout)} layout, {len(overlap)} text/data warnings"
        )
    fig.savefig(
        FIGURES / "fig3_regular_tree_regimes.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig3_regular_tree_regimes.png", dpi=350)
    plt.close(fig)


def mean_std(frame: pd.DataFrame, column: str) -> tuple[float, float]:
    values = frame[column].to_numpy(float)
    return float(values.mean()), float(values.std(ddof=1))


def panel_tasks(ax: plt.Axes) -> None:
    frame = pd.read_csv(DATA / "competence_summary.csv")
    datasets = [
        ("mnist", "MN"),
        ("fashion_mnist", "FMN"),
        ("context_gating", "FG"),
    ]
    specs = [
        ("dendritic_shunting", "standard", "shunt. BP", BP, -0.22),
        ("dendritic_shunting", "local_ca", "shunting", SHUNT, 0.00),
        ("dendritic_additive", "local_ca", "additive", ADD, 0.22),
    ]
    x = np.arange(len(datasets), dtype=float)
    for network, strategy, label, color, offset in specs:
        means, errors = [], []
        for dataset, _ in datasets:
            row = frame[
                frame["dataset"].eq(dataset)
                & frame["network_type"].eq(network)
                & frame["strategy"].eq(strategy)
            ]
            means.append(float(row.iloc[0]["test_accuracy_mean"]))
            errors.append(float(row.iloc[0]["test_accuracy_std"]))
        ax.bar(
            x + offset,
            means,
            0.20,
            yerr=errors,
            color=color,
            edgecolor="white",
            linewidth=LW_HAIR,
            capsize=2.0,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in datasets])
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
    ax.set_ylabel("test accuracy")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax, "A", "Classification tasks")
    style_axis(ax, grid="y")
    clean_legend(
        ax,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        fontsize=PT_SMALL,
        handlelength=0.8,
        handletextpad=0.25,
        columnspacing=0.42,
    )


def panel_inhibition(ax: plt.Axes) -> None:
    frame = pd.read_csv(DATA / "inhibition_dose_summary.csv")
    specs = [
        ("dendritic_shunting", "mnist", SHUNT, "-", "o"),
        ("dendritic_additive", "mnist", ADD, "-", "s"),
        ("dendritic_shunting", "noise_resilience", SHUNT, "--", "^"),
        ("dendritic_additive", "noise_resilience", ADD, "--", "v"),
    ]
    for network, dataset, color, linestyle, marker in specs:
        sub = frame[
            frame["network_type"].eq(network) & frame["dataset"].eq(dataset)
        ].sort_values("ie_value")
        ax.errorbar(
            sub["ie_value"],
            sub["test_accuracy_mean"],
            yerr=sub["test_accuracy_std"],
            color=color,
            ls=linestyle,
            marker=marker,
            markersize=3.8,
            lw=LW_DATA,
            capsize=1.8,
        )
    ax.set_xlabel("inhibitory synapses / branch")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.25, 1.00)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax, "B", "Inhibitory dose")
    style_axis(ax, grid="y")
    clean_legend(
        ax,
        handles=[
            Line2D([0], [0], color=SHUNT, marker="o", label="shunting"),
            Line2D([0], [0], color=ADD, marker="s", label="additive"),
            Line2D([0], [0], color=COLORS["mute"], ls="-", label="MNIST"),
            Line2D([0], [0], color=COLORS["mute"], ls="--", label="noise task"),
        ],
        loc="lower right",
        ncol=2,
        fontsize=PT_SMALL,
        handlelength=1.1,
        handletextpad=0.3,
        columnspacing=0.5,
    )


def panel_depth(ax: plt.Axes) -> None:
    frame = pd.read_csv(DATA / "depth_scaling_summary.csv")
    frame["depth"] = frame["branch_factors"].astype(str).map(
        lambda value: len(value.strip("[]").split(","))
    )
    for strategy, linestyle, alpha in [("local_ca", "-", 1.0), ("standard", "--", 0.32)]:
        for network, color, label in [
            ("dendritic_shunting", SHUNT, "shunting"),
            ("dendritic_additive", ADD, "additive"),
        ]:
            sub = frame[
                frame["strategy"].eq(strategy) & frame["network_type"].eq(network)
            ].sort_values("depth")
            ax.errorbar(
                sub["depth"],
                sub["test_accuracy_mean"],
                yerr=sub["test_accuracy_std"],
                color=color,
                ls=linestyle,
                marker="o",
                markersize=3.6,
                lw=LW_DATA,
                capsize=1.8,
                alpha=alpha,
                label=f"{label} {'local' if strategy == 'local_ca' else 'BP'}",
            )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("dendritic layers")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.15, 0.98)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax, "C", "Depth stress")
    style_axis(ax, grid="y")


def panel_noise(ax: plt.Axes) -> None:
    frame = pd.read_csv(DATA / "broadcast_noise_summary.csv")
    for network, color, label in [
        ("dendritic_shunting", SHUNT, "shunting"),
        ("dendritic_additive", ADD, "additive"),
    ]:
        sub = frame[frame["network_type"].eq(network)].sort_values("error_noise_sigma")
        ax.errorbar(
            sub["error_noise_sigma"],
            sub["test_accuracy_mean"],
            yerr=sub["test_accuracy_std"],
            color=color,
            marker="o",
            markersize=3.6,
            lw=LW_DATA,
            capsize=1.8,
            label=label,
        )
    ax.set_xlabel(r"broadcast-noise $\sigma$")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0.05, 0.72)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax, "D", "Noisy teaching signal")
    style_axis(ax, grid="y")
    clean_legend(ax, loc="lower left", fontsize=PT_SMALL, handlelength=1.0)


def panel_rules(ax: plt.Axes) -> None:
    frame = pd.read_csv(DATA / "rule_family_summary_source.csv")
    sub = frame[
        frame["dataset"].eq("mnist")
        & frame["network_type"].eq("dendritic_shunting")
        & frame["error_broadcast_mode"].eq("per_soma")
        & frame["decoder_update_mode"].eq("local")
        & frame["rule_variant"].isin(["3f", "4f", "5f"])
    ]
    rows = []
    for rule in ["3f", "4f", "5f"]:
        cell = sub[sub["rule_variant"].eq(rule)]
        means = cell["test_accuracy_mean"].to_numpy(float)
        stds = cell["test_accuracy_std"].to_numpy(float)
        if not np.allclose(means, means[0]) or not np.allclose(stds, stds[0]):
            raise ValueError(f"inert base-rate duplicates disagree for {rule}")
        rows.append((means[0], stds[0]))
    x = np.arange(3)
    ax.bar(
        x,
        [row[0] for row in rows],
        yerr=[row[1] for row in rows],
        color=[RULE3, RULE4, RULE5],
        edgecolor="white",
        linewidth=LW_HAIR,
        capsize=2.0,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["3F", "4F", "5F"])
    ax.set_ylabel("MNIST accuracy")
    ax.set_ylim(0.84, 0.94)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax, "E", "Rule family")
    style_axis(ax, grid="y")


def panel_error_source(ax: plt.Axes) -> None:
    frame = pd.read_csv(DATA / "error_source_runs.csv")
    groups = [("per_soma", "soma error"), ("local_mismatch", "local mismatch")]
    x = np.arange(2, dtype=float)
    width = 0.30
    for index, (network, color, label) in enumerate([
        ("dendritic_shunting", SHUNT, "shunting"),
        ("dendritic_additive", ADD, "additive"),
    ]):
        means, stds = [], []
        for mode, _ in groups:
            mean, std = mean_std(
                frame[
                    frame["network_type"].eq(network)
                    & frame["error_broadcast_mode"].eq(mode)
                ],
                "test_accuracy",
            )
            means.append(mean)
            stds.append(std)
        ax.bar(
            x + (index - 0.5) * width,
            means,
            width * 0.92,
            yerr=stds,
            color=color,
            edgecolor="white",
            linewidth=LW_HAIR,
            capsize=2.0,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["soma error", "mismatch"])
    ax.set_ylabel("MNIST accuracy")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax, "F", "Error source")
    style_axis(ax, grid="y")
    clean_legend(ax, loc="upper right", fontsize=PT_SMALL, handlelength=0.8)


def panel_controls(ax: plt.Axes) -> None:
    exact = pd.read_csv(DATA / "exact_transport_summary.csv")
    bp = pd.read_csv(DATA / "backprop_reference_summary.csv").iloc[0]
    additive = pd.read_csv(DATA / "additive_controls_summary.csv")
    react = pd.read_csv(DATA / "reactivation_controls_summary.csv")
    transport = exact[
        exact["rule_variant"].eq("5f") & exact["decoder_update_mode"].eq("local")
    ].iloc[0]
    identity = react[react["reactivation_enabled"].eq(False)].iloc[0]
    tanh = react[react["reactivation_enabled"].eq(True)].iloc[0]
    add_none = additive[
        additive["core"].eq("additive") & additive["additive_gain_mode"].eq("none")
    ].iloc[0]
    gain = additive[
        additive["core"].eq("additive") & additive["additive_gain_mode"].ne("none")
    ].sort_values("test_acc_mean", ascending=False).iloc[0]
    norm = additive[additive["core"].eq("normalized_additive")].sort_values(
        "test_acc_mean", ascending=False
    ).iloc[0]
    rows = [
        ("backprop", bp, BP),
        ("transport", transport, ORACLE),
        ("identity", identity, "#72B795"),
        ("tanh", tanh, SHUNT),
        ("additive", add_none, ADD),
        ("add. gain", gain, "#6F88C6"),
        ("add. norm", norm, "#5B8AC4"),
    ]
    y = np.arange(len(rows), dtype=float)
    values = [float(row[1]["test_acc_mean"]) for row in rows]
    errors = [float(row[1]["test_acc_std"]) for row in rows]
    ax.barh(
        y,
        values,
        xerr=errors,
        color=[row[2] for row in rows],
        edgecolor="white",
        linewidth=LW_HAIR,
        capsize=1.8,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([row[0] for row in rows], fontsize=PT_TICK)
    ax.invert_yaxis()
    ax.set_xlim(0.86, 0.98)
    ax.set_xlabel("MNIST accuracy")
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax, "G", "Mechanism controls")
    style_axis(ax, grid="x")


def panel_noise_feedback(ax: plt.Axes) -> None:
    frame = pd.read_csv(DATA / "noise_feedback_ladder_summary.csv")
    specs = [
        ("per_soma", 4, False, "scalar", SHUNT),
        ("per_soma", 4, True, "path", "#4DAF4A"),
        ("low_rank", 1, False, "rank 1", "#F6B94A"),
        ("low_rank", 2, False, "rank 2", "#F39C12"),
        ("low_rank", 4, False, "rank 4", "#E67E22"),
        ("low_rank", 8, False, "rank 8", "#D35400"),
        ("path_transport", 4, False, "exact", ORACLE),
    ]
    values, errors, labels, colors = [], [], [], []
    for mode, rank, propagation, label, color in specs:
        row = frame[
            frame["broadcast_mode"].eq(mode)
            & frame["broadcast_rank"].eq(rank)
            & frame["use_path_propagation"].eq(propagation)
        ].iloc[0]
        values.append(float(row["test_accuracy_mean"]))
        errors.append(float(row["test_accuracy_std"]))
        labels.append(label)
        colors.append(color)
    x = np.arange(len(values))
    ax.bar(
        x,
        values,
        yerr=errors,
        color=colors,
        edgecolor="white",
        linewidth=LW_HAIR,
        capsize=1.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=32, ha="right")
    ax.set_ylim(0.35, 0.90)
    ax.set_ylabel("noise-task accuracy")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax, "H", "Feedback bandwidth")
    style_axis(ax, grid="y")


def panel_cifar(ax: plt.Axes) -> None:
    frame = pd.read_csv(DATA / "cifar10_control_ladder_runs.csv")
    specs = [
        ("cifar10_shunting_5f_per_soma_learned_i", "scalar", SHUNT),
        ("cifar10_shunting_5f_low_rank4_learned_i", "rank 4", "#E67E22"),
        ("cifar10_shunting_5f_path_transport_learned_i", "exact", ORACLE),
        ("cifar10_shunting_standard_learned_i", "backprop", BP),
    ]
    x = np.arange(len(specs))
    for index, (condition, label, color) in enumerate(specs):
        values = frame[frame["condition"].eq(condition)]["test_accuracy"].to_numpy(float)
        mean = values.mean()
        std = values.std(ddof=1)
        ax.bar(
            index,
            mean,
            yerr=std,
            color=color,
            edgecolor="white",
            linewidth=LW_HAIR,
            capsize=1.8,
        )
        ax.scatter(
            np.full(values.size, index) + np.linspace(-0.08, 0.08, values.size),
            values,
            s=7,
            facecolor="white",
            edgecolor=color,
            linewidth=0.5,
            zorder=4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label, _ in specs], rotation=28, ha="right")
    ax.set_ylim(0, 0.56)
    ax.set_ylabel("CIFAR-10 accuracy")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax, "I", "Harder-data control")
    style_axis(ax, grid="y")


def main() -> None:
    # Compact the vertical panel boxes rather than shrinking the complete
    # figure in LaTeX; this preserves the canonical paper-wide type scale and
    # keeps the figure-plus-caption within one page.
    fig = plt.figure(figsize=(FIG_W, 6.65))
    gs = fig.add_gridspec(
        3,
        3,
        left=0.135,
        right=0.985,
        bottom=0.075,
        top=0.945,
        wspace=0.66,
        hspace=0.96,
    )
    axes = [fig.add_subplot(gs[row, column]) for row in range(3) for column in range(3)]
    panel_tasks(axes[0])
    panel_inhibition(axes[1])
    panel_depth(axes[2])
    panel_noise(axes[3])
    panel_rules(axes[4])
    panel_error_source(axes[5])
    panel_controls(axes[6])
    panel_noise_feedback(axes[7])
    panel_cifar(axes[8])
    save(fig)


if __name__ == "__main__":
    main()

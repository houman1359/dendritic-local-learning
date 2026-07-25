#!/usr/bin/env python3
"""Generate the supplementary fixed-state and learning-step audit figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neurips_style import (
    apply_neurips_style,
    COLORS,
    MAIN_W,
    panel_label,
    style_axis,
)


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = DRAFT_DIR / "figures"
DATA_DIR = FIGURES_DIR / "data"
FIXED_STATE_CSV = (
    DATA_DIR / "fixed_state_followups" / "fixed_state_factorial_seed.csv"
)
IDENTITY_CSV = (
    DATA_DIR / "identity_transfer_replication" / "identity_transfer_seed.csv"
)
FEEDBACK_RELEVANCE_CSV = DATA_DIR / "feedback_learning_relevance_runs.csv"

COLOR_SHUNTING = COLORS["shunting"]
COLOR_ADDITIVE = COLORS["additive"]


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(
            FIGURES_DIR / f"{name}.{extension}",
            dpi=350,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    print(f"Saved {name}.{{pdf,png}}")


def _plot_fixed_state(ax: plt.Axes, frame: pd.DataFrame) -> None:
    selected = [
        ("exact_full", "Exact reconstruction", COLORS["oracle"]),
        (
            "transport_R_no_inhibition",
            "No-I backward $R$",
            "#70B68F",
        ),
        (
            "eligibility_R_no_inhibition",
            "No-I eligibility $R$",
            "#4C9B70",
        ),
        (
            "synaptic_voltage_proxy",
            "Voltage proxy",
            COLORS["additive"],
        ),
        (
            "transport_no_parent_derivative",
            "No parent derivative",
            COLORS["local"],
        ),
        (
            "submitted_full",
            "Submitted feedback",
            COLORS["scalar"],
        ),
    ]
    y = np.arange(len(selected), dtype=float)
    means, sems, labels, colors = [], [], [], []
    rng = np.random.default_rng(17)
    for row_index, (condition, label, color) in enumerate(selected):
        values = frame.loc[
            frame["condition"] == condition,
            "branch_numel_weighted_cosine",
        ].to_numpy(dtype=float)
        means.append(float(np.mean(values)))
        sems.append(float(np.std(values, ddof=1) / np.sqrt(len(values))))
        labels.append(label)
        colors.append(color)
        jitter = rng.uniform(-0.10, 0.10, size=len(values))
        ax.scatter(
            values,
            np.full(len(values), row_index) + jitter,
            s=11,
            facecolor="white",
            edgecolor=color,
            linewidth=0.65,
            zorder=4,
        )
    ax.barh(
        y,
        means,
        xerr=sems,
        height=0.57,
        color=colors,
        edgecolor="white",
        linewidth=0.7,
        capsize=2.0,
        error_kw={"lw": 0.85},
        zorder=2,
    )
    ax.axvline(1.0, color=COLORS["mute"], lw=0.8, ls="--", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.7)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.05)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_xlabel("Cosine to exact gradient", fontsize=9.4)
    ax.set_title("Fixed-state factor audit", fontsize=10.2)
    style_axis(ax, grid="x")
    panel_label(ax, "A", dx=-26, dy=5)


def _plot_identity_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    metric: str,
    ylabel: str,
    scale: float,
    show_xlabels: bool,
    title: str | None = None,
) -> None:
    pivot = frame.pivot(
        index="seed",
        columns="network_type",
        values=metric,
    ).dropna()
    additive = pivot["dendritic_additive"].to_numpy(dtype=float) * scale
    shunting = pivot["dendritic_shunting"].to_numpy(dtype=float) * scale
    for left, right in zip(additive, shunting):
        ax.plot([0, 1], [left, right], color=COLORS["mute"], lw=0.55, alpha=0.50)
    ax.scatter(
        np.zeros_like(additive),
        additive,
        s=12,
        facecolor="white",
        edgecolor=COLOR_ADDITIVE,
        linewidth=0.65,
        zorder=3,
    )
    ax.scatter(
        np.ones_like(shunting),
        shunting,
        s=12,
        facecolor="white",
        edgecolor=COLOR_SHUNTING,
        linewidth=0.65,
        zorder=3,
    )
    ax.scatter(
        [0, 1],
        [np.mean(additive), np.mean(shunting)],
        s=30,
        marker="s",
        color=[COLOR_ADDITIVE, COLOR_SHUNTING],
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    delta = float(np.mean(shunting - additive))
    unit = " pp" if scale == 100.0 else ""
    ax.text(
        0.50,
        0.04,
        rf"$\Delta={delta:+.3f}$" + unit,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.0,
        color=COLORS["ink"],
    )
    ax.set_xlim(-0.28, 1.28)
    ax.set_xticks([0, 1])
    if show_xlabels:
        ax.set_xticklabels(["Add.", "Shunt."], fontsize=8.7)
    else:
        ax.set_xticklabels([])
    ax.set_ylabel(ylabel, fontsize=8.9)
    if title:
        ax.set_title(title, fontsize=10.0, linespacing=0.92)
    style_axis(ax, grid="y")


def _plot_feedback_relevance(
    ax: plt.Axes,
    frame: pd.DataFrame,
) -> None:
    selected = frame.loc[np.isclose(frame["relative_step"], 1e-5)].copy()
    checkpoint = (
        selected.groupby(
            ["run_dir", "feedback_family"],
            as_index=False,
        )[
            [
                "field_scaled_capture",
                "gradient_scaled_capture",
                "norm_matched_fraction_of_exact",
            ]
        ]
        .mean()
    )
    families = [
        ("submitted_mw", "MW"),
        ("ancestry_available", "Anc."),
        ("exact_transport", "PT"),
    ]
    metrics = [
        ("field_scaled_capture", "Error field", COLORS["scalar"]),
        ("gradient_scaled_capture", "Gradient", COLORS["local"]),
        ("norm_matched_fraction_of_exact", "One step", COLORS["oracle"]),
    ]
    x = np.arange(len(families), dtype=float)
    width = 0.23
    for metric_index, (metric, label, color) in enumerate(metrics):
        means: list[float] = []
        sems: list[float] = []
        for family, _family_label in families:
            values = checkpoint.loc[
                checkpoint["feedback_family"] == family,
                metric,
            ].to_numpy(dtype=float)
            means.append(float(np.mean(values)))
            sems.append(float(np.std(values, ddof=1) / np.sqrt(len(values))))
        offset = (metric_index - 1) * width
        ax.bar(
            x + offset,
            means,
            yerr=sems,
            width=width,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            capsize=1.8,
            label=label,
            zorder=2,
        )
    ax.axhline(0.0, color=COLORS["mute"], lw=0.7, ls="-")
    ax.axhline(1.0, color=COLORS["mute"], lw=0.8, ls="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [family_label for _family, family_label in families],
        fontsize=8.2,
    )
    ax.set_ylim(-0.03, 1.10)
    ax.set_ylabel("Fraction of exact", fontsize=9.1)
    ax.set_title("Feedback-to-step audit", fontsize=10.0)
    ax.legend(
        loc="upper left",
        ncol=1,
        fontsize=7.8,
        handlelength=1.0,
        borderaxespad=0.3,
    )
    style_axis(ax, grid="y")
    panel_label(ax, "C", dx=-39, dy=5)


def build_figure() -> plt.Figure:
    apply_neurips_style()
    fixed_state = pd.read_csv(FIXED_STATE_CSV)
    identity = pd.read_csv(IDENTITY_CSV)
    feedback_relevance = pd.read_csv(FEEDBACK_RELEVANCE_CSV)

    fig = plt.figure(figsize=(MAIN_W, 3.15))
    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.42, 0.92, 1.26],
        wspace=0.72,
    )
    ax_a = fig.add_subplot(outer[0, 0])
    middle = outer[0, 1].subgridspec(2, 1, hspace=0.52)
    ax_b_top = fig.add_subplot(middle[0, 0])
    ax_b_bottom = fig.add_subplot(middle[1, 0])
    ax_c = fig.add_subplot(outer[0, 2])

    _plot_fixed_state(ax_a, fixed_state)
    _plot_identity_axis(
        ax_b_top,
        identity,
        metric="branch_numel_weighted_cosine",
        ylabel="Branch cosine",
        scale=1.0,
        show_xlabels=False,
        title="Identity transfer\n($n=15$)",
    )
    panel_label(ax_b_top, "B", dx=-39, dy=5)
    _plot_identity_axis(
        ax_b_bottom,
        identity,
        metric="test_accuracy",
        ylabel="Test accuracy (%)",
        scale=100.0,
        show_xlabels=True,
    )
    _plot_feedback_relevance(ax_c, feedback_relevance)

    fig.subplots_adjust(left=0.17, right=0.985, top=0.84, bottom=0.20)
    return fig


def main() -> None:
    figure = build_figure()
    _save(figure, "fig_s_mechanistic_audit")
    plt.close(figure)


if __name__ == "__main__":
    main()

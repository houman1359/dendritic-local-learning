#!/usr/bin/env python3
"""Generate the one-layer input-mode path-gain schematic and result figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from neurips_style import COLORS, apply_neurips_style, clean_schematic_axis, panel_label, style_axis


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = DRAFT_DIR / "figures"
SUMMARY_CSV = (
    DRAFT_DIR
    / "analysis"
    / "input_mode_onelayer_probe_summary_20260425"
    / "input_mode_onelayer_grouped.csv"
)

DPI = 350
COLOR_DIRECT = COLORS["shunting"]
COLOR_EXPLICIT = COLORS["pathway"]
COLOR_INH = COLORS["inh"]
COLOR_EXC = COLORS["exc"]
COLOR_ORACLE = COLORS["oracle"]
COLOR_PER_SOMA = COLORS["per_soma"]
COLOR_BP = COLORS["bp"]
COLOR_MUTED = COLORS["mute"]
COLOR_PANEL = "#F7F8FA"
COLOR_LIGHT_GREEN = "#E8F3EC"
COLOR_LIGHT_RED = "#F7E6E8"
COLOR_LIGHT_BLUE = "#E7EEF9"
COLOR_LIGHT_PURPLE = "#EFE8F6"
COLOR_LIGHT_GOLD = "#F8EEDC"


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    face: str,
    edge: str = "#555555",
    fontsize: float = 7.1,
    lw: float = 0.9,
    weight: str = "normal",
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["ink"],
    )
    return patch


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    rad: float = 0.0,
    lw: float = 1.2,
    style: str = "-|>",
    ls: str = "-",
    mutation_scale: float = 11,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=lw,
            linestyle=ls,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def _panel_title(ax: plt.Axes, title: str, subtitle: str | None = None) -> None:
    ax.text(
        0.0,
        1.03,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.9,
        fontweight="bold",
        color=COLORS["ink"],
    )
    if subtitle:
        ax.text(
            0.0,
            0.965,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.4,
            color=COLOR_MUTED,
        )


def _draw_path_gain_panel(ax: plt.Axes) -> None:
    clean_schematic_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _panel_title(
        ax,
        "Inhibition gates path gain",
        r"$G_I(x)$ lowers $R^{tot}$ and attenuates all upstream paths",
    )

    soma = Circle((0.50, 0.82), 0.055, facecolor=COLORS["soma"], edgecolor="#8A4B21", lw=1.0)
    ax.add_patch(soma)
    ax.text(0.50, 0.82, "soma", ha="center", va="center", fontsize=6.7, color="white", fontweight="bold")

    # Dendritic tree with two proximal branches and three distal leaves each.
    proximal = [(0.34, 0.58), (0.66, 0.58)]
    distal = [(0.20, 0.36), (0.34, 0.34), (0.48, 0.36), (0.56, 0.36), (0.70, 0.34), (0.84, 0.36)]
    for p in proximal:
        ax.plot([0.50, p[0]], [0.77, p[1]], color=COLORS["dend"], lw=4.0, solid_capstyle="round", zorder=1)
    for p, leaves in zip(proximal, [distal[:3], distal[3:]]):
        for leaf in leaves:
            ax.plot([p[0], leaf[0]], [p[1], leaf[1]], color=COLORS["dend"], lw=3.0, solid_capstyle="round", zorder=1)
            ax.add_patch(Circle(leaf, 0.018, facecolor=COLOR_EXC, edgecolor="white", lw=0.4, zorder=3))

    # Highlight one upstream path and one inhibitory gate.
    highlighted = [(0.20, 0.36), (0.34, 0.58), (0.50, 0.82)]
    ax.plot(
        [p[0] for p in highlighted],
        [p[1] for p in highlighted],
        color=COLOR_ORACLE,
        lw=2.0,
        alpha=0.90,
        zorder=4,
    )
    ax.add_patch(Circle((0.29, 0.48), 0.030, facecolor=COLOR_LIGHT_RED, edgecolor=COLOR_INH, lw=1.2, zorder=5))
    ax.text(0.29, 0.48, "I", ha="center", va="center", fontsize=8.0, color=COLOR_INH, fontweight="bold", zorder=6)
    _arrow(ax, (0.12, 0.50), (0.25, 0.49), COLOR_INH, lw=1.0)
    ax.text(0.12, 0.53, r"$G_I(x)$", ha="center", va="bottom", fontsize=7.8, color=COLOR_INH)

    _arrow(ax, (0.23, 0.39), (0.31, 0.53), COLOR_ORACLE, rad=-0.05, lw=1.0)
    _arrow(ax, (0.38, 0.61), (0.47, 0.78), COLOR_ORACLE, rad=-0.05, lw=1.0)
    ax.text(0.16, 0.30, "distal synapse", ha="left", va="center", fontsize=6.7, color=COLOR_MUTED)
    ax.text(0.62, 0.78, "teaching signal\ntravels down path", ha="left", va="center", fontsize=6.8, color=COLOR_ORACLE)

    _box(
        ax,
        (0.08, 0.05),
        (0.84, 0.16),
        r"$\alpha_n(x)=\prod_{(i\to k)\in path(n\to 0)} R_k^{tot}(x)g_{i\to k}^{den}$"
        "\n"
        r"$\partial \log \alpha_n / \partial G_k^I = -R_k^{tot}$ on the path",
        "white",
        edge="#D4D7DD",
        fontsize=8.1,
    )


def _draw_input_modes_panel(ax: plt.Axes) -> None:
    clean_schematic_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _panel_title(
        ax,
        "Two inhibitory input modes",
        "Both routes end as nonnegative I-to-E conductance on the dendrite",
    )

    # Direct input-driven inhibition.
    ax.text(0.03, 0.80, "Direct-I stream", fontsize=8.2, fontweight="bold", color=COLOR_DIRECT, ha="left")
    _box(ax, (0.04, 0.62), (0.16, 0.10), "input\n$x\\geq0$", COLOR_LIGHT_BLUE, edge=COLOR_EXC)
    _box(ax, (0.33, 0.69), (0.19, 0.09), "E stream", COLOR_LIGHT_BLUE, edge=COLOR_EXC)
    _box(ax, (0.33, 0.55), (0.19, 0.09), "I stream", COLOR_LIGHT_RED, edge=COLOR_INH)
    _box(ax, (0.68, 0.61), (0.23, 0.13), "E dendrite\nwith I synapses", COLOR_LIGHT_GREEN, edge=COLOR_DIRECT)
    _arrow(ax, (0.20, 0.67), (0.33, 0.735), COLOR_EXC)
    _arrow(ax, (0.20, 0.67), (0.33, 0.595), COLOR_INH)
    _arrow(ax, (0.52, 0.735), (0.68, 0.69), COLOR_EXC)
    _arrow(ax, (0.52, 0.595), (0.68, 0.64), COLOR_INH)
    ax.text(0.79, 0.55, r"$G_I^{branch}(x)$", ha="center", fontsize=7.6, color=COLOR_INH)

    # Explicit inhibitory cells.
    ax.text(0.03, 0.42, "Explicit-I cell", fontsize=8.2, fontweight="bold", color=COLOR_EXPLICIT, ha="left")
    _box(ax, (0.04, 0.22), (0.16, 0.10), "input\n$x\\geq0$", COLOR_LIGHT_BLUE, edge=COLOR_EXC)
    _box(ax, (0.32, 0.30), (0.20, 0.10), "I neuron\n$[3,3]$ tree", COLOR_LIGHT_PURPLE, edge=COLOR_EXPLICIT)
    _box(ax, (0.32, 0.13), (0.20, 0.10), "E drive", COLOR_LIGHT_BLUE, edge=COLOR_EXC)
    _box(ax, (0.68, 0.20), (0.23, 0.13), "E dendrite\nwith I synapses", COLOR_LIGHT_GREEN, edge=COLOR_DIRECT)
    _arrow(ax, (0.20, 0.27), (0.32, 0.35), COLOR_EXC)
    _arrow(ax, (0.20, 0.27), (0.32, 0.18), COLOR_EXC)
    _arrow(ax, (0.52, 0.35), (0.68, 0.29), COLOR_INH)
    _arrow(ax, (0.52, 0.18), (0.68, 0.25), COLOR_EXC)
    ax.text(0.60, 0.39, "learned\nE->I", ha="center", fontsize=6.4, color=COLOR_EXPLICIT)
    ax.text(0.61, 0.17, "E->E", ha="center", fontsize=6.4, color=COLOR_EXC)
    ax.text(0.79, 0.14, r"$G_I^{branch}(h_I(x))$", ha="center", fontsize=7.3, color=COLOR_INH)


def _draw_clean_probe_panel(ax: plt.Axes) -> None:
    clean_schematic_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _panel_title(
        ax,
        "One dendritic layer probe",
        "Path gain needs a branched tree, not two feedforward layers",
    )

    # Clean one-layer probe.
    _box(
        ax,
        (0.04, 0.61),
        (0.42, 0.16),
        "Clean mechanism test\n1 E population\n$[3,3]$ dendritic tree",
        COLOR_LIGHT_GREEN,
        edge=COLOR_DIRECT,
        fontsize=6.8,
        weight="bold",
    )
    ax.text(0.25, 0.54, r"$E=[128]$, optional $I=[128]$", ha="center", fontsize=7.0, color=COLOR_MUTED)
    # Mini tree inside/right of clean box.
    root = (0.66, 0.71)
    branches = [(0.58, 0.58), (0.74, 0.58)]
    leaves = [(0.53, 0.47), (0.62, 0.46), (0.70, 0.46), (0.79, 0.47)]
    ax.add_patch(Circle(root, 0.032, facecolor=COLORS["soma"], edgecolor="#8A4B21", lw=0.8))
    for b in branches:
        ax.plot([root[0], b[0]], [root[1] - 0.02, b[1]], color=COLORS["dend"], lw=2.7, solid_capstyle="round")
    for b, leaf_pair in zip(branches, [leaves[:2], leaves[2:]]):
        for leaf in leaf_pair:
            ax.plot([b[0], leaf[0]], [b[1], leaf[1]], color=COLORS["dend"], lw=2.2, solid_capstyle="round")
            ax.add_patch(Circle(leaf, 0.012, facecolor=COLOR_EXC, edgecolor="white", lw=0.3))
    ax.text(0.78, 0.67, "path gain\ninside tree", fontsize=6.8, color=COLOR_ORACLE, ha="left")
    _arrow(ax, (0.54, 0.47), (0.65, 0.68), COLOR_ORACLE, rad=-0.08, lw=1.0)

    # Confounded two-layer graph.
    _box(
        ax,
        (0.05, 0.16),
        (0.36, 0.17),
        "Confounded diagnostic\nextra feedforward E/I layer",
        "#F2F2F2",
        edge="#B9B9B9",
        fontsize=7.3,
    )
    ax.text(
        0.76,
        0.39,
        "old two-layer graph",
        ha="center",
        va="center",
        fontsize=7.0,
        color=COLOR_MUTED,
        fontweight="bold",
    )
    for x in [0.59, 0.78]:
        _box(ax, (x - 0.050, 0.25), (0.10, 0.075), "E", COLOR_LIGHT_GREEN, edge=COLOR_DIRECT, fontsize=7.8, weight="bold")
        _box(ax, (x - 0.050, 0.11), (0.10, 0.075), "I", COLOR_LIGHT_PURPLE, edge=COLOR_EXPLICIT, fontsize=7.8, weight="bold")
    _arrow(ax, (0.64, 0.285), (0.73, 0.285), COLOR_EXC)
    _arrow(ax, (0.64, 0.145), (0.73, 0.145), COLOR_INH)
    ax.plot([0.59, 0.59], [0.25, 0.19], color=COLOR_INH, lw=1.0)
    ax.plot([0.78, 0.78], [0.185, 0.245], color=COLOR_INH, lw=1.0)
    _arrow(ax, (0.91, 0.47), (0.82, 0.30), COLOR_MUTED, rad=0.10, lw=0.9)
    ax.text(
        0.94,
        0.47,
        "extra graph-credit\nproblem",
        ha="right",
        va="center",
        fontsize=6.8,
        color=COLOR_MUTED,
    )


def _load_grouped(summary_csv: Path) -> pd.DataFrame:
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary CSV: {summary_csv}")
    df = pd.read_csv(summary_csv)
    df["test_accuracy_mean"] = pd.to_numeric(df["test_accuracy_mean"])
    df["test_accuracy_std"] = pd.to_numeric(df["test_accuracy_std"])
    return df


def _draw_results_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    order = [
        ("direct_i_stream__localca_per_soma", "Direct-I\nper-soma"),
        ("direct_i_stream__localca_path_transport", "Direct-I\npath"),
        ("explicit_i_cells__localca_path_transport__i_updates_True", "Explicit-I\npath"),
        ("explicit_i_cells__localca_path_transport__i_updates_False", "Explicit-I\npath\nI frozen"),
        ("explicit_i_cells__standard_bp", "Explicit-I\nBP"),
    ]
    palette = [COLOR_PER_SOMA, COLOR_ORACLE, COLOR_DIRECT, COLOR_EXPLICIT, COLOR_BP]
    hatches = ["", "", "", "//", ""]

    values = []
    errors = []
    labels = []
    for condition, label in order:
        row = df[df["condition"] == condition]
        if row.empty:
            raise RuntimeError(f"Missing condition in summary: {condition}")
        labels.append(label)
        values.append(float(row.iloc[0]["test_accuracy_mean"]) * 100.0)
        errors.append(float(row.iloc[0]["test_accuracy_std"]) * 100.0)

    x = np.arange(len(values))
    bars = ax.bar(
        x,
        values,
        yerr=errors,
        capsize=2.5,
        color=palette,
        edgecolor="white",
        linewidth=0.7,
        width=0.72,
        zorder=3,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
        if hatch:
            bar.set_edgecolor("#5F4778")
            bar.set_linewidth(0.7)

    style_axis(ax, grid="y")
    ax.set_ylim(78, 97.2)
    ax.set_ylabel("Noise-resilience accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(values[-1], color=COLOR_BP, linestyle="--", linewidth=1.0, alpha=0.65)
    ax.text(
        len(values) - 0.15,
        values[-1] + 0.25,
        "BP ceiling",
        ha="right",
        va="bottom",
        fontsize=7.0,
        color=COLOR_BP,
    )
    for xi, val in zip(x, values):
        ax.text(xi, val + 0.55, f"{val:.1f}", ha="center", va="bottom", fontsize=7.0, color=COLORS["ink"])
    ax.set_title("Probe result", fontsize=9.2, fontweight="bold")


def build_figure(summary_csv: Path = SUMMARY_CSV) -> plt.Figure:
    apply_neurips_style()
    df = _load_grouped(summary_csv)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14.4, 3.25),
        gridspec_kw={"wspace": 0.42, "width_ratios": [1.20, 1.18, 1.05, 1.0]},
    )

    _draw_path_gain_panel(axes[0])
    _draw_input_modes_panel(axes[1])
    _draw_clean_probe_panel(axes[2])
    _draw_results_panel(axes[3], df)

    for label, ax in zip(["A", "B", "C", "D"], axes):
        panel_label(ax, label, x=-0.12, y=1.08, fontsize=11.5)

    fig.subplots_adjust(left=0.035, right=0.992, bottom=0.22, top=0.86, wspace=0.42)

    out = FIGURES_DIR / "fig_s_input_mode_path_gain"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=DPI)
    print(f"Saved {out}.{{pdf,png}}")
    return fig


if __name__ == "__main__":
    build_figure()

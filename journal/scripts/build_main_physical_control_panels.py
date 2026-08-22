#!/usr/bin/env python3
"""Render the two promoted physical-depth controls at final main-panel aspect.

The underlying summaries are frozen outputs of the confirmatory analyses.  The
modular source figures place these plots in three-column grids; this renderer
uses the identical means and confidence intervals in the half-width geometry
used by Figure 5, avoiding either nonuniform stretching or large side margins.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from journal_style import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_ERR,
    MARKER_MS,
    PT_LEGEND,
    PT_SMALL,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
DEPTH = ROOT / "source_data" / "nonlinear_physical_depth_confirmatory"
POINT = ROOT / "source_data" / "point_dendrite_credit_controls"
OUT = ROOT / "figures" / "generated" / "fig_physical_controls_main.pdf"
SCHEMATIC_OUT = ROOT / "figures" / "generated" / "fig_figure5_schematics.pdf"
ARCHITECTURE_OUT = (
    ROOT / "figures" / "generated" / "fig_figure5_architecture_schematic.pdf"
)


def _diagram_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _edge(
    ax: plt.Axes,
    parent: tuple[float, float],
    child: tuple[float, float],
    color: str,
    *,
    lw: float = 0.85,
) -> None:
    ax.plot(
        [parent[0], child[0]],
        [parent[1], child[1]],
        color=color,
        lw=lw,
        solid_capstyle="round",
        zorder=1,
    )


def _unit(
    ax: plt.Axes,
    xy: tuple[float, float],
    color: str,
    *,
    radius: float = 0.011,
) -> None:
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=12.0 * (radius / 0.011) ** 2,
        facecolor="white",
        edgecolor=color,
        linewidth=0.9,
        zorder=3,
        clip_on=False,
    )


def _soma(ax: plt.Axes, xy: tuple[float, float], *, radius: float = 0.025) -> None:
    ax.scatter(
        [xy[0]],
        [xy[1]],
        s=58.0 * (radius / 0.025) ** 2,
        facecolor=COLORS["soma"],
        edgecolor="white",
        linewidth=0.8,
        zorder=4,
        clip_on=False,
    )


def _depth_inventory(
    ax: plt.Axes,
    center: float,
    depth: int,
    *,
    soma_y: float = 0.23,
) -> None:
    """Draw exactly eight nonsomatic units at one, two or three stages."""
    soma = (center, soma_y)
    _soma(ax, soma)
    stage_colors = (COLORS["shunting"], COLORS["additive"], COLORS["oracle"])

    if depth == 1:
        nodes = [(center + dx, 0.61) for dx in np.linspace(-0.105, 0.105, 8)]
        for node in nodes:
            _edge(ax, soma, node, stage_colors[0])
            _unit(ax, node, stage_colors[0])
    elif depth == 2:
        proximal = [(center - 0.050, 0.42), (center + 0.050, 0.42)]
        for node in proximal:
            _edge(ax, soma, node, stage_colors[0])
            _unit(ax, node, stage_colors[0])
            for dx in (-0.042, 0.0, 0.042):
                distal = (node[0] + dx, 0.65)
                _edge(ax, node, distal, stage_colors[1])
                _unit(ax, distal, stage_colors[1])
    elif depth == 3:
        proximal = [(center - 0.052, 0.37), (center + 0.052, 0.37)]
        for node in proximal:
            _edge(ax, soma, node, stage_colors[0])
            _unit(ax, node, stage_colors[0])
            middle = (node[0], 0.51)
            _edge(ax, node, middle, stage_colors[1])
            _unit(ax, middle, stage_colors[1])
            for dx in (-0.036, 0.036):
                distal = (middle[0] + dx, 0.67)
                _edge(ax, middle, distal, stage_colors[2])
                _unit(ax, distal, stage_colors[2])
    else:  # pragma: no cover - fixed publication inventory
        raise ValueError(depth)


def _resource_bracket(
    ax: plt.Axes,
    x0: float,
    x1: float,
    y: float,
    label: str,
    *,
    color: str = COLORS["mute"],
) -> None:
    ax.plot([x0, x1], [y, y], color=color, lw=0.7, clip_on=False)
    ax.plot([x0, x0], [y, y - 0.025], color=color, lw=0.7, clip_on=False)
    ax.plot([x1, x1], [y, y - 0.025], color=color, lw=0.7, clip_on=False)
    ax.text(
        (x0 + x1) / 2,
        y + 0.025,
        label,
        ha="center",
        va="bottom",
        fontsize=PT_SMALL,
        color=color,
    )


def _dense_mlp(ax: plt.Axes, center: float) -> None:
    rows = [
        [(center + dx, 0.67) for dx in (-0.060, 0.0, 0.060)],
        [(center + dx, 0.52) for dx in (-0.075, -0.025, 0.025, 0.075)],
        [(center + dx, 0.38) for dx in (-0.055, 0.0, 0.055)],
    ]
    for upper, lower in zip(rows, rows[1:]):
        for parent in upper:
            for child in lower:
                _edge(ax, parent, child, COLORS["point_mlp"], lw=0.45)
    for row in rows:
        for node in row:
            _unit(ax, node, COLORS["point_mlp"], radius=0.010)
    output = (center, 0.23)
    for parent in rows[-1]:
        _edge(ax, parent, output, COLORS["point_mlp"], lw=0.65)
    _soma(ax, output)


def _parallel_modules(ax: plt.Axes, center: float) -> None:
    output = (center, 0.23)
    nodes = [(center + dx, 0.58) for dx in np.linspace(-0.105, 0.105, 8)]
    for node in nodes:
        _edge(ax, node, output, COLORS["oracle"], lw=0.8)
        _unit(ax, node, COLORS["oracle"])
    _soma(ax, output)


def _draw_architecture_schematic(ax: plt.Axes) -> None:
    _diagram_axis(ax)
    _resource_bracket(
        ax,
        0.385,
        0.955,
        0.875,
        "same 8 branch modules",
    )
    ax.text(
        0.19,
        0.76,
        "parameter-matched",
        ha="center",
        va="center",
        fontsize=PT_SMALL,
        color=COLORS["mute"],
    )
    _dense_mlp(ax, 0.16)
    _parallel_modules(ax, 0.51)
    _depth_inventory(ax, 0.84, 3)
    for center, label, color in (
        (0.16, "point MLP", COLORS["point_mlp"]),
        (0.51, "grouped star", COLORS["oracle"]),
        (0.84, "serial tree", COLORS["shunting"]),
    ):
        ax.text(
            center,
            0.070,
            label,
            ha="center",
            va="center",
            fontsize=7.0,
            color=color,
        )
    ax.text(
        0.51,
        0.76,
        "parallel",
        ha="center",
        va="center",
        fontsize=PT_SMALL,
        color=COLORS["oracle"],
    )
    ax.text(
        0.84,
        0.76,
        "serial",
        ha="center",
        va="center",
        fontsize=PT_SMALL,
        color=COLORS["shunting"],
    )


def build_schematics() -> None:
    """Build the two Figure 5 definition panels at their final aspect."""
    apply_neurips_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(FIG_W, 1.85),
        gridspec_kw={
            "left": 0.025,
            "right": 0.985,
            "bottom": 0.04,
            "top": 0.96,
            "wspace": 0.08,
        },
    )
    for ax in (ax_a, ax_b):
        _diagram_axis(ax)

    _resource_bracket(
        ax_a,
        0.055,
        0.945,
        0.875,
        "same 8 branch units, contacts and parameters",
    )
    for center, depth, inventory in zip(
        (0.17, 0.50, 0.83),
        (1, 2, 3),
        ("[8]", "[2,3]", "[2,1,2]"),
        strict=True,
    ):
        _depth_inventory(ax_a, center, depth)
        ax_a.text(
            center,
            0.125,
            rf"$D_{{\mathrm{{p}}}}={depth}$",
            ha="center",
            va="center",
            fontsize=7.2,
            color=COLORS["ink"],
        )
        ax_a.text(
            center,
            0.065,
            inventory,
            ha="center",
            va="center",
            fontsize=PT_SMALL,
            color=COLORS["mute"],
        )

    _draw_architecture_schematic(ax_b)

    fig.canvas.draw()
    audit_layout(fig, "fig_figure5_schematics")
    SCHEMATIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SCHEMATIC_OUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Wrote {SCHEMATIC_OUT}")

    # Figure 5G occupies one third of the publication canvas.  Author it in a
    # one-third source cell so labels remain at their intended print size.
    fig_arch, axes = plt.subplots(
        1,
        3,
        figsize=(FIG_W, 1.85),
        gridspec_kw={
            "left": 0.008,
            "right": 0.992,
            "bottom": 0.04,
            "top": 0.96,
            "wspace": 0.08,
        },
    )
    _draw_architecture_schematic(axes[0])
    for ax in axes[1:]:
        _diagram_axis(ax)
    fig_arch.canvas.draw()
    audit_layout(fig_arch, "fig_figure5_architecture_schematic")
    fig_arch.savefig(
        ARCHITECTURE_OUT,
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig_arch)
    print(f"Wrote {ARCHITECTURE_OUT}")


def _line(
    ax: plt.Axes,
    frame: pd.DataFrame,
    mask: pd.Series,
    *,
    color: str,
    marker: str,
) -> None:
    part = frame[mask].sort_values("depth")
    x = part.depth.to_numpy(float)
    mean = part.mean_test_accuracy.to_numpy(float)
    low = part.ci95_low_test_accuracy.to_numpy(float)
    high = part.ci95_high_test_accuracy.to_numpy(float)
    ax.errorbar(
        x,
        mean,
        yerr=np.vstack([mean - low, high - mean]),
        color=color,
        lw=LW_DATA,
        elinewidth=LW_ERR,
        capsize=ERR_CAPSIZE,
        marker=marker,
        ms=MARKER_MS,
        markeredgecolor="white",
        markeredgewidth=0.5,
    )


def main() -> None:
    build_schematics()
    depth = pd.read_csv(DEPTH / "condition_summary.csv")
    point = pd.read_csv(POINT / "condition_summary.csv")

    apply_neurips_style()
    fig, (ax_a, ax_b) = plt.subplots(
        1,
        2,
        figsize=(FIG_W, 2.55),
        gridspec_kw={
            "left": 0.09,
            "right": 0.985,
            "bottom": 0.18,
            "top": 0.84,
            "wspace": 0.42,
        },
    )

    _line(
        ax_a,
        depth,
        depth.regime.eq("aligned")
        & depth.mechanism.eq("shunting")
        & depth.method.eq("bp")
        & depth.transport.eq("backpropagation"),
        color=COLORS["shunting"],
        marker="o",
    )
    _line(
        ax_a,
        depth,
        depth.regime.eq("aligned")
        & depth.mechanism.eq("additive")
        & depth.method.eq("bp")
        & depth.transport.eq("backpropagation"),
        color=COLORS["additive"],
        marker="s",
    )
    ax_a.set_xlim(0.7, 3.3)
    ax_a.set_xticks([1, 2, 3])
    ax_a.set_ylim(0.44, 1.06)
    ax_a.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_a.set_xlabel(r"physical stage count $D_{\mathrm{p}}$")
    ax_a.set_ylabel("test accuracy")
    panel_title(ax_a, "A", "Divisive control")
    style_axis(ax_a, grid="y")
    ax_a.text(2.1, 0.815, "shunting", ha="right", va="bottom",
              fontsize=PT_LEGEND, color=COLORS["shunting"])
    ax_a.text(3.0, 0.492, "raw additive", ha="right", va="top",
              fontsize=PT_LEGEND, color=COLORS["additive"])

    _line(
        ax_b,
        point,
        point.regime.eq("aligned")
        & point.architecture.eq("serial_tree")
        & point.credit.eq("full_bp")
        & point.depth.gt(0),
        color=COLORS["shunting"],
        marker="o",
    )
    _line(
        ax_b,
        point,
        point.regime.eq("aligned")
        & point.architecture.eq("all_active_star")
        & point.credit.eq("full_bp")
        & point.depth.gt(0),
        color=COLORS["oracle"],
        marker="^",
    )
    ax_b.set_xlim(0.7, 3.3)
    ax_b.set_xticks([1, 2, 3])
    ax_b.set_ylim(0.44, 1.06)
    ax_b.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax_b.set_xlabel(r"physical stage count $D_{\mathrm{p}}$")
    ax_b.set_ylabel("test accuracy")
    panel_title(ax_b, "B", "Serial composition")
    style_axis(ax_b, grid="y")
    ax_b.text(3.0, 0.955, "serial tree", ha="right", va="bottom",
              fontsize=PT_LEGEND, color=COLORS["shunting"])
    ax_b.text(3.0, 0.572, "grouped star", ha="right", va="top",
              fontsize=PT_LEGEND, color=COLORS["oracle"])

    fig.canvas.draw()
    audit_layout(fig, "fig_physical_controls_main")
    audit_text_over_data(fig, "fig_physical_controls_main")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()

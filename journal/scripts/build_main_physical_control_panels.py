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

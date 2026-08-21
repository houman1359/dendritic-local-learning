#!/usr/bin/env python3
"""Render the transparent operating-point calibration for physical depth."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_REF,
    PT_LEGEND,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
FIGURES = ROOT / "figures" / "generated"
DEPTH_COLORS = {1: COLORS["mute"], 2: COLORS["pathway"], 3: COLORS["shunting"]}
DEPTH_MARKERS = {1: "o", 2: "s", 3: "D"}


def _read(folder: str, file: str = "bp_seed_rows.csv") -> pd.DataFrame:
    return pd.read_csv(SOURCE / folder / file)


def _depth_lines(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    *,
    xlabel: str,
    ylim: tuple[float, float] = (0.48, 0.98),
) -> None:
    grouped = (
        frame.groupby([x_column, "depth"], as_index=False)
        .test_accuracy.agg(["mean", "min", "max"])
        .reset_index()
    )
    for depth in (1, 2, 3):
        part = grouped[grouped.depth.eq(depth)].sort_values(x_column)
        x = part[x_column].to_numpy(float)
        mean = part["mean"].to_numpy(float)
        low = part["min"].to_numpy(float)
        high = part["max"].to_numpy(float)
        ax.plot(
            x,
            mean,
            marker=DEPTH_MARKERS[depth],
            ms=4.0,
            lw=LW_DATA,
            color=DEPTH_COLORS[depth],
            label=f"D{depth}",
        )
        ax.fill_between(x, low, high, color=DEPTH_COLORS[depth], alpha=0.10, linewidth=0)
    ax.axhline(0.5, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(*ylim)
    style_axis(ax, grid="y")


def main() -> None:
    apply_neurips_style()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_W, 5.0),
        gridspec_kw={
            "left": 0.10,
            "right": 0.985,
            "bottom": 0.10,
            "top": 0.91,
            "wspace": 0.43,
            "hspace": 0.67,
        },
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    severe = _read("nonlinear_physical_depth_canary", "bp_aligned_seed_rows.csv")
    severe = severe.melt(
        id_vars=["depth", "seed"],
        value_vars=["train_accuracy", "test_accuracy"],
        var_name="split",
        value_name="accuracy",
    )
    means = severe.groupby(["depth", "split"], as_index=False).accuracy.mean()
    width = 0.34
    for offset, split, color in (
        (-width / 2, "train_accuracy", COLORS["pathway"]),
        (width / 2, "test_accuracy", COLORS["mute"]),
    ):
        part = means[means.split.eq(split)].sort_values("depth")
        ax_a.bar(
            part.depth.to_numpy(float) + offset,
            part.accuracy.to_numpy(float),
            width=width,
            facecolor=color,
            edgecolor="white",
            label="train" if split.startswith("train") else "severe-shift test",
        )
    for _, row in severe.iterrows():
        offset = -width / 2 if row["split"] == "train_accuracy" else width / 2
        ax_a.plot(row.depth + offset, row.accuracy, "o", ms=2.3, color=COLORS["ink"], alpha=0.60)
    ax_a.axhline(0.5, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_a.set_xticks([1, 2, 3])
    ax_a.set_xlabel(r"physical stage count $D_{\mathrm{p}}$")
    ax_a.set_ylabel("accuracy")
    ax_a.set_ylim(0.48, 0.72)
    panel_title(ax_a, "A", "Original shift canary failed")
    style_axis(ax_a, grid="y")
    clean_legend(ax_a, fontsize=PT_LEGEND, loc="upper left")

    accessibility = _read("nonlinear_physical_depth_accessibility")
    _depth_lines(ax_b, accessibility, "test_gain_sigma", xlabel="test gain SD")
    ax_b.axvline(0.25, color=COLORS["ink"], ls=":", lw=LW_REF)
    panel_title(ax_b, "B", "Unseen gain shift erodes access")
    clean_legend(ax_b, fontsize=PT_LEGEND, loc="upper right")

    coupling = _read("nonlinear_physical_depth_coupling")
    _depth_lines(ax_c, coupling, "child_conductance", xlabel="initial child conductance")
    ax_c.set_xscale("log", base=4)
    ax_c.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
    ax_c.axvline(16, color=COLORS["ink"], ls=":", lw=LW_REF)
    panel_title(ax_c, "C", "Coupling unlocks serial depth")

    signal = _read("nonlinear_physical_depth_signal")
    boundary = _read("nonlinear_physical_depth_boundary")
    combined = pd.concat([signal, boundary], ignore_index=True)
    _depth_lines(ax_d, combined, "signal_delta", xlabel="excitatory signal contrast")
    ax_d.axvline(0.80, color=COLORS["ink"], ls=":", lw=LW_REF)
    panel_title(ax_d, "D", "Accessible non-ceiling boundary")

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_supp_nonlinear_depth_calibration")
    audit_text_over_data(fig, "fig_supp_nonlinear_depth_calibration")
    fig.savefig(
        FIGURES / "fig_supp_nonlinear_depth_calibration.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_supp_nonlinear_depth_calibration.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()

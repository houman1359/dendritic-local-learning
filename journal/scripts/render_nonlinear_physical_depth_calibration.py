#!/usr/bin/env python3
"""Render the transparent operating-point calibration for physical depth."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
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
# D1 is drawn last so the smallest series is never buried under D2/D3 where
# the three depths coincide (child conductance 1, test gain SD 0.8).
DEPTH_ORDER = (3, 2, 1)
CHANCE = 0.5
# Frozen accessibility criterion (analysis/NONLINEAR_PHYSICAL_DEPTH_SIGNAL_
# CONTRACT_20260812.md, BOUNDARY_CONTRACT_20260812.md): every depth's two-seed
# mean test accuracy in [0.60, 0.95], each seed above 0.57, D3 - D1 >= 0.02.
CRITERION_LOW = 0.60
CRITERION_HIGH = 0.95
# Panels A-C share one expanded ordinate; D keeps the full range because its
# D3 curve reaches 0.92.
YLIM_NARROW = (0.48, 0.70)
YLIM_WIDE = (0.48, 0.98)
SEED_MS = 2.6
SEED_DODGE = 0.011  # x units of the linear panels B and D (spans 0.55 / 0.56)
SEED_MEW = LW_EDGE
MEAN_MEW = LW_HAIR
MEAN_MS = 3.8


def _read(folder: str, file: str = "bp_seed_rows.csv") -> pd.DataFrame:
    return pd.read_csv(SOURCE / folder / file)


def _seed_offsets(x: np.ndarray, n_seeds: int, *, log_base: float | None) -> list[np.ndarray]:
    """Small deterministic horizontal displacements, one array per seed rank.

    Linear axes are dodged by a fixed fraction of the panel's x span
    (``SEED_DODGE`` in axis units set by the caller); log axes by a fixed
    fraction of one decade of the base.
    """
    ranks = (np.arange(n_seeds) - (n_seeds - 1) / 2.0)
    if log_base is None:
        return [x + r * SEED_DODGE for r in ranks]
    return [x * log_base ** (0.045 * r) for r in ranks]


def _depth_lines(
    ax: plt.Axes,
    frame: pd.DataFrame,
    x_column: str,
    *,
    xlabel: str,
    ylim: tuple[float, float],
    log_base: float | None = None,
) -> None:
    grouped = (
        frame.groupby([x_column, "depth"], as_index=False)
        .test_accuracy.agg(["mean"])
        .reset_index()
    )
    for z, depth in enumerate(DEPTH_ORDER):
        color = DEPTH_COLORS[depth]
        part = grouped[grouped.depth.eq(depth)].sort_values(x_column)
        x = part[x_column].to_numpy(float)
        mean = part["mean"].to_numpy(float)
        ax.plot(
            x,
            mean,
            marker=DEPTH_MARKERS[depth],
            ms=MEAN_MS,
            mec="white",
            mew=MEAN_MEW,
            lw=LW_DATA,
            color=color,
            label=f"D{depth}",
            zorder=3 + z,
        )
        # Individual exploratory seeds: open markers, dodged slightly in x so
        # the two values stay visible where they are closer than a marker.
        # Seeds are ranked within each x position (the 0.80 boundary in panel
        # D used a different seed pair from the ladder).
        sx, sy = [], []
        for x_value, cell in frame[frame.depth.eq(depth)].groupby(x_column):
            cell = cell.sort_values("seed")
            offsets = _seed_offsets(np.array([float(x_value)]), len(cell), log_base=log_base)
            sx.extend(float(o[0]) for o in offsets)
            sy.extend(cell.test_accuracy.to_numpy(float))
        ax.plot(
            sx,
            sy,
            linestyle="none",
            marker="o",
            ms=SEED_MS,
            mfc="white",
            mec=color,
            mew=SEED_MEW,
            zorder=6 + z,
        )
    ax.axhline(CHANCE, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("test accuracy")
    ax.set_ylim(*ylim)
    style_axis(ax, grid="y")


def _seed_handle() -> Line2D:
    return Line2D(
        [], [], linestyle="none", marker="o", ms=SEED_MS, mfc="white",
        mec=COLORS["ink"], mew=SEED_MEW, label="seed (n = 2)",
    )


def _token_weights(fig: plt.Figure) -> None:
    """Pin spine, tick and grid weights onto the journal line-weight tokens.

    ``journal_style.style_axis`` predates the 2026-09-08 token set: it still
    hard-sets 0.8 pt spines/ticks and 0.6 pt grid lines, neither of which is a
    line-weight token, so the strict canvas audit reports every panel.  This
    pass runs after the panels are final and re-states the same hierarchy in
    tokens -- spine and major tick to ``LW_EDGE`` (the edge weight the native
    canvas uses for exactly these marks), minor tick and grid line to
    ``LW_HAIR``.  Geometry, data and colour are untouched.
    """
    for ax in fig.get_axes():
        for spine in ax.spines.values():
            spine.set_linewidth(LW_EDGE)
        ax.tick_params(axis="both", which="major", width=LW_EDGE,
                       grid_linewidth=LW_HAIR)
        ax.tick_params(axis="both", which="minor", width=LW_HAIR,
                       grid_linewidth=LW_HAIR)
        for line in (*ax.get_xgridlines(), *ax.get_ygridlines()):
            line.set_linewidth(LW_HAIR)


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

    # A: bars grow from chance so bar length is the above-chance accuracy;
    # colour encodes depth (as in B-D) and fill encodes the split.
    severe = _read("nonlinear_physical_depth_canary", "bp_aligned_seed_rows.csv")
    severe = severe.melt(
        id_vars=["depth", "seed"],
        value_vars=["train_accuracy", "test_accuracy"],
        var_name="split",
        value_name="accuracy",
    )
    means = severe.groupby(["depth", "split"], as_index=False).accuracy.mean()
    width = 0.34
    for offset, split in ((-width / 2, "train_accuracy"), (width / 2, "test_accuracy")):
        part = means[means.split.eq(split)].sort_values("depth")
        for depth, value in zip(part.depth.to_numpy(int), part.accuracy.to_numpy(float)):
            color = DEPTH_COLORS[depth]
            filled = split == "train_accuracy"
            ax_a.bar(
                depth + offset,
                value - CHANCE,
                bottom=CHANCE,
                width=width,
                facecolor=color if filled else "white",
                edgecolor=color,
                linewidth=LW_EDGE,
                zorder=2,
            )
    for _, row in severe.iterrows():
        offset = -width / 2 if row["split"] == "train_accuracy" else width / 2
        ax_a.plot(
            row.depth + offset, row.accuracy, linestyle="none", marker="o", ms=SEED_MS,
            mfc="white", mec=COLORS["ink"], mew=SEED_MEW, zorder=4,
        )
    ax_a.axhline(CHANCE, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=1)
    ax_a.set_xticks([1, 2, 3], ["D1", "D2", "D3"])
    ax_a.set_xlabel("physical depth")
    ax_a.set_ylabel("accuracy")
    ax_a.set_ylim(*YLIM_NARROW)
    ax_a.set_yticks(np.arange(0.50, 0.701, 0.05))
    panel_title(ax_a, "A", "Severe gain shift at test time")
    style_axis(ax_a, grid="y")
    handles = [
        Patch(facecolor=COLORS["dend"], edgecolor=COLORS["dend"], linewidth=LW_EDGE,
              label="train (filled)"),
        Patch(facecolor="white", edgecolor=COLORS["mute"], linewidth=LW_EDGE,
              label="severe-shift test (open)"),
    ]
    clean_legend(ax_a, handles=handles, fontsize=PT_LEGEND, loc="upper left")

    accessibility = _read("nonlinear_physical_depth_accessibility")
    _depth_lines(ax_b, accessibility, "test_gain_sigma", xlabel="test gain SD",
                 ylim=YLIM_NARROW)
    ax_b.set_yticks(np.arange(0.50, 0.701, 0.05))
    ax_b.axvline(0.25, color=COLORS["ink"], ls=":", lw=LW_REF, zorder=1)
    panel_title(ax_b, "B", "Unseen gain shift erodes access")
    handles_b = [
        Line2D([], [], color=DEPTH_COLORS[d], marker=DEPTH_MARKERS[d], ms=MEAN_MS,
               mec="white", mew=MEAN_MEW, lw=LW_DATA, label=f"D{d}")
        for d in (1, 2, 3)
    ] + [_seed_handle()]
    clean_legend(ax_b, handles=handles_b, fontsize=PT_LEGEND, loc="upper right")

    coupling = _read("nonlinear_physical_depth_coupling")
    _depth_lines(ax_c, coupling, "child_conductance", xlabel="initial child conductance",
                 ylim=YLIM_NARROW, log_base=4.0)
    ax_c.set_xscale("log", base=4)
    ax_c.set_xticks([1, 4, 16, 64], ["1", "4", "16", "64"])
    ax_c.set_yticks(np.arange(0.50, 0.701, 0.05))
    ax_c.axvline(16, color=COLORS["ink"], ls=":", lw=LW_REF, zorder=1)
    panel_title(ax_c, "C", "Coupling unlocks serial depth")

    signal = _read("nonlinear_physical_depth_signal")
    boundary = _read("nonlinear_physical_depth_boundary")
    combined = pd.concat([signal, boundary], ignore_index=True)
    _depth_lines(ax_d, combined, "signal_delta", xlabel="excitatory signal contrast",
                 ylim=YLIM_WIDE)
    # Criterion window across the full x range, behind every series.
    x_lo, x_hi = ax_d.get_xlim()
    ax_d.fill_between(
        [x_lo, x_hi], CRITERION_LOW, CRITERION_HIGH,
        color=COLORS["grid"], alpha=0.30, linewidth=0, zorder=-1,
    )
    ax_d.set_xlim(x_lo, x_hi)
    ax_d.axhline(CRITERION_LOW, color=COLORS["ink"], ls="-", lw=LW_REF, zorder=1)
    ax_d.axhline(CRITERION_HIGH, color=COLORS["ink"], ls="-", lw=LW_REF, zorder=1)
    ax_d.axvline(0.80, color=COLORS["ink"], ls=":", lw=LW_REF, zorder=1)
    ax_d.text(
        0.04, 0.925, "criterion window:\ndepth means in [0.60, 0.95]",
        transform=ax_d.transAxes, ha="left", va="top", fontsize=PT_LEGEND,
        color=COLORS["ink"],
    )
    # The D1 pass/fail margin that this panel adjudicates.
    d1 = combined[combined.depth.eq(1)].groupby("signal_delta").test_accuracy.mean()
    fail_x, fail_y = 0.72, float(d1.loc[0.72])
    pass_x, pass_y = 0.80, float(d1.loc[0.80])
    assert fail_y < CRITERION_LOW <= pass_y
    for (px, py), (tx, ty), ha, text in (
        ((fail_x, fail_y), (0.56, 0.548), "center", f"D1 {fail_y:.3f} < 0.60"),
        ((pass_x, pass_y), (pass_x - 0.008, 0.548), "right", f"D1 {pass_y:.3f}"),
    ):
        # Leader drawn as a plain hairline from just outside the marker to the
        # label's top edge.
        lx = tx if ha == "center" else tx - 0.03
        ax_d.plot([px, lx], [py - 0.006, ty + 0.009], color=COLORS["ink"], lw=LW_HAIR,
                  solid_capstyle="butt", zorder=2)
        ax_d.text(tx, ty, text, ha=ha, va="top", fontsize=PT_LEGEND, color=COLORS["ink"])
    panel_title(ax_d, "D", "Accessible non-ceiling boundary")

    FIGURES.mkdir(parents=True, exist_ok=True)
    _token_weights(fig)
    fig.canvas.draw()
    audit_layout(fig, "fig_supp_nonlinear_depth_calibration")
    audit_text_over_data(fig, "fig_supp_nonlinear_depth_calibration")
    from panel_letter_layout import finish_panel_letters
    metadata=finish_panel_letters(fig,[(chr(65+i),i//2,i%2,[ax])
                                      for i,ax in enumerate(axes.ravel())])
    fig.savefig(
        FIGURES / "fig_supp_nonlinear_depth_calibration.pdf",
        metadata=metadata,
    )
    fig.savefig(FIGURES / "fig_supp_nonlinear_depth_calibration.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()

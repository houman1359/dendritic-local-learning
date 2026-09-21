#!/usr/bin/env python3
"""Render the adaptive local conductance-reliability experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from journal_style import style_direct_color_labels
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    SEED_ALPHA,
    SEED_MS,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)
from figure_canvas import enforce_tokens


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "adaptive_conductance_reliability"
FIGURE = ROOT / "figures" / "generated" / "fig_adaptive_conductance_reliability.pdf"


def token_run(ax, x, y, parts, *, size=PT_ANNOT, color=None, drop_pt=1.6,
              ha="center", zorder=5):
    """One line of base + subscript spans, drawn at token type sizes only.

    Mathtext shrinks a subscript to 0.7x its base, so a 7.0 pt annotation
    carried its branch index at 4.90 pt -- below the 7.0 pt floor and the
    smallest type in the volume.  Here every span is a real ``size`` pt text
    placed on the line's own baseline, a subscript span dropped by
    ``drop_pt``, the same chained-span idiom as
    :func:`figure_canvas.token_subscript`.  ``parts`` is a sequence of
    ``(text, is_subscript)`` pairs; the run is centred on ``x`` (display
    metrics, so the centring survives a re-measure).
    """
    from matplotlib.font_manager import FontProperties

    color = COLORS["ink"] if color is None else color
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    prop = FontProperties(size=size)
    widths = [renderer.get_text_width_height_descent(text, prop, False)[0]
              for text, _ in parts]
    baseline_text = "".join(text for text, is_sub in parts if not is_sub)
    _, height_px, descent_px = renderer.get_text_width_height_descent(
        baseline_text or "Xy", prop, False)
    box = ax.get_window_extent(renderer=renderer)
    drop = drop_pt * fig.dpi / 72.0
    anchor = {"center": 0.5, "right": 1.0}.get(ha, 0.0)
    cursor = x - anchor * sum(widths) / box.width
    # ``y`` keeps its meaning from the mathtext span it replaces: the vertical
    # centre of the line, so the run sits where the old string sat.
    baseline = y - (height_px / 2.0 - descent_px) / box.height
    for (text, is_sub), width in zip(parts, widths):
        ax.text(cursor, baseline - (drop / box.height if is_sub else 0.0),
                text, fontsize=size, color=color, ha="left", va="baseline",
                zorder=zorder, clip_on=False)
        cursor += width / box.width


def tokenise_axis(ax, grid="none"):
    """``style_axis``, then the canvas weights for spines, ticks and grid.

    :func:`journal_style.style_axis` still writes the pre-token weights (a
    0.8 pt spine and tick, a 0.6 pt grid line); the canvas token set puts a
    spine and a tick at ``LW_EDGE`` and a grid line at ``LW_HAIR``, which is
    what the natively drawn sheets of this supplement print.
    """
    style_axis(ax, grid=grid)
    if grid in {"x", "y", "both"}:
        ax.grid(True, axis=grid, zorder=0, linewidth=LW_HAIR, alpha=0.55,
                color=COLORS["grid"])
    for spine in ax.spines.values():
        spine.set_linewidth(LW_EDGE)
    ax.tick_params(axis="both", which="major", width=LW_EDGE)
    ax.tick_params(axis="both", which="minor", width=LW_HAIR)


def estimator_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "A", "Local reliability estimate")
    # Each box carries a word line and a symbol line.  Both were mathtext set
    # at 6.50 pt, which put every subscript at 4.55 pt; the words are now at
    # the type token and the symbols are token-size spans with a real dropped
    # subscript.  The hat on the moments is carried by the word "estimates"
    # instead: matplotlib draws \widehat from Cmex10, a maths fallback face
    # that must not reach a print PDF.
    boxes = [
        (0.03, 0.55, 0.25, 0.20, "paired noisy",
         [("credit g", False), ("1", True), (", g", False), ("2", True)],
         COLORS["local"]),
        (0.38, 0.55, 0.25, 0.20, "local estimates",
         [("S", False), ("b", True), (", N", False), ("b", True)],
         COLORS["oracle"]),
        (0.73, 0.55, 0.24, 0.20, "adaptive shunt",
         [("\u03ba", False), ("b", True), (" \u2265 0", False)],
         COLORS["shunting"]),
    ]
    line = 8.75 / 2.0            # half a 7.0 pt line at linespacing 1.25, in pt
    for x, y, width, height, label, symbols, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height, boxstyle="round,pad=0.018",
                facecolor="white", edgecolor=color, lw=LW_DATA,
            )
        )
        centre = (x + width / 2, y + height / 2)
        offset = line / (ax.get_window_extent().height * 72.0 / ax.figure.dpi)
        ax.text(centre[0], centre[1] + offset, label, ha="center",
                va="center", fontsize=PT_ANNOT, color=color)
        token_run(ax, centre[0], centre[1] - offset, symbols,
                  size=PT_ANNOT, color=color)
    for left, right in ((0.28, 0.38), (0.63, 0.73)):
        ax.add_patch(
            FancyArrowPatch(
                (left, 0.65), (right, 0.65), arrowstyle="-|>", mutation_scale=8,
                lw=LW_REF, color=COLORS["mute"],
            )
        )
    # The same two definitions, set with token-size spans.  The inner product
    # is written with a centre dot and the norm with the double bar that the
    # journal face carries, so no glyph is fetched from STIXGeneral or Cmex10.
    token_run(
        ax, 0.5, 0.35,
        [("S", False), ("b", True), (" = g", False), ("1", True),
         (" \u00b7 g", False), ("2", True), ("     N", False), ("b", True),
         (" = \u2225g", False), ("1", True), (" \u2212 g", False), ("2", True),
         ("\u2225\u00b2 / 2", False)],
        size=PT_ANNOT, color=COLORS["ink"],
    )
    token_run(
        ax, 0.5, 0.18,
        [("a", False), ("b", True), (" = min{1,  S", False), ("b", True),
         (" / [c(S", False), ("b", True), (" + N", False), ("b", True),
         (")]}", False)],
        size=PT_ANNOT, color=COLORS["shunting"],
    )


def main() -> None:
    apply_neurips_style()
    branch = pd.read_csv(SOURCE / "branch_estimates.csv")
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    outcomes = pd.read_csv(SOURCE / "seed_outcomes.csv")
    contrasts = pd.read_csv(SOURCE / "paired_contrasts.csv")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_W, 4.65),
        gridspec_kw={
            "left": 0.10,
            "right": 0.985,
            "bottom": 0.175,
            "top": 0.905,
            "wspace": 0.40,
            "hspace": 0.62,
        },
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    estimator_schematic(ax_a)

    high = branch[np.isclose(branch.reliability_heterogeneity, 2.0)]
    gain_summary = high.groupby("branch", as_index=False).agg(
        oracle=("oracle_initial_gain", "mean"),
        adaptive=("adaptive_final_gain", "mean"),
        adaptive_sd=("adaptive_final_gain", "std"),
    )
    x = gain_summary.branch.to_numpy(int) + 1
    ax_b.plot(x, gain_summary.oracle, color=COLORS["oracle"], marker="D", ms=3.5,
              lw=LW_DATA, label="fixed oracle")
    ax_b.errorbar(x, gain_summary.adaptive, yerr=gain_summary.adaptive_sd,
                  color=COLORS["shunting"], marker="o", ms=3.5, lw=LW_DATA,
                  elinewidth=LW_ERR, capsize=1.8, label="adaptive local")
    ax_b.set_xticks(x)
    ax_b.set_xlabel("branch index")
    ax_b.set_ylabel("attenuation gain")
    panel_title(ax_b, "B", "Estimated branch ordering")
    ax_b.text(.03,.96,"mean ± SD",transform=ax_b.transAxes,va="top",
              fontsize=PT_LEGEND,color=COLORS["mute"])
    tokenise_axis(ax_b, grid="y")
    clean_legend(ax_b, fontsize=PT_LEGEND, loc="lower right")

    # One colour per condition, shared with the fixed-profile study (the
    # consolidated supplement pastes both contrast panels on one sheet): the
    # global control takes the same salmon there, which also separates it from
    # the shuffled control where the two cross at low heterogeneity.
    methods = [
        ("noisy_no_shunt", "no shunt", COLORS["ink"], "o"),
        ("adaptive_global_shunt", "adaptive global", COLORS["per_soma"], "s"),
        ("adaptive_shuffled_shunt", "adaptive shuffled", COLORS["mute"], "^"),
        ("adaptive_local_shunt", "adaptive local", COLORS["shunting"], "D"),
        ("initial_oracle_shunt", "fixed oracle", COLORS["oracle"], "P"),
    ]
    for method, label, color, marker in methods:
        part = summary[summary.method.eq(method)].sort_values("reliability_heterogeneity")
        mean = part.mean_final_test_loss.to_numpy(float)
        low = part.ci95_low_final_test_loss.to_numpy(float)
        high_ci = part.ci95_high_final_test_loss.to_numpy(float)
        h = part.reliability_heterogeneity.to_numpy(float)
        ax_c.plot(h, mean, color=color, marker=marker, ms=3.5, lw=LW_DATA, label=label)
        ax_c.fill_between(h, low, high_ci, color=color, alpha=0.08, linewidth=0)
    ax_c.set_xticks([0, 1, 2])
    ax_c.set_xlabel("credit-reliability heterogeneity")
    ax_c.set_ylabel("test loss after 40 updates")
    panel_title(ax_c, "C", "No endpoint gain over no shunt")
    ax_c.text(.03,.95,"mean / 95% CI",transform=ax_c.transAxes,va="top",
              fontsize=PT_LEGEND,color=COLORS["mute"])
    tokenise_axis(ax_c, grid="both")
    # The key belongs to C alone (B has its own), so it sits under C's axis
    # label, spanning C's width, rather than between the rows.
    handles, labels = ax_c.get_legend_handles_labels()
    ax_c.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        ncol=3,
        frameon=False,
        fontsize=PT_LEGEND,
        handlelength=1.25,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0.0,
    )

    # The point-gate control given the adaptive gains ties the conductance
    # rule exactly in every seed (final loss difference 0, 50/50 ties); it is
    # reported as a number rather than a zero-spread column.
    controls = [
        ("adaptive_global_shunt", "global", COLORS["per_soma"]),
        ("adaptive_shuffled_shunt", "shuffled", COLORS["mute"]),
        ("noisy_no_shunt", "no shunt", COLORS["ink"]),
        ("initial_oracle_shunt", "fixed oracle", COLORS["oracle"]),
    ]
    high_outcomes = outcomes[np.isclose(outcomes.reliability_heterogeneity, 2.0)]
    wide = high_outcomes.pivot(index="seed", columns="method", values="final_test_loss")
    point_gate = (wide["adaptive_point_gate"] - wide["adaptive_local_shunt"]).dropna()
    assert len(point_gate) == 50 and float(np.abs(point_gate).max()) == 0.0
    positions = np.arange(len(controls))
    for index, (control, _, color) in enumerate(controls):
        values = (wide[control] - wide["adaptive_local_shunt"]).to_numpy(float)
        jitter = np.linspace(-0.13, 0.13, len(values))
        ax_d.scatter(index + jitter, values, s=SEED_MS**2, color=color,
                     alpha=SEED_ALPHA * 0.72, edgecolors="none", zorder=1)
        row = contrasts[
            contrasts.left_minus_right.eq(f"adaptive_local_shunt - {control}")
            & contrasts.metric.eq("final_test_loss")
        ].iloc[0]
        mean = -float(row.mean_difference)
        low = -float(row.ci95_high)
        high_ci = -float(row.ci95_low)
        ax_d.errorbar(index, mean, yerr=[[mean - low], [high_ci - mean]], fmt="D",
                      ms=4.0, color=color, markeredgecolor="white",
                      markeredgewidth=LW_HAIR, elinewidth=LW_ERR, capsize=2.0,
                      zorder=3)
    ax_d.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    # Name the zero reference in a short right-hand margin, clear of the last
    # column's seed dots.
    ax_d.set_xlim(-0.5, len(controls) - 0.5 + 0.72)
    ax_d.text(len(controls) - 0.5 + 0.68, 0, "adaptive\nlocal", ha="right",
              va="bottom", fontsize=PT_SMALL, color=COLORS["mute"],
              linespacing=1.1)
    ax_d.set_xticks(positions, [entry[1] for entry in controls])
    ax_d.tick_params(axis="x", labelsize=PT_SMALL, pad=2.0)
    ax_d.set_ylabel("control loss $-$ adaptive-local loss")
    panel_title(ax_d, "D", "Final-loss boundary at high heterogeneity")
    tokenise_axis(ax_d, grid="y")

    style_direct_color_labels(fig)
    # Snap whatever a shared helper still sets off the token set (legend and
    # marker furniture) before the audits read the figure.
    enforce_tokens(fig)
    fig.canvas.draw()
    audit_layout(fig, "fig_adaptive_conductance_reliability")
    audit_text_over_data(fig, "fig_adaptive_conductance_reliability")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    main()

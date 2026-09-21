#!/usr/bin/env python3
"""Render the state-matched positive-conductance reliability experiment."""

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
SOURCE = ROOT / "source_data" / "positive_conductance_reliability_step_consistent"
FIGURES = ROOT / "figures" / "generated"


METHODS = [
    ("noisy_no_shunt", "no shunt", COLORS["ink"], "o"),
    ("best_global_shunt", "best global", COLORS["per_soma"], "s"),
    ("reliability_aligned_shunt", "SNR-aligned", COLORS["shunting"], "D"),
    ("shuffled_shunt", "shuffled", COLORS["mute"], "^"),
    ("anti_aligned_shunt", "anti-aligned", COLORS["additive"], "v"),
]


def schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # The caption names the mechanism as a supplied compensating current; the
    # same name is used here so the sheet has one word for it.
    panel_title(ax, "A", "Supplied state clamp")
    # Three boxes with a gap wide enough for an arrow that touches neither
    # stroke nor glyph; text sits inside with its own padding.
    # The chain spans 0.896 of the axes so the schematic's crop is as wide as
    # the adaptive study's panel C on the consolidated sheet.
    span = 0.896
    width, gap, y0, height = 0.27 * span, 0.095 * span, 0.52, 0.40
    boxes = [
        ("positive\ninput rates\n$x\\geq0$", COLORS["dend"]),
        ("branch\nvoltage $V$", COLORS["oracle"]),
        ("local\neligibility", COLORS["shunting"]),
    ]
    x = 0.0
    for label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.012, y0), width - 0.024, height,
                boxstyle="round,pad=0.012", facecolor="white", edgecolor=color,
                lw=LW_DATA, clip_on=False,
            )
        )
        ax.text(x + width / 2, y0 + height / 2, label, ha="center", va="center",
                fontsize=PT_SMALL, color=color, linespacing=1.25)
        x += width + gap
    y_arrow = y0 + height / 2
    for start in (width, 2 * width + gap):
        ax.add_patch(FancyArrowPatch((start + 0.016, y_arrow),
                                     (start + gap - 0.016, y_arrow),
                                     arrowstyle="-|>", mutation_scale=8,
                                     lw=LW_REF, color=COLORS["mute"], clip_on=False))
    centre = (3 * width + 2 * gap) / 2
    # Both lines were mathtext, whose subscripts printed at 4.90 pt; they are
    # the same words and symbols set as token-size spans with a real dropped
    # subscript instead.
    token_run(ax, centre, 0.34,
              [("shunt \u03ba", False), ("b", True),
               (" \u2265 0  +  clamp current \u03ba", False), ("b", True),
               ("V", False)],
              size=PT_ANNOT, color=COLORS["ink"])
    token_run(ax, centre, 0.16,
              [("V\u2032 = V   but   eligibility \u00d7 G/(G + \u03ba", False),
               ("b", True), (")", False)],
              size=PT_ANNOT, color=COLORS["shunting"])


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


def line_with_interval(
    ax: plt.Axes,
    frame: pd.DataFrame,
    method: str,
    metric: str,
    label: str,
    color: str,
    marker: str,
) -> None:
    part = frame[frame.method.eq(method)].sort_values("reliability_heterogeneity")
    x = part.reliability_heterogeneity.to_numpy(float)
    mean = part[f"mean_{metric}"].to_numpy(float)
    low = part[f"ci95_low_{metric}"].to_numpy(float)
    high = part[f"ci95_high_{metric}"].to_numpy(float)
    ax.plot(x, mean, color=color, marker=marker, ms=3.4, lw=LW_DATA, label=label)
    ax.fill_between(x, low, high, color=color, alpha=0.09, linewidth=0)


def main() -> None:
    apply_neurips_style()
    branch = pd.read_csv(SOURCE / "branch_reliability.csv")
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    contrasts = pd.read_csv(SOURCE / "extended_contrasts.csv")

    # Four column units: the schematic (A) and the paired endpoint panel (F)
    # each take two so their crops match the width of a half-page panel in
    # the consolidated supplement; B-E are single-unit line plots.
    fig = plt.figure(figsize=(FIG_W, 4.95))
    grid = fig.add_gridspec(
        2, 4, left=0.092, right=0.985, bottom=0.115, top=0.90,
        wspace=0.62, hspace=0.95,
    )
    ax_a = fig.add_subplot(grid[0, 0:2])
    ax_b = fig.add_subplot(grid[0, 2])
    ax_c = fig.add_subplot(grid[0, 3])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_e = fig.add_subplot(grid[1, 1])
    ax_f = fig.add_subplot(grid[1, 2:4])
    # F is trimmed by 12.7 pt so its crop is as wide as the adaptive study's
    # panel D on the consolidated sheet (the two are pasted as twins).
    pos = ax_f.get_position()
    ax_f.set_position([pos.x0, pos.y0, pos.width - 12.7 / (FIG_W * 72.0), pos.height])
    schematic(ax_a)

    high = branch[np.isclose(branch.reliability_heterogeneity, 2.0)]
    branch_summary = high.groupby("branch", as_index=False).agg(
        reliability=("optimal_reliability_gain", "mean"),
        reliability_sd=("optimal_reliability_gain", "std"),
        snr=("signal_to_noise", "mean"),
    )
    ax_b.plot(
        branch_summary.branch + 1,
        branch_summary.reliability,
        color=COLORS["shunting"], marker="o", ms=3.4, lw=LW_DATA,
    )
    ax_b.set_xticks(np.arange(1, 9))
    ax_b.set_xlabel("branch index")
    # $a_b^*$ set its subscript and star at 5.60 pt on top of each other;
    # the axis says the same thing in words.
    ax_b.set_ylabel("optimal fixed-step gain")
    panel_title(ax_b, "B", "Step-consistent profile")
    tokenise_axis(ax_b)

    for method, label, color, marker in METHODS:
        line_with_interval(
            ax_c, summary, method, "one_step_test_loss_decrease",
            label, color, marker,
        )
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_c.set_xlabel("SNR heterogeneity")
    ax_c.set_ylabel("one-step test-loss decrease")
    panel_title(ax_c, "C", "Immediate step")
    tokenise_axis(ax_c)

    comparison_styles = [
        ("reliability_aligned_shunt - best_global_shunt", "aligned $-$ global",
         COLORS["shunting"], "o"),
        ("reliability_aligned_shunt - noisy_no_shunt", "aligned $-$ no shunt",
         COLORS["oracle"], "s"),
    ]
    for comparison, label, color, marker in comparison_styles:
        part = contrasts[
            contrasts.left_minus_right.eq(comparison)
            & contrasts.metric.eq("one_step_test_loss_decrease")
        ].sort_values("heterogeneity")
        ax_d.plot(part.heterogeneity, part.mean_difference, color=color, marker=marker,
                  ms=3.4, lw=LW_DATA, label=label)
        ax_d.fill_between(part.heterogeneity, part.ci95_low, part.ci95_high,
                          color=color, alpha=0.10, linewidth=0)
    ax_d.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_d.set_xlabel("SNR heterogeneity")
    ax_d.set_ylabel("paired loss-decrease difference")
    panel_title(ax_d, "D", "Alignment contrast")
    tokenise_axis(ax_d)
    # the 7.0 pt key needs a band of its own: both contrast curves cross every
    # corner of the data box, so the panel is given headroom instead of letting
    # the key sit on a line (no datum moves; only the window grows upward)
    lo, hi = ax_d.get_ylim()
    ax_d.set_ylim(lo, hi + 0.42 * (hi - lo))
    clean_legend(ax_d, fontsize=PT_LEGEND, loc="upper left")

    for method, label, color, marker in METHODS:
        line_with_interval(
            ax_e, summary, method, "final_test_loss", label, color, marker,
        )
    line_with_interval(
        ax_e, summary, "exact_clean_bp", "final_test_loss", "exact clean BP",
        COLORS["oracle"], "*",
    )
    ax_e.set_xlabel("SNR heterogeneity")
    ax_e.set_ylabel("test loss after 40 updates")
    panel_title(ax_e, "E", "Training horizon")
    tokenise_axis(ax_e)
    handles, labels = ax_e.get_legend_handles_labels()
    # The shared key sits between the rows under C and F, to the right of the
    # schematic column, so that the schematic's own crop is not widened by it.
    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.745, 0.505),
        ncol=3,
        frameon=False,
        fontsize=PT_LEGEND,
        handlelength=1.3,
        handletextpad=0.35,
        columnspacing=0.9,
    )

    # The explicit point-gate control ties the aligned conductance rule to
    # machine precision in every seed (final loss difference 0, 50/50 ties); it
    # is reported as a number rather than a zero-spread column.
    controls = [
        ("best_global_shunt", "global", COLORS["per_soma"]),
        ("shuffled_shunt", "shuffled", COLORS["mute"]),
        ("anti_aligned_shunt", "anti-aligned", COLORS["additive"]),
        ("noisy_no_shunt", "no shunt", COLORS["ink"]),
    ]
    values, lows, highs = [], [], []
    for control, _, _ in controls:
        row = contrasts[
            np.isclose(contrasts.heterogeneity, 2.0)
            & contrasts.left_minus_right.eq(
                f"reliability_aligned_shunt - {control}"
            )
            & contrasts.metric.eq("final_test_loss")
        ].iloc[0]
        # Convert aligned-minus-control loss into control-minus-aligned so that
        # positive bars favor aligned shunting.
        values.append(-float(row.mean_difference))
        lows.append(-float(row.ci95_high))
        highs.append(-float(row.ci95_low))
    x = np.arange(len(controls))
    values = np.asarray(values)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    seed_rows = pd.read_csv(SOURCE / "seed_outcomes.csv")
    high = seed_rows[np.isclose(seed_rows.reliability_heterogeneity, 2)]
    wide = high.pivot(index="seed", columns="method", values="final_test_loss")
    point_gate = (wide["explicit_point_gate"] - wide["reliability_aligned_shunt"]).dropna()
    assert len(point_gate) == 50 and float(np.abs(point_gate).max()) == 0.0
    for i, (control, label, color) in enumerate(controls):
        paired = (wide[control] - wide["reliability_aligned_shunt"]).dropna().to_numpy()
        # Translucent seed dots under a white-edged mean diamond, the same
        # grammar as the adaptive study's paired-contrast panel.
        ax_f.scatter(i + np.linspace(-.13, .13, len(paired)), paired,
                     s=SEED_MS**2, color=color, alpha=SEED_ALPHA * 0.72,
                     edgecolors="none", zorder=1)
        ax_f.errorbar(i, values[i], yerr=[[values[i] - lows[i]], [highs[i] - values[i]]],
                      color=color, fmt="D", ms=4.0, markeredgecolor="white",
                      markeredgewidth=LW_HAIR, elinewidth=LW_ERR, capsize=2.0,
                      zorder=3)
    ax_f.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    # Name the zero reference in a short right-hand margin, clear of the last
    # column's seed dots.
    ax_f.set_xlim(-0.5, len(controls) - 0.5 + 0.62)
    ax_f.text(len(controls) - 0.5 + 0.58, 0, "aligned", ha="right", va="bottom",
              fontsize=PT_SMALL, color=COLORS["mute"])
    ax_f.set_xticks(x, [entry[1] for entry in controls])
    ax_f.tick_params(axis="x", labelsize=PT_SMALL, pad=2.0)
    ax_f.set_ylabel("control loss $-$ aligned loss")
    panel_title(ax_f, "F", "Paired endpoint effects")
    tokenise_axis(ax_f, grid="y")

    FIGURES.mkdir(parents=True, exist_ok=True)
    style_direct_color_labels(fig)
    # Snap whatever a shared helper still sets off the token set (legend and
    # marker furniture) before the audits read the figure.
    enforce_tokens(fig)
    fig.canvas.draw()
    audit_layout(fig, "fig_positive_conductance_reliability")
    audit_text_over_data(fig, "fig_positive_conductance_reliability")
    fig.savefig(
        FIGURES / "fig_positive_conductance_reliability.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / "fig_positive_conductance_reliability.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()

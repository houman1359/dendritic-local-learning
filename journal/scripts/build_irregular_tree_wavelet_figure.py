#!/usr/bin/env python3
"""Render the irregular-tree multiscale route-field analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    ORDINAL_RAMP,
    PANEL_LABEL_PT,
    PT_LEGEND,
    SEED_ALPHA,
    SEED_MS,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    panel_title,
    style_axis,
    tint_pct,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "irregular_tree_wavelets"
FIGURE = ROOT / "figures" / "generated" / "fig_irregular_tree_wavelets.pdf"

SCALES = ["coarse", "intermediate", "fine"]
SCALE_LABELS = ["coarse\n$>1/4$", "intermediate\n$1/16$–$1/4$", "fine\n$\\leq1/16$"]
# Green / blue / rose are the SCALE-BIN hues of this figure (A and C).  B and
# D therefore draw nothing in them: B's series are a teal "actual routes" line
# against grey open null marks, and D's two cohorts are a dark and a light
# tint of that same teal, so no hue carries two meanings on the sheet.
SCALE_COLORS = [COLORS["shunting"], COLORS["additive"], COLORS["highlight"]]
ROUTE_COLOR = ORDINAL_RAMP[3]              # 47-cell cohort, actual routes
ORIGINAL_COLOR = tint_pct(ROUTE_COLOR, 40)  # 8-cell original cohort (D)
NULL_COLOR = COLORS["point_mlp"]            # the one grey control hue
INTERVAL_COLOR = COLORS["ink"]              # 95% CI whiskers, drawn over marks

# Cell-bootstrap of the cohort mean, the same percentile procedure that
# produced the *_ci_low/_ci_high columns of cohort_scale_summary.csv (checked:
# it reproduces those columns to 1e-3 for the actual-route series).  The
# source table carries no interval for the two null series, so B computes
# theirs here for display only; no plotted mean changes.
BOOTSTRAP_DRAWS = 20_000   # matches summary.json "bootstrap_draws"
BOOTSTRAP_SEED = 20260911


def cell_bootstrap_ci(values: np.ndarray, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAP_DRAWS, len(values)), replace=True).mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(lo), float(hi)


def compact_panel_title(ax: plt.Axes, letter: str, title: str) -> None:
    """Place the panel letter beside, not across, a vertical axis label."""

    artist = panel_title(ax, letter, title)
    artist.set_visible(False)
    ax.set_title(title, loc="left", x=0.09, pad=8)
    ax.text(0.0, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=PANEL_LABEL_PT, fontweight="bold", color=COLORS["ink"],
            clip_on=False)


def spread(n: int, half_width: float) -> np.ndarray:
    """Deterministic, value-independent horizontal jitter for a strip of n cells."""

    if n == 1:
        return np.zeros(1)
    # Interleave so consecutive rows do not walk left-to-right in table order.
    order = np.argsort((np.arange(n) * 0.6180339887) % 1.0)
    return np.linspace(-half_width, half_width, n)[order]


def mean_with_interval(ax: plt.Axes, x: float, mean: float, lo: float, hi: float,
                       *, marker: str, color: str, filled: bool = True,
                       ms: float = 4.0) -> None:
    """A mean mark with its 95% interval drawn OVER it in ink.

    The interval is on top, with caps wider than the marker, so it stays
    visible even where it is narrower than the marker body.
    """

    face = color if filled else "white"
    ax.plot([x], [mean], marker=marker, ms=ms, color=color, mfc=face,
            mec=color if not filled else "white", mew=LW_HAIR if filled else LW_EDGE,
            ls="none", zorder=3)
    ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                ecolor=INTERVAL_COLOR, elinewidth=LW_ERR, capsize=2.4,
                capthick=LW_ERR, zorder=4)


def draw_tree_basis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    compact_panel_title(ax, "A", "Irregular-tree scale basis")
    # The tree occupies the left ~55% of the panel; the colour key the rest.
    points = {
        0: (0.30, 0.08), 1: (0.29, 0.25), 2: (0.15, 0.45), 3: (0.42, 0.43),
        4: (0.07, 0.68), 5: (0.23, 0.67), 6: (0.36, 0.65), 7: (0.51, 0.65),
        8: (0.02, 0.88), 9: (0.11, 0.86), 10: (0.20, 0.89), 11: (0.28, 0.84),
        12: (0.33, 0.87), 13: (0.40, 0.84), 14: (0.49, 0.88), 15: (0.57, 0.83),
    }
    edges = [(0, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7),
             (4, 8), (4, 9), (5, 10), (5, 11), (6, 12), (6, 13), (7, 14), (7, 15)]
    bin_of = {**{c: 0 for c in (3, 6, 7, 12, 13, 14, 15)},
              **{c: 1 for c in (5, 10, 11)}, **{c: 2 for c in (8, 9)}}
    for parent, child in edges:
        x0, y0 = points[parent]
        x1, y1 = points[child]
        if child in bin_of:
            color, width = SCALE_COLORS[bin_of[child]], LW_DATA
        else:
            color, width = COLORS["mute"], LW_REF
        ax.plot([x0, x1], [y0, y1], color=color, lw=width,
                solid_capstyle="round", transform=ax.transAxes)
    ax.add_patch(Circle(points[0], 0.034, transform=ax.transAxes,
                        fc=COLORS["soma"], ec=COLORS["edge"], lw=LW_EDGE))
    # Colour key: which hue is which bin (descendant excitatory-weight
    # fraction), plus the two uncoloured elements.
    handles = [
        Line2D([], [], color=SCALE_COLORS[0], lw=LW_DATA, label="coarse  $>1/4$"),
        Line2D([], [], color=SCALE_COLORS[1], lw=LW_DATA, label="intermediate  $1/16$–$1/4$"),
        Line2D([], [], color=SCALE_COLORS[2], lw=LW_DATA, label="fine  $\\leq1/16$"),
        Line2D([], [], color=COLORS["mute"], lw=LW_REF, label="trunk to the example subtrees"),
        Line2D([], [], marker="o", ms=5, mfc=COLORS["soma"], mec=COLORS["edge"],
               mew=LW_EDGE, ls="none", label="soma"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.60, 0.46),
              frameon=False, fontsize=PT_LEGEND, handlelength=1.3,
              handletextpad=0.45, labelspacing=0.32, borderaxespad=0.0,
              title="descendant weight fraction", title_fontsize=PT_LEGEND,
              alignment="left")


def token_axis(ax: plt.Axes, grid: str = "none") -> None:
    """``style_axis`` finished on the journal line-weight tokens.

    ``journal_style.style_axis`` still writes the pre-token furniture widths
    (0.8 pt spines and major ticks, 0.6 pt grid rules, and the rc default on
    minor ticks), none of which is a line-weight token, so the strict audit
    reports them.  Re-set exactly those three widths to the tokens the native
    canvas uses for the same roles (``figure_canvas.style_panel``): spines and
    major ticks at ``LW_EDGE``, grid rules and minor ticks at ``LW_HAIR``.
    Nothing here touches data, limits, ticks or their positions.
    """

    style_axis(ax, grid=grid)
    if grid in {"x", "y", "both"}:
        ax.grid(True, axis=grid, zorder=0, linewidth=LW_HAIR, alpha=0.9,
                color=COLORS["grid"])
    ax.tick_params(axis="both", which="major", width=LW_EDGE)
    ax.tick_params(axis="both", which="minor", width=LW_HAIR)
    for spine in ax.spines.values():
        if spine.get_visible():
            spine.set_linewidth(LW_EDGE)


def main() -> None:
    apply_neurips_style()
    cell = pd.read_csv(SOURCE / "cell_scale_summary.csv")
    cohort = pd.read_csv(SOURCE / "cohort_scale_summary.csv")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(FIG_W, 4.45),
        gridspec_kw={"left": 0.095, "right": 0.985, "bottom": 0.105,
                     "top": 0.90, "wspace": 0.42, "hspace": 0.80},
    )
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    draw_tree_basis(ax_a)

    # ---- B: cohort route-energy fraction by scale, actual vs two nulls ----
    primary = cohort[cohort.cohort.eq("public_v661_47")].set_index("scale").loc[SCALES]
    primary_cell = cell[cell.cohort.eq("public_v661_47")]
    x = np.arange(len(SCALES), dtype=float)
    series = [
        # (label, cell column, cohort mean column, dodge, marker, colour, filled)
        ("actual routes", "actual_energy_fraction", "mean_actual_energy_fraction",
         -0.25, "o", ROUTE_COLOR, True),
        ("permuted ancestry", "shuffled_energy_fraction_mean", "mean_shuffled_energy_fraction",
         0.0, "s", NULL_COLOR, False),
        ("isotropic noise", "isotropic_energy_fraction", "mean_isotropic_energy_fraction",
         0.25, "^", NULL_COLOR, False),
    ]
    for label, cell_col, mean_col, dodge, marker, color, filled in series:
        for index, scale in enumerate(SCALES):
            values = primary_cell[primary_cell.scale.eq(scale)][cell_col].to_numpy(float)
            ax_b.scatter(index + dodge + spread(len(values), 0.07), values,
                         s=(SEED_MS * 0.8) ** 2, marker=marker, color=color,
                         alpha=SEED_ALPHA * 0.55, edgecolors="none",
                         rasterized=True, zorder=1)
            row = primary.loc[scale]
            mean = float(row[mean_col])
            if label == "actual routes":
                lo = float(row.actual_energy_fraction_ci_low)
                hi = float(row.actual_energy_fraction_ci_high)
            else:
                lo, hi = cell_bootstrap_ci(values)
            mean_with_interval(ax_b, index + dodge, mean, lo, hi, marker=marker,
                               color=color, filled=filled)
    ax_b.set_xticks(x, SCALE_LABELS)
    ax_b.set_xlim(-0.55, len(SCALES) - 0.45)
    ax_b.set_ylim(0, 0.88)
    ax_b.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
    ax_b.set_ylabel("non-scalar route-energy fraction")
    compact_panel_title(ax_b, "B", "Disjoint same-animal cohort (47 cells)")
    token_axis(ax_b, grid="y")
    key = [
        Line2D([], [], marker="o", ms=4.0, color=ROUTE_COLOR, mec="white", mew=LW_HAIR,
               ls="none", label="actual routes"),
        Line2D([], [], marker="s", ms=4.0, color=NULL_COLOR, mfc="white", mew=LW_EDGE,
               ls="none", label="permuted ancestry"),
        Line2D([], [], marker="^", ms=4.0, color=NULL_COLOR, mfc="white", mew=LW_EDGE,
               ls="none", label="isotropic noise"),
    ]
    # The key sits under B's tick labels, in one row, so nothing is drawn
    # over the data.
    ax_b.legend(handles=key, loc="upper center", bbox_to_anchor=(0.5, -0.30),
                ncol=3, frameon=False, fontsize=PT_LEGEND, handlelength=1.0,
                handletextpad=0.35, columnspacing=0.9, borderaxespad=0.0)

    # ---- C: per-cell signal / isotropic-noise power by scale ----
    for index, (scale, color) in enumerate(zip(SCALES, SCALE_COLORS, strict=True)):
        values = primary_cell[primary_cell.scale.eq(scale)].isotropic_enrichment.to_numpy(float)
        ax_c.scatter(index + spread(len(values), 0.30), values, s=SEED_MS**2,
                     color=color, alpha=SEED_ALPHA * 0.75, edgecolors="none",
                     rasterized=True, zorder=1)
        row = primary.loc[scale]
        mean_with_interval(ax_c, index, float(row.mean_isotropic_enrichment),
                           float(row.isotropic_enrichment_ci_low),
                           float(row.isotropic_enrichment_ci_high),
                           marker="D", color=color, ms=4.0)
    ax_c.axhline(1, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=2)
    ax_c.set_yscale("log", base=2)
    ax_c.set_ylim(2.0 ** -7, 2.0 ** 4.3)
    ax_c.set_yticks([2.0 ** k for k in range(-6, 5, 2)],
                    ["1/64", "1/16", "1/4", "1", "4", "16"])
    ax_c.set_yticks([2.0 ** k for k in range(-7, 5)], minor=True)
    ax_c.set_yticklabels([], minor=True)
    ax_c.set_xticks(x, SCALE_LABELS)
    ax_c.set_xlim(-0.55, len(SCALES) - 0.45)
    ax_c.set_ylabel("signal / isotropic-noise power")
    compact_panel_title(ax_c, "C", "Coarse modes carry excess power")
    token_axis(ax_c, grid="y")

    # ---- D: coarse actual - permuted contrast, both cohorts ----
    cohorts = [("original_8", "original cohort\n(n=8)", ORIGINAL_COLOR, 0.16),
               ("public_v661_47", "disjoint cohort\n(n=47)", ROUTE_COLOR, 0.26)]
    for index, (name, _, color, half_width) in enumerate(cohorts):
        values = cell[cell.cohort.eq(name) & cell.scale.eq("coarse")].actual_minus_shuffled.to_numpy(float)
        ax_d.scatter(index + spread(len(values), half_width), values, s=SEED_MS**2,
                     color=color, alpha=SEED_ALPHA * 0.78 if name != "original_8" else 0.95,
                     edgecolors="none", rasterized=True, zorder=1)
        row = cohort[cohort.cohort.eq(name) & cohort.scale.eq("coarse")].iloc[0]
        mean_with_interval(ax_d, index, float(row.mean_actual_minus_shuffled),
                           float(row.actual_minus_shuffled_ci_low),
                           float(row.actual_minus_shuffled_ci_high),
                           marker="D", color=color, ms=4.0)
    ax_d.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=2)
    ax_d.set_xticks([0, 1], [item[1] for item in cohorts])
    ax_d.set_xlim(-0.6, 1.6)
    ax_d.set_ylabel("actual $-$ permuted coarse energy")
    compact_panel_title(ax_d, "D", "Coarse excess: original vs disjoint cohort")
    token_axis(ax_d, grid="y")

    fig.canvas.draw()
    audit_layout(fig, "fig_irregular_tree_wavelets")
    audit_text_over_data(fig, "fig_irregular_tree_wavelets")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    from panel_letter_layout import finish_panel_letters
    metadata = finish_panel_letters(fig, [(chr(65+i), i//2, i%2, [ax])
                                         for i, ax in enumerate(axes.ravel())])
    fig.savefig(FIGURE, metadata=metadata)
    plt.close(fig)


if __name__ == "__main__":
    main()

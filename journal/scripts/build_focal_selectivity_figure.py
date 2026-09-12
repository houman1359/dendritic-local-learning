#!/usr/bin/env python3
"""Render Supplementary Figure S11 from the frozen passive focal matrix.

2026-09-11 (S29 visual review): ported to the native canvas.  Panel A now
draws every background level (the old render silently plotted the
background-multiplier-0 slice, the one slice in which the R_m = 15,000 curve
is flat) on a symmetric-log y axis, with R_m as the ordinal teal ramp and
background as three column groups; panel B colours the 909 site-regime points
by R_m and encodes background by marker; panel C draws the signed census of
the focal shunt beside its matched current injection, so the panel shows a
contrast rather than a constant.  Only ``source_data/`` tables are read.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

from figure_canvas import (
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
    LW_ERR,
    LW_REF,
    MARKER_MS,
    ORDINAL_RAMP,
    PT_BASE,
    Margins,
    NativeCanvas,
    token_subscript,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "focal_selectivity_phase1"
FIGURES = ROOT / "figures" / "generated"

# R_m is ordinal: one hue, three lightness steps, plus a marker and a dash
# pattern each so the ramp survives a deutan simulation and a grey print.
RM_LEVELS = (300.0, 1000.0, 15000.0)
RM_COLOR = {300.0: ORDINAL_RAMP[0], 1000.0: ORDINAL_RAMP[1],
            15000.0: ORDINAL_RAMP[3]}
RM_MARKER = {300.0: "o", 1000.0: "s", 15000.0: "^"}
RM_DASH = {300.0: (None, None), 1000.0: (4.0, 1.6), 15000.0: (1.6, 1.4)}
RM_LABEL = {300.0: "300", 1000.0: "1,000", 15000.0: "15,000"}
BG_LEVELS = (0.0, 1.0, 4.0)
DOSES = (0.05, 0.5, 5.0)
# ``linthresh`` sits just under the smallest 0.05 nS contrast (0.0005 in
# magnitude) so every drawn mean is on the log part of the axis.
LINTHRESH = 1e-3


def _rm_key_handles():
    return [Line2D([], [], color=RM_COLOR[rm], marker=RM_MARKER[rm],
                   dashes=RM_DASH[rm], lw=LW_DATA, ms=MARKER_MS - 1.0,
                   label=f"Rm {RM_LABEL[rm]}") for rm in RM_LEVELS]


def draw_dose_panel(ax, contrasts):
    """A: shunt minus injection across dose x R_m x background."""
    absolute = contrasts[
        contrasts.dose_scheme.eq("fixed_absolute_ns")
        & contrasts.metric.eq("localization_index")
    ]
    assert len(absolute) == len(RM_LEVELS) * len(BG_LEVELS) * len(DOSES)
    group_w = len(DOSES)
    group_gap = 1.5
    centres = []
    for g, bg in enumerate(BG_LEVELS):
        x0 = g * (group_w + group_gap)
        xs = np.arange(group_w) + x0
        centres.append(xs.mean())
        for rm in RM_LEVELS:
            part = absolute[
                np.isclose(absolute.membrane_resistance_ohm_cm2, rm)
                & np.isclose(absolute.background_leak_multiplier, bg)
            ].sort_values("dose_value")
            assert np.allclose(part.dose_value.to_numpy(), DOSES)
            mean = part.mean_shunt_minus_additive.to_numpy()
            ax.errorbar(
                xs, mean,
                yerr=np.vstack([mean - part.ci95_low.to_numpy(),
                                part.ci95_high.to_numpy() - mean]),
                color=RM_COLOR[rm], marker=RM_MARKER[rm],
                dashes=RM_DASH[rm], lw=LW_DATA, elinewidth=LW_ERR,
                ms=MARKER_MS - 1.0, capsize=ERR_CAPSIZE, zorder=3,
            )
        ax.annotate(f"background ×{bg:g}", xy=(xs.mean(), 1.0),
                    xycoords=("data", "axes fraction"), xytext=(0.0, 2.0),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=PT_BASE, color=COLORS["mute"],
                    annotation_clip=False)
        if g:
            ax.axvline(x0 - group_gap / 2.0, color=COLORS["grid"],
                       lw=LW_REF, zorder=0.5)
    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.8)
    ax.set_ylim(-0.02, 0.3)
    ticks = [-0.01, -0.001, 0.0, 0.001, 0.01, 0.1]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["−0.01", "−0.001", "0", "0.001",
                        "0.01", "0.1"])
    ax.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=1)
    xt = [g * (group_w + group_gap) + i
          for g in range(len(BG_LEVELS)) for i in range(group_w)]
    ax.set_xticks(xt, [f"{d:g}" for d in DOSES] * len(BG_LEVELS))
    ax.set_xlim(-0.7, xt[-1] + 0.7)
    # a little extra label pad keeps this crop the tallest of the three, so
    # the supplement paste keeps A and B on one row (see build.py layout)
    ax.set_xlabel("fixed shunt conductance (nS)", labelpad=7.0)
    ax.set_ylabel("shunt − current-injection\nlocalization")
    ax.legend(handles=_rm_key_handles(), loc="lower right", frameon=False,
              fontsize=PT_BASE, handlelength=2.2, handletextpad=0.5,
              labelspacing=0.25, borderaxespad=0.3)


def draw_selectivity_panel(ax, rows):
    """B: transport selectivity versus localization, 909 site-regime points."""
    selected = rows[
        rows.perturbation.eq("focal shunt")
        & rows.dose_scheme.eq("input_conductance_normalized")
        & np.isclose(rows.dose_value, 1.0)
    ]
    sites = selected.groupby(
        ["root_id", "focal_segment_id", "membrane_resistance_ohm_cm2",
         "background_leak_multiplier"],
        as_index=False,
    )[["transport_selectivity", "localization_index"]].mean()
    assert len(sites) == 909
    bg_style = {
        0.0: dict(marker="o", filled=True),
        1.0: dict(marker="o", filled=False),
        4.0: dict(marker="+", filled=True),
    }
    for rm in RM_LEVELS:
        for bg in BG_LEVELS:
            part = sites[
                np.isclose(sites.membrane_resistance_ohm_cm2, rm)
                & np.isclose(sites.background_leak_multiplier, bg)
            ]
            assert len(part) == 101
            st = bg_style[bg]
            colour = RM_COLOR[rm]
            if st["marker"] == "+":
                ax.scatter(part.transport_selectivity,
                           part.localization_index, s=14, marker="+",
                           color=colour, linewidths=LW_ERR, alpha=0.7,
                           zorder=2)
            elif st["filled"]:
                ax.scatter(part.transport_selectivity,
                           part.localization_index, s=9, marker="o",
                           color=colour, alpha=0.6, edgecolors="none",
                           zorder=2)
            else:
                ax.scatter(part.transport_selectivity,
                           part.localization_index, s=11, marker="o",
                           facecolors="none", edgecolors=colour,
                           linewidths=LW_ERR, alpha=0.7, zorder=2)
    ax.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=1)
    ax.axvline(1, color=COLORS["mute"], ls=":", lw=LW_REF, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.6, 1.2e5)
    ax.xaxis.set_major_locator(FixedLocator([1, 100, 10000]))
    ax.xaxis.set_minor_locator(FixedLocator([10, 1000]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(["1", "100", "10,000"])
    ax.set_ylim(-0.04, 0.74)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("transport selectivity S")
    ax.annotate("k", xy=(1.0, 0.0), xycoords=ax.xaxis.label,
                xytext=(0.4, -1.6), textcoords="offset points",
                fontsize=PT_BASE, color=COLORS["ink"], ha="left",
                va="baseline", annotation_clip=False)
    ax.set_ylabel("localization index")
    # the reference is named at the axis top, outside the point cloud
    token_subscript(ax, 1.0, 1.0, "S", "k", " = 1", size=PT_BASE,
                    color=COLORS["mute"], ha="left", va="bottom",
                    transform=ax.get_xaxis_transform(), clip_on=False)
    handles = [
        Line2D([], [], ls="none", marker="o", color=COLORS["mute"],
               ms=SEED_KEY_MS, label="background ×0"),
        Line2D([], [], ls="none", marker="o", markerfacecolor="none",
               markeredgecolor=COLORS["mute"], markeredgewidth=LW_ERR,
               ms=SEED_KEY_MS, label="×1"),
        Line2D([], [], ls="none", marker="+", color=COLORS["mute"],
               markeredgewidth=LW_ERR, ms=SEED_KEY_MS + 0.8,
               label="×4"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              bbox_to_anchor=(1.0, 0.06), fontsize=PT_BASE,
              handlelength=1.0, handletextpad=0.5, labelspacing=0.25,
              borderaxespad=0.3)


SEED_KEY_MS = 3.4


def draw_signed_panel(ax, summary, rows):
    """C: signed census, focal shunt beside its matched current injection."""
    central = summary[
        summary.dose_scheme.eq("input_conductance_normalized")
        & np.isclose(summary.dose_value, 1.0)
        & np.isclose(summary.membrane_resistance_ohm_cm2, 1000.0)
        & np.isclose(summary.background_leak_multiplier, 1.0)
    ]
    cols = ["mean_descendant_attenuated_fraction",
            "mean_descendant_enhanced_fraction",
            "mean_descendant_sign_flip_fraction"]
    series = [("focal shunt", "shunt", COLORS["shunting"], -0.19),
              ("matched additive", "current injection", COLORS["additive"],
               0.19)]
    for perturbation, label, colour, offset in series:
        rec = central[central.perturbation.eq(perturbation)]
        assert len(rec) == 1
        values = rec.iloc[0][cols].to_numpy(dtype=float)
        ax.bar(np.arange(3) + offset, values, width=0.36, color=colour,
               label=label, zorder=2)
    central_sites = rows[
        rows.dose_scheme.eq("input_conductance_normalized")
        & np.isclose(rows.dose_value, 1.0)
        & np.isclose(rows.membrane_resistance_ohm_cm2, 1000.0)
        & np.isclose(rows.background_leak_multiplier, 1.0)
        & rows.perturbation.eq("focal shunt")
    ]
    assert len(central_sites) == 101 and central_sites.root_id.nunique() == 8
    ax.set_xticks(range(3), ["attenuated", "enhanced", "sign flip"],
                  rotation=20, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel("descendant fraction")
    ax.set_ylim(0, 1.38)
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.legend(loc="upper right", frameon=False, fontsize=PT_BASE,
              handlelength=1.0, handletextpad=0.5, labelspacing=0.25,
              borderaxespad=0.3)


def main() -> None:
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    rows = pd.read_csv(SOURCE / "site_outcomes.csv.gz")
    contrasts = pd.read_csv(SOURCE / "paired_contrasts.csv")
    # One row whose axes box is as tall as the S21 native rows (129.5 pt), so
    # the supplement sheet pastes S11 A and S21 A on one top and one baseline.
    cv = NativeCanvas(
        185.5 / 72.0, 1, hgutter_pt=32, vgutter_pt=32,
        margins=Margins(left=45, right=10, top=22, bottom=30),
    )
    # The supplement paste (scripts/supplement_consolidation/build.py) cuts
    # each panel at its letter's x - 3 pt, so every panel's y decorations
    # must sit right of its own letter: the letters go 24 pt left of their
    # module column and B carries a declared reserve.
    cv.letter_dx = 24.0
    ax_a = cv.panel("A", 0, 0, 5, grid="y")
    ax_b = cv.panel("B", 0, 5, 4, title="Cable selectivity", grid="none")
    ax_c = cv.panel("C", 0, 9, 3, title="Signed outcomes", grid="y")
    cv.declare_reserve("B", left=12.0)
    draw_dose_panel(ax_a, contrasts)
    draw_selectivity_panel(ax_b, rows)
    draw_signed_panel(ax_c, summary, rows)
    FIGURES.mkdir(parents=True, exist_ok=True)
    problems = cv.save(FIGURES / "fig_focal_selectivity_matrix.pdf",
                       name="fig_focal_selectivity_matrix")
    for problem in problems:
        print("  layout:", problem)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Supplementary sheet S29 (ident ``shunt_sensitivity``) -- dose, cable
parameters and input mapping of focal-shunt selectivity -- rebuilt as ONE
native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet is a paste of six panels cropped from two upstream renders
(``figures/generated/fig_focal_selectivity_matrix.pdf`` A-C and
``figures/supplementary/figure_S21_panels_A-D.pdf`` A, C, D) at scale 0.912 --
the last sheet of the thirty-six that cannot be pasted at 1.0, and the only
remaining ``SCALE_EXEMPTIONS`` entry.  This builder draws all six panels on
one canvas at the canonical 518.4 pt width, so the sheet pastes whole at 1.0
and every glyph is a canvas type token (the pasted panel A carried 80 spans of
6.38 pt type, below the 7.0 pt floor).

Panel letters and plotted quantities are unchanged:

* **A** (was S11 A) shunt-minus-injection localization across three fixed
  absolute doses, three membrane resistances and three background multipliers;
* **B** (was S21 A) shunt / matched injection / reassigned-template
  localization, cells paired across the two matched conditions;
* **C** (was S11 B) transport selectivity against localization, 909 site-regime
  points at unit input-conductance-normalized dose;
* **D** (was S11 C) signed census of the same 101 sites, shunt beside its
  matched injection;
* **E** (was S21 D) all mapped versus directly typed contacts, cells paired;
* **F** (was S21 C) E/I scale and inhibitory-reversal sensitivity by row.

Only frozen tables under ``source_data/`` are read
(``focal_selectivity_phase1/`` and ``figure4/``); every printed or plotted
number is asserted against the table it comes from, and the cell means,
intervals and counts are recomputed from the per-cell rows and checked against
the study's recorded summaries.  Nothing computes or trains.

Drawing changes over the pasted sheet (all from
``analysis/figure_visual_review_20260910/mixed_S29.json``; the upstream fixes
of 2026-09-11 are preserved verbatim in the values and in the design):

* the eight per-cell contrasts behind every mean in A are drawn as a fan, with
  the three membrane resistances dodged inside each dose so the fans and the
  intervals do not pile up, and the y axis is tight to those per-cell values;
* C is the widest panel on the sheet (7 modules against the 123 pt it had in
  the paste) and D, whose three bars carry one distinct value per condition,
  is the narrowest of its row;
* the zero bars of D print "0" beside the bar that is not there;
* one meaning per colour across the whole sheet: the teal ordinal ramp is the
  membrane resistance in A and C and nothing else, green is the focal shunt,
  blue the matched current injection, grey the reassigned template and purple
  the direct-typed contact set;
* A and B share one axes-box top and one baseline by construction (one canvas,
  one row), and every gutter on the sheet is the canvas's single 26 pt
  horizontal gutter.

2026-09-23 clarity pass (review_20260923/si_pass): no panel titles, no
reference-rule names ("S_k = 1", "reference interval") and no legend title in
C; sign counts are compact ("8/8 > 0"); axis labels are in sentence case.
Plotted values, limits and colours are unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    ORDINAL_RAMP,
    PT_BASE,
    SEED_ALPHA,
    Margins,
    NativeCanvas,
    tint_patch,
)

ROOT = SCRIPT_DIR.parent
PHASE1 = ROOT / "source_data" / "focal_selectivity_phase1"
FIG4 = ROOT / "source_data" / "figure4"
OUT = ROOT / "figures" / "supplementary" / "figure_shunt_sensitivity_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
SHUNT = COLORS["shunting"]          # green: the focal shunt
INJECT = COLORS["additive"]         # blue: the baseline-current-matched injection
REASSIGN = COLORS["point_mlp"]      # grey: the reassigned relation template
DIRECT = COLORS["pathway"]          # purple: the direct-typed contact set

# R_m is ordinal: one hue, three lightness steps, plus a marker and a dash
# pattern each, so the ramp survives a deutan simulation and a grey print.
RM_LEVELS = (300.0, 1000.0, 15000.0)
RM_COLOR = {300.0: ORDINAL_RAMP[0], 1000.0: ORDINAL_RAMP[1],
            15000.0: ORDINAL_RAMP[3]}
RM_MARKER = {300.0: "o", 1000.0: "s", 15000.0: "^"}
RM_DASH = {300.0: (None, None), 1000.0: (4.0, 1.6), 15000.0: (1.6, 1.4)}
RM_LABEL = {300.0: "300", 1000.0: "1,000", 15000.0: "15,000"}
BG_LEVELS = (0.0, 1.0, 4.0)
DOSES = (0.05, 0.5, 5.0)
# ``linthresh`` sits just under the smallest per-cell contrast that matters
# (1e-4 in magnitude), so every drawn mean and every fan point is resolvable.
LINTHRESH = 1e-3
RM_DODGE = {300.0: -0.24, 1000.0: 0.0, 15000.0: 0.24}

CONTRAST_LABEL = "Shunt − current-injection\nlocalization"
CONTRAST_LABEL_1 = "Shunt − current-injection localization"
CENTRAL = dict(rm=1000.0, bg=1.0)   # the census condition of panel D


# ── shared helpers ───────────────────────────────────────────────────────
def interval(values, seed):
    """``build_supplementary_figure_s21_native.interval`` verbatim.

    Same seeds as the upstream render, so B, E and F keep the intervals the
    accepted 2026-09-11 panels drew.
    """
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    lo, hi = np.quantile(draws, [.025, .975])
    return float(values.mean()), float(lo), float(hi)


def paired(ax, frames, colors, labels, *, seeds, pair=None, xpos=None):
    """Per-cell points, a light pairing line and a mean ± 95 % CI diamond."""
    xpos = list(range(len(frames))) if xpos is None else list(xpos)
    pair = list(range(len(frames))) if pair is None else list(pair)
    cells = np.column_stack([np.asarray(f, float) for f in frames])
    for row in cells:
        ax.plot([xpos[i] for i in pair], [row[i] for i in pair],
                color=MUTE, lw=LW_HAIR, alpha=.4, zorder=1)
    stats = []
    for i, (vals, color, seed) in enumerate(zip(cells.T, colors, seeds)):
        mean, lo, hi = interval(vals, seed)
        stats.append((mean, lo, hi))
        ax.scatter(xpos[i] + np.linspace(-.06, .06, len(vals)), vals, s=12,
                   color=color, alpha=SEED_ALPHA + .05, zorder=2,
                   edgecolors="none")
        ax.errorbar(xpos[i], mean, yerr=[[mean - lo], [hi - mean]], fmt="D",
                    ms=3.8, color=color, capsize=ERR_CAPSIZE, lw=LW_ERR,
                    zorder=3)
    ax.set_xticks(xpos, labels)
    ax.axhline(0, color=MUTE, ls="--", lw=LW_REF)
    return cells, stats


# ── A: dose × membrane resistance × background ───────────────────────────
def panel_dose(ax, contrasts, percell):
    """Shunt minus matched injection at three fixed absolute doses."""
    absolute = contrasts[contrasts.dose_scheme.eq("fixed_absolute_ns")
                         & contrasts.metric.eq("localization_index")]
    assert len(absolute) == len(RM_LEVELS) * len(BG_LEVELS) * len(DOSES) == 27
    assert set(absolute.n_cells) == {8}

    group_w, group_gap = len(DOSES), 1.9
    fan_min, fan_max = np.inf, -np.inf
    for g, bg in enumerate(BG_LEVELS):
        x0 = g * (group_w + group_gap)
        xs = np.arange(group_w) + x0
        for rm in RM_LEVELS:
            part = absolute[
                np.isclose(absolute.membrane_resistance_ohm_cm2, rm)
                & np.isclose(absolute.background_leak_multiplier, bg)
            ].sort_values("dose_value")
            assert np.allclose(part.dose_value.to_numpy(), DOSES)
            mean = part.mean_shunt_minus_additive.to_numpy()
            lo = part.ci95_low.to_numpy()
            hi = part.ci95_high.to_numpy()
            xd = xs + RM_DODGE[rm]
            # the eight per-cell contrasts behind every mean
            for xi, dose in zip(xd, DOSES):
                cells = percell[
                    np.isclose(percell.membrane_resistance_ohm_cm2, rm)
                    & np.isclose(percell.background_leak_multiplier, bg)
                    & np.isclose(percell.dose_value, dose)
                ].contrast.to_numpy(float)
                assert len(cells) == 8
                fan_min, fan_max = min(fan_min, cells.min()), max(fan_max, cells.max())
                ax.plot(xi + np.linspace(-.09, .09, len(cells)), cells,
                        linestyle="none", marker="o", markersize=2.1,
                        markerfacecolor=RM_COLOR[rm], markeredgecolor="none",
                        alpha=0.45, zorder=1.6)
            ax.errorbar(xd, mean, yerr=np.vstack([mean - lo, hi - mean]),
                        color=RM_COLOR[rm], marker=RM_MARKER[rm],
                        dashes=RM_DASH[rm], lw=LW_DATA, elinewidth=LW_ERR,
                        ms=MARKER_MS - 1.4, capsize=ERR_CAPSIZE, zorder=3)
        # the group name sits under its own dose ticks, not over the panel
        ax.annotate(f"background ×{bg:g}", xy=(xs.mean(), 0.0),
                    xycoords=("data", "axes fraction"), xytext=(0.0, -11.5),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=PT_BASE, color=MUTE, annotation_clip=False)
        if g:
            ax.axvline(x0 - group_gap / 2.0, color=COLORS["grid"], lw=LW_REF,
                       zorder=0.5)
    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.8)
    # tight to the per-cell fan (-0.042 .. 0.179), not to a round number
    ax.set_ylim(-0.065, 0.26)   # ~2 pt of white under the lowest per-cell point
    ticks = [-0.01, -0.001, 0.0, 0.001, 0.01, 0.1]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["−0.01", "−0.001", "0", "0.001", "0.01", "0.1"])
    ax.axhline(0, color=MUTE, ls="--", lw=LW_REF, zorder=1)
    xt = [g * (group_w + group_gap) + i
          for g in range(len(BG_LEVELS)) for i in range(group_w)]
    ax.set_xticks(xt, [f"{d:g}" for d in DOSES] * len(BG_LEVELS))
    ax.set_xlim(-0.85, xt[-1] + 0.85)
    ax.set_xlabel("Fixed shunt conductance (nS)", labelpad=12.0)
    ax.set_ylabel(CONTRAST_LABEL)
    handles = [Line2D([], [], color=RM_COLOR[rm], marker=RM_MARKER[rm],
                      dashes=RM_DASH[rm], lw=LW_DATA, ms=MARKER_MS - 1.4,
                      label=f"Rm {RM_LABEL[rm]}") for rm in RM_LEVELS]
    ax.legend(handles=handles, loc="lower right", frameon=False,
              fontsize=PT_BASE, handlelength=2.2, handletextpad=0.5,
              labelspacing=0.22, borderaxespad=0.25)
    print(f"[A] 27 conditions; per-cell fan {fan_min:+.4f} .. {fan_max:+.4f}; "
          f"means {absolute.mean_shunt_minus_additive.min():+.4f} .. "
          f"{absolute.mean_shunt_minus_additive.max():+.4f}")
    return fan_min, fan_max


# ── B: within-cell controls ──────────────────────────────────────────────
def panel_controls(ax, primary, summary):
    cols = ["focal_shunt_localization", "matched_additive_localization",
            "shunt_depth_shuffled_localization"]
    cells, stats = paired(
        ax, [primary[k] for k in cols], [SHUNT, INJECT, REASSIGN],
        ["shunt", "current\ninjection", "reassigned"],
        seeds=(442, 440, 441), pair=[0, 1], xpos=[0, 1, 2.35])
    ref = summary["primary_contrast"]
    np.testing.assert_allclose(stats[0][0], ref["mean_shunt_localization_index"],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(stats[1][0],
                               ref["mean_matched_additive_localization_index"],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        stats[2][0],
        ref["depth_shuffled_relation_control"]["mean_shunt_depth_shuffled_localization"],
        rtol=0, atol=1e-12)
    assert int((cells[:, 0] > cells[:, 1]).sum()) == ref["cells_positive"] == 8
    ax.set_xlim(-.5, 2.85)
    ax.set_ylim(-0.004, 0.152)
    ax.set_yticks([0, 0.05, 0.10, 0.15], ["0", "0.05", "0.10", "0.15"])
    ax.set_ylabel("Localization index")
    print(f"[B] shunt {stats[0][0]:.4f} [{stats[0][1]:.4f}, {stats[0][2]:.4f}], "
          f"injection {stats[1][0]:.4f} [{stats[1][1]:.4f}, {stats[1][2]:.4f}], "
          f"reassigned {stats[2][0]:.4f} [{stats[2][1]:.4f}, {stats[2][2]:.4f}]")


# ── C: transport selectivity against localization ────────────────────────
BG_KEY_MS = 3.4


def panel_selectivity(ax, rows):
    selected = rows[rows.perturbation.eq("focal shunt")
                    & rows.dose_scheme.eq("input_conductance_normalized")
                    & np.isclose(rows.dose_value, 1.0)]
    sites = selected.groupby(
        ["root_id", "focal_segment_id", "membrane_resistance_ohm_cm2",
         "background_leak_multiplier"], as_index=False,
    )[["transport_selectivity", "localization_index"]].mean()
    assert len(sites) == 909
    assert sites.groupby(["root_id", "focal_segment_id"]).ngroups == 101
    for rm in RM_LEVELS:
        for bg in BG_LEVELS:
            part = sites[np.isclose(sites.membrane_resistance_ohm_cm2, rm)
                         & np.isclose(sites.background_leak_multiplier, bg)]
            assert len(part) == 101
            colour = RM_COLOR[rm]
            if bg == 4.0:
                ax.scatter(part.transport_selectivity, part.localization_index,
                           s=13, marker="+", color=colour, linewidths=LW_ERR,
                           alpha=0.7, zorder=2)
            elif bg == 0.0:
                ax.scatter(part.transport_selectivity, part.localization_index,
                           s=9, marker="o", color=colour, alpha=0.6,
                           edgecolors="none", zorder=2)
            else:
                ax.scatter(part.transport_selectivity, part.localization_index,
                           s=11, marker="o", facecolors="none",
                           edgecolors=colour, linewidths=LW_ERR, alpha=0.7,
                           zorder=2)
    ax.axhline(0, color=MUTE, ls="--", lw=LW_REF, zorder=1)
    ax.axvline(1, color=MUTE, ls=":", lw=LW_REF, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.62, 1.1e5)
    ax.xaxis.set_major_locator(FixedLocator([1, 100, 10000]))
    ax.xaxis.set_minor_locator(FixedLocator([10, 1000]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels(["1", "100", "10,000"])
    ax.set_ylim(-0.03, 0.73)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("Transport selectivity S")
    ax.annotate("k", xy=(1.0, 0.0), xycoords=ax.xaxis.label, xytext=(0.4, -1.6),
                textcoords="offset points", fontsize=PT_BASE, color=INK,
                ha="left", va="baseline", annotation_clip=False)
    ax.set_ylabel("Localization index")
    # the dotted S_k = 1 rule is named in the legend text, not in the artwork
    handles = [
        Line2D([], [], ls="none", marker="o", color=MUTE, ms=BG_KEY_MS,
               label="background ×0"),
        Line2D([], [], ls="none", marker="o", markerfacecolor="none",
               markeredgecolor=MUTE, markeredgewidth=LW_ERR, ms=BG_KEY_MS,
               label="×1"),
        Line2D([], [], ls="none", marker="+", color=MUTE,
               markeredgewidth=LW_ERR, ms=BG_KEY_MS + 0.8, label="×4"),
    ]
    # colour is R_m as in A: the legend text says so, the key names markers only
    ax.legend(handles=handles, loc="lower right", frameon=False,
              bbox_to_anchor=(1.0, 0.04), fontsize=PT_BASE, handlelength=1.0,
              handletextpad=0.5, labelspacing=0.22, borderaxespad=0.3)
    stratum = sites.groupby(["membrane_resistance_ohm_cm2",
                             "background_leak_multiplier"])
    print(f"[C] {len(sites)} site-regime points, 101 per stratum; "
          f"S_k {sites.transport_selectivity.min():.2f}-"
          f"{sites.transport_selectivity.max():.0f}; stratum medians "
          f"{np.array2string(stratum.transport_selectivity.median().to_numpy(), precision=2)}")


# ── D: signed census ─────────────────────────────────────────────────────
CENSUS_COLS = ("mean_descendant_attenuated_fraction",
               "mean_descendant_enhanced_fraction",
               "mean_descendant_sign_flip_fraction")
SITE_COLS = ("descendant_attenuated_fraction", "descendant_enhanced_fraction",
             "descendant_sign_flip_fraction")


def panel_census(ax, summary, rows):
    central = summary[summary.dose_scheme.eq("input_conductance_normalized")
                      & np.isclose(summary.dose_value, 1.0)
                      & np.isclose(summary.membrane_resistance_ohm_cm2, CENTRAL["rm"])
                      & np.isclose(summary.background_leak_multiplier, CENTRAL["bg"])]
    series = [("focal shunt", "shunt", SHUNT, -0.19),
              ("matched additive", "current injection", INJECT, 0.19)]
    for perturbation, label, colour, offset in series:
        rec = central[central.perturbation.eq(perturbation)]
        assert len(rec) == 1
        values = rec.iloc[0][list(CENSUS_COLS)].to_numpy(dtype=float)
        # the same 101 sites the panel names: the cell means are the site
        # values exactly, so the bar height is the census, not an average
        sites = rows[rows.dose_scheme.eq("input_conductance_normalized")
                     & np.isclose(rows.dose_value, 1.0)
                     & np.isclose(rows.membrane_resistance_ohm_cm2, CENTRAL["rm"])
                     & np.isclose(rows.background_leak_multiplier, CENTRAL["bg"])
                     & rows.perturbation.eq(perturbation)]
        assert len(sites) == 101 and sites.root_id.nunique() == 8
        np.testing.assert_allclose(sites[list(SITE_COLS)].to_numpy(float).mean(axis=0),
                                   values, rtol=0, atol=1e-12)
        ax.bar(np.arange(3) + offset, values, width=0.34, color=colour,
               label=label, zorder=2)
        for i, value in enumerate(values):
            # a zero bar draws nothing: say so beside the place it would be
            ax.annotate("1.00" if value == 1 else "0",
                        xy=(i + offset, value), xycoords="data",
                        xytext=(0.0, 2.0), textcoords="offset points",
                        ha="center", va="bottom", fontsize=PT_BASE,
                        color=colour, alpha=1.0 if value else 0.75)
    ax.set_xticks(range(3), ["attenuated", "enhanced", "sign flip"])
    ax.set_xlim(-0.62, 2.62)
    ax.set_ylabel("Descendant-gradient fraction")
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.legend(loc="upper center", frameon=False, fontsize=PT_BASE,
              handlelength=1.0, handletextpad=0.5, labelspacing=0.22,
              borderaxespad=0.1, ncol=2, columnspacing=1.1)


# ── E: all mapped versus directly typed contacts ─────────────────────────
def panel_contacts(ax, primary, direct, summary, direct_summary):
    assert list(primary.root_id) == list(direct.root_id)
    cells, stats = paired(ax, [primary.shunt_minus_additive,
                               direct.shunt_minus_additive],
                          [SHUNT, DIRECT], ["all mapped", "direct typed"],
                          seeds=(510, 511))
    rises = int((cells[:, 1] > cells[:, 0]).sum())
    assert rises == len(cells) == 8
    np.testing.assert_allclose(stats[0][0],
                               summary["primary_contrast"]["mean_shunt_minus_additive"],
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(stats[1][0],
                               direct_summary["primary_contrast"]["mean_shunt_minus_additive"],
                               rtol=0, atol=1e-12)
    # the drawn intervals are the same bootstrap the upstream panel drew; they
    # agree with the study's recorded cell-bootstrap intervals to resampling
    # noise, which is what a 10,000-draw bootstrap of eight cells carries
    for stat, rec in ((stats[0], summary), (stats[1], direct_summary)):
        recorded = rec["primary_contrast"]["cell_bootstrap_ci95"]
        np.testing.assert_allclose(stat[1:], recorded, rtol=0, atol=1.0e-3)
    ax.set_xlim(-.45, 1.45)
    ax.set_ylim(-0.006, 0.188)
    ax.set_yticks([0, 0.05, 0.10, 0.15], ["0", "0.05", "0.10", "0.15"])
    ax.set_ylabel(CONTRAST_LABEL)
    ax.annotate(f"{rises}/{len(cells)} increase", xy=(0.03, 0.97),
                xycoords="axes fraction", ha="left", va="top",
                fontsize=PT_BASE, color=MUTE)
    print(f"[E] all mapped {stats[0][0]:.4f} [{stats[0][1]:.4f}, {stats[0][2]:.4f}], "
          f"direct typed {stats[1][0]:.4f} [{stats[1][1]:.4f}, {stats[1][2]:.4f}]; "
          f"{rises}/8 cells increase")


# ── F: synaptic scales and inhibitory reversal ───────────────────────────
SENSITIVITY = (("scale0p1_summary.json", "E/I 0.10\nreversal −0.2", 0.0,
                0.1, -0.2),
               ("summary.json", "E/I 0.35\nreversal −0.2", 1.0, 0.35, -0.2),
               ("scale1p0_summary.json", "E/I 1.00\nreversal −0.2", 2.0,
                1.0, -0.2),
               ("irevm0p5_summary.json", "E/I 0.35\nreversal −0.5", 3.45,
                0.35, -0.5),
               ("irev0_summary.json", "E/I 0.35\nreversal 0.0", 4.45,
                0.35, 0.0))


def panel_sensitivity(ax, cv):
    rows = []
    for name, label, y, e_scale, reversal in SENSITIVITY:
        blob = json.loads((FIG4 / name).read_text())
        rec = blob["primary_contrast"]
        params = blob["parameters"]
        # the row label states the two parameters this row holds
        assert np.isclose(float(params["e_scale"]), e_scale)
        assert np.isclose(float(params["i_scale"]), e_scale)
        assert np.isclose(float(params["inhibitory_reversal"]), reversal)
        rows.append(dict(label=label, y=y,
                         mean=rec["mean_shunt_minus_additive"],
                         lo=rec["cell_bootstrap_ci95"][0],
                         hi=rec["cell_bootstrap_ci95"][1],
                         positive=int(rec["cells_positive"]),
                         n=int(rec["n_cells"])))
    ref = rows[1]
    assert ref["label"].startswith("E/I 0.35\nreversal −0.2")
    assert [r["positive"] for r in rows] == [8, 8, 7, 7, 8]
    assert {r["n"] for r in rows} == {8}
    xlo = -0.012
    xhi = 1.05 * max(r["hi"] for r in rows)
    assert xhi > max(r["hi"] for r in rows)          # every cap inside the axis
    # the band's label is gone (the legend names the reference interval), so
    # the rows keep the same 0.62 of headroom above as below
    ytop, ybot = -0.62, rows[-1]["y"] + 0.62
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ybot, ytop)
    ax.set_yticks([])
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)
    tint_patch(ax, ("rect", ref["lo"], ytop, ref["hi"] - ref["lo"], ybot - ytop),
               color=SHUNT, pct=14, edge=False, radius_pt=0.0, zorder=0.3,
               clip_on=True)
    ax.axvline(0, color=MUTE, ls="--", lw=LW_REF, zorder=1)
    for r in rows:
        y = r["y"]
        ax.plot([xlo, xlo], [y - .3, y + .3], color=COLORS["edge"], lw=LW_HAIR,
                clip_on=False, zorder=1.5, solid_capstyle="butt")
        ax.errorbar(r["mean"], y,
                    xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]],
                    fmt="o", ms=3.8, color=SHUNT, capsize=ERR_CAPSIZE,
                    lw=LW_ERR, zorder=3,
                    markerfacecolor="white" if r is ref else SHUNT,
                    markeredgewidth=LW_ERR)
        ax.annotate(r["label"], xy=(0.0, y), xycoords=("axes fraction", "data"),
                    xytext=(-4.0, 0.0), textcoords="offset points", ha="right",
                    va="center", fontsize=PT_BASE, color=INK, linespacing=1.15,
                    annotation_clip=False)
        ax.annotate(f"{r['positive']}/{r['n']} > 0", xy=(1.0, y),
                    xycoords=("axes fraction", "data"), xytext=(3.0, 0.0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=PT_BASE, color=MUTE, annotation_clip=False)
    ax.set_xticks([0, .03, .06, .09], ["0", "0.03", "0.06", "0.09"])
    ax.set_xlabel(CONTRAST_LABEL_1)
    ax.tick_params(axis="y", length=0)
    # the "8/8 > 0" column (21.6 pt + 3 pt offset) fits a 26 pt reserve; the
    # reserve is locked on the whole right column edge, so B and D share it
    cv.declare_reserve("F", left=44.0, right=26.0)
    print("[F] " + "; ".join(
        f"{r['label'].replace(chr(10), ' ')} {r['mean']:.4f} "
        f"[{r['lo']:.4f}, {r['hi']:.4f}] {r['positive']}/{r['n']}"
        for r in rows) + f"; xlim {xlo:.3f}..{xhi:.4f}")
    return rows


# ── cross-panel assertions against the frozen tables ─────────────────────
def check_tables(contrasts, metrics, summary, sites, primary, direct):
    """Recompute every summary this sheet draws from the per-cell rows."""
    # A: the 27 drawn means and the cells-positive counts are the per-cell
    # shunt-minus-injection contrasts of cell_condition_metrics.csv
    wide = metrics[metrics.dose_scheme.eq("fixed_absolute_ns")].pivot_table(
        index=["root_id", "membrane_resistance_ohm_cm2",
               "background_leak_multiplier", "dose_value"],
        columns="perturbation", values="localization_index")
    contrast = (wide["focal shunt"] - wide["matched additive"]).rename("contrast")
    percell = contrast.reset_index()
    grouped = percell.groupby(["membrane_resistance_ohm_cm2",
                               "background_leak_multiplier", "dose_value"])
    recomputed = grouped.contrast.agg(["mean", "size",
                                       lambda s: int((s > 0).sum())])
    recomputed.columns = ["mean", "n", "positive"]
    absolute = contrasts[contrasts.dose_scheme.eq("fixed_absolute_ns")
                         & contrasts.metric.eq("localization_index")]
    joined = absolute.set_index(["membrane_resistance_ohm_cm2",
                                 "background_leak_multiplier",
                                 "dose_value"]).join(recomputed)
    assert len(joined) == 27 and joined["mean"].notna().all()
    np.testing.assert_allclose(joined.mean_shunt_minus_additive.to_numpy(),
                               joined["mean"].to_numpy(), rtol=0, atol=1e-9)
    assert (joined.cells_positive.to_numpy() == joined.positive.to_numpy()).all()
    assert (joined.n_cells.to_numpy() == joined["n"].to_numpy()).all()
    assert (joined.ci95_low <= joined.mean_shunt_minus_additive).all()
    assert (joined.mean_shunt_minus_additive <= joined.ci95_high).all()

    # D: the census is one distinct value per perturbation, in every one of
    # the 81 passive conditions, not only in the one the panel draws
    for perturbation, expected in (("focal shunt", (1.0, 0.0, 0.0)),
                                   ("matched additive", (0.0, 1.0, 0.0))):
        block = summary[summary.perturbation.eq(perturbation)]
        assert len(block) == 81
        values = block[list(CENSUS_COLS)].to_numpy(float)
        assert np.array_equal(np.unique(values, axis=0), np.array([expected]))
        site_block = sites[sites.perturbation.eq(perturbation)
                           & sites.dose_scheme.eq("input_conductance_normalized")
                           & np.isclose(sites.dose_value, 1.0)
                           & np.isclose(sites.membrane_resistance_ohm_cm2, CENTRAL["rm"])
                           & np.isclose(sites.background_leak_multiplier, CENTRAL["bg"])]
        assert len(site_block) == 101 and site_block.root_id.nunique() == 8
        assert np.array_equal(
            np.unique(site_block[list(SITE_COLS)].to_numpy(float), axis=0),
            np.array([expected]))

    # B, E: the per-cell contrast columns are the differences they claim
    for table in (primary, direct):
        assert len(table) == 8 and table.root_id.nunique() == 8
        np.testing.assert_allclose(
            table.shunt_minus_additive.to_numpy(),
            (table.focal_shunt_localization
             - table.matched_additive_localization).to_numpy(),
            rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            table.shunt_topology_minus_depth_shuffle.to_numpy(),
            (table.focal_shunt_localization
             - table.shunt_depth_shuffled_localization).to_numpy(),
            rtol=0, atol=1e-12)
    print(f"[tables] 27 A conditions, 81 census conditions and both eight-cell "
          f"contrast tables recomputed from the per-cell rows; "
          f"max |A error| {np.abs(joined.mean_shunt_minus_additive.to_numpy() - joined['mean'].to_numpy()).max():.2e}")
    return percell


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 489.0
HGUTTER_PT = 26.0
VGUTTER_PT = 34.0
MARGINS = Margins(left=46.0, right=10.0, top=15.0, bottom=33.0)


def build(path: Path = OUT):
    contrasts = pd.read_csv(PHASE1 / "paired_contrasts.csv")
    metrics = pd.read_csv(PHASE1 / "cell_condition_metrics.csv")
    summary = pd.read_csv(PHASE1 / "condition_summary.csv")
    sites = pd.read_csv(PHASE1 / "site_outcomes.csv.gz")
    primary = pd.read_csv(FIG4 / "cell_primary_contrasts.csv")
    direct = pd.read_csv(FIG4 / "direct_typed_cell_primary_contrasts.csv")
    fig4_summary = json.loads((FIG4 / "summary.json").read_text())
    direct_summary = json.loads((FIG4 / "direct_typed_summary.json").read_text())
    assert fig4_summary["n_cells"] == 8 and fig4_summary["n_focal_sites"] == 101
    percell = check_tables(contrasts, metrics, summary, sites, primary, direct)

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 3, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS)
    # 2026-09-23 clarity pass: no panel titles; what they said is in the legend
    ax_a = cv.panel("A", 0, 0, 7, grid="y")
    ax_b = cv.panel("B", 0, 7, 5, grid="y")
    ax_c = cv.panel("C", 1, 0, 7, grid="none")
    ax_d = cv.panel("D", 1, 7, 5, grid="y")
    ax_e = cv.panel("E", 2, 0, 5, grid="y")
    ax_f = cv.panel("F", 2, 5, 7, grid="x")
    for name in "ABCDE":
        cv.declare_reserve(name, left=14.0, right=9.0)

    panel_dose(ax_a, contrasts, percell)
    panel_controls(ax_b, primary, fig4_summary)
    panel_selectivity(ax_c, sites)
    panel_census(ax_d, summary, sites)
    panel_contacts(ax_e, primary, direct, fig4_summary, direct_summary)
    panel_sensitivity(ax_f, cv)

    problems = cv.save(path, name="figure_shunt_sensitivity_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

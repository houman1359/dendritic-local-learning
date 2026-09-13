#!/usr/bin/env python3
"""Supplementary sheet S30 (ident ``shunt_replication``) -- focal-shunt
controls replicate structurally and under weak-channel linearization --
rebuilt as ONE native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/shunt_replication.pdf``)
pastes three panels of the legacy S10 render (``scripts/
build_microns_v661_replication_figure.py`` F, G, H) above two panels of the
frozen S47 render ``figure_shunt_weak_channels.pdf``, which has no generator
in this repository.  This builder reads ONLY the frozen tables under
``source_data/microns_v661_replication``, ``source_data/
focal_selectivity_active_ensemble`` and ``source_data/focal_selectivity_
phase1`` and redraws the same five panels with the same plotted quantities.
Nothing about the numbers changes: every printed or plotted value is
asserted against the table it comes from, the A/B intervals are recomputed
with the S10 builder's own bootstrap (``mean_ci``: 20,000 cell draws,
``default_rng(3000 + k)`` / ``default_rng(4000 + k)``, replicated verbatim
because that module imports a second analysis module at import time), and
every mean the study tabulates is recomputed from its per-cell rows.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
mixed_S30.json), panel by panel:

* A, B, C share one row of three equal four-module panels (C no longer sits
  alone in a full-width band), with common left/right edges, letters on the
  module grid and one type scale for the whole sheet.  A and B keep their
  own letters (main.tex and si_08_shunting.tex cite S30A,B and S30D,E), share
  one y range, and B's tick label is set natively as 'true relation'; the
  cells-positive counts are set in the band above each plot box, off the
  data field.  B's true-relation column repeats A's focal-shunt values for
  its forty cells (asserted; stated in the caption).
* C keeps the earlier pass's fixes: selected (capped) sites on the y axis,
  cell classes on the ordinal ramp with distinct markers and counts in the
  key; the key now sits in the band above the axes, in the finding's order
  L2IT, L3IT, L4IT, L5ET, so the axis is tight to the data.
* D draws the eight per-cell values at every dose as a fan behind each
  mean (the cell spread the old panel hid), keeps the mean +/- 95 % whiskers
  and the two coloured direct labels, drops the grey method notes from the
  plot box, and replaces the invisible dashed passive curve by an inset on
  its own scale: active minus passive localization per cell and per dose,
  from the phase-1 table at the same calibration (R_m = 1,000 ohm cm^2,
  background = leak, dose normalized to local input conductance), with the
  mean difference per dose.  Both axes of the sheet name the dose as
  'dose / local input conductance'.
* E is a three-row categorical forest (half-row padding, no log padding):
  the eight cells are a vertical fan centred on the row behind the hollow
  diamond and its 95 % whisker, the zero rule is labelled, every row prints
  its '8/8 positive' count, and the row height matches D by construction.
* Second pass (checker report mixed_wave_reports.json, S30): D prints the
  direction of the effect again, 'attenuates descendants' under the focal-
  shunt label and 'enhances descendants' under the current-injection label
  (mute ink, in the whitespace beside each curve, measured clear of every
  line, fan, whisker and rule after the reserve lock); the 'focal shunt'
  label is lowered off the top of the data box; the inset names its axes
  ('dose'; 'active − passive localization, per cell'), has an opaque face
  so D's y = 1.0 grid rule no longer shows through it, and labels its y
  ticks at 0 and 0.10 (minor tick at 0.05) so no inset label sits on a
  grid rule of D; A/B's zero rule is a
  hairline behind the summary diamond so the marker reads on top of it; E's
  per-row count column leaves the data field for the band above the axes
  ('8/8 cells positive at every dose') and the axis ends at 1.25.
* One palette register: focal shunt green (``shunting``), the matched
  current injection the additive blue (``additive``, main Fig. 8's hue for
  this control), the reassigned relation the rose permutation-control slot
  (``highlight``), cell classes on ``ORDINAL_RAMP`` + ink.  No colour
  carries a second meaning anywhere on the sheet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

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
    PT_BASE,
    PT_EMPH,
    SEED_ALPHA,
    SEED_MS,
    Margins,
    NativeCanvas,
    style_panel,
)
from journal_style import ORDINAL_RAMP, style_direct_color_labels  # noqa: E402

ROOT = SCRIPT_DIR.parent
MICRONS = ROOT / "source_data" / "microns_v661_replication"
ACTIVE = ROOT / "source_data" / "focal_selectivity_active_ensemble"
PHASE1 = ROOT / "source_data" / "focal_selectivity_phase1"
OUT = ROOT / "figures" / "supplementary" / "figure_shunt_replication_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
SHUNT = COLORS["shunting"]          # focal shunt
INJECT = COLORS["additive"]         # baseline-current-matched injection
REASSIGN = COLORS["highlight"]      # reassigned relation template (permutation control)
DASHED = (0, (2.6, 2.0))
M_SHUNT, M_INJECT, M_CONTRAST = "o", "s", "D"
MEAN_MS = MARKER_MS + 0.6
DOSES = (0.25, 1.0, 4.0)
DOSE_LABELS = ("0.25", "1", "4")
DOSE_AXIS = "dose / local input conductance"
LOCAL_LABEL = "localization index"
# The passive reference of the frozen S47 sheet: the phase-1 passive matrix
# at the active ensemble's own electrical calibration (main Fig. 8 E).
PASSIVE_CALIBRATION = {
    "membrane_resistance_ohm_cm2": 1000,
    "background_leak_multiplier": 1,
    "dose_scheme": "input_conductance_normalized",
}
# cell classes of C: colour (ordinal ramp + ink), marker, in the key's order
CLASS_STYLE = (("L2IT", ORDINAL_RAMP[1], "o"), ("L3IT", ORDINAL_RAMP[2], "s"),
               ("L4IT", ORDINAL_RAMP[3], "^"), ("L5ET", INK, "D"))
SITE_CAP = 16
E_XMAX = 1.25                       # E's axis end (the dose-4 fan tops out at 1.12)
TITLE_PAD_KEYED = 24.0              # row-1 titles lifted over the two-row key / count band
TITLE_PAD_BAND = 14.0               # row-2 titles lifted over E's one-line count band
LABEL_CLEAR_PT = 2.0                # a printed label keeps this from every rule and mark


def mean_ci(values, seed, n_boot=20_000):
    """``build_microns_v661_replication_figure.mean_ci`` verbatim."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def band_text(ax, text, *, x=0.0, ha="left", color=MUTE, dy=2.5):
    """A one-line statement in the band between the axes top and the title."""
    return ax.annotate(text, xy=(x, 1.0), xycoords="axes fraction", xytext=(0.0, dy),
                       textcoords="offset points", ha=ha, va="bottom", fontsize=PT_BASE,
                       color=color, annotation_clip=False)


def fan_v(ax, x, values, color, *, half=0.0, ms=SEED_MS, alpha=SEED_ALPHA, zorder=2.0,
          log=False):
    """Per-cell values as a fan behind the mean at column ``x``; ``half`` is
    the horizontal spread (additive, or multiplicative on a log axis)."""
    values = np.asarray(values, float)
    n = len(values)
    j = np.linspace(-half, half, n) if (n > 1 and half > 0) else np.zeros(n)
    xs = x * np.exp(j) if log else x + j
    ax.plot(xs, values, linestyle="none", marker="o", markersize=ms, markerfacecolor=color,
            markeredgecolor="none", alpha=alpha, zorder=zorder)


def hollow_mean(ax, x, mean, lo, hi, color, marker, *, ms=MEAN_MS, zorder=5.0, horizontal=False):
    err = [[mean - lo], [hi - mean]]
    # the face is open, so the per-cell fan behind the mean stays visible
    kw = dict(marker=marker, ms=ms, color=color, markerfacecolor="none",
              markeredgecolor=color, markeredgewidth=LW_ERR, elinewidth=LW_ERR,
              capsize=ERR_CAPSIZE, capthick=LW_ERR, linestyle="none", zorder=zorder)
    if horizontal:
        ax.errorbar([mean], [x], xerr=err, **kw)
    else:
        ax.errorbar([x], [mean], yerr=err, **kw)


# ── A, B: paired cell plots (S10 F, G) ───────────────────────────────────
def paired_panel(ax, frame, left, right, labels, colors, seed, intervals):
    """``plot_pair`` of the S10 builder: the same pairs, the same bootstrap."""
    paired = frame[[left, right]].dropna()
    xs = np.array([0.0, 1.0])
    for _, row in paired.iterrows():
        values = row[[left, right]].to_numpy(dtype=float)
        ax.plot(xs, values, color=MUTE, alpha=0.32, lw=LW_HAIR, zorder=1)
    for index, column in enumerate([left, right]):
        vals = paired[column].to_numpy(dtype=float)
        ax.plot(np.full(len(vals), xs[index]), vals, linestyle="none", marker="o",
                markersize=SEED_MS, markerfacecolor=colors[index], markeredgecolor="none",
                alpha=SEED_ALPHA, zorder=2)
        mean, low, high = mean_ci(vals, seed + index)
        hollow_mean(ax, xs[index], mean, low, high, colors[index], M_CONTRAST)
        intervals.append((ax, labels[index].replace("\n", " "), "y", low, high))
    # a hairline zero rule under everything: the injection mean of A (0.013)
    # lies on it, so the rule must read as lighter than the diamond's edge
    ax.axhline(0, color=MUTE, ls=DASHED, lw=LW_HAIR, zorder=0.6)   # above the y grid rule at 0, under the data
    ax.set_xticks(xs, labels)
    ax.tick_params(axis="x", length=0, pad=2.5)
    ax.set_xlim(-0.42, 1.42)
    ax.set_ylabel(LOCAL_LABEL)
    return paired


def panel_matched(ax, focal, summary, intervals):
    direct = summary["focal_primary_dose_one"]["shunt_minus_additive"]
    paired = paired_panel(ax, focal, "matched_additive_localization", "focal_shunt_localization",
                          ("matched current\ninjection", "focal\nshunt"), (INJECT, SHUNT), 3000, intervals)
    diff = (paired.focal_shunt_localization - paired.matched_additive_localization).to_numpy(float)
    sites = focal.loc[paired.index, "n_focal_sites"].to_numpy(float)
    assert len(paired) == direct["n_cells"] == 45
    assert int(sites.sum()) == direct["n_focal_sites"] == 235
    assert int((diff > direct["zero_tolerance"]).sum()) == direct["cells_positive"] == 45
    np.testing.assert_allclose(diff.mean(), direct["mean_difference"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(paired.focal_shunt_localization.mean(),
                               summary["focal_primary_dose_one"]["mean_shunt_localization"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(paired.matched_additive_localization.mean(),
                               summary["focal_primary_dose_one"]["mean_additive_localization"], rtol=0, atol=1e-12)
    band_text(ax, f"{direct['cells_positive']}/{direct['n_cells']} cells positive", color=SHUNT)
    print(f"[A] n = {len(paired)} cells, {int(sites.sum())} sites; shunt {paired.focal_shunt_localization.min():.4f}-"
          f"{paired.focal_shunt_localization.max():.4f}, injection {paired.matched_additive_localization.min():.4f}-"
          f"{paired.matched_additive_localization.max():.4f}; shunt - injection {diff.mean():.4f}, positive "
          f"{int((diff > 0).sum())}/{len(diff)}")
    return paired


def panel_relation(ax, focal, summary, paired_a, intervals):
    topology = summary["focal_primary_dose_one"]["topology_minus_depth_shuffle"]
    paired = paired_panel(ax, focal, "shunt_depth_shuffled_localization", "focal_shunt_localization",
                          ("reassigned\nrelation", "true\nrelation"), (REASSIGN, SHUNT), 4000, intervals)
    diff = (paired.focal_shunt_localization - paired.shunt_depth_shuffled_localization).to_numpy(float)
    sites = focal.loc[paired.index, "n_focal_sites"].to_numpy(float)
    tol = topology["zero_tolerance"]
    assert len(paired) == topology["n_cells"] == 40
    assert int(sites.sum()) == topology["n_focal_sites"] == 230
    assert int((diff > tol).sum()) == topology["cells_positive"] == 39
    assert int((np.abs(diff) <= tol).sum()) == topology["cells_tied_within_tolerance"] == 1
    np.testing.assert_allclose(diff.mean(), topology["mean_difference"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(paired.shunt_depth_shuffled_localization.mean(),
                               summary["focal_primary_dose_one"]["mean_depth_shuffled_localization"], rtol=0, atol=1e-12)
    # the true-relation column IS A's focal-shunt column for these forty cells
    assert set(paired.index) <= set(paired_a.index)
    np.testing.assert_allclose(paired.focal_shunt_localization.to_numpy(float),
                               paired_a.loc[paired.index, "focal_shunt_localization"].to_numpy(float), rtol=0, atol=0)
    band_text(ax, f"{topology['cells_positive']} positive, {topology['cells_tied_within_tolerance']} tie / "
                  f"{topology['n_cells']} cells", color=SHUNT)
    print(f"[B] n = {len(paired)} cells, {int(sites.sum())} sites; reassigned {paired.shunt_depth_shuffled_localization.min():.4f}-"
          f"{paired.shunt_depth_shuffled_localization.max():.4f}; true - reassigned {diff.mean():.4f}, positive "
          f"{int((diff > tol).sum())}/{len(diff)}, ties {int((np.abs(diff) <= tol).sum())}")
    return paired


# ── C: direct-label coverage and selected focal sites (S10 H) ────────────
def panel_coverage(ax, cohort):
    included = cohort[cohort["routing_included"].astype(str).str.lower().eq("true")].copy()
    for col in ("direct_type_coverage", "n_selected_focal_sites", "n_eligible_focal_sites"):
        included[col] = pd.to_numeric(included[col], errors="coerce")
    assert len(included) == 47 and included.direct_type_coverage.notna().all()
    counts = included.cell_type.value_counts().to_dict()
    assert counts == {"L2IT": 34, "L4IT": 10, "L3IT": 2, "L5ET": 1}, counts
    assert int(included.n_selected_focal_sites.sum()) == 235          # the caption's 235 sites
    assert int(included.n_eligible_focal_sites.sum()) == 266
    assert included.n_selected_focal_sites.max() == SITE_CAP
    l5et = included[included.cell_type.eq("L5ET")].iloc[0]
    assert l5et.n_eligible_focal_sites == 47 and l5et.n_selected_focal_sites == SITE_CAP
    capped = included[included.n_eligible_focal_sites > included.n_selected_focal_sites]
    assert len(capped) == 1 and capped.iloc[0].cell_type == "L5ET"
    handles = []
    for cell_type, color, marker in CLASS_STYLE:
        group = included[included.cell_type.eq(cell_type)]
        assert len(group) == counts[cell_type]
        ax.plot(100.0 * group.direct_type_coverage, group.n_selected_focal_sites, linestyle="none",
                marker=marker, markersize=MARKER_MS * 0.85, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=LW_HAIR, alpha=0.9, zorder=3)
        handles.append(Line2D([], [], linestyle="none", marker=marker, markersize=MARKER_MS * 0.85,
                              markerfacecolor=color, markeredgecolor="white", markeredgewidth=LW_HAIR,
                              label=f"{cell_type} ({counts[cell_type]})"))
    x = 100.0 * included.direct_type_coverage
    ax.set_xlim(2.0, 10.6)
    ax.set_xticks([4, 6, 8, 10])
    ax.set_ylim(-0.9, SITE_CAP + 1.2)
    ax.set_yticks([0, 5, 10, 15])
    ax.set_xlabel("direct E/I labels (% inputs)")
    ax.set_ylabel(f"selected focal sites (max {SITE_CAP})")
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2,
              frameon=False, fontsize=PT_BASE, handlelength=1.0, columnspacing=0.9,
              handletextpad=0.35, borderaxespad=0.0, borderpad=0.0, labelspacing=0.15)
    print(f"[C] n = {len(included)} cells; coverage {x.min():.2f}-{x.max():.2f} %; selected sites "
          f"{int(included.n_selected_focal_sites.min())}-{int(included.n_selected_focal_sites.max())} "
          f"(sum 235; eligible sum 266; L5ET 47 eligible capped at {SITE_CAP})")


# ── D: weak-channel dose response with the passive difference inset ──────
def load_passive_reference():
    table = pd.read_csv(PHASE1 / "condition_summary.csv")
    cells = pd.read_csv(PHASE1 / "cell_condition_metrics.csv")
    for column, value in PASSIVE_CALIBRATION.items():
        table = table[table[column].eq(value)]
        cells = cells[cells[column].eq(value)]
    assert len(table) == 6 and len(cells) == 48
    return table, cells


def per_cell(cells, dose, perturbation, dose_col):
    z = cells[np.isclose(cells[dose_col], dose) & cells.perturbation.eq(perturbation)]
    assert len(z) == 8 and z.root_id.nunique() == 8
    return z.set_index("root_id").localization_index.sort_index()


def panel_dose(ax, summary, cells, passive_summary, passive_cells, intervals):
    series = (("focal shunt", SHUNT, M_SHUNT), ("matched additive", INJECT, M_INJECT))
    diffs = {}
    printed = []
    for perturbation, color, marker in series:
        part = summary[summary.perturbation.eq(perturbation)].sort_values("dose_relative_to_local_input_conductance")
        x = part.dose_relative_to_local_input_conductance.to_numpy(float)
        np.testing.assert_allclose(x, DOSES, rtol=0, atol=0)
        assert (part.n_cells == 8).all()
        means = part.mean_localization_index.to_numpy(float)
        lows = part.ci95_low_localization_index.to_numpy(float)
        highs = part.ci95_high_localization_index.to_numpy(float)
        ref = passive_summary[passive_summary.perturbation.eq(perturbation)].sort_values("dose_value")
        np.testing.assert_allclose(ref.dose_value.to_numpy(float), DOSES, rtol=0, atol=0)
        ref_means = ref.mean_localization_index.to_numpy(float)
        for i, dose in enumerate(DOSES):
            active = per_cell(cells, dose, perturbation, "dose_relative_to_local_input_conductance")
            passive = per_cell(passive_cells, dose, perturbation, "dose_value")
            assert list(active.index) == list(passive.index)
            np.testing.assert_allclose(active.mean(), means[i], rtol=0, atol=1e-9)
            np.testing.assert_allclose(passive.mean(), ref_means[i], rtol=0, atol=1e-9)
            assert lows[i] <= means[i] <= highs[i]
            # the sign claim of the caption: signed localization negative in
            # every cell for the shunt (attenuates descendants), positive for
            # the injection (enhances them)
            signed = cells[np.isclose(cells.dose_relative_to_local_input_conductance, dose)
                           & cells.perturbation.eq(perturbation)].signed_localization.to_numpy(float)
            assert (signed < 0).all() if perturbation == "focal shunt" else (signed > 0).all()
            fan_v(ax, dose, active.to_numpy(float), color, half=0.045, log=True, zorder=2.0)
            diffs[(perturbation, dose)] = (active - passive).to_numpy(float)
            printed.append(f"{perturbation} dose {dose:g}: mean {means[i]:.4f} [{lows[i]:.4f}, {highs[i]:.4f}] "
                           f"(width {highs[i] - lows[i]:.4f}), cells {active.min():.4f}-{active.max():.4f}, "
                           f"active - passive mean {means[i] - ref_means[i]:+.4f}")
        ax.plot(x, means, color=color, lw=LW_DATA, zorder=3.0)
        for i in range(len(DOSES)):
            hollow_mean(ax, x[i], means[i], lows[i], highs[i], color, marker)
            intervals.append((ax, f"{perturbation} dose {DOSES[i]:g}", "y", lows[i], highs[i]))
    # the caption's discrepancy bounds (active minus passive, mean per dose)
    shunt_gap = max(abs(diffs[("focal shunt", d)].mean()) for d in DOSES)
    inject_gap = max(abs(diffs[("matched additive", d)].mean()) for d in DOSES)
    assert shunt_gap <= 0.010 and inject_gap <= 0.023, (shunt_gap, inject_gap)
    print("[D] " + "; ".join(printed))
    print(f"[D] largest |active - passive| mean: shunt {shunt_gap:.4f}, injection {inject_gap:.4f}; "
          f"per-cell range {min(v.min() for v in diffs.values()):+.4f}..{max(v.max() for v in diffs.values()):+.4f}")
    ax.set_xscale("log")
    ax.set_xticks(list(DOSES), list(DOSE_LABELS))
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_xlim(0.2, 5.0)
    ax.set_ylim(-0.04, 1.44)
    ax.set_yticks([0.0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.set_xlabel(DOSE_AXIS)
    ax.set_ylabel(LOCAL_LABEL)
    # direct labels in the series colour, in clear whitespace beside each
    # curve, each with the direction of the effect (the old panel's tags)
    # in mute ink beneath it; ``label_checks`` measures their clearance from
    # every line, fan, whisker and rule once the reserve lock has settled
    labels = [
        ax.text(2.9, 1.33, "focal shunt", color=SHUNT, fontsize=PT_BASE, ha="right", va="bottom", zorder=6),
        ax.text(2.9, 1.24, "attenuates descendants", color=MUTE, fontsize=PT_BASE, ha="right", va="bottom", zorder=6),
        ax.text(3.5, 0.635, "current injection", color=INJECT, fontsize=PT_BASE, ha="right", va="bottom", zorder=6),
        ax.text(3.5, 0.545, "enhances descendants", color=MUTE, fontsize=PT_BASE, ha="right", va="bottom", zorder=6),
    ]

    # inset: active minus passive per cell, on its own scale
    # opaque face: D's pale y = 1.0 grid rule no longer shows through the
    # inset's data field; the inset's labels ('0', '0.10', 'dose') are
    # placed clear of that rule and of the 0.5 rule (``label_checks``)
    sub = ax.inset_axes([0.15, 0.53, 0.33, 0.33])
    sub.set_facecolor("white")
    style_panel(sub)
    allv = np.concatenate(list(diffs.values()))
    for k, (perturbation, color, marker) in enumerate(series):
        dx = -0.17 if k == 0 else 0.17
        m = []
        for i, dose in enumerate(DOSES):
            v = diffs[(perturbation, dose)]
            fan_v(sub, i + dx, v, color, half=0.07, ms=SEED_MS * 0.8, zorder=2.0)
            m.append(v.mean())
        sub.plot(np.arange(3) + dx, m, linestyle="none", marker=marker, markersize=MARKER_MS * 0.7,
                 markerfacecolor="none", markeredgecolor=color, markeredgewidth=LW_ERR, zorder=4.0)
    sub.axhline(0.0, color=MUTE, ls=DASHED, lw=LW_REF, zorder=1.0)
    sub.set_xlim(-0.55, 2.55)
    sub.set_xticks([0, 1, 2], list(DOSE_LABELS))
    sub.set_ylim(min(allv.min(), 0.0) - 0.01, allv.max() + 0.012)
    sub.set_yticks([0.0, 0.10], ["0", "0.10"])
    sub.set_yticks([0.05], minor=True)
    sub.tick_params(axis="both", labelsize=PT_BASE, pad=1.5, length=2.2)
    sub.set_xlabel("dose", fontsize=PT_BASE, labelpad=1.5)
    labels.append(ax.annotate("active − passive localization,\nper cell", xy=(0.15, 0.53 + 0.33),
                              xycoords="axes fraction", xytext=(-14.0, 3.0), textcoords="offset points",
                              ha="left", va="bottom", fontsize=PT_BASE, color=INK, linespacing=1.15,
                              annotation_clip=False))
    return sub, labels


# ── E: paired shunt-minus-current contrasts, three dose rows ─────────────
def panel_contrast(ax, contrasts, cells, intervals):
    rows = contrasts[contrasts.metric.eq("localization_index")].sort_values("dose_relative_to_local_input_conductance")
    np.testing.assert_allclose(rows.dose_relative_to_local_input_conductance.to_numpy(float), DOSES, rtol=0, atol=0)
    printed = []
    widths = []
    for y, (_, row) in enumerate(rows.iterrows()):
        dose = float(row.dose_relative_to_local_input_conductance)
        shunt = per_cell(cells, dose, "focal shunt", "dose_relative_to_local_input_conductance")
        inject = per_cell(cells, dose, "matched additive", "dose_relative_to_local_input_conductance")
        diff = (shunt - inject).to_numpy(float)
        mean, lo, hi = float(row.mean_shunt_minus_additive), float(row.ci95_low), float(row.ci95_high)
        np.testing.assert_allclose(diff.mean(), mean, rtol=0, atol=1e-9)
        assert int(row.n_cells) == 8 == len(diff)
        assert int((diff > 0).sum()) == int(row.cells_positive) == 8
        assert lo <= mean <= hi
        np.testing.assert_allclose(float(row.wilcoxon_p_two_sided), 0.0078125, rtol=0, atol=1e-12)
        # eight cells as a vertical fan on the row, behind the diamond
        jitter = np.linspace(-0.2, 0.2, len(diff))
        ax.plot(diff, y + jitter, linestyle="none", marker="o", markersize=SEED_MS, markerfacecolor=SHUNT,
                markeredgecolor="none", alpha=SEED_ALPHA, zorder=2.0)
        hollow_mean(ax, y, mean, lo, hi, SHUNT, M_CONTRAST, horizontal=True)
        intervals.append((ax, f"contrast dose {dose:g}", "x", lo, hi))
        assert diff.max() < 1.13                 # the dose-4 fan ends at 1.12, inside E_XMAX
        widths.append(hi - lo)
        printed.append(f"dose {dose:g}: {mean:.4f} [{lo:.4f}, {hi:.4f}] (width {hi - lo:.4f}), cells "
                       f"{diff.min():.4f}-{diff.max():.4f}, positive {int((diff > 0).sum())}/8, p {row.wilcoxon_p_two_sided:.4g}")
    print("[E] " + "; ".join(printed))
    # the paired count, once for the three rows (asserted 8/8 in each above),
    # in the band above the axes as in A and B rather than in the data field
    assert (rows.cells_positive == 8).all() and (rows.n_cells == 8).all()
    band_text(ax, "8/8 cells positive at every dose", color=SHUNT)
    ax.axvline(0.0, color=MUTE, ls=DASHED, lw=LW_REF, zorder=1.0)
    ax.set_ylim(-0.5, len(DOSES) - 0.5)
    # the rule's label sits inside the box beside the rule, above the top
    # row's fan (which spans y = 2 +/- 0.2 and starts at x = 0.72)
    ax.annotate("no effect", xy=(0.0, len(DOSES) - 0.5), xycoords="data", xytext=(2.5, -1.5),
                textcoords="offset points", ha="left", va="top", fontsize=PT_BASE, color=MUTE, zorder=6)
    ax.set_yticks(range(len(DOSES)), list(DOSE_LABELS))
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.set_xlim(-0.05, E_XMAX)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1.0"])
    ax.set_ylabel(DOSE_AXIS)
    ax.set_xlabel("shunt − current-injection localization")
    return widths


def whisker_report(cv, intervals):
    """Interval extent in points against the mean marker, per summary mark,
    so the caption can say exactly which whiskers the marker hides."""
    hidden, shown = [], []
    for ax, label, axis, lo, hi in intervals:
        box = ax.get_position()
        if axis == "y":
            pt_per_unit = box.height * cv.height_pt / np.ptp(ax.get_ylim())
        else:
            pt_per_unit = box.width * cv.width_pt / np.ptp(ax.get_xlim())
        extent = (hi - lo) * pt_per_unit
        name = f"{cv_name(cv, ax)} {label}"
        (shown if extent > MEAN_MS else hidden).append(f"{name} ({extent:.1f} pt)")
    print(f"[whiskers] marker {MEAN_MS:.1f} pt; hidden under the marker: " + "; ".join(hidden))
    print("[whiskers] wider than the marker: " + "; ".join(shown))


def cv_name(cv, ax):
    return next(name for name, a in cv.axes.items() if a is ax)


def label_checks(cv, ax, sub, labels, clear_pt=LABEL_CLEAR_PT):
    """Every printed label of D keeps ``clear_pt`` from every data line,
    marker, fan dot, whisker, the inset (with its own labels) and the box.

    Measured on the locked axes in display points: a text's bbox, grown by
    the clearance, may not contain any data-line sample or marker centre
    (markers/dots grown by their radius), and must lie inside the data box.
    """
    fig = cv.fig
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = 72.0 / fig.dpi
    box = ax.get_window_extent(renderer)
    sub_box = sub.get_tightbbox(renderer)
    marks = []                                   # (x_px, y_px, radius_pt)

    def polyline(pts, half_width_pt):
        # sample a polyline every ~0.5 pt so a sloping segment is covered
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) * scale / 0.5))
            for t in np.linspace(0.0, 1.0, n):
                marks.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), half_width_pt))

    for line in ax.get_lines():
        xy = line.get_xydata()
        if len(xy) == 0:
            continue
        pts = ax.transData.transform(xy)
        if line.get_linestyle() not in ("None", "none", "") and len(pts) > 1:
            polyline(pts, line.get_linewidth() / 2.0)
        if line.get_marker() not in ("None", "none", ""):
            r = line.get_markersize() / 2.0 + line.get_markeredgewidth() / 2.0
            for x, y in pts:
                marks.append((x, y, r))
    for coll in ax.collections:                  # whisker segments (errorbar LineCollections)
        for seg in coll.get_segments():
            polyline(ax.transData.transform(seg), LW_ERR / 2.0)
    for y in ax.get_yticks():                    # the pale y grid rules are rules too
        if ax.get_ylim()[0] < y < ax.get_ylim()[1]:
            polyline(ax.transData.transform([[ax.get_xlim()[0], y], [ax.get_xlim()[1], y]]), LW_HAIR / 2.0)
    marks = np.asarray(marks, float)
    report = []
    for text in labels:
        bb = text.get_window_extent(renderer)
        name = text.get_text().replace("\n", " / ")
        # inside the data box by the clearance (the inset title may touch
        # the inset's own top: it is that inset's label)
        edge = min(bb.x0 - box.x0, box.x1 - bb.x1, bb.y0 - box.y0, box.y1 - bb.y1) * scale
        assert edge >= clear_pt, (name, edge)
        dx = np.clip(marks[:, 0], bb.x0, bb.x1) - marks[:, 0]
        dy = np.clip(marks[:, 1], bb.y0, bb.y1) - marks[:, 1]
        gap = np.hypot(dx, dy) * scale - marks[:, 2]
        k = int(np.argmin(gap))
        nearest = float(gap[k])
        near_xy = ax.transData.inverted().transform(marks[k, :2])
        assert nearest >= clear_pt, (name, nearest, near_xy)
        if text.axes is ax and not text.get_text().startswith("active"):
            # clear of the inset and everything hung on it
            ox = max(0.0, max(sub_box.x0 - bb.x1, bb.x0 - sub_box.x1)) * scale
            oy = max(0.0, max(sub_box.y0 - bb.y1, bb.y0 - sub_box.y1)) * scale
            inset_gap = float(np.hypot(ox, oy))
            assert inset_gap >= clear_pt, (name, inset_gap)
        else:
            inset_gap = float("nan")
        report.append(f"'{name}': box edge {edge:.1f} pt, nearest mark {nearest:.1f} pt at "
                      f"({near_xy[0]:.2f}, {near_xy[1]:.3f})"
                      + ("" if np.isnan(inset_gap) else f", inset {inset_gap:.1f} pt"))
    print("[D labels] " + "; ".join(report))


# ── the canvas ───────────────────────────────────────────────────────────
ROW_PT = [150.0, 158.0]
HGUTTER_PT = 30.0
VGUTTER_PT = 50.0
MARGINS = Margins(left=44.0, right=6.0, top=20.0, bottom=36.0)
CANVAS_H_PT = MARGINS.top + sum(ROW_PT) + VGUTTER_PT + MARGINS.bottom


def build(path: Path = OUT):
    focal = pd.read_csv(MICRONS / "supp_figure_focal_cells.csv")
    summary = json.loads((MICRONS / "replication_summary.json").read_text(encoding="utf-8"))
    cohort = pd.read_csv(MICRONS / "cohort_manifest.csv", keep_default_na=False)
    active_summary = pd.read_csv(ACTIVE / "condition_summary.csv")
    active_contrasts = pd.read_csv(ACTIVE / "paired_contrasts.csv")
    active_cells = pd.read_csv(ACTIVE / "cell_condition_metrics.csv")
    active_meta = json.loads((ACTIVE / "summary.json").read_text())
    assert active_meta["n_cells"] == 8 and active_meta["accepted_draws_per_cell"] == 64
    assert len(active_cells) == 48 and active_cells.root_id.nunique() == 8
    passive_summary, passive_cells = load_passive_reference()
    assert len(focal) == 47

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 2, row_weights=ROW_PT, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS)
    ax_a = cv.panel("A", 0, 0, 4, grid="y", title="Matched perturbations")
    ax_b = cv.panel("B", 0, 4, 4, grid="y", title="True versus reassigned relation")
    ax_c = cv.panel("C", 0, 8, 4, grid="y", title="Direct-label coverage")
    ax_d = cv.panel("D", 1, 0, 7, grid="y", title="Weak-channel linearization versus passive")
    ax_e = cv.panel("E", 1, 7, 5, title="Weak-channel localization contrasts")
    for ax in (ax_a, ax_b, ax_c):
        ax.set_title(ax.get_title(), fontsize=PT_EMPH, color=INK, pad=TITLE_PAD_KEYED, fontweight="normal")
    for ax in (ax_d, ax_e):
        ax.set_title(ax.get_title(), fontsize=PT_EMPH, color=INK, pad=TITLE_PAD_BAND, fontweight="normal")
    for name in "ABCDE":
        cv.declare_reserve(name, left=14.0, right=6.0)

    intervals = []                           # (panel, label, axis, lo, hi) for the marker check
    paired_a = panel_matched(ax_a, focal, summary, intervals)
    panel_relation(ax_b, focal, summary, paired_a, intervals)
    for ax in (ax_a, ax_b):
        ax.set_ylim(-0.09, 0.56)
        ax.set_yticks([0.0, 0.2, 0.4], ["0", "0.2", "0.4"])
    panel_coverage(ax_c, cohort)
    sub, d_labels = panel_dose(ax_d, active_summary, active_cells, passive_summary, passive_cells, intervals)
    panel_contrast(ax_e, active_contrasts, active_cells, intervals)

    style_direct_color_labels(cv.fig)
    cv.lock_reserves()                       # settle the boxes, then measure the whiskers
    whisker_report(cv, intervals)
    label_checks(cv, ax_d, sub, d_labels + [sub.xaxis.label] + sub.get_xticklabels() + sub.get_yticklabels())
    problems = cv.save(path, name="figure_shunt_replication_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

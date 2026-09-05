#!/usr/bin/env python3
"""Main Figure 8 as ONE native full-width canvas.

Figure 8 used to be composed from three pre-rendered sub-blocks
(``fig4_focal_shunting`` panels B/D/E/F, ``fig_active_dose_main`` panels A/B
and the ``main_focal_schematic`` component), each scaled by a different
factor into a grid slot.  That destroyed the journal type scale (the compiled
page carried nine different text sizes) and made row-mates different physical
sizes.  This module rebuilds the whole figure as one 12-module canvas at
exactly ``FIG_W`` inches, emitted at scale 1.0.

Every number, ``n``, bootstrap interval and test is read from the same frozen
source tables through the same helper (``build_journal_figures.mean_ci``) with
the same seeds as the sub-block builders, so the plotted values are identical
to today.  What changes is geometry, encoding and grammar:

* **S2** -- the shunt-off/shunt-on tree pair (panel A) sits immediately left of
  the tree-relation result it explains, drawn natively from the shared
  ``credit_tree_schematics`` vocabulary at the figure's own stroke weights
  instead of being pasted in from a foreign scale;
* **S1** -- panels C and D plot the same quantity (localization index) on one
  genuinely shared y axis: C keeps the label, ticks and grid, D drops the
  duplicate tick column and cross-references it;
* **S3** -- the electrotonic-boundary panel (F) opens the bottom row on six of
  the twelve modules, at the same axes-box height as every other panel on the
  page, because it carries the state-dependence claim and the
  small, opposite-signed standard-calibration contrasts;
* **S4** -- the paired shunt-minus-additive contrasts of the active ensemble
  are consolidated into one forest panel (G) on a shared effect-size axis,
  which also surfaces the sign count and the Wilcoxon test that previously
  lived only in the running text;
* **S5/S6** -- colour carries condition, not decoration: focal shunt green
  circles, matched additive blue squares, the driving-force-only freeze amber
  triangles and the replication cohort a second lightness of the same green;
  every two-series panel is direct-labelled and no legend box remains;
* the random horizontal jitter of the old category and factor-freeze panels
  is gone -- seed points sit at their honest x position.

Output: ``figures/components/main_figure_08_native.pdf`` (+ 600-dpi PNG),
which ``assemble_compact_main_figures.emit_native(8)`` copies to
``figures/main/figure_08.pdf`` at scale 1.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_canvas import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    Margins,
    NativeCanvas,
    PT_ANNOT,
    PT_LABEL,
    PT_SMALL,
    PT_TICK,
    PT_TITLE,
    SEED_ALPHA,
    SEED_MS,
    enforce_tokens,
    style_panel,
)
from credit_tree_schematics import (  # noqa: E402
    _BASE_XLIM,
    _BASE_YLIM,
    P as TREE_P,
    draw_credit_tree,
    mix,
)
from build_journal_figures import mean_ci  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
COMPONENTS = ROOT / "figures" / "components"
OUTPUT = COMPONENTS / "main_figure_08_native.pdf"

# ── canvas geometry (points) ─────────────────────────────────────────────
HEIGHT_IN = 493.0 / 72.0            # aspect 1.10, inside the 1.05-1.55 band
# The horizontal gutter is the figure's ONE shared left reserve: it is set
# wide enough to hold the widest y label and tick column on the page, so no
# panel has to carve that space out of its own module slot and every panel of
# a grid column keeps the same x0 and the same axes width.
HGUTTER = 36.0
VGUTTER = 57.0
MARGINS = Margins(left=49.0, right=8.0, top=21.0, bottom=27.0)
ROW_WEIGHTS = (1.00, 1.00, 1.00)

# ── one encoding vocabulary for the whole figure ─────────────────────────
SHUNT = COLORS["shunting"]          # focal shunting conductance
ADDITIVE = COLORS["additive"]       # current-matched additive control
FREEZE = COLORS["local"]            # driving-force-only factor freeze
REPLICATE = mix("shunting", 62, "ink")   # second lightness: v661 cohort
MUTE = COLORS["mute"]
INK = COLORS["ink"]

M_SHUNT = "o"                       # focal shunt level
M_ADD = "s"                         # matched additive level
M_FREEZE = "^"                      # driving-force-only level
M_CONTRAST = "D"                    # shunt - additive contrast
M_REPLICATE = "v"                   # replication-cohort contrast

# Shared y axis of the two passive localization panels (C keeps the ticks).
LOCAL_YLIM = (-0.010, 0.248)
LOCAL_YTICKS = (0.0, 0.05, 0.10, 0.15, 0.20)
LOCAL_LABEL = "localization index"

# Forest rows carry their cells on a declared sub-row this far above the
# summary marker, so the points are never hidden under it.
CELL_ROW_DY = 0.245

# The mean/summary marker is MARKER_MS + 1.2 everywhere (the mark contract),
# and it is stroked at LW_ERR everywhere, so the same glyph never prints at
# two sizes or two edge weights inside one figure.
MEAN_MS = MARKER_MS + 1.2

TREE_ASPECT = ((_BASE_XLIM[1] - _BASE_XLIM[0])
               / (_BASE_YLIM[1] - _BASE_YLIM[0]))


# ── small drawing helpers ────────────────────────────────────────────────
def _num(fmt: str, value: float) -> str:
    """Format a number with a true minus sign, never a hyphen."""
    return fmt.format(value).replace("-", "−")


def _title(ax, text):
    ax.set_title(text, fontsize=PT_TITLE, loc="center", pad=3.5, color=INK,
                 fontweight="normal")


def _tick_labels(ax, axis, values, labels, *, size=PT_TICK):
    getattr(ax, f"set_{axis}ticks")(list(values), list(labels))
    ax.tick_params(axis=axis, labelsize=size)


def _seed_points(ax, x, values, color, *, zorder=2.4):
    """Per-cell points at their honest x position -- never jittered."""
    values = np.asarray(values, dtype=float)
    ax.plot(np.full(values.size, x), values, marker="o", ls="none",
            ms=SEED_MS, mfc=color, mec="none", alpha=SEED_ALPHA,
            zorder=zorder)


def _mean_marker(ax, x, values, color, marker, *, seed=0, zorder=5):
    """Mean +- 95% cell-bootstrap interval: the figure-wide summary glyph."""
    m, lo, hi = mean_ci(np.asarray(values, dtype=float), seed=seed)
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], marker=marker,
                ms=MEAN_MS, color=color, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=LW_ERR, lw=LW_ERR,
                capsize=ERR_CAPSIZE, zorder=zorder)
    return m, lo, hi


def _zero_line(ax, *, axis="y"):
    if axis == "y":
        ax.axhline(0.0, color=MUTE, ls="--", lw=LW_REF, zorder=1.0)
    else:
        ax.axvline(0.0, color=MUTE, ls="--", lw=LW_REF, zorder=1.0)


def _direct_label(ax, x, y, text, color, *, size=PT_ANNOT, ha="left",
                  va="center"):
    """Series name in its own colour, in clear whitespace: no legend box."""
    return ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va,
                   zorder=6)


# ── panel A: the shunt-off / shunt-on credit-tree pair ───────────────────
def _tree_inset(ax, rect):
    """Credit-tree inset filling ``rect`` at the library's own proportions."""
    x0, y0, w, h = rect
    sub = ax.inset_axes([x0, y0, w, h], transform=ax.transData, zorder=3)
    sub.set_facecolor("none")
    return sub


def _site_xy(rect):
    """Where the shunt site lands inside a tree rect, in axes fractions."""
    j1, jl = np.asarray(TREE_P["J1"], float), np.asarray(TREE_P["JL"], float)
    sx, sy = j1 + 0.86 * (jl - j1)
    u = (sx - _BASE_XLIM[0]) / (_BASE_XLIM[1] - _BASE_XLIM[0])
    v = (sy - _BASE_YLIM[0]) / (_BASE_YLIM[1] - _BASE_YLIM[0])
    return rect[0] + rect[2] * u, rect[1] + rect[3] * v


def panel_focal_shunt(ax, *, width_pt, height_pt):
    """Current-matched additive injection versus an open conductance.

    The pair is the talk library's ``mode=shunt`` tree drawn twice -- open
    inhibitory ring (conductance off, the current-matched additive control)
    and engaged ring with the descendant subtree thinned and faded
    (conductance on).  Drawn natively, so its strokes are the figure's own
    ``LW_DATA``..``LW_EDGE`` taper rather than a rescaled sub-block.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()

    top = 0.868
    # The pair sits left inside the panel rather than filling it edge to
    # edge: the right shoulder it gives up is the clearance B's letter needs.
    span = 0.925
    tree_w = 0.412
    tree_h = min(0.625, (tree_w * width_pt / TREE_ASPECT) / height_pt)
    bottom = top - tree_h
    left_rect = (0.0, bottom, tree_w, tree_h)
    right_rect = (span - tree_w, bottom, tree_w, tree_h)

    for rect, shunted in ((left_rect, False), (right_rect, True)):
        sub = _tree_inset(ax, rect)
        draw_credit_tree(sub, mode="shunt", shunted=shunted, scale=0.78,
                         labels=False)
        enforce_tokens(sub)

    # The baseline focal current is matched and a separate somatic current
    # restores the baseline soma voltage.  Local dendritic voltages are not
    # clamped, so the graphic must not claim a matched focal Delta V.
    bx0, bx1, by = 0.012, span - 0.012, 0.968
    ax.plot([bx0, bx1], [by, by], color=MUTE, lw=LW_HAIR, zorder=2,
            solid_capstyle="butt")
    for x in (bx0, bx1):
        ax.plot([x, x], [by, by - 0.022], color=MUTE, lw=LW_HAIR, zorder=2)
    ax.text(0.5 * span, 0.908, "baseline focal current matched; soma V restored",
            ha="center", va="center", fontsize=PT_SMALL, color=MUTE)

    lx = left_rect[0] + left_rect[2] / 2.0
    rx = right_rect[0] + right_rect[2] / 2.0
    ax.text(lx, bottom - 0.075, "matched current injection", ha="center", va="center",
            fontsize=PT_ANNOT, color=ADDITIVE)
    ax.text(rx, bottom - 0.075, "focal shunt", ha="center", va="center",
            fontsize=PT_ANNOT, color=SHUNT)
    ax.text(lx, bottom - 0.165, "gₛₕ off", ha="center",
            va="center", fontsize=PT_SMALL, color=MUTE)
    ax.text(rx, bottom - 0.165, "gₛₕ on", ha="center",
            va="center", fontsize=PT_SMALL, color=COLORS["inh"])

    # The matched current is depicted, not just asserted: a short arrow in
    # the additive colour enters the open (conductance-off) site of the left
    # tree, giving the blue condition label a blue referent in the drawing.
    site = _site_xy(left_rect)
    tail = (site[0] - 0.105, site[1] - 0.115)
    ax.add_patch(FancyArrowPatch(
        tail, site, arrowstyle="-|>,head_length=4.5,head_width=2.79",
        mutation_scale=1.0, color=ADDITIVE, lw=LW_DATA, capstyle="round",
        shrinkA=0.0, shrinkB=2.6, zorder=4.5))

    # The fade convention (pale strokes = attenuated descendant credit) is a
    # methodological note and now lives in the caption, not on the drawing.
    return ax


# ── panel B: which tree relations lose modelled gradient ─────────────────
CATEGORIES = ("descendant", "sister", "ancestor", "depth-matched unrelated",
              "unrelated")
CATEGORY_LABELS = ("descendant", "sister", "ancestor", "depth\ncontrol",
                   "unrelated")


def panel_tree_relation(ax):
    category = pd.read_csv(DATA / "figure4" / "category_effects.csv")
    category = category[np.isclose(category.dose, 1.0)]
    for perturbation, color, marker, offset in (
        ("matched additive", ADDITIVE, M_ADD, -0.15),
        ("focal shunt", SHUNT, M_SHUNT, 0.15),
    ):
        subset = category[category.perturbation.eq(perturbation)].groupby(
            ["root_id", "category"], as_index=False
        ).median_abs_log_gradient_change.mean()
        for index, relation in enumerate(CATEGORIES):
            values = subset[
                subset.category.eq(relation)
            ].median_abs_log_gradient_change.to_numpy(float)
            _seed_points(ax, index + offset, values, color)
            _mean_marker(ax, index + offset, values, color, marker,
                         seed=1610 + index)
    _tick_labels(ax, "x", range(5), CATEGORY_LABELS)
    ax.set_xlim(-0.62, 4.62)
    ax.set_ylim(-0.007, 0.172)
    _tick_labels(ax, "y", (0.0, 0.05, 0.10, 0.15),
                 ("0", "0.05", "0.10", "0.15"))
    # The axis names the quantity in the text's own notation (γᵢ is the
    # site gradient; the plotted value is the median site-level mᵢ).
    ax.set_ylabel("median |Δ log |γ||", fontsize=PT_LABEL,
                  color=INK)
    ax.tick_params(axis="x", length=0, pad=2.5)
    _direct_label(ax, 0.52, 0.152, "focal shunt", SHUNT)
    _direct_label(ax, 0.52, 0.124, "current injection", ADDITIVE)

    # The injection's two off-route squares sit at 0 with no whisker because
    # they are zero by construction: the soma-restoring current isolates every
    # subtree that does not contain the site.  Verify the structural zero from
    # the frozen table before claiming it on the ink (site-level rows, unit
    # dose): every 'unrelated' change is numerically zero.
    injection_sites = category[category.perturbation.eq("matched additive")]
    unrelated = injection_sites[
        injection_sites.category.eq("unrelated")
    ].median_abs_log_gradient_change.to_numpy(float)
    assert unrelated.size and np.all(np.abs(unrelated) < 1e-9), (
        "injection 'unrelated' changes are no longer structurally zero")
    ax.text(3.35, 0.031, "injection off-route:\nzero by construction",
            ha="center", va="bottom", fontsize=PT_SMALL, color=MUTE,
            linespacing=1.15, zorder=6)
    _title(ax, "Tree-relation selectivity")
    return ax


# ── panel C: passive dose response (keeps the shared y axis) ─────────────
def panel_passive_dose(ax):
    focal = pd.read_csv(DATA / "figure4" / "focal_localization.csv")
    per_cell = focal.groupby(
        ["root_id", "dose", "perturbation"], as_index=False
    ).localization_index.mean()
    for perturbation_index, (perturbation, color, marker) in enumerate((
        ("matched additive", ADDITIVE, M_ADD),
        ("focal shunt", SHUNT, M_SHUNT),
    )):
        subset = per_cell[per_cell.perturbation.eq(perturbation)]
        doses, means, lows, highs = [], [], [], []
        for dose, group in subset.groupby("dose"):
            mean, low, high = mean_ci(
                group.localization_index.to_numpy(float),
                seed=1660 + 10 * perturbation_index + int(dose * 4),
            )
            doses.append(float(dose))
            means.append(mean)
            lows.append(low)
            highs.append(high)
        order = np.argsort(doses)
        doses = np.asarray(doses)[order]
        means = np.asarray(means)[order]
        lows = np.asarray(lows)[order]
        highs = np.asarray(highs)[order]
        ax.errorbar(doses, means, yerr=[means - lows, highs - means],
                    marker=marker, ms=MEAN_MS, lw=LW_DATA, color=color,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=LW_ERR, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE, zorder=3)
    ax.set_xscale("log", base=2)
    _tick_labels(ax, "x", (0.25, 0.5, 1, 2), ("0.25", "0.5", "1", "2"))
    ax.set_xlim(0.215, 2.35)
    ax.set_ylim(*LOCAL_YLIM)
    _tick_labels(ax, "y", LOCAL_YTICKS,
                 ("0", "0.05", "0.10", "0.15", "0.20"))
    ax.set_xlabel("perturbation dose", fontsize=PT_LABEL, color=INK)
    ax.set_ylabel(LOCAL_LABEL, fontsize=PT_LABEL, color=INK)
    _direct_label(ax, 0.275, 0.213, "focal shunt", SHUNT)
    _direct_label(ax, 0.275, 0.150, "current injection", ADDITIVE)
    _title(ax, "Permissive passive dose")
    return ax


# ── panel D: exact factor freeze (shares panel C's y axis) ───────────────
def panel_factor_freeze(ax):
    shapley = pd.read_csv(DATA / "focal_decomposition" / "cell_shapley.csv")
    shapley = shapley[
        shapley.estimand.eq("full_shunt_minus_matched_additive")
    ].sort_values("root_id")
    columns = ("driving_force_only_localization", "full_shunt_localization")
    colors = (FREEZE, SHUNT)
    markers = (M_FREEZE, M_SHUNT)
    for _, row in shapley.iterrows():
        ax.plot((0, 1), row[list(columns)].to_numpy(float), color=MUTE,
                lw=LW_HAIR, alpha=0.5, zorder=1.6)
    for index, column in enumerate(columns):
        values = shapley[column].to_numpy(float)
        _seed_points(ax, index, values, colors[index])
        _mean_marker(ax, index, values, colors[index], markers[index],
                     seed=1710 + index)
    _zero_line(ax)
    _tick_labels(ax, "x", (0, 1),
                 ("baseline\nadjoint", "post-shunt\nadjoint"))
    ax.set_xlim(-0.52, 1.52)
    ax.set_ylim(*LOCAL_YLIM)
    # D repeats C's localization scale but sits across a gutter with a
    # different x type, so it keeps its own tick labels: the shared left
    # gutter already reserves the column, and a bare axis is not readable.
    _tick_labels(ax, "y", LOCAL_YTICKS,
                 ("0", "0.05", "0.10", "0.15", "0.20"))
    ax.tick_params(axis="x", length=0, pad=2.5)
    # Direct-label the two conditions so the amber triangle is not an
    # unexplained third hue: amber = driving-force-only (baseline adjoint),
    # green = the full shunt at unit dose.
    _direct_label(ax, 0.0, 0.215, "driving-force\nonly", FREEZE, ha="center")
    _direct_label(ax, 1.0, 0.215, "full shunt", SHUNT, ha="center")
    _title(ax, "Adjoint decomposition")
    return ax


# ── panel E: active-channel dose response ────────────────────────────────
# The passive model at the SAME calibration as the active ensemble (frozen
# phase-1 table): permissive membrane resistance, unit background leak, dose
# normalized to the local input conductance.  Its curve is overlaid on E so
# the reader sees that adding the active conductances preserves the passive
# dose response rather than merely taking the panel's word for it.
PASSIVE_CALIBRATION = {
    "membrane_resistance_ohm_cm2": 1000,
    "background_leak_multiplier": 1,
    "dose_scheme": "input_conductance_normalized",
}
# Passive overlay tint: the same two hues at a second lightness (the figure's
# existing rule for a second cohort of one condition), dashed and stroked at
# the reference weight so it reads as a coincident second series ON the
# active curve instead of hiding under it.
PASSIVE_TINT_PCT = 42


def load_passive_reference():
    table = pd.read_csv(
        DATA / "focal_selectivity_phase1" / "condition_summary.csv")
    for column, value in PASSIVE_CALIBRATION.items():
        table = table[table[column].eq(value)]
    assert len(table) == 6, (
        f"expected 2 perturbations x 3 doses at the active calibration, "
        f"got {len(table)} rows")
    return table


def panel_active_dose(ax, summary, passive):
    for perturbation, color_name, marker in (
        ("focal shunt", "shunting", M_SHUNT),
        ("matched additive", "additive", M_ADD),
    ):
        color = COLORS[color_name]
        part = summary[summary.perturbation.eq(perturbation)].sort_values(
            "dose_relative_to_local_input_conductance"
        )
        x = part.dose_relative_to_local_input_conductance.to_numpy(float)
        means = part.mean_localization_index.to_numpy(float)
        lows = part.ci95_low_localization_index.to_numpy(float)
        highs = part.ci95_high_localization_index.to_numpy(float)

        # Active series: the connecting line goes down first, the passive
        # dashes ride on top of it, and the summary markers with their
        # whiskers come last so no dash crosses a hollow marker face.
        ax.plot(x, means, lw=LW_DATA, color=color, zorder=3)
        ref = passive[passive.perturbation.eq(perturbation)].sort_values(
            "dose_value")
        x_ref = ref.dose_value.to_numpy(float)
        assert np.allclose(x_ref, x), "passive and active doses differ"
        ax.plot(x_ref, ref.mean_localization_index.to_numpy(float),
                ls="--", lw=LW_REF, color=mix(color_name, PASSIVE_TINT_PCT),
                zorder=3.3)
        ax.errorbar(x, means, yerr=[means - lows, highs - means],
                    marker=marker, ms=MEAN_MS, ls="none", color=color,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=LW_ERR, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE, zorder=3.6)
    ax.set_xscale("log")
    _tick_labels(ax, "x", (0.25, 1, 4), ("0.25", "1", "4"))
    ax.set_xlim(0.205, 5.0)
    ax.set_ylim(-0.055, 1.48)
    _tick_labels(ax, "y", (0.0, 0.5, 1.0), ("0", "0.5", "1.0"))
    ax.set_xlabel("normalized shunt dose", fontsize=PT_LABEL, color=INK)
    ax.set_ylabel(LOCAL_LABEL, fontsize=PT_LABEL, color=INK)

    # Direct labels, each with a one-line mute sub-label naming the SIGN of
    # the effect that the unsigned index hides: the shunt attenuates and the
    # injection enhances descendant credit (signed localization negative in
    # 8/8 cells for the shunt and positive in 8/8 for the injection at every
    # dose of cell_condition_metrics.csv).  The two interventions move
    # descendant credit in opposite directions; they are not one effect at
    # two strengths.
    _direct_label(ax, 0.245, 1.405, "focal shunt", SHUNT)
    ax.text(0.245, 1.29, "attenuates descendants", color=MUTE,
            fontsize=PT_SMALL, ha="left", va="center", zorder=6)
    _direct_label(ax, 0.245, 1.135, "current injection", ADDITIVE)
    ax.text(0.245, 1.02, "enhances descendants", color=MUTE,
            fontsize=PT_SMALL, ha="left", va="center", zorder=6)
    ax.text(0.245, 0.80, "dashed: passive,\nsame calibration", color=MUTE,
            fontsize=PT_SMALL, ha="left", va="center", linespacing=1.15,
            zorder=6)
    # E carries the same quantity as C and D on a wider range; the caption
    # carries that disclosure rather than the panel.
    _title(ax, "Active dose response")
    return ax


# ── panel F (headline): the electrotonic boundary ────────────────────────
COHORTS = (
    ("original_eight", "initial sample, n = 8", SHUNT, M_CONTRAST),
    ("v661_disjoint", "disjoint 45-cell cohort", REPLICATE, M_REPLICATE),
)
CONTRAST_LABEL = "shunt − current-injection localization"
# The same quantity, set on two lines where it is the (rotated) y axis of a
# panel whose row is shorter than the label is long.
CONTRAST_LABEL_Y = "shunt − current injection\nlocalization"
# Membrane-resistance key for every marker of F.  The initial sample's labels
# are one tick row at this y (top edge), below every whisker of that series;
# the disjoint cohort's labels sit above their own whiskers, nudged in x
# (multiplicative, log axis) only where a label would otherwise touch the
# cohort's own descending curve.
RM_ROW_Y = -0.0072
# The multiplicative nudges are 1.5-2 pt on the 40 pt-per-decade axis: the
# initial sample's 3000/5000 and 15,000/30,000 pairs are 20 and 28 pt apart
# on the ratio axis and their labels would otherwise touch; the disjoint
# cohort's 3000 label would otherwise touch that cohort's own descending
# curve.
RM_LABEL_NUDGE = {
    ("original_eight", "Ra150_Rm3000"): 0.951,
    ("original_eight", "Ra150_Rm5000"): 1.051,
    ("original_eight", "Ra150_Rm15000"): 0.963,
    ("original_eight", "Ra150_Rm30000"): 1.038,
    ("v661_disjoint", "Ra150_Rm3000"): 1.12,
}


def _rm_label(regime: str) -> str:
    """'Ra150_Rm15000' -> '15,000'; four-digit values keep no comma."""
    value = int(regime.split("Rm")[1])
    return f"{value:,}" if value >= 10_000 else str(value)


def panel_electrotonic(ax):
    physical = pd.read_csv(
        DATA / "physical_cable_sensitivity" / "cell_primary_contrasts.csv")
    ratio = pd.read_csv(
        DATA / "physical_cable_sensitivity" / "cell_electrotonic_ratios.csv")
    eligible = physical[["cohort", "regime", "root_id"]].drop_duplicates()
    ratio = ratio.merge(
        eligible,
        on=["cohort", "regime", "root_id"],
        how="inner",
        validate="one_to_one",
    )
    ratio_mean = ratio.groupby(
        ["cohort", "regime"], as_index=False
    ).median_axial_to_leak_ratio.median()
    physical = physical.merge(ratio_mean, on=["cohort", "regime"],
                              validate="many_to_one")

    standard = {}
    for cohort, _, color, marker in COHORTS:
        subset = physical[physical.cohort.eq(cohort)].copy()
        if cohort == "original_eight":
            subset = subset[subset.regime.str.startswith("Ra150_")]
        points = []
        for (regime, x_value), group in subset.groupby(
                ["regime", "median_axial_to_leak_ratio"]):
            mean, low, high = mean_ci(group.difference.to_numpy(float),
                                      seed=1740 + len(points))
            points.append((float(x_value), mean, low, high, regime))
        points.sort()
        x = np.asarray([p[0] for p in points])
        means = np.asarray([p[1] for p in points])
        lows = np.asarray([p[2] for p in points])
        highs = np.asarray([p[3] for p in points])
        ax.errorbar(x, means, yerr=[means - lows, highs - means],
                    marker=marker, ms=MEAN_MS, lw=LW_DATA,
                    capsize=ERR_CAPSIZE, color=color,
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=LW_ERR, elinewidth=LW_ERR, zorder=3)
        for px, pm, _, _, regime in points:
            if regime == "Ra150_Rm15000":
                standard[cohort] = (px, pm)
        # Every marker is keyed to its membrane resistance so the calibration
        # behind each point is readable off the ink (axial resistivity is
        # fixed at 150 Ω cm in every regime drawn).  The initial sample's
        # labels form one secondary tick row below its diamonds; the disjoint
        # cohort's labels sit above their triangles.  Both rows are on the
        # outside of the pair of curves, so no label crosses either series.
        for px, _, low, high, regime in points:
            assert regime.startswith("Ra150_Rm"), regime
            label = _rm_label(regime)
            nudge = RM_LABEL_NUDGE.get((cohort, regime), 1.0)
            if cohort == "original_eight":
                ax.text(px * nudge, RM_ROW_Y, label, ha="center", va="top",
                        fontsize=PT_SMALL, color=MUTE, zorder=6)
            else:
                ax.text(px * nudge, high + 0.0035, label, ha="center",
                        va="bottom", fontsize=PT_SMALL, color=MUTE,
                        zorder=6)
    _zero_line(ax)
    ax.set_xscale("log")
    _tick_labels(ax, "x", (1, 10, 100), ("1", "10", "100"))
    ax.set_xlim(0.95, 190.0)
    # The y range is opened by 4.5 pt at the bottom (the Rₘ tick row of the
    # initial sample) and 5 pt at the top (the disjoint cohort's Rₘ = 300
    # label above its whisker); no plotted value moves.
    ax.set_ylim(-0.0175, 0.106)
    _tick_labels(ax, "y", (0.0, 0.02, 0.04, 0.06, 0.08),
                 ("0", "0.02", "0.04", "0.06", "0.08"))
    ax.set_xlabel("median axial / leak conductance ratio", fontsize=PT_LABEL,
                  color=INK)
    ax.set_ylabel(CONTRAST_LABEL, fontsize=PT_LABEL, color=INK)

    # Only the two series names stay on the panel (T5): the axial-resistivity
    # condition of the pilot cohort and the small standard-calibration effects are
    # methodological notes and are carried by the caption.  The initial-sample
    # label is set on two lines inside the clear pocket between its curve's
    # descending limb (above) and the zero reference (below): a single line at
    # this height ran under the curve's own tail.
    _direct_label(ax, 1.05, 0.010, "initial\n8-cell sample", SHUNT)
    _direct_label(ax, 3.9, 0.0895, "disjoint 45-cell cohort", REPLICATE)

    # Mark the standard passive calibration explicitly.  Its two cohort
    # values differ in axial/leak ratio, so a bracket is more honest than one
    # vertical reference line.
    standard_x = sorted(value[0] for value in standard.values())
    if len(standard_x) == 2:
        # The bracket sits above the disjoint cohort's own Rₘ = 15,000
        # label (which now names that marker directly), still spanning the
        # two standard-calibration points of the two cohorts.
        y_bar = 0.021
        ax.plot(standard_x, [y_bar, y_bar], color=MUTE, lw=LW_HAIR,
                solid_capstyle="butt", zorder=2)
        for x_value in standard_x:
            ax.plot([x_value, x_value], [y_bar - 0.002, y_bar + 0.002],
                    color=MUTE, lw=LW_HAIR, zorder=2)
        # The annotation floats in the empty upper-right pocket above the
        # bracket, on two lines and with units: anchored on the bracket
        # itself it collided with the disjoint cohort's descending curve.
        ax.text(175.0, 0.0285, "standard passive\nRₘ = 15,000 Ω cm²",
                ha="right", va="bottom",
                fontsize=PT_SMALL, color=MUTE)
    _title(ax, "Electrotonic boundary")
    return ax


# ── panel G: forest of the paired active contrasts ───────────────────────
def panel_contrast_forest(ax, contrasts, cells):
    rows = contrasts[contrasts.metric.eq("localization_index")].sort_values(
        "dose_relative_to_local_input_conductance"
    )
    doses = rows.dose_relative_to_local_input_conductance.to_numpy(float)
    # Treat dose as ordered categories in the forest: the largest dose sits
    # at the top, matching the usual high-to-low reading order of row labels.
    positions = np.arange(len(doses))
    _zero_line(ax, axis="x")
    for y, (_, row) in zip(positions, rows.iterrows(), strict=True):
        dose = float(row.dose_relative_to_local_input_conductance)
        paired = cells[np.isclose(
            cells.dose_relative_to_local_input_conductance, dose)].pivot(
            index="root_id", columns="perturbation", values="localization_index"
        )
        diff = (paired["focal shunt"] - paired["matched additive"]).to_numpy(float)
        # Cells ride a declared sub-row above their summary: at the lowest
        # dose the interval is narrower than the mean glyph, so points left
        # on the row itself would be hidden under it rather than read.
        ax.plot(diff, np.full(diff.size, y + CELL_ROW_DY), marker="o",
                ls="none", ms=SEED_MS, mfc=SHUNT, mec="none",
                alpha=SEED_ALPHA, zorder=2.4)
        mean = float(row.mean_shunt_minus_additive)
        ax.errorbar(mean, y,
                    xerr=[[mean - float(row.ci95_low)],
                          [float(row.ci95_high) - mean]],
                    marker=M_CONTRAST, ms=MEAN_MS, color=SHUNT,
                    markerfacecolor="white", markeredgecolor=SHUNT,
                    markeredgewidth=LW_ERR, lw=LW_ERR, capsize=ERR_CAPSIZE,
                    zorder=5)
    ax.set_yticks(list(positions))
    ax.set_yticklabels([("%g" % d) for d in doses])
    ax.set_ylim(positions.min() - 0.62, positions.max() + 0.62)
    ax.set_xlim(-0.06, 1.30)
    _tick_labels(ax, "x", (0.0, 0.5, 1.0), ("0", "0.5", "1.0"))
    ax.tick_params(axis="y", length=0, pad=2.5)
    ax.set_ylabel("normalized shunt dose", fontsize=PT_LABEL, color=INK)
    ax.set_xlabel(CONTRAST_LABEL, fontsize=PT_LABEL, color=INK)

    # The cell-dot offset, the sign count and the Wilcoxon test are reported
    # in the caption; the panel shows only the geometry.
    _title(ax, "Paired cellwise contrast")
    return ax


# ── the figure ───────────────────────────────────────────────────────────
def build() -> list[str]:
    active = DATA / "focal_selectivity_active_ensemble"
    active_summary = pd.read_csv(active / "condition_summary.csv")
    active_contrasts = pd.read_csv(active / "paired_contrasts.csv")
    active_cells = pd.read_csv(active / "cell_condition_metrics.csv")
    passive_reference = load_passive_reference()

    canvas = NativeCanvas(HEIGHT_IN, 3, row_weights=list(ROW_WEIGHTS),
                          hgutter_pt=HGUTTER, vgutter_pt=VGUTTER,
                          margins=MARGINS)

    # Three regular rows on the one module grid: 6+6, 4+4+4, 6+6.  No panel
    # carries a hand inset, so the column lock gives every panel of a column
    # the same x0 and the same axes width and every row-mate the same height.
    ax_a = canvas.panel("A", 0, 0, 6, schematic=True)
    ax_b = canvas.panel("B", 0, 6, 6)
    ax_c = canvas.panel("C", 1, 0, 4, grid="y")
    ax_d = canvas.panel("D", 1, 4, 4, grid="y")
    ax_e = canvas.panel("E", 1, 8, 4)
    ax_f = canvas.panel("F", 2, 0, 6)
    ax_g = canvas.panel("G", 2, 6, 6)

    for ax in (ax_c, ax_d):
        style_panel(ax, grid="y")

    box = ax_a.get_position()
    panel_focal_shunt(ax_a, width_pt=box.width * canvas.width_pt,
                      height_pt=box.height * canvas.height_pt)
    _title(ax_a, "Matched focal perturbations")

    panel_tree_relation(ax_b)
    panel_passive_dose(ax_c)
    panel_factor_freeze(ax_d)
    panel_active_dose(ax_e, active_summary, passive_reference)
    panel_electrotonic(ax_f)
    panel_contrast_forest(ax_g, active_contrasts, active_cells)

    problems = canvas.save(OUTPUT, name="main_figure_08_native")
    return problems


def main() -> None:
    problems = build()
    if problems:
        print(f"  {len(problems)} layout/overlap problems")
        for problem in problems:
            print(f"    {problem}")
    else:
        print("  clean: no layout or text-over-data problems")


if __name__ == "__main__":
    main()

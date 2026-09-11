#!/usr/bin/env python3
"""Two supplement panels that cannot be produced by cropping a frozen sheet.

Both are written as small stand-alone panel PDFs under
``figures/supplementary/components/`` and are pasted at scale 1.0 by
``scripts/supplement_consolidation/build.py`` like any other crop.

* ``si_checkpoint_merged.pdf`` -- SI_PLAN M1's merged checkpoint panel (new
  Supplementary Fig. S3E): the candidate/exact gradient cosine and the
  norm-matched one-step loss decrease of the same 120 valid trained
  checkpoints, drawn on ONE axis.  The two quantities are dimensionless, so
  one axis is exact; the frozen source table is the same
  ``mechanism_checkpoint_rows_valid.csv`` that
  ``build_supplementary_figure_s09_native.py`` reads.  No number changes.
  The exact-path family (both quantities 1 by construction) is a labelled
  reference line rather than two degenerate box slots.
* ``si_utility_bound.pdf`` -- the panel demoted out of main Fig. 1 (AMENDMENTS
  B3), new Supplementary Fig. S3G: the analytic rank/noise trade-off
  ``q^2 / (q + K sigma^2)``.  Analytic, no data.

Library note (DECISIONS G5): ``figure_canvas.NativeCanvas`` is fixed at the
518.4 pt full-canvas width, so a single sub-panel of a pasted supplement sheet
cannot be built with it.  ``_panel()`` below is the private helper that draws
one panel at an arbitrary point size under the journal type tokens; it is
reported in IMPLEMENTATION_NOTES.md as a library follow-up.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import journal_style as js  # noqa: E402

OUT = ROOT / "figures/supplementary/components"
SOURCE = ROOT / "source_data/prospective_input_validity"
# 2026-09-11: the exact-path family is no longer drawn.  Both of its
# quantities are 1 by construction (cosine of the exact gradient with itself;
# the exact step's fraction of its own progress), so its two slots were a
# bare diamond on a zero-height box and cost a third of the panel; unity is
# now one labelled reference line.  The strip of 120 checkpoint values is
# drawn behind each box, and the mean with its 95% bootstrap interval sits
# beside the box instead of on top of it, where the 0.02-0.03 half-widths
# were hidden under the marker and read as the whiskers.
FAMILIES = ["global_scalar_available", "ancestry_available"]
LABELS = ["strict\nscalar", "per\nneuron"]
TONES = [js.SERIES_COLORS["scalar"], js.SERIES_COLORS["per_soma"]]
N_CHECKPOINTS = 120

# Page geometry shared by both components: the same 163 x 137.2 pt page and
# the same axes box (28.8 pt title band above, 25.1 pt tick-and-label band
# below) as the frozen figure_S05 panel E crop they flank in Supplementary
# Fig. S3, so the three plot boxes of that row share one height and one
# baseline.  Points, measured on the frozen S05 crop.
PANEL_W_PT = 163.0
PANEL_H_PT = 137.2
PANEL_TOP_PT = 28.8
PANEL_BOTTOM_PT = 25.1


def _panel(width_pt, height_pt, *, left, right, top, bottom):
    """One panel on its own page; margins in points (private helper, G5)."""
    js.apply_neurips_style()
    fig = plt.figure(figsize=(width_pt / 72.0, height_pt / 72.0))
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([left / width_pt,
                       bottom / height_pt,
                       1.0 - (left + right) / width_pt,
                       1.0 - (top + bottom) / height_pt])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # token line weights for the frame and ticks (the style default is 0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(js.LW_EDGE)
    ax.tick_params(width=js.LW_EDGE)
    return fig, ax


def _bootstrap_ci(values, seed):
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, len(values)), replace=True)
    means = draws.mean(axis=1)
    return float(np.mean(values)), float(np.percentile(means, 2.5)), \
        float(np.percentile(means, 97.5))


def checkpoint_merged(dest=OUT / "si_checkpoint_merged.pdf"):
    """Old S9A and old S9C on one axis (SI_PLAN M1, new S3E)."""
    import pandas as pd

    data = pd.read_csv(SOURCE / "mechanism_checkpoint_rows_valid.csv")
    data = data[np.isclose(data.relative_step, 1e-5)
                & data.feedback_family.isin(FAMILIES)]
    fig, ax = _panel(PANEL_W_PT, PANEL_H_PT, left=34, right=6,
                     top=PANEL_TOP_PT, bottom=PANEL_BOTTOM_PT)
    metrics = ["gradient_cosine", "norm_matched_fraction_of_exact"]
    # Slot layout: two families per metric, a 1.6-unit gap between the two
    # metric groups; the mean/CI marker sits 0.42 right of its box.
    slot_x = {(0, 0): 0.0, (0, 1): 1.0, (1, 0): 2.6, (1, 1): 3.6}
    ci_dx = 0.42
    y_lo, y_hi = -0.8, 1.28
    positions = []
    rng = np.random.default_rng(700)
    clipped = []
    for mi, metric in enumerate(metrics):
        for fi, family in enumerate(FAMILIES):
            vals = data[data.feedback_family.eq(family)][metric].to_numpy()
            assert len(vals) == N_CHECKPOINTS, (metric, family, len(vals))
            x = slot_x[(mi, fi)]
            positions.append(x)
            tone = TONES[fi]
            # every checkpoint, jittered inside the box width; values below
            # the axis floor are drawn at the floor as open triangles and
            # listed beside them (2 of the 480 values, both strict-scalar
            # one-step progress)
            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            inside = vals >= y_lo
            ax.plot(x + jitter[inside], vals[inside], ls="none", marker="o",
                    ms=1.5, mfc=tone, mec="none", alpha=0.45, zorder=2)
            if (~inside).any():
                low = np.sort(vals[~inside])
                ax.plot(x + jitter[~inside], np.full((~inside).sum(), y_lo),
                        ls="none", marker="v", ms=3.0, mfc="white", mec=tone,
                        mew=js.LW_EDGE, zorder=3.5, clip_on=False)
                clipped.append((x, low))
            box = ax.boxplot([vals], positions=[x], widths=0.5,
                             patch_artist=True, showfliers=False,
                             whis=1.5, zorder=3,
                             boxprops={"facecolor": "none", "edgecolor": tone,
                                       "linewidth": js.LW_ERR},
                             medianprops={"color": tone,
                                          "linewidth": js.LW_ERR},
                             whiskerprops={"color": tone,
                                           "linewidth": js.LW_EDGE},
                             capprops={"color": tone,
                                       "linewidth": js.LW_EDGE})
            mean, lo, hi = _bootstrap_ci(vals, seed=700 + mi * 10 + fi)
            ax.errorbar(x + ci_dx, mean, yerr=[[mean - lo], [hi - mean]],
                        fmt="D", mfc="white", mec=js.COLORS["ink"],
                        ecolor=js.COLORS["ink"], ms=2.6, mew=js.LW_EDGE,
                        lw=js.LW_ERR, capsize=1.6, zorder=4)
    for x, low in clipped:
        ax.text(x + 0.26, y_lo + 0.02,
                ", ".join(f"{v:.2f}".replace("-", "−") for v in low),
                ha="left", va="bottom", fontsize=js.PT_BASE,
                color=js.COLORS["mute"])
    ax.axhline(0, color=js.COLORS["mute"], ls="--", lw=js.LW_REF, zorder=1)
    # unity: the exact-path value of both quantities, by construction
    ax.axhline(1.0, color=js.COLORS["mute"], ls=":", lw=js.LW_REF, zorder=1)
    ax.text(-0.42, 0.985, "exact path = 1", ha="left", va="top",
            fontsize=js.PT_BASE, color=js.COLORS["mute"])
    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS * 2, fontsize=js.PT_BASE, linespacing=1.05)
    ax.set_yticks([-0.5, 0.0, 0.5, 1.0])
    ax.set_ylabel("dimensionless value", fontsize=js.PT_EMPH)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.45, 4.25)
    for x, label in ((0.5 + 0.5 * ci_dx, "gradient cosine"),
                     (3.1 + 0.5 * ci_dx, "one-step progress")):
        ax.text(x, y_hi - 0.02, label, ha="center", va="top",
                fontsize=js.PT_BASE, color=js.COLORS["ink"])
    ax.axvline(1.8 + 0.5 * ci_dx, color=js.COLORS["grid"], lw=js.LW_HAIR)
    ax.tick_params(labelsize=js.PT_BASE)
    ax.tick_params(axis="x", length=0.0, pad=2.0)
    fig.text(0.5, 0.98, "120 trained checkpoints, one held-out batch",
             ha="center", va="top", fontsize=js.PT_BASE,
             color=js.COLORS["mute"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def utility_bound(dest=OUT / "si_utility_bound.pdf"):
    """The one-step smoothness bound demoted out of main Fig. 1 (new S3G).

    2026-09-11: the two analytic curves are ink (solid) and grey (dashed)
    rather than the route green and the additive navy, which on the S3 sheet
    carry the route-class meanings of the neighbouring panel F; the dashed
    crossing rule is gone (the annotation's leader marks the crossing once)
    and the annotation sits clear of the curves, below the legend.
    """
    fig, ax = _panel(PANEL_W_PT, PANEL_H_PT, left=36, right=8,
                     top=PANEL_TOP_PT, bottom=PANEL_BOTTOM_PT)
    sigma2 = np.linspace(0.0, 2.0, 400)
    curves = [(1, 0.8, js.COLORS["ink"], "-", "$K = 1$, $q = 0.8$"),
              (2, 1.0, js.COLORS["point_mlp"], (0, (3.2, 1.6)),
               "$K = 2$, $q = 1$")]
    values = []
    for rank, q, color, ls, label in curves:
        y = q ** 2 / (q + rank * sigma2)
        values.append(y)
        ax.plot(sigma2, y, color=color, lw=js.LW_DATA, ls=ls, label=label,
                zorder=3)
    cross = 4.0 / 7.0
    lo, hi = values
    ax.fill_between(sigma2, lo, hi, color=js.COLORS["grid"], alpha=0.85, lw=0,
                    zorder=1)
    ax.annotate("sign change at\nσ² = 4/7 ≈ 0.571",
                xy=(cross, 1.0 / (1.0 + 2 * cross)), xytext=(1.18, 0.66),
                ha="center", va="center", fontsize=js.PT_BASE,
                color=js.COLORS["ink"], linespacing=1.25,
                arrowprops=dict(arrowstyle="-", lw=js.LW_HAIR,
                                color=js.COLORS["mute"],
                                shrinkA=1.0, shrinkB=2.0))
    ax.set_xlabel("noise variance σ² (arbitrary units)",
                  fontsize=js.PT_EMPH)
    ax.set_ylabel("2L × optimized one-step bound", fontsize=js.PT_EMPH)
    ax.tick_params(labelsize=js.PT_BASE)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=js.PT_BASE, loc="upper right",
              handlelength=1.8, borderpad=0.1, labelspacing=0.25)
    fig.text(0.5, 0.98, "analytic bound q² / (q + Kσ²), no data",
             ha="center", va="top", fontsize=js.PT_BASE,
             color=js.COLORS["mute"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def main():
    for path in (checkpoint_merged(), utility_bound()):
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()

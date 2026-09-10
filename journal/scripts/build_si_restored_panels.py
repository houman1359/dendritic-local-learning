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
FAMILIES = ["global_scalar_available", "ancestry_available", "exact_transport"]
LABELS = ["strict scalar", "per-neuron", "exact path"]
TONES = [js.SERIES_COLORS["scalar"], js.SERIES_COLORS["per_soma"],
         js.SERIES_COLORS["oracle"]]


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
    fig, ax = _panel(163.0, 148.0, left=34, right=9, top=26, bottom=30)
    metrics = [("gradient_cosine", "cosine with exact gradient"),
               ("norm_matched_fraction_of_exact", "fraction of exact progress")]
    positions = []
    for mi, (metric, _) in enumerate(metrics):
        for fi, family in enumerate(FAMILIES):
            vals = data[data.feedback_family.eq(metric and family)][metric]
            vals = vals.to_numpy()
            assert len(vals) == 120, (metric, family, len(vals))
            x = mi * 3.6 + fi
            positions.append(x)
            box = ax.boxplot([vals], positions=[x], widths=0.62,
                             patch_artist=True, showfliers=False,
                             medianprops={"color": "white", "linewidth": 1.0})
            box["boxes"][0].set_facecolor(TONES[fi])
            box["boxes"][0].set_alpha(0.6)
            box["boxes"][0].set_linewidth(js.LW_EDGE)
            for key in ("whiskers", "caps"):
                for art in box[key]:
                    art.set_linewidth(js.LW_EDGE)
            mean, lo, hi = _bootstrap_ci(vals, seed=700 + mi * 10 + fi)
            ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="D",
                        mfc="white", color=js.COLORS["ink"], ms=2.6,
                        lw=js.LW_ERR, capsize=1.6)
    ax.axhline(0, color=js.COLORS["mute"], ls="--", lw=js.LW_REF)
    ax.set_xticks(positions)
    ax.set_xticklabels(LABELS * 2, rotation=38, ha="right",
                       fontsize=js.PT_BASE)
    ax.set_ylabel("dimensionless value", fontsize=js.PT_EMPH)
    ax.set_ylim(-0.62, 1.32)
    ax.set_xlim(-0.75, 5.75)
    ax.text(0.24, 0.985, "gradient cosine", ha="center", va="top",
            transform=ax.transAxes, fontsize=js.PT_BASE,
            color=js.COLORS["ink"])
    ax.text(0.76, 0.985, "one-step progress", ha="center", va="top",
            transform=ax.transAxes, fontsize=js.PT_BASE,
            color=js.COLORS["ink"])
    ax.axvline(2.3, color=js.COLORS["grid"], lw=js.LW_HAIR)
    ax.tick_params(labelsize=js.PT_BASE)
    fig.text(0.5, 0.985, "120 trained checkpoints, relative step $10^{-5}$",
             ha="center", va="top", fontsize=js.PT_BASE,
             color=js.COLORS["mute"])
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def utility_bound(dest=OUT / "si_utility_bound.pdf"):
    """The one-step smoothness bound demoted out of main Fig. 1 (new S3G)."""
    fig, ax = _panel(163.0, 148.0, left=36, right=6, top=22, bottom=32)
    sigma2 = np.linspace(0.0, 2.0, 400)
    curves = [(1, 0.8, js.SERIES_COLORS["shunting"], "$K = 1$, $q = 0.8$"),
              (2, 1.0, js.SERIES_COLORS["additive"], "$K = 2$, $q = 1$")]
    values = []
    for rank, q, color, label in curves:
        y = q ** 2 / (q + rank * sigma2)
        values.append(y)
        ax.plot(sigma2, y, color=color, lw=js.LW_DATA, label=label)
    cross = 4.0 / 7.0
    lo, hi = values
    ax.fill_between(sigma2, lo, hi, color=js.COLORS["grid"], alpha=0.85, lw=0)
    ax.axvline(cross, color=js.COLORS["mute"], ls="--", lw=js.LW_REF)
    ax.annotate("sign change at\n$\\sigma^2 = 4/7 \\approx 0.571$",
                xy=(cross, q ** 2 / (q + 2 * cross)), xytext=(0.09, 0.10),
                fontsize=js.PT_BASE, color=js.COLORS["ink"],
                arrowprops=dict(arrowstyle="-", lw=js.LW_HAIR,
                                color=js.COLORS["mute"]))
    ax.set_xlabel("noise variance $\\sigma^2$ (arbitrary units)",
                  fontsize=js.PT_EMPH)
    ax.set_ylabel("$2L\\times$ optimized one-step bound", fontsize=js.PT_EMPH)
    ax.tick_params(labelsize=js.PT_BASE)
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=js.PT_BASE, loc="upper right",
              handlelength=1.4, borderpad=0.1, labelspacing=0.25)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)
    return dest


def main():
    for path in (checkpoint_merged(), utility_bound()):
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()

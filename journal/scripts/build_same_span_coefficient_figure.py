#!/usr/bin/env python3
"""Render the same-span coefficient-learning bias--variance experiment (S14).

One native full-width canvas, three rows of two panels:

  A  schematic: one rank-8 route span, three coordinate systems
  B  positive Gram spectrum of the three dictionaries (deterministic
     construction from the frozen config; no fit, no sampling)
  C  low-data trajectories (n = 4)         D  high-data trajectories (n = 256)
  E  final loss versus effective sample size on a log axis, with the
     per-seed predicted crossover sample sizes drawn as a strip above the
     axes and the per-seed medians dropped into the panel as guides
  F  paired nested-minus-Haar contrasts with the exact finite-time risk

Every number is read from source_data/same_span_coefficient_learning; the
runner module is imported only for ``dictionaries()``, which builds the three
frozen dictionaries from the config without training anything.

2026-09-11 (supplement visual review, sheet S5): rebuilt on the native
canvas so the two rows the supplement pastes (A/B and E/F) fill the sheet
width; E moved to a log y axis so the n = 64 crossing is legible; the
per-seed predicted crossovers (a declared source of the sheet) are drawn;
series keys carry marker glyphs and E carries direct labels; the grey
gate note left panel A (the caption states the number once); the schematic
labels fit their boxes and use the legend's names.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import NullLocator

from figure_canvas import (
    COLORS,
    LW_DATA,
    LW_HAIR,
    LW_REF,
    PT_ANNOT,
    PT_LEGEND,
    PT_TITLE,
    Margins,
    NativeCanvas,
    tint_patch,
)
from journal_style import style_direct_color_labels


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "same_span_coefficient_learning"
CONFIG = ROOT / "configs" / "credit_phase_theory" / "same_span_learning_confirmatory.json"
FIGURES = ROOT / "figures" / "generated"
STEM = "fig_same_span_coefficient_learning"

# the per-seed crossover strip above panel E, in points
STRIP_H_PT = 28.0
STRIP_GAP_PT = 4.0
TITLE_PAD_PT = STRIP_GAP_PT + STRIP_H_PT + 5.0


def load_runner():
    path = ROOT / "scripts" / "run_same_span_coefficient_learning.py"
    spec = importlib.util.spec_from_file_location("same_span_figure_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()
# one name per series everywhere: schematic boxes, legends, direct labels
STYLES = {
    "tree_haar": ("Haar", COLORS["shunting"], "o"),
    "raw_nested_indicators": ("raw nested", COLORS["per_soma"], "s"),
    "static_gain_scaled_nested": ("scaled nested", COLORS["additive"], "D"),
}
MS = 3.4
# text set inside its own tint patch is an intentional overlay, not text on
# a datum; the live audit recognises the (invisible) bbox patch
ON_PATCH = dict(facecolor="none", edgecolor="none", pad=0.0)
MARK = dict(ms=MS, lw=LW_DATA, markeredgecolor="white", markeredgewidth=LW_HAIR)


def log2_ticks(ax, values):
    ax.set_xscale("log", base=2)
    ax.set_xticks(values, [str(v) for v in values])
    ax.xaxis.set_minor_locator(NullLocator())


def plain_log_yticks(ax, values):
    ax.set_yscale("log")
    ax.set_yticks(values, [f"{v:g}" for v in values])
    ax.yaxis.set_minor_locator(NullLocator())


def schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # one span box on the left, three coordinate boxes on the right
    tint_patch(ax, ("rect", 0.02, 0.34, 0.36, 0.32), color="dend")
    ax.text(0.20, 0.50, "rank-8\nroute span", ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["ink"], linespacing=1.15,
            bbox=ON_PATCH)
    ax.text(0.20, 0.13, "same projector\ndifferent Gram matrix",
            ha="center", va="center", fontsize=PT_ANNOT,
            color=COLORS["mute"], linespacing=1.15)
    rows = [(0.84, "tree_haar"), (0.50, "raw_nested_indicators"),
            (0.16, "static_gain_scaled_nested")]
    for y, key in rows:
        label, color, _ = STYLES[key]
        tint_patch(ax, ("rect", 0.62, y - 0.13, 0.36, 0.26), color=color)
        ax.text(0.80, y, label, ha="center", va="center", fontsize=PT_ANNOT,
                color=color, bbox=ON_PATCH)
        ax.add_patch(FancyArrowPatch((0.39, 0.50), (0.61, y), arrowstyle="-|>",
                                     mutation_scale=6, lw=LW_REF,
                                     color=COLORS["mute"],
                                     shrinkA=0, shrinkB=0))


def trajectory(ax: plt.Axes, summary: pd.DataFrame, sample_size: int) -> None:
    part = summary[
        summary.effective_sample_size.eq(sample_size)
        & summary.optimizer.eq("vanilla_local")
    ]
    for parameterization, (label, color, marker) in STYLES.items():
        curve = part[part.parameterization.eq(parameterization)].sort_values("checkpoint")
        ax.fill_between(curve.checkpoint, curve.ci95_low_population_loss,
                        curve.ci95_high_population_loss, color=color, alpha=0.14,
                        linewidth=0)
        ax.plot(curve.checkpoint, curve.mean_population_loss, color=color,
                marker=marker, label=label, **MARK)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 20, 80], ["1", "5", "20", "80"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("coefficient updates")
    ax.set_ylabel("population loss")


def crossover_strip(fig, ax_e, crossovers, xlim):
    """Per-seed predicted crossover sample sizes, one row per nested series.

    Drawn as a satellite axes above panel E on the same log2 x axis, so a
    crossing the sampled sizes 4/16/64/256 only interpolate is shown with
    its seed spread and its median.
    """
    box = ax_e.get_position()
    h_pt = fig.get_size_inches()[1] * 72.0
    y0 = box.y1 + STRIP_GAP_PT / h_pt
    strip = fig.add_axes([box.x0, y0, box.width, STRIP_H_PT / h_pt])
    strip.set_xscale("log", base=2)
    strip.set_xlim(*xlim)
    strip.set_ylim(-0.75, 1.75)
    strip.set_xticks([])
    strip.set_yticks([])
    strip.xaxis.set_minor_locator(NullLocator())
    for spine in strip.spines.values():
        spine.set_visible(False)
    strip.patch.set_alpha(0.0)
    rng = np.random.default_rng(20260911)   # visual jitter only
    rows = [("raw_nested_indicators", 1.0, "left"),
            ("static_gain_scaled_nested", 0.0, "right")]
    medians = {}
    for key, y, side in rows:
        label, color, marker = STYLES[key]
        values = np.sort(crossovers[crossovers.parameterization.eq(key)]
                         .predicted_effective_sample_crossover.to_numpy())
        jitter = rng.uniform(-0.3, 0.3, size=len(values))
        strip.scatter(values, y + jitter, s=5.0, color=color, alpha=0.45,
                      linewidths=0, zorder=2)
        median = float(np.median(values))
        medians[key] = median
        strip.plot([median], [y], marker=marker, color=color, ms=MS + 0.6,
                   markeredgecolor="white", markeredgewidth=LW_HAIR, lw=0,
                   zorder=3)
        strip.axvline(median, color=color, ls=":", lw=LW_HAIR, zorder=1)
        if side == "left":
            strip.text(xlim[0] * 1.04, y, label, ha="left", va="center",
                       fontsize=PT_ANNOT, color=color)
        else:
            strip.text(xlim[1] / 1.08, y, label, ha="right", va="center",
                       fontsize=PT_ANNOT, color=color)
    return strip, medians


def main() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    contrasts = pd.read_csv(SOURCE / "paired_contrasts.csv")
    predictions = pd.read_csv(SOURCE / "theory_predictions.csv")
    crossovers = pd.read_csv(SOURCE / "predicted_sample_crossovers.csv")
    dictionaries = RUNNER.dictionaries(cfg)
    sizes = [int(v) for v in cfg["effective_sample_sizes"]]

    # Two module columns.  The gutter equals the widest y decoration of the
    # right column (28.3 pt: tick labels plus the rotated label), so that
    # column reserves exactly the 8 pt lock pad on its left while the left
    # column reserves the same 8 pt on its right, and the two columns keep
    # one identical axes width; the right column's y label then also starts
    # to the right of the supplement's letter-based column split.
    cv = NativeCanvas(6.85, 3, row_weights=[1.0, 1.0, 1.32], module_cols=2,
                      hgutter_pt=28.3,
                      margins=Margins(left=40.0, right=13.0, top=16.0, bottom=26.0))
    ax_a = cv.panel("A", 0, 0, 1, title="Same address span", schematic=True)
    ax_b = cv.panel("B", 0, 1, 1, title="Different conditioning")
    ax_c = cv.panel("C", 1, 0, 1, title="Low data, n = 4")
    ax_d = cv.panel("D", 1, 1, 1, title="High data, n = 256")
    ax_e = cv.panel("E", 2, 0, 1, title="Final-loss crossover")
    ax_f = cv.panel("F", 2, 1, 1, title="Risk reversal")
    for ax in (ax_e, ax_f):
        ax.set_title(ax.get_title(), fontsize=PT_TITLE, color=COLORS["ink"],
                     pad=TITLE_PAD_PT, fontweight="normal")
    cv.declare_reserve("E", top=STRIP_H_PT + STRIP_GAP_PT)
    cv.declare_reserve("F", top=STRIP_H_PT + STRIP_GAP_PT)

    schematic(ax_a)

    # B: the Gram spectrum of each frozen dictionary
    for parameterization, dictionary in dictionaries.items():
        eigenvalues = np.linalg.eigvalsh(dictionary @ dictionary.T)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        eigenvalues = np.sort(eigenvalues / eigenvalues.max())[::-1]
        label, color, marker = STYLES[parameterization]
        ax_b.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, color=color,
                  marker=marker, label=label, **MARK)
    plain_log_yticks(ax_b, [0.001, 0.01, 0.1, 1])
    ax_b.set_ylim(0.0012, 1.6)
    ax_b.set_xticks(range(1, 9))
    ax_b.set_xlim(0.6, 8.4)
    ax_b.set_xlabel("positive Gram mode")
    ax_b.set_ylabel("eigenvalue / largest eigenvalue")
    ax_b.legend(loc="lower left", frameon=False, fontsize=PT_LEGEND,
                handlelength=1.6, handletextpad=0.5, labelspacing=0.25,
                borderaxespad=0.2)

    trajectory(ax_c, summary, 4)
    trajectory(ax_d, summary, 256)
    ax_d.legend(loc="upper right", frameon=False, fontsize=PT_LEGEND,
                handlelength=1.6, handletextpad=0.5, labelspacing=0.25,
                borderaxespad=0.2)

    # E: final loss on a log axis, direct labels, per-seed crossover guides
    final = summary[
        summary.checkpoint.eq(int(cfg["iterations"]))
        & summary.optimizer.eq("vanilla_local")
    ]
    xlim = (sizes[0] / 1.25, sizes[-1] * 1.25)
    ends = {}
    for parameterization, (label, color, marker) in STYLES.items():
        part = final[final.parameterization.eq(parameterization)].sort_values(
            "effective_sample_size"
        )
        ax_e.fill_between(part.effective_sample_size, part.ci95_low_population_loss,
                          part.ci95_high_population_loss, color=color, alpha=0.14,
                          linewidth=0)
        ax_e.plot(part.effective_sample_size, part.mean_population_loss,
                  color=color, marker=marker, label=label, zorder=3, **MARK)
        last = part.iloc[-1]
        ends[parameterization] = (float(last.ci95_low_population_loss),
                                  float(last.ci95_high_population_loss))
    log2_ticks(ax_e, sizes)
    ax_e.set_xlim(*xlim)
    plain_log_yticks(ax_e, [0.01, 0.03, 0.1, 0.3])
    lo = float(final.ci95_low_population_loss.min())
    hi = float(final.ci95_high_population_loss.max())
    ax_e.set_ylim(lo / 1.9, hi * 1.25)
    ax_e.set_xlabel("effective sample size")
    ax_e.set_ylabel("loss after 80 updates")
    # direct end-of-curve labels: the two nested series above their band,
    # Haar below its band, all right-aligned on the last sampled size
    for key, above in (("static_gain_scaled_nested", True),
                       ("raw_nested_indicators", True), ("tree_haar", False)):
        label, color, _ = STYLES[key]
        band_lo, band_hi = ends[key]
        y = band_hi * 1.16 if above else band_lo / 1.16
        ax_e.text(sizes[-1], y, label, ha="right", va="bottom" if above else "top",
                  fontsize=PT_ANNOT, color=color, zorder=4)

    # F: paired contrasts with the exact finite-time risk
    pair_styles = [
        ("raw_nested", "raw nested − Haar", COLORS["per_soma"], "s"),
        ("static_gain_scaled_nested", "scaled nested − Haar", COLORS["additive"], "D"),
    ]
    prediction_final = predictions[
        predictions.checkpoint.eq(int(cfg["iterations"]))
        & predictions.optimizer.eq("vanilla_local")
    ]
    pred_means = prediction_final.groupby(
        ["effective_sample_size", "parameterization"]
    ).total_expected_loss.mean().unstack()
    for prefix, label, color, marker in pair_styles:
        part = contrasts[
            contrasts.left_minus_right.str.startswith(prefix)
            & contrasts.left_minus_right.str.endswith("tree_haar / vanilla_local")
        ].sort_values("effective_sample_size")
        ax_f.errorbar(
            part.effective_sample_size,
            part.mean_loss_difference,
            yerr=np.vstack([
                part.mean_loss_difference - part.ci95_low,
                part.ci95_high - part.mean_loss_difference,
            ]),
            color=color, marker=marker, capsize=2, label=label, zorder=3, **MARK,
        )
        parameterization = (
            "raw_nested_indicators" if prefix == "raw_nested"
            else "static_gain_scaled_nested"
        )
        predicted_difference = (
            pred_means[parameterization] - pred_means["tree_haar"]
        )
        ax_f.plot(predicted_difference.index, predicted_difference.values,
                  color=color, ls="--", lw=LW_REF, zorder=2)
    ax_f.axhline(0, color=COLORS["mute"], ls=":", lw=LW_REF)
    log2_ticks(ax_f, sizes)
    ax_f.set_xlim(*xlim)
    ax_f.set_xlabel("effective sample size")
    ax_f.set_ylabel("nested − Haar loss")
    handles, labels = ax_f.get_legend_handles_labels()
    handles = [Line2D([], [], color=h.lines[0].get_color(), marker=m,
                      **MARK) for h, (_, _, _, m) in zip(handles, pair_styles)]
    handles.append(Line2D([], [], color=COLORS["mute"], ls="--", lw=LW_REF))
    labels.append("exact finite-time risk")

    # lock the grid first so the strip can be placed in E's reserved band
    cv.lock_reserves()
    # F's key sits in the same band above its axes, so the row's two
    # reserved bands read as one register: seed strip on the left, key on
    # the right, and no key competes with the data below.
    box_f = ax_f.get_position()
    h_pt = cv.fig.get_size_inches()[1] * 72.0
    ax_f.legend(handles, labels, loc="lower left", ncol=2, frameon=False,
                bbox_to_anchor=(0.0, 1.0 + STRIP_GAP_PT / (box_f.height * h_pt)),
                fontsize=PT_LEGEND, handlelength=1.8, handletextpad=0.5,
                labelspacing=0.3, columnspacing=1.2, borderaxespad=0.0)
    strip, medians = crossover_strip(cv.fig, ax_e, crossovers, xlim)
    cv.bind_satellite(strip, ax_e)
    for key, median in medians.items():
        ax_e.axvline(median, color=STYLES[key][1], ls=":", lw=LW_HAIR, zorder=1)

    style_direct_color_labels(cv.fig)
    FIGURES.mkdir(parents=True, exist_ok=True)
    problems = cv.save(FIGURES / f"{STEM}.pdf", name=STEM)
    if problems:
        print("\n".join(str(p) for p in problems))
    plt.close(cv.fig)


if __name__ == "__main__":
    main()

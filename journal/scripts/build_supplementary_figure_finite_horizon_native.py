#!/usr/bin/env python3
"""Supplementary sheet S35 (ident ``finite_horizon``) -- finite-horizon
prediction improves selection but strong simple baselines remain -- rebuilt as
ONE native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/finite_horizon.pdf``) is a
paste of the upstream render ``figure_S38_panels_A-D.pdf`` + ``figure_S39_
panels_A-F.pdf`` whose generator is the legacy port
``scripts/build_morphology_followup_figures.py``; this builder reads ONLY the
frozen tables under ``source_data/morphology_finite_horizon`` and redraws the
same six panels with the same plotted quantities.  Nothing about the numbers
changes; every printed or plotted value is asserted against the table it
comes from, and every mean and 95 % interval is recomputed from the raw
per-seed rows with the study's own bootstrap (``investigate.py:bootstrap``,
10,000 whole-seed draws, ``default_rng(991000)``) before it is drawn.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
frozen_S35.json), panel by panel:

* A, B are eight-row forests on a SYMLOG regret axis (linear from 0 to 0.01,
  logarithmic above, regret x 1,000), so the five rows that collapsed into
  half a pixel on the old linear axis now spread over four decades, and the
  factor-two ordering among the strong selectors is readable.  Every row
  draws its twenty whole-seed means as a fan behind the mean and the 95 %
  seed-bootstrap whisker; the zero-valued seeds pile up on the dashed zero
  rule, which starts clear of the spine.  The two numerically identical
  feedback-arm rows (Gaussian full-batch, context count) are kept as two
  rows -- they are two methods -- and the coincidence is printed on the
  context-count row of A (same candidate in 320/320 tasks, verified).
* C is on the same symlog regret scale, tight to the data (the largest
  seed mean sets the top), with every seed mean drawn as a fan, the two
  Gaussian variants on different marker shapes and the four-entry key set
  ABOVE the axes, in the title band, so no data head-room is spent on it.
* D is sorted by measured seconds, draws median AND interquartile range of
  the raw timing draws (320-1,280 per method, recomputed and asserted
  against ``comparative_computational_cost.csv``), reuses the A/B row
  vocabulary and glyph key, and draws the all-candidate 256-update training
  cost as the labelled reference rule + IQR band it is, not as a row.
* E, F share one hexagonal grid (identical extent, identical (nx, ny) cell
  counts), identical limits, one box size and EQUAL data scale, so the
  identity line is at 45 degrees and the hexagons are the same size in both;
  the y axis starts one half-hexagon below zero (observed half-MSE is
  non-negative) and the shared logarithmic colour bar is printed beside F
  with ticks 1, 10, 100, 1,000 fits per hexagon.  Each panel states its
  n (6,400 fits) and its RMSE (asserted against ``prediction_summary.csv``).
* One palette register for the methods across the sheet, none of it the
  arm colours of the neighbouring S34 (feedback green / joint blue): the
  three original baselines (S34's moment selector, rank-only and fixed /
  max-budget rows on the independent fresh seeds) are the one grey control
  series with three marker shapes; 16-step pilot amber; Gaussian full-batch
  rose; Gaussian SGD (the primary method) the dark red ``bp``; context count
  violet; empirical full-batch (D only) salmon; the privileged population
  oracle is ink, HOLLOW, dashed wherever a line is drawn.  Hexagon fills are
  the journal sequential ramp.  No colour carries a second meaning.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    LW_DATA,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_BASE,
    PT_EMPH,
    SEED_ALPHA,
    SEED_MS,
    SEQ_CMAP,
    Margins,
    NativeCanvas,
    _text_width_pt,
    tint_patch,
    tint_pct,
)

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "morphology_finite_horizon"
SUMMARY = SOURCE / "summaries" / "fresh"
RUNS = SOURCE / "runs" / "fresh"
OUT = ROOT / "figures" / "supplementary" / "figure_finite_horizon_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
DASHED = (0, (2.6, 1.8))
SCALE = 1000.0                 # regret is drawn x 1,000
LINTHRESH = 0.01               # symlog: linear below 0.01 (x 1,000), log above
LINSCALE = 0.6
ARMS = ("feedback_only", "joint_forward_feedback")
ARM_TITLE = {"feedback_only": "Feedback only", "joint_forward_feedback": "Joint transfer"}
CHECKPOINT = 256
N_SEEDS = 20
N_TASKS = 320
FITS_PER_PANEL = 6400

# methods: table key -> (row label, colour, marker, hollow)
METHODS = {
    "original_scalar": ("original\nscalar", COLORS["point_mlp"], "o", False),
    "development_best": ("fixed /\nmax-budget", COLORS["point_mlp"], "s", False),
    "rank_only": ("rank-only", COLORS["point_mlp"], "^", False),
    "pilot16": ("16-step pilot", COLORS["local"], "D", False),
    "gaussian_plugin_fullbatch": ("Gaussian\nfull-batch", COLORS["highlight"], "s", False),
    "gaussian_plugin_sgd": ("Gaussian SGD", COLORS["bp"], "o", False),
    "observed_count_cheapest": ("context count", COLORS["oracle"], "D", False),
    "gaussian_oracle_sgd": ("population\noracle", INK, "o", True),
    "empirical_split_fullbatch": ("empirical\nfull-batch", COLORS["per_soma"], "v", False),
}
ROWS_AB = ("original_scalar", "development_best", "rank_only", "pilot16",
           "gaussian_plugin_fullbatch", "gaussian_plugin_sgd",
           "observed_count_cheapest", "gaussian_oracle_sgd")
ROWS_C = ("gaussian_plugin_sgd", "gaussian_plugin_fullbatch",
          "observed_count_cheapest", "gaussian_oracle_sgd")
ROWS_D = ("observed_count_cheapest", "gaussian_plugin_fullbatch",
          "empirical_split_fullbatch", "pilot16", "original_scalar",
          "gaussian_plugin_sgd")
REFERENCE_D = "candidate_training"
RANKS = (1, 2, 4, 8)
FORECASTS_EF = (("E", "original_scalar", "Feedback only: original scalar forecast"),
                ("F", "gaussian_plugin_sgd", "Feedback only: Gaussian SGD forecast"))


def bootstrap(values, draws=10000):
    """``investigate.py:bootstrap`` verbatim: the study's whole-seed bootstrap."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(991000)
    sample = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    return (float(values.mean()), float(np.quantile(sample, .025)),
            float(np.quantile(sample, .975)))


def csv(name, base=SUMMARY):
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def regret_scale(ax, axis):
    """The one regret scale of the sheet: x 1,000, linear to 0.01, log above."""
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    setter("symlog", linthresh=LINTHRESH, linscale=LINSCALE, base=10)
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_minor_locator(NullLocator())


def seed_means(policies, arm, method, rank=None):
    """Twenty whole-seed mean regrets, in table (natural) units."""
    z = policies[policies.arm.eq(arm) & policies.method.eq(method)]
    if rank is not None:
        z = z[z["rank"].eq(rank)]
    assert z.seed.nunique() == N_SEEDS and len(z) == (N_TASKS if rank is None else N_TASKS // 4)
    assert (z.regret >= 0).all()
    return z.groupby("seed").regret.mean().sort_index().to_numpy(float)


def summary_row(summary, arm, method, rank):
    r = summary[summary.arm.eq(arm) & summary.method.eq(method)
                & summary["rank"].astype(str).eq(str(rank))]
    assert len(r) == 1, (arm, method, rank)
    r = r.iloc[0]
    assert int(r.n_seed_blocks) == N_SEEDS
    return r


def mean_ci(policies, summary, arm, method, rank=None):
    """Recompute mean and 95 % interval from the per-seed rows; assert the table."""
    seeds = seed_means(policies, arm, method, rank)
    m, lo, hi = bootstrap(seeds)
    r = summary_row(summary, arm, method, "all" if rank is None else rank)
    np.testing.assert_allclose([m, lo, hi], [r.mean_regret, r.ci95_low, r.ci95_high],
                               rtol=0, atol=1e-12)
    assert int(r.n_tasks) == (N_TASKS if rank is None else N_TASKS // 4)
    return seeds, m, lo, hi


def fan_h(ax, y, values, color, *, half=0.16, zorder=2.0):
    """Per-seed values as a jittered fan along a horizontal row."""
    values = np.asarray(values, float)
    jitter = np.linspace(-half, half, len(values)) if len(values) > 1 else np.zeros(1)
    ax.plot(values, y + jitter, linestyle="none", marker="o", markersize=SEED_MS,
            markerfacecolor=color, markeredgecolor="none", alpha=SEED_ALPHA,
            zorder=zorder, clip_on=True)


def mean_marker(ax, x, y, color, marker, hollow, *, zorder=4.0, ms=MARKER_MS):
    ax.plot([x], [y], linestyle="none", marker=marker, markersize=ms,
            markerfacecolor=("white" if hollow else color),
            markeredgecolor=(color if hollow else "white"),
            markeredgewidth=(LW_ERR if hollow else LW_HAIR), zorder=zorder)


def whisker_h(ax, y, lo, hi, color, *, zorder=3.0):
    ax.plot([lo, hi], [y, y], color=color, lw=LW_ERR, zorder=zorder, solid_capstyle="butt")
    for xb in (lo, hi):
        ax.plot([xb, xb], [y - 0.13, y + 0.13], color=color, lw=LW_ERR, zorder=zorder,
                solid_capstyle="butt")


def row_label(ax, y, key, *, glyph_dx=-5.5, text_dx=-11.0):
    """Row label in the left gutter, ink, with the series glyph beside it."""
    from matplotlib import transforms
    label, color, marker, hollow = METHODS[key]
    glyph_tr = transforms.offset_copy(ax.get_yaxis_transform(), fig=ax.figure,
                                      x=glyph_dx, y=0.0, units="points")
    ax.plot([0.0], [y], transform=glyph_tr, linestyle="none", marker=marker,
            markersize=MARKER_MS * 0.8,
            markerfacecolor=("white" if hollow else color),
            markeredgecolor=(color if hollow else "white"),
            markeredgewidth=(LW_ERR if hollow else LW_HAIR), clip_on=False, zorder=5.0)
    return ax.annotate(label, xy=(0.0, y), xycoords=("axes fraction", "data"),
                       xytext=(text_dx, 0.0), textcoords="offset points",
                       ha="right", va="center", fontsize=PT_BASE, color=INK,
                       linespacing=0.95, annotation_clip=False)


def tag(ax, text, *, ha="right", x=1.0, dy=2.0):
    return ax.annotate(text, xy=(x, 1.0), xycoords="axes fraction", xytext=(0.0, dy),
                       textcoords="offset points", ha=ha, va="bottom", fontsize=PT_BASE,
                       color=MUTE, annotation_clip=False)


def row_bands(ax, keys, x0, x1):
    """6 % tint band per row, as a fill (a patch would read as a bar to the
    live text-over-data audit, which cannot tell a band from a datum)."""
    for i, key in enumerate(keys):
        ax.fill_between([x0, x1], i - 0.42, i + 0.42, facecolor=tint_pct(METHODS[key][1], 6),
                        edgecolor="none", linewidth=0.0, zorder=0.2, clip_on=True)


# ── A, B: final selection, one arm per panel ─────────────────────────────
def panel_regret(ax, policies, summary, arm, *, note_row=None, note=None):
    labels = []
    printed = []
    x0, x1 = -0.004, 200.0
    ax.set_xlim(x0, x1)
    ax.set_ylim(len(ROWS_AB) - 0.4, -0.6)
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    row_bands(ax, ROWS_AB, x0, x1)
    ax.axvline(0.0, color=MUTE, lw=LW_REF, zorder=1.0, dashes=(2.6, 2.0))
    for y, key in enumerate(ROWS_AB):
        label, color, marker, hollow = METHODS[key]
        seeds, m, lo, hi = mean_ci(policies, summary, arm, key)
        fan_h(ax, y, seeds * SCALE, color)
        whisker_h(ax, y, lo * SCALE, hi * SCALE, color)
        mean_marker(ax, m * SCALE, y, color, marker, hollow)
        labels.append(row_label(ax, y, key))
        zeros = int((seeds == 0.0).sum())
        printed.append(f"{key} {m:.6f} [{lo:.6f}, {hi:.6f}] seeds {seeds.min():.6f}-"
                       f"{seeds.max():.6f} zeros {zeros}/{N_SEEDS}")
    regret_scale(ax, "x")
    ax.set_xlim(x0, x1)
    ax.set_xticks([0.0, 0.01, 0.1, 1.0, 10.0, 100.0], ["0", "0.01", "0.1", "1", "10", "100"])
    ax.set_xlabel("test loss + cost regret (×1,000)")
    ax.tick_params(axis="y", length=0)
    tag(ax, f"n = {N_SEEDS} seed blocks; mean [95 % CI]", dy=11.0)
    ax.annotate("zero regret", xy=(0.0, 1.0), xycoords=("data", "axes fraction"),
                xytext=(2.5, 2.0), textcoords="offset points", ha="left", va="bottom",
                fontsize=PT_BASE, color=MUTE, annotation_clip=False)
    if note_row is not None:
        y = ROWS_AB.index(note_row)
        art = ax.annotate(note, xy=(1.0, y), xycoords=("axes fraction", "data"),
                          xytext=(-1.0, 0.0), textcoords="offset points", ha="right",
                          va="center", fontsize=PT_BASE, color=MUTE, linespacing=1.05)
        art._finite_horizon_note = note_row
    print(f"[{ARM_TITLE[arm]}] " + "; ".join(printed))
    return labels


def assert_identical_choices(policies, arm, a, b):
    """Two methods that chose the same candidate in every task of an arm."""
    za = policies[policies.arm.eq(arm) & policies.method.eq(a)].set_index("task_id")
    zb = policies[policies.arm.eq(arm) & policies.method.eq(b)].set_index("task_id")
    assert len(za) == len(zb) == N_TASKS
    same = int((za.candidate_id == zb.candidate_id.reindex(za.index)).sum())
    return same


# ── C: strong selectors by rank, joint arm ───────────────────────────────
def panel_by_rank(ax, policies, summary):
    arm = "joint_forward_feedback"
    xs = np.arange(len(RANKS), dtype=float)
    dodge = np.linspace(-0.27, 0.27, len(ROWS_C))
    top = 0.0
    handles = []
    printed = []
    for dx, key in zip(dodge, ROWS_C):
        label, color, marker, hollow = METHODS[key]
        ms_, los, his = [], [], []
        for i, rank in enumerate(RANKS):
            seeds, m, lo, hi = mean_ci(policies, summary, arm, key, rank)
            top = max(top, float(seeds.max()))
            jitter = np.linspace(-0.07, 0.07, len(seeds))
            ax.plot(xs[i] + dx + jitter, seeds * SCALE, linestyle="none", marker="o",
                    markersize=SEED_MS * 0.85, markerfacecolor=color, markeredgecolor="none",
                    alpha=SEED_ALPHA, zorder=2.0)
            ms_.append(m); los.append(lo); his.append(hi)
            printed.append(f"{key} r{rank} {m:.7f} [{lo:.7f}, {hi:.7f}] zeros "
                           f"{int((seeds == 0).sum())}/{N_SEEDS}")
        ms_ = np.array(ms_) * SCALE
        los = np.array(los) * SCALE
        his = np.array(his) * SCALE
        ls = DASHED if hollow else "-"
        ax.plot(xs + dx, ms_, color=color, lw=LW_DATA, ls=ls, zorder=3.0)
        for i in range(len(RANKS)):
            ax.plot([xs[i] + dx, xs[i] + dx], [los[i], his[i]], color=color, lw=LW_ERR,
                    zorder=3.4, solid_capstyle="butt")
            mean_marker(ax, xs[i] + dx, ms_[i], color, marker, hollow, zorder=4.0,
                        ms=MARKER_MS * 0.82)
        handles.append(Line2D([], [], color=color, lw=LW_DATA, ls=ls, marker=marker,
                              ms=MARKER_MS * 0.82,
                              markerfacecolor=("white" if hollow else color),
                              markeredgecolor=(color if hollow else "white"),
                              markeredgewidth=(LW_ERR if hollow else LW_HAIR),
                              label=label.replace("\n", " ")))
    print("[C] " + "; ".join(printed))
    regret_scale(ax, "y")
    ax.set_ylim(-0.004, top * SCALE * 1.5)
    ax.set_yticks([0.0, 0.01, 0.1, 1.0], ["0", "0.01", "0.1", "1"])
    ax.set_xlim(-0.55, len(RANKS) - 0.45)
    ax.set_xticks(xs, [str(r) for r in RANKS])
    ax.set_xlabel("generating rank")
    ax.set_ylabel("test loss + cost\nregret (×1,000)")
    ax.axhline(0.0, color=MUTE, lw=LW_REF, zorder=1.0, dashes=(2.6, 2.0))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2,
              frameon=False, fontsize=PT_BASE, handlelength=2.4, columnspacing=1.2,
              handletextpad=0.5, borderaxespad=0.15, borderpad=0.0, labelspacing=0.25)
    return top


# ── D: measured single-CPU cost ──────────────────────────────────────────
def timing_draws(timings, observed, key):
    """Raw per-decision timing draws of one method, as the study defines them."""
    if key == "observed_count_cheapest":
        assert (observed.method == key).all()
        return observed.seconds.to_numpy(float)
    if key == "pilot16":
        g = timings[timings.method.eq("candidate_training")]
        return (g.pilot_update_seconds + g.pilot_evaluation_seconds).to_numpy(float)
    g = timings[timings.method.eq(key)]
    s = g.seconds.copy()
    if key in ("gaussian_plugin_sgd", "gaussian_plugin_fullbatch"):
        s = s + g.calibration_fitting_shared_seconds
    return s.to_numpy(float)


def panel_cost(ax, timings, observed, cost):
    cost = cost.set_index("method")
    rows = []
    for key in ROWS_D + (REFERENCE_D,):
        d = timing_draws(timings, observed, key)
        r = cost.loc[key]
        assert len(d) == int(r.n_measurements) and np.isfinite(d).all() and (d > 0).all()
        med = float(np.median(d))
        np.testing.assert_allclose(med, r.median_seconds_per_20_candidates, rtol=0, atol=1e-12)
        np.testing.assert_allclose(d.mean(), r.mean_seconds_per_20_candidates, rtol=0, atol=1e-12)
        q1, q3 = (float(v) for v in np.quantile(d, [0.25, 0.75]))
        rows.append((key, med, q1, q3, len(d)))
    ref = rows.pop()
    assert ref[0] == REFERENCE_D
    order = sorted(rows, key=lambda t: t[1])
    assert [t[0] for t in order] == list(ROWS_D), [t[0] for t in order]
    assert all(t[1] < ref[2] for t in order), "every method is cheaper than full training"
    ns = sorted({t[4] for t in rows} | {ref[4]})
    print("[D] " + "; ".join(f"{k} median {m:.5f} IQR {a:.5f}-{b:.5f} n={n}"
                             for k, m, a, b, n in order + [ref]))
    x0, x1 = 0.0011, 0.25
    ax.set_xscale("log")
    ax.set_xlim(x0, x1)
    ax.set_ylim(len(order) - 0.4, -0.6)
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    row_bands(ax, [t[0] for t in order], x0, x1)
    # the reference cost: training every candidate for the full 256 updates
    tint_patch(ax, ("rect", ref[2], -0.6, ref[3] - ref[2], len(order) + 0.2), color=MUTE,
               pct=16, edge=False, radius_pt=0.0, zorder=0.3, clip_on=True)
    ax.axvline(ref[1], color=MUTE, lw=LW_REF, zorder=1.0, dashes=(2.6, 2.0))
    ax.annotate("all 256-update fits (reference)", xy=(ref[1], 1.0),
                xycoords=("data", "axes fraction"), xytext=(-2.5, 2.0),
                textcoords="offset points", ha="right", va="bottom", fontsize=PT_BASE,
                color=MUTE, annotation_clip=False)
    labels = []
    for y, (key, med, q1, q3, n) in enumerate(order):
        label, color, marker, hollow = METHODS[key]
        whisker_h(ax, y, q1, q3, color)
        mean_marker(ax, med, y, color, marker, hollow)
        labels.append(row_label(ax, y, key))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks([0.001, 0.01, 0.1], ["0.001", "0.01", "0.1"])
    ax.set_xlabel("seconds per 20-candidate decision")
    ax.tick_params(axis="y", length=0)
    tag(ax, f"median [IQR] of {ns[0]:,}–{ns[-1]:,} timing draws per row", dy=11.0)
    return labels


# ── E, F: endpoint forecasts against outcomes, feedback-only arm ─────────
def load_forecasts():
    key = ["task_id", "arm", "candidate_id", "checkpoint"]
    outcomes = pd.concat([pd.read_csv(p) for p in sorted(RUNS.glob("seed_*_outcomes.csv"))],
                         ignore_index=True)
    predictions = pd.concat([pd.read_csv(p) for p in sorted(RUNS.glob("seed_*_predictions.csv"))],
                            ignore_index=True)
    assert outcomes.seed.nunique() == N_SEEDS and len(outcomes) == 76800
    actual = outcomes[outcomes.arm.eq("feedback_only") & outcomes.regime.eq("fixed_cache")
                      & outcomes.checkpoint.eq(CHECKPOINT)]
    assert len(actual) == FITS_PER_PANEL and actual.task_id.nunique() == N_TASKS
    out = {}
    for _, method, _ in FORECASTS_EF:
        z = actual.merge(predictions[predictions.method.eq(method)][key + ["predicted_loss"]],
                         on=key, validate="one_to_one")
        assert len(z) == FITS_PER_PANEL and z.candidate_id.nunique() == 20
        out[method] = z
    return out


def panel_forecast(ax, z, method, prediction_summary, *, extent, gridsize, cmap, norm):
    ps = prediction_summary
    r = ps[ps.arm.eq("feedback_only") & ps.regime.eq("fixed_cache")
           & ps.checkpoint.eq(CHECKPOINT) & ps.method.eq(method)]
    assert len(r) == 1
    r = r.iloc[0]
    rmse = float(np.sqrt(((z.predicted_loss - z.test_loss) ** 2).mean()))
    np.testing.assert_allclose(rmse, r.rmse, rtol=0, atol=1e-9)
    assert (z.test_loss >= 0).all()
    x, y = z.predicted_loss.to_numpy(float), z.test_loss.to_numpy(float)
    assert extent[0] < x.min() and x.max() < extent[1] and extent[2] <= y.min() and y.max() < extent[3]
    hb = ax.hexbin(x, y, gridsize=gridsize, extent=extent, mincnt=1, cmap=cmap, norm=norm,
                   linewidths=0.0, edgecolors="none", zorder=2.0)
    counts = hb.get_array()
    assert int(counts.sum()) == FITS_PER_PANEL and counts.min() >= 1
    print(f"[{method}] n={len(z)} x {x.min():.4f}..{x.max():.4f} y {y.min():.2e}..{y.max():.4f} "
          f"RMSE {rmse:.5f}; {len(counts)} occupied hexagons, max {int(counts.max())} per hexagon")
    ax.annotate(f"RMSE {rmse:.3f}", xy=(0.03, 0.97), xycoords="axes fraction", ha="left",
                va="top", fontsize=PT_BASE, color=INK)
    tag(ax, f"n = {FITS_PER_PANEL:,} candidate fits", ha="left", x=0.0)
    return hb, rmse


def equal_scale_limits(ax, *, xmin, ymin, ymax, x_need):
    """x range = y range x (box w / box h): one data scale on both axes."""
    box = ax.get_position()
    w_pt = box.width * ax.figure.get_size_inches()[0] * 72.0
    h_pt = box.height * ax.figure.get_size_inches()[1] * 72.0
    xr = (ymax - ymin) * w_pt / h_pt
    ax.set_xlim(xmin, xmin + xr)
    ax.set_ylim(ymin, ymax)
    assert xmin + xr >= x_need, (xmin + xr, x_need)
    return w_pt, h_pt, xmin + xr


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 476.0
ROW_PT = [138.0, 112.0, 110.0]
HGUTTER_PT = 30.0
VGUTTER_PT = 36.0
MARGINS = Margins(left=30.0, right=10.0, top=14.0, bottom=30.0)
TITLE_PAD = 11.0
TITLE_PAD_KEYED = 22.0


def build(path: Path = OUT):
    summary = csv("endpoint_summary_with_secondary_baselines.csv")
    policies = csv("endpoint_policies_with_secondary_baselines.csv")
    cost = csv("comparative_computational_cost.csv")
    timings = csv("timings.csv")
    observed = csv("observed_count_timings.csv")
    prediction_summary = csv("prediction_summary.csv")
    summary = summary[summary.regime.eq("fixed_cache")]
    policies = policies[policies.regime.eq("fixed_cache") & policies.checkpoint.eq(CHECKPOINT)]
    assert len(policies) == 7680 and policies.seed.nunique() == N_SEEDS
    same_a = assert_identical_choices(policies, "feedback_only", "gaussian_plugin_fullbatch",
                                      "observed_count_cheapest")
    same_b = assert_identical_choices(policies, "joint_forward_feedback",
                                      "gaussian_plugin_fullbatch", "observed_count_cheapest")
    assert same_a == N_TASKS and same_b == 310, (same_a, same_b)
    print(f"[choices] feedback: full-batch == context count in {same_a}/{N_TASKS} tasks; "
          f"joint: {same_b}/{N_TASKS}")
    forecasts = load_forecasts()

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 3, row_weights=ROW_PT, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS)
    ax_a = cv.panel("A", 0, 0, 6, title="Feedback only: final selection")
    ax_b = cv.panel("B", 0, 6, 6, title="Joint transfer: final selection")
    ax_c = cv.panel("C", 1, 0, 6, grid="y", title="Strong baselines in the joint arm")
    ax_d = cv.panel("D", 1, 6, 6, title="Measured single-CPU cost")
    ax_e = cv.panel("E", 2, 0, 5, title=FORECASTS_EF[0][2])
    ax_f = cv.panel("F", 2, 6, 5, title=FORECASTS_EF[1][2])
    for ax in (ax_e, ax_f):
        ax.set_title(ax.get_title(), fontsize=PT_EMPH, color=INK, pad=TITLE_PAD, fontweight="normal")
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_title(ax.get_title(), fontsize=PT_EMPH, color=INK, pad=TITLE_PAD_KEYED, fontweight="normal")

    labels = panel_regret(ax_a, policies, summary, "feedback_only",
                          note_row="observed_count_cheapest",
                          note=f"= Gaussian full-batch\n({same_a}/{N_TASKS} tasks)")
    labels += panel_regret(ax_b, policies, summary, "joint_forward_feedback")
    panel_by_rank(ax_c, policies, summary)
    labels += panel_cost(ax_d, timings, observed, cost)

    # E, F: one hexagonal grid, one colour scale, one box, one data scale
    ext_x = (-0.075, 0.90)
    ext_y = (0.0, 0.80)
    nx = 21
    # ---- first pass: lock the boxes so the data scale can be equalised
    cv.fig.canvas.draw()
    renderer = cv.fig.canvas.get_renderer()
    gutter = max(_text_width_pt(t, renderer) for t in labels) + 11.0 + 6.0
    for name in "ABCDEF":
        cv.declare_reserve(name, left=gutter)
    for name in "ABCD":                 # one right reserve for both 6-module columns
        cv.declare_reserve(name, right=6.0)
    cv.lock_reserves()
    box_e, box_f = ax_e.get_position(), ax_f.get_position()
    print(f"[lock] gutter {gutter:.1f} pt; E box {box_e.width * cv.width_pt:.1f} x {box_e.height * cv.height_pt:.1f} pt; "
          + "; ".join(f"{n} {cv.axes[n].get_position().width * cv.width_pt:.1f}x{cv.axes[n].get_position().height * cv.height_pt:.1f}" for n in "ABCD"))
    np.testing.assert_allclose([box_e.width, box_e.height], [box_f.width, box_f.height], rtol=0, atol=1e-9)
    w_pt = box_e.width * cv.width_pt
    h_pt = box_e.height * cv.height_pt
    # regular hexagons at one data scale: sy = sqrt(3) * sx in data units
    ny = int(round((ext_y[1] - ext_y[0]) * nx / (np.sqrt(3.0) * (ext_x[1] - ext_x[0]))))
    sy = (ext_y[1] - ext_y[0]) / ny
    ymin = -sy / 3.0 - 0.003            # one half-hexagon below zero, so the y = 0 row is whole
    ymax = 0.80 + sy / 3.0 + 0.003
    cmap = LinearSegmentedColormap.from_list("journal_seq_hex", SEQ_CMAP(np.linspace(0.30, 1.0, 256)))
    hexes = []
    rmses = {}
    for (name, method, _), ax in zip(FORECASTS_EF, (ax_e, ax_f)):
        z = forecasts[method]
        hb, rmse = panel_forecast(ax, z, method, prediction_summary,
                                  extent=(*ext_x, *ext_y), gridsize=(nx, ny), cmap=cmap,
                                  norm=LogNorm(vmin=1.0, vmax=10.0))
        hexes.append(hb)
        rmses[name] = rmse
    vmax = max(float(h.get_array().max()) for h in hexes)
    assert vmax > 100.0
    norm = LogNorm(vmin=1.0, vmax=vmax)
    for h in hexes:
        h.set_norm(norm)
    sx = (ext_x[1] - ext_x[0]) / nx
    x_need = max(float(forecasts[m].predicted_loss.max()) for _, m, _ in FORECASTS_EF) + sx + 0.02
    for ax in (ax_e, ax_f):
        w_pt, h_pt, xmax = equal_scale_limits(ax, xmin=ext_x[0] - 0.012, ymin=ymin, ymax=ymax,
                                              x_need=x_need)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8], ["0", "0.2", "0.4", "0.6", "0.8"])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8], ["0", "0.2", "0.4", "0.6", "0.8"])
        ax.set_xlabel("predicted final half-MSE")
        ax.set_ylabel("observed test half-MSE")
        ax.plot([0.0, 0.80], [0.0, 0.80], color=MUTE, lw=LW_REF, dashes=(2.6, 2.0), zorder=3.0)
        ax.annotate("equality", xy=(0.80, 0.80), xycoords="data", xytext=(2.0, 1.5),
                    textcoords="offset points", ha="left", va="bottom", fontsize=PT_BASE,
                    color=MUTE, annotation_clip=False)
    print(f"[E,F] box {w_pt:.1f} x {h_pt:.1f} pt; x {ax_e.get_xlim()[0]:.3f}..{xmax:.3f}, y {ymin:.3f}..{ymax:.3f}; "
          f"grid {nx} x {ny} (cell {(ext_x[1] - ext_x[0]) / nx:.4f} x {sy:.4f}); colour 1..{int(vmax)}")
    # the shared colour bar, in the empty sixth module right of F
    bar_w = 4.5
    x_bar = (box_f.x0 + box_f.width) * cv.width_pt + 7.0
    cax = cv.fig.add_axes([x_bar / cv.width_pt, box_f.y0 + 0.12 * box_f.height,
                           bar_w / cv.width_pt, 0.76 * box_f.height])
    cv.bind_satellite(cax, ax_f)
    cbar = cv.fig.colorbar(hexes[1], cax=cax, ticks=[1.0, 10.0, 100.0, 1000.0])
    cbar.outline.set_linewidth(LW_HAIR)
    cbar.outline.set_edgecolor(COLORS["edge"])
    cbar.ax.yaxis.set_minor_locator(NullLocator())
    cbar.ax.set_yticklabels(["1", "10", "100", "1,000"])
    cbar.ax.tick_params(labelsize=PT_BASE, width=LW_HAIR, length=2.2, pad=1.5,
                        color=COLORS["edge"], labelcolor=INK)
    cbar.set_label("fits per hexagon (E, F)", fontsize=PT_BASE, labelpad=2.5, color=INK)

    problems = cv.save(path, name="figure_finite_horizon_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

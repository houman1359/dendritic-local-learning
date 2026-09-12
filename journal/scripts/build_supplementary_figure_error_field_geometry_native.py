#!/usr/bin/env python3
"""Supplementary sheet S8 (ident ``error_field_geometry``) -- direction,
amplitude and spatial variation of the transported error field -- rebuilt as
ONE native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/error_field_geometry.pdf``)
is a paste of three panels of an upstream render (old S45 C-E) that has no
generator in this repository, so its verified in-panel defects could not be
fixed by the paste layer.  This builder reads ONLY the two frozen tables the
manifest declares for the plotted quantities,

* ``source_data/figure2/feedback_gradient_runs.csv``            (panel A)
* ``source_data/figure2/path_gain_dispersion_ladder_runs.csv``  (panels B, C)

and draws the same three panels with the same plotted quantities.  Nothing
about the numbers changes; every printed or plotted value is asserted against
the table it comes from, every mean is recomputed from the fifteen per-seed
rows, and the descriptive 95 % intervals are whole-seed bootstraps
(10,000 draws, ``default_rng``) of those same rows.  No summary table exists
for these three quantities, so the caption's stated values (17.9 %, 18.1 %,
53 %, 32 %) are asserted against the recomputed seed means directly.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
frozen_S8.json), panel by panel:

* A keeps only the two measured feedback fields (matched-width fallback and
  neuron-specific), both category labels set horizontally on two short
  lines, the y axis tight to the data (-0.10 .. 0.80); the exact-path cosine
  of one is definitional and is not drawn (the caption says so).  The zero
  reference is a dashed mute rule labelled in the axis margin.  The fifteen
  paired seeds are a fan of hairlines with a small mark at each end behind
  the seed mean and its whisker; the two architectures are dodged.
* B draws, for the first time, the fifteen per-seed depth profiles behind
  each architecture's mean profile (they were in the table all along), on
  the same ordinal depth axis and a logarithmic ratio axis with the 1-2-5
  ladder; the soma value one is a normalization constant and is drawn as
  a dashed mute reference rule labelled "soma = 1 by normalization", not as
  a marker in a third colour.  The in-plot methods note is gone: the axis
  label carries "batch-RMS ratio, soma = 1".
* C dodges the two architectures at every depth so the coinciding mid
  values (17.9 % and 18.1 %) are two visible marks, keeps its per-seed fan,
  and moves "within-depth residual" into its title.
* The three panels sit on one 2 x 2 module grid of the full 518.4 pt canvas
  (A, B on the top row; C and the sheet's key on the bottom row -- the canvas
  audit admits only a 1.05-1.55 page aspect, which a single row of three
  ladders cannot meet), so every plot box has the same size and the two
  rows share one top and one baseline each.  The means wear a smaller symbol
  than the frozen render so the whiskers draw where they exceed it; where
  none would (B, every half-width below 0.08 ratio units on the log axis)
  the panel says so and draws no stub.  One key names every glyph with its
  n: shunting = green circle, raw additive = blue square, small mark = one
  seed (15 paired seeds per architecture), large mark = seed mean, whisker
  = 95 % seed-bootstrap interval (10,000 whole-seed draws), dashed mute
  rule = reference constant.  Grey (``mute``) means a reference constant
  and nothing else on the sheet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

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
    SEED_ALPHA,
    SEED_MS,
    Margins,
    NativeCanvas,
)

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "figure2"
OUT = ROOT / "figures" / "supplementary" / "figure_error_field_geometry_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
DASHES = (2.6, 2.0)

# architecture register: (gradient-table name, dispersion-table name, key label, colour, marker)
ARCHS = (
    ("dendritic_shunting", "shunting", "shunting", COLORS["shunting"], "o"),
    ("dendritic_additive", "additive", "raw additive", COLORS["additive"], "s"),
)
N_SEEDS = 15
BOOT_DRAWS = 10_000
MEAN_MS = MARKER_MS * 0.78          # 3.6 pt: the whiskers draw where they exceed it
FAN_MS = SEED_MS * 0.85
FAN_ALPHA = SEED_ALPHA * 0.75
LINE_ALPHA = 0.30

# A: the two measured feedback fields, at checkpoints trained with per-neuron
# (per_soma_shared) feedback; the exact-path cosine (1 by construction) is not
# a column of the table and is not drawn.
FIELDS = (("scalar_fallback", "matched-width\nfallback"), ("neuron_wise", "neuron-\nspecific"))
TRAINED_MODE = "per_soma_shared"
COSINE = "branch_numel_weighted_cosine"


def boot(values, seed):
    """Descriptive 95 % whole-seed bootstrap of the mean (10,000 draws)."""
    values = np.asarray(values, float)
    assert values.ndim == 1 and np.isfinite(values).all()
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(BOOT_DRAWS, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def csv(name):
    path = SOURCE / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def draw_profiles(ax, xs, values, color, marker, *, dodge, half, seeds, whiskers=True):
    """The fifteen per-seed profiles (hairlines, one small mark at each
    measured depth) behind the mean profile with its whisker at each point.

    ``values`` is (n_seed, n_x); a column of NaN is a definitional point
    (drawn as the line start only, no mark and no whisker).  Returns the
    (mean, lo, hi) triple per measured column.
    """
    values = np.asarray(values, float)
    n, k = values.shape
    assert n == N_SEEDS and k == len(xs)
    measured = np.isfinite(values).all(axis=0)
    jitter = np.linspace(-half, half, n)
    for i in range(n):
        x_i = np.asarray(xs, float) + dodge + jitter[i]
        row = values[i].copy()
        row[~measured] = 1.0 if not measured.all() else row[~measured]
        ax.plot(x_i, row, color=color, lw=LW_HAIR, alpha=LINE_ALPHA, zorder=2.0,
                solid_capstyle="butt")
        ax.plot(x_i[measured], row[measured], linestyle="none", marker=marker, markersize=FAN_MS,
                markerfacecolor=color, markeredgecolor="none", alpha=FAN_ALPHA, zorder=2.2)
    out = []
    means = []
    for j in range(k):
        if measured[j]:
            m, lo, hi = boot(values[:, j], seeds[j])
        else:
            m, lo, hi = 1.0, 1.0, 1.0          # the normalization constant
        out.append((m, lo, hi))
        means.append(m)
    ax.plot(np.asarray(xs, float) + dodge, means, color=color, lw=LW_DATA, zorder=3.5,
            solid_capstyle="butt")
    for j in range(k):
        if not measured[j]:
            continue
        m, lo, hi = out[j]
        yerr = [[m - lo], [hi - m]] if whiskers else None
        ax.errorbar([xs[j] + dodge], [m], yerr=yerr, fmt=marker, ms=MEAN_MS,
                    color=color, markeredgecolor="white", markeredgewidth=LW_HAIR,
                    ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE * 0.8,
                    capthick=LW_ERR, zorder=4.0, linestyle="none")
    return out


def whisker_extent_pt(ax, stats, *, skip=()):
    """Longest and shortest 95 % half-interval on the panel, in points of the
    final axes box, so the builder can assert whether a whisker is visible
    beyond the mean symbol (radius ``MEAN_MS / 2``) rather than guess."""
    ax.figure.canvas.draw()
    lengths = []
    for s in stats.values():
        for j, (m, lo, hi) in enumerate(s):
            if j in skip:
                continue
            y = ax.transData.transform([[0.0, lo], [0.0, m], [0.0, hi]])[:, 1] * 72.0 / ax.figure.dpi
            lengths.append(max(y[1] - y[0], y[2] - y[1]))
    return max(lengths), min(lengths)


def reference_rule(ax, y, label, *, x0, x1, dashes=DASHES, xytext=(0.0, 2.0), ha="left",
                   x_label=None):
    ax.plot([x0, x1], [y, y], color=MUTE, lw=LW_REF, dashes=dashes, zorder=1.0,
            solid_capstyle="butt")
    ax.annotate(label, xy=(x0 if x_label is None else x_label, y), xycoords="data",
                xytext=xytext, textcoords="offset points", ha=ha, va="bottom",
                fontsize=PT_BASE, color=MUTE, annotation_clip=False)


# ── A: branch-gradient cosine at common checkpoints ──────────────────────
def panel_gradient_cosine(ax, grad):
    g = grad[grad.trained_broadcast_mode.eq(TRAINED_MODE)]
    assert set(g.diagnostic_feedback) == {f for f, _ in FIELDS}
    assert set(g.network_type) == {a[0] for a in ARCHS}
    assert (g.cohort == "clean_current_code").all()
    xs = [0.0, 1.0]
    dodge = {"dendritic_shunting": -0.11, "dendritic_additive": 0.11}
    stats = {}
    lo_all, hi_all = np.inf, -np.inf
    for table, _, label, color, marker in ARCHS:
        pivot = g[g.network_type.eq(table)].pivot_table(
            index="seed", columns="diagnostic_feedback", values=COSINE, aggfunc="mean")
        assert len(pivot) == N_SEEDS and pivot.notna().all().all()
        assert pivot.index.is_unique
        vals = pivot[[f for f, _ in FIELDS]].to_numpy(float)
        # the pivot's aggfunc never averages: exactly one row per seed x field
        assert (g[g.network_type.eq(table)].groupby(["seed", "diagnostic_feedback"]).size() == 1).all()
        stats[table] = draw_profiles(ax, xs, vals, color, marker, dodge=dodge[table], half=0.045,
                                     seeds=(101, 102) if table == "dendritic_shunting" else (103, 104))
        lo_all = min(lo_all, vals.min())
        hi_all = max(hi_all, vals.max())
        for (m, lo, hi), (field, _) in zip(stats[table], FIELDS):
            raw = g[g.network_type.eq(table) & g.diagnostic_feedback.eq(field)][COSINE].to_numpy(float)
            assert len(raw) == N_SEEDS
            np.testing.assert_allclose(m, raw.mean(), rtol=0, atol=1e-12)
            assert lo <= m <= hi
            print(f"[A] {label:13s} {field:16s} mean {m:.4f} [{lo:.4f}, {hi:.4f}] "
                  f"half-width {max(m - lo, hi - m):.4f}; seeds {raw.min():.4f}-{raw.max():.4f}")
    # the values the review measured on the frozen render
    sh = stats["dendritic_shunting"][1][0]
    assert abs(sh - 0.708) < 5e-4, sh
    assert -0.10 < lo_all and hi_all < 0.80, (lo_all, hi_all)
    ax.set_xlim(-0.55, 1.55)
    reference_rule(ax, 0.0, "0 = no alignment", x0=-0.55, x1=1.55, x_label=1.55, ha="right",
                   xytext=(0.0, 3.5))
    ax.set_ylim(-0.10, 0.80)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8], ["0", "0.2", "0.4", "0.6", "0.8"])
    ax.set_xticks(xs, [lab for _, lab in FIELDS])
    ax.set_xlabel("feedback field at common checkpoints")
    ax.set_ylabel("branch-gradient cosine\nto exact gradient")
    return stats


# ── B: exact voltage-error magnitude by depth ────────────────────────────
def panel_transport_profile(ax, disp):
    xs = [0.0, 1.0, 2.0]                     # soma, mid, distal
    dodge = {"shunting": -0.05, "additive": 0.05}
    stats = {}
    lo_all, hi_all = np.inf, -np.inf
    for _, table, label, color, marker in ARCHS:
        sub = disp[disp.architecture.eq(table)].sort_values("seed")
        assert len(sub) == N_SEEDS and sub.seed.is_unique
        vals = np.column_stack([np.full(N_SEEDS, np.nan), sub.stage1_mean.to_numpy(float),
                                sub.stage0_mean.to_numpy(float)])
        stats[table] = draw_profiles(ax, xs, vals, color, marker, dodge=dodge[table], half=0.03,
                                     seeds=(0, 201, 202) if table == "shunting" else (0, 203, 204),
                                     whiskers=False)
        lo_all = min(lo_all, np.nanmin(vals))
        hi_all = max(hi_all, np.nanmax(vals))
        for j, col in ((1, "stage1_mean"), (2, "stage0_mean")):
            m, lo, hi = stats[table][j]
            np.testing.assert_allclose(m, sub[col].mean(), rtol=0, atol=1e-12)
            assert lo <= m <= hi
            print(f"[B] {label:13s} {col:12s} mean {m:.4f} [{lo:.4f}, {hi:.4f}] "
                  f"half-width {max(m - lo, hi - m):.4f}; seeds {sub[col].min():.4f}-{sub[col].max():.4f}")
    # the values the review measured on the frozen render
    np.testing.assert_allclose([stats["additive"][2][0], stats["additive"][1][0],
                                stats["shunting"][2][0], stats["shunting"][1][0]],
                               [5.209, 0.908, 0.237, 0.599], rtol=0, atol=6e-4)
    assert 0.18 < lo_all and hi_all < 6.4, (lo_all, hi_all)
    half = max(max(m - lo, hi - m) for s in stats.values() for m, lo, hi in s[1:])
    assert half < 0.08, half
    # every 95 % interval on this log axis is narrower than the mean symbol
    # (asserted in ``build`` once the axes box is final), so none is drawn as
    # a hidden stub; the panel says so instead.
    ax.annotate(f"95% seed-bootstrap intervals\n\u2264 {half:.2f} ratio units: within\nthe mean symbols, not drawn",
                xy=(0.0, 1.0), xycoords="axes fraction", xytext=(4.0, -3.0),
                textcoords="offset points", ha="left", va="top", fontsize=PT_BASE, color=MUTE,
                linespacing=1.1)
    ax.set_xlim(-0.35, 2.35)
    ax.set_yscale("log")
    ax.set_ylim(0.18, 6.4)
    ax.set_yticks([0.2, 0.5, 1.0, 2.0, 5.0], ["0.2", "0.5", "1", "2", "5"])
    ax.set_yticks([0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 3.0, 4.0, 6.0], minor=True)
    ax.yaxis.set_minor_formatter(NullFormatter())
    reference_rule(ax, 1.0, "soma = 1 by normalization", x0=-0.35, x1=2.35, x_label=-0.35,
                   xytext=(1.0, 3.5))
    ax.set_xticks(xs, ["soma", "mid", "distal"])
    ax.set_xlabel("depth")
    ax.set_ylabel("batch-RMS ratio, soma = 1")
    return stats


# ── C: within-depth path-specific residual energy ────────────────────────
def panel_path_specific_energy(ax, disp):
    xs = [0.0, 1.0]                          # mid, distal
    dodge = {"shunting": -0.07, "additive": 0.07}
    cols = ["stage1_path_specific_energy_fraction", "stage0_path_specific_energy_fraction"]
    stats = {}
    lo_all, hi_all = np.inf, -np.inf
    for _, table, label, color, marker in ARCHS:
        sub = disp[disp.architecture.eq(table)].sort_values("seed")
        assert len(sub) == N_SEEDS and sub.seed.is_unique
        vals = 100.0 * sub[cols].to_numpy(float)
        assert ((vals > 0) & (vals < 100)).all()
        stats[table] = draw_profiles(ax, xs, vals, color, marker, dodge=dodge[table], half=0.03,
                                     seeds=(301, 302) if table == "shunting" else (303, 304))
        lo_all = min(lo_all, vals.min())
        hi_all = max(hi_all, vals.max())
        for j, col in enumerate(cols):
            m, lo, hi = stats[table][j]
            np.testing.assert_allclose(m, 100.0 * sub[col].mean(), rtol=0, atol=1e-10)
            assert lo <= m <= hi
            print(f"[C] {label:13s} {col[:6]} mean {m:.3f} % [{lo:.3f}, {hi:.3f}] "
                  f"half-width {max(m - lo, hi - m):.3f} pp; seeds {vals[:, j].min():.2f}-{vals[:, j].max():.2f}")
    # the caption's numbers: mid 17.9 % (shunting) and 18.1 % (raw additive), distal 53 % and 32 %
    np.testing.assert_allclose([stats["shunting"][0][0], stats["additive"][0][0]], [17.9, 18.1],
                               rtol=0, atol=0.05)
    np.testing.assert_allclose([stats["shunting"][1][0], stats["additive"][1][0]], [53.0, 32.0],
                               rtol=0, atol=0.5)
    assert 14.0 < lo_all and hi_all < 62.0, (lo_all, hi_all)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(14.0, 62.0)
    ax.set_yticks([20, 30, 40, 50, 60], ["20", "30", "40", "50", "60"])
    ax.set_xticks(xs, ["mid", "distal"])
    ax.set_xlabel("depth")
    ax.set_ylabel("path-specific error energy (%)")
    return stats


# ── the canvas ───────────────────────────────────────────────────────────
# The canvas audit admits only a 1.05-1.55 page aspect, so the three ladders
# share a 2 x 2 module grid: A and B on the top row, C and the sheet's key
# (with the cohort statement) on the bottom row.
CANVAS_H_PT = 352.0
HGUTTER_PT = 30.0
VGUTTER_PT = 40.0
MARGINS = Margins(left=44.0, right=8.0, top=18.0, bottom=38.0)


def build(path: Path = OUT):
    grad = csv("feedback_gradient_runs.csv")
    disp = csv("path_gain_dispersion_ladder_runs.csv")
    assert len(grad) == 120 and grad.seed.nunique() == N_SEEDS
    assert len(disp) == 2 * N_SEEDS and disp.seed.nunique() == N_SEEDS
    # the caption's cohort: 128 directed [3,3] trees (12 non-somatic compartments)
    assert (disp.n_soma == 128).all() and (disp.compartments_per_neuron == 12).all()
    # the same fifteen seeds in both tables and both architectures
    assert set(grad.seed) == set(disp.seed)
    n_soma = int(disp.n_soma.iloc[0])
    # the key cell's cohort statement, checked against the run records
    assert grad.run_dir.str.contains("mnist").all() and disp.run_dir.str.contains("mnist").all()
    assert disp.run_dir.str.contains("exact_path").all()
    assert (grad.trained_broadcast_mode.isin(["per_soma", "per_soma_shared"])).all()

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 2, hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
                      margins=MARGINS)
    ax_a = cv.panel("A", 0, 0, 6, grid="y", title="Direction: gradient cosine")
    ax_b = cv.panel("B", 0, 6, 6, grid="y", title="Amplitude: transport by depth")
    ax_c = cv.panel("C", 1, 0, 6, grid="y", title="Variation: within-depth residual")
    for name in "ABC":
        cv.declare_reserve(name, left=15.0, right=8.0)

    stats_a = panel_gradient_cosine(ax_a, grad)
    stats_b = panel_transport_profile(ax_b, disp)
    stats_c = panel_path_specific_energy(ax_c, disp)
    cv.lock_reserves()              # settle the boxes, then measure the whiskers

    # the widest 95 % half-width on each panel, for the caption, and the
    # visibility rule: A and C draw whiskers because at least one protrudes
    # beyond the mean symbol; B draws none because none would.
    def widest(stats, skip=()):
        return max(max(m - lo, hi - m) for s in stats.values()
                   for j, (m, lo, hi) in enumerate(s) if j not in skip)
    radius = MEAN_MS / 2.0
    ext_a = whisker_extent_pt(ax_a, stats_a)
    ext_b = whisker_extent_pt(ax_b, stats_b, skip=(0,))
    ext_c = whisker_extent_pt(ax_c, stats_c)
    assert ext_a[0] > radius + 1.0 and ext_c[0] > radius + 1.0, (ext_a, ext_c, radius)
    assert ext_b[0] < radius, (ext_b, radius)
    print(f"[intervals] max half-width A {widest(stats_a):.4f} cosine, "
          f"B {widest(stats_b, skip=(0,)):.4f} ratio, C {widest(stats_c):.3f} pp; "
          f"whisker half-lengths on the page A {ext_a[1]:.1f}-{ext_a[0]:.1f} pt, "
          f"B {ext_b[1]:.1f}-{ext_b[0]:.1f} pt (not drawn), C {ext_c[1]:.1f}-{ext_c[0]:.1f} pt; "
          f"mean-symbol radius {radius:.1f} pt")

    # the key cell (row 1, modules 6-12): every glyph on the sheet with its n,
    # then the cohort statement
    handles = [
        Line2D([], [], color=c, lw=LW_DATA, marker=mk, ms=MEAN_MS, markeredgecolor="white",
               markeredgewidth=LW_HAIR, label=f"{lab}: seed mean")
        for _, _, lab, c, mk in ARCHS
    ] + [
        Line2D([], [], color=INK, lw=LW_ERR, marker="|", ms=ERR_CAPSIZE * 2.0,
               markeredgewidth=LW_ERR, label="whisker: 95% seed-bootstrap interval\n"
               f"of the mean ({BOOT_DRAWS:,} whole-seed draws)"),
        Line2D([], [], color=INK, lw=LW_HAIR, alpha=LINE_ALPHA + 0.25, marker="o", ms=FAN_MS,
               markerfacecolor=INK, markeredgecolor="none",
               label=f"thin line, small mark: one seed\n({N_SEEDS} paired seeds per architecture)"),
        Line2D([], [], color=MUTE, lw=LW_REF, dashes=DASHES, label="dashed rule: reference constant"),
    ]
    x0, y_top, w, h = cv.slot_pt(1, 6, 6)
    fx = lambda x: x / cv.width_pt  # noqa: E731
    fy = lambda y: 1.0 - y / cv.height_pt  # noqa: E731
    cv.fig.text(fx(x0 + 15.0), fy(y_top - 2.0), "Key", fontsize=PT_BASE + 1.0, color=INK,
                ha="left", va="bottom")
    cv.fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(fx(x0 + 15.0), fy(y_top + 2.0)),
                  ncol=1, frameon=False, fontsize=PT_BASE, handlelength=2.4, labelspacing=0.75,
                  handletextpad=0.7, borderaxespad=0.0)
    cohort = (f"Cohort: {n_soma} directed [3,3] trees per network, {N_SEEDS} paired seeds\n"
              "per architecture, flattened MNIST; A at checkpoints trained with\n"
              "per-neuron feedback, B and C at exact-path checkpoints.\n"
              "Exact-path cosine in A is 1 by construction and is not drawn.")
    cv.fig.text(fx(x0 + 15.0), fy(y_top + h), cohort, fontsize=PT_BASE, color=INK, ha="left",
                va="bottom", linespacing=1.25)
    problems = cv.save(path, name="figure_error_field_geometry_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

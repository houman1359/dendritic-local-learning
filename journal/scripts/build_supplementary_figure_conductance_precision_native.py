#!/usr/bin/env python3
"""Supplementary sheet S19 (ident ``conductance_precision``) -- the first
conductance task's small precision benefit from exact credit -- rebuilt as
ONE native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/conductance_precision.pdf``)
is a paste of an upstream render (``figure_S50_conductance_small_effect``)
with no generator in this repository, so its verified in-panel defects could
not be fixed by the paste layer.  This builder reads ONLY the frozen tables
under ``source_data/conductance_credit_demand/`` and redraws the same four
panels with the same plotted quantities.  Nothing about the numbers changes;
every printed or plotted value is asserted against the table it comes from,
and every mean is recomputed from the per-seed rows before it is drawn.

Tables read
-----------
``figures/supplement_first_conductance_source.csv``
    the sheet's declared numerical source: per-checkpoint means and 95 %
    paired-seed bootstrap intervals for A/B, the two C contrasts, the four
    D marks;
``summaries/all_curves.csv``
    per-seed test NMSE at checkpoints 0-4,096 (phase ``fresh``, Adam at the
    selected rate);
``extension/seed_<s>_curves.csv``
    per-seed test NMSE at checkpoints 4,096-16,384 (phase ``extension``);
``summaries/paired_seed_contrasts.csv`` and ``extension/seed_<s>_endpoints.csv``
    the 20 paired seed differences behind each C mark, and the 20 exact-rule
    best-validation states behind each D mark.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
frozen_S19.json, the "cannot" list of frozen_drafts.json), panel by panel:

* A, B  updates on a LOGARITHMIC axis from 64 with update 0 as a pip outside
        a broken axis (ticks 0, 64, 256, 1,024, 4,096, 16,384; minor ticks at
        2,048, 8,192, 12,288; a marker at every one of the nine checkpoints);
        the y axis is the base-ten logarithm of test NMSE on a linear scale
        that holds every value drawn, initialization included; the 4,096
        budget rule is dotted and named; all 20 per-seed trajectories of
        every rule are drawn as a hairline fan behind the mean; the two
        panels share one y axis (ticks and label on A only) and, by the
        column lock, one axes width.
* C     the Adam and SGD contrasts are two facets with INDEPENDENT y limits,
        each with its own zero rule, so both seed clouds resolve; the seed
        points are the rule colour at 60 % over white, above the spine in
        contrast; the 20/20 positive count is printed under each facet; the
        in-box caption prose is gone.
* D     the 20 per-seed capture values are a fan behind each mark; the y
        axis is tight to the data (0.866-1.021); the ungated rank-one mark
        is labelled "= 1 by construction"; the note that three of the four
        intervals are narrower than the marker is printed on the panel.
* One palette register for the whole sheet: exact-path credit and the path
  field it produces are dark red (``bp``); unit broadcast is amber
  (``local``); the initial profile -- the fixed calibrated broadcast --
  and every quantity of it (its curve in A/B, its gap against exact in C,
  its eligibility-weighted capture in D) is blue (``additive``).  Line
  style and marker repeat the rule identity (solid circle, dashed square,
  dotted triangle).  No colour carries a second meaning anywhere.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
    LW_EDGE,
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
SOURCE = ROOT / "source_data" / "conductance_credit_demand"
OUT = ROOT / "figures" / "supplementary" / "figure_conductance_precision_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]
DASHED = (2.6, 1.8)
DOTTED = (0.9, 1.7)

# credit rules: (table key, key label, colour, dash pattern, marker)
RULES = (
    ("exact", "exact path", COLORS["bp"], None, "o"),
    ("unit_broadcast", "unit broadcast", COLORS["local"], DASHED, "s"),
    ("calibrated_broadcast", "initial profile", COLORS["additive"], DOTTED, "^"),
)
RULE_DODGE = {"exact": -0.17, "unit_broadcast": 0.0, "calibrated_broadcast": 0.17}
STEPS = (0, 64, 256, 1024, 2048, 4096, 8192, 12288, 16384)
FRESH_STEPS = (0, 64, 256, 1024, 2048, 4096)
EXT_STEPS = (4096, 8192, 12288, 16384)
SEEDS = tuple(range(1101, 1121))
TASKS = (("A", "ungated_independent", "Ungated, independent inputs"),
         ("B", "gated_conflict", "Gated, conflicting inputs"))
BUDGET = 4096
ADAM_RATE = 0.03
SGD_RATE = 0.3

# the update axis: log from 64, update 0 as a pip outside a broken axis
PIP_X = 24.0
X_BREAK = (32.0, 46.0)
XLIM = (16.0, 26000.0)
YLIM_LOG = (-9.6, 1.95)          # log10 test NMSE; holds every per-seed value
FAN_ALPHA = 0.25

D_YLIM = (0.866, 1.021)
D_METRICS = (("path_rank_one_capture", "best rank-one\npath capture", COLORS["bp"], "o", -0.17),
             ("eligibility_calibrated_oracle_capture", "eligibility-weighted\ninitial-profile capture",
              COLORS["additive"], "^", 0.17))


# ── tables ───────────────────────────────────────────────────────────────
def csv(rel):
    path = SOURCE / rel
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def boot_check(values, lo, hi, *, draws=20000, seed=20260912, tol=0.15):
    """The study's intervals are percentile bootstraps of the seed mean
    (20,000 resamples; ``conductance_demand_methods.tex``) whose generator
    state is not recorded, so they cannot be reproduced bit for bit.  This
    recomputes one and requires the frozen bounds to lie within ``tol`` of
    the interval width of it, and the mean to lie inside the frozen bounds."""
    values = np.asarray(values, float)
    assert lo <= values.mean() <= hi, (values.mean(), lo, hi)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
    my_lo, my_hi = np.quantile(means, [0.025, 0.975])
    width = hi - lo
    if width <= 1e-9:                       # a constant (1 by construction)
        assert np.ptp(values) <= 1e-9
        return
    assert abs(my_lo - lo) <= tol * width and abs(my_hi - hi) <= tol * width, \
        (lo, hi, my_lo, my_hi)


def load_curves():
    """Per-seed Adam test NMSE at the nine checkpoints, both tasks, all rules:
    ``{(task, rule): DataFrame[seed x step]}``."""
    fresh = csv("summaries/all_curves.csv")
    fresh = fresh[fresh.phase.eq("fresh") & fresh.optimizer.eq("adam") & fresh.selected_rate]
    assert (fresh.rate == ADAM_RATE).all() and len(fresh) == 720
    ext = pd.concat([csv(f"extension/seed_{s}_curves.csv") for s in SEEDS], ignore_index=True)
    ext = ext[ext.optimizer.eq("adam")]
    assert (ext.rate == ADAM_RATE).all() and ext.selected_rate.all() and (ext.phase == "extension").all()
    assert len(ext) == 20 * 2 * 3 * 4
    # the extension rows are also carried by the summary table: same values
    summary_ext = csv("summaries/all_curves.csv")
    summary_ext = summary_ext[summary_ext.phase.eq("extension") & summary_ext.optimizer.eq("adam")]
    m = ext.merge(summary_ext, on=["seed", "task", "rule", "step"], suffixes=("", "_s"))
    assert len(m) == len(ext)
    np.testing.assert_allclose(m.test_nmse, m.test_nmse_s, rtol=1e-12, atol=0)
    out = {}
    for _, task, _ in TASKS:
        for key, *_ in RULES:
            f = fresh[fresh.task.eq(task) & fresh.rule.eq(key)].pivot(index="seed", columns="step", values="test_nmse")
            e = ext[ext.task.eq(task) & ext.rule.eq(key)].pivot(index="seed", columns="step", values="test_nmse")
            assert list(f.columns) == list(FRESH_STEPS) and list(e.columns) == list(EXT_STEPS)
            assert list(f.index) == list(SEEDS) and list(e.index) == list(SEEDS)
            # the continuation starts from the saved 4,096 state: identical
            np.testing.assert_allclose(f[4096], e[4096], rtol=1e-12, atol=0)
            out[(task, key)] = pd.concat([f, e.drop(columns=[4096])], axis=1)[list(STEPS)]
            assert (out[(task, key)] > 0).all().all()
        # all three rules share the seed's initialization error
        z0 = np.column_stack([out[(task, k)][0].to_numpy() for k, *_ in RULES])
        assert np.ptp(z0, axis=1).max() == 0.0
    return out


# ── A, B ─────────────────────────────────────────────────────────────────
def xpos(step, dodge=0.0):
    return (PIP_X if step == 0 else float(step)) * 2.0 ** dodge


def panel_learning(ax, source, curves, panel, task, *, show_y):
    src = source[source.panel.eq(panel)]
    assert src.task.eq(task).all() and src.optimizer.eq("adam").all() and (src.n == 20).all()
    endpoint = {}
    for key, label, color, dashes, marker in RULES:
        rows = src[src.rule.eq(key)].set_index("step").loc[list(STEPS)]
        per_seed = curves[(task, key)]
        # the table's means ARE the seed means; its intervals bracket them
        np.testing.assert_allclose(per_seed.mean(axis=0).to_numpy(), rows["mean"].to_numpy(), rtol=1e-12, atol=0)
        for step in STEPS:
            boot_check(per_seed[step].to_numpy(), rows.loc[step, "ci_low"], rows.loc[step, "ci_high"])
        assert np.log10(per_seed.to_numpy()).min() > YLIM_LOG[0] + 0.3
        assert np.log10(per_seed.to_numpy()).max() < YLIM_LOG[1] - 0.25
        dodge = RULE_DODGE[key]
        xs = np.array([xpos(s, dodge) for s in STEPS])
        # the fan: every seed's trajectory as a hairline (checkpoints 64 on)
        # and a dot at the update-0 pip
        for seed in SEEDS:
            y = np.log10(per_seed.loc[seed].to_numpy())
            ax.plot(xs[1:], y[1:], color=color, lw=LW_HAIR, alpha=FAN_ALPHA, zorder=1.8,
                    solid_capstyle="butt")
            ax.plot([xs[0]], [y[0]], linestyle="none", marker="o", markersize=SEED_MS * 0.7,
                    markerfacecolor=color, markeredgecolor="none", alpha=FAN_ALPHA, zorder=1.8)
        mean = np.log10(rows["mean"].to_numpy())
        lo = np.log10(rows["ci_low"].to_numpy())
        hi = np.log10(rows["ci_high"].to_numpy())
        line_kw = dict(color=color, lw=LW_DATA, zorder=3.0)
        if dashes:
            line_kw["dashes"] = dashes
        ax.plot(xs[1:], mean[1:], **line_kw)
        ax.errorbar(xs, mean, yerr=[mean - lo, hi - mean], fmt=marker, ms=MARKER_MS * 0.78,
                    color=color, markeredgecolor="white", markeredgewidth=LW_HAIR,
                    ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE * 0.8,
                    capthick=LW_ERR, zorder=3.5, linestyle="none")
        endpoint[key] = (rows.loc[16384, "mean"], rows.loc[16384, "ci_low"], rows.loc[16384, "ci_high"])
        print(f"[{panel}] {label}: mean {rows.loc[0, 'mean']:.5g} at 0, {rows.loc[64, 'mean']:.5g} at 64, "
              f"{rows.loc[BUDGET, 'mean']:.4g} at {BUDGET}, {rows.loc[16384, 'mean']:.4g} "
              f"[{rows.loc[16384, 'ci_low']:.3g}, {rows.loc[16384, 'ci_high']:.3g}] at 16,384; "
              f"seeds {per_seed.min().min():.2e}-{per_seed.max().max():.2e}")
    # the budget rule, named
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM_LOG)
    ax.plot([BUDGET, BUDGET], list(YLIM_LOG), color=MUTE, lw=LW_REF, dashes=DOTTED, zorder=1.0,
            solid_capstyle="butt")
    ax.annotate("4,096-update budget", xy=(BUDGET, YLIM_LOG[1]), xycoords="data",
                xytext=(-3.0, -1.5), textcoords="offset points", ha="right", va="top",
                fontsize=PT_BASE, color=MUTE)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM)
    ax.set_xticks([PIP_X, 64, 256, 1024, 4096, 16384],
                  ["0", "64", "256", "1,024", "4,096", "16,384"])
    ax.set_xticks([2048, 8192, 12288], minor=True)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([-8, -6, -4, -2, 0], ["−8", "−6", "−4", "−2", "0"])
    ax.set_xlabel("training update")
    if show_y:
        ax.set_ylabel("log10 test NMSE")
    else:
        ax.tick_params(axis="y", labelleft=False)
    return endpoint


def axis_break_x(ax, lo, hi, *, size_pt=2.2):
    """White out the bottom spine between ``lo`` and ``hi`` (data units) and
    draw the two diagonal break marks (the boolean-learning H precedent)."""
    bb = ax.get_window_extent()
    h_pt = bb.height * 72.0 / ax.figure.dpi
    y0, y1 = ax.get_ylim()
    dy = size_pt / h_pt * (y1 - y0)
    ax.add_patch(Rectangle((lo, y0 - 1.6 * dy), hi - lo, 3.2 * dy, facecolor="white",
                           edgecolor="none", zorder=6, clip_on=False))
    for x in (lo, hi):
        f = 0.35 * np.log10(hi / lo)
        ax.plot([x * 10.0 ** (-f), x * 10.0 ** f], [y0 - dy, y0 + dy], color=EDGE, lw=LW_EDGE,
                zorder=7, clip_on=False, solid_capstyle="butt")


def rule_key(ax):
    handles = [Line2D([], [], color=c, lw=LW_DATA, dashes=d if d else (None, None), marker=m,
                      ms=MARKER_MS * 0.78, markeredgecolor="white", markeredgewidth=LW_HAIR, label=lab)
               for _, lab, c, d, m in RULES]
    handles.append(Line2D([], [], color=MUTE, lw=LW_HAIR, alpha=0.7, label="single seeds (20 per rule)"))
    leg = ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=PT_BASE,
                    handlelength=2.4, handletextpad=0.6, labelspacing=0.35, borderaxespad=0.4,
                    title="means, 95 % CI whiskers", title_fontsize=PT_BASE, alignment="left")
    leg.get_title().set_color(MUTE)
    return leg


# ── C: the paired endpoint gaps ──────────────────────────────────────────
def panel_gap(ax, source, seed_contrasts, endpoints, optimizer, *, ylim, yticks, show_y):
    row = source[source.panel.eq("C") & source.optimizer.eq(optimizer)]
    assert len(row) == 1
    row = row.iloc[0]
    assert row.rule == "calibrated_broadcast" and row.contrast == "target_gap" and row.phase == "extension"
    assert row.rate_scope == "selected" and row.n == 20 and row.n_positive == 20 and row.n_negative == 0
    z = seed_contrasts[seed_contrasts.phase.eq("extension") & seed_contrasts.optimizer.eq(optimizer)
                       & seed_contrasts.rate_scope.eq("selected") & seed_contrasts.rule.eq("calibrated_broadcast")
                       & seed_contrasts.contrast.eq("target_gap")].sort_values("seed")
    seeds = z.value.to_numpy(float)
    assert len(seeds) == 20 and list(z.seed) == list(SEEDS)
    # the same 20 differences from the per-seed endpoint files
    ep = endpoints[endpoints.optimizer.eq(optimizer) & endpoints.task.eq("gated_conflict") & endpoints.selected_rate]
    assert (ep.rate == (ADAM_RATE if optimizer == "adam" else SGD_RATE)).all()
    assert (ep.phase == "extension").all() and ep.best_step.max() <= 16384
    diff = (ep[ep.rule.eq("calibrated_broadcast")].set_index("seed").test_nmse
            - ep[ep.rule.eq("exact")].set_index("seed").test_nmse).loc[list(SEEDS)].to_numpy()
    np.testing.assert_allclose(diff, seeds, rtol=0, atol=1e-15)
    np.testing.assert_allclose(seeds.mean(), row["mean"], rtol=1e-12, atol=0)
    assert int((seeds > 0).sum()) == 20
    boot_check(seeds, row.ci_low, row.ci_high)
    scale = 1000.0                     # drawn in thousandths of NMSE
    assert seeds.max() * scale < ylim[1] and ylim[0] < 0
    color = COLORS["additive"]         # the initial-profile rule's own gap
    jitter = np.linspace(-0.2, 0.2, len(seeds))
    ax.plot(jitter, seeds * scale, linestyle="none", marker="o", markersize=SEED_MS,
            markerfacecolor=color, markeredgecolor="none", alpha=SEED_ALPHA, zorder=2.0)
    ax.errorbar([0.0], [row["mean"] * scale], yerr=[[(row["mean"] - row.ci_low) * scale],
                                                     [(row.ci_high - row["mean"]) * scale]],
                fmt="^", ms=MARKER_MS, color=color, markeredgecolor="white", markeredgewidth=LW_HAIR,
                ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=4.0)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(*ylim)
    ax.plot([-0.6, 0.6], [0.0, 0.0], color=MUTE, lw=LW_REF, dashes=DASHED, zorder=1.0,
            solid_capstyle="butt")
    ax.annotate("no gap", xy=(0.6, 0.0), xycoords="data", xytext=(-1.5, 2.5),
                textcoords="offset points", ha="right", va="bottom", fontsize=PT_BASE, color=MUTE)
    ax.set_xticks([0.0], ["20/20 seeds > 0"])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks(yticks, [f"{t:g}" for t in yticks])
    if show_y:
        ax.set_ylabel("initial profile − exact,\ngated-task test NMSE × 1000")
    print(f"[C] {optimizer}: gap {row['mean']:.6f} [{row.ci_low:.6f}, {row.ci_high:.6f}], "
          f"{int(row.n_positive)}/{int(row.n)} seeds > 0, seeds {seeds.min():.2e}-{seeds.max():.2e}, "
          f"best steps {int(ep.best_step.min())}-{int(ep.best_step.max())}")
    return row["mean"], row.ci_low, row.ci_high


# ── D: path-field capture at the exact-rule states ───────────────────────
def panel_capture(ax, source, endpoints):
    ep = endpoints[endpoints.optimizer.eq("adam") & endpoints.rule.eq("exact") & endpoints.selected_rate]
    assert (ep.phase == "extension").all() and (ep.rate == ADAM_RATE).all()
    xs = {"ungated_independent": 0.0, "gated_conflict": 1.0}
    halves = []
    for metric, label, color, marker, dx in D_METRICS:
        for task, x in xs.items():
            row = source[source.panel.eq("D") & source.task.eq(task) & source.metric.eq(metric)]
            assert len(row) == 1
            row = row.iloc[0]
            g = ep[ep.task.eq(task)].sort_values("seed")
            assert list(g.seed) == list(SEEDS)
            seeds = g[metric].to_numpy(float)
            np.testing.assert_allclose(seeds.mean(), row["mean"], rtol=1e-12, atol=0)
            boot_check(seeds, row.ci_low, row.ci_high)
            assert seeds.min() > D_YLIM[0] and seeds.max() <= 1.0 + 1e-9
            if metric == "path_rank_one_capture" and task == "ungated_independent":
                # rank one by construction: every seed's path field has
                # effective rank exactly 1 at this state
                np.testing.assert_allclose(seeds, 1.0, rtol=0, atol=1e-12)
                np.testing.assert_allclose(g.path_effective_rank.to_numpy(), 1.0, rtol=0, atol=1e-9)
            half = 0.5 * (row.ci_high - row.ci_low)
            halves.append(half)
            jitter = np.linspace(-0.07, 0.07, len(seeds))
            ax.plot(x + dx + jitter, seeds, linestyle="none", marker="o", markersize=SEED_MS * 0.85,
                    markerfacecolor=color, markeredgecolor="none", alpha=SEED_ALPHA, zorder=2.0)
            ax.errorbar([x + dx], [row["mean"]], yerr=[[row["mean"] - row.ci_low], [row.ci_high - row["mean"]]],
                        fmt=marker, ms=MARKER_MS, color=color, markeredgecolor="white",
                        markeredgewidth=LW_HAIR, ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                        capthick=LW_ERR, zorder=4.0)
            print(f"[D] {task} {metric}: {row['mean']:.5f} [{row.ci_low:.5f}, {row.ci_high:.5f}], "
                  f"seeds {seeds.min():.5f}-{seeds.max():.5f}; 95 % CI half-width {half:.2e}")
    ax.set_xlim(-0.55, 1.5)
    ax.set_ylim(*D_YLIM)
    ax.set_xticks([0.0, 1.0], ["ungated", "gated conflict"])
    ax.set_yticks([0.90, 0.95, 1.00], ["0.90", "0.95", "1.00"])
    ax.set_ylabel("captured squared energy")
    ax.annotate("= 1 by construction", xy=(xs["ungated_independent"] + D_METRICS[0][4], 1.0),
                xycoords="data", xytext=(0.0, 5.0), textcoords="offset points", ha="center",
                va="bottom", fontsize=PT_BASE, color=COLORS["bp"])
    handles = [Line2D([], [], linestyle="none", marker=m, ms=MARKER_MS, color=c,
                      markeredgecolor="white", markeredgewidth=LW_HAIR, label=lab)
               for _, lab, c, m, _ in D_METRICS]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=PT_BASE,
              handletextpad=0.6, labelspacing=0.5, borderaxespad=0.4,
              title="20 seeds per mean", title_fontsize=PT_BASE,
              alignment="left").get_title().set_color(MUTE)
    return halves


def narrow_note(ax, halves):
    """How many of D's four 95 % intervals are hidden inside their marker,
    measured against the FINAL axes box, and printed on the panel (top
    right, clear of the ungated fans at x <= 0.24 and the gated fans at
    y <= 0.978)."""
    h_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
    radius = 0.5 * MARKER_MS / h_pt * (D_YLIM[1] - D_YLIM[0])   # marker radius, data units
    narrow = int(sum(half < radius for half in halves))
    print(f"[D] marker radius {radius:.2e}; {narrow} of {len(halves)} intervals narrower than it")
    assert narrow == 3, narrow
    ax.annotate(f"95 % CI within the marker\nfor {narrow} of {len(halves)} means", xy=(1.0, 1.0),
                xycoords=("axes fraction", "data"), xytext=(-2.0, 0.0), textcoords="offset points",
                ha="right", va="center", fontsize=PT_BASE, color=MUTE, linespacing=1.1)
    return narrow


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 370.0
ROW_PT = [172.0, 106.0]
HGUTTER_PT = 30.0
VGUTTER_PT = 44.0
MARGINS = Margins(left=46.0, right=12.0, top=18.0, bottom=30.0)


def build(path: Path = OUT, *, png=False):
    source = csv("figures/supplement_first_conductance_source.csv")
    assert len(source) == 60 and source.panel.value_counts().to_dict() == {"A": 27, "B": 27, "C": 2, "D": 4}
    curves = load_curves()
    seed_contrasts = csv("summaries/paired_seed_contrasts.csv")
    endpoints = pd.concat([csv(f"extension/seed_{s}_endpoints.csv") for s in SEEDS], ignore_index=True)
    assert len(endpoints) == 240 and endpoints.seed.nunique() == 20

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 2, row_weights=ROW_PT, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS)
    ax_a = cv.panel("A", 0, 0, 6, grid="y", title=TASKS[0][2])
    ax_b = cv.panel("B", 0, 6, 6, grid="y", title=TASKS[1][2])
    ax_c = cv.panel("C", 1, 0, 3, grid="y", title="Adam")
    ax_c2 = cv.panel("C_sgd", 1, 3, 3, letter="", grid="y", title="SGD, own y scale")
    ax_d = cv.panel("D", 1, 6, 6, grid="y", title="Path-field capture at exact-rule states")

    end_a = panel_learning(ax_a, source, curves, "A", "ungated_independent", show_y=True)
    end_b = panel_learning(ax_b, source, curves, "B", "gated_conflict", show_y=False)
    # the caption's endpoint claims
    np.testing.assert_allclose(end_a["exact"], (5.892e-6, 1.461e-6, 1.211e-5), rtol=1e-3)
    np.testing.assert_allclose(end_a["unit_broadcast"], (1.643e-6, 7.007e-7, 2.736e-6), rtol=1e-3)
    np.testing.assert_allclose(end_a["calibrated_broadcast"], (3.335e-6, 3.364e-7, 8.314e-6), rtol=1e-3)
    np.testing.assert_allclose(end_b["exact"], (1.215e-6, 6.343e-7, 1.956e-6), rtol=1e-3)
    np.testing.assert_allclose(end_b["calibrated_broadcast"], (5.160e-4, 2.806e-4, 7.898e-4), rtol=1e-3)
    # the intervals overlap on the ungated task at 16,384 updates
    assert end_a["exact"][1] < end_a["unit_broadcast"][2] and end_a["unit_broadcast"][1] < end_a["exact"][2]
    # ... and do not on the gated task
    assert end_b["exact"][2] < end_b["unit_broadcast"][1] and end_b["exact"][2] < end_b["calibrated_broadcast"][1]
    rule_key(ax_a)

    adam = panel_gap(ax_c, source, seed_contrasts, endpoints, "adam", ylim=(-0.12, 2.2),
                     yticks=[0.0, 0.5, 1.0, 1.5, 2.0], show_y=True)
    sgd = panel_gap(ax_c2, source, seed_contrasts, endpoints, "sgd", ylim=(-0.02, 0.37),
                    yticks=[0.0, 0.1, 0.2, 0.3], show_y=False)
    np.testing.assert_allclose(adam, (0.000502, 0.000267, 0.000772), rtol=0, atol=5e-7)
    np.testing.assert_allclose(sgd, (0.000114, 0.000076, 0.000157), rtol=0, atol=5e-7)
    assert adam[0] < 6e-4

    halves = panel_capture(ax_d, source, endpoints)

    # Settle the boxes.  The measured locks differ by column (D's y label
    # spends the gutter that A's and C_sgd's right sides would otherwise
    # use), so one uniform reserve -- the largest measured on any side -- is
    # declared on every panel: every column then shares one x0 and every
    # same-span pair one axes width by construction.
    cv.lock_reserves()
    left = max(v[0] for v in cv._locks.values())
    right = max(v[1] for v in cv._locks.values())
    for name in ("A", "B", "C", "C_sgd", "D"):
        cv.declare_reserve(name, left=left, right=right)
    cv.lock_reserves()
    narrow_note(ax_d, halves)
    for ax in (ax_a, ax_b):
        axis_break_x(ax, *X_BREAK)
    boxes = {k: cv.axes[k].get_position() for k in ("A", "B", "C", "C_sgd", "D")}
    assert abs(boxes["A"].width - boxes["B"].width) * cv.width_pt < 0.5
    assert abs(boxes["C"].width - boxes["C_sgd"].width) * cv.width_pt < 0.5
    assert abs(boxes["B"].x0 - boxes["D"].x0) * cv.width_pt < 0.5
    assert cv.height_pt <= 540.0

    problems = cv.save(path, name="figure_conductance_precision_native", png=png)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

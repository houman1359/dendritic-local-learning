#!/usr/bin/env python3
"""Supplementary sheet S18 (ident ``conductance_grouping``) -- input grouping
and restricted feedback in positive-conductance trees -- rebuilt as ONE native
full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/conductance_grouping.pdf``)
is a paste of an upstream render with no generator in this repository; this
builder reads ONLY the frozen tables under ``source_data/morphology_conductance``
and redraws the same six panels with the same plotted quantities.  Nothing
about the numbers changes; every printed or plotted value is asserted against
the table it comes from, and every summary is recomputed from the raw
per-seed rows with the study's own bootstrap (``run.py:boot``, 10,000
whole-seed draws, ``default_rng(159_000)``) before it is drawn.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
frozen_S18.json), panel by panel:

* A draws all THREE input groupings (01|23, 02|13, 03|12) as three seven-
  compartment trees in one slot, the leaf-pair assignment carried by the leaf
  labels; the four excitatory and six inhibitory contacts are marked on the
  first tree with the anatomy glyphs, the six couplings are the six drawn
  edges, and the two subtrees are two neutral greys (ADDRESS_RAMP), never a
  series hue.  Both in-panel caption lines are gone.
* B is a third-width panel: the 120 incompatible task/grouping pairs are
  drawn behind the 20 seed means and the seed-bootstrap mean; the compatible
  column is the 20 superimposed seeds at exactly zero with a value label.
  The in-plot sentence is gone, the axis is tight to the data and the
  quadrature caveat lives in the y-axis label.  One neutral colour.
* C and D draw the four rules as five-checkpoint trajectories on an ordinal
  update axis (0, 10, 100, 300, 1,000 all ticked) over a log NMSE axis, with
  95 % whiskers, and replace the eight duplicated incompatible marks by ONE
  tinted band per panel (the union of the four rules' 95 % intervals).  Full
  rule names live in the shared key; 'Adam' is set once, in sentence case.
* E is the paired compatible-tree contrast against exact path for the fixed
  broadcast AND the two oracle controls, under both optimizers, so the six
  strips fill the width; colour means the credit rule, the optimizer is on
  the x axis.
* F is on the same ordinal update axis, tight to the data (0.81-1.005), with
  every per-seed mean drawn as a fan behind the mean and a 95 % seed-
  bootstrap whisker; the exact rule (1.0 by construction in all 60 records
  at every checkpoint) is a labelled dashed reference at 1, not a series.
* One palette register for the credit rules across the sheet, the one the
  cross-figure review asked for: exact path dark red (``bp``), calibrated
  broadcast blue (``additive``, the main Fig. 4 hue), two subtree profiles
  purple (``oracle``) and the one-profile oracle control the neutral grey
  control series (``point_mlp``).  Oracle controls are dashed wherever a
  line is drawn.  No colour carries a second meaning anywhere on the sheet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

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
    tint_pct,
)
from journal_style import ADDRESS_RAMP  # noqa: E402
from native_schematics import Frame  # noqa: E402

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "morphology_conductance"
SUMMARY = SOURCE / "summaries" / "fresh"
BOUND = SOURCE / "interaction_bound"
OUT = ROOT / "figures" / "supplementary" / "figure_conductance_grouping_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
DASHED = (0, (2.6, 1.8))

# credit rules: (table key, key label, colour, line style, marker)
RULES = (
    ("exact_path", "exact path", COLORS["bp"], "-", "o"),
    ("calibrated_broadcast", "fixed calibrated broadcast", COLORS["additive"], "-", "s"),
    ("broadcast_projection", "one oracle profile", COLORS["point_mlp"], DASHED, "^"),
    ("subtree_projection", "two subtree profiles (oracle)", COLORS["oracle"], DASHED, "D"),
)
STEPS = (0, 10, 100, 300, 1000)
STEP_X = {s: i for i, s in enumerate(STEPS)}
OPTIMIZERS = (("adam", "Adam"), ("sgd", "SGD"))
GROUPINGS = (("01_23", ((0, 1), (2, 3))), ("02_13", ((0, 2), (1, 3))),
             ("03_12", ((0, 3), (1, 2))))
BOUND_KEY = "additive_subtree_nmse_lower_bound"


def boot(values):
    """``run.py:boot`` verbatim: the study's whole-seed bootstrap."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(159_000)
    draws = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
    return (float(values.mean()), float(np.quantile(draws, .025)),
            float(np.quantile(draws, .975)))


def csv(name, base=SUMMARY):
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fan(ax, x, values, color, *, half=0.14, ms=SEED_MS, alpha=SEED_ALPHA, zorder=2.0,
        shuffle=False):
    """Per-seed (or per-pair) values as a jittered fan behind the mean; the
    jitter is the table order, or a fixed permutation of it when duplicate
    rows would otherwise line up side by side (presentation only)."""
    values = np.asarray(values, float)
    jitter = np.linspace(-half, half, len(values)) if len(values) > 1 else np.zeros(1)
    if shuffle:
        jitter = jitter[np.random.default_rng(20260912).permutation(len(values))]
    ax.plot(x + jitter, values, linestyle="none", marker="o", markersize=ms,
            markerfacecolor=color, markeredgecolor="none", alpha=alpha, zorder=zorder)


def mean_whisker(ax, x, mean, lo, hi, color, *, marker="o", zorder=4.0):
    ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt=marker, ms=MARKER_MS,
                color=color, markeredgecolor="white", markeredgewidth=LW_HAIR,
                ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE, capthick=LW_ERR,
                zorder=zorder)


# ── A: three groupings of one physical tree ──────────────────────────────
def panel_groupings(ax, protocol, curves):
    """Seven compartments, two proximal subtrees, four leaves; the grouping
    is which pair of inputs feeds each proximal subtree."""
    assert protocol["compartments"] == 7 and protocol["couplings"] == 6
    assert protocol["excitatory_contacts"] == 4 and protocol["inhibitory_contacts"] == 6
    assert protocol["trainable_positive_conductances"] == 16
    names = sorted(curves.task_group_name.unique())
    assert names == sorted(g[0] for g in GROUPINGS), names
    assert sorted(curves.student_group_name.unique()) == names

    f = Frame(ax)
    W, H = f.w_pt, f.h_pt
    X, Y = f.fx, f.fy
    key_pt = 7.0                       # glyph key band at the bottom
    head_pt = 12.0                     # grouping name above each tree
    n = len(GROUPINGS)
    gap = 10.0
    cell_w = (W - (n - 1) * gap) / n
    tree_top = H - head_pt - 12.0      # leaf labels and contacts sit above the leaves
    tree_bot = key_pt + 14.0
    tree_h = tree_top - tree_bot
    greys = (ADDRESS_RAMP[0], ADDRESS_RAMP[1])
    for i, (name, pairs) in enumerate(GROUPINGS):
        x0 = i * (cell_w + gap)
        cx = x0 + cell_w / 2.0
        # geometry in points: soma at the bottom centre, two proximal
        # compartments at mid height, four leaves across the top
        soma = (cx, tree_bot + 3.0)
        prox = [(cx - 0.26 * cell_w, tree_bot + 0.46 * tree_h),
                (cx + 0.26 * cell_w, tree_bot + 0.46 * tree_h)]
        leaf_dx = 0.13 * cell_w
        leaves = [(prox[0][0] - leaf_dx, tree_top), (prox[0][0] + leaf_dx, tree_top),
                  (prox[1][0] - leaf_dx, tree_top), (prox[1][0] + leaf_dx, tree_top)]
        P = lambda p: (X(p[0]), Y(p[1]))  # noqa: E731
        f.text((X(cx), Y(H - 3.0)), name.replace("_", " | "), size=PT_BASE, color=INK,
               va="top")
        for s in range(2):
            f.dendrite(P(soma), P(prox[s]), level=0, color=greys[s])
            for k in range(2):
                f.dendrite(P(prox[s]), P(leaves[2 * s + k]), level=2, color=greys[s])
        for s in range(2):
            f.junction(P(prox[s]))
            for k in range(2):
                leaf = leaves[2 * s + k]
                f.disc(P(leaf), 1.65, fill=greys[s], zorder=3.5)
                f.text((X(leaf[0]), Y(leaf[1] + 9.0)), str(pairs[s][k]), size=PT_BASE,
                       color=INK, va="center")
        if i == 0:
            # the sixteen conductances: E + I on every leaf, I on each proximal
            # compartment, and the six couplings are the six edges drawn
            for leaf in leaves:
                f.contact(P((leaf[0] - 3.0, leaf[1] + 3.4)), kind="exc")
                f.contact(P((leaf[0] + 3.0, leaf[1] + 3.4)), kind="inh")
            for s, sgn in ((0, -1.0), (1, 1.0)):
                f.contact(P((prox[s][0] + sgn * 5.0, prox[s][1] + 0.6)), kind="inh")
        f.soma(P(soma))
        f.error_in(P(soma), side="right")
    f.require_delta0()
    # glyph key: one line, three entries
    ky = 3.5
    entries = (("exc", f"excitatory contact ×{protocol['excitatory_contacts']}"),
               ("inh", f"inhibitory contact ×{protocol['inhibitory_contacts']}"),
               ("edge", f"coupling ×{protocol['couplings']}"))
    x = 4.0
    for kind, label in entries:
        if kind == "edge":
            f.dendrite((X(x), Y(ky)), (X(x + 9.0), Y(ky)), level=2, color=greys[0])
            x += 9.0
        else:
            f.contact((X(x + 2.0), Y(ky)), kind=kind)
            x += 4.0
        t = f.text((X(x + 2.5), Y(ky)), label, size=PT_BASE, color=INK, ha="left")
        f.ax.figure.canvas.draw()
        bb = t.get_window_extent(f.ax.figure.canvas.get_renderer())
        x += 2.5 + bb.width * 72.0 / f.ax.figure.dpi + 11.0


# ── B: interaction bound ─────────────────────────────────────────────────
def panel_bound(ax, bounds, seed_summary, report):
    inc = bounds[~bounds.compatible]
    com = bounds[bounds.compatible]
    assert len(inc) == 120 and len(com) == 60 and bounds.seed.nunique() == 20
    assert inc.groupby("seed").size().eq(6).all() and com.groupby("seed").size().eq(3).all()
    assert (com[BOUND_KEY] == 0.0).all(), "compatible bound must be exactly zero"
    assert report["all_compatible_bounds_zero"] is True
    pairs = inc[BOUND_KEY].to_numpy(float)
    np.testing.assert_allclose(pairs.min(), report["min_incompatible_nmse_lower_bound"], rtol=0, atol=1e-15)
    distinct = inc.groupby("seed")[BOUND_KEY].nunique()
    assert distinct.eq(2).all(), "each seed's six incompatible pairs take two distinct bound values"
    seeds = inc.groupby("seed")[BOUND_KEY].mean().sort_index().to_numpy(float)
    assert len(seeds) == 20
    mean, lo, hi = boot(seeds)
    np.testing.assert_allclose(mean, seed_summary["mean_incompatible_population_nmse_lower_bound"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(lo, seed_summary["ci95_low"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(hi, seed_summary["ci95_high"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(mean, report["mean_incompatible_nmse_lower_bound"], rtol=0, atol=1e-12)
    assert abs(mean - 0.013222) < 5e-7, mean          # the caption's number
    print(f"[B] incompatible pairs n={len(pairs)} range {pairs.min():.6f}-{pairs.max():.6f}; "
          f"seed means n={len(seeds)} range {seeds.min():.6f}-{seeds.max():.6f}; "
          f"mean {mean:.6f} [{lo:.6f}, {hi:.6f}]; compatible: {len(com)} rows == 0")

    # the 120 pairs (small, pale) behind the 20 seed means (the fan) and the
    # seed-bootstrap mean; the compatible column is 20 seeds at exactly zero
    fan(ax, 0.0, pairs, INK, half=0.26, ms=1.7, alpha=0.22, zorder=1.8, shuffle=True)
    fan(ax, 0.0, seeds, INK, half=0.13)
    mean_whisker(ax, 0.0, mean, lo, hi, INK)
    zero = com.groupby("seed")[BOUND_KEY].mean().to_numpy(float)
    assert len(zero) == 20 and (zero == 0.0).all()
    fan(ax, 1.0, zero, INK, half=0.13)
    ax.plot([1.0], [0.0], marker="o", ms=MARKER_MS, color=INK, markeredgecolor="white",
            markeredgewidth=LW_HAIR, linestyle="none", zorder=4.0)
    ax.annotate("0 exactly,\nall 20 seeds", xy=(1.0, 0.0), xycoords="data",
                xytext=(0.0, 9.0), textcoords="offset points", ha="center", va="bottom",
                fontsize=PT_BASE, color=MUTE, linespacing=1.1)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks([0, 1], ["incompatible", "compatible"])
    top = 1.06 * pairs.max()
    ax.set_ylim(-0.0009, top)
    ax.set_yticks([0.0, 0.005, 0.010, 0.015, 0.020], ["0", "0.005", "0.010", "0.015", "0.020"])
    ax.set_xlabel("student grouping vs task")
    ax.set_ylabel("population NMSE\nlower bound (quadrature)")


# ── C, D: learning trajectories under each optimizer ─────────────────────
def recompute_learning_summary(curves, summary):
    """Rebuild ``learning_summary.csv`` from the raw fit rows and assert it."""
    err = 0.0
    for _, r in summary.iterrows():
        g = curves[curves.optimizer.eq(r.optimizer) & curves.credit_rule.eq(r.credit_rule)
                   & curves.compatible.eq(r.compatible) & curves.step.eq(r.step)]
        assert len(g) == r.n_fits and g.seed.nunique() == r.n_seed_blocks == 20
        m, lo, hi = boot(g.groupby("seed").test_nmse.mean().sort_index())
        err = max(err, abs(m - r.mean_test_nmse), abs(lo - r.ci95_low), abs(hi - r.ci95_high))
    assert err < 1e-12, err
    return err


def panel_learning(ax, summary, optimizer):
    s = summary[summary.optimizer.eq(optimizer)]
    assert sorted(s.step.unique()) == list(STEPS) and s.credit_rule.nunique() == 4
    # incompatible: one band, the union of the four rules' 95 % intervals
    inc = s[~s.compatible]
    lo = inc.groupby("step").ci95_low.min().loc[list(STEPS)].to_numpy()
    hi = inc.groupby("step").ci95_high.max().loc[list(STEPS)].to_numpy()
    final = inc[inc.step.eq(1000)].mean_test_nmse
    xs = np.arange(len(STEPS), dtype=float)
    # a 16 % tint fill with a hairline edge (the tint_patch recipe, drawn as
    # a fill_between so its bounding box is not mistaken for a data mark)
    ax.fill_between(xs, lo, hi, facecolor=tint_pct(MUTE, 16), edgecolor=tint_pct(MUTE, 40),
                    linewidth=LW_HAIR, zorder=1.2)
    print(f"[{optimizer}] incompatible band: step-1000 means {final.min():.6f}-{final.max():.6f}, "
          f"95 % union {lo[-1]:.6f}-{hi[-1]:.6f}; step-0 union {lo[0]:.4f}-{hi[0]:.4f}")
    # compatible: four rules, dodged so the whiskers stay apart
    com = s[s.compatible]
    dodge = np.linspace(-0.18, 0.18, len(RULES))
    for dx, (key, _, color, ls, marker) in zip(dodge, RULES):
        z = com[com.credit_rule.eq(key)].set_index("step").loc[list(STEPS)]
        m = z.mean_test_nmse.to_numpy()
        ax.plot(xs + dx, m, color=color, lw=LW_DATA, ls=ls, zorder=3.0)
        ax.errorbar(xs + dx, m, yerr=[m - z.ci95_low.to_numpy(), z.ci95_high.to_numpy() - m],
                    fmt=marker, ms=MARKER_MS * 0.78, color=color, markeredgecolor="white",
                    markeredgewidth=LW_HAIR, ecolor=color, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE * 0.8, capthick=LW_ERR, zorder=3.5, linestyle="none")
    # all rules share the initialization error (paired conditions)
    for compatible in (True, False):
        z0 = s[s.step.eq(0) & s.compatible.eq(compatible)].mean_test_nmse
        assert np.ptp(z0.to_numpy()) == 0.0
    ax.set_yscale("log")
    ax.set_ylim(2.0e-5, 1.4)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0], ["0.0001", "0.001", "0.01", "0.1", "1"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.45, 4.45)
    ax.set_xticks(xs, [str(s) for s in STEPS])
    ax.set_xlabel("training update")
    ax.set_ylabel("test NMSE")
    # direct labels: the band above its flat tail, the curves below theirs
    ax.annotate("incompatible\n(all four rules)", xy=(4.0, hi[-1]), xycoords="data",
                xytext=(0.0, 7.0), textcoords="offset points", ha="right", va="bottom",
                fontsize=PT_BASE, color=MUTE, linespacing=1.1)
    ax.annotate("compatible", xy=(4.18, 2.0e-5), xycoords="data", xytext=(0.0, 1.5),
                textcoords="offset points", ha="right", va="bottom", fontsize=PT_BASE,
                color=MUTE)


# ── E: paired contrasts against exact path on compatible trees ───────────
def panel_contrasts(ax, contrasts, seed_contrasts):
    rules = [r for r in RULES if r[0] != "exact_path"]
    xs = {}
    x = 0.0
    printed = []
    for oi, (opt, _) in enumerate(OPTIMIZERS):
        for ri, (key, _, color, _, marker) in enumerate(rules):
            name = f"{key}_minus_exact_path__compatible_True"
            r = contrasts[contrasts.optimizer.eq(opt) & contrasts.contrast.eq(name)]
            assert len(r) == 1
            r = r.iloc[0]
            vals = seed_contrasts[seed_contrasts.optimizer.eq(opt) & seed_contrasts.contrast.eq(name)]
            vals = vals.sort_values("seed").difference.to_numpy(float)
            assert len(vals) == 20 == r.n_seed_blocks
            m, lo, hi = boot(vals)
            np.testing.assert_allclose([m, lo, hi], [r.mean_difference, r.ci95_low, r.ci95_high], rtol=0, atol=1e-12)
            assert int((vals > 0).sum()) == r.positive_seeds
            xs[(opt, key)] = x
            fan(ax, x, vals, color, half=0.2)
            mean_whisker(ax, x, m, lo, hi, color, marker=marker)
            printed.append(f"{opt} {key}: {m:.6f} [{lo:.6f}, {hi:.6f}] positive {r.positive_seeds}/20")
            x += 1.0
        x += 0.6
    # the two headline contrasts the caption states
    adam = contrasts[contrasts.optimizer.eq("adam") & contrasts.contrast.eq("calibrated_broadcast_minus_exact_path__compatible_True")].iloc[0]
    sgd = contrasts[contrasts.optimizer.eq("sgd") & contrasts.contrast.eq("calibrated_broadcast_minus_exact_path__compatible_True")].iloc[0]
    assert adam.ci95_low < 0 < adam.ci95_high and sgd.ci95_low > 0
    print("[E] " + "; ".join(printed))
    x0, x1 = -0.6, x - 0.6
    ax.set_xlim(x0, x1)
    ax.plot([x0, x1], [0.0, 0.0], color=MUTE, lw=LW_REF, dashes=(2.6, 2.0), zorder=1.0)
    ax.set_xticks([np.mean([xs[(o, r[0])] for r in rules]) for o, _ in OPTIMIZERS],
                  [lab for _, lab in OPTIMIZERS])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0.0, 0.001, 0.002], ["0", "0.001", "0.002"])
    ax.set_ylabel("credit rule − exact path\ntest NMSE")


# ── F: gradient alignment at common exact-trained states ─────────────────
def panel_alignment(ax, diagnostics, geometry):
    d = diagnostics[diagnostics.optimizer.eq("adam") & diagnostics.compatible]
    g = geometry[geometry.optimizer.eq("adam") & geometry.compatible]
    ex = d[d.credit_rule.eq("exact_path")]
    assert len(ex) == 60 * len(STEPS)
    np.testing.assert_allclose(ex.gradient_cosine.to_numpy(), 1.0, rtol=0, atol=1e-12)  # 1 by construction
    rules = [r for r in RULES if r[0] != "exact_path"]
    xs = np.arange(len(STEPS), dtype=float)
    dodge = np.linspace(-0.22, 0.22, len(rules))
    record_lo = 1.0
    seed_lo = 1.0
    curves = {}
    for dx, (key, _, color, ls, marker) in zip(dodge, rules):
        ms_, los, his = [], [], []
        for step in STEPS:
            z = d[d.credit_rule.eq(key) & d.step.eq(step)]
            assert len(z) == 60 and z.seed.nunique() == 20
            seeds = z.groupby("seed").gradient_cosine.mean().sort_index().to_numpy(float)
            m, lo, hi = boot(seeds)
            ref = g[g.credit_rule.eq(key) & g.step.eq(step)].gradient_cosine
            assert len(ref) == 1
            np.testing.assert_allclose(m, float(ref.iloc[0]), rtol=0, atol=1e-12)
            record_lo = min(record_lo, float(z.gradient_cosine.min()))
            seed_lo = min(seed_lo, float(seeds.min()))
            fan(ax, STEP_X[step] + dx, seeds, color, half=0.07, ms=SEED_MS * 0.85)
            ms_.append(m); los.append(lo); his.append(hi)
        ms_ = np.array(ms_)
        curves[key] = ms_
        ax.plot(xs + dx, ms_, color=color, lw=LW_DATA, ls=ls, zorder=3.0)
        ax.errorbar(xs + dx, ms_, yerr=[ms_ - np.array(los), np.array(his) - ms_], fmt=marker,
                    ms=MARKER_MS * 0.78, color=color, markeredgecolor="white",
                    markeredgewidth=LW_HAIR, ecolor=color, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE * 0.8, capthick=LW_ERR, zorder=3.5, linestyle="none")
    gap = np.abs(curves["calibrated_broadcast"] - curves["broadcast_projection"])
    print(f"[F] record minimum {record_lo:.4f}, seed-mean minimum {seed_lo:.4f}; fixed broadcast vs "
          f"one oracle profile |gap| per checkpoint {np.array2string(gap, precision=4)}")
    assert record_lo > 0.785 and gap.max() < 0.0065
    # exact path: 1 by construction -> a labelled reference, not a series
    ax.set_xlim(-0.5, 4.5)
    ax.plot([-0.5, 4.5], [1.0, 1.0], color=COLORS["bp"], lw=LW_REF, dashes=(2.6, 2.0), zorder=1.0)
    ax.annotate("exact path = 1 by construction", xy=(4.5, 1.0), xycoords="data",
                xytext=(0.0, 2.0), textcoords="offset points", ha="right", va="bottom",
                fontsize=PT_BASE, color=COLORS["bp"], annotation_clip=False)
    ax.set_ylim(0.81, 1.005)
    ax.set_yticks([0.85, 0.90, 0.95, 1.00], ["0.85", "0.90", "0.95", "1.00"])
    ax.set_xticks(xs, [str(s) for s in STEPS])
    ax.set_xlabel("training update")
    ax.set_ylabel("calibration-gradient\ncosine")


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 484.0
ROW_PT = [122.0, 104.0, 104.0]
HGUTTER_PT = 30.0
VGUTTER_PT = 46.0
MARGINS = Margins(left=46.0, right=10.0, top=18.0, bottom=46.0)


def build(path: Path = OUT):
    protocol = json.loads((SOURCE / "development_protocol.json").read_text())
    assert list(protocol["checkpoints"]) == list(STEPS) and protocol["steps"] == 1000
    curves = csv("all_learning_curves.csv")
    summary = csv("learning_summary.csv")
    contrasts = csv("paired_contrasts.csv")
    seed_contrasts = csv("paired_seed_contrasts.csv")
    diagnostics = csv("all_credit_diagnostics.csv")
    geometry = csv("credit_geometry_summary.csv")
    bounds = csv("population_bounds.csv", BOUND)
    seed_summary = json.loads((BOUND / "seed_summary.json").read_text())
    report = json.loads((BOUND / "report.json").read_text())
    assert len(curves) == 7200 and curves.seed.nunique() == 20 and len(diagnostics) == 7200
    print(f"[tables] learning_summary recomputed from raw rows, max |error| "
          f"{recompute_learning_summary(curves, summary):.2e}")
    inc_final = summary[~summary.compatible & summary.step.eq(1000)].mean_test_nmse
    assert 0.0289 < inc_final.min() and inc_final.max() < 0.0319, (inc_final.min(), inc_final.max())

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 3, row_weights=ROW_PT, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS, letter_clearance=True)
    ax_a = cv.panel("A", 0, 0, 7, schematic=True, title="Fixed physical shape; three input groupings")
    ax_b = cv.panel("B", 0, 7, 5, grid="y", title="Interaction bound")
    ax_c = cv.panel("C", 1, 0, 6, grid="y", title="Adam: test error by credit rule")
    ax_d = cv.panel("D", 1, 6, 6, grid="y", title="SGD: test error by credit rule")
    ax_e = cv.panel("E", 2, 0, 6, grid="y", title="Compatible trees: paired credit effect")
    ax_f = cv.panel("F", 2, 6, 6, grid="y", title="Adam: common-state gradient alignment")
    # one declared reserve on every panel: the column lock then gives every
    # same-span pair one axes width by construction (the S21 precedent)
    for name in "ABCDEF":
        cv.declare_reserve(name, left=15.0, right=8.0)
    # F's reference label sits above its axis top: lift the row-2 titles
    for ax in (ax_e, ax_f):
        ax.set_title(ax.get_title(), fontsize=PT_EMPH, color=INK, pad=11.0, fontweight="normal")

    panel_bound(ax_b, bounds, seed_summary, report)
    panel_learning(ax_c, summary, "adam")
    panel_learning(ax_d, summary, "sgd")
    panel_contrasts(ax_e, contrasts, seed_contrasts)
    panel_alignment(ax_f, diagnostics, geometry)
    cv.lock_reserves()              # settle the boxes before drawing A in points
    panel_groupings(ax_a, protocol, curves)

    # one shared key for the credit rules of C-F, below the panels
    handles = [Line2D([], [], color=c, lw=LW_DATA, ls=ls, marker=mk, ms=MARKER_MS * 0.78,
                      markeredgecolor="white", markeredgewidth=LW_HAIR, label=lab)
               for _, lab, c, ls, mk in RULES]
    cv.fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0),
                  ncol=4, frameon=False, fontsize=PT_BASE, handlelength=2.6,
                  columnspacing=1.6, handletextpad=0.6, borderaxespad=0.4)
    problems = cv.save(path, name="figure_conductance_grouping_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

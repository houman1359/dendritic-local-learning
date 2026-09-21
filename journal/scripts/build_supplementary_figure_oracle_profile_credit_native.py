#!/usr/bin/env python3
"""Supplementary sheet S13 (ident ``oracle_profile_credit``) -- task-dependent
credit requirements in bounded multi-affine trees -- rebuilt as ONE native
full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/oracle_profile_credit.pdf``)
is a paste of an upstream render with no generator in this repository; this
builder reads ONLY the frozen tables under ``source_data/morphology_credit``
and redraws the same six panels with the same plotted quantities.  Nothing
about the numbers changes; every printed or plotted value is asserted against
the table it comes from, and every summary is recomputed from the frozen
per-seed rows with the study's own bootstrap recipes before it is drawn:

* ``figure_condition_summary.csv`` / ``figure_gradient_summary.csv`` --
  ``build_morphology_credit_figure_tables.py:bootstrap``: 10,000 whole-seed
  draws, ``default_rng(202609052)`` restarted per condition, seeds in order;
* ``summaries/fresh/paired_contrasts.csv`` -- ``experiment.py:summarize``:
  ONE ``default_rng(127999)`` consumed in loop order, replayed here verbatim.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
frozen_S13.json), panel by panel:

* A, B: log NMSE axis (0.02-2.8) so the matching family and both noise
  floors resolve; every one of the 20 seeds per condition is a fan behind
  its mean; the floors are dotted rules over the family they apply to only
  (0.0225 = 0.15^2 / 1 for matching and nested, 0.045 = 0.15^2 / 0.5 for
  quartic, recomputed from the protocol noise SD and the runs' ``variance``
  column) and are keyed in the shared legend, not in caption prose inside
  the plot box.
* C: the 20 paired per-seed differences per condition are drawn behind each
  mean (the SGD trimodality is visible); the optimizer is named on the x
  axis, so the panel uses ONE colour -- the exact-path red-brown, because
  every value is exact credit -- and the rule palette is never re-bound; the
  Adam quartic interval, narrower than its marker, is printed beside the
  point; the positive-seed count of every strip is printed with its
  denominator; the axis label says ``input-reassigned - compatible`` so the
  word ``shuffled`` keeps its one meaning (the shuffled-profiles rule).
* D: ordinal update axis with all six checkpoints (0, 1, 16, 64, 256, 1,024)
  labelled and no minor ticks; y axis tight to the drawn values (the seed fan
  reaches -0.24); the exact path, 1.0 in all 120 records by construction, is
  a labelled dashed reference and not a series; title says the panel is the
  oracle-compatible arm.
* E, F: one common log NMSE axis; the three tested rates are three
  categorical positions with no minor ticks and leading-zero labels; every
  point carries its 20-seed fan and a 95 % whole-seed bootstrap whisker;
  rules are joined by a light connector in the rule's own dash pattern;
  the development-selected rate is the FILLED marker, the other tested rates
  are open -- the one fill meaning on the sheet, which A-D (all drawn at the
  selected rate) obey as well.
* Whole sheet: every rule has its own hue, marker AND dash pattern (exact
  path red-brown / circle / solid; root broadcast amber / square / dashed;
  one oracle profile grey / triangle / dash-dot; two subtree profiles violet
  / diamond / dotted; two shuffled profiles blue / cross / long dash); the
  exact-path hue is the main-text ``bp`` red-brown, as the cross-figure
  review asked; the optimizer is 'Adam' everywhere; one shared key.
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
    SEED_ALPHA,
    SEED_MS,
    Margins,
    NativeCanvas,
)

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "morphology_credit"
SUMMARY = SOURCE / "summaries" / "fresh"
OUT = ROOT / "figures" / "supplementary" / "figure_oracle_profile_credit_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
DASHED = (0, (2.6, 1.8))
DASHDOT = (0, (3.2, 1.4, 0.9, 1.4))
DOTTED = (0, (0.9, 1.5))
LONGDASH = (0, (4.6, 2.0))
FLOOR_DASH = (0, (0.9, 1.5))

# credit rules: (table key, key label, colour, line style, marker)
RULES = (
    ("exact", "exact path", COLORS["bp"], "-", "o"),
    ("broadcast", "root broadcast", COLORS["scalar"], DASHED, "s"),
    ("global_projection", "one oracle profile", COLORS["point_mlp"], DASHDOT, "^"),
    ("subtree_projection", "two subtree profiles", COLORS["oracle"], DOTTED, "D"),
    ("shuffled_projection", "two shuffled profiles", COLORS["additive"], LONGDASH, "X"),
)
RULE = {r[0]: r for r in RULES}
FAMILIES = (("matching", "matching"), ("quartet", "quartic"), ("nested", "nested"))
OPTIMIZERS = (("adam", "Adam"), ("sgd", "SGD"))
STEPS = (0, 1, 16, 64, 256, 1024)
RATES = (0.003, 0.01, 0.03)
N_SEEDS = 20


# ── the study's two bootstrap recipes ────────────────────────────────────
def bootstrap(values):
    """``build_morphology_credit_figure_tables.py:bootstrap`` verbatim."""
    x = np.asarray(values, float)
    rng = np.random.default_rng(202609052)
    means = x[rng.integers(len(x), size=(10000, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def replay_paired_contrasts(end):
    """Replay ``experiment.py:summarize``'s contrast loop with its single
    ``default_rng(127999)`` stream, returning every row it writes."""
    rules = [r[0] for r in RULES]
    rng = np.random.default_rng(127999)
    out = {}
    for family in ("matching", "quartet", "nested", "all"):
        z = end if family == "all" else end[end.family.eq(family)]
        for optimizer in ("sgd", "adam"):
            for structure in ("compatible", "assignment_shuffled"):
                part = z[z.optimizer.eq(optimizer) & z.structure.eq(structure)]
                wide = part.groupby(["seed", "rule"]).test_nmse.mean().unstack()
                for baseline in rules[1:]:
                    values = (wide[baseline] - wide["exact"]).to_numpy()
                    boot = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
                    out[(family, optimizer, structure, baseline + " minus exact")] = (
                        values, float(values.mean()), float(np.quantile(boot, .025)),
                        float(np.quantile(boot, .975)), int(np.sum(values > 0)))
            for rule in rules:
                part = z[z.optimizer.eq(optimizer) & z.rule.eq(rule)]
                wide = part.groupby(["seed", "structure"]).test_nmse.mean().unstack()
                values = (wide.assignment_shuffled - wide.compatible).to_numpy()
                boot = values[rng.integers(len(values), size=(10000, len(values)))].mean(axis=1)
                out[(family, optimizer, "paired", "shuffled minus compatible: " + rule)] = (
                    values, float(values.mean()), float(np.quantile(boot, .025)),
                    float(np.quantile(boot, .975)), int(np.sum(values > 0)))
    return out


def csv(name, base=SUMMARY):
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# ── shared marks ─────────────────────────────────────────────────────────
def fan(ax, x, values, color, *, half=0.14, ms=SEED_MS, alpha=SEED_ALPHA, zorder=2.0):
    """Per-seed values as a jittered fan behind the mean (seed order)."""
    values = np.asarray(values, float)
    jitter = np.linspace(-half, half, len(values)) if len(values) > 1 else np.zeros(1)
    ax.plot(x + jitter, values, linestyle="none", marker="o", markersize=ms,
            markerfacecolor=color, markeredgecolor="none", alpha=alpha, zorder=zorder)


def mean_whisker(ax, x, mean, lo, hi, color, *, marker="o", ms=MARKER_MS, hollow=False,
                 zorder=4.0):
    ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt=marker, ms=ms,
                color=color, markerfacecolor=("white" if hollow else color),
                markeredgecolor=(color if hollow else "white"),
                markeredgewidth=(LW_ERR if hollow else LW_HAIR),
                ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE, capthick=LW_ERR,
                zorder=zorder)


# ── A, B: final test NMSE on compatible trees by family and rule ─────────
def panel_final_nmse(ax, end, cond, optimizer, floors):
    """Every seed of every (family, rule) at the development-selected rate,
    behind the seed-bootstrap mean and 95 % interval; log NMSE axis."""
    z = end[end.structure.eq("compatible") & end.optimizer.eq(optimizer)]
    assert z.step.eq(1024).all() and np.isclose(z.rate, z.selected_rate).all()
    dodge = np.linspace(-0.34, 0.34, len(RULES))
    seed_lo, seed_hi = np.inf, -np.inf
    lines = []
    for fi, (fam, _) in enumerate(FAMILIES):
        # the noise floor of THIS family, over this family's strip only
        ax.plot([fi - 0.46, fi + 0.46], [floors[fam]] * 2, color=MUTE, lw=LW_REF,
                dashes=FLOOR_DASH[1], zorder=1.0, solid_capstyle="butt")
        for dx, (key, _, color, _, marker) in zip(dodge, RULES):
            g = z[z.family.eq(fam) & z.rule.eq(key)].sort_values("seed")
            assert len(g) == N_SEEDS and g.seed.nunique() == N_SEEDS
            vals = g.test_nmse.to_numpy(float)
            m, lo, hi = bootstrap(vals)
            row = cond[cond.family.eq(fam) & cond.structure.eq("compatible")
                       & cond.optimizer.eq(optimizer) & cond.rule.eq(key)]
            assert len(row) == 1 and int(row.n_seeds.iloc[0]) == N_SEEDS
            row = row.iloc[0]
            np.testing.assert_allclose([m, lo, hi], [row["mean"], row.ci95_low, row.ci95_high],
                                       rtol=0, atol=1e-12)
            assert (vals > 0).all()
            seed_lo, seed_hi = min(seed_lo, vals.min()), max(seed_hi, vals.max())
            fan(ax, fi + dx, vals, color, half=0.05, ms=SEED_MS * 0.8)
            mean_whisker(ax, fi + dx, m, lo, hi, color, marker=marker, ms=MARKER_MS * 0.82)
            lines.append(f"{fam}/{key} {m:.4f} [{lo:.4f}, {hi:.4f}]")
    print(f"[{optimizer} A/B] seeds {seed_lo:.4f}-{seed_hi:.4f}; " + "; ".join(lines))
    ax.set_yscale("log")
    ax.set_ylim(0.0185, 2.85)
    assert 0.0185 < seed_lo and seed_hi < 2.85
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
                  ["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.55, len(FAMILIES) - 0.45)
    ax.set_xticks(range(len(FAMILIES)), [lab for _, lab in FAMILIES])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("test NMSE")


# ── C: paired exact-credit cost of reassigning the leaf inputs ───────────
def panel_assignment(ax, replay, contrasts):
    color = RULE["exact"][2]
    x = 0.0
    strip = 0.55
    pos = {}
    printed = []
    for fam, flab in FAMILIES:
        for opt, olab in OPTIMIZERS:
            key = (fam, opt, "paired", "shuffled minus compatible: exact")
            vals, m, lo, hi, npos = replay[key]
            row = contrasts[contrasts.family.eq(fam) & contrasts.optimizer.eq(opt)
                            & contrasts.structure.eq("paired")
                            & contrasts.contrast.eq("shuffled minus compatible: exact")]
            assert len(row) == 1
            row = row.iloc[0]
            np.testing.assert_allclose([m, lo, hi], [row["mean"], row.ci95_low, row.ci95_high],
                                       rtol=0, atol=1e-12)
            assert len(vals) == N_SEEDS == int(row.n_seeds) and npos == int(row.positive_seeds)
            pos[(fam, opt)] = x
            fan(ax, x, vals, color, half=0.16)
            mean_whisker(ax, x, m, lo, hi, color, marker="o")
            # the positive-seed count of every strip, staggered on two lines
            # because a strip is narrower than its own count label
            ax.annotate(f"{npos}/{N_SEEDS} > 0", xy=(x, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(0.0, -1.0 - (8.5 if opt == "sgd" else 0.0)),
                        textcoords="offset points", ha="center", va="top",
                        fontsize=PT_BASE, color=MUTE)
            if fam == "quartet" and opt == "adam":
                # the interval is narrower than the marker: print it under the point
                assert hi - lo < 0.012
                ax.annotate(f"[{lo:.3f}, {hi:.3f}]", xy=(x + 0.2, min(vals)), xycoords="data",
                            xytext=(0.0, -4.0), textcoords="offset points", ha="right",
                            va="top", fontsize=PT_BASE, color=MUTE)
                assert abs(lo - 0.507) < 5e-4 and abs(hi - 0.517) < 5e-4
            printed.append(f"{flab}/{olab} {m:.4f} [{lo:.4f}, {hi:.4f}] {npos}/{N_SEEDS} > 0 "
                           f"seeds {vals.min():.3f}-{vals.max():.3f}")
            x += strip
        x += 0.45
    print("[C] " + "; ".join(printed))
    x0, x1 = -0.38, x - 0.45 - strip + 0.38
    ax.set_xlim(x0, x1)
    ax.plot([x0, x1], [0.0, 0.0], color=MUTE, lw=LW_REF, dashes=(2.6, 2.0), zorder=1.0)
    ax.set_ylim(-0.06, 1.30)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xticks([pos[(f, o)] for f, _ in FAMILIES for o, _ in OPTIMIZERS],
                  [olab for _ in FAMILIES for _, olab in OPTIMIZERS])
    ax.tick_params(axis="x", length=0)
    for fam, flab in FAMILIES:
        xc = np.mean([pos[(fam, o)] for o, _ in OPTIMIZERS])
        ax.annotate(flab, xy=(xc, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0.0, -13.5), textcoords="offset points", ha="center", va="top",
                    fontsize=PT_BASE, color=INK, annotation_clip=False)
    ax.set_ylabel("input-reassigned − compatible\ntest NMSE (exact credit)")


# ── D: gradient alignment at each rule's own state (Adam, compatible) ────
def panel_alignment(ax, traj, grad):
    t = traj[traj.structure.eq("compatible") & traj.optimizer.eq("adam")]
    assert np.isclose(t.rate, t.selected_rate).all()
    ex = t[t.rule.eq("exact")]
    assert len(ex) == N_SEEDS * len(FAMILIES) * len(STEPS)
    np.testing.assert_allclose(ex.gradient_cosine.to_numpy(), 1.0, rtol=0, atol=1e-12)
    for step in STEPS:
        row = grad[grad.structure.eq("compatible") & grad.optimizer.eq("adam")
                   & grad.rule.eq("exact") & grad.step.eq(step)].iloc[0]
        assert row["mean"] == row.ci95_low == row.ci95_high == 1.0
    rules = [r for r in RULES if r[0] != "exact"]
    xs = np.arange(len(STEPS), dtype=float)
    dodge = np.linspace(-0.24, 0.24, len(rules))
    seed_lo, seed_hi = np.inf, -np.inf
    lines = []
    for dx, (key, _, color, ls, marker) in zip(dodge, rules):
        ms_, los, his = [], [], []
        for step in STEPS:
            z = t[t.rule.eq(key) & t.step.eq(step)]
            assert len(z) == N_SEEDS * len(FAMILIES) and z.seed.nunique() == N_SEEDS
            seeds = z.groupby("seed").gradient_cosine.mean().sort_index().to_numpy(float)
            m, lo, hi = bootstrap(seeds)
            row = grad[grad.structure.eq("compatible") & grad.optimizer.eq("adam")
                       & grad.rule.eq(key) & grad.step.eq(step)]
            assert len(row) == 1 and int(row.n_seeds.iloc[0]) == N_SEEDS
            row = row.iloc[0]
            np.testing.assert_allclose([m, lo, hi], [row["mean"], row.ci95_low, row.ci95_high],
                                       rtol=0, atol=1e-12)
            seed_lo, seed_hi = min(seed_lo, seeds.min()), max(seed_hi, seeds.max())
            fan(ax, xs[STEPS.index(step)] + dx, seeds, color, half=0.07, ms=SEED_MS * 0.8)
            ms_.append(m); los.append(lo); his.append(hi)
        ms_ = np.array(ms_)
        ax.plot(xs + dx, ms_, color=color, lw=LW_DATA, ls=ls, zorder=3.0)
        ax.errorbar(xs + dx, ms_, yerr=[ms_ - np.array(los), np.array(his) - ms_], fmt=marker,
                    ms=MARKER_MS * 0.78, color=color, markeredgecolor="white",
                    markeredgewidth=LW_HAIR, ecolor=color, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE * 0.8, capthick=LW_ERR, zorder=3.5, linestyle="none")
        lines.append(f"{key} " + np.array2string(ms_, precision=4))
    print(f"[D] seed-mean range {seed_lo:.4f}-{seed_hi:.4f}; " + "; ".join(lines))
    ax.set_xlim(-0.55, len(STEPS) - 0.45)
    ax.plot([-0.55, len(STEPS) - 0.45], [1.0, 1.0], color=RULE["exact"][2], lw=LW_REF,
            dashes=(2.6, 2.0), zorder=1.0)
    ax.annotate("exact path = 1 by definition", xy=(len(STEPS) - 0.45, 1.0), xycoords="data",
                xytext=(0.0, 2.2), textcoords="offset points", ha="right", va="bottom",
                fontsize=PT_BASE, color=RULE["exact"][2], annotation_clip=False)
    lo_lim = -0.28
    assert lo_lim < seed_lo and seed_hi <= 1.0
    ax.set_ylim(lo_lim, 1.03)
    ax.set_yticks([-0.25, 0.0, 0.25, 0.5, 0.75, 1.0], ["−0.25", "0", "0.25", "0.5", "0.75", "1"])
    ax.set_xticks(xs, [f"{s:,}" for s in STEPS])
    ax.set_xlabel("training update")
    ax.set_ylabel("population-gradient\ncosine")


# ── E, F: every tested learning rate, averaged over assignments and families
def panel_rates(ax, runs, sens, fit, optimizer):
    z = runs[runs.optimizer.eq(optimizer) & runs.step.eq(1024)]
    assert len(z) == N_SEEDS * len(FAMILIES) * 2 * len(RULES) * len(RATES)
    rules = list(RULES)
    dodge = np.linspace(-0.30, 0.30, len(rules))
    seed_lo, seed_hi = np.inf, -np.inf
    lines = []
    for dx, (key, _, color, ls, marker) in zip(dodge, rules):
        selected = float(fit[optimizer][key])
        assert any(np.isclose(selected, r) for r in RATES)
        means = []
        for ri, rate in enumerate(RATES):
            g = z[z.rule.eq(key) & np.isclose(z.rate, rate)]
            assert len(g) == N_SEEDS * len(FAMILIES) * 2 and g.seed.nunique() == N_SEEDS
            assert g.groupby("seed").size().eq(len(FAMILIES) * 2).all()
            seeds = g.groupby("seed").test_nmse.mean().sort_index().to_numpy(float)
            m, lo, hi = bootstrap(seeds)
            # the frozen sensitivity table holds the six (family x structure)
            # means; their average is the same 120-fit mean
            s = sens[sens.optimizer.eq(optimizer) & sens.rule.eq(key) & np.isclose(sens.rate, rate)]
            assert len(s) == len(FAMILIES) * 2
            np.testing.assert_allclose(m, s.test_nmse.mean(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(m, g.test_nmse.mean(), rtol=0, atol=1e-12)
            seed_lo, seed_hi = min(seed_lo, seeds.min()), max(seed_hi, seeds.max())
            x = ri + dx
            fan(ax, x, seeds, color, half=0.06, ms=SEED_MS * 0.8)
            chosen = bool(np.isclose(rate, selected))
            mean_whisker(ax, x, m, lo, hi, color, marker=marker, ms=MARKER_MS * 0.82,
                         hollow=not chosen, zorder=4.5 if chosen else 4.0)
            means.append(m)
            lines.append(f"{key}@{rate:g}{'*' if chosen else ''} {m:.4f} [{lo:.4f}, {hi:.4f}]")
        ax.plot(np.arange(len(RATES)) + dx, means, color=color, lw=LW_REF, ls=ls, zorder=3.0,
                alpha=0.75)
    print(f"[{optimizer} E/F] seeds {seed_lo:.4f}-{seed_hi:.4f}; " + "; ".join(lines))
    ax.set_yscale("log")
    ax.set_ylim(0.25, 1.95)
    assert 0.25 < seed_lo and seed_hi < 1.95
    ax.set_yticks([0.3, 0.5, 0.7, 1.0, 1.5], ["0.3", "0.5", "0.7", "1", "1.5"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(-0.55, len(RATES) - 0.45)
    ax.set_xticks(range(len(RATES)), [f"{r:g}" for r in RATES])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("mean test NMSE")


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 490.0                 # the canvas audit caps the aspect at 1.05
ROW_PT = [120.0, 108.0, 108.0]
HGUTTER_PT = 30.0
VGUTTER_PT = 42.0
MARGINS = Margins(left=40.0, right=6.0, top=18.0, bottom=52.0)


def build(path: Path = OUT):
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    assert list(protocol["checkpoints"]) == list(STEPS) and protocol["steps"] == 1024
    assert list(protocol["rates"]) == list(RATES) and list(protocol["rules"]) == [r[0] for r in RULES]
    assert len(protocol["fresh_seeds"]) == N_SEEDS
    fit = json.loads((SOURCE / "development_fit.json").read_text())
    validation = json.loads((SOURCE / "validation.json").read_text())
    assert validation["fresh_selections_match_frozen_development"] is True
    end = csv("selected_endpoints.csv")
    traj = csv("selected_trajectories.csv")
    cond = csv("figure_condition_summary.csv", SOURCE)
    grad = csv("figure_gradient_summary.csv", SOURCE)
    contrasts = csv("paired_contrasts.csv")
    sens = csv("all_rate_sensitivity.csv")
    runs = pd.concat([pd.read_csv(SOURCE / "runs" / "fresh" / f"seed_{s}.csv")
                      for s in protocol["fresh_seeds"]], ignore_index=True)
    assert len(runs) == validation["checkpoint_rows"] == 21600
    assert len(end) == validation["selected_rate_fits"] == 1200 and end.seed.nunique() == N_SEEDS
    assert len(traj) == 7200 and len(cond) == 60 and len(grad) == 120 and len(sens) == 180
    # the frozen selected rates are the development choices, condition by condition
    for (o, r), g in end.groupby(["optimizer", "rule"]):
        assert np.isclose(g.selected_rate, fit[o][r]).all() and np.isclose(g.rate, fit[o][r]).all()
    # noise floors: test noise SD^2 over the family's known target variance
    sd = float(protocol["test_noise_sd"])
    floors = {}
    for fam, _ in FAMILIES:
        var = runs[runs.family.eq(fam)].variance.unique()
        assert len(var) == 1
        floors[fam] = sd ** 2 / float(var[0])
    np.testing.assert_allclose([floors[f] for f, _ in FAMILIES], [0.0225, 0.045, 0.0225],
                               rtol=0, atol=1e-15)
    replay = replay_paired_contrasts(end)
    # the replay reproduces EVERY row of the frozen contrasts table
    err = 0.0
    for _, r in contrasts.iterrows():
        vals, m, lo, hi, npos = replay[(r.family, r.optimizer, r.structure, r.contrast)]
        err = max(err, abs(m - r["mean"]), abs(lo - r.ci95_low), abs(hi - r.ci95_high))
        assert len(vals) == int(r.n_seeds) == N_SEEDS and npos == int(r.positive_seeds)
    assert err < 1e-12, err
    print(f"[tables] paired_contrasts replayed from the endpoints, {len(contrasts)} rows, "
          f"max |error| {err:.2e}; floors {floors}")

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 3, row_weights=ROW_PT, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS, letter_clearance=True)
    ax_a = cv.panel("A", 0, 0, 6, grid="y", title="Adam, compatible trees: final test error")
    ax_b = cv.panel("B", 0, 6, 6, grid="y", title="SGD, compatible trees: final test error")
    ax_c = cv.panel("C", 1, 0, 6, grid="y", title="Exact credit: input assignment matters")
    ax_d = cv.panel("D", 1, 6, 6, grid="y", title="Adam, compatible trees: gradient alignment")
    ax_e = cv.panel("E", 2, 0, 6, grid="y", title="Adam: all three learning rates")
    ax_f = cv.panel("F", 2, 6, 6, grid="y", title="SGD: all three learning rates")
    # one declared reserve on every panel, wide enough to cover the widest
    # measured y decoration (D's two-line label + '−0.25'), so the column lock
    # gives both module columns one axes width by construction
    for name in "ABCDEF":
        cv.declare_reserve(name, left=22.0, right=8.0)
    cv.declare_reserve("C", bottom=14.0)      # the family names under the optimizer ticks
    ax_d.set_title(ax_d.get_title(), fontsize=ax_d.title.get_fontsize(), color=INK, pad=11.0,
                   fontweight="normal")      # room for the reference label above the axis

    panel_final_nmse(ax_a, end, cond, "adam", floors)
    panel_final_nmse(ax_b, end, cond, "sgd", floors)
    panel_assignment(ax_c, replay, contrasts)
    panel_alignment(ax_d, traj, grad)
    panel_rates(ax_e, runs, sens, fit, "adam")
    panel_rates(ax_f, runs, sens, fit, "sgd")
    ax_f.sharey(ax_e)

    # one shared key: the five rules, the floor rule and the fill meaning
    handles = [Line2D([], [], color=c, lw=LW_DATA, ls=ls, marker=mk, ms=MARKER_MS * 0.78,
                      markeredgecolor="white", markeredgewidth=LW_HAIR, label=lab)
               for _, lab, c, ls, mk in RULES]
    handles.append(Line2D([], [], color=MUTE, lw=LW_REF, dashes=FLOOR_DASH[1],
                          label="label-noise floor (A,B)"))
    handles.append(Line2D([], [], color=INK, lw=0, marker="o", ms=MARKER_MS * 0.78,
                          markerfacecolor=INK, markeredgecolor="white", markeredgewidth=LW_HAIR,
                          label="development-selected rate (A–F)"))
    handles.append(Line2D([], [], color=INK, lw=0, marker="o", ms=MARKER_MS * 0.78,
                          markerfacecolor="white", markeredgecolor=INK, markeredgewidth=LW_ERR,
                          label="other tested rate (E,F)"))
    cv.fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0),
                  ncol=4, frameon=False, fontsize=PT_BASE, handlelength=2.6,
                  columnspacing=1.5, handletextpad=0.6, borderaxespad=0.4, labelspacing=0.45)
    problems = cv.save(path, name="figure_oracle_profile_credit_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

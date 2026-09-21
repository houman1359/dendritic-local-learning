#!/usr/bin/env python3
"""Supplementary sheet S16 (ident ``fixed_profile_budget``) -- fixed profiles
and exact credit under longer matched training budgets -- rebuilt as ONE
native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/fixed_profile_budget.pdf``)
is a paste of an upstream render with no generator in this repository; this
builder reads ONLY the frozen tables under ``source_data/credit_rule_extension``
(plus the noise-floor table of ``source_data/credit_rule_bridge``) and redraws
the same six lettered panels with the same plotted quantities.  Nothing about
the numbers changes: every mean and every 95 % interval is recomputed from the
per-seed rows with the study's own bootstrap (protocol_freeze.json:
``default_rng(210999)``, 10,000 whole-seed draws, percentile 2.5 / 97.5) and
asserted against ``figures/figure_source.csv`` and the summary tables before
it is drawn.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
frozen_S16.json and the "cannot" list of frozen_drafts.json), panel by panel:

* A, B  pairwise test NMSE at the selected rates and at the common rate: the
        64 measured checkpoints are drawn as dots on every curve; every
        per-seed trajectory is a hairline fan behind its mean and band; the
        1,024-update cap and the noise-only NMSE are thin solid grey
        reference rules (the dotted texture now means initial sign only).
* C     the two quartic rate views merged into one panel: exact path and unit
        broadcast are the same runs under both rate choices (asserted
        bitwise) and are drawn once; the initial-profile and initial-sign
        controls are drawn twice, thick at the rates selected per rule and
        thin at the common rate.
* D     the freed quadrant shows the terminal state that A and B cannot
        resolve: all twenty pairwise seeds at 16,384 updates for every
        distinct condition, on a linear axis with the noise floor.
* E     rate view recoloured onto a violet / green pair that no other panel
        uses (dark red and blue mean credit rule everywhere else); the four
        translucent bands are replaced by dodged capped 95 % whiskers; all
        four budgets are ticked; the zero line is labelled; the seed count
        is printed with its denominator.
* F     the quartic seed scatter keeps its diagonal and floor guides, adds rug
        ticks for every seed on both axes and prints n = 20 per rule; the
        companion zoom subpanel (letter-less, same row) resolves the sixteen
        exact seeds that overplot at the floor, as open circles at 20x.

Colour meanings on this sheet (one meaning per hue): dark red (``bp``) exact
path, amber (``local``) unit broadcast, blue (``additive``) initial profile,
ink dotted initial sign -- in A-D and F; violet (``oracle``) rates selected
per rule and green (``shunting``) common rate -- in E only; mute grey thin
solid lines are reference rules everywhere (1,024-update cap, noise-only
NMSE, unchanged-error diagonal, zero deficit).  Dashed means validation-
selected (E only); thin coloured curves mean the common rate (C only).
"""
from __future__ import annotations

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
    COLORS, ERR_CAPSIZE, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR, LW_REF, MARKER_MS,
    PT_BASE, PT_EMPH, SEED_ALPHA, SEED_MS, Margins, NativeCanvas)

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "credit_rule_extension"
BRIDGE = ROOT / "source_data" / "credit_rule_bridge"
OUT = ROOT / "figures" / "supplementary" / "figure_fixed_profile_budget_native.pdf"

# ── the study's own bootstrap (protocol_freeze.json "uncertainty") ────────
BOOT_SEED = 210999
BOOT_DRAWS = 10000
N_SEEDS = 20
BUDGETS = [1024, 4096, 8192, 16384]
CAP = 1024                      # the historical update cap
COMMON_RATE = 0.003

# credit rules: (table key, key label, colour, line style)
RULES = [
    ("exact", "exact path", COLORS["bp"], "-"),
    ("unit_broadcast", "unit broadcast", COLORS["local"], "-"),
    ("calibrated_broadcast", "initial profile", COLORS["additive"], "-"),
    ("sign_broadcast", "initial sign", COLORS["ink"], ":"),
]
RULE_COLOR = {k: c for k, _, c, _ in RULES}
RULE_LABEL = {k: lab for k, lab, _, _ in RULES}
SELECTED = COLORS["oracle"]     # E: rates selected per rule
COMMON = COLORS["shunting"]     # E: the common rate
MUTE = COLORS["mute"]
INK = COLORS["ink"]
EDGE = COLORS["edge"]

DOTTED = (1.0, 1.8)             # initial sign
DASHED = (3.2, 1.8)             # validation-selected (E)
BAND_ALPHA = 0.16               # 95 % interval fill
FAN_ALPHA = 0.16                # per-seed hairline
DOT_MS = 1.7                    # checkpoint dot on a mean curve
LW_THIN = LW_HAIR               # common-rate curves in C


def csv(base, name):
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def boot_index():
    return np.random.default_rng(BOOT_SEED).integers(0, N_SEEDS, (BOOT_DRAWS, N_SEEDS))


def boot_interval(values, idx):
    values = np.asarray(values, float)
    assert values.shape[-1] == N_SEEDS
    draws = values[..., idx].mean(axis=-1)          # (..., BOOT_DRAWS)
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=-1)
    return values.mean(axis=-1), lo, hi


# ── shared glyphs ─────────────────────────────────────────────────────────
def reference(ax, *, y=None, x=None, zorder=1.1):
    """A thin solid mute rule between the fixed limits (call after set_*lim)."""
    kw = dict(color=MUTE, lw=LW_REF, zorder=zorder, solid_capstyle="butt")
    if y is not None:
        x0, x1 = ax.get_xlim()
        ax.plot([x0, x1], [y, y], **kw)
    if x is not None:
        y0, y1 = ax.get_ylim()
        ax.plot([x, x], [y0, y1], **kw)


def fan(ax, x, seeds, color, *, half=0.20):
    seeds = np.asarray(seeds, float)
    jit = np.linspace(-half, half, len(seeds))
    ax.plot(x + jit, seeds, linestyle="none", marker="o", markersize=SEED_MS,
            markerfacecolor=color, markeredgecolor="none", alpha=SEED_ALPHA,
            zorder=2.0)


def mean_rule(ax, x, mean, lo, hi, color, *, half_pt=4.5):
    ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                capthick=LW_ERR, zorder=3.5)
    for sign in (-1.0, 1.0):
        ax.annotate("", xy=(x, mean), xycoords="data", xytext=(sign * half_pt, 0.0),
                    textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=color, lw=LW_DATA,
                                    shrinkA=0, shrinkB=0), zorder=4.0)


# ── A-C: mean curves with checkpoint dots, bands and seed fans ────────────
class Curves:
    """Per-seed trajectory matrices (20 x 64) for Adam at steps >= 64."""

    def __init__(self, curves, protocol_steps):
        c = curves[curves.optimizer.eq("adam") & curves.step.ge(64)]
        self.steps = [int(s) for s in protocol_steps if s >= 64]
        assert len(self.steps) == 64 and self.steps[:5] == [64, 256, 512, 1024, 1280]
        assert self.steps[4:] == list(range(1280, 16385, 256))
        self.seeds = sorted(c.seed.unique())
        assert len(self.seeds) == N_SEEDS
        self.c = c

    def matrix(self, task, rule, view):
        z = self.c[self.c.task.eq(task) & self.c.rule.eq(rule) & self.c[view]]
        assert z.rate.nunique() == 1
        m = z.pivot(index="seed", columns="step", values="test_nmse")
        m = m.loc[self.seeds, self.steps]
        assert m.shape == (N_SEEDS, 64) and np.isfinite(m.to_numpy()).all()
        return m.to_numpy(float), float(z.rate.iloc[0])


def frozen_curve(fs, panel, rule):
    z = fs[fs.panel.eq(panel) & fs.rule.eq(rule)].sort_values("step")
    assert len(z) == 64
    return z


def draw_curve(ax, steps, mat, frozen, color, *, lw=LW_DATA, ls="-",
               fan_lines=True, zorder=3.0, idx):
    """One rule: recompute mean and interval, assert, draw fan + band + mean."""
    mean, lo, hi = boot_interval(mat.T, idx)
    np.testing.assert_allclose(mean, frozen["mean"].to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(lo, frozen.ci95_low.to_numpy(), rtol=0, atol=1e-9)
    np.testing.assert_allclose(hi, frozen.ci95_high.to_numpy(), rtol=0, atol=1e-9)
    assert list(frozen.step.astype(int)) == list(steps)
    x = np.asarray(steps, float)
    if fan_lines:
        for row in mat:
            ax.plot(x, row, color=color, lw=LW_HAIR, alpha=FAN_ALPHA,
                    zorder=zorder - 1.5, solid_capstyle="butt")
    ax.fill_between(x, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=zorder - 1.0)
    kw = dict(color=color, lw=lw, zorder=zorder, marker="o", ms=DOT_MS,
              markeredgecolor="none", markerfacecolor=color)
    if ls == ":":
        kw["dashes"] = DOTTED
    ax.plot(x, mean, **kw)
    return mean, lo, hi


def style_step_axis(ax, ylim, floor):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(64 / 1.12, 16384 * 1.12)
    ax.set_ylim(*ylim)
    ax.set_xticks([64, 256, 1024, 4096, 16384], ["64", "256", "1,024", "4,096", "16,384"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2] if ylim[0] < 0.03 else [0.05, 0.1, 0.2, 0.5, 1, 2],
                  ["0.02", "0.05", "0.1", "0.2", "0.5", "1", "2"] if ylim[0] < 0.03
                  else ["0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("training update")
    ax.set_ylabel("test NMSE")
    reference(ax, x=CAP)
    reference(ax, y=floor)


def panel_pairwise(ax, view, fs_panel, cv_curves, fs, idx, floor):
    tails = {}
    for key, _, color, ls in RULES:
        mat, rate = cv_curves.matrix("matching", key, view)
        frozen = frozen_curve(fs, fs_panel, key)
        assert set(frozen.rate_view) == {view}
        # the sign control is the profile control at the same rate: bitwise
        # to 1e-6 per seed, so its fan would be the blue fan drawn twice
        fan_lines = key != "sign_broadcast"
        mean, lo, hi = draw_curve(ax, cv_curves.steps, mat, frozen, color, ls=ls,
                                  fan_lines=fan_lines, idx=idx,
                                  zorder=3.4 if key == "sign_broadcast" else 3.0)
        tails[key] = (mean[-1], rate, mat)
    prof, _ = cv_curves.matrix("matching", "calibrated_broadcast", view)
    sign, _ = cv_curves.matrix("matching", "sign_broadcast", view)
    np.testing.assert_allclose(prof, sign, rtol=0, atol=2e-6)
    lo_all = min(m[2].min() for m in tails.values())
    hi_all = max(m[2].max() for m in tails.values())
    assert 0.0195 < lo_all and hi_all < 2.3, (lo_all, hi_all)
    style_step_axis(ax, (0.0195, 2.3), floor)
    print(f"[{fs_panel}] pairwise {view}: terminal means "
          + ", ".join(f"{RULE_LABEL[k]} {v[0]:.5f} (rate {v[1]:g})" for k, v in tails.items())
          + f"; seeds span {lo_all:.4f}-{hi_all:.4f}")
    return tails


def panel_quartic(ax, cv_curves, fs, idx, floor):
    # exact and unit broadcast: the same runs under both rate views
    for key in ("exact", "unit_broadcast"):
        sel, r_sel = cv_curves.matrix("quartet", key, "selected_rate")
        com, r_com = cv_curves.matrix("quartet", key, "common_rate")
        assert r_sel == r_com == COMMON_RATE
        np.testing.assert_array_equal(sel, com)
        fc, fd = frozen_curve(fs, "C", key), frozen_curve(fs, "D", key)
        for col in ("mean", "ci95_low", "ci95_high"):
            np.testing.assert_array_equal(fc[col].to_numpy(), fd[col].to_numpy())
        draw_curve(ax, cv_curves.steps, sel, fc, RULE_COLOR[key], idx=idx,
                   zorder=3.0 if key == "exact" else 2.9)
    # profile and sign: thick at the selected rate (0.01), thin at the common
    # rate (0.003); at the common rate the two coincide per seed
    for key in ("calibrated_broadcast", "sign_broadcast"):
        ls = ":" if key == "sign_broadcast" else "-"
        sel, r_sel = cv_curves.matrix("quartet", key, "selected_rate")
        com, r_com = cv_curves.matrix("quartet", key, "common_rate")
        assert r_sel == 0.01 and r_com == COMMON_RATE
        # the sign fan is not drawn: at the common rate it is the profile fan
        # (asserted below); at the selected rate its 20 hairlines would sit
        # in the same 0.34-1.62 band as the profile fan and only darken it
        draw_curve(ax, cv_curves.steps, sel, frozen_curve(fs, "C", key), RULE_COLOR[key],
                   ls=ls, fan_lines=key != "sign_broadcast", idx=idx,
                   zorder=3.3 if key == "sign_broadcast" else 3.1)
        if key == "sign_broadcast":
            assert 0.34 < sel.min() and sel.max() < 1.62, (sel.min(), sel.max())
        draw_curve(ax, cv_curves.steps, com, frozen_curve(fs, "D", key), RULE_COLOR[key],
                   lw=LW_THIN, ls=ls, fan_lines=key != "sign_broadcast", idx=idx,
                   zorder=3.25 if key == "sign_broadcast" else 3.05)
    prof_c, _ = cv_curves.matrix("quartet", "calibrated_broadcast", "common_rate")
    sign_c, _ = cv_curves.matrix("quartet", "sign_broadcast", "common_rate")
    np.testing.assert_allclose(prof_c, sign_c, rtol=0, atol=1e-5)
    gap = np.abs(frozen_curve(fs, "C", "calibrated_broadcast")["mean"].to_numpy()
                 - frozen_curve(fs, "C", "sign_broadcast")["mean"].to_numpy()).max()
    assert gap < 0.03, gap
    lo_all = min(cv_curves.matrix("quartet", k, v)[0].min()
                 for k in RULE_COLOR for v in ("selected_rate", "common_rate"))
    hi_all = max(cv_curves.matrix("quartet", k, v)[0].max()
                 for k in RULE_COLOR for v in ("selected_rate", "common_rate"))
    assert 0.040 < lo_all and hi_all < 3.5, (lo_all, hi_all)
    style_step_axis(ax, (0.040, 3.5), floor)
    # the thick / thin rate-view key lives in the shared key under the sheet:
    # every band of this panel holds a seed hairline somewhere
    print(f"[C] quartic both rate views: profile/sign means differ by at most {gap:.4f} "
          f"at the selected rate; seeds span {lo_all:.4f}-{hi_all:.4f}")


# ── D: pairwise terminal state, every seed ────────────────────────────────
D_COLS = [("exact", "selected_rate", "exact\npath", COLORS["bp"]),
          ("unit_broadcast", "selected_rate", "unit\nbroadcast", COLORS["local"]),
          ("calibrated_broadcast", "selected_rate", "profile / sign\nrate 0.01", COLORS["additive"]),
          ("calibrated_broadcast", "common_rate", "profile / sign\nrate 0.003", COLORS["additive"])]


def panel_terminal(ax, outcomes, cond, idx, floor):
    t = outcomes[outcomes.budget.eq(16384) & outcomes.endpoint.eq("terminal")
                 & outcomes.task.eq("matching") & outcomes.optimizer.eq("adam")]
    c = cond[cond.budget.eq(16384) & cond.endpoint.eq("terminal") & cond.task.eq("matching")
             & cond.optimizer.eq("adam") & cond.metric.eq("test_nmse")]
    ymin, ymax = np.inf, -np.inf
    for x, (rule, view, label, color) in enumerate(D_COLS):
        z = t[t.rule.eq(rule) & t[view]].sort_values("seed")
        v = z.test_nmse.to_numpy(float)
        assert len(v) == N_SEEDS and z.seed.is_unique and z.rate.nunique() == 1
        mean, lo, hi = boot_interval(v, idx)
        r = c[c.rule.eq(rule) & c.rate_view.eq(view)].iloc[0]
        assert int(r.n_seeds) == N_SEEDS
        np.testing.assert_allclose([mean, lo, hi, v.min(), v.max()],
                                   [r["mean"], r.ci95_low, r.ci95_high, r.minimum, r.maximum],
                                   rtol=0, atol=1e-12)
        assert float(z.rate.iloc[0]) == float(r.rate)
        assert label.endswith(f"{float(r.rate):g}") or rule in ("exact", "unit_broadcast")
        if rule == "calibrated_broadcast":
            s = t[t.rule.eq("sign_broadcast") & t[view]].sort_values("seed").test_nmse.to_numpy()
            np.testing.assert_allclose(v, s, rtol=0, atol=2e-6)
        # exact and unit broadcast: identical runs under both views
        if rule in ("exact", "unit_broadcast"):
            o = t[t.rule.eq(rule) & t.common_rate].sort_values("seed").test_nmse.to_numpy()
            np.testing.assert_array_equal(v, o)
        fan(ax, x, v, color)
        mean_rule(ax, x, mean, lo, hi, color)
        ymin, ymax = min(ymin, v.min()), max(ymax, v.max())
        print(f"[D] {label.replace(chr(10), ' ')}: mean {mean:.5f} [{lo:.5f}, {hi:.5f}] "
              f"seeds {v.min():.5f}-{v.max():.5f}")
    ax.set_xlim(-0.55, len(D_COLS) - 0.45)
    assert 0.0215 < ymin and ymax < 0.0275, (ymin, ymax)
    ax.set_ylim(0.0215, 0.0275)
    reference(ax, y=floor)
    ax.set_xticks(range(len(D_COLS)), [c[2] for c in D_COLS])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0.022, 0.024, 0.026], ["0.022", "0.024", "0.026"])
    ax.set_ylabel("test NMSE at 16,384 updates")
    # the floor value cannot be printed on the rule (seeds straddle it in every
    # column); the shared key names the rule and the caption gives 0.0225
    ax.text(0.0, 1.0, f"n = {N_SEEDS} seeds per condition", transform=ax.transAxes,
            ha="left", va="top", fontsize=PT_BASE, color=INK, zorder=6)


# ── E: task-by-credit contrast over budget ────────────────────────────────
E_SERIES = [("selected_rate", "terminal"), ("selected_rate", "validation_selected"),
            ("common_rate", "terminal"), ("common_rate", "validation_selected")]
E_LABEL = {"selected_rate": "selected rates", "common_rate": "common rate"}
E_STATE = {"terminal": "terminal", "validation_selected": "validation"}


def panel_contrast(ax, fs, pairs, contrasts, idx):
    E = fs[fs.panel.eq("E")]
    assert len(E) == 16 and set(E.contrast) == {"calibrated_broadcast minus exact interaction"}
    dodge = np.linspace(-0.13, 0.13, len(E_SERIES))
    band_lo, band_hi = np.inf, -np.inf
    for dx, (view, state) in zip(dodge, E_SERIES):
        color = SELECTED if view == "selected_rate" else COMMON
        marker = "o" if state == "terminal" else "s"
        means, los, his = [], [], []
        for budget in BUDGETS:
            r = E[E.rate_view.eq(view) & E.endpoint.eq(state) & E.budget.eq(budget)].iloc[0]
            z = pairs[pairs.task.eq("quartet_minus_matching") & pairs.optimizer.eq("adam")
                      & pairs.budget.eq(budget) & pairs.endpoint.eq(state)
                      & pairs.rate_view.eq(view) & pairs.metric.eq("test_nmse")
                      & pairs.contrast.eq("calibrated_broadcast minus exact interaction")]
            d = z.sort_values("seed").difference.to_numpy(float)
            assert len(d) == N_SEEDS and z.seed.is_unique
            mean, lo, hi = boot_interval(d, idx)
            np.testing.assert_allclose([mean, lo, hi, np.median(d)],
                                       [r["mean"], r.ci95_low, r.ci95_high, r["median"]],
                                       rtol=0, atol=1e-12)
            assert int(r.n_seeds) == N_SEEDS and int(r.positive_seeds) == N_SEEDS
            assert int((d > 0).sum()) == N_SEEDS
            s = contrasts[contrasts.task.eq("quartet_minus_matching") & contrasts.optimizer.eq("adam")
                          & contrasts.budget.eq(budget) & contrasts.endpoint.eq(state)
                          & contrasts.rate_view.eq(view) & contrasts.metric.eq("test_nmse")
                          & contrasts.contrast.eq("calibrated_broadcast minus exact interaction")].iloc[0]
            np.testing.assert_allclose([s["mean"], s.ci95_low, s.ci95_high], [mean, lo, hi],
                                       rtol=0, atol=1e-12)
            means.append(mean); los.append(lo); his.append(hi)
        means, los, his = map(np.asarray, (means, los, his))
        band_lo, band_hi = min(band_lo, los.min()), max(band_hi, his.max())
        x = np.array(BUDGETS, float) * 2.0 ** dx
        kw = dict(color=color, lw=LW_DATA, zorder=3.0)
        if state == "validation_selected":
            kw["dashes"] = DASHED
        ax.plot(x, means, **kw)
        ax.errorbar(x, means, yerr=[means - los, his - means], fmt=marker, ms=MARKER_MS * 0.7,
                    color=color, markeredgecolor="white", markeredgewidth=LW_HAIR, ecolor=color,
                    elinewidth=LW_ERR, capsize=ERR_CAPSIZE * 0.8, capthick=LW_ERR, zorder=3.5,
                    linestyle="none")
        print(f"[E] {view}/{state}: " + ", ".join(
            f"{b}: {m:.4f} [{lo:.4f}, {hi:.4f}]" for b, m, lo, hi in zip(BUDGETS, means, los, his)))
    assert 0.64 < band_lo and band_hi < 1.02, (band_lo, band_hi)
    ax.set_xscale("log", base=2)
    ax.set_xlim(1024 / 1.45, 16384 * 1.45)
    ax.set_ylim(0.0, 1.06)
    ax.set_xticks(BUDGETS, ["1,024", "4,096", "8,192", "16,384"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    ax.set_xlabel("maximum training updates")
    ax.set_ylabel("quartic − pairwise credit deficit")
    reference(ax, y=0.0)
    ax.annotate("zero deficit", xy=(1.0, 0.0), xycoords=("axes fraction", "data"),
                xytext=(-1.5, 2.2), textcoords="offset points", ha="right", va="bottom",
                fontsize=PT_BASE, color=MUTE)
    ax.text(0.02, 0.60, f"positive in {N_SEEDS}/{N_SEEDS} seeds\nat every budget",
            transform=ax.transAxes, ha="left", va="top", fontsize=PT_BASE, color=INK,
            linespacing=1.1, zorder=6)
    handles = []
    for view, state in E_SERIES:
        color = SELECTED if view == "selected_rate" else COMMON
        handles.append(Line2D([], [], color=color, lw=LW_DATA,
                              dashes=DASHED if state == "validation_selected" else (None, None),
                              marker="o" if state == "terminal" else "s", ms=MARKER_MS * 0.7,
                              markeredgecolor="white", markeredgewidth=LW_HAIR,
                              label=f"{E_LABEL[view]}, {E_STATE[state]}"))
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.07), frameon=False,
              fontsize=PT_BASE, handlelength=2.0, handletextpad=0.5, borderaxespad=0.0,
              labelspacing=0.3)


# ── F: quartic seeds at 1,024 and 16,384 updates ──────────────────────────
F_LIM = (0.036, 1.45)
ZOOM = (0.0432, 0.0504)          # the floor cluster window (both axes)


def seed_table(fs, outcomes):
    F = fs[fs.panel.eq("F")]
    out = {}
    for rule in ("exact", "calibrated_broadcast"):
        z = F[F.rule.eq(rule)].sort_values("seed")
        assert len(z) == N_SEEDS and z.seed.is_unique
        for budget, col in ((1024, "test_nmse1024"), (16384, "test_nmse16384")):
            o = outcomes[outcomes.budget.eq(budget) & outcomes.endpoint.eq("terminal")
                         & outcomes.task.eq("quartet") & outcomes.optimizer.eq("adam")
                         & outcomes.rule.eq(rule) & outcomes.selected_rate].sort_values("seed")
            assert list(o.seed) == [int(s) for s in z.seed]
            np.testing.assert_array_equal(o.test_nmse.to_numpy(), z[col].to_numpy())
        out[rule] = (z.test_nmse1024.to_numpy(float), z.test_nmse16384.to_numpy(float))
    return out


def panel_seeds(ax, seeds, floor, dist):
    for rule, (x, y) in seeds.items():
        color = RULE_COLOR[rule]
        assert F_LIM[0] < min(x.min(), y.min()) and max(x.max(), y.max()) < F_LIM[1]
        ax.plot(x, y, linestyle="none", marker="o", ms=SEED_MS, markerfacecolor=color,
                markeredgecolor="none", alpha=SEED_ALPHA, zorder=3.0)
        # rug: one tick per seed on each axis, in the rule's colour
        for xi, yi in zip(x, y):
            ax.plot([xi, xi], [0.0, 0.035], transform=ax.get_xaxis_transform(),
                    color=color, lw=LW_HAIR, alpha=SEED_ALPHA, zorder=2.5, solid_capstyle="butt")
            ax.plot([0.0, 0.035], [yi, yi], transform=ax.get_yaxis_transform(),
                    color=color, lw=LW_HAIR, alpha=SEED_ALPHA, zorder=2.5, solid_capstyle="butt")
    ex, ey = seeds["exact"]
    px, py = seeds["calibrated_broadcast"]
    stalled = int((ex > 0.5).sum())
    assert stalled == 3 and int((ey < 0.05).sum()) == N_SEEDS
    near_1024 = int(dist[dist.budget.eq(1024) & dist.endpoint.eq("terminal") & dist.optimizer.eq("adam")
                         & dist.rule.eq("exact")].near_floor_count_0_065.iloc[0])
    near_16384 = int(dist[dist.budget.eq(16384) & dist.endpoint.eq("terminal") & dist.optimizer.eq("adam")
                          & dist.rule.eq("exact")].near_floor_count_0_065.iloc[0])
    assert near_1024 == N_SEEDS - stalled == 17 and near_16384 == N_SEEDS
    assert py.min() > 0.6 and px.min() > 0.7
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*F_LIM)
    ax.set_ylim(*F_LIM)
    ticks = [0.05, 0.1, 0.2, 0.5, 1.0]
    labels = ["0.05", "0.1", "0.2", "0.5", "1"]
    ax.set_xticks(ticks, labels)
    ax.set_yticks(ticks, labels)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("test NMSE at 1,024 updates")
    ax.set_ylabel("test NMSE at 16,384 updates")
    ax.plot(F_LIM, F_LIM, color=MUTE, lw=LW_REF, zorder=1.1, solid_capstyle="butt")
    reference(ax, x=floor)
    reference(ax, y=floor)
    # texts sit where no seed, rug tick or rule runs: the top-left corner
    # right of the vertical floor guide (x > 0.05) and above every profile
    # seed's x (0.72), and the strip under the diagonal between the exact
    # cluster and the three stalled seeds (0.2 < x < 0.5, 0.06 < y < 0.13)
    ax.text(0.10, 0.97, f"n = {N_SEEDS} seeds per rule", transform=ax.transAxes,
            ha="left", va="top", fontsize=PT_BASE, color=INK, zorder=6)
    ax.text(0.2, 0.128, f"{stalled}/{N_SEEDS} exact seeds\nstalled at 1,024", ha="left", va="top",
            fontsize=PT_BASE, color=RULE_COLOR["exact"], linespacing=1.1, zorder=6)
    ax.text(0.062, 0.56, "unchanged error", ha="left", va="center", fontsize=PT_BASE,
            color=MUTE, zorder=6)
    ax.annotate("", xy=(0.31, 0.31), xycoords="data", xytext=(0.17, 0.46), textcoords="data",
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=LW_HAIR, shrinkA=0, shrinkB=0),
                zorder=5)
    print(f"[F] exact seeds at 1,024: {ex.min():.4f}-{ex.max():.4f} ({stalled} stalled above 0.5), "
          f"at 16,384: {ey.min():.4f}-{ey.max():.4f}; profile seeds {px.min():.3f}-{px.max():.3f} "
          f"and {py.min():.3f}-{py.max():.3f}")
    return stalled


def panel_zoom(ax, seeds, floor):
    ex, ey = seeds["exact"]
    inside = (ex > ZOOM[0]) & (ex < ZOOM[1]) & (ey > ZOOM[0]) & (ey < ZOOM[1])
    n_in = int(inside.sum())
    assert n_in == 16
    px, py = seeds["calibrated_broadcast"]
    assert not ((px < ZOOM[1]) | (py < ZOOM[1])).any()
    # the four exact seeds outside the window: 3 stalled at 1,024 and one at 0.059
    out_x = np.sort(ex[~inside])
    assert out_x[0] > 0.055 and (out_x[1:] > 0.5).all()
    ax.plot(ex[inside], ey[inside], linestyle="none", marker="o", ms=MARKER_MS * 0.8,
            markerfacecolor="none", markeredgecolor=RULE_COLOR["exact"],
            markeredgewidth=LW_ERR, zorder=3.0)
    ax.set_xlim(*ZOOM)
    ax.set_ylim(*ZOOM)
    ax.plot(ZOOM, ZOOM, color=MUTE, lw=LW_REF, zorder=1.1, solid_capstyle="butt")
    reference(ax, x=floor)
    reference(ax, y=floor)
    ticks = [0.044, 0.046, 0.048, 0.050]
    ax.set_xticks(ticks, ["0.044", "0.046", "0.048", "0.050"])
    ax.set_yticks(ticks, ["0.044", "0.046", "0.048", "0.050"])
    ax.set_xlabel("test NMSE at 1,024 updates")
    ax.set_ylabel("test NMSE at 16,384 updates")
    ax.text(0.04, 0.97, f"{n_in}/{N_SEEDS} exact seeds\nwithin {ZOOM[0]:.3f}–{ZOOM[1]:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=PT_BASE,
            color=RULE_COLOR["exact"], linespacing=1.1, zorder=6)
    ax.annotate(f"noise-only {floor:g}", xy=(1.0, floor), xycoords=("axes fraction", "data"),
                xytext=(-1.5, 2.2), textcoords="offset points", ha="right", va="bottom",
                fontsize=PT_BASE, color=MUTE)
    print(f"[F zoom] {n_in}/{N_SEEDS} exact seeds inside {ZOOM}")


# ── the canvas ────────────────────────────────────────────────────────────
CANVAS_H_PT = 493.0
HGUTTER_PT = 30.0
VGUTTER_PT = 34.0
MARGINS = Margins(left=42.0, right=10.0, top=20.0, bottom=54.0)
RESERVE = dict(left=14.0, right=8.0)


def build(path: Path = OUT):
    import json
    fs = csv(SOURCE, "figures/figure_source.csv")
    curves = csv(SOURCE, "summaries/all_curves.csv")
    outcomes = csv(SOURCE, "summaries/all_budget_outcomes.csv")
    pairs = csv(SOURCE, "summaries/paired_seed_contrasts.csv")
    contrasts = csv(SOURCE, "summaries/paired_contrasts.csv")
    cond = csv(SOURCE, "summaries/condition_summary.csv")
    dist = csv(SOURCE, "summaries/quartic_distribution_summary.csv")
    noise = csv(BRIDGE, "summaries/task_variance_and_noise_floor.csv").set_index("task")
    protocol = json.loads((SOURCE / "protocol_freeze.json").read_text())
    assert protocol["max_updates"] == 16384 and "seed210999" in protocol["uncertainty"]
    assert "10000 draws" in protocol["uncertainty"]
    floor_pair = float(noise.loc["matching", "expected_label_noise_nmse"])
    floor_quartic = float(noise.loc["quartet", "expected_label_noise_nmse"])
    assert floor_pair == 0.0225 and floor_quartic == 0.045
    assert len(curves) == 48240 and len(outcomes) == 5760
    idx = boot_index()
    cv_curves = Curves(curves, protocol["checkpoints"])

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 3, hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
                      margins=MARGINS, letter_clearance=True)
    ax_a = cv.panel("A", 0, 0, 6, grid="y", title="Pairwise: rates selected per rule")
    ax_b = cv.panel("B", 0, 6, 6, grid="y", title=f"Pairwise: common rate {COMMON_RATE:g}")
    ax_c = cv.panel("C", 1, 0, 6, grid="y", title="Quartic: selected rates and common rate")
    ax_d = cv.panel("D", 1, 6, 6, grid="y", title="Pairwise at 16,384 updates: every seed")
    ax_e = cv.panel("E", 2, 0, 4, grid="y", title="Credit deficit over budget")
    ax_f = cv.panel("F", 2, 4, 4, grid="none", title="Quartic seeds, two budgets")
    ax_z = cv.panel("F_zoom", 2, 8, 4, letter="", grid="none", title="Exact seeds at the floor, 20×")
    for name in ("A", "B", "C", "D", "E", "F", "F_zoom"):
        cv.declare_reserve(name, **RESERVE)

    tails_a = panel_pairwise(ax_a, "selected_rate", "A", cv_curves, fs, idx, floor_pair)
    tails_b = panel_pairwise(ax_b, "common_rate", "B", cv_curves, fs, idx, floor_pair)
    # exact and unit broadcast are the same runs in A and B
    for key in ("exact", "unit_broadcast"):
        np.testing.assert_array_equal(tails_a[key][2], tails_b[key][2])
    panel_quartic(ax_c, cv_curves, fs, idx, floor_quartic)
    panel_terminal(ax_d, outcomes, cond, idx, floor_pair)
    panel_contrast(ax_e, fs, pairs, contrasts, idx)
    seeds = seed_table(fs, outcomes)
    panel_seeds(ax_f, seeds, floor_quartic, dist)
    panel_zoom(ax_z, seeds, floor_quartic)

    # one shared key for the credit rules and the reference rules of A-D, F
    handles = [Line2D([], [], color=c, lw=LW_DATA, dashes=DOTTED if ls == ":" else (None, None),
                      marker="o", ms=DOT_MS, markeredgecolor="none", markerfacecolor=c, label=lab)
               for _, lab, c, ls in RULES]
    handles.append(Line2D([], [], color=MUTE, lw=0, marker="|", ms=7.0, markeredgewidth=LW_REF,
                          label="1,024-update cap"))
    handles.append(Line2D([], [], color=MUTE, lw=LW_REF, label="noise-only NMSE"))
    handles.append(Line2D([], [], color=INK, lw=LW_DATA, label="thick (C): rates selected per rule"))
    handles.append(Line2D([], [], color=INK, lw=LW_THIN, label=f"thin (C): common rate {COMMON_RATE:g}"))
    cv.fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=4,
                  frameon=False, fontsize=PT_BASE, handlelength=2.4, columnspacing=1.6,
                  handletextpad=0.6, borderaxespad=0.5, labelspacing=0.35)
    problems = cv.save(path, name="figure_fixed_profile_budget_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

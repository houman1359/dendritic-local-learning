#!/usr/bin/env python3
"""Supplementary sheet S20 (ident ``conductance_optimization``) -- parameter-
range, development and rate controls of the opponent conductance study --
rebuilt as ONE native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/conductance_optimization
.pdf``) is a paste of two upstream renders (S51 B-D, S52 A-B) whose builders
are frozen by the source registry, so its verified in-panel defects could
not be fixed by the paste layer.  This builder reads ONLY the frozen tables
under ``source_data/conductance_credit_demand/opponent`` and redraws the same
five panels with the same plotted quantities.  Nothing about the numbers
changes; every printed or plotted value is asserted against the table it
comes from, and every mean, interval and count is recomputed from the frozen
per-seed rows with the study's own bootstrap (``report.py:interval``, 20,000
whole-seed draws, ``default_rng(982211)``) before it is drawn.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
frozen_S20.json), panel by panel:

* A is a forest of all TWELVE task-by-credit interaction rows of
  ``bound_sensitivity/summaries/paired_contrasts.csv`` (three credit rules
  minus exact path x Adam/SGD x log-conductance bounds +/-7 and +/-20),
  grouped by rule with the rule named and keyed in the group header, every
  one of the 20 paired seed values drawn as a fan behind the mean and 95 %
  paired-bootstrap whisker, the axis anchored at 0 and tight to the seeds,
  the n-positive count printed with its denominator on every row, and the
  oracle rows -- whose means and intervals all lie within +/-0.000005 of
  zero, too narrow to draw -- stated as such.  One marker style; the bound
  is in the row label.  The stranded 'Log-conductance bounds' legend title
  is gone.  A shares the top row with B instead of holding a band alone.
* B plots the twenty per-seed calibrated-broadcast-minus-exact gaps against
  the CONTINUOUS update of first bound contact on a log axis (every 2^k
  position ticked), so no two seeds coincide and all 20 are countable; it
  carries its 20/20 count and its zero reference labelled.
* C keeps the four development regimes under Adam but draws the three
  per-seed development values behind every mean and names the 4,096-update
  budget in its title.
* D and E draw the three per-seed values behind every six-rate mean, name
  the 16,384-update budget in their titles, label the dotted rule as the
  originally selected rate, share one tick convention with C (composed
  10^k labels every two decades, set from token-size spans because Nimbus
  Sans has no superscript minus) and, in D, state that the two broadcast
  series coincide.
* One palette register for the credit rules across the sheet, the one the
  cross-figure review asked for: exact path dark red (``bp``), calibrated
  broadcast blue (``additive``, the main Fig. 4 hue; the frozen render's
  'Initial profile'), unit broadcast amber (``local``), three-profile oracle
  violet (``oracle``).  The two control rules (oracle, unit broadcast) are
  dashed wherever a line is drawn; each rule has one marker shape.  In A the
  hue is the rule whose deficit against exact path is drawn; in B it is the
  calibrated-broadcast gap.  No colour carries a second meaning anywhere on
  the sheet, and one key at the foot serves C-E (A keys its own groups).

2026-09-23 clarity pass (analysis/figure_visual_review_20260910/
review_20260923/si_pass/ledger/conductance_optimization.md): the artwork keeps
axis labels, tick labels, the rule labels of A, A's per-row sign counts, the
short C-E condition titles, the 'selected rate' rule label and the key.  The
finding titles of A and B, the statistics notes ('n = ... seeds ...'), the
zero-rule labels ('no interaction', 'no gap'), B's 20/20 count, A's oracle
footnote and D's coincidence note moved to the legend; the assertions behind
them still run.  Axis labels are in sentence case.  The key is laid out on the
panel-letter grid so that no key mark falls in the strip beside a row-1 letter
where the supplement's whole-sheet paste assigns it to that letter's panel.
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
    LW_DATA,
    LW_EDGE,
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
)

ROOT = SCRIPT_DIR.parent
STUDY = ROOT / "source_data" / "conductance_credit_demand" / "opponent"
BOUNDS = STUDY / "bound_sensitivity" / "summaries"
RATES = STUDY / "expanded_rates"
OUT = ROOT / "figures" / "supplementary" / "figure_conductance_optimization_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
DASHED = (0, (2.6, 1.8))
DOTTED = (0, (0.9, 1.6))
SERIES_MS = MARKER_MS * 0.78
LOG_TICK_PAD_PT = 11.0          # room for the raised exponent between "10" and the spine
TITLE_PAD_PT = 4.0              # condition title to axes top (no n tag since 2026-09-23)

# credit rules: (table key, key label, colour, line style, marker).  The two
# control rules are dashed AND hollow, so a control mean that lands on a
# primary mean (the broadcast pair in D, exact/oracle in C) still shows both.
RULES = (
    ("exact", "exact path", COLORS["bp"], "-", "o"),
    ("ancestry_three_oracle", "three-profile oracle", COLORS["oracle"], DASHED, "D"),
    ("calibrated_broadcast", "calibrated broadcast", COLORS["additive"], "-", "s"),
    ("unit_broadcast", "unit broadcast", COLORS["local"], DASHED, "^"),
)
RULE = {r[0]: r for r in RULES}
CONTROL = {"ancestry_three_oracle", "unit_broadcast"}


def mean_marker_kw(key, color):
    """Filled with a white edge for a primary rule, hollow for a control."""
    if key in CONTROL:      # no fill at all, so a primary mean underneath stays visible
        return dict(markerfacecolor="none", markeredgecolor=color, markeredgewidth=LW_ERR)
    return dict(markerfacecolor=color, markeredgecolor="white", markeredgewidth=LW_HAIR)
OPTIMIZERS = (("adam", "Adam"), ("sgd", "SGD"))
# (table key, gating level); the task alignment is the second tick level
REGIMES = (("aligned_strong", "strong"), ("opposed_strong", "strong"),
           ("opposed_moderate", "moderate"), ("opposed_ungated", "ungated"))
REGIME_GROUPS = (("aligned", (0, 0)), ("opposed", (1, 3)))
GROUP_ROW_PT = 21.0             # second tick level, below the gating labels
GROUP_RULE_PT = 18.5
ORACLE_BAND = 5e-6              # |mean|, |lo|, |hi| of every oracle row lie inside this
SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def interval(values):
    """``report.py:interval`` verbatim: the study's whole-seed bootstrap."""
    x = np.asarray(values, float)
    rng = np.random.default_rng(982211)
    boot = x[rng.integers(len(x), size=(20000, len(x)))].mean(1)
    return float(x.mean()), *np.quantile(boot, [.025, .975]).tolist()


def csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fan_x(ax, y, values, color, *, half=0.16, ms=SEED_MS, alpha=SEED_ALPHA, zorder=2.0):
    """Per-seed values as a jittered fan (horizontal panel)."""
    values = np.asarray(values, float)
    jitter = np.linspace(-half, half, len(values)) if len(values) > 1 else np.zeros(1)
    ax.plot(values, y + jitter, linestyle="none", marker="o", markersize=ms,
            markerfacecolor=color, markeredgecolor="none", alpha=alpha, zorder=zorder)


def fan_y(ax, x, values, color, *, half=0.05, log_x=False, ms=SEED_MS, alpha=SEED_ALPHA,
          zorder=2.0):
    """Per-seed values as a jittered fan (vertical panel); the jitter is
    multiplicative on a log x axis so it reads the same at every rate."""
    values = np.asarray(values, float)
    jitter = np.linspace(-half, half, len(values)) if len(values) > 1 else np.zeros(1)
    xs = x * 10.0 ** jitter if log_x else x + jitter
    ax.plot(xs, values, linestyle="none", marker="o", markersize=ms,
            markerfacecolor=color, markeredgecolor="none", alpha=alpha, zorder=zorder)


def log_y_axis(ax, lo, hi, exponents):
    """Log y axis with composed ``10^k`` tick labels at token size.

    Nimbus Sans has no SUPERSCRIPT MINUS, so a literal '10⁻⁶' loses its sign
    and mathtext would shrink the exponent below the 7 pt floor; the label is
    therefore two token-size spans -- the tick label '10', padded off the
    spine, and a raised '−k' annotated in the pad -- the idiom of
    :func:`figure_canvas.token_subscript`.  The tick label is a real tick
    label so the column lock measures it.
    """
    ax.set_yscale("log")
    ticks = [10.0 ** k for k in exponents]
    ax.set_yticks(ticks, ["1" if k == 0 else "10" for k in exponents])
    ax.set_ylim(lo, hi)
    assert all(lo < t < hi for t in ticks), (lo, hi, ticks)
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", pad=LOG_TICK_PAD_PT)
    for k, t in zip(exponents, ticks):
        if k == 0:
            continue
        ax.annotate(f"−{-k}" if k < 0 else str(k), xy=(0.0, t),
                    xycoords=("axes fraction", "data"), xytext=(-4.6, 2.4),
                    textcoords="offset points", ha="right", va="baseline",
                    fontsize=PT_BASE, color=INK, annotation_clip=False)


def series_line(ax, xs, means, key, *, zorder=3.0):
    _, _, color, ls, marker = RULE[key]
    ax.plot(xs, means, color=color, lw=LW_DATA, ls=ls, zorder=zorder)
    ax.plot(xs, means, linestyle="none", marker=marker, ms=SERIES_MS, color=color,
            zorder=zorder + 0.5, **mean_marker_kw(key, color))


# ── A: task-by-credit interaction under both bounds, all rules ───────────
def panel_interaction(ax, contrasts, seed_contrasts, sheet):
    groups = (("calibrated_broadcast", "calibrated broadcast − exact path"),
              ("unit_broadcast", "unit broadcast − exact path"),
              ("ancestry_three_oracle", "three-profile oracle − exact path"))
    cells = (("adam", 7.0, "Adam ±7"), ("adam", 20.0, "Adam ±20"),
             ("sgd", 7.0, "SGD ±7"), ("sgd", 20.0, "SGD ±20"))
    pitch = 5.6                          # rows per group incl. the header row
    t = contrasts[contrasts.contrast.eq("task_difference_in_gap")]
    assert len(t) == 12 and t.n.eq(20).all()
    ts = seed_contrasts[seed_contrasts.contrast.eq("task_difference_in_gap")]
    assert len(ts) == 240 and ts.seed.nunique() == 20
    old = sheet[sheet.panel.eq("B")]
    assert len(old) == 4 and old.rule.eq("calibrated_broadcast").all()
    labels, printed, oracle_extreme = [], [], 0.0
    fan_max = 0.0
    for g, (rule, header) in enumerate(groups):
        _, _, color, _, marker = RULE[rule]
        y_h = g * pitch
        ax.plot([0.03], [y_h], linestyle="none", marker=marker, ms=SERIES_MS, color=color,
                zorder=4.0, clip_on=False, **mean_marker_kw(rule, color))
        ax.annotate(header, xy=(0.03, y_h), xycoords="data", xytext=(4.5, 0.0),
                    textcoords="offset points", ha="left", va="center", fontsize=PT_BASE,
                    color=color, zorder=5.0)
        for i, (opt, bound, label) in enumerate(cells):
            y = y_h + 1 + i
            r = t[t.optimizer.eq(opt) & t.bound.eq(bound) & t.rule.eq(rule)]
            assert len(r) == 1
            r = r.iloc[0]
            v = ts[ts.optimizer.eq(opt) & ts.bound.eq(bound) & ts.rule.eq(rule)]
            v = v.sort_values("seed").value.to_numpy(float)
            assert len(v) == 20
            m, lo, hi = interval(v)
            np.testing.assert_allclose([m, lo, hi], [r["mean"], r.ci_low, r.ci_high],
                                       rtol=0, atol=1e-12)
            n_pos = int((v > 0).sum())
            assert n_pos == int(r.n_positive)
            if rule == "calibrated_broadcast":      # the four rows the old sheet drew
                o = old[old.optimizer.eq(opt) & old.bound.eq(bound)].iloc[0]
                np.testing.assert_allclose([m, lo, hi], [o["mean"], o.ci_low, o.ci_high],
                                           rtol=0, atol=1e-12)
                assert int(o.n_positive) == n_pos == 20
            if rule == "ancestry_three_oracle":
                oracle_extreme = max(oracle_extreme, abs(m), abs(lo), abs(hi))
            fan_max = max(fan_max, float(v.max()))
            # hairline row tick at the left edge, fan, whisker, mean, label, count
            ax.plot([0, 0], [y - 0.3, y + 0.3], color=COLORS["edge"], lw=LW_HAIR,
                    transform=ax.get_yaxis_transform(), clip_on=False, zorder=1.5,
                    solid_capstyle="butt")
            fan_x(ax, y, v, color)
            ax.plot([lo, hi], [y, y], color=color, lw=LW_ERR, zorder=3.0, solid_capstyle="butt")
            for xb in (lo, hi):
                ax.plot([xb, xb], [y - 0.13, y + 0.13], color=color, lw=LW_ERR, zorder=3.0,
                        solid_capstyle="butt")
            ax.plot([m], [y], linestyle="none", marker=marker, markersize=MARKER_MS,
                    color=color, zorder=4.0, **mean_marker_kw(rule, color))
            labels.append((y, label))
            ax.annotate(f"{n_pos}/20 > 0", xy=(1.0, y), xycoords=("axes fraction", "data"),
                        xytext=(3.0, 0.0), textcoords="offset points", ha="left", va="center",
                        fontsize=PT_BASE, color=MUTE, annotation_clip=False)
            printed.append(f"{rule} {label}: {m:.6f} [{lo:.6f}, {hi:.6f}] {n_pos}/20 > 0; "
                           f"seeds {v.min():.6f}-{v.max():.6f}")
    # the oracle rows: means and intervals inside +/-0.000005 of zero -- too
    # narrow to draw; the legend states it (2026-09-23), the check stays here
    assert oracle_extreme < ORACLE_BAND, oracle_extreme
    print("[A] " + "\n    ".join(printed))
    n_rows = len(groups) * pitch - (pitch - 5)
    ax.set_ylim(n_rows - 0.4, -0.7)
    ax.set_yticks([y for y, _ in labels], [lab for _, lab in labels])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(-0.04, 1.06)
    assert fan_max < 1.06
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.axvline(0.0, color=MUTE, lw=LW_REF, zorder=1.0, dashes=(2.6, 2.0))
    ax.set_xlabel("Task-by-credit interaction: opposed − aligned\ndifference in (rule − exact path) test NMSE")
    ax.tick_params(axis="y", length=0, pad=4.0)


# ── B: the gap before the first bound contact ────────────────────────────
def panel_precontact(ax, gaps, first, sheet):
    g = gaps[gaps.optimizer.eq("adam") & gaps.rule.eq("calibrated_broadcast")].sort_values("seed")
    assert len(g) == 20 and g.seed.nunique() == 20
    old = sheet[sheet.panel.eq("C")].sort_values("seed")
    assert len(old) == 20
    np.testing.assert_array_equal(old.seed.to_numpy(int), g.seed.to_numpy(int))
    np.testing.assert_allclose(old.gap.to_numpy(), g.gap.to_numpy(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(old.first_contact_step.to_numpy(), g.first_contact_step.to_numpy())
    np.testing.assert_allclose(old.last_precontact_checkpoint.to_numpy(),
                               g.last_precontact_checkpoint.to_numpy())
    np.testing.assert_allclose(g.gap.to_numpy(), (g.broadcast_nmse - g.exact_nmse).to_numpy(),
                               rtol=0, atol=1e-12)
    f = first[first.optimizer.eq("adam") & first.rule.eq("calibrated_broadcast")].sort_values("seed")
    np.testing.assert_array_equal(f.first_contact_step.to_numpy(int), g.first_contact_step.to_numpy(int))
    x = g.first_contact_step.to_numpy(float)
    y = g.gap.to_numpy(float)
    assert (y > 0).all() and int((y > 0).sum()) == 20
    assert x.min() == 384 and x.max() == 8326 and (x < 16385).all()
    # every seed's checkpoint is the last saved checkpoint below its contact
    checkpoints = np.array([0, 64, 256, 1024, 2048, 4096, 8192, 12288, 16384])
    for xi, ci in zip(x, g.last_precontact_checkpoint.to_numpy(float)):
        assert ci == checkpoints[checkpoints < xi].max()
    col = RULE["calibrated_broadcast"][2]
    counts = g.groupby("last_precontact_checkpoint").size()
    print(f"[B] n=20 gaps {y.min():.4f}-{y.max():.4f}, all > 0; first contact {x.min():.0f}-"
          f"{x.max():.0f}; checkpoint occupancy {counts.to_dict()}")
    ax.set_xscale("log")
    ax.plot(x, y, linestyle="none", marker="o", markersize=SEED_MS, markerfacecolor=col,
            markeredgecolor="none", alpha=SEED_ALPHA, zorder=2.0)
    ax.set_xlim(x.min() / 1.18, x.max() * 1.18)
    ax.set_xticks([512, 1024, 2048, 4096, 8192], ["512", "1,024", "2,048", "4,096", "8,192"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.045, 1.06)
    assert y.max() < 1.06
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.axhline(0.0, color=MUTE, lw=LW_REF, zorder=1.0, dashes=(2.6, 2.0))
    ax.set_xlabel("Update of first bound contact (Adam)")
    ax.set_ylabel("Calibrated broadcast − exact path\ntest NMSE, last checkpoint before contact")


# ── C: every development regime under Adam ───────────────────────────────
def panel_regimes(ax, selection, endpoints, sheet):
    sel = selection[selection.optimizer.eq("adam")]
    dev = endpoints[endpoints.phase.eq("development") & endpoints.optimizer.eq("adam")]
    assert dev.seed.nunique() == 3 and sorted(dev.seed.unique()) == [201, 202, 203]
    old = sheet[sheet.panel.eq("D")]
    assert len(old) == 16
    xs = np.arange(len(REGIMES), dtype=float)
    dodge = dict(zip([r[0] for r in RULES], np.linspace(-0.27, 0.27, len(RULES))))
    seed_lo, seed_hi = np.inf, 0.0
    printed = []
    for key, _, color, _, marker in RULES:
        means = []
        for i, (task, _) in enumerate(REGIMES):
            p = sel[sel.task.eq(task) & sel.rule.eq(key)]
            assert len(p) == 3, "three equally budgeted rates per rule"
            r = p.loc[p.validation_nmse.idxmin()]
            o = old[old.task.eq(task) & old.rule.eq(key)].iloc[0]
            np.testing.assert_allclose(r.validation_nmse, o.validation_nmse, rtol=0, atol=1e-15)
            assert r.rate == o.rate
            seeds = dev[dev.task.eq(task) & dev.rule.eq(key) & dev.rate.eq(r.rate)]
            seeds = seeds.sort_values("seed").validation_nmse.to_numpy(float)
            assert len(seeds) == 3
            np.testing.assert_allclose(seeds.mean(), r.validation_nmse, rtol=1e-12, atol=0)
            seed_lo, seed_hi = min(seed_lo, seeds.min()), max(seed_hi, seeds.max())
            fan_y(ax, xs[i] + dodge[key], seeds, color, half=0.045)
            means.append(float(r.validation_nmse))
            printed.append(f"{key} {task} rate {r.rate}: {r.validation_nmse:.3e} "
                           f"(seeds {seeds.min():.2e}-{seeds.max():.2e})")
        ax.plot(xs + dodge[key], means, linestyle="none", marker=marker, ms=MARKER_MS,
                color=color, zorder=4.0, **mean_marker_kw(key, color))
    print("[C] " + "\n    ".join(printed))
    lo, hi = seed_lo / 1.8, seed_hi * 1.9
    log_y_axis(ax, lo, hi, [-4, -2, 0])
    ax.set_xlim(-0.45, len(REGIMES) - 0.55)
    ax.set_xticks(xs, [lab for _, lab in REGIMES])
    # second tick level: the task alignment, a hairline over each group
    from matplotlib.transforms import offset_copy
    for name, (i0, i1) in REGIME_GROUPS:
        rule = offset_copy(ax.get_xaxis_transform(), fig=ax.figure, x=0.0, y=-GROUP_RULE_PT,
                           units="points")
        ax.plot([i0 - 0.38, i1 + 0.38], [0.0, 0.0], transform=rule, color=COLORS["edge"],
                lw=LW_HAIR, clip_on=False, zorder=1.5, solid_capstyle="butt")
        ax.annotate(name, xy=((i0 + i1) / 2.0, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0.0, -GROUP_ROW_PT), textcoords="offset points", ha="center",
                    va="top", fontsize=PT_BASE, color=INK, annotation_clip=False)
    ax.set_xlabel("Gating regime", labelpad=GROUP_ROW_PT - 2.0)
    ax.set_ylabel("Development validation NMSE")


# ── D, E: six-rate opposed-tuning sweeps ─────────────────────────────────
def panel_rates(ax, source, endpoints, protocol, optimizer, selected_rate, *, note_y,
                note_side="right", note=False):
    rates = [float(r) for r in protocol["rates"][optimizer]]
    assert len(rates) == 6 and selected_rate in rates
    panel = "A" if optimizer == "adam" else "B"
    src = source[source.panel.eq(panel)]
    assert len(src) == 24 and src.task.eq("opposed_strong").all() and src.optimizer.eq(optimizer).all()
    ends = endpoints[endpoints.task.eq("opposed_strong") & endpoints.optimizer.eq(optimizer)]
    assert len(ends) == 72 and sorted(ends.seed.unique()) == protocol["development_seeds"]
    seed_lo, seed_hi = np.inf, 0.0
    curves = {}
    ratio_max = 0.0
    # the four fans of one rate are dodged in log x so coincident rules (the
    # broadcast pair, exact/oracle) show six seeds side by side, not three
    fan_dodge = dict(zip([r[0] for r in RULES], np.linspace(-0.045, 0.045, len(RULES))))
    for key, _, color, _, _ in RULES:
        means = []
        for rate in rates:
            r = src[src.rule.eq(key) & src.rate.eq(rate)]
            assert len(r) == 1
            r = r.iloc[0]
            seeds = ends[ends.rule.eq(key) & ends.rate.eq(rate)].sort_values("seed")
            seeds = seeds.validation_nmse.to_numpy(float)
            assert len(seeds) == 3
            np.testing.assert_allclose(seeds.mean(), r.validation_nmse, rtol=1e-12, atol=0)
            assert bool(r.original_grid) == (rate in rates[:3])
            seed_lo, seed_hi = min(seed_lo, seeds.min()), max(seed_hi, seeds.max())
            ratio_max = max(ratio_max, seeds.max() / seeds.min())
            fan_y(ax, rate * 10.0 ** fan_dodge[key], seeds, color, half=0.018, log_x=True)
            means.append(float(r.validation_nmse))
        curves[key] = np.array(means)
    # the primary rules first so the dashed controls read on top of them
    for key in ("exact", "calibrated_broadcast", "ancestry_three_oracle", "unit_broadcast"):
        series_line(ax, rates, curves[key], key)
    pair = np.abs(curves["unit_broadcast"] / curves["calibrated_broadcast"] - 1.0)
    print(f"[{optimizer}] seeds {seed_lo:.2e}-{seed_hi:.2e}, max three-seed max/min {ratio_max:.1f}; "
          f"unit/calibrated relative difference max {pair.max():.2e}; exact at selected rate "
          f"{curves['exact'][rates.index(selected_rate)]:.3e}")
    log_y_axis(ax, 4e-9, 3.0, [-8, -6, -4, -2, 0])
    assert 4e-9 < seed_lo and seed_hi < 3.0
    ax.set_xscale("log")
    ax.set_xlim(rates[0] / 1.9, rates[-1] * 1.9)
    ax.set_xticks(rates, [f"{r:g}" for r in rates])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(selected_rate, color=MUTE, lw=LW_REF, zorder=1.0, dashes=(0.9, 1.6))
    # the label sits in the empty band between the exact/oracle curves and
    # the broadcast band: under Adam those curves stay below 4e-7 at rates up
    # to 0.1, so it sits right of the rule at 1e-4; under SGD they run at
    # 1.5e-4 and climb through 1e-2 right of rate 1, so it sits LEFT of the
    # rule at 3e-3, over the 0.03-0.1 columns whose seeds are below 3e-4
    ax.annotate("selected\nrate", xy=(selected_rate, note_y), xycoords="data",
                xytext=(2.5 if note_side == "right" else -2.5, 0.0), textcoords="offset points",
                ha="left" if note_side == "right" else "right", va="center",
                fontsize=PT_BASE, color=MUTE, linespacing=1.15, zorder=5.0)
    if note:        # the legend states the coincidence (0.03 %); the check stays
        assert pair.max() < 3e-4
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Development validation NMSE")
    return pair.max(), ratio_max


# ── the shared key ───────────────────────────────────────────────────────
KEY_HANDLE_PT = 2.0 * PT_BASE   # handle length, pad and entry spacing in em of
KEY_PAD_PT = 0.5 * PT_BASE      # the 7 pt key type, as a legend would set them
KEY_SPACE_PT = 1.2 * PT_BASE
KEY_LETTER_GAP_PT = 6.0         # the left pair ends this far short of letter E


def rule_key(cv, ax_c, ax_e):
    """The credit-rule key, drawn as two pairs on the foot line of row 1.

    The supplement pastes this sheet whole and gives each row-1 panel the
    strip that starts 3 pt left of its letter, so a key mark between letter
    E and E's y label would count as E's leftmost ink and the letter could
    not clear it.  The first pair (exact path, three-profile oracle) therefore
    ends ``KEY_LETTER_GAP_PT`` before letter E and the second pair (calibrated
    and unit broadcast) starts at E's y label, the panel's own leftmost mark.
    The key sits level with C's axis label, on the line D and E leave free
    under their x labels.  Called after the letters have their final place.
    """
    fig = cv.fig
    W, H = cv.width_pt, cv.height_pt
    cv.lock_reserves()
    cv.reserve_letter_clearance()
    cv.align_letters()
    renderer = fig.canvas.get_renderer()
    s = 72.0 / fig.dpi
    letter = {it["letter"]: it["art"].get_window_extent(renderer) for it in cv._letters}
    le, ld = letter["E"], letter["D"]
    ylab = ax_e.yaxis.label.get_window_extent(renderer)
    xlab = ax_c.xaxis.label.get_window_extent(renderer)
    y = 0.5 * (xlab.y0 + xlab.y1) * s                      # points from the foot
    entries = []
    for key, label, color, ls, marker in RULES:
        text = fig.text(0.0, y / H, label, fontsize=PT_BASE, color=INK, ha="left",
                        va="center_baseline")
        width = text.get_window_extent(renderer).width * s
        entries.append((key, color, ls, marker, text,
                        KEY_HANDLE_PT + KEY_PAD_PT + width))

    def place(pair, x):
        for key, color, ls, marker, text, width in pair:
            fig.add_artist(Line2D([x / W, (x + KEY_HANDLE_PT) / W], [y / H, y / H],
                                  transform=fig.transFigure, color=color, lw=LW_DATA, ls=ls))
            fig.add_artist(Line2D([(x + 0.5 * KEY_HANDLE_PT) / W], [y / H],
                                  transform=fig.transFigure, linestyle="none", marker=marker,
                                  ms=SERIES_MS, color=color, **mean_marker_kw(key, color)))
            text.set_x((x + KEY_HANDLE_PT + KEY_PAD_PT) / W)
            x += width + KEY_SPACE_PT
        return x - KEY_SPACE_PT

    first, second = entries[:2], entries[2:]
    first_w = sum(e[-1] for e in first) + KEY_SPACE_PT
    x_first = le.x0 * s - KEY_LETTER_GAP_PT - first_w
    x_second = max(ylab.x0 * s, le.x1 * s + 3.6)
    place(first, x_first)
    end = place(second, x_second)
    # both pairs stay clear of the strips beside letters D and E, and on the page
    assert x_first > ld.x1 * s + 3.6, (x_first, ld.x1 * s)
    assert end < W - 4.0, (end, W)
    print(f"[key] pairs at {x_first:.1f}-{x_first + first_w:.1f} and {x_second:.1f}-{end:.1f} pt; "
          f"letter E {le.x0 * s:.1f}-{le.x1 * s:.1f} pt")


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 486.0             # 2026-09-23: 6 pt shorter once the titles and notes left
ROW_PT = [216.0, 160.0]         # C-E keep an axes aspect near the 0.74 they had
HGUTTER_PT = 20.0
VGUTTER_PT = 56.0               # A's two-line x label stays nearer A than row 1's titles
MARGINS = Margins(left=40.0, right=6.0, top=10.0, bottom=44.0)


def build(path: Path = OUT):
    contrasts = csv(BOUNDS / "paired_contrasts.csv")
    seed_contrasts = csv(BOUNDS / "paired_seed_contrasts.csv")
    gaps = csv(BOUNDS / "precontact_gaps.csv")
    first = csv(BOUNDS / "first_contact.csv")
    sheet = csv(STUDY / "supplementary_figures" / "supplement_opponent_controls_source.csv")
    selection = csv(STUDY / "development_rate_selection.csv")
    endpoints = csv(STUDY / "summaries" / "all_endpoints.csv")
    rate_source = csv(RATES / "figure_expanded_rates_source.csv")
    rate_endpoints = csv(RATES / "all_endpoints.csv")
    rate_protocol = json.loads((RATES / "protocol_freeze.json").read_text())
    bound_protocol = json.loads((STUDY / "bound_sensitivity" / "protocol_freeze.json").read_text())
    protocol = json.loads((STUDY / "protocol.json").read_text())
    selection_freeze = json.loads((STUDY / "selection_freeze.json").read_text())
    assert bound_protocol["bounds"] == [7.0, 20.0] and bound_protocol["steps"] == 16384
    assert rate_protocol["steps"] == 16384 and rate_protocol["development_seeds"] == [201, 202, 203]
    assert protocol["checkpoints"][-1] == 4096 and protocol["development_seeds"] == [201, 202, 203]
    assert len(rate_endpoints) == 288 and len(rate_source) == 96
    selected = selection_freeze["rates"]["opposed_strong"]
    assert selected["adam"]["exact"] == 0.03 and selected["sgd"]["exact"] == 0.3
    assert all(v == 0.03 for v in selected["adam"].values())
    assert all(v == 0.3 for v in selected["sgd"].values())

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 2, row_weights=ROW_PT, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS)
    # A and B carry no title (their findings are the legend's); C-E keep the
    # short condition labels that tell the three otherwise-identical panels apart
    ax_a = cv.panel("A", 0, 0, 7, grid="x")
    ax_b = cv.panel("B", 0, 7, 5, grid="both")
    ax_c = cv.panel("C", 1, 0, 4, grid="y", title="Adam, 4,096 updates")
    ax_d = cv.panel("D", 1, 4, 4, grid="y", title="Adam, opposed, 16,384 updates")
    ax_e = cv.panel("E", 1, 8, 4, grid="y", title="SGD, opposed, 16,384 updates")

    for ax in (ax_c, ax_d, ax_e):
        ax.set_title(ax.get_title(), fontsize=PT_EMPH, color=INK, pad=TITLE_PAD_PT,
                     fontweight="normal")
    panel_interaction(ax_a, contrasts, seed_contrasts, sheet)
    panel_precontact(ax_b, gaps, first, sheet)
    panel_regimes(ax_c, selection, endpoints, sheet)
    exact_c = sheet[sheet.panel.eq("D") & sheet.task.eq("opposed_strong") & sheet.rule.eq("exact")]
    exact_d = rate_source[rate_source.panel.eq("A") & rate_source.rule.eq("exact") & rate_source.rate.eq(0.03)]
    ratio = float(exact_c.validation_nmse.iloc[0] / exact_d.validation_nmse.iloc[0])
    assert abs(ratio - 192.8) < 0.1, ratio                       # the caption's 193x
    print(f"[C vs D] exact, Adam, opposed strong, rate 0.03: {exact_c.validation_nmse.iloc[0]:.4e} "
          f"at 4,096 updates vs {exact_d.validation_nmse.iloc[0]:.4e} at 16,384 (x{ratio:.1f})")
    pair_d, _ = panel_rates(ax_d, rate_source, rate_endpoints, rate_protocol, "adam",
                                   selected["adam"]["exact"], note_y=1e-4, note=True)
    pair_e, _ = panel_rates(ax_e, rate_source, rate_endpoints, rate_protocol, "sgd",
                                   selected["sgd"]["exact"], note_y=3e-3, note_side="left")
    assert pair_d < 3e-4 and pair_e < 0.075, (pair_d, pair_e)   # the caption's 0.03 % and 7.4 %
    # A's per-row counts hang outside its right spine, unseen by the lock
    cv.declare_reserve("A", right=34.0)
    cv.lock_reserves()
    # the three panels of row 1 start in three different grid columns, so
    # they measure three different left needs (C's y label sits in the outer
    # margin, D's and E's in a gutter): give all three the largest lock so
    # they share one axes width, as the audit's row contract asks
    left = max(cv._locks[k][0] for k in "CDE")
    right = max(cv._locks[k][1] for k in "CDE")      # E ends at the margin, C and D at a gutter
    for name in "CDE":
        cv.declare_reserve(name, left=left, right=right)
    cv.lock_reserves()
    boxes = {k: cv.axes[k].get_position() for k in "CDE"}
    widths = [boxes[k].width * cv.width_pt for k in "CDE"]
    assert max(widths) - min(widths) < 0.5, widths

    # one shared key for the credit rules, on the foot line beside C's axis label
    rule_key(cv, ax_c, ax_e)
    problems = cv.save(path, name="figure_conductance_optimization_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

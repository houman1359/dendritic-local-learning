#!/usr/bin/env python3
"""Supplementary sheet S21 (ident ``local_gate_controls``) -- the complete
local-gate rate/budget grid of the prospective conductance-gate study --
rebuilt as ONE native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/local_gate_controls.pdf``)
is a paste of an upstream heat-map render with no generator in this
repository.  This builder reads ONLY the frozen tables under
``source_data/conductance_local_gate/summaries`` and redraws the same four
task x budget panels with the same plotted quantities.  Nothing about the
numbers changes: every plotted or printed value is asserted against the
table it comes from, every cell summary (mean, median, minimum, maximum) is
recomputed from the 20 per-seed rows of ``all_endpoints.csv``, and the 108
tabulated 95 % intervals are reproduced bit for bit by replaying the study's
own bootstrap (``report.py:summarize``: one ``default_rng(2026090800)``
stream, 20,000 whole-seed draws per cell, cells in ``groupby(sort=True)``
order) before anything is drawn.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
frozen_S21.json), sheet-wide:

* Each heat map becomes a per-rule strip plot on one shared log axis of
  test NMSE: for every rule and every Adam rate the 20 seed values are drawn
  as a fan behind the arithmetic mean and its 95 % whole-seed bootstrap
  interval (findings 1 and 3: the seed spread and the outlier seeds that
  inflate an arithmetic mean are visible instead of folded into one digit
  string; finding 4: no 8-decade colour ramp carries the reading).
* The byte-identical 'Unit broadcast' and 'Initial profile' rows collapse to
  ONE row, 'Unit broadcast (≈ calibrated)': the builder asserts that the
  calibrated-broadcast mean agrees with the unit-broadcast mean to two
  significant figures in all 12 cells and reports the largest per-seed
  disagreement (finding 2 and the cross-figure duplicate-row finding).
* Rows are ordered as the caption groups them -- five rules that learn
  accurately, a hairline, then the three that fail on at least one task
  (finding 8) -- and carry the main Fig. 5 names (finding 7): 'Two-profile
  oracle' for the two leaf patterns with unit proximal credit, 'Local
  distal gate', 'Gate also proximal', 'Swapped gate'.
* The three Adam rates are three strips inside each rule's row, top to
  bottom 0.01 / 0.03 / 0.1, keyed by marker shape with the primary 0.03
  rate named as such in the key (finding 5); the x axis is labelled 'Adam
  rate (fixed)' nowhere because the rate is no longer an axis.
* The x axis is shared by all four panels, its tick labels and axis label
  are set once per column, on C and D (finding 6), and the unit is one
  quantity, test NMSE, on a log scale (finding 9).
* Colour is the rule FAMILY, in the register main Fig. 5 uses: exact path
  ``bp``, oracle projections ``oracle``, local gates ``shunting``, broadcast
  ``scalar``, misplaced gate ``highlight``, swapped gate ``point_mlp``.
  Every mark is filled; no colour or fill carries a second meaning.
* An interval narrower than the mean marker is not drawn as a stub under
  it; the count of such intervals, with its denominator, is printed in the
  key band.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    LW_ERR,
    LW_HAIR,
    MARKER_MS,
    PT_BASE,
    SEED_ALPHA,
    SEED_MS,
    Margins,
    NativeCanvas,
    tint_patch,
)
from native_schematics import _text_w_pt  # noqa: E402

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "conductance_local_gate"
SUMMARY = SOURCE / "summaries"
OUT = ROOT / "figures" / "supplementary" / "figure_local_gate_controls_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]

# the study's bootstrap (protocol.json: statistics.bootstrap_seed / _draws)
BOOT_SEED = 2026090800
BOOT_DRAWS = 20000
N_SEEDS = 20

# rows, top first: (table rule id, row label, colour key).  The first five
# learn accurately on both tasks; the last three fail on at least one.
RULES = (
    ("exact", "Exact path", "bp"),
    ("ancestry_three_oracle", "Three-pattern\noracle", "oracle"),
    ("ancestry_two_leaf_oracle_unit_proximal", "Two-profile\noracle", "oracle"),
    ("hard_distal_unit_proximal", "Local distal gate", "shunting"),
    ("shunt_proportional_unit_proximal", "Shunt-proportional\ngate", "shunting"),
    ("unit_broadcast", "Unit broadcast\n(≈ calibrated)", "scalar"),
    ("hard_distal_and_proximal", "Gate also\nproximal", "highlight"),
    ("swapped_distal_unit_proximal", "Swapped gate", "point_mlp"),
)
N_ACCURATE = 5
CALIBRATED = "calibrated_broadcast"          # collapsed into the unit row
RATES = ((0.01, "^"), (0.03, "D"), (0.10, "v"))   # top strip first
PRIMARY_RATE = 0.03
STRIP_DY = 0.28                               # strip offset inside a row
ROW_GAP = 0.30                                # extra gap at the block break
FAN_HALF = 0.07                               # seed fan half-height (rows)
MEAN_MS = MARKER_MS * 0.78
PANELS = (                                    # letter, task, budget, title
    ("A", "aligned_strong", 4096, "Aligned task, 4,096 updates"),
    ("B", "aligned_strong", 16384, "Aligned task, 16,384 updates"),
    ("C", "opposed_strong", 4096, "Opposed task, 4,096 updates"),
    ("D", "opposed_strong", 16384, "Opposed task, 16,384 updates"),
)
X_TICKS = (1e-9, 1e-7, 1e-5, 1e-3, 1e-1)

# the frozen render's printed cells (two significant figures of the mean),
# transcribed from figures/supplementary/curated/local_gate_controls.pdf,
# so the redraw is asserted to carry exactly the old sheet's content.
OLD_PRINTED = {
    ("aligned_strong", 4096): {
        "exact": ("4.3e-05", "7.5e-06", "5.0e-06"),
        "unit_broadcast": ("7.9e-04", "5.5e-04", "3.0e-04"),
        "calibrated_broadcast": ("7.9e-04", "5.5e-04", "3.0e-04"),
        "ancestry_three_oracle": ("4.0e-05", "9.2e-06", "4.4e-06"),
        "hard_distal_unit_proximal": ("3.9e-05", "1.1e-05", "1.8e-06"),
        "swapped_distal_unit_proximal": ("5.1e-03", "5.4e-03", "4.3e-03"),
        "hard_distal_and_proximal": ("3.9e-05", "1.2e-05", "5.7e-06"),
        "ancestry_two_leaf_oracle_unit_proximal": ("4.0e-05", "9.1e-06", "7.1e-06"),
        "shunt_proportional_unit_proximal": ("3.9e-05", "1.1e-05", "4.2e-06"),
    },
    ("aligned_strong", 16384): {
        "exact": ("2.1e-07", "8.4e-07", "5.8e-07"),
        "unit_broadcast": ("2.7e-04", "2.4e-04", "2.6e-04"),
        "calibrated_broadcast": ("2.7e-04", "2.4e-04", "2.6e-04"),
        "ancestry_three_oracle": ("2.5e-07", "2.9e-07", "1.7e-07"),
        "hard_distal_unit_proximal": ("3.4e-07", "2.7e-07", "2.2e-07"),
        "swapped_distal_unit_proximal": ("5.1e-03", "5.4e-03", "4.3e-03"),
        "hard_distal_and_proximal": ("1.4e-06", "2.4e-06", "3.3e-06"),
        "ancestry_two_leaf_oracle_unit_proximal": ("2.3e-07", "1.6e-07", "9.8e-08"),
        "shunt_proportional_unit_proximal": ("3.9e-07", "1.3e-07", "3.0e-07"),
    },
    ("opposed_strong", 4096): {
        "exact": ("7.9e-05", "2.2e-05", "7.0e-06"),
        "unit_broadcast": ("5.6e-01", "4.9e-01", "5.0e-01"),
        "calibrated_broadcast": ("5.6e-01", "4.9e-01", "5.0e-01"),
        "ancestry_three_oracle": ("7.3e-05", "2.3e-05", "8.5e-06"),
        "hard_distal_unit_proximal": ("7.8e-05", "2.3e-05", "9.1e-06"),
        "swapped_distal_unit_proximal": ("9.5e-01", "9.5e-01", "9.5e-01"),
        "hard_distal_and_proximal": ("1.3e-01", "1.2e-01", "1.3e-01"),
        "ancestry_two_leaf_oracle_unit_proximal": ("7.4e-05", "2.3e-05", "9.5e-06"),
        "shunt_proportional_unit_proximal": ("7.8e-05", "2.3e-05", "4.8e-06"),
    },
    ("opposed_strong", 16384): {
        "exact": ("1.5e-07", "8.5e-07", "4.2e-07"),
        "unit_broadcast": ("4.8e-01", "4.7e-01", "4.9e-01"),
        "calibrated_broadcast": ("4.8e-01", "4.7e-01", "4.9e-01"),
        "ancestry_three_oracle": ("9.0e-08", "7.0e-08", "1.2e-06"),
        "hard_distal_unit_proximal": ("2.3e-07", "2.9e-07", "3.9e-07"),
        "swapped_distal_unit_proximal": ("9.5e-01", "9.5e-01", "9.5e-01"),
        "hard_distal_and_proximal": ("1.3e-01", "1.2e-01", "1.3e-01"),
        "ancestry_two_leaf_oracle_unit_proximal": ("7.5e-08", "2.4e-07", "8.7e-07"),
        "shunt_proportional_unit_proximal": ("2.2e-07", "1.1e-07", "1.6e-06"),
    },
}


# ── tables ───────────────────────────────────────────────────────────────
def boot(values, rng, draws=BOOT_DRAWS):
    """``report.py:boot`` verbatim: percentile interval of the seed mean."""
    v = np.asarray(values, float)
    dist = v[rng.integers(len(v), size=(draws, len(v)))].mean(1)
    return np.quantile(dist, [.025, .975])


def load_tables():
    ep = pd.read_csv(SUMMARY / "all_endpoints.csv", float_precision="round_trip")
    cm = pd.read_csv(SUMMARY / "condition_means.csv", float_precision="round_trip")
    assert len(ep) == 2160 and len(cm) == 108
    assert ep.seed.nunique() == N_SEEDS
    assert not ep.duplicated(["seed", "task", "budget", "rate", "rule"]).any()
    assert ep.groupby(["task", "budget", "rate", "rule"]).size().eq(N_SEEDS).all()
    assert sorted(ep.rule.unique()) == sorted([r[0] for r in RULES] + [CALIBRATED])
    assert sorted(ep.rate.unique()) == [r[0] for r in RATES]
    assert (ep.optimizer == "adam").all()
    assert (cm.n == N_SEEDS).all()
    return ep, cm


def recompute_condition_means(ep, cm):
    """Every summary the sheet draws, rebuilt from the per-seed rows.

    The mean, median, minimum and maximum are recomputed exactly; the 95 %
    intervals are reproduced by replaying the study's single bootstrap
    stream over the cells in the order ``report.py`` visits them.
    """
    rng = np.random.default_rng(BOOT_SEED)
    cells = {}
    worst = 0.0
    for (task, budget, rate, rule), f in ep.groupby(["task", "budget", "rate", "rule"], sort=True):
        f = f.sort_values("seed")
        seeds = f.test_nmse.to_numpy(float)
        ci = boot(seeds, rng)
        row = cm[cm.task.eq(task) & cm.budget.eq(budget) & cm.rate.eq(rate) & cm.rule.eq(rule)]
        assert len(row) == 1
        row = row.iloc[0]
        np.testing.assert_allclose(seeds.mean(), row["mean"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(np.median(seeds), row["median"], rtol=1e-12, atol=0)
        np.testing.assert_allclose(seeds.min(), row["minimum"], rtol=0, atol=0)
        np.testing.assert_allclose(seeds.max(), row["maximum"], rtol=0, atol=0)
        np.testing.assert_allclose(ci, [row.ci_low, row.ci_high], rtol=1e-12, atol=0)
        worst = max(worst, abs(ci[0] - row.ci_low) / row.ci_low, abs(ci[1] - row.ci_high) / row.ci_high)
        cells[(task, budget, rate, rule)] = dict(
            seeds=seeds, seed_ids=f.seed.to_numpy(), mean=float(row["mean"]),
            median=float(row["median"]), lo=float(row.ci_low), hi=float(row.ci_high),
            n=int(row.n))
    assert len(cells) == 108
    return cells, worst


def check_old_sheet(cells):
    """The redraw carries the frozen render's 108 printed cells exactly."""
    n = 0
    for (task, budget), rows in OLD_PRINTED.items():
        assert len(rows) == 9
        for rule, printed in rows.items():
            for (rate, _), text in zip(RATES, printed):
                mean = cells[(task, budget, rate, rule)]["mean"]
                assert f"{mean:.1e}" == text, (task, budget, rate, rule, mean, text)
                n += 1
    assert n == 108
    return n


def check_broadcast_collapse(ep, cells):
    """Unit and calibrated broadcast: one drawn row, the agreement asserted."""
    for task in ("aligned_strong", "opposed_strong"):
        for budget in (4096, 16384):
            for rate, _ in RATES:
                u = cells[(task, budget, rate, "unit_broadcast")]["mean"]
                c = cells[(task, budget, rate, CALIBRATED)]["mean"]
                assert f"{u:.1e}" == f"{c:.1e}", (task, budget, rate, u, c)
    p = ep.pivot_table(index=["seed", "task", "budget", "rate"], columns="rule", values="test_nmse")
    ratio = (p["unit_broadcast"] / p[CALIBRATED]).to_numpy(float)
    assert len(ratio) == 240
    fold = float(np.exp(np.abs(np.log(ratio)).max()))
    assert 3.0 < fold < 3.1, fold                   # the caption's 3.1-fold
    rel = np.abs(p["unit_broadcast"] - p[CALIBRATED]) / p[CALIBRATED]
    return fold, float(rel.median())


def check_caption_numbers(cells):
    """The outlier statements the caption makes about B and D."""
    out = {}
    for rate, _ in RATES:
        c = cells[("aligned_strong", 16384, rate, "hard_distal_and_proximal")]
        out[rate] = (c["mean"], c["median"], c["mean"] / c["median"])
    means = [out[r][0] for r, _ in RATES]
    ratios = [out[r][2] for r, _ in RATES]
    assert 1.3e-6 < min(means) and max(means) < 3.4e-6
    assert 17.0 < min(ratios) and max(ratios) < 24.7
    acc = [cells[("aligned_strong", 16384, r, k)]["median"]
           for r, _ in RATES for k, _, _ in RULES[:N_ACCURATE]]
    assert 4.7e-8 < min(acc) and max(acc) < 1.9e-7
    two = cells[("opposed_strong", 16384, 0.03, "ancestry_two_leaf_oracle_unit_proximal")]
    sh = cells[("opposed_strong", 16384, 0.10, "shunt_proportional_unit_proximal")]
    assert 10.7 < two["mean"] / two["median"] < 10.9
    assert 10.0 < sh["mean"] / sh["median"] < 10.2
    return out


# ── drawing ──────────────────────────────────────────────────────────────
def row_y(i):
    """Row centre, top row first; the failing block sits below a gap."""
    return float(i) + (ROW_GAP if i >= N_ACCURATE else 0.0)


def plain_log_xticks(ax, ticks=X_TICKS, labels=True):
    """Log x ticks printed as ``10`` with a raised exponent, both at 7 pt.

    Mathtext is not used anywhere on the page; the exponent is a real text
    span offset from the right edge of its own ``10``.
    """
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.set_xticks(list(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    if not labels:
        ax.set_xticklabels([""] * len(ticks))
        return
    ax.set_xticklabels(["10"] * len(ticks))
    ax.tick_params(axis="x", pad=2.2)
    for label, tick in zip(ax.get_xticklabels(), ticks):
        exp = str(int(round(math.log10(tick)))).replace("-", "−")
        ax.annotate(exp, xy=(1.0, 0.45), xycoords=label, xytext=(0.3, 2.4),
                    textcoords="offset points", fontsize=PT_BASE, color=INK,
                    ha="left", va="center", annotation_clip=False)


def draw_panel(ax, cells, task, budget, *, xlim, labels, ticklabels, pt_per_decade):
    """One task x budget panel: eight rule rows, three rate strips each."""
    n_rows = len(RULES)
    ax.set_xlim(*xlim)
    ax.set_ylim(row_y(n_rows - 1) + 0.55, -0.55)      # top row first
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    x0, x1 = ax.get_xlim()
    hidden = 0
    drawn = 0
    half_marker_dec = (MEAN_MS / 2.0) / pt_per_decade   # marker radius, decades
    for i, (rule, label, ckey) in enumerate(RULES):
        col = COLORS[ckey]
        y = row_y(i)
        tint_patch(ax, ("rect", x0, y - 0.44, x1 - x0, 0.88), color=col, pct=6,
                   edge=False, radius_pt=1.5, zorder=0.2, clip_on=True)
        ax.plot([x0, x0], [y - 0.30, y + 0.30], color=EDGE, lw=LW_HAIR,
                clip_on=False, zorder=1.5, solid_capstyle="butt")
        if labels:
            ax.annotate(label, xy=(0.0, y), xycoords=("axes fraction", "data"),
                        xytext=(-4.0, 0.0), textcoords="offset points", ha="right",
                        va="center", fontsize=PT_BASE, color=INK, linespacing=1.15,
                        annotation_clip=False)
        for k, (rate, marker) in enumerate(RATES):
            ys = y + (k - 1) * STRIP_DY
            c = cells[(task, budget, rate, rule)]
            seeds = c["seeds"]
            assert len(seeds) == N_SEEDS == c["n"]
            jitter = np.linspace(-FAN_HALF, FAN_HALF, len(seeds))
            ax.plot(seeds, ys + jitter, linestyle="none", marker="o", markersize=SEED_MS,
                    markerfacecolor=col, markeredgecolor="none", alpha=SEED_ALPHA,
                    zorder=2.0, clip_on=True)
            lo, hi, mean = c["lo"], c["hi"], c["mean"]
            assert lo <= mean <= hi
            # the interval is drawn only where it reaches past the marker;
            # a stub hidden under the marker is counted and stated instead
            reach = max(math.log10(mean) - math.log10(lo), math.log10(hi) - math.log10(mean))
            if reach > half_marker_dec + 0.15 / pt_per_decade:
                ax.plot([lo, hi], [ys, ys], color=col, lw=LW_ERR, zorder=3.0,
                        solid_capstyle="butt")
                for xb in (lo, hi):
                    ax.plot([xb, xb], [ys - 0.09, ys + 0.09], color=col, lw=LW_ERR,
                            zorder=3.0, solid_capstyle="butt")
                drawn += 1
            else:
                hidden += 1
            ax.plot([mean], [ys], linestyle="none", marker=marker, markersize=MEAN_MS,
                    markerfacecolor=col, markeredgecolor="white", markeredgewidth=LW_HAIR,
                    zorder=4.0)
    # the block break: a hairline between the accurate and the failing rules
    ysep = (row_y(N_ACCURATE - 1) + row_y(N_ACCURATE)) / 2.0
    ax.plot([x0, x1], [ysep, ysep], color=EDGE, lw=LW_HAIR, zorder=1.4,
            solid_capstyle="butt", clip_on=False)
    plain_log_xticks(ax, labels=ticklabels)
    ax.grid(True, axis="x", zorder=0, linewidth=LW_HAIR, alpha=0.9, color=COLORS["grid"])
    ax.set_axisbelow(True)
    if ticklabels:
        ax.set_xlabel("test NMSE at the validation-selected checkpoint")
    return hidden, drawn


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 492.0
HGUTTER_PT = 24.0
VGUTTER_PT = 26.0
MARGINS = Margins(left=8.0, right=10.0, top=22.0, bottom=52.0)
LEFT_SPAN, RIGHT_SPAN = 7, 5


def build(path: Path = OUT):
    ep, cm = load_tables()
    cells, worst = recompute_condition_means(ep, cm)
    n_old = check_old_sheet(cells)
    fold, rel_median = check_broadcast_collapse(ep, cells)
    outliers = check_caption_numbers(cells)
    print(f"[tables] 108 cells x {N_SEEDS} seeds; condition_means.csv recomputed from "
          f"all_endpoints.csv (mean/median/min/max exact; bootstrap replay max rel "
          f"error {worst:.1e}); {n_old} printed cells of the frozen sheet reproduced")
    print(f"[broadcast] unit vs calibrated: all 12 cell means equal at 2 s.f.; per-seed "
          f"max fold difference {fold:.2f}, median relative difference {rel_median:.2e}; "
          f"calibrated row collapsed into the unit row")
    for rate, (m, md, r) in outliers.items():
        print(f"[B outliers] gate also proximal, aligned 16,384, rate {rate}: mean {m:.2e}, "
              f"median {md:.2e}, mean/median {r:.1f}")

    # one x axis for the whole sheet, tight to the extreme seed values
    lo_seed = float(ep.test_nmse.min())
    hi_seed = float(ep.test_nmse.max())
    xlim = (lo_seed / 2.2, hi_seed * 2.2)
    decades = math.log10(xlim[1]) - math.log10(xlim[0])

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 2, hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
                      margins=MARGINS)
    axes = {}
    for letter, task, budget, title in PANELS:
        row = 0 if letter in "AB" else 1
        col, span = (0, LEFT_SPAN) if letter in "AC" else (LEFT_SPAN, RIGHT_SPAN)
        axes[letter] = cv.panel(letter, row, col, span, title=title)
    # the row-label column of A and C is a declared reserve: measured from
    # the widest label, so the column lock gives A and C one axes width
    widest = max(_text_w_pt(axes["A"], line, PT_BASE)
                 for _, label, _ in RULES for line in label.split("\n"))
    gutter = widest + 8.0
    for name in ("A", "C"):
        cv.declare_reserve(name, left=gutter)
    for name in "ABCD":
        cv.declare_reserve(name, right=3.0)
    cv.lock_reserves()
    stats = {}
    for letter, task, budget, _ in PANELS:
        ax = axes[letter]
        w_pt = ax.get_position().width * cv.width_pt
        stats[letter] = draw_panel(ax, cells, task, budget, xlim=xlim, labels=letter in "AC",
                                   ticklabels=letter in "CD", pt_per_decade=w_pt / decades)
        print(f"[{letter}] axes {w_pt:.1f} pt wide, {w_pt / decades:.1f} pt per decade; "
              f"{stats[letter][1]} intervals drawn, {stats[letter][0]} within the marker")
    hidden = sum(v[0] for v in stats.values())
    total = sum(v[0] + v[1] for v in stats.values())
    assert total == 96

    # the key, in the bottom margin: rate strips by marker and the seed fan
    # on one line, the interval and the hidden-interval count on the next
    marks = [
        Line2D([], [], linestyle="none", marker="^", ms=MEAN_MS, mfc=INK, mec="white",
               mew=LW_HAIR, label="mean at Adam rate 0.01 (top strip of a rule)"),
        Line2D([], [], linestyle="none", marker="D", ms=MEAN_MS, mfc=INK, mec="white",
               mew=LW_HAIR, label="mean at 0.03, the primary rate (middle strip)"),
        Line2D([], [], linestyle="none", marker="v", ms=MEAN_MS, mfc=INK, mec="white",
               mew=LW_HAIR, label="mean at 0.1 (bottom strip)"),
        Line2D([], [], linestyle="none", marker="o", ms=SEED_MS, mfc=INK, mec="none",
               alpha=SEED_ALPHA, label=f"one seed ({N_SEEDS} per strip)"),
    ]
    interval = [
        Line2D([], [], color=INK, lw=LW_ERR, marker="|", ms=4.0, mew=LW_ERR,
               label=f"95 % CI of the mean ({BOOT_DRAWS:,} whole-seed bootstrap draws); "
                     f"{hidden} of {total} CIs lie entirely under their marker and are not drawn"),
    ]
    common = dict(frameon=False, fontsize=PT_BASE, handlelength=1.6, columnspacing=1.4,
                  handletextpad=0.5, borderaxespad=0.0, labelcolor=INK)
    cv.fig.legend(handles=marks, loc="lower center", bbox_to_anchor=(0.5, 13.0 / cv.height_pt),
                  ncol=4, **common)
    cv.fig.legend(handles=interval, loc="lower center", bbox_to_anchor=(0.5, 2.0 / cv.height_pt),
                  ncol=1, numpoints=2, **common)
    problems = cv.save(path, name="figure_local_gate_controls_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    print(f"  canvas {cv.width_pt:.1f} x {cv.height_pt:.1f} pt")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

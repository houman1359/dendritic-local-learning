#!/usr/bin/env python3
"""Supplementary sheet S17 (ident ``credit_optimizer_controls``) -- nested
targets and SGD controls in the balanced budget extension -- rebuilt as ONE
native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/credit_optimizer_controls.pdf``)
is a paste of the upstream render ``credit_rule_extension_controls.pdf``,
which has no generator in this repository.  This builder reads ONLY the
frozen tables under ``source_data/`` and redraws the same four panels with
the same plotted quantities:

* the panel table ``credit_rule_extension/figures/controls_figure_source.csv``
  (mean test NMSE and its pointwise 95 % whole-seed bootstrap interval at the
  64 checkpoints from update 64 to 16,384, per panel and rule);
* the per-seed rows of ``credit_rule_extension/summaries/all_curves.csv``
  (20 seeds x 67 checkpoints per task x optimizer x rule x rate), which are
  drawn as fans behind every mean and from which every mean and interval is
  RECOMPUTED with the study's own bootstrap (protocol_freeze.json:
  "10000 draws, seed 210999" -- one ``default_rng(210999).integers(20,
  size=(10000, 20))`` index matrix shared by every condition and step) and
  asserted against the panel table to 1e-12;
* the noise-only NMSE of each task from
  ``credit_rule_bridge/summaries/task_variance_and_noise_floor.csv``
  (label-noise variance 0.0225 over clean target variance 1.0 or 0.5), the
  study whose tasks, seeds and trajectories this extension replays.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
frozen_S17.json), panel by panel:

* Every panel draws all 20 per-seed traces of every series as a fan behind
  the mean (finding 1), so the unit-broadcast mean in B that no seed occupies
  (18 seeds at the floor, 2 divergent) is seen for what it is, and the
  counts are printed with their denominators on B and D.
* D no longer redraws the exact-path and initial-profile runs of C (they are
  the same runs: selected = common = 0.03 for those two rules, finding 2);
  D carries only the two broadcast arms that differ between the rate views,
  on an axis tight to their seeds, and says where the other two rules are.
* C keeps its four series but its three coinciding broadcast controls are
  deliberately separated by line style (finding 8: amber dashed, blue solid,
  ink dotted) and the coincidence is stated on the panel (finding 5).
* The noise floor is a light grey SOLID rule with its value printed at the
  left end of every rule (findings 7 and 9), keyed in the shared key; dotted
  is reserved for the initial-sign series.
* One meaning per colour and per channel: dark red exact path, amber unit
  broadcast, blue initial profile (the calibrated broadcast of main Fig. 4),
  ink initial sign; line style is the rule (solid / dashed / solid / dotted);
  a pale fill is always the 95 % interval of a mean; a hairline is always
  one seed; grey solid is the noise floor and mute dashed the 1,024 cap.
* Titles name task, optimizer and rate view on all four panels; the
  vertical axes are log and tight to the data of each panel.
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
    COLORS,
    LW_DATA,
    LW_HAIR,
    LW_REF,
    PT_BASE,
    Margins,
    NativeCanvas,
)
from journal_style import ADDRESS_RAMP  # noqa: E402

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "credit_rule_extension"
BRIDGE = ROOT / "source_data" / "credit_rule_bridge"
OUT = ROOT / "figures" / "supplementary" / "figure_credit_optimizer_controls_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
FLOOR_GREY = ADDRESS_RAMP[1]          # the neutral structural grey, never a series
DASHED = (3.2, 1.8)
DOTTED = (1.0, 1.8)
CAP_DASH = (2.6, 2.0)

# credit rules: (table key, key label, colour, dash pattern or None)
RULES = (
    ("exact", "exact path", COLORS["bp"], None),
    ("unit_broadcast", "unit broadcast", COLORS["local"], DASHED),
    ("calibrated_broadcast", "initial profile (calibrated broadcast)", COLORS["additive"], None),
    ("sign_broadcast", "initial-sign broadcast", INK, DOTTED),
)
RULE = {key: (label, colour, dash) for key, label, colour, dash in RULES}

# the four panels: (task, optimizer, rate-view column of all_curves.csv)
PANELS = {
    "A": ("nested", "adam", "selected_rate"),
    "B": ("matching", "sgd", "common_rate"),
    "C": ("quartet", "sgd", "selected_rate"),
    "D": ("quartet", "sgd", "common_rate"),
}
TITLES = {
    "A": "Nested targets, Adam, selected rates",
    "B": "Pairwise targets, SGD, common rate 0.03",
    "C": "Quartic targets, SGD, selected rates",
    "D": "Quartic targets, SGD, common rate 0.03",
}
# the rate every rule uses in every panel (protocol_freeze.json records)
RATES = {
    "A": {"exact": 0.003, "unit_broadcast": 0.003, "calibrated_broadcast": 0.01, "sign_broadcast": 0.01},
    "B": {"exact": 0.03, "unit_broadcast": 0.03, "calibrated_broadcast": 0.03, "sign_broadcast": 0.03},
    "C": {"exact": 0.03, "unit_broadcast": 0.01, "calibrated_broadcast": 0.03, "sign_broadcast": 0.01},
    "D": {"exact": 0.03, "unit_broadcast": 0.03, "calibrated_broadcast": 0.03, "sign_broadcast": 0.03},
}
# D draws only the rules whose runs differ from C's
DRAWN = {p: tuple(k for k, *_ in RULES) for p in "ABC"}
DRAWN["D"] = ("unit_broadcast", "sign_broadcast")

CAP = 1024
FINAL = 16384
XTICKS = (64, 256, 1024, 4096, 16384)
XLIM = (64 / 1.10, FINAL * 1.14)
BAND_ALPHA = 0.14
SEED_LW = LW_HAIR
SEED_ALPHA_LINE = 0.32
BOOT_DRAWS = 10000
BOOT_SEED = 210999


def csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fmt_step(s):
    return f"{s:,}"


def boot_index():
    """The study's whole-seed bootstrap (protocol_freeze.json 'uncertainty'):
    10,000 draws of 20 seeds from ``default_rng(210999)``, one index matrix
    for every condition and checkpoint."""
    rng = np.random.default_rng(BOOT_SEED)
    return rng.integers(20, size=(BOOT_DRAWS, 20))


def series(curves, panel, rule, steps):
    """Per-seed matrix (20 x len(steps)) of test NMSE for one panel series."""
    task, opt, view = PANELS[panel]
    g = curves[curves.task.eq(task) & curves.optimizer.eq(opt) & curves.rule.eq(rule)
               & curves[view].eq(True)]
    assert g.seed.nunique() == 20 and len(g) == 20 * 67, (panel, rule, len(g))
    rates = g.rate.unique()
    assert len(rates) == 1 and float(rates[0]) == RATES[panel][rule], (panel, rule, rates)
    piv = g.pivot(index="seed", columns="step", values="test_nmse").sort_index()
    assert list(piv.columns) == sorted(piv.columns) and piv.shape == (20, 67)
    return piv[list(steps)].to_numpy(float)


def draw_series(ax, steps, seeds, mean, lo, hi, rule, *, zorder):
    label, colour, dash = RULE[rule]
    x = np.asarray(steps, float)
    ax.fill_between(x, lo, hi, facecolor=colour, alpha=BAND_ALPHA, lw=0, zorder=1.2)
    for row in seeds:                                 # the fan: one hairline per seed
        ax.plot(x, row, color=colour, lw=SEED_LW, alpha=SEED_ALPHA_LINE, zorder=1.6,
                solid_capstyle="butt")
    if dash is None:
        ax.plot(x, mean, color=colour, lw=LW_DATA, zorder=zorder)
    else:
        ax.plot(x, mean, color=colour, lw=LW_DATA, dashes=dash, zorder=zorder)


def floor_rule(ax, floor, xlim, *, dy_pt):
    """The noise-only NMSE as a light grey solid rule over the whole update
    range, its value printed under its left end (``dy_pt`` below the rule)."""
    ax.plot(list(xlim), [floor, floor], color=FLOOR_GREY, lw=LW_REF, zorder=1.0,
            solid_capstyle="butt")
    ax.annotate(f"noise-only NMSE {floor:g}", xy=(xlim[0] * 1.06, floor), xycoords="data",
                xytext=(0.0, -dy_pt), textcoords="offset points", ha="left", va="top",
                fontsize=PT_BASE, color=FLOOR_GREY, zorder=5)


def cap_rule(ax, ylim):
    ax.plot([CAP, CAP], list(ylim), color=MUTE, lw=LW_REF, dashes=CAP_DASH, zorder=1.0)


def note(ax, x, y, text, colour, *, ha="left", va="top"):
    ax.annotate(text, xy=(x, y), xycoords="data", ha=ha, va=va, fontsize=PT_BASE,
                color=colour, linespacing=1.15, zorder=6)


def style_axes(ax, ylim, yticks, ylabels):
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ticks before limits: set_yticks widens the view to include every tick
    ax.set_xticks(list(XTICKS), [fmt_step(s) for s in XTICKS])
    ax.set_yticks(yticks, ylabels)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*ylim)
    assert ax.get_ylim() == ylim and all(ylim[0] < t < ylim[1] for t in yticks), (ylim, yticks)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("training updates")
    ax.set_ylabel("test NMSE")


def build(path: Path = OUT):
    table = csv(SOURCE / "figures" / "controls_figure_source.csv")
    curves = csv(SOURCE / "summaries" / "all_curves.csv")
    floors = csv(BRIDGE / "summaries" / "task_variance_and_noise_floor.csv")
    assert len(curves) == 48240 and curves.seed.nunique() == 20
    assert set(curves.phase) == {"fresh"} and set(curves.model) == {"algebraic"}
    steps = sorted(table.step.unique())
    assert len(steps) == 64 and steps[0] == 64 and steps[-1] == FINAL and CAP in steps
    idx = boot_index()
    floor_of = {}
    for _, r in floors[floors.model.eq("algebraic")].iterrows():
        np.testing.assert_allclose(r.expected_label_noise_nmse,
                                   r.label_noise_variance / r.clean_target_variance, rtol=0, atol=1e-15)
        floor_of[r.task] = float(r.expected_label_noise_nmse)
    assert floor_of == {"matching": 0.0225, "quartet": 0.045, "nested": 0.0225}, floor_of

    # ── recompute every plotted mean and interval from the per-seed rows ──
    data = {}
    worst = 0.0
    for panel, (task, opt, view) in PANELS.items():
        for rule, *_ in RULES:
            seeds = series(curves, panel, rule, steps)
            mean = seeds.mean(axis=0)
            draws = seeds[idx].mean(axis=1)                      # (10000, 64)
            lo = np.quantile(draws, 0.025, axis=0)
            hi = np.quantile(draws, 0.975, axis=0)
            t = table[table.panel.eq(panel) & table.rule.eq(rule)].sort_values("step")
            assert list(t.step) == steps and t.task.eq(task).all() and t.optimizer.eq(opt).all()
            assert t.rate_view.eq(view).all()
            np.testing.assert_allclose(mean, t["mean"].to_numpy(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(lo, t.ci95_low.to_numpy(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(hi, t.ci95_high.to_numpy(), rtol=0, atol=1e-12)
            worst = max(worst, np.abs(mean - t["mean"].to_numpy()).max(),
                        np.abs(lo - t.ci95_low.to_numpy()).max(), np.abs(hi - t.ci95_high.to_numpy()).max())
            data[(panel, rule)] = (seeds, mean, lo, hi)
    print(f"[tables] 16 series x 64 checkpoints recomputed from all_curves.csv "
          f"(20 seeds, {BOOT_DRAWS} draws, seed {BOOT_SEED}); max |error| vs "
          f"controls_figure_source.csv {worst:.2e}")
    # the C/D identity the caption states: same runs, same rows
    for rule in ("exact", "calibrated_broadcast"):
        np.testing.assert_array_equal(data[("C", rule)][0], data[("D", rule)][0])
        tc = table[table.panel.eq("C") & table.rule.eq(rule)].sort_values("step")
        td = table[table.panel.eq("D") & table.rule.eq(rule)].sort_values("step")
        np.testing.assert_array_equal(tc[["mean", "ci95_low", "ci95_high"]].to_numpy(),
                                      td[["mean", "ci95_low", "ci95_high"]].to_numpy())
    for rule in ("unit_broadcast", "sign_broadcast"):
        assert not np.array_equal(data[("C", rule)][0], data[("D", rule)][0])

    # ── the printed counts and statements ────────────────────────────────
    j_cap = steps.index(CAP)
    j_end = steps.index(FINAL)
    # A: the three broadcast controls after 1,024 updates
    ctrl = ("unit_broadcast", "calibrated_broadcast", "sign_broadcast")
    means_a = np.array([data[("A", r)][1] for r in ctrl])
    spread_a = 10 ** np.ptp(np.log10(means_a[:, j_cap:]), axis=0).max() - 1.0
    assert 0.05 < spread_a < 0.06, spread_a                     # "within 6 %"
    # B: unit broadcast is 18 seeds at the floor and 2 divergent; initial profile 5 above 0.05
    unit_b = data[("B", "unit_broadcast")][0][:, j_end]
    n_unit_div = int((unit_b > 1.0).sum())
    assert n_unit_div == 2 and sorted(unit_b[unit_b > 1.0].round(2)) == [2.28, 4.16], unit_b
    assert abs(np.median(unit_b) - 0.0230) < 5e-4 and abs(unit_b.mean() - 0.3425) < 5e-4
    cal_b = data[("B", "calibrated_broadcast")][0][:, j_end]
    n_cal_high = int((cal_b > 0.05).sum())
    assert n_cal_high == 5 and 0.0500 <= cal_b[cal_b > 0.05].min() and cal_b.max() < 0.125, cal_b
    sign_b = data[("B", "sign_broadcast")][0][:, j_end]
    exact_b = data[("B", "exact")][0][:, j_end]
    assert (sign_b < 0.025).all() and (exact_b < 0.025).all()
    # C: the three controls stay between 0.94 and 1.09
    means_c = np.array([data[("C", r)][1] for r in ctrl])
    assert 0.94 < means_c.min() and means_c.max() < 1.09, (means_c.min(), means_c.max())
    # D: the divergent arms at 16,384
    unit_d = data[("D", "unit_broadcast")][0][:, j_end]
    sign_d = data[("D", "sign_broadcast")][0][:, j_end]
    n_unit_d = int((unit_d > 1.0).sum())
    n_sign_d = int((sign_d > 1.0).sum())
    assert n_unit_d == 18 and n_sign_d == 15, (n_unit_d, n_sign_d)
    assert abs(unit_d.max() - 64.66) < 5e-3 and abs(sign_d.max() - 71.63) < 5e-3
    print(f"[A] broadcast controls after {CAP:,}: max spread {100 * spread_a:.1f} %")
    print(f"[B] unit broadcast at {FINAL:,}: mean {unit_b.mean():.4f}, median {np.median(unit_b):.4f}, "
          f"{n_unit_div}/20 seeds > 1 ({np.sort(unit_b)[-2:].round(2)}); initial profile "
          f"{n_cal_high}/20 seeds > 0.05 ({cal_b[cal_b > 0.05].min():.4f}-{cal_b.max():.4f})")
    print(f"[C] broadcast-control means {means_c.min():.4f}-{means_c.max():.4f}")
    print(f"[D] at {FINAL:,}: unit broadcast {n_unit_d}/20 seeds > 1 (max {unit_d.max():.2f}), "
          f"initial sign {n_sign_d}/20 seeds > 1 (max {sign_d.max():.2f})")

    # ── axes limits, tight to what each panel draws ──────────────────────
    lim = {}
    for panel in PANELS:
        allv = np.concatenate([data[(panel, r)][0].ravel() for r in DRAWN[panel]]
                              + [data[(panel, r)][2] for r in DRAWN[panel]]
                              + [data[(panel, r)][3] for r in DRAWN[panel]])
        lim[panel] = (float(allv.min()), float(allv.max()))
    floor = {p: floor_of[PANELS[p][0]] for p in PANELS}
    ylim = {}
    for panel in "ABC":
        assert lim[panel][0] > floor[panel] / 1.06                 # seeds sit at the floor
        # room for the floor label below (under the 0.02 grid rule in A and B)
        # and, in C, for the control note above
        ylim[panel] = (floor[panel] / (1.42 if panel == "C" else 1.62),
                       lim[panel][1] * (1.32 if panel == "C" else 1.25))
    assert lim["D"][0] > 0.3 and lim["D"][1] < 90.0, lim["D"]
    ylim["D"] = (0.28, 95.0)
    print("[axes] data ranges " + "; ".join(f"{p} {lo:.4f}-{hi:.3f}" for p, (lo, hi) in lim.items()))

    # ── the canvas ───────────────────────────────────────────────────────
    cv = NativeCanvas(CANVAS_H_PT / 72.0, 2, hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
                      margins=MARGINS, letter_clearance=True)
    axes = {}
    for i, panel in enumerate("ABCD"):
        axes[panel] = cv.panel(panel, i // 2, 6 * (i % 2), 6, grid="y", title=TITLES[panel])
        cv.declare_reserve(panel, left=14.0, right=8.0)
    # C has no tick within its bottom strip: a 0.05 tick 4 pt above the 0.045
    # floor rule would read as a second reference; the printed floor value
    # names the strip instead
    yt = {"A": ([0.02, 0.1, 1.0], ["0.02", "0.1", "1"]), "C": ([0.1, 1.0], ["0.1", "1"])}
    yt["B"] = ([0.02, 0.1, 1.0, 5.0], ["0.02", "0.1", "1", "5"])
    yt["D"] = ([0.5, 1.0, 10.0, 50.0], ["0.5", "1", "10", "50"])
    for panel in "ABCD":
        ax = axes[panel]
        style_axes(ax, ylim[panel], *yt[panel])
        cap_rule(ax, ylim[panel])
        if panel != "D":
            floor_rule(ax, floor[panel], XLIM, dy_pt=3.0 if panel == "C" else 7.5)
        # draw order: the sign series last so its dots sit over the blue it coincides with
        for z, rule in enumerate(DRAWN[panel]):
            draw_series(ax, steps, *data[(panel, rule)], rule, zorder=3.0 + 0.1 * z)

    # ── in-panel statements (each a count with its denominator or a range) ──
    a, b, c, d = (axes[p] for p in "ABCD")
    note(a, 1350.0, 0.30, f"3 broadcast controls within {100 * spread_a:.0f} %\nafter {CAP:,} updates",
         MUTE)
    # B's and D's notes stand left of the 1,024 cap, above every seed trace
    note(b, 72.0, ylim["B"][1] / 1.10, f"unit broadcast:\n{n_unit_div}/20 seeds end above 1",
         COLORS["local"])
    note(b, 72.0, ylim["B"][1] / 2.05, f"initial profile:\n{n_cal_high}/20 seeds end above 0.05",
         COLORS["additive"])
    # C's note sits left of the 1,024 cap and above the controls' seed fans
    note(c, 76.0, ylim["C"][1] / 1.05, f"3 broadcast controls\n{means_c.min():.2f}–{means_c.max():.2f}, none diverges",
         MUTE)
    note(d, 72.0, ylim["D"][1] / 1.10, f"unit broadcast:\n{n_unit_d}/20 seeds end above 1",
         COLORS["local"])
    note(d, 72.0, ylim["D"][1] / 2.05, f"initial sign:\n{n_sign_d}/20 seeds end above 1", INK)
    note(d, 72.0, ylim["D"][1] / 3.85, "exact path and initial profile:\nrate 0.03 in C and D\n(same runs), drawn in C",
         MUTE)

    # ── one shared key ───────────────────────────────────────────────────
    handles = []
    for _, label, colour, dash in RULES:
        h = Line2D([], [], color=colour, lw=LW_DATA, label=label)
        if dash is not None:
            h.set_dashes(dash)
        handles.append(h)
    handles.append(Line2D([], [], color=FLOOR_GREY, lw=LW_REF, label="noise-only NMSE (value printed)"))
    cap_h = Line2D([], [], color=MUTE, lw=LW_REF, label=f"original {CAP:,}-update cap")
    cap_h.set_dashes(CAP_DASH)
    handles.append(cap_h)
    cv.fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0), ncol=3,
                  frameon=False, fontsize=PT_BASE, handlelength=2.8, columnspacing=1.8,
                  handletextpad=0.6, borderaxespad=0.5, labelspacing=0.45)
    problems = cv.save(path, name="figure_credit_optimizer_controls_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


CANVAS_H_PT = 470.0
HGUTTER_PT = 34.0
VGUTTER_PT = 40.0
MARGINS = Margins(left=44.0, right=12.0, top=20.0, bottom=58.0)


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

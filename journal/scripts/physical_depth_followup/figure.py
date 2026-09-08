#!/usr/bin/env python3
"""Main Fig. 6 (fig:physicaldepth): physical depth, alignment and training budget.

Grid (DESIGN_SPEC §6 + errata): ``NativeCanvas`` 492 pt, three rows of
117 / 108 / 108 pt, hgutter 40, vgutter 48, margins 46 / 16 / 25 / 38 (the
spec's 485-pt 118/112/112 grid with 36/40 gutters and a 50-pt left margin
left 5.8 pt of ink-free band between rows -- the audit floor is 8.5 pt
under an x-labelled row -- and could not hold a 30-pt y-decoration plus a
panel letter in one gutter; the rows stay inside D2's 104-136 pt band).  Every locked panel declares the same
6-pt left reserve so the three 4-module columns keep one axes width and
every letter clears its left neighbour.
Row 0 is the schematic row -- A (nested gain task card) and B (D1 [8]
versus D3 [2,1,2] stage pair) -- beside C, the four-tier depth ladder.
Row 1 holds D | E | F (4 / 4 / 4 modules) on one shared y axis (test
accuracy, 48-100 %); row 2 holds G | H | I (4 / 4 / 4) on one shared x
axis (epoch, 0-600 with a label rail).

Waiver (spec D3): D/E/F are one comparison family sharing y and G/H/I one
family sharing x, column-locked at modules 0, 4 and 8 (errata #3).

Deviations from the spec's panel table (each reported in TEXT.md):
* Row 0 is 3 / 4 / 5 modules instead of 3 / 3 / 6.  The glyph library
  sizes the stage pair at >= 4 modules before the D3 tier tags fit
  (GLYPH_LIBRARY_REPORT, sizing notes); at 3 modules the two eight-
  compartment trees would sit in 40-pt cards with a 4-pt ring pitch.
* ``journal_style.ORDINAL_RAMP`` (D8) is not defined yet, so the three
  depth tints are a private constant with the spec's recipe.
* ``native_schematics.draw_stage_pair`` draws a shared/exact rule key and
  no excitatory contacts, which the Fig. 6 brief forbids / requires, so B
  is composed here from the same primitives (``Frame.stage_tree`` with the
  [8] and [2,1,2] fans, ``Frame.contact``, ``Frame.soma``, ``Frame.error_in``,
  ``Frame.task_card``, ``Frame.footer``).
* C and D print means with 95 % intervals only (no seed fans): C stacks six
  series and D three series within 5 pp of one another at most x positions,
  where 10-seed fans pile into unreadable clusters; F keeps the seed fans.
* E carries a 2 x 2 key because it introduces the D1 exact-BP trajectory;
  C's key names the four-tier series.
* C has no x-axis label: its D1-D4 tick labels name the depth, and the label
  band would leave < 8.5 pt between rows 0 and 1.
* Panel titles are shortened to the <= 26-character budget of a 4-5 module
  panel (spec rule 0.4.5); they state the same finding as the spec's longer
  panel-table titles (e.g. I: 'No plateau by epoch 600').
* Chance / zero is labelled once per shared-axis row (D in row 1), the
  idiom the spec states for Fig. 2 F-H; the dashed reference is drawn in
  every data panel.

Every printed number is derived from the frozen Source Data named in the
provenance record and re-checked against the independent summary tables;
this builder performs no fitting and writes nothing under the frozen
study directories except its own ``figures/`` outputs and the caption.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0]))
import analyze  # noqa: E402
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR,  # noqa: E402
                           LW_REF, MARKER_MS, PT_ANNOT, PT_LEGEND, PT_SMALL,
                           SEED_ALPHA, SEED_MS, Margins, NativeCanvas,
                           style_panel)
from journal_style import label_color  # noqa: E402
from credit_tree_schematics import mix  # noqa: E402
from native_schematics import (JUNCTION_R_PT, Frame, _text_w_pt,  # noqa: E402
                               reference_line)

J = analyze.J
OUT = analyze.OUT                      # source_data/physical_depth_followup
BUDGET = analyze.SRC                   # source_data/physical_depth_budget/canonical
H4 = J / "source_data/physical_depth_h4_factorial"
DOSE = J / "source_data/physical_alignment_dose"
DEST = OUT / "figures"
INK, MUTE = COLORS["ink"], COLORS["mute"]
DASH = (0, (2.2, 1.8))
DOT = (0, (1.0, 1.6))
# Spec D8 ordinal ramp (journal_style.ORDINAL_RAMP is not defined yet).
ORDINAL_RAMP = [mix("edge", 45), mix("edge", 75), COLORS["edge"]]
H4_SEEDS = list(range(10400, 10410))
DOSE_SEEDS = list(range(10200, 10210))

# C: four-tier factorial series (label <= 22 characters for the key).
H4_SERIES = [
    # key, key label, (regime, architecture, mechanism, credit), colour, ls, marker
    ("exact_bp", "exact BP", ("aligned", "serial_tree", "shunting", "full_bp"), "ink", DASH, "o"),
    ("exact_path", "exact path", ("aligned", "serial_tree", "shunting", "local_path"), "bp", "-", "o"),
    ("shared_soma", "shared soma", ("aligned", "serial_tree", "shunting", "local_shared"), "scalar", "-", "o"),
    ("grouped_point", "grouped point", ("aligned", "grouped_point", "shunting", "full_bp"), "point_mlp", "-", "o"),
    ("raw_additive", "raw additive", ("aligned", "serial_tree", "raw_additive", "full_bp"), "additive", DOT, "s"),
    ("reversed", "reversed", ("rewired_tree", "serial_tree", "shunting", "full_bp"), "mute", DOT, "o"),
]
# E / F / I: the six 600-epoch conditions.  ``open`` marks broadcast autograd.
COND = {
    ("exact_autograd_bp_recipe", 1): dict(label="D1 exact BP", recipe="BP", color="point_mlp", ls=DASH, open=False),
    ("exact_autograd_bp_recipe", 3): dict(label="D3 exact BP", recipe="BP", color="ink", ls=DASH, open=False),
    ("broadcast_autograd_bp_recipe", 3): dict(label="broadcast autograd", recipe="BP", color="scalar", ls=DOT, open=True),
    ("broadcast_autograd_localca_recipe", 3): dict(label="broadcast autograd", recipe="LocalCA", color="scalar", ls=DOT, open=True),
    ("per_soma_shared", 3): dict(label="shared soma", recipe="LocalCA", color="scalar", ls="-", open=False),
    ("path_transport", 3): dict(label="exact path", recipe="LocalCA", color="bp", ls="-", open=False),
}
F_ORDER = [("exact_autograd_bp_recipe", 1), ("exact_autograd_bp_recipe", 3), ("broadcast_autograd_bp_recipe", 3),
           ("broadcast_autograd_localca_recipe", 3), ("per_soma_shared", 3), ("path_transport", 3)]
E_ORDER = [("exact_autograd_bp_recipe", 1), ("exact_autograd_bp_recipe", 3), ("path_transport", 3), ("per_soma_shared", 3)]


def signed(value, decimals=2):
    return f"{value:+.{decimals}f}".replace("-", "−")


def plain(value, decimals=2):
    return f"{value:.{decimals}f}".replace("-", "−")


def read(path):
    return pd.read_csv(path, float_precision="round_trip")


# ── private glyph helpers (library geometry; errata #7) ─────────────────
def chain(f, xy, parts, *, size=PT_ANNOT, color=None, ha="left", va="center", zorder=6):
    """Plain strings and (base, sub) pairs laid out left-to-right (token subscripts)."""
    color = INK if color is None else color
    base_size = PT_ANNOT if size <= PT_SMALL else size
    widths = []
    for p in parts:
        if isinstance(p, tuple):
            widths.append(_text_w_pt(f.ax, p[0], base_size) + 0.4 + _text_w_pt(f.ax, p[1], PT_SMALL))
        else:
            widths.append(_text_w_pt(f.ax, p, size))
    total = sum(widths)
    x, y = xy
    if ha == "center":
        x -= f.fx(total / 2.0)
    elif ha == "right":
        x -= f.fx(total)
    for p, w in zip(parts, widths):
        if isinstance(p, tuple):
            f.subscript((x, y), p[0], p[1], size=size, color=color, ha="left", va=va, zorder=zorder)
        else:
            f.text((x, y), p, size=size, color=color, ha="left", va=va, zorder=zorder)
        x += f.fx(w)
    return total


def bracket(f, x0_pt, x1_pt, y_pt, *, tick_pt=2.6):
    """Mute hairline bracket over [x0, x1] at height y (points), ticks down."""
    X, Y = f.fx, f.fy
    f.rule(Y(y_pt), X(x0_pt), X(x1_pt))
    for x in (x0_pt, x1_pt):
        f.ax.plot([X(x), X(x)], [Y(y_pt), Y(y_pt - tick_pt)], color=MUTE, lw=f.lw(LW_HAIR),
                  solid_capstyle="round", zorder=1.2)


def lerp(a, b, t):
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


# ── A: nested gain task ───────────────────────────────────────────────────
def nested_gain_task(ax):
    """Three bracket tiers over eight input slots, sensors per tier, the gain product."""
    f = Frame(ax)
    X, Y = f.fx, f.fy
    inh_text = label_color(COLORS["inh"])
    f.task_card((0.0, 0.0, 1.0, 1.0))
    slots = [9.0 + 8.6 * i for i in range(8)]
    sensor_x = 78.5
    tiers = (("global ×1", [(0, 7)], 105.0),
             ("coarse ×2", [(0, 3), (4, 7)], 90.5),
             ("fine ×4", [(0, 1), (2, 3), (4, 5), (6, 7)], 76.0))
    mid = (slots[0] + slots[-1]) / 2.0
    for name, groups, y in tiers:
        for a, b in groups:
            bracket(f, slots[a] - 3.2, slots[b] + 3.2, y)
        f.text((X(mid), Y(y + 6.2)), name, size=PT_SMALL, color=MUTE)
        f.contact((X(sensor_x), Y(y)), kind="inh")
    for x in slots:
        f.contact((X(x), Y(64.5)), kind="exc")
    f.text((X(sensor_x), Y(64.5)), "h", size=PT_SMALL, color=inh_text)
    tag = "class signal m × gains"
    if not f._fits(tag, PT_SMALL, f.w_pt - 6.0):
        tag = "signal m × gains"
    f.text((X(mid), Y(54.0)), tag, size=PT_SMALL, color=INK)
    f.text((X(mid), Y(40.0)), "sensors report h", size=PT_SMALL, color=inh_text)
    f.badge((X(f.w_pt - 5.0), Y(40.0)), "control", text="α", ha="right", va="center")
    chain(f, (0.5, Y(13.0)), [("x", "E"), " = m ", ("h", "f"), " ", ("h", "c"), " ", ("h", "g")],
          size=PT_SMALL, color=INK, ha="center")
    return {"slots": 8, "tiers": {"global": 1, "coarse": 2, "fine": 4}}


# ── B: D1 [8] versus D3 [2,1,2] ───────────────────────────────────────────
def stage_pair(ax):
    """Same eight compartments in one fan versus three serial stages (hero)."""
    f = Frame(ax)
    X, Y = f.fx, f.fy
    inh_text = label_color(COLORS["inh"])
    foot = f.footer("contacts and parameters matched")
    gap_pt, d1_w_pt = 8.0, 50.0
    d3_w_pt = f.w_pt - d1_w_pt - gap_pt
    y0, h = Y(foot), 1.0 - Y(foot)
    cells = [(0.0, y0, X(d1_w_pt), h), (X(d1_w_pt + gap_pt), y0, X(d3_w_pt), h)]
    specs = (("D1  [8]", [8], False, 40.0, ("all tiers",)),
             ("D3  [2,1,2]", [2, 1, 2], True, 60.0, ("global", "coarse", "fine")))
    drawn = {}
    for cell, (title, stages, hero, height_pt, tags) in zip(cells, specs):
        core = f.task_card(cell, title=title, emphasis=hero)
        cx0, cy0, cw, ch = core
        w_pt = cw * f.w_pt
        if len(stages) == 1:
            tree_w_pt = w_pt - 12.0
            centre = cx0 + X(w_pt / 2.0 - 2.0)
        else:
            tag_w = max(_text_w_pt(ax, t, PT_SMALL) for t in tags)
            tree_w_pt = w_pt - 8.0 - tag_w - 10.0
            centre = cx0 + X(7.0 + tree_w_pt / 2.0)
        base = (centre, cy0 + Y(16.0))
        tree = f.stage_tree(base, Y(height_pt), len(stages), branching=stages,
                            width=X(tree_w_pt), rings=True)
        f.soma(base, output=11.0, label="y")
        f.error_in(base, side="right")
        # inhibitory sensors on every incoming segment (staggered where dense)
        for s, comps in enumerate(tree):
            parents = [base] if s == 0 else tree[s - 1]
            fan = max(int(stages[s]), 1)
            for i, child in enumerate(comps):
                parent = parents[i // fan]
                if len(stages) == 1:
                    t = (0.36, 0.56, 0.76)[i % 3]
                else:
                    t = 0.55 if fan == 1 else (0.42, 0.70)[i % 2]
                f.contact(lerp(parent, child, t), kind="inh")
        # excitatory contacts on the distal compartments (short terminal stubs)
        distal = tree[-1]
        for px, py in distal:
            f.dendrite((px, py + Y(JUNCTION_R_PT)), (px, py + Y(6.5)), level=3)
            f.contact((px, py + Y(8.0)), kind="exc")
        top = max(p[1] for p in distal) + Y(8.0)
        if len(stages) == 1:
            f.text((centre, top + Y(6.5)), tags[0], size=PT_SMALL, color=inh_text)
        else:
            for comps, tag in zip(tree, tags):
                right = max(p[0] for p in comps)
                f.text((right + X(JUNCTION_R_PT + 4.5), comps[0][1]), tag, size=PT_SMALL,
                       color=inh_text, ha="left")
        drawn[title.split()[0]] = dict(stages=stages, compartments=sum(len(c) for c in tree),
                                       tree_width_pt=round(tree_w_pt, 1), height_pt=height_pt)
    return drawn


# ── data loaders (assertions on the frozen Source Data) ──────────────────
def load_h4():
    cs = read(H4 / "condition_summary.csv")
    pc = read(H4 / "paired_contrasts.csv")
    so = read(H4 / "seed_outcomes.csv")
    assert len(so) == 360 and sorted(so.seed.unique()) == H4_SEEDS
    rows = []
    for key, label, (regime, arch, mech, credit), *_ in H4_SERIES:
        for depth in (1, 2, 3, 4):
            c = cs[cs.regime.eq(regime) & cs.architecture.eq(arch) & cs.mechanism.eq(mech)
                   & cs.credit.eq(credit) & cs.depth.eq(depth)]
            s = so[so.regime.eq(regime) & so.architecture.eq(arch) & so.mechanism.eq(mech)
                   & so.credit.eq(credit) & so.depth.eq(depth)]
            assert len(c) == 1 and int(c.iloc[0].n_seeds) == 10 and len(s) == 10, (key, depth)
            assert sorted(s.seed) == H4_SEEDS
            assert abs(s.test_accuracy.mean() - c.iloc[0].mean_test_accuracy) < 1e-9
            r = c.iloc[0]
            rows.append(dict(series=key, label=label, regime=regime, architecture=arch, mechanism=mech,
                             credit=credit, depth=depth, n_seeds=10, mean=float(r.mean_test_accuracy),
                             ci_low=float(r.ci_low), ci_high=float(r.ci_high)))
    d43 = pc[pc.contrast.eq("depth__serial_bp__aligned__d4_d3")].iloc[0]
    seeds = np.array([float(v) for v in str(d43.seed_values_pp).split(";")])
    assert len(seeds) == 10 and abs(seeds.mean() - d43.mean_pp) < 1e-6 and int(d43.negative_pairs) == 10
    contrast = dict(contrast=str(d43.contrast), mean_pp=float(d43.mean_pp), ci_low_pp=float(d43.ci_low_pp),
                    ci_high_pp=float(d43.ci_high_pp), positive_pairs=int(d43.positive_pairs),
                    negative_pairs=int(d43.negative_pairs), n_seeds=int(d43.n_seeds))
    return pd.DataFrame(rows), contrast


def load_dose():
    cs = read(DOSE / "condition_summary.csv")
    pc = read(DOSE / "paired_contrasts.csv")
    so = read(DOSE / "combined_seed_outcomes.csv")
    assert len(cs) == 15 and len(so) == 150
    alphas = sorted(cs.alignment_alpha.unique())
    assert alphas == [0.0, 0.25, 0.5, 0.75, 1.0]
    rows, gains = [], []
    for alpha in alphas:
        for depth in (1, 2, 3):
            c = cs[cs.alignment_alpha.eq(alpha) & cs.depth.eq(depth)]
            s = so[so.alignment_alpha.eq(alpha) & so.depth.eq(depth)]
            assert len(c) == 1 and int(c.iloc[0].n_seeds) == 10 and len(s) == 10
            assert sorted(s.seed) == DOSE_SEEDS
            assert abs(s.test_accuracy.mean() - c.iloc[0].mean_test_accuracy) < 1e-9
            r = c.iloc[0]
            rows.append(dict(alignment_alpha=alpha, depth=depth, n_seeds=10, mean=float(r.mean_test_accuracy),
                             ci_low=float(r.ci95_low_test_accuracy), ci_high=float(r.ci95_high_test_accuracy)))
        g = pc[pc.estimand.eq(f"depth_effect_alpha_{alpha:.2f}")].iloc[0]
        p = so[so.alignment_alpha.eq(alpha)].pivot(index="seed", columns="depth", values="test_accuracy")
        assert abs((p[3] - p[1]).mean() - g.mean_difference) < 1e-9 and int(g.n_pairs) == 10
        assert int((p[3] - p[1] > 0).sum()) == int(g.positive_pairs)
        gains.append(dict(alignment_alpha=alpha, estimand=str(g.estimand), mean_pp=100 * float(g.mean_difference),
                          ci_low_pp=100 * float(g.ci95_low), ci_high_pp=100 * float(g.ci95_high),
                          positive_pairs=int(g.positive_pairs), n_pairs=10))
    return pd.DataFrame(rows), pd.DataFrame(gains)


def load_budget():
    valid = json.loads((OUT / "analysis_validation.json").read_text())
    assert valid["status"] == "passed"
    s = read(OUT / "condition_trajectory_summary.csv")
    g = read(OUT / "paired_trajectory_summary.csv")
    end = read(BUDGET / "extension_endpoints.csv")
    es = read(BUDGET / "extension_summary.csv")
    lc = read(BUDGET / "extension_loss_contrasts.csv")
    psc = read(BUDGET / "extension_paired_seed_contrasts.csv")
    stops = read(OUT / "stopping_by_seed.csv")
    traj = read(OUT / "validation_selected_seed_trajectories.csv")
    assert len(end) == 120 and len(stops) == 60 and len(traj) == 36000
    for (arm, depth) in COND:
        for budget in (180, 600):
            e = end[end.arm.eq(arm) & end.depth.eq(depth) & end.budget.eq(budget)]
            m = s[s.arm.eq(arm) & s.depth.eq(depth) & s.metric.eq("test_accuracy") & s.epoch.eq(budget)].iloc[0]
            r = es[es.arm.eq(arm) & es.depth.eq(depth) & es.budget.eq(budget)].iloc[0]
            assert len(e) == 10 and sorted(e.seed) == DOSE_SEEDS and int(r.n) == 10
            assert abs(e.test_accuracy.mean() - m["mean"]) < 1e-9 and abs(r.mean_test_accuracy - m["mean"]) < 1e-9
            v = s[s.arm.eq(arm) & s.depth.eq(depth) & s.metric.eq("best_validation_loss") & s.epoch.eq(budget)].iloc[0]
            assert abs(e.best_valid_loss.mean() - v["mean"]) < 1e-9
    ga = g[g.metric.eq("test_accuracy")].set_index("epoch")
    gc = g[g.metric.eq("test_cross_entropy")].set_index("epoch")
    for rec in valid["accuracy_budgets"]:
        assert abs(ga.loc[rec["epoch"], "mean"] - rec["mean"]) < 1e-9
    for rec in valid["cross_entropy_budgets"]:
        assert abs(gc.loc[rec["epoch"], "mean"] - rec["mean"]) < 1e-9
    for budget in (180, 600):
        lr = lc[lc.budget.eq(budget) & lc.metric.eq("test_loss") & lc.contrast.eq("localca_path_minus_shared")].iloc[0]
        assert abs(lr["mean"] - gc.loc[budget, "mean"]) < 1e-9
        pr = psc[psc.budget.eq(budget) & psc.contrast.eq("localca_path_minus_shared")]
        assert len(pr) == 10 and abs(pr.difference_pp.mean() - ga.loc[budget, "mean"]) < 1e-9
    d1 = stops[stops.arm.eq("exact_autograd_bp_recipe") & stops.depth.eq(1)]
    assert len(d1) == 10 and int(d1.stopped_before600.sum()) == valid["d1_stopped_before600"] == 8
    assert int(stops[stops.depth.eq(3)].reached_cap600.sum()) == 50
    # the accuracy crossing computed from the condition means must match the paired summary
    path = s[s.arm.eq("path_transport") & s.depth.eq(3) & s.metric.eq("test_accuracy")].set_index("epoch")["mean"]
    shared = s[s.arm.eq("per_soma_shared") & s.depth.eq(3) & s.metric.eq("test_accuracy")].set_index("epoch")["mean"]
    diff = (path - shared).loc[180:]
    crossing = int(diff[diff < 0].index[0])
    assert crossing == valid["accuracy_crossing"]["first_negative_after180"] == 312
    stop_pts = []
    for r in d1[d1.stopped_before600].itertuples():
        t = traj[traj.arm.eq(r.arm) & traj.depth.eq(1) & traj.seed.eq(r.seed) & traj.epoch.eq(r.epochs_run)].iloc[0]
        assert int(t.last_training_epoch) == int(r.epochs_run) < 600
        stop_pts.append(dict(seed=int(r.seed), epochs_run=int(r.epochs_run), best_validation_loss=float(t.best_validation_loss)))
    return dict(valid=valid, s=s, g=g, end=end, stops=stops, crossing=crossing, stop_pts=pd.DataFrame(stop_pts))


# ── C: four-tier depth ladder ─────────────────────────────────────────────
def depth_ladder(ax, h4, contrast):
    handles = []
    for key, label, _, cname, ls, marker in H4_SERIES:
        color = COLORS[cname]
        r = h4[h4.series.eq(key)].sort_values("depth")
        x = r.depth.to_numpy()
        y = 100 * r["mean"].to_numpy()
        lw = LW_REF if cname == "ink" else LW_DATA
        ax.plot(x, y, color=color, ls=ls, lw=lw, zorder=3)
        ax.errorbar(x, y, yerr=[y - 100 * r.ci_low.to_numpy(), 100 * r.ci_high.to_numpy() - y], fmt=marker,
                    color=color, ms=MARKER_MS, mfc="white", mew=LW_ERR, elinewidth=LW_ERR, capsize=2,
                    ls="none", zorder=4)
        handles.append(Line2D([], [], color=color, ls=ls, lw=lw, marker=marker, ms=MARKER_MS, mfc="white",
                              mew=LW_ERR, label=label))
    ax.set(xlim=(0.6, 4.4), ylim=(44, 100), xticks=[1, 2, 3, 4], xticklabels=["D1", "D2", "D3", "D4"],
           yticks=[50, 75, 100], ylabel="Test accuracy (%)")
    style_panel(ax, grid="y")
    reference_line(ax, 50, label=None, span=(0.6, 4.4))
    ax.text(4.4, 49.2, "chance", ha="right", va="top", fontsize=PT_SMALL, color=MUTE, zorder=5)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 0.995), frameon=False, fontsize=PT_LEGEND,
              handlelength=1.6, handletextpad=0.5, labelspacing=0.12, borderpad=0.1, borderaxespad=0.1)
    tag = (f"D4 − D3: {signed(contrast['mean_pp'])} pp", f"({contrast['positive_pairs']}/{contrast['n_seeds']} positive)")
    ax.text(4.4, 97.5, tag[0], ha="right", va="center", fontsize=PT_ANNOT, color=INK, zorder=6)
    ax.text(4.4, 93.0, tag[1], ha="right", va="center", fontsize=PT_ANNOT, color=INK, zorder=6)
    ax.plot([4.0, 4.0], [90.6, 100 * h4[h4.series.eq("exact_bp") & h4.depth.eq(4)].iloc[0].ci_high + 1.2],
            color=MUTE, lw=LW_HAIR, zorder=2)
    ax.text(0.68, 46.4, f"four-tier task, seeds {H4_SEEDS[0]}–{H4_SEEDS[-1]}", ha="left", va="center",
            fontsize=PT_SMALL, color=MUTE, zorder=6)
    return tag


# ── D: alignment dose ─────────────────────────────────────────────────────
def alignment_dose(ax, dose, gains):
    for depth, color in zip((1, 2, 3), ORDINAL_RAMP):
        r = dose[dose.depth.eq(depth)].sort_values("alignment_alpha")
        x = r.alignment_alpha.to_numpy()
        y = 100 * r["mean"].to_numpy()
        ax.plot(x, y, color=color, lw=LW_DATA, zorder=3)
        ax.errorbar(x, y, yerr=[y - 100 * r.ci_low.to_numpy(), 100 * r.ci_high.to_numpy() - y], fmt="o", color=color,
                    ms=MARKER_MS, mfc="white", mew=LW_ERR, elinewidth=LW_ERR, capsize=2, ls="none", zorder=4)
        ax.text(1.05, y[-1], f"D{depth}", ha="left", va="center", fontsize=PT_SMALL, color=label_color(color), zorder=6)
    top = 100 * dose.groupby("alignment_alpha").ci_high.max()
    printed = []
    for k, g in enumerate(gains.itertuples()):
        # alternate heights: the alpha pitch (~22 pt) equals a PT_ANNOT tag's width
        y = top[g.alignment_alpha] + (2.6 if k % 2 == 0 else 7.2)
        text = signed(g.mean_pp)
        # the D3 segment rises through the tag band right of alpha = 0.75, so
        # that tag ends before the curve; the alpha = 1 tag hugs the frame.
        ha = "right" if g.alignment_alpha >= 0.75 else "center"
        x = 1.02 if g.alignment_alpha >= 1.0 else (0.735 if g.alignment_alpha >= 0.75 else g.alignment_alpha)
        ax.text(x, y, text, ha=ha, va="bottom", fontsize=PT_ANNOT, color=INK, zorder=6)
        printed.append(text)
    ax.set(xlim=(-0.14, 1.21), ylim=(48, 100), xticks=[0, 0.25, 0.5, 0.75, 1], xticklabels=["0", ".25", ".5", ".75", "1"],
           yticks=[50, 75, 100], xlabel="Sensor alignment α", ylabel="Test accuracy (%)")
    style_panel(ax, grid="y")
    reference_line(ax, 50, label="chance", span=(-0.14, 1.21))
    # cohort lines in the upper-left band; the pair count goes to the empty
    # band under the flat D1 curve, clear of the steep D3 segment.
    lines = [("three-tier task, 180 epochs", 90.5), (f"seeds {DOSE_SEEDS[0]}–{DOSE_SEEDS[-1]}", 85.5),
             ("tags: D3 − D1 (pp)", 80.5), ("10/10 pairs from α = 0.5", 53.6)]
    for line, y in lines:
        ax.text(-0.11, y, line, ha="left", va="center", fontsize=PT_SMALL, color=MUTE, zorder=6)
    return printed


# ── E: trajectories ───────────────────────────────────────────────────────
def trajectories(ax, s, crossing):
    handles = []
    for (arm, depth) in E_ORDER:
        spec = COND[(arm, depth)]
        color = COLORS[spec["color"]]
        p = s[s.arm.eq(arm) & s.depth.eq(depth) & s.metric.eq("test_accuracy")].sort_values("epoch")
        lw = LW_REF if spec["color"] == "ink" else LW_DATA
        ax.fill_between(p.epoch, 100 * p.ci95_low, 100 * p.ci95_high, color=color, alpha=0.12, lw=0, zorder=1)
        ax.plot(p.epoch, 100 * p["mean"], color=color, ls=spec["ls"], lw=lw, zorder=3)
        if depth == 1:
            y_d1 = 100 * p[p.epoch.eq(400)].iloc[0]["mean"]
            ax.text(400, y_d1 - 2.4, spec["label"], ha="center", va="top", fontsize=PT_LEGEND,
                    color=label_color(color), zorder=6)
        else:
            handles.append(Line2D([], [], color=color, ls=spec["ls"], lw=lw, label=spec["label"]))
    path = s[s.arm.eq("path_transport") & s.depth.eq(3) & s.metric.eq("test_accuracy")].set_index("epoch")["mean"]
    y_cross = 100 * path.loc[crossing]
    ax.plot([crossing], [y_cross], marker="o", ms=MARKER_MS, mfc="white", mec=MUTE, mew=LW_EDGE, ls="none", zorder=5)
    ax.text(crossing + 14, y_cross - 4.0, str(crossing), ha="left", va="top", fontsize=PT_SMALL, color=MUTE, zorder=6)
    ax.plot([180, 180], [60, 100], color=MUTE, ls=":", lw=LW_REF, zorder=2)
    ax.set(xlim=(0, 640), ylim=(48, 100), xticks=[0, 180, 400, 600], yticks=[50, 75, 100], xlabel="Epoch")
    style_panel(ax, grid="y")
    reference_line(ax, 50, label=None, span=(0, 640))
    # key in the empty band above the flat D1 trajectory (epochs 200-520, 63-76 %)
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(200 / 640, 15 / 52), ncol=1, frameon=False,
              fontsize=PT_LEGEND, handlelength=1.4, handletextpad=0.4, labelspacing=0.15,
              borderpad=0.1, borderaxespad=0.0)
    return float(y_cross)


# ── F: selected states at 180 and 600 ─────────────────────────────────────
def budget_states(ax, s, end, stops):
    xs = {"BP": (0.0, 1.0), "LocalCA": (2.6, 3.6)}
    dodge = {("exact_autograd_bp_recipe", 1): -0.2, ("exact_autograd_bp_recipe", 3): 0.0,
             ("broadcast_autograd_bp_recipe", 3): 0.2, ("broadcast_autograd_localca_recipe", 3): -0.2,
             ("per_soma_shared", 3): 0.0, ("path_transport", 3): 0.2}
    fan = np.linspace(-0.07, 0.07, 10)
    rows = []
    for key in F_ORDER:
        arm, depth = key
        spec = COND[key]
        color = COLORS[spec["color"]]
        means = []
        for budget, x0 in zip((180, 600), xs[spec["recipe"]]):
            x = x0 + dodge[key]
            e = end[end.arm.eq(arm) & end.depth.eq(depth) & end.budget.eq(budget)].sort_values("seed")
            vals = 100 * e.test_accuracy.to_numpy()
            if spec["open"]:
                ax.plot(x + fan, vals, ls="none", marker="o", ms=SEED_MS, mfc="white", mec=color, mew=LW_EDGE,
                        alpha=SEED_ALPHA, zorder=2)
            else:
                ax.plot(x + fan, vals, ls="none", marker="o", ms=SEED_MS, mfc=color, mec="none", alpha=SEED_ALPHA, zorder=2)
            m = s[s.arm.eq(arm) & s.depth.eq(depth) & s.metric.eq("test_accuracy") & s.epoch.eq(budget)].iloc[0]
            y = 100 * m["mean"]
            ax.errorbar(x, y, yerr=[[y - 100 * m.ci95_low], [100 * m.ci95_high - y]], fmt="o", color=color, ms=MARKER_MS,
                        mfc="white", mew=LW_ERR, elinewidth=LW_ERR, capsize=2, ls="none", zorder=4)
            means.append((x, y))
            rows.append(dict(arm=arm, depth=depth, budget=budget, recipe=spec["recipe"], label=spec["label"],
                             mean=float(m["mean"]), ci95_low=float(m.ci95_low), ci95_high=float(m.ci95_high), n_seeds=10))
        (xa, ya), (xb, yb) = means
        lw = LW_REF if spec["color"] in ("ink", "point_mlp") else LW_EDGE
        ax.plot([xa, xb], [ya, yb], color=color, ls=spec["ls"], lw=lw, zorder=3)
    n_stop = int(stops[stops.arm.eq("exact_autograd_bp_recipe") & stops.depth.eq(1)].stopped_before600.sum())
    d1_600 = [r for r in rows if r["arm"] == "exact_autograd_bp_recipe" and r["depth"] == 1 and r["budget"] == 600][0]
    ax.text(0.75, 68.5, f"{n_stop}/10 D1 stop early", ha="center", va="center", fontsize=PT_SMALL, color=INK, zorder=6)
    ax.plot([0.8, 0.8], [100 * d1_600["ci95_high"] + 1.2, 66.2], color=MUTE, lw=LW_HAIR, zorder=2)
    for recipe, (a, b) in xs.items():
        ax.text((a + b) / 2.0, 55.0, f"{recipe} recipe", ha="center", va="center", fontsize=PT_SMALL, color=MUTE, zorder=6)
    ax.set(xlim=(-0.55, 4.15), ylim=(48, 100), xticks=[0, 1, 2.6, 3.6], xticklabels=["180", "600", "180", "600"],
           yticks=[50, 75, 100], xlabel="Budget (epochs)")
    style_panel(ax, grid="y")
    # chance is labelled once for the shared-y row, in D; the label would sit
    # under the 'LocalCA recipe' tag here.
    reference_line(ax, 50, label=None, span=(-0.55, 4.15))
    return pd.DataFrame(rows), n_stop


# ── G / H: paired gaps ────────────────────────────────────────────────────
def gap_panel(ax, g, metric, *, ylabel, ylim, yticks, yticklabels, xlim):
    p = g[g.metric.eq(metric)].sort_values("epoch")
    ax.fill_between(p.epoch, p.ci95_low, p.ci95_high, color=COLORS["bp"], alpha=0.14, lw=0, zorder=1)
    ax.plot(p.epoch, p["mean"], color=COLORS["bp"], lw=LW_DATA, zorder=3)
    ax.axvline(180, color=MUTE, ls=":", lw=LW_REF, zorder=2)
    ax.set(xlim=xlim, ylim=ylim, xticks=[0, 180, 400, 600], yticks=yticks, xlabel="Epoch", ylabel=ylabel)
    ax.set_yticklabels(yticklabels)
    style_panel(ax, grid="y")
    reference_line(ax, 0, label=None, span=xlim)
    return p.set_index("epoch")


def accuracy_gap(ax, g, valid, xlim):
    p = gap_panel(ax, g, "test_accuracy", ylabel="Exact − shared accuracy (pp)", ylim=(-6, 15.5),
                  yticks=[-5, 0, 5, 10, 15], yticklabels=["−5", "0", "5", "10", "15"], xlim=xlim)
    crossing = valid["accuracy_crossing"]["first_negative_after180"]
    lowest = int(valid["accuracy_most_negative_epoch"])
    marks = {}
    for epoch in (180, lowest, 600):
        r = p.loc[epoch]
        ax.plot([epoch], [r["mean"]], marker="o", ms=MARKER_MS, mfc="white", mec=COLORS["bp"], mew=LW_ERR, ls="none", zorder=5)
        marks[epoch] = signed(r["mean"])
    ax.text(205, 12.6, marks[180], ha="left", va="center", fontsize=PT_ANNOT, color=INK, zorder=6)
    ax.plot([184, 203], [p.loc[180, "mean"] + 0.5, 12.1], color=MUTE, lw=LW_HAIR, zorder=2)
    ax.text(lowest, p.loc[lowest, "ci95_low"] - 0.9, marks[lowest], ha="center", va="top", fontsize=PT_ANNOT, color=INK, zorder=6)
    ax.text(610, p.loc[600, "mean"], marks[600], ha="left", va="center", fontsize=PT_ANNOT, color=INK, zorder=6)
    ax.text(crossing + 16, 2.6, f"{crossing}: first negative", ha="left", va="center", fontsize=PT_SMALL, color=MUTE, zorder=6)
    ax.plot([crossing, crossing + 14], [p.loc[crossing, "ci95_high"] + 0.3, 2.3], color=MUTE, lw=LW_HAIR, zorder=2)
    neg600 = int(p.loc[600, "negative_seeds"])
    ax.text(720, 8.0, f"{neg600}/10 < 0 at 600", ha="right", va="center", fontsize=PT_SMALL, color=MUTE, zorder=6)
    return dict(marks=marks, crossing=crossing, lowest=lowest, negative_seeds_600=neg600)


def loss_gap(ax, g, valid, xlim):
    p = gap_panel(ax, g, "test_cross_entropy", ylabel="Exact − shared cross-entropy", ylim=(-0.1, 0.018),
                  yticks=[-0.1, -0.05, 0.0], yticklabels=["−.1", "−.05", "0"], xlim=xlim)
    marks = {}
    for epoch in (180, 315, 600):
        r = p.loc[epoch]
        ax.plot([epoch], [r["mean"]], marker="o", ms=MARKER_MS, mfc="white", mec=COLORS["bp"], mew=LW_ERR, ls="none", zorder=5)
        marks[epoch] = plain(r["mean"], 3)
    ax.text(180, p.loc[180:240, "ci95_low"].min() - 0.005, marks[180], ha="center", va="top",
            fontsize=PT_ANNOT, color=INK, zorder=6)
    ax.text(610, p.loc[600, "mean"], marks[600], ha="left", va="center", fontsize=PT_ANNOT, color=INK, zorder=6)
    ax.text(340, -0.0675, marks[315], ha="left", va="center", fontsize=PT_ANNOT, color=INK, zorder=6)
    ax.plot([318, 333], [p.loc[315, "ci95_low"] - 0.0015, -0.0665], color=MUTE, lw=LW_HAIR, zorder=2)
    ax.text(205, 0.0065, "below zero: exact lower", ha="left", va="bottom", fontsize=PT_SMALL, color=MUTE, zorder=6)
    assert valid["exact_lower_mean_cross_entropy_all180_to600"]
    return marks


# ── I: validation loss ────────────────────────────────────────────────────
def validation_loss(ax, s, stop_pts, xlim):
    ends = {}
    for key in F_ORDER:
        arm, depth = key
        spec = COND[key]
        color = COLORS[spec["color"]]
        p = s[s.arm.eq(arm) & s.depth.eq(depth) & s.metric.eq("best_validation_loss")].sort_values("epoch")
        lw = LW_REF if spec["color"] == "ink" else LW_DATA
        ax.plot(p.epoch, p["mean"], color=color, ls=spec["ls"], lw=lw, zorder=3)
        ends[key] = float(p[p.epoch.eq(600)].iloc[0]["mean"])
    ax.plot(stop_pts.epochs_run, stop_pts.best_validation_loss, ls="none", marker="o", ms=SEED_MS, mfc="white",
            mec=COLORS["point_mlp"], mew=LW_EDGE, zorder=4)
    ax.axvline(180, color=MUTE, ls=":", lw=LW_REF, zorder=2)
    clusters = {"BP": [ends[k] for k in F_ORDER if COND[k]["recipe"] == "BP" and k[1] == 3],
                "LocalCA": [ends[k] for k in F_ORDER if COND[k]["recipe"] == "LocalCA"]}
    for name, vals in clusters.items():
        lo, hi = min(vals), max(vals)
        ax.plot([606, 606], [lo, hi], color=MUTE, lw=LW_HAIR, solid_capstyle="butt", zorder=2)
        ax.text(611, (lo + hi) / 2.0, name, ha="left", va="center", fontsize=PT_SMALL, color=INK, zorder=6)
    ax.text(611, ends[("exact_autograd_bp_recipe", 1)], "D1", ha="left", va="center", fontsize=PT_SMALL,
            color=label_color(COLORS["point_mlp"]), zorder=6)
    ax.text(205, ends[("exact_autograd_bp_recipe", 1)] - 0.045, "open: D1 stop epochs", ha="left", va="top",
            fontsize=PT_SMALL, color=MUTE, zorder=6)
    ax.set(xlim=xlim, ylim=(0.0, 0.72), xticks=[0, 180, 400, 600], yticks=[0, 0.2, 0.4, 0.6], xlabel="Epoch",
           ylabel="Best validation loss")
    style_panel(ax, grid="y")
    return ends


CAPTION = r"""\textbf{Physical depth, sensor alignment and training budget.}
\textbf{A}, Nested gain task: eight excitatory slots (blue) carry the class signal $m$ times one global, two coarse and four fine gains (mute brackets); one inhibitory sensor (red) per tier reports its gain $h$ with alignment $\alpha$, so $x_E = m\,h_{\rm f}h_{\rm c}h_{\rm g}$.
\textbf{B}, The same eight compartments (rings) as one fan (D1 $[8]$, sensors of all tiers) or three serial stages (D3 $[2,1,2]$, one tier per stage; emphasised card): red, inhibitory sensors; blue, excitatory contacts on the distal compartments; output $y$; somatic error $\delta_0$; contacts and parameters matched. Masks describe the three-tier cohort of \textbf{D}--\textbf{I}.
\textbf{C}, Four-tier task, ten seeds (10400--10409), 180 epochs: test accuracy versus physical depth D1--D4 for aligned exact BP (black dashed reference), exact-path LocalCA (dark red), shared-soma LocalCA (amber), grouped point (gray), raw additive (blue dotted) and reversed placement (gray dotted; Source Data regime \texttt{rewired\_tree}); tag, paired D4 minus D3 exact-BP contrast.
\textbf{D}, Three-tier task, ten seeds (10200--10209), 180 epochs: exact-BP accuracy versus $\alpha$ for D1--D3 (light to dark); tags, paired D3 minus D1 gains (pp). The $\alpha=1$ cohort is extended in \textbf{E}--\textbf{I}.
\textbf{E}, Validation-selected accuracy against epoch; the open circle marks epoch 312, where exact-path LocalCA first falls below shared-soma LocalCA.
\textbf{F}, Selected states at both budgets, grouped by recipe (small dots, seeds; open dots, broadcast autograd; lines join one condition).
\textbf{G}, Paired exact-path minus shared-soma accuracy gap (pp), marked at epochs 180, 486 and 600.
\textbf{H}, The same pairs' cross-entropy gap, negative throughout, marked at 180, 315 and 600.
\textbf{I}, Best validation loss for the six conditions (open markers, the eight D1 stop epochs; brackets, recipe); D3 losses were still falling at 600, convergence is not established.
Circles are means of ten paired seeds; error bars and shading (\textbf{E}, \textbf{G}, \textbf{H}) are pointwise 95\% whole-seed bootstrap intervals; dotted verticals mark the 180-epoch budget; dashed lines mark chance (50\%) or zero. Both recipes use Adam at their original rates."""


def main():
    h4, contrast = load_h4()
    dose, gains = load_dose()
    b = load_budget()
    s, g, valid = b["s"], b["g"], b["valid"]
    DEST.mkdir(exist_ok=True)

    canvas = NativeCanvas(492 / 72, 3, row_weights=[117, 108, 108], hgutter_pt=40, vgutter_pt=48,
                          margins=Margins(left=46, right=16, top=25, bottom=38))
    a = canvas.panel("A", 0, 0, 3, schematic=True, lock=False, title="Nested gain task")
    bx = canvas.panel("B", 0, 3, 4, schematic=True, lock=False, title="D1 [8] versus D3 [2,1,2]")
    c = canvas.panel("C", 0, 7, 5, title="Depth pays only if aligned")
    d = canvas.panel("D", 1, 0, 4, title="Gain needs alignment")
    e = canvas.panel("E", 1, 4, 4, title="Depth gain lasts to 600")
    f_ = canvas.panel("F", 1, 8, 4, title="Budget reorders the rules")
    gx = canvas.panel("G", 2, 0, 4, title="Accuracy ranking flips")
    hx = canvas.panel("H", 2, 4, 4, title="Exact credit: lower loss")
    ix = canvas.panel("I", 2, 8, 4, title="No plateau by epoch 600")
    xlim_row2 = (0, 800)
    a_info = nested_gain_task(a)
    b_info = stage_pair(bx)
    c_tag = depth_ladder(c, h4, contrast)
    d_tags = alignment_dose(d, dose, gains)
    e_cross = trajectories(e, s, b["crossing"])
    f_rows, n_stop = budget_states(f_, s, b["end"], b["stops"])
    g_info = accuracy_gap(gx, g, valid, xlim_row2)
    h_marks = loss_gap(hx, g, valid, xlim_row2)
    i_ends = validation_loss(ix, s, b["stop_pts"], xlim_row2)
    # titles must stay inside their own slot (centred titles of neighbours meet in the gutter)
    canvas.fig.canvas.draw()
    renderer = canvas.fig.canvas.get_renderer()
    for rec in canvas._records:
        ax = canvas.axes[rec["name"]]
        w = ax.title.get_window_extent(renderer).width / canvas.fig.dpi * 72.0
        slot_w = canvas.slot_pt(rec["row"], rec["col"], rec["colspan"])[2]
        assert w <= slot_w + 20.0, (rec["name"], w, slot_w)
    for name in "CDEFGHI":            # one reserve for every locked panel: equal widths, letters clear
        canvas.declare_reserve(name, left=6.0, right=2.0)
    canvas.lock_reserves()
    canvas.fig.canvas.draw()
    overflow = []                     # every in-panel artist stays inside its own axes box
    for name in "CDEFGHI":
        ax = canvas.axes[name]
        box = ax.get_window_extent(renderer)
        arts = list(ax.texts) + ([ax.get_legend()] if ax.get_legend() else [])
        for art in arts:
            bb = art.get_window_extent(renderer)
            if bb.x1 > box.x1 + 1.0 or bb.x0 < box.x0 - 1.0:
                overflow.append((name, getattr(art, "get_text", lambda: "legend")(),
                                 round((bb.x0 - box.x0) * 72 / canvas.fig.dpi, 1), round((bb.x1 - box.x1) * 72 / canvas.fig.dpi, 1)))
    assert not overflow, overflow
    findings = list(canvas.align_letters())
    if findings:                      # name the axes a flagged letter actually touches
        boxes = {rec["name"]: canvas.axes[rec["name"]].get_tightbbox(renderer) for rec in canvas._records}
        for item in canvas._letters:
            lb = item["art"].get_window_extent(renderer)
            hits = [n for n, ob in boxes.items() if canvas.axes[n] is not item["ax"] and ob is not None
                    and min(lb.x1, ob.x1) - max(lb.x0, ob.x0) > 1.0 and min(lb.y1, ob.y1) - max(lb.y0, ob.y0) > 1.0]
            if hits:
                print(f"  letter {item['letter']} touches {hits}: letter x {lb.x0 * 72 / canvas.fig.dpi:.1f}-{lb.x1 * 72 / canvas.fig.dpi:.1f}, "
                      + ", ".join(f"{n} x {boxes[n].x0 * 72 / canvas.fig.dpi:.1f}-{boxes[n].x1 * 72 / canvas.fig.dpi:.1f} y {boxes[n].y0 * 72 / canvas.fig.dpi:.1f}-{boxes[n].y1 * 72 / canvas.fig.dpi:.1f}" for n in hits)
                      + f" letter y {lb.y0 * 72 / canvas.fig.dpi:.1f}-{lb.y1 * 72 / canvas.fig.dpi:.1f}")
    path = DEST / "physical_depth_followup.pdf"
    problems = findings + list(canvas.save(path, name="physical_depth_followup", dpi=200))
    plt.close(canvas.fig)

    # -- render-time source table (every plotted value, one row each) ------
    table = []
    table.extend(dict(panel="C", source="source_data/physical_depth_h4_factorial/condition_summary.csv", **r) for r in h4.to_dict("records"))
    table.append(dict(panel="C", source="source_data/physical_depth_h4_factorial/paired_contrasts.csv", **contrast))
    table.extend(dict(panel="D", source="source_data/physical_alignment_dose/condition_summary.csv", **r) for r in dose.to_dict("records"))
    table.extend(dict(panel="D", source="source_data/physical_alignment_dose/paired_contrasts.csv", **r) for r in gains.to_dict("records"))
    for key in E_ORDER:
        p = s[s.arm.eq(key[0]) & s.depth.eq(key[1]) & s.metric.eq("test_accuracy")]
        table.extend(dict(panel="E", source="source_data/physical_depth_followup/condition_trajectory_summary.csv", **r) for r in p.to_dict("records"))
    table.extend(dict(panel="F", source="source_data/physical_depth_budget/canonical/extension_endpoints.csv", **r) for r in f_rows.to_dict("records"))
    for metric, panel in (("test_accuracy", "G"), ("test_cross_entropy", "H")):
        table.extend(dict(panel=panel, source="source_data/physical_depth_followup/paired_trajectory_summary.csv", **r)
                     for r in g[g.metric.eq(metric)].to_dict("records"))
    for key in F_ORDER:
        p = s[s.arm.eq(key[0]) & s.depth.eq(key[1]) & s.metric.eq("best_validation_loss")]
        table.extend(dict(panel="I", source="source_data/physical_depth_followup/condition_trajectory_summary.csv", **r) for r in p.to_dict("records"))
    table.extend(dict(panel="I", source="source_data/physical_depth_followup/stopping_by_seed.csv", **r) for r in b["stop_pts"].to_dict("records"))
    pd.DataFrame(table).to_csv(DEST / "figure_source.csv", index=False)

    panelmap = {
        "A": "Schematic (task_card): eight excitatory input slots under three bracket tiers (global x1, coarse x2, fine x4), one inhibitory sensor contact per tier reporting h, alignment alpha badge, x_E = m h_f h_c h_g. No measured data.",
        "B": "Schematic: D1 [8] one fan versus D3 [2,1,2] three serial stages (hero card), inhibitory sensors on every incoming segment tagged all tiers (D1) and global/coarse/fine per stage (D3), excitatory contacts on the distal compartments, output y, somatic error delta_0; contacts and parameters matched. Masks describe the three-tier cohort of D-I.",
        "C": "Four-tier factorial (seeds 10400-10409, 180 epochs): 180-epoch test accuracy versus physical depth D1-D4 for aligned exact BP (reference), exact-path LocalCA, shared-soma LocalCA, grouped point, raw additive and reversed placement (Source Data regime rewired_tree); means with 95% bootstrap intervals; printed D4 - D3 exact-BP contrast.",
        "D": "Alignment dose (three-tier task, seeds 10200-10209, 180 epochs): exact-BP test accuracy versus alpha for D1/D2/D3 with 95% intervals; printed paired D3 - D1 gains (pp) and positive-pair counts from paired_contrasts.csv.",
        "E": "Validation-selected test accuracy trajectories to 600 epochs: D1 exact BP, D3 exact BP, D3 exact-path LocalCA, D3 shared-soma LocalCA; 12% bands are pointwise 95% whole-seed intervals; the exact-path/shared-soma crossing epoch is marked.",
        "F": "Validation-selected test accuracy at the 180 and 600 budgets for the six conditions grouped by recipe: seed dots (open = broadcast autograd), white-faced means with 95% intervals, connectors per condition; D1 early-stop count.",
        "G": "Paired D3 exact-path minus shared-soma test-accuracy gap (pp) with pointwise 95% intervals; marked means at 180, 486 and 600; first negative epoch 312; negative-seed count at 600.",
        "H": "Paired exact-path minus shared-soma test cross-entropy gap with pointwise 95% intervals; marked means at 180, 315 and 600; negative throughout.",
        "I": "Best validation loss trajectories for the six conditions (stopped D1 runs carried forward; open markers at the eight D1 stop epochs); recipe clusters bracketed at 600.",
    }
    inputs = [Path(__file__), J / "scripts/figure_canvas.py", J / "scripts/journal_style.py", J / "scripts/native_schematics.py",
              J / "scripts/credit_tree_schematics.py",
              OUT / "condition_trajectory_summary.csv", OUT / "paired_trajectory_summary.csv", OUT / "analysis_validation.json",
              OUT / "stopping_by_seed.csv", OUT / "validation_selected_seed_trajectories.csv",
              BUDGET / "extension_protocol.json", BUDGET / "extension_endpoints.csv", BUDGET / "extension_summary.csv",
              BUDGET / "extension_loss_contrasts.csv", BUDGET / "extension_paired_seed_contrasts.csv",
              H4 / "condition_summary.csv", H4 / "paired_contrasts.csv", H4 / "seed_outcomes.csv", H4 / "audit.json",
              DOSE / "condition_summary.csv", DOSE / "paired_contrasts.csv", DOSE / "combined_seed_outcomes.csv", DOSE / "audit.json"]
    printed = dict(C=c_tag, D=d_tags, E=dict(crossing_epoch=b["crossing"], exact_path_accuracy_at_crossing=e_cross),
                   F=dict(d1_stopped_before600=n_stop), G=g_info, H=h_marks, I={f"{k[0]}_D{k[1]}": v for k, v in i_ends.items()})
    analyze.write(DEST / "figure_provenance.json", {
        "figure_sha256": analyze.sha(path), "builder_sha256": analyze.sha(__file__),
        "sources_sha256": {str(p.relative_to(J)): analyze.sha(p) for p in inputs},
        "panel_map": panelmap, "printed_numbers": printed, "schematics": {"A": a_info, "B": b_info},
        "layout": {"canvas_pt": [518.4, 492.0], "rows_pt": [117, 108, 108], "panels": "A3 B4 C5 | D4 E4 F4 | G4 H4 I4",
                   "layout_findings": [str(p) for p in problems]},
        "all_original_outcomes_preserved": True, "new_training_runs": 0,
        "scope": "Figure overhaul 2026-09-08 (DESIGN_SPEC section 6). Nine panels: task and stage schematics (A, B), the "
                 "four-tier depth factorial at 180 epochs (C), the three-tier alignment dose (D) and the 600-epoch budget "
                 "follow-up (E-I). No new training; every printed number is replayed from frozen Source Data and re-checked "
                 "against the independent summary tables. CIs are pointwise descriptive whole-seed bootstrap intervals, not "
                 "simultaneous bands. Original main6 PDF retained in the parent directory; the 180-epoch plot remains S31.",
    })
    (OUT / "FIGURE_CAPTION.md").write_text(CAPTION + "\n")
    print(path)
    for problem in problems:
        print("  layout:", problem)
    return problems


if __name__ == "__main__":
    main()

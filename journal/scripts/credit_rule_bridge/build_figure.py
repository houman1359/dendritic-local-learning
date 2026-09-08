#!/usr/bin/env python3
"""Main Fig. 4: prospective targets and credit geometry (credit calibration bridge).

Grid (DESIGN_SPEC §4): ``NativeCanvas`` 490 pt, rows 118 / 116 / 116 pt,
hgutter 37, vgutter 40, margins 58 / 15 / 24 / 36.  Row 0 is the schematic
row -- A (5 modules, two targets on one shared tree) and B (7, three credit
rules over the six sites); row 1 holds the learning curves C, D, E (4 / 4 / 4,
one log-NMSE range, y ticks on C only); row 2 holds the endpoint strips F
(4, same log-NMSE range, column-locked with C), the deficit forest G (4) and
the energy-capture panel H (4).

Waiver (spec D3): F, G and H share neither axis (F repeats row 1's y range,
G is a forest on NMSE differences, H is a capture fraction); the spec records
this 4/4/4 row as an explicit exception ("waiver D3 recorded (F/G/H)").
Panel I of the spec (effective-rank trajectory) is demoted to the Supplement;
its numbers are derived here and written to figure_source.csv for the caption.

Every schematic element is drawn with the shared glyph library
(``native_schematics.Frame``: task_card, balanced_tree, soma, error_in,
badge, subscript, disc, arrow).  Private helpers (errata #7) cover what the
library does not draw at this size: the spec's D5 broadcast bus sourced at
the soma with drops into the six junctions (``Frame.credit_delivery(mode=
'scalar')`` puts its source dot at the bus end and targets terminals), the
same bus with weight-encoding drop dots for the fixed profile, the operator
badges on the junction rings (``balanced_tree(badges=...)`` drops them when
sibling junctions are closer than 9.5 pt, which they always are in a 77-pt
card), and token-subscript text chains.  Row 2 declares a 10 pt top reserve
so its letters clear row 1's x labels (row-separation floor 8.5 pt), as the
Fig. 1 builder does.

This builder performs no fitting: every printed number is replayed from the
frozen Source Data named in the provenance record and re-derived as an
assertion.  The only render-time file it writes is
``figures/initial_profile_source.csv`` (the mean calibrated six-site
profile drawn in B), registered in ``figure_provenance.json``.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import run  # noqa: E402  (frozen bootstrap, sha, write, OUT, JOURNAL)
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR, LW_REF,  # noqa: E402
                           MARKER_MS, PT_ANNOT, PT_LABEL, PT_LEGEND, PT_SMALL,
                           PT_TICK, SEED_ALPHA, SEED_MS, Margins, NativeCanvas,
                           style_panel)
from journal_style import label_color  # noqa: E402
from credit_tree_schematics import AMBER_TEXT, GHOST, mix  # noqa: E402
from native_schematics import BADGE_STYLE, Frame, _text_w_pt, reference_line  # noqa: E402

INK, MUTE = COLORS["ink"], COLORS["mute"]
DASH = (0, (2.2, 1.8))
RULE_COLORS = {"exact": COLORS["bp"], "unit_broadcast": COLORS["scalar"],
               "calibrated_broadcast": COLORS["per_soma"]}
RULE_NAMES = {"exact": "Exact path", "unit_broadcast": "Unit broadcast",
              "calibrated_broadcast": "Initial profile"}
TASK_NAMES = {"matching": "Pairwise", "quartet": "Quartic", "nested": "Nested"}
TASK_MARKS = {"matching": ("o", INK, INK), "quartet": ("s", INK, "white"),
              "nested": ("^", MUTE, MUTE)}          # §0.3: sole ink-marker use
YLIM_NMSE = (0.016, 1.65)
Y_TICKS = ([0.02, 0.1, 1.0], ["0.02", "0.1", "1"])
STEPS = 1024
COMMON_RATE = 0.003
# model node -> library junction (children 10=[8,9], 13=[11,12], 14=[10,13];
# the six nonsomatic sites are path columns 0..5 = nodes 8..13)
SITE_JUNCTIONS = ["JLL", "JLR", "JL", "JRL", "JRR", "JR"]
SOURCE = run.OUT / "summaries"
CAPTURE = run.JOURNAL / "source_data/credit_resolution_bridge/capture"
DEST = run.OUT / "figures"
MAIN_PDF = run.JOURNAL / "figures/main/figure_04.pdf"


def signed(value, decimals=3):
    return f"{value:+.{decimals}f}".replace("-", "−")


# ── loaders (frozen Source Data) ──────────────────────────────────────────
def load_curves():
    curves = pd.read_csv(SOURCE / "all_curves.csv")
    curves = curves[(curves.model == "algebraic") & (curves.optimizer == "adam")]
    assert curves.seed.nunique() == 20
    return curves


def load_floors():
    floors = pd.read_csv(SOURCE / "task_variance_and_noise_floor.csv")
    out = {r.task: float(r.expected_label_noise_nmse) for r in floors[floors.model == "algebraic"].itertuples()}
    assert out == {"matching": 0.0225, "quartet": 0.045, "nested": 0.0225}
    return out


def load_contrasts():
    contrasts = pd.read_csv(SOURCE / "paired_contrasts.csv")
    seeds = pd.read_csv(SOURCE / "paired_seed_contrasts.csv")
    keep = lambda d: d[(d.model == "algebraic") & (d.optimizer == "adam")]
    return keep(contrasts), keep(seeds)


def load_selection():
    sel = json.loads((run.OUT / "selection_freeze.json").read_text())
    assert sel["common_rates"]["algebraic"]["adam"] == COMMON_RATE
    return sel


def load_capture():
    """Per-seed cumulative energy fractions of the exact-trained path field."""
    ev = pd.read_csv(CAPTURE / "matched_bridge_eigenvalues.csv")
    ev = ev[(ev.cohort == "matched_fresh_algebraic") & (ev.optimizer == "adam") & (ev.rule == "exact")
            & ev.selected_rate & (ev.field == "path_q") & ev.step.isin([0, STEPS])].copy()
    ev = ev.sort_values(["family", "step", "seed", "index"])
    ev["cumulative"] = ev.groupby(["family", "step", "seed"]).fraction.cumsum()
    assert ev.groupby(["family", "step"]).seed.nunique().eq(20).all()
    assert np.allclose(ev[ev["index"] == 6].cumulative, 1.0, atol=1e-9)
    cap = pd.read_csv(CAPTURE / "matched_bridge_capture.csv")
    cap = cap[(cap.cohort == "matched_fresh_algebraic") & (cap.optimizer == "adam") & (cap.rule == "exact")
              & cap.selected_rate & (cap.step == STEPS) & (cap.dictionary == "uniform_projection")]
    assert cap.groupby("family").seed.nunique().eq(20).all()
    spectra = pd.read_csv(CAPTURE / "matched_bridge_spectra.csv")
    spectra = spectra[(spectra.cohort == "matched_fresh_algebraic") & (spectra.optimizer == "adam")
                      & (spectra.rule == "exact") & spectra.selected_rate & (spectra.field == "path_q")]
    gauge = pd.read_csv(CAPTURE / "matched_bridge_gauge_spectra.csv")
    gauge = gauge[(gauge.cohort == "matched_fresh_algebraic") & (gauge.optimizer == "adam")
                  & (gauge.rule == "exact") & gauge.selected_rate]
    return ev, cap, spectra, gauge


def load_profile():
    """Mean calibrated six-site profile over the 20 fresh seeds (drawn in B)."""
    rows, children = [], set()
    for task in TASK_NAMES:
        for path in sorted(glob.glob(str(run.OUT / f"runs/fresh/algebraic/seed_*_task_{task}_states.npz"))):
            with np.load(path) as d:
                profiles = d["initial_profiles"]
            assert profiles.shape[1] == 6 and np.allclose(profiles, profiles[0])
            meta = json.loads(Path(path.replace("_states.npz", "_metadata.json")).read_text())
            if task != "nested":          # the shared balanced tree: leaves permute, ancestry does not
                children.add(json.dumps({k: v for k, v in meta["children"].items() if k in ("10", "13", "14")}, sort_keys=True))
            rows.append(dict(task=task, seed=int(Path(path).name.split("_")[1]), **{f"site_{k}": float(v) for k, v in enumerate(profiles[0])}))
    assert len(children) == 1 and json.loads(next(iter(children))) == {"10": [8, 9], "13": [11, 12], "14": [10, 13]}
    frame = pd.DataFrame(rows)
    assert len(frame) == 60
    wide = frame[frame.task == "matching"].sort_values("seed")
    assert np.allclose(wide[[f"site_{k}" for k in range(6)]].to_numpy(),
                       frame[frame.task == "quartet"].sort_values("seed")[[f"site_{k}" for k in range(6)]].to_numpy())
    cols = [f"site_{k}" for k in range(6)]
    summary = pd.DataFrame(dict(site=range(6), node=range(8, 14), junction=SITE_JUNCTIONS,
                                mean_profile=wide[cols].mean().to_numpy(),
                                mean_abs_profile=wide[cols].abs().mean().to_numpy(), n_seeds=len(wide)))
    return summary


# ── private glyph helpers (errata #7; same geometry as the library) ──────
def chain(f, xy, parts, *, size=PT_ANNOT, color=None, ha="left", va="center", zorder=6):
    """Plain strings and (base, sub[, tail]) tuples laid out left-to-right."""
    color = INK if color is None else color
    base_size = PT_ANNOT if size <= PT_SMALL else size
    widths = []
    for p in parts:
        if isinstance(p, tuple):
            w = _text_w_pt(f.ax, p[0], base_size) + 0.4 + _text_w_pt(f.ax, p[1], PT_SMALL)
            if len(p) > 2 and p[2]:
                w += 0.6 + _text_w_pt(f.ax, p[2], base_size)
            widths.append(w)
        else:
            widths.append(_text_w_pt(f.ax, p, size))
    x, y = xy
    total = sum(widths)
    if ha == "center":
        x -= f.fx(total / 2.0)
    elif ha == "right":
        x -= f.fx(total)
    for p, w in zip(parts, widths):
        if isinstance(p, tuple):
            f.subscript((x, y), p[0], p[1], p[2] if len(p) > 2 else "", size=size, color=color, ha="left", va=va, zorder=zorder)
        else:
            f.text((x, y), p, size=size, color=color, ha="left", va=va, zorder=zorder)
        x += f.fx(w)
    return total


def tree_scale_pt(f, nodes):
    """Points per library tree unit (T1..T8 span 4.54 units)."""
    return (nodes["T8"][0] - nodes["T1"][0]) * f.w_pt / 4.54


# bus attachment offsets (tree units) so the drops fall through canopy gaps
BUS_OFFSETS = {"JLL": 0.0, "JLR": 0.0, "JL": -0.23, "JRL": 0.0, "JRR": 0.23, "JR": 0.32}


def junction_bus(f, nodes, *, color, radii=None, signs=None):
    """Spec D5 broadcast glyph for the six junction sites.

    Source dot (r 1.6) at the soma, hairline riser up the left of the tree,
    LW_HAIR bus above the canopy, one hairline drop into every junction (head
    2.6).  ``radii`` (pt) replaces the arrowheads by drop dots whose radius
    encodes a fixed weight; ``signs`` < 0 draws that dot open (white face,
    rule-colour rim).
    """
    s = tree_scale_pt(f, nodes)
    top = max(nodes[t][1] for t in nodes.terminals) + f.fy(6.0)
    xs = {j: nodes[j][0] + f.fx(BUS_OFFSETS[j] * s) for j in SITE_JUNCTIONS}
    x_feed = min(nodes[t][0] for t in nodes.terminals) - f.fx(3.5)
    x_hi = max(xs.values()) + f.fx(2.0)
    sx, sy = nodes.soma
    src = (sx - f.fx(nodes.soma_r_pt + 0.4), sy)
    f.disc(src, 1.6, fill=color, zorder=6)
    f.ax.plot([src[0], x_feed, x_feed, x_hi], [sy, sy, top, top], color=color, lw=f.lw(LW_HAIR),
              solid_capstyle="round", solid_joinstyle="round", zorder=5)
    for k, j in enumerate(SITE_JUNCTIONS):
        p0 = (xs[j], top)
        if radii is None:
            f.arrow(p0, (nodes[j][0], nodes[j][1] + f.fy(2.8)), color=color, lw=LW_HAIR, head=2.6, zorder=5)
        else:
            end = (nodes[j][0], nodes[j][1] + f.fy(3.4))
            f.leader(p0, end, color=color, lw=LW_HAIR)
            open_dot = signs is not None and signs[k] < 0
            f.disc(end, radii[k], fill="white" if open_dot else color, edge=color if open_dot else "none",
                   lw=LW_EDGE, zorder=6)


def operator_badges(f, nodes, badges, *, r_pt=3.8):
    """+ / × badges on the junction rings and the root's + on the soma."""
    for name, text in badges.items():
        if name == "S":
            f.text(nodes.soma, text, size=PT_ANNOT, color=INK, zorder=6, force=True)
            continue
        f.disc(nodes[name], r_pt, fill="white", edge=COLORS["dend"], lw=LW_EDGE, zorder=3.6)
        f.text(nodes[name], text, size=PT_ANNOT, color=INK, zorder=6, force=True)


def badge_on_axes(ax, xy, kind, *, ha="left", va="center"):
    """``Frame.badge`` geometry (BADGE_STYLE, PT_SMALL, LW_HAIR edge) on a data axes."""
    key, face, edge = BADGE_STYLE[kind]
    try:
        colour = label_color(COLORS[key], background=face)
    except ValueError:
        colour = COLORS[key]
    return ax.text(xy[0], xy[1], kind, fontsize=PT_SMALL, color=colour, ha=ha, va=va, zorder=7,
                   bbox=dict(boxstyle=f"round,pad=0.28,rounding_size={2.0 / PT_SMALL:.3f}", facecolor=face,
                             edgecolor=edge, linewidth=LW_HAIR))


def fit_first(f, options, size, width_pt):
    for s in options:
        if _text_w_pt(f.ax, s, size) <= width_pt:
            return s
    return options[-1]


# ── A: two targets on one shared tree ─────────────────────────────────────
OPERATORS = {
    "matching": {"JLL": "×", "JLR": "×", "JRL": "×", "JRR": "×", "JL": "+", "JR": "+", "S": "+"},
    "quartet": {"JLL": "×", "JLR": "×", "JRL": "×", "JRR": "×", "JL": "×", "JR": "×", "S": "+"},
}
SUB_PT, LIFT_PT = 13.0, 12.0


def panel_targets(ax):
    f = Frame(ax)
    chain(f, (0.5, 1.0 - f.fy(SUB_PT * 0.5)), ["same tree, weights, examples; ", ("I", "8", "/4")],
          size=PT_ANNOT, color=MUTE, ha="center")
    cells = f.split(2, axis="x", gap_pt=10.0, pad_pt=(0, 0, SUB_PT, 0))
    footers = {"matching": ["four pair products, ±0.5", "four pair products"],
               "quartet": ["two complementary quartets", "two quartets"]}
    for cell, task in zip(cells, ("matching", "quartet")):
        w_pt = cell[2] * f.w_pt
        core = f.task_card(cell, title=f"{TASK_NAMES[task]} target",
                           footer=fit_first(f, footers[task], PT_SMALL, w_pt - 8.0))
        body = (core[0] + f.fx(2.0), core[1] + f.fy(LIFT_PT), core[2] - f.fx(4.0), core[3] - f.fy(LIFT_PT))
        nodes = f.balanced_tree(body, trunk=False, mode="forward", output="f",
                                input_labels=[("x", str(i)) for i in range(1, 9)])
        operator_badges(f, nodes, OPERATORS[task])
        f.error_in(nodes.soma, side="right")
    return ax


# ── B: three credit rules over the six sites ─────────────────────────────
def panel_rules(ax, profile):
    f = Frame(ax)
    f.text((0.5, 1.0 - f.fy(SUB_PT * 0.5)), "delivery over the six nonsomatic sites", size=PT_ANNOT, color=MUTE)
    cells = f.split(3, axis="x", gap_pt=10.0, pad_pt=(0, 0, SUB_PT, 0))
    cards = [("exact", "q(x) per site", "exact"),
             ("unit_broadcast", "equal at all sites", "control"),
             ("calibrated_broadcast", "fixed at step 0, 256 examples", "control")]
    cores = [f.task_card(cell, title=RULE_NAMES[rule], footer=foot) for cell, (rule, foot, _) in zip(cells, cards)]
    body_h = min(c[3] for c in cores) - f.fy(LIFT_PT)      # identical tree geometry
    weights = profile.sort_values("site").mean_profile.to_numpy()
    radii = 1.2 + 1.2 * np.abs(weights) / np.abs(weights).max()
    for cell, core, (rule, _, badge) in zip(cells, cores, cards):
        body = (core[0] + f.fx(2.0), core[1] + core[3] - body_h, core[2] - f.fx(4.0), body_h)
        nodes = f.balanced_tree(body, trunk=False, mode="forward", output=None)
        color = RULE_COLORS[rule]
        if rule == "exact":
            f.credit_delivery(nodes, mode="exact", targets=["JLL", "JLR", "JRL", "JRR"])
        elif rule == "unit_broadcast":
            junction_bus(f, nodes, color=color)
        else:
            junction_bus(f, nodes, color=color, radii=radii, signs=np.sign(weights))
        f.error_in(nodes.soma, side="right")
        f.badge((core[0] + f.fx(4.0), body[1] - f.fy(LIFT_PT - 1.5)), badge, ha="left", va="bottom")
    return ax


# ── C / D / E: learning curves ────────────────────────────────────────────
def log_axis(ax, *, labels=True):
    ax.set_yscale("log")
    ax.set_ylim(*YLIM_NMSE)
    ax.yaxis.set_major_locator(FixedLocator(Y_TICKS[0]))
    ax.yaxis.set_major_formatter(FixedFormatter(Y_TICKS[1] if labels else [""] * 3))
    ax.minorticks_off()


def curves(ax, part, floor, *, legend=False, ylabel=False, table, panel, note=None):
    for rule in RULE_NAMES:
        sub = part[part.rule == rule]
        assert sub.seed.nunique() == 20 and sub.rate.nunique() == 1
        records = []
        for step, group in sub.groupby("step"):
            res = run.bootstrap(group.sort_values("seed").test_nmse)
            records.append(dict(step=step, **res))
            table.append(dict(panel=panel, task=str(part.task.iloc[0]), rule=rule, rate=float(sub.rate.iloc[0]), **records[-1]))
        frame = pd.DataFrame(records)
        ax.fill_between(frame.step, frame.ci95_low, frame.ci95_high, color=RULE_COLORS[rule], alpha=.13, lw=0)
        ax.plot(frame.step, frame["mean"], color=RULE_COLORS[rule], lw=LW_DATA, label=RULE_NAMES[rule],
                zorder=4 if rule == "exact" else 3)
    ax.set_xlim(0, 1130)
    log_axis(ax, labels=ylabel)
    reference_line(ax, floor, axis="y", label=None, span=(0, 1130))
    ax.text(30, floor * 1.18, "noise", ha="left", va="bottom", fontsize=PT_SMALL, color=MUTE)
    ax.set_xticks([0, 512, 1024], ["0", "512", "1,024"])
    ax.set_xlabel("Training step")
    if ylabel:
        ax.set_ylabel("Held-out NMSE")
    if legend:
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.88), frameon=False, fontsize=PT_LEGEND, handlelength=1.5,
                  borderaxespad=.2, labelspacing=.35)
    if note:
        for k, (line, size, color) in enumerate(note):
            ax.text(0.97, 0.95 - 0.105 * k, line, transform=ax.transAxes, ha="right", va="top", fontsize=size, color=color)
    style_panel(ax, grid="y")


# ── F: per-seed endpoints ─────────────────────────────────────────────────
STRIPS = [("matching", "exact", 0.0), ("matching", "unit_broadcast", 1.0), ("matching", "calibrated_broadcast", 2.0),
          ("quartet", "exact", 3.4), ("quartet", "calibrated_broadcast", 4.4),
          ("nested", "exact", 5.8), ("nested", "calibrated_broadcast", 6.8)]


def endpoints(ax, curves_df, floors, *, table):
    end = curves_df[(curves_df.step == STEPS) & curves_df.selected_rate]
    stalled = None
    for task, rule, x in STRIPS:
        values = end[(end.task == task) & (end.rule == rule)].sort_values("seed").test_nmse.to_numpy()
        assert len(values) == 20
        color = RULE_COLORS[rule]
        ax.plot(x + np.linspace(-.28, .28, 20), values, ls="none", marker="o", ms=SEED_MS, mfc=color, mec="none",
                alpha=SEED_ALPHA, zorder=2)
        res = run.bootstrap(values)
        ax.errorbar(x, res["mean"], yerr=[[res["mean"] - res["ci95_low"]], [res["ci95_high"] - res["mean"]]], fmt="o",
                    color=color, ms=MARKER_MS, mfc="white", mew=LW_ERR, elinewidth=LW_ERR, capsize=2, zorder=4)
        table.append(dict(panel="F", task=task, rule=rule, step=STEPS, **res))
        if task == "quartet" and rule == "exact":
            near = int((values < 2.0 * floors[task]).sum())
            stalled = (near, len(values) - near, values[values >= 2.0 * floors[task]].min())
    assert stalled[:2] == (17, 3)
    # unit broadcast omitted on quartic / nested: its mean coincides with the profile
    wide = end.pivot_table(index=["task", "seed"], columns="rule", values="test_nmse")
    for task in ("quartet", "nested"):
        diff = (wide.loc[task, "unit_broadcast"] - wide.loc[task, "calibrated_broadcast"])
        table.append(dict(panel="F", task=task, rule="unit_broadcast minus calibrated_broadcast", step=STEPS, **run.bootstrap(diff)))
    for task, (lo, hi) in {"matching": (-.5, 2.5), "quartet": (2.9, 4.9), "nested": (5.3, 7.3)}.items():
        reference_line(ax, floors[task], axis="y", label=None, span=(lo, hi))
    ax.set_xlim(-.6, 7.4)
    log_axis(ax, labels=True)
    ax.set_xticks([1.0, 3.9, 6.3], ["Pairwise", "Quartic", "Nested\n(separate tree)"])
    ax.set_ylabel("NMSE at 1,024 steps")
    ax.text(-.45, 0.62, f"{stalled[0]} near floor", ha="left", va="center", fontsize=PT_SMALL, color=INK)
    ax.text(-.45, 0.40, f"{stalled[1]} stalled", ha="left", va="center", fontsize=PT_SMALL, color=INK)
    ax.plot([1.65, 3.12], [0.40, stalled[2] * 0.92], color=MUTE, lw=LW_HAIR, zorder=3)
    style_panel(ax, grid="y")
    ax.tick_params(axis="x", pad=2.2)


# ── G: profile deficit forest ─────────────────────────────────────────────
# y in points from the axes bottom (ylim = axes height), so every band is a
# fixed number of points: label lines never pile onto marker rows.
ROWS = [("Pairwise", "selected_rate", "matching", 85.5), ("Quartic", "selected_rate", "quartet", 68.5),
        ("Nested", "selected_rate", "nested", 51.5),
        ("Rule-specific rates", "selected_rate", "quartet_minus_matching", 22.5),
        ("Common rate 0.003", "common_rate", "quartet_minus_matching", 5.5)]
ROW2_RESERVE_PT = 10.0
FOREST_H = 116.0 - ROW2_RESERVE_PT      # axes height after the row-2 top reserve
LABEL_LIFT = 6.0


def forest(ax, contrasts, seeds, *, table):
    xlim = (-.2, 1.13)
    x_text = 0.03
    ax.plot([0, 0], [0.0, 97.0], color=MUTE, lw=LW_REF, ls=DASH, zorder=0)
    for label, sens, task, y in ROWS:
        contrast = "calibrated_broadcast minus exact" + (" interaction" if task == "quartet_minus_matching" else "")
        row = contrasts[(contrasts.sensitivity == sens) & (contrasts.task == task) & (contrasts.contrast == contrast)]
        assert len(row) == 1
        row = row.iloc[0]
        part = seeds[(seeds.sensitivity == sens) & (seeds.task == task) & (seeds.contrast == contrast)].sort_values("seed")
        assert len(part) == 20 and abs(part.difference.mean() - row["mean"]) < 1e-9
        assert part.difference.min() > xlim[0] and part.difference.max() < xlim[1]     # no seed hidden
        ax.plot(part.difference, y + np.linspace(-2.3, 2.3, 20), ls="none", marker="o", ms=SEED_MS, mfc=MUTE, mec="none",
                alpha=SEED_ALPHA, zorder=2)
        ax.errorbar(row["mean"], y, xerr=[[row["mean"] - row.ci95_low], [row.ci95_high - row["mean"]]], fmt="D", color=INK,
                    ms=MARKER_MS, mfc="white", mew=LW_ERR, elinewidth=LW_ERR, capsize=2, zorder=4)
        ax.text(x_text, y + LABEL_LIFT, label, ha="left", va="bottom", fontsize=PT_SMALL, color=INK)
        if task == "quartet_minus_matching":
            assert int(row.positive_seeds) == 20
            ax.text(xlim[1] - .01, y + LABEL_LIFT, f"{int(row.positive_seeds)}/{int(row.n_seeds)}", ha="right", va="bottom",
                    fontsize=PT_SMALL, color=MUTE)
        table.append(dict(panel="G", **row.to_dict()))
    ax.text(x_text, 102.8, "per task, descriptive", ha="left", va="center", fontsize=PT_SMALL, color=MUTE)
    ax.plot(xlim, [38.5, 38.5], color=MUTE, lw=LW_HAIR, zorder=1)
    ax.text(x_text, 42.5, "predefined quartic − pairwise", ha="left", va="center", fontsize=PT_SMALL, color=MUTE)
    ax.set(xlim=xlim, ylim=(0.0, FOREST_H), xticks=[0, .5, 1], yticks=[], xlabel="Profile − exact NMSE")
    ax.set_xticklabels(["0", "0.5", "1"])
    style_panel(ax, grid="none", spines=("bottom",))


# ── H: energy captured by the best k directions ───────────────────────────
KX = 0.35          # k = 1..6 sit at k + KX so the 'uniform' label clears '1'


def capture(ax, ev, cap, spectra, gauge, *, table):
    ks = np.arange(1, 7)
    init = ev[ev.step == 0]
    init_mean = init.groupby(["family", "index"]).cumulative.mean().unstack("family")
    assert np.allclose(init_mean["matching"], init_mean["quartet"], atol=1e-9)
    ghost, = ax.plot(ks + KX, init_mean["matching"].to_numpy(), color=GHOST, lw=LW_REF, zorder=2, label="Initial, shared")
    ghost.set_dashes((2.2, 1.8))
    for k, v in zip(ks, init_mean["matching"].to_numpy()):
        table.append(dict(panel="H", task="matching", step=0, k=int(k), mean=float(v)))
    handles = []
    for task, (marker, color, face) in TASK_MARKS.items():
        means, lo, hi = [], [], []
        for k in ks:
            values = ev[(ev.step == STEPS) & (ev.family == task) & (ev["index"] == k)].sort_values("seed").cumulative.to_numpy()
            assert len(values) == 20
            res = run.bootstrap(values)
            means.append(res["mean"]); lo.append(res["ci95_low"]); hi.append(res["ci95_high"])
            table.append(dict(panel="H", task=task, step=STEPS, k=int(k), **res))
        means, lo, hi = map(np.array, (means, lo, hi))
        ax.errorbar(ks + KX, means, yerr=[means - lo, hi - means], color=color, marker=marker, ls="-", lw=LW_DATA, ms=MARKER_MS,
                    mfc=face, mew=LW_EDGE, elinewidth=LW_ERR, capsize=2, zorder=4)
        uni = cap[cap.family == task].sort_values("seed").field_energy_fidelity.to_numpy()
        res = run.bootstrap(uni)
        ax.errorbar([0], [res["mean"]], yerr=[[res["mean"] - res["ci95_low"]], [res["ci95_high"] - res["mean"]]], color=color,
                    marker=marker, ls="none", ms=MARKER_MS, mfc=face, mew=LW_EDGE, elinewidth=LW_ERR, capsize=2, zorder=4)
        table.append(dict(panel="H", task=task, step=STEPS, k="uniform", **res))
        handles.append(Line2D([], [], color=color, marker=marker, mfc=face, mew=LW_EDGE, lw=LW_DATA, ms=MARKER_MS,
                              label=TASK_NAMES[task]))
    handles.append(ghost)
    # effective ranks quoted in the caption (spec panel I, demoted to the SI)
    for name, frame in (("effective_rank", spectra), ("gauge_effective_rank", gauge)):
        for (family, step), g in frame.groupby(["family", "step"]):
            if step in (0, STEPS):
                assert g.seed.nunique() == 20
                table.append(dict(panel="caption", task=family, step=int(step), k=name, **run.bootstrap(g.effective_rank)))
    x_sep = 0.5 + KX / 2.0
    for lo_y, hi_y in ((0.0, 0.31), (0.50, 1.09)):        # hairline skips the 'not fitted' tag
        ax.plot([x_sep, x_sep], [lo_y, hi_y], color=MUTE, lw=LW_HAIR, zorder=1)
    ax.text(0, 0.335, "not\nfitted", ha="center", va="bottom", fontsize=PT_SMALL, color=MUTE, linespacing=1.1)
    reference_line(ax, 0.95, axis="y", label=None, span=(x_sep, 6.5 + KX))
    ax.text(6.4 + KX, 0.915, "0.95", ha="right", va="top", fontsize=PT_SMALL, color=MUTE)
    ax.set(xlim=(-.78, 6.5 + KX), ylim=(0, 1.09), xticks=[0] + list(ks + KX), yticks=[0, .5, 1],
           xlabel="Best k fitted directions", ylabel="Energy captured")
    ax.set_xticklabels(["uniform", "1", "2", "3", "4", "5", "6"])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.03, -0.01), frameon=False, fontsize=PT_LEGEND,
              handlelength=1.6, labelspacing=.25, borderaxespad=.1, handletextpad=.5)
    style_panel(ax, grid="y")
    ax.yaxis.labelpad = 1.0
    ax.tick_params(axis="y", pad=1.0, length=2.0)     # keeps H's letter clear of D's axes box
    badge_on_axes(ax, (1.0 + KX, 0.30), "oracle")


# ── assembly ──────────────────────────────────────────────────────────────
def build():
    DEST.mkdir(exist_ok=True)
    curves_df = load_curves()
    floors = load_floors()
    contrasts, seeds = load_contrasts()
    load_selection()
    ev, cap, spectra, gauge = load_capture()
    profile = load_profile()
    profile.to_csv(DEST / "initial_profile_source.csv", index=False)
    table = []

    canvas = NativeCanvas(490 / 72, 3, row_weights=[118, 116, 116], hgutter_pt=37, vgutter_pt=40,
                          margins=Margins(left=58, right=15, top=24, bottom=36))
    a = canvas.panel("A", 0, 0, 5, schematic=True, lock=False, title="Two targets on one shared tree")
    panel_targets(a)
    b = canvas.panel("B", 0, 5, 7, schematic=True, lock=False, title="Three credit rules over the six sites")
    panel_rules(b, profile)

    selected = curves_df[curves_df.selected_rate]
    common = curves_df[curves_df.common_rate]
    c = canvas.panel("C", 1, 0, 4, title="Pairwise: profile suffices")
    curves(c, selected[selected.task == "matching"], floors["matching"], legend=True, ylabel=True, table=table, panel="C")
    d = canvas.panel("D", 1, 4, 4, title="Quartic: only exact learns")
    curves(d, selected[selected.task == "quartet"], floors["quartet"], table=table, panel="D")
    e = canvas.panel("E", 1, 8, 4, title="Common rate: profile ahead")
    row = contrasts[(contrasts.sensitivity == "common_rate") & (contrasts.task == "matching")
                    & (contrasts.contrast == "calibrated_broadcast minus exact")].iloc[0]
    lower = int(row.n_seeds - row.positive_seeds)
    assert lower == 18 and abs(row["mean"] + 0.0089) < 5e-4
    table.append(dict(panel="E", **row.to_dict()))
    curves(e, common[common.task == "matching"], floors["matching"], table=table, panel="E",
           note=[(f"all rules {COMMON_RATE}", PT_SMALL, MUTE), ("profile − exact", PT_SMALL, INK),
                 (f"{signed(row['mean'])} ({lower}/{int(row.n_seeds)})", PT_ANNOT, INK)])

    fpanel = canvas.panel("F", 2, 0, 4, title="Exact quartic: 17/20 at floor")
    endpoints(fpanel, curves_df, floors, table=table)
    g = canvas.panel("G", 2, 4, 4, title="Quartic pays a larger deficit")
    forest(g, contrasts, seeds, table=table)
    h = canvas.panel("H", 2, 8, 4, title="One vs four credit directions")
    capture(h, ev, cap, spectra, gauge, table=table)

    canvas.declare_reserve("F", top=ROW2_RESERVE_PT)      # row 2 letters clear row 1's x labels
    canvas.lock_reserves()
    findings = list(canvas.align_letters())
    path = DEST / "credit_interaction_bridge_native.pdf"
    problems = findings + list(canvas.save(path, name="credit_interaction_bridge_native", dpi=200, lock=False))
    pd.DataFrame(table).to_csv(DEST / "figure_source.csv", index=False)
    inputs = [SOURCE / "all_curves.csv", SOURCE / "paired_contrasts.csv", SOURCE / "paired_seed_contrasts.csv",
              SOURCE / "task_variance_and_noise_floor.csv", run.OUT / "protocol_freeze.json", run.OUT / "selection_freeze.json",
              CAPTURE / "matched_bridge_eigenvalues.csv", CAPTURE / "matched_bridge_capture.csv",
              CAPTURE / "matched_bridge_spectra.csv", CAPTURE / "matched_bridge_gauge_spectra.csv"]
    states = sorted(glob.glob(str(run.OUT / "runs/fresh/algebraic/seed_*_task_*_states.npz")))
    run.write(DEST / "figure_provenance.json", dict(
        figure=str(path.relative_to(run.JOURNAL)), builder_sha256=run.sha(Path(__file__)),
        sources_sha256={str(p.relative_to(run.JOURNAL)): run.sha(p) for p in inputs},
        profile_states_sha256={str(Path(p).relative_to(run.JOURNAL)): run.sha(p) for p in states},
        render_time_files={str((DEST / n).relative_to(run.JOURNAL)): run.sha(DEST / n)
                           for n in ("initial_profile_source.csv", "figure_source.csv")},
        figure_sha256=run.sha(path), n_seed_blocks=20, optimizer="Adam", panels="A–H",
        selection="C, D, F use frozen per-rule rates; E the frozen common rate 0.003; G both; H the exact-rule selected rate",
        field="Unweighted six-site nonsomatic path field q(x); H: cumulative energy of the best k fitted directions "
              "(oracle directions and per-example amplitudes) and the uniform-profile projection; B profile: mean of "
              "the 20 fresh seeds' calibrated six-site profiles (initial_profile_source.csv)",
        layout_notes=problems))
    plt.close(canvas.fig)
    MAIN_PDF.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, MAIN_PDF)
    return problems


if __name__ == "__main__":
    notes = build()
    if notes:
        print("layout notes:", *notes, sep="\n  ")

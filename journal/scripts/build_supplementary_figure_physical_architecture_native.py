#!/usr/bin/env python3
"""Supplementary sheet S22 (ident ``physical_architecture``) -- task-matched
serial computation versus grouped and flexible point controls -- rebuilt as
ONE native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/physical_architecture.pdf``)
is a paste of frozen S31 panels B, C, D, G (no generator in the repository)
and S18 panels A-D (``build_supplementary_figures_s17_s20_native.py``).  This
builder reads ONLY the frozen tables under ``source_data/`` and redraws the
same eight panels, same letters, same plotted quantities.  Nothing about the
numbers changes: every printed or plotted value is asserted against the table
it comes from, every mean is recomputed from the per-seed rows and asserted
against the study's condition summary, and every per-seed value is drawn as
a fan behind its mean.

Tables read (all frozen):

* ``nonlinear_physical_depth_confirmatory``  condition_summary, seed_outcomes
  (B: the four serial arms of the three-tier ladder);
* ``remaining_physical_experiments``  condition_summary,
  seed_outcomes_with_h3_reference (B: the literal grouped-point control);
* ``physical_depth_h4_factorial``  condition_summary, seed_outcomes (C: the
  four-tier grid -- the work list names ``remaining_physical_experiments``
  for C, but that study holds only the two- and three-tier tasks; the
  four-tier cells live here, as ``build_main_figure_06.py`` reads them);
* ``task_family_alignment``  architecture_effects, seed_outcomes (D);
* ``point_dendrite_credit_controls``  condition_summary, paired_contrasts,
  combined_seed_outcomes (B ceiling, E, F);
* ``physical_alignment_dose``  condition_summary, paired_contrasts,
  combined_seed_outcomes (G, H).

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
mixed_S22.json), panel by panel:

* A is a one-third-width schematic flush left, drawn with the paper's own
  glyphs (``native_schematics.Frame``): the serial D3 tree [2,1,2], the
  resource-identical grouped point (the same eight compartments in one
  stage) and the parameter-matched point MLP.  Both trained somata carry the
  delta-0 arrow; the eight excitatory contacts are drawn on both dendritic
  models; one naming ("8 modules") is used for both; a glyph key sits under
  the drawings.
* B keeps the five-arm D1-D3 ladder but in main Fig. 6D's register: exact
  (serial) BP black, exact-path LocalCA dark red (``bp``), shared-soma
  LocalCA amber (``scalar``), grouped point grey (``point_mlp``), raw
  additive blue (``additive``); accuracy in per cent; the ten seeds of every
  arm at every depth drawn as a fan (each arm dodged as a whole so the four
  arms that coincide at D1 stay four visible marks); the arms keyed in the
  empty upper-left; the point-network ceiling drawn as a solid mute
  hairline (the grouped-point trace owns the grey dashes) with its interval
  as a grey band and its label set clear above the rule at the right; the
  chance rule labelled.
* C stacks the aligned block over the reversed block so each cell is wide
  enough to print; the ramp runs over the observed range (52-87 %) so the
  aligned gradient uses the full ramp; the colour key is a vertical rail
  with its label rotated beside the bar, clear of the block titles; the
  raw-additive row of the reversed block prints "not run"; the outline rule
  (best depth of its row whose 95 % interval clears the runner-up's) is
  applied to the ALIGNED block only, as ``build_main_figure_06.py`` and the
  old sheet do -- the reversed block is flat by design and its 0.8-point D3
  blip stays unmarked; row labels take the arm hues of B.
* D: every paired seed difference drawn behind its mean, the three task
  families keyed in the empty upper-left with the main Fig. 6C ramp and
  markers, the axis tight to the data.
* E, F, G, H preserve the earlier pass (seed fans, broken axes, the slope
  contrast moved to the caption, y limits tight) and take one full-width
  row pair with the same outer margins as the rows above.  G's depth ladder
  is drawn in neutral ink with three line styles and three filled markers
  (open markers stay reserved for the derived row of E and for B's point
  controls) so no hue or glyph on the sheet carries two meanings (the
  ordinal ramp names task families in D, as it does in main Fig. 6C).
* All accuracies on the sheet are in per cent; all differences in points.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D

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
    PT_EMPH,
    SEED_ALPHA,
    SEED_MS,
    SEQ_CMAP,
    Margins,
    NativeCanvas,
    _text_width_pt,
    tint_pct,
)
from journal_style import (  # noqa: E402
    ORDINAL_RAMP,
    label_color,
    style_axis,
    style_direct_color_labels,
    tint_patch,
)
from native_schematics import Frame  # noqa: E402
import build_supplementary_figures_s17_s20_native as s18  # noqa: E402

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data"
OUT = ROOT / "figures" / "supplementary" / "figure_physical_architecture_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
DASHED = (0, (4.2, 2.0))
DOTTED = (0, (1.2, 1.6))
N_SEEDS = 10
ATOL = 1e-6           # summary means are printed to 1e-9; seeds to 1e-4

# credit-rule / architecture register of main Fig. 6D (one meaning per hue)
ARMS = {
    "serial_bp": dict(label="serial BP", color=INK, marker="o", ls=DASHED, filled=True),
    "path": dict(label="exact path", color=COLORS["bp"], marker="^", ls="-", filled=True),
    "shared": dict(label="shared soma", color=COLORS["scalar"], marker="s", ls="-", filled=True),
    "grouped": dict(label="grouped point", color=COLORS["point_mlp"], marker="D", ls=DASHED, filled=False),
    "additive": dict(label="raw additive", color=COLORS["additive"], marker="X", ls=DOTTED, filled=False),
}
C_CEILING = COLORS["point_mlp"]      # the point network is a point control
H4_ROWS = (
    ("serial_bp", "serial_tree", "shunting", "full_bp"),
    ("path", "serial_tree", "shunting", "local_path"),
    ("shared", "serial_tree", "shunting", "local_shared"),
    ("grouped", "grouped_point", "shunting", "full_bp"),
    ("additive", "serial_tree", "raw_additive", "full_bp"),
)
FAMILIES = (  # main Fig. 6C register: ordinal ramp + markers
    ("nested_factor", "nested factors", ORDINAL_RAMP[3], "o"),
    ("flat_factor", "flat factors", ORDINAL_RAMP[2], "s"),
    ("local_ratio", "local ratios", ORDINAL_RAMP[1], "^"),
)
DEPTH_STYLE = {1: ("D1", DOTTED, "o", True), 2: ("D2", DASHED, "s", True),
               3: ("D3", "-", "^", True)}   # all filled: open = derived (E)
ALPHA_LABEL = "task–sensor alignment α"
DODGE_B = {"serial_bp": -0.16, "path": -0.08, "shared": 0.0, "grouped": 0.08, "additive": 0.16}


def csv(folder, name):
    path = SOURCE / folder / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pick(frame, **filters):
    out = frame
    for key, value in filters.items():
        out = out[out[key].eq(value)]
    return out


def depth_axis_label(ax, *, drop_pt=14.0):
    """'serial physical depth D' + a token-size subscript p, centred under
    the axis (mathtext and the precomposed subscript glyph are both off the
    journal face)."""
    from figure_canvas import token_subscript
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    probe = ax.text(0.0, 0.0, "serial physical depth D", fontsize=PT_EMPH, transform=ax.transAxes)
    w = _text_width_pt(probe, renderer)
    probe.remove()
    ext = ax.get_window_extent(renderer)
    axes_w = ext.width * 72.0 / ax.figure.dpi
    axes_h = ext.height * 72.0 / ax.figure.dpi
    x = 0.5 - (w + 4.0) / (2.0 * axes_w)
    token_subscript(ax, x, -drop_pt / axes_h, "serial physical depth D", "p", size=PT_EMPH,
                    sub_size=PT_BASE, color=INK, ha="left", va="top", transform=ax.transAxes,
                    clip_on=False)


def fan(ax, x, values, color, *, half=0.10, ms=SEED_MS, alpha=SEED_ALPHA, zorder=2.0):
    """Per-seed values as a jittered fan behind their mean (sorted, so the
    jitter order is a function of the values alone)."""
    values = np.sort(np.asarray(values, float))
    jitter = np.linspace(-half, half, len(values)) if len(values) > 1 else np.zeros(1)
    ax.plot(x + jitter, values, linestyle="none", marker="o", markersize=ms,
            markerfacecolor=color, markeredgecolor="none", alpha=alpha, zorder=zorder)


def mean_whisker(ax, x, m, lo, hi, color, *, marker="o", filled=True, zorder=4.0,
                 ms=MARKER_MS):
    ax.errorbar([x], [m], yerr=[[m - lo], [hi - m]], fmt=marker, ms=ms,
                markerfacecolor=color if filled else "white",
                markeredgecolor="white" if filled else color,
                markeredgewidth=LW_EDGE, ecolor=color, elinewidth=LW_ERR,
                capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=zorder, linestyle="none")


# ── A: architectures compared ────────────────────────────────────────────
def panel_architectures(ax):
    """Serial D3 [2,1,2], the grouped point (same eight modules, one stage)
    and the parameter-matched point MLP, in the paper's own glyphs."""
    f = Frame(ax)
    W, H = f.w_pt, f.h_pt
    X, Y = f.fx, f.fy
    key_pt = 17.0                    # two-line glyph key at the bottom
    label_pt = 31.0                  # four-line cell labels
    top_pt = 23.0                    # bracket and its two-line caption
    gap = 2.0
    shares = (0.30, 0.40, 0.30)
    avail = W - 2 * gap
    widths = [s * avail for s in shares]
    x0s = [0.0, widths[0] + gap, widths[0] + widths[1] + 2 * gap]
    centres = [x0 + w / 2.0 for x0, w in zip(x0s, widths)]
    glyph_bot = key_pt + label_pt + 23.0      # delta-0 arrow and tag below the soma,
    # the tag's subscript about 3.5 pt clear of the first cell-label line
    glyph_top = H - top_pt - 6.0              # contacts sit above the leaves
    assert glyph_top - glyph_bot > 34.0, (glyph_top, glyph_bot)
    dia = 3.0                                 # contact diameter in this panel

    # serial tree: three stages, [2,1,2] compartments
    base = (X(centres[0]), Y(glyph_bot))
    stages = f.stage_tree(base, Y(glyph_top - glyph_bot - 4.0), 3,
                          branching=[2, 1, 2], width=X(widths[0] - 3.6), rings=True)
    assert [len(s) for s in stages] == [2, 2, 4]
    assert min(p[1] for s in stages for p in s) > base[1], "soma must be the lowest node"
    for px, py in stages[-1]:
        for o in (-1.8, 1.8):
            f.contact((px + X(o), py + Y(3.2)), kind="exc", dia_pt=dia)
    f.error_in(base, side="below")

    # grouped point: the same eight compartments, one stage, in parallel
    base2 = (X(centres[1]), Y(glyph_bot))
    f.soma(base2)
    top_y = Y(glyph_top - 4.0)
    xs = np.linspace(centres[1] - 0.45 * widths[1], centres[1] + 0.45 * widths[1], 8)
    for mx in xs:
        f.dendrite(base2, (X(mx), top_y), level=3)
    for mx in xs:
        f.junction((X(mx), top_y))
        f.contact((X(mx), top_y + Y(3.2)), kind="exc", dia_pt=dia)
    f.error_in(base2, side="below")
    f.require_delta0()

    # point MLP: 3-4-3 fully connected, no soma glyph (it is not a neuron model)
    ys = np.linspace(glyph_bot + 1.0, glyph_top - 6.0, 3)
    layers = [[(centres[2] + dx, y) for dx in np.linspace(-0.40 * widths[2], 0.40 * widths[2], n)]
              for y, n in zip(ys, (3, 4, 3))]
    for lower, upper in zip(layers[:-1], layers[1:]):
        for px, py in lower:
            for qx, qy in upper:
                ax.plot([X(px), X(qx)], [Y(py), Y(qy)], color=tint_pct(INK, 45),
                        lw=LW_HAIR, solid_capstyle="round", zorder=1)
    for nodes in layers:
        for px, py in nodes:
            ax.plot([X(px)], [Y(py)], marker="o", ms=3.2, mfc="white", mec=INK,
                    mew=LW_EDGE, ls="none", zorder=4)

    # cell labels: one naming for the two dendritic models
    labels = (("serial", "tree", "3 stages"),
              ("grouped", "point", "1 stage"),
              ("point", "MLP", "matched", "params"))
    colours = (INK, label_color(COLORS["point_mlp"]), INK)
    for cx, lines, col in zip(centres, labels, colours):
        for i, line in enumerate(lines):
            f.text((X(cx), Y(key_pt + label_pt - 4.0 - i * 7.6)), line, size=PT_BASE,
                   color=col if i < 2 else MUTE)
    # bracket over the two models that share modules and contacts
    by = H - top_pt + 2.0
    bx0, bx1 = x0s[0] + 2.0, x0s[1] + widths[1] - 2.0
    ax.plot([X(bx0), X(bx1)], [Y(by), Y(by)], color=MUTE, lw=LW_HAIR, zorder=1)
    for bx in (bx0, bx1):
        ax.plot([X(bx), X(bx)], [Y(by), Y(by - 3.0)], color=MUTE, lw=LW_HAIR, zorder=1)
    f.text((X((bx0 + bx1) / 2.0), Y(by + 3.0)), "same 8 modules\nand contacts",
           size=PT_BASE, color=MUTE, va="bottom", linespacing=1.1)
    # glyph key, two lines: the two glyphs the caption does not already name
    for ky, kind, label in ((key_pt - 4.0, "exc", "excitatory contact ×8"), (4.0, "module", "module")):
        x = 2.0
        if kind == "exc":
            f.contact((X(x + 1.5), Y(ky)), kind="exc", dia_pt=dia)
        else:
            f.junction((X(x + 1.6), Y(ky)))
        x += 3.2
        t = f.text((X(x + 2.2), Y(ky)), label, size=PT_BASE, color=INK, ha="left")
        ax.figure.canvas.draw()
        x += 2.2 + _text_width_pt(t, ax.figure.canvas.get_renderer())
        print(f"[A] frame {W:.1f} x {H:.1f} pt; key line '{label}' ends at {x:.1f} pt")
        assert x <= W + 0.5, "key wider than the panel"


# ── B: the three-tier ladder ─────────────────────────────────────────────
def load_ladder():
    conf = csv("nonlinear_physical_depth_confirmatory", "condition_summary.csv")
    conf_seed = csv("nonlinear_physical_depth_confirmatory", "seed_outcomes.csv")
    rem = csv("remaining_physical_experiments", "condition_summary.csv")
    rem_seed = csv("remaining_physical_experiments", "seed_outcomes_with_h3_reference.csv")
    ceil = csv("point_dendrite_credit_controls", "condition_summary.csv")
    ceil_seed = csv("point_dendrite_credit_controls", "combined_seed_outcomes.csv")
    assert len(conf_seed) == 270 and conf_seed.seed.nunique() == N_SEEDS
    sel = {
        "serial_bp": dict(regime="aligned", mechanism="shunting", method="bp", transport="backpropagation"),
        "path": dict(regime="aligned", mechanism="shunting", method="local3f", transport="path_transport"),
        "shared": dict(regime="aligned", mechanism="shunting", method="local3f", transport="per_soma_shared"),
        "additive": dict(regime="aligned", mechanism="additive", method="bp", transport="backpropagation"),
    }
    series = {}
    for key, flt in sel.items():
        rows = pick(conf, **flt).sort_values("depth")
        assert list(rows.depth) == [1, 2, 3] and (rows.n_seeds == N_SEEDS).all()
        seeds = []
        for depth, r in zip((1, 2, 3), rows.itertuples()):
            z = pick(conf_seed, depth=depth, **flt).test_accuracy.to_numpy(float)
            assert len(z) == N_SEEDS
            np.testing.assert_allclose(z.mean(), r.mean_test_accuracy, rtol=0, atol=ATOL)
            assert r.ci95_low_test_accuracy <= r.mean_test_accuracy <= r.ci95_high_test_accuracy
            seeds.append(100.0 * z)
        series[key] = dict(x=rows.depth.to_numpy(float),
                           m=100.0 * rows.mean_test_accuracy.to_numpy(float),
                           lo=100.0 * rows.ci95_low_test_accuracy.to_numpy(float),
                           hi=100.0 * rows.ci95_high_test_accuracy.to_numpy(float),
                           seeds=seeds)
    rows = pick(rem, hierarchy=3, regime="aligned", architecture="grouped_point",
                credit="full_bp").sort_values("depth")
    assert list(rows.depth) == [1, 2, 3] and (rows.n_seeds == N_SEEDS).all()
    seeds = []
    for depth, r in zip((1, 2, 3), rows.itertuples()):
        z = pick(rem_seed, hierarchy=3, regime="aligned", architecture="grouped_point",
                 credit="full_bp", depth=depth).test_accuracy.to_numpy(float)
        assert len(z) == N_SEEDS
        np.testing.assert_allclose(z.mean(), r.mean_test_accuracy, rtol=0, atol=ATOL)
        seeds.append(100.0 * z)
    series["grouped"] = dict(x=rows.depth.to_numpy(float),
                             m=100.0 * rows.mean_test_accuracy.to_numpy(float),
                             lo=100.0 * rows.ci_low.to_numpy(float),
                             hi=100.0 * rows.ci_high.to_numpy(float), seeds=seeds)
    top = pick(ceil, regime="aligned", architecture="point_mlp_total", credit="full_bp")
    assert len(top) == 1 and int(top.iloc[0].n_seeds) == N_SEEDS
    top = top.iloc[0]
    z = pick(ceil_seed, regime="aligned", architecture="point_mlp_total",
             credit="full_bp").test_accuracy.to_numpy(float)
    assert len(z) == N_SEEDS
    np.testing.assert_allclose(z.mean(), top.mean_test_accuracy, rtol=0, atol=ATOL)
    ceiling = dict(m=100.0 * float(top.mean_test_accuracy),
                   lo=100.0 * float(top.ci95_low_test_accuracy),
                   hi=100.0 * float(top.ci95_high_test_accuracy), seeds=100.0 * z)
    # the serial BP D3 run is one run: the ladder, the point-control reference
    # and the dose endpoint all index it
    ref = pick(ceil, regime="aligned", architecture="serial_tree", credit="full_bp", depth=3).iloc[0]
    np.testing.assert_allclose(ref.mean_test_accuracy, series["serial_bp"]["m"][2] / 100.0, rtol=0, atol=ATOL)
    return series, ceiling


def panel_ladder(ax, series, ceiling):
    dodge = DODGE_B
    ends = {}
    for key, spec in ARMS.items():
        s = series[key]
        col = spec["color"]
        # every arm is dodged as a whole (fan, trace and marker) so the four
        # arms that coincide at D1 stay four visible marks
        for x, seeds in zip(s["x"], s["seeds"]):
            fan(ax, x + dodge[key], seeds, col, half=0.045, ms=SEED_MS * 0.8, alpha=0.5)
        ax.plot(s["x"] + dodge[key], s["m"], color=col, lw=LW_DATA, ls=spec["ls"],
                solid_capstyle="round", zorder=2.4)
        ax.errorbar(s["x"] + dodge[key], s["m"], yerr=[s["m"] - s["lo"], s["hi"] - s["m"]],
                    fmt=spec["marker"], ms=MARKER_MS,
                    markerfacecolor=col if spec["filled"] else "white",
                    markeredgecolor="white" if spec["filled"] else col,
                    markeredgewidth=LW_EDGE, ecolor=col, elinewidth=LW_ERR,
                    capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=3.0, linestyle="none")
        ends[key] = float(s["m"][-1])
        print(f"[B] {spec['label']}: " + ", ".join(
            f"D{int(x)} {m:.2f} [{lo:.2f}, {hi:.2f}] seeds {z.min():.2f}-{z.max():.2f}"
            for x, m, lo, hi, z in zip(s["x"], s["m"], s["lo"], s["hi"], s["seeds"])))
    # point-network ceiling: interval band + solid mute hairline (the grey
    # dashes belong to the grouped-point trace), label clear above the rule
    # at the right end (the upper-left holds the key)
    x0, x1 = 0.72, 3.36
    tint_patch(ax, ("rect", x0, ceiling["lo"], x1 - x0, ceiling["hi"] - ceiling["lo"]),
               color=C_CEILING, pct=10, edge=False, radius_pt=0.6, zorder=0.6, clip_on=True)
    ax.plot([x0, x1], [ceiling["m"]] * 2, color=MUTE, lw=LW_HAIR, ls="-",
            zorder=1.4, solid_capstyle="butt")
    ax.annotate(f"point network {ceiling['m']:.1f} %", xy=(x1 - 0.04, ceiling["m"]),
                xycoords="data", xytext=(0.0, 2.5), textcoords="offset points",
                ha="right", va="bottom", fontsize=PT_BASE, color=label_color(C_CEILING))
    print(f"[B] ceiling {ceiling['m']:.2f} [{ceiling['lo']:.2f}, {ceiling['hi']:.2f}]")
    # chance for the balanced binary task; its label sits under the rule,
    # clear of the additive D3 seeds that start at 50.0
    ax.plot([x0, x1], [50.0, 50.0], color=MUTE, lw=LW_REF, dashes=(2.6, 2.0), zorder=1.0)
    ax.annotate("chance", xy=(x1 - 0.04, 50.0), xycoords="data", xytext=(0.0, -2.6),
                textcoords="offset points", ha="right", va="top", fontsize=PT_BASE, color=MUTE)
    ax.set_xlim(x0, x1)
    ax.set_xticks([1, 2, 3], ["D1", "D2", "D3"])
    lo_seed = min(float(z.min()) for s in series.values() for z in s["seeds"])
    ax.set_ylim(min(lo_seed - 1.6, 48.0), ceiling["m"] + 6.0)
    ax.set_yticks([50, 60, 70, 80, 90, 100])
    ax.set_ylabel("test accuracy (%)")
    depth_axis_label(ax)
    handles = [Line2D([], [], color=spec["color"], lw=LW_DATA, ls=spec["ls"], marker=spec["marker"],
                      ms=MARKER_MS * 0.9,
                      markerfacecolor=spec["color"] if spec["filled"] else "white",
                      markeredgecolor="white" if spec["filled"] else spec["color"],
                      markeredgewidth=LW_EDGE, label=spec["label"])
               for spec in ARMS.values()]
    # the key sits in the empty upper-left: below the ceiling band, above the
    # serial BP trace where it passes the key's right edge (asserted below)
    y_top = 96.0
    leg = ax.legend(handles=handles, loc="upper left",
                    bbox_to_anchor=(0.0, (y_top - ax.get_ylim()[0]) / np.diff(ax.get_ylim())[0]),
                    frameon=False, fontsize=PT_BASE, handlelength=1.2, handletextpad=0.4,
                    borderaxespad=0.0, labelspacing=0.15, borderpad=0.0)
    return leg


def check_ladder_key(ax, cv, leg, series, ceiling):
    """The key's box must clear the ceiling band above and every trace below."""
    dodge = DODGE_B
    cv.fig.canvas.draw()
    bb = leg.get_window_extent(cv.fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    assert y1 < ceiling["lo"] - 0.8, (y1, ceiling["lo"])
    for key, s in series.items():
        # traces are straight between depths: the highest point under the key is at x1
        y_at = np.interp(min(x1, 3.0) - dodge[key], s["x"], s["m"])
        assert y_at < y0 - 0.8, (key, y_at, y0)
        for x, seeds in zip(s["x"], s["seeds"]):
            if x0 - 0.2 <= x <= x1 + 0.2:
                assert float(np.max(seeds)) < y0 - 0.8, (key, x, float(np.max(seeds)), y0)
    print(f"[B] key box x {x0:.2f}-{x1:.2f}, y {y0:.1f}-{y1:.1f}")


# ── C: the four-tier grid ────────────────────────────────────────────────
def load_h4():
    h4 = csv("physical_depth_h4_factorial", "condition_summary.csv")
    seed = csv("physical_depth_h4_factorial", "seed_outcomes.csv")
    assert len(h4) == 36 and len(seed) == 360 and seed.seed.nunique() == N_SEEDS
    assert (h4.hierarchy == 4).all() and (h4.n_seeds == N_SEEDS).all()
    blocks = {}
    for regime in ("aligned", "rewired_tree"):
        m = np.full((5, 4), np.nan)
        lo = np.full((5, 4), np.nan)
        hi = np.full((5, 4), np.nan)
        for i, (_, arch, mech, credit) in enumerate(H4_ROWS):
            rows = pick(h4, regime=regime, architecture=arch, mechanism=mech, credit=credit)
            for r in rows.itertuples():
                col = int(r.depth) - 1
                z = pick(seed, regime=regime, architecture=arch, mechanism=mech,
                         credit=credit, depth=int(r.depth)).test_accuracy.to_numpy(float)
                assert len(z) == N_SEEDS
                np.testing.assert_allclose(z.mean(), r.mean_test_accuracy, rtol=0, atol=ATOL)
                assert r.ci_low <= r.mean_test_accuracy <= r.ci_high
                m[i, col] = 100.0 * r.mean_test_accuracy
                lo[i, col] = 100.0 * r.ci_low
                hi[i, col] = 100.0 * r.ci_high
        blocks[regime] = (m, lo, hi)
    a, r = blocks["aligned"][0], blocks["rewired_tree"][0]
    assert np.isfinite(a).all() and np.isfinite(r[:4]).all() and np.isnan(r[4]).all(), \
        "raw additive ran under aligned placement only"
    assert np.isfinite(a).sum() + np.isfinite(r).sum() == 36
    return blocks


def panel_h4(ax, cv, blocks):
    aligned, a_lo, a_hi = blocks["aligned"]
    rev, r_lo, r_hi = blocks["rewired_tree"]
    combined = np.vstack([aligned, np.full((1, 4), np.nan), rev])     # 11 x 4
    finite = combined[np.isfinite(combined)]
    vmin, vmax = float(np.floor(finite.min())), float(np.ceil(finite.max()))
    assert vmin == 52.0 and vmax == 87.0, (vmin, vmax)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = SEQ_CMAP.copy()
    cmap.set_bad((0, 0, 0, 0))
    image = ax.imshow(np.ma.masked_invalid(combined), cmap=cmap, norm=norm,
                      aspect="auto", interpolation="nearest", zorder=1)
    outlined = []
    for block, (m, lo, hi), off in (("aligned", blocks["aligned"], 0),
                                     ("reversed", blocks["rewired_tree"], 6)):
        for row in range(5):
            cols = np.flatnonzero(np.isfinite(m[row]))
            for col in cols:
                v = m[row, col]
                rgba = cmap(norm(v))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(col, row + off, f"{v:.0f}", ha="center", va="center",
                        fontsize=PT_BASE, color="white" if lum < 0.48 else INK, zorder=3)
            # An outline claims a best depth; as in build_main_figure_06.py it
            # is drawn for the aligned block only -- the reversed block is
            # flat by design (its largest lead is the 0.8-point D3 blip of
            # serial BP, asserted below) and marking it would contradict the
            # result that reversing sensor order removes the depth benefit.
            if block == "reversed" or len(cols) < 2:
                continue
            order = cols[np.argsort(m[row, cols])[::-1]]
            best, runner = int(order[0]), int(order[1])
            if lo[row, best] > hi[row, runner]:
                x0, x1 = best - 0.48, best + 0.48
                y0, y1 = row + off - 0.47, row + off + 0.47
                # four open segments (projecting caps close the corners): a
                # closed rectangle path would be read as a box on the value
                for xs, ys in (((x0, x1), (y0, y0)), ((x0, x1), (y1, y1)),
                               ((x0, x0), (y0, y1)), ((x1, x1), (y0, y1))):
                    ax.plot(xs, ys, color=INK, lw=LW_ERR, solid_capstyle="projecting", zorder=5)
                outlined.append((block, ARMS[H4_ROWS[row][0]]["label"], f"D{best + 1}",
                                 m[row, best], lo[row, best], hi[row, runner]))
    print("[C] outlined: " + "; ".join(
        f"{b} {lab} {d} ({v:.1f} %, lower {l:.1f} > runner-up upper {u:.1f})"
        for b, lab, d, v, l, u in outlined))
    assert [(b, lab, d) for b, lab, d, *_ in outlined] == [
        ("aligned", "serial BP", "D3"), ("aligned", "shared soma", "D4")]
    # the unmarked reversed block: the largest best-over-runner-up lead of
    # any of its rows is serial BP D3 over D1, 0.8 points (the caption's number)
    leads = []
    for row in range(4):                      # the raw-additive row was not run
        order = np.argsort(rev[row])[::-1]
        leads.append((float(rev[row, order[0]] - rev[row, order[1]]), row,
                      int(order[0]) + 1, int(order[1]) + 1))
    lead, row, best, runner = max(leads)
    assert (row, best, runner) == (0, 3, 1) and abs(lead - 0.8) < 0.05, leads
    print(f"[C] reversed block unmarked; largest lead {lead:.2f} points "
          f"({ARMS[H4_ROWS[row][0]]['label']} D{best} over D{runner})")
    ax.text(1.5, 10.0, "not run", ha="center", va="center", fontsize=PT_BASE, color=MUTE, zorder=3)
    # block titles: the aligned block's above its first row, the reversed
    # block's in the gap row, both left-aligned on the grid
    ax.text(-0.5, -1.05, "aligned sensors", ha="left", va="center", fontsize=PT_BASE,
            color=INK, zorder=3)
    ax.text(-0.5, 5.0, "reversed tier placement", ha="left", va="center", fontsize=PT_BASE,
            color=INK, zorder=3)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(10.5, -1.6)
    ax.set_xticks(range(4), ["D1", "D2", "D3", "D4"])
    ticks = list(range(5)) + list(range(6, 11))
    labels = [ARMS[k]["label"] for k, *_ in H4_ROWS] * 2
    ax.set_yticks(ticks, labels)
    for tick, (key, *_) in zip(ax.get_yticklabels(), list(H4_ROWS) * 2):
        tick.set_color(label_color(ARMS[key]["color"]))
    ax.tick_params(axis="both", length=0, pad=2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    depth_axis_label(ax)
    return image


def colour_rail(cv, ax, image):
    """The accuracy key: a 4 pt rail in the right outer margin beside C, its
    label rotated alongside the bar so nothing sits on the block titles."""
    from matplotlib.ticker import NullLocator
    box = ax.get_position()
    x = box.x1 * cv.width_pt + 5.0
    cax = cv.fig.add_axes([x / cv.width_pt, box.y0 + 0.16 * box.height, 4.0 / cv.width_pt,
                           0.68 * box.height])
    cv.bind_satellite(cax, ax)
    cbar = cv.fig.colorbar(image, cax=cax, ticks=[55, 65, 75, 85])
    cbar.outline.set_linewidth(LW_HAIR)
    cbar.outline.set_edgecolor(COLORS["edge"])
    cbar.ax.yaxis.set_minor_locator(NullLocator())
    cbar.ax.set_yticklabels(["55", "65", "75", "85"])
    cbar.ax.tick_params(labelsize=PT_BASE, width=LW_HAIR, length=2.2, pad=1.5,
                        color=COLORS["edge"], labelcolor=INK)
    cbar.set_label("test accuracy (%)", fontsize=PT_BASE, labelpad=2.0, color=INK)
    return cbar


# ── D: task families under exact-path LocalCA ────────────────────────────
def panel_families(ax):
    effects = csv("task_family_alignment", "architecture_effects.csv")
    seed = csv("task_family_alignment", "seed_outcomes.csv")
    assert len(seed) == 360 and len(effects) == 18
    handles = []
    lo_all, hi_all = [], []
    dodge = {"nested_factor": -0.035, "flat_factor": 0.035, "local_ratio": 0.0}
    for family, label, color, marker in FAMILIES:
        rows = pick(effects, family=family, credit="local3f",
                    estimand="serial_minus_grouped").sort_values("alignment_alpha")
        assert list(rows.alignment_alpha) == [0.0, 0.5, 1.0] and (rows.n_pairs == N_SEEDS).all()
        x = rows.alignment_alpha.to_numpy(float)
        m = 100.0 * rows.mean_difference.to_numpy(float)
        lo = 100.0 * rows.ci95_low.to_numpy(float)
        hi = 100.0 * rows.ci95_high.to_numpy(float)
        printed = []
        for alpha, mm, r in zip(x, m, rows.itertuples()):
            s = pick(seed, family=family, credit="local3f", alignment_alpha=alpha,
                     architecture="serial").set_index("seed").test_accuracy
            g = pick(seed, family=family, credit="local3f", alignment_alpha=alpha,
                     architecture="grouped_point").set_index("seed").test_accuracy
            d = 100.0 * (s - g).dropna()
            assert len(d) == N_SEEDS
            np.testing.assert_allclose(d.mean(), mm, rtol=0, atol=ATOL)
            assert int((d > 0).sum()) == int(r.positive_pairs)
            fan(ax, alpha + dodge[family], d.to_numpy(), color, half=0.024)
            lo_all.append(float(d.min())); hi_all.append(float(d.max()))
            printed.append(f"α={alpha:g} {mm:.2f} [{100 * r.ci95_low:.2f}, {100 * r.ci95_high:.2f}] "
                           f"seeds {d.min():.2f}-{d.max():.2f} ({int(r.positive_pairs)}/10 > 0)")
        print(f"[D] {label}: " + "; ".join(printed))
        ax.plot(x + dodge[family], m, color=color, lw=LW_DATA, zorder=2.4, solid_capstyle="round")
        ax.errorbar(x + dodge[family], m, yerr=[m - lo, hi - m], fmt=marker, ms=MARKER_MS, markerfacecolor=color,
                    markeredgecolor="white", markeredgewidth=LW_EDGE, ecolor=color,
                    elinewidth=LW_ERR, capsize=ERR_CAPSIZE, capthick=LW_ERR, zorder=3.0,
                    linestyle="none")
        handles.append(Line2D([], [], color=color, lw=LW_DATA, marker=marker, ms=MARKER_MS,
                              markerfacecolor=color, markeredgecolor="white",
                              markeredgewidth=LW_EDGE, label=label))
        lo_all += list(lo); hi_all += list(hi)
    ax.axhline(0.0, color=MUTE, lw=LW_REF, dashes=(2.6, 2.0), zorder=1.0)
    ax.set_xlim(-0.09, 1.09)
    ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    ymin, ymax = min(lo_all), max(hi_all)
    pad = 0.06 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_yticks([-10, 0, 10, 20, 30])
    ax.set_xlabel(ALPHA_LABEL)
    ax.set_ylabel("serial − grouped point (pp)")
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=PT_BASE,
              handlelength=2.0, handletextpad=0.5, borderaxespad=0.2, labelspacing=0.35)


def seed_forest(host, segments, rows, *, xlabel, ticks, reference=0.0, seed_jitter=.16):
    """``build_supplementary_figures_s17_s20_native._seed_forest`` with one
    change: a row's seed fan is drawn on the segment that holds it only.  The
    original drew every fan on every segment and let the clip hide the copies,
    which leaves invisible marks in the PDF under the neighbouring panel."""
    segs = s18._broken_x_segments(host, segments)

    def seg_for(x):
        return next(s for s in segs if s.get_xlim()[0] <= x <= s.get_xlim()[1])
    for index, row in enumerate(rows):
        color = COLORS.get(row.get("color", "ink"), row.get("color"))
        seeds = np.sort(np.asarray(row["seeds"], float))
        assert len(seeds) == N_SEEDS
        ys = index + np.linspace(-seed_jitter, seed_jitter, len(seeds))
        s = seg_for(float(seeds.min()))
        assert s is seg_for(float(seeds.max()))
        s.plot(seeds, ys, linestyle="none", marker="o", markersize=SEED_MS, markerfacecolor=color,
               markeredgecolor="none", alpha=SEED_ALPHA, zorder=2, clip_on=True)
        m, lo, hi = row["mean"], row["lo"], row["hi"]
        s = seg_for(m)
        assert s is seg_for(lo) is seg_for(hi)
        s.plot([lo, hi], [index, index], color=color, lw=LW_ERR, zorder=3, solid_capstyle="butt")
        for xb in (lo, hi):
            s.plot([xb, xb], [index - .13, index + .13], color=color, lw=LW_ERR, zorder=3,
                   solid_capstyle="butt")
        hollow = row.get("hollow", False)
        s.plot([m], [index], linestyle="none", marker="o", markersize=MARKER_MS,
               markerfacecolor="white" if hollow else color,
               markeredgecolor=color if hollow else "white",
               markeredgewidth=LW_ERR if hollow else LW_HAIR, zorder=4)
    for s in segs:
        s.set_ylim(len(rows) - .45, -.55)
    if reference is not None:
        seg_for(reference).axvline(reference, color=COLORS["mute"], lw=LW_HAIR, zorder=0)
    segs[0].set_yticks(range(len(rows)), [r["label"] for r in rows])
    for s, t in zip(segs, ticks):
        s.set_xticks(t)
    host.set_xlabel(xlabel)
    host.xaxis.set_label_coords(.5, -.16)
    return segs


def host_row_labels(host, segs, rows):
    """Move the forest's row labels from the first inset to the host axes:
    the host's y axis is what the column lock measures, so the label column
    is reserved by measurement rather than by a hand-declared width."""
    segs[0].set_yticks([])
    host.set_ylim(*segs[0].get_ylim())
    host.set_yticks(range(len(rows)), [r["label"] for r in rows])
    host.tick_params(axis="y", length=0, pad=4.0, labelsize=PT_BASE, labelcolor=INK)
    for spine in host.spines.values():
        spine.set_visible(False)


# ── E, F: the point-control panels (earlier pass, preserved) ─────────────
def panel_star(ax, cv):
    pc = csv("point_dendrite_credit_controls", "paired_contrasts.csv").set_index("contrast")
    ps = csv("point_dendrite_credit_controls", "combined_seed_outcomes.csv")
    conf_seed = csv("nonlinear_physical_depth_confirmatory", "seed_outcomes.csv")
    rows = []
    for depth in (1, 2, 3):
        seeds = s18._paired_seed_diff(
            ps, dict(architecture="serial_tree", regime="aligned", credit="full_bp", depth=depth),
            dict(architecture="all_active_star", regime="aligned", credit="full_bp", depth=depth))
        r = pc.loc[f"serial_minus_star__aligned__d{depth}"]
        assert int(r.n_pairs) == N_SEEDS
        np.testing.assert_allclose(seeds.mean(), 100 * r.mean_difference, rtol=0, atol=ATOL)
        # the serial side of every pair is the ladder's serial BP run
        ladder = pick(conf_seed, regime="aligned", mechanism="shunting", method="bp",
                      depth=depth).set_index("seed").test_accuracy
        star = pick(ps, architecture="serial_tree", regime="aligned", credit="full_bp",
                    depth=depth).set_index("seed").test_accuracy
        np.testing.assert_allclose(ladder.sort_index().to_numpy(), star.sort_index().to_numpy(),
                                   rtol=0, atol=1e-9)
        rows.append(dict(label=f"D{depth}", mean=100 * r.mean_difference, lo=100 * r.ci95_low,
                         hi=100 * r.ci95_high, seeds=seeds.to_numpy(), color="ink"))
    rev = s18._paired_seed_diff(
        ps, dict(architecture="serial_tree", regime="rewired_tree", credit="full_bp", depth=3),
        dict(architecture="all_active_star", regime="rewired_tree", credit="full_bp", depth=3))
    ali = s18._paired_seed_diff(
        ps, dict(architecture="serial_tree", regime="aligned", credit="full_bp", depth=3),
        dict(architecture="all_active_star", regime="aligned", credit="full_bp", depth=3))
    inter = (ali - rev).dropna()
    r = pc.loc["serial_star_alignment_interaction__d3"]
    assert len(inter) == N_SEEDS == int(r.n_pairs)
    np.testing.assert_allclose(inter.mean(), 100 * r.mean_difference, rtol=0, atol=ATOL)
    rows.append(dict(label="D3, aligned −\nreversed", mean=100 * r.mean_difference,
                     lo=100 * r.ci95_low, hi=100 * r.ci95_high, seeds=inter.to_numpy(),
                     color="ink", hollow=True))
    segments = [(-1.2, 9.0), (28.5, 33.5)]
    for row in rows:
        assert any(lo <= min(row["seeds"]) and max(row["seeds"]) <= hi for lo, hi in segments), row["label"]
        assert any(lo <= row["lo"] and row["hi"] <= hi for lo, hi in segments), row["label"]
    print("[E] " + "; ".join(f"{r['label'].replace(chr(10), ' ')}: {r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] "
                             f"seeds {min(r['seeds']):.2f}-{max(r['seeds']):.2f}" for r in rows))
    segs = seed_forest(ax, segments, rows, xlabel="serial − all-active star (pp)",
                       ticks=[[0, 5], [30]])
    host_row_labels(ax, segs, rows)
    ax.xaxis.set_label_coords(0.5, -0.20)


def panel_point_controls(ax):
    point = csv("point_dendrite_credit_controls", "condition_summary.csv")
    ps = csv("point_dendrite_credit_controls", "combined_seed_outcomes.csv")
    pc = csv("point_dendrite_credit_controls", "paired_contrasts.csv").set_index("contrast")
    specs = (("point_mlp_active", 0, "active\nmatch", COLORS["point_mlp"]),
             ("point_mlp_total", 0, "total\nmatch", COLORS["point_mlp"]),
             ("serial_tree", 3, "serial\nD3", INK))
    lo_all, hi_all = [], []
    means = {}
    for i, (arch, depth, label, color) in enumerate(specs):
        r = pick(point, regime="aligned", architecture=arch, credit="full_bp", depth=depth)
        assert len(r) == 1
        r = r.iloc[0]
        assert int(r.n_seeds) == N_SEEDS
        z = pick(ps, regime="aligned", architecture=arch, credit="full_bp",
                 depth=depth).test_accuracy.to_numpy(float)
        assert len(z) == N_SEEDS
        np.testing.assert_allclose(z.mean(), r.mean_test_accuracy, rtol=0, atol=ATOL)
        m, lo, hi = (100 * r.mean_test_accuracy, 100 * r.ci95_low_test_accuracy,
                     100 * r.ci95_high_test_accuracy)
        fan(ax, i, 100 * z, color, half=0.16)
        mean_whisker(ax, i, m, lo, hi, color)
        means[arch] = z.mean()
        lo_all.append(float(100 * z.min())); hi_all.append(float(100 * z.max()))
        print(f"[F] {label.replace(chr(10), ' ')}: {m:.2f} [{lo:.2f}, {hi:.2f}] seeds {100 * z.min():.2f}-{100 * z.max():.2f}")
    for name, arch in (("serial_d3_minus_active_matched_point_mlp", "point_mlp_active"),
                       ("serial_d3_minus_total_matched_point_mlp", "point_mlp_total")):
        np.testing.assert_allclose(means["serial_tree"] - means[arch], pc.loc[name].mean_difference,
                                   rtol=0, atol=ATOL)
    ax.set_xticks(range(3), [s[2] for s in specs])
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(min(lo_all) - 0.6, 100.4)
    ax.set_yticks([90, 92, 94, 96, 98, 100])
    ax.set_ylabel("test accuracy (%)")
    style_axis(ax, grid="y")


# ── G, H: the alignment dose ─────────────────────────────────────────────
def load_dose():
    dose = csv("physical_alignment_dose", "condition_summary.csv")
    dseed = csv("physical_alignment_dose", "combined_seed_outcomes.csv")
    dc = csv("physical_alignment_dose", "paired_contrasts.csv").set_index("estimand")
    conf_seed = csv("nonlinear_physical_depth_confirmatory", "seed_outcomes.csv")
    assert len(dose) == 15 and len(dseed) == 150 and (dose.n_seeds == N_SEEDS).all()
    wide = dseed.pivot_table(index=["alignment_alpha", "seed"], columns="depth", values="test_accuracy")
    for r in dose.itertuples():
        z = wide.loc[r.alignment_alpha][int(r.depth)].to_numpy(float)
        assert len(z) == N_SEEDS
        np.testing.assert_allclose(z.mean(), r.mean_test_accuracy, rtol=0, atol=ATOL)
    # the alpha = 1 endpoint is the ladder's serial BP run; alpha = 0 is its
    # zero-alignment control
    for alpha, regime in ((1.0, "aligned"), (0.0, "zero_alignment")):
        for depth in (1, 2, 3):
            a = wide.loc[alpha][depth].sort_index().to_numpy(float)
            b = pick(conf_seed, regime=regime, mechanism="shunting", method="bp",
                     depth=depth).set_index("seed").test_accuracy.sort_index().to_numpy(float)
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
    return dose, wide, dc


def panel_dose(ax, dose, wide):
    ends = {}
    lo_all, hi_all = [], []
    for depth in (1, 2, 3):
        label, ls, marker, filled = DEPTH_STYLE[depth]
        p = dose[dose.depth.eq(depth)].sort_values("alignment_alpha")
        x = p.alignment_alpha.to_numpy(float)
        m = 100 * p.mean_test_accuracy.to_numpy(float)
        lo = 100 * p.ci95_low_test_accuracy.to_numpy(float)
        hi = 100 * p.ci95_high_test_accuracy.to_numpy(float)
        dx = {1: -0.028, 2: 0.0, 3: 0.028}[depth]
        for alpha in x:
            z = 100 * wide.loc[alpha][depth].to_numpy(float)
            fan(ax, alpha + dx, z, INK, half=0.012, ms=SEED_MS * 0.85, alpha=0.45)
            lo_all.append(float(z.min())); hi_all.append(float(z.max()))
        ax.plot(x + dx, m, color=INK, lw=LW_DATA, ls=ls, zorder=2.4, solid_capstyle="round")
        ax.errorbar(x + dx, m, yerr=[m - lo, hi - m], fmt=marker, ms=MARKER_MS,
                    markerfacecolor=INK if filled else "white",
                    markeredgecolor="white" if filled else INK, markeredgewidth=LW_EDGE,
                    ecolor=INK, elinewidth=LW_ERR, capsize=ERR_CAPSIZE, capthick=LW_ERR,
                    zorder=3.0, linestyle="none")
        ends[depth] = float(m[-1])
        print(f"[G] {label}: " + ", ".join(f"α={a:g} {mm:.2f} [{l:.2f}, {h:.2f}]"
                                          for a, mm, l, h in zip(x, m, lo, hi)))
    ax.set_xlim(-0.06, 1.17)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(min(lo_all) - 1.5, max(hi_all) + 1.5)
    ax.set_yticks([60, 70, 80, 90])
    ax.set_xlabel(ALPHA_LABEL)
    ax.set_ylabel("test accuracy (%)")
    style_axis(ax, grid="y")
    # direct labels at alpha = 1 (D1 and D2 end 6.6 points apart, one line)
    for depth, dy in ((1, -1.3), (2, 1.3), (3, 0.0)):
        ax.text(1.075, ends[depth] + dy, DEPTH_STYLE[depth][0], ha="left", va="center",
                fontsize=PT_BASE, color=INK)


def panel_depth_benefit(ax, cv, wide, dc):
    rows = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        seeds = 100 * (wide.loc[alpha][3] - wide.loc[alpha][1]).dropna()
        r = dc.loc[f"depth_effect_alpha_{alpha:.2f}"]
        assert len(seeds) == N_SEEDS == int(r.n_pairs)
        np.testing.assert_allclose(seeds.mean(), 100 * r.mean_difference, rtol=0, atol=ATOL)
        assert int((seeds > 0).sum()) == int(r.positive_pairs)
        rows.append(dict(label=f"α = {alpha:g}", mean=100 * r.mean_difference, lo=100 * r.ci95_low,
                         hi=100 * r.ci95_high, seeds=seeds.to_numpy(), color="ink"))
    segments = [(-1.2, 4.5), (28.5, 33.5)]
    for row in rows:
        assert any(lo <= min(row["seeds"]) and max(row["seeds"]) <= hi for lo, hi in segments), row["label"]
    # the two summary contrasts the old panel H drew on the wrong axis are
    # stated in the caption; assert them here so the caption's numbers are sourced
    d = dc.loc["alpha_0.75_minus_0.25"]
    s = dc.loc["within_seed_linear_slope_per_unit_alpha"]
    np.testing.assert_allclose([d.mean_difference, d.ci95_low, d.ci95_high], [0.02919, 0.02649, 0.03174], atol=5e-5)
    np.testing.assert_allclose([s.mean_difference, s.ci95_low, s.ci95_high], [0.25836, 0.25426, 0.26262], atol=5e-5)
    print("[H] " + "; ".join(f"{r['label']}: {r['mean']:.2f} [{r['lo']:.2f}, {r['hi']:.2f}] "
                             f"seeds {min(r['seeds']):.2f}-{max(r['seeds']):.2f}" for r in rows)
          + f"; 0.75−0.25 {100 * d.mean_difference:.2f} [{100 * d.ci95_low:.2f}, {100 * d.ci95_high:.2f}]"
          + f"; slope {100 * s.mean_difference:.1f} [{100 * s.ci95_low:.1f}, {100 * s.ci95_high:.1f}] per unit α")
    segs = seed_forest(ax, segments, rows, xlabel="D3 − D1 accuracy (pp)", ticks=[[0, 2, 4], [30]])
    host_row_labels(ax, segs, rows)
    ax.xaxis.set_label_coords(0.5, -0.20)


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 484.0
ROW_PT = [136.0, 98.0, 102.0]
HGUTTER_PT = 26.0
VGUTTER_PT = 42.0
MARGINS = Margins(left=24.0, right=32.0, top=20.0, bottom=44.0)
# one declared reserve on every panel: the column lock then gives every
# panel of a row one axes width by construction (the S21 precedent).  The
# left value is what the widest measured label column needs beyond the gutter
# (C's row labels, E's row labels); the right value is the last x tick label.
RESERVE = dict(left=32.0, right=8.0)


def build(path: Path = OUT):
    series, ceiling = load_ladder()
    blocks = load_h4()
    dose, wide, dc = load_dose()

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 3, row_weights=ROW_PT, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS)
    ax_a = cv.panel("A", 0, 0, 4, schematic=True, title="Architectures compared")
    ax_b = cv.panel("B", 0, 4, 4, grid="y", title="Three-tier task")
    ax_c = cv.panel("C", 0, 8, 4, title="Four-tier task")
    ax_d = cv.panel("D", 1, 0, 4, grid="y", title="Exact path at D3")
    ax_e = cv.panel("E", 1, 4, 4, title="Serial vs star, BP")
    ax_f = cv.panel("F", 1, 8, 4, title="Flexible point controls")
    ax_g = cv.panel("G", 2, 0, 6, grid="y", title="Alignment dose, serial BP")
    ax_h = cv.panel("H", 2, 6, 6, title="Depth benefit versus alignment")
    for name in "ABCDEFGH":
        cv.declare_reserve(name, **RESERVE)

    leg_b = panel_ladder(ax_b, series, ceiling)
    image = panel_h4(ax_c, cv, blocks)
    panel_families(ax_d)
    panel_star(ax_e, cv)
    panel_point_controls(ax_f)
    panel_dose(ax_g, dose, wide)
    panel_depth_benefit(ax_h, cv, wide, dc)
    cv.lock_reserves()
    check_ladder_key(ax_b, cv, leg_b, series, ceiling)
    colour_rail(cv, ax_c, image)
    panel_architectures(ax_a)

    # one key for the estimate glyphs used on every data panel
    seed_h = Line2D([], [], linestyle="none", marker="o", markersize=SEED_MS, markerfacecolor=INK,
                    markeredgecolor="none", alpha=SEED_ALPHA, label="one seed (10 per condition)")
    mean_line = Line2D([], [], linestyle="none", marker="o", markersize=MARKER_MS,
                       markerfacecolor=INK, markeredgecolor="white", markeredgewidth=LW_EDGE)
    cap = Line2D([], [], linestyle="none", marker="|", markersize=ERR_CAPSIZE * 2.0,
                 markeredgewidth=LW_ERR, color=INK)
    bar = LineCollection([], colors=INK, linewidths=LW_ERR)
    mean_h = ErrorbarContainer((mean_line, (cap,), (bar,)), has_xerr=True,
                               label="mean and 95 % seed-bootstrap interval")
    open_h = Line2D([], [], linestyle="none", marker="o", markersize=MARKER_MS,
                    markerfacecolor="white", markeredgecolor=INK, markeredgewidth=LW_ERR,
                    label="derived difference of two rows (E)")
    cv.fig.legend(handles=[seed_h, mean_h, open_h], loc="lower center",
                  bbox_to_anchor=(0.53, 0.0), ncol=3, frameon=False, fontsize=PT_BASE,
                  handlelength=2.2, columnspacing=1.8, handletextpad=0.6, borderaxespad=0.5)
    style_direct_color_labels(cv.fig)
    problems = cv.save(path, name="figure_physical_architecture_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

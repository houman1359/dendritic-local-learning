#!/usr/bin/env python3
"""Credit-first Fig. 1: task-derived credit, neuronal identity versus resolution.

Grid (DESIGN_SPEC §1): ``NativeCanvas`` 490 pt, rows 126 / 114 / 108 pt,
hgutter 40, vgutter 40, margins 36 / 12 / 24 / 38 (the spec's 120 / 116 /
110 split puts the 8-module B at aspect 2.50, above PANEL_ASPECT_MAX 2.40;
the row weights stay inside D2's 104-136 pt band).  Row 2 declares an 11 pt
top reserve so its letters clear D's two-line tick labels (row separation
floor 8.5 pt).  Row 0 is the schematic
row -- A (4 modules, credit entry) and B (8, three deliveries); row 1 holds
the dictionary schematic C (5) beside the six-rule MNIST ladder D (7); row 2
is the cohort family E | F | G (4 modules each).

Waiver (spec D3): E and F are the two halves of one cohort comparison
(identical cohort rows, identical y limits, column-locked at modules 0 and
4) and G is the capture panel beside that pair.  The single 8-module E|F
host the spec prefers cannot pass the panel-aspect audit (300 x 110 pt =
2.73 > PANEL_ASPECT_MAX 2.40), so the spec's recorded fallback is used.

Every schematic element is drawn with the shared glyph library
(``native_schematics.Frame``).  Private helpers (errata #7) cover what the
library does not draw: the D5 amber bus fed from the soma's own error
(``Frame.credit_delivery(mode='neuron')`` still draws the retired barrier
glyph) and the layer bus spanning a ghost neighbour; token-subscript chains
for equation lines; and the three-hue K cycle (``journal_style.K_CYCLE`` is
not defined yet).  This builder performs no fitting, rate selection, or
interval fitting: every number is replayed from frozen Source Data and
re-derived as an assertion.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR, LW_REF,
                           MARKER_MS, PT_ANNOT, PT_LABEL, PT_LEGEND, PT_SMALL,
                           PT_TICK, SEED_ALPHA, SEED_MS, Margins, NativeCanvas,
                           style_panel)
from journal_style import label_color
from credit_tree_schematics import AMBER_TEXT
from native_schematics import CONTACT_DIA_PT, Frame, _text_w_pt

SOURCE = JOURNAL / "source_data"
FRESH = SOURCE / "image_ladder_controls/summaries"
OUT = JOURNAL / "figures/components/credit_first_figure_01.pdf"
RECORDS = SOURCE / "credit_first_figures"
ARMS = ("strict_scalar", "neuron_shared", "projected_k1", "subtree_k3", "exact_path", "decoder_only")
ARM_LABELS = ("Layer\nscalar", "Per\nneuron", "Projected\nK = 1", "Subtrees\nK = 3", "Exact\npath", "Decoder\nonly")
ARCHITECTURES = ("shunting", "additive")
CONTRASTS = ("neuron_shared_minus_strict_scalar", "subtree_k3_minus_projected_k1",
             "exact_path_minus_subtree_k3", "exact_path_minus_neuron_shared")
# Addressed-subtree K cycle (spec D7: dend, soma, exc; never a series hue).
K3_CYCLE = ("dend", "soma", "exc")
CAPTURE_COORDINATE = "voltage"      # G is parameterised on the coordinate
CAPTURE_BASES = (("broadcast_k1", "K = 1"), ("subtrees_k3", "K = 3"), ("exact_k12", "K = 12"))
INK, MUTE = COLORS["ink"], COLORS["mute"]
DASH = (0, (2.2, 1.8))
DIGIT = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], float)


# ── private glyph helpers (library geometry, spec §0.2) ───────────────────
def chain(f, xy, parts, *, size=PT_ANNOT, color=None, ha="left", va="center",
          zorder=6):
    """Lay out plain strings and (base, sub) pairs left-to-right at ``xy``.

    Uses ``Frame.subscript`` for every pair (token subscripts, no mathtext)
    and measures each piece so the run can be centred or right-anchored.
    A PT_SMALL base steps up to PT_ANNOT inside ``Frame.subscript``, so the
    measured width uses the same rule.  Returns (x_start, x_end, {index:
    x_centre}) in frame fractions.
    """
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
    x_start, centres = x, {}
    for i, (p, w) in enumerate(zip(parts, widths)):
        centres[i] = x + f.fx(w / 2.0)
        if isinstance(p, tuple):
            f.subscript((x, y), p[0], p[1], size=size, color=color, ha="left", va=va, zorder=zorder)
        else:
            f.text((x, y), p, size=size, color=color, ha="left", va=va, zorder=zorder)
        x += f.fx(w)
    return x_start, x, centres


def amber_bus(f, points, *, x_lo, x_hi, y, source_at="right", clip=None):
    """Amber credit bus (spec D5): LW_HAIR bus, hairline drops into ``points``.

    ``points`` are frame xy of the targets; the bus runs from ``x_lo`` to
    ``x_hi`` at height ``y`` (frame fractions).  ``source_at`` 'right' /
    'left' puts the r 1.6 source dot at that end of the bus (None: no dot,
    the caller draws the source, e.g. at the soma).  ``clip`` (a patch)
    keeps a layer bus inside its card.  Returns the drawn artists.
    """
    color = COLORS["scalar"]
    line, = f.ax.plot([x_lo, x_hi], [y, y], color=color, lw=f.lw(LW_HAIR),
                      solid_capstyle="round", zorder=5)
    arts = [line]
    if source_at:
        arts.append(f.disc((x_hi if source_at == "right" else x_lo, y), 1.6, fill=color, zorder=6))
    for (px, py) in points:
        arts.append(f.arrow((px, y), (px, py + f.fy(2.8)), color=color, lw=LW_HAIR, head=2.6, zorder=5))
    if clip is not None:
        for art in arts:
            art.set_clip_path(clip)
    return arts


def card_footer(f, cell, lines, *, band_pt=20.0):
    """PT_SMALL mute footer lines (token subscripts) centred in the card's bottom band.

    ``lines`` is a list of chains; one line sits on the band's lower line so
    every card's footer ends on the same baseline, two lines fill the band.
    """
    x0, y0, w, _ = cell
    pitch = band_pt / 2.0
    ys = [y0 + f.fy(pitch * 0.5)] if len(lines) == 1 else [y0 + f.fy(pitch * 1.5), y0 + f.fy(pitch * 0.5)]
    for y, parts in zip(ys, lines):
        chain(f, (x0 + w / 2.0, y), parts, size=PT_SMALL, color=MUTE, ha="center")


# ── A: credit enters at the soma ──────────────────────────────────────────
def credit_entry(ax):
    f = Frame(ax)
    X, Y = f.fx, f.fy
    # MNIST tile (top left) and its arrow into the canopy
    tile = ax.inset_axes((X(1), Y(94), X(18), Y(18)), transform=ax.transData)
    tile.imshow(DIGIT, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    tile.set_xticks([]); tile.set_yticks([])
    for spine in tile.spines.values():
        spine.set_color(COLORS["grid"]); spine.set_linewidth(LW_HAIR)
    f.text((X(10), Y(91.5)), "MNIST", size=PT_SMALL, color=MUTE, va="top")
    # three [3,3]-style trees: two ghosted behind, the front tree the hero
    base = (X(22), Y(52), X(46), Y(44))
    for k in (2, 1):
        f.balanced_tree((base[0] + X(7.0 * k), base[1] + Y(9.0 * k), base[2], base[3]),
                        depth=2, mode="plain", ghost=True, labels=False)
    nodes = f.balanced_tree(base, depth=2, mode="inputs", labels=False)
    terms = [nodes[t] for t in nodes.terminals]
    top = max(p[1] for p in terms)
    left = min(p[0] for p in terms)
    f.arrow((X(19.5), Y(100)), (left - X(1.0), top + Y(5.0)), color=MUTE, lw=LW_EDGE, head=3.4)
    f.text((max(p[0] for p in terms) + X(16.0), top + Y(20.0)), "×128", size=PT_SMALL, color=MUTE, ha="left")
    # one contact tagged e_i (left of the leftmost terminal)
    t1 = nodes["T1"]
    f.subscript((t1[0] - X(CONTACT_DIA_PT * 0.5 + 1.6), t1[1]), "e", "i", size=PT_SMALL,
                color=COLORS["exc"], ha="right", va="center")
    # forward path: soma -> readout card -> loss card
    sx, sy = nodes.soma
    cards = ((73.0, "readout", "ŷ"), (104.0, "loss", "L"))
    cw, ch = 25.0, 26.0
    for x_pt, head, body in cards:
        core = f.task_card((X(x_pt), sy - Y(ch / 2.0), X(cw), Y(ch)))
        f.text((core[0] + core[2] / 2.0, core[1] + core[3] - Y(3.5)), head, size=PT_SMALL, color=MUTE, va="top")
        f.text((core[0] + core[2] / 2.0, core[1] + Y(7.5)), body, size=PT_ANNOT, color=INK)
    z0 = sx + X(nodes.soma_r_pt + 1.5)
    f.arrow((z0, sy), (X(cards[0][0] - 1.0), sy), color=MUTE, lw=LW_EDGE, head=3.4)
    f.text(((z0 + X(cards[0][0])) / 2.0, sy + Y(2.6)), "z", size=PT_SMALL, color=INK, va="bottom")
    f.arrow((X(cards[0][0] + cw + 1.0), sy), (X(cards[1][0] - 1.0), sy), color=MUTE, lw=LW_EDGE, head=3.4)
    # the error returns from the loss to every soma: ink hairline loop, then
    # the rule-agnostic δ0 entry arrow (D9); δ_u is named once on the loop
    tail = (sx + X(nodes.soma_r_pt + 11.0), sy - Y(nodes.soma_r_pt + 7.0))
    loop_y = sy - Y(ch / 2.0 + 12.5)
    lx = X(cards[1][0] + cw / 2.0)
    ax.plot([lx, lx, tail[0], tail[0]], [sy - Y(ch / 2.0), loop_y, loop_y, tail[1]],
            color=INK, lw=f.lw(LW_HAIR), solid_capstyle="round", solid_joinstyle="round", zorder=4.8)
    f.error_in(nodes.soma, label="δ0")
    chain(f, (X(cards[1][0] + cw), loop_y + Y(1.6)), [("δ", "u"), " = ∂L/∂", ("y", "u")],
          size=PT_ANNOT, color=INK, ha="right", va="bottom")
    # equation line and its operands
    chain(f, (X(2), Y(23.5)), ["Δ", ("g", "i"), " = −η ", ("e", "i"), " ", ("ε", "n")],
          size=PT_LABEL, color=INK)
    chain(f, (X(2), Y(12.5)), [("e", "i"), " = ", ("x", "i"), " ", ("R", "n"), " (", ("E", "i"),
                               " − ", ("V", "n"), ")  local eligibility"], size=PT_SMALL, color=MUTE)
    chain(f, (X(2), Y(3.5)), [("ε", "n"), "  delivered credit"], size=PT_SMALL, color=MUTE)
    return ax


# ── B: three ways to spread one error ─────────────────────────────────────
def deliveries(ax):
    f = Frame(ax)
    cells = f.split(3, axis="x", gap_pt=10.0)
    foot_pt, eq_pt, bus_pt = 20.0, 12.0, 9.0
    specs = (
        ("Layer scalar", False, [("ε", "n"), " = s"], [["one s per layer"]]),
        ("Per neuron", True, [("ε", "n"), " = ", ("δ", "0")],
         [["one ", ("δ", "u"), " per neuron,"], ["spread evenly"]]),
        ("Exact path", False, [("ε", "n"), " = ", ("α", "3"), ("α", "2"), ("α", "1"), ("δ", "0")],
         [["one ", ("ε", "n"), " per compartment"]]),
    )
    scalar = COLORS["scalar"]
    for i, (cell, (title, hero, equation, footer)) in enumerate(zip(cells, specs)):
        patch_before = list(ax.patches)
        core = f.task_card(cell, title=title, emphasis=hero)
        card = [p for p in ax.patches if p not in patch_before][0]
        card_footer(f, cell, footer, band_pt=foot_pt)
        core = (core[0], core[1] + f.fy(foot_pt), core[2], core[3] - f.fy(foot_pt))
        # identical tree geometry in every card: same rect offsets
        rect = (core[0] + f.fx(8.0), core[1] + f.fy(eq_pt), core[2] - f.fx(16.0),
                core[3] - f.fy(eq_pt + bus_pt))
        ghost = None
        if i == 0:
            # a ghost neighbour behind the hero, entering from the right edge
            ghost_rect = (rect[0] + f.fx(60.0), rect[1], rect[2], rect[3])
            n_patches = len(ax.patches)
            ghost = f.balanced_tree(ghost_rect, depth=3, mode="plain", ghost=True, labels=False)
            for art in [*ghost.edges.values(), *ghost.rings.values(), *ax.patches[n_patches:]]:
                art.set_clip_path(card)
        nodes = f.balanced_tree(rect, depth=3, mode="forward", output="z")
        chain(f, (core[0] + f.fx(5.0), core[1] + f.fy(5.0)), equation, size=PT_ANNOT, color=INK)
        terms = [nodes[t] for t in nodes.terminals]
        top = max(p[1] for p in terms)
        bus_y = top + f.fy(bus_pt)
        x_left = min(p[0] for p in terms) - f.fx(2.0)
        if i == 0:
            x_hi = cell[0] + cell[2] - f.fx(9.0)
            visible = [ghost[t] for t in ghost.terminals if ghost[t][0] < x_hi - f.fx(3.0)]
            amber_bus(f, terms + visible, x_lo=x_left, x_hi=x_hi, y=bus_y, source_at="right", clip=card)
            f.text((x_hi + f.fx(2.6), bus_y), "s", size=PT_SMALL, color=AMBER_TEXT, ha="left", va="center")
        elif i == 1:
            f.error_in(nodes.soma, label="δ0")
            # source dot at the soma (D5), hairline up the left side to the bus
            sx, sy = nodes.soma
            src = (sx - f.fx(nodes.soma_r_pt + 0.4), sy)
            f.disc(src, 1.6, fill=scalar, zorder=6)
            x_feed = x_left - f.fx(3.5)
            ax.plot([src[0], x_feed, x_feed], [sy, sy, bus_y], color=scalar, lw=f.lw(LW_HAIR),
                    solid_capstyle="round", solid_joinstyle="round", zorder=5)
            amber_bus(f, terms, x_lo=x_feed, x_hi=max(p[0] for p in terms) + f.fx(2.0),
                      y=bus_y, source_at=None)
        else:
            f.error_in(nodes.soma, label="δ0")
            f.credit_delivery(nodes, mode="exact", alpha_tags=True)
    return ax


# ── C: profiles at K = 1, 3, 12 ───────────────────────────────────────────
def dictionaries(ax):
    """Actual three-proximal/nine-distal morphology and projection matrices.

    Native row indices follow the model: 0--2 proximal, 3--11 distal. The
    matrices are displayed in subtree order (each proximal site immediately
    before its three distal children), the order the site tree's rows use.
    """
    f = Frame(ax)
    X, Y = f.fx, f.fy
    rows_h = 72.0                       # 12 rows at 6 pt
    y0 = 30.0                           # matrices and tree share this band
    tree_rect = (X(8), Y(y0), X(36), Y(rows_h))
    nodes = f.site_tree(tree_rect, subtree_colors=K3_CYCLE, orient="right")
    f.partition(nodes, [nodes.subtree(k) for k in range(3)], colors=K3_CYCLE)
    f.error_in(nodes.soma, label=None, side="below")
    sx, sy = nodes.soma
    f.subscript((sx - X(2.4), sy - Y(nodes.soma_r_pt + 12.0)), "δ", "0", size=PT_ANNOT,
                color=INK, ha="right", va="center")
    chain(f, (X(2), Y(y0 + rows_h + 7.0)), ["δ̂ = A c"], size=PT_LABEL, color=INK)
    order = np.array(nodes.order)
    assert order.tolist() == [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11]
    a1 = np.ones((12, 1))
    a3 = np.zeros((12, 3))
    for k in range(3):
        a3[[k, 3 + 3 * k, 4 + 3 * k, 5 + 3 * k], k] = 1
    a12 = np.eye(12)
    boxes = ((50.0, 6.0, a1[order], dict(color="scalar"), "K = 1", "right"),
             (66.0, 18.0, a3[order], dict(col_colors=K3_CYCLE), "K = 3", "center"),
             (96.0, 72.0, a12[order][:, order], dict(color="bp"), "K = 12", "center"))
    for x, w, data, kw, header, ha in boxes:
        rect = (X(x), Y(y0), X(w), Y(rows_h))
        f.dictionary_matrix(rect, data, row_groups=[4, 4, 4], label=None, **kw)
        hx = X(x + w + 4.0) if ha == "right" else X(x + w / 2.0)
        f.text((hx, Y(y0 + rows_h + 3.0)), header, size=PT_ANNOT, color=INK, ha=ha, va="bottom")
    # coefficient source per dictionary (one line each; oracle badge on the projected rules)
    chain(f, (X(2), Y(22.0)), ["K = 1: c = ", ("δ", "u"), " (per neuron)"], size=PT_SMALL, color=INK)
    chain(f, (X(168), Y(22.0)), ["K = 12: c = ε (exact)"], size=PT_SMALL, color=INK, ha="right")
    _, x_end, _ = chain(f, (X(2), Y(13.0)), ["K = 1, 3: c = ⟨exact field⟩ (projected)"],
                        size=PT_SMALL, color=INK)
    f.badge((x_end + X(4.0), Y(13.0)), "oracle", ha="left", va="center")
    f.text((X(2), Y(4.0)), "K counts profiles, not external errors", size=PT_SMALL, color=MUTE, ha="left")
    return dict(broadcast=a1, subtrees=a3, resolved=a12, display_row_order=order)


# ── data loaders (assertions retained from the frozen six-rule cohort) ───
def read_fresh():
    conditions = pd.read_csv(FRESH / "condition_summary_six_rules.csv", float_precision="round_trip")
    seeds = pd.read_csv(FRESH / "fresh_analysis_rows_six_rules.csv", float_precision="round_trip")
    paired = pd.read_csv(FRESH / "paired_contrasts_six_rules.csv", float_precision="round_trip")
    conditions = conditions[conditions.metric.eq("test_accuracy") & conditions.rate_policy.eq("selected")].copy()
    seeds = seeds[seeds.rate_policy.eq("selected")].copy()
    paired = paired[paired.metric.eq("test_accuracy") & paired.rate_policy.eq("selected") & paired.contrast.isin(CONTRASTS)].copy()
    assert len(seeds) == 120 and len(conditions) == 12 and len(paired) == 2 * len(CONTRASTS)
    assert seeds.epochs.eq(180).all()
    assert seeds.groupby(["architecture", "seed"]).initialized_model_sha256.nunique().eq(1).all()
    assert seeds[seeds.arm.eq("decoder_only")].decoder_only_core_unchanged.all()
    for architecture in ARCHITECTURES:
        for arm in ARMS:
            s = seeds[seeds.architecture.eq(architecture) & seeds.arm.eq(arm)]
            row = conditions[conditions.architecture.eq(architecture) & conditions.arm.eq(arm)]
            assert len(s) == 10 and s.seed.nunique() == 10, (architecture, arm, len(s))
            assert len(row) == 1 and int(row.iloc[0].n) == 10
            assert abs(s.test_accuracy.mean() - row.iloc[0]["mean"]) < 1e-12
        p = seeds[seeds.architecture.eq(architecture)].pivot(index="seed", columns="arm", values="test_accuracy")
        assert p.shape == (10, 6) and not p.isna().any().any()
        for name in CONTRASTS:
            lhs, rhs = name.split("_minus_")
            row = paired[paired.architecture.eq(architecture) & paired.contrast.eq(name)]
            assert len(row) == 1 and int(row.iloc[0].n) == 10
            assert abs((p[lhs] - p[rhs]).mean() - row.iloc[0]["mean"]) < 1e-12
    return conditions, seeds, paired


def cohort_contrasts(paired):
    """E/F rows: per-neuron − scalar and exact − per-neuron in four cohorts.

    Fresh MNIST rows come from the six-rule cohort; the DFA, Fashion-MNIST
    and CIFAR-10 rows replay frozen paired contrasts and re-check each seed
    count against the cohort's ``seed_outcomes.csv``.
    """
    rows = []
    fresh_keys = {"identity": "neuron_shared_minus_strict_scalar", "exact": "exact_path_minus_neuron_shared"}
    for architecture in ARCHITECTURES:
        for kind, key in fresh_keys.items():
            r = paired[paired.architecture.eq(architecture) & paired.contrast.eq(key)].iloc[0]
            rows.append(dict(cohort="mnist_fresh", task="MNIST", protocol="fresh", architecture=architecture, kind=kind,
                             mean_pp=100 * r["mean"], low_pp=100 * r.ci_low, high_pp=100 * r.ci_high, n_seeds=int(r.n),
                             positive_seeds=int(r.positive),
                             source_table="source_data/image_ladder_controls/summaries/paired_contrasts_six_rules.csv",
                             source_contrast=key))
    factorial = pd.read_csv(SOURCE / "mnist_between_within_factorial/paired_contrasts.csv", float_precision="round_trip")
    fo = pd.read_csv(SOURCE / "mnist_between_within_factorial/seed_outcomes.csv")
    for architecture in ARCHITECTURES:
        for kind, key in (("identity", "dfa within: neuron - scalar"), ("exact", "dfa within: exact path - neuron")):
            r = factorial[factorial.architecture.eq(architecture) & factorial.contrast.eq(key)].iloc[0]
            n_seed = fo[fo.architecture.eq(architecture) & fo.between.eq("dfa") & fo.within.eq("neuron specific")].seed.nunique()
            assert int(r.n_seeds) == n_seed == 15, (architecture, key, n_seed)
            rows.append(dict(cohort="mnist_dfa", task="MNIST", protocol="DFA", architecture=architecture, kind=kind,
                             mean_pp=100 * r.mean_difference, low_pp=100 * r.ci95_low, high_pp=100 * r.ci95_high,
                             n_seeds=int(r.n_seeds), positive_seeds=int(round(r.positive_seed_fraction * r.n_seeds)),
                             source_table="source_data/mnist_between_within_factorial/paired_contrasts.csv", source_contrast=key))
    fashion = pd.read_csv(SOURCE / "fashion_feedback_ladder/paired_contrasts.csv", float_precision="round_trip")
    fs = pd.read_csv(SOURCE / "fashion_feedback_ladder/seed_outcomes.csv")
    for architecture in ARCHITECTURES:
        for kind, key in (("identity", "neuron indexed - scalar fallback"), ("exact", "exact path - neuron indexed")):
            r = fashion[fashion.architecture.eq(architecture) & fashion.contrast.eq(key)].iloc[0]
            n_seed = fs[fs.architecture.eq(architecture) & fs.feedback.eq("neuron indexed")].seed.nunique()
            assert int(r.n_seeds) == n_seed == 10, (architecture, key, n_seed)
            rows.append(dict(cohort="fashion", task="Fashion-MNIST", protocol="matched-width scalar fallback",
                             architecture=architecture, kind=kind, mean_pp=100 * r.mean_difference,
                             low_pp=100 * r.ci95_low, high_pp=100 * r.ci95_high, n_seeds=int(r.n_seeds),
                             positive_seeds=int(r.positive_seeds),
                             source_table="source_data/fashion_feedback_ladder/paired_contrasts.csv", source_contrast=key))
    cifar = pd.read_csv(SOURCE / "cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv", float_precision="round_trip")
    cs = pd.read_csv(SOURCE / "cifar10_additive_feedback_ladder_confirmatory/seed_outcomes.csv")
    for kind, key in (("identity", "neuron specific minus strict scalar"), ("exact", "exact path minus neuron specific")):
        r = cifar[cifar.contrast.eq(key)].iloc[0]
        diffs = np.array([float(v) for v in str(r.seed_differences).split(";")])
        n_seed = cs[cs.feedback.eq("neuron specific")].seed.nunique()
        assert len(diffs) == int(r.n_seeds) == n_seed == 20 and abs(diffs.mean() - r.mean_difference) < 1e-8
        rows.append(dict(cohort="cifar10", task="CIFAR-10", protocol="additive [3,3,3,3]", architecture="additive", kind=kind,
                         mean_pp=100 * r.mean_difference, low_pp=100 * r.ci95_low_difference, high_pp=100 * r.ci95_high_difference,
                         n_seeds=int(r.n_seeds), positive_seeds=int(r.seeds_positive),
                         source_table="source_data/cifar10_additive_feedback_ladder_confirmatory/paired_contrasts.csv",
                         source_contrast=key))
    return pd.DataFrame(rows)


def read_capture(coordinate=CAPTURE_COORDINATE):
    per_seed = pd.read_csv(FRESH / "delivery_coordinate_capture.csv", float_precision="round_trip")
    summary = pd.read_csv(FRESH / "delivery_coordinate_capture_summary.csv", float_precision="round_trip")
    bases = [b for b, _ in CAPTURE_BASES]
    per_seed = per_seed[per_seed.cohort.eq("fresh") & per_seed.checkpoint.eq("trained")
                        & per_seed.coordinate.eq(coordinate) & per_seed.basis.isin(bases)].copy()
    summary = summary[summary.checkpoint.eq("trained") & summary.coordinate.eq(coordinate)
                      & summary.basis.isin(bases) & summary.metric.eq("mean_capture")].copy()
    assert len(per_seed) == 60 and len(summary) == 6
    assert per_seed.forward_max_difference.abs().max() < 1e-4
    for architecture in ARCHITECTURES:
        for basis in bases:
            s = per_seed[per_seed.architecture.eq(architecture) & per_seed.basis.eq(basis)]
            row = summary[summary.architecture.eq(architecture) & summary.basis.eq(basis)].iloc[0]
            assert len(s) == 10 and s.seed.nunique() == 10 and int(row.n) == 10
            assert abs(s.mean_capture.mean() - row["mean"]) < 1e-9
    return per_seed, summary


def signed(value, *, decimals=1):
    return f"{value:+.{decimals}f}".replace("-", "−")


# ── D: six rules on MNIST ─────────────────────────────────────────────────
def accuracy(ax, conditions, seeds, paired):
    positions = np.array([0, 1, 2, 3, 4, 5.6])
    ax.axvspan(4.95, 6.15, color=COLORS["panel_bg"], zorder=0, lw=0)
    for architecture, offset, marker in [("shunting", -.09, "o"), ("additive", .09, "s")]:
        color = COLORS[architecture]
        group = seeds[seeds.architecture.eq(architecture)]
        pivot = group.pivot(index="seed", columns="arm", values="test_accuracy").loc[:, list(ARMS)]
        for _, seed in pivot.iterrows():
            ax.plot(positions[:5] + offset, 100 * seed.iloc[:5], color=color, alpha=.16, lw=LW_HAIR, zorder=1)
        for x, arm in zip(positions, ARMS):
            values = 100 * group[group.arm.eq(arm)].sort_values("seed").test_accuracy.to_numpy()
            ax.plot(x + offset + np.linspace(-.038, .038, 10), values, ls="none", marker="o", ms=SEED_MS,
                    mfc=color, mec="none", alpha=SEED_ALPHA, zorder=2)
            row = conditions[conditions.architecture.eq(architecture) & conditions.arm.eq(arm)].iloc[0]
            ax.errorbar(x + offset, 100 * row["mean"], yerr=[[100 * (row["mean"] - row.ci_low)], [100 * (row.ci_high - row["mean"])]],
                        fmt=marker, color=color, ms=MARKER_MS, mfc="white", mew=LW_ERR,
                        elinewidth=LW_ERR, capsize=2, zorder=4)
    # the identity gain (per neuron − layer scalar, shunting / additive), derived
    # from the frozen paired contrast and set in clear whitespace with a leader
    gains = [100 * paired[paired.architecture.eq(a) & paired.contrast.eq("neuron_shared_minus_strict_scalar")].iloc[0]["mean"]
             for a in ARCHITECTURES]
    tag = " / ".join(signed(v) for v in gains) + " pp"
    ax.text(0.5, 98.75, tag, ha="center", va="center", fontsize=PT_ANNOT, color=INK, zorder=6)
    ax.plot([0.5, 0.5], [98.0, 94.6], color=MUTE, lw=LW_HAIR, zorder=3)
    ax.set(xlim=(-.5, 6.15), ylim=(78, 99.5), xticks=positions, xticklabels=ARM_LABELS,
           yticks=[80, 85, 90, 95], ylabel="Test accuracy (%)")
    style_panel(ax, grid="y")
    ax.tick_params(axis="x", labelsize=PT_TICK, pad=2.5)
    ax.yaxis.labelpad = 2.0
    handles = [Line2D([], [], color=COLORS[a], marker=m, mfc="white", lw=LW_DATA, ms=MARKER_MS, label=l)
               for a, m, l in [("shunting", "o", "Shunting"), ("additive", "s", "Additive")]]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(.01, .0), frameon=False, fontsize=PT_LEGEND,
              ncol=2, columnspacing=1.2, handlelength=1.5, handletextpad=.5)


# ── E / F: the four-cohort forest ─────────────────────────────────────────
COHORT_ROWS = (("mnist_fresh", "MNIST", "n = 10, fresh"),
               ("mnist_dfa", "MNIST", "n = 15, DFA"),
               ("fashion", "Fashion-MNIST", "n = 10, † fallback"),
               ("cifar10", "CIFAR-10", "n = 20, additive [3,3,3,3]"))
FOREST_YLIM = (-0.55, 4.25)     # note band above the MNIST row in F


def cohort_forest(ax, rows, kind, *, xlabel, xlim, xticks, labels, note=None):
    ax.plot([0, 0], [FOREST_YLIM[0], 3.36], color=MUTE, lw=LW_REF, ls=DASH, zorder=0)
    ys = {cohort: 3 - i for i, (cohort, _, _) in enumerate(COHORT_ROWS)}
    for architecture, offset, marker in [("shunting", .11, "o"), ("additive", -.11, "s")]:
        color = COLORS[architecture]
        sub = rows[rows.architecture.eq(architecture) & rows.kind.eq(kind)]
        for r in sub.itertuples():
            single = len(rows[rows.cohort.eq(r.cohort) & rows.kind.eq(kind)]) == 1
            y = ys[r.cohort] + (0.0 if single else offset)
            ax.errorbar(r.mean_pp, y, xerr=[[r.mean_pp - r.low_pp], [r.high_pp - r.mean_pp]], fmt=marker,
                        color=color, ms=MARKER_MS, mfc="white", mew=LW_ERR, elinewidth=LW_ERR, capsize=2, zorder=3)
    if labels:
        for cohort, name, tag in COHORT_ROWS:
            y = ys[cohort] + .40
            ax.text(xlim[0] + .052 * (xlim[1] - xlim[0]), y, name, ha="left", va="bottom", fontsize=PT_SMALL, color=INK)
            ax.text(xlim[1] - .02 * (xlim[1] - xlim[0]), y, tag, ha="right", va="bottom", fontsize=PT_SMALL, color=MUTE)
    if note:
        for k, line in enumerate(note):
            ax.text(xlim[0] + .03 * (xlim[1] - xlim[0]), 4.0 - 0.38 * k, line, ha="left", va="center",
                    fontsize=PT_SMALL, color=INK)
    ax.set(xlim=xlim, ylim=FOREST_YLIM, xticks=xticks, yticks=[], xlabel=xlabel)
    ax.set_xticklabels([str(t).replace("-", "−") for t in xticks])
    style_panel(ax, grid="x", spines=("bottom",))


def within_tree_note(paired):
    """The two within-tree MNIST contrasts (shunting / additive) quoted in F."""
    out = []
    for label, key in (("K = 3 − K = 1", "subtree_k3_minus_projected_k1"), ("exact − K = 3", "exact_path_minus_subtree_k3")):
        vals = [100 * paired[paired.architecture.eq(a) & paired.contrast.eq(key)].iloc[0]["mean"] for a in ARCHITECTURES]
        out.append(f"{label}: " + " / ".join(signed(v, decimals=2) for v in vals))
    return out


# ── G: dictionary capture of the trained fields ───────────────────────────
def capture(ax, per_seed, summary, coordinate=CAPTURE_COORDINATE):
    xs = np.arange(len(CAPTURE_BASES))
    for architecture, offset, marker, (lx, ly, va) in [("shunting", -.12, "o", (.62, 49.0, "top")),
                                                       ("additive", .12, "s", (.38, 68.5, "bottom"))]:
        color = COLORS[architecture]
        means = []
        for x, (basis, _) in zip(xs, CAPTURE_BASES):
            s = per_seed[per_seed.architecture.eq(architecture) & per_seed.basis.eq(basis)].sort_values("seed")
            ax.plot(x + offset + np.linspace(-.04, .04, len(s)), 100 * s.mean_capture.to_numpy(), ls="none",
                    marker="o", ms=SEED_MS, mfc=color, mec="none", alpha=SEED_ALPHA, zorder=2)
            row = summary[summary.architecture.eq(architecture) & summary.basis.eq(basis)].iloc[0]
            means.append(100 * row["mean"])
            ax.errorbar(x + offset, 100 * row["mean"], yerr=[[100 * (row["mean"] - row.ci_low)], [100 * (row.ci_high - row["mean"])]],
                        fmt=marker, color=color, ms=MARKER_MS, mfc="white", mew=LW_ERR, elinewidth=LW_ERR, capsize=2, zorder=4)
        ax.plot(xs + offset, means, color=color, lw=LW_DATA, zorder=3)
        ax.text(lx, ly, architecture.capitalize(), ha="center", va=va, fontsize=PT_LEGEND, color=label_color(color))
    ax.set(xlim=(-.45, 2.45), ylim=(0, 100), xticks=xs, xticklabels=[lab for _, lab in CAPTURE_BASES],
           yticks=[0, 25, 50, 75, 100], ylabel="Capture (%)")
    style_panel(ax, grid="y")
    ax.yaxis.labelpad = 0.8
    ax.tick_params(axis="y", pad=1.0, length=2.0)   # keeps G's letter clear of F's axes box
    ax.text(-.38, 11.5, "trained exact-path checkpoints,", ha="left", va="bottom", fontsize=PT_SMALL, color=MUTE)
    ax.text(-.38, 3, f"10 seeds, {coordinate} coordinate", ha="left", va="bottom", fontsize=PT_SMALL, color=MUTE)


def main():
    conditions, seeds, paired = read_fresh()
    cohorts = cohort_contrasts(paired)
    cap_seed, cap_summary = read_capture()
    RECORDS.mkdir(exist_ok=True)
    canvas = NativeCanvas(490 / 72, 3, row_weights=[126, 114, 108], hgutter_pt=40, vgutter_pt=40,
                          margins=Margins(left=36, right=12, top=24, bottom=38))
    a = canvas.panel("A", 0, 0, 4, schematic=True, lock=False, title="Credit enters at the soma")
    b = canvas.panel("B", 0, 4, 8, schematic=True, lock=False, title="Three ways to spread one error")
    c = canvas.panel("C", 1, 0, 5, schematic=True, lock=False, title="Profiles at K = 1, 3, 12")
    d = canvas.panel("D", 1, 5, 7, title="Per-neuron credit carries the MNIST gain")
    e = canvas.panel("E", 2, 0, 4, title="Identity gain is large")
    f_ = canvas.panel("F", 2, 4, 4, title="Exact ≤ 0.2 pp; CIFAR −0.9")
    g = canvas.panel("G", 2, 8, 4, title="Voltage capture rises with K")
    credit_entry(a)
    deliveries(b)
    matrices = dictionaries(c)
    accuracy(d, conditions, seeds, paired)
    cohort_forest(e, cohorts, "identity", xlabel="Per neuron − scalar (pp)", xlim=(-.6, 18.4),
                  xticks=[0, 5, 10, 15], labels=True)
    cohort_forest(f_, cohorts, "exact", xlabel="Exact − per neuron (pp)", xlim=(-1.3, .45),
                  xticks=[-1, -0.5, 0], labels=False, note=within_tree_note(paired))
    capture(g, cap_seed, cap_summary)
    canvas.declare_reserve(e, top=11.0)     # row 2 letters clear D's two-line ticks
    canvas.lock_reserves()
    findings = list(canvas.align_letters())
    problems = findings + list(canvas.save(OUT, name="credit_first_figure_01", dpi=180))
    np.savez_compressed(RECORDS / "figure_01_illustrative_dictionaries.npz", **matrices)
    cohorts.to_csv(RECORDS / "figure_01_contrasts.csv", index=False)
    plotted = []
    for panel, table, rows in [("D", "condition_summary_six_rules.csv", conditions),
                               ("D", "paired_contrasts_six_rules.csv", paired)]:
        plotted.extend(dict(panel=panel, source_table=str((FRESH / table).relative_to(JOURNAL)), **r) for r in rows.to_dict("records"))
    plotted.extend(dict(panel="E" if r["kind"] == "identity" else "F", **r) for r in cohorts.to_dict("records"))
    pd.DataFrame(plotted).to_csv(RECORDS / "figure_01_six_arm_source.csv", index=False)
    seeds.to_csv(RECORDS / "figure_01_six_arm_seed_source.csv", index=False)
    cap_seed.assign(panel="G").to_csv(RECORDS / "figure_01_capture_source.csv", index=False)
    files = [Path(__file__), JOURNAL / "scripts/figure_canvas.py", JOURNAL / "scripts/journal_style.py",
             JOURNAL / "scripts/native_schematics.py", JOURNAL / "scripts/credit_tree_schematics.py",
             *[FRESH / name for name in ["condition_summary_six_rules.csv", "fresh_analysis_rows_six_rules.csv",
                                         "paired_contrasts_six_rules.csv", "delivery_coordinate_capture.csv",
                                         "delivery_coordinate_capture_summary.csv"]],
             *[SOURCE / f"{study}/{name}" for study in ("mnist_between_within_factorial", "fashion_feedback_ladder",
                                                         "cifar10_additive_feedback_ladder_confirmatory")
               for name in ("paired_contrasts.csv", "seed_outcomes.csv")],
             *[SOURCE / "image_ladder_controls" / name for name in ["protocol.json", "selection.json", "projected_k1/protocol.json", "projected_k1/selection.json"]]]
    payload = dict(panel_sources={
        "A": "Schematic: MNIST inputs at excitatory contacts of 128 trees, readout and loss, neuronal error delta_u returning as the somatic error delta_0 to every soma; eligibility x delivered credit (main.tex Eqs. factorization, dendriticlocalrule). No measured data.",
        "B": "Schematic: three deliveries on one generic depth-3 tree (layer scalar bus spanning a ghost neighbour, per-neuron bus sourced at the soma, exact path chain with alpha tags; Eq. pathgain). The MNIST model has two path factors. No measured data.",
        "C": "Actual 12 nonsomatic sites: three proximal sites and nine distal children, soma outside the basis. K1 (ones), K3 (subtree indicators, rows in subtree order), K12 (identity). K1/K3 projected rules take oracle coefficients from the exact field (badge); the per-neuron rule uses the same all-ones profile with delta_u.",
        "D": "Complete six-arm fresh MNIST selected-rate cohort: 10 paired seeds per architecture; 180 epochs; validation-selected checkpoint. The printed tag is the paired per-neuron minus layer-scalar contrast (shunting / additive). Decoder-only core remains fixed.",
        "E": "Per-neuron minus scalar (pp) in four separately trained cohorts: fresh MNIST (10 seeds), legacy DFA MNIST (15 seeds), Fashion-MNIST with the matched-width scalar fallback (10 seeds), flattened CIFAR-10 additive [3,3,3,3] (20 seeds). Cohorts are not one ladder.",
        "F": "Exact path minus per-neuron (pp) in the same four cohorts and rows as E; the printed note gives the fresh-cohort within-tree contrasts K3 minus projected K1 and exact minus K3 (shunting / additive).",
        "G": f"Mean {CAPTURE_COORDINATE}-coordinate capture of D's trained exact-path fields by the K1/K3/K12 dictionaries of C; fresh cohort, trained checkpoints, 10 seeds per architecture."},
        source_sha256={str(p.relative_to(JOURNAL)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        numerical_scope="No fitting, selection, bootstrap, or source-outcome mutation. Existing 95% bootstrap intervals replayed exactly; seed means, paired contrast means, cohort seed counts and capture means independently recomputed as assertions.",
        plot_units={"D": "Test accuracy (%); source fractions multiplied by 100; tag in percentage points",
                    "E": "Paired per-neuron minus scalar test-accuracy differences (percentage points)",
                    "F": "Paired exact-path minus per-neuron test-accuracy differences (percentage points); note in percentage points",
                    "G": "Mean per-field capture (%); source fractions multiplied by 100"},
        capture_coordinate=CAPTURE_COORDINATE,
        cohort_seed_counts={"mnist_fresh": 10, "mnist_dfa": 15, "fashion": 10, "cifar10": 20},
        fresh_seed_count_per_architecture=10, normal_font_minimum_pt=PT_SMALL, layout_findings=problems)
    (RECORDS / "figure_01_sources.json").write_text(json.dumps(payload, indent=2) + "\n")
    caption = r"""\textbf{Task-derived credit separates neuronal identity from resolution within a dendritic tree.}
\textbf{A}, MNIST inputs reach excitatory contacts on 128 dendritic trees (two ghosted); readout $\hat y$ and loss $\mathcal L$ return each neuron's error $\delta_u=\partial\mathcal L/\partial y_u$, entering its soma as the somatic error $\delta_0$ (ink arrow). Updates multiply local eligibility $e_i=x_iR_n(E_i-V_n)$ by delivered credit $\varepsilon_n$.
\textbf{B}, Three deliveries on a generic depth-3 tree (three path factors; the MNIST model has two): a layer scalar $s$ shared with a ghost neighbour (amber bus), one error per neuron spread evenly (bus sourced at the soma), and exact path transport $\varepsilon_n=\alpha_3\alpha_2\alpha_1\delta_0$ (red chain).
\textbf{C}, Dictionaries over the twelve nonsomatic sites, rows in subtree order (capsules and $K=3$ columns share tints). The per-neuron rule uses the all-ones profile with $\delta_u$; the projected $K=1$ and $K=3$ rules take oracle coefficients from the exact field (badge).
\textbf{D}, MNIST test accuracy, six rules at selected rates, ten fresh paired seeds per architecture (green circles shunting, blue squares additive; dots seeds, hairlines pair seeds, open symbols means with 95\% bootstrap intervals; tag, per-neuron minus layer-scalar gain, shunting / additive; band, decoder-only reference).
\textbf{E}, Per-neuron minus scalar (pp) in four cohorts, each row naming its seed count and protocol: fresh MNIST (10 seeds), DFA (15-seed legacy cohort), Fashion-MNIST (\dag, matched-width scalar fallback keeping identity at the somatic stage, 640 of 2,413,312 parameters; 10 seeds), flattened CIFAR-10 (additive $[3,3,3,3]$ tree, 20 seeds); cohorts differ in feedback source, scalar construction and tree.
\textbf{F}, Exact minus per-neuron (pp) in the rows of \textbf{E}; note, fresh-cohort within-tree contrasts $K=3$ minus projected $K=1$ (+0.05 / +0.04) and exact minus $K=3$ ($-0.09$ / $-0.04$).
\textbf{G}, Mean voltage-error capture of \textbf{D}'s trained exact-path fields by the dictionaries of \textbf{C} (seeds, means, 95\% intervals; in activation space, the trained coordinate, 0.622$\to$0.655 and 0.764$\to$0.859, Supplementary Fig.~S49F).
Marks in \textbf{E}--\textbf{G} follow \textbf{D}; intervals are paired 95\% bootstrap intervals; pp, percentage points; rates selected on three development seeds (Supplementary Fig.~S49).
"""
    (RECORDS / "figure_01_caption.tex").write_text(caption)
    (RECORDS / "figure_01_caption.md").write_text(caption)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

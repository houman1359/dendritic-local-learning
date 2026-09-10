#!/usr/bin/env python3
"""Main Fig. 1 (`fig:framework`): neuronal identity, not finer spatial address.

Built to `analysis/figure_overhaul_20260908/v2/fig1/PLAN.md` as amended by
`v2/AMENDMENTS.md` (Figure 1 items 1-8, blocking items B9/B10/B12/B14) and
ruled by `v2/DECISIONS.md` (G1-G7, Figure 1).  Precedence: DECISIONS >
AMENDMENTS > PLAN.

Cross-figure rules in force (AMENDMENTS section 3), copied verbatim:

* CF-1 Canvas.  518.4 pt wide; height on the ladder 340 / 415 / 490 only;
  aspect >= 1.05.  This figure: 518.4 x 490.0 pt, aspect 1.058.
* CF-2 Type.  Exactly three sizes 7.0 / 8.0 / 9.0-bold; nothing below 7.0; no
  DejaVu; subscripts via Frame.subscript / token_subscript, never mathtext.
* CF-3 Strokes.  Only 0.55 / 0.70 / 0.85 / 0.95 / 1.25 pt; every area mark is
  a 16 % tint_patch with a 0.55 pt edge; no open stroke >= 1.35 pt.
* CF-4 Glyphs.  Soma = filled soma disc, lowest node of its tree, ink rim.
  Tapered dend tree fanning up, open white junction rings.  exc filled blue /
  inh filled carmine contacts, 3.6 pt.  Ghosts at GHOST_PCT 45.  Exactly four
  delivery modes; A, B and C use scalar, neuron and exact.  One ink delta-0
  arrow into every soma and no DELTA0_EXEMPTIONS entry for this figure.
* CF-5 Legends.  Zero legend boxes inside data axes, zero figure-level
  legends, direct labels only; the set's single sanctioned key is Fig 5C.
* CF-6 Forest idiom.  E and F go through figure_canvas.forest(); centred row
  labels in the declared gutter, 0.55 pt row tick / 6 % band, seed fan always
  drawn, per-row n, one right-aligned footer tag; the second arm is
  overplotted at +0.22 rows with an open marker.
* CF-7 References.  Every reference line dashed mute at LW_REF with the label
  right-aligned on the line; zero drawn once, never grid + rule.
* CF-8 Captions, CF-9 Titles: see analysis/.../v2/fig1/TEXT.md.
* CF-10 Schematic area, one common formula
  (sum of schematic panel SLOT areas) / (live_w_pt x live_h_pt)
  = 126 x (130.13 + 300.27 + 130.13) / (470.4 x 430) = 34.9 % -- waiver W1.
* CF-11 Matrices.  check_matrix_cells >= 6.0 pt per row and per column
  (C: 6.5 pt rows, 7.0 pt columns); no col_labels; no raster below 300 dpi
  (panel A's stimulus tile is drawn, not rastered -- DECISIONS Fig 1).
* CF-12 Letters.  9 pt bold, canvas.align_letters() unconditional inside
  save(), no hand-placed letters, no hand alignment loop.

Waivers recorded here and in the canvas manifest:

* W1 schematic area 34.9 % > 30 % (DECISIONS G4, restated on the B12 formula).
* W2 row heights 126 / 126 / 98 pt, not the review's 124 / 116 / 108: an
  8-module panel on a 116 pt row is aspect 2.59, above PANEL_ASPECT_MAX 2.40.
* W3 [build-time] the forest label gutter is a 38 pt declared reserve on E, F
  and G.  A declared reserve is locked per module COLUMN, so A and C draw in
  92.1 pt and B and D in 262.3 pt rather than the plan's 130.1 / 300.3.
  Without the shared lock the panel-emphasis audit fails (slot-fill spread
  1.56x, allowed 1.35x), and audit_letter_alignment.py forbids any column-0
  ink left of the letter column, so the row labels cannot hang into the
  margin instead.

Private helpers (SPEC_ERRATA #7 / DECISIONS G5; nothing is added to
journal_style.py, figure_canvas.py or native_schematics.py):

* chain()        token-subscript runs laid out left to right (equations).
* hat()          the circumflex of delta-hat; the face has no combining
                 circumflex (y-hat itself is a real glyph, U+0177).
* gain_dial()    the Gamma gain ring on the delta-0 shaft
                 (proposed: native_schematics.Frame.gain_dial).
* overlay_arm()  the additive arm at +0.22 rows with an open marker
                 (proposed: figure_canvas.forest(row_offset=, jitter_rows=)).
* row_note()     a per-row tag placed on the emptier side of its own row.
* ghost_clip()   clips a ghost neighbour to its card and drops the glyphs the
                 cut would halve.
* _fade_ghost()  re-tints a ghost arbor's strokes and junction rings to
                 mix('dend', GHOST_PCT).  LIBRARY DEFECT, reported, not
                 patched here: native_schematics.GHOST is mix('mute', 45) =
                 RGB(0.645, 0.654, 0.665), which is DARKER than COLORS['dend']
                 #A9ABB1, so balanced_tree(ghost=True) draws a neighbour at
                 the hero's own tone and CF-4's GHOST_PCT 45 fade never
                 reaches an arbor -- only the soma fill is faded.  GHOST
                 should be mix('dend', GHOST_PCT).
* write_curated() / register_curated()
                 the Source Data display table the caption names,
                 source_data/curated_publication/figure_01_plotted.csv, one
                 record per drawn mark (panels C--G; A and B are schematics
                 and contribute no rows), and its manifest.json record.

This builder performs no fitting, rate selection or interval estimation:
every number is replayed from frozen Source Data and re-derived as an
assertion.  Frozen protocol and summary files under source_data/ are never
modified; the render-time copies this builder writes live in
source_data/credit_first_figures/ and are re-hashed in figure_01_sources.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib.patches import Ellipse, Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
JOURNAL = HERE.parents[1]
sys.path.insert(0, str(JOURNAL / "scripts"))
from figure_canvas import (COLORS, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR, LW_REF,
                           MARKER_MS, PT_BASE, PT_EMPH, SEED_ALPHA, SEED_MS,
                           Margins, NativeCanvas, style_panel, tint_patch)
from journal_style import K_CYCLE, label_color
from credit_tree_schematics import AMBER_TEXT, mix
from native_schematics import (CONTACT_DIA_PT, GHOST_PCT, Frame,
                               _text_w_pt)

SOURCE = JOURNAL / "source_data"
FRESH = SOURCE / "image_ladder_controls/summaries"
OUT = JOURNAL / "figures/components/credit_first_figure_01.pdf"
MAIN = JOURNAL / "figures/main/figure_01.pdf"
RECORDS = SOURCE / "credit_first_figures"
CURATED = SOURCE / "curated_publication/figure_01_plotted.csv"

# ── vocabulary lock (AMENDMENTS section 4a; DECISIONS G3) ─────────────────
ARMS = ("strict_scalar", "neuron_shared", "projected_k1", "subtree_k3",
        "exact_path", "decoder_only")
ARM_LABELS = ("Strict\nscalar", "Per\nneuron", "Projected\nK = 1",
              "Projected\nK = 3", "Exact\npath", "Decoder\nonly")
ARCHITECTURES = ("shunting", "additive")
CONTRASTS = ("neuron_shared_minus_strict_scalar", "subtree_k3_minus_projected_k1",
             "exact_path_minus_subtree_k3", "exact_path_minus_neuron_shared")
CAPTURE_COORDINATE = "activation"      # D-1: the coordinate L134 publishes
CAPTURE_BASES = (("broadcast_k1", "1"), ("subtrees_k3", "3"),
                 ("exact_k12", "12"))
COHORT_ROWS = (("mnist_fresh", "MNIST\nfresh"), ("mnist_dfa", "MNIST\nDFA"),
               ("fashion", "Fashion-\nMNIST"), ("cifar10", "CIFAR-10"))
ARM_OFFSET_ROWS = 0.22                 # CF-6: the one second-arm offset
FOREST_GUTTER_PT = 38.0                # W3
INK, MUTE = COLORS["ink"], COLORS["mute"]
DASH = (0, (2.2, 1.8))
SCHEMATIC_FRACTION = 126.0 * (130.13 + 300.27 + 130.13) / (470.4 * 430.0)


# ── private glyph helpers ─────────────────────────────────────────────────
def chain(f, xy, parts, *, size=PT_BASE, color=None, ha="left", va="center",
          zorder=6):
    """Lay out plain strings and (base, sub) pairs left to right at ``xy``.

    Every pair goes through ``Frame.subscript`` (token subscripts, never
    mathtext).  Returns ``(x_start, x_end, [x_start of each piece])`` in frame
    fractions so a caller can hang an accent or a badge off one piece.
    """
    color = INK if color is None else color
    widths = []
    for p in parts:
        if isinstance(p, tuple):
            widths.append(_text_w_pt(f.ax, p[0], size) + 0.4
                          + _text_w_pt(f.ax, p[1], PT_BASE))
        else:
            widths.append(_text_w_pt(f.ax, p, size))
    total = sum(widths)
    x, y = xy
    if ha == "center":
        x -= f.fx(total / 2.0)
    elif ha == "right":
        x -= f.fx(total)
    starts, x_start = [], x
    for p, w in zip(parts, widths):
        starts.append(x)
        if isinstance(p, tuple):
            f.subscript((x, y), p[0], p[1], size=size, color=color, ha="left",
                        va=va, zorder=zorder)
        else:
            f.text((x, y), p, size=size, color=color, ha="left", va=va,
                   zorder=zorder)
        x += f.fx(w)
    return x_start, x, starts


def hat(f, x, y, w_pt, *, size=PT_BASE, color=None):
    """The circumflex of delta-hat, as a LW_HAIR chevron over a ``w_pt`` glyph."""
    color = INK if color is None else color
    cx = x + f.fx(w_pt / 2.0)
    half, rise = 0.28 * size, 0.20 * size
    y0 = y + f.fy(0.42 * size)
    f.ax.plot([cx - f.fx(half), cx, cx + f.fx(half)],
              [y0, y0 + f.fy(rise), y0], color=color, lw=f.lw(LW_HAIR),
              solid_capstyle="round", solid_joinstyle="round", zorder=6)


def gain_dial(f, xy, *, r_pt=3.2, label="Γ"):
    """The diagonal gain as a scaffolding dial on the delta-0 shaft.

    An open mute ring with one radial tick: it is drawn in the scaffolding
    colour, so it cannot be read as a credit rule, and it gives L110's Gamma
    a picture.  Proposed for the library as ``Frame.gain_dial``.
    """
    f.disc(xy, r_pt, fill="white", edge=MUTE, lw=LW_HAIR, zorder=5.5)
    ang = np.deg2rad(40.0)
    f.ax.plot([xy[0], xy[0] + f.fx(r_pt * 0.95 * np.cos(ang))],
              [xy[1], xy[1] + f.fy(r_pt * 0.95 * np.sin(ang))], color=MUTE,
              lw=f.lw(LW_HAIR), solid_capstyle="round", zorder=5.6)
    f.text((xy[0], xy[1] - f.fy(r_pt + 3.4)), label, size=PT_BASE, color=MUTE,
           va="top")


def _fade_ghost(nodes):
    """Re-tint a ghost arbor's strokes and junction rings to the fade tone.

    ``native_schematics.GHOST`` (imported from the legacy
    ``credit_tree_schematics``) is ``mix('mute', 45)`` = RGB(0.645, 0.654,
    0.665) -- slightly DARKER than ``COLORS['dend']`` #A9ABB1 -- so
    ``balanced_tree(ghost=True)`` draws a neighbour at the hero's own tone and
    only the soma fill is faded.  CF-4's GHOST_PCT 45 convention is therefore
    not in force for arbors anywhere in the library.  Reported as a library
    defect (GHOST should be ``mix('dend', GHOST_PCT)``); until it is fixed,
    every ghost arbor this builder draws is repainted here.  The ghost soma is
    left as the library draws it.  This is a tint, not a stroke weight, so
    there is no CF-3 exposure.
    """
    tone = mix("dend", GHOST_PCT)
    for line in nodes.edges.values():
        if line is not None:
            line.set_color(tone)
    for ring in nodes.rings.values():
        if ring is not None:
            ring.set_edgecolor(tone)
    return nodes


def ghost_clip(f, cell, artists, *, pad_pt=2.0):
    """Clip a ghost neighbour to its card, dropping glyphs the cut halves."""
    x0, y0, w, h = cell
    clip = Rectangle((x0 + f.fx(pad_pt), y0 + f.fy(pad_pt)),
                     w - 2 * f.fx(pad_pt), h - 2 * f.fy(pad_pt),
                     transform=f.ax.transData, facecolor="none",
                     edgecolor="none", zorder=0)
    f.ax.add_patch(clip)
    cut = x0 + w - f.fx(pad_pt)
    for art in artists:
        if isinstance(art, Ellipse) and art.center[0] > cut - f.fx(2.6):
            art.remove()
        else:
            art.set_clip_path(clip)
    return cut


def new_artists(ax, before):
    """Artists added to ``ax`` since ``before`` = snapshot(ax)."""
    return [*ax.patches[before[0]:], *ax.lines[before[1]:],
            *ax.texts[before[2]:]]


def snapshot(ax):
    return (len(ax.patches), len(ax.lines), len(ax.texts))


def signed(value, decimals=2):
    return f"{value:+.{decimals}f}".replace("-", "−")


def minus(text):
    return str(text).replace("-", "−")


def close(value, target, tol=5e-3):
    assert abs(float(value) - float(target)) <= tol, (value, target)
    return float(value)


# ── data ──────────────────────────────────────────────────────────────────
def read_fresh():
    """The frozen six-rule fresh MNIST cohort, with its assertions kept."""
    conditions = pd.read_csv(FRESH / "condition_summary_six_rules.csv",
                             float_precision="round_trip")
    seeds = pd.read_csv(FRESH / "fresh_analysis_rows_six_rules.csv",
                        float_precision="round_trip")
    paired = pd.read_csv(FRESH / "paired_contrasts_six_rules.csv",
                         float_precision="round_trip")
    seed_diffs = pd.read_csv(FRESH / "paired_seed_contrasts_six_rules.csv",
                             float_precision="round_trip")
    conditions = conditions[conditions.metric.eq("test_accuracy")
                            & conditions.rate_policy.eq("selected")].copy()
    seeds = seeds[seeds.rate_policy.eq("selected")].copy()
    paired = paired[paired.metric.eq("test_accuracy")
                    & paired.rate_policy.eq("selected")
                    & paired.contrast.isin(CONTRASTS)].copy()
    seed_diffs = seed_diffs[seed_diffs.metric.eq("test_accuracy")
                            & seed_diffs.rate_policy.eq("selected")].copy()
    assert len(seeds) == 120 and len(conditions) == 12
    assert len(paired) == 2 * len(CONTRASTS)
    assert seeds.epochs.eq(180).all()
    assert seeds.groupby(["architecture", "seed"]).initialized_model_sha256.nunique().eq(1).all()
    assert seeds[seeds.arm.eq("decoder_only")].decoder_only_core_unchanged.all()
    for architecture in ARCHITECTURES:
        for arm in ARMS:
            s = seeds[seeds.architecture.eq(architecture) & seeds.arm.eq(arm)]
            row = conditions[conditions.architecture.eq(architecture)
                             & conditions.arm.eq(arm)]
            assert len(s) == 10 and s.seed.nunique() == 10
            assert len(row) == 1 and int(row.iloc[0].n) == 10
            assert abs(s.test_accuracy.mean() - row.iloc[0]["mean"]) < 1e-12
        p = seeds[seeds.architecture.eq(architecture)].pivot(
            index="seed", columns="arm", values="test_accuracy")
        assert p.shape == (10, 6) and not p.isna().any().any()
        for name in CONTRASTS:
            lhs, rhs = name.split("_minus_")
            row = paired[paired.architecture.eq(architecture)
                         & paired.contrast.eq(name)]
            assert len(row) == 1 and int(row.iloc[0].n) == 10
            assert abs((p[lhs] - p[rhs]).mean() - row.iloc[0]["mean"]) < 1e-12
            fan = seed_diffs[seed_diffs.architecture.eq(architecture)
                             & seed_diffs.contrast.eq(name)]
            assert len(fan) == 10
            assert abs(fan.difference.mean() - row.iloc[0]["mean"]) < 1e-9
    return conditions, seeds, paired, seed_diffs


def cohort_contrasts(paired, seed_diffs):
    """E and F rows: per neuron - strict scalar, exact path - per neuron.

    Four separately trained cohorts, each replayed from its own frozen paired
    contrast table and re-checked against that cohort's seed outcomes; the
    per-seed fan is recomputed from the seed-level table in every case.
    """
    rows = []
    fresh_keys = {"identity": "neuron_shared_minus_strict_scalar",
                  "exact": "exact_path_minus_neuron_shared"}
    for architecture in ARCHITECTURES:
        for kind, key in fresh_keys.items():
            r = paired[paired.architecture.eq(architecture)
                       & paired.contrast.eq(key)].iloc[0]
            fan = seed_diffs[seed_diffs.architecture.eq(architecture)
                             & seed_diffs.contrast.eq(key)].sort_values("seed")
            rows.append(dict(
                cohort="mnist_fresh", task="MNIST fresh", protocol="fresh",
                architecture=architecture, kind=kind, mean_pp=100 * r["mean"],
                low_pp=100 * r.ci_low, high_pp=100 * r.ci_high,
                n_seeds=int(r.n), positive_seeds=int(r.positive),
                seed_pp=[100 * v for v in fan.difference.to_numpy()],
                source_table="source_data/image_ladder_controls/summaries/"
                             "paired_contrasts_six_rules.csv",
                source_contrast=key))
    factorial = pd.read_csv(SOURCE / "mnist_between_within_factorial/paired_contrasts.csv",
                            float_precision="round_trip")
    fo = pd.read_csv(SOURCE / "mnist_between_within_factorial/seed_outcomes.csv",
                     float_precision="round_trip")
    fo = fo[fo.between.eq("dfa")]
    for architecture in ARCHITECTURES:
        piv = fo[fo.architecture.eq(architecture)].pivot(
            index="seed", columns="within", values="test_accuracy")
        for kind, key, pair in (
                ("identity", "dfa within: neuron - scalar",
                 ("neuron specific", "scalar broadcast")),
                ("exact", "dfa within: exact path - neuron",
                 ("exact path", "neuron specific"))):
            r = factorial[factorial.architecture.eq(architecture)
                          & factorial.contrast.eq(key)].iloc[0]
            fan = (piv[pair[0]] - piv[pair[1]]).sort_index()
            assert int(r.n_seeds) == len(fan) == 15
            assert abs(fan.mean() - r.mean_difference) < 1e-9
            rows.append(dict(
                cohort="mnist_dfa", task="MNIST DFA", protocol="DFA",
                architecture=architecture, kind=kind,
                mean_pp=100 * r.mean_difference, low_pp=100 * r.ci95_low,
                high_pp=100 * r.ci95_high, n_seeds=int(r.n_seeds),
                positive_seeds=int(round(r.positive_seed_fraction * r.n_seeds)),
                seed_pp=[100 * v for v in fan.to_numpy()],
                source_table="source_data/mnist_between_within_factorial/"
                             "paired_contrasts.csv", source_contrast=key))
    fashion = pd.read_csv(SOURCE / "fashion_feedback_ladder/paired_contrasts.csv",
                          float_precision="round_trip")
    fs = pd.read_csv(SOURCE / "fashion_feedback_ladder/seed_outcomes.csv",
                     float_precision="round_trip")
    for architecture in ARCHITECTURES:
        piv = fs[fs.architecture.eq(architecture)].pivot(
            index="seed", columns="feedback", values="test_accuracy")
        for kind, key, pair in (
                ("identity", "neuron indexed - scalar fallback",
                 ("neuron indexed", "scalar fallback")),
                ("exact", "exact path - neuron indexed",
                 ("exact path", "neuron indexed"))):
            r = fashion[fashion.architecture.eq(architecture)
                        & fashion.contrast.eq(key)].iloc[0]
            fan = (piv[pair[0]] - piv[pair[1]]).sort_index()
            assert int(r.n_seeds) == len(fan) == 10
            assert abs(fan.mean() - r.mean_difference) < 1e-9
            rows.append(dict(
                cohort="fashion", task="Fashion-MNIST",
                protocol="matched-width scalar fallback",
                architecture=architecture, kind=kind,
                mean_pp=100 * r.mean_difference, low_pp=100 * r.ci95_low,
                high_pp=100 * r.ci95_high, n_seeds=int(r.n_seeds),
                positive_seeds=int(r.positive_seeds),
                seed_pp=[100 * v for v in fan.to_numpy()],
                source_table="source_data/fashion_feedback_ladder/"
                             "paired_contrasts.csv", source_contrast=key))
    cifar = pd.read_csv(SOURCE / "cifar10_additive_feedback_ladder_confirmatory/"
                        "paired_contrasts.csv", float_precision="round_trip")
    cs = pd.read_csv(SOURCE / "cifar10_additive_feedback_ladder_confirmatory/"
                     "seed_outcomes.csv", float_precision="round_trip")
    for kind, key in (("identity", "neuron specific minus strict scalar"),
                      ("exact", "exact path minus neuron specific")):
        r = cifar[cifar.contrast.eq(key)].iloc[0]
        fan = np.array([float(v) for v in str(r.seed_differences).split(";")])
        n_seed = cs[cs.feedback.eq("neuron specific")].seed.nunique()
        assert len(fan) == int(r.n_seeds) == n_seed == 20
        assert abs(fan.mean() - r.mean_difference) < 1e-8
        rows.append(dict(
            cohort="cifar10", task="CIFAR-10", protocol="additive [3,3,3,3]",
            architecture="additive", kind=kind,
            mean_pp=100 * r.mean_difference, low_pp=100 * r.ci95_low_difference,
            high_pp=100 * r.ci95_high_difference, n_seeds=int(r.n_seeds),
            positive_seeds=int(r.seeds_positive),
            seed_pp=[100 * v for v in fan],
            source_table="source_data/cifar10_additive_feedback_ladder_"
                         "confirmatory/paired_contrasts.csv",
            source_contrast=key))
    return pd.DataFrame(rows)


def read_bp_equivalence():
    """CIFAR exact path - backpropagation and its prespecified TOST decision."""
    cifar = pd.read_csv(SOURCE / "cifar10_additive_feedback_ladder_confirmatory/"
                        "paired_contrasts.csv", float_precision="round_trip")
    r = cifar[cifar.contrast.eq("exact path minus backpropagation")].iloc[0]
    decision = json.loads((SOURCE / "cifar10_additive_feedback_ladder_"
                           "confirmatory/summary.json").read_text())
    eq = decision["decision"]["exact_vs_bp_equivalence"]
    assert eq["equivalent"] and abs(eq["margin"] - 0.01) < 1e-12
    return dict(mean_pp=100 * r.mean_difference, low_pp=100 * r.ci95_low_difference,
                high_pp=100 * r.ci95_high_difference, n=int(r.n_seeds),
                margin_pp=100 * eq["margin"], p_tost=eq["p_tost"])


def read_capture(coordinate=CAPTURE_COORDINATE):
    """G: dictionary capture of the trained (and initial) exact-path fields."""
    per_seed = pd.read_csv(FRESH / "delivery_coordinate_capture.csv",
                           float_precision="round_trip")
    summary = pd.read_csv(FRESH / "delivery_coordinate_capture_summary.csv",
                          float_precision="round_trip")
    bases = [b for b, _ in CAPTURE_BASES]
    per_seed = per_seed[per_seed.cohort.eq("fresh")
                        & per_seed.coordinate.eq(coordinate)
                        & per_seed.basis.isin(bases)].copy()
    summary = summary[summary.coordinate.eq(coordinate)
                      & summary.basis.isin(bases)
                      & summary.metric.eq("mean_capture")].copy()
    assert len(summary) == 12 and set(summary.checkpoint) == {"initial", "trained"}
    assert per_seed.forward_max_difference.abs().max() < 1e-4
    for architecture in ARCHITECTURES:
        for checkpoint in ("initial", "trained"):
            for basis in bases:
                s = per_seed[per_seed.architecture.eq(architecture)
                             & per_seed.basis.eq(basis)
                             & per_seed.checkpoint.eq(checkpoint)]
                row = summary[summary.architecture.eq(architecture)
                              & summary.basis.eq(basis)
                              & summary.checkpoint.eq(checkpoint)].iloc[0]
                assert len(s) == 10 and s.seed.nunique() == 10 and int(row.n) == 10
                assert abs(s.mean_capture.mean() - row["mean"]) < 1e-9
    return per_seed, summary


# ── A: from loss to one arbor ─────────────────────────────────────────────
def credit_entry(ax):
    """One arbor of 128, its readout, its loss and the return of its error.

    Bands from the cell floor: the update rule and its two operands (0-27 pt),
    the somatic-error shaft carrying the Gamma gain dial (~41 pt), the neuron
    with two ghosted neighbours behind it (46-92), the stimulus tile (103-118).
    The two ghosts are offset up and to the right so the forward path leaves
    the hero soma into clear space; the error returns from the RIGHT edge of
    the loss card, so it never crosses the tag that defines delta_u.
    """
    f = Frame(ax)
    X, Y = f.fx, f.fy
    W = f.w_pt
    # -- update rule and operands -----------------------------------------
    x0, w_delta = X(1.0), _text_w_pt(ax, "δ", PT_BASE)
    _, _, starts = chain(f, (x0, Y(22.0)),
                         ["Δ", ("g", "i"), " = −η ", ("e", "i"), " ", ("δ", "n")],
                         size=PT_BASE)
    hat(f, starts[5], Y(22.0), w_delta)
    chain(f, (x0, Y(12.0)),
          [("e", "i"), " = ", ("x", "i"), " ", ("R", "n"), " (", ("E", "i"),
           " − ", ("V", "n"), ")"], size=PT_BASE, color=MUTE)
    _, _, starts = chain(f, (x0, Y(2.0)),
                         ["δ", " = A c,   c = Γ ", ("c", "source")],
                         size=PT_BASE, color=MUTE)
    hat(f, starts[0], Y(2.0), w_delta, color=MUTE)
    # -- the neuron and its two ghosted neighbours ------------------------
    base = (X(0.5), Y(46.0), X(0.50 * W), Y(46.0))
    for k in (2, 1):
        _fade_ghost(f.balanced_tree(
            (base[0] + X(8.5 * k), base[1] + Y(6.0 * k), base[2], base[3]),
            depth=2, ghost=True, labels=False, soma_r_pt=2.4))
    nodes = f.balanced_tree(base, depth=2, mode="plain", labels=False)
    sx, sy = nodes.soma
    exc = nodes["T1"]
    inh = (0.5 * (nodes["JR"][0] + nodes["T3"][0]),
           0.5 * (nodes["JR"][1] + nodes["T3"][1]))
    f.contact(exc, kind="exc")
    f.contact(inh, kind="inh")
    f.text(f._off(exc, -2.6, 1.4), "E", size=PT_BASE, color=COLORS["exc"],
           ha="right", va="bottom")
    f.text(f._off(inh, 2.6, 0.8), "I", size=PT_BASE, color=COLORS["inh"],
           ha="left")
    f.text((X(W - 1.0), Y(97.0)), "×128", size=PT_BASE, color=MUTE, ha="right")
    # -- stimulus: drawn, never a sub-300 dpi raster (DECISIONS Fig 1) -----
    tile = (X(1.0), Y(103.0), X(15.0), Y(15.0))
    tint_patch(ax, ("rect", *tile), color="grid", pct=70, edge=True,
               lw=LW_HAIR, radius_pt=1.0, zorder=1.0)
    for k in (1, 2, 3):
        ax.plot([tile[0] + X(2.0), tile[0] + tile[2] - X(2.0)],
                [tile[1] + tile[3] * k / 4.0] * 2, color=COLORS["grid"],
                lw=f.lw(LW_HAIR), zorder=1.2)
    f.text((tile[0] + tile[2] + X(2.5), tile[1] + tile[3] / 2.0), "x (MNIST)",
           size=PT_BASE, color=INK, ha="left")
    f.leader((tile[0] + tile[2] / 2.0, tile[1] - Y(0.5)),
             (exc[0], exc[1] + Y(3.2)), color=MUTE)
    # -- readout, loss, and the error that returns to this soma -----------
    card_w, card_x = 0.455 * W, W - 0.455 * W - 1.0
    read_y, loss_y = 76.0, 46.0
    f.task_card((X(card_x), Y(read_y), X(card_w), Y(16.0)))
    f.task_card((X(card_x), Y(loss_y), X(card_w), Y(24.0)))
    f.text((X(card_x + card_w / 2.0), Y(read_y + 8.0)), "ŷ = W z + b",
           size=PT_BASE, color=INK)
    f.text((X(card_x + card_w / 2.0), Y(loss_y + 17.0)), "loss L",
           size=PT_BASE, color=INK)
    chain(f, (X(card_x + card_w / 2.0), Y(loss_y + 7.0)),
          [("δ", "u"), " = ∂L/∂", ("y", "u")], size=PT_BASE, color=INK,
          ha="center")
    riser_x = X(card_x - 2.5)
    ax.plot([sx + X(nodes.soma_r_pt + 1.5), riser_x, riser_x],
            [sy, sy, Y(read_y + 8.0)], color=MUTE, lw=f.lw(LW_EDGE),
            solid_capstyle="round", solid_joinstyle="round", zorder=3)
    f.arrow((riser_x, Y(read_y + 8.0)), (X(card_x + 0.4), Y(read_y + 8.0)),
            color=MUTE, lw=LW_EDGE, head=3.4)
    f.text((riser_x - X(1.5), sy + Y(2.4)), "z", size=PT_BASE, color=INK,
           ha="right", va="bottom")
    f.arrow((X(card_x + card_w / 2.0), Y(read_y - 1.0)),
            (X(card_x + card_w / 2.0), Y(loss_y + 25.0)), color=MUTE,
            lw=LW_EDGE, head=3.4)
    tail = (sx + X(nodes.soma_r_pt + 11.0), sy - Y(nodes.soma_r_pt + 7.0))
    # QA 2026-09-09: the return shaft runs 3.5 pt below the delta-0 tag's
    # baseline and rises into the arrow tail, so it never cuts the tag
    shaft_y, right = tail[1] - Y(3.5), X(card_x + card_w)
    ax.plot([right, right, tail[0], tail[0]],
            [Y(loss_y), shaft_y, shaft_y, tail[1]], color=INK,
            lw=f.lw(LW_HAIR), solid_capstyle="round", solid_joinstyle="round",
            zorder=4.8)
    f.error_in(nodes.soma, label="δ0")
    gain_dial(f, (X(0.72 * W), shaft_y))
    f.require_soma_lowest()
    f.require_delta0()
    return ax


# ── B: three ways to spread one somatic error ─────────────────────────────
def deliveries(ax):
    """Three deliveries of one somatic error on one generic depth-3 tree.

    Card order matches panel D's x order (strict scalar, per neuron, exact
    path) and the emphasis stays on per neuron, so the hero card sits second
    (deviation D-5 from DESIGN_SPEC 0.5.5).  The tree geometry is identical in
    every card: one rect size, one floor, one error_in call site.
    """
    f = Frame(ax)
    X, Y = f.fx, f.fy
    foot_pt = f.footer("projected K = 1, 3: oracle coefficients on C's "
                       "dictionaries\ndecoder only: frozen core (D)", size=PT_BASE, band_pt=10.5, color=MUTE)
    gap_pt, share = 5.0, np.array([104.0, 88.0, 98.0])
    widths = share / share.sum() * (f.w_pt - 2 * gap_pt)
    cells, x_pt = [], 0.0
    for w_pt in widths:
        cells.append((X(x_pt), Y(foot_pt + 1.0), X(w_pt),
                      1.0 - Y(foot_pt + 1.0)))
        x_pt += w_pt + gap_pt
    specs = (("Strict scalar", False, ["one s across neurons"]),
             ("Per neuron", True, ["one ", ("δ", "u"), ", spread evenly"]),
             ("Exact path", False, None))
    tree_w_pt = 72.0
    for i, (cell, (title, hero, footer)) in enumerate(zip(cells, specs)):
        core = f.task_card(cell, title=title, emphasis=hero)
        core_pt = core[3] * f.h_pt
        band_pt = 11.0                      # the card's own footer line
        delta_pt = 12.0                     # room under the soma for delta-0
        rect = (core[0] + (core[2] - X(tree_w_pt)) / 2.0,
                core[1] + Y(band_pt + delta_pt), X(tree_w_pt),
                Y(core_pt - band_pt - delta_pt - 8.0))
        if i == 0:
            rect = (core[0] + X(3.0), rect[1], rect[2], rect[3])
        ghost = None
        if i == 0:
            mark = snapshot(ax)
            ghost = _fade_ghost(f.balanced_tree(
                (rect[0] + X(30.0), rect[1], rect[2], rect[3]), depth=3,
                ghost=True, labels=False))
            cut = ghost_clip(f, cell, new_artists(ax, mark))
        nodes = f.balanced_tree(rect, depth=3, labels=False)
        f.error_in(nodes.soma, label="δ0")
        if i == 0:
            keep = [t for t in ghost.terminals if ghost[t][0] < cut - X(16.0)]
            for j, t in enumerate(keep):
                nodes[f"G{j}"] = ghost[t]
            f.credit_delivery(nodes, mode="scalar",
                              rule_color=COLORS["scalar"],
                              targets=list(nodes.terminals)
                              + [f"G{j}" for j in range(len(keep))])
        elif i == 1:
            f.credit_delivery(nodes, mode="neuron", rule_color=COLORS["scalar"])
            f.badge((cell[0] + cell[2] - X(3.5), cell[1] + cell[3] - Y(3.0)),
                    "local rule")
        else:
            f.credit_delivery(nodes, mode="exact", rule_color=COLORS["bp"],
                              alpha_tags=True)
            f.badge((cell[0] + cell[2] - X(3.5), cell[1] + cell[3] - Y(3.0)),
                    "exact")
        if footer is None:                  # card 3: the path-gain equation
            # one typography for delta-hat-n: the same (base, sub) token
            # pair panel A's equation band uses, never a bare "n" span.
            _, _, starts = chain(f, (cell[0] + cell[2] / 2.0,
                                     core[1] + Y(3.5)),
                                 [("δ", "n"), " = ", ("α", "3"), ("α", "2"),
                                  ("α", "1"), ("δ", "0")], size=PT_BASE,
                                 color=INK, ha="center")
            hat(f, starts[0], core[1] + Y(3.5), _text_w_pt(ax, "δ", PT_BASE))
        else:
            chain(f, (cell[0] + cell[2] / 2.0, core[1] + Y(3.5)), footer,
                  size=PT_BASE, color=MUTE, ha="center")
    f.require_soma_lowest()
    f.require_delta0()
    return ax


# ── C: profiles at K = 1, 3, 12 ───────────────────────────────────────────
def dictionaries(ax):
    """The twelve nonsomatic sites, their display order, and two dictionaries.

    The arbor is drawn upright (soma lowest); the row order a matrix aligns to
    is carried by the site strip beside it, which is also the K = 12 dictionary
    (A = I is one profile per site, decision D-2).  Both matrices use ONE
    encoding: nonzero cells in neutral mute on panel_bg, so nothing inside a
    matrix can be read as a rule or an architecture; subtree identity is
    carried only by the strip's 16 % address bands.
    """
    f = Frame(ax)
    X, Y = f.fx, f.fy
    W = f.w_pt
    rows_h, y0 = 78.0, 38.0             # 12 rows at 6.5 pt
    address = list(K_CYCLE[:3])
    # QA 2026-09-09: the arbor's subtrees run left-to-right in the same
    # order as the strip's bands run top-to-bottom
    arbor_colors = address[::-1]
    nodes = f.site_tree((X(1.0), Y(y0), X(0.33 * W), Y(rows_h)),
                        branching=(3, 3), subtree_colors=arbor_colors,
                        orient="up")
    f.partition(nodes, [nodes.subtree(k) for k in range(3)],
                colors=arbor_colors, pct=16)
    f.error_in(nodes.soma, label="δ0")
    strip_x, strip_w = 0.42 * W, 0.145 * W
    f.site_strip((X(strip_x), Y(y0), X(strip_w), Y(rows_h)), branching=(3, 3),
                 subtree_colors=address, band=True, pct=16)
    f.text((X(strip_x + strip_w / 2.0), Y(y0 + rows_h + 1.5)), "K = 12: A = I",
           size=PT_BASE, color=MUTE, va="bottom")
    order = np.array(nodes.order)
    assert order.tolist() == [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11]
    a1 = np.ones((12, 1))
    a3 = np.zeros((12, 3))
    for k in range(3):
        a3[[k, 3 + 3 * k, 4 + 3 * k, 5 + 3 * k], k] = 1.0
    a12 = np.eye(12)
    boxes = ((0.615 * W, 0.075 * W, a1[order], "K = 1", "right"),
             (0.745 * W, 0.225 * W, a3[order], "K = 3", "left"))
    for x_pt, w_pt, data, name, ha in boxes:
        f.dictionary_matrix((X(x_pt), Y(y0), X(w_pt), Y(rows_h)), data,
                            color="mute", row_groups=[4, 4, 4], label=None,
                            col_labels=None)
        f.text((X(x_pt + w_pt) if ha == "right" else X(x_pt), Y(y0 - 8.0)),
               name, size=PT_BASE, color=INK, ha=ha, va="center")
    chain(f, (X(1.5), Y(24.0)), ["per neuron: c = ", ("δ", "u")], size=PT_BASE)
    _, x_end, _ = chain(f, (X(1.5), Y(13.0)), ["projected K = 1, 3:"],
                        size=PT_BASE)
    f.badge((x_end + X(4.0), Y(13.0)), "oracle", ha="left", va="center")
    f.text((X(1.5), Y(3.0)), "c from the exact field", size=PT_BASE, color=INK,
           ha="left")
    f.require_soma_lowest()
    f.require_delta0()
    return dict(broadcast=a1, subtrees=a3, resolved=a12, display_row_order=order)


# ── D: six rules on fresh MNIST ───────────────────────────────────────────
def accuracy(ax, conditions, seeds, paired, within):
    """Test accuracy for the six rules, ten fresh paired seeds per architecture."""
    xs = np.arange(6.0)
    tint_patch(ax, ("rect", 4.55, 78.0, 0.9, 20.5), color="mute", pct=10,
               edge=True, lw=LW_HAIR, radius_pt=1.5, zorder=0.15,
               clip_on=True)
    ax.text(5.42, 79.1, "decoder only: frozen core", ha="right", va="center",
            fontsize=PT_BASE, color=MUTE, zorder=6)
    printed = {}
    # Fill means ARCHITECTURE in D, E and F (filled shunting circle, open
    # additive square); only G re-uses open, for its initial checkpoint, and
    # says so in its own footer.
    for architecture, offset, marker, face in (
            ("shunting", -0.10, "o", "fill"), ("additive", 0.10, "s", "open")):
        color = COLORS[architecture]
        group = seeds[seeds.architecture.eq(architecture)]
        pivot = group.pivot(index="seed", columns="arm",
                            values="test_accuracy").loc[:, list(ARMS)]
        for _, seed in pivot.iterrows():
            ax.plot(xs + offset, 100 * seed.to_numpy(), color=color,
                    alpha=0.16, lw=LW_HAIR, zorder=1)
        for x, arm in zip(xs, ARMS):
            values = 100 * group[group.arm.eq(arm)].sort_values("seed").test_accuracy.to_numpy()
            ax.plot(x + offset + np.linspace(-0.05, 0.05, 10), values,
                    ls="none", marker="o", ms=SEED_MS, mfc=color, mec="none",
                    alpha=SEED_ALPHA, zorder=2)
            row = conditions[conditions.architecture.eq(architecture)
                             & conditions.arm.eq(arm)].iloc[0]
            ax.errorbar(x + offset, 100 * row["mean"],
                        yerr=[[100 * (row["mean"] - row.ci_low)],
                              [100 * (row.ci_high - row["mean"])]],
                        fmt=marker, color=color, ms=MARKER_MS,
                        mfc=color if face == "fill" else "white",
                        mec="white" if face == "fill" else color,
                        mew=LW_HAIR if face == "fill" else LW_ERR,
                        elinewidth=LW_ERR, capsize=2, zorder=4)
            printed[(architecture, arm)] = 100 * row["mean"]
        ax.text(5.58, 100 * conditions[conditions.architecture.eq(architecture)
                                       & conditions.arm.eq("decoder_only")].iloc[0]["mean"],
                architecture.capitalize(), ha="left", va="center",
                fontsize=PT_BASE, color=label_color(color), zorder=6)
    gains = [100 * paired[paired.architecture.eq(a)
                          & paired.contrast.eq("neuron_shared_minus_strict_scalar")].iloc[0]["mean"]
             for a in ARCHITECTURES]
    # one block on PT_BASE leading (1.37 data units = 9.0 pt on this
    # y-scale), so the number and its name read as one tagged contrast; the
    # slash order is stated because the two hue labels sit at the far right.
    ax.text(1.28, 93.6, " / ".join(signed(v) for v in gains) + " pp",
            ha="left", va="center", fontsize=PT_BASE, color=INK, zorder=6)
    ax.text(1.28, 92.23, "per neuron − strict scalar",
            ha="left", va="center", fontsize=PT_BASE, color=MUTE, zorder=6)
    credit = [100 * conditions[conditions.architecture.eq(a) & conditions.arm.eq(arm)].iloc[0]["mean"]
              for a in ARCHITECTURES for arm in ARMS[1:5]]
    spread = max(credit) - min(credit)
    ax.text(0.62, 81.8, f"four credit rules within {spread:.2f} pp — see E, F",
            ha="left", va="center", fontsize=PT_BASE, color=MUTE, zorder=6)
    ax.text(0.62, 89.0, "within tree, fresh cohort:\n"
            + "\n".join(f"{k}  {signed(v[0])} / {signed(v[1])}"
                        for k, v in within.items()),
            ha="left", va="top", fontsize=PT_BASE, color=MUTE, zorder=6,
            linespacing=1.35)
    ax.set(xlim=(-0.5, 6.3), ylim=(78, 98.5), xticks=xs, xticklabels=ARM_LABELS,
           yticks=[80, 85, 90, 95], ylabel="Test accuracy (%)")
    ax.set_yticks(np.arange(79, 98.5, 1.0), minor=True)
    style_panel(ax, grid="y")
    ax.tick_params(axis="x", labelsize=PT_BASE, pad=1.4, length=2.0)
    ax.yaxis.labelpad = 2.0
    return printed, spread


# ── E and F: the four-cohort forests ──────────────────────────────────────
def overlay_arm(ax, y, row, color, *, offset=ARM_OFFSET_ROWS):
    """The additive arm at +0.22 rows, open marker (CF-6, decision D-4)."""
    yy = y + offset
    seeds = np.asarray(row["seeds"], dtype=float)
    if len(seeds):
        jitter = np.linspace(-0.06, 0.06, len(seeds))
        ax.plot(seeds, yy + jitter, ls="none", marker="o", ms=SEED_MS,
                mfc=color, mec="none", alpha=SEED_ALPHA, zorder=2.0)
    ax.plot([row["lo"], row["hi"]], [yy, yy], color=color, lw=LW_ERR,
            zorder=3.0, solid_capstyle="butt")
    for xb in (row["lo"], row["hi"]):
        ax.plot([xb, xb], [yy - 0.13, yy + 0.13], color=color, lw=LW_ERR,
                zorder=3.0, solid_capstyle="butt")
    ax.plot([row["mean"]], [yy], ls="none", marker="s", ms=MARKER_MS,
            mfc="white", mec=color, mew=LW_ERR, zorder=4.0)


def row_note(ax, y, text, lo, hi, xlim, *, pad=0.03, side="right",
             below=False):
    """A per-row tag set ABOVE its row, right-aligned inside the right spine.

    QA 2026-09-09: rows 1 and 2 of E have no empty side (fans span 7-13 pp on
    a 0-20 axis), so an in-row tag always landed on a mark.  The tag now sits
    0.40 rows above the row (between rows; the second arm is 0.22 rows below
    the row), where no mark of any row lies.
    """
    span = xlim[1] - xlim[0]
    yy = y + 0.42 if below else y - 0.45     # last row: below (nothing there)
    if side == "right":
        ax.text(xlim[1] - pad * span, yy, text, ha="right", va="center",
                fontsize=PT_BASE, color=MUTE, zorder=6)
    else:                       # F: the right edge lies inside the band
        ax.text(xlim[0] + pad * span, yy, text, ha="left", va="center",
                fontsize=PT_BASE, color=MUTE, zorder=6)
    return _text_w_pt(ax, text, PT_BASE)


def cohort_forest(canvas, ax, cohorts, kind, *, value_label, xlim, xticks,
                  arch_code=False):
    """One half of the cohort comparison, through figure_canvas.forest()."""
    rows, extras = [], []
    for cohort, label in COHORT_ROWS:
        sub = cohorts[cohorts.cohort.eq(cohort) & cohorts.kind.eq(kind)]
        shunt = sub[sub.architecture.eq("shunting")]
        add = sub[sub.architecture.eq("additive")]
        lead = (shunt if len(shunt) else add).iloc[0]
        rows.append(dict(label=label, mean=lead.mean_pp, lo=lead.low_pp,
                         hi=lead.high_pp, seeds=list(lead.seed_pp),
                         n=int(lead.n_seeds),
                         color="shunting" if len(shunt) else "additive",
                         marker="o" if len(shunt) else "s"))
        second = add.iloc[0] if (len(shunt) and len(add)) else None
        # one form for every row: a two-arm row always prints both arms'
        # positive-seed counts, so only the genuinely single-arm CIFAR-10 row
        # carries one token (never a collapsed pair that reads as one arm).
        note = f"{int(lead.positive_seeds)}/{int(lead.n_seeds)}"
        if second is not None:
            pair = f"{int(second.positive_seeds)}/{int(second.n_seeds)}"
            # 'both 10/10' where the two arms agree: the shorter tag clears
            # the fan on the 0-20 axis (QA 2026-09-09)
            note = f"both {note}" if pair == note else f"{note} · {pair}"
        extras.append((second, note))
    # QA 2026-09-09: no row bands (the per-row tags sit between rows, and
    # the bands hid F's equivalence corridor); the tick still ties label to row
    out = canvas.forest(ax, rows, value_label=value_label, reference=0.0,
                        reference_label="", xlim=xlim, tag="", band=False,
                        tick=True, gutter_pt=FOREST_GUTTER_PT)
    ax.set_xticks(xticks)
    ax.set_xticklabels([minus(t) for t in xticks], fontsize=PT_BASE)
    lo, hi = ax.get_ylim()                       # inverted: lo > hi
    ax.set_ylim(lo + 0.35, hi - 0.55)            # bands above row 0 and below
                                                 # the last row for the two
                                                 # reference labels
    ax.annotate("no effect", xy=(0.0, hi - 0.50), xycoords=("data", "data"),
                xytext=(2.5, 0.0), textcoords="offset points", ha="left",
                va="center", fontsize=PT_BASE, color=MUTE, zorder=6)
    last = out["ypos"][-1]
    for y, row, (second, note) in zip(out["ypos"], rows, extras):
        span_lo, span_hi = row["lo"], row["hi"]
        if row["seeds"]:
            span_lo = min(span_lo, min(row["seeds"]))
            span_hi = max(span_hi, max(row["seeds"]))
        if second is not None:
            overlay_arm(ax, y, dict(mean=second.mean_pp, lo=second.low_pp,
                                    hi=second.high_pp,
                                    seeds=list(second.seed_pp)),
                        COLORS["additive"])
            span_lo = min(span_lo, second.low_pp, min(second.seed_pp))
            span_hi = max(span_hi, second.high_pp, max(second.seed_pp))
        row_note(ax, y, note, span_lo, span_hi, xlim,
                 side="left" if kind == "exact" else "right",
                 below=(y == last))
    # the arm key lives in the footnote (an in-row key collided with the
    # per-row tags): filled circle shunting, open square additive
    foot = "mean [95 % CI]" + ("; ● shunting, □ additive" if arch_code else "")
    ax.annotate(foot, xy=(1.0, 0.0), xycoords="axes fraction",
                xytext=(0.0, -25.0), textcoords="offset points", ha="right",
                va="top", fontsize=PT_BASE, color=MUTE, annotation_clip=False)
    return out


# ── G: dictionary capture of the trained field ────────────────────────────
def capture(ax, per_seed, summary):
    """Mean capture of D's exact-path fields by C's dictionaries, over K."""
    xs = np.arange(3.0)
    ax.axhline(1.0, color=MUTE, lw=LW_REF, dashes=(2.6, 2.0), zorder=1.0)
    ax.text(-0.42, 1.045, "exact field (A = I)", ha="left", va="center",
            fontsize=PT_BASE, color=MUTE, zorder=6)
    printed = {}
    for architecture, offset, marker in (("shunting", -0.06, "o"),
                                         ("additive", 0.06, "s")):
        color = COLORS[architecture]
        for checkpoint in ("initial", "trained"):
            means = []
            for x, (basis, _) in zip(xs, CAPTURE_BASES):
                row = summary[summary.architecture.eq(architecture)
                              & summary.basis.eq(basis)
                              & summary.checkpoint.eq(checkpoint)].iloc[0]
                means.append(float(row["mean"]))
                printed[(architecture, checkpoint, basis)] = float(row["mean"])
                if checkpoint == "trained":
                    s = per_seed[per_seed.architecture.eq(architecture)
                                 & per_seed.basis.eq(basis)
                                 & per_seed.checkpoint.eq(checkpoint)].sort_values("seed")
                    ax.plot(x + offset + np.linspace(-0.05, 0.05, len(s)),
                            s.mean_capture.to_numpy(), ls="none", marker="o",
                            ms=SEED_MS, mfc=color, mec="none",
                            alpha=SEED_ALPHA, zorder=2)
                    ax.errorbar(x + offset, float(row["mean"]),
                                yerr=[[float(row["mean"] - row.ci_low)],
                                      [float(row.ci_high - row["mean"])]],
                                fmt=marker, color=color, ms=MARKER_MS,
                                mfc=color, mec="white", mew=LW_HAIR,
                                elinewidth=LW_ERR, capsize=2, zorder=4)
                else:
                    ax.plot([x + offset], [float(row["mean"])], ls="none",
                            marker=marker, ms=MARKER_MS, mfc="white",
                            mec=color, mew=LW_ERR, zorder=3.6)
            if checkpoint == "trained":
                ax.plot(xs + offset, means, color=color, lw=LW_DATA, zorder=3)
            else:
                line, = ax.plot(xs + offset, means, color=color, lw=LW_HAIR,
                                zorder=2.6)
                line.set_dashes((2.2, 1.8))
    # direct labels right of the K = 12 markers, stacked (both arms reach
    # 1.0 there); the K = 3 side is crossed by the rising trained lines
    # colour-keyed names head the lower-left key block: every other placement
    # crossed a trained line or the canvas edge (QA 2026-09-09)
    ax.text(-0.42, 0.445, "Additive", ha="left", va="center", fontsize=PT_BASE,
            color=label_color(COLORS["additive"]), zorder=6)
    ax.text(-0.42, 0.375, "Shunting", ha="left", va="center", fontsize=PT_BASE,
            color=label_color(COLORS["shunting"]), zorder=6)
    for k, line in enumerate(("filled: trained", "open: initial",
                              "10 seeds per point", "activation coordinate")):
        ax.text(-0.42, 0.30 - 0.075 * k, line, ha="left", va="center",
                fontsize=PT_BASE, color=MUTE, zorder=6)
    ax.set(xlim=(-0.5, 2.5), ylim=(0.0, 1.10), xticks=xs,
           xticklabels=[lab for _, lab in CAPTURE_BASES],
           yticks=[0, 0.25, 0.50, 0.75, 1.00], xlabel="Profiles K",
           ylabel="Mean capture (fraction)")
    style_panel(ax, grid="y")
    ax.tick_params(axis="both", labelsize=PT_BASE, pad=1.4, length=2.0)
    ax.yaxis.labelpad = 1.6
    ax.xaxis.labelpad = 1.6
    return printed


# ── Source Data: the curated display table the caption names ─────────────
CURATED_COLUMNS = [
    "panel", "record", "architecture", "arm", "cohort", "contrast", "basis",
    "profiles_k", "checkpoint", "site_display_position", "site_index",
    "profile_column", "seed", "value", "mean", "ci_low", "ci_high", "n",
    "positive_seeds", "unit", "source_table",
]
CURATED_SCOPE = ("Display summaries from existing inputs; panel column "
                 "identifies the selected data. One record per drawn mark. "
                 "Panels A and B are schematics and contribute no rows. "
                 "Not an additional experiment.")


def write_curated(conditions, seeds, paired, within, cohorts, equivalence,
                  cap_seed, cap_summary, matrices):
    """Emit ``source_data/curated_publication/figure_01_plotted.csv``.

    One record per drawn mark, in panel order C -> G.  A and B are schematics
    (no data), so they contribute no rows; the panel column carries the letter
    the artwork actually uses, which the HEAD-era table -- written against the
    retired panel semantics, with a panel C of one-step-bound curves that is
    now Supplementary S3G -- did not.
    """
    rows = []
    order = list(np.asarray(matrices["display_row_order"], dtype=int))
    for name, key, k in (("K = 1", "broadcast", 1), ("K = 3", "subtrees", 3),
                         ("K = 12", "resolved", 12)):
        block = np.asarray(matrices[key], dtype=float)[order]
        for pos, site in enumerate(order):
            for col in range(block.shape[1]):
                rows.append(dict(
                    panel="C", record="indicator support", basis=name,
                    profiles_k=k, site_display_position=pos, site_index=site,
                    profile_column=col, value=float(block[pos, col]),
                    unit="indicator (0 or 1)",
                    source_table="source_data/credit_first_figures/"
                                 "figure_01_illustrative_dictionaries.npz"))
    cond_table = "source_data/image_ladder_controls/summaries/" \
                 "condition_summary_six_rules.csv"
    seed_table = "source_data/image_ladder_controls/summaries/" \
                 "fresh_analysis_rows_six_rules.csv"
    pair_table = "source_data/image_ladder_controls/summaries/" \
                 "paired_contrasts_six_rules.csv"
    for architecture in ARCHITECTURES:
        for arm in ARMS:
            r = conditions[conditions.architecture.eq(architecture)
                           & conditions.arm.eq(arm)].iloc[0]
            rows.append(dict(
                panel="D", record="arm mean", architecture=architecture,
                arm=arm, mean=100 * r["mean"], ci_low=100 * r.ci_low,
                ci_high=100 * r.ci_high, n=int(r.n),
                unit="test accuracy (%)", source_table=cond_table))
            fan = seeds[seeds.architecture.eq(architecture)
                        & seeds.arm.eq(arm)].sort_values("seed")
            for _, sr in fan.iterrows():
                rows.append(dict(
                    panel="D", record="seed", architecture=architecture,
                    arm=arm, seed=int(sr.seed), value=100 * sr.test_accuracy,
                    unit="test accuracy (%)", source_table=seed_table))
    for label, key in (("headline contrast", "neuron_shared_minus_strict_scalar"),
                       ("within-tree note", "subtree_k3_minus_projected_k1"),
                       ("within-tree note", "exact_path_minus_subtree_k3")):
        for architecture in ARCHITECTURES:
            r = paired[paired.architecture.eq(architecture)
                       & paired.contrast.eq(key)].iloc[0]
            rows.append(dict(
                panel="D", record=label, architecture=architecture,
                contrast=key, mean=100 * r["mean"], ci_low=100 * r.ci_low,
                ci_high=100 * r.ci_high, n=int(r.n),
                positive_seeds=int(r.positive), unit="percentage points",
                source_table=pair_table))
    for _, r in cohorts.iterrows():
        panel = "E" if r.kind == "identity" else "F"
        rows.append(dict(
            panel=panel, record="cohort contrast", architecture=r.architecture,
            cohort=r.cohort, contrast=r.source_contrast, mean=r.mean_pp,
            ci_low=r.low_pp, ci_high=r.high_pp, n=int(r.n_seeds),
            positive_seeds=int(r.positive_seeds), unit="percentage points",
            source_table=r.source_table))
        for i, v in enumerate(r.seed_pp):
            rows.append(dict(
                panel=panel, record="seed", architecture=r.architecture,
                cohort=r.cohort, contrast=r.source_contrast, seed=i,
                value=float(v), unit="percentage points",
                source_table=r.source_table))
    rows.append(dict(
        panel="F", record="equivalence margin and contrast",
        architecture="additive", cohort="cifar10",
        contrast="exact path minus backpropagation",
        mean=equivalence["mean_pp"], ci_low=equivalence["low_pp"],
        ci_high=equivalence["high_pp"], n=equivalence["n"],
        value=equivalence["margin_pp"], unit="percentage points",
        source_table="source_data/cifar10_additive_feedback_ladder_"
                     "confirmatory/paired_contrasts.csv"))
    cap_table = "source_data/image_ladder_controls/summaries/" \
                "delivery_coordinate_capture_summary.csv"
    seed_cap_table = "source_data/image_ladder_controls/summaries/" \
                     "delivery_coordinate_capture.csv"
    for architecture in ARCHITECTURES:
        for checkpoint in ("initial", "trained"):
            for basis, k in CAPTURE_BASES:
                r = cap_summary[cap_summary.architecture.eq(architecture)
                                & cap_summary.basis.eq(basis)
                                & cap_summary.checkpoint.eq(checkpoint)].iloc[0]
                rows.append(dict(
                    panel="G", record="capture mean",
                    architecture=architecture, basis=basis, profiles_k=int(k),
                    checkpoint=checkpoint, mean=float(r["mean"]),
                    ci_low=float(r.ci_low), ci_high=float(r.ci_high),
                    n=int(r.n), unit=f"mean {CAPTURE_COORDINATE}-error "
                                     "capture (fraction)",
                    source_table=cap_table))
                if checkpoint != "trained":
                    continue        # only the trained fan is drawn
                fan = cap_seed[cap_seed.architecture.eq(architecture)
                               & cap_seed.basis.eq(basis)
                               & cap_seed.checkpoint.eq(checkpoint)].sort_values("seed")
                for _, sr in fan.iterrows():
                    rows.append(dict(
                        panel="G", record="seed", architecture=architecture,
                        basis=basis, profiles_k=int(k), checkpoint=checkpoint,
                        seed=int(sr.seed), value=float(sr.mean_capture),
                        unit=f"mean {CAPTURE_COORDINATE}-error capture "
                             "(fraction)", source_table=seed_cap_table))
    frame = pd.DataFrame(rows, columns=CURATED_COLUMNS)
    assert set(frame.panel) == {"C", "D", "E", "F", "G"}
    CURATED.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CURATED, index=False)
    return CURATED


def register_curated(path):
    """Refresh the figure-1 record in ``curated_publication/manifest.json``."""
    manifest = path.parent / "manifest.json"
    data = json.loads(manifest.read_text())
    record = {"figure": 1,
              "source": "scripts/credit_first_figures/build_framework.py",
              "path": "source_data/curated_publication/figure_01_plotted.csv",
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "scope": CURATED_SCOPE}
    records = [r for r in data["records"] if r.get("figure") != 1]
    records.append(record)
    data["records"] = sorted(records, key=lambda r: r.get("figure", 0))
    manifest.write_text(json.dumps(data, indent=2) + "\n")
    readme = path.parent / "README.md"
    line = ("Figure 1 panels C-G are emitted by "
            "`scripts/credit_first_figures/build_framework.py`; panels A and B "
            "are schematics and contribute no rows.")
    text = readme.read_text()
    if line not in text:
        readme.write_text(text.rstrip("\n") + "\n\n" + line + "\n")
    return record


# ── build ─────────────────────────────────────────────────────────────────
def main():
    conditions, seeds, paired, seed_diffs = read_fresh()
    cohorts = cohort_contrasts(paired, seed_diffs)
    equivalence = read_bp_equivalence()
    cap_seed, cap_summary = read_capture()
    RECORDS.mkdir(exist_ok=True)

    canvas = NativeCanvas(490 / 72, 3, row_weights=[126, 126, 98],
                          hgutter_pt=40, vgutter_pt=40,
                          margins=Margins(left=36, right=12, top=22, bottom=38))
    a = canvas.panel("A", 0, 0, 4, schematic=True, title="From loss to one arbor")
    b = canvas.panel("B", 0, 4, 8, schematic=True,
                     title="Three ways to spread one somatic error")
    c = canvas.panel("C", 1, 0, 4, schematic=True,
                     title="Profiles at K = 1, 3, 12")
    d = canvas.panel("D", 1, 4, 8, title="Per-neuron credit carries the MNIST gain")
    e = canvas.panel("E", 2, 0, 4, title="Identity: +4 to +16 pp")
    f_ = canvas.panel("F", 2, 4, 4, title="Resolution: ≤ 0.2 pp")
    g = canvas.panel("G", 2, 8, 4, title="Capture rises with K")
    for ax in (e, f_, g):
        # left: the forest label gutter (W3).  top: 4 pt so row 2's titles and
        # letters clear D's two-line category ticks (audit_row_separation.py
        # floor 8.5 pt; 7.0 pt without it).
        canvas.declare_reserve(ax, left=FOREST_GUTTER_PT, top=4.0)

    within = {}
    for label, key in (("K = 3 − K = 1", "subtree_k3_minus_projected_k1"),
                       ("exact − K = 3", "exact_path_minus_subtree_k3")):
        within[label] = [100 * paired[paired.architecture.eq(a_)
                                      & paired.contrast.eq(key)].iloc[0]["mean"]
                         for a_ in ARCHITECTURES]
    printed_d, spread = accuracy(d, conditions, seeds, paired, within)
    out_e = cohort_forest(canvas, e, cohorts, "identity",
                          value_label="Per neuron − strict scalar (pp)",
                          xlim=(0.0, 20.2), xticks=[0, 5, 10, 15, 20],
                          arch_code=True)
    out_f = cohort_forest(canvas, f_, cohorts, "exact",
                          value_label="Exact path − per neuron (pp)",
                          xlim=(-2.35, 1.05), xticks=[-2, -1, 0, 1])
    # F carries the prespecified equivalence margin and names its referent
    tint_patch(f_, ("rect", -equivalence["margin_pp"], f_.get_ylim()[1],
                    2 * equivalence["margin_pp"],
                    f_.get_ylim()[0] - f_.get_ylim()[1]), color="mute", pct=10,
               edge=True, lw=LW_HAIR, radius_pt=1.5, zorder=1.6, clip_on=True)
    f_.text(1.0, f_.get_ylim()[0] - 0.30, "±1 pp equivalence vs BP",
            ha="right", va="center", fontsize=PT_BASE, color=MUTE, zorder=6)
    printed_g = capture(g, cap_seed, cap_summary)

    canvas.lock_reserves()          # settle the boxes before drawing in points
    credit_entry(a)
    deliveries(b)
    matrices = dictionaries(c)

    # -- every printed number, re-derived and asserted (check 10) ---------
    close(printed_d[("shunting", "strict_scalar")], 87.89, 5e-3)
    close(printed_d[("shunting", "neuron_shared")], 97.12, 5e-3)
    close(printed_d[("shunting", "decoder_only")], 80.96, 5e-3)
    close(printed_d[("additive", "strict_scalar")], 89.66, 5e-3)
    close(printed_d[("additive", "subtree_k3")], 97.34, 5e-3)
    close(spread, 0.23, 5e-3)
    for arch, cohort, kind, target in (
            ("shunting", "mnist_fresh", "identity", 9.230),
            ("additive", "mnist_fresh", "identity", 7.456),
            ("shunting", "mnist_dfa", "identity", 11.033),
            ("additive", "mnist_dfa", "identity", 10.797),
            ("shunting", "fashion", "identity", 4.463),
            ("additive", "fashion", "identity", 3.781),
            ("additive", "cifar10", "identity", 16.393),
            ("shunting", "mnist_fresh", "exact", 0.010),
            ("additive", "mnist_fresh", "exact", 0.182),
            ("shunting", "mnist_dfa", "exact", 0.028),
            ("additive", "mnist_dfa", "exact", 0.087),
            ("shunting", "fashion", "exact", -0.107),
            ("additive", "fashion", "exact", 0.132),
            ("additive", "cifar10", "exact", -0.859)):
        row = cohorts[cohorts.architecture.eq(arch) & cohorts.cohort.eq(cohort)
                      & cohorts.kind.eq(kind)].iloc[0]
        close(row.mean_pp, target, 5e-3)
    close(equivalence["mean_pp"], -0.115, 5e-3)
    for key, target in ((("shunting", "trained", "broadcast_k1"), 0.622),
                        (("shunting", "trained", "subtrees_k3"), 0.655),
                        (("shunting", "trained", "exact_k12"), 1.000),
                        (("additive", "trained", "broadcast_k1"), 0.764),
                        (("additive", "trained", "subtrees_k3"), 0.859),
                        (("shunting", "initial", "broadcast_k1"), 0.532),
                        (("shunting", "initial", "subtrees_k3"), 0.532),
                        (("additive", "initial", "broadcast_k1"), 0.851),
                        (("additive", "initial", "subtrees_k3"), 0.893)):
        close(printed_g[key], target, 5e-4)

    problems = list(canvas.save(OUT, name="credit_first_figure_01", dpi=180))
    MAIN.parent.mkdir(parents=True, exist_ok=True)
    MAIN.write_bytes(OUT.read_bytes())

    # -- render-time records and provenance -------------------------------
    np.savez_compressed(RECORDS / "figure_01_illustrative_dictionaries.npz",
                        **matrices)
    flat = cohorts.assign(seed_pp=cohorts.seed_pp.map(
        lambda v: ";".join(f"{x:.6f}" for x in v)))
    flat.to_csv(RECORDS / "figure_01_contrasts.csv", index=False)
    plotted = []
    for table, frame in (("condition_summary_six_rules.csv", conditions),
                         ("paired_contrasts_six_rules.csv", paired)):
        plotted.extend(dict(panel="D", source_table=str((FRESH / table).relative_to(JOURNAL)),
                            **r) for r in frame.to_dict("records"))
    plotted.extend(dict(panel="E" if r["kind"] == "identity" else "F",
                        **{k: v for k, v in r.items() if k != "seed_pp"})
                   for r in cohorts.to_dict("records"))
    pd.DataFrame(plotted).to_csv(RECORDS / "figure_01_six_arm_source.csv",
                                 index=False)
    seeds.to_csv(RECORDS / "figure_01_six_arm_seed_source.csv", index=False)
    cap_seed.assign(panel="G").to_csv(RECORDS / "figure_01_capture_source.csv",
                                      index=False)
    curated = write_curated(conditions, seeds, paired, within, cohorts,
                            equivalence, cap_seed, cap_summary, matrices)
    curated_record = register_curated(curated)
    files = [Path(__file__), JOURNAL / "scripts/figure_canvas.py",
             JOURNAL / "scripts/journal_style.py",
             JOURNAL / "scripts/native_schematics.py",
             JOURNAL / "scripts/credit_tree_schematics.py",
             *[FRESH / name for name in
               ["condition_summary_six_rules.csv", "fresh_analysis_rows_six_rules.csv",
                "paired_contrasts_six_rules.csv", "paired_seed_contrasts_six_rules.csv",
                "delivery_coordinate_capture.csv",
                "delivery_coordinate_capture_summary.csv"]],
             *[SOURCE / f"{study}/{name}"
               for study in ("mnist_between_within_factorial", "fashion_feedback_ladder",
                             "cifar10_additive_feedback_ladder_confirmatory")
               for name in ("paired_contrasts.csv", "seed_outcomes.csv")],
             SOURCE / "cifar10_additive_feedback_ladder_confirmatory/summary.json",
             *[SOURCE / "image_ladder_controls" / name for name in
               ["protocol.json", "selection.json", "projected_k1/protocol.json",
                "projected_k1/selection.json"]]]
    payload = dict(panel_sources={
        "A": "Schematic, no data: MNIST input at one excitatory and one "
             "inhibitory contact on one of 128 trees (two ghosted); readout "
             "and loss return that neuron's error delta_u, entering the soma "
             "as delta_0; the Gamma dial on the shaft is the diagonal gain of "
             "c = Gamma c_source (main.tex Eqs. dendriticlocalrule, routedfield).",
        "B": "Schematic, no data: three deliveries of one somatic error on one "
             "generic depth-3 tree -- strict scalar (amber bus spanning a ghost "
             "neighbour, source outside both trees), per neuron (the same bus "
             "sourced at the soma), exact path (bp chain with alpha tags). The "
             "MNIST model has two path factors.",
        "C": "Schematic, no data: the twelve nonsomatic sites (three proximal, "
             "nine distal) in display order, the K = 1 and K = 3 indicator "
             "dictionaries, and the site strip as the K = 12 identity "
             "dictionary; projected rules take oracle coefficients from the "
             "exact field.",
        "D": "Complete six-arm fresh MNIST selected-rate cohort: 10 paired "
             "seeds per architecture, 180 epochs, validation-selected state. "
             "The tag is the paired per-neuron minus strict-scalar contrast.",
        "E": "Per neuron minus strict scalar (pp), means with paired 95% "
             "seed-bootstrap intervals in four separately trained cohorts: "
             "fresh MNIST (10 seeds), MNIST DFA (15), Fashion-MNIST (10), "
             "flattened CIFAR-10 additive (20). Cohorts are not one ladder.",
        "F": "Exact path minus per neuron (pp) in the same rows; band, the "
             "prespecified +-1 pp equivalence margin against backpropagation; "
             "notes, the CIFAR exact-minus-backprop contrast with its TOST "
             "result and the fresh-cohort within-tree contrasts.",
        "G": f"Mean {CAPTURE_COORDINATE}-error capture of D's exact-path "
             "fields by C's dictionaries at K = 1, 3, 12; fresh cohort, 10 "
             "seeds per architecture, trained (filled) and initial (open) "
             "checkpoints."},
        source_sha256={str(p.relative_to(JOURNAL)):
                       hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        numerical_scope="No fitting, selection, bootstrap or source-outcome "
                        "mutation. Frozen 95% intervals replayed exactly; seed "
                        "means, paired contrast means, per-seed fans, cohort "
                        "seed counts and capture means recomputed as assertions.",
        plot_units={"D": "Test accuracy (%); source fractions x 100",
                    "E": "Paired per-neuron minus strict-scalar differences (pp)",
                    "F": "Paired exact-path minus per-neuron differences (pp)",
                    "G": "Mean per-field capture (fraction)"},
        capture_coordinate=CAPTURE_COORDINATE,
        curated_source_data=curated_record,
        cohort_seed_counts={"mnist_fresh": 10, "mnist_dfa": 15, "fashion": 10,
                            "cifar10": 20},
        schematic_fraction=round(SCHEMATIC_FRACTION, 4),
        waivers={"W1": f"schematic area {SCHEMATIC_FRACTION * 100:.1f} % > 30 % "
                       "(DECISIONS G4, on the AMENDMENTS B12 formula)",
                 "W2": "row heights 126/126/98 pt (an 8-module panel on a "
                       "116 pt row is aspect 2.59 > 2.40)",
                 "W3": f"forest label gutter declared at {FOREST_GUTTER_PT:.0f} pt "
                       "and locked per module column, so A and C draw in "
                       "92.1 pt and B and D in 262.3 pt"},
        normal_font_minimum_pt=PT_BASE, layout_findings=problems,
        layout_findings_note="save_native's live audit_text_over_data compares "
                             "bounding boxes and flags every tag that shares a "
                             "box with a mark; the compiled-page check "
                             "(figure_canvas.py --audit --strict, which tests "
                             "real path geometry and excludes tint bands) "
                             "reports 0 violations for this figure.")
    (RECORDS / "figure_01_sources.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "main": str(MAIN),
                      "schematic_fraction": round(SCHEMATIC_FRACTION, 4),
                      "layout_findings": problems}, indent=2))


if __name__ == "__main__":
    main()

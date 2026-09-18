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
* row_tag()      a per-row tag centred ON its own row and hung just outside
                 that row's own marks (proposed: figure_canvas.forest(
                 row_tags=)), with _units_per_pt() for the data-per-point
                 conversion it needs.
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
import re
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
from journal_style import ADDRESS_RAMP, K_CYCLE, label_color
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
# G draws the first two; the identity is the labelled rule (QA 2026-09-10).
CAPTURE_SEED_MS = 2.0                  # under the 2.9 pt panel default
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
    # QA 2026-09-10: hung 3.4 pt UNDER the ring, the label landed on the same
    # baseline, at the same size and in the same face as the update rule
    # immediately to its left, so it read as a trailing factor of that
    # equation rather than as the name of the dial above it.  It now sits
    # against the ring's own upper right shoulder, 8 pt off the equation
    # baseline and touching nothing else.
    f.text((xy[0] + f.fx(r_pt + 1.2), xy[1] + f.fy(1.4)), label, size=PT_BASE,
           color=MUTE, ha="left", va="bottom")


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
                seed_ids=[int(seed) for seed in fan.seed],
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
                seed_ids=[int(seed) for seed in fan.index],
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
                seed_ids=[int(seed) for seed in fan.index],
                seed_pp=[100 * v for v in fan.to_numpy()],
                source_table="source_data/fashion_feedback_ladder/"
                             "paired_contrasts.csv", source_contrast=key))
    cifar = pd.read_csv(SOURCE / "cifar10_additive_feedback_ladder_confirmatory/"
                        "paired_contrasts.csv", float_precision="round_trip")
    cs = pd.read_csv(SOURCE / "cifar10_additive_feedback_ladder_confirmatory/"
                     "seed_outcomes.csv", float_precision="round_trip")
    cifar_pairs = cs.pivot(index="seed", columns="feedback",
                          values="test_accuracy").sort_index()
    for kind, key in (("identity", "neuron specific minus strict scalar"),
                      ("exact", "exact path minus neuron specific")):
        r = cifar[cifar.contrast.eq(key)].iloc[0]
        fan = np.array([float(v) for v in str(r.seed_differences).split(";")])
        paired_values = cifar_pairs[r.left] - cifar_pairs[r.right]
        assert not paired_values.isna().any()
        # The frozen display values are rounded to nine decimal places.
        # Verify their ordering against actual seed IDs without changing them.
        np.testing.assert_allclose(fan, paired_values.to_numpy(),
                                   rtol=0, atol=5e-10)
        n_seed = len(paired_values)
        assert len(fan) == int(r.n_seeds) == n_seed == 20
        assert abs(fan.mean() - r.mean_difference) < 1e-8
        rows.append(dict(
            cohort="cifar10", task="CIFAR-10", protocol="additive [3,3,3,3]",
            architecture="additive", kind=kind,
            mean_pp=100 * r.mean_difference, low_pp=100 * r.ci95_low_difference,
            high_pp=100 * r.ci95_high_difference, n_seeds=int(r.n_seeds),
            positive_seeds=int(r.seeds_positive),
            seed_ids=[int(seed) for seed in paired_values.index],
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

    Bands from the cell floor: the update rule and its two operands (0-28 pt),
    the somatic-error shaft carrying the Gamma gain dial (32.5 pt), the hero
    neuron with ONE ghosted neighbour behind it (44-90 pt), and the stimulus
    tile (105-120 pt).

    QA 2026-09-09: the forward path is a mute HORIZONTAL arrow that leaves the
    soma rightward at soma height into the clear strip under the readout card
    (the previous riser turned up through the canopy and crossed a junction
    ring), the readout card is therefore the LOWER of the two cards, and the
    single ghost is offset (16, 20) pt -- 25.6 pt away, so the hero arbor, its
    two contacts and the z tag sit on clean ground.
    """
    f = Frame(ax)
    X, Y = f.fx, f.fy
    W = f.w_pt
    # 2026-09-14: the cell is no longer confined to the 92 pt beside the
    # forest gutter (the panel is unlocked), so the drawing's horizontal
    # geometry is set from the cell width: the arbor takes the left 45 %,
    # the two cards the right 46 pt, and the type stays on its tokens.
    k = max(W / 92.0, 1.0)
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
    # -- the neuron and its one ghosted neighbour -------------------------
    tree_w, tree_h, tree_y = min(44.0 * k, 0.46 * W), 52.0 + 4.0 * (k - 1.0), 44.0
    base = (X(0.5), Y(tree_y), X(tree_w), Y(tree_h))
    # QA 2026-09-09: no ghost neighbour in A.  A 92 pt panel that also holds
    # two task cards leaves a 7.5 pt window in which a ghost soma is both
    # clear of the hero canopy (which ends at x = 40.5 pt) and clear of the
    # cards (x >= 48 pt); at every offset that fits, the ghost's own soma
    # reads as a second hero soma inside the hero's branches.  The population
    # is stated by the x128 tag, and the ghost NEIGHBOUR is drawn where it
    # carries the argument -- B card 1, the strict scalar shared across cells.
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
    # -- stimulus: drawn, never a sub-300 dpi raster (DECISIONS Fig 1) -----
    tile = (X(1.0), Y(102.0), X(15.0 * min(k, 1.2)), Y(15.0 * min(k, 1.2)))
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
    card_w = 42.0 + 4.0 * min(k - 1.0, 1.0)
    card_x = W - card_w - 2.0                # flush with the cell's right edge
    read_y, read_h = 38.0, 16.0             # centred on the soma (y = 46)
    loss_y, loss_h = 62.0, 18.0
    f.task_card((X(card_x), Y(read_y), X(card_w), Y(read_h)))
    f.task_card((X(card_x), Y(loss_y), X(card_w), Y(loss_h)))
    f.text((X(card_x + card_w / 2.0), Y(read_y + read_h / 2.0)), "ŷ = Wz + b",
           size=PT_BASE, color=INK)
    f.text((X(card_x + card_w / 2.0), Y(loss_y + 12.0)), "loss L",
           size=PT_BASE, color=INK)
    chain(f, (X(card_x + card_w / 2.0), Y(loss_y + 4.5)),
          [("δ", "u"), " = ∂L/∂", ("z", "u")], size=PT_BASE, color=INK,
          ha="center")
    f.text((X(card_x + card_w), Y(loss_y + loss_h + 14.0)), "×128",
           size=PT_BASE, color=MUTE, ha="right")
    # forward: one mute HORIZONTAL arrow, soma height, no canopy crossed.
    # QA 2026-09-10: mute #363B41 and ink #232323 are within 0.08 in grey
    # value, so at print the forward shaft and the delta-0 shaft read as the
    # same stroke and the forward/error contrast rests on direction alone.
    # The forward arrow is therefore DASHED (the figure's own reference-line
    # dash) and dropped to LW_HAIR; delta-0 keeps its solid ink shaft and its
    # 4.5 pt head.  No new colour is spent.
    fwd_y = sy
    fwd = f.arrow((sx + X(nodes.soma_r_pt + 2.0), fwd_y), (X(card_x - 1.2), fwd_y),
                  color=MUTE, lw=LW_HAIR, head=3.4)
    fwd.set_linestyle(DASH)
    f.text(((sx + X(nodes.soma_r_pt + 2.0) + X(card_x - 1.2)) / 2.0,
            fwd_y + Y(3.0)), "z", size=PT_BASE, color=INK, va="bottom")
    # the second leg of the same forward path: one idiom, so it is dashed too
    up = f.arrow((X(card_x + card_w / 2.0), Y(read_y + read_h + 0.8)),
                 (X(card_x + card_w / 2.0), Y(loss_y - 1.0)), color=MUTE,
                 lw=LW_HAIR, head=3.4)
    up.set_linestyle(DASH)
    tail = (sx + X(nodes.soma_r_pt + 11.0), sy - Y(nodes.soma_r_pt + 7.0))
    # the return leaves the loss card's RIGHT edge, drops outside both cards
    # and runs left along the delta-0 shaft, 3.5 pt below the tag's baseline
    shaft_y, right = tail[1] - Y(7.5), X(card_x + card_w + 1.6)   # clear of the delta-0 subscript
    ax.plot([X(card_x + card_w), right, right, tail[0], tail[0]],
            [Y(loss_y + 6.0), Y(loss_y + 6.0), shaft_y, shaft_y, tail[1]],
            color=INK, lw=f.lw(LW_HAIR), solid_capstyle="round",
            solid_joinstyle="round", zorder=4.8)
    f.error_in(nodes.soma, label="δ0")
    gain_dial(f, (X(0.62 * W), shaft_y))
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
    # QA 2026-09-10: the two-line footer is deleted.  "projected K = 1, 3:
    # oracle coefficients on C's dictionaries" is C's own in-panel line, set
    # 250 pt below it in the same column pair, and "decoder only: frozen core"
    # is D's band key -- so both definitions were printed twice inside one
    # figure.  Each is now stated once, beside the thing it defines, and B's
    # three cards take the 21 pt the footer band held.
    foot_pt = 0.0
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
        band_pt = 17.0                      # footer, including subscript clearance
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
        # QA 2026-09-10, declined: re-pointing the amber drops at the junction
        # rings.  Measured on the built page, every drop in cards 1 and 2 ends
        # 3.33 pt above its OWN terminal tip (hero and ghost alike; twelve
        # drops checked in card 1, nine in card 2), i.e. on the canopy, not in
        # mid-air and never over empty space.  The 8-12 pt figure is the
        # distance to the nearest JUNCTION ring, and the drop endpoint
        # (target + 2.8 pt along the tree axis) and the terminal target set are
        # both owned by Frame.credit_delivery, which this build may not edit.
        # Library request: credit_delivery(drop_clearance_pt=..., to='junction').
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
        # Keep the subscript descenders inside the card, not on its lower
        # border.  The equation tokens extend below their nominal baseline.
        footer_y = core[1] + Y(9.5)
        if footer is None:                  # card 3: the path-gain equation
            # one typography for delta-hat-n: the same (base, sub) token
            # pair panel A's equation band uses, never a bare "n" span.
            _, _, starts = chain(f, (cell[0] + cell[2] / 2.0, footer_y),
                                 [("δ", "n"), " = ", ("α", "3"), ("α", "2"),
                                  ("α", "1"), ("δ", "0")], size=PT_BASE,
                                 color=INK, ha="center")
            hat(f, starts[0], footer_y, _text_w_pt(ax, "δ", PT_BASE))
        else:
            chain(f, (cell[0] + cell[2] / 2.0, footer_y), footer,
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
    # QA 2026-09-09: the stack keeps its top (117.5 pt, 2.7 pt under the
    # title) and its floor rises 3 pt, so the delta-0 tag clears the
    # matrix-name row and the coefficient band under it.
    rows_h, y0 = 75.0, 41.0             # 12 rows at 6.25 pt
    # QA 2026-09-11 (major): these were K_CYCLE[:3], i.e. the shunting,
    # additive and scalar inks, so C's three subtree tints came out
    # pixel-identical to the architecture series drawn in D-G and to B's
    # delivered-scalar arrow.  One hue then meant "which subtree" here and
    # "which architecture" two panels away.  Structure now uses the neutral
    # address ladder, which claims no series hue.
    address = list(ADDRESS_RAMP)
    # QA 2026-09-09: the arbor's subtrees run left-to-right in the same
    # order as the strip's bands run top-to-bottom
    arbor_colors = address[::-1]
    # QA 2026-09-10 (blocker): site_tree spreads its twelve display rows over
    # the rect's WIDTH, so the arbor's terminal pitch is rect_w / 12.  At the
    # old 0.33 W rect that pitch was 2.53 pt against a 3.30 pt disc: adjacent
    # terminals overlapped by 0.77 pt and each subtree's three sites fused
    # into one lozenge, so the 3 proximal + 9 distal = 12 correspondence the
    # panel exists to establish could not be counted.  The right-hand block
    # (strip, gap, K = 1, gap, K = 3) is now laid out from its own minimum
    # widths and the arbor takes every point that is left.
    strip_w, m1_w, m2_w = 11.0, 6.2, 18.6       # matrix cells 6.2 pt >= 6.0
    gap_arbor, gap_strip, gap_mat, pad_r = 2.5, 3.0, 5.5, 2.5
    right_block = (gap_arbor + strip_w + gap_strip + m1_w + gap_mat + m2_w
                   + pad_r)
    arbor_x = 0.5
    arbor_w = W - right_block - arbor_x
    assert arbor_w / 12.0 >= 3.2, arbor_w        # pitch floor
    nodes = f.site_tree((X(arbor_x), Y(y0), X(arbor_w), Y(rows_h)),
                        branching=(3, 3), subtree_colors=arbor_colors,
                        orient="up")
    # QA 2026-09-10: the site discs were painted at full shunting / additive /
    # scalar saturation, i.e. the three series hues D-G use for architecture
    # and delivery, which gives green and blue a second meaning inside one
    # figure.  Section 0.4 sanctions K_CYCLE only as 16 % address tints in the
    # anatomy register, so the discs are redrawn as 16 % faces on a dend edge;
    # the capsules and the strip bands stay the carriers of the address code.
    # The distal discs also drop from r 1.65 to 1.1 pt so that the sibling
    # clearance at the new pitch is a full point.
    for site, ring in nodes.rings.items():
        colour = arbor_colors[(site if site < 3 else (site - 3) // 3)
                              % len(arbor_colors)]
        ring.set_facecolor(mix(colour, 16))
        ring.set_edgecolor(COLORS["dend"])
        ring.set_linewidth(f.lw(LW_HAIR))
        if nodes.level[site] == 3:
            ring.set_width(2 * X(1.1 * f.scale))
            ring.set_height(2 * Y(1.1 * f.scale))
    f.partition(nodes, [nodes.subtree(k) for k in range(3)],
                colors=arbor_colors, pct=16)
    f.error_in(nodes.soma, label="δ0")
    strip_x = arbor_x + arbor_w + gap_arbor
    f.site_strip((X(strip_x), Y(y0), X(strip_w), Y(rows_h)), branching=(3, 3),
                 subtree_colors=address, band=True, pct=16)
    # QA 2026-09-10: the strip's name used to be a 48 pt mute line reaching
    # from over the arbor to over the K = 1 column, so it read as a title over
    # all three dictionaries rather than as the third dictionary's own label.
    # It is now the bare budget, in the same ink and the same words as "K = 1"
    # and "K = 3", and it is narrow enough to sit on the strip it names; that
    # A = I is the identity is the caption's sentence.  It cannot join the
    # other two on the y0 - 8 baseline: strip + matrices are 44.3 pt of ruler
    # and three such labels need 62 pt.
    f.text((X(strip_x + strip_w / 2.0), Y(y0 + rows_h + 1.5)), "K = 12",
           size=PT_BASE, color=INK, va="bottom")
    order = np.array(nodes.order)
    assert order.tolist() == [0, 3, 4, 5, 1, 6, 7, 8, 2, 9, 10, 11]
    a1 = np.ones((12, 1))
    a3 = np.zeros((12, 3))
    for k in range(3):
        a3[[k, 3 + 3 * k, 4 + 3 * k, 5 + 3 * k], k] = 1.0
    a12 = np.eye(12)
    m1_x = strip_x + strip_w + gap_strip
    m2_x = m1_x + m1_w + gap_mat
    boxes = ((m1_x, m1_w, a1[order], "K = 1", "right"),
             (m2_x, m2_w, a3[order], "K = 3", "left"))
    for x_pt, w_pt, data, name, ha in boxes:
        # QA 2026-09-09: COLORS['mute'] #363B41 is near-black at 4 modules --
        # the two matrices became the heaviest ink on the page and swallowed
        # their own hairline block separators.  A 62 % mute keeps the single
        # neutral encoding the plan mandates and lets the separators read.
        f.dictionary_matrix((X(x_pt), Y(y0), X(w_pt), Y(rows_h)), data,
                            color=mix("mute", 62), row_groups=[4, 4, 4],
                            label=None, col_labels=None)
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
def plateau_inset(ax, conditions, seeds):
    """The four credit rules again, y-magnified, in D's own empty floor.

    QA 2026-09-10 (major): the four middle arms span 97.113-97.340 = 0.227 pp,
    which is 1.1 % of D's 20.5 pp axis and 3 pt on the page, so the panel that
    exists to show the collapse could not show how flat the collapse is, could
    not show the ten seeds inside it (fan 0.1 pp = 1.4 pt, entirely under the
    mean symbol) and could not show the intervals.  The emptied floor now
    carries a y-magnified copy of columns 2-5: the x mapping is D's own, so
    each pair sits directly under the column it repeats, and no x axis is
    spent restating labels the panel already prints.  Nothing is added to the
    record -- every mark here is a mark of the main panel.
    """
    x0, x1, y0, y1 = 0.70, 4.50, 78.9, 86.2       # in D's own data coordinates
    box = ax.inset_axes([(x0 + 0.5) / 6.8, (y0 - 78.0) / 20.5,
                         (x1 - x0) / 6.8, (y1 - y0) / 20.5])
    box.set_zorder(5.0)
    box.set_facecolor("white")
    arms = ARMS[1:5]
    xs = np.arange(1.0, 5.0)
    for architecture, offset, marker, face in (
            ("shunting", -0.10, "o", "fill"), ("additive", 0.10, "s", "open")):
        color = COLORS[architecture]
        group = seeds[seeds.architecture.eq(architecture)]
        for x, arm in zip(xs, arms):
            values = 100 * group[group.arm.eq(arm)].sort_values("seed").test_accuracy.to_numpy()
            box.plot(x + offset + np.linspace(-0.05, 0.05, 10), values,
                     ls="none", marker="o", ms=SEED_MS, mfc=color, mec="none",
                     alpha=SEED_ALPHA, zorder=2)
            row = conditions[conditions.architecture.eq(architecture)
                             & conditions.arm.eq(arm)].iloc[0]
            box.errorbar(x + offset, 100 * row["mean"],
                         yerr=[[100 * (row["mean"] - row.ci_low)],
                               [100 * (row.ci_high - row["mean"])]],
                         fmt=marker, color=color, ms=MARKER_MS,
                         mfc=color if face == "fill" else "white",
                         mec="white" if face == "fill" else color,
                         mew=LW_HAIR if face == "fill" else LW_ERR,
                         elinewidth=LW_ERR, capsize=2, zorder=4)
    box.set(xlim=(x0, x1), ylim=(96.78, 97.58), xticks=[],
            yticks=[97.0, 97.2, 97.4])
    box.set_yticks(np.arange(96.8, 97.58, 0.05), minor=True)
    style_panel(box, grid="y")
    box.tick_params(axis="y", labelsize=PT_BASE, pad=1.4, length=2.0)
    box.tick_params(axis="x", length=0.0)
    return box


def accuracy(ax, conditions, seeds, paired, within):
    """Test accuracy for the six rules, ten fresh paired seeds per architecture."""
    xs = np.arange(6.0)
    tint_patch(ax, ("rect", 4.55, 78.0, 0.9, 20.5), color="mute", pct=10,
               edge=True, lw=LW_HAIR, radius_pt=1.5, zorder=0.15,
               clip_on=True)
    # QA 2026-09-10: the band's key used to be one 25-character line whose left
    # half lay outside the band, on white ground over the exact-path column, so
    # it read as another line of body text rather than as the band's label.  The
    # x tick under the band already says "Decoder only"; the band adds only what
    # was frozen, set in two short lines that fit inside the band's own width.
    ax.text(5.0, 97.9, "frozen\ncore", ha="center", va="top",
            fontsize=PT_BASE, color=MUTE, zorder=6, linespacing=1.35)
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
        ax.text(5.52, 100 * conditions[conditions.architecture.eq(architecture)
                                       & conditions.arm.eq("decoder_only")].iloc[0]["mean"],
                architecture.capitalize(), ha="left", va="center",
                fontsize=PT_BASE, color=label_color(color), zorder=6)
    # QA 2026-09-10 (major): the identity-gain block that used to sit here,
    # "+9.23 / +7.46 pp / per neuron - strict scalar", is deleted.  Both
    # numbers are E's own MNIST-fresh row (9.230 and 7.456), drawn there with
    # their intervals and their ten seeds, so the block was a verbatim second
    # printing of another panel of this figure inside D's emptied floor.
    credit = [100 * conditions[conditions.architecture.eq(a) & conditions.arm.eq(arm)].iloc[0]["mean"]
              for a in ARCHITECTURES for arm in ARMS[1:5]]
    spread = max(credit) - min(credit)
    # the within-tree contrasts are printed nowhere else in the figure (F plots
    # exact path - per neuron, a different pair), so they stay -- but as an
    # aligned two-column table on one leading, name at x = 1.15 and the
    # shunting / additive pair at a single x = 2.72, not as ragged prose.
    ax.text(1.15, 93.2, "within tree, fresh cohort:", ha="left", va="center",
            fontsize=PT_BASE, color=MUTE, zorder=6)
    for i, (name, values) in enumerate(within.items()):
        ax.text(1.15, 91.8 - 1.37 * i, name, ha="left", va="center",
                fontsize=PT_BASE, color=MUTE, zorder=6)
        ax.text(2.72, 91.8 - 1.37 * i,
                f"{signed(values[0])} / {signed(values[1])}", ha="left",
                va="center", fontsize=PT_BASE, color=MUTE, zorder=6)
    plateau_inset(ax, conditions, seeds)
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


def _units_per_pt(ax, xlim):
    """Data units per point on ``ax``'s x axis (the axes box is already locked)."""
    box = ax.get_window_extent()
    w_pt = box.width / ax.figure.dpi * 72.0
    return (xlim[1] - xlim[0]) / w_pt


def row_tag(ax, y, text, span_lo, span_hi, xlim, side, *, pad_pt=5.6):
    """A per-row tag centred ON its own row, hung off the end of that row.

    QA 2026-09-09: the tags used to sit 0.45 rows above their rows, i.e. in the
    inter-row gutter, where a reader cannot tell which row they belong to.  One
    rule now governs every row of both panels: the tag sits on the row's own
    baseline and hangs just outside the row's own marks, on whichever end of
    that row has room (``side`` is decided per row from the data spans).

    QA 2026-09-10: ``pad_pt`` is measured from the row's data span, but the
    span's extreme mark is a seed dot with its own radius, so the old 4.0 pt
    pad left only 1.38-1.43 pt of white between the tag box and the nearest
    dot (about 0.5 mm at print).  The pad is 5.6 pt, which puts every tag in
    both panels at >= 2.5 pt of measured clearance; the x-limit headroom for
    it already existed (E draws to 20.2 with the widest fan at 19.22).
    """
    upp = _units_per_pt(ax, xlim)
    pad = pad_pt * upp
    if side == "right":
        ax.text(span_hi + pad, y, text, ha="left", va="center",
                fontsize=PT_BASE, color=MUTE, zorder=6)
    else:
        ax.text(span_lo - pad, y, text, ha="right", va="center",
                fontsize=PT_BASE, color=MUTE, zorder=6)
    return _text_w_pt(ax, text, PT_BASE)


def cohort_forest(canvas, ax, cohorts, kind, *, value_label, xlim, xticks,
                  tag_sides, arch_code=False):
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
                         marker="o" if len(shunt) else "s",
                         # QA 2026-09-10: the additive arm is open in every
                         # row that has both arms, so the additive-only row
                         # must be open too.  Fill otherwise carried two
                         # meanings inside one figure: arm here, and
                         # trained-versus-initial in G.
                         hollow=not len(shunt)))
        second = add.iloc[0] if (len(shunt) and len(add)) else None
        # the plan's own row note: positive seeds out of the row's seed total,
        # for the arm the row label sits on; the single-arm CIFAR row says so
        # once, in E.
        note = f"{int(lead.positive_seeds)}/{int(lead.n_seeds)}"
        if second is None and kind == "identity":
            note += ", additive only"
        extras.append((second, note))
    # QA 2026-09-09: no row bands (they hid F's equivalence corridor); the
    # 0.55 pt row tick still ties the label to its row
    out = canvas.forest(ax, rows, value_label=value_label, reference=0.0,
                        reference_label="", xlim=xlim, tag="", band=False,
                        tick=True, gutter_pt=FOREST_GUTTER_PT)
    ax.set_xticks(xticks)
    ax.set_xticklabels([minus(t) for t in xticks], fontsize=PT_BASE)
    lo, hi = ax.get_ylim()                       # inverted: lo > hi
    ax.set_ylim(lo + 0.35, hi - 0.55)            # bands above row 0 and below
                                                 # the last row for the two
                                                 # reference labels
    # QA 2026-09-09: 2 pt lower than the top of the rule, so the mute label
    # clears the panel title box instead of crowding it
    ax.annotate("no effect", xy=(0.0, hi - 0.36), xycoords=("data", "data"),
                xytext=(2.5, 0.0), textcoords="offset points", ha="left",
                va="center", fontsize=PT_BASE, color=MUTE, zorder=6)
    for i, (y, row, (second, note)) in enumerate(zip(out["ypos"], rows,
                                                     extras)):
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
        row_tag(ax, y, note, span_lo, span_hi, xlim, tag_sides[i])
    if arch_code:
        # CF-5: the architecture code is stated ONCE, as two direct labels in
        # the series hues on row 1 -- shunting above the row line, additive
        # below the open arm that hangs 0.22 rows under it.
        edge = xlim[1] - 4.0 * _units_per_pt(ax, xlim)
        for name, key, dy in (("shunting", "shunting", -0.42),
                              ("additive", "additive", 0.30)):
            ax.text(edge, out["ypos"][0] + dy, name, ha="right", va="center",
                    fontsize=PT_BASE, color=label_color(COLORS[key]), zorder=6)
    # 2026-09-14: the "mean [95 % CI]" sub-label under the axis is gone; the
    # legend defines the marks and the line cost the forest a text row.
    return out


# ── G: dictionary capture of the trained field ────────────────────────────
def capture(ax, per_seed, summary):
    """Mean capture of D's exact-path fields by C's dictionaries, over K.

    QA 2026-09-10 (blocking): the K = 12 column was a constant.  Every one of
    its forty per-seed values is exactly 1.0 and its interval has zero width,
    because that dictionary is the identity.  Drawn as a third marker column it
    carried the panel's largest movement -- 25.9 pt for shunting against the
    2.49 pt of the only measured change -- and stacked 28 marks on top of the
    reference rule that already states the same fact.  It is now the labelled
    rule alone, and the panel draws the two budgets that were measured.
    """
    xs = np.arange(2.0)
    drawn = CAPTURE_BASES[:2]
    # QA 2026-09-10 (major): the axis used to run -0.78 to 1.22 so that the two
    # hue labels could hang in a left gutter.  Measured on the built page that
    # gutter was 39 % of the 73.8 pt plot width, all of it at profile counts
    # K <= 0, which cannot exist -- and it still was not wide enough: the
    # y spine struck the "S" of "Shunting" and the 0.75 rule ran through
    # "Additive".  The label column moves to the RIGHT of the K = 3 marks,
    # where each name sits on the end of its own trained arm, and the left pad
    # shrinks to the 0.07 profile units the seed fan needs.  The K = 12
    # identity stays the labelled rule (QA 2026-09-10, blocking): it is a
    # constant by construction and is not drawn as a third marker column.
    # QA 2026-09-11 (regression repair, two entries): the -0.26 left pad clipped
    # the outermost K = 1 shunting seed (fan edge at -0.23, dot radius 1 pt)
    # a third of the way under the y spine, and the 2.32 right edge left ~22 %
    # of the axis as bare rule past "Shunting" (which ends at x = 1.98).  The
    # left pad widens to -0.34 (~1.5 pt of white between dot and spine) and
    # the right edge tightens to 2.05, keeping label_x and jitter as measured
    # on 2026-09-10; the fan itself is not narrowed.
    xlo, xhi = -0.34, 2.55            # 2026-09-14: the panel is 3 modules wide
    label_x, jitter = 1.30, 0.17       # the fan is rescaled with the axis
    ax.axhline(1.0, color=MUTE, lw=LW_REF, dashes=(2.6, 2.0), zorder=1.0)
    # on TOP of its own rule, right-aligned: below the rule the label would
    # now sit on the additive initial marker at K = 3.
    ax.text(xhi - 0.02, 1.012, "identity, K = 12", ha="right", va="bottom",
            fontsize=PT_BASE, color=MUTE, zorder=6)
    # The rule stands for the identity column, so its level is read back from
    # the same table the markers use and asserted with them, not hard-coded.
    identity = {}
    for architecture in ARCHITECTURES:
        for checkpoint in ("initial", "trained"):
            row = summary[summary.architecture.eq(architecture)
                          & summary.basis.eq("exact_k12")
                          & summary.checkpoint.eq(checkpoint)].iloc[0]
            identity[(architecture, checkpoint, "exact_k12")] = float(row["mean"])
    assert set(identity.values()) == {1.0}, identity
    # QA 2026-09-10: the four mute lines that used to sit inside these axes
    # were verbatim duplicates of the caption six lines below the figure, and
    # they held the y range open so that data filled only 44 % of it.  They are
    # deleted; the caption carries them.
    printed = {}
    for architecture, offset, marker in (("shunting", -0.06, "o"),
                                         ("additive", 0.06, "s")):
        color = COLORS[architecture]
        for checkpoint in ("initial", "trained"):
            means = []
            for x, (basis, _) in zip(xs, drawn):
                row = summary[summary.architecture.eq(architecture)
                              & summary.basis.eq(basis)
                              & summary.checkpoint.eq(checkpoint)].iloc[0]
                means.append(float(row["mean"]))
                printed[(architecture, checkpoint, basis)] = float(row["mean"])
                if checkpoint == "trained":
                    s = per_seed[per_seed.architecture.eq(architecture)
                                 & per_seed.basis.eq(basis)
                                 & per_seed.checkpoint.eq(checkpoint)].sort_values("seed")
                    # QA 2026-09-10: the fan was +-0.05 ordinal units = 2.31 pt
                    # for ten 2.9 pt dots, an elevenfold overlap that printed as
                    # one smear.  Widened, and the dot is smaller than the
                    # panel default so the ten separate.
                    ax.plot(x + offset + np.linspace(-jitter, jitter, len(s)),
                            s.mean_capture.to_numpy(), ls="none", marker="o",
                            ms=CAPTURE_SEED_MS, mfc=color, mec="none",
                            alpha=SEED_ALPHA, zorder=2)
                    # QA 2026-09-10: no yerr.  The widest 95 % interval here
                    # is 0.007 capture units = 0.52 pt, 11 % of the 4.6 pt
                    # marker, and the caption already states that the intervals
                    # are below symbol size.  A drawn mark the caption calls
                    # invisible is worse than none.
                    ax.plot([x + offset], [float(row["mean"])], ls="none",
                            marker=marker, ms=MARKER_MS, mfc=color,
                            mec="white", mew=LW_HAIR, zorder=4)
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
    # true direct labels: each name is set at the END of its own trained arm,
    # inside the box and past the K = 3 seed fan, so the nearest mark to a
    # label is the series it names and no label crosses a spine or a rule.
    for architecture in ARCHITECTURES:
        row = summary[summary.architecture.eq(architecture)
                      & summary.basis.eq("subtrees_k3")
                      & summary.checkpoint.eq("trained")].iloc[0]
        ax.text(label_x, float(row["mean"]), architecture.capitalize(),
                ha="left", va="center", fontsize=PT_BASE,
                color=label_color(COLORS[architecture]), zorder=6)
    printed.update(identity)
    ax.set(xlim=(xlo, xhi), ylim=(0.45, 1.10), xticks=xs,
           xticklabels=[lab for _, lab in drawn],
           yticks=[0.50, 0.75, 1.00], xlabel="Profiles K",
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
        assert len(r.seed_ids) == len(r.seed_pp) == int(r.n_seeds)
        assert len(set(r.seed_ids)) == int(r.n_seeds)
        for seed, v in zip(r.seed_ids, r.seed_pp):
            rows.append(dict(
                panel=panel, record="seed", architecture=r.architecture,
                cohort=r.cohort, contrast=r.source_contrast, seed=int(seed),
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
    # 2026-09-10: the manifest now also carries the supplement's "S1".."S36"
    # records, so sort main figures first by number and SI figures after.
    def _order(record):
        value = str(record.get("figure", ""))
        digits = re.sub(r"\D", "", value)
        return (value.upper().startswith("S"), int(digits) if digits else 0, value)

    data["records"] = sorted(records, key=_order)
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

    # Design pass 2026-09-14: panel area follows content.  G draws eight
    # points and becomes a 3-module panel at the right of the bottom row; E,
    # whose axis spans 20 pp, takes five modules and F four.  The data panels
    # carry no sentence titles (the legend states the claims) and the row
    # gutter is 30 pt.  Letters keep their reading order, so no pointer in the
    # text moves.
    canvas = NativeCanvas(472 / 72, 3, row_weights=[126, 126, 100],
                          hgutter_pt=40, vgutter_pt=30,
                          margins=Margins(left=36, right=12, top=22, bottom=38))
    a = canvas.panel("A", 0, 0, 4, schematic=True, title="DendriNet image classifier",
                     lock=False)
    b = canvas.panel("B", 0, 4, 8, schematic=True,
                     title="From shared feedback to exact paths")
    # C and D start in the same grid columns as E and F below them, so the
    # forest label gutter locked on those columns is shared rather than paid
    # by the forests alone (the canvas locks reserves per column).
    c = canvas.panel("C", 1, 0, 5, schematic=True,
                     title="Profiles at K = 1, 3, 12", lock=False)
    d = canvas.panel("D", 1, 5, 7)
    e = canvas.panel("E", 2, 0, 5)
    f_ = canvas.panel("F", 2, 5, 4)
    g = canvas.panel("G", 2, 9, 3)
    for ax in (e, f_):
        # the forest label gutter (W3), locked per row; G is not a forest
        canvas.declare_reserve(ax, left=FOREST_GUTTER_PT)

    within = {}
    for label, key in (("K = 3 − K = 1", "subtree_k3_minus_projected_k1"),
                       ("exact − K = 3", "exact_path_minus_subtree_k3")):
        within[label] = [100 * paired[paired.architecture.eq(a_)
                                      & paired.contrast.eq(key)].iloc[0]["mean"]
                         for a_ in ARCHITECTURES]
    printed_d, spread = accuracy(d, conditions, seeds, paired, within)
    # tag_sides: the end of each row that its own marks leave free (see
    # row_tag).  E rows 1, 2 and 4 hang left, row 3 right; F rows 1-3 hang
    # left and the CIFAR row right.
    # QA 2026-09-10, declined: flipping E's row 3 to the left to make the rule
    # one-side-per-panel.  Measured on the built page, row 3's leftmost mark is
    # at x = 85.59 pt and the 'Fashion-MNIST' row label ends at x = 70.0 pt, so
    # a left-hung '10/10' (17.5 pt wide plus the 5.6 pt pad) would start at
    # x = 62.5 pt and overlap its own row label by 7.5 pt.  That trades a
    # non-overlap for a real collision; the tag stays on the row's own baseline
    # with its 0.55 pt row tick, a full 17.6 pt row below row 2's marks.
    out_e = cohort_forest(canvas, e, cohorts, "identity",
                          value_label="Per neuron − scalar baseline (pp)",
                          xlim=(0.0, 20.2), xticks=[0, 5, 10, 15, 20],
                          tag_sides=("left", "left", "right", "left"),
                          arch_code=True)
    out_f = cohort_forest(canvas, f_, cohorts, "exact",
                          value_label="Exact path − per neuron (pp)",
                          xlim=(-2.35, 1.05), xticks=[-2, -1, 0, 1],
                          tag_sides=("left", "left", "left", "right"))
    # QA 2026-09-10: F's title claims a resolution of 0.2 pp and its coarsest
    # tick was 1 pp, so the three near-zero rows could not be read against any
    # gradation finer than the effect being claimed.  Unlabelled 0.25 pp minor
    # ticks; the axis is NOT broken -- the CIFAR seed cloud reaches -2.11 and
    # legitimately needs the range.
    f_.set_xticks(np.arange(-2.25, 1.01, 0.25), minor=True)
    # F carries the prespecified equivalence margin and names its referent.
    # QA 2026-09-09 (blocker): the band is painted UNDER the zero rule
    # (zorder 0.12 < forest()'s axvline at 1.0), so the panel's load-bearing
    # anchor stays visible.
    # QA 2026-09-10 (major): the margin is the CIFAR-10 exact-path-MINUS-BP
    # contrast, and only that cohort has a BP arm at all.  Painted across all
    # four rows it certified three cohorts against a test never run on them,
    # on an axis that plots a different contrast.  It is now confined to the
    # CIFAR-10 row, with that contrast drawn inside it as its own mark.
    cifar_y = out_f["ypos"][-1]
    # a slim strip under the CIFAR row, so the row's own marks and its seed
    # tag stay outside the margin that does not apply to them
    tint_patch(f_, ("rect", -equivalence["margin_pp"], cifar_y + 0.04,
                    2 * equivalence["margin_pp"], 0.52), color="mute", pct=10,
               edge=True, lw=LW_HAIR, radius_pt=1.5, zorder=0.12, clip_on=True)
    f_.plot([equivalence["mean_pp"]], [cifar_y + 0.30], linestyle="none",
            marker="D", ms=MARKER_MS - 1.0, mfc="white",
            mec=COLORS["bp"], mew=LW_ERR, zorder=4.2)
    f_.plot([equivalence["low_pp"], equivalence["high_pp"]],
            [cifar_y + 0.30] * 2, color=COLORS["bp"], lw=LW_ERR, zorder=4.0,
            solid_capstyle="butt")
    f_.text(equivalence["margin_pp"] - 0.08, cifar_y + 0.30, "vs BP",
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
    flat = cohorts.assign(
        seed_ids=cohorts.seed_ids.map(lambda v: ";".join(str(x) for x in v)),
        seed_pp=cohorts.seed_pp.map(lambda v: ";".join(f"{x:.6f}" for x in v)))
    flat.to_csv(RECORDS / "figure_01_contrasts.csv", index=False)
    plotted = []
    for table, frame in (("condition_summary_six_rules.csv", conditions),
                         ("paired_contrasts_six_rules.csv", paired)):
        plotted.extend(dict(panel="D", source_table=str((FRESH / table).relative_to(JOURNAL)),
                            **r) for r in frame.to_dict("records"))
    plotted.extend(dict(panel="E" if r["kind"] == "identity" else "F",
                        **{k: v for k, v in r.items()
                           if k not in {"seed_ids", "seed_pp"}})
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
        "E": "Per neuron minus each cohort's scalar baseline (pp), means with paired 95% "
             "seed-bootstrap intervals in four separately trained cohorts: "
             "fresh MNIST (10 seeds), MNIST DFA (15), Fashion-MNIST (10), "
             "flattened CIFAR-10 additive (20). Cohorts are not one ladder.",
        "F": "Exact path minus per neuron (pp) in the same rows; band, the "
             "prespecified +-1 pp equivalence margin against backpropagation; "
             "notes, the CIFAR exact-minus-backprop contrast with its TOST "
             "result and the fresh-cohort within-tree contrasts.",
        "G": f"Mean {CAPTURE_COORDINATE}-error capture of D's exact-path "
             "fields by C's dictionaries at K = 1 and K = 3; fresh cohort, 10 "
             "seeds per architecture, trained (filled) and initial (open) "
             "checkpoints. K = 12 is the identity dictionary and reproduces "
             "the field exactly, so it is the reference line at 1 and not a "
             "drawn column; intervals are at most 0.007 and are not drawn."},
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

#!/usr/bin/env python3
"""Supplementary sheet S3 (ident ``utility_signal_noise``) -- restricted routes
trade signal retention against admitted noise, and the one-step quantity is
measurable at trained states -- rebuilt as ONE native full-width
:class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/utility_signal_noise.pdf``)
is a paste of three upstream renders: panels A-D are crops of
``figure_S46_utility.pdf`` (``scripts/build_utility_supplement.py``), E and G
are the two stand-alone components of ``scripts/build_si_restored_panels.py``,
and F is a crop of ``figure_S05_panels_A-H.pdf`` whose only generator
(``scripts/run_alignment_controlled_learning.py``) trains models and is
treated as FROZEN.  This builder reads ONLY frozen tables under
``source_data/`` and redraws the same seven panels with the same plotted
quantities.  Nothing about the numbers changes; every printed or plotted
value is asserted against the table it comes from, and every mean and
interval is recomputed from the raw per-seed / per-cell / per-stream rows
with the study's own bootstrap before it is drawn:

* B, C, D -- ``run_credit_phase_theory_experiment.py::bootstrap`` (20,000
  whole-seed draws, ``default_rng(base + 100 * group + metric)``), which
  reproduces every ``*_summary.csv`` row to its printed precision;
* E -- the 10,000-draw checkpoint bootstrap of
  ``build_si_restored_panels.py::_bootstrap_ci`` (``default_rng(700 + 10 *
  metric + family)``), whose means are asserted against the frozen
  ``mechanism_feedback_summary_valid.csv`` rows;
* F -- ``run_alignment_controlled_learning.py::hierarchical_interval``
  (20,000 draws: cells with replacement, then one Monte Carlo stream within
  each drawn cell, ``default_rng(20260731 + 9_000_001)`` advanced group by
  group in the study's own order), which reproduces every interval of
  ``alignment_controlled_curves.csv`` bit for bit.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
mixed_S3.json), panel by panel:

* Whole sheet: one module grid, one outer margin, one horizontal and one
  vertical gutter; A/C/E start in module column 0 and B/D in column 6, so
  the row-1/row-2 split repeats exactly; the three row-3 panels share one
  axes-box height and one baseline by construction; every panel is set in
  the 7/8/9 pt Nimbus tokens (the frozen S5 crop's 7.6/8.4 pt DejaVu is
  gone with the crop).
* A is the shared operator schematic of ``build_utility_supplement.py``
  (imported, not copied), sized from its own cell.
* B, D print every cell (no colour key); B keeps the K = 16 column and the
  rho = 0 row as the by-construction controls.
* C draws the four task hierarchies on the journal ORDINAL ramp (teal, a
  hue no series on this sheet claims; the old navy H_c = 3 matched F's
  depth-bins blue), every one of the 50 paired seeds as a fan behind each
  mean, the 95 % seed-bootstrap interval as a whisker, and the four series
  dodged by 0.12 of a resolution step so the fans stay apart.
* D is on TRUE numeric axes: the sampled fractions (0.25, 0.50, 0.75, 0.90,
  1.00 and 0.10, 0.25, 0.50, 0.75, 1.00) are cells centred on the samples
  and bounded by the midpoints between them, so the dashed rule is the
  exact boundary f_noise = f_sig, a 45-degree line, not a staircase.
* E drops the two exact-path slots (both quantities 1 by construction);
  unity is a dark-red reference rule labelled as such; all 120 checkpoints
  are drawn behind each box, the mean sits beside its box, the two
  strict-scalar values below the axis floor are drawn at the floor and
  printed; the axis is tight to the strips.
* F (previously frozen) now has an in-panel key with the four marker
  shapes, draws the eight per-cell values behind every mean, and states
  n = 8 arbors; the hierarchical 95 % intervals are whiskers.
* G is analytic and carries no series hue: both curves are ink (solid vs
  dashed), the crossing is marked once by the annotation leader, and the
  label sits clear of both curves.
* One palette register: route green (A's route capsules, F's
  morphology-selected routes), control grey (random paths), additive blue
  (depth bins), rose (ancestry-shuffled) in F; broadcast amber (strict
  scalar) and per-neuron salmon in E; exact path dark red as the reference
  rule in E; the ordinal teal ramp in C; the diverging journal ramp in B
  and D.  No colour carries a second meaning on the sheet.

2026-09-23 clarity pass (analysis/figure_visual_review_20260910/
review_20260923/si_pass/ledger/utility_signal_noise.md): the panel titles,
the method tags above B-G, the "exact path = 1" and "full gradient = 1" rule
labels and G's crossing callout moved to the caption; axis labels are in
sentence case, the "(= full rank)" notes left the B and C axis labels and G's
y label reads "2L × optimized bound", the caption's wording.
Every plotted mark, scale and colour is unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    DIV_CMAP,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_BASE,
    SEED_ALPHA,
    SEED_MS,
    Margins,
    NativeCanvas,
)
from journal_style import ORDINAL_RAMP, style_direct_color_labels  # noqa: E402
from build_utility_supplement import _MinusFmt, operator_schematic, signed_heatmap  # noqa: E402

ROOT = SCRIPT_DIR.parent
PHASE = ROOT / "source_data" / "credit_phase_theory"
VALIDITY = ROOT / "source_data" / "prospective_input_validity"
ALIGNED = ROOT / "source_data" / "alignment_controlled"
OUT = ROOT / "figures" / "supplementary" / "figure_utility_signal_noise_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
DASHED = (0, (2.6, 1.8))
N_SEEDS = 50
N_CHECKPOINTS = 120
N_CELLS = 8
N_STREAMS = 40
PHASE_DRAWS = 20000
CHECKPOINT_DRAWS = 10000
CELL_DRAWS = 20000

# ── the study bootstraps, verbatim ───────────────────────────────────────
def phase_bootstrap(values, seed, draws=PHASE_DRAWS):
    """``run_credit_phase_theory_experiment.py::bootstrap`` verbatim."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(int(draws), len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def checkpoint_bootstrap(values, seed, draws=CHECKPOINT_DRAWS):
    """``build_si_restored_panels.py::_bootstrap_ci`` verbatim."""
    rng = np.random.default_rng(seed)
    sample = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return (float(np.mean(values)), float(np.percentile(sample, 2.5)),
            float(np.percentile(sample, 97.5)))


def hierarchical_interval(frame, value, rng, n_boot=CELL_DRAWS):
    """``run_alignment_controlled_learning.py::hierarchical_interval``: cells
    with replacement, then one Monte Carlo stream within each drawn cell.

    The study draws ``rng.choice(roots, 8)`` and then ``rng.choice(streams)``
    once per drawn root; ``Generator.choice`` on an index array is
    ``integers(0, n)``, so the two ``integers`` calls below consume the bit
    stream in exactly the study's order (verified: every one of the 72
    intervals of ``alignment_controlled_curves.csv`` is reproduced to 1e-16).
    """
    grouped = {int(root): g[value].to_numpy(float) for root, g in frame.groupby("root_id")}
    roots = np.asarray(sorted(grouped), dtype=np.int64)
    matrix = np.stack([grouped[int(r)] for r in roots])
    assert matrix.shape == (N_CELLS, N_STREAMS)
    draws = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        selected = rng.integers(0, len(roots), size=len(roots), dtype=np.int64)
        streams = rng.integers(0, N_STREAMS, size=len(roots), dtype=np.int64)
        draws[i] = matrix[selected, streams].mean()
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def phase_summary_check(seed_frame, summary, groups, metrics, base, *, bootstrap_metrics=()):
    """Every summary mean (and, for ``bootstrap_metrics``, every interval)
    recomputed from the seed rows with the study's own seeds."""
    err = 0.0
    for gi, (keys, part) in enumerate(seed_frame.groupby(groups, sort=True)):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = summary
        for g, k in zip(groups, keys):
            row = row[row[g].eq(k)]
        assert len(row) == 1, keys
        row = row.iloc[0]
        assert int(row.n_seeds) == part.seed.nunique() == N_SEEDS, keys
        for mi, metric in enumerate(metrics):
            vals = part[metric].to_numpy(float)
            err = max(err, abs(vals.mean() - row[f"mean_{metric}"]) / max(abs(row[f"mean_{metric}"]), 1e-9))
            if metric in bootstrap_metrics:
                m, lo, hi = phase_bootstrap(vals, base + 100 * gi + mi)
                for a, b in ((m, row[f"mean_{metric}"]), (lo, row[f"ci95_low_{metric}"]),
                             (hi, row[f"ci95_high_{metric}"])):
                    err = max(err, abs(a - b) / max(abs(b), 1e-9))
    assert err < 5e-9, err            # the summaries are printed at %.10g
    return err


def csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def fan(ax, x, values, color, *, half=0.12, ms=SEED_MS * 0.6, alpha=SEED_ALPHA * 0.7,
        zorder=2.0, seed=None):
    """Per-seed (or per-cell) values as a jittered fan behind the mean; the
    jitter is the table order, or a fixed permutation of it (presentation
    only) so sorted tables do not line up as a ramp."""
    values = np.asarray(values, float)
    jitter = np.linspace(-half, half, len(values)) if len(values) > 1 else np.zeros(1)
    if seed is not None:
        jitter = jitter[np.random.default_rng(seed).permutation(len(values))]
    ax.plot(x + jitter, values, linestyle="none", marker="o", markersize=ms,
            markerfacecolor=color, markeredgecolor="none", alpha=alpha, zorder=zorder)


def mean_marker(ax, x, y, color, marker, *, ms=MARKER_MS * 0.82, zorder=4.0, hollow=False):
    ax.plot([x], [y], linestyle="none", marker=marker, markersize=ms,
            markerfacecolor=("white" if hollow else color),
            markeredgecolor=(color if hollow else "white"),
            markeredgewidth=(LW_EDGE if hollow else LW_HAIR), zorder=zorder)


# ── B: subtree-minus-random spectral capture ─────────────────────────────
def panel_spectral(ax, spectral, spectral_seed):
    phase_summary_check(spectral_seed, spectral, ["alignment", "budget_k", "method"],
                        ["spectral_capture", "morphology_regret"], 2_700_000)
    wide = spectral.pivot_table(index=["alignment", "budget_k"], columns="method",
                                values="mean_spectral_capture")
    advantage = (wide.ancestry - wide.random_rank).unstack("budget_k")
    # the same matrix from the paired per-seed rows
    sw = spectral_seed.pivot_table(index=["seed", "alignment", "budget_k"], columns="method",
                                   values="spectral_capture")
    diff = (sw.ancestry - sw.random_rank).groupby(level=["alignment", "budget_k"])
    assert diff.size().eq(N_SEEDS).all()
    np.testing.assert_allclose(diff.mean().unstack("budget_k").to_numpy(), advantage.to_numpy(),
                               rtol=0, atol=3e-9)
    assert list(advantage.columns) == [1, 2, 4, 8, 16]
    np.testing.assert_allclose(advantage.index.to_numpy(), [0.0, 0.25, 0.5, 0.75, 1.0])
    matrix = advantage.to_numpy()
    # the two by-construction controls: K = 16 is full rank (no advantage),
    # rho = 0 is isotropic (nothing to capture)
    assert np.abs(matrix[:, -1]).max() < 5e-3 and (matrix[0, :] <= 0.0).all()
    fmt = _MinusFmt(2)
    printed = [[fmt.format(v) for v in row] for row in matrix]
    assert printed[4][2] == "0.55" and printed[4][1] == "0.38" and printed[0][0] == "−0.01"
    assert all(s == "0.00" for s in (r[-1] for r in printed))
    signed_heatmap(ax, matrix, [str(v) for v in advantage.columns],
                   [f"{v:.2f}" for v in advantage.index])
    ax.set_xlabel("Route budget K", labelpad=2.0)
    ax.set_ylabel("Covariance mixture ρ", labelpad=1.5)
    print("[B] subtree − random spectral capture, rows rho 0..1, cols K 1..16:")
    for lab, row in zip(advantage.index, printed):
        print(f"      rho {lab:.2f}: " + "  ".join(row))
    return matrix


# ── C: final loss versus route resolution, one series per hierarchy ──────
def panel_depth(ax, depth, depth_seed):
    phase_summary_check(depth_seed, depth, ["task_depth", "model_depth", "method"],
                        ["final_population_loss", "target_signal_capture", "admitted_noise"],
                        2_710_000, bootstrap_metrics=("final_population_loss",))
    aligned = depth[depth.method.eq("aligned_tree")]
    seeds = depth_seed[depth_seed.method.eq("aligned_tree")]
    levels = sorted(aligned.task_depth.unique())
    assert levels == [1, 2, 3, 4] and sorted(aligned.model_depth.unique()) == [1, 2, 3, 4]
    dodge = np.linspace(-0.12, 0.12, len(levels))
    markers = ("o", "s", "^", "D")
    handles = []
    lo_all, hi_all = np.inf, -np.inf
    printed = []
    for dx, level, color, marker in zip(dodge, levels, ORDINAL_RAMP, markers):
        part = aligned[aligned.task_depth.eq(level)].sort_values("model_depth")
        xs = part.model_depth.to_numpy(float)
        ys = part.mean_final_population_loss.to_numpy(float)
        los = part.ci95_low_final_population_loss.to_numpy(float)
        his = part.ci95_high_final_population_loss.to_numpy(float)
        for x, y in zip(xs, ys):
            vals = seeds[seeds.task_depth.eq(level) & seeds.model_depth.eq(x)]
            vals = vals.sort_values("seed").final_population_loss.to_numpy(float)
            assert len(vals) == N_SEEDS and (vals > 0).all()
            np.testing.assert_allclose(vals.mean(), y, rtol=3e-9, atol=0)
            lo_all, hi_all = min(lo_all, vals.min()), max(hi_all, vals.max())
            fan(ax, x + dx, vals, color, half=0.045, seed=int(1000 * level + x))
        ax.plot(xs + dx, ys, color=color, lw=LW_DATA, zorder=3.0)
        for x, y, lo, hi in zip(xs, ys, los, his):
            ax.plot([x + dx, x + dx], [lo, hi], color=color, lw=LW_ERR, zorder=3.4,
                    solid_capstyle="butt")
            mean_marker(ax, x + dx, y, color, marker, zorder=4.0)
        handles.append(Line2D([], [], color=color, lw=LW_DATA, marker=marker,
                              ms=MARKER_MS * 0.82, markeredgecolor="white",
                              markeredgewidth=LW_HAIR, label=f"hierarchy {level}"))
        printed.append(f"H{level}: " + ", ".join(f"{y:.4f} [{lo:.4f}, {hi:.4f}]"
                                                 for y, lo, hi in zip(ys, los, his)))
    # the interior optimum: each hierarchy's loss is smallest at its own resolution
    means = aligned.pivot(index="task_depth", columns="model_depth", values="mean_final_population_loss")
    assert all(int(means.loc[h].idxmin()) == h for h in levels)
    print("[C] " + "; ".join(printed) + f"; seed range {lo_all:.2e}-{hi_all:.2f}")
    ax.set_yscale("log")
    ax.set_ylim(lo_all / 1.6, hi_all * 1.6)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1.0, 10.0], ["0.001", "0.01", "0.1", "1", "10"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xlim(0.6, 4.4)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Route resolution")
    ax.set_ylabel("Final loss")
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False, fontsize=PT_BASE,
              handlelength=1.6, handletextpad=0.5, labelspacing=0.25, columnspacing=1.0,
              borderaxespad=0.2, borderpad=0.0)


# ── D: the one-step projection boundary on numeric axes ─────────────────
def cell_edges(samples):
    s = np.asarray(samples, float)
    mid = 0.5 * (s[1:] + s[:-1])
    return np.concatenate([[s[0] - (s[1] - s[0]) / 2.0], mid, [s[-1] + (s[-1] - s[-2]) / 2.0]])


def panel_projection(ax, projection, projection_seed):
    phase_summary_check(projection_seed, projection, ["signal_retention", "noise_retention", "method"],
                        ["population_loss_after_one_step", "expected_population_loss"], 2_720_000)
    wide = projection.pivot_table(index=["signal_retention", "noise_retention"], columns="method",
                                  values="mean_expected_population_loss")
    delta = (wide.bp_plus_route_projection - wide.full_stochastic_bp).unstack("signal_retention")
    xs = delta.columns.to_numpy(float)
    ys = delta.index.to_numpy(float)
    np.testing.assert_allclose(xs, [0.25, 0.5, 0.75, 0.9, 1.0])
    np.testing.assert_allclose(ys, [0.1, 0.25, 0.5, 0.75, 1.0])
    matrix = delta.to_numpy()
    # the exact identity the caption states, and its per-seed constancy
    np.testing.assert_allclose(matrix, (ys[:, None] - xs[None, :]) / 2.0, rtol=0, atol=2e-9)
    ps = projection_seed[projection_seed.method.isin(["bp_plus_route_projection", "full_stochastic_bp"])]
    assert ps.groupby(["signal_retention", "noise_retention", "method"]).expected_population_loss.nunique().eq(1).all()
    sw = ps.pivot_table(index=["seed", "signal_retention", "noise_retention"], columns="method",
                        values="expected_population_loss")
    per_seed = (sw.bp_plus_route_projection - sw.full_stochastic_bp).groupby(level=["signal_retention", "noise_retention"])
    assert per_seed.size().eq(N_SEEDS).all()
    np.testing.assert_allclose(per_seed.mean().unstack("signal_retention").to_numpy(), matrix, rtol=0, atol=2e-9)
    fmt = _MinusFmt(2)
    printed = [[fmt.format(v) for v in row] for row in matrix]
    assert printed[4][0] == "0.38" and printed[0][4] == "−0.45" and printed[4][2] == "0.13"
    assert all(printed[r][c] == "0.00" for r, c in ((1, 0), (2, 1), (3, 2), (4, 4)))  # f_noise = f_sig

    limit = float(np.max(np.abs(matrix)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    xe, ye = cell_edges(xs), cell_edges(ys)
    # a raster at >= 300 dpi whose pixels take their cell's exact colour, so
    # each cell spans its own sampled interval (widths follow the data)
    px_x = np.linspace(xe[0], xe[-1], 1400, endpoint=False) + (xe[-1] - xe[0]) / 2800.0
    px_y = np.linspace(ye[0], ye[-1], 900, endpoint=False) + (ye[-1] - ye[0]) / 1800.0
    ci = np.clip(np.searchsorted(xe, px_x, side="right") - 1, 0, len(xs) - 1)
    ri = np.clip(np.searchsorted(ye, px_y, side="right") - 1, 0, len(ys) - 1)
    rgba = DIV_CMAP(norm(matrix))[ri[:, None], ci[None, :]]
    ax.imshow(rgba, origin="lower", aspect="auto", interpolation="nearest",
              extent=(xe[0], xe[-1], ye[0], ye[-1]), zorder=1.0)
    for (r, c), val in np.ndenumerate(matrix):
        rgb = DIV_CMAP(norm(val))
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        ax.text(0.5 * (xe[c] + xe[c + 1]), 0.5 * (ye[r] + ye[r + 1]), printed[r][c],
                ha="center", va="center", fontsize=PT_BASE,
                color="white" if lum < 0.45 else INK, zorder=3.0)
    # cell boundaries (hairline) and the exact sign boundary f_noise = f_sig
    for x in xe[1:-1]:
        ax.plot([x, x], [ye[0], ye[-1]], color="white", lw=LW_HAIR, zorder=2.0)
    for y in ye[1:-1]:
        ax.plot([xe[0], xe[-1]], [y, y], color="white", lw=LW_HAIR, zorder=2.0)
    lo = max(xe[0], ye[0])
    hi = min(xe[-1], ye[-1])
    # the exact boundary f_noise = f_sig, a 45-degree line in data units; it
    # passes through the centre of every zero cell, so it is interrupted
    # around each printed "0.00" (the label sits on the line it belongs to)
    zeros = [float(v) for v in xs if any(np.isclose(v, ys))]
    assert zeros == [0.25, 0.5, 0.75, 1.0]
    gap = 0.065
    cuts = [lo] + [c + s * gap for c in zeros for s in (-1.0, 1.0)] + [hi]
    for a, b in zip(cuts[0::2], cuts[1::2]):
        if b > a:
            ax.plot([a, b], [a, b], color=INK, lw=LW_REF, dashes=(2.6, 1.8), zorder=4.0,
                    solid_capstyle="butt")
    ax.plot([xe[0], xe[-1], xe[-1], xe[0], xe[0]], [ye[0], ye[0], ye[-1], ye[-1], ye[0]],
            color=COLORS["grid"], lw=LW_HAIR, solid_joinstyle="miter", zorder=4.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0.0)
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(ye[0], ye[-1])
    ax.set_xticks(xs, [f"{v:.2f}" for v in xs])
    ax.set_yticks(ys, [f"{v:.2f}" for v in ys])
    ax.set_xlabel("Retained signal fraction", labelpad=2.0)
    ax.set_ylabel("Retained noise fraction", labelpad=1.5)
    print("[D] (f_noise − f_sig)/2, rows f_noise 0.10..1.00, cols f_sig 0.25..1.00:")
    for lab, row in zip(ys, printed):
        print(f"      f_noise {lab:.2f}: " + "  ".join(row))
    return matrix


# ── E: gradient cosine and one-step progress at 120 trained checkpoints ──
FAMILIES_E = (("global_scalar_available", "strict\nscalar", COLORS["scalar"]),
              ("ancestry_available", "per\nneuron", COLORS["per_soma"]))
METRICS_E = (("gradient_cosine", "gradient cosine"),
             ("norm_matched_fraction_of_exact", "one-step progress"))
FAMILY_INDEX = {"global_scalar_available": 0, "ancestry_available": 1, "exact_transport": 2}


def panel_checkpoints(ax, rows, feedback_summary):
    data = rows[np.isclose(rows.relative_step, 1e-5)]
    exact = data[data.feedback_family.eq("exact_transport")]
    assert len(exact) == N_CHECKPOINTS
    np.testing.assert_allclose(exact.gradient_cosine, 1.0, rtol=0, atol=1e-6)
    np.testing.assert_allclose(exact.norm_matched_fraction_of_exact, 1.0, rtol=0, atol=1e-4)
    slot_x = {(0, 0): 0.0, (0, 1): 1.0, (1, 0): 2.6, (1, 1): 3.6}
    ci_dx = 0.42
    y_lo, y_hi = -0.8, 1.28
    positions, clipped, printed = [], [], []
    rng = np.random.default_rng(700)
    for mi, (metric, metric_label) in enumerate(METRICS_E):
        for fi, (family, label, tone) in enumerate(FAMILIES_E):
            vals = data[data.feedback_family.eq(family)][metric].to_numpy(float)
            assert len(vals) == N_CHECKPOINTS, (metric, family, len(vals))
            mean, lo, hi = checkpoint_bootstrap(vals, seed=700 + mi * 10 + FAMILY_INDEX[family])
            fs = feedback_summary[feedback_summary.feedback_family.eq(family)]
            assert len(fs) == 1 and int(fs.n_checkpoints.iloc[0]) == N_CHECKPOINTS
            ref = fs.gradient_cosine_mean if metric == "gradient_cosine" else fs.one_step_progress_mean
            np.testing.assert_allclose(mean, float(ref.iloc[0]), rtol=0, atol=1e-12)
            x = slot_x[(mi, fi)]
            positions.append(x)
            jitter = rng.uniform(-0.16, 0.16, size=len(vals))
            inside = vals >= y_lo
            ax.plot(x + jitter[inside], vals[inside], ls="none", marker="o", ms=1.5,
                    mfc=tone, mec="none", alpha=0.45, zorder=2)
            if (~inside).any():
                low = np.sort(vals[~inside])
                ax.plot(x + jitter[~inside], np.full((~inside).sum(), y_lo), ls="none",
                        marker="v", ms=3.0, mfc="white", mec=tone, mew=LW_EDGE, zorder=3.5,
                        clip_on=False)
                clipped.append((x, low))
            ax.boxplot([vals], positions=[x], widths=0.5, patch_artist=True, showfliers=False,
                       whis=1.5, zorder=3,
                       boxprops={"facecolor": "none", "edgecolor": tone, "linewidth": LW_ERR},
                       medianprops={"color": tone, "linewidth": LW_ERR},
                       whiskerprops={"color": tone, "linewidth": LW_EDGE},
                       capprops={"color": tone, "linewidth": LW_EDGE})
            ax.errorbar(x + ci_dx, mean, yerr=[[mean - lo], [hi - mean]], fmt="D", mfc="white",
                        mec=INK, ecolor=INK, ms=2.6, mew=LW_EDGE, lw=LW_ERR, capsize=1.6, zorder=4)
            assert max(mean - lo, hi - mean) < 0.06     # narrower than the 2.6 pt marker
            q1, med, q3 = np.quantile(vals, [0.25, 0.5, 0.75])
            printed.append(f"{label.replace(chr(10), ' ')} {metric_label}: mean {mean:.4f} [{lo:.4f}, {hi:.4f}], "
                           f"median {med:.4f} IQR {q1:.4f}-{q3:.4f}, range {vals.min():.3f}-{vals.max():.3f}")
    assert len(clipped) == 1 and clipped[0][0] == 2.6 and len(clipped[0][1]) == 2
    np.testing.assert_allclose(clipped[0][1], [-1.4762118050624338, -1.3445762745738097], rtol=1e-12)
    for x, low in clipped:
        ax.text(x + 0.30, y_lo + 0.03, ", ".join(f"{v:.2f}".replace("-", "−") for v in low),
                ha="left", va="bottom", fontsize=PT_BASE, color=MUTE)
    print("[E] " + "; ".join(printed))
    ax.axhline(0, color=MUTE, ls="--", lw=LW_REF, zorder=1)
    # unity: the exact-path value of both quantities, by construction
    ax.plot([-0.45, 4.25], [1.0, 1.0], color=COLORS["bp"], lw=LW_REF, dashes=(2.6, 2.0), zorder=1)
    ax.set_xticks(positions, [lab for _, lab, _ in FAMILIES_E] * 2)
    ax.tick_params(axis="x", length=0.0, pad=2.0)
    ax.set_yticks([-0.5, 0.0, 0.5, 1.0], ["−0.5", "0", "0.5", "1.0"])
    ax.set_ylabel("Dimensionless value")
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.45, 4.25)
    ax.axvline(1.8 + 0.5 * ci_dx, color=COLORS["grid"], lw=LW_HAIR, zorder=0.5)
    for x, label in ((0.5 + 0.5 * ci_dx, "Gradient cosine"), (3.1 + 0.5 * ci_dx, "One-step progress")):
        ax.annotate(label, xy=(x, 0.0), xycoords=("data", "axes fraction"), xytext=(0.0, -21.0),
                    textcoords="offset points", ha="center", va="top", fontsize=PT_BASE, color=INK,
                    annotation_clip=False)
    return positions


# ── F: eight reconstructed arbors, twenty projected steps ────────────────
METHODS_F = (("morphology-selected paths", "morphology-selected", COLORS["shunting"], "o"),
             ("random paths", "random paths", COLORS["point_mlp"], "s"),
             ("depth bins", "depth bins", COLORS["additive"], "^"),
             ("ancestry-shuffled paths", "ancestry-shuffled", COLORS["highlight"], "D"))
METRIC_F = "iterative_progress"


def check_alignment_tables(runs, cells, curves, contrasts, summary):
    """Cell means from the 7,680 stream rows, curve means from the cell rows,
    every hierarchical interval from the stream rows with the study's rng."""
    assert len(runs) == 7680 and runs.root_id.nunique() == N_CELLS and runs.stream.nunique() == N_STREAMS
    assert int(summary["n_microns_cells"]) == N_CELLS and int(summary["monte_carlo_streams_per_cell"]) == N_STREAMS
    assert int(summary["quadratic_objective"]["iterative_steps"]) == 20
    assert sorted(runs.root_id.unique()) == sorted(int(v) for v in summary["cell_ids"])
    metrics = ("credit_capture", "one_step_progress", "iterative_progress")
    cell = (runs.groupby(["root_id", "alignment", "method"], as_index=False)
            .agg({m: "mean" for m in metrics}).sort_values(["alignment", "method", "root_id"]))
    ref = cells.sort_values(["alignment", "method", "root_id"]).reset_index(drop=True)
    assert len(cell) == len(ref) == 192
    for m in metrics:
        np.testing.assert_allclose(cell[m].to_numpy(), ref[m].to_numpy(), rtol=0, atol=1e-12)
    rng = np.random.default_rng(int(summary["provenance"]["base_random_seed"]) + 9_000_001)
    err = 0.0
    for (alignment, method), group in runs.groupby(["alignment", "method"], sort=True):
        row = curves[curves.alignment.eq(alignment) & curves.method.eq(method)]
        assert len(row) == 1 and int(row.n_cells.iloc[0]) == N_CELLS
        row = row.iloc[0]
        cg = ref[ref.alignment.eq(alignment) & ref.method.eq(method)]
        for m in metrics:
            np.testing.assert_allclose(cg[m].mean(), row[m], rtol=0, atol=1e-12)
            lo, hi = hierarchical_interval(group, m, rng)
            err = max(err, abs(lo - row[f"{m}_ci_low"]), abs(hi - row[f"{m}_ci_high"]))
    assert err < 1e-12, err
    # the paired contrasts are differences of the same cell rows
    wide = ref.pivot(index=["root_id", "alignment"], columns="method", values=METRIC_F)
    for control in ("random paths", "depth bins", "ancestry-shuffled paths"):
        d = (wide["morphology-selected paths"] - wide[control]).rename("d").reset_index()
        c = contrasts[contrasts.control.eq(control)].merge(d, on=["root_id", "alignment"])
        assert len(c) == 48
        np.testing.assert_allclose(c[f"morphology_minus_control_{METRIC_F}"], c.d, rtol=0, atol=1e-12)
    return err


def panel_arbors(ax, cells, curves):
    handles = []
    dodge = (-2.4, -0.8, 0.8, 2.4)      # in alignment percent
    printed = []
    lo_all, hi_all = np.inf, -np.inf
    for dx, (key, label, color, marker) in zip(dodge, METHODS_F):
        part = curves[curves.method.eq(key)].sort_values("alignment")
        xs = 100.0 * part.alignment.to_numpy(float)
        ys = part[METRIC_F].to_numpy(float)
        los = part[f"{METRIC_F}_ci_low"].to_numpy(float)
        his = part[f"{METRIC_F}_ci_high"].to_numpy(float)
        np.testing.assert_allclose(xs, [0, 20, 40, 60, 80, 100])
        for x, y in zip(xs, ys):
            vals = cells[cells.method.eq(key) & np.isclose(cells.alignment, x / 100.0)]
            vals = vals.sort_values("root_id")[METRIC_F].to_numpy(float)
            assert len(vals) == N_CELLS
            np.testing.assert_allclose(vals.mean(), y, rtol=0, atol=1e-12)
            lo_all, hi_all = min(lo_all, vals.min()), max(hi_all, vals.max())
            fan(ax, x + dx, vals, color, half=0.7, seed=int(x) + len(key))
        ax.plot(xs + dx, ys, color=color, lw=LW_DATA, zorder=3.0)
        for x, y, lo, hi in zip(xs, ys, los, his):
            ax.plot([x + dx, x + dx], [lo, hi], color=color, lw=LW_ERR, zorder=3.4,
                    solid_capstyle="butt")
            mean_marker(ax, x + dx, y, color, marker, zorder=4.0)
        handles.append(Line2D([], [], color=color, lw=LW_DATA, marker=marker, ms=MARKER_MS * 0.82,
                              markeredgecolor="white", markeredgewidth=LW_HAIR, label=label))
        printed.append(f"{key}: " + ", ".join(f"{y:.3f} [{lo:.3f}, {hi:.3f}]" for y, lo, hi in zip(ys, los, his)))
    morph = curves[curves.method.eq("morphology-selected paths")].sort_values("alignment")[METRIC_F].to_numpy()
    assert abs(morph[0]) < 1e-15 and 0.96 < morph[-1] < 0.98
    print("[F] " + "; ".join(printed) + f"; cell range {lo_all:.2e}-{hi_all:.3f}")
    ax.set_xlim(-6.0, 106.0)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_ylim(-0.03, 1.09)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlabel("Credit aligned to routes (%)")
    ax.set_ylabel("Relative 20-step progress")
    # the full-gradient sequence is 1 by definition (the caption names the rule)
    ax.plot([-6.0, 106.0], [1.0, 1.0], color=MUTE, lw=LW_REF, dashes=(2.6, 2.0), zorder=1.0)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.01, 0.905), ncol=1,
              frameon=False, fontsize=PT_BASE, handlelength=1.6, handletextpad=0.5,
              labelspacing=0.25, borderaxespad=0.0, borderpad=0.0)


# ── G: the analytic rank-noise trade-off ─────────────────────────────────
CURVES_G = ((1, 0.8, "-", "K = 1, q = 0.8"), (2, 1.0, DASHED, "K = 2, q = 1"))


def panel_bound(ax):
    sigma2 = np.linspace(0.0, 2.0, 400)
    values = []
    for rank, q, ls, label in CURVES_G:
        y = q ** 2 / (q + rank * sigma2)
        values.append(y)
        ax.plot(sigma2, y, color=INK, lw=LW_DATA, ls=ls, label=label, zorder=3)
    cross = 4.0 / 7.0
    y_cross = 0.8 ** 2 / (0.8 + cross)
    np.testing.assert_allclose(y_cross, 1.0 / (1.0 + 2.0 * cross), rtol=0, atol=1e-12)
    lo, hi = values
    assert (lo[sigma2 < cross] < hi[sigma2 < cross]).all() and (lo[sigma2 > cross] > hi[sigma2 > cross]).all()
    ax.fill_between(sigma2, lo, hi, color=COLORS["grid"], alpha=0.85, lw=0, zorder=1)
    print(f"[G] analytic q²/(q + Kσ²); curves cross at σ² = 4/7 = {cross:.4f}, value {y_cross:.4f}")
    ax.set_xlabel("Noise variance σ² (arbitrary units)")
    # 2026-09-23: the caption's own wording; the 110 pt "... one-step bound"
    # outran the 95 pt axes and rose above G's letter
    ax.set_ylabel("2L × optimized bound")
    ax.set_xlim(0, 2.0)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0], ["0", "0.5", "1.0", "1.5", "2.0"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.50", "0.75", "1.00"])
    ax.legend(frameon=False, fontsize=PT_BASE, loc="upper right", handlelength=2.2,
              borderpad=0.1, labelspacing=0.25, borderaxespad=0.2)


# ── the canvas ───────────────────────────────────────────────────────────
# 2026-09-23: with the titles and tags gone the canvas keeps every axes box
# (A/B 104 pt, C/D 120 pt, E-G 95.2 pt tall) and drops only the title bands:
# the top margin and the vertical gutter now hold the letter band (13 pt) plus
# the lock's 8 pt pad, and row 2's weight carries the 2.2 pt the lock still
# carves above it.  One more point of left margin lets column 0 take the same
# 6 pt declared reserve as columns 4, 6 and 8, so equal spans have equal
# axes widths (the 0.9 pt row-alignment spread is gone).
CANVAS_H_PT = 458.4
ROW_PT = [104.0, 120.0, 97.4]
HGUTTER_PT = 30.0
VGUTTER_PT = 40.0
MARGINS = Margins(left=41.0, right=10.0, top=21.0, bottom=36.0)


def build(path: Path = OUT):
    spectral = csv(PHASE / "spectral_phase_summary.csv")
    spectral_seed = csv(PHASE / "spectral_phase_seed.csv")
    depth = csv(PHASE / "depth_training_summary.csv")
    depth_seed = csv(PHASE / "depth_training_seed.csv")
    projection = csv(PHASE / "projection_phase_summary.csv")
    projection_seed = csv(PHASE / "projection_phase_seed.csv")
    rows = csv(VALIDITY / "mechanism_checkpoint_rows_valid.csv")
    feedback_summary = csv(VALIDITY / "mechanism_feedback_summary_valid.csv")
    runs = pd.read_csv(ALIGNED / "alignment_controlled_runs.csv.gz")
    cells = csv(ALIGNED / "cell_alignment_metrics.csv")
    curves = csv(ALIGNED / "alignment_controlled_curves.csv")
    contrasts = csv(ALIGNED / "cell_paired_contrasts.csv")
    summary = json.loads((ALIGNED / "summary.json").read_text())
    print(f"[F tables] hierarchical intervals recomputed from the stream rows, max |error| "
          f"{check_alignment_tables(runs, cells, curves, contrasts, summary):.1e}")

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 3, row_weights=ROW_PT, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS, letter_clearance=True)
    # 2026-09-23 clarity pass: no panel titles or method tags on the artwork;
    # what they said is in the caption (ledger si_pass/ledger/utility_signal_noise.md)
    ax_a = cv.panel("A", 0, 0, 6, schematic=True)
    ax_b = cv.panel("B", 0, 6, 6)
    ax_c = cv.panel("C", 1, 0, 6, grid="y")
    ax_d = cv.panel("D", 1, 6, 6)
    ax_e = cv.panel("E", 2, 0, 4, grid="y")
    ax_f = cv.panel("F", 2, 4, 4, grid="y")
    ax_g = cv.panel("G", 2, 8, 4, grid="y")
    for name in "ABCDEFG":
        cv.declare_reserve(name, left=6.0, right=6.0)

    panel_spectral(ax_b, spectral, spectral_seed)
    panel_depth(ax_c, depth, depth_seed)
    panel_projection(ax_d, projection, projection_seed)
    panel_checkpoints(ax_e, rows, feedback_summary)
    panel_arbors(ax_f, cells, curves)
    panel_bound(ax_g)
    cv.lock_reserves()              # settle the boxes before drawing A in points
    operator_schematic(ax_a)
    style_direct_color_labels(cv.fig)

    problems = cv.save(path, name="figure_utility_signal_noise_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

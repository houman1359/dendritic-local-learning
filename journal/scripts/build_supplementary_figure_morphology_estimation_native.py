#!/usr/bin/env python3
"""Supplementary sheet ``morphology_estimation`` (S36) rebuilt as one native canvas.

The frozen render (``figures/supplementary/curated/morphology_estimation.pdf``)
has no builder anywhere in the repository, so its verified in-panel defects
could not be fixed by the paste layer.  This builder reads ONLY the frozen
tables under ``source_data/morphology_calibration/`` and draws the same eight
panels with the same plotted quantities on the paper's
:class:`figure_canvas.NativeCanvas`:

* A  the sealed calibration protocol as a drawing: the 256-label bar split
     192 : 64 (``protocol.json``), the two selection branches (estimate and
     score versus fit pilots and gate) as parallel paths, and the sealed
     training step they merge into.
* B  the two prespecified pooled contrasts (``primary_contrasts.csv``) with
     the 80 per-task paired differences of ``paired_contrast_*.csv`` as a fan,
     the four family means as open marks, the pooled mean as a short rule
     with its Bonferroni 97.5 % whisker and a solid zero rule with a 0 tick.
     The 0.01 meaningful margin is 0.7 pt on this axis, less than the width
     of the zero rule, so it is stated in a corner note (with both lower
     bounds, which exceed it) rather than drawn as a second rule that would
     fuse with the first.  Fixed baseline left, pilot right (the caption's
     order, and H's); the row labels name the baselines, the fixed one by
     its table alias ``fixed / estimated-rank`` (``observed_rank_only`` is
     the same tree in every task, asserted).
* C  pooled menu regret against calibration size on a logarithmic label
     axis with the recorded 95 % seed-bootstrap bands
     (``policy_summary.csv``, recomputed from ``policy_outcomes.csv``).  The
     fixed / estimated-rank baseline does not depend on the calibration
     sample, so its two noise levels are one identical curve, drawn once.
* D  family-specific primary regret with the 20 per-task regrets of every
     cell drawn behind the mean rule and its 95 % whisker; the one interval
     narrower than the rule (quartic estimated cut) is stated on the panel.
* E  secondary adaptive construction, clean-test NMSE on a log axis tight to
     the data, 20 per-task values behind each mean, 95 % whiskers from
     ``figure_absolute_error_summary.csv`` (recomputed).  Both comparators
     are retrospective oracles (the best trained candidate of the menu, the
     true-target construction), named as such in the panel's own note.
* F  selection cost: the 80 per-task timings of
     ``calibration_selection_records.csv`` at the primary condition behind
     the median rule and interquartile whisker, on an axis from zero.
* G  the fresh Adam cohort (``end_to_end/summary.csv`` recomputed from
     ``end_to_end/endpoints.csv``): three structures x two credit rules in
     four families, 20 per-seed endpoints behind each mean, and the
     0.0225 noise floor drawn as a dashed rule.
* H  the three pooled paired contrasts of the fresh cohort
     (``end_to_end/contrasts.csv``, ``end_to_end/paired_contrasts.csv``): the
     20 pooled seed differences as a fan, the four family means as open
     marks, the pooled mean rule with its Bonferroni 97.5 % (rows 1-2) or
     pointwise 95 % (row 3) whisker, coverage printed in each category label.

Colour meanings on this sheet (one meaning per hue, keyed once at the top):
green = the estimated interactions / estimated tree; grey = a fixed tree of
the twelve-candidate menu (the development-best / estimated-rank baseline in
B-D, the best trained candidate in E, the development-fixed tree in G);
orange = the two-sweep pilot; purple = the target-informed (true-target)
construction.  Contrast rows in B take the colour of the baseline being
compared; H's three heterogeneous contrasts are drawn in neutral ink.
Marker fill in G names the credit rule (filled exact, open broadcast); line
style in C names the label noise (solid SD 0.5, dashed noiseless).  Every
printed or plotted number is asserted against the table it comes from.
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
    COLORS, ERR_CAPSIZE, LW_DATA, LW_ERR, LW_HAIR, LW_REF, MARKER_MS, PT_BASE,
    SEED_ALPHA, SEED_MS, Margins, NativeCanvas, tint_patch)
from native_schematics import Frame  # noqa: E402

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "morphology_calibration"
E2E = SOURCE / "end_to_end"
OUT = ROOT / "figures" / "supplementary" / "figure_morphology_estimation_native.pdf"

FAMILIES = ["matching", "quartet", "nested_prefix", "random_interactions"]
FAMILY_NAMES = ["matching", "quartic", "nested", "random"]
SIZES = [64, 256, 1024]
NOISES = [0.5, 0.0]
PRIMARY = (256, 0.5)
MARGIN = 0.01
N_SEEDS = 20

EST = COLORS["shunting"]      # estimated interactions / estimated tree
FIXED = COLORS["point_mlp"]   # a fixed tree of the twelve-candidate menu
PILOT = COLORS["local"]       # the two-sweep pilot
TARGET = COLORS["oracle"]     # target-informed (true-target) construction
MUTE = COLORS["mute"]
INK = COLORS["ink"]

MEAN_HALF_PT = 4.0            # half-width of a mean rule, in points
BAND_ALPHA = 0.16
DOTTED = (1.0, 1.8)
DASHED = (2.6, 2.0)
LINE_DASH = (3.2, 1.8)


# ── the study's bootstraps, verbatim ─────────────────────────────────────
def boot_run(values, confidence=0.95):
    """``scripts/morphology_calibration/run.py:bootstrap`` (rng 650905)."""
    values = np.asarray(values, float)
    g = np.random.default_rng(650905)
    means = values[g.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    q = (1.0 - confidence) / 2.0
    return float(np.quantile(means, q)), float(np.quantile(means, 1.0 - q))


def boot_tables(values):
    """``scripts/build_morphology_credit_figure_tables.py:bootstrap`` (rng 202609052)."""
    x = np.asarray(values, float)
    rng = np.random.default_rng(202609052)
    means = x[rng.integers(len(x), size=(10000, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def csv(name, base=SOURCE):
    path = base / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# ── shared glyphs ─────────────────────────────────────────────────────────
def fan(ax, x, values, color, *, half=0.16, ms=SEED_MS, alpha=SEED_ALPHA, zorder=2.0):
    """Every per-task / per-seed value as a jittered fan behind the mean."""
    values = np.asarray(values, float)
    jitter = np.linspace(-half, half, len(values)) if len(values) > 1 else np.zeros(1)
    ax.plot(x + jitter, values, linestyle="none", marker="o", markersize=ms,
            markerfacecolor=color, markeredgecolor="none", alpha=alpha, zorder=zorder)


def mean_rule(ax, x, mean, lo, hi, color, *, zorder=4.0):
    """Mean as a short horizontal rule (drawn in points, so it is the same
    width on a linear and a logarithmic axis) with a capped whisker."""
    ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                capthick=LW_ERR, zorder=zorder - 0.5)
    for sign in (-1.0, 1.0):
        ax.annotate("", xy=(x, mean), xycoords="data",
                    xytext=(sign * MEAN_HALF_PT, 0.0), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", color=color, lw=LW_DATA,
                                    shrinkA=0, shrinkB=0), zorder=zorder)


def open_marks(ax, x, values, color, *, half=0.10, zorder=3.0):
    """Family means as small open squares beside the pooled rule."""
    values = np.asarray(values, float)
    xs = x + np.linspace(-half, half, len(values))
    for xi, v in zip(xs, values):    # one artist per mark, so the live overlap
        ax.plot([xi], [v], linestyle="none", marker="s", markersize=MARKER_MS * 0.62,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=LW_ERR,
                zorder=zorder)       # audit sees each square, not the whole spread


def reference(ax, y, *, style="solid", color=MUTE, zorder=1.0):
    """A full-width reference rule between the fixed x limits."""
    x0, x1 = ax.get_xlim()
    kw = dict(color=color, lw=LW_REF, zorder=zorder, solid_capstyle="butt")
    if style == "dotted":
        kw["dashes"] = DOTTED
    elif style == "dashed":
        kw["dashes"] = DASHED
    ax.plot([x0, x1], [y, y], **kw)


def note(ax, text, *, corner="tl", color=INK, dx=3.0, dy=-2.5):
    """In-panel note in a free corner, offset a few points from the frame."""
    xa, ha = (0.0, "left") if corner[1] == "l" else (1.0, "right")
    ya, va = (1.0, "top") if corner[0] == "t" else (0.0, "bottom")
    if corner[1] == "r":
        dx = -dx
    if corner[0] == "b":
        dy = -dy
    return ax.annotate(text, xy=(xa, ya), xycoords="axes fraction", xytext=(dx, dy),
                       textcoords="offset points", ha=ha, va=va, fontsize=PT_BASE,
                       color=color, linespacing=1.15, zorder=6)


def note_lines(ax, lines, *, corner="tr", color=INK, dx=3.0, dy=-2.5):
    """A top-corner note drawn one line per artist.

    Both overlap audits (the live one in ``save`` and the compiled-page one)
    then see each line's own box, so a ragged block can stand beside a
    column of data where the rectangle spanning its widest line could not.
    The line pitch is measured from a two-line probe at the note's own
    ``linespacing``, so the block lays out exactly as one :func:`note`.
    """
    assert corner[0] == "t", corner
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    heights = []
    for probe_text in ("lp", "lp\nlp"):
        probe = note(ax, probe_text, corner=corner, color=color, dx=dx, dy=dy)
        heights.append(probe.get_window_extent(renderer).height * 72.0 / fig.dpi)
        probe.remove()
    pitch = heights[1] - heights[0]
    assert 0.9 * PT_BASE < pitch < 1.4 * PT_BASE, pitch
    return [note(ax, line, corner=corner, color=color, dx=dx, dy=dy - i * pitch)
            for i, line in enumerate(lines)]


# ── A: the sealed protocol ───────────────────────────────────────────────
def panel_a(ax, protocol, e2e_protocol):
    # the primary condition and the 75 : 25 split come from the calibration
    # protocol; the end-to-end protocol records the same split as row counts
    assert protocol["primary_condition"] == {"calibration_noise_sd": PRIMARY[1],
                                             "calibration_rows": PRIMARY[0]}
    assert protocol["calibration_split"].startswith("First75% fit, remaining25%")
    n_rows = int(e2e_protocol["calibration_rows"])
    n_fit = int(e2e_protocol["calibration_fit_rows"])
    n_gate = n_rows - n_fit
    assert (n_rows, n_fit, n_gate) == (PRIMARY[0], int(0.75 * PRIMARY[0]), int(0.25 * PRIMARY[0]))
    assert (n_rows, n_fit, n_gate) == (256, 192, 64)
    assert float(e2e_protocol["calibration_noise_sd"]) == PRIMARY[1]
    assert protocol["pilot"].startswith("2ALS sweeps")
    pilot_sweeps = 2
    n_cand = len(protocol["candidates"])
    assert protocol["estimator"].startswith("Lasso all255 Walsh monomials")
    n_terms = 255
    assert (pilot_sweeps, n_cand, n_terms) == (2, 12, 255)
    assert float(protocol["meaningful_margin_nmse"]) == MARGIN

    f = Frame(ax)
    W, H = f.w_pt, f.h_pt
    X, Y = f.fx, f.fy

    def box(x0, y0, w, h, color, pct, text, *, tcolor=INK):
        tint_patch(ax, ("rect", X(x0), Y(y0), X(w), Y(h)), color=color, pct=pct,
                   edge=True, radius_pt=2.0, zorder=0.6, clip_on=False)
        f.text((X(x0 + w / 2.0), Y(y0 + h / 2.0)), text, size=PT_BASE, color=tcolor,
               linespacing=1.1)

    # geometry in points from the bottom-left of the axes box
    head_h = 8.5                 # the "256 labels" line
    bar_h = 10.0
    gap = 5.0
    box1_h = 17.0
    box2_h = 11.0
    final_h = 11.0
    used = head_h + bar_h + gap + box1_h + gap + box2_h + gap + final_h
    assert used <= H, (used, H)
    slack = (H - used) / 3.0
    gap += slack
    y_final = 0.0
    y_box2 = y_final + final_h + gap
    y_box1 = y_box2 + box2_h + gap
    y_bar = y_box1 + box1_h + gap
    y_head = y_bar + bar_h + head_h / 2.0
    f.text((X(W / 2.0), Y(y_head)), f"{n_rows} noisy labels (noise SD {PRIMARY[1]:g})",
           size=PT_BASE, color=INK)
    # the label bar, split in proportion 192 : 64
    fit_w = W * n_fit / n_rows
    box(0.0, y_bar, fit_w, bar_h, "mute", 12, f"{n_fit} labels: fit the estimator or the pilots")
    box(fit_w, y_bar, W - fit_w, bar_h, "mute", 30, f"{n_gate}: gate")
    # two branches
    col_gap = 9.0
    col_w = (W - col_gap) / 2.0
    xl, xr = 0.0, col_w + col_gap
    rail = 8.0                   # the 64-label rail down the right edge
    box(xl, y_box1, col_w, box1_h, "shunting", 16,
        f"estimate interactions\n(Lasso, {n_terms} Walsh terms)")
    box(xl, y_box2, col_w, box2_h, "shunting", 16, f"score the {n_cand} candidate cuts")
    box(xr, y_box1, col_w - rail, box1_h, "local", 16,
        f"fit the {n_cand} pilots\n({pilot_sweeps} sweeps each)")
    box(xr, y_box2, col_w - rail, box2_h, "local", 16, f"gate on the {n_gate} held-out")
    box(0.0, y_final, W, final_h, "mute", 12, "seal choices; reset; train every candidate afresh")
    # arrows: 192 -> both branches, 64 -> the gate (down the rail), both -> seal
    P = lambda x, y: (X(x), Y(y))  # noqa: E731
    head = 3.2
    f.arrow(P(xl + col_w / 2.0, y_bar), P(xl + col_w / 2.0, y_box1 + box1_h), head=head)
    x_r = xr + (col_w - rail) / 2.0
    f.arrow(P(x_r, y_bar), P(x_r, y_box1 + box1_h), head=head)
    f.arrow(P(xl + col_w / 2.0, y_box1), P(xl + col_w / 2.0, y_box2 + box2_h), head=head)
    f.arrow(P(x_r, y_box1), P(x_r, y_box2 + box2_h), head=head)
    x_rail = W - rail / 2.0
    y_gate = y_box2 + box2_h / 2.0
    assert x_rail > fit_w + 2.0          # the rail starts under the 64 segment
    f.leader(P(x_rail, y_bar), P(x_rail, y_gate), color=MUTE, lw=LW_REF)
    f.arrow(P(x_rail, y_gate), P(xr + col_w - rail, y_gate), head=head)
    f.arrow(P(xl + col_w / 2.0, y_box2), P(xl + col_w / 2.0, y_final + final_h), head=head)
    f.arrow(P(x_r, y_box2), P(x_r, y_final + final_h), head=head)
    f.require_delta0(allow_no_delta0=True, reason="protocol flow chart, no neuron drawn")
    print(f"[A] {n_rows} labels = {n_fit} fit + {n_gate} gate; {n_cand} candidates; "
          f"{n_terms} Walsh terms; {pilot_sweeps} pilot sweeps")


# ── B: the two prespecified pooled contrasts ─────────────────────────────
B_ROWS = [("development_best_fixed", "fixed / estimated-rank", FIXED),
          ("two_sweep_pilot", "two-sweep pilot", PILOT)]
B_YLIM = (-0.33, 0.78)
B_ALIAS = "observed_rank_only"     # the table row the alias in the label names


def panel_b(ax, primary, outcomes, summary):
    assert len(primary) == 4
    prim = outcomes[outcomes.calibration_rows.eq(PRIMARY[0])
                    & outcomes.calibration_noise_sd.eq(PRIMARY[1])]
    est = prim[prim.policy.eq("estimated_cut")].set_index(["seed", "family"]).regret
    # the alias "fixed / estimated-rank": the estimated-rank policy picks the
    # development-best tree in every task and condition, so its rows of the
    # outcome and contrast tables are the fixed baseline's rows
    keys = ["seed", "family", "calibration_rows", "calibration_noise_sd"]
    fixed_all = outcomes[outcomes.policy.eq("development_best_fixed")].set_index(keys).sort_index()
    alias_all = outcomes[outcomes.policy.eq(B_ALIAS)].set_index(keys).sort_index()
    assert len(fixed_all) == 480 and fixed_all.index.equals(alias_all.index)
    assert (alias_all.selected_candidate == fixed_all.selected_candidate).all()
    assert alias_all.selected_candidate.nunique() == 1
    np.testing.assert_allclose(alias_all.regret.to_numpy(), fixed_all.regret.to_numpy(), rtol=0, atol=0)
    alias_row = primary[primary.baseline.eq(B_ALIAS)].iloc[0]
    fixed_row = primary[primary.baseline.eq("development_best_fixed")].iloc[0]
    for col in ("mean_improvement", "ci95_low", "ci95_high", "ci975_low", "ci975_high"):
        assert float(alias_row[col]) == float(fixed_row[col]), col
    ax.set_xlim(-0.45, 2.0)          # room at the right for the criterion note
    ax.set_ylim(*B_YLIM)
    lower_bounds = []
    for x, (key, label, color) in enumerate(B_ROWS):
        r = primary[primary.baseline.eq(key)].iloc[0]
        assert bool(r.primary_comparison) and bool(r.meaningful_superiority)
        assert int(r.seed_count) == N_SEEDS
        pairs = csv(f"paired_contrast_{key}.csv")
        assert len(pairs) == 80 and (pairs.baseline == key).all()
        assert pairs.seed.nunique() == N_SEEDS and pairs.family.nunique() == 4
        # the 80 task differences are exact subtractions of the primary regrets
        base = prim[prim.policy.eq(key)].set_index(["seed", "family"]).regret
        idx = pd.MultiIndex.from_frame(pairs[["seed", "family"]])
        np.testing.assert_allclose(pairs.baseline_minus_selector_regret.to_numpy(),
                                   (base.loc[idx] - est.loc[idx]).to_numpy(), rtol=0, atol=1e-12)
        tasks = pairs.baseline_minus_selector_regret.to_numpy(float)
        units = pairs.groupby("seed").baseline_minus_selector_regret.mean().sort_index()
        assert len(units) == N_SEEDS
        np.testing.assert_allclose(units.mean(), r.mean_improvement, rtol=0, atol=1e-12)
        lo95, hi95 = boot_run(units.to_numpy(), 0.95)
        lo, hi = boot_run(units.to_numpy(), 0.975)
        np.testing.assert_allclose([lo95, hi95, lo, hi],
                                   [r.ci95_low, r.ci95_high, r.ci975_low, r.ci975_high],
                                   rtol=0, atol=1e-9)
        assert lo > 0 and r.mean_improvement >= MARGIN
        assert lo > MARGIN               # the corner note's claim, per row
        lower_bounds.append(lo)
        # the family means are the differences of the family regret means
        fam = pairs.groupby("family").baseline_minus_selector_regret.mean()
        for family in FAMILIES:
            s = summary[summary.family.eq(family) & summary.calibration_rows.eq(PRIMARY[0])
                        & summary.calibration_noise_sd.eq(PRIMARY[1])].set_index("policy").mean_regret
            np.testing.assert_allclose(fam[family], s[key] - s["estimated_cut"], rtol=0, atol=1e-12)
        fam = fam.loc[FAMILIES].to_numpy(float)
        assert tasks.min() > B_YLIM[0] and tasks.max() < B_YLIM[1]
        assert fam.min() > B_YLIM[0] and fam.max() < B_YLIM[1]
        fan(ax, x, tasks, color, half=0.20, ms=SEED_MS * 0.9)
        open_marks(ax, x, fam, color, half=0.26)
        mean_rule(ax, x, float(r.mean_improvement), lo, hi, color)
        print(f"[B] {key}: mean {r.mean_improvement:.5f} [{lo:.5f}, {hi:.5f}] (97.5%); "
              f"tasks {tasks.min():.3f}-{tasks.max():.3f}, {(tasks == 0).sum()}/80 zero; "
              f"family means {np.array2string(fam, precision=3)}")
    reference(ax, 0.0, style="solid")
    # the 0.01 meaningful margin is narrower than the zero rule on this axis
    # (a second rule at 0.01 would fuse with it), so the criterion is stated:
    # the note stands right of the pilot column (x > 1.26 holds no datum),
    # its two longer lines above the column's highest task (0.669 at x 0.91)
    h_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
    margin_pt = MARGIN / (B_YLIM[1] - B_YLIM[0]) * h_pt
    assert margin_pt < LW_REF, (margin_pt, LW_REF)
    assert len(lower_bounds) == 2 and min(lower_bounds) > MARGIN
    ax.set_xticks([0, 1], [row[1] for row in B_ROWS])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([-0.25, 0.0, 0.25, 0.5, 0.75], ["−0.25", "0.00", "0.25", "0.50", "0.75"])
    ax.set_ylabel("baseline − estimated regret\n(mean; Bonferroni 97.5% CI)")
    note(ax, "dots: 80 tasks per column\nopen: family means (n = 4)", corner="tl")
    note_lines(ax, ["lower bounds", f"> {MARGIN:g} NMSE"], corner="tr")
    print(f"[B] margin {MARGIN:g} NMSE = {margin_pt:.2f} pt on the axis (< LW_REF {LW_REF} pt): "
          f"stated, not drawn; lower bounds {lower_bounds[0]:.3f}, {lower_bounds[1]:.3f}")


# ── C: calibration size and label noise ──────────────────────────────────
C_POLICIES = [("estimated_cut", EST), ("two_sweep_pilot", PILOT), ("development_best_fixed", FIXED)]
C_XLIM = (64 / 1.35, 1024 * 1.35)
C_YLIM = (0.0, 0.30)


def panel_c(ax, summary, outcomes):
    pooled = summary[summary.family.eq("all")]
    ax.set_xscale("log")
    ax.set_xlim(*C_XLIM)
    ax.set_ylim(*C_YLIM)
    curves = {}
    for key, color in C_POLICIES:
        for noise in NOISES:
            means, los, his = [], [], []
            for size in SIZES:
                r = pooled[pooled.policy.eq(key) & pooled.calibration_rows.eq(size)
                           & pooled.calibration_noise_sd.eq(noise)]
                assert len(r) == 1
                r = r.iloc[0]
                assert int(r.seed_count) == N_SEEDS
                z = outcomes[outcomes.policy.eq(key) & outcomes.calibration_rows.eq(size)
                             & outcomes.calibration_noise_sd.eq(noise)]
                assert len(z) == 80
                units = z.groupby("seed").regret.mean().sort_index()
                np.testing.assert_allclose(units.mean(), r.mean_regret, rtol=0, atol=1e-12)
                lo, hi = boot_run(units.to_numpy())
                np.testing.assert_allclose([lo, hi], [r.regret_ci_low, r.regret_ci_high],
                                           rtol=0, atol=1e-9)
                assert lo <= r.mean_regret <= hi
                means.append(float(r.mean_regret)); los.append(lo); his.append(hi)
            curves[(key, noise)] = (np.array(means), np.array(los), np.array(his))
    # the fixed baseline never sees the calibration sample: one curve
    for size in SIZES:
        z = outcomes[outcomes.policy.eq("development_best_fixed") & outcomes.calibration_rows.eq(size)]
        piv = z.pivot(index=["seed", "family"], columns="calibration_noise_sd", values="regret")
        np.testing.assert_allclose(piv[0.0].to_numpy(), piv[0.5].to_numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(curves[("development_best_fixed", 0.5)][0],
                               curves[("development_best_fixed", 0.0)][0], rtol=0, atol=0)
    top = 0.0
    for (key, noise), (m, lo, hi) in curves.items():
        color = dict(C_POLICIES)[key]
        if key == "development_best_fixed" and noise == 0.0:
            continue                     # identical to the SD 0.5 curve
        assert hi.max() < C_YLIM[1]
        top = max(top, hi.max())
        ax.fill_between(SIZES, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=1.5)
        solid = noise == 0.5
        ax.plot(SIZES, m, color=color, lw=LW_DATA, ls="-" if solid else "--",
                dashes=(None, None) if solid else LINE_DASH, marker="o", ms=MARKER_MS * 0.7,
                markerfacecolor=color if solid else "white", markeredgecolor=color,
                markeredgewidth=LW_ERR, zorder=3.0 if solid else 3.2)
        print(f"[C] {key} SD {noise:g}: regret {np.array2string(m, precision=4)} "
              f"bands {np.array2string(lo, precision=4)}-{np.array2string(hi, precision=4)}")
    ax.set_xticks(SIZES, [f"{s} labels" for s in SIZES])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0.0, 0.1, 0.2], ["0.0", "0.1", "0.2"])
    ax.set_ylabel("pooled menu regret (NMSE)")
    # every datum stays below the note band
    band_top = C_YLIM[1] - 26.0 / (ax.get_window_extent().height * 72.0 / ax.figure.dpi) * C_YLIM[1]
    assert top < band_top, (top, band_top)
    note(ax, "solid: noise SD 0.5; dashed: noiseless\n"
             f"bands: 95% seed bootstrap (n = {N_SEEDS} seeds)\n"
             "grey fixed baseline: one curve, both noise levels", corner="tr")


# ── D: family differences at the primary condition ───────────────────────
D_DODGE = {"estimated_cut": -0.26, "development_best_fixed": 0.0, "two_sweep_pilot": 0.26}
D_COLOR = {"estimated_cut": EST, "development_best_fixed": FIXED, "two_sweep_pilot": PILOT}
D_YLIM = (-0.03, 0.95)          # marks at 0.003 clear the spine


def panel_d(ax, summary, outcomes):
    prim = outcomes[outcomes.calibration_rows.eq(PRIMARY[0]) & outcomes.calibration_noise_sd.eq(PRIMARY[1])]
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(*D_YLIM)
    narrow = None
    for i, family in enumerate(FAMILIES):
        for key, dx in D_DODGE.items():
            r = summary[summary.family.eq(family) & summary.policy.eq(key)
                        & summary.calibration_rows.eq(PRIMARY[0])
                        & summary.calibration_noise_sd.eq(PRIMARY[1])]
            assert len(r) == 1
            r = r.iloc[0]
            z = prim[prim.policy.eq(key) & prim.family.eq(family)].sort_values("seed")
            assert len(z) == N_SEEDS and z.seed.is_unique
            seeds = z.regret.to_numpy(float)
            np.testing.assert_allclose(seeds.mean(), r.mean_regret, rtol=0, atol=1e-12)
            lo, hi = boot_run(seeds)
            np.testing.assert_allclose([lo, hi], [r.regret_ci_low, r.regret_ci_high],
                                       rtol=0, atol=1e-9)
            assert seeds.min() >= 0.0 > D_YLIM[0] and seeds.max() < D_YLIM[1]
            fan(ax, i + dx, seeds, D_COLOR[key], half=0.07, ms=SEED_MS * 0.85)
            mean_rule(ax, i + dx, float(r.mean_regret), lo, hi, D_COLOR[key])
            if family == "quartet" and key == "estimated_cut":
                narrow = (float(r.mean_regret), lo, hi)
            print(f"[D] {family}/{key}: regret {r.mean_regret:.4f} [{lo:.4f}, {hi:.4f}]; "
                  f"tasks {seeds.min():.3f}-{seeds.max():.3f}, {(seeds == 0).sum()}/20 zero")
    # the one interval narrower than the rule
    h_pt = ax.get_window_extent().height * 72.0 / ax.figure.dpi
    span_pt = (narrow[2] - narrow[1]) / (D_YLIM[1] - D_YLIM[0]) * h_pt
    assert span_pt < LW_DATA, span_pt
    np.testing.assert_allclose(narrow, [0.002692, 0.001373, 0.004586], rtol=0, atol=5e-7)
    ax.set_xticks(range(4), FAMILY_NAMES)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0.0, 0.3, 0.6, 0.9], ["0.0", "0.3", "0.6", "0.9"])
    ax.set_ylabel("menu regret (NMSE)")
    note(ax, f"dots: {N_SEEDS} tasks per mark; whisker 95%\n"
             f"green quartic CI {narrow[1]:.4f}–{narrow[2]:.4f}\n"
             "is narrower than the rule", corner="tl")


# ── E: secondary adaptive construction ───────────────────────────────────
E_SERIES = [("estimated_adaptive_dp", -0.27, EST), ("oracle_best_trained_menu", 0.0, FIXED),
            ("oracle_adaptive_dp", 0.27, TARGET)]
E_YLIM = (4.0e-4, 1.0)


def panel_e(ax, absolute, outcomes):
    prim = outcomes[outcomes.calibration_rows.eq(PRIMARY[0]) & outcomes.calibration_noise_sd.eq(PRIMARY[1])]
    ax.set_yscale("log")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(*E_YLIM)
    for i, family in enumerate(FAMILIES):
        for key, dx, color in E_SERIES:
            r = absolute[absolute.family.eq(family) & absolute.policy.eq(key)
                         & absolute.calibration_rows.eq(PRIMARY[0])
                         & absolute.calibration_noise_sd.eq(PRIMARY[1])]
            assert len(r) == 1
            r = r.iloc[0]
            assert int(r.n_seeds) == N_SEEDS
            z = prim[prim.policy.eq(key) & prim.family.eq(family)].sort_values("seed")
            assert len(z) == N_SEEDS and z.seed.is_unique
            seeds = z.test_nmse.to_numpy(float)
            m, lo, hi = boot_tables(seeds)
            np.testing.assert_allclose([m, lo, hi], [r.mean_test_nmse, r.ci95_low, r.ci95_high],
                                       rtol=0, atol=1e-9)
            assert seeds.min() > E_YLIM[0] and seeds.max() < E_YLIM[1]
            fan(ax, i + dx, seeds, color, half=0.07, ms=SEED_MS * 0.85)
            mean_rule(ax, i + dx, m, lo, hi, color)
            print(f"[E] {family}/{key}: NMSE {m:.5f} [{lo:.5f}, {hi:.5f}]; "
                  f"tasks {seeds.min():.5f}-{seeds.max():.4f}")
    ax.set_xticks(range(4), FAMILY_NAMES)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1.0], ["0.001", "0.01", "0.1", "1"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("clean-test NMSE")
    # the 0.003-0.05 band holds no datum in any of the four families: the
    # note (three lines, 22.8 pt of the band's 32.7 pt) sits in it, naming
    # both comparators as the retrospective oracles they are
    struct = prim[prim.family.isin(FAMILIES[:3]) & prim.policy.isin([s[0] for s in E_SERIES])].test_nmse
    assert not ((struct > 0.0026) & (struct < 0.055)).any()
    every = prim[prim.family.isin(FAMILIES) & prim.policy.isin([s[0] for s in E_SERIES])].test_nmse
    assert len(every) == 240 and not ((every > 0.0026) & (every < 0.055)).any()
    ax.text(-0.45, 0.012, f"dots: {N_SEEDS} tasks per mark; whisker: 95% seed bootstrap\n"
                          "grey: best trained of twelve (oracle)\n"
                          "purple: target-informed tree (oracle)",
            ha="left", va="center", fontsize=PT_BASE, color=INK, linespacing=1.15, zorder=6)


# ── F: selection cost ────────────────────────────────────────────────────
F_YLIM = (0.0, 0.082)


def panel_f(ax, records, timing):
    z = records[records.calibration_rows.eq(PRIMARY[0]) & records.calibration_noise_sd.eq(PRIMARY[1])]
    assert len(z) == 80 and z.seed.nunique() == N_SEEDS and z.family.nunique() == 4
    t = timing[timing.calibration_rows.eq(PRIMARY[0]) & timing.calibration_noise_sd.eq(PRIMARY[1])].iloc[0]
    for col in ("estimator_seconds", "score_seconds", "pilot_seconds"):
        np.testing.assert_allclose(z[col].median(), t[col], rtol=0, atol=1e-12)
    arms = [("estimate + score twelve", (z.estimator_seconds + z.score_seconds).to_numpy(float), EST),
            ("twelve two-sweep pilots", z.pilot_seconds.to_numpy(float), PILOT)]
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(*F_YLIM)
    medians = []
    for x, (label, values, color) in enumerate(arms):
        assert len(values) == 80 and values.min() > 0 and values.max() < F_YLIM[1]
        med = float(np.median(values))
        q1, q3 = (float(v) for v in np.percentile(values, [25, 75]))
        medians.append(med)
        fan(ax, x, values, color, half=0.22, ms=SEED_MS * 0.9)
        mean_rule(ax, x, med, q1, q3, color)
        print(f"[F] {label.replace(chr(10), ' ')}: median {med:.4f} s, IQR {q1:.4f}-{q3:.4f}, "
              f"range {values.min():.4f}-{values.max():.4f}")
    np.testing.assert_allclose(medians, [0.0464, 0.0661], rtol=0, atol=6e-5)
    print(f"[F] ratio of medians {medians[1] / medians[0]:.3f}")
    ax.set_xticks([0, 1], [a[0] for a in arms])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0.0, 0.02, 0.04, 0.06, 0.08], ["0.00", "0.02", "0.04", "0.06", "0.08"])
    ax.set_ylabel("elapsed time on one CPU (s)")
    note(ax, "dots: 80 tasks per arm\nrule: median; whisker: IQR", corner="tl")


# ── G: the fresh Adam cohort, all six conditions ─────────────────────────
G_STRUCT = [("estimated_dp", -0.30, EST), ("development_fixed", 0.0, FIXED), ("oracle_dp", 0.30, TARGET)]
G_RULE = [("exact", -0.075), ("broadcast", 0.075)]
G_YLIM = (-0.07, 1.85)          # marks at 0.024 and the floor clear the spine


def panel_g(ax, summary, endpoints, protocol):
    floor = float(protocol["test_noise_sd"]) ** 2
    np.testing.assert_allclose(floor, 0.0225, rtol=0, atol=1e-12)
    rates = protocol["rates"]
    assert rates == {"broadcast": 0.003, "exact": 0.01}
    assert len(endpoints) == 480 and (endpoints.step == 1024).all()
    assert (endpoints.status == "completed").all() and (endpoints.optimizer == "adam").all()
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(*G_YLIM)
    for i, family in enumerate(FAMILIES):
        for struct, dx, color in G_STRUCT:
            for rule, dr in G_RULE:
                r = summary[summary.family.eq(family) & summary.structure.eq(struct) & summary.rule.eq(rule)]
                assert len(r) == 1
                r = r.iloc[0]
                assert int(r.seed_count) == N_SEEDS
                z = endpoints[endpoints.family.eq(family) & endpoints.structure.eq(struct)
                              & endpoints.rule.eq(rule)].sort_values("seed")
                assert len(z) == N_SEEDS and z.seed.is_unique
                assert (z.rate == rates[rule]).all()
                seeds = z.test_nmse.to_numpy(float)
                np.testing.assert_allclose(seeds.mean(), r.mean_test_nmse, rtol=0, atol=1e-12)
                lo, hi = boot_run(seeds)
                np.testing.assert_allclose([lo, hi], [r.ci95_low, r.ci95_high], rtol=0, atol=1e-9)
                assert seeds.min() > 0.0 > G_YLIM[0] and seeds.max() < G_YLIM[1]
                x = i + dx + dr
                fan(ax, x, seeds, color, half=0.035, ms=SEED_MS * 0.7)
                exact = rule == "exact"
                ax.errorbar([x], [r.mean_test_nmse], yerr=[[r.mean_test_nmse - lo], [hi - r.mean_test_nmse]],
                            fmt="o", ms=MARKER_MS * 0.72, color=color,
                            markerfacecolor=color if exact else "white", markeredgecolor=color,
                            markeredgewidth=LW_ERR, ecolor=color, elinewidth=LW_ERR,
                            capsize=ERR_CAPSIZE * 0.7, capthick=LW_ERR, zorder=3.5, linestyle="none")
                print(f"[G] {family}/{struct}/{rule}: NMSE {r.mean_test_nmse:.4f} [{lo:.4f}, {hi:.4f}]; "
                      f"seeds {seeds.min():.4f}-{seeds.max():.4f}")
    reference(ax, floor, style="dashed", zorder=1.0)
    ax.set_xticks(range(4), FAMILY_NAMES)
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5], ["0.0", "0.5", "1.0", "1.5"])
    ax.set_ylabel(f"noisy-test NMSE\n({N_SEEDS} seeds per mark; 95% CI)")
    # the note band (top ~30 pt) holds only the one 1.76 outlier, at nested
    high = endpoints[endpoints.test_nmse > 1.1]
    assert len(high) == 1 and high.family.iloc[0] == "nested_prefix"
    note(ax, f"filled: exact credit ({rates['exact']:g})\n"
             f"open: root broadcast ({rates['broadcast']:g})\n"
             f"dashed: noise floor {floor:g}", corner="tl")


# ── H: pooled paired contrasts of the fresh cohort ───────────────────────
H_ROWS = [("fixed_exact_minus_estimated_exact", "fixed − estimated\n(exact credit)\n97.5% CI", 0.975),
          ("estimated_broadcast_minus_exact", "broadcast − exact\n(estimated tree)\n97.5% CI", 0.975),
          ("estimated_exact_minus_oracle_exact", "estimated −\ntarget-informed\n(exact) 95% CI", 0.95)]
H_YLIM = (-0.2, 1.40)


def panel_h(ax, contrasts, paired):
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(*H_YLIM)
    for x, (key, label, coverage) in enumerate(H_ROWS):
        r = contrasts[contrasts.family.eq("all") & contrasts.contrast.eq(key)].iloc[0]
        assert bool(r.primary) == (coverage == 0.975) and int(r.seed_count) == N_SEEDS
        z = paired[paired.family.eq("all") & paired.contrast.eq(key)].sort_values("seed")
        assert len(z) == N_SEEDS and z.seed.is_unique
        seeds = z.difference.to_numpy(float)
        # the pooled seed difference is the mean of the four family differences
        fam = paired[paired.family.ne("all") & paired.contrast.eq(key)]
        assert len(fam) == 80
        per_seed = fam.groupby("seed").difference.mean().loc[z.seed].to_numpy()
        np.testing.assert_allclose(seeds, per_seed, rtol=0, atol=1e-12)
        np.testing.assert_allclose(seeds.mean(), r.mean_difference, rtol=0, atol=1e-12)
        lo95, hi95 = boot_run(seeds, 0.95)
        lo975, hi975 = boot_run(seeds, 0.975)
        np.testing.assert_allclose([lo95, hi95, lo975, hi975],
                                   [r.ci95_low, r.ci95_high, r.ci975_low, r.ci975_high],
                                   rtol=0, atol=1e-9)
        lo, hi = (lo975, hi975) if coverage == 0.975 else (lo95, hi95)
        fam_means = []
        for family in FAMILIES:
            rf = contrasts[contrasts.family.eq(family) & contrasts.contrast.eq(key)].iloc[0]
            f_seeds = fam[fam.family.eq(family)].difference.to_numpy(float)
            assert len(f_seeds) == N_SEEDS
            np.testing.assert_allclose(f_seeds.mean(), rf.mean_difference, rtol=0, atol=1e-12)
            fam_means.append(float(rf.mean_difference))
        fam_means = np.array(fam_means)
        assert seeds.min() > H_YLIM[0] and seeds.max() < H_YLIM[1]
        assert fam_means.min() > H_YLIM[0] and fam_means.max() < H_YLIM[1]
        fan(ax, x, seeds, INK, half=0.16)
        open_marks(ax, x, fam_means, INK, half=0.28)
        mean_rule(ax, x, float(r.mean_difference), lo, hi, INK)
        print(f"[H] {key}: mean {r.mean_difference:.4f} [{lo:.4f}, {hi:.4f}] ({coverage:.3f}); "
              f"seeds {seeds.min():.3f}-{seeds.max():.3f}; family means "
              f"{np.array2string(fam_means, precision=3)}")
    # the matching sign reversal the caption states
    m = contrasts[contrasts.family.eq("matching") & contrasts.contrast.eq("estimated_broadcast_minus_exact")].iloc[0]
    assert m.mean_difference < 0 and m.ci95_high < 0
    reference(ax, 0.0, style="solid")
    ax.set_xticks(range(3), [row[1] for row in H_ROWS])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
    ax.set_ylabel("paired noisy-test NMSE\ndifference (mean; CI as labelled)")
    note(ax, f"dots: {N_SEEDS} seed means (4 families averaged)\nopen: family means (n = 4)",
         corner="tr")


# ── the sheet key ────────────────────────────────────────────────────────
def sheet_key(cv):
    handles = [
        Line2D([], [], linestyle="none", marker="o", ms=MARKER_MS * 0.8, color=c,
               markeredgecolor="none", label=label)
        for c, label in ((EST, "estimated interactions / tree"),
                         (FIXED, "fixed / estimated-rank tree (menu of twelve)"),
                         (PILOT, "two-sweep pilot"),
                         (TARGET, "target-informed tree (oracle)"))
    ]
    cv.fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0 - 3.0 / cv.height_pt),
                  ncol=len(handles), frameon=False, fontsize=PT_BASE, handlelength=1.2,
                  handletextpad=0.4, columnspacing=1.4, borderaxespad=0.0)


# ── the canvas ────────────────────────────────────────────────────────────
CANVAS_H_PT = 493.0           # the supplement's 540 pt cap and the audit's 1.05 aspect floor
HGUTTER_PT = 30.0
VGUTTER_PT = 36.0
MARGINS = Margins(left=40.0, right=8.0, top=34.0, bottom=38.0)
RESERVE = dict(left=32.0)     # one declared reserve on every panel: same width by construction


def build(path: Path = OUT, *, png=False):
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    e2e_protocol = json.loads((E2E / "protocol.json").read_text())
    summary = csv("policy_summary.csv")
    outcomes = csv("policy_outcomes.csv")
    primary = csv("primary_contrasts.csv")
    absolute = csv("figure_absolute_error_summary.csv")
    records = csv("calibration_selection_records.csv")
    timing = csv("calibration_timing.csv")
    e2e_summary = csv("summary.csv", E2E)
    endpoints = csv("endpoints.csv", E2E)
    contrasts = csv("contrasts.csv", E2E)
    paired = csv("paired_contrasts.csv", E2E)
    assert len(summary) == 310 and len(outcomes) == 4960 and len(absolute) == 310
    assert len(records) == 480 and len(timing) == 6

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 4, hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
                      margins=MARGINS)
    a = cv.panel("A", 0, 0, 6, schematic=True, title="Sealed calibration protocol")
    b = cv.panel("B", 0, 6, 6, grid="y", title="Two prespecified pooled comparisons")
    c = cv.panel("C", 1, 0, 6, grid="y", title="Regret by calibration labels and label noise")
    d = cv.panel("D", 1, 6, 6, grid="y", title="Primary condition: regret by family")
    e = cv.panel("E", 2, 0, 6, grid="y", title="Secondary adaptive construction (ALS fits)")
    f = cv.panel("F", 2, 6, 6, grid="y", title="Selection cost before final training")
    g = cv.panel("G", 3, 0, 6, grid="y", title="Fresh Adam cohort: all six conditions")
    h = cv.panel("H", 3, 6, 6, grid="y", title="Fresh Adam cohort: pooled paired contrasts")
    for name in "ABCDEFGH":
        cv.declare_reserve(name, **RESERVE)
    cv.lock_reserves()               # settle the boxes before drawing in points

    panel_a(a, protocol, e2e_protocol)
    panel_b(b, primary, outcomes, summary)
    panel_c(c, summary, outcomes)
    panel_d(d, summary, outcomes)
    panel_e(e, absolute, outcomes)
    panel_f(f, records, timing)
    panel_g(g, e2e_summary, endpoints, e2e_protocol)
    panel_h(h, contrasts, paired)
    sheet_key(cv)
    # the two columns measure different right needs (a last tick label into
    # the gutter versus into the outer margin): give every panel the larger
    # lock on each side so all eight share one axes width
    cv.lock_reserves()
    left = max(cv._locks[k][0] for k in "ABCDEFGH")
    right = max(cv._locks[k][1] for k in "ABCDEFGH")
    for name in "ABCDEFGH":
        cv.declare_reserve(name, left=left, right=right)
    cv.lock_reserves()
    widths = [cv.axes[k].get_position().width * cv.width_pt for k in "ABCDEFGH"]
    assert max(widths) - min(widths) < 0.5, widths

    problems = cv.save(path, name="figure_morphology_estimation_native", png=png)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

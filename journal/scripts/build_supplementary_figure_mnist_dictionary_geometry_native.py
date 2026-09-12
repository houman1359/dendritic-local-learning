#!/usr/bin/env python3
"""Supplementary sheet S7 (ident ``mnist_dictionary_geometry``) -- the six-rule
MNIST ladder, its within-tree paired contrasts and the dictionary capture of
the same ten-seed cohort -- rebuilt as ONE native full-width
:class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/mnist_dictionary_geometry.pdf``)
is a paste of the upstream S49 render, which has no generator in this
repository.  This builder reads ONLY the frozen tables under
``source_data/image_ladder_controls/summaries`` and redraws the same four
panels with the same plotted quantities.  Nothing about the numbers changes:
every mean is recomputed from the per-seed rows and asserted against the
study's condition / contrast / capture summaries, every positive-seed count
is recomputed from the per-seed paired differences and asserted against the
contrast table, and the frozen intervals are re-checked against a fresh
50,000-draw whole-seed bootstrap (the study's own recipe, see
``summaries/completeness_audit.json``) to Monte-Carlo tolerance.

2026-09-12 visual-review fixes (analysis/figure_visual_review_20260910/
frozen_S7.json), panel by panel:

* A and B keep one colour each -- shunting green in A, raw-additive blue in
  B -- so blue no longer means both 'projected K = 1' (A/B) and 'raw
  additive' (C/D) on one sheet.  At the five arms whose development-selected
  multiplier equals the original common rate (identical ``final_model_sha256``
  in 10/10 seeds) the ten runs are drawn ONCE, as a filled circle inside a
  thin ring, instead of as a filled/open pair of the same runs; the seven
  arms whose rates differ keep the dodged pair with its own seed fan each.
  Tick labels are two short lines with the between-category gap at least
  three times the word space.  The y axis is tight to the data (74-98.5 %)
  and shared by A and B.
* C is on a single categorical axis with one blank unit between the two
  contrast groups (0-3 | 5-8), draws the ten per-seed paired differences as
  a fan behind every mean, prints the positive-seed count k/10 under every
  mark, and carries two-line group labels that stay inside the axis span.
  The freed width comes from D.
* D is a third-width slope panel on the 0-1 capture axis that main Fig. 1G
  uses: the K = 1 and K = 3 marks are dodged horizontally at each checkpoint
  (the shunting initial means 0.53219 and 0.53229 no longer hide one
  another), the ten per-seed capture values are drawn behind every mean --
  the content main Fig. 1G does not carry -- and the 95 % intervals, all
  narrower than 0.007 capture units, are stated on the panel instead of
  being drawn as sub-marker stubs.  Its y axis reads 'mean activation-error
  capture', the plotted ``mean_capture`` column.
* One shared key under the sheet names every glyph: filled circle =
  development-selected rate, open square = original common rate, ringed
  circle = one fit for both rates, small dot = one seed; colour patches for
  shunting and raw additive; solid = K = 1 and dashed = K = 3 in D.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
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

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "image_ladder_controls" / "summaries"
OUT = ROOT / "figures" / "supplementary" / "figure_mnist_dictionary_geometry_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
ARCH = (("shunting", "shunting", COLORS["shunting"]),
        ("additive", "raw additive", COLORS["additive"]))
DASHED = (0, (2.6, 1.8))
N_SEEDS = 10
BOOT_DRAWS = 50_000               # completeness_audit.json: "50,000 paired
                                  # whole-seed bootstrap resamples; descriptive 95% CI"
BOOT_TOL = 2.5e-4                 # Monte-Carlo tolerance on a re-drawn quantile

# the six-rule ladder, in the sheet's left-to-right order
ARMS = (("strict_scalar", "Strict\nscalar"),
        ("neuron_shared", "Per\nneuron"),
        ("projected_k1", "Proj.\nK = 1"),
        ("subtree_k3", "K = 3\nsubtrees"),
        ("exact_path", "Exact\npath"),
        ("decoder_only", "Decoder\nonly"))
POLICIES = (("selected", "o", "full"), ("common_original", "s", "none"))
CONTRASTS = (("subtree_k3_minus_projected_k1", "K = 3 subtrees minus\nprojected K = 1"),
             ("exact_path_minus_subtree_k3", "exact path minus\nK = 3 subtrees"))
BASES = (("broadcast_k1", "-", -0.13), ("subtrees_k3", DASHED, 0.13))
CHECKPOINTS = (("initial", "initial"), ("trained", "trained"))
GROUP_GAP = 1.6                    # blank units between C's two contrast groups
RING_MS = MARKER_MS + 2.8


def csv(name):
    path = SOURCE / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def boot(values, seed=20260912):
    """The study's whole-seed bootstrap of the mean, re-drawn: 50,000 resamples
    of the ten seeds, descriptive 2.5 / 97.5 % quantiles."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(BOOT_DRAWS, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def check_interval(values, mean, lo, hi, *, what):
    """Mean exact; frozen interval brackets the mean and agrees with a fresh
    bootstrap of the same rows to Monte-Carlo tolerance."""
    m, blo, bhi = boot(values)
    np.testing.assert_allclose(m, mean, rtol=0, atol=1e-12, err_msg=what)
    assert lo <= mean <= hi, (what, lo, mean, hi)
    assert abs(blo - lo) < BOOT_TOL and abs(bhi - hi) < BOOT_TOL, (what, blo, lo, bhi, hi)


def fan(ax, x, values, color, *, half=0.14, ms=SEED_MS, alpha=SEED_ALPHA, zorder=2.0):
    """Per-seed values as a jittered fan behind the mean (jitter = seed order)."""
    values = np.asarray(values, float)
    jitter = np.linspace(-half, half, len(values)) if len(values) > 1 else np.zeros(1)
    ax.plot(x + jitter, values, linestyle="none", marker="o", markersize=ms,
            markerfacecolor=color, markeredgecolor="none", alpha=alpha, zorder=zorder)


def mean_glyph(ax, x, mean, color, *, marker="o", fill="full", ring=False, zorder=4.0):
    face = color if fill == "full" else "white"
    ax.plot([x], [mean], linestyle="none", marker=marker, ms=MARKER_MS, markerfacecolor=face,
            markeredgecolor=color, markeredgewidth=LW_ERR, zorder=zorder)
    if ring:
        ax.plot([x], [mean], linestyle="none", marker="o", ms=RING_MS, markerfacecolor="none",
                markeredgecolor=color, markeredgewidth=LW_HAIR, zorder=zorder - 0.1)


# ── A, B: the six-rule ladder, one architecture per panel ────────────────
def panel_ladder(ax, rows, summary, arch, color):
    r = rows[rows.architecture.eq(arch)]
    s = summary[summary.architecture.eq(arch) & summary.metric.eq("test_accuracy")]
    assert len(r) == 120 and r.seed.nunique() == N_SEEDS and len(s) == 12
    printed = []
    lo_all, hi_all = 1.0, 0.0
    for xi, (arm, _) in enumerate(ARMS):
        sel = r[r.arm.eq(arm) & r.rate_policy.eq("selected")].sort_values("seed")
        com = r[r.arm.eq(arm) & r.rate_policy.eq("common_original")].sort_values("seed")
        assert len(sel) == len(com) == N_SEEDS
        assert list(sel.seed) == list(com.seed)
        assert com.multiplier.eq(1.0).all(), "the original common rate is multiplier 1"
        same = bool(sel.multiplier.eq(1.0).all())
        if same:
            assert (sel.final_model_sha256.to_numpy() == com.final_model_sha256.to_numpy()).all(), \
                f"{arch} {arm}: selected multiplier 1.0 but not the same fits"
            np.testing.assert_array_equal(sel.test_accuracy.to_numpy(), com.test_accuracy.to_numpy())
        else:
            assert sel.multiplier.nunique() == 1 and float(sel.multiplier.iloc[0]) != 1.0
            assert (sel.final_model_sha256.to_numpy() != com.final_model_sha256.to_numpy()).all()
        for policy, marker, fill in POLICIES:
            g = sel if policy == "selected" else com
            row = s[s.arm.eq(arm) & s.rate_policy.eq(policy)]
            assert len(row) == 1 and int(row.n.iloc[0]) == N_SEEDS
            row = row.iloc[0]
            vals = g.test_accuracy.to_numpy(float)
            check_interval(vals, row["mean"], row.ci_low, row.ci_high, what=f"{arch} {arm} {policy}")
            lo_all, hi_all = min(lo_all, vals.min()), max(hi_all, vals.max())
            pct = 100.0 * vals
            if same and policy == "common_original":
                continue                        # the same ten fits, drawn once
            x = xi if same else xi + (-0.19 if policy == "selected" else 0.19)
            fan(ax, x, pct, color, half=0.11 if not same else 0.14)
            mean_glyph(ax, x, 100.0 * row["mean"], color, marker=marker, fill=fill, ring=same)
            printed.append(f"{arm} {policy} x{sel.multiplier.iloc[0] if policy == 'selected' else 1.0:g}: "
                           f"{100 * row['mean']:.2f} % [{100 * row.ci_low:.2f}, {100 * row.ci_high:.2f}]"
                           + (" (one fit for both rates)" if same else ""))
    print(f"[{arch}] seed range {100 * lo_all:.2f}-{100 * hi_all:.2f} %; " + "; ".join(printed))
    ax.set_xlim(-0.6, len(ARMS) - 0.4)
    ax.set_xticks(range(len(ARMS)), [lab for _, lab in ARMS])
    ax.set_ylabel("test accuracy (%)")
    return lo_all, hi_all


# ── C: paired within-tree contrasts ──────────────────────────────────────
def panel_contrasts(ax, contrasts, seed_contrasts, rows):
    pc = contrasts[contrasts.metric.eq("test_accuracy")]
    ps = seed_contrasts[seed_contrasts.metric.eq("test_accuracy")]
    xs, counts, seed_lo, seed_hi = [], [], 0.0, 0.0
    printed = []
    x = 0.0
    group_centres = []
    for name, _ in CONTRASTS:
        left = x
        minuend, subtrahend = name.split("_minus_")
        for arch, _, color in ARCH:
            for policy, marker, fill in POLICIES:
                r = pc[pc.contrast.eq(name) & pc.architecture.eq(arch) & pc.rate_policy.eq(policy)]
                assert len(r) == 1
                r = r.iloc[0]
                d = ps[ps.contrast.eq(name) & ps.architecture.eq(arch) & ps.rate_policy.eq(policy)]
                d = d.sort_values("seed")
                assert len(d) == N_SEEDS == int(r.n)
                # the paired difference rebuilt from the per-seed accuracy rows
                acc = rows[rows.architecture.eq(arch) & rows.rate_policy.eq(policy)] \
                    .set_index(["arm", "seed"]).test_accuracy
                rebuilt = np.array([acc[(minuend, sd)] - acc[(subtrahend, sd)] for sd in d.seed])
                vals = d.difference.to_numpy(float)
                np.testing.assert_allclose(vals, rebuilt, rtol=0, atol=1e-12)
                check_interval(vals, r["mean"], r.ci_low, r.ci_high, what=f"C {name} {arch} {policy}")
                positive = int((rebuilt > 0).sum())      # a tie is not positive
                assert positive == int(r.positive), (name, arch, policy, positive, r.positive)
                pp = 100.0 * vals
                seed_lo, seed_hi = min(seed_lo, pp.min()), max(seed_hi, pp.max())
                fan(ax, x, pp, color, half=0.22)
                ax.errorbar([x], [100.0 * r["mean"]], yerr=[[100.0 * (r["mean"] - r.ci_low)],
                                                            [100.0 * (r.ci_high - r["mean"])]],
                            fmt="none", ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                            capthick=LW_ERR, zorder=3.5)
                mean_glyph(ax, x, 100.0 * r["mean"], color, marker=marker, fill=fill)
                xs.append(x)
                counts.append((x, positive, int(r.n)))
                printed.append(f"{name} {arch} {policy}: {100 * r['mean']:+.3f} pp "
                               f"[{100 * r.ci_low:+.3f}, {100 * r.ci_high:+.3f}] positive {positive}/{int(r.n)}")
                x += 1.0
        group_centres.append((left + x - 1.0) / 2.0)
        x += GROUP_GAP                             # the corridor carries the count-row label
    print("[C] " + "; ".join(printed))
    print(f"[C] per-seed differences span {seed_lo:+.2f} to {seed_hi:+.2f} pp")
    x0, x1 = -0.6, xs[-1] + 0.6
    ax.set_xlim(x0, x1)
    ax.plot([x0, x1], [0.0, 0.0], color=MUTE, lw=LW_REF, dashes=(2.6, 2.0), zorder=1.0)
    ax.set_xticks(group_centres, [lab for _, lab in CONTRASTS])
    ax.tick_params(axis="x", length=0)
    count_y = -0.46
    ax.set_ylim(-0.52, 0.34)
    ax.set_yticks([-0.4, -0.2, 0.0, 0.2], ["−0.4", "−0.2", "0", "0.2"])
    assert seed_lo > -0.42 and seed_hi < 0.31, (seed_lo, seed_hi)   # the count row is clear of every datum
    for xc, k, n in counts:
        ax.annotate(f"{k}/{n}", xy=(xc, count_y), xycoords="data", ha="center", va="center",
                    fontsize=PT_BASE, color=MUTE)
    corridor = (xs[3] + xs[4]) / 2.0            # the blank unit between the groups
    ax.annotate("seeds > 0", xy=(corridor, count_y), xycoords="data", ha="center", va="center",
                fontsize=PT_BASE, color=MUTE)
    ax.set_ylabel("paired accuracy difference (pp)")


# ── D: dictionary capture at initialization and after exact-path training ─
def panel_capture(ax, capture, capture_summary):
    c = capture[capture.coordinate.eq("activation")]
    s = capture_summary[capture_summary.coordinate.eq("activation")
                        & capture_summary.metric.eq("mean_capture")]
    assert c.cohort.eq("fresh").all() and c.seed.nunique() == N_SEEDS
    # the exact twelve-column basis captures everything by construction; it is
    # the reference the caption states, not a drawn series
    np.testing.assert_allclose(c[c.basis.eq("exact_k12")].mean_capture.to_numpy(), 1.0, rtol=0, atol=0)
    widest = 0.0
    printed = []
    for arch, _, color in ARCH:
        for basis, ls, dx in BASES:
            means = []
            for ci, (ck, _) in enumerate(CHECKPOINTS):
                g = c[c.architecture.eq(arch) & c.basis.eq(basis) & c.checkpoint.eq(ck)].sort_values("seed")
                r = s[s.architecture.eq(arch) & s.basis.eq(basis) & s.checkpoint.eq(ck)]
                assert len(g) == N_SEEDS and len(r) == 1 and int(r.n.iloc[0]) == N_SEEDS
                r = r.iloc[0]
                vals = g.mean_capture.to_numpy(float)
                check_interval(vals, r["mean"], r.ci_low, r.ci_high, what=f"D {arch} {basis} {ck}")
                widest = max(widest, float(r.ci_high - r.ci_low))
                fan(ax, ci + dx, vals, color, half=0.07, ms=SEED_MS * 0.85)
                means.append(float(r["mean"]))
                printed.append(f"{arch} {basis} {ck}: {r['mean']:.4f} [{r.ci_low:.4f}, {r.ci_high:.4f}] "
                               f"seeds {vals.min():.4f}-{vals.max():.4f}")
            xs = np.arange(len(CHECKPOINTS)) + dx
            ax.plot(xs, means, color=color, lw=LW_DATA, ls=ls, zorder=3.0)
            for xv, m in zip(xs, means):
                mean_glyph(ax, xv, m, color)
    assert widest < 0.007, widest                 # 'at most 0.007 capture units'
    print("[D] " + "; ".join(printed))
    print(f"[D] widest 95 % interval {widest:.5f} capture units (not drawn)")
    ax.set_xlim(-0.5, len(CHECKPOINTS) - 0.5)
    ax.set_xticks(range(len(CHECKPOINTS)), [lab for _, lab in CHECKPOINTS])
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], ["0", "0.2", "0.4", "0.6", "0.8", "1"])
    ax.set_ylabel("mean activation-error\ncapture (fraction)")
    ax.annotate("10 seeds per mark\n95% CI ≤ 0.007, not drawn", xy=(0.0, 0.0),
                xycoords="axes fraction", xytext=(3.0, 3.0), textcoords="offset points",
                ha="left", va="bottom", fontsize=PT_BASE, color=MUTE, linespacing=1.1)
    return widest


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 392.0
HGUTTER_PT = 26.0
VGUTTER_PT = 50.0
MARGINS = Margins(left=40.0, right=6.0, top=18.0, bottom=62.0)


def build(path: Path = OUT):
    rows = csv("fresh_analysis_rows_six_rules.csv")
    summary = csv("condition_summary_six_rules.csv")
    contrasts = csv("paired_contrasts_six_rules.csv")
    seed_contrasts = csv("paired_seed_contrasts_six_rules.csv")
    capture = csv("delivery_coordinate_capture.csv")
    capture_summary = csv("delivery_coordinate_capture_summary.csv")
    assert len(rows) == 240 and rows.stage.eq("fresh").all() and rows.seed.nunique() == N_SEEDS
    assert len(seed_contrasts) == 560 and len(capture) == 240

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 2, hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
                      margins=MARGINS)
    ax_a = cv.panel("A", 0, 0, 6, grid="y", title="Shunting: ten fresh paired seeds")
    ax_b = cv.panel("B", 0, 6, 6, grid="y", title="Raw additive: ten fresh paired seeds")
    ax_c = cv.panel("C", 1, 0, 8, grid="y", title="Added within-tree resolution")
    ax_d = cv.panel("D", 1, 8, 4, grid="y", title="Dictionary capture")
    for name in "ABCD":
        cv.declare_reserve(name, left=15.0, right=8.0)

    lo_a, hi_a = panel_ladder(ax_a, rows, summary, "shunting", COLORS["shunting"])
    lo_b, hi_b = panel_ladder(ax_b, rows, summary, "additive", COLORS["additive"])
    lo, hi = 100.0 * min(lo_a, lo_b), 100.0 * max(hi_a, hi_b)
    assert 74.0 < lo and hi < 98.0, (lo, hi)
    for ax in (ax_a, ax_b):
        ax.set_ylim(73.6, 98.6)
        ax.set_yticks([75, 80, 85, 90, 95], ["75", "80", "85", "90", "95"])
    panel_contrasts(ax_c, contrasts, seed_contrasts, rows)
    panel_capture(ax_d, capture, capture_summary)

    # one shared key: glyphs (rate policy, seed), colour (architecture), line style (profile)
    glyph = dict(linestyle="none", ms=MARKER_MS, markeredgewidth=LW_ERR, color=INK)
    handles = [
        Line2D([], [], marker="o", markerfacecolor=INK, label="development-selected rate", **glyph),
        Line2D([], [], marker="s", markerfacecolor="white", label="original common rate", **glyph),
        Line2D([], [], marker="o", markerfacecolor=INK, markeredgecolor=INK, label="one fit for both rates", **glyph),
        Line2D([], [], marker="o", linestyle="none", ms=SEED_MS, markerfacecolor=INK,
               markeredgecolor="none", alpha=SEED_ALPHA, label="one seed"),
        Patch(facecolor=COLORS["shunting"], edgecolor="none", label="shunting"),
        Patch(facecolor=COLORS["additive"], edgecolor="none", label="raw additive"),
        Line2D([], [], color=INK, lw=LW_DATA, ls="-", label="K = 1, one profile (D)"),
        Line2D([], [], color=INK, lw=LW_DATA, ls=DASHED, label="K = 3, three profiles (D)"),
    ]
    leg = cv.fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.53, 0.0),
                        ncol=4, frameon=False, fontsize=PT_BASE, handlelength=2.2,
                        columnspacing=1.5, handletextpad=0.6, borderaxespad=0.5,
                        labelspacing=0.45)
    # the ringed glyph: a second, larger open circle drawn in the same legend
    # handle box as its filled partner (matplotlib keeps each handle in a
    # DrawingArea, so the ring is positioned and clipped with it)
    ring = leg.legend_handles[2]
    ring_line = Line2D(*ring.get_data(), marker="o", linestyle="none", ms=RING_MS,
                       markerfacecolor="none", markeredgecolor=INK, markeredgewidth=LW_HAIR,
                       markevery=ring.get_markevery())
    for box in leg.findobj(match=lambda a: a.__class__.__name__ == "DrawingArea"):
        if ring in box.get_children():
            box.add_artist(ring_line)
            break
    else:
        raise RuntimeError("legend handle box for the ringed glyph not found")

    problems = cv.save(path, name="figure_mnist_dictionary_geometry_native", png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

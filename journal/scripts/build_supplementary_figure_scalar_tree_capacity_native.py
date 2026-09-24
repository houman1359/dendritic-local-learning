#!/usr/bin/env python3
"""Supplementary sheet S12 (ident ``scalar_tree_capacity``) -- interaction
structure constrains scalar trees at matched input spectra -- rebuilt as ONE
native full-width :class:`figure_canvas.NativeCanvas`.

The curated sheet (``figures/supplementary/curated/scalar_tree_capacity.pdf``)
is a paste of two upstream renders (old S36 A, B, D and old S37 A, B, C) that
have no generator in this repository, so the verified in-panel defects of the
2026-09-10 visual review (``analysis/figure_visual_review_20260910/
frozen_S12.json``) could not be fixed by the paste layer.  This builder reads
ONLY the frozen tables under ``source_data/morphology_structure/`` and draws
the same six panels, with the same plotted quantities, on the paper's canvas.
Nothing about the numbers changes; every printed or plotted value is
asserted against the table it comes from before it is drawn.

Panels (same letters and content as the curated sheet):

* A  the two exhaustive target families and the nested-prefix control, each
     with its mark (the sheet's family key), its name and its defining
     formula, spread over the cell; the counts, the shared input-gradient
     second moment and the tree budget are asserted here
     (``protocol.json``, ``constructive/constructed_trees.json``,
     ``constructive/adaptive_constructions.csv``) and stated in the caption.
* B  the full rank-two and centered rank-one cut-tail lower bounds versus
     the best fitted population NMSE of all 1,680 candidate fits
     (``candidate_outcomes.csv``), aggregated on their 8 and 9 distinct
     points: mark area proportional to the number of fits, every count
     printed beside its mark (so no mark-area key), the origin marks at a
     legible floor.  Two families at ONE point never leave it on the
     quantitative axes: unequal marks are superposed (the smaller on top,
     white hairline edge) and the two equal floor-size origin marks are
     dodged along the equality rule, with one ``n + n`` label per point.
     The two panel titles name the bound; the curated sheet's interpretive
     super-title, the equality label and the fit census moved to the caption
     (2026-09-23 clarity pass).
* C  mean excess NMSE above the best of the twelve fixed candidates for all
     SEVEN selection policies of ``policy_summary.csv`` (the fixed balanced
     candidate is now drawn), with the 105 matching and 35 quartic
     per-target values of ``policy_outcomes.csv`` as fans behind the family
     means and the per-row count of targets at exactly zero excess.
* D, E  the constructed trees for ``matching_000`` and ``nested_prefix_000``
     (``constructive/constructed_trees.json``, identical in
     ``constructive_dp_v2``), inputs at the top and the root readout at the
     foot as in main Fig. 4A, at one common level pitch and leaf pitch with
     the roots on one line and a depth ruler on each; the readout label is
     centred on the root in both.
* F  the minimum achievable maximum centered-cut bound over all labeled
     binary trees at depth limits 3, 4 and 5
     (``design/constructive_depth_certificate.csv``) on an ordinal axis with
     no connecting segments, the three families offset side by side at each
     limit so the coincident zeros are all visible; the one non-zero value
     (0.146) and the per-family n are given in the caption.

2026-09-23 clarity pass: headline titles (A, C-F), explanatory lines and
restated numbers moved to the caption; axis labels in sentence case.

Colour and shape carry ONE meaning each across the sheet: green circle =
matching targets, purple triangle = quartic targets, amber square = nested-
prefix controls; mute grey = reference rules (equality in B, zero in C and
F).  Every mark is filled; translucent small dots are per-target values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS, LW_EDGE, LW_HAIR, LW_REF, MARKER_MS, PT_BASE, SEED_ALPHA, Margins,
    NativeCanvas)

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "morphology_structure"
OUT = ROOT / "figures" / "supplementary" / "figure_scalar_tree_capacity_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]

# the three families: one hue, one shape, one count each, sheet-wide
FAMILIES = {
    "quadratic_matching": dict(short="matching", color=COLORS["shunting"],
                               marker="o", n=105),
    "quartic_partition": dict(short="quartic", color=COLORS["oracle"],
                              marker="^", n=35),
    "nested_prefix_control": dict(short="nested", color=COLORS["local"],
                                  marker="s", n=24),
}
PRIMARY = ("quadratic_matching", "quartic_partition")
DASHED = (2.6, 2.0)
FAN_MS = 2.0                  # per-target dot diameter, pt
SUB_DROP_PT = 1.6             # baseline drop of a subscript span
A_FOOT_PT = 3.0               # A: last subscript baseline above the axes bottom


def csv(name):
    path = SOURCE / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def pt_per_unit(ax):
    """(x, y) points per data unit of ``ax`` at its current box."""
    bb = ax.get_window_extent()
    fig = ax.figure
    w_pt = bb.width * 72.0 / fig.dpi
    h_pt = bb.height * 72.0 / fig.dpi
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    return w_pt / (x1 - x0), h_pt / (y1 - y0)


def text_width_pt(ax, s, size):
    fig = ax.figure
    t = ax.text(0, 0, s, fontsize=size, alpha=0.0)
    fig.canvas.draw()
    w = t.get_window_extent(fig.canvas.get_renderer()).width * 72.0 / fig.dpi
    t.remove()
    return w


def chain(ax, x, y, parts, *, size=PT_BASE, color=INK, ha="left",
          zorder=6, gap_pt=0.35):
    """A run of plain-text spans with token-size subscripts.

    ``parts`` is a list of ``(text, level)`` with level 0 a base span and
    level 1 a subscript; every span is a real 7 pt token (mathtext would
    shrink a subscript to 4.9 pt, below the type floor).  ``x``, ``y`` are
    in the axes' data units (points on a schematic axes), ``y`` is the base
    baseline; ``ha`` aligns the whole run.  Returns the run's width in pt.
    """
    widths = [text_width_pt(ax, s, size) for s, _ in parts]
    total = sum(widths) + gap_pt * (len(parts) - 1)
    if ha == "center":
        x = x - total / 2.0
    elif ha == "right":
        x = x - total
    cursor = x
    for (s, level), w in zip(parts, widths):
        ax.text(cursor, y - (SUB_DROP_PT if level else 0.0), s, fontsize=size,
                color=color, ha="left", va="baseline", zorder=zorder)
        cursor += w + gap_pt
    return total


def key_mark(ax, x, y, fam, *, ms=MARKER_MS, zorder=6):
    f = FAMILIES[fam]
    ax.plot([x], [y], linestyle="none", marker=f["marker"], markersize=ms,
            markerfacecolor=f["color"], markeredgecolor="white",
            markeredgewidth=LW_HAIR, zorder=zorder, clip_on=False)


# ── A: the families, their marks and their definitions ────────────────────
def panel_a(ax, protocol, trees, constructions):
    """Schematic axes in points: the family key plus the definitions."""
    assert protocol["restarts"] == 4 and protocol["sweeps"] == 32
    assert protocol["model"].startswith("8 scalar input leaves;7 bilinear internal units;"
                                        "28 trainable coefficients;14 edges;root readout only")
    assert protocol["tasks"].startswith("All105 perfect matchings and35 unordered4+4 partitions")
    # the spectra: every primary target I8/4, every nested control the one
    # anisotropic spectrum, from the 164 constructed trees
    spectra = {}
    for rec in trees:
        fam = rec["task_id"].rsplit("_", 1)[0]
        spectra.setdefault(fam, []).append(tuple(sorted(rec["input_gradient_second_moment_spectrum"])))
    assert set(spectra) == {"matching", "quartet", "nested_prefix"}
    assert len(spectra["matching"]) == 105 and len(spectra["quartet"]) == 35 and len(spectra["nested_prefix"]) == 24
    for fam in ("matching", "quartet"):
        for s in spectra[fam]:
            np.testing.assert_allclose(s, [0.25] * 8, rtol=0, atol=1e-12)
    nested = [0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0]
    for s in spectra["nested_prefix"]:
        np.testing.assert_allclose(s, nested, rtol=0, atol=1e-12)
    assert (constructions.parameters == 28).all() and (constructions.edges == 14).all()
    assert (constructions["rank"] == 8).all()
    counts = constructions.family.value_counts()
    assert counts["quadratic_matching"] == 105 and counts["quartic_partition"] == 35 \
        and counts["nested_prefix_control"] == 24

    box = ax.get_position()
    fig = ax.figure
    W = box.width * fig.get_size_inches()[0] * 72.0
    H = box.height * fig.get_size_inches()[1] * 72.0
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    x_mark = 3.0
    x_text = 10.0
    line = 9.2                       # baseline pitch, pt
    # three key entries of (family name, defining formula); the counts, the
    # shared spectra and the tree budget are stated in the caption
    blocks = [
        ("quadratic_matching", [("perfect matchings M of x", 0), ("1", 1), (" … x", 0), ("8", 1)],
         [("f", 0), ("M", 1), (" = ½ Σ", 0), ("(i, j) ∈ M", 1), (" x", 0), ("i", 1), (" x", 0), ("j", 1)]),
        ("quartic_partition", [("four-plus-four partitions A | A′", 0)],
         [("f", 0), ("A", 1), (" = ½ (Π", 0), ("i ∈ A", 1), (" x", 0), ("i", 1),
          (" + Π", 0), ("i ∉ A", 1), (" x", 0), ("i", 1), (")", 0)]),
        ("nested_prefix_control", [("nested-prefix controls", 0)],
         [("f", 0), ("π", 1), (" = ½ Σ", 0), ("k ∈ {2, 4, 6, 8}", 1), (" Π", 0), ("j ≤ k", 1),
          (" x", 0), ("π(j)", 1)]),
    ]
    # the entries spread over the whole cell: the first name line's cap
    # height at the axes top, the last formula's subscript foot A_FOOT_PT
    # above the axes bottom (a schematic cell must be filled by its ink)
    y_first = H - 6.0
    pitch = (y_first - line - SUB_DROP_PT - A_FOOT_PT) / (len(blocks) - 1)
    for k, (fam, name, parts) in enumerate(blocks):
        y = y_first - k * pitch
        key_mark(ax, x_mark, y + 2.4, fam, ms=MARKER_MS * 0.95)
        chain(ax, x_text, y, name, size=PT_BASE, color=FAMILIES[fam]["color"])
        chain(ax, x_text, y - line, parts, size=PT_BASE, color=INK)
    print("[A] families 105 / 35 / 24; spectra I8/4 x 140 and anisotropic x 24; "
          "28 coefficients and 14 edges in all 164 constructions")


# ── B: cut-tail lower bounds versus achieved NMSE ─────────────────────────
B_LIM = (-0.085, 0.93)
B_DMAX_PT = 15.0              # diameter of the n = 725 mark
B_DFLOOR_PT = 3.2             # a mark is never drawn smaller than this (n <= 18)
EXPECTED = {
    "full_cut_bound": {(0.0, 0.0, "quadratic_matching"): 4, (0.0, 0.0, "quartic_partition"): 4,
                       (0.0, 0.5, "quartic_partition"): 416, (0.25, 0.25, "quadratic_matching"): 52,
                       (0.25, 0.5, "quadratic_matching"): 88, (0.25, 0.75, "quadratic_matching"): 36,
                       (0.5, 0.5, "quadratic_matching"): 355, (0.5, 0.75, "quadratic_matching"): 725},
    "centered_cut_bound": {(0.0, 0.0, "quadratic_matching"): 4, (0.0, 0.0, "quartic_partition"): 4,
                           (0.25, 0.25, "quadratic_matching"): 52, (0.25, 0.5, "quadratic_matching"): 88,
                           (0.25, 0.75, "quadratic_matching"): 12, (0.5, 0.5, "quadratic_matching"): 355,
                           (0.5, 0.5, "quartic_partition"): 416, (0.5, 0.75, "quadratic_matching"): 173,
                           (0.75, 0.75, "quadratic_matching"): 576},
}


def aggregate(outcomes, column):
    """{(bound, nmse, family): n fits}, values rounded to 6 dp."""
    g = outcomes.groupby([outcomes[column].round(6), outcomes.normalized_mse.round(6),
                          "family"]).size()
    return {(float(b), float(m), str(f)): int(n) for (b, m, f), n in g.items()}


def diameter_pt(n):
    k = B_DMAX_PT ** 2 / 725.0
    return max(float(np.sqrt(k * n)), B_DFLOOR_PT)


def panel_b(axes, outcomes):
    assert len(outcomes) == 1680
    assert outcomes.task_id.nunique() == 140 and outcomes.candidate_id.nunique() == 12
    assert set(outcomes.family) == set(PRIMARY)
    assert (outcomes.family.value_counts()[["quadratic_matching", "quartic_partition"]]
            == [1260, 420]).all()
    # the bound never exceeds the achieved error (a valid lower bound) beyond
    # the optimizer's 1e-9 tolerance
    for column in ("full_cut_bound", "centered_cut_bound"):
        assert (outcomes.normalized_mse - outcomes[column]).min() > -1e-8
    for ax, column in ((axes[0], "full_cut_bound"), (axes[1], "centered_cut_bound")):
        points = aggregate(outcomes, column)
        assert points == EXPECTED[column], (column, points)
        assert sum(points.values()) == 1680
        sx, sy = pt_per_unit(ax)
        # group the families that share a point
        by_xy = {}
        for (b, m, fam), n in points.items():
            by_xy.setdefault((b, m), []).append((fam, n))
        for (b, m), members in sorted(by_xy.items()):
            members.sort(key=lambda t: PRIMARY.index(t[0]))
            diam = [diameter_pt(n) for _, n in members]
            if len(members) == 1:
                offsets = [(0.0, 0.0)]
            elif abs(diam[0] - diam[1]) < 0.5:
                # two equal (floor-size) marks at one point would hide each
                # other: dodge them ALONG the equality rule, touching, so
                # both still read bound = NMSE and neither leaves the rule
                assert b == m, (b, m, members)
                step = (max(diam) + 0.6) / 2.0 / np.hypot(sx, sy)   # data units
                offsets = [(-step * sx, -step * sy), (step * sx, step * sy)]
            else:
                # unequal marks at one point: superposed at the TRUE point,
                # the smaller drawn on top with its white hairline edge (a
                # horizontal dodge would move a mark off its true bound)
                offsets = [(0.0, 0.0), (0.0, 0.0)]
            for (fam, n), d, (ox, oy) in zip(members, diam, offsets):
                f = FAMILIES[fam]
                x = b + ox / sx
                y = m + oy / sy
                ax.scatter([x], [y], s=d ** 2, marker=f["marker"], color=f["color"],
                           edgecolors="white", linewidths=LW_HAIR,
                           zorder=3.1 if d < max(diam) - 0.5 else 3.0)
                if len(members) == 2:
                    continue          # one shared label per point, below
                if b == m:        # on the equality rule: label up its left side
                    ax.annotate(f"{n}", xy=(x, m), xycoords="data",
                                xytext=(-0.72 * (d / 2.0 + 1.8), 0.72 * (d / 2.0 + 1.8)),
                                textcoords="offset points", ha="right", va="bottom",
                                fontsize=PT_BASE, color=MUTE, zorder=6)
                else:
                    ax.annotate(f"{n}", xy=(x, m), xycoords="data",
                                xytext=(0.0, d / 2.0 + 1.4), textcoords="offset points",
                                ha="center", va="bottom", fontsize=PT_BASE, color=MUTE,
                                zorder=6)
            if len(members) == 2:
                # the shared ``n + n`` label, in the empty half-plane below
                # the equality rule, clear of the pair's right-most extent
                reach = max(ox + d / 2.0 for d, (ox, _) in zip(diam, offsets))
                ax.annotate(" + ".join(str(n) for _, n in members), xy=(b, m),
                            xycoords="data", xytext=(reach + 2.6, -2.4),
                            textcoords="offset points", ha="left", va="top",
                            fontsize=PT_BASE, color=MUTE, zorder=6)
        ax.plot(list(B_LIM), list(B_LIM), color=MUTE, lw=LW_REF, dashes=DASHED,
                zorder=1.0, solid_capstyle="butt")
        print(f"[B] {column}: {len(points)} distinct points, counts "
              f"{sorted(points.values())}")
    axes[0].set_ylabel("Best fitted population NMSE")
    # every mark carries its fit count, so no mark-area key is drawn; the
    # area scaling, the equality rule and the fit census are in the caption


def setup_b(ax):
    ax.set_xlim(*B_LIM)
    ax.set_ylim(*B_LIM)
    ax.set_xticks([0, 0.25, 0.5, 0.75], ["0", "0.25", "0.5", "0.75"])
    ax.set_yticks([0, 0.25, 0.5, 0.75], ["0", "0.25", "0.5", "0.75"])
    ax.set_aspect("auto")


# ── C: policy regret with the per-target fans ─────────────────────────────
POLICIES = [
    ("centered_cut_bound", "centered-cut bound"),
    ("full_cut_bound", "full-cut bound"),
    ("centered_cut_sum", "sum of centered tails"),
    ("two_sweep_pilot", "two-sweep fitting pilot"),
    ("fixed_balanced_p0", "fixed balanced candidate"),
    ("best_fixed_in_hindsight", "best fixed in hindsight"),
    ("uniform_random_expectation", "random expectation"),
]
C_GAP_AFTER = 3               # a half-row gap after the fourth (selection) row
C_XLIM = (-0.022, 0.80)
SUB = 0.21                    # family sub-row offset, row units
JIT = 0.12                    # fan jitter half-width, row units


def row_y(i):
    return i + (0.5 if i > C_GAP_AFTER else 0.0)


def panel_c(ax, outcomes, summary):
    assert len(outcomes) == 980 and outcomes.task_id.nunique() == 140
    assert set(outcomes.policy) == {p for p, _ in POLICIES} and len(summary) == 14
    ys = [row_y(i) for i in range(len(POLICIES))]
    ax.set_ylim(ys[-1] + 0.6, -0.6)
    ax.set_xlim(*C_XLIM)
    ax.set_yticks([])
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    x0 = C_XLIM[0]
    ax.axvline(0.0, color=MUTE, lw=LW_REF, dashes=DASHED, zorder=1.0)
    rng = np.random.default_rng(20260912)
    for (policy, label), y in zip(POLICIES, ys):
        ax.plot([x0, x0], [y - 0.36, y + 0.36], color=EDGE, lw=LW_HAIR, clip_on=False,
                zorder=1.5, solid_capstyle="butt")
        ax.annotate(label, xy=(0.0, y), xycoords=("axes fraction", "data"),
                    xytext=(-4.0, 0.0), textcoords="offset points", ha="right", va="center",
                    fontsize=PT_BASE, color=INK, annotation_clip=False)
        zeros = []
        for k, fam in enumerate(PRIMARY):
            f = FAMILIES[fam]
            rows = outcomes[outcomes.policy.eq(policy) & outcomes.family.eq(fam)]
            vals = rows.regret.to_numpy(float)
            assert len(vals) == f["n"] and rows.task_id.is_unique
            rec = summary[summary.policy.eq(policy) & summary.family.eq(fam)].iloc[0]
            np.testing.assert_allclose(vals.mean(), rec.mean_regret, rtol=0, atol=1e-12)
            np.testing.assert_allclose(vals.max(), rec.max_regret, rtol=0, atol=1e-12)
            assert vals.min() > -1e-12 and vals.max() < C_XLIM[1]
            n_zero = int((vals < 1e-9).sum())
            zeros.append(n_zero)
            yc = y + (-SUB if k == 0 else SUB)
            jitter = rng.permutation(np.linspace(-JIT, JIT, len(vals)))
            ax.plot(vals, yc + jitter, linestyle="none", marker="o", markersize=FAN_MS,
                    markerfacecolor=f["color"], markeredgecolor="none", alpha=SEED_ALPHA,
                    zorder=2.0)
            ax.plot([float(rec.mean_regret)], [yc], linestyle="none", marker=f["marker"],
                    markersize=MARKER_MS, markerfacecolor=f["color"], markeredgecolor="white",
                    markeredgewidth=LW_HAIR, zorder=4.0)
            print(f"[C] {policy:27s} {f['short']:8s} mean {rec.mean_regret:.6f} "
                  f"max {rec.max_regret:.4f} zero {n_zero}/{f['n']}")
        ax.annotate(f"{zeros[0]}/105, {zeros[1]}/35", xy=(1.0, y),
                    xycoords=("axes fraction", "data"), xytext=(4.0, 0.0),
                    textcoords="offset points", ha="left", va="center", fontsize=PT_BASE,
                    color=MUTE, annotation_clip=False)
    ax.annotate("at zero excess", xy=(1.0, 1.0), xycoords="axes fraction",
                xytext=(4.0, 2.0), textcoords="offset points", ha="left", va="bottom",
                fontsize=PT_BASE, color=MUTE, annotation_clip=False)
    ax.set_xticks([0, 0.25, 0.5, 0.75], ["0", "0.25", "0.5", "0.75"])
    ax.set_xlabel("Excess population NMSE above the best of the twelve candidates")


# ── D, E: the constructed trees ───────────────────────────────────────────
def tree_layout(children):
    """Leaf order by in-order traversal, node depth and x (leaf pitch units)."""
    children = {int(k): tuple(v) for k, v in children.items()}
    root = max(children)
    assert root == 14 and set(children) == set(range(8, 15))
    order, depth, x = [], {}, {}

    def walk(u, d):
        depth[u] = d
        if u in children:
            for c in children[u]:
                walk(c, d + 1)
            x[u] = (x[children[u][0]] + x[children[u][1]]) / 2.0
        else:
            x[u] = float(len(order))
            order.append(u)

    walk(root, 0)
    assert sorted(order) == list(range(8))
    return children, root, order, depth, x


def panel_tree(ax, rec, fam, *, expected_depth, pitch_pt, leaf_pt, ruler=True):
    children, root, order, depth, x = tree_layout(rec["children"])
    max_depth = max(depth[u] for u in range(8))
    assert max_depth == expected_depth
    f = FAMILIES[fam]
    box = ax.get_position()
    fig = ax.figure
    W = box.width * fig.get_size_inches()[0] * 72.0
    H = box.height * fig.get_size_inches()[1] * 72.0
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    top_pad = 9.0                    # leaf labels above the top leaves
    y_leaf_top = H - top_pad
    ruler_w = 22.0 if ruler else 0.0
    tree_w = 7.0 * leaf_pt
    x_left = ruler_w + (W - ruler_w - tree_w) / 2.0
    X = lambda u: x_left + x[u] * leaf_pt                       # noqa: E731
    Y = lambda d: y_leaf_top - (4 - d) * pitch_pt              # noqa: E731
    # edges
    for u, (a, b) in children.items():
        for c in (a, b):
            ax.plot([X(u), X(c)], [Y(depth[u]), Y(depth[c])], color=MUTE, lw=LW_EDGE,
                    solid_capstyle="round", zorder=2)
    # internal nodes: filled discs; leaves: open squares with their input
    for u in children:
        ax.plot([X(u)], [Y(depth[u])], linestyle="none", marker="o", markersize=4.6,
                markerfacecolor=f["color"], markeredgecolor="white", markeredgewidth=LW_HAIR,
                zorder=4)
    for u in range(8):
        ax.plot([X(u)], [Y(depth[u])], linestyle="none", marker="s", markersize=4.0,
                markerfacecolor="white", markeredgecolor=f["color"], markeredgewidth=LW_EDGE,
                zorder=4)
        chain(ax, X(u), Y(depth[u]) + 4.6, [("x", 0), (str(u + 1), 1)], size=PT_BASE,
              color=INK, ha="center")
    # root readout, centred on the root
    ax.annotate("", xy=(X(root), Y(0) - 3.2), xytext=(X(root), Y(0) - 9.0),
                arrowprops=dict(arrowstyle="<|-", color=f["color"], lw=LW_EDGE,
                                shrinkA=0, shrinkB=0, mutation_scale=6.0), zorder=3)
    ax.text(X(root), Y(0) - 10.5, "root readout", ha="center", va="top", fontsize=PT_BASE,
            color=f["color"], zorder=6)
    if ruler:
        xr = 2.0
        ax.plot([xr + 8.0, xr + 8.0], [Y(0), Y(4)], color=EDGE, lw=LW_HAIR, zorder=1)
        for d in range(5):
            ax.plot([xr + 6.5, xr + 8.0], [Y(d), Y(d)], color=EDGE, lw=LW_HAIR, zorder=1)
            ax.text(xr + 5.0, Y(d), f"{d}", ha="right", va="center", fontsize=PT_BASE,
                    color=MUTE, zorder=6)
        ax.text(xr, Y(4) + 4.6, "depth", ha="left", va="bottom", fontsize=PT_BASE,
                color=MUTE, zorder=6)
    leaves = [u + 1 for u in order]
    print(f"[{fam}] {rec['task_id']}: depth {max_depth}, leaf order x{leaves}")
    return leaves


# ── F: depth-limited certificate ──────────────────────────────────────────
F_DEPTHS = (3, 4, 5)
F_OFF = (-0.24, 0.0, 0.24)
F_YLIM = (-0.016, 0.172)


def panel_f(ax, cert, constructions):
    assert len(cert) == 492 and cert.task_id.nunique() == 164
    assert sorted(cert.maximum_depth.unique()) == list(F_DEPTHS)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(*F_YLIM)
    ax.plot([-0.5, 2.5], [0.0, 0.0], color=MUTE, lw=LW_REF, dashes=DASHED, zorder=1.0,
            solid_capstyle="butt")
    sx, sy = pt_per_unit(ax)
    values = {}
    for fam, off in zip(FAMILIES, F_OFF):
        f = FAMILIES[fam]
        for i, d in enumerate(F_DEPTHS):
            rows = cert[cert.family.eq(fam) & cert.maximum_depth.eq(d)]
            assert len(rows) == f["n"] and rows.task_id.is_unique
            vals = rows.best_centered_cut_bound.to_numpy(float)
            assert vals.min() == vals.max(), (fam, d)     # identical within family
            v = float(vals[0])
            values[(fam, d)] = v
            ax.plot([i + off], [v], linestyle="none", marker=f["marker"], markersize=MARKER_MS,
                    markerfacecolor=f["color"], markeredgecolor="white",
                    markeredgewidth=LW_HAIR, zorder=4)
    # the one non-zero value: nested controls at depth limit 3 (0.146, given
    # in the caption rather than printed beside the mark)
    for (fam, d), v in values.items():
        if fam == "nested_prefix_control" and d == 3:
            assert 0.1464 < v < 0.1465
        else:
            assert v == 0.0, (fam, d, v)
    # the constructed depths agree with the certificate: the minimum depth
    # with a zero bound is 3 for both primaries and 4 for the controls
    for fam, depth in (("quadratic_matching", 3), ("quartic_partition", 3),
                       ("nested_prefix_control", 4)):
        assert (constructions[constructions.family.eq(fam)].depth == depth).all()
        assert min(d for d in F_DEPTHS if values[(fam, d)] == 0.0) == depth
    ax.set_xticks(range(3), [str(d) for d in F_DEPTHS])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0, 0.05, 0.10, 0.15], ["0", "0.05", "0.10", "0.15"])
    ax.set_xlabel("Depth limit (root-to-leaf edges)")
    ax.set_ylabel("Minimum cut bound (NMSE)")
    # family key in the empty upper right (the per-family n is in the caption)
    kx, ky = 0.95, 0.118
    for k, fam in enumerate(FAMILIES):
        f = FAMILIES[fam]
        yk = ky - k * 0.021
        ax.plot([kx], [yk], linestyle="none", marker=f["marker"], markersize=MARKER_MS,
                markerfacecolor=f["color"], markeredgecolor="white", markeredgewidth=LW_HAIR,
                zorder=6)
        ax.text(kx + 0.16, yk, f["short"], ha="left", va="center",
                fontsize=PT_BASE, color=INK, zorder=6)
    print(f"[F] nested at depth 3: {values[('nested_prefix_control', 3)]:.9f}; "
          f"all other cells 0")


# ── the canvas ────────────────────────────────────────────────────────────
CANVAS_H_PT = 492.0
ROW_PT = [138.0, 126.0, 118.0]
HGUTTER_PT = 30.0
VGUTTER_PT = 40.0
MARGINS = Margins(left=34.0, right=12.0, top=16.0, bottom=26.0)
C_LEFT_PT = 84.0              # the row-label gutter of C (inside its slot)
C_RIGHT_PT = 54.0             # the zero-count column of C
C_TOP_PT = 10.0               # the count-column header band C does not get
                              # from the row lock


def build(path: Path = OUT, *, png=False):
    outcomes = csv("candidate_outcomes.csv")
    policy_outcomes = csv("policy_outcomes.csv")
    policy_summary = csv("policy_summary.csv")
    cert = csv("design/constructive_depth_certificate.csv")
    constructions = csv("constructive/adaptive_constructions.csv")
    protocol = json.loads((SOURCE / "protocol.json").read_text())
    trees = json.loads((SOURCE / "constructive" / "constructed_trees.json").read_text())
    trees_v2 = json.loads((SOURCE / "constructive_dp_v2" / "constructed_trees.json").read_text())
    assert len(trees) == 164 and len(trees_v2) == 164 and len(constructions) == 164
    by_id = {t["task_id"]: t for t in trees}
    by_id_v2 = {t["task_id"]: t for t in trees_v2}
    for tid in ("matching_000", "nested_prefix_000"):
        assert by_id[tid]["children"] == by_id_v2[tid]["children"], tid
    # every construction is exact: the caption's 7e-30 error and unit coefficients
    assert (constructions.normalized_mse < 7e-30).all()
    v2 = csv("constructive_dp_v2/adaptive_constructions.csv")
    assert (v2.max_abs_parameter < 1.0 + 1e-12).all() and (v2.normalized_mse < 7e-30).all()

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 3, row_weights=ROW_PT, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS, letter_clearance=True)
    # panel titles only where they name the condition that tells two
    # otherwise identical panels apart (the two bounds of B)
    a = cv.panel("A", 0, 0, 4, schematic=True)
    b1 = cv.panel("B", 0, 4, 4, letter="B", grid="both", title="Full-cut bound, rank ≤ 2")
    b2 = cv.panel("B_centered", 0, 8, 4, letter="", grid="both",
                  title="Centered-cut bound, rank ≤ 1")
    c = cv.panel("C", 1, 0, 12, letter="C", lock=False,
                 inset_pt=(C_LEFT_PT, C_RIGHT_PT, C_TOP_PT, 0.0), grid="x")
    d = cv.panel("D", 2, 0, 4, letter="D", schematic=True)
    e = cv.panel("E", 2, 4, 4, letter="E", schematic=True)
    f = cv.panel("F", 2, 8, 4, letter="F", grid="y")
    for ax in (b1, b2):
        setup_b(ax)
    # one declared reserve on every four-module panel: the column lock then
    # gives every same-span panel of a row one axes width by construction
    for name in ("A", "B", "B_centered", "D", "E", "F"):
        cv.declare_reserve(name, left=12.0, right=9.0)
    # decorations first, so the column lock measures them; data drawn in
    # points afterwards
    panel_c(c, policy_outcomes, policy_summary)
    b1.set_ylabel("Best fitted population NMSE")
    for ax in (b1, b2):
        ax.set_xlabel("Cut-tail lower bound / target variance")
    f.set_ylabel("Minimum cut bound (NMSE)")
    f.set_xlabel("Depth limit (root-to-leaf edges)")
    cv.lock_reserves()
    panel_a(a, protocol, trees, constructions)
    panel_b([b1, b2], outcomes)
    # one level pitch and one leaf pitch for both trees, from the shared row box
    boxes = {k: cv.axes[k].get_position() for k in "DE"}
    h_pt = min(bx.height for bx in boxes.values()) * CANVAS_H_PT
    w_pt = min(bx.width for bx in boxes.values()) * cv.width_pt
    pitch = (h_pt - 9.0 - 19.0) / 4.0
    leaf = (w_pt - 22.0 - 12.0) / 7.0
    lv_d = panel_tree(d, by_id["matching_000"], "quadratic_matching", expected_depth=3,
                      pitch_pt=pitch, leaf_pt=leaf)
    lv_e = panel_tree(e, by_id["nested_prefix_000"], "nested_prefix_control", expected_depth=4,
                      pitch_pt=pitch, leaf_pt=leaf)
    assert lv_d == [1, 2, 3, 4, 5, 6, 7, 8] and lv_e == [1, 6, 2, 5, 4, 7, 3, 8]
    panel_f(f, cert, constructions)
    problems = cv.save(path, name="figure_scalar_tree_capacity_native", png=png)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

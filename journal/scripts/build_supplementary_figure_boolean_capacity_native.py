#!/usr/bin/env python3
"""Supplementary sheet S14 (ident ``boolean_capacity``) rebuilt as ONE native
full-width canvas from the frozen ``source_data/boolean_theory/`` tables.

The pasted sheet (``figures/supplementary/curated/boolean_capacity.pdf``,
upstream asset ``figure_S43_panels_A-F.pdf``) has no generator anywhere in
the repository, so the in-panel defects the 2026-09-10 visual review
confirmed (``analysis/figure_visual_review_20260910/frozen_S14.json``) could
only be patched by caption.  This builder draws the same six panels, with the
same letters and the same plotted quantities, on :class:`figure_canvas.
NativeCanvas`; every number it draws is read from the frozen tables and
asserted against the study's own summaries.  The old render is left in place
as the reference.

What changed in the drawing (finding -> fix):

* A  columns are in numeric order of ``abcd`` (``a`` the most significant
     bit) with a hairline separator after every four patterns; the binary
     fill is a neutral slate (structure register), not the green series hue;
     the ``1`` / ``0`` key is a two-swatch key in the panel band, not the
     x-axis label.
* B  the obstruction ramp is the journal's single-hue sequential map under a
     power (gamma 0.6) mapping, so 0-0.2 takes about half of the lightness
     span and the nested row (0.105-0.199 on 14 of 15 trees) is legibly
     positive; the ramp starts at a faint tint and every cell has a hairline
     border, so a zero cell is still a cell; the exact-construction marker is
     an open ring with a white fill (a marker, not a ramp value); each row's
     maximum bound is printed at the right edge; the two column groups are
     bracketed ``balanced`` / ``comb`` above the heatmap; the axis label names
     the axis and the marker key sits beside the colourbar.
* C  the three internal nodes of both trees are labelled with their gate in
     the sheet's gate colours (AND green, OR amber, XOR violet); the free-text
     line ``12 coefficients and 6 edges for every tree`` is gone (the caption
     carries it); trees follow the paper's root-lowest orientation.
* D  transposed: families are unrotated rows in the A/B order, minimum exact
     depth is a two-position ordinal axis (2 | 3), and the compatible-tree
     counts are printed with their denominators (``exact``: ``15/15`` of the
     15 trees; ``balanced``: ``3/3`` of the 3 balanced trees) in aligned
     columns outside the box.  D is kept (same letters as the current
     sheet); the redundancy-with-B finding is an editorial decision
     reported, not taken here.
* E  all seven families on the six input pairs as a dot matrix (dot area =
     projection energy, the exact value printed beside each dot, exact zeros
     as open rings), columns grouped ``aligned`` (ab, cd) / ``crossed``; no
     connecting lines; the axis is named ``input pair``; the quantity is
     named on the left edge (``projection energy``) and the area encoding
     has a one-line key (``dot area = energy``) in the raised band.
* F  the three canonical derivatives are drawn from the 101-point grids in
     ``gate_credit_fields.csv`` in the same three gate colours as C and
     labelled directly at their right ends; the zero rule spans the axis;
     the y range is padded to +-1.06 so the end points clear the frame.
* Colour registers on this sheet: green / amber / violet = AND / OR / XOR
  gates only (C, F); the blue sequential ramp = NMSE lower bound only (B);
  slate = truth value 1 (A); ink marks in D and E carry no colour meaning.

Run from the journal directory::

    python3 scripts/build_supplementary_figure_boolean_capacity_native.py
    python3 scripts/figure_canvas.py --audit \
        figures/supplementary/figure_boolean_capacity_native.pdf --strict
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import transforms
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_BASE,
    PT_TITLE,
    SEQ_CMAP,
    Margins,
    NativeCanvas,
    slim_colorbar,
)
from journal_style import ADDRESS_RAMP  # noqa: E402
from native_schematics import Frame  # noqa: E402

ROOT = SCRIPT_DIR.parent
SRC = ROOT / "source_data" / "boolean_theory"
OUT = ROOT / "figures" / "supplementary" / "figure_boolean_capacity_native.pdf"

INK = COLORS["ink"]
MUTE = COLORS["mute"]
EDGE = COLORS["edge"]
GRID = COLORS["grid"]

# ── the sheet's colour registers ─────────────────────────────────────────
GATE_COLOR = {"AND": COLORS["shunting"], "OR": COLORS["local"],
              "XOR": COLORS["oracle"]}
TRUTH_ONE = ADDRESS_RAMP[0]          # slate: truth value 1 (structure)
TRUTH_ZERO = "white"

# Family order of the old sheet (rows of A, B, D, E, top to bottom).
FAMILIES = ["and4", "or4", "parity4", "or_of_ands", "xor_of_ands",
            "and_of_xors", "nested"]
FAMILY_LABEL = {"and4": "AND", "or4": "OR", "parity4": "parity",
                "or_of_ands": "OR(AND)", "xor_of_ands": "XOR(AND)",
                "and_of_xors": "AND(XOR)", "nested": "nested"}
# Column order of the old sheet's B: the three balanced (depth-2) trees,
# then the twelve comb (depth-3) trees.
TREE_ORDER = ["T10", "T11", "T12", "T01", "T02", "T03", "T04", "T05", "T06",
              "T07", "T08", "T09", "T13", "T14", "T15"]
BALANCED = ["T10", "T11", "T12"]
PAIR_ORDER = ["ab", "cd", "ac", "ad", "bc", "bd"]     # aligned, then crossed
ALIGNED = ["ab", "cd"]

TITLE_PAD = 16.0        # one raised title band for every row (group labels,
                        # column headers and keys live in it)
TITLE_PAD_KEYED = 25.0  # row 2's band carries a third line (E's dot key)
                        # between the group labels and the title
BAND_RULE_PT = 3.0      # bracket rule height above the axes top, in points
BAND_TEXT_PT = 5.5      # bracket label baseline above the axes top
BAND_KEY_PT = 15.0      # key-line baseline above the axes top (row 2)

NMSE_VMAX = 2.0 / 3.0   # largest bound in tree_capacity.csv (asserted)
NMSE_GAMMA = 0.6        # power mapping: 0.2/0.667 -> 0.49 of the ramp
RAMP_START = 0.12       # the ramp begins at a faint tint, not at the page


def _csv(name):
    path = SRC / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _rational(s):
    return Fraction(str(s).strip())


# ── panel-band furniture ─────────────────────────────────────────────────
def _band_transform(ax, dy_pt):
    """Data x, axes-fraction y, shifted ``dy_pt`` points upward."""
    base = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    return transforms.offset_copy(base, fig=ax.figure, x=0.0, y=dy_pt,
                                  units="points")


def group_labels(ax, groups):
    """Bracket labels over column groups, in the raised title band.

    ``groups`` is ``[(first_col, last_col, label), ...]`` in data columns.
    """
    rule = _band_transform(ax, BAND_RULE_PT)
    for x0, x1, label in groups:
        # A figure-level artist: it is band furniture, not a data line of
        # the axes (the live overlap audit reads every ax.lines vertex in
        # data coordinates, which a blended transform is not).
        ax.figure.add_artist(Line2D([x0 - 0.42, x1 + 0.42], [1.0, 1.0],
                                    transform=rule, color=EDGE, lw=LW_HAIR,
                                    clip_on=False, solid_capstyle="butt"))
        ax.annotate(label, xy=((x0 + x1) / 2.0, 1.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0.0, BAND_TEXT_PT), textcoords="offset points",
                    ha="center", va="baseline", fontsize=PT_BASE, color=INK,
                    annotation_clip=False)


def column_divider(ax, x, n_rows):
    """A rule between two column groups, over the rows only."""
    ax.plot([x, x], [-0.5, n_rows - 0.5], color=EDGE, lw=LW_REF,
            solid_capstyle="butt", zorder=3.5, clip_on=False)


def category_axes(ax, row_labels, col_labels, *, col_rotation=0):
    """Heatmap-style categorical axes: no spines, no tick marks."""
    ax.set_xticks(range(len(col_labels)), list(col_labels),
                  rotation=col_rotation)
    ax.set_yticks(range(len(row_labels)), list(row_labels))
    ax.tick_params(axis="both", length=0, pad=2.0, labelsize=PT_BASE)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-0.5, len(col_labels) - 0.5)
    ax.set_ylim(len(row_labels) - 0.5, -0.5)     # first row at the top
    ax.grid(False)


# ── A: truth tables ──────────────────────────────────────────────────────
def panel_truth_tables(ax, truth):
    """A: 7 x 16 binary truth tables, patterns in numeric order of abcd."""
    assert len(truth) == 112 and set(truth.family) == set(FAMILIES)
    truth = truth.copy()
    truth["pattern"] = [f"{a}{b}{c}{d}" for a, b, c, d in
                        zip(truth.a, truth.b, truth.c, truth.d)]
    patterns = sorted(truth.pattern.unique())            # 0000 ... 1111
    assert len(patterns) == 16 and patterns[0] == "0000" \
        and patterns[-1] == "1111"
    table = truth.pivot(index="family", columns="pattern",
                        values="target_raw").loc[FAMILIES, patterns]
    matrix = table.to_numpy(dtype=int)
    assert set(np.unique(matrix)) <= {0, 1}
    # The frozen truth values are the formulas the study names.
    bits = np.array([[int(ch) for ch in p] for p in patterns])
    a, b, c, d = bits.T
    expected = {
        "and4": a & b & c & d, "or4": a | b | c | d,
        "parity4": a ^ b ^ c ^ d,
        "or_of_ands": (a & b) | (c & d), "xor_of_ands": (a & b) ^ (c & d),
        "and_of_xors": (a ^ b) & (c ^ d), "nested": a & (b | (c & d)),
    }
    for i, fam in enumerate(FAMILIES):
        np.testing.assert_array_equal(matrix[i], expected[fam])
    # ... and the normalised target is (y - mean) / sd of the raw column.
    for fam in FAMILIES:
        sub = truth[truth.family == fam]
        y = sub.target_raw.to_numpy(float)
        z = (y - y.mean()) / np.sqrt(((y - y.mean()) ** 2).mean())
        np.testing.assert_allclose(sub.target_normalized.to_numpy(float), z,
                                   atol=1e-12)
    n_rows, n_cols = matrix.shape
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                   facecolor=TRUTH_ONE if matrix[i, j]
                                   else TRUTH_ZERO,
                                   edgecolor=GRID, linewidth=LW_HAIR,
                                   zorder=2))
    # A light grouping rule after every four patterns (a, b fixed per group).
    for x in (3.5, 7.5, 11.5):
        ax.plot([x, x], [-0.5, n_rows - 0.5], color=MUTE, lw=LW_HAIR,
                solid_capstyle="butt", zorder=3, clip_on=False)
    ax.plot([-0.5, n_cols - 0.5, n_cols - 0.5, -0.5, -0.5],
            [-0.5, -0.5, n_rows - 0.5, n_rows - 0.5, -0.5], color=EDGE,
            lw=LW_HAIR, zorder=3, clip_on=False)
    category_axes(ax, [FAMILY_LABEL[f] for f in FAMILIES], patterns,
                  col_rotation=90)
    ax.set_xlabel("input pattern abcd")
    # The two-swatch key sits in the raised title band, right of the title.
    handles = [Patch(facecolor=TRUTH_ONE, edgecolor=EDGE, linewidth=LW_HAIR),
               Patch(facecolor=TRUTH_ZERO, edgecolor=EDGE, linewidth=LW_HAIR)]
    ax.legend(handles, ["1", "0"], loc="lower right", bbox_to_anchor=(1.0, 1.0),
              bbox_transform=ax.transAxes, borderaxespad=0.25, frameon=False,
              ncol=2, handlelength=1.0, handleheight=0.85, handletextpad=0.4,
              columnspacing=0.9, fontsize=PT_BASE)
    print(f"[A] {n_rows} families x {n_cols} patterns; ones per family: "
          + ", ".join(f"{FAMILY_LABEL[f]} {int(matrix[i].sum())}"
                      for i, f in enumerate(FAMILIES)))
    return matrix


# ── B: tree-capacity obstruction heatmap ─────────────────────────────────
def _ramp():
    return LinearSegmentedColormap.from_list(
        "journal_seq_faint", SEQ_CMAP(np.linspace(RAMP_START, 1.0, 256)))


def panel_tree_capacity(canvas, ax, capacity, depth_summary):
    """B: normalised-MSE lower bound for all 105 family-tree pairs."""
    assert len(capacity) == 105
    bound = capacity.pivot(index="family", columns="tree_id",
                           values="normalized_mse_lower_bound")
    bound = bound.loc[FAMILIES, TREE_ORDER]
    exact = capacity.pivot(index="family", columns="tree_id",
                           values="exact_construction_verified")
    exact = exact.loc[FAMILIES, TREE_ORDER].to_numpy(dtype=bool)
    zero = capacity.pivot(index="family", columns="tree_id",
                          values="zero_bound_exact")
    zero = zero.loc[FAMILIES, TREE_ORDER].to_numpy(dtype=bool)
    values = bound.to_numpy(dtype=float)
    # The study's own classification: zero bound <=> exact construction, 49.
    np.testing.assert_array_equal(exact, zero)
    np.testing.assert_array_equal(exact, values == 0.0)
    assert int(exact.sum()) == 49
    depth = capacity.pivot(index="family", columns="tree_id",
                           values="depth").loc[FAMILIES, TREE_ORDER]
    assert (depth.nunique(axis=0) == 1).all()
    depth = depth.iloc[0]
    assert all(depth[t] == 2 for t in BALANCED)
    assert all(depth[t] == 3 for t in TREE_ORDER if t not in BALANCED)
    assert values.max() <= NMSE_VMAX + 1e-12
    np.testing.assert_allclose(values.max(), NMSE_VMAX, atol=1e-12)
    # Row facts the caption states.
    row_max = values.max(axis=1)
    per_family = capacity.groupby("family").normalized_mse_lower_bound.max()
    np.testing.assert_allclose(row_max, per_family.loc[FAMILIES].to_numpy(),
                               atol=1e-12)
    nested = values[FAMILIES.index("nested")]
    assert int((nested > 0).sum()) == 14 and nested[TREE_ORDER.index("T15")] == 0
    np.testing.assert_allclose([nested[nested > 0].min(),
                                nested[nested > 0].max()],
                               [0.1046, 0.1992], atol=5e-4)
    for fam in ("or_of_ands", "xor_of_ands", "and_of_xors"):
        row = values[FAMILIES.index(fam)]
        assert row[TREE_ORDER.index("T10")] == 0.0 and (row > 0).sum() == 14
    ds = depth_summary.set_index("family").loc[FAMILIES]
    np.testing.assert_array_equal(exact.sum(axis=1),
                                  ds.exact_compatible_trees.to_numpy())

    cmap = _ramp()
    norm = PowerNorm(gamma=NMSE_GAMMA, vmin=0.0, vmax=NMSE_VMAX)
    n_rows, n_cols = values.shape
    for i in range(n_rows):
        for j in range(n_cols):
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0,
                                   facecolor=cmap(norm(values[i, j])),
                                   edgecolor="white", linewidth=LW_HAIR,
                                   zorder=2))
    # Exact-construction marker: an open ring, white fill, ink outline.
    ys, xs = np.nonzero(exact)
    ax.plot(xs, ys, linestyle="none", marker="o", ms=3.4,
            markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=LW_EDGE, zorder=4)
    column_divider(ax, len(BALANCED) - 0.5, n_rows)
    category_axes(ax, [FAMILY_LABEL[f] for f in FAMILIES],
                  [t.lstrip("T").lstrip("0") for t in TREE_ORDER])
    ax.set_xlabel("labeled binary tree, T index")
    group_labels(ax, [(0, len(BALANCED) - 1, "balanced"),
                      (len(BALANCED), n_cols - 1, "comb")])
    # Row maxima at the right edge: legible without reading colour.
    for i, fam in enumerate(FAMILIES):
        text = "0" if row_max[i] == 0.0 else f"{row_max[i]:.3f}"
        ax.annotate(text, xy=(1.0, i), xycoords=("axes fraction", "data"),
                    xytext=(3.0, 0.0), textcoords="offset points",
                    ha="left", va="center", fontsize=PT_BASE, color=INK,
                    annotation_clip=False)
    ax.annotate("max", xy=(1.0, 1.0), xycoords="axes fraction",
                xytext=(3.0, BAND_TEXT_PT), textcoords="offset points",
                ha="left", va="baseline", fontsize=PT_BASE, color=INK,
                annotation_clip=False)
    mappable = plt_scalar_mappable(cmap, norm)
    cbar = slim_colorbar(canvas.fig, ax, mappable, label="NMSE lower bound",
                         pad_pt=24.0, ticks=[0.0, 0.1, 0.2, 0.4, 0.6])
    cbar.ax.set_yticklabels(["0", "0.1", "0.2", "0.4", "0.6"])
    # Marker key beside the colourbar, in the same raised band.
    cbar.ax.annotate("exact", xy=(0.5, 1.0), xycoords="axes fraction",
                     xytext=(2.5, BAND_TEXT_PT), textcoords="offset points",
                     ha="left", va="baseline", fontsize=PT_BASE, color=INK,
                     annotation_clip=False)
    cbar.ax.plot([0.5], [1.0], transform=transforms.offset_copy(
        cbar.ax.transAxes, fig=canvas.fig, x=-0.5, y=BAND_TEXT_PT + 2.6,
        units="points"), linestyle="none", marker="o", ms=3.4,
        markerfacecolor="white", markeredgecolor=INK,
        markeredgewidth=LW_EDGE, clip_on=False, zorder=6)
    print("[B] row maxima: " + ", ".join(
        f"{FAMILY_LABEL[f]} {row_max[i]:.4f}" for i, f in enumerate(FAMILIES))
        + f"; exact rings {int(exact.sum())}; nested positive on "
          f"{int((nested > 0).sum())}/15 trees "
          f"({nested[nested > 0].min():.4f}-{nested[nested > 0].max():.4f})")
    return values, exact


def plt_scalar_mappable(cmap, norm):
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


# ── C: two gate trees, equal resources, different minimum depth ──────────
def panel_gate_trees(ax, depth_summary):
    """C: OR(AND) and nested: two AND gates + one OR gate each, depth 2 / 3."""
    ds = depth_summary.set_index("family")
    for fam in ("or_of_ands", "nested"):
        assert int(ds.loc[fam, "and_gates"]) == 2
        assert int(ds.loc[fam, "or_gates"]) == 1
        assert int(ds.loc[fam, "xor_gates"]) == 0
    assert int(ds.loc["or_of_ands", "minimum_exact_depth"]) == 2
    assert int(ds.loc["nested", "minimum_exact_depth"]) == 3
    assert ds.loc["or_of_ands", "formula"] == "(a AND b) OR (c AND d)"
    assert ds.loc["nested", "formula"] == "a AND (b OR (c AND d))"

    f = Frame(ax)
    W, H = f.w_pt, f.h_pt
    head_pt = 9.0                   # tree name band at the top
    foot_pt = 13.0                  # 'minimum depth' band at the bottom
    leaf_pt = 8.0                   # leaf letter band above the squares
    levels = 4                      # root, two gate levels, leaves (nested)
    span = H - head_pt - foot_pt - leaf_pt - 6.0
    step = span / (levels - 1)
    cell_w = W / 2.0
    R = 2.6                         # gate disc radius, pt
    LEAF = 2.2                      # leaf square half-side, pt

    def P(cx_pt, dx_pt, level):
        """Frame xy of a node ``dx_pt`` right of the cell centre at ``level``
        (0 = root at the bottom, higher = further from the root)."""
        return (f.fx(cx_pt + dx_pt), f.fy(foot_pt + 3.0 + level * step))

    def edge(p, q):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=EDGE, lw=f.lw(LW_EDGE),
                solid_capstyle="round", zorder=2)

    def gate(p, name, *, side=1):
        f.disc(p, R, fill=GATE_COLOR[name], edge="white", lw=LW_HAIR,
               zorder=4)
        f.text(f._off(p, side * (R + 2.2), 0.0), name, size=PT_BASE,
               color=GATE_COLOR[name], ha="left" if side > 0 else "right",
               va="center")

    def leaf(p, name):
        ax.add_patch(Rectangle((p[0] - f.fx(LEAF), p[1] - f.fy(LEAF)),
                               2 * f.fx(LEAF), 2 * f.fy(LEAF),
                               facecolor="white", edgecolor=INK,
                               linewidth=LW_EDGE, zorder=4, clip_on=False))
        f.text(f._off(p, 0.0, LEAF + 1.8), name, size=PT_BASE, color=INK,
               va="bottom")

    # left cell: OR(AND) = (a AND b) OR (c AND d), depth 2
    cx = cell_w / 2.0
    root = P(cx, 0.0, 0)
    l_and = P(cx, -20.0, 1)
    r_and = P(cx, 20.0, 1)
    leaves = [P(cx, x, 2) for x in (-31.0, -9.0, 9.0, 31.0)]
    for p, q in ((root, l_and), (root, r_and), (l_and, leaves[0]),
                 (l_and, leaves[1]), (r_and, leaves[2]), (r_and, leaves[3])):
        edge(p, q)
    gate(root, "OR")
    gate(l_and, "AND", side=-1)
    gate(r_and, "AND")
    for p, name in zip(leaves, "abcd"):
        leaf(p, name)
    f.text((f.fx(cx), 1.0 - f.fy(1.0)), "OR(AND)", size=PT_BASE, color=INK,
           va="top")
    f.text((f.fx(cx), f.fy(1.0)), "minimum depth 2", size=PT_BASE,
           color=INK, va="bottom")

    # right cell: nested = a AND (b OR (c AND d)), depth 3
    cx = cell_w + cell_w / 2.0
    root = P(cx, 0.0, 0)
    a = P(cx, -22.0, 1)
    n_or = P(cx, 10.0, 1)
    b = P(cx, -8.0, 2)
    n_and = P(cx, 24.0, 2)
    c = P(cx, 12.0, 3)
    d = P(cx, 36.0, 3)
    for p, q in ((root, a), (root, n_or), (n_or, b), (n_or, n_and),
                 (n_and, c), (n_and, d)):
        edge(p, q)
    gate(root, "AND")
    gate(n_or, "OR", side=-1)
    gate(n_and, "AND", side=-1)
    for p, name in ((a, "a"), (b, "b"), (c, "c"), (d, "d")):
        leaf(p, name)
    f.text((f.fx(cx), 1.0 - f.fy(1.0)), "nested", size=PT_BASE, color=INK,
           va="top")
    f.text((f.fx(cx), f.fy(1.0)), "minimum depth 3", size=PT_BASE,
           color=INK, va="bottom")
    # The cell's own vertical rule between the two trees.
    ax.plot([0.5, 0.5], [f.fy(foot_pt), 1.0 - f.fy(head_pt)], color=GRID,
            lw=LW_HAIR, zorder=1)
    print("[C] OR(AND): 2 AND + 1 OR, depth 2; nested: 2 AND + 1 OR, depth 3")


# ── D: minimum exact depth strip with compatible-tree counts ─────────────
def panel_depth_strip(ax, depth_summary, target_summary, capacity):
    """D: minimum exact depth (2 | 3) per family; counts printed outside."""
    ds = depth_summary.set_index("family").loc[FAMILIES]
    ts = target_summary.set_index("family").loc[FAMILIES]
    depth = ds.minimum_exact_depth.to_numpy(int)
    n_exact = ds.exact_compatible_trees.to_numpy(int)
    n_bal = ds.exact_compatible_balanced_trees.to_numpy(int)
    np.testing.assert_array_equal(depth, ts.minimum_exact_depth.to_numpy(int))
    np.testing.assert_array_equal(n_exact,
                                  ts.exact_compatible_trees.to_numpy(int))
    np.testing.assert_array_equal(
        n_bal, ts.exact_compatible_balanced_trees.to_numpy(int))
    # Recount both columns and the minimum depth from the 105 pairs.
    ok = capacity[capacity.exact_construction_verified.astype(bool)]
    recount = ok.groupby("family").tree_id.count().reindex(FAMILIES).fillna(0)
    np.testing.assert_array_equal(n_exact, recount.to_numpy(int))
    recount_bal = (ok[ok.tree_id.isin(BALANCED)].groupby("family").tree_id
                   .count().reindex(FAMILIES).fillna(0))
    np.testing.assert_array_equal(n_bal, recount_bal.to_numpy(int))
    min_depth = ok.groupby("family").depth.min().reindex(FAMILIES)
    np.testing.assert_array_equal(depth, min_depth.to_numpy(int))
    assert list(depth) == [2, 2, 2, 2, 2, 2, 3]
    assert list(n_exact) == [15, 15, 15, 1, 1, 1, 1]
    assert list(n_bal) == [3, 3, 3, 1, 1, 1, 0]

    n_rows = len(FAMILIES)
    ys = np.arange(n_rows)
    xs = depth - 2                                     # ordinal 0 | 1
    for y in ys:                                       # row hairlines
        ax.plot([-0.5, 1.5], [y, y], color=GRID, lw=LW_HAIR, zorder=1,
                solid_capstyle="butt")
    ax.plot(xs, ys, linestyle="none", marker="o", ms=MARKER_MS,
            markerfacecolor=INK, markeredgecolor="white",
            markeredgewidth=LW_HAIR, zorder=4)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks([0, 1], ["2", "3"])
    ax.set_yticks(ys, [FAMILY_LABEL[f] for f in FAMILIES])
    ax.tick_params(axis="y", length=0, pad=2.0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("minimum exact depth")
    # Two aligned count columns outside the box; headers in the title band.
    # Each count is printed with its denominator (15 trees; 3 balanced
    # trees), as the old sheet's ``15/15`` / ``1/15``.
    cols = ((13.0, "exact", n_exact, len(TREE_ORDER)),
            (41.0, "balanced", n_bal, len(BALANCED)))
    for dx, head, counts, denom in cols:
        ax.annotate(head, xy=(1.0, 1.0), xycoords="axes fraction",
                    xytext=(dx, BAND_TEXT_PT), textcoords="offset points",
                    ha="center", va="baseline", fontsize=PT_BASE, color=INK,
                    annotation_clip=False)
        for y, n in zip(ys, counts):
            ax.annotate(f"{int(n)}/{denom}", xy=(1.0, y),
                        xycoords=("axes fraction", "data"),
                        xytext=(dx, 0.0), textcoords="offset points",
                        ha="center", va="center", fontsize=PT_BASE,
                        color=INK, annotation_clip=False)
    print("[D] depth " + ", ".join(
        f"{FAMILY_LABEL[f]} {d} ({e}/15, {b}/3 balanced)"
        for f, d, e, b in zip(FAMILIES, depth, n_exact, n_bal)))


# ── E: post hoc projection energy on the six input pairs ─────────────────
def panel_projection_energy(ax, energy, posthoc):
    """E: 7 families x 6 pairs; dot area = energy, exact value printed."""
    assert len(energy) == 98
    assert posthoc["full_domain_conditional_expectation_crosschecks"] == 98
    assert energy.independent_conditional_mean_check_exact.all()
    # Exact rationals agree with the float column everywhere.
    rat = energy.projection_energy_normalized_rational.map(_rational)
    np.testing.assert_allclose(
        energy.projection_energy_normalized.to_numpy(float),
        rat.map(float).to_numpy(), atol=1e-12)
    # Parity: zero on all 14 proper subsets (the caption's claim).
    par = energy[energy.family == "parity4"]
    assert len(par) == 14 and (par.projection_energy_normalized == 0).all()
    assert posthoc["parity_all_14_proper_subsets_exactly_zero"] is True
    pairs = energy[energy.subset_size == 2]
    assert len(pairs) == 42
    table = pairs.pivot(index="family", columns="subset",
                        values="projection_energy_normalized")
    table = table.loc[FAMILIES, PAIR_ORDER]
    values = table.to_numpy(float)
    xo = pairs[pairs.family == "xor_of_ands"]
    assert (xo.projection_energy_normalized_rational == "1/5").all()
    assert posthoc["xor_of_ands_all_six_pair_projection_energies_exactly"] \
        == "1/5"
    ax_row = values[FAMILIES.index("and_of_xors")]
    np.testing.assert_allclose(ax_row[:2], 1.0 / 3.0, atol=1e-12)
    np.testing.assert_array_equal(ax_row[2:], 0.0)
    nested = values[FAMILIES.index("nested")]
    np.testing.assert_allclose([nested.min(), nested.max()],
                               [3.0 / 55.0, 43.0 / 55.0], atol=1e-12)

    n_rows, n_cols = values.shape
    vmax = values.max()
    # Column layout, in column units (one column = the axes width / 6, about
    # 26.7 pt): the dot at j + DOT_DX, the printed value left-aligned at
    # j + LABEL_DX.  With the largest dot D_MAX pt across, the value in the
    # cd column ends >= 2 pt before the aligned/crossed divider at x = 1.5,
    # the largest ac dot starts >= 2 pt after it, and every dot clears its
    # own label; measured on the PDF, not assumed.
    DOT_DX, LABEL_DX, D_MAX = -0.295, -0.105, 7.5
    for y in range(n_rows):
        ax.plot([-0.5, n_cols - 0.5], [y, y], color=GRID, lw=LW_HAIR,
                zorder=1, solid_capstyle="butt")
    for i in range(n_rows):
        for j in range(n_cols):
            v = values[i, j]
            x_dot = j + DOT_DX
            if v == 0.0:
                ax.plot([x_dot], [i], linestyle="none", marker="o", ms=3.0,
                        markerfacecolor="white", markeredgecolor=INK,
                        markeredgewidth=LW_EDGE, zorder=4)
                label = "0"
            else:
                d = D_MAX * np.sqrt(v / vmax)
                ax.scatter([x_dot], [i], s=d * d, color=INK, zorder=4,
                           edgecolors="none")
                label = f"{v:.2f}"
            ax.text(j + LABEL_DX, i, label, ha="left", va="center",
                    fontsize=PT_BASE, color=INK, zorder=5)
    column_divider(ax, len(ALIGNED) - 0.5, n_rows)
    category_axes(ax, [FAMILY_LABEL[f] for f in FAMILIES], PAIR_ORDER)
    ax.set_xlabel("input pair")
    # The plotted quantity is named on the left edge (the old sheet's y-axis
    # label, shortened to fit the axes height); the area encoding has its
    # own one-line key in the raised band, above the group labels.
    ax.set_ylabel("projection energy", labelpad=2.0)
    group_labels(ax, [(0, len(ALIGNED) - 1, "aligned"),
                      (len(ALIGNED), n_cols - 1, "crossed")])
    ax.annotate("dot area = energy", xy=(1.0, 1.0), xycoords="axes fraction",
                xytext=(0.0, BAND_KEY_PT), textcoords="offset points",
                ha="right", va="baseline", fontsize=PT_BASE, color=INK,
                annotation_clip=False)
    print("[E] pair energies (rows AND..nested; cols " + " ".join(PAIR_ORDER)
          + "):\n" + "\n".join("     " + " ".join(f"{v:.3f}" for v in row)
                               for row in values))
    return values


# ── F: canonical conditional credit ──────────────────────────────────────
def panel_gate_derivatives(ax, fields, corners):
    """F: dF/du versus the other branch output v for AND, OR, XOR."""
    assert len(fields) == 30603 and set(fields.gate) == {"AND", "OR", "XOR"}
    expect = {"AND": lambda v: v, "OR": lambda v: 1.0 - v,
              "XOR": lambda v: 1.0 - 2.0 * v}
    ends = {}
    for gate in ("AND", "OR", "XOR"):
        sub = fields[fields.gate == gate]
        # dF/du depends on v only: one value per v across all 101 u.
        by_v = sub.groupby("right").d_output_d_left
        assert int(by_v.nunique().max()) == 1
        v = np.array(sorted(sub.right.unique()), float)
        assert len(v) == 101 and v[0] == 0.0 and v[-1] == 1.0
        dfdu = by_v.first().loc[v].to_numpy(float)
        np.testing.assert_allclose(dfdu, expect[gate](v), atol=1e-12)
        # ... and the twelve Boolean corners agree with the grid ends.
        corner = corners[corners.gate == gate]
        for _, row in corner.iterrows():
            np.testing.assert_allclose(row.d_output_d_left,
                                       expect[gate](row.right), atol=1e-12)
        ax.plot(v, dfdu, color=GATE_COLOR[gate], lw=LW_DATA, zorder=3,
                solid_capstyle="butt")
        ends[gate] = float(dfdu[-1])
    ax.axhline(0.0, color=MUTE, ls=(0, (2.2, 1.8)), lw=LW_REF, zorder=1.5)
    for gate, y in ends.items():
        ax.annotate(gate, xy=(1.0, y), xycoords="data", xytext=(3.0, 0.0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=PT_BASE, color=GATE_COLOR[gate],
                    annotation_clip=False)
    ax.set_xlim(0.0, 1.0)
    # Padded so the XOR end (1, -1) and the OR end on the zero rule do not
    # merge with the bottom spine / the dashed rule; the ticks stay at +-1.
    ax.set_ylim(-1.06, 1.06)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_yticks([-1.0, 0.0, 1.0], ["−1", "0", "1"])
    ax.set_xlabel("other branch output v")
    ax.set_ylabel("∂F/∂u")
    print(f"[F] dF/du at v = 1: " + ", ".join(f"{g} {y:+.0f}"
                                              for g, y in ends.items()))


# ── the canvas ───────────────────────────────────────────────────────────
CANVAS_H_PT = 493.0     # 484 + the 9 pt of row 2's taller band; the audit
                        # caps the page at aspect 1.05 (493.7 pt)
HGUTTER_PT = 38.0
VGUTTER_PT = 64.5       # keeps every row's axes box at 100 pt
# left: the family tick labels (36.6 pt) + E's rotated axis label + pad,
# so the column lock leaves column 0 its full module width.
MARGINS = Margins(left=55.0, right=10.0, top=32.0, bottom=32.0)


def build(path: Path = OUT):
    truth = _csv("truth_tables.csv")
    capacity = _csv("tree_capacity.csv")
    depth_summary = _csv("depth_summary.csv")
    target_summary = _csv("target_summary.csv")
    energy = _csv("proper_subtree_projection_energy.csv")
    fields = _csv("gate_credit_fields.csv")
    corners = _csv("gate_derivative_corners.csv")
    import json
    posthoc = json.loads((SRC / "posthoc_projection_validation.json")
                         .read_text())
    report = json.loads((SRC / "report.json").read_text())
    assert report["exact_constructions"] == 49
    assert report["family_tree_pairs"] == 105 and report["families"] == 7
    assert report["balanced_trees"] == 3 and report["depth_3_trees"] == 12

    canvas = NativeCanvas(CANVAS_H_PT / 72.0, 3, hgutter_pt=HGUTTER_PT,
                          vgutter_pt=VGUTTER_PT, margins=MARGINS, letter_clearance=True)
    ax_a = canvas.panel("A", 0, 0, 5, title="Seven exact Boolean truth tables")
    ax_b = canvas.panel("B", 0, 5, 7, title="All trees: regression obstruction")
    ax_c = canvas.panel("C", 1, 0, 5, schematic=True,
                        title="Equal resources; different minimum depth")
    ax_d = canvas.panel("D", 1, 5, 7, title="Associative gates are structure controls")
    ax_e = canvas.panel("E", 2, 0, 5, title="Post hoc: target information in a pair")
    ax_f = canvas.panel("F", 2, 5, 7, grid="y", title="Canonical conditional credit")
    # One raised title band for every row: group labels, column headers and
    # the swatch / marker keys live between the axes top and the title.
    for ax in (ax_a, ax_b, ax_c, ax_d):
        ax.set_title(ax.get_title(), fontsize=PT_TITLE, color=INK,
                     pad=TITLE_PAD, fontweight="normal")
    # Row 2's band has a third line (E's ``dot area = energy`` key between
    # the group labels and the title); F shares the row's title height.
    for ax in (ax_e, ax_f):
        ax.set_title(ax.get_title(), fontsize=PT_TITLE, color=INK,
                     pad=TITLE_PAD_KEYED, fontweight="normal")
    # D's count columns and F's right-end labels are drawn as artists.
    canvas.declare_reserve("D", right=58.0)
    canvas.declare_reserve("F", right=20.0)

    panel_truth_tables(ax_a, truth)
    panel_tree_capacity(canvas, ax_b, capacity, depth_summary)
    panel_depth_strip(ax_d, depth_summary, target_summary, capacity)
    panel_projection_energy(ax_e, energy, posthoc)
    panel_gate_derivatives(ax_f, fields, corners)
    canvas.lock_reserves()           # settle the boxes before drawing in points
    panel_gate_trees(ax_c, depth_summary)

    problems = canvas.save(path, name="figure_boolean_capacity_native",
                           png=False)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

#!/usr/bin/env python3
"""Supplementary sheet ``boolean_learning`` (S15) rebuilt as one native canvas.

The frozen render (``figures/supplementary/curated/boolean_learning.pdf``)
had no native builder, so its verified in-panel defects could not be fixed
by the paste layer.  This builder reads ONLY the frozen tables under
``source_data/boolean_morphology/`` and draws the same eight panels with the
same plotted quantities on the paper's :class:`figure_canvas.NativeCanvas`:

* A-D  clean population NMSE, twenty-seed means, seven families x four
       trees, for Adam/SGD x exact/broadcast at the development-selected
       rates (``condition_summary.csv``; the means are recomputed from the
       2,240 per-seed endpoints of ``selected_endpoints.csv``).  Cells are
       vector rectangles with white hairline separators; ONE colour key
       stands in the right margin beside the whole 2 x 2 block, so the four
       maps share one axes width and B registers with D.
* E    the two prespecified Adam XOR-of-AND contrasts
       (``paired_primary_contrasts.csv``, ``primary_contrasts.csv``): twenty
       paired seed differences as a fan, the mean as a short rule with the
       Bonferroni 97.5 % whisker, the 0.01 margin amber dotted.  The grouping
       subpanel has a broken y axis: a labelled strip at the foot carries 0
       and the 0.01 margin, the upper segment 0.50-0.68 carries the data.
* F    accuracy and balanced accuracy versus NMSE for the 112 conditions,
       axes tight to the data, the ceiling population counted in-panel, the
       AND/OR constant-majority references from ``target_normalization.csv``.
* G    same-rate broadcast-minus-exact contrasts
       (``paired_same_rate_contrasts.csv``, ``same_rate_contrast_summary.csv``)
       on a logarithmic rate axis with every seed drawn; the SGD subpanel is
       symmetric-log so the -0.94..0.61 seed tails at rate 0.003 are visible
       while the 0.01 margin and zero still separate.
* H    broadcast gradient cosine on the aligned tree
       (``trajectory_summary.csv``; means recomputed from the 17,920 per-seed
       rows of ``selected_trajectories.csv``) on a logarithmic update axis
       from update 1, update 0 as a pip outside a broken axis, markers at
       every checkpoint, bands the recorded pointwise 95 % intervals.

Colour meanings on this sheet (one meaning per hue): green = the XOR-of-AND
task series (E, G, H); violet = the parity series (H); amber = a reference
quantity (0.01 margin in E and G, cosine 1 in H, constant-majority stars in
F); grey = accuracy and blue = balanced accuracy in F; mute = zero.  Line
style and marker fill name the optimizer in H (solid filled Adam, dashed
open SGD).  Every printed or plotted number is asserted against the table it
comes from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter, NullLocator

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from figure_canvas import (  # noqa: E402
    COLORS, ERR_CAPSIZE, LW_DATA, LW_EDGE, LW_ERR, LW_HAIR, LW_REF, MARKER_MS,
    PT_BASE, SEED_ALPHA, SEED_MS, SEQ_CMAP, Margins, NativeCanvas)

ROOT = SCRIPT_DIR.parent
SOURCE = ROOT / "source_data" / "boolean_morphology"
OUT = ROOT / "figures" / "supplementary" / "figure_boolean_learning_native.pdf"

FAMILIES = ["and4", "or4", "parity4", "or_of_ands", "xor_of_ands",
            "and_of_xors", "nested"]
FAMILY_NAMES = ["AND", "OR", "parity", "OR(AND)", "XOR(AND)", "AND(XOR)",
                "nested"]
TREES = ["balanced_ab_cd", "balanced_ac_bd", "balanced_ad_bc", "comb_a_b_cd"]
TREE_LABELS = ["ab|cd", "ac|bd", "ad|bc", "a|(b|cd)"]
CHECKPOINTS = [0, 1, 16, 64, 256, 512, 1024, 2048]
RATES = [0.003, 0.01, 0.03]
MARGIN = 0.01

XOR = COLORS["shunting"]      # the XOR-of-AND series (E, G, H)
PARITY = COLORS["oracle"]     # the parity series (H)
REF = COLORS["local"]         # every reference quantity (margins, cosine 1, majority)
ACC = COLORS["point_mlp"]     # accuracy (F)
BAL = COLORS["additive"]      # balanced accuracy (F)
MUTE = COLORS["mute"]
INK = COLORS["ink"]
EDGE = COLORS["edge"]

MEAN_HALF_PT = 4.0            # half-width of the mean rule, in points
BAND_ALPHA = 0.16
DOTTED = (1.0, 1.8)


def csv(name):
    path = SOURCE / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


# ── glyphs shared by E and G ──────────────────────────────────────────────
def fan(ax, x, seeds, color, *, log_x=False, half=0.16):
    """Twenty per-seed values as a jittered fan behind the mean."""
    seeds = np.asarray(seeds, float)
    jit = np.linspace(-half, half, len(seeds))
    xs = x * 10.0 ** (jit * 0.35) if log_x else x + jit
    ax.plot(xs, seeds, linestyle="none", marker="o", markersize=SEED_MS,
            markerfacecolor=color, markeredgecolor="none", alpha=SEED_ALPHA,
            zorder=2.0)


def mean_rule(ax, x, mean, lo, hi, color, *, log_x=False):
    """Mean as a short horizontal rule with a capped whisker, so an interval
    narrower than a marker still resolves (a diamond swallowed the 0.034-wide
    grouping interval in the frozen render)."""
    ax.errorbar([x], [mean], yerr=[[mean - lo], [hi - mean]], fmt="none",
                ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                capthick=LW_ERR, zorder=3.5)
    # the rule is drawn in points around x so it has the same width on a
    # linear and a logarithmic axis
    ax.annotate("", xy=(x, mean), xycoords="data", xytext=(-MEAN_HALF_PT, 0.0),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color=color, lw=LW_DATA,
                                shrinkA=0, shrinkB=0), zorder=4.0)
    ax.annotate("", xy=(x, mean), xycoords="data", xytext=(MEAN_HALF_PT, 0.0),
                textcoords="offset points",
                arrowprops=dict(arrowstyle="-", color=color, lw=LW_DATA,
                                shrinkA=0, shrinkB=0), zorder=4.0)


def reference(ax, y, *, style="dotted", color=REF, zorder=1.2):
    """A full-width reference rule.  Drawn between the fixed x limits (call
    after ``set_xlim``) rather than as an axhline, so its two vertices sit at
    the axes edges and never under a label placed at the panel centre."""
    x0, x1 = ax.get_xlim()
    kw = dict(color=color, lw=LW_REF, zorder=zorder, solid_capstyle="butt")
    if style == "dotted":
        kw["dashes"] = DOTTED
    ax.plot([x0, x1], [y, y], **kw)


def axis_break(ax, *, axis, lo, hi, size_pt=2.2):
    """White out a spine between ``lo`` and ``hi`` (data units) and draw the
    two diagonal break marks; ``axis`` is 'y' (left spine) or 'x' (bottom).
    Everything is drawn in data coordinates (clip off) so the live overlap
    audit, which reads every line vertex through ``transData``, sees the
    marks where they really are -- on the spine, not at the panel centre."""
    fig = ax.figure
    bb = ax.get_window_extent()
    w_pt = bb.width * 72.0 / fig.dpi
    h_pt = bb.height * 72.0 / fig.dpi
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    if axis == "y":
        assert ax.get_xscale() == "linear"
        dx = size_pt / w_pt * (x1 - x0)
        ax.add_patch(Rectangle((x0 - 1.6 * dx, lo), 3.2 * dx, hi - lo,
                               facecolor="white", edgecolor="none", zorder=6,
                               clip_on=False))
        for y in (lo, hi):
            dy = 0.35 * (hi - lo)
            ax.plot([x0 - dx, x0 + dx], [y - dy, y + dy], color=EDGE,
                    lw=LW_EDGE, zorder=7, clip_on=False, solid_capstyle="butt")
    else:
        assert ax.get_yscale() == "linear"
        dy = size_pt / h_pt * (y1 - y0)
        ax.add_patch(Rectangle((lo, y0 - 1.6 * dy), hi - lo, 3.2 * dy,
                               facecolor="white", edgecolor="none", zorder=6,
                               clip_on=False))
        for x in (lo, hi):
            if ax.get_xscale() == "log":
                f = 0.35 * np.log10(hi / lo)
                xa, xb = x * 10.0 ** (-f), x * 10.0 ** f
            else:
                f = 0.35 * (hi - lo)
                xa, xb = x - f, x + f
            ax.plot([xa, xb], [y0 - dy, y0 + dy], color=EDGE, lw=LW_EDGE,
                    zorder=7, clip_on=False, solid_capstyle="butt")


# ── A-D: the four heatmaps ────────────────────────────────────────────────
def heatmaps(cv, axes, summary, endpoints, rates):
    assert len(summary) == 112 and len(endpoints) == 2240
    # the recorded means are the means of the twenty per-seed endpoints
    recomputed = (endpoints.groupby(["family", "tree", "optimizer", "rule"])
                  .population_nmse.agg(["mean", "size"]).reset_index())
    merged = summary.merge(recomputed, on=["family", "tree", "optimizer", "rule"])
    assert len(merged) == 112 and (merged["size"] == 20).all()
    np.testing.assert_allclose(merged["mean"], merged.mean_population_nmse,
                               rtol=0, atol=1e-12)
    values = summary.mean_population_nmse.to_numpy(float)
    assert np.isfinite(values).all() and values.min() > 1e-4 and values.max() <= 1.0
    # one decade of headroom below the smallest cell keeps the ~0.001 cells
    # a visible tint rather than white, so the separators read in every row
    norm = LogNorm(vmin=1e-4, vmax=1.0)
    mesh = None
    for ax, (opt, rule) in zip(axes, [("adam", "exact"), ("adam", "broadcast"),
                                      ("sgd", "exact"), ("sgd", "broadcast")]):
        sub = summary[summary.optimizer.eq(opt) & summary.rule.eq(rule)]
        z = sub.pivot(index="family", columns="tree",
                      values="mean_population_nmse").loc[FAMILIES, TREES]
        assert z.shape == (7, 4) and np.isfinite(z.to_numpy()).all()
        # the rate each map was fitted at is the development-selected one
        rate_rows = endpoints[endpoints.optimizer.eq(opt) & endpoints.rule.eq(rule)].rate.unique()
        assert len(rate_rows) == 1 and float(rate_rows[0]) == rates[opt][rule]
        mesh = ax.pcolormesh(np.arange(5) - 0.5, np.arange(8) - 0.5,
                             z.to_numpy(), cmap=SEQ_CMAP, norm=norm,
                             edgecolors="white", linewidth=LW_HAIR,
                             shading="flat", rasterized=False)
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(6.5, -0.5)
        ax.set_xticks(range(4), TREE_LABELS)
        # the row names stand once per row, on the left maps; B and D share
        # them by the row lock (same y0, same height), which keeps the gutter
        # free and every map the same width
        ax.set_yticks(range(7), FAMILY_NAMES if opt == "adam" and rule == "exact"
                      or opt == "sgd" and rule == "exact" else [])
        ax.tick_params(axis="both", length=0, pad=2.2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        title = (f"{'Adam' if opt == 'adam' else 'SGD'}, "
                 f"{rule} credit, rate {rates[opt][rule]:g}")
        ax.set_title(title, fontsize=ax.title.get_fontsize(), color=INK,
                     pad=3.0, fontweight="normal")
    return mesh, norm


def colour_rail(cv, mesh, top_ax, bottom_ax):
    """One key for A-D in the right outer margin, spanning the 2 x 2 block."""
    top = top_ax.get_position()
    bottom = bottom_ax.get_position()
    x0 = top.x1 + 3.0 / cv.width_pt
    cax = cv.fig.add_axes([x0, bottom.y0, 4.5 / cv.width_pt, top.y1 - bottom.y0])
    cbar = cv.fig.colorbar(mesh, cax=cax)
    cbar.outline.set_linewidth(LW_EDGE)
    cbar.outline.set_edgecolor(EDGE)
    cbar.set_ticks([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    cbar.set_ticklabels(["0.0001", "0.001", "0.01", "0.1", "1"])
    cbar.minorticks_off()
    cbar.ax.tick_params(labelsize=PT_BASE, width=LW_EDGE, length=2.2, pad=1.2,
                        color=EDGE, labelcolor=INK)
    cbar.ax.annotate("clean\nNMSE", xy=(0.0, 0.0), xycoords="axes fraction",
                     xytext=(-1.0, -3.0), textcoords="offset points",
                     ha="left", va="top", fontsize=PT_BASE, color=INK,
                     linespacing=1.1, annotation_clip=False)
    return cax


# ── E: the two prespecified contrasts ─────────────────────────────────────
E_STRIP_TOP = 0.02            # data range of the foot strip
E_STRIP_V = 0.028             # its virtual height
E_GAP_V = 0.014               # the break
E_UPPER = (0.50, 0.68)        # data range of the upper segment


def e_map(y):
    """Broken-axis mapping for the grouping subpanel (data -> virtual)."""
    y = np.asarray(y, float)
    low = y * (E_STRIP_V / E_STRIP_TOP)
    high = y - E_UPPER[0] + E_STRIP_V + E_GAP_V
    return np.where(y <= E_STRIP_TOP, low, high)


def panel_e(ax_group, ax_credit, primary, pairs):
    assert len(primary) == 2 and len(pairs) == 40
    rows = {}
    for key in ("crossed_minus_compatible_exact", "broadcast_minus_exact_compatible"):
        r = primary[primary.contrast.eq(key)].iloc[0]
        z = pairs[pairs.contrast.eq(key)].sort_values("seed")
        seeds = z.difference.to_numpy(float)
        assert len(seeds) == 20 and z.seed.is_unique
        np.testing.assert_allclose(seeds, z.comparator_nmse - z.reference_nmse,
                                   rtol=0, atol=1e-12)
        np.testing.assert_allclose(seeds.mean(), r.mean_difference, rtol=0, atol=1e-12)
        assert r.ci975_low < r.mean_difference < r.ci975_high
        assert r.ci975_low <= r.ci95_low and r.ci95_high <= r.ci975_high
        assert float(r.margin_nmse) == MARGIN and bool(r.primary)
        passes = bool(r.ci975_low > 0 and r.mean_difference >= MARGIN)
        assert passes == bool(r.passes_adjusted_interval_and_mean_margin)
        rows[key] = (r, seeds, passes)
    # grouping: broken y axis, strip at the foot for 0 and the margin
    r, seeds, passes = rows["crossed_minus_compatible_exact"]
    assert passes and seeds.min() > E_UPPER[0] and seeds.max() < E_UPPER[1]
    assert r.ci975_low > E_UPPER[0] and r.ci975_high < E_UPPER[1]
    ax = ax_group
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(0.0, float(e_map(E_UPPER[1])))
    fan(ax, 0.0, e_map(seeds), XOR)
    mean_rule(ax, 0.0, float(e_map(r.mean_difference)), float(e_map(r.ci975_low)),
              float(e_map(r.ci975_high)), XOR)
    reference(ax, float(e_map(MARGIN)))
    ax.annotate("0.01 margin", xy=(1.0, float(e_map(MARGIN))),
                xycoords=("axes fraction", "data"), xytext=(-1.5, 1.0),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=PT_BASE, color=REF)
    ticks = [0.0, 0.50, 0.55, 0.60, 0.65]
    ax.set_yticks(list(e_map(ticks)), ["0", "0.50", "0.55", "0.60", "0.65"])
    ax.set_xticks([0.0], ["passes"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("NMSE difference")
    print(f"[E] grouping: mean {r.mean_difference:.6f} "
          f"[{r.ci975_low:.6f}, {r.ci975_high:.6f}] seeds {seeds.min():.4f}-{seeds.max():.4f}")
    # credit: linear axis, own range
    r, seeds, passes = rows["broadcast_minus_exact_compatible"]
    assert not passes and r.ci975_low > 0 and r.mean_difference < MARGIN
    ax = ax_credit
    ylim = (-0.0006, 0.0125)
    assert seeds.min() > ylim[0] and seeds.max() < ylim[1]
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(*ylim)
    fan(ax, 0.0, seeds, XOR)
    mean_rule(ax, 0.0, float(r.mean_difference), float(r.ci975_low),
              float(r.ci975_high), XOR)
    reference(ax, MARGIN)
    reference(ax, 0.0, style="solid", color=MUTE, zorder=1.0)
    ax.annotate("0.01 margin", xy=(1.0, MARGIN), xycoords=("axes fraction", "data"),
                xytext=(-1.5, 1.0), textcoords="offset points", ha="right",
                va="bottom", fontsize=PT_BASE, color=REF)
    ax.text(-0.28, ylim[1], "own y scale", ha="left", va="top",
            fontsize=PT_BASE, color=MUTE, zorder=5)
    ax.set_yticks([0.0, 0.005, 0.010], ["0.000", "0.005", "0.010"])
    ax.set_xticks([0.0], ["below margin"])
    ax.tick_params(axis="x", length=0)
    print(f"[E] credit: mean {r.mean_difference:.6f} "
          f"[{r.ci975_low:.6f}, {r.ci975_high:.6f}] seeds {seeds.min():.5f}-{seeds.max():.5f}")


# ── F: classification versus regression ───────────────────────────────────
F_XLIM = (5.0e-4, 1.25)
F_YLIM = (0.45, 1.025)


def panel_f(ax, summary, normalization):
    assert len(summary) == 112
    x = summary.mean_population_nmse.to_numpy(float)
    acc = summary.mean_accuracy.to_numpy(float)
    bal = summary.mean_balanced_accuracy.to_numpy(float)
    assert x.min() > F_XLIM[0] and x.max() < F_XLIM[1]
    assert min(acc.min(), bal.min()) > F_YLIM[0] and max(acc.max(), bal.max()) <= 1.0
    ceiling = int(((acc == 1.0) & (bal == 1.0)).sum())
    assert ceiling == 60 and int((acc == 1.0).sum()) == 60 and int((bal == 1.0).sum()) == 60
    # the AND/OR constant-majority references
    ref = normalization.set_index("family")
    for fam in ("and4", "or4"):
        assert float(ref.loc[fam, "constant_mean_nmse"]) == 1.0
        assert float(ref.loc[fam, "majority_class_accuracy"]) == 15.0 / 16.0
        assert float(ref.loc[fam, "constant_balanced_accuracy"]) == 0.5
    ax.scatter(x, acc, s=9, color=ACC, marker="o", alpha=0.55, lw=0, zorder=2)
    ax.scatter(x, bal, s=11, color=BAL, marker="x", alpha=0.6, lw=LW_ERR, zorder=3)
    ax.scatter([1.0, 1.0], [15.0 / 16.0, 0.5], marker="*", s=30, color=REF,
               lw=0, zorder=5)
    ax.set_xscale("log")
    ax.set_xlim(*F_XLIM)
    ax.set_ylim(*F_YLIM)
    ax.set_xticks([1e-3, 1e-2, 1e-1, 1.0], ["0.001", "0.01", "0.1", "1"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([0.5, 0.75, 1.0], ["0.50", "0.75", "1.00"])
    ax.set_xlabel("clean population NMSE")
    ax.set_ylabel("threshold performance")
    # the middle column of the NMSE axis holds no condition: the key and the
    # ceiling count go there, clear of every mark
    free = (x > 0.012) & (x < 0.125)
    assert not free.any()
    ax.text(0.014, 0.985, f"{ceiling}/112 at 1.00\non both metrics", ha="left",
            va="top", fontsize=PT_BASE, color=INK, linespacing=1.1, zorder=6)
    ax.text(0.014, 0.80, "accuracy", ha="left", va="top", fontsize=PT_BASE,
            color=ACC, zorder=6)
    ax.text(0.014, 0.735, "balanced\naccuracy", ha="left", va="top",
            fontsize=PT_BASE, color=BAL, linespacing=1.1, zorder=6)
    ax.scatter([0.0125], [0.585], marker="*", s=30, color=REF, lw=0, zorder=6)
    ax.text(0.0165, 0.615, "AND/OR\nmajority\nreference", ha="left", va="top",
            fontsize=PT_BASE, color=REF, linespacing=1.1, zorder=6)
    print(f"[F] 112 conditions; {ceiling} at accuracy = balanced accuracy = 1; "
          f"NMSE {x.min():.5f}-{x.max():.5f}; lowest accuracy {acc.min():.4f}, "
          f"lowest balanced {bal.min():.4f}")


# ── G: same-rate credit contrasts ─────────────────────────────────────────
G_XLIM = (0.0019, 0.048)
G_ADAM_YLIM = (-0.0062, 0.0215)
G_SGD_YLIM = (-1.3, 1.3)
G_LINTHRESH = 0.002


def panel_g(ax_adam, ax_sgd, same, paired):
    for ax, opt in ((ax_adam, "adam"), (ax_sgd, "sgd")):
        rows = same[same.optimizer.eq(opt)
                    & same.contrast.eq("broadcast_minus_exact_compatible")].sort_values("rate")
        assert list(rows.rate) == RATES and (rows.n_seeds == 20).all()
        assert not rows.primary.any() and (rows.margin_nmse == MARGIN).all()
        ax.set_xscale("log")
        ax.set_xlim(*G_XLIM)
        means, los, his = [], [], []
        for _, r in rows.iterrows():
            z = paired[paired.optimizer.eq(opt) & paired.rate.eq(r.rate)
                       & paired.contrast.eq("broadcast_minus_exact_compatible")].sort_values("seed")
            seeds = z.difference.to_numpy(float)
            assert len(seeds) == 20 and z.seed.is_unique
            np.testing.assert_allclose(seeds, z.comparator_nmse - z.reference_nmse,
                                       rtol=0, atol=1e-12)
            np.testing.assert_allclose(seeds.mean(), r.mean_difference, rtol=0, atol=1e-12)
            assert r.ci95_low < r.mean_difference < r.ci95_high
            ylim = G_ADAM_YLIM if opt == "adam" else G_SGD_YLIM
            assert seeds.min() > ylim[0] and seeds.max() < ylim[1], (opt, r.rate)
            fan(ax, r.rate, seeds, XOR, log_x=True)
            means.append(r.mean_difference); los.append(r.ci95_low); his.append(r.ci95_high)
            print(f"[G] {opt} rate {r.rate:g}: mean {r.mean_difference:.6f} "
                  f"[{r.ci95_low:.6f}, {r.ci95_high:.6f}] seeds {seeds.min():.4f}-{seeds.max():.4f}")
        ax.plot(RATES, means, color=XOR, lw=LW_HAIR, zorder=3.0)
        for rate, m, lo, hi in zip(RATES, means, los, his):
            mean_rule(ax, rate, m, lo, hi, XOR, log_x=True)
        reference(ax, 0.0, style="solid", color=MUTE, zorder=1.0)
        reference(ax, MARGIN)
        ax.set_xticks(RATES, ["0.003", "0.01", "0.03"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("common rate")
        if opt == "adam":
            ax.set_ylim(*G_ADAM_YLIM)
            ax.set_yticks([0.0, 0.01, 0.02], ["0.00", "0.01", "0.02"])
            ax.set_ylabel("broadcast − exact NMSE")
        else:
            ax.set_yscale("symlog", linthresh=G_LINTHRESH, linscale=1.0)
            ax.set_ylim(*G_SGD_YLIM)
            ax.set_yticks([-1, -0.1, -0.01, 0, 0.01, 0.1, 1],
                          ["−1", "−0.1", "−0.01", "0", "0.01", "0.1", "1"])
            ax.yaxis.set_minor_locator(NullLocator())
            # the note sits over the 0.01 and 0.03 columns, whose seeds stay
            # below 0.016; the 0.003 column on the left reaches 0.61
            ax.text(1.0, 1.0, "symlog y axis", transform=ax.transAxes, ha="right",
                    va="top", fontsize=PT_BASE, color=MUTE, zorder=5)


# ── H: broadcast gradient cosine along training ───────────────────────────
H_PIP_X = 0.6
H_XLIM = (0.44, 2700.0)
H_YLIM = (0.0, 1.09)
H_SERIES = [("xor_of_ands", "adam"), ("xor_of_ands", "sgd"),
            ("parity4", "adam"), ("parity4", "sgd")]


def panel_h(ax, trajectory, per_seed):
    ax.set_xscale("log")
    ax.set_xlim(*H_XLIM)
    ax.set_ylim(*H_YLIM)
    reference(ax, 1.0)
    for family, opt in H_SERIES:
        z = trajectory[trajectory.family.eq(family) & trajectory.optimizer.eq(opt)
                       & trajectory.rule.eq("broadcast")
                       & trajectory.tree.eq("balanced_ab_cd")].sort_values("step")
        assert list(z.step) == CHECKPOINTS and (z.n_seeds == 20).all()
        raw = per_seed[per_seed.family.eq(family) & per_seed.optimizer.eq(opt)
                       & per_seed.rule.eq("broadcast") & per_seed.tree.eq("balanced_ab_cd")]
        assert len(raw) == 160 and raw.seed.nunique() == 20
        rec = raw.groupby("step").gradient_cosine.mean().loc[CHECKPOINTS]
        np.testing.assert_allclose(rec.to_numpy(), z.mean_gradient_cosine.to_numpy(),
                                   rtol=0, atol=1e-12)
        mean = z.mean_gradient_cosine.to_numpy(float)
        lo = z.gradient_cosine_ci95_low.to_numpy(float)
        hi = z.gradient_cosine_ci95_high.to_numpy(float)
        assert (lo <= mean).all() and (mean <= hi).all()
        assert lo.min() > H_YLIM[0] and hi.max() < 1.0
        color = XOR if family == "xor_of_ands" else PARITY
        ls = "-" if opt == "adam" else (3.2, 1.8)
        face = color if opt == "adam" else "white"
        steps = np.array(CHECKPOINTS[1:], float)
        ax.fill_between(steps, lo[1:], hi[1:], color=color, alpha=BAND_ALPHA,
                        lw=0, zorder=1.5)
        ax.plot(steps, mean[1:], color=color, lw=LW_DATA,
                ls="-" if opt == "adam" else "--", dashes=(None, None) if opt == "adam" else ls,
                marker="o", ms=3.4, markerfacecolor=face, markeredgecolor=color,
                markeredgewidth=LW_ERR, zorder=3 if opt == "adam" else 3.2)
        # update 0 as a pip outside the broken axis, with its own interval;
        # the two optimizers share the initial state, so their pips carry the
        # same value and stand side by side under the one '0' tick
        pip_x = H_PIP_X * 10.0 ** (-0.05 if opt == "adam" else 0.05)
        ax.errorbar([pip_x], [mean[0]], yerr=[[mean[0] - lo[0]], [hi[0] - mean[0]]],
                    fmt="o", ms=3.4, color=color, markerfacecolor=face,
                    markeredgecolor=color, markeredgewidth=LW_ERR,
                    elinewidth=LW_ERR, capsize=0, zorder=3 if opt == "adam" else 3.2)
        print(f"[H] {family}/{opt}: cosine {mean[0]:.4f} at 0, {mean[-1]:.4f} at 2048 "
              f"(band {lo[-1]:.4f}-{hi[-1]:.4f})")
    # both optimizers share initial weights, so the step-0 pips coincide
    for family in ("xor_of_ands", "parity4"):
        z = trajectory[trajectory.family.eq(family) & trajectory.rule.eq("broadcast")
                       & trajectory.tree.eq("balanced_ab_cd") & trajectory.step.eq(0)]
        assert z.optimizer.nunique() == 2 and z.mean_gradient_cosine.nunique() == 1
    ax.set_xticks([H_PIP_X, 1, 16, 256, 2048], ["0", "1", "16", "256", "2048"])
    ax.set_xticks([64, 512, 1024], minor=True)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([0.0, 0.5, 1.0], ["0.0", "0.5", "1.0"])
    ax.set_xlabel("training update")
    ax.set_ylabel("population-gradient cosine")
    # keys: family by hue (top left, clear of every band), optimizer by line
    # style and fill (bottom right, below the lowest parity band)
    ax.text(1.05, 0.985, "XOR(AND)", ha="left", va="top", fontsize=PT_BASE,
            color=XOR, zorder=6)
    ax.text(1.05, 0.90, "parity", ha="left", va="top", fontsize=PT_BASE,
            color=PARITY, zorder=6)
    ax.annotate("cosine 1", xy=(1.0, 1.0), xycoords=("axes fraction", "data"),
                xytext=(-1.5, 1.0), textcoords="offset points", ha="right",
                va="bottom", fontsize=PT_BASE, color=REF)
    for y, label, ls, face in ((0.19, "Adam", "-", INK), (0.09, "SGD", "--", "white")):
        ax.plot([300.0, 700.0], [y, y], color=INK, lw=LW_DATA, ls=ls,
                dashes=(None, None) if ls == "-" else (3.2, 1.8), marker="o", ms=3.4,
                markerfacecolor=face, markeredgecolor=INK, markeredgewidth=LW_ERR,
                markevery=[1], zorder=6)
        ax.text(820.0, y, label, ha="left", va="center", fontsize=PT_BASE,
                color=INK, zorder=6)


# ── the canvas ────────────────────────────────────────────────────────────
CANVAS_H_PT = 493.0           # the supplement's 540 pt cap and the 1.05 aspect floor
HGUTTER_PT = 36.0
VGUTTER_PT = 40.0
MARGINS = Margins(left=46.0, right=42.0, top=21.0, bottom=32.0)
HEAT_RESERVE_PT = 14.0        # one left reserve for all four maps


def build(path: Path = OUT, *, png=False):
    summary = csv("condition_summary.csv")
    endpoints = csv("selected_endpoints.csv")
    primary = csv("primary_contrasts.csv")
    pairs = csv("paired_primary_contrasts.csv")
    same = csv("same_rate_contrast_summary.csv")
    paired_same = csv("paired_same_rate_contrasts.csv")
    trajectory = csv("trajectory_summary.csv")
    per_seed = csv("selected_trajectories.csv")
    normalization = csv("target_normalization.csv")
    import json
    rates = json.loads((SOURCE / "selected_rates.json").read_text())
    assert rates == {"adam": {"broadcast": 0.01, "exact": 0.003},
                     "sgd": {"broadcast": 0.01, "exact": 0.03}}

    cv = NativeCanvas(CANVAS_H_PT / 72.0, 4, hgutter_pt=HGUTTER_PT,
                      vgutter_pt=VGUTTER_PT, margins=MARGINS)
    a = cv.panel("A", 0, 0, 6)
    b = cv.panel("B", 0, 6, 6)
    c = cv.panel("C", 1, 0, 6)
    d = cv.panel("D", 1, 6, 6)
    e1 = cv.panel("E", 2, 0, 3, title="Grouping contrast", grid="y")
    e2 = cv.panel("E_credit", 2, 3, 3, letter="", title="Credit contrast", grid="y")
    f = cv.panel("F", 2, 6, 6, title="Classification versus regression", grid="y")
    g1 = cv.panel("G", 3, 0, 3, title="Adam", grid="y")
    g2 = cv.panel("G_sgd", 3, 3, 3, letter="", title="SGD", grid="y")
    h = cv.panel("H", 3, 6, 6, title="Broadcast gradients at own trained states", grid="y")
    for name in "ABCD":
        cv.declare_reserve(name, left=HEAT_RESERVE_PT)
    # the audit wants same-span panels of one row at one width: the second
    # subpanel of E and of G carries the same left reserve as the first
    for name in ("E_credit", "G_sgd"):
        cv.declare_reserve(name, left=HEAT_RESERVE_PT)

    mesh, norm = heatmaps(cv, [a, b, c, d], summary, endpoints, rates)
    panel_e(e1, e2, primary, pairs)
    panel_f(f, summary, normalization)
    panel_g(g1, g2, same, paired_same)
    panel_h(h, trajectory, per_seed)
    cv.lock_reserves()            # settle the boxes before drawing in points
    # the two map columns measure different left needs (the G y label sits in
    # column 0, the F y label in column 6): give both columns the larger lock
    # so all four maps share one axes width
    left = max(cv._locks[k][0] for k in "ABCD")
    for name in ("A", "B", "C", "D", "E_credit", "G_sgd"):
        cv.declare_reserve(name, left=left)
    cv.lock_reserves()
    colour_rail(cv, mesh, b, d)
    axis_break(e1, axis="y", lo=E_STRIP_V, hi=E_STRIP_V + E_GAP_V)
    axis_break(h, axis="x", lo=0.75, hi=0.89)
    # the four maps must share one axes width and B must register with D
    boxes = {k: cv.axes[k].get_position() for k in "ABCD"}
    widths = [boxes[k].width * cv.width_pt for k in "ABCD"]
    assert max(widths) - min(widths) < 0.5, widths
    assert abs(boxes["B"].x0 - boxes["D"].x0) * cv.width_pt < 0.5

    problems = cv.save(path, name="figure_boolean_learning_native", png=png)
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

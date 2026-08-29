#!/usr/bin/env python3
"""Main Figure 2 -- feedback coordinates, ownership and address -- as ONE
native full-width canvas.

Figure 2 used to be three separately rendered sub-blocks (``fig2_feedback``
panels B/C, ``fig_prospective_learning_benefits`` panels G/J/M and
``fig_fashion_feedback_ladder`` panels P/Q) that the compositor scaled into a
grid at three different factors, so one page carried nine different type
sizes and a dozen different stroke weights.  Here every panel is drawn onto
one 12-module :class:`figure_canvas.NativeCanvas` at exactly 518.4 pt wide and the
component is copied into ``figures/main/figure_02.pdf`` at scale 1.0.

Structure (three rows on one 12-module grid, one axes-box height per row):

* row 0 -- the task and the three feedback resolutions as four schematic
  cells sharing one panel letter: the classification pipeline ends in the
  readout error, and three cards define the ladder's resolution levels (one
  value for the layer; one coordinate per neuron; the complete
  per-compartment field), so the data rows under them read without a
  legend;
* row 1 -- the identity bottleneck as small multiples: matched three-rung
  feedback ladders for MNIST and Fashion-MNIST use separately labeled
  accuracy axes because their baselines differ, and the exact-gradient cosine
  sits beside them; three panels, four modules each;
* row 2 -- the ownership/address schematic (five modules), drawn natively
  from the shared credit-tree vocabulary, immediately left of the forest it
  explains (seven modules): every paired "X minus Y" contrast of the figure
  on ONE effect-size axis, grouped and labelled, carrying the paired seed
  differences and the published estimate with its 95% interval;
The former fourth row mixed two later questions into the standard-task
baseline: a depth robustness audit and the context-gated branch-conflict
experiment.  The robustness audit remains supplementary, while the conflict
experiment now has its own main figure after the credit-operator theory.

The geometry is column-locked: the left reserve of a grid column is the
maximum any panel of that column needs, so the three panels of row 1 are one
width, the label-heavy panels of one grid column share one reserve, and every panel of a row has
one axes-box height.  Numbers that only repeat a mark's own position (a
right-hand "mean [95% CI]" column beside a forest, a value printed next to a
bar) are not drawn: the geometry is the report and the exact values live in
Source Data.  Notes about dodging, fanning and coincidence live in the
caption, never inside a panel.

Every number, n, interval and test is the published one: summary rows come
from the frozen source-data tables and the three panels whose intervals are
bootstrapped in code call the inherited helpers with the inherited seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.ticker import PercentFormatter

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
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    SEED_ALPHA,
    SEED_MS,
    Margins,
    NativeCanvas,
    enforce_tokens,
    style_panel,
)
from credit_tree_schematics import (  # noqa: E402
    AMBER_TEXT,
    draw_credit_tree,
    draw_deranged_pair,
    mix,
)
from native_schematics import Frame, MIN_CORE_PT  # noqa: E402
from build_main_figure_01 import mini_tree  # noqa: E402
from journal_style import paired_lines  # noqa: E402

# Inherited panel logic, imported rather than re-implemented so that the
# marks and the bootstrap intervals stay identical to the published blocks.
from build_journal_figures import (  # noqa: E402
    errorbar_mean,
    jitter,
)
from analyze_prospective_learning_results import bootstrap_ci  # noqa: E402


ROOT = SCRIPT_DIR.parent
DATA = ROOT / "source_data"
COMPONENTS = ROOT / "figures" / "components"

SHUNT = COLORS["shunting"]
ADD = COLORS["additive"]
MUTE = COLORS["mute"]
INK = COLORS["ink"]

CORE_COLOR = {"dendritic_shunting": SHUNT, "dendritic_additive": ADD}
CORE_MARKER = {"dendritic_shunting": "o", "dendritic_additive": "s"}

# Dataset-specific accuracy axes expose the within-dataset feedback effect.
# Cross-dataset effect sizes are compared on the common percentage-point axis
# in panel F, so forcing B and C onto one absolute-accuracy range is unnecessary.
MNIST_ACC_LIM = (0.820, 0.982)
MNIST_ACC_TICKS = [0.84, 0.88, 0.92, 0.96]
FASHION_ACC_LIM = (0.824, 0.894)
FASHION_ACC_TICKS = [0.84, 0.86, 0.88]
ACC_LABEL = "held-out accuracy"

# One contrast axis, shared by the forest (x) and the depth panel (y).
# Seed dots span [-0.53, 5.89] and every mean/CI lies in [-0.21, 4.88],
# so the old 8.85 upper bound left ~30% of the axis empty and
# compressed the near-zero assignment contrasts that carry the point.
GAIN_LIM = (-0.85, 6.35)
GAIN_TICKS = [0, 2, 4, 6]
GAIN_LABEL = "accuracy difference (pp)"

STRICT_LADDER_TICKS = ["strict\nscalar", "neuron\nspecific", "exact\npath"]
FALLBACK_LADDER_TICKS = ["scalar\nfallback", "neuron\nspecific", "exact\npath"]

# One declared convention for the two categorical panels: the per-seed cloud
# is drawn as a symmetric deterministic fan a quarter-row BELOW its own mean
# marker, in table order (never sorted, which would make the cloud encode
# rank).  The convention is disclosed in the caption, not on the plot.
SEED_DY = 0.27
SEED_FAN = 0.05


def _fan(n):
    """Symmetric deterministic spread for a per-seed cloud."""
    return np.linspace(-SEED_FAN, SEED_FAN, n) if n > 1 else np.zeros(n)


# ── row 0: the task and the three feedback resolutions ───────────────────
# The row answers the two questions every data row below assumes: what the
# network is trained on, and what one rung of the feedback ladder means.
# The task cell runs image -> dendritic layer -> readout error, and three
# cards carry the ladder's exact x-category wording with the manuscript's
# own definitions.  Each card shows the same two-unit layer receiving that
# error at one resolution, so the contrast IS the cards' only difference.
# The four cells are separate grid panels (a full-width band would be a
# letterbox strip) but share the single letter A.  Signal hues follow the
# manuscript-wide semantics: amber = the scalar level, additive blue = the
# per-neuron coordinate, red-brown = the exact / backpropagated field.

BP = COLORS["bp"]
AMBER = COLORS["local"]
PIXEL = mix("ink", 82)

UNIT_HALF_PT = 8.0       # half-height of one mini unit inside a card
UNIT_GAP_PT = 21.0       # vertical spacing of the two stacked units

# 5x5 ink mask for the input glyph: a low-resolution digit.
_DIGIT = ("01110", "10001", "00110", "01000", "11111")


def _digit_glyph(f, cx, cy, side_pt):
    """A pixel-grid input image centred at (cx, cy), frame fractions."""
    w, h = f.fx(side_pt), f.fy(side_pt)
    x0, y0 = cx - w / 2.0, cy - h / 2.0
    f.group((x0, y0, w, h), tint="white", edge=COLORS["grid"],
            lw=LW_HAIR, radius_pt=1.2)
    cw, ch = w / 5.0, h / 5.0
    for r, row in enumerate(_DIGIT):
        for c, bit in enumerate(row):
            if bit == "1":
                f.ax.add_patch(Rectangle(
                    (x0 + (c + 0.07) * cw, y0 + (4 - r + 0.07) * ch),
                    0.86 * cw, 0.86 * ch, facecolor=PIXEL,
                    edgecolor="none", zorder=3))


def _unit_pair(f, x_soma, y_mid):
    """Two stacked mini units with somas at one x; returns soma y's."""
    ys = (y_mid + f.fy(UNIT_GAP_PT) / 2.0, y_mid - f.fy(UNIT_GAP_PT) / 2.0)
    for cy in ys:
        mini_tree(f, x_soma, cy, f.fy(UNIT_HALF_PT))
    return ys


def _card_scalar(f, core):
    """One error value for the whole layer: one source, both somas."""
    x0, y0, w, h = core
    sx, mid = x0 + 0.42 * w, y0 + 0.52 * h
    top, bot = _unit_pair(f, sx, mid)
    src = (sx + f.fx(24.0), mid)
    f.arrow(src, (sx + f.fx(4.5), top), color=AMBER, lw=LW_EDGE, head=4.0,
            rad=0.22)
    f.arrow(src, (sx + f.fx(4.5), bot), color=AMBER, lw=LW_EDGE, head=4.0,
            rad=-0.22)
    f.disc(src, 2.1, fill=AMBER)
    f.text((src[0], src[1] + f.fy(8.0)), "δ₀", size=PT_ANNOT,
           color=AMBER_TEXT)


def _card_neuron(f, core):
    """One coordinate per neuron: two distinct arrows, one per soma."""
    x0, y0, w, h = core
    sx, mid = x0 + 0.42 * w, y0 + 0.52 * h
    top, bot = _unit_pair(f, sx, mid)
    tail = sx + f.fx(24.0)
    for cy, tag in ((top, "δᵤ"), (bot, "δᵥ")):
        f.arrow((tail, cy), (sx + f.fx(4.5), cy), color=ADD, lw=LW_EDGE,
                head=4.0)
        f.text((tail + f.fx(3.5), cy), tag, size=PT_ANNOT, color=ADD,
               ha="left", va="center")


def _card_exact(f, core):
    """The complete field: every compartment carries its own error."""
    x0, y0, w, h = core
    sx, mid = x0 + 0.42 * w, y0 + 0.52 * h
    top, bot = _unit_pair(f, sx, mid)
    tail = sx + f.fx(24.0)
    dy = f.fy(UNIT_HALF_PT) / 1.5
    for cy in (top, bot):
        f.arrow((tail, cy), (sx + f.fx(4.5), cy), color=BP, lw=LW_EDGE,
                head=4.0)
        for my in (cy + 0.98 * dy, cy - 0.98 * dy):
            f.disc((sx - f.fx(9.5), my), 1.15, fill=BP, zorder=5)


def panel_task(ax):
    """Image -> dendritic layer -> readout error, one schematic cell."""
    f = Frame(ax)
    cell = (0.0, 0.0, 1.0, 1.0)
    f.group(cell, tint=None, edge=COLORS["grid"])
    core = f.cell_text(cell, title="task", subtitle="image → class label",
                       min_core_pt=MIN_CORE_PT)
    x0, y0, w, h = core
    cy = y0 + 0.50 * h
    start = x0 + (w - f.fx(96.0)) / 2.0

    def px(pt):
        return start + f.fx(pt)

    _digit_glyph(f, px(10.0), cy, 16.0)
    f.arrow((px(20.0), cy), (px(27.0), cy), color=MUTE, lw=LW_EDGE,
            head=3.6)
    soma_x = px(48.0)
    top, bot = _unit_pair(f, soma_x, cy)
    for uy, bend in ((top, 0.14), (bot, -0.14)):
        f.arrow((soma_x + f.fx(3.5), uy), (px(58.5), cy), color=MUTE,
                lw=LW_HAIR, head=3.2, rad=bend)
    box = (px(60.5), cy - f.fy(6.5), f.fx(15.0), f.fy(13.0))
    f.group(box, tint=mix("mute", 8), edge=COLORS["grid"], lw=LW_HAIR,
            radius_pt=2.0)
    f.text((px(68.0), cy), "ŷ", size=PT_ANNOT, color=INK)
    # the readout error: what the three cards deliver back at three grains
    f.arrow((px(77.5), cy), (px(84.5), cy), color=MUTE, lw=LW_EDGE,
            head=3.6)
    f.disc((px(89.0), cy), 2.2, fill=BP)
    f.text((px(89.0), cy + f.fy(8.0)), "δ₀", size=PT_ANNOT, color=BP)
    return ax


# One card per RESOLUTION LEVEL, with the gloss its one-line definition;
# that is the panel's whole prose.  The scalar level now exists in two
# implementations (the strict rung of the MNIST ladder in B and the
# historical matched-width fallback of the Fashion and gradient panels in C
# and D), so the first card carries the level's bare name "scalar" that both
# tick wordings extend; the other two cards still carry their x-categories'
# exact wording.
RESOLUTION_CARDS = (
    ("scalar", "scalar", AMBER_TEXT,
     "one value for the layer", _card_scalar),
    ("neuron", "neuron specific", ADD,
     "one δᵤ per neuron", _card_neuron),
    ("exact", "exact path", BP,
     "full field ∂ℒ/∂Vₙ", _card_exact),
)


def panel_resolution_card(ax, title, tone, gloss, draw):
    """One feedback-resolution card: the two-unit layer, one delivery."""
    f = Frame(ax)
    cell = (0.0, 0.0, 1.0, 1.0)
    f.group(cell, tint=None, edge=COLORS["grid"])
    core = f.cell_text(cell, title=title, title_color=tone, subtitle=gloss,
                       min_core_pt=MIN_CORE_PT)
    draw(f, core)
    return ax


# ── row 1: the feedback ladder, one accuracy axis ────────────────────────
def _paired_ladder(ax, data, metric, *, gradient=False):
    """Ported verbatim from ``build_journal_figures.paired_feedback_panel``.

    Two deliberate departures, both restyling: the mean glyph wears the
    architecture's own marker (so shunting is a circle and additive a square
    in every panel of the figure instead of both being diamonds here), the
    two means are joined so the panel reads as a ladder like its row-mates,
    and the seed dots drop their white keyline, which the token snap would
    otherwise widen to 0.55 pt on a 2.9 pt dot.  The values, the pairing, the
    jitter seeds and the bootstrap seeds are unchanged.
    """
    archs = ["dendritic_shunting", "dendritic_additive"]
    colors = {archs[0]: SHUNT, archs[1]: ADD}
    offsets = {archs[0]: -0.08, archs[1]: 0.08}
    if gradient:
        data = data[data["trained_broadcast_mode"].eq("per_soma_shared")].copy()
        data["condition"] = data["diagnostic_feedback"].map(
            {"scalar_fallback": "scalar", "neuron_wise": "ancestry"})
    else:
        data = data.copy()
        data["condition"] = data["feedback"].map(
            {"scalar_fallback": "scalar", "ancestry_shared": "ancestry"})
    for arch in archs:
        pivot = data[data["network_type"].eq(arch)].pivot_table(
            index="seed", columns="condition", values=metric,
            aggfunc="mean").dropna()
        xs = np.array([0, 1], float) + offsets[arch]
        paired_lines(ax, xs[0], xs[1], pivot["scalar"], pivot["ancestry"],
                     color=colors[arch], lw=LW_HAIR, alpha=0.24)
        columns = ["scalar", "ancestry"]
        for i, cond in enumerate(columns):
            vals = pivot[cond].to_numpy(float)
            ax.scatter(np.full(vals.size, xs[i])
                       + jitter(vals.size, 30 + i, 0.018), vals,
                       s=SEED_MS ** 2, color=colors[arch], alpha=SEED_ALPHA,
                       marker=CORE_MARKER[arch], edgecolors="none", zorder=3)
        ax.plot(xs, [float(np.mean(pivot[c].to_numpy(float)))
                     for c in columns],
                color=colors[arch], lw=LW_DATA, zorder=4)
        for i, cond in enumerate(columns):
            errorbar_mean(ax, xs[i], pivot[cond].to_numpy(float),
                          colors[arch], seed=40 + i, marker=CORE_MARKER[arch])
    return ax


def panel_mnist_ladder(ax):
    """Matched MNIST scalar, neuron-specific and exact-path ladder."""
    seeds = pd.read_csv(DATA / "mnist_feedback_ladder" / "seed_outcomes.csv")
    summary = pd.read_csv(
        DATA / "mnist_feedback_ladder" / "condition_summary.csv")
    order = ["scalar broadcast", "neuron specific", "exact path"]
    offsets = {"shunting": -0.06, "additive": 0.06}
    colors = {"shunting": SHUNT, "additive": ADD}
    markers = {"shunting": "o", "additive": "s"}
    for architecture in ("shunting", "additive"):
        wide = (seeds[seeds.architecture.eq(architecture)]
                .pivot(index="seed", columns="feedback",
                       values="test_accuracy")
                .loc[:, order])
        x = np.arange(3, dtype=float) + offsets[architecture]
        for values in wide.to_numpy(float):
            ax.plot(x, values, color=colors[architecture], alpha=0.24,
                    lw=LW_HAIR, zorder=2)
        for column, xi in zip(order, x):
            values = wide[column].to_numpy(float)
            ax.scatter(np.full(values.size, xi)
                       + jitter(values.size, 30 + order.index(column), 0.018),
                       values, s=SEED_MS ** 2, color=colors[architecture],
                       alpha=SEED_ALPHA, marker=markers[architecture],
                       edgecolors="none", zorder=3)
        part = (summary[summary.architecture.eq(architecture)]
                .set_index("feedback").loc[order])
        mean = part.mean_accuracy.to_numpy(float)
        ax.errorbar(
            x, mean,
            yerr=np.vstack([mean - part.ci95_low, part.ci95_high - mean]),
            color=colors[architecture], marker=markers[architecture],
            markerfacecolor="white", markeredgecolor=colors[architecture],
            markeredgewidth=LW_ERR, ms=MARKER_MS, lw=LW_DATA,
            elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(STRICT_LADDER_TICKS)
    ax.set_xlim(-0.52, 2.52)
    ax.set_ylim(*MNIST_ACC_LIM)
    ax.set_yticks(MNIST_ACC_TICKS)
    ax.set_ylabel(ACC_LABEL)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style_panel(ax)
    return ax


def panel_fashion_ladder(ax):
    """Fashion-MNIST replication of the ladder, on the same accuracy axis."""
    seeds = pd.read_csv(DATA / "fashion_feedback_ladder" / "seed_outcomes.csv")
    summary = pd.read_csv(
        DATA / "fashion_feedback_ladder" / "condition_summary.csv")
    order = ["scalar fallback", "neuron indexed", "exact path"]
    offsets = {"shunting": -0.06, "additive": 0.06}
    colors = {"shunting": SHUNT, "additive": ADD}
    markers = {"shunting": "o", "additive": "s"}
    for architecture in ("shunting", "additive"):
        wide = (seeds[seeds.architecture.eq(architecture)]
                .pivot(index="seed", columns="feedback",
                       values="test_accuracy")
                .loc[:, order])
        x = np.arange(3, dtype=float) + offsets[architecture]
        for values in wide.to_numpy(float):
            ax.plot(x, values, color=colors[architecture], alpha=0.24,
                    lw=LW_HAIR, zorder=2)
        # Same uncertainty convention as the row-mates: per-seed dots under
        # the published mean and interval (B and D scatter their seeds too).
        for column, xi in zip(order, x):
            vals = wide[column].to_numpy(float)
            ax.scatter(np.full(vals.size, xi)
                       + jitter(vals.size, 30 + order.index(column), 0.018),
                       vals, s=SEED_MS ** 2, color=colors[architecture],
                       alpha=SEED_ALPHA, marker=markers[architecture],
                       edgecolors="none", zorder=3)
        part = (summary[summary.architecture.eq(architecture)]
                .set_index("feedback").loc[order])
        y = part.mean_accuracy.to_numpy(float)
        ax.errorbar(x, y,
                    yerr=np.vstack([y - part.ci95_low, part.ci95_high - y]),
                    color=colors[architecture], marker=markers[architecture],
                    markerfacecolor="white",
                    markeredgecolor=colors[architecture],
                    markeredgewidth=LW_ERR, ms=MARKER_MS, lw=LW_DATA,
                    elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(FALLBACK_LADDER_TICKS)
    ax.set_xlim(-0.52, 2.52)
    ax.set_ylim(*FASHION_ACC_LIM)
    ax.set_yticks(FASHION_ACC_TICKS)
    ax.set_ylabel(ACC_LABEL)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style_panel(ax)
    return ax


def panel_gradient(ax):
    """Exact-gradient cosine at matched checkpoints, on the same ladder."""
    grad = pd.read_csv(DATA / "figure2" / "feedback_gradient_runs.csv")
    _paired_ladder(ax, grad, "branch_numel_weighted_cosine", gradient=True)
    ax.axhline(0, color=MUTE, ls="--", lw=LW_REF, zorder=0)
    # Exact path transport is the exact gradient, so its cosine is one by
    # definition. Show that ceiling as a reference marker rather than as an
    # empirical seed cloud or a third trained cohort.
    ax.scatter([2.0], [1.0], s=MARKER_MS ** 2, marker="D",
               facecolor="white", edgecolor=COLORS["oracle"],
               linewidth=LW_ERR, zorder=5)
    ax.text(2.0, 0.945, "by definition", color=COLORS["oracle"],
            fontsize=PT_SMALL, ha="center", va="top")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(FALLBACK_LADDER_TICKS)
    ax.set_xlim(-0.52, 2.52)
    ax.set_ylim(-0.16, 1.08)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("exact-gradient cosine")
    style_panel(ax)
    # Two series: direct labels at the separated right endpoints replace a
    # legend box.  This is the figure's colour and marker key.
    ax.text(1.20, 0.712, "shunting", color=SHUNT, fontsize=PT_LEGEND,
            ha="left", va="center")
    ax.text(1.20, 0.575, "additive", color=ADD, fontsize=PT_LEGEND,
            ha="left", va="center")
    return ax


# ── row 2, left: the schematic ───────────────────────────────────────────
# The deranged-pair glyph direct-labels its two rows ("correct", "deranged")
# to the right of the trees, so the frame it is drawn in has to be wide
# enough to hold those labels: the library's own frame stops at the trees and
# would let the labels run out of the cell and into the gutter.
DERANGED_XL = (-0.95, 5.75)
DERANGED_YL = (-1.60, 4.15)
DERANGED_ASPECT = ((DERANGED_XL[1] - DERANGED_XL[0])
                   / (DERANGED_YL[1] - DERANGED_YL[0]))


def _fit(frame, rect, aspect):
    """Largest sub-rectangle of ``rect`` at ``aspect``, centred in it."""
    x0, y0, w, h = rect
    w_pt, h_pt = w * frame.w_pt, h * frame.h_pt
    if w_pt / h_pt > aspect:
        fit_h, fit_w = h_pt, h_pt * aspect
    else:
        fit_w, fit_h = w_pt, w_pt / aspect
    return (x0 + (w - frame.fx(fit_w)) / 2.0,
            y0 + (h - frame.fy(fit_h)) / 2.0,
            frame.fx(fit_w), frame.fy(fit_h))


def panel_ownership_address(ax):
    """Coordinate-to-arbor assignment beside within-tree address.

    Drawn natively on this axes from the shared credit-tree vocabulary.  The
    two cells sit side by side because the panel is now one row high like
    every other panel of its row; the arrow between them carries the same
    ownership -> address reading the stacked version carried downwards.  The
    two footers it used to print ("same bandwidth, deranged map", "routes
    refine δᵤ → δᵤ,ₖ") are prose and have moved to the caption.
    """
    f = Frame(ax)
    # The two cells are NOT equal: the assignment cell carries two mini-trees,
    # four coordinate labels and two right-hand row names, and at an even
    # split its "deranged" label overran the cell border by 3 pt.  The address
    # cell holds one tree and reads comfortably narrower.  The module grid
    # itself is left alone: row 2 is locked to a 5/7 split by the canvas
    # contract (equal spans there force equal axes-box widths, which a
    # gutter-free schematic beside a data panel cannot satisfy).
    gap = f.fx(9.0)
    left_w = 0.585 * (1.0 - gap)
    right_w = (1.0 - gap) - left_w
    left = (0.0, 0.0, left_w, 1.0)
    right = (left_w + gap, 0.0, right_w, 1.0)
    for rect in (left, right):
        f.group(rect, tint=None, edge=COLORS["grid"])

    core = f.cell_text(
        left, title="arbor assignment", title_color=SHUNT,
        subtitle="which arbor gets δᵤ?",
        min_core_pt=MIN_CORE_PT + 10.0)
    sub = ax.inset_axes(_fit(f, Frame.inset(core, bottom=0.04),
                             DERANGED_ASPECT),
                        transform=ax.transData, zorder=3)
    sub.set_facecolor("none")
    draw_deranged_pair(sub, xlim=DERANGED_XL, ylim=DERANGED_YL)
    enforce_tokens(sub)

    core = f.cell_text(
        right, title="within-tree address", title_color=COLORS["oracle"],
        subtitle="where in the tree?",
        min_core_pt=MIN_CORE_PT + 10.0)
    f.tree(Frame.inset(core, left=0.06, right=0.06, bottom=0.04),
           mode="address", K=4)
    f.arrow((left_w + gap * 0.16, 0.5), (left_w + gap * 0.84, 0.5),
            color=MUTE, lw=LW_HAIR, head=4.6)
    return ax


# ── row 2, right: the forest ─────────────────────────────────────────────
def _fashion_rows():
    seeds = pd.read_csv(DATA / "fashion_feedback_ladder" / "seed_outcomes.csv")
    contrasts = pd.read_csv(
        DATA / "fashion_feedback_ladder" / "paired_contrasts.csv")
    pairs = {
        "neuron indexed - scalar fallback":
            ("neuron indexed", "scalar fallback"),
        "exact path - neuron indexed": ("exact path", "neuron indexed"),
    }
    out = {}
    for name, (high, low) in pairs.items():
        rows = []
        for architecture in ("shunting", "additive"):
            row = contrasts[contrasts.contrast.eq(name)
                            & contrasts.architecture.eq(architecture)].iloc[0]
            wide = (seeds[seeds.architecture.eq(architecture)]
                    .pivot(index="seed", columns="feedback",
                           values="test_accuracy"))
            rows.append({
                "label": f"Fashion {architecture}",
                "color": SHUNT if architecture == "shunting" else ADD,
                "marker": "o" if architecture == "shunting" else "s",
                "mean": 100 * float(row.mean_difference),
                "lo": 100 * float(row.ci95_low),
                "hi": 100 * float(row.ci95_high),
                "seeds": 100 * (wide[high] - wide[low]).to_numpy(float),
            })
        out[name] = rows
    return out


def _ownership_rows():
    contrasts = pd.read_csv(
        DATA / "prospective_input_validity"
        / "routing_valid_paired_contrasts.csv")
    outcomes = pd.read_csv(
        DATA / "prospective_input_validity"
        / "followup_publication_seed_outcomes.csv")
    outcomes = outcomes[outcomes.family.eq("routing")]
    task_label = {"mnist": "MNIST", "noise_resilience": "noise"}
    order = [("mnist", "dendritic_shunting", 2),
             ("mnist", "dendritic_additive", 2),
             ("mnist", "dendritic_shunting", 4),
             ("mnist", "dendritic_additive", 4),
             ("noise_resilience", "dendritic_additive", 2),
             ("noise_resilience", "dendritic_additive", 4)]
    rows = []
    for task, core, depth in order:
        row = contrasts[contrasts.task.eq(task) & contrasts.core.eq(core)
                        & contrasts.depth.eq(depth)].iloc[0]
        part = outcomes[outcomes.task.eq(task) & outcomes.core.eq(core)
                        & outcomes.depth.eq(depth)]
        wide = part.pivot_table(index="seed", columns="routing",
                                values="test_accuracy")
        paired = 100 * (wide["correct"] - wide["shuffled"]).dropna().to_numpy(float)
        rows.append({
            # Uppercase D-notation: the manuscript defines D1-D4 and never
            # uses a lowercase variant, so the row labels match the text.
            "label": f"{task_label[task]} D{depth} {core.split('_')[1]}",
            "color": CORE_COLOR[core],
            "marker": CORE_MARKER[core],
            "mean": 100 * float(row.mean_difference),
            "lo": 100 * float(row.ci95_low),
            "hi": 100 * float(row.ci95_high),
            "seeds": paired,
        })
    return rows


# The three group headers name the contrast in one word each; what each
# contrast subtracts from what ("neuron-indexed minus scalar", "exact path
# minus neuron-indexed", "correct minus deranged neuron-to-tree map") is a
# definition and belongs in the caption, not in a sentence-long in-panel
# header that has to be set at half the width of the plot.
HEADER_LEFT_PT = 52.0


def panel_forest(ax):
    """Every paired contrast of the figure on one effect-size axis.

    The point and its whisker ARE the estimate and the interval, so no
    right-hand "mean [95% CI]" column repeats them; the exact values are in
    Source Data and in the running text.  The reclaimed width goes back into
    the effect-size axis.
    """
    from matplotlib import transforms as mtransforms

    fashion = _fashion_rows()
    groups = [
        ("neuron-specific", fashion["neuron indexed - scalar fallback"]),
        ("transport", fashion["exact path - neuron indexed"]),
        ("assignment", _ownership_rows()),
    ]
    header_trans = mtransforms.offset_copy(
        ax.get_yaxis_transform(), fig=ax.get_figure(), x=-HEADER_LEFT_PT,
        y=0.0, units="points")

    # The dashed zero reference and an x=0 gridline would print as one
    # mottled stroke, so the grid is drawn here without its zero member.
    for xt in GAIN_TICKS:
        if xt:
            ax.axvline(xt, color=COLORS["grid"], lw=LW_HAIR, zorder=0)
    ax.axvline(0.0, color=MUTE, ls="--", lw=LW_REF, zorder=0.1)
    y = 0.0
    ticks, labels = [], []
    for index, (header, rows) in enumerate(groups):
        # A blank half-row before every block but the first, so the mute
        # group name reads as the head of the rows under it rather than as
        # one more row in the same column.
        y += 0.0 if index == 0 else 0.55
        ax.text(0.0, y - 0.04, header, transform=header_trans,
                fontsize=PT_ANNOT, color=MUTE, ha="left", va="center",
                zorder=6, clip_on=False)
        y += 1.0
        for row in rows:
            seeds = np.asarray(row["seeds"], dtype=float)
            ax.scatter(seeds,
                       np.full(seeds.size, y - SEED_DY)
                       + _fan(seeds.size),
                       s=SEED_MS ** 2, color=row["color"], alpha=SEED_ALPHA,
                       edgecolors="none", zorder=3)
            ax.errorbar(row["mean"], y,
                        xerr=[[row["mean"] - row["lo"]],
                              [row["hi"] - row["mean"]]],
                        color=row["color"], marker=row["marker"],
                        markerfacecolor="white", markeredgecolor=row["color"],
                        markeredgewidth=LW_ERR, ms=MARKER_MS, lw=LW_ERR,
                        elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=5)
            ticks.append(y)
            labels.append(row["label"])
            y += 1.0
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y - 0.5, -0.62)
    ax.set_xlim(*GAIN_LIM)
    ax.set_xticks(GAIN_TICKS)
    ax.set_xlabel(GAIN_LABEL)
    style_panel(ax)
    ax.tick_params(axis="y", length=0, labelsize=PT_SMALL)
    ax.spines["left"].set_visible(False)
    return ax


# ── row 3: a real-image path-necessity boundary ─────────────────────────
# The old two-stream endpoint remains in Supplementary Fig. S19G.  The main
# panel now shows the stronger task-family result: the point at which a shared
# somatic coordinate becomes harmful shifts with the number of simultaneously
# driven branches exactly as predicted before the confirmatory seeds ran.
PATH_BRANCH_STYLE = {
    2: (COLORS["additive"], "o"),
    4: (COLORS["local"], "s"),
    8: (COLORS["oracle"], "^"),
}


def panel_path_necessity(ax):
    """Correct-path benefit across the prespecified credit-conflict family."""
    contrasts = pd.read_csv(
        DATA / "path_necessity_fashion" / "paired_contrasts.csv"
    )
    selected = contrasts[
        contrasts.contrast.eq("correct - shared")
        & contrasts.endpoint.eq("test_accuracy")
    ]
    for branches, (color, marker) in PATH_BRANCH_STYLE.items():
        part = selected[selected.branches.eq(branches)].sort_values(
            "conflict_probability"
        )
        x = part.conflict_probability.to_numpy(float)
        y = 100.0 * part.mean_difference.to_numpy(float)
        low = 100.0 * part.ci95_low.to_numpy(float)
        high = 100.0 * part.ci95_high.to_numpy(float)
        ax.plot(
            x,
            y,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=LW_ERR,
            ms=MARKER_MS,
            lw=LW_DATA,
            label=f"$B={branches}$",
            zorder=4,
        )
        ax.fill_between(x, low, high, color=color, alpha=0.11, linewidth=0)
        boundary = branches / (2.0 * (branches - 1))
        ax.scatter(
            [boundary],
            [1.1],
            marker="v",
            s=(MARKER_MS + 0.4) ** 2,
            facecolor=color,
            edgecolor="white",
            linewidth=LW_HAIR,
            clip_on=False,
            zorder=6,
        )
    ax.axhline(0, color=MUTE, ls="--", lw=LW_REF, zorder=0)
    ax.set_xlim(-0.025, 1.025)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_ylim(-2.5, 64.0)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel(r"credit conflict $\chi$")
    ax.set_ylabel("correct path $-$ shared (pp)")
    style_panel(ax)
    ax.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=PT_SMALL,
        handlelength=1.2,
        handletextpad=0.35,
        columnspacing=0.72,
        borderaxespad=0.12,
    )
    ax.text(
        0.02,
        0.08,
        r"$\blacktriangledown$ predicted $\chi_c$",
        transform=ax.transAxes,
        fontsize=PT_SMALL,
        color=MUTE,
        ha="left",
        va="bottom",
    )
    return ax


def panel_identity_depth(ax):
    """Neuron-indexed minus scalar accuracy across dendritic stage count."""
    contrast = pd.read_csv(
        DATA / "prospective_input_validity"
        / "central_valid_paired_contrasts.csv")
    identity = contrast[contrast.family.eq("feedback")
                        & contrast.contrast.eq("neuron-indexed - scalar")
                        & contrast.task.eq("mnist")]
    for core in ("dendritic_shunting", "dendritic_additive"):
        part = identity[identity.core.eq(core)].sort_values("depth")
        x = part.depth.to_numpy(float)
        y = 100 * part.mean_difference.to_numpy(float)
        lo = 100 * part.ci95_low.to_numpy(float)
        hi = 100 * part.ci95_high.to_numpy(float)
        ax.errorbar(x, y, yerr=np.vstack([y - lo, hi - y]),
                    color=CORE_COLOR[core], marker=CORE_MARKER[core],
                    markerfacecolor="white", markeredgecolor=CORE_COLOR[core],
                    markeredgewidth=LW_ERR, ms=MARKER_MS, lw=LW_DATA,
                    elinewidth=LW_ERR, capsize=ERR_CAPSIZE)
    ax.axhline(0, color=MUTE, ls="--", lw=LW_REF, zorder=0)
    # D-ticks make this panel the figure's definition of the D-notation:
    # the axis label says what a stage count is, the ticks name the levels,
    # and the D2/D4 rows of the forest index into them.
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["D1", "D2", "D3", "D4"])
    ax.set_xlim(0.62, 4.38)
    ax.set_ylim(*GAIN_LIM)
    ax.set_yticks(GAIN_TICKS)
    ax.set_xlabel("dendritic stage count")
    ax.set_ylabel(GAIN_LABEL)
    style_panel(ax)
    ax.text(1.16, 7.9, "shunting", color=SHUNT, fontsize=PT_LEGEND,
            ha="left", va="center")
    ax.text(1.16, 3.4, "additive", color=ADD, fontsize=PT_LEGEND,
            ha="left", va="center")
    return ax


# ── the canvas ───────────────────────────────────────────────────────────
# Three rows on one 12-module grid: the schematic band, the three ladders,
# and the schematic-plus-forest row.  No panel carves
# its own reserve: every left reserve is the column lock the canvas
# measures, topped up by ``_equalise_row`` so panels of one row that start
# in different grid columns still share one axes-box width.
CANVAS_H_PT = 398.0                       # compact standard-task figure
ROW_H_PT = (68.0, 87.0, 108.0)
VGUTTER_PT = 62.0 / 3.0 + 25.0


def _equalise_row(canvas, names_cols):
    """Declare left reserves so same-row panels share one axes width.

    The column lock equalises panels that start in the SAME grid column;
    panels of one row that start in different columns can still differ by a
    point or two of tick-label width.  Measure after the first lock, then
    top the narrow reserves up to the widest so the row is exact.
    """
    boxes = {name: canvas.axes[name].get_position() for name in names_cols}
    widths = {name: box.width * canvas.width_pt
              for name, box in boxes.items()}
    target = min(widths.values())
    for name, col in names_cols.items():
        extra = widths[name] - target
        if extra <= 0.05:
            continue
        slot_x = (canvas.margins.left
                  + col * (canvas._module_w + canvas.hgutter))
        left_now = boxes[name].x0 * canvas.width_pt - slot_x
        canvas.declare_reserve(name, left=left_now + extra)


def build(height_in=CANVAS_H_PT / 72.0, path=None):
    canvas = NativeCanvas(height_in, 3, row_weights=list(ROW_H_PT),
                          hgutter_pt=32.0, vgutter_pt=VGUTTER_PT,
                          margins=Margins(left=46.0, right=12.0, top=18.0,
                                          bottom=25.0))

    ax_task = canvas.panel("task", 0, 0, 3, schematic=True, letter="",
                           inset_pt=(2.0, 2.0, 2.0, 2.0), lock=False)
    card_axes = []
    for index, (name, *_rest) in enumerate(RESOLUTION_CARDS):
        card_axes.append(canvas.panel(f"card_{name}", 0, 3 * (index + 1), 3,
                                      schematic=True, letter="",
                                      inset_pt=(2.0, 2.0, 2.0, 2.0)))
    ax_a = canvas.panel("mnist", 1, 0, 4, title="MNIST", letter="B",
                        inset_pt=(1.0, 1.0, 1.0, 1.0))
    ax_b = canvas.panel("fashion", 1, 4, 4, title="Fashion-MNIST",
                        letter="C", inset_pt=(1.0, 1.0, 1.0, 1.0))
    ax_c = canvas.panel("gradient", 1, 8, 4,
                        title="Gradient alignment", letter="",
                        inset_pt=(1.0, 1.0, 1.0, 1.0))
    ax_d = canvas.panel("schematic", 2, 0, 5, schematic=True, letter="")
    ax_e = canvas.panel("forest", 2, 5, 7, title="Paired accuracy contrasts",
                        letter="", inset_pt=(0.0, 2.0, 0.0, 0.0))
    canvas.add_letter("A", ax_task)
    # The rotated gradient-axis label reaches the top of its panel; place D
    # in the inter-panel gutter so the two glyphs cannot collide.
    canvas.add_letter("D", ax_c, dx_pt=46.0)
    canvas.add_letter("E", ax_d)
    canvas.add_letter("F", ax_e)

    panel_mnist_ladder(ax_a)
    panel_fashion_ladder(ax_b)
    panel_gradient(ax_c)
    panel_ownership_address(ax_d)
    panel_forest(ax_e)

    # The ladder row starts in three different grid columns; measure the
    # first lock, equalise the row, then draw the schematic cells on final
    # geometry (their Frames capture the axes box at draw time).
    canvas.lock_reserves()
    _equalise_row(canvas, {"mnist": 0, "fashion": 4, "gradient": 8})
    canvas.lock_reserves()
    panel_task(ax_task)
    for ax_card, (_name, title, tone, gloss, draw) in zip(card_axes,
                                                          RESOLUTION_CARDS):
        panel_resolution_card(ax_card, title, tone, gloss, draw)

    target = Path(path) if path else COMPONENTS / "main_figure_02_native.pdf"
    problems = canvas.save(target, name="main_figure_02_native")
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

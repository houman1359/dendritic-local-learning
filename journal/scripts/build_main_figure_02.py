#!/usr/bin/env python3
"""Main Figure 2 -- feedback coordinates, ownership and address -- as ONE
native full-width canvas.

Figure 2 used to be three separately rendered sub-blocks (``fig2_feedback``
panels B/C, ``fig_prospective_learning_benefits`` panels G/J/M and
``fig_fashion_feedback_ladder`` panels P/Q) that the compositor scaled into a
grid at three different factors, so one page carried nine different type
sizes and a dozen different stroke weights.  Here every panel is drawn onto
one 12-module :class:`figure_canvas.NativeCanvas` at exactly 518.4 pt and the
component is copied into ``figures/main/figure_02.pdf`` at scale 1.0.

Structure (three rows on one 12-module grid, one axes-box height per row):

* row 0 -- the identity bottleneck as small multiples: MNIST and
  Fashion-MNIST held-out accuracy share ONE accuracy axis (label, range and
  tick labels appear once, at the left of the row) and the exact-gradient
  cosine sits beside them on the same feedback ladder; three panels, four
  modules each;
* row 1 -- the ownership/address schematic (five modules), drawn natively
  from the shared credit-tree vocabulary, immediately left of the forest it
  explains (seven modules): every paired "X minus Y" contrast of the figure
  on ONE effect-size axis, grouped and labelled, carrying the paired seed
  differences and the published estimate with its 95% interval;
* row 2 -- the within-tree credit reversal and the identity gain across
  dendritic stage count, six modules each, the latter on exactly the same
  contrast scale as the forest.

The geometry is column-locked: the left reserve of a grid column is the
maximum any panel of that column needs, so the three panels of row 0 are one
width, the two panels of row 2 are one width, and every panel of a row has
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
from credit_tree_schematics import draw_credit_tree, draw_deranged_pair  # noqa: E402
from native_schematics import Frame, MIN_CORE_PT  # noqa: E402
from journal_style import paired_lines  # noqa: E402

# Inherited panel logic, imported rather than re-implemented so that the
# marks and the bootstrap intervals stay identical to the published blocks.
from build_journal_figures import (  # noqa: E402
    _temper_glyph_arrows,
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

# One accuracy axis for the held-out-accuracy panels of the top row.
ACC_LIM = (0.828, 0.982)
ACC_TICKS = [0.84, 0.88, 0.92, 0.96]
ACC_LABEL = "held-out accuracy"

# One contrast axis, shared by the forest (x) and the depth panel (y).
GAIN_LIM = (-0.85, 8.85)
GAIN_TICKS = [0, 2, 4, 6, 8]
GAIN_LABEL = "accuracy difference (pp)"

LADDER_TICKS = ["scalar\nfallback", "neuron\nindexed", "exact\npath"]

# One declared convention for the two categorical panels: the per-seed cloud
# is drawn as a symmetric deterministic fan a quarter-row BELOW its own mean
# marker, in table order (never sorted, which would make the cloud encode
# rank).  The convention is disclosed in the caption, not on the plot.
SEED_DY = 0.27
SEED_FAN = 0.05


def _fan(n):
    """Symmetric deterministic spread for a per-seed cloud."""
    return np.linspace(-SEED_FAN, SEED_FAN, n) if n > 1 else np.zeros(n)


# ── row 0: the feedback ladder, one accuracy axis ────────────────────────
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
    """MNIST held-out accuracy under scalar vs neuron-indexed feedback."""
    acc = pd.read_csv(DATA / "figure2" / "feedback_accuracy_runs.csv")
    _paired_ladder(ax, acc, "test_accuracy")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(LADDER_TICKS[:2])
    ax.set_xlim(-0.52, 1.52)
    ax.set_ylim(*ACC_LIM)
    ax.set_yticks(ACC_TICKS)
    ax.set_ylabel(ACC_LABEL)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style_panel(ax)
    # The manipulation itself, under its own category: a global scalar
    # broadcast reaching one soma, and a neuron-indexed coordinate.  The two
    # glyphs sit on a tinted key strip that spans the panel, so they read as
    # an annotation of the x categories and not as marks at 84-88% accuracy.
    strip = Rectangle((0.0, 0.035), 1.0, 0.345, transform=ax.transAxes,
                      facecolor=COLORS["panel_bg"], edgecolor="none",
                      zorder=0.3)
    ax.add_patch(strip)
    for centre, mode in ((0.0, "scalar"), (1.0, "coordinate")):
        x0 = (centre + 0.52) / 2.04 - 0.17
        glyph = ax.inset_axes([x0, 0.055, 0.34, 0.30])
        glyph.set_facecolor("none")
        draw_credit_tree(glyph, mode=mode, scale=0.60, labels=False)
        _temper_glyph_arrows(glyph, 0.45)
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
        # the published mean and interval (A and C scatter their seeds too).
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
    ax.set_xticklabels(LADDER_TICKS)
    ax.set_xlim(-0.52, 2.52)
    ax.set_ylim(*ACC_LIM)
    ax.set_yticks(ACC_TICKS)
    style_panel(ax)
    # Shared axis: label, tick labels and numbers live once, on panel a.
    ax.set_yticklabels([])
    return ax


def panel_gradient(ax):
    """Exact-gradient cosine at matched checkpoints, on the same ladder."""
    grad = pd.read_csv(DATA / "figure2" / "feedback_gradient_runs.csv")
    _paired_ladder(ax, grad, "branch_numel_weighted_cosine", gradient=True)
    ax.axhline(0, color=MUTE, ls="--", lw=LW_REF, zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(LADDER_TICKS[:2])
    ax.set_xlim(-0.52, 1.88)
    ax.set_ylim(-0.16, 0.86)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.set_ylabel("dendritic-gradient cosine")
    style_panel(ax)
    # Two series: direct labels at the separated right endpoints replace a
    # legend box.  This is the figure's colour and marker key.
    ax.text(1.20, 0.712, "shunting", color=SHUNT, fontsize=PT_LEGEND,
            ha="left", va="center")
    ax.text(1.20, 0.575, "additive", color=ADD, fontsize=PT_LEGEND,
            ha="left", va="center")
    return ax


# ── row 1, left: the schematic ───────────────────────────────────────────
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
    """Neuron ownership (which tree) beside within-tree address (where).

    Drawn natively on this axes from the shared credit-tree vocabulary.  The
    two cells sit side by side because the panel is now one row high like
    every other panel of its row; the arrow between them carries the same
    ownership -> address reading the stacked version carried downwards.  The
    two footers it used to print ("same bandwidth, deranged map", "routes
    refine δᵤ → δᵤ,ₖ") are prose and have moved to the caption.
    """
    f = Frame(ax)
    gap = f.fx(11.0)
    cell_w = (1.0 - gap) / 2.0
    left = (0.0, 0.0, cell_w, 1.0)
    right = (cell_w + gap, 0.0, cell_w, 1.0)
    for rect in (left, right):
        f.group(rect, tint=None, edge=COLORS["grid"])

    core = f.cell_text(
        left, title="neuron ownership", title_color=SHUNT,
        subtitle="which tree gets δᵤ?",
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
    f.arrow((cell_w + gap * 0.16, 0.5), (cell_w + gap * 0.84, 0.5),
            color=MUTE, lw=LW_HAIR, head=4.6)
    return ax


# ── row 1, right: the forest ─────────────────────────────────────────────
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
            "label": f"{task_label[task]} d{depth} {core.split('_')[1]}",
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
        ("identity", fashion["neuron indexed - scalar fallback"]),
        ("transport", fashion["exact path - neuron indexed"]),
        ("ownership", _ownership_rows()),
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


# ── row 2 ────────────────────────────────────────────────────────────────
# One colour and one marker per routing condition, following the
# manuscript-wide routing taxonomy shared with Figs. 3, 5 and 8: red-brown =
# exact transport (the ceiling), green = anatomy-correct routing, rose =
# broken permutation control, blue = random rank-K, amber = neuron-shared,
# gray = gated point control.
CONTROL_DARK = COLORS["point_mlp"]        # #686868
CONTROL_LIGHT = "#989898"                 # the second control lightness

SUBTREE_COLORS = {
    "neuron_shared_k1": CONTROL_LIGHT,
    "correct_subtree_k2": COLORS["shunting"],
    "within_neuron_deranged_k2": CONTROL_DARK,
    "random_dense_rank2": CONTROL_DARK,
    "exact_transport": COLORS["bp"],
    "gated_point_emulation": CONTROL_LIGHT,
}
SUBTREE_MARKERS = {
    "neuron_shared_k1": "X",
    "correct_subtree_k2": "o",
    "within_neuron_deranged_k2": "s",
    "random_dense_rank2": "v",
    "exact_transport": "D",
    "gated_point_emulation": "P",
}
# Shortest unambiguous form for every row label: the category column is a
# tick column like any other, so it is locked to the grid column's reserve
# and must not grow into the panel beside it.
SUBTREE_LABEL = {
    "neuron_shared_k1": "neuron\nshared",
    "correct_subtree_k2": "correct\nancestry",
    "within_neuron_deranged_k2": "route\nderanged",
    "random_dense_rank2": "random\nrank-2",
    "exact_transport": "exact\ntransport",
    "gated_point_emulation": "gated\npoint",
}
SUBTREE_ORDER = ["correct_subtree_k2", "exact_transport",
                 "gated_point_emulation", "neuron_shared_k1",
                 "random_dense_rank2", "within_neuron_deranged_k2"]


def panel_credit_reversal(ax):
    """Held-out accuracy in the two-stream within-tree credit reversal.

    The four-line note this panel used to print (that its top three rows
    coincide on 10/10 seeds, that its accuracy range is wider than A and B,
    and that its seed dots follow the convention of E) is disclosure, not
    graphic content: it has moved to the caption.
    """
    subtree = pd.read_csv(
        DATA / "trained_subtree_address" / "seed_outcomes.csv")
    for index, condition in enumerate(SUBTREE_ORDER):
        values = subtree[
            subtree.condition.eq(condition)].test_accuracy.to_numpy(float)
        color = SUBTREE_COLORS[condition]
        ax.scatter(values, index - SEED_DY + _fan(values.size),
                   s=SEED_MS ** 2, color=color, alpha=SEED_ALPHA,
                   edgecolors="none", zorder=3)
        mean, low, high = bootstrap_ci(values, seed=42_000 + index)
        ax.errorbar(mean, index, xerr=[[mean - low], [high - mean]],
                    color=color, marker=SUBTREE_MARKERS[condition],
                    markerfacecolor="white", markeredgecolor=color,
                    markeredgewidth=LW_ERR, ms=MARKER_MS, lw=LW_ERR,
                    elinewidth=LW_ERR, capsize=ERR_CAPSIZE, zorder=5)
    ax.set_yticks(range(len(SUBTREE_ORDER)))
    ax.set_yticklabels([SUBTREE_LABEL[c] for c in SUBTREE_ORDER])
    for tick_label in ax.get_yticklabels():
        tick_label.set_linespacing(1.02)
    ax.set_ylim(len(SUBTREE_ORDER) - 0.45, -0.62)
    ax.set_xlim(0.10, 0.92)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel(ACC_LABEL)
    # Same unit as panels A and B: held-out accuracy is a percentage
    # everywhere on the page.  Only the range differs, and the panel says so.
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style_panel(ax, grid="x")
    ax.tick_params(axis="y", length=0, labelsize=PT_SMALL)
    ax.spines["left"].set_visible(False)
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
    ax.set_xticks([1, 2, 3, 4])
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
# Three equal rows on one 12-module grid.  No panel carves its own reserve:
# every left reserve is the column lock the canvas measures, so the three
# panels of row 0 share one width, the two panels of row 2 share one width,
# and the forest -- the only panel whose row-label column is wider than a
# gutter -- has a grid column of its own to take that reserve out of.
CANVAS_H_PT = 463.0                       # 518.4 / 463.0 = 1.12 aspect


def build(height_in=CANVAS_H_PT / 72.0, path=None):
    canvas = NativeCanvas(height_in, 3, row_weights=[1.0, 1.0, 1.0],
                          hgutter_pt=32.0, vgutter_pt=38.0,
                          margins=Margins(left=46.0, right=12.0, top=16.0,
                                          bottom=26.0))

    ax_a = canvas.panel("mnist", 0, 0, 4, title="MNIST feedback ladder")
    ax_b = canvas.panel("fashion", 0, 4, 4, title="Fashion-MNIST replication")
    ax_c = canvas.panel("gradient", 0, 8, 4, title="Exact-gradient alignment")
    ax_d = canvas.panel("schematic", 1, 0, 5, schematic=True, letter="")
    ax_e = canvas.panel("forest", 1, 5, 7, title="Paired accuracy contrasts",
                        letter="")
    ax_f = canvas.panel("reversal", 2, 0, 6, title="Within-tree reversal",
                        letter="")
    ax_g = canvas.panel("depth", 2, 6, 6, title="Identity gain across depth",
                        letter="")
    canvas.add_letter("D", ax_d)
    canvas.add_letter("E", ax_e)
    canvas.add_letter("F", ax_f)
    canvas.add_letter("G", ax_g)

    panel_mnist_ladder(ax_a)
    panel_fashion_ladder(ax_b)
    panel_gradient(ax_c)
    panel_ownership_address(ax_d)
    panel_forest(ax_e)
    panel_credit_reversal(ax_f)
    panel_identity_depth(ax_g)

    target = Path(path) if path else COMPONENTS / "main_figure_02_native.pdf"
    problems = canvas.save(target, name="main_figure_02_native")
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

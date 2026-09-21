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
* row 1 -- four matched diagnostic panels: the three-rung MNIST feedback
  ladder, exact-gradient alignment, mean transported-error magnitude by
  depth and the within-depth path-specific error-energy fraction; three
  modules each;
* row 2 -- the ownership/address schematic (five modules), drawn natively
  from the shared credit-tree vocabulary, immediately left of the forest it
  explains (seven modules): every paired "X minus Y" contrast of the figure
  on ONE effect-size axis, grouped and labelled, carrying the paired seed
  differences and the published estimate with its 95% interval;
Fashion-MNIST and the former depth-robustness row remain supplementary.  The
context-gated branch-conflict experiment has its own main figure after the
credit-operator theory.

The geometry is column-locked: the left reserve of a grid column is the
maximum any panel of that column needs, so the four panels of row 1 are one
width, the label-heavy panels of one grid column share one reserve, and every panel of a row has
one axes-box height.  Numbers that only repeat a mark's own position (a
right-hand "mean [95% CI]" column beside a forest, a value printed next to a
bar) are not drawn: the geometry is the report and the exact values live in
Source Data.  Notes about dodging, fanning and coincidence live in the
caption, never inside a panel.

Every number, sample size, interval and test is read from the retained
source-data tables; intervals generated here use the documented seeds and
the shared bootstrap helpers.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter, PercentFormatter

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


ROOT = SCRIPT_DIR.parent
DATA = ROOT / "source_data"
COMPONENTS = ROOT / "figures" / "components"

SHUNT = COLORS["shunting"]
ADD = COLORS["additive"]
MUTE = COLORS["mute"]
INK = COLORS["ink"]

CORE_COLOR = {"dendritic_shunting": SHUNT, "dendritic_additive": ADD}
CORE_MARKER = {"dendritic_shunting": "o", "dendritic_additive": "s"}

# The primary feedback ladder is the matched 15-seed MNIST cohort.  The
# Fashion-MNIST replication now appears with the other cross-dataset controls
# in Supplementary Fig. S4.
MNIST_ACC_LIM = (0.820, 0.982)
MNIST_ACC_TICKS = [0.84, 0.88, 0.92, 0.96]
ACC_LABEL = "held-out accuracy"

# Effect-size axes of the forest.  Neither strip's limits are typed in:
# each is derived from the rows it draws (every paired seed difference and
# every interval end) padded by STRIP_PAD of that span, so a refreshed
# table cannot silently push a seed dot past the axis edge -- the fixed
# limits this replaces clipped five upper-tail dots of the ladder strip and
# five negative dots of the readout-minus-DFA block.  The two strips keep
# their own pp scales, but the zero of both sits at ONE axis fraction
# (solved in ``_align_zero`` from whichever strip needs the larger left
# share), so the stacked dashed references align vertically instead of
# reading as a drafting slip.  Ticks are typed, then checked against the
# derived limits in ``panel_forest``.
STRIP_PAD = 0.04
ASSIGN_TICKS = [0, 0.4, 0.8]
FOREST_LABEL_RESERVE_PT = 99.0   # widest row name + tick pad + header hang
GAIN_TICKS = [0, 4, 8, 12]
GAIN_LABEL = "accuracy difference (pp)"

# Keep the scientifically distinct feedback conditions visible in the
# artwork.  The axis gives the three scientific resolution levels; the caption
# states that the scalar arm is strict in B.  The fixed-checkpoint alignment
# diagnostic uses the separately defined matched-width scalar field.
STRICT_LADDER_TICKS = ["strict\nscalar", "neuron-\nspecific", "exact\npath"]
ALIGNMENT_LADDER_TICKS = [
    "matched-width\nfallback", "neuron-\nspecific", "exact\npath"
]

# One declared convention for the two categorical panels: the per-seed cloud
# is drawn as a symmetric deterministic fan a quarter-row BELOW its own mean
# marker, in table order (never sorted, which would make the cloud encode
# rank).  The convention is disclosed in the caption, not on the plot.
SEED_DY = 0.27
SEED_FAN = 0.05


def _fan(n):
    """Symmetric deterministic spread for a per-seed cloud."""
    return np.linspace(-SEED_FAN, SEED_FAN, n) if n > 1 else np.zeros(n)


def _definitional_marker(ax, x, y, *, color, marker):
    """One open marker with a dashed edge: a value fixed by definition.

    Measured means are open markers with a solid edge at LW_ERR; a
    definitional constant (cosine one for exact transport in C, the
    normalised soma ratio in D) gets the dashed edge at the reference
    weight and a mute "def." tag beside it, so the two are never confused.
    """
    ax.scatter([x], [y], s=MARKER_MS ** 2, marker=marker,
               facecolor="white", edgecolor=color, linewidth=LW_REF,
               linestyle=(0.0, (1.4, 1.1)), zorder=5)


def _style_feedback_ticks(ax):
    """Keep the three feedback conditions legible in quarter-width panels."""
    ax.tick_params(axis="x", labelsize=PT_SMALL, pad=1.5)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")
        label.set_rotation_mode("anchor")


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


def _unit_pair(f, x_soma, y_mid, *, avail_pt=None):
    """Two stacked mini units with somas at one x; returns soma y's.

    ``avail_pt`` is the core height the pair has to live in.  A card whose
    subtitle wraps to two lines keeps less core than one whose subtitle fits
    on a single line, and at the fixed size the lower unit's tips were drawn
    straight through the card floor.  Given the room it has, the pair scales
    to fit and every card's arbor stays whole.
    """
    half, gap = UNIT_HALF_PT, UNIT_GAP_PT
    if avail_pt is not None:
        # tips reach 1.55/1.5 of the half-height beyond the outer somas, so
        # the arbor pair spans gap + 2 * 1.033 * half -- and the exact card
        # then centres a 1.15 pt disc ON each outer tip.  Reserve that disc
        # plus a hair of clearance at both ends of the core, or the lowest
        # mark of a tight card is bisected by the card's own bottom border.
        tip_pad = 2.4
        need = gap + 2.07 * half
        room = avail_pt - 2.0 * tip_pad
        if need > room > 0:
            k = room / need
            half, gap = half * k, gap * k
    ys = (y_mid + f.fy(gap) / 2.0, y_mid - f.fy(gap) / 2.0)
    for cy in ys:
        mini_tree(f, x_soma, cy, f.fy(half))
    return ys, half


def _card_scalar(f, core):
    """One error value for the whole layer: one source, both somas."""
    x0, y0, w, h = core
    sx, mid = x0 + 0.42 * w, y0 + 0.52 * h
    (top, bot), unit_half = _unit_pair(f, sx, mid, avail_pt=h * f.h_pt)
    src = (sx + f.fx(24.0), mid)
    f.arrow(src, (sx + f.fx(4.5), top), color=AMBER, lw=LW_EDGE, head=4.0,
            rad=0.22)
    f.arrow(src, (sx + f.fx(4.5), bot), color=AMBER, lw=LW_EDGE, head=4.0,
            rad=-0.22)
    f.disc(src, 2.1, fill=AMBER)
    f.text((src[0], src[1] + f.fy(8.0)), "s", size=PT_ANNOT,
           color=AMBER_TEXT)


def _card_neuron(f, core):
    """One coordinate per neuron: two distinct arrows, one per soma."""
    x0, y0, w, h = core
    sx, mid = x0 + 0.42 * w, y0 + 0.52 * h
    (top, bot), unit_half = _unit_pair(f, sx, mid, avail_pt=h * f.h_pt)
    tail = sx + f.fx(24.0)
    for cy, tag in ((top, "δᵤ"), (bot, "δᵥ")):
        f.arrow((tail, cy), (sx + f.fx(4.5), cy), color=ADD, lw=LW_EDGE,
                head=4.0)
        f.text((tail + f.fx(3.5), cy), tag, size=PT_ANNOT, color=ADD,
               ha="left", va="center")


def _card_exact(f, core):
    """The complete field: every compartment carries its own error.

    Drawn as one arrow into the soma this card was the neuron card in a
    different colour -- same geometry, same single delivery -- and nothing
    on it said "per compartment".  The transported field now runs OUT along
    the arbor's own edges and terminates in a mark on every compartment, so
    the three cards differ in the one way that matters: one value for the
    layer, one per neuron, one per compartment.
    """
    x0, y0, w, h = core
    sx, mid = x0 + 0.42 * w, y0 + 0.52 * h
    (top, bot), unit_half = _unit_pair(f, sx, mid, avail_pt=h * f.h_pt)
    tail = sx + f.fx(24.0)
    dy = f.fy(unit_half) / 1.5
    tip_x, mid_x = sx - f.fx(18.0), sx - f.fx(9.5)
    for cy in (top, bot):
        f.arrow((tail, cy), (sx + f.fx(4.5), cy), color=BP, lw=LW_EDGE,
                head=4.0)
        mid_ys = (cy + 0.98 * dy, cy - 0.98 * dy)
        tip_ys = (cy + 1.55 * dy, cy + 0.50 * dy,
                  cy - 0.50 * dy, cy - 1.55 * dy)
        # Marks only, not a red overlay of the edges: painted along the
        # branches the field hid the green the rest of the paper uses for
        # dendrite, and the card stopped reading as an arbor at all.
        for my in mid_ys:
            f.disc((mid_x, my), 1.15, fill=BP, zorder=5)
        for ly in tip_ys:
            f.disc((tip_x, ly), 1.15, fill=BP, zorder=5)


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
    (top, bot), _ = _unit_pair(f, soma_x, cy)
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
    f.text((px(89.0), cy + f.fy(8.0)), "δₒᵤₜ", size=PT_ANNOT,
           color=BP)
    return ax


# One card per RESOLUTION LEVEL, with the gloss its one-line definition;
# that is the panel's whole prose.  The scalar level now exists in two
# implementations (the strict rung of the MNIST ladder in B and the
# matched-width scalar-fallback field in the fixed-checkpoint diagnostic C),
# so the first card carries the level's bare name "scalar" that both
# tick wordings extend; the other two cards still carry their x-categories'
# exact wording.
RESOLUTION_CARDS = (
    # "one scalar s for the layer" wrapped with a one-word widow, the only
    # ragged gloss in the card row; the title already names the level
    # "scalar", so the gloss drops the repeated word and sets on one line.
    ("scalar", "scalar", AMBER_TEXT,
     "one s per layer", _card_scalar),
    ("neuron", "neuron-specific", ADD,
     "one δᵤ / neuron", _card_neuron),
    ("exact", "exact path", BP,
     "one ∂ℒ/∂Vₙ per compartment", _card_exact),
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


def panel_transport_profile(ax):
    """D: mean transported-error magnitude at each dendritic depth.

    The same batch-RMS definition is used for both architectures and each
    branch is normalized to the RMS error at its own soma. This panel shows
    the mean branchwise profile; panel E separately measures the fraction of
    exact field energy that remains after removing the mean at each depth.
    """
    cv = pd.read_csv(DATA / "figure2" / "path_gain_dispersion_ladder_runs.csv")
    colors = {"additive": ADD, "shunting": SHUNT}
    marker = {"additive": "s", "shunting": "o"}
    xs = np.array([0.0, 1.0, 2.0])          # soma, mid, distal
    ax.axhline(1.0, color=COLORS["grid"], lw=LW_REF, zorder=1)
    offsets = {"additive": -0.025, "shunting": 0.025}
    for arch in ("additive", "shunting"):
        sub = cv[cv.architecture.eq(arch)]
        prof = np.column_stack([np.ones(len(sub)),
                                sub.stage1_mean.to_numpy(float),
                                sub.stage0_mean.to_numpy(float)])
        x_arch = xs + offsets[arch]
        for row in prof:
            ax.plot(x_arch, row, color=colors[arch], lw=LW_HAIR,
                    alpha=SEED_ALPHA * 0.6, zorder=2)
        mean = prof.mean(axis=0)
        ax.plot(x_arch, mean, color=colors[arch], lw=LW_DATA, zorder=4)
        # The soma column is skipped: it is identically 1 (each branch is
        # normalised to its own soma), and drawing it through the mean
        # helper printed a zero-width interval as if it had been estimated.
        for index, x in enumerate(x_arch):
            if index == 0:
                continue
            errorbar_mean(ax, x, prof[:, index], colors[arch],
                          seed=580 + 10 * index + (arch == "shunting"),
                          marker=marker[arch])
    # One definitional marker at the soma for both architectures, in the
    # same open-dashed-mute "def." token as panel C's reference diamond: the
    # two mean lines leave it in their own colours.
    _definitional_marker(ax, xs[0], 1.0, color=MUTE, marker="D")
    ax.text(xs[0], 1.30, "def.", fontsize=PT_SMALL, color=MUTE,
            ha="center", va="bottom", zorder=5)
    ax.set_xlim(-0.25, 2.25)
    ax.set_xticks(xs, ["soma", "mid", "distal"])
    # Each branch value is its batch-RMS exact compartment error divided by
    # the batch-RMS error at the branch's own soma.  The plotted stage value
    # then averages this ratio over branches and neurons within a seed.
    ax.text(
        0.06, 0.97,
        "soma-normalized\nbatch RMS",
        transform=ax.transAxes, fontsize=PT_SMALL, color=MUTE,
        ha="left", va="top",
    )
    # The raw-additive distal field is amplified above the soma whereas the
    # shunting field is attenuated. A logarithmic ordinate keeps both regimes
    # legible without compressing the smaller shunting values against zero.
    # The 1-2-5 major ladder and unlabelled minor ticks at the remaining
    # integers of each decade declare the scale; three ticks at 0.2/1/5 did
    # not, and a reader took the ordinate for linear.
    ax.set_yscale("log")
    ax.set_ylim(0.16, 6.4)
    ax.set_yticks([0.2, 0.5, 1.0, 2.0, 5.0])
    ax.set_yticklabels(["0.2", "0.5", "1", "2", "5"])
    ax.set_yticks([0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 3.0, 4.0, 6.0], minor=True)
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("mean batch-RMS ratio")
    return ax


def panel_path_specific_energy(ax):
    """E: exact-error energy outside one depth-shared coordinate.

    For every example and neuron, the branch mean is removed separately at
    each depth.  The squared residual is divided by the exact field energy,
    so the ordinate is the fraction of transported-error energy that a
    depth-shared signal cannot represent.  Each faint line is one independently
    trained exact-path checkpoint; thick open markers show the seed mean and
    its bootstrap interval.
    """
    frame = pd.read_csv(
        DATA / "figure2" / "path_gain_dispersion_ladder_runs.csv")
    columns = ["stage1_path_specific_energy_fraction",
               "stage0_path_specific_energy_fraction"]  # mid, distal
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(
            "path-specific transported-error fields are missing: "
            + ", ".join(missing))

    xs = np.array([0.0, 1.0])
    colors = {"additive": ADD, "shunting": SHUNT}
    markers = {"additive": "s", "shunting": "o"}
    for arch in ("additive", "shunting"):
        sub = frame[frame.architecture.eq(arch)].sort_values("seed")
        values = 100.0 * sub[columns].to_numpy(float)
        for row in values:
            ax.plot(xs, row, color=colors[arch], lw=LW_HAIR,
                    alpha=SEED_ALPHA * 0.55, zorder=2)
        ax.plot(xs, values.mean(axis=0), color=colors[arch], lw=LW_DATA,
                zorder=4)
        for index, x in enumerate(xs):
            errorbar_mean(ax, x, values[:, index], colors[arch],
                          seed=620 + 10 * index + (arch == "shunting"),
                          marker=markers[arch])

    ax.set_xlim(-0.22, 1.22)
    ax.set_xticks(xs, ["mid", "distal"])
    ax.text(
        0.06, 0.985,
        "within-depth\nresidual",
        transform=ax.transAxes, fontsize=PT_SMALL, color=MUTE,
        ha="left", va="top",
    )
    ax.set_ylim(0.0, 59.0)
    ax.set_yticks([0, 20, 40])
    ax.set_ylabel("path-specific error energy (%)")
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
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(*MNIST_ACC_LIM)
    ax.set_yticks(MNIST_ACC_TICKS)
    ax.set_ylabel(ACC_LABEL)
    # The figure's one series legend, on its first panel: the lower-right
    # field under the rising ladder is the only region every seed cloud
    # leaves empty, here and in the panels that inherit the key.
    ax.text(2.20, MNIST_ACC_LIM[0] + 0.028, "shunting", color=SHUNT,
            fontsize=PT_LEGEND, ha="right", va="center")
    ax.text(2.20, MNIST_ACC_LIM[0] + 0.010, "raw additive", color=ADD,
            fontsize=PT_LEGEND, ha="right", va="center")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    style_panel(ax)
    _style_feedback_ticks(ax)
    return ax


def panel_gradient(ax):
    """Branch-parameter gradient cosine at matched checkpoints."""
    grad = pd.read_csv(DATA / "figure2" / "feedback_gradient_runs.csv")
    _paired_ladder(ax, grad, "branch_numel_weighted_cosine", gradient=True)
    ax.axhline(0, color=MUTE, ls="--", lw=LW_REF, zorder=0)
    # Exact path transport is the exact gradient, so its cosine is one by
    # definition: no cohort was trained or measured there.  The reference
    # marker therefore wears the figure's definitional-constant token (open,
    # dashed edge at the reference weight, mute "def." tag) rather than the
    # solid-edged open marker every measured mean uses, so it cannot be read
    # as a third seed cloud or a mean with a vanishing interval.
    _definitional_marker(ax, 2.0, 1.0, color=COLORS["oracle"], marker="D")
    ax.text(2.0, 0.905, "def.", fontsize=PT_SMALL, color=MUTE,
            ha="center", va="top", zorder=5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(ALIGNMENT_LADDER_TICKS)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-0.16, 1.08)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("branch-gradient cosine")
    style_panel(ax)
    _style_feedback_ticks(ax)
    # Two series: direct labels at the separated right endpoints replace a
    # legend box.  This is the figure's colour and marker key.
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
    """Coordinate-to-arbor ownership beside within-arbor address.

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
        left, title="arbor ownership", title_color=SHUNT,
        subtitle="which arbor gets δᵤ?",
        min_core_pt=MIN_CORE_PT + 10.0)
    sub = ax.inset_axes(_fit(f, Frame.inset(core, bottom=0.04),
                             DERANGED_ASPECT),
                        transform=ax.transData, zorder=3)
    sub.set_facecolor("none")
    draw_deranged_pair(sub, xlim=DERANGED_XL, ylim=DERANGED_YL)
    enforce_tokens(sub)

    core = f.cell_text(
        right, title="within-arbor address", title_color=COLORS["oracle"],
        subtitle="where in the tree?",
        min_core_pt=MIN_CORE_PT + 10.0)
    tree = f.tree(Frame.inset(core, left=0.06, right=0.06, bottom=0.04),
                  mode="address", K=4)
    # The library paints the K=4 address capsules in pale signal hues
    # (shunting green, additive blue, amber, oracle purple) that the rest of
    # THIS figure spends on architectures and ceilings, so here the four
    # patches are re-toned to one neutral hue at four lightness steps: still
    # four distinct addresses, ordered left to right, with no false mapping
    # onto the series key.  Capsules are the only fat pale Line2Ds on the
    # inset, matched by their exact source tints.
    retone = {
        mix("shunting", 16): mix("mute", 14),
        mix("additive", 15): mix("mute", 24),
        mix("local", 24): mix("mute", 34),
        mix("oracle", 16): mix("mute", 45),
    }
    for line in tree.lines:
        rgb = to_rgb(line.get_color())
        for src, dst in retone.items():
            if np.allclose(rgb, src, atol=1e-6):
                line.set_color(dst)
                break
    f.arrow((left_w + gap * 0.16, 0.5), (left_w + gap * 0.84, 0.5),
            color=MUTE, lw=LW_HAIR, head=4.6)
    return ax


# ── row 2, right: the forest ─────────────────────────────────────────────
def _mnist_rows():
    """Panel B's own ladder as paired per-seed effect sizes.

    G first summarized the Fashion-MNIST ladder here while its assignment block
    ran on MNIST, so the one-scale comparison mixed tasks for no reason the
    figure could state. Every block now uses MNIST; the Fashion-MNIST
    replication is retained in Supplementary Fig. S4E.
    """
    seeds = pd.read_csv(DATA / "mnist_feedback_ladder" / "seed_outcomes.csv")
    contrasts = pd.read_csv(
        DATA / "mnist_feedback_ladder" / "paired_contrasts.csv")
    pairs = {
        "neuron specific - scalar broadcast":
            ("neuron specific", "scalar broadcast"),
        "exact path - neuron specific": ("exact path", "neuron specific"),
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
                "label": f"MNIST {'raw additive' if architecture == 'additive' else architecture}",
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
    order = [("mnist", "dendritic_shunting", 2),
             ("mnist", "dendritic_additive", 2),
             ("mnist", "dendritic_shunting", 4),
             ("mnist", "dendritic_additive", 4)]
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
            # Spell out stage count at its first use rather than introducing
            # D2/D4 notation three figures before the physical-depth section.
            "label": f"MNIST {depth}-stage "
                     f"{'raw additive' if core == 'dendritic_additive' else 'shunting'}",
            "color": CORE_COLOR[core],
            "marker": CORE_MARKER[core],
            "mean": 100 * float(row.mean_difference),
            "lo": 100 * float(row.ci95_low),
            "hi": 100 * float(row.ci95_high),
            "seeds": paired,
        })
    return rows


def _factorial_rows():
    """The between-by-within factorial's headline contrasts (Supp. Fig. S30).

    Two families join panel G so the between-neuron test is visible in the
    main figure: the DFA-source ladder step (does neuron identity survive a
    fixed random between-neuron map?) on the ladder pp scale, and the
    readout-minus-DFA gap at matched within-rungs on the sub-point scale.
    The exact-readout rows above them are the frozen ladder release; these
    rows share its seeds, architectures, colors and markers.
    """
    seeds = pd.read_csv(
        DATA / "mnist_between_within_factorial" / "seed_outcomes.csv")
    contrasts = pd.read_csv(
        DATA / "mnist_between_within_factorial" / "paired_contrasts.csv")

    def rows_for(name, high, low, label_prefix=""):
        rows = []
        for architecture in ("shunting", "additive"):
            row = contrasts[contrasts.contrast.eq(name)
                            & contrasts.architecture.eq(architecture)].iloc[0]
            part = seeds[seeds.architecture.eq(architecture)]
            wide = part.pivot_table(index="seed",
                                    columns=["between", "within"],
                                    values="test_accuracy")
            paired = 100 * (wide[high] - wide[low]).dropna().to_numpy(float)
            rows.append({
                "label": (label_prefix
                          + ("raw additive" if architecture == "additive"
                             else "shunting")),
                "color": SHUNT if architecture == "shunting" else ADD,
                "marker": "o" if architecture == "shunting" else "s",
                "mean": 100 * float(row.mean_difference),
                "lo": 100 * float(row.ci95_low),
                "hi": 100 * float(row.ci95_high),
                "seeds": paired,
            })
        return rows

    return {
        "dfa_neuron_minus_scalar": rows_for(
            "dfa within: neuron - scalar",
            ("dfa", "neuron specific"), ("dfa", "scalar broadcast"),
            label_prefix="MNIST "),
        "bp_minus_dfa": (
            rows_for("between at neuron: readout - dfa",
                     ("readout backprop", "neuron specific"),
                     ("dfa", "neuron specific"),
                     label_prefix="neuron rung, ")
            + rows_for("between at exact path: readout - dfa",
                       ("readout backprop", "exact path"),
                       ("dfa", "exact path"),
                       label_prefix="path rung, ")),
    }


# The group headers name the contrast in one word each; what each
# contrast subtracts from what ("neuron-specific minus scalar", "exact path
# minus neuron-specific", "correct minus deranged neuron-to-tree map") is a
# definition and belongs in the caption, not in a sentence-long in-panel
# header that has to be set at half the width of the plot.
HEADER_LEFT_PT = 80.0


def _row_extent(row):
    """Every x this row puts ink at: seed dots, mean and interval ends."""
    return np.concatenate([np.asarray(row["seeds"], dtype=float),
                           [row["mean"], row["lo"], row["hi"]]])


def _strip_extent(groups):
    """Padded data extent of one strip over every row it draws."""
    drawn = np.concatenate([_row_extent(row) for _h, rows in groups
                            for row in rows])
    lo, hi = float(drawn.min()), float(drawn.max())
    pad = STRIP_PAD * (hi - lo)
    return lo - pad, hi + pad


def _align_zero(extents):
    """Widen each strip on the left until x = 0 sits at one shared fraction.

    Every strip straddles zero; the one whose padded extent needs the
    largest left share fixes the fraction, and the others gain empty room
    on the left (never on the right, where the data end) to match it.
    """
    for lo, hi in extents:
        if not lo < 0.0 < hi:
            raise AssertionError(f"forest strip does not straddle zero: "
                                 f"({lo:.3f}, {hi:.3f})")
    frac = max((0.0 - lo) / (hi - lo) for lo, hi in extents)
    return [(-frac / (1.0 - frac) * hi, hi) for _lo, hi in extents]


def panel_forest(ax):
    """Every paired contrast of the figure on one effect-size axis.

    Two stacked strips, one x scale each: the ladder blocks live on a
    ~15 pp axis, and the assignment block gets its own ~1 pp axis --
    on the shared scale its half-point effects sat inside one marker width
    of zero and their ordering was unreadable.  The point and its whisker
    ARE the estimate and the interval; the exact values are in Source Data
    and the running text.  Both strips' limits are derived from the rows
    they draw (``_strip_extent``, ``_align_zero``), never typed in.
    """
    from matplotlib import transforms as mtransforms

    mnist = _mnist_rows()
    # Each header carries the panel it summarizes: the first two blocks are
    # B's ladder steps re-expressed as paired within-seed effect sizes, and
    # the third is the correct-minus-deranged test that F draws.  Without
    # the pointers the rows read as a second dataset rather than as B's own
    # contrasts on an inferential scale.
    # Both strips label their own x axis: they share one unit but differ in
    # scale by ~13x, and with the label printed only under the lower strip
    # the upper strip's bare tick numbers read as the same scale.  The upper
    # strip therefore gives up a little height to make room for its own
    # tick-plus-label band above the lower strip's hanging group header.
    factorial = _factorial_rows()
    strips = [
        ((0.58, 0.42),
         [("neuron-specific (B)",
           mnist["neuron specific - scalar broadcast"]),
          ("path resolution (B)", mnist["exact path - neuron specific"]),
          ("neuron-specific under DFA source (S30)",
           factorial["dfa_neuron_minus_scalar"])],
         GAIN_TICKS, GAIN_LABEL, False),
        ((0.0, 0.36),
         [("ownership: correct − deranged (F)", _ownership_rows()),
          # Only the neuron-rung gap is drawn; the path-rung gap (at most
          # 0.2 pp) stays in Supplementary Fig. S30 to keep the strip
          # legible.
          ("readout BP − DFA source (S30)",
           factorial["bp_minus_dfa"][:2])],
         ASSIGN_TICKS, GAIN_LABEL, True),
    ]
    xlims = _align_zero([_strip_extent(groups) for _g, groups, *_r in strips])
    ax.set_axis_off()
    for ((y0, height), groups, xticks, xlabel, zero_below), xlim in zip(
            strips, xlims):
        sub = ax.inset_axes([0.0, y0, 1.0, height])
        sub.set_facecolor("none")
        header_trans = mtransforms.offset_copy(
            sub.get_yaxis_transform(), fig=sub.get_figure(),
            x=-HEADER_LEFT_PT, y=0.0, units="points")
        # The dashed zero reference and an x=0 gridline would print as one
        # mottled stroke, so the grid is drawn without its zero member.
        for xt in xticks:
            if xt:
                sub.axvline(xt, color=COLORS["grid"], lw=LW_HAIR, zorder=0)
        y = 0.0
        ticks, labels = [], []
        header_ys = []
        for index, (header, rows) in enumerate(groups):
            # A blank half-row before every block but the first, so the mute
            # group name reads as the head of the rows under it rather than
            # as one more row in the same column.
            y += 0.0 if index == 0 else 0.55
            header_ys.append(y - 0.04)
            sub.text(0.0, y - 0.04, header, transform=header_trans,
                     fontsize=PT_ANNOT, color=MUTE, ha="left", va="center",
                     zorder=6, clip_on=False)
            y += 1.0
            for row in rows:
                seeds = np.asarray(row["seeds"], dtype=float)
                sub.scatter(seeds,
                            np.full(seeds.size, y - SEED_DY)
                            + _fan(seeds.size),
                            s=SEED_MS ** 2, color=row["color"],
                            alpha=SEED_ALPHA, edgecolors="none", zorder=3)
                sub.errorbar(row["mean"], y,
                             xerr=[[row["mean"] - row["lo"]],
                                   [row["hi"] - row["mean"]]],
                             color=row["color"], marker=row["marker"],
                             markerfacecolor="white",
                             markeredgecolor=row["color"],
                             markeredgewidth=LW_ERR, ms=MARKER_MS, lw=LW_ERR,
                             elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                             zorder=5)
                ticks.append(y)
                labels.append(row["label"])
                y += 1.0
        sub.set_yticks(ticks)
        sub.set_yticklabels(labels)
        sub.set_ylim(y - 0.5, -0.62)
        # The dashed zero reference.  The lower strip's hanging group header
        # crosses x = 0, so there the line starts just below the header band
        # instead of striking straight through the words.
        if zero_below:
            # Draw the dashed zero reference in segments that dodge every
            # hanging group header rather than striking through the words.
            spans, previous = [], -0.62
            for header_y in header_ys:
                spans.append((previous, header_y - 0.34))
                previous = header_y + 0.46
            spans.append((previous, y - 0.5))
            for lo, hi in spans:
                if hi > lo:
                    sub.plot([0.0, 0.0], [lo, hi], color=MUTE, ls="--",
                             lw=LW_REF, zorder=0.1)
        else:
            sub.axvline(0.0, color=MUTE, ls="--", lw=LW_REF, zorder=0.1)
        sub.set_xlim(*xlim)
        # Nothing this strip draws may lie outside its own axis: a clipped
        # seed dot silently misreports the paired distribution.
        axis_lo, axis_hi = sub.get_xlim()
        for _header, rows in groups:
            for row in rows:
                drawn = _row_extent(row)
                if drawn.min() < axis_lo or drawn.max() > axis_hi:
                    raise AssertionError(
                        f"forest row {row['label']!r} draws outside its "
                        f"strip: [{drawn.min():.3f}, {drawn.max():.3f}] vs "
                        f"xlim ({axis_lo:.3f}, {axis_hi:.3f})")
        if any(t < axis_lo or t > axis_hi for t in xticks):
            raise AssertionError(
                f"forest ticks {xticks} fall outside the derived xlim "
                f"({axis_lo:.3f}, {axis_hi:.3f})")
        sub.set_xticks(xticks)
        if xlabel:
            # A tight pad on both strips: the upper strip's label lives in
            # the inter-strip band just above the lower strip's hanging
            # group header, and the pad is measured from the tick labels,
            # so pulling it in cannot collide with them.
            sub.set_xlabel(xlabel, labelpad=1.5)
        style_panel(sub)
        sub.tick_params(axis="y", length=0, labelsize=PT_SMALL)
        sub.spines["left"].set_visible(False)
    return ax


# ── the canvas ───────────────────────────────────────────────────────────
# Three rows on one 12-module grid: the schematic band, the three ladders,
# and the schematic-plus-forest row.  No panel carves
# its own reserve: every left reserve is the column lock the canvas
# measures, topped up by ``_equalise_row`` so panels of one row that start
# in different grid columns still share one axes-box width.
CANVAS_H_PT = 428.0                       # common 0.92-textwidth placement
ROW_H_PT = (75.0, 87.0, 170.0)
VGUTTER_PT = 38.0


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
                          hgutter_pt=24.0, vgutter_pt=VGUTTER_PT,
                          margins=Margins(left=48.0, right=12.0, top=18.0,
                                          bottom=25.0))

    ax_task = canvas.panel("task", 0, 0, 3, schematic=True, letter="",
                           inset_pt=(2.0, 2.0, 2.0, 2.0), lock=False)
    card_axes = []
    for index, (name, *_rest) in enumerate(RESOLUTION_CARDS):
        card_axes.append(canvas.panel(f"card_{name}", 0, 3 * (index + 1), 3,
                                      schematic=True, letter="",
                                      inset_pt=(2.0, 2.0, 2.0, 2.0)))
    ax_a = canvas.panel("mnist", 1, 0, 3, title="MNIST", letter="B",
                        inset_pt=(1.0, 1.0, 1.0, 1.0))
    ax_b = canvas.panel("gradient", 1, 3, 3,
                        title="Gradient cosine", letter="",
                        inset_pt=(1.0, 1.0, 1.0, 1.0))
    ax_c = canvas.panel("transport", 1, 6, 3,
                         title="Transport by depth",
                         letter="", inset_pt=(1.0, 1.0, 1.0, 1.0))
    ax_cv = canvas.panel("path_specific", 1, 9, 3,
                         title="Path-specific fraction",
                         letter="", inset_pt=(1.0, 1.0, 1.0, 1.0))
    ax_d = canvas.panel("schematic", 2, 0, 5, schematic=True, letter="")
    ax_e = canvas.panel("forest", 2, 5, 7, title="Paired MNIST contrasts",
                        letter="", inset_pt=(0.0, 2.0, 0.0, 0.0))
    # The forest is two inset strips on a switched-off host, so the column
    # lock can no longer measure its row labels off host ticks; the label
    # column is declared instead, sized for its widest row name plus the
    # header overhang.
    canvas.declare_reserve("forest", left=FOREST_LABEL_RESERVE_PT)
    canvas.add_letter("A", ax_task)
    # The rotated gradient-axis label reaches the top of its panel; place C
    # in the inter-panel gutter so the two glyphs cannot collide.
    canvas.add_letter("C", ax_b, dx_pt=46.0)
    canvas.add_letter("D", ax_c, dx_pt=46.0)
    canvas.add_letter("E", ax_cv, dx_pt=46.0)
    canvas.add_letter("F", ax_d)
    canvas.add_letter("G", ax_e)

    panel_mnist_ladder(ax_a)
    panel_gradient(ax_b)
    panel_transport_profile(ax_c)
    panel_path_specific_energy(ax_cv)
    panel_ownership_address(ax_d)
    panel_forest(ax_e)

    # The ladder row starts in three different grid columns; measure the
    # first lock, equalise the row, then draw the schematic cells on final
    # geometry (their Frames capture the axes box at draw time).
    canvas.lock_reserves()
    _equalise_row(canvas, {"mnist": 0, "gradient": 3, "transport": 6,
                           "path_specific": 9})
    canvas.lock_reserves()
    panel_task(ax_task)
    for ax_card, (_name, title, tone, gloss, draw) in zip(card_axes,
                                                          RESOLUTION_CARDS):
        panel_resolution_card(ax_card, title, tone, gloss, draw)

    target = Path(path) if path else COMPONENTS / "main_figure_02_native.pdf"
    from journal_style import style_direct_color_labels
    style_direct_color_labels(canvas.fig)
    problems = canvas.save(target, name="main_figure_02_native")
    for problem in problems:
        print(f"    {problem}")
    return problems


if __name__ == "__main__":
    raise SystemExit(1 if build() else 0)

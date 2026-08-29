#!/usr/bin/env python3
"""Main Figure 3 -- credit-operator phase theory -- as ONE native canvas.

The figure is authored at exactly ``journal_style.FIG_W`` (518.4 pt) and
emitted at scale 1.0, so every tick label is 7.6 pt on the compiled page and
every panel in a row is the same physical size.  Nothing here recomputes a
result: the panels read the same frozen source tables, apply the same
pivots and the same summary statistics as
``scripts/build_credit_phase_figure.py`` (panels A, B, D, E, F, I) and
``scripts/build_main_panel_redesigns.build_phase_plane`` (the phase plane),
whose sub-blocks the compact assembler used to scale into grid slots.

Layout (12-column module, one horizontal and one vertical gutter)::

    row 0   A operator schematic  | B spectral advantage | C route-resolution
    row 1   D projection boundary | E reliability gains
    row 2   F predictive utility  | G alignment x bandwidth

The letters run in reading order and A-E now run as one uninterrupted
theoretical argument -- operator utility, spectral alignment, hierarchy x
resolution, the signal-noise boundary, the reliability gain -- before F
validates the theory against observed one-step progress and G synthesises
the plane.  The observed-progress scatter used to sit at C, inside the
theory sequence; moving it to F costs it nothing but the shape of its slot
(it now reads across six modules instead of four, and the route-resolution
curves read across four instead of six).

Three module widths only -- four modules in row 0, six in rows 1 and 2 -- so
every panel of a row owns an identical axes box and the whole figure starts
at one of four column edges.  The synthesis panel G used to run the full
twelve modules as a 474 x 124 pt letterbox strip: normalised by the modules
it spans it was the largest panel on the page and its aspect was outside any
sane band.  It now takes the same six-module share as its row-mate, which
costs it nothing (its five families occupy a small part of the plane) and
brings the module-normalised area spread of the figure to 1.16x.

Every explanatory or methodological note that used to be printed inside a
panel -- the marker dodges in C and E, the misrouted-control callout in F,
the alignment rule and the filled/open marker key in G -- now lives in the
caption, which is where the reader looks for how a panel was drawn.  The two
5x5 signed heatmaps still share one diverging treatment centred on zero, one
cell-annotation format and one geometrically identical colour key, drawn as
a horizontal rail inside the panel's own height rather than as a vertical bar
carved out of its width.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from credit_tree_schematics import draw_credit_tree
from figure_canvas import (
    COLORS,
    DIV_CMAP,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    MARKERS,
    PT_ANNOT,
    PT_LABEL,
    PT_LEGEND,
    PT_SMALL,
    PT_TICK,
    SEQ_CMAP,
    Margins,
    NativeCanvas,
    audit_native_pdf,
)
from journal_style import annotate_heatmap


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "source_data" / "credit_phase_theory"
EXISTING = ROOT / "source_data" / "credit_phase_existing"
FACTORIAL = ROOT / "source_data" / "trained_subtree_address_full_factorial"
PLANE = ROOT / "source_data" / "credit_phase_plane"
OUT = ROOT / "figures" / "components" / "main_figure_03_native.pdf"

HEIGHT_IN = 442.0 / 72.0
# Every reserve this figure needs is paid for by the outer margins and by the
# two uniform gutters, never by a slice of one panel: the horizontal gutter
# carries the next panel's y label and tick column (34.8 pt for the widest
# panel) and the panel letter that sits left of it, and the vertical gutter
# carries a row's x label band plus the next row's letter and title band.  So
# no panel is ever carved on its own and every panel that starts in one grid
# column keeps one x0 and one axes width.
HGUTTER = 38.0
VGUTTER = 58.0
MARGINS = Margins(left=51.0, right=12.0, top=23.0, bottom=27.0)
# Row 1 carries the sweep panels and gets the extra 6 pt of height; the module
# grid does the rest, so the module-normalised areas stay inside 1.16x.
ROW_WEIGHTS = (106.0, 112.0, 106.0)

# The colour key of a signed heatmap, as a rail reserved along the TOP of the
# panel's own module allocation.  Every dimension is in points and is shared
# by both heatmaps, so B and D carry one identical key and keep the full
# 4-module axes box their row-mates get.
KEY_BAR_PT = 98.0         # bar length
KEY_BAR_H_PT = 4.5        # bar thickness (the old vertical rail's width)
KEY_TOP_PT = 1.0          # axes top -> key label
KEY_LABEL_PT = 9.5        # key-label line
KEY_LABEL_GAP_PT = 1.5    # key label -> bar
KEY_TICK_PT = 2.0         # tick length below the bar
KEY_TICKGAP_PT = 1.0      # tick -> tick number
KEY_TICKLAB_PT = 9.0      # tick-number line
KEY_GRID_GAP_PT = 5.5     # key -> matrix
KEY_STRIP_PT = (KEY_TOP_PT + KEY_LABEL_PT + KEY_LABEL_GAP_PT + KEY_BAR_H_PT
                + KEY_TICK_PT + KEY_TICKGAP_PT + KEY_TICKLAB_PT
                + KEY_GRID_GAP_PT)

def matrix_label_y(ax) -> float:
    """Axes-fraction height of the MATRIX centre in a keyed heatmap panel.

    The colour key is a rail reserved inside the axes box, so the grid only
    occupies the lower ``h_pt - KEY_STRIP_PT`` points of it.  ``set_ylabel``
    resets the label to y=0.5 -- the centre of the whole box -- unless the
    caller passes ``y``, which left the label ~15 pt above the grid it names.
    """
    figure = ax.get_figure()
    h_pt = ax.get_position().height * figure.get_size_inches()[1] * 72.0
    return 0.5 * (1.0 - KEY_STRIP_PT / h_pt)


# One colour per family, fixed for the whole figure.
C_ROUTE = COLORS["shunting"]      # the anatomy/route-of-interest series
C_CTRL = COLORS["point_mlp"]      # neutral control gray
C_SHUFFLE = COLORS["highlight"]   # shuffle rose (palette semantics)
C_ANTI = COLORS["additive"]       # anti-aligned route
C_ORACLE = COLORS["oracle"]
C_BP = COLORS["bp"]
# Task hierarchy H is an ordered factor, so it gets the sequential ramp
# rather than four unrelated hues (S5).
H_COLORS = [SEQ_CMAP(v) for v in (0.34, 0.56, 0.78, 1.00)]


class _MinusFmt:
    """Heatmap cell values with a true minus sign (and never a bare "-0")."""

    def __init__(self, spec: str = "{:.2f}", half_away: bool = False):
        self.spec = spec
        self.half_away = half_away

    def format(self, value: float) -> str:
        v = float(value)
        if self.half_away:
            v = math.copysign(math.floor(abs(v) + 0.5), v)
        s = self.spec.format(v)
        if s.lstrip("-").strip("0").strip(".") == "":
            s = s.lstrip("-")
        return s.replace("-", "−")


def signed_heatmap(ax, matrix, xlabels, ylabels, *, label):
    """The figure's ONE signed-heatmap treatment: DIV, zero-centred, keyed.

    The matrix keeps the panel's whole axes box -- the full 4-module share
    every other panel in its row gets -- and the colour key takes a rail
    reserved along the top of that same box.  A vertical bar on the right
    cannot do this: its bar, tick numbers and rotated label need 46 pt, and
    there is no thirteenth module to pay for them, so the rail can only come
    out of the matrix itself.  Taken vertically the same furniture is paid
    for out of the panel's own height, and both heatmaps print at their
    row-mates' width with one identical key.
    """
    limit = float(np.max(np.abs(matrix)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=DIV_CMAP,
                      norm=norm)
    ax.set_xticks(np.arange(len(xlabels)), xlabels)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    annotate_heatmap(ax, image, matrix, fmt=_MinusFmt("{:.0f}", True),
                     fontsize=PT_SMALL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0.0)

    # The key rail, in points converted to data units from the panel's own
    # axes box, so both heatmaps get a geometrically identical key.
    rows, cols = matrix.shape
    fig_w, fig_h = ax.figure.get_size_inches()
    box = ax.get_position()
    w_pt = box.width * fig_w * 72.0
    h_pt = box.height * fig_h * 72.0
    ux = cols / w_pt                          # data units per point, x
    uy = rows / (h_pt - KEY_STRIP_PT)         # data units per point, y
    top = rows - 0.5 + KEY_STRIP_PT * uy
    bar_top = top - (KEY_TOP_PT + KEY_LABEL_PT + KEY_LABEL_GAP_PT) * uy
    bar_bottom = bar_top - KEY_BAR_H_PT * uy
    centre = 0.5 * (cols - 1)
    bar_x0 = centre - 0.5 * KEY_BAR_PT * ux
    bar_x1 = centre + 0.5 * KEY_BAR_PT * ux

    ramp = np.linspace(-limit, limit, 256)[None, :]
    ax.imshow(ramp, origin="lower", aspect="auto", cmap=DIV_CMAP, norm=norm,
              extent=(bar_x0, bar_x1, bar_bottom, bar_top),
              interpolation="bilinear", zorder=3)
    ax.add_patch(Rectangle(
        (bar_x0, bar_bottom), bar_x1 - bar_x0, bar_top - bar_bottom,
        facecolor="none", edgecolor=COLORS["edge"], lw=LW_EDGE, zorder=4))

    # The two 5x5 grids present as a matched pair but normalise to their own
    # maxima, so each key declares its own end value: nobody should compare a
    # hue across the two panels.
    fmt = _MinusFmt("{:.0f}", True)
    step = limit / 2.0
    tick_y = bar_bottom - KEY_TICK_PT * uy
    label_y = tick_y - KEY_TICKGAP_PT * uy
    for value in (-limit, -step, 0.0, step, limit):
        x = bar_x0 + (value + limit) / (2.0 * limit) * (bar_x1 - bar_x0)
        ax.add_patch(Rectangle(
            (x - 0.5 * LW_EDGE * ux, tick_y), LW_EDGE * ux,
            KEY_TICK_PT * uy, facecolor=COLORS["edge"], edgecolor="none",
            zorder=4))
        ax.text(x, label_y, fmt.format(value), ha="center", va="top",
                fontsize=PT_TICK, color=COLORS["ink"])
    ax.text(centre, top - (KEY_TOP_PT + 0.5 * KEY_LABEL_PT) * uy, label,
            ha="center", va="center", fontsize=PT_LABEL, color=COLORS["ink"])

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, top)
    # The y label belongs to the matrix, not to the matrix plus its key rail.
    ax.yaxis.label.set_y(0.5 * (h_pt - KEY_STRIP_PT) / h_pt)
    return image


# ── A: the credit operator ───────────────────────────────────────────────
def operator_schematic(ax) -> None:
    """Restricted-route credit tree feeding the guaranteed-utility ratio.

    The drawing is the shared credit-tree vocabulary in *address* mode (the
    K = 4 nested route capsules this figure manipulates): the task gradient
    enters at the soma, the route capsules carry it, and the routed update
    leaves at the canopy.  Underneath, on a recessive card that spans the
    cell, sits the guarantee the phase panels sweep.  The world frame is
    sized in points from the cell itself, so the schematic fills its cell
    instead of floating inside it.
    """
    fig_w, fig_h = ax.figure.get_size_inches()
    box = ax.get_position()
    w_pt = box.width * fig_w * 72.0
    h_pt = box.height * fig_h * 72.0
    unit = 15.5                                   # points per tree unit
    x_lo = -2.62
    x_hi = x_lo + w_pt / unit
    y_hi = 3.50
    y_lo = y_hi - h_pt / unit
    draw_credit_tree(ax, mode="address", K=4, labels=False, scale=0.85,
                     xlim=(x_lo, x_hi), ylim=(y_lo, y_hi))

    # Right column, top to bottom: the operator flow.
    lab_x = 4.35
    ax.add_patch(FancyArrowPatch(
        (2.32, 2.30), (3.32, 2.70), arrowstyle="-|>", mutation_scale=7,
        connectionstyle="arc3,rad=0.12", lw=LW_REF, color=COLORS["ink"],
        capstyle="round", zorder=4.5))
    ax.text(3.46, 2.66, "update\n−η M(μ + ξ)", ha="left",
            va="center", fontsize=PT_SMALL, color=COLORS["ink"],
            linespacing=1.25)
    # An arrow, like this panel's two other annotations, and stopped short of
    # the label: as a bare line starting at 3.55 it ran 2.4 pt into the "routes
    # M" text and pointed at nothing in particular.  The head now lands on the
    # route capsules the label names.
    ax.add_patch(FancyArrowPatch(
        (3.20, 1.47), (2.34, 1.79), arrowstyle="-|>", mutation_scale=7,
        connectionstyle="arc3,rad=-0.10", lw=LW_REF, color=C_ROUTE,
        capstyle="round", zorder=4.5))
    ax.text(lab_x, 1.45, "routes M", ha="center", va="center",
            fontsize=PT_SMALL, color=C_ROUTE)
    ax.add_patch(FancyArrowPatch(
        (2.95, 0.14), (0.36, 0.02), arrowstyle="-|>", mutation_scale=7,
        connectionstyle="arc3,rad=-0.10", lw=LW_REF, color=C_ORACLE,
        capstyle="round", zorder=4.5))
    ax.text(lab_x, 0.32, "mean\ngradient μ", ha="center", va="center",
            fontsize=PT_SMALL, color=C_ORACLE, linespacing=1.25)

    # The guarantee, on a recessive card that spans the cell.
    card_x0, card_x1 = x_lo + 0.06, x_hi - 0.06
    card_y1 = -0.30
    card_y0 = -1.72
    ax.add_patch(FancyBboxPatch(
        (card_x0, card_y0), card_x1 - card_x0, card_y1 - card_y0,
        boxstyle="round,pad=0.10", facecolor=COLORS["panel_bg"],
        edgecolor=COLORS["grid"], lw=LW_HAIR, zorder=0.5))
    # Render the complete ratio as one mathematical object.  Independent text
    # lines collide after physical-point coordinates are converted into this
    # schematic's tree coordinate system.
    frac_x = 0.5 * (card_x0 + card_x1)
    ax.text(
        frac_x, card_y1 - 0.73,
        r"$U(M)=\frac{[\mu^{\mathsf{T}}M\mu]^2}"
        r"{2L\,[\Vert M\mu\Vert_2^2+\mathrm{tr}(M\Sigma M^{\mathsf{T}})]}$",
        ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["ink"],
    )


def main() -> None:
    spectral = pd.read_csv(PHASE / "spectral_phase_summary.csv")
    depth = pd.read_csv(PHASE / "depth_training_summary.csv")
    projection = pd.read_csv(PHASE / "projection_phase_summary.csv")
    reliability = pd.read_csv(PHASE / "reliability_phase_summary.csv")
    operator = pd.read_csv(EXISTING / "operator_metrics.csv")
    outcomes = pd.read_csv(FACTORIAL / "seed_outcomes.csv")
    audit = json.loads((EXISTING / "summary.json").read_text())
    points = pd.read_csv(PLANE / "points.csv", keep_default_na=False)

    canvas = NativeCanvas(
        HEIGHT_IN, 3, row_weights=list(ROW_WEIGHTS),
        hgutter_pt=HGUTTER, vgutter_pt=VGUTTER, margins=MARGINS,
    )
    ax_a = canvas.panel("A", 0, 0, 4, schematic=True,
                        title="Credit-operator utility")
    ax_b = canvas.panel("B", 0, 4, 4, title="Spectral alignment")
    ax_c = canvas.panel("C", 0, 8, 4, title="Route-resolution crossover")
    ax_d = canvas.panel("D", 1, 0, 6, title="Projection boundary")
    ax_e = canvas.panel("E", 1, 6, 6, title="Reliability gains")
    ax_f = canvas.panel("F", 2, 0, 6, title="Predictive utility")
    ax_g = canvas.panel("G", 2, 6, 6, title="Alignment × bandwidth")

    # ── A ───────────────────────────────────────────────────────────────
    operator_schematic(ax_a)

    # ── B: ancestry-minus-random spectral capture (signed) ──────────────
    wide = spectral.pivot_table(
        index=["alignment", "budget_k"], columns="method",
        values="mean_spectral_capture")
    advantage = (wide.ancestry - wide.random_rank).unstack("budget_k")
    signed_heatmap(
        ax_b, advantage.to_numpy() * 100.0,
        [str(v) for v in advantage.columns],
        [f"{v:.2f}" for v in advantage.index],
        label="capture advantage (×10⁻²)")
    ax_b.set_xlabel("route budget K   (16 = full rank)", labelpad=2.0)
    ax_b.set_ylabel("task–tree alignment ρ", labelpad=1.5,
                    y=matrix_label_y(ax_b), ha="center")

    # ── C: final loss versus route resolution, one series per task depth ─
    depth_marker_dx = {3: -0.20, 4: 0.20}
    aligned = depth[depth.method.eq("aligned_tree")].pivot(
        index="task_depth", columns="model_depth",
        values="mean_final_population_loss")
    for task_depth, color, marker in zip(sorted(aligned.index), H_COLORS,
                                         MARKERS):
        part = depth[depth.method.eq("aligned_tree")
                     & depth.task_depth.eq(task_depth)].sort_values(
                         "model_depth")
        xs = part.model_depth.to_numpy(dtype=float)
        ys = part.mean_final_population_loss.to_numpy(dtype=float)
        # Only the MARKER column is dodged, exactly as panel E does it: the
        # line, and the 95% ribbon under it, stay at the true x, so a ribbon
        # can never sit beside the mean it belongs to and no segment of a
        # log-loss curve is given a distorted run.
        fanned = (np.isin(xs, (1.0, 2.0)) if task_depth in depth_marker_dx
                  else np.zeros(xs.shape, dtype=bool))
        ax_c.plot(xs, ys, color=color, marker=marker, ms=MARKER_MS,
                  lw=LW_DATA, mec="white", mew=LW_HAIR,
                  markevery=list(np.flatnonzero(~fanned)),
                  label=rf"$H_{{\rm c}}={task_depth}$")
        if fanned.any():
            ax_c.scatter(xs[fanned] + depth_marker_dx[task_depth], ys[fanned],
                         color=color, marker=marker, s=MARKER_MS ** 2,
                         zorder=3, edgecolors="white", linewidths=LW_HAIR)
        ax_c.fill_between(part.model_depth,
                          part.ci95_low_final_population_loss,
                          part.ci95_high_final_population_loss, color=color,
                          alpha=0.10, linewidth=0)
    ax_c.set_yscale("log")
    ax_c.set_yticks([0.03, 0.1, 0.3, 1.0, 3.0])
    ax_c.set_yticklabels(["0.03", "0.1", "0.3", "1", "3"])
    ax_c.set_xticks([1, 2, 3, 4])
    ax_c.set_xlim(0.70, 4.30)
    ax_c.set_xlabel("route resolution Dᵣ   (4 = full rank)")
    ax_c.set_ylabel("final loss")
    ax_c.legend(loc="upper left", ncol=2, frameon=False, fontsize=PT_LEGEND,
                handlelength=1.3, handletextpad=0.4, labelspacing=0.25,
                columnspacing=0.8, borderaxespad=0.2)

    # ── D: signed loss change of the common route projection ────────────
    projection_wide = projection.pivot_table(
        index=["signal_retention", "noise_retention"], columns="method",
        values="mean_expected_population_loss")
    delta = (projection_wide.bp_plus_route_projection
             - projection_wide.full_stochastic_bp).unstack("signal_retention")
    signed_heatmap(
        ax_d, delta.to_numpy() * 100.0,
        [f"{v:.2f}" for v in delta.columns],
        [f"{v:.2f}" for v in delta.index],
        label="Δ loss (×10⁻²)")
    ax_d.plot([-0.5, 0.5, 0.5, 1.5, 1.5, 3.5, 3.5],
              [1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5],
              ls="--", color=COLORS["mute"], lw=LW_REF, zorder=3)
    ax_d.set_xlabel(r"retained signal fraction  $f_{\rm sig}$", labelpad=2.0)
    ax_d.set_ylabel(r"retained noise fraction  $f_{\rm noise}$", labelpad=1.5,
                    y=matrix_label_y(ax_d), ha="center")

    # ── E: one-step reliability gains for the four gain policies ────────
    reliability_styles = [
        ("reliability_aligned", "SNR-aligned", C_ROUTE, MARKERS[0]),
        ("best_global_gain", "best global", C_CTRL, MARKERS[1]),
        ("shuffled_shunting", "shuffled", C_SHUFFLE, MARKERS[2]),
        ("anti_aligned_shunting", "anti-aligned", C_ANTI, MARKERS[3]),
    ]
    reliability_marker_dx = {
        "reliability_aligned": -0.165, "best_global_gain": -0.055,
        "shuffled_shunting": 0.055, "anti_aligned_shunting": 0.165,
    }
    for method, label, color, marker in reliability_styles:
        part = reliability[reliability.method.eq(method)].sort_values(
            "reliability_heterogeneity")
        xs = part.reliability_heterogeneity.to_numpy(dtype=float)
        ys = part.mean_population_loss_decrease.to_numpy(dtype=float)
        fanned = xs == 0.0
        ax_e.plot(xs, ys, color=color, marker=marker, ms=MARKER_MS,
                  lw=LW_DATA, label=label,
                  markevery=list(np.flatnonzero(~fanned)),
                  mec="white", mew=LW_HAIR)
        if fanned.any():
            ax_e.scatter(xs[fanned] + reliability_marker_dx[method],
                         ys[fanned], color=color, marker=marker,
                         s=MARKER_MS ** 2, zorder=3, edgecolors="white",
                         linewidths=LW_HAIR)
        ax_e.fill_between(part.reliability_heterogeneity,
                          part.ci95_low_population_loss_decrease,
                          part.ci95_high_population_loss_decrease,
                          color=color, alpha=0.08, linewidth=0)
    ax_e.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_e.margins(x=0.115)
    # Pinned, not auto-located: one tick per swept heterogeneity level, so the
    # axis reads the same whatever width the module grid gives this panel.
    ax_e.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_e.set_ylim(-4.0, 3.9)
    ax_e.set_xlabel("branch-SNR heterogeneity")
    ax_e.set_ylabel("one-step loss decrease")
    ax_e.legend(loc="lower left", ncol=1, frameon=False, fontsize=PT_LEGEND,
                handlelength=1.3, handletextpad=0.4, labelspacing=0.28,
                borderaxespad=0.2)

    # ── F: phase utility predicts observed one-step progress ────────────
    merged = operator.merge(
        outcomes,
        on=["seed", "condition_id", "architecture", "feedback_family",
            "budget_k"],
        validate="one_to_one")
    merged = merged[merged.architecture.eq("dendritic_tree")]
    ax_f.scatter(merged.maximum_guaranteed_decrease,
                 merged.norm_matched_one_step_progress,
                 s=7, alpha=0.30, edgecolors="none", color=COLORS["additive"])
    ax_f.set_ylim(-1.42, 1.22)
    ax_f.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    # The Spearman coefficient, its seed-block interval and n are reported in
    # the caption; printing them on the panel duplicated the caption text.
    ax_f.set_xlabel("operator utility")
    ax_f.set_ylabel("observed progress")

    # ── G: alignment x bandwidth synthesis ──────────────────────────────
    families = [
        ("factorial", "trained bandwidth", C_ROUTE, MARKERS[0], True),
        ("sweep", "spectral theory", COLORS["additive"], MARKERS[1], True),
        ("microns", "imposed alignment", C_ORACLE, MARKERS[2], True),
        ("measured", None, C_CTRL, MARKERS[3], False),
        ("reversal", "two-stream reversal (S19G)", C_BP, MARKERS[4], False),
    ]
    ax_g.set_yscale("log")
    ax_g.set_xlim(-0.045, 1.12)
    ax_g.set_ylim(0.088, 7.0)
    span_x = [-0.045, 1.12]
    # K/r_eff is an ordinal axis, so the three regimes take an ordered light slate
    # ramp (lightest at low K/r) rather than three near-invisible hues at
    # alpha 0.07-0.10, which were impossible to tell apart and competed with
    # the condition colours of the data.
    for lo, hi, tint in ((0.088, 0.24, "#F2F4F7"),
                         (0.24, 1.2, "#E3E8ED"),
                         (1.2, 7.0, "#D4DBE3")):
        ax_g.fill_between(span_x, [lo] * 2, [hi] * 2, color=tint,
                          zorder=0, linewidth=0)
    # Same reference treatment as panels D and E: dashed, mute, LW_REF.
    ax_g.axhline(1.0, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=1)
    # Three observations sit at alignment exactly 1 and are drawn fanned in x
    # by the frozen source table; the rule marks where alignment 1 really is.
    ax_g.axvline(1.0, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=1)
    for family, label, color, marker, has_line in families:
        part = points[points.family.eq(family)].sort_values("x_plot")
        if has_line:
            ax_g.plot(part.x_plot, part.y_plot, color=color, lw=LW_DATA,
                      alpha=0.85, zorder=2)
        for row in part.itertuples(index=False):
            filled = str(row.outcome) not in {"loss", "null"}
            ax_g.plot(row.x_plot, row.y_plot, marker=marker, ms=MARKER_MS,
                      ls="none", mfc=color if filled else "white", mec=color,
                      mew=LW_REF, zorder=4)
    # The lone measured-response diamond is already labelled directly, so it
    # is not repeated as a key entry (S6).
    handles = [Line2D([], [], color=color, marker=marker,
                      lw=LW_DATA if has_line else 0, markersize=MARKER_MS,
                      markeredgewidth=LW_REF, label=label)
               for _, label, color, marker, has_line in families
               if label is not None]
    ax_g.legend(handles=handles, loc="upper left", ncol=1, frameon=False,
                fontsize=PT_LEGEND, handlelength=1.3, handletextpad=0.4,
                labelspacing=0.28, borderaxespad=0.15)
    # Regime names label the quiet background bands directly, in their
    # shortest unambiguous form; the bands themselves are defined in the
    # caption, and so are the alignment rule and the filled/open marker key
    # that used to be printed here as two blocks of running text.
    # One right-hand column, each name vertically centred in the band it
    # names, so the background tints have an unambiguous key.  "misaligned"
    # is gone: it named an x-region while its three neighbours named y-bands,
    # and it shared the green band with "matched regime".
    for band_y, band_name in ((4.20, "span saturated"),
                              (0.33, "matched regime"),
                              (0.145, "bandwidth limited")):
        ax_g.text(1.10, band_y, band_name, color=COLORS["mute"],
                  fontsize=PT_ANNOT, style="italic", ha="right", va="center")
    # Anchored in axes fractions on the right: placed in the upper left it
    # sat directly under the legend's last row and read as a fifth key entry.
    ax_g.annotate("rank-saturated null", xy=(0.474526, 3.73089),
                  xytext=(0.375, 0.52), textcoords="axes fraction", ha="left",
                  va="center", fontsize=PT_SMALL, color=C_CTRL,
                  arrowprops={"arrowstyle": "-", "color": C_CTRL,
                              "lw": LW_HAIR, "shrinkA": 1, "shrinkB": 3})
    ax_g.annotate("routing required", xy=(1.06, 1.00036), xytext=(-3, 13),
                  textcoords="offset points", ha="right", va="bottom",
                  fontsize=PT_SMALL, color=C_BP,
                  arrowprops={"arrowstyle": "-", "color": C_BP,
                              "lw": LW_HAIR})
    ax_g.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_g.set_yticks([0.125, 0.25, 0.5, 1, 2, 4])
    ax_g.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4"])
    ax_g.minorticks_off()
    ax_g.set_xlabel("task–anatomy alignment")
    # Set on two lines: rotated, one long line overran the panel and climbed
    # into the row above, past G's own letter.  Two lines stack as adjacent
    # columns, each about half as tall, and stay inside the axes.
    ax_g.set_ylabel("bandwidth / effective\ntask rank  " r"$K/r_{\rm eff}$")

    problems = canvas.save(OUT, name="main_figure_03_native")
    for violation in audit_native_pdf(OUT):
        print(f"    {violation}")
    return problems


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Main Figure 6 as ONE native full-width canvas.

Figure 6 was already authored at final size, so nothing here recomputes an
estimate: every mean, ``n``, 95 % interval and paired contrast is read from
the same frozen source tables through the same helpers as
``build_main_panel_redesigns.build_figure6``.  What this module owns is
geometry and encoding hygiene, and it now carries the two structural moves
the other main figures received.

Structure
---------
*A schematic column beside the result it explains.*  Panel A draws the H4
construction natively in the ``native_schematics`` vocabulary -- the nested
task hierarchy of depth :math:`H=4`, the ladder of physical stage counts
D1--D4 it is mapped onto, and the aligned versus reversed sensor placements
-- and sits immediately left of panel B, the accuracy matrix that tests it.
It is drawn as a landscape composition (the hierarchy across the top, the
stage ladder and the two placements beneath it) because it now owns half of
the top row rather than a third of it.

*Modest emphasis.*  The claim of this figure is that accuracy saturates once
the physical stage count reaches D3.  That matrix (B) is still the largest
panel on the page, but it is larger only by spanning more modules: six of
twelve on the top row, so it prints 1.6x the area of a panel on the rows
below instead of the 2.9x it used to take, and its module-normalised area is
within 1.07x of every other panel.  Its printed values keep the one type step
they had.

Three module widths, three column edges::

    row 0   A  H4 construction (6 modules)   B  H4, aligned placement (6)
    row 1   C  reversed placement  D  seed-paired contrasts  E  depth x hierarchy
    row 2   F  serial benefit, BP  G  serial benefit, LocalCA  H  interaction

Every panel of a row owns an identical axes box, and every reserve the figure
needs is paid for by the outer margins and the two uniform gutters rather
than by a slice of one panel: the row-label columns of B, C and F are set in
two lines so they fit the gutter or the left margin, the contrast names of D
are set inside D's own plotting rectangle above the bars they name, and both
colour keys sit in space the grid already owns -- the accuracy key in the
right margin beside B, the effect key in the gutter between the two panels it
explains.  So no panel carves its own reserve and the whole figure starts at
one of four column edges.

* B, C and E are one shared-scale small multiple: the same accuracy norm, the
  same D1--D4 column axis and one colour key;
* F and G are the second shared-scale pair, with their key between them;
* the reversed-placement control (C) takes the neutral gray title tone the
  palette reserves for controls, and the mechanism-interaction contrast in D
  keeps the neutral gray it already had;
* H recolours by *credit rule* (backprop red-brown, LocalCA amber) with two
  direct end-of-line labels instead of a legend.

The paired differences in D are read off the geometry -- point, interval and
the zero reference -- not from a printed value beside each bar; the exact
numbers live in Source Data and the running text.

Output: ``figures/components/main_figure_06_native.pdf`` (+ 600-dpi PNG),
which ``assemble_compact_main_figures.emit_native(6)`` copies to
``figures/main/figure_06.pdf`` at scale 1.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import Normalize, TwoSlopeNorm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figure_canvas import (  # noqa: E402
    COLORS,
    DIV_CMAP,
    ERR_CAPSIZE,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    Margins,
    NativeCanvas,
    PT_LABEL,
    PT_SMALL,
    PT_TICK,
    PT_TITLE,
    SEQ_CMAP,
    style_panel,
)
from credit_tree_schematics import AMBER_TEXT, mix  # noqa: E402
from native_schematics import Frame  # noqa: E402
from build_main_panel_redesigns import (  # noqa: E402
    H4_ROWS,
    _h4_matrix,
    _hierarchy_depth_matrix,
    _rgba,
    _task_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
COMPONENTS = ROOT / "figures" / "components"
OUTPUT = COMPONENTS / "main_figure_06_native.pdf"

# ── canvas geometry (points) ─────────────────────────────────────────────
HEIGHT_IN = 440.0 / 72.0            # aspect 1.18, inside the 1.05-1.55 band
# The horizontal gutter carries a panel's two-line row-label column, its
# panel letter and (between F and G) the shared effect key; the vertical
# gutter carries a row's x label band plus the next row's letter and title
# band.  Nothing is carved out of a panel, so every panel that starts in one
# grid column keeps one x0 and one axes width.
HGUTTER = 38.0
VGUTTER = 38.0
# The left margin holds the two-line row-label column of the panels that
# start a row at module 0; the right margin holds the accuracy key rail.
MARGINS = Margins(left=44.0, right=44.0, top=18.0, bottom=26.0)
ROW_WEIGHTS = (104.0, 108.0, 108.0)
RAIL_W = 6.0                        # colour-key bar width
RAIL_GAP = 4.0                      # bar left edge beyond the module grid

# The key spans the data it encodes: panel E reaches 0.967 and panel B 0.867,
# so a 0.88 ceiling printed a 0.97 cell and a 0.87 cell in the same blue.
ACCURACY_NORM = Normalize(vmin=0.52, vmax=0.97)
ACCURACY_TICKS = (0.55, 0.65, 0.75, 0.85, 0.95)
ACCURACY_TICK_LABELS = ("0.55", "0.65", "0.75", "0.85", "0.95")
EFFECT_NORM = TwoSlopeNorm(vmin=-8, vcenter=0, vmax=32)
EFFECT_TICKS = (-8, 0, 10, 20, 30)
EFFECT_TICK_LABELS = ("−8", "0", "+10", "+20", "+30")

DEPTH_LABELS = ("D1", "D2", "D3", "D4")
DEPTH_AXIS = "physical stage count"
FAMILY_LABELS = ("nested\nfactors", "flat\nfactors", "local\nratios")
# Panel H puts three categories across a 4-module slot, so it uses the short
# form; panels F and G carry the full names on the shared row-label column.
FAMILY_SHORT = ("nested", "flat", "local")
ALPHA_LABELS = ("0", "0.5", "1")
ALPHA_AXIS = "sensor alignment α"

# The five H4 conditions.  Order and meaning are exactly ``H4_ROWS``; the
# names are set in two lines so the label column of a matrix is 24 pt wide
# and fits the gutter or the outer margin the grid already provides, instead
# of being carved out of the panel and making it narrower than its row-mates.
H4_ROW_LABELS = [row[0].replace(" ", "\n") for row in H4_ROWS]

# Colour carries the CREDIT RULE throughout figure 6 (red-brown = backprop,
# amber = LocalCA, gray = the mechanism interaction), exactly as panels D, F,
# G and H use it; the row label carries the mechanism.
CONTRAST_SPECS = [
    ("depth__serial_bp__aligned__d4_d3", "serial BP, D4 − D3", "bp"),
    ("depth__serial_bp__aligned__d4_d1", "serial BP, D4 − D1", "bp"),
    ("serial_minus_grouped__aligned__d4", "serial − point, D4", "bp"),
    ("depth__shared_local__aligned__d4_d3", "shared local, D4 − D3", "local"),
    ("depth__path_local__aligned__d4_d3", "path local, D4 − D3", "local"),
    ("shunting_additive_depth_interaction__aligned__d4_d3", "shunt × depth",
     "point_mlp"),
]

# Amber is legible as a fill but not as 6.8 pt type on white, so every amber
# TEXT element on the page uses the darkened text tone.
TEXT_COLOR = {"local": AMBER_TEXT}

# Value ramp for the four nested task factors in the schematic: one hue (the
# palette's oracle violet, which already carries "nested divisive task"
# throughout the paper) at four tints, so the ladder reads as four levels of
# one construct rather than four different things.
FACTOR_TINTS = (20, 34, 48, 62)

# The hierarchy tags are set as literal Unicode subscript digits rather than
# as mathtext: every other figure in the paper does the same, and mathtext
# shrinks a subscript to 0.7 of its base (4.76 pt here), which is below the
# 6.8 pt type floor and illegible in print.
SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"


# ── drawing helpers ──────────────────────────────────────────────────────
def _num(fmt: str, value: float) -> str:
    """Format a number with a true minus sign, never a hyphen or a "+0"."""
    text = fmt.format(value).replace("-", "−")
    if text.lstrip("+−").strip("0").strip(".") == "":
        text = text.lstrip("+−")
    return text


def _heatmap(ax, matrix, col_labels, *, row_labels=None, cmap, norm, fmt,
             best_by_row=False, value_size=PT_SMALL):
    """Annotated matrix: identical encoding to the frozen Figure 6 panels."""
    masked = np.ma.masked_invalid(matrix)
    cmap_local = cmap.copy()
    cmap_local.set_bad("#F2F3F5")
    image = ax.imshow(masked, cmap=cmap_local, norm=norm, aspect="auto",
                      interpolation="nearest")
    for row in range(matrix.shape[0]):
        finite = np.flatnonzero(np.isfinite(matrix[row]))
        best = finite[np.argmax(matrix[row, finite])] if len(finite) else None
        # A row whose whole spread is under one unit of the printed precision
        # has no readable optimum, so it gets no "best" outline: the heavy box
        # would assert a best stage count the printed numbers do not support.
        if best is not None and len(finite) > 1 and (
                matrix[row, finite].max() - matrix[row, finite].min() < 0.01):
            best = None
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                ax.text(col, row, "—", ha="center", va="center",
                        fontsize=value_size, color=COLORS["mute"])
                continue
            rgba = cmap_local(norm(value))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(col, row, _num(fmt, value), ha="center", va="center",
                    fontsize=value_size,
                    color="white" if lum < 0.48 else COLORS["ink"])
            if best_by_row and col == best:
                # Drawn as a stroked path rather than a patch: it is a
                # pointer, not a datum, and the text-over-data audit rightly
                # treats patches under labels as marks.
                x0, x1 = col - 0.46, col + 0.46
                y0, y1 = row - 0.43, row + 0.43
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                        color=COLORS["ink"], lw=LW_ERR, solid_joinstyle="miter",
                        solid_capstyle="butt", zorder=5)
    ax.set_xticks(range(len(col_labels)), col_labels)
    if row_labels is None:
        ax.set_yticks(range(matrix.shape[0]), [""] * matrix.shape[0])
    else:
        ax.set_yticks(range(len(row_labels)), list(row_labels))
    ax.tick_params(axis="both", length=0, pad=2.0, labelsize=PT_SMALL)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _title(ax, text, *, color=None):
    ax.set_title(text, fontsize=PT_TITLE, loc="center", pad=3.5,
                 color=color or COLORS["ink"], fontweight="normal")


def _key_rail_left(canvas, mappable, ax, label, ticks, tick_labels):
    """Colour key in the host panel's own left reserve, ticks facing left.

    The key for a small-multiple pair belongs beside the panels it explains.
    Drawn here it abuts the left edge of the right-hand panel and looks across
    the gutter at the left-hand one, instead of sitting in a right-edge rail
    next to an unrelated bar chart.
    """
    fig = canvas.fig
    box = ax.get_position()
    # The rail stands in the uniform gutter, not in a reserve carved out of a
    # panel, so it is pushed as close to its host as the bar allows: what is
    # left of the gutter is exactly what the tick numbers and the key label
    # need, and F's own last tick label still clears them.
    x1_pt = box.x0 * canvas.width_pt - 3.0
    cax = fig.add_axes([
        (x1_pt - RAIL_W) / canvas.width_pt, box.y0,
        RAIL_W / canvas.width_pt, box.height,
    ])
    cbar = fig.colorbar(mappable, cax=cax, ticks=list(ticks))
    cbar.outline.set_linewidth(LW_EDGE)
    cbar.outline.set_edgecolor(COLORS["edge"])
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("left")
    cbar.ax.set_yticklabels(list(tick_labels))
    cbar.ax.tick_params(labelsize=PT_SMALL, width=LW_EDGE, length=2.2, pad=1.5,
                        color=COLORS["edge"], labelcolor=COLORS["ink"])
    cbar.set_label(label, fontsize=PT_LABEL, labelpad=1.5,
                   color=COLORS["ink"])
    canvas.bind_satellite(cax, ax)
    return cbar


def _key_rail(canvas, mappable, row, rowspan, label, ticks, tick_labels):
    """A slim labelled colour key in the reserved right-edge rail.

    Taking the rail out of the outer margin instead of out of a panel is what
    lets small-multiple siblings keep byte-identical widths: the 12-module
    pitch cannot be split, so a colorbar carved from one panel of a pair
    always makes that panel visibly narrower than its twin.
    """
    fig = canvas.fig
    x0, y_top, w, h = canvas.slot_pt(row, 0, canvas.module_cols, rowspan)
    rail_x = x0 + w + RAIL_GAP
    cax = fig.add_axes([
        rail_x / canvas.width_pt,
        (canvas.height_pt - y_top - h) / canvas.height_pt,
        RAIL_W / canvas.width_pt,
        h / canvas.height_pt,
    ])
    cbar = fig.colorbar(mappable, cax=cax, ticks=list(ticks))
    cbar.outline.set_linewidth(LW_EDGE)
    cbar.outline.set_edgecolor(COLORS["edge"])
    cbar.ax.set_yticklabels(list(tick_labels))
    cbar.ax.tick_params(labelsize=PT_TICK, width=LW_EDGE, length=2.2, pad=1.5,
                        color=COLORS["edge"], labelcolor=COLORS["ink"])
    cbar.set_label(label, fontsize=PT_LABEL, labelpad=3.0,
                   color=COLORS["ink"])
    return cbar


# ── A: the H4 construction, drawn natively ───────────────────────────────
def panel_construction(ax) -> None:
    """Task hierarchy depth, physical stage count, and the two placements.

    A landscape composition for a landscape cell: the nested task factors
    (violet, one hue at four tints) run across the top of the cell, and under
    them sit the ladder of physical stage counts whose D1--D4 tags are
    exactly panel B's column axis (left) and the aligned/reversed sensor maps
    in the route green and the control gray (right).  Everything is laid out
    in points measured down from the top of the cell, so the drawing keeps its
    clearances whatever slot it is given.
    """
    f = Frame(ax, labels=True)
    green = COLORS["shunting"]
    mute = COLORS["mute"]
    height = f.h_pt

    def y(pt: float) -> float:
        """Points measured down from the top of the cell -> y fraction."""
        return 1.0 - pt / height

    # 1. the task: four nested factors, across the top of the cell -- the
    #    fourth hierarchy this figure adds to the H2/H3 cohorts of Figure 5.
    f.text((0.5, y(6.0)), r"task hierarchy $H=4$", size=PT_SMALL)
    for index in range(4):
        top = 13.0 + index * 8.8
        f.group((0.02, y(top + 7.0), 0.96 - 0.175 * index, f.fy(7.0)),
                tint=mix("oracle", FACTOR_TINTS[index]),
                edge=mix("oracle", 62), lw=LW_EDGE, radius_pt=1.5,
                zorder=1.0)
        f.text((0.02 + f.fx(8.0), y(top + 3.5)),
               f"G{SUBSCRIPT_DIGITS[index + 1]}", size=PT_SMALL)

    # 2. bottom left: the ladder of physical stage counts the task is mapped
    #    onto, tagged with panel B's own column axis.
    f.text((0.225, y(57.0)), "physical stages", size=PT_SMALL)
    base = 92.0
    for index, depth in enumerate((1, 2, 3, 4)):
        cx = 0.055 + 0.115 * index
        for level in range(depth):
            top = base - (level + 1) * 4.8 - level * 1.5
            f.group((cx - f.fx(6.5), y(top + 4.8), f.fx(13.0), f.fy(4.8)),
                    tint=mix("shunting", 26), edge=green, lw=LW_EDGE,
                    radius_pt=1.1, zorder=1.0)
        f.text((cx, y(97.0)), f"D{depth}", size=PT_SMALL)

    # 3. bottom right: the two placements -- which factor each stage sees.
    f.text((0.77, y(57.0)), "sensor placement", size=PT_SMALL)
    for x0, x1, color, name, order in (
        (0.565, 0.725, green, "aligned", (0, 1, 2, 3)),
        (0.815, 0.975, mute, "reversed", (3, 2, 1, 0)),
    ):
        for src, dst in enumerate(order):
            ax.plot([x0, x1], [y(70.0 + 5.5 * src), y(70.0 + 5.5 * dst)],
                    color=color, lw=f.lw(LW_HAIR), solid_capstyle="round",
                    zorder=2)
        for index in range(4):
            f.disc((x0, y(70.0 + 5.5 * index)), 1.5,
                   fill=mix("oracle", FACTOR_TINTS[index]),
                   edge=mix("oracle", 62), lw=LW_HAIR, zorder=4)
            f.disc((x1, y(70.0 + 5.5 * index)), 1.5,
                   fill=mix("shunting", 55), edge=green, lw=LW_HAIR,
                   zorder=4)
        f.text(((x0 + x1) / 2.0, y(97.0)), name, size=PT_SMALL, color=color)


# ── the figure ───────────────────────────────────────────────────────────
def build() -> list[str]:
    summary = pd.read_csv(
        SOURCE / "physical_depth_h4_factorial" / "condition_summary.csv"
    )
    contrasts = pd.read_csv(
        SOURCE / "physical_depth_h4_factorial" / "paired_contrasts.csv"
    ).set_index("contrast")
    h4_seed = pd.read_csv(
        SOURCE / "physical_depth_h4_factorial" / "seed_outcomes.csv"
    )
    task_effects = pd.read_csv(
        SOURCE / "task_family_alignment" / "architecture_effects.csv"
    )
    task_contrasts = pd.read_csv(
        SOURCE / "task_family_alignment" / "paired_contrasts.csv"
    )

    canvas = NativeCanvas(HEIGHT_IN, 3, row_weights=list(ROW_WEIGHTS),
                          hgutter_pt=HGUTTER, vgutter_pt=VGUTTER,
                          margins=MARGINS)

    # Row 0: the design, then the headline matrix it explains.  Six modules
    # each: B is the largest panel on the page by spanning more modules, not
    # by taking space out of its neighbours.
    ax_a = canvas.panel("A", 0, 0, 6, schematic=True, letter="")
    ax_b = canvas.panel("B", 0, 6, 6, letter="")
    # Row 1: the placement control, the inferential summary, the sweep.
    ax_c = canvas.panel("C", 1, 0, 4, letter="")
    ax_d = canvas.panel("D", 1, 4, 4, grid="x", letter="")
    style_panel(ax_d, grid="x", spines=("bottom",))
    ax_e = canvas.panel("E", 1, 8, 4, letter="")
    # Row 2: the task-family experiment.
    ax_f = canvas.panel("F", 2, 0, 4, letter="")
    ax_g = canvas.panel("G", 2, 4, 4, letter="")
    ax_h = canvas.panel("H", 2, 8, 4, grid="y", letter="")
    # One letter offset per column: wide where a two-line row-label column
    # occupies the gutter or the margin, narrow where nothing does.
    for name, dx in (("A", 34.0), ("B", 34.0), ("C", 34.0), ("D", 14.0),
                     ("E", 30.0), ("F", 34.0), ("G", 14.0), ("H", 30.0)):
        canvas.add_letter(name, canvas.axes[name], dx_pt=dx)

    # ── A: the manipulation this figure tests.
    panel_construction(ax_a)
    _title(ax_a, "H4 factorial design")

    # ── B: the headline. H4 accuracy under aligned sensor placement, on the
    # widest slot of the tallest row, with its values one type step up.
    image_acc = _heatmap(ax_b, _h4_matrix(summary, "aligned"), DEPTH_LABELS,
                         row_labels=H4_ROW_LABELS, cmap=SEQ_CMAP,
                         norm=ACCURACY_NORM, fmt="{:.2f}", best_by_row=True,
                         value_size=PT_LABEL)
    _title(ax_b, "H4, aligned placement")
    ax_b.set_xlabel(DEPTH_AXIS, fontsize=PT_LABEL, color=COLORS["ink"])

    # ── C: the placement control, same rows, same scale, neutral title.
    _heatmap(ax_c, _h4_matrix(summary, "rewired_tree"), DEPTH_LABELS,
             row_labels=H4_ROW_LABELS, cmap=SEQ_CMAP, norm=ACCURACY_NORM,
             fmt="{:.2f}", best_by_row=True)
    _title(ax_c, "Reversed placement", color=COLORS["mute"])
    ax_c.set_xlabel(DEPTH_AXIS, fontsize=PT_LABEL, color=COLORS["ink"])

    # ── D: seed-paired frozen contrasts on one effect-size axis.
    # The contrast name is set inside the panel, above the bar it names, so
    # the row-label column costs the panel nothing and D keeps the same axes
    # box as C and E.  No value is printed beside a bar: the point, its
    # interval and the zero reference already report the difference, and the
    # exact numbers live in Source Data and the running text.
    y = np.arange(len(CONTRAST_SPECS))[::-1]
    ax_d.axvline(0, color=COLORS["mute"], lw=LW_REF, ls="--", zorder=0)
    for yi, (name, label, key) in zip(y, CONTRAST_SPECS, strict=True):
        color = COLORS[key]
        row = contrasts.loc[name]
        mean = float(row.mean_pp)
        low = float(row.ci_low_pp)
        high = float(row.ci_high_pp)
        ax_d.barh(yi, mean, height=0.34, color=_rgba(color, 0.20),
                  edgecolor=color, lw=LW_EDGE, zorder=1)
        ax_d.errorbar(
            mean, yi, xerr=[[mean - low], [high - mean]], fmt="D",
            ms=4.2, mfc="white", mec=color, mew=LW_ERR, color=color,
            lw=LW_ERR, capsize=ERR_CAPSIZE, zorder=3,
        )
        ax_d.text(-4.4, yi + 0.26, label, ha="left", va="bottom",
                  fontsize=PT_SMALL, color=TEXT_COLOR.get(key, color))
    ax_d.set_yticks([])
    ax_d.set_ylim(-0.55, len(CONTRAST_SPECS) - 1 + 0.92)
    ax_d.tick_params(axis="y", length=0, pad=2.0, labelsize=PT_SMALL)
    ax_d.tick_params(axis="x", labelsize=PT_TICK)
    ax_d.set_xlabel("paired accuracy difference (pp)", fontsize=PT_LABEL,
                    color=COLORS["ink"])
    ax_d.set_xlim(-5.0, 36.0)
    ax_d.set_xticks([0, 10, 20, 30])
    _title(ax_d, "Seed-paired H4 contrasts")

    # ── E: the same accuracy scale across task hierarchy and stage count.
    _heatmap(ax_e, _hierarchy_depth_matrix(h4_seed), DEPTH_LABELS,
             row_labels=("H2", "H3", "H4"), cmap=SEQ_CMAP,
             norm=ACCURACY_NORM, fmt="{:.2f}", best_by_row=True)
    ax_e.set_xlabel(DEPTH_AXIS, fontsize=PT_LABEL, color=COLORS["ink"])
    _title(ax_e, "Depth × hierarchy")

    # ── F, G: serial-minus-grouped-point advantage; shared labels and key.
    image_eff = _heatmap(ax_f, _task_matrix(task_effects, "bp"), ALPHA_LABELS,
                         row_labels=FAMILY_LABELS, cmap=DIV_CMAP,
                         norm=EFFECT_NORM, fmt="{:+.1f}")
    _heatmap(ax_g, _task_matrix(task_effects, "local3f"), ALPHA_LABELS,
             row_labels=None, cmap=DIV_CMAP, norm=EFFECT_NORM, fmt="{:+.1f}")
    _title(ax_f, "Serial benefit, BP", color=COLORS["bp"])
    _title(ax_g, "Serial benefit, LocalCA", color=AMBER_TEXT)
    for ax in (ax_f, ax_g):
        ax.set_xlabel(ALPHA_AXIS, fontsize=PT_LABEL, color=COLORS["ink"])

    # ── H: the alpha = 0 -> 1 change in that advantage, by credit rule.
    # Colour now carries the credit rule (the thing that differs between F
    # and G); the task family is already carried by position.
    interaction = task_contrasts[
        task_contrasts.estimand.eq("alignment_interaction")
    ]
    families = ("nested_factor", "flat_factor", "local_ratio")
    positions = np.arange(3)
    width = 0.34
    for credit, key, offset in (("bp", "bp", -width / 2),
                                ("local3f", "local", width / 2)):
        color = COLORS[key]
        for index, family in enumerate(families):
            row = interaction[
                interaction.family.eq(family) & interaction.credit.eq(credit)
            ].iloc[0]
            mean = 100 * float(row.mean_difference)
            low = 100 * float(row.ci95_low)
            high = 100 * float(row.ci95_high)
            xpos = positions[index] + offset
            ax_h.bar(xpos, mean, width=width * 0.88,
                     color=_rgba(color, 0.24), edgecolor=color, lw=LW_EDGE,
                     zorder=2)
            ax_h.errorbar(xpos, mean, yerr=[[mean - low], [high - mean]],
                          fmt="none", ecolor=color, elinewidth=LW_ERR,
                          capsize=ERR_CAPSIZE, zorder=3)
    ax_h.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--", zorder=1)
    ax_h.set_xticks(positions, list(FAMILY_SHORT))
    ax_h.set_xlim(-0.62, 2.62)
    ax_h.set_ylim(-19.0, 34.0)
    ax_h.set_yticks([-10, 0, 10, 20, 30])
    ax_h.tick_params(axis="x", length=0, pad=2.0, labelsize=PT_SMALL)
    ax_h.tick_params(axis="y", labelsize=PT_TICK)
    ax_h.set_ylabel("interaction (pp)", fontsize=PT_LABEL,
                    color=COLORS["ink"], labelpad=1.5)
    _title(ax_h, "Alignment interaction")
    # Direct labels in clear whitespace over the local-ratio group; no box.
    ax_h.text(2.46, 31.0, "BP", ha="right", va="center", fontsize=PT_SMALL,
              color=COLORS["bp"])
    ax_h.text(2.46, 24.0, "LocalCA", ha="right", va="center",
              fontsize=PT_SMALL, color=AMBER_TEXT)

    # ── colour keys: the accuracy scale in the reserved right rail beside
    # the headline, the effect scale between the two panels that share it.
    _key_rail(canvas, image_acc, 0, 1, "mean test accuracy", ACCURACY_TICKS,
              ACCURACY_TICK_LABELS)
    _key_rail_left(canvas, image_eff, ax_g, "serial − point (pp)",
                   EFFECT_TICKS, EFFECT_TICK_LABELS)

    problems = canvas.save(OUTPUT, name="main_figure_06_native")
    return problems


def main() -> None:
    problems = build()
    if problems:
        print(f"  {len(problems)} layout/overlap problems")
    else:
        print("  clean: no layout or text-over-data problems")


if __name__ == "__main__":
    main()

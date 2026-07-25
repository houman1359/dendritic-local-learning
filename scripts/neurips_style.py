"""
Unified NeurIPS figure style for local_credit_assignment paper.

Usage:
    from neurips_style import apply_neurips_style, COLORS, panel_label
    apply_neurips_style()
"""

from __future__ import annotations

import matplotlib as mpl

# ── Color palette (colorblind-safe, publication-oriented) ─────────────────
COLORS = {
    # architecture / anatomy
    "shunting":  "#1F7A4C",   # forest green
    "additive":  "#2657A2",   # steel blue
    "point_mlp": "#8A8A8A",   # neutral gray
    "exc":       "#2C6CB0",   # excitatory-synapse blue
    "inh":       "#B13138",   # inhibitory-synapse red
    "dend":      "#3B9668",   # dendrite green
    "soma":      "#EE8A3B",   # soma orange
    # optimization strategies
    "bp":        "#B0402F",   # muted red-brown
    "local":     "#C47A24",   # amber orange
    "oracle":    "#6D597A",   # muted purple
    # rules
    "rule_3f":   "#5BB39A",   # teal
    "rule_4f":   "#E88B69",   # salmon
    "rule_5f":   "#7A8CC4",   # muted lavender
    # broadcast modes
    "scalar":    "#B13138",   # scalar broadcast
    "per_soma":  "#E88B69",   # per-soma
    "low_rank":  "#D08C2F",   # low-rank warm gold
    "pathway":   "#7C5AA6",   # pathway violet
    # neutral / UI
    "grid":      "#D7DCE2",   # light grid
    "panel_bg":  "#F7F8FA",   # subtle panel background
    "ink":       "#1C1C1C",   # dark text
    "mute":      "#6B7280",   # mid-gray secondary text
    "edge":      "#4A4A4A",   # neutral outline
    "highlight": "#C15A8A",   # callout rose
}

# ── Figure sizing (NeurIPS text width ≈ 5.5 in; full width ≈ 7 in) ──────
SINGLE_COL = 5.5   # inches
DOUBLE_COL = 7.0   # inches
# Wider headline figures (will occupy full textwidth 0.98x in LaTeX)
WIDE_FIG   = 13.0

# Canonical authored width for every MAIN figure.
#
# Every main figure is included with \includegraphics[width=\textwidth]; NeurIPS
# \textwidth is 5.5 in = 397 pt.  A figure authored at MAIN_W is therefore
# rescaled by 397/(72*MAIN_W).  Authoring the main figures at *different* widths
# (previously 5.5 / 6.95 / 7.0 / 7.35) made the same nominal font render at a
# different printed size in every figure.  Author every main figure at MAIN_W so
# they all take the identical scale factor and the type is consistent.
MAIN_W = 7.2                        # inches
MAIN_SCALE = 397.0 / (72.0 * MAIN_W)  # ≈ 0.766 → printed pt = nominal * MAIN_SCALE

# Canonical authored width for EVERY figure in the paper, main and supplementary.
#
# Printed type size = nominal_pt * (latex_width_pt) / (72 * authored_width_in).
# Authoring supplementary figures at 6.3 / 10.45 in while including them at
# 0.62-0.98\textwidth made the same nominal 10.8 pt font print anywhere between
# 5.6 pt and 12.4 pt across the figure set.  Author every figure at FIG_W and
# include every figure at \textwidth: one scale factor, one printed type size.
FIG_W = MAIN_W

# Canonical panel-box geometry, in inches.  Panels are laid out on a fixed grid
# whose margins are specified in inches (not figure fractions), so a panel box
# in a 5-panel figure is exactly as tall as a panel box in a 2-panel figure and
# axis line weights, tick lengths and fonts all render at one scale.
PANEL_H = 1.62          # height of one panel box
PANEL_GAP_W = 0.72      # horizontal gap between panel boxes
PANEL_GAP_H = 0.58      # vertical gap between panel rows
MARGIN_L = 0.62         # left margin (room for y label + ticks)
MARGIN_R = 0.30         # room for right-edge value labels
MARGIN_T = 0.34         # room for panel title
MARGIN_B = 0.52         # room for x label + ticks

# Uniform element sizing, referenced by every generator.
BAR_LW = 0.9            # bar / patch edge width
ERR_LW = 1.15           # error-bar line width
ERR_CAPSIZE = 2.6
REF_LW = 1.25           # reference / threshold line width
SEED_MS = 3.4           # per-seed scatter marker size
SEED_ALPHA = 0.85
PANEL_LABEL_PT = 12.0   # panel letter size (single value, every figure)
PANEL_TITLE_PT = 10.4   # panel header size
PANEL_TITLE_PAD = 5.0   # header -> axes gap, points


def panel_title(ax, letter, title="", *, loc="left", pad=None, fontsize=None):
    """Uniform panel header: a bold letter followed by the panel title.

    Every panel in every figure uses this one call, so the letter has the same
    size, weight, alignment and baseline everywhere.  Left-aligning the letter
    with the title (rather than floating a letter outside the axes and centring
    the title) is what keeps narrow multi-panel rows free of collisions: the
    header grows to the right into free space instead of upward into the row
    above.
    """
    text = f"{letter}  {title}".rstrip() if title else str(letter)
    ax.set_title(
        text, loc=loc, fontweight="bold",
        fontsize=PANEL_TITLE_PT if fontsize is None else fontsize,
        pad=PANEL_TITLE_PAD if pad is None else pad,
    )


def printed_pt(nominal_pt: float) -> float:
    """Printed size (pt) of a FIG_W-authored element after LaTeX rescaling."""
    return float(nominal_pt) * MAIN_SCALE


def grid_figure(ncols, nrows=1, *, panel_h=None, width=None,
                gap_w=None, gap_h=None, margin_l=None, margin_r=None,
                margin_t=None, margin_b=None, width_ratios=None,
                height_ratios=None, squeeze=True):
    """Create a figure whose panel boxes have identical geometry everywhere.

    Margins and gaps are given in *inches* and converted to figure fractions,
    so the drawable panel box is the same physical size in every figure that
    uses the same ``panel_h``.  This is what makes axis weights, tick lengths
    and type render consistently across the whole figure set.

    Returns ``(fig, axes)`` with ``axes`` shaped like ``plt.subplots``.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    panel_h = PANEL_H if panel_h is None else float(panel_h)
    width = FIG_W if width is None else float(width)
    gap_w = PANEL_GAP_W if gap_w is None else float(gap_w)
    gap_h = PANEL_GAP_H if gap_h is None else float(gap_h)
    ml = MARGIN_L if margin_l is None else float(margin_l)
    mr = MARGIN_R if margin_r is None else float(margin_r)
    mt = MARGIN_T if margin_t is None else float(margin_t)
    mb = MARGIN_B if margin_b is None else float(margin_b)

    # Total figure height from the panel geometry.
    height = mt + mb + nrows * panel_h + (nrows - 1) * gap_h
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(
        nrows, ncols,
        left=ml / width, right=1.0 - mr / width,
        top=1.0 - mt / height, bottom=mb / height,
        wspace=gap_w / ((width - ml - mr) / ncols),
        hspace=gap_h / panel_h,
        width_ratios=width_ratios, height_ratios=height_ratios,
    )
    axes = np.empty((nrows, ncols), dtype=object)
    for r in range(nrows):
        for c in range(ncols):
            axes[r, c] = fig.add_subplot(gs[r, c])
    if squeeze:
        if nrows == 1 and ncols == 1:
            return fig, axes[0, 0]
        if nrows == 1:
            return fig, axes[0]
        if ncols == 1:
            return fig, axes[:, 0]
    return fig, axes


def label_panels(axes, labels=None, **kwargs):
    """Attach uniform panel letters to a flat or nested sequence of axes."""
    import numpy as np

    flat = list(np.asarray(axes, dtype=object).ravel())
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(flat))]
    for ax, lab in zip(flat, labels):
        if ax is None or not lab:
            continue
        panel_label(ax, lab, **kwargs)


def clean_legend(ax, *, loc="best", ncol=1, **kwargs):
    """Standard legend: no frame, tight spacing, consistent type size."""
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("handlelength", 1.4)
    kwargs.setdefault("handletextpad", 0.4)
    kwargs.setdefault("labelspacing", 0.3)
    kwargs.setdefault("columnspacing", 0.8)
    kwargs.setdefault("borderaxespad", 0.25)
    leg = ax.legend(loc=loc, ncol=ncol, **kwargs)
    return leg


def axis_break_note(ax, text="axis truncated", *, loc="lower right"):
    """Small italic note marking a truncated axis, placed inside the panel."""
    xy = {"lower right": (0.98, 0.02, "right", "bottom"),
          "lower left": (0.02, 0.02, "left", "bottom"),
          "upper right": (0.98, 0.98, "right", "top")}[loc]
    ax.text(xy[0], xy[1], text, transform=ax.transAxes,
            fontsize=7.4, style="italic", color=COLORS["mute"],
            ha=xy[2], va=xy[3])


def paired_lines(ax, x0, x1, y0, y1, *, color=None, lw=0.55, alpha=0.42,
                 zorder=1):
    """Draw per-seed pairing lines between two conditions."""
    color = COLORS["mute"] if color is None else color
    for a, b in zip(y0, y1):
        ax.plot([x0, x1], [a, b], color=color, lw=lw, alpha=alpha,
                zorder=zorder, solid_capstyle="round")


def apply_neurips_style():
    """Set matplotlib rcParams for a consistent, crisp NeurIPS look."""
    mpl.rcParams.update({
        # Font — sans-serif for crisp figure text at small sizes
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica", "Arial", "Liberation Sans",
            "DejaVu Sans", "Bitstream Vera Sans",
        ],
        "mathtext.fontset": "dejavusans",
        "font.size": 10.8,
        "axes.labelsize": 11.3,
        "axes.titlesize": 11.6,
        "axes.titleweight": "bold",
        "xtick.labelsize": 9.8,
        "ytick.labelsize": 9.8,
        "legend.fontsize": 9.3,
        "legend.title_fontsize": 9.5,

        # Lines / markers
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",

        # Axes
        "axes.linewidth": 1.05,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.labelpad": 2.5,
        # Room for the title to clear the panel letter.  With titlepad=4 the
        # title sat in the same band as a panel letter placed just above the
        # axes, which collided whenever a title wrapped to two lines.
        "axes.titlepad": 7,
        "axes.edgecolor": "#4A4A4A",
        "axes.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(
            "color",
            [COLORS["shunting"], COLORS["additive"], COLORS["bp"],
             COLORS["local"], COLORS["oracle"], COLORS["rule_3f"],
             COLORS["rule_4f"], COLORS["rule_5f"]],
        ),

        # Ticks
        "xtick.major.width": 0.95,
        "ytick.major.width": 0.95,
        "xtick.major.size": 3.9,
        "ytick.major.size": 3.9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.color": "#2A2A2A",
        "ytick.color": "#2A2A2A",

        # Grid
        "grid.linewidth": 0.78,
        "grid.alpha": 0.34,
        "grid.color": COLORS["grid"],

        # Legend
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.35,
        "legend.columnspacing": 0.9,

        # Figure
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": 350,
        # Fixed canvas: identical printed type size in every figure.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "savefig.facecolor": "white",
        "savefig.transparent": False,

        # PDF embedding (Type 42 = editable text in PDF)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,

        # Hatch
        "hatch.linewidth": 0.72,
    })


def panel_label(ax, label, x=None, y=None, *, dx=-26.0, dy=4.0, **kwargs):
    """Add a bold panel label (A, B, C, …) at the axes' upper-left.

    The label is anchored to the axes' top-left corner and offset in *points*
    (``dx``/``dy``), not axes fractions.  This matters: an axes-fraction offset
    scales with panel size, so the same nominal offset drifted into the title on
    narrow panels and floated away on wide ones.  A points offset is identical
    on every panel regardless of its width.

    The label sits to the LEFT of the axes (clearing the tick labels) and only
    slightly above it, so a centred title — which grows *upward* when it wraps —
    never occupies the same space.  Legacy ``x``/``y`` (axes-fraction) args are
    still honoured if a call site passes them explicitly.
    """
    fontsize = kwargs.pop("fontsize", PANEL_LABEL_PT)
    color = kwargs.pop("color", COLORS["ink"])
    if x is not None or y is not None:  # legacy axes-fraction placement
        ax.text(
            -0.11 if x is None else x, 1.05 if y is None else y, label,
            transform=ax.transAxes, fontsize=fontsize, fontweight="bold",
            va="top", ha="left", color=color, **kwargs,
        )
        return
    ax.annotate(
        label,
        xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(dx, dy), textcoords="offset points",
        fontsize=fontsize, fontweight="bold",
        va="bottom", ha="left", color=color,
        annotation_clip=False,
        **kwargs,
    )


def style_axis(ax, grid="none", spine_color=None):
    """Apply standard panel polish to an axes."""
    if grid in {"x", "y", "both"}:
        ax.grid(True, axis=grid, zorder=0, linewidth=0.78, alpha=0.34,
                color=COLORS["grid"])
    else:
        ax.grid(False)
    ax.tick_params(direction="out", length=3.9, width=0.95)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(1.05)
    if spine_color is not None:
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(spine_color)


def despine(ax, top=True, right=True, left=False, bottom=False):
    """Hide specific spines on an axes."""
    if top:
        ax.spines["top"].set_visible(False)
    if right:
        ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


def clean_schematic_axis(ax):
    """Configure an axes for pure schematic drawing (no spines, no ticks)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect("equal")


def add_panel_background(ax, color=None, alpha=0.0, radius=0.02):
    """Reserved hook for rounded panel backgrounds."""
    del ax, color, alpha, radius
    return None

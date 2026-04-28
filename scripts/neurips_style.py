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
        "font.size": 11.2,
        "axes.labelsize": 11.7,
        "axes.titlesize": 11.8,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.3,
        "legend.title_fontsize": 9.5,

        # Lines / markers
        "lines.linewidth": 2.0,
        "lines.markersize": 5.2,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",

        # Axes
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.labelpad": 2.5,
        "axes.titlepad": 5,
        "axes.edgecolor": "#4A4A4A",
        "axes.facecolor": "white",
        "axes.prop_cycle": mpl.cycler(
            "color",
            [COLORS["shunting"], COLORS["additive"], COLORS["bp"],
             COLORS["local"], COLORS["oracle"], COLORS["rule_3f"],
             COLORS["rule_4f"], COLORS["rule_5f"]],
        ),

        # Ticks
        "xtick.major.width": 0.75,
        "ytick.major.width": 0.75,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,
        "xtick.color": "#2A2A2A",
        "ytick.color": "#2A2A2A",

        # Grid
        "grid.linewidth": 0.65,
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
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "savefig.facecolor": "white",
        "savefig.transparent": False,

        # PDF embedding (Type 42 = editable text in PDF)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "pdf.compression": 9,

        # Hatch
        "hatch.linewidth": 0.55,
    })


def panel_label(ax, label, x=-0.11, y=1.05, **kwargs):
    """Add a bold panel label (A, B, C, …) to an axes.

    Default position is outside upper-left, suitable for most axes.
    """
    fontsize = kwargs.pop("fontsize", 14.0)
    color = kwargs.pop("color", COLORS["ink"])
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize, fontweight="bold",
        va="top", ha="left", color=color,
        **kwargs,
    )


def style_axis(ax, grid="none", spine_color=None):
    """Apply standard panel polish to an axes."""
    if grid in {"x", "y", "both"}:
        ax.grid(True, axis=grid, zorder=0, linewidth=0.65, alpha=0.34,
                color=COLORS["grid"])
    else:
        ax.grid(False)
    ax.tick_params(direction="out", length=3.5, width=0.75)
    ax.set_axisbelow(True)
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

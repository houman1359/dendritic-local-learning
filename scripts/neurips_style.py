"""
Unified NeurIPS figure style for local_credit_assignment paper.

Usage:
    from neurips_style import apply_neurips_style, COLORS
    apply_neurips_style()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Color palette (muted, colorblind-safe, publication-oriented) ───────────
COLORS = {
    "shunting": "#1C7C54",       # deep green
    "additive": "#2C5A88",       # steel blue
    "point_mlp": "#8A8A8A",      # neutral gray
    "bp": "#B04A3C",             # muted red-brown
    "local": "#C47A24",          # amber orange
    "oracle": "#6D597A",         # muted purple
    "highlight": "#C15A8A",      # rose
    "neutral": "#7A8352",        # olive gray
    "low_rank": "#D08C2F",       # warm orange-gold
    "pathway": "#7C5AA6",        # pathway-structured violet
    "grid": "#D7DCE2",           # light cool gray
    "ink": "#222222",            # dark text
}

# ── Line / marker defaults ───────────────────────────────────────────────────
LINEWIDTH = 1.45
MARKERSIZE = 4.2

# ── Figure sizing (NeurIPS column = 5.5 in, full width ≈ text width) ────────
SINGLE_COL = 5.5   # inches
DOUBLE_COL = 7.0   # inches (roughly 0.95 * textwidth at NeurIPS)


def apply_neurips_style():
    """Set matplotlib rcParams for a consistent NeurIPS look."""
    mpl.rcParams.update({
        # Font
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7,
        "legend.title_fontsize": 7.5,

        # Lines / markers
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,

        # Axes
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.labelpad": 3,
        "axes.titlepad": 6,
        "axes.edgecolor": "#666666",
        "axes.facecolor": "white",

        # Ticks
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.pad": 2,
        "ytick.major.pad": 2,

        # Grid (off by default, enable per-panel if needed)
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "grid.color": COLORS["grid"],

        # Legend
        "legend.frameon": True,
        "legend.framealpha": 0.96,
        "legend.facecolor": "white",
        "legend.edgecolor": "#D0D5DD",
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3,

        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "savefig.facecolor": "white",
        "savefig.transparent": False,

        # PDF embedding (Type 42 = editable text in PDF)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, label, x=-0.12, y=1.08, **kwargs):
    """Add a bold panel label (A, B, C, …) to an axes."""
    fontsize = kwargs.pop("fontsize", 11)
    color = kwargs.pop("color", COLORS["ink"])
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
        color=color,
        **kwargs,
    )


def style_axis(ax, grid="none"):
    """Apply common panel polish to an axes."""
    if grid in {"x", "y", "both"}:
        ax.grid(True, axis=grid, zorder=0)
    else:
        ax.grid(False)
    ax.tick_params(direction="out", length=3, width=0.5)
    ax.set_axisbelow(True)

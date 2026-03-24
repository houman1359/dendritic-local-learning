"""
Unified NeurIPS figure style for local_credit_assignment paper.

Usage:
    from neurips_style import apply_neurips_style, COLORS
    apply_neurips_style()
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Color palette (colorblind-safe) ──────────────────────────────────────────
COLORS = {
    "shunting": "#2ca02c",       # green
    "additive": "#1f77b4",       # blue
    "point_mlp": "#7f7f7f",      # gray
    "bp": "#d62728",             # red
    "local": "#ff7f0e",          # orange
    "oracle": "#9467bd",         # purple
    "highlight": "#e377c2",      # pink
    "neutral": "#bcbd22",        # olive
}

# ── Line / marker defaults ───────────────────────────────────────────────────
LINEWIDTH = 1.4
MARKERSIZE = 4

# ── Figure sizing (NeurIPS column = 5.5 in, full width ≈ text width) ────────
SINGLE_COL = 5.5   # inches
DOUBLE_COL = 7.0   # inches (roughly 0.95 * textwidth at NeurIPS)


def apply_neurips_style():
    """Set matplotlib rcParams for a consistent NeurIPS look."""
    mpl.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
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
        "axes.titlepad": 5,

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
        "grid.alpha": 0.3,

        # Legend
        "legend.frameon": False,
        "legend.borderpad": 0.3,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.labelspacing": 0.3,

        # Figure
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        # PDF embedding (Type 42 = editable text in PDF)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def panel_label(ax, label, x=-0.12, y=1.08, **kwargs):
    """Add a bold panel label (A, B, C, …) to an axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left", **kwargs)

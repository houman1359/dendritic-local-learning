"""Journal-specific extensions to the frozen NeurIPS visual system.

The canonical canvas, panel geometry and audit helpers are inherited verbatim
from :mod:`neurips_style` (hash-frozen by the lineage audit).  This layer owns
the *journal look*: a CVD-validated palette, a lighter type scale with
regular-weight sentence-case panel titles, thinner marks, one sequential and
one diverging colormap for every heatmap, and letter placement in a fixed
gutter so long y labels never push letters into a neighbouring panel.

Palette provenance: every pair of series colours that shares a panel was
checked in OKLab under Machado severity-1.0 protan/deutan/tritan simulation
(worst-pair ΔE·100 ≥ 8, normal-vision ΔE·100 ≥ 15).  Do not swap hues here
without re-running that check.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, to_rgb

import neurips_style as _base
from neurips_style import *  # noqa: F401,F403 - deliberate style re-export


# ── Journal palette (CVD-validated; keeps the NeurIPS hue semantics) ──────
_JOURNAL_COLORS = {
    "shunting":  "#3FA26C",   # primary green — lightened for tritan/deutan sep
    "additive":  "#20509E",   # primary blue — deepened against the green
    "bp":        "#932F1E",   # backprop / exact red-brown, darkened
    "local":     "#E2A23F",   # amber, lightened away from the green
    "oracle":    "#8F66CD",   # violet, raised chroma + lightness vs gray/blue
    "highlight": "#E28FBC",   # rose, lightened away from mid-gray
    "point_mlp": "#686868",   # neutral gray control
    "scalar":    "#E2A23F",   # scalar broadcast shares the amber slot; never
                              # plot scalar and 'local' as sibling series
    "per_soma":  "#D98A75",
    "low_rank":  "#C7862B",
    "pathway":   "#8F66CD",
    # anatomy (schematics)
    "exc":       "#2C6CB0",
    "inh":       "#B13138",
    "dend":      "#3E8E63",
    "soma":      "#E8873C",
    # neutral / UI
    "grid":      "#E3E7EC",
    "panel_bg":  "#F7F8FA",
    "ink":       "#232323",
    "mute":      "#69707A",
    "edge":      "#55595E",
}
COLORS.update(_JOURNAL_COLORS)  # in-place: frozen NeurIPS components see it too

# Fixed marker order for multi-series panels: shape is the secondary encoding
# that keeps 6-8 ΔE pairs legal and survives grayscale printing.
MARKERS = ("o", "s", "^", "D", "v", "P", "X")

# ── Journal type scale (authored 7.2 in, printed at ~0.96 in this draft) ──
PT_TITLE = 8.8          # panel title (regular weight, sentence case)
PT_LABEL = 8.4          # axis label
PT_TICK = 7.6           # tick label
PT_LEGEND = 7.4         # legend entry
PT_ANNOT = 7.2          # in-panel callout
PT_SMALL = 6.8          # dense schematic text (floor)
PANEL_LABEL_PT = 10.5   # panel letter (bold)
PANEL_TITLE_PT = PT_TITLE

# ── Journal line weights: thinner marks, recessive references ─────────────
LW_DATA = 1.25
LW_REF = 0.85
LW_ERR = 0.95
LW_EDGE = 0.7
LW_HAIR = 0.55
BAR_LW = LW_EDGE
ERR_LW = LW_ERR
REF_LW = LW_REF
ERR_CAPSIZE = 2.0
SEED_MS = 2.9
SEED_ALPHA = 0.60
MARKER_MS = 4.6

PANEL_LETTER_X = -30.0

# ── One sequential + one diverging colormap for every heatmap ─────────────
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "journal_seq",
    ["#F4F7FB", "#CBDAEE", "#93B3DB", "#5580BE", "#20509E", "#122E63"],
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "journal_div",
    ["#122E63", "#3F6BB0", "#9FB9DC", "#F1EFEA", "#DA9A82", "#B2492F", "#6E2113"],
)


def snap_pt(value: float) -> float:
    """Snap an arbitrary font size to the journal type scale."""
    scale = (PT_SMALL, PT_ANNOT, PT_LEGEND, PT_TICK, PT_LABEL, PT_TITLE)
    v = float(value)
    if v >= PT_TITLE:
        return PT_TITLE
    return min(scale, key=lambda s: abs(s - v))


def snap_lw(value: float) -> float:
    """Snap an arbitrary line width to the journal weights."""
    scale = (LW_HAIR, LW_EDGE, LW_ERR, LW_REF, LW_DATA)
    v = float(value)
    if v <= 0:
        return v
    if v >= LW_DATA:
        return LW_DATA
    return min(scale, key=lambda s: abs(s - v))


def _journal_finalize_panel_letters(fig) -> None:
    """Align panel letters in one gutter and lift them above panel titles."""
    try:
        renderer = fig.canvas.get_renderer()
    except Exception:
        return
    dpi = fig.dpi
    for ax in fig.axes:
        artist = getattr(ax, "_neurips_panel_letter", None)
        if artist is None or not artist.get_visible():
            continue
        try:
            axes_box = ax.get_window_extent(renderer=renderer)
        except Exception:
            continue
        top = axes_box.y1
        for title in _base._title_artists(ax):
            try:
                top = max(top, title.get_window_extent(renderer=renderer).y1)
            except Exception:
                pass
        artist.xyann = (
            PANEL_LETTER_X,
            ((top - axes_box.y1) / dpi) * 72.0 + _base.PANEL_LETTER_RISE,
        )


def panel_title(ax, letter, title="", *, loc="left", pad=None, fontsize=None):
    """Journal panel header: bold letter, regular-weight sentence-case title."""
    artist = _base.panel_title(
        ax,
        letter,
        title,
        loc=loc,
        pad=pad,
        fontsize=PANEL_TITLE_PT if fontsize is None else fontsize,
    )
    for t in _base._title_artists(ax):
        t.set_fontweight("normal")
        t.set_color(COLORS["ink"])
    if artist is not None:
        artist.set_fontsize(PANEL_LABEL_PT)
        artist.xyann = (PANEL_LETTER_X, 8.0)
    return artist


def panel_label(ax, label, x=None, y=None, **kwargs):
    """Journal panel letter at the journal size (call sites without a title)."""
    kwargs.setdefault("fontsize", PANEL_LABEL_PT)
    return _base.panel_label(ax, label, x=x, y=y, **kwargs)


def add_colorbar(fig, ax, mappable, *, label="", width=0.035, pad=0.02):
    """Slim colorbar hugging the panel's right edge, journal type sizes."""
    box = ax.get_position()
    cax = fig.add_axes(
        [box.x1 + pad * box.width, box.y0, width * box.width, box.height]
    )
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=PT_ANNOT, width=0.7, length=2.4, pad=1.5)
    if label:
        cbar.set_label(label, fontsize=PT_ANNOT, labelpad=2.5)
    return cbar


def annotate_heatmap(ax, im, data, *, fmt="{:.2f}", fontsize=None):
    """Cell-value annotations with luminance-aware ink/white text."""
    import numpy as np

    fontsize = PT_ANNOT if fontsize is None else fontsize
    arr = np.asarray(data, dtype=float)
    norm = im.norm
    cmap = im.get_cmap()
    for (r, c), val in np.ndenumerate(arr):
        if not np.isfinite(val):
            continue
        rgba = cmap(norm(val))
        lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
        ax.text(
            c, r, fmt.format(val),
            ha="center", va="center", fontsize=fontsize,
            color="white" if lum < 0.45 else COLORS["ink"],
        )


def wrap_ticklabels(labels, width=10):
    """Break category names onto two lines so they can stay horizontal."""
    import textwrap

    out = []
    for lab in labels:
        lab = str(lab)
        if len(lab) <= width or " " not in lab:
            out.append(lab)
        else:
            out.append("\n".join(textwrap.wrap(lab, width=width, max_lines=2,
                                               placeholder="…")))
    return out


def apply_neurips_style() -> None:
    """Frozen base style, then the journal overrides, then the letter hook."""
    _base.apply_neurips_style()
    _base._finalize_panel_letters = _journal_finalize_panel_letters
    mpl.rcParams.update({
        "font.size": PT_TICK,
        "axes.labelsize": PT_LABEL,
        "axes.titlesize": PT_TITLE,
        "axes.titleweight": "normal",
        "xtick.labelsize": PT_TICK,
        "ytick.labelsize": PT_TICK,
        "legend.fontsize": PT_LEGEND,
        "legend.title_fontsize": PT_LEGEND,
        "lines.linewidth": LW_DATA,
        "lines.markersize": MARKER_MS,
        "axes.linewidth": 0.8,
        "axes.edgecolor": COLORS["edge"],
        "axes.labelcolor": COLORS["ink"],
        "axes.labelpad": 2.2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.color": COLORS["edge"],
        "ytick.color": COLORS["edge"],
        "xtick.labelcolor": COLORS["ink"],
        "ytick.labelcolor": COLORS["ink"],
        "grid.linewidth": 0.6,
        "grid.alpha": 0.55,
        "grid.color": COLORS["grid"],
        "axes.prop_cycle": mpl.cycler(
            "color",
            [COLORS["shunting"], COLORS["additive"], COLORS["bp"],
             COLORS["local"], COLORS["oracle"], COLORS["highlight"],
             COLORS["point_mlp"]],
        ),
    })


def style_axis(ax, grid="none", spine_color=None):
    """Journal panel polish: thinner spines and ticks than the frozen base."""
    if grid in {"x", "y", "both"}:
        ax.grid(True, axis=grid, zorder=0, linewidth=0.6, alpha=0.55,
                color=COLORS["grid"])
    else:
        ax.grid(False)
    ax.tick_params(direction="out", length=3.0, width=0.8)
    ax.set_axisbelow(True)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color(
            COLORS["edge"] if spine_color is None else spine_color
        )

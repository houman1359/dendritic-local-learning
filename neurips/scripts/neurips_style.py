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
PANEL_GAP_W = 0.95      # horizontal gap between panel boxes
PANEL_GAP_H = 0.58      # vertical gap between panel rows
MARGIN_L = 0.62         # left margin (room for y label + ticks)
MARGIN_R = 0.46         # room for right-edge value labels
MARGIN_T = 0.50         # room for panel title + panel letter above it
MARGIN_B = 0.62         # room for x label + ticks

# ── Canonical type scale ──────────────────────────────────────────────────
#
# Every figure is authored at FIG_W and printed at MAIN_SCALE (~0.766), so one
# nominal size means one printed size everywhere.  Call sites must use these
# tokens rather than literals: the previous 30 distinct hardcoded `fontsize=`
# values were tuned when figures were authored at 5.0-14.8 in and are now just
# noise.  PT_SMALL is the floor -- 7.0 nominal prints at 5.4 pt, and anything
# below that is unreadable at NeurIPS column width.
PT_TITLE = 10.4         # panel header
PT_LABEL = 9.6          # axis label
PT_TICK = 8.6           # tick label
PT_LEGEND = 8.2         # legend entry
PT_ANNOT = 7.8          # in-panel value label / callout
PT_SMALL = 7.0          # dense schematic text (floor)

# ── Canonical line weights ────────────────────────────────────────────────
LW_DATA = 1.6           # primary data line
LW_REF = 1.25           # reference / threshold line
LW_ERR = 1.15           # error bar
LW_EDGE = 0.8           # bar / patch edge
LW_HAIR = 0.6           # pairing line, seed connector, schematic rule

# Legacy aliases kept so existing call sites keep working.
BAR_LW = LW_EDGE
ERR_LW = LW_ERR
ERR_CAPSIZE = 2.6
REF_LW = LW_REF
SEED_MS = 3.4           # per-seed scatter marker size
SEED_ALPHA = 0.85
# Panel letter is the dominant element of the header and is set larger than the
# panel title; the title is demoted a step so the letter reads first.
PANEL_LABEL_PT = 12.0   # panel letter
PANEL_TITLE_PT = 9.6    # panel title (deliberately smaller than the letter)
PANEL_TITLE_PAD = 5.0   # title -> axes gap, points
PANEL_LETTER_RISE = 3.0 # letter baseline above the title top, points


def snap_pt(value: float) -> float:
    """Snap an arbitrary font size to the nearest canonical token."""
    scale = (PT_SMALL, PT_ANNOT, PT_LEGEND, PT_TICK, PT_LABEL, PT_TITLE)
    v = float(value)
    if v >= PT_TITLE:
        return PT_TITLE
    return min(scale, key=lambda s: abs(s - v))


def snap_lw(value: float) -> float:
    """Snap an arbitrary line width to the nearest canonical weight."""
    scale = (LW_HAIR, LW_EDGE, LW_ERR, LW_REF, LW_DATA)
    v = float(value)
    if v <= 0:
        return v
    if v >= LW_DATA:
        return LW_DATA
    return min(scale, key=lambda s: abs(s - v))


def panel_title(ax, letter, title="", *, loc="left", pad=None, fontsize=None):
    """Uniform panel header: a large bold letter plus a smaller panel title.

    The letter is a separate artist, not a prefix on the title string, so it can
    be (a) set larger than the title and (b) aligned with the *y-axis label*
    rather than the axes spine.  Aligning to the spine left the letter floating
    inside the panel's own tick/label gutter and made the column of letters look
    ragged when panels had different y-label widths.  Final placement happens at
    save time in ``_finalize_panel_letters`` once the y label and tick labels
    have their real extents.
    """
    if title:
        ax.set_title(
            title, loc=loc, fontweight="bold",
            fontsize=PANEL_TITLE_PT if fontsize is None else fontsize,
            pad=PANEL_TITLE_PAD if pad is None else pad,
        )
    else:
        ax.set_title("")
    if not str(letter):
        return None
    art = ax.annotate(
        str(letter),
        xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(-30.0, 8.0), textcoords="offset points",
        fontsize=PANEL_LABEL_PT, fontweight="bold",
        va="bottom", ha="left", color=COLORS["ink"],
        annotation_clip=False, zorder=100,
    )
    ax._neurips_panel_letter = art
    return art


def _title_artists(ax):
    """All three title slots: with loc="left" matplotlib uses ax._left_title,
    not ax.title, so code that reads only ax.title silently sees an empty
    centre title."""
    out = []
    for name in ("title", "_left_title", "_right_title"):
        art = getattr(ax, name, None)
        if art is not None and art.get_text().strip():
            out.append(art)
    return out


def _finalize_panel_letters(fig):
    """Align each panel letter to its y-axis label and lift it above the title."""
    try:
        r = fig.canvas.get_renderer()
    except Exception:
        return
    dpi = fig.dpi
    for ax in fig.axes:
        art = getattr(ax, "_neurips_panel_letter", None)
        if art is None or not art.get_visible():
            continue
        try:
            ab = ax.get_window_extent(renderer=r)
        except Exception:
            continue
        left, top = ab.x0, ab.y1
        # extend left over the y label and the y tick labels
        ylab = ax.yaxis.label
        if ylab is not None and ylab.get_text().strip():
            try:
                left = min(left, ylab.get_window_extent(renderer=r).x0)
            except Exception:
                pass
        if getattr(ax, "axison", True):
            lo, hi = ax.get_ylim()
            for pos, lb in zip(ax.get_yticks(), ax.get_yticklabels()):
                if min(lo, hi) <= pos <= max(lo, hi) and lb.get_text().strip():
                    try:
                        left = min(left, lb.get_window_extent(renderer=r).x0)
                    except Exception:
                        pass
        # sit above the panel title (which lives in _left_title for loc="left")
        for t in _title_artists(ax):
            try:
                top = max(top, t.get_window_extent(renderer=r).y1)
            except Exception:
                pass
        art.xyann = (((left - ab.x0) / dpi) * 72.0,
                     ((top - ab.y1) / dpi) * 72.0 + PANEL_LETTER_RISE)


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


def add_headroom(ax, frac=0.24, *, bottom=False):
    """Expand the y range so an in-panel legend never sits on the data."""
    lo, hi = ax.get_ylim()
    if ax.get_yscale() == "log":
        return
    span = hi - lo
    if bottom:
        ax.set_ylim(lo - frac * span, hi)
    else:
        ax.set_ylim(lo, hi + frac * span)


def tidy_ticks(ax, *, nx=None, ny=None):
    """Cap tick counts so extreme labels stay inside the panel box."""
    from matplotlib.ticker import MaxNLocator
    if nx:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=nx, prune=None))
    if ny:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=ny, prune="both"))


def _legend_data_hits(ax, leg):
    """Number of plotted points falling inside a legend's box."""
    import numpy as np
    fig = ax.get_figure()
    try:
        r = fig.canvas.get_renderer()
    except Exception:
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
    try:
        bb = leg.get_window_extent(renderer=r)
    except Exception:
        return 0
    pts = []
    for ln in ax.lines:
        xy = ln.get_xydata()
        if xy is not None and len(xy):
            arr = np.asarray(xy, dtype=float)
            arr = arr[np.isfinite(arr).all(axis=1)]
            if len(arr):
                pts.append(ax.transData.transform(arr))
    for coll in ax.collections:
        try:
            off = np.asarray(coll.get_offsets(), dtype=float)
        except Exception:
            continue
        if off.ndim == 2 and len(off):
            off = off[np.isfinite(off).all(axis=1)]
            if len(off):
                pts.append(coll.get_offset_transform().transform(off))
    hits = 0
    if pts:
        d = np.vstack(pts)
        hits += int((((d[:, 0] > bb.x0) & (d[:, 0] < bb.x1)
                      & (d[:, 1] > bb.y0) & (d[:, 1] < bb.y1))).sum())
    # Bars are patches, not lines; a legend over a bar chart must count them
    # or the auto-placement silently believes it is clear.
    for patch in ax.patches:
        try:
            ob = patch.get_window_extent(renderer=r)
        except Exception:
            continue
        if (min(bb.x1, ob.x1) - max(bb.x0, ob.x0) > 2
                and min(bb.y1, ob.y1) - max(bb.y0, ob.y0) > 2):
            hits += 1
    return hits


def clean_legend(ax, *, loc="best", ncol=1, auto_clear=False, **kwargs):
    """Standard legend: no frame, tight spacing, consistent type size.

    ``auto_clear`` re-places the legend if the requested corner sits on the
    data.  It is opt-in: relocation optimises only for data overlap and can push
    a legend past the panel edge or into the header, so hand-tuned placements
    are kept unless a call site asks for automatic clearing.
    """
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("handlelength", 1.4)
    kwargs.setdefault("handletextpad", 0.4)
    kwargs.setdefault("labelspacing", 0.3)
    kwargs.setdefault("columnspacing", 0.8)
    kwargs.setdefault("borderaxespad", 0.25)
    leg = ax.legend(loc=loc, ncol=ncol, **kwargs)
    if not auto_clear or ax.get_yscale() == "log":
        return leg

    if _legend_data_hits(ax, leg) == 0:
        return leg
    best, best_hits = leg, _legend_data_hits(ax, leg)
    for cand in ("upper left", "upper right", "lower left", "lower right",
                 "center left", "center right"):
        if cand == loc:
            continue
        trial = ax.legend(loc=cand, ncol=ncol, **kwargs)
        try:
            r2 = ax.get_figure().canvas.get_renderer()
            lb = trial.get_window_extent(renderer=r2)
            ab = ax.get_window_extent(renderer=r2)
            if (lb.x0 < ab.x0 - 1 or lb.x1 > ab.x1 + 1
                    or lb.y0 < ab.y0 - 1 or lb.y1 > ab.y1 + 1):
                continue
        except Exception:
            pass
        h = _legend_data_hits(ax, trial)
        if h < best_hits:
            best, best_hits = trial, h
        if h == 0:
            return trial
    # Nothing is clear: make room at the top and re-place there.
    add_headroom(ax, 0.26)
    trial = ax.legend(loc="upper right", ncol=ncol, **kwargs)
    if _legend_data_hits(ax, trial) <= best_hits:
        return trial
    return best


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


_SAVEFIG_PATCHED = False


def _patch_savefig_once():
    """Make every savefig align panel letters just before rendering."""
    global _SAVEFIG_PATCHED
    if _SAVEFIG_PATCHED:
        return
    from matplotlib.figure import Figure
    original = Figure.savefig

    def savefig(self, *args, **kwargs):
        try:
            self.canvas.draw()
            _finalize_panel_letters(self)
        except Exception:
            pass
        return original(self, *args, **kwargs)

    Figure.savefig = savefig
    _SAVEFIG_PATCHED = True


def apply_neurips_style():
    """Set matplotlib rcParams for a consistent, crisp NeurIPS look."""
    _patch_savefig_once()
    mpl.rcParams.update({
        # Font — sans-serif for crisp figure text at small sizes
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Helvetica", "Arial", "Liberation Sans",
            "DejaVu Sans", "Bitstream Vera Sans",
        ],
        "mathtext.fontset": "dejavusans",
        # rcParams follow the same canonical scale as the call sites; leaving
        # them at the old 10.8/11.3/11.6 made axis labels render larger than
        # every explicitly sized element and pushed them into the next panel.
        "font.size": PT_TICK,
        "axes.labelsize": PT_LABEL,
        "axes.titlesize": PT_TITLE,
        "axes.titleweight": "bold",
        "xtick.labelsize": PT_TICK,
        "ytick.labelsize": PT_TICK,
        "legend.fontsize": PT_LEGEND,
        "legend.title_fontsize": PT_LEGEND,

        # Lines / markers
        "lines.linewidth": LW_DATA,
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


def audit_layout(fig, name=""):
    """Report text that leaves the canvas or intrudes into a neighbouring panel.

    Called from the figure generators just before saving.  Catches the two
    failure modes that a fixed canvas makes silent: an artist drawn outside the
    figure (clipped away) and an axis label or annotation whose bounding box
    lands inside a different panel's box.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches() * fig.dpi
    axes = [a for a in fig.axes if a.get_visible()]
    boxes = {a: a.get_window_extent(renderer=r) for a in axes}
    problems = []

    for ax in axes:
        own = boxes[ax]
        artists = _title_artists(ax) + [ax.xaxis.label, ax.yaxis.label]
        artists += list(ax.texts)
        # Axes with axis("off") still carry tick-label artists at stale
        # positions; they are never drawn, so auditing them is pure noise.
        if getattr(ax, "axison", True):
            # Locators emit label artists for ticks outside the view; those are
            # positioned far off-figure and never drawn, so auditing them
            # produces phantom "off-canvas" hits.  Keep only in-view ticks.
            def _in_view(locs, labs, lo, hi):
                span = abs(hi - lo)
                pad = 1e-9 if span == 0 else 1e-6 * span
                return [lb for pos, lb in zip(locs, labs)
                        if min(lo, hi) - pad <= pos <= max(lo, hi) + pad]
            artists += _in_view(ax.get_xticks(), ax.get_xticklabels(), *ax.get_xlim())
            artists += _in_view(ax.get_yticks(), ax.get_yticklabels(), *ax.get_ylim())
        leg = ax.get_legend()
        if leg is not None:
            artists.append(leg)
        for art in artists:
            if art is None or not art.get_visible():
                continue
            txt = getattr(art, "get_text", lambda: "")()
            if isinstance(txt, str) and not txt.strip():
                continue
            try:
                bb = art.get_window_extent(renderer=r)
            except Exception:
                continue
            if bb.width <= 0 or bb.height <= 0:
                continue
            tol = 5.0
            over = max(-bb.x0, -bb.y0, bb.x1 - fw, bb.y1 - fh)
            if over > tol:
                side = ("left" if -bb.x0 == over else
                        "bottom" if -bb.y0 == over else
                        "right" if bb.x1 - fw == over else "top")
                problems.append(
                    f"OFF-CANVAS {txt!r} {side} by {over/fig.dpi*72:.1f}pt")
                continue
            for other in axes:
                if other is ax:
                    continue
                ob = boxes[other]
                ix = max(0.0, min(bb.x1, ob.x1) - max(bb.x0, ob.x0))
                iy = max(0.0, min(bb.y1, ob.y1) - max(bb.y0, ob.y0))
                if ix > 2 and iy > 2:
                    problems.append(f"INTRUDES {txt!r} -> other panel")
                    break

    if problems:
        print(f"  [layout] {name}: " + "; ".join(sorted(set(problems))[:8]))
    return problems


def audit_text_over_data(fig, name=""):
    """Report in-panel text (annotations, legends) drawn on top of data.

    Complements audit_layout, which only sees text-vs-panel geometry.  Here we
    transform each line's vertices, each bar's rectangle and each scatter offset
    into display space and test them against the text bounding boxes in the same
    axes.  This is the check that catches a value callout sitting on the curve
    it labels, or a legend overlapping the series it describes.
    """
    import numpy as np

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []

    for ax in fig.axes:
        if not ax.get_visible() or not getattr(ax, "axison", True):
            continue

        labels = []
        for t in ax.texts:
            txt = t.get_text()
            if not (t.get_visible() and isinstance(txt, str) and txt.strip()):
                continue
            # A light-coloured label or one drawn on its own patch is an
            # intentional overlay (value inside a bar, boxed callout).
            try:
                import matplotlib.colors as mcolors
                rgb = mcolors.to_rgb(t.get_color())
                if sum(rgb) / 3.0 > 0.7:
                    continue
            except Exception:
                pass
            if t.get_bbox_patch() is not None:
                continue
            try:
                labels.append((txt, t.get_window_extent(renderer=r)))
            except Exception:
                pass
        leg = ax.get_legend()
        if leg is not None and leg.get_visible():
            try:
                labels.append(("<legend>", leg.get_window_extent(renderer=r)))
            except Exception:
                pass
        if not labels:
            continue

        pts = []
        for ln in ax.lines:
            if not ln.get_visible():
                continue
            xy = ln.get_xydata()
            if xy is None or len(xy) == 0:
                continue
            arr = np.asarray(xy, dtype=float)
            arr = arr[np.isfinite(arr).all(axis=1)]
            if len(arr):
                pts.append(ax.transData.transform(arr))
        for coll in ax.collections:
            try:
                off = coll.get_offsets()
            except Exception:
                continue
            arr = np.asarray(off, dtype=float)
            if arr.ndim == 2 and len(arr):
                arr = arr[np.isfinite(arr).all(axis=1)]
                if len(arr):
                    pts.append(coll.get_offset_transform().transform(arr))
        data_pts = np.vstack(pts) if pts else np.empty((0, 2))

        bars = []
        for patch in ax.patches:
            try:
                bars.append(patch.get_window_extent(renderer=r))
            except Exception:
                pass

        for txt, bb in labels:
            hit = False
            if len(data_pts):
                inside = ((data_pts[:, 0] > bb.x0) & (data_pts[:, 0] < bb.x1)
                          & (data_pts[:, 1] > bb.y0) & (data_pts[:, 1] < bb.y1))
                need = max(2, int(0.01 * len(data_pts))) if txt == "<legend>" else 1
                hit = int(inside.sum()) >= need
            if not hit:
                for ob in bars:
                    ix = min(bb.x1, ob.x1) - max(bb.x0, ob.x0)
                    iy = min(bb.y1, ob.y1) - max(bb.y0, ob.y0)
                    if ix > 2 and iy > 2:
                        hit = True
                        break
            if hit:
                problems.append(f"TEXT-ON-DATA {txt!r}")

    if problems:
        print(f"  [overlap] {name}: " + "; ".join(sorted(set(problems))[:10]))
    return problems

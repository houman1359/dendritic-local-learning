"""Main Figure 2 -- route dictionaries over morphology, natively.

    row 0   A one dictionary, five instantiations | B gallery on a [3,3] arbor
    row 1   C anatomical routes (MICrONS)         | D task -> dictionary map

The figure introduces the paper's method up front: it names the route
dictionary once (A), shows what the candidate dictionaries look like and how
much of a trained credit field each captures (B, with the additive and
shunting field profiles side by side), paints the same construction on a
reconstructed arbor with a metric scale bar (C), and previews which
dictionary each experimental task family demands (D, a roadmap schematic
mirroring Fig. 1E's experimental-logic role). Every "Fig. N" cross-reference
baked into A and D is derived from the assembly map's ``FIGURE_SOURCES``,
so a renumbering there cannot leave these labels stale.

The component keeps its historical file name (main_figure_10_native.pdf);
the assembly map in ``assemble_compact_main_figures.py`` emits it as
``figures/main/figure_02.pdf``. Quantitative marks read frozen source
tables: the gallery captures and the example field come from
``source_data/route_dictionary_atlas`` (computed on the paper's own trained
exact-path MNIST checkpoints), and the arbor geometry is the
``segment_metrics.csv`` cell drawn in the anatomy figure. Panels A and D
are labelled schematics and draw no data. The alignment-by-bandwidth plane
with the utility-argmax exhibit lives in the theory figure's panel H, not
here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from assemble_compact_main_figures import FIGURE_SOURCES
from credit_tree_schematics import draw_credit_tree, mix
from figure_canvas import (
    COLORS,
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
    Margins,
    NativeCanvas,
    audit_native_pdf,
)

ROOT = Path(__file__).resolve().parents[1]
ATLAS = ROOT / "source_data" / "route_dictionary_atlas"
PLANE = ROOT / "source_data" / "credit_phase_plane"
SEGMENTS = ROOT / "source_data" / "figure3" / "segment_metrics.csv"
OUT = ROOT / "figures" / "components" / "main_figure_10_native.pdf"

HEIGHT_IN = 336.0 / 72.0
HGUTTER = 38.0
VGUTTER = 58.0
MARGINS = Margins(left=51.0, right=13.5, top=23.0, bottom=21.0)
ROW_WEIGHTS = (150.0, 160.0)

C_ROUTE = COLORS["shunting"]      # route-of-interest series (Fig. 7 semantics)
C_CTRL = COLORS["point_mlp"]      # neutral control gray
C_ORACLE = COLORS["oracle"]
C_BP = COLORS["bp"]

# Manuscript figure number of each native component, inverted from the
# assembly map (manuscript number -> component number) so the "Fig. N"
# strings drawn in A and D follow any renumbering made there.
FIG_OF = {component: number for number, component in FIGURE_SOURCES.items()}
COMPONENT = {
    "framework": 1,    # main_figure_01_native: framework
    "ladder": 2,       # main_figure_02_native: MNIST feedback ladder
    "theory": 3,       # main_figure_03_native: credit operator theory
    "conflict": 4,     # main_figure_04_native: branch conflict
    "factorial": 5,    # main_figure_05_native: eight-context factorial
    "arbors": 7,       # main_figure_07_native: reconstructed-arbor capacity
    "boundary": 9,     # main_figure_09_native: measured boundary
}


def fig_ref(*roles):
    """``Fig. N`` / ``Figs N, M`` for the named components, as printed."""
    numbers = ", ".join(str(FIG_OF[COMPONENT[role]]) for role in roles)
    return f"Figs {numbers}" if len(roles) > 1 else f"Fig. {numbers}"

# Categorical shades for the four anatomical routes in C.  Subtree identity
# is categorical, but the palette reserves distinct hues for other series, so
# the four routes take an ordered ramp of the route green -- the caption
# declares the ordering (routes sorted by size, darkest = largest), and the
# ramp floor stays high enough that the palest route separates from the
# neutral minor-route gray.
ROUTE_SHADES = [mix(C_ROUTE, pct) for pct in (100, 80, 62, 48)]

# The [3,3] arbor of the gallery: compartments 0-2 are proximal, compartment
# 3 + 3*k .. 5 + 3*k are the children of proximal k.  This matches the
# compartment ordering documented in analyze_route_dictionary_atlas.py.
N_COMP = 12
SUBTREES = [np.array([k, 3 + 3 * k, 4 + 3 * k, 5 + 3 * k]) for k in range(3)]


def dictionary_matrices():
    """The three gallery dictionaries as dense arrays over 12 compartments."""
    broadcast = np.ones((N_COMP, 1))
    subtree = np.zeros((N_COMP, 3))
    for k, members in enumerate(SUBTREES):
        subtree[members, k] = 1.0
    exact = np.eye(N_COMP)
    # "one per neuron", not "broadcast": the ladder's strict layer-wide rung
    # is the one its Source Data calls "scalar broadcast"; this column is the
    # per-neuron coefficient repeated over the arbor.
    return [("broadcast_k1", "$K{=}1$\none per\nneuron", broadcast),
            ("subtrees_k3", "$K{=}3$\nsubtrees", subtree),
            ("exact_k12", "$K{=}12$\nexact", exact)]


def panel_definition(ax):
    """A: the shared credit tree plus the five named instantiations."""
    tree = ax.inset_axes([0.0, 0.14, 0.40, 0.84])
    tree.set_axis_off()
    draw_credit_tree(tree, mode="address", K=4, scale=0.86, labels=False)
    ax.text(0.19, 0.035, "addresses $A_u$ ($K{=}4$)\non one arbor", ha="center",
            va="bottom", fontsize=PT_SMALL, color=COLORS["ink"],
            transform=ax.transAxes, linespacing=1.0)

    entries = [
        (r"$A_u$ address matrix",
         f"delivery model ({fig_ref('framework', 'ladder')})"),
        ("Haar route basis", f"operator screens ({fig_ref('theory')})"),
        (r"$\Phi^{(K)}$ context routes",
         f"eight-context task ({fig_ref('factorial')})"),
        (r"$A_K$ anatomical subtrees",
         f"reconstructed arbors ({fig_ref('arbors')})"),
        ("fitted subtree routes",
         f"measured responses ({fig_ref('boundary')})"),
    ]
    x0 = 0.47
    top, bottom = 0.93, 0.06
    ys = np.linspace(top, bottom, len(entries))
    for y, (name, home) in zip(ys, entries):
        ax.add_patch(Rectangle((x0, y - 0.055), 0.018, 0.11,
                               transform=ax.transAxes,
                               facecolor=C_ROUTE, edgecolor="none"))
        ax.text(x0 + 0.035, y, name, ha="left", va="center",
                fontsize=PT_ANNOT, color=COLORS["ink"],
                transform=ax.transAxes)
        ax.text(x0 + 0.035, y - 0.062, home, ha="left", va="top",
                fontsize=PT_SMALL, color=COLORS["mute"],
                transform=ax.transAxes)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def panel_gallery(ax):
    """B: the trained fields next to each dictionary and its measured capture."""
    summary = pd.read_csv(ATLAS / "capture_summary.csv")
    field = pd.read_csv(ATLAS / "example_field.csv")
    profiles = {
        dynamics: (field[field.dynamics.eq(dynamics)]
                   .sort_values("compartment_index")
                   .mean_abs_error.to_numpy(float).reshape(-1, 1))
        for dynamics in ("additive", "shunting")}

    cap = {(row.dynamics, row.basis): float(row.mean_capture)
           for row in summary.itertuples(index=False)}

    seq = LinearSegmentedColormap.from_list(
        "atlas_field", ["#FFFFFF", mix(C_ROUTE, 88)])
    # 0.74, not 0.78: the three-line headers (the field pair's quantity line
    # and the K=1 column's "one per neuron") need that headroom under the
    # panel title, whose pad is only 3 pt.
    y0, y1 = 0.30, 0.74
    bar_y, bar_h = 0.125, 0.055

    def block(x, width, data, cmap, vmax):
        inset = ax.inset_axes([x, y0, width, y1 - y0])
        inset.imshow(data, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax,
                     interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(LW_HAIR)
            spine.set_color(COLORS["edge"])
        return inset

    # Two thin strips under one header: the additive profile (proximal low,
    # distal high) beside the shunting profile, whose depth weighting is the
    # reverse.  Each strip is the frozen table's own max-normalised column on
    # one 0-1 ramp; a rotated tag under each strip names its dynamics, and
    # the quantity symbol rides the shared header so the strip column stays
    # as narrow as one dictionary column.
    x, strip_w, strip_gap = 0.045, 0.04, 0.006
    for i, dynamics in enumerate(("additive", "shunting")):
        xs = x + i * (strip_w + strip_gap)
        block(xs, strip_w, profiles[dynamics], seq, 1.0)
        ax.text(xs + strip_w / 2, y0 - 0.02, dynamics, ha="center", va="top",
                rotation=90, fontsize=PT_SMALL, color=COLORS["mute"],
                transform=ax.transAxes)
    pair_w = 2 * strip_w + strip_gap
    ax.text(x + pair_w / 2, y1 + 0.035,
            "trained\nfield\n" + r"$|\partial\mathcal{L}/\partial V|$",
            ha="center", va="bottom", fontsize=PT_SMALL, color=COLORS["ink"],
            transform=ax.transAxes, linespacing=0.95)
    # Row-structure bracket: each strip's 12 rows are 3 proximal compartments
    # (top) above 9 distal ones; two hairline brackets with rotated tags make
    # the light/dark split -- and its reversal between strips -- decodable
    # without the caption.
    bx, cap_w, pad = x - 0.012, 0.006, 0.008
    y_split = y1 - (y1 - y0) * 3.0 / 12.0
    groups = [("prox", y_split + pad, y1 - pad),
              ("distal", y0 + pad, y_split - pad)]
    for tag, ylo, yhi in groups:
        ax.plot([bx, bx], [ylo, yhi], transform=ax.transAxes,
                color=COLORS["mute"], lw=LW_HAIR, clip_on=False)
        for ycap in (ylo, yhi):
            ax.plot([bx, bx + cap_w], [ycap, ycap], transform=ax.transAxes,
                    color=COLORS["mute"], lw=LW_HAIR, clip_on=False)
        ax.text(bx - 0.024, (ylo + yhi) / 2, tag, ha="center", va="center",
                rotation=90, fontsize=PT_SMALL, color=COLORS["mute"],
                transform=ax.transAxes)

    binary = LinearSegmentedColormap.from_list(
        "atlas_dict", ["#FFFFFF", C_ROUTE])
    # 0.185 and a 0.08 gap (not 0.175 / 0.085): the second field strip
    # widens the first column, and the right-hand legend cannot move.
    x = 0.185
    mat_ws = {1: 0.05, 3: 0.105, 12: 0.24}
    for key, label, matrix in dictionary_matrices():
        width = mat_ws[matrix.shape[1]]
        block(x, width, matrix, binary, 1.0)
        ax.text(x + width / 2, y1 + 0.035, label, ha="center", va="bottom",
                fontsize=PT_SMALL, color=COLORS["ink"],
                transform=ax.transAxes, linespacing=0.95)
        add = cap[("additive", key)]
        shunt = cap[("shunting", key)]
        ax.add_patch(Rectangle((x, bar_y), width, bar_h,
                               transform=ax.transAxes, facecolor="#EFF2F5",
                               edgecolor=COLORS["edge"], lw=LW_HAIR))
        ax.add_patch(Rectangle((x, bar_y), width * add, bar_h,
                               transform=ax.transAxes, facecolor=C_ROUTE,
                               edgecolor="none"))
        # The shunting capture rides the same bar as a thin ink tick, so the
        # graphic encodes both quantities the value line reports.
        ax.add_patch(Rectangle((x + width * shunt - 0.0015, bar_y), 0.003,
                               bar_h, transform=ax.transAxes,
                               facecolor=COLORS["ink"], edgecolor="none",
                               zorder=5))
        ax.text(x + width / 2, bar_y - 0.045,
                f"{100 * add:.0f}%\n({100 * shunt:.0f}%)",
                ha="center", va="top", fontsize=PT_SMALL,
                color=COLORS["ink"], transform=ax.transAxes,
                linespacing=1.05)
        x += width + 0.08

    ax.text(0.755, bar_y + bar_h / 2 - 0.02,
            "field-energy capture\nbar: additive\ntick: shunting", ha="left",
            va="center", fontsize=PT_SMALL, color=COLORS["mute"],
            transform=ax.transAxes, linespacing=1.05)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def arbor_routes():
    """The Fig. 7A cell with each segment assigned to a depth-1 subtree."""
    segments = pd.read_csv(SEGMENTS)
    sizes = segments.groupby("root_id").size()
    root = int((sizes - sizes.median()).abs().sort_values(kind="stable")
               .index[0])
    cell = segments[segments.root_id.eq(root)].copy()
    xyz = cell[["x_um", "y_um", "z_um"]].to_numpy(float)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ basis[:2].T
    # The same isotropic SVD projection and span normalisation as the
    # anatomy figure's morphology_geometry(), so span_um makes a metric
    # scale bar on this drawing too.
    span_um = max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
    projected /= span_um
    pos = {int(s): p for s, p in zip(cell.segment_id.to_numpy(int),
                                     projected, strict=True)}
    parent = {int(r.segment_id): int(r.parent_segment_id)
              for r in cell.itertuples(index=False)}
    root_seg = next(s for s, p in parent.items() if p < 0)

    # Walk each segment up to its depth-1 ancestor (child of the root).
    def depth1(seg):
        while parent[seg] >= 0 and parent[seg] != root_seg:
            seg = parent[seg]
        return seg if parent[seg] == root_seg else None

    membership = {}
    for seg in parent:
        if seg == root_seg:
            continue
        head = depth1(seg)
        if head is not None:
            membership.setdefault(head, []).append(seg)
    # Largest first; equal sizes (ranks 4 and 5 tie on this cell) break by
    # segment id so the drawn choice is explicit rather than iteration order.
    heads = sorted(membership, key=lambda h: (-len(membership[h]), h))
    return cell, pos, parent, root_seg, membership, heads, span_um


def panel_arbor_routes(ax):
    """C: the reconstructed arbor coloured by its four largest subtree routes."""
    cell, pos, parent, root_seg, membership, heads, span_um = arbor_routes()
    shade_of = {}
    for k, head in enumerate(heads[:4]):
        for seg in membership[head]:
            shade_of[seg] = ROUTE_SHADES[k]
    for row in cell.itertuples(index=False):
        seg, par = int(row.segment_id), int(row.parent_segment_id)
        if par < 0:
            continue
        color = shade_of.get(seg, "#EBEEF2")
        a, b = pos[par], pos[seg]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                lw=1.05 if seg in shade_of else LW_HAIR,
                solid_capstyle="round",
                zorder=3 if seg in shade_of else 2)
    sx, sy = pos[root_seg]
    ax.plot([sx], [sy], marker="o", ms=4.2, mfc=COLORS["soma"],
            mec=COLORS["edge"], mew=LW_HAIR, zorder=5, ls="none")
    # Route tags sit at each subtree's centroid, pushed radially away from
    # the soma so four labels near a central soma cannot pile up on it, plus
    # a small tangential offset (to whichever side clears the drawn strokes
    # better) so a near-radial terminal branch cannot end exactly on its tag.
    ink = [np.array(list(pos.values()))]
    ink += [(np.array([pos[s] for s in parent if parent[s] >= 0])
             + np.array([pos[parent[s]] for s in parent if parent[s] >= 0]))
            / 2.0]
    ink = np.vstack(ink)
    for k, head in enumerate(heads[:4]):
        pts = np.array([pos[s] for s in membership[head]])
        cx, cy = pts.mean(axis=0)
        norm = max(np.hypot(cx - sx, cy - sy), 1e-6)
        ux, uy = (cx - sx) / norm, (cy - sy) / norm
        bx, by = sx + ux * (norm + 0.24), sy + uy * (norm + 0.24)
        tx, ty = max(
            ((bx - uy * s, by + ux * s) for s in (-0.06, 0.06)),
            key=lambda c: np.hypot(ink[:, 0] - c[0], ink[:, 1] - c[1]).min())
        ax.text(tx, ty, f"$k{{=}}{k + 1}$", ha="center", va="center",
                fontsize=PT_SMALL, color=mix(ROUTE_SHADES[k], 65, "black"))
    # Metric scale bar in the arbor's empty lower-left corner: positions are
    # normalised by span_um under an isotropic projection, so 50/span_um data
    # units is 50 um, the bar the anatomy figure (Fig. 7A) carries.
    xy = np.array(list(pos.values()))
    x_lo, y_lo = xy.min(axis=0)
    bar_len = 50.0 / span_um
    ax.plot([x_lo, x_lo + bar_len], [y_lo, y_lo], color=COLORS["ink"],
            lw=LW_DATA, solid_capstyle="butt", zorder=7)
    ax.text(x_lo + bar_len / 2, y_lo + 0.02, "50 µm", ha="center",
            va="bottom", fontsize=PT_SMALL, color=COLORS["ink"], zorder=7)
    # Disclose a size tie at the drawn/undrawn boundary (ranks 4 and 5), so
    # "four largest" is not read as a strict ordering.  The short second
    # line keeps the note clear of the branch that ends in this corner.
    sizes = [len(membership[h]) for h in heads]
    if len(sizes) > 4 and sizes[3] == sizes[4]:
        ax.text(1.0, 0.02, f"ranks 4 and 5 tie\nat {sizes[3]} segments",
                ha="right", va="bottom", fontsize=PT_SMALL,
                color=COLORS["mute"], transform=ax.transAxes,
                linespacing=1.05)
    n_route = sum(len(membership[h]) for h in heads[:4])
    # The summary sits in the row gutter under the axes box, clear of the
    # arbor strokes that would otherwise run through it.
    ax.text(0.5, -0.02,
            f"4 of {len(heads)} depth-1 routes cover "
            f"{n_route}/{len(parent) - 1} segments",
            ha="center", va="top", fontsize=PT_SMALL, color=COLORS["mute"],
            transform=ax.transAxes, clip_on=False)
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.05)
    ax.set_axis_off()


def panel_roadmap(ax):
    """D: which dictionary each task family demands -- a labelled preview.

    A roadmap schematic in the same role as Fig. 1E: each row pairs one
    experimental task family with the dictionary the trained results select,
    so the reader carries the thesis into the evidence sections. The mini
    dictionaries are drawn to the conventions of panel B.
    """
    binary = LinearSegmentedColormap.from_list(
        "roadmap_dict", ["#FFFFFF", C_ROUTE])
    # Each mini carries its matrix shape (height scales with the row count,
    # and a two-line dimension tag reads "rows / x columns" in one
    # convention: what is addressed over what route), so the two 4-step
    # staircases -- 4x4 branch selectors vs 8x4 sibling pairs over terminal
    # parameter blocks (Fig. 6's leaves, not serial compartments) -- cannot
    # be mistaken for one another.  The third verdict states the frozen
    # factorial's contrast honestly: ancestry beats the best matched control
    # only at K=4 (+1.3 pp), not that K=4 is the accuracy optimum.
    rows = [
        ("image classes", f"MNIST ladder ({fig_ref('ladder')})",
         np.ones((12, 1)), 0.035, 0.22, "12 comp\n× 1 coefficient",
         "one coefficient\nper neuron suffices"),
        ("conflicting branches",
         r"$\chi>\chi_{\rm c}$" + f" ({fig_ref('conflict')})",
         np.eye(4), 0.10, 0.12, "4 branches\n× 4 selectors",
         "update gating\nbecomes necessary"),
        ("nested contexts", f"eight-context task ({fig_ref('factorial')})",
         np.kron(np.eye(4), np.ones((2, 1))), 0.10, 0.19,
         "8 blocks\n× 4 sibling pairs",
         "ancestry beats best\ncontrol only at $K{=}4$\n(+1.3 pp)"),
    ]
    # Rows sit 0.02 higher than before so the two-line tag under the third
    # mini clears the canvas's bottom margin.
    ys = (1.0, 0.66, 0.32)
    row_h = 0.30
    for (task, home, matrix, width, height, dims, verdict), y1 in zip(rows, ys):
        y0 = y1 - row_h
        mid = (y0 + y1) / 2
        ax.text(0.0, mid + 0.035, task, ha="left", va="center",
                fontsize=PT_ANNOT, color=COLORS["ink"],
                transform=ax.transAxes)
        ax.text(0.0, mid - 0.045, home, ha="left", va="center",
                fontsize=PT_SMALL, color=COLORS["mute"],
                transform=ax.transAxes)
        ax.annotate("", xy=(0.475, mid), xytext=(0.375, mid),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops={"arrowstyle": "-|>", "color": COLORS["mute"],
                                "lw": LW_REF, "shrinkA": 0, "shrinkB": 0})
        inset = ax.inset_axes([0.51, mid - height / 2, width, height])
        inset.imshow(matrix, aspect="auto", cmap=binary, vmin=0.0, vmax=1.0,
                     interpolation="nearest")
        if matrix.shape[0] == 8:
            # Hairline separators split each subtree block into its two
            # member compartments, distinguishing it from a branch selector.
            for yline in (0.5, 2.5, 4.5, 6.5):
                inset.axhline(yline, color="#FFFFFF", lw=LW_HAIR)
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(LW_HAIR)
            spine.set_color(COLORS["edge"])
        ax.text(0.51 + width / 2, mid - height / 2 - 0.018, dims,
                ha="center", va="top", fontsize=PT_SMALL,
                color=COLORS["mute"], transform=ax.transAxes,
                linespacing=1.0)
        ax.text(0.67, mid, verdict, ha="left", va="center",
                fontsize=PT_SMALL, color=COLORS["ink"],
                transform=ax.transAxes, linespacing=1.05)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


def main():
    canvas = NativeCanvas(
        HEIGHT_IN, nrows=2, row_weights=ROW_WEIGHTS,
        hgutter_pt=HGUTTER, vgutter_pt=VGUTTER, margins=MARGINS)
    ax_a = canvas.panel("A", 0, 0, 5, schematic=True,
                        title="One dictionary, five instantiations")
    ax_b = canvas.panel("B", 0, 5, 7, schematic=True,
                        title="Dictionary gallery on a [3,3] arbor")
    ax_c = canvas.panel("C", 1, 0, 5, schematic=True,
                        title="Anatomical routes (MICrONS)")
    ax_d = canvas.panel("D", 1, 5, 7, schematic=True,
                        title="Task-dependent routing results")
    panel_definition(ax_a)
    panel_gallery(ax_b)
    panel_arbor_routes(ax_c)
    panel_roadmap(ax_d)
    problems = canvas.save(OUT, name="main_figure_10_native")
    for violation in audit_native_pdf(OUT):
        print(f"    {violation}")
    return problems


if __name__ == "__main__":
    main()

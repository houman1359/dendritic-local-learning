"""Main Figure 2 -- route dictionaries over morphology, natively.

    row 0   A one dictionary, five instantiations | B gallery on a [3,3] arbor
    row 1   C anatomical routes (MICrONS)         | D task -> dictionary map

The figure introduces the paper's method up front: it names the route
dictionary once (A), shows what the candidate dictionaries look like and how
much of a trained credit field each captures (B), paints the same
construction on a reconstructed arbor (C), and previews which dictionary
each experimental task family demands (D, a roadmap schematic mirroring
Fig. 1E's experimental-logic role).

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

# Categorical shades for the four anatomical routes in C.  Subtree identity
# is categorical, but the palette reserves distinct hues for other series, so
# the four routes take an ordered ramp of the route green -- the caption
# states that shade encodes route identity, not magnitude.
ROUTE_SHADES = [mix(C_ROUTE, pct) for pct in (100, 76, 56, 40)]

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
    return [("broadcast_k1", "$K{=}1$\nbroadcast", broadcast),
            ("subtrees_k3", "$K{=}3$\nsubtrees", subtree),
            ("exact_k12", "$K{=}12$\nexact", exact)]


def panel_definition(ax):
    """A: the shared credit tree plus the five named instantiations."""
    tree = ax.inset_axes([0.0, 0.14, 0.40, 0.84])
    tree.set_axis_off()
    draw_credit_tree(tree, mode="address", K=4, scale=0.86, labels=False)
    ax.text(0.19, 0.035, "addresses $A_u$\non one arbor", ha="center",
            va="bottom", fontsize=PT_SMALL, color=COLORS["ink"],
            transform=ax.transAxes, linespacing=1.0)

    entries = [
        (r"$A_u$ address matrix", "delivery model (Figs 1, 3)"),
        ("Haar route basis", "operator screens (Fig. 4)"),
        (r"$\Phi^{(K)}$ context routes", "eight-context task (Fig. 6)"),
        (r"$A_K$ anatomical subtrees", "reconstructed arbors (Fig. 7)"),
        ("fitted subtree routes", "measured responses (Fig. 9)"),
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
    """B: the trained field next to each dictionary and its measured capture."""
    summary = pd.read_csv(ATLAS / "capture_summary.csv")
    field = pd.read_csv(ATLAS / "example_field.csv")
    field = field[field.dynamics.eq("additive")].sort_values("compartment_index")
    profile = field.mean_abs_error.to_numpy(float).reshape(-1, 1)

    cap = {(row.dynamics, row.basis): float(row.mean_capture)
           for row in summary.itertuples(index=False)}

    seq = LinearSegmentedColormap.from_list(
        "atlas_field", ["#FFFFFF", mix(C_ROUTE, 88)])
    y0, y1 = 0.30, 0.78
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

    x, strip_w = 0.02, 0.05
    block(x, strip_w, profile, seq, 1.0)
    ax.text(x + strip_w / 2, y1 + 0.035, "trained\nfield", ha="center",
            va="bottom", fontsize=PT_SMALL, color=COLORS["ink"],
            transform=ax.transAxes, linespacing=0.95)
    ax.text(x + strip_w / 2, y0 - 0.045,
            r"$|\partial\mathcal{L}/\partial V|$", ha="center", va="top",
            fontsize=PT_SMALL, color=COLORS["mute"], transform=ax.transAxes)

    binary = LinearSegmentedColormap.from_list(
        "atlas_dict", ["#FFFFFF", C_ROUTE])
    x = 0.175
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
        ax.text(x + width / 2, bar_y - 0.045,
                f"{100 * add:.0f}%\n({100 * shunt:.0f}%)",
                ha="center", va="top", fontsize=PT_SMALL,
                color=COLORS["ink"], transform=ax.transAxes,
                linespacing=1.05)
        x += width + 0.085

    ax.text(0.755, bar_y + bar_h / 2 - 0.02,
            "field-energy capture\nadditive (shunting)", ha="left",
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
    projected /= max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
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
    heads = sorted(membership, key=lambda h: len(membership[h]), reverse=True)
    return cell, pos, parent, root_seg, membership, heads


def panel_arbor_routes(ax):
    """C: the reconstructed arbor coloured by its four largest subtree routes."""
    cell, pos, parent, root_seg, membership, heads = arbor_routes()
    shade_of = {}
    for k, head in enumerate(heads[:4]):
        for seg in membership[head]:
            shade_of[seg] = ROUTE_SHADES[k]
    for row in cell.itertuples(index=False):
        seg, par = int(row.segment_id), int(row.parent_segment_id)
        if par < 0:
            continue
        color = shade_of.get(seg, "#E3E7EC")
        a, b = pos[par], pos[seg]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                lw=1.05 if seg in shade_of else 0.7,
                solid_capstyle="round",
                zorder=3 if seg in shade_of else 2)
    sx, sy = pos[root_seg]
    ax.plot([sx], [sy], marker="o", ms=4.2, mfc=COLORS["soma"],
            mec=COLORS["edge"], mew=LW_HAIR, zorder=5, ls="none")
    # Route tags sit at each subtree's centroid, pushed radially away from
    # the soma so four labels near a central soma cannot pile up on it.
    for k, head in enumerate(heads[:4]):
        pts = np.array([pos[s] for s in membership[head]])
        cx, cy = pts.mean(axis=0)
        norm = max(np.hypot(cx - sx, cy - sy), 1e-6)
        tx = sx + (cx - sx) / norm * (norm + 0.16)
        ty = sy + (cy - sy) / norm * (norm + 0.16)
        ax.text(tx, ty, f"$k{{=}}{k + 1}$", ha="center", va="center",
                fontsize=PT_SMALL, color=mix(ROUTE_SHADES[k], 100, "black"))
    n_route = sum(len(membership[h]) for h in heads[:4])
    ax.text(0.02, 0.01,
            f"4 of {len(heads)} depth-1 routes cover "
            f"{n_route}/{len(parent) - 1} segments",
            ha="left", va="bottom", fontsize=PT_SMALL, color=COLORS["mute"],
            transform=ax.transAxes)
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
    rows = [
        ("image classes", "MNIST ladder (Fig. 3)",
         np.ones((12, 1)), 0.035,
         "one coefficient\nper neuron suffices"),
        ("conflicting branches", r"$\chi>\chi_{\rm c}$ (Fig. 5)",
         np.eye(4), 0.10,
         "branch selectors\nbecome necessary"),
        ("nested contexts", "eight-context task (Fig. 6)",
         np.kron(np.eye(4), np.ones((2, 1))), 0.10,
         "subtrees win in a\n$K{=}4$ window"),
    ]
    ys = (1.0, 0.65, 0.30)
    row_h = 0.30
    for (task, home, matrix, width, verdict), y1 in zip(rows, ys):
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
        inset = ax.inset_axes([0.51, y0 + 0.035, width, row_h - 0.07])
        inset.imshow(matrix, aspect="auto", cmap=binary, vmin=0.0, vmax=1.0,
                     interpolation="nearest")
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_linewidth(LW_HAIR)
            spine.set_color(COLORS["edge"])
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
                        title="Which dictionary a task demands")
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

#!/usr/bin/env python3
"""Build the integrated journal figures from frozen source data.

The visual system is intentionally the same one used for the NeurIPS paper:
fixed 7.2-inch canvases, one typography and line-weight scale, uppercase panel
letters, consistent condition colours, and no tight-bounding-box rescaling.
Figures are expanded for the journal format by showing examples, biological
replicates, controls, and mechanism-level analyses rather than only summary
statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

from figure1_neurips_components import panel_a as neurips_panel_a
from figure1_neurips_components import panel_b as neurips_panel_b
from inherited_neurips.generate_theory_diagnostics_figures import (
    _plot_path_gain_map as neurips_path_gain_panel,
)
from journal_style import (
    COLORS,
    DIV_CMAP,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKERS,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    SEED_ALPHA,
    SEED_MS,
    add_colorbar,
    add_headroom,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    paired_lines,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data"
FIGURES = ROOT / "figures" / "generated"

apply_neurips_style()

MORPH = COLORS["shunting"]
RANDOM = COLORS["point_mlp"]
DEPTH = COLORS["additive"]
SHUFFLE = COLORS["highlight"]
DENSE = COLORS["oracle"]
EXACT = COLORS["bp"]

METHOD_COLORS = {
    "dense PCA oracle": DENSE,
    "morphology-aware paths": MORPH,
    "morphology-selected paths": MORPH,
    "random paths": RANDOM,
    "random nonempty paths": RANDOM,
    "depth-only bins": DEPTH,
    "depth bins": DEPTH,
    "shuffled ancestry": SHUFFLE,
    "ancestry-shuffled paths": SHUFFLE,
    "scalar broadcast": COLORS["scalar"],
    "exact backprop": EXACT,
}


def mean_ci(values: np.ndarray, seed: int = 0, n_boot: int = 20_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(lo), float(hi)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    layout = audit_layout(fig, stem)
    overlap = audit_text_over_data(fig, stem)
    if layout or overlap:
        print(f"  review {stem}: {len(layout)} layout, {len(overlap)} text/data warnings")
    fig.savefig(
        FIGURES / f"{stem}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / f"{stem}.png", dpi=600)
    plt.close(fig)


def journal_grid(nrows: int, ncols: int, *, height: float = 5.00,
                 width_ratios=None, height_ratios=None,
                 wspace: float = 0.72, hspace: float = 0.76):
    fig = plt.figure(figsize=(FIG_W, height))
    gs = fig.add_gridspec(
        nrows,
        ncols,
        left=0.088,
        right=0.985,
        bottom=0.090,
        top=0.925,
        wspace=wspace,
        hspace=hspace,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
    )
    return fig, gs


def eight_panel_grid(*, height: float = 6.35):
    """Three journal rows: 3 + 3 + 2 panels at the canonical width."""
    fig = plt.figure(figsize=(FIG_W, height))
    gs = fig.add_gridspec(
        3, 6, left=0.088, right=0.985, bottom=0.068, top=0.935,
        wspace=0.94, hspace=0.88,
    )
    spans = [
        (0, slice(0, 2)), (0, slice(2, 4)), (0, slice(4, 6)),
        (1, slice(0, 2)), (1, slice(2, 4)), (1, slice(4, 6)),
        (2, slice(0, 3)), (2, slice(3, 6)),
    ]
    return fig, [fig.add_subplot(gs[r, c]) for r, c in spans]


def ten_panel_grid(*, height: float = 6.25):
    """Four compact journal rows at the canonical full-width print scale.

    Keeping the original 3 + 3 + 2 + 2 grouping avoids squeezing quantitative
    panels horizontally. The shorter canvas replaces the former LaTeX
    down-scaling and therefore preserves the paper-wide type size.
    """
    fig = plt.figure(figsize=(FIG_W, height))
    gs = fig.add_gridspec(
        4, 6, left=0.088, right=0.985, bottom=0.078, top=0.940,
        wspace=0.94, hspace=1.08,
    )
    spans = [
        (0, slice(0, 2)), (0, slice(2, 4)), (0, slice(4, 6)),
        (1, slice(0, 2)), (1, slice(2, 4)), (1, slice(4, 6)),
        (2, slice(0, 3)), (2, slice(3, 6)),
        (3, slice(0, 3)), (3, slice(3, 6)),
    ]
    return fig, [fig.add_subplot(gs[row, columns]) for row, columns in spans]


def nine_panel_grid(*, height: float = 6.55):
    fig, gs = journal_grid(3, 3, height=height, wspace=0.92, hspace=0.90)
    return fig, [fig.add_subplot(gs[row, column]) for row in range(3) for column in range(3)]


def jitter(n: int, seed: int, scale: float = 0.045) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, int(n))


def errorbar_mean(ax, x, values, color, *, seed=0, marker="D", zorder=5):
    m, lo, hi = mean_ci(np.asarray(values, dtype=float), seed=seed)
    ax.errorbar(
        x, m, yerr=[[m - lo], [hi - m]], marker=marker, ms=4.6,
        color=color, markeredgecolor="white", markeredgewidth=0.45,
        lw=LW_ERR, capsize=ERR_CAPSIZE, zorder=zorder,
    )
    return m, lo, hi


def draw_compact_tree(ax, x0: float, y0: float, scale: float = 1.0,
                      *, color=COLORS["mute"], selected=None):
    nodes = {
        "soma": (x0 + 0.86 * scale, y0),
        "p": (x0 + 0.64 * scale, y0),
        "m1": (x0 + 0.40 * scale, y0 + 0.20 * scale),
        "m2": (x0 + 0.40 * scale, y0 - 0.20 * scale),
        "d1": (x0 + 0.10 * scale, y0 + 0.34 * scale),
        "d2": (x0 + 0.10 * scale, y0 + 0.08 * scale),
        "d3": (x0 + 0.10 * scale, y0 - 0.08 * scale),
        "d4": (x0 + 0.10 * scale, y0 - 0.34 * scale),
    }
    edges = [("d1", "m1"), ("d2", "m1"), ("d3", "m2"), ("d4", "m2"),
             ("m1", "p"), ("m2", "p"), ("p", "soma")]
    selected = set() if selected is None else set(selected)
    for a, b in edges:
        ec = MORPH if a in selected and b in selected else color
        ax.plot([nodes[a][0], nodes[b][0]], [nodes[a][1], nodes[b][1]],
                color=ec, lw=2.2 if ec == MORPH else 1.25,
                solid_capstyle="round", zorder=2)
    for name, (x, y) in nodes.items():
        if name == "soma":
            ax.add_patch(Circle((x, y), 0.055 * scale, fc=COLORS["soma"],
                                ec=COLORS["edge"], lw=LW_EDGE, zorder=4))
        else:
            ax.add_patch(Circle((x, y), 0.022 * scale,
                                fc="#EAF5EF" if name in selected else "white",
                                ec=MORPH if name in selected else color,
                                lw=LW_EDGE, zorder=4))
    return nodes


# -------------------------------------------------------------------------
# Figure 1: exact NeurIPS framework components
# -------------------------------------------------------------------------


def _panel_credit_hierarchy(ax: plt.Axes) -> None:
    """Port the workshop's coordinate--address--gain tree strip to Figure 1.

    The workshop CreditTree schematic is deliberately redrawn with the
    journal's vector primitives so the paper and talk share the same visual
    grammar without introducing a raster or external TeX build dependency.
    """

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "C", "Coordinate → address → gain")

    ax.text(0.50, 0.86,
            r"$\partial\mathcal{L}/\partial g_i=e_i\;\delta_u\;\widetilde\alpha_n$",
            ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["ink"])

    x_origins = (0.00, 0.34, 0.68)
    trees = [draw_compact_tree(ax, x, 0.50, scale=0.29,
                               color=COLORS["grid"])
             for x in x_origins]

    # Coordinate: one neuronal value reaches the soma without resolving the
    # interior of the arbor.
    coordinate = trees[0]
    ax.add_patch(FancyArrowPatch((0.31, 0.50),
                                 (coordinate["soma"][0] + 0.018, 0.50),
                                 arrowstyle="-|>", mutation_scale=7,
                                 linewidth=1.0, color=COLORS["exc"]))
    ax.text(0.292, 0.60, r"$\delta_u$", ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["exc"])

    def route(nodes, names, color, width):
        for left, right in zip(names[:-1], names[1:]):
            ax.plot([nodes[left][0], nodes[right][0]],
                    [nodes[left][1], nodes[right][1]],
                    color=color, lw=width, solid_capstyle="round", zorder=5)

    # Address: independently colored descendant domains instantiate the
    # workshop's within-tree routing variable.
    addressed = trees[1]
    for leaf in ("d1", "d2"):
        route(addressed, (leaf, "m1"), MORPH, 2.4)
    for leaf in ("d3", "d4"):
        route(addressed, (leaf, "m2"), COLORS["oracle"], 2.4)
    ax.text(addressed["d1"][0] - 0.008, addressed["d1"][1] + 0.075,
            r"$\delta_{u,1}$", ha="center", va="center", fontsize=PT_SMALL,
            color=MORPH)
    ax.text(addressed["d4"][0] - 0.008, addressed["d4"][1] - 0.075,
            r"$\delta_{u,2}$", ha="center", va="center", fontsize=PT_SMALL,
            color=COLORS["oracle"])

    # Gain: the address remains, while line width makes route-specific
    # conductance gain visible at a glance.
    gained = trees[2]
    route(gained, ("d1", "m1", "p", "soma"), MORPH, 3.4)
    route(gained, ("d4", "m2", "p"), COLORS["oracle"], 1.25)
    ax.add_patch(Circle(gained["m1"], 0.030, fc="white", ec=MORPH,
                        lw=1.2, zorder=7))
    ax.text(gained["m1"][0] + 0.002, gained["m1"][1] + 0.078,
            r"$\widetilde\alpha_n$", ha="center", va="center",
            fontsize=PT_SMALL, color=MORPH)

    for left, right in ((0.294, 0.337), (0.634, 0.677)):
        ax.add_patch(FancyArrowPatch((left, 0.50), (right, 0.50),
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.8, color=COLORS["mute"], zorder=8))

    labels = (
        (0.145, "coordinate", "which neuron"),
        (0.485, "address", "which subtree"),
        (0.825, "gain", "how strongly"),
    )
    for x, label, detail in labels:
        ax.text(x, 0.225, label, ha="center", va="center", fontsize=PT_ANNOT,
                color=COLORS["ink"])
        ax.text(x, 0.155, detail, ha="center", va="center", fontsize=PT_SMALL,
                color=COLORS["mute"])

    ax.text(0.50, 0.025, "Tree contribution after neuronal credit arrives",
            ha="center", va="bottom", fontsize=PT_SMALL, color=COLORS["mute"])


def _panel_general_adjoint(ax: plt.Axes) -> None:
    """Teach the eligibility-by-transport factorization pictorially."""

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "D", "Local eligibility × transported error")

    # Local synaptic information.
    ax.plot([0.07, 0.39], [0.61, 0.61], color=COLORS["mute"], lw=2.0,
            solid_capstyle="round")
    ax.add_patch(Circle((0.22, 0.61), 0.030, fc=COLORS["exc"], ec="white",
                        lw=0.6, zorder=4))
    ax.add_patch(FancyArrowPatch((0.08, 0.82), (0.20, 0.65), arrowstyle="-|>",
                                 mutation_scale=8, lw=1.2, color=COLORS["exc"]))
    ax.text(0.07, 0.86, "presynaptic\ndrive", ha="left", va="center",
            fontsize=PT_SMALL, color=COLORS["exc"])
    ax.text(0.23, 0.49, "LOCAL ELIGIBILITY", ha="center", va="center",
            fontsize=PT_SMALL, color=COLORS["exc"])
    ax.text(0.23, 0.39, "activity · driving force · input resistance",
            ha="center", va="center", fontsize=PT_SMALL, color=COLORS["mute"])

    # Multiplication is the key operation.
    ax.add_patch(Circle((0.49, 0.61), 0.055, fc="white", ec=COLORS["edge"],
                        lw=1.0, zorder=4))
    ax.text(0.49, 0.61, "×", ha="center", va="center", fontsize=12,
            color=COLORS["ink"], zorder=5)

    # A transported error follows the unique tree path in the directed model.
    nodes = draw_compact_tree(ax, 0.58, 0.61, scale=0.38,
                              color=COLORS["mute"], selected={"d1", "m1", "p"})
    soma = nodes["soma"]
    ax.add_patch(FancyArrowPatch((0.94, soma[1] + 0.05),
                                 (nodes["p"][0] + 0.015, nodes["p"][1]),
                                 arrowstyle="-|>", mutation_scale=8, lw=1.2,
                                 color=COLORS["oracle"], connectionstyle="arc3,rad=.12"))
    ax.text(0.96, soma[1] + 0.13, "loss-derived\ncoordinate",
            ha="right", va="center", fontsize=PT_SMALL, color=COLORS["oracle"])
    ax.text(0.74, 0.39, "PATH TRANSPORT", ha="center", va="center",
            fontsize=PT_SMALL, color=MORPH)
    ax.text(0.74, 0.30, "route-specific gain", ha="center", va="center",
            fontsize=PT_SMALL, color=COLORS["mute"])

    result = FancyBboxPatch(
        (0.24, 0.05), 0.52, 0.13,
        boxstyle="round,pad=0.010,rounding_size=0.025",
        facecolor="#F6F8FA", edgecolor=COLORS["edge"], linewidth=LW_EDGE,
    )
    ax.add_patch(result)
    ax.text(0.50, 0.115, r"synaptic update  $\propto$  eligibility $\times$ error",
            ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["ink"])


def _panel_evidence_ladder(ax: plt.Axes) -> None:
    """A neutral roadmap from mathematical object to empirical boundary."""

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "E", "Evidence path")

    rows = [
        (0.82, "Exact factorization", "eligibility × compartment error", COLORS["bp"]),
        (0.66, "Signal–noise theory", "useful route resolution", COLORS["oracle"]),
        (0.50, "Trained route tests", "identity → ownership → address → depth", COLORS["per_soma"]),
        (0.34, "Anatomical capacity", "available sparse ancestry fields", MORPH),
        (0.18, "Functional boundary", "measured task–route alignment", COLORS["highlight"]),
    ]
    for index, (y, tag, question, color) in enumerate(rows):
        ax.add_patch(Circle((0.07, y), 0.027, facecolor=color, edgecolor="white",
                            linewidth=0.5, zorder=3))
        ax.text(0.12, y + 0.027, tag, ha="left", va="center",
                fontsize=PT_SMALL, color=color)
        ax.text(0.12, y - 0.032, question, ha="left", va="center",
                fontsize=PT_SMALL, color=COLORS["ink"])
        if index < len(rows) - 1:
            next_y = rows[index + 1][0]
            ax.add_patch(FancyArrowPatch((0.07, y - 0.035), (0.07, next_y + 0.035),
                                         arrowstyle="-|>", mutation_scale=7,
                                         linewidth=0.8, color=COLORS["grid"], zorder=1))


def _restyle_neuron_schematic(ax: plt.Axes) -> None:
    """Unify the frozen A/B panels with the paper's schematic vocabulary.

    The inherited generators draw somas as gray discs with heavy near-black
    rings plus amber halos, dendrites in a taupe brown, and mixed neutral
    grays for labels.  The journal glyph set (figs 2E, 3A/B, 4A) is an
    orange COLORS['soma'] disc with a thin COLORS['edge'] outline over
    slate-neutral dendrites.  The frozen module stays untouched; its
    artists are recolored post hoc here.
    """
    from matplotlib.colors import to_hex, to_rgb

    def hex_of(color) -> str:
        try:
            return to_hex(to_rgb(color)).lower()
        except (TypeError, ValueError):
            return ""

    for line in ax.lines:
        if hex_of(line.get_color()) == "#8a7a6a":
            line.set_color(COLORS["mute"])
    for patch in ax.patches:
        if not isinstance(patch, Circle):
            continue
        face = patch.get_facecolor()
        fc = hex_of(face) if (len(face) < 4 or face[3] > 0) else ""
        ec = hex_of(patch.get_edgecolor())
        if fc in {"#e8e8e8", "#efefef"}:
            # Soma discs: orange fill, thin gray outline.
            patch.set_facecolor(COLORS["soma"])
            patch.set_edgecolor(COLORS["edge"])
            patch.set_linewidth(min(patch.get_linewidth(), 0.8))
        elif fc == "#fff3d8":
            # Amber halo -> recessive open highlight ring.
            patch.set_facecolor("none")
            patch.set_edgecolor(COLORS["local"])
            patch.set_linewidth(0.85)
        elif ec in {"#555555", "#6e5d4f", "#666666"}:
            patch.set_edgecolor(COLORS["edge"])
    ink_sources = {"#111111", "#333333", "#444444"}
    mute_sources = {"#555555", "#777777"}
    for text in ax.texts:
        source = hex_of(text.get_color())
        if source in ink_sources:
            text.set_color(COLORS["ink"])
        elif source in mute_sources:
            text.set_color(COLORS["mute"])
        arrow = getattr(text, "arrow_patch", None)
        if arrow is not None and hex_of(arrow.get_edgecolor()) in {"#333333", "#666666"}:
            arrow.set_color(COLORS["edge"])


def _restyle_inherited_panel(ax: plt.Axes, letter: str, title: str) -> None:
    """Bring a frozen NeurIPS schematic panel onto the journal type system.

    The inherited generators set bold panel titles and bold in-panel text;
    the journal reserves bold for panel letters.  The artists are restyled
    here rather than in the (hash-frozen) component module.
    """

    old_letter = getattr(ax, "_neurips_panel_letter", None)
    if old_letter is not None:
        old_letter.remove()
        ax._neurips_panel_letter = None
    letter_artist = panel_title(ax, letter, title)
    for text in ax.texts:
        if text is letter_artist:
            continue
        text.set_fontweight("normal")


def figure1() -> None:
    # A and B preserve the polished NeurIPS model language. C--E add the
    # journal-specific hierarchy, unifying theorem, and evidential scope.
    # The left margin clears the fixed -30 pt panel-letter gutter so the
    # A and D letters cannot clip at the canvas edge.
    fig = plt.figure(figsize=(FIG_W, 5.10))
    grid = fig.add_gridspec(
        2, 6, height_ratios=[1.08, 0.92], hspace=0.34, wspace=0.20,
        left=0.078, right=0.985, top=0.925, bottom=0.045,
    )
    ax_a = fig.add_subplot(grid[0, 0:2])
    ax_b = fig.add_subplot(grid[0, 2:4])
    ax_c = fig.add_subplot(grid[0, 4:6])
    ax_d = fig.add_subplot(grid[1, 0:3])
    ax_e = fig.add_subplot(grid[1, 3:6])
    neurips_panel_a(ax_a)
    neurips_panel_b(ax_b)
    _restyle_inherited_panel(ax_a, "A", "Dendritic E/I unit")
    _restyle_inherited_panel(ax_b, "B", "Network layer")
    _restyle_neuron_schematic(ax_a)
    _restyle_neuron_schematic(ax_b)
    _panel_credit_hierarchy(ax_c)
    _panel_general_adjoint(ax_d)
    _panel_evidence_ladder(ax_e)

    save(fig, "fig1_framework")


# -------------------------------------------------------------------------
# Figure 2: exactness, feedback identity, and learning
# -------------------------------------------------------------------------


ARCH_MARKERS = {"dendritic_shunting": "o", "dendritic_additive": "s"}


def paired_feedback_panel(ax, data, metric, *, ylabel, ylim, gradient=False):
    archs = ["dendritic_shunting", "dendritic_additive"]
    colors = {archs[0]: COLORS["shunting"], archs[1]: COLORS["additive"]}
    offsets = {archs[0]: -0.08, archs[1]: 0.08}
    if gradient:
        data = data[data["trained_broadcast_mode"].eq("per_soma_shared")].copy()
        data["condition"] = data["diagnostic_feedback"].map(
            {"scalar_fallback": "scalar", "neuron_wise": "ancestry"})
        index = "seed"
    else:
        data = data.copy()
        data["condition"] = data["feedback"].map(
            {"scalar_fallback": "scalar", "ancestry_shared": "ancestry"})
        index = "seed"
    for arch in archs:
        pivot = data[data["network_type"].eq(arch)].pivot_table(
            index=index, columns="condition", values=metric, aggfunc="mean").dropna()
        xs = np.array([0, 1], float) + offsets[arch]
        paired_lines(ax, xs[0], xs[1], pivot["scalar"], pivot["ancestry"],
                     color=colors[arch], lw=LW_HAIR, alpha=0.24)
        for i, cond in enumerate(["scalar", "ancestry"]):
            vals = pivot[cond].to_numpy(float)
            ax.scatter(np.full(vals.size, xs[i]) + jitter(vals.size, 30 + i, 0.018), vals,
                       s=SEED_MS ** 2, color=colors[arch], alpha=SEED_ALPHA,
                       marker=ARCH_MARKERS[arch],
                       edgecolor="white", linewidth=0.25, zorder=3)
            errorbar_mean(ax, xs[i], vals, colors[arch], seed=40 + i)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["scalar\nfallback", "neuron\nindexed"])
    ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    style_axis(ax)


def _restyle_path_gain_schematic(ax: plt.Axes, cv_summary: pd.DataFrame) -> None:
    """Repaint the inherited viridis path-gain trees in the journal palette.

    The frozen generator colours the three paths of each tree with a shared
    viridis ramp and prints bold annotations.  Here each tree keeps its own
    architecture hue (additive blue, shunting green) whose lightness encodes
    the same log path-gain scale, the annotations drop the bold weight, and
    the delta_0 error label moves off the soma disc so both junctions read
    as the same orange node.
    """
    from matplotlib.colors import Normalize, to_rgb

    cvs = cv_summary.set_index("network_type")["path_gain_cv_mean_mean"]
    add_cv = float(cvs["dendritic_additive"])
    shunt_cv = float(cvs["dendritic_shunting"])
    add_gains = [1.0 - add_cv / 3.0, 1.0, 1.0 + add_cv]
    shunt_gains = [1.0 - shunt_cv / 3.0, 1.0, 1.0 + shunt_cv]
    logs = np.log10(np.clip(np.asarray(add_gains + shunt_gains), 1e-5, None))
    norm = Normalize(vmin=float(logs.min()), vmax=float(logs.max()))

    def shade(base: str, gain: float) -> tuple[float, ...]:
        weight = 0.35 + 0.65 * float(norm(np.log10(max(gain, 1e-5))))
        return tuple(1.0 - weight * (1.0 - channel) for channel in to_rgb(base))

    shades = [shade(COLORS["additive"], gain) for gain in add_gains]
    shades += [shade(COLORS["shunting"], gain) for gain in shunt_gains]
    for line, color in zip(list(ax.lines)[:6], shades):
        line.set_color(color)
    circles = [patch for patch in ax.patches if isinstance(patch, Circle)]
    # Patch order: additive leaves 0-2, additive soma, shunting leaves 4-6.
    for shade_index, circle_index in enumerate((0, 1, 2, 4, 5, 6)):
        circles[circle_index].set_facecolor(shades[shade_index])
    letter_artist = getattr(ax, "_neurips_panel_letter", None)
    for text in ax.texts:
        if text is letter_artist:
            continue
        text.set_fontweight("normal")
        if r"\delta_0" in text.get_text():
            text.set_position((0.94, 0.578))
            text.set_ha("center")
            text.set_va("bottom")
            text.set_bbox(dict(facecolor="none", edgecolor="none", pad=0))
    ax.text(0.34, 0.578, r"$\delta_0$", ha="center", va="bottom",
            fontsize=PT_LEGEND, color=COLORS["ink"], zorder=6)


def figure2() -> None:
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_W, 5.15),
        gridspec_kw={
            "left": 0.09,
            "right": 0.985,
            "bottom": 0.095,
            "top": 0.93,
            "wspace": 0.55,
            "hspace": 0.58,
        },
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    diag = pd.read_csv(DATA / "figure2" / "exact_gradient_reconstruction_runs.csv")
    # Exact visual and numerical definition of NeurIPS Fig. 2A. The source
    # generator is preserved byte-for-byte under scripts/inherited_neurips/.
    rel_error = diag["factorization_weighted_relative_l2"].to_numpy(float)
    scale_error = diag["factorization_weighted_scale_mismatch"].to_numpy(float)
    rng = np.random.default_rng(7)
    for i, (values, color) in enumerate([
        (rel_error, COLORS["rule_3f"]),
        (scale_error, COLORS["rule_5f"]),
    ]):
        ax_a.scatter(
            np.full_like(values, i, dtype=float) + rng.uniform(-0.08, 0.08, len(values)),
            values,
            s=32,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=LW_EDGE,
            zorder=3,
        )
        ax_a.hlines(np.median(values), i - 0.18, i + 0.18,
                    color=COLORS["ink"], lw=LW_DATA, zorder=4)
    ax_a.set_xticks([0, 1])
    ax_a.set_xticklabels(["rel. $L_2$", "rel. norm"])
    ax_a.set_ylabel("reconstruction error")
    ax_a.set_yscale("log")
    ax_a.set_ylim(1e-10, 1e-4)
    panel_title(ax_a, "A", "Reconstruction")
    style_axis(ax_a)

    acc = pd.read_csv(DATA / "figure2" / "feedback_accuracy_runs.csv")
    paired_feedback_panel(ax_b, acc, "test_accuracy", ylabel="MNIST accuracy",
                          ylim=(0.875, 0.978))
    ax_b.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax_b, "B", "Neuron-indexed feedback")
    clean_legend(
        ax_b, handles=[
            Line2D([0], [0], color=COLORS["shunting"], marker="o", label="shunting"),
            Line2D([0], [0], color=COLORS["additive"], marker="s", label="additive"),
        ], loc="lower right",
    )

    grad = pd.read_csv(DATA / "figure2" / "feedback_gradient_runs.csv")
    paired_feedback_panel(ax_c, grad, "branch_numel_weighted_cosine",
                          ylabel="dendritic-gradient cosine",
                          ylim=(-0.14, 0.80), gradient=True)
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF, zorder=0)
    panel_title(ax_c, "C", "Gradient alignment")
    clean_legend(
        ax_c, handles=[
            Line2D([0], [0], color=COLORS["shunting"], marker="o", label="shunting"),
            Line2D([0], [0], color=COLORS["additive"], marker="s", label="additive"),
        ], loc="lower right", auto_clear=True,
    )

    exact = pd.read_csv(DATA / "figure2" / "exact_transport_and_backprop_runs.csv")
    # The four factorization conditions use a lightness-only ramp of the
    # non-rule amber family (dark -> light: 3F local, 3F BP decoder, 5F local,
    # 5F BP decoder); identity is carried by the category axis.  Green/blue
    # tints are avoided here because panels B, C and F of this figure bind
    # green = shunting and blue = additive, and only backprop may wear the
    # manuscript-wide bp red-brown.
    conditions = [
        ("3f", "local", "3F\nlocal", "#9C6B1C"),
        ("3f", "backprop", "3F\nBP decoder", "#B58433"),
        ("5f", "local", "5F\nlocal", "#CFA04F"),
        ("5f", "backprop", "5F\nBP decoder", "#E4BC72"),
        ("backpropagation", "backprop", "backprop", COLORS["bp"]),
    ]
    for i, (rule, decoder, label, color) in enumerate(conditions):
        vals_i = exact[exact["rule"].eq(rule) & exact["decoder_mode"].eq(decoder)]["test_accuracy"].to_numpy(float)
        ax_d.scatter(i + jitter(vals_i.size, 90 + i, 0.055), vals_i,
                     s=SEED_MS ** 2, color=color, edgecolor="white", linewidth=0.3, zorder=3)
        errorbar_mean(ax_d, i, vals_i, color, seed=100 + i)
    ax_d.set_xticks(range(5)); ax_d.set_xticklabels(["3F\nlocal", "3F\nBP", "5F\nlocal", "5F\nBP", "BP"])
    ax_d.set_ylim(0.967, 0.975); ax_d.set_ylabel("MNIST accuracy")
    ax_d.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=1))
    panel_title(ax_d, "D", "Exact error transport")
    style_axis(ax_d)

    cv = pd.read_csv(DATA / "figure2" / "path_gain_cv_runs.csv")
    cv_summary = (
        cv.groupby("network_type", as_index=False)["path_gain_cv_mean"]
        .mean()
        .rename(columns={"path_gain_cv_mean": "path_gain_cv_mean_mean"})
    )
    cv_summary["dataset"] = "mnist"
    cv_summary["ie_value"] = 5
    # Direct call into the byte-identical NeurIPS mechanism generator. Only
    # the panel letter is relabeled after rendering because its journal slot is E.
    neurips_path_gain_panel(ax_e, cv_summary, cv)
    inherited_letter = getattr(ax_e, "_neurips_panel_letter", None)
    if inherited_letter is not None:
        inherited_letter.remove()
        ax_e._neurips_panel_letter = None
    panel_title(ax_e, "E", "Path gains")
    _restyle_path_gain_schematic(ax_e, cv_summary)

    init = pd.read_csv(DATA / "figure2" / "initialization_factorial_runs.csv")
    positions = {"analytical": 0, "occupancy_quantile": 1}
    offsets = {"dendritic_additive": -0.09, "dendritic_shunting": 0.09}
    for arch, color in [("dendritic_additive", COLORS["additive"]),
                        ("dendritic_shunting", COLORS["shunting"])]:
        pivot = init[init["network_type"].eq(arch)].pivot_table(
            index="seed", columns="init_policy", values="test_accuracy").dropna()
        xs = np.array([positions["analytical"], positions["occupancy_quantile"]]) + offsets[arch]
        paired_lines(ax_f, xs[0], xs[1], pivot["analytical"], pivot["occupancy_quantile"],
                     color=color, alpha=0.22)
        for j, policy in enumerate(["analytical", "occupancy_quantile"]):
            arr = pivot[policy].to_numpy(float)
            ax_f.scatter(xs[j] + jitter(arr.size, 150 + j, 0.016), arr,
                         s=SEED_MS ** 2, color=color, alpha=SEED_ALPHA,
                         marker=ARCH_MARKERS[arch],
                         edgecolor="white", linewidth=0.25, zorder=3)
            errorbar_mean(ax_f, xs[j], arr, color, seed=160 + j)
    ax_f.set_xticks([0, 1]); ax_f.set_xticklabels(["analytical", "occupancy\nquantile"])
    ax_f.set_ylabel("MNIST accuracy"); ax_f.set_ylim(0.895, 0.932)
    ax_f.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    panel_title(ax_f, "F", "Initialization factorial")
    style_axis(ax_f)
    clean_legend(
        ax_f, handles=[
            Line2D([0], [0], color=COLORS["shunting"], marker="o", label="shunting"),
            Line2D([0], [0], color=COLORS["additive"], marker="s", label="additive"),
        ], loc="upper left", auto_clear=True,
    )

    save(fig, "fig2_feedback")


# -------------------------------------------------------------------------
# Figure 4: topology-derived routing in the original MICrONS cohort
# -------------------------------------------------------------------------


def descendants(parent: dict[int, int], node: int) -> set[int]:
    out = set()
    for candidate in parent:
        current = candidate
        seen = set()
        while current != -1 and current in parent and current not in seen:
            seen.add(current)
            if current == node:
                out.add(candidate)
                break
            current = parent[current]
    return out


def exemplar_cell():
    seg = pd.read_csv(DATA / "figure3" / "segment_metrics.csv")
    root = int(seg.groupby("root_id").size().idxmax())
    return seg[seg["root_id"].eq(root)].copy(), root


def plot_morphology(ax, cell, *, show_domains=False, show_contacts=False):
    rows = {int(r.segment_id): r for r in cell.itertuples(index=False)}
    parent = {int(r.segment_id): int(r.parent_segment_id) for r in cell.itertuples(index=False)}
    selected_sets = []
    if show_domains:
        candidates = cell[(cell.topological_depth >= 2) & cell.credit_domain_fraction.between(0.04, 0.22)].copy()
        for target in [0.16, 0.07]:
            if len(candidates) == 0:
                break
            idx = (candidates.credit_domain_fraction - target).abs().idxmin()
            node = int(cell.loc[idx, "segment_id"])
            selected_sets.append(descendants(parent, node))
            candidates = candidates[~candidates.segment_id.isin(selected_sets[-1])]
    # Show the principal plane of the measured 3D arbor. This preserves metric
    # aspect while using the panel area more effectively than an arbitrary x--y
    # projection of this obliquely oriented cell.
    xyz = cell[["x_um", "y_um", "z_um"]].to_numpy(float)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    _, _, axes_3d = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ axes_3d[:2].T
    scale = max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
    projected /= scale
    positions = {
        int(segment_id): point
        for segment_id, point in zip(cell.segment_id.to_numpy(int), projected)
    }
    for sid, row in rows.items():
        pid = parent[sid]
        if pid not in rows:
            continue
        point, parent_point = positions[sid], positions[pid]
        color = "#C7CBD0"; lw = LW_HAIR
        for selected, sc in zip(selected_sets, [MORPH, COLORS["pathway"]]):
            if sid in selected:
                color, lw = sc, LW_DATA
        ax.plot([point[0], parent_point[0]], [point[1], parent_point[1]],
                color=color, lw=lw, solid_capstyle="round", zorder=1)
    if show_contacts:
        shown = cell[(cell.E_count > 0) | (cell.I_count > 0)]
        shown_xy = np.asarray([positions[int(value)] for value in shown.segment_id])
        sizes_e = np.clip(shown.E_count.to_numpy(float), 0, 5) * 2.0
        sizes_i = np.clip(shown.I_count.to_numpy(float), 0, 5) * 2.0
        # E and I contacts land on the same segments, so a fixed-offset pair
        # of filled discs silently occludes one class.  Instead the two
        # classes share one true position with complementary glyphs: filled
        # blue discs for E underneath open red rings for I.
        e_mask = shown.E_count.to_numpy(float) > 0
        i_mask = shown.I_count.to_numpy(float) > 0
        ax.scatter(shown_xy[e_mask, 0], shown_xy[e_mask, 1],
                   s=sizes_e[e_mask], facecolor=COLORS["exc"],
                   edgecolor="none", alpha=0.80, zorder=3)
        ax.scatter(shown_xy[i_mask, 0], shown_xy[i_mask, 1],
                   s=sizes_i[i_mask] + 6.0, facecolor="none",
                   edgecolor=COLORS["inh"], linewidth=0.5, alpha=0.80,
                   zorder=4)
    soma = cell.loc[cell.topological_depth.idxmin()]
    soma_xy = positions[int(soma.segment_id)]
    ax.scatter(soma_xy[0], soma_xy[1],
               s=28, color=COLORS["soma"], edgecolor=COLORS["edge"], linewidth=0.5, zorder=5)
    xpad = max(0.04, np.ptp(projected[:, 0]) * 0.05)
    ypad = max(0.04, np.ptp(projected[:, 1]) * 0.05)
    ax.set_xlim(projected[:, 0].min() - xpad, projected[:, 0].max() + xpad)
    ax.set_ylim(projected[:, 1].min() - ypad, projected[:, 1].max() + ypad)
    ax.set_aspect("equal"); ax.axis("off")


def routing_curves_by_cell():
    curves = pd.read_csv(DATA / "figure3" / "routing_capacity_curves.csv.gz")
    return curves.groupby(["root_id", "channels", "method"], as_index=False).agg(
        credit_capture=("credit_capture", "mean"), wiring_density=("wiring_density", "mean"))


def _figure3_detailed() -> None:
    fig, axes = ten_panel_grid(height=6.25)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i, ax_j = axes
    cell, _ = exemplar_cell()
    plot_morphology(ax_a, cell, show_contacts=True)
    panel_title(ax_a, "A", "Tree and mapped contacts")
    # The equal-aspect drawing fills the panel height, so the key gets its
    # own whitespace band left of the arbor instead of sitting on nodes.
    x_low, x_high = ax_a.get_xlim()
    ax_a.set_xlim(x_low - 0.52 * (x_high - x_low), x_high)
    clean_legend(ax_a, handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["exc"],
               markeredgecolor="none", label="excitatory"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=COLORS["inh"], markeredgewidth=0.7,
               label="inhibitory"),
    ], loc="upper left", fontsize=PT_SMALL, handlelength=0.7, handletextpad=0.2)
    ax_a.text(0.02, 0.16, "E/I contacts co-located;\nI shown as open rings",
              transform=ax_a.transAxes, ha="left", va="top",
              fontsize=PT_SMALL, color=COLORS["mute"], linespacing=1.15)

    plot_morphology(ax_b, cell, show_domains=True)
    panel_title(ax_b, "B", "Ancestry routes")
    x_low, x_high = ax_b.get_xlim()
    ax_b.set_xlim(x_low - 0.52 * (x_high - x_low), x_high)
    clean_legend(ax_b, handles=[
        Line2D([0], [0], color=MORPH, lw=LW_DATA, label="route 1"),
        Line2D([0], [0], color=COLORS["pathway"], lw=LW_DATA, label="route 2"),
    ], loc="upper left", fontsize=PT_SMALL, handlelength=1.0)

    seg = pd.read_csv(DATA / "figure3" / "segment_metrics.csv")
    seg_e = seg[seg["E_count"].gt(0)].copy()
    seg_e["depth_bin"] = pd.cut(seg_e.topological_depth, bins=[-1, 2, 4, 6, 9, np.inf],
                                labels=["0-2", "3-4", "5-6", "7-9", "10+"])
    cell_depth = seg_e.groupby(["root_id", "depth_bin"], observed=True, as_index=False).credit_domain_fraction.mean()
    groups = [g.credit_domain_fraction.to_numpy(float) for _, g in cell_depth.groupby("depth_bin", observed=True)]
    for i, arr in enumerate(groups):
        ax_c.scatter(i + jitter(arr.size, 210 + i, 0.07), arr, s=SEED_MS ** 2,
                     color=MORPH, alpha=SEED_ALPHA, edgecolor="white",
                     linewidth=0.2)
        errorbar_mean(ax_c, i, arr, MORPH, seed=220 + i)
    ax_c.set_xticks(range(len(groups))); ax_c.set_xticklabels(["0-2", "3-4", "5-6", "7-9", "10+"])
    ax_c.set_xlabel("topological depth"); ax_c.set_ylabel("domain fraction")
    panel_title(ax_c, "C", "Domains versus depth")
    style_axis(ax_c)

    cell_curves = routing_curves_by_cell()
    methods = ["dense PCA oracle", "morphology-aware paths", "random paths",
               "depth-only bins", "shuffled ancestry"]
    method_short = ["dense", "ances.", "random", "depth", "shuffle"]
    for mi, method in enumerate(methods):
        sub = cell_curves[cell_curves.method.eq(method)]
        xs, means, lows, highs = [], [], [], []
        for ch, group in sub.groupby("channels"):
            m, lo, hi = mean_ci(group.credit_capture.to_numpy(float), seed=250 + 10 * mi + int(ch))
            xs.append(ch); means.append(m); lows.append(lo); highs.append(hi)
        order = np.argsort(xs); xs = np.asarray(xs)[order]; means = np.asarray(means)[order]
        lows = np.asarray(lows)[order]; highs = np.asarray(highs)[order]
        color = METHOD_COLORS[method]
        ax_d.plot(xs, means, marker=MARKERS[mi], ms=3.5, color=color, lw=LW_DATA,
                  markeredgecolor="white", markeredgewidth=0.3,
                  label=method_short[mi])
        ax_d.fill_between(xs, lows, highs, color=color, alpha=0.10, linewidth=0)
    ax_d.set_xscale("log", base=2); ax_d.set_xticks([1, 2, 4, 8, 16])
    ax_d.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_d.set_xlabel("feedback channels"); ax_d.set_ylabel("field capture")
    ax_d.set_ylim(0, 0.90)
    panel_title(ax_d, "D", "Route capacity")
    style_axis(ax_d)
    clean_legend(ax_d, loc="upper left", ncol=2, fontsize=PT_SMALL,
                 auto_clear=True)

    focus = cell_curves[cell_curves.channels.eq(8)]
    for mi, method in enumerate(methods):
        sub = focus[focus.method.eq(method)]
        color = METHOD_COLORS[method]
        ax_e.scatter(sub.wiring_density, sub.credit_capture, s=12, color=color,
                     marker=MARKERS[mi], alpha=0.35, edgecolors="none")
        ax_e.scatter(sub.wiring_density.mean(), sub.credit_capture.mean(),
                     marker=MARKERS[mi],
                     s=30, color=color, edgecolor="white", linewidth=0.4, zorder=4)
    # Dedicated headroom band keeps the five-method key off the point cloud.
    ax_e.set_xscale("log"); ax_e.set_xlim(0.008, 1.35); ax_e.set_ylim(0.0, 1.34)
    ax_e.set_yticks([0.0, 0.5, 1.0])
    ax_e.set_xlabel("wiring density"); ax_e.set_ylabel("field capture")
    panel_title(ax_e, "E", "Wiring-capture trade-off")
    style_axis(ax_e)
    clean_legend(ax_e, handles=[
        Line2D([0], [0], marker=MARKERS[mi], color="none",
               markerfacecolor=METHOD_COLORS[method], markeredgecolor="none",
               markersize=4.2, label=method_short[mi])
        for mi, method in enumerate(methods)
    ], loc="upper left", ncol=3, fontsize=PT_SMALL, columnspacing=0.6,
        handlelength=0.9, handletextpad=0.25)

    for i, method in enumerate(methods):
        arr = focus[focus.method.eq(method)].credit_capture.to_numpy(float)
        ax_f.scatter(i + jitter(arr.size, 300 + i, 0.06), arr, s=SEED_MS ** 2,
                     color=METHOD_COLORS[method], alpha=0.68, edgecolor="white", linewidth=0.2)
        errorbar_mean(ax_f, i, arr, METHOD_COLORS[method], seed=310 + i)
    ax_f.set_xticks(range(5)); ax_f.set_xticklabels(method_short)
    ax_f.tick_params(axis="x", labelsize=PT_SMALL)
    ax_f.set_ylabel("field capture"); ax_f.set_ylim(0.05, 0.78)
    panel_title(ax_f, "F", "Eight-channel capture")
    style_axis(ax_f)

    pivot = focus.pivot(index="root_id", columns="method", values="credit_capture")
    controls = ["random paths", "depth-only bins", "shuffled ancestry"]
    # Horizontal orientation: the continuous advantage axis spans the wide
    # panel, so three categories no longer leave empty bands across it.
    for i, control in enumerate(controls):
        diff = (pivot["morphology-aware paths"] - pivot[control]).dropna().to_numpy(float)
        ax_g.scatter(diff, i + jitter(diff.size, 340 + i, 0.055), s=SEED_MS ** 2,
                     color=METHOD_COLORS[control], alpha=0.65, edgecolor="white", linewidth=0.2)
        m, lo, hi = mean_ci(diff, seed=350 + i)
        ax_g.errorbar(m, i, xerr=[[m - lo], [hi - m]], marker="D", ms=4.6,
                      color=METHOD_COLORS[control], markeredgecolor="white",
                      markeredgewidth=0.45, lw=LW_ERR, capsize=ERR_CAPSIZE, zorder=5)
    ax_g.axvline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_g.set_yticks(range(3)); ax_g.set_yticklabels(["random", "depth", "shuffle"])
    ax_g.set_ylim(2.6, -0.6)
    ax_g.set_xlabel("field-capture advantage of ancestry"); ax_g.set_xlim(-0.04, 0.42)
    panel_title(ax_g, "G", "Topology controls")
    style_axis(ax_g)

    typed = pd.read_csv(DATA / "figure3" / "typed_only_compression_curves.csv")
    typed = typed[typed.channels.eq(8)].copy()
    typed["credit_capture"] = 1.0 - typed.residual.astype(float) ** 2
    typed_methods = ["morphology-aware paths", "random paths", "depth-only bins", "shuffled ancestry"]
    for i, method in enumerate(typed_methods):
        arr = typed[typed.method.eq(method)].groupby("root_id").credit_capture.mean().to_numpy(float)
        ax_h.scatter(i + jitter(arr.size, 370 + i, 0.05), arr, s=SEED_MS ** 2,
                     color=METHOD_COLORS[method], alpha=0.60, edgecolor="white", linewidth=0.2)
        errorbar_mean(ax_h, i, arr, METHOD_COLORS[method], seed=380 + i)
    ax_h.set_xticks(range(4)); ax_h.set_xticklabels(["ances.", "random", "depth", "shuffle"])
    # Top limit clears the max per-cell value (0.867) plus marker radius so
    # no seed dot is truncated by the axes box.
    ax_h.set_ylabel("field capture"); ax_h.set_ylim(0, 0.92)
    panel_title(ax_h, "H", "Directly typed inputs")
    style_axis(ax_h)

    reciprocal = pd.read_csv(DATA / "reciprocal_routing" / "cell_method_capture.csv")
    # The surrogate-tree control takes the amber slot: COLORS["pathway"] is
    # the same violet as the dense oracle, which made the two curves
    # indistinguishable wherever they shared a panel.
    reciprocal_colors = {
        "dense SVD oracle": DENSE,
        "morphology paths": MORPH,
        "random real paths": RANDOM,
        "depth bins": DEPTH,
        "row-shuffled paths": SHUFFLE,
        "degree-depth surrogate tree": COLORS["local"],
    }
    reciprocal_labels = {
        "dense SVD oracle": "dense",
        "morphology paths": "ances.",
        "random real paths": "random",
        "depth bins": "depth",
        "row-shuffled paths": "row shuffle",
        "degree-depth surrogate tree": "matched tree",
    }
    for mi, method in enumerate(reciprocal_labels):
        part = reciprocal[reciprocal.method.eq(method)].sort_values("channels")
        grouped = part.groupby("channels").capture
        xvals = np.asarray(sorted(grouped.groups), dtype=float)
        means, lows, highs = [], [], []
        for channel in xvals:
            m, lo, hi = mean_ci(grouped.get_group(channel).to_numpy(float), seed=700 + int(channel))
            means.append(m); lows.append(lo); highs.append(hi)
        means = np.asarray(means); lows = np.asarray(lows); highs = np.asarray(highs)
        ax_i.plot(xvals, means, marker=MARKERS[mi], ms=3.2, lw=LW_DATA,
                  markeredgecolor="white", markeredgewidth=0.3,
                  color=reciprocal_colors[method], label=reciprocal_labels[method])
        ax_i.fill_between(xvals, lows, highs, color=reciprocal_colors[method], alpha=0.10, linewidth=0)
    ax_i.set_xscale("log", base=2); ax_i.set_xticks([1, 2, 4, 8, 16])
    ax_i.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_i.set_xlabel("feedback channels"); ax_i.set_ylabel("field capture")
    ax_i.set_ylim(0, 1.10)
    panel_title(ax_i, "I", "Independent reciprocal field")
    style_axis(ax_i)
    clean_legend(ax_i, loc="upper left", fontsize=PT_SMALL, ncol=3,
                 auto_clear=True)

    reciprocal_focus = reciprocal[reciprocal.channels.eq(8)]
    reciprocal_order = list(reciprocal_labels)
    for i, method in enumerate(reciprocal_order):
        arr = reciprocal_focus[reciprocal_focus.method.eq(method)].capture.to_numpy(float)
        ax_j.scatter(i + jitter(arr.size, 730 + i, 0.05), arr, s=SEED_MS ** 2,
                     color=reciprocal_colors[method], alpha=0.68,
                     edgecolor="white", linewidth=0.2)
        errorbar_mean(ax_j, i, arr, reciprocal_colors[method], seed=740 + i)
    ax_j.set_xticks(range(len(reciprocal_order)))
    ax_j.set_xticklabels(["dense", "ances.", "random", "depth", "row\nshuffle", "matched\ntree"])
    ax_j.set_ylabel("field capture")
    panel_title(ax_j, "J", "Matched topology controls")
    style_axis(ax_j)
    save(fig, "fig3_microns_topology")


def figure3() -> None:
    """Main morphology figure: anatomy, sparse capacity, independent boundary."""

    fig, gs = journal_grid(2, 3, height=4.80, wspace=0.62, hspace=0.72)
    gs.update(bottom=0.13)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = [
        fig.add_subplot(gs[row, column])
        for row in range(2)
        for column in range(3)
    ]

    cell, _ = exemplar_cell()
    plot_morphology(ax_a, cell, show_contacts=True)
    panel_title(ax_a, "A", "Reconstructed tree")
    ax_a.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["exc"], label="excitatory"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["inh"], label="inhibitory"),
        ],
        loc="lower left",
        fontsize=PT_SMALL,
        handlelength=0.7,
        handletextpad=0.2,
    )

    plot_morphology(ax_b, cell, show_domains=True)
    panel_title(ax_b, "B", "Ancestry addresses")
    ax_b.legend(
        handles=[
            Line2D([0], [0], color=MORPH, lw=LW_DATA, label="route 1"),
            Line2D([0], [0], color=COLORS["pathway"], lw=LW_DATA, label="route 2"),
        ],
        loc="lower left",
        fontsize=PT_SMALL,
        handlelength=1.0,
    )

    cell_curves = routing_curves_by_cell()
    methods = [
        "dense PCA oracle",
        "morphology-aware paths",
        "random paths",
        "depth-only bins",
        "shuffled ancestry",
    ]
    for method_index, method in enumerate(methods):
        subset = cell_curves[cell_curves.method.eq(method)]
        x_values, means, lows, highs = [], [], [], []
        for channels, group in subset.groupby("channels"):
            mean, low, high = mean_ci(
                group.credit_capture.to_numpy(float),
                seed=1200 + 10 * method_index + int(channels),
            )
            x_values.append(channels)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        order = np.argsort(x_values)
        x_values = np.asarray(x_values)[order]
        means = np.asarray(means)[order]
        lows = np.asarray(lows)[order]
        highs = np.asarray(highs)[order]
        color = METHOD_COLORS[method]
        ax_c.plot(
            x_values,
            means,
            marker="o",
            ms=3.5,
            color=color,
            lw=LW_DATA,
            label=method.replace("-aware", ""),
        )
        ax_c.fill_between(x_values, lows, highs, color=color, alpha=0.10, linewidth=0)
    ax_c.set_xscale("log", base=2)
    ax_c.set_xticks([1, 2, 4, 8, 16])
    ax_c.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_c.set_xlabel("feedback channels")
    ax_c.set_ylabel("model-field capture")
    ax_c.set_ylim(0, 0.90)
    panel_title(ax_c, "C", "Sparse route capacity")
    style_axis(ax_c)

    focus = cell_curves[cell_curves.channels.eq(8)]
    for index, method in enumerate(methods):
        values = focus[focus.method.eq(method)].credit_capture.to_numpy(float)
        color = METHOD_COLORS[method]
        ax_d.scatter(
            index + jitter(values.size, 1300 + index, 0.055),
            values,
            s=SEED_MS ** 2,
            color=color,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.2,
        )
        errorbar_mean(ax_d, index, values, color, seed=1310 + index)
    ax_d.set_xticks(range(5))
    ax_d.set_xticklabels(["dense", "ances.", "random", "depth", "shuffle"], rotation=28, ha="right")
    ax_d.set_ylabel("model-field capture")
    ax_d.set_ylim(0.05, 0.78)
    panel_title(ax_d, "D", "Model-field controls")
    style_axis(ax_d)

    reciprocal = pd.read_csv(DATA / "reciprocal_routing" / "cell_method_capture.csv")
    reciprocal_colors = {
        "dense SVD oracle": DENSE,
        "morphology paths": MORPH,
        "random real paths": RANDOM,
        "depth bins": DEPTH,
        "row-shuffled paths": SHUFFLE,
        "degree-depth surrogate tree": COLORS["pathway"],
    }
    reciprocal_labels = {
        "dense SVD oracle": "dense",
        "morphology paths": "morphology",
        "random real paths": "random",
        "depth bins": "depth",
        "row-shuffled paths": "row shuffle",
        "degree-depth surrogate tree": "matched tree",
    }
    for method, label in reciprocal_labels.items():
        part = reciprocal[reciprocal.method.eq(method)].sort_values("channels")
        grouped = part.groupby("channels").capture
        x_values = np.asarray(sorted(grouped.groups), dtype=float)
        means, lows, highs = [], [], []
        for channels in x_values:
            mean, low, high = mean_ci(
                grouped.get_group(channels).to_numpy(float), seed=1400 + int(channels)
            )
            means.append(mean)
            lows.append(low)
            highs.append(high)
        means = np.asarray(means)
        lows = np.asarray(lows)
        highs = np.asarray(highs)
        color = reciprocal_colors[method]
        ax_e.plot(x_values, means, marker="o", ms=3.2, lw=LW_DATA, color=color, label=label)
        ax_e.fill_between(x_values, lows, highs, color=color, alpha=0.10, linewidth=0)
    ax_e.set_xscale("log", base=2)
    ax_e.set_xticks([1, 2, 4, 8, 16])
    ax_e.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_e.set_xlabel("feedback channels")
    ax_e.set_ylabel("cable-response capture")
    panel_title(ax_e, "E", "Reciprocal cable field")
    style_axis(ax_e)

    reciprocal_focus = reciprocal[reciprocal.channels.eq(8)]
    for index, method in enumerate(reciprocal_labels):
        values = reciprocal_focus[reciprocal_focus.method.eq(method)].capture.to_numpy(float)
        color = reciprocal_colors[method]
        ax_f.scatter(
            index + jitter(values.size, 1500 + index, 0.05),
            values,
            s=SEED_MS ** 2,
            color=color,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.2,
        )
        errorbar_mean(ax_f, index, values, color, seed=1510 + index)
    ax_f.set_xticks(range(len(reciprocal_labels)))
    ax_f.set_xticklabels(
        ["dense", "ances.", "random", "depth", "row\nshuffle", "matched\ntree"],
    )
    ax_f.tick_params(axis="x", labelsize=PT_SMALL)
    ax_f.set_ylabel("cable-response capture")
    panel_title(ax_f, "F", "Topology controls")
    style_axis(ax_f)

    save(fig, "fig3_microns_topology")


# -------------------------------------------------------------------------
# Figure 5: focal perturbations and mechanism decomposition
# -------------------------------------------------------------------------


def focal_schematic(ax):
    """Focal-shunt intervention drawn in the shared tree vocabulary.

    The tree fills the cell vertically; each label sits in its own clear
    whitespace (green route above its branches, matched-state note in the
    empty band under the soma) with a hair leader to the focal site.
    """
    panel_title(ax, "A", "Matched focal shunt")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    nodes = draw_compact_tree(ax, 0.02, 0.44, 1.02,
                              selected={"d1", "d2", "m1"})
    focus = nodes["m1"]
    ax.add_patch(Circle(focus, 0.052, fc=COLORS["inh"], ec="white", lw=LW_EDGE, zorder=7))
    # Leader line into clear whitespace above the proximal edge.
    ax.plot([focus[0] + 0.028, focus[0] + 0.115],
            [focus[1] + 0.052, focus[1] + 0.135],
            color=COLORS["mute"], lw=LW_HAIR, zorder=6)
    ax.text(focus[0] + 0.13, focus[1] + 0.15, "focal shunt",
            color=COLORS["inh"], ha="left", va="center", fontsize=PT_SMALL)
    ax.text(0.02, 0.93, "shunted route: descendants", color=MORPH,
            ha="left", va="center", fontsize=PT_SMALL)
    soma = nodes["soma"]
    ax.plot([soma[0] - 0.004, soma[0] - 0.018],
            [soma[1] - 0.075, soma[1] - 0.245],
            color=COLORS["mute"], lw=LW_HAIR, zorder=6)
    ax.text(0.955, 0.095, "soma and output\nstate matched",
            color=COLORS["mute"], ha="right", va="center",
            fontsize=PT_SMALL, linespacing=1.15)


def _figure4_detailed() -> None:
    fig, axes = nine_panel_grid(height=6.15)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i = axes
    focal_schematic(ax_a)

    cat = pd.read_csv(DATA / "figure4" / "category_effects.csv")
    cat = cat[np.isclose(cat.dose, 1.0)]
    categories = ["descendant", "sister", "ancestor", "depth-matched unrelated", "unrelated"]
    labels = ["desc.", "sister", "ancestor", "depth ctrl.", "unrelated"]
    # Horizontal strips: the five relation labels stay horizontal on the y
    # axis, where the wide inter-column gutter gives them room.
    for perturb, color, marker, offset in [
        ("matched additive", COLORS["additive"], "s", -0.17),
        ("focal shunt", COLORS["shunting"], "o", 0.17),
    ]:
        sub = cat[cat.perturbation.eq(perturb)].groupby(["root_id", "category"], as_index=False).median_abs_log_gradient_change.mean()
        for i, category in enumerate(categories):
            arr = sub[sub.category.eq(category)].median_abs_log_gradient_change.to_numpy(float)
            ax_b.scatter(arr, i + offset + jitter(arr.size, 400 + i, 0.045),
                         s=SEED_MS ** 2, color=color, marker=marker,
                         alpha=0.65, edgecolor="white", linewidth=0.2)
            m, lo, hi = mean_ci(arr, seed=410 + i)
            ax_b.errorbar(m, i + offset, xerr=[[m - lo], [hi - m]], marker="D",
                          ms=4.6, color=color, markeredgecolor="white",
                          markeredgewidth=0.45, lw=LW_ERR, capsize=ERR_CAPSIZE,
                          zorder=5)
    ax_b.set_yticks(range(5)); ax_b.set_yticklabels(labels)
    ax_b.tick_params(axis="y", labelsize=PT_SMALL)
    ax_b.set_ylim(4.6, -0.6)
    ax_b.set_xlabel(r"median $|\Delta\log |\nabla||$")
    panel_title(ax_b, "B", "Change by tree relation")
    style_axis(ax_b)
    clean_legend(
        ax_b, handles=[
            Line2D([0], [0], color=COLORS["additive"], marker="s",
                   linestyle="none", label="matched additive"),
            Line2D([0], [0], color=COLORS["shunting"], marker="o",
                   linestyle="none", label="focal shunt"),
        ], loc="lower right", fontsize=PT_SMALL, auto_clear=True,
    )

    primary = pd.read_csv(DATA / "figure4" / "cell_primary_contrasts.csv")
    cols = ["matched_additive_localization", "shunt_depth_shuffled_localization", "focal_shunt_localization"]
    xs = np.arange(3)
    for _, row in primary.iterrows():
        ax_c.plot(xs, row[cols].to_numpy(float), color=COLORS["mute"], lw=LW_HAIR, alpha=0.48)
    colors = [COLORS["additive"], SHUFFLE, COLORS["shunting"]]
    for i, col in enumerate(cols):
        arr = primary[col].to_numpy(float)
        ax_c.scatter(i + jitter(arr.size, 430 + i, 0.045), arr, s=SEED_MS ** 2,
                     color=colors[i], alpha=0.75, edgecolor="white", linewidth=0.2)
        errorbar_mean(ax_c, i, arr, colors[i], seed=440 + i)
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_c.set_xticks(xs); ax_c.set_xticklabels(["additive", "reassigned", "shunt"])
    ax_c.tick_params(axis="x", labelsize=PT_SMALL)
    ax_c.set_ylabel("localization index")
    panel_title(ax_c, "C", "Within-cell controls")
    style_axis(ax_c)

    focal = pd.read_csv(DATA / "figure4" / "focal_localization.csv")
    per_cell = focal.groupby(["root_id", "dose", "perturbation"], as_index=False).localization_index.mean()
    for pi, (perturb, color, marker) in enumerate([
        ("matched additive", COLORS["additive"], "s"),
        ("focal shunt", COLORS["shunting"], "o"),
    ]):
        sub = per_cell[per_cell.perturbation.eq(perturb)]
        xvals, means, lows, highs = [], [], [], []
        for dose, group in sub.groupby("dose"):
            m, lo, hi = mean_ci(group.localization_index.to_numpy(float), seed=460 + 10 * pi + int(dose * 4))
            xvals.append(dose); means.append(m); lows.append(lo); highs.append(hi)
        order = np.argsort(xvals); xvals = np.asarray(xvals)[order]; means = np.asarray(means)[order]
        lows = np.asarray(lows)[order]; highs = np.asarray(highs)[order]
        ax_d.plot(xvals, means, marker=marker, ms=3.5, lw=LW_DATA, color=color,
                  markeredgecolor="white", markeredgewidth=0.3, label=perturb)
        ax_d.fill_between(xvals, lows, highs, color=color, alpha=0.12, linewidth=0)
    ax_d.set_xscale("log", base=2); ax_d.set_xticks([0.25, 0.5, 1, 2])
    ax_d.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_d.set_xlabel("perturbation dose"); ax_d.set_ylabel("localization index")
    panel_title(ax_d, "D", "Dose response")
    style_axis(ax_d); clean_legend(ax_d, loc="upper left", fontsize=PT_SMALL)

    shunt_sites = focal[focal.perturbation.eq("focal shunt")]
    depth_cell = shunt_sites.groupby(["root_id", "focal_topological_depth"], as_index=False).localization_index.mean()
    ax_e.scatter(depth_cell.focal_topological_depth, depth_cell.localization_index,
                 s=10, color=MORPH, alpha=0.35, edgecolors="none")
    bins = sorted(depth_cell.focal_topological_depth.unique())
    means = [depth_cell[depth_cell.focal_topological_depth.eq(b)].localization_index.mean() for b in bins]
    ax_e.plot(bins, means, color=MORPH, marker="o", ms=3.2, lw=LW_DATA)
    ax_e.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_e.set_xlabel("focal topological depth"); ax_e.set_ylabel("localization index")
    panel_title(ax_e, "E", "Effect across focal depth")
    style_axis(ax_e)

    specs = [
        ("scale0p1_summary.json", "scale 0.10"),
        ("summary.json", "scale 0.35"),
        ("scale1p0_summary.json", "scale 1.00"),
        ("irevm0p5_summary.json", r"$E_I=-0.5$"),
        ("irev0_summary.json", r"$E_I=0.0$"),
    ]
    for i, (fname, label) in enumerate(specs):
        obj = json.loads((DATA / "figure4" / fname).read_text())
        pc = obj["primary_contrast"]
        m = pc["mean_shunt_minus_additive"]; lo, hi = pc["cell_bootstrap_ci95"]
        ax_f.errorbar(m, i, xerr=[[m - lo], [hi - m]], marker="o",
                      color=COLORS["shunting"], ms=4.2, lw=LW_ERR, capsize=ERR_CAPSIZE)
    ax_f.axvline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_f.set_yticks(range(len(specs)))
    ax_f.set_yticklabels([label for _, label in specs])
    ax_f.tick_params(axis="y", labelsize=PT_SMALL)
    ax_f.invert_yaxis(); ax_f.set_xlabel("localization difference")
    panel_title(ax_f, "F", "Parameter sweep")
    style_axis(ax_f)

    direct = pd.read_csv(DATA / "figure4" / "direct_typed_cell_primary_contrasts.csv")
    contrasts = [(primary.shunt_minus_additive.to_numpy(float), "all mapped", MORPH),
                 (direct.shunt_minus_additive.to_numpy(float), "direct typed", COLORS["pathway"])]
    for i, (arr, label, color) in enumerate(contrasts):
        ax_g.scatter(i + jitter(arr.size, 500 + i, 0.045), arr, s=SEED_MS ** 2,
                     color=color, alpha=0.70, edgecolor="white", linewidth=0.2)
        errorbar_mean(ax_g, i, arr, color, seed=510 + i)
    ax_g.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_g.set_xticks([0, 1]); ax_g.set_xticklabels(["all mapped", "direct typed"])
    ax_g.set_ylabel("localization difference")
    panel_title(ax_g, "G", "Direct synapse-type control")
    style_axis(ax_g)

    factor_states = pd.read_csv(DATA / "focal_decomposition" / "cell_shapley.csv")
    factor_states = factor_states[
        factor_states.estimand.eq("full_shunt_minus_matched_additive")
    ].sort_values("root_id")
    order = ["driving_force_only_localization", "full_shunt_localization"]
    labels = ["driving force\nonly", "full shunt"]
    colors = [COLORS["local"], COLORS["shunting"]]
    for _, row in factor_states.iterrows():
        ax_h.plot(range(2), row[order].to_numpy(float), color=COLORS["mute"], lw=LW_HAIR, alpha=0.45)
    for i, col in enumerate(order):
        arr = factor_states[col].to_numpy(float)
        ax_h.scatter(i + jitter(arr.size, 530 + i, 0.04), arr, s=SEED_MS ** 2,
                     color=colors[i], alpha=0.68, edgecolor="white", linewidth=0.2)
        errorbar_mean(ax_h, i, arr, colors[i], seed=540 + i)
    ax_h.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_h.set_xticks(range(2)); ax_h.set_xticklabels(labels)
    ax_h.set_ylabel("localization index")
    panel_title(ax_h, "H", "Exact transport contrast")
    style_axis(ax_h)

    physical = pd.read_csv(DATA / "physical_cable_sensitivity" / "cell_primary_contrasts.csv")
    ratio = pd.read_csv(DATA / "physical_cable_sensitivity" / "cell_electrotonic_ratios.csv")
    ratio_mean = ratio.groupby(["cohort", "regime"], as_index=False).median_axial_to_leak_ratio.median()
    physical = physical.merge(ratio_mean, on=["cohort", "regime"], validate="many_to_one")
    for cohort, label, color, marker in [
        ("original_eight", "original 8", COLORS["shunting"], "o"),
        ("v661_disjoint", "v661 45", COLORS["pathway"], "s"),
    ]:
        part = physical[physical.cohort.eq(cohort)].copy()
        if cohort == "original_eight":
            part = part[part.regime.str.startswith("Ra150_")]
        points = []
        for (regime, xvalue), group in part.groupby(["regime", "median_axial_to_leak_ratio"]):
            m, lo, hi = mean_ci(group.difference.to_numpy(float), seed=800 + len(points))
            points.append((float(xvalue), m, lo, hi))
        points.sort()
        xvals = np.asarray([item[0] for item in points]); means = np.asarray([item[1] for item in points])
        lows = np.asarray([item[2] for item in points]); highs = np.asarray([item[3] for item in points])
        ax_i.errorbar(xvals, means, yerr=[means - lows, highs - means], marker=marker,
                      ms=3.7, lw=LW_DATA, capsize=ERR_CAPSIZE, color=color, label=label)
    ax_i.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_i.set_xscale("log")
    ax_i.set_xlabel("median axial / leak")
    ax_i.set_ylabel("localization difference")
    panel_title(ax_i, "I", "Cable calibration")
    style_axis(ax_i); clean_legend(ax_i, loc="upper right", fontsize=PT_SMALL)
    save(fig, "fig4_focal_shunting")


def figure4() -> None:
    """Main focal-shunting figure: intervention, mechanism, and physical boundary."""

    fig, gs = journal_grid(2, 3, height=4.80, wspace=0.64, hspace=0.72)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = [
        fig.add_subplot(gs[row, column])
        for row in range(2)
        for column in range(3)
    ]
    focal_schematic(ax_a)

    category = pd.read_csv(DATA / "figure4" / "category_effects.csv")
    category = category[np.isclose(category.dose, 1.0)]
    categories = ["descendant", "sister", "ancestor", "depth-matched unrelated", "unrelated"]
    labels = ["desc.", "sister", "ancestor", "depth ctrl.", "unrelated"]
    for perturbation, color, offset in [
        ("matched additive", COLORS["additive"], -0.10),
        ("focal shunt", COLORS["shunting"], 0.10),
    ]:
        subset = category[category.perturbation.eq(perturbation)].groupby(
            ["root_id", "category"], as_index=False
        ).median_abs_log_gradient_change.mean()
        for index, relation in enumerate(categories):
            values = subset[
                subset.category.eq(relation)
            ].median_abs_log_gradient_change.to_numpy(float)
            ax_b.scatter(
                index + offset + jitter(values.size, 1600 + index, 0.025),
                values,
                s=SEED_MS ** 2,
                color=color,
                alpha=0.65,
                edgecolor="white",
                linewidth=0.2,
            )
            errorbar_mean(ax_b, index + offset, values, color, seed=1610 + index)
    ax_b.set_xticks(range(5))
    ax_b.set_xticklabels(labels, rotation=30, ha="right")
    ax_b.set_ylabel(r"median $|\Delta\log |\nabla||$")
    panel_title(ax_b, "B", "Tree-relation selectivity")
    style_axis(ax_b)

    primary = pd.read_csv(DATA / "figure4" / "cell_primary_contrasts.csv")
    columns = [
        "matched_additive_localization",
        "shunt_depth_shuffled_localization",
        "focal_shunt_localization",
    ]
    colors = [COLORS["additive"], SHUFFLE, COLORS["shunting"]]
    x_values = np.arange(3)
    for _, row in primary.iterrows():
        ax_c.plot(x_values, row[columns].to_numpy(float), color=COLORS["mute"], lw=LW_HAIR, alpha=0.48)
    for index, column in enumerate(columns):
        values = primary[column].to_numpy(float)
        ax_c.scatter(
            index + jitter(values.size, 1630 + index, 0.045),
            values,
            s=SEED_MS ** 2,
            color=colors[index],
            alpha=0.75,
            edgecolor="white",
            linewidth=0.2,
        )
        errorbar_mean(ax_c, index, values, colors[index], seed=1640 + index)
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_c.set_xticks(x_values)
    ax_c.set_xticklabels(["additive", "reassigned", "shunt"], rotation=24, ha="right")
    ax_c.set_ylabel("localization index")
    panel_title(ax_c, "C", "Within-cell controls")
    style_axis(ax_c)

    focal = pd.read_csv(DATA / "figure4" / "focal_localization.csv")
    per_cell = focal.groupby(["root_id", "dose", "perturbation"], as_index=False).localization_index.mean()
    for perturbation_index, (perturbation, color) in enumerate(
        [("matched additive", COLORS["additive"]), ("focal shunt", COLORS["shunting"])]
    ):
        subset = per_cell[per_cell.perturbation.eq(perturbation)]
        x_values, means, lows, highs = [], [], [], []
        for dose, group in subset.groupby("dose"):
            mean, low, high = mean_ci(
                group.localization_index.to_numpy(float),
                seed=1660 + 10 * perturbation_index + int(dose * 4),
            )
            x_values.append(dose)
            means.append(mean)
            lows.append(low)
            highs.append(high)
        order = np.argsort(x_values)
        x_values = np.asarray(x_values)[order]
        means = np.asarray(means)[order]
        lows = np.asarray(lows)[order]
        highs = np.asarray(highs)[order]
        ax_d.plot(x_values, means, marker="o", ms=3.5, lw=LW_DATA, color=color, label=perturbation)
        ax_d.fill_between(x_values, lows, highs, color=color, alpha=0.12, linewidth=0)
    ax_d.set_xscale("log", base=2)
    ax_d.set_xticks([0.25, 0.5, 1, 2])
    ax_d.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_d.set_xlabel("perturbation dose")
    ax_d.set_ylabel("localization index")
    panel_title(ax_d, "D", "Dose response")
    style_axis(ax_d)
    clean_legend(ax_d, loc="upper left", fontsize=PT_SMALL)

    factor_states = pd.read_csv(DATA / "focal_decomposition" / "cell_shapley.csv")
    factor_states = factor_states[
        factor_states.estimand.eq("full_shunt_minus_matched_additive")
    ].sort_values("root_id")
    state_columns = ["driving_force_only_localization", "full_shunt_localization"]
    state_labels = ["driving force\nonly", "full shunt"]
    state_colors = [COLORS["local"], COLORS["shunting"]]
    for _, row in factor_states.iterrows():
        ax_e.plot(range(2), row[state_columns].to_numpy(float), color=COLORS["mute"], lw=LW_HAIR, alpha=0.45)
    for index, column in enumerate(state_columns):
        values = factor_states[column].to_numpy(float)
        ax_e.scatter(
            index + jitter(values.size, 1700 + index, 0.04),
            values,
            s=SEED_MS ** 2,
            color=state_colors[index],
            alpha=0.68,
            edgecolor="white",
            linewidth=0.2,
        )
        errorbar_mean(ax_e, index, values, state_colors[index], seed=1710 + index)
    ax_e.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_e.set_xticks(range(2))
    ax_e.set_xticklabels(state_labels)
    ax_e.set_ylabel("localization index")
    panel_title(ax_e, "E", "Adjoint transport")
    style_axis(ax_e)

    physical = pd.read_csv(DATA / "physical_cable_sensitivity" / "cell_primary_contrasts.csv")
    ratio = pd.read_csv(DATA / "physical_cable_sensitivity" / "cell_electrotonic_ratios.csv")
    ratio_mean = ratio.groupby(["cohort", "regime"], as_index=False).median_axial_to_leak_ratio.median()
    physical = physical.merge(ratio_mean, on=["cohort", "regime"], validate="many_to_one")
    for cohort, label, color, marker in [
        ("original_eight", "original 8", COLORS["shunting"], "o"),
        ("v661_disjoint", "v661 45", COLORS["pathway"], "s"),
    ]:
        subset = physical[physical.cohort.eq(cohort)].copy()
        if cohort == "original_eight":
            subset = subset[subset.regime.str.startswith("Ra150_")]
        points = []
        for (_, x_value), group in subset.groupby(["regime", "median_axial_to_leak_ratio"]):
            mean, low, high = mean_ci(group.difference.to_numpy(float), seed=1740 + len(points))
            points.append((float(x_value), mean, low, high))
        points.sort()
        x_values = np.asarray([item[0] for item in points])
        means = np.asarray([item[1] for item in points])
        lows = np.asarray([item[2] for item in points])
        highs = np.asarray([item[3] for item in points])
        ax_f.errorbar(
            x_values,
            means,
            yerr=[means - lows, highs - means],
            marker=marker,
            ms=3.7,
            lw=LW_DATA,
            capsize=ERR_CAPSIZE,
            color=color,
            label=label,
        )
    ax_f.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_f.set_xscale("log")
    ax_f.set_xlabel("median axial / leak")
    ax_f.set_ylabel("shunt - additive localization")
    panel_title(ax_f, "F", "Electrotonic limit")
    style_axis(ax_f)
    clean_legend(ax_f, loc="upper right", fontsize=PT_SMALL)

    save(fig, "fig4_focal_shunting")


# -------------------------------------------------------------------------
# Figure 7 in the manuscript: measured-response boundary
# -------------------------------------------------------------------------


def _figure5_detailed() -> None:
    fig, axes = eight_panel_grid(height=6.35)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h = axes
    functional = pd.read_csv(DATA / "figure5" / "functional_target_metrics.csv")
    y = np.arange(len(functional))
    ax_a.barh(y, functional.n_partners, color=MORPH, alpha=0.82)
    ax_a.set_yticks(y); ax_a.set_yticklabels([f"target {i + 1}" for i in y])
    ax_a.invert_yaxis(); ax_a.set_xlabel("presynaptic partners")
    panel_title(ax_a, "A", "Visual-response cohort")
    style_axis(ax_a, grid="x")

    vals = functional.partial_shared_path_r.to_numpy(float)
    ax_b.scatter(vals, y, s=SEED_MS ** 2 * 1.4, color=MORPH, alpha=0.78,
                 edgecolor="white", linewidth=0.25)
    ax_b.axvline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_b.set_yticks(y); ax_b.set_yticklabels([str(i + 1) for i in y])
    ax_b.set_ylabel("target cell")
    ax_b.invert_yaxis()
    ax_b.set_xlabel("partial shared-ancestry effect")
    panel_title(ax_b, "B", "Structure-function relation")
    style_axis(ax_b)

    metrics = functional[["shared_path_r", "negative_tree_distance_r", "partial_shared_path_r",
                          "same_major_branch_delta"]].copy()
    names = ["path", "dist.", "partial", "branch"]
    im = ax_c.imshow(metrics.to_numpy(float).T, aspect="auto", cmap=DIV_CMAP, vmin=-0.6, vmax=0.6)
    ax_c.set_xticks(range(len(functional))); ax_c.set_xticklabels([str(i + 1) for i in range(len(functional))])
    ax_c.set_yticks(range(4)); ax_c.set_yticklabels(names)
    ax_c.set_xlabel("target cell")
    panel_title(ax_c, "C", "Topology metrics")
    for spine in ax_c.spines.values(): spine.set_visible(False)
    # A slim colorbar defines the signed scale; the panel cedes a little
    # width so the bar and its tick labels stay inside the canvas.
    box_c = ax_c.get_position()
    ax_c.set_position([box_c.x0, box_c.y0, box_c.width * 0.70, box_c.height])
    add_colorbar(fig, ax_c, im, label=r"effect size ($r$ or $\Delta$)", width=0.05)

    target = pd.read_csv(DATA / "figure5" / "task_target_method_means_ch4.csv")
    methods = ["exact backprop", "dense PCA oracle", "morphology-aware paths",
               "random nonempty paths", "depth-only bins", "shuffled ancestry", "scalar broadcast"]
    short = ["exact", "dense", "ances.", "random", "depth", "shuffle", "scalar"]
    # Horizontal strips keep every category label horizontal: methods run
    # down the y axis and the metric spans the x axis.
    for ax, metric, xlabel, title, xlim in [
        (ax_d, "heldout_credit_capture", "held-out field capture", "Task-field capacity", (0, 1.06)),
        (ax_e, "heldout_normalized_mse", "held-out normalized MSE", "Learning outcome", (0.60, 1.01)),
    ]:
        for i, method in enumerate(methods):
            arr = target[target.method.eq(method)][metric].to_numpy(float)
            color = METHOD_COLORS[method]
            ax.scatter(arr, i + jitter(arr.size, 600 + i, 0.09), s=SEED_MS ** 2,
                       color=color, alpha=0.65, edgecolor="white", linewidth=0.2)
            m, lo, hi = mean_ci(arr, seed=610 + i)
            ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], marker="D", ms=4.6,
                        color=color, markeredgecolor="white", markeredgewidth=0.45,
                        lw=LW_ERR, capsize=ERR_CAPSIZE, zorder=5)
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(short)
        ax.set_ylim(len(methods) - 0.4, -0.6)
        ax.set_xlabel(xlabel); ax.set_xlim(*xlim)
        panel_title(ax, "D" if ax is ax_d else "E", title)
        style_axis(ax)
    # Honest coincidence: the exact-backprop captures are identically 1.00
    # and the dense-oracle captures span 0.996-1.00, so the per-target dots
    # sit beneath the mean diamond rather than being spread out.
    ax_d.text(0.955, 0, "all targets $=$ 1.00", ha="right", va="center",
              fontsize=PT_SMALL, color=COLORS["mute"])
    ax_d.text(0.955, 1, r"targets $\geq$ 0.996", ha="right", va="center",
              fontsize=PT_SMALL, color=COLORS["mute"])

    paired = target.pivot(index="target_root_id", columns="method", values=["heldout_credit_capture", "heldout_normalized_mse"])
    xdiff = paired["heldout_credit_capture"]["morphology-aware paths"] - paired["heldout_credit_capture"]["shuffled ancestry"]
    ydiff = paired["heldout_normalized_mse"]["shuffled ancestry"] - paired["heldout_normalized_mse"]["morphology-aware paths"]
    ax_f.scatter(xdiff, ydiff, s=24, color=MORPH, alpha=0.78, edgecolor="white", linewidth=0.35)
    for i, (xv, yv) in enumerate(zip(xdiff, ydiff), start=1):
        ax_f.annotate(str(i), (xv, yv), xytext=(3, 2), textcoords="offset points", fontsize=PT_SMALL)
    ax_f.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax_f.axvline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax_f.set_xlabel(r"capture: ancestry $-$ shuffle"); ax_f.set_ylabel("MSE gain")
    panel_title(ax_f, "F", "Target dependence")
    style_axis(ax_f)

    structural = target[~target.method.isin(["exact backprop", "dense PCA oracle"])].copy()
    structural["capture_centered"] = structural.heldout_credit_capture - structural.groupby("target_root_id").heldout_credit_capture.transform("mean")
    structural["mse_centered"] = structural.heldout_normalized_mse - structural.groupby("target_root_id").heldout_normalized_mse.transform("mean")
    structural_short = dict(zip(methods, short))
    structural_markers = dict(zip(methods, MARKERS))
    for method, group in structural.groupby("method"):
        ax_g.scatter(group.capture_centered, group.mse_centered, s=14,
                     color=METHOD_COLORS[method], alpha=0.65,
                     marker=structural_markers[method],
                     label=structural_short[method])
    ax_g.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_g.axvline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_g.set_xlabel("within-target centered capture"); ax_g.set_ylabel("centered MSE")
    panel_title(ax_g, "G", "Capture versus utility")
    style_axis(ax_g)
    add_headroom(ax_g, 0.22)
    clean_legend(ax_g, loc="upper left", ncol=3, fontsize=PT_SMALL,
                 auto_clear=True)

    channels = [1, 2, 4, 8]
    cap_m, cap_lo, cap_hi, learn_m, learn_lo, learn_hi = [], [], [], [], [], []
    for ch in channels:
        obj = json.loads((DATA / "figure5" / f"task_summary_ch{ch}.json").read_text())
        cap = obj["primary_credit_capture_contrast"]
        learn = obj["primary_learning_contrast"]
        cap_m.append(cap["mean_difference"]); cap_lo.append(cap["target_bootstrap_ci95"][0]); cap_hi.append(cap["target_bootstrap_ci95"][1])
        learn_m.append(-learn["mean_difference"]); learn_lo.append(-learn["target_bootstrap_ci95"][1]); learn_hi.append(-learn["target_bootstrap_ci95"][0])
    ax_h.errorbar(np.asarray(channels) - 0.08, cap_m,
                  yerr=[np.asarray(cap_m) - cap_lo, np.asarray(cap_hi) - cap_m],
                  marker="o", color=MORPH, lw=LW_DATA, capsize=ERR_CAPSIZE, label="capture")
    ax_h.errorbar(np.asarray(channels) + 0.08, learn_m,
                  yerr=[np.asarray(learn_m) - learn_lo, np.asarray(learn_hi) - learn_m],
                  marker="s", color=SHUFFLE, lw=LW_DATA, capsize=ERR_CAPSIZE, label="MSE improvement")
    ax_h.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax_h.set_xscale("log", base=2); ax_h.set_xticks(channels)
    ax_h.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_h.set_xlabel("feedback channels"); ax_h.set_ylabel("morphology advantage")
    panel_title(ax_h, "H", "Across feedback bandwidth")
    style_axis(ax_h)
    # One-row legend above the tallest error bar: headroom keeps the key
    # clear of the wide bootstrap intervals.
    add_headroom(ax_h, 0.30)
    clean_legend(ax_h, loc="upper right", ncol=2, fontsize=PT_SMALL)
    save(fig, "fig5_alignment_boundary")


def figure5() -> None:
    """Main alignment figure: measured boundary plus controlled sufficiency test."""

    fig, gs = journal_grid(2, 3, height=4.85, wspace=0.64, hspace=0.72)
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = [
        fig.add_subplot(gs[row, column])
        for row in range(2)
        for column in range(3)
    ]

    functional = pd.read_csv(DATA / "figure5" / "functional_target_metrics.csv")
    y_values = np.arange(len(functional))
    ax_a.barh(y_values, functional.n_partners, color=MORPH, alpha=0.82)
    ax_a.set_yticks(y_values)
    ax_a.set_yticklabels([f"target {index + 1}" for index in y_values])
    ax_a.invert_yaxis()
    ax_a.set_xlabel("presynaptic partners")
    panel_title(ax_a, "A", "Measured cohort")
    style_axis(ax_a, grid="x")

    values = functional.partial_shared_path_r.to_numpy(float)
    ax_b.scatter(
        values,
        y_values,
        s=SEED_MS ** 2 * 1.4,
        color=MORPH,
        alpha=0.78,
        edgecolor="white",
        linewidth=0.25,
    )
    mean, low, high = mean_ci(values, seed=1800)
    ax_b.errorbar(
        mean,
        len(y_values) + 0.10,
        xerr=[[mean - low], [high - mean]],
        color=COLORS["ink"],
        marker="D",
        markerfacecolor="white",
        ms=4.2,
        capsize=ERR_CAPSIZE,
        lw=LW_ERR,
    )
    ax_b.axvline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_b.set_yticks(y_values)
    ax_b.set_yticklabels([])
    ax_b.invert_yaxis()
    ax_b.set_xlabel("partial shared-ancestry effect")
    panel_title(ax_b, "B", "Structure-function boundary")
    style_axis(ax_b)

    target = pd.read_csv(DATA / "figure5" / "task_target_method_means_ch4.csv")
    methods = [
        "exact backprop",
        "dense PCA oracle",
        "morphology-aware paths",
        "random nonempty paths",
        "depth-only bins",
        "shuffled ancestry",
        "scalar broadcast",
    ]
    short_labels = ["exact", "dense", "ances.", "random", "depth", "shuffle", "scalar"]
    for ax, metric, ylabel, letter, title, limits in [
        (ax_c, "heldout_credit_capture", "held-out field capture", "C", "Task-field capture", (0, 1.06)),
        (ax_d, "heldout_normalized_mse", "held-out normalized MSE", "D", "Held-out learning", (0.62, 1.01)),
    ]:
        for index, method in enumerate(methods):
            metric_values = target[target.method.eq(method)][metric].to_numpy(float)
            color = METHOD_COLORS[method]
            ax.scatter(
                index + jitter(metric_values.size, 1820 + index, 0.045),
                metric_values,
                s=SEED_MS ** 2,
                color=color,
                alpha=0.65,
                edgecolor="white",
                linewidth=0.2,
            )
            errorbar_mean(ax, index, metric_values, color, seed=1830 + index)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(short_labels, rotation=32, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*limits)
        panel_title(ax, letter, title)
        style_axis(ax)

    controlled = pd.read_csv(DATA / "alignment_controlled" / "alignment_controlled_curves.csv")
    controlled_colors = {
        "morphology-selected paths": MORPH,
        "random paths": RANDOM,
        "depth bins": DEPTH,
        "ancestry-shuffled paths": SHUFFLE,
    }
    controlled_labels = {
        "morphology-selected paths": "morphology",
        "random paths": "random",
        "depth bins": "depth",
        "ancestry-shuffled paths": "shuffle",
    }
    for method, label in controlled_labels.items():
        part = controlled[controlled.method.eq(method)].sort_values("alignment")
        ax_e.plot(
            part.alignment,
            part.credit_capture,
            color=controlled_colors[method],
            marker="o",
            ms=3.5,
            lw=LW_DATA,
            label=label,
        )
        ax_e.fill_between(
            part.alignment,
            part.credit_capture_ci_low,
            part.credit_capture_ci_high,
            color=controlled_colors[method],
            alpha=0.10,
            linewidth=0,
        )
    ax_e.set_xlabel("imposed morphology alignment")
    ax_e.set_ylabel("field capture")
    ax_e.set_xlim(-0.02, 1.02)
    ax_e.set_ylim(-0.03, 1.04)
    panel_title(ax_e, "E", "Controlled alignment")
    style_axis(ax_e)
    clean_legend(ax_e, fontsize=PT_SMALL, loc="upper left")

    cell_metrics = pd.read_csv(DATA / "alignment_controlled" / "cell_alignment_metrics.csv")
    for method, label in controlled_labels.items():
        part = cell_metrics[cell_metrics.method.eq(method)]
        ax_f.scatter(
            part.credit_capture,
            part.one_step_progress,
            s=8,
            color=controlled_colors[method],
            alpha=0.28,
            edgecolors="none",
            label=label,
            rasterized=True,
        )
    summary = json.loads((DATA / "alignment_controlled" / "summary.json").read_text())
    rho = summary["capture_progress_relation_non_morphology"]["median_within_cell_spearman_r"]
    ax_f.text(
        0.04,
        0.94,
        rf"median within-cell $\rho={rho:.2f}$",
        transform=ax_f.transAxes,
        va="top",
        fontsize=PT_ANNOT,
        color=COLORS["ink"],
    )
    ax_f.set_xlabel("field capture")
    ax_f.set_ylabel("one-step progress")
    panel_title(ax_f, "F", "Capture vs progress")
    style_axis(ax_f)

    save(fig, "fig5_alignment_boundary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figures", nargs="*", type=int, default=[1, 2, 3, 4, 5],
                        choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    # The journal keeps the comprehensive, full-panel figures. Compact
    # alternates are retained only as possible presentation assets.
    builders = {
        1: figure1,
        2: figure2,
        3: _figure3_detailed,
        4: _figure4_detailed,
        5: _figure5_detailed,
    }
    for number in args.figures:
        builders[number]()


if __name__ == "__main__":
    main()

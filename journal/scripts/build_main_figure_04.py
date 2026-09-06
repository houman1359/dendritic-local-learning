#!/usr/bin/env python3
"""Final Main Figure 4 -- branch-specific credit under local conflict.

This dedicated figure separates the continuously controlled branch-conflict
experiment from the hierarchical subtree-routing experiment.  It uses the
same source tables and journal-wide vector tokens as the earlier combined
Figure 3, but gives the task, its analytic boundary and its implementation
controls enough room to be understood without relying on the caption.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import routing_figure_panels
from routing_figure_panels import (
    B_STYLE,
    PATH_NECESSITY,
    path_accuracy_facets,
    shared_mode_boundary,
)
from credit_tree_schematics import mix
from figure_canvas import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_ANNOT,
    PT_SMALL,
    SEED_ALPHA,
    SEED_MS,
    Margins,
    NativeCanvas,
)
from native_schematics import Frame
from routing_figure_panels import draw_conflict_neuron, draw_credit_fan


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "figures" / "components" / "main_figure_04_native.pdf"
# The published figures/main copy is emitted ONLY by
# assemble_compact_main_figures.py, whose FIGURE_SOURCES map assigns
# this component its publication number; a builder-side copy would
# bypass the 2026-09-04 dictionary-forward renumbering.

CANVAS_H_PT = 450.0
HEIGHT_IN = CANVAS_H_PT / 72.0
ROW_PT = [137.0, 120.0, 111.0]
HGUTTER_PT = 34.0
VGUTTER_PT = 44.0

INK = COLORS["ink"]
MUTE = COLORS["mute"]
GREEN = COLORS["shunting"]
AMBER = COLORS["local"]
PURPLE = COLORS["oracle"]
GRAY = COLORS["point_mlp"]


def _sub_token(ax, x, y, base, sub, tail="", *, size=PT_ANNOT,
               sub_size=PT_SMALL, color=None, ha="left", va="center",
               drop_pt=1.8, zorder=5, clip_on=True, transform=None):
    """``token_subscript`` with the tail restored to the base baseline.

    The shared helper chains every span on the *previous* span's bounding
    box, so its tail inherits the subscript's drop and the base glyph reads
    as a superscript.  Here the subscript aligns its box bottom to the
    base's box bottom before dropping, and the tail takes its x from the
    subscript but its y from the BASE box, so only the subscript leaves the
    line -- the same corrected baseline logic as the boundary-value tags.
    Signature-compatible with ``token_subscript`` so it can stand in for it
    while this figure's panels draw.

    A subscript should also be visibly smaller than its base, but the token
    set is the only legal size vocabulary and PT_SMALL is its floor, so a
    sub below a PT_SMALL base cannot exist (``enforce_tokens`` would snap it
    back up).  When a caller asks for a subscript at or above its base size,
    step the base and tail up one token (PT_SMALL -> PT_ANNOT) instead: the
    same base-over-sub differential, entirely inside the token set, and one
    consistent treatment for panel A's x-stream labels, B's equations and
    D's boundary tags.
    """
    if sub_size >= size:
        if size <= PT_SMALL:
            size = PT_ANNOT
        sub_size = PT_SMALL
    color = COLORS["ink"] if color is None else color
    kwargs = {"transform": transform} if transform is not None else {}
    base_text = ax.text(x, y, base, fontsize=size, color=color, ha=ha,
                        va=va, zorder=zorder, clip_on=clip_on, **kwargs)
    sub_text = ax.annotate(sub, xy=(1.0, 0.0), xycoords=base_text,
                           xytext=(0.4, -drop_pt), textcoords="offset points",
                           fontsize=sub_size, color=color, ha="left",
                           va="bottom", zorder=zorder,
                           annotation_clip=clip_on)
    if tail:
        ax.annotate(tail, xy=(1.0, 0.0), xycoords=(sub_text, base_text),
                    xytext=(0.6, 0.0), textcoords="offset points",
                    fontsize=size, color=color, ha="left", va="bottom",
                    zorder=zorder, annotation_clip=clip_on)
    return base_text


def _box(frame: Frame, center, width, height, *, face, edge, radius=3.0):
    """Rounded box whose corner radius is specified in physical points."""
    cx, cy = center
    patch = FancyBboxPatch(
        (cx - width / 2.0, cy - height / 2.0), width, height,
        boxstyle=f"round,pad=0.002,rounding_size={frame.fy(radius)}",
        facecolor=face, edgecolor=edge, linewidth=LW_EDGE,
        transform=frame.ax.transData, clip_on=False, zorder=3,
    )
    frame.ax.add_patch(patch)
    return patch


def _trial_card(frame: Frame, rect, *, conflict: bool) -> None:
    """One compatible or conflicting trial, drawn as a dendritic unit."""
    x0, y0, width, height = rect
    # Tint and border strength follow the manuscript's emphasis card (Fig. 5's
    # "matched subtrees": fill #F3F9F6, edge #A9D5BD).  At mix 52 these two
    # borders were the most saturated card edges in the figure set.
    # White cards with the figure set's hairline border: the colored washes
    # this card used to carry were the most saturated fills in the set, and
    # they muted the strokes of the very glyph they framed.
    frame.group(rect, tint="white", edge=COLORS["grid"])
    frame.text(
        (x0 + frame.fx(6.0), y0 + height - frame.fy(5.0)),
        "compatible: χ = 0" if not conflict else "conflicting: χ = 1",
        size=PT_ANNOT, color=GREEN if not conflict else COLORS["highlight"],
        ha="left", va="top",
    )
    # The glyph fills the card: at 0.70 of the card height a dead band of a
    # quarter card sat between the header and the input-label row while the
    # soma arrow crowded the footer text.
    draw_conflict_neuron(
        frame,
        (x0 + 0.04 * width, y0 + 0.10 * height,
         0.92 * width, 0.80 * height),
        conflict=conflict,
    )
    frame.text(
        (x0 + width - frame.fx(5.0), y0 + frame.fy(6.0)),
        "compatible shared updates" if not conflict else "misdirected shared updates",
        size=PT_SMALL, color=GREEN if not conflict else COLORS["highlight"],
        ha="right", va="bottom",
    )


def branch_conflict_task(ax) -> None:
    """Side-by-side endpoints of the continuous conflict-dose family."""
    frame = Frame(ax, labels=True, scale=0.94)
    # An 11 pt band below the cards defines the doses BETWEEN the endpoints:
    # the cards draw only chi = 0 and chi = 1, and without this line neither
    # the ink nor the axis tells a reader whether an intermediate dose flips
    # views independently or a fixed fraction of branches.
    cells = frame.split(2, axis="x", gap_pt=9.0, pad_pt=(0, 0, 0, 11.0))
    _trial_card(frame, cells[0], conflict=False)
    _trial_card(frame, cells[1], conflict=True)
    frame.text((0.5, 0.0), "0 < χ < 1: each nonselected view flips "
               "with prob. χ", size=PT_SMALL, color=MUTE,
               ha="center", va="bottom")


def backward_credit_schematic(ax) -> None:
    """Separate forward selection from branch-resolved learning signals."""
    frame = Frame(ax, labels=True, scale=0.96)
    rows = frame.split(3, axis="y", gap_pt=6.0, pad_pt=(0, 0, 1.5, 1.5))
    # Each row names the quantity it delivers, in the notation the text and
    # Supplementary Fig. S29 use.  The earlier forms printed a bare
    # "δ·1[b = c]" beside "(δ/B)·1", where the two "1"s meant different
    # things (an indicator and a ones-vector) and neither expression said
    # what was being defined.
    specs = (
        ("local eligibility", ("e", "b", " ≠ 0 for every branch"),
         MUTE, "eligibility"),
        ("inactive exact gradient = 0", ("δ", "b", " = δ · 1[b = c]"),
         GREEN, "selective"),
        ("neuron-shared credit", ("δ", "b", " = δ / B for every branch"),
         AMBER, "shared"),
    )
    for index, (rect, (title, equation, color, glyph_mode)) in enumerate(
            zip(rows, specs, strict=True)):
        x0, y0, width, height = rect
        # A hairline rule above every row but the first, in place of the
        # gray card fills: the fills made three heavy slabs of a ladder that
        # the rest of the paper draws as ruled rows (Fig. 1C).
        if index:
            frame.ax.plot([x0, x0 + width],
                          [y0 + height + frame.fy(2.6)] * 2,
                          color=COLORS["grid"], lw=LW_HAIR, zorder=1)
        frame.text((x0 + frame.fx(5.0), y0 + 0.66 * height), title,
                   size=PT_SMALL, color=color, ha="left")
        base, sub, tail = equation
        if sub:
            _sub_token(frame.ax, x0 + frame.fx(5.0),
                       y0 + 0.30 * height, base, sub, tail,
                       size=PT_SMALL, sub_size=PT_SMALL, color=INK,
                       ha="left", va="center")
        else:
            frame.text((x0 + frame.fx(5.0), y0 + 0.30 * height), base,
                       size=PT_SMALL, color=INK, ha="left")
        draw_credit_fan(
            frame,
            (x0 + 0.55 * width, y0 + 0.02 * height,
             0.43 * width, 0.96 * height),
            mode=glyph_mode, color=color,
        )


def full_conflict_controls(ax, summary: pd.DataFrame,
                           seeds: pd.DataFrame | None = None) -> None:
    """Full-conflict trained accuracy with every implementation control.

    This panel merges what used to be two: the paired correct-minus-shared
    effect and the equivalence/derangement controls were both slices of the
    same chi = 1 endpoint, so they are now one summary carrying the levels,
    the per-seed spread and all five conditions.  The chi = 0 null they used
    to state separately is the left edge of panel D.
    """
    endpoint = summary[np.isclose(summary.conflict_probability, 1.0)]
    branches = np.asarray([2, 4, 8], dtype=float)
    # Equally spaced categorical positions, exactly as boundary_test() below:
    # side by side the two bottom panels used to encode the same "branches B"
    # axis two different ways (linear here, categorical there).
    xpos = np.arange(len(branches), dtype=float)

    correct = endpoint[endpoint.condition.eq("correct_path")].set_index(
        "branches").loc[branches]
    mean = correct.mean_test_accuracy.to_numpy(float)
    low = correct.ci95_low_test_accuracy.to_numpy(float)
    high = correct.ci95_high_test_accuracy.to_numpy(float)
    ax.errorbar(xpos, mean, yerr=[mean - low, high - mean],
                color=GREEN, marker="o", markerfacecolor="white",
                markeredgecolor=GREEN, markeredgewidth=LW_EDGE,
                ms=MARKER_MS + 0.6, lw=LW_DATA, capsize=ERR_CAPSIZE, zorder=4)
    # The three series are numerically identical.  Concentric markers preserve
    # that fact visually without horizontal jitter that would imply a dose.
    ax.plot(xpos, mean, lw=0, marker="s", ms=MARKER_MS + 3.0,
            markerfacecolor="none", markeredgecolor=PURPLE,
            markeredgewidth=LW_EDGE, zorder=3)
    ax.plot(xpos, mean, lw=0, marker="+", ms=MARKER_MS + 1.4,
            color=COLORS["bp"], markeredgewidth=LW_EDGE, zorder=5)

    for condition, color, marker, label in (
        ("neuron_shared_k1", AMBER, "D", "neuron-shared"),
        ("within_neuron_deranged", GRAY, "v", "deranged route"),
    ):
        part = endpoint[endpoint.condition.eq(condition)].set_index(
            "branches").loc[branches]
        y = part.mean_test_accuracy.to_numpy(float)
        lo = part.ci95_low_test_accuracy.to_numpy(float)
        hi = part.ci95_high_test_accuracy.to_numpy(float)
        ax.errorbar(xpos, y, yerr=[y - lo, hi - y], color=color,
                    marker=marker, markerfacecolor="white",
                    markeredgecolor=color, markeredgewidth=LW_EDGE,
                    ms=MARKER_MS, lw=LW_DATA, capsize=ERR_CAPSIZE,
                    label=label, zorder=3)

    # Per-seed spread for the two conditions the contrast is about, drawn as
    # a deterministic fan so the reader sees the 20 paired fits behind each
    # mean rather than a bare interval.
    if seeds is not None:
        full = seeds[np.isclose(seeds.conflict_probability, 1.0)]
        for condition, tone in (("correct_path", GREEN),
                                ("neuron_shared_k1", AMBER)):
            part = full[full.condition.eq(condition)]
            for position, branch in zip(xpos, branches, strict=True):
                values = part[part.branches.eq(int(branch))].test_accuracy
                values = values.to_numpy(float)
                if not values.size:
                    continue
                # Constant fan width, scaled to the unit categorical spacing
                # (the +-0.24 of the old linear axis covered the same share
                # of the axis span).
                offset = np.linspace(-0.12, 0.12, values.size)
                ax.plot(np.full(values.size, position) + offset,
                        values, linestyle="none", marker="o", ms=SEED_MS,
                        color=tone, alpha=SEED_ALPHA,
                        markeredgecolor="none", zorder=2)

    # A compact in-panel key occupies the empty middle band; a conventional
    # legend would cover the two low-accuracy curves.
    key_specs = (
        (0.705, GREEN, "o", "branch-specific = BP = gated point"),
        (0.640, AMBER, "D", "neuron-shared"),
        (0.575, GRAY, "v", "deranged route"),
    )
    for ypos, color, marker, label in key_specs:
        ax.plot([0.04, 0.20], [ypos, ypos], color=color, lw=LW_DATA,
                marker=marker, markerfacecolor="white",
                markeredgewidth=LW_EDGE, ms=MARKER_MS - 0.6,
                markevery=[1], clip_on=False, zorder=6)
        ax.text(0.34, ypos, label, fontsize=PT_SMALL, color=color,
                ha="left", va="center")
    # The green row stands for three coincident implementations that the panel
    # draws as concentric glyphs.  Repeat the gated-point square and the
    # backpropagation plus on its key marker, so both are identified where the
    # reader first meets them rather than appearing as unexplained overprints.
    ax.plot([0.20], [key_specs[0][0]], lw=0, marker="s",
            ms=MARKER_MS + 2.4, markerfacecolor="none",
            markeredgecolor=PURPLE, markeredgewidth=LW_EDGE,
            clip_on=False, zorder=6)
    ax.plot([0.20], [key_specs[0][0]], lw=0, marker="+",
            ms=MARKER_MS + 0.8, color=COLORS["bp"],
            markeredgewidth=LW_EDGE, clip_on=False, zorder=7)
    ax.axhline(0.5, color=MUTE, lw=LW_REF, dashes=(2.2, 1.8), zorder=0)
    ax.text(2.29, 0.507, "chance", fontsize=PT_SMALL, color=MUTE,
            ha="right", va="bottom")
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.18, 0.84)
    ax.set_xticks(xpos, ["2", "4", "8"])
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_xlabel("branches B")
    ax.set_ylabel("held-out accuracy")


def boundary_test(ax, crossings: pd.DataFrame,
                  seed_boundaries: pd.DataFrame) -> None:
    """Observed chance crossing against the analytic boundary, per B.

    This is the figure's central prediction actually being tested: panel C
    derives chi_c = B/[2(B-1)] and panel D marks it, but only this panel
    compares it with where the trained shared rule really fell to chance.
    It was previously available only as Supplementary Fig. S29c.
    """
    crossings = crossings.sort_values("branches")
    branches = crossings.branches.to_numpy(int)
    x = np.arange(len(branches), dtype=float)
    predicted = crossings.predicted_boundary.to_numpy(float)
    observed = crossings.trained_mean_curve_chance_crossing.to_numpy(float)

    # Per-seed crossings, grid-quantized by the dose sweep: with them the
    # agreement stops being two coincident means and becomes visible mass on
    # the boundary-adjacent doses.  Seeds that never reached chance inside
    # the sweep (5/20 at B = 2) are absent by construction.
    rng = np.random.default_rng(41)
    for xpos, branch in zip(x, branches, strict=True):
        doses = (seed_boundaries[seed_boundaries.branches.eq(branch)]
                 .first_at_or_below_chance_accuracy_dose.dropna()
                 .to_numpy(float))
        ax.scatter(np.full(doses.size, xpos)
                   + rng.uniform(-0.10, 0.10, doses.size), doses,
                   s=SEED_MS ** 2, color=AMBER, alpha=0.35,
                   edgecolors="none", zorder=2.5)
        n_missing = int(seed_boundaries[seed_boundaries.branches.eq(branch)]
                        .first_at_or_below_chance_accuracy_dose.isna().sum())
        if n_missing:
            ax.scatter(xpos + np.linspace(-.09,.09,n_missing),
                       np.full(n_missing,1.055),marker="^",s=SEED_MS**2,
                       facecolors="white",edgecolors=AMBER,lw=LW_EDGE,zorder=5)
            ax.text(xpos+.12,1.075,f"{n_missing}/20 > 1",fontsize=PT_SMALL,
                    color=AMBER,ha="left",va="center")

    for xpos, theory, trained in zip(x, predicted, observed, strict=True):
        ax.plot([xpos, xpos], [theory, trained], color=MUTE, lw=LW_HAIR,
                alpha=0.7, zorder=1)
    ax.plot(x, predicted, color=INK, lw=LW_REF, dashes=(2.4, 1.8), zorder=2)
    ax.plot(x, observed, color=AMBER, lw=LW_DATA, zorder=3)
    ax.plot(x, predicted, linestyle="none", marker="D", ms=MARKER_MS,
            markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=LW_EDGE, zorder=4, label="analytic boundary")
    ax.plot(x, observed, linestyle="none", marker="o", ms=MARKER_MS,
            markerfacecolor=AMBER, markeredgecolor="white",
            markeredgewidth=LW_EDGE, zorder=5,
            label="mean-curve crossing")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), frameon=False,
              handlelength=1.5, handletextpad=0.45, borderaxespad=0.0,
              fontsize=PT_SMALL)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.50, 1.105)
    ax.set_xticks(x, [str(branch) for branch in branches])
    ax.set_yticks([0.50, 0.75, 1.00])
    ax.set_xlabel("branches B")
    # One name for the one variable: C and D call the x quantity "conflict
    # dose", so this axis reports the chance-crossing dose, with the same
    # subscripted boundary symbol the other panels tag.  The subscript chains
    # on the rotated label: for a 90-degree label the glyph-down (drop)
    # direction is +x in display space and the advance direction is +y.
    ax.set_ylabel("chance-crossing dose χ")
    ax.annotate("c", xy=(1.0, 1.0), xycoords=ax.yaxis.label,
                xytext=(-0.2, 0.4), textcoords="offset points",
                fontsize=PT_SMALL, color=INK, rotation=90,
                rotation_mode="anchor", ha="left", va="baseline",
                annotation_clip=False, zorder=5)


def build() -> list:
    mpl.rcParams["lines.markeredgewidth"] = LW_EDGE
    summary = pd.read_csv(PATH_NECESSITY / "condition_summary.csv")
    seeds = pd.read_csv(PATH_NECESSITY / "seed_outcomes.csv")
    crossings = pd.read_csv(PATH_NECESSITY / "plotted_crossings.csv")
    seed_boundaries = pd.read_csv(PATH_NECESSITY / "boundary_by_seed.csv")

    # The caption states that the conflict interaction is positive in every
    # paired seed (20/20 at each B).  No panel plots that statistic any more,
    # so verify it here rather than letting the claim drift from the data.
    interactions = pd.read_csv(PATH_NECESSITY / "interaction_summary.csv")
    positive = interactions.set_index("branches").loc[[2, 4, 8]].positive_pairs
    if not np.all(positive.to_numpy(int) == 20):
        raise ValueError(
            "Fig. 4 caption claims 20/20 positive interaction slopes at each "
            f"B; source table reports {positive.to_dict()}"
        )

    canvas = NativeCanvas(
        HEIGHT_IN, 3, row_weights=ROW_PT,
        hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
        # Left margin 44, not 36: at 36 the C/E y labels reached the canvas edge,
        # so their panel letters hit the 2.5 pt clamp and could not sit left of
        # their own labels.
        margins=Margins(left=51.0, right=13.0, top=22.0, bottom=26.0),
    )
    fig = canvas.fig
    # Row 0 is 7/5, not 8/4: at 8/4 the two task cards were the largest
    # cards in the manuscript (146x131 pt) while B's three cards were so
    # narrow that their glyphs drew at a quarter of A's linear scale.
    ax_a = canvas.panel("A", 0, 0, 7, schematic=True,
                        title="Context-gated branch-conflict task")
    ax_b = canvas.panel("B", 0, 7, 5, schematic=True,
                        title="Forward selection, backward credit")
    ax_c = canvas.panel("C", 1, 0, 5,
                        title="Predicted shared-mode boundary")
    ax_d = canvas.panel("D", 1, 5, 7, schematic=True,
                        title="Branch count shifts the trained transition")
    ax_e = canvas.panel("E", 2, 0, 6,
                        title="Full conflict: routing decides the outcome")
    ax_f = canvas.panel("F", 2, 6, 6,
                        title="Chance crossings shift with branch count")

    # The shared token_subscript raises the base glyph above the line (its
    # tail inherits the subscript's drop); stand in the corrected helper for
    # every panel this figure draws through routing_figure_panels -- panel A's
    # x-subscripts and panel D's boundary tags -- and restore it so sibling
    # builders in the same process keep the module untouched.
    _shared_subscript = routing_figure_panels.token_subscript
    routing_figure_panels.token_subscript = _sub_token
    try:
        branch_conflict_task(ax_a)
        backward_credit_schematic(ax_b)
        shared_mode_boundary(ax_c)
        path_accuracy_facets(ax_d, summary)
    finally:
        routing_figure_panels.token_subscript = _shared_subscript

    # One name for the one variable chi across C, D, F and the caption.
    ax_c.set_xlabel("conflict dose χ")
    # C's zero crossings ARE the analytic boundary that F labels with an open
    # ink diamond; give the same quantity the same glyph (the line shades
    # already distinguish B, so the per-B marker shapes said nothing).
    for line in ax_c.get_lines():
        if line.get_marker() not in ("", "None", None) \
                and len(line.get_xdata()) == 1:
            line.set_marker("D")
            line.set_color(INK)
            line.set_markerfacecolor("white")
            line.set_markeredgecolor(INK)
            line.set_markersize(MARKER_MS)
    for text in ax_d.texts:
        if text.get_text() == "held-out accuracy":
            text.set_x(-0.035)
        elif text.get_text() == "credit-conflict probability χ":
            text.set_text("conflict dose χ")
        elif text.get_text() == "branch-specific = BP = gated point":
            # Anchored top-right this label sat beside the B=8 facet's tags
            # and read as facet-specific; moved to the host's top-left it
            # shared a 5 pt baseline gap with the facet tag row, and the 'g'
            # descender of "gated" struck the 'B' cap of the B=4 tag.  The
            # header band cannot hold two text rows, so the green condition
            # name joins "neuron-shared" inside the first facet's empty band
            # (the full equivalence string stays in panel E's key and in the
            # caption), and this host-level label is retired.
            text.set_visible(False)
    ax_d.child_axes[0].text(0.05, 0.645, "branch-specific",
                            fontsize=PT_SMALL, color=GREEN,
                            ha="left", va="top")
    # The facet boundary tags carried the per-B slate shades; the lightest
    # (B=2) had weak contrast on white, and the same quantity is inked in C
    # and F.  Ink the tags, keep the dashed boundary lines in the shades.
    shade_hexes = {shade.lower() for shade, _ in B_STYLE.values()}
    for inset in ax_d.child_axes:
        for text in inset.texts:
            if mpl.colors.to_hex(text.get_color()).lower() in shade_hexes:
                text.set_color(INK)
    full_conflict_controls(ax_e, summary, seeds)
    boundary_test(ax_f, crossings, seed_boundaries)

    COMPONENT.parent.mkdir(parents=True, exist_ok=True)
    from journal_style import style_direct_color_labels
    style_direct_color_labels(canvas.fig)
    problems = canvas.save(COMPONENT, name="main_figure_04_native")
    return problems


def main() -> None:
    problems = build()
    for problem in problems:
        print(f"  {problem}")
    print(f"  canvas width {FIG_W * 72:.1f} pt, height {CANVAS_H_PT:.1f} pt")


if __name__ == "__main__":
    main()

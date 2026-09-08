#!/usr/bin/env python3
"""Main Figure 2 -- branch selection versus shared credit (``fig:branchconflict``).

Component ``figures/components/main_figure_04_native.pdf`` (the historical
component name is kept so ``rebuild_final_publication_figures.py`` and the
provenance map stay valid).  Eight panels on one ``NativeCanvas``:

  row 0  A task cards (compatible / conflicting)      B three deliveries
  row 1  C initial signed utility vs dose             D cosine vs dose
         E analytic vs trained chance crossings
  row 2  F/G/H held-out accuracy vs dose, one facet per branch count

Waiver (design spec D3): row 1 places the 4-module crossings panel E
(x = branches) beside the column-locked shared-x pair C/D (x = dose); the
three panels share one width, so the letters C/D/E align with F/G/H.

Every schematic element comes from the shared glyph library
(:mod:`native_schematics`); the only private drawing is the B-branch fan
tree, assembled from ``Frame.dendrite`` / ``junction`` / ``contact`` /
``gate`` / ``soma`` / ``error_in`` because the library's balanced tree is
binary and this task has B parallel branches.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex

from credit_tree_schematics import AMBER_TEXT, mix
from figure_canvas import (
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    SEED_MS,
    Margins,
    NativeCanvas,
)
from journal_style import label_color
from native_schematics import (
    CONTACT_DIA_PT,
    LINE_BAND_PT,
    SOMA_R_PT,
    Frame,
    Nodes,
    _text_w_pt,
)
from routing_figure_panels import PATH_NECESSITY


ROOT = Path(__file__).resolve().parents[1]
COMPONENT = ROOT / "figures" / "components" / "main_figure_04_native.pdf"
PUBLISHED = ROOT / "figures" / "main" / "figure_02.pdf"
TRAJECTORIES = ROOT / "source_data" / "review_branch_trajectories"

# Grid.  The design spec asks for 484 pt with a 40-pt vertical gutter; the
# row-separation audit needs 8.5 pt of clear ink between row 1's x labels
# (22.6 pt of tick + label) and row 2's title band (13 pt), which a 40-42 pt
# gutter cannot give (46 pt measured 8.2), so the gutter is 48 pt and the
# rows are trimmed in the spec's proportions:
# 22 + 110 + 48 + 110 + 48 + 116 + 32 = 486 (in the 484-492 band).
CANVAS_H_PT = 486.0
HEIGHT_IN = CANVAS_H_PT / 72.0
ROW_PT = [110.0, 110.0, 116.0]
HGUTTER_PT = 34.0
VGUTTER_PT = 48.0
MARGINS = Margins(left=51.0, right=13.0, top=22.0, bottom=32.0)

INK = COLORS["ink"]
MUTE = COLORS["mute"]
GREEN = COLORS["shunting"]
AMBER = COLORS["scalar"]
GRAY = COLORS["point_mlp"]
HIGHLIGHT = COLORS["highlight"]

BRANCHES = (2, 4, 8)
# The paper's one ordinal ramp (spec D8).  ``journal_style`` does not yet
# register ORDINAL_RAMP and per-figure tasks may not edit it, so the same
# definition lives here until the library owner absorbs it.
ORDINAL_RAMP = [mix("edge", 45), mix("edge", 75), COLORS["edge"]]
RAMP = dict(zip(BRANCHES, ORDINAL_RAMP))
RAMP_TEXT = {b: label_color(to_hex(c)) for b, c in RAMP.items()}
B_MARKER = {2: "o", 4: "s", 8: "^"}
CHI_C = {b: b / (2.0 * (b - 1)) for b in BRANCHES}
DOSE_TICKS = ([0.0, 0.5, 1.0], ["0", "0.5", "1"])
# C/D share this x range: the data end at chi = 1 and the band beyond it
# carries the direct series labels (inside the axes, never in the gutter).
XLIM_DOSE = (-0.02, 1.27)
LABEL_X = 1.045
DASHES = (2.2, 1.8)


def _pt_to_data_x(ax, pt):
    """Points -> data units along x, at the axes' CURRENT box."""
    fig = ax.get_figure()
    w_pt = ax.get_position().width * fig.get_size_inches()[0] * 72.0
    lo, hi = ax.get_xlim()
    return pt * (hi - lo) / w_pt


def _tag_chain(ax, xy, dx_pt, base, sub, tail, *, color, size=PT_ANNOT,
               va="top", zorder=5):
    """``base``+subscript+``tail`` anchored ``dx_pt`` points from a data point.

    The base is an annotation offset in POINTS from ``xy``, so the tag keeps
    its distance from the mark it labels when the reserve lock moves the
    axes box; the subscript and tail chain on the base exactly as
    :func:`figure_canvas.token_subscript` does (sub at PT_SMALL dropped
    1.6 pt; tail back on the base line).
    """
    base_text = ax.annotate(base, xy=xy, xytext=(dx_pt, 0.0),
                            textcoords="offset points", fontsize=size,
                            color=color, ha="left", va=va, zorder=zorder,
                            annotation_clip=False)
    sub_text = ax.annotate(sub, xy=(1.0, 0.0), xycoords=base_text,
                           xytext=(0.4, -1.6), textcoords="offset points",
                           fontsize=PT_SMALL, color=color, ha="left",
                           va="bottom", zorder=zorder, annotation_clip=False)
    if tail:
        ax.annotate(tail, xy=(1.0, 0.0), xycoords=(sub_text, base_text),
                    xytext=(0.6, 0.0), textcoords="offset points",
                    fontsize=size, color=color, ha="left", va="bottom",
                    zorder=zorder, annotation_clip=False)
    return base_text


def _chain_w_pt(ax, base, sub, tail, size=PT_ANNOT):
    return (_text_w_pt(ax, base, size) + 0.4 + _text_w_pt(ax, sub, PT_SMALL)
            + ((0.6 + _text_w_pt(ax, tail, size)) if tail else 0.0))


# -- the B-branch fan tree (private; library geometry, library glyphs) ------
def fan_tree(frame: Frame, rect, *, B=4, selected=1, ghost=False,
             stream_tags=None, gate=True, fade_unselected=False,
             output="z", delta="δ0", bottom_pt=15.0) -> Nodes:
    """Soma at the bottom, B parallel branches, one junction per branch.

    Each branch is a proximal segment soma -> junction (taper level 1) and a
    vertical distal segment junction -> terminal (level 3) with an exc
    contact at the terminal.  ``stream_tags`` is a list of (base, sub,
    colour) per terminal; ``gate`` puts the context ring on the selected
    junction; ``fade_unselected`` attenuates the non-selected proximal
    segments.  Returns a :class:`Nodes` so the library's delivery glyphs can
    address the tree.
    """
    x0, y0, w, h = rect
    band_pt = LINE_BAND_PT if stream_tags else 0.0
    top_pad = 4.0 + CONTACT_DIA_PT * 0.5 + band_pt
    y_soma = y0 + frame.fy(bottom_pt)
    y_top = y0 + h - frame.fy(top_pad)
    y_junc = y_soma + 0.52 * (y_top - y_soma)
    xs = np.linspace(x0 + 0.12 * w, x0 + 0.88 * w, B)
    soma = (x0 + 0.5 * w, y_soma)
    nodes = Nodes()
    nodes["S"] = soma
    nodes.children["S"] = []
    for i, x in enumerate(xs):
        j, t = f"J{i + 1}", f"T{i + 1}"
        nodes[j] = (float(x), y_junc)
        nodes[t] = (float(x), y_top)
        nodes.parent[j] = "S"
        nodes.parent[t] = j
        nodes.children["S"].append(j)
        nodes.children[j] = [t]
        nodes.children[t] = []
        nodes.level[j] = 1
        nodes.level[t] = 3
    nodes.terminals = [f"T{i + 1}" for i in range(B)]
    nodes.soma = soma
    nodes.orient = "up"
    nodes.pitch_pt = float((xs[1] - xs[0]) * frame.w_pt) if B > 1 else 20.0
    nodes.rect = rect
    for name, par in nodes.parent.items():
        p0 = nodes[par]
        nodes.edges[(par, name)] = frame.dendrite(
            p0, nodes[name], level=nodes.level[name], ghost=ghost)
    if fade_unselected:
        frame.fade([nodes.edges[("S", f"J{i + 1}")]
                    for i in range(B) if i != selected])
    for i in range(B):
        j = f"J{i + 1}"
        if gate and i == selected:
            frame.gate(nodes[j], badge_offset=(-4.6, 3.4))
        else:
            nodes.rings[j] = frame.junction(nodes[j], ghost=ghost)
        frame.contact(nodes[f"T{i + 1}"], kind="exc")
    if stream_tags:
        lift = CONTACT_DIA_PT * 0.5 + 2.0
        for i, (base, sub, colour) in enumerate(stream_tags):
            t = nodes[f"T{i + 1}"]
            frame.subscript((t[0], t[1] + frame.fy(lift)), base, sub,
                            size=PT_SMALL, color=colour, ha="center",
                            va="bottom")
    frame.soma(soma, output=bool(output), label=output)
    if delta:
        frame.error_in(soma, label=delta)
    return nodes


# -- A: the task ------------------------------------------------------------
def branch_conflict_task(ax) -> None:
    """Compatible (χ = 0) and conflicting (χ = 1) endpoints, identical cards."""
    frame = Frame(ax)
    band = frame.footer("0 < χ < 1: each nonselected view flips "
                        "independently with probability χ")
    cells = frame.split(2, axis="x", gap_pt=9.0, pad_pt=(0, 0, 0, band))
    for cell, conflict in zip(cells, (False, True)):
        core = frame.task_card(cell, title="conflicting, χ = 1" if conflict
                               else "compatible, χ = 0")
        core = Frame.inset(core, left=0.03, right=0.03, bottom=0.02)
        tags = []
        for i in range(4):
            if i == 1:
                tags.append(("x", "y", INK))
            elif conflict:
                tags.append(("x", "1−y", HIGHLIGHT))
            else:
                tags.append(("x", "y", MUTE))
        fan_tree(frame, core, B=4, selected=1, stream_tags=tags, gate=True,
                 fade_unselected=True)


# -- B: three deliveries of the same error ----------------------------------
def backward_credit_schematic(ax) -> None:
    """Branch-specific, neuron-shared and deranged delivery on ghost fans.

    Three fans side by side over a three-line key.  The spec's stacked rows
    would give each fan 32 pt, which cannot hold a soma-at-bottom tree plus
    the δ0 arrow (15 pt below the soma); columns keep every glyph at the
    library's size and the key lines carry name, equation and badge.
    """
    frame = Frame(ax)
    key_pt = 3 * LINE_BAND_PT + 2.0
    cols = frame.split(3, axis="x", gap_pt=4.0, pad_pt=(0, 0, 2.0, key_pt))
    specs = (
        ("branch-specific", GREEN, "subtree", "exact"),
        ("neuron-shared", AMBER, "scalar", None),
        ("deranged", GRAY, "deranged", "control"),
    )
    for cell, (name, colour, mode, badge) in zip(cols, specs):
        nodes = fan_tree(frame, cell, B=4, selected=1, ghost=True,
                         gate=True, output="z")
        if mode == "subtree":
            # K_CYCLE capsule on the selected branch, shunting entry arrow
            frame.credit_delivery(nodes, mode="subtree", targets=["J2"],
                                  rule_color=colour)
        elif mode == "scalar":
            # D5: amber bus with hairline drops, fed from this soma's own δ0
            # arrow (the same feed line as Fig. 1B's per-neuron card).
            frame.credit_delivery(nodes, mode="scalar", rule_color=colour)
            sx, sy = nodes.soma
            tail = (sx + frame.fx(SOMA_R_PT + 11.0),
                    sy - frame.fy(SOMA_R_PT + 7.0))
            x_bus = nodes["T4"][0] + frame.fx(6.0)
            y_bus = nodes["T4"][1] + frame.fy(8.0)
            tag_w = (_text_w_pt(ax, "δ", PT_ANNOT) + 0.4
                     + _text_w_pt(ax, "0", PT_SMALL) + 1.8)
            x_from = tail[0] + frame.fx(tag_w + 2.0)
            ax.plot([x_from, x_bus, x_bus], [tail[1], tail[1], y_bus],
                    color=colour, lw=frame.lw(LW_HAIR), solid_capstyle="round",
                    solid_joinstyle="round", zorder=4.8)
        else:
            # cyclic derangement: the same capsule glyph on branch c + 1 with
            # the entry arrow in the control colour, plus the b -> b + 1 hop
            frame.credit_delivery(nodes, mode="subtree", targets=["J3"],
                                  rule_color=colour)
            j2, j3 = nodes["J2"], nodes["J3"]
            frame.arrow((j2[0] + frame.fx(4.0), j2[1] + frame.fy(4.0)),
                        (j3[0] - frame.fx(4.0), j3[1] + frame.fy(4.0)),
                        color=MUTE, lw=LW_HAIR, head=3.0, rad=-0.45,
                        zorder=4.6)
    # key lines: name (rule colour), equation, badge
    x_left = frame.fx(3.0)
    y = frame.fy(key_pt - LINE_BAND_PT * 0.5)
    equations = (("δ", "b", " = δ · 1[b = c]"),
                 ("δ", "b", " = δ / B"),
                 ("δ", "b", " = δ · 1[b = c + 1]"))
    # The spec's badge text "= BP = gated point" does not fit the 169-pt
    # key line beside the name and equation; the tie is stated in panel F
    # and the caption, and the badge keeps its one-word kind.
    for (name, colour, _mode, badge), (base, sub, tail) in zip(specs, equations):
        text_colour = AMBER_TEXT if colour == AMBER else label_color(colour)
        frame.text((x_left, y), name, size=PT_SMALL, color=text_colour,
                   ha="left")
        w_name = _text_w_pt(ax, name, PT_SMALL)
        x_eq = x_left + frame.fx(w_name + 7.0)
        frame.subscript((x_eq, y), base, sub, tail, size=PT_ANNOT, ha="left")
        if badge:
            x_badge = x_eq + frame.fx(_chain_w_pt(ax, base, sub, tail) + 8.0)
            frame.badge((x_badge, y), badge, ha="left", va="center")
        y -= frame.fy(LINE_BAND_PT)


# -- C: initial signed utility against the mean-field line ------------------
def initial_utility_boundary(ax, summary: pd.DataFrame) -> None:
    chi = np.linspace(0.0, 1.0, 301)
    shared = summary[summary.condition.eq("neuron_shared_k1")]
    for b in BRANCHES:
        colour = RAMP[b]
        ax.plot(chi, 1.0 - 2.0 * (b - 1) * chi / b, color=colour, lw=LW_DATA,
                zorder=2)
        part = shared[shared.branches.eq(b)].sort_values("conflict_probability")
        x = part.conflict_probability.to_numpy(float)
        m = part.mean_initial_signed_utility.to_numpy(float)
        lo = part.ci95_low_initial_signed_utility.to_numpy(float)
        hi = part.ci95_high_initial_signed_utility.to_numpy(float)
        if np.max(np.abs(m - (1.0 - 2.0 * (b - 1) * x / b))) > 0.02:
            raise ValueError(f"B = {b}: initial signed utility departs from "
                             "the mean-field line by more than 0.02")
        ax.errorbar(x, m, yerr=[m - lo, hi - m], linestyle="none",
                    marker=B_MARKER[b], ms=MARKER_MS, markerfacecolor="white",
                    markeredgecolor=colour, markeredgewidth=LW_EDGE,
                    ecolor=colour, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
                    capthick=LW_ERR, zorder=4)
        ax.text(LABEL_X, 1.0 - 2.0 * (b - 1) / b, f"B = {b}",
                fontsize=PT_SMALL, color=RAMP_TEXT[b], ha="left", va="center")
        ax.plot([CHI_C[b]], [0.0], linestyle="none", marker="D",
                ms=MARKER_MS, markerfacecolor="white", markeredgecolor=INK,
                markeredgewidth=LW_EDGE, zorder=5)
    ax.plot([0.0, 1.0], [0.0, 0.0], color=MUTE, lw=LW_REF, dashes=DASHES,
            zorder=0, solid_capstyle="butt")
    # the right end of the zero line carries the B = 2 crossing diamond, so
    # the reference label sits at the free left end
    ax.text(0.02, 0.035, "zero", fontsize=PT_SMALL, color=MUTE,
            ha="left", va="bottom")
    ax.text(0.025, 0.05, "s(χ) = 1 − 2χ(B − 1)/B", transform=ax.transAxes,
            fontsize=PT_ANNOT, color=INK, ha="left", va="bottom")
    ax.set_xlim(*XLIM_DOSE)
    ax.set_ylim(-0.86, 1.10)
    ax.set_xticks(*DOSE_TICKS)
    ax.set_yticks([-0.5, 0.0, 0.5, 1.0], ["−0.5", "0", "0.5", "1"])
    ax.set_xlabel("conflict dose χ")
    ax.set_ylabel("initial signed utility")


# -- D: cosine with the exact update, epoch 0 and epoch 250 -----------------
def alignment_collapse(ax, intervals: pd.DataFrame,
                       crossings: pd.DataFrame) -> dict:
    common = intervals[intervals.state_comparison.eq("common_exact_state")]
    zero = crossings[crossings.state_comparison.eq("common_exact_state")
                     & crossings.epoch.eq(250)]
    trained_zero = {}
    for b in BRANCHES:
        colour = RAMP[b]
        e0 = common[common.branches.eq(b) & common.epoch.eq(0)] \
            .sort_values("conflict_probability")
        e250 = common[common.branches.eq(b) & common.epoch.eq(250)] \
            .sort_values("conflict_probability")
        ax.plot(e0.conflict_probability, e0.mean_cosine, linestyle="none",
                marker=B_MARKER[b], ms=MARKER_MS, markerfacecolor="white",
                markeredgecolor=colour, markeredgewidth=LW_EDGE, zorder=3)
        x = e250.conflict_probability.to_numpy(float)
        ax.fill_between(x, e250.ci95_low, e250.ci95_high, color=colour,
                        alpha=0.13, linewidth=0, zorder=1)
        ax.plot(x, e250.mean_cosine, color=colour, lw=LW_DATA, zorder=4)
        dose = float(zero[zero.branches.eq(b)].interpolated_zero_alignment.mean())
        trained_zero[b] = dose
        ax.plot([dose], [0.0], linestyle="none", marker="|", ms=7.0,
                markeredgecolor=colour, markeredgewidth=LW_ERR, zorder=5)
        ax.plot([CHI_C[b]], [0.0], linestyle="none", marker="D",
                ms=MARKER_MS, markerfacecolor="white", markeredgecolor=INK,
                markeredgewidth=LW_EDGE, zorder=5)
    ax.plot([0.0, 1.0], [0.0, 0.0], color=MUTE, lw=LW_REF, dashes=DASHES,
            zorder=0, solid_capstyle="butt")
    ax.text(0.98, 1.11, "epoch 0", fontsize=PT_SMALL, color=MUTE,
            ha="right", va="bottom")
    # epoch-0 markers fall to -1 beyond chi_c, so the epoch-250 tag sits in
    # the lower-left region every trained curve is still above
    ax.text(0.03, -0.64, "epoch 250", fontsize=PT_SMALL, color=MUTE,
            ha="left", va="top")
    # the one sub-title line: which state the two updates are compared at
    ax.text(0.0, -1.23, "common exact-rule state", fontsize=PT_SMALL,
            color=MUTE, ha="left", va="bottom")
    ax.set_xlim(*XLIM_DOSE)
    ax.set_ylim(-1.25, 1.40)
    ax.set_xticks(*DOSE_TICKS)
    ax.set_yticks([-1.0, 0.0, 1.0], ["−1", "0", "1"])
    ax.set_xlabel("conflict dose χ")
    ax.set_ylabel("cosine with exact update")
    return trained_zero


# -- E: analytic boundary versus trained chance crossing --------------------
def boundary_test(ax, crossings: pd.DataFrame,
                  seed_boundaries: pd.DataFrame) -> None:
    """Observed chance crossing against the analytic boundary, per B."""
    crossings = crossings.sort_values("branches")
    branches = crossings.branches.to_numpy(int)
    x = np.arange(len(branches), dtype=float)
    predicted = crossings.predicted_boundary.to_numpy(float)
    observed = crossings.trained_mean_curve_chance_crossing.to_numpy(float)

    rng = np.random.default_rng(41)
    for xpos, branch in zip(x, branches, strict=True):
        column = seed_boundaries[seed_boundaries.branches.eq(branch)] \
            .first_at_or_below_chance_accuracy_dose
        doses = column.dropna().to_numpy(float)
        ax.scatter(np.full(doses.size, xpos)
                   + rng.uniform(-0.10, 0.10, doses.size), doses,
                   s=SEED_MS ** 2, color=MUTE, alpha=0.35,
                   edgecolors="none", zorder=2.5)
        n_missing = int(column.isna().sum())
        if n_missing:
            ax.scatter(xpos + np.linspace(-0.09, 0.09, n_missing),
                       np.full(n_missing, 1.055), marker="^", s=SEED_MS ** 2,
                       facecolors="white", edgecolors=AMBER, lw=LW_EDGE,
                       zorder=5)
            ax.text(xpos + 0.14, 1.058, f"{n_missing}/{len(column)} > 1",
                    fontsize=PT_SMALL, color=AMBER_TEXT, ha="left",
                    va="center")

    for xpos, theory, trained in zip(x, predicted, observed, strict=True):
        ax.plot([xpos, xpos], [theory, trained], color=MUTE, lw=LW_HAIR,
                alpha=0.7, zorder=1)
    ax.plot(x, predicted, color=INK, lw=LW_REF, dashes=DASHES, zorder=2)
    ax.plot(x, observed, color=AMBER, lw=LW_DATA, zorder=3)
    ax.plot(x, predicted, linestyle="none", marker="D", ms=MARKER_MS,
            markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=LW_EDGE, zorder=4, label="analytic boundary")
    ax.plot(x, observed, linestyle="none", marker="o", ms=MARKER_MS,
            markerfacecolor=AMBER, markeredgecolor="white",
            markeredgewidth=LW_EDGE, zorder=5, label="mean-curve crossing")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.02), frameon=False,
              handlelength=1.5, handletextpad=0.45, borderaxespad=0.0,
              labelspacing=0.3, fontsize=PT_LEGEND)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.50, 1.105)
    ax.set_xticks(x, [str(branch) for branch in branches])
    ax.set_yticks([0.50, 0.75, 1.00], ["0.50", "0.75", "1.00"])
    ax.set_xlabel("branches B")
    # "chance-crossing dose χ_c": the subscript chains on the rotated label
    # (glyph-down is +x in display space for a 90-degree label).
    ax.set_ylabel("chance-crossing dose χ")
    ax.annotate("c", xy=(1.0, 1.0), xycoords=ax.yaxis.label,
                xytext=(-0.2, 0.4), textcoords="offset points",
                fontsize=PT_SMALL, color=INK, rotation=90,
                rotation_mode="anchor", ha="left", va="baseline",
                annotation_clip=False, zorder=5)


# -- F/G/H: accuracy facets -------------------------------------------------
SERIES = (
    ("correct_path", GREEN, "o"),
    ("neuron_shared_k1", AMBER, "s"),
    ("within_neuron_deranged", GRAY, "v"),
)
BAND_ALPHA = 0.13
YLIM_ACC = (0.16, 0.92)
DASH_TOP = 0.845          # the boundary dash stops under its own tag


def accuracy_facet(ax, summary: pd.DataFrame, branches: int, *,
                   first: bool, note=None) -> None:
    for condition, colour, marker in SERIES:
        part = summary[summary.condition.eq(condition)
                       & summary.branches.eq(branches)] \
            .sort_values("conflict_probability")
        x = part.conflict_probability.to_numpy(float)
        m = part.mean_test_accuracy.to_numpy(float)
        ax.fill_between(x, part.ci95_low_test_accuracy,
                        part.ci95_high_test_accuracy, color=colour,
                        alpha=BAND_ALPHA, linewidth=0, zorder=1)
        ax.plot(x, m, color=colour, lw=LW_DATA, marker=marker,
                ms=MARKER_MS - 0.8, markerfacecolor="white",
                markeredgecolor=colour, markeredgewidth=LW_EDGE, zorder=3)
    boundary = CHI_C[branches]
    ax.plot([boundary, boundary], [YLIM_ACC[0], DASH_TOP], color=MUTE,
            lw=LW_REF, dashes=DASHES, zorder=0, solid_capstyle="butt")
    ax.axhline(0.5, color=MUTE, lw=LW_REF, dashes=DASHES, zorder=0)
    if first:
        # the chance label sits where neither the falling shared curve nor
        # the deranged curve passes (0.55 < chi < 0.75 above the line)
        ax.text(0.76, 0.51, "chance", fontsize=PT_SMALL, color=MUTE,
                ha="right", va="bottom")
        ax.text(0.02, 0.815, "branch-specific", fontsize=PT_SMALL,
                color=GREEN, ha="left", va="bottom")
        ax.text(0.80, 0.63, "neuron-shared", fontsize=PT_SMALL,
                color=AMBER, ha="right", va="top")
        ax.text(0.02, 0.575, "deranged", fontsize=PT_SMALL, color=GRAY,
                ha="left", va="bottom")
    if note:
        ax.text(*note[0], note[1], fontsize=PT_SMALL, color=MUTE,
                ha=note[2], va=note[3], linespacing=1.15)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(*YLIM_ACC)
    ax.set_xticks(*DOSE_TICKS)
    ax.set_yticks([0.2, 0.5, 0.8])
    if not first:
        ax.tick_params(axis="y", labelleft=False)
    else:
        ax.set_ylabel("held-out accuracy")
    ax.set_xlabel("conflict dose χ")


def boundary_tag(ax, branches: int) -> None:
    """``χ_c = value`` beside the boundary dash, placed after the reserve
    lock so its point offsets are final; B = 2's dash sits on the right
    spine, so that tag hangs to the LEFT of its line."""
    boundary = CHI_C[branches]
    tail = f" = {boundary:.2f}"
    y = DASH_TOP + 0.012
    if boundary > 0.85:
        dx = -(_chain_w_pt(ax, "χ", "c", tail) + 2.5)
    else:
        dx = 2.5
    _tag_chain(ax, (boundary, y), dx, "χ", "c", tail, color=MUTE,
               va="bottom")


def _check_identities(contrasts: pd.DataFrame, interactions: pd.DataFrame,
                      intervals: pd.DataFrame, trajectories: pd.DataFrame):
    """Assertions behind the caption's exact claims."""
    acc = contrasts[contrasts.endpoint.eq("test_accuracy")]
    ties = acc[acc.contrast.isin(["correct - BP", "correct - gated point"])]
    if not (ties.ties.eq(20).all() and np.allclose(ties.mean_difference, 0.0)):
        raise ValueError("branch-specific, BP and gated point no longer tie "
                         "in all 20 pairs")
    positive = interactions.set_index("branches").loc[list(BRANCHES)] \
        .positive_pairs
    if not np.all(positive.to_numpy(int) == 20):
        raise ValueError("Fig. 2 caption claims 20/20 positive interaction "
                         f"slopes at each B; source reports {positive.to_dict()}")
    # The S29 intervals file carries no ``condition`` column; pin it to the
    # neuron-shared rule against the per-condition trajectory summary.
    ref = trajectories[trajectories.condition.eq("neuron_shared_k1")
                       & trajectories.state_comparison.eq("common_exact_state")
                       & trajectories.epoch.isin([0, 250])]
    got = intervals[intervals.state_comparison.eq("common_exact_state")
                    & intervals.epoch.isin([0, 250])]
    keys = ["branches", "epoch", "conflict_probability"]
    merged = got.merge(ref, on=keys, suffixes=("", "_ref"))
    if len(merged) != len(got) or not np.allclose(
            merged.mean_cosine, merged.mean_cosine_ref, atol=1e-6):
        raise ValueError("figure_S29_gradient_intervals.csv does not match "
                         "the neuron_shared_k1 trajectory summary")


def _equalize_row_widths(canvas: NativeCanvas, rows) -> None:
    """One axes width for every 4-module panel of a row (audit L2).

    The column lock carves each panel's left and right reserves separately
    (a y label on the left, a neighbour's y label eating the shared gutter
    on the right), so the three panels of a row can end up a few points
    apart.  Declare the row's largest left + right total as extra RIGHT
    reserve on the narrower columns so every panel yields the same width;
    the declared reserve is keyed by grid column, so rows 1 and 2 stay
    identical too.
    """
    locks = canvas.lock_reserves()
    for names in rows:
        totals = {n: locks[n][0] + locks[n][1] for n in names}
        target = max(totals.values())
        for n in names:
            short = target - totals[n]
            if short > 0.05:
                canvas.declare_reserve(n, right=locks[n][1] + short)
    canvas.lock_reserves()


def build() -> list:
    mpl.rcParams["lines.markeredgewidth"] = LW_EDGE
    summary = pd.read_csv(PATH_NECESSITY / "condition_summary.csv")
    crossings = pd.read_csv(PATH_NECESSITY / "plotted_crossings.csv")
    seed_boundaries = pd.read_csv(PATH_NECESSITY / "boundary_by_seed.csv")
    contrasts = pd.read_csv(PATH_NECESSITY / "paired_contrasts.csv")
    interactions = pd.read_csv(PATH_NECESSITY / "interaction_summary.csv")
    intervals = pd.read_csv(TRAJECTORIES / "figure_S29_gradient_intervals.csv")
    zero_crossings = pd.read_csv(TRAJECTORIES / "zero_alignment_crossings.csv")
    trajectories = pd.read_csv(TRAJECTORIES / "condition_summary.csv")
    _check_identities(contrasts, interactions, intervals, trajectories)
    n_pos = int(interactions.set_index("branches").loc[4].positive_pairs)
    n_seeds = int(interactions.set_index("branches").loc[4].n_seeds)
    n_ties = int(contrasts[contrasts.endpoint.eq("test_accuracy")
                           & contrasts.contrast.eq("correct - BP")].ties.min())

    canvas = NativeCanvas(
        HEIGHT_IN, 3, row_weights=ROW_PT,
        hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT, margins=MARGINS,
    )
    ax_a = canvas.panel("A", 0, 0, 7, schematic=True, lock=False,
                        title="Context selects one branch; conflict corrupts the rest")
    ax_b = canvas.panel("B", 0, 7, 5, schematic=True, lock=False,
                        title="Three ways to deliver the same error")
    ax_c = canvas.panel("C", 1, 0, 4, title="Boundary holds at epoch 0")
    ax_d = canvas.panel("D", 1, 4, 4, title="Alignment lost in training",
                        sharex=ax_c)
    ax_e = canvas.panel("E", 1, 8, 4, title="Trained order as predicted")
    ax_f = canvas.panel("F", 2, 0, 4, title="B = 2: shared still learns")
    ax_g = canvas.panel("G", 2, 4, 4, title="B = 4: boundary moves left",
                        sharey=ax_f)
    ax_h = canvas.panel("H", 2, 8, 4, title="B = 8: earliest collapse",
                        sharey=ax_f)

    branch_conflict_task(ax_a)
    backward_credit_schematic(ax_b)
    initial_utility_boundary(ax_c, summary)
    alignment_collapse(ax_d, intervals, zero_crossings)
    boundary_test(ax_e, crossings, seed_boundaries)
    accuracy_facet(ax_f, summary, 2, first=True,
                   note=((0.06, 0.40), "BP and gated point\n= branch-specific\n"
                         f"(all {n_ties} pairs tie)", "left", "top"))
    accuracy_facet(ax_g, summary, 4, first=False,
                   note=((0.03, 0.44), "selection ×\nconflict slopes\n"
                         f"positive {n_pos}/{n_seeds}", "left", "top"))
    accuracy_facet(ax_h, summary, 8, first=False)

    from journal_style import style_direct_color_labels
    style_direct_color_labels(canvas.fig)
    _equalize_row_widths(canvas, (("C", "D", "E"), ("F", "G", "H")))
    for ax, b in ((ax_f, 2), (ax_g, 4), (ax_h, 8)):
        boundary_tag(ax, b)
    findings = canvas.align_letters()
    COMPONENT.parent.mkdir(parents=True, exist_ok=True)
    problems = canvas.save(COMPONENT, name="main_figure_04_native")
    PUBLISHED.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(COMPONENT, PUBLISHED)
    return list(findings) + list(problems)


def main() -> None:
    problems = build()
    for problem in problems:
        print(f"  {problem}")
    print(f"  canvas width {FIG_W * 72:.1f} pt, height {CANVAS_H_PT:.1f} pt")


if __name__ == "__main__":
    main()

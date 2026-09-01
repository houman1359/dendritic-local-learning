#!/usr/bin/env python3
"""Neutral plotting primitives for the two trained routing figures.

This module owns no figure number and writes no asset.  It contains only the
frozen data locations and reusable panels shared by the branch-conflict and
hierarchical-routing builders.  Keeping these primitives here prevents the
final Figure 4 and Figure 5 builders from depending on whichever scientific
result happens to occupy Figure 3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from credit_tree_schematics import (
    _BASE_XLIM,
    _Tree,
    _setup_axes,
    draw_credit_tree,
    mix,
)
from figure_canvas import (
    COLORS,
    ERR_CAPSIZE,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_ANNOT,
    PT_LABEL,
    PT_SMALL,
    SEED_ALPHA,
    SEED_MS,
    enforce_tokens,
    token_subscript,
)
from native_schematics import Frame


ROOT = Path(__file__).resolve().parents[1]
PATH_NECESSITY = ROOT / "source_data" / "path_necessity_fashion"
SUBTREE = ROOT / "source_data" / "trained_subtree_address_full_factorial"

GREEN = COLORS["shunting"]
PURPLE = COLORS["oracle"]
MUTE = COLORS["mute"]
INK = COLORS["ink"]
GRAY_D = COLORS["point_mlp"]
AMBER = COLORS["local"]

K_TICKS = (1, 2, 4, 8)
# The branch count B is an ordinal design property, not a condition, so it
# wears a graded slate ramp (light -> dark) that repurposes no architecture
# or condition hue; conditions keep their manuscript-wide colors (green =
# correct routing, amber = the shared coordinate, gray = deranged).
B_STYLE = {
    2: ("#9AA5B4", "o"),
    4: ("#5F6B7E", "s"),
    8: ("#2E3947", "^"),
}


def bootstrap(values: np.ndarray, seed: int, draws: int = 20_000):
    """Seed-bootstrap mean and 95% interval used by the frozen analysis."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


# ── Shared dendritic-fan schematic vocabulary ───────────────────────────
# The branch-conflict task is a within-neuron story, so its schematics use
# the same tree grammar as every other figure: green dendrite strokes, open
# junction rings, the orange soma disc, blue presynaptic dots, and delivery
# arrows in the condition hues.  Both the main figure 4 and Supplementary
# Fig. S29 draw from these helpers.
DEND = COLORS["dend"]
SOMA = COLORS["soma"]
EXC = COLORS["exc"]
RIM = mix("ink", 30)
GHOST = mix("mute", 45)


def draw_conflict_neuron(frame, rect, *, conflict, selected=1,
                         n_branches=4):
    """One dendritic unit: B branches fan from the soma; context gates the
    selected branch's path to the soma.  Solid distal segments carry every
    input; the proximal segment is solid only for the gated branch."""
    x0, y0, w, h = rect
    cx = x0 + 0.46 * w
    y_soma = y0 + 0.14 * h
    y_mid = y0 + 0.46 * h
    y_tip = y0 + 0.74 * h
    tips = np.linspace(x0 + 0.10 * w, x0 + 0.82 * w, n_branches)
    for index, x_tip in enumerate(tips):
        chosen = index == selected
        opposite = conflict and not chosen
        x_mid = x_tip + 0.42 * (cx - x_tip)
        # distal segment: every branch carries its input
        frame.ax.plot([x_tip, x_mid], [y_tip, y_mid],
                      color=DEND if chosen else mix("dend", 55),
                      lw=LW_EDGE, solid_capstyle="round", zorder=3)
        # proximal segment: only the gated branch reaches the soma solidly
        line, = frame.ax.plot([x_mid, cx], [y_mid, y_soma],
                              color=GREEN if chosen else GHOST,
                              lw=LW_DATA if chosen else LW_HAIR,
                              solid_capstyle="round", zorder=3)
        if not chosen:
            line.set_dashes((2.0, 1.8))
        # junction ring at the gate point
        frame.disc((x_mid, y_mid), 1.9, fill="white",
                   edge=GREEN if chosen else mix("dend", 55), lw=LW_HAIR,
                   zorder=4)
        if chosen:
            frame.disc((x_mid, y_mid), 3.6, fill="none", edge=GREEN,
                       lw=LW_EDGE, zorder=4)
        # active input synapse and its stream label: the branches receive
        # class VIEWS, not labels, so the streams read x with a class
        # subscript rather than a bare target.
        frame.disc((x_tip, y_tip), 2.2, fill=EXC, zorder=4)
        stream_color = (COLORS["highlight"] if opposite
                        else (GREEN if chosen else MUTE))
        sub = "1−y" if opposite else "y"
        token_subscript(frame.ax,
                        x_tip - frame.fx(7.5 if opposite else 3.0),
                        y_tip + frame.fy(6.5), "x", sub,
                        size=PT_SMALL, sub_size=PT_SMALL,
                        color=stream_color, ha="left", va="bottom")
    # The double ring plus the tiny c badge is the CONTEXT-GATE glyph: the
    # badge distinguishes this hard selector from the conductance gain ring
    # of Fig. 1E, which shares only the ring geometry.
    gate_x = tips[selected] + 0.42 * (cx - tips[selected])
    frame.text((gate_x - frame.fx(4.6), y_mid + frame.fy(3.4)), "c",
               size=PT_SMALL, color=GREEN, ha="right", va="bottom",
               zorder=6)
    frame.disc((cx, y_soma), 3.8, fill=SOMA, edge=RIM, lw=LW_HAIR, zorder=5)
    frame.arrow((cx + frame.fx(5.0), y_soma),
                (cx + frame.fx(16.0), y_soma), color=MUTE, lw=LW_EDGE,
                head=3.4)
    frame.text((cx + frame.fx(19.5), y_soma), "z", size=PT_SMALL,
               color=INK, ha="left", va="center")
    return tips


def draw_credit_fan(frame, rect, *, mode, color, selected=1, n_branches=4):
    """Row-scale delivery glyph, in the manuscript's dendritic orientation.

    Soma at the bottom, B branches fanning upward through junction rings --
    the same grammar as :func:`draw_conflict_neuron` and as every tree in
    Figs. 1, 2 and 5.  The earlier version drew the soma at the LEFT with
    branches running right, so Figure 4 carried two different orientations
    of the same object on one page.

    ``mode``: "eligibility" puts a presynaptic dot on every tip; "selective"
    delivers one arrow to the selected branch; "shared" fans one source into
    every branch.
    """
    x0, y0, w, h = rect
    cx = x0 + 0.40 * w
    y_soma = y0 + 0.16 * h
    y_mid = y0 + 0.54 * h
    y_tip = y0 + 0.90 * h
    tips = np.linspace(x0 + 0.10 * w, x0 + 0.70 * w, n_branches)
    for x_tip in tips:
        x_mid = x_tip + 0.42 * (cx - x_tip)
        frame.ax.plot([x_tip, x_mid], [y_tip, y_mid], color=mix("dend", 45),
                      lw=LW_EDGE, solid_capstyle="round", zorder=3)
        frame.ax.plot([x_mid, cx], [y_mid, y_soma], color=mix("dend", 45),
                      lw=LW_EDGE, solid_capstyle="round", zorder=3)
        frame.disc((x_mid, y_mid), 1.5, fill="white",
                   edge=mix("dend", 45), lw=LW_HAIR, zorder=4)
        if mode == "eligibility":
            frame.disc((x_tip, y_tip), 1.9, fill=EXC, zorder=5)
    frame.disc((cx, y_soma), 2.9, fill=SOMA, edge=RIM, lw=LW_HAIR, zorder=5)

    # Delivery is drawn ABOVE the junction row.  All B junctions sit at one
    # height, so a fan of arrows from a single right-hand source crossed
    # itself and passed over junctions it was not addressing.
    mids = [x_tip + 0.42 * (cx - x_tip) for x_tip in tips]
    y_bus = y_mid + frame.fy(9.0)
    if mode == "selective":
        x_sel = mids[selected]
        frame.arrow((x_sel, y_bus), (x_sel, y_mid + frame.fy(2.6)),
                    color=color, lw=LW_EDGE, head=3.0)
    elif mode == "shared":
        # One source, tapped once per branch: a bus with a drop into each
        # junction says "the same delta reaches every branch" without four
        # crossing arrows.
        source = (x0 + 0.93 * w, y_bus)
        frame.ax.plot([mids[0], source[0]], [y_bus, y_bus], color=color,
                      lw=LW_HAIR, solid_capstyle="round", zorder=5)
        frame.disc(source, 1.6, fill=color, zorder=6)
        for x_mid in mids:
            frame.arrow((x_mid, y_bus), (x_mid, y_mid + frame.fy(2.6)),
                        color=color, lw=LW_HAIR, head=2.6)


# ── Branch-conflict panels ──────────────────────────────────────────────
def shared_mode_boundary(ax) -> None:
    """Analytic useful-signal boundary of the normalized shared coordinate."""
    chi = np.linspace(0.0, 1.0, 301)
    for branches, (color, marker) in B_STYLE.items():
        signal = 1.0 - 2.0 * (branches - 1) * chi / branches
        boundary = branches / (2.0 * (branches - 1))
        ax.plot(chi, signal, color=color, lw=LW_DATA, zorder=2)
        ax.plot([boundary], [0.0], marker=marker, color=color,
                markerfacecolor="white", markeredgewidth=LW_EDGE,
                ms=MARKER_MS, zorder=4)
        label_y = 1.0 - 2.0 * (branches - 1) / branches
        ax.text(1.035, label_y, f"B = {branches}", fontsize=PT_SMALL,
                color=color, ha="left", va="center", clip_on=False)
    # The zero reference stops at the data range so its dashes never strike
    # through the B = 2 label sitting on the y = 0 line beyond it.
    ax.plot([0.0, 1.0], [0.0, 0.0], color=MUTE, lw=LW_REF,
            dashes=(2.4, 2.0), zorder=0)
    ax.text(0.025, 0.055, "s(χ) = 1 − 2χ(B − 1)/B",
            transform=ax.transAxes, fontsize=PT_SMALL, color=INK,
            ha="left", va="bottom")
    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(-0.82, 1.08)
    ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
    ax.set_yticks([-0.5, 0.0, 1.0])
    ax.set_xlabel("credit conflict χ")
    ax.set_ylabel("shared useful signal")


def _boundary_token(ax, x_right, y_baseline, value, color) -> None:
    """Right-aligned 'chi-sub-c = value' set with a true dropped subscript.

    ``token_subscript`` chains every span on the *previous* span's bounding
    box, so its tail inherits the subscript's drop and the whole group reads
    as one lowered string.  Here each span takes its x from the neighbour it
    abuts and its y straight from the axes, so only the subscript leaves the
    base line.  Chaining leftward from ``x_right`` right-aligns the group
    without having to measure it.
    """
    gap_word, gap_sub, drop = 2.2, 0.4, 1.8   # points
    tail = ax.text(x_right, y_baseline, f"= {value}", transform=ax.transAxes,
                   fontsize=PT_SMALL, color=color, ha="right", va="baseline",
                   zorder=5, clip_on=False)
    sub = ax.annotate("c", xy=(0.0, y_baseline),
                      xycoords=(tail, ax.transAxes),
                      xytext=(-gap_word, -drop), textcoords="offset points",
                      fontsize=PT_SMALL, color=color, ha="right",
                      va="baseline", zorder=5, annotation_clip=False)
    ax.annotate("χ", xy=(0.0, y_baseline), xycoords=(sub, ax.transAxes),
                xytext=(-gap_sub, 0.0), textcoords="offset points",
                fontsize=PT_SMALL, color=color, ha="right", va="baseline",
                zorder=5, annotation_clip=False)


def path_accuracy_facets(host, summary: pd.DataFrame) -> None:
    """Three identical-scale facets; exact-equivalent curves appear once."""
    host.set_xlim(0, 1)
    host.set_ylim(0, 1)
    host.text(0.995, 1.000, "branch-specific = BP = gated point",
              fontsize=PT_SMALL, color=GREEN, ha="right", va="top")
    # Anchored on the host's own bottom edge this label rose into the three
    # facets' tick rows, overlapping five tick numbers by 4 pt.  The panel is
    # schematic=True, so the canvas collision check does not police it.
    host.text(0.515, -0.050, "credit-conflict probability χ",
              fontsize=PT_LABEL, color=INK, ha="center", va="bottom",
              clip_on=False)
    host.text(0.008, 0.515, "held-out accuracy", rotation=90,
              fontsize=PT_LABEL, color=INK, ha="center", va="center")

    xstarts = (0.070, 0.382, 0.694)
    for index, (branches, x0) in enumerate(zip((2, 4, 8), xstarts, strict=True)):
        color, marker = AMBER, "s"
        dash_color = B_STYLE[branches][0]
        ax = host.inset_axes([x0, 0.16, 0.286, 0.70], transform=host.transAxes)
        from figure_canvas import style_panel
        style_panel(ax, grid="none")
        correct = summary[
            summary.condition.eq("correct_path")
            & summary.branches.eq(branches)
        ].sort_values("conflict_probability")
        shared = summary[
            summary.condition.eq("neuron_shared_k1")
            & summary.branches.eq(branches)
        ].sort_values("conflict_probability")
        x = shared.conflict_probability.to_numpy(float)
        ax.fill_between(
            x,
            correct.ci95_low_test_accuracy.to_numpy(float),
            correct.ci95_high_test_accuracy.to_numpy(float),
            color=GREEN, alpha=0.13, linewidth=0, zorder=1,
        )
        ax.plot(x, correct.mean_test_accuracy, color=GREEN,
                lw=LW_DATA, zorder=3)
        ax.fill_between(
            x,
            shared.ci95_low_test_accuracy.to_numpy(float),
            shared.ci95_high_test_accuracy.to_numpy(float),
            color=color, alpha=0.11, linewidth=0, zorder=1,
        )
        ax.plot(x, shared.mean_test_accuracy, color=color, marker=marker,
                ms=MARKER_MS - 0.8, markerfacecolor="white",
                markeredgecolor=color, markeredgewidth=LW_EDGE,
                lw=LW_DATA, zorder=4)
        boundary = branches / (2.0 * (branches - 1))
        ax.axvline(boundary, color=dash_color, lw=LW_REF, alpha=0.72,
                   dashes=(2.4, 2.0), zorder=0)
        ax.axhline(0.5, color=MUTE, lw=LW_HAIR, alpha=0.8,
                   dashes=(2.0, 2.0), zorder=0)
        ax.text(0.04, 1.035, f"B = {branches}", transform=ax.transAxes,
                fontsize=PT_ANNOT, color=INK, ha="left", va="bottom",
                clip_on=False)
        # The predicted boundary value sits ABOVE the axes on its own dashed
        # line, where no curve can strike through it.
        # Side-adaptive anchor: a boundary near the right edge grows its
        # label leftward to end at the dashed line; earlier boundaries grow
        # rightward from it, so the label never reaches a neighbour's tag.
        # Anchor the boundary value in AXES fractions, not at the dashed line.
        # These labels are drawn with clip_on=False, so a data anchor that
        # tracked the boundary ran into a "B = n" tag whenever the boundary
        # sat far left or far right: at 0.67 it collided with the neighbouring
        # B=8 tag, and pushing it leftward then collided with its own B=4 tag.
        # A fixed right-of-tag anchor puts every facet's label in the same
        # place, clear of its own tag and unable to reach the next facet's.
        token_subscript(ax, 0.50, 1.035, "χ", "c",
                        f" = {boundary:.2f}", size=PT_SMALL,
                        sub_size=PT_SMALL, color=dash_color, ha="left",
                        va="bottom", clip_on=False, transform=ax.transAxes)
        ax.set_xlim(-0.025, 1.035)
        ax.set_ylim(0.18, 0.84)
        ax.set_xticks([0.0, 0.5, 1.0], ["0", "0.5", "1"])
        ax.set_yticks([0.2, 0.5, 0.8])
        if index:
            ax.tick_params(axis="y", labelleft=False)
        else:
            ax.text(0.02, 0.505, "chance", fontsize=PT_SMALL, color=MUTE,
                    ha="left", va="bottom")
            ax.text(0.05, 0.735, "neuron-shared", fontsize=PT_SMALL,
                    color=AMBER, ha="left", va="top")


def path_effect_summary(ax, contrasts: pd.DataFrame,
                        interactions: pd.DataFrame) -> None:
    """Paired endpoint effects at compatible and fully conflicting limits."""
    frame = contrasts[
        contrasts.contrast.eq("correct - shared")
        & contrasts.endpoint.eq("test_accuracy")
        & contrasts.conflict_probability.isin([0.0, 1.0])
    ]
    xpos = np.arange(3, dtype=float)
    specs = (
        (0.0, GRAY_D, "o", -0.055, "white"),
        (1.0, GREEN, "D", 0.055, GREEN),
    )
    seeds = pd.read_csv(PATH_NECESSITY / "seed_outcomes.csv")
    for dose, color, marker, offset, face in specs:
        rows = frame[np.isclose(frame.conflict_probability, dose)].set_index(
            "branches"
        ).loc[[2, 4, 8]]
        mean = 100.0 * rows.mean_difference.to_numpy(float)
        low = 100.0 * rows.ci95_low.to_numpy(float)
        high = 100.0 * rows.ci95_high.to_numpy(float)
        # Per-seed paired differences under the published mean, following
        # the manuscript-wide seed-cloud convention.
        for x_position, branches in zip(xpos, (2, 4, 8), strict=True):
            cell = seeds[
                seeds.branches.eq(branches)
                & np.isclose(seeds.conflict_probability, dose)
                & seeds.condition.isin(["correct_path", "neuron_shared_k1"])
            ].pivot(index="seed", columns="condition",
                    values="test_accuracy")
            diffs = 100.0 * (cell["correct_path"]
                             - cell["neuron_shared_k1"]).to_numpy(float)
            ax.scatter(
                np.full(diffs.size, x_position + offset)
                + np.linspace(-0.045, 0.045, diffs.size),
                diffs, s=SEED_MS ** 2, color=color, alpha=SEED_ALPHA,
                edgecolors="none", zorder=2,
            )
        ax.errorbar(
            xpos + offset, mean, yerr=[mean - low, high - mean],
            color=color, marker=marker, markerfacecolor=face,
            markeredgecolor=color if face == "white" else "white",
            markeredgewidth=LW_EDGE, ms=MARKER_MS, lw=LW_ERR,
            capsize=ERR_CAPSIZE, zorder=3,
        )
    ax.axhline(0.0, color=MUTE, lw=LW_REF, dashes=(2.4, 2.0), zorder=0)
    # The seed count is a statistic and belongs in the caption, which reports
    # "positive in every paired seed (20/20 at each B)".  Guard it here so the
    # figure fails loudly if the data stop supporting the caption's claim.
    positives = interactions.set_index("branches").loc[[2, 4, 8]].positive_pairs
    if not np.all(positives.to_numpy(int) == 20):
        raise ValueError(
            "Fig. 4E caption claims 20/20 positive interaction slopes at each "
            f"B; source table reports {positives.to_dict()}"
        )
    ax.text(0.04, 0.90, "χ = 1", transform=ax.transAxes,
            fontsize=PT_SMALL, color=GREEN, ha="left", va="top")
    ax.text(0.04, 0.16, "χ = 0", transform=ax.transAxes,
            fontsize=PT_SMALL, color=GRAY_D, ha="left", va="bottom")
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(-5.0, 66.0)
    ax.set_xticks(xpos, ["2", "4", "8"])
    ax.set_yticks([0, 30, 60])
    ax.set_xlabel("branches B")
    ax.set_ylabel("correct − shared (pp)")


# ── Hierarchical-routing panels ─────────────────────────────────────────
ADDRESS_YLIM = (-0.55, 3.62)
ADDRESS_ASPECT = ((_BASE_XLIM[1] - _BASE_XLIM[0])
                  / (ADDRESS_YLIM[1] - ADDRESS_YLIM[0]))
TREE_SCALE = 0.62
K1_CHAINS = (
    ((0.02, 0.34), "J1", "JL", "JLL", "T1"), ("JLL", "T2"),
    ("JL", "JLR", "T3"), ("JLR", "T4"),
    ("J1", "JR", "JRL", "T5"), ("JRL", "T6"),
    ("JR", "JRR", "T7"), ("JRR", "T8"),
)


def _tree_inset(frame: Frame, rect):
    x0, y0, width, height = rect
    width_pt, height_pt = width * frame.w_pt, height * frame.h_pt
    if width_pt / height_pt > ADDRESS_ASPECT:
        fit_h_pt, fit_w_pt = height_pt, height_pt * ADDRESS_ASPECT
    else:
        fit_w_pt, fit_h_pt = width_pt, width_pt / ADDRESS_ASPECT
    fit = (x0 + (width - frame.fx(fit_w_pt)) / 2.0,
           y0 + (height - frame.fy(fit_h_pt)) / 2.0,
           frame.fx(fit_w_pt), frame.fy(fit_h_pt))
    sub = frame.ax.inset_axes(fit, transform=frame.ax.transData, zorder=3)
    sub.set_facecolor("none")
    return sub


def _address_tree(ax, budget_k: int) -> None:
    if budget_k == 1:
        _setup_axes(ax, _BASE_XLIM, ADDRESS_YLIM)
        tree = _Tree(ax, TREE_SCALE, False)
        tree.capsule(mix("point_mlp", 15), 13, list(K1_CHAINS))
        tree.tree(COLORS["dend"])
        tree.junctions()
        tree.soma(COLORS["soma"], mix("ink", 30))
    else:
        draw_credit_tree(ax, mode="address", K=budget_k, scale=TREE_SCALE,
                         labels=False, xlim=_BASE_XLIM, ylim=ADDRESS_YLIM)
    enforce_tokens(ax)


def _share_tree_frame(subs) -> None:
    cy = 0.5 * (ADDRESS_YLIM[0] + ADDRESS_YLIM[1])
    half_w = max(max(abs(value) for value in sub.get_xlim()) for sub in subs)
    half_h = max(max(abs(value - cy) for value in sub.get_ylim()) for sub in subs)
    half_w = max(half_w, half_h * ADDRESS_ASPECT)
    half_h = half_w / ADDRESS_ASPECT
    for sub in subs:
        sub.set_aspect("auto")
        sub.set_xlim(-half_w, half_w)
        sub.set_ylim(cy - half_h, cy + half_h)


def address_ladder_compact(ax) -> None:
    """K=1,2,4,8 address fields in a consistent two-by-two tree grid."""
    frame = Frame(ax, labels=True, scale=0.86)
    cells = (
        (0.01, 0.52, 0.47, 0.46),
        (0.52, 0.52, 0.47, 0.46),
        (0.01, 0.02, 0.47, 0.46),
        (0.52, 0.02, 0.47, 0.46),
    )
    subs = []
    for rect, budget in zip(cells, K_TICKS, strict=True):
        x0, y0, width, height = rect
        emphasis = budget == 4
        frame.group(
            rect,
            tint=mix("shunting", 9) if emphasis else COLORS["panel_bg"],
            edge=mix("shunting", 45) if emphasis else COLORS["grid"],
        )
        frame.text((x0 + frame.fx(3.0), y0 + height - frame.fy(5.2)),
                   f"K = {budget}", size=PT_SMALL,
                   color=GREEN if emphasis else INK, ha="left", va="top")
        sub = _tree_inset(
            frame,
            (x0 + 0.06 * width, y0 + 0.04 * height,
             0.88 * width, 0.73 * height),
        )
        _address_tree(sub, int(budget))
        subs.append(sub)
    _share_tree_frame(subs)


def _best_control_by_budget(outcomes: pd.DataFrame):
    """Per-seed oracle over the four frozen matched non-anatomical controls."""
    controls = [
        "within_neuron_route_derangement",
        "depth_interleaved_bins",
        "random_sparse_matched",
        "random_rank_k",
    ]
    dendritic = outcomes[outcomes.architecture.eq("dendritic_tree")]
    rows = []
    for index, budget in enumerate(K_TICKS):
        wide = dendritic[
            dendritic.feedback_family.isin(controls)
            & dendritic.budget_k.eq(budget)
        ].pivot(index="seed", columns="feedback_family",
                values="heldout_accuracy")
        values = wide.max(axis=1).to_numpy(float)
        rows.append((budget, *bootstrap(values, 82_000 + index)))
    return np.asarray(rows, dtype=float)


def bandwidth_sweep_compact(ax, outcomes: pd.DataFrame,
                            dendritic: pd.DataFrame) -> None:
    """Correct routes, their strongest matched control and derangement."""
    x = np.arange(4, dtype=float)
    # At K=1 a single channel leaves no routing to get right, so the correct
    # and deranged means are identical (0.1866).  Equal-size opaque markers
    # made the later-drawn square hide the green circle completely, so the
    # correct series looked as though it had no K=1 point.  Draw the control
    # first and slightly larger, and the correct series last and smaller, so a
    # coincident pair reads as concentric rather than as one missing point.
    specs = (
        ("within_neuron_route_derangement", GRAY_D, "s", MARKER_MS + 1.0, 3),
        ("correct_ancestry_subtrees", GREEN, "o", MARKER_MS - 0.5, 5),
    )
    for family, color, marker, marker_size, layer in specs:
        part = dendritic[dendritic.feedback_family.eq(family)].set_index(
            "budget_k"
        ).loc[list(K_TICKS)]
        mean = part.mean_heldout_accuracy.to_numpy(float)
        low = part.ci95_low_heldout_accuracy.to_numpy(float)
        high = part.ci95_high_heldout_accuracy.to_numpy(float)
        ax.errorbar(
            x, mean, yerr=[mean - low, high - mean], color=color,
            marker=marker, markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=LW_EDGE, ms=marker_size,
            lw=LW_DATA, capsize=ERR_CAPSIZE, zorder=layer,
        )
    best = _best_control_by_budget(outcomes)
    # Correct equals the best control at K=8 to the last decimal, so this
    # series is drawn largest and lowest for the same reason as the control
    # above: the smaller green circle then sits inside it instead of erasing
    # it.  Markers stay at their true K, so a coincidence reads as one.
    ax.errorbar(
        x, best[:, 1], yerr=[best[:, 1] - best[:, 2],
                            best[:, 3] - best[:, 1]],
        color=PURPLE, marker="D", markerfacecolor="white",
        markeredgecolor=PURPLE, markeredgewidth=LW_EDGE,
        ms=MARKER_MS + 1.6, lw=LW_DATA, capsize=ERR_CAPSIZE, zorder=2,
    )
    ax.text(0.10, 0.74, "best control", color=PURPLE,
            fontsize=PT_SMALL, ha="left", va="bottom")
    ax.text(1.62, 0.845, "correct", color=GREEN,
            fontsize=PT_SMALL, ha="left", va="bottom")
    ax.text(2.95, 0.34, "deranged", color=GRAY_D,
            fontsize=PT_SMALL, ha="right", va="bottom")
    ax.set_xlim(-0.18, 3.18)
    ax.set_ylim(0.10, 0.88)
    ax.set_xticks(x, ["1", "2", "4", "8"])
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_xlabel("channels K")
    ax.set_ylabel("accuracy")


def route_contrasts_compact(ax, contrasts: pd.DataFrame) -> None:
    """The two route-assignment comparisons that determine the conclusion."""
    frame = contrasts[
        contrasts.architecture.eq("dendritic_tree")
        & contrasts.endpoint.eq("heldout_accuracy")
        & contrasts.contrast.isin([
            "correct - best_matched_nonanatomical_oracle",
            "correct - within_neuron_route_derangement",
        ])
    ]
    x = np.arange(4, dtype=float)
    specs = (
        ("correct - best_matched_nonanatomical_oracle", PURPLE, "D"),
        ("correct - within_neuron_route_derangement", GREEN, "o"),
    )
    for name, color, marker in specs:
        part = frame[frame.contrast.eq(name)].set_index("budget_k").loc[
            list(K_TICKS)
        ]
        mean = 100.0 * part.mean_difference.to_numpy(float)
        low = 100.0 * part.ci95_low.to_numpy(float)
        high = 100.0 * part.ci95_high.to_numpy(float)
        ax.errorbar(
            x, mean, yerr=[mean - low, high - mean], color=color,
            marker=marker, markerfacecolor="white", markeredgecolor=color,
            markeredgewidth=LW_EDGE, ms=MARKER_MS - 0.5,
            lw=LW_DATA, capsize=ERR_CAPSIZE, zorder=3,
        )
    ax.axhline(0.0, color=MUTE, lw=LW_REF, dashes=(2.4, 2.0), zorder=0)
    ax.text(1.85, 63.5, "vs deranged", color=GREEN, fontsize=PT_SMALL,
            ha="left", va="bottom")
    ax.text(0.03, -43.0, "vs best", color=PURPLE, fontsize=PT_SMALL,
            ha="left", va="bottom")
    ax.set_xlim(-0.18, 3.18)
    ax.set_ylim(-50.0, 70.0)
    ax.set_xticks(x, ["1", "2", "4", "8"])
    ax.set_yticks([-40, 0, 40])
    ax.set_xlabel("channels K")
    ax.set_ylabel("route effect (pp)")


def topology_alignment_compact(ax, outcomes: pd.DataFrame) -> None:
    """Mean paired matched-minus-rewired effect with bootstrap intervals."""
    x = np.arange(4, dtype=float)
    correct = outcomes[outcomes.feedback_family.eq("correct_ancestry_subtrees")]
    means, lows, highs = [], [], []
    for index, budget in enumerate(K_TICKS):
        left = correct[
            correct.architecture.eq("dendritic_tree")
            & correct.budget_k.eq(budget)
        ].set_index("seed").heldout_accuracy
        right = correct[
            correct.architecture.eq("degree_depth_matched_rewired_tree")
            & correct.budget_k.eq(budget)
        ].set_index("seed").heldout_accuracy
        mean, low, high = bootstrap((left - right).to_numpy(float),
                                    70_000 + index)
        means.append(100.0 * mean)
        lows.append(100.0 * low)
        highs.append(100.0 * high)
    means = np.asarray(means)
    lows = np.asarray(lows)
    highs = np.asarray(highs)
    ax.errorbar(
        x, means, yerr=[means - lows, highs - means], color=GREEN,
        marker="D", markerfacecolor="white", markeredgecolor=GREEN,
        markeredgewidth=LW_EDGE, ms=MARKER_MS, lw=LW_DATA,
        capsize=ERR_CAPSIZE, zorder=3,
    )
    ax.axhline(0.0, color=MUTE, lw=LW_REF, dashes=(2.4, 2.0), zorder=0)
    ax.set_xlim(-0.18, 3.18)
    ax.set_ylim(-3.0, 35.0)
    ax.set_xticks(x, ["1", "2", "4", "8"])
    ax.set_yticks([0, 10, 20, 30])
    ax.set_xlabel("channels K")
    ax.set_ylabel("topology effect (pp)")

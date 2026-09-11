#!/usr/bin/env python3
"""Build the native-vector Fashion-MNIST path-necessity component.

This is deliberately a standalone component: it does not alter Figure 3 or
the manuscript figure map.  Every mark is a matplotlib vector primitive and
every plotted value is read from ``source_data/path_necessity_fashion``.

Panels
------
A. All B branches receive nonzero views and a downstream context gate selects
   one branch for the somatic readout.  Nonselected
   views vary from class-compatible (alpha=0) to class-conflicting (alpha=1).
   Correct transport addresses the selected path; neuron-shared transport
   applies the same coordinate to every branch.
B. Held-out accuracy of the correct-path and neuron-shared rules across the
   frozen conflict doses.  The nearly coincident correct-path curves are
   represented by their joint 95% envelope to avoid three redundant traces.
   Coloured triangles and vertical rules mark the analytic shared-mode zeros.
C. Analytic boundaries are compared with the linearly interpolated conflict
   dose at which the mean trained shared-credit curve reaches chance.  This
   interpolation is descriptive and is computed directly from the plotted
   means; it is not a new inferential test.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyBboxPatch

from credit_tree_schematics import mix
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
    PT_LABEL,
    PT_LEGEND,
    PT_SMALL,
    SEED_MS,
    Margins,
    NativeCanvas,
)
from native_schematics import Frame
from routing_figure_panels import draw_conflict_neuron


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "path_necessity_fashion"
OUTPUT = ROOT / "figures" / "supplementary" / "figure_S29_panels_A-C.pdf"

CANVAS_H_PT = 530.0
HEIGHT_IN = CANVAS_H_PT / 72.0
ROW_WEIGHTS = (132.0, 132.0, 145.0)
HGUTTER_PT = 31.0
VGUTTER_PT = 37.0

INK = COLORS["ink"]
MUTE = COLORS["mute"]
GRID = COLORS["grid"]
GREEN = COLORS["shunting"]
AMBER = COLORS["local"]

# Branch count is an ordinal design property, not a condition: it wears the
# same graded slate ramp as main Fig. 4, repurposing no condition hue.
BRANCH_STYLES = {
    2: ("#9AA5B4", "o"),
    4: ("#5F6B7E", "s"),
    8: ("#2E3947", "^"),
}


def _round_box(ax, center, width, height, *, face, edge, lw=LW_EDGE,
               radius=0.012, zorder=3):
    """Rounded box in an axes-normalized schematic frame."""
    cx, cy = center
    patch = FancyBboxPatch(
        (cx - width / 2.0, cy - height / 2.0), width, height,
        boxstyle=f"round,pad=0.003,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=lw,
        transform=ax.transData, clip_on=False, zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def _draw_gate_card(frame: Frame, rect, *, conflict: bool) -> None:
    """One B=4 realization of the shared forward gate, drawn as the same
    dendritic unit the main figures use (branches, junction rings, soma)."""
    x0, y0, width, height = rect
    title = "conflicting, χ = 1" if conflict else "compatible, χ = 0"
    # Figure 4's _trial_card draws the same "conflicting" card in the
    # highlight pink; this sheet must share that vocabulary exactly, and
    # red-brown is reserved for exact backpropagation.
    title_color = COLORS["highlight"] if conflict else GREEN
    tint = mix("highlight", 7) if conflict else mix("shunting", 7)
    edge = mix("highlight", 38) if conflict else mix("shunting", 38)
    frame.group(rect, tint=tint, edge=edge)
    frame.text((x0 + 0.04 * width, y0 + 0.91 * height), title,
               size=PT_ANNOT, color=title_color, ha="left")
    draw_conflict_neuron(
        frame,
        (x0 + 0.08 * width, y0 + 0.06 * height,
         0.84 * width, 0.66 * height),
        conflict=conflict,
    )


def path_task_schematic(ax) -> None:
    """Panel A: compatibility sweep, downstream gate and route equations."""
    frame = Frame(ax, labels=True, scale=0.96)
    upper = (0.02, 0.58, 0.96, 0.39)
    middle = (0.02, 0.19, 0.96, 0.34)
    _draw_gate_card(frame, upper, conflict=False)
    _draw_gate_card(frame, middle, conflict=True)

    frame.text((0.02, 0.145), "same forward readout:  z = wᵀx of the gated branch",
               size=PT_ANNOT, color=INK, ha="left")
    frame.text((0.02, 0.085), "correct path", size=PT_SMALL,
               color=GREEN, ha="left")
    frame.text((0.32, 0.085),
               "δ·1[b = c]",
               size=PT_ANNOT, color=GREEN, ha="left")
    frame.text((0.02, 0.025), "neuron-shared", size=PT_SMALL,
               color=AMBER, ha="left")
    frame.text((0.32, 0.025), "δ/B per branch",
               size=PT_ANNOT, color=AMBER, ha="left")


def _accuracy_panel(ax, summary: pd.DataFrame) -> None:
    """Panel B: exact-path envelope and three shared-credit curves."""
    doses = np.sort(summary.conflict_probability.unique().astype(float))
    correct = summary[summary.condition.eq("correct_path")]
    lows, highs, centers = [], [], []
    for dose in doses:
        part = correct[np.isclose(correct.conflict_probability, dose)]
        lows.append(float(part.ci95_low_test_accuracy.min()))
        highs.append(float(part.ci95_high_test_accuracy.max()))
        centers.append(float(part.mean_test_accuracy.mean()))
    ax.fill_between(doses, lows, highs, color=GREEN, alpha=0.15,
                    linewidth=0, zorder=1)
    ax.plot(doses, centers, color=GREEN, lw=LW_DATA, zorder=3)

    for branches, (color, marker) in BRANCH_STYLES.items():
        part = summary[
            summary.condition.eq("neuron_shared_k1")
            & summary.branches.eq(branches)
        ].sort_values("conflict_probability")
        x = part.conflict_probability.to_numpy(float)
        y = part.mean_test_accuracy.to_numpy(float)
        lo = part.ci95_low_test_accuracy.to_numpy(float)
        hi = part.ci95_high_test_accuracy.to_numpy(float)
        ax.fill_between(x, lo, hi, color=color, alpha=0.10,
                        linewidth=0, zorder=1)
        ax.plot(x, y, color=color, marker=marker, ms=MARKER_MS - 0.6,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=LW_EDGE, lw=LW_DATA, zorder=4)
        boundary = branches / (2.0 * (branches - 1))
        # Clip the rule to the plotted band: a full-height axvline ran down
        # through the legend rows and cut their final glyphs.
        ax.axvline(boundary, color=color, lw=LW_REF, alpha=0.55,
                   ymin=0.30, ymax=1.0, dashes=(2.4, 2.0), zorder=0)
        ax.plot([boundary], [0.585], marker="v", ms=MARKER_MS - 0.2,
                color=color, zorder=5)
        ax.text(boundary, 0.604, f"{branches}", ha="center", va="bottom",
                fontsize=PT_SMALL, color=color)

    ax.axhline(0.5, color=MUTE, lw=LW_HAIR, dashes=(2.0, 2.0), zorder=0)
    ax.text(0.015, 0.505, "chance", fontsize=PT_SMALL, color=MUTE,
            ha="left", va="bottom")
    ax.text(0.25, 0.625, "triangles: predicted boundary (B)", fontsize=PT_SMALL,
            color=MUTE, ha="center", va="bottom")

    handles = [
        Line2D([], [], color=GREEN, lw=LW_DATA, label="correct path"),
        *[
            Line2D([], [], color=color, marker=marker, ms=MARKER_MS - 0.8,
                   markerfacecolor="white", markeredgewidth=LW_EDGE,
                   lw=LW_DATA, label=f"shared, B = {branches}")
            for branches, (color, marker) in BRANCH_STYLES.items()
        ],
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.01, 0.015),
              ncol=2, columnspacing=0.9, handlelength=1.5, handletextpad=0.4,
              borderaxespad=0.0, frameon=False, fontsize=PT_SMALL)
    ax.set_xlim(-0.025, 1.035)
    ax.set_ylim(0.18, 0.855)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.2, 0.5, 0.8])
    ax.set_xlabel("credit-conflict probability χ")
    ax.set_ylabel("held-out accuracy")


def _boundary_panel(ax, crossings: pd.DataFrame) -> None:
    """Panel C: analytic and observed order of the shared-credit transition."""
    crossings = crossings.sort_values("branches")
    branches = crossings.branches.to_numpy(int)
    x = np.arange(len(branches), dtype=float)
    predicted = crossings.predicted_boundary.to_numpy(float)
    observed = crossings.trained_mean_curve_chance_crossing.to_numpy(float)

    ax.plot(x, predicted, color=INK, lw=LW_REF,
            dashes=(2.4, 1.8), zorder=2)
    ax.plot(x, observed, color=AMBER, lw=LW_DATA, zorder=3)
    for xpos, theory, trained in zip(x, predicted, observed, strict=True):
        ax.plot([xpos, xpos], [theory, trained], color=MUTE, lw=LW_HAIR,
                alpha=0.7, zorder=1)
    ax.plot(x, predicted, linestyle="none", marker="D", ms=MARKER_MS,
            markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=LW_EDGE, zorder=4, label="analytic boundary")
    ax.plot(x, observed, linestyle="none", marker="o", ms=MARKER_MS,
            markerfacecolor=AMBER, markeredgecolor="white",
            markeredgewidth=LW_EDGE, zorder=5,
            label="trained chance crossing")

    # Values are shown once because all three comparisons are the conclusion
    # of this compact panel; the coloured series and marker shape carry the
    # identity, avoiding a second prose key inside the data region.
    for xpos, theory, trained in zip(x, predicted, observed, strict=True):
        ax.text(xpos - 0.07, theory + 0.025, f"{theory:.2f}",
                fontsize=PT_SMALL, color=INK,
                ha="right", va="bottom")
        # Placed below-right the label landed on the amber segment itself
        # (same hue over a 1.25 pt stroke); sit it above the point instead.
        ax.text(xpos + 0.055, trained + 0.030, f"{trained:.2f}",
                fontsize=PT_SMALL, color=AMBER, ha="left", va="bottom")

    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.98), frameon=False,
              handlelength=1.5, handletextpad=0.45, borderaxespad=0.0,
              fontsize=PT_SMALL)
    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0.50, 1.035)
    ax.set_xticks(x, [f"{branch}" for branch in branches])
    ax.set_yticks([0.50, 0.75, 1.00])
    ax.set_xlabel("branches receiving input B")
    ax.set_ylabel("conflict threshold")


def _boundary_test(ax, crossings: pd.DataFrame,
                   seed_boundaries: pd.DataFrame) -> None:
    """Observed chance crossing against the analytic boundary, per B.

    Local copy of the ``boundary_test`` panel this render was registered
    with (commit e30b10b of build_main_figure_04.py); the main-figure module
    since replaced it with ``boundary_order``, which has another signature
    and look, so the import it used to satisfy no longer exists.
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
    """Write the publication PDF and its 600-dpi review PNG."""
    mpl.rcParams["lines.markeredgewidth"] = LW_EDGE
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    crossings = pd.read_csv(SOURCE / "plotted_crossings.csv")
    seed_boundaries=pd.read_csv(SOURCE/"boundary_by_seed.csv")
    trajectories=pd.read_csv(ROOT/"source_data/review_branch_trajectories/gradient_trajectories.csv")
    required = {
        "correct_path", "neuron_shared_k1", "backpropagation",
        "gated_point_emulation", "within_neuron_deranged",
    }
    missing = required.difference(summary.condition.unique())
    if missing:
        raise ValueError(f"path-necessity source table lacks conditions: {missing}")

    canvas = NativeCanvas(
        HEIGHT_IN, 3, row_weights=list(ROW_WEIGHTS),
        hgutter_pt=HGUTTER_PT, vgutter_pt=VGUTTER_PT,
        margins=Margins(left=34.0, right=12.0, top=16.0, bottom=26.0),
        letters=False,
    )
    ax_a = canvas.panel(
        "A", 0, 0, 6, rowspan=2, schematic=True,
        title="Credit conflict creates path demand",
    )
    ax_b = canvas.panel(
        "B", 0, 6, 6, grid="none",
        title="Shared credit fails with conflict",
    )
    ax_c = canvas.panel(
        "C", 1, 6, 6, grid="y",
        title="Predicted and learned thresholds",
    )

    path_task_schematic(ax_a)
    _accuracy_panel(ax_b, summary)
    _boundary_test(ax_c,crossings,seed_boundaries)

    interval_rows=[]
    specs=[("common_exact_state",0,"#9AA5B4","-","common state, epoch 0"),
           ("common_exact_state",50,"#5F6B7E","-","common state, epoch 50"),
           ("common_exact_state",250,"#202936","-","common state, epoch 250"),
           ("own_state",250,AMBER,"--","own state, epoch 250")]
    # The eight sampled doses are drawn at equal spacing (an ordinal dose
    # axis) with chi as the tick label: on a linear axis five of them fall in
    # [0.5, 0.75] and their markers merge, while [0.75, 1] holds no sample.
    dose_levels=np.sort(trajectories.conflict_probability.unique().astype(float))
    dose_labels=["0","0.25","0.5","4/7","0.6","2/3","0.75","1"]
    assert np.allclose(dose_levels,[0,.25,.5,4/7,.6,2/3,.75,1],atol=1e-6)
    BOUNDARY=COLORS["oracle"]
    for col,branch,letter in [(0,2,"D"),(4,4,"E"),(8,8,"F")]:
        ax=canvas.panel(letter,2,col,4,grid="y",title=f"Update alignment, {branch} branches")
        for state,epoch,color,ls,label in specs:
            part=trajectories[trajectories.branches.eq(branch)&trajectories.condition.eq("neuron_shared_k1")
                              &trajectories.state_comparison.eq(state)&trajectories.epoch.eq(epoch)]
            xx=[];means=[];low=[];high=[]
            for dose,group in part.groupby("conflict_probability"):
                vals=group.sort_values("seed").gradient_cosine.to_numpy();assert len(vals)==20
                rng=np.random.default_rng(29000+branch*100+epoch+(state=="own_state"))
                lo,hi=np.quantile(rng.choice(vals,(5000,len(vals)),replace=True).mean(axis=1),[.025,.975])
                xx.append(dose);means.append(vals.mean());low.append(lo);high.append(hi)
                interval_rows.append(dict(branches=branch,epoch=epoch,state_comparison=state,
                                          conflict_probability=dose,mean_cosine=vals.mean(),ci95_low=lo,ci95_high=hi,n_seeds=20))
            pos=np.interp(xx,dose_levels,np.arange(len(dose_levels)))
            assert np.allclose(pos,np.round(pos))
            ax.plot(pos,means,color=color,ls=ls,marker="o",ms=2.8,lw=LW_DATA,label=label)
            ax.fill_between(pos,low,high,color=color,alpha=.10,lw=0)
        # Mean-field boundary chi_c = B / (2 (B - 1)) = 1, 2/3, 4/7: the odd
        # doses 4/7 and 2/3 were sampled to bracket it, so it is a tick.
        boundary=branch/(2.0*(branch-1))
        bpos=float(np.interp(boundary,dose_levels,np.arange(len(dose_levels))))
        assert np.isclose(bpos,round(bpos))
        ax.axvline(bpos,color=BOUNDARY,lw=LW_REF,dashes=(2.4,2.0),zorder=0,
                   label="mean-field boundary χc" if letter=="D" else None)
        ax.axhline(0,color=MUTE,lw=LW_REF,ls=":")
        ax.set_ylim(-1.05,1.08);ax.set_yticks([-1,0,1])
        ax.set_xlim(-0.4,len(dose_levels)-0.6)
        ax.set_xticks(np.arange(len(dose_levels)),dose_labels)
        ax.tick_params(axis="x",labelsize=PT_SMALL)
        ax.set_xlabel("credit conflict χ (sampled doses)")
        if letter=="D":
            ax.set_ylabel("cosine with exact update")
            ax.legend(loc="lower left",frameon=False,fontsize=PT_SMALL,
                       handlelength=1.2,labelspacing=.12,borderaxespad=.15,
                       handletextpad=.5)
        # Tag sits inside the axes beside the rule, on the side away from
        # the nearer x limit, where no curve reaches the top band.
        right=bpos<len(dose_levels)/2
        ax.text(bpos+(.14 if right else -.14),1.0,"χc",ha="left" if right else "right",
                va="top",fontsize=PT_SMALL,color=BOUNDARY)
        canvas.add_letter(letter,ax)
    pd.DataFrame(interval_rows).to_csv(ROOT/"source_data/review_branch_trajectories/figure_S29_gradient_intervals.csv",index=False)

    # A spans both rows.  The generic row-leading-letter alignment otherwise
    # places C at A's left edge, next to the wrong schematic.
    canvas.add_letter("A", ax_a)
    canvas.add_letter("B", ax_b)
    ax_c.text(-0.14, 1.14, "C", transform=ax_c.transAxes,
              fontsize=10.5, fontweight="bold", ha="left", va="bottom",
              color=INK, clip_on=False)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    return canvas.save(OUTPUT, name="figure_S29_panels_A-F")


def main() -> None:
    problems = build()
    for problem in problems:
        print(f"  {problem}")
    print(f"  canvas width {FIG_W * 72:.1f} pt, height {CANVAS_H_PT:.1f} pt")


if __name__ == "__main__":
    main()

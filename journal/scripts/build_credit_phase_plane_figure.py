#!/usr/bin/env python3
"""Build the credit phase-plane synthesis figure (fig_credit_phase_plane).

One panel places every feedback-routing experiment in the paper on a shared
plane: x is task--anatomy alignment (the fraction of task-gradient energy in
the anatomical route span) and y is feedback bandwidth relative to task
effective rank (K / r_eff, log scale).  Region tints are *theory-derived*
from the credit-operator bound and drawn as graded bands: their boundaries
are conceptual, not fitted to the plotted points.

Every plotted number is read from frozen source tables at build time; the
exact values and their provenance are also exported to
``source_data/credit_phase_plane/points.csv``.

Provenance of every number
--------------------------
Trained factorial (green circles, K in {1, 2, 4, 8}):
  x  = ``mean_initial_gradient_capture`` for architecture ``dendritic_tree``,
       family ``correct_ancestry_subtrees`` in
       source_data/trained_subtree_address_full_factorial/condition_summary.csv
       (capture of the ancestry span = task-gradient energy fraction in it).
  y  = K / r_eff.  r_eff = (sum lambda)^2 / sum lambda^2 over the task
       gradient-covariance spectrum in
       source_data/credit_phase_existing/task_spectra.csv (columns ``seed``,
       ``eigenvalue``; mean over the 20 frozen seeds; r_eff = 7.993).
  annotation = ``mean_difference`` of the heldout_accuracy contrast
       ``correct - best_matched_nonanatomical_oracle`` in
       source_data/trained_subtree_address_full_factorial/paired_contrasts.csv
       (K=1: -0.358, K=2: -0.154, K=4: +0.0127, K=8: 0.000 tie).

Spectral alignment sweep at K=4 (blue diamonds):
  x  = imposed mixture alignment ``alignment`` (0, 0.25, 0.5, 0.75, 1).
  y  = 4 / r_eff(alignment).  r_eff is estimated per seed from the dense-PCA
       cumulative captures at ranks {1, 2, 4, 8, 16} in
       source_data/credit_phase_theory/spectral_phase_seed.csv (method
       ``dense_pca_upper_bound``, column ``spectral_capture``), assuming the
       spectrum is uniform within each rank group; mean of 50 seeds.  At the
       mixture endpoints this recovers the designed spectrum exactly
       (r_eff = 5.000); at rho = 0.5 the mixture decoheres to r_eff = 7.79.
  annotation = ancestry-minus-random ``mean_spectral_capture`` difference at
       budget_k = 4 in
       source_data/credit_phase_theory/spectral_phase_summary.csv
       (-0.011, +0.130, +0.271, +0.411, +0.552).

MICrONS alignment-controlled (violet squares, K = 8 channels):
  x  = imposed ``alignment`` energy fraction (0, 0.2, ..., 1.0) from
       source_data/alignment_controlled/cell_paired_contrasts.csv.
  y  = 8 / r_eff(a) per cell, aggregated by geometric mean over the eight
       cells.  The gradient-family spectrum is not stored per se, but the
       generator (scripts/run_alignment_controlled_learning.py,
       ``alignment_controlled_fields``) draws fields isotropically inside the
       8-dim morphology span and isotropically in its orthogonal complement,
       so the population covariance has 8 modes at a/8 and (n-8) modes at
       (1-a)/(n-8), giving K/r_eff = a^2 + 8 (1-a)^2 / (n-8) with n =
       ``n_weighted_coordinates`` per cell from
       source_data/alignment_controlled/summary.json (dictionary_metadata).
       This y is therefore *derived from the documented generator*, not read
       from a results table.
  annotation = mean over cells of ``morphology_minus_control_credit_capture``
       against the ``random paths`` control in
       source_data/alignment_controlled/cell_paired_contrasts.csv
       (-0.135 at a=0 up to +0.710 at a=1).

Measured responses, MICrONS visual task (gray X, K = 4 channels):
  x  = mean ``heldout_credit_capture`` of ``morphology-aware paths`` over the
       7 target cells in source_data/figure5/task_target_method_means_ch4.csv
       (0.4745; the shuffled-ancestry control is 0.4463 -- the null).
  y  = 4 / r_eff.  r_eff = 1.072 is estimated from the dense-PCA-oracle mean
       captures at 1/2/4/8 channels (0.9654 / 0.9915 / 0.9987 / 1.0000) in
       source_data/figure5/task_target_method_means_ch{1,2,4,8}.csv, with the
       same within-group uniformity assumption (bounds 1.0715--1.0722, so the
       estimate is tight): the measured-response credit family is nearly
       rank-1, which places this null in the spans-coincide region.

Credit reversal (red-brown triangle, K = 2):
  x  = ``mean_initial_gradient_scaled_capture`` of ``correct_subtree_k2`` in
       source_data/trained_subtree_address/condition_summary.csv (1.000).
  y  = 2 / r_eff.  r_eff = 1/(c^2 + (1-c)^2) = 1.999 with c = 0.5095, the
       capture of the single shared neuronal coordinate
       (``neuron_shared_k1``, same file): two subtree coordinates capture
       1.000, so the credit family is rank-2 with an approximately even
       energy split.  Derived, and flagged as such in points.csv.
  annotation = ``mean_difference`` (+0.0439, 10/10 seeds) of the
       test_accuracy contrast ``correct - neuron shared`` in
       source_data/trained_subtree_address/paired_contrasts.csv.

Markers inside the small cluster at (1, ~1) are dodged horizontally for
legibility; the exported points.csv keeps the exact coordinates.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

from journal_style import (
    COLORS,
    FIG_W,
    LW_HAIR,
    LW_REF,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    PT_TICK,
    PT_TITLE,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    panel_title,
    style_axis,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
FACTORIAL = SOURCE / "trained_subtree_address_full_factorial"
EXISTING = SOURCE / "credit_phase_existing"
THEORY = SOURCE / "credit_phase_theory"
CONTROLLED = SOURCE / "alignment_controlled"
MEASURED = SOURCE / "figure5"
REVERSAL = SOURCE / "trained_subtree_address"
OUT = SOURCE / "credit_phase_plane"
FIGURES = ROOT / "figures" / "generated"

FAMILY_COLORS = {
    "factorial": COLORS["shunting"],
    "sweep": COLORS["additive"],
    "microns": COLORS["oracle"],
    "measured": COLORS["point_mlp"],
    "reversal": COLORS["bp"],
}
FAMILY_MARKERS = {
    "factorial": "o",
    "sweep": "D",
    "microns": "s",
    "measured": "X",
    "reversal": "^",
}


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, name)
    audit_text_over_data(fig, name)
    fig.savefig(
        FIGURES / f"{name}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / f"{name}.png", dpi=600)
    plt.close(fig)


def effective_rank(eigenvalues: np.ndarray) -> float:
    lam = np.asarray(eigenvalues, dtype=float)
    return float(lam.sum() ** 2 / np.sum(lam**2))


def grouped_effective_rank(cumulative: dict[int, float], total_rank: int) -> float:
    """r_eff from cumulative captures at nested ranks, uniform within groups."""
    ranks = sorted(cumulative)
    fractions: list[float] = []
    previous_rank, previous_capture = 0, 0.0
    for rank in ranks:
        block = rank - previous_rank
        fractions += [(cumulative[rank] - previous_capture) / block] * block
        previous_rank, previous_capture = rank, cumulative[rank]
    if total_rank > previous_rank:
        block = total_rank - previous_rank
        fractions += [max(0.0, 1.0 - previous_capture) / block] * block
    return effective_rank(np.asarray(fractions))


# ── Point families, straight from the frozen tables ───────────────────────


def factorial_points() -> list[dict]:
    spectra = pd.read_csv(EXISTING / "task_spectra.csv")
    r_eff = float(
        spectra.groupby("seed")["eigenvalue"].apply(effective_rank).mean()
    )
    summary = pd.read_csv(FACTORIAL / "condition_summary.csv")
    summary = summary[
        summary.architecture.eq("dendritic_tree")
        & summary.feedback_family.eq("correct_ancestry_subtrees")
    ]
    contrasts = pd.read_csv(FACTORIAL / "paired_contrasts.csv")
    contrasts = contrasts[
        contrasts.architecture.eq("dendritic_tree")
        & contrasts.contrast.eq("correct - best_matched_nonanatomical_oracle")
        & contrasts.endpoint.eq("heldout_accuracy")
    ].set_index("budget_k")
    points = []
    for row in summary.itertuples():
        k = int(row.budget_k)
        delta = float(contrasts.loc[k, "mean_difference"])
        tie = int(contrasts.loc[k, "ties"]) == int(contrasts.loc[k, "n_pairs"])
        points.append(
            {
                "family": "factorial",
                "label": f"factorial K={k}",
                "x": float(row.mean_initial_gradient_capture),
                "y": k / r_eff,
                "value": delta,
                "value_definition": (
                    "heldout accuracy, ancestry minus best "
                    "bandwidth-matched non-anatomical control"
                ),
                "outcome": "tie" if tie else ("win" if delta > 0 else "loss"),
                "derived": f"y = K/r_eff with r_eff = {r_eff:.3f} "
                "from task_spectra.csv",
                "source": "trained_subtree_address_full_factorial/"
                "condition_summary.csv + paired_contrasts.csv; "
                "credit_phase_existing/task_spectra.csv",
            }
        )
    return points


def sweep_points() -> list[dict]:
    seed = pd.read_csv(THEORY / "spectral_phase_seed.csv")
    dense = seed[seed.method.eq("dense_pca_upper_bound")]
    dense = dense.pivot_table(
        index=["seed", "alignment"], columns="budget_k", values="spectral_capture"
    )
    r_eff = (
        dense.apply(
            lambda row: grouped_effective_rank(
                {k: float(row[k]) for k in (1, 2, 4, 8, 16)}, 16
            ),
            axis=1,
        )
        .groupby(level="alignment")
        .mean()
    )
    summary = pd.read_csv(THEORY / "spectral_phase_summary.csv")
    summary = summary[summary.budget_k.eq(4)]
    capture = summary.pivot_table(
        index="alignment", columns="method", values="mean_spectral_capture"
    )
    points = []
    for rho in sorted(capture.index):
        delta = float(capture.loc[rho, "ancestry"] - capture.loc[rho, "random_rank"])
        points.append(
            {
                "family": "sweep",
                "label": f"spectral sweep rho={rho:g}",
                "x": float(rho),
                "y": 4.0 / float(r_eff.loc[rho]),
                "value": delta,
                "value_definition": "spectral capture, ancestry minus random rank-4",
                "outcome": "win" if delta > 0 else "loss",
                "derived": "y = 4/r_eff(rho) estimated from dense-PCA cumulative "
                f"captures (r_eff = {float(r_eff.loc[rho]):.3f})",
                "source": "credit_phase_theory/spectral_phase_summary.csv + "
                "spectral_phase_seed.csv",
            }
        )
    return points


def microns_points() -> list[dict]:
    meta = json.loads((CONTROLLED / "summary.json").read_text())
    channels = int(meta["channels"])
    coordinates = np.array(
        [entry["n_weighted_coordinates"] for entry in meta["dictionary_metadata"]],
        dtype=float,
    )
    paired = pd.read_csv(CONTROLLED / "cell_paired_contrasts.csv")
    paired = paired[paired.control.eq("random paths")]
    advantage = paired.groupby("alignment")[
        "morphology_minus_control_credit_capture"
    ].mean()
    points = []
    for a in sorted(advantage.index):
        # Population K/r_eff of the documented field generator, per cell.
        k_over_r = a**2 + channels * (1.0 - a) ** 2 / (coordinates - channels)
        y = float(np.exp(np.mean(np.log(k_over_r))))
        delta = float(advantage.loc[a])
        points.append(
            {
                "family": "microns",
                "label": f"MICrONS controlled a={a:g}",
                "x": float(a),
                "y": y,
                "value": delta,
                "value_definition": "field capture, ancestry routes minus random paths "
                "(mean of 8 cells)",
                "outcome": "win" if delta > 0 else "loss",
                "derived": "y = geometric-mean over cells of a^2 + "
                "8(1-a)^2/(n-8), the population K/r_eff of the documented "
                "field generator (not read from a results table)",
                "source": "alignment_controlled/cell_paired_contrasts.csv + "
                "summary.json (n_weighted_coordinates)",
            }
        )
    return points


def measured_point() -> dict:
    oracle = {}
    for ch in (1, 2, 4, 8):
        table = pd.read_csv(MEASURED / f"task_target_method_means_ch{ch}.csv")
        oracle[ch] = float(
            table[table.method.eq("dense PCA oracle")].heldout_credit_capture.mean()
        )
    r_eff = grouped_effective_rank(oracle, 8)
    table = pd.read_csv(MEASURED / "task_target_method_means_ch4.csv")
    morphology = float(
        table[table.method.eq("morphology-aware paths")].heldout_credit_capture.mean()
    )
    shuffle = float(
        table[table.method.eq("shuffled ancestry")].heldout_credit_capture.mean()
    )
    return {
        "family": "measured",
        "label": "measured responses (null)",
        "x": morphology,
        "y": 4.0 / r_eff,
        "value": morphology - shuffle,
        "value_definition": "heldout field capture, ancestry routes (0.475) minus "
        "shuffled ancestry (0.446); interval crosses zero",
        "outcome": "null",
        "derived": "y = 4/r_eff with r_eff = "
        f"{r_eff:.3f} estimated from dense-oracle captures at 1/2/4/8 channels",
        "source": "figure5/task_target_method_means_ch{1,2,4,8}.csv",
    }


def reversal_point() -> dict:
    summary = pd.read_csv(REVERSAL / "condition_summary.csv").set_index("condition")
    shared = float(
        summary.loc["neuron_shared_k1", "mean_initial_gradient_scaled_capture"]
    )
    correct = float(
        summary.loc["correct_subtree_k2", "mean_initial_gradient_scaled_capture"]
    )
    contrasts = pd.read_csv(REVERSAL / "paired_contrasts.csv")
    row = contrasts[
        contrasts.contrast.eq("correct - neuron shared")
        & contrasts.endpoint.eq("test_accuracy")
    ].iloc[0]
    r_eff = 1.0 / (shared**2 + (1.0 - shared) ** 2)
    return {
        "family": "reversal",
        "label": "credit reversal K=2",
        "x": correct,
        "y": 2.0 / r_eff,
        "value": float(row.mean_difference),
        "value_definition": "test accuracy, correct two-subtree routing minus "
        "one shared neuronal coordinate (10/10 seeds)",
        "outcome": "win",
        "derived": "y = 2/r_eff with r_eff = 1/(c^2+(1-c)^2) = "
        f"{r_eff:.3f}, c = {shared:.4f} the shared-coordinate capture "
        "(rank-2 family, near-even split); derived, not read from a table",
        "source": "trained_subtree_address/condition_summary.csv + "
        "paired_contrasts.csv",
    }


# ── Region tints: graded bands from the credit-operator bound ─────────────


def region_image(xlim, ylim, n: int = 480) -> np.ndarray:
    """RGBA tint field in axes fractions; boundaries are soft by design."""

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    fx = np.linspace(0.0, 1.0, n)[None, :]
    fy = np.linspace(0.0, 1.0, n)[:, None]
    x = xlim[0] + fx * (xlim[1] - xlim[0])
    u = np.log(ylim[0]) + fy * (np.log(ylim[1]) - np.log(ylim[0]))

    weight_bottom = sigmoid((np.log(0.28) - u) / 0.24) + 0.0 * x
    weight_top = sigmoid((u - np.log(1.0)) / 0.24) + 0.0 * x
    weight_middle = np.clip(1.0 - weight_bottom - weight_top, 0.0, 1.0)
    right = sigmoid((x - 0.45) / 0.10) + 0.0 * u
    weights = {
        "bottom": weight_bottom,
        "mid_left": weight_middle * (1.0 - right),
        "mid_right": weight_middle * right,
        "top": weight_top,
    }
    tints = {
        "bottom": (COLORS["local"], 0.14),
        "mid_left": (COLORS["point_mlp"], 0.11),
        "mid_right": (COLORS["shunting"], 0.15),
        "top": (COLORS["bp"], 0.09),
    }
    image = np.zeros((n, n, 4))
    for name, weight in weights.items():
        color, alpha = tints[name]
        rgb = np.asarray(to_rgb(color))
        image[..., 3] += weight * alpha
        image[..., :3] += weight[..., None] * alpha * rgb[None, None, :]
    scale = np.maximum(image[..., 3:], 1e-9)
    image[..., :3] /= scale
    return image


# ── Figure assembly ────────────────────────────────────────────────────────


def annotate(ax, text, xy, offset, *, color=None, ha="left", va="center",
             fontsize=PT_SMALL):
    ax.annotate(
        text,
        xy,
        xytext=offset,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=COLORS["ink"] if color is None else color,
    )


def marker_kwargs(family: str, outcome: str, size: float = 6.4) -> dict:
    color = FAMILY_COLORS[family]
    open_marker = outcome == "loss"
    return {
        "marker": FAMILY_MARKERS[family],
        "markersize": size,
        "markerfacecolor": "white" if open_marker else color,
        "markeredgecolor": color,
        "markeredgewidth": 1.1,
        "linestyle": "none",
        "zorder": 6,
    }


def main() -> None:
    apply_neurips_style()

    points = (
        factorial_points()
        + sweep_points()
        + microns_points()
        + [measured_point(), reversal_point()]
    )
    frame = pd.DataFrame(points)

    xlim = (-0.045, 1.14)
    ylim = (0.088, 6.2)

    fig = plt.figure(figsize=(FIG_W, 4.55))
    ax = fig.add_axes([0.075, 0.145, 0.600, 0.760])
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.imshow(
        region_image(xlim, ylim),
        transform=ax.transAxes,
        extent=(0.0, 1.0, 0.0, 1.0),
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        zorder=0,
    )

    # K = r reference line: a definition, not a fitted boundary.
    ax.axhline(1.0, color=COLORS["mute"], lw=LW_HAIR, ls=(0, (1, 2)), zorder=1)
    annotate(ax, "K = r", (xlim[0], 1.0), (4, 6), color=COLORS["mute"],
             fontsize=PT_SMALL)

    # Region labels (mute ink; the bands are theory-derived and schematic).
    region_label = dict(color=COLORS["mute"], fontsize=PT_ANNOT, zorder=2,
                        linespacing=1.25)
    ax.text(0.60, 0.130, "neuron identity binds:\ntoo few feedback channels",
            ha="center", va="center", style="italic", **region_label)
    ax.text(0.025, 0.435, "non-anatomical\nlow-rank routes win",
            ha="left", va="center", style="italic", **region_label)
    ax.text(0.995, 0.325, "anatomical routes win\n(operating regime)",
            ha="right", va="center", style="italic", **region_label)
    ax.text(0.025, 3.1,
            "spans coincide: anatomy ties unconstrained feedback\n"
            "and stochastic BP; only noise rejection differs",
            ha="left", va="center", style="italic", **region_label)

    # Display positions: dodge the three points that coincide at (1, ~1).
    dodge = {
        "factorial K=8": (-0.060, 0.0),
        "MICrONS controlled a=1": (0.0, 0.0),
        "credit reversal K=2": (0.060, 0.0),
    }
    frame["x_plot"] = frame.x + frame.label.map(
        lambda s: dodge.get(s, (0.0, 0.0))[0]
    )
    frame["y_plot"] = frame.y * np.exp(
        frame.label.map(lambda s: dodge.get(s, (0.0, 0.0))[1])
    )

    # Series hairlines connect each experiment family.
    for family in ("factorial", "sweep", "microns"):
        sub = frame[frame.family.eq(family)].sort_values("x")
        ax.plot(sub.x_plot, sub.y_plot, color=FAMILY_COLORS[family],
                lw=LW_HAIR, alpha=0.55, zorder=3)

    for row in frame.itertuples():
        ax.plot([row.x_plot], [row.y_plot],
                **marker_kwargs(row.family, row.outcome))

    # ── Per-point annotations (selective, short) ──────────────────────────
    get = frame.set_index("label")

    def at(label):
        row = get.loc[label]
        return float(row.x_plot), float(row.y_plot)

    green = FAMILY_COLORS["factorial"]
    annotate(ax, "K=1: $-$0.36", at("factorial K=1"), (7, -6), color=green)
    annotate(ax, "K=2: $-$0.15", at("factorial K=2"), (-7, 2), ha="right",
             color=green)
    annotate(ax, "K=4: +0.013", at("factorial K=4"), (-2, 16), ha="center",
             color=green)
    annotate(ax, "K=8: tie", at("factorial K=8"), (-2, 9), ha="right",
             color=green)

    blue = FAMILY_COLORS["sweep"]
    annotate(ax, r"$\rho$=0: $-$0.01", at("spectral sweep rho=0"), (6, 7),
             color=blue)
    annotate(ax, r"$\rho$=1: +0.55", at("spectral sweep rho=1"), (0, -10),
             ha="center", color=blue)

    violet = FAMILY_COLORS["microns"]
    annotate(ax, "a=0: $-$0.13", at("MICrONS controlled a=0"), (7, 1),
             color=violet)
    annotate(ax, "a=1: +0.71", at("MICrONS controlled a=1"), (2, 9),
             ha="left", color=violet)

    annotate(ax, "measured responses: null\n(ancestry 0.475 vs shuffle 0.446)",
             at("measured responses (null)"), (0, 11), ha="center",
             color=COLORS["mute"])
    # The reversal label moves to the free lower-right corner; a hairline
    # leader (drawn separately so its endpoints stay clear of the text box)
    # ties it back to the dodged triangle.
    reversal_x, reversal_y = at("credit reversal K=2")
    ax.text(1.115, 0.56, "routing necessary:\n+4.4 pts", ha="right", va="top",
            fontsize=PT_SMALL, color=FAMILY_COLORS["reversal"], zorder=5)
    ax.plot([reversal_x + 0.004, 1.082], [reversal_y * 0.93, 0.60],
            color=FAMILY_COLORS["reversal"], lw=LW_HAIR, alpha=0.6, zorder=4)

    # ── Axes cosmetics ─────────────────────────────────────────────────────
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.125, 0.25, 0.5, 1.0, 2.0, 4.0])
    ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel("task–anatomy alignment\n"
                  "(fraction of task-gradient energy in the anatomical route span)")
    ax.set_ylabel("feedback bandwidth / task rank  ($K/r_{\\mathrm{eff}}$)")
    panel_title(ax, "O", "Alignment × bandwidth phase plane")
    style_axis(ax)

    # ── Legend column ──────────────────────────────────────────────────────
    def handle(family, label, outcome="win", with_line=True):
        kwargs = marker_kwargs(family, outcome, size=5.4)
        kwargs["linestyle"] = "-" if with_line else "none"
        kwargs["lw"] = LW_HAIR
        kwargs["color"] = FAMILY_COLORS[family]
        return Line2D([], [], label=label, **kwargs)

    neutral = dict(marker="o", markersize=5.4, markeredgecolor=COLORS["edge"],
                   markeredgewidth=1.1, linestyle="none")
    handles = [
        handle("factorial", "Trained factorial, $K\\in\\{1,2,4,8\\}$"),
        handle("sweep", "Spectral sweep, $K$=4 (theory)"),
        handle("microns", "MICrONS controlled, $K$=8"),
        handle("measured", "Measured responses, $K$=4",
               outcome="null", with_line=False),
        handle("reversal", "Credit reversal, $K$=2", with_line=False),
        Line2D([], [], markerfacecolor=COLORS["edge"],
               label="anatomical routes win / tie", **neutral),
        Line2D([], [], markerfacecolor="white",
               label="anatomical routes lose", **neutral),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.005),
        frameon=False,
        fontsize=PT_LEGEND,
        handlelength=1.4,
        handletextpad=0.7,
        labelspacing=0.85,
        borderaxespad=0.0,
    )
    legend.set_zorder(7)

    fig.text(
        0.693, 0.360,
        "Bands: credit-operator regimes\n"
        "(conceptual boundaries, not fitted).\n\n"
        "Labels: ancestry-minus-best-control\n"
        "endpoint. Exact point coordinates and\n"
        "display offsets are in Source Data.",
        ha="left", va="top", fontsize=PT_SMALL, color=COLORS["mute"],
        linespacing=1.45,
    )

    # ── Export the plotted points with provenance ─────────────────────────
    OUT.mkdir(parents=True, exist_ok=True)
    frame[
        ["label", "family", "x", "y", "x_plot", "y_plot", "value",
         "value_definition", "outcome", "derived", "source"]
    ].to_csv(OUT / "points.csv", index=False, float_format="%.6g")

    save(fig, "fig_credit_phase_plane")


if __name__ == "__main__":
    main()

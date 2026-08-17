#!/usr/bin/env python3
"""Render the unified credit-operator phase figure from source tables."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from journal_style import (
    COLORS,
    DIV_CMAP,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    MARKERS,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    SEQ_CMAP,
    add_colorbar,
    annotate_heatmap,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "source_data" / "credit_phase_theory"
EXISTING = ROOT / "source_data" / "credit_phase_existing"
FACTORIAL = ROOT / "source_data" / "trained_subtree_address_full_factorial"
FIGURES = ROOT / "figures" / "generated"


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


class _MinusFmt:
    """Format heatmap cell values with a true minus sign (and no "-0")."""

    def __init__(self, spec: str = "{:.2f}", half_away: bool = False):
        self.spec = spec
        self.half_away = half_away

    def format(self, value: float) -> str:
        v = float(value)
        if self.half_away:
            v = math.copysign(math.floor(abs(v) + 0.5), v)
        s = self.spec.format(v)
        if s.lstrip("-").strip("0").strip(".") == "":
            s = s.lstrip("-")
        return s.replace("-", "−")


def heatmap_with_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    *,
    cmap,
    norm=None,
    fmt: str = "{:.2f}",
    half_away: bool = False,
    label: str = "",
):
    """Journal heatmap: shrink the axes to keep the slim colorbar in-panel."""
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.84, box.height])
    image = ax.imshow(matrix, origin="lower", aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(xlabels)), xlabels)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    annotate_heatmap(ax, image, matrix, fmt=_MinusFmt(fmt, half_away),
                     fontsize=PT_SMALL)
    add_colorbar(fig, ax, image, label=label)
    return image


def operator_schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "A", "Credit-operator utility")

    # Recessive card grouping the utility expression: panel_bg fill with a
    # hairline grid-tone outline, so the math reads as one designed unit
    # beside the flow column without competing with the stage boxes.
    ax.add_patch(
        FancyBboxPatch(
            (0.418, 0.35), 0.562, 0.43, boxstyle="round,pad=0.015",
            facecolor=COLORS["panel_bg"], edgecolor=COLORS["grid"],
            lw=LW_HAIR, zorder=0.5,
        )
    )

    # Left column: the three-stage flow, evenly spaced, consistent arrows.
    box_x, box_w, box_h = 0.02, 0.34, 0.20
    boxes = [
        (0.76, "task\ngradient $g$", COLORS["oracle"]),
        (0.45, "route/gain\n$M$", COLORS["additive"]),
        (0.14, "update\n$-\\eta M(g{+}\\xi)$", COLORS["shunting"]),
    ]
    for y, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (box_x, y), box_w, box_h, boxstyle="round,pad=0.015",
                facecolor="white", edgecolor=color, lw=LW_DATA,
            )
        )
        ax.text(box_x + box_w / 2, y + box_h / 2, label, ha="center",
                va="center", fontsize=PT_SMALL, color=color, linespacing=1.3)
    arrow_x = box_x + box_w / 2
    for top, bottom in ((0.743, 0.667), (0.433, 0.357)):
        ax.add_patch(
            FancyArrowPatch(
                (arrow_x, top), (arrow_x, bottom), arrowstyle="-|>",
                mutation_scale=7, lw=LW_REF, color=COLORS["mute"],
            )
        )

    # Right column: the utility as a stacked fraction with an explicit rule.
    eq_x = 0.70
    ax.text(eq_x, 0.70, r"guarantee $\propto$", ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["ink"])
    ax.text(eq_x, 0.565, r"$[g^{\mathrm{T}}Mg]^{2}$", ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["ink"])
    ax.plot([0.44, 0.96], [0.505, 0.505], color=COLORS["ink"], lw=LW_HAIR)
    ax.text(eq_x, 0.435,
            r"$\|Mg\|^{2}{+}\mathrm{tr}(M\Sigma M^{\mathrm{T}})$",
            ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["ink"])
    ax.text(0.5, 0.035, "signal² / (gain + noise)",
            ha="center", va="center", fontsize=PT_SMALL, style="italic",
            color=COLORS["mute"])


def main() -> None:
    apply_neurips_style()
    spectral = pd.read_csv(PHASE / "spectral_phase_summary.csv")
    depth = pd.read_csv(PHASE / "depth_training_summary.csv")
    projection = pd.read_csv(PHASE / "projection_phase_summary.csv")
    reliability = pd.read_csv(PHASE / "reliability_phase_summary.csv")
    span = pd.read_csv(PHASE / "same_span_diagnostics.csv")
    operator = pd.read_csv(EXISTING / "operator_metrics.csv")
    outcomes = pd.read_csv(FACTORIAL / "seed_outcomes.csv")
    audit = json.loads((EXISTING / "summary.json").read_text())

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(FIG_W, 6.72),
        gridspec_kw={
            "left": 0.08,
            "right": 0.985,
            "bottom": 0.07,
            "top": 0.925,
            "wspace": 0.62,
            "hspace": 0.68,
        },
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h, ax_i = axes.ravel()
    # The schematic has no tick or axis-label rows below it, so stretch its
    # axes down to the flanking heatmaps' full visual extent (axes + labels);
    # otherwise the top row ends with a dead whitespace band under panel A.
    box_a = ax_a.get_position()
    ax_a.set_position(
        [box_a.x0, box_a.y0 - 0.045, box_a.width, box_a.height + 0.045]
    )
    operator_schematic(ax_a)

    # ── B: ancestry-over-random spectral-capture advantage (signed) ───────
    wide = spectral.pivot_table(
        index=["alignment", "budget_k"], columns="method",
        values="mean_spectral_capture",
    )
    advantage = (wide.ancestry - wide.random_rank).unstack("budget_k")
    budgets = list(advantage.columns)
    alignments = list(advantage.index)
    advantage_scaled = advantage.to_numpy() * 100.0
    advantage_limit = float(np.max(np.abs(advantage_scaled)))
    heatmap_with_colorbar(
        fig, ax_b, advantage_scaled,
        [str(value) for value in budgets],
        [f"{value:.2f}" for value in alignments],
        cmap=DIV_CMAP,
        norm=TwoSlopeNorm(vmin=-advantage_limit, vcenter=0.0,
                          vmax=advantage_limit),
        fmt="{:.0f}", half_away=True,
        label=r"capture advantage ($\times 10^{-2}$)",
    )
    ax_b.set_xlabel("route budget $K$")
    ax_b.set_ylabel(r"alignment $\rho$")
    panel_title(ax_b, "B", "Spectral alignment")
    style_axis(ax_b)

    # ── C: final loss across task depth x model depth (magnitude) ─────────
    aligned = depth[depth.method.eq("aligned_tree")].pivot(
        index="task_depth", columns="model_depth", values="mean_final_population_loss"
    )
    heatmap_with_colorbar(
        fig, ax_c, aligned.to_numpy(),
        [str(v) for v in aligned.columns], [str(v) for v in aligned.index],
        cmap=SEQ_CMAP, fmt="{:.2f}", label="final loss",
    )
    for i in range(min(aligned.shape)):  # outline the depth-matched D == H cells
        ax_c.plot(
            [i - 0.5, i + 0.5, i + 0.5, i - 0.5, i - 0.5],
            [i - 0.5, i - 0.5, i + 0.5, i + 0.5, i - 0.5],
            color=COLORS["ink"], lw=LW_EDGE, zorder=3,
            solid_joinstyle="miter",
        )
    ax_c.set_xlabel(r"routed depth $D_{\mathrm{r}}$")
    ax_c.set_ylabel("task depth $H$")
    panel_title(ax_c, "C", "Depth matching")
    style_axis(ax_c)
    # Key for the outlined diagonal, kept with the panel so it stands alone.
    ax_c.text(0.5, -0.285, r"boxes: $D_{\mathrm{r}}=H$",
              transform=ax_c.transAxes, ha="center", va="top",
              fontsize=PT_SMALL, style="italic", color=COLORS["mute"])

    # ── D: training loss versus model depth, one series per task depth ────
    depth_series = [
        (COLORS["shunting"], MARKERS[0]),
        (COLORS["additive"], MARKERS[1]),
        (COLORS["local"], MARKERS[2]),
        (COLORS["oracle"], MARKERS[3]),
    ]
    # At D = 1 the H=3 and H=4 means nearly coincide, and at D = 2 the H=1,
    # H=3 and H=4 means do; fan the H=3 / H=4 markers horizontally there
    # (panel-G convention, values exact) instead of silently occluding them.
    depth_marker_dx = {3: -0.14, 4: 0.14}
    for task_depth, (color, marker) in zip(sorted(aligned.index), depth_series):
        part = depth[
            depth.method.eq("aligned_tree") & depth.task_depth.eq(task_depth)
        ].sort_values("model_depth")
        xs = part.model_depth.to_numpy(dtype=float)
        ys = part.mean_final_population_loss.to_numpy(dtype=float)
        fanned = (
            np.isin(xs, (1.0, 2.0))
            if task_depth in depth_marker_dx
            else np.zeros(len(xs), dtype=bool)
        )
        ax_d.plot(xs, ys, color=color, marker=marker, ms=MARKER_MS, lw=LW_DATA,
                  markevery=list(np.flatnonzero(~fanned)), mec="white", mew=0.4,
                  label=f"$H={task_depth}$")
        if fanned.any():
            ax_d.scatter(xs[fanned] + depth_marker_dx[task_depth], ys[fanned],
                         color=color, marker=marker, s=MARKER_MS ** 2, zorder=3,
                         edgecolors="white", linewidths=0.4)
        ax_d.fill_between(part.model_depth, part.ci95_low_final_population_loss,
                          part.ci95_high_final_population_loss, color=color,
                          alpha=0.10, linewidth=0)
    ax_d.set_yscale("log")
    ax_d.set_xticks([1, 2, 3, 4])
    ax_d.set_xlabel(r"routed depth $D_{\mathrm{r}}$")
    ax_d.set_ylabel("final loss")
    panel_title(ax_d, "D", "Depth crossover")
    style_axis(ax_d)
    clean_legend(ax_d, loc="upper left", fontsize=PT_LEGEND, ncol=2)
    ax_d.text(0.98, 0.03, "coincident markers\noffset",
              transform=ax_d.transAxes, ha="right", va="bottom",
              fontsize=PT_SMALL, style="italic", color=COLORS["mute"],
              linespacing=1.25)

    # ── E: signed loss change of the common projection (diverging) ────────
    projection_wide = projection.pivot_table(
        index=["signal_retention", "noise_retention"], columns="method",
        values="mean_expected_population_loss",
    )
    delta = (
        projection_wide.bp_plus_route_projection
        - projection_wide.full_stochastic_bp
    ).unstack("signal_retention")
    # Rows are noise retention and columns are signal retention.  Negative
    # entries mean that the common projection improves stochastic BP.
    scaled = delta.to_numpy() * 100.0
    limit = float(np.max(np.abs(scaled)))
    # Five 2-decimal x tick labels need breathing room: widen the panel a
    # touch into the wide inter-column gutter before the colorbar carve-out.
    box_e = ax_e.get_position()
    ax_e.set_position(
        [box_e.x0 - 0.012, box_e.y0, box_e.width + 0.014, box_e.height]
    )
    heatmap_with_colorbar(
        fig, ax_e, scaled,
        [f"{v:.2f}" for v in delta.columns], [f"{v:.2f}" for v in delta.index],
        cmap=DIV_CMAP, norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        fmt="{:.0f}", half_away=True,
        label=r"$\Delta$ loss ($\times 10^{-2}$)",
    )
    # Zero boundary between cells where the projection helps and where it
    # hurts, drawn along cell edges so it never crosses a value label.
    ax_e.plot(
        [-0.5, 0.5, 0.5, 1.5, 1.5, 3.5, 3.5],
        [1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 4.5],
        ls="--", color=COLORS["mute"], lw=LW_REF, zorder=3,
    )
    ax_e.set_xlabel("retained signal fraction $r$")
    ax_e.set_ylabel("noise retention $n$")
    panel_title(ax_e, "E", "Projection boundary")
    style_axis(ax_e)
    # One step down the type scale keeps the dense 2-decimal ticks separated.
    ax_e.tick_params(axis="both", labelsize=PT_SMALL)

    # ── F: one-step reliability gains for the four gain policies ──────────
    reliability_styles = [
        ("reliability_aligned", "SNR-aligned", COLORS["shunting"], MARKERS[0]),
        ("best_global_gain", "best global", COLORS["per_soma"], MARKERS[1]),
        ("shuffled_shunting", "shuffled", COLORS["point_mlp"], MARKERS[2]),
        ("anti_aligned_shunting", "anti-aligned", COLORS["additive"], MARKERS[3]),
    ]
    # All four policies share one exact value at zero heterogeneity; fan the
    # x = 0 markers horizontally (panel-D convention, values exact) instead
    # of silently stacking them, and widen the x margin so the fan clears
    # the left spine.
    reliability_marker_dx = {
        "reliability_aligned": -0.165,
        "best_global_gain": -0.055,
        "shuffled_shunting": 0.055,
        "anti_aligned_shunting": 0.165,
    }
    for method, label, color, marker in reliability_styles:
        part = reliability[reliability.method.eq(method)].sort_values(
            "reliability_heterogeneity"
        )
        xs = part.reliability_heterogeneity.to_numpy(dtype=float)
        ys = part.mean_population_loss_decrease.to_numpy(dtype=float)
        fanned = xs == 0.0
        ax_f.plot(xs, ys, color=color, marker=marker,
                  ms=MARKER_MS, lw=LW_DATA, label=label,
                  markevery=list(np.flatnonzero(~fanned)),
                  mec="white", mew=0.4)
        if fanned.any():
            ax_f.scatter(xs[fanned] + reliability_marker_dx[method], ys[fanned],
                         color=color, marker=marker, s=MARKER_MS ** 2, zorder=3,
                         edgecolors="white", linewidths=0.4)
        ax_f.fill_between(part.reliability_heterogeneity,
                          part.ci95_low_population_loss_decrease,
                          part.ci95_high_population_loss_decrease,
                          color=color, alpha=0.08, linewidth=0)
    ax_f.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_f.margins(x=0.115)
    ax_f.set_ylim(-4.0, 3.9)
    ax_f.set_xlabel("branch-SNR heterogeneity")
    ax_f.set_ylabel("one-step loss decrease")
    panel_title(ax_f, "F", "Reliability gains")
    style_axis(ax_f)
    clean_legend(ax_f, fontsize=PT_LEGEND, loc="lower left")
    # Under-axis key note (panel-C convention): inside the panel the note's
    # baseline sat level with the legend's last entry and read as one line.
    ax_f.text(0.5, -0.285,
              "point gate ≡ SNR-aligned\ncoincident markers at 0 offset",
              transform=ax_f.transAxes, ha="center", va="top",
              fontsize=PT_SMALL, style="italic", color=COLORS["mute"],
              linespacing=1.25)

    # ── G: spectral capture is identical for the three bases ──────────────
    span_styles = [
        ("tree_haar", "tree Haar", COLORS["oracle"], MARKERS[0]),
        ("raw_nested_indicators", "raw nested", COLORS["additive"], MARKERS[1]),
        ("static_gain_scaled_nested", "gain-scaled", COLORS["shunting"], MARKERS[2]),
    ]
    common = span[span.basis.eq("tree_haar")].sort_values("depth")
    ax_g.plot(common.depth, common.spectral_capture, color=COLORS["mute"],
              lw=LW_DATA, zorder=1)
    for offset, (basis, label, color, marker) in zip([-0.16, 0.0, 0.16], span_styles):
        part = span[span.basis.eq(basis)].sort_values("depth")
        ax_g.scatter(part.depth + offset, part.spectral_capture, color=color,
                     marker=marker, s=MARKER_MS ** 2, label=label, zorder=3,
                     edgecolors="white", linewidths=0.4)
    ax_g.set_xticks([1, 2, 3, 4])
    ax_g.set_xlabel("address depth")
    ax_g.set_ylabel("spectral capture")
    panel_title(ax_g, "G", "Span invariance")
    style_axis(ax_g)
    clean_legend(ax_g, fontsize=PT_LEGEND, loc="upper left")
    ax_g.text(0.97, 0.05, "curves coincide\n(markers offset)",
              transform=ax_g.transAxes, ha="right", va="bottom",
              fontsize=PT_SMALL, style="italic", color=COLORS["mute"],
              linespacing=1.25)

    # ── H: Gram conditioning separates the same-span bases ────────────────
    for basis, label, color, marker in span_styles:
        part = span[span.basis.eq(basis)].sort_values("depth")
        ax_h.plot(part.depth, part.condition_number, color=color, marker=marker,
                  ms=MARKER_MS, lw=LW_DATA, label=label)
    ax_h.set_yscale("log")
    ax_h.set_xticks([1, 2, 3, 4])
    ax_h.set_xlabel("address depth")
    ax_h.set_ylabel("Gram condition number")
    panel_title(ax_h, "H", "Route conditioning")
    style_axis(ax_h)
    # Direct per-series labels in place of a legend: no clear legend corner
    # exists on this log panel, and colored labels keep it standalone.
    ax_h.text(2.17, 700.0, "gain-scaled", color=COLORS["shunting"],
              ha="left", va="center", fontsize=PT_LEGEND)
    ax_h.text(3.4, 5.6, "raw nested", color=COLORS["additive"],
              ha="center", va="bottom", fontsize=PT_LEGEND)
    ax_h.text(2.5, 1.45, "tree Haar", color=COLORS["oracle"],
              ha="center", va="bottom", fontsize=PT_LEGEND)

    # ── I: phase utility predicts observed one-step progress ──────────────
    merged = operator.merge(
        outcomes,
        on=["seed", "condition_id", "architecture", "feedback_family", "budget_k"],
        validate="one_to_one",
    )
    merged = merged[merged.architecture.eq("dendritic_tree")]
    ax_i.scatter(merged.maximum_guaranteed_decrease,
                 merged.norm_matched_one_step_progress,
                 s=7, alpha=0.30, edgecolors="none", color=COLORS["additive"])
    # Extra headroom so the stats block clears the dense y = 1.0 data ceiling.
    ax_i.set_ylim(-1.42, 2.08)
    ax_i.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ci_low, ci_high = audit["seed_block_ci95_utility_vs_one_step_progress"]
    ax_i.text(0.02, 0.97,
              rf"Spearman $\rho={audit['spearman_utility_vs_one_step_progress']:.2f}$"
              + "\n"
              + rf"seed-block 95% CI {ci_low:.2f}–{ci_high:.2f}",
              transform=ax_i.transAxes, ha="left", va="top",
              fontsize=PT_ANNOT, color=COLORS["mute"], linespacing=1.3)
    ax_i.annotate(
        "misrouted controls",
        xy=(0.03, -0.96), xytext=(0.14, -0.96), textcoords="data",
        ha="left", va="center", fontsize=PT_SMALL, style="italic",
        color=COLORS["mute"],
        arrowprops=dict(arrowstyle="-", color=COLORS["mute"], lw=LW_HAIR,
                        shrinkA=2, shrinkB=0),
    )
    ax_i.set_xlabel("phase utility")
    ax_i.set_ylabel("observed progress")
    panel_title(ax_i, "I", "Predictive utility")
    style_axis(ax_i)

    save(fig, "fig_credit_phase_theory")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build publication-facing replacements for the remaining weak main panels.

The underlying analyses and frozen source tables are unchanged.  This script
only improves the visual encoding used in the compact main figures:

* Figure 4G: a quieter alignment-by-bandwidth boundary map;
* Figure 5B: an explicit ordered divisive-task schematic;
* Figure 6: one coherent matrix/effect-size figure instead of seven small
  diagnostic plots;
* Figure 7A/B/E/G: clean measured-arbor schematics, a Pareto view of wiring,
  and a high-contrast cross-animal comparison;
* Figure 8A: a causal matched-current versus shunt schematic; and
* Figure 9A: a two-estimand target-level summary rather than an unlabeled
  strip of points.

All outputs are vector PDFs written to ``figures/generated`` and published by
``sync_canonical_figures.py`` under stable component names.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

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
    MARKER_MS,
    PT_ANNOT,
    PT_LABEL,
    PT_LEGEND,
    PT_SMALL,
    PT_TICK,
    SEQ_CMAP,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data"
OUT = ROOT / "figures" / "generated"


def save(fig: plt.Figure, stem: str, *, audit_overlap: bool = True) -> None:
    """Save a deterministic vector panel after the common visual audits."""

    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    layout = audit_layout(fig, stem)
    overlap = audit_text_over_data(fig, stem) if audit_overlap else []
    if layout or overlap:
        print(f"  review {stem}: {len(layout)} layout, {len(overlap)} text/data warnings")
    fig.savefig(
        OUT / f"{stem}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def mean_ci(values: np.ndarray, seed: int, n_boot: int = 20_000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 1:
        return float(values[0]), float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    return to_rgba(color, alpha)


# ---------------------------------------------------------------------------
# Figure 4G: alignment x bandwidth
# ---------------------------------------------------------------------------


def build_phase_plane() -> None:
    points = pd.read_csv(
        SOURCE / "credit_phase_plane" / "points.csv", keep_default_na=False
    )
    colors = {
        "factorial": COLORS["shunting"],
        "sweep": COLORS["additive"],
        "microns": COLORS["oracle"],
        "measured": COLORS["point_mlp"],
        "reversal": COLORS["bp"],
    }
    markers = {
        "factorial": "o",
        "sweep": "D",
        "microns": "s",
        "measured": "X",
        "reversal": "^",
    }
    labels = {
        "factorial": "trained bandwidth",
        "sweep": "spectral theory",
        "microns": "imposed alignment",
        "measured": "measured responses",
        "reversal": "credit reversal",
    }

    fig = plt.figure(figsize=(FIG_W, 2.42))
    ax = fig.add_axes([0.140, 0.29, 0.835, 0.66])
    ax.set_yscale("log")
    ax.set_xlim(-0.04, 1.10)
    ax.set_ylim(0.09, 5.2)

    # Three quiet, theory-derived regions.  They communicate the phase logic
    # without the previous panel's dense field of overlapping prose.
    ax.axhspan(0.09, 0.24, color=_rgba(COLORS["local"], 0.10), zorder=0)
    ax.axhspan(1.2, 5.2, color=_rgba(COLORS["point_mlp"], 0.07), zorder=0)
    ax.fill_between(
        [0.0, 0.42, 0.42, 1.10],
        [0.24, 0.24, 0.24, 0.24],
        [1.2, 1.2, 1.2, 1.2],
        color=_rgba(COLORS["shunting"], 0.07),
        zorder=0,
    )
    ax.axhline(1.0, color=COLORS["mute"], lw=LW_HAIR, ls=(0, (2, 2)), zorder=1)

    ax.text(
        0.02, 0.13, "bandwidth bottleneck", color=COLORS["mute"],
        fontsize=PT_ANNOT, style="italic", va="center",
    )
    ax.text(
        0.18, 0.48, "misaligned routes", color=COLORS["mute"],
        fontsize=PT_ANNOT, style="italic", ha="center", va="center",
    )
    ax.text(
        0.77, 0.34, "matched operating region", color=COLORS["shunting"],
        fontsize=PT_ANNOT, style="italic", ha="center", va="center",
    )
    ax.text(
        0.03, 2.45, "route span already saturated", color=COLORS["mute"],
        fontsize=PT_ANNOT, style="italic", va="center",
    )
    ax.text(
        1.085, 4.55, "filled: benefit or tie\nopen: no benefit",
        ha="right", va="top", fontsize=PT_SMALL, color=COLORS["mute"],
    )

    # Small fixed dodges separate the three full-alignment observations.
    dodge = {
        "factorial K=8": -0.035,
        "MICrONS controlled a=1": 0.0,
        "credit reversal K=2": 0.035,
    }
    points["xp"] = points.x + points.label.map(lambda x: dodge.get(x, 0.0))
    for family in ("factorial", "sweep", "microns"):
        part = points[points.family.eq(family)].sort_values("x")
        ax.plot(
            part.xp, part.y, color=colors[family], lw=LW_DATA,
            alpha=0.78, zorder=2,
        )

    handles: list[Line2D] = []
    for family in ("factorial", "sweep", "microns", "measured", "reversal"):
        part = points[points.family.eq(family)]
        for row in part.itertuples(index=False):
            filled = str(row.outcome) not in {"loss", "null"}
            ax.plot(
                row.xp,
                row.y,
                marker=markers[family],
                ms=5.8,
                ls="none",
                mfc=colors[family] if filled else "white",
                mec=colors[family],
                mew=LW_ERR,
                zorder=4,
            )
        handles.append(
            Line2D(
                [], [], color=colors[family], marker=markers[family],
                lw=LW_DATA if family in {"factorial", "sweep", "microns"} else 0,
                markersize=5.2, markeredgewidth=LW_ERR, label=labels[family],
            )
        )

    ax.annotate(
        "K=4\n+1.3 pp",
        xy=(0.522873, 0.500452), xytext=(0, 12), textcoords="offset points",
        ha="center", va="bottom", fontsize=PT_SMALL, color=COLORS["shunting"],
    )
    ax.annotate(
        "rank-saturated null",
        xy=(0.474526, 3.73089), xytext=(10, -2), textcoords="offset points",
        ha="left", va="center", fontsize=PT_SMALL, color=COLORS["point_mlp"],
    )
    ax.annotate(
        "routing required",
        xy=(1.035, 1.00036), xytext=(-4, 16), textcoords="offset points",
        ha="right", va="bottom", fontsize=PT_SMALL, color=COLORS["bp"],
        arrowprops={"arrowstyle": "-", "color": COLORS["bp"], "lw": LW_HAIR},
    )

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.125, 0.25, 0.5, 1, 2, 4])
    ax.set_yticklabels(["1/8", "1/4", "1/2", "1", "2", "4"])
    ax.minorticks_off()
    ax.set_xlabel("task–anatomy alignment")
    ax.set_ylabel(r"bandwidth / task rank  ($K/r_{\mathrm{eff}}$)")
    style_axis(ax, grid="none")
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.015),
        ncol=5, frameon=False, fontsize=PT_LEGEND, handlelength=1.25,
        handletextpad=0.4, columnspacing=1.05,
    )
    save(fig, "fig_main_phase_plane_clean")


# ---------------------------------------------------------------------------
# Figure 5B: nested divisive task
# ---------------------------------------------------------------------------


def _rounded_stage(ax, center, width, height, color, label, gain) -> None:
    x, y = center
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - height / 2), width, height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor=_rgba(color, 0.10), edgecolor=color, lw=LW_EDGE,
        )
    )
    ax.text(x, y + 0.015, label, ha="center", va="center",
            fontsize=PT_SMALL, color=color)
    ax.text(x, y - 0.055, gain, ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["ink"])


def build_physical_task_schematic() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W / 2, 1.40))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.03, top=0.98)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stage_colors = (COLORS["oracle"], COLORS["additive"], COLORS["shunting"])
    centers = [(0.36, 0.73), (0.55, 0.53), (0.72, 0.33)]
    labels = (("fine", r"$G_f$"), ("coarse", r"$G_c$"), ("global", r"$G_g$"))

    # A single distal signal passes through three ordered divisive stages.
    ax.annotate(
        "", xy=(centers[0][0] - 0.105, centers[0][1] + 0.055),
        xytext=(0.10, 0.88),
        arrowprops={"arrowstyle": "-|>", "color": COLORS["bp"], "lw": LW_DATA},
    )
    ax.text(0.085, 0.91, "class signal", ha="left", va="center",
            fontsize=PT_SMALL, color=COLORS["bp"])
    ax.text(0.16, 0.78, r"$s_y$", ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["bp"])

    for first, second in zip(centers[:-1], centers[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (first[0] + 0.07, first[1] - 0.07),
                (second[0] - 0.07, second[1] + 0.07),
                arrowstyle="-|>", mutation_scale=8, color=COLORS["dend"],
                lw=LW_DATA,
            )
        )
    for center, color, (label, gain) in zip(centers, stage_colors, labels, strict=True):
        _rounded_stage(ax, center, 0.22, 0.19, color, label, gain)

    # Gain-sensor streams enter the matching stage rather than the soma.
    for index, (center, color) in enumerate(zip(centers, stage_colors, strict=True)):
        source = (center[0] + 0.18, center[1] + 0.14)
        ax.plot(source[0], source[1], marker="o", ms=4.0, mfc="white",
                mec=color, mew=LW_EDGE)
        ax.annotate(
            "", xy=(center[0] + 0.09, center[1] + 0.06), xytext=source,
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": LW_EDGE},
        )
        if index == 2:
            ax.text(source[0] - 0.025, source[1], f"sensor {index + 1}",
                    ha="right", va="center", fontsize=PT_SMALL, color=color)
        else:
            ax.text(source[0] + 0.025, source[1], f"sensor {index + 1}",
                    ha="left", va="center", fontsize=PT_SMALL, color=color)

    soma = (0.84, 0.18)
    ax.plot([centers[-1][0] + 0.07, soma[0] - 0.04],
            [centers[-1][1] - 0.07, soma[1] + 0.025],
            color=COLORS["dend"], lw=LW_DATA, solid_capstyle="round")
    ax.add_patch(Circle(soma, 0.035, facecolor=COLORS["soma"],
                        edgecolor=COLORS["edge"], lw=LW_EDGE))
    ax.text(
        0.49, 0.07, r"$E_{\mathrm{distal}}=s_y\,G_f\,G_c\,G_g$",
        ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["ink"],
    )
    save(fig, "fig_main_physical_task_schematic", audit_overlap=False)


# ---------------------------------------------------------------------------
# Figure 6: coherent phase boundary and generalization figure
# ---------------------------------------------------------------------------


H4_ROWS = [
    ("serial BP", "serial_tree", "shunting", "full_bp"),
    ("grouped point", "grouped_point", "shunting", "full_bp"),
    ("shared LocalCA", "serial_tree", "shunting", "local_shared"),
    ("path LocalCA", "serial_tree", "shunting", "local_path"),
    ("raw additive", "serial_tree", "raw_additive", "full_bp"),
]


def _annotated_matrix(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    cmap,
    norm,
    fmt: str,
    best_by_row: bool = False,
) -> None:
    masked = np.ma.masked_invalid(matrix)
    cmap_local = cmap.copy()
    cmap_local.set_bad("#F2F3F5")
    ax.imshow(masked, cmap=cmap_local, norm=norm, aspect="auto", interpolation="nearest")
    for row in range(matrix.shape[0]):
        finite = np.flatnonzero(np.isfinite(matrix[row]))
        best = finite[np.argmax(matrix[row, finite])] if len(finite) else None
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            if not np.isfinite(value):
                ax.text(col, row, "—", ha="center", va="center",
                        fontsize=PT_SMALL, color=COLORS["mute"])
                continue
            rgba = cmap_local(norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(col, row, fmt.format(value), ha="center", va="center",
                    fontsize=PT_SMALL, color="white" if luminance < 0.48 else COLORS["ink"])
            if best_by_row and col == best:
                ax.add_patch(
                    FancyBboxPatch(
                        (col - 0.46, row - 0.43), 0.92, 0.86,
                        boxstyle="round,pad=0.0,rounding_size=0.04",
                        facecolor="none", edgecolor=COLORS["ink"], lw=LW_ERR,
                    )
                )
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.tick_params(axis="both", length=0, labelsize=PT_SMALL)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _h4_matrix(summary: pd.DataFrame, regime: str) -> np.ndarray:
    matrix = np.full((len(H4_ROWS), 4), np.nan)
    for row_index, (_, architecture, mechanism, credit) in enumerate(H4_ROWS):
        part = summary[
            summary.regime.eq(regime)
            & summary.architecture.eq(architecture)
            & summary.mechanism.eq(mechanism)
            & summary.credit.eq(credit)
        ]
        for row in part.itertuples(index=False):
            matrix[row_index, int(row.depth) - 1] = float(row.mean_test_accuracy)
    return matrix


def _hierarchy_depth_matrix(h4_seed: pd.DataFrame) -> np.ndarray:
    h23 = pd.read_csv(
        SOURCE / "physical_depth_clean_source_replication" / "seed_outcomes.csv"
    )
    common = pd.concat([h23, h4_seed], ignore_index=True, sort=False)
    common = common[
        common.hierarchy.isin([2, 3, 4])
        & common.regime.eq("aligned")
        & common.architecture.eq("serial_tree")
        & common.mechanism.eq("shunting")
        & common.credit.eq("full_bp")
    ]
    matrix = np.full((3, 4), np.nan)
    for (hierarchy, depth), part in common.groupby(["hierarchy", "depth"]):
        matrix[int(hierarchy) - 2, int(depth) - 1] = part.test_accuracy.mean()
    return matrix


def _task_matrix(effects: pd.DataFrame, credit: str) -> np.ndarray:
    families = ["nested_factor", "flat_factor", "local_ratio"]
    alphas = [0.0, 0.5, 1.0]
    matrix = np.full((3, 3), np.nan)
    for r, family in enumerate(families):
        for c, alpha in enumerate(alphas):
            row = effects[
                effects.family.eq(family)
                & effects.credit.eq(credit)
                & np.isclose(effects.alignment_alpha, alpha)
            ]
            if len(row):
                matrix[r, c] = 100 * float(row.iloc[0].mean_difference)
    return matrix


def build_figure6() -> None:
    summary = pd.read_csv(
        SOURCE / "physical_depth_h4_factorial" / "condition_summary.csv"
    )
    contrasts = pd.read_csv(
        SOURCE / "physical_depth_h4_factorial" / "paired_contrasts.csv"
    ).set_index("contrast")
    h4_seed = pd.read_csv(
        SOURCE / "physical_depth_h4_factorial" / "seed_outcomes.csv"
    )
    task_effects = pd.read_csv(
        SOURCE / "task_family_alignment" / "architecture_effects.csv"
    )
    task_contrasts = pd.read_csv(
        SOURCE / "task_family_alignment" / "paired_contrasts.csv"
    )

    fig = plt.figure(figsize=(FIG_W, 5.52))
    grid = fig.add_gridspec(
        3, 6, left=0.150, right=0.985, bottom=0.075, top=0.945,
        wspace=1.14, hspace=0.93, height_ratios=[1.0, 1.0, 0.94],
    )
    ax_a = fig.add_subplot(grid[0, 0:3])
    ax_b = fig.add_subplot(grid[0, 3:6])
    ax_c = fig.add_subplot(grid[1, 0:3])
    ax_d = fig.add_subplot(grid[1, 3:6])
    ax_e = fig.add_subplot(grid[2, 0:2])
    ax_f = fig.add_subplot(grid[2, 2:4])
    ax_g = fig.add_subplot(grid[2, 4:6])

    accuracy_norm = Normalize(vmin=0.52, vmax=0.88)
    row_labels = [row[0] for row in H4_ROWS]
    for ax, regime, letter, title in (
        (ax_a, "aligned", "A", "H4: aligned hierarchy"),
        (ax_b, "rewired_tree", "B", "H4: reversed placement"),
    ):
        _annotated_matrix(
            ax, _h4_matrix(summary, regime), row_labels,
            ["D1", "D2", "D3", "D4"], cmap=SEQ_CMAP,
            norm=accuracy_norm, fmt="{:.2f}", best_by_row=True,
        )
        panel_title(ax, letter, title)
    ax_b.set_yticklabels([])
    ax_b.tick_params(axis="y", length=0)
    ax_a.text(
        0.99, 1.035, "outlined = best depth", ha="right", va="bottom",
        fontsize=PT_SMALL, color=COLORS["mute"], transform=ax_a.transAxes,
    )

    # C: retain only the contrasts needed to interpret saturation, matched
    # architecture, and local-credit transport.
    contrast_specs = [
        ("depth__serial_bp__aligned__d4_d3", "serial BP, D4 − D3", COLORS["shunting"]),
        ("depth__serial_bp__aligned__d4_d1", "serial BP, D4 − D1", COLORS["shunting"]),
        ("serial_minus_grouped__aligned__d4", "serial − point, D4", COLORS["bp"]),
        ("depth__shared_local__aligned__d4_d3", "shared local, D4 − D3", COLORS["local"]),
        ("depth__path_local__aligned__d4_d3", "path local, D4 − D3", COLORS["oracle"]),
        ("shunting_additive_depth_interaction__aligned__d4_d3", "shunt × depth", COLORS["point_mlp"]),
    ]
    y = np.arange(len(contrast_specs))[::-1]
    ax_c.axvline(0, color=COLORS["mute"], lw=LW_REF, ls="--", zorder=0)
    for yi, (name, _, color) in zip(y, contrast_specs, strict=True):
        row = contrasts.loc[name]
        mean = float(row.mean_pp)
        low = float(row.ci_low_pp)
        high = float(row.ci_high_pp)
        ax_c.barh(yi, mean, height=0.42, color=_rgba(color, 0.20),
                  edgecolor=color, lw=LW_EDGE, zorder=1)
        ax_c.errorbar(
            mean, yi, xerr=[[mean - low], [high - mean]], fmt="D",
            ms=4.1, mfc="white", mec=color, mew=LW_ERR, color=color,
            lw=LW_ERR, capsize=ERR_CAPSIZE, zorder=3,
        )
        ax_c.text(
            mean + (0.8 if mean >= 0 else -0.8), yi, f"{mean:+.1f}",
            ha="left" if mean >= 0 else "right", va="center",
            fontsize=PT_SMALL, color=color,
        )
    ax_c.set_yticks(y, [spec[1] for spec in contrast_specs])
    ax_c.tick_params(axis="y", labelsize=PT_SMALL)
    ax_c.set_xlabel("paired accuracy difference (pp)")
    ax_c.set_xlim(-5.0, 30.5)
    panel_title(ax_c, "C", "Seed-paired H4 contrasts")
    style_axis(ax_c, grid="x")

    depth_matrix = _hierarchy_depth_matrix(h4_seed)
    _annotated_matrix(
        ax_d, depth_matrix, ["H2", "H3", "H4"],
        ["D1", "D2", "D3", "D4"], cmap=SEQ_CMAP,
        norm=accuracy_norm, fmt="{:.2f}", best_by_row=True,
    )
    ax_d.set_xlabel("physical stage count")
    ax_d.set_ylabel("task hierarchy")
    panel_title(ax_d, "D", "Depth tracks hierarchy, then saturates")

    effect_norm = mpl.colors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=32)
    family_labels = ["nested\nfactors", "flat\nfactors", "local\nratios"]
    for ax, credit, letter, title in (
        (ax_e, "bp", "E", "Serial benefit under BP"),
        (ax_f, "local3f", "F", "Serial benefit under LocalCA"),
    ):
        _annotated_matrix(
            ax, _task_matrix(task_effects, credit), family_labels,
            ["0", ".5", "1"], cmap=DIV_CMAP,
            norm=effect_norm, fmt="{:+.1f}", best_by_row=False,
        )
        ax.set_xlabel(r"sensor alignment $\alpha$")
        ax.set_ylabel("task family")
        panel_title(ax, letter, title)

    # G: the change from alpha=0 to alpha=1 is the direct interaction.  Bars
    # and interval whiskers read faster than the previous six-point forest.
    interaction = task_contrasts[
        task_contrasts.estimand.eq("alignment_interaction")
    ].copy()
    families = ["nested_factor", "flat_factor", "local_ratio"]
    family_labels_short = ["nested", "flat", "local ratio"]
    family_colors = [COLORS["shunting"], COLORS["point_mlp"], COLORS["local"]]
    positions = np.arange(3)
    width = 0.34
    for credit, label, offset, hatch in (
        ("bp", "BP", -width / 2, ""),
        ("local3f", "LocalCA", width / 2, "///"),
    ):
        means, lows, highs = [], [], []
        for family in families:
            row = interaction[
                interaction.family.eq(family) & interaction.credit.eq(credit)
            ].iloc[0]
            means.append(100 * float(row.mean_difference))
            lows.append(100 * float(row.ci95_low))
            highs.append(100 * float(row.ci95_high))
        for index, (mean, low, high, color) in enumerate(
            zip(means, lows, highs, family_colors, strict=True)
        ):
            xpos = positions[index] + offset
            ax_g.bar(
                xpos, mean, width=width * 0.88, color=_rgba(color, 0.24),
                edgecolor=color, lw=LW_EDGE, hatch=hatch, label=label if index == 0 else None,
            )
            ax_g.errorbar(
                xpos, mean, yerr=[[mean - low], [high - mean]], fmt="none",
                ecolor=color, elinewidth=LW_ERR, capsize=ERR_CAPSIZE,
            )
    ax_g.axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax_g.set_xticks(positions, family_labels_short)
    ax_g.tick_params(axis="x", labelsize=PT_SMALL)
    ax_g.set_ylabel(r"alignment interaction (pp)")
    panel_title(ax_g, "G", "Alignment interaction")
    style_axis(ax_g, grid="y")
    clean_legend(ax_g, fontsize=PT_SMALL, loc="lower left", ncol=1)

    save(fig, "fig_main_figure6_redesigned", audit_overlap=False)


# ---------------------------------------------------------------------------
# Figure 7: anatomy panels and biological replication
# ---------------------------------------------------------------------------


def _morphology_geometry():
    segments = pd.read_csv(SOURCE / "figure3" / "segment_metrics.csv")
    root = int(segments.groupby("root_id").size().idxmax())
    cell = segments[segments.root_id.eq(root)].copy()
    xyz = cell[["x_um", "y_um", "z_um"]].to_numpy(float)
    centered = xyz - xyz.mean(axis=0, keepdims=True)
    _, _, basis = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ basis[:2].T
    projected /= max(np.ptp(projected[:, 0]), np.ptp(projected[:, 1]))
    positions = {
        int(segment): point
        for segment, point in zip(cell.segment_id.to_numpy(int), projected, strict=True)
    }
    rows = {int(row.segment_id): row for row in cell.itertuples(index=False)}
    parent = {int(row.segment_id): int(row.parent_segment_id) for row in cell.itertuples(index=False)}
    return cell, positions, rows, parent


def _descendants(parent: dict[int, int], node: int) -> set[int]:
    result: set[int] = set()
    for candidate in parent:
        current = candidate
        seen: set[int] = set()
        while current != -1 and current in parent and current not in seen:
            if current == node:
                result.add(candidate)
                break
            seen.add(current)
            current = parent[current]
    return result


def _draw_arbor(ax, positions, rows, parent, *, base_color="#C5CBD2", lw=0.55):
    for segment, row in rows.items():
        p = parent[segment]
        if p not in rows:
            continue
        start, end = positions[segment], positions[p]
        ax.plot([start[0], end[0]], [start[1], end[1]], color=base_color,
                lw=lw, solid_capstyle="round", zorder=1)
    soma_id = min(rows, key=lambda key: rows[key].topological_depth)
    soma = positions[soma_id]
    ax.scatter(soma[0], soma[1], s=45, color=COLORS["soma"],
               edgecolor="white", linewidth=0.6, zorder=7)


def _fit_arbor(ax, positions) -> None:
    xy = np.asarray(list(positions.values()))
    xpad = 0.06 * np.ptp(xy[:, 0])
    ypad = 0.06 * np.ptp(xy[:, 1])
    ax.set_xlim(xy[:, 0].min() - xpad, xy[:, 0].max() + xpad)
    ax.set_ylim(xy[:, 1].min() - ypad, xy[:, 1].max() + ypad)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def build_mapped_reconstruction() -> None:
    cell, positions, rows, parent = _morphology_geometry()
    fig, ax = plt.subplots(figsize=(FIG_W / 3, 1.62))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.98)

    xy = np.asarray(list(positions.values()))
    center = xy.mean(axis=0)
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]))

    def transformed(segment: int, panel_center: float):
        px, py = (positions[segment] - center) / span
        return panel_center + 0.41 * px, 0.54 + 0.72 * py

    for panel_center, count_column, color, label in (
        (0.25, "E_count", COLORS["exc"], "excitatory"),
        (0.75, "I_count", COLORS["inh"], "inhibitory"),
    ):
        counts = cell[count_column].to_numpy(float)
        scale = np.log1p(counts)
        scale /= max(scale.max(), 1e-12)
        count_by_segment = dict(zip(cell.segment_id.to_numpy(int), scale, strict=True))
        for segment, row in rows.items():
            p = parent[segment]
            if p not in rows:
                continue
            start, end = transformed(segment, panel_center), transformed(p, panel_center)
            burden = count_by_segment[segment]
            edge_color = _rgba(color, 0.18 + 0.72 * burden) if burden > 0 else "#D6DADF"
            ax.plot(
                [start[0], end[0]], [start[1], end[1]], color=edge_color,
                lw=0.45 + 1.45 * burden, solid_capstyle="round", zorder=2,
            )
        soma_id = min(rows, key=lambda key: rows[key].topological_depth)
        soma = transformed(soma_id, panel_center)
        ax.scatter(soma[0], soma[1], s=28, color=COLORS["soma"],
                   edgecolor="white", linewidth=0.5, zorder=4)
        ax.text(panel_center, 0.08, label, ha="center", va="center",
                fontsize=PT_ANNOT, color=color)
    ax.plot([0.50, 0.50], [0.12, 0.92], color=COLORS["grid"], lw=LW_HAIR)
    ax.text(0.50, 0.985, "branch color/width ∝ mapped contacts",
            ha="center", va="top", fontsize=PT_SMALL, color=COLORS["mute"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    save(fig, "fig_main_mapped_reconstruction", audit_overlap=False)


def build_ancestry_addresses() -> None:
    cell, positions, rows, parent = _morphology_geometry()
    fig, ax = plt.subplots(figsize=(FIG_W / 3, 1.62))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.03, top=0.98)
    _draw_arbor(ax, positions, rows, parent, base_color="#D0D4D9", lw=0.55)

    candidates = cell[
        (cell.topological_depth >= 2)
        & cell.credit_domain_fraction.between(0.14, 0.32)
    ]
    outer_node = int(
        candidates.loc[(candidates.credit_domain_fraction - 0.22).abs().idxmin(), "segment_id"]
    )
    outer = _descendants(parent, outer_node)
    inner_candidates = cell[
        cell.segment_id.isin(outer)
        & (cell.topological_depth > rows[outer_node].topological_depth)
        & cell.credit_domain_fraction.between(0.025, 0.13)
    ]
    inner_node = int(
        inner_candidates.loc[
            (inner_candidates.credit_domain_fraction - 0.07).abs().idxmin(),
            "segment_id",
        ]
    )
    inner = _descendants(parent, inner_node)

    for selected, color, halo, line in (
        (outer, COLORS["shunting"], 4.4, 1.55),
        (inner, COLORS["oracle"], 3.6, 1.65),
    ):
        for segment in selected:
            p = parent.get(segment, -1)
            if p not in rows or p not in selected:
                continue
            start, end = positions[segment], positions[p]
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color=_rgba(color, 0.14), lw=halo,
                    solid_capstyle="round", zorder=2)
            ax.plot([start[0], end[0]], [start[1], end[1]],
                    color=color, lw=line, solid_capstyle="round", zorder=3)
    for number, node, color in (("1", outer_node, COLORS["shunting"]),
                                ("2", inner_node, COLORS["oracle"])):
        x, y = positions[node]
        ax.scatter(x, y, s=36, facecolor="white", edgecolor=color,
                   linewidth=LW_ERR, zorder=6)
        ax.text(x, y, number, ha="center", va="center", fontsize=PT_SMALL,
                color=color, zorder=7)
    _fit_arbor(ax, positions)
    ax.legend(
        handles=[
            Line2D([], [], color=COLORS["shunting"], lw=2.2, label="route 1"),
            Line2D([], [], color=COLORS["oracle"], lw=2.2, label="route 2 ⊂ route 1"),
        ],
        loc="lower left", fontsize=PT_SMALL, frameon=False,
        handlelength=1.3, handletextpad=0.4, borderaxespad=0.0,
    )
    save(fig, "fig_main_ancestry_addresses", audit_overlap=False)


def build_wire_efficiency() -> None:
    cell = pd.read_csv(SOURCE / "capture_per_wire" / "cell_method_channel.csv")
    eight = cell[cell.channels.eq(8)].copy()
    methods = [
        ("dense PCA oracle", "dense", COLORS["oracle"], "o"),
        ("morphology-aware paths", "ancestry", COLORS["shunting"], "s"),
        ("random paths", "random", COLORS["point_mlp"], "D"),
        ("depth-only bins", "depth", COLORS["additive"], "^"),
        ("shuffled ancestry", "shuffle", COLORS["highlight"], "v"),
    ]
    fig, ax = plt.subplots(figsize=(FIG_W / 2, 1.86))
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.25, top=0.96)
    ax.set_xscale("log")
    for method, label, color, marker in methods:
        part = eight[eight.method.eq(method)]
        ax.scatter(
            part.wiring_density, part.credit_capture, marker=marker, s=11,
            color=color, alpha=0.22, edgecolors="none", zorder=2,
        )
        ax.scatter(
            part.wiring_density.mean(), part.credit_capture.mean(),
            marker=marker, s=42, color=color, edgecolor="white",
            linewidth=0.6, zorder=5, label=label,
        )
    dense = eight[eight.method.eq("dense PCA oracle")]
    ancestry = eight[eight.method.eq("morphology-aware paths")]
    start = (dense.wiring_density.mean(), dense.credit_capture.mean())
    end = (ancestry.wiring_density.mean(), ancestry.credit_capture.mean())
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops={
            "arrowstyle": "-|>", "color": COLORS["shunting"],
            "lw": LW_DATA, "connectionstyle": "arc3,rad=-0.16",
        },
    )
    ax.text(
        0.34, 0.79, "85% of dense capture\nat 6.9% of wiring",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=PT_ANNOT, color=COLORS["shunting"],
        bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "none", "alpha": 0.88},
    )
    ax.text(
        0.03, 0.06, "14.2× capture per wire", transform=ax.transAxes,
        ha="left", va="bottom", fontsize=PT_ANNOT, color=COLORS["shunting"],
    )
    ax.set_xlim(0.008, 1.35)
    ax.set_ylim(0.0, 0.86)
    ax.set_xlabel("wiring density")
    ax.set_ylabel("field capture")
    style_axis(ax, grid="both")
    clean_legend(
        ax, loc="lower right", ncol=2, fontsize=PT_SMALL,
        handlelength=0.9, handletextpad=0.25, columnspacing=0.65,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.88,
    )
    save(fig, "fig_main_wire_efficiency")


def build_cross_animal() -> None:
    contrasts = pd.read_csv(
        SOURCE / "pinky_v185_replication" / "routing" / "k4_cross_animal_contrasts.csv"
    )
    controls = ["random paths", "depth-only bins", "shuffled ancestry"]
    labels = ["random routes", "depth bins", "shuffled ancestry"]
    animals = [
        ("minnie65 v661", "minnie65", COLORS["additive"], "s", -0.12),
        ("Pinky v185", "Pinky v185", COLORS["local"], "o", 0.12),
    ]
    fig, ax = plt.subplots(figsize=(FIG_W / 2, 1.88))
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.23, top=0.96)
    y = np.arange(3)[::-1]
    for animal, label, color, marker, offset in animals:
        subset = contrasts[contrasts.animal.eq(animal)]
        for index, control in enumerate(controls):
            values = subset[
                subset.control.eq(control)
            ].morphology_capture_advantage.to_numpy(float)
            mean, low, high = mean_ci(values, 20260830 + index + (0 if offset < 0 else 20))
            ypos = y[index] + offset
            # Cell values are a quiet rug; the high-contrast mean/interval is
            # the primary token and remains separable in grayscale by shape.
            ax.scatter(values, np.full(values.size, ypos), s=7, color=color,
                       alpha=0.18, edgecolors="none", zorder=1)
            ax.errorbar(
                mean, ypos, xerr=[[mean - low], [high - mean]], fmt=marker,
                ms=5.2, color=color, mfc=color, mec="white", mew=0.5,
                lw=LW_ERR, capsize=ERR_CAPSIZE, zorder=4,
                label=label if index == 0 else None,
            )
    ax.axvline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax.set_yticks(y, labels)
    ax.set_xlabel("ancestry-route capture advantage")
    ax.set_xlim(-0.06, 0.96)
    style_axis(ax, grid="x")
    clean_legend(
        ax, loc="lower right", fontsize=PT_SMALL, ncol=1,
        frameon=True, facecolor="white", edgecolor="none", framealpha=0.88,
    )
    save(fig, "fig_main_cross_animal")


# ---------------------------------------------------------------------------
# Figure 8A: focal perturbation schematic
# ---------------------------------------------------------------------------


TREE_NODES = {
    "s": (0.50, 0.08), "j1": (0.50, 0.28),
    "l": (0.28, 0.46), "r": (0.72, 0.46),
    "ll": (0.14, 0.72), "lr": (0.38, 0.72),
    "rl": (0.62, 0.72), "rr": (0.86, 0.72),
    "t1": (0.06, 0.94), "t2": (0.21, 0.95), "t3": (0.33, 0.96),
    "t4": (0.45, 0.94), "t5": (0.56, 0.94), "t6": (0.68, 0.96),
    "t7": (0.80, 0.94), "t8": (0.94, 0.92),
}
TREE_EDGES = [
    ("s", "j1"), ("j1", "l"), ("j1", "r"),
    ("l", "ll"), ("l", "lr"), ("r", "rl"), ("r", "rr"),
    ("ll", "t1"), ("ll", "t2"), ("lr", "t3"), ("lr", "t4"),
    ("rl", "t5"), ("rl", "t6"), ("rr", "t7"), ("rr", "t8"),
]


def _mini_tree(ax, box, *, shunt: bool) -> None:
    x0, y0, w, h = box
    descendant = {"l", "ll", "lr", "t1", "t2", "t3", "t4"}
    def point(name):
        x, y = TREE_NODES[name]
        return x0 + w * x, y0 + h * y
    for a, b in TREE_EDGES:
        selected = a in descendant and b in descendant
        if selected:
            color = COLORS["shunting"]
            alpha = 0.26 if shunt else 0.90
            lw = 0.75 if shunt else 1.25
        else:
            color = COLORS["dend"]
            alpha = 0.85
            lw = 1.0
        pa, pb = point(a), point(b)
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, alpha=alpha,
                lw=lw, solid_capstyle="round", zorder=2)
    soma = point("s")
    ax.add_patch(Circle(soma, 0.025, facecolor=COLORS["soma"],
                        edgecolor=COLORS["edge"], lw=LW_EDGE, zorder=4))
    focal = (
        0.55 * point("j1")[0] + 0.45 * point("l")[0],
        0.55 * point("j1")[1] + 0.45 * point("l")[1],
    )
    ax.scatter(focal[0], focal[1], s=26, facecolor="white",
               edgecolor=COLORS["inh"], linewidth=LW_ERR, zorder=5)
    ax.scatter(focal[0], focal[1], s=8, color=COLORS["inh"], zorder=6)
    return focal


def build_focal_schematic() -> None:
    fig, ax = plt.subplots(figsize=(FIG_W / 3, 1.62))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    left = _mini_tree(ax, (0.02, 0.23, 0.43, 0.62), shunt=False)
    right = _mini_tree(ax, (0.55, 0.23, 0.43, 0.62), shunt=True)

    ax.plot([0.10, 0.90], [0.91, 0.91], color=COLORS["mute"], lw=LW_HAIR)
    ax.plot([0.10, 0.10], [0.91, 0.885], color=COLORS["mute"], lw=LW_HAIR)
    ax.plot([0.90, 0.90], [0.91, 0.885], color=COLORS["mute"], lw=LW_HAIR)
    ax.text(0.50, 0.945, r"same focal $\Delta V$", ha="center", va="center",
            fontsize=PT_SMALL, color=COLORS["mute"])

    ax.annotate(
        "matched current", xy=left, xytext=(0.06, 0.18),
        textcoords="axes fraction", ha="left", va="center",
        fontsize=PT_SMALL, color=COLORS["additive"],
        arrowprops={"arrowstyle": "-|>", "color": COLORS["additive"], "lw": LW_EDGE},
    )
    ax.annotate(
        r"shunt $g_{\rm sh}$", xy=right, xytext=(0.62, 0.18),
        textcoords="axes fraction", ha="left", va="center",
        fontsize=PT_SMALL, color=COLORS["inh"],
        arrowprops={"arrowstyle": "-|>", "color": COLORS["inh"], "lw": LW_EDGE},
    )
    ax.text(0.24, 0.06, "matched additive", ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["additive"])
    ax.text(0.76, 0.06, "focal shunt", ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["inh"])
    ax.text(0.74, 0.31, "descendant credit\nattenuated", ha="center", va="center",
            fontsize=PT_SMALL, color=COLORS["shunting"])
    save(fig, "fig_main_focal_schematic", audit_overlap=False)


# ---------------------------------------------------------------------------
# Figure 9A: structure-function boundary
# ---------------------------------------------------------------------------


def build_structure_function_summary() -> None:
    original = json.loads(
        (SOURCE / "figure5" / "functional_summary.json").read_text(encoding="utf-8")
    )["tests"]["partial_shared_path_r"]
    expanded = json.loads(
        (SOURCE / "functional_topology_all_scans" / "summary.json").read_text(
            encoding="utf-8"
        )
    )["metrics"]["shared_path_partial_r"]
    rows = [
        (
            "prespecified scan",
            float(original["mean"]),
            *map(float, original["bootstrap_95_ci_mean"]),
            "3/7 positive",
        ),
        (
            "all eligible scans",
            float(expanded["mean"]),
            *map(float, expanded["target_bootstrap_ci95"]),
            "3/7 positive",
        ),
    ]
    fig, ax = plt.subplots(figsize=(FIG_W / 3, 1.64))
    fig.subplots_adjust(left=0.43, right=0.98, bottom=0.28, top=0.94)
    ax.axvline(0, color=COLORS["mute"], lw=LW_REF, ls="--", zorder=0)
    colors = [COLORS["shunting"], COLORS["additive"]]
    for index, ((label, mean, low, high, sign), color) in enumerate(
        zip(rows, colors, strict=True)
    ):
        ypos = 1 - index
        ax.errorbar(
            mean, ypos, xerr=[[mean - low], [high - mean]], fmt="D",
            ms=5.0, color=color, mfc="white", mec=color, mew=LW_ERR,
            lw=LW_ERR, capsize=ERR_CAPSIZE, zorder=3,
        )
    ax.set_yticks(
        [1, 0],
        ["one scan / target\n3/7 positive", "all eligible scans\n3/7 positive"],
    )
    ax.set_xlim(-0.46, 0.34)
    ax.set_ylim(-0.55, 1.55)
    ax.set_xlabel("partial ancestry effect")
    style_axis(ax, grid="x")
    save(fig, "fig_main_structure_function_summary")


def main() -> None:
    apply_neurips_style()
    mpl.rcParams["lines.markeredgewidth"] = LW_EDGE
    build_phase_plane()
    build_physical_task_schematic()
    build_figure6()
    build_mapped_reconstruction()
    build_ancestry_addresses()
    build_wire_efficiency()
    build_cross_animal()
    build_focal_schematic()
    build_structure_function_summary()


if __name__ == "__main__":
    main()

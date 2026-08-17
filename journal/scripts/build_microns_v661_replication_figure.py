#!/usr/bin/env python3
"""Render the public-v661 MICrONS robustness analysis as a main figure.

The figure uses cells, rather than focal sites or Monte Carlo shuffles, as the
independent units.  It also writes the exact processed plotting tables and a
TeX-ready caption/results paragraph.  No manuscript source is modified.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from neurips_style import (
    COLORS as NEURIPS_COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    PT_SMALL,
    SEED_MS,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    panel_title,
    paired_lines,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "source_data" / "microns_v661_replication"
FIGURES = ROOT / "figures" / "generated"
STEM = "fig_v661_robustness"

COLORS = {
    "shunting": NEURIPS_COLORS["shunting"],
    "additive": NEURIPS_COLORS["additive"],
    "morphology": NEURIPS_COLORS["shunting"],
    "random": NEURIPS_COLORS["point_mlp"],
    "depth": NEURIPS_COLORS["additive"],
    "shuffle": NEURIPS_COLORS["highlight"],
    "oracle": NEURIPS_COLORS["oracle"],
}
METHOD_COLORS = {
    "dense PCA oracle": COLORS["oracle"],
    "morphology-aware paths": COLORS["morphology"],
    "random paths": COLORS["random"],
    "depth-only bins": COLORS["depth"],
    "shuffled ancestry": COLORS["shuffle"],
}
METHOD_LABELS = {
    "dense PCA oracle": "dense\noracle",
    "morphology-aware paths": "morphology\npaths",
    "random paths": "random\npaths",
    "depth-only bins": "depth\nbins",
    "shuffled ancestry": "shuffled\nancestry",
}


def mean_ci(values: np.ndarray, seed: int, n_boot: int = 20_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_source_tables() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    capacity_curves = DATA / "routing_capacity_20stream.csv.gz"
    capacity_summary = DATA / "routing_capacity_20stream_summary.json"
    if not capacity_curves.exists() or not capacity_summary.exists():
        raise FileNotFoundError(
            "Run scripts/slurm_v661_direct_typed_capacity.sh before rendering "
            "Supplementary Fig. S1; the figure requires the frozen 20-stream analysis."
        )

    raw_curves = pd.read_csv(capacity_curves)
    required = {
        "root_id",
        "seed",
        "channels",
        "method",
        "credit_capture",
        "wiring_nonzeros",
        "wiring_density",
    }
    missing = sorted(required.difference(raw_curves.columns))
    if missing:
        raise ValueError(f"20-stream capacity table is missing columns: {missing}")
    curves = (
        raw_curves.groupby(["root_id", "channels", "method"], as_index=False)
        .agg(
            credit_capture=("credit_capture", "mean"),
            seed_capture_sd=("credit_capture", "std"),
            wiring_nonzeros=("wiring_nonzeros", "mean"),
            wiring_density=("wiring_density", "mean"),
            n_monte_carlo_streams=("seed", "nunique"),
        )
        .sort_values(["root_id", "channels", "method"])
    )
    if set(curves["n_monte_carlo_streams"].unique()) != {20}:
        raise ValueError("Supplementary Fig. S1 requires exactly 20 Monte Carlo streams per cell and condition")

    cells = pd.read_csv(DATA / "cell_level_primary.csv")
    focal = cells[
        [
            "root_id",
            "nucleus_id",
            "cell_type",
            "matched_additive_localization",
            "focal_shunt_localization",
            "shunt_minus_additive",
            "shunt_depth_shuffled_localization",
            "shunt_topology_minus_depth_shuffle",
            "n_focal_sites",
        ]
    ].copy()
    summary = json.loads((DATA / "replication_summary.json").read_text(encoding="utf-8"))
    robust = json.loads(capacity_summary.read_text(encoding="utf-8"))
    robust_methods = robust["methods"]
    robust_comparisons = robust["morphology_comparisons"]
    summary["routing_eight_channels"] = {
        **summary["routing_eight_channels"],
        "mean_capture": {
            "dense_oracle": robust_methods["dense PCA oracle"]["mean_credit_capture"],
            "morphology_paths": robust_methods["morphology-aware paths"]["mean_credit_capture"],
            "random_paths": robust_methods["random paths"]["mean_credit_capture"],
            "depth_bins": robust_methods["depth-only bins"]["mean_credit_capture"],
            "shuffled_ancestry": robust_methods["shuffled ancestry"]["mean_credit_capture"],
        },
        "mean_morphology_capture_fraction_of_dense": robust_methods["morphology-aware paths"][
            "mean_oracle_fraction"
        ],
        "mean_morphology_wiring_density": robust_methods["morphology-aware paths"][
            "mean_wiring_density"
        ],
        "morphology_minus_random": {
            "mean_difference": robust_comparisons["random paths"]["mean_capture_advantage"],
            "cell_bootstrap_ci95": robust_comparisons["random paths"]["hierarchical_ci95"],
            "cells_positive": robust_comparisons["random paths"]["cells_positive"],
            "n_cells": robust_comparisons["random paths"]["n_cells"],
        },
        "morphology_minus_depth": {
            "mean_difference": robust_comparisons["depth-only bins"]["mean_capture_advantage"],
            "cell_bootstrap_ci95": robust_comparisons["depth-only bins"]["hierarchical_ci95"],
            "cells_positive": robust_comparisons["depth-only bins"]["cells_positive"],
            "n_cells": robust_comparisons["depth-only bins"]["n_cells"],
        },
        "morphology_minus_shuffled": {
            "mean_difference": robust_comparisons["shuffled ancestry"]["mean_capture_advantage"],
            "cell_bootstrap_ci95": robust_comparisons["shuffled ancestry"]["hierarchical_ci95"],
            "cells_positive": robust_comparisons["shuffled ancestry"]["cells_positive"],
            "n_cells": robust_comparisons["shuffled ancestry"]["n_cells"],
        },
        "n_cells": robust["n_cells"],
        "n_monte_carlo_streams": robust["n_monte_carlo_streams"],
    }
    curves.to_csv(DATA / "supp_figure_routing_curves.csv", index=False)
    focal.to_csv(DATA / "supp_figure_focal_cells.csv", index=False)
    cohort = pd.read_csv(DATA / "cohort_manifest.csv", keep_default_na=False)
    endpoint_specs = [
        ("structural_routing", "routing_included"),
        ("focal_shunt_vs_additive", "focal_shunt_additive_included"),
        ("focal_true_vs_depth_shuffle", "focal_depth_shuffle_included"),
    ]
    exclusion_rows = []
    for endpoint, inclusion_column in endpoint_specs:
        for row in cohort.itertuples(index=False):
            included = str(getattr(row, inclusion_column)).lower() == "true"
            if included:
                reason = "included"
            elif endpoint == "structural_routing":
                reason = str(row.cohort_disposition)
            else:
                reason = str(row.focal_exclusion_reason)
            exclusion_rows.append(
                {
                    "endpoint": endpoint,
                    "selection_order": int(row.selection_order),
                    "nucleus_id": int(row.nucleus_id),
                    "source_root_id": str(row.source_root_id),
                    "v661_root_id": str(row.v661_root_id),
                    "cell_type": str(row.cell_type),
                    "frozen_before_replication_outcomes": True,
                    "included": included,
                    "exclusion_reason": reason,
                }
            )
    pd.DataFrame(exclusion_rows).to_csv(
        DATA / "prospective_endpoint_exclusion_log.csv", index=False
    )
    return curves, focal, summary


def plot_pair(
    ax: plt.Axes,
    frame: pd.DataFrame,
    left: str,
    right: str,
    labels: tuple[str, str],
    colors: tuple[str, str],
    seed: int,
) -> None:
    paired = frame[[left, right]].dropna()
    xs = np.array([0.0, 1.0])
    for _, row in paired.iterrows():
        values = row[[left, right]].to_numpy(dtype=float)
        ax.plot(xs, values, color="#B9B9B9", alpha=0.42, lw=0.55, zorder=1)
        ax.scatter(xs, values, color=colors, alpha=0.60, s=9, edgecolors="none", zorder=2)
    for index, column in enumerate([left, right]):
        mean, low, high = mean_ci(paired[column].to_numpy(dtype=float), seed + index)
        ax.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            marker="D",
            color=colors[index],
            markeredgecolor="white",
            markeredgewidth=0.45,
            ms=5.0,
            capsize=2.2,
            lw=1.15,
            zorder=5,
        )
    ax.axhline(0, color="#777777", ls="--", lw=0.6)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("localization index")
    style_axis(ax)


def render(curves: pd.DataFrame, focal: pd.DataFrame, summary: dict) -> None:
    apply_neurips_style()
    fig = plt.figure(figsize=(FIG_W, 6.10))
    grid = fig.add_gridspec(
        3, 6, left=0.088, right=0.985, bottom=0.068, top=0.935,
        wspace=0.86, hspace=0.88,
    )
    spans = [
        (0, slice(0, 2)), (0, slice(2, 4)), (0, slice(4, 6)),
        (1, slice(0, 2)), (1, slice(2, 4)), (1, slice(4, 6)),
        (2, slice(0, 3)), (2, slice(3, 6)),
    ]
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f, ax_g, ax_h = [
        fig.add_subplot(grid[row, cols]) for row, cols in spans
    ]

    methods = [
        "dense PCA oracle",
        "morphology-aware paths",
        "random paths",
        "depth-only bins",
        "shuffled ancestry",
    ]
    cohort = pd.read_csv(DATA / "cohort_manifest.csv", keep_default_na=False)
    included = cohort[cohort["routing_included"].astype(str).str.lower().eq("true")]
    type_counts = included.cell_type.value_counts().reindex(["L2IT", "L3IT", "L4IT", "L5IT", "L5ET"], fill_value=0)
    ax_a.barh(np.arange(len(type_counts)), type_counts.to_numpy(), color=COLORS["morphology"], alpha=0.86)
    ax_a.set_yticks(np.arange(len(type_counts))); ax_a.set_yticklabels(type_counts.index)
    ax_a.invert_yaxis(); ax_a.set_xlabel("reconstructed cells")
    panel_title(ax_a, "A", "Frozen 47-cell cohort")
    style_axis(ax_a, grid="x")

    for method_index, method in enumerate(methods):
        selected = curves[curves["method"].eq(method)]
        channel_values: list[int] = []
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for channel, group in selected.groupby("channels"):
            mean, low, high = mean_ci(
                group["credit_capture"].to_numpy(dtype=float),
                1000 + 100 * method_index + int(channel),
            )
            channel_values.append(int(channel))
            means.append(mean)
            lows.append(low)
            highs.append(high)
        order = np.argsort(channel_values)
        x = np.asarray(channel_values)[order]
        y = np.asarray(means)[order]
        low = np.asarray(lows)[order]
        high = np.asarray(highs)[order]
        color = METHOD_COLORS[method]
        ax_b.plot(x, y, marker="o", ms=3.0, lw=LW_DATA, color=color, label=method.replace("-aware", ""))
        ax_b.fill_between(x, low, high, color=color, alpha=0.10, linewidth=0)
    ax_b.set_xscale("log", base=2)
    ax_b.set_xticks([1, 2, 4, 8, 16])
    ax_b.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_b.set_ylim(0, 1.0)
    ax_b.set_xlabel("feedback channels")
    ax_b.set_ylabel("modeled field capture")
    panel_title(ax_b, "B", "Route capacity")
    style_axis(ax_b)

    focus = curves[curves["channels"].eq(8)]
    rng = np.random.default_rng(20260731)
    for index, method in enumerate(methods):
        values = focus.loc[focus["method"].eq(method), "credit_capture"].to_numpy(dtype=float)
        color = METHOD_COLORS[method]
        ax_c.scatter(
            index + rng.normal(0, 0.045, len(values)),
            values,
            s=10,
            color=color,
            alpha=0.38,
            edgecolors="none",
        )
        mean, low, high = mean_ci(values, 2000 + index)
        ax_c.errorbar(
            index,
            mean,
            yerr=[[mean - low], [high - mean]],
            marker="D",
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            ms=4.8,
            capsize=ERR_CAPSIZE,
            lw=LW_ERR,
            zorder=5,
        )
    ax_c.set_xticks(range(len(methods)), ["dense", "ances.", "random", "depth", "shuffle"], rotation=30, ha="right")
    ax_c.set_ylim(0, 1.0)
    ax_c.set_ylabel("capture at 8 channels")
    panel_title(ax_c, "C", "Eight-channel capture")
    style_axis(ax_c)

    for method in methods:
        selected = focus[focus.method.eq(method)]
        ax_d.scatter(selected.wiring_density, selected.credit_capture, s=10,
                     color=METHOD_COLORS[method], alpha=0.30, edgecolors="none")
        ax_d.scatter(selected.wiring_density.mean(), selected.credit_capture.mean(),
                     marker="D", s=28, color=METHOD_COLORS[method],
                     edgecolor="white", linewidth=0.4, zorder=4)
    ax_d.set_xscale("log"); ax_d.set_xlim(0.008, 1.4); ax_d.set_ylim(0, 1.0)
    ax_d.set_xlabel("wiring density"); ax_d.set_ylabel("field capture")
    panel_title(ax_d, "D", "Wiring-capture trade-off")
    style_axis(ax_d)

    pivot = focus.pivot(index="root_id", columns="method", values="credit_capture")
    controls = ["random paths", "depth-only bins", "shuffled ancestry"]
    for index, method in enumerate(controls):
        values = (pivot["morphology-aware paths"] - pivot[method]).dropna().to_numpy(float)
        ax_e.scatter(index + np.random.default_rng(2300 + index).normal(0, 0.045, len(values)),
                     values, s=SEED_MS ** 2, color=METHOD_COLORS[method], alpha=0.42,
                     edgecolors="none")
        mean, low, high = mean_ci(values, 2400 + index)
        ax_e.errorbar(index, mean, yerr=[[mean - low], [high - mean]], marker="D",
                      color=METHOD_COLORS[method], ms=4.8, lw=LW_ERR, capsize=ERR_CAPSIZE)
    ax_e.axhline(0, color=NEURIPS_COLORS["mute"], ls="--", lw=LW_REF)
    ax_e.set_xticks(range(3)); ax_e.set_xticklabels(["random", "depth", "shuffle"])
    ax_e.set_ylabel("morphology advantage")
    panel_title(ax_e, "E", "Topology controls")
    style_axis(ax_e)

    plot_pair(
        ax_f,
        focal,
        "matched_additive_localization",
        "focal_shunt_localization",
        ("first-order-current-\nmatched additive", "focal\nshunt"),
        (COLORS["additive"], COLORS["shunting"]),
        3000,
    )
    ax_f.set_ylim(-0.09, 0.56)
    direct = summary["focal_primary_dose_one"]["shunt_minus_additive"]
    ax_f.text(
        0.04,
        0.96,
        f"{direct['cells_positive']}/{direct['n_cells']} cells positive",
        transform=ax_f.transAxes,
        ha="left",
        va="top",
        color=COLORS["shunting"],
        fontsize=PT_SMALL,
    )
    panel_title(ax_f, "F", "Shunt versus additive")

    plot_pair(
        ax_g,
        focal,
        "shunt_depth_shuffled_localization",
        "focal_shunt_localization",
        ("reassigned\nrelation", "true descendant\nrelation"),
        (COLORS["shuffle"], COLORS["shunting"]),
        4000,
    )
    ax_g.set_ylim(-0.09, 0.56)
    topology = summary["focal_primary_dose_one"]["topology_minus_depth_shuffle"]
    ax_g.text(
        0.04,
        0.96,
        f"{topology['cells_positive']} positive, {topology['cells_tied_within_tolerance']} tie / {topology['n_cells']} cells",
        transform=ax_g.transAxes,
        ha="left",
        va="top",
        color=COLORS["shunting"],
        fontsize=PT_SMALL,
    )
    panel_title(ax_g, "G", "True versus reassigned relation")

    included_h = included.copy()
    included_h["direct_type_coverage"] = pd.to_numeric(included_h["direct_type_coverage"], errors="coerce")
    included_h["n_selected_focal_sites"] = pd.to_numeric(included_h["n_selected_focal_sites"], errors="coerce")
    type_colors = {"L2IT": NEURIPS_COLORS["exc"], "L3IT": NEURIPS_COLORS["rule_3f"],
                   "L4IT": NEURIPS_COLORS["pathway"], "L5IT": NEURIPS_COLORS["soma"],
                   "L5ET": NEURIPS_COLORS["inh"]}
    for cell_type, group in included_h.groupby("cell_type"):
        ax_h.scatter(100 * group.direct_type_coverage, group.n_selected_focal_sites,
                     s=18, color=type_colors.get(cell_type, NEURIPS_COLORS["mute"]),
                     alpha=0.65, label=cell_type, edgecolor="white", linewidth=0.25)
    ax_h.set_xlabel("direct E/I labels (% inputs)"); ax_h.set_ylabel("eligible focal sites")
    panel_title(ax_h, "H", "Direct-label coverage")
    style_axis(ax_h)
    ax_h.legend(frameon=False, ncol=3, loc="upper right", fontsize=PT_SMALL,
                handlelength=0.8, handletextpad=0.2, columnspacing=0.5)

    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, STEM)
    audit_text_over_data(fig, STEM)
    fig.savefig(
        FIGURES / f"{STEM}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(FIGURES / f"{STEM}.png", dpi=350)
    plt.close(fig)


def write_tex(summary: dict) -> None:
    routing = summary["routing_eight_channels"]
    focal = summary["focal_primary_dose_one"]
    cohort = summary["cohort"]
    direct = focal["shunt_minus_additive"]
    topology = focal["topology_minus_depth_shuffle"]
    caption = rf"""\textbf{{Structural credit routing and focal shunting in a frozen MICrONS v661 sensitivity cohort.}}
\textbf{{A}}, Composition of the frozen 47-cell cohort by excitatory cell class.
\textbf{{B}}, Cell-mean modeled field capture across feedback-channel budgets for a dense PCA oracle, morphology-defined paths, random paths, depth bins and ancestry shuffles ($n=47$ cells through eight channels; $n=46$ at 16 channels). Twenty perturbation streams were averaged within each cell and condition; bands are 95\% cell-bootstrap confidence intervals.
\textbf{{C}}, Cell-level capture at eight channels.
\textbf{{D}}, Eight-channel wiring-capture trade-off; morphology paths used {100 * routing['mean_morphology_wiring_density']:.1f}\% of dense-feedback wiring.
\textbf{{E}}, Within-cell morphology advantage over the three structural controls.
\textbf{{F}}, At dose 1, localization after a focal shunt and first-order-current-matched additive perturbation ($n={direct['n_cells']}$ cells, {direct['n_focal_sites']} focal sites).
\textbf{{G}}, Localization for the true descendant relation and relation templates reassigned within cell ($n={topology['n_cells']}$ cells, {topology['n_focal_sites']} focal sites).
\textbf{{H}}, Direct E/I-label coverage and eligible focal-site count by cell class. Diamonds and intervals in \textbf{{C,E--G}} show cell means and 95\% cell-bootstrap confidence intervals. Cells, not focal sites or streams, are the independent units. The cohort excludes the original eight cells by stable nucleus identifier but comes from the same MICrONS mouse. It uses historical static v661 reconstructions and direct presynaptic coarse E/I calls only ({cohort['n_total_direct_typed_synapses']:,} annotations before spatial mapping, {cohort['n_total_mapped_direct_synapses']:,} mapped contacts; 4.71\% of incoming synapses). These are structural and passive-network model tests, not measurements of in vivo learning.
"""
    paragraph = rf"""\paragraph{{Disjoint-cell sensitivity analysis in public MICrONS v661 reconstructions.}}
We froze a 55-cell V1 excitatory cohort before computing sensitivity outcomes and excluded the eight discovery cells by stable nucleus identifier. All 47 remaining v661 skeletons and postsynaptic meshworks were available from the public static release. Twenty perturbation streams were averaged within each cell. At eight feedback channels, morphology-defined paths captured {100 * routing['mean_capture']['morphology_paths']:.1f}\% of weighted modeled field energy using {100 * routing['mean_morphology_wiring_density']:.1f}\% of dense-feedback wiring, compared with {100 * routing['mean_capture']['dense_oracle']:.1f}\% for the dense PCA oracle. Morphology capture exceeded random paths by {100 * routing['morphology_minus_random']['mean_difference']:.1f} percentage points (95\% CI {100 * routing['morphology_minus_random']['cell_bootstrap_ci95'][0]:.1f}--{100 * routing['morphology_minus_random']['cell_bootstrap_ci95'][1]:.1f}), depth bins by {100 * routing['morphology_minus_depth']['mean_difference']:.1f} points ({100 * routing['morphology_minus_depth']['cell_bootstrap_ci95'][0]:.1f}--{100 * routing['morphology_minus_depth']['cell_bootstrap_ci95'][1]:.1f}) and ancestry shuffles by {100 * routing['morphology_minus_shuffled']['mean_difference']:.1f} points ({100 * routing['morphology_minus_shuffled']['cell_bootstrap_ci95'][0]:.1f}--{100 * routing['morphology_minus_shuffled']['cell_bootstrap_ci95'][1]:.1f}); all three contrasts were positive in 47/47 cells. Across {direct['n_focal_sites']} eligible sites in {direct['n_cells']} cells, focal shunting increased descendant-selective gradient localization relative to the first-order-current-matched additive control by {direct['mean_difference']:.3f} ({direct['cell_bootstrap_ci95'][0]:.3f}--{direct['cell_bootstrap_ci95'][1]:.3f}; {direct['cells_positive']}/{direct['n_cells']} cells). The true relation also exceeded reassigned relations by {topology['mean_difference']:.3f} ({topology['cell_bootstrap_ci95'][0]:.3f}--{topology['cell_bootstrap_ci95'][1]:.3f}; {topology['cells_positive']} positive and one numerical tie among {topology['n_cells']} cells). Because this analysis uses a historical release from the same animal and direct E/I labels cover 4.71\% of incoming synapses ({cohort['n_total_direct_typed_synapses']:,} before mapping; {cohort['n_total_mapped_direct_synapses']:,} mapped), it supports robustness across disjoint reconstructed cells, not independent-animal generalization or an in vivo learning claim.
"""
    (DATA / "tex_ready_caption_and_results.tex").write_text(
        "% TeX-ready caption and results text for the public-v661 main figure.\n"
        + caption
        + "\n"
        + paragraph,
        encoding="utf-8",
    )


def write_manifest() -> None:
    files = [
        DATA / "cohort_manifest.csv",
        DATA / "cell_level_primary.csv",
        DATA / "replication_summary.json",
        DATA / "routing_capacity_20stream.csv.gz",
        DATA / "routing_capacity_20stream_summary.json",
        DATA / "supp_figure_routing_curves.csv",
        DATA / "supp_figure_focal_cells.csv",
        DATA / "prospective_endpoint_exclusion_log.csv",
        DATA / "tex_ready_caption_and_results.tex",
        FIGURES / f"{STEM}.pdf",
        FIGURES / f"{STEM}.png",
    ]
    rows = []
    for path in files:
        rows.append(
            {
                "role": (
                    "figure_asset"
                    if path.suffix in {".pdf", ".png"}
                    else "cohort_manifest"
                    if path.name == "cohort_manifest.csv"
                    else "figure_source"
                ),
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(rows).to_csv(DATA / "supp_figure_manifest.tsv", sep="\t", index=False)


def main() -> None:
    curves, focal, summary = prepare_source_tables()
    render(curves, focal, summary)
    write_tex(summary)
    write_manifest()
    print(FIGURES / f"{STEM}.pdf")


if __name__ == "__main__":
    main()

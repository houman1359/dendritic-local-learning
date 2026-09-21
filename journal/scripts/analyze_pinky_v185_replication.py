#!/usr/bin/env python3
"""Run and summarize the frozen independent-animal Pinky-v185 replication."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from journal_style import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_ERR,
    LW_HAIR,
    MARKER_MS,
    PT_LEGEND,
    PT_SMALL,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


DEFAULT_PREPARED = ROOT / "data" / "pinky_v185" / "prepared"
DEFAULT_COHORT = ROOT / "source_data" / "pinky_v185_replication" / "cohort_manifest.csv"
DEFAULT_OUTDIR = ROOT / "source_data" / "pinky_v185_replication" / "routing"
DEFAULT_FIGURE = (
    ROOT / "figures" / "supplementary" / "figure_S27_panels_A-D"
)
MAIN_CURVES = ROOT / "source_data" / "microns_v661_replication" / "routing" / "feedback_compression_curves.csv"
COMMON_ANALYSIS = ROOT / "code" / "reconstructed_tree" / "analyze_microns_morphology_credit.py"
CONTROLS = ("random paths", "depth-only bins", "shuffled ancestry")
METHOD_ORDER = (
    "dense PCA oracle",
    "morphology-aware paths",
    "random paths",
    "shuffled ancestry",
    "depth-only bins",
)
METHOD_STYLE = {
    "dense PCA oracle": (COLORS["oracle"], "--", "D"),
    "morphology-aware paths": (COLORS["dend"], "-", "o"),
    "random paths": (COLORS["point_mlp"], ":", "s"),
    "shuffled ancestry": (COLORS["highlight"], "-.", "^"),
    # COLORS["pathway"] is the same #8F66CD as COLORS["oracle"], so the
    # depth control and the dense oracle drew as two crossing violet
    # lines.  Depth is blue on the other anatomy sheets (S20, S22).
    "depth-only bins": (COLORS["additive"], (0, (3, 1, 1, 1)), "v"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _common_module():
    spec = importlib.util.spec_from_file_location("common_microns_routing", COMMON_ANALYSIS)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {COMMON_ANALYSIS}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 50_000) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan"), float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(values.mean()), float(low), float(high)]


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    null = [
        abs(float(np.mean(values * np.asarray(signs, dtype=float))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def k4_contrasts(
    curves: pd.DataFrame, *, compute_exact_descriptive_p: bool = True
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fixed = curves[curves["channels"].eq(4)].pivot_table(
        index="root_id", columns="method", values="residual"
    )
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for control_index, control in enumerate(CONTROLS):
        difference = fixed[control] - fixed["morphology-aware paths"]
        # Positive means that morphology captures more field energy.
        for root_id, value in difference.items():
            rows.append(
                {
                    "root_id": int(root_id),
                    "control": control,
                    "morphology_capture_advantage": float(value),
                }
            )
        descriptive_p = (
            exact_sign_flip_p(difference.to_numpy(dtype=float))
            if compute_exact_descriptive_p
            else None
        )
        summary[control] = {
            "n_cells": int(len(difference)),
            "mean_ci95_cell_bootstrap": bootstrap_mean(
                difference.to_numpy(dtype=float), 20260820 + control_index
            ),
            "median": float(difference.median()),
            "positive_cells": int((difference > 0).sum()),
            "descriptive_cell_sign_flip_p_two_sided": descriptive_p,
            "descriptive_cell_sign_flip_method": (
                "exhaustive 2^n mean sign-flip"
                if compute_exact_descriptive_p
                else "not recomputed for reference cohort"
            ),
        }
    return pd.DataFrame(rows), summary


def analyze(
    prepared: Path,
    cohort_path: Path,
    outdir: Path,
    *,
    max_mapping_distance_um: float,
    n_shuffles: int,
    seed: int,
) -> dict[str, Any]:
    common = _common_module()
    cohort = pd.read_csv(cohort_path)
    metric_rows: list[dict[str, Any]] = []
    segments_all: list[pd.DataFrame] = []
    curves_all: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in cohort.itertuples(index=False):
        root_id = int(row.root_id)
        print(f"Analyzing Pinky root {root_id}", flush=True)
        try:
            metrics, segments, _mapped, curves = common.analyze_cell(
                root_id=root_id,
                datadir=prepared,
                max_mapping_distance_um=max_mapping_distance_um,
                e_scale=0.5,
                i_scale=0.5,
                n_shuffles=n_shuffles,
                classification_mode="typed_only",
                min_target_probability=1.0,
                seed=seed,
            )
        except Exception as error:  # preserve every selected-cell failure
            errors.append(
                {
                    "root_id": root_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            continue
        n_typed = int(metrics["n_e_synapses"] + metrics["n_i_synapses"])
        included = n_typed >= 20 and int(metrics["n_i_synapses"]) >= 3
        metrics.update(
            {
                "selection_order": int(row.selection_order),
                "qc_typed_synapses": n_typed,
                "qc_included": included,
                "qc_exclusion_reason": (
                    ""
                    if included
                    else "fewer than 20 mapped typed inputs or 3 inhibitory inputs"
                ),
            }
        )
        metric_rows.append(metrics)
        segments["qc_included"] = included
        segments_all.append(segments)
        for curve in curves:
            curve["qc_included"] = included
            curves_all.append(curve)

    cell_metrics = pd.DataFrame(metric_rows)
    if cell_metrics.empty:
        raise RuntimeError(f"All selected cells failed: {errors}")
    segments = pd.concat(segments_all, ignore_index=True)
    curves = pd.DataFrame(curves_all)
    included_roots = set(
        cell_metrics.loc[cell_metrics["qc_included"], "root_id"].astype(int)
    )
    included_curves = curves[curves["root_id"].isin(included_roots)].copy()
    if len(included_roots) < 2:
        raise RuntimeError(f"Only {len(included_roots)} cells passed the frozen QC gate")

    outdir.mkdir(parents=True, exist_ok=True)
    cell_metrics.to_csv(outdir / "cell_metrics.csv", index=False)
    segments.to_csv(outdir / "segment_metrics.csv", index=False)
    curves.to_csv(outdir / "feedback_compression_curves_all_selected.csv", index=False)
    included_curves.to_csv(outdir / "feedback_compression_curves.csv", index=False)
    pinky_contrasts, pinky_summary = k4_contrasts(included_curves)
    pinky_contrasts["animal"] = "Pinky v185"

    main_curves = pd.read_csv(MAIN_CURVES)
    main_contrasts, main_summary = k4_contrasts(
        main_curves, compute_exact_descriptive_p=False
    )
    main_contrasts["animal"] = "minnie65 v661"
    contrasts = pd.concat([main_contrasts, pinky_contrasts], ignore_index=True)
    contrasts.to_csv(outdir / "k4_cross_animal_contrasts.csv", index=False)

    agreement = {
        control: {
            "minnie65_mean": main_summary[control]["mean_ci95_cell_bootstrap"][0],
            "pinky_mean": pinky_summary[control]["mean_ci95_cell_bootstrap"][0],
            "same_positive_direction": bool(
                main_summary[control]["mean_ci95_cell_bootstrap"][0] > 0
                and pinky_summary[control]["mean_ci95_cell_bootstrap"][0] > 0
            ),
        }
        for control in CONTROLS
    }
    payload = {
        "status": "complete" if not errors else "complete_with_selected_cell_errors",
        "dataset": "MICrONS phase-1 Pinky v185",
        "biological_unit": "one independent P36 male mouse",
        "n_selected": int(len(cohort)),
        "n_analyzed": int(len(cell_metrics)),
        "n_qc_included": int(len(included_roots)),
        "n_errors": int(len(errors)),
        "errors": errors,
        "qc_gate": "at least 20 mapped typed incoming synapses and at least 3 inhibitory",
        "classification": "presynaptic soma-valence labels only",
        "pinky_k4": pinky_summary,
        "minnie65_k4_reference": main_summary,
        "cross_animal_directional_replication": agreement,
        "claim_boundary": (
            "The animal is the biological replication unit. Cell-bootstrap intervals "
            "quantify within-volume stability and are not population-level animal inference."
        ),
        "parameters": {
            "max_mapping_distance_um": float(max_mapping_distance_um),
            "e_scale": 0.5,
            "i_scale": 0.5,
            "n_shuffles": int(n_shuffles),
            "seed": int(seed),
        },
        "source_code": {
            Path(__file__).name: sha256(Path(__file__).resolve()),
            COMMON_ANALYSIS.name: sha256(COMMON_ANALYSIS),
        },
    }
    (outdir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "summary": payload,
        "cohort": cohort,
        "cell_metrics": cell_metrics,
        "curves": included_curves,
        "contrasts": contrasts,
    }


def make_figure(result: dict[str, Any], figure_stem: Path) -> None:
    apply_neurips_style()
    cohort = result["cohort"]
    cell_metrics = result["cell_metrics"]
    curves = result["curves"]
    contrasts = result["contrasts"]
    included = set(cell_metrics.loc[cell_metrics["qc_included"], "root_id"].astype(int))

    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 5.45))
    ax = axes[0, 0]
    panel_title(ax, "A", "Independent-animal sample")
    selected = cohort.copy()
    colors = np.where(selected["root_id"].isin(included), COLORS["dend"], COLORS["mute"])
    ax.scatter(selected["x_nm"] / 1_000, selected["y_nm"] / 1_000, c=colors, s=25)
    for row in selected.itertuples(index=False):
        ax.text(row.x_nm / 1_000 + 1.2, row.y_nm / 1_000, str(row.selection_order + 1), fontsize=PT_SMALL)
    ax.set_xlabel("volume x (µm)")
    ax.set_ylabel("volume y (µm)")
    ax.set_aspect("equal", adjustable="datalim")
    style_axis(ax, grid="none")

    ax = axes[0, 1]
    panel_title(ax, "B", "Sparse route reconstruction")
    grouped = curves.groupby(["method", "channels"])["residual"]
    summary = grouped.agg(["mean", "sem"]).reset_index()
    for method in METHOD_ORDER:
        part = summary[summary["method"].eq(method)].sort_values("channels")
        if part.empty:
            continue
        color, linestyle, marker = METHOD_STYLE[method]
        ax.errorbar(
            part["channels"],
            part["mean"],
            yerr=part["sem"].fillna(0),
            color=color,
            ls=linestyle,
            marker=marker,
            ms=MARKER_MS,
            lw=LW_DATA,
            elinewidth=LW_ERR,
            capsize=ERR_CAPSIZE,
            label=method.replace("morphology-aware paths", "ancestry routes"),
        )
    ax.set_xscale("log", base=2)
    # Ticks and limits follow the measured budgets (1-8 channels); a fixed
    # tick at 16 left a dead right band beyond the last data point.
    channel_values = sorted(int(v) for v in curves["channels"].unique())
    ax.set_xticks(channel_values)
    ax.set_xlim(channel_values[0] * 0.85, channel_values[-1] * 1.18)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("feedback channels")
    ax.set_ylabel("weighted residual (lower is better)")
    style_axis(ax, grid="y")
    # The frameless legend sat on the ancestry curve and its error
    # bars; auto_clear re-places it off the data.
    clean_legend(ax, fontsize=PT_LEGEND, ncol=1, auto_clear=True)

    ax = axes[1, 0]
    panel_title(ax, "C", "K=4 direction in two animals")
    animal_style = {
        "minnie65 v661": (COLORS["point_mlp"], "s", -0.12),
        "Pinky v185": (COLORS["dend"], "o", 0.12),
    }
    for animal, (color, marker, offset) in animal_style.items():
        subset = contrasts[contrasts["animal"].eq(animal)]
        for x, control in enumerate(CONTROLS):
            values = subset.loc[
                subset["control"].eq(control), "morphology_capture_advantage"
            ].to_numpy(dtype=float)
            mean, low, high = bootstrap_mean(values, 20260830 + x)
            ax.scatter(
                np.full(len(values), x + offset),
                values,
                s=8,
                color=color,
                alpha=0.32,
                linewidths=0,
            )
            ax.errorbar(
                x + offset,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt=marker,
                color=color,
                ms=MARKER_MS,
                lw=LW_ERR,
                capsize=ERR_CAPSIZE,
                label=animal if x == 0 else None,
            )
    ax.axhline(0, color=COLORS["edge"], lw=LW_HAIR, ls="--")
    ax.set_xticks(range(3), ["random\nroutes", "depth\nbins", "shuffled\nancestry"])
    ax.set_ylabel("capture advantage")
    style_axis(ax, grid="y")
    clean_legend(ax, fontsize=PT_LEGEND)

    ax = axes[1, 1]
    panel_title(ax, "D", "Capture per nonzero coefficient")
    fixed = curves[curves["channels"].eq(4)].copy()
    fixed["capture_per_1000_nonzeros"] = (
        1.0 - fixed["residual"]
    ) / fixed["wiring_nonzeros"].clip(lower=1e-12) * 1_000
    plot_methods = ["dense PCA oracle", "morphology-aware paths", "random paths"]
    for x, method in enumerate(plot_methods):
        values = fixed.loc[
            fixed["method"].eq(method), "capture_per_1000_nonzeros"
        ].to_numpy(dtype=float)
        color, _ls, marker = METHOD_STYLE[method]
        jitter = np.linspace(-0.055, 0.055, len(values)) if len(values) else np.asarray([])
        ax.scatter(x + jitter, values, s=9, color=color, alpha=0.45, linewidths=0)
        mean, low, high = bootstrap_mean(values, 20260840 + x)
        ax.errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt=marker,
            color=color,
            ms=MARKER_MS,
            lw=LW_ERR,
            capsize=ERR_CAPSIZE,
        )
    ax.set_xticks(range(3), ["dense\nPCA", "ancestry\nroutes", "random\nroutes"])
    ax.set_ylabel("capture per 1,000 nonzeros")
    style_axis(ax, grid="y")

    fig.subplots_adjust(left=0.105, right=0.98, top=0.93, bottom=0.12, wspace=0.34, hspace=0.44)
    fig.canvas.draw()
    layout = audit_layout(fig, figure_stem.name)
    overlap = audit_text_over_data(fig, figure_stem.name)
    if layout or overlap:
        print(f"layout audit: {len(layout)} layout and {len(overlap)} text/data warnings")
    figure_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(figure_stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--max-mapping-distance-um", type=float, default=5.0)
    parser.add_argument("--n-shuffles", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()
    result = analyze(
        args.prepared,
        args.cohort,
        args.outdir,
        max_mapping_distance_um=args.max_mapping_distance_um,
        n_shuffles=args.n_shuffles,
        seed=args.seed,
    )
    make_figure(result, args.figure_stem)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

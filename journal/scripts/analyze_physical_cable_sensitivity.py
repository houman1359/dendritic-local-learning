#!/usr/bin/env python3
"""Calibrate the focal-credit perturbation to physical passive-cable units.

The original reconstructed-tree analysis intentionally normalized axial,
leak, and synaptic scales within each cell.  That is useful as a dimensionless
mechanism test, but it does not preserve the electrotonic ratio implied by
specific axial resistance and specific membrane resistance.  This script
reruns the identical focal-site selection, state-matching compensatory somatic
current, exact adjoint, and
additive control after computing axial and leak conductances in nS from cable
geometry.  Synaptic conductance remains a documented relative calibration
because MICrONS synapse size is not a conductance measurement.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PROJECT = Path(__file__).resolve().parents[1]
RECONSTRUCTED = PROJECT / "code" / "reconstructed_tree"
if str(RECONSTRUCTED) not in sys.path:
    sys.path.insert(0, str(RECONSTRUCTED))

from analyze_microns_morphology_credit import parent_map  # noqa: E402
from run_focal_shunting_credit_perturbation import analyze_cell  # noqa: E402


DEFAULT_ORIGINAL = PROJECT / "source_data" / "figure3" / "segment_metrics.csv"
DEFAULT_V661 = (
    PROJECT
    / "source_data"
    / "microns_v661_replication"
    / "routing"
    / "segment_metrics.csv"
)
DEFAULT_OUTDIR = PROJECT / "source_data" / "physical_cable_sensitivity"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def physical_conductance_system(
    segments: pd.DataFrame,
    e_scale: float,
    i_scale: float,
    excitatory_reversal: float,
    inhibitory_reversal: float,
    *,
    axial_resistivity_ohm_cm: float,
    membrane_resistance_ohm_cm2: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, int, dict[int, int], dict[int, list[int]]]:
    """Build a reciprocal passive cable matrix in nS.

    Non-root segments are cylindrical.  The root is assigned a spherical
    membrane area because its compressed edge length is zero.  Axial coupling
    uses the child segment's cross-section and length.  E/I conductances retain
    the original within-cell synapse-size calibration, but their reference is
    the median physical leak conductance rather than an independently
    normalized unit.
    """

    if axial_resistivity_ohm_cm <= 0 or membrane_resistance_ohm_cm2 <= 0:
        raise ValueError("physical cable resistances must be positive")
    electrical = segments.copy().sort_values("topological_depth").reset_index(drop=True)
    root, parents, children = parent_map(electrical)
    ids = electrical["segment_id"].astype(int).tolist()
    index = {segment: position for position, segment in enumerate(ids)}
    radius_cm = electrical["mean_radius_um"].to_numpy(dtype=float) * 1e-4
    length_cm = electrical["edge_length_um"].to_numpy(dtype=float) * 1e-4
    is_root = electrical["segment_id"].astype(int).to_numpy() == int(root)
    degenerate_nonroot = (~is_root) & (length_cm <= 0)
    effective_length_cm = length_cm.copy()
    effective_length_cm[degenerate_nonroot] = radius_cm[degenerate_nonroot]

    lateral_area_cm2 = 2.0 * np.pi * radius_cm * np.maximum(effective_length_cm, 0.0)
    soma_area_cm2 = 4.0 * np.pi * radius_cm * radius_cm
    membrane_area_cm2 = np.where(is_root, soma_area_cm2, lateral_area_cm2)
    g_leak_ns = membrane_area_cm2 / float(membrane_resistance_ohm_cm2) * 1e9
    if np.any(g_leak_ns <= 0):
        raise ValueError("non-positive physical leak conductance")

    g_edge_ns = np.zeros(len(electrical), dtype=float)
    nonroot = ~is_root
    g_edge_ns[nonroot] = (
        np.pi * radius_cm[nonroot] ** 2
        / (float(axial_resistivity_ohm_cm) * np.maximum(effective_length_cm[nonroot], 1e-12))
        * 1e9
    )

    total_synapse_size = (
        electrical["E_size"].to_numpy(dtype=float)
        + electrical["I_size"].to_numpy(dtype=float)
    )
    positive_synapse_size = total_synapse_size[total_synapse_size > 0]
    synapse_size_reference = (
        float(np.median(positive_synapse_size)) if len(positive_synapse_size) else 1.0
    )
    leak_reference_ns = float(np.median(g_leak_ns[nonroot])) if np.any(nonroot) else float(np.median(g_leak_ns))
    g_e_ns = (
        float(e_scale)
        * electrical["E_size"].to_numpy(dtype=float)
        / synapse_size_reference
        * leak_reference_ns
    )
    g_i_ns = (
        float(i_scale)
        * electrical["I_size"].to_numpy(dtype=float)
        / synapse_size_reference
        * leak_reference_ns
    )

    electrical["g_edge"] = g_edge_ns
    electrical["g_leak"] = g_leak_ns
    electrical["g_e"] = g_e_ns
    electrical["g_i"] = g_i_ns
    electrical["membrane_area_cm2"] = membrane_area_cm2
    electrical["degenerate_length_imputed"] = degenerate_nonroot
    electrical["axial_to_leak_ratio"] = np.divide(
        g_edge_ns,
        g_leak_ns,
        out=np.zeros_like(g_edge_ns),
        where=g_leak_ns > 0,
    )

    matrix = np.zeros((len(electrical), len(electrical)), dtype=float)
    rhs = g_e_ns * float(excitatory_reversal) + g_i_ns * float(inhibitory_reversal)
    matrix[np.diag_indices(len(electrical))] += g_leak_ns + g_e_ns + g_i_ns
    for child, parent in parents.items():
        child_index = index[int(child)]
        parent_index = index[int(parent)]
        coupling = float(g_edge_ns[child_index])
        matrix[child_index, child_index] += coupling
        matrix[parent_index, parent_index] += coupling
        matrix[child_index, parent_index] -= coupling
        matrix[parent_index, child_index] -= coupling
    return electrical, matrix, rhs, index[int(root)], parents, children


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> list[float]:
    draws = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def summarize_regime(focal: pd.DataFrame, regime: dict[str, Any], seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    primary = focal[np.isclose(focal["dose"], 1.0)].pivot_table(
        index=["root_id", "focal_segment_id"],
        columns="perturbation",
        values="localization_index",
    ).dropna().reset_index()
    primary["difference"] = primary["focal shunt"] - primary["matched additive"]
    cells = primary.groupby("root_id", as_index=False).agg(
        shunting_localization=("focal shunt", "mean"),
        additive_localization=("matched additive", "mean"),
        difference=("difference", "mean"),
        n_sites=("difference", "size"),
    )
    values = cells["difference"].to_numpy(dtype=float)
    if len(values) > 1 and np.any(values != 0):
        wilcoxon_p = float(stats.wilcoxon(values, alternative="two-sided", zero_method="wilcox").pvalue)
    else:
        wilcoxon_p = float("nan")
    summary = {
        **regime,
        "n_cells": int(len(cells)),
        "n_focal_sites": int(primary.shape[0]),
        "mean_shunting_localization": float(cells["shunting_localization"].mean()),
        "mean_additive_localization": float(cells["additive_localization"].mean()),
        "mean_shunting_minus_additive": float(values.mean()),
        "cell_bootstrap_ci95": bootstrap_interval(values, np.random.default_rng(seed)),
        "cells_positive": int((values > 0).sum()),
        "wilcoxon_two_sided_p": wilcoxon_p,
    }
    return cells, summary


def physical_ratio_rows(
    segments: pd.DataFrame,
    cohort: str,
    regime_name: str,
    ra: float,
    rm: float,
    e_scale: float,
    i_scale: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    builder = partial(
        physical_conductance_system,
        axial_resistivity_ohm_cm=ra,
        membrane_resistance_ohm_cm2=rm,
    )
    for root_id, group in segments.groupby("root_id", sort=True):
        electrical, *_ = builder(group.copy(), e_scale, i_scale, 1.0, -0.2)
        nonroot = electrical["g_edge"] > 0
        rows.append(
            {
                "cohort": cohort,
                "regime": regime_name,
                "root_id": int(root_id),
                "median_axial_to_leak_ratio": float(electrical.loc[nonroot, "axial_to_leak_ratio"].median()),
                "minimum_axial_to_leak_ratio": float(electrical.loc[nonroot, "axial_to_leak_ratio"].min()),
                "maximum_axial_to_leak_ratio": float(electrical.loc[nonroot, "axial_to_leak_ratio"].max()),
                "median_leak_ns": float(electrical["g_leak"].median()),
                "median_axial_ns": float(electrical.loc[nonroot, "g_edge"].median()),
                "n_degenerate_lengths_imputed": int(electrical["degenerate_length_imputed"].sum()),
            }
        )
    return rows


def run_cohort(
    segments: pd.DataFrame,
    cohort: str,
    regimes: list[dict[str, Any]],
    seed: int,
    max_focal_sites: int,
    minimum_sites: int,
    e_scale: float,
    i_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    focal_frames: list[pd.DataFrame] = []
    cell_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    ratios: list[dict[str, Any]] = []
    for regime_index, regime in enumerate(regimes):
        print(f"{cohort}: {regime['name']} ({regime_index + 1}/{len(regimes)})", flush=True)
        builder = partial(
            physical_conductance_system,
            axial_resistivity_ohm_cm=float(regime["axial_resistivity_ohm_cm"]),
            membrane_resistance_ohm_cm2=float(regime["membrane_resistance_ohm_cm2"]),
        )
        regime_rows: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for root_id, group in segments.groupby("root_id", sort=True):
            try:
                _, focal, _ = analyze_cell(
                    int(root_id),
                    group.copy(),
                    [1.0],
                    seed,
                    max_focal_sites,
                    minimum_sites,
                    e_scale,
                    i_scale,
                    1.0,
                    -0.2,
                    conductance_builder=builder,
                )
                regime_rows.extend(focal)
            except Exception as exc:
                errors.append({"root_id": str(root_id), "error": f"{type(exc).__name__}: {exc}"})
        frame = pd.DataFrame(regime_rows)
        if frame.empty:
            raise RuntimeError(f"no physical focal results for {cohort}/{regime['name']}: {errors}")
        frame.insert(0, "cohort", cohort)
        frame.insert(1, "regime", regime["name"])
        focal_frames.append(frame)
        cells, summary = summarize_regime(
            frame,
            {
                "cohort": cohort,
                "regime": regime["name"],
                "axial_resistivity_ohm_cm": float(regime["axial_resistivity_ohm_cm"]),
                "membrane_resistance_ohm_cm2": float(regime["membrane_resistance_ohm_cm2"]),
                "errors": errors,
            },
            seed + regime_index + (0 if cohort == "original_eight" else 10_000),
        )
        cells.insert(0, "cohort", cohort)
        cells.insert(1, "regime", regime["name"])
        cell_frames.append(cells)
        summaries.append(summary)
        ratios.extend(
            physical_ratio_rows(
                segments,
                cohort,
                regime["name"],
                float(regime["axial_resistivity_ohm_cm"]),
                float(regime["membrane_resistance_ohm_cm2"]),
                e_scale,
                i_scale,
            )
        )
    return (
        pd.concat(focal_frames, ignore_index=True),
        pd.concat(cell_frames, ignore_index=True),
        summaries,
        pd.DataFrame(ratios),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-segments", type=Path, default=DEFAULT_ORIGINAL)
    parser.add_argument("--v661-segments", type=Path, default=DEFAULT_V661)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--e-scale", type=float, default=0.35)
    parser.add_argument("--i-scale", type=float, default=0.35)
    parser.add_argument("--max-focal-sites", type=int, default=16)
    parser.add_argument("--minimum-sites", type=int, default=3)
    parser.add_argument("--skip-v661", action="store_true")
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    original_regimes = [
        {"name": "Ra100_Rm15000", "axial_resistivity_ohm_cm": 100.0, "membrane_resistance_ohm_cm2": 15_000.0},
        {"name": "Ra150_Rm300", "axial_resistivity_ohm_cm": 150.0, "membrane_resistance_ohm_cm2": 300.0},
        {"name": "Ra150_Rm1000", "axial_resistivity_ohm_cm": 150.0, "membrane_resistance_ohm_cm2": 1_000.0},
        {"name": "Ra150_Rm3000", "axial_resistivity_ohm_cm": 150.0, "membrane_resistance_ohm_cm2": 3_000.0},
        {"name": "Ra150_Rm5000", "axial_resistivity_ohm_cm": 150.0, "membrane_resistance_ohm_cm2": 5_000.0},
        {"name": "Ra150_Rm15000", "axial_resistivity_ohm_cm": 150.0, "membrane_resistance_ohm_cm2": 15_000.0},
        {"name": "Ra150_Rm30000", "axial_resistivity_ohm_cm": 150.0, "membrane_resistance_ohm_cm2": 30_000.0},
        {"name": "Ra250_Rm15000", "axial_resistivity_ohm_cm": 250.0, "membrane_resistance_ohm_cm2": 15_000.0},
    ]
    v661_names = {"Ra150_Rm300", "Ra150_Rm3000", "Ra150_Rm15000"}
    v661_regimes = [item for item in original_regimes if item["name"] in v661_names]

    original = pd.read_csv(args.original_segments)
    focal, cells, summaries, ratios = run_cohort(
        original,
        "original_eight",
        original_regimes,
        args.seed,
        args.max_focal_sites,
        args.minimum_sites,
        args.e_scale,
        args.i_scale,
    )
    if not args.skip_v661:
        v661 = pd.read_csv(args.v661_segments)
        v_focal, v_cells, v_summaries, v_ratios = run_cohort(
            v661,
            "v661_disjoint",
            v661_regimes,
            args.seed,
            args.max_focal_sites,
            args.minimum_sites,
            args.e_scale,
            args.i_scale,
        )
        focal = pd.concat([focal, v_focal], ignore_index=True)
        cells = pd.concat([cells, v_cells], ignore_index=True)
        summaries.extend(v_summaries)
        ratios = pd.concat([ratios, v_ratios], ignore_index=True)

    focal.to_csv(args.outdir / "focal_localization.csv.gz", index=False, compression="gzip")
    cells.to_csv(args.outdir / "cell_primary_contrasts.csv", index=False)
    ratios.to_csv(args.outdir / "cell_electrotonic_ratios.csv", index=False)
    payload = {
        "status": "complete",
        "analysis": "physical passive-cable sensitivity of focal exact-credit perturbations",
        "synaptic_calibration": (
            "MICrONS synapse size is scaled so the median nonzero total synapse size corresponds "
            "to e_scale or i_scale times the median physical leak conductance; it is not a measured conductance"
        ),
        "focal_dose": "one times local leak plus excitatory plus inhibitory conductance, excluding axial coupling",
        "regimes": summaries,
        "limitations": [
            "Specific membrane and axial resistances are sensitivity parameters, not fitted cell-specific measurements.",
            "The model is passive and steady-state; it does not include active channels or temporal plasticity.",
            "The v661 cohort is from the same MICrONS mouse and is not an independent-animal replication.",
            "Two zero-length non-root segments in the v661 historical reconstructions use their radius as an explicit positive cable-length fallback.",
        ],
    }
    write_json(args.outdir / "summary.json", payload)
    lines = [
        "# Physical passive-cable sensitivity",
        "",
        "Focal sites, state-matching compensatory somatic current, exact adjoint, and the first-order-current-matched additive control are identical to the normalized analysis.",
        "Axial and leak conductances are computed in nS from reconstructed radius and length plus explicit specific resistances.",
        "",
        "| Cohort | Regime | Median axial/leak | Shunt - additive | 95% CI | Positive cells |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ratio_lookup = ratios.groupby(["cohort", "regime"])["median_axial_to_leak_ratio"].median()
    for item in summaries:
        ratio = float(ratio_lookup.loc[(item["cohort"], item["regime"])])
        lines.append(
            f"| {item['cohort']} | {item['regime']} | {ratio:.2f} | "
            f"{item['mean_shunting_minus_additive']:.4f} | "
            f"[{item['cell_bootstrap_ci95'][0]:.4f}, {item['cell_bootstrap_ci95'][1]:.4f}] | "
            f"{item['cells_positive']}/{item['n_cells']} |"
        )
    (args.outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

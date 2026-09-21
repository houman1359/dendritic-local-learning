#!/usr/bin/env python3
"""Package the public-v661 MICrONS replication and its provenance.

The routing and focal scripts predate this replication and summarize slightly
different cell sets for their focal contrasts.  This script leaves those raw
outputs unchanged, constructs the intended cell-level estimands explicitly,
and writes publication-facing source tables plus a human-readable audit.

The primary routing estimand is the per-cell captured energy at eight feedback
channels, ``1 - residual**2``.  The primary focal estimand compares focal
shunting with a current-matched additive perturbation at dose 1, averaging
first across eligible focal sites within each cell.  All 45 cells with an
eligible site enter that comparison.  The within-cell depth-shuffled relation
control needs at least two eligible sites and therefore contains 40 cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


JOURNAL = Path(__file__).resolve().parents[1]
REPO = JOURNAL.parents[2]
DEFAULT_SELECTION = JOURNAL / "data" / "microns_expanded" / "prospective_selection.csv"
DEFAULT_STATIC_DATA = JOURNAL / "data" / "microns_v661_replication"
DEFAULT_ROUTING = JOURNAL / "source_data" / "microns_v661_replication" / "routing"
DEFAULT_FOCAL = JOURNAL / "source_data" / "microns_v661_replication" / "focal"
DEFAULT_TYPED_PILOT_ROUTING = (
    REPO
    / "drafts"
    / "dendritic-credit-routing"
    / "results"
    / "microns_morphology_credit_typed_only"
)
ROUTING_SCRIPT = (
    REPO
    / "drafts"
    / "dendritic-credit-routing"
    / "analysis"
    / "analyze_microns_morphology_credit.py"
)
FOCAL_SCRIPT = (
    REPO
    / "drafts"
    / "dendritic-credit-routing"
    / "analysis"
    / "run_focal_shunting_credit_perturbation.py"
)
DEFAULT_PILOT = (
    REPO
    / "drafts"
    / "dendritic-credit-routing"
    / "data"
    / "microns_morphology"
    / "cell_manifest.csv"
)
DEFAULT_CURRENT_CACHE = DEFAULT_PILOT.parent
DEFAULT_OUTDIR = JOURNAL / "source_data" / "microns_v661_replication"
DEFAULT_REPORT = JOURNAL / "analysis" / "microns_v661_replication_report.md"
DEFAULT_ROUTING_MULTISTREAM = DEFAULT_OUTDIR / "routing_capacity_20stream.csv.gz"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    """Return a portable repository-relative path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO.resolve()))
    except ValueError:
        return str(resolved)


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_draws: int = 20_000) -> list[float]:
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(int(n_draws), len(values)), replace=True).mean(axis=1)
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def paired_summary(values: pd.Series, seed: int) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    zero_tolerance = 1e-12
    clean[np.abs(clean) < zero_tolerance] = 0.0
    test = (
        stats.wilcoxon(clean, alternative="two-sided")
        if len(clean) >= 2 and not np.allclose(clean, 0.0)
        else None
    )
    return {
        "n_cells": int(len(clean)),
        "mean_difference": float(np.mean(clean)),
        "median_difference": float(np.median(clean)),
        "cell_bootstrap_ci95": bootstrap_mean_ci(clean, seed),
        "cells_positive": int(np.sum(clean > 0)),
        "cells_tied_within_tolerance": int(np.sum(clean == 0)),
        "zero_tolerance": zero_tolerance,
        "wilcoxon_two_sided_p": float(test.pvalue) if test is not None else float("nan"),
    }


def current_cache_inventory(cache_dir: Path, pilot: pd.DataFrame) -> pd.DataFrame:
    pilot_by_current = pilot.set_index("root_id").to_dict("index")
    pilot_by_source = pilot.set_index("source_root_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for skeleton in sorted(cache_dir.glob("skeleton_*.csv.gz")):
        root_id = int(skeleton.name.removeprefix("skeleton_").removesuffix(".csv.gz"))
        synapses = cache_dir / f"synapses_{root_id}.csv.gz"
        pilot_row = pilot_by_current.get(root_id) or pilot_by_source.get(root_id)
        pilot_current_root = (
            root_id
            if root_id in pilot_by_current
            else int(pilot_row["root_id"])
            if pilot_row is not None
            else pd.NA
        )
        if root_id in pilot_by_current:
            disposition = "complete original-pilot cell"
        elif root_id in pilot_by_source:
            disposition = "historical root of an original-pilot cell; skeleton only"
        elif synapses.exists():
            disposition = "complete non-pilot cached cell"
        else:
            disposition = "non-pilot functional target; skeleton only"
        rows.append(
            {
                "root_id": root_id,
                "has_skeleton": True,
                "has_synapses": synapses.exists(),
                "skeleton_bytes": skeleton.stat().st_size,
                "synapse_bytes": synapses.stat().st_size if synapses.exists() else 0,
                "skeleton_sha256": file_sha256(skeleton),
                "synapse_sha256": file_sha256(synapses) if synapses.exists() else "",
                "pilot_nucleus_id": (
                    str(int(pilot_row["nucleus_id"]))
                    if pilot_row is not None
                    else "not_applicable"
                ),
                "pilot_current_root_id": (
                    str(pilot_current_root)
                    if pilot_row is not None
                    else "not_applicable"
                ),
                "cache_disposition": disposition,
            }
        )
    return pd.DataFrame(rows)


def routing_table(curves: pd.DataFrame, metrics: pd.DataFrame, channels: int) -> pd.DataFrame:
    selected = curves[curves["channels"].eq(int(channels))].copy()
    selected["capture"] = 1.0 - selected["residual"].astype(float) ** 2
    capture = selected.pivot(index="root_id", columns="method", values="capture")
    residual = selected.pivot(index="root_id", columns="method", values="residual")
    morph = selected[selected["method"].eq("morphology-aware paths")][
        ["root_id", "wiring_nonzeros"]
    ].copy()
    out = metrics[[
        "root_id",
        "n_segments",
        "n_e_segments",
        "n_i_segments",
        "mapping_pass_fraction",
        "credit_kernel_participation_rank",
        "max_topological_depth",
    ]].merge(morph, on="root_id", how="inner", validate="one_to_one")
    out["morphology_wiring_density"] = out["wiring_nonzeros"] / (
        out["n_e_segments"] * int(channels)
    )
    rename_capture = {
        "dense PCA oracle": "capture_dense_oracle",
        "morphology-aware paths": "capture_morphology_paths",
        "random paths": "capture_random_paths",
        "shuffled ancestry": "capture_shuffled_ancestry",
        "depth-only bins": "capture_depth_bins",
    }
    rename_residual = {
        method: name.replace("capture_", "residual_")
        for method, name in rename_capture.items()
    }
    out = out.merge(capture.rename(columns=rename_capture), on="root_id", validate="one_to_one")
    out = out.merge(residual.rename(columns=rename_residual), on="root_id", validate="one_to_one")
    out["morphology_capture_fraction_of_dense"] = (
        out["capture_morphology_paths"] / out["capture_dense_oracle"]
    )
    out["morphology_minus_random_capture"] = (
        out["capture_morphology_paths"] - out["capture_random_paths"]
    )
    out["morphology_minus_depth_capture"] = (
        out["capture_morphology_paths"] - out["capture_depth_bins"]
    )
    out["morphology_minus_shuffled_capture"] = (
        out["capture_morphology_paths"] - out["capture_shuffled_ancestry"]
    )
    return out.sort_values("root_id").reset_index(drop=True)


def routing_table_multistream(
    curves: pd.DataFrame, metrics: pd.DataFrame, channels: int
) -> pd.DataFrame:
    """Average Monte Carlo streams within cell before forming routing contrasts."""
    selected = curves[curves["channels"].eq(int(channels))].copy()
    required = {
        "root_id",
        "method",
        "credit_capture",
        "residual",
        "wiring_nonzeros",
    }
    missing = required.difference(selected.columns)
    if missing:
        raise ValueError(f"multistream routing table lacks columns: {sorted(missing)}")
    selected = (
        selected.groupby(["root_id", "method"], as_index=False)
        .agg(
            capture=("credit_capture", "mean"),
            residual=("residual", "mean"),
            wiring_nonzeros=("wiring_nonzeros", "mean"),
        )
    )
    capture = selected.pivot(index="root_id", columns="method", values="capture")
    residual = selected.pivot(index="root_id", columns="method", values="residual")
    morph = selected[selected["method"].eq("morphology-aware paths")][
        ["root_id", "wiring_nonzeros"]
    ].copy()
    out = metrics[[
        "root_id",
        "n_segments",
        "n_e_segments",
        "n_i_segments",
        "mapping_pass_fraction",
        "credit_kernel_participation_rank",
        "max_topological_depth",
    ]].merge(morph, on="root_id", how="inner", validate="one_to_one")
    out["morphology_wiring_density"] = out["wiring_nonzeros"] / (
        out["n_e_segments"] * int(channels)
    )
    rename_capture = {
        "dense PCA oracle": "capture_dense_oracle",
        "morphology-aware paths": "capture_morphology_paths",
        "random paths": "capture_random_paths",
        "shuffled ancestry": "capture_shuffled_ancestry",
        "depth-only bins": "capture_depth_bins",
    }
    rename_residual = {
        method: name.replace("capture_", "residual_")
        for method, name in rename_capture.items()
    }
    out = out.merge(capture.rename(columns=rename_capture), on="root_id", validate="one_to_one")
    out = out.merge(residual.rename(columns=rename_residual), on="root_id", validate="one_to_one")
    out["morphology_capture_fraction_of_dense"] = (
        out["capture_morphology_paths"] / out["capture_dense_oracle"]
    )
    out["morphology_minus_random_capture"] = (
        out["capture_morphology_paths"] - out["capture_random_paths"]
    )
    out["morphology_minus_depth_capture"] = (
        out["capture_morphology_paths"] - out["capture_depth_bins"]
    )
    out["morphology_minus_shuffled_capture"] = (
        out["capture_morphology_paths"] - out["capture_shuffled_ancestry"]
    )
    return out.sort_values("root_id").reset_index(drop=True)


def focal_tables(
    focal: pd.DataFrame,
    validation: list[dict[str, Any]],
    primary_dose: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    checks = pd.DataFrame(
        {
            "root_id": int(row["root_id"]),
            "n_eligible_focal_sites": int(row["n_eligible_focal_sites"]),
            "n_selected_focal_sites": int(row["n_selected_focal_sites"]),
        }
        for row in validation
    )
    selected = focal[np.isclose(focal["dose"].astype(float), float(primary_dose))].copy()
    site = selected.pivot_table(
        index=["root_id", "focal_segment_id"],
        columns="perturbation",
        values="localization_index",
    ).dropna(subset=["focal shunt", "matched additive"])
    site["shunt_minus_additive"] = site["focal shunt"] - site["matched additive"]
    direct = site.groupby(level="root_id").agg(
        focal_shunt_localization=("focal shunt", "mean"),
        matched_additive_localization=("matched additive", "mean"),
        shunt_minus_additive=("shunt_minus_additive", "mean"),
        n_focal_sites=("shunt_minus_additive", "size"),
    ).reset_index()

    shunt = selected[selected["perturbation"].eq("focal shunt")].copy()
    shunt["shunt_topology_minus_depth_shuffle"] = (
        shunt["localization_index"] - shunt["depth_shuffled_localization_mean"]
    )
    # A relation template drawn from "other" focal sites is only a meaningful
    # within-cell control when the cell has at least two eligible sites.
    eligible_roots = set(checks.loc[checks["n_selected_focal_sites"] >= 2, "root_id"].astype(int))
    shuffled = (
        shunt[shunt["root_id"].isin(eligible_roots)]
        .groupby("root_id", as_index=False)
        .agg(
            focal_shunt_localization=("localization_index", "mean"),
            shunt_depth_shuffled_localization=("depth_shuffled_localization_mean", "mean"),
            shunt_topology_minus_depth_shuffle=("shunt_topology_minus_depth_shuffle", "mean"),
            n_focal_sites=("focal_segment_id", "nunique"),
        )
    )
    return checks.sort_values("root_id"), direct.sort_values("root_id"), shuffled.sort_values("root_id")


def finite_difference_summary(validation: list[dict[str, Any]]) -> dict[str, Any]:
    checks: list[dict[str, float]] = []
    clamp: list[float] = []
    eigenvalues: list[float] = []
    for cell in validation:
        clamp.append(float(cell["maximum_soma_clamp_error"]))
        eigenvalues.append(float(cell["matrix_minimum_eigenvalue"]))
        for row in cell.get("finite_difference_checks", []):
            analytic = float(row["analytic"])
            finite = float(row["finite_difference"])
            checks.append(
                {
                    "analytic": analytic,
                    "absolute_error": abs(finite - analytic),
                    "relative_error": abs(finite - analytic) / max(abs(analytic), 1e-12),
                }
            )
    frame = pd.DataFrame(checks)
    stable = frame[np.abs(frame["analytic"]) >= 1e-8]
    return {
        "n_finite_difference_checks": int(len(frame)),
        "maximum_absolute_error": float(frame["absolute_error"].max()),
        "maximum_raw_relative_error": float(frame["relative_error"].max()),
        "maximum_relative_error_for_abs_analytic_at_least_1e-8": float(
            stable["relative_error"].max()
        ),
        "n_checks_abs_analytic_at_least_1e-8": int(len(stable)),
        "maximum_soma_clamp_error": float(max(clamp)),
        "minimum_matrix_eigenvalue": float(min(eigenvalues)),
    }


def make_cohort_manifest(
    eligibility: pd.DataFrame,
    pilot: pd.DataFrame,
    fetched: pd.DataFrame,
    routing: pd.DataFrame,
    focal_checks: pd.DataFrame,
) -> pd.DataFrame:
    pilot_lookup = pilot[["nucleus_id", "root_id", "source_root_id", "pilot_order"]].rename(
        columns={
            "root_id": "original_pilot_current_root_id",
            "source_root_id": "original_pilot_source_root_id",
        }
    )
    for column in ["original_pilot_current_root_id", "original_pilot_source_root_id"]:
        pilot_lookup[column] = pilot_lookup[column].astype("Int64")
    fetch_keep = [
        "root_id",
        "nucleus_id",
        "status",
        "n_skeleton_nodes",
        "n_synapses",
        "n_direct_typed_synapses",
        "n_direct_e_synapses",
        "n_direct_i_synapses",
        "direct_type_coverage",
        "swc_url",
        "meshwork_url",
        "swc_bytes",
        "meshwork_bytes",
        "swc_sha256",
        "meshwork_sha256",
    ]
    fetch = fetched[fetch_keep].rename(
        columns={"root_id": "replication_root_id", "status": "fetch_status"}
    )
    fetch["replication_root_id"] = fetch["replication_root_id"].astype("Int64")
    route_keep = routing[[
        "root_id",
        "mapping_pass_fraction",
        "n_e_segments",
        "n_i_segments",
        "capture_dense_oracle",
        "capture_morphology_paths",
        "capture_random_paths",
        "capture_depth_bins",
        "capture_shuffled_ancestry",
        "morphology_wiring_density",
        "morphology_capture_fraction_of_dense",
    ]].rename(columns={"root_id": "replication_root_id"})
    route_keep["replication_root_id"] = route_keep["replication_root_id"].astype("Int64")
    focal_keep = focal_checks.rename(columns={"root_id": "replication_root_id"})
    focal_keep["replication_root_id"] = focal_keep["replication_root_id"].astype("Int64")
    out = eligibility.merge(pilot_lookup, on="nucleus_id", how="left", validate="one_to_one")
    out = out.merge(fetch, on="nucleus_id", how="left", validate="one_to_one")
    # The eight excluded rows all have a missing replication root.  The right
    # tables are unique, but repeated missing keys make pandas reject a strict
    # one-to-one validation on the left.
    out = out.merge(route_keep, on="replication_root_id", how="left", validate="many_to_one")
    out = out.merge(focal_keep, on="replication_root_id", how="left", validate="many_to_one")
    out["cohort_disposition"] = np.select(
        [
            out["excluded_original_pilot"].astype(bool),
            out["fetch_status"].eq("fetched") & out["capture_morphology_paths"].notna(),
            out["replication_eligible"].astype(bool),
        ],
        [
            "excluded: member of original eight-cell pilot",
            "included: public-v661 disjoint replication",
            "eligible but replication output unavailable",
        ],
        default="excluded: no v661 mapping",
    )
    out["routing_included"] = out["capture_morphology_paths"].notna()
    out["focal_shunt_additive_included"] = out["n_selected_focal_sites"].fillna(0).ge(1)
    out["focal_depth_shuffle_included"] = out["n_selected_focal_sites"].fillna(0).ge(2)
    out["focal_exclusion_reason"] = np.select(
        [
            out["excluded_original_pilot"].astype(bool),
            out["n_selected_focal_sites"].fillna(0).eq(0),
            out["n_selected_focal_sites"].fillna(0).eq(1),
        ],
        [
            "original pilot excluded from replication",
            "no site met descendant and matched-unrelated eligibility criteria",
            "eligible for shunt-additive contrast; insufficient second site for within-cell depth shuffle",
        ],
        default="included in both focal contrasts",
    )
    # Root IDs are larger than IEEE-754's exact integer range.  Columns that
    # contain missing values are emitted as strings with explicit sentinels so
    # a default CSV reader cannot silently round them through float64.
    out["replication_root_id"] = (
        out["replication_root_id"].astype("Int64").astype("string").fillna("not_in_replication")
    )
    for column in ["original_pilot_current_root_id", "original_pilot_source_root_id"]:
        out[column] = (
            out[column].astype("Int64").astype("string").fillna("not_applicable")
        )
    return out.sort_values("selection_order").reset_index(drop=True)


def markdown_report(summary: dict[str, Any]) -> str:
    routing = summary["routing_eight_channels"]
    focal = summary["focal_primary_dose_one"]
    validation = summary["numerical_validation"]
    return f"""# Public-v661 MICrONS replication audit

## Decision

The local cache did not contain enough complete, non-overlapping current-materialization reconstructions for a larger replication. It contained 11 skeletons but only eight complete skeleton-plus-synapse pairs. One extra skeleton is the historical root of a cell already represented among those eight, and two extra functional targets have no cached whole-cell synapse file.

An official public static route does support a disjoint historical replication without CAVE authentication. The frozen 55-cell V1 L2-L5 IT/ET list maps uniquely to MICrONS version 661. Excluding the original eight by stable nucleus identifier leaves 47 cells. All 47 public SWC skeletons and postsynaptic meshworks were fetched successfully and analyzed with direct presynaptic coarse E/I labels only.

This is valid as a disjoint-cell robustness analysis of structural routing and the focal in-model mechanism. It is not an independent animal or population sample: all cells come from the same MICrONS mouse, the 55-cell universe descends from a convenience structural pilot, and v661 is a historical reconstruction rather than materialization 1822.

## Cohort provenance

- Frozen universe: {summary['cohort']['n_selection_universe']} V1 excitatory L2-L5 IT/ET cells.
- Original pilot excluded by nucleus ID: {summary['cohort']['n_original_pilot_excluded']}.
- Public-v661 replication cells fetched and routed: {summary['cohort']['n_routing_cells']}.
- Incoming synapses in meshworks: {summary['cohort']['n_total_postsynaptic_synapses']:,}.
- Direct E/I calls before spatial mapping: {summary['cohort']['n_total_direct_typed_synapses']:,} ({100 * summary['cohort']['direct_type_coverage']:.2f}%).
- Direct E/I synapses passing the spatial map: {summary['cohort']['n_total_mapped_direct_synapses']:,}.
- Per cell, the direct-type set contained a median of {summary['cohort']['median_direct_typed_synapses_per_cell']:.0f} synapses (range {summary['cohort']['range_direct_typed_synapses_per_cell'][0]}-{summary['cohort']['range_direct_typed_synapses_per_cell'][1]}); every routed tree retained at least {summary['cohort']['minimum_e_segments_per_cell']} excitatory-bearing and {summary['cohort']['minimum_i_segments_per_cell']} inhibitory-bearing segments.
- Replication cell types: 34 L2IT, 2 L3IT, 10 L4IT, and 1 L5ET; the cohort is therefore layer/type imbalanced.
- Classification: matching v661 `baylor_log_reg_cell_type_coarse_v1` calls only; no spine/shaft proxy.
- Access: versioned public static URLs and SHA-256 hashes are recorded in `cohort_manifest.csv` and `replication_summary.json`. The parent MICrONS dataset is described by DOI 10.1038/s41586-025-08790-w. DANDI:000402 is functional imaging data and is not an accession for the structural files used here.

The original eight were a hand-authored, layer-diverse pilot selected from an existing anatomy/functional-coregistration cohort. The list was present before its recorded routing outcomes, but it was not an externally preregistered or population-random sample. The focal endpoint was specified after selection of the structural pilot and before the focal run.

The inherited 55-cell universe also is not population-wide. It consists of the V1 excitatory cells with stable nuclei in a 64-target export. Those 64 targets were the most represented targets in the first 50,000 rows returned from `vortex_compartment_targets` at materialization 1718. Accordingly, the journal should call the 47 cells a frozen, disjoint replication cohort, not a representative MICrONS sample.

## Structural routing replication

At eight feedback channels, morphology paths captured {100 * routing['mean_capture']['morphology_paths']:.1f}% of modeled field energy with {100 * routing['mean_morphology_wiring_density']:.1f}% of dense feedback wiring. The dense PCA oracle captured {100 * routing['mean_capture']['dense_oracle']:.1f}%; the cellwise morphology-to-oracle capture ratio was {100 * routing['mean_morphology_capture_fraction_of_dense']:.1f}% on average.

The directly typed reanalysis of the original eight cells captured {100 * summary['direct_typed_discovery_comparison']['mean_capture']['morphology_paths']:.1f}% with morphology paths, compared with {100 * summary['direct_typed_discovery_comparison']['mean_capture']['dense_oracle']:.1f}% for its dense oracle. Thus the disjoint v661 cohort reproduces the ordering under the same direct-type-first classification, although its effect size should not be pooled with the current-materialization cohort.

Morphology capture exceeded random paths by {100 * routing['morphology_minus_random']['mean_difference']:.1f} percentage points (95% cell-bootstrap CI {100 * routing['morphology_minus_random']['cell_bootstrap_ci95'][0]:.1f} to {100 * routing['morphology_minus_random']['cell_bootstrap_ci95'][1]:.1f}; {routing['morphology_minus_random']['cells_positive']}/{routing['morphology_minus_random']['n_cells']} cells), depth bins by {100 * routing['morphology_minus_depth']['mean_difference']:.1f} points (CI {100 * routing['morphology_minus_depth']['cell_bootstrap_ci95'][0]:.1f} to {100 * routing['morphology_minus_depth']['cell_bootstrap_ci95'][1]:.1f}; {routing['morphology_minus_depth']['cells_positive']}/{routing['morphology_minus_depth']['n_cells']}), and ancestry shuffles by {100 * routing['morphology_minus_shuffled']['mean_difference']:.1f} points (CI {100 * routing['morphology_minus_shuffled']['cell_bootstrap_ci95'][0]:.1f} to {100 * routing['morphology_minus_shuffled']['cell_bootstrap_ci95'][1]:.1f}; {routing['morphology_minus_shuffled']['cells_positive']}/{routing['morphology_minus_shuffled']['n_cells']}).

The earlier exploratory association between maximum depth and credit-kernel participation rank does not replicate: Spearman rho = {routing['depth_rank_spearman_r']:.3f}, p = {routing['depth_rank_spearman_p']:.3f}, n = {routing['n_cells']}. It should not be promoted as a journal-level result.

## Focal perturbation replication

Forty-five of 47 cells had at least one eligible focal site, giving {focal['shunt_minus_additive']['n_focal_sites']} sites. At dose 1, shunting localization exceeded the current-matched additive perturbation by {focal['shunt_minus_additive']['mean_difference']:.3f} on average across cells (95% cell-bootstrap CI {focal['shunt_minus_additive']['cell_bootstrap_ci95'][0]:.3f} to {focal['shunt_minus_additive']['cell_bootstrap_ci95'][1]:.3f}; {focal['shunt_minus_additive']['cells_positive']}/{focal['shunt_minus_additive']['n_cells']} cells; two-sided Wilcoxon p = {focal['shunt_minus_additive']['wilcoxon_two_sided_p']:.3g}). Mean localization was {focal['mean_shunt_localization']:.3f} for shunting and {focal['mean_additive_localization']:.3f} for the additive control.

The depth-shuffled relation control is defined for 40 cells and {focal['topology_minus_depth_shuffle']['n_focal_sites']} focal sites. True descendant relations exceeded shuffled relations by {focal['topology_minus_depth_shuffle']['mean_difference']:.3f} (95% CI {focal['topology_minus_depth_shuffle']['cell_bootstrap_ci95'][0]:.3f} to {focal['topology_minus_depth_shuffle']['cell_bootstrap_ci95'][1]:.3f}; {focal['topology_minus_depth_shuffle']['cells_positive']}/{focal['topology_minus_depth_shuffle']['n_cells']} cells, with one numerical tie; p = {focal['topology_minus_depth_shuffle']['wilcoxon_two_sided_p']:.3g}). Five additional one-site cells remain valid for the shunt-additive comparison but cannot supply an independent within-cell shuffled template. Two cells had no eligible focal site; all exclusions are explicit in `cohort_manifest.csv`.

The original focal summary silently reduced the shunt-additive comparison to the 40-cell subset used by the shuffle control. The packaged primary table corrects this: shunt versus additive uses all 45 eligible cells; only the topology-shuffle comparison uses 40.

## Numerical checks

Across {validation['n_finite_difference_checks']} finite-difference checks, the maximum absolute gradient error was {validation['maximum_absolute_error']:.3g}. The maximum relative error among checks with absolute analytic gradient at least 1e-8 was {validation['maximum_relative_error_for_abs_analytic_at_least_1e-8']:.3g}. The larger raw relative maximum ({validation['maximum_raw_relative_error']:.3g}) is caused by division by an analytic gradient near numerical zero and should not be quoted without that qualification. The maximum soma-clamp error was {validation['maximum_soma_clamp_error']:.3g}, and the minimum conductance-matrix eigenvalue was {validation['minimum_matrix_eigenvalue']:.3f}.

## Other local MICrONS caches examined

- `drafts/dendritic-credit-routing/data/microns_morphology`: 11 skeleton files, eight complete skeleton-plus-synapse pairs, and one coarse cell-type table. The exact cell-level disposition is in `local_current_cache_inventory.csv`.
- `drafts/dendritic-pop-draft/results/microns_cave_export_scaled` and its copied `dendritic-credit-routing/imported/population` version: 9,486 compartment-tagged rows across 64 targets. These provide soma/shaft/spine pools, not dendritic parent-child connectivity, so they cannot support ancestry or descendant perturbation tests. The key tables in the two locations are byte-identical.
- `drafts/dendritic-pop-draft/results/microns_hf_dataset/microns.h5`: a 20.6-GB functional stimulus/session cache with top-level groups `brain_areas`, `sessions`, `types`, and `videos`; it does not contain skeleton topology or whole-cell synapse locations.
- Existing routing result directories are alternative analyses of the same original eight cells, not additional biological observations.

No other complete local skeleton-plus-synapse cohort was found in the drafts tree.

## Recommended journal use

1. Present the current-materialization eight-cell cohort as the discovery pilot and this historical v661 cohort as a frozen, disjoint-cell replication.
2. Report routing at eight channels as captured energy, with all three matched controls and wiring density. Do not pool the two cohorts as 55 statistically independent observations.
3. Use 45 cells for the shunt-additive focal contrast and 40 for the depth-shuffle control.
4. State the 4.71% direct-type coverage and the historical-release/convenience-cohort limitations next to the result, not only in a general limitations section.
5. Do not retain the exploratory depth-rank claim, which was null in the larger cohort.
6. Describe the focal experiment as a mechanistic passive-network intervention on measured anatomy, not an in vivo perturbation or evidence of biological learning.
7. Before making the replication central, archive the normalized source tables or their exact static URLs and hashes with the submission. The source tables here already contain those identifiers.

## Files

- `cohort_manifest.csv`: all 55 frozen candidates, original-pilot exclusions, v661 roots, file hashes, routing inclusion, and focal eligibility.
- `local_current_cache_inventory.csv`: all locally cached current-pipeline skeleton roots and whether a matching synapse file exists.
- `cell_level_primary.csv`: the 47 replication cells and their eight-channel routing outcomes, plus focal contrasts when available.
- `replication_summary.json`: machine-readable estimands, uncertainty intervals, validation, sources, and limitations.
- Raw routing and focal outputs remain in their respective subdirectories.
"""


def build(args: argparse.Namespace) -> dict[str, Any]:
    args.outdir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    selection = pd.read_csv(args.selection)
    eligibility = pd.read_csv(args.static_data / "eligibility_frozen_before_outcomes.csv")
    fetched = pd.read_csv(args.static_data / "cell_manifest.csv")
    fetch_summary = read_json(args.static_data / "fetch_summary.json")
    pilot = pd.read_csv(args.pilot_manifest)
    routing_metrics = pd.read_csv(args.routing / "cell_metrics.csv")
    routing_curves = pd.read_csv(args.routing_multistream)
    routing_summary_raw = read_json(args.routing / "summary.json")
    typed_pilot_metrics = pd.read_csv(args.typed_pilot_routing / "cell_metrics.csv")
    typed_pilot_curves = pd.read_csv(args.typed_pilot_routing / "feedback_compression_curves.csv")
    focal = pd.read_csv(args.focal / "focal_localization.csv")
    focal_summary_raw = read_json(args.focal / "summary.json")

    if selection["nucleus_id"].duplicated().any():
        raise ValueError("frozen selection contains duplicate nucleus IDs")
    if pilot["nucleus_id"].duplicated().any():
        raise ValueError("original pilot contains duplicate nucleus IDs")

    routing = routing_table_multistream(routing_curves, routing_metrics, args.channels)
    typed_pilot_routing = routing_table(
        typed_pilot_curves, typed_pilot_metrics, args.channels
    )
    focal_checks, focal_direct, focal_shuffled = focal_tables(
        focal, focal_summary_raw["validation"]["cell_checks"], args.primary_dose
    )
    cohort = make_cohort_manifest(eligibility, pilot, fetched, routing, focal_checks)
    cache_inventory = current_cache_inventory(DEFAULT_CURRENT_CACHE, pilot)

    pilot_nuclei = set(pilot["nucleus_id"].astype(int))
    replication_nuclei = set(fetched["nucleus_id"].astype(int))
    if pilot_nuclei & replication_nuclei:
        raise ValueError("replication cohort overlaps original pilot by stable nucleus ID")
    if set(routing["root_id"].astype(int)) != set(fetched["root_id"].astype(int)):
        raise ValueError("routing output does not contain exactly the fetched replication roots")

    cell_level = routing.merge(focal_direct, on="root_id", how="left", validate="one_to_one")
    cell_level = cell_level.merge(
        focal_shuffled[
            ["root_id", "shunt_depth_shuffled_localization", "shunt_topology_minus_depth_shuffle"]
        ],
        on="root_id",
        how="left",
        validate="one_to_one",
    )
    cell_level = cell_level.merge(
        fetched[["root_id", "nucleus_id", "source_root_id", "cell_type"]],
        on="root_id",
        validate="one_to_one",
    )

    route_capture_means = {
        "dense_oracle": float(routing["capture_dense_oracle"].mean()),
        "morphology_paths": float(routing["capture_morphology_paths"].mean()),
        "random_paths": float(routing["capture_random_paths"].mean()),
        "depth_bins": float(routing["capture_depth_bins"].mean()),
        "shuffled_ancestry": float(routing["capture_shuffled_ancestry"].mean()),
    }
    direct_stats = paired_summary(focal_direct["shunt_minus_additive"], args.seed + 11)
    direct_stats["n_focal_sites"] = int(focal_direct["n_focal_sites"].sum())
    shuffle_stats = paired_summary(
        focal_shuffled["shunt_topology_minus_depth_shuffle"], args.seed + 12
    )
    shuffle_stats["n_focal_sites"] = int(focal_shuffled["n_focal_sites"].sum())
    summary: dict[str, Any] = {
        "status": "complete",
        "analysis_version": "public-v661 disjoint-cell replication",
        "estimand_definition": {
            "routing": (
                f"per-cell captured energy 1-residual^2 at {args.channels} channels; "
                "cell is the unit of inference"
            ),
            "focal": (
                f"dose-{args.primary_dose:g} localization averaged across sites within cell; "
                "cell is the unit of inference"
            ),
        },
        "cohort": {
            "n_selection_universe": int(len(selection)),
            "n_original_pilot_excluded": int(eligibility["excluded_original_pilot"].sum()),
            "n_routing_cells": int(len(routing)),
            "n_total_postsynaptic_synapses": int(fetch_summary["n_total_postsynaptic_synapses"]),
            "n_total_direct_typed_synapses": int(fetch_summary["n_total_direct_typed_synapses"]),
            "n_total_mapped_direct_synapses": int(routing_summary_raw["n_total_mapped_synapses"]),
            "direct_type_coverage": float(fetch_summary["direct_type_coverage"]),
            "median_direct_typed_synapses_per_cell": float(
                fetched["n_direct_typed_synapses"].median()
            ),
            "range_direct_typed_synapses_per_cell": [
                int(fetched["n_direct_typed_synapses"].min()),
                int(fetched["n_direct_typed_synapses"].max()),
            ],
            "minimum_e_segments_per_cell": int(routing["n_e_segments"].min()),
            "minimum_i_segments_per_cell": int(routing["n_i_segments"].min()),
            "same_mouse_as_discovery": True,
            "disjoint_from_original_by_stable_nucleus_id": True,
            "population_random_sample": False,
            "historical_static_release": "MICrONS minnie65 v661",
            "replication_cell_type_counts": {
                str(key): int(value)
                for key, value in fetched["cell_type"].value_counts().sort_index().items()
            },
        },
        "routing_eight_channels": {
            "n_cells": int(len(routing)),
            "channels": int(args.channels),
            "mean_capture": route_capture_means,
            "mean_morphology_wiring_density": float(routing["morphology_wiring_density"].mean()),
            "mean_morphology_capture_fraction_of_dense": float(
                routing["morphology_capture_fraction_of_dense"].mean()
            ),
            "median_morphology_capture_fraction_of_dense": float(
                routing["morphology_capture_fraction_of_dense"].median()
            ),
            "morphology_minus_random": paired_summary(
                routing["morphology_minus_random_capture"], args.seed + 1
            ),
            "morphology_minus_depth": paired_summary(
                routing["morphology_minus_depth_capture"], args.seed + 2
            ),
            "morphology_minus_shuffled": paired_summary(
                routing["morphology_minus_shuffled_capture"], args.seed + 3
            ),
            "depth_rank_spearman_r": float(
                routing_summary_raw["depth_vs_credit_kernel_rank"]["spearman_r"]
            ),
            "depth_rank_spearman_p": float(
                routing_summary_raw["depth_vs_credit_kernel_rank"]["spearman_p"]
            ),
        },
        "direct_typed_discovery_comparison": {
            "n_cells": int(len(typed_pilot_routing)),
            "mean_capture": {
                "dense_oracle": float(typed_pilot_routing["capture_dense_oracle"].mean()),
                "morphology_paths": float(
                    typed_pilot_routing["capture_morphology_paths"].mean()
                ),
                "random_paths": float(typed_pilot_routing["capture_random_paths"].mean()),
                "depth_bins": float(typed_pilot_routing["capture_depth_bins"].mean()),
                "shuffled_ancestry": float(
                    typed_pilot_routing["capture_shuffled_ancestry"].mean()
                ),
            },
            "mean_morphology_wiring_density": float(
                typed_pilot_routing["morphology_wiring_density"].mean()
            ),
            "scope": (
                "same direct-presynaptic-type classification principle; different materialization "
                "and reconstruction cohort, so effect sizes are not pooled"
            ),
        },
        "focal_primary_dose_one": {
            "dose": float(args.primary_dose),
            "mean_shunt_localization": float(focal_direct["focal_shunt_localization"].mean()),
            "mean_additive_localization": float(
                focal_direct["matched_additive_localization"].mean()
            ),
            "shunt_minus_additive": direct_stats,
            "mean_depth_shuffled_localization": float(
                focal_shuffled["shunt_depth_shuffled_localization"].mean()
            ),
            "topology_minus_depth_shuffle": shuffle_stats,
            "n_cells_without_eligible_focal_site": int(
                focal_checks["n_selected_focal_sites"].eq(0).sum()
            ),
            "n_one_site_cells_excluded_only_from_shuffle_control": int(
                focal_checks["n_selected_focal_sites"].eq(1).sum()
            ),
        },
        "numerical_validation": finite_difference_summary(
            focal_summary_raw["validation"]["cell_checks"]
        ),
        "source_provenance": {
            "primary_dataset_doi": "10.1038/s41586-025-08790-w",
            "official_static_repository_documentation": (
                "https://tutorial.microns-explorer.org/static-repositories.html"
            ),
            "official_v661_release_manifest": (
                "https://tutorial.microns-explorer.org/release_manifests/version-661.html"
            ),
            "static_release": fetch_summary["release"],
            "cave_authentication_used": False,
            "table_sources": fetch_summary["table_sources"],
            "input_files": {
                "frozen_selection": {
                    "path": repository_path(args.selection),
                    "sha256": file_sha256(args.selection),
                },
                "original_pilot_manifest": {
                    "path": repository_path(args.pilot_manifest),
                    "sha256": file_sha256(args.pilot_manifest),
                },
                "eligibility": {
                    "path": repository_path(
                        args.static_data / "eligibility_frozen_before_outcomes.csv"
                    ),
                    "sha256": file_sha256(
                        args.static_data / "eligibility_frozen_before_outcomes.csv"
                    ),
                },
                "fetch_manifest": {
                    "path": repository_path(args.static_data / "cell_manifest.csv"),
                    "sha256": file_sha256(args.static_data / "cell_manifest.csv"),
                },
                "routing_curves": {
                    "path": repository_path(args.routing_multistream),
                    "sha256": file_sha256(args.routing_multistream),
                },
                "focal_localization": {
                    "path": repository_path(args.focal / "focal_localization.csv"),
                    "sha256": file_sha256(args.focal / "focal_localization.csv"),
                },
                "direct_typed_pilot_curves": {
                    "path": repository_path(
                        args.typed_pilot_routing / "feedback_compression_curves.csv"
                    ),
                    "sha256": file_sha256(
                        args.typed_pilot_routing / "feedback_compression_curves.csv"
                    ),
                },
            },
            "analysis_scripts": {
                "static_fetch_and_normalization": {
                    "path": repository_path(
                        JOURNAL / "scripts" / "build_static_microns_replication.py"
                    ),
                    "sha256": file_sha256(
                        JOURNAL / "scripts" / "build_static_microns_replication.py"
                    ),
                },
                "routing": {
                    "path": repository_path(ROUTING_SCRIPT),
                    "sha256": file_sha256(ROUTING_SCRIPT),
                },
                "focal_perturbation": {
                    "path": repository_path(FOCAL_SCRIPT),
                    "sha256": file_sha256(FOCAL_SCRIPT),
                },
                "publication_summary": {
                    "path": repository_path(Path(__file__).resolve()),
                    "sha256": file_sha256(Path(__file__).resolve()),
                },
            },
        },
        "limitations": [
            "The 47 replication cells are disjoint from the original eight by stable nucleus ID but come from the same mouse.",
            "The frozen selection universe descends from a 64-target convenience export and is not a population-random sample.",
            "The replication cohort is imbalanced by layer/type (34 L2IT, 2 L3IT, 10 L4IT, and 1 L5ET).",
            "Version 661 is a historical static reconstruction, not materialization 1822 used for the original eight.",
            "Direct v661 presynaptic E/I calls cover 4.71% of incoming synapses; no target-structure proxy is used.",
            "Routing is a modeled field analysis and focal shunting is a passive-network intervention, not an in vivo learning test.",
            "The larger cohort does not replicate the exploratory depth-versus-rank association.",
        ],
    }

    cohort.to_csv(args.outdir / "cohort_manifest.csv", index=False)
    cache_inventory.to_csv(args.outdir / "local_current_cache_inventory.csv", index=False)
    cell_level.to_csv(args.outdir / "cell_level_primary.csv", index=False)
    write_json(args.outdir / "replication_summary.json", summary)
    args.report.write_text(markdown_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--static-data", type=Path, default=DEFAULT_STATIC_DATA)
    parser.add_argument("--pilot-manifest", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--routing", type=Path, default=DEFAULT_ROUTING)
    parser.add_argument(
        "--routing-multistream", type=Path, default=DEFAULT_ROUTING_MULTISTREAM
    )
    parser.add_argument("--focal", type=Path, default=DEFAULT_FOCAL)
    parser.add_argument(
        "--typed-pilot-routing", type=Path, default=DEFAULT_TYPED_PILOT_ROUTING
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--primary-dose", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

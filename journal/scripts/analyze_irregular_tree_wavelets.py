#!/usr/bin/env python3
"""Quantify multiscale route-field energy on irregular MICrONS trees.

The analysis constructs a weighted unbalanced tree-Haar basis directly from
each compressed reconstruction.  It then resolves the non-scalar energy of the
conductance-weighted ancestry dictionary across coarse, intermediate and fine
anatomical scales.  The result is a structural capacity diagnostic, not an
in-vivo measurement of credit or learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


JOURNAL = Path(__file__).resolve().parents[1]
DEFAULT_PRIMARY = JOURNAL / "source_data" / "microns_v661_replication" / "routing" / "segment_metrics.csv"
DEFAULT_SECONDARY = JOURNAL / "source_data" / "figure3" / "segment_metrics.csv"
DEFAULT_OUTDIR = JOURNAL / "source_data" / "irregular_tree_wavelets"

SCALES = ("coarse", "intermediate", "fine")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_rng(seed: int, root_id: int, stream: int = 0) -> np.random.Generator:
    root = int(root_id)
    entropy = [
        int(seed) & 0xFFFFFFFF,
        root & 0xFFFFFFFF,
        (root >> 32) & 0xFFFFFFFF,
        int(stream) & 0xFFFFFFFF,
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def parent_map(segments: pd.DataFrame) -> dict[int, int]:
    roots = segments.loc[segments["parent_segment_id"] < 0, "segment_id"]
    if len(roots) != 1:
        raise ValueError(f"expected one compressed-tree root, found {len(roots)}")
    return {
        int(segment): int(parent)
        for segment, parent in segments[["segment_id", "parent_segment_id"]].itertuples(index=False)
        if int(parent) >= 0
    }


def descendants_by_segment(
    segments: pd.DataFrame,
    excitatory_ids: list[int],
) -> dict[int, frozenset[int]]:
    """Return the excitatory coordinates descended from every tree segment."""

    parents = parent_map(segments)
    result: dict[int, set[int]] = {int(value): set() for value in segments["segment_id"]}
    for coordinate, segment in enumerate(excitatory_ids):
        cursor = int(segment)
        while True:
            result[cursor].add(int(coordinate))
            if cursor not in parents:
                break
            cursor = parents[cursor]
    return {segment: frozenset(values) for segment, values in result.items()}


def scale_label(support_fraction: float) -> str:
    value = float(support_fraction)
    if value > 0.25:
        return "coarse"
    if value > 0.0625:
        return "intermediate"
    return "fine"


@dataclass(frozen=True)
class WaveletMetadata:
    mode: int
    scale: str
    support_fraction: float
    support_count: int
    split_balance: float
    split_segment_id: int


def irregular_tree_haar(
    segments: pd.DataFrame,
    excitatory_ids: list[int],
    weights: np.ndarray,
) -> tuple[np.ndarray, list[WaveletMetadata]]:
    """Construct a deterministic weighted unbalanced tree-Haar basis.

    Column zero is the weighted constant.  Every subsequent column contrasts
    the two children of a recursively nested anatomical partition.  Candidate
    partitions are descendant sets of compressed-tree segments, and the split
    closest to half of the current region's weight is chosen with stable ties.
    """

    mass = np.asarray(weights, dtype=float)
    if mass.ndim != 1 or len(mass) != len(excitatory_ids):
        raise ValueError("weights must match excitatory coordinates")
    if np.any(mass <= 0) or not np.all(np.isfinite(mass)):
        raise ValueError("all excitatory weights must be finite and positive")
    n_coordinates = len(excitatory_ids)
    if n_coordinates < 2:
        raise ValueError("at least two excitatory coordinates are required")

    descendants = descendants_by_segment(segments, excitatory_ids)
    total = float(mass.sum())
    columns = [np.sqrt(mass / total)]
    metadata: list[WaveletMetadata] = []

    def recurse(region: frozenset[int]) -> None:
        if len(region) <= 1:
            return
        indices = np.asarray(sorted(region), dtype=int)
        region_mass = float(mass[indices].sum())
        candidates: list[tuple[float, float, int, frozenset[int]]] = []
        for segment_id, descended in descendants.items():
            side = region.intersection(descended)
            if not side or len(side) == len(region):
                continue
            side_indices = np.asarray(sorted(side), dtype=int)
            side_mass = float(mass[side_indices].sum())
            imbalance = abs(side_mass / region_mass - 0.5)
            # Prefer balanced partitions, then the smaller side and stable ID.
            candidates.append((imbalance, min(side_mass, region_mass - side_mass), int(segment_id), frozenset(side)))
        if not candidates:
            # This should be unreachable for a valid rooted tree, but provides
            # a deterministic numerical fallback for malformed compression.
            ordered = sorted(region)
            side = frozenset(ordered[: len(ordered) // 2])
            split_segment = -1
        else:
            _, _, split_segment, side = min(candidates, key=lambda item: (item[0], -item[1], item[2]))
        other = frozenset(region.difference(side))
        a = np.asarray(sorted(side), dtype=int)
        b = np.asarray(sorted(other), dtype=int)
        weight_a = float(mass[a].sum())
        weight_b = float(mass[b].sum())
        vector = np.zeros(n_coordinates, dtype=float)
        vector[a] = np.sqrt(mass[a]) * math.sqrt(weight_b / (weight_a * region_mass))
        vector[b] = -np.sqrt(mass[b]) * math.sqrt(weight_a / (weight_b * region_mass))
        columns.append(vector)
        metadata.append(
            WaveletMetadata(
                mode=len(columns) - 1,
                scale=scale_label(region_mass / total),
                support_fraction=region_mass / total,
                support_count=len(region),
                split_balance=min(weight_a, weight_b) / region_mass,
                split_segment_id=int(split_segment),
            )
        )
        recurse(side)
        recurse(other)

    recurse(frozenset(range(n_coordinates)))
    basis = np.column_stack(columns)
    if basis.shape != (n_coordinates, n_coordinates):
        raise RuntimeError(f"incomplete Haar basis: {basis.shape}, expected {(n_coordinates, n_coordinates)}")
    return basis, metadata


def ancestry_matrix(
    row_segments: list[int],
    column_segments: list[int],
    parents: dict[int, int],
) -> np.ndarray:
    lookup = {int(segment): index for index, segment in enumerate(column_segments)}
    matrix = np.zeros((len(row_segments), len(column_segments)), dtype=float)
    for row, segment in enumerate(row_segments):
        cursor = int(segment)
        while True:
            if cursor in lookup:
                matrix[row, lookup[cursor]] = 1.0
            if cursor not in parents:
                break
            cursor = parents[cursor]
    return matrix


def route_dictionary(segments: pd.DataFrame) -> tuple[list[int], np.ndarray, np.ndarray]:
    indexed = segments.set_index("segment_id")
    excitatory = [int(value) for value in segments.loc[segments["E_size"] > 0, "segment_id"]]
    inhibitory = [int(value) for value in segments.loc[segments["I_size"] > 0, "segment_id"]]
    if len(excitatory) < 2 or not inhibitory:
        raise ValueError("cell lacks enough excitatory or inhibitory route segments")
    weights = indexed.loc[excitatory, "E_size"].to_numpy(dtype=float)
    ancestry = ancestry_matrix(excitatory, inhibitory, parent_map(segments))
    beta = (
        indexed.loc[inhibitory, "g_i"].to_numpy(dtype=float)
        / indexed.loc[inhibitory, "g_total"].to_numpy(dtype=float)
    )
    return excitatory, weights, ancestry * beta[None, :]


def centered_weighted_dictionary(raw: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    normalized_weight = np.asarray(weights, dtype=float) / float(np.sum(weights))
    constant = np.sqrt(normalized_weight)
    weighted = np.sqrt(normalized_weight)[:, None] * np.asarray(raw, dtype=float)
    full_energy = float(np.sum(weighted * weighted))
    centered = weighted - constant[:, None] * (constant @ weighted)[None, :]
    centered_energy = float(np.sum(centered * centered))
    removed = 1.0 - centered_energy / full_energy if full_energy > 0 else float("nan")
    return centered, removed


def scale_energy(
    centered: np.ndarray,
    basis: np.ndarray,
    metadata: list[WaveletMetadata],
) -> tuple[pd.DataFrame, float]:
    wavelets = basis[:, 1:]
    coefficients = wavelets.T @ centered
    powers = np.sum(coefficients * coefficients, axis=1)
    total = float(np.sum(centered * centered))
    decomposition_error = abs(float(powers.sum()) - total) / max(total, 1e-15)
    rows = []
    for meta, power in zip(metadata, powers, strict=True):
        rows.append(
            {
                "mode": meta.mode,
                "scale": meta.scale,
                "support_fraction": meta.support_fraction,
                "support_count": meta.support_count,
                "split_balance": meta.split_balance,
                "split_segment_id": meta.split_segment_id,
                "power": float(power),
                "energy_fraction": float(power / total) if total > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows), decomposition_error


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> list[float]:
    data = np.asarray(values, dtype=float)
    draws = np.empty(int(n_boot), dtype=float)
    for index in range(int(n_boot)):
        draws[index] = float(np.mean(rng.choice(data, size=len(data), replace=True)))
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def analyze_cell(
    segments: pd.DataFrame,
    cohort: str,
    seed: int,
    shuffles: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root_id = int(segments["root_id"].iloc[0])
    excitatory, weights, raw = route_dictionary(segments)
    basis, metadata = irregular_tree_haar(segments, excitatory, weights)
    centered, scalar_removed = centered_weighted_dictionary(raw, weights)
    modes, decomposition_error = scale_energy(centered, basis, metadata)
    modes.insert(0, "root_id", root_id)
    modes.insert(0, "cohort", cohort)

    rng = stable_rng(seed, root_id)
    null_scale = {scale: [] for scale in SCALES}
    for _ in range(int(shuffles)):
        shuffled = np.empty_like(raw)
        for column in range(raw.shape[1]):
            shuffled[:, column] = rng.permutation(raw[:, column])
        shuffled_centered, _ = centered_weighted_dictionary(shuffled, weights)
        shuffled_modes, _ = scale_energy(shuffled_centered, basis, metadata)
        grouped = shuffled_modes.groupby("scale")["energy_fraction"].sum()
        for scale in SCALES:
            null_scale[scale].append(float(grouped.get(scale, 0.0)))

    scale_rows: list[dict[str, Any]] = []
    grouped_actual = modes.groupby("scale")["energy_fraction"].sum()
    grouped_counts = modes.groupby("scale").size()
    n_wavelets = len(excitatory) - 1
    for scale in SCALES:
        actual = float(grouped_actual.get(scale, 0.0))
        count = int(grouped_counts.get(scale, 0))
        isotropic = float(count / n_wavelets)
        null_values = np.asarray(null_scale[scale], dtype=float)
        null_mean = float(np.mean(null_values))
        scale_rows.append(
            {
                "cohort": cohort,
                "root_id": root_id,
                "scale": scale,
                "n_modes": count,
                "actual_energy_fraction": actual,
                "isotropic_energy_fraction": isotropic,
                "isotropic_enrichment": actual / isotropic if isotropic > 0 else float("nan"),
                "shuffled_energy_fraction_mean": null_mean,
                "shuffled_energy_fraction_sd": float(np.std(null_values, ddof=1)),
                "actual_minus_shuffled": actual - null_mean,
                "actual_over_shuffled": actual / null_mean if null_mean > 0 else float("nan"),
            }
        )

    orthogonality_error = float(np.max(np.abs(basis.T @ basis - np.eye(len(excitatory)))))
    audit = {
        "cohort": cohort,
        "root_id": root_id,
        "n_excitatory_coordinates": len(excitatory),
        "n_inhibitory_routes": int(raw.shape[1]),
        "basis_rank": int(np.linalg.matrix_rank(basis)),
        "orthonormality_error": orthogonality_error,
        "energy_decomposition_relative_error": decomposition_error,
        "scalar_energy_fraction_removed": scalar_removed,
        "scale_mode_count": int(sum(row["n_modes"] for row in scale_rows)),
    }
    return modes, pd.DataFrame(scale_rows), audit


def cohort_statistics(
    cell_scales: pd.DataFrame,
    seed: int,
    bootstrap: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + 7_100_019)
    rows: list[dict[str, Any]] = []
    tests: dict[str, Any] = {}
    for (cohort, scale), group in cell_scales.groupby(["cohort", "scale"], sort=True):
        row: dict[str, Any] = {
            "cohort": str(cohort),
            "scale": str(scale),
            "n_cells": int(len(group)),
            "mean_actual_energy_fraction": float(group["actual_energy_fraction"].mean()),
            "mean_isotropic_energy_fraction": float(group["isotropic_energy_fraction"].mean()),
            "mean_isotropic_enrichment": float(group["isotropic_enrichment"].mean()),
            "mean_shuffled_energy_fraction": float(group["shuffled_energy_fraction_mean"].mean()),
            "mean_actual_minus_shuffled": float(group["actual_minus_shuffled"].mean()),
        }
        for column in ("actual_energy_fraction", "isotropic_enrichment", "actual_minus_shuffled"):
            low, high = bootstrap_interval(group[column].to_numpy(dtype=float), rng, bootstrap)
            row[f"{column}_ci_low"] = low
            row[f"{column}_ci_high"] = high
        rows.append(row)

        key = f"{cohort}:{scale}"
        delta = group["actual_minus_shuffled"].to_numpy(dtype=float)
        enrichment_delta = group["isotropic_enrichment"].to_numpy(dtype=float) - 1.0
        tests[key] = {
            "actual_minus_shuffled": {
                "mean": float(np.mean(delta)),
                "cells_positive": int(np.sum(delta > 0)),
                "n_cells": int(len(delta)),
                "two_sided_wilcoxon_p": float(stats.wilcoxon(delta).pvalue),
            },
            "isotropic_enrichment_minus_one": {
                "mean": float(np.mean(enrichment_delta)),
                "cells_positive": int(np.sum(enrichment_delta > 0)),
                "n_cells": int(len(enrichment_delta)),
                "two_sided_wilcoxon_p": float(stats.wilcoxon(enrichment_delta).pvalue),
            },
        }
    return pd.DataFrame(rows), tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, default=DEFAULT_PRIMARY)
    parser.add_argument("--secondary", type=Path, default=DEFAULT_SECONDARY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--shuffles", type=int, default=256)
    parser.add_argument("--bootstrap", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    mode_frames: list[pd.DataFrame] = []
    scale_frames: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    for cohort, path in (("public_v661_47", args.primary), ("original_8", args.secondary)):
        frame = pd.read_csv(path)
        for _, group in frame.groupby("root_id", sort=True):
            modes, scales, audit = analyze_cell(group.copy(), cohort, args.seed, args.shuffles)
            mode_frames.append(modes)
            scale_frames.append(scales)
            audits.append(audit)

    modes = pd.concat(mode_frames, ignore_index=True)
    cell_scales = pd.concat(scale_frames, ignore_index=True)
    audit = pd.DataFrame(audits).sort_values(["cohort", "root_id"]).reset_index(drop=True)
    cohort_scales, tests = cohort_statistics(cell_scales, args.seed, args.bootstrap)

    max_orthogonality = float(audit["orthonormality_error"].max())
    max_decomposition = float(audit["energy_decomposition_relative_error"].max())
    rank_gate = bool((audit["basis_rank"] == audit["n_excitatory_coordinates"]).all())
    count_gate = bool((audit["scale_mode_count"] == audit["n_excitatory_coordinates"] - 1).all())
    gates = {
        "full_rank_basis": rank_gate,
        "complete_scale_mode_count": count_gate,
        "orthonormality_at_most_1e-10": max_orthogonality <= 1e-10,
        "energy_decomposition_at_most_1e-10": max_decomposition <= 1e-10,
        "finite_cell_scale_summary": bool(np.isfinite(cell_scales.select_dtypes(include=[np.number])).all().all()),
    }
    summary = {
        "status": "complete" if all(gates.values()) else "failed_numerical_gate",
        "analysis_type": "frozen structural multiscale capacity analysis",
        "seed": int(args.seed),
        "permutation_draws_per_cell": int(args.shuffles),
        "bootstrap_draws": int(args.bootstrap),
        "input_sha256": {"primary": sha256(args.primary), "secondary": sha256(args.secondary)},
        "n_cells": {str(key): int(value) for key, value in audit.groupby("cohort").size().items()},
        "numerical_gates": gates,
        "max_orthonormality_error": max_orthogonality,
        "max_energy_decomposition_relative_error": max_decomposition,
        "cell_level_tests": tests,
        "boundaries": [
            "Route fields are modeled conductance-weighted ancestry columns, not measured biological teaching signals.",
            "The isotropic and ancestry-permuted references define noise geometries; they are not estimates of in-vivo noise.",
            "All reconstructed cells come from one mouse; cells are the inferential units and permutation draws are nested.",
            "A point system with the same independently addressable coordinates can emulate the linear operation.",
        ],
    }

    modes.to_csv(args.outdir / "mode_spectrum.csv", index=False)
    cell_scales.to_csv(args.outdir / "cell_scale_summary.csv", index=False)
    cohort_scales.to_csv(args.outdir / "cohort_scale_summary.csv", index=False)
    audit.to_csv(args.outdir / "numerical_audit.csv", index=False)
    write_json(args.outdir / "summary.json", summary)
    if not all(gates.values()):
        raise SystemExit("irregular-tree wavelet numerical gate failed")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

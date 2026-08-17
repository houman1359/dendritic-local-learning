#!/usr/bin/env python3
"""Analyze inhibitory route organization in the MICrONS census overlap.

The analysis joins the paper's independent 47-cell v661 reconstruction cohort
to stable cell IDs in the published v795 inhibitory census.  It then maps all
input locations and all known within-column inhibitory contacts onto the v795
dendritic skeleton of every overlapping target.

The primary structural tests were frozen in
``analysis/MICRONS_FUNCTIONAL_INHIBITORY_CONTRACT.md`` before this script was
run: topology of repeated contacts from one inhibitory axon, descendant-domain
size at observed contacts, and sparse capture by the actual inhibitory route
dictionary.  These are anatomy/capacity tests in one mouse, not evidence that
the animal used the modeled learning rule.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
import math
from pathlib import Path
import sys
import zipfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "reconstructed_tree"))
from analyze_microns_morphology_credit import (  # noqa: E402
    ancestry_matrix,
    compress_dendritic_tree,
    parent_map,
)

sys.path.insert(0, str(ROOT / "scripts"))
from journal_style import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    PT_ANNOT,
    PT_LEGEND,
    PT_SMALL,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


DEFAULT_COHORT = ROOT / "source_data" / "microns_v661_replication" / "cohort_manifest.csv"
DEFAULT_CENSUS = ROOT / "external_data" / "microns_inhibitory_census"
DEFAULT_DIGITAL = ROOT / "external_data" / "microns_cave_functional_annotations" / "digital_twin_rows.csv"
DEFAULT_OUTDIR = ROOT / "source_data" / "microns_inhibitory_routes"
DEFAULT_FIGURE_STEM = ROOT / "figures" / "generated" / "fig_microns_inhibitory_routes"
INH = COLORS["inh"]
GENERIC = COLORS["point_mlp"]
MORPH = COLORS["shunting"]
RANDOM = COLORS["point_mlp"]
DEPTH = COLORS["additive"]
SHUFFLE = COLORS["highlight"]
DENSE = COLORS["oracle"]


def stable_rng(seed: int, *identifiers: int) -> np.random.Generator:
    words = [int(seed) & 0xFFFFFFFF]
    for value in identifiers:
        value = int(value)
        words.extend([value & 0xFFFFFFFF, (value >> 32) & 0xFFFFFFFF])
    return np.random.default_rng(np.random.SeedSequence(words))


def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.columns = [str(column).strip() for column in out.columns]
    out = out.loc[:, ~out.columns.str.match(r"^Unnamed")]
    return out


def fit_slanted_coordinate_transform(inhibitory: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Recover the published raw-voxel to slanted-micrometre transform.

    The census synapse table contains the same cleft locations in both frames.
    We use those paired coordinates to recover the affine map applied by the
    MICrONS standard transform, and fail loudly if it does not reproduce the
    published micrometre coordinates to numerical precision.
    """
    raw_columns = ["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]
    um_columns = ["ctr_pt_position_um_x", "ctr_pt_position_um_y", "ctr_pt_position_um_z"]
    paired = inhibitory[raw_columns + um_columns].dropna().copy()
    raw = paired[raw_columns].to_numpy(dtype=float)
    target = paired[um_columns].to_numpy(dtype=float)
    design = np.column_stack([raw, np.ones(len(raw), dtype=float)])
    affine, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = np.linalg.norm(design @ affine - target, axis=1)
    if float(residual.max()) > 1e-6:
        raise ValueError(f"raw-to-slanted coordinate transform residual is too large: {residual.max():.3g} um")
    audit = {
        "n_paired_locations": int(len(raw)),
        "maximum_residual_um": float(residual.max()),
        "median_residual_um": float(np.median(residual)),
        "affine_raw_xyz1_to_slanted_um": affine.tolist(),
    }
    return affine, audit


def apply_affine_positions(raw_positions: np.ndarray, affine: np.ndarray) -> np.ndarray:
    raw_positions = np.asarray(raw_positions, dtype=float)
    return np.column_stack([raw_positions, np.ones(len(raw_positions), dtype=float)]) @ affine


def read_swc(archive: zipfile.ZipFile, root_id: int) -> pd.DataFrame:
    member = f"swc/{int(root_id)}.swc"
    with archive.open(member) as handle:
        frame = pd.read_csv(
            io.BytesIO(handle.read()),
            sep=r"\s+",
            comment="#",
            header=None,
            names=["id", "type", "x", "y", "z", "radius", "parent"],
        )
    # Census SWCs use type 3 at the root and type 4 for excitatory dendrite.
    # The inherited, audited compressor expects one dendrite code (3).
    frame["type"] = np.where(frame["type"].astype(int).isin([3, 4]), 3, frame["type"].astype(int))
    return frame


def input_compartment(frame: pd.DataFrame) -> pd.Series:
    soma = frame["is_soma"].fillna(False).astype(bool)
    apical = frame["is_apical"].fillna(False).astype(bool)
    return pd.Series(np.where(soma, "soma", np.where(apical, "apical", "other_dendrite")), index=frame.index)


def map_positions(
    positions_um: np.ndarray,
    rows: pd.DataFrame,
    nodes: pd.DataFrame,
    max_distance_um: float,
) -> pd.DataFrame:
    tree = cKDTree(nodes[["x", "y", "z"]].to_numpy(dtype=float))
    distance, index = tree.query(np.asarray(positions_um, dtype=float), k=1)
    out = rows.copy().reset_index(drop=True)
    out["nearest_skeleton_um"] = distance
    out["segment_id"] = nodes.iloc[index]["segment_id"].to_numpy(dtype="int64")
    out["mapping_pass"] = np.isfinite(distance) & (distance <= float(max_distance_um))
    return out


def add_tree_annotations(segments: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], dict[int, list[int]]]:
    root, parent, children = parent_map(segments)
    out = segments.copy().set_index("segment_id", drop=False)
    major: dict[int, int] = {root: root}
    order = out.sort_values("topological_depth")["segment_id"].astype(int).tolist()
    for segment in order:
        if segment == root:
            continue
        par = parent[segment]
        major[segment] = segment if par == root else major[par]
    out["major_branch_id"] = out["segment_id"].map(major).astype("int64")

    descendants: dict[int, list[int]] = {}
    for segment in reversed(order):
        values = [segment]
        for child in children.get(segment, []):
            values.extend(descendants[child])
        descendants[segment] = values
    return out.reset_index(drop=True), parent, descendants


def aggregate_segment_inputs(
    segments: pd.DataFrame,
    mapped_inputs: pd.DataFrame,
    mapped_inhibitory: pd.DataFrame,
    descendants: dict[int, list[int]],
) -> pd.DataFrame:
    out = segments.copy().set_index("segment_id", drop=False)
    inputs = mapped_inputs[mapped_inputs["mapping_pass"]].copy()
    inhibitory = mapped_inhibitory[mapped_inhibitory["mapping_pass"]].copy()
    input_count = inputs.groupby("segment_id")["id"].size()
    input_size = inputs.groupby("segment_id")["size"].sum()
    inh_count = inhibitory.groupby("segment_id")["synapse_id"].size()
    inh_size = inhibitory.groupby("segment_id")["synapse_size"].sum()
    out["input_count"] = input_count.reindex(out.index).fillna(0).astype(int)
    out["input_size"] = input_size.reindex(out.index).fillna(0.0).astype(float)
    out["known_inhibitory_count"] = inh_count.reindex(out.index).fillna(0).astype(int)
    out["known_inhibitory_size"] = inh_size.reindex(out.index).fillna(0.0).astype(float)
    for field in ["input_count", "input_size"]:
        values = out[field].to_dict()
        out[f"descendant_{field}"] = [
            float(sum(values[node] for node in descendants[int(segment)]))
            for segment in out["segment_id"]
        ]
    total_count = max(float(out["input_count"].sum()), 1.0)
    total_size = max(float(out["input_size"].sum()), 1.0)
    out["descendant_input_fraction"] = out["descendant_input_count"] / total_count
    out["descendant_size_fraction"] = out["descendant_input_size"] / total_size
    out["route_specificity_bits"] = -np.log2(out["descendant_input_fraction"].clip(lower=1.0 / total_count))
    return out.reset_index(drop=True)


def distance_matrices(
    segments: pd.DataFrame, parent: dict[int, int]
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    ids = segments["segment_id"].astype(int).tolist()
    index = {segment: idx for idx, segment in enumerate(ids)}
    path_length = segments.set_index("segment_id")["path_length_um"].astype(float).to_dict()
    ancestors: dict[int, list[int]] = {}
    for segment in ids:
        chain = [segment]
        while chain[-1] in parent:
            chain.append(parent[chain[-1]])
        ancestors[segment] = chain
    distance = np.zeros((len(ids), len(ids)), dtype=float)
    shared = np.zeros_like(distance)
    for i, left in enumerate(ids):
        left_set = set(ancestors[left])
        for j in range(i, len(ids)):
            right = ids[j]
            lca = next(node for node in ancestors[right] if node in left_set)
            lca_path = float(path_length[lca])
            left_path = float(path_length[left])
            right_path = float(path_length[right])
            d = left_path + right_path - 2.0 * lca_path
            denom = left_path + right_path
            s = 1.0 if denom <= 0 else 2.0 * lca_path / denom
            distance[i, j] = distance[j, i] = d
            shared[i, j] = shared[j, i] = s
    # A route is the weighted set of mapped input locations descended from a
    # candidate control site.  Its cosine overlap is the fraction of the same
    # input-weighted credit domain addressed by two sites.  This makes the
    # connection-level topology test directly interpretable as co-control of a
    # common descendant domain rather than only geometric proximity.
    input_frame = segments.loc[segments["input_count"] > 0].copy()
    input_segments = input_frame["segment_id"].astype(int).tolist()
    route = ancestry_matrix(input_segments, ids, parent)
    weight = np.sqrt(np.clip(input_frame["input_size"].to_numpy(dtype=float), 0.0, None))
    if not np.any(weight > 0):
        weight = np.sqrt(input_frame["input_count"].to_numpy(dtype=float))
    route = weight[:, None] * route
    norm = np.linalg.norm(route, axis=0)
    route = np.divide(route, norm[None, :], out=np.zeros_like(route), where=norm[None, :] > 0)
    route_overlap = route.T @ route
    return ids, distance, shared, route_overlap


def connection_pair_mean(segment_indices: np.ndarray, matrix: np.ndarray) -> float:
    segment_indices = np.asarray(segment_indices, dtype=int)
    if len(segment_indices) < 2:
        return float("nan")
    rows, cols = np.triu_indices(len(segment_indices), k=1)
    return float(matrix[segment_indices[rows], segment_indices[cols]].mean())


def build_candidate_segments(
    contacts: pd.DataFrame,
    generic_inputs: pd.DataFrame,
    n_candidates: int,
    spatially_matched: bool = False,
) -> dict[int, np.ndarray]:
    candidates: dict[int, np.ndarray] = {}
    pools = {
        compartment: group.copy()
        for compartment, group in generic_inputs[generic_inputs["mapping_pass"]].groupby("compartment")
    }
    all_pool = generic_inputs[generic_inputs["mapping_pass"]].copy()
    for row in contacts.itertuples(index=False):
        pool = pools.get(str(row.compartment), all_pool)
        if len(pool) < 8:
            pool = all_pool
        delta = np.abs(
            np.log1p(pd.to_numeric(pool["dist_to_root"], errors="coerce").fillna(0.0).to_numpy(dtype=float))
            - math.log1p(max(float(row.dist_to_root), 0.0))
        )
        if spatially_matched:
            coordinates = pool[["x_um", "y_um", "z_um"]].to_numpy(dtype=float)
            focal_coordinate = np.asarray([row.x_um, row.y_um, row.z_um], dtype=float)
            euclidean = np.linalg.norm(coordinates - focal_coordinate[None, :], axis=1)
            # Percentile ranks put path-distance and three-dimensional proximity
            # on the same transparent scale without fitting a nuisance model.
            path_rank = stats.rankdata(delta, method="average") / max(len(delta), 1)
            euclidean_rank = stats.rankdata(euclidean, method="average") / max(len(euclidean), 1)
            score = path_rank + euclidean_rank
        else:
            score = delta
        order = np.argsort(score, kind="stable")[: min(int(n_candidates), len(pool))]
        values = pool.iloc[order]["segment_id"].to_numpy(dtype="int64")
        if values.size == 0:
            raise RuntimeError(f"no matched candidates for synapse {row.synapse_id}")
        candidates[int(row.synapse_id)] = values
    return candidates


def representative_clump_contacts(mapped_inhibitory: pd.DataFrame) -> pd.DataFrame:
    passed = mapped_inhibitory[mapped_inhibitory["mapping_pass"]].copy()
    passed["synapse_size"] = pd.to_numeric(passed["synapse_size"], errors="coerce").fillna(0.0)
    passed = passed.sort_values(
        ["pre_soma_id", "post_soma_id", "syn_clump_comp", "synapse_size", "synapse_id"],
        ascending=[True, True, True, False, True],
    )
    return passed.drop_duplicates(["pre_soma_id", "post_soma_id", "syn_clump_comp"], keep="first").copy()


def analyze_connections(
    cell_id: int,
    representatives: pd.DataFrame,
    candidate_segments: dict[int, np.ndarray],
    segment_ids: list[int],
    distance_matrix: np.ndarray,
    shared_matrix: np.ndarray,
    route_overlap_matrix: np.ndarray,
    n_shuffles: int,
    seed: int,
) -> pd.DataFrame:
    index = {segment: idx for idx, segment in enumerate(segment_ids)}
    rows: list[dict] = []
    for pre_id, group in representatives.groupby("pre_soma_id", sort=True):
        if len(group) < 2:
            continue
        synapse_ids = group["synapse_id"].astype(int).tolist()
        actual_idx = np.asarray([index[int(value)] for value in group["segment_id"]], dtype=int)
        actual_distance = connection_pair_mean(actual_idx, distance_matrix)
        actual_shared = connection_pair_mean(actual_idx, shared_matrix)
        actual_route_overlap = connection_pair_mean(actual_idx, route_overlap_matrix)
        rng = stable_rng(seed, cell_id, int(pre_id))
        sampled = np.empty((int(n_shuffles), len(group)), dtype=int)
        for column, synapse_id in enumerate(synapse_ids):
            options = candidate_segments[synapse_id]
            sampled[:, column] = [index[int(value)] for value in rng.choice(options, size=int(n_shuffles), replace=True)]
        null_distance = np.asarray([connection_pair_mean(values, distance_matrix) for values in sampled])
        null_shared = np.asarray([connection_pair_mean(values, shared_matrix) for values in sampled])
        null_route_overlap = np.asarray(
            [connection_pair_mean(values, route_overlap_matrix) for values in sampled]
        )
        rows.append(
            {
                "post_soma_id": int(cell_id),
                "pre_soma_id": int(pre_id),
                "m_type_pre": str(group["m_type_pre"].iloc[0]),
                "n_contacts": int(group["synapse_id"].nunique()),
                "n_axonal_clumps": int(len(group)),
                "actual_pair_tree_distance_um": actual_distance,
                "matched_pair_tree_distance_um": float(null_distance.mean()),
                "tree_distance_delta_um": actual_distance - float(null_distance.mean()),
                "tree_distance_null_sd": float(null_distance.std(ddof=1)),
                "actual_shared_path_fraction": actual_shared,
                "matched_shared_path_fraction": float(null_shared.mean()),
                "shared_path_delta": actual_shared - float(null_shared.mean()),
                "shared_path_null_sd": float(null_shared.std(ddof=1)),
                "actual_route_overlap": actual_route_overlap,
                "matched_route_overlap": float(null_route_overlap.mean()),
                "route_overlap_delta": actual_route_overlap - float(null_route_overlap.mean()),
                "route_overlap_null_sd": float(null_route_overlap.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows)


def reconstruction_residual(dictionary: np.ndarray, target: np.ndarray) -> float:
    if dictionary.size == 0:
        return 1.0
    coefficient, *_ = np.linalg.lstsq(dictionary, target, rcond=None)
    reconstruction = dictionary @ coefficient
    denominator = float(np.linalg.norm(target))
    return float(np.linalg.norm(target - reconstruction) / denominator) if denominator > 0 else float("nan")


def route_capture_curves(
    segments: pd.DataFrame,
    parent: dict[int, int],
    cell_id: int,
    n_random: int,
    seed: int,
) -> pd.DataFrame:
    input_segments = segments.loc[segments["input_count"] > 0, "segment_id"].astype(int).tolist()
    inhibitory_segments = segments.loc[segments["known_inhibitory_count"] > 0, "segment_id"].astype(int).tolist()
    if len(input_segments) < 2 or len(inhibitory_segments) < 1:
        return pd.DataFrame()
    route = ancestry_matrix(input_segments, inhibitory_segments, parent)
    seg = segments.set_index("segment_id")
    row_weight = np.sqrt(np.clip(seg.loc[input_segments, "input_size"].to_numpy(dtype=float), 0.0, None))
    if not np.any(row_weight > 0):
        row_weight = np.sqrt(seg.loc[input_segments, "input_count"].to_numpy(dtype=float))
    row_weight /= np.sqrt(np.mean(row_weight * row_weight))
    column_weight = np.sqrt(
        np.clip(seg.loc[inhibitory_segments, "known_inhibitory_size"].to_numpy(dtype=float), 0.0, None)
    )
    if not np.any(column_weight > 0):
        column_weight = np.sqrt(seg.loc[inhibitory_segments, "known_inhibitory_count"].to_numpy(dtype=float))
    target = row_weight[:, None] * route * column_weight[None, :]
    leverage = np.linalg.norm(target, axis=0)
    selected_order = np.argsort(-leverage, kind="stable")
    singular = np.linalg.svd(target, compute_uv=False)
    singular_energy = singular * singular
    total_energy = float(np.sum(singular_energy))
    rng = stable_rng(seed, cell_id, 991)
    input_depth = seg.loc[input_segments, "path_length_um"].to_numpy(dtype=float)
    depth_edges = np.unique(np.quantile(input_depth, np.linspace(0.0, 1.0, 9)))
    depth_strata = np.digitize(input_depth, depth_edges[1:-1], right=True)
    rows: list[dict] = []
    for count in [1, 2, 4, 8, 16]:
        if count > min(len(input_segments), len(inhibitory_segments)):
            continue
        dense_capture = float(np.sum(singular_energy[:count]) / total_energy)
        rows.append({"post_soma_id": cell_id, "routes": count, "method": "dense SVD oracle", "capture": dense_capture})

        selected = selected_order[:count]
        dictionary = target[:, selected]
        morph_capture = 1.0 - reconstruction_residual(dictionary, target) ** 2
        rows.append({"post_soma_id": cell_id, "routes": count, "method": "morphology-selected actual routes", "capture": morph_capture})

        random_values = []
        shuffled_values = []
        for _ in range(int(n_random)):
            choice = rng.choice(len(inhibitory_segments), size=count, replace=False)
            random_dictionary = target[:, choice]
            random_values.append(1.0 - reconstruction_residual(random_dictionary, target) ** 2)

            shuffled = dictionary.copy()
            for column in range(shuffled.shape[1]):
                for stratum in np.unique(depth_strata):
                    loc = np.flatnonzero(depth_strata == stratum)
                    shuffled[loc, column] = rng.permutation(shuffled[loc, column])
            shuffled_values.append(1.0 - reconstruction_residual(shuffled, target) ** 2)
        rows.append({"post_soma_id": cell_id, "routes": count, "method": "random actual routes", "capture": float(np.mean(random_values)), "null_sd": float(np.std(random_values, ddof=1))})
        rows.append({"post_soma_id": cell_id, "routes": count, "method": "depth-preserving ancestry shuffle", "capture": float(np.mean(shuffled_values)), "null_sd": float(np.std(shuffled_values, ddof=1))})

        labels = pd.qcut(pd.Series(input_depth), q=min(count, len(np.unique(input_depth))), labels=False, duplicates="drop").to_numpy(dtype=int)
        depth_dictionary = np.eye(int(labels.max()) + 1, dtype=float)[labels] * row_weight[:, None]
        depth_capture = 1.0 - reconstruction_residual(depth_dictionary, target) ** 2
        rows.append({"post_soma_id": cell_id, "routes": count, "method": "depth bins", "capture": depth_capture})
    return pd.DataFrame(rows)


def summarize_target(
    cell: pd.Series,
    segments: pd.DataFrame,
    mapped_inputs: pd.DataFrame,
    mapped_inhibitory: pd.DataFrame,
    representatives: pd.DataFrame,
    candidate_segments: dict[int, np.ndarray],
    connections: pd.DataFrame,
) -> dict:
    seg = segments.set_index("segment_id")
    contact_domain_rows = []
    for row in representatives.itertuples(index=False):
        actual = seg.loc[int(row.segment_id)]
        candidates = seg.loc[candidate_segments[int(row.synapse_id)]]
        contact_domain_rows.append(
            {
                "actual_descendant_fraction": float(actual.descendant_input_fraction),
                "matched_descendant_fraction": float(candidates.descendant_input_fraction.mean()),
                "actual_specificity_bits": float(actual.route_specificity_bits),
                "matched_specificity_bits": float(candidates.route_specificity_bits.mean()),
            }
        )
    domain = pd.DataFrame(contact_domain_rows)
    class_domain = (
        representatives[representatives["m_type_pre"].isin(["DTC", "PTC"])]
        .groupby("m_type_pre")["descendant_input_fraction"]
        .median()
    )
    record = {
        "post_soma_id": int(cell["nucleus_id"]),
        "v661_root_id": int(cell["replication_root_id"]),
        "v795_root_id": int(cell["pt_root_id"]),
        "cell_type": str(cell["cell_type"]),
        "m_type": str(cell["m_type"]),
        "n_segments": int(len(segments)),
        "n_mapped_inputs": int(mapped_inputs["mapping_pass"].sum()),
        "input_mapping_fraction": float(mapped_inputs["mapping_pass"].mean()),
        "n_known_inhibitory_contacts": int(mapped_inhibitory["mapping_pass"].sum()),
        "inhibitory_mapping_fraction": float(mapped_inhibitory["mapping_pass"].mean()),
        "n_inhibitory_cells": int(mapped_inhibitory.loc[mapped_inhibitory.mapping_pass, "pre_soma_id"].nunique()),
        "n_inhibitory_connections": int(mapped_inhibitory.loc[mapped_inhibitory.mapping_pass, ["pre_soma_id", "post_soma_id"]].drop_duplicates().shape[0]),
        "n_connections_with_two_clumps": int(len(connections)),
        "actual_connection_tree_distance_um": float(connections["actual_pair_tree_distance_um"].mean()) if len(connections) else np.nan,
        "matched_connection_tree_distance_um": float(connections["matched_pair_tree_distance_um"].mean()) if len(connections) else np.nan,
        "connection_tree_distance_delta_um": float(connections["tree_distance_delta_um"].mean()) if len(connections) else np.nan,
        "actual_connection_shared_path": float(connections["actual_shared_path_fraction"].mean()) if len(connections) else np.nan,
        "matched_connection_shared_path": float(connections["matched_shared_path_fraction"].mean()) if len(connections) else np.nan,
        "connection_shared_path_delta": float(connections["shared_path_delta"].mean()) if len(connections) else np.nan,
        "actual_connection_route_overlap": float(connections["actual_route_overlap"].mean()) if len(connections) else np.nan,
        "matched_connection_route_overlap": float(connections["matched_route_overlap"].mean()) if len(connections) else np.nan,
        "connection_route_overlap_delta": float(connections["route_overlap_delta"].mean()) if len(connections) else np.nan,
        "actual_contact_descendant_fraction": float(domain["actual_descendant_fraction"].mean()),
        "matched_contact_descendant_fraction": float(domain["matched_descendant_fraction"].mean()),
        "contact_descendant_fraction_delta": float((domain["actual_descendant_fraction"] - domain["matched_descendant_fraction"]).mean()),
        "actual_contact_specificity_bits": float(domain["actual_specificity_bits"].mean()),
        "matched_contact_specificity_bits": float(domain["matched_specificity_bits"].mean()),
        "contact_specificity_delta_bits": float((domain["actual_specificity_bits"] - domain["matched_specificity_bits"]).mean()),
        "dtc_median_descendant_fraction": float(class_domain.get("DTC", np.nan)),
        "ptc_median_descendant_fraction": float(class_domain.get("PTC", np.nan)),
        "dtc_minus_ptc_median_descendant_fraction": float(
            class_domain.get("DTC", np.nan) - class_domain.get("PTC", np.nan)
        ),
    }
    return record


def paired_test(values: np.ndarray, alternative: str) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    statistic, p_value = stats.wilcoxon(values, alternative=alternative)
    rng = np.random.default_rng(4242)
    boot = rng.choice(values, size=(50_000, len(values)), replace=True).mean(axis=1)
    return {
        "n_targets": int(len(values)),
        "mean_delta": float(values.mean()),
        "median_delta": float(np.median(values)),
        "bootstrap_95_ci": [float(value) for value in np.quantile(boot, [0.025, 0.975])],
        "ordering_positive": int(np.sum(values > 0)),
        "ordering_negative": int(np.sum(values < 0)),
        "wilcoxon_alternative": alternative,
        "wilcoxon_p": float(p_value),
    }


def aggregate_digital_twin(digital: pd.DataFrame) -> pd.DataFrame:
    if digital.empty:
        return pd.DataFrame(columns=["post_soma_id"])
    metrics = [column for column in ["gOSI", "gDSI", "OSI", "DSI", "cc_abs", "cc_max", "cc_norm"] if column in digital]
    aggregate = digital.groupby("target_id")[metrics].mean().reset_index().rename(columns={"target_id": "post_soma_id"})
    aggregate["n_digital_twin_rows"] = digital.groupby("target_id").size().reindex(aggregate.post_soma_id).to_numpy()
    return aggregate


def plot_examples(ax: plt.Axes, example_payloads: list[dict]) -> None:
    ax.axis("off")
    # Keep this heading compact: panel A is narrow and its title sits directly
    # to the left at manuscript scale.
    panel_title(ax, "B", "Inhibitory contacts on six trees")
    slots = [(0.02, 0.68), (0.52, 0.68), (0.02, 0.36), (0.52, 0.36), (0.02, 0.04), (0.52, 0.04)]
    for payload, (x0, y0) in zip(example_payloads[:6], slots):
        nodes = payload["nodes"]
        contacts = payload["contacts"]
        x = nodes["x"].to_numpy(dtype=float)
        y = nodes["y"].to_numpy(dtype=float)
        x_mid = 0.5 * (np.nanmin(x) + np.nanmax(x))
        y_mid = 0.5 * (np.nanmin(y) + np.nanmax(y))
        scale = min(
            0.42 / max(np.nanmax(x) - np.nanmin(x), 1e-9),
            0.22 / max(np.nanmax(y) - np.nanmin(y), 1e-9),
        )
        x_plot = x0 + 0.22 + scale * (x - x_mid)
        y_plot = y0 + 0.12 + scale * (y - y_mid)
        ax.scatter(x_plot, y_plot, s=0.20, color="#8FBFA5", alpha=0.52, rasterized=True)
        cx = contacts["x_um"].to_numpy(dtype=float)
        cy = contacts["y_um"].to_numpy(dtype=float)
        ax.scatter(
            x0 + 0.22 + scale * (cx - x_mid),
            y0 + 0.12 + scale * (cy - y_mid),
            s=2.4,
            color=INH,
            alpha=0.72,
            linewidth=0,
            rasterized=True,
        )
        ax.text(
            x0 + 0.22,
            y0 - 0.005,
            f"{payload['m_type']} | n={len(contacts)}",
            ha="center",
            va="top",
            fontsize=PT_SMALL,
        )
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.02, 1.02)


def make_figure(
    cohort: pd.DataFrame,
    contacts: pd.DataFrame,
    target: pd.DataFrame,
    axon: pd.DataFrame,
    spatial_target: pd.DataFrame,
    route: pd.DataFrame,
    example_payloads: list[dict],
    stem: Path,
) -> None:
    apply_neurips_style()
    # Keep all ten endpoints in one readable, page-safe figure.  The broad
    # reconstruction panel spans three columns; the quantitative panels then
    # form two compact rows.  This avoids shrinking a five-row portrait figure
    # to the point that labels and example morphologies become unreadable.
    fig = plt.figure(figsize=(FIG_W, 6.35))
    gs = fig.add_gridspec(
        3,
        4,
        left=0.075,
        right=0.985,
        bottom=0.08,
        top=0.915,
        wspace=0.78,
        hspace=0.72,
        height_ratios=[0.86, 1.0, 1.0],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[1, 2])
    ax_f = fig.add_subplot(gs[1, 3])
    ax_g = fig.add_subplot(gs[2, 0])
    ax_h = fig.add_subplot(gs[2, 1])
    ax_i = fig.add_subplot(gs[2, 2])
    ax_j = fig.add_subplot(gs[2, 3])

    panel_title(ax_a, "A", "Cohort overlap")
    counts = [47, len(cohort), int(cohort["has_digital_twin"].sum())]
    labels = ["v661", "inh. census", "functional"]
    ax_a.bar(range(3), counts, color=[GENERIC, INH, COLORS["oracle"]], edgecolor=COLORS["edge"], lw=LW_EDGE)
    for idx, value in enumerate(counts):
        ax_a.text(idx, value + 1.0, str(value), ha="center", va="bottom", fontsize=PT_ANNOT, fontweight="bold")
    ax_a.set_xticks(range(3), labels)
    for label in ax_a.get_xticklabels():
        label.set_rotation(24)
        label.set_ha("right")
    ax_a.set_ylabel("target cells")
    ax_a.set_ylim(0, 54)
    style_axis(ax_a, grid="y")

    plot_examples(ax_b, example_payloads)

    panel_title(ax_c, "C", "Presynaptic class")
    class_counts = contacts["m_type_pre"].value_counts().reindex(["DTC", "PTC", "STC", "ITC"]).dropna()
    ax_c.bar(range(len(class_counts)), class_counts.to_numpy(), color=[INH, COLORS["additive"], COLORS["oracle"], COLORS["point_mlp"]][: len(class_counts)], edgecolor=COLORS["edge"], lw=LW_EDGE)
    ax_c.set_xticks(range(len(class_counts)), class_counts.index)
    for label in ax_c.get_xticklabels():
        label.set_rotation(24)
        label.set_ha("right")
    ax_c.set_ylabel("contacts")
    style_axis(ax_c, grid="y")

    primary_sensitivity = target.rename(
        columns={
            "connection_tree_distance_delta_um": "tree_distance_delta_um",
            "connection_shared_path_delta": "shared_path_delta",
            "connection_route_overlap_delta": "route_overlap_delta",
        }
    )

    def contrast_panel(ax, letter: str, title: str, column: str, ylabel: str, seed: int) -> None:
        frames = [primary_sensitivity, axon, spatial_target]
        colors = [INH, COLORS["additive"], COLORS["oracle"]]
        rng = np.random.default_rng(seed)
        for index, (frame, color) in enumerate(zip(frames, colors, strict=True)):
            values = frame[column].dropna().to_numpy(dtype=float)
            x = index + rng.uniform(-0.055, 0.055, size=len(values))
            ax.scatter(x, values, s=8, color=color, alpha=0.35, edgecolor="none", rasterized=True)
            draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
            lo, hi = np.quantile(draws, [0.025, 0.975])
            mean = float(values.mean())
            ax.errorbar(
                index,
                mean,
                yerr=[[mean - lo], [hi - mean]],
                fmt="D",
                ms=4.3,
                color=color,
                mec="white",
                mew=0.4,
                capsize=ERR_CAPSIZE,
                lw=LW_ERR,
                zorder=5,
            )
        ax.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
        ax.set_xticks([0, 1, 2], ["target", "axon", "3D"])
        ax.set_ylabel(ylabel)
        panel_title(ax, letter, title)
        style_axis(ax, grid="y")

    contrast_panel(
        ax_d,
        "D",
        "Tree proximity",
        "tree_distance_delta_um",
        "observed - matched (µm)",
        202608041,
    )
    contrast_panel(
        ax_e,
        "E",
        "Path overlap",
        "shared_path_delta",
        "observed - matched",
        202608042,
    )
    contrast_panel(
        ax_f,
        "F",
        "Domain match",
        "route_overlap_delta",
        "observed - matched",
        202608043,
    )
    panel_title(ax_g, "G", "Class-specific domains")
    representative = contacts["is_clump_representative"].fillna(False).astype(bool)
    for mtype, color in [("DTC", INH), ("PTC", COLORS["additive"])]:
        x = contacts.loc[
            contacts.mapping_pass & representative & contacts.m_type_pre.eq(mtype),
            "descendant_input_fraction",
        ].clip(lower=1e-4).to_numpy(dtype=float)
        if len(x):
            ordered = np.sort(x)
            ax_g.plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), color=color, lw=LW_DATA, label=mtype)
    ax_g.set_xscale("log")
    ax_g.set_xlabel("descendant input fraction")
    ax_g.set_ylabel("cumulative fraction")
    clean_legend(ax_g, loc="upper left", fontsize=PT_LEGEND)
    style_axis(ax_g, grid="x")

    panel_title(ax_h, "H", "Placement control")
    for row in target.itertuples(index=False):
        ax_h.plot(
            [0, 1],
            [row.matched_contact_descendant_fraction, row.actual_contact_descendant_fraction],
            color=COLORS["mute"],
            lw=LW_HAIR,
            alpha=0.48,
        )
        ax_h.scatter(
            [0, 1],
            [row.matched_contact_descendant_fraction, row.actual_contact_descendant_fraction],
            s=12,
            color=[GENERIC, INH],
            zorder=3,
        )
    ax_h.set_xticks([0, 1], ["matched", "obs."])
    ax_h.set_ylabel("domain fraction")
    style_axis(ax_h, grid="y")

    panel_title(ax_i, "I", "Route capacity")
    method_style = {
        "dense SVD oracle": (DENSE, "-"),
        "morphology-selected actual routes": (MORPH, "-"),
        "random actual routes": (RANDOM, "--"),
        "depth bins": (DEPTH, ":"),
        "depth-preserving ancestry shuffle": (SHUFFLE, "-."),
    }
    for method, (color, linestyle) in method_style.items():
        frame = route[route.method == method]
        summary = frame.groupby("routes")["capture"].agg(["mean", "sem"]).reset_index()
        ax_i.plot(
            summary.routes,
            summary["mean"],
            color=color,
            ls=linestyle,
            marker="o",
            ms=3.4,
            lw=LW_DATA,
            label=method.replace("morphology-selected actual routes", "selected sites").replace(
                "depth-preserving ancestry shuffle", "ancestry shuffle"
            ),
        )
        ax_i.fill_between(
            summary.routes,
            summary["mean"] - summary["sem"],
            summary["mean"] + summary["sem"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )
        label = {
            "dense SVD oracle": "oracle",
            "morphology-selected actual routes": "selected",
            "random actual routes": "random",
            "depth bins": "depth",
            "depth-preserving ancestry shuffle": "shuffle",
        }[method]
        offset = {
            "dense SVD oracle": 6,
            "morphology-selected actual routes": -7,
            "random actual routes": 1,
            "depth bins": 7,
            "depth-preserving ancestry shuffle": -7,
        }[method]
        ax_i.annotate(
            label,
            xy=(float(summary.routes.iloc[-1]), float(summary["mean"].iloc[-1])),
            xytext=(4, offset),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=PT_SMALL,
            color=color,
        )
    ax_i.set_xscale("log", base=2)
    ax_i.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax_i.set_xlabel("selected inhibitory sites")
    # Reserve a right-hand labeling lane so direct curve labels remain inside
    # this panel and do not collide with panel J.
    ax_i.set_xlim(0.85, 96)
    ax_i.set_ylabel("energy capture")
    style_axis(ax_i, grid="both")

    panel_title(ax_j, "J", "8-site capture")
    fixed = route[route.routes == 8].pivot(index="post_soma_id", columns="method", values="capture").dropna()
    for _, row in fixed.iterrows():
        ax_j.plot(
            [0, 1, 2],
            [
                row["morphology-selected actual routes"],
                row["random actual routes"],
                row["depth-preserving ancestry shuffle"],
            ],
            color=COLORS["mute"],
            lw=LW_HAIR,
            alpha=0.4,
        )
    for x, method, color in [(0, "morphology-selected actual routes", MORPH), (1, "random actual routes", RANDOM), (2, "depth-preserving ancestry shuffle", SHUFFLE)]:
        vals = fixed[method].to_numpy(dtype=float)
        ax_j.scatter(np.full(len(vals), x), vals, color=color, s=11, alpha=0.75, zorder=3)
        ax_j.errorbar(x, vals.mean(), yerr=stats.sem(vals), fmt="D", ms=4.3, color=color, mec="white", mew=0.4, capsize=ERR_CAPSIZE, lw=LW_ERR, zorder=5)
    ax_j.set_xticks([0, 1, 2], ["selected", "random", "shuff."])
    for label in ax_j.get_xticklabels():
        label.set_rotation(24)
        label.set_ha("right")
    ax_j.set_ylabel("energy capture")
    style_axis(ax_j, grid="y")

    fig.canvas.draw()
    layout = audit_layout(fig, stem.name)
    overlap = audit_text_over_data(fig, stem.name)
    if layout or overlap:
        print(f"layout audit: {len(layout)} layout and {len(overlap)} text/data warnings")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def make_streamlined_figure(
    contacts: pd.DataFrame,
    target: pd.DataFrame,
    axon: pd.DataFrame,
    spatial_target: pd.DataFrame,
    route: pd.DataFrame,
    example_payloads: list[dict],
    stem: Path,
) -> None:
    """Replace the ten-panel census display with a six-claim main figure."""

    apply_neurips_style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FIG_W, 4.85),
        gridspec_kw={
            "left": 0.085,
            "right": 0.985,
            "bottom": 0.105,
            "top": 0.91,
            "wspace": 0.58,
            "hspace": 0.68,
        },
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()

    panel_title(ax_a, "A", "Inhibitory contacts")
    ax_a.axis("off")
    example = example_payloads[0]
    nodes = example["nodes"]
    example_contacts = example["contacts"]
    x = nodes["x"].to_numpy(float)
    y = nodes["y"].to_numpy(float)
    x_mid = 0.5 * (np.nanmin(x) + np.nanmax(x))
    y_mid = 0.5 * (np.nanmin(y) + np.nanmax(y))
    scale = 0.90 / max(np.nanmax(x) - np.nanmin(x), np.nanmax(y) - np.nanmin(y), 1e-9)
    ax_a.scatter(
        0.5 + scale * (x - x_mid),
        0.5 + scale * (y - y_mid),
        s=0.45,
        color="#8FBFA5",
        alpha=0.55,
        rasterized=True,
    )
    ax_a.scatter(
        0.5 + scale * (example_contacts["x_um"].to_numpy(float) - x_mid),
        0.5 + scale * (example_contacts["y_um"].to_numpy(float) - y_mid),
        s=4.0,
        color=INH,
        alpha=0.72,
        linewidth=0,
        rasterized=True,
    )
    ax_a.text(
        0.03,
        0.05,
        f"{example['m_type']} target · {len(example_contacts)} mapped contacts",
        transform=ax_a.transAxes,
        fontsize=PT_SMALL,
        color=COLORS["ink"],
    )
    ax_a.set_xlim(0, 1)
    ax_a.set_ylim(0, 1)

    panel_title(ax_b, "B", "Class composition")
    representative = contacts["is_clump_representative"].fillna(False).astype(bool)
    class_counts = contacts.loc[
        contacts.mapping_pass & representative, "m_type_pre"
    ].value_counts().reindex(["DTC", "PTC", "STC", "ITC"]).dropna()
    colors = [INH, COLORS["additive"], COLORS["oracle"], COLORS["point_mlp"]]
    ax_b.bar(
        range(len(class_counts)),
        class_counts.to_numpy(),
        color=colors[: len(class_counts)],
        edgecolor=COLORS["edge"],
        lw=LW_EDGE,
    )
    ax_b.set_xticks(range(len(class_counts)), class_counts.index)
    ax_b.set_ylabel("independent contact clumps")
    style_axis(ax_b, grid="y")

    primary = target.rename(
        columns={
            "connection_tree_distance_delta_um": "tree_distance_delta_um",
            "connection_shared_path_delta": "shared_path_delta",
            "connection_route_overlap_delta": "route_overlap_delta",
        }
    )
    frames = [(primary, "target", INH), (axon, "axon", COLORS["additive"]), (spatial_target, "3D", COLORS["oracle"])]

    def add_interval(ax: plt.Axes, position: float, values: np.ndarray, color: str, seed: int) -> None:
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        rng = np.random.default_rng(seed)
        ax.scatter(
            position + rng.uniform(-0.05, 0.05, size=len(values)),
            values,
            s=7,
            color=color,
            alpha=0.30,
            edgecolors="none",
            rasterized=True,
        )
        draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
        low, high = np.quantile(draws, [0.025, 0.975])
        mean = float(values.mean())
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            marker="D",
            ms=4.3,
            color=color,
            markeredgecolor="white",
            markeredgewidth=0.4,
            capsize=ERR_CAPSIZE,
            lw=LW_ERR,
            zorder=5,
        )

    panel_title(ax_c, "C", "Tree proximity")
    for index, (frame, _, color) in enumerate(frames):
        add_interval(ax_c, index, frame.tree_distance_delta_um.to_numpy(float), color, 1900 + index)
    ax_c.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_c.set_xticks(range(3), [label for _, label, _ in frames])
    ax_c.set_ylabel("observed - matched distance (µm)")
    style_axis(ax_c, grid="y")

    panel_title(ax_d, "D", "Fine-route boundary")
    offsets = {"shared_path_delta": -0.09, "route_overlap_delta": 0.09}
    endpoint_colors = {"shared_path_delta": MORPH, "route_overlap_delta": COLORS["pathway"]}
    for endpoint, endpoint_label in [
        ("shared_path_delta", "shared path"),
        ("route_overlap_delta", "domain overlap"),
    ]:
        means = []
        for index, (frame, _, _) in enumerate(frames):
            values = frame[endpoint].dropna().to_numpy(float)
            rng = np.random.default_rng(1930 + index + (0 if endpoint == "shared_path_delta" else 10))
            draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
            low, high = np.quantile(draws, [0.025, 0.975])
            mean = float(values.mean())
            means.append(mean)
            ax_d.errorbar(
                index + offsets[endpoint],
                mean,
                yerr=[[mean - low], [high - mean]],
                marker="o" if endpoint == "shared_path_delta" else "s",
                ms=4.2,
                color=endpoint_colors[endpoint],
                capsize=ERR_CAPSIZE,
                lw=LW_ERR,
                label=endpoint_label if index == 0 else None,
            )
        ax_d.plot(
            np.arange(3) + offsets[endpoint],
            means,
            color=endpoint_colors[endpoint],
            lw=LW_HAIR,
        )
    ax_d.axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    ax_d.set_xticks(range(3), [label for _, label, _ in frames])
    ax_d.set_ylabel("observed - matched")
    style_axis(ax_d, grid="y")
    clean_legend(ax_d, loc="upper right", fontsize=PT_LEGEND)

    panel_title(ax_e, "E", "Class-specific domains")
    for m_type, color in [("DTC", INH), ("PTC", COLORS["additive"])]:
        values = contacts.loc[
            contacts.mapping_pass & representative & contacts.m_type_pre.eq(m_type),
            "descendant_input_fraction",
        ].clip(lower=1e-4).to_numpy(float)
        ordered = np.sort(values)
        ax_e.plot(
            ordered,
            np.arange(1, len(ordered) + 1) / len(ordered),
            color=color,
            lw=LW_DATA,
            label=m_type,
        )
    ax_e.set_xscale("log")
    ax_e.set_xlabel("descendant input fraction")
    ax_e.set_ylabel("cumulative fraction")
    style_axis(ax_e, grid="x")
    clean_legend(ax_e, loc="upper left", fontsize=PT_LEGEND)

    panel_title(ax_f, "F", "Route capacity")
    method_style = {
        "dense SVD oracle": (DENSE, "SVD oracle"),
        "morphology-selected actual routes": (MORPH, "selected sites"),
        "random actual routes": (RANDOM, "random sites"),
        "depth bins": (DEPTH, "depth bins"),
        "depth-preserving ancestry shuffle": (SHUFFLE, "ancestry shuffle"),
    }
    for method, (color, label) in method_style.items():
        frame = route[route.method.eq(method)]
        summary = frame.groupby("routes")["capture"].agg(["mean", "sem"]).reset_index()
        ax_f.plot(
            summary.routes,
            summary["mean"],
            color=color,
            marker="o",
            ms=3.3,
            lw=LW_DATA,
            label=label,
        )
        ax_f.fill_between(
            summary.routes,
            summary["mean"] - summary["sem"],
            summary["mean"] + summary["sem"],
            color=color,
            alpha=0.10,
            linewidth=0,
        )
    ax_f.set_xscale("log", base=2)
    ax_f.set_xticks([1, 2, 4, 8, 16], ["1", "2", "4", "8", "16"])
    ax_f.set_xlabel("selected inhibitory sites")
    ax_f.set_ylabel("ancestry energy captured")
    style_axis(ax_f, grid="both")

    fig.canvas.draw()
    audit_layout(fig, stem.name)
    audit_text_over_data(fig, stem.name)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=600)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--census-dir", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--digital-rows", type=Path, default=DEFAULT_DIGITAL)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    parser.add_argument("--max-mapping-distance-um", type=float, default=5.0)
    parser.add_argument("--matched-candidates", type=int, default=64)
    parser.add_argument("--connection-shuffles", type=int, default=500)
    parser.add_argument("--route-random", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the comprehensive journal figure from frozen source-data tables.",
    )
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.plot_only:
        cohort_output = pd.read_csv(args.outdir / "cohort_overlap.csv")
        contacts = pd.read_csv(args.outdir / "mapped_inhibitory_contacts.csv")
        targets = pd.read_csv(args.outdir / "target_metrics.csv")
        axon_metrics = pd.read_csv(args.outdir / "inhibitory_axon_metrics.csv")
        spatial_target_metrics = pd.read_csv(
            args.outdir / "inhibitory_3d_matched_target_metrics.csv"
        )
        routes = pd.read_csv(args.outdir / "route_capture_curves.csv")
        segments = pd.read_csv(args.outdir / "segment_metrics.csv")
        examples = []
        for m_type in ["L2a", "L2c", "L3a", "L4a", "L4c", "L5ET"]:
            ids = sorted(targets.loc[targets.m_type.eq(m_type), "post_soma_id"].astype(int))
            if not ids:
                continue
            example_id = ids[0]
            examples.append(
                {
                    "nodes": segments[segments.post_soma_id.eq(example_id)].rename(
                        columns={"x_um": "x", "y_um": "y"}
                    ),
                    "contacts": contacts[
                        contacts.post_soma_id.eq(example_id) & contacts.mapping_pass
                    ],
                    "m_type": m_type,
                    "cell_id": example_id,
                }
            )
        make_figure(
            cohort_output,
            contacts,
            targets,
            axon_metrics,
            spatial_target_metrics,
            routes,
            examples,
            args.figure_stem,
        )
        return

    cohort = pd.read_csv(args.cohort)
    cohort = cohort[cohort["replication_eligible"].astype(bool)].copy()
    cells = clean_columns(pd.read_csv(args.census_dir / "cell_types.csv"))
    overlap = cohort.merge(cells, left_on="nucleus_id", right_on="cell_id", how="inner", suffixes=("_v661", "_v795"), validate="one_to_one")
    if len(overlap) != 20:
        raise ValueError(f"expected frozen 20-cell inhibitory-census overlap, found {len(overlap)}")

    inhibitory_all = clean_columns(pd.read_csv(args.census_dir / "inhibitory_synapses_onto_column.csv"))
    coordinate_affine, coordinate_audit = fit_slanted_coordinate_transform(inhibitory_all)
    inhibitory = inhibitory_all[inhibitory_all["post_soma_id"].isin(overlap["nucleus_id"])].copy()
    known_inhibitory_ids = set(inhibitory["synapse_id"].astype("int64"))
    roots = set(overlap["pt_root_id"].astype("int64"))
    inputs = pd.read_feather(
        args.census_dir / "baseline_synapses_slanted_update_v795_post.feather",
        columns=["id", "post_pt_root_id", "ctr_pt_position", "size", "dist_to_root", "is_apical", "is_soma", "is_dendrite"],
    )
    inputs = inputs[inputs["post_pt_root_id"].isin(roots)].copy()
    if not known_inhibitory_ids.issubset(set(inputs["id"].astype("int64"))):
        raise ValueError("known inhibitory synapse IDs are not a subset of all-input census IDs")

    digital = pd.read_csv(args.digital_rows) if args.digital_rows.exists() else pd.DataFrame()
    digital_aggregate = aggregate_digital_twin(digital[digital.get("target_id", pd.Series(dtype=int)).isin(overlap["nucleus_id"])].copy()) if not digital.empty else pd.DataFrame(columns=["post_soma_id"])
    overlap["has_digital_twin"] = overlap["nucleus_id"].isin(digital_aggregate.get("post_soma_id", pd.Series(dtype=int)))

    segment_tables: list[pd.DataFrame] = []
    contact_tables: list[pd.DataFrame] = []
    connection_tables: list[pd.DataFrame] = []
    spatial_connection_tables: list[pd.DataFrame] = []
    target_rows: list[dict] = []
    route_tables: list[pd.DataFrame] = []
    example_payloads: list[dict] = []
    with zipfile.ZipFile(args.census_dir / "skeletons_swc.zip") as archive:
        for order, cell in overlap.sort_values(["m_type", "nucleus_id"]).reset_index(drop=True).iterrows():
            cell_id = int(cell["nucleus_id"])
            root_id = int(cell["pt_root_id"])
            print(f"[{order + 1}/{len(overlap)}] stable cell {cell_id} / v795 root {root_id}", flush=True)
            swc = read_swc(archive, root_id)
            segments, nodes = compress_dendritic_tree(swc)
            segments, parent, descendants = add_tree_annotations(segments)

            cell_inputs = inputs[inputs["post_pt_root_id"] == root_id].copy()
            input_positions = apply_affine_positions(
                np.vstack(cell_inputs["ctr_pt_position"].to_numpy()), coordinate_affine
            )
            mapped_inputs = map_positions(input_positions, cell_inputs, nodes, args.max_mapping_distance_um)
            mapped_inputs["compartment"] = input_compartment(mapped_inputs)
            mapped_inputs["x_um"] = input_positions[:, 0]
            mapped_inputs["y_um"] = input_positions[:, 1]
            mapped_inputs["z_um"] = input_positions[:, 2]

            cell_inhibitory = inhibitory[inhibitory["post_soma_id"] == cell_id].copy()
            inhibitory_positions = cell_inhibitory[["ctr_pt_position_um_x", "ctr_pt_position_um_y", "ctr_pt_position_um_z"]].to_numpy(dtype=float)
            mapped_inhibitory = map_positions(inhibitory_positions, cell_inhibitory, nodes, args.max_mapping_distance_um)
            mapped_inhibitory["compartment"] = input_compartment(mapped_inhibitory)
            mapped_inhibitory["x_um"] = inhibitory_positions[:, 0]
            mapped_inhibitory["y_um"] = inhibitory_positions[:, 1]
            mapped_inhibitory["z_um"] = inhibitory_positions[:, 2]

            segments = aggregate_segment_inputs(segments, mapped_inputs, mapped_inhibitory, descendants)
            segment_lookup = segments.set_index("segment_id")
            for field in ["path_length_um", "topological_depth", "major_branch_id", "descendant_input_count", "descendant_input_fraction", "descendant_input_size", "descendant_size_fraction", "route_specificity_bits"]:
                mapped_inhibitory[field] = mapped_inhibitory["segment_id"].map(segment_lookup[field])
            representatives = representative_clump_contacts(mapped_inhibitory)
            mapped_inhibitory["is_clump_representative"] = mapped_inhibitory["synapse_id"].isin(
                representatives["synapse_id"]
            )
            generic_inputs = mapped_inputs[~mapped_inputs["id"].isin(known_inhibitory_ids)].copy()
            candidates = build_candidate_segments(representatives, generic_inputs, args.matched_candidates)
            spatial_candidates = build_candidate_segments(
                representatives,
                generic_inputs,
                args.matched_candidates,
                spatially_matched=True,
            )
            seg_ids, distance_matrix, shared_matrix, route_overlap_matrix = distance_matrices(segments, parent)
            connections = analyze_connections(
                cell_id,
                representatives,
                candidates,
                seg_ids,
                distance_matrix,
                shared_matrix,
                route_overlap_matrix,
                args.connection_shuffles,
                args.seed,
            )
            spatial_connections = analyze_connections(
                cell_id,
                representatives,
                spatial_candidates,
                seg_ids,
                distance_matrix,
                shared_matrix,
                route_overlap_matrix,
                args.connection_shuffles,
                args.seed + 10_000_019,
            )
            if len(spatial_connections):
                spatial_connections["control"] = "joint path-distance and 3D matched"
            route = route_capture_curves(segments, parent, cell_id, args.route_random, args.seed)
            target_rows.append(summarize_target(cell, segments, mapped_inputs, mapped_inhibitory, representatives, candidates, connections))

            segments.insert(0, "post_soma_id", cell_id)
            segment_tables.append(segments)
            contact_tables.append(mapped_inhibitory)
            if len(connections):
                connection_tables.append(connections)
                spatial_connection_tables.append(spatial_connections)
            if len(route):
                route_tables.append(route)
            example_payloads.append(
                {
                    "nodes": nodes,
                    "contacts": mapped_inhibitory[mapped_inhibitory.mapping_pass],
                    "m_type": str(cell["m_type"]),
                    "cell_id": cell_id,
                }
            )

    segments_all = pd.concat(segment_tables, ignore_index=True)
    contacts_all = pd.concat(contact_tables, ignore_index=True)
    connections_all = pd.concat(connection_tables, ignore_index=True)
    spatial_connections_all = pd.concat(spatial_connection_tables, ignore_index=True)
    targets = pd.DataFrame(target_rows).merge(digital_aggregate, on="post_soma_id", how="left")
    routes = pd.concat(route_tables, ignore_index=True)
    cohort_output = overlap[["nucleus_id", "replication_root_id", "pt_root_id", "cell_type", "cell_type_manual", "m_type", "has_digital_twin"]].copy().rename(columns={"nucleus_id": "post_soma_id", "replication_root_id": "v661_root_id", "pt_root_id": "v795_root_id"})

    fixed = routes[routes.routes == 8].pivot(index="post_soma_id", columns="method", values="capture")
    tests = {
        "connection_tree_distance": paired_test(targets["connection_tree_distance_delta_um"].to_numpy(), "less"),
        "connection_shared_path": paired_test(targets["connection_shared_path_delta"].to_numpy(), "greater"),
        "connection_route_overlap": paired_test(targets["connection_route_overlap_delta"].to_numpy(), "greater"),
        "contact_descendant_fraction": paired_test(targets["contact_descendant_fraction_delta"].to_numpy(), "two-sided"),
        "contact_specificity": paired_test(targets["contact_specificity_delta_bits"].to_numpy(), "two-sided"),
        "dtc_minus_ptc_descendant_fraction": paired_test(
            targets["dtc_minus_ptc_median_descendant_fraction"].to_numpy(), "less"
        ),
    }
    axon_metrics = connections_all.groupby("pre_soma_id", as_index=False).agg(
        n_target_connections=("post_soma_id", "size"),
        n_targets=("post_soma_id", "nunique"),
        tree_distance_delta_um=("tree_distance_delta_um", "mean"),
        shared_path_delta=("shared_path_delta", "mean"),
        route_overlap_delta=("route_overlap_delta", "mean"),
    )
    tests["axon_clustered_connection_tree_distance"] = paired_test(
        axon_metrics["tree_distance_delta_um"].to_numpy(), "less"
    )
    tests["axon_clustered_connection_shared_path"] = paired_test(
        axon_metrics["shared_path_delta"].to_numpy(), "greater"
    )
    tests["axon_clustered_connection_route_overlap"] = paired_test(
        axon_metrics["route_overlap_delta"].to_numpy(), "greater"
    )
    spatial_target_metrics = spatial_connections_all.groupby("post_soma_id", as_index=False).agg(
        tree_distance_delta_um=("tree_distance_delta_um", "mean"),
        shared_path_delta=("shared_path_delta", "mean"),
        route_overlap_delta=("route_overlap_delta", "mean"),
    )
    tests["spatially_matched_connection_tree_distance"] = paired_test(
        spatial_target_metrics["tree_distance_delta_um"].to_numpy(), "less"
    )
    tests["spatially_matched_connection_shared_path"] = paired_test(
        spatial_target_metrics["shared_path_delta"].to_numpy(), "greater"
    )
    tests["spatially_matched_connection_route_overlap"] = paired_test(
        spatial_target_metrics["route_overlap_delta"].to_numpy(), "greater"
    )
    if len(fixed):
        for control in ["random actual routes", "depth bins", "depth-preserving ancestry shuffle"]:
            delta = fixed["morphology-selected actual routes"] - fixed[control]
            tests[f"eight_routes_selected_minus_{control}"] = paired_test(delta.to_numpy(), "greater")
    summary = {
        "claim_level": "one-mouse structural organization and route-capacity analysis",
        "cohort": {
            "n_independent_v661_targets": 47,
            "n_inhibitory_census_overlap": int(len(cohort_output)),
            "n_functionally_annotated_overlap": int(cohort_output.has_digital_twin.sum()),
            "n_all_input_locations": int(len(inputs)),
            "n_mapped_input_locations": int(targets.n_mapped_inputs.sum()),
            "n_known_column_inhibitory_contacts": int(len(inhibitory)),
            "n_mapped_known_inhibitory_contacts": int(targets.n_known_inhibitory_contacts.sum()),
            "n_known_inhibitory_cells": int(inhibitory.pre_soma_id.nunique()),
            "n_mapped_known_inhibitory_cells": int(
                contacts_all.loc[contacts_all.mapping_pass, "pre_soma_id"].nunique()
            ),
            "n_known_inhibitory_connections": int(inhibitory[["pre_soma_id", "post_soma_id"]].drop_duplicates().shape[0]),
            "n_connections_with_two_or_more_axonal_clumps": int(len(connections_all)),
        },
        "mapping": {
            "max_distance_um": float(args.max_mapping_distance_um),
            "all_input_pass_fraction": float(targets.input_mapping_fraction.mean()),
            "inhibitory_pass_fraction": float(targets.inhibitory_mapping_fraction.mean()),
            "coordinate_transform": coordinate_audit,
        },
        "tests": tests,
        "limitations": [
            "The MICrONS volume is one mouse; target cells are nested samples, not animal replicates.",
            "Known inhibitory contacts include only presynaptic inhibitory neurons whose somata are in the census column.",
            "Five of 20 targets changed root IDs between v661 and v795 and were joined by stable nucleus identity; unchanged segmentation is not assumed.",
            "The packaged census table does not provide a per-target proofreading-status field, so no proofreading claim is made.",
            "All-input locations are used as a generic eligibility proxy; unlabelled inputs are not asserted to be excitatory.",
            "Structural route organization and capacity do not show that the routes carried learning signals in vivo.",
        ],
    }

    cohort_output.to_csv(args.outdir / "cohort_overlap.csv", index=False)
    segments_all.to_csv(args.outdir / "segment_metrics.csv", index=False)
    contacts_all.to_csv(args.outdir / "mapped_inhibitory_contacts.csv", index=False)
    connections_all.to_csv(args.outdir / "inhibitory_connection_topology.csv", index=False)
    spatial_connections_all.to_csv(
        args.outdir / "inhibitory_connection_topology_3d_matched.csv", index=False
    )
    axon_metrics.to_csv(args.outdir / "inhibitory_axon_metrics.csv", index=False)
    spatial_target_metrics.to_csv(args.outdir / "inhibitory_3d_matched_target_metrics.csv", index=False)
    targets.to_csv(args.outdir / "target_metrics.csv", index=False)
    routes.to_csv(args.outdir / "route_capture_curves.csv", index=False)
    pd.DataFrame(
        coordinate_affine,
        index=["raw_x", "raw_y", "raw_z", "intercept"],
        columns=["slanted_x_um", "slanted_y_um", "slanted_z_um"],
    ).to_csv(args.outdir / "coordinate_transform.csv")
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# MICrONS inhibitory route analysis",
        "",
        "This analysis joins the independent reconstruction cohort to the published inhibitory census by stable cell ID.",
        "",
        f"- Targets: {len(cohort_output)} of 47; functionally annotated: {int(cohort_output.has_digital_twin.sum())}.",
        f"- Known column-interneuron contacts before mapping: {len(inhibitory):,} from {inhibitory.pre_soma_id.nunique()} inhibitory cells.",
        f"- Successfully mapped contacts: {int(targets.n_known_inhibitory_contacts.sum()):,} from {contacts_all.loc[contacts_all.mapping_pass, 'pre_soma_id'].nunique()} inhibitory cells.",
        f"- Multi-clump inhibitory connections tested: {len(connections_all)}.",
        f"- Target-level tree-distance delta: {tests['connection_tree_distance']['mean_delta']:.2f} um, one-sided Wilcoxon p={tests['connection_tree_distance']['wilcoxon_p']:.4g}.",
        f"- Target-level shared-path delta: {tests['connection_shared_path']['mean_delta']:.3f}, one-sided Wilcoxon p={tests['connection_shared_path']['wilcoxon_p']:.4g}.",
        f"- Target-level descendant-route overlap delta: {tests['connection_route_overlap']['mean_delta']:.3f}, one-sided Wilcoxon p={tests['connection_route_overlap']['wilcoxon_p']:.4g}.",
        f"- Presynaptic-axon-level shared-path sensitivity: p={tests['axon_clustered_connection_shared_path']['wilcoxon_p']:.4g}; route-overlap sensitivity: p={tests['axon_clustered_connection_route_overlap']['wilcoxon_p']:.4g}.",
        f"- Joint path-distance/3D-matched shared-path sensitivity: p={tests['spatially_matched_connection_shared_path']['wilcoxon_p']:.4g}; route-overlap sensitivity: p={tests['spatially_matched_connection_route_overlap']['wilcoxon_p']:.4g}.",
        f"- DTC minus PTC median descendant-domain fraction: {tests['dtc_minus_ptc_descendant_fraction']['mean_delta']:.3f}; ordering in {tests['dtc_minus_ptc_descendant_fraction']['ordering_negative']}/{tests['dtc_minus_ptc_descendant_fraction']['n_targets']} targets.",
        "",
        "Interpretation is restricted to structural organization and available route capacity in one reconstructed mouse.",
    ]
    (args.outdir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    # Prefer a layer-diverse display while keeping the complete cohort in source data.
    display_types = ["L2a", "L2c", "L3a", "L4a", "L4c", "L5ET"]
    examples = []
    for m_type in display_types:
        candidates = sorted(
            [value for value in example_payloads if value["m_type"] == m_type],
            key=lambda value: value["cell_id"],
        )
        if candidates:
            examples.append(candidates[0])
    make_figure(
        cohort_output,
        contacts_all,
        targets,
        axon_metrics,
        spatial_target_metrics,
        routes,
        examples,
        args.figure_stem,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Pilot: real MICrONS morphology as a hierarchical credit-routing substrate.

The analysis maps incoming synapses to precomputed MICrONS dendritic skeletons,
compresses each skeleton into branch-to-branch cable segments, and instantiates
the exact ancestry structure implied by the conductance-tree gradient theorem.

This is structural evidence, not evidence that the recorded mouse implemented
the modeled learning rule. The analysis asks three narrower questions:

1. Do real dendrites expose multiple nested, shunt-controllable credit domains?
2. Does real inhibitory placement reshape path/update-gain geometry beyond a
   path-distance-matched shuffle?
3. Can a morphology-aware sparse feedback dictionary compress the resulting
   credit perturbations more efficiently than scalar, depth-only, or random
   branch feedback?
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATADIR = PROJECT / "data" / "microns_morphology"
DEFAULT_OUTDIR = PROJECT / "results" / "microns_morphology_credit"
VOXEL_TO_UM = np.asarray([0.004, 0.004, 0.040], dtype=float)
COLORS = {
    "E": "#d95f5f",
    "I": "#4169a1",
    "morphology": "#26828e",
    "depth": "#7a5195",
    "random": "#999999",
    "pca": "#e69f00",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(keep):
        return float("nan")
    order = np.argsort(values[keep])
    x = values[keep][order]
    w = weights[keep][order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(np.interp(float(q), cdf, x))


def weighted_cosine_to_constant(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(keep):
        return float("nan")
    x = values[keep]
    w = weights[keep]
    scale = np.max(np.abs(x))
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    x = x / scale
    return float(np.sum(w * x) / np.sqrt(np.sum(w * x * x) * np.sum(w)))


def participation_rank(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    if matrix.size == 0 or min(matrix.shape) == 0:
        return float("nan"), np.asarray([], dtype=float)
    singular = np.linalg.svd(matrix, compute_uv=False)
    power = singular * singular
    denom = np.sum(power * power)
    rank = float(np.sum(power) ** 2 / denom) if denom > 0 else 0.0
    return rank, singular


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 10_000) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [float("nan"), float("nan")]
    samples = rng.choice(values, size=(int(n_boot), len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def cell_rng(seed: int, root_id: int, stream: int) -> np.random.Generator:
    """Return a stable cell-specific stream independent of loop order/settings."""

    root_id = int(root_id)
    entropy = [
        int(seed) & 0xFFFFFFFF,
        root_id & 0xFFFFFFFF,
        (root_id >> 32) & 0xFFFFFFFF,
        int(stream) & 0xFFFFFFFF,
    ]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def compress_dendritic_tree(swc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse degree-two dendritic chains into branch-to-branch segments."""

    frame = swc.copy()
    for col in ["id", "type", "parent"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["id", "type", "parent", "x", "y", "z", "radius"]).copy()
    frame[["id", "type", "parent"]] = frame[["id", "type", "parent"]].astype("int64")
    frame = frame.drop_duplicates("id", keep="first").set_index("id", drop=False)
    ids = {int(x) for x in frame.index}
    soma_candidates = frame.index[frame["type"] == 1].tolist()
    root_candidates = frame.index[frame["parent"] < 0].tolist()
    if soma_candidates:
        root = int(soma_candidates[0])
    elif root_candidates:
        root = int(root_candidates[0])
    else:
        raise ValueError("skeleton has no soma/root node")

    full_children: dict[int, list[int]] = defaultdict(list)
    for node, parent in frame[["id", "parent"]].itertuples(index=False):
        if int(parent) in ids:
            full_children[int(parent)].append(int(node))

    component: set[int] = set()
    stack = [root]
    while stack:
        node = int(stack.pop())
        if node in component:
            continue
        component.add(node)
        for child in full_children.get(node, []):
            if int(frame.loc[child, "type"]) in (1, 3):
                stack.append(child)
    if len(component) < 10:
        raise ValueError(f"dendritic component too small: {len(component)} nodes")

    children: dict[int, list[int]] = defaultdict(list)
    parent: dict[int, int] = {}
    for node in component:
        if node == root:
            continue
        par = int(frame.loc[node, "parent"])
        if par not in component:
            raise ValueError(f"dendritic node {node} disconnected from soma component")
        parent[node] = par
        children[par].append(node)

    retained = {root}
    retained.update(node for node in component if node != root and len(children.get(node, [])) != 1)
    node_to_segment: dict[int, int] = {root: root}
    records: list[dict[str, Any]] = [
        {
            "segment_id": root,
            "parent_segment_id": -1,
            "edge_length_um": 0.0,
            "mean_radius_um": float(frame.loc[root, "radius"]),
            "x_um": float(frame.loc[root, "x"]),
            "y_um": float(frame.loc[root, "y"]),
            "z_um": float(frame.loc[root, "z"]),
            "n_skeleton_nodes": 1,
        }
    ]

    coord = frame[["x", "y", "z"]].astype(float)
    for distal in sorted(retained - {root}):
        chain = [int(distal)]
        cursor = int(distal)
        while parent[cursor] not in retained:
            cursor = parent[cursor]
            chain.append(cursor)
        proximal = int(parent[cursor])
        lengths: list[float] = []
        radii: list[float] = []
        for node in chain:
            par = int(parent[node])
            delta = coord.loc[node].to_numpy() - coord.loc[par].to_numpy()
            length = float(np.linalg.norm(delta))
            if length > 0:
                lengths.append(length)
                radii.append(0.5 * (float(frame.loc[node, "radius"]) + float(frame.loc[par, "radius"])))
            node_to_segment[node] = int(distal)
        total_length = float(np.sum(lengths))
        mean_radius = (
            float(np.average(radii, weights=lengths))
            if lengths and total_length > 0
            else float(frame.loc[distal, "radius"])
        )
        records.append(
            {
                "segment_id": int(distal),
                "parent_segment_id": proximal,
                "edge_length_um": total_length,
                "mean_radius_um": mean_radius,
                "x_um": float(frame.loc[distal, "x"]),
                "y_um": float(frame.loc[distal, "y"]),
                "z_um": float(frame.loc[distal, "z"]),
                "n_skeleton_nodes": int(len(chain)),
            }
        )

    missing_nodes = component - set(node_to_segment)
    if missing_nodes:
        raise ValueError(f"{len(missing_nodes)} dendritic nodes lacked a compressed segment")

    segments = pd.DataFrame(records).set_index("segment_id", drop=False)
    seg_children: dict[int, list[int]] = defaultdict(list)
    for seg, par in segments[["segment_id", "parent_segment_id"]].itertuples(index=False):
        if int(par) >= 0:
            seg_children[int(par)].append(int(seg))
    depth = {root: 0}
    path_length = {root: 0.0}
    order = [root]
    for node in order:
        for child in seg_children.get(node, []):
            depth[child] = depth[node] + 1
            path_length[child] = path_length[node] + float(segments.loc[child, "edge_length_um"])
            order.append(child)
    segments["topological_depth"] = segments["segment_id"].map(depth).astype(int)
    segments["path_length_um"] = segments["segment_id"].map(path_length).astype(float)
    segments["n_children"] = segments["segment_id"].map(lambda x: len(seg_children.get(int(x), []))).astype(int)
    segments["is_leaf"] = segments["n_children"] == 0
    segments["is_branchpoint"] = segments["n_children"] > 1

    nodes = frame.loc[sorted(component), ["id", "x", "y", "z", "radius", "type"]].copy()
    nodes["segment_id"] = nodes["id"].map(node_to_segment).astype("int64")
    return segments.reset_index(drop=True), nodes.reset_index(drop=True)


def map_synapses_to_segments(
    synapses: pd.DataFrame,
    nodes: pd.DataFrame,
    max_distance_um: float,
) -> pd.DataFrame:
    position_vox = synapses[
        ["post_pt_position_x", "post_pt_position_y", "post_pt_position_z"]
    ].to_numpy(dtype=float)
    position_um = position_vox * VOXEL_TO_UM[None, :]
    tree = cKDTree(nodes[["x", "y", "z"]].to_numpy(dtype=float))
    distance, index = tree.query(position_um, k=1)
    out = synapses.copy()
    out["x_um"] = position_um[:, 0]
    out["y_um"] = position_um[:, 1]
    out["z_um"] = position_um[:, 2]
    out["nearest_dendrite_distance_um"] = distance
    out["segment_id"] = nodes.iloc[index]["segment_id"].to_numpy(dtype="int64")
    out["mapping_pass"] = np.isfinite(distance) & (distance <= float(max_distance_um))
    return out


def add_segment_synapses(segments: pd.DataFrame, mapped: pd.DataFrame) -> pd.DataFrame:
    passed = mapped[mapped["mapping_pass"] & mapped["synapse_class"].isin(["E", "I"])].copy()
    passed["size"] = pd.to_numeric(passed["size"], errors="coerce").fillna(0.0).clip(lower=0.0)
    aggregate = (
        passed.groupby(["segment_id", "synapse_class"])
        .agg(synapse_count=("id", "size"), synapse_size=("size", "sum"))
        .reset_index()
    )
    count = aggregate.pivot(index="segment_id", columns="synapse_class", values="synapse_count").fillna(0)
    size = aggregate.pivot(index="segment_id", columns="synapse_class", values="synapse_size").fillna(0.0)
    result = segments.copy().set_index("segment_id", drop=False)
    for cls in ["E", "I"]:
        result[f"{cls}_count"] = count.get(cls, pd.Series(dtype=float)).reindex(result.index).fillna(0).astype(int)
        result[f"{cls}_size"] = size.get(cls, pd.Series(dtype=float)).reindex(result.index).fillna(0.0).astype(float)
    return result.reset_index(drop=True)


def parent_map(segments: pd.DataFrame) -> tuple[int, dict[int, int], dict[int, list[int]]]:
    root_rows = segments[segments["parent_segment_id"] < 0]
    if len(root_rows) != 1:
        raise ValueError(f"expected one root segment, found {len(root_rows)}")
    root = int(root_rows.iloc[0]["segment_id"])
    parent: dict[int, int] = {}
    children: dict[int, list[int]] = defaultdict(list)
    for seg, par in segments[["segment_id", "parent_segment_id"]].itertuples(index=False):
        if int(par) >= 0:
            parent[int(seg)] = int(par)
            children[int(par)].append(int(seg))
    return root, parent, children


def ancestry_matrix(row_segments: list[int], column_segments: list[int], parent: dict[int, int]) -> np.ndarray:
    matrix = np.zeros((len(row_segments), len(column_segments)), dtype=float)
    lookup = {seg: j for j, seg in enumerate(column_segments)}
    for i, segment in enumerate(row_segments):
        cursor = int(segment)
        while True:
            if cursor in lookup:
                matrix[i, lookup[cursor]] = 1.0
            if cursor not in parent:
                break
            cursor = parent[cursor]
    return matrix


def electrical_geometry(
    segments: pd.DataFrame,
    e_scale: float,
    i_scale: float,
    i_override: np.ndarray | None = None,
) -> pd.DataFrame:
    result = segments.copy().set_index("segment_id", drop=False)
    nonroot = result["parent_segment_id"] >= 0
    raw_coupling = (
        result.loc[nonroot, "mean_radius_um"].to_numpy(dtype=float) ** 2
        / np.maximum(result.loc[nonroot, "edge_length_um"].to_numpy(dtype=float), 1e-3)
    )
    coupling_scale = float(np.median(raw_coupling[raw_coupling > 0])) if np.any(raw_coupling > 0) else 1.0
    result["g_edge"] = 0.0
    result.loc[nonroot, "g_edge"] = np.clip(raw_coupling / coupling_scale, 0.05, 20.0)

    raw_leak = result["mean_radius_um"].to_numpy(dtype=float) * np.maximum(
        result["edge_length_um"].to_numpy(dtype=float),
        result["mean_radius_um"].to_numpy(dtype=float),
    )
    leak_scale = float(np.median(raw_leak[raw_leak > 0])) if np.any(raw_leak > 0) else 1.0
    result["g_leak"] = np.clip(raw_leak / leak_scale, 0.1, 10.0)

    total_syn = result["E_size"].to_numpy(dtype=float) + result["I_size"].to_numpy(dtype=float)
    syn_scale = float(np.median(total_syn[total_syn > 0])) if np.any(total_syn > 0) else 1.0
    result["g_e"] = float(e_scale) * result["E_size"].to_numpy(dtype=float) / syn_scale
    i_size = result["I_size"].to_numpy(dtype=float) if i_override is None else np.asarray(i_override, dtype=float)
    result["g_i"] = float(i_scale) * i_size / syn_scale

    root, parent, children = parent_map(result.reset_index(drop=True))
    child_coupling = {
        seg: float(sum(result.loc[child, "g_edge"] for child in children.get(seg, [])))
        for seg in result.index
    }
    result["g_children"] = result["segment_id"].map(child_coupling).astype(float)
    result["g_total"] = result["g_leak"] + result["g_e"] + result["g_i"] + result["g_children"]

    log_alpha = {root: 0.0}
    ordered = result.sort_values("topological_depth")
    for segment in ordered["segment_id"]:
        segment = int(segment)
        if segment == root:
            continue
        par = parent[segment]
        factor = float(result.loc[segment, "g_edge"] / result.loc[par, "g_total"])
        log_alpha[segment] = log_alpha[par] + math.log(max(factor, 1e-300))
    result["log_path_gain"] = result["segment_id"].map(log_alpha).astype(float)
    result["log_update_gain"] = result["log_path_gain"] - np.log(result["g_total"].clip(lower=1e-300))
    return result.reset_index(drop=True)


def depth_matched_i_shuffle(segments: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    result = segments.copy()
    candidate = (result["E_size"] + result["I_size"]) > 0
    values = result["I_size"].to_numpy(dtype=float).copy()
    distance = result["path_length_um"].to_numpy(dtype=float)
    valid_distance = distance[candidate.to_numpy()]
    if len(np.unique(valid_distance)) < 4:
        bins = result["topological_depth"].to_numpy(dtype=int)
    else:
        edges = np.unique(np.quantile(valid_distance, [0.0, 0.25, 0.5, 0.75, 1.0]))
        bins = np.digitize(distance, edges[1:-1], right=True)
    shuffled = values.copy()
    for group in np.unique(bins[candidate.to_numpy()]):
        indices = np.flatnonzero(candidate.to_numpy() & (bins == group))
        shuffled[indices] = rng.permutation(values[indices])
    return shuffled


def reconstruction_curves(
    kernel: np.ndarray,
    e_weights: np.ndarray,
    e_depth: np.ndarray,
    leverage: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
    n_random: int,
) -> list[dict[str, Any]]:
    n_rows, n_columns = kernel.shape
    if n_rows < 2 or n_columns < 1:
        return []
    perturbation = rng.standard_normal((int(n_samples), n_columns))
    target = perturbation @ kernel.T
    sqrt_w = np.sqrt(np.clip(np.asarray(e_weights, dtype=float), 0.0, None))
    sqrt_w /= np.sqrt(np.mean(sqrt_w * sqrt_w)) if np.mean(sqrt_w * sqrt_w) > 0 else 1.0
    weighted_target = target * sqrt_w[None, :]
    denominator = float(np.linalg.norm(weighted_target))
    if denominator <= 0:
        return []

    _, singular, _ = np.linalg.svd(weighted_target, full_matrices=False)
    singular_power = singular * singular
    counts = [1, 2, 4, 8, 16]
    counts = [count for count in counts if count <= min(n_rows, n_columns) and count > 0]
    order = np.argsort(-np.asarray(leverage, dtype=float))
    curves: list[dict[str, Any]] = []

    def residual_for_dictionary(dictionary: np.ndarray) -> float:
        weighted_dictionary = dictionary * sqrt_w[:, None]
        coefficient, *_ = np.linalg.lstsq(weighted_dictionary, weighted_target.T, rcond=None)
        reconstruction = (weighted_dictionary @ coefficient).T
        return float(np.linalg.norm(weighted_target - reconstruction) / denominator)

    for count in counts:
        pca_residual = math.sqrt(max(0.0, 1.0 - float(np.sum(singular_power[:count]) / np.sum(singular_power))))
        curves.append(
            {
                "channels": count,
                "method": "dense PCA oracle",
                "residual": pca_residual,
                "wiring_nonzeros": int(n_rows * count),
            }
        )

        selected = order[:count]
        morph_dictionary = kernel[:, selected]
        curves.append(
            {
                "channels": count,
                "method": "morphology-aware paths",
                "residual": residual_for_dictionary(morph_dictionary),
                "wiring_nonzeros": int(np.count_nonzero(morph_dictionary)),
            }
        )

        random_residuals: list[float] = []
        random_nonzeros: list[int] = []
        for _ in range(int(n_random)):
            choice = rng.choice(n_columns, size=count, replace=False)
            random_dictionary = kernel[:, choice]
            random_residuals.append(residual_for_dictionary(random_dictionary))
            random_nonzeros.append(int(np.count_nonzero(random_dictionary)))
        curves.append(
            {
                "channels": count,
                "method": "random paths",
                "residual": float(np.mean(random_residuals)),
                "residual_sd": float(np.std(random_residuals, ddof=1)) if len(random_residuals) > 1 else 0.0,
                "wiring_nonzeros": float(np.mean(random_nonzeros)),
            }
        )

        shuffled_ancestry_residuals: list[float] = []
        for _ in range(int(n_random)):
            shuffled_dictionary = morph_dictionary.copy()
            for column in range(shuffled_dictionary.shape[1]):
                shuffled_dictionary[:, column] = rng.permutation(
                    shuffled_dictionary[:, column]
                )
            shuffled_ancestry_residuals.append(
                residual_for_dictionary(shuffled_dictionary)
            )
        curves.append(
            {
                "channels": count,
                "method": "shuffled ancestry",
                "residual": float(np.mean(shuffled_ancestry_residuals)),
                "residual_sd": (
                    float(np.std(shuffled_ancestry_residuals, ddof=1))
                    if len(shuffled_ancestry_residuals) > 1
                    else 0.0
                ),
                "wiring_nonzeros": int(np.count_nonzero(morph_dictionary)),
            }
        )

        if count == 1:
            depth_dictionary = np.ones((n_rows, 1), dtype=float)
        else:
            edges = np.unique(np.quantile(e_depth, np.linspace(0.0, 1.0, count + 1)))
            labels = np.digitize(e_depth, edges[1:-1], right=True)
            n_bins = int(labels.max()) + 1
            depth_dictionary = np.eye(n_bins, dtype=float)[labels]
        curves.append(
            {
                "channels": count,
                "method": "depth-only bins",
                "residual": residual_for_dictionary(depth_dictionary),
                "actual_channels": int(depth_dictionary.shape[1]),
                "wiring_nonzeros": int(np.count_nonzero(depth_dictionary)),
            }
        )
    return curves


def analyze_cell(
    root_id: int,
    datadir: Path,
    max_mapping_distance_um: float,
    e_scale: float,
    i_scale: float,
    n_shuffles: int,
    classification_mode: str,
    min_target_probability: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    # Keep null shuffles and perturbation reconstruction on separate streams.
    # This makes compression results invariant to --n-shuffles and manifest order.
    null_rng = cell_rng(seed, root_id, stream=1)
    compression_rng = cell_rng(seed, root_id, stream=2)
    swc = pd.read_csv(datadir / f"skeleton_{root_id}.csv.gz")
    synapses = pd.read_csv(datadir / f"synapses_{root_id}.csv.gz")
    target_confident = (
        pd.to_numeric(synapses["target_tag_probability"], errors="coerce").fillna(0.0)
        >= float(min_target_probability)
    )
    if classification_mode == "typed_only":
        synapses["synapse_class"] = synapses["typed_class"]
        synapses["class_source"] = "presynaptic_cell_type"
    elif classification_mode == "target_proxy_only":
        synapses["synapse_class"] = np.where(
            target_confident, synapses["target_proxy_class"], "?"
        )
        synapses["class_source"] = "target_proxy"
    elif classification_mode == "hybrid":
        typed_available = synapses["typed_class"].isin(["E", "I"])
        synapses["synapse_class"] = np.where(
            typed_available,
            synapses["typed_class"],
            np.where(target_confident, synapses["target_proxy_class"], "?"),
        )
        synapses["class_source"] = np.where(
            typed_available, "presynaptic_cell_type", "target_proxy"
        )
    else:
        raise ValueError(f"unknown classification mode: {classification_mode}")
    segments, nodes = compress_dendritic_tree(swc)
    mapped = map_synapses_to_segments(synapses, nodes, max_mapping_distance_um)
    segments = add_segment_synapses(segments, mapped)
    _, parent, _ = parent_map(segments)

    electrical_no_i = electrical_geometry(segments, e_scale=e_scale, i_scale=0.0)
    electrical = electrical_geometry(segments, e_scale=e_scale, i_scale=i_scale)
    lookup_no_i = electrical_no_i.set_index("segment_id")
    lookup = electrical.set_index("segment_id")

    e_segments = [int(x) for x in segments.loc[segments["E_size"] > 0, "segment_id"]]
    i_segments = [int(x) for x in segments.loc[segments["I_size"] > 0, "segment_id"]]
    e_lookup = segments.set_index("segment_id").loc[e_segments]
    i_lookup = segments.set_index("segment_id").loc[i_segments]
    ancestry = ancestry_matrix(e_segments, i_segments, parent)
    beta = (
        lookup.loc[i_segments, "g_i"].to_numpy(dtype=float)
        / lookup.loc[i_segments, "g_total"].to_numpy(dtype=float)
    )
    kernel = ancestry * beta[None, :]
    e_weight = e_lookup["E_size"].to_numpy(dtype=float)
    weighted_kernel = np.sqrt(e_weight / max(np.sum(e_weight), 1e-12))[:, None] * kernel
    kernel_rank, singular = participation_rank(weighted_kernel)

    descendant_e = ancestry.T @ e_weight if len(e_segments) and len(i_segments) else np.asarray([], dtype=float)
    domain_fraction = descendant_e / max(float(np.sum(e_weight)), 1e-12)
    i_weight = i_lookup["I_size"].to_numpy(dtype=float) if len(i_segments) else np.asarray([], dtype=float)
    weighted_domain = (
        float(np.average(domain_fraction, weights=i_weight))
        if len(i_weight) and np.sum(i_weight) > 0
        else float("nan")
    )

    shuffled_domains: list[float] = []
    shuffled_cosines: list[float] = []
    for _ in range(int(n_shuffles)):
        shuffled_i = depth_matched_i_shuffle(segments, null_rng)
        shunted = electrical_geometry(segments, e_scale=e_scale, i_scale=i_scale, i_override=shuffled_i)
        shunted_lookup = shunted.set_index("segment_id")
        shuffled_cosines.append(
            weighted_cosine_to_constant(
                np.exp(
                    shunted_lookup.loc[e_segments, "log_update_gain"].to_numpy(dtype=float)
                    - np.max(shunted_lookup.loc[e_segments, "log_update_gain"].to_numpy(dtype=float))
                ),
                e_weight,
            )
        )
        nonzero = np.flatnonzero(shuffled_i > 0)
        if len(nonzero):
            shuffled_sites = [int(segments.iloc[index]["segment_id"]) for index in nonzero]
            shuffled_ancestry = ancestry_matrix(e_segments, shuffled_sites, parent)
            shuffled_fraction = (shuffled_ancestry.T @ e_weight) / max(float(np.sum(e_weight)), 1e-12)
            shuffled_domains.append(float(np.average(shuffled_fraction, weights=shuffled_i[nonzero])))

    no_i_gain = lookup_no_i.loc[e_segments, "log_update_gain"].to_numpy(dtype=float)
    actual_gain = lookup.loc[e_segments, "log_update_gain"].to_numpy(dtype=float)
    cosine_no_i = weighted_cosine_to_constant(np.exp(no_i_gain - np.max(no_i_gain)), e_weight)
    cosine_actual = weighted_cosine_to_constant(np.exp(actual_gain - np.max(actual_gain)), e_weight)

    passed = mapped[mapped["mapping_pass"] & mapped["synapse_class"].isin(["E", "I"])].copy()
    segment_meta = segments.set_index("segment_id")[["path_length_um", "topological_depth"]]
    passed = passed.join(segment_meta, on="segment_id")
    e_syn = passed[passed["synapse_class"] == "E"]
    i_syn = passed[passed["synapse_class"] == "I"]
    typed = passed[
        passed["typed_class"].isin(["E", "I"])
        & passed["target_proxy_class"].isin(["E", "I"])
    ]
    concordance = float((typed["typed_class"] == typed["target_proxy_class"]).mean()) if len(typed) else float("nan")

    leverage = beta * domain_fraction if len(beta) else np.asarray([], dtype=float)
    curve_rows = reconstruction_curves(
        kernel,
        e_weight,
        e_lookup["path_length_um"].to_numpy(dtype=float),
        leverage,
        compression_rng,
        n_samples=384,
        n_random=64,
    )
    for row in curve_rows:
        row["root_id"] = int(root_id)

    segment_output = segments.copy()
    segment_output["root_id"] = int(root_id)
    segment_output["g_total"] = segment_output["segment_id"].map(lookup["g_total"])
    segment_output["g_i"] = segment_output["segment_id"].map(lookup["g_i"])
    segment_output["log_path_gain"] = segment_output["segment_id"].map(lookup["log_path_gain"])
    segment_output["log_update_gain"] = segment_output["segment_id"].map(lookup["log_update_gain"])
    domain_map = dict(zip(i_segments, domain_fraction))
    segment_output["credit_domain_fraction"] = segment_output["segment_id"].map(domain_map)

    metrics = {
        "root_id": int(root_id),
        "n_skeleton_nodes": int(len(swc)),
        "n_dendrite_nodes": int(len(nodes)),
        "n_segments": int(len(segments)),
        "n_branchpoints": int(segments["is_branchpoint"].sum()),
        "n_leaves": int(segments["is_leaf"].sum()),
        "max_topological_depth": int(segments["topological_depth"].max()),
        "max_path_length_um": float(segments["path_length_um"].max()),
        "total_cable_um": float(segments["edge_length_um"].sum()),
        "n_synapses": int(len(mapped)),
        "mapping_pass_fraction": float(mapped["mapping_pass"].mean()),
        "mapping_distance_median_um": float(mapped["nearest_dendrite_distance_um"].median()),
        "n_e_synapses": int(len(e_syn)),
        "n_i_synapses": int(len(i_syn)),
        "typed_proxy_validation_n": int(len(typed)),
        "typed_proxy_concordance": concordance,
        "e_path_distance_median_um": weighted_quantile(
            e_syn["path_length_um"].to_numpy(dtype=float),
            e_syn["size"].to_numpy(dtype=float),
            0.5,
        ),
        "i_path_distance_median_um": weighted_quantile(
            i_syn["path_length_um"].to_numpy(dtype=float),
            i_syn["size"].to_numpy(dtype=float),
            0.5,
        ),
        "e_topological_depth_median": weighted_quantile(
            e_syn["topological_depth"].to_numpy(dtype=float),
            e_syn["size"].to_numpy(dtype=float),
            0.5,
        ),
        "i_topological_depth_median": weighted_quantile(
            i_syn["topological_depth"].to_numpy(dtype=float),
            i_syn["size"].to_numpy(dtype=float),
            0.5,
        ),
        "n_e_segments": int(len(e_segments)),
        "n_i_segments": int(len(i_segments)),
        "credit_kernel_participation_rank": kernel_rank,
        "credit_kernel_rank_fraction": float(kernel_rank / max(1, min(len(e_segments), len(i_segments)))),
        "credit_domain_weighted_mean": weighted_domain,
        "credit_domain_median": float(np.median(domain_fraction)) if len(domain_fraction) else float("nan"),
        "credit_domain_p10": float(np.quantile(domain_fraction, 0.1)) if len(domain_fraction) else float("nan"),
        "credit_domain_p90": float(np.quantile(domain_fraction, 0.9)) if len(domain_fraction) else float("nan"),
        "credit_domain_depthmatched_null_mean": float(np.mean(shuffled_domains)) if shuffled_domains else float("nan"),
        "credit_domain_depthmatched_null_sd": float(np.std(shuffled_domains, ddof=1)) if len(shuffled_domains) > 1 else float("nan"),
        "credit_domain_depthmatched_z": (
            float((weighted_domain - np.mean(shuffled_domains)) / np.std(shuffled_domains, ddof=1))
            if len(shuffled_domains) > 1 and np.std(shuffled_domains, ddof=1) > 0
            else float("nan")
        ),
        "scalar_broadcast_cosine_no_inhibition": cosine_no_i,
        "scalar_broadcast_cosine_actual_inhibition": cosine_actual,
        "scalar_broadcast_cosine_change": cosine_actual - cosine_no_i,
        "scalar_broadcast_cosine_depthmatched_null_mean": float(np.mean(shuffled_cosines)),
        "scalar_broadcast_cosine_actual_minus_null": cosine_actual - float(np.mean(shuffled_cosines)),
        "log_update_gain_sd_no_inhibition": float(np.sqrt(np.average((no_i_gain - np.average(no_i_gain, weights=e_weight)) ** 2, weights=e_weight))),
        "log_update_gain_sd_actual_inhibition": float(np.sqrt(np.average((actual_gain - np.average(actual_gain, weights=e_weight)) ** 2, weights=e_weight))),
        "kernel_singular_value_1_fraction": (
            float(singular[0] ** 2 / np.sum(singular**2)) if len(singular) and np.sum(singular**2) > 0 else float("nan")
        ),
    }
    return metrics, segment_output, passed, curve_rows


def paired_test(frame: pd.DataFrame, first: str, second: str) -> dict[str, Any]:
    pair = frame[[first, second]].dropna()
    difference = pair[first].to_numpy(dtype=float) - pair[second].to_numpy(dtype=float)
    if len(pair) < 2 or np.allclose(difference, 0):
        statistic, pvalue = float("nan"), float("nan")
    else:
        result = stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
        statistic, pvalue = float(result.statistic), float(result.pvalue)
    return {
        "n_cells": int(len(pair)),
        "mean_difference": float(np.mean(difference)) if len(difference) else float("nan"),
        "median_difference": float(np.median(difference)) if len(difference) else float("nan"),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": pvalue,
    }


def make_figure(
    cell_metrics: pd.DataFrame,
    segments: pd.DataFrame,
    synapses: pd.DataFrame,
    curves: pd.DataFrame,
    outdir: Path,
) -> None:
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.2))
    ax = axes[0, 0]
    example_root = int(cell_metrics.iloc[0]["root_id"])
    example_segments = segments[segments["root_id"] == example_root].set_index("segment_id")
    for _, row in example_segments.iterrows():
        if int(row["parent_segment_id"]) < 0:
            continue
        parent = example_segments.loc[int(row["parent_segment_id"])]
        ax.plot([parent["x_um"], row["x_um"]], [parent["y_um"], row["y_um"]], color="#666666", lw=0.7, alpha=0.7)
    ex_syn = synapses[synapses["root_id"] == example_root]
    for cls in ["E", "I"]:
        part = ex_syn[ex_syn["synapse_class"] == cls]
        ax.scatter(part["x_um"], part["y_um"], s=4 if cls == "E" else 7, c=COLORS[cls], alpha=0.35, label=cls)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("A  Real morphology and mapped inputs")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.legend(frameon=False, markerscale=2)

    ax = axes[0, 1]
    x = np.arange(len(cell_metrics))
    ax.plot(x, cell_metrics["e_path_distance_median_um"], "o-", color=COLORS["E"], label="E proxy")
    ax.plot(x, cell_metrics["i_path_distance_median_um"], "o-", color=COLORS["I"], label="I proxy")
    ax.set_xticks(x, cell_metrics["cell_type"].fillna("?").astype(str), rotation=45, ha="right")
    ax.set_ylabel("size-weighted median path distance (µm)")
    ax.set_title("B  E and I occupy similar cable depths")
    ax.legend(frameon=False)

    ax = axes[0, 2]
    domain_rows = segments.dropna(subset=["credit_domain_fraction"])
    for i, (_, group) in enumerate(domain_rows.groupby("root_id")):
        y = group["credit_domain_fraction"].to_numpy(dtype=float)
        ax.scatter(np.full_like(y, i, dtype=float) + np.random.default_rng(i).normal(0, 0.04, len(y)), y, s=8, alpha=0.35, color=COLORS["morphology"])
    ax.set_yscale("log")
    ax.set_ylabel("fraction of E input in gated subtree")
    ax.set_title("C  Inhibition spans nested credit domains")

    ax = axes[1, 0]
    ax.scatter(cell_metrics["max_topological_depth"], cell_metrics["credit_kernel_participation_rank"], s=45, c=cell_metrics["n_segments"], cmap="viridis")
    for _, row in cell_metrics.iterrows():
        ax.annotate(str(row["cell_type"]), (row["max_topological_depth"], row["credit_kernel_participation_rank"]), xytext=(3, 2), textcoords="offset points", fontsize=6)
    ax.axhline(1.0, ls="--", lw=1, color="#555555", label="point unit")
    ax.set_xlabel("maximum topological depth")
    ax.set_ylabel("credit-kernel participation rank")
    ax.set_title("D  Morphology supplies multiple control modes")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    before = cell_metrics["scalar_broadcast_cosine_no_inhibition"].to_numpy(dtype=float)
    after = cell_metrics["scalar_broadcast_cosine_actual_inhibition"].to_numpy(dtype=float)
    for i in range(len(cell_metrics)):
        ax.plot([0, 1], [before[i], after[i]], "-o", color="#555555", alpha=0.7, ms=4)
    ax.set_xticks([0, 1], ["no I", "actual I"])
    ax.set_ylabel("scalar-broadcast cosine")
    ax.set_title("E  Shunting does not rescue scalar feedback")

    ax = axes[1, 2]
    if not curves.empty:
        summary = curves.groupby(["channels", "method"], as_index=False)["residual"].agg(["mean", "sem"]).reset_index()
        styles = {
            "dense PCA oracle": (COLORS["pca"], "--"),
            "morphology-aware paths": (COLORS["morphology"], "-"),
            "depth-only bins": (COLORS["depth"], "-."),
            "random paths": (COLORS["random"], ":"),
            "shuffled ancestry": ("#cc79a7", (0, (3, 1, 1, 1))),
        }
        for method, part in summary.groupby("method"):
            color, linestyle = styles.get(method, ("black", "-"))
            ax.errorbar(part["channels"], part["mean"], yerr=part["sem"].fillna(0), marker="o", ms=3, color=color, ls=linestyle, label=method)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("feedback channels")
        ax.set_ylabel("weighted reconstruction residual")
        ax.legend(frameon=False, fontsize=6)
    ax.set_title("F  Sparse morphology-aware feedback")

    fig.suptitle("MICrONS topology pilot: dendritic morphology defines hierarchical credit routing", fontsize=12, weight="bold")
    fig.tight_layout()
    fig.savefig(outdir / "microns_morphology_credit_pilot.png", dpi=260, bbox_inches="tight")
    fig.savefig(outdir / "microns_morphology_credit_pilot.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=Path, default=DEFAULT_DATADIR)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--max-mapping-distance-um", type=float, default=5.0)
    parser.add_argument("--e-scale", type=float, default=0.5)
    parser.add_argument("--i-scale", type=float, default=0.5)
    parser.add_argument("--n-shuffles", type=int, default=500)
    parser.add_argument(
        "--classification-mode",
        choices=["hybrid", "typed_only", "target_proxy_only"],
        default="hybrid",
    )
    parser.add_argument("--min-target-probability", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.datadir / "cell_manifest.csv")
    manifest = manifest[manifest["status"].isin(["fetched", "cached"])].copy()
    if manifest.empty:
        raise ValueError("no completed cells in cell_manifest.csv")

    metrics_rows: list[dict[str, Any]] = []
    segment_frames: list[pd.DataFrame] = []
    synapse_frames: list[pd.DataFrame] = []
    curve_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for _, cell in manifest.iterrows():
        root_id = int(cell["root_id"])
        print(f"Analyzing MICrONS root {root_id}", flush=True)
        try:
            metrics, segments, synapses, curves = analyze_cell(
                root_id,
                args.datadir,
                args.max_mapping_distance_um,
                args.e_scale,
                args.i_scale,
                args.n_shuffles,
                args.classification_mode,
                args.min_target_probability,
                args.seed,
            )
        except Exception as exc:
            errors.append({"root_id": root_id, "error_type": type(exc).__name__, "error": str(exc)})
            continue
        for col in ["cell_type", "functional_area", "has_coregistration", "has_digital_twin"]:
            metrics[col] = cell.get(col)
        metrics_rows.append(metrics)
        segment_frames.append(segments)
        synapses["root_id"] = root_id
        synapse_frames.append(synapses)
        curve_rows.extend(curves)

    if not metrics_rows:
        raise RuntimeError(f"all cells failed: {errors}")
    cell_metrics = pd.DataFrame(metrics_rows)
    segment_metrics = pd.concat(segment_frames, ignore_index=True)
    mapped_synapses = pd.concat(synapse_frames, ignore_index=True)
    curves = pd.DataFrame(curve_rows)

    cell_metrics.to_csv(args.outdir / "cell_metrics.csv", index=False)
    segment_metrics.to_csv(args.outdir / "segment_metrics.csv", index=False)
    mapped_synapses.to_csv(args.outdir / "mapped_synapses.csv.gz", index=False, compression="gzip")
    curves.to_csv(args.outdir / "feedback_compression_curves.csv", index=False)
    make_figure(cell_metrics, segment_metrics, mapped_synapses, curves, args.outdir)

    proxy_weight = cell_metrics["typed_proxy_validation_n"].to_numpy(dtype=float)
    pooled_proxy = (
        float(np.average(cell_metrics["typed_proxy_concordance"], weights=proxy_weight))
        if np.sum(proxy_weight) > 0
        else float("nan")
    )
    depth_rank = stats.spearmanr(
        cell_metrics["max_topological_depth"],
        cell_metrics["credit_kernel_participation_rank"],
    )
    rng_summary = np.random.default_rng(args.seed + 1)
    cosine_change = cell_metrics["scalar_broadcast_cosine_change"].to_numpy(dtype=float)
    compression_tests: dict[str, Any] = {}
    if not curves.empty:
        pivot = curves.pivot_table(
            index=["root_id", "channels"], columns="method", values="residual"
        ).reset_index()
        for channels in [1, 2, 4, 8, 16]:
            subset = pivot[pivot["channels"] == channels]
            if subset.empty:
                continue
            morphology_rows = curves[
                (curves["channels"] == channels)
                & (curves["method"] == "morphology-aware paths")
            ].merge(
                cell_metrics[["root_id", "n_e_segments"]],
                on="root_id",
                how="left",
            )
            wiring_density = (
                morphology_rows["wiring_nonzeros"]
                / (morphology_rows["n_e_segments"] * channels)
            )
            compression_tests[str(channels)] = {
                "morphology_vs_random": paired_test(
                    subset,
                    "morphology-aware paths",
                    "random paths",
                ),
                "morphology_vs_depth": paired_test(
                    subset,
                    "morphology-aware paths",
                    "depth-only bins",
                ),
                "morphology_vs_shuffled_ancestry": paired_test(
                    subset,
                    "morphology-aware paths",
                    "shuffled ancestry",
                ),
                "mean_residuals": {
                    method: float(subset[method].mean())
                    for method in [
                        "dense PCA oracle",
                        "morphology-aware paths",
                        "random paths",
                        "shuffled ancestry",
                        "depth-only bins",
                    ]
                    if method in subset
                },
                "morphology_wiring_density_mean": float(wiring_density.mean()),
                "morphology_wiring_sparsity_mean": float(1.0 - wiring_density.mean()),
            }
    summary = {
        "status": "pilot_complete",
        "claim_level": (
            "real MICrONS morphology and synapse-location pilot; structural credit-routing evidence, not an in-vivo learning-rule test"
        ),
        "n_cells": int(len(cell_metrics)),
        "n_errors": int(len(errors)),
        "errors": errors,
        "n_total_mapped_synapses": int(
            (
                mapped_synapses["mapping_pass"]
                & mapped_synapses["synapse_class"].isin(["E", "I"])
            ).sum()
        ),
        "mapping_pass_fraction_mean": float(cell_metrics["mapping_pass_fraction"].mean()),
        "typed_proxy_validation_n": int(cell_metrics["typed_proxy_validation_n"].sum()),
        "typed_proxy_concordance_pooled": pooled_proxy,
        "e_vs_i_path_distance": paired_test(
            cell_metrics,
            "i_path_distance_median_um",
            "e_path_distance_median_um",
        ),
        "shunting_scalar_broadcast_change": {
            **paired_test(
                cell_metrics,
                "scalar_broadcast_cosine_actual_inhibition",
                "scalar_broadcast_cosine_no_inhibition",
            ),
            "mean_change_ci95_bootstrap": bootstrap_mean_ci(cosine_change, rng_summary),
        },
        "depth_vs_credit_kernel_rank": {
            "spearman_r": float(depth_rank.statistic),
            "spearman_p": float(depth_rank.pvalue),
        },
        "credit_kernel_rank": {
            "mean": float(cell_metrics["credit_kernel_participation_rank"].mean()),
            "range": [
                float(cell_metrics["credit_kernel_participation_rank"].min()),
                float(cell_metrics["credit_kernel_participation_rank"].max()),
            ],
            "point_neuron_reference": 1.0,
        },
        "credit_domain_fraction": {
            "median_across_inhibitory_segments": float(segment_metrics["credit_domain_fraction"].median()),
            "p10": float(segment_metrics["credit_domain_fraction"].quantile(0.1)),
            "p90": float(segment_metrics["credit_domain_fraction"].quantile(0.9)),
        },
        "feedback_compression": compression_tests,
        "classification": {
            "mode": args.classification_mode,
            "minimum_target_probability": float(args.min_target_probability),
        },
        "electrical_calibration": {
            "description": (
                "passive cable proxy: axial coupling proportional to radius^2/length; leak proportional to radius*length; synaptic area normalized within cell"
            ),
            "e_scale": float(args.e_scale),
            "i_scale": float(args.i_scale),
            "scope": "sensitivity-analysis parameterization, not a fitted biophysical model",
        },
        "important_boundaries": [
            "The target-structure prediction table is marked in development by MICrONS.",
            "Spine/shaft/soma labels are an E/I proxy when presynaptic cell type is unavailable.",
            "The electrical scale is normalized rather than fitted to electrophysiology.",
            "MICrONS is a single-mouse resource and these cells were selected from an existing matched cohort.",
            "Functional learning evidence requires trial-level or longitudinal error/plasticity measurements beyond this structural pilot.",
        ],
        "outputs": {
            "cell_metrics": str(args.outdir / "cell_metrics.csv"),
            "segment_metrics": str(args.outdir / "segment_metrics.csv"),
            "feedback_curves": str(args.outdir / "feedback_compression_curves.csv"),
            "figure": str(args.outdir / "microns_morphology_credit_pilot.png"),
        },
    }
    write_json(args.outdir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Prepare the frozen Pinky-v185 cohort for the common routing analysis.

The script extracts only the prospectively selected fixed meshes, skeletonizes
them with the published meshparty TEASAR implementation, ray-estimates cable
caliber, and converts the dense synapse table to the column contract used by
the minnie65 analysis. Raw public inputs and bulky per-cell derivatives remain
under ignored ``journal/data``; a checksum manifest is written to source data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW = ROOT / "data" / "pinky_v185"
DEFAULT_PREPARED = DEFAULT_RAW / "prepared"
DEFAULT_COHORT = ROOT / "source_data" / "pinky_v185_replication" / "cohort_manifest.csv"
DEFAULT_MANIFEST = ROOT / "source_data" / "pinky_v185_replication" / "preparation_manifest.json"
VOXEL_NM = np.asarray([4.0, 4.0, 40.0], dtype=float)
ZENODO_INPUTS = {
    "layer23_v185.tar.gz": (10_983_643_501, "c120366230a2d4ce94f213f60353e0f1"),
    "pni_synapses_v185.csv": (388_210_532, "f2382e0606ca72b46062d4c56b15351b"),
    "soma_valence_v185.csv": (30_881, "a8ce8aa4e5cdf4202caa5ae10411c333"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_zenodo_input(path: Path) -> dict[str, Any]:
    expected_size, expected_md5 = ZENODO_INPUTS[path.name]
    observed_size = path.stat().st_size
    observed_md5 = md5(path)
    if observed_size != expected_size or observed_md5 != expected_md5:
        raise ValueError(
            f"Zenodo checksum mismatch for {path}: size {observed_size} "
            f"(expected {expected_size}), md5 {observed_md5} "
            f"(expected {expected_md5})"
        )
    return {
        "size_bytes": observed_size,
        "zenodo_md5": observed_md5,
        "sha256": sha256(path),
    }


def _member_root_id(name: str, selected: set[int]) -> int | None:
    basename = Path(name).name
    # The public archive contains AppleDouble metadata entries named
    # ``._<root>.h5`` before the actual mesh files.  Match only exact mesh
    # basenames so these small resource-fork records can never be selected.
    if basename.startswith("._"):
        return None
    for root_id in selected:
        if basename.lower() in {f"{root_id}.h5", f"{root_id}.hdf5"}:
            return root_id
    return None


def extract_selected_meshes(archive: Path, mesh_dir: Path, root_ids: list[int]) -> dict[int, Path]:
    """Stream the archive once and materialize exactly the frozen meshes."""

    mesh_dir.mkdir(parents=True, exist_ok=True)
    found = {
        root_id: mesh_dir / f"mesh_{root_id}.h5"
        for root_id in root_ids
        if (mesh_dir / f"mesh_{root_id}.h5").is_file()
    }
    missing = set(root_ids) - set(found)
    if not missing:
        return found
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            root_id = _member_root_id(member.name, missing)
            if root_id is None or not member.isfile():
                continue
            source = bundle.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not read {member.name}")
            destination = mesh_dir / f"mesh_{root_id}.h5"
            with destination.open("wb") as target:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    target.write(block)
            found[root_id] = destination
            missing.remove(root_id)
            print(f"Extracted {member.name} -> {destination.name}", flush=True)
            if not missing:
                break
    if missing:
        raise FileNotFoundError(f"Archive lacked selected roots: {sorted(missing)}")
    return found


def face_edges(faces: np.ndarray, link_edges: np.ndarray | None = None) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    if link_edges is not None and np.asarray(link_edges).size:
        edges = np.concatenate(
            [edges, np.asarray(link_edges, dtype=np.int64).reshape(-1, 2)], axis=0
        )
    edges = np.sort(edges, axis=1)
    edges = edges[edges[:, 0] != edges[:, 1]]
    return np.unique(edges, axis=0)


@dataclass
class SimpleMesh:
    """Minimal meshparty-compatible graph without cloud-volume dependencies."""

    vertices: np.ndarray
    faces: np.ndarray
    link_edges: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        self.edges = face_edges(self.faces, self.link_edges)
        edge_lengths = np.linalg.norm(
            self.vertices[self.edges[:, 0]] - self.vertices[self.edges[:, 1]], axis=1
        ).astype(np.float32)
        directed_edges = np.concatenate([self.edges.T, self.edges.T[[1, 0]]], axis=1)
        weights = np.concatenate([edge_lengths, edge_lengths])
        self.csgraph = sparse.csr_matrix(
            (weights, directed_edges), shape=(len(self.vertices), len(self.vertices))
        )
        self.node_mask = np.ones(len(self.vertices), dtype=bool)

    def map_indices_to_unmasked(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)

    def map_boolean_to_unmasked(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=bool)

    def filter_unmasked_indices_padded(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.int64)


def retain_largest_component(mesh: SimpleMesh) -> tuple[SimpleMesh, dict[str, Any]]:
    """Remove detached mesh fragments before rooting and TEASAR skeletonization."""

    n_components, labels = connected_components(mesh.csgraph, directed=False)
    counts = np.bincount(labels)
    largest_label = int(np.argmax(counts))
    keep = labels == largest_label
    old_to_new = np.full(len(keep), -1, dtype=np.int64)
    old_to_new[keep] = np.arange(int(keep.sum()), dtype=np.int64)
    face_keep = np.all(keep[mesh.faces], axis=1)
    faces = old_to_new[mesh.faces[face_keep]]
    link_edges = None
    if mesh.link_edges is not None and np.asarray(mesh.link_edges).size:
        links = np.asarray(mesh.link_edges, dtype=np.int64).reshape(-1, 2)
        link_keep = np.all(keep[links], axis=1)
        link_edges = old_to_new[links[link_keep]]
    retained = SimpleMesh(mesh.vertices[keep], faces, link_edges)
    return retained, {
        "n_mesh_components_before_filter": int(n_components),
        "n_vertices_before_component_filter": int(len(mesh.vertices)),
        "n_vertices_largest_component": int(keep.sum()),
        "largest_component_fraction": float(keep.mean()),
        "component_filter": "largest face-plus-link-edge connected component",
    }


def read_fixed_mesh(
    path: Path, soma_nm: np.ndarray
) -> tuple[SimpleMesh, str, float, dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        vertices = np.asarray(handle["vertices"], dtype=float)
        faces = np.asarray(handle["faces"], dtype=np.int64)
        link_edges = (
            np.asarray(handle["link_edges"], dtype=np.int64)
            if "link_edges" in handle
            else None
        )
    unscaled_mesh = SimpleMesh(vertices, faces, link_edges)
    component_mesh, component_qc = retain_largest_component(unscaled_mesh)
    candidates = {
        "stored_nm": component_mesh.vertices,
        "stored_voxels_scaled_4_4_40_nm": component_mesh.vertices * VOXEL_NM[None, :],
    }
    distances = {
        name: float(cKDTree(candidate).query(soma_nm, k=1)[0])
        for name, candidate in candidates.items()
    }
    units = min(distances, key=distances.get)
    scaled_mesh = SimpleMesh(
        candidates[units], component_mesh.faces, component_mesh.link_edges
    )
    return scaled_mesh, units, distances[units], component_qc


def ray_radius_nm(mesh: SimpleMesh, mesh_indices: np.ndarray, soma_index: int) -> np.ndarray:
    """Estimate local radius as half the opposite-surface ray distance."""

    import trimesh
    from trimesh.ray import has_embree
    from trimesh.ray.ray_pyembree import RayMeshIntersector

    if not has_embree:
        raise RuntimeError(
            "Pinky radius estimation requires the frozen Embree backend; the "
            "triangle backend has prohibitive memory scaling on these meshes"
        )

    tri = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        process=False,
        validate=False,
    )
    mesh_indices = np.asarray(mesh_indices, dtype=np.int64)
    valid = (mesh_indices >= 0) & (mesh_indices < len(mesh.vertices))
    output = np.full(len(mesh_indices), np.nan, dtype=float)
    query_rows = np.flatnonzero(valid)
    if len(query_rows):
        anchors = mesh_indices[query_rows]
        normals = np.asarray(tri.vertex_normals[anchors], dtype=float)
        origins = mesh.vertices[anchors] - normals
        directions = -normals
        locations, ray_rows, _ = RayMeshIntersector(tri).intersects_location(
            origins,
            directions,
            multiple_hits=False,
        )
        if len(ray_rows):
            distance = np.linalg.norm(locations - mesh.vertices[anchors[ray_rows]], axis=1)
            good = np.isfinite(distance) & (distance >= 50.0)
            # The inward ray crosses the local cable to the opposite surface,
            # so its travel distance estimates diameter rather than radius.
            output[query_rows[ray_rows[good]]] = 0.5 * distance[good]
    finite = output[np.isfinite(output)]
    fallback = float(np.median(finite)) if len(finite) else 500.0
    output[~np.isfinite(output)] = fallback
    output = np.clip(output, 50.0, 20_000.0)
    output[int(soma_index)] = max(output[int(soma_index)], 7_500.0)
    return output


def skeletonize_mesh(mesh: SimpleMesh, soma_nm: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    from meshparty import skeletonize

    root_vertex = int(cKDTree(mesh.vertices).query(soma_nm, k=1)[1])
    print(
        f"Starting TEASAR: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces, "
        f"root {root_vertex}",
        flush=True,
    )
    skel = skeletonize.skeletonize_mesh(
        mesh,
        root_index=root_vertex,
        soma_radius=7_500,
        collapse_soma=True,
        collapse_function="sphere",
        invalidation_d=12_000,
        compute_radius=False,
        compute_original_index=True,
        smooth_vertices=False,
        cc_vertex_thresh=100,
        verbose=False,
    )
    print(f"TEASAR complete: {len(skel.vertices)} skeleton nodes", flush=True)
    indices = np.arange(len(skel.vertices), dtype=int)
    parents_raw = np.asarray(skel.parent_nodes(indices), dtype=object)
    parents = np.asarray(
        [-1 if value is None else int(value) for value in parents_raw], dtype=int
    )
    print("Starting opposite-surface radius estimation", flush=True)
    radius_nm = ray_radius_nm(mesh, np.asarray(skel.mesh_index), int(skel.root))
    print("Opposite-surface radius estimation complete", flush=True)
    frame = pd.DataFrame(
        {
            "id": indices + 1,
            "type": np.where(indices == int(skel.root), 1, 3),
            "x": skel.vertices[:, 0] / 1_000.0,
            "y": skel.vertices[:, 1] / 1_000.0,
            "z": skel.vertices[:, 2] / 1_000.0,
            "radius": radius_nm / 1_000.0,
            "parent": np.where(parents < 0, -1, parents + 1),
        }
    )
    qc = {
        "n_mesh_vertices": int(len(mesh.vertices)),
        "n_mesh_faces": int(len(mesh.faces)),
        "n_mesh_graph_edges": int(len(mesh.edges)),
        "nearest_soma_mesh_vertex_nm": float(
            np.linalg.norm(mesh.vertices[root_vertex] - soma_nm)
        ),
        "n_skeleton_nodes": int(len(frame)),
        "root_index_zero_based": int(skel.root),
        "radius_method": "half opposite-surface ray distance at TEASAR mesh anchors",
        "radius_backend": "trimesh.ray.ray_pyembree with embreex 2.17.7.post7",
        "radius_median_um": float(frame["radius"].median()),
    }
    return frame, qc


def write_synapse_derivatives(
    source: Path,
    soma_table: Path,
    root_ids: list[int],
    prepared: Path,
    chunksize: int = 1_000_000,
) -> dict[int, dict[str, Any]]:
    soma = pd.read_csv(soma_table)
    type_map = {
        int(root): ("E" if cell_type == "e" else "I" if cell_type == "i" else "?")
        for root, cell_type in soma[["pt_root_id", "cell_type"]].itertuples(index=False)
        if pd.notna(root)
    }
    selected = set(root_ids)
    buckets: dict[int, list[pd.DataFrame]] = {root_id: [] for root_id in root_ids}
    usecols = [
        "id",
        "pre_root_id",
        "post_root_id",
        "cleft_vx",
        "post_pos_x_vx",
        "post_pos_y_vx",
        "post_pos_z_vx",
    ]
    for chunk in pd.read_csv(source, usecols=usecols, chunksize=chunksize):
        kept = chunk[chunk["post_root_id"].isin(selected)].copy()
        for root_id, part in kept.groupby("post_root_id"):
            buckets[int(root_id)].append(part)
    summaries: dict[int, dict[str, Any]] = {}
    for root_id in root_ids:
        if not buckets[root_id]:
            raise ValueError(f"No incoming synapses found for selected root {root_id}")
        raw = pd.concat(buckets[root_id], ignore_index=True)
        typed = raw["pre_root_id"].map(type_map).fillna("?")
        output = pd.DataFrame(
            {
                "id": raw["id"].astype("int64"),
                "post_pt_position_x": raw["post_pos_x_vx"],
                "post_pt_position_y": raw["post_pos_y_vx"],
                "post_pt_position_z": raw["post_pos_z_vx"],
                "size": pd.to_numeric(raw["cleft_vx"], errors="coerce").fillna(0.0),
                "typed_class": typed,
                "target_tag_probability": 0.0,
                "target_proxy_class": "?",
                "pre_root_id": raw["pre_root_id"],
            }
        )
        path = prepared / f"synapses_{root_id}.csv.gz"
        output.to_csv(path, index=False, compression="gzip")
        summaries[root_id] = {
            "n_incoming_synapses": int(len(output)),
            "n_typed_e": int(output["typed_class"].eq("E").sum()),
            "n_typed_i": int(output["typed_class"].eq("I").sum()),
            "n_untyped": int(output["typed_class"].eq("?").sum()),
            "derivative": path.name,
            "derivative_sha256": sha256(path),
        }
    return summaries


def prepare_skeleton_derivative(
    root_id: int,
    soma_nm: np.ndarray,
    mesh_path: Path,
    prepared: Path,
) -> dict[str, Any]:
    """Prepare one cell so high-memory ray indices die with the worker process."""

    mesh, units, soma_distance, component_qc = read_fixed_mesh(mesh_path, soma_nm)
    skeleton, qc = skeletonize_mesh(mesh, soma_nm)
    path = prepared / f"skeleton_{root_id}.csv.gz"
    skeleton.to_csv(path, index=False, compression="gzip")
    record = {
        **qc,
        **component_qc,
        "mesh_file": mesh_path.name,
        "mesh_sha256": sha256(mesh_path),
        "mesh_coordinate_interpretation": units,
        "unit_choice_nearest_soma_distance_nm": soma_distance,
        "skeleton_derivative": path.name,
        "skeleton_derivative_sha256": sha256(path),
    }
    print(f"Skeletonized {root_id}: {len(skeleton)} nodes", flush=True)
    return record


def reusable_worker_record(
    record_path: Path, prepared: Path, mesh_path: Path
) -> dict[str, Any] | None:
    """Return a completed worker record only after derivative hash validation."""

    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    skeleton_name = record.get("skeleton_derivative")
    if not skeleton_name:
        return None
    skeleton_path = prepared / str(skeleton_name)
    required = {
        "mesh_sha256": sha256(mesh_path),
        "skeleton_derivative_sha256": sha256(skeleton_path)
        if skeleton_path.is_file()
        else None,
        "radius_backend": "trimesh.ray.ray_pyembree with embreex 2.17.7.post7",
    }
    if any(record.get(key) != value for key, value in required.items()):
        return None
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--skip-skeletons", action="store_true")
    parser.add_argument("--skeleton-worker-root", type=int)
    parser.add_argument("--worker-qc", type=Path)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort)
    root_ids = [int(value) for value in cohort["root_id"]]
    args.prepared.mkdir(parents=True, exist_ok=True)
    if args.skeleton_worker_root is not None:
        root_id = int(args.skeleton_worker_root)
        if root_id not in root_ids:
            raise ValueError(f"Worker root {root_id} is outside the frozen cohort")
        if args.worker_qc is None:
            raise ValueError("--worker-qc is required with --skeleton-worker-root")
        cohort_lookup = cohort.set_index("root_id")
        soma_nm = cohort_lookup.loc[
            root_id, ["x_nm", "y_nm", "z_nm"]
        ].to_numpy(dtype=float)
        record = prepare_skeleton_derivative(
            root_id,
            soma_nm,
            args.raw / "meshes" / f"mesh_{root_id}.h5",
            args.prepared,
        )
        args.worker_qc.parent.mkdir(parents=True, exist_ok=True)
        args.worker_qc.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return

    archive = args.raw / "layer23_v185.tar.gz"
    synapse_source = args.raw / "pni_synapses_v185.csv"
    soma_source = args.raw / "soma_valence_v185.csv"
    raw_inputs = {
        path.name: validate_zenodo_input(path)
        for path in (archive, synapse_source, soma_source)
    }
    meshes = extract_selected_meshes(archive, args.raw / "meshes", root_ids)

    cells: dict[str, dict[str, Any]] = {}
    if not args.skip_skeletons:
        worker_qc_dir = args.prepared / "worker_qc"
        worker_qc_dir.mkdir(parents=True, exist_ok=True)
        for root_id in root_ids:
            worker_qc = worker_qc_dir / f"cell_{root_id}.json"
            reusable = reusable_worker_record(
                worker_qc, args.prepared, meshes[root_id]
            )
            if reusable is None:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--raw",
                        str(args.raw),
                        "--prepared",
                        str(args.prepared),
                        "--cohort",
                        str(args.cohort),
                        "--skeleton-worker-root",
                        str(root_id),
                        "--worker-qc",
                        str(worker_qc),
                    ],
                    check=True,
                )
                reusable = reusable_worker_record(
                    worker_qc, args.prepared, meshes[root_id]
                )
                if reusable is None:
                    raise RuntimeError(f"Worker derivative failed validation: {root_id}")
            else:
                print(f"Reusing hash-validated skeleton {root_id}", flush=True)
            cells[str(root_id)] = reusable
    synapse_qc = write_synapse_derivatives(
        synapse_source, soma_source, root_ids, args.prepared
    )
    for root_id in root_ids:
        cells.setdefault(str(root_id), {}).update(synapse_qc[root_id])

    payload = {
        "status": "prepared_from_frozen_outcome_independent_cohort",
        "dataset": "MICrONS phase-1 Pinky v185",
        "n_selected": len(root_ids),
        "parameters": {
            "soma_radius_nm": 7_500,
            "teasar_invalidation_distance_nm": 12_000,
            "minimum_component_vertices": 100,
            "component_selection": "largest face-plus-link-edge connected component",
            "ray_backend": "embreex 2.17.7.post7",
            "memory_isolation": "one skeletonization subprocess per selected cell",
            "synapse_classification": "presynaptic soma-valence label only",
        },
        "source_code": {
            Path(__file__).name: sha256(Path(__file__).resolve()),
        },
        "raw_inputs": raw_inputs,
        "cells": cells,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        print(args.manifest.relative_to(ROOT))
    except ValueError:
        print(args.manifest)


if __name__ == "__main__":
    main()

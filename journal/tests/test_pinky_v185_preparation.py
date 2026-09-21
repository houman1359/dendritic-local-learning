from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


JOURNAL = Path(__file__).resolve().parents[1]
SCRIPT = JOURNAL / "scripts" / "prepare_pinky_v185_replication.py"


def _module():
    spec = importlib.util.spec_from_file_location("pinky_preparation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_archive_member_match_rejects_appledouble_metadata() -> None:
    module = _module()
    root_id = 648518346349538440
    selected = {root_id}
    assert module._member_root_id(f"layer23_v185/{root_id}.h5", selected) == root_id
    assert module._member_root_id(f"layer23_v185/{root_id}.hdf5", selected) == root_id
    assert module._member_root_id(f"layer23_v185/._{root_id}.h5", selected) is None
    assert module._member_root_id(f"layer23_v185/prefix_{root_id}.h5", selected) is None


def test_largest_component_filter_removes_detached_fragment() -> None:
    module = _module()
    vertices = np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [10, 0, 0],
            [11, 0, 0],
            [10, 1, 0],
        ],
        dtype=float,
    )
    faces = np.asarray(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3], [4, 5, 6]], dtype=int
    )
    retained, qc = module.retain_largest_component(module.SimpleMesh(vertices, faces))
    assert len(retained.vertices) == 4
    assert len(retained.faces) == 4
    assert retained.faces.max() == 3
    assert qc["n_mesh_components_before_filter"] == 2
    assert qc["n_vertices_largest_component"] == 4


def test_worker_reuse_requires_matching_mesh_and_skeleton_hashes(tmp_path: Path) -> None:
    module = _module()
    mesh = tmp_path / "mesh.h5"
    skeleton = tmp_path / "skeleton.csv.gz"
    mesh.write_bytes(b"mesh")
    skeleton.write_bytes(b"skeleton")
    record_path = tmp_path / "record.json"
    record = {
        "mesh_sha256": module.sha256(mesh),
        "skeleton_derivative": skeleton.name,
        "skeleton_derivative_sha256": module.sha256(skeleton),
        "radius_backend": "trimesh.ray.ray_pyembree with embreex 2.17.7.post7",
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")
    assert module.reusable_worker_record(record_path, tmp_path, mesh) == record
    skeleton.write_bytes(b"changed")
    assert module.reusable_worker_record(record_path, tmp_path, mesh) is None

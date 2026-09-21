from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def toy_segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "root_id": [1] * 7,
            "segment_id": [0, 1, 2, 3, 4, 5, 6],
            "parent_segment_id": [-1, 0, 0, 1, 1, 2, 2],
            "topological_depth": [0, 1, 1, 2, 2, 2, 2],
            "n_children": [2, 2, 2, 0, 0, 0, 0],
            "mean_radius_um": [5.0, 1.2, 1.0, 0.8, 0.7, 0.75, 0.65],
            "edge_length_um": [0.0, 20.0, 25.0, 30.0, 22.0, 28.0, 24.0],
            "E_size": [1.0, 2.0, 0.5, 1.0, 0.0, 1.5, 0.8],
            "I_size": [0.5, 0.2, 0.7, 0.0, 0.4, 0.1, 0.3],
        }
    )


def test_degree_depth_surrogate_preserves_declared_invariants() -> None:
    module = load_script("analyze_reciprocal_routing_controls.py")
    original = toy_segments()
    surrogate = module.degree_depth_matched_surrogate(
        original, np.random.default_rng(20260804)
    )

    assert surrogate.segment_id.tolist() == original.segment_id.tolist()
    assert surrogate.topological_depth.tolist() == original.topological_depth.tolist()
    assert sorted(surrogate.n_children.tolist()) == sorted(original.n_children.tolist())
    parent_depth = surrogate.set_index("segment_id").topological_depth.to_dict()
    for row in surrogate.itertuples(index=False):
        if row.parent_segment_id >= 0:
            assert row.topological_depth == parent_depth[row.parent_segment_id] + 1


def test_physical_cable_builder_is_reciprocal_and_positive() -> None:
    module = load_script("analyze_physical_cable_sensitivity.py")
    electrical, matrix, rhs, root_index, parents, children = (
        module.physical_conductance_system(
            toy_segments(),
            0.35,
            0.35,
            1.0,
            -0.2,
            axial_resistivity_ohm_cm=150.0,
            membrane_resistance_ohm_cm2=15_000.0,
        )
    )

    assert root_index == 0
    assert len(parents) == len(electrical) - 1
    assert sum(len(value) for value in children.values()) == len(electrical) - 1
    np.testing.assert_allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
    assert np.linalg.eigvalsh(matrix).min() > 0
    assert np.isfinite(rhs).all()
    assert (electrical.g_leak > 0).all()
    assert (electrical.loc[electrical.segment_id.ne(0), "g_edge"] > 0).all()
    assert int(electrical.degenerate_length_imputed.sum()) == 0

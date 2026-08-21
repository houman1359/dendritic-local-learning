from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_irregular_tree_wavelets.py"
SPEC = importlib.util.spec_from_file_location("irregular_tree_wavelets", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def synthetic_tree() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": [0, 1, 2, 3, 4, 5, 6],
            "parent_segment_id": [-1, 0, 0, 1, 1, 2, 2],
            "E_size": [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "I_size": [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            "g_i": [0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0],
            "g_total": [1.0] * 7,
            "root_id": [11] * 7,
        }
    )


def test_irregular_tree_haar_is_complete_and_orthonormal() -> None:
    frame = synthetic_tree()
    excitatory = [3, 4, 5, 6]
    weights = frame.set_index("segment_id").loc[excitatory, "E_size"].to_numpy()
    basis, metadata = MODULE.irregular_tree_haar(frame, excitatory, weights)
    assert basis.shape == (4, 4)
    assert len(metadata) == 3
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(basis[:, 0], np.sqrt(weights / weights.sum()), atol=1e-12)


def test_centered_energy_decomposes_over_wavelets() -> None:
    frame = synthetic_tree()
    excitatory, weights, raw = MODULE.route_dictionary(frame)
    basis, metadata = MODULE.irregular_tree_haar(frame, excitatory, weights)
    centered, removed = MODULE.centered_weighted_dictionary(raw, weights)
    spectrum, error = MODULE.scale_energy(centered, basis, metadata)
    assert 0.0 <= removed <= 1.0
    assert error < 1e-12
    np.testing.assert_allclose(spectrum["energy_fraction"].sum(), 1.0, atol=1e-12)


def test_cell_analysis_is_deterministic() -> None:
    frame = synthetic_tree()
    first_modes, first_scales, first_audit = MODULE.analyze_cell(frame, "test", 17, 8)
    second_modes, second_scales, second_audit = MODULE.analyze_cell(frame, "test", 17, 8)
    pd.testing.assert_frame_equal(first_modes, second_modes)
    pd.testing.assert_frame_equal(first_scales, second_scales)
    assert first_audit == second_audit
    assert first_audit["basis_rank"] == 4
    assert first_audit["scale_mode_count"] == 3

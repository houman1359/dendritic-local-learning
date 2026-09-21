from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "analyze_trained_partition_residual.py"
SPEC = importlib.util.spec_from_file_location("trained_partition_residual", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


def test_exact_route_has_zero_address_residual() -> None:
    exact = np.eye(4)
    values = MODEL.weighted_field_decomposition(
        exact,
        exact,
        np.ones_like(exact),
        np.array([0.2, -0.4, 0.7, -0.1]),
    )
    assert values["address_residual_energy"] < 1e-14
    assert np.isclose(values["address_capture"], 1.0)


def test_scalar_route_captures_one_quarter_of_uniform_four_coordinate_field() -> None:
    exact = np.eye(4)
    shared = np.ones((4, 4)) / 2.0
    values = MODEL.weighted_field_decomposition(
        exact,
        shared,
        np.ones_like(exact),
        np.ones(4),
    )
    assert np.isclose(values["address_capture"], 0.25)
    assert values["decomposition_error"] < 1e-12


def test_decomposition_remains_exact_with_heterogeneous_weights() -> None:
    rng = np.random.default_rng(41)
    exact = np.eye(5)
    route = np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0]), (5, 1))
    values = MODEL.weighted_field_decomposition(
        exact,
        route,
        rng.uniform(0.2, 2.0, size=exact.shape),
        rng.normal(size=5),
    )
    assert values["decomposition_error"] < 1e-12

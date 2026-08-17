from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "code" / "theory" / "operator_spectral.py"
SPEC = importlib.util.spec_from_file_location("operator_spectral", PATH)
assert SPEC is not None and SPEC.loader is not None
THEORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(THEORY)


def test_ky_fan_bound_dominates_random_rank_projectors() -> None:
    rng = np.random.default_rng(53)
    rotation, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    covariance = rotation @ np.diag(np.linspace(3.0, 0.1, 10)) @ rotation.T
    bound = THEORY.ky_fan_capture_bound(covariance, 3)
    for _ in range(200):
        q, _ = np.linalg.qr(rng.normal(size=(10, 3)))
        projector = q @ q.T
        capture = np.trace(projector @ covariance) / np.trace(covariance)
        assert capture <= bound + 1e-12


def test_affine_projector_crossover_is_exact() -> None:
    first = np.diag([1.0, 0.0])
    second = np.diag([0.0, 1.0])
    c0 = np.diag([0.2, 0.8])
    c1 = np.diag([0.9, 0.1])
    crossover = THEORY.affine_projector_crossover(first, second, c0, c1)
    mixture = (1.0 - crossover) * c0 + crossover * c1
    assert 0.0 < crossover < 1.0
    assert np.isclose(np.trace((first - second) @ mixture), 0.0, atol=1e-14)


def test_focal_inverse_norm_identity_and_bound() -> None:
    conductance = np.array(
        [[3.0, -1.0, 0.0], [-1.0, 3.5, -0.7], [0.0, -0.7, 2.2]]
    )
    dose = 0.8
    result = THEORY.focal_inverse_change_bound(conductance, 1, dose)
    updated = conductance.copy()
    updated[1, 1] += dose
    observed = np.linalg.norm(np.linalg.inv(updated) - np.linalg.inv(conductance), 2)
    assert np.isclose(observed, result["exact_inverse_change_norm"], atol=1e-14)
    assert observed <= result["spectral_upper_bound"] + 1e-14

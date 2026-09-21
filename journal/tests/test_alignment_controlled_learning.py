from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_alignment_controlled_learning.py"
SPEC = importlib.util.spec_from_file_location("alignment_controlled", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_alignment_generator_fixes_energy_and_route_fraction() -> None:
    basis = np.eye(7, 3)
    rng = np.random.default_rng(7)
    fields = MODULE.alignment_controlled_fields(
        basis,
        n_fields=128,
        alignments=(0.0, 0.25, 0.7, 1.0),
        rng=rng,
    )
    for alignment, values in fields.items():
        np.testing.assert_allclose(np.sum(values * values, axis=1), 1.0, atol=1e-12)
        capture = MODULE.projection_capture(values, basis)
        np.testing.assert_allclose(capture, alignment, atol=1e-12)


def test_one_step_progress_matches_explicit_quadratic_update() -> None:
    rng = np.random.default_rng(11)
    target = rng.standard_normal(9)
    target /= np.linalg.norm(target)
    basis = MODULE.orthonormal_basis(rng.standard_normal((9, 4)))
    projected = basis @ (basis.T @ target)
    fraction = 0.1
    curvature = np.geomspace(0.5, 1.5, len(target))
    reported, _ = MODULE.quadratic_progress(
        target[None, :], basis, curvature, fraction, learning_rate=0.25, steps=20
    )

    full_update = fraction * target
    restricted_update = fraction * projected / np.linalg.norm(projected)
    optimum = target / curvature
    initial_loss = 0.5 * np.sum(curvature * optimum**2)
    full_decrease = initial_loss - 0.5 * np.sum(curvature * (full_update - optimum) ** 2)
    restricted_decrease = initial_loss - 0.5 * np.sum(
        curvature * (restricted_update - optimum) ** 2
    )
    np.testing.assert_allclose(reported[0], restricted_decrease / full_decrease, atol=1e-12)


def test_iterative_projected_progress_matches_explicit_updates() -> None:
    rng = np.random.default_rng(19)
    target = rng.standard_normal(12)
    target /= np.linalg.norm(target)
    basis = MODULE.orthonormal_basis(rng.standard_normal((12, 5)))
    curvature = np.geomspace(0.5, 1.5, len(target))
    learning_rate = 0.25
    steps = 20

    optimum = target / curvature
    full = np.zeros_like(target)
    restricted = np.zeros_like(target)
    for _ in range(steps):
        full += learning_rate * (target - curvature * full)
        residual_gradient = target - curvature * restricted
        restricted += learning_rate * basis @ (basis.T @ residual_gradient)
    initial_loss = 0.5 * np.sum(curvature * optimum**2)
    full_progress = initial_loss - 0.5 * np.sum(curvature * (full - optimum) ** 2)
    restricted_progress = initial_loss - 0.5 * np.sum(
        curvature * (restricted - optimum) ** 2
    )
    _, reported = MODULE.quadratic_progress(
        target[None, :],
        basis,
        curvature,
        step_fraction=0.1,
        learning_rate=learning_rate,
        steps=steps,
    )
    np.testing.assert_allclose(reported[0], restricted_progress / full_progress, atol=1e-12)


def test_depth_dictionary_has_one_assignment_per_coordinate() -> None:
    depths = np.linspace(0.0, 100.0, 37)
    dictionary = MODULE.depth_dictionary(depths, channels=8)
    assert dictionary.shape == (37, 8)
    np.testing.assert_array_equal(dictionary.sum(axis=1), np.ones(37))

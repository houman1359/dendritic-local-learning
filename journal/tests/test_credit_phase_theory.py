from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "theory" / "credit_phase.py"
SPEC = importlib.util.spec_from_file_location("credit_phase", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PHASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE)


def test_credit_operator_bound_matches_quadratic_monte_carlo() -> None:
    rng = np.random.default_rng(5)
    g = np.array([0.8, -0.3, 0.2])
    m = np.array([[1.0, 0.1, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.2]])
    sigma = np.diag([0.4, 0.2, 0.1])
    eta = 0.15
    samples = rng.multivariate_normal(np.zeros(3), sigma, size=300_000)
    updates = -eta * (g[None, :] + samples) @ m.T
    # L(x)=g^T x + ||x||^2/2 at x=0 has smoothness one.
    losses = updates @ g + 0.5 * np.sum(updates * updates, axis=1)
    bound = PHASE.expected_loss_upper_bound(0.0, g, m, sigma, 1.0, eta)
    assert abs(losses.mean() - bound) < 2e-3


def test_partition_variance_locates_branching_benefit() -> None:
    coefficients = np.array([1.0, 1.0, -1.0, -1.0])
    weights = np.ones(4)
    point, _ = PHASE.weighted_partition_residual(
        coefficients, weights, np.zeros(4, dtype=int)
    )
    branches, _ = PHASE.weighted_partition_residual(
        coefficients, weights, np.array([0, 0, 1, 1])
    )
    assert point == 4.0
    assert branches == 0.0


def test_spectral_capture_and_static_column_scaling() -> None:
    dictionary = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    scaled = dictionary @ np.diag([0.2, 7.0])
    p1 = PHASE.weighted_projector(dictionary, np.ones(4))
    p2 = PHASE.weighted_projector(scaled, np.ones(4))
    assert np.allclose(p1, p2, atol=1e-12)
    covariance = np.diag([3.0, 3.0, 1.0, 1.0])
    assert 0.0 <= PHASE.spectral_capture(p1, covariance) <= 1.0


def test_coefficient_error_pythagorean_decomposition() -> None:
    c = np.array([1.0, 0.8, -0.2, -0.4])
    d = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    address, estimation, total = PHASE.coefficient_error_decomposition(
        c, d, np.array([0.6, -0.1]), np.ones(4)
    )
    assert np.isclose(address + estimation, total, atol=1e-12)


def test_branch_specific_reliability_is_never_worse_than_global() -> None:
    signal = np.array([9.0, 4.0, 1.0, 0.25])
    noise = np.array([1.0, 4.0, 9.0, 16.0])
    gains = PHASE.reliability_shrinkage(signal, noise)
    branch, global_value = PHASE.reliability_guarantees(signal, noise)
    assert np.all((0.0 <= gains) & (gains <= 1.0))
    assert branch > global_value
    equal_signal = np.ones(4)
    equal_noise = np.ones(4)
    branch_equal, global_equal = PHASE.reliability_guarantees(
        equal_signal, equal_noise
    )
    assert np.isclose(branch_equal, global_equal)


def test_fixed_step_reliability_uses_actual_step_and_beats_common_gain() -> None:
    signal = np.array([9.0, 4.0, 1.0, 0.25])
    noise = np.array([1.0, 4.0, 9.0, 16.0])
    gains = PHASE.fixed_step_reliability_gain(signal, noise, 0.5)
    expected = np.minimum(1.0, 2.0 * signal / (signal + noise))
    assert np.allclose(gains, expected)
    branch, common, common_gain, returned = PHASE.fixed_step_reliability_guarantees(
        signal, noise, 0.5
    )
    assert np.allclose(returned, gains)
    assert 0.0 <= common_gain <= 1.0
    assert branch >= common


def test_fixed_step_formula_reduces_to_reliability_at_full_step() -> None:
    signal = np.array([3.0, 1.0])
    noise = np.array([2.0, 4.0])
    assert np.allclose(
        PHASE.fixed_step_reliability_gain(signal, noise, 1.0),
        PHASE.reliability_shrinkage(signal, noise),
    )


def test_focal_shunt_attenuates_positive_m_matrix_adjoint() -> None:
    conductance = np.array(
        [[3.0, -1.0, 0.0], [-1.0, 3.0, -1.0], [0.0, -1.0, 2.0]]
    )
    source = np.array([1.0, 0.0, 0.0])
    before, after = PHASE.focal_shunt_adjoint(conductance, source, 1, 0.8)
    assert np.all(after >= -1e-14)
    assert np.all(after <= before + 1e-14)


def test_depth_penalty_and_tree_haar_orthogonality() -> None:
    assert np.isclose(
        PHASE.shared_feedback_lognormal_cosine(np.array([0.2, 0.3])),
        np.exp(-0.25),
    )
    basis, levels = PHASE.tree_haar_basis(16)
    assert basis.shape == (16, 16)
    assert levels.max() == 4
    assert np.allclose(basis.T @ basis, np.eye(16), atol=1e-12)

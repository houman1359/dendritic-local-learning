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

BUDGETS = (1, 2, 4, 8)


def hierarchical_covariance(eigenvalues: np.ndarray) -> np.ndarray:
    basis = THEORY.tree_haar_basis(len(eigenvalues))
    return basis @ np.diag(eigenvalues) @ basis.T


def test_ancestry_span_equals_coarse_haar_span() -> None:
    haar = THEORY.tree_haar_basis(8)
    for k in BUDGETS:
        ancestry = THEORY.balanced_ancestry_dictionary(8, k)
        qa, _ = np.linalg.qr(ancestry)
        qh, _ = np.linalg.qr(haar[:, :k])
        assert np.allclose(qa @ qa.T, qh @ qh.T, atol=1e-12)


def test_tree_structured_covariance_attains_ky_fan_at_every_budget() -> None:
    eigenvalues = np.array([4.0, 2.5, 1.8, 1.2, 0.6, 0.4, 0.25, 0.1])
    covariance = hierarchical_covariance(eigenvalues)
    for k in BUDGETS:
        ancestry = THEORY.balanced_ancestry_dictionary(8, k)
        capture = THEORY.dictionary_capture(covariance, ancestry)
        bound = THEORY.ky_fan_capture_bound(covariance, k)
        assert abs(capture - bound) < 1e-12


def test_fine_scale_energy_breaks_attainment_and_shortfall_is_off_span_energy() -> None:
    # Put the largest eigenvalue on the finest Haar mode: it lies outside the
    # K=2 ancestry span, so the shortfall must equal exactly that missing
    # leading energy relative to the Ky-Fan optimum.
    eigenvalues = np.array([1.0, 0.8, 0.3, 0.2, 0.1, 0.05, 0.02, 5.0])
    covariance = hierarchical_covariance(eigenvalues)
    ancestry = THEORY.balanced_ancestry_dictionary(8, 2)
    shortfall = THEORY.laminar_shortfall(covariance, ancestry)
    total = eigenvalues.sum()
    expected = (5.0 + 1.0) / total - (1.0 + 0.8) / total
    assert shortfall > 0
    assert abs(shortfall - expected) < 1e-12


def test_shortfall_nonnegative_for_generic_covariance() -> None:
    rng = np.random.default_rng(20260812)
    for _ in range(50):
        raw = rng.normal(size=(8, 8))
        covariance = raw @ raw.T
        for k in BUDGETS:
            ancestry = THEORY.balanced_ancestry_dictionary(8, k)
            assert THEORY.laminar_shortfall(covariance, ancestry) >= -1e-12


def test_shortfall_decomposition_is_exact_for_generic_covariance() -> None:
    # Shortfall = top-K eigenspace energy outside the span MINUS trailing
    # energy the span retains (normalized); it is only BOUNDED by the first
    # term, not equal to it.
    rng = np.random.default_rng(31415)
    for _ in range(50):
        raw = rng.normal(size=(8, 8))
        covariance = raw @ raw.T
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        total = eigenvalues.sum()
        for k in BUDGETS:
            ancestry = THEORY.balanced_ancestry_dictionary(8, k)
            q, _ = np.linalg.qr(ancestry)
            projector = q @ q.T
            outside = sum(
                eigenvalues[i]
                * np.linalg.norm((np.eye(8) - projector) @ eigenvectors[:, i]) ** 2
                for i in range(k)
            )
            retained_trailing = sum(
                eigenvalues[j] * np.linalg.norm(projector @ eigenvectors[:, j]) ** 2
                for j in range(k, 8)
            )
            shortfall = THEORY.laminar_shortfall(covariance, ancestry)
            assert abs(shortfall - (outside - retained_trailing) / total) < 1e-10
            assert shortfall <= outside / total + 1e-10


def test_marginal_condition_matches_direct_utility_comparison() -> None:
    rng = np.random.default_rng(7)
    for _ in range(2000):
        s = float(rng.uniform(0.01, 5.0))
        n = float(rng.uniform(0.0, 5.0))
        ds = float(rng.uniform(0.0, 3.0))
        dn = float(rng.uniform(0.0, 3.0))
        direct = THEORY.bound_utility(s + ds, n + dn) > THEORY.bound_utility(s, n)
        assert direct == THEORY.utility_increases(s, n, ds, dn)


def test_utility_interior_maximum_under_fast_signal_decay() -> None:
    # Signal spectrum decays geometrically; noise is admitted uniformly per
    # added route. The utility along the nested ladder then rises and falls
    # around an interior budget.
    signal_per_scale = {1: 4.0, 2: 1.2, 4: 0.15, 8: 0.02}
    noise_per_route = 0.25
    utilities = []
    s = n = 0.0
    for k in BUDGETS:
        s += signal_per_scale[k]
        n = noise_per_route * k
        utilities.append(THEORY.bound_utility(s, n))
    peak = int(np.argmax(utilities))
    assert 0 < peak < len(BUDGETS) - 1

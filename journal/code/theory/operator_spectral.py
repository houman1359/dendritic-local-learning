"""Spectral bounds for routed credit and passive conductance operators."""

from __future__ import annotations

import numpy as np


def ky_fan_capture_bound(covariance: np.ndarray, rank: int) -> float:
    """Maximum normalized capture attainable by any projector of given rank."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.min(initial=0.0) < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    k = int(rank)
    if not 0 <= k <= len(eigenvalues):
        raise ValueError("rank must lie between zero and covariance dimension")
    total = float(eigenvalues.sum())
    if total <= 0 or k == 0:
        return 0.0
    return float(np.sort(eigenvalues)[::-1][:k].sum() / total)


def affine_projector_crossover(
    first_projector: np.ndarray,
    second_projector: np.ndarray,
    covariance_at_zero: np.ndarray,
    covariance_at_one: np.ndarray,
) -> float:
    """Return rho where two captures cross under an affine covariance mixture.

    The mixture is ``C(rho)=(1-rho)C0+rho*C1``.  NaN is returned when the
    captures are parallel/equal or when their crossing lies outside [0, 1].
    The normalization by trace is irrelevant when ``trace(C0)=trace(C1)``.
    """

    first = np.asarray(first_projector, dtype=float)
    second = np.asarray(second_projector, dtype=float)
    c0 = np.asarray(covariance_at_zero, dtype=float)
    c1 = np.asarray(covariance_at_one, dtype=float)
    if first.shape != second.shape or first.shape != c0.shape or c0.shape != c1.shape:
        raise ValueError("projectors and covariances must have one common shape")
    if not np.isclose(np.trace(c0), np.trace(c1), rtol=1e-10, atol=1e-12):
        raise ValueError("affine normalized crossover requires equal covariance trace")
    difference = first - second
    delta_zero = float(np.trace(difference @ c0))
    delta_one = float(np.trace(difference @ c1))
    slope = delta_one - delta_zero
    if abs(slope) <= 1e-15:
        return float("nan")
    crossover = -delta_zero / slope
    return float(crossover) if -1e-12 <= crossover <= 1.0 + 1e-12 else float("nan")


def focal_inverse_change_bound(
    conductance: np.ndarray, site: int, dose: float
) -> dict[str, float]:
    """Exact and spectral upper bound for a focal diagonal conductance update."""

    matrix = np.asarray(conductance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("conductance must be square")
    if not np.allclose(matrix, matrix.T, atol=1e-12):
        raise ValueError("conductance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues.min() <= 0:
        raise ValueError("conductance must be positive definite")
    kappa = float(dose)
    if kappa < 0:
        raise ValueError("dose must be nonnegative")
    k = int(site)
    if not 0 <= k < len(matrix):
        raise ValueError("site outside matrix")
    inverse = np.linalg.inv(matrix)
    column = inverse[:, k]
    exact = kappa * float(column @ column) / (1.0 + kappa * inverse[k, k])
    spectral = (
        kappa
        * float(np.linalg.norm(inverse, ord=2) ** 2)
        / (1.0 + kappa / float(eigenvalues.max()))
    )
    return {
        "exact_inverse_change_norm": exact,
        "spectral_upper_bound": spectral,
        "minimum_conductance_eigenvalue": float(eigenvalues.min()),
        "maximum_conductance_eigenvalue": float(eigenvalues.max()),
    }


def balanced_ancestry_dictionary(n_leaves: int, budget_k: int) -> np.ndarray:
    """Nested ancestry indicator columns of a balanced binary tree.

    Column ``j`` is the indicator of the ``j``-th contiguous subtree of
    ``n_leaves / budget_k`` leaves, matching the paper's route families at
    budgets 1, 2, 4, ..., ``n_leaves``.
    """

    n = int(n_leaves)
    k = int(budget_k)
    if n <= 0 or k <= 0 or n % k != 0:
        raise ValueError("budget must divide the leaf count")
    block = n // k
    dictionary = np.zeros((n, k))
    for j in range(k):
        dictionary[j * block : (j + 1) * block, j] = 1.0
    return dictionary


def tree_haar_basis(n_leaves: int) -> np.ndarray:
    """Orthonormal tree-Haar basis ordered coarse to fine.

    The first column is the constant; each subsequent column is a
    normalized sibling-subtree difference. The span of the first ``k``
    columns equals the span of the ``k``-route ancestry dictionary for
    every admissible budget ``k`` in ``{1, 2, 4, ..., n_leaves}``.
    """

    n = int(n_leaves)
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("leaf count must be a power of two")
    columns = [np.full(n, 1.0 / np.sqrt(n))]
    scale = n
    while scale > 1:
        half = scale // 2
        for start in range(0, n, scale):
            vec = np.zeros(n)
            vec[start : start + half] = 1.0
            vec[start + half : start + scale] = -1.0
            columns.append(vec / np.sqrt(scale))
        scale = half
    return np.column_stack(columns)


def dictionary_capture(covariance: np.ndarray, dictionary: np.ndarray) -> float:
    """Normalized energy of ``covariance`` in the column span of ``dictionary``."""

    covariance = np.asarray(covariance, dtype=float)
    q, _ = np.linalg.qr(np.asarray(dictionary, dtype=float))
    projector = q @ q.T
    total = float(np.trace(covariance))
    if total <= 0:
        return 0.0
    return float(np.trace(projector @ covariance) / total)


def laminar_shortfall(covariance: np.ndarray, dictionary: np.ndarray) -> float:
    """Ky--Fan capture at the dictionary rank minus the dictionary capture.

    Nonnegative for every dictionary; zero exactly when the span of the
    top-``k`` eigenvectors of ``covariance`` lies inside the dictionary
    span (``k`` = dictionary rank).
    """

    dictionary = np.asarray(dictionary, dtype=float)
    rank = int(np.linalg.matrix_rank(dictionary))
    return ky_fan_capture_bound(covariance, rank) - dictionary_capture(
        covariance, dictionary
    )


def bound_utility(signal: float, noise: float, smoothness: float = 1.0) -> float:
    """Guaranteed one-step decrease U = S^2 / (2 L (S + N)) for a projector.

    ``signal`` is the retained task-gradient energy g'Mg = ||Mg||^2 and
    ``noise`` the admitted stochastic energy tr(M Sigma M'); both must be
    nonnegative.
    """

    signal = float(signal)
    noise = float(noise)
    if signal <= 0.0:
        return 0.0
    return signal**2 / (2.0 * float(smoothness) * (signal + noise))


def utility_increases(
    signal: float, noise: float, d_signal: float, d_noise: float
) -> bool:
    """Exact marginal condition for one rung up a nested projector ladder.

    U(K+1) > U(K) if and only if
    ``dS [ (S + N)(2 S + dS) - S^2 ] > S^2 dN`` where ``dS, dN >= 0`` are
    the marginal retained signal and admitted noise of the added routes.
    """

    s = float(signal)
    n = float(noise)
    ds = float(d_signal)
    dn = float(d_noise)
    if ds < 0 or dn < 0:
        raise ValueError("marginal signal and noise must be nonnegative")
    return ds * ((s + n) * (2.0 * s + ds) - s**2) > s**2 * dn

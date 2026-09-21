"""Numerical reference functions for the dendritic credit phase theory.

The functions in this module mirror the propositions used in the journal
manuscript.  They are deliberately small and NumPy-only so that the analytical
claims can be checked independently of the training code.
"""

from __future__ import annotations

import numpy as np


def expected_loss_upper_bound(
    loss: float,
    gradient: np.ndarray,
    operator: np.ndarray,
    covariance: np.ndarray,
    smoothness: float,
    step_size: float,
) -> float:
    """Smoothness upper bound for ``w - step_size * M * g_hat``.

    ``g_hat`` is assumed unbiased with covariance ``covariance`` and the
    operator is held fixed at the current state.
    """

    g = np.asarray(gradient, dtype=float)
    m = np.asarray(operator, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    mg = m @ g
    retained = float(g @ mg)
    second_moment = float(mg @ mg + np.trace(m @ sigma @ m.T))
    return float(
        loss
        - step_size * retained
        + 0.5 * smoothness * step_size * step_size * second_moment
    )


def optimal_credit_step(
    gradient: np.ndarray,
    operator: np.ndarray,
    covariance: np.ndarray,
    smoothness: float,
) -> tuple[float, float]:
    """Return the optimal bound step and its guaranteed decrease.

    A non-descent operator returns ``(0, 0)``.
    """

    g = np.asarray(gradient, dtype=float)
    m = np.asarray(operator, dtype=float)
    sigma = np.asarray(covariance, dtype=float)
    mg = m @ g
    retained = float(g @ mg)
    second_moment = float(mg @ mg + np.trace(m @ sigma @ m.T))
    if retained <= 0.0 or second_moment <= 0.0:
        return 0.0, 0.0
    eta = retained / (float(smoothness) * second_moment)
    decrease = retained * retained / (2.0 * float(smoothness) * second_moment)
    return float(eta), float(decrease)


def weighted_projector(dictionary: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Return the projector onto ``col(dictionary)`` in a diagonal W metric."""

    d = np.asarray(dictionary, dtype=float)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or d.shape[0] != len(w):
        raise ValueError("weights must match dictionary rows")
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive")
    root = np.sqrt(w)
    transformed = root[:, None] * d
    # Nested route dictionaries are deliberately redundant.  An unpivoted QR
    # can therefore retain the wrong columns (and, for wide dictionaries,
    # even report the wrong numerical rank).  The left singular vectors give
    # the unique projector onto the route span without choosing a particular
    # redundant parameterization.
    u, singular_values, _ = np.linalg.svd(transformed, full_matrices=False)
    if not len(singular_values):
        return np.zeros((d.shape[0], d.shape[0]), dtype=float)
    tolerance = max(transformed.shape) * np.finfo(float).eps * singular_values[0]
    keep = singular_values > tolerance
    if not np.any(keep):
        return np.zeros((d.shape[0], d.shape[0]), dtype=float)
    q = u[:, keep]
    euclidean = q @ q.T
    return (euclidean / root[:, None]) * root[None, :]


def spectral_capture(projector: np.ndarray, covariance: np.ndarray) -> float:
    """Expected energy fraction captured by an orthogonal route projector."""

    p = np.asarray(projector, dtype=float)
    c = np.asarray(covariance, dtype=float)
    total = float(np.trace(c))
    if total <= 0:
        return 0.0
    return float(np.trace(p @ c) / total)


def weighted_partition_residual(
    coefficients: np.ndarray, weights: np.ndarray, groups: np.ndarray
) -> tuple[float, np.ndarray]:
    """Minimum residual from one coefficient per declared synaptic group."""

    c = np.asarray(coefficients, dtype=float)
    w = np.asarray(weights, dtype=float)
    labels = np.asarray(groups)
    if c.shape != w.shape or c.shape != labels.shape:
        raise ValueError("coefficients, weights, and groups must match")
    if np.any(w <= 0):
        raise ValueError("weights must be strictly positive")
    means = np.empty_like(c)
    for label in np.unique(labels):
        mask = labels == label
        means[mask] = np.sum(w[mask] * c[mask]) / np.sum(w[mask])
    residual = float(np.sum(w * (c - means) ** 2))
    return residual, means


def coefficient_error_decomposition(
    coefficients: np.ndarray,
    dictionary: np.ndarray,
    estimated_route_coefficients: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, float]:
    """Split realized field error into address and coefficient errors."""

    c = np.asarray(coefficients, dtype=float)
    d = np.asarray(dictionary, dtype=float)
    zhat = np.asarray(estimated_route_coefficients, dtype=float)
    w = np.asarray(weights, dtype=float)
    gram = d.T @ (w[:, None] * d)
    rhs = d.T @ (w * c)
    zstar = np.linalg.pinv(gram, rcond=1e-12) @ rhs
    address = float(np.sum(w * (c - d @ zstar) ** 2))
    estimation = float(np.sum(w * (d @ (zhat - zstar)) ** 2))
    total = float(np.sum(w * (c - d @ zhat) ** 2))
    return address, estimation, total


def route_gram_diagnostics(
    dictionary: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Return rank, positive-spectrum condition number, and coherence."""

    d = np.asarray(dictionary, dtype=float)
    w = np.asarray(weights, dtype=float)
    transformed = np.sqrt(w)[:, None] * d
    gram = transformed.T @ transformed
    eigenvalues = np.linalg.eigvalsh(gram)
    positive = eigenvalues[eigenvalues > 1e-12]
    condition = float(positive.max() / positive.min()) if len(positive) else np.inf
    norms = np.linalg.norm(transformed, axis=0)
    normalized = np.divide(
        transformed,
        norms[None, :],
        out=np.zeros_like(transformed),
        where=norms[None, :] > 1e-15,
    )
    correlations = np.abs(normalized.T @ normalized)
    np.fill_diagonal(correlations, 0.0)
    return {
        "rank": float(np.linalg.matrix_rank(transformed, tol=1e-12)),
        "lambda_min_positive": float(positive.min()) if len(positive) else 0.0,
        "condition_number": condition,
        "coherence": float(correlations.max(initial=0.0)),
    }


def reliability_shrinkage(
    signal_energy: np.ndarray, noise_energy: np.ndarray
) -> np.ndarray:
    """Optimal branch attenuation relative to a global ``1/L`` step."""

    signal = np.asarray(signal_energy, dtype=float)
    noise = np.asarray(noise_energy, dtype=float)
    if np.any(signal < 0) or np.any(noise < 0):
        raise ValueError("energies must be nonnegative")
    return np.divide(
        signal,
        signal + noise,
        out=np.zeros_like(signal),
        where=(signal + noise) > 0,
    )


def fixed_step_reliability_gain(
    signal_energy: np.ndarray,
    noise_energy: np.ndarray,
    step_fraction: float,
    maximum_gain: float = 1.0,
) -> np.ndarray:
    """Optimal attenuation for the fixed step ``eta=step_fraction/L``.

    The smoothness guarantee for one block is proportional to
    ``a*S - step_fraction*a**2*(S+N)/2``.  A conductance can attenuate but not
    amplify, so the unconstrained optimum is clipped at ``maximum_gain``.
    ``reliability_shrinkage`` is the special case ``step_fraction=1``.
    """

    signal = np.asarray(signal_energy, dtype=float)
    noise = np.asarray(noise_energy, dtype=float)
    fraction = float(step_fraction)
    ceiling = float(maximum_gain)
    if np.any(signal < 0) or np.any(noise < 0):
        raise ValueError("energies must be nonnegative")
    if fraction <= 0:
        raise ValueError("step_fraction must be positive")
    if ceiling <= 0:
        raise ValueError("maximum_gain must be positive")
    denominator = fraction * (signal + noise)
    unconstrained = np.divide(
        signal,
        denominator,
        out=np.zeros_like(signal),
        where=denominator > 0,
    )
    return np.minimum(ceiling, unconstrained)


def fixed_step_reliability_guarantees(
    signal_energy: np.ndarray,
    noise_energy: np.ndarray,
    step_fraction: float,
    smoothness: float = 1.0,
    maximum_gain: float = 1.0,
) -> tuple[float, float, float, np.ndarray]:
    """Return branch/global guarantees and the corresponding fixed-step gains."""

    signal = np.asarray(signal_energy, dtype=float)
    noise = np.asarray(noise_energy, dtype=float)
    fraction = float(step_fraction)
    if float(smoothness) <= 0:
        raise ValueError("smoothness must be positive")
    branch_gains = fixed_step_reliability_gain(
        signal, noise, fraction, maximum_gain
    )
    total_signal = float(np.sum(signal))
    total_noise = float(np.sum(noise))
    global_gain = float(
        fixed_step_reliability_gain(
            np.array([total_signal]),
            np.array([total_noise]),
            fraction,
            maximum_gain,
        )[0]
    )

    def guarantee(gains: np.ndarray) -> float:
        return float(
            fraction
            / float(smoothness)
            * np.sum(
                gains * signal
                - 0.5 * fraction * gains * gains * (signal + noise)
            )
        )

    branch_value = guarantee(branch_gains)
    global_value = guarantee(np.full_like(signal, global_gain))
    return branch_value, global_value, global_gain, branch_gains


def reliability_guarantees(
    signal_energy: np.ndarray, noise_energy: np.ndarray, smoothness: float = 1.0
) -> tuple[float, float]:
    """Best branch-specific and best global smoothness guarantees."""

    signal = np.asarray(signal_energy, dtype=float)
    noise = np.asarray(noise_energy, dtype=float)
    denominator = signal + noise
    branch = 0.5 / smoothness * float(
        np.sum(
            np.divide(
                signal * signal,
                denominator,
                out=np.zeros_like(signal),
                where=denominator > 0,
            )
        )
    )
    total_signal = float(np.sum(signal))
    total_second = float(np.sum(denominator))
    global_value = (
        0.5 / smoothness * total_signal * total_signal / total_second
        if total_second > 0
        else 0.0
    )
    return branch, global_value


def focal_shunt_adjoint(
    conductance: np.ndarray,
    source: np.ndarray,
    site: int,
    dose: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pre/post focal-shunt passive adjoints."""

    g = np.asarray(conductance, dtype=float)
    s = np.asarray(source, dtype=float)
    q = np.linalg.solve(g, s)
    updated = g.copy()
    updated[site, site] += float(dose)
    q_updated = np.linalg.solve(updated, s)
    return q, q_updated


def shared_feedback_lognormal_cosine(log_gain_variances: np.ndarray) -> float:
    """Depth penalty from independent lognormal transport-gain factors."""

    variances = np.asarray(log_gain_variances, dtype=float)
    if np.any(variances < 0):
        raise ValueError("variances must be nonnegative")
    return float(np.exp(-0.5 * np.sum(variances)))


def tree_haar_basis(n_leaves: int) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal coarse-to-fine Haar basis for a balanced binary tree.

    Returns ``(basis, levels)`` with basis vectors in columns.  Level zero is
    the global mean; level one is the root-child contrast.
    """

    n = int(n_leaves)
    if n < 2 or n & (n - 1):
        raise ValueError("n_leaves must be a power of two >= 2")
    vectors = [np.ones(n, dtype=float) / np.sqrt(n)]
    levels = [0]
    depth = int(np.log2(n))
    for level in range(1, depth + 1):
        block = n // (2 ** (level - 1))
        half = block // 2
        for start in range(0, n, block):
            vector = np.zeros(n, dtype=float)
            vector[start : start + half] = 1.0
            vector[start + half : start + block] = -1.0
            vector /= np.sqrt(block)
            vectors.append(vector)
            levels.append(level)
    basis = np.column_stack(vectors)
    return basis, np.asarray(levels, dtype=int)

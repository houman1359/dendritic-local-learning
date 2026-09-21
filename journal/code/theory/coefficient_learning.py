"""Exact finite-time risk for noisy rate-based route-coefficient learning."""

from __future__ import annotations

import numpy as np


def linear_field_learning_risk(
    field_operator: np.ndarray,
    target: np.ndarray,
    observation_noise_variance: float,
    step_fraction: float,
    iterations: int,
) -> dict[str, float]:
    """Return exact bias and variance of a linear noisy field iteration.

    The iteration is

    ``f[t+1] = f[t] - c A (f[t] - target - noise[t])``,

    starting at zero, with independent isotropic noise of per-coordinate
    variance ``observation_noise_variance``.  ``A`` must be symmetric positive
    semidefinite.  The target is required to lie in its positive eigenspace.
    Returned risks use the population-loss convention ``||f-target||^2/2``.
    """

    operator = np.asarray(field_operator, dtype=float)
    target = np.asarray(target, dtype=float)
    variance = float(observation_noise_variance)
    step = float(step_fraction)
    steps = int(iterations)
    if operator.shape != (len(target), len(target)):
        raise ValueError("field_operator must be square and match target")
    if not np.allclose(operator, operator.T, atol=1e-12):
        raise ValueError("field_operator must be symmetric")
    if variance < 0 or step <= 0 or steps < 0:
        raise ValueError("variance must be nonnegative, step positive, iterations nonnegative")
    eigenvalues, eigenvectors = np.linalg.eigh(operator)
    if np.min(eigenvalues) < -1e-10:
        raise ValueError("field_operator must be positive semidefinite")
    eigenvalues = np.maximum(eigenvalues, 0.0)
    coefficients = eigenvectors.T @ target
    inactive = eigenvalues <= 1e-12
    if np.sum(coefficients[inactive] ** 2) > 1e-10:
        raise ValueError("target must lie in the positive eigenspace")
    transition = 1.0 - step * eigenvalues
    if np.any(np.abs(transition[~inactive]) >= 1.0):
        raise ValueError("unstable field iteration")
    power = transition ** (2 * steps)
    bias_by_mode = power * coefficients * coefficients
    innovation = (step * eigenvalues) ** 2
    denominator = 1.0 - transition * transition
    variance_multiplier = np.divide(
        innovation * (1.0 - power),
        denominator,
        out=np.zeros_like(eigenvalues),
        where=np.abs(denominator) > 1e-15,
    )
    bias_loss = 0.5 * float(np.sum(bias_by_mode))
    variance_loss = 0.5 * variance * float(np.sum(variance_multiplier))
    return {
        "bias_loss": bias_loss,
        "variance_loss": variance_loss,
        "total_expected_loss": bias_loss + variance_loss,
        "variance_multiplier": 0.5 * float(np.sum(variance_multiplier)),
    }


def pairwise_effective_sample_crossover(
    first_operator: np.ndarray,
    second_operator: np.ndarray,
    target: np.ndarray,
    observation_noise_variance_at_n1: float,
    step_fraction: float,
    iterations: int,
) -> float:
    """Effective sample size at which two exact expected risks cross.

    Observation variance is assumed to be ``variance_at_n1 / n``.  Returns
    NaN when no positive finite crossover exists.
    """

    first = linear_field_learning_risk(
        first_operator, target, 0.0, step_fraction, iterations
    )
    second = linear_field_learning_risk(
        second_operator, target, 0.0, step_fraction, iterations
    )
    first_unit = linear_field_learning_risk(
        first_operator, target, 1.0, step_fraction, iterations
    )["variance_loss"]
    second_unit = linear_field_learning_risk(
        second_operator, target, 1.0, step_fraction, iterations
    )["variance_loss"]
    bias_difference = first["bias_loss"] - second["bias_loss"]
    variance_difference = first_unit - second_unit
    if abs(bias_difference) <= 1e-15:
        return float("nan")
    crossover = -float(observation_noise_variance_at_n1) * variance_difference / bias_difference
    return crossover if np.isfinite(crossover) and crossover > 0 else float("nan")

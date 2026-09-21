"""Reserved positive Bernstein conductance mixtures, with independent quadrature.

Labels use a high-precision analytic antiderivative, checked independently by
adaptive high-precision quadrature. This evaluator changes no target functions,
seeds or optimization choices from the frozen positive-density plan.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import mpmath as mp
import numpy as np
import torch
from scipy.special import roots_legendre


@lru_cache(maxsize=2)
def gauss_rule(order):
    # Float64 eigensolver weights can lose enough accuracy to affect very small
    # label differences. Refine polynomial roots/weights independently at70digits,
    # then perform the full-grid integration in NumPy float64.
    initial, _ = roots_legendre(order)
    nodes, weights = [], []
    with mp.workdps(70):

        def polynomial(z):
            previous, current = mp.mpf(1), z
            for k in range(2, order + 1):
                previous, current = (
                    current,
                    ((2 * k - 1) * z * current - (k - 1) * previous) / k,
                )
            derivative = order * (z * current - previous) / (z * z - 1)
            return current, derivative

        for value in initial:
            z = mp.mpf(float(value))
            for _ in range(8):
                p, derivative = polynomial(z)
                delta = p / derivative
                z -= delta
                if abs(delta) < mp.mpf("1e-65"):
                    break
            p, derivative = polynomial(z)
            assert abs(p) < mp.mpf("1e-60")
            nodes.append(float((z + 1) / 2))
            weights.append(float(1 / ((1 - z * z) * derivative * derivative)))
    return np.array(nodes), np.array(weights)


@dataclass(frozen=True)
class PositiveDensity:
    lower: float
    upper: float
    coefficients: tuple[float, ...]
    seed: int | None = None

    def __post_init__(self):
        if not 0 < self.lower < self.upper or len(self.coefficients) != 5:
            raise ValueError(
                "Positive interval and degree-four Bernstein coefficients required"
            )
        if not all(math.isfinite(c) and c > 0 for c in self.coefficients):
            raise ValueError("Density coefficients must be finite and positive")

    @classmethod
    def from_seed(cls, lower, upper, seed):
        rng = np.random.Generator(np.random.PCG64(seed))
        return cls(lower, upper, tuple(np.exp(rng.uniform(-1, 1, 5)).tolist()), seed)

    def receipt(self):
        return {
            "family": "positive_degree4_bernstein",
            "interval": [self.lower, self.upper],
            "coefficients": list(self.coefficients),
            "seed": self.seed,
            "rng": "numpy.Generator(PCG64); five U[-1,1] draws then float64 exp",
            "normalization": "Integral on t in[0,1] is sum(coefficients)/5; z=a+(b-a)t.",
            "label_evaluator": "70-digit analytic antiderivative; independent adaptive high-precision quadrature qualification",
            "independent_numpy_evaluator": "Positive256-node Gauss-Legendre integration with70-digit refined nodes and weights",
        }

    def _polynomial(self):
        return [
            mp.fsum(
                mp.mpf(self.coefficients[j])
                * math.comb(4, j)
                * math.comb(4 - j, k - j)
                * (-1) ** (k - j)
                for j in range(k + 1)
            )
            for k in range(5)
        ]

    def value_mp(self, x):
        """Use the caller's precision, with exact conversion of stored float parameters."""
        a, d = mp.mpf(self.lower), mp.mpf(self.upper) - mp.mpf(self.lower)
        norm = mp.fsum(mp.mpf(c) for c in self.coefficients) / 5
        base = mp.log1p(d / (x + a)) / d
        moments = [base]
        for k in range(1, 5):
            moments.append(1 / (d * k) - (x + a) * moments[-1] / d)
        return (
            x
            * mp.fsum(c * value for c, value in zip(self._polynomial(), moments))
            / norm
        )

    def quadrature_mp(self, x):
        a, d = mp.mpf(self.lower), mp.mpf(self.upper) - mp.mpf(self.lower)
        coefficients = [mp.mpf(c) for c in self.coefficients]
        norm = mp.fsum(coefficients) / 5

        def integrand(t):
            density = (
                mp.fsum(
                    c * math.comb(4, j) * t**j * (1 - t) ** (4 - j)
                    for j, c in enumerate(coefficients)
                )
                / norm
            )
            return x * density / (x + a + d * t)

        return mp.quad(integrand, [0, mp.mpf("0.01"), mp.mpf("0.1"), 1])

    def labels(self, x, dps=70):
        array = (
            x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        )
        with mp.workdps(dps):
            values = [float(self.value_mp(mp.mpf(float(v)))) for v in array.flatten()]
        return torch.tensor(values, dtype=torch.float64).reshape(array.shape)

    def numpy_quadrature(self, x, order=256):
        nodes, weights = gauss_rule(order)
        density = sum(
            c * math.comb(4, j) * nodes**j * (1 - nodes) ** (4 - j)
            for j, c in enumerate(self.coefficients)
        )
        density /= sum(self.coefficients) / 5
        poles = self.lower + (self.upper - self.lower) * nodes
        result = []
        for block in np.array_split(
            np.asarray(x).reshape(-1, 1), max(1, math.ceil(np.size(x) / 4096))
        ):
            result.append((block / (block + poles)) @ (weights * density))
        return np.concatenate(result).reshape(np.shape(x))


def qualify_reserved_targets():
    """Function-only checks, no training, TEST grid, or optimizer selection."""
    records = []
    # These diagnostic coordinates are not the final TEST midpoint grid.
    points = [0.0, 1e-8, 1e-5, 0.001, 0.01, 0.13, 0.41, 0.73, 1.0]
    for lower, upper in [(1.0, 2.0), (0.01, 4.0)]:
        for seed in [7301, 7307, 7319]:
            density = PositiveDensity.from_seed(lower, upper, seed)
            with mp.workdps(70):
                error = max(
                    abs(density.value_mp(mp.mpf(x)) - density.quadrature_mp(mp.mpf(x)))
                    for x in points
                )
            values = density.labels(np.array(points))
            independent = density.numpy_quadrature(np.array(points))
            difference = float(np.max(abs(values.numpy() - independent)))
            assert error < mp.mpf("1e-60") and difference < 3e-15
            records.append(
                {
                    **density.receipt(),
                    "diagnostic_points": points,
                    "max_analytic_vs_adaptive_quadrature_difference": str(error),
                    "max_float64_vs_numpy_quadrature_difference": difference,
                }
            )
    return records

"""Paired intrinsic-rank teachers on a common rotation-invariant input law."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def raw_response(t, intervals, tilts, response="mixture"):
    t = np.asarray(t, dtype=np.float64)
    if response == "sine":
        return t + 0.1 * np.sin(2 * np.pi * t)
    if response != "mixture":
        raise ValueError("Unknown response family")
    a, b = np.asarray(intervals).T
    delta = b - a
    logarithm = np.log1p(delta / (a + t))
    return (
        t
        / delta
        * (logarithm + np.asarray(tilts) * (2 - (a + b + 2 * t) * logarithm / delta))
    )


def sphere_inputs(seed, n, blocks=4, dimension=3):
    if n < 1 or dimension != 3:
        raise ValueError("Positive row count and dimension3 required")
    # Independent streams give exact prefixes for BOTH directions and radii.
    directions = np.random.default_rng(np.random.SeedSequence([seed, 11])).normal(
        size=(n, blocks, dimension)
    )
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    radii = 0.5 * np.random.default_rng(np.random.SeedSequence([seed, 13])).random(
        (n, blocks, 1)
    ) ** (1 / 3)
    return directions * radii


def moment_integrals(intervals, tilts, response="mixture", order=96, angles=384):
    """Marginal Beta(2,2) and joint disk-marginal Gaussian quadrature."""
    nodes, weights = np.polynomial.legendre.leggauss(256)
    t, w = (nodes + 1) / 2, weights / 2
    h = raw_response(t[:, None], intervals, tilts, response)
    w = w * 6 * t * (1 - t)
    means = w @ h
    seconds = w @ (h * h)
    nodes, weights = np.polynomial.legendre.leggauss(order)
    u, w = (nodes + 1) / 2, weights / 2
    # Projecting uniform3D ball onto a plane gives radial density
    # 3*rho*sqrt(R^2-rho^2)/R^3. Withu=sqrt(1-(rho/R)^2), weight=3u^2.
    rho = 0.5 * np.sqrt(1 - u * u)
    phi = 2 * np.pi * np.arange(angles) / angles
    t1 = (0.5 + rho[:, None] * np.cos(phi)[None, :]).reshape(-1, 1)
    t2 = (0.5 + rho[:, None] * np.sin(phi)[None, :]).reshape(-1, 1)
    joint_weights = np.repeat(3 * u * u * w / angles, angles)
    h1, h2 = (
        raw_response(t1, intervals, tilts, response),
        raw_response(t2, intervals, tilts, response),
    )
    cross = joint_weights @ (h1 * h2)
    variance1 = seconds - means * means
    variance2 = 2 * seconds + 2 * cross - 4 * means * means
    if not (np.all(variance1 > 0) and np.all(variance2 > 0)):
        raise FloatingPointError("Nonpositive teacher variance")
    return means, np.stack([variance1, variance2]), cross


@dataclass(frozen=True)
class TeacherConfig:
    index: int
    seed: int
    response: str = "mixture"


class RankTeacher:
    """Private generation object; no directions or moments enter fitting."""

    def __init__(self, config: TeacherConfig):
        self.config = config
        rng = np.random.default_rng(config.seed)
        base = np.array([[1.0, 2.0], [0.25, 4.0], [0.05, 4.0], [0.01, 4.0]])
        self.intervals = base * np.exp(rng.uniform(np.log(0.8), np.log(1.25), (4, 2)))
        self.tilts = rng.uniform(-0.8, 0.8, 4)
        self.rotations = np.stack(
            [np.linalg.qr(rng.normal(size=(3, 3)))[0] for _ in range(4)]
        )
        self.means, self.variances, self.cross = moment_integrals(
            self.intervals, self.tilts, config.response
        )
        _, check_variances, check_cross = moment_integrals(
            self.intervals, self.tilts, config.response, order=64, angles=256
        )
        self.moment_check = {
            "max_relative_variance_difference": float(
                np.max(abs(check_variances / self.variances - 1))
            ),
            "max_cross_difference": float(np.max(abs(check_cross - self.cross))),
            "main_joint_orders": [96, 384],
            "check_joint_orders": [64, 256],
            "marginal_beta_quadrature_order": 256,
        }
        if self.moment_check["max_relative_variance_difference"] > 1e-8:
            raise FloatingPointError("Teacher quadrature did not converge")

    def components(self, x, intrinsic_rank):
        if intrinsic_rank not in (1, 2):
            raise ValueError("Intrinsic rank must be1 or2")
        x = np.asarray(x, dtype=np.float64)
        if (
            x.ndim != 3
            or x.shape[1:] != (4, 3)
            or np.max(np.linalg.norm(x, axis=2)) > 0.5 + 1e-12
        ):
            raise ValueError("Input outside the common four-block ball")
        latent = np.einsum("nki,kij->nkj", x, self.rotations)
        values = sum(
            raw_response(
                0.5 + latent[:, :, j], self.intervals, self.tilts, self.config.response
            )
            for j in range(intrinsic_rank)
        )
        return (values - intrinsic_rank * self.means) / np.sqrt(
            4 * self.variances[intrinsic_rank - 1]
        )

    def evaluate(self, x, intrinsic_rank):
        return self.components(x, intrinsic_rank).sum(axis=1)

    def specification(self):
        return {
            "index": self.config.index,
            "seed": self.config.seed,
            "response": self.config.response,
            "intervals": self.intervals.tolist(),
            "density_tilt": self.tilts.tolist(),
            "rotations": self.rotations.tolist(),
            "component_means": self.means.tolist(),
            "component_variances_q1_q2": self.variances.tolist(),
            "cross_moments": self.cross.tolist(),
            "moment_check": self.moment_check,
            "input_law": "Independent uniform3D balls radius.5; identical raw law acrossallteachers/ranks. CovarianceI/20. One projection plus.5 hasBeta(2,2) marginal; two orthogonal projections are dependent.",
        }


def generate_data(
    teacher, seed, intrinsic_rank, train_n=2048, endpoint="test", endpoint_n=8192
):
    train_x = sphere_inputs(seed * 100 + 1, train_n)
    endpoint_x = sphere_inputs(seed * 100 + 3, endpoint_n)
    return {
        "train_x": train_x,
        "train_y_clean": teacher.evaluate(train_x, intrinsic_rank),
        "train_epsilon": np.random.default_rng(
            np.random.SeedSequence([seed, 17])
        ).normal(size=train_n),
        endpoint + "_x": endpoint_x,
        endpoint + "_y_clean": teacher.evaluate(endpoint_x, intrinsic_rank),
        endpoint
        + "_epsilon": np.random.default_rng(np.random.SeedSequence([seed, 19])).normal(
            size=endpoint_n
        ),
    }

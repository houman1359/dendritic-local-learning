"""Private radial-plane teachers and public scalar-label ball observations.

The power, input radius, supplied block structure and normalization are public
task assumptions. Projectors are private generation/audit state. No fitting API
in this module consumes a private teacher.
"""

from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _radius(value):
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Radius must be positive and finite")
    return float(value)


def array_sha256(value):
    """Bind dtype and shape as well as array bytes, without pickle."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def normalized_moments(m, radius=0.5):
    """Exact rational unit-ball coefficients, scaled to the supplied radius."""
    m, radius = _positive_integer(m, "Power"), _radius(radius)

    def coefficient(j):
        return Fraction(
            3 * 4**j * math.factorial(j) ** 2,
            (2 * j + 3) * math.factorial(2 * j + 1),
        )

    mean_unit = coefficient(m)
    second_unit = coefficient(2 * m)
    variance_unit = second_unit - mean_unit**2
    mean = float(mean_unit) * radius ** (2 * m)
    variance = float(variance_unit) * radius ** (4 * m)
    if not (math.isfinite(mean) and math.isfinite(variance) and variance > 0):
        raise ValueError("Power/radius combination has unrepresentable moments")
    return {
        "power": m,
        "radius": radius,
        "mean": mean,
        "variance": variance,
        "sigma": math.sqrt(variance),
        "unit_radius_mean_exact": str(mean_unit),
        "unit_radius_second_moment_exact": str(second_unit),
        "unit_radius_variance_exact": str(variance_unit),
        "normalization": "Center each radial component, divide by its population sigma, sum over blocks divided by sqrt(K).",
    }


def sample_inputs(seed, n, blocks=4, radius=0.5):
    """Independent uniform three-balls; exact row prefixes across sample sizes."""
    n, blocks = _positive_integer(n, "Rows"), _positive_integer(blocks, "Blocks")
    radius = _radius(radius)
    if int(seed) != seed or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    directions = np.random.default_rng(np.random.SeedSequence([int(seed), 11])).normal(
        size=(n, blocks, 3)
    )
    directions /= np.linalg.norm(directions, axis=2, keepdims=True)
    radii = radius * np.random.default_rng(
        np.random.SeedSequence([int(seed), 13])
    ).random((n, blocks, 1)) ** (1 / 3)
    return directions * radii


def validate_inputs(x, blocks, radius):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (blocks, 3) or len(x) < 1:
        raise ValueError("Inputs must have nonempty shape [N,K,3]")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite input")
    if np.any(np.linalg.norm(x, axis=2) > radius * (1 + 1e-12)):
        raise ValueError("Input outside the declared ball")
    return x


class RadialTeacher:
    """Private teacher; changing m with the same seed preserves all projectors."""

    def __init__(self, m, seed, blocks=4, radius=0.5):
        self.blocks = _positive_integer(blocks, "Blocks")
        self.radius = _radius(radius)
        self.moments = normalized_moments(m, radius)
        self.m = self.moments["power"]
        if int(seed) != seed or seed < 0:
            raise ValueError("Seed must be a nonnegative integer")
        self.seed = int(seed)
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, 211]))
        banks = np.stack(
            [
                np.linalg.qr(rng.normal(size=(3, 3)))[0][:, :2]
                for _ in range(self.blocks)
            ]
        )
        self._projectors = banks @ banks.transpose(0, 2, 1)
        self._projectors.flags.writeable = False

    def components(self, x):
        """Private audit components, already divided by sqrt(K)."""
        x = validate_inputs(x, self.blocks, self.radius)
        radial = np.einsum("nki,kij,nkj->nk", x, self._projectors, x)
        values = radial**self.m
        return (values - self.moments["mean"]) / (
            math.sqrt(self.blocks) * self.moments["sigma"]
        )

    def evaluate(self, x):
        return self.components(x).sum(axis=1)

    def specification(self):
        """Write this private record separately from any public observation NPZ."""
        return {
            "schema": "radial_width_private_teacher_v1",
            "power": self.m,
            "seed": self.seed,
            "blocks": self.blocks,
            "radius": self.radius,
            "projectors": self._projectors.tolist(),
            "projectors_sha256": array_sha256(self._projectors),
            "moments": dict(self.moments),
            "input_law": "Independent uniform three-dimensional balls; power and seed do not affect the input law.",
            "teacher_variation": "Orientation only within each fixed radial power; m changes the response shape.",
        }

    @classmethod
    def from_specification(cls, specification):
        if specification.get("schema") != "radial_width_private_teacher_v1":
            raise ValueError("Unknown private teacher schema")
        result = cls.__new__(cls)
        result.blocks = _positive_integer(specification["blocks"], "Blocks")
        result.radius = _radius(specification["radius"])
        result.moments = normalized_moments(specification["power"], result.radius)
        if specification["moments"] != result.moments:
            raise ValueError(
                "Private archived moments disagree with exact public normalization"
            )
        result.m = result.moments["power"]
        result.seed = int(specification["seed"])
        projectors = np.array(specification["projectors"], dtype=np.float64)
        if (
            projectors.shape != (result.blocks, 3, 3)
            or not np.isfinite(projectors).all()
        ):
            raise ValueError("Invalid private projectors")
        if not (
            np.allclose(projectors, projectors.transpose(0, 2, 1), atol=1e-12, rtol=0)
            and np.allclose(projectors @ projectors, projectors, atol=1e-12, rtol=0)
            and np.allclose(
                np.trace(projectors, axis1=1, axis2=2), 2, atol=1e-12, rtol=0
            )
        ):
            raise ValueError("Private matrices must be rank-two orthogonal projectors")
        if array_sha256(projectors) != specification["projectors_sha256"]:
            raise ValueError("Private projector hash mismatch")
        projectors.flags.writeable = False
        result._projectors = projectors
        return result


def teacher(m, seed, blocks=4, radius=0.5):
    return RadialTeacher(m, seed, blocks, radius)


def evaluate(private_teacher, x):
    """Generation/audit only; deliberately requires the private state."""
    if isinstance(private_teacher, dict):
        private_teacher = RadialTeacher.from_specification(private_teacher)
    return private_teacher.evaluate(x)


def sample_data(private_teacher, seed, train_n=2048, validation_n=2048, test_n=8192):
    """Convenience coordinator; returned arrays contain no private metadata."""
    output = {}
    for name, n, stream in (
        ("train", train_n, 301),
        ("validation", validation_n, 303),
        ("test", test_n, 305),
    ):
        split_seed = int(
            np.random.SeedSequence([int(seed), stream]).generate_state(1)[0]
        )
        x = sample_inputs(split_seed, n, private_teacher.blocks, private_teacher.radius)
        output[f"x_{name}"] = x
        output[f"y_{name}"] = private_teacher.evaluate(x)
    return output

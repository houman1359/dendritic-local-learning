"""Private, continuously varied positive three-cosine response functions.

The scalar-observation and normalization API is shared with the original task.
Neither response coefficients nor analytic population quantities enter fitting.
The family fixes three separated lines but varies their angles, frequencies and
amplitudes independently of the ambient plane and observations. No rejection or
screening by initializer alignment, error, or analytic floor is performed.
"""

from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from . import cosine_learning_tasks as original

FAMILY = "positive_three_cosine_v1"


class CosineFamilyTeacher(original.ThreeCosineTeacher):
    """Independent ambient orientation and latent response random streams."""

    def __init__(self, config, teacher_seed, response_seed):
        super().__init__(config, teacher_seed)
        self.response_seed = original._integer(response_seed, "Response seed")
        rng = np.random.default_rng(
            np.random.SeedSequence([self.response_seed, 7411])
        )
        phase = rng.uniform(0, math.pi)
        angles = phase + np.arange(3) * math.pi / 3 + rng.uniform(
            -math.pi / 12, math.pi / 12, 3
        )
        frequencies = rng.uniform(0.75, 1.25, 3)
        amplitudes = rng.uniform(0.75, 1.25, 3)
        amplitudes /= np.linalg.norm(amplitudes)
        self._set_response(
            np.column_stack((np.cos(angles), np.sin(angles)))
            * frequencies[:, None],
            amplitudes,
        )

    def _set_response(self, frequencies, amplitudes):
        self._frequencies = np.array(frequencies, dtype=np.float64, copy=True)
        self._amplitudes = np.array(amplitudes, dtype=np.float64, copy=True)
        self._frequencies.flags.writeable = False
        self._amplitudes.flags.writeable = False

    def evaluate(self, x):
        z = self.private_latent(x)
        phases = np.einsum("nq,jq->nj", z, self._frequencies, optimize=False)
        return np.sum(np.cos(phases) * self._amplitudes[None, :], axis=1)

    def public_specification(self):
        return {
            "schema": "positive_three_cosine_public_task_v1",
            **asdict(self.config),
            "family": FAMILY,
            "blocks": 1,
            "intrinsic_rank": 2,
            "input_law": "Independent N(0,1) raw coordinates, shape [N,1,d].",
            "response": "Sum of three positive weighted cosine ridge profiles.",
            "angle_law": "Uniform common phase [0,pi), offsets j*pi/3, independent jitter Uniform[-pi/12,pi/12].",
            "frequency_law": "Independent Uniform[0.75,1.25] magnitudes.",
            "amplitude_law": "Independent Uniform[0.75,1.25], divided by their Euclidean norm.",
            "teacher_variation": "Independent random ambient plane and continuously varied latent response; no target screening.",
            "labels": "Scalar raw responses and their TRAIN-normalized versions only.",
            "normalization_scope": "Empirical mean and ddof=0 scale from TRAIN only.",
        }

    def specification(self):
        payload = {
            "schema": "positive_three_cosine_private_teacher_v1",
            "config": asdict(self.config),
            "teacher_seed": self.teacher_seed,
            "response_seed": self.response_seed,
            "basis": self._basis.tolist(),
            "basis_sha256": original.array_sha256(self._basis),
            "frequencies": self._frequencies.tolist(),
            "amplitudes": self._amplitudes.tolist(),
        }
        return {**payload, "payload_sha256": original._json_sha256(payload)}

    @classmethod
    def from_specification(cls, specification):
        if specification.get("schema") != "positive_three_cosine_private_teacher_v1":
            raise ValueError("Unknown private response-family teacher schema")
        payload = {k: v for k, v in specification.items() if k != "payload_sha256"}
        if original._json_sha256(payload) != specification.get("payload_sha256"):
            raise ValueError("Private response-family teacher payload hash mismatch")
        # Reproduce the declared seed law instead of accepting arbitrary
        # rehashed targets with different directions or response coefficients.
        result = cls(
            original.CosineTaskConfig(**specification["config"]),
            specification["teacher_seed"],
            specification["response_seed"],
        )
        if result.specification() != specification:
            raise ValueError("Private teacher does not reproduce its declared seed law")
        return result

    def two_direction_raw_floor(self):
        """Fourth-chaos rank floor, evaluated for the archived floating basis.

        The ideal formula is lambda_min(K)^2 / 4!, where
        K_ij=sqrt(c_i*c_j)*(v_i dot v_j)^2 and
        c_j=a_j*exp(-||v_j||^2/2). Numerical evaluation is not an interval
        certificate. The lower bound is for sums of two ridge functions, not
        unrestricted growing-width or deep networks.
        """
        vectors = self._frequencies @ self._basis.T
        gram = vectors @ vectors.T
        coefficients = self._amplitudes * np.exp(-np.diag(gram) / 2)
        kernel = np.sqrt(np.outer(coefficients, coefficients)) * gram**2
        smallest = float(np.linalg.eigvalsh(kernel)[0])
        if not math.isfinite(smallest) or smallest <= 0:
            raise ValueError("Positive fourth-chaos rank-three floor required")
        return smallest**2 / math.factorial(4)


def validate_campaign(config):
    family = config.get("response_family", "fixed_three_cosine")
    if family not in {"fixed_three_cosine", FAMILY}:
        raise ValueError("Unknown cosine response family")
    if family == FAMILY:
        original._integer(config["response_seed_base"], "Response seed base")


def make_teacher(config, case):
    validate_campaign(config)
    task_config = original.CosineTaskConfig(case["dimension"])
    seed = config["teacher_seed_base"] + case["teacher"]
    if config.get("response_family") == FAMILY:
        return CosineFamilyTeacher(
            task_config, seed, config["response_seed_base"] + case["teacher"]
        )
    return original.ThreeCosineTeacher(task_config, seed)


def restore_teacher(specification):
    if specification.get("schema") == "positive_three_cosine_private_teacher_v1":
        return CosineFamilyTeacher.from_specification(specification)
    return original.ThreeCosineTeacher.from_specification(specification)


def raw_floor(teacher):
    if isinstance(teacher, CosineFamilyTeacher):
        return teacher.two_direction_raw_floor()
    if type(teacher) is not original.ThreeCosineTeacher:
        raise TypeError("Unknown cosine teacher")
    return math.exp(-1) * (1 - 1 / math.sqrt(2)) ** 2 / 72

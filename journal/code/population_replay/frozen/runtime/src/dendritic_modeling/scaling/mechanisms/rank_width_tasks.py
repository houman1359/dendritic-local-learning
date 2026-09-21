"""New rank/width tasks; the completed IID teacher implementation is unchanged."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

if __package__:
    from . import rank_study_tasks as original
else:
    import rank_study_tasks as original


@dataclass(frozen=True)
class TaskConfig:
    response: str
    teacher_index: int
    teacher_seed: int


class QuadraticTeacher:
    """Private rotated quadratic on four independent radius-one-half balls.

    Rotations are independent experimental orientation draws. Unlike the mixture
    task, these draws do not change the response shape under the isotropic law.
    The learner sees only scalar labels and raw supplied blocks.
    """

    def __init__(self, config: TaskConfig):
        if config.response != "quadratic":
            raise ValueError("Quadratic teacher requires quadratic response")
        self.config = config
        rng = np.random.default_rng(np.random.SeedSequence([config.teacher_seed, 61]))
        matrices = rng.normal(size=(4, 3, 3))
        rotations = []
        for matrix in matrices:
            q, r = np.linalg.qr(matrix)
            signs = np.where(np.diag(r) < 0, -1.0, 1.0)
            rotations.append(q * signs[None, :])
        self.rotations = np.stack(rotations)

    @staticmethod
    def moments(intrinsic_rank):
        if intrinsic_rank not in (1, 2):
            raise ValueError("Intrinsic rank must be one or two")
        radius, dimension = 0.5, 3
        mean = intrinsic_rank * radius**2 / (dimension + 2)
        c0 = radius**4 / ((dimension + 2) * (dimension + 4))
        variance = 2 * c0 * (intrinsic_rank - intrinsic_rank**2 / (dimension + 2))
        return mean, variance

    def components(self, x, intrinsic_rank):
        x = np.asarray(x, dtype=np.float64)
        if (
            x.ndim != 3
            or x.shape[1:] != (4, 3)
            or not np.isfinite(x).all()
            or np.max(np.linalg.norm(x, axis=2)) > 0.5 + 1e-12
        ):
            raise ValueError("Finite inputs in four radius-one-half balls required")
        mean, variance = self.moments(intrinsic_rank)
        latent = np.einsum("nki,kij->nkj", x, self.rotations)
        values = np.square(latent[:, :, :intrinsic_rank]).sum(axis=2)
        return (values - mean) / np.sqrt(4 * variance)

    def evaluate(self, x, intrinsic_rank):
        return self.components(x, intrinsic_rank).sum(axis=1)

    def specification(self):
        return {
            "response": "quadratic",
            "index": self.config.teacher_index,
            "seed": self.config.teacher_seed,
            "rotations": self.rotations.tolist(),
            "population_block_moments": {
                str(q): dict(zip(("mean", "variance"), self.moments(q), strict=True))
                for q in (1, 2)
            },
            "normalization": "Subtract each block's qR^2/5 and divide by sqrt(4*Var); independent blocks make whole scalar target mean0,var1.",
            "scope": "Quadratic orientation draws on identical four-block ball law. For q2, one accessible projection per rawblock has population normalized MSE at least5/8. Multiple independently directed nodes per block evade that restriction. Rotating the two active axes within their plane does not change this target.",
        }


def make_teacher(config):
    if config.response == "mixture":
        return original.RankTeacher(
            original.TeacherConfig(config.teacher_index, config.teacher_seed, "mixture")
        )
    if config.response == "quadratic":
        return QuadraticTeacher(config)
    raise ValueError("Supported responses are mixture and quadratic")


def generate_data(
    teacher, observation_seed, intrinsic_rank, train_n, endpoint, endpoint_n
):
    if endpoint not in ("validation", "test"):
        raise ValueError("Explicit validation or test endpoint required")
    return original.generate_data(
        teacher, observation_seed, intrinsic_rank, train_n, endpoint, endpoint_n
    )

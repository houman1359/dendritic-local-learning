"""Scalar-label learning bridge for the explicit Gaussian three-cosine target.

Only generation and an explicitly named oracle diagnostic can access the private
plane. The campaign's primary initializer uses signed, empirically centered TRAIN
labels. A separate raw-label Hermite initializer estimates E[y (xx^T-I)] without
centering labels. Neither estimator inserts population moments or uses held-out
observations; the theory distinguishes their finite-sample fluctuations.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        valid = int(value) == value and value >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _canonical_signs(vectors):
    """Choose positive largest-magnitude coordinates; first index breaks ties."""
    result = np.array(vectors, dtype=np.float64, copy=True)
    pivots = result[np.argmax(np.abs(result), axis=0), np.arange(result.shape[1])]
    result *= np.where(pivots < 0, -1.0, 1.0)[None, :]
    return result


@dataclass(frozen=True)
class CosineTaskConfig:
    ambient_dim: int = 3

    def __post_init__(self):
        object.__setattr__(
            self, "ambient_dim", _integer(self.ambient_dim, "Ambient dimension", 2)
        )


def sample_inputs(seed, n, ambient_dim=3):
    """Teacher-independent Gaussian draws with exact sample-size prefixes."""
    seed = _integer(seed, "Input seed")
    n = _integer(n, "Rows", 1)
    ambient_dim = _integer(ambient_dim, "Ambient dimension", 2)
    return np.random.default_rng(np.random.SeedSequence([seed, 7301])).normal(
        size=(n, 1, ambient_dim)
    )


def _inputs(x, ambient_dim=None, minimum_rows=1):
    x = np.asarray(x, dtype=np.float64)
    if (
        x.ndim != 3
        or x.shape[1] != 1
        or x.shape[2] < 2
        or len(x) < minimum_rows
        or (ambient_dim is not None and x.shape[2] != ambient_dim)
        or not np.isfinite(x).all()
    ):
        raise ValueError("Expected finite raw inputs [N,1,d], with d >= 2")
    return x


class ThreeCosineTeacher:
    """Private orthonormal plane; the three latent response profiles are fixed."""

    def __init__(self, config: CosineTaskConfig, teacher_seed):
        if not isinstance(config, CosineTaskConfig):
            raise TypeError("config must be a CosineTaskConfig")
        self.config = config
        self.teacher_seed = _integer(teacher_seed, "Teacher seed")
        rng = np.random.default_rng(np.random.SeedSequence([self.teacher_seed, 7311]))
        basis, triangular = np.linalg.qr(rng.normal(size=(config.ambient_dim, 2)))
        # QR's diagonal convention preserves the intended random-frame law.
        basis *= np.where(np.diag(triangular) < 0, -1.0, 1.0)[None, :]
        self._set_basis(basis)

    def _set_basis(self, basis):
        self._basis = np.array(basis, dtype=np.float64, copy=True)
        self._basis.flags.writeable = False

    def private_latent(self, x):
        # Unoptimized einsum keeps each row's reduction independent of N;
        # BLAS can choose a different dot-product order for short/long prefixes.
        return np.einsum(
            "nd,dq->nq",
            _inputs(x, self.config.ambient_dim)[:, 0, :],
            self._basis,
            optimize=False,
        )

    def evaluate(self, x):
        z = self.private_latent(x)
        return (
            np.cos(z[:, 0])
            + np.cos(z[:, 1])
            + np.cos((z[:, 0] + z[:, 1]) / math.sqrt(2.0))
        ) / math.sqrt(3.0)

    def public_specification(self):
        return {
            "schema": "three_cosine_public_task_v1",
            **asdict(self.config),
            "blocks": 1,
            "intrinsic_rank": 2,
            "input_law": "Independent N(0,1) raw coordinates, shape [N,1,d].",
            "response": "(cos(z1)+cos(z2)+cos((z1+z2)/sqrt(2)))/sqrt(3)",
            "labels": "Scalar raw responses and their TRAIN-normalized versions only.",
            "teacher_variation": "Private random orthonormal plane; fixed latent response.",
            "normalization_scope": "Empirical mean and ddof=0 scale from TRAIN only.",
        }

    def specification(self):
        """Private archive; never an input to generic estimation or fitting."""
        payload = {
            "schema": "three_cosine_private_teacher_v1",
            "config": asdict(self.config),
            "teacher_seed": self.teacher_seed,
            "basis": self._basis.tolist(),
            "basis_sha256": array_sha256(self._basis),
        }
        return {**payload, "payload_sha256": _json_sha256(payload)}

    @classmethod
    def from_specification(cls, specification):
        if specification.get("schema") != "three_cosine_private_teacher_v1":
            raise ValueError("Unknown private teacher schema")
        payload = {k: v for k, v in specification.items() if k != "payload_sha256"}
        if _json_sha256(payload) != specification.get("payload_sha256"):
            raise ValueError("Private teacher payload hash mismatch")
        config = CosineTaskConfig(**specification["config"])
        basis = np.asarray(specification["basis"], dtype=np.float64)
        if (
            basis.shape != (config.ambient_dim, 2)
            or not np.isfinite(basis).all()
            or array_sha256(basis) != specification["basis_sha256"]
            or not np.allclose(basis.T @ basis, np.eye(2), atol=1e-12, rtol=0)
        ):
            raise ValueError("Invalid private orthonormal basis")
        result = cls.__new__(cls)
        result.config = config
        result.teacher_seed = _integer(specification["teacher_seed"], "Teacher seed")
        result._set_basis(basis)
        return result


def fit_train_normalization(y_train_raw):
    y = np.asarray(y_train_raw, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("At least two finite scalar TRAIN labels required")
    mean = float(np.mean(y))
    scale = float(np.sqrt(np.mean((y - mean) ** 2)))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("TRAIN target scale must be positive and finite")
    return {
        "schema": "three_cosine_train_normalization_v1",
        "train_rows": len(y),
        "raw_train_labels_sha256": array_sha256(y),
        "mean": mean,
        "scale": scale,
        "scope": "TRAIN-only empirical mean and ddof=0 scale; unchanged on held-out rows.",
    }


def apply_normalization(y_raw, normalization):
    y = np.asarray(y_raw, dtype=np.float64)
    mean, scale = normalization["mean"], normalization["scale"]
    if (
        y.ndim != 1
        or not np.isfinite(y).all()
        or not math.isfinite(mean)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Finite scalar labels and valid normalization required")
    return (y - mean) / scale


def generate_dataset(
    private_teacher, observation_seed, train_n=2048, validation_n=4096, test_n=0
):
    """Return public observations and receipt, omitting TEST unless requested.

    Raw observations have exact N-prefixes. Each TRAIN prefix receives its own
    empirical normalization; use raw MSE for comparisons across TRAIN sizes.
    Raw labels are scalar observations, not latent coordinates or components.
    """
    observation_seed = _integer(observation_seed, "Observation seed")
    train_n = _integer(train_n, "TRAIN rows", 2)
    validation_n = _integer(validation_n, "Validation rows", 1)
    test_n = _integer(test_n, "TEST rows")
    arrays, split_seeds = {}, {}
    for split, count, stream in (
        ("train", train_n, 7321),
        ("validation", validation_n, 7323),
        ("test", test_n, 7327),
    ):
        if count == 0:
            continue
        seed = int(
            np.random.SeedSequence([observation_seed, stream]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        arrays[f"x_{split}"] = sample_inputs(
            seed, count, private_teacher.config.ambient_dim
        )
        arrays[f"y_{split}_raw"] = private_teacher.evaluate(arrays[f"x_{split}"])
        split_seeds[split] = seed
    normalization = fit_train_normalization(arrays["y_train_raw"])
    for split in split_seeds:
        arrays[f"y_{split}"] = apply_normalization(
            arrays[f"y_{split}_raw"], normalization
        )
    receipt = {
        "schema": "three_cosine_public_observations_v1",
        "task": private_teacher.public_specification(),
        "observation_seed": observation_seed,
        "split_seeds": split_seeds,
        "normalization": normalization,
        "array_hashes": {key: array_sha256(value) for key, value in arrays.items()},
        "test_generated": test_n > 0,
        "raw_mse_conversion": "Raw MSE = normalized MSE times TRAIN scale squared.",
    }
    return arrays, receipt


def second_hermite_matrix(x_train, y_train_raw):
    """Direct TRAIN average y*(xx^T-I), without empirical centering."""
    x = _inputs(x_train, minimum_rows=4)
    y = np.asarray(y_train_raw, dtype=np.float64)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("Finite raw scalar TRAIN labels must match TRAIN rows")
    flat = x[:, 0, :]
    matrix = flat.T @ (flat * y[:, None]) / len(x)
    matrix -= float(np.mean(y)) * np.eye(flat.shape[1])
    return (matrix + matrix.T) / 2.0


def estimate_bank(x_train, y_train_raw, method="signed_second_hermite", seed=0):
    """Generic TRAIN-only estimated or random plane, in wrapper shape [1,d,2].

    Eigenvectors are ordered by decreasing absolute eigenvalue, retaining the
    signed spectrum. Sign conventions are deterministic; repeated eigenvalues
    admit multiple equivalent bases, so cross-LAPACK bit identity is not claimed.
    The centered method uses the existing signed-centered scalar-label moment;
    its finite-N matrix differs from the raw Hermite average. No oracle option
    exists here. Use oracle_bank with an explicit private teacher.
    """
    x = _inputs(x_train, minimum_rows=4)
    y = np.asarray(y_train_raw, dtype=np.float64)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("Finite raw scalar TRAIN labels must match TRAIN rows")
    seed = _integer(seed, "Initializer seed")
    eigenvalues, selected = [], []
    matrix_sha256 = None
    if method in ("signed_second_hermite", "centered_second_moment"):
        if method == "signed_second_hermite":
            matrix = second_hermite_matrix(x, y)
        else:
            flat = x[:, 0, :]
            matrix = flat.T @ (flat * (y - y.mean())[:, None]) / len(x)
            matrix = (matrix + matrix.T) / 2.0
        spectrum, vectors = np.linalg.eigh(matrix)
        order = np.argsort(-np.abs(spectrum), kind="stable")
        vectors = vectors[:, order[:2]]
        eigenvalues = spectrum.tolist()
        selected = spectrum[order[:2]].tolist()
        matrix_sha256 = array_sha256(matrix)
    elif method == "random":
        rng = np.random.default_rng(np.random.SeedSequence([seed, 7331]))
        vectors = np.linalg.qr(rng.normal(size=(x.shape[2], 2)))[0]
    else:
        raise ValueError("Unknown generic initializer; no oracle access is permitted")
    bank = _canonical_signs(vectors)[None, :, :]
    receipt = {
        "schema": "three_cosine_train_initializer_v1",
        "method": method,
        "seed": seed,
        "raw_train_inputs_sha256": array_sha256(x),
        "raw_train_labels_sha256": array_sha256(y),
        "train_rows": len(x),
        "eigenvalues_ascending": eigenvalues,
        "selected_eigenvalues_by_absolute_magnitude": selected,
        "second_hermite_matrix_sha256": matrix_sha256,
        "bank_sha256": array_sha256(bank),
        "bank_shape": list(bank.shape),
        "matrix_formula": {
            "signed_second_hermite": "mean_TRAIN[y_raw*(xx^T-I)]",
            "centered_second_moment": "mean_TRAIN[(y_raw-mean_TRAIN[y_raw])*xx^T]",
            "random": None,
        }[method],
        "scope": "Raw TRAIN scalar observations only; no population moments. Centered method is the existing signed-centered recipe with absolute eigenvalue ordering; raw Hermite method is distinct at finite N. Random initialization ignores labels after validation.",
    }
    return bank, receipt


def oracle_bank(private_teacher, seed=0):
    """True-plane diagnostic, with independently randomized in-plane frame.

    The teacher's two columns coincide with two response directions. Randomizing
    the coordinate frame avoids handing the generic coverage rule those exact
    directions in addition to the plane. This remains a private-information arm.
    """
    if not isinstance(private_teacher, ThreeCosineTeacher):
        raise TypeError("Oracle diagnostic requires an explicit private teacher")
    seed = _integer(seed, "Oracle rotation seed")
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7337]))
    angle = rng.uniform(0.0, 2.0 * math.pi)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    bank = (private_teacher._basis @ rotation)[None, :, :]
    return bank, {
        "schema": "three_cosine_private_oracle_initializer_v1",
        "method": "oracle_private_plane",
        "rotation_seed": seed,
        "bank_sha256": array_sha256(bank),
        "bank_shape": list(bank.shape),
        "scope": "Private true plane only, with an independent uniform in-plane rotation. No exact response directions supplied to generic coverage; unavailable to generic fitting or selection.",
    }


def bank_frame_rotation_angle(*, seed_base, teacher, ambient_dim):
    """Independent frame angle shared across matched budgets and geometries.

    The teacher identifier and dimension index the stream; no observations,
    target coefficients, estimated basis, or model/allocation seed enters it.
    Different initializer methods receive this same angle in their respective
    basis coordinates, which need not be the same ambient coordinate frame.
    """
    seed_base = _integer(seed_base, "Bank frame rotation seed base")
    teacher = _integer(teacher, "Teacher identifier")
    ambient_dim = _integer(ambient_dim, "Ambient dimension", 2)
    rng = np.random.default_rng(
        np.random.SeedSequence([seed_base, teacher, ambient_dim, 7351])
    )
    return float(rng.uniform(0.0, math.pi))


def rotate_bank_frame(bank, receipt, *, seed_base, teacher, ambient_dim):
    """Rotate an already obtained bank before rebuilding generic hinge features.

    This preserves the supplied plane and adds no trainable scalar. It changes
    coverage directions and TRAIN quantile knots, with no compensating feature
    rotation. Generic banks remain generic; the explicit oracle arm stays a
    private-information diagnostic. Existing bank extraction is unchanged.
    """
    angle = bank_frame_rotation_angle(
        seed_base=seed_base, teacher=teacher, ambient_dim=ambient_dim
    )
    bank = np.asarray(bank, dtype=np.float64)
    if (
        bank.shape != (1, int(ambient_dim), 2)
        or not np.isfinite(bank).all()
        or not np.allclose(bank.swapaxes(-1, -2) @ bank, np.eye(2), atol=1e-12, rtol=0)
    ):
        raise ValueError("Finite orthonormal bank with shape [1,d,2] required")
    if receipt.get("method") not in {
        "centered_second_moment",
        "signed_second_hermite",
        "random",
        "oracle_private_plane",
    }:
        raise ValueError("Known generic or explicit oracle initializer required")
    if receipt.get("bank_sha256") != array_sha256(bank):
        raise ValueError("Source bank differs from initializer receipt")
    if "bank_frame_rotation" in receipt:
        raise ValueError("A bank frame rotation may be applied only once")
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
    )
    rotated = bank @ rotation
    return rotated, {
        **receipt,
        "bank_sha256": array_sha256(rotated),
        "bank_frame_rotation": {
            "schema": "three_cosine_common_bank_frame_rotation_v1",
            "seed_base": int(seed_base),
            "teacher": int(teacher),
            "ambient_dim": int(ambient_dim),
            "stream_tag": 7351,
            "angle_radians": angle,
            "angle_law": "Uniform[0,pi), SeedSequence([seed_base,teacher,ambient_dim,7351])",
            "source_bank_sha256": receipt["bank_sha256"],
            "scope": "One angle shared across geometries, budgets and initializer methods for a teacher/dimension. Methods rotate in their respective bank frames, which need not coincide. Plane extraction, allocation seed and scalar parameter inventory are unchanged; coverage and TRAIN quantile knots are rebuilt. No target or held-out information enters this rotation. Oracle provenance remains private.",
        },
    }

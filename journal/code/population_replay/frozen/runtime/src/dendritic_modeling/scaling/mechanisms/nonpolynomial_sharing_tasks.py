"""Unknown response families for a new shared-subspace scaling study.

Teacher state is generation/audit-only. Public observations contain raw inputs
and scalar labels, never latent coordinates, component labels, profiles, true
supports or teacher seeds. Inputs are iid standard Gaussian in observed
coordinates: their law neither depends on the teacher nor reveals its blocks.
This differs from the older bounded-ball task, whose approximation certificate
must not be transferred to these families without an additional argument.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np


def _integer(value, name, minimum=1):
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
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _json_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class SharingTaskConfig:
    family: str = "fourier_ridges"
    ambient_dim: int = 3
    intrinsic_rank: int = 2
    blocks: int = 4
    directions: int = 8
    harmonics: int = 4
    bumps: int = 6
    smoothness: float = 2.0
    frequency_scale: float = 1.0
    support: str = "known"

    def __post_init__(self):
        if self.family not in ("fourier_ridges", "smooth_bumps", "radial_bumps"):
            raise ValueError("Unknown nonpolynomial family")
        for name in (
            "ambient_dim",
            "intrinsic_rank",
            "blocks",
            "directions",
            "harmonics",
            "bumps",
        ):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        if self.intrinsic_rank > self.ambient_dim:
            raise ValueError("Intrinsic rank must not exceed ambient dimension")
        if self.family == "fourier_ridges" and self.directions < self.intrinsic_rank:
            raise ValueError("Fourier directional bank must span the intrinsic rank")
        for name in ("smoothness", "frequency_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.support not in ("known", "scrambled"):
            raise ValueError("Support must be known or scrambled")
        if self.support == "scrambled" and (self.blocks < 2 or self.ambient_dim < 2):
            raise ValueError(
                "Scrambled support requires at least two blocks and two coordinates per block"
            )


def sample_inputs(seed, n, blocks=4, ambient_dim=3):
    """Teacher-independent iid N(0,1) inputs; sample sizes have exact prefixes."""
    seed = _integer(seed, "Seed", 0)
    n, blocks, ambient_dim = (
        _integer(n, "Rows"),
        _integer(blocks, "Blocks"),
        _integer(ambient_dim, "Ambient dimension"),
    )
    return np.random.default_rng(np.random.SeedSequence([seed, 7101])).normal(
        size=(n, blocks, ambient_dim)
    )


class NonpolynomialSharingTeacher:
    """Private random basis AND profile draw, independent of observation seeds.

    ``fourier_ridges`` sums smooth random ridge profiles in a common q-space.
    ``smooth_bumps`` uses anisotropic off-center q-dimensional Gaussian bumps.
    ``radial_bumps`` uses a random mixture of centered isotropic Gaussian bumps.
    No family is claimed to uniquely favor a dendritic implementation.
    """

    def __init__(self, config: SharingTaskConfig, teacher_seed):
        self.config = config
        self.teacher_seed = _integer(teacher_seed, "Teacher seed", 0)
        blocks, dimension, rank = (
            config.blocks,
            config.ambient_dim,
            config.intrinsic_rank,
        )
        basis_rng = np.random.default_rng(
            np.random.SeedSequence([self.teacher_seed, 7111])
        )
        basis = []
        for _ in range(blocks):
            q, r = np.linalg.qr(basis_rng.normal(size=(dimension, rank)))
            basis.append(q * np.where(np.diag(r) < 0, -1.0, 1.0)[None, :])
        profile_rng = np.random.default_rng(
            np.random.SeedSequence([self.teacher_seed, 7113])
        )
        arrays = {"basis": np.stack(basis)}
        if config.family == "fourier_ridges":
            directions = profile_rng.normal(size=(blocks, config.directions, rank))
            directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
            # A declared full-rank bank prevents an accidental rank-deficient task.
            # Its physical directions are still private and uniformly rotated.
            directions[:, :rank, :] = np.eye(rank)[None, :, :]
            coefficients = profile_rng.normal(
                size=(blocks, config.directions, config.harmonics, 2)
            )
            coefficients /= (
                np.arange(1, config.harmonics + 1)[None, None, :, None]
                ** config.smoothness
            )
            coefficients /= np.linalg.norm(coefficients, axis=(2, 3), keepdims=True)
            arrays.update(
                directions=directions,
                coefficients=coefficients,
                frequencies=config.frequency_scale
                * profile_rng.uniform(0.7, 1.3, size=(blocks, config.directions, 1))
                * np.arange(1, config.harmonics + 1)[None, None, :],
            )
        else:
            centers = profile_rng.uniform(-0.8, 0.8, size=(blocks, config.bumps, rank))
            widths = profile_rng.uniform(0.35, 1.4, size=(blocks, config.bumps, rank))
            if config.family == "radial_bumps":
                centers[:] = 0
                widths[:] = widths[:, :, :1]
            weights = profile_rng.normal(size=(blocks, config.bumps))
            weights /= np.linalg.norm(weights, axis=1, keepdims=True)
            arrays.update(centers=centers, widths=widths, weights=weights)
        permutation = np.arange(blocks * dimension, dtype=np.int64)
        if config.support == "scrambled":
            support_rng = np.random.default_rng(
                np.random.SeedSequence([self.teacher_seed, 7117])
            )
            # Resample until at least one true block crosses observed groups.
            while True:
                permutation = support_rng.permutation(blocks * dimension)
                inverse = np.argsort(permutation)
                memberships = inverse.reshape(blocks, dimension) // dimension
                if any(len(np.unique(row)) > 1 for row in memberships):
                    break
        arrays["observed_permutation"] = permutation
        self._set_arrays(arrays)

    def _set_arrays(self, arrays):
        self._arrays = {}
        for name, value in arrays.items():
            value = np.array(value, copy=True)
            value.flags.writeable = False
            self._arrays[name] = value
        self._inverse_permutation = np.argsort(self._arrays["observed_permutation"])
        self._inverse_permutation.flags.writeable = False

    def _inputs(self, x):
        x = np.asarray(x, dtype=np.float64)
        if (
            x.ndim != 3
            or x.shape[1:] != (self.config.blocks, self.config.ambient_dim)
            or not len(x)
        ):
            raise ValueError("Expected nonempty raw inputs [N, blocks, ambient_dim]")
        if not np.isfinite(x).all():
            raise ValueError("Inputs must be finite")
        return x

    def private_latent(self, x):
        """Private oracle coordinates; never included in public observations."""
        x = self._inputs(x)
        canonical = x.reshape(len(x), -1)[:, self._inverse_permutation].reshape(x.shape)
        return np.einsum("nkd,kdq->nkq", canonical, self._arrays["basis"])

    def private_components(self, x):
        """Private unnormalized block labels, for generation and audits only."""
        latent = self.private_latent(x)
        arrays = self._arrays
        if self.config.family == "fourier_ridges":
            projected = np.einsum("nkq,klq->nkl", latent, arrays["directions"])
            phase = projected[:, :, :, None] * arrays["frequencies"][None, :, :, :]
            values = (
                np.cos(phase) * arrays["coefficients"][None, :, :, :, 0]
                + np.sin(phase) * arrays["coefficients"][None, :, :, :, 1]
            ).sum(axis=(2, 3)) / math.sqrt(self.config.directions)
        else:
            offsets = (
                latent[:, :, None, :] - arrays["centers"][None, :, :, :]
            ) / arrays["widths"][None, :, :, :]
            bumps = np.exp(-0.5 * np.square(offsets).sum(axis=-1))
            values = (bumps * arrays["weights"][None, :, :]).sum(axis=-1)
        return values / math.sqrt(self.config.blocks)

    def evaluate(self, x):
        return self.private_components(x).sum(axis=1)

    def public_specification(self):
        """Declared family assumptions, excluding all recoverable teacher state."""
        return {
            "schema": "nonpolynomial_sharing_public_task_v1",
            **asdict(self.config),
            "input_law": "Independent N(0,1) observed coordinates; covariance I at every ambient dimension.",
            "labels": "Scalar target only, centered and scaled from declared TRAIN observations only.",
            "teacher_variation": "Independent draws vary basis, directions, and response coefficients or bump shapes.",
            "support_scope": (
                "True additive blocks coincide with supplied raw blocks; active subspaces remain private."
                if self.config.support == "known"
                else "True coordinates are privately permuted across supplied raw blocks; all learners see the same scrambled coordinates."
            ),
            "profile_scope": "Family, rank and profile complexity are public; coefficients, true directions, centers, widths and teacher seed are private.",
            "approximation_scope": "No lower bound or P^-4 certificate is asserted for this Gaussian-input task.",
        }

    def specification(self):
        """Private archive. Keep separate from fitting inputs and public metadata."""
        payload = {
            "schema": "nonpolynomial_sharing_private_teacher_v1",
            "config": asdict(self.config),
            "teacher_seed": self.teacher_seed,
            "arrays": {name: value.tolist() for name, value in self._arrays.items()},
            "array_hashes": {
                name: array_sha256(value) for name, value in self._arrays.items()
            },
        }
        return {**payload, "payload_sha256": _json_sha256(payload)}

    @classmethod
    def from_specification(cls, specification):
        if specification.get("schema") != "nonpolynomial_sharing_private_teacher_v1":
            raise ValueError("Unknown private teacher schema")
        payload = {
            key: value
            for key, value in specification.items()
            if key != "payload_sha256"
        }
        if _json_sha256(payload) != specification.get("payload_sha256"):
            raise ValueError("Private teacher payload hash mismatch")
        config = SharingTaskConfig(**specification["config"])
        result = cls.__new__(cls)
        result.config = config
        result.teacher_seed = _integer(specification["teacher_seed"], "Teacher seed", 0)
        shapes = {
            "basis": (config.blocks, config.ambient_dim, config.intrinsic_rank),
            "observed_permutation": (config.blocks * config.ambient_dim,),
        }
        if config.family == "fourier_ridges":
            shapes.update(
                directions=(config.blocks, config.directions, config.intrinsic_rank),
                coefficients=(config.blocks, config.directions, config.harmonics, 2),
                frequencies=(config.blocks, config.directions, config.harmonics),
            )
        else:
            shapes.update(
                centers=(config.blocks, config.bumps, config.intrinsic_rank),
                widths=(config.blocks, config.bumps, config.intrinsic_rank),
                weights=(config.blocks, config.bumps),
            )
        if set(specification["arrays"]) != set(shapes) or set(
            specification["array_hashes"]
        ) != set(shapes):
            raise ValueError("Private teacher array inventory differs from family")
        arrays = {}
        for name, shape in shapes.items():
            value = np.asarray(
                specification["arrays"][name],
                dtype=np.int64 if name == "observed_permutation" else np.float64,
            )
            if (
                value.shape != shape
                or not np.isfinite(value).all()
                or array_sha256(value) != specification["array_hashes"][name]
            ):
                raise ValueError(f"Invalid private array: {name}")
            arrays[name] = value
        basis = arrays["basis"]
        if not np.allclose(
            basis.transpose(0, 2, 1) @ basis,
            np.eye(config.intrinsic_rank),
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("Private basis must have orthonormal columns")
        if not np.array_equal(
            np.sort(arrays["observed_permutation"]),
            np.arange(config.blocks * config.ambient_dim),
        ):
            raise ValueError("Private support map must be a permutation")
        if config.support == "known" and not np.array_equal(
            arrays["observed_permutation"],
            np.arange(config.blocks * config.ambient_dim),
        ):
            raise ValueError("Known support must use the identity map")
        if "widths" in arrays and np.any(arrays["widths"] <= 0):
            raise ValueError("Private bump widths must be positive")
        result._set_arrays(arrays)
        return result


def fit_train_normalization(y_train):
    """Fit population-style empirical variance using TRAIN labels alone."""
    y_train = np.asarray(y_train, dtype=np.float64)
    if y_train.ndim != 1 or len(y_train) < 2 or not np.isfinite(y_train).all():
        raise ValueError("At least two finite scalar TRAIN labels required")
    mean = float(y_train.mean())
    scale = float(np.sqrt(np.mean(np.square(y_train - mean))))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("TRAIN target variance must be positive and finite")
    return {
        "schema": "nonpolynomial_sharing_train_normalization_v1",
        "train_rows": len(y_train),
        "raw_train_labels_sha256": array_sha256(y_train),
        "mean": mean,
        "scale": scale,
        "scope": "Mean and ddof=0 scale from TRAIN labels only; reuse unchanged on validation and test.",
    }


def apply_normalization(y, normalization):
    y = np.asarray(y, dtype=np.float64)
    mean, scale = normalization["mean"], normalization["scale"]
    if (
        not np.isfinite(y).all()
        or not math.isfinite(mean)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Finite labels and valid normalization required")
    return (y - mean) / scale


def generate_dataset(
    private_teacher, observation_seed, train_n=2048, validation_n=8192, test_n=32768
):
    """Return public arrays and a separate public receipt; no private state.

    For nested N comparisons raw inputs/labels have exact prefixes. Normalized
    labels differ when TRAIN N changes because each dataset refits its own
    TRAIN-only scale. Set ``test_n=0`` to withhold TEST completely. The recorded
    mean/scale recover raw-unit MSE for valid
    sample-size comparisons. Use one frozen TRAIN calibration if a common scale
    across N is required; never calibrate on validation/test.
    """
    observation_seed = _integer(observation_seed, "Observation seed", 0)
    _integer(train_n, "TRAIN rows", 2)
    test_n = _integer(test_n, "TEST rows", 0)
    arrays, raw_labels, seeds = {}, {}, {}
    config = private_teacher.config
    for split, n, stream in (
        ("train", train_n, 7121),
        ("validation", validation_n, 7123),
        ("test", test_n, 7127),
    ):
        if split == "test" and n == 0:
            continue
        split_seed = int(
            np.random.SeedSequence([observation_seed, stream]).generate_state(
                1, dtype=np.uint64
            )[0]
        )
        x = sample_inputs(split_seed, n, config.blocks, config.ambient_dim)
        arrays[f"x_{split}"] = x
        raw_labels[split] = private_teacher.evaluate(x)
        seeds[split] = split_seed
    normalization = fit_train_normalization(raw_labels["train"])
    for split, y in raw_labels.items():
        arrays[f"y_{split}"] = apply_normalization(y, normalization)
    receipt = {
        "schema": "nonpolynomial_sharing_observations_v1",
        "task": private_teacher.public_specification(),
        "observation_seed": observation_seed,
        "split_seeds": seeds,
        "normalization": normalization,
        "array_hashes": {name: array_sha256(value) for name, value in arrays.items()},
        "test_generated": test_n > 0,
        "normalization_comparison": "Convert normalized MSE to raw MSE by multiplying by scale squared when comparing independently normalized TRAIN sizes.",
    }
    return arrays, receipt


def sample_data(
    private_teacher, observation_seed, train_n=2048, validation_n=8192, test_n=32768
):
    """Array-only convenience API; use generate_dataset to archive normalization."""
    return generate_dataset(
        private_teacher, observation_seed, train_n, validation_n, test_n
    )[0]


def parameter_inventory(
    branches, architecture, *, blocks=4, ambient_dim=3, rank=2, directions=1
):
    """Count all stored FP64 slots, including fixed soma slots and gauge freedom.

    ``shared_rank`` and ``low_rank_bottleneck`` are the SAME function class and
    stored parameterization. A mapping between them is an equivalence audit,
    not evidence of a distinct dendritic advantage. Fixed support/group maps
    are accounted separately as int64 storage, never as trainable parameters.
    """
    branches, blocks, ambient_dim, rank, directions = (
        _integer(branches, "Branches"),
        _integer(blocks, "Blocks"),
        _integer(ambient_dim, "Ambient dimension"),
        _integer(rank, "Rank"),
        _integer(directions, "Directions"),
    )
    if architecture not in (
        "independent",
        "shared_rank",
        "low_rank_bottleneck",
        "fixed_directions",
    ):
        raise ValueError("Unknown architecture")
    if rank > ambient_dim:
        raise ValueError("Rank must not exceed ambient dimension")
    nodes = blocks * directions if architecture == "fixed_directions" else blocks
    if branches < nodes:
        raise ValueError("Every node requires at least one branch")
    if architecture == "independent":
        branch_cost, overhead = ambient_dim + 2, blocks + 1
    elif architecture == "fixed_directions":
        branch_cost, overhead = 3, blocks * directions * (ambient_dim + 1) + 1
    else:
        branch_cost, overhead = rank + 2, blocks * (ambient_dim * rank + 1) + 1
    stored = branch_cost * branches + overhead
    integer_entries = branches + nodes + blocks * ambient_dim
    return {
        "architecture": architecture,
        "branches": branches,
        "blocks": blocks,
        "ambient_dim": ambient_dim,
        "rank": rank,
        "directions": directions,
        "nodes": nodes,
        "branch_parameter_cost": branch_cost,
        "fixed_parameter_overhead": overhead,
        "stored_parameters": stored,
        "fixed_soma_parameter_slots": nodes,
        "gradient_parameter_slots": stored - nodes,
        "support_integer_entries": blocks * ambient_dim,
        "branch_group_integer_entries": branches,
        "node_map_integer_entries": nodes,
        "stored_float64_bytes": 8 * stored,
        "stored_int64_bytes": 8 * integer_entries,
        "total_counted_bytes": 8 * (stored + integer_entries),
        "scope": "Unconstrained stored projection slots counted; no gauge/rank redundancy discount. Fixed integer maps counted separately from floating parameters.",
    }


def branches_under_ceiling(ceiling, architecture, **geometry):
    ceiling = _integer(ceiling, "Ceiling")
    blocks = _integer(geometry.get("blocks", 4), "Blocks")
    directions = _integer(geometry.get("directions", 1), "Directions")
    minimum = blocks * directions if architecture == "fixed_directions" else blocks
    inventory = parameter_inventory(minimum, architecture, **geometry)
    if ceiling < inventory["stored_parameters"]:
        raise ValueError("Parameter ceiling cannot fit every declared node")
    return (ceiling - inventory["fixed_parameter_overhead"]) // inventory[
        "branch_parameter_cost"
    ]


def shared_to_bottleneck_state(
    basis, gain, threshold, internal_readout, soma_readout, group, bias=0.0
):
    """Map a shared node to ordinary blockwise Linear(d,q)-Linear(q,M)-readout.

    No bias/nonlinearity is inserted after the first linear bottleneck. Doing
    either changes the architecture. Soma slots remain explicit and counted.
    """
    basis, gain, threshold, internal_readout, soma_readout = [
        np.asarray(value, dtype=np.float64)
        for value in (basis, gain, threshold, internal_readout, soma_readout)
    ]
    raw_group = np.asarray(group)
    if raw_group.ndim != 1 or not np.issubdtype(raw_group.dtype, np.integer):
        raise ValueError("Branch group must be a one-dimensional integer map")
    group = raw_group.astype(np.int64, copy=True)
    if basis.ndim != 3 or gain.ndim != 2 or not len(group):
        raise ValueError("Expected basis [K,d,q], gain [M,q] and nonempty group")
    blocks, dimension, rank = basis.shape
    branches = len(group)
    if (
        gain.shape != (branches, rank)
        or threshold.shape != (branches,)
        or internal_readout.shape != (branches,)
        or soma_readout.shape != (blocks,)
    ):
        raise ValueError("Shared-state shapes disagree")
    if not (
        np.isfinite(bias)
        and all(
            np.isfinite(value).all()
            for value in (basis, gain, threshold, internal_readout, soma_readout)
        )
    ):
        raise ValueError("Shared-state floats must be finite")
    if np.any(group < 0) or np.any(group >= blocks) or len(np.unique(group)) != blocks:
        raise ValueError("Every block must receive branches through a valid group map")
    return {
        "schema": "ordinary_shared_bottleneck_equivalence_v1",
        "linear_bottleneck_weight": basis.transpose(0, 2, 1).copy(),
        "branch_weight": gain.copy(),
        "branch_bias": threshold.copy(),
        "branch_readout": internal_readout.copy(),
        "soma_readout": soma_readout.copy(),
        "group": group,
        "bias": float(bias),
        "output_divisor": math.sqrt(blocks),
        "inventory": parameter_inventory(
            branches,
            "low_rank_bottleneck",
            blocks=blocks,
            ambient_dim=dimension,
            rank=rank,
        ),
    }


def bottleneck_predict(state, x, activation="relu"):
    """Ordinary blockwise MLP evaluation of the equivalence-map output."""
    x = np.asarray(x, dtype=np.float64)
    weights = state["linear_bottleneck_weight"]
    if (
        x.ndim != 3
        or x.shape[1:] != (weights.shape[0], weights.shape[2])
        or not np.isfinite(x).all()
    ):
        raise ValueError("Invalid raw inputs for ordinary bottleneck")
    output = np.full(len(x), state["bias"], dtype=np.float64)
    for block, weight in enumerate(weights):
        selected = np.flatnonzero(state["group"] == block)
        hidden = (x[:, block, :] @ weight.T) @ state["branch_weight"][
            selected
        ].T + state["branch_bias"][selected]
        if activation == "relu":
            hidden = np.maximum(hidden, 0)
        elif activation == "tanh":
            hidden = np.tanh(hidden)
        elif activation == "shunt":
            excitation = np.maximum(hidden, 0)
            hidden = excitation / (1 + excitation)
        else:
            raise ValueError("Unknown activation")
        output += (
            (hidden @ state["branch_readout"][selected])
            * state["soma_readout"][block]
            / state["output_divisor"]
        )
    return output

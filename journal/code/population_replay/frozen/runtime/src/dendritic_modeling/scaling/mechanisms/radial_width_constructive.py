"""Raw-label plane estimation and an explicit known-profile ReLU learner.

No teacher or private projector enters these APIs. The construction knows the
radial power and uniform-ball normalization. Its native NPZ can be loaded by the
unchanged rank_learning.load_state; it is not a ridge-optimizer fit.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def load_tasks(source_path=None):
    path = Path(
        source_path or Path(__file__).with_name("radial_width_tasks.py")
    ).resolve()
    name = "radial_width_tasks_" + _sha(path)[:20]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ball_stein_kernel(x, radius=0.5):
    """Hessian of (R²-||x||²)², divided by its known population mean."""
    tasks = load_tasks()
    radius = tasks._radius(radius)
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1:] != (3,) or not np.isfinite(x).all():
        raise ValueError("Finite three-dimensional inputs required")
    squared = np.sum(x * x, axis=-1)
    if np.any(squared > radius**2 * (1 + 2e-12)):
        raise ValueError("Input outside the declared ball")
    raw = 8 * x[..., :, None] * x[..., None, :] - 4 * (radius**2 - squared)[
        ..., None, None
    ] * np.eye(3)
    return raw / (8 * radius**4 / 35)


def _signed_bank(bank):
    bank = np.array(bank, dtype=np.float64, copy=True)
    for block in range(len(bank)):
        for j in range(2):
            row = np.argmax(np.abs(bank[block, :, j]))
            if bank[block, row, j] < 0:
                bank[block, :, j] *= -1
    return bank


def estimate_planes(train_x, train_y, *, radius=0.5):
    """TRAIN raw-label moments; select the two largest algebraic eigenvalues."""
    tasks = load_tasks()
    x = np.asarray(train_x, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("TRAIN inputs must have shape [N,K,3]")
    radius = tasks._radius(radius)
    x = tasks.validate_inputs(x, x.shape[1], radius)
    y = np.asarray(train_y, dtype=np.float64)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError("One finite scalar TRAIN label per observation required")
    moment = np.einsum("n,nkij->kij", y, ball_stein_kernel(x, radius)) / len(x)
    values, vectors = np.linalg.eigh(moment)
    values = values[:, ::-1]
    bank = _signed_bank(vectors[:, :, ::-1][:, :, :2])
    receipt = {
        "method": "Raw scalar TRAIN-label normalized ball-Stein moment; top two algebraic eigenvectors.",
        "rows": len(x),
        "blocks": x.shape[1],
        "radius": radius,
        "train_x_sha256": tasks.array_sha256(x),
        "train_y_sha256": tasks.array_sha256(y),
        "train_hash_encoding": "dtype_shape_bytes",
        "moment": moment.tolist(),
        "eigenvalues_descending_algebraic": values.tolist(),
        "active_null_gap": (values[:, 1] - values[:, 2]).tolist(),
        "within_active_gap": (values[:, 0] - values[:, 1]).tolist(),
        "bank_sha256": tasks.array_sha256(bank),
        "centering_or_residualization": "None: raw Y is used without estimating or subtracting any intercept.",
        "tie_policy": "numpy.linalg.eigh column order for exact ties, reversed algebraic order; each vector sign fixed by its largest absolute coordinate (first index on ties).",
        "structural_prior": "Supplied K blocks, uniform three-ball radius and rank two; no profile, degree or teacher state.",
        "source_sha256": _sha(__file__),
    }
    return bank, receipt


def random_planes(seed, blocks=4):
    tasks = load_tasks()
    blocks = tasks._positive_integer(blocks, "Blocks")
    if int(seed) != seed or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 419]))
    bank = _signed_bank(
        np.stack(
            [np.linalg.qr(rng.normal(size=(3, 3)))[0][:, :2] for _ in range(blocks)]
        )
    )
    return bank, {
        "method": "Seeded Gaussian QR random planes; no data or labels.",
        "seed": int(seed),
        "blocks": blocks,
        "bank_sha256": tasks.array_sha256(bank),
        "source_sha256": _sha(__file__),
    }


def spline_population_mean(m, intervals, radius=0.5):
    """Exact rational mean on the unit interval coordinates, then radius scale."""
    tasks = load_tasks()
    m = tasks._positive_integer(m, "Power")
    intervals = tasks._positive_integer(intervals, "Intervals")
    radius = tasks._radius(radius)
    if intervals < 2:
        raise ValueError("At least two intervals ensure every ReLU branch is used")
    result = Fraction(0)
    for j in range(intervals):
        left, right = (
            Fraction(2 * j - intervals, intervals),
            Fraction(2 * (j + 1) - intervals, intervals),
        )
        slope = (right ** (2 * m) - left ** (2 * m)) / (right - left)
        intercept = left ** (2 * m) - slope * left
        result += Fraction(3, 4) * (
            slope * ((right**2 - left**2) / 2 - (right**4 - left**4) / 4)
            + intercept * ((right - left) - (right**3 - left**3) / 3)
        )
    return float(result) * radius ** (2 * m), str(result)


def _validate_bank(bank):
    bank = np.asarray(bank, dtype=np.float64)
    if (
        bank.ndim != 3
        or bank.shape[1:] != (3, 2)
        or len(bank) < 1
        or not np.isfinite(bank).all()
    ):
        raise ValueError("Orthonormal banks must have shape [K,3,2]")
    if not np.allclose(bank.transpose(0, 2, 1) @ bank, np.eye(2), atol=2e-12, rtol=0):
        raise ValueError("Construction requires an orthonormal plane basis")
    return bank


class ConstructiveRadialModel:
    """Pure NumPy predictor with exactly the base rank-two model's stored slots."""

    PARAMETER_KEYS = (
        "basis",
        "branch_coefficients",
        "threshold",
        "internal_readout",
        "soma_readout",
        "bias",
    )

    def __init__(self, arrays, metadata):
        self.arrays = {key: np.array(value, copy=True) for key, value in arrays.items()}
        self.metadata = json.loads(json.dumps(metadata))
        self.capacities = tuple(metadata["capacities"])
        self._validate()

    def _validate(self):
        blocks, branches = len(self.capacities), sum(self.capacities)
        required = {
            "basis": (blocks, 3, 2),
            "branch_coefficients": (branches, 2),
            "threshold": (branches,),
            "internal_readout": (branches,),
            "soma_readout": (blocks,),
            "bias": (),
            "group": (branches,),
        }
        if set(self.arrays) != set(required):
            raise ValueError("Unexpected or missing model arrays")
        for key, shape in required.items():
            value = self.arrays[key]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid model array: {key}")
            if value.dtype != (np.int64 if key == "group" else np.float64):
                raise ValueError(f"Invalid model dtype: {key}")
        if (
            self.metadata["family"] != "relu"
            or self.metadata["architecture"] != "rank2"
        ):
            raise ValueError("Construction schema requires rank-two ReLU")
        if not np.array_equal(
            self.arrays["group"], np.repeat(np.arange(blocks), self.capacities)
        ):
            raise ValueError("Branch grouping mismatch")
        if self.parameter_count != 4 * branches + 7 * blocks + 1:
            raise ValueError("Counted parameter mismatch")

    @property
    def parameter_count(self):
        return sum(self.arrays[key].size for key in self.PARAMETER_KEYS)

    def effective_projection(self):
        a = self.arrays
        return np.einsum("mdr,mr->md", a["basis"][a["group"]], a["branch_coefficients"])

    def features(self, x):
        radius = self.metadata["initialization_receipt"]["profile"]["radius"]
        x = load_tasks().validate_inputs(x, len(self.capacities), radius)
        projection = self.effective_projection()
        pre = np.empty((len(x), len(projection)), dtype=np.float64)
        start = 0
        for block, count in enumerate(self.capacities):
            stop = start + count
            pre[:, start:stop] = x[:, block] @ projection[start:stop].T
            start = stop
        return np.maximum(pre + self.arrays["threshold"], 0)

    def effective_readout(self):
        a = self.arrays
        return (
            a["internal_readout"]
            * a["soma_readout"][a["group"]]
            / math.sqrt(len(self.capacities))
        )

    def components(self, x, *, include_bias=True, batch_size=2048):
        batch_size = load_tasks()._positive_integer(batch_size, "Prediction batch")
        output = []
        for start in range(0, len(x), batch_size):
            weighted = (
                self.features(x[start : start + batch_size]) * self.effective_readout()
            )
            values = np.column_stack(
                [
                    weighted[:, self.arrays["group"] == i].sum(axis=1)
                    for i in range(len(self.capacities))
                ]
            )
            if include_bias:
                values += float(self.arrays["bias"]) / len(self.capacities)
            output.append(values)
        if not output:
            raise ValueError("Nonempty prediction inputs required")
        return np.concatenate(output)

    def predict(self, x, *, batch_size=2048):
        batch_size = load_tasks()._positive_integer(batch_size, "Prediction batch")
        output = [
            self.features(x[start : start + batch_size]) @ self.effective_readout()
            + float(self.arrays["bias"])
            for start in range(0, len(x), batch_size)
        ]
        if not output:
            raise ValueError("Nonempty prediction inputs required")
        return np.concatenate(output)

    __call__ = predict


def construct_model(
    bank, m, *, radius=0.5, intervals=8, estimator_receipt=None, source_bindings=None
):
    """Construct a known-profile predictor from a supplied estimated plane only."""
    tasks = load_tasks()
    bank = _validate_bank(bank)
    profile = tasks.normalized_moments(m, radius)
    m, radius = profile["power"], profile["radius"]
    intervals = tasks._positive_integer(intervals, "Intervals")
    mean_spline, mean_exact = spline_population_mean(m, intervals, radius)
    blocks = len(bank)
    if estimator_receipt is not None and estimator_receipt.get(
        "bank_sha256"
    ) != tasks.array_sha256(bank):
        raise ValueError("Supplied bank does not match its estimator receipt")
    lines = m + 1
    capacity = lines * intervals
    branches = blocks * capacity
    cm = math.comb(2 * m, m) / 4**m
    angles = np.arange(lines) * np.pi / lines
    coefficients = np.repeat(
        np.column_stack([np.cos(angles), np.sin(angles)]), intervals, axis=0
    )
    knots = np.linspace(-radius, radius, intervals + 1)
    slopes = np.diff(knots ** (2 * m)) / np.diff(knots)
    slope_changes = np.concatenate([slopes[:1], np.diff(slopes)])
    if not np.all(slope_changes != 0):
        raise FloatingPointError("Construction lost a functionally used branch")
    arrays = {
        "basis": bank.copy(),
        "branch_coefficients": np.tile(coefficients, (blocks, 1)),
        "threshold": np.tile(-knots[:-1], blocks * lines),
        "internal_readout": np.tile(
            slope_changes / (lines * cm * profile["sigma"]), blocks * lines
        ),
        "soma_readout": np.ones(blocks, dtype=np.float64),
        "bias": np.array(
            math.sqrt(blocks)
            * (radius ** (2 * m) - mean_spline)
            / (cm * profile["sigma"])
        ),
        "group": np.repeat(np.arange(blocks, dtype=np.int64), capacity),
    }
    inventory = {
        "geometry": "rank2",
        "raw_blocks": blocks,
        "inputs_per_block": 3,
        "nodes_per_raw_block": 1,
        "nodes": blocks,
        "branches": branches,
        "branch_parameter_cost": 4,
        "fixed_parameter_overhead": 7 * blocks + 1,
        "stored_parameters": 4 * branches + 7 * blocks + 1,
        "fixed_soma_parameter_slots": blocks,
        "fixed_node_map_integer_entries": 0,
        "fixed_branch_group_integer_entries": branches,
        "capacities": [capacity] * blocks,
        "output_normalization": "Native base internal*soma/sqrt(K), with one node per raw block; no hidden output rescaling.",
        "source_bindings": dict(source_bindings or {}),
    }
    receipt = {
        "method": "Explicit known radial profile; equal-angle line decomposition and centered uniform-knot biased-ReLU interpolation. No optimization or ridge solve.",
        "profile": profile,
        "intervals_per_line": intervals,
        "lines_per_block": lines,
        "one_dimensional_spline_mean": mean_spline,
        "unit_radius_spline_mean_exact": mean_exact,
        "raw_local_approximation_mean": mean_spline / cm,
        "bank_sha256": tasks.array_sha256(bank),
        "direction_bank": {
            "estimator": estimator_receipt,
            "bank_sha256": tasks.array_sha256(bank),
        },
        "_radial_width": inventory,
        "construction_source_sha256": _sha(__file__),
        "learner_source_sha256": _sha(__file__),
        "task_source_sha256": _sha(tasks.__file__),
        "observations_used": "No additional scalar-label queries. The caller supplies the bank; its estimator receipt records any TRAIN exposure. Analytic centering uses only the declared input law.",
        "bank_provenance": (
            "Supplied estimator receipt; its recorded inputs determine the learning claim."
            if estimator_receipt is not None
            else "Externally supplied plane bank with no estimator receipt: representation construction only, no learned-plane claim."
        ),
        "centering": "Known uniform-ball marginal 3*(1-t²/R²)/(4R); constant folded into global bias.",
    }
    metadata = {
        "capacities": [capacity] * blocks,
        "family": "relu",
        "architecture": "rank2",
        "inputs_per_block": 3,
        "threshold_mode": "quantile",
        "initialization_spread": 0.0,
        "initialization_receipt": receipt,
    }
    return ConstructiveRadialModel(arrays, metadata)


def construction_under_ceiling(bank, m, ceiling, *, radius=0.5, **kwargs):
    blocks = len(_validate_bank(bank))
    m = load_tasks()._positive_integer(m, "Power")
    ceiling = load_tasks()._positive_integer(ceiling, "Ceiling")
    intervals = (ceiling - 7 * blocks - 1) // (4 * blocks * (m + 1))
    if intervals < 2:
        raise ValueError("Ceiling cannot fit two intervals per required line")
    return construct_model(bank, m, radius=radius, intervals=intervals, **kwargs)


def save_state(model, path):
    path = Path(path)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            **model.arrays,
            metadata_json=np.array(json.dumps(model.metadata, sort_keys=True)),
        )
    return {
        "path": str(path),
        "sha256": _sha(path),
        "stored_parameters": model.parameter_count,
        "schema": "Native RankBlockModel NPZ; constructive known-profile endpoint, no fit.json or ridge objective.",
    }


def load_state(path):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        arrays = {
            key: archive[key].copy() for key in archive.files if key != "metadata_json"
        }
    return ConstructiveRadialModel(arrays, metadata)

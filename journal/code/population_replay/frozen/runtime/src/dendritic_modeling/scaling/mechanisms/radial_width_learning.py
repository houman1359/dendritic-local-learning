"""Generic plane-covered initializations and counted width controls.

Only supplied TRAIN inputs/scalar labels and an observed or random plane enter
this module. It has no radial-profile, teacher-direction or test-data API. The
unchanged rank-learning model, reduced objective and QR/restart fitter remain
the numerical implementation. Stored fixed soma gauge slots are still counted.
"""

import hashlib
import importlib.util
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

GEOMETRIES = (
    "full",
    "rank2",
    "width1",
    "width2",
    "width3",
    "width4",
    "width5",
    "width8",
)
WIDTH_METADATA_KEY = "_radial_width"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_hash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def train_fingerprints(x, y, encoding="raw_bytes"):
    if encoding not in ("raw_bytes", "dtype_shape_bytes"):
        raise ValueError("Unknown TRAIN-array hash encoding")

    def digest(value):
        value = np.ascontiguousarray(value)
        prefix = (
            str(value.dtype).encode() + str(value.shape).encode()
            if encoding == "dtype_shape_bytes"
            else b""
        )
        return hashlib.sha256(prefix + value.tobytes()).hexdigest()

    return {
        "train_x_sha256": digest(x),
        "train_y_sha256": digest(y),
        "rows": len(x),
        "train_hash_encoding": encoding,
    }


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


@lru_cache(maxsize=8)
def load_module(source_path):
    path = Path(source_path).resolve()
    name = "radial_width_dependency_" + sha256(path)[:20]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_base_module(source_path=None):
    return load_module(source_path or Path(__file__).with_name("rank_learning.py"))


def geometry_spec(geometry):
    if geometry not in GEOMETRIES:
        raise ValueError("Unknown full/rank/width geometry")
    return (
        ("rank1", int(geometry[5:])) if geometry.startswith("width") else (geometry, 1)
    )


def _integer(value, minimum=1):
    if int(value) != value or value < minimum:
        raise ValueError("Whole nonnegative seed or positive geometry count required")
    return int(value)


def parameter_inventory(branches, geometry, *, raw_blocks=4, inputs_per_block=3):
    architecture, width = geometry_spec(geometry)
    branches, raw_blocks, dimension = map(
        _integer, (branches, raw_blocks, inputs_per_block)
    )
    rank = 0 if architecture == "full" else int(architecture[-1])
    if dimension < max(2, rank):
        raise ValueError("The supplied input coordinates must support a plane")
    nodes = raw_blocks * width
    if branches < nodes:
        raise ValueError("Every stored node needs at least one branch")
    cost = dimension + 2 if rank == 0 else rank + 2
    overhead = nodes + 1 + dimension * rank * nodes
    parameters = cost * branches + overhead
    return {
        "geometry": geometry,
        "base_architecture": architecture,
        "raw_blocks": raw_blocks,
        "inputs_per_block": dimension,
        "nodes_per_raw_block": width,
        "nodes": nodes,
        "branches": branches,
        "branch_parameter_cost": cost,
        "fixed_parameter_overhead": overhead,
        "stored_parameters": parameters,
        "fixed_soma_parameter_slots": nodes,
        "gradient_parameter_slots": parameters - nodes,
        "fixed_node_map_integer_entries": nodes,
        "fixed_branch_group_integer_entries": branches,
    }


def branches_under_ceiling(ceiling, geometry, *, raw_blocks=4, inputs_per_block=3):
    ceiling = _integer(ceiling)
    _, width = geometry_spec(geometry)
    minimum = parameter_inventory(
        raw_blocks * width,
        geometry,
        raw_blocks=raw_blocks,
        inputs_per_block=inputs_per_block,
    )
    if ceiling < minimum["stored_parameters"]:
        raise ValueError("Ceiling cannot fit nonempty declared nodes")
    return (ceiling - minimum["fixed_parameter_overhead"]) // minimum[
        "branch_parameter_cost"
    ]


def _balanced(total, groups, seed):
    counts = np.full(groups, total // groups, dtype=np.int64)
    counts[
        np.random.default_rng(np.random.SeedSequence(seed)).permutation(groups)[
            : total % groups
        ]
    ] += 1
    return counts


def allocate_capacities(branches, *, raw_blocks=4, nodes_per_block=1, seed=0):
    branches, raw_blocks, nodes_per_block = map(
        _integer, (branches, raw_blocks, nodes_per_block)
    )
    seed = _integer(seed, 0)
    if branches < raw_blocks * nodes_per_block:
        raise ValueError("Not enough branches for every node")
    totals = _balanced(branches, raw_blocks, [seed, 431])
    return tuple(
        int(c)
        for k, total in enumerate(totals)
        for c in _balanced(int(total), nodes_per_block, [seed, 433, k])
    )


def matched_ceiling_inventory(
    ceilings, *, tolerance=0.01, raw_blocks=4, inputs_per_block=3
):
    rows = []
    for ceiling in ceilings:
        entries = [
            parameter_inventory(
                branches_under_ceiling(
                    ceiling, g, raw_blocks=raw_blocks, inputs_per_block=inputs_per_block
                ),
                g,
                raw_blocks=raw_blocks,
                inputs_per_block=inputs_per_block,
            )
            for g in GEOMETRIES
        ]
        counts = [e["stored_parameters"] for e in entries]
        gap = (max(counts) - min(counts)) / min(counts)
        if gap > tolerance:
            raise ValueError("Actual parameter gap exceeds declared tolerance")
        rows.append(
            {
                "ceiling": ceiling,
                "maximum_relative_pair_gap": gap,
                "geometries": entries,
            }
        )
    return rows


def _train_arrays(train_data):
    if train_data is None or len(train_data) != 2:
        raise ValueError("Unchanged TRAIN inputs and scalar labels required")
    x, y = map(_array, train_data)
    if x.ndim != 3 or x.shape[2] < 2 or len(x) < 4 or y.shape != (len(x),):
        raise ValueError("Expected at least four [N,K,d] rows and scalar TRAIN labels")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite TRAIN inputs and labels required")
    return x, y


def estimate_bank(method, train_data, *, seed=0, radius=0.5, estimation_module=None):
    """Estimate from TRAIN only; random controls also bind the TRAIN prefix."""
    x, y = _train_arrays(train_data)
    estimator = estimation_module or load_module(
        Path(__file__).with_name("radial_width_constructive.py")
    )
    if method == "stein":
        bank, receipt = estimator.estimate_planes(x, y, radius=radius)
    elif method == "random":
        if x.shape[-1] != 3:
            raise ValueError(
                "The declared random-plane estimator uses three input coordinates"
            )
        bank, receipt = estimator.random_planes(_integer(seed, 0), blocks=x.shape[1])
    else:
        raise ValueError("Initializer method must be stein or random")
    receipt = dict(receipt)
    receipt.update(
        **train_fingerprints(x, y),
        source_sha256=sha256(estimator.__file__),
        training_prefix_bound=True,
    )
    return bank, receipt


def _validate_bank(bank, x):
    bank = _array(bank)
    if bank.shape != (x.shape[1], x.shape[2], 2) or not np.isfinite(bank).all():
        raise ValueError("One finite [d,2] plane per original raw block required")
    if not np.allclose(
        bank.transpose(0, 2, 1) @ bank, np.eye(2), atol=1e-10, rtol=1e-10
    ):
        raise ValueError("Supplied direction bank must be orthonormal")
    return bank


def initialized_model(
    capacities,
    family,
    geometry,
    *,
    bank,
    train_data,
    seed=0,
    coverage_lines=8,
    gain_range=(0.1, 30.0),
    initializer_receipt=None,
    base_module=None,
):
    """Build a generic trainable model, independently of any target formula.

    Full and rank2 split each block's branches as evenly as possible among
    coverage_lines angles j*pi/coverage_lines. A widthL arm uses L node angles
    j*pi/L. Every direction gets centered TRAIN quantiles (i+.5)/local_count.
    ReLU uses gain=1/TRAIN span; smooth primitives use centered log-grid gains
    over the same fixed generic range, divided by that span. Singleton groups
    get the geometric midpoint. No target-derived knot, gain or readout enters.
    """
    x, y = _train_arrays(train_data)
    bank = _validate_bank(bank, x)
    seed, coverage_lines = _integer(seed, 0), _integer(coverage_lines)
    architecture, width = geometry_spec(geometry)
    capacities = tuple(_integer(c) for c in capacities)
    blocks, dimension = x.shape[1:]
    if len(capacities) != blocks * width:
        raise ValueError("Capacities must describe all block-major expanded nodes")
    low_gain, high_gain = map(float, gain_range)
    if not (
        math.isfinite(low_gain)
        and math.isfinite(high_gain)
        and 0 < low_gain <= high_gain
    ):
        raise ValueError("Finite positive increasing generic gain range required")
    estimator = dict(initializer_receipt or {"method": "supplied_bank"})
    fingerprints = train_fingerprints(
        x, y, estimator.get("train_hash_encoding", "raw_bytes")
    )
    for key, value in fingerprints.items():
        if key in estimator and estimator[key] != value:
            raise ValueError("Initializer receipt differs from supplied TRAIN prefix")
    estimator.update(fingerprints)
    node_map = np.repeat(np.arange(blocks), width)
    angles = np.tile(np.arange(width) * math.pi / width, blocks)
    node_vectors = np.stack(
        [
            math.cos(a) * bank[k, :, 0] + math.sin(a) * bank[k, :, 1]
            for k, a in zip(node_map, angles, strict=True)
        ]
    )
    base = base_module or load_base_module()
    model = base.RankBlockModel(
        capacities,
        family,
        architecture,
        inputs_per_block=dimension,
        seed=seed,
        train_data=(x[:, node_map, :], y),
        effective_directions=node_vectors,
        initialization_spread=0.0,
        threshold_mode="quantile",
    )
    coefficients, thresholds, coverage = [], [], []
    for node, capacity in enumerate(capacities):
        block = int(node_map[node])
        lines = coverage_lines if architecture != "rank1" else 1
        counts = _balanced(capacity, lines, [seed, 641, block])
        for line, count in enumerate(counts):
            if count == 0:
                continue
            angle = (
                line * math.pi / lines
                if architecture != "rank1"
                else float(angles[node])
            )
            unit = (
                math.cos(angle) * bank[block, :, 0]
                + math.sin(angle) * bank[block, :, 1]
            )
            projected = x[:, block, :] @ unit
            span = max(float(projected.max() - projected.min()), 1e-6)
            quantile_levels = (np.arange(count, dtype=float) + 0.5) / count
            knots = np.quantile(projected, quantile_levels, method="linear")
            gains = (
                np.ones(count)
                if family == "relu"
                else np.exp(
                    math.log(low_gain)
                    + quantile_levels * math.log(high_gain / low_gain)
                )
            ) / span
            if architecture == "rank1":
                coefficients.extend(gains[:, None])
            else:
                coefficients.extend(
                    gains[:, None] * np.array([math.cos(angle), math.sin(angle)])
                )
            thresholds.extend(-gains * knots)
            coverage.append(
                {
                    "node": node,
                    "raw_block": block,
                    "line": line,
                    "angle_radians": angle,
                    "branches": int(count),
                    "train_projection_span": span,
                    "quantile_levels": quantile_levels.tolist(),
                    "knots": knots.tolist(),
                    "gains": gains.tolist(),
                }
            )
    coefficients = torch.tensor(np.asarray(coefficients), dtype=torch.float64)
    with torch.no_grad():
        model.threshold.copy_(torch.tensor(thresholds, dtype=torch.float64))
        if architecture == "rank1":
            model.basis.copy_(
                torch.tensor(node_vectors[:, :, None], dtype=torch.float64)
            )
            model.branch_coefficients.copy_(coefficients)
        else:
            planes = torch.tensor(bank, dtype=torch.float64)
            if architecture == "rank2":
                model.basis.copy_(planes)
                model.branch_coefficients.copy_(coefficients)
            else:
                model.projection.copy_(
                    torch.einsum("mdr,mr->md", planes[model.group], coefficients)
                )
    model.initialization_receipt = {
        "seed": seed,
        "initializer": "TRAIN_plane_coverage",
        "threshold_mode": "quantile",
        "soma_gauge": "Stored soma=1 remains fixed as in unchanged fitter; every stored scalar counted. All body and functional readout parameters remain free.",
        "direction_bank": {
            "sha256": array_hash(bank),
            "basis": bank.tolist(),
            "estimator": estimator,
            "coverage": "uniform_plane_lines",
            "coverage_lines": coverage_lines,
            "gain_range": [low_gain, high_gain],
            "direction_groups": coverage,
            "coverage_rule": "Full/rank2: j*pi/coverage_lines, balanced groups per raw block. WidthL: node j*pi/L. TRAIN-centered quantile positions (i+.5)/count for every primitive; smooth log-grid gains share those fixed positions. No radial/profile information.",
            "matching": "Full/rank2 use identical effective projections, thresholds and readouts at equal capacities and seed. Different width geometries retain unchanged base output normalization; no cross-width identical-feature claim.",
        },
    }
    return RadialWidthModel.from_model(
        model, geometry, raw_blocks=blocks, base_module=base
    )


class RadialWidthModel(nn.Module):
    @classmethod
    def from_model(cls, model, geometry, *, raw_blocks=4, base_module=None):
        architecture, width = geometry_spec(geometry)
        if (
            model.architecture != architecture
            or len(model.capacities) != raw_blocks * width
        ):
            raise ValueError("Base geometry differs from declared fixed mapping")
        wrapper = cls()
        wrapper.geometry, wrapper.raw_blocks, wrapper.nodes_per_block = (
            geometry,
            int(raw_blocks),
            width,
        )
        wrapper.inputs_per_block = model.inputs_per_block
        wrapper.base_module = base_module or sys.modules[type(model).__module__]
        wrapper.model = model
        wrapper.register_buffer(
            "node_to_raw",
            torch.arange(raw_blocks, device=model.bias.device).repeat_interleave(width),
        )
        model.initialization_receipt[WIDTH_METADATA_KEY] = wrapper.counted_metadata()
        return wrapper

    @property
    def core(self):
        return self.model

    @property
    def family(self):
        return self.model.family

    @property
    def parameter_count(self):
        return self.model.parameter_count

    @property
    def gradient_parameter_count(self):
        return self.model.gradient_parameter_count

    def counted_metadata(self):
        inventory = parameter_inventory(
            sum(self.model.capacities),
            self.geometry,
            raw_blocks=self.raw_blocks,
            inputs_per_block=self.inputs_per_block,
        )
        assert inventory["stored_parameters"] == self.parameter_count
        assert inventory["gradient_parameter_slots"] == self.gradient_parameter_count
        return inventory | {
            "capacities": list(self.model.capacities),
            "node_to_raw": self.node_to_raw.detach().cpu().tolist(),
            "output_normalization": "Unchanged base internal*soma/sqrt(expanded_node_count); no hidden output rescaling.",
            "base_source_sha256": sha256(self.base_module.__file__),
            "wrapper_source_sha256": sha256(__file__),
            "representation_scope": "WidthL has L independently trainable rank1 nodes per supplied raw block. Width2/3/4/5/8 can expose multiple directions and do not inherit the single-direction floor. Full/rank2 plane initialization is not a fixed-plane constraint.",
        }

    def metadata(self):
        return self.counted_metadata()

    def _validate_input_shape(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (
            self.raw_blocks,
            self.inputs_per_block,
        ):
            raise ValueError("Expected raw [N,raw_blocks,inputs_per_block] inputs")

    def expanded_inputs(self, raw_x):
        x = torch.as_tensor(
            raw_x, dtype=self.model.bias.dtype, device=self.model.bias.device
        )
        self._validate_input_shape(x)
        return x.index_select(1, self.node_to_raw.to(device=x.device))

    def expanded_numpy(self, raw_x):
        x = _array(raw_x)
        self._validate_input_shape(x)
        return x[:, self.node_to_raw.detach().cpu().numpy(), :]

    def forward(self, raw_x):
        return self.model(self.expanded_inputs(raw_x))

    def predict(self, raw_x):
        return self.forward(raw_x)

    def features(self, raw_x):
        return self.model.features(self.expanded_inputs(raw_x))

    def components(self, raw_x):
        values = self.model.components(self.expanded_inputs(raw_x))
        return values.reshape(len(values), self.raw_blocks, self.nodes_per_block).sum(2)


def model_under_ceiling(
    ceiling,
    family,
    geometry,
    *,
    bank,
    train_data,
    seed=0,
    allocation_seed=0,
    coverage_lines=8,
    gain_range=(0.1, 30.0),
    initializer_receipt=None,
    base_module=None,
    raw_blocks=None,
    inputs_per_block=None,
):
    x, y = _train_arrays(train_data)
    if (raw_blocks is not None and raw_blocks != x.shape[1]) or (
        inputs_per_block is not None and inputs_per_block != x.shape[2]
    ):
        raise ValueError("Declared raw geometry differs from TRAIN inputs")
    _, width = geometry_spec(geometry)
    branches = branches_under_ceiling(
        ceiling, geometry, raw_blocks=x.shape[1], inputs_per_block=x.shape[2]
    )
    capacities = allocate_capacities(
        branches, raw_blocks=x.shape[1], nodes_per_block=width, seed=allocation_seed
    )
    return initialized_model(
        capacities,
        family,
        geometry,
        bank=bank,
        train_data=(x, y),
        seed=seed,
        coverage_lines=coverage_lines,
        gain_range=gain_range,
        initializer_receipt=initializer_receipt,
        base_module=base_module,
    )


def save_state(wrapper, state_path, descriptor_path=None):
    wrapper.model.initialization_receipt[WIDTH_METADATA_KEY] = (
        wrapper.counted_metadata()
    )
    result = wrapper.base_module.save_state(wrapper.model, state_path)
    descriptor = {
        "state_path": str(Path(state_path).resolve()),
        "state_sha256": sha256(state_path),
        "width_metadata": wrapper.counted_metadata(),
    }
    if descriptor_path is not None:
        with Path(descriptor_path).open("x") as f:
            json.dump(descriptor, f, indent=2, allow_nan=False)
            f.write("\n")
    return result | descriptor


def load_state(path, *, base_module=None, device="cpu"):
    path = Path(path)
    descriptor = json.loads(path.read_text()) if path.suffix == ".json" else None
    state_path = Path(descriptor["state_path"]) if descriptor else path
    if descriptor and sha256(state_path) != descriptor["state_sha256"]:
        raise ValueError("Descriptor state hash changed")
    base = base_module or load_base_module()
    model = base.load_state(state_path, device)
    metadata = model.initialization_receipt.get(WIDTH_METADATA_KEY)
    if not isinstance(metadata, dict):
        raise ValueError("Missing radial fixed-map metadata")
    # Copy before from_model installs freshly reconstructed metadata.
    metadata = dict(metadata)
    wrapper = RadialWidthModel.from_model(
        model, metadata["geometry"], raw_blocks=metadata["raw_blocks"], base_module=base
    )
    if metadata != wrapper.counted_metadata():
        raise ValueError(
            "Saved map/counts/source fingerprints differ from reconstruction"
        )
    if descriptor and descriptor["width_metadata"] != metadata:
        raise ValueError("Descriptor metadata differs from stored state")
    return wrapper

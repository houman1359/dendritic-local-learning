"""Counted node-width controls around the unchanged rank-learning implementation.

The wrapper duplicates supplied input blocks by a fixed integer map. Each copy
receives its own independently trainable rank-one node. No target, fitting
objective, trainable pooling, or hidden normalization is added. The base fitter
and base NPZ states remain unchanged and can be used with expanded inputs.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import nn

GEOMETRIES = ("full", "rank2", "width1", "width2", "width4")
WIDTH_METADATA_KEY = "_rank_width"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=8)
def load_base_module(source_path=None):
    """Load the adjacent frozen numerical source, or an explicitly bound source."""
    path = Path(source_path or Path(__file__).with_name("rank_learning.py")).resolve()
    name = "rank_width_base_" + sha256(path)[:20]
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def geometry_spec(geometry):
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown geometry: {geometry}")
    return (
        ("rank1", int(geometry[-1])) if geometry.startswith("width") else (geometry, 1)
    )


def parameter_inventory(branches, geometry, *, raw_blocks=4, inputs_per_block=3):
    base_architecture, nodes_per_block = geometry_spec(geometry)
    nodes = raw_blocks * nodes_per_block
    if any(
        int(value) != value or value < 1
        for value in [raw_blocks, inputs_per_block, branches]
    ):
        raise ValueError("Positive integer geometry and branch counts required")
    if branches < nodes:
        raise ValueError("Every stored node must receive at least one branch")
    rank = 0 if base_architecture == "full" else int(base_architecture[-1])
    branch_cost = inputs_per_block + 2 if rank == 0 else rank + 2
    overhead = nodes + 1 + (inputs_per_block * rank * nodes if rank else 0)
    return {
        "geometry": geometry,
        "base_architecture": base_architecture,
        "raw_blocks": int(raw_blocks),
        "inputs_per_block": int(inputs_per_block),
        "nodes_per_raw_block": nodes_per_block,
        "nodes": nodes,
        "branches": int(branches),
        "branch_parameter_cost": branch_cost,
        "fixed_parameter_overhead": overhead,
        "stored_parameters": branch_cost * int(branches) + overhead,
        "fixed_soma_parameter_slots": nodes,
        "gradient_parameter_slots": branch_cost * int(branches) + overhead - nodes,
        "fixed_node_map_integer_entries": nodes,
        "fixed_branch_group_integer_entries": int(branches),
    }


def branches_under_ceiling(ceiling, geometry, *, raw_blocks=4, inputs_per_block=3):
    _, width = geometry_spec(geometry)
    minimum = parameter_inventory(
        raw_blocks * width,
        geometry,
        raw_blocks=raw_blocks,
        inputs_per_block=inputs_per_block,
    )
    if int(ceiling) != ceiling or ceiling < minimum["stored_parameters"]:
        raise ValueError("Ceiling cannot fit all declared nodes with nonempty branches")
    return (int(ceiling) - minimum["fixed_parameter_overhead"]) // minimum[
        "branch_parameter_cost"
    ]


def allocate_capacities(branches, *, raw_blocks=4, nodes_per_block=1, seed=0):
    """Balance original-block totals first, then node totals; no data access."""
    nodes = raw_blocks * nodes_per_block
    if (
        any(
            int(value) != value or value < 1
            for value in (raw_blocks, nodes_per_block, branches)
        )
        or branches < nodes
    ):
        raise ValueError(
            "Positive integer dimensions and at least one branch per node required"
        )
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 431]))
    block_totals = np.full(raw_blocks, branches // raw_blocks, dtype=np.int64)
    block_totals[rng.permutation(raw_blocks)[: branches % raw_blocks]] += 1
    capacities = []
    for block, total in enumerate(block_totals):
        node_rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), 433, block])
        )
        values = np.full(nodes_per_block, int(total) // nodes_per_block, dtype=np.int64)
        values[
            node_rng.permutation(nodes_per_block)[: int(total) % nodes_per_block]
        ] += 1
        capacities.extend(int(value) for value in values)
    assert (
        len(capacities) == nodes and sum(capacities) == branches and min(capacities) > 0
    )
    return tuple(capacities)


def matched_ceiling_inventory(
    ceilings, *, tolerance=0.01, raw_blocks=4, inputs_per_block=3
):
    result = []
    for ceiling in ceilings:
        entries = [
            parameter_inventory(
                branches_under_ceiling(
                    ceiling,
                    geometry,
                    raw_blocks=raw_blocks,
                    inputs_per_block=inputs_per_block,
                ),
                geometry,
                raw_blocks=raw_blocks,
                inputs_per_block=inputs_per_block,
            )
            for geometry in GEOMETRIES
        ]
        parameters = [entry["stored_parameters"] for entry in entries]
        mismatch = (max(parameters) - min(parameters)) / min(parameters)
        if mismatch > tolerance:
            raise ValueError(
                f"Ceiling {ceiling} has actual parameter mismatch {mismatch:.6g}, above {tolerance}"
            )
        result.append(
            {
                "ceiling": ceiling,
                "maximum_relative_pair_gap": mismatch,
                "geometries": entries,
            }
        )
    return result


class RankWidthModel(nn.Module):
    def __init__(
        self,
        capacities,
        family,
        geometry,
        *,
        raw_blocks=4,
        inputs_per_block=3,
        train_data=None,
        effective_directions=None,
        base_module=None,
        **initialization,
    ):
        super().__init__()
        architecture, nodes_per_block = geometry_spec(geometry)
        if len(capacities) != raw_blocks * nodes_per_block:
            raise ValueError(
                "Capacities must describe every expanded node, in original-block-major order"
            )
        self.geometry = geometry
        self.raw_blocks = int(raw_blocks)
        self.nodes_per_block = nodes_per_block
        self.inputs_per_block = int(inputs_per_block)
        self.base_module = base_module or load_base_module()
        self.register_buffer(
            "node_to_raw", torch.arange(raw_blocks).repeat_interleave(nodes_per_block)
        )
        expanded_train = None
        if train_data is not None:
            x, y = train_data
            matrix = torch.as_tensor(x, dtype=torch.float64)
            self._validate_input_shape(matrix)
            expanded_train = (matrix.index_select(1, self.node_to_raw), y)
        self.model = self.base_module.RankBlockModel(
            capacities,
            family,
            architecture,
            inputs_per_block=inputs_per_block,
            train_data=expanded_train,
            effective_directions=effective_directions,
            **initialization,
        )
        self.model.initialization_receipt[WIDTH_METADATA_KEY] = self.counted_metadata()
        assert (
            self.parameter_count
            == parameter_inventory(
                sum(capacities),
                geometry,
                raw_blocks=raw_blocks,
                inputs_per_block=inputs_per_block,
            )["stored_parameters"]
        )

    @classmethod
    def _from_loaded_model(cls, model, base_module):
        metadata = model.initialization_receipt.get(WIDTH_METADATA_KEY)
        if not isinstance(metadata, dict):
            raise ValueError(
                "Base state lacks the fixed raw-block mapping; provide a width-exported state"
            )
        wrapper = cls.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.geometry = metadata["geometry"]
        architecture, width = geometry_spec(wrapper.geometry)
        wrapper.raw_blocks = int(metadata["raw_blocks"])
        wrapper.nodes_per_block = width
        wrapper.inputs_per_block = int(metadata["inputs_per_block"])
        wrapper.base_module = base_module
        wrapper.model = model
        wrapper.register_buffer(
            "node_to_raw",
            torch.tensor(
                metadata["node_to_raw"], dtype=torch.int64, device=model.bias.device
            ),
        )
        expected_map = torch.arange(
            wrapper.raw_blocks, device=model.bias.device
        ).repeat_interleave(width)
        if not torch.equal(wrapper.node_to_raw, expected_map):
            raise ValueError(
                "Saved fixed map violates declared block-major duplication"
            )
        if model.architecture != architecture or len(model.capacities) != len(
            expected_map
        ):
            raise ValueError("Expanded base geometry differs from saved mapping")
        if metadata != wrapper.counted_metadata():
            raise ValueError(
                "Saved width metadata, source hashes or counts differ from reconstruction"
            )
        return wrapper

    @classmethod
    def from_model(cls, model, geometry, *, raw_blocks=4, base_module=None):
        """Wrap an initializer-created base model; change no numeric parameter."""
        architecture, width = geometry_spec(geometry)
        if (
            model.architecture != architecture
            or len(model.capacities) != raw_blocks * width
        ):
            raise ValueError("Base model must match the declared expanded geometry")
        wrapper = cls.__new__(cls)
        nn.Module.__init__(wrapper)
        wrapper.geometry = geometry
        wrapper.raw_blocks = int(raw_blocks)
        wrapper.nodes_per_block = width
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
    def parameter_count(self):
        return self.model.parameter_count

    @property
    def gradient_parameter_count(self):
        return self.model.gradient_parameter_count

    @property
    def family(self):
        return self.model.family

    def counted_metadata(self):
        return {
            **parameter_inventory(
                sum(self.model.capacities),
                self.geometry,
                raw_blocks=self.raw_blocks,
                inputs_per_block=self.inputs_per_block,
            ),
            "capacities": list(self.model.capacities),
            "node_to_raw": self.node_to_raw.detach().cpu().tolist(),
            "output_normalization": "Unchanged base effective_readout=internal*soma/sqrt(expanded_node_count). No extra learned or fixed output rescaling in wrapper.",
            "base_source_sha256": sha256(self.base_module.__file__),
            "wrapper_source_sha256": sha256(__file__),
            "representation_scope": "Each node has within-node rank1 for width arms; several nodes expose several independently trainable directions per original block. No rank1 information floor is asserted for width2/4.",
        }

    def _validate_input_shape(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (
            self.raw_blocks,
            self.inputs_per_block,
        ):
            raise ValueError(
                "Expected raw inputs with shape [N, raw_blocks, inputs_per_block]"
            )

    def expanded_inputs(self, raw_x):
        x = torch.as_tensor(
            raw_x, dtype=self.model.bias.dtype, device=self.model.bias.device
        )
        self._validate_input_shape(x)
        return x.index_select(1, self.node_to_raw.to(device=x.device))

    def expanded_numpy(self, raw_x):
        x = np.asarray(raw_x)
        self._validate_input_shape(x)
        return x[:, self.node_to_raw.detach().cpu().numpy(), :]

    def forward(self, raw_x):
        return self.model(self.expanded_inputs(raw_x))

    def predict(self, raw_x):
        return self.forward(raw_x)

    def features(self, raw_x):
        return self.model.features(self.expanded_inputs(raw_x))

    def components(self, raw_x):
        components = self.model.components(self.expanded_inputs(raw_x))
        return components.reshape(
            len(components), self.raw_blocks, self.nodes_per_block
        ).sum(dim=2)


def model_under_ceiling(
    ceiling,
    family,
    geometry,
    *,
    allocation_seed=0,
    raw_blocks=4,
    inputs_per_block=3,
    bank=None,
    initialization_module=None,
    coverage="legacy",
    node_pattern="equally_spaced",
    initializer_receipt=None,
    **kwargs,
):
    _, width = geometry_spec(geometry)
    branches = branches_under_ceiling(
        ceiling, geometry, raw_blocks=raw_blocks, inputs_per_block=inputs_per_block
    )
    capacities = allocate_capacities(
        branches, raw_blocks=raw_blocks, nodes_per_block=width, seed=allocation_seed
    )
    if bank is not None:
        if initialization_module is None or kwargs.get("train_data") is None:
            raise ValueError(
                "A direction bank requires its initializer module and unchanged TRAIN data"
            )
        initialization = dict(kwargs)
        x, y = initialization.pop("train_data")
        base_module = initialization.pop("base_module", None)
        x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
        y = y.detach().cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
        bank = np.asarray(bank, dtype=np.float64)
        if x.shape[1:] != (raw_blocks, inputs_per_block) or bank.shape != (
            raw_blocks,
            inputs_per_block,
            2,
        ):
            raise ValueError(
                "Bank and TRAIN inputs must describe the original supplied blocks"
            )
        node_map = np.repeat(np.arange(raw_blocks), width)
        expanded_bank = (
            initialization_module.node_banks(
                bank,
                width,
                pattern=node_pattern,
                spread=initialization.get("initialization_spread", 0.1),
            )
            if geometry.startswith("width")
            else bank
        )
        architecture, _ = geometry_spec(geometry)
        model = initialization_module.make_initialized_model(
            capacities,
            family,
            architecture,
            x[:, node_map, :],
            y,
            expanded_bank,
            coverage=coverage,
            initializer_receipt=initializer_receipt,
            **initialization,
        )
        model.initialization_receipt["width_bank_context"] = {
            "original_bank_sha256": hashlib.sha256(
                np.ascontiguousarray(bank).tobytes()
            ).hexdigest(),
            "node_pattern": node_pattern,
            "scope": "Fixed per-node rotation of a TRAIN-estimated plane; no private directions. Exact full/rank2 feature matching is supplied by their initializer, not asserted across different width geometries.",
        }
        return RankWidthModel.from_model(
            model, geometry, raw_blocks=raw_blocks, base_module=base_module
        )
    return RankWidthModel(
        capacities,
        family,
        geometry,
        raw_blocks=raw_blocks,
        inputs_per_block=inputs_per_block,
        **kwargs,
    )


def fit(wrapper, raw_x, y, config=None, *, optimizer="reduced", output_dir=None):
    """Delegate the unchanged fitter and its state/failure preservation to base."""
    return wrapper.base_module.fit(
        wrapper.model,
        wrapper.expanded_inputs(raw_x),
        y,
        config,
        optimizer=optimizer,
        output_dir=output_dir,
    )


def state_descriptor(wrapper, state_path):
    state_path = Path(state_path).resolve()
    return {
        "state_path": str(state_path),
        "state_sha256": sha256(state_path),
        "width_metadata": wrapper.counted_metadata(),
    }


def save_state(wrapper, state_path, descriptor_path=None):
    """Export ordinary base NPZ plus optional explicit fixed-map descriptor."""
    wrapper.model.initialization_receipt[WIDTH_METADATA_KEY] = (
        wrapper.counted_metadata()
    )
    base_receipt = wrapper.base_module.save_state(wrapper.model, state_path)
    descriptor = state_descriptor(wrapper, state_path)
    if descriptor_path is not None:
        with Path(descriptor_path).open("x") as stream:
            json.dump(descriptor, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return {**base_receipt, **descriptor}


def load_state(path, *, base_module=None, device="cpu"):
    """Accept a width descriptor or an unchanged fitter-produced base NPZ."""
    path = Path(path)
    descriptor = json.loads(path.read_text()) if path.suffix == ".json" else None
    if descriptor is not None:
        state_path = Path(descriptor["state_path"])
        if sha256(state_path) != descriptor["state_sha256"]:
            raise ValueError("Saved base state hash differs from descriptor")
    else:
        state_path = path
    base = base_module or load_base_module()
    wrapper = RankWidthModel._from_loaded_model(
        base.load_state(state_path, device), base
    )
    if (
        descriptor is not None
        and descriptor["width_metadata"] != wrapper.counted_metadata()
    ):
        raise ValueError("Descriptor mapping/counts differ from the serialized state")
    return wrapper

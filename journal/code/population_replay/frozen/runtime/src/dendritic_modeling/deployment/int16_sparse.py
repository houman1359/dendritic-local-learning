"""Review-only frozen eager INT16 resident-index prototype.

Not installed into the package or connected to a model loader. No training,
Triton, packed-value, bitmask-resident, or latency claim. Reuses the existing
frozen eager projection helper, including its per-tile INT64 scratch conversion.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

SCHEMA = "dendritic_frozen_eager_int16_linear/v1"


class FrozenInt16Linear(nn.Module):
    def __init__(
        self, in_features, indices, values, *, output_chunk_size=128, workspace_mb=16.0
    ):
        super().__init__()
        if isinstance(in_features, bool) or not isinstance(in_features, Integral):
            raise TypeError("in_features must be an integer")
        if not 1 <= in_features <= 32768:
            raise ValueError(
                "signed INT16 domain supports 1..32768 inputs, indices 0..32767"
            )
        if indices.dtype not in {torch.int16, torch.int32, torch.int64}:
            raise TypeError(
                "indices must be signed INT16/INT32/INT64; unsigned inputs are not reinterpreted"
            )
        if indices.ndim != 2 or indices.shape != values.shape or not indices.numel():
            raise ValueError(
                "indices and values must be nonempty matching two-dimensional tensors"
            )
        if not values.is_floating_point() or not bool(torch.isfinite(values).all()):
            raise ValueError("values must be finite floating-point tensors")
        if indices.device != values.device:
            raise ValueError("indices and values must share one device")
        # Validate in the wide representation BEFORE casting, preventing wraparound.
        if int(indices.min()) < 0 or int(indices.max()) >= in_features:
            raise ValueError("index is outside the declared input domain")
        ordered = indices.sort(dim=1).values
        if ordered.shape[1] > 1 and bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError("indices must be unique within a row")
        if (
            isinstance(output_chunk_size, bool)
            or not isinstance(output_chunk_size, Integral)
            or output_chunk_size < 1
        ):
            raise ValueError("output_chunk_size must be a positive integer")
        if not math.isfinite(float(workspace_mb)) or workspace_mb <= 0:
            raise ValueError("workspace_mb must be finite and positive")
        self.in_features = int(in_features)
        self.out_features, self.K = map(int, values.shape)
        self.output_chunk_size = int(output_chunk_size)
        self.workspace_mb = float(workspace_mb)
        self.weight_transform = "identity"
        self.register_buffer(
            "connection_indices", indices.detach().to(torch.int16).contiguous().clone()
        )
        self.register_buffer("pre_w", values.detach().contiguous().clone())
        super().train(False)

    @classmethod
    def from_indexed(cls, layer, *, output_chunk_size=128, workspace_mb=16.0):
        if type(layer) is not IndexedSparseLinear:
            raise TypeError(
                "only an already frozen fixed IndexedSparseLinear is accepted"
            )
        if layer.training or layer.pre_w.requires_grad:
            raise ValueError("source layer must be eval and have frozen values")
        if layer.weight_transform != "identity" or layer.weight_norm_order is not None:
            raise ValueError(
                "prototype supports only signed identity weights without normalization"
            )
        if (
            layer._inference_folded_weight is not None
            or layer._recurrent_weight_cache_enabled
            or layer._recurrent_cached_weight is not None
        ):
            raise ValueError(
                "drop folded or recurrent effective weights before conversion"
            )
        return cls(
            layer.in_features,
            layer.connection_indices,
            layer.pre_w,
            output_chunk_size=output_chunk_size,
            workspace_mb=workspace_mb,
        )

    def train(self, mode=True):
        if mode:
            raise RuntimeError("this prototype is frozen inference only")
        return super().train(False)

    def forward(self, inputs):
        if not torch.is_tensor(inputs) or not inputs.is_floating_point():
            raise TypeError("floating input tensor required")
        if inputs.requires_grad:
            raise RuntimeError(
                "input gradients are unsupported by the frozen prototype"
            )
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("input shape mismatch")
        if inputs.device != self.pre_w.device:
            raise ValueError("input and resident state must share a device")
        if self.connection_indices.dtype != torch.int16:
            raise RuntimeError("resident index representation drifted")
        shape = inputs.shape
        flat = inputs.reshape(-1, self.in_features)
        result = IndexedSparseLinear._forward_frozen_chunked_transform(
            self, flat, self.connection_indices, self.output_chunk_size
        )
        return result.reshape(*shape[:-1], self.out_features)

    def payload(self):
        return {
            "schema": SCHEMA,
            "in_features": self.in_features,
            "output_chunk_size": self.output_chunk_size,
            "workspace_mb": self.workspace_mb,
            "state_dict": {
                k: v.detach().cpu().clone() for k, v in self.state_dict().items()
            },
        }

    @classmethod
    def from_payload(cls, payload):
        if (
            set(payload)
            != {
                "schema",
                "in_features",
                "output_chunk_size",
                "workspace_mb",
                "state_dict",
            }
            or payload["schema"] != SCHEMA
        ):
            raise ValueError("unknown or incomplete prototype payload")
        state = payload["state_dict"]
        if (
            set(state) != {"pre_w", "connection_indices"}
            or state["connection_indices"].dtype != torch.int16
        ):
            raise ValueError(
                "native payload must contain exactly values and INT16 topology"
            )
        return cls(
            payload["in_features"],
            state["connection_indices"],
            state["pre_w"],
            output_chunk_size=payload["output_chunk_size"],
            workspace_mb=payload["workspace_mb"],
        )

    def accounting(self):
        return {
            "value_bytes": self.pre_w.numel() * self.pre_w.element_size(),
            "index_bytes": self.connection_indices.numel() * 2,
            "resident_registered_bytes": sum(
                t.numel() * t.element_size() for t in self.buffers()
            ),
            "maximum_index_scratch_bytes": min(
                self.output_chunk_size, self.out_features
            )
            * self.K
            * 8,
            "index_scratch_scope": "Conservative upper bound on one per-output-tile INT64 cast; actual helper may reduce tile size. Repeated each input/output tile. Excludes gather/product/output workspace and allocator reservation.",
            "backend": "existing frozen eager helper; no direct INT16 Triton support",
        }


def convert_fixed_signed_indices_to_int16_(
    root, *, output_chunk_size=128, workspace_mb=16.0
):
    """Replace fixed signed sparse children in-place; preserve outer cell wrappers.

    This is an explicit inference conversion, never an automatic dtype switch.
    Returned metadata is required when rebuilding a native outer cell from config.
    """
    candidates = []
    for path, child in root.named_modules():
        if type(child) is IndexedSparseLinear:
            if not path:
                raise ValueError(
                    "convert a containing cell; use from_indexed for a root projection"
                )
            replacement = FrozenInt16Linear.from_indexed(
                child, output_chunk_size=output_chunk_size, workspace_mb=workspace_mb
            )
            candidates.append((path, replacement))
    if not candidates:
        raise ValueError("no eligible fixed signed indexed children")
    # Construct and validate every child before the first mutation.
    for path, replacement in candidates:
        parent_name, _, name = path.rpartition(".")
        parent = root.get_submodule(parent_name) if parent_name else root
        parent._modules[name] = replacement
    return {
        "schema": "dendritic_frozen_int16_conversion/v1",
        "modules": {
            path: {
                "in_features": m.in_features,
                "out_features": m.out_features,
                "k": m.K,
                "output_chunk_size": m.output_chunk_size,
                "workspace_mb": m.workspace_mb,
                "weight_transform": "identity",
                "accounting": m.accounting(),
            }
            for path, m in candidates
        },
    }


def load_converted_int16_cell_state_(root, state, conversion):
    """Load native whole-cell state after reconstructing the exact outer wrapper.

    The destination must already be explicitly converted. Reject a legacy INT32
    destination instead of letting load_state_dict silently widen resident indices.
    """
    if (
        set(conversion) != {"schema", "modules"}
        or conversion["schema"] != "dendritic_frozen_int16_conversion/v1"
    ):
        raise ValueError("unknown conversion manifest")
    observed = {
        p: m for p, m in root.named_modules() if isinstance(m, FrozenInt16Linear)
    }
    if set(observed) != set(conversion["modules"]):
        raise ValueError(
            "destination was not rebuilt with the exact frozen INT16 modules"
        )
    for path, module in observed.items():
        entry = conversion["modules"][path]
        expected = {
            "in_features": module.in_features,
            "out_features": module.out_features,
            "k": module.K,
            "output_chunk_size": module.output_chunk_size,
            "workspace_mb": module.workspace_mb,
            "weight_transform": "identity",
            "accounting": module.accounting(),
        }
        if entry != expected:
            raise ValueError("destination conversion geometry/ledger differs")
        key = path + ".connection_indices"
        if key not in state or state[key].dtype != torch.int16:
            raise ValueError("saved converted topology must remain INT16")
        # Re-run dtype, finite-value, bounds and uniqueness validation before mutation.
        FrozenInt16Linear(
            module.in_features,
            state[key],
            state[path + ".pre_w"],
            output_chunk_size=module.output_chunk_size,
            workspace_mb=module.workspace_mb,
        )
    expected_state = root.state_dict()
    if set(state) != set(expected_state):
        raise ValueError("whole-cell state key set differs")
    for key, tensor in state.items():
        if (
            tensor.shape != expected_state[key].shape
            or tensor.dtype != expected_state[key].dtype
        ):
            raise ValueError("whole-cell shape/dtype differs at " + key)
    root.load_state_dict(state, strict=True)
    return root

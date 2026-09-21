"""Physical frozen value packing for complete PopulationNetwork cells.

BF16 values, unsigned INT8 codes, or two INT4 codes per byte remain resident.
Only a bounded tile is decoded to the original weight dtype during inference.
The stored quantity is the effective transformed/normalized weight, never a
log-parameter. Positive E/I conductances stay nonnegative; the enclosing graph
retains its E-minus-I, shunting, gating, activation and state-update operations.
This module makes a storage claim, not an integer-kernel or latency claim.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import inspect
import json
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
    EfficientBlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

SCHEMA = "packed_population_cell/v1"
_COMPACT_SCHEMA = "packed_population_cell/v2"
_SCHEMAS = (SCHEMA, _COMPACT_SCHEMA)
_TOPOLOGY_FORMATS = ("indexed", "periodic_uint16")
FORMATS = ("bf16", "int8", "int4")
_DTYPES = {
    str(dtype): dtype
    for dtype in (torch.float32, torch.bfloat16, torch.float16, torch.float64)
}


def _class_name(value):
    return type(value).__module__ + "." + type(value).__qualname__


def _tensor_digest(value):
    value = value.detach().contiguous().cpu()
    h = hashlib.sha256(json.dumps([str(value.dtype), list(value.shape)]).encode())
    h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _frozen(module):
    if module.training or any(p.requires_grad for p in module.parameters()):
        raise ValueError("Packing requires an explicitly frozen eval cell")


def _index_long(indices):
    # UInt16 indexing/reductions are not available on every Torch backend.
    # Signed reinterpretation followed by masking preserves all sixteen bits.
    if indices.dtype == torch.uint16:
        return indices.view(torch.int16).long().bitwise_and_(65535)
    return indices.long()


def _validate_indices(indices, in_features):
    if indices.ndim != 2 or not indices.numel():
        raise ValueError("Nonempty matrix topology required")
    if indices.dtype not in (torch.int16, torch.uint16, torch.int32, torch.int64):
        raise ValueError("Integral resident topology required")
    if in_features > 2**31:
        raise ValueError("Input domain exceeds supported int32 topology")
    for block in indices.split(max(1, (1 << 20) // indices.shape[1])):
        values = _index_long(block)
        if int(values.min()) < 0 or int(values.max()) >= in_features:
            raise ValueError(
                "Validate topology before casting: index outside input domain"
            )


def _exact_row_period(indices, limit=1024):
    """Find an exact period up to limit; a failed search keeps explicit rows."""
    n, columns = indices.shape
    values = indices.view(torch.int16) if indices.dtype == torch.uint16 else indices
    for lo in range(1, min(n, limit + 1), 64):
        matches = (values[lo : min(lo + 64, n, limit + 1)] == values[0]).all(1)
        for offset in matches.nonzero().flatten().tolist():
            period = lo + offset
            # Reject coincidental prefix matches before scanning a large bank.
            starts = (period, max(period, n - 64), max(period, n // 2))
            if any(
                not torch.equal(
                    values[start : min(start + 64, n)],
                    values[
                        torch.arange(start, min(start + 64, n), device=values.device)
                        % period
                    ],
                )
                for start in starts
            ):
                continue
            chunk = max(1, (1 << 20) // columns)
            if all(
                torch.equal(
                    values[start : min(start + chunk, n)],
                    values[
                        torch.arange(start, min(start + chunk, n), device=values.device)
                        % period
                    ],
                )
                for start in range(0, n, chunk)
            ):
                return period
    return None


def _compact_indices(indices, in_features):
    _validate_indices(indices, in_features)
    period = _exact_row_period(indices)
    dtype = (
        torch.int16
        if in_features <= 32768
        else torch.uint16 if in_features <= 65536 else torch.int32
    )
    selected = indices if period is None else indices[:period]
    return {
        "layout": "explicit" if period is None else "periodic",
        "period": period,
        "index_dtype": str(dtype),
    }, selected.to(dtype)


class PackedValueMatrix(nn.Module):
    """Per-row active-value groups with exact-zero symmetric/nonnegative PTQ."""

    def __init__(self, spec, values, scales=None):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.format = spec["format"]
        self.rows, self.columns = spec["shape"]
        self.group_size = spec["group_size"]
        self.nonnegative = spec["nonnegative"]
        self.compute_dtype = _DTYPES.get(spec["compute_dtype"])
        if self.format not in FORMATS or self.compute_dtype is None:
            raise ValueError("Unknown packed format or compute dtype")
        if any(
            type(n) is not int or n < 1
            for n in (self.rows, self.columns, self.group_size)
        ):
            raise ValueError("Positive integral matrix dimensions/group size required")
        if type(self.nonnegative) is not bool:
            raise ValueError("Explicit sign convention required")
        expected = (
            self.rows,
            self.columns if self.format != "int4" else (self.columns + 1) // 2,
        )
        if tuple(values.shape) != expected:
            raise ValueError("Packed value shape changed")
        if self.format == "bf16":
            if (
                values.dtype != torch.bfloat16
                or scales is not None
                or not bool(torch.isfinite(values).all())
            ):
                raise ValueError("BF16 format requires only finite BF16 values")
            if self.nonnegative and bool((values < 0).any()):
                raise ValueError("Positive conductance became negative")
        else:
            if values.dtype != torch.uint8:
                raise ValueError("Integer format must contain actual UINT8 storage")
            if (
                scales is None
                or scales.dtype != torch.float32
                or tuple(scales.shape)
                != (self.rows, math.ceil(self.columns / self.group_size))
            ):
                raise ValueError("One FP32 scale per active-value group required")
            if (
                values.device != scales.device
                or not bool(torch.isfinite(scales).all())
                or not bool((scales > 0).all())
            ):
                raise ValueError("Finite positive colocated scales required")
            if (
                self.format == "int4"
                and self.columns % 2
                and bool((values[:, -1] >> 4).any())
            ):
                raise ValueError("Noncanonical nonzero INT4 padding")
            if not self.nonnegative:
                codes = (
                    values
                    if self.format == "int8"
                    else self._unpack(values, self.columns)
                )
                if bool((codes == 0).any()):
                    raise ValueError("Reserved asymmetric negative code is invalid")
        self.register_buffer("values", values.detach().contiguous().clone())
        self.register_buffer(
            "scales", None if scales is None else scales.detach().contiguous().clone()
        )
        super().train(False)

    @staticmethod
    def _unpack(values, columns):
        return torch.stack((values & 15, values >> 4), dim=-1).flatten(-2)[:, :columns]

    @classmethod
    def from_effective(cls, weight, *, format="int8", group_size=128):
        if (
            weight.ndim != 2
            or not weight.numel()
            or str(weight.dtype) not in _DTYPES
            or not bool(torch.isfinite(weight).all())
        ):
            raise ValueError("Finite floating effective weight matrix required")
        if format not in FORMATS or type(group_size) is not int or group_size < 1:
            raise ValueError("BF16/INT8/INT4 and a positive group size required")
        w = weight.detach()
        spec = {
            "format": format,
            "shape": list(w.shape),
            "group_size": group_size,
            "nonnegative": bool((w >= 0).all()),
            "compute_dtype": str(w.dtype),
            "stored_quantity": "effective_weight_after_transform_and_normalization",
        }
        if format == "bf16":
            return cls(spec, w.to(torch.bfloat16))
        bits = 8 if format == "int8" else 4
        positive = spec["nonnegative"]
        limit = (1 << bits) - 1 if positive else (1 << (bits - 1)) - 1
        zero = 0 if positive else 1 << (bits - 1)
        groups = math.ceil(w.shape[1] / group_size)
        padded = F.pad(w.float(), (0, groups * group_size - w.shape[1])).reshape(
            w.shape[0], groups, group_size
        )
        maxima = padded.abs().amax(dim=-1)
        scales = torch.where(maxima > 0, maxima / limit, torch.ones_like(maxima))
        codes = (
            (padded / scales.unsqueeze(-1))
            .round()
            .clamp(0 if positive else -limit, limit)
            + zero
        ).to(torch.uint8)
        codes = codes.flatten(1)[:, : w.shape[1]].contiguous()
        if format == "int4":
            codes = F.pad(codes, (0, w.shape[1] % 2))
            codes = codes[:, ::2] | (codes[:, 1::2] << 4)
        return cls(spec, codes, scales)

    def train(self, mode=True):
        if mode:
            raise RuntimeError(
                "Packed values are inference only; recover FP32 masters separately"
            )
        return super().train(False)

    def _apply(self, fn, recurse=True):
        for value in (self.values, self.scales):
            if value is not None and fn(value.new_empty(0)).dtype != value.dtype:
                raise ValueError(
                    "Device moves are supported; implicit packed-format dtype changes are not"
                )
        return super()._apply(fn, recurse=recurse)

    def decode(self, start=0, end=None):
        end = self.rows if end is None else end
        if not 0 <= start <= end <= self.rows:
            raise ValueError("Invalid decoded row interval")
        values = self.values[start:end]
        if self.format == "bf16":
            if values.dtype != torch.bfloat16:
                raise RuntimeError("Resident BF16 format drifted")
            return values.to(self.compute_dtype)
        if values.dtype != torch.uint8 or self.scales.dtype != torch.float32:
            raise RuntimeError("Resident integer format drifted")
        codes = values if self.format == "int8" else self._unpack(values, self.columns)
        zero = 0 if self.nonnegative else (128 if self.format == "int8" else 8)
        groups = torch.arange(self.columns, device=values.device) // self.group_size
        return ((codes.float() - zero) * self.scales[start:end, groups]).to(
            self.compute_dtype
        )


class PackedPopulationLinear(nn.Module):
    """Fixed indexed, implicit block, or dense projection with packed values."""

    def __init__(self, spec, matrix, connection_indices=None, bias=None):
        super().__init__()
        self.spec = copy.deepcopy(spec)
        self.kind = spec["kind"]
        self.in_features = spec["in_features"]
        self.out_features = spec["out_features"]
        self.output_chunk_size = spec["output_chunk_size"]
        self.workspace_mb = spec["workspace_mb"]
        self.matrix = matrix
        # Runtime-only opt-in. Archives and restored modules retain the generic
        # reference until the caller explicitly selects a qualified backend.
        self.inference_backend = "generic"
        self.weight_transform = "identity"
        if self.kind not in ("indexed", "block_reduce", "block_dense", "dense"):
            raise ValueError("Unknown packed projection kind")
        if (
            any(
                type(n) is not int or n < 1
                for n in (self.in_features, self.out_features, self.output_chunk_size)
            )
            or not math.isfinite(self.workspace_mb)
            or self.workspace_mb <= 0
        ):
            raise ValueError("Invalid projection dimensions/workspace")
        if self.out_features != matrix.rows:
            raise ValueError("Packed matrix output shape changed")
        self.index_period = None
        if self.kind == "indexed":
            topology = spec.get("topology")
            if topology is not None:
                if set(topology) != {"layout", "period", "index_dtype"} or topology[
                    "layout"
                ] not in ("explicit", "periodic"):
                    raise ValueError("Unknown compact topology descriptor")
                period = topology["period"]
                if topology["layout"] == "periodic":
                    if type(period) is not int or not 1 <= period < matrix.rows:
                        raise ValueError("Invalid topology period")
                    self.index_period = period
                elif period is not None:
                    raise ValueError("Explicit topology cannot declare a period")
            if connection_indices is None:
                raise ValueError("Explicit resident topology required")
            expected_rows = self.index_period or matrix.rows
            if (
                tuple(connection_indices.shape) != (expected_rows, matrix.columns)
                or connection_indices.device != matrix.values.device
            ):
                raise ValueError("Indexed topology shape/device changed")
            expected = torch.int16 if self.in_features <= 32768 else torch.int32
            if topology is not None and 32768 < self.in_features <= 65536:
                expected = torch.uint16
            if connection_indices.dtype != expected or (
                topology is not None and topology["index_dtype"] != str(expected)
            ):
                raise ValueError(
                    "Topology does not use the declared safe compact representation"
                )
            _validate_indices(connection_indices, self.in_features)
            self.K = matrix.columns
        else:
            if connection_indices is not None or "topology" in spec:
                raise ValueError("Dense/block connectivity must be implicit")
            if self.kind.startswith("block"):
                self.block_size = matrix.columns
                if self.in_features != self.out_features * self.block_size:
                    raise ValueError("Block connectivity changed")
            elif matrix.columns != self.in_features:
                raise ValueError("Dense input dimension changed")
        if bias is not None and (
            self.kind != "dense"
            or bias.shape != (self.out_features,)
            or bias.device != matrix.values.device
            or not bool(torch.isfinite(bias).all())
        ):
            raise ValueError("Invalid dense auxiliary bias")
        self.register_buffer(
            "connection_indices",
            (
                None
                if connection_indices is None
                else connection_indices.detach().contiguous().clone()
            ),
        )
        self.register_buffer(
            "bias", None if bias is None else bias.detach().contiguous().clone()
        )
        super().train(False)

    @classmethod
    def from_module(
        cls,
        source,
        *,
        format="int8",
        group_size=128,
        output_chunk_size=128,
        workspace_mb=16.0,
        topology_format="indexed",
    ):
        _frozen(source)
        if topology_format not in _TOPOLOGY_FORMATS:
            raise ValueError("Unknown topology format")
        indices = bias = topology = None
        if type(source) is IndexedSparseLinear:
            if not getattr(source, "_topology_ready", True) and not bool(
                source._topology_initialized.item()
            ):
                raise ValueError("Topology must be initialized before packing")
            weight = source._inference_folded_weight
            if weight is None:
                weight = source._forward_sparse_weight()
            indices = source.connection_indices
            _validate_indices(indices, source.in_features)
            if topology_format == "periodic_uint16":
                topology, indices = _compact_indices(indices, source.in_features)
            else:
                indices = indices.to(
                    torch.int16 if source.in_features <= 32768 else torch.int32
                )
            kind = "indexed"
        elif type(source) in (BlockLinear, EfficientBlockLinear):
            weight = source.weight()
            kind = (
                "block_reduce"
                if type(source) is EfficientBlockLinear
                else "block_dense"
            )
        elif type(source) is nn.Linear:
            weight, bias, kind = source.weight, source.bias, "dense"
        else:
            raise TypeError(
                "Only fixed IndexedSparseLinear, native BlockLinear, or exact nn.Linear is supported"
            )
        matrix = PackedValueMatrix.from_effective(
            weight, format=format, group_size=group_size
        )
        spec = {
            "kind": kind,
            "in_features": source.in_features,
            "out_features": source.out_features,
            "source_class": _class_name(source),
            "source_weight_transform": getattr(source, "weight_transform", "identity"),
            "source_weight_norm_order": getattr(source, "weight_norm_order", None),
            "source_gamma": getattr(source, "gamma", None),
            "retired_source_buffers": [
                {
                    "name": name,
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "bytes": value.numel() * value.element_size(),
                }
                for name, value in source.named_buffers()
                if name != "connection_indices"
            ],
            "output_chunk_size": output_chunk_size,
            "workspace_mb": float(workspace_mb),
            "matrix": matrix.spec,
        }
        if topology is not None:
            spec["topology"] = topology
        return cls(spec, matrix, indices, bias)

    def train(self, mode=True):
        if mode:
            raise RuntimeError(
                "Packed PopulationNetwork projections are inference only"
            )
        return super().train(False)

    def weight(self):
        if self.kind != "indexed":
            return self.matrix.decode()
        dense = self.matrix.values.new_zeros(
            (self.out_features, self.in_features), dtype=self.matrix.compute_dtype
        )
        return dense.scatter_add_(1, self.decode_indices(), self.matrix.decode())

    def decode_indices(self, start=0, end=None):
        """Decode only the requested output rows; no full-index runtime cache."""
        end = self.out_features if end is None else end
        if self.kind != "indexed" or not 0 <= start <= end <= self.out_features:
            raise ValueError("Invalid decoded topology row interval")
        resident = self.connection_indices
        if resident.dtype == torch.uint16:
            resident = resident.view(torch.int16)
        if self.index_period is None:
            selected = resident[start:end]
        else:
            rows = torch.arange(start, end, device=resident.device) % self.index_period
            selected = resident[rows]
        values = selected.long()
        if self.connection_indices.dtype == torch.uint16:
            values.bitwise_and_(65535)
        return values

    def sum_conductances(self):
        if not self.kind.startswith("block"):
            raise TypeError("Only native block aggregation has this conductance API")
        return self.matrix.decode().sum(dim=1)

    def set_inference_backend(self, backend="generic"):
        """Select a runtime backend without changing stored values or topology.

        ``triton_bf16`` supports indexed BF16 values whose reference compute
        dtype is FP32. CUDA availability is checked at forward, allowing a
        caller to select the backend before moving a restored cell to CUDA.
        Other projection kinds keep their existing generic implementation.
        """
        if backend not in ("generic", "triton_bf16", "triton_bf16_block"):
            raise ValueError("Unknown packed inference backend")
        if backend == "triton_bf16" and (
            self.kind != "indexed"
            or self.matrix.format != "bf16"
            or self.matrix.compute_dtype != torch.float32
        ):
            raise ValueError("Triton requires indexed BF16 values with FP32 compute")
        if backend == "triton_bf16_block" and (
            self.kind != "block_reduce"
            or self.matrix.format != "bf16"
            or self.matrix.compute_dtype != torch.float32
            or self.weight_transform != "identity"
            or self.matrix.columns > 4096
        ):
            raise ValueError(
                "Triton block reduction requires effective BF16 values with FP32 compute"
            )
        self.inference_backend = backend
        return self

    def forward(self, inputs):
        if (
            not torch.is_tensor(inputs)
            or not inputs.is_floating_point()
            or inputs.ndim < 1
            or inputs.shape[-1] != self.in_features
        ):
            raise ValueError(
                "Floating inputs with the exact original feature dimension required"
            )
        if inputs.requires_grad or self.training:
            raise RuntimeError("Packed inference does not accept gradients")
        if inputs.device != self.matrix.values.device:
            raise ValueError("Packed values and inputs must share a device")
        shape = inputs.shape
        flat = inputs.reshape(-1, self.in_features)
        if self.inference_backend == "triton_bf16":
            from dendritic_modeling.deployment._packed_triton import packed_bf16_gather

            return packed_bf16_gather(
                flat, self.matrix.values, self.connection_indices, self.index_period
            ).reshape(*shape[:-1], self.out_features)
        if self.inference_backend == "triton_bf16_block":
            from dendritic_modeling.deployment._packed_triton import (
                packed_bf16_block_reduce,
            )

            result = packed_bf16_block_reduce(flat, self.matrix.values)
            # Preserve the native block path's historical rank-one convention.
            return (
                result
                if len(shape) == 1
                else result.reshape(*shape[:-1], self.out_features)
            )
        output_dtype = (
            torch.promote_types(inputs.dtype, self.matrix.compute_dtype)
            if self.kind.startswith("block")
            else inputs.dtype
        )
        output = flat.new_empty((flat.shape[0], self.out_features), dtype=output_dtype)
        elements = max(
            1,
            int(self.workspace_mb * 1024**2)
            // max(
                inputs.element_size(),
                torch.empty((), dtype=self.matrix.compute_dtype).element_size(),
            ),
        )
        columns = (
            self.in_features if self.kind == "block_dense" else self.matrix.columns
        )
        if self.kind == "block_dense" and columns > elements:
            raise ValueError(
                "Workspace cannot hold even one exact native dense-block row"
            )
        width = min(
            self.output_chunk_size,
            self.out_features,
            max(1, elements // max(1, columns)),
        )
        batch = max(1, elements // (width * columns))
        for start in range(0, self.out_features, width):
            end = min(start + width, self.out_features)
            weights = self.matrix.decode(start, end)
            if self.kind == "indexed":
                indices = self.decode_indices(start, end)
            elif self.kind == "block_dense":
                dense = weights.new_zeros((end - start, self.in_features))
                dense[
                    torch.arange(end - start, device=weights.device)[:, None],
                    torch.arange(
                        start * self.block_size,
                        end * self.block_size,
                        device=weights.device,
                    ).reshape(end - start, self.block_size),
                ] = weights
            for row in range(0, flat.shape[0], batch):
                block = flat[row : row + batch]
                if self.kind == "indexed":
                    selected = block[:, indices.reshape(-1)].reshape(
                        len(block), end - start, self.K
                    )
                    result = (selected * weights.unsqueeze(0)).sum(-1)
                elif self.kind == "block_reduce":
                    selected = block[
                        :, start * self.block_size : end * self.block_size
                    ].reshape(len(block), end - start, self.block_size)
                    result = (selected * weights).sum(-1)
                elif self.kind == "block_dense":
                    result = block @ dense.t()
                else:
                    result = F.linear(
                        block.to(self.matrix.compute_dtype),
                        weights,
                        (
                            None
                            if self.bias is None
                            else self.bias[start:end].to(self.matrix.compute_dtype)
                        ),
                    )
                output[row : row + batch, start:end] = result
        if self.kind.startswith("block") and len(shape) == 1:
            return output
        return output.reshape(*shape[:-1], self.out_features)


def set_packed_population_backend_(root, backend="generic"):
    """Opt compatible indexed banks into fused inference; return their paths.

    ``triton_bf16`` selects indexed banks only. ``triton_bf16_all`` additionally
    selects implicit native block reductions. INT8/INT4, dense block and dense
    projections stay generic. The indexed kernel is identical in both modes.
    No weights, indices, module buffers or archive descriptors are changed.
    """
    if backend not in ("generic", "triton_bf16", "triton_bf16_all"):
        raise ValueError("Unknown packed inference backend")
    selected = []
    for name, module in root.named_modules():
        if not isinstance(module, PackedPopulationLinear):
            continue
        # Presets replace the previous selection. In particular, switching
        # all -> indexed-only must retire the optional block backend.
        module.set_inference_backend("generic")
        selected_backend = None
        if backend == "generic":
            selected_backend = "generic"
        elif (
            module.matrix.format == "bf16"
            and module.matrix.compute_dtype == torch.float32
        ):
            if module.kind == "indexed":
                selected_backend = "triton_bf16"
            elif (
                backend == "triton_bf16_all"
                and module.kind == "block_reduce"
                and module.weight_transform == "identity"
                and module.matrix.columns <= 4096
            ):
                selected_backend = "triton_bf16_block"
        if selected_backend is not None:
            module.set_inference_backend(selected_backend)
            selected.append(name)
    return selected


def _candidates(root):
    result = []
    seen = set()
    for path, module in root.named_modules(remove_duplicate=False):
        if type(module) in (
            IndexedSparseLinear,
            BlockLinear,
            EfficientBlockLinear,
            nn.Linear,
        ):
            if not path:
                raise ValueError(
                    "Pass the complete containing cell, not its root projection"
                )
            if id(module) in seen:
                raise ValueError(
                    "Aliased projection modules must be untied explicitly before packing: "
                    + path
                )
            seen.add(id(module))
            result.append((path, module))
        elif isinstance(
            module, (IndexedSparseLinear, BlockLinear, EfficientBlockLinear, nn.Linear)
        ):
            raise TypeError(
                "Unsupported projection subclass at "
                + path
                + ": "
                + _class_name(module)
            )
        elif isinstance(module, PackedPopulationLinear):
            raise ValueError("Cell is already packed")
        elif ".synapse." in type(module).__module__:
            raise TypeError(
                "Unsupported synapse bank at "
                + path
                + ": "
                + _class_name(module)
                + "; explicitly freeze dynamic/masked topology first"
            )
    if not result:
        raise ValueError("No supported PopulationNetwork projection banks found")
    _reject_retained_projection_aliases(root, result)
    return result


def _reject_retained_projection_aliases(root, candidates):
    """Reject outer aliases that would retain an original projection or storage."""
    candidate_ids = {id(module) for _, module in candidates}
    candidate_paths = tuple(path + "." for path, _ in candidates)
    stores = {
        (str(t.device), t.untyped_storage().data_ptr())
        for _, module in candidates
        for t in [*module.parameters(), *module.buffers()]
        if t.numel()
    }
    for name, value in root.named_parameters(remove_duplicate=False):
        if (
            not name.startswith(candidate_paths)
            and value.numel()
            and (str(value.device), value.untyped_storage().data_ptr()) in stores
        ):
            raise ValueError(
                "Registered outer alias retains a source projection value: " + name
            )
    visited = set()

    def inspect(value, name):
        if id(value) in visited:
            return
        visited.add(id(value))
        if isinstance(value, nn.Module):
            if id(value) in candidate_ids:
                raise ValueError(
                    "Unregistered outer alias retains a source projection: " + name
                )
        elif torch.is_tensor(value):
            if (
                value.numel()
                and (str(value.device), value.untyped_storage().data_ptr()) in stores
            ):
                raise ValueError(
                    "Outer runtime alias retains source projection storage: " + name
                )
        elif isinstance(value, (tuple, list, dict)):
            iterator = value.items() if isinstance(value, dict) else enumerate(value)
            for key, item in iterator:
                inspect(item, name + "." + str(key))

    for path, module in root.named_modules():
        if id(module) in candidate_ids:
            continue
        for name, value in vars(module).items():
            if name not in ("_parameters", "_modules"):
                inspect(value, path + "." + name)


def _replace(root, path, module):
    parent, _, name = path.rpartition(".")
    (root.get_submodule(parent) if parent else root)._modules[name] = module


def packed_population_ledger(root):
    """Count every unique registered tensor, including nonpersistent buffers."""
    packed_paths = {
        name
        for name, module in root.named_modules()
        if isinstance(module, PackedPopulationLinear)
    }
    rows, seen = [], set()
    totals = {
        "value_bytes": 0,
        "scale_bytes": 0,
        "topology_bytes": 0,
        "auxiliary_bytes": 0,
    }
    for kind, iterator in (
        ("parameter", root.named_parameters()),
        ("buffer", root.named_buffers()),
    ):
        for name, tensor in iterator:
            if id(tensor) in seen:
                continue
            seen.add(id(tensor))
            owner = next((p for p in packed_paths if name.startswith(p + ".")), None)
            category = "auxiliary_bytes"
            if owner is not None:
                suffix = name[len(owner) + 1 :]
                category = {
                    "matrix.values": "value_bytes",
                    "matrix.scales": "scale_bytes",
                    "connection_indices": "topology_bytes",
                }.get(suffix, category)
            size = tensor.numel() * tensor.element_size()
            totals[category] += size
            parent, _, key = name.rpartition(".")
            module = root.get_submodule(parent) if parent else root
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "bytes": size,
                    "category": category,
                    "persistent": kind == "parameter"
                    or key not in module._non_persistent_buffers_set,
                }
            )
    return dict(
        **totals,
        total_bytes=sum(totals.values()),
        tensors=rows,
        scope="All unique registered values, indices, scales and auxiliary state, including nonpersistent buffers; transient decode workspace and allocator memory are separate",
    )


def packed_population_fingerprint(root):
    rows = []
    for kind, iterator in (
        ("parameter", root.named_parameters()),
        ("buffer", root.named_buffers()),
    ):
        for name, tensor in iterator:
            rows.append({"name": name, "kind": kind, "sha256": _tensor_digest(tensor)})
    structure = [
        (name, module.spec)
        for name, module in root.named_modules()
        if isinstance(module, PackedPopulationLinear)
    ]
    return hashlib.sha256(
        json.dumps(
            {
                "state": rows,
                "packed_structure": structure,
                "outer_graph_semantics": _outer_graph_signature(
                    root, {name for name, _ in structure}
                ),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _outer_graph_signature(cell, bank_paths):
    """Bind native computation settings as well as same-shaped state tensors.

    Native configuration dataclasses and scalar/container attributes are copied.
    Registered tensor values are independently owned by the payload. Transient
    tensors/modules and torch's module registries are not configuration values.
    """
    missing = object()

    def value_signature(value):
        if value is None or type(value) in (str, int, bool):
            return value
        if type(value) is float:
            return value if math.isfinite(value) else {"float": repr(value)}
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                "dataclass": _class_name(value),
                "fields": value_signature(dataclasses.asdict(value)),
            }
        if isinstance(value, Mapping):
            pairs = {}
            for key, item in value.items():
                result = value_signature(item)
                if result is missing:
                    return missing
                pairs[str(key)] = result
            return pairs
        if isinstance(value, (list, tuple, set, frozenset)):
            result = [value_signature(item) for item in value]
            if any(item is missing for item in result):
                return missing
            if isinstance(value, (set, frozenset)):
                result.sort(key=lambda item: json.dumps(item, sort_keys=True))
            return {"container": type(value).__name__, "items": result}
        if inspect.ismethod(value) or inspect.isfunction(value):
            fn = value.__func__ if inspect.ismethod(value) else value
            return {
                "callable": fn.__module__ + "." + fn.__qualname__,
                "code_sha256": hashlib.sha256(fn.__code__.co_code).hexdigest(),
            }
        return missing

    result = {}
    for path, module in cell.named_modules():
        if any(path == bank or path.startswith(bank + ".") for bank in bank_paths):
            continue
        attrs = {}
        for name, value in vars(module).items():
            if (
                name == "training"
                or name
                in (
                    "_parameters",
                    "_buffers",
                    "_modules",
                    "_non_persistent_buffers_set",
                )
                or "hook" in name
            ):
                continue
            if name.startswith(("_last", "_cache")):
                continue
            converted = value_signature(value)
            if converted is not missing:
                attrs[name] = converted
        result[path] = {"class": _class_name(module), "attributes": attrs}
    return result


def pack_population_cell_(
    cell,
    *,
    format="int8",
    group_size=128,
    output_chunk_size=128,
    workspace_mb=16.0,
    topology_format="indexed",
):
    """Replace every supported projection atomically, preserving the outer graph."""
    _frozen(cell)
    if topology_format not in _TOPOLOGY_FORMATS:
        raise ValueError("Unknown topology format")
    candidates = _candidates(cell)
    paths = {path for path, _ in candidates}
    outer_classes = {
        path: _class_name(module)
        for path, module in cell.named_modules()
        if path not in paths
    }
    outer_semantics = _outer_graph_signature(cell, paths)
    before = packed_population_ledger(cell)
    replacements = [
        (
            path,
            PackedPopulationLinear.from_module(
                module,
                format=format,
                group_size=group_size,
                output_chunk_size=output_chunk_size,
                workspace_mb=workspace_mb,
                topology_format=topology_format,
            ),
        )
        for path, module in candidates
    ]
    # Construction/validation succeeds for all banks before any graph mutation.
    for path, replacement in replacements:
        _replace(cell, path, replacement)
    manifest = {
        "schema": SCHEMA if topology_format == "indexed" else _COMPACT_SCHEMA,
        "root_class": _class_name(cell),
        "format": format,
        "group_size": group_size,
        "outer_module_classes": outer_classes,
        "outer_graph_semantics": outer_semantics,
        "modules": {path: module.spec for path, module in replacements},
        "original_registered_ledger": before,
        "effective_weight_quantization": True,
        "auxiliary_policy": "preserve_exact_dtype_and_values",
        "topology_policy": (
            "signed_int16_for_input_domain_at_most32768_else_int32"
            if topology_format == "indexed"
            else "exact_periodic_rows_up_to1024_signed16_to32768_unsigned16_to65536_else_int32"
        ),
        "training_supported": False,
    }
    manifest["packed_registered_ledger"] = packed_population_ledger(cell)
    manifest["packed_fingerprint"] = packed_population_fingerprint(cell)
    return manifest


def population_cell_payload(cell, conversion):
    _validate_conversion(cell, conversion)
    persistent = {
        name: value.detach().cpu().clone() for name, value in cell.state_dict().items()
    }
    runtime = {}
    for path, module in cell.named_modules():
        for key in module._non_persistent_buffers_set:
            value = module._buffers.get(key)
            if value is not None:
                runtime[(path + "." if path else "") + key] = (
                    value.detach().cpu().clone()
                )
    return {
        "schema": conversion["schema"],
        "conversion": copy.deepcopy(conversion),
        "state_dict": persistent,
        "runtime_buffers": runtime,
    }


def _validate_conversion(cell, conversion):
    if conversion["schema"] not in _SCHEMAS or conversion["root_class"] != _class_name(
        cell
    ):
        raise ValueError("Packed cell and manifest disagree")
    if (
        _outer_graph_signature(cell, conversion["modules"])
        != conversion["outer_graph_semantics"]
    ):
        raise ValueError(
            "Packed outer graph computation settings changed after conversion"
        )
    if packed_population_fingerprint(cell) != conversion["packed_fingerprint"]:
        raise ValueError("Packed cell changed after conversion")


def repack_population_topology_(cell, conversion):
    """Losslessly compact existing packed indices without requantizing values.

    The search stores exact row periods up to 1024 when present. Other banks
    retain explicit rows. This never merges weights or removes connections.
    """
    _frozen(cell)
    _validate_conversion(cell, conversion)
    before = packed_population_ledger(cell)
    replacements = []
    for path, module in cell.named_modules():
        if not isinstance(module, PackedPopulationLinear) or module.kind != "indexed":
            continue
        if "topology" in module.spec:
            continue
        topology, indices = _compact_indices(
            module.connection_indices, module.in_features
        )
        spec = copy.deepcopy(module.spec)
        spec["topology"] = topology
        replacements.append(
            (path, PackedPopulationLinear(spec, module.matrix, indices, module.bias))
        )
    # All banks validate before mutating the graph. Matrices/scales are reused.
    for path, module in replacements:
        _replace(cell, path, module)
    result = copy.deepcopy(conversion)
    result["schema"] = _COMPACT_SCHEMA
    result["modules"] = {
        path: copy.deepcopy(module.spec)
        for path, module in cell.named_modules()
        if isinstance(module, PackedPopulationLinear)
    }
    result.setdefault("pre_topology_registered_ledger", before)
    result["topology_policy"] = (
        "exact_periodic_rows_up_to1024_signed16_to32768_unsigned16_to65536_else_int32"
    )
    result["packed_registered_ledger"] = packed_population_ledger(cell)
    result["packed_fingerprint"] = packed_population_fingerprint(cell)
    return result


def restore_packed_population_cell_(cell, payload):
    """Restore an artifact-owned packed state into its fresh native outer graph."""
    if (
        set(payload) != {"schema", "conversion", "state_dict", "runtime_buffers"}
        or payload["schema"] not in _SCHEMAS
    ):
        raise ValueError("Unknown or incomplete packed-cell payload")
    manifest = payload["conversion"]
    if manifest["schema"] != payload["schema"] or manifest["root_class"] != _class_name(
        cell
    ):
        raise ValueError("Fresh outer graph differs from the packed-cell artifact")
    if manifest["schema"] == SCHEMA and any(
        "topology" in s for s in manifest["modules"].values()
    ):
        raise ValueError("Compact topology requires the version2 artifact schema")
    cell.eval().requires_grad_(False)
    candidates = dict(_candidates(cell))
    if set(candidates) != set(manifest["modules"]):
        raise ValueError("Every source synapse/readout bank must be restored exactly")
    if {
        path: _class_name(module)
        for path, module in cell.named_modules()
        if path not in candidates
    } != manifest["outer_module_classes"]:
        raise ValueError(
            "Fresh activation/population/wrapper classes differ from the artifact"
        )
    if _outer_graph_signature(cell, candidates) != manifest["outer_graph_semantics"]:
        raise ValueError(
            "Fresh outer graph computation settings differ from the artifact"
        )
    state = payload["state_dict"]
    if not isinstance(state, Mapping) or not all(
        torch.is_tensor(value) for value in state.values()
    ):
        raise ValueError("Tensor-only packed state required")
    if any(
        value.device.type != "cpu"
        for value in [*state.values(), *payload["runtime_buffers"].values()]
    ):
        raise ValueError(
            "Restore requires a complete CPU artifact payload before device placement"
        )
    if any(
        value.device.type != "cpu" for value in [*cell.parameters(), *cell.buffers()]
    ):
        raise ValueError("Restore the fresh outer graph on CPU before device placement")
    replacements = []
    for path, spec in manifest["modules"].items():
        source = candidates[path]
        if (
            _class_name(source) != spec["source_class"]
            or source.in_features != spec["in_features"]
            or source.out_features != spec["out_features"]
        ):
            raise ValueError("Fresh projection configuration differs at " + path)
        prefix = path + "."
        matrix = PackedValueMatrix(
            spec["matrix"],
            state[prefix + "matrix.values"],
            state.get(prefix + "matrix.scales"),
        )
        projection = PackedPopulationLinear(
            spec,
            matrix,
            state.get(prefix + "connection_indices"),
            state.get(prefix + "bias"),
        )
        replacements.append((path, projection))
    for path, module in replacements:
        _replace(cell, path, module)
    expected = cell.state_dict()
    if set(expected) != set(state):
        raise ValueError("Missing or unexpected persistent cell state")
    for name, tensor in expected.items():
        actual = state[name]
        if tensor.shape != actual.shape or tensor.dtype != actual.dtype:
            raise ValueError("Persistent shape/dtype drift at " + name)
    cell.load_state_dict(state, strict=True)
    expected_runtime = {}
    for path, module in cell.named_modules():
        for name in module._non_persistent_buffers_set:
            value = module._buffers.get(name)
            if value is not None:
                expected_runtime[(path + "." if path else "") + name] = value
    if set(expected_runtime) != set(payload["runtime_buffers"]):
        raise ValueError("Nonpersistent runtime-buffer set changed")
    for name, expected_tensor in expected_runtime.items():
        value = payload["runtime_buffers"][name]
        if value.shape != expected_tensor.shape or value.dtype != expected_tensor.dtype:
            raise ValueError("Nonpersistent shape/dtype drift at " + name)
        parent, _, key = name.rpartition(".")
        (cell.get_submodule(parent) if parent else cell)._buffers[key] = (
            value.detach().clone().to(expected_tensor.device)
        )
    if (
        packed_population_fingerprint(cell) != manifest["packed_fingerprint"]
        or packed_population_ledger(cell) != manifest["packed_registered_ledger"]
    ):
        raise ValueError("Restored packed state/final registered ledger changed")
    return cell


__all__ = [
    "PackedPopulationLinear",
    "PackedValueMatrix",
    "pack_population_cell_",
    "packed_population_fingerprint",
    "packed_population_ledger",
    "population_cell_payload",
    "repack_population_topology_",
    "restore_packed_population_cell_",
    "set_packed_population_backend_",
]

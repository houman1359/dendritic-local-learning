"""Public storage ledgers for deployed dendritic models.

The compact-checkpoint exporter historically computed its byte accounting with
private helpers. Deployment decisions need the same numbers per top-level
module (retained dense backbone vs dendritic core vs adapters vs index
buffers), so this module exposes an itemized, dtype-aware ledger usable by the
exporter report, the inference benchmark, and analysis collection alike.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

__all__ = [
    "deployment_storage_manifest",
    "model_storage_ledger",
    "module_storage_ledger",
    "serialized_state_ledger",
    "sparse_projection_ledger",
]


def module_storage_ledger(module: nn.Module) -> dict[str, int]:
    """Return exact byte counts for one module subtree.

    Keys: ``parameter_bytes``, ``buffer_bytes``, ``index_bytes`` (the subset of
    buffer bytes held by ``connection_indices`` buffers), and ``total_bytes``.
    Only persistent state is counted — non-persistent inference folds are
    runtime caches, not deployed storage.
    """

    parameter_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
    buffer_bytes = 0
    index_bytes = 0
    for submodule in module.modules():
        non_persistent = getattr(submodule, "_non_persistent_buffers_set", set())
        for name, buffer in submodule._buffers.items():
            if buffer is None or name in non_persistent:
                continue
            size = buffer.numel() * buffer.element_size()
            buffer_bytes += size
            if name.endswith(("connection_indices", "crow_indices", "col_indices")):
                index_bytes += size
    return {
        "parameter_bytes": int(parameter_bytes),
        "buffer_bytes": int(buffer_bytes),
        "index_bytes": int(index_bytes),
        "total_bytes": int(parameter_bytes + buffer_bytes),
    }


def model_storage_ledger(model: nn.Module) -> dict[str, dict[str, int] | dict]:
    """Return per-top-level-module and per-dtype byte ledgers for a model."""

    per_module: dict[str, dict[str, int]] = {}
    for name, child in model.named_children():
        per_module[name] = module_storage_ledger(child)

    per_dtype: dict[str, int] = defaultdict(int)
    for tensor in list(model.parameters()) + list(model.buffers()):
        per_dtype[str(tensor.dtype)] += tensor.numel() * tensor.element_size()

    return {
        "total": module_storage_ledger(model),
        "per_top_level_module": per_module,
        "per_dtype_bytes": dict(per_dtype),
    }


def serialized_state_ledger(
    state_dict: Mapping[str, torch.Tensor],
    *,
    parameter_names: set[str] | None = None,
) -> dict[str, Any]:
    """Account for every tensor byte in one encoded state dictionary.

    The ledger distinguishes learned parameters, topology payload, bitmask
    reconstruction metadata, and other persistent buffers.  Its total is the
    tensor payload size, not the on-disk checkpoint size; the latter must be
    measured from the finished artifact and is reported separately.
    """

    known_parameters = parameter_names or set()
    categories: dict[str, int] = defaultdict(int)
    per_dtype: dict[str, int] = defaultdict(int)
    tensor_count = 0
    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        size = int(tensor.numel() * tensor.element_size())
        tensor_count += 1
        per_dtype[str(tensor.dtype)] += size
        if name.endswith(".__bitmask_metadata"):
            category = "topology_metadata_bytes"
        elif any(
            marker in name
            for marker in ("connection_indices", "crow_indices", "col_indices")
        ):
            category = "topology_bytes"
        elif name in known_parameters:
            category = "parameter_bytes"
        else:
            category = "other_buffer_bytes"
        categories[category] += size
    for category in (
        "parameter_bytes",
        "topology_bytes",
        "topology_metadata_bytes",
        "other_buffer_bytes",
    ):
        categories.setdefault(category, 0)
    total = sum(categories.values())
    return {
        **dict(categories),
        "total_tensor_bytes": int(total),
        "tensor_count": int(tensor_count),
        "per_dtype_bytes": dict(per_dtype),
    }


def sparse_projection_ledger(model: nn.Module) -> dict[str, Any]:
    """Describe stored and active contacts in every sparse projection.

    A training-time candidate pool and its active topology are deliberately
    separate quantities.  After fixed-topology export they must agree; before
    export the difference is retained so compression claims cannot count a
    learnable candidate slot as an already removed contact.
    """

    projections: list[dict[str, Any]] = []
    for path, module in model.named_modules():
        values = getattr(module, "values", None)
        columns = getattr(module, "col_indices", None)
        rows = getattr(module, "crow_indices", None)
        if all(isinstance(value, torch.Tensor) for value in (values, columns, rows)):
            degrees = rows[1:] - rows[:-1]
            active_contacts = int(values.numel())
            topology_bytes = int(
                columns.numel() * columns.element_size()
                + rows.numel() * rows.element_size()
            )
            projections.append(
                {
                    "path": path or "core",
                    "module_type": type(module).__name__,
                    "in_features": int(module.in_features),
                    "out_features": int(module.out_features),
                    "stored_slots_per_output": None,
                    "active_contacts_per_output": None,
                    "realized_k_min": int(degrees.min()),
                    "realized_k_max": int(degrees.max()),
                    "realized_k_mean": float(degrees.float().mean()),
                    "stored_weight_slots": active_contacts,
                    "active_contacts": active_contacts,
                    "weight_bytes": int(values.numel() * values.element_size()),
                    "runtime_index_bytes": topology_bytes,
                    "weight_dtype": str(values.dtype),
                    "index_dtype": str(columns.dtype),
                    "weight_transform": "identity",
                    "projection_backend": "torch_csr",
                    "fixed_topology": True,
                }
            )
            continue
        raw_weight = getattr(module, "pre_w", None)
        if not isinstance(raw_weight, torch.Tensor) or raw_weight.ndim != 2:
            continue
        out_features, stored_per_output = (int(value) for value in raw_weight.shape)
        configured_k = getattr(module, "current_k", getattr(module, "K", None))
        if callable(configured_k):
            configured_k = configured_k()
        active_per_output = (
            stored_per_output if configured_k is None else int(configured_k)
        )
        active_per_output = min(active_per_output, stored_per_output)
        indices = getattr(module, "connection_indices", None)
        index_bytes = (
            int(indices.numel() * indices.element_size())
            if isinstance(indices, torch.Tensor)
            else 0
        )
        record = {
            "path": path or "core",
            "module_type": type(module).__name__,
            "in_features": int(getattr(module, "in_features", stored_per_output)),
            "out_features": out_features,
            "stored_slots_per_output": stored_per_output,
            "active_contacts_per_output": active_per_output,
            "stored_weight_slots": int(raw_weight.numel()),
            "active_contacts": int(out_features * active_per_output),
            "weight_bytes": int(raw_weight.numel() * raw_weight.element_size()),
            "runtime_index_bytes": index_bytes,
            "weight_dtype": str(raw_weight.dtype),
            "index_dtype": (
                str(indices.dtype) if isinstance(indices, torch.Tensor) else None
            ),
            "weight_transform": str(getattr(module, "weight_transform", "unknown")),
            "projection_backend": str(getattr(module, "projection_backend", "unknown")),
            "fixed_topology": bool(
                isinstance(indices, torch.Tensor)
                and stored_per_output == active_per_output
            ),
        }
        projections.append(record)
    return {
        "projections": projections,
        "projection_count": len(projections),
        "stored_weight_slots": int(
            sum(record["stored_weight_slots"] for record in projections)
        ),
        "active_contacts": int(
            sum(record["active_contacts"] for record in projections)
        ),
        "weight_bytes": int(sum(record["weight_bytes"] for record in projections)),
        "runtime_index_bytes": int(
            sum(record["runtime_index_bytes"] for record in projections)
        ),
    }


def deployment_storage_manifest(
    model: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    topology_encoding: str,
    sparse_topology_manifest: list[Mapping[str, Any]] | None = None,
    scope: str = "model",
) -> dict[str, Any]:
    """Build one complete runtime and serialized-tensor deployment ledger."""

    compiled_plans = {
        path or "core": dict(plan)
        for path, module in model.named_modules()
        if isinstance(
            (plan := getattr(module, "compiled_replacement_plan", None)), Mapping
        )
    }
    return {
        "schema": "dendritic_deployment_storage_manifest/v1",
        "scope": str(scope),
        "topology_encoding": str(topology_encoding),
        "runtime_storage": model_storage_ledger(model),
        "serialized_state": serialized_state_ledger(
            state_dict,
            parameter_names=set(dict(model.named_parameters())),
        ),
        "sparse_projections": sparse_projection_ledger(model),
        "compiled_replacement_plans": compiled_plans,
        "analytical_pathway_contracts": {
            path: plan.get("export_contract", {})
            for path, plan in compiled_plans.items()
        },
        "frozen_sparse_topology": [
            dict(record) for record in (sparse_topology_manifest or [])
        ],
    }

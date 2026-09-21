"""Checkpoint loading helpers for transformer replacements."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import random
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from dendritic_modeling.deployment import prepare_model_for_compact_state_
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.architectures.transformer import (
    EIStackMLPSlot,
    unwrap_shared_population_replacement,
)
from dendritic_modeling.networks.checkpoints import (
    atomic_torch_save as _atomic_torch_save,
    compact_sparse_bitmask_state_dict as _compact_sparse_bitmask_state_dict,
    compact_sparse_index_state_dict as _compact_sparse_index_state_dict,
    decode_sparse_bitmask_state_dict as _decode_sparse_bitmask_state_dict,
    sha256_file,
)
from dendritic_modeling.training.replacement_common import ReplacementTrainingHistory

from .distributed import (
    is_transformer_main_process,
    transformer_process_rank,
    transformer_process_world_size,
)

logger = logging.getLogger(__name__)
BEST_REPLACEMENT_CHECKPOINT = "best_replacement_checkpoint.pt"
JOINT_TRAINING_CHECKPOINT_SCHEMA_VERSION = 1


def tensor_state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Content-address tensor state independently of ``torch.save`` metadata.

    Compact topology encodings and their decoded runtime state are intentionally
    different serializations.  This digest therefore identifies whichever
    representation is supplied; callers must not equate a compact digest with
    the digest of its decoded module state.
    """

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"state_dict entry {name!r} is not a tensor")
        cpu = tensor.detach().cpu().contiguous()
        metadata = json.dumps(
            {
                "name": name,
                "dtype": str(cpu.dtype),
                "shape": list(cpu.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _state_key_set_sha256(keys: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(sorted(keys), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _canonical_compiled_plan(value: Mapping[str, Any]) -> str:
    """Canonicalize complete compiler plans while retaining tiny test plans."""

    normalized: Mapping[str, Any] = value
    if "core_config" in value:
        from dendritic_modeling.networks.architectures.replacement import (
            compiled_replacement_plan_from_mapping,
        )

        normalized = compiled_replacement_plan_from_mapping(value).as_dict()
    return json.dumps(
        dict(normalized),
        sort_keys=True,
        separators=(",", ":"),
    )


def validate_prospective_single_cell_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    layer_index: int,
    compiled_plan: Mapping[str, Any],
    compiled_plan_sha256: str,
) -> dict[str, Any]:
    """Fail closed on the modern standalone replacement export contract.

    A prospectively frozen ladder seed must be more than loadable: its plan,
    compact encoding, reconstructive topology, and deployment ledger must all
    describe the same single physical cell.  Legacy and shared/span exports
    remain supported by general checkpoint loading but are inadmissible here.
    """

    from dendritic_modeling.networks.architectures.replacement import (
        compiled_replacement_plan_from_mapping,
    )

    if payload.get("schema_version") != 3:
        raise ValueError("prospective seed checkpoint must use schema_version 3")
    if payload.get("layer_index") != int(layer_index):
        raise ValueError("prospective seed checkpoint embeds another layer")
    if "compiled_plan" in payload:
        raise ValueError("prospective seed checkpoint uses a legacy compiled plan")
    embedded_plan = payload.get("compiled_replacement_plan")
    if not isinstance(embedded_plan, Mapping) or not embedded_plan:
        raise ValueError("prospective seed checkpoint has no compiled replacement plan")
    expected_plan = compiled_replacement_plan_from_mapping(compiled_plan).as_dict()
    observed_plan = compiled_replacement_plan_from_mapping(embedded_plan).as_dict()
    expected_plan_sha256 = _canonical_json_sha256(expected_plan)
    if compiled_plan_sha256 != expected_plan_sha256 or observed_plan != expected_plan:
        raise ValueError("prospective seed checkpoint compiled plan differs")
    if (
        payload.get("parameter_tied_replacement") is not None
        or payload.get("collapsed_replacement_span") is not None
    ):
        raise ValueError("prospective seed checkpoint is not one standalone cell")

    selection = payload.get("selection_manifest")
    if not isinstance(selection, Mapping) or set(selection) != {
        "schema",
        "source",
        "layer_index",
        "plan_sha256",
        "status",
    }:
        raise ValueError("prospective seed checkpoint selection manifest is invalid")
    if selection != {
        "schema": "dendritic_frozen_layer_plan/v1",
        "source": "compiled_plans_by_layer",
        "layer_index": int(layer_index),
        "plan_sha256": expected_plan_sha256,
        "status": "prospectively_frozen_before_training",
    }:
        raise ValueError("prospective seed checkpoint selection binding differs")

    state = payload.get("state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("prospective seed checkpoint has no tensor state")
    serialized_state_sha256 = tensor_state_dict_sha256(state)
    selected_encoding = payload.get("topology_encoding_selected")
    requested_encoding = payload.get("topology_encoding_requested")
    if selected_encoding not in {"uint", "bitmask"} or requested_encoding not in {
        selected_encoding,
        "auto",
    }:
        raise ValueError("prospective seed checkpoint topology encoding is invalid")
    if any(
        key == "_last_forward_param_tensor"
        or key.endswith("._last_forward_param_tensor")
        for key in state
    ):
        raise ValueError("prospective seed checkpoint retains a forward cache tensor")
    decoded_state = _decode_sparse_bitmask_state_dict(state)
    if selected_encoding == "bitmask":
        reencoded_state, expected_index_encoding = _compact_sparse_bitmask_state_dict(
            decoded_state
        )
    else:
        reencoded_state, expected_index_encoding = _compact_sparse_index_state_dict(
            decoded_state
        )
    if (
        tensor_state_dict_sha256(reencoded_state) != serialized_state_sha256
        or payload.get("sparse_index_encoding") != expected_index_encoding
        or not expected_index_encoding
    ):
        raise ValueError("prospective seed checkpoint compact encoding differs")

    topology = payload.get("sparse_topology_manifest")
    if (
        not isinstance(topology, list)
        or not topology
        or any(not isinstance(record, Mapping) for record in topology)
    ):
        raise ValueError("prospective seed checkpoint topology manifest is invalid")
    topology_paths = [record.get("path") for record in topology]
    if any(not isinstance(path, str) or not path for path in topology_paths) or len(
        topology_paths
    ) != len(set(topology_paths)):
        raise ValueError("prospective seed checkpoint topology paths are invalid")
    encoded_paths = {
        key.removesuffix(".connection_indices")
        for key in expected_index_encoding
        if key.endswith(".connection_indices")
    }
    if set(topology_paths) != encoded_paths:
        raise ValueError("prospective seed checkpoint topology coverage differs")
    for record in topology:
        if (
            record.get("target_type") != "IndexedSparseLinear"
            or record.get("source_type") == "IndexedSparseLinear"
            or not isinstance(record.get("in_features"), int)
            or int(record["in_features"]) < 1
            or not isinstance(record.get("out_features"), int)
            or int(record["out_features"]) < 1
            or not isinstance(record.get("synapses_per_output"), int)
            or int(record["synapses_per_output"]) < 1
        ):
            raise ValueError("prospective seed checkpoint topology record is invalid")

    deployment = payload.get("deployment_storage_manifest")
    if not isinstance(deployment, Mapping) or (
        deployment.get("schema") != "dendritic_deployment_storage_manifest/v1"
        or deployment.get("topology_encoding") != selected_encoding
        or deployment.get("frozen_sparse_topology") != topology
    ):
        raise ValueError("prospective seed checkpoint deployment manifest differs")
    deployment_plans = deployment.get("compiled_replacement_plans")
    if not isinstance(deployment_plans, Mapping) or set(deployment_plans) != {"core"}:
        raise ValueError("prospective seed checkpoint deployment plan is invalid")
    if (
        compiled_replacement_plan_from_mapping(deployment_plans["core"]).as_dict()
        != expected_plan
    ):
        raise ValueError("prospective seed checkpoint deployment plan differs")
    sparse_projections = deployment.get("sparse_projections")
    if not isinstance(sparse_projections, Mapping) or not isinstance(
        sparse_projections.get("projections"), list
    ):
        raise ValueError("prospective seed checkpoint deployment topology is invalid")
    projection_paths = [
        projection.get("path")
        for projection in sparse_projections["projections"]
        if isinstance(projection, Mapping)
    ]
    if (
        len(projection_paths) != len(sparse_projections["projections"])
        or sparse_projections.get("projection_count") != len(topology_paths)
        or projection_paths != topology_paths
    ):
        raise ValueError("prospective seed checkpoint deployment coverage differs")

    return {
        "schema": "dendritic_prospective_single_cell_checkpoint_receipt/v1",
        "layer_index": int(layer_index),
        "payload_schema_version": 3,
        "compiled_plan_sha256": expected_plan_sha256,
        "serialized_state_sha256": serialized_state_sha256,
        "topology_encoding_requested": requested_encoding,
        "topology_encoding_selected": selected_encoding,
        "sparse_index_encoding": dict(expected_index_encoding),
        "topology_paths": topology_paths,
        "topology_manifest_sha256": _canonical_json_sha256(topology),
        "deployment_manifest_sha256": _canonical_json_sha256(deployment),
        "verified": True,
    }


def _verify_loaded_replacement_tensors(
    replacement: nn.Module,
    loaded_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Prove every normalized checkpoint tensor reached its runtime buffer.

    ``load_state_dict`` may perform a deliberate dtype/device conversion.  We
    compare against that exact conversion and separately hash the complete
    realized module state, including any explicitly recorded legacy defaults.
    """

    realized = replacement.state_dict()
    missing = sorted(set(loaded_state) - set(realized))
    if missing:
        raise RuntimeError(
            "normalized replacement checkpoint has no realized destination for "
            f"keys {missing[:5]}"
        )
    dtype_casts: list[dict[str, str]] = []
    device_transfers: list[dict[str, str]] = []
    realized_loaded: dict[str, torch.Tensor] = {}
    for key in sorted(loaded_state):
        source = loaded_state[key]
        target = realized[key]
        expected = source.detach().to(device=target.device, dtype=target.dtype)
        if expected.shape != target.shape or not torch.equal(expected, target):
            raise RuntimeError(
                f"replacement checkpoint tensor differs after normalized loading: {key}"
            )
        if source.dtype != target.dtype:
            dtype_casts.append(
                {"key": key, "source": str(source.dtype), "target": str(target.dtype)}
            )
        if source.device != target.device:
            device_transfers.append(
                {
                    "key": key,
                    "source": str(source.device),
                    "target": str(target.device),
                }
            )
        realized_loaded[key] = target
    return {
        "loaded_tensor_count": len(loaded_state),
        "loaded_key_set_sha256": _state_key_set_sha256(list(loaded_state)),
        "realized_loaded_state_sha256": tensor_state_dict_sha256(realized_loaded),
        "dtype_casts": dtype_casts,
        "device_transfers": device_transfers,
        "exact_post_conversion_match": True,
    }


def _resolve_replacement_artifact_layout(
    checkpoint_dir: str,
    layer_indices: Sequence[int],
) -> tuple[dict[int, str], dict[int, int]]:
    """Resolve logical sites to artifacts and their declared physical leaders.

    Manifest v1 describes independent layer artifacts, v2 is the alias-aware
    parameter-tying format, and v3 describes independent exit artifacts for
    collapsed spans. All content-addressed formats must cover the configured
    artifact layers exactly and pass byte/hash verification. No manifest
    retains the legacy one-file-per-layer convention.
    """

    expected = [int(layer) for layer in layer_indices]
    if len(expected) != len(set(expected)):
        raise ValueError("replacement artifact layers must be unique")
    manifest_path = os.path.join(checkpoint_dir, "replacement_export_manifest.json")
    if not os.path.isfile(manifest_path):
        paths = {
            layer: os.path.join(checkpoint_dir, f"layer_{layer}_replacement.pt")
            for layer in expected
        }
        return paths, {layer: layer for layer in expected}
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest_schema = manifest.get("schema")
    if manifest_schema not in {
        "dendritic_replacement_artifact_manifest/v1",
        "dendritic_replacement_artifact_manifest/v2",
        "dendritic_replacement_artifact_manifest/v3",
    }:
        paths = {
            layer: os.path.join(checkpoint_dir, f"layer_{layer}_replacement.pt")
            for layer in expected
        }
        return paths, {layer: layer for layer in expected}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("shared replacement artifact manifest has no artifacts")
    resolved: dict[int, str] = {}
    leaders: dict[int, int] = {}
    for artifact_index, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise RuntimeError(
                f"shared replacement artifact {artifact_index} is not a mapping"
            )
        leader = artifact.get("layer_index")
        aliases = artifact.get("alias_layers", [])
        if manifest_schema == "dendritic_replacement_artifact_manifest/v3" and aliases:
            raise RuntimeError(
                "collapsed replacement artifacts cannot declare parameter aliases"
            )
        if isinstance(leader, bool) or not isinstance(leader, Integral):
            raise RuntimeError("shared replacement artifact leader must be an integer")
        if isinstance(aliases, (str, bytes)) or not isinstance(aliases, Sequence):
            raise RuntimeError("shared replacement alias_layers must be a sequence")
        if any(
            isinstance(alias, bool) or not isinstance(alias, Integral)
            for alias in aliases
        ):
            raise RuntimeError("shared replacement aliases must be integers")
        leader = int(leader)
        aliases = [int(alias) for alias in aliases]
        if aliases != sorted(aliases) or len(aliases) != len(set(aliases)):
            raise RuntimeError(
                "shared replacement artifact aliases must be sorted and unique"
            )
        if leader in aliases:
            raise RuntimeError("shared replacement leader cannot also be an alias")
        checkpoint = artifact.get("checkpoint")
        expected_name = f"layer_{leader}_replacement.pt"
        if checkpoint != expected_name:
            raise RuntimeError(
                "shared replacement artifact checkpoint must use its leader name"
            )
        path = os.path.join(checkpoint_dir, expected_name)
        expected_bytes = artifact.get("compact_checkpoint_bytes")
        expected_sha256 = artifact.get("sha256")
        if (
            isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, Integral)
            or int(expected_bytes) < 1
        ):
            raise RuntimeError(
                "replacement artifact manifest must record a positive byte size"
            )
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise RuntimeError(
                "replacement artifact manifest must record a SHA-256 digest"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        observed_bytes = os.path.getsize(path)
        if observed_bytes != int(expected_bytes):
            raise RuntimeError(
                f"replacement artifact size mismatch for {expected_name}: "
                f"expected {int(expected_bytes)}, observed {observed_bytes}"
            )
        observed_sha256 = sha256_file(path)
        if observed_sha256 != expected_sha256.lower():
            raise RuntimeError(
                f"replacement artifact SHA-256 mismatch for {expected_name}: "
                f"expected {expected_sha256.lower()}, observed {observed_sha256}"
            )
        for layer in [leader, *aliases]:
            if layer in resolved:
                raise RuntimeError(
                    f"shared replacement artifact layer {layer} is declared twice"
                )
            resolved[layer] = path
            leaders[layer] = leader
    if set(resolved) != set(expected):
        missing = sorted(set(expected) - set(resolved))
        extra = sorted(set(resolved) - set(expected))
        raise RuntimeError(
            "shared replacement artifact coverage does not match configured layers; "
            f"missing={missing}, extra={extra}"
        )
    return (
        {layer: resolved[layer] for layer in expected},
        {layer: leaders[layer] for layer in expected},
    )


def _resolve_replacement_artifact_paths(
    checkpoint_dir: str,
    layer_indices: Sequence[int],
) -> dict[int, str]:
    """Resolve logical replacement sites to compact checkpoint paths."""

    paths, _leaders = _resolve_replacement_artifact_layout(
        checkpoint_dir,
        layer_indices,
    )
    return paths


def _configured_replacement_alias_leaders(
    records: Sequence[Any],
) -> dict[int, int]:
    """Describe the physical alias graph installed by the current config."""

    by_physical_id: dict[int, list[Any]] = {}
    for record in records:
        physical = unwrap_shared_population_replacement(record.replacement)
        by_physical_id.setdefault(id(physical), []).append(record)

    leaders: dict[int, int] = {}
    for members in by_physical_id.values():
        member_layers = sorted(int(member.layer_index) for member in members)
        physical_leader = member_layers[0]
        if len(members) > 1:
            expected_group = tuple(member_layers)
            for member in members:
                declared_leader = getattr(member, "tied_group_leader", None)
                declared_layers = tuple(
                    int(layer) for layer in getattr(member, "tied_group_layers", ())
                )
                if (
                    declared_leader is None
                    or int(declared_leader) != physical_leader
                    or declared_layers != expected_group
                ):
                    raise RuntimeError(
                        "configured shared replacement metadata does not match "
                        "the installed physical alias graph"
                    )
        else:
            member = members[0]
            declared_leader = getattr(member, "tied_group_leader", None)
            if declared_leader is not None:
                raise RuntimeError(
                    "configured parameter-tied replacement group is incomplete in the "
                    "checkpoint load set"
                )
        for layer in member_layers:
            leaders[layer] = physical_leader
    return leaders


def _remap_single_layer_stack_state_dict(
    replacement: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    """Map a single-layer EI-stack checkpoint into one slot of a shared stack."""
    if not isinstance(replacement, EIStackMLPSlot):
        return None

    slot_index = int(replacement.layer_index)
    remapped: dict[str, torch.Tensor] = {}
    prefix_map = {
        "stack.layers.0.": f"stack.layers.{slot_index}.",
        "stack.output_projections.0.": f"stack.output_projections.{slot_index}.",
        "stack.pre_norm.0.": f"stack.pre_norm.{slot_index}.",
    }
    for key, value in state_dict.items():
        for old_prefix, new_prefix in prefix_map.items():
            if key.startswith(old_prefix):
                remapped[f"{new_prefix}{key[len(old_prefix) :]}"] = value
                break

    return remapped or None


def _load_replacement_state_dict(
    replacement: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    sparse_topology_manifest: Sequence[Mapping[str, Any]] | None = None,
    *,
    verify_semantic_identity: bool = False,
) -> dict[str, Any] | None:
    """Load a replacement checkpoint and return a verified semantic receipt.

    The serialized state may use a compact bitmask topology and therefore must
    not be compared byte-for-byte with ``replacement.state_dict()``.  The
    receipt binds the compact source, its deterministic decoded form, the
    effective key mapping, every deliberate dtype/device conversion, and the
    complete realized module state.
    """
    replacement = unwrap_shared_population_replacement(replacement)
    serialized_state_sha256 = (
        tensor_state_dict_sha256(state_dict) if verify_semantic_identity else None
    )
    # Older identity-transform checkpoints could register this non-persistent
    # forward cache as a duplicate alias of ``pre_w``. It is never model state.
    ignored_cache_keys = sorted(
        key
        for key in state_dict
        if key.endswith("._last_forward_param_tensor")
        or key == "_last_forward_param_tensor"
    )
    compact_bitmask_keys = sorted(
        key for key in state_dict if key.endswith(".__bitmask")
    )
    filtered_state = {
        key: value for key, value in state_dict.items() if key not in ignored_cache_keys
    }
    normalized_state = _decode_sparse_bitmask_state_dict(filtered_state)
    normalized_state_sha256 = (
        tensor_state_dict_sha256(normalized_state) if verify_semantic_identity else None
    )
    compact_roundtrip_state_sha256 = None
    compact_roundtrip_encoding: dict[str, str] | None = None
    if compact_bitmask_keys and verify_semantic_identity:
        reencoded_state, compact_roundtrip_encoding = (
            _compact_sparse_bitmask_state_dict(normalized_state)
        )
        compact_roundtrip_state_sha256 = tensor_state_dict_sha256(reencoded_state)
        if set(reencoded_state) != set(
            filtered_state
        ) or compact_roundtrip_state_sha256 != tensor_state_dict_sha256(filtered_state):
            raise RuntimeError(
                "decoded replacement bitmask state does not reproduce its exact "
                "compact serialization"
            )
    topology_manifest_sha256 = (
        _canonical_json_sha256(list(sparse_topology_manifest))
        if sparse_topology_manifest and verify_semantic_identity
        else None
    )
    if sparse_topology_manifest:
        prepare_model_for_compact_state_(
            replacement,
            list(sparse_topology_manifest),
            normalized_state,
        )
        replacement._loaded_sparse_topology_manifest = [
            dict(record) for record in sparse_topology_manifest
        ]
    remapped = _remap_single_layer_stack_state_dict(replacement, normalized_state)
    mapping_mode = "identity"
    effective_state = normalized_state
    compatibility_initialized_keys: list[str] = []
    if remapped is not None and set(normalized_state) != set(replacement.state_dict()):
        mapping_mode = "single_layer_ei_stack_slot"
        effective_state = remapped
        _copy_replacement_state_dict(replacement, remapped)
    else:
        try:
            replacement.load_state_dict(normalized_state)
        except RuntimeError as exc:
            if remapped is not None:
                mapping_mode = "single_layer_ei_stack_slot"
                effective_state = remapped
                _copy_replacement_state_dict(replacement, remapped)
            else:
                current_keys = set(replacement.state_dict())
                loaded_keys = set(normalized_state)
                missing = sorted(current_keys - loaded_keys)
                unexpected = sorted(loaded_keys - current_keys)
                if (
                    unexpected
                    or not missing
                    or not all(
                        key == "rewire_step" or key.endswith(".rewire_step")
                        for key in missing
                    )
                ):
                    raise exc
                incompatible = replacement.load_state_dict(
                    normalized_state, strict=False
                )
                if sorted(incompatible.missing_keys) != missing or (
                    incompatible.unexpected_keys
                ):
                    raise exc
                missing_set = set(missing)
                for name, module in replacement.named_modules():
                    key = f"{name}.rewire_step" if name else "rewire_step"
                    if key in missing_set and isinstance(module, IndexedRewireLinear):
                        module.freeze_connectivity = True
                replacement._loaded_frozen_rewire_compatibility = missing
                compatibility_initialized_keys = missing

    if not verify_semantic_identity:
        return None

    verification = _verify_loaded_replacement_tensors(replacement, effective_state)
    realized_state = replacement.state_dict()
    semantic_complete = set(effective_state) == set(realized_state)
    receipt = {
        "schema": "dendritic_replacement_checkpoint_load_receipt/v1",
        "serialized_state_sha256": serialized_state_sha256,
        "serialized_tensor_count": len(state_dict),
        "ignored_nonpersistent_cache_keys": ignored_cache_keys,
        "compact_bitmask_keys": compact_bitmask_keys,
        "compact_roundtrip_state_sha256": compact_roundtrip_state_sha256,
        "compact_roundtrip_encoding": compact_roundtrip_encoding,
        "compact_roundtrip_exact": bool(compact_bitmask_keys),
        "normalized_state_sha256": normalized_state_sha256,
        "normalized_tensor_count": len(normalized_state),
        "normalized_key_set_sha256": _state_key_set_sha256(list(normalized_state)),
        "topology_manifest_sha256": topology_manifest_sha256,
        "mapping_mode": mapping_mode,
        "compatibility_initialized_keys": compatibility_initialized_keys,
        "semantic_state_complete": semantic_complete,
        "realized_module_tensor_count": len(realized_state),
        "realized_module_state_sha256": tensor_state_dict_sha256(realized_state),
        **verification,
    }
    replacement._checkpoint_load_receipt = receipt
    return receipt


def _copy_replacement_state_dict(
    replacement: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    current_state = replacement.state_dict()
    missing = [key for key in state_dict if key not in current_state]
    if missing:
        raise RuntimeError(
            f"EI-stack checkpoint keys do not match the replacement slot: {missing[:5]}"
        )
    with torch.no_grad():
        for key, value in state_dict.items():
            target = current_state[key]
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def _load_replacement_record_checkpoints(
    records: Sequence[Any],
    checkpoint_dir: str,
    *,
    device: torch.device,
) -> None:
    if not checkpoint_dir:
        return
    layer_indices = [int(record.layer_index) for record in records]
    artifact_paths, artifact_leaders = _resolve_replacement_artifact_layout(
        checkpoint_dir,
        layer_indices,
    )
    configured_leaders = _configured_replacement_alias_leaders(records)
    if artifact_leaders != configured_leaders:
        raise RuntimeError(
            "replacement artifact alias graph does not exactly match the "
            "configured physical alias graph"
        )
    loaded: set[int] = set()
    for record in records:
        physical = unwrap_shared_population_replacement(record.replacement)
        if id(physical) in loaded:
            continue
        loaded.add(id(physical))
        leader = getattr(record, "tied_group_leader", None)
        checkpoint_layer = int(record.layer_index) if leader is None else int(leader)
        checkpoint_path = artifact_paths[int(record.layer_index)]
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=device)
        if isinstance(payload, Mapping) and "layer_index" in payload:
            if int(payload["layer_index"]) != checkpoint_layer:
                raise RuntimeError(
                    "replacement checkpoint layer does not match its configured site"
                )
        state_dict = payload.get("state_dict", payload)
        tied_manifest = payload.get("parameter_tied_replacement", {})
        if leader is not None:
            expected_layers = list(getattr(record, "tied_group_layers", ()))
            if (
                int(tied_manifest.get("leader_layer", -1)) != checkpoint_layer
                or list(tied_manifest.get("alias_layers", [])) != expected_layers[1:]
            ):
                raise RuntimeError(
                    "parameter-tied replacement checkpoint alias manifest does not "
                    f"match configured group {expected_layers}"
                )
        collapsed_manifest = payload.get("collapsed_replacement_span", {})
        expected_span = list(getattr(record, "collapsed_span_layers", ()))
        if expected_span:
            expected_norm_attr = getattr(
                record, "collapsed_span_post_mlp_norm_attr", ""
            )
            if (
                collapsed_manifest.get("post_mlp_norm_attr", "") != expected_norm_attr
                or list(collapsed_manifest.get("post_mlp_norm_removed_layers", []))
                != (expected_span if expected_norm_attr else [])
                or (
                    expected_norm_attr
                    and collapsed_manifest.get("cell_output_boundary")
                    != "post_mlp_norm_residual_branch"
                )
            ):
                raise RuntimeError(
                    "collapsed replacement checkpoint post-MLP norm boundary does "
                    f"not match configured span {expected_span}"
                )
            if (
                list(collapsed_manifest.get("span_layers", [])) != expected_span
                or int(collapsed_manifest.get("exit_layer", -1))
                != int(record.layer_index)
                or int(collapsed_manifest.get("cell_application_count", -1)) != 1
            ):
                raise RuntimeError(
                    "collapsed replacement checkpoint manifest does not match "
                    f"configured span {expected_span}"
                )
            checkpoint_plan = payload.get("compiled_replacement_plan")
            configured_plan = getattr(
                record.replacement,
                "compiled_replacement_plan",
                None,
            )
            if not isinstance(checkpoint_plan, Mapping) or not isinstance(
                configured_plan, Mapping
            ):
                raise RuntimeError(
                    "collapsed replacement checkpoint and configured cell must "
                    "both expose compiler plans"
                )
            checkpoint_plan_canonical = _canonical_compiled_plan(checkpoint_plan)
            configured_plan_canonical = _canonical_compiled_plan(configured_plan)
            if checkpoint_plan_canonical != configured_plan_canonical:
                raise RuntimeError(
                    "collapsed replacement checkpoint compiler plan does not "
                    f"match configured span {expected_span}"
                )
        elif collapsed_manifest:
            raise RuntimeError(
                "collapsed replacement checkpoint cannot load into an ordinary layer"
            )
        else:
            checkpoint_plan = payload.get("compiled_replacement_plan")
            configured_plan = getattr(
                physical,
                "compiled_replacement_plan",
                None,
            )
            if checkpoint_plan or configured_plan:
                if not isinstance(checkpoint_plan, Mapping) or not isinstance(
                    configured_plan, Mapping
                ):
                    raise RuntimeError(
                        "ordinary replacement checkpoint and configured cell must "
                        "both expose compiler plans"
                    )
                checkpoint_plan_canonical = _canonical_compiled_plan(checkpoint_plan)
                configured_plan_canonical = _canonical_compiled_plan(configured_plan)
                if checkpoint_plan_canonical != configured_plan_canonical:
                    raise RuntimeError(
                        "ordinary replacement checkpoint compiler plan does not "
                        f"match configured layer {checkpoint_layer}"
                    )
        _load_replacement_state_dict(
            physical,
            state_dict,
            (
                payload.get("sparse_topology_manifest")
                if isinstance(payload, dict)
                else None
            ),
        )


def _unique_replacement_checkpoint_payload(
    units: Sequence[Any],
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, Any]]]:
    """Serialize each physical replacement once and describe every alias site."""

    states: list[dict[str, torch.Tensor]] = []
    layout: list[dict[str, Any]] = []
    state_index_by_id: dict[int, int] = {}
    leader_by_id: dict[int, int] = {}
    for unit in units:
        physical = unwrap_shared_population_replacement(unit.replacement)
        identity = id(physical)
        state_index = state_index_by_id.get(identity)
        if state_index is None:
            state_index = len(states)
            state_index_by_id[identity] = state_index
            leader = getattr(unit, "tied_group_leader", None)
            leader_by_id[identity] = (
                int(unit.layer_index) if leader is None else int(leader)
            )
            states.append(_compact_sparse_index_state_dict(physical.state_dict())[0])
        layout.append(
            {
                "layer_index": int(unit.layer_index),
                "state_index": int(state_index),
                "leader_layer": int(leader_by_id[identity]),
                **(
                    {
                        "collapsed_span_post_mlp_norm_attr": unit.collapsed_span_post_mlp_norm_attr,
                        "collapsed_span_layers": list(unit.collapsed_span_layers),
                    }
                    if getattr(unit, "collapsed_span_post_mlp_norm_attr", "")
                    else {}
                ),
            }
        )
    return states, layout


def _load_replacement_checkpoint_payload(
    units: Sequence[Any],
    states: Sequence[Mapping[str, torch.Tensor]],
    layout: Sequence[Mapping[str, Any]] | None,
) -> None:
    """Restore unique states and fail closed on an incompatible alias graph."""

    physical = [
        unwrap_shared_population_replacement(unit.replacement) for unit in units
    ]
    if layout is None:
        if any(
            getattr(unit, "collapsed_span_post_mlp_norm_attr", "") for unit in units
        ):
            raise RuntimeError(
                "legacy replacement checkpoint cannot prove the post-MLP norm boundary"
            )
        if len({id(module) for module in physical}) != len(physical):
            raise RuntimeError(
                "legacy replacement checkpoint duplicates a configured shared "
                "parameter owner and cannot prove a valid alias graph"
            )
        if len(states) != len(units):
            raise RuntimeError(
                "Replacement checkpoint unit count does not match the current config"
            )
        for module, state_dict in zip(physical, states):
            _load_replacement_state_dict(module, state_dict)
        return

    if len(layout) != len(units):
        raise RuntimeError("Replacement checkpoint alias-layout count mismatch")
    state_modules: dict[int, nn.Module] = {}
    module_state_indices: dict[int, int] = {}
    for unit, module, entry in zip(units, physical, layout):
        norm_attr = getattr(unit, "collapsed_span_post_mlp_norm_attr", "")
        if entry.get("collapsed_span_post_mlp_norm_attr", "") != norm_attr or (
            norm_attr
            and list(entry.get("collapsed_span_layers", []))
            != list(unit.collapsed_span_layers)
        ):
            raise RuntimeError("replacement checkpoint post-MLP norm boundary mismatch")
        if int(entry.get("layer_index", -1)) != int(unit.layer_index):
            raise RuntimeError("Replacement checkpoint layer ordering mismatch")
        state_index = int(entry.get("state_index", -1))
        if state_index < 0 or state_index >= len(states):
            raise RuntimeError("Replacement checkpoint state index is invalid")
        expected_leader = getattr(unit, "tied_group_leader", None)
        expected_leader = (
            int(unit.layer_index) if expected_leader is None else int(expected_leader)
        )
        if int(entry.get("leader_layer", -1)) != expected_leader:
            raise RuntimeError("Replacement checkpoint alias leader mismatch")
        prior = state_modules.get(state_index)
        if prior is not None and prior is not module:
            raise RuntimeError(
                "Checkpoint aliases one state across independently constructed modules"
            )
        prior_state_index = module_state_indices.get(id(module))
        if prior_state_index is not None and prior_state_index != state_index:
            raise RuntimeError(
                "Configured shared module maps to multiple checkpoint states"
            )
        state_modules[state_index] = module
        module_state_indices[id(module)] = state_index
    if set(state_modules) != set(range(len(states))):
        raise RuntimeError("Replacement checkpoint contains unreferenced states")
    for state_index, module in state_modules.items():
        _load_replacement_state_dict(module, states[state_index])


def _worker_checkpoint_path(path: str, rank: int) -> str:
    root, extension = os.path.splitext(path)
    return f"{root}.rank{int(rank)}{extension or '.pt'}"


def _save_layerwise_training_checkpoint(
    path: str,
    units: Sequence[Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    history: ReplacementTrainingHistory,
    *,
    step: int,
    device: torch.device,
) -> None:
    """Save common model state plus rank-local RNG state at one DDP barrier."""
    rank = transformer_process_rank()
    worker_payload: dict[str, Any] = {
        "schema_version": 1,
        "rank": rank,
        "step": int(step),
        "torch_rng_state": torch.get_rng_state(),
    }
    if device.type == "cuda":
        worker_payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    _atomic_torch_save(worker_payload, _worker_checkpoint_path(path, rank))

    if is_transformer_main_process():
        replacement_states, replacement_layout = _unique_replacement_checkpoint_payload(
            units
        )
        has_layout_metadata = len(replacement_states) < len(units) or any(
            getattr(unit, "collapsed_span_post_mlp_norm_attr", "") for unit in units
        )
        _atomic_torch_save(
            {
                "schema_version": 2 if has_layout_metadata else 1,
                "step": int(step),
                "replacement_state_dicts": replacement_states,
                **(
                    {"replacement_alias_layout": replacement_layout}
                    if has_layout_metadata
                    else {}
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": dataclasses.asdict(history),
            },
            path,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _load_layerwise_training_checkpoint(
    path: str,
    units: Sequence[Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    device: torch.device,
) -> tuple[int, ReplacementTrainingHistory]:
    """Restore a checkpoint produced by `_save_layerwise_training_checkpoint`."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    states = payload.get("replacement_state_dicts", [])
    _load_replacement_checkpoint_payload(
        units,
        states,
        payload.get("replacement_alias_layout"),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload.get("scaler_state_dict", {}))
    history = ReplacementTrainingHistory(**payload["history"])

    worker_path = _worker_checkpoint_path(path, transformer_process_rank())
    if os.path.isfile(worker_path):
        worker = torch.load(worker_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(worker["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in worker:
            torch.cuda.set_rng_state(worker["cuda_rng_state"], device)
    else:
        logger.warning("Rank-local checkpoint not found: %s", worker_path)
    return int(payload["step"]), history


def _save_joint_training_checkpoint(
    path: str,
    units: Sequence[Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    history: ReplacementTrainingHistory,
    validation_metrics_history: Sequence[Mapping[str, Any]],
    token_source: Any,
    restart_contract: Mapping[str, Any],
    *,
    step: int,
    device: torch.device,
    save_dir: str,
    ddp_integrity_history: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    """Atomically save an exact joint-DDP restart at one global step."""

    world_size = transformer_process_world_size()
    candidate_paths = [path] + [
        _worker_checkpoint_path(path, candidate_rank)
        for candidate_rank in range(world_size)
    ]
    existing = [candidate for candidate in candidate_paths if os.path.exists(candidate)]
    if existing:
        raise FileExistsError(
            "immutable joint checkpoint path already exists: " + ", ".join(existing)
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    rank = transformer_process_rank()
    stream_state = (
        token_source.state_dict() if hasattr(token_source, "state_dict") else None
    )
    worker_payload: dict[str, Any] = {
        "schema_version": JOINT_TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "step": int(step),
        "torch_rng_state": torch.get_rng_state(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "token_source_type": type(token_source).__name__,
        "token_stream_state": stream_state,
    }
    if device.type == "cuda":
        worker_payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    _atomic_torch_save(worker_payload, _worker_checkpoint_path(path, rank))

    if is_transformer_main_process():
        replacement_states, replacement_layout = _unique_replacement_checkpoint_payload(
            units
        )
        has_layout_metadata = len(replacement_states) < len(units) or any(
            getattr(unit, "collapsed_span_post_mlp_norm_attr", "") for unit in units
        )
        best_path = os.path.join(save_dir, BEST_REPLACEMENT_CHECKPOINT)
        best_payload = None
        if os.path.isfile(best_path):
            best_payload = torch.load(
                best_path,
                map_location="cpu",
                weights_only=False,
            )
        _atomic_torch_save(
            {
                "schema_version": JOINT_TRAINING_CHECKPOINT_SCHEMA_VERSION,
                "step": int(step),
                "world_size": world_size,
                "replacement_state_dicts": replacement_states,
                **(
                    {"replacement_alias_layout": replacement_layout}
                    if has_layout_metadata
                    else {}
                ),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": dataclasses.asdict(history),
                "validation_metrics_history": [
                    dict(record) for record in validation_metrics_history
                ],
                "restart_contract": dict(restart_contract),
                "ddp_integrity_history": [
                    dict(record) for record in (ddp_integrity_history or [])
                ],
                "best_replacement_checkpoint": best_payload,
            },
            path,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _load_joint_training_checkpoint(
    path: str,
    units: Sequence[Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    token_source: Any,
    expected_restart_contract: Mapping[str, Any],
    *,
    device: torch.device,
    save_dir: str,
    ddp_integrity_history: list[dict[str, Any]] | None = None,
) -> tuple[int, ReplacementTrainingHistory, list[dict[str, Any]]]:
    """Restore a complete joint-DDP checkpoint and its rank-local stream."""

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("schema_version") != JOINT_TRAINING_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("joint checkpoint schema version differs")
    step = int(payload.get("step", -1))
    world_size = transformer_process_world_size()
    if int(payload.get("world_size", -1)) != world_size:
        raise RuntimeError("joint checkpoint world size differs")
    if payload.get("restart_contract") != dict(expected_restart_contract):
        raise RuntimeError("joint checkpoint restart contract differs")
    states = payload.get("replacement_state_dicts", [])
    _load_replacement_checkpoint_payload(
        units,
        states,
        payload.get("replacement_alias_layout"),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload.get("scaler_state_dict", {}))
    history = ReplacementTrainingHistory(**payload["history"])
    if len(history.train_losses) != step:
        raise RuntimeError("joint checkpoint history does not end at its step")
    validation_history_raw = payload.get("validation_metrics_history")
    if not isinstance(validation_history_raw, list) or not validation_history_raw:
        raise RuntimeError("joint checkpoint validation history is missing")
    validation_history = [dict(record) for record in validation_history_raw]
    if any(
        not isinstance(record.get("step"), Integral)
        or not 0 <= int(record["step"]) <= step
        for record in validation_history
    ):
        raise RuntimeError("joint checkpoint validation history step is invalid")

    saved_integrity_raw = payload.get("ddp_integrity_history", [])
    if not isinstance(saved_integrity_raw, list) or any(
        not isinstance(record, Mapping) for record in saved_integrity_raw
    ):
        raise RuntimeError("joint checkpoint DDP integrity history is invalid")
    saved_integrity = [dict(record) for record in saved_integrity_raw]
    execution = expected_restart_contract.get("execution_contract", {})
    cascade_restart = (
        isinstance(execution, Mapping)
        and execution.get("schema")
        == "dendritic_modelopt_then_dendritic_restart_identity/v1"
    )
    if cascade_restart and not saved_integrity:
        raise RuntimeError("cascade restart checkpoint omits DDP integrity history")
    if ddp_integrity_history is not None:
        if ddp_integrity_history:
            raise ValueError("DDP integrity restore target must initially be empty")
        ddp_integrity_history.extend(saved_integrity)

    best_payload = payload.get("best_replacement_checkpoint")
    if best_payload is not None and is_transformer_main_process():
        _atomic_torch_save(
            best_payload,
            os.path.join(save_dir, BEST_REPLACEMENT_CHECKPOINT),
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    rank = transformer_process_rank()
    worker_path = _worker_checkpoint_path(path, rank)
    if not os.path.isfile(worker_path):
        raise FileNotFoundError(worker_path)
    worker = torch.load(worker_path, map_location="cpu", weights_only=False)
    expected_worker = {
        "schema_version": JOINT_TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "rank": rank,
        "world_size": world_size,
        "step": step,
        "token_source_type": type(token_source).__name__,
    }
    for key, expected in expected_worker.items():
        if worker.get(key) != expected:
            raise RuntimeError(
                f"joint rank-local checkpoint {key} differs: "
                f"{worker.get(key)!r} != {expected!r}"
            )
    stream_state = worker.get("token_stream_state")
    if stream_state is None:
        if hasattr(token_source, "load_state_dict"):
            raise RuntimeError("joint checkpoint omits a restartable token stream")
    elif not hasattr(token_source, "load_state_dict"):
        raise RuntimeError("joint checkpoint stream cannot be restored by this source")
    else:
        token_source.load_state_dict(stream_state)

    torch.set_rng_state(worker["torch_rng_state"])
    random.setstate(worker["python_rng_state"])
    np.random.set_state(worker["numpy_rng_state"])
    if device.type == "cuda":
        if "cuda_rng_state" not in worker:
            raise RuntimeError("joint CUDA checkpoint omits rank-local CUDA RNG state")
        torch.cuda.set_rng_state(worker["cuda_rng_state"], device)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return step, history, validation_history


def _save_best_replacement_checkpoint(
    save_dir: str,
    units: Sequence[Any],
    *,
    step: int,
    valid_loss: float,
) -> str:
    """Persist a validation-best layerwise state, including the initial state."""
    path = os.path.join(save_dir, BEST_REPLACEMENT_CHECKPOINT)
    if is_transformer_main_process():
        replacement_states, replacement_layout = _unique_replacement_checkpoint_payload(
            units
        )
        has_layout_metadata = len(replacement_states) < len(units) or any(
            getattr(unit, "collapsed_span_post_mlp_norm_attr", "") for unit in units
        )
        _atomic_torch_save(
            {
                "schema_version": 2 if has_layout_metadata else 1,
                "step": int(step),
                "valid_loss": float(valid_loss),
                "replacement_state_dicts": replacement_states,
                **(
                    {"replacement_alias_layout": replacement_layout}
                    if has_layout_metadata
                    else {}
                ),
            },
            path,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return path


def _restore_best_replacement_checkpoint(
    save_dir: str,
    units: Sequence[Any],
    *,
    device: torch.device,
) -> tuple[int, float] | None:
    """Restore the exported layerwise modules to their best validation state."""
    path = os.path.join(save_dir, BEST_REPLACEMENT_CHECKPOINT)
    if not os.path.isfile(path):
        logger.warning("Best replacement checkpoint not found: %s", path)
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    states = payload.get("replacement_state_dicts", [])
    _load_replacement_checkpoint_payload(
        units,
        states,
        payload.get("replacement_alias_layout"),
    )
    return int(payload["step"]), float(payload["valid_loss"])


__all__ = [
    "_compact_sparse_bitmask_state_dict",
    "_compact_sparse_index_state_dict",
    "_copy_replacement_state_dict",
    "_decode_sparse_bitmask_state_dict",
    "_load_joint_training_checkpoint",
    "_load_layerwise_training_checkpoint",
    "_load_replacement_record_checkpoints",
    "_load_replacement_state_dict",
    "_remap_single_layer_stack_state_dict",
    "_resolve_replacement_artifact_paths",
    "_restore_best_replacement_checkpoint",
    "_save_best_replacement_checkpoint",
    "_save_joint_training_checkpoint",
    "_save_layerwise_training_checkpoint",
    "tensor_state_dict_sha256",
    "validate_prospective_single_cell_checkpoint_payload",
]

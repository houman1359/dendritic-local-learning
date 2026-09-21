"""Fail-closed dense-only ModelOpt packing around PopulationNetwork cells.

The dendritic replacement modules are custom executable structures, not
ModelOpt-native dense layers.  This module therefore treats the outermost
canonical PopulationNetwork replacement at every model site as an immutable
protected root.  Only explicitly enumerated ``torch.nn.Linear`` modules outside
those roots may receive a ModelOpt quantizer, and every protected tensor,
topology buffer, module type, and compiled-plan identity must remain unchanged.

``modelopt.torch.quantization.quantize`` alone is fake-Q/DQ.  A physical
artifact additionally requires ``modelopt.torch.quantization.compress`` and
must contain packed ``QTensorWrapper`` weights at exactly the frozen target
paths.  A ModelOpt ``mto.save`` checkpoint is reloadable only when the caller
first reconstructs the same custom PopulationNetwork architecture; an ordinary
Hugging Face model factory cannot infer those cells from the base OLMo config.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationNetwork,
)
from dendritic_modeling.networks.architectures.transformer.modules import (
    CollapsedPopulationNetworkSpanExit,
    GatedPopulationNetworkFFNReplacement,
    PopulationNetworkFFNReplacement,
    TiedPopulationNetworkFFNSite,
)

DENSE_ONLY_SCHEMA = "dendritic_modelopt_dense_only_physical/v1"
PROTECTED_CELL_SCHEMA = "dendritic_modelopt_protected_population_cell/v1"
MODELOPT_ONLY_SCHEMA = "dendritic_modelopt_zero_population_cells/v1"
PACKED_AUDIT_SCHEMA = "dendritic_modelopt_dense_only_packed_audit/v1"
DEVICE_AUDIT_SCHEMA = "dendritic_modelopt_dense_only_device_audit/v1"
INT4_BLOCK_SIZE = 128

FAKE_QDQ_CLAIM_BOUNDARY = (
    "ModelOpt quantize() is calibration/fake-Q-DQ evidence only; it is not a "
    "physical compressed artifact."
)
PHYSICAL_ARTIFACT_CLAIM_BOUNDARY = (
    "A physical ModelOpt PyTorch artifact requires compress(), packed "
    "QTensorWrapper weights at exactly the frozen native-dense allowlist, an "
    "exact protected-cell hash bracket, and save/restore parity from the same "
    "custom composed-model factory. It is not a standard Hugging Face or "
    "TensorRT-LLM artifact."
)
MODELOPT_ONLY_CLAIM_BOUNDARY = (
    "A zero-cell physical ModelOpt PyTorch artifact requires compress(), packed "
    "QTensorWrapper weights at exactly the frozen native-dense allowlist, no "
    "PopulationNetwork roots before or after packing, and save/restore parity "
    "from the same native model architecture. This audit alone establishes "
    "neither standalone loading nor quantized capability or runtime."
)

_PROTECTED_TYPES = (
    PopulationNetworkFFNReplacement,
    GatedPopulationNetworkFFNReplacement,
    CollapsedPopulationNetworkSpanExit,
    TiedPopulationNetworkFFNSite,
    PopulationNetwork,
)


@dataclass(frozen=True)
class DenseOnlyInventory:
    """Frozen protected roots and native-dense target paths."""

    protected_roots: tuple[str, ...]
    target_linear_paths: tuple[str, ...]
    excluded_linear_paths: tuple[str, ...]
    protected_linear_paths: tuple[str, ...]
    modelopt_only: bool = False

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        # Preserve the established hybrid receipt representation and hashes.
        if not self.modelopt_only:
            result.pop("modelopt_only")
        return result


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value):
        return _canonicalize(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    raise TypeError(f"value of type {type(value).__name__} is not canonical JSON")


def canonical_sha256(value: Any) -> str:
    """Return a stable SHA-256 for JSON-compatible scientific metadata."""

    payload = json.dumps(
        _canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_value(value: Any) -> Any:
    """Return the deterministic JSON-safe representation used for hashing."""

    return _canonicalize(value)


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, logical shape, and exact contiguous bytes."""

    if tensor.device.type == "meta":
        raise ValueError("cannot fingerprint a meta tensor")
    cpu = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    metadata = json.dumps(
        {"dtype": str(cpu.dtype), "shape": list(cpu.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _type_name(value: object) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_at_or_below(path: str, root: str) -> bool:
    return path == root or (bool(root) and path.startswith(root + "."))


def _has_selected_ancestor(path: str, selected: Sequence[str]) -> bool:
    return any(root != path and _is_at_or_below(path, root) for root in selected)


def discover_protected_population_roots(model: nn.Module) -> tuple[str, ...]:
    """Find outermost canonical PopulationNetwork roots at every model path.

    ``remove_duplicate=False`` is deliberate: parameter-tied wrappers occur at
    multiple transformer sites and every executable site must be protected even
    when the underlying cell object is shared.
    """

    candidates = [
        name
        for name, module in model.named_modules(remove_duplicate=False)
        if isinstance(module, _PROTECTED_TYPES)
    ]
    candidates.sort(key=lambda item: (item.count("."), item))
    selected: list[str] = []
    for path in candidates:
        if not _has_selected_ancestor(path, selected):
            selected.append(path)
    return tuple(sorted(selected))


def _is_excluded(path: str, excluded_roots: Sequence[str]) -> bool:
    return any(_is_at_or_below(path, root) for root in excluded_roots)


def inventory_dense_only_targets(
    model: nn.Module,
    *,
    excluded_roots: Sequence[str] = ("lm_head",),
    require_population_cells: bool = True,
    reject_dense_descendants: bool = True,
    modelopt_only: bool = False,
) -> DenseOnlyInventory:
    """Freeze the exact native ``nn.Linear`` allowlist outside dendritic cells.

    The strict v1 contract rejects a canonical cell containing any registered
    ``nn.Linear`` descendant.  ModelOpt converts registered dense modules even
    when their quantizers are disabled, so merely appending a disable pattern is
    insufficient to prove that the protected cell's executable structure was
    untouched.

    ``modelopt_only=True`` declares a separate baseline with no canonical cells;
    it rejects an accidental hybrid instead of merely permitting empty roots.
    The default remains the strict hybrid contract.
    """

    protected = discover_protected_population_roots(model)
    if modelopt_only and protected:
        raise ValueError("ModelOpt-only baseline contains PopulationNetwork roots")
    if require_population_cells and not modelopt_only and not protected:
        raise ValueError("no canonical PopulationNetwork replacement roots were found")

    target: list[str] = []
    excluded: list[str] = []
    protected_dense: list[str] = []
    for name, module in model.named_modules(remove_duplicate=False):
        if not name or not isinstance(module, nn.Linear):
            continue
        if _is_excluded(name, protected):
            protected_dense.append(name)
        elif _is_excluded(name, excluded_roots):
            excluded.append(name)
        else:
            target.append(name)
    if reject_dense_descendants and protected_dense:
        raise ValueError(
            "protected PopulationNetwork roots contain ModelOpt-registered "
            "nn.Linear descendants: " + ", ".join(sorted(protected_dense))
        )
    if not target:
        raise ValueError("the native-dense ModelOpt allowlist is empty")
    return DenseOnlyInventory(
        protected_roots=tuple(sorted(protected)),
        target_linear_paths=tuple(sorted(set(target))),
        excluded_linear_paths=tuple(sorted(set(excluded))),
        protected_linear_paths=tuple(sorted(set(protected_dense))),
        modelopt_only=modelopt_only,
    )


def _cell_metadata(module: nn.Module) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for name in (
        "compiled_replacement_plan",
        "population_network_config",
        "selection_manifest",
    ):
        if hasattr(module, name):
            fields[name] = _canonicalize(getattr(module, name))
    return fields


def fingerprint_protected_cells(
    model: nn.Module,
    roots: Sequence[str] | None = None,
    *,
    modelopt_only: bool = False,
) -> dict[str, Any]:
    """Fingerprint all state and topology under canonical protected roots."""

    modules = dict(model.named_modules(remove_duplicate=False))
    selected = (
        discover_protected_population_roots(model) if roots is None else tuple(roots)
    )
    if modelopt_only:
        if selected or discover_protected_population_roots(model):
            raise ValueError("ModelOpt-only baseline contains PopulationNetwork roots")
        result = {"schema": MODELOPT_ONLY_SCHEMA, "roots": [], "cells": []}
        result["sha256"] = canonical_sha256(result)
        return result
    if not selected:
        raise ValueError("protected-cell fingerprint requires at least one root")
    cells: list[dict[str, Any]] = []
    for path in selected:
        if path not in modules:
            raise ValueError(f"protected root {path!r} is absent from the model")
        module = modules[path]
        if not isinstance(module, _PROTECTED_TYPES):
            raise TypeError(f"protected root {path!r} is no longer canonical")

        tree = [
            {"path": name, "type": _type_name(child)}
            for name, child in module.named_modules(remove_duplicate=False)
        ]
        tensors: list[dict[str, Any]] = []
        for kind, named_values in (
            (
                "parameter",
                module.named_parameters(recurse=True, remove_duplicate=False),
            ),
            ("buffer", module.named_buffers(recurse=True, remove_duplicate=False)),
        ):
            for name, tensor in named_values:
                tensors.append(
                    {
                        "kind": kind,
                        "path": name,
                        "dtype": str(tensor.dtype),
                        "shape": list(tensor.shape),
                        "numel": int(tensor.numel()),
                        "sha256": tensor_sha256(tensor),
                    }
                )
        tensors.sort(key=lambda row: (row["kind"], row["path"]))
        metadata = _cell_metadata(module)
        payload = {
            "path": path,
            "root_type": _type_name(module),
            "module_tree": tree,
            "tensors": tensors,
            "metadata": metadata,
        }
        payload["sha256"] = canonical_sha256(payload)
        cells.append(payload)
    result = {
        "schema": PROTECTED_CELL_SCHEMA,
        "roots": list(selected),
        "cells": cells,
    }
    result["sha256"] = canonical_sha256(result)
    return result


def assert_protected_cells_unchanged(
    expected: Mapping[str, Any],
    model: nn.Module,
) -> dict[str, Any]:
    """Fail unless the complete protected-cell fingerprint is unchanged."""

    roots = tuple(str(item) for item in expected.get("roots", ()))
    if discover_protected_population_roots(model) != tuple(sorted(roots)):
        raise RuntimeError(
            "PopulationNetwork root set changed across the ModelOpt operation"
        )
    observed = fingerprint_protected_cells(
        model,
        roots,
        modelopt_only=expected.get("schema") == MODELOPT_ONLY_SCHEMA,
    )
    if canonical_sha256(expected) != canonical_sha256(observed):
        raise RuntimeError(
            "PopulationNetwork value/topology/module fingerprint changed across "
            "the ModelOpt operation"
        )
    return observed


@contextmanager
def protected_modelopt_traversal_shield(
    model: nn.Module,
    expected_protected_fingerprint: Mapping[str, Any],
):
    """Shield callable ``weight`` APIs from ModelOpt 0.46's pack traversal.

    ModelOpt 0.46's physical packer visits every submodule and assumes that any
    object exposing ``weight`` exposes a tensor with ``is_meta``.  Canonical
    ``IndexedSparseLinear`` instead has a callable ``weight()`` compatibility
    API.  During only the third-party traversal, this context shadows such
    callable class attributes with ``None`` so ModelOpt skips them.  It refuses
    pre-existing instance shadows or arbitrary non-tensor values, removes every
    temporary shadow even on failure, and closes with the full protected-cell
    hash bracket.
    """

    roots = tuple(str(item) for item in expected_protected_fingerprint["roots"])
    assert_protected_cells_unchanged(expected_protected_fingerprint, model)
    shadows: list[nn.Module] = []
    try:
        for path, module in model.named_modules(remove_duplicate=False):
            if not _is_excluded(path, roots) or not hasattr(module, "weight"):
                continue
            value = module.weight
            if value is None or isinstance(value, torch.Tensor):
                continue
            if "weight" in module.__dict__:
                raise RuntimeError(
                    f"protected module {path!r} already shadows a non-tensor weight"
                )
            if not callable(value):
                raise RuntimeError(
                    f"protected module {path!r} exposes unsupported non-tensor "
                    f"weight type {type(value).__name__}"
                )
            object.__setattr__(module, "weight", None)
            shadows.append(module)
        yield
    finally:
        for module in reversed(shadows):
            object.__delattr__(module, "weight")
        assert_protected_cells_unchanged(expected_protected_fingerprint, model)


def build_int4_blockwise_weight_only_config(
    target_linear_paths: Sequence[str],
    *,
    block_size: int = INT4_BLOCK_SIZE,
) -> dict[str, Any]:
    """Build an exact-path, deny-all-first ModelOpt INT4 configuration."""

    targets = tuple(sorted({str(path) for path in target_linear_paths}))
    if not targets or any(not path for path in targets):
        raise ValueError("target_linear_paths must contain non-empty module paths")
    if int(block_size) < 1:
        raise ValueError("block_size must be positive")
    quant_cfg: list[dict[str, Any]] = [{"quantizer_name": "*", "enable": False}]
    for path in targets:
        quant_cfg.extend(
            (
                {
                    "quantizer_name": f"{path}.weight_quantizer",
                    "cfg": {
                        "num_bits": 4,
                        "block_sizes": {-1: int(block_size), "type": "static"},
                    },
                },
                {
                    "quantizer_name": f"{path}.input_quantizer",
                    "enable": False,
                },
            )
        )
    return {"quant_cfg": quant_cfg, "algorithm": "max"}


def _storage_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _concrete_device(device: torch.device | str) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def audit_packed_modelopt_artifact(
    model: nn.Module,
    *,
    inventory: DenseOnlyInventory,
    expected_protected_fingerprint: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify real packed weights exist at exactly the frozen allowlist."""

    try:
        from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
        from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper
    except ImportError as exc:  # pragma: no cover - exercised in optional env
        raise RuntimeError(
            "pinned nvidia-modelopt is required for packed audit"
        ) from exc

    assert_protected_cells_unchanged(expected_protected_fingerprint, model)
    if tuple(expected_protected_fingerprint["roots"]) != inventory.protected_roots:
        raise RuntimeError("inventory and protected-cell fingerprint roots disagree")
    if inventory.modelopt_only != (
        expected_protected_fingerprint.get("schema") == MODELOPT_ONLY_SCHEMA
    ):
        raise RuntimeError("inventory and protected-cell fingerprint modes disagree")
    modules = dict(model.named_modules(remove_duplicate=False))
    expected_targets = set(inventory.target_linear_paths)
    missing = sorted(path for path in expected_targets if path not in modules)
    if missing:
        raise RuntimeError("packed target modules are absent: " + ", ".join(missing))

    packed_rows: list[dict[str, Any]] = []
    packed_paths: set[str] = set()
    real_quant_paths: set[str] = set()
    for path, module in modules.items():
        if isinstance(module, RealQuantLinear):
            real_quant_paths.add(path)
        weight = getattr(module, "weight", None)
        if isinstance(weight, QTensorWrapper):
            packed_paths.add(path)
            metadata = dict(weight.metadata)
            auxiliary_tensors: list[dict[str, Any]] = []
            for kind, named_values in (
                ("parameter", module.named_parameters(recurse=True)),
                ("buffer", module.named_buffers(recurse=True)),
            ):
                for name, tensor in named_values:
                    if kind == "parameter" and name == "weight":
                        continue
                    auxiliary_tensors.append(
                        {
                            "kind": kind,
                            "path": name,
                            "dtype": str(tensor.dtype),
                            "shape": list(tensor.shape),
                            "physical_bytes": _storage_bytes(tensor),
                            "sha256": tensor_sha256(tensor),
                        }
                    )
            auxiliary_tensors.sort(key=lambda row: (row["kind"], row["path"]))
            weight_bytes = _storage_bytes(weight)
            auxiliary_bytes = sum(
                int(row["physical_bytes"]) for row in auxiliary_tensors
            )
            packed_rows.append(
                {
                    "path": path,
                    "physical_dtype": str(weight.dtype),
                    "physical_shape": list(weight.shape),
                    "physical_bytes": weight_bytes,
                    "physical_sha256": tensor_sha256(weight),
                    "logical_dtype": str(metadata.get("dtype")),
                    "logical_shape": list(metadata.get("shape", ())),
                    "qtensor_class": _canonicalize(metadata.get("qtensor_class")),
                    "auxiliary_tensors": auxiliary_tensors,
                    "quantizer_auxiliary_bytes": auxiliary_bytes,
                    "physical_target_state_bytes": weight_bytes + auxiliary_bytes,
                }
            )
    if packed_paths != expected_targets:
        raise RuntimeError(
            "packed QTensor paths differ from the frozen native-dense allowlist: "
            f"missing={sorted(expected_targets - packed_paths)}, "
            f"unexpected={sorted(packed_paths - expected_targets)}"
        )
    if not expected_targets.issubset(real_quant_paths):
        raise RuntimeError("a packed target is not represented by RealQuantLinear")

    packed_rows.sort(key=lambda row: row["path"])
    report = {
        "schema": PACKED_AUDIT_SCHEMA,
        "artifact_kind": "physical_modelopt_qtensor_pytorch",
        "is_fake_qdq_only": False,
        "inventory": inventory.as_dict(),
        "packed_targets": packed_rows,
        "packed_weight_bytes": sum(row["physical_bytes"] for row in packed_rows),
        "quantizer_auxiliary_bytes": sum(
            row["quantizer_auxiliary_bytes"] for row in packed_rows
        ),
        "packed_target_state_bytes": sum(
            row["physical_target_state_bytes"] for row in packed_rows
        ),
        "real_quant_module_paths": sorted(real_quant_paths),
        "protected_cell_sha256": str(expected_protected_fingerprint["sha256"]),
        "claim_boundary": (
            MODELOPT_ONLY_CLAIM_BOUNDARY
            if inventory.modelopt_only
            else PHYSICAL_ARTIFACT_CLAIM_BOUNDARY
        ),
    }
    report["sha256"] = canonical_sha256(report)
    return report


def audit_packed_device_coherence(
    model: nn.Module,
    *,
    target_linear_paths: Sequence[str],
    expected_device: torch.device | str,
) -> dict[str, Any]:
    """Require packed values and every quantizer auxiliary on one device."""

    try:
        from modelopt.torch.quantization.qtensor.base_qtensor import QTensorWrapper
    except ImportError as exc:  # pragma: no cover - exercised in optional env
        raise RuntimeError(
            "pinned nvidia-modelopt is required for device audit"
        ) from exc
    expected = _concrete_device(expected_device)
    rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for path in sorted(set(target_linear_paths)):
        module = model.get_submodule(path)
        weight = getattr(module, "weight", None)
        if not isinstance(weight, QTensorWrapper):
            raise RuntimeError(f"target {path!r} is not physically packed")
        tensors = [("weight", weight)]
        tensors.extend(
            (f"parameter:{name}", tensor)
            for name, tensor in module.named_parameters(recurse=True)
            if name != "weight"
        )
        tensors.extend(
            (f"buffer:{name}", tensor)
            for name, tensor in module.named_buffers(recurse=True)
        )
        devices = {name: str(tensor.device) for name, tensor in tensors}
        wrong = sorted(name for name, tensor in tensors if tensor.device != expected)
        if wrong:
            mismatches.append(f"{path}: {', '.join(wrong)}")
        rows.append({"path": path, "tensor_devices": devices})
    if mismatches:
        raise RuntimeError(
            "packed ModelOpt weights/auxiliaries are not device-coherent: "
            + "; ".join(mismatches)
        )
    report = {
        "schema": DEVICE_AUDIT_SCHEMA,
        "expected_device": str(expected),
        "target_count": len(rows),
        "targets": rows,
    }
    report["sha256"] = canonical_sha256(report)
    return report


def restore_packed_modelopt_checkpoint(
    model: nn.Module,
    checkpoint: Path | str,
    *,
    device: torch.device | str,
    expected_protected_fingerprint: Mapping[str, Any],
    target_linear_paths: Sequence[str],
) -> tuple[nn.Module, dict[str, Any]]:
    """Restore a physical checkpoint with explicit target-device materialization.

    ModelOpt 0.46 otherwise defaults ``mto.restore`` to CPU even when the
    supplied custom model is already on CUDA, which separates INT4 scales from
    packed weights.  The explicit ``map_location`` and post-restore ``to`` are
    followed by protected-cell and packed-auxiliary audits before execution.
    """

    try:
        import modelopt.torch.opt as mto
    except ImportError as exc:  # pragma: no cover - exercised in optional env
        raise RuntimeError("pinned nvidia-modelopt is required for restore") from exc
    target_device = _concrete_device(device)
    with protected_modelopt_traversal_shield(model, expected_protected_fingerprint):
        restored = mto.restore(model, checkpoint, map_location=target_device)
    restored.to(target_device)
    assert_protected_cells_unchanged(expected_protected_fingerprint, restored)
    device_audit = audit_packed_device_coherence(
        restored,
        target_linear_paths=target_linear_paths,
        expected_device=target_device,
    )
    return restored, device_audit


__all__ = [
    "DENSE_ONLY_SCHEMA",
    "DEVICE_AUDIT_SCHEMA",
    "FAKE_QDQ_CLAIM_BOUNDARY",
    "INT4_BLOCK_SIZE",
    "MODELOPT_ONLY_CLAIM_BOUNDARY",
    "MODELOPT_ONLY_SCHEMA",
    "PACKED_AUDIT_SCHEMA",
    "PHYSICAL_ARTIFACT_CLAIM_BOUNDARY",
    "PROTECTED_CELL_SCHEMA",
    "DenseOnlyInventory",
    "assert_protected_cells_unchanged",
    "audit_packed_device_coherence",
    "audit_packed_modelopt_artifact",
    "build_int4_blockwise_weight_only_config",
    "canonical_json_value",
    "canonical_sha256",
    "discover_protected_population_roots",
    "fingerprint_protected_cells",
    "inventory_dense_only_targets",
    "protected_modelopt_traversal_shield",
    "restore_packed_modelopt_checkpoint",
    "tensor_sha256",
]

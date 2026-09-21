"""Optional NVIDIA Model Optimizer fake-Q/DQ for indexed synapses.

This integration is deliberately narrow.  It supports calibration-aware INT8
fake quantization of :class:`IndexedSparseLinear` inputs and *effective*
synaptic weights with NVIDIA Model Optimizer 0.46.0.  It does not pack weights,
change indexed topology, export a deployment graph, or provide a runtime,
storage, energy, or quality claim.

ModelOpt is imported lazily so the base dendritic-modeling package remains
usable without the optional dependency.  Registration is scoped through
``indexed_sparse_modelopt_registration``; callers restoring a ModelOpt
checkpoint must enter the same context before invoking ModelOpt restore APIs.
"""

from __future__ import annotations

import importlib
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from dendritic_modeling.integrations.modelopt import (
    PINNED_MODELOPT_VERSION,
    ModelOptEnvironment,
    require_pinned_modelopt,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

INDEXED_SPARSE_FAKE_QDQ_SCHEMA = "dendritic_modelopt_indexed_sparse_fake_qdq/v1"
INDEXED_SPARSE_FAKE_QDQ_CLAIM_BOUNDARY = (
    "ModelOpt fake-Q/DQ compatibility only: effective values remain represented by "
    "the original floating-point pre_w parameters plus quantizer state and indexed "
    "topology. This is not packed compression, export, acceleration, energy, or "
    "quality evidence."
)


@dataclass(frozen=True)
class IndexedSparseModelOptRegistration:
    """Identity of one active custom-module registration."""

    modelopt_version: str
    modelopt_module_path: str
    registry_key: str
    quantized_class: type[nn.Module]
    schema: str = INDEXED_SPARSE_FAKE_QDQ_SCHEMA
    claim_boundary: str = INDEXED_SPARSE_FAKE_QDQ_CLAIM_BOUNDARY


@dataclass(frozen=True)
class _ModelOptAPI:
    package: ModuleType
    quantization: ModuleType
    tensor_quantizer_class: type[nn.Module]
    quant_module_class: type[nn.Module]
    quant_module_registry: Any
    input_quant_descriptor: Any
    weight_quant_descriptor: Any
    environment: ModelOptEnvironment


_REGISTRATION_LOCK = threading.RLock()
_ACTIVE_REGISTRATION: IndexedSparseModelOptRegistration | None = None
_ACTIVE_API: _ModelOptAPI | None = None


def _resolved_module_path(module: ModuleType) -> Path:
    raw = getattr(module, "__file__", None)
    if raw is None:
        raise RuntimeError("loaded ModelOpt package has no __file__ identity")
    return Path(raw).resolve(strict=True)


def _load_pinned_modelopt_api() -> _ModelOptAPI:
    """Import and validate the exact optional API used by this adapter."""

    environment = require_pinned_modelopt()
    package = importlib.import_module("modelopt")
    loaded_version = str(getattr(package, "__version__", ""))
    if loaded_version != PINNED_MODELOPT_VERSION:
        raise RuntimeError(
            "loaded ModelOpt version drift: "
            f"{loaded_version!r}, required {PINNED_MODELOPT_VERSION!r}"
        )
    if environment.module_path is None:
        raise RuntimeError("inspected ModelOpt package has no module path")
    inspected_path = Path(environment.module_path).resolve(strict=True)
    loaded_path = _resolved_module_path(package)
    if loaded_path != inspected_path:
        raise RuntimeError(
            "loaded ModelOpt code does not match the inspected package: "
            f"loaded={loaded_path}, inspected={inspected_path}"
        )

    quantization = importlib.import_module("modelopt.torch.quantization")
    quant_nn = importlib.import_module("modelopt.torch.quantization.nn")
    quant_module = importlib.import_module(
        "modelopt.torch.quantization.nn.modules.quant_module"
    )
    tensor_quant = importlib.import_module("modelopt.torch.quantization.tensor_quant")
    required = {
        "register": getattr(quantization, "register", None),
        "unregister": getattr(quantization, "unregister", None),
        "quantize": getattr(quantization, "quantize", None),
        "TensorQuantizer": getattr(quant_nn, "TensorQuantizer", None),
        "QuantModule": getattr(quant_module, "QuantModule", None),
        "QuantModuleRegistry": getattr(quant_module, "QuantModuleRegistry", None),
        "input descriptor": getattr(tensor_quant, "QUANT_DESC_8BIT_PER_TENSOR", None),
        "weight descriptor": getattr(
            tensor_quant,
            "QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW",
            None,
        ),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(f"ModelOpt 0.46.0 API mismatch; missing {missing}")
    if not all(
        callable(required[name]) for name in ("register", "unregister", "quantize")
    ):
        raise RuntimeError(
            "ModelOpt 0.46.0 registration or quantization API is not callable"
        )

    return _ModelOptAPI(
        package=package,
        quantization=quantization,
        tensor_quantizer_class=required["TensorQuantizer"],
        quant_module_class=required["QuantModule"],
        quant_module_registry=required["QuantModuleRegistry"],
        input_quant_descriptor=required["input descriptor"],
        weight_quant_descriptor=required["weight descriptor"],
        environment=environment,
    )


def _build_quantized_indexed_sparse_class(api: _ModelOptAPI) -> type[nn.Module]:
    tensor_quantizer_class = api.tensor_quantizer_class
    quant_module_class = api.quant_module_class
    input_descriptor = api.input_quant_descriptor
    weight_descriptor = api.weight_quant_descriptor

    class _QuantIndexedSparseLinear(quant_module_class):
        """Dynamic ModelOpt mixin preserving indexed-synapse equations."""

        _dendritic_modelopt_adapter_schema = INDEXED_SPARSE_FAKE_QDQ_SCHEMA

        def _setup(self) -> None:
            self._register_temp_attribute(
                "indexed_input_quantizer",
                tensor_quantizer_class(input_descriptor),
            )
            self._register_temp_attribute(
                "effective_weight_quantizer",
                tensor_quantizer_class(weight_descriptor),
            )
            # A fold made before conversion would bypass effective-weight Q/DQ.
            self.unfreeze_inference_fold()
            self._recurrent_cached_weight = None

        def _normalized_sparse_weight(self) -> torch.Tensor:
            effective_weight = super()._normalized_sparse_weight()
            return self.effective_weight_quantizer(effective_weight)

        def iter_weights_for_calibration(self):
            # Bypass this override to calibrate on the transformed and normalized
            # conductance, without recursively applying fake quantization.
            yield (
                super()._normalized_sparse_weight(),
                self.effective_weight_quantizer,
            )

        def _can_use_frozen_chunked_transform(self, flat_x: torch.Tensor) -> bool:
            # The original chunked path transforms pre_w directly and therefore
            # cannot observe the effective-weight quantizer.
            if self.effective_weight_quantizer.is_enabled:
                return False
            return super()._can_use_frozen_chunked_transform(flat_x)

        def freeze_for_inference(self) -> None:
            if self.effective_weight_quantizer.is_enabled:
                raise RuntimeError(
                    "freeze_for_inference is unsupported while ModelOpt indexed-sparse "
                    "fake weight quantization is enabled; it would hide explicit Q/DQ "
                    "behind a non-persistent folded buffer"
                )
            super().freeze_for_inference()

        def fold_weight(self, keep_attrs: bool = False) -> None:
            del keep_attrs
            raise RuntimeError(
                "ModelOpt weight folding/real compression is unsupported for "
                "IndexedSparseLinear because quantization is applied after the "
                "pre_w transform; use explicit fake-Q/DQ only"
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return super().forward(self.indexed_input_quantizer(x))

    _QuantIndexedSparseLinear.__module__ = __name__
    _QuantIndexedSparseLinear.__qualname__ = "_QuantIndexedSparseLinear"
    return _QuantIndexedSparseLinear


def build_indexed_sparse_int8_fake_qdq_config() -> dict[str, Any]:
    """Return the only audited ModelOpt recipe for indexed synapses.

    The ordered deny-all entry is essential: ModelOpt may dynamically convert
    other registered module types while walking a containing model, but their
    quantizers must remain disabled in this narrowly scoped experiment.
    """

    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*effective_weight_quantizer",
                "cfg": {"num_bits": 8, "axis": 0},
            },
            {
                "quantizer_name": "*indexed_input_quantizer",
                "cfg": {"num_bits": 8, "axis": None},
            },
        ],
        "algorithm": "max",
    }


def register_indexed_sparse_modelopt_adapter() -> IndexedSparseModelOptRegistration:
    """Register the custom dynamic class, rejecting ambiguous registry state."""

    global _ACTIVE_API, _ACTIVE_REGISTRATION
    with _REGISTRATION_LOCK:
        if _ACTIVE_REGISTRATION is not None or _ACTIVE_API is not None:
            raise RuntimeError(
                "IndexedSparseLinear ModelOpt adapter is already registered"
            )
        api = _load_pinned_modelopt_api()
        registry = api.quant_module_registry
        if IndexedSparseLinear in registry:
            raise RuntimeError(
                "IndexedSparseLinear already has a ModelOpt registration; refusing "
                "to replace or reuse an unverified adapter"
            )
        quantized_class = _build_quantized_indexed_sparse_class(api)
        api.quantization.register(
            original_cls=IndexedSparseLinear,
            quantized_cls=quantized_class,
        )
        try:
            if IndexedSparseLinear not in registry:
                raise RuntimeError(
                    "ModelOpt did not retain the IndexedSparseLinear registration"
                )
            resolved_class = registry[IndexedSparseLinear]
            if not issubclass(resolved_class, quantized_class):
                raise RuntimeError(
                    "ModelOpt resolved a different IndexedSparseLinear adapter"
                )
            registry_key = str(registry.get_key(IndexedSparseLinear))
            if registry_key != "IndexedSparseLinear":
                raise RuntimeError(
                    f"unexpected IndexedSparseLinear ModelOpt registry key: {registry_key!r}"
                )
        except BaseException:
            api.quantization.unregister(IndexedSparseLinear)
            raise

        registration = IndexedSparseModelOptRegistration(
            modelopt_version=str(api.package.__version__),
            modelopt_module_path=str(_resolved_module_path(api.package)),
            registry_key=registry_key,
            quantized_class=quantized_class,
        )
        _ACTIVE_API = api
        _ACTIVE_REGISTRATION = registration
        return registration


def unregister_indexed_sparse_modelopt_adapter() -> None:
    """Remove exactly the registration installed by this module."""

    global _ACTIVE_API, _ACTIVE_REGISTRATION
    with _REGISTRATION_LOCK:
        if _ACTIVE_REGISTRATION is None or _ACTIVE_API is None:
            raise RuntimeError("IndexedSparseLinear ModelOpt adapter is not registered")
        api = _ACTIVE_API
        registration = _ACTIVE_REGISTRATION
        registry = api.quant_module_registry
        if IndexedSparseLinear not in registry:
            raise RuntimeError(
                "active IndexedSparseLinear ModelOpt registration disappeared"
            )
        resolved_class = registry[IndexedSparseLinear]
        if not issubclass(resolved_class, registration.quantized_class):
            raise RuntimeError(
                "IndexedSparseLinear ModelOpt registry ownership changed"
            )
        api.quantization.unregister(IndexedSparseLinear)
        if IndexedSparseLinear in registry:
            raise RuntimeError("ModelOpt failed to unregister IndexedSparseLinear")
        _ACTIVE_REGISTRATION = None
        _ACTIVE_API = None


@contextmanager
def indexed_sparse_modelopt_registration() -> (
    Iterator[IndexedSparseModelOptRegistration]
):
    """Temporarily register the audited ModelOpt custom module."""

    registration = register_indexed_sparse_modelopt_adapter()
    try:
        yield registration
    finally:
        unregister_indexed_sparse_modelopt_adapter()


def _indexed_targets(model: nn.Module) -> dict[str, IndexedSparseLinear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, IndexedSparseLinear)
    }


def quantize_indexed_sparse_fake_qdq(
    model: nn.Module,
    *,
    forward_loop: Callable[[nn.Module], Any],
) -> nn.Module:
    """Apply the audited INT8/max fake-Q/DQ recipe in place.

    This helper intentionally offers no format or algorithm switches.  New
    formats, output quantization, or calibration algorithms require independent
    compatibility and scientific validation.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model)}")
    if not callable(forward_loop):
        raise TypeError("forward_loop must be callable")
    targets_before = _indexed_targets(model)
    if not targets_before:
        raise ValueError("model contains no IndexedSparseLinear target")
    already_converted = [
        name
        for name, module in targets_before.items()
        if getattr(module, "_dendritic_modelopt_adapter_schema", None)
        == INDEXED_SPARSE_FAKE_QDQ_SCHEMA
        or hasattr(module, "effective_weight_quantizer")
        or hasattr(module, "indexed_input_quantizer")
        or hasattr(module, "input_quantizer")
    ]
    if already_converted:
        raise RuntimeError(
            "refusing to re-quantize already converted IndexedSparseLinear targets: "
            f"{already_converted}"
        )
    reserved_quantizer_suffixes = (
        "effective_weight_quantizer",
        "indexed_input_quantizer",
    )
    collisions = [
        name
        for name, _module in model.named_modules()
        if any(name.endswith(suffix) for suffix in reserved_quantizer_suffixes)
    ]
    if collisions:
        raise RuntimeError(
            "reserved indexed-sparse ModelOpt quantizer names already exist before "
            f"conversion: {collisions}"
        )
    topology_before = {
        name: module.connection_indices.detach().clone()
        for name, module in targets_before.items()
    }
    raw_parameter_ids = {
        name: id(module.pre_w) for name, module in targets_before.items()
    }

    with indexed_sparse_modelopt_registration() as registration:
        api = _ACTIVE_API
        if api is None:
            raise RuntimeError("ModelOpt registration context lost its API state")
        api.quantization.quantize(
            model,
            build_indexed_sparse_int8_fake_qdq_config(),
            forward_loop=forward_loop,
        )
        targets_after = _indexed_targets(model)
        if targets_after.keys() != targets_before.keys():
            raise RuntimeError(
                "IndexedSparseLinear module paths changed during quantization"
            )
        enabled_target_quantizer_ids: set[int] = set()
        for name, module in targets_after.items():
            if not isinstance(module, registration.quantized_class):
                raise RuntimeError(
                    f"IndexedSparseLinear target was not converted: {name}"
                )
            if id(module.pre_w) != raw_parameter_ids[name]:
                raise RuntimeError(f"raw pre_w parameter identity changed: {name}")
            if not torch.equal(module.connection_indices, topology_before[name]):
                raise RuntimeError(
                    f"indexed topology changed during quantization: {name}"
                )
            if module._inference_folded_weight is not None:
                raise RuntimeError(f"stale inference fold survived conversion: {name}")
            if not module.indexed_input_quantizer.is_enabled:
                raise RuntimeError(
                    f"input quantizer is disabled after calibration: {name}"
                )
            if not module.effective_weight_quantizer.is_enabled:
                raise RuntimeError(
                    f"effective-weight quantizer is disabled after calibration: {name}"
                )
            enabled_target_quantizer_ids.update(
                (
                    id(module.indexed_input_quantizer),
                    id(module.effective_weight_quantizer),
                )
            )

        for name, module in model.named_modules():
            if (
                isinstance(module, api.tensor_quantizer_class)
                and module.is_enabled
                and id(module) not in enabled_target_quantizer_ids
            ):
                raise RuntimeError(
                    "a non-target ModelOpt quantizer is enabled despite the deny-all recipe: "
                    f"{name}"
                )
    return model


__all__ = [
    "INDEXED_SPARSE_FAKE_QDQ_CLAIM_BOUNDARY",
    "INDEXED_SPARSE_FAKE_QDQ_SCHEMA",
    "IndexedSparseModelOptRegistration",
    "build_indexed_sparse_int8_fake_qdq_config",
    "indexed_sparse_modelopt_registration",
    "quantize_indexed_sparse_fake_qdq",
    "register_indexed_sparse_modelopt_adapter",
    "unregister_indexed_sparse_modelopt_adapter",
]

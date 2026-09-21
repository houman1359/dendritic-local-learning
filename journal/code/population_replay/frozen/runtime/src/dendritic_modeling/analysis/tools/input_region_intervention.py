"""Paired input-region interventions for frozen image classifiers.

The analysis in this module separates two different region definitions:

* ``central_square`` and ``surround`` are fixed spatial regions shared by all
  examples;
* ``foreground`` and ``background`` are sample-dependent intensity regions
  computed from each unmodified input using an explicit threshold.

An intervention either removes the named region or retains only that region by
replacing the complementary features.  Replacement uses numerical
feature-space zero or a featurewise mean estimated once from a named reference
split.  The latter is held fixed while the evaluation split is processed.

The analyzer reports paired accuracy and negative-log-likelihood effects.  It
also summarizes changes in E/I soma-population outputs at each network layer
and changes in excitatory/inhibitory branch currents at each soma-relative
dendritic depth when those signals exist.  Later-layer measurements are
therefore response propagation from an input intervention, not literal pixel
receptive fields.  No per-sample activation or sample-by-synapse array is
returned.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

from dendritic_modeling.analysis.utils.dendritic_depth import (
    SOMA_RELATIVE_DEPTH_REFERENCE,
    soma_relative_dendritic_depth,
)
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    iter_analysis_batches,
)
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.networks import DendriticBranchLayer, PopulationLayer
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.utils.general import save_dict

RegionKind = Literal[
    "central_square",
    "surround",
    "foreground",
    "background",
]
InterventionOperation = Literal["remove_only", "retain_only"]
ReplacementMethod = Literal["zero", "reference_mean"]

_REGION_KINDS = {"central_square", "surround", "foreground", "background"}
_OPERATIONS = {"remove_only", "retain_only"}
_REPLACEMENTS = {"zero", "reference_mean"}
_POPULATION_POLARITIES = {"excitatory", "inhibitory"}


@dataclass(frozen=True)
class ImageRegionSpec:
    """Definition of one fixed-spatial or sample-dependent input region."""

    name: str
    kind: RegionKind
    center_fraction: float = 0.5
    foreground_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("image-region names must be non-empty")
        if self.kind not in _REGION_KINDS:
            raise ValueError(f"unknown image-region kind {self.kind!r}")
        if self.kind in {"central_square", "surround"} and not (
            0.0 < float(self.center_fraction) <= 1.0
        ):
            raise ValueError("center_fraction must lie in (0, 1]")
        if not math.isfinite(float(self.foreground_threshold)):
            raise ValueError("foreground_threshold must be finite")

    @property
    def sample_dependent(self) -> bool:
        return self.kind in {"foreground", "background"}


@dataclass(frozen=True)
class InputRegionInterventionSettings:
    """Runtime-independent settings for paired input-region interventions."""

    image_shape: tuple[int, ...]
    regions: tuple[ImageRegionSpec, ...]
    operations: tuple[InterventionOperation, ...] = (
        "remove_only",
        "retain_only",
    )
    replacement_methods: tuple[ReplacementMethod, ...] = (
        "zero",
        "reference_mean",
    )
    reference_split: str = "train"
    evaluation_split: str = "test"
    max_reference_samples: int | None = None
    max_evaluation_samples: int | None = None
    target_population_polarities: tuple[str, ...] = ("excitatory",)
    identity_soma_class_mapping: tuple[int, ...] | None = None
    identity_soma_population_name: str | None = None
    identity_soma_network_layer_index: int | None = None

    def __post_init__(self) -> None:
        _canonical_image_shape(self.image_shape)
        if not self.regions:
            raise ValueError("at least one input region is required")
        names = [str(region.name) for region in self.regions]
        if len(names) != len(set(names)):
            raise ValueError(f"image-region names must be unique, got {names}")
        if not self.operations or any(op not in _OPERATIONS for op in self.operations):
            raise ValueError(f"operations must be selected from {sorted(_OPERATIONS)}")
        if not self.replacement_methods or any(
            method not in _REPLACEMENTS for method in self.replacement_methods
        ):
            raise ValueError(
                f"replacement_methods must be selected from {sorted(_REPLACEMENTS)}"
            )
        if not str(self.evaluation_split).strip():
            raise ValueError("evaluation_split must be named")
        if "reference_mean" in self.replacement_methods:
            if not str(self.reference_split).strip():
                raise ValueError("reference_split must be named")
            if self.reference_split == self.evaluation_split:
                raise ValueError(
                    "reference_mean replacement requires distinct named reference "
                    "and evaluation splits"
                )
        for name, value in (
            ("max_reference_samples", self.max_reference_samples),
            ("max_evaluation_samples", self.max_evaluation_samples),
        ):
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive when provided")
        polarities = tuple(str(value) for value in self.target_population_polarities)
        if not polarities or len(set(polarities)) != len(polarities):
            raise ValueError(
                "target_population_polarities must be non-empty and unique"
            )
        unknown_polarities = sorted(set(polarities) - _POPULATION_POLARITIES)
        if unknown_polarities:
            raise ValueError(
                "target_population_polarities must be selected from "
                f"{sorted(_POPULATION_POLARITIES)}, got {unknown_polarities}"
            )
        self._validate_identity_mapping()
        self._validate_unique_region_operation_masks()

    def _validate_identity_mapping(self) -> None:
        mapping = self.identity_soma_class_mapping
        selectors = (
            self.identity_soma_population_name,
            self.identity_soma_network_layer_index,
        )
        if mapping is None:
            if any(value is not None for value in selectors):
                raise ValueError(
                    "identity soma selectors require identity_soma_class_mapping"
                )
            return
        classes = tuple(int(value) for value in mapping)
        if not classes or len(classes) != len(set(classes)) or min(classes) < 0:
            raise ValueError(
                "identity_soma_class_mapping must contain unique non-negative classes"
            )
        if not str(self.identity_soma_population_name or "").strip():
            raise ValueError(
                "identity_soma_population_name is required with an identity mapping"
            )
        if (
            self.identity_soma_network_layer_index is None
            or int(self.identity_soma_network_layer_index) < 0
        ):
            raise ValueError(
                "a non-negative identity_soma_network_layer_index is required with "
                "an identity mapping"
            )

    def _validate_unique_region_operation_masks(self) -> None:
        """Reject pairs that replace the same features by construction."""

        seen: dict[tuple[str, float, bool], tuple[str, str]] = {}
        for region in self.regions:
            if region.kind in {"central_square", "surround"}:
                family = "central_square"
                parameter = float(region.center_fraction)
                named_region_is_base = region.kind == "central_square"
            else:
                family = "foreground"
                parameter = float(region.foreground_threshold)
                named_region_is_base = region.kind == "foreground"
            for operation in self.operations:
                replacement_is_base = (
                    named_region_is_base
                    if operation == "remove_only"
                    else not named_region_is_base
                )
                signature = (family, parameter, replacement_is_base)
                previous = seen.get(signature)
                if previous is not None:
                    raise ValueError(
                        "statically duplicate input-region replacement masks: "
                        f"{previous[0]}/{previous[1]} and "
                        f"{region.name}/{operation}"
                    )
                seen[signature] = (region.name, operation)


def _canonical_image_shape(image_shape: tuple[int, ...]) -> tuple[int, int, int]:
    shape = tuple(int(value) for value in image_shape)
    if len(shape) == 2:
        shape = (1, *shape)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError(
            "image_shape must be (height, width) or (channels, height, width)"
        )
    return shape


def _as_image_batch(
    inputs: torch.Tensor,
    image_shape: tuple[int, ...],
) -> torch.Tensor:
    if not isinstance(inputs, torch.Tensor) or inputs.ndim < 2:
        raise TypeError("image intervention requires a batched input tensor")
    channels, height, width = _canonical_image_shape(image_shape)
    expected = channels * height * width
    if inputs.ndim == 2 and int(inputs.shape[1]) == expected:
        return inputs.reshape(inputs.shape[0], channels, height, width)
    if tuple(inputs.shape[1:]) == (channels, height, width):
        return inputs
    if channels == 1 and tuple(inputs.shape[1:]) == (height, width):
        return inputs.unsqueeze(1)
    raise ValueError(
        f"input shape {tuple(inputs.shape)} is incompatible with image_shape "
        f"{(channels, height, width)}"
    )


def build_input_region_mask(
    inputs: torch.Tensor,
    *,
    image_shape: tuple[int, ...],
    region: ImageRegionSpec,
) -> torch.Tensor:
    """Return a boolean mask with the same shape as ``inputs``.

    Foreground/background membership is computed from the mean input value over
    channels at each spatial location, then broadcast back over channels.  The
    mask always describes the named region in the original, unmodified input;
    the intervention operation separately decides which features are replaced.
    """

    image = _as_image_batch(inputs, image_shape)
    _, channels, height, width = image.shape
    if region.kind in {"central_square", "surround"}:
        center_height = max(1, min(height, round(height * region.center_fraction)))
        center_width = max(1, min(width, round(width * region.center_fraction)))
        row_start = (height - center_height) // 2
        col_start = (width - center_width) // 2
        spatial = torch.zeros(
            (1, 1, height, width), dtype=torch.bool, device=inputs.device
        )
        spatial[
            :,
            :,
            row_start : row_start + center_height,
            col_start : col_start + center_width,
        ] = True
        if region.kind == "surround":
            spatial = ~spatial
        mask = spatial.expand(image.shape[0], channels, height, width)
    else:
        intensity = image.to(dtype=torch.float32).mean(dim=1, keepdim=True)
        spatial = intensity > float(region.foreground_threshold)
        if region.kind == "background":
            spatial = ~spatial
        mask = spatial.expand(image.shape[0], channels, height, width)
    return mask.reshape_as(inputs)


def apply_input_region_intervention(
    inputs: torch.Tensor,
    *,
    region_mask: torch.Tensor,
    operation: InterventionOperation,
    replacement: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the requested region and return ``(altered, replacement_mask)``."""

    if operation not in _OPERATIONS:
        raise ValueError(f"unknown input-region operation {operation!r}")
    if region_mask.dtype is not torch.bool or region_mask.shape != inputs.shape:
        raise ValueError("region_mask must be boolean and match the input shape")
    replacement_mask = region_mask if operation == "remove_only" else ~region_mask
    try:
        replacement_value = replacement.to(device=inputs.device, dtype=inputs.dtype)
        replacement_value = torch.broadcast_to(replacement_value, inputs.shape)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("replacement is not broadcastable to the input") from exc
    altered = torch.where(replacement_mask, replacement_value, inputs)
    return altered, replacement_mask


@dataclass
class _ReferenceMean:
    values: torch.Tensor
    sample_count: int


def estimate_featurewise_reference_mean(
    dataset: torch.utils.data.Dataset,
    *,
    image_shape: tuple[int, ...],
    runtime: EvaluationRuntimeConfig | None = None,
    max_samples: int | None = None,
) -> _ReferenceMean:
    """Estimate a deterministic featurewise mean by streaming a reference split."""

    running_sum: torch.Tensor | None = None
    sample_count = 0
    for batch in iter_analysis_batches(
        dataset,
        runtime,
        explicit_max_samples=max_samples,
        device="cpu",
    ):
        if not batch:
            raise TypeError("reference data must provide input tensors")
        inputs = batch[0]
        image = _as_image_batch(inputs, image_shape).detach().cpu().to(torch.float64)
        batch_sum = image.sum(dim=0)
        running_sum = batch_sum if running_sum is None else running_sum + batch_sum
        sample_count += int(image.shape[0])
    if running_sum is None or sample_count < 1:
        raise RuntimeError("reference dataset is empty")
    return _ReferenceMean(
        values=(running_sum / sample_count).to(torch.float32),
        sample_count=sample_count,
    )


def _sha256_tensor(values: torch.Tensor) -> str:
    array = np.ascontiguousarray(values.detach().cpu().numpy())
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


@dataclass(frozen=True)
class _CapturedResponse:
    values: torch.Tensor
    metadata: dict[str, Any]


def _module_network_layer_index(module_name: str) -> int | None:
    parts = module_name.split(".")
    for index, value in enumerate(parts[:-1]):
        if value == "layers":
            try:
                return int(parts[index + 1])
            except ValueError:
                return None
    return None


def _legacy_target_population(module_name: str) -> tuple[str | None, str | None]:
    if ".excitatory_cells." in module_name:
        return "excitatory_cells", "excitatory"
    if ".inhibitory_cells." in module_name:
        return "inhibitory_cells", "inhibitory"
    return None, None


def _mechanism_metadata(module: object) -> dict[str, Any]:
    config = getattr(module, "pop_config", None)

    def _setting(name: str, default: object = None) -> object:
        value = getattr(module, name, None)
        if value is not None:
            return value
        return getattr(config, name, default)

    use_shunting = bool(_setting("use_shunting", False))
    reactivate = bool(_setting("reactivate", False))
    additive_mode = str(_setting("additive_mode", "raw"))
    reactivation_type = _setting("reactivation_type")
    if reactivation_type is None:
        reactivation = getattr(module, "reactivation", None)
        reactivation_type = (
            None if reactivation is None else type(reactivation).__name__
        )
    if use_shunting:
        inhibitory_role = "positive_drive_added_to_shunting_denominator"
        integration = "shunting"
    elif additive_mode == "raw":
        inhibitory_role = "positive_drive_subtracted_before_reactivation"
        integration = "additive"
    else:
        inhibitory_role = (
            f"positive_drive_in_additive_comparison_control_{additive_mode}"
        )
        integration = "additive"
    return {
        "integration_mechanism": integration,
        "use_shunting": use_shunting,
        "additive_mode": additive_mode,
        "reactivation_enabled": reactivate,
        "reactivation_type": (
            None if reactivation_type is None else str(reactivation_type)
        ),
        "inhibitory_drive_semantics": inhibitory_role,
        "pre_gate_voltage_semantics": (
            "after_configured_EI_integration_before_branch_reactivation"
        ),
        "post_gate_output_semantics": "after_configured_branch_reactivation",
    }


def _unsupported_branch_capture_features(model: torch.nn.Module) -> list[str]:
    unsupported: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, StatefulDendriNet):
            continue
        enabled = []
        if bool(getattr(module, "synapse_types_enabled", False)):
            enabled.append("typed synapses")
        if getattr(module, "spiking_soma", None) is not None:
            enabled.append("spiking soma dynamics")
        if bool(getattr(module, "dendritic_spikes_enabled", False)):
            enabled.append("dendritic spikes")
        if bool(getattr(module, "soma_feedback_enabled", False)):
            enabled.append("soma feedback")
        if enabled:
            unsupported.append(f"{name}: {', '.join(enabled)}")
    return unsupported


class _ModelResponseCapture:
    """Capture one forward pass of population outputs and branch currents."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        target_population_polarities: tuple[str, ...],
    ):
        self._current: dict[str, _CapturedResponse] = {}
        self._handles: list[Any] = []
        self._branch_state: list[tuple[DendriticBranchLayer, object]] = []
        self._population_metadata: dict[int, dict[str, Any]] = {}
        self._target_population_polarities = set(target_population_polarities)

        population_modules = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, PopulationLayer)
        ]
        for fallback_index, (name, module) in enumerate(population_modules):
            network_index = _module_network_layer_index(name)
            if network_index is None:
                network_index = fallback_index
            polarities = {
                str(definition.name): str(definition.polarity)
                for definition in module.population_definitions
            }
            mechanism_by_population = {
                str(population_name): _mechanism_metadata(population)
                for population_name, population in module.populations.items()
            }
            self._population_metadata[id(module)] = {
                "module_name": name,
                "network_layer_index": int(network_index),
                "network_layer_name": str(getattr(module.config, "name", name)),
                "polarities": polarities,
                "mechanism_by_population": mechanism_by_population,
            }
            self._handles.append(module.register_forward_hook(self._population_hook))

        for name, module in model.named_modules():
            if not isinstance(module, DendriticBranchLayer):
                continue
            if module.branch_excitation is None and module.branch_inhibition is None:
                continue
            previous = getattr(module, "_store_analysis_currents", False)
            module._store_analysis_currents = True
            self._branch_state.append((module, previous))
            self._handles.append(
                module.register_forward_hook(
                    lambda branch, _inputs, _output, module_name=name: (
                        self._branch_hook(module_name, branch)
                    )
                )
            )

    def begin_forward(self) -> None:
        self._current = {}

    def snapshot(self) -> dict[str, _CapturedResponse]:
        return dict(self._current)

    def close(self) -> None:
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        for module, previous in self._branch_state:
            module._store_analysis_currents = previous
            if hasattr(module, "_last_analysis_currents"):
                delattr(module, "_last_analysis_currents")
        self._branch_state.clear()

    def _store(self, key: str, values: torch.Tensor, metadata: dict[str, Any]) -> None:
        if key in self._current:
            raise RuntimeError(
                f"response site {key!r} ran more than once in one image forward; "
                "recurrent/temporal models are outside this analyzer's contract"
            )
        if values.ndim < 2 or not torch.isfinite(values).all():
            raise ValueError(f"response site {key!r} produced invalid values")
        self._current[key] = _CapturedResponse(
            values=values.detach(), metadata=metadata
        )

    def _population_hook(self, module: PopulationLayer, _inputs, _output) -> None:
        metadata = self._population_metadata[id(module)]
        for population_name, values in module._last_outputs.items():
            if not isinstance(values, torch.Tensor):
                continue
            polarity = metadata["polarities"].get(str(population_name), "unknown")
            key = f"{metadata['module_name']}|soma|{population_name}"
            self._store(
                key,
                values,
                {
                    "response_type": "soma_population_output",
                    "signal": "post_gate_soma_output",
                    "module_name": metadata["module_name"],
                    "network_layer_index": metadata["network_layer_index"],
                    "network_layer_number": metadata["network_layer_index"] + 1,
                    "network_layer_name": metadata["network_layer_name"],
                    "population_name": str(population_name),
                    "population_polarity": polarity,
                    **metadata["mechanism_by_population"].get(str(population_name), {}),
                    "interpretation": "network-layer response propagation",
                },
            )

    def _branch_hook(
        self,
        module_name: str,
        module: DendriticBranchLayer,
    ) -> None:
        cached = getattr(module, "_last_analysis_currents", None)
        if not isinstance(cached, dict):
            raise RuntimeError(f"branch-current capture failed for {module_name}")
        network_index = _module_network_layer_index(module_name)
        population_name, population_polarity = _legacy_target_population(module_name)
        if ".populations." in module_name:
            suffix = module_name.split(".populations.", 1)[1]
            population_name = suffix.split(".", 1)[0]
            for metadata in self._population_metadata.values():
                if module_name.startswith(f"{metadata['module_name']}."):
                    population_polarity = metadata["polarities"].get(
                        population_name, "unknown"
                    )
                    network_index = metadata["network_layer_index"]
                    break
        if population_polarity not in self._target_population_polarities:
            return
        depth = soma_relative_dendritic_depth(module)
        mechanism = _mechanism_metadata(module)
        signals = (
            (
                "pre_gate_excitation_current",
                "branch_current",
                "pre_gate_synaptic_current",
                "excitatory",
            ),
            (
                "pre_gate_inhibition_current",
                "branch_current",
                "pre_gate_synaptic_current",
                "inhibitory",
            ),
            (
                "pre_gate_voltage",
                "branch_voltage",
                "pre_gate_compartment_voltage",
                None,
            ),
            (
                "post_gate_output",
                "branch_output",
                "post_gate_compartment_output",
                None,
            ),
        )
        for signal, response_type, response_stage, source_polarity in signals:
            values = cached.get(signal)
            if not isinstance(values, torch.Tensor):
                continue
            key = f"{module_name}|{signal}"
            record: dict[str, Any] = {
                "response_type": response_type,
                "response_stage": response_stage,
                "signal": signal,
                "module_name": module_name,
                "population_name": population_name,
                "population_polarity": population_polarity,
                "target_population_name": population_name,
                "target_population_polarity": population_polarity,
                "soma_relative_dendritic_depth": int(depth),
                "dendritic_depth_reference": SOMA_RELATIVE_DEPTH_REFERENCE,
                **mechanism,
                "interpretation": "response propagation, not a pixel receptive field",
            }
            if source_polarity is not None:
                record["current_polarity"] = source_polarity
                record["source_current_polarity"] = source_polarity
            if network_index is not None:
                record["network_layer_index"] = int(network_index)
                record["network_layer_number"] = int(network_index) + 1
            self._store(key, values, record)


@dataclass
class _PairedResponseAccumulator:
    metadata: dict[str, Any]
    value_count: int = 0
    sample_count: int = 0
    sum_baseline: float = 0.0
    sum_intervention: float = 0.0
    sum_absolute_delta: float = 0.0
    sum_squared_delta: float = 0.0
    sum_baseline_squared: float = 0.0

    def update(self, baseline: torch.Tensor, intervention: torch.Tensor) -> None:
        if baseline.shape != intervention.shape:
            raise RuntimeError("paired response shapes differ")
        baseline64 = baseline.detach().to(dtype=torch.float64)
        intervention64 = intervention.detach().to(dtype=torch.float64)
        delta = intervention64 - baseline64
        self.value_count += int(delta.numel())
        self.sample_count += int(delta.shape[0])
        self.sum_baseline += float(baseline64.sum().item())
        self.sum_intervention += float(intervention64.sum().item())
        self.sum_absolute_delta += float(delta.abs().sum().item())
        self.sum_squared_delta += float(delta.square().sum().item())
        self.sum_baseline_squared += float(baseline64.square().sum().item())

    def result(self) -> dict[str, Any]:
        if self.value_count < 1:
            raise RuntimeError("cannot summarize an empty paired response")
        delta_norm = math.sqrt(self.sum_squared_delta)
        baseline_norm = math.sqrt(self.sum_baseline_squared)
        return {
            **self.metadata,
            "sample_count": int(self.sample_count),
            "value_count": int(self.value_count),
            "baseline_mean": float(self.sum_baseline / self.value_count),
            "intervention_mean": float(self.sum_intervention / self.value_count),
            "mean_signed_change": float(
                (self.sum_intervention - self.sum_baseline) / self.value_count
            ),
            "mean_absolute_change": float(self.sum_absolute_delta / self.value_count),
            "root_mean_square_change": float(
                math.sqrt(self.sum_squared_delta / self.value_count)
            ),
            "relative_l2_change": float(
                delta_norm / max(baseline_norm, torch.finfo(torch.float64).eps)
            ),
        }


@dataclass
class _ClassificationAccumulator:
    sample_count: int = 0
    correct: int = 0
    nll_sum: float = 0.0

    def update(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        if logits.ndim != 2 or labels.ndim != 1:
            raise ValueError(
                "classification analysis requires 2-D logits and 1-D labels"
            )
        if logits.shape[0] != labels.shape[0]:
            raise RuntimeError("logit and label sample counts differ")
        self.sample_count += int(labels.numel())
        self.correct += int((logits.argmax(dim=1) == labels).sum().item())
        self.nll_sum += float(F.cross_entropy(logits, labels, reduction="sum").item())

    def result(self) -> dict[str, Any]:
        if self.sample_count < 1:
            raise RuntimeError("cannot summarize an empty evaluation")
        return {
            "sample_count": int(self.sample_count),
            "accuracy": float(self.correct / self.sample_count),
            "mean_negative_log_likelihood": float(self.nll_sum / self.sample_count),
        }


@dataclass
class _FractionAccumulator:
    count: int = 0
    total: float = 0.0
    total_squared: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, mask: torch.Tensor) -> None:
        fractions = mask.reshape(mask.shape[0], -1).to(torch.float64).mean(dim=1)
        self.count += int(fractions.numel())
        self.total += float(fractions.sum().item())
        self.total_squared += float(fractions.square().sum().item())
        self.minimum = min(self.minimum, float(fractions.min().item()))
        self.maximum = max(self.maximum, float(fractions.max().item()))

    def result(self) -> dict[str, float | int]:
        if self.count < 1:
            raise RuntimeError("cannot summarize empty region coverage")
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        return {
            "sample_count": int(self.count),
            "mean": float(mean),
            "standard_deviation": float(math.sqrt(variance)),
            "minimum": float(self.minimum),
            "maximum": float(self.maximum),
        }


def _paired_response_update(
    accumulators: dict[str, _PairedResponseAccumulator],
    class_accumulators: dict[str, _PairedResponseAccumulator],
    assigned_soma_accumulators: dict[str, _PairedResponseAccumulator],
    assigned_soma_class_accumulators: dict[str, _PairedResponseAccumulator],
    identity_matched_sites: set[str],
    baseline: dict[str, _CapturedResponse],
    intervention: dict[str, _CapturedResponse],
    labels: torch.Tensor,
    settings: InputRegionInterventionSettings,
) -> None:
    if baseline.keys() != intervention.keys():
        missing = sorted(set(baseline) - set(intervention))
        extra = sorted(set(intervention) - set(baseline))
        raise RuntimeError(
            f"baseline/intervention response sites differ: missing={missing}, extra={extra}"
        )
    for key, baseline_record in baseline.items():
        intervention_record = intervention[key]
        if baseline_record.metadata != intervention_record.metadata:
            raise RuntimeError(f"response metadata changed for {key}")
        accumulator = accumulators.setdefault(
            key,
            _PairedResponseAccumulator(metadata=dict(baseline_record.metadata)),
        )
        accumulator.update(baseline_record.values, intervention_record.values)
        if baseline_record.values.shape[0] != labels.shape[0]:
            raise RuntimeError(f"response/label sample counts differ for {key}")
        for class_label in sorted(int(value) for value in labels.unique().tolist()):
            selection = labels == class_label
            class_key = f"{key}|class={class_label}"
            class_accumulator = class_accumulators.setdefault(
                class_key,
                _PairedResponseAccumulator(
                    metadata={
                        **baseline_record.metadata,
                        "class_label": class_label,
                        "aggregation_scope": "within_true_class",
                    }
                ),
            )
            class_accumulator.update(
                baseline_record.values[selection],
                intervention_record.values[selection],
            )

        if not _is_identity_soma_site(baseline_record.metadata, settings):
            continue
        identity_matched_sites.add(key)
        baseline_assigned = _true_class_assigned_soma_values(
            baseline_record.values,
            labels,
            settings.identity_soma_class_mapping,
        )
        intervention_assigned = _true_class_assigned_soma_values(
            intervention_record.values,
            labels,
            settings.identity_soma_class_mapping,
        )
        assigned_metadata = {
            **baseline_record.metadata,
            "response_type": "true_class_assigned_soma_output",
            "aggregation_scope": "true_class_assigned_soma",
            "identity_soma_class_mapping": list(
                settings.identity_soma_class_mapping or ()
            ),
        }
        assigned_accumulator = assigned_soma_accumulators.setdefault(
            key,
            _PairedResponseAccumulator(metadata=assigned_metadata),
        )
        assigned_accumulator.update(baseline_assigned, intervention_assigned)
        for class_label in sorted(int(value) for value in labels.unique().tolist()):
            selection = labels == class_label
            class_key = f"{key}|assigned_class={class_label}"
            assigned_class_accumulator = assigned_soma_class_accumulators.setdefault(
                class_key,
                _PairedResponseAccumulator(
                    metadata={
                        **assigned_metadata,
                        "class_label": class_label,
                        "aggregation_scope": "true_class_assigned_soma_within_class",
                    }
                ),
            )
            assigned_class_accumulator.update(
                baseline_assigned[selection],
                intervention_assigned[selection],
            )


def _is_identity_soma_site(
    metadata: dict[str, Any],
    settings: InputRegionInterventionSettings,
) -> bool:
    if settings.identity_soma_class_mapping is None:
        return False
    return (
        metadata.get("response_type") == "soma_population_output"
        and metadata.get("population_name") == settings.identity_soma_population_name
        and metadata.get("network_layer_index")
        == settings.identity_soma_network_layer_index
    )


def _true_class_assigned_soma_values(
    values: torch.Tensor,
    labels: torch.Tensor,
    mapping: tuple[int, ...] | None,
) -> torch.Tensor:
    if mapping is None:
        raise RuntimeError("identity soma extraction requires an explicit mapping")
    if values.ndim != 2 or values.shape[1] != len(mapping):
        raise ValueError(
            "identity-mapped soma output must be [batch, n_somas] with one "
            "mapping entry per soma"
        )
    mapping_tensor = torch.as_tensor(mapping, device=labels.device, dtype=labels.dtype)
    matches = labels[:, None] == mapping_tensor[None, :]
    match_count = matches.sum(dim=1)
    if not torch.all(match_count == 1):
        missing = sorted(
            {int(value) for value in labels.tolist()} - {int(v) for v in mapping}
        )
        raise ValueError(
            "each evaluation class must map to exactly one soma; "
            f"unmapped classes={missing}"
        )
    soma_indices = matches.to(torch.int64).argmax(dim=1)
    return values.gather(1, soma_indices[:, None])


def _region_metadata(region: ImageRegionSpec) -> dict[str, Any]:
    result: dict[str, Any] = {
        "region_name": region.name,
        "region_kind": region.kind,
        "mask_scope": (
            "sample_dependent_intensity" if region.sample_dependent else "fixed_spatial"
        ),
        "sample_dependent": region.sample_dependent,
    }
    if region.sample_dependent:
        result.update(
            {
                "foreground_threshold": float(region.foreground_threshold),
                "foreground_rule": (
                    "mean_over_input_channels_strictly_greater_than_threshold"
                ),
                "background_rule": "complement_of_foreground_rule",
            }
        )
    else:
        result["center_fraction"] = float(region.center_fraction)
    return result


class InputRegionInterventionAnalyzer:
    """Evaluate paired input-region interventions on a frozen classifier."""

    def __init__(self, settings: InputRegionInterventionSettings):
        self.settings = settings

    def analyze(
        self,
        *,
        model: torch.nn.Module,
        test_dataset: torch.utils.data.Dataset,
        reference_dataset: torch.utils.data.Dataset | None = None,
        runtime: EvaluationRuntimeConfig | None = None,
        reference_runtime: EvaluationRuntimeConfig | None = None,
        device: str | torch.device = "cpu",
        save_path: str | None = None,
        filename: str = "final",
    ) -> dict[str, Any]:
        """Run all requested region/operation/replacement combinations.

        ``reference_dataset`` is mandatory for ``reference_mean`` replacement
        and must not be the evaluation dataset object.  The model is evaluated
        under ``torch.no_grad()`` and restored to its original device and mode.
        """

        settings = self.settings
        is_recurrent = getattr(
            getattr(model, "core_network", None), "is_recurrent", False
        )
        if callable(is_recurrent):
            is_recurrent = is_recurrent()
        if bool(is_recurrent):
            raise ValueError(
                "input-region intervention currently requires a feedforward model"
            )
        unsupported = _unsupported_branch_capture_features(model)
        if unsupported:
            raise ValueError(
                "input-region branch-response capture does not support stateful "
                "branch dynamics that bypass canonical DendriticBranchLayer.forward: "
                + "; ".join(unsupported)
            )

        reference: _ReferenceMean | None = None
        if "reference_mean" in settings.replacement_methods:
            if reference_dataset is None:
                raise ValueError(
                    "reference_mean replacement requires reference_dataset"
                )
            if reference_dataset is test_dataset:
                raise ValueError(
                    "reference_dataset and test_dataset must be distinct objects to "
                    "prevent evaluation-split calibration"
                )
            reference = estimate_featurewise_reference_mean(
                reference_dataset,
                image_shape=settings.image_shape,
                runtime=reference_runtime or runtime,
                max_samples=settings.max_reference_samples,
            )

        interventions: list[dict[str, Any]] = []
        shared_baseline: dict[str, Any] | None = None
        with analysis_device_context(model, device) as analysis_device:
            capture = _ModelResponseCapture(
                model,
                target_population_polarities=settings.target_population_polarities,
            )
            try:
                with torch.no_grad():
                    for region in settings.regions:
                        for operation in settings.operations:
                            for replacement_method in settings.replacement_methods:
                                record = self._evaluate_one(
                                    model=model,
                                    capture=capture,
                                    test_dataset=test_dataset,
                                    region=region,
                                    operation=operation,
                                    replacement_method=replacement_method,
                                    reference=reference,
                                    runtime=runtime,
                                    device=analysis_device,
                                )
                                baseline = record.pop("baseline")
                                if shared_baseline is None:
                                    shared_baseline = baseline
                                elif baseline != shared_baseline:
                                    raise RuntimeError(
                                        "baseline metrics changed across paired intervention passes"
                                    )
                                interventions.append(record)
            finally:
                capture.close()

        if shared_baseline is None:
            raise RuntimeError("no input-region interventions were evaluated")

        canonical_shape = _canonical_image_shape(settings.image_shape)
        result: dict[str, Any] = {
            "schema_version": 1,
            "analysis_type": "input_region_intervention",
            "analysis_contract": (
                "paired frozen-checkpoint input intervention; fixed spatial masks "
                "and sample-dependent intensity masks are distinct estimands"
            ),
            "image_shape_channels_height_width": list(canonical_shape),
            "evaluation_split": settings.evaluation_split,
            "evaluation_max_samples": settings.max_evaluation_samples,
            "region_definitions": [
                _region_metadata(region) for region in settings.regions
            ],
            "operations": list(settings.operations),
            "replacement_methods": list(settings.replacement_methods),
            "branch_target_population_polarities": list(
                settings.target_population_polarities
            ),
            "replacement_semantics": {
                "zero": "numerical feature-space zero",
                "reference_mean": (
                    "featurewise mean estimated on the named reference split and "
                    "held fixed on the evaluation split"
                ),
            },
            "baseline": shared_baseline,
            "interventions": interventions,
            "response_interpretation": (
                "layer and dendritic-depth records quantify response propagation "
                "from the input intervention; later-layer records are not literal "
                "pixel receptive fields"
            ),
            "serialization_policy": (
                "aggregate paired metrics only; no per-sample activation or "
                "sample-by-synapse arrays"
            ),
        }
        if reference is not None:
            result["reference"] = {
                "split": settings.reference_split,
                "max_samples": settings.max_reference_samples,
                "sample_count": int(reference.sample_count),
                "featurewise_mean_sha256": _sha256_tensor(reference.values),
                "featurewise_mean_global_mean": float(reference.values.mean().item()),
                "featurewise_mean_global_standard_deviation": float(
                    reference.values.std(unbiased=False).item()
                ),
            }
        if settings.identity_soma_class_mapping is not None:
            result["identity_soma_mapping"] = {
                "network_layer_index": int(
                    settings.identity_soma_network_layer_index  # type: ignore[arg-type]
                ),
                "population_name": settings.identity_soma_population_name,
                "soma_index_to_class": list(settings.identity_soma_class_mapping),
            }
        if save_path is not None:
            save_dict(result, save_path, f"{Path(filename).stem}.json")
        return result

    def _evaluate_one(
        self,
        *,
        model: torch.nn.Module,
        capture: _ModelResponseCapture,
        test_dataset: torch.utils.data.Dataset,
        region: ImageRegionSpec,
        operation: InterventionOperation,
        replacement_method: ReplacementMethod,
        reference: _ReferenceMean | None,
        runtime: EvaluationRuntimeConfig | None,
        device: torch.device,
    ) -> dict[str, Any]:
        baseline_metrics = _ClassificationAccumulator()
        intervention_metrics = _ClassificationAccumulator()
        response_accumulators: dict[str, _PairedResponseAccumulator] = {}
        class_response_accumulators: dict[str, _PairedResponseAccumulator] = {}
        assigned_soma_accumulators: dict[str, _PairedResponseAccumulator] = {}
        assigned_soma_class_accumulators: dict[str, _PairedResponseAccumulator] = {}
        identity_matched_sites: set[str] = set()
        region_coverage = _FractionAccumulator()
        replacement_coverage = _FractionAccumulator()

        for batch in iter_analysis_batches(
            test_dataset,
            runtime,
            explicit_max_samples=self.settings.max_evaluation_samples,
            device=device,
        ):
            if len(batch) < 2:
                raise TypeError(
                    "input-region intervention requires (input, label) data"
                )
            inputs = batch[0].to(device)
            labels = batch[1].to(device).reshape(-1).long()
            region_mask = build_input_region_mask(
                inputs,
                image_shape=self.settings.image_shape,
                region=region,
            )
            if replacement_method == "zero":
                replacement = torch.zeros((), device=device, dtype=inputs.dtype)
            elif reference is not None:
                image = _as_image_batch(inputs, self.settings.image_shape)
                replacement_image = reference.values.to(
                    device=device, dtype=inputs.dtype
                ).expand(image.shape[0], -1, -1, -1)
                replacement = replacement_image.reshape_as(inputs)
            else:
                raise RuntimeError("reference mean was not calibrated")
            altered, replacement_mask = apply_input_region_intervention(
                inputs,
                region_mask=region_mask,
                operation=operation,
                replacement=replacement,
            )

            capture.begin_forward()
            baseline_logits = model(inputs)
            baseline_responses = capture.snapshot()
            capture.begin_forward()
            intervention_logits = model(altered)
            intervention_responses = capture.snapshot()

            baseline_metrics.update(baseline_logits, labels)
            intervention_metrics.update(intervention_logits, labels)
            _paired_response_update(
                response_accumulators,
                class_response_accumulators,
                assigned_soma_accumulators,
                assigned_soma_class_accumulators,
                identity_matched_sites,
                baseline_responses,
                intervention_responses,
                labels,
                self.settings,
            )
            region_coverage.update(region_mask)
            replacement_coverage.update(replacement_mask)

        baseline = baseline_metrics.result()
        intervention = intervention_metrics.result()
        if (
            self.settings.identity_soma_class_mapping is not None
            and len(identity_matched_sites) != 1
        ):
            raise RuntimeError(
                "identity soma mapping must match exactly one captured population, "
                f"matched={sorted(identity_matched_sites)}"
            )
        classification = {
            "accuracy": intervention["accuracy"],
            "accuracy_change": float(intervention["accuracy"] - baseline["accuracy"]),
            "accuracy_drop": float(baseline["accuracy"] - intervention["accuracy"]),
            "mean_negative_log_likelihood": intervention[
                "mean_negative_log_likelihood"
            ],
            "mean_negative_log_likelihood_change": float(
                intervention["mean_negative_log_likelihood"]
                - baseline["mean_negative_log_likelihood"]
            ),
        }
        return {
            **_region_metadata(region),
            "operation": operation,
            "operation_semantics": (
                "replace_named_region"
                if operation == "remove_only"
                else "replace_complement_and_retain_named_region"
            ),
            "replacement_method": replacement_method,
            "baseline": baseline,
            "classification": classification,
            "named_region_feature_fraction": region_coverage.result(),
            "replaced_feature_fraction": replacement_coverage.result(),
            "response_records": [
                response_accumulators[key].result()
                for key in sorted(response_accumulators)
            ],
            "response_class_records": [
                class_response_accumulators[key].result()
                for key in sorted(class_response_accumulators)
            ],
            "assigned_soma_response_records": [
                assigned_soma_accumulators[key].result()
                for key in sorted(assigned_soma_accumulators)
            ],
            "assigned_soma_class_response_records": [
                assigned_soma_class_accumulators[key].result()
                for key in sorted(assigned_soma_class_accumulators)
            ],
        }


__all__ = [
    "ImageRegionSpec",
    "InputRegionInterventionAnalyzer",
    "InputRegionInterventionSettings",
    "apply_input_region_intervention",
    "build_input_region_mask",
    "estimate_featurewise_reference_mean",
]

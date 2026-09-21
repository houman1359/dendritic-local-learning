"""Phase-specific, same-checkpoint interventions for population-network RNNs.

The utilities in this module intervene on the *output current* of a selected
sparse synaptic pathway.  They do not edit weights or retrain the model.  A
manual timestep loop is required because each trial in a variable-delay task
can occupy a different phase at the same padded timestep.

Only the current rate-mode classification wrappers around
``PopulationNetwork`` are supported.  This includes the ``RecurrentClassifier``
created by the real recurrent factory as well as the scale-aware ``Classifier``
wrapper. Typed conductances, spiking dynamics, stochastic selectors, and
forward-mutating sparsification schedules are rejected rather than producing
an ambiguous causal comparison.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral
from typing import Literal

import torch

from dendritic_modeling.analysis.utils.dendritic_depth import (
    SOMA_RELATIVE_DEPTH_REFERENCE,
    soma_relative_dendritic_depth,
)
from dendritic_modeling.models.classifier import Classifier
from dendritic_modeling.models.recurrent_classifier import RecurrentClassifier
from dendritic_modeling.networks.architectures.classical.identity import Identity
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    apply_recurrent_weight_cache,
)
from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationNetwork,
)
from dendritic_modeling.networks.architectures.recurrent.population_state import (
    PopulationLayerState,
)
from dendritic_modeling.utils.stable_hash import stable_seed_offset

PathwayName = Literal[
    "ff_excitatory",
    "rec_excitatory",
    "rec_inhibitory",
]
InterventionMode = Literal["zero", "shuffle", "fixed_mean"]
PopulationClassificationModel = Classifier | RecurrentClassifier

_PATHWAY_ATTRIBUTES: dict[str, str] = {
    "ff_excitatory": "branch_excitation",
    "rec_excitatory": "branch_recurrent",
    "rec_inhibitory": "branch_rec_inhibition",
}
_INTERVENTION_MODES = frozenset({"zero", "shuffle", "fixed_mean"})
_RUNTIME_ATTRIBUTES = (
    "_routing_info",
    "_last_outputs",
    "_last_routing_info",
    "_last_forward_weight_mask",
    "_last_forward_active_sparse_mask",
    "_last_forward_param_tensor",
    "_recurrent_weight_cache_enabled",
    "_recurrent_cached_weight_mask",
    "_recurrent_cached_pruned_weight",
)
ROMO_PHASE_ORDER = (
    "first_stimulus",
    "delay",
    "second_stimulus",
    "response",
)


@dataclass(frozen=True)
class PathwayInterventionTarget:
    """One sparse pathway identified in scientific, soma-relative coordinates."""

    layer: int | str
    target_population: str
    soma_relative_depth: int
    pathway: PathwayName

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, (Integral, str)):
            raise TypeError("target layer must be an integer index or layer name")
        if isinstance(self.layer, Integral) and int(self.layer) < 0:
            raise ValueError("target layer index must be non-negative")
        if isinstance(self.layer, str) and not self.layer:
            raise ValueError("target layer name must be non-empty")
        if not str(self.target_population):
            raise ValueError("target_population must be non-empty")
        if isinstance(self.soma_relative_depth, bool) or not isinstance(
            self.soma_relative_depth, Integral
        ):
            raise TypeError("soma_relative_depth must be an integer")
        if int(self.soma_relative_depth) < 0:
            raise ValueError("soma_relative_depth must be non-negative")
        if self.pathway not in _PATHWAY_ATTRIBUTES:
            raise ValueError(
                f"pathway must be one of {sorted(_PATHWAY_ATTRIBUTES)}, got "
                f"{self.pathway!r}"
            )


@dataclass(frozen=True)
class ResolvedPathwayTarget:
    """A validated intervention target and its concrete sparse module."""

    layer_index: int
    layer_name: str
    target_population: str
    soma_relative_depth: int
    pathway: str
    module: torch.nn.Module

    @property
    def key(self) -> str:
        return (
            f"layer={self.layer_name}|population={self.target_population}|"
            f"depth={self.soma_relative_depth}|pathway={self.pathway}"
        )

    def metadata(self) -> dict[str, object]:
        return {
            "key": self.key,
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "target_population": self.target_population,
            "soma_relative_depth": self.soma_relative_depth,
            "depth_reference": SOMA_RELATIVE_DEPTH_REFERENCE,
            "pathway": self.pathway,
            "module_type": type(self.module).__name__,
        }


@dataclass(frozen=True)
class CausalInterventionBatch:
    """A reference/evaluation batch with an explicit per-trial phase mask."""

    inputs: torch.Tensor
    phase_mask: torch.Tensor
    seq_lengths: torch.Tensor | None = None


@dataclass(frozen=True)
class FixedMeanCalibration:
    """Feature-wise pathway means estimated from explicit reference batches."""

    references: dict[str, torch.Tensor]
    counts: dict[str, int]
    source_id: str

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("fixed-mean calibration requires a non-empty source_id")
        if set(self.references) != set(self.counts):
            raise ValueError("fixed-mean reference and count keys must match")
        for key, reference in self.references.items():
            if not isinstance(reference, torch.Tensor) or reference.ndim != 1:
                raise ValueError(
                    f"fixed-mean reference {key!r} must be a one-dimensional tensor"
                )
            if int(self.counts[key]) <= 0:
                raise ValueError(f"fixed-mean reference {key!r} has no observations")
        object.__setattr__(
            self,
            "references",
            {
                str(key): reference.detach()
                .to(dtype=torch.float64, device="cpu")
                .clone()
                for key, reference in self.references.items()
            },
        )
        object.__setattr__(
            self,
            "counts",
            {str(key): int(count) for key, count in self.counts.items()},
        )

    def metadata(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "counts": dict(self.counts),
            "reference_shapes": {
                key: list(reference.shape) for key, reference in self.references.items()
            },
        }


@dataclass(frozen=True)
class InterventionAudit:
    """Direct hook applications, separated from downstream causal effects."""

    targets: tuple[dict[str, object], ...]
    touched_masks: dict[str, torch.Tensor]
    directly_changed_masks: dict[str, torch.Tensor]
    calls_per_timestep: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class ManualCausalForwardResult:
    """Logits and direct-intervention audit from one manual sequence unroll."""

    logits: torch.Tensor
    audit: InterventionAudit


@dataclass(frozen=True)
class SameCheckpointInterventionResult:
    """Per-trial predictions and aggregate drop for one causal comparison."""

    baseline_logits: torch.Tensor
    intervened_logits: torch.Tensor
    labels: torch.Tensor
    baseline_predictions: torch.Tensor
    intervened_predictions: torch.Tensor
    baseline_correct: torch.Tensor
    intervened_correct: torch.Tensor
    per_trial_accuracy_drop: torch.Tensor
    baseline_accuracy: float
    intervened_accuracy: float
    accuracy_drop: float
    mode: str
    audit: InterventionAudit
    calibration_metadata: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe result with one record per evaluated trial."""

        touched_per_trial = {
            key: mask.sum(dim=1).tolist()
            for key, mask in self.audit.touched_masks.items()
        }
        changed_per_trial = {
            key: mask.sum(dim=1).tolist()
            for key, mask in self.audit.directly_changed_masks.items()
        }
        trial_records = []
        for index in range(self.labels.shape[0]):
            trial_records.append(
                {
                    "trial_index": index,
                    "label": int(self.labels[index].item()),
                    "baseline_logits": self.baseline_logits[index].tolist(),
                    "intervened_logits": self.intervened_logits[index].tolist(),
                    "baseline_prediction": int(self.baseline_predictions[index].item()),
                    "intervened_prediction": int(
                        self.intervened_predictions[index].item()
                    ),
                    "baseline_correct": bool(self.baseline_correct[index].item()),
                    "intervened_correct": bool(self.intervened_correct[index].item()),
                    "accuracy_drop": float(self.per_trial_accuracy_drop[index].item()),
                    "directly_touched_timesteps": {
                        key: int(counts[index])
                        for key, counts in touched_per_trial.items()
                    },
                    "directly_changed_timesteps": {
                        key: int(counts[index])
                        for key, counts in changed_per_trial.items()
                    },
                }
            )
        return {
            "analysis_type": "same_checkpoint_phase_pathway_intervention",
            "mode": self.mode,
            "baseline_accuracy": self.baseline_accuracy,
            "intervened_accuracy": self.intervened_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "targets": list(self.audit.targets),
            "direct_intervention_audit": {
                "calls_per_timestep": {
                    key: list(counts)
                    for key, counts in self.audit.calls_per_timestep.items()
                },
                "touched_timesteps_per_trial": touched_per_trial,
                "changed_timesteps_per_trial": changed_per_trial,
            },
            "calibration": self.calibration_metadata,
            "trials": trial_records,
        }


def romo_phase_masks(
    delays: torch.Tensor,
    *,
    max_sequence_length: int,
    stimulus_duration: int,
    response_duration: int,
    seq_lengths: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build non-overlapping per-sample masks for variable-delay Romo trials."""

    if not isinstance(delays, torch.Tensor) or delays.ndim != 1:
        raise ValueError("delays must be a one-dimensional tensor")
    if delays.dtype == torch.bool or delays.is_floating_point():
        raise TypeError("delays must use an integer dtype")
    if max_sequence_length < 1:
        raise ValueError("max_sequence_length must be positive")
    if stimulus_duration < 1 or response_duration < 1:
        raise ValueError("stimulus_duration and response_duration must be positive")
    delays = delays.long()
    if bool((delays < 0).any()):
        raise ValueError("Romo delays must be non-negative")
    expected_lengths = 2 * int(stimulus_duration) + delays + int(response_duration)
    if bool((expected_lengths > int(max_sequence_length)).any()):
        raise ValueError("max_sequence_length is shorter than at least one Romo trial")
    if seq_lengths is not None:
        if seq_lengths.shape != delays.shape:
            raise ValueError("seq_lengths and delays must have the same shape")
        if not torch.equal(
            seq_lengths.to(device=delays.device).long(), expected_lengths
        ):
            raise ValueError(
                "seq_lengths do not match 2*stimulus_duration + delay + "
                "response_duration"
            )

    time = torch.arange(max_sequence_length, device=delays.device).unsqueeze(0)
    first_end = torch.full_like(delays, int(stimulus_duration)).unsqueeze(1)
    second_start = first_end + delays.unsqueeze(1)
    second_end = second_start + int(stimulus_duration)
    response_end = expected_lengths.unsqueeze(1)
    masks = {
        "first_stimulus": time < first_end,
        "delay": (time >= first_end) & (time < second_start),
        "second_stimulus": (time >= second_start) & (time < second_end),
        "response": (time >= second_end) & (time < response_end),
    }
    masks["all_valid"] = time < response_end
    return masks


def romo_phase_mask(
    delays: torch.Tensor,
    phases: str | Sequence[str],
    *,
    max_sequence_length: int,
    stimulus_duration: int,
    response_duration: int,
    seq_lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the union of one or more named variable-delay Romo phases."""

    masks = romo_phase_masks(
        delays,
        max_sequence_length=max_sequence_length,
        stimulus_duration=stimulus_duration,
        response_duration=response_duration,
        seq_lengths=seq_lengths,
    )
    requested = (phases,) if isinstance(phases, str) else tuple(phases)
    if not requested:
        raise ValueError("at least one Romo phase must be selected")
    unknown = sorted(set(requested).difference(masks))
    if unknown:
        raise ValueError(f"unknown Romo phases: {unknown}")
    combined = torch.zeros_like(next(iter(masks.values())))
    for phase in requested:
        combined |= masks[phase]
    return combined


def _validate_supported_model(
    model: PopulationClassificationModel,
) -> PopulationNetwork:
    if not isinstance(model, (Classifier, RecurrentClassifier)):
        raise TypeError(
            "causal pathway interventions require Classifier or "
            "RecurrentClassifier, got "
            f"{type(model).__name__}"
        )
    core = model.core_network
    if not isinstance(core, PopulationNetwork):
        raise TypeError(
            "causal pathway interventions require PopulationNetwork core, got "
            f"{type(core).__name__}"
        )
    if not core.is_recurrent or not bool(getattr(model, "_is_recurrent_core", False)):
        raise ValueError("PopulationNetwork core must be recurrent")
    if not isinstance(model.encoder_network, torch.nn.Module) or not isinstance(
        model.decoder_network, torch.nn.Module
    ):
        raise TypeError(
            "Classification-model encoder and decoder must be torch modules"
        )

    for layer in core.layers:
        for population_name, population in layer.populations.items():
            enabled = []
            if getattr(population, "spiking_soma", None) is not None:
                enabled.append("spiking_soma")
            if bool(getattr(population, "synapse_types_enabled", False)):
                enabled.append("typed_synapses")
            if bool(getattr(population, "dendritic_spikes_enabled", False)):
                enabled.append("dendritic_spikes")
            if bool(getattr(population, "soma_feedback_enabled", False)):
                enabled.append("soma_feedback")
            if enabled:
                raise ValueError(
                    f"population {layer.config.name}.{population_name} is not "
                    f"supported rate mode; enabled: {', '.join(enabled)}"
                )

    for name, module in core.named_modules():
        selection = str(getattr(module, "selection", "")).lower()
        class_name = type(module).__name__.lower()
        noise_level = float(getattr(module, "noise_level", 0.0) or 0.0)
        if (
            "stochastic" in class_name
            or selection in {"stochastic", "rank_probabilistic"}
            or noise_level > 0.0
        ):
            raise ValueError(
                f"module {name!r} ({type(module).__name__}) has stochastic "
                "forward-time connectivity"
            )
        if getattr(module, "_supports_recurrent_weight_cache", None) is False:
            raise ValueError(
                f"module {name!r} ({type(module).__name__}) uses a "
                "forward-mutating sparsification policy"
            )
    return core


def resolve_pathway_targets(
    core: PopulationNetwork,
    targets: Sequence[PathwayInterventionTarget],
) -> tuple[ResolvedPathwayTarget, ...]:
    """Resolve scientific target coordinates to exact sparse modules."""

    if not isinstance(core, PopulationNetwork):
        raise TypeError("target resolution requires a PopulationNetwork")
    if not targets:
        return ()
    resolved: list[ResolvedPathwayTarget] = []
    seen_modules: set[int] = set()
    for target in targets:
        if not isinstance(target, PathwayInterventionTarget):
            raise TypeError("targets must contain PathwayInterventionTarget objects")
        if isinstance(target.layer, Integral):
            layer_index = int(target.layer)
            if layer_index >= len(core.layers):
                raise ValueError(f"target layer index {layer_index} does not exist")
        else:
            matches = [
                index
                for index, layer in enumerate(core.layers)
                if layer.config.name == target.layer
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"target layer name {target.layer!r} matched {len(matches)} layers"
                )
            layer_index = matches[0]
        layer = core.layers[layer_index]
        if target.target_population not in layer.populations:
            raise ValueError(
                f"population {target.target_population!r} does not exist in "
                f"layer {layer.config.name!r}"
            )
        population = layer.populations[target.target_population]
        depth_matches = [
            branch_layer
            for branch_layer in population.branch_layers
            if soma_relative_dendritic_depth(branch_layer)
            == int(target.soma_relative_depth)
        ]
        if len(depth_matches) != 1:
            raise ValueError(
                f"population {layer.config.name}.{target.target_population} has "
                f"{len(depth_matches)} branches at soma-relative depth "
                f"{target.soma_relative_depth}"
            )
        module = getattr(depth_matches[0], _PATHWAY_ATTRIBUTES[target.pathway], None)
        if module is None:
            raise ValueError(
                f"target pathway {target.pathway} is absent at "
                f"{layer.config.name}.{target.target_population} depth "
                f"{target.soma_relative_depth}"
            )
        if id(module) in seen_modules:
            raise ValueError(
                "the same sparse pathway module was targeted more than once"
            )
        seen_modules.add(id(module))
        resolved.append(
            ResolvedPathwayTarget(
                layer_index=layer_index,
                layer_name=layer.config.name,
                target_population=target.target_population,
                soma_relative_depth=int(target.soma_relative_depth),
                pathway=target.pathway,
                module=module,
            )
        )
    return tuple(resolved)


def _validate_sequence_inputs(
    inputs: torch.Tensor,
    phase_mask: torch.Tensor | None,
    seq_lengths: torch.Tensor | None,
) -> torch.Tensor | None:
    if not isinstance(inputs, torch.Tensor) or inputs.ndim != 3:
        raise ValueError("inputs must have shape [batch, timesteps, features]")
    batch_size, timesteps, _ = inputs.shape
    if timesteps < 1:
        raise ValueError("inputs must contain at least one timestep")
    if seq_lengths is not None:
        if not isinstance(seq_lengths, torch.Tensor) or seq_lengths.shape != (
            batch_size,
        ):
            raise ValueError("seq_lengths must have shape [batch]")
        if seq_lengths.dtype == torch.bool or seq_lengths.is_floating_point():
            raise TypeError("seq_lengths must use an integer dtype")
    if phase_mask is None:
        return None
    if not isinstance(phase_mask, torch.Tensor) or phase_mask.dtype != torch.bool:
        raise TypeError("phase_mask must be a boolean tensor")
    if phase_mask.shape != (batch_size, timesteps):
        raise ValueError(
            f"phase_mask must have shape {(batch_size, timesteps)}, got "
            f"{tuple(phase_mask.shape)}"
        )
    phase_mask = phase_mask.to(device=inputs.device)
    if seq_lengths is not None:
        lengths = seq_lengths.to(device=inputs.device).long().clamp(0, timesteps)
        valid = torch.arange(timesteps, device=inputs.device).unsqueeze(
            0
        ) < lengths.unsqueeze(1)
        if bool((phase_mask & ~valid).any()):
            raise ValueError("phase_mask selects padded timesteps")
    return phase_mask


@contextmanager
def _preserve_analysis_model_state(model: PopulationClassificationModel):
    """Use eval mode temporarily and restore hooks/caches/diagnostics on exit."""

    training_flags = [(module, module.training) for module in model.modules()]
    runtime_values = []
    for module in model.modules():
        for attribute in _RUNTIME_ATTRIBUTES:
            if hasattr(module, attribute):
                runtime_values.append((module, attribute, getattr(module, attribute)))
    model.eval()
    try:
        yield
    finally:
        for module, attribute, value in runtime_values:
            setattr(module, attribute, value)
        # Direct assignment preserves intentionally mixed train/eval submodule modes.
        for module, was_training in training_flags:
            module.training = was_training


def _encode_classifier_sequence(
    model: PopulationClassificationModel, inputs: torch.Tensor
) -> torch.Tensor:
    """Match ``BaseModel.forward`` encoder semantics exactly for recurrent input."""

    if not isinstance(model.encoder_network, Identity) and inputs.dim() == 3:
        batch_size, timesteps, feature_dim = inputs.shape
        encoded = model.encoder_network(
            inputs.reshape(batch_size * timesteps, feature_dim)
        )
        return encoded.reshape(batch_size, timesteps, -1)
    return model.encoder_network(inputs)


def _decode_and_scale_classifier(
    model: PopulationClassificationModel, core_output: torch.Tensor
) -> torch.Tensor:
    """Match both classification wrappers' decoder/scale ordering exactly."""

    logits = model.decoder_network(core_output)
    logits = float(getattr(model, "fixed_output_scale", 1.0)) * logits
    if hasattr(model, "log_output_scale"):
        logits = model.log_output_scale.exp() * logits
    elif hasattr(model, "log_global_output_scale"):
        logits = model.log_global_output_scale.exp() * logits
    return logits


class _TimestepPhaseContext:
    def __init__(self, phase_mask: torch.Tensor | None = None) -> None:
        self.phase_mask = phase_mask
        self.timestep = -1
        self.current_mask: torch.Tensor | None = None

    def reset(self, phase_mask: torch.Tensor | None) -> None:
        self.phase_mask = phase_mask
        self.timestep = -1
        self.current_mask = None

    def set_timestep(self, timestep: int) -> None:
        self.timestep = int(timestep)
        self.current_mask = (
            None if self.phase_mask is None else self.phase_mask[:, self.timestep]
        )

    def clear(self) -> None:
        self.timestep = -1
        self.current_mask = None


def _manual_population_core_unroll(
    core: PopulationNetwork,
    encoded: torch.Tensor,
    *,
    seq_lengths: torch.Tensor | None,
    timestep_context: _TimestepPhaseContext,
) -> torch.Tensor:
    """Mirror ``PopulationNetwork._forward_recurrent`` one timestep at a time."""

    batch_size, sequence_length, _ = encoded.shape
    hidden = core.init_state(
        batch_size=batch_size,
        device=encoded.device,
        dtype=encoded.dtype,
    )
    outputs = []
    core._routing_info = []
    apply_recurrent_weight_cache(core, True)
    try:
        for timestep in range(sequence_length):
            timestep_context.set_timestep(timestep)
            default_inputs = core._project_external_inputs(encoded[:, timestep, :])
            previous_layer_outputs = {
                layer.config.name: hidden[layer_index].outputs
                for layer_index, layer in enumerate(core.layers)
            }
            current_layer_outputs: dict[str, dict[str, torch.Tensor]] = {}
            timestep_hidden = []
            readout = None
            timestep_info = {}
            for layer_index, layer in enumerate(core.layers):
                external_inputs = core._same_step_external_inputs_for_layer(
                    layer_index,
                    default_inputs,
                    current_layer_outputs,
                )
                delayed_external_inputs = core._delayed_external_inputs_for_layer(
                    layer_index,
                    previous_layer_outputs,
                )
                layer_output, new_layer_state = layer(
                    external_inputs,
                    hidden[layer_index],
                    delayed_external_inputs=delayed_external_inputs,
                )
                if new_layer_state is None:
                    new_layer_state = PopulationLayerState(
                        population_states=hidden[layer_index].population_states,
                        outputs=layer._last_outputs,
                    )
                timestep_hidden.append(new_layer_state)
                current_layer_outputs[layer.config.name] = new_layer_state.outputs
                if layer.config.name == core.readout_layer:
                    readout = layer_output
                if core._store_routing:
                    timestep_info[layer.config.name] = layer._last_routing_info
                default_inputs = core._single_stream_external_inputs(layer_output)
            hidden = timestep_hidden
            outputs.append(layer_output if readout is None else readout)
            if core._store_routing:
                core._routing_info.append(timestep_info)
    finally:
        timestep_context.clear()
        apply_recurrent_weight_cache(core, False)
    stacked = torch.stack(outputs, dim=1)
    return core._reduce_outputs(stacked, seq_lengths=seq_lengths)


class _InterventionController:
    def __init__(
        self,
        *,
        resolved_targets: tuple[ResolvedPathwayTarget, ...],
        phase_mask: torch.Tensor,
        mode: str,
        shuffle_seed: int,
        calibration: FixedMeanCalibration | None,
    ) -> None:
        if mode not in _INTERVENTION_MODES:
            raise ValueError(
                f"intervention mode must be one of {sorted(_INTERVENTION_MODES)}, "
                f"got {mode!r}"
            )
        if not resolved_targets:
            raise ValueError("an intervention requires at least one pathway target")
        if mode == "fixed_mean" and calibration is None:
            raise ValueError(
                "fixed_mean requires an explicit FixedMeanCalibration from a "
                "separate reference set; per-batch fallback is not allowed"
            )
        self.resolved_targets = resolved_targets
        self.phase_context = _TimestepPhaseContext(phase_mask)
        self.mode = mode
        self.shuffle_seed = int(shuffle_seed)
        self.calibration = calibration
        self.touched = {
            target.key: torch.zeros_like(phase_mask) for target in resolved_targets
        }
        self.changed = {
            target.key: torch.zeros_like(phase_mask) for target in resolved_targets
        }
        self.calls = {
            target.key: [0 for _ in range(phase_mask.shape[1])]
            for target in resolved_targets
        }

        if calibration is not None and mode == "fixed_mean":
            missing = sorted(
                {target.key for target in resolved_targets}.difference(
                    calibration.references
                )
            )
            if missing:
                raise ValueError(
                    f"fixed-mean calibration is missing targets: {missing}"
                )

    def _shuffle_active(
        self,
        output: torch.Tensor,
        active: torch.Tensor,
        target: ResolvedPathwayTarget,
    ) -> torch.Tensor:
        active_indices = active.nonzero(as_tuple=False).flatten()
        if active_indices.numel() <= 1:
            return output
        seed = (
            self.shuffle_seed
            + stable_seed_offset(
                "phase_pathway_shuffle",
                target.key,
                self.phase_context.timestep,
            )
        ) % ((1 << 63) - 1)
        generator = torch.Generator(device=output.device)
        generator.manual_seed(seed)
        order = torch.randperm(
            active_indices.numel(),
            generator=generator,
            device=output.device,
        )
        result = output.clone()
        result[active_indices] = output[active_indices[order]]
        return result

    def _hook(self, target: ResolvedPathwayTarget):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise TypeError(
                    f"target {target.key} must produce [batch, features] tensor"
                )
            active = self.phase_context.current_mask
            timestep = self.phase_context.timestep
            if active is None or timestep < 0:
                raise RuntimeError(
                    "pathway hook fired outside the manual timestep loop"
                )
            if output.shape[0] != active.shape[0]:
                raise ValueError("pathway output batch does not match phase mask")
            self.touched[target.key][:, timestep] |= active
            self.calls[target.key][timestep] += 1
            if not bool(active.any()):
                return output

            expanded = active.unsqueeze(1)
            if self.mode == "zero":
                modified = torch.where(expanded, torch.zeros_like(output), output)
            elif self.mode == "shuffle":
                modified = self._shuffle_active(output, active, target)
            else:
                assert self.calibration is not None
                reference = self.calibration.references[target.key]
                if reference.shape != output.shape[1:]:
                    raise ValueError(
                        f"fixed-mean reference for {target.key} has shape "
                        f"{tuple(reference.shape)}, expected {tuple(output.shape[1:])}"
                    )
                reference = reference.to(device=output.device, dtype=output.dtype)
                modified = torch.where(
                    expanded,
                    reference.unsqueeze(0).expand_as(output),
                    output,
                )
            direct_change = (modified != output).reshape(output.shape[0], -1).any(dim=1)
            self.changed[target.key][:, timestep] |= direct_change
            return modified

        return hook

    @contextmanager
    def installed(self):
        handles = []
        try:
            for target in self.resolved_targets:
                handles.append(
                    target.module.register_forward_hook(
                        self._hook(target),
                        prepend=True,
                    )
                )
            yield self.phase_context
        finally:
            for handle in reversed(handles):
                handle.remove()

    def audit(self) -> InterventionAudit:
        return InterventionAudit(
            targets=tuple(target.metadata() for target in self.resolved_targets),
            touched_masks={
                key: mask.detach().cpu() for key, mask in self.touched.items()
            },
            directly_changed_masks={
                key: mask.detach().cpu() for key, mask in self.changed.items()
            },
            calls_per_timestep={
                key: tuple(counts) for key, counts in self.calls.items()
            },
        )


def _manual_classifier_logits(
    model: PopulationClassificationModel,
    inputs: torch.Tensor,
    *,
    seq_lengths: torch.Tensor | None,
    timestep_context: _TimestepPhaseContext,
) -> torch.Tensor:
    encoded = _encode_classifier_sequence(model, inputs)
    if encoded.ndim != 3:
        raise ValueError("Classifier encoder must preserve a sequence dimension")
    core_output = _manual_population_core_unroll(
        model.core_network,
        encoded,
        seq_lengths=seq_lengths,
        timestep_context=timestep_context,
    )
    return _decode_and_scale_classifier(model, core_output)


def manual_population_classifier_forward(
    model: PopulationClassificationModel,
    inputs: torch.Tensor,
    *,
    seq_lengths: torch.Tensor | None = None,
    targets: Sequence[PathwayInterventionTarget] = (),
    phase_mask: torch.Tensor | None = None,
    mode: str | None = None,
    shuffle_seed: int = 0,
    fixed_mean_calibration: FixedMeanCalibration | None = None,
) -> ManualCausalForwardResult:
    """Run an exact manual classifier unroll, optionally with an intervention."""

    core = _validate_supported_model(model)
    validated_phase = _validate_sequence_inputs(inputs, phase_mask, seq_lengths)
    resolved = resolve_pathway_targets(core, targets)
    if not resolved:
        if (
            mode is not None
            or validated_phase is not None
            or fixed_mean_calibration is not None
        ):
            raise ValueError(
                "mode/phase_mask must be omitted when no intervention targets are set"
            )
    else:
        if mode is None or validated_phase is None:
            raise ValueError("targeted interventions require mode and phase_mask")
        if mode != "fixed_mean" and fixed_mean_calibration is not None:
            raise ValueError(
                "fixed_mean_calibration may only be supplied for fixed_mean mode"
            )

    with _preserve_analysis_model_state(model), torch.no_grad():
        if not resolved:
            context = _TimestepPhaseContext()
            logits = _manual_classifier_logits(
                model,
                inputs,
                seq_lengths=seq_lengths,
                timestep_context=context,
            )
            audit = InterventionAudit((), {}, {}, {})
        else:
            assert validated_phase is not None and mode is not None
            controller = _InterventionController(
                resolved_targets=resolved,
                phase_mask=validated_phase,
                mode=mode,
                shuffle_seed=shuffle_seed,
                calibration=fixed_mean_calibration,
            )
            with controller.installed() as context:
                logits = _manual_classifier_logits(
                    model,
                    inputs,
                    seq_lengths=seq_lengths,
                    timestep_context=context,
                )
            audit = controller.audit()
    return ManualCausalForwardResult(logits=logits.detach(), audit=audit)


def calibrate_fixed_mean_pathways(
    model: PopulationClassificationModel,
    reference_batches: Iterable[CausalInterventionBatch],
    *,
    targets: Sequence[PathwayInterventionTarget],
    source_id: str,
) -> FixedMeanCalibration:
    """Calibrate pathway means only from explicit, phase-labelled references."""

    core = _validate_supported_model(model)
    resolved = resolve_pathway_targets(core, targets)
    if not resolved:
        raise ValueError("fixed-mean calibration requires pathway targets")
    sums: dict[str, torch.Tensor | None] = {target.key: None for target in resolved}
    counts = {target.key: 0 for target in resolved}
    context = _TimestepPhaseContext()

    def collector(target: ResolvedPathwayTarget):
        def hook(_module, _inputs, output):
            if not isinstance(output, torch.Tensor) or output.ndim != 2:
                raise TypeError(
                    f"calibration target {target.key} must produce a 2D tensor"
                )
            active = context.current_mask
            if active is None:
                raise RuntimeError("calibration hook fired outside manual unroll")
            selected = output[active]
            if selected.numel() == 0:
                return None
            batch_sum = selected.detach().to(dtype=torch.float64).sum(dim=0).cpu()
            previous = sums[target.key]
            sums[target.key] = batch_sum if previous is None else previous + batch_sum
            counts[target.key] += int(selected.shape[0])
            return None

        return hook

    handles = []
    saw_batch = False
    try:
        for target in resolved:
            handles.append(target.module.register_forward_hook(collector(target)))
        with _preserve_analysis_model_state(model), torch.no_grad():
            for batch in reference_batches:
                saw_batch = True
                if not isinstance(batch, CausalInterventionBatch):
                    raise TypeError(
                        "reference_batches must contain CausalInterventionBatch objects"
                    )
                phase_mask = _validate_sequence_inputs(
                    batch.inputs,
                    batch.phase_mask,
                    batch.seq_lengths,
                )
                assert phase_mask is not None
                context.reset(phase_mask)
                _manual_classifier_logits(
                    model,
                    batch.inputs,
                    seq_lengths=batch.seq_lengths,
                    timestep_context=context,
                )
    finally:
        context.clear()
        for handle in reversed(handles):
            handle.remove()
    if not saw_batch:
        raise ValueError("reference_batches must not be empty")

    references = {}
    for target in resolved:
        count = counts[target.key]
        total = sums[target.key]
        if count <= 0 or total is None:
            raise ValueError(
                f"reference phase selected no observations for {target.key}"
            )
        references[target.key] = total / count
    return FixedMeanCalibration(
        references=references,
        counts=counts,
        source_id=source_id,
    )


def evaluate_same_checkpoint_intervention(
    model: PopulationClassificationModel,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    *,
    targets: Sequence[PathwayInterventionTarget],
    phase_mask: torch.Tensor,
    mode: InterventionMode,
    seq_lengths: torch.Tensor | None = None,
    shuffle_seed: int = 0,
    fixed_mean_calibration: FixedMeanCalibration | None = None,
) -> SameCheckpointInterventionResult:
    """Compare ordinary baseline and intervened logits from the same checkpoint."""

    _validate_supported_model(model)
    _validate_sequence_inputs(inputs, phase_mask, seq_lengths)
    if not isinstance(labels, torch.Tensor) or labels.shape != (inputs.shape[0],):
        raise ValueError("labels must have shape [batch]")
    if labels.dtype == torch.bool or labels.is_floating_point():
        raise TypeError("classification labels must use an integer dtype")
    if model.core_network.output_mode == "all":
        raise ValueError(
            "per-trial classification requires output_mode 'last' or 'mean'"
        )

    with _preserve_analysis_model_state(model), torch.no_grad():
        baseline_logits = model(inputs, seq_lengths=seq_lengths)
        intervention = manual_population_classifier_forward(
            model,
            inputs,
            seq_lengths=seq_lengths,
            targets=targets,
            phase_mask=phase_mask,
            mode=mode,
            shuffle_seed=shuffle_seed,
            fixed_mean_calibration=fixed_mean_calibration,
        )
        intervened_logits = intervention.logits
    if baseline_logits.ndim != 2 or intervened_logits.shape != baseline_logits.shape:
        raise ValueError("Classifier must return [batch, classes] logits")

    labels_on_device = labels.to(device=baseline_logits.device).long()
    baseline_predictions = baseline_logits.argmax(dim=-1)
    intervened_predictions = intervened_logits.argmax(dim=-1)
    baseline_correct = baseline_predictions.eq(labels_on_device)
    intervened_correct = intervened_predictions.eq(labels_on_device)
    per_trial_drop = baseline_correct.float() - intervened_correct.float()
    baseline_accuracy = float(baseline_correct.float().mean().cpu())
    intervened_accuracy = float(intervened_correct.float().mean().cpu())
    calibration_metadata = (
        fixed_mean_calibration.metadata()
        if mode == "fixed_mean" and fixed_mean_calibration is not None
        else None
    )
    return SameCheckpointInterventionResult(
        baseline_logits=baseline_logits.detach().cpu(),
        intervened_logits=intervened_logits.detach().cpu(),
        labels=labels.detach().cpu().long(),
        baseline_predictions=baseline_predictions.detach().cpu(),
        intervened_predictions=intervened_predictions.detach().cpu(),
        baseline_correct=baseline_correct.detach().cpu(),
        intervened_correct=intervened_correct.detach().cpu(),
        per_trial_accuracy_drop=per_trial_drop.detach().cpu(),
        baseline_accuracy=baseline_accuracy,
        intervened_accuracy=intervened_accuracy,
        accuracy_drop=baseline_accuracy - intervened_accuracy,
        mode=mode,
        audit=intervention.audit,
        calibration_metadata=calibration_metadata,
    )


__all__ = [
    "ROMO_PHASE_ORDER",
    "CausalInterventionBatch",
    "FixedMeanCalibration",
    "InterventionAudit",
    "ManualCausalForwardResult",
    "PathwayInterventionTarget",
    "PopulationClassificationModel",
    "ResolvedPathwayTarget",
    "SameCheckpointInterventionResult",
    "calibrate_fixed_mean_pathways",
    "evaluate_same_checkpoint_intervention",
    "manual_population_classifier_forward",
    "resolve_pathway_targets",
    "romo_phase_mask",
    "romo_phase_masks",
]

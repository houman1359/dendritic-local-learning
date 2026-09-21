"""Matrix-free linearization of recurrent population-network state dynamics.

The recurrent state analyzed here is the state that is actually carried from
one :class:`PopulationNetwork` timestep to the next.  In rate mode that state
contains, for every population, all five dendritic trace banks and the
population output used by delayed projections.  Keeping the output tensors is
important: differentiating only the trace banks would omit recurrent routes
that consume the previous population outputs.

This module deliberately supports only the basic rate-mode state.  Optional
typed-conductance, soma-feedback, and spiking state is rejected rather than
being silently left out of the flattened vector.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real

import torch

from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState
from dendritic_modeling.networks.architectures.recurrent.heterogeneous_leak_ctrnn import (
    HeterogeneousLeakCTRNN,
    HeterogeneousLeakCTRNNState,
)
from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationNetwork,
)
from dendritic_modeling.networks.architectures.recurrent.population_state import (
    PopulationLayerState,
)

_TRACE_FIELDS = (
    "trace_E",
    "trace_E_rec",
    "trace_I",
    "trace_I_rec",
    "trace_branch",
)
_OPTIONAL_STATE_FIELDS = (
    "branch_voltage",
    "typed_traces",
    "soma_output",
    "v_soma",
    "refractory_counter",
    "spike_readout",
    "dendritic_spike_plateau",
    "dendritic_spike_refractory",
    "dendritic_spike_events",
)


class UnsupportedPopulationStateError(ValueError):
    """Raised when flattening would omit an active recurrent state tensor."""


@dataclass(frozen=True)
class StateTensorSpec:
    """Location and shape of one tensor in the canonical state vector."""

    layer_index: int
    layer_name: str
    population_name: str
    field: str
    level_index: int | None
    shape: tuple[int, ...]
    start: int
    stop: int

    @property
    def numel(self) -> int:
        return self.stop - self.start

    @property
    def path(self) -> str:
        prefix = f"layers[{self.layer_index}].{self.layer_name}.{self.population_name}"
        if self.level_index is None:
            return f"{prefix}.{self.field}"
        return f"{prefix}.{self.field}[{self.level_index}]"


def _unsupported_fields(state: DendriNetState) -> list[str]:
    return [
        field
        for field in _OPTIONAL_STATE_FIELDS
        if getattr(state, field, None) is not None
    ]


def _validate_rate_population(population: torch.nn.Module, path: str) -> None:
    unsupported_modes = []
    if getattr(population, "spiking_soma", None) is not None:
        unsupported_modes.append("spiking_soma")
    if bool(getattr(population, "synapse_types_enabled", False)):
        unsupported_modes.append("typed_synapses")
    if bool(getattr(population, "dendritic_spikes_enabled", False)):
        unsupported_modes.append("dendritic_spikes")
    if bool(getattr(population, "soma_feedback_enabled", False)):
        unsupported_modes.append("soma_feedback")
    if unsupported_modes:
        raise UnsupportedPopulationStateError(
            f"{path} is not a supported basic rate-mode population; enabled: "
            + ", ".join(unsupported_modes)
        )


def _validate_deterministic_network(network: torch.nn.Module) -> None:
    """Reject forward-time randomness that does not define one fixed Jacobian."""

    for name, module in network.named_modules():
        selection = str(getattr(module, "selection", "")).lower()
        class_name = type(module).__name__.lower()
        noise_level = float(getattr(module, "noise_level", 0.0) or 0.0)
        stochastic = (
            "stochastic" in class_name
            or selection in {"stochastic", "rank_probabilistic"}
            or noise_level > 0.0
        )
        if stochastic:
            raise ValueError(
                "State-transition linearization requires a deterministic forward; "
                f"module {name!r} ({type(module).__name__}) is stochastic"
            )
        if getattr(module, "_supports_recurrent_weight_cache", None) is False:
            raise ValueError(
                "State-transition linearization requires a fixed forward map; "
                f"module {name!r} ({type(module).__name__}) uses a dynamic "
                "selection/sparsification policy"
            )
        if isinstance(module, torch.nn.modules.dropout._DropoutNd) and module.training:
            raise ValueError(
                "State-transition linearization requires a deterministic forward; "
                f"dropout module {name!r} is in training mode"
            )


class PopulationStateCodec:
    """Deterministically flatten and rebuild the supported recurrent state.

    The order is network layer order, population declaration order, the five
    trace-bank names in ``TRACE_FIELDS`` order, level order within a bank, and
    finally that population's delayed output.  Dictionary insertion order in a
    runtime state therefore cannot change the vector convention.
    """

    TRACE_FIELDS = _TRACE_FIELDS

    def __init__(
        self,
        network: PopulationNetwork,
        reference_state: Sequence[PopulationLayerState],
    ) -> None:
        if not isinstance(network, PopulationNetwork):
            raise TypeError(
                "PopulationStateCodec supports PopulationNetwork, got "
                f"{type(network).__name__}"
            )
        if not network.is_recurrent:
            raise ValueError("PopulationNetwork must have recurrent state")

        self.network = network
        self._population_names = tuple(
            tuple(layer.populations.keys()) for layer in network.layers
        )
        self._layer_names = tuple(layer.config.name for layer in network.layers)
        self._specs = self._build_specs(reference_state)
        self._dimension = self._specs[-1].stop if self._specs else 0
        if self._dimension == 0:
            raise ValueError("PopulationNetwork recurrent state is empty")

        reference_tensors = self._ordered_tensors(reference_state)
        self._dtype = reference_tensors[0].dtype
        self._device = reference_tensors[0].device
        self._batch_size = int(reference_tensors[0].shape[0])

    @property
    def specs(self) -> tuple[StateTensorSpec, ...]:
        return self._specs

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _validate_structure(
        self, state: Sequence[PopulationLayerState]
    ) -> list[torch.Tensor]:
        if not isinstance(state, (list, tuple)):
            raise TypeError(
                "PopulationNetwork state must be a list or tuple of "
                "PopulationLayerState objects"
            )
        if len(state) != len(self.network.layers):
            raise ValueError(
                f"expected {len(self.network.layers)} layer states, got {len(state)}"
            )

        tensors: list[torch.Tensor] = []
        for layer_index, (layer_state, population_names) in enumerate(
            zip(state, self._population_names)
        ):
            if not isinstance(layer_state, PopulationLayerState):
                raise TypeError(
                    f"state layer {layer_index} must be PopulationLayerState, got "
                    f"{type(layer_state).__name__}"
                )
            expected_names = set(population_names)
            if set(layer_state.population_states) != expected_names:
                raise ValueError(
                    f"layer {layer_index} population-state keys differ: expected "
                    f"{sorted(expected_names)}, got "
                    f"{sorted(layer_state.population_states)}"
                )
            if set(layer_state.outputs) != expected_names:
                raise ValueError(
                    f"layer {layer_index} output keys differ: expected "
                    f"{sorted(expected_names)}, got {sorted(layer_state.outputs)}"
                )

            layer = self.network.layers[layer_index]
            for population_name in population_names:
                population = layer.populations[population_name]
                path = (
                    f"layers[{layer_index}].{self._layer_names[layer_index]}."
                    f"{population_name}"
                )
                _validate_rate_population(population, path)
                population_state = layer_state.population_states[population_name]
                if not isinstance(population_state, DendriNetState):
                    raise TypeError(
                        f"{path} state must be DendriNetState, got "
                        f"{type(population_state).__name__}"
                    )
                unsupported = _unsupported_fields(population_state)
                if unsupported:
                    raise UnsupportedPopulationStateError(
                        f"{path} has unsupported optional state fields: "
                        + ", ".join(unsupported)
                    )

                n_levels: int | None = None
                for field in _TRACE_FIELDS:
                    traces = getattr(population_state, field)
                    if not isinstance(traces, list):
                        raise TypeError(f"{path}.{field} must be a list of tensors")
                    if n_levels is None:
                        n_levels = len(traces)
                    elif len(traces) != n_levels:
                        raise ValueError(
                            f"{path} trace banks do not have equal level counts"
                        )
                    if not traces:
                        raise ValueError(f"{path}.{field} must not be empty")
                    expected_level_dims = tuple(
                        int(dim) for dim in getattr(population, "_level_dims", ())
                    )
                    if expected_level_dims and len(traces) != len(expected_level_dims):
                        raise ValueError(
                            f"{path}.{field} has {len(traces)} levels, expected "
                            f"{len(expected_level_dims)}"
                        )
                    for level_index, tensor in enumerate(traces):
                        if not isinstance(tensor, torch.Tensor):
                            raise TypeError(
                                f"{path}.{field}[{level_index}] must be a tensor"
                            )
                        if expected_level_dims and (
                            tensor.ndim != 2
                            or tensor.shape[1] != expected_level_dims[level_index]
                        ):
                            raise ValueError(
                                f"{path}.{field}[{level_index}] must have shape "
                                f"[batch, {expected_level_dims[level_index]}], got "
                                f"{tuple(tensor.shape)}"
                            )
                        tensors.append(tensor)

                output = layer_state.outputs[population_name]
                if not isinstance(output, torch.Tensor):
                    raise TypeError(f"{path}.outputs must be a tensor")
                expected_output_dim = int(getattr(population, "_n_soma", 0))
                if expected_output_dim and (
                    output.ndim != 2 or output.shape[1] != expected_output_dim
                ):
                    raise ValueError(
                        f"{path}.outputs must have shape "
                        f"[batch, {expected_output_dim}], got {tuple(output.shape)}"
                    )
                tensors.append(output)

        if not tensors:
            raise ValueError("PopulationNetwork recurrent state has no tensors")
        first = tensors[0]
        if not first.is_floating_point():
            raise TypeError("recurrent state tensors must be floating point")
        if first.ndim < 1:
            raise ValueError("recurrent state tensors must include a batch dimension")
        for tensor in tensors:
            if not tensor.is_floating_point():
                raise TypeError("recurrent state tensors must be floating point")
            if tensor.device != first.device or tensor.dtype != first.dtype:
                raise ValueError(
                    "all recurrent state tensors must share one device and dtype"
                )
            if tensor.ndim < 1 or tensor.shape[0] != first.shape[0]:
                raise ValueError(
                    "all recurrent state tensors must share a batch dimension"
                )
        return tensors

    def _build_specs(
        self, state: Sequence[PopulationLayerState]
    ) -> tuple[StateTensorSpec, ...]:
        tensors = self._validate_structure(state)
        specs: list[StateTensorSpec] = []
        offset = 0
        tensor_index = 0
        for layer_index, population_names in enumerate(self._population_names):
            layer_state = state[layer_index]
            for population_name in population_names:
                population_state = layer_state.population_states[population_name]
                for field in _TRACE_FIELDS:
                    traces = getattr(population_state, field)
                    for level_index, _ in enumerate(traces):
                        tensor = tensors[tensor_index]
                        tensor_index += 1
                        stop = offset + tensor.numel()
                        specs.append(
                            StateTensorSpec(
                                layer_index=layer_index,
                                layer_name=self._layer_names[layer_index],
                                population_name=population_name,
                                field=field,
                                level_index=level_index,
                                shape=tuple(tensor.shape),
                                start=offset,
                                stop=stop,
                            )
                        )
                        offset = stop
                tensor = tensors[tensor_index]
                tensor_index += 1
                stop = offset + tensor.numel()
                specs.append(
                    StateTensorSpec(
                        layer_index=layer_index,
                        layer_name=self._layer_names[layer_index],
                        population_name=population_name,
                        field="outputs",
                        level_index=None,
                        shape=tuple(tensor.shape),
                        start=offset,
                        stop=stop,
                    )
                )
                offset = stop
        return tuple(specs)

    def _ordered_tensors(
        self, state: Sequence[PopulationLayerState]
    ) -> list[torch.Tensor]:
        tensors = self._validate_structure(state)
        if hasattr(self, "_specs"):
            if len(tensors) != len(self._specs):
                raise ValueError("recurrent state tensor count changed")
            for tensor, spec in zip(tensors, self._specs):
                if tuple(tensor.shape) != spec.shape:
                    raise ValueError(
                        f"state tensor {spec.path} shape changed: expected "
                        f"{spec.shape}, got {tuple(tensor.shape)}"
                    )
        return tensors

    def flatten(self, state: Sequence[PopulationLayerState]) -> torch.Tensor:
        """Return the canonical one-dimensional differentiable state vector."""

        tensors = self._ordered_tensors(state)
        return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0)

    def unflatten(self, vector: torch.Tensor) -> list[PopulationLayerState]:
        """Rebuild a state container from the canonical vector without detaching."""

        if not isinstance(vector, torch.Tensor):
            raise TypeError("state vector must be a tensor")
        if vector.ndim != 1 or vector.numel() != self.dimension:
            raise ValueError(
                f"state vector must have shape ({self.dimension},), got "
                f"{tuple(vector.shape)}"
            )
        if vector.device != self.device or vector.dtype != self.dtype:
            raise ValueError(
                f"state vector must use {self.device}/{self.dtype}, got "
                f"{vector.device}/{vector.dtype}"
            )

        values = {
            spec.path: vector[spec.start : spec.stop].view(spec.shape)
            for spec in self.specs
        }
        rebuilt: list[PopulationLayerState] = []
        for layer_index, population_names in enumerate(self._population_names):
            population_states: dict[str, DendriNetState] = {}
            outputs: dict[str, torch.Tensor] = {}
            for population_name in population_names:
                prefix = (
                    f"layers[{layer_index}].{self._layer_names[layer_index]}."
                    f"{population_name}"
                )
                traces_by_field: dict[str, list[torch.Tensor]] = {}
                for field in _TRACE_FIELDS:
                    matching_specs = [
                        spec
                        for spec in self.specs
                        if spec.layer_index == layer_index
                        and spec.population_name == population_name
                        and spec.field == field
                    ]
                    traces_by_field[field] = [
                        values[spec.path] for spec in matching_specs
                    ]
                population_states[population_name] = DendriNetState(
                    trace_E=traces_by_field["trace_E"],
                    trace_E_rec=traces_by_field["trace_E_rec"],
                    trace_I=traces_by_field["trace_I"],
                    trace_I_rec=traces_by_field["trace_I_rec"],
                    trace_branch=traces_by_field["trace_branch"],
                )
                outputs[population_name] = values[f"{prefix}.outputs"]
            rebuilt.append(
                PopulationLayerState(
                    population_states=population_states,
                    outputs=outputs,
                )
            )
        return rebuilt

    def metadata(self) -> list[dict[str, object]]:
        """Return a JSON-safe description of the canonical state convention."""

        return [
            {
                "path": spec.path,
                "shape": list(spec.shape),
                "start": spec.start,
                "stop": spec.stop,
            }
            for spec in self.specs
        ]


class HeterogeneousLeakStateCodec:
    """Flatten the true leaky-channel and delayed-output CTRNN state."""

    def __init__(
        self,
        network: HeterogeneousLeakCTRNN,
        reference_state: HeterogeneousLeakCTRNNState,
    ) -> None:
        if not isinstance(network, HeterogeneousLeakCTRNN):
            raise TypeError(
                "HeterogeneousLeakStateCodec supports HeterogeneousLeakCTRNN, got "
                f"{type(network).__name__}"
            )
        self.network = network
        tensors = self._validate_structure(reference_state)
        self._dtype = tensors[0].dtype
        self._device = tensors[0].device
        self._batch_size = int(tensors[0].shape[0])
        offset = 0
        specs = []
        for field, tensor in zip(("voltage", "outputs"), tensors, strict=True):
            stop = offset + tensor.numel()
            specs.append(
                StateTensorSpec(
                    layer_index=0,
                    layer_name="heterogeneous_leak_ctrnn",
                    population_name="point_channels",
                    field=field,
                    level_index=None,
                    shape=tuple(tensor.shape),
                    start=offset,
                    stop=stop,
                )
            )
            offset = stop
        self._specs = tuple(specs)
        self._dimension = offset

    @property
    def specs(self) -> tuple[StateTensorSpec, ...]:
        return self._specs

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def _validate_structure(
        self,
        state: HeterogeneousLeakCTRNNState,
    ) -> list[torch.Tensor]:
        if not isinstance(state, HeterogeneousLeakCTRNNState):
            raise TypeError(
                f"state must be HeterogeneousLeakCTRNNState, got {type(state).__name__}"
            )
        expected = (
            ("voltage", state.voltage, self.network.config.hidden_dim),
            ("outputs", state.outputs, self.network.config.n_outputs),
        )
        tensors = []
        for field, tensor, width in expected:
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                raise ValueError(f"state.{field} must be a rank-two tensor")
            if tensor.shape[-1] != width:
                raise ValueError(
                    f"state.{field} must have width {width}, got {tensor.shape[-1]}"
                )
            if not tensor.is_floating_point():
                raise TypeError(f"state.{field} must be floating point")
            tensors.append(tensor)
        first = tensors[0]
        for tensor in tensors[1:]:
            if (
                tensor.shape[0] != first.shape[0]
                or tensor.device != first.device
                or tensor.dtype != first.dtype
            ):
                raise ValueError(
                    "all recurrent state tensors must share batch, device, and dtype"
                )
        if hasattr(self, "_specs"):
            for tensor, spec in zip(tensors, self._specs, strict=True):
                if tuple(tensor.shape) != spec.shape:
                    raise ValueError(
                        f"state tensor {spec.path} shape changed: expected "
                        f"{spec.shape}, got {tuple(tensor.shape)}"
                    )
        return tensors

    def flatten(self, state: HeterogeneousLeakCTRNNState) -> torch.Tensor:
        tensors = self._validate_structure(state)
        return torch.cat([tensor.reshape(-1) for tensor in tensors])

    def unflatten(self, vector: torch.Tensor) -> HeterogeneousLeakCTRNNState:
        if not isinstance(vector, torch.Tensor):
            raise TypeError("state vector must be a tensor")
        if vector.ndim != 1 or vector.numel() != self.dimension:
            raise ValueError(
                f"state vector must have shape ({self.dimension},), got "
                f"{tuple(vector.shape)}"
            )
        if vector.device != self.device or vector.dtype != self.dtype:
            raise ValueError(
                f"state vector must use {self.device}/{self.dtype}, got "
                f"{vector.device}/{vector.dtype}"
            )
        voltage_spec, outputs_spec = self.specs
        voltage = vector[voltage_spec.start : voltage_spec.stop].view(
            voltage_spec.shape
        )
        outputs = vector[outputs_spec.start : outputs_spec.stop].view(
            outputs_spec.shape
        )
        return HeterogeneousLeakCTRNNState(voltage=voltage, outputs=outputs)

    def metadata(self) -> list[dict[str, object]]:
        return [
            {
                "path": spec.path,
                "shape": list(spec.shape),
                "start": spec.start,
                "stop": spec.stop,
            }
            for spec in self.specs
        ]


class StateTransitionLinearization:
    """Matrix-free Jacobian of ``q_t = F(q_{t-1}, x_t)`` at one point."""

    def __init__(
        self,
        network: PopulationNetwork | HeterogeneousLeakCTRNN,
        state: Sequence[PopulationLayerState] | HeterogeneousLeakCTRNNState,
        input_t: torch.Tensor,
    ) -> None:
        if isinstance(network, PopulationNetwork):
            if not isinstance(state, (list, tuple)):
                raise TypeError("PopulationNetwork state must be a list or tuple")
            codec = PopulationStateCodec(network, state)
        elif isinstance(network, HeterogeneousLeakCTRNN):
            if not isinstance(state, HeterogeneousLeakCTRNNState):
                raise TypeError(
                    "HeterogeneousLeakCTRNN state must be HeterogeneousLeakCTRNNState"
                )
            codec = HeterogeneousLeakStateCodec(network, state)
        else:
            raise TypeError(
                "StateTransitionLinearization supports PopulationNetwork or "
                "HeterogeneousLeakCTRNN, got "
                f"{type(network).__name__}"
            )
        _validate_deterministic_network(network)
        self.network = network
        self.codec = codec
        if not isinstance(input_t, torch.Tensor):
            raise TypeError("input_t must be a tensor")
        if input_t.ndim != 2:
            raise ValueError(
                "input_t must have shape [batch, input_dim], got "
                f"{tuple(input_t.shape)}"
            )
        if input_t.shape[0] != self.codec.batch_size:
            raise ValueError(
                f"input batch {input_t.shape[0]} differs from state batch "
                f"{self.codec.batch_size}"
            )
        if input_t.shape[1] != network.input_dim:
            raise ValueError(
                f"input feature dimension must be {network.input_dim}, got "
                f"{input_t.shape[1]}"
            )
        if input_t.device != self.codec.device or input_t.dtype != self.codec.dtype:
            raise ValueError(
                "input_t and recurrent state must share device and dtype; got "
                f"{input_t.device}/{input_t.dtype} and "
                f"{self.codec.device}/{self.codec.dtype}"
            )
        self.input_t = input_t.detach().clone()
        self.state_vector = self.codec.flatten(state).detach().clone()

        # Validate that the actual transition returns exactly the same supported
        # state structure before any expensive iterative analysis starts.
        with torch.no_grad():
            next_vector = self.transition(self.state_vector)
        if next_vector.shape != self.state_vector.shape:
            raise RuntimeError(
                "state transition changed the flattened state dimension: "
                f"{self.state_vector.shape} -> {next_vector.shape}"
            )
        self.next_state_vector = next_vector.detach()

    @property
    def dimension(self) -> int:
        return self.codec.dimension

    @property
    def device(self) -> torch.device:
        return self.codec.device

    @property
    def dtype(self) -> torch.dtype:
        return self.codec.dtype

    def transition(self, state_vector: torch.Tensor) -> torch.Tensor:
        """Evaluate ``F(q, x_t)`` and return the canonical next-state vector."""

        return self._transition(state_vector, self.input_t)

    def transition_from_input(self, input_t: torch.Tensor) -> torch.Tensor:
        """Evaluate ``F(q_{t-1}, x)`` at the fixed reference state."""

        self._validate_input(input_t, "input_t")
        return self._transition(self.state_vector, input_t)

    def _transition(
        self,
        state_vector: torch.Tensor,
        input_t: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the transition while retaining both differentiable arguments."""

        state = self.codec.unflatten(state_vector)
        if isinstance(self.network, HeterogeneousLeakCTRNN):
            _, next_state = self.network.step(input_t, state)
        else:
            _, next_state = self.network(
                input_t,
                hidden=state,
                return_hidden=True,
            )
        if next_state is None:
            raise RuntimeError("PopulationNetwork did not return recurrent state")
        return self.codec.flatten(next_state)

    def _validate_vector(self, vector: torch.Tensor, name: str) -> None:
        if not isinstance(vector, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if vector.shape != self.state_vector.shape:
            raise ValueError(
                f"{name} must have shape {tuple(self.state_vector.shape)}, got "
                f"{tuple(vector.shape)}"
            )
        if vector.device != self.device or vector.dtype != self.dtype:
            raise ValueError(
                f"{name} must use {self.device}/{self.dtype}, got "
                f"{vector.device}/{vector.dtype}"
            )

    def _validate_input(self, input_t: torch.Tensor, name: str) -> None:
        if not isinstance(input_t, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if input_t.shape != self.input_t.shape:
            raise ValueError(
                f"{name} must have shape {tuple(self.input_t.shape)}, got "
                f"{tuple(input_t.shape)}"
            )
        if input_t.device != self.device or input_t.dtype != self.dtype:
            raise ValueError(
                f"{name} must use {self.device}/{self.dtype}, got "
                f"{input_t.device}/{input_t.dtype}"
            )

    def jvp(self, vector: torch.Tensor, *, create_graph: bool = False) -> torch.Tensor:
        """Apply ``(d q_t / d q_{t-1})`` to ``vector`` without building it."""

        self._validate_vector(vector, "vector")
        _, product = torch.autograd.functional.jvp(
            self.transition,
            self.state_vector,
            vector,
            create_graph=create_graph,
            strict=False,
        )
        return product

    def vjp(
        self, cotangent: torch.Tensor, *, create_graph: bool = False
    ) -> torch.Tensor:
        """Apply the transpose state Jacobian to ``cotangent`` matrix-free."""

        self._validate_vector(cotangent, "cotangent")
        _, product = torch.autograd.functional.vjp(
            self.transition,
            self.state_vector,
            cotangent,
            create_graph=create_graph,
            strict=False,
        )
        return product

    def input_jvp(
        self,
        direction: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Apply ``(d q_t / d x_t)`` to an input-space direction."""

        self._validate_input(direction, "direction")
        _, product = torch.autograd.functional.jvp(
            self.transition_from_input,
            self.input_t,
            direction,
            create_graph=create_graph,
            strict=False,
        )
        return product

    def input_vjp(
        self,
        cotangent: torch.Tensor,
        *,
        create_graph: bool = False,
    ) -> torch.Tensor:
        """Apply ``(d q_t / d x_t)^T`` to a state-space cotangent."""

        self._validate_vector(cotangent, "cotangent")
        _, product = torch.autograd.functional.vjp(
            self.transition_from_input,
            self.input_t,
            cotangent,
            create_graph=create_graph,
            strict=False,
        )
        return product


@dataclass(frozen=True)
class SingularValueEstimate:
    """Result of two-sided power iteration on a matrix-free operator."""

    singular_value: float
    right_vector: torch.Tensor
    left_vector: torch.Tensor
    iterations: int
    converged: bool
    residual_norm: float

    def metadata(self) -> dict[str, object]:
        return {
            "singular_value": self.singular_value,
            "iterations": self.iterations,
            "converged": self.converged,
            "residual_norm": self.residual_norm,
            "method": "two_sided_power_iteration_with_jvp_vjp",
            "estimate_scope": "randomized_leading_singular_value_estimate",
            "stopping_criterion_scope": (
                "successive_singular_value_stability_not_global_leading_certification"
            ),
            "residual_certificate_scope": "returned_singular_triplet_only",
            "global_leading_certificate": False,
        }


@dataclass(frozen=True)
class BlockSubspaceSingularValueEstimate(SingularValueEstimate):
    """Randomized block estimate with a Ritz-triplet residual certificate.

    ``iterations`` counts block updates, while ``matvec_calls`` and
    ``rmatvec_calls`` count calls to the supplied forward and adjoint
    operators.  The latter make the numerical budget explicit even when one
    outer operator application is itself a product of many state Jacobians.
    """

    input_dimension: int
    block_size: int
    subspace_dimension: int
    maximum_subspace_dimension: int
    matvec_calls: int
    rmatvec_calls: int
    maximum_matvec_calls: int
    maximum_rmatvec_calls: int
    left_residual_norm: float
    right_residual_norm: float
    relative_residual_norm: float

    def metadata(self) -> dict[str, object]:
        full_input_space_coverage = self.subspace_dimension == self.input_dimension
        return {
            **super().metadata(),
            "method": ("block_krylov_rayleigh_ritz_with_full_reorthogonalization"),
            "convergence_criterion": (
                "ritz_singular_triplet_residual_le_absolute_plus_relative_sigma;"
                "explicit_initial_block_requires_independent_orthogonal_extension;"
                "zero_requires_full_input_space_coverage"
            ),
            "estimate_scope": (
                "exact_full_space_leading_singular_value"
                if full_input_space_coverage
                else "randomized_leading_singular_value_estimate"
            ),
            "residual_certificate_scope": "extracted_ritz_singular_triplet_only",
            "global_leading_certificate": full_input_space_coverage,
            "full_input_space_coverage": full_input_space_coverage,
            "input_dimension": self.input_dimension,
            "block_size": self.block_size,
            "subspace_dimension": self.subspace_dimension,
            "maximum_subspace_dimension": self.maximum_subspace_dimension,
            "left_residual_norm": self.left_residual_norm,
            "right_residual_norm": self.right_residual_norm,
            "relative_residual_norm": self.relative_residual_norm,
            "operator_calls": {
                "forward": self.matvec_calls,
                "adjoint": self.rmatvec_calls,
                "total": self.matvec_calls + self.rmatvec_calls,
            },
            "operator_call_budget": {
                "maximum_forward": self.maximum_matvec_calls,
                "maximum_adjoint": self.maximum_rmatvec_calls,
                "maximum_total": (
                    self.maximum_matvec_calls + self.maximum_rmatvec_calls
                ),
            },
        }


def _normalized(vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.vector_norm(vector)
    eps = torch.finfo(vector.dtype).eps
    if bool(norm <= eps):
        return torch.zeros_like(vector), norm
    return vector / norm, norm


def _power_iteration(
    *,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    jvp: Callable[[torch.Tensor], torch.Tensor],
    vjp: Callable[[torch.Tensor], torch.Tensor],
    max_iterations: int,
    relative_tolerance: float,
    absolute_tolerance: float,
    seed: int,
    initial_vector: torch.Tensor | None,
) -> SingularValueEstimate:
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("power-iteration tolerances must be non-negative")

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))

    def random_right_vector() -> torch.Tensor:
        candidate = torch.randn(
            dimension,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        normalized, norm = _normalized(candidate)
        if bool(norm == 0):  # pragma: no cover - RNG cannot practically hit this
            raise RuntimeError("could not draw a nonzero power-iteration restart")
        return normalized

    def expanded_right_vector(anchor: torch.Tensor) -> torch.Tensor:
        """Retain an explicit start while adding a seeded orthogonal component."""

        for _ in range(4):
            candidate = torch.randn(
                dimension,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            candidate = candidate - anchor * torch.dot(anchor, candidate)
            orthogonal, orthogonal_norm = _normalized(candidate)
            if bool(orthogonal_norm > 0):
                expanded, _ = _normalized(anchor + orthogonal)
                return expanded
        raise RuntimeError("could not draw an orthogonal power-iteration probe")

    explicit_initial_vector = initial_vector is not None
    if initial_vector is None:
        right = random_right_vector()
    else:
        if initial_vector.shape != (dimension,):
            raise ValueError(
                f"initial_vector must have shape ({dimension},), got "
                f"{tuple(initial_vector.shape)}"
            )
        if initial_vector.device != device or initial_vector.dtype != dtype:
            raise ValueError("initial_vector device/dtype does not match the operator")
        right = initial_vector.detach().clone()
        right, right_norm = _normalized(right)
        if bool(right_norm == 0):
            raise ValueError("initial_vector must be nonzero")

    previous_sigma: float | None = None
    converged = False
    iterations = 0
    explicit_extension_required = explicit_initial_vector and dimension > 1
    with torch.enable_grad():
        for iteration in range(1, max_iterations + 1):
            left, sigma_tensor = _normalized(jvp(right))
            iterations = iteration
            sigma = float(sigma_tensor.detach().cpu())
            if explicit_extension_required:
                # Stability inside a user-supplied invariant direction says
                # nothing about a larger singular value outside that direction.
                # Before convergence is eligible, retain the supplied direction
                # while injecting a deterministic seeded orthogonal component.
                explicit_extension_required = False
                previous_sigma = None
                if iteration < max_iterations:
                    right = expanded_right_vector(right)
                    continue
                break
            if sigma == 0.0:
                if dimension == 1:
                    return SingularValueEstimate(
                        singular_value=0.0,
                        right_vector=right.detach(),
                        left_vector=left.detach(),
                        iterations=iteration,
                        converged=True,
                        residual_norm=0.0,
                    )
                # A null result for one direction does not certify a zero
                # operator in a multidimensional input space. Restart from an
                # independent seeded direction, or return fail-closed if the
                # iteration budget cannot support another probe.
                previous_sigma = None
                if iteration < max_iterations:
                    right = random_right_vector()
                    continue
                break
            next_right, transpose_norm = _normalized(vjp(left))
            if bool(transpose_norm == 0):
                break
            right = next_right
            if previous_sigma is not None:
                threshold = absolute_tolerance + relative_tolerance * abs(sigma)
                if abs(sigma - previous_sigma) <= threshold:
                    converged = True
                    break
            previous_sigma = sigma

        left, sigma_tensor = _normalized(jvp(right))
        sigma = float(sigma_tensor.detach().cpu())
        transpose_product = vjp(left) if sigma > 0.0 else torch.zeros_like(right)
        residual = torch.linalg.vector_norm(transpose_product - sigma_tensor * right)

    return SingularValueEstimate(
        singular_value=sigma,
        right_vector=right.detach(),
        left_vector=left.detach(),
        iterations=iterations,
        converged=converged,
        residual_norm=float(residual.detach().cpu()),
    )


def _operator_block(
    block: torch.Tensor,
    operator: Callable[[torch.Tensor], torch.Tensor],
    *,
    output_dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Apply a vector-only matrix-free operator to every block column."""

    columns: list[torch.Tensor] = []
    for index in range(block.shape[1]):
        value = operator(block[:, index])
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must return a tensor")
        if value.shape != (output_dimension,):
            raise ValueError(
                f"{name} output must have shape ({output_dimension},), got "
                f"{tuple(value.shape)}"
            )
        if value.device != device or value.dtype != dtype:
            raise ValueError(f"{name} output device/dtype differs from its input")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} returned non-finite values")
        columns.append(value)
    return torch.stack(columns, dim=1)


def _orthogonal_block_extension(
    candidate: torch.Tensor,
    basis: torch.Tensor,
    *,
    block_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Return a full-rank block orthogonal to an existing Krylov basis."""

    def project(value: torch.Tensor, selected: torch.Tensor | None) -> torch.Tensor:
        for _ in range(2):
            value = value - basis @ (basis.mT @ value)
            if selected is not None and selected.shape[1] > 0:
                value = value - selected @ (selected.mT @ value)
        return value

    projected = project(candidate, None)
    left, singular_values, _right = torch.linalg.svd(
        projected,
        full_matrices=False,
    )
    scale = max(float(singular_values.max().detach().cpu()), 1.0)
    tolerance = torch.finfo(candidate.dtype).eps * max(candidate.shape) * scale
    rank = int(torch.count_nonzero(singular_values > tolerance).item())
    selected = left[:, : min(rank, block_size)]

    for _ in range(4):
        missing = block_size - selected.shape[1]
        if missing == 0:
            break
        random = torch.randn(
            candidate.shape[0],
            missing,
            device=candidate.device,
            dtype=candidate.dtype,
            generator=generator,
        )
        random = project(random, selected)
        orthogonal, triangular = torch.linalg.qr(random, mode="reduced")
        diagonal = torch.abs(torch.diagonal(triangular))
        random_scale = max(float(diagonal.max().detach().cpu()), 1.0)
        random_tolerance = (
            torch.finfo(candidate.dtype).eps * max(random.shape) * random_scale
        )
        random_rank = int(torch.count_nonzero(diagonal > random_tolerance).item())
        if random_rank > 0:
            selected = torch.cat(
                (selected, orthogonal[:, :random_rank]),
                dim=1,
            )
    if selected.shape[1] != block_size:
        raise RuntimeError("could not extend the block Krylov basis")
    selected = project(selected, None)
    selected, triangular = torch.linalg.qr(selected, mode="reduced")
    if bool(torch.any(torch.abs(torch.diagonal(triangular)) <= tolerance)):
        raise RuntimeError("block Krylov extension lost numerical rank")
    return selected


def leading_matrix_free_singular_value_block_subspace(
    *,
    input_dimension: int,
    output_dimension: int,
    device: torch.device,
    dtype: torch.dtype,
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rmatvec: Callable[[torch.Tensor], torch.Tensor],
    block_size: int | None = None,
    max_iterations: int | None = None,
    relative_residual_tolerance: float = 1e-3,
    absolute_residual_tolerance: float = 1e-7,
    seed: int = 0,
    initial_block: torch.Tensor | None = None,
) -> BlockSubspaceSingularValueEstimate:
    """Estimate a leading singular triplet without materializing the matrix.

    This builds the block Krylov space of ``A.T @ A`` with full
    reorthogonalization and performs a Rayleigh--Ritz extraction after every
    block expansion.  Retaining the full Krylov basis resolves clustered or
    degenerate leading spectra without the slow one-vector competition that
    can leave a stable singular value with a poor singular-vector residual.

    Convergence is certified directly from the two singular-triplet equations,
    not from stabilization of the singular value.  Each iteration uses exactly
    ``block_size`` forward and ``block_size`` adjoint operator calls; the result
    records both the realized and maximum call budgets.
    """

    if (
        isinstance(input_dimension, bool)
        or not isinstance(input_dimension, Integral)
        or input_dimension < 1
    ):
        raise ValueError("input_dimension must be a positive integer")
    if (
        isinstance(output_dimension, bool)
        or not isinstance(output_dimension, Integral)
        or output_dimension < 1
    ):
        raise ValueError("output_dimension must be a positive integer")
    block_was_provided = block_size is not None
    iterations_were_provided = max_iterations is not None
    if block_was_provided and (
        isinstance(block_size, bool)
        or not isinstance(block_size, Integral)
        or not 1 <= block_size <= input_dimension
    ):
        raise ValueError("block_size must be in [1, input_dimension]")
    if iterations_were_provided and (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, Integral)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    input_dimension = int(input_dimension)
    output_dimension = int(output_dimension)
    if not block_was_provided and not iterations_were_provided:
        resolved_block_size = min(8, input_dimension)
        while input_dimension % resolved_block_size:
            resolved_block_size -= 1
        resolved_max_iterations = min(
            24,
            input_dimension // resolved_block_size,
        )
    elif block_was_provided and not iterations_were_provided:
        assert block_size is not None
        resolved_block_size = int(block_size)
        resolved_max_iterations = min(24, input_dimension // resolved_block_size)
    elif not block_was_provided and iterations_were_provided:
        assert max_iterations is not None
        resolved_max_iterations = int(max_iterations)
        if resolved_max_iterations > input_dimension:
            raise ValueError(
                "max_iterations must not exceed input_dimension when block_size "
                "is omitted"
            )
        resolved_block_size = min(
            8,
            input_dimension // resolved_max_iterations,
        )
    else:
        assert block_size is not None and max_iterations is not None
        resolved_block_size = int(block_size)
        resolved_max_iterations = int(max_iterations)
    block_size = resolved_block_size
    max_iterations = resolved_max_iterations
    if block_size * max_iterations > input_dimension:
        raise ValueError("block_size * max_iterations must not exceed input_dimension")
    resolved_tolerances: dict[str, float] = {}
    for name, value in (
        ("relative_residual_tolerance", relative_residual_tolerance),
        ("absolute_residual_tolerance", absolute_residual_tolerance),
    ):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a real number")
        resolved = float(value)
        if not math.isfinite(resolved):
            raise ValueError(f"{name} must be finite")
        if resolved < 0.0:
            raise ValueError(f"{name} must be non-negative")
        resolved_tolerances[name] = resolved
    relative_residual_tolerance = resolved_tolerances["relative_residual_tolerance"]
    absolute_residual_tolerance = resolved_tolerances["absolute_residual_tolerance"]
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(
            "block matrix-free singular solver requires torch.float32 or "
            f"torch.float64, got {dtype}"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    explicit_initial_block = initial_block is not None
    if initial_block is None:
        candidate = torch.randn(
            input_dimension,
            block_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    else:
        if not isinstance(initial_block, torch.Tensor):
            raise TypeError("initial_block must be a tensor")
        if initial_block.shape != (input_dimension, block_size):
            raise ValueError(
                "initial_block must have shape "
                f"({input_dimension}, {block_size}), got {tuple(initial_block.shape)}"
            )
        if initial_block.device != device or initial_block.dtype != dtype:
            raise ValueError("initial_block device/dtype does not match the operator")
        if not bool(torch.isfinite(initial_block).all()):
            raise ValueError("initial_block must be finite")
        candidate = initial_block.detach().clone()

    right_block, triangular = torch.linalg.qr(candidate, mode="reduced")
    diagonal = torch.abs(torch.diagonal(triangular))
    rank_scale = max(float(diagonal.max().detach().cpu()), 1.0)
    rank_tolerance = (
        torch.finfo(dtype).eps * max(input_dimension, block_size) * rank_scale
    )
    if int(torch.count_nonzero(diagonal > rank_tolerance).item()) != block_size:
        raise ValueError("initial_block must have full column rank")

    maximum_calls = block_size * max_iterations
    maximum_subspace_dimension = block_size * max_iterations
    matvec_calls = 0
    rmatvec_calls = 0
    latest: BlockSubspaceSingularValueEstimate | None = None
    tiny = torch.finfo(dtype).tiny
    right_basis = right_block
    forward_basis = torch.empty(
        output_dimension,
        0,
        device=device,
        dtype=dtype,
    )
    normal_basis = torch.empty(
        input_dimension,
        0,
        device=device,
        dtype=dtype,
    )

    with torch.enable_grad():
        for iteration in range(1, max_iterations + 1):
            forward_block = _operator_block(
                right_block,
                matvec,
                output_dimension=output_dimension,
                device=device,
                dtype=dtype,
                name="matvec",
            )
            matvec_calls += block_size
            normal_block = _operator_block(
                forward_block,
                rmatvec,
                output_dimension=input_dimension,
                device=device,
                dtype=dtype,
                name="rmatvec",
            )
            rmatvec_calls += block_size

            forward_basis = torch.cat((forward_basis, forward_block), dim=1)
            normal_basis = torch.cat((normal_basis, normal_block), dim=1)

            projected = right_basis.mT @ normal_basis
            projected = 0.5 * (projected + projected.mT)
            _eigenvalues, eigenvectors = torch.linalg.eigh(projected)
            coefficient = eigenvectors[:, -1]
            right = right_basis @ coefficient
            forward = forward_basis @ coefficient
            sigma_tensor = torch.linalg.vector_norm(forward)
            sigma = float(sigma_tensor.detach().cpu())

            if sigma <= tiny:
                left = torch.zeros(
                    output_dimension,
                    device=device,
                    dtype=dtype,
                )
                # A zero Ritz value certifies only that the current subspace
                # lies in A's nullspace. It certifies A == 0 only after the
                # accumulated orthonormal basis spans the full input space.
                # Until then, continue with an independently extended block;
                # if the call budget ends first, return fail-closed.
                zero_certified = right_basis.shape[1] == input_dimension
                latest = BlockSubspaceSingularValueEstimate(
                    singular_value=0.0,
                    right_vector=right.detach(),
                    left_vector=left,
                    iterations=iteration,
                    converged=zero_certified,
                    residual_norm=0.0,
                    input_dimension=input_dimension,
                    block_size=block_size,
                    subspace_dimension=right_basis.shape[1],
                    maximum_subspace_dimension=maximum_subspace_dimension,
                    matvec_calls=matvec_calls,
                    rmatvec_calls=rmatvec_calls,
                    maximum_matvec_calls=maximum_calls,
                    maximum_rmatvec_calls=maximum_calls,
                    left_residual_norm=0.0,
                    right_residual_norm=0.0,
                    relative_residual_norm=0.0,
                )
                if zero_certified:
                    return latest
            else:
                left = forward / sigma_tensor
                adjoint = (normal_basis @ coefficient) / sigma_tensor
                left_residual = torch.linalg.vector_norm(forward - sigma_tensor * left)
                right_residual = torch.linalg.vector_norm(
                    adjoint - sigma_tensor * right
                )
                residual = torch.maximum(left_residual, right_residual)
                residual_value = float(residual.detach().cpu())
                relative_residual = residual_value / max(abs(sigma), tiny)
                threshold = absolute_residual_tolerance + (
                    relative_residual_tolerance * abs(sigma)
                )
                independent_extension_probed = (
                    not explicit_initial_block
                    or right_basis.shape[1] > block_size
                    or right_basis.shape[1] == input_dimension
                )
                converged = residual_value <= threshold and independent_extension_probed
                latest = BlockSubspaceSingularValueEstimate(
                    singular_value=sigma,
                    right_vector=right.detach(),
                    left_vector=left.detach(),
                    iterations=iteration,
                    converged=converged,
                    residual_norm=residual_value,
                    input_dimension=input_dimension,
                    block_size=block_size,
                    subspace_dimension=right_basis.shape[1],
                    maximum_subspace_dimension=maximum_subspace_dimension,
                    matvec_calls=matvec_calls,
                    rmatvec_calls=rmatvec_calls,
                    maximum_matvec_calls=maximum_calls,
                    maximum_rmatvec_calls=maximum_calls,
                    left_residual_norm=float(left_residual.detach().cpu()),
                    right_residual_norm=float(right_residual.detach().cpu()),
                    relative_residual_norm=relative_residual,
                )
                if converged:
                    return latest

            if iteration < max_iterations:
                right_block = _orthogonal_block_extension(
                    normal_block,
                    right_basis,
                    block_size=block_size,
                    generator=generator,
                )
                right_basis = torch.cat((right_basis, right_block), dim=1)

    if latest is None:  # pragma: no cover - max_iterations validation guarantees this
        raise RuntimeError("block subspace iteration did not execute")
    return latest


def leading_state_transition_singular_value(
    linearization: StateTransitionLinearization,
    *,
    max_iterations: int = 50,
    relative_tolerance: float = 1e-5,
    absolute_tolerance: float = 1e-7,
    seed: int = 0,
    initial_vector: torch.Tensor | None = None,
) -> SingularValueEstimate:
    """Estimate ``sigma_max(d q_t / d q_{t-1})`` without forming the Jacobian."""

    return _power_iteration(
        dimension=linearization.dimension,
        device=linearization.device,
        dtype=linearization.dtype,
        jvp=linearization.jvp,
        vjp=linearization.vjp,
        max_iterations=max_iterations,
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        seed=seed,
        initial_vector=initial_vector,
    )


def leading_state_transition_singular_value_block_subspace(
    linearization: StateTransitionLinearization,
    *,
    block_size: int | None = None,
    max_iterations: int | None = None,
    relative_residual_tolerance: float = 1e-3,
    absolute_residual_tolerance: float = 1e-7,
    seed: int = 0,
    initial_block: torch.Tensor | None = None,
) -> BlockSubspaceSingularValueEstimate:
    """Randomized block estimate with a state-Jacobian Ritz residual."""

    return leading_matrix_free_singular_value_block_subspace(
        input_dimension=linearization.dimension,
        output_dimension=linearization.dimension,
        device=linearization.device,
        dtype=linearization.dtype,
        matvec=linearization.jvp,
        rmatvec=linearization.vjp,
        block_size=block_size,
        max_iterations=max_iterations,
        relative_residual_tolerance=relative_residual_tolerance,
        absolute_residual_tolerance=absolute_residual_tolerance,
        seed=seed,
        initial_block=initial_block,
    )


class StateTransitionProduct:
    """Matrix-free product of one-step Jacobians along a fixed trajectory."""

    def __init__(self, linearizations: Sequence[StateTransitionLinearization]) -> None:
        if not linearizations:
            raise ValueError("StateTransitionProduct requires at least one step")
        self.linearizations = tuple(linearizations)
        first = self.linearizations[0]
        previous = first
        for linearization in self.linearizations[1:]:
            if linearization.network is not first.network:
                raise ValueError(
                    "all state-transition linearizations must reference the "
                    "same network object"
                )
            if (
                linearization.dimension != first.dimension
                or linearization.device != first.device
                or linearization.dtype != first.dtype
            ):
                raise ValueError(
                    "all state-transition linearizations must share shape/device/dtype"
                )
            if not torch.equal(
                previous.next_state_vector,
                linearization.state_vector,
            ):
                raise ValueError(
                    "state-transition linearizations must be consecutive: each "
                    "next_state_vector must exactly equal the following "
                    "state_vector"
                )
            previous = linearization

    @property
    def dimension(self) -> int:
        return self.linearizations[0].dimension

    @property
    def device(self) -> torch.device:
        return self.linearizations[0].device

    @property
    def dtype(self) -> torch.dtype:
        return self.linearizations[0].dtype

    def jvp(self, vector: torch.Tensor) -> torch.Tensor:
        product = vector
        for linearization in self.linearizations:
            product = linearization.jvp(product)
        return product

    def vjp(self, cotangent: torch.Tensor) -> torch.Tensor:
        product = cotangent
        for linearization in reversed(self.linearizations):
            product = linearization.vjp(product)
        return product

    def leading_singular_value(
        self,
        *,
        max_iterations: int = 50,
        relative_tolerance: float = 1e-5,
        absolute_tolerance: float = 1e-7,
        seed: int = 0,
        initial_vector: torch.Tensor | None = None,
    ) -> SingularValueEstimate:
        return _power_iteration(
            dimension=self.dimension,
            device=self.device,
            dtype=self.dtype,
            jvp=self.jvp,
            vjp=self.vjp,
            max_iterations=max_iterations,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            seed=seed,
            initial_vector=initial_vector,
        )

    def leading_singular_value_block_subspace(
        self,
        *,
        block_size: int | None = None,
        max_iterations: int | None = None,
        relative_residual_tolerance: float = 1e-3,
        absolute_residual_tolerance: float = 1e-7,
        seed: int = 0,
        initial_block: torch.Tensor | None = None,
    ) -> BlockSubspaceSingularValueEstimate:
        """Randomized block estimate with a finite-horizon Ritz residual."""

        return leading_matrix_free_singular_value_block_subspace(
            input_dimension=self.dimension,
            output_dimension=self.dimension,
            device=self.device,
            dtype=self.dtype,
            matvec=self.jvp,
            rmatvec=self.vjp,
            block_size=block_size,
            max_iterations=max_iterations,
            relative_residual_tolerance=relative_residual_tolerance,
            absolute_residual_tolerance=absolute_residual_tolerance,
            seed=seed,
            initial_block=initial_block,
        )


__all__ = [
    "BlockSubspaceSingularValueEstimate",
    "HeterogeneousLeakStateCodec",
    "PopulationStateCodec",
    "SingularValueEstimate",
    "StateTensorSpec",
    "StateTransitionLinearization",
    "StateTransitionProduct",
    "UnsupportedPopulationStateError",
    "leading_matrix_free_singular_value_block_subspace",
    "leading_state_transition_singular_value",
    "leading_state_transition_singular_value_block_subspace",
]

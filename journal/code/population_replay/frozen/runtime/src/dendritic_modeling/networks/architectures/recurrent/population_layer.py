"""Single-layer population-network routing and population modules."""

from __future__ import annotations

from dataclasses import replace

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    sample_probability_mask,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState
from dendritic_modeling.networks.architectures.recurrent.population_common import (
    PopulationDefinitionConfig,
    as_population_definition_config as _as_population_definition_config,
    population_config_from_definition as _population_config_from_definition,
    population_level_dims as _population_level_dims,
    population_requires_temporal_state as _population_requires_temporal_state,
)
from dendritic_modeling.networks.architectures.recurrent.population_configs import (
    PopulationLayerConfig,
    PopulationProjectionConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_constants import (
    _DELAYED,
    _EXTERNAL_INPUT_SOURCES,
    _FF_EXC,
    _FF_INH,
    _INPUT_E_SOURCE,
    _INPUT_I_SOURCE,
    _INPUT_SOURCE,
    _REC_EXC,
    _REC_INH,
    _SAME_STEP,
)
from dendritic_modeling.networks.architectures.recurrent.population_sources import (
    _as_projection_config,
    _canonical_external_source,
    _concat_or_none,
)
from dendritic_modeling.networks.architectures.recurrent.population_state import (
    PopulationLayerState,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.utils.stable_hash import stable_seed_offset


class PopulationLayer(nn.Module):
    """A directional population-network layer.

    Same-step population-to-population edges must form a DAG. Delayed edges use
    previous-timestep outputs and may be cyclic.
    """

    def __init__(
        self,
        config: PopulationLayerConfig,
        input_dim: int | dict[str, int],
        external_source_polarities: dict[str, str] | None = None,
    ):
        super().__init__()
        self.config = config
        self.external_input_dims = self._normalize_external_input_dims(input_dim)
        self.external_source_polarities = self._normalize_external_source_polarities(
            external_source_polarities or {}
        )
        self.input_dim = self.external_input_dims[_INPUT_SOURCE]
        self.recurrent = bool(config.recurrent)
        self._store_routing = bool(config.store_routing)
        self._last_routing_info: dict[str, dict] = {}
        self._last_outputs: dict[str, torch.Tensor] = {}

        self.population_definitions = [
            _as_population_definition_config(population)
            for population in config.populations
        ]
        names = [population.name for population in self.population_definitions]
        if len(set(names)) != len(names):
            raise ValueError(f"population names must be unique per layer, got {names}")
        self._population_by_name = {
            population.name: population for population in self.population_definitions
        }
        self._population_dims = {
            population.name: int(population.n_neurons)
            for population in self.population_definitions
        }
        self.excitatory_names = [
            population.name
            for population in self.population_definitions
            if population.polarity == "excitatory"
        ]
        self.inhibitory_names = [
            population.name
            for population in self.population_definitions
            if population.polarity == "inhibitory"
        ]
        if not self.excitatory_names:
            raise ValueError(
                "population-network layer requires an excitatory population"
            )

        self.readout_population = config.readout_population or self.excitatory_names[0]
        if self.readout_population not in self._population_by_name:
            raise ValueError(
                f"readout_population {self.readout_population!r} is not a population"
            )

        raw_connections = [_as_projection_config(conn) for conn in config.connections]
        self.connections = [
            self._normalize_connection(conn) for conn in raw_connections
        ]
        self._validate_connections()
        self._same_step_order = self._topological_same_step_order()

        self._sources_by_target_pathway: dict[tuple[str, str, str], list[str]] = {}
        self._connection_by_target_pathway_source: dict[
            tuple[str, str, str, str], PopulationProjectionConfig
        ] = {}
        for conn in self.connections:
            if not conn.enabled:
                continue
            key = (conn.target, conn.timing, conn.pathway)
            self._sources_by_target_pathway.setdefault(key, []).append(conn.source)
            self._connection_by_target_pathway_source[
                (conn.target, conn.timing, conn.pathway, conn.source)
            ] = conn
        for population in self.population_definitions:
            if not self._stream_sources(population.name, _SAME_STEP, _FF_EXC):
                raise ValueError(
                    f"population {population.name!r} has no same-step ff_excitatory input"
                )

        self.population_configs: dict[str, PopulationConfig] = {}
        self.populations = nn.ModuleDict()
        self._has_temporal_population_state = False
        defaults = dict(config.population_defaults)
        for idx, population in enumerate(self.population_definitions):
            pop_cfg = _population_config_from_definition(
                population, defaults, layer_idx=idx
            )
            if pop_cfg.indexed_seed is not None:
                pop_cfg = replace(
                    pop_cfg,
                    indexed_seed=(
                        int(pop_cfg.indexed_seed)
                        + stable_seed_offset(
                            "population_network",
                            config.name,
                            population.name,
                        )
                    )
                    % ((1 << 63) - 1),
                )
            if (
                pop_cfg.initialization_seed is not None
                and not pop_cfg.initialization_namespace
            ):
                pop_cfg = replace(
                    pop_cfg,
                    initialization_namespace=(
                        f"population_network.{config.name}.{population.name}"
                    ),
                )
            self.population_configs[population.name] = pop_cfg
            self._has_temporal_population_state = (
                self._has_temporal_population_state
                or _population_requires_temporal_state(pop_cfg)
            )
            connection_masks = self._build_connection_masks_for_population(
                population, pop_cfg
            )
            self.populations[population.name] = StatefulDendriNet(
                pop_config=pop_cfg,
                excitatory_input_dim=self._stream_dim(
                    population.name, _SAME_STEP, _FF_EXC
                ),
                inhibitory_input_dim=self._stream_dim(
                    population.name, _SAME_STEP, _FF_INH
                ),
                recurrent_excitatory_input_dim=self._stream_dim(
                    population.name, _DELAYED, _REC_EXC
                ),
                recurrent_inhibitory_input_dim=self._stream_dim(
                    population.name, _DELAYED, _REC_INH
                ),
                recurrent_excitatory_is_self_population=False,
                recurrent_inhibitory_is_self_population=False,
                connection_masks_by_pathway=connection_masks,
                dt=config.dt,
            )

        self.output_dim = self._population_dims[self.readout_population]
        self._is_recurrent = (
            self.recurrent
            or self._has_temporal_population_state
            or any(conn.timing == _DELAYED for conn in self.connections if conn.enabled)
        )

    @property
    def is_recurrent(self) -> bool:
        return self._is_recurrent

    @property
    def store_routing(self) -> bool:
        return self._store_routing

    @store_routing.setter
    def store_routing(self, value: bool) -> None:
        self._store_routing = bool(value)
        for population in self.populations.values():
            population.store_routing = self._store_routing

    def _normalize_connection(
        self,
        conn: PopulationProjectionConfig,
    ) -> PopulationProjectionConfig:
        source = _canonical_external_source(conn.source) or conn.source
        source_is_external = source in self.external_input_dims
        if not source_is_external and source not in self._population_by_name:
            raise ValueError(f"connection source {source!r} is not a population")
        if conn.target not in self._population_by_name:
            raise ValueError(f"connection target {conn.target!r} is not a population")
        if source in _EXTERNAL_INPUT_SOURCES and conn.timing == _DELAYED:
            raise ValueError(
                f"source={source!r} cannot use timing='delayed'; delayed edges "
                "must come from a population output"
            )

        pathway = conn.pathway
        if not pathway:
            source_polarity = self._source_polarity(source)
            if conn.timing == _SAME_STEP:
                pathway = _FF_INH if source_polarity == "inhibitory" else _FF_EXC
            else:
                pathway = _REC_INH if source_polarity == "inhibitory" else _REC_EXC
        if conn.timing == _SAME_STEP and pathway not in {_FF_EXC, _FF_INH}:
            raise ValueError(
                "same_step connections must use pathway 'ff_excitatory' "
                f"or 'ff_inhibitory', got {pathway!r}"
            )
        if conn.timing == _DELAYED and pathway not in {_REC_EXC, _REC_INH}:
            raise ValueError(
                "delayed connections must use pathway 'rec_excitatory' "
                f"or 'rec_inhibitory', got {pathway!r}"
            )

        payload = dict(conn.__dict__)
        payload["source"] = source
        payload["pathway"] = pathway
        return PopulationProjectionConfig(**payload)

    def _validate_connections(self) -> None:
        seen = set()
        for conn in self.connections:
            key = (conn.source, conn.target, conn.timing, conn.pathway)
            if key in seen:
                raise ValueError(
                    "duplicate population-network connection "
                    f"{conn.source}->{conn.target} ({conn.timing}, {conn.pathway})"
                )
            seen.add(key)
            if conn.timing == _DELAYED and not self.recurrent:
                raise ValueError(
                    "delayed population connections require layer.recurrent=true"
                )

    def _topological_same_step_order(self) -> list[str]:
        incoming = dict.fromkeys(self._population_by_name, 0)
        outgoing = {name: [] for name in self._population_by_name}
        for conn in self.connections:
            if not conn.enabled or conn.timing != _SAME_STEP:
                continue
            if conn.source in self.external_input_dims:
                continue
            outgoing[conn.source].append(conn.target)
            incoming[conn.target] += 1

        queue = [name for name in self._population_by_name if incoming[name] == 0]
        order = []
        while queue:
            name = queue.pop(0)
            order.append(name)
            for target in outgoing[name]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    queue.append(target)

        if len(order) != len(self._population_by_name):
            cyclic = [name for name, degree in incoming.items() if degree > 0]
            raise ValueError(
                "Same-step within-layer population connections must be acyclic; "
                f"cycle involves {cyclic}"
            )
        return order

    def _source_dim(self, source: str) -> int:
        if source in self.external_input_dims:
            return self.external_input_dims[source]
        return self._population_dims[source]

    def _source_polarity(self, source: str) -> str:
        if source in self.external_input_dims:
            return self.external_source_polarities.get(source, "excitatory")
        return self._population_by_name[source].polarity

    def _normalize_external_input_dims(
        self,
        input_dim: int | dict[str, int],
    ) -> dict[str, int]:
        if isinstance(input_dim, dict):
            dims: dict[str, int] = {}
            for source, dim in input_dim.items():
                canonical = _canonical_external_source(source) or str(source)
                if not canonical:
                    raise ValueError("external input source names must be non-empty")
                dims[canonical] = int(dim)
            if _INPUT_SOURCE not in dims:
                raise ValueError("external input dimensions must include 'input'")
        else:
            dims = {_INPUT_SOURCE: int(input_dim)}

        for source in _EXTERNAL_INPUT_SOURCES:
            dims.setdefault(source, dims[_INPUT_SOURCE])
        for source, dim in dims.items():
            if int(dim) <= 0:
                raise ValueError(
                    f"external input dimension for {source!r} must be > 0, got {dim}"
                )
            dims[source] = int(dim)
        return dims

    def _normalize_external_source_polarities(
        self,
        external_source_polarities: dict[str, str],
    ) -> dict[str, str]:
        polarities = {
            _INPUT_SOURCE: "excitatory",
            _INPUT_E_SOURCE: "excitatory",
            _INPUT_I_SOURCE: "inhibitory",
        }
        for source, polarity in external_source_polarities.items():
            canonical = _canonical_external_source(source) or str(source)
            polarity = str(polarity).lower()
            if polarity not in {"excitatory", "inhibitory"}:
                raise ValueError(
                    "external source polarity must be 'excitatory' or "
                    f"'inhibitory', got {polarity!r} for {canonical!r}"
                )
            polarities[canonical] = polarity
        for source in self.external_input_dims:
            polarities.setdefault(source, "excitatory")
        return polarities

    def _normalize_external_inputs(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if isinstance(x, dict):
            external_inputs: dict[str, torch.Tensor] = {}
            for source, value in x.items():
                canonical = _canonical_external_source(source) or str(source)
                if canonical not in self.external_input_dims:
                    raise ValueError(
                        "external input tensors may only be provided for configured "
                        f"sources {sorted(self.external_input_dims)}, got {source!r}"
                    )
                external_inputs[canonical] = value
            if _INPUT_SOURCE in external_inputs:
                for source in _EXTERNAL_INPUT_SOURCES:
                    external_inputs.setdefault(source, external_inputs[_INPUT_SOURCE])
        else:
            external_inputs = dict.fromkeys(_EXTERNAL_INPUT_SOURCES, x)

        reference_shape = None
        for source, value in external_inputs.items():
            expected = self.external_input_dims[source]
            if value.shape[-1] != expected:
                raise ValueError(
                    "PopulationLayer expected external source "
                    f"{source!r} to have last dimension {expected}, "
                    f"got {value.shape[-1]}"
                )
            prefix_shape = tuple(value.shape[:-1])
            if reference_shape is None:
                reference_shape = prefix_shape
            elif prefix_shape != reference_shape:
                raise ValueError(
                    "PopulationLayer external sources must share batch/time "
                    f"dimensions, got {reference_shape} and {prefix_shape}"
                )
        return external_inputs

    def _stream_sources(self, target: str, timing: str, pathway: str) -> list[str]:
        return self._sources_by_target_pathway.get((target, timing, pathway), [])

    def _stream_dim(self, target: str, timing: str, pathway: str) -> int | None:
        sources = self._stream_sources(target, timing, pathway)
        if not sources:
            return None
        return sum(self._source_dim(source) for source in sources)

    def _level_dims(self, population: PopulationDefinitionConfig) -> list[int]:
        return _population_level_dims(population.n_neurons, population.branch_factors)

    def _build_connection_masks_for_population(
        self,
        population: PopulationDefinitionConfig,
        pop_cfg: PopulationConfig,
    ) -> dict[str, list[torch.Tensor]]:
        masks_by_pathway = {}
        for timing, pathway, pop_pathway in (
            (_SAME_STEP, _FF_EXC, pop_cfg.ff_excitatory_pathway),
            (_SAME_STEP, _FF_INH, pop_cfg.ff_inhibitory_pathway),
            (_DELAYED, _REC_EXC, pop_cfg.rec_excitatory_pathway),
            (_DELAYED, _REC_INH, pop_cfg.rec_inhibitory_pathway),
        ):
            masks = self._build_masks_for_stream(
                target=population.name,
                timing=timing,
                pathway=pathway,
                level_dims=self._level_dims(population),
                allow_self_recurrence=pop_cfg.allow_self_recurrence,
            )
            if masks is not None:
                masks_by_pathway[pop_pathway] = masks
        return masks_by_pathway

    def _build_masks_for_stream(
        self,
        *,
        target: str,
        timing: str,
        pathway: str,
        level_dims: list[int],
        allow_self_recurrence: bool,
    ) -> list[torch.Tensor] | None:
        sources = self._stream_sources(target, timing, pathway)
        if not sources:
            return None
        masks = [
            self._build_mask_level(
                target=target,
                timing=timing,
                pathway=pathway,
                sources=sources,
                out_dim=out_dim,
                level_idx=level_idx,
                allow_self_recurrence=allow_self_recurrence,
            )
            for level_idx, out_dim in enumerate(level_dims)
        ]
        if all(bool(mask.all()) for mask in masks):
            return None
        return masks

    def _build_mask_level(
        self,
        *,
        target: str,
        timing: str,
        pathway: str,
        sources: list[str],
        out_dim: int,
        level_idx: int,
        allow_self_recurrence: bool,
    ) -> torch.Tensor:
        blocks = []
        for source in sources:
            conn = self._connection_by_target_pathway_source[
                (target, timing, pathway, source)
            ]
            probability = self._connection_probability(conn)
            source_dim = self._source_dim(source)
            block = self._sample_connection_block(
                source=source,
                target=target,
                timing=timing,
                pathway=pathway,
                probability=probability,
                out_dim=out_dim,
                source_dim=source_dim,
                level_idx=level_idx,
                seed=conn.seed,
            )
            if not allow_self_recurrence and timing == _DELAYED and source == target:
                target_dim = self._population_dims[target]
                if out_dim % target_dim != 0 or source_dim != target_dim:
                    raise ValueError(
                        "Cannot construct no-self mask for mismatched recurrent "
                        f"shape out={out_dim}, source={source_dim}, target={target_dim}"
                    )
                repeats_per_target = out_dim // target_dim
                owner = torch.arange(out_dim, dtype=torch.long) // repeats_per_target
                block[torch.arange(out_dim), owner] = False
            blocks.append(block)
        return torch.cat(blocks, dim=1)

    def _connection_probability(self, conn: PopulationProjectionConfig) -> float:
        if not conn.enabled:
            return 0.0
        if conn.probability is None:
            return 1.0
        return float(conn.probability)

    def _sample_connection_block(
        self,
        *,
        source: str,
        target: str,
        timing: str,
        pathway: str,
        probability: float,
        out_dim: int,
        source_dim: int,
        level_idx: int,
        seed: int | None,
    ) -> torch.Tensor:
        if probability <= 0.0:
            return torch.zeros(out_dim, source_dim, dtype=torch.bool)
        if probability >= 1.0:
            return torch.ones(out_dim, source_dim, dtype=torch.bool)

        base_seed = seed if seed is not None else self.config.connection_seed
        if base_seed is None:
            base_seed = 0
        generator = torch.Generator()
        generator.manual_seed(
            (
                int(base_seed)
                + stable_seed_offset(
                    self.config.name,
                    source,
                    target,
                    timing,
                    pathway,
                    level_idx,
                    out_dim,
                    source_dim,
                )
            )
            % ((1 << 63) - 1)
        )
        return sample_probability_mask(
            out_features=out_dim,
            in_features=source_dim,
            probability=float(probability),
            generator=generator,
        )

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> PopulationLayerState:
        population_states = {
            name: population.init_state(
                batch_size=batch_size, device=device, dtype=dtype
            )
            for name, population in self.populations.items()
        }
        outputs = {
            name: torch.zeros(
                batch_size,
                self._population_dims[name],
                device=device,
                dtype=dtype,
            )
            for name in self.populations
        }
        return PopulationLayerState(
            population_states=population_states,
            outputs=outputs,
        )

    def _collect_stream(
        self,
        *,
        target: str,
        timing: str,
        pathway: str,
        external_inputs: dict[str, torch.Tensor],
        delayed_external_inputs: dict[str, torch.Tensor] | None,
        current_outputs: dict[str, torch.Tensor],
        previous_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        values = []
        source_outputs = current_outputs if timing == _SAME_STEP else previous_outputs
        for source in self._stream_sources(target, timing, pathway):
            if source in self.external_input_dims:
                source_inputs = (
                    delayed_external_inputs if timing == _DELAYED else external_inputs
                )
                if source_inputs is None:
                    source_inputs = {}
                if source not in source_inputs:
                    raise ValueError(
                        f"missing external input source {source!r} for "
                        f"population {target!r}"
                    )
                values.append(source_inputs[source])
            else:
                values.append(source_outputs[source])
        return _concat_or_none(values)

    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        state: PopulationLayerState | None = None,
        delayed_external_inputs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, PopulationLayerState | None]:
        if self.is_recurrent:
            return self.step(
                x,
                state=state,
                delayed_external_inputs=delayed_external_inputs,
            )

        external_inputs = self._normalize_external_inputs(x)
        current_outputs: dict[str, torch.Tensor] = {}
        previous_outputs: dict[str, torch.Tensor] = {}
        for name in self._same_step_order:
            population = self.populations[name]
            x_ff = self._collect_stream(
                target=name,
                timing=_SAME_STEP,
                pathway=_FF_EXC,
                external_inputs=external_inputs,
                delayed_external_inputs=None,
                current_outputs=current_outputs,
                previous_outputs=previous_outputs,
            )
            ff_inh = self._collect_stream(
                target=name,
                timing=_SAME_STEP,
                pathway=_FF_INH,
                external_inputs=external_inputs,
                delayed_external_inputs=None,
                current_outputs=current_outputs,
                previous_outputs=previous_outputs,
            )
            if x_ff is None:
                raise ValueError(
                    f"population {name!r} has no same-step ff_excitatory input"
                )
            current_outputs[name] = population(x_ff, ff_inh)

        self._last_outputs = current_outputs
        return current_outputs[self.readout_population], None

    def step(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        state: PopulationLayerState | None = None,
        delayed_external_inputs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, PopulationLayerState]:
        external_inputs = self._normalize_external_inputs(x)
        delayed_external_inputs = self._normalize_external_inputs(
            delayed_external_inputs or {}
        )
        reference_input = next(iter(external_inputs.values()))
        if state is None:
            state = self.init_state(
                batch_size=reference_input.shape[0],
                device=reference_input.device,
                dtype=reference_input.dtype,
            )

        current_outputs: dict[str, torch.Tensor] = {}
        new_population_states: dict[str, DendriNetState] = {}
        for name in self._same_step_order:
            population = self.populations[name]
            x_ff = self._collect_stream(
                target=name,
                timing=_SAME_STEP,
                pathway=_FF_EXC,
                external_inputs=external_inputs,
                delayed_external_inputs=delayed_external_inputs,
                current_outputs=current_outputs,
                previous_outputs=state.outputs,
            )
            ff_inh = self._collect_stream(
                target=name,
                timing=_SAME_STEP,
                pathway=_FF_INH,
                external_inputs=external_inputs,
                delayed_external_inputs=delayed_external_inputs,
                current_outputs=current_outputs,
                previous_outputs=state.outputs,
            )
            rec_exc = self._collect_stream(
                target=name,
                timing=_DELAYED,
                pathway=_REC_EXC,
                external_inputs=external_inputs,
                delayed_external_inputs=delayed_external_inputs,
                current_outputs=current_outputs,
                previous_outputs=state.outputs,
            )
            rec_inh = self._collect_stream(
                target=name,
                timing=_DELAYED,
                pathway=_REC_INH,
                external_inputs=external_inputs,
                delayed_external_inputs=delayed_external_inputs,
                current_outputs=current_outputs,
                previous_outputs=state.outputs,
            )
            if x_ff is None:
                raise ValueError(
                    f"population {name!r} has no same-step ff_excitatory input"
                )
            out, new_state = population.step(
                x=x_ff,
                inhibitory_input=ff_inh,
                recurrent_input=rec_exc,
                rec_inhibitory_input=rec_inh,
                state=state.population_states[name],
            )
            current_outputs[name] = out
            new_population_states[name] = new_state

        new_layer_state = PopulationLayerState(
            population_states=new_population_states,
            outputs=current_outputs,
        )
        self._last_outputs = current_outputs
        if self.store_routing:
            self._last_routing_info = {
                name: population._last_routing_info
                for name, population in self.populations.items()
            }
        return current_outputs[self.readout_population], new_layer_state


__all__ = ["PopulationLayer"]

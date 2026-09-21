"""Layer and network config builders for structured recurrent E/I networks."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.conversion import layer_value
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_layer_policy import (
    _direct_ff_inhibitory_enabled,
    _effective_layer_recurrence_enabled,
    _layer_recurrence_enabled,
    _resolve_layer_inhibitory_synapse_counts,
    _resolve_layer_recurrent_timing,
    _resolve_recurrent_input_projection_dims,
    _resolve_recurrent_store_routing,
    _resolve_recurrent_transfer_params,
    _should_build_inhibitory_population,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_population_builders import (
    _build_structured_excitatory_population_config,
    _build_structured_inhibitory_population_config,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_types import (
    _RecurrentFeatureFlags,
    _RecurrentReactivationOptions,
    _RecurrentSynapseCounts,
    _StructuredRecurrentSections,
)


def _layer_value(
    value: Any, layer_idx: int, default: Any = 0, *, allow_empty_list: bool = False
) -> Any:
    """Read scalar/list per-layer values with repeat-last semantics."""
    return layer_value(
        value,
        layer_idx,
        default,
        allow_empty_list=allow_empty_list,
    )


def _build_structured_recurrent_layer_config(
    *,
    ei_layer_config_cls: type[Any],
    population_config_cls: type[Any],
    sections: _StructuredRecurrentSections,
    counts: _RecurrentSynapseCounts,
    flags: _RecurrentFeatureFlags,
    layer_idx: int,
    n_exc: Any,
    n_inh: int,
    input_mode: int,
    transfer_inhibitory_mode: str,
    explicit_recurrent_layers: Any,
    use_shunting: bool,
    use_additive_normalization: bool,
    reactivation_options: _RecurrentReactivationOptions,
) -> Any:
    """Build one recurrent E/I layer config from structured sections."""
    architecture = sections.architecture
    connectivity = sections.connectivity
    recurrent_cfg = sections.recurrent_cfg

    # Per-layer recurrence
    rec_enabled = _layer_recurrence_enabled(
        layer_idx=layer_idx,
        explicit_recurrent_layers=explicit_recurrent_layers,
        counts=counts,
    )

    timing = _resolve_layer_recurrent_timing(
        recurrent_cfg=recurrent_cfg,
        layer_idx=layer_idx,
    )

    exc_cfg = _build_structured_excitatory_population_config(
        population_config_cls=population_config_cls,
        sections=sections,
        counts=counts,
        layer_idx=layer_idx,
        n_exc=n_exc,
        use_shunting=use_shunting,
        use_additive_normalization=use_additive_normalization,
        reactivation_options=reactivation_options,
        tau_base=timing.tau_base,
        tau_ratio=timing.tau_ratio,
    )

    inh_synapses = _resolve_layer_inhibitory_synapse_counts(
        connectivity=connectivity,
        counts=counts,
        layer_idx=layer_idx,
    )
    build_inhibitory_population = _should_build_inhibitory_population(
        n_inh=n_inh,
        synapses=inh_synapses,
    )

    inh_cfg = None
    if build_inhibitory_population:
        inh_cfg = _build_structured_inhibitory_population_config(
            excitatory_config=exc_cfg,
            architecture=architecture,
            layer_idx=layer_idx,
            n_inh=n_inh,
            synapses=inh_synapses,
        )

    rec_enabled = _effective_layer_recurrence_enabled(
        base_recurrent=rec_enabled,
        flags=flags,
        exc_cfg=exc_cfg,
        inh_cfg=inh_cfg,
    )

    direct_ff_inh = _direct_ff_inhibitory_enabled(
        input_mode=input_mode,
        transfer_inhibitory_mode=transfer_inhibitory_mode,
        layer_idx=layer_idx,
    )

    return ei_layer_config_cls(
        excitatory=exc_cfg,
        inhibitory=inh_cfg,
        recurrent=rec_enabled,
        dt=timing.dt,
        direct_ff_inhibitory_to_excitatory=direct_ff_inh,
    )


def _build_structured_recurrent_layers(
    *,
    ei_layer_config_cls: type[Any],
    population_config_cls: type[Any],
    sections: _StructuredRecurrentSections,
    counts: _RecurrentSynapseCounts,
    flags: _RecurrentFeatureFlags,
    excitatory_sizes: list[Any],
    inhibitory_sizes: list[Any],
    input_mode: int,
    transfer_inhibitory_mode: str,
    explicit_recurrent_layers: Any,
    use_shunting: bool,
    use_additive_normalization: bool,
    reactivation_options: _RecurrentReactivationOptions,
) -> list[Any]:
    """Build the per-layer recurrent E/I config sequence."""
    layers: list[Any] = []
    for layer_idx, n_exc in enumerate(excitatory_sizes):
        n_inh = int(_layer_value(inhibitory_sizes, layer_idx, default=0))
        layers.append(
            _build_structured_recurrent_layer_config(
                ei_layer_config_cls=ei_layer_config_cls,
                population_config_cls=population_config_cls,
                sections=sections,
                counts=counts,
                flags=flags,
                layer_idx=layer_idx,
                n_exc=n_exc,
                n_inh=n_inh,
                input_mode=input_mode,
                transfer_inhibitory_mode=transfer_inhibitory_mode,
                explicit_recurrent_layers=explicit_recurrent_layers,
                use_shunting=use_shunting,
                use_additive_normalization=use_additive_normalization,
                reactivation_options=reactivation_options,
            )
        )
    return layers


def _build_structured_recurrent_network_config(
    *,
    network_config_cls: type[Any],
    sections: _StructuredRecurrentSections,
    layers: list[Any],
    input_dim: int,
    use_transfer: bool,
) -> Any:
    """Assemble the recurrent network config from resolved sections."""
    transfer_params = _resolve_recurrent_transfer_params(
        transfer=sections.transfer,
        recurrent_cfg=sections.recurrent_cfg,
    )
    input_projection_dims = _resolve_recurrent_input_projection_dims(
        architecture=sections.architecture,
        recurrent_cfg=sections.recurrent_cfg,
    )
    store_routing = _resolve_recurrent_store_routing(
        implementation=sections.implementation,
        recurrent_cfg=sections.recurrent_cfg,
    )

    return network_config_cls(
        layers=layers,
        input_dim=input_dim,
        use_transfer=use_transfer,
        transfer_params=transfer_params,
        input_projection_dims=input_projection_dims,
        output_mode=str(sections.recurrent_cfg.get("output_mode", "last")),
        store_routing=store_routing,
    )


__all__ = [
    "_build_structured_recurrent_layer_config",
    "_build_structured_recurrent_layers",
    "_build_structured_recurrent_network_config",
    "_layer_value",
]

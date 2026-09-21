"""Population config builders for structured recurrent E/I networks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.networks.architectures.recurrent.structured_population_kwargs import (
    _structured_population_adaptive_init_kwargs,
    _structured_population_blocklinear_kwargs,
    _structured_population_deepst_kwargs,
    _structured_population_dendritic_spike_kwargs,
    _structured_population_dynamics_kwargs,
    _structured_population_morphology_kwargs,
    _structured_population_reactivation_kwargs,
    _structured_population_soma_feedback_kwargs,
    _structured_population_sparsity_kwargs,
    _structured_population_timing_kwargs,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_layer_policy import (
    _resolve_layer_excitatory_synapse_counts,
)
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_types import (
    _LayerPopulationSynapseCounts,
    _RecurrentReactivationOptions,
    _RecurrentSynapseCounts,
    _StructuredRecurrentSections,
)


def _structured_branch_factors(
    architecture: dict[str, Any], field_name: str
) -> list[int]:
    """Resolve structured morphology without conflating ``[]`` with missing.

    An explicit empty list represents a true one-level point population.  The
    historical ``[1]`` fallback remains only for configs that omit the field or
    set it to ``None``.
    """
    configured = architecture.get(field_name)
    return [1] if configured is None else list(configured)


def _build_structured_inhibitory_population_config(
    *,
    excitatory_config: Any,
    architecture: dict[str, Any],
    layer_idx: int,
    n_inh: int,
    synapses: _LayerPopulationSynapseCounts,
) -> Any:
    """Create inhibitory PopulationConfig by replacing pathway-specific fields."""
    replacements = {
        "n_neurons": n_inh,
        "branch_factors": _structured_branch_factors(
            architecture, "inhibitory_branch_factors"
        ),
        "ff_excitatory_synapses": synapses.ff_excitatory,
        "ff_inhibitory_synapses": synapses.ff_inhibitory,
        "rec_excitatory_synapses": synapses.rec_excitatory,
        "rec_inhibitory_synapses": synapses.rec_inhibitory,
        "structured_layer_idx": layer_idx,
        "ff_excitatory_pathway": "ei",
        "ff_inhibitory_pathway": "ii",
        "rec_excitatory_pathway": "rec_ei",
        "rec_inhibitory_pathway": "rec_ii",
    }
    if hasattr(excitatory_config, "initialization_namespace"):
        replacements["initialization_namespace"] = (
            f"structured.layer.{layer_idx}.inhibitory"
        )
    return replace(excitatory_config, **replacements)


def _build_structured_excitatory_population_config(
    *,
    population_config_cls: type[Any],
    sections: _StructuredRecurrentSections,
    counts: _RecurrentSynapseCounts,
    layer_idx: int,
    n_exc: Any,
    use_shunting: bool,
    use_additive_normalization: bool,
    reactivation_options: _RecurrentReactivationOptions,
    tau_base: float,
    tau_ratio: float,
) -> Any:
    architecture = sections.architecture
    connectivity = sections.connectivity
    sparsity = sections.sparsity
    morphology = sections.morphology
    reactivation = sections.reactivation
    blocklinear = sections.blocklinear
    implementation = sections.implementation
    recurrent_cfg = sections.recurrent_cfg
    deepst = sections.deepst
    dense_to_sparse = sections.dense_to_sparse
    indexed = sections.indexed
    dynamics = sections.dynamics
    dendritic_spikes = sections.dendritic_spikes
    soma_feedback = sections.soma_feedback
    synapses = _resolve_layer_excitatory_synapse_counts(
        connectivity=connectivity,
        counts=counts,
        layer_idx=layer_idx,
    )

    return population_config_cls(
        n_neurons=int(n_exc),
        branch_factors=_structured_branch_factors(
            architecture, "excitatory_branch_factors"
        ),
        ff_excitatory_synapses=synapses.ff_excitatory,
        ff_inhibitory_synapses=synapses.ff_inhibitory,
        rec_excitatory_synapses=synapses.rec_excitatory,
        rec_inhibitory_synapses=synapses.rec_inhibitory,
        **_structured_population_sparsity_kwargs(
            sparsity=sparsity,
            dense_to_sparse=dense_to_sparse,
            indexed=indexed,
        ),
        **_structured_population_morphology_kwargs(
            morphology=morphology,
            use_shunting=use_shunting,
            use_additive_normalization=use_additive_normalization,
        ),
        **_structured_population_reactivation_kwargs(
            reactivation=reactivation,
            reactivation_options=reactivation_options,
        ),
        **_structured_population_blocklinear_kwargs(
            blocklinear=blocklinear,
            implementation=implementation,
        ),
        **_structured_population_timing_kwargs(
            recurrent_cfg=recurrent_cfg,
            tau_base=tau_base,
            tau_ratio=tau_ratio,
        ),
        **_structured_population_deepst_kwargs(deepst),
        **_structured_population_adaptive_init_kwargs(implementation),
        structured_connectivity=to_plain_dict(sections.structured_connectivity),
        structured_layer_idx=layer_idx,
        initialization_seed=sections.initialization_seed,
        initialization_namespace=f"structured.layer.{layer_idx}.excitatory",
        ff_excitatory_pathway="ee",
        ff_inhibitory_pathway="ie",
        rec_excitatory_pathway="rec_ee",
        rec_inhibitory_pathway="rec_ie",
        synapse_types=sections.synapse_types,
        **_structured_population_dynamics_kwargs(dynamics),
        **_structured_population_dendritic_spike_kwargs(
            dendritic_spikes=dendritic_spikes,
            dynamics=dynamics,
        ),
        **_structured_population_soma_feedback_kwargs(
            soma_feedback=soma_feedback,
            dynamics=dynamics,
        ),
    )


__all__ = [
    "_build_structured_excitatory_population_config",
    "_build_structured_inhibitory_population_config",
    "_structured_branch_factors",
]

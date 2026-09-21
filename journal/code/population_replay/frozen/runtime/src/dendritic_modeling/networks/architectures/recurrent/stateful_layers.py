"""Branch-layer construction helpers for stateful recurrent populations."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_config import (
    DendriticBranchConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.spatial_morphology import (
    sample_configured_indices,
    uses_spatial_morphology,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    sample_configured_mask,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_setup import (
    _active_input_dim,
    _active_synapse_count,
    _resolve_stateful_branch_synapse_layout,
)


@dataclass(frozen=True)
class _StatefulBranchLayerBuildResult:
    """Constructed branch layers and their shared synapse configuration."""

    branch_layers: nn.ModuleList
    synapse_config: DendriticSynapseConfig


def _stateful_connection_mask(
    *,
    pop_config: PopulationConfig,
    direct_connection_masks,
    pathway: str,
    in_dim: int | None,
    out_dim: int,
    level_idx: int,
    n_levels: int,
) -> torch.Tensor | None:
    """Resolve a direct or structured connectivity mask for one pathway."""
    if in_dim is None:
        return None
    direct_masks = direct_connection_masks.get(pathway)
    if direct_masks is not None:
        if level_idx >= len(direct_masks):
            raise ValueError(
                f"connection_masks_by_pathway[{pathway!r}] must include "
                f"{n_levels} level masks"
            )
        return direct_masks[level_idx]
    return sample_configured_mask(
        pop_config.structured_connectivity,
        pathway=pathway,
        out_features=out_dim,
        in_features=in_dim,
        layer_idx=pop_config.structured_layer_idx,
        level_idx=level_idx,
        n_levels=n_levels,
    )


def _stateful_connection_indices(
    *,
    pop_config: PopulationConfig,
    direct_connection_masks,
    pathway: str,
    in_dim: int | None,
    out_dim: int,
    synapses_per_branch: int | None,
    level_idx: int,
) -> torch.Tensor | None:
    """Resolve compact spatial indices for one feedforward pathway."""
    if (
        in_dim is None
        or synapses_per_branch is None
        or synapses_per_branch <= 0
        or direct_connection_masks.get(pathway) is not None
    ):
        return None
    return sample_configured_indices(
        pop_config.structured_connectivity,
        pathway=pathway,
        out_features=out_dim,
        in_features=in_dim,
        synapses_per_branch=synapses_per_branch,
        owner_count=pop_config.n_neurons,
        branch_factors=pop_config.branch_factors,
        layer_idx=pop_config.structured_layer_idx,
        level_idx=level_idx,
        index_dtype=pop_config.indexed_index_dtype,
    )


def _reject_spatial_recurrent_pathway(
    *,
    pop_config: PopulationConfig,
    direct_connection_masks,
    pathway: str,
    in_dim: int | None,
) -> None:
    """Reject image-grid routing on active recurrent population states."""
    if in_dim is None or direct_connection_masks.get(pathway) is not None:
        return
    if uses_spatial_morphology(pop_config.structured_connectivity, pathway):
        raise ValueError(
            "spatial_morphology is supported for feedforward image pathways "
            "only; configure it under the feedforward pathway keys instead "
            f"of active recurrent pathway {pathway!r}"
        )


def _build_stateful_branch_layers(
    *,
    pop_config: PopulationConfig,
    n_levels: int,
    level_dims: list[int],
    output_owner_index_per_level: list[torch.Tensor],
    excitatory_input_dim: int,
    inhibitory_input_dim: int | None,
    recurrent_excitatory_input_dim: int | None,
    recurrent_inhibitory_input_dim: int | None,
    recurrent_excitatory_is_self_population: bool,
    recurrent_inhibitory_is_self_population: bool,
    connection_masks_by_pathway=None,
) -> _StatefulBranchLayerBuildResult:
    """Build distal-to-soma branch layers for a stateful population."""
    branch_layers = nn.ModuleList()
    synapse_layout = _resolve_stateful_branch_synapse_layout(
        pop_config,
        n_levels,
    )
    direct_connection_masks = connection_masks_by_pathway or {}
    common_synapse_config = DendriticSynapseConfig.from_population_config(pop_config)
    tangent_n0_by_depth = [
        float(value) for value in pop_config.additive_tangent_n0_by_depth
    ]
    tangent_t0_by_depth = [
        float(value) for value in pop_config.additive_tangent_t0_by_depth
    ]
    if bool(tangent_n0_by_depth) != bool(tangent_t0_by_depth):
        raise ValueError(
            "additive_tangent_n0_by_depth and additive_tangent_t0_by_depth "
            "must be provided together"
        )
    if tangent_n0_by_depth:
        if (
            pop_config.additive_tangent_n0 is not None
            or pop_config.additive_tangent_t0 is not None
        ):
            raise ValueError(
                "configure either scalar additive tangent anchors or "
                "soma-relative per-depth anchors, not both"
            )
        if len(tangent_n0_by_depth) != n_levels:
            raise ValueError(
                "soma-relative additive tangent anchors must contain exactly "
                f"{n_levels} values, found {len(tangent_n0_by_depth)}"
            )
        if not all(torch.isfinite(torch.tensor(tangent_n0_by_depth))):
            raise ValueError("additive_tangent_n0_by_depth must be finite")
        if not all(torch.isfinite(torch.tensor(tangent_t0_by_depth))):
            raise ValueError("additive_tangent_t0_by_depth must be finite")
        if any(1 + value + pop_config.epsilon <= 0 for value in tangent_t0_by_depth):
            raise ValueError(
                "every per-depth tangent denominator must satisfy "
                "1 + t0 + epsilon > 0"
            )
    is_soma_level = n_levels - 1

    _reject_spatial_recurrent_pathway(
        pop_config=pop_config,
        direct_connection_masks=direct_connection_masks,
        pathway=pop_config.rec_excitatory_pathway,
        in_dim=(
            recurrent_excitatory_input_dim
            if any(count > 0 for count in synapse_layout.rec_exc_by_level)
            else None
        ),
    )
    _reject_spatial_recurrent_pathway(
        pop_config=pop_config,
        direct_connection_masks=direct_connection_masks,
        pathway=pop_config.rec_inhibitory_pathway,
        in_dim=(
            recurrent_inhibitory_input_dim
            if any(count > 0 for count in synapse_layout.rec_inh_by_level)
            else None
        ),
    )

    for level_idx, out_dim in enumerate(level_dims):
        recurrent_self_forbidden = None
        rec_inhibitory_self_forbidden = None
        if not pop_config.allow_self_recurrence:
            owner_idx = output_owner_index_per_level[level_idx]
            if recurrent_excitatory_is_self_population:
                recurrent_self_forbidden = owner_idx
            if recurrent_inhibitory_is_self_population:
                rec_inhibitory_self_forbidden = owner_idx

        if level_idx == is_soma_level and not pop_config.somatic_synapses:
            exc_in = None
            exc_syn = None
            inh_in = None
            inh_syn = None
            rec_in = None
            rec_syn = None
            rec_inh_in = None
            rec_inh_syn = None
        else:
            ff_exc_syn = synapse_layout.ff_exc_by_level[level_idx]
            ff_inh_syn_count = synapse_layout.ff_inh_by_level[level_idx]
            rec_exc_syn_count = synapse_layout.rec_exc_by_level[level_idx]
            rec_inh_syn_count = synapse_layout.rec_inh_by_level[level_idx]

            exc_in = _active_input_dim(excitatory_input_dim, ff_exc_syn)
            exc_syn = _active_synapse_count(excitatory_input_dim, ff_exc_syn)
            inh_in = _active_input_dim(inhibitory_input_dim, ff_inh_syn_count)
            inh_syn = _active_synapse_count(inhibitory_input_dim, ff_inh_syn_count)
            rec_in = _active_input_dim(
                recurrent_excitatory_input_dim,
                rec_exc_syn_count,
            )
            rec_syn = _active_synapse_count(
                recurrent_excitatory_input_dim,
                rec_exc_syn_count,
            )
            rec_inh_in = _active_input_dim(
                recurrent_inhibitory_input_dim,
                rec_inh_syn_count,
            )
            rec_inh_syn = _active_synapse_count(
                recurrent_inhibitory_input_dim,
                rec_inh_syn_count,
            )

        synapse_config = common_synapse_config
        if tangent_n0_by_depth:
            soma_relative_depth = n_levels - level_idx - 1
            synapse_config = replace(
                common_synapse_config,
                morphology=replace(
                    common_synapse_config.morphology,
                    additive_tangent_n0=tangent_n0_by_depth[soma_relative_depth],
                    additive_tangent_t0=tangent_t0_by_depth[soma_relative_depth],
                ),
            )

        branch_layer = DendriticBranchLayer(
            branch_config=DendriticBranchConfig(
                output_dim=out_dim,
                excitatory_input_dim=exc_in,
                excitatory_synapses_per_branch=exc_syn,
                excitatory_connection_indices=_stateful_connection_indices(
                    pop_config=pop_config,
                    direct_connection_masks=direct_connection_masks,
                    pathway=pop_config.ff_excitatory_pathway,
                    in_dim=exc_in,
                    out_dim=out_dim,
                    synapses_per_branch=exc_syn,
                    level_idx=level_idx,
                ),
                excitatory_connection_mask=_stateful_connection_mask(
                    pop_config=pop_config,
                    direct_connection_masks=direct_connection_masks,
                    pathway=pop_config.ff_excitatory_pathway,
                    in_dim=exc_in,
                    out_dim=out_dim,
                    level_idx=level_idx,
                    n_levels=n_levels,
                ),
                inhibitory_input_dim=inh_in,
                inhibitory_synapses_per_branch=inh_syn,
                inhibitory_connection_indices=_stateful_connection_indices(
                    pop_config=pop_config,
                    direct_connection_masks=direct_connection_masks,
                    pathway=pop_config.ff_inhibitory_pathway,
                    in_dim=inh_in,
                    out_dim=out_dim,
                    synapses_per_branch=inh_syn,
                    level_idx=level_idx,
                ),
                inhibitory_connection_mask=_stateful_connection_mask(
                    pop_config=pop_config,
                    direct_connection_masks=direct_connection_masks,
                    pathway=pop_config.ff_inhibitory_pathway,
                    in_dim=inh_in,
                    out_dim=out_dim,
                    level_idx=level_idx,
                    n_levels=n_levels,
                ),
                input_branch_factor=synapse_layout.input_branch_factors[level_idx],
                recurrent_input_dim=rec_in,
                recurrent_synapses_per_branch=rec_syn,
                recurrent_forbidden_input_index_per_output=recurrent_self_forbidden,
                recurrent_connection_mask=_stateful_connection_mask(
                    pop_config=pop_config,
                    direct_connection_masks=direct_connection_masks,
                    pathway=pop_config.rec_excitatory_pathway,
                    in_dim=rec_in,
                    out_dim=out_dim,
                    level_idx=level_idx,
                    n_levels=n_levels,
                ),
                rec_inhibitory_input_dim=rec_inh_in,
                rec_inhibitory_synapses_per_branch=rec_inh_syn,
                rec_inhibitory_forbidden_input_index_per_output=(
                    rec_inhibitory_self_forbidden
                ),
                rec_inhibitory_connection_mask=_stateful_connection_mask(
                    pop_config=pop_config,
                    direct_connection_masks=direct_connection_masks,
                    pathway=pop_config.rec_inhibitory_pathway,
                    in_dim=rec_inh_in,
                    out_dim=out_dim,
                    level_idx=level_idx,
                    n_levels=n_levels,
                ),
                # ``level_dims`` is ordered distal-to-soma and includes the
                # soma as its final level.  Keep the shared analysis convention
                # exact: soma=0, proximal=1, increasing toward distal levels.
                layer_idx=n_levels - level_idx - 1,
            ),
            synapse_config=synapse_config,
        )
        if pop_config.topk_strategy == "conductance_dynamic":
            branch_layer.set_forward_dynamic_grad_scaling(True)
        branch_layers.append(branch_layer)

    return _StatefulBranchLayerBuildResult(
        branch_layers=branch_layers,
        synapse_config=common_synapse_config,
    )


__all__ = [
    "_StatefulBranchLayerBuildResult",
    "_build_stateful_branch_layers",
]

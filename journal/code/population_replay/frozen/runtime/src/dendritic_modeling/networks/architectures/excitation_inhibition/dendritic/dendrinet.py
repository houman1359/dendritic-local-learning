"""
DendriNet architectures for dendritic neural networks.

This module implements the DendriNet and DendriNetWithOutputs classes
which generate linear layers of neurons with sequential dendritic branches.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_config import (
    DendriticBranchConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_options import (
    resolve_dendrinet_synapse_config,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet_config import (
    DendriNetConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.spatial_morphology import (
    sample_configured_indices,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    sample_configured_mask,
)


def _sample_dendrinet_structured_mask(
    structured_connectivity,
    *,
    pathway,
    out_dim,
    in_dim,
    synapses_per_branch,
    structured_layer_idx,
    level_idx,
):
    """Sample an optional structured mask for one DendriNet pathway."""
    if in_dim is None or synapses_per_branch is None or synapses_per_branch <= 0:
        return None
    return sample_configured_mask(
        structured_connectivity,
        pathway=pathway,
        out_features=out_dim,
        in_features=in_dim,
        layer_idx=structured_layer_idx,
        level_idx=level_idx,
    )


def _sample_dendrinet_structured_masks(
    structured_connectivity,
    *,
    structured_excitatory_pathway,
    structured_inhibitory_pathway,
    out_dim,
    excitatory_input_dim,
    excitatory_synapses_per_branch,
    inhibitory_input_dim,
    inhibitory_synapses_per_branch,
    structured_layer_idx,
    level_idx,
):
    """Sample optional excitatory and inhibitory masks for one DendriNet level."""

    excitatory_connection_mask = _sample_dendrinet_structured_mask(
        structured_connectivity,
        pathway=structured_excitatory_pathway,
        out_dim=out_dim,
        in_dim=excitatory_input_dim,
        synapses_per_branch=excitatory_synapses_per_branch,
        structured_layer_idx=structured_layer_idx,
        level_idx=level_idx,
    )
    inhibitory_connection_mask = _sample_dendrinet_structured_mask(
        structured_connectivity,
        pathway=structured_inhibitory_pathway,
        out_dim=out_dim,
        in_dim=inhibitory_input_dim,
        synapses_per_branch=inhibitory_synapses_per_branch,
        structured_layer_idx=structured_layer_idx,
        level_idx=level_idx,
    )
    return excitatory_connection_mask, inhibitory_connection_mask


def _sample_dendrinet_structured_indices(
    structured_connectivity,
    *,
    structured_excitatory_pathway,
    structured_inhibitory_pathway,
    out_dim,
    excitatory_input_dim,
    excitatory_synapses_per_branch,
    inhibitory_input_dim,
    inhibitory_synapses_per_branch,
    n_soma,
    branch_factors,
    index_dtype,
    structured_layer_idx,
    level_idx,
):
    """Sample optional compact excitatory and inhibitory indices."""

    excitatory_connection_indices = None
    if (
        excitatory_input_dim is not None
        and excitatory_synapses_per_branch is not None
        and excitatory_synapses_per_branch > 0
    ):
        excitatory_connection_indices = sample_configured_indices(
            structured_connectivity,
            pathway=structured_excitatory_pathway,
            out_features=out_dim,
            in_features=excitatory_input_dim,
            synapses_per_branch=excitatory_synapses_per_branch,
            owner_count=n_soma,
            branch_factors=branch_factors,
            layer_idx=structured_layer_idx,
            level_idx=level_idx,
            index_dtype=index_dtype,
        )

    inhibitory_connection_indices = None
    if (
        inhibitory_input_dim is not None
        and inhibitory_synapses_per_branch is not None
        and inhibitory_synapses_per_branch > 0
    ):
        inhibitory_connection_indices = sample_configured_indices(
            structured_connectivity,
            pathway=structured_inhibitory_pathway,
            out_features=out_dim,
            in_features=inhibitory_input_dim,
            synapses_per_branch=inhibitory_synapses_per_branch,
            owner_count=n_soma,
            branch_factors=branch_factors,
            layer_idx=structured_layer_idx,
            level_idx=level_idx,
            index_dtype=index_dtype,
        )
    return excitatory_connection_indices, inhibitory_connection_indices


def _make_dendrinet_branch_layer(
    *,
    output_dim,
    excitatory_input_dim,
    excitatory_synapses_per_branch,
    inhibitory_input_dim,
    inhibitory_synapses_per_branch,
    input_branch_factor,
    layer_idx,
    synapse_config,
    branch_extra_kwargs,
    excitatory_connection_indices=None,
    inhibitory_connection_indices=None,
    excitatory_connection_mask=None,
    inhibitory_connection_mask=None,
):
    """Create one DendriNet branch layer with the shared synapse config."""
    return DendriticBranchLayer(
        branch_config=DendriticBranchConfig(
            output_dim=output_dim,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            input_branch_factor=input_branch_factor,
            layer_idx=layer_idx,
            excitatory_connection_indices=excitatory_connection_indices,
            inhibitory_connection_indices=inhibitory_connection_indices,
            excitatory_connection_mask=excitatory_connection_mask,
            inhibitory_connection_mask=inhibitory_connection_mask,
        ),
        synapse_config=synapse_config,
        **branch_extra_kwargs,
    )


def _plan_dendrinet_branch_layout(
    n_soma: int, branch_factors
) -> tuple[list[int], list]:
    """Return construction-order layer sizes and input branch factors."""

    layer_sizes = [n_soma]
    for branch_factor in branch_factors:
        layer_sizes.append(layer_sizes[-1] * branch_factor)
    return list(reversed(layer_sizes)), [None, *reversed(branch_factors)]


def _validate_dendrinet_dimensions(
    *,
    n_soma,
    branch_factors,
    excitatory_input_dim,
    excitatory_synapses_per_branch,
    inhibitory_input_dim,
    inhibitory_synapses_per_branch,
) -> None:
    """Validate DendriNet dimensions and synapse counts."""

    if not isinstance(n_soma, int):
        raise TypeError("n_soma must be an integer")
    if n_soma < 1:
        raise ValueError("n_soma must be >= 1")

    for bf in branch_factors:
        if not isinstance(bf, int):
            raise TypeError("branch_factors must be integers")
        if bf < 1:
            raise ValueError("branch_factors must be >= 1")

    if inhibitory_input_dim is not None and inhibitory_synapses_per_branch is not None:
        if not isinstance(inhibitory_synapses_per_branch, int):
            raise TypeError("inhibitory_synapses_per_branch must be int")
        # if inhibitory_synapses_per_branch < 1:
        #     raise ValueError("inhibitory_synapses_per_branch must be >= 1")

    if excitatory_input_dim is not None and excitatory_synapses_per_branch is not None:
        if not isinstance(excitatory_synapses_per_branch, int):
            raise TypeError("excitatory_synapses_per_branch must be int")
        if excitatory_synapses_per_branch < 1:
            raise ValueError("excitatory_synapses_per_branch must be >= 1")


def _build_dendrinet_branch_layers(
    *,
    n_soma,
    n_branch_layers,
    branch_factors,
    layer_sizes,
    input_branch_factors,
    excitatory_input_dim,
    excitatory_synapses_per_branch,
    inhibitory_input_dim,
    inhibitory_synapses_per_branch,
    somatic_synapses,
    structured_connectivity,
    structured_layer_idx,
    structured_excitatory_pathway,
    structured_inhibitory_pathway,
    synapse_config,
    branch_extra_kwargs,
):
    """Build DendriNet branch layers in forward construction order."""

    branch_layers = []
    indexed_config = getattr(synapse_config, "indexed", None)
    index_dtype = getattr(indexed_config, "index_dtype", "int64")
    soma_synapse_config = _soma_override_synapse_config(synapse_config)
    for i in range(n_branch_layers):
        (
            excitatory_connection_mask,
            inhibitory_connection_mask,
        ) = _sample_dendrinet_structured_masks(
            structured_connectivity,
            structured_excitatory_pathway=structured_excitatory_pathway,
            structured_inhibitory_pathway=structured_inhibitory_pathway,
            out_dim=layer_sizes[i],
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            structured_layer_idx=structured_layer_idx,
            level_idx=i,
        )
        (
            excitatory_connection_indices,
            inhibitory_connection_indices,
        ) = _sample_dendrinet_structured_indices(
            structured_connectivity,
            structured_excitatory_pathway=structured_excitatory_pathway,
            structured_inhibitory_pathway=structured_inhibitory_pathway,
            out_dim=layer_sizes[i],
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            n_soma=n_soma,
            branch_factors=branch_factors,
            index_dtype=index_dtype,
            structured_layer_idx=structured_layer_idx,
            level_idx=i,
        )
        lyr = _make_dendrinet_branch_layer(
            output_dim=layer_sizes[i],
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            excitatory_connection_indices=excitatory_connection_indices,
            inhibitory_connection_indices=inhibitory_connection_indices,
            excitatory_connection_mask=excitatory_connection_mask,
            inhibitory_connection_mask=inhibitory_connection_mask,
            input_branch_factor=input_branch_factors[i],
            layer_idx=n_branch_layers - i,
            synapse_config=synapse_config,
            branch_extra_kwargs=branch_extra_kwargs,
        )
        branch_layers.append(lyr)

    if somatic_synapses:
        soma_level_idx = n_branch_layers
        (
            excitatory_connection_mask,
            inhibitory_connection_mask,
        ) = _sample_dendrinet_structured_masks(
            structured_connectivity,
            structured_excitatory_pathway=structured_excitatory_pathway,
            structured_inhibitory_pathway=structured_inhibitory_pathway,
            out_dim=n_soma,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            structured_layer_idx=structured_layer_idx,
            level_idx=soma_level_idx,
        )
        (
            excitatory_connection_indices,
            inhibitory_connection_indices,
        ) = _sample_dendrinet_structured_indices(
            structured_connectivity,
            structured_excitatory_pathway=structured_excitatory_pathway,
            structured_inhibitory_pathway=structured_inhibitory_pathway,
            out_dim=n_soma,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            n_soma=n_soma,
            branch_factors=branch_factors,
            index_dtype=index_dtype,
            structured_layer_idx=structured_layer_idx,
            level_idx=soma_level_idx,
        )
        branch_layers.append(
            _make_dendrinet_branch_layer(
                output_dim=n_soma,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch,
                inhibitory_input_dim=inhibitory_input_dim,
                inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
                excitatory_connection_indices=excitatory_connection_indices,
                inhibitory_connection_indices=inhibitory_connection_indices,
                excitatory_connection_mask=excitatory_connection_mask,
                inhibitory_connection_mask=inhibitory_connection_mask,
                input_branch_factor=input_branch_factors[-1],
                layer_idx=0,
                synapse_config=soma_synapse_config,
                branch_extra_kwargs=branch_extra_kwargs,
            )
        )
    else:
        branch_layers.append(
            _make_dendrinet_branch_layer(
                output_dim=n_soma,
                excitatory_input_dim=None,
                excitatory_synapses_per_branch=None,
                inhibitory_input_dim=None,
                inhibitory_synapses_per_branch=None,
                excitatory_connection_indices=None,
                inhibitory_connection_indices=None,
                input_branch_factor=input_branch_factors[-1],
                layer_idx=0,
                synapse_config=soma_synapse_config,
                branch_extra_kwargs=branch_extra_kwargs,
            )
        )

    return branch_layers


def _soma_override_synapse_config(synapse_config):
    """Return the synapse config to use for the soma output layer.

    When ``reactivation.soma_type`` is set, the soma (``layer_idx=0``) stage
    uses that activation type with ``soma_init_m`` / ``soma_init_b`` as its
    initialization parameters, while every other branch level keeps the base
    gate. ``soma_type=None`` (the default) returns the shared config object
    unchanged, so historical construction is byte-identical.
    """

    reactivation = getattr(synapse_config, "reactivation", None)
    soma_type = getattr(reactivation, "soma_type", None)
    if not soma_type:
        return synapse_config
    from dataclasses import replace

    return replace(
        synapse_config,
        reactivation=replace(
            reactivation,
            type=str(soma_type),
            init_m=reactivation.soma_init_m,
            init_b=reactivation.soma_init_b,
            soma_type=None,
        ),
    )


class DendriNet(nn.Module):
    """
    Generates a linear layer of neurons, each of which consists of a sequential
    structure of dendritic branches and inhibitory/excitatory inputs.

    This network models the complex interactions between excitatory and
    inhibitory inputs across multiple dendritic branches, simulating dendritic
    processing in biological neurons.
    """

    def __init__(
        self,
        n_soma=None,
        branch_factors=None,
        excitatory_input_dim=None,
        excitatory_synapses_per_branch=None,
        inhibitory_input_dim=None,
        inhibitory_synapses_per_branch=None,
        reactivate=False,
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        topk_noise_level=0.0,
        topk_type="standard",  # TopK implementation strategy.
        topk_weight_norm_order=None,
        topk_gamma=1.0,
        weight_transform="exp",  # Weight transformation for positive weights
        topk_temperature=0.5,  # Stochastic TopK temperature.
        topk_ultrafast=True,  # ultrafast stochastic-topk default (set False for rank-probabilistic)
        topk_strategy="none",
        use_shunting=True,
        reactivation_type="param_tanh",
        dendritic_activation=None,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        print_hooks=False,
        reactivation_init_m=1,
        reactivation_init_b=0.5,
        reactivation_init_policy="analytical",
        reactivation_sigma_aware_k=0.25,
        dbl_init_method="analytical_expectation",
        # DeepST-compatible sparse-rewiring parameters.
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        rewire_frequency=1,
        synapses_per_branch=None,
        efficient_blocklinear: bool = False,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method=None,
        weight_threshold=1e-6,
        adaptive_initialization=True,
        adaptive_initialization_policy="preserve_shunting_center",
        adaptive_target_conductance=5.0,
        initial_child_conductance=1.0,
        structured_connectivity=None,
        structured_layer_idx: int = 0,
        structured_excitatory_pathway: str = "ee",
        structured_inhibitory_pathway: str = "ie",
        synapse_config=None,
        config=None,
        **kwargs,
    ):
        super().__init__()

        if config is not None:
            dendrinet_config = DendriNetConfig.from_config(config)
            config_kwargs = dendrinet_config.to_kwargs()
            n_soma = config_kwargs["n_soma"]
            branch_factors = config_kwargs["branch_factors"]
            excitatory_input_dim = config_kwargs["excitatory_input_dim"]
            excitatory_synapses_per_branch = config_kwargs[
                "excitatory_synapses_per_branch"
            ]
            inhibitory_input_dim = config_kwargs["inhibitory_input_dim"]
            inhibitory_synapses_per_branch = config_kwargs[
                "inhibitory_synapses_per_branch"
            ]
            somatic_synapses = config_kwargs["somatic_synapses"]
            structured_connectivity = config_kwargs["structured_connectivity"]
            structured_layer_idx = config_kwargs["structured_layer_idx"]
            structured_excitatory_pathway = config_kwargs[
                "structured_excitatory_pathway"
            ]
            structured_inhibitory_pathway = config_kwargs[
                "structured_inhibitory_pathway"
            ]
            synapse_config = dendrinet_config.synapse_config

        _validate_dendrinet_dimensions(
            n_soma=n_soma,
            branch_factors=branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
        )

        # Store branch factors for network-aware initialization
        self.branch_factors = deepcopy(branch_factors)

        self.n_branch_layers = len(branch_factors)
        layer_sizes, input_branch_factors = _plan_dendrinet_branch_layout(
            n_soma,
            branch_factors,
        )
        formal_synapse_kwargs = DendriticSynapseConfig.legacy_kwargs_from_mapping(
            locals()
        )
        (
            synapse_config,
            branch_extra_kwargs,
            self.dendritic_activation,
            reactivate,
            reactivation_type,
        ) = resolve_dendrinet_synapse_config(
            formal_synapse_kwargs=formal_synapse_kwargs,
            extra_kwargs=kwargs,
            synapse_config=synapse_config,
            branch_factors=self.branch_factors,
            dendritic_activation=dendritic_activation,
            reactivate=reactivate,
            reactivation_type=reactivation_type,
        )
        self.synapse_config = synapse_config

        branch_layers = _build_dendrinet_branch_layers(
            n_soma=n_soma,
            n_branch_layers=self.n_branch_layers,
            branch_factors=self.branch_factors,
            layer_sizes=layer_sizes,
            input_branch_factors=input_branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            somatic_synapses=somatic_synapses,
            structured_connectivity=structured_connectivity,
            structured_layer_idx=structured_layer_idx,
            structured_excitatory_pathway=structured_excitatory_pathway,
            structured_inhibitory_pathway=structured_inhibitory_pathway,
            synapse_config=synapse_config,
            branch_extra_kwargs=branch_extra_kwargs,
        )
        self.branch_layers = nn.ModuleList(branch_layers)
        self.branch_configs = tuple(layer.branch_config for layer in branch_layers)

        self.reactivate = reactivate
        self.layer_sizes = layer_sizes
        self.input_inhibitory = inhibitory_input_dim is not None
        self.n_soma = n_soma
        self.somatic_synapses = somatic_synapses
        self.efficient_blocklinear = (
            self.synapse_config.morphology.efficient_blocklinear
        )

    def apply_rewiring(self):
        for layer in self.branch_layers:
            layer.apply_rewiring()

    def decay_weights(self, weight_decay, weight_boosting=False):
        for branch_layer in self.branch_layers:
            branch_layer.decay_weights(weight_decay, weight_boosting)

    def forward(self, x, inhibitory_input=None):
        output = None
        for i in range(self.n_branch_layers + 1):
            layer = self.branch_layers[i]
            # Store excitatory and inhibitory inputs for visualization if needed
            # The excitatory input for the current layer is always x
            if hasattr(layer, "excitatory_input"):
                layer.excitatory_input = x

            # The inhibitory input is whatever was passed in or None
            if hasattr(layer, "inhibitory_input"):
                layer.inhibitory_input = inhibitory_input

            output = layer(x, inhibitory_input, output)
        return output

    def sum_weights(self, pruned=False):
        exc_total = 0
        if self.input_inhibitory:
            inh_total = 0

        n_layers = self.n_branch_layers + 1
        for i in range(n_layers):
            layer: DendriticBranchLayer = self.branch_layers[i]

            if layer.input_inhibitory and layer.input_excitatory:
                weights = layer.get_pruned_weights() if pruned else layer.get_weights()
                exc = weights["exc"]
                inh = weights["inh"]
                exc_chunk = exc.chunk(self.n_soma, dim=0)
                inh_chunk = inh.chunk(self.n_soma, dim=0)

                exc_total = exc_total + torch.stack(
                    [chunk.sum(0) for chunk in exc_chunk], dim=0
                )
                inh_total = inh_total + torch.stack(
                    [chunk.sum(0) for chunk in inh_chunk], dim=0
                )

            elif layer.input_excitatory:
                weights = layer.get_pruned_weights() if pruned else layer.get_weights()
                exc = weights["exc"]
                exc_chunk = exc.chunk(self.n_soma, dim=0)
                exc_total = exc_total + torch.stack(
                    [chunk.sum(0) for chunk in exc_chunk], dim=0
                )

        if self.input_inhibitory:
            return exc_total, inh_total
        else:
            return exc_total

    def log_sum_weights(self, pruned=False):
        if self.input_inhibitory:
            exc_total, inh_total = self.sum_weights(pruned=pruned)
            return (exc_total + 1e-8).log(), (inh_total + 1e-8).log()
        else:
            exc_total = self.sum_weights(pruned=pruned)
            return (exc_total + 1e-8).log()


__all__ = ["DendriNet", "DendriNetConfig"]

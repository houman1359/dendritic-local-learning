"""
Excitation-Inhibition Layer for dendritic networks.

This module implements the ExcitationInhibitionLayer which models
excitatory and inhibitory cell populations.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
    DendriNetConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)


def _build_common_synapse_config(
    dendrinet_extra_kwargs: dict[str, Any],
    *,
    provided_synapse_config,
    reactivate,
    reactivation_init_m,
    reactivation_init_b,
    reactivation_init_policy,
    reactivation_sigma_aware_k,
    dbl_init_method,
    topk_init_method,
    use_shunting,
    reactivation_type,
    reactivation_strategy,
    blocklinear_strategy,
    topk_noise_level,
    topk_strategy,
    topk_type,
    topk_weight_norm_order,
    topk_gamma,
    weight_transform,
    print_hooks,
    excitatory_target_density,
    inhibitory_target_density,
    use_noise,
    sigma,
    rewiring_mode,
    rewire_frequency,
    synapses_per_branch,
    freeze_excitatory_connectivity,
    freeze_inhibitory_connectivity,
    init_method,
    weight_threshold,
    adaptive_initialization,
    adaptive_initialization_policy,
    adaptive_target_conductance,
    initial_child_conductance,
) -> DendriticSynapseConfig:
    """Normalize EI-layer synapse kwargs into the shared typed config."""
    if provided_synapse_config is not None:
        return DendriticSynapseConfig.from_config(provided_synapse_config)

    return DendriticSynapseConfig.from_kwargs(
        {
            **dendrinet_extra_kwargs,
            "reactivate": reactivate,
            "reactivation_init_m": reactivation_init_m,
            "reactivation_init_b": reactivation_init_b,
            "reactivation_init_policy": reactivation_init_policy,
            "reactivation_sigma_aware_k": reactivation_sigma_aware_k,
            "dbl_init_method": dbl_init_method,
            "topk_init_method": topk_init_method,
            "use_shunting": use_shunting,
            "reactivation_type": reactivation_type,
            "reactivation_strategy": reactivation_strategy,
            "blocklinear_strategy": blocklinear_strategy,
            "topk_noise_level": topk_noise_level,
            "topk_strategy": topk_strategy,
            "topk_type": topk_type,
            "topk_weight_norm_order": topk_weight_norm_order,
            "topk_gamma": topk_gamma,
            "weight_transform": weight_transform,
            "print_hooks": print_hooks,
            "excitatory_target_density": excitatory_target_density,
            "inhibitory_target_density": inhibitory_target_density,
            "use_noise": use_noise,
            "sigma": sigma,
            "rewiring_mode": rewiring_mode,
            "rewire_frequency": rewire_frequency,
            "synapses_per_branch": synapses_per_branch,
            "freeze_excitatory_connectivity": freeze_excitatory_connectivity,
            "freeze_inhibitory_connectivity": freeze_inhibitory_connectivity,
            "init_method": init_method,
            "weight_threshold": weight_threshold,
            "adaptive_initialization": adaptive_initialization,
            "adaptive_initialization_policy": adaptive_initialization_policy,
            "adaptive_target_conductance": adaptive_target_conductance,
            "initial_child_conductance": initial_child_conductance,
        }
    )


def _structured_connectivity_enabled(structured_connectivity) -> bool:
    """Return whether structured connectivity is explicitly enabled."""

    return bool(
        isinstance(structured_connectivity, dict)
        and structured_connectivity.get("enabled", False)
    ) or bool(
        hasattr(structured_connectivity, "enabled") and structured_connectivity.enabled
    )


def _resolve_excitatory_inhibitory_input_dim(
    *,
    n_inhibitory_cells,
    build_inhibitory_cells,
    inhibitory_input_dim,
    ie_synapses_per_branch,
) -> tuple[int | None, bool]:
    """Resolve inhibitory drive into the excitatory population."""

    excitatory_inhibitory_input_dim = (
        n_inhibitory_cells
        if (build_inhibitory_cells and n_inhibitory_cells is not None)
        else inhibitory_input_dim
    )
    concat_inhibitory_to_excitatory = bool(
        excitatory_inhibitory_input_dim is not None
        and int(ie_synapses_per_branch or 0) > 0
    )
    if not concat_inhibitory_to_excitatory:
        excitatory_inhibitory_input_dim = None
    return excitatory_inhibitory_input_dim, concat_inhibitory_to_excitatory


def _build_dendrinet_cells(
    *,
    n_soma,
    branch_factors,
    excitatory_input_dim,
    excitatory_synapses_per_branch,
    inhibitory_input_dim=None,
    inhibitory_synapses_per_branch=None,
    somatic_synapses,
    synapse_config,
    dendrinet_extra_kwargs,
    structured_connectivity=None,
    structured_layer_idx=0,
    structured_excitatory_pathway="ee",
    structured_inhibitory_pathway="ie",
) -> DendriNet:
    """Build one DendriNet population from grouped construction metadata."""

    return DendriNet(
        config=DendriNetConfig(
            n_soma=n_soma,
            branch_factors=tuple(branch_factors),
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
        ),
        **dendrinet_extra_kwargs,
    )


def _build_inhibitory_cells(
    *,
    n_inhibitory_cells,
    inhibitory_branch_factors,
    excitatory_input_dim,
    ei_synapses_per_branch,
    build_inhibitory_cells,
    inhibitory_network_type,
    inhibitory_input_dim,
    ii_synapses_per_branch,
    somatic_synapses,
    synapse_mode,
    common_synapse_config,
    structured_connectivity,
    structured_layer_idx,
    dendrinet_extra_kwargs,
    mlp_inhibitory_network_params,
):
    """Build the optional inhibitory population for one EI layer."""

    if not build_inhibitory_cells or n_inhibitory_cells is None:
        return None

    # The soma reactivation override targets the propagated excitatory output
    # code; inhibitory populations always keep the base gate so the inhibitory
    # mechanism is not silently altered by an output-contract setting.
    if getattr(getattr(common_synapse_config, "reactivation", None), "soma_type", None):
        from dataclasses import replace

        common_synapse_config = replace(
            common_synapse_config,
            reactivation=replace(common_synapse_config.reactivation, soma_type=None),
        )

    assert inhibitory_network_type in ["dendritic", "mlp"], (
        "inhibitory_network_type must be either 'dendritic' or 'mlp'."
        + f"Got inhibitory_network_type: {inhibitory_network_type}"
    )
    if inhibitory_network_type == "dendritic":
        if synapse_mode == "ei":
            return _build_dendrinet_cells(
                n_soma=n_inhibitory_cells,
                branch_factors=inhibitory_branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=ei_synapses_per_branch,
                inhibitory_input_dim=inhibitory_input_dim,
                inhibitory_synapses_per_branch=ii_synapses_per_branch,
                somatic_synapses=somatic_synapses,
                structured_connectivity=structured_connectivity,
                structured_layer_idx=structured_layer_idx,
                structured_excitatory_pathway="ei",
                structured_inhibitory_pathway="ii",
                synapse_config=common_synapse_config,
                dendrinet_extra_kwargs=dendrinet_extra_kwargs,
            )

        inh_in_dim = int(excitatory_input_dim) + int(inhibitory_input_dim or 0)
        k_total = int(ei_synapses_per_branch or 0) + int(ii_synapses_per_branch or 0)
        return _build_dendrinet_cells(
            n_soma=n_inhibitory_cells,
            branch_factors=inhibitory_branch_factors,
            excitatory_input_dim=inh_in_dim,
            excitatory_synapses_per_branch=k_total,
            inhibitory_input_dim=None,
            inhibitory_synapses_per_branch=None,
            somatic_synapses=somatic_synapses,
            synapse_config=common_synapse_config,
            dendrinet_extra_kwargs=dendrinet_extra_kwargs,
        )

    inh_mlp = MLP(
        input_dim=excitatory_input_dim,
        hidden_dims=mlp_inhibitory_network_params.get("hidden_dims", []),
        activation=mlp_inhibitory_network_params.get("activation", "relu"),
        output_dim=n_inhibitory_cells,
    )
    nn.init.xavier_normal_(inh_mlp.layers[-1].weight)
    nn.init.zeros_(inh_mlp.layers[-1].bias)
    return nn.Sequential(inh_mlp, nn.Sigmoid())


def _build_excitatory_cells(
    *,
    n_excitatory_cells,
    excitatory_branch_factors,
    excitatory_input_dim,
    ee_synapses_per_branch,
    excitatory_inhibitory_input_dim,
    ie_synapses_per_branch,
    somatic_synapses,
    synapse_mode,
    common_synapse_config,
    structured_connectivity,
    structured_layer_idx,
    dendrinet_extra_kwargs,
):
    """Build the excitatory population for one EI layer."""

    if synapse_mode == "ei":
        return _build_dendrinet_cells(
            n_soma=n_excitatory_cells,
            branch_factors=excitatory_branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=ee_synapses_per_branch,
            inhibitory_input_dim=excitatory_inhibitory_input_dim,
            inhibitory_synapses_per_branch=ie_synapses_per_branch,
            structured_connectivity=structured_connectivity,
            structured_layer_idx=structured_layer_idx,
            structured_excitatory_pathway="ee",
            structured_inhibitory_pathway="ie",
            somatic_synapses=somatic_synapses,
            synapse_config=common_synapse_config,
            dendrinet_extra_kwargs=dendrinet_extra_kwargs,
        )

    exc_in_dim = int(excitatory_input_dim) + int(excitatory_inhibitory_input_dim or 0)
    k_total = int(ee_synapses_per_branch or 0) + int(ie_synapses_per_branch or 0)
    return _build_dendrinet_cells(
        n_soma=n_excitatory_cells,
        branch_factors=excitatory_branch_factors,
        excitatory_input_dim=exc_in_dim,
        excitatory_synapses_per_branch=k_total,
        inhibitory_input_dim=None,
        inhibitory_synapses_per_branch=None,
        somatic_synapses=somatic_synapses,
        synapse_config=common_synapse_config,
        dendrinet_extra_kwargs=dendrinet_extra_kwargs,
    )


class ExcitationInhibitionLayer(nn.Module):
    def __init__(
        self,
        n_excitatory_cells,
        n_inhibitory_cells,
        excitatory_branch_factors,
        inhibitory_branch_factors,
        excitatory_input_dim,
        ee_synapses_per_branch,
        ei_synapses_per_branch,
        build_inhibitory_cells=True,
        inhibitory_network_type="dendritic",
        inhibitory_input_dim=None,
        ie_synapses_per_branch=None,
        ii_synapses_per_branch=None,
        reactivate=True,
        reactivation_init_m=1.5,
        reactivation_init_b=0.5,
        reactivation_init_policy="analytical",
        reactivation_sigma_aware_k=0.25,
        dbl_init_method="analytical_expectation",
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        use_shunting=True,
        synapse_mode: str = "ei",
        reactivation_strategy="none",
        blocklinear_strategy="none",
        reactivation_type="param_tanh",
        topk_noise_level=0.0,
        topk_type="standard",
        topk_weight_norm_order=None,
        topk_gamma=1.0,
        weight_transform="exp",  # Weight transformation for positive weights
        topk_strategy="none",
        print_hooks=False,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        rewire_frequency=1,
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method="xavier_normal",
        weight_threshold=1e-6,
        adaptive_initialization=True,
        adaptive_initialization_policy="preserve_shunting_center",
        adaptive_target_conductance=5.0,
        initial_child_conductance=1.0,
        structured_connectivity=None,
        structured_layer_idx: int = 0,
        **kwargs,
    ):
        super().__init__()
        if synapse_mode not in ["ei", "mlp"]:
            raise ValueError(f"synapse_mode must be 'ei' or 'mlp', got: {synapse_mode}")
        self.synapse_mode = synapse_mode
        structured_enabled = _structured_connectivity_enabled(structured_connectivity)
        if self.synapse_mode == "mlp" and structured_enabled:
            raise ValueError(
                "Structured connectivity is only supported for synapse_mode='ei'."
            )
        dendrinet_extra_kwargs = dict(kwargs)
        provided_synapse_config = dendrinet_extra_kwargs.pop("synapse_config", None)
        common_synapse_config = _build_common_synapse_config(
            dendrinet_extra_kwargs,
            provided_synapse_config=provided_synapse_config,
            reactivate=reactivate,
            reactivation_init_m=reactivation_init_m,
            reactivation_init_b=reactivation_init_b,
            reactivation_init_policy=reactivation_init_policy,
            reactivation_sigma_aware_k=reactivation_sigma_aware_k,
            dbl_init_method=dbl_init_method,
            topk_init_method=topk_init_method,
            use_shunting=use_shunting,
            reactivation_type=reactivation_type,
            reactivation_strategy=reactivation_strategy,
            blocklinear_strategy=blocklinear_strategy,
            topk_noise_level=topk_noise_level,
            topk_strategy=topk_strategy,
            topk_type=topk_type,
            topk_weight_norm_order=topk_weight_norm_order,
            topk_gamma=topk_gamma,
            weight_transform=weight_transform,
            print_hooks=print_hooks,
            excitatory_target_density=excitatory_target_density,
            inhibitory_target_density=inhibitory_target_density,
            use_noise=use_noise,
            sigma=sigma,
            rewiring_mode=rewiring_mode,
            rewire_frequency=rewire_frequency,
            synapses_per_branch=synapses_per_branch,
            freeze_excitatory_connectivity=freeze_excitatory_connectivity,
            freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
            init_method=init_method,
            weight_threshold=weight_threshold,
            adaptive_initialization=adaptive_initialization,
            adaptive_initialization_policy=adaptive_initialization_policy,
            adaptive_target_conductance=adaptive_target_conductance,
            initial_child_conductance=initial_child_conductance,
        )

        self.inhibitory_cells = _build_inhibitory_cells(
            n_inhibitory_cells=n_inhibitory_cells,
            inhibitory_branch_factors=inhibitory_branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            ei_synapses_per_branch=ei_synapses_per_branch,
            build_inhibitory_cells=build_inhibitory_cells,
            inhibitory_network_type=inhibitory_network_type,
            inhibitory_input_dim=inhibitory_input_dim,
            ii_synapses_per_branch=ii_synapses_per_branch,
            somatic_synapses=somatic_synapses,
            synapse_mode=self.synapse_mode,
            common_synapse_config=common_synapse_config,
            structured_connectivity=structured_connectivity,
            structured_layer_idx=structured_layer_idx,
            dendrinet_extra_kwargs=dendrinet_extra_kwargs,
            mlp_inhibitory_network_params=kwargs.get(
                "mlp_inhibitory_network_params",
                {},
            ),
        )

        # Excitatory cells may receive inhibitory input from the same layer or transferred pathway.
        # In MLP-synapse mode this must be gated by IE connectivity; otherwise concatenating
        # stale inhibitory tensors can create shape mismatches across layers.
        (
            excitatory_inhibitory_input_dim,
            self.concat_inhibitory_to_excitatory,
        ) = _resolve_excitatory_inhibitory_input_dim(
            n_inhibitory_cells=n_inhibitory_cells,
            build_inhibitory_cells=build_inhibitory_cells,
            inhibitory_input_dim=inhibitory_input_dim,
            ie_synapses_per_branch=ie_synapses_per_branch,
        )
        self.excitatory_cells = _build_excitatory_cells(
            n_excitatory_cells=n_excitatory_cells,
            excitatory_branch_factors=excitatory_branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            ee_synapses_per_branch=ee_synapses_per_branch,
            excitatory_inhibitory_input_dim=excitatory_inhibitory_input_dim,
            ie_synapses_per_branch=ie_synapses_per_branch,
            somatic_synapses=somatic_synapses,
            synapse_mode=self.synapse_mode,
            common_synapse_config=common_synapse_config,
            structured_connectivity=structured_connectivity,
            structured_layer_idx=structured_layer_idx,
            dendrinet_extra_kwargs=dendrinet_extra_kwargs,
        )

        self.synapse_config = common_synapse_config
        self.excitatory_synapse_config = getattr(
            self.excitatory_cells,
            "synapse_config",
            common_synapse_config,
        )
        self.inhibitory_synapse_config = getattr(
            self.inhibitory_cells,
            "synapse_config",
            None,
        )
        self.excitatory_branch_configs = getattr(
            self.excitatory_cells,
            "branch_configs",
            (),
        )
        self.inhibitory_branch_configs = getattr(
            self.inhibitory_cells,
            "branch_configs",
            (),
        )

    def decay_weights(self, weight_decay: float, weight_boosting=False):
        if self.inhibitory_cells is not None and hasattr(
            self.inhibitory_cells, "decay_weights"
        ):
            self.inhibitory_cells.decay_weights(weight_decay, weight_boosting)
        self.excitatory_cells.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self):
        if self.inhibitory_cells is not None and hasattr(
            self.inhibitory_cells, "apply_rewiring"
        ):
            self.inhibitory_cells.apply_rewiring()
        self.excitatory_cells.apply_rewiring()

    def forward(
        self,
        excitatory_x: torch.Tensor,
        inhibitory_x: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: shape [batch, excitatory_input_dim]
        inhibitory_input: shape [batch, inhibitory_input_dim], or None
        Returns:
            excitatory_output, inhibitory_output
        """
        if self.synapse_mode == "mlp":
            # Merge inputs into one signed synapse bank.
            def _cat(a: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
                return torch.cat([a, b], dim=-1) if b is not None else a

            if isinstance(self.inhibitory_cells, DendriNet):
                inh_in = _cat(excitatory_x, inhibitory_x)
                inhibitory_x = self.inhibitory_cells(inh_in, None)
            elif isinstance(self.inhibitory_cells, nn.Sequential):
                inhibitory_x = self.inhibitory_cells(excitatory_x)

            exc_in = (
                _cat(excitatory_x, inhibitory_x)
                if self.concat_inhibitory_to_excitatory
                else excitatory_x
            )
            excitatory_x = self.excitatory_cells(exc_in, None)
        else:
            if isinstance(self.inhibitory_cells, DendriNet):
                inhibitory_x = self.inhibitory_cells(excitatory_x, inhibitory_x)
            elif isinstance(self.inhibitory_cells, nn.Sequential):
                inhibitory_x = self.inhibitory_cells(excitatory_x)

            excitatory_x = self.excitatory_cells(excitatory_x, inhibitory_x)

        return excitatory_x, inhibitory_x


__all__ = ["ExcitationInhibitionLayer"]

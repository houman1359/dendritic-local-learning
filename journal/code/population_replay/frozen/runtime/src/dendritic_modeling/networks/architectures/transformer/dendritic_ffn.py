"""Compatibility imports for transformer dendritic replacements.

Implementation lives in smaller transformer modules:
``estimates``, ``modules``, ``config_translation``, and ``patching``.
This file preserves the historical import path.
"""

from dendritic_modeling.networks.architectures.transformer.config_translation import (
    build_dendritic_ffn_kwargs_from_core_config,
    build_ei_stack_kwargs_from_core_config,
    build_population_network_ffn_kwargs_from_core_config,
)
from dendritic_modeling.networks.architectures.transformer.estimates import (
    estimate_deepseek_dense_ffn_params,
    estimate_dendritic_ffn_params,
    estimate_gated_dendritic_ffn_params,
)
from dendritic_modeling.networks.architectures.transformer.modules import (
    CollapsedPopulationNetworkSpanExit,
    DendriticFFNReplacement,
    EIStackMLPSlot,
    GatedDendriticFFNReplacement,
    GatedPopulationNetworkFFNReplacement,
    LayerwiseEIStackReplacement,
    PopulationNetworkFFNReplacement,
    TiedPopulationNetworkFFNSite,
    ZeroFFNResidualBranch,
    unwrap_collapsed_population_replacement,
    unwrap_shared_population_replacement,
)
from dendritic_modeling.networks.architectures.transformer.patching import (
    ReplacementRecord,
    apply_transformer_replacement_config,
    load_transformer_with_dendritic_replacements,
    replace_transformer_mlp_layers,
    replace_transformer_mlp_layers_with_ei_stack,
    resolve_transformer_layers,
)
from dendritic_modeling.networks.architectures.transformer.scaling import (
    TransformerParameterSpec,
    estimate_dendritic_replacement_from_config,
    profile_dendritic_lm,
    profile_dendritic_lm_config,
)

__all__ = [
    "CollapsedPopulationNetworkSpanExit",
    "DendriticFFNReplacement",
    "EIStackMLPSlot",
    "GatedDendriticFFNReplacement",
    "GatedPopulationNetworkFFNReplacement",
    "LayerwiseEIStackReplacement",
    "PopulationNetworkFFNReplacement",
    "ReplacementRecord",
    "TiedPopulationNetworkFFNSite",
    "TransformerParameterSpec",
    "ZeroFFNResidualBranch",
    "apply_transformer_replacement_config",
    "build_dendritic_ffn_kwargs_from_core_config",
    "build_ei_stack_kwargs_from_core_config",
    "build_population_network_ffn_kwargs_from_core_config",
    "estimate_deepseek_dense_ffn_params",
    "estimate_dendritic_ffn_params",
    "estimate_dendritic_replacement_from_config",
    "estimate_gated_dendritic_ffn_params",
    "load_transformer_with_dendritic_replacements",
    "profile_dendritic_lm",
    "profile_dendritic_lm_config",
    "replace_transformer_mlp_layers",
    "replace_transformer_mlp_layers_with_ei_stack",
    "resolve_transformer_layers",
    "unwrap_collapsed_population_replacement",
    "unwrap_shared_population_replacement",
]

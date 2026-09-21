import logging
import math
from collections.abc import Callable
from typing import Any, Optional, Union

import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_mapping
from dendritic_modeling.config.model_aliases import get_core_morphology_alias_overrides
from dendritic_modeling.networks.architectures.classical.autoencoder import (
    CNNAutoencoder,
    CNNVariationalAutoencoder,
    MLPAutoencoder,
    MLPVariationalAutoencoder,
)
from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNNDownsample,
    CNNUpsample,
)
from dendritic_modeling.networks.architectures.classical.identity import Identity
from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    DirectActiveMatchedPointBottleneck,
    EffectiveMatchedParamMLP,
    MatchedActiveParamMLP,
    MatchedTotalParamMLP,
    PointMatchedParamMLP,
    SparseActiveMatchedPointAffine,
    SparseStructuredMLP,
)
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ConfigurableEINetwork,
)
from dendritic_modeling.networks.architectures.factory_recurrent import (
    _BASELINE_RNN_TYPES,
    _HETEROGENEOUS_LEAK_CTRNN_TYPES,
    _LEGENDRE_MEMORY_TYPES,
    _POPULATION_NETWORK_TYPES,
    _build_baseline_rnn_architecture,
    _build_heterogeneous_leak_ctrnn_architecture,
    _build_legendre_memory_architecture,
    _build_population_network_architecture,
)
from dendritic_modeling.networks.architectures.factory_spatial import (
    _SPATIAL_DENDRITIC_CONV_TYPES,
    _SPATIAL_DENDRITIC_TYPES,
    _SPATIAL_PATCH_POINT_TYPES,
    _build_spatial_dendritic_architecture,
    _build_spatial_dendritic_conv_architecture,
    _build_spatial_patch_point_architecture,
)
from dendritic_modeling.networks.architectures.factory_unified import (
    _build_unified_ei_architecture,
)
from dendritic_modeling.networks.architectures.recurrent.structured_einet_factory import (
    _build_recurrent_einet_from_structured_config,
)
from dendritic_modeling.networks.architectures.registry import (
    build_registered_architecture,
    get_registered_architecture_names,
    reserve_architecture_names,
)

logger = logging.getLogger(__name__)

# All EINet type strings handled via ConfigurableEINetwork (+ dendritic_mlp special-cased)
_EINET_TYPES: set[str] = {
    "einet",
    "dendritic_shunting",
    "dendritic_additive",
    "dendritic_normalized_additive",
    "dendritic_mlp",
    "dendritic_signed",
    "flat_shunting",
    "flat_additive",
    "flat_normalized_additive",
    "flat_mlp",
    "flat_signed",
}

_UNIFIED_EI_TYPES: set[str] = {
    "ei_unified",
    "unified_ei",
    "ei_net",
    "unified_einet",
    # Parallel aliases that mirror the BP / feedforward vocabulary
    # (`dendritic_shunting`, `dendritic_additive`, `flat_shunting`,
    # `flat_additive`). An `rnn_` prefix disambiguates them from the
    # non-recurrent BP types. The factory below canonicalizes each alias
    # to the appropriate `use_shunting` / `use_additive_normalization`
    # pair on every configured population, leaving the `ei_unified` path
    # unchanged for backward compatibility.
    "rnn_dendritic_shunting",
    "rnn_dendritic_additive",
    "rnn_dendritic_normalized_additive",
    "rnn_flat_shunting",
    "rnn_flat_additive",
    "rnn_flat_normalized_additive",
}

_BUILTIN_ARCHITECTURE_TYPES: frozenset[str] = frozenset(
    set(_EINET_TYPES)
    | set(_UNIFIED_EI_TYPES)
    | set(_BASELINE_RNN_TYPES)
    | set(_HETEROGENEOUS_LEAK_CTRNN_TYPES)
    | set(_LEGENDRE_MEMORY_TYPES)
    | set(_POPULATION_NETWORK_TYPES)
    | set(_SPATIAL_DENDRITIC_TYPES)
    | set(_SPATIAL_DENDRITIC_CONV_TYPES)
    | set(_SPATIAL_PATCH_POINT_TYPES)
    | {
        "active_param_mlp",
        "alexnet",
        "cnn",
        "cnn_down",
        "cnn_downsample",
        "cnn_up",
        "cnn_upsample",
        "cnnautoencoder",
        "cnnvariationalautoencoder",
        "cnndownsample",
        "cnnupsample",
        "direct_active_matched_point",
        "dendritic_alexnet",
        "effectivematchedparammlp",
        "identity",
        "mlp",
        "mlpautoencoder",
        "mlpvariationalautoencoder",
        "none",
        "pathway_router",
        "point_mlp",
        "router",
        "sparse_active_matched_point",
        "ss_mlp",
        "ss_mlp_flat",
        "total_param_mlp",
    }
)

reserve_architecture_names(_BUILTIN_ARCHITECTURE_TYPES)


def get_available_architectures() -> list[str]:
    """Return built-in and extension architecture names."""
    return sorted(
        set(_BUILTIN_ARCHITECTURE_TYPES) | set(get_registered_architecture_names())
    )


ArchitectureBuilder = Callable[
    [str, Union[dict[str, Any], DictConfig], Optional[int], Optional[int]],
    nn.Module,
]


def _build_einet_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build structured feedforward/recurrent E/I architectures."""
    del suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if not ("architecture" in raw_params and "connectivity" in raw_params):
        raise ValueError(
            "EINet now requires structured config with 'architecture', "
            "'connectivity', etc. sections. Flat parameters are no longer supported."
        )
    if input_dim is None:
        raise ValueError("input_dim required when passing structured config to EINet")

    recurrent_einet = _build_recurrent_einet_from_structured_config(
        type=type, raw_params=raw_params, input_dim=input_dim
    )
    if recurrent_einet is not None:
        return recurrent_einet

    synapse_mode: Optional[str] = None
    use_shunting: Optional[bool] = None
    weight_transform: Optional[str] = None
    flatten_dendrites = False
    core_morphology_overrides = get_core_morphology_alias_overrides(type)

    if "use_shunting" in core_morphology_overrides:
        use_shunting = bool(core_morphology_overrides["use_shunting"])

    if "flat" in type:
        flatten_dendrites = True

    if "signed" in type:
        weight_transform = "identity"
        use_shunting = False

    if "mlp" in type:
        synapse_mode = "mlp"
        weight_transform = "identity"
        use_shunting = False

    params_for_build = _to_plain_mapping(parameters)
    params_for_build.setdefault("type", type)

    return ConfigurableEINetwork(
        config=params_for_build,
        input_dim=input_dim,
        synapse_mode=synapse_mode,
        use_shunting=use_shunting,
        weight_transform=weight_transform,
        flatten_dendrites=flatten_dendrites,
    )


def _build_point_mlp_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build the point-neuron MLP baseline from structured E/I config."""
    del type, suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if raw_params.get("population_network"):
        reference = _build_population_network_architecture(
            "population_network", raw_params, input_dim, None
        )
        return _build_reference_matched_mlp(
            reference=reference,
            input_dim=input_dim,
            match_mode="effective",
        )
    if not ("architecture" in raw_params and "connectivity" in raw_params):
        raise ValueError(
            "point_mlp requires structured architecture/connectivity config or "
            "a population_network reference config"
        )
    if input_dim is None:
        raise ValueError(
            "input_dim required when passing structured config to point_mlp"
        )

    from dendritic_modeling.scripts.script_utils.config_utils import (
        prepare_ei_network_params,
    )

    params = prepare_ei_network_params(raw_params, input_dim)
    return PointMatchedParamMLP(**params)


def _build_direct_active_matched_point_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build a one-affine-layer point control matched to active E/I params."""

    del type
    raw_params = _to_plain_mapping(parameters)
    if not ("architecture" in raw_params and "connectivity" in raw_params):
        raise ValueError(
            "direct_active_matched_point requires structured "
            "architecture/connectivity reference config"
        )
    if input_dim is None:
        raise ValueError("input_dim required when building direct_active_matched_point")

    reference_output_sizes = raw_params["architecture"].get(
        "excitatory_layer_sizes", []
    )
    if not reference_output_sizes:
        raise ValueError("direct_active_matched_point requires excitatory_layer_sizes")
    configured_output_dim = int(reference_output_sizes[-1])
    if suffix_input_dim is not None and configured_output_dim != int(suffix_input_dim):
        raise ValueError(
            "direct_active_matched_point zero-padded interface does not match "
            f"the suffix: configured={configured_output_dim}, "
            f"suffix_input_dim={int(suffix_input_dim)}"
        )

    from dendritic_modeling.scripts.script_utils.config_utils import (
        prepare_ei_network_params,
    )

    params = prepare_ei_network_params(raw_params, input_dim)
    return DirectActiveMatchedPointBottleneck(**params)


def _build_sparse_active_matched_point_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build an all-output-active fixed-index affine point control."""

    del type
    raw_params = _to_plain_mapping(parameters)
    if not ("architecture" in raw_params and "connectivity" in raw_params):
        raise ValueError(
            "sparse_active_matched_point requires structured "
            "architecture/connectivity reference config"
        )
    if input_dim is None:
        raise ValueError("input_dim required when building sparse_active_matched_point")

    reference_output_sizes = raw_params["architecture"].get(
        "excitatory_layer_sizes", []
    )
    if not reference_output_sizes:
        raise ValueError("sparse_active_matched_point requires excitatory_layer_sizes")
    configured_output_dim = int(reference_output_sizes[-1])
    if suffix_input_dim is not None and configured_output_dim != int(suffix_input_dim):
        raise ValueError(
            "sparse_active_matched_point output does not match the suffix: "
            f"configured={configured_output_dim}, "
            f"suffix_input_dim={int(suffix_input_dim)}"
        )

    from dendritic_modeling.scripts.script_utils.config_utils import (
        prepare_ei_network_params,
    )

    params = prepare_ei_network_params(raw_params, input_dim)
    if "target_active_parameters" in raw_params:
        params["target_active_parameters"] = raw_params["target_active_parameters"]
    return SparseActiveMatchedPointAffine(**params)


def _build_reference_matched_mlp(
    *,
    reference: nn.Module,
    input_dim: Optional[int],
    match_mode: str,
    activation: str = "relu",
) -> nn.Module:
    if input_dim is None:
        raise ValueError("input_dim required when building matched MLP baselines")

    output_dim = int(reference.output_dim)
    if match_mode == "effective" and hasattr(reference, "get_effective_params"):
        target_params = int(reference.get_effective_params())
    elif match_mode in {"effective", "total"}:
        target_params = sum(
            p.numel() for p in reference.parameters() if p.requires_grad
        )
    else:
        raise ValueError(f"Unknown matched MLP mode: {match_mode!r}")

    # One hidden layer, matching PointMatchedParamMLP semantics:
    # (input_dim * H + H) + (H * output_dim + output_dim) >= target_params.
    denom = int(input_dim) + output_dim + 1
    hidden_width = max(1, math.ceil(max(0, target_params - output_dim) / denom))
    return MLP(
        input_dim=int(input_dim),
        hidden_dims=[hidden_width],
        activation=activation,
        output_dim=output_dim,
    )


def _ensure_flat_parameters(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
) -> dict[str, Any]:
    if isinstance(parameters, DictConfig):
        raise ValueError(
            f"Architecture {type} expects flat parameters, not structured config"
        )
    return parameters


def _build_flat_mlp_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build flat MLP, or point-MLP when passed structured config."""
    del suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if "architecture" in raw_params and "connectivity" in raw_params:
        return _build_point_mlp_architecture(type, parameters, input_dim, None)
    return MLP(**_ensure_flat_parameters(type, parameters))


def _build_flat_architecture(
    cls: type[nn.Module],
) -> ArchitectureBuilder:
    def _builder(
        type: str,
        parameters: Union[dict[str, Any], DictConfig],
        input_dim: Optional[int],
        suffix_input_dim: Optional[int],
    ) -> nn.Module:
        del input_dim, suffix_input_dim
        return cls(**_ensure_flat_parameters(type, parameters))

    return _builder


def _build_total_param_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del type, suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if raw_params.get("population_network"):
        reference = _build_population_network_architecture(
            "population_network", raw_params, input_dim, None
        )
        return _build_reference_matched_mlp(
            reference=reference,
            input_dim=input_dim,
            match_mode="total",
        )
    return MatchedTotalParamMLP(parameters, input_dim)


def _build_active_param_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del type, suffix_input_dim
    raw_params = _to_plain_mapping(parameters)
    if raw_params.get("population_network"):
        reference = _build_population_network_architecture(
            "population_network", raw_params, input_dim, None
        )
        return _build_reference_matched_mlp(
            reference=reference,
            input_dim=input_dim,
            match_mode="effective",
        )
    return MatchedActiveParamMLP(parameters, input_dim)


def _build_sparse_structured_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del type, suffix_input_dim
    return SparseStructuredMLP(parameters, input_dim)


def _build_sparse_structured_flat_mlp(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del type, suffix_input_dim
    return SparseStructuredMLP(parameters, input_dim, flatten_dendrites=True)


def _build_pathway_router(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del input_dim, suffix_input_dim
    return PathwayRouter(**_ensure_flat_parameters(type, parameters))


def _build_identity(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    del input_dim, suffix_input_dim
    return Identity(**_ensure_flat_parameters(type, parameters))


def _build_dendritic_alexnet_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int],
    suffix_input_dim: Optional[int],
) -> nn.Module:
    """Build an end-to-end AlexNet-shaped dendritic vision network."""

    del type, suffix_input_dim
    from dendritic_modeling.networks.architectures.vision import DendriticAlexNet

    raw_params = _to_plain_mapping(parameters)
    input_channels = int(
        raw_params.get("vision", {}).get(
            "input_channels", 3 if input_dim is None else input_dim
        )
    )
    return DendriticAlexNet(raw_params, input_channels=input_channels)


def _build_builtin_architecture_registry() -> dict[str, ArchitectureBuilder]:
    builders: dict[str, ArchitectureBuilder] = {}
    builders.update(dict.fromkeys(_UNIFIED_EI_TYPES, _build_unified_ei_architecture))
    builders.update(
        dict.fromkeys(_BASELINE_RNN_TYPES, _build_baseline_rnn_architecture)
    )
    builders.update(
        dict.fromkeys(
            _HETEROGENEOUS_LEAK_CTRNN_TYPES,
            _build_heterogeneous_leak_ctrnn_architecture,
        )
    )
    builders.update(
        dict.fromkeys(_LEGENDRE_MEMORY_TYPES, _build_legendre_memory_architecture)
    )
    builders.update(
        dict.fromkeys(_POPULATION_NETWORK_TYPES, _build_population_network_architecture)
    )
    builders.update(
        dict.fromkeys(_SPATIAL_DENDRITIC_TYPES, _build_spatial_dendritic_architecture)
    )
    builders.update(
        dict.fromkeys(
            _SPATIAL_DENDRITIC_CONV_TYPES, _build_spatial_dendritic_conv_architecture
        )
    )
    builders.update(
        dict.fromkeys(
            _SPATIAL_PATCH_POINT_TYPES, _build_spatial_patch_point_architecture
        )
    )
    builders.update(dict.fromkeys(_EINET_TYPES, _build_einet_architecture))
    builders.update(
        {
            "direct_active_matched_point": (
                _build_direct_active_matched_point_architecture
            ),
            "dendritic_alexnet": _build_dendritic_alexnet_architecture,
            "sparse_active_matched_point": (
                _build_sparse_active_matched_point_architecture
            ),
            "point_mlp": _build_point_mlp_architecture,
            "mlp": _build_flat_mlp_architecture,
            "total_param_mlp": _build_total_param_mlp,
            "active_param_mlp": _build_active_param_mlp,
            "ss_mlp": _build_sparse_structured_mlp,
            "ss_mlp_flat": _build_sparse_structured_flat_mlp,
            "alexnet": _build_flat_architecture(AlexNet),
            "cnn": _build_flat_architecture(CNNDownsample),
            "cnn_down": _build_flat_architecture(CNNDownsample),
            "cnndownsample": _build_flat_architecture(CNNDownsample),
            "cnn_downsample": _build_flat_architecture(CNNDownsample),
            "cnn_up": _build_flat_architecture(CNNUpsample),
            "cnnupsample": _build_flat_architecture(CNNUpsample),
            "cnn_upsample": _build_flat_architecture(CNNUpsample),
            "effectivematchedparammlp": _build_flat_architecture(
                EffectiveMatchedParamMLP
            ),
            "mlpautoencoder": _build_flat_architecture(MLPAutoencoder),
            "cnnautoencoder": _build_flat_architecture(CNNAutoencoder),
            "mlpvariationalautoencoder": _build_flat_architecture(
                MLPVariationalAutoencoder
            ),
            "cnnvariationalautoencoder": _build_flat_architecture(
                CNNVariationalAutoencoder
            ),
            "pathway_router": _build_pathway_router,
            "router": _build_pathway_router,
            "identity": _build_identity,
            "none": _build_identity,
        }
    )
    return builders


_BUILTIN_ARCHITECTURE_BUILDERS: dict[str, ArchitectureBuilder] = (
    _build_builtin_architecture_registry()
)


def get_architecture(
    type: str,
    parameters: Union[dict[str, Any], DictConfig],
    input_dim: Optional[int] = None,
    suffix_input_dim: Optional[int] = None,
) -> nn.Module:
    """
    Factory method for creating network architectures.

    Args:
        type: Architecture type name
        parameters: Either flat parameters dict or structured config
        input_dim: Optional input dimension (needed for EINet from config)
        suffix_input_dim: Optional suffix input dimension. Used by spatial
            dendritic cores (e.g., ``hierarchical_dendritic_tensor_map``) to
            add an output projection when the EINet output dim doesn't match
            the downstream decoder's expected input.

    Returns:
        Initialized network module
    """
    type = (type or "").lower()
    if type in {"ei_net", "unified_einet"}:
        logger.warning(
            "Architecture type %r is deprecated; use 'unified_ei' instead.",
            type,
        )
    builder = _BUILTIN_ARCHITECTURE_BUILDERS.get(type)
    if builder is not None:
        return builder(type, parameters, input_dim, suffix_input_dim)

    registered = build_registered_architecture(
        type,
        parameters,
        input_dim=input_dim,
        suffix_input_dim=suffix_input_dim,
    )
    if registered is not None:
        return registered

    raise ValueError(f"Invalid architecture type: {type}")

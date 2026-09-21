"""Option normalization helpers for dendritic branch layers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dendritic_modeling.networks.activations import resolve_dendritic_activation
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_config import (
    DendriticBranchConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)


def resolve_branch_config(branch_config: Any) -> DendriticBranchConfig | None:
    """Return a typed branch-layer config when one was provided."""

    if branch_config is None:
        return None
    return DendriticBranchConfig.from_config(branch_config)


def resolve_branch_options(branch_config: Any) -> dict[str, Any]:
    """Return legacy branch-layer kwargs from an optional typed config."""

    normalized_config = resolve_branch_config(branch_config)
    if normalized_config is None:
        return {}
    return normalized_config.to_kwargs()


def resolve_synapse_options(
    *,
    formal_synapse_kwargs: Mapping[str, Any],
    extra_kwargs: Mapping[str, Any],
    synapse_config: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize synapse options and return unconsumed constructor kwargs."""

    normalized_config, remaining_kwargs = resolve_synapse_config(
        formal_synapse_kwargs=formal_synapse_kwargs,
        extra_kwargs=extra_kwargs,
        synapse_config=synapse_config,
    )
    return normalized_config.to_kwargs(), remaining_kwargs


def resolve_synapse_config(
    *,
    formal_synapse_kwargs: Mapping[str, Any],
    extra_kwargs: Mapping[str, Any],
    synapse_config: Any,
) -> tuple[DendriticSynapseConfig, dict[str, Any]]:
    """Normalize synapse options and return a typed config plus extra kwargs."""

    if synapse_config is None:
        config_source = {**extra_kwargs, **formal_synapse_kwargs}
        normalized_config = DendriticSynapseConfig.from_kwargs(config_source)
    else:
        normalized_config = DendriticSynapseConfig.from_config(synapse_config)
    _, remaining_kwargs = DendriticSynapseConfig.split_kwargs(extra_kwargs)
    return normalized_config, remaining_kwargs


def resolve_branch_layer_synapse_config(
    *,
    formal_synapse_kwargs: Mapping[str, Any],
    extra_kwargs: Mapping[str, Any],
    synapse_config: Any,
) -> tuple[DendriticSynapseConfig, dict[str, Any], dict[str, Any]]:
    """Normalize branch-layer synapse config and dendritic activation aliases."""

    resolved_config, remaining_kwargs = resolve_synapse_config(
        formal_synapse_kwargs=formal_synapse_kwargs,
        extra_kwargs=extra_kwargs,
        synapse_config=synapse_config,
    )
    synapse_options = resolved_config.to_kwargs()
    reactivate, reactivation_type = resolve_dendritic_activation(
        synapse_options["dendritic_activation"],
        synapse_options["reactivate"],
        synapse_options["reactivation_type"],
    )
    synapse_options["reactivate"] = reactivate
    synapse_options["reactivation_type"] = reactivation_type
    return (
        DendriticSynapseConfig.from_kwargs(synapse_options),
        synapse_options,
        remaining_kwargs,
    )


def resolve_dendrinet_synapse_config(
    *,
    formal_synapse_kwargs: Mapping[str, Any],
    extra_kwargs: Mapping[str, Any],
    synapse_config: Any,
    branch_factors: Any,
    dendritic_activation: Any,
    reactivate: bool,
    reactivation_type: str,
) -> tuple[DendriticSynapseConfig, dict[str, Any], str, bool, str]:
    """Normalize shared DendriNet synapse options for all branch layers."""

    formal_options = dict(formal_synapse_kwargs)
    formal_options["branch_factors"] = branch_factors
    if synapse_config is None:
        reactivate, reactivation_type = resolve_dendritic_activation(
            dendritic_activation,
            reactivate,
            reactivation_type,
        )
        formal_options["reactivate"] = reactivate
        formal_options["reactivation_type"] = reactivation_type
        resolved_config = DendriticSynapseConfig.from_kwargs(
            {**extra_kwargs, **formal_options}
        )
    else:
        resolved_config = DendriticSynapseConfig.from_config(synapse_config)
        synapse_options = resolved_config.to_kwargs()
        reactivate, reactivation_type = resolve_dendritic_activation(
            synapse_options["dendritic_activation"],
            synapse_options["reactivate"],
            synapse_options["reactivation_type"],
        )
        synapse_options["reactivate"] = reactivate
        synapse_options["reactivation_type"] = reactivation_type
        resolved_config = DendriticSynapseConfig.from_kwargs(synapse_options)
    resolved_config = resolved_config.with_branch_factors(branch_factors)
    _, branch_extra_kwargs = DendriticSynapseConfig.split_kwargs(extra_kwargs)
    dendritic_activation_name = reactivation_type if reactivate else "none"
    return (
        resolved_config,
        branch_extra_kwargs,
        dendritic_activation_name,
        reactivate,
        reactivation_type,
    )


__all__ = [
    "resolve_branch_config",
    "resolve_branch_layer_synapse_config",
    "resolve_branch_options",
    "resolve_dendrinet_synapse_config",
    "resolve_synapse_config",
    "resolve_synapse_options",
]

"""Unified recurrent E/I architecture factory helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_mapping
from dendritic_modeling.config.legacy import canonicalize_unified_ei_config
from dendritic_modeling.config.model_aliases import (
    get_unified_ei_alias_overrides,
    warn_alias_conflicts,
)


def _apply_population_alias_overrides(
    population: dict[str, Any],
    alias_overrides: dict[str, object],
    *,
    core_type: str,
    context: str,
) -> dict[str, Any]:
    """Apply recurrent type-alias semantics to one PopulationConfig payload."""

    if not alias_overrides:
        return population
    updated = dict(population)
    warn_alias_conflicts(core_type, updated, alias_overrides, context=context)
    for key, value in alias_overrides.items():
        updated[key] = value
    return updated


def _coerce_population_config(
    population: Any,
    *,
    population_config_cls: type[Any],
    alias_overrides: dict[str, object],
    core_type: str,
    context: str,
    initialization_seed: int | None = None,
    initialization_namespace: str = "",
) -> Any:
    """Normalize one optional E/I population config payload."""

    if population is not None and not isinstance(population, dict):
        plain_population = _to_plain_mapping(population)
        if plain_population:
            population = plain_population
    if isinstance(population, dict):
        population = dict(population)
        if initialization_seed is not None:
            population.setdefault("initialization_seed", initialization_seed)
        if not population.get("initialization_namespace"):
            population["initialization_namespace"] = initialization_namespace
        population = _apply_population_alias_overrides(
            population,
            alias_overrides,
            core_type=core_type,
            context=context,
        )
        return population_config_cls(**population)
    return population


def build_unified_ei_layer_configs(
    *,
    layers: Sequence[Any],
    ei_layer_config_cls: type[Any],
    population_config_cls: type[Any],
    alias_overrides: dict[str, object],
    core_type: str,
    initialization_seed: int | None = None,
) -> list[Any]:
    """Build typed unified E/I layer configs from mappings or config objects."""

    layers_cfg = []
    for layer_idx, layer in enumerate(layers):
        if isinstance(layer, ei_layer_config_cls):
            if not alias_overrides:
                layers_cfg.append(layer)
                continue
            layer = asdict(layer)
        if not isinstance(layer, dict):
            raise ValueError(
                "Each unified E-I layer config must be a dict or EILayerConfig"
            )
        layer = dict(layer)
        exc = _coerce_population_config(
            layer.get("excitatory"),
            population_config_cls=population_config_cls,
            alias_overrides=alias_overrides,
            core_type=core_type,
            context=f"model.core.unified_ei.layers[{layer_idx}].excitatory",
            initialization_seed=initialization_seed,
            initialization_namespace=f"unified.layer.{layer_idx}.excitatory",
        )
        inh = _coerce_population_config(
            layer.get("inhibitory"),
            population_config_cls=population_config_cls,
            alias_overrides=alias_overrides,
            core_type=core_type,
            context=f"model.core.unified_ei.layers[{layer_idx}].inhibitory",
            initialization_seed=initialization_seed,
            initialization_namespace=f"unified.layer.{layer_idx}.inhibitory",
        )
        if isinstance(exc, population_config_cls):
            layer["excitatory"] = exc
        if isinstance(inh, population_config_cls):
            layer["inhibitory"] = inh
        layers_cfg.append(ei_layer_config_cls(**layer))
    return layers_cfg


def _build_unified_ei_architecture(
    type: str,
    parameters: dict[str, Any],
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> Any:
    """Build the unified recurrent E/I architecture family."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.recurrent import (
        EILayerConfig,
        EINetwork,
        EINetworkConfig,
        PopulationConfig,
    )

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating unified E-I architecture (type='{type}')"
        )

    raw_params = _to_plain_mapping(parameters)
    unified_cfg = raw_params.get("unified_ei", raw_params.get("ei_unified", {}))
    if not unified_cfg and "layers" in raw_params:
        # Allow passing EINetworkConfig-shaped dict directly.
        unified_cfg = raw_params

    if not isinstance(unified_cfg, dict):
        raise ValueError(
            "Unified E-I config must be a mapping, got "
            f"{unified_cfg.__class__.__name__}"
        )

    unified_cfg = canonicalize_unified_ei_config(unified_cfg)
    unified_cfg["input_dim"] = input_dim

    alias_overrides = get_unified_ei_alias_overrides(type)
    layers_cfg = build_unified_ei_layer_configs(
        layers=unified_cfg.get("layers", []),
        ei_layer_config_cls=EILayerConfig,
        population_config_cls=PopulationConfig,
        alias_overrides=alias_overrides,
        core_type=type,
        initialization_seed=raw_params.get("initialization_seed"),
    )

    if not layers_cfg:
        raise ValueError(
            "Unified E-I architecture requires at least one layer in unified_ei.layers"
        )

    unified_cfg["layers"] = layers_cfg
    return EINetwork(EINetworkConfig(**unified_cfg))


__all__ = [
    "_apply_population_alias_overrides",
    "_build_unified_ei_architecture",
    "build_unified_ei_layer_configs",
]

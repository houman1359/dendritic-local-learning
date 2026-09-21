"""Population-network config translation for transformer replacements."""

from __future__ import annotations

import logging
from typing import Any

from dendritic_modeling.config.conversion import normalize_sparsity_type
from dendritic_modeling.config.model_aliases import (
    get_core_morphology_alias_overrides,
    warn_alias_conflicts,
)
from dendritic_modeling.networks.architectures.transformer.utils import (
    _to_plain_mapping,
)

logger = logging.getLogger(__name__)


def _normalize_topk_type(value: Any) -> str:
    return normalize_sparsity_type(value)


def _population_network_payload(
    core: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]] | None:
    population_network = _to_plain_mapping(core.get("population_network", {}))
    layers = population_network.get("layers", [])
    if not isinstance(layers, list) or not layers:
        return None
    layer = _to_plain_mapping(layers[0])
    populations = [_to_plain_mapping(pop) for pop in layer.get("populations", [])]
    return population_network, layer, populations


def _population_override(population: dict[str, Any]) -> dict[str, Any]:
    """Return explicit per-population overrides from either accepted shape."""
    nested = _to_plain_mapping(population.get("population", {}))
    direct = {
        key: value
        for key, value in population.items()
        if key not in {"name", "polarity", "population"}
    }
    return {**nested, **direct}


def _merged_population_config(
    layer: dict[str, Any],
    population: dict[str, Any],
) -> dict[str, Any]:
    defaults = _to_plain_mapping(layer.get("population_defaults", {}))
    return {**defaults, **_population_override(population)}


def _select_population(
    layer: dict[str, Any],
    populations: list[dict[str, Any]],
    *,
    polarity: str | None = None,
) -> dict[str, Any]:
    readout_name = layer.get("readout_population")
    if readout_name is not None:
        for population in populations:
            if str(population.get("name", "")) == str(readout_name):
                if (
                    polarity is None
                    or str(population.get("polarity", "")).lower() == polarity
                ):
                    return population

    if polarity is not None:
        for population in populations:
            if str(population.get("polarity", "")).lower() == polarity:
                return population

    return populations[0] if populations else {}


def _population_sparsity_config(payload: dict[str, Any]) -> dict[str, Any]:
    return _to_plain_mapping(payload.get("sparsity", {}))


def _population_dense_to_sparse_config(
    payload: dict[str, Any],
    sparsity: dict[str, Any],
) -> dict[str, Any]:
    if "annealed_topk" in sparsity and "dense_to_sparse" not in sparsity:
        logger.warning(
            "population_network sparsity.annealed_topk is deprecated; use "
            "dense_to_sparse instead."
        )
    return _to_plain_mapping(
        payload.get(
            "dense_to_sparse",
            sparsity.get("dense_to_sparse", sparsity.get("annealed_topk", {})),
        )
    )


def _population_indexed_config(
    payload: dict[str, Any],
    sparsity: dict[str, Any],
) -> dict[str, Any]:
    indexed = _to_plain_mapping(sparsity.get("indexed", {}))
    indexed.update(_to_plain_mapping(payload.get("indexed", {})))
    field_aliases = {
        "seed": "indexed_seed",
        "candidate_size": "indexed_candidate_size",
        "selection": "indexed_selection",
        "output_chunk_size": "indexed_output_chunk_size",
        "index_dtype": "indexed_index_dtype",
        "workspace_mb": "indexed_workspace_mb",
        "cache_transformed_weights": "indexed_cache_transformed_weights",
        "recompute_backward": "indexed_recompute_backward",
        "projection_backend": "indexed_projection_backend",
        "persistent_indices": "indexed_persistent_indices",
        "init_mode": "indexed_init_mode",
        "rewire_frequency": "indexed_rewire_frequency",
        "rewire_quantile": "indexed_rewire_quantile",
        "rewire_until_step": "indexed_rewire_until_step",
    }
    for indexed_key, population_key in field_aliases.items():
        if population_key in payload:
            indexed[indexed_key] = payload[population_key]
    return indexed


def _topk_value(
    payload: dict[str, Any],
    sparsity: dict[str, Any],
    payload_key: str,
    sparsity_key: str,
    default: Any,
) -> Any:
    if payload_key in payload:
        return payload[payload_key]
    return sparsity.get(sparsity_key, default)


def _input_transform_from_population_network(
    population_network: dict[str, Any],
    tr: dict[str, Any],
    transfer: dict[str, Any] | None = None,
) -> str:
    transfer_params = _to_plain_mapping(population_network.get("transfer_params", {}))
    transfer = transfer or {}
    return tr.get(
        "input_transform",
        population_network.get(
            "input_transform",
            transfer_params.get(
                "input_transform",
                transfer.get("input_transform", "signed_split"),
            ),
        ),
    )


def _common_population_replacement_kwargs(
    payload: dict[str, Any],
    *,
    topk_temperature_default: float,
) -> dict[str, Any]:
    sparsity = _population_sparsity_config(payload)
    indexed = _population_indexed_config(payload, sparsity)
    dense_to_sparse = _population_dense_to_sparse_config(payload, sparsity)
    return {
        "topk_type": _normalize_topk_type(
            _topk_value(payload, sparsity, "topk_type", "type", "indexed_rewire")
        ),
        "topk_init_method": _topk_value(
            payload,
            sparsity,
            "topk_init_method",
            "init_method",
            "xavier_normal",
        ),
        "topk_noise_level": float(
            _topk_value(payload, sparsity, "topk_noise_level", "noise_level", 0.0)
        ),
        "topk_temperature": float(
            _topk_value(
                payload,
                sparsity,
                "topk_temperature",
                "temperature",
                topk_temperature_default,
            )
        ),
        "topk_ultrafast": bool(
            _topk_value(payload, sparsity, "topk_ultrafast", "ultrafast", False)
        ),
        "topk_weight_norm_order": _topk_value(
            payload,
            sparsity,
            "topk_weight_norm_order",
            "weight_norm_order",
            None,
        ),
        "topk_gamma": float(_topk_value(payload, sparsity, "topk_gamma", "gamma", 1.0)),
        "dense_to_sparse_initial_density": float(
            dense_to_sparse.get("initial_density", 1.0)
        ),
        "dense_to_sparse_initial_k": dense_to_sparse.get("initial_k"),
        "dense_to_sparse_start_step": int(dense_to_sparse.get("start_step", 0)),
        "dense_to_sparse_end_step": int(dense_to_sparse.get("end_step", 1000)),
        "dense_to_sparse_update_interval": int(
            dense_to_sparse.get("update_interval", 1)
        ),
        "dense_to_sparse_schedule": str(dense_to_sparse.get("schedule", "cubic")),
        "dense_to_sparse_freeze_on_end": bool(
            dense_to_sparse.get("freeze_on_end", False)
        ),
        "dense_to_sparse_advance_on_forward": bool(
            dense_to_sparse.get("advance_on_forward", True)
        ),
        "dense_to_sparse_prune_metric": str(
            dense_to_sparse.get("prune_metric", "weight")
        ),
        "indexed_seed": indexed.get("seed", sparsity.get("seed")),
        "indexed_candidate_size": indexed.get("candidate_size"),
        "indexed_selection": indexed.get("selection", "standard"),
        "indexed_output_chunk_size": int(indexed.get("output_chunk_size", 2048)),
        "indexed_index_dtype": str(indexed.get("index_dtype", "int64")),
        "indexed_workspace_mb": indexed.get("workspace_mb"),
        "indexed_cache_transformed_weights": bool(
            indexed.get("cache_transformed_weights", False)
        ),
        "indexed_recompute_backward": bool(indexed.get("recompute_backward", False)),
        "indexed_projection_backend": str(indexed.get("projection_backend", "eager")),
        "indexed_persistent_indices": bool(indexed.get("persistent_indices", True)),
        "indexed_init_mode": str(indexed.get("init_mode", "per_rank")),
        "indexed_rewire_frequency": int(indexed.get("rewire_frequency", 100)),
        "indexed_rewire_quantile": float(indexed.get("rewire_quantile", 0.05)),
        "indexed_rewire_until_step": indexed.get("rewire_until_step"),
    }


def _population_reactivation_kwargs(
    payload: dict[str, Any],
    reactivation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reactivation = reactivation or {}
    return {
        "reactivate": bool(
            payload.get("reactivate", reactivation.get("enabled", True))
        ),
        "reactivation_type": payload.get(
            "reactivation_type",
            reactivation.get("type", "param_tanh"),
        ),
        "reactivation_init_m": float(
            payload.get("reactivation_init_m", reactivation.get("init_m", 1.0))
        ),
        "reactivation_init_b": float(
            payload.get("reactivation_init_b", reactivation.get("init_b", 0.5))
        ),
        "reactivation_init_policy": payload.get(
            "reactivation_init_policy",
            reactivation.get("init_policy", "analytical"),
        ),
        "reactivation_occupancy_quantile_low": payload.get(
            "reactivation_occupancy_quantile_low",
            reactivation.get("occupancy_quantile_low"),
        ),
        "reactivation_occupancy_quantile_high": payload.get(
            "reactivation_occupancy_quantile_high",
            reactivation.get("occupancy_quantile_high"),
        ),
        "reactivation_occupancy_target_low": payload.get(
            "reactivation_occupancy_target_low",
            reactivation.get("occupancy_target_low"),
        ),
        "reactivation_occupancy_target_high": payload.get(
            "reactivation_occupancy_target_high",
            reactivation.get("occupancy_target_high"),
        ),
        "reactivation_calibration_min_quantile_width": float(
            payload.get(
                "reactivation_calibration_min_quantile_width",
                reactivation.get("calibration_min_quantile_width", 1e-3),
            )
        ),
        "reactivation_calibration_max_m": float(
            payload.get(
                "reactivation_calibration_max_m",
                reactivation.get("calibration_max_m", 50.0),
            )
        ),
        "reactivation_calibration_revert_on_invalid": bool(
            payload.get(
                "reactivation_calibration_revert_on_invalid",
                reactivation.get("calibration_revert_on_invalid", True),
            )
        ),
        "reactivation_sigma_aware_k": float(
            payload.get(
                "reactivation_sigma_aware_k",
                reactivation.get("sigma_aware_k", 0.25),
            )
        ),
        "reactivation_memory_efficient": bool(
            payload.get(
                "reactivation_memory_efficient",
                reactivation.get("memory_efficient", False),
            )
        ),
    }


def _population_morphology_kwargs(
    payload: dict[str, Any],
    core_type: str,
    *,
    default_use_shunting: bool,
) -> dict[str, Any]:
    requested = {
        "use_shunting": payload.get("use_shunting", default_use_shunting),
        "use_additive_normalization": payload.get(
            "use_additive_normalization",
            False,
        ),
    }
    morphology_overrides = get_core_morphology_alias_overrides(core_type)
    warn_alias_conflicts(
        core_type,
        requested,
        morphology_overrides,
        context="model.core.population_network",
    )
    return {
        "use_shunting": bool(
            morphology_overrides.get("use_shunting", requested["use_shunting"])
        ),
        "use_additive_normalization": bool(
            morphology_overrides.get(
                "use_additive_normalization",
                requested["use_additive_normalization"],
            )
        ),
        "weight_transform": payload.get("weight_transform", "softplus"),
        "somatic_synapses": bool(payload.get("somatic_synapses", True)),
    }


def _population_blocklinear_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Return branch-aggregation options from population-network payloads."""
    blocklinear = _to_plain_mapping(payload.get("blocklinear", {}))
    return {
        "efficient_blocklinear": bool(
            payload.get(
                "efficient_blocklinear",
                blocklinear.get("efficient", False),
            )
        ),
    }


def _dendritic_ffn_kwargs_from_population_network(
    core: dict[str, Any],
    tr: dict[str, Any],
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    population_payload = _population_network_payload(core)
    if population_payload is None:
        raise ValueError("model.core.population_network.layers must be non-empty")
    population_network, layer, populations = population_payload
    population = _select_population(layer, populations, polarity="excitatory")
    payload = _merged_population_config(layer, population)
    core_type = str(core.get("type", "")).lower()

    kwargs: dict[str, Any] = {
        "dendritic_units": int(payload.get("n_neurons", 512)),
        "branch_factors": list(payload.get("branch_factors", [3, 3])),
        "synapses_per_branch": int(
            payload.get(
                "synapses_per_branch",
                payload.get("ff_excitatory_synapses", 40),
            )
        ),
        "input_transform": _input_transform_from_population_network(
            population_network,
            tr,
        ),
        **_common_population_replacement_kwargs(
            payload,
            topk_temperature_default=0.5,
        ),
        **_population_morphology_kwargs(
            payload,
            core_type,
            default_use_shunting=True,
        ),
        **_population_blocklinear_kwargs(payload),
        **_population_reactivation_kwargs(
            payload,
            _to_plain_mapping(core.get("reactivation", {})),
        ),
        "output_bias": bool(tr.get("output_bias", False)),
        "output_init_std": float(tr.get("output_init_std", 0.02)),
        "output_scale": float(tr.get("output_scale", 1.0)),
        "pre_norm": bool(tr.get("pre_norm", False)),
    }
    replacement_overrides = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    replacement_overrides.pop("kind", None)
    replacement_overrides.pop("type", None)
    kwargs.update(replacement_overrides)
    if overrides:
        kwargs.update(overrides)
    return kwargs


def _ei_stack_kwargs_from_population_network(
    core: dict[str, Any],
    tr: dict[str, Any],
    *,
    num_layers: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    population_payload = _population_network_payload(core)
    if population_payload is None:
        raise ValueError("model.core.population_network.layers must be non-empty")
    population_network, layer, populations = population_payload
    excitatory = _merged_population_config(
        layer,
        _select_population(layer, populations, polarity="excitatory"),
    )
    inhibitory_population = _select_population(
        layer,
        populations,
        polarity="inhibitory",
    )
    inhibitory = _merged_population_config(layer, inhibitory_population)
    core_type = str(core.get("type", "")).lower()

    kwargs: dict[str, Any] = {
        "num_layers": int(num_layers),
        "excitatory_cells": int(excitatory.get("n_neurons", 200)),
        "inhibitory_cells": int(inhibitory.get("n_neurons", 50)),
        "excitatory_branch_factors": list(excitatory.get("branch_factors", [2, 2])),
        "inhibitory_branch_factors": list(inhibitory.get("branch_factors", [2])),
        "ee_synapses_per_branch": int(
            excitatory.get(
                "ee_synapses_per_branch",
                excitatory.get("ff_excitatory_synapses", 32),
            )
        ),
        "ei_synapses_per_branch": int(
            inhibitory.get(
                "ei_synapses_per_branch",
                inhibitory.get("ff_excitatory_synapses", 16),
            )
        ),
        "ie_synapses_per_branch": int(
            excitatory.get(
                "ie_synapses_per_branch",
                excitatory.get("ff_inhibitory_synapses", 16),
            )
        ),
        "ii_synapses_per_branch": int(
            inhibitory.get(
                "ii_synapses_per_branch",
                inhibitory.get("ff_inhibitory_synapses", 8),
            )
        ),
        "input_transform": _input_transform_from_population_network(
            population_network,
            tr,
        ),
        **_population_morphology_kwargs(
            excitatory,
            core_type,
            default_use_shunting=True,
        ),
        **_common_population_replacement_kwargs(
            excitatory,
            topk_temperature_default=0.25,
        ),
        **_population_blocklinear_kwargs(excitatory),
        **_population_reactivation_kwargs(
            excitatory,
            _to_plain_mapping(core.get("reactivation", {})),
        ),
        "output_bias": bool(tr.get("output_bias", False)),
        "output_init_std": float(tr.get("output_init_std", 0.02)),
        "output_scale": float(tr.get("output_scale", 1.0)),
        "pre_norm": bool(tr.get("pre_norm", False)),
    }
    replacement_overrides = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    replacement_overrides.pop("kind", None)
    replacement_overrides.pop("type", None)
    kwargs.update(replacement_overrides)
    if overrides:
        kwargs.update(overrides)
    return kwargs

"""Config-to-kwargs translation for transformer dendritic replacements."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from dendritic_modeling.config.conversion import normalize_sparsity_type
from dendritic_modeling.config.model_aliases import (
    get_core_morphology_alias_overrides,
    warn_alias_conflicts,
)
from dendritic_modeling.networks.architectures.transformer.config_translation_population import (
    _dendritic_ffn_kwargs_from_population_network,
    _ei_stack_kwargs_from_population_network,
    _population_network_payload,
    _population_reactivation_kwargs,
)
from dendritic_modeling.networks.architectures.transformer.utils import (
    _first_list_value,
    _last_list_value,
    _to_plain_mapping,
)

logger = logging.getLogger(__name__)


def _normalize_topk_type(value: Any) -> str:
    return normalize_sparsity_type(value)


def _structured_blocklinear_kwargs(core: dict[str, Any]) -> dict[str, Any]:
    """Return branch-aggregation options from structured core configs."""
    blocklinear = _to_plain_mapping(core.get("blocklinear", {}))
    return {"efficient_blocklinear": bool(blocklinear.get("efficient", False))}


def _dense_to_sparse_config(sparsity: dict[str, Any]) -> dict[str, Any]:
    """Return dense-to-sparse config while preserving legacy alias warnings."""
    if "annealed_topk" in sparsity and "dense_to_sparse" not in sparsity:
        logger.warning(
            "model.core.sparsity.annealed_topk is deprecated; use "
            "model.core.sparsity.dense_to_sparse instead."
        )
    return _to_plain_mapping(
        sparsity.get("dense_to_sparse", sparsity.get("annealed_topk", {}))
    )


def _replacement_overrides(transformer_replacement: dict[str, Any]) -> dict[str, Any]:
    """Return replacement kwargs with selector-only keys removed."""
    replacement_overrides = _to_plain_mapping(
        transformer_replacement.get("replacement_kwargs", {})
    )
    replacement_overrides.pop("kind", None)
    replacement_overrides.pop("type", None)
    return replacement_overrides


def _structured_sparsity_kwargs(
    sparsity: dict[str, Any],
    *,
    default_temperature: float,
    include_ultrafast: bool = False,
) -> dict[str, Any]:
    """Translate structured sparsity config into replacement constructor kwargs."""
    indexed = _to_plain_mapping(sparsity.get("indexed", {}))
    dense_to_sparse = _dense_to_sparse_config(sparsity)
    kwargs: dict[str, Any] = {
        "topk_type": _normalize_topk_type(sparsity.get("type", "indexed_rewire")),
        "topk_init_method": sparsity.get("init_method", "xavier_normal"),
        "topk_noise_level": float(sparsity.get("noise_level", 0.0)),
        "topk_temperature": float(sparsity.get("temperature", default_temperature)),
        "topk_weight_norm_order": sparsity.get("weight_norm_order"),
        "topk_gamma": float(sparsity.get("gamma", 1.0)),
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
        # Transformer replacements default to the recompute backend: the
        # eager path keeps the gathered [tokens, units, K] operand alive for
        # backward, which at LM token budgets (e.g. OLMo-1B, seq 2048 x
        # batch 8, units 8192, K 256) is ~128 GiB per core — untrainable.
        # Recompute saves only the flat activations and re-gathers in
        # backward. Explicit config always wins.
        "indexed_projection_backend": str(
            indexed.get("projection_backend", "recompute")
        ),
        "indexed_persistent_indices": bool(indexed.get("persistent_indices", True)),
        "indexed_init_mode": str(indexed.get("init_mode", "per_rank")),
        "indexed_rewire_frequency": int(indexed.get("rewire_frequency", 100)),
        "indexed_rewire_quantile": float(indexed.get("rewire_quantile", 0.05)),
        "indexed_rewire_until_step": indexed.get("rewire_until_step"),
    }
    if include_ultrafast:
        kwargs["topk_ultrafast"] = bool(sparsity.get("ultrafast", False))
    return kwargs


def build_dendritic_ffn_kwargs_from_core_config(
    core_config: Any,
    *,
    transformer_replacement: Any = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a structured ``model.core`` config into FFN replacement kwargs.

    This mirrors the AlexNet replacement convention: the replacement target is
    described separately, while the dendritic module itself is still described
    by the shared ``model.core`` fields.
    """
    core = _to_plain_mapping(core_config)
    tr = _to_plain_mapping(transformer_replacement)
    if _population_network_payload(core) is not None:
        return _dendritic_ffn_kwargs_from_population_network(
            core,
            tr,
            overrides=overrides,
        )
    architecture = _to_plain_mapping(core.get("architecture", {}))
    connectivity = _to_plain_mapping(core.get("connectivity", {}))
    transfer = _to_plain_mapping(core.get("transfer", {}))
    morphology = _to_plain_mapping(core.get("morphology", {}))
    sparsity = _to_plain_mapping(core.get("sparsity", {}))
    reactivation = _to_plain_mapping(core.get("reactivation", {}))
    core_type = str(core.get("type", "")).lower()
    morphology_overrides = get_core_morphology_alias_overrides(core_type)
    warn_alias_conflicts(
        core_type,
        morphology,
        morphology_overrides,
        context="model.core.morphology",
    )

    dendritic_units = int(
        architecture.get(
            "dendritic_units",
            _last_list_value(architecture.get("excitatory_layer_sizes"), 512),
        )
    )
    branch_factors = architecture.get(
        "branch_factors",
        architecture.get("excitatory_branch_factors", [3, 3]),
    )
    synapses_per_branch = int(
        connectivity.get(
            "synapses_per_branch",
            _first_list_value(
                connectivity.get("ee_synapses_per_branch_per_layer"),
                40,
            ),
        )
    )

    kwargs: dict[str, Any] = {
        "dendritic_units": dendritic_units,
        "branch_factors": list(branch_factors),
        "synapses_per_branch": synapses_per_branch,
        "input_transform": tr.get(
            "input_transform",
            transfer.get("input_transform", "signed_split"),
        ),
        **_structured_sparsity_kwargs(
            sparsity,
            default_temperature=0.5,
            include_ultrafast=True,
        ),
        "use_shunting": bool(
            morphology_overrides.get(
                "use_shunting", morphology.get("use_shunting", True)
            )
        ),
        "use_additive_normalization": bool(
            morphology_overrides.get(
                "use_additive_normalization",
                morphology.get("use_additive_normalization", False),
            )
        ),
        "weight_transform": morphology.get("weight_transform", "softplus"),
        **_structured_blocklinear_kwargs(core),
        **_population_reactivation_kwargs({}, reactivation),
        "output_bias": bool(tr.get("output_bias", False)),
        "output_init_std": float(tr.get("output_init_std", 0.02)),
        "output_scale": float(tr.get("output_scale", 1.0)),
        "pre_norm": bool(tr.get("pre_norm", False)),
    }
    kwargs.update(_replacement_overrides(tr))
    if overrides:
        kwargs.update(overrides)
    return kwargs


def build_ei_stack_kwargs_from_core_config(
    core_config: Any,
    *,
    transformer_replacement: Any = None,
    num_layers: int,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate a shared ``model.core`` config into EI-stack kwargs."""
    core = _to_plain_mapping(core_config)
    tr = _to_plain_mapping(transformer_replacement)
    if _population_network_payload(core) is not None:
        return _ei_stack_kwargs_from_population_network(
            core,
            tr,
            num_layers=num_layers,
            overrides=overrides,
        )
    architecture = _to_plain_mapping(core.get("architecture", {}))
    connectivity = _to_plain_mapping(core.get("connectivity", {}))
    transfer = _to_plain_mapping(core.get("transfer", {}))
    morphology = _to_plain_mapping(core.get("morphology", {}))
    sparsity = _to_plain_mapping(core.get("sparsity", {}))
    reactivation = _to_plain_mapping(core.get("reactivation", {}))
    core_type = str(core.get("type", "")).lower()
    morphology_overrides = get_core_morphology_alias_overrides(core_type)
    warn_alias_conflicts(
        core_type,
        morphology,
        morphology_overrides,
        context="model.core.morphology",
    )

    kwargs: dict[str, Any] = {
        "num_layers": int(num_layers),
        "excitatory_cells": _last_list_value(
            architecture.get("excitatory_layer_sizes"),
            200,
        ),
        "inhibitory_cells": _last_list_value(
            architecture.get("inhibitory_layer_sizes"),
            50,
        ),
        "excitatory_branch_factors": list(
            architecture.get("excitatory_branch_factors", [2, 2])
        ),
        "inhibitory_branch_factors": list(
            architecture.get("inhibitory_branch_factors", [2])
        ),
        "ee_synapses_per_branch": _first_list_value(
            connectivity.get("ee_synapses_per_branch_per_layer"),
            32,
        ),
        "ei_synapses_per_branch": _first_list_value(
            connectivity.get("ei_synapses_per_branch_per_layer"),
            16,
        ),
        "ie_synapses_per_branch": _first_list_value(
            connectivity.get("ie_synapses_per_branch_per_layer"),
            16,
        ),
        "ii_synapses_per_branch": _first_list_value(
            connectivity.get("ii_synapses_per_branch_per_layer"),
            8,
        ),
        "input_transform": tr.get(
            "input_transform",
            transfer.get("input_transform", "signed_split"),
        ),
        "use_shunting": bool(
            morphology_overrides.get(
                "use_shunting", morphology.get("use_shunting", True)
            )
        ),
        "use_additive_normalization": bool(
            morphology_overrides.get(
                "use_additive_normalization",
                morphology.get("use_additive_normalization", False),
            )
        ),
        "weight_transform": morphology.get("weight_transform", "softplus"),
        **_structured_blocklinear_kwargs(core),
        **_structured_sparsity_kwargs(sparsity, default_temperature=0.25),
        **_population_reactivation_kwargs({}, reactivation),
        "output_bias": bool(tr.get("output_bias", False)),
        "output_init_std": float(tr.get("output_init_std", 0.02)),
        "output_scale": float(tr.get("output_scale", 1.0)),
        "pre_norm": bool(tr.get("pre_norm", False)),
    }
    kwargs.update(_replacement_overrides(tr))
    if overrides:
        kwargs.update(overrides)
    return kwargs


def build_population_network_ffn_kwargs_from_core_config(
    core_config: Any,
    *,
    transformer_replacement: Any = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Preserve a complete ``population_network`` as a transformer FFN core.

    The older compatibility builders intentionally project a population config
    onto either one DendriNet population or one E/I stack.  This builder is the
    non-lossy route: all named populations, directed connections, branch
    morphologies, pathway roles, and sparse-training options reach the canonical
    :class:`PopulationNetwork` implementation unchanged.
    """

    core = _to_plain_mapping(core_config)
    tr = _to_plain_mapping(transformer_replacement)
    population_network = _to_plain_mapping(core.get("population_network", {}))
    layers = population_network.get("layers", [])
    if not isinstance(layers, list) or not layers:
        raise ValueError(
            "population-network transformer replacement requires "
            "model.core.population_network.layers"
        )

    input_transform = tr.get(
        "input_transform",
        population_network.get("input_transform", "signed_split"),
    )
    population_network = deepcopy(population_network)
    population_network.pop("input_transform", None)
    kwargs: dict[str, Any] = {
        "population_network": population_network,
        "input_transform": input_transform,
        "biological_neuron": core.get("biological_neuron"),
        "output_bias": bool(tr.get("output_bias", False)),
        "output_init_std": float(tr.get("output_init_std", 0.02)),
        "output_scale": float(tr.get("output_scale", 1.0)),
        "pre_norm": bool(tr.get("pre_norm", False)),
    }
    kwargs.update(_replacement_overrides(tr))
    if overrides:
        kwargs.update(overrides)
    return kwargs

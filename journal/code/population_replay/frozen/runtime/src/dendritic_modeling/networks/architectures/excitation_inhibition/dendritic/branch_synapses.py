"""Sparse synapse layer construction for dendritic branch layers."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory import (
    get_sparse_layer,
    normalize_sparse_layer_type,
)
from dendritic_modeling.utils.stable_hash import stable_seed_offset


def _indexed_synapse_seed(
    owner: Any,
    *,
    pathway: str,
) -> int | None:
    """Derive a stable, independent indexed seed for one synaptic pathway.

    ``synapse_type`` alone is not a sufficient namespace: feedforward and
    recurrent excitation previously received the same seed (as did the two
    inhibitory pathways).  The population-network builder namespaces the base
    seed by layer and target population; this final offset separates pathway
    and dendritic depth without depending on Python's randomized ``hash``.
    """
    if owner.indexed_seed is None:
        return None
    return (
        int(owner.indexed_seed)
        + stable_seed_offset("branch_synapse", pathway, int(owner.layer_idx))
    ) % ((1 << 63) - 1)


def _pathway_gamma(owner: Any, synapse_type: str) -> float:
    """Use inhibitory gamma only for inhibitory sparse pathways."""
    return owner.topk_gamma if synapse_type == "inh" else 1.0


def _indexed_runtime_kwargs(owner: Any) -> dict[str, Any]:
    """Return config-controlled indexed storage and execution options."""
    return {
        "index_dtype": owner.indexed_index_dtype,
        "workspace_mb": owner.indexed_workspace_mb,
        "cache_transformed_weights": owner.indexed_cache_transformed_weights,
        "recompute_backward": owner.indexed_recompute_backward,
        "projection_backend": owner.indexed_projection_backend,
        "persistent_indices": owner.indexed_persistent_indices,
        "init_mode": owner.indexed_init_mode,
        "support_group_rows": getattr(owner, "indexed_support_group_rows", 1),
        "support_col_block": getattr(owner, "indexed_support_col_block", 1),
    }


def create_branch_sparse_layer(
    owner: Any,
    *,
    in_features: int,
    out_features: int,
    K: int | None,
    init_method: str,
    noise_level: float,
    synapse_type: str = "exc",
    pathway: str | None = None,
    forbidden_input_index_per_output: Any = None,
    connection_indices: Any = None,
    connection_mask: Any = None,
):
    """Create a sparse synapse layer using a branch layer's configuration."""
    pathway = pathway or synapse_type
    # Keep invalid sweep configs runnable by capping requested synapse count to
    # available presynaptic features for this layer.
    if K is not None and K > in_features:
        K = in_features

    layer_type = normalize_sparse_layer_type(owner.topk_type)
    if connection_indices is not None and layer_type != "indexed":
        raise ValueError(
            "Direct structured connection indices currently require "
            "sparsity.type='indexed'. Adaptive rewiring would leave the "
            "configured spatial regions."
        )

    if layer_type == "standard":
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            weight_transform=owner.weight_transform,
            weight_norm_order=owner.topk_weight_norm_order,
            gamma=_pathway_gamma(owner, synapse_type),
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
        )

    # Alternative strategies only incur overhead if explicitly requested.
    if layer_type == "stochastic":
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            temperature=getattr(owner, "topk_temperature", 0.5),
            ultrafast=getattr(owner, "topk_ultrafast", True),
            weight_transform=owner.weight_transform,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
        )

    if layer_type == "variance":
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            weight_transform=owner.weight_transform,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
        )

    if layer_type in {"dense_to_sparse", "annealed_topk"}:
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            weight_transform=owner.weight_transform,
            weight_norm_order=owner.topk_weight_norm_order,
            gamma=_pathway_gamma(owner, synapse_type),
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
            initial_density=owner.dense_to_sparse_initial_density,
            initial_k=owner.dense_to_sparse_initial_k,
            start_step=owner.dense_to_sparse_start_step,
            end_step=owner.dense_to_sparse_end_step,
            update_interval=owner.dense_to_sparse_update_interval,
            schedule=owner.dense_to_sparse_schedule,
            freeze_on_end=owner.dense_to_sparse_freeze_on_end,
            advance_on_forward=owner.dense_to_sparse_advance_on_forward,
            prune_metric=owner.dense_to_sparse_prune_metric,
        )

    if layer_type in {"deepst", "credit_deepst"}:
        owner.step_counter = 0
        constant_branch = owner.rewiring_mode == "constant_branch"
        resolved_synapses_per_branch = (
            K if constant_branch and K is not None else owner.synapses_per_branch
        )
        resolved_target_density = (
            float(K) / int(in_features)
            if constant_branch and K is not None
            else (
                owner.excitatory_target_density
                if synapse_type == "exc"
                else owner.inhibitory_target_density
            )
        )
        credit_kwargs = {}
        if layer_type == "credit_deepst":
            credit_kwargs = {
                "credit_trace_decay": owner.credit_trace_decay,
                "credit_candidate_pool_size": owner.credit_candidate_pool_size,
                "credit_turnover_fraction": owner.credit_turnover_fraction,
                "credit_weak_active_pool_fraction": (
                    owner.credit_weak_active_pool_fraction
                ),
                "credit_min_observations": owner.credit_min_observations,
                "credit_swap_margin": owner.credit_swap_margin,
                "credit_force_turnover": owner.credit_force_turnover,
                "credit_selection": owner.credit_selection,
                "credit_warmup_steps": owner.credit_warmup_steps,
            }
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            target_density=resolved_target_density,
            sigma=owner.sigma,
            use_noise=owner.use_noise,
            rewiring_mode=owner.rewiring_mode,
            synapses_per_branch=resolved_synapses_per_branch,
            param_space="log",
            init_method=owner.init_method,
            freeze_connectivity=(
                owner.freeze_excitatory_connectivity
                if synapse_type == "exc"
                else owner.freeze_inhibitory_connectivity
            ),
            weight_threshold=owner.weight_threshold,
            weight_transform=owner.weight_transform,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
            **credit_kwargs,
        )

    if layer_type in {"indexed", "indexed_rewire"}:
        extra_kwargs = {}
        if layer_type == "indexed_rewire":
            extra_kwargs = {
                "rewire_frequency": owner.indexed_rewire_frequency,
                "rewire_quantile": owner.indexed_rewire_quantile,
                "rewire_until_step": owner.indexed_rewire_until_step,
            }
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            weight_transform=owner.weight_transform,
            weight_norm_order=owner.topk_weight_norm_order,
            gamma=_pathway_gamma(owner, synapse_type),
            connection_indices=connection_indices,
            connection_mask=connection_mask,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            seed=_indexed_synapse_seed(owner, pathway=pathway),
            output_chunk_size=owner.indexed_output_chunk_size,
            **_indexed_runtime_kwargs(owner),
            **extra_kwargs,
        )

    if layer_type == "indexed_dynamic":
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            candidate_size=owner.indexed_candidate_size,
            selection=owner.indexed_selection,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            temperature=owner.topk_temperature,
            ultrafast=owner.topk_ultrafast,
            weight_transform=owner.weight_transform,
            weight_norm_order=owner.topk_weight_norm_order,
            gamma=_pathway_gamma(owner, synapse_type),
            connection_mask=connection_mask,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            seed=_indexed_synapse_seed(owner, pathway=pathway),
            output_chunk_size=owner.indexed_output_chunk_size,
            **_indexed_runtime_kwargs(owner),
        )

    return get_sparse_layer(
        layer_type,
        in_features,
        out_features,
        K,
        param_space="log",
        init_method=init_method,
        noise_level=noise_level,
        weight_transform=owner.weight_transform,
        weight_norm_order=owner.topk_weight_norm_order,
        gamma=_pathway_gamma(owner, synapse_type),
        forbidden_input_index_per_output=forbidden_input_index_per_output,
        connection_mask=connection_mask,
    )


def create_registered_branch_sparse_layer(
    owner: Any,
    gradient_scaler: Any,
    *,
    in_features: int,
    out_features: int,
    K: int | None,
    init_method: str,
    noise_level: float,
    synapse_type: str = "exc",
    pathway: str | None = None,
    forbidden_input_index_per_output: Any = None,
    connection_indices: Any = None,
    connection_mask: Any = None,
):
    """Create a sparse branch pathway and register its dynamic gradient scaler."""
    layer = create_branch_sparse_layer(
        owner,
        in_features=in_features,
        out_features=out_features,
        K=K,
        init_method=init_method,
        noise_level=noise_level,
        synapse_type=synapse_type,
        pathway=pathway,
        forbidden_input_index_per_output=forbidden_input_index_per_output,
        connection_indices=connection_indices,
        connection_mask=connection_mask,
    )
    gradient_scaler.register_topk_dynamic(layer)
    return layer


__all__ = [
    "create_branch_sparse_layer",
    "create_registered_branch_sparse_layer",
]

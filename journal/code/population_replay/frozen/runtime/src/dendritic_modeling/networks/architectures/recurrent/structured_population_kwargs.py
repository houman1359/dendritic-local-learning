"""PopulationConfig kwarg translators for structured recurrent E/I configs."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.conversion import normalize_sparsity_type
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_types import (
    _RecurrentReactivationOptions,
)


def _structured_population_sparsity_kwargs(
    *,
    sparsity: dict[str, Any],
    dense_to_sparse: dict[str, Any],
    indexed: dict[str, Any],
) -> dict[str, Any]:
    """Translate structured sparsity config into PopulationConfig kwargs."""
    return {
        "topk_init_method": str(sparsity.get("init_method", "xavier_normal")),
        "topk_noise_level": float(sparsity.get("noise_level", 0.0)),
        "topk_type": normalize_sparsity_type(sparsity.get("type", "standard")),
        "topk_weight_norm_order": sparsity.get("weight_norm_order"),
        "topk_gamma": float(sparsity.get("gamma", 1.0)),
        "topk_temperature": float(sparsity.get("temperature", 0.0)),
        "topk_ultrafast": bool(sparsity.get("ultrafast", False)),
        "topk_strategy": str(sparsity.get("gradient_scaling", "none")),
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
        "indexed_selection": str(indexed.get("selection", "standard")),
        "indexed_output_chunk_size": int(indexed.get("output_chunk_size", 2048)),
        "indexed_rewire_frequency": int(indexed.get("rewire_frequency", 100)),
        "indexed_rewire_quantile": float(indexed.get("rewire_quantile", 0.05)),
        "indexed_rewire_until_step": indexed.get("rewire_until_step"),
        "indexed_index_dtype": str(indexed.get("index_dtype", "int64")),
        "indexed_workspace_mb": indexed.get("workspace_mb"),
        "indexed_cache_transformed_weights": bool(
            indexed.get("cache_transformed_weights", False)
        ),
        "indexed_recompute_backward": bool(indexed.get("recompute_backward", False)),
        "indexed_projection_backend": str(indexed.get("projection_backend", "eager")),
        "indexed_persistent_indices": bool(indexed.get("persistent_indices", True)),
        "indexed_init_mode": str(indexed.get("init_mode", "per_rank")),
    }


def _structured_population_deepst_kwargs(deepst: dict[str, Any]) -> dict[str, Any]:
    """Translate structured DeepST config into PopulationConfig kwargs."""
    return {
        "excitatory_target_density": float(
            deepst.get("excitatory_target_density", 0.1)
        ),
        "inhibitory_target_density": float(
            deepst.get("inhibitory_target_density", 0.1)
        ),
        "use_noise": bool(deepst.get("use_noise", True)),
        "sigma": float(deepst.get("sigma", 0.05)),
        "rewiring_mode": str(deepst.get("rewiring_mode", "global")),
        "rewire_frequency": int(deepst.get("rewire_frequency", 1)),
        "synapses_per_branch": deepst.get("synapses_per_branch"),
        "freeze_excitatory_connectivity": bool(
            deepst.get("freeze_excitatory_connectivity", False)
        ),
        "freeze_inhibitory_connectivity": bool(
            deepst.get("freeze_inhibitory_connectivity", False)
        ),
        "init_method": str(deepst.get("init_method", "xavier_normal")),
        "weight_threshold": float(deepst.get("weight_threshold", 1e-6)),
        "credit_trace_decay": float(deepst.get("credit_trace_decay", 0.95)),
        "credit_candidate_pool_size": int(deepst.get("credit_candidate_pool_size", 8)),
        "credit_turnover_fraction": float(deepst.get("credit_turnover_fraction", 0.1)),
        "credit_weak_active_pool_fraction": float(
            deepst.get("credit_weak_active_pool_fraction", 0.5)
        ),
        "credit_min_observations": int(deepst.get("credit_min_observations", 5)),
        "credit_swap_margin": float(deepst.get("credit_swap_margin", 0.0)),
        "credit_force_turnover": bool(deepst.get("credit_force_turnover", False)),
        "credit_selection": str(deepst.get("credit_selection", "credit")),
        "credit_warmup_steps": int(deepst.get("credit_warmup_steps", 5)),
    }


def _structured_population_adaptive_init_kwargs(
    implementation: dict[str, Any],
) -> dict[str, Any]:
    """Translate adaptive-initialization implementation settings."""
    return {
        "adaptive_initialization": bool(
            implementation.get("adaptive_initialization", True)
        ),
        "adaptive_initialization_policy": str(
            implementation.get(
                "adaptive_initialization_policy",
                "preserve_shunting_center",
            )
        ),
        "adaptive_target_conductance": float(
            implementation.get("adaptive_target_conductance", 5.0)
        ),
        "initial_child_conductance": float(
            implementation.get("initial_child_conductance", 1.0)
        ),
    }


def _structured_population_dynamics_kwargs(
    dynamics: dict[str, Any],
) -> dict[str, Any]:
    """Translate somatic spiking dynamics config into PopulationConfig kwargs."""
    return {
        "dynamics_mode": str(
            dynamics.get("mode", dynamics.get("dynamics_mode", "rate"))
        ),
        "spike_threshold": float(
            dynamics.get("threshold", dynamics.get("spike_threshold", 1.0))
        ),
        "spike_reset": float(dynamics.get("reset", dynamics.get("spike_reset", 0.0))),
        "spike_tau": float(dynamics.get("tau", dynamics.get("spike_tau", 20.0))),
        "spike_refractory_steps": int(
            dynamics.get("refractory_steps", dynamics.get("spike_refractory_steps", 0))
        ),
        "spike_surrogate_beta": float(
            dynamics.get("surrogate_beta", dynamics.get("spike_surrogate_beta", 10.0))
        ),
        "spike_readout": str(
            dynamics.get("readout", dynamics.get("spike_readout", "spikes"))
        ),
        "spike_readout_tau": float(
            dynamics.get("readout_tau", dynamics.get("spike_readout_tau", 20.0))
        ),
    }


def _structured_population_dendritic_spike_kwargs(
    *,
    dendritic_spikes: dict[str, Any],
    dynamics: dict[str, Any],
) -> dict[str, Any]:
    """Translate dendritic-spike config into PopulationConfig kwargs."""
    return {
        "dendritic_spikes_enabled": bool(
            dendritic_spikes.get(
                "enabled",
                dynamics.get("dendritic_spikes_enabled", False),
            )
        ),
        "dendritic_spike_mode": str(
            dendritic_spikes.get(
                "mode",
                dynamics.get("dendritic_spike_mode", "plateau"),
            )
        ),
        "dendritic_spike_levels": dendritic_spikes.get(
            "levels",
            dynamics.get("dendritic_spike_levels", "non_soma"),
        ),
        "dendritic_spike_threshold": float(
            dendritic_spikes.get(
                "threshold",
                dynamics.get("dendritic_spike_threshold", 0.7),
            )
        ),
        "dendritic_spike_plateau_amplitude": float(
            dendritic_spikes.get(
                "plateau_amplitude",
                dynamics.get("dendritic_spike_plateau_amplitude", 1.0),
            )
        ),
        "dendritic_spike_plateau_tau": float(
            dendritic_spikes.get(
                "plateau_tau",
                dynamics.get("dendritic_spike_plateau_tau", 20.0),
            )
        ),
        "dendritic_spike_refractory_steps": int(
            dendritic_spikes.get(
                "refractory_steps",
                dynamics.get("dendritic_spike_refractory_steps", 0),
            )
        ),
        "dendritic_spike_surrogate_beta": float(
            dendritic_spikes.get(
                "surrogate_beta",
                dynamics.get("dendritic_spike_surrogate_beta", 10.0),
            )
        ),
        "dendritic_spike_propagation": str(
            dendritic_spikes.get(
                "propagation",
                dynamics.get("dendritic_spike_propagation", "additive"),
            )
        ),
    }


def _structured_population_soma_feedback_kwargs(
    *,
    soma_feedback: dict[str, Any],
    dynamics: dict[str, Any],
) -> dict[str, Any]:
    """Translate soma-feedback config into PopulationConfig kwargs."""
    return {
        "soma_feedback_enabled": bool(
            soma_feedback.get(
                "enabled",
                dynamics.get("soma_feedback_enabled", False),
            )
        ),
        "soma_feedback_mode": str(
            soma_feedback.get(
                "mode",
                dynamics.get("soma_feedback_mode", "additive"),
            )
        ),
        "soma_feedback_source": str(
            soma_feedback.get(
                "source",
                dynamics.get("soma_feedback_source", "output"),
            )
        ),
        "soma_feedback_levels": soma_feedback.get(
            "levels",
            dynamics.get("soma_feedback_levels", "non_soma"),
        ),
        "soma_feedback_strength": float(
            soma_feedback.get(
                "strength",
                dynamics.get("soma_feedback_strength", 0.1),
            )
        ),
        "soma_feedback_learnable_strength": bool(
            soma_feedback.get(
                "learnable_strength",
                dynamics.get("soma_feedback_learnable_strength", False),
            )
        ),
        "soma_feedback_per_level": bool(
            soma_feedback.get(
                "per_level",
                dynamics.get("soma_feedback_per_level", True),
            )
        ),
        "soma_feedback_init_std": float(
            soma_feedback.get(
                "init_std",
                dynamics.get("soma_feedback_init_std", 0.0),
            )
        ),
        "soma_feedback_reversal": float(
            soma_feedback.get(
                "reversal",
                dynamics.get("soma_feedback_reversal", 1.0),
            )
        ),
    }


def _structured_population_morphology_kwargs(
    *,
    morphology: dict[str, Any],
    use_shunting: bool,
    use_additive_normalization: bool,
) -> dict[str, Any]:
    """Translate morphology config into PopulationConfig kwargs."""
    return {
        "weight_transform": str(morphology.get("weight_transform", "softplus")),
        "use_shunting": use_shunting,
        "use_additive_normalization": use_additive_normalization,
        "additive_mode": str(morphology.get("additive_mode", "raw")),
        "additive_tangent_n0": morphology.get("additive_tangent_n0"),
        "additive_tangent_t0": morphology.get("additive_tangent_t0"),
        "additive_tangent_n0_by_depth": list(
            morphology.get("additive_tangent_n0_by_depth", [])
        ),
        "additive_tangent_t0_by_depth": list(
            morphology.get("additive_tangent_t0_by_depth", [])
        ),
        "allow_self_recurrence": bool(morphology.get("allow_self_recurrence", True)),
        "somatic_synapses": bool(morphology.get("somatic_synapses", False)),
        "dbl_init_method": str(
            morphology.get("dbl_init_method", "analytical_expectation")
        ),
    }


def _structured_population_reactivation_kwargs(
    *,
    reactivation: dict[str, Any],
    reactivation_options: _RecurrentReactivationOptions,
) -> dict[str, Any]:
    """Translate reactivation config into PopulationConfig kwargs."""
    return {
        "reactivate": reactivation_options.reactivate,
        "reactivation_type": reactivation_options.reactivation_type,
        "reactivation_init_m": reactivation_options.init_m,
        "reactivation_init_b": reactivation_options.init_b,
        "reactivation_init_policy": reactivation_options.init_policy,
        "reactivation_occupancy_quantile_low": reactivation.get(
            "occupancy_quantile_low"
        ),
        "reactivation_occupancy_quantile_high": reactivation.get(
            "occupancy_quantile_high"
        ),
        "reactivation_occupancy_target_low": reactivation.get("occupancy_target_low"),
        "reactivation_occupancy_target_high": reactivation.get("occupancy_target_high"),
        "reactivation_calibration_min_quantile_width": float(
            reactivation.get("calibration_min_quantile_width", 1e-3)
        ),
        "reactivation_calibration_max_m": float(
            reactivation.get("calibration_max_m", 50.0)
        ),
        "reactivation_calibration_revert_on_invalid": bool(
            reactivation.get("calibration_revert_on_invalid", True)
        ),
        "reactivation_sigma_aware_k": float(reactivation.get("sigma_aware_k", 0.25)),
        "reactivation_strategy": str(reactivation.get("gradient_scaling", "none")),
        "reactivation_memory_efficient": bool(
            reactivation.get("memory_efficient", False)
        ),
    }


def _structured_population_blocklinear_kwargs(
    *,
    blocklinear: dict[str, Any],
    implementation: dict[str, Any],
) -> dict[str, Any]:
    """Translate BlockLinear implementation settings into PopulationConfig kwargs."""
    return {
        "blocklinear_strategy": str(blocklinear.get("gradient_scaling", "none")),
        "efficient_blocklinear": bool(blocklinear.get("efficient", False)),
        "print_hooks": bool(implementation.get("print_hooks", False)),
    }


def _structured_population_timing_kwargs(
    *,
    recurrent_cfg: dict[str, Any],
    tau_base: float,
    tau_ratio: float,
) -> dict[str, Any]:
    """Translate recurrent timing config into PopulationConfig kwargs."""
    return {
        "level_taus": list(recurrent_cfg.get("level_taus", [])),
        "tau_base": tau_base,
        "tau_ratio": tau_ratio,
        "learnable_tau": bool(recurrent_cfg.get("learnable_tau", False)),
        "epsilon": float(recurrent_cfg.get("epsilon", 1e-8)),
    }


__all__ = [
    "_structured_population_adaptive_init_kwargs",
    "_structured_population_blocklinear_kwargs",
    "_structured_population_deepst_kwargs",
    "_structured_population_dendritic_spike_kwargs",
    "_structured_population_dynamics_kwargs",
    "_structured_population_morphology_kwargs",
    "_structured_population_reactivation_kwargs",
    "_structured_population_soma_feedback_kwargs",
    "_structured_population_sparsity_kwargs",
    "_structured_population_timing_kwargs",
]

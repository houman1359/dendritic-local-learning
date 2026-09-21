"""
General utilities for dendritic modeling.

Common utility functions for data serialization, logging, seed management,
and configuration validation.
"""

from __future__ import annotations

import importlib
from typing import Any

# Importing ``dendritic_modeling`` initializes logging through this package.
# Keep the namespace lazy so that lightweight imports do not transitively load
# configuration validation, network architectures, PyTorch, and torchvision.
_ATTRIBUTE_MODULES = {
    "ForwardHookRemovalMixin": "hooks",
    "HookHandle": "hooks",
    "ModelArchitectureValidator": "model_architecture_validator",
    "ResolvedExperimentSeeds": "reproducibility",
    "Shaper": "general",
    "apply_automatic_safeguards": "training_stability_validator",
    "apply_nonnegative_lda": "lda",
    "apply_random_weights": "lda",
    "calculate_effective_sparsity": "sparse_ops",
    "compute_lda_importance_scores": "lda",
    "convert_to_serializable": "general",
    "count_active_parameters": "sparse_ops",
    "estimate_memory_usage": "sparse_ops",
    "fit_nonnegative_lda": "lda",
    "generate_random_nonnegative_weights": "lda",
    "get_available_sparse_types": "sparse_ops",
    "get_logger": "logging_config",
    "get_sparse_layer": "sparse_ops",
    "get_validation_summary": "config_validation",
    "isolated_random_seed": "reproducibility",
    "iter_child_modules_of_type": "hooks",
    "iter_modules_matching": "hooks",
    "iter_modules_of_type": "hooks",
    "iter_named_modules_matching": "hooks",
    "iter_named_modules_of_type": "hooks",
    "normalize_sparse_layer_type": "sparse_ops",
    "parallel_topk_selection": "sparse_ops",
    "preserved_random_state": "reproducibility",
    "register_forward_hook_groups": "hooks",
    "register_hook_groups": "hooks",
    "register_named_forward_hook_groups": "hooks",
    "register_named_forward_hooks": "hooks",
    "remove_hook_handles": "hooks",
    "resolve_experiment_seeds": "reproducibility",
    "roc_auc_score": "math",
    "run_with_forward_hooks": "hooks",
    "save_dict": "general",
    "select_topk_features_by_lda": "lda",
    "set_seed": "general",
    "setup_logging": "logging_config",
    "suggest_optimal_config_for_model_size": "training_stability_validator",
    "validate_architecture_only": "config_validation",
    "validate_full_config": "config_validation",
    "validate_model_architecture": "model_architecture_validator",
    "validate_training_only": "config_validation",
    "validate_training_stability": "training_stability_validator",
}


def __getattr__(name: str) -> Any:
    """Import a public utility on first access."""
    module_name = _ATTRIBUTE_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals()) + list(_ATTRIBUTE_MODULES))


__all__ = [
    "ForwardHookRemovalMixin",
    "HookHandle",
    "ModelArchitectureValidator",
    "ResolvedExperimentSeeds",
    "Shaper",
    "apply_automatic_safeguards",
    "apply_nonnegative_lda",
    "apply_random_weights",
    "compute_lda_importance_scores",
    "convert_to_serializable",
    "fit_nonnegative_lda",
    "generate_random_nonnegative_weights",
    "get_logger",
    "get_validation_summary",
    "isolated_random_seed",
    "iter_child_modules_of_type",
    "iter_modules_matching",
    "iter_modules_of_type",
    "iter_named_modules_matching",
    "iter_named_modules_of_type",
    "normalize_sparse_layer_type",
    "preserved_random_state",
    "register_forward_hook_groups",
    "register_hook_groups",
    "register_named_forward_hook_groups",
    "register_named_forward_hooks",
    "remove_hook_handles",
    "resolve_experiment_seeds",
    "roc_auc_score",
    "run_with_forward_hooks",
    "save_dict",
    "select_topk_features_by_lda",
    "set_seed",
    "setup_logging",
    "suggest_optimal_config_for_model_size",
    "validate_architecture_only",
    "validate_full_config",
    "validate_model_architecture",
    "validate_training_only",
    "validate_training_stability",
]

"""Shared option parsing for unified dataset loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

_SEQUENCE_DATASET_NAMES = (
    "sequential_mnist",
    "copy_task",
    "adding_problem",
    "variable_delay_dms",
    "speeded_distractor_dms",
    "speeded_distractor_dms_positive",
    "multi_frequency",
    "hierarchical_temporal",
    "bayesian_ready_set_go",
    "bayesian_ready_set_go_positive",
    "factual_bayesian_ready_set_go",
    "factual_bayesian_ready_set_go_positive",
    "ready_set_go",
    "ready_set_go_positive",
    "contextual_ready_set_go",
    "contextual_ready_set_go_positive",
    "context_dependent_decision",
    "context_dependent_decision_hard",
    "context_dependent_decision_positive",
    "context_dependent_decision_hard_positive",
    "switching_context_decision",
    "switching_context_decision_positive",
    "timescale_generalization",
    "timescale_generalization_positive",
    "odr_generalization",
    "odr_generalization_positive",
    "romo_delay_comparison",
    "romo_delay_comparison_positive",
    "gain_modulated_contextual_comparison",
    "gain_modulated_contextual_comparison_positive",
    "lorenz",
    "lorenz_sequence",
    "switching_lds",
    "multi_sine_forecast",
    "stringer_v1",
)

_THEORETICAL_SYNTHETIC_DATASET_NAMES = (
    "correlated_gaussian",
    "nonlinear_interaction",
    "linear_separable",
    "nonlinear_separable",
    "block_correlated",
    "balanced_information",
    "contextual_stream_gain_shift",
    "branch_local_gain_load",
    "hierarchical_gain_load",
)

_EXPERIMENTAL_DATASET_IMPORT_CANDIDATES = (
    "experiments.data_generation.synthetic_datasets",
    "data_generation.synthetic_datasets",
)

_DatasetTriplet = tuple[Dataset, Dataset, Dataset]


@dataclass(frozen=True)
class _UnifiedDatasetOptions:
    dataset_name: str
    task_data_path: str | None
    parameters: dict[str, Any]
    train_valid_split: float
    flatten: bool
    normalize: bool
    base_dataset: str
    poisson_sampling: bool
    stimulus_duration: float
    multiplicative_gain: bool
    fixed_gain_factor: Any
    max_gain_factor: Any
    max_gain_tau_ratio: Any
    uniform_gain: bool
    gain_sampling: str
    split_seed: int
    label_noise_rate: float
    label_noise_seed: int


def _get_config_value(config: dict | object, key: str, default: Any = None) -> Any:
    """Safely get a value from a config object or mapping."""
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_unified_dataset_options(task_cfg: Any) -> _UnifiedDatasetOptions:
    dataset_name = _get_config_value(task_cfg, "dataset")
    task_data_path = _get_config_value(task_cfg, "data_path", None)
    parameters = _get_config_value(task_cfg, "parameters", {})
    if task_data_path == "":
        task_data_path = None

    return _UnifiedDatasetOptions(
        dataset_name=dataset_name,
        task_data_path=task_data_path,
        parameters=parameters,
        train_valid_split=_get_config_value(task_cfg, "train_valid_split", 1.0),
        flatten=parameters.get("flatten", True),
        normalize=parameters.get("normalize", False),
        base_dataset=parameters.get("base_dataset", "mnist"),
        poisson_sampling=parameters.get("poisson_sampling", True),
        stimulus_duration=parameters.get("stimulus_duration", 1.0),
        multiplicative_gain=parameters.get("multiplicative_gain", True),
        fixed_gain_factor=parameters.get("fixed_gain_factor", None),
        max_gain_factor=parameters.get("max_gain_factor", None),
        max_gain_tau_ratio=parameters.get("max_gain_tau_ratio", None),
        uniform_gain=parameters.get("uniform_gain", True),
        gain_sampling=parameters.get("gain_sampling", "log_uniform"),
        split_seed=int(parameters.get("split_seed", parameters.get("seed", 0)) or 0),
        label_noise_rate=parameters.get("label_noise_rate", 0.0),
        label_noise_seed=parameters.get("label_noise_seed", 0),
    )


__all__ = [
    "_EXPERIMENTAL_DATASET_IMPORT_CANDIDATES",
    "_SEQUENCE_DATASET_NAMES",
    "_THEORETICAL_SYNTHETIC_DATASET_NAMES",
    "_DatasetTriplet",
    "_UnifiedDatasetOptions",
    "_get_config_value",
    "_resolve_unified_dataset_options",
]

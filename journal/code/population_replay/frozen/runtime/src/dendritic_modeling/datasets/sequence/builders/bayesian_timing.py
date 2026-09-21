"""Builders for Bayesian interval-reproduction datasets."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from dendritic_modeling.datasets.sequence.bayesian_timing import (
    BayesianReadySetGoDataset,
    FactualBayesianReadySetGoDataset,
    default_observed_interval_bounds,
    derive_factual_nested_distractor_seed,
    finite_gaussian_motor_kernel,
)
from dendritic_modeling.datasets.sequence.builders.common import (
    SequenceDatasetBuild,
    require_distinct_explicit_split_seeds,
    validate_noop_standard_processing,
    validated_split_overrides,
)

_SPLIT_NAMES = ("train", "valid", "test")


def _validated_even_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("Bayesian Ready-Set-Go split sizes must be positive integers")
    if int(value) % 2:
        raise ValueError(
            "Bayesian Ready-Set-Go split sizes must be even for paired trials"
        )
    return int(value)


def _validated_positive_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(
            "Factual Bayesian Ready-Set-Go split sizes must be positive integers"
        )
    return int(value)


def _split_default_size(n_train: int) -> int:
    size = max(n_train // 5, 2000)
    return size + size % 2


def _base_parameters(dataset_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "prior_supports": tuple(
            tuple(support)
            for support in kwargs.get(
                "prior_supports",
                ((24, 28, 32, 36, 40), (40, 45, 50, 55, 60)),
            )
        ),
        "prior_probabilities": kwargs.get("prior_probabilities"),
        "weber_fraction": kwargs.get("weber_fraction", 0.12),
        "measurement_noise_floor": kwargs.get("measurement_noise_floor", 0.0),
        "likelihood_tail_sigma": kwargs.get("likelihood_tail_sigma", 4.0),
        "production_scale": kwargs.get("production_scale", 1.0),
        "pulse_duration": kwargs.get("pulse_duration", 2),
        "context_start_step": kwargs.get("context_start_step", 0),
        "context_duration": kwargs.get("context_duration", 10),
        "context_gain": kwargs.get("context_gain", 1.0),
        "target_sigma": kwargs.get("target_sigma", 2.0),
        "target_tail_sigma": kwargs.get("target_tail_sigma", 4.0),
        "post_go_duration": kwargs.get("post_go_duration", 20),
        "cue_gain_min": kwargs.get("cue_gain_min", 0.7),
        "cue_gain_max": kwargs.get("cue_gain_max", 1.3),
        "input_noise_std": kwargs.get("input_noise_std", 0.05),
        "mean_distractors_per_trial": kwargs.get("mean_distractors_per_trial", 1.0),
        "distractor_amplitude": kwargs.get("distractor_amplitude", 1.0),
        "distractor_pulse_duration": kwargs.get("distractor_pulse_duration", 2),
        "positive_input_encoding": dataset_name.endswith("_positive")
        or kwargs.get("positive_input_encoding", False),
    }


def build_bayesian_ready_set_go(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    """Build seeded Bayesian RSG splits with shared timing geometry."""

    validate_noop_standard_processing(
        kwargs,
        dataset_label="Bayesian Ready-Set-Go",
    )

    n_train = _validated_even_size(kwargs.get("n_samples", 10000))
    base_seed = kwargs.get("seed", 0)
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise TypeError("Bayesian Ready-Set-Go seed must be an integer")
    base = _base_parameters(dataset_name, kwargs)
    split_overrides = validated_split_overrides(
        kwargs,
        dataset_label="Bayesian Ready-Set-Go",
    )
    forbidden = {
        "context_duration",
        "context_start_step",
        "likelihood_tail_sigma",
        "measurement_noise_floor",
        "observed_interval_max",
        "observed_interval_min",
        "prior_probabilities",
        "prior_supports",
        "production_scale",
        "sequence_duration",
        "set_max_step",
        "set_min_step",
        "target_sigma",
        "target_tail_sigma",
        "weber_fraction",
    }
    if any(forbidden.intersection(override) for override in split_overrides.values()):
        raise ValueError(
            "Bayesian observer semantics and absolute timing must be shared across "
            "Ready-Set-Go splits; categorical prior cues cannot specify changed "
            "prior or likelihood parameters"
        )

    default_sizes = {
        "train": n_train,
        "valid": _split_default_size(n_train),
        "test": _split_default_size(n_train),
    }
    parameters_by_split: dict[str, tuple[int, dict[str, Any]]] = {}
    seed_offsets = (0, 1, 2) if split_overrides else (0, 1, 1)
    for seed_offset, split_name in zip(seed_offsets, _SPLIT_NAMES, strict=True):
        override = dict(split_overrides.get(split_name, {}))
        n_split = _validated_even_size(
            override.pop("n_samples", default_sizes[split_name])
        )
        parameters = {**base, **override}
        parameters.setdefault("seed", int(base_seed) + seed_offset)
        parameters_by_split[split_name] = (n_split, parameters)
    if split_overrides:
        require_distinct_explicit_split_seeds(
            {
                name: parameters["seed"]
                for name, (_size, parameters) in parameters_by_split.items()
            },
            dataset_label="Bayesian Ready-Set-Go",
        )

    derived_bounds = tuple(
        default_observed_interval_bounds(
            parameters["prior_supports"],
            weber_fraction=parameters["weber_fraction"],
            measurement_noise_floor=parameters["measurement_noise_floor"],
            likelihood_tail_sigma=parameters["likelihood_tail_sigma"],
            minimum_interval=parameters["pulse_duration"],
        )
        for _, parameters in parameters_by_split.values()
    )
    observed_min = kwargs.get(
        "observed_interval_min", min(bounds[0] for bounds in derived_bounds)
    )
    observed_max = kwargs.get(
        "observed_interval_max", max(bounds[1] for bounds in derived_bounds)
    )
    configured_set_min = kwargs.get("set_min_step")
    set_min_step = (
        observed_max + base["context_start_step"] + base["context_duration"] + 10
        if configured_set_min is None
        else configured_set_min
    )
    configured_set_max = kwargs.get("set_max_step")
    default_translation_span = max(40, observed_max - observed_min)
    set_max_step = (
        set_min_step + default_translation_span
        if configured_set_max is None
        else configured_set_max
    )

    required_duration = max(
        set_max_step
        + max(
            round(parameters["production_scale"] * interval)
            for support in parameters["prior_supports"]
            for interval in support
        )
        + parameters["post_go_duration"]
        + 1
        for _, parameters in parameters_by_split.values()
    )
    configured_duration = kwargs.get("sequence_duration")
    sequence_duration = (
        required_duration if configured_duration is None else configured_duration
    )
    shared_geometry = {
        "observed_interval_min": observed_min,
        "observed_interval_max": observed_max,
        "set_min_step": set_min_step,
        "set_max_step": set_max_step,
        "sequence_duration": sequence_duration,
    }

    def construct(split_name: str) -> BayesianReadySetGoDataset:
        n_split, parameters = parameters_by_split[split_name]
        return BayesianReadySetGoDataset(
            n_samples=n_split,
            **parameters,
            **shared_geometry,
        )

    if split_overrides:
        return SequenceDatasetBuild(
            split_datasets=tuple(construct(name) for name in _SPLIT_NAMES)
        )
    return SequenceDatasetBuild(
        train_full=construct("train"),
        test_ds=construct("test"),
    )


def _factual_base_parameters(
    dataset_name: str, kwargs: dict[str, Any]
) -> dict[str, Any]:
    return {
        "prior_supports": tuple(
            tuple(support)
            for support in kwargs.get(
                "prior_supports",
                ((24, 28, 32, 36, 40), (40, 45, 50, 55, 60)),
            )
        ),
        "prior_probabilities": kwargs.get("prior_probabilities"),
        "weber_fraction": kwargs.get("weber_fraction", 0.12),
        "measurement_noise_floor": kwargs.get("measurement_noise_floor", 0.0),
        "likelihood_tail_sigma": kwargs.get("likelihood_tail_sigma", 4.0),
        "production_scale": kwargs.get("production_scale", 1.0),
        "motor_sigma": kwargs.get("motor_sigma", 2.0),
        "motor_tail_sigma": kwargs.get("motor_tail_sigma", 4.0),
        "pulse_duration": kwargs.get("pulse_duration", 2),
        "context_start_step": kwargs.get("context_start_step", 0),
        "context_duration": kwargs.get("context_duration", 10),
        "context_gain": kwargs.get("context_gain", 1.0),
        "post_go_duration": kwargs.get("post_go_duration", 0),
        "cue_gain_min": kwargs.get("cue_gain_min", 0.7),
        "cue_gain_max": kwargs.get("cue_gain_max", 1.3),
        "input_noise_std": kwargs.get("input_noise_std", 0.05),
        "mean_distractors_per_trial": kwargs.get("mean_distractors_per_trial", 1.0),
        "distractor_amplitude": kwargs.get("distractor_amplitude", 1.0),
        "distractor_pulse_duration": kwargs.get("distractor_pulse_duration", 2),
        "nested_distractor_pairs_per_trial": kwargs.get(
            "nested_distractor_pairs_per_trial", 0
        ),
        "nested_distractor_interval_values": tuple(
            kwargs.get("nested_distractor_interval_values", (3, 4, 5))
        ),
        "nested_distractor_clearance": kwargs.get("nested_distractor_clearance", 2),
        "positive_input_encoding": dataset_name.endswith("_positive")
        or kwargs.get("positive_input_encoding", False),
    }


def build_factual_bayesian_ready_set_go(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    """Build independent factual RSG train/validation/test datasets."""

    label = "Factual Bayesian Ready-Set-Go"
    validate_noop_standard_processing(kwargs, dataset_label=label)
    n_train = _validated_positive_size(kwargs.get("n_samples", 10000))
    base_seed = kwargs.get("seed", 0)
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise TypeError(f"{label} seed must be an integer")

    base = _factual_base_parameters(dataset_name, kwargs)
    split_overrides = validated_split_overrides(kwargs, dataset_label=label)
    forbidden = {
        "context_duration",
        "context_start_step",
        "likelihood_tail_sigma",
        "measurement_noise_floor",
        "motor_sigma",
        "motor_tail_sigma",
        "nested_distractor_clearance",
        "nested_distractor_interval_values",
        "nested_distractor_pairs_per_trial",
        "observed_interval_max",
        "observed_interval_min",
        "post_go_duration",
        "prior_probabilities",
        "prior_supports",
        "production_scale",
        "pulse_duration",
        "sequence_duration",
        "set_max_step",
        "set_min_step",
        "set_step_values",
        "weber_fraction",
    }
    if any(forbidden.intersection(override) for override in split_overrides.values()):
        raise ValueError(
            "Factual observer, motor-kernel, and absolute timing semantics must "
            "be shared across Ready-Set-Go splits"
        )

    default_sizes = {
        "train": n_train,
        "valid": max(n_train // 5, 2000),
        "test": max(n_train // 5, 2000),
    }
    base_motor_seed = kwargs.get("motor_seed")
    if base_motor_seed is not None and (
        isinstance(base_motor_seed, bool) or not isinstance(base_motor_seed, Integral)
    ):
        raise TypeError(f"{label} motor_seed must be an integer or None")
    base_nested_seed = kwargs.get("nested_distractor_seed")
    if base_nested_seed is not None and (
        isinstance(base_nested_seed, bool) or not isinstance(base_nested_seed, Integral)
    ):
        raise TypeError(f"{label} nested_distractor_seed must be an integer or None")
    parameters_by_split: dict[str, tuple[int, dict[str, Any]]] = {}
    for seed_offset, split_name in enumerate(_SPLIT_NAMES):
        override = dict(split_overrides.get(split_name, {}))
        n_split = _validated_positive_size(
            override.pop("n_samples", default_sizes[split_name])
        )
        parameters = {**base, **override}
        parameters.setdefault("seed", int(base_seed) + seed_offset)
        if "motor_seed" not in parameters:
            parameters["motor_seed"] = (
                None if base_motor_seed is None else int(base_motor_seed) + seed_offset
            )
        if "nested_distractor_seed" not in parameters:
            parameters["nested_distractor_seed"] = (
                None
                if base_nested_seed is None
                else int(base_nested_seed) + seed_offset
            )
        parameters_by_split[split_name] = (n_split, parameters)
    require_distinct_explicit_split_seeds(
        {
            name: parameters["seed"]
            for name, (_size, parameters) in parameters_by_split.items()
        },
        dataset_label=label,
    )
    effective_motor_seeds = {}
    effective_nested_seeds = {}
    for name, (_size, parameters) in parameters_by_split.items():
        candidate = parameters["motor_seed"]
        if candidate is not None and (
            isinstance(candidate, bool) or not isinstance(candidate, Integral)
        ):
            raise TypeError(f"{label} {name} motor_seed must be an integer or None")
        effective_motor_seeds[name] = (
            (int(parameters["seed"]) * 6364136223846793005 + 1442695040888963407)
            % (2**63 - 1)
            if candidate is None
            else int(candidate)
        )
        nested_candidate = parameters["nested_distractor_seed"]
        if nested_candidate is not None and (
            isinstance(nested_candidate, bool)
            or not isinstance(nested_candidate, Integral)
        ):
            raise TypeError(
                f"{label} {name} nested_distractor_seed must be an integer or None"
            )
        effective_nested_seeds[name] = (
            derive_factual_nested_distractor_seed(int(parameters["seed"]))
            if nested_candidate is None
            else int(nested_candidate)
        )
    require_distinct_explicit_split_seeds(
        effective_motor_seeds,
        dataset_label=f"{label} motor-outcome",
    )
    require_distinct_explicit_split_seeds(
        effective_nested_seeds,
        dataset_label=f"{label} nested-distractor",
    )
    data_seeds = {
        int(parameters["seed"]) for _size, parameters in parameters_by_split.values()
    }
    if data_seeds.intersection(effective_motor_seeds.values()):
        raise ValueError(f"{label} trial-data and motor-outcome seeds must be disjoint")
    if data_seeds.intersection(effective_nested_seeds.values()) or set(
        effective_motor_seeds.values()
    ).intersection(effective_nested_seeds.values()):
        raise ValueError(
            f"{label} nested-distractor seeds must be disjoint from trial-data "
            "and motor-outcome seeds"
        )

    derived_bounds = tuple(
        default_observed_interval_bounds(
            parameters["prior_supports"],
            weber_fraction=parameters["weber_fraction"],
            measurement_noise_floor=parameters["measurement_noise_floor"],
            likelihood_tail_sigma=parameters["likelihood_tail_sigma"],
            minimum_interval=parameters["pulse_duration"],
        )
        for _, parameters in parameters_by_split.values()
    )
    observed_min = kwargs.get(
        "observed_interval_min", min(bounds[0] for bounds in derived_bounds)
    )
    observed_max = kwargs.get(
        "observed_interval_max", max(bounds[1] for bounds in derived_bounds)
    )
    configured_set_values = kwargs.get("set_step_values")
    if configured_set_values is not None:
        if (
            kwargs.get("set_min_step") is not None
            or kwargs.get("set_max_step") is not None
        ):
            raise ValueError(
                "set_step_values is mutually exclusive with Set-time bounds"
            )
        set_step_values = tuple(configured_set_values)
        if not set_step_values:
            raise ValueError("set_step_values must not be empty")
        maximum_set_step = max(set_step_values)
        shared_set_geometry = {"set_step_values": set_step_values}
    else:
        configured_set_min = kwargs.get("set_min_step")
        set_min_step = (
            observed_max + base["context_start_step"] + base["context_duration"] + 10
            if configured_set_min is None
            else configured_set_min
        )
        configured_set_max = kwargs.get("set_max_step")
        default_translation_span = max(40, observed_max - observed_min)
        set_max_step = (
            set_min_step + default_translation_span
            if configured_set_max is None
            else configured_set_max
        )
        maximum_set_step = set_max_step
        shared_set_geometry = {
            "set_min_step": set_min_step,
            "set_max_step": set_max_step,
        }

    maximum_production = max(
        round(parameters["production_scale"] * interval)
        + int(
            finite_gaussian_motor_kernel(
                sigma=parameters["motor_sigma"],
                tail_sigma=parameters["motor_tail_sigma"],
            )[0]
            .abs()
            .max()
        )
        for _, parameters in parameters_by_split.values()
        for support in parameters["prior_supports"]
        for interval in support
    )
    required_duration = max(
        maximum_set_step + maximum_production + parameters["post_go_duration"] + 1
        for _, parameters in parameters_by_split.values()
    )
    configured_duration = kwargs.get("sequence_duration")
    sequence_duration = (
        required_duration if configured_duration is None else configured_duration
    )
    shared_geometry = {
        "observed_interval_min": observed_min,
        "observed_interval_max": observed_max,
        "sequence_duration": sequence_duration,
        **shared_set_geometry,
    }

    def construct(split_name: str) -> FactualBayesianReadySetGoDataset:
        n_split, parameters = parameters_by_split[split_name]
        return FactualBayesianReadySetGoDataset(
            n_samples=n_split,
            **parameters,
            **shared_geometry,
        )

    return SequenceDatasetBuild(
        split_datasets=tuple(construct(name) for name in _SPLIT_NAMES)
    )


BAYESIAN_TIMING_SEQUENCE_BUILDERS = {
    "bayesian_ready_set_go": build_bayesian_ready_set_go,
    "bayesian_ready_set_go_positive": build_bayesian_ready_set_go,
    "factual_bayesian_ready_set_go": build_factual_bayesian_ready_set_go,
    "factual_bayesian_ready_set_go_positive": build_factual_bayesian_ready_set_go,
}


__all__ = [
    "BAYESIAN_TIMING_SEQUENCE_BUILDERS",
    "build_bayesian_ready_set_go",
    "build_factual_bayesian_ready_set_go",
]

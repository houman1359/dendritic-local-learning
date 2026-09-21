"""Builders for decision and temporal-generalization sequence datasets."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from dendritic_modeling.datasets.sequence.builders.common import (
    SequenceDatasetBuild,
    build_deferred_train_test_split,
    build_explicit_or_deferred_split,
    positive_input_encoding,
    require_distinct_explicit_split_seeds,
    validate_noop_standard_processing,
    validated_split_overrides,
)
from dendritic_modeling.datasets.sequence.decision_tasks import (
    ContextDependentDecisionDataset,
    ContextualReadySetGoDataset,
    HierarchicalTemporalDataset,
    ReadySetGoDataset,
    SwitchingContextDecisionDataset,
    TimescaleGeneralizationDataset,
)


def build_hierarchical_temporal(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common = {
        "n_contexts": kwargs.get("n_contexts", 2),
        "n_block_types": kwargs.get("n_block_types", 4),
        "n_blocks": kwargs.get("n_blocks", 8),
        "context_duration": kwargs.get("context_duration", 20),
        "block_duration": kwargs.get("block_duration", 25),
        "signal_dim": kwargs.get("signal_dim", 16),
        "noise_std": kwargs.get("noise_std", 0.5),
    }
    return build_deferred_train_test_split(
        dataset_cls=HierarchicalTemporalDataset,
        n_train=n_train,
        common_defaults=common,
        min_test_size=2000,
    )


def build_context_dependent_decision(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common = {
        "signal_dim": kwargs.get("signal_dim", 8),
        "context_duration": kwargs.get("context_duration", 10),
        "integration_duration": kwargs.get("integration_duration", 50),
        "coherence_levels": tuple(
            kwargs.get("coherence_levels", (0.05, 0.1, 0.2, 0.4))
        ),
        "noise_std": kwargs.get("noise_std", 1.0),
        "variable_delay": kwargs.get("variable_delay", True),
        "min_integration": kwargs.get("min_integration", 20),
        "max_integration": kwargs.get("max_integration", 80),
        "delay_duration": kwargs.get("delay_duration", 0),
        "min_delay": kwargs.get("min_delay", None),
        "max_delay": kwargs.get("max_delay", None),
        "conflict_fraction": kwargs.get("conflict_fraction", 0.0),
        "distractor_burst_prob": kwargs.get("distractor_burst_prob", 0.0),
        "distractor_burst_scale": kwargs.get("distractor_burst_scale", 2.0),
        "distractor_burst_duration": kwargs.get("distractor_burst_duration", 5),
        "cue_dropout_prob": kwargs.get("cue_dropout_prob", 0.0),
        "cue_keep_duration": kwargs.get("cue_keep_duration", 1),
        "late_context_switch_prob": kwargs.get("late_context_switch_prob", 0.0),
        "context_signal_scale": kwargs.get("context_signal_scale", 1.0),
        "return_seq_lengths": kwargs.get("return_seq_lengths", False),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
    }
    return build_deferred_train_test_split(
        dataset_cls=ContextDependentDecisionDataset,
        n_train=n_train,
        common_defaults=common,
        min_test_size=2000,
    )


def build_switching_context_decision(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 16000)
    common = {
        "signal_dim": kwargs.get("signal_dim", 8),
        "seq_len": kwargs.get("seq_len", 360),
        "trial_duration": kwargs.get("trial_duration", 12),
        "min_context_duration": kwargs.get("min_context_duration", 90),
        "max_context_duration": kwargs.get("max_context_duration", 150),
        "coherence_levels": tuple(
            kwargs.get("coherence_levels", (0.03, 0.06, 0.12, 0.24))
        ),
        "noise_std": kwargs.get("noise_std", 1.1),
        "conflict_fraction": kwargs.get("conflict_fraction", 0.75),
        "distractor_burst_prob": kwargs.get("distractor_burst_prob", 0.35),
        "distractor_burst_scale": kwargs.get("distractor_burst_scale", 2.0),
        "distractor_burst_duration": kwargs.get("distractor_burst_duration", 3),
        "context_cue_duration": kwargs.get("context_cue_duration", 10),
        "switch_cue_duration": kwargs.get("switch_cue_duration", 4),
        "context_signal_scale": kwargs.get("context_signal_scale", 1.0),
        "switch_signal_scale": kwargs.get("switch_signal_scale", 0.35),
        "min_blocks": kwargs.get("min_blocks", 2),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
    }
    return build_deferred_train_test_split(
        dataset_cls=SwitchingContextDecisionDataset,
        n_train=n_train,
        common_defaults=common,
        min_test_size=2000,
    )


def build_timescale_generalization(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 20000)
    common_defaults = {
        "stimulus_dim": kwargs.get("stimulus_dim", 16),
        "stimulus_duration": kwargs.get("stimulus_duration", 10),
        "min_delay": kwargs.get("min_delay", 10),
        "max_delay": kwargs.get("max_delay", 100),
        "probe_duration": kwargs.get("probe_duration", 10),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
        "noise_std": kwargs.get("noise_std", 0.3),
    }
    return build_explicit_or_deferred_split(
        dataset_cls=TimescaleGeneralizationDataset,
        kwargs=kwargs,
        common_defaults=common_defaults,
        n_train=n_train,
        min_test_size=2000,
    )


def build_ready_set_go(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    """Build independently seeded Ready-Set-Go splits on one fixed horizon."""

    validate_noop_standard_processing(kwargs, dataset_label="Ready-Set-Go")

    n_train = kwargs.get("n_samples", 10000)
    default_intervals = tuple(
        kwargs.get("sample_intervals", (40, 50, 60, 70, 80, 90, 100))
    )
    split_names = ("train", "valid", "test")
    split_overrides = validated_split_overrides(
        kwargs,
        dataset_label="Ready-Set-Go",
    )
    all_intervals = list(default_intervals)
    for override in split_overrides.values():
        all_intervals.extend(override.get("sample_intervals", ()))
    default_set_min = max(all_intervals) + 10
    set_min_step = kwargs.get("set_min_step", default_set_min)
    set_max_step = kwargs.get("set_max_step", set_min_step + 20)
    common_defaults = {
        "sample_intervals": default_intervals,
        "production_scale": kwargs.get("production_scale", 1.0),
        "set_min_step": set_min_step,
        "set_max_step": set_max_step,
        "pulse_duration": kwargs.get("pulse_duration", 2),
        "target_sigma": kwargs.get("target_sigma", 2.0),
        "post_go_duration": kwargs.get("post_go_duration", 20),
        "cue_gain_min": kwargs.get("cue_gain_min", 0.7),
        "cue_gain_max": kwargs.get("cue_gain_max", 1.3),
        "input_noise_std": kwargs.get("input_noise_std", 0.05),
        "mean_distractors_per_trial": kwargs.get("mean_distractors_per_trial", 1.0),
        "distractor_amplitude": kwargs.get("distractor_amplitude", 1.0),
        "distractor_pulse_duration": kwargs.get("distractor_pulse_duration", 2),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
    }
    base_seed = kwargs.get("seed", 0)
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise TypeError("Ready-Set-Go seed must be an integer")

    if split_overrides:
        forbidden_split_geometry = {
            "sequence_duration",
            "set_max_step",
            "set_min_step",
        }
        if any(
            forbidden_split_geometry.intersection(override)
            for override in split_overrides.values()
        ):
            raise ValueError(
                "sequence_duration and Set-time bounds must be shared across splits"
            )
        default_sizes = {
            "train": n_train,
            "valid": max(n_train // 5, 2000),
            "test": max(n_train // 5, 2000),
        }
        split_params: dict[str, tuple[int, dict[str, Any]]] = {}
        for seed_offset, split_name in enumerate(split_names):
            override = dict(split_overrides.get(split_name, {}))
            n_split = override.pop("n_samples", default_sizes[split_name])
            params = {**common_defaults, **override}
            params.setdefault("seed", base_seed + seed_offset)
            split_params[split_name] = (n_split, params)
        require_distinct_explicit_split_seeds(
            {
                name: parameters["seed"]
                for name, (_size, parameters) in split_params.items()
            },
            dataset_label="Ready-Set-Go",
        )

        required_horizon = max(
            params["set_max_step"]
            + round(params["production_scale"] * max(params["sample_intervals"]))
            + params["post_go_duration"]
            + 1
            for _, params in split_params.values()
        )
        configured_horizon = kwargs.get("sequence_duration")
        fixed_horizon = (
            required_horizon if configured_horizon is None else configured_horizon
        )
        datasets = []
        for split_name in split_names:
            n_split, params = split_params[split_name]
            params["sequence_duration"] = fixed_horizon
            datasets.append(ReadySetGoDataset(n_samples=n_split, **params))
        return SequenceDatasetBuild(split_datasets=tuple(datasets))

    required_horizon = (
        set_max_step
        + round(common_defaults["production_scale"] * max(default_intervals))
        + common_defaults["post_go_duration"]
        + 1
    )
    configured_horizon = kwargs.get("sequence_duration")
    fixed_horizon = (
        required_horizon if configured_horizon is None else configured_horizon
    )
    train_params = {
        **common_defaults,
        "sequence_duration": fixed_horizon,
        "seed": base_seed,
    }
    test_params = {**train_params, "seed": base_seed + 1}
    return SequenceDatasetBuild(
        train_full=ReadySetGoDataset(n_samples=n_train, **train_params),
        test_ds=ReadySetGoDataset(n_samples=max(n_train // 5, 2000), **test_params),
    )


def build_contextual_ready_set_go(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    """Build paired context-selected timing splits on one fixed horizon."""

    validate_noop_standard_processing(
        kwargs,
        dataset_label="contextual Ready-Set-Go",
    )

    n_train = kwargs.get("n_samples", 10000)
    default_intervals = tuple(
        kwargs.get("sample_intervals", (40, 50, 60, 70, 80, 90, 100))
    )
    split_names = ("train", "valid", "test")
    split_overrides = validated_split_overrides(
        kwargs,
        dataset_label="contextual Ready-Set-Go",
    )
    all_intervals = list(default_intervals)
    for override in split_overrides.values():
        all_intervals.extend(override.get("sample_intervals", ()))
    default_set_min = max(all_intervals) + 30
    set_min_step = kwargs.get("set_min_step", default_set_min)
    set_max_step = kwargs.get("set_max_step", set_min_step + 130)
    common_defaults = {
        "sample_intervals": default_intervals,
        "production_scale": kwargs.get("production_scale", 1.0),
        "set_min_step": set_min_step,
        "set_max_step": set_max_step,
        "pulse_duration": kwargs.get("pulse_duration", 2),
        "context_start_step": kwargs.get("context_start_step", 0),
        "context_duration": kwargs.get("context_duration", 10),
        "context_gain": kwargs.get("context_gain", 1.0),
        "minimum_competing_go_separation": kwargs.get(
            "minimum_competing_go_separation", 10
        ),
        "target_sigma": kwargs.get("target_sigma", 2.0),
        "post_go_duration": kwargs.get("post_go_duration", 20),
        "cue_gain_min": kwargs.get("cue_gain_min", 0.7),
        "cue_gain_max": kwargs.get("cue_gain_max", 1.3),
        "input_noise_std": kwargs.get("input_noise_std", 0.05),
        "mean_distractors_per_trial": kwargs.get("mean_distractors_per_trial", 1.0),
        "distractor_amplitude": kwargs.get("distractor_amplitude", 1.0),
        "distractor_pulse_duration": kwargs.get("distractor_pulse_duration", 2),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
    }
    base_seed = kwargs.get("seed", 0)
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise TypeError("contextual Ready-Set-Go seed must be an integer")

    if split_overrides:
        forbidden_split_geometry = {
            "context_duration",
            "context_start_step",
            "sequence_duration",
            "set_max_step",
            "set_min_step",
        }
        if any(
            forbidden_split_geometry.intersection(override)
            for override in split_overrides.values()
        ):
            raise ValueError(
                "sequence duration, context timing, and Set-time bounds must be "
                "shared across splits"
            )
        default_sizes = {
            "train": n_train,
            "valid": max(n_train // 5, 2000),
            "test": max(n_train // 5, 2000),
        }
        split_params: dict[str, tuple[int, dict[str, Any]]] = {}
        for seed_offset, split_name in enumerate(split_names):
            override = dict(split_overrides.get(split_name, {}))
            n_split = override.pop("n_samples", default_sizes[split_name])
            params = {**common_defaults, **override}
            params.setdefault("seed", base_seed + seed_offset)
            split_params[split_name] = (n_split, params)
        require_distinct_explicit_split_seeds(
            {
                name: parameters["seed"]
                for name, (_size, parameters) in split_params.items()
            },
            dataset_label="contextual Ready-Set-Go",
        )

        required_horizon = max(
            params["set_max_step"]
            + round(params["production_scale"] * max(params["sample_intervals"]))
            + params["post_go_duration"]
            + 1
            for _, params in split_params.values()
        )
        configured_horizon = kwargs.get("sequence_duration")
        fixed_horizon = (
            required_horizon if configured_horizon is None else configured_horizon
        )
        datasets = []
        for split_name in split_names:
            n_split, params = split_params[split_name]
            params["sequence_duration"] = fixed_horizon
            datasets.append(ContextualReadySetGoDataset(n_samples=n_split, **params))
        return SequenceDatasetBuild(split_datasets=tuple(datasets))

    required_horizon = (
        set_max_step
        + round(common_defaults["production_scale"] * max(default_intervals))
        + common_defaults["post_go_duration"]
        + 1
    )
    configured_horizon = kwargs.get("sequence_duration")
    fixed_horizon = (
        required_horizon if configured_horizon is None else configured_horizon
    )
    train_params = {
        **common_defaults,
        "sequence_duration": fixed_horizon,
        "seed": base_seed,
    }
    test_params = {**train_params, "seed": base_seed + 1}
    return SequenceDatasetBuild(
        train_full=ContextualReadySetGoDataset(n_samples=n_train, **train_params),
        test_ds=ContextualReadySetGoDataset(
            n_samples=max(n_train // 5, 2000), **test_params
        ),
    )


DECISION_SEQUENCE_BUILDERS = {
    "context_dependent_decision": build_context_dependent_decision,
    "context_dependent_decision_hard": build_context_dependent_decision,
    "context_dependent_decision_hard_positive": build_context_dependent_decision,
    "context_dependent_decision_positive": build_context_dependent_decision,
    "contextual_ready_set_go": build_contextual_ready_set_go,
    "contextual_ready_set_go_positive": build_contextual_ready_set_go,
    "hierarchical_temporal": build_hierarchical_temporal,
    "ready_set_go": build_ready_set_go,
    "ready_set_go_positive": build_ready_set_go,
    "switching_context_decision": build_switching_context_decision,
    "switching_context_decision_positive": build_switching_context_decision,
    "timescale_generalization": build_timescale_generalization,
    "timescale_generalization_positive": build_timescale_generalization,
}


__all__ = [
    "DECISION_SEQUENCE_BUILDERS",
    "build_context_dependent_decision",
    "build_contextual_ready_set_go",
    "build_hierarchical_temporal",
    "build_ready_set_go",
    "build_switching_context_decision",
    "build_timescale_generalization",
]

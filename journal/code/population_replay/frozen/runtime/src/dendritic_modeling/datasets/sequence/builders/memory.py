"""Builders for memory-oriented sequence datasets."""

from __future__ import annotations

from numbers import Integral
from typing import Any

from dendritic_modeling.datasets.sequence.builders.common import (
    SequenceDatasetBuild,
    build_deferred_train_test_split,
    deferred_split,
    positive_input_encoding,
    test_size,
    validate_noop_standard_processing,
    validated_split_overrides,
)
from dendritic_modeling.datasets.sequence.memory_tasks import (
    AddingProblemDataset,
    CopyTaskDataset,
    SequentialMNISTDataset,
    VariableDelayDMSDataset,
)
from dendritic_modeling.datasets.sequence.multiscale_working_memory import (
    SpeededDistractorDMSDataset,
)

_SPEED_DMS_SPLITS = ("train", "valid", "test")
_SPEED_DMS_PARAMETERS = {
    "circular_noise_std",
    "delay_support",
    "distractor_identity_mode",
    "flatten",
    "intervention_seed",
    "label_noise_rate",
    "label_noise_seed",
    "lure_fraction",
    "n_distractors",
    "n_pairs",
    "n_samples",
    "normalize",
    "positive_input_encoding",
    "salience_mu",
    "seed",
    "sequence_duration",
    "split_seed",
    *_SPEED_DMS_SPLITS,
}


def _validated_even_size(value: int, *, split_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{split_name} n_samples must be an integer")
    result = int(value)
    if result <= 0 or result % 2:
        raise ValueError(
            f"{split_name} n_samples must be positive and even for paired trials"
        )
    return result


def _resolved_sd_dms_size(
    values: dict[str, Any],
    *,
    split_name: str,
    default_n_samples: int,
) -> int:
    has_samples = "n_samples" in values
    has_pairs = "n_pairs" in values
    if has_samples and has_pairs:
        raise ValueError(
            f"{split_name} must specify exactly one of n_samples or n_pairs"
        )
    if has_pairs:
        n_pairs = values["n_pairs"]
        if isinstance(n_pairs, bool) or not isinstance(n_pairs, Integral):
            raise TypeError(f"{split_name} n_pairs must be an integer")
        if int(n_pairs) <= 0:
            raise ValueError(f"{split_name} n_pairs must be positive")
        return 2 * int(n_pairs)
    return _validated_even_size(
        values.get("n_samples", default_n_samples),
        split_name=split_name,
    )


def build_sequential_mnist(
    _dataset_name: str,
    data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    mode = kwargs.get("mode", "pixel")
    normalize = kwargs.get("normalize", True)
    return deferred_split(
        SequentialMNISTDataset(
            mode=mode,
            train=True,
            data_path=data_path,
            normalize=normalize,
        ),
        SequentialMNISTDataset(
            mode=mode,
            train=False,
            data_path=data_path,
            normalize=normalize,
        ),
    )


def build_copy_task(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {
        "seq_len": kwargs.get("seq_len", 10),
        "delay": kwargs.get("delay", 50),
        "n_symbols": kwargs.get("n_symbols", 8),
    }
    return build_deferred_train_test_split(
        dataset_cls=CopyTaskDataset,
        n_train=n_train,
        common_defaults=common,
        min_test_size=1000,
    )


def build_adding_problem(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {"seq_len": kwargs.get("seq_len", 200)}
    return build_deferred_train_test_split(
        dataset_cls=AddingProblemDataset,
        n_train=n_train,
        common_defaults=common,
        min_test_size=1000,
    )


def build_variable_delay_dms(
    _dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    n_train = kwargs.get("n_samples", 10000)
    common = {
        "stimulus_dim": kwargs.get("stimulus_dim", 16),
        "stimulus_duration": kwargs.get("stimulus_duration", 10),
        "min_delay": kwargs.get("min_delay", 10),
        "max_delay": kwargs.get("max_delay", 100),
        "probe_duration": kwargs.get("probe_duration", 10),
    }
    return build_deferred_train_test_split(
        dataset_cls=VariableDelayDMSDataset,
        n_train=n_train,
        common_defaults=common,
        min_test_size=1000,
    )


def build_speeded_distractor_dms(
    dataset_name: str,
    _data_path: str | None,
    kwargs: dict[str, Any],
) -> SequenceDatasetBuild:
    """Build independently seeded, explicit SD-DMS dataset splits."""

    unexpected_top_level = set(kwargs).difference(_SPEED_DMS_PARAMETERS)
    if unexpected_top_level:
        fields = ", ".join(sorted(unexpected_top_level))
        raise ValueError(f"unsupported SD-DMS parameter(s): {fields}")
    validate_noop_standard_processing(kwargs, dataset_label="SD-DMS")
    n_train = _resolved_sd_dms_size(
        kwargs,
        split_name="train",
        default_n_samples=10000,
    )
    base_seed = kwargs.get("seed", 0)
    if isinstance(base_seed, bool) or not isinstance(base_seed, Integral):
        raise TypeError("speeded distractor DMS seed must be an integer")
    base_parameters = {
        "delay_support": tuple(kwargs.get("delay_support", (128, 160, 192, 224))),
        "n_distractors": kwargs.get("n_distractors", 4),
        "lure_fraction": kwargs.get("lure_fraction", 0.25),
        "distractor_identity_mode": kwargs.get(
            "distractor_identity_mode", "sample_conditioned_lures"
        ),
        "salience_mu": kwargs.get("salience_mu", 0.8),
        "circular_noise_std": kwargs.get("circular_noise_std", 0.04),
        "sequence_duration": kwargs.get("sequence_duration", 360),
        "positive_input_encoding": positive_input_encoding(dataset_name, kwargs),
        "intervention_seed": kwargs.get("intervention_seed"),
    }
    default_eval_size = test_size(n_train, 2000)
    default_eval_size += default_eval_size % 2
    default_sizes = {
        "train": n_train,
        "valid": default_eval_size,
        "test": default_eval_size,
    }
    split_overrides = validated_split_overrides(
        kwargs,
        dataset_label="SD-DMS",
    )

    split_specs: list[tuple[str, int, int]] = []
    for seed_offset, split_name in enumerate(_SPEED_DMS_SPLITS):
        override = split_overrides.get(split_name, {})
        unexpected = set(override).difference({"n_pairs", "n_samples", "seed"})
        if unexpected:
            fields = ", ".join(sorted(unexpected))
            raise ValueError(
                "SD-DMS task geometry must be shared across splits; "
                f"unsupported {split_name} override(s): {fields}"
            )
        split_size = _resolved_sd_dms_size(
            override,
            split_name=split_name,
            default_n_samples=default_sizes[split_name],
        )
        split_seed = override.get("seed", int(base_seed) + seed_offset)
        if isinstance(split_seed, bool) or not isinstance(split_seed, Integral):
            raise TypeError(f"{split_name} seed must be an integer")
        split_specs.append((split_name, split_size, int(split_seed)))

    split_seeds = [seed for _name, _size, seed in split_specs]
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError(
            "SD-DMS train, valid, and test seeds must be pairwise distinct"
        )

    datasets = []
    for split_name, split_size, split_seed in split_specs:
        datasets.append(
            SpeededDistractorDMSDataset(
                n_samples=split_size,
                seed=split_seed,
                split_name=split_name,
                **base_parameters,
            )
        )
    content_hashes = [dataset.content_sha256 for dataset in datasets]
    if len(set(content_hashes)) != len(content_hashes):
        raise RuntimeError(
            "SD-DMS train, valid, and test content hashes must be pairwise distinct"
        )
    return SequenceDatasetBuild(split_datasets=tuple(datasets))


MEMORY_SEQUENCE_BUILDERS = {
    "adding_problem": build_adding_problem,
    "copy_task": build_copy_task,
    "sequential_mnist": build_sequential_mnist,
    "speeded_distractor_dms": build_speeded_distractor_dms,
    "speeded_distractor_dms_positive": build_speeded_distractor_dms,
    "variable_delay_dms": build_variable_delay_dms,
}


__all__ = [
    "MEMORY_SEQUENCE_BUILDERS",
    "build_adding_problem",
    "build_copy_task",
    "build_sequential_mnist",
    "build_speeded_distractor_dms",
    "build_variable_delay_dms",
]

"""Shared helpers for sequence dataset builders."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True)
class SequenceDatasetBuild:
    """Datasets returned by a sequence dataset builder.

    Builders either return an explicit train/valid/test triplet or a full
    training set plus a test set for the public factory to split.
    """

    train_full: Any | None = None
    test_ds: Any | None = None
    split_datasets: tuple[Any, Any, Any] | None = None


SequenceDatasetBuilder = Callable[
    [str, str | None, dict[str, Any]], SequenceDatasetBuild
]

_EXPLICIT_SPLIT_NAMES = ("train", "valid", "test")


def validate_noop_standard_processing(
    kwargs: dict[str, Any],
    *,
    dataset_label: str,
) -> None:
    """Validate generic loader fields that a task intentionally does not use."""

    for field_name in ("flatten", "normalize"):
        if field_name not in kwargs:
            continue
        value = kwargs[field_name]
        if not isinstance(value, bool):
            raise TypeError(f"{dataset_label} {field_name} must be a bool")
        if value:
            raise ValueError(f"{dataset_label} requires {field_name}=false")

    if "label_noise_rate" in kwargs:
        rate = kwargs["label_noise_rate"]
        if isinstance(rate, bool) or not isinstance(rate, Real):
            raise TypeError(f"{dataset_label} label_noise_rate must be a real number")
        if not math.isfinite(float(rate)):
            raise ValueError(f"{dataset_label} label_noise_rate must be finite")
        if float(rate) != 0.0:
            raise ValueError(f"{dataset_label} requires label_noise_rate=0")

    # These generic seeds are validated but intentionally ignored. Explicit
    # task-owned train/valid/test seeds define scientific content and may differ
    # from the unified loader's split seed.
    for field_name in ("label_noise_seed", "split_seed"):
        if field_name not in kwargs:
            continue
        value = kwargs[field_name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{dataset_label} {field_name} must be an integer")


def require_distinct_explicit_split_seeds(
    seeds_by_split: dict[str, Any],
    *,
    dataset_label: str,
) -> None:
    """Require independent seeds for an explicit train/valid/test triplet."""

    if set(seeds_by_split) != set(_EXPLICIT_SPLIT_NAMES):
        raise ValueError(f"{dataset_label} requires train, valid, and test seeds")
    seeds: list[int] = []
    for split_name in _EXPLICIT_SPLIT_NAMES:
        value = seeds_by_split[split_name]
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{dataset_label} {split_name} seed must be an integer")
        seeds.append(int(value))
    if len(set(seeds)) != len(seeds):
        raise ValueError(
            f"{dataset_label} explicit train, valid, and test seeds must be distinct"
        )


def test_size(n_train: int, minimum: int) -> int:
    return max(n_train // 5, minimum)


def deferred_split(train_full: Any, test_ds: Any) -> SequenceDatasetBuild:
    return SequenceDatasetBuild(train_full=train_full, test_ds=test_ds)


def build_deferred_train_test_split(
    *,
    dataset_cls: Callable[..., Any],
    n_train: int,
    common_defaults: dict[str, Any],
    min_test_size: int,
) -> SequenceDatasetBuild:
    return deferred_split(
        dataset_cls(n_samples=n_train, **common_defaults),
        dataset_cls(n_samples=test_size(n_train, min_test_size), **common_defaults),
    )


def explicit_split(
    train_ds: Any,
    valid_ds: Any,
    test_ds: Any,
) -> SequenceDatasetBuild:
    return SequenceDatasetBuild(split_datasets=(train_ds, valid_ds, test_ds))


def positive_input_encoding(dataset_name: str, kwargs: dict[str, Any]) -> bool:
    return dataset_name.endswith("_positive") or kwargs.get(
        "positive_input_encoding", False
    )


def validated_split_overrides(
    kwargs: dict[str, Any],
    *,
    dataset_label: str,
) -> dict[str, dict[str, Any]]:
    """Copy present split mappings and reject split-local variant changes."""

    overrides: dict[str, dict[str, Any]] = {}
    for split_name in _EXPLICIT_SPLIT_NAMES:
        if split_name not in kwargs:
            continue
        raw_override = kwargs[split_name]
        if not isinstance(raw_override, Mapping):
            raise TypeError(
                f"{dataset_label} {split_name} split parameters must be a mapping"
            )
        override = dict(raw_override)
        if "positive_input_encoding" in override:
            raise ValueError(
                f"{dataset_label} positive_input_encoding is a dataset-level "
                "variant and cannot be overridden within a split"
            )
        overrides[split_name] = override
    return overrides


def has_explicit_splits(kwargs: dict[str, Any]) -> bool:
    return bool(
        validated_split_overrides(
            kwargs,
            dataset_label="sequence dataset",
        )
    )


def _split_dataset_kwargs(
    split_overrides: dict[str, dict[str, Any]],
    split_name: str,
    common_defaults: dict[str, Any],
    default_n_samples: int,
) -> tuple[int, dict[str, Any]]:
    split_kwargs = split_overrides.get(split_name, {})
    split_params = {
        **common_defaults,
        **{key: value for key, value in split_kwargs.items() if key != "n_samples"},
    }
    return split_kwargs.get("n_samples", default_n_samples), split_params


def build_explicit_or_deferred_split(
    *,
    dataset_cls: Callable[..., Any],
    kwargs: dict[str, Any],
    common_defaults: dict[str, Any],
    n_train: int,
    min_test_size: int,
) -> SequenceDatasetBuild:
    split_overrides = validated_split_overrides(
        kwargs,
        dataset_label=getattr(dataset_cls, "__name__", "sequence dataset"),
    )
    if split_overrides:

        def build_split(split_name: str, default_n_samples: int) -> Any:
            split_n, split_params = _split_dataset_kwargs(
                split_overrides,
                split_name,
                common_defaults,
                default_n_samples,
            )
            return dataset_cls(n_samples=split_n, **split_params)

        return explicit_split(
            build_split("train", n_train),
            build_split("valid", test_size(n_train, min_test_size)),
            build_split("test", test_size(n_train, min_test_size)),
        )

    return build_deferred_train_test_split(
        dataset_cls=dataset_cls,
        n_train=n_train,
        common_defaults=common_defaults,
        min_test_size=min_test_size,
    )


__all__ = [
    "SequenceDatasetBuild",
    "SequenceDatasetBuilder",
    "build_deferred_train_test_split",
    "build_explicit_or_deferred_split",
    "deferred_split",
    "explicit_split",
    "has_explicit_splits",
    "positive_input_encoding",
    "require_distinct_explicit_split_seeds",
    "test_size",
    "validate_noop_standard_processing",
    "validated_split_overrides",
]

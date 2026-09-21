"""Registry for built-in sequence dataset builders."""

from __future__ import annotations

from collections.abc import Mapping

from dendritic_modeling.datasets.sequence.builders.bayesian_timing import (
    BAYESIAN_TIMING_SEQUENCE_BUILDERS,
)
from dendritic_modeling.datasets.sequence.builders.common import SequenceDatasetBuilder
from dendritic_modeling.datasets.sequence.builders.decision import (
    DECISION_SEQUENCE_BUILDERS,
)
from dendritic_modeling.datasets.sequence.builders.memory import (
    MEMORY_SEQUENCE_BUILDERS,
)
from dendritic_modeling.datasets.sequence.builders.neuro import NEURO_SEQUENCE_BUILDERS
from dendritic_modeling.datasets.sequence.builders.signal import (
    SIGNAL_SEQUENCE_BUILDERS,
)


def merge_sequence_dataset_builders(
    *builder_maps: Mapping[str, SequenceDatasetBuilder],
) -> dict[str, SequenceDatasetBuilder]:
    """Merge builder maps while rejecting accidental duplicate dataset names."""
    merged: dict[str, SequenceDatasetBuilder] = {}
    for builder_map in builder_maps:
        duplicates = sorted(set(merged).intersection(builder_map))
        if duplicates:
            duplicate_names = ", ".join(duplicates)
            raise ValueError(
                f"duplicate sequence dataset builder names: {duplicate_names}"
            )
        merged.update(builder_map)
    return merged


SEQUENCE_DATASET_BUILDERS = merge_sequence_dataset_builders(
    BAYESIAN_TIMING_SEQUENCE_BUILDERS,
    DECISION_SEQUENCE_BUILDERS,
    MEMORY_SEQUENCE_BUILDERS,
    NEURO_SEQUENCE_BUILDERS,
    SIGNAL_SEQUENCE_BUILDERS,
)


def register_sequence_dataset_builder(
    name: str,
    builder: SequenceDatasetBuilder,
    *,
    allow_override: bool = False,
) -> None:
    """Register a sequence dataset builder by exact dataset name."""
    key = str(name)
    if not key.strip():
        raise ValueError("sequence dataset builder name must be non-empty")
    if not callable(builder):
        raise TypeError(f"Sequence dataset builder for {key!r} must be callable")
    if key in SEQUENCE_DATASET_BUILDERS and not allow_override:
        raise ValueError(f"Sequence dataset builder '{key}' is already registered")
    SEQUENCE_DATASET_BUILDERS[key] = builder


def unregister_sequence_dataset_builder(name: str) -> None:
    """Remove a sequence dataset builder if present."""
    SEQUENCE_DATASET_BUILDERS.pop(str(name), None)


def get_sequence_dataset_builder(name: str) -> SequenceDatasetBuilder:
    """Return a registered sequence dataset builder."""
    key = str(name)
    builder = SEQUENCE_DATASET_BUILDERS.get(key)
    if builder is None:
        raise ValueError(f"Unknown sequence dataset: {key}")
    return builder


def has_sequence_dataset_builder(name: str) -> bool:
    """Return True when a sequence dataset builder is registered."""
    return str(name) in SEQUENCE_DATASET_BUILDERS


def get_registered_sequence_dataset_names() -> list[str]:
    """Return registered sequence dataset names."""
    return sorted(SEQUENCE_DATASET_BUILDERS)


__all__ = [
    "SEQUENCE_DATASET_BUILDERS",
    "get_registered_sequence_dataset_names",
    "get_sequence_dataset_builder",
    "has_sequence_dataset_builder",
    "merge_sequence_dataset_builders",
    "register_sequence_dataset_builder",
    "unregister_sequence_dataset_builder",
]

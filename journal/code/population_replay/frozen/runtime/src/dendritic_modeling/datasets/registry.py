"""Dataset-loader registry for project extensions.

Built-in datasets continue through the established loaders in
``standard_datasets`` and ``sequence_datasets``.  This registry gives new
datasets a config-only entry point without editing those long dispatch chains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch.utils.data import Dataset

DatasetTriplet = tuple[Dataset, Dataset, Dataset]
DatasetBuilder = Callable[[Any], DatasetTriplet]


@dataclass(frozen=True)
class DatasetSpec:
    """Metadata for a registered dataset loader."""

    name: str
    builder: DatasetBuilder
    description: str = ""
    sequence: bool = False


_DATASET_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset(
    name: str,
    builder: DatasetBuilder,
    *,
    aliases: tuple[str, ...] | list[str] = (),
    description: str = "",
    sequence: bool = False,
    allow_override: bool = False,
) -> None:
    """Register a dataset builder.

    Builders receive the original task/data config object and must return
    ``(train_ds, valid_ds, test_ds)``.
    """
    names = [name, *aliases]
    if not names or any(not str(item).strip() for item in names):
        raise ValueError("dataset name and aliases must be non-empty")
    for raw_name in names:
        key = str(raw_name).lower()
        if key in _DATASET_REGISTRY and not allow_override:
            raise ValueError(f"Dataset '{key}' is already registered")
        _DATASET_REGISTRY[key] = DatasetSpec(
            name=key,
            builder=builder,
            description=description,
            sequence=sequence,
        )


def unregister_dataset(name: str) -> None:
    """Remove a registered dataset if present."""
    _DATASET_REGISTRY.pop(str(name).lower(), None)


def get_dataset_spec(name: str) -> DatasetSpec | None:
    """Return a registered dataset spec, if any."""
    return _DATASET_REGISTRY.get(str(name).lower())


def get_registered_dataset_names(*, sequence: bool | None = None) -> list[str]:
    """Return registered dataset names, optionally filtered by sequence flag."""
    if sequence is None:
        return sorted(_DATASET_REGISTRY)
    return sorted(
        name for name, spec in _DATASET_REGISTRY.items() if spec.sequence == sequence
    )


def build_registered_dataset(name: str, task_cfg: Any) -> DatasetTriplet | None:
    """Build a registered dataset, or ``None`` if the name is unregistered."""
    spec = get_dataset_spec(name)
    if spec is None:
        return None
    return spec.builder(task_cfg)


__all__ = [
    "DatasetBuilder",
    "DatasetSpec",
    "DatasetTriplet",
    "build_registered_dataset",
    "get_dataset_spec",
    "get_registered_dataset_names",
    "register_dataset",
    "unregister_dataset",
]

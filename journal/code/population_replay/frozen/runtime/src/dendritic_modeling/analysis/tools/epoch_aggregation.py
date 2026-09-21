"""Shared helpers for aggregating per-epoch analysis JSON files."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

from dendritic_modeling.utils.epoch_files import epoch_files_by_number


def _preserved_leaf_key_set(preserved_leaf_keys: Iterable[str]) -> set[str]:
    return set(preserved_leaf_keys)


def initialize_epoch_aggregation_structure(
    data: dict[str, Any],
    *,
    preserved_leaf_keys: Iterable[str],
) -> dict[str, Any]:
    """Return a nested aggregation structure with lists at metric leaves."""
    preserved_keys = _preserved_leaf_key_set(preserved_leaf_keys)
    agg_structure: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            agg_structure[key] = initialize_epoch_aggregation_structure(
                value,
                preserved_leaf_keys=preserved_keys,
            )
        elif key in preserved_keys:
            agg_structure[key] = value
        else:
            agg_structure[key] = []
    return agg_structure


def populate_epoch_aggregation(
    agg_dict: dict[str, Any],
    epoch_data: dict[str, Any],
    *,
    preserved_leaf_keys: Iterable[str],
) -> None:
    """Append epoch leaves into an initialized aggregation dictionary."""
    preserved_keys = _preserved_leaf_key_set(preserved_leaf_keys)
    for key, value in epoch_data.items():
        if isinstance(value, dict):
            if key in agg_dict:
                populate_epoch_aggregation(
                    agg_dict[key],
                    value,
                    preserved_leaf_keys=preserved_keys,
                )
        elif key in agg_dict and key not in preserved_keys:
            agg_dict[key].append(value)


def load_epoch_aggregation(
    save_path: str,
    *,
    preserved_leaf_keys: Iterable[str],
) -> tuple[dict[str, Any] | None, list[int] | None]:
    """Load and aggregate ``epoch*.json`` files from ``save_path/epochs``."""
    training_path = os.path.join(save_path, "epochs")
    if not os.path.exists(training_path):
        return None, None

    agg_dict_initialized = False
    agg_dict: dict[str, Any] = {}
    epoch_numbers = []

    for epoch_number, filename in epoch_files_by_number(training_path):
        epoch_numbers.append(epoch_number)
        file_path = os.path.join(training_path, filename)
        with open(file_path) as f:
            epoch_data = json.load(f)

        if not agg_dict_initialized:
            agg_dict = initialize_epoch_aggregation_structure(
                epoch_data,
                preserved_leaf_keys=preserved_leaf_keys,
            )
            agg_dict_initialized = True

        populate_epoch_aggregation(
            agg_dict,
            epoch_data,
            preserved_leaf_keys=preserved_leaf_keys,
        )

    return agg_dict, epoch_numbers


__all__ = [
    "initialize_epoch_aggregation_structure",
    "load_epoch_aggregation",
    "populate_epoch_aggregation",
]

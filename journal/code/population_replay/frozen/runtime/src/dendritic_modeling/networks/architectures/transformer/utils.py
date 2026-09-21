"""Small shared helpers for transformer replacement modules."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch.nn as nn

from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_mapping
from dendritic_modeling.utils.hooks import iter_modules_of_type

__all__ = [
    "_as_list",
    "_first_list_value",
    "_first_parameter",
    "_get_attr_path",
    "_infer_mlp_dims",
    "_last_list_value",
    "_set_attr_path",
    "_to_plain_mapping",
]


def _first_parameter(module: nn.Module) -> nn.Parameter | None:
    return next(module.parameters(), None)


def _get_attr_path(root: object, path: str) -> object:
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            raise AttributeError(path)
        current = getattr(current, part)
    return current


def _set_attr_path(root: object, path: str, value: object) -> None:
    parent_path, _, attr = path.rpartition(".")
    parent = _get_attr_path(root, parent_path) if parent_path else root
    if not hasattr(parent, attr):
        raise AttributeError(path)
    setattr(parent, attr, value)


def _infer_mlp_dims(mlp: nn.Module) -> tuple[int, int | None]:
    declared_hidden = getattr(mlp, "hidden_size", None)
    declared_intermediate = getattr(mlp, "intermediate_size", None)
    if declared_hidden is not None:
        return int(declared_hidden), (
            None if declared_intermediate is None else int(declared_intermediate)
        )

    hidden_candidates = []
    intermediate_candidates = []

    for attr in ("gate_proj", "up_proj", "fc1", "dense_h_to_4h"):
        layer = getattr(mlp, attr, None)
        if isinstance(layer, nn.Linear):
            hidden_candidates.append(layer.in_features)
            intermediate_candidates.append(layer.out_features)

    for attr in ("down_proj", "fc2", "dense_4h_to_h"):
        layer = getattr(mlp, attr, None)
        if isinstance(layer, nn.Linear):
            hidden_candidates.append(layer.out_features)
            intermediate_candidates.append(layer.in_features)

    if not hidden_candidates:
        linears = list(iter_modules_of_type(mlp, nn.Linear))
        if not linears:
            raise ValueError("Could not infer MLP dimensions; no Linear modules found")
        hidden_candidates.append(linears[0].in_features)
        hidden_candidates.append(linears[-1].out_features)
        if len(linears) > 1:
            intermediate_candidates.append(linears[0].out_features)

    hidden_size = max(set(hidden_candidates), key=hidden_candidates.count)
    intermediate_size = (
        max(set(intermediate_candidates), key=intermediate_candidates.count)
        if intermediate_candidates
        else None
    )
    return int(hidden_size), (
        None if intermediate_size is None else int(intermediate_size)
    )


def _as_list(values: Iterable[int]) -> list[int]:
    return [int(v) for v in values]


def _first_list_value(value: Any, default: int) -> int:
    if isinstance(value, list):
        if not value:
            return int(default)
        return int(value[0])
    if value is None:
        return int(default)
    return int(value)


def _last_list_value(value: Any, default: int) -> int:
    if isinstance(value, list):
        if not value:
            return int(default)
        return int(value[-1])
    if value is None:
        return int(default)
    return int(value)

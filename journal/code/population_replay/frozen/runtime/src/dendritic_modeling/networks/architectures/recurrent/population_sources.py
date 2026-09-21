"""Population-network source normalization and stream helpers."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.networks.architectures.recurrent.population_configs import (
    PopulationProjectionConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_constants import (
    _EXTERNAL_INPUT_ALIASES,
)


def _as_projection_config(
    connection: PopulationProjectionConfig | dict[str, Any],
) -> PopulationProjectionConfig:
    if isinstance(connection, PopulationProjectionConfig):
        return connection
    if isinstance(connection, dict):
        payload = dict(connection)
        if "probability" not in payload and "p_connect" in payload:
            payload["probability"] = payload.pop("p_connect")
        return PopulationProjectionConfig(**payload)
    raise TypeError(
        "connections entries must be PopulationProjectionConfig or dict, "
        f"got {type(connection).__name__}"
    )


def _concat_or_none(values: list[torch.Tensor]) -> torch.Tensor | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return torch.cat(values, dim=-1)


def _canonical_external_source(source: str) -> str | None:
    return _EXTERNAL_INPUT_ALIASES.get(str(source).lower())


def _split_qualified_source(source: str) -> tuple[str, str] | None:
    if "." not in source:
        return None
    layer_name, population_name = source.split(".", 1)
    if not layer_name or not population_name:
        raise ValueError(
            "qualified population sources must use 'layer.population', "
            f"got {source!r}"
        )
    return layer_name, population_name


__all__ = [
    "_as_projection_config",
    "_canonical_external_source",
    "_concat_or_none",
    "_split_qualified_source",
]

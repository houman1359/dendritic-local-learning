"""Shared named-population configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.conversion import (
    deep_merge_dicts,
    has_enabled_synapse_types,
)
from dendritic_modeling.config.reactivation import (
    DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)


@dataclass
class PopulationDefinitionConfig(BaseConfig):
    """Configuration for one named population in a population-network layer."""

    name: str = "pyr"
    polarity: str = "excitatory"  # excitatory | inhibitory
    n_neurons: int = 64
    branch_factors: list[int] = field(default_factory=lambda: [1])
    population: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValueError("population name must be non-empty")
        self.polarity = str(self.polarity).lower()
        if self.polarity not in {"excitatory", "inhibitory"}:
            raise ValueError(
                f"polarity must be 'excitatory' or 'inhibitory', got {self.polarity!r}"
            )
        if self.n_neurons <= 0:
            raise ValueError(f"n_neurons must be > 0, got {self.n_neurons}")
        for branch_factor in self.branch_factors:
            if int(branch_factor) <= 0:
                raise ValueError(
                    f"branch_factors entries must be positive, got {branch_factor}"
                )


def as_population_definition_config(
    population: PopulationDefinitionConfig | dict[str, Any],
    *,
    field_name: str = "populations",
) -> PopulationDefinitionConfig:
    """Coerce a population payload into ``PopulationDefinitionConfig``."""

    if isinstance(population, PopulationDefinitionConfig):
        return population
    if isinstance(population, dict):
        return PopulationDefinitionConfig(**population)
    raise TypeError(
        f"{field_name} entries must be PopulationDefinitionConfig or dict, "
        f"got {type(population).__name__}"
    )


def population_level_dims(n_soma: int, branch_factors: list[int]) -> list[int]:
    """Return leaf-to-soma level dimensions for a population morphology."""

    sizes = [int(n_soma)]
    for branch_factor in branch_factors:
        sizes.append(sizes[-1] * int(branch_factor))
    sizes.reverse()
    return sizes


def population_config_from_definition(
    population: PopulationDefinitionConfig,
    defaults: dict[str, Any],
    *,
    layer_idx: int,
) -> PopulationConfig:
    """Merge layer defaults with a population override and build config."""

    payload = deep_merge_dicts(defaults, dict(population.population))
    if (
        "reactivation_init_policy" not in payload
        and payload.get("use_shunting") is False
    ):
        payload["reactivation_init_policy"] = DEFAULT_ADDITIVE_REACTIVATION_INIT_POLICY
    payload["n_neurons"] = int(population.n_neurons)
    payload["branch_factors"] = list(population.branch_factors)
    payload.setdefault("structured_layer_idx", layer_idx)
    return PopulationConfig(**payload)


def population_requires_temporal_state(pop_config: PopulationConfig) -> bool:
    """Return whether population dynamics need state across sequence timesteps."""

    return (
        str(pop_config.dynamics_mode).lower() == "spike"
        or has_enabled_synapse_types(pop_config.synapse_types)
        or bool(pop_config.dendritic_spikes_enabled)
        or bool(pop_config.soma_feedback_enabled)
    )


__all__ = [
    "PopulationDefinitionConfig",
    "as_population_definition_config",
    "population_config_from_definition",
    "population_level_dims",
    "population_requires_temporal_state",
]

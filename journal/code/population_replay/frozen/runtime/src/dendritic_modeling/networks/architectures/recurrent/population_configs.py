"""Configuration dataclasses for population-network layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.networks.architectures.recurrent.population_common import (
    PopulationDefinitionConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_constants import (
    _DELAYED,
    _INPUT_SOURCE,
    _SAME_STEP,
)


@dataclass
class PopulationProjectionConfig(BaseConfig):
    """Directed connection between one source and one target population."""

    source: str = _INPUT_SOURCE
    target: str = ""
    pathway: str = ""  # inferred from source polarity when omitted
    timing: str = _SAME_STEP  # same_step | delayed
    enabled: bool = True
    probability: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.source = str(self.source)
        self.target = str(self.target)
        self.pathway = str(self.pathway).lower()
        self.timing = str(self.timing).lower()
        if not self.source:
            raise ValueError("connection source must be non-empty")
        if not self.target:
            raise ValueError("connection target must be non-empty")
        if self.timing not in {_SAME_STEP, _DELAYED}:
            raise ValueError(
                "connection timing must be 'same_step' or 'delayed', "
                f"got {self.timing!r}"
            )
        if self.probability is not None:
            self.probability = float(self.probability)
            if self.probability < 0.0 or self.probability > 1.0:
                raise ValueError(
                    f"connection probability must be in [0, 1], got {self.probability}"
                )


@dataclass
class PopulationLayerConfig(BaseConfig):
    """One layer containing multiple named dendritic populations."""

    name: str = "layer0"
    populations: list[PopulationDefinitionConfig | dict[str, Any]] = field(
        default_factory=list
    )
    connections: list[PopulationProjectionConfig | dict[str, Any]] = field(
        default_factory=list
    )
    population_defaults: dict[str, Any] = field(default_factory=dict)
    readout_population: str | None = None
    recurrent: bool = False
    dt: float = 1.0
    connection_seed: int | None = None
    store_routing: bool = False

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValueError("layer name must be non-empty")
        if not self.populations:
            raise ValueError("population-network layer requires populations")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")


@dataclass
class PopulationNetworkConfig(BaseConfig):
    """Stack of population-network dendritic layers."""

    layers: list[PopulationLayerConfig | dict[str, Any]] = field(default_factory=list)
    input_dim: int = 64
    # Adapter applied before optional TransferLayer/input projection. Identity
    # preserves historical behavior; signed_split maps arbitrary hidden states
    # to concatenated nonnegative positive/negative channels.
    input_transform: str = "identity"
    use_transfer: bool = False
    transfer_params: dict[str, Any] = field(default_factory=dict)
    input_projection_dims: list[int] = field(default_factory=list)
    readout_layer: str | None = None
    readout_population: str | None = None
    output_mode: str = "last"  # recurrent only: last | mean | all
    store_routing: bool = False

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if not self.layers:
            raise ValueError("population network requires at least one layer")
        for hidden_dim in self.input_projection_dims:
            if int(hidden_dim) <= 0:
                raise ValueError(
                    f"input_projection_dims entries must be positive, got {hidden_dim}"
                )
        if self.output_mode not in {"last", "mean", "all"}:
            raise ValueError(
                f"output_mode must be one of ('last', 'mean', 'all'), got {self.output_mode!r}"
            )


__all__ = [
    "PopulationLayerConfig",
    "PopulationNetworkConfig",
    "PopulationProjectionConfig",
]

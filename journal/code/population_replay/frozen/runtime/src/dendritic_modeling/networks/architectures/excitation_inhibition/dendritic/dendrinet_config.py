"""Typed configuration for DendriNet construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)


def _require_config_value(source: Mapping[str, Any], key: str) -> Any:
    """Return a required config value, rejecting omitted or explicit None values."""
    value = source.get(key, None)
    if value is None:
        raise ValueError(f"DendriNetConfig requires {key}")
    return value


def _normalize_branch_factors(branch_factors: Sequence[int]) -> tuple[int, ...]:
    """Return branch factors in an immutable, dataclass-friendly form."""
    return tuple(branch_factors)


@dataclass(frozen=True)
class DendriNetConfig:
    """Geometry, shared synapses, and structured-mask options for ``DendriNet``."""

    n_soma: int
    branch_factors: tuple[int, ...]
    excitatory_input_dim: int | None = None
    excitatory_synapses_per_branch: int | None = None
    inhibitory_input_dim: int | None = None
    inhibitory_synapses_per_branch: int | None = None
    somatic_synapses: bool = True
    structured_connectivity: Any | None = None
    structured_layer_idx: int = 0
    structured_excitatory_pathway: str = "ee"
    structured_inhibitory_pathway: str = "ie"
    synapse_config: DendriticSynapseConfig = field(
        default_factory=DendriticSynapseConfig
    )

    @classmethod
    def from_config(
        cls,
        config: DendriNetConfig | Mapping[str, Any],
    ) -> DendriNetConfig:
        """Normalize an existing config object or mapping."""
        if isinstance(config, cls):
            return config
        if isinstance(config, Mapping):
            return cls.from_kwargs(config)
        raise TypeError("config must be a DendriNetConfig or mapping")

    @classmethod
    def from_kwargs(
        cls,
        mapping: Mapping[str, Any] | None = None,
        /,
        **kwargs: Any,
    ) -> DendriNetConfig:
        """Build from legacy ``DendriNet`` kwargs or a nested config mapping."""
        source = dict(mapping or {})
        source.update(kwargs)
        synapse_source = source.get("synapse_config", source.get("synapse", None))
        synapse_config = (
            DendriticSynapseConfig.from_config(synapse_source)
            if synapse_source is not None
            else DendriticSynapseConfig.from_config(source)
        )

        return cls(
            n_soma=_require_config_value(source, "n_soma"),
            branch_factors=_normalize_branch_factors(
                _require_config_value(source, "branch_factors")
            ),
            excitatory_input_dim=source.get("excitatory_input_dim", None),
            excitatory_synapses_per_branch=source.get(
                "excitatory_synapses_per_branch", None
            ),
            inhibitory_input_dim=source.get("inhibitory_input_dim", None),
            inhibitory_synapses_per_branch=source.get(
                "inhibitory_synapses_per_branch", None
            ),
            somatic_synapses=source.get("somatic_synapses", True),
            structured_connectivity=source.get("structured_connectivity", None),
            structured_layer_idx=source.get("structured_layer_idx", 0),
            structured_excitatory_pathway=source.get(
                "structured_excitatory_pathway", "ee"
            ),
            structured_inhibitory_pathway=source.get(
                "structured_inhibitory_pathway", "ie"
            ),
            synapse_config=synapse_config,
        )

    def to_kwargs(self) -> dict[str, Any]:
        """Convert to the legacy ``DendriNet`` constructor keyword shape."""
        return {
            **self.synapse_config.to_kwargs(),
            "n_soma": self.n_soma,
            "branch_factors": list(self.branch_factors),
            "excitatory_input_dim": self.excitatory_input_dim,
            "excitatory_synapses_per_branch": self.excitatory_synapses_per_branch,
            "inhibitory_input_dim": self.inhibitory_input_dim,
            "inhibitory_synapses_per_branch": self.inhibitory_synapses_per_branch,
            "somatic_synapses": self.somatic_synapses,
            "structured_connectivity": self.structured_connectivity,
            "structured_layer_idx": self.structured_layer_idx,
            "structured_excitatory_pathway": self.structured_excitatory_pathway,
            "structured_inhibitory_pathway": self.structured_inhibitory_pathway,
        }


__all__ = ["DendriNetConfig"]

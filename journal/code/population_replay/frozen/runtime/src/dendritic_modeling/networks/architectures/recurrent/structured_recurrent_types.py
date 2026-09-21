"""Typed helper records for structured recurrent E/I factory construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _StructuredRecurrentSections:
    initialization_seed: int | None
    recurrent_cfg: dict[str, Any]
    architecture: dict[str, Any]
    connectivity: dict[str, Any]
    structured_connectivity: Any
    transfer: dict[str, Any]
    morphology: dict[str, Any]
    sparsity: dict[str, Any]
    reactivation: dict[str, Any]
    blocklinear: dict[str, Any]
    implementation: dict[str, Any]
    synapse_types: dict[str, Any]
    dynamics: dict[str, Any]
    dendritic_spikes: dict[str, Any]
    soma_feedback: dict[str, Any]
    deepst: dict[str, Any]
    dense_to_sparse: dict[str, Any]
    indexed: dict[str, Any]


@dataclass(frozen=True)
class _RecurrentSynapseCounts:
    rec_ee: Any
    rec_ie: Any
    rec_ei: Any
    rec_ii: Any


@dataclass(frozen=True)
class _LayerPopulationSynapseCounts:
    ff_excitatory: int
    ff_inhibitory: int
    rec_excitatory: int
    rec_inhibitory: int

    @property
    def has_incoming_synapses(self) -> bool:
        return any(
            value > 0
            for value in (
                self.ff_excitatory,
                self.ff_inhibitory,
                self.rec_excitatory,
                self.rec_inhibitory,
            )
        )


@dataclass(frozen=True)
class _LayerRecurrentTiming:
    dt: float
    tau_base: float
    tau_ratio: float


@dataclass(frozen=True)
class _RecurrentFeatureFlags:
    has_rec_synapses: bool
    has_soma_feedback: bool
    has_spiking_dynamics: bool
    has_dendritic_spikes: bool
    has_synapse_types: bool

    @property
    def has_stateful_features(self) -> bool:
        return (
            self.has_soma_feedback
            or self.has_spiking_dynamics
            or self.has_dendritic_spikes
            or self.has_synapse_types
        )


@dataclass(frozen=True)
class _RecurrentReactivationOptions:
    reactivate: bool
    reactivation_type: str
    init_m: float
    init_b: float
    init_policy: str


@dataclass(frozen=True)
class _StructuredRecurrentBuildOptions:
    excitatory_sizes: list[Any]
    inhibitory_sizes: list[Any]
    input_mode: int
    use_transfer: bool
    transfer_inhibitory_mode: str
    explicit_recurrent_layers: Any
    use_shunting: bool
    use_additive_normalization: bool
    reactivation_options: _RecurrentReactivationOptions


__all__ = [
    "_LayerPopulationSynapseCounts",
    "_LayerRecurrentTiming",
    "_RecurrentFeatureFlags",
    "_RecurrentReactivationOptions",
    "_RecurrentSynapseCounts",
    "_StructuredRecurrentBuildOptions",
    "_StructuredRecurrentSections",
]

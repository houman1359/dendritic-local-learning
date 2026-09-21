"""Setup helpers for stateful recurrent dendritic populations."""

from __future__ import annotations

from dataclasses import dataclass

from dendritic_modeling.networks.architectures.recurrent.dendritic_geometry import (
    compute_input_branch_factors,
    resolve_synapses_by_level,
)
from dendritic_modeling.networks.architectures.recurrent.dendritic_spikes import (
    DendriticSpikeDynamics,
    resolve_dendritic_spike_levels,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.spiking_soma import LIFSoma


@dataclass(frozen=True)
class _StatefulBranchSynapseLayout:
    """Per-level branch factors and synapse counts for a stateful population."""

    input_branch_factors: list[int | None]
    ff_exc_by_level: list[int]
    ff_inh_by_level: list[int]
    rec_exc_by_level: list[int]
    rec_inh_by_level: list[int]


@dataclass(frozen=True)
class _StatefulDendriticSpikeSetup:
    """Resolved dendritic spike settings and runtime dynamics."""

    enabled: bool
    mode: str
    threshold: float
    plateau_amplitude: float
    plateau_tau: float
    refractory_steps: int
    surrogate_beta: float
    propagation: str
    level_indices: set[int]
    dynamics: DendriticSpikeDynamics


def _resolve_stateful_branch_synapse_layout(
    pop_config: PopulationConfig,
    n_levels: int,
) -> _StatefulBranchSynapseLayout:
    """Resolve branch factors and per-level synapse counts from population config."""
    return _StatefulBranchSynapseLayout(
        input_branch_factors=compute_input_branch_factors(pop_config.branch_factors),
        ff_exc_by_level=resolve_synapses_by_level(
            per_level=pop_config.ff_excitatory_synapses_by_level,
            fallback=pop_config.ff_excitatory_synapses,
            n_levels=n_levels,
            field_name="ff_excitatory_synapses_by_level",
        ),
        ff_inh_by_level=resolve_synapses_by_level(
            per_level=pop_config.ff_inhibitory_synapses_by_level,
            fallback=pop_config.ff_inhibitory_synapses,
            n_levels=n_levels,
            field_name="ff_inhibitory_synapses_by_level",
        ),
        rec_exc_by_level=resolve_synapses_by_level(
            per_level=pop_config.rec_excitatory_synapses_by_level,
            fallback=pop_config.rec_excitatory_synapses,
            n_levels=n_levels,
            field_name="rec_excitatory_synapses_by_level",
        ),
        rec_inh_by_level=resolve_synapses_by_level(
            per_level=pop_config.rec_inhibitory_synapses_by_level,
            fallback=pop_config.rec_inhibitory_synapses,
            n_levels=n_levels,
            field_name="rec_inhibitory_synapses_by_level",
        ),
    )


def _active_input_dim(input_dim: int | None, synapses: int) -> int | None:
    """Return an input dimension only when the corresponding pathway is active."""
    return input_dim if input_dim is not None and synapses > 0 else None


def _active_synapse_count(input_dim: int | None, synapses: int) -> int | None:
    """Return a synapse count only when the corresponding pathway is active."""
    return synapses if input_dim is not None and synapses > 0 else None


def _build_stateful_spiking_soma(
    pop_config: PopulationConfig,
    *,
    dynamics_mode: str,
    dt: float,
) -> LIFSoma | None:
    """Build the optional soma spiking dynamics module."""
    if dynamics_mode != "spike":
        return None

    return LIFSoma(
        threshold=pop_config.spike_threshold,
        reset=pop_config.spike_reset,
        tau=pop_config.spike_tau,
        dt=dt,
        refractory_steps=pop_config.spike_refractory_steps,
        surrogate_beta=pop_config.spike_surrogate_beta,
        output_mode=pop_config.spike_readout,
        readout_tau=pop_config.spike_readout_tau,
    )


def _resolve_stateful_dendritic_spike_setup(
    pop_config: PopulationConfig,
    *,
    n_levels: int,
    dt: float,
) -> _StatefulDendriticSpikeSetup:
    """Resolve optional dendritic spike settings and dynamics."""
    enabled = bool(pop_config.dendritic_spikes_enabled)
    mode = str(pop_config.dendritic_spike_mode).lower()
    threshold = float(pop_config.dendritic_spike_threshold)
    plateau_amplitude = float(pop_config.dendritic_spike_plateau_amplitude)
    plateau_tau = float(pop_config.dendritic_spike_plateau_tau)
    refractory_steps = int(pop_config.dendritic_spike_refractory_steps)
    surrogate_beta = float(pop_config.dendritic_spike_surrogate_beta)
    propagation = str(pop_config.dendritic_spike_propagation).lower()
    level_indices = resolve_dendritic_spike_levels(
        enabled,
        pop_config.dendritic_spike_levels,
        n_levels,
    )
    dynamics = DendriticSpikeDynamics(
        enabled=enabled,
        level_indices=level_indices,
        mode=mode,
        threshold=threshold,
        plateau_amplitude=plateau_amplitude,
        plateau_tau=plateau_tau,
        refractory_steps=refractory_steps,
        surrogate_beta=surrogate_beta,
        propagation=propagation,
        dt=dt,
    )
    return _StatefulDendriticSpikeSetup(
        enabled=enabled,
        mode=mode,
        threshold=threshold,
        plateau_amplitude=plateau_amplitude,
        plateau_tau=plateau_tau,
        refractory_steps=refractory_steps,
        surrogate_beta=surrogate_beta,
        propagation=propagation,
        level_indices=level_indices,
        dynamics=dynamics,
    )


def _resolve_stateful_level_taus(
    pop_config: PopulationConfig,
    n_levels: int,
) -> list[float]:
    """Resolve explicit or geometric per-level time constants."""
    if pop_config.level_taus:
        if len(pop_config.level_taus) != n_levels:
            raise ValueError(
                "level_taus length must match number of dendritic levels "
                f"({n_levels}), got {len(pop_config.level_taus)}"
            )
        return list(pop_config.level_taus)

    ratio = pop_config.tau_ratio
    return [
        pop_config.tau_base * (ratio ** (n_levels - 1 - i)) for i in range(n_levels)
    ]


def _resolve_stateful_soma_feedback_levels(
    enabled: bool,
    levels,
    n_levels: int,
) -> set[int]:
    """Resolve soma-feedback level selectors."""
    if not enabled:
        return set()
    dendritic_levels = list(range(max(n_levels - 1, 0)))
    if isinstance(levels, str):
        key = levels.lower()
        if key in {"all", "*"}:
            return set(range(n_levels))
        if key in {"dendritic", "non_soma", "non-soma"}:
            return set(dendritic_levels)
        if key == "distal":
            return {0} if dendritic_levels else set()
        if key == "proximal":
            return {dendritic_levels[-1]} if dendritic_levels else set()
        if key == "soma":
            return {n_levels - 1}
        if key in {"none", ""}:
            return set()
        raise ValueError(
            "soma_feedback_levels must be one of 'non_soma', 'all', "
            "'distal', 'proximal', 'soma', 'none', or a list of level "
            f"indices, got {levels!r}"
        )
    resolved = set()
    for value in levels:
        level_idx = int(value)
        if level_idx < 0 or level_idx >= n_levels:
            raise ValueError(
                "soma_feedback_levels entries are distal-to-soma level "
                f"indices in [0, {n_levels - 1}], got {level_idx}"
            )
        resolved.add(level_idx)
    return resolved


__all__ = [
    "_StatefulBranchSynapseLayout",
    "_StatefulDendriticSpikeSetup",
    "_active_input_dim",
    "_active_synapse_count",
    "_build_stateful_spiking_soma",
    "_resolve_stateful_branch_synapse_layout",
    "_resolve_stateful_dendritic_spike_setup",
    "_resolve_stateful_level_taus",
    "_resolve_stateful_soma_feedback_levels",
]

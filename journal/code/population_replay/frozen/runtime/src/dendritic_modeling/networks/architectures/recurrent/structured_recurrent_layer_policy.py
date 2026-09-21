"""Per-layer policy helpers for structured recurrent E/I factory configs."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.conversion import layer_value, to_plain_dict
from dendritic_modeling.config.legacy import normalize_transfer_config
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_types import (
    _LayerPopulationSynapseCounts,
    _LayerRecurrentTiming,
    _RecurrentFeatureFlags,
    _RecurrentSynapseCounts,
)


def _layer_recurrence_enabled(
    *,
    layer_idx: int,
    explicit_recurrent_layers: Any,
    counts: _RecurrentSynapseCounts,
) -> bool:
    if explicit_recurrent_layers is not None:
        return bool(layer_value(explicit_recurrent_layers, layer_idx, default=True))
    layer_rec_total = sum(
        int(layer_value(value, layer_idx, default=0))
        for value in (counts.rec_ee, counts.rec_ie, counts.rec_ei, counts.rec_ii)
    )
    return layer_rec_total > 0


def _direct_ff_inhibitory_enabled(
    *,
    input_mode: int,
    transfer_inhibitory_mode: str,
    layer_idx: int,
) -> bool:
    if input_mode != 1:
        return False
    if transfer_inhibitory_mode == "all":
        return True
    if transfer_inhibitory_mode == "first":
        return layer_idx == 0
    return False


def _resolve_layer_recurrent_timing(
    *,
    recurrent_cfg: dict[str, Any],
    layer_idx: int,
) -> _LayerRecurrentTiming:
    """Resolve per-layer recurrent integration and time-constant values."""
    return _LayerRecurrentTiming(
        dt=float(layer_value(recurrent_cfg.get("dt", 1.0), layer_idx, default=1.0)),
        tau_base=float(
            layer_value(recurrent_cfg.get("tau_base", 50.0), layer_idx, default=50.0)
        ),
        tau_ratio=float(
            layer_value(recurrent_cfg.get("tau_ratio", 3.0), layer_idx, default=3.0)
        ),
    )


def _resolve_recurrent_transfer_params(
    *,
    transfer: dict[str, Any],
    recurrent_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Merge transfer settings with legacy recurrent transfer override sections."""
    transfer_params = dict(transfer)
    transfer_params.update(to_plain_dict(recurrent_cfg.get("transfer", {})))
    transfer_params.update(to_plain_dict(recurrent_cfg.get("transfer_params", {})))
    return normalize_transfer_config(transfer_params)


def _resolve_recurrent_input_projection_dims(
    *,
    architecture: dict[str, Any],
    recurrent_cfg: dict[str, Any],
) -> list[Any]:
    """Resolve network-level input projection dims with legacy fallback semantics."""
    raw_ipd = architecture.get(
        "input_projection_dims",
        recurrent_cfg.get("input_projection_dims", []),
    )
    return list(raw_ipd) if isinstance(raw_ipd, list) else []


def _resolve_recurrent_store_routing(
    *,
    implementation: dict[str, Any],
    recurrent_cfg: dict[str, Any],
) -> bool:
    """Resolve whether recurrent routing should be retained for analysis."""
    return bool(
        implementation.get("store_routing", recurrent_cfg.get("store_routing", False))
    )


def _resolve_layer_inhibitory_synapse_counts(
    *,
    connectivity: dict[str, Any],
    counts: _RecurrentSynapseCounts,
    layer_idx: int,
) -> _LayerPopulationSynapseCounts:
    """Resolve feedforward/recurrent synapse counts for one inhibitory population."""
    return _LayerPopulationSynapseCounts(
        ff_excitatory=int(
            layer_value(
                connectivity.get("ei_synapses_per_branch_per_layer", []),
                layer_idx,
                default=0,
            )
        ),
        ff_inhibitory=int(
            layer_value(
                connectivity.get("ii_synapses_per_branch_per_layer", []),
                layer_idx,
                default=0,
            )
        ),
        rec_excitatory=int(layer_value(counts.rec_ei, layer_idx, default=0)),
        rec_inhibitory=int(layer_value(counts.rec_ii, layer_idx, default=0)),
    )


def _resolve_layer_excitatory_synapse_counts(
    *,
    connectivity: dict[str, Any],
    counts: _RecurrentSynapseCounts,
    layer_idx: int,
) -> _LayerPopulationSynapseCounts:
    """Resolve feedforward/recurrent synapse counts for one excitatory population."""
    return _LayerPopulationSynapseCounts(
        ff_excitatory=int(
            layer_value(
                connectivity.get("ee_synapses_per_branch_per_layer", []),
                layer_idx,
                default=0,
            )
        ),
        ff_inhibitory=int(
            layer_value(
                connectivity.get("ie_synapses_per_branch_per_layer", []),
                layer_idx,
                default=0,
            )
        ),
        rec_excitatory=int(layer_value(counts.rec_ee, layer_idx, default=0)),
        rec_inhibitory=int(layer_value(counts.rec_ie, layer_idx, default=0)),
    )


def _should_build_inhibitory_population(
    *,
    n_inh: int,
    synapses: _LayerPopulationSynapseCounts,
) -> bool:
    # Positive inhibitory sizes without incoming synapses preserve legacy direct-stream
    # behavior: the size can be present in the config without constructing a population.
    return n_inh > 0 and synapses.has_incoming_synapses


def _stateful_layer_features_enabled(
    *,
    flags: _RecurrentFeatureFlags,
    exc_cfg: Any,
    inh_cfg: Any | None,
) -> bool:
    """Return whether stateful dynamics require recurrent layer execution."""
    return (
        flags.has_spiking_dynamics
        or exc_cfg.dendritic_spikes_enabled
        or flags.has_synapse_types
        or exc_cfg.soma_feedback_enabled
        or (
            inh_cfg is not None
            and (
                inh_cfg.dendritic_spikes_enabled
                or flags.has_synapse_types
                or inh_cfg.soma_feedback_enabled
            )
        )
    )


def _effective_layer_recurrence_enabled(
    *,
    base_recurrent: bool,
    flags: _RecurrentFeatureFlags,
    exc_cfg: Any,
    inh_cfg: Any | None,
) -> bool:
    """Combine configured recurrence with stateful feature requirements."""
    return base_recurrent or _stateful_layer_features_enabled(
        flags=flags,
        exc_cfg=exc_cfg,
        inh_cfg=inh_cfg,
    )


__all__ = [
    "_direct_ff_inhibitory_enabled",
    "_effective_layer_recurrence_enabled",
    "_layer_recurrence_enabled",
    "_resolve_layer_excitatory_synapse_counts",
    "_resolve_layer_inhibitory_synapse_counts",
    "_resolve_layer_recurrent_timing",
    "_resolve_recurrent_input_projection_dims",
    "_resolve_recurrent_store_routing",
    "_resolve_recurrent_transfer_params",
    "_should_build_inhibitory_population",
    "_stateful_layer_features_enabled",
]

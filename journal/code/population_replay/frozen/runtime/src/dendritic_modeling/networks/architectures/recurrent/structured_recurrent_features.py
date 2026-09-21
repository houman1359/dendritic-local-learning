"""Feature detection helpers for structured recurrent E/I factory configs."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.conversion import has_enabled_synapse_types
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_types import (
    _RecurrentFeatureFlags,
    _RecurrentSynapseCounts,
    _StructuredRecurrentSections,
)


def _resolve_recurrent_synapse_counts(
    *,
    connectivity: dict[str, Any],
    recurrent_cfg: dict[str, Any],
) -> _RecurrentSynapseCounts:
    def _rec_syn(new_key: str, *old_keys: str, fallback: Any = 0) -> Any:
        if new_key in connectivity:
            return connectivity[new_key]
        for key in old_keys:
            if key in recurrent_cfg:
                return recurrent_cfg[key]
        return fallback

    rec_ee = _rec_syn(
        "rec_ee_synapses_per_branch",
        "rec_excitatory_synapses_per_layer",
        "rec_excitatory_synapses",
    )
    rec_ie = _rec_syn(
        "rec_ie_synapses_per_branch",
        "rec_inhibitory_synapses_per_layer",
        "rec_inhibitory_synapses",
    )
    rec_ei = _rec_syn(
        "rec_ei_synapses_per_branch",
        "inh_rec_excitatory_synapses_per_layer",
        "inh_rec_excitatory_synapses",
        fallback=rec_ee,
    )
    rec_ii = _rec_syn(
        "rec_ii_synapses_per_branch",
        "inh_rec_inhibitory_synapses_per_layer",
        "inh_rec_inhibitory_synapses",
        fallback=rec_ie,
    )
    return _RecurrentSynapseCounts(
        rec_ee=rec_ee,
        rec_ie=rec_ie,
        rec_ei=rec_ei,
        rec_ii=rec_ii,
    )


def _has_positive_layer_value(value: Any) -> bool:
    if isinstance(value, list):
        return any(v > 0 for v in value)
    return value is not None and value > 0


def _detect_recurrent_feature_flags(
    sections: _StructuredRecurrentSections,
    counts: _RecurrentSynapseCounts,
) -> _RecurrentFeatureFlags:
    has_rec_synapses = any(
        _has_positive_layer_value(value)
        for value in (counts.rec_ee, counts.rec_ie, counts.rec_ei, counts.rec_ii)
    )
    has_soma_feedback = bool(
        sections.soma_feedback.get(
            "enabled", sections.dynamics.get("soma_feedback_enabled", False)
        )
    )
    has_spiking_dynamics = (
        str(
            sections.dynamics.get(
                "mode", sections.dynamics.get("dynamics_mode", "rate")
            )
        ).lower()
        == "spike"
    )
    has_dendritic_spikes = bool(
        sections.dendritic_spikes.get(
            "enabled",
            sections.dynamics.get("dendritic_spikes_enabled", False),
        )
    )
    has_synapse_types = has_enabled_synapse_types(sections.synapse_types)
    return _RecurrentFeatureFlags(
        has_rec_synapses=has_rec_synapses,
        has_soma_feedback=has_soma_feedback,
        has_spiking_dynamics=has_spiking_dynamics,
        has_dendritic_spikes=has_dendritic_spikes,
        has_synapse_types=has_synapse_types,
    )


def _should_build_structured_recurrent_einet(
    sections: _StructuredRecurrentSections,
    flags: _RecurrentFeatureFlags,
) -> bool:
    explicit_enabled = sections.recurrent_cfg.get("enabled", None)
    return (
        flags.has_rec_synapses or bool(explicit_enabled) or flags.has_stateful_features
    )


__all__ = [
    "_detect_recurrent_feature_flags",
    "_has_positive_layer_value",
    "_resolve_recurrent_synapse_counts",
    "_should_build_structured_recurrent_einet",
]

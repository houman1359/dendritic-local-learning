"""Shared recurrent-core detection for model setup and validation."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.config.conversion import (
    deep_merge_dicts,
    has_enabled_synapse_types,
    to_plain_dict,
)

_BASELINE_RNN_TYPES = {"gru", "lstm", "vanilla_rnn", "rnn"}
_HETEROGENEOUS_LEAK_CTRNN_TYPES = {
    "heterogeneous_leak_ctrnn",
    "heterogeneous_ctrnn",
}
_LEGENDRE_MEMORY_TYPES = {"legendre_memory"}
_POPULATION_NETWORK_TYPES = {
    "population_network",
}
_UNIFIED_EI_TYPES = {
    "ei_unified",
    "unified_ei",
    "ei_net",
    "unified_einet",
    "rnn_dendritic_shunting",
    "rnn_dendritic_additive",
    "rnn_dendritic_normalized_additive",
    "rnn_flat_shunting",
    "rnn_flat_additive",
    "rnn_flat_normalized_additive",
}
_STRUCTURED_EI_TYPES = {
    "einet",
    "dendritic_shunting",
    "dendritic_additive",
    "dendritic_normalized_additive",
    "flat_shunting",
    "flat_additive",
    "flat_normalized_additive",
    "dendritic_mlp",
}


def _get_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _core_type(core_cfg: Any) -> str:
    return str(_get_value(core_cfg, "type", "") or "").lower()


def _is_positive(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_is_positive(item) for item in value)
    if value is None:
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _any_positive_in(mapping: Any, keys: tuple[str, ...]) -> bool:
    for key in keys:
        if _is_positive(_get_value(mapping, key, 0)):
            return True
    return False


def _legacy_recurrent_enabled(core_cfg: Any) -> bool:
    rec = _get_value(core_cfg, "recurrent", {})
    if isinstance(rec, dict):
        return bool(rec.get("enabled", False))
    return bool(getattr(rec, "enabled", False))


def _has_unified_recurrent_layers(core_cfg: Any) -> bool:
    unified = _get_value(core_cfg, "unified_ei", None)
    if unified is None:
        unified = _get_value(core_cfg, "ei_unified", {})
    unified = to_plain_dict(unified)
    if not unified:
        return False
    if bool(unified.get("recurrent", False)):
        return True
    for layer in unified.get("layers", []) or []:
        if isinstance(layer, dict):
            if bool(layer.get("recurrent", False)):
                return True
            if _is_positive(layer.get("rec_synapses", 0)):
                return True
            if _is_positive(layer.get("rec_excitatory_synapses", 0)):
                return True
            if _is_positive(layer.get("rec_inhibitory_synapses", 0)):
                return True
        elif bool(getattr(layer, "recurrent", False)):
            return True
    return False


def _has_structured_einet_recurrence(core_cfg: Any) -> bool:
    conn = _get_value(core_cfg, "connectivity", {})
    if _any_positive_in(
        conn,
        (
            "rec_ee_synapses_per_branch",
            "rec_ie_synapses_per_branch",
            "rec_ei_synapses_per_branch",
            "rec_ii_synapses_per_branch",
        ),
    ):
        return True

    dynamics = to_plain_dict(_get_value(core_cfg, "dynamics", {}))
    soma_feedback = to_plain_dict(_get_value(core_cfg, "soma_feedback", {}))
    if bool(
        soma_feedback.get(
            "enabled",
            dynamics.get("soma_feedback_enabled", False),
        )
    ):
        return True
    if (
        str(dynamics.get("mode", dynamics.get("dynamics_mode", "rate"))).lower()
        == "spike"
    ):
        return True

    dendritic_spikes = to_plain_dict(_get_value(core_cfg, "dendritic_spikes", {}))
    if bool(
        dendritic_spikes.get(
            "enabled",
            dynamics.get("dendritic_spikes_enabled", False),
        )
    ):
        return True
    if has_enabled_synapse_types(_get_value(core_cfg, "synapse_types", {})):
        return True

    recurrent_ei = _get_value(core_cfg, "recurrent_ei", {})
    recurrent_feedback = to_plain_dict(_get_value(recurrent_ei, "soma_feedback", {}))
    if bool(recurrent_feedback.get("enabled", False)):
        return True
    if not recurrent_ei:
        return False

    enabled = bool(_get_value(recurrent_ei, "enabled", False))
    if enabled:
        per_layer = _get_value(
            recurrent_ei,
            "recurrent_layers",
            _get_value(recurrent_ei, "recurrent_per_layer", None),
        )
        if per_layer is None:
            return True
        if isinstance(per_layer, list):
            return any(bool(v) for v in per_layer)
        return bool(per_layer)

    return _any_positive_in(
        recurrent_ei,
        (
            "rec_excitatory_synapses",
            "rec_inhibitory_synapses",
            "rec_excitatory_synapses_per_layer",
            "rec_inhibitory_synapses_per_layer",
        ),
    )


def _population_has_temporal_state(population: dict[str, Any]) -> bool:
    """Return whether a population needs state across sequence timesteps."""
    if str(population.get("dynamics_mode", "rate")).lower() == "spike":
        return True
    if has_enabled_synapse_types(population.get("synapse_types", {})):
        return True
    if bool(population.get("dendritic_spikes_enabled", False)):
        return True
    if bool(population.get("soma_feedback_enabled", False)):
        return True
    return False


def _population_network_requires_recurrent_wrapper(core_cfg: Any) -> bool:
    mp = to_plain_dict(_get_value(core_cfg, "population_network", {}))
    if not mp:
        mp = to_plain_dict(core_cfg)
    output_mode = str(mp.get("output_mode", "last")).lower()
    if output_mode in {"all", "mean"}:
        return True
    for layer in mp.get("layers", []) or []:
        layer_payload = to_plain_dict(layer)
        if bool(layer_payload.get("recurrent", False)):
            return True
        defaults = to_plain_dict(layer_payload.get("population_defaults", {}))
        if _population_has_temporal_state(defaults):
            return True
        for cell in layer_payload.get("populations", []) or []:
            cell_payload = to_plain_dict(cell)
            population = deep_merge_dicts(
                defaults,
                to_plain_dict(cell_payload.get("population", {})),
            )
            if _population_has_temporal_state(population):
                return True
        for conn in layer_payload.get("connections", []) or []:
            conn_payload = to_plain_dict(conn)
            if str(conn_payload.get("timing", "same_step")).lower() == "delayed":
                return True
    return False


def is_recurrent_core_config(core_cfg: Any) -> bool:
    """Detect whether a model core config should use recurrent wrappers."""
    core_type = _core_type(core_cfg)
    if core_type in _BASELINE_RNN_TYPES:
        return True
    if core_type in _HETEROGENEOUS_LEAK_CTRNN_TYPES:
        return True
    if core_type in _LEGENDRE_MEMORY_TYPES:
        return True
    if core_type in _POPULATION_NETWORK_TYPES:
        return _population_network_requires_recurrent_wrapper(core_cfg)
    if core_type in _UNIFIED_EI_TYPES:
        return _has_unified_recurrent_layers(core_cfg)
    if core_type in _STRUCTURED_EI_TYPES:
        return _has_structured_einet_recurrence(core_cfg) or _legacy_recurrent_enabled(
            core_cfg
        )
    return _legacy_recurrent_enabled(core_cfg)


__all__ = ["is_recurrent_core_config"]

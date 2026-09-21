"""Config-time compatibility diagnostics for dendritic operator settings.

The dendritic operator space (input domain, weight domain, integration rule,
gates, morphology, topology, output contract) contains combinations that are
legal but subtle. This module surfaces those interactions as warnings at
model-build time so they are visible in every run log, without changing any
behavior: nothing here raises, and each distinct message is emitted once per
process to stay quiet inside sweeps.

Interactions that are genuinely invalid keep failing exactly where they always
did (for example the runtime non-negative-drive check for shunting networks);
this module only makes the run log explain the configuration earlier.
"""

from __future__ import annotations

import logging
from typing import Any

from dendritic_modeling.networks.utils.weight_transforms import (
    POSITIVE_WEIGHT_TRANSFORMS,
)

logger = logging.getLogger(__name__)

_TANH_FAMILY_GATES = {"param_tanh", "param_tanh_only_m"}
_emitted: set[str] = set()


def _warn_once(message: str) -> None:
    if message not in _emitted:
        _emitted.add(message)
        logger.warning("operator config: %s", message)


def _get(section: Any, key: str, default: Any = None) -> Any:
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def collect_operator_config_warnings(core_config: Any) -> list[str]:
    """Return diagnostic messages for one core configuration (no side effects)."""

    morphology = _get(core_config, "morphology", {})
    reactivation = _get(core_config, "reactivation", {})
    warnings: list[str] = []

    use_shunting = bool(_get(morphology, "use_shunting", True))
    weight_transform = str(_get(morphology, "weight_transform", "exp")).lower()
    gate_type = str(_get(reactivation, "type", "param_tanh")).lower()
    gate_enabled = bool(_get(reactivation, "enabled", True))
    init_policy = str(_get(reactivation, "init_policy", "analytical")).lower()
    soma_type = _get(reactivation, "soma_type", None)
    additive_mode = str(_get(morphology, "additive_mode", "raw")).lower()

    if use_shunting and weight_transform not in POSITIVE_WEIGHT_TRANSFORMS:
        warnings.append(
            "shunting with a signed weight transform "
            f"({weight_transform!r}) skips the non-negative drive validation; "
            "the shunting denominator is then not guaranteed positive."
        )
    if (
        gate_enabled
        and init_policy == "occupancy_quantile"
        and gate_type not in _TANH_FAMILY_GATES
    ):
        warnings.append(
            "occupancy_quantile calibration derives (m, b) from tanh occupancy "
            f"formulas but the gate type is {gate_type!r}; the fitted values "
            "are applied through the gate's initialize() interface."
        )
    if soma_type and not gate_enabled:
        warnings.append(
            "reactivation.soma_type is set but reactivation.enabled is false; "
            "the soma override has no effect."
        )
    if use_shunting and additive_mode != "raw":
        warnings.append(
            f"additive_mode {additive_mode!r} is ignored while " "use_shunting is true."
        )
    return warnings


def collect_biological_violations(core_config: Any) -> list[str]:
    """Return violations of the declared biological constraint set.

    The strict contract covers input channels, hidden synapses, and population
    outputs. Signed source features must be split or rectified; all effective
    synaptic weights must be nonnegative; E and I effects remain separate; and
    population reactivations must return nonnegative activity. A signed target
    interface may still be represented by the difference of two positive E/I
    output pathways.
    """

    morphology = _get(core_config, "morphology", {})
    violations: list[str] = []
    weight_transform = str(_get(morphology, "weight_transform", "exp")).lower()
    if weight_transform not in POSITIVE_WEIGHT_TRANSFORMS:
        violations.append(
            f"morphology.weight_transform={weight_transform!r} permits signed "
            "synaptic weights (biological synapses require a positive "
            "transform: softplus, exp, or relu)."
        )
    synapse_mode = str(_get(core_config, "synapse_mode", "ei")).lower()
    if synapse_mode == "mlp":
        violations.append(
            "synapse_mode='mlp' merges excitatory and inhibitory inputs into "
            "one signed synapse bank, which is not Dale-compliant."
        )
    population_network = _get(core_config, "population_network", {})
    if population_network:
        input_transform = str(
            _get(population_network, "input_transform", "identity")
        ).lower()
        if input_transform not in {"relu", "signed_split"}:
            violations.append(
                "population_network.input_transform must be 'relu' or "
                "'signed_split' to enforce nonnegative dendritic inputs."
            )
        for layer_index, layer in enumerate(_get(population_network, "layers", [])):
            defaults = _get(layer, "population_defaults", {})
            for population in _get(layer, "populations", []):
                options = _get(population, "population", {})
                transform = str(
                    _get(
                        options,
                        "weight_transform",
                        _get(defaults, "weight_transform", weight_transform),
                    )
                ).lower()
                if transform not in POSITIVE_WEIGHT_TRANSFORMS:
                    violations.append(
                        f"population_network layer {layer_index} population "
                        f"{_get(population, 'name', '?')!r} uses signed "
                        f"weight_transform={transform!r}."
                    )
    return violations


def enforce_biological_declaration(core_config: Any) -> None:
    """Apply the ``biological_neuron`` declaration for one core config.

    ``true`` raises on any violation; undefined (``None``) logs violations as
    warnings so historical configs keep working; ``false`` silences the
    biological-purity diagnostics entirely (the configuration explicitly opts
    into non-biological settings).
    """

    declared = _get(core_config, "biological_neuron", None)
    if declared is False:
        return
    violations = collect_biological_violations(core_config)
    if not violations:
        return
    if declared is True:
        raise ValueError(
            "core.biological_neuron=true but the configuration violates the "
            "biological constraint set:\n- " + "\n- ".join(violations) + "\n"
            "Set core.biological_neuron: false to explicitly permit "
            "non-biological settings."
        )
    for message in violations:
        _warn_once(
            message + " (set core.biological_neuron explicitly to enforce or "
            "permit this)"
        )


def log_operator_config_warnings(core_config: Any) -> None:
    """Emit each distinct diagnostic once per process."""

    enforce_biological_declaration(core_config)
    for message in collect_operator_config_warnings(core_config):
        _warn_once(message)

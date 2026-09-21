"""Maintenance and inspection helpers for dendritic branch layers."""

from __future__ import annotations

from typing import Any


def decay_branch_synapse_weights(
    owner: Any,
    weight_decay,
    weight_boosting: bool = False,
) -> None:
    """Apply custom decay to all active sparse synapse sublayers."""
    if owner.branch_excitation is not None:
        owner.branch_excitation.decay_weights(weight_decay, weight_boosting)
    if owner.branch_inhibition is not None:
        owner.branch_inhibition.decay_weights(weight_decay, weight_boosting)
    if owner.branch_recurrent is not None:
        owner.branch_recurrent.decay_weights(weight_decay, weight_boosting)
    if owner.branch_rec_inhibition is not None:
        owner.branch_rec_inhibition.decay_weights(weight_decay, weight_boosting)


def apply_branch_rewiring(owner: Any) -> None:
    """Apply rewiring/noise hooks to all active sparse synapse sublayers."""

    def _apply_synapse_rewiring(layer):
        if layer is None:
            return
        if rewire_condition and hasattr(layer, "apply_noise_and_enforce_constraints"):
            layer.apply_noise_and_enforce_constraints()
        elif hasattr(layer, "apply_rewiring"):
            layer.apply_rewiring()

    if not hasattr(owner, "step_counter"):
        owner.step_counter = 0
    owner.step_counter += 1
    rewire_condition = owner.step_counter % owner.rewire_frequency == 0
    _apply_synapse_rewiring(owner.branch_excitation)
    _apply_synapse_rewiring(owner.branch_inhibition)
    _apply_synapse_rewiring(owner.branch_recurrent)
    _apply_synapse_rewiring(owner.branch_rec_inhibition)


def collect_from_branch_synapse_layers(owner: Any, method_name: str):
    """Collect a named method result from all active sparse synapse sublayers."""
    result = {}
    if owner.branch_excitation is not None:
        result["exc"] = getattr(owner.branch_excitation, method_name)()
    if owner.branch_recurrent is not None:
        result["rec_exc"] = getattr(owner.branch_recurrent, method_name)()
    if owner.branch_inhibition is not None:
        result["inh"] = getattr(owner.branch_inhibition, method_name)()
    if owner.branch_rec_inhibition is not None:
        result["rec_inh"] = getattr(owner.branch_rec_inhibition, method_name)()
    return result if result else None


__all__ = [
    "apply_branch_rewiring",
    "collect_from_branch_synapse_layers",
    "decay_branch_synapse_weights",
]

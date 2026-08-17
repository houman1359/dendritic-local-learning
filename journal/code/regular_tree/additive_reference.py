"""Executable reference equations for the raw additive dendritic control.

The publication configuration named ``dendritic_additive`` sets
``use_shunting=False``, ``use_additive_normalization=False``, and
``additive_mode='raw'``. This module isolates that forward equation and its
analytic derivatives from the larger training framework. It is a reference
implementation, not the implementation used to generate the archived results.
The source-of-truth file paths and hashes are recorded in
``../../reproducibility/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def softplus(raw: Array) -> Array:
    """Numerically stable nonnegative conductance transform."""

    raw = np.asarray(raw, dtype=float)
    return np.logaddexp(0.0, raw)


def softplus_derivative(raw: Array) -> Array:
    """Derivative of ``softplus(raw)``."""

    raw = np.asarray(raw, dtype=float)
    return np.where(
        raw >= 0.0,
        1.0 / (1.0 + np.exp(-raw)),
        np.exp(raw) / (1.0 + np.exp(raw)),
    )


def parametric_tanh(voltage: Array, slope: Array, midpoint: Array) -> Array:
    """Reactivation ``(tanh(m(V-b))+1)/2`` used in the experiments."""

    return 0.5 * (np.tanh(slope * (voltage - midpoint)) + 1.0)


def parametric_tanh_derivative(
    voltage: Array, slope: Array, midpoint: Array
) -> Array:
    """Voltage derivative of :func:`parametric_tanh`."""

    z = np.tanh(slope * (voltage - midpoint))
    return 0.5 * slope * (1.0 - z * z)


@dataclass(frozen=True)
class AdditiveState:
    """Forward intermediates for one additive compartment."""

    excitation: Array
    inhibition: Array
    child_current: Array
    voltage: Array
    activity: Array


def additive_compartment(
    excitatory_activity: Array,
    excitatory_conductance: Array,
    inhibitory_activity: Array,
    inhibitory_conductance: Array,
    child_activity: Array,
    child_conductance: Array,
    slope: Array,
    midpoint: Array,
) -> AdditiveState:
    """Evaluate ``V=E-I+C`` followed by the parametric-tanh gate.

    The last axis indexes contacts or children. Empty arrays may be supplied
    for an absent pathway. All preceding axes are broadcast batch/compartment
    dimensions.
    """

    excitation = np.sum(
        np.asarray(excitatory_activity) * np.asarray(excitatory_conductance),
        axis=-1,
    )
    inhibition = np.sum(
        np.asarray(inhibitory_activity) * np.asarray(inhibitory_conductance),
        axis=-1,
    )
    child_current = np.sum(
        np.asarray(child_activity) * np.asarray(child_conductance), axis=-1
    )
    voltage = excitation - inhibition + child_current
    activity = parametric_tanh(voltage, slope, midpoint)
    return AdditiveState(excitation, inhibition, child_current, voltage, activity)


def transformed_parameter_gradients(
    voltage_error: Array,
    excitatory_activity: Array,
    inhibitory_activity: Array,
    child_activity: Array,
    raw_excitatory: Array,
    raw_inhibitory: Array,
    raw_child: Array,
) -> tuple[Array, Array, Array]:
    """Return exact gradients with respect to raw softplus parameters.

    ``voltage_error`` is the exact ``dL/dV`` at the host compartment. The
    leading dimensions are broadcast over each pathway's final contact axis.
    """

    delta = np.expand_dims(np.asarray(voltage_error), axis=-1)
    grad_exc = (
        delta
        * np.asarray(excitatory_activity)
        * softplus_derivative(raw_excitatory)
    )
    grad_inh = (
        -delta
        * np.asarray(inhibitory_activity)
        * softplus_derivative(raw_inhibitory)
    )
    grad_child = (
        delta * np.asarray(child_activity) * softplus_derivative(raw_child)
    )
    return grad_exc, grad_inh, grad_child


def transport_error_to_child(
    parent_voltage_error: Array,
    child_voltage: Array,
    child_conductance: Array,
    child_slope: Array,
    child_midpoint: Array,
) -> Array:
    """Apply the exact additive child-to-parent error recursion."""

    return (
        np.asarray(parent_voltage_error)
        * np.asarray(child_conductance)
        * parametric_tanh_derivative(
            child_voltage, child_slope, child_midpoint
        )
    )


__all__ = [
    "AdditiveState",
    "additive_compartment",
    "parametric_tanh",
    "parametric_tanh_derivative",
    "softplus",
    "softplus_derivative",
    "transformed_parameter_gradients",
    "transport_error_to_child",
]

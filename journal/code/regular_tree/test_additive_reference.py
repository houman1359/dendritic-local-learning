#!/usr/bin/env python3
"""Finite-difference checks for the standalone raw-additive equations."""

from __future__ import annotations

import numpy as np

from additive_reference import (
    additive_compartment,
    parametric_tanh,
    softplus,
    transformed_parameter_gradients,
    transport_error_to_child,
)


def main() -> None:
    rng = np.random.default_rng(20260731)
    x_e = rng.uniform(0.0, 1.0, 5)
    x_i = rng.uniform(0.0, 1.0, 4)
    a_child = rng.uniform(0.0, 1.0, 3)
    raw_e = rng.normal(size=5)
    raw_i = rng.normal(size=4)
    raw_child = rng.normal(size=3)
    slope = np.asarray(1.7)
    midpoint = np.asarray(0.2)
    delta = np.asarray(-0.63)
    epsilon = 1e-6

    def voltage(re: np.ndarray, ri: np.ndarray, rc: np.ndarray) -> float:
        state = additive_compartment(
            x_e,
            softplus(re),
            x_i,
            softplus(ri),
            a_child,
            softplus(rc),
            slope,
            midpoint,
        )
        return float(state.voltage)

    analytic = transformed_parameter_gradients(
        delta, x_e, x_i, a_child, raw_e, raw_i, raw_child
    )
    numerical: list[np.ndarray] = []
    raw_groups = [raw_e, raw_i, raw_child]
    for group_index, values in enumerate(raw_groups):
        current = np.empty_like(values)
        for index in range(len(values)):
            plus = [value.copy() for value in raw_groups]
            minus = [value.copy() for value in raw_groups]
            plus[group_index][index] += epsilon
            minus[group_index][index] -= epsilon
            current[index] = delta * (
                voltage(*plus) - voltage(*minus)
            ) / (2.0 * epsilon)
        numerical.append(current)

    maximum_parameter_error = max(
        float(np.max(np.abs(exact - finite)))
        for exact, finite in zip(analytic, numerical)
    )

    parent_delta = np.asarray(0.81)
    child_voltage = np.asarray(-0.14)
    child_conductance = np.asarray(0.72)
    exact_transport = transport_error_to_child(
        parent_delta,
        child_voltage,
        child_conductance,
        slope,
        midpoint,
    )
    finite_transport = parent_delta * child_conductance * (
        parametric_tanh(child_voltage + epsilon, slope, midpoint)
        - parametric_tanh(child_voltage - epsilon, slope, midpoint)
    ) / (2.0 * epsilon)
    transport_error = float(np.abs(exact_transport - finite_transport))

    if maximum_parameter_error >= 1e-8 or transport_error >= 1e-8:
        raise AssertionError(
            "finite-difference validation failed: "
            f"parameter={maximum_parameter_error:.3e}, "
            f"transport={transport_error:.3e}"
        )
    print(
        "raw-additive reference validated; "
        f"max parameter error={maximum_parameter_error:.3e}, "
        f"transport error={transport_error:.3e}"
    )


if __name__ == "__main__":
    main()

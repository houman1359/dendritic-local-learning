"""Exact local-gain sensitivity and curvature of divisive branch integration.

The helpers describe the pre-reactivation scalar law used by a shunting branch
under a controlled multiplicative-gain perturbation.  They are analytic
diagnostics, not claims about a trained recurrent network's operating point.
"""

from __future__ import annotations

import torch


def _floating_vector(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.numel():
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _matched_vectors(
    signal: torch.Tensor,
    conductance: torch.Tensor,
    gains: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values = (
        _floating_vector(signal, name="signal"),
        _floating_vector(conductance, name="conductance"),
        _floating_vector(gains, name="gains"),
    )
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError("signal, conductance, and gains must have matched shapes")
    if (
        len({value.dtype for value in values}) != 1
        or len({value.device for value in values}) != 1
    ):
        raise ValueError("signal, conductance, and gains must share dtype and device")
    if bool((values[0] < 0).any()) or bool((values[1] < 0).any()):
        raise ValueError("signal and conductance must be non-negative")
    if bool((values[2] <= 0).any()):
        raise ValueError("gains must be positive")
    return values


def _baseline_tensor(
    reference: torch.Tensor, baseline: float | torch.Tensor
) -> torch.Tensor:
    result = torch.as_tensor(baseline, dtype=reference.dtype, device=reference.device)
    if result.ndim > 1 or (result.ndim == 1 and result.shape != reference.shape):
        raise ValueError("baseline must be scalar or match the branch vector")
    if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
        raise ValueError("baseline must be finite and positive")
    return result


def local_shunting_outputs(
    signal: torch.Tensor,
    conductance: torch.Tensor,
    gains: torch.Tensor,
    *,
    baseline: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return branch-local outputs ``g*N / (b + g*D)``."""

    signal, conductance, gains = _matched_vectors(signal, conductance, gains)
    baseline_tensor = _baseline_tensor(signal, baseline)
    return gains * signal / (baseline_tensor + gains * conductance)


def local_log_gain_jacobian(
    signal: torch.Tensor,
    conductance: torch.Tensor,
    gains: torch.Tensor,
    *,
    baseline: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return ``d local_output / d log(gain)`` for independent branch gains."""

    signal, conductance, gains = _matched_vectors(signal, conductance, gains)
    baseline_tensor = _baseline_tensor(signal, baseline)
    diagonal = (
        baseline_tensor
        * gains
        * signal
        / (baseline_tensor + gains * conductance).square()
    )
    return torch.diag(diagonal)


def global_shunting_output(
    signal: torch.Tensor,
    conductance: torch.Tensor,
    gains: torch.Tensor,
    *,
    baseline: float = 1.0,
) -> torch.Tensor:
    """Return one global divider applied after pooling all gain groups."""

    signal, conductance, gains = _matched_vectors(signal, conductance, gains)
    baseline_tensor = _baseline_tensor(signal, baseline)
    if baseline_tensor.ndim != 0:
        raise ValueError("global baseline must be scalar")
    numerator = torch.sum(gains * signal)
    denominator = baseline_tensor + torch.sum(gains * conductance)
    return numerator / denominator


def global_log_gain_gradient(
    signal: torch.Tensor,
    conductance: torch.Tensor,
    gains: torch.Tensor,
    *,
    baseline: float = 1.0,
) -> torch.Tensor:
    """Return the gradient of the global divider with respect to log gains."""

    signal, conductance, gains = _matched_vectors(signal, conductance, gains)
    baseline_tensor = _baseline_tensor(signal, baseline)
    if baseline_tensor.ndim != 0:
        raise ValueError("global baseline must be scalar")
    numerator = torch.sum(gains * signal)
    denominator = baseline_tensor + torch.sum(gains * conductance)
    return (
        gains * (signal * denominator - numerator * conductance) / denominator.square()
    )


def shunting_mixed_ei_curvature(
    excitation: torch.Tensor,
    inhibition: torch.Tensor,
    *,
    baseline: float = 1.0,
) -> torch.Tensor:
    """Return ``d²[N/(b+N+I)] / dN dI`` before reactivation."""

    if not isinstance(excitation, torch.Tensor) or not isinstance(
        inhibition, torch.Tensor
    ):
        raise TypeError("excitation and inhibition must be torch.Tensor objects")
    if excitation.shape != inhibition.shape or not excitation.is_floating_point():
        raise ValueError("excitation and inhibition must be matched floating tensors")
    if excitation.dtype != inhibition.dtype or excitation.device != inhibition.device:
        raise ValueError("excitation and inhibition must share dtype and device")
    if not bool(torch.isfinite(excitation).all()) or not bool(
        torch.isfinite(inhibition).all()
    ):
        raise ValueError("excitation and inhibition must be finite")
    if bool((excitation < 0).any()) or bool((inhibition < 0).any()):
        raise ValueError("excitation and inhibition must be non-negative")
    baseline_tensor = _baseline_tensor(excitation.reshape(-1), baseline)
    denominator = baseline_tensor + excitation + inhibition
    return (excitation - baseline_tensor - inhibition) / denominator.pow(3)


__all__ = [
    "global_log_gain_gradient",
    "global_shunting_output",
    "local_log_gain_jacobian",
    "local_shunting_outputs",
    "shunting_mixed_ei_curvature",
]

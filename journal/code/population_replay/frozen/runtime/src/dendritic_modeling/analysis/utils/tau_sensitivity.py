"""Exact local sensitivity of exponential recurrent time constants."""

from __future__ import annotations

from numbers import Real

import torch


def _positive_dt(dt: Real) -> float:
    if isinstance(dt, bool) or not isinstance(dt, Real):
        raise TypeError("dt must be a real number")
    value = float(dt)
    if not torch.isfinite(torch.tensor(value)) or value <= 0.0:
        raise ValueError("dt must be finite and positive")
    return value


def _positive_tau(tau: torch.Tensor) -> None:
    if not isinstance(tau, torch.Tensor):
        raise TypeError("tau must be a tensor")
    if not tau.is_floating_point():
        raise TypeError("tau must be floating point")
    if not bool(torch.isfinite(tau).all()) or bool((tau <= 0).any()):
        raise ValueError("tau must be finite and positive")


def exponential_decay(tau: torch.Tensor, *, dt: Real = 1.0) -> torch.Tensor:
    """Return the codebase decay ``rho = exp(-dt/tau)``."""

    _positive_tau(tau)
    return torch.exp(-_positive_dt(dt) / tau)


def decay_log_tau_sensitivity(
    tau: torch.Tensor,
    *,
    dt: Real = 1.0,
) -> torch.Tensor:
    r"""Return ``d rho / d log(tau) = (dt/tau) exp(-dt/tau)``."""

    resolved_dt = _positive_dt(dt)
    rho = exponential_decay(tau, dt=resolved_dt)
    return (resolved_dt / tau) * rho


def trace_update_log_tau_sensitivity(
    previous_trace: torch.Tensor,
    current_drive: torch.Tensor,
    tau: torch.Tensor,
    *,
    dt: Real = 1.0,
) -> torch.Tensor:
    r"""Return the local derivative of one trace update with respect to log tau.

    For ``T_t = rho T_{t-1} + (1-rho) J_t``, this is
    ``(d rho / d log(tau)) (T_{t-1} - J_t)`` before later BPTT factors.
    """

    if not isinstance(previous_trace, torch.Tensor) or not isinstance(
        current_drive, torch.Tensor
    ):
        raise TypeError("previous_trace and current_drive must be tensors")
    if previous_trace.shape != current_drive.shape:
        raise ValueError("previous_trace and current_drive must share shape")
    if (
        previous_trace.device != current_drive.device
        or previous_trace.dtype != current_drive.dtype
    ):
        raise ValueError("previous_trace and current_drive must share device/dtype")
    sensitivity = decay_log_tau_sensitivity(tau, dt=dt)
    return sensitivity * (previous_trace - current_drive)


__all__ = [
    "decay_log_tau_sensitivity",
    "exponential_decay",
    "trace_update_log_tau_sensitivity",
]

"""Activation-derivative helpers for local learning."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from dendritic_modeling.networks.activations.parametric import (
    ParametricTanh,
    ParametricTanhOnlyM,
)


def _parametric_tanh_derivative(
    react_module: ParametricTanh | ParametricTanhOnlyM,
    v_n: torch.Tensor,
    v_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the local slope for positive parametric tanh activations."""

    y = (
        v_out.to(device=v_n.device, dtype=v_n.dtype)
        if isinstance(v_out, torch.Tensor)
        else react_module(v_n)
    )
    m = react_module.log_m.detach().exp().to(device=v_n.device, dtype=v_n.dtype)
    return 0.5 * m.unsqueeze(0) * (1.0 - (2.0 * y - 1.0).square())


def compute_reactivation_derivative(
    react_module: nn.Module | None,
    v_n: torch.Tensor,
    v_out: torch.Tensor | None = None,
    *,
    logger: logging.Logger | None = None,
) -> torch.Tensor:
    """Return the elementwise local slope d a / d V for a branch reactivation."""

    if react_module is None or isinstance(react_module, nn.Identity):
        return torch.ones_like(v_n)

    if isinstance(react_module, nn.ReLU):
        return (v_n > 0).to(device=v_n.device, dtype=v_n.dtype)

    if isinstance(react_module, nn.Sigmoid):
        y = (
            v_out.to(device=v_n.device, dtype=v_n.dtype)
            if isinstance(v_out, torch.Tensor)
            else torch.sigmoid(v_n)
        )
        return y * (1.0 - y)

    if isinstance(react_module, nn.Tanh):
        y = (
            v_out.to(device=v_n.device, dtype=v_n.dtype)
            if isinstance(v_out, torch.Tensor)
            else torch.tanh(v_n)
        )
        return 1.0 - y.square()

    if isinstance(react_module, ParametricTanh):
        return _parametric_tanh_derivative(react_module, v_n, v_out)

    if isinstance(react_module, ParametricTanhOnlyM):
        return _parametric_tanh_derivative(react_module, v_n, v_out)

    try:
        with torch.enable_grad():
            x = v_n.detach().clone().requires_grad_(True)
            y = react_module(x)
            grad = torch.autograd.grad(
                y,
                x,
                grad_outputs=torch.ones_like(y),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]
        return grad.detach().to(device=v_n.device, dtype=v_n.dtype)
    except Exception:
        if logger is not None:
            logger.warning(
                "Falling back to unit reactivation derivative for module %s.",
                react_module.__class__.__name__,
            )
        return torch.ones_like(v_n)


__all__ = [
    "_parametric_tanh_derivative",
    "compute_reactivation_derivative",
]

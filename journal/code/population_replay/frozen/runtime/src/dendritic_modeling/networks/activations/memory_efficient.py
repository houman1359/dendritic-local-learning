"""Opt-in custom autograd paths for positive-tanh reactivation.

These functions preserve the model equation while retaining fewer full-size
tensors for backward.  They are deliberately configuration controlled; the
standard eager spelling remains the default and diagnostic paths stay eager.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch


def _feature_reduction_dims(value: torch.Tensor) -> tuple[int, ...]:
    return tuple(range(max(value.ndim - 1, 0)))


def _center_tensor(
    center: torch.Tensor | float, reference: torch.Tensor
) -> torch.Tensor:
    """Match eager scalar promotion while preserving tensor parameter precision."""
    if torch.is_tensor(center):
        return center.to(device=reference.device)
    return torch.as_tensor(center, dtype=reference.dtype, device=reference.device)


class _MemoryEfficientPositiveTanh(torch.autograd.Function):
    """Compute ``sigmoid(2 * scale * (x - center))``, saving only its output."""

    @staticmethod
    def forward(ctx, x, scale, center):
        center_tensor = _center_tensor(center, x)
        output = torch.sigmoid((scale + scale) * (x - center_tensor))
        ctx.save_for_backward(output, scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, scale = ctx.saved_tensors
        needs_x, needs_scale, needs_center = ctx.needs_input_grad[:3]
        twice_scale = scale + scale
        reduction_dims = _feature_reduction_dims(output)

        sigmoid_grad = output * output
        torch.sub(output, sigmoid_grad, out=sigmoid_grad)
        sigmoid_grad.mul_(grad_output)

        grad_scale = None
        if needs_scale:
            logits = torch.logit(output, torch.finfo(output.dtype).eps)
            scaled = sigmoid_grad * logits / scale
            grad_scale = scaled.sum(dim=reduction_dims) if reduction_dims else scaled

        grad_center = None
        if needs_center:
            reduced = (
                sigmoid_grad.sum(dim=reduction_dims) if reduction_dims else sigmoid_grad
            )
            grad_center = -reduced * twice_scale

        grad_x = sigmoid_grad.mul_(twice_scale) if needs_x else None
        return grad_x, grad_scale, grad_center


class _MemoryEfficientShuntPositiveTanh(torch.autograd.Function):
    """Fuse shunt division with positive-tanh and retain two tensors."""

    @staticmethod
    def forward(ctx, numerator, denominator, epsilon, scale, center):
        shifted_denominator = denominator + epsilon
        center_tensor = _center_tensor(center, numerator)
        output = torch.sigmoid(
            (scale + scale) * (numerator / shifted_denominator - center_tensor)
        )
        ctx.save_for_backward(shifted_denominator, output, scale, center_tensor)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        shifted_denominator, output, scale, center = ctx.saved_tensors
        needs_numerator, needs_denominator, _, needs_scale, needs_center = (
            ctx.needs_input_grad[:5]
        )
        twice_scale = scale + scale
        reduction_dims = _feature_reduction_dims(output)

        grad_voltage = output * output
        torch.sub(output, grad_voltage, out=grad_voltage)
        grad_voltage.mul_(grad_output).mul_(twice_scale)

        logits = None
        if needs_scale or needs_denominator:
            logits = torch.logit(output, torch.finfo(output.dtype).eps)

        grad_scale = None
        if needs_scale:
            assert logits is not None
            scaled = grad_voltage * logits / (twice_scale * scale)
            grad_scale = scaled.sum(dim=reduction_dims) if reduction_dims else scaled

        grad_center = None
        if needs_center:
            grad_center = (
                -grad_voltage.sum(dim=reduction_dims)
                if reduction_dims
                else -grad_voltage
            )

        grad_numerator = grad_voltage / shifted_denominator if needs_numerator else None
        grad_denominator = None
        if needs_denominator:
            assert logits is not None
            voltage = logits.div_(twice_scale).add_(center)
            if grad_numerator is not None:
                grad_denominator = voltage.mul_(grad_numerator).neg_()
            else:
                grad_denominator = (
                    voltage.mul_(grad_voltage).div_(shifted_denominator).neg_()
                )

        return grad_numerator, grad_denominator, None, grad_scale, grad_center


def positive_tanh_centered(
    scale: torch.Tensor,
    value: torch.Tensor,
    center: torch.Tensor | float,
    *,
    memory_efficient: bool,
) -> torch.Tensor:
    """Apply a centered positive tanh using eager or memory-saving autograd."""
    if memory_efficient:
        return _MemoryEfficientPositiveTanh.apply(value, scale, center)
    return (torch.tanh(scale * (value - center)) + 1) / 2


def fused_shunt_reactivation(
    reactivation: Any,
) -> Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor] | None:
    """Return a fused shunt callable when an activation explicitly opts in."""
    if not getattr(reactivation, "memory_efficient", False) or not getattr(
        reactivation,
        "supports_memory_efficient_shunt",
        False,
    ):
        return None

    center = reactivation.b if hasattr(reactivation, "b") else reactivation.fixed_b

    def run(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        return _MemoryEfficientShuntPositiveTanh.apply(
            numerator,
            denominator,
            float(epsilon),
            reactivation.log_m.exp(),
            center,
        )

    return run


__all__ = ["fused_shunt_reactivation", "positive_tanh_centered"]

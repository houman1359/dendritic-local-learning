"""Pure tensor helpers for local-learning gradient writeback."""

from __future__ import annotations

from typing import Any

import torch


def _accumulate_parameter_grad(param: torch.Tensor, grad: torch.Tensor) -> None:
    """Accumulate a local-rule gradient using the parameter's dtype."""
    grad = grad.to(param.dtype)
    if param.grad is None:
        param.grad = grad
    else:
        param.grad = param.grad + grad


def _scale_reactivation_error(
    error_signal: torch.Tensor,
    *,
    rho: Any,
    phi: float,
    branch_scale: Any,
) -> torch.Tensor:
    """Apply local-rule modulation to a reactivation error signal."""
    scaled = error_signal
    if isinstance(rho, torch.Tensor):
        scaled = scaled * rho.unsqueeze(0)
    else:
        scaled = scaled * float(rho)
    scaled = scaled * float(phi)
    if isinstance(branch_scale, torch.Tensor):
        scaled = scaled * branch_scale.unsqueeze(0)
    else:
        scaled = scaled * float(branch_scale)
    return scaled


def _reactivation_output_slope(output: torch.Tensor) -> torch.Tensor:
    """Derivative factor for the normalized tanh-style reactivation output."""
    return 1 - (2 * output - 1) ** 2


def _apply_optional_alignment(
    grad: torch.Tensor,
    alignment: torch.Tensor | None,
) -> torch.Tensor:
    """Apply a same-shaped alignment tensor to a local gradient."""
    if isinstance(alignment, torch.Tensor) and alignment.size() == grad.size():
        return grad * alignment.to(device=grad.device, dtype=grad.dtype)
    return grad


def _normalize_batch_gradient(
    grad: torch.Tensor,
    *,
    normalize_by_batch: bool,
    batch_size: int,
) -> torch.Tensor:
    """Average a summed local gradient when the local rule requests it."""
    if normalize_by_batch:
        return grad / float(batch_size)
    return grad


def _factor_pre_weight_grad(
    factor: torch.Tensor,
    x_pre: torch.Tensor,
    *,
    normalize_by_batch: bool,
    batch_size: int,
) -> torch.Tensor:
    """Compute a local pre-synaptic gradient with legacy batch scaling."""
    return _normalize_batch_gradient(
        factor.t() @ x_pre,
        normalize_by_batch=normalize_by_batch,
        batch_size=batch_size,
    )


def _factor_synapse_weight_grad(
    layer: Any,
    factor: torch.Tensor,
    x_pre: torch.Tensor,
    *,
    normalize_by_batch: bool,
    batch_size: int,
) -> torch.Tensor:
    """Compute a dense or compact synaptic conductance gradient."""

    compact_gradient = getattr(layer, "local_weight_gradient", None)
    if callable(compact_gradient):
        if int(batch_size) != int(factor.shape[0]):
            raise ValueError(
                "Configured batch size does not match the local factor batch: "
                f"{batch_size} vs {factor.shape[0]}"
            )
        return compact_gradient(
            factor,
            x_pre,
            normalize_by_batch=normalize_by_batch,
        )
    return _factor_pre_weight_grad(
        factor,
        x_pre,
        normalize_by_batch=normalize_by_batch,
        batch_size=batch_size,
    )


def _reduce_batch_mean_gradient(
    factor: torch.Tensor,
    *,
    normalize_by_batch: bool,
    batch_size: int,
) -> torch.Tensor:
    """Reduce per-example local factors using the legacy batch scaling rule."""
    grad = factor.mean(dim=0)
    if not normalize_by_batch:
        grad = grad * float(batch_size)
    return grad


def _topk_pre_weight_grad(
    local_grad: torch.Tensor,
    *,
    pre_weight: torch.Tensor,
    weight_transform: str,
    derivative_fn: Any,
) -> torch.Tensor:
    """Map a local conductance gradient through the TopK weight transform."""
    chain = derivative_fn(pre_weight.detach(), weight_transform)
    return local_grad * chain


__all__ = [
    "_accumulate_parameter_grad",
    "_apply_optional_alignment",
    "_factor_pre_weight_grad",
    "_factor_synapse_weight_grad",
    "_normalize_batch_gradient",
    "_reactivation_output_slope",
    "_reduce_batch_mean_gradient",
    "_scale_reactivation_error",
    "_topk_pre_weight_grad",
]

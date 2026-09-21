"""HSIC auxiliary-gradient methods for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_hsic import (
    compute_hsic_gradient,
    compute_kernel_matrix,
)


def _clip_hsic_gradient(
    grad_z: torch.Tensor,
    *,
    clip_value: float,
) -> torch.Tensor:
    """Apply the configured HSIC gradient clipping rule."""
    if clip_value > 0:
        return torch.clamp(grad_z, min=-clip_value, max=clip_value)
    return grad_z


def _hsic_topk_param_grad(
    grad_z: torch.Tensor,
    x_exc: torch.Tensor,
    current_weight: torch.Tensor,
    *,
    normalize_by_batch: bool,
    batch_size: int,
) -> torch.Tensor:
    """Project an HSIC output gradient onto TopK pre-weights."""
    grad_g = grad_z.t() @ x_exc
    if normalize_by_batch:
        grad_g = grad_g / float(batch_size)
    return grad_g * current_weight


def _resolve_hsic_warmup_scale(
    *,
    warmup_epochs: int,
    epoch_counter: int,
    apply_last_layer_only: bool,
    layer_idx: int,
    num_layers: int,
) -> float:
    """Resolve the HSIC scalar warmup for a layer update."""
    hsic_warm = 1.0
    if warmup_epochs > 0 and epoch_counter < warmup_epochs:
        hsic_warm = max(0.0, float(epoch_counter) / float(max(1, warmup_epochs)))

    if apply_last_layer_only:
        is_last = layer_idx == num_layers - 1
        if not is_last:
            hsic_warm = 0.0

    return hsic_warm


class LocalLearningHSICMixin:
    """Stateful HSIC gradient helpers used by ``LocalCreditAssignment``."""

    def _compute_kernel_matrix(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute kernel matrix K(X, Y) based on configured kernel type."""

        return compute_kernel_matrix(self.local_cfg, X, Y)

    def _compute_hsic_gradient(
        self, z: torch.Tensor, y: torch.Tensor, weight: float, grad_type: str
    ) -> torch.Tensor:
        """Compute HSIC gradient with respect to z using specified kernel."""

        return compute_hsic_gradient(self.local_cfg, z, y, weight, grad_type)

    def _apply_hsic_auxiliary_gradients(
        self,
        *,
        rec: dict[str, Any],
        exc_layer: TopKLinear | None,
        layer_idx: int,
        num_layers: int,
        y_target: torch.Tensor | None,
        batch_size: int,
    ) -> None:
        if not self.local_cfg.hsic.enabled:
            return

        v_out = rec.get("v_out")
        if v_out is None:
            return

        z = v_out - v_out.mean(dim=0, keepdim=True)
        hsic_warm = _resolve_hsic_warmup_scale(
            warmup_epochs=self.local_cfg.hsic.warmup_epochs,
            epoch_counter=self.epoch_counter,
            apply_last_layer_only=self.local_cfg.hsic.apply_last_layer_only,
            layer_idx=layer_idx,
            num_layers=num_layers,
        )

        if self.local_cfg.hsic.self_weight != 0.0:
            grad_z_self = self._compute_hsic_gradient(
                z, z, self.local_cfg.hsic.self_weight * hsic_warm, "self"
            )
            self._add_hsic_topk_grad(
                grad_z_self,
                rec=rec,
                exc_layer=exc_layer,
                batch_size=batch_size,
            )

        if self.local_cfg.hsic.target_weight != 0.0 and y_target is not None:
            y_centered = y_target - y_target.mean(dim=0, keepdim=True)
            grad_z_tgt = self._compute_hsic_gradient(
                z,
                y_centered,
                self.local_cfg.hsic.target_weight * hsic_warm,
                "target",
            )
            self._add_hsic_topk_grad(
                grad_z_tgt,
                rec=rec,
                exc_layer=exc_layer,
                batch_size=batch_size,
            )

    def _add_hsic_topk_grad(
        self,
        grad_z: torch.Tensor,
        *,
        rec: dict[str, Any],
        exc_layer: TopKLinear | None,
        batch_size: int,
    ) -> None:
        """Add an HSIC auxiliary gradient to the feedforward excitatory TopK path."""
        grad_z = _clip_hsic_gradient(
            grad_z,
            clip_value=self.local_cfg.hsic.grad_clip_value,
        )
        if exc_layer is None or rec.get("x_exc") is None:
            return

        grad_param = _hsic_topk_param_grad(
            grad_z,
            rec["x_exc"],
            exc_layer.weight().detach(),
            normalize_by_batch=self.local_cfg.normalize_by_batch,
            batch_size=batch_size,
        )
        self._add_topk_grad(exc_layer, grad_param, rec.get("exc_mask"))


__all__ = ["LocalLearningHSICMixin"]

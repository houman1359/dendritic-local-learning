"""
Custom optimizer implementations.

This module contains custom optimizer wrappers and implementations for
dendritic network training.
"""

from collections.abc import Iterator

import torch
from torch.optim import Optimizer

from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils.hooks import iter_modules_matching


def _is_rewired_optimizer_module(module) -> bool:
    """Return whether a module exposes rewiring masks and optimizer weights."""
    consume = getattr(module, "consume_rewired_mask", None)
    pre_w = getattr(module, "pre_w", None)
    return callable(consume) and pre_w is not None


def _iter_rewired_optimizer_modules(
    model,
) -> Iterator[tuple[object, torch.nn.Parameter]]:
    """Yield modules that expose rewiring masks and optimizer-backed weights."""
    for module in iter_modules_matching(model, _is_rewired_optimizer_module):
        yield module, module.pre_w


def _clear_rewired_optimizer_state(model, optimizer: Optimizer) -> None:
    """Clear optimizer moments for sparse slots replaced during rewiring."""
    for module, pre_w in _iter_rewired_optimizer_modules(model):
        consume = module.consume_rewired_mask
        mask = consume()
        if mask is None:
            continue
        state = optimizer.state.get(pre_w)
        if not state:
            continue
        mask = mask.to(device=pre_w.device)
        for value in state.values():
            if getattr(value, "shape", None) == pre_w.shape:
                value[mask] = 0


def apply_indexed_rewiring_after_step(model, optimizer: Optimizer) -> int:
    """Rewire indexed sparse leaves once and clear their optimizer moments.

    This helper is used by training paths that construct a normal PyTorch
    optimizer directly instead of wrapping a full ``BaseModel`` in
    ``CustomWeightDecayOptimizer``. Traversing rewired leaf modules avoids
    calling shared or nested replacement containers more than once.
    """
    rewired_modules = list(_iter_rewired_optimizer_modules(model))
    for module, _ in rewired_modules:
        apply_rewiring = getattr(module, "apply_rewiring", None)
        if callable(apply_rewiring):
            apply_rewiring()
    _clear_rewired_optimizer_state(model, optimizer)
    return len(rewired_modules)


def apply_sparse_topology_updates_after_step(model, optimizer: Optimizer) -> int:
    """Apply all optimizer-step-coupled sparse topology updates exactly once.

    Indexed rewiring first updates its fixed-size candidate contacts and clears
    optimizer moments for replaced slots. Developmental dense-to-sparse layers
    configured with ``advance_on_forward=False`` then advance their pruning
    schedules. Keeping this hook optimizer-step based prevents calibration,
    validation, or recomputation forwards from silently changing the schedule.
    """

    updated = apply_indexed_rewiring_after_step(model, optimizer)

    def is_manual_schedule(module) -> bool:
        return callable(
            getattr(module, "advance_sparsity_schedule", None)
        ) and not bool(getattr(module, "advance_on_forward", True))

    scheduled_modules = list(iter_modules_matching(model, is_manual_schedule))
    for module in scheduled_modules:
        module.advance_sparsity_schedule()
    return int(updated + len(scheduled_modules))


def apply_sparse_topology_updates_after_scaled_step(
    model,
    optimizer: Optimizer,
    scaler,
    *,
    scale_before: float,
) -> int:
    """Update topology only when ``GradScaler`` executed the optimizer step.

    PyTorch lowers the scale when non-finite gradients make ``scaler.step``
    skip the underlying optimizer update. Rewiring or pruning on that skipped
    step would desynchronize the topology from the optimizer-step schedule.
    """

    scale_after = float(scaler.get_scale())
    if scale_after < float(scale_before):
        return 0
    return apply_sparse_topology_updates_after_step(model, optimizer)


class CustomWeightDecayOptimizer:
    """
    Custom weight decay optimizer wrapper.

    This optimizer wrapper applies weight decay directly to the model parameters
    by calling the model's decay_weights method, rather than using PyTorch's
    built-in weight decay. This allows for more flexible weight decay strategies
    that are model-specific.

    Args:
        model: The model to apply weight decay to
        optimizer: The underlying PyTorch optimizer
        weight_decay: Weight decay coefficient
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: Optimizer,
        weight_decay: float = 0.1,
        weight_boosting: bool = False,
    ):
        self.model = model
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.weight_boosting = weight_boosting

    @property
    def param_groups(self):
        """Expose the underlying optimizer's param_groups."""
        return self.optimizer.param_groups

    @property
    def defaults(self):
        """Expose the underlying optimizer's defaults."""
        return self.optimizer.defaults

    def add_param_group(self, param_group: dict):
        """Delegate add_param_group to the underlying optimizer."""
        self.optimizer.add_param_group(param_group)

    def zero_grad(self):
        """Zero gradients of the underlying optimizer."""
        self.optimizer.zero_grad()

    def step(self):
        """
        Perform optimization step with custom weight decay.

        This method first applies model-specific weight decay using the current
        learning rate, then performs the standard optimizer step.
        """
        lr = self.optimizer.param_groups[0]["lr"]
        if self.weight_decay > 0:
            self.model.decay_weights(
                weight_decay=lr * self.weight_decay,
                weight_boosting=self.weight_boosting,
            )
        self.optimizer.step()
        self.model.apply_rewiring()
        _clear_rewired_optimizer_state(self.model, self.optimizer)


__all__ = [
    "CustomWeightDecayOptimizer",
    "apply_indexed_rewiring_after_step",
    "apply_sparse_topology_updates_after_scaled_step",
    "apply_sparse_topology_updates_after_step",
]

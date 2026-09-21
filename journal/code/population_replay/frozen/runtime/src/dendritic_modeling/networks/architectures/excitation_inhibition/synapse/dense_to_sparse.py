"""
Dense-to-sparse TopK linear layer.

This module implements a developmental pruning variant of ``TopKLinear``:
all allowed synapses can participate early in training, then the active
synapse count is gradually reduced to the configured target ``K``.
"""

from __future__ import annotations

import math

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.utils.weight_transforms import WeightTransformType


class DenseToSparseLinear(TopKLinear):
    """
    TopK layer with an annealed active-synapse count.

    The parent ``TopKLinear`` receives ``K`` as the final target number of
    active synapses per output.  ``DenseToSparseLinear`` starts with
    ``initial_k`` active synapses, or ``ceil(initial_density * in_features)``
    when ``initial_k`` is not supplied, and monotonically anneals down to ``K``.

    By default the schedule advances once per training forward pass.  This
    keeps the feature usable across standard, local, feedforward, and recurrent
    trainers without requiring trainer-specific hooks.
    """

    # The annealing schedule can change the active-K (and thus the mask) on each
    # forward, so the pruned weight must not be reused across recurrent timesteps.
    _supports_recurrent_weight_cache = False

    def __init__(
        self,
        in_features,
        out_features,
        K,
        param_space="log",
        init_method="xavier_normal",
        init_gain=1.0,
        noise_level=0.0,
        weight_transform: WeightTransformType = "exp",
        weight_norm_order: int | None = None,
        gamma: float = 1.0,
        forbidden_input_index_per_output: torch.Tensor | None = None,
        connection_mask: torch.Tensor | None = None,
        initial_density: float = 1.0,
        initial_k: int | None = None,
        start_step: int = 0,
        end_step: int = 1000,
        update_interval: int = 1,
        schedule: str = "cubic",
        freeze_on_end: bool = False,
        advance_on_forward: bool = True,
        prune_metric: str = "weight",
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            K=K,
            param_space=param_space,
            init_method=init_method,
            init_gain=init_gain,
            noise_level=noise_level,
            weight_transform=weight_transform,
            weight_norm_order=weight_norm_order,
            gamma=gamma,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
        )

        self.target_k = int(self.K)
        self.initial_k = self._resolve_initial_k(initial_k, initial_density)
        self.start_step = max(0, int(start_step))
        self.end_step = max(self.start_step, int(end_step))
        self.update_interval = max(1, int(update_interval))
        self.schedule = str(schedule).lower()
        self.freeze_on_end = bool(freeze_on_end)
        self.advance_on_forward = bool(advance_on_forward)
        self.prune_metric = str(prune_metric).lower()

        if self.schedule not in {"linear", "cubic", "cosine", "step"}:
            raise ValueError(
                "dense-to-sparse schedule must be one of "
                "('linear', 'cubic', 'cosine', 'step')"
            )
        if self.prune_metric not in {
            "weight",
            "magnitude",
            "transformed_weight",
            "pre_w",
            "raw",
        }:
            raise ValueError(
                "dense-to-sparse prune_metric must be one of "
                "('weight', 'magnitude', 'transformed_weight', 'pre_w', 'raw')"
            )

        self.register_buffer(
            "sparsity_step", torch.zeros((), dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "_frozen_weight_mask",
            torch.zeros(out_features, in_features, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "_mask_is_frozen", torch.zeros((), dtype=torch.bool), persistent=True
        )

    def _resolve_initial_k(self, initial_k: int | None, initial_density: float) -> int:
        if initial_k is None:
            density = float(initial_density)
            if not 0.0 < density <= 1.0:
                raise ValueError("initial_density must satisfy 0 < density <= 1")
            initial = math.ceil(self.in_features * density)
        else:
            initial = int(initial_k)

        if initial < 1:
            raise ValueError("initial_k must be >= 1")
        if initial > self.in_features:
            raise ValueError("initial_k must be <= in_features")
        if initial < self.target_k:
            raise ValueError("initial_k/initial_density must be >= target K")
        return initial

    @property
    def current_k(self) -> int:
        """Return the active synapse count for the current schedule step."""
        step = int(self.sparsity_step.item())
        if step < self.start_step:
            return self.initial_k
        if self.end_step <= self.start_step:
            return self.target_k

        interval_step = ((step - self.start_step) // self.update_interval) * (
            self.update_interval
        )
        progress = max(
            0.0,
            min(
                1.0,
                interval_step / float(self.end_step - self.start_step),
            ),
        )
        if self.schedule == "step":
            shaped = 0.0 if progress < 1.0 else 1.0
        elif self.schedule == "cosine":
            shaped = 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.schedule == "cubic":
            # Cubic gradual pruning: stay denser early, then commit near the end.
            return max(
                self.target_k,
                min(
                    self.initial_k,
                    math.ceil(
                        self.target_k
                        + (self.initial_k - self.target_k) * ((1.0 - progress) ** 3)
                    ),
                ),
            )
        else:
            shaped = progress

        current = self.initial_k - (self.initial_k - self.target_k) * shaped
        return max(self.target_k, min(self.initial_k, math.ceil(current)))

    def sparsity_progress(self) -> float:
        """Return raw schedule progress in ``[0, 1]``."""
        if self.end_step <= self.start_step:
            return 1.0
        step = int(self.sparsity_step.item())
        return max(
            0.0,
            min(1.0, (step - self.start_step) / float(self.end_step - self.start_step)),
        )

    def _scores_for_dense_to_sparse(self) -> torch.Tensor:
        if self.prune_metric in {"weight", "transformed_weight", "magnitude"}:
            scores = self.weight()
            if self.prune_metric == "magnitude" or self.weight_transform == "identity":
                scores = scores.abs()
        else:
            scores = self.pre_w
            if self.weight_transform == "identity":
                scores = scores.abs()
        if self.noise_level > 0:
            scores = scores + torch.randn_like(scores) * self.noise_level
        return self._apply_forbidden_scores(scores)

    def _mask_for_k(self, k_value: int) -> torch.Tensor:
        topk_indices = torch.topk(
            self._scores_for_dense_to_sparse(),
            int(k_value),
            dim=-1,
            largest=True,
            sorted=False,
        )[1]
        mask = torch.zeros_like(
            self.pre_w,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        rows = torch.arange(self.pre_w.shape[0], device=self.pre_w.device)[:, None]
        mask[rows, topk_indices] = 1
        return self._apply_forbidden_mask(mask)

    def weight_mask(self):
        """Return the current dense-to-sparse mask."""
        if bool(self._mask_is_frozen.item()):
            return self._frozen_weight_mask.to(
                device=self.pre_w.device,
                dtype=self.pre_w.dtype,
            )
        return self._mask_for_k(self.current_k)

    def _maybe_freeze_final_mask(self) -> None:
        if not self.freeze_on_end or bool(self._mask_is_frozen.item()):
            return
        if int(self.sparsity_step.item()) < self.end_step:
            return
        with torch.no_grad():
            final_mask = self._mask_for_k(self.target_k).to(dtype=torch.bool)
            self._frozen_weight_mask.copy_(final_mask)
            self._mask_is_frozen.fill_(True)

    def advance_sparsity_schedule(self, steps: int = 1) -> None:
        """Advance the pruning schedule by ``steps``."""
        steps = int(steps)
        if steps < 0:
            raise ValueError("steps must be non-negative")
        if steps == 0:
            return
        with torch.no_grad():
            self.sparsity_step.add_(steps)
        self._maybe_freeze_final_mask()

    def set_sparsity_step(self, step: int) -> None:
        """Set the schedule step explicitly."""
        step = int(step)
        if step < 0:
            raise ValueError("step must be non-negative")
        with torch.no_grad():
            self.sparsity_step.fill_(step)
            if step < self.end_step:
                self._mask_is_frozen.fill_(False)
                self._frozen_weight_mask.zero_()
        self._maybe_freeze_final_mask()

    def apply_rewiring(self):
        """Compatibility hook used by model-wide rewiring calls."""
        if not self.advance_on_forward:
            self.advance_sparsity_schedule()

    def forward(self, x):
        out = super().forward(x)
        if self.training and self.advance_on_forward:
            self.advance_sparsity_schedule()
        return out


__all__ = ["DenseToSparseLinear"]

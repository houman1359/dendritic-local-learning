"""Homeostasis delegates for local credit assignment."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_homeostasis import (
    compute_gate_homeostasis_aux_grads,
    compute_inhibitory_homeostasis_factor,
    compute_voltage_homeostasis_error,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_rho_mixin import (
    LocalLearningRhoMixin,
)


class LocalLearningHomeostasisMixin(LocalLearningRhoMixin):
    """Trainer-bound wrappers for local homeostasis helper functions."""

    def _compute_inhibitory_homeostasis_factor(
        self,
        rec: dict[str, Any],
        R_tot: torch.Tensor | float,
        v_n: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute an auxiliary local inhibitory-homeostasis gradient factor."""

        return compute_inhibitory_homeostasis_factor(self.local_cfg, rec, R_tot, v_n)

    def _compute_voltage_homeostasis_error(
        self, v_n: torch.Tensor
    ) -> torch.Tensor | None:
        """Return a local voltage-centering error signal."""

        return compute_voltage_homeostasis_error(self.local_cfg, v_n)

    def _compute_gate_homeostasis_aux_grads(
        self,
        react_module: nn.Module | None,
        v_n: torch.Tensor | None,
        v_out: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return strictly local auxiliary grads for reactivation parameters."""

        return compute_gate_homeostasis_aux_grads(
            self.local_cfg, react_module, v_n, v_out
        )


__all__ = ["LocalLearningHomeostasisMixin"]

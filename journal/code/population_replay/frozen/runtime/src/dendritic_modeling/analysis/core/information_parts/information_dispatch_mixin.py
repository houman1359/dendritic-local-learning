"""Computation-level dispatch helpers for information analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from dendritic_modeling.models import BaseModel


class InformationDispatchMixin:
    """Dispatches information analysis to the configured granularity."""

    def _compute_results_for_level(
        self,
        *,
        model: BaseModel,
        inputs: torch.Tensor,
        C: np.ndarray,
        entropy_C: float | None,
    ) -> dict[str, Any] | None:
        """Dispatch to the configured information-analysis granularity."""
        if self.computation_level == "single_branch":
            return self._compute_single_branch_results(C, entropy_C)

        if self.computation_level == "layer_branch":
            return self._compute_layer_branch_results(C)

        return self._compute_all_branch_results(
            model=model,
            inputs=inputs,
            C=C,
        )


__all__ = ["InformationDispatchMixin"]

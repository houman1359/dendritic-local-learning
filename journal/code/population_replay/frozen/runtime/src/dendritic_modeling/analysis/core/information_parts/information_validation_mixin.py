"""Input validation helpers for information analysis."""

from __future__ import annotations

import numpy as np

from dendritic_modeling.analysis.core.information_parts.information_helpers import (
    ensure_2d,
)
from dendritic_modeling.analysis.core.information_parts.information_result_helpers_mixin import (
    InformationResultHelpersMixin,
)
from dendritic_modeling.analysis.utils.einet_core import has_einet_core
from dendritic_modeling.models import BaseModel


class InformationValidationMixin(InformationResultHelpersMixin):
    """Validates and normalizes array shapes before metric computation."""

    def _validate_inputs(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
    ):
        """Validate input arrays."""
        n_samples = len(E)

        if len(I_var) != n_samples:
            raise ValueError(
                f"I must have same number of samples as E: {len(I_var)} vs {n_samples}"
            )
        if len(Vout) != n_samples:
            raise ValueError(
                f"Vout must have same number of samples as E: {len(Vout)} vs {n_samples}"
            )
        if len(C) != n_samples:
            raise ValueError(
                f"C must have same number of samples as E: {len(C)} vs {n_samples}"
            )
        if Vinf is not None and len(Vinf) != n_samples:
            raise ValueError(
                f"Vinf must have same number of samples as E: {len(Vinf)} vs {n_samples}"
            )

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D."""
        return ensure_2d(arr)

    def _model_supports_information_analysis(self, model: BaseModel) -> bool:
        """Return whether a model exposes the EI core needed for information analysis."""
        if has_einet_core(model):
            return True

        self.logger.warning(
            "Model does not expose an ExcitationInhibitionNetwork. Skipping information analysis."
        )
        return False


__all__ = ["InformationValidationMixin"]

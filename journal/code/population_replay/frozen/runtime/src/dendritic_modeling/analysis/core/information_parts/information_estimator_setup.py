"""Estimator construction helpers for information analysis."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.analysis.core.information_parts.information_config import (
    _build_information_estimator_method_params,
)
from dendritic_modeling.utils.information.estimators import (
    EstimatorConfig,
    InformationEstimatorFactory,
)


def _build_information_estimators(
    *,
    params: Any,
    estimator_cfg: Any,
    method: str,
    n_neighbors: int,
    n_bins: int,
    copula_type: str,
    verbose: bool,
) -> dict[str, Any]:
    """Build the estimator registry used by the analyzer."""
    method_params = _build_information_estimator_method_params(
        params=params,
        estimator_cfg=estimator_cfg,
        method=method,
    )
    estimator_config = EstimatorConfig(
        method=method,
        n_neighbors=n_neighbors,
        n_bins=n_bins,
        copula_type=copula_type,
        verbose=verbose,
        method_params=method_params,
    )
    return {method: InformationEstimatorFactory.create(estimator_config)}


__all__ = ["_build_information_estimators"]

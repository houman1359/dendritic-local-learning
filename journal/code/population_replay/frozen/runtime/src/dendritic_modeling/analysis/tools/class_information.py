"""Matched-support mutual information between representations and class labels.

This reusable array-level analyzer is intended for depth comparisons where the
feature support is selected independently of labels and has the same width at
every depth.  It estimates the *joint* information ``I(X; C)`` between the
complete supplied feature vector and a discrete class label using the
continuous-discrete Kraskov/Ross k-nearest-neighbour estimator already used by
the codebase.  A label-permutation distribution exposes finite-sample bias.

The analyzer does not choose features, depths, checkpoints, or mechanisms.  A
caller must establish and record that design before passing arrays here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dendritic_modeling.utils.information.estimators import (
    EstimationMethod,
    EstimatorConfig,
    InformationEstimatorFactory,
)


def _validated_information_inputs(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    targets = np.asarray(labels).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError("Information features must be two-dimensional")
    if matrix.shape[0] != targets.shape[0]:
        raise ValueError("Information feature/label counts do not match")
    if matrix.shape[0] < 3 or matrix.shape[1] == 0:
        raise ValueError("Information features require samples and coordinates")
    if not np.isfinite(matrix).all():
        raise ValueError("Information features must all be finite")
    if not np.isfinite(targets.astype(float)).all():
        raise ValueError("Information labels must all be finite")
    unique, counts = np.unique(targets, return_counts=True)
    if unique.size < 2 or int(counts.min()) < 2:
        raise ValueError("Information labels require at least two populated classes")
    return matrix, targets.astype(np.int64, copy=False)


def matched_support_class_information(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    n_neighbors: int,
    n_label_shuffles: int,
    seed: int,
    standardize: bool = True,
) -> dict[str, Any]:
    """Estimate joint ``I(X; C)`` and a label-permutation baseline in bits.

    The Ross estimator uses distances in the joint representation. Its
    finite-sample estimate therefore depends on arbitrary coordinate units
    unless coordinates are put on a common scale. By default, every coordinate
    is centered and divided by its population standard deviation using the
    same label-free sample supplied for estimation. Constant coordinates stay
    zero. The transformed matrix is reused for the observed and all shuffled
    label estimates.
    """

    matrix, targets = _validated_information_inputs(values, labels)
    if int(n_neighbors) < 1:
        raise ValueError("n_neighbors must be positive")
    if int(n_label_shuffles) < 1:
        raise ValueError("At least one label shuffle is required")
    zero_variance_coordinate_count = 0
    if bool(standardize):
        center = np.mean(matrix, axis=0, dtype=np.float64)
        scale = np.std(matrix, axis=0, dtype=np.float64)
        zero_variance = scale <= np.finfo(np.float64).eps
        zero_variance_coordinate_count = int(np.sum(zero_variance))
        safe_scale = scale.copy()
        safe_scale[zero_variance] = 1.0
        matrix = (matrix - center) / safe_scale
        if zero_variance_coordinate_count:
            matrix[:, zero_variance] = 0.0
        if not np.isfinite(matrix).all():
            raise ValueError("Standardized information features must all be finite")
    estimator = InformationEstimatorFactory.create(
        EstimatorConfig(
            method=EstimationMethod.KRASKOV,
            n_neighbors=int(n_neighbors),
        )
    )
    labels_column = targets.reshape(-1, 1)
    observed_nats = float(estimator.mutual_information(matrix, labels_column))
    null_mean_nats, null_std_nats, null_values_nats = estimator.mutual_information_null(
        matrix,
        labels_column,
        n_shuffles=int(n_label_shuffles),
        seed=int(seed),
        return_values=True,
    )
    nats_to_bits = 1.0 / np.log(2.0)
    observed_bits = observed_nats * nats_to_bits
    null_mean_bits = float(null_mean_nats) * nats_to_bits
    null_std_bits = float(null_std_nats) * nats_to_bits
    null_values_bits = np.asarray(null_values_nats, dtype=float) * nats_to_bits
    return {
        "estimator": "Kraskov/Ross continuous-discrete k-nearest-neighbour MI",
        "estimand": "joint mutual information I(X; C)",
        "information_units": "bits",
        "feature_support_semantics": (
            "all supplied coordinates are one joint representation; not a sum "
            "or mean of featurewise mutual information"
        ),
        "preprocessing": (
            "per-coordinate validation-sample z-score (population standard "
            "deviation); constant coordinates set to zero"
            if bool(standardize)
            else "none"
        ),
        "preprocessing_uses_labels": False,
        "standardized_per_coordinate": bool(standardize),
        "zero_variance_coordinate_count": zero_variance_coordinate_count,
        "n_samples": int(matrix.shape[0]),
        "feature_count": int(matrix.shape[1]),
        "n_neighbors": int(n_neighbors),
        "n_label_shuffles": int(n_label_shuffles),
        "label_shuffle_seed": int(seed),
        "observed_mi_bits": float(observed_bits),
        "label_shuffle_mi_mean_bits": float(null_mean_bits),
        "label_shuffle_mi_std_bits": float(null_std_bits),
        "label_shuffle_mi_bits": null_values_bits.tolist(),
        # Do not clip: a signed excess estimate makes estimator noise and a
        # below-null result visible instead of manufacturing positive signal.
        "excess_mi_over_label_shuffle_bits": float(observed_bits - null_mean_bits),
    }


@dataclass(frozen=True)
class MatchedClassInformationAnalyzer:
    """Configured facade for matched-support class-information estimates."""

    n_neighbors: int = 10
    n_label_shuffles: int = 20
    seed: int = 0
    standardize: bool = True

    def estimate(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        *,
        seed_offset: int = 0,
    ) -> dict[str, Any]:
        return matched_support_class_information(
            values,
            labels,
            n_neighbors=self.n_neighbors,
            n_label_shuffles=self.n_label_shuffles,
            seed=int(self.seed) + int(seed_offset),
            standardize=self.standardize,
        )


__all__ = [
    "MatchedClassInformationAnalyzer",
    "matched_support_class_information",
]

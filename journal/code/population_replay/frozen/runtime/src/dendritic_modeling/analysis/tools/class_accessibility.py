"""Held-out linear class-accessibility analysis for internal representations.

This module deliberately calls the quantity *class accessibility*, not mutual
information.  A fixed linear decoder can establish that class structure is
available to a downstream readout under a declared feature budget; it does not
decompose information or prove that the trained network uses that structure.

The validation-to-test estimator is the primary publication-safe path.  Its
feature support must be selected without test labels before this module is
called.  Label-shuffle controls permute only decoder-fitting labels and are
then evaluated against the untouched true test labels.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_SKLEARN_RANDOM_STATE_MODULUS = 2**32


def _bounded_sklearn_seed(seed: int) -> int:
    """Map deterministic integer seeds into scikit-learn's uint32 domain."""

    return int(seed) % _SKLEARN_RANDOM_STATE_MODULUS


def _aligned_feature_matrix(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float32)
    targets = np.asarray(labels).reshape(-1)
    if matrix.ndim != 2:
        raise ValueError(f"{name} features must be two-dimensional")
    if matrix.shape[0] != targets.shape[0]:
        raise ValueError(f"{name} feature/label counts do not match")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} features cannot be empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} features must all be finite")
    return matrix, targets


def _ridge_decoder(ridge_alpha: float):
    return make_pipeline(
        StandardScaler(),
        RidgeClassifier(alpha=float(ridge_alpha), class_weight="balanced"),
    )


def _mlp_decoder(
    *,
    hidden_dims: tuple[int, ...],
    activation: str,
    alpha: float,
    learning_rate_init: float,
    batch_size: int,
    max_iter: int,
    early_stopping: bool,
    validation_fraction: float,
    n_iter_no_change: int,
    seed: int,
):
    if not hidden_dims or any(int(width) < 1 for width in hidden_dims):
        raise ValueError("MLP hidden_dims must contain positive widths")
    if activation not in {"identity", "logistic", "tanh", "relu"}:
        raise ValueError(f"Unsupported MLP activation: {activation!r}")
    if float(alpha) < 0.0:
        raise ValueError("MLP alpha cannot be negative")
    if float(learning_rate_init) <= 0.0:
        raise ValueError("MLP learning_rate_init must be positive")
    if int(batch_size) < 1 or int(max_iter) < 1 or int(n_iter_no_change) < 1:
        raise ValueError("MLP batch_size and iteration limits must be positive")
    if not 0.0 < float(validation_fraction) < 1.0:
        raise ValueError("MLP validation_fraction must lie strictly between 0 and 1")
    return make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=tuple(int(width) for width in hidden_dims),
            activation=activation,
            solver="adam",
            alpha=float(alpha),
            batch_size=int(batch_size),
            learning_rate_init=float(learning_rate_init),
            max_iter=int(max_iter),
            shuffle=True,
            random_state=_bounded_sklearn_seed(seed),
            early_stopping=bool(early_stopping),
            validation_fraction=float(validation_fraction),
            n_iter_no_change=int(n_iter_no_change),
        ),
    )


def cross_validated_linear_decode(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    folds: int,
    seed: int,
    ridge_alpha: float,
    n_label_shuffles: int,
) -> dict[str, Any]:
    """Return stratified fold-held-out ridge decoding and a shuffle floor."""

    values, labels = _aligned_feature_matrix(values, labels, name="Decoder")
    if int(folds) < 2:
        raise ValueError("At least two cross-validation folds are required")
    _, class_counts = np.unique(labels, return_counts=True)
    if int(class_counts.min()) < int(folds):
        raise ValueError(
            f"Every class needs at least {int(folds)} samples for stratified CV"
        )
    if int(n_label_shuffles) < 0:
        raise ValueError("n_label_shuffles cannot be negative")

    splitter = StratifiedKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=_bounded_sklearn_seed(seed),
    )
    splits = list(splitter.split(values, labels))

    def _scores(targets: np.ndarray) -> tuple[list[float], list[float]]:
        balanced_scores: list[float] = []
        accuracy_scores: list[float] = []
        for train_indices, heldout_indices in splits:
            decoder = _ridge_decoder(ridge_alpha)
            decoder.fit(values[train_indices], targets[train_indices])
            predictions = decoder.predict(values[heldout_indices])
            balanced_scores.append(
                float(
                    balanced_accuracy_score(
                        targets[heldout_indices], predictions, adjusted=False
                    )
                )
            )
            accuracy_scores.append(
                float(accuracy_score(targets[heldout_indices], predictions))
            )
        return balanced_scores, accuracy_scores

    observed_balanced, observed_accuracy = _scores(labels)
    rng = np.random.default_rng(int(seed) + 104729)
    null_balanced: list[list[float]] = []
    null_accuracy: list[list[float]] = []
    for _ in range(int(n_label_shuffles)):
        balanced, accuracy = _scores(rng.permutation(labels))
        null_balanced.append(balanced)
        null_accuracy.append(accuracy)

    flat_null_balanced = [score for scores in null_balanced for score in scores]
    flat_null_accuracy = [score for scores in null_accuracy for score in scores]
    return {
        "estimator": "StandardScaler + RidgeClassifier",
        "interpretation": "cross-validated linear decoding; not mutual information",
        "folds": int(folds),
        "splitter": "StratifiedKFold(shuffle=True)",
        "cv_seed": int(seed),
        "ridge_alpha": float(ridge_alpha),
        "n_label_shuffles": int(n_label_shuffles),
        "observed_balanced_accuracy": float(np.mean(observed_balanced)),
        "observed_accuracy": float(np.mean(observed_accuracy)),
        "observed_balanced_accuracy_by_fold": observed_balanced,
        "observed_accuracy_by_fold": observed_accuracy,
        "label_shuffle_balanced_accuracy": (
            float(np.mean(flat_null_balanced)) if flat_null_balanced else None
        ),
        "label_shuffle_accuracy": (
            float(np.mean(flat_null_accuracy)) if flat_null_accuracy else None
        ),
        "label_shuffle_balanced_accuracy_by_shuffle_and_fold": null_balanced,
        "label_shuffle_accuracy_by_shuffle_and_fold": null_accuracy,
    }


def validation_to_test_linear_decode(
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
    ridge_alpha: float,
    n_label_shuffles: int,
) -> dict[str, Any]:
    """Fit a fixed ridge probe on validation and evaluate on untouched test."""

    validation_values, validation_labels = _aligned_feature_matrix(
        validation_values,
        validation_labels,
        name="Validation",
    )
    test_values, test_labels = _aligned_feature_matrix(
        test_values,
        test_labels,
        name="Test",
    )
    if validation_values.shape[1] != test_values.shape[1]:
        raise ValueError("Validation and test feature counts must match")
    if int(n_label_shuffles) < 0:
        raise ValueError("n_label_shuffles cannot be negative")

    def _fit_score(targets: np.ndarray) -> tuple[float, float]:
        decoder = _ridge_decoder(ridge_alpha)
        decoder.fit(validation_values, targets)
        predictions = decoder.predict(test_values)
        return (
            float(balanced_accuracy_score(test_labels, predictions, adjusted=False)),
            float(accuracy_score(test_labels, predictions)),
        )

    observed_balanced, observed_accuracy = _fit_score(validation_labels)
    rng = np.random.default_rng(int(seed) + 130363)
    null_scores = [
        _fit_score(rng.permutation(validation_labels))
        for _ in range(int(n_label_shuffles))
    ]
    return {
        "primary_estimator": "validation-fit StandardScaler + RidgeClassifier",
        "primary_interpretation": (
            "fixed validation-to-test linear decoding; not mutual information"
        ),
        "primary_fit_split": "valid",
        "primary_evaluation_split": "test",
        "primary_test_used_for_selection": False,
        "heldout_test_balanced_accuracy": observed_balanced,
        "heldout_test_accuracy": observed_accuracy,
        "heldout_test_label_shuffle_balanced_accuracy": (
            float(np.mean([score[0] for score in null_scores])) if null_scores else None
        ),
        "heldout_test_label_shuffle_accuracy": (
            float(np.mean([score[1] for score in null_scores])) if null_scores else None
        ),
        "heldout_test_label_shuffle_balanced_accuracy_by_shuffle": [
            score[0] for score in null_scores
        ],
        "heldout_test_label_shuffle_accuracy_by_shuffle": [
            score[1] for score in null_scores
        ],
        "label_shuffle_semantics": (
            "permute validation training labels; score against true test labels"
        ),
    }


def validation_to_test_probe_decode(
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    test_values: np.ndarray,
    test_labels: np.ndarray,
    *,
    probe_type: str,
    seed: int,
    n_label_shuffles: int,
    ridge_alpha: float = 1.0,
    mlp_hidden_dims: tuple[int, ...] = (32, 16),
    mlp_activation: str = "relu",
    mlp_alpha: float = 1e-4,
    mlp_learning_rate_init: float = 1e-3,
    mlp_batch_size: int = 256,
    mlp_max_iter: int = 300,
    mlp_early_stopping: bool = True,
    mlp_validation_fraction: float = 0.1,
    mlp_n_iter_no_change: int = 20,
) -> dict[str, Any]:
    """Fit a declared posthoc probe on validation and score untouched test.

    This function measures representation accessibility. The MLP option is a
    nonlinear sensitivity analysis and must not be interpreted as mutual
    information or as evidence that the trained network uses the fitted probe.
    """

    probe_type = str(probe_type).lower()
    if probe_type == "linear":
        probe_type = "ridge"
    if probe_type == "ridge":
        result = validation_to_test_linear_decode(
            validation_values,
            validation_labels,
            test_values,
            test_labels,
            seed=seed,
            ridge_alpha=ridge_alpha,
            n_label_shuffles=n_label_shuffles,
        )
        return {
            "probe_type": "ridge",
            "probe_family": "linear",
            "probe_seed": int(seed),
            "probe_hyperparameters": {"ridge_alpha": float(ridge_alpha)},
            **result,
        }
    if probe_type != "mlp":
        raise ValueError(f"Unsupported accessibility probe: {probe_type!r}")

    validation_values, validation_labels = _aligned_feature_matrix(
        validation_values,
        validation_labels,
        name="Validation",
    )
    test_values, test_labels = _aligned_feature_matrix(
        test_values,
        test_labels,
        name="Test",
    )
    if validation_values.shape[1] != test_values.shape[1]:
        raise ValueError("Validation and test feature counts must match")
    if int(n_label_shuffles) < 0:
        raise ValueError("n_label_shuffles cannot be negative")

    hidden_dims = tuple(int(width) for width in mlp_hidden_dims)
    hyperparameters = {
        "hidden_dims": list(hidden_dims),
        "activation": str(mlp_activation),
        "alpha": float(mlp_alpha),
        "learning_rate_init": float(mlp_learning_rate_init),
        "batch_size": int(mlp_batch_size),
        "max_iter": int(mlp_max_iter),
        "early_stopping": bool(mlp_early_stopping),
        "validation_fraction": float(mlp_validation_fraction),
        "n_iter_no_change": int(mlp_n_iter_no_change),
    }

    def _fit_score(
        targets: np.ndarray,
        *,
        fit_seed: int,
    ) -> tuple[float, float, dict[str, Any]]:
        fit_seed = _bounded_sklearn_seed(fit_seed)
        decoder = _mlp_decoder(
            hidden_dims=hidden_dims,
            activation=str(mlp_activation),
            alpha=mlp_alpha,
            learning_rate_init=mlp_learning_rate_init,
            batch_size=mlp_batch_size,
            max_iter=mlp_max_iter,
            early_stopping=mlp_early_stopping,
            validation_fraction=mlp_validation_fraction,
            n_iter_no_change=mlp_n_iter_no_change,
            seed=fit_seed,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            decoder.fit(validation_values, targets)
        predictions = decoder.predict(test_values)
        fitted_mlp = decoder.named_steps["mlpclassifier"]
        fit_metadata = {
            "fit_seed": int(fit_seed),
            "iterations": int(fitted_mlp.n_iter_),
            "final_loss": float(fitted_mlp.loss_),
            "convergence_warning": any(
                issubclass(warning.category, ConvergenceWarning) for warning in caught
            ),
        }
        return (
            float(balanced_accuracy_score(test_labels, predictions, adjusted=False)),
            float(accuracy_score(test_labels, predictions)),
            fit_metadata,
        )

    observed_balanced, observed_accuracy, observed_fit = _fit_score(
        validation_labels,
        fit_seed=int(seed),
    )
    rng = np.random.default_rng(int(seed) + 130363)
    null_scores = [
        _fit_score(
            rng.permutation(validation_labels),
            fit_seed=int(seed) + 104729 * (shuffle_index + 1),
        )
        for shuffle_index in range(int(n_label_shuffles))
    ]
    return {
        "probe_type": "mlp",
        "probe_family": "nonlinear",
        "probe_seed": int(seed),
        "probe_hyperparameters": hyperparameters,
        "primary_estimator": "validation-fit StandardScaler + MLPClassifier",
        "primary_interpretation": (
            "fixed validation-to-test nonlinear decoding; not mutual information"
        ),
        "primary_fit_split": "valid",
        "primary_evaluation_split": "test",
        "primary_test_used_for_selection": False,
        "heldout_test_balanced_accuracy": observed_balanced,
        "heldout_test_accuracy": observed_accuracy,
        "heldout_test_label_shuffle_balanced_accuracy": (
            float(np.mean([score[0] for score in null_scores])) if null_scores else None
        ),
        "heldout_test_label_shuffle_accuracy": (
            float(np.mean([score[1] for score in null_scores])) if null_scores else None
        ),
        "heldout_test_label_shuffle_balanced_accuracy_by_shuffle": [
            score[0] for score in null_scores
        ],
        "heldout_test_label_shuffle_accuracy_by_shuffle": [
            score[1] for score in null_scores
        ],
        "observed_fit": observed_fit,
        "label_shuffle_fit": [score[2] for score in null_scores],
        "label_shuffle_semantics": (
            "permute validation training labels; score against true test labels"
        ),
    }


@dataclass(frozen=True)
class LinearClassAccessibilityAnalyzer:
    """Configured facade for the reusable array-level accessibility estimators."""

    ridge_alpha: float = 1.0
    n_label_shuffles: int = 20
    seed: int = 0

    def validation_to_test(
        self,
        validation_values: np.ndarray,
        validation_labels: np.ndarray,
        test_values: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict[str, Any]:
        return validation_to_test_linear_decode(
            validation_values,
            validation_labels,
            test_values,
            test_labels,
            seed=self.seed,
            ridge_alpha=self.ridge_alpha,
            n_label_shuffles=self.n_label_shuffles,
        )

    def cross_validated(
        self,
        values: np.ndarray,
        labels: np.ndarray,
        *,
        folds: int,
    ) -> dict[str, Any]:
        return cross_validated_linear_decode(
            values,
            labels,
            folds=folds,
            seed=self.seed,
            ridge_alpha=self.ridge_alpha,
            n_label_shuffles=self.n_label_shuffles,
        )


@dataclass(frozen=True)
class ClassAccessibilityProbeAnalyzer:
    """Array-level validation-to-test facade for linear and nonlinear probes."""

    probe_type: str = "ridge"
    n_label_shuffles: int = 20
    seed: int = 0
    ridge_alpha: float = 1.0
    mlp_hidden_dims: tuple[int, ...] = (32, 16)
    mlp_activation: str = "relu"
    mlp_alpha: float = 1e-4
    mlp_learning_rate_init: float = 1e-3
    mlp_batch_size: int = 256
    mlp_max_iter: int = 300
    mlp_early_stopping: bool = True
    mlp_validation_fraction: float = 0.1
    mlp_n_iter_no_change: int = 20

    def validation_to_test(
        self,
        validation_values: np.ndarray,
        validation_labels: np.ndarray,
        test_values: np.ndarray,
        test_labels: np.ndarray,
    ) -> dict[str, Any]:
        return validation_to_test_probe_decode(
            validation_values,
            validation_labels,
            test_values,
            test_labels,
            probe_type=self.probe_type,
            seed=self.seed,
            n_label_shuffles=self.n_label_shuffles,
            ridge_alpha=self.ridge_alpha,
            mlp_hidden_dims=self.mlp_hidden_dims,
            mlp_activation=self.mlp_activation,
            mlp_alpha=self.mlp_alpha,
            mlp_learning_rate_init=self.mlp_learning_rate_init,
            mlp_batch_size=self.mlp_batch_size,
            mlp_max_iter=self.mlp_max_iter,
            mlp_early_stopping=self.mlp_early_stopping,
            mlp_validation_fraction=self.mlp_validation_fraction,
            mlp_n_iter_no_change=self.mlp_n_iter_no_change,
        )


__all__ = [
    "ClassAccessibilityProbeAnalyzer",
    "LinearClassAccessibilityAnalyzer",
    "cross_validated_linear_decode",
    "validation_to_test_linear_decode",
    "validation_to_test_probe_decode",
]

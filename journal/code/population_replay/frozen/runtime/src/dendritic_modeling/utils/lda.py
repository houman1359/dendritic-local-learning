"""
Non-negative LDA utilities for dendritic network analysis and initialization.

This module provides functions for fitting Linear Discriminant Analysis (LDA)
with non-negative weight constraints, which is appropriate for neural network
analysis where synaptic weights are non-negative.
"""

from typing import Optional

import numpy as np
from scipy.optimize import minimize


def fit_nonnegative_lda(
    X: np.ndarray,
    y: np.ndarray,
    weight_norm_order: Optional[int] = None,
    gamma: float = 1.0,
) -> tuple[np.ndarray, float]:
    """Fit LDA with non-negative weight constraint.

    Finds the optimal non-negative linear combination of features that
    maximizes class separability (Fisher criterion). This is appropriate
    for neural network analysis where synaptic weights are non-negative.

    Args:
        X: Input features, shape [n_samples, n_features]
        y: Class labels, shape [n_samples,]
        weight_norm_order: Lp norm order for weight normalization.
                           If None, weights are normalized to sum to 1 (L1 norm = 1).
                           Common values: 1 (L1), 2 (L2).
        gamma: Scaling factor after normalization.

    Returns:
        Tuple of (weights, intercept) where weights are non-negative and properly normalized.
        The intercept is always 0.0 (included for API compatibility).
    """
    n_features = X.shape[1]
    classes = np.unique(y)
    n_classes = len(classes)

    if n_classes <= 1 or n_features == 0:
        # Cannot do LDA with single class or no features
        return np.ones(n_features) / max(n_features, 1), 0.0

    # Compute class means and overall mean
    class_means = np.array([X[y == c].mean(axis=0) for c in classes])
    overall_mean = X.mean(axis=0)

    # Compute between-class scatter matrix S_b
    # S_b = sum_c n_c * (mu_c - mu)(mu_c - mu)^T
    S_b = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        n_c = np.sum(y == c)
        diff = (class_means[i] - overall_mean).reshape(-1, 1)
        S_b += n_c * (diff @ diff.T)

    # Compute within-class scatter matrix S_w
    # S_w = sum_c sum_{x in c} (x - mu_c)(x - mu_c)^T
    S_w = np.zeros((n_features, n_features))
    for i, c in enumerate(classes):
        X_c = X[y == c]
        diff = X_c - class_means[i]
        S_w += diff.T @ diff

    # Add regularization for numerical stability
    S_w += 1e-6 * np.eye(n_features)

    # Objective: maximize w^T S_b w / w^T S_w w
    # Equivalent to minimizing -w^T S_b w subject to w^T S_w w = 1 and w >= 0
    def objective(w):
        w = w.reshape(-1)
        numerator = w @ S_b @ w
        denominator = w @ S_w @ w + 1e-10
        return -numerator / denominator

    def gradient(w):
        w = w.reshape(-1)
        Sb_w = S_b @ w
        Sw_w = S_w @ w
        num = w @ Sb_w
        den = w @ Sw_w + 1e-10
        # d/dw [-num/den] = -2*(Sb_w*den - num*Sw_w) / den^2
        grad = -2 * (Sb_w * den - num * Sw_w) / (den * den)
        return grad

    # Initialize with uniform weights
    w0 = np.ones(n_features) / n_features

    # Bounds: weights must be non-negative
    bounds = [(0, None) for _ in range(n_features)]

    # Optimize
    result = minimize(
        objective,
        w0,
        method="L-BFGS-B",
        jac=gradient,
        bounds=bounds,
        options={"maxiter": 100},
    )

    w_opt = result.x

    # Apply normalization based on weight_norm_order
    if weight_norm_order is not None:
        # Use Lp norm normalization (matching TopKLinear.pruned_weight())
        w_norm = np.linalg.norm(w_opt, ord=weight_norm_order)
        if w_norm > 1e-10:
            w_opt = (w_opt / w_norm) * gamma
        else:
            # Fallback: uniform weights with the requested norm and gamma scaling.
            w_opt = np.ones(n_features, dtype=float)
            w_norm = np.linalg.norm(w_opt, ord=weight_norm_order)
            w_opt = (w_opt / max(w_norm, 1e-10)) * gamma
    else:
        # Fallback to sum to 1 if no specific norm order
        w_sum = np.sum(w_opt)
        if w_sum > 1e-10:
            w_opt = w_opt / w_sum
        else:
            w_opt = np.ones(n_features) / n_features

    return w_opt, 0.0


def apply_nonnegative_lda(
    X: np.ndarray,
    y: np.ndarray,
    weight_norm_order: Optional[int] = None,
    gamma: float = 1.0,
) -> np.ndarray:
    """Apply non-negative LDA to get projected values.

    Args:
        X: Input features, shape [n_samples, n_features]
        y: Class labels, shape [n_samples,]
        weight_norm_order: Lp norm order for weight normalization.
        gamma: Scaling factor after normalization.

    Returns:
        Projected values, shape [n_samples,]
    """
    weights, _ = fit_nonnegative_lda(X, y, weight_norm_order, gamma)
    return X @ weights


def generate_random_nonnegative_weights(
    n_features: int,
    weight_norm_order: Optional[int] = None,
    gamma: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate random non-negative weights with same normalization as LDA.

    Creates random weights from a uniform distribution and normalizes them
    using the same scheme as LDA weights, providing a proper random baseline.

    Args:
        n_features: Number of features (weights)
        weight_norm_order: Lp norm order for weight normalization.
                           If None, weights are normalized to sum to 1 (L1 norm = 1).
        gamma: Scaling factor after normalization.
        rng: Random number generator. If None, uses numpy default.

    Returns:
        Non-negative normalized weights, shape [n_features,]
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate random non-negative weights (uniform)
    w = rng.uniform(0, 1, n_features)

    # Apply the same normalization as LDA weights
    if weight_norm_order is not None:
        w_norm = np.linalg.norm(w, ord=weight_norm_order)
        if w_norm > 1e-10:
            w = (w / w_norm) * gamma
        else:
            # Fallback: uniform weights with the requested norm and gamma scaling.
            w = np.ones(n_features, dtype=float)
            w_norm = np.linalg.norm(w, ord=weight_norm_order)
            w = (w / max(w_norm, 1e-10)) * gamma
    else:
        # Fallback to sum to 1
        w_sum = np.sum(w)
        if w_sum > 1e-10:
            w = w / w_sum
        else:
            w = np.ones(n_features) / n_features

    return w


def apply_random_weights(
    X: np.ndarray,
    weight_norm_order: Optional[int] = None,
    gamma: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply random non-negative weights to get projected values (baseline).

    Args:
        X: Input features, shape [n_samples, n_features]
        weight_norm_order: Lp norm order for weight normalization.
        gamma: Scaling factor after normalization.
        rng: Random number generator.

    Returns:
        Projected values, shape [n_samples,]
    """
    n_features = X.shape[1]
    weights = generate_random_nonnegative_weights(
        n_features, weight_norm_order, gamma, rng
    )
    return X @ weights


def compute_lda_importance_scores(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Compute per-feature importance scores using LDA Fisher criterion.

    For each feature, computes a score based on:
    - Between-class variance (how much the feature separates classes)
    - Within-class variance (how consistent the feature is within classes)

    Higher scores indicate features that are more discriminative for classification.
    This can be used for feature selection / sparsity mask initialization.

    Args:
        X: Input features, shape [n_samples, n_features]
        y: Class labels, shape [n_samples,]

    Returns:
        Importance scores for each feature, shape [n_features,]
    """
    n_features = X.shape[1]
    classes = np.unique(y)
    n_classes = len(classes)

    if n_classes <= 1 or n_features == 0:
        # Cannot compute importance with single class
        return np.ones(n_features) / max(n_features, 1)

    # Compute class means and overall mean
    class_means = np.array([X[y == c].mean(axis=0) for c in classes])
    overall_mean = X.mean(axis=0)

    # Compute between-class variance per feature
    # This measures how much the class means differ from the overall mean
    between_var = np.zeros(n_features)
    for i, c in enumerate(classes):
        n_c = np.sum(y == c)
        between_var += n_c * (class_means[i] - overall_mean) ** 2

    # Compute within-class variance per feature
    # This measures how spread out the data is within each class
    within_var = np.zeros(n_features)
    for i, c in enumerate(classes):
        X_c = X[y == c]
        within_var += np.sum((X_c - class_means[i]) ** 2, axis=0)

    # Add regularization to avoid division by zero
    within_var += 1e-10

    # Fisher criterion: between-class variance / within-class variance
    importance_scores = between_var / within_var

    return importance_scores


def select_topk_features_by_lda(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
) -> np.ndarray:
    """Select the top K most discriminative features using LDA importance.

    Args:
        X: Input features, shape [n_samples, n_features]
        y: Class labels, shape [n_samples,]
        K: Number of features to select

    Returns:
        Indices of the K most important features, shape [K,]
    """
    importance_scores = compute_lda_importance_scores(X, y)
    n_features = len(importance_scores)
    K = min(K, n_features)
    topk_indices = np.argsort(importance_scores)[-K:][::-1]  # Descending order
    return topk_indices

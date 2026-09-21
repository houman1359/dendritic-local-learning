"""
Training utilities module.

This module provides utilities for training dendritic models including:
- Backward hooks for gradient manipulation
- Evaluation functions for model performance
"""

from .evaluation import (
    accuracy_score,
    categorical_loglikelihood_score,
    evaluate_accuracy,
    evaluate_auc,
    evaluate_categorical_loglikelihood,
    evaluate_cosine_similarity,
    evaluate_mse,
    mse_score,
)

__all__ = [
    # Evaluation
    "accuracy_score",
    "categorical_loglikelihood_score",
    "evaluate_accuracy",
    "evaluate_auc",
    "evaluate_categorical_loglikelihood",
    "evaluate_cosine_similarity",
    "evaluate_mse",
    "mse_score",
]

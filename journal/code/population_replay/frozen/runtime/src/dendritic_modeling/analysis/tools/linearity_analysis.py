"""
Linearity Analysis for Dendritic Networks

Computes linearity index L(R) = 1 - E[||Hess(R)||] / (E[||grad(R)||] + eps)
to quantify how linear the branch output is relative to inputs.

Tests theoretical prediction: Shunting produces more linear outputs than non-shunting
for nonlinear input tasks.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    get_model_device,
)
from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.models.base import BaseModel

logger = logging.getLogger(__name__)


class LinearityAnalyzer(AbstractAnalyzer):
    """
    Analyze linearity of branch outputs.

    Computes linearity index via finite-difference approximation of gradients and Hessians.
    """

    def __init__(self, params: Optional[BaseConfig] = None):
        super().__init__()
        self.params = params if params is not None else BaseConfig()
        self.n_samples = getattr(self.params, "n_samples", 500)
        self.finite_diff_h = getattr(self.params, "finite_diff_h", 1e-4)
        self.epsilon = getattr(self.params, "epsilon", 1e-8)

        self.results = {}

    def analyze(
        self,
        model: BaseModel,
        data_loader: DataLoader | Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Compute linearity index for model outputs.

        Args:
            model: Trained model
            data_loader: Data loader or dataset for validation/test set
            device: Device to run analysis on
            save_path: Where to save results

        Returns:
            Dictionary with linearity metrics
        """
        logger.info("Computing linearity index...")

        data_loader = self._ensure_dataloader(data_loader)
        analysis_device = self._resolve_analysis_device(model, device)
        inputs, labels = self._collect_samples(data_loader)
        inputs = inputs.to(analysis_device)

        logger.info(f"Analyzing {len(inputs)} samples, input_dim = {inputs.shape[1]}")

        with analysis_device_context(model, analysis_device):
            linearity_index = self._compute_linearity_index(model, inputs)
            gradient_norm = self._compute_gradient_norm(model, inputs)
            hessian_norm = self._compute_hessian_norm(model, inputs)
            per_class_linearity = self._compute_per_class_linearity(
                model, inputs, labels
            )

        # Store results
        self.results = {
            "linearity_index": float(linearity_index),
            "gradient_norm_mean": float(gradient_norm),
            "hessian_norm_mean": float(hessian_norm),
            "per_class_linearity": per_class_linearity,
            "n_samples_analyzed": len(inputs),
        }

        # Save if path provided
        if save_path is not None:
            self.save_results(self.results, save_path)

        logger.info(f"Linearity index: {linearity_index:.4f}")
        logger.info(f"Gradient norm: {gradient_norm:.4f}")
        logger.info(f"Hessian norm: {hessian_norm:.4f}")

        return self.results

    @staticmethod
    def _ensure_dataloader(data_loader: DataLoader | Dataset) -> DataLoader:
        if isinstance(data_loader, DataLoader):
            return data_loader
        return DataLoader(data_loader, batch_size=256, shuffle=False)

    def _resolve_analysis_device(self, model: nn.Module, device: str) -> torch.device:
        model_device = get_model_device(model)
        analysis_device = torch.device(device)
        if analysis_device != model_device:
            analysis_device = model_device
        return analysis_device

    def _collect_samples(self, data_loader: DataLoader) -> tuple[torch.Tensor, ...]:
        all_inputs = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                x_batch = batch[0]
                all_inputs.append(x_batch)
                all_labels.append(batch[1])

                if len(all_inputs) * x_batch.shape[0] >= self.n_samples:
                    break

        inputs = torch.cat(all_inputs, dim=0)[: self.n_samples]
        labels = torch.cat(all_labels, dim=0)[: self.n_samples]
        return inputs, labels

    def _compute_per_class_linearity(
        self, model: nn.Module, inputs: torch.Tensor, labels: torch.Tensor
    ) -> dict[int, float]:
        per_class_linearity = {}
        for c in torch.unique(labels):
            mask = labels == c
            if mask.sum() > 0:
                class_inputs = inputs[mask]
                per_class_linearity[int(c)] = self._compute_linearity_index(
                    model, class_inputs
                )
        return per_class_linearity

    def _compute_linearity_index(self, model: nn.Module, inputs: torch.Tensor) -> float:
        """Compute L(R) = 1 - E[||Hess||] / (E[||grad||] + eps)"""
        grad_norm = self._compute_gradient_norm(model, inputs)
        hess_norm = self._compute_hessian_norm(model, inputs)

        linearity = 1.0 - hess_norm / (grad_norm + self.epsilon)
        return max(0.0, min(1.0, linearity))  # Clamp to [0,1]

    def _compute_gradient_norm(self, model: nn.Module, inputs: torch.Tensor) -> float:
        """Compute E[||grad R||] via finite differences"""
        h = self.finite_diff_h
        input_dim = inputs.shape[1]
        grad_norms = []

        for sample_idx in range(len(inputs)):
            x = inputs[sample_idx : sample_idx + 1]  # Keep batch dim
            grad = torch.zeros(input_dim)

            # Central difference for each dimension
            for dim in range(input_dim):
                x_plus = x.clone()
                x_minus = x.clone()
                x_plus[0, dim] += h
                x_minus[0, dim] -= h

                with torch.no_grad():
                    y_plus = model(x_plus).squeeze()
                    y_minus = model(x_minus).squeeze()

                # Handle multi-output (take norm if needed)
                if y_plus.ndim > 0:
                    grad[dim] = ((y_plus - y_minus) / (2 * h)).norm().item()
                else:
                    grad[dim] = ((y_plus - y_minus) / (2 * h)).item()

            grad_norms.append(grad.norm().item())

        return float(np.mean(grad_norms))

    def _compute_hessian_norm(self, model: nn.Module, inputs: torch.Tensor) -> float:
        """Compute E[||Hess R||_F] via finite differences"""
        h = self.finite_diff_h
        input_dim = inputs.shape[1]
        hess_norms = []

        for sample_idx in range(len(inputs)):
            x = inputs[sample_idx : sample_idx + 1]

            # Compute Hessian via second-order finite differences
            # Only compute diagonal and a subset of off-diagonal elements (full Hessian is expensive)
            hess_elements = []

            # Diagonal elements
            for dim in range(input_dim):
                x_plus = x.clone()
                x_minus = x.clone()
                x_center = x.clone()
                x_plus[0, dim] += h
                x_minus[0, dim] -= h

                with torch.no_grad():
                    y_plus = model(x_plus).squeeze()
                    y_minus = model(x_minus).squeeze()
                    y_center = model(x_center).squeeze()

                # Second derivative: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h^2
                if y_plus.ndim > 0:
                    hess_diag = ((y_plus - 2 * y_center + y_minus) / h**2).norm().item()
                else:
                    hess_diag = ((y_plus - 2 * y_center + y_minus) / h**2).item()

                hess_elements.append(hess_diag)

            # Sample some off-diagonal elements (mixed partials)
            n_off_diag_samples = min(10, input_dim * (input_dim - 1) // 2)
            for _ in range(n_off_diag_samples):
                i = np.random.randint(0, input_dim)
                j = np.random.randint(0, input_dim)
                if i == j:
                    continue

                # Mixed partial: ∂²f/∂xi∂xj ≈ [f(x+hi+hj) - f(x+hi) - f(x+hj) + f(x)] / h²
                x_ij = x.clone()
                x_i = x.clone()
                x_j = x.clone()
                x_base = x.clone()

                x_ij[0, i] += h
                x_ij[0, j] += h
                x_i[0, i] += h
                x_j[0, j] += h

                with torch.no_grad():
                    y_ij = model(x_ij).squeeze()
                    y_i = model(x_i).squeeze()
                    y_j = model(x_j).squeeze()
                    y_base = model(x_base).squeeze()

                if y_ij.ndim > 0:
                    hess_mixed = ((y_ij - y_i - y_j + y_base) / h**2).norm().item()
                else:
                    hess_mixed = ((y_ij - y_i - y_j + y_base) / h**2).item()

                hess_elements.append(hess_mixed)

            # Frobenius norm approximation (sum of squared elements)
            hess_norm = np.sqrt(np.sum(np.array(hess_elements) ** 2))
            hess_norms.append(hess_norm)

        return float(np.mean(hess_norms))

    def save_results(self, results: dict[str, Any], save_path: str):
        """Save results to file."""
        import pickle

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Linearity analysis results saved to {save_path}")


def compute_linear_decoder_performance(
    branch_outputs: np.ndarray,
    labels: np.ndarray,
    test_fraction: float = 0.3,
) -> dict[str, float]:
    """
    Train linear decoder on branch outputs and measure performance.

    Tests whether shunting outputs are more linearly separable than non-shunting.

    Args:
        branch_outputs: Branch activations (n_samples, n_branches)
        labels: Class labels (n_samples,)
        test_fraction: Fraction to use for testing

    Returns:
        Dictionary with accuracy, margin, and other metrics
    """
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import LinearSVC

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        branch_outputs,
        labels,
        test_size=test_fraction,
        random_state=42,
        stratify=labels,
    )

    # Train linear SVM
    clf = LinearSVC(max_iter=5000, random_state=42)
    clf.fit(X_train, y_train)

    # Predict
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Compute metrics
    results = {
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "margin": float(np.abs(clf.decision_function(X_test)).min()),  # Minimum margin
        "n_support_vectors": _count_linear_svc_margin_samples(clf, X_train, y_train),
    }

    logger.info(
        f"Linear decoder - Train acc: {results['train_accuracy']:.3f}, Test acc: {results['test_accuracy']:.3f}"
    )

    return results


def _count_linear_svc_margin_samples(
    clf: Any, branch_outputs: np.ndarray, labels: np.ndarray
) -> int:
    """Estimate support-like samples for LinearSVC from signed margins.

    LinearSVC does not expose explicit support vectors. For the binary case,
    samples with signed margin <= 1 are the hinge-loss active points. For
    multiclass one-vs-rest scores, use the true-class score as the same
    deterministic support-like diagnostic.
    """
    decision = clf.decision_function(branch_outputs)
    labels = np.asarray(labels)

    if np.ndim(decision) == 1:
        class_to_sign = {clf.classes_[0]: -1.0, clf.classes_[1]: 1.0}
        signs = np.array([class_to_sign[label] for label in labels], dtype=float)
        margins = signs * decision
        return int(np.sum(margins <= 1.0))

    class_to_idx = {label: idx for idx, label in enumerate(clf.classes_)}
    true_class_indices = np.array([class_to_idx[label] for label in labels])
    true_class_scores = decision[np.arange(len(labels)), true_class_indices]
    return int(np.sum(true_class_scores <= 1.0))

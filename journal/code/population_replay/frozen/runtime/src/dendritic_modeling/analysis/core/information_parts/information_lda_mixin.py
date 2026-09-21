"""LDA comparison helpers for information analysis."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from dendritic_modeling.analysis.utils.runtime import get_model_device


class InformationLdaComparisonMixin:
    """Compare trained network predictions against chained LDA projections."""

    def compare_lda_vs_trained_performance(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device = None,
    ) -> dict:
        """Compare classification performance using trained weights vs LDA-optimal weights.

        This method uses the data already collected in self.data_dict from the main analysis.
        It computes:
        1. Performance with trained network weights (original model)
        2. Performance with LDA-optimized input aggregation (Vout_lin passed through decoder)

        Args:
            model: Trained model with encoder, core_network, and decoder
            inputs: Input tensor (already processed in main analysis)
            labels: Label tensor (already processed in main analysis)
            device: Device to run computations on

        Returns:
            Dictionary with:
            - trained_accuracy: Accuracy with trained weights
            - lda_accuracy: Accuracy with LDA-optimal aggregation
            - accuracy_difference: LDA - Trained accuracy
        """
        from sklearn.metrics import accuracy_score

        if device is None:
            device = get_model_device(model)

        # Ensure model is in eval mode (same state used for all comparisons)
        model.eval()

        self.logger.info("Computing LDA vs trained weights performance comparison...")
        self.logger.info(
            "Note: Using the SAME model for both trained and LDA predictions. "
            "Only the intermediate representations differ."
        )

        labels_np = self._labels_to_numpy(labels)

        trained_preds = self._predict_trained_model_classes(model, inputs, device)
        trained_accuracy = accuracy_score(labels_np, trained_preds)

        final_vout_lin = self._compute_chained_lda_output(labels_np)
        lda_preds = self._predict_lda_model_classes(model, final_vout_lin, device)
        lda_accuracy = (
            accuracy_score(labels_np, lda_preds) if lda_preds is not None else None
        )
        results = {
            "trained_accuracy": trained_accuracy,
            "lda_accuracy": lda_accuracy,
            "trained_predictions": trained_preds.tolist(),
            "lda_predictions": lda_preds.tolist() if lda_preds is not None else None,
            "accuracy_difference": (
                lda_accuracy - trained_accuracy if lda_accuracy is not None else None
            ),
        }

        if lda_accuracy is not None:
            self.logger.info(
                f"Difference (LDA - Trained): {results['accuracy_difference']:+.4f}"
            )

        return results

    @staticmethod
    def _labels_to_numpy(
        labels: torch.Tensor | np.ndarray | Sequence[Any],
    ) -> np.ndarray:
        if hasattr(labels, "cpu"):
            return labels.cpu().numpy()
        if hasattr(labels, "numpy"):
            return labels.numpy()
        return np.array(labels)

    @staticmethod
    def _predict_trained_model_classes(
        model: torch.nn.Module, inputs: torch.Tensor, device: torch.device
    ) -> np.ndarray:
        with torch.no_grad():
            trained_outputs = model(inputs.to(device))
            return trained_outputs.argmax(dim=1).cpu().numpy()

    def _ordered_branch_layer_names(self) -> list[str]:
        layers_with_idx = []
        for layer_name in list(self.data_dict.keys()):
            idx = self._parse_branch_layer_index(layer_name)
            if idx is not None:
                layers_with_idx.append((idx, layer_name))
        layers_with_idx.sort(key=lambda x: x[0])
        return [layer_name for _idx, layer_name in layers_with_idx]

    def _compute_chained_lda_output(self, labels_np: np.ndarray) -> np.ndarray | None:
        prev_vout_lin = None
        final_vout_lin = None

        for layer_name in self._ordered_branch_layer_names():
            layer_data = self.data_dict[layer_name]
            if "excitation" not in layer_data:
                continue

            lda_result = self.compute_lda_aggregated_signals_chained(
                layer_data, labels_np, prev_vout_lin
            )

            if lda_result[0] is not None:
                _E_lin, _I_lin, _Vb_lin, Vout_lin = lda_result
                prev_vout_lin = Vout_lin
                final_vout_lin = Vout_lin
            else:
                prev_vout_lin = layer_data["output"].detach().cpu().numpy()
                final_vout_lin = prev_vout_lin

        return final_vout_lin

    @staticmethod
    def _predict_lda_model_classes(
        model: torch.nn.Module,
        final_vout_lin: np.ndarray | None,
        device: torch.device,
    ) -> np.ndarray | None:
        if final_vout_lin is None:
            return None

        with torch.no_grad():
            vout_tensor = torch.tensor(final_vout_lin, dtype=torch.float32).to(device)

            if hasattr(model, "decoder") and model.decoder is not None:
                lda_logits = model.decoder(vout_tensor)
            else:
                lda_logits = vout_tensor

            return lda_logits.argmax(dim=1).cpu().numpy()


__all__ = ["InformationLdaComparisonMixin"]

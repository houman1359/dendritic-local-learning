"""Signal aggregation helpers for information analysis."""

from __future__ import annotations

import numpy as np
import torch

from dendritic_modeling.analysis.core.information_parts.information_random_signal_mixin import (
    InformationRandomSignalMixin,
)
from dendritic_modeling.analysis.core.information_parts.information_signal_projection import (
    compute_projected_branch_output,
)
from dendritic_modeling.analysis.core.information_parts.information_signal_summary_mixin import (
    InformationSignalSummaryMixin,
)
from dendritic_modeling.utils.lda import apply_nonnegative_lda, fit_nonnegative_lda


class InformationSignalAggregationMixin(
    InformationRandomSignalMixin,
    InformationSignalSummaryMixin,
):
    """Builds LDA, random-weight, per-neuron, and E/I signal summaries."""

    def _fit_nonnegative_lda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight_norm_order: int | None = None,
        gamma: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Fit LDA with non-negative weight constraint.

        Delegates to the utility function in dendritic_modeling.utils.lda.
        """
        return fit_nonnegative_lda(X, y, weight_norm_order, gamma)

    def _apply_nonnegative_lda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight_norm_order: int | None = None,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Apply non-negative LDA to get projected values.

        Delegates to the utility function in dendritic_modeling.utils.lda.
        """
        return apply_nonnegative_lda(X, y, weight_norm_order, gamma)

    def _compute_masked_lda_projection(
        self,
        *,
        output: np.ndarray,
        raw_input: torch.Tensor,
        fallback: torch.Tensor,
        source_module,
        n_branches: int,
        C: np.ndarray,
        signal_name: str,
    ) -> np.ndarray:
        """Fit per-branch LDA on active TopK inputs and fall back to network weights."""
        raw_input_np = raw_input.detach().cpu().numpy()
        weight_norm_order = getattr(source_module, "weight_norm_order", None)
        gamma = getattr(source_module, "gamma", 1.0)
        mask = source_module.weight_mask().detach().cpu().numpy()

        for branch_idx in range(n_branches):
            branch_mask = mask[branch_idx, :]
            active_indices = np.where(branch_mask > 0)[0]

            if len(active_indices) > 0:
                branch_inputs = raw_input_np[:, active_indices]
                try:
                    output[:, branch_idx] = self._apply_nonnegative_lda(
                        branch_inputs,
                        C,
                        weight_norm_order=weight_norm_order,
                        gamma=gamma,
                    )
                except Exception as e:
                    self.logger.warning(
                        f"LDA failed for {signal_name} branch {branch_idx}: {e}. Using network weights."
                    )
                    output[:, branch_idx] = (
                        fallback[:, branch_idx].detach().cpu().numpy()
                    )
            else:
                output[:, branch_idx] = fallback[:, branch_idx].detach().cpu().numpy()

        return output

    def compute_lda_aggregated_signals(
        self,
        layer_data: dict,
        C: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute LDA-weighted versions of E, I, Vb, and Vout using TopK-masked features.

        For each branch, fit non-negative constrained LDA only on the input
        dimensions that pass through the TopK mask (i.e., the synapses that are
        actually connected). Non-negative weights match the network's positive
        synaptic weights.

        Vout_lin is computed by applying the same dendritic computation
        (shunting or additive) to E_lin, I_lin, Vb_lin, then passing through
        the network's reactivation function.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,]

        Returns:
            Tuple of (E_lin, I_lin, Vb_lin, Vout_lin) each shape [n_samples, n_branches]
        """
        module = layer_data.get("module")
        if module is None:
            return None, None, None, None

        raw_exc = layer_data.get("raw_excitatory_input")
        raw_inh = layer_data.get("raw_inhibitory_input")
        raw_branch = layer_data.get("raw_branch_input")

        n_samples = C.shape[0]
        n_branches = layer_data["excitation"].shape[1]

        E_lin = np.zeros((n_samples, n_branches))
        I_lin = np.zeros((n_samples, n_branches))
        Vb_lin = np.zeros((n_samples, n_branches))

        # Check if we have enough classes for LDA
        if len(np.unique(C)) <= 1:
            # Fall back to network weights
            E_lin = layer_data["excitation"].detach().cpu().numpy()
            I_lin = layer_data["inhibition"].detach().cpu().numpy()
            vb_fallback = layer_data.get("branch_input")
            if vb_fallback is not None:
                Vb_lin = vb_fallback.detach().cpu().numpy()
            else:
                Vb_lin = np.zeros_like(E_lin)
            Vout_lin = layer_data["output"].detach().cpu().numpy()
            return E_lin, I_lin, Vb_lin, Vout_lin

        # For E: Fit LDA on TopK-masked excitatory inputs for each branch
        if raw_exc is not None and module.branch_excitation is not None:
            E_lin = self._compute_masked_lda_projection(
                output=E_lin,
                raw_input=raw_exc,
                fallback=layer_data["excitation"],
                source_module=module.branch_excitation,
                n_branches=n_branches,
                C=C,
                signal_name="E",
            )

        # For I: Fit LDA on TopK-masked inhibitory inputs for each branch
        if raw_inh is not None and module.branch_inhibition is not None:
            I_lin = self._compute_masked_lda_projection(
                output=I_lin,
                raw_input=raw_inh,
                fallback=layer_data["inhibition"],
                source_module=module.branch_inhibition,
                n_branches=n_branches,
                C=C,
                signal_name="I",
            )

        # For Vb: Fit LDA on branch inputs using BlockLinear's block structure
        # branches_to_output is a BlockLinear where each output branch receives input
        # from a contiguous block of upstream branches
        if raw_branch is not None and module.input_branches:
            raw_branch_np = (
                raw_branch.detach().cpu().numpy()
            )  # [n_samples, n_upstream_branches]

            # BlockLinear doesn't have weight_norm_order/gamma like TopKLinear
            # Use defaults (L1 norm, gamma=1.0)
            vb_weight_norm_order = getattr(
                module.branches_to_output, "weight_norm_order", None
            )
            vb_gamma = getattr(module.branches_to_output, "gamma", 1.0)

            # Get block_size from branches_to_output (BlockLinear)
            if hasattr(module.branches_to_output, "block_size"):
                block_size = module.branches_to_output.block_size

                # Fit separate LDA for each output branch using its upstream block
                for branch_idx in range(n_branches):
                    # Each output branch receives from upstream indices [branch_idx * block_size : (branch_idx+1) * block_size]
                    start_idx = branch_idx * block_size
                    end_idx = min((branch_idx + 1) * block_size, raw_branch_np.shape[1])

                    if start_idx < raw_branch_np.shape[1]:
                        # Extract the upstream branches for this output branch
                        branch_inputs = raw_branch_np[:, start_idx:end_idx]

                        if branch_inputs.shape[1] > 0:
                            try:
                                # Fit non-negative LDA on the upstream branches
                                # Use same weight normalization as the network
                                Vb_lin[:, branch_idx] = self._apply_nonnegative_lda(
                                    branch_inputs,
                                    C,
                                    weight_norm_order=vb_weight_norm_order,
                                    gamma=vb_gamma,
                                )

                            except Exception as e:
                                self.logger.warning(
                                    f"LDA failed for Vb branch {branch_idx}: {e}. Using network weights."
                                )
                                Vb_lin[:, branch_idx] = (
                                    layer_data["branch_input"][:, branch_idx]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                        else:
                            Vb_lin[:, branch_idx] = (
                                layer_data["branch_input"][:, branch_idx]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                    else:
                        Vb_lin[:, branch_idx] = (
                            layer_data["branch_input"][:, branch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
            else:
                # Fallback: no block structure, use network weights
                Vb_lin = layer_data["branch_input"].detach().cpu().numpy()

        Vout_lin = compute_projected_branch_output(module, E_lin, I_lin, Vb_lin)

        return E_lin, I_lin, Vb_lin, Vout_lin

    def compute_lda_aggregated_signals_chained(
        self,
        layer_data: dict,
        C: np.ndarray,
        prev_layer_vout_lin: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute LDA-weighted signals with optional chained branch input.

        This is similar to compute_lda_aggregated_signals but allows overriding
        the branch input with Vout_lin from a previous layer, enabling proper
        chained LDA comparison across all network layers.

        Uses non-negative LDA weights to match network's non-negative synaptic weights.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,]
            prev_layer_vout_lin: Vout_lin from previous layer, to use as branch input.
                                 If None, uses the raw_branch_input from layer_data.

        Returns:
            Tuple of (E_lin, I_lin, Vb_lin, Vout_lin) each shape [n_samples, n_branches]
        """
        # The "chained" path is identical to the standard LDA aggregation, except
        # that the upstream branch input can be replaced with the previous layer's
        # LDA-computed output (Vout_lin). To avoid duplicated logic (and to ensure
        # we still respect TopK masks / BlockLinear block structure), we delegate
        # to `compute_lda_aggregated_signals` after injecting the overridden
        # `raw_branch_input` when provided.

        if prev_layer_vout_lin is None:
            return self.compute_lda_aggregated_signals(layer_data, C)

        try:
            # Convert override to a torch Tensor so downstream code can `.detach()`.
            if isinstance(prev_layer_vout_lin, torch.Tensor):
                raw_branch_override = prev_layer_vout_lin
            else:
                raw_branch_override = torch.as_tensor(
                    prev_layer_vout_lin, dtype=torch.float32
                )

            # Move override to the same device as existing layer tensors if possible.
            device = None
            for key in (
                "raw_excitatory_input",
                "raw_inhibitory_input",
                "excitation",
                "inhibition",
                "output",
            ):
                t = layer_data.get(key)
                if isinstance(t, torch.Tensor):
                    device = t.device
                    break
            if device is not None:
                raw_branch_override = raw_branch_override.to(device)

            tmp = dict(layer_data)
            tmp["raw_branch_input"] = raw_branch_override
            return self.compute_lda_aggregated_signals(tmp, C)
        except Exception as e:
            self.logger.warning(
                "Chained LDA branch-input override failed; falling back to raw branch input. Error: %s",
                e,
            )
            return self.compute_lda_aggregated_signals(layer_data, C)


__all__ = ["InformationSignalAggregationMixin"]

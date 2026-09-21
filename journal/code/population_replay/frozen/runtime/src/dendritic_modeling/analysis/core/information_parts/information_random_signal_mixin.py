"""Random-weight signal baselines for information analysis."""

from __future__ import annotations

import numpy as np

from dendritic_modeling.analysis.core.information_parts.information_signal_projection import (
    compute_projected_branch_output,
)
from dendritic_modeling.utils.lda import apply_random_weights


class InformationRandomSignalMixin:
    """Compute random-weight aggregate signal baselines."""

    def compute_shuffled_aggregated_signals(
        self,
        layer_data: dict,
        C: np.ndarray,
        shuffle_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute random-weighted versions of E, I, Vb for baseline comparison.

        Uses random non-negative weights with same normalization as LDA weights.
        This provides a random baseline to compare against the LDA-derived signals,
        showing how much improvement LDA provides over chance.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,] (unused but kept for API consistency)
            shuffle_idx: Random seed offset for reproducibility across shuffles

        Returns:
            Tuple of (E_sh, I_sh, Vb_sh, Vout_sh) each shape [n_samples, n_branches]
        """
        module = layer_data.get("module")
        if module is None:
            return None, None, None, None

        raw_exc = layer_data.get("raw_excitatory_input")
        raw_inh = layer_data.get("raw_inhibitory_input")
        raw_branch = layer_data.get("raw_branch_input")

        n_samples = C.shape[0]
        n_branches = layer_data["excitation"].shape[1]

        E_sh = np.zeros((n_samples, n_branches))
        I_sh = np.zeros((n_samples, n_branches))
        Vb_sh = np.zeros((n_samples, n_branches))

        # Create RNG with reproducible seed
        rng = np.random.default_rng(int(self.random_weights_seed) + int(shuffle_idx))

        # For E: Apply random weights on TopK-masked excitatory inputs
        if raw_exc is not None and module.branch_excitation is not None:
            raw_exc_np = raw_exc.detach().cpu().numpy()
            exc_weight_norm_order = getattr(
                module.branch_excitation, "weight_norm_order", None
            )
            exc_gamma = getattr(module.branch_excitation, "gamma", 1.0)
            exc_mask = module.branch_excitation.weight_mask().detach().cpu().numpy()

            for branch_idx in range(n_branches):
                branch_mask = exc_mask[branch_idx, :]
                active_indices = np.where(branch_mask > 0)[0]
                if len(active_indices) > 0:
                    branch_inputs = raw_exc_np[:, active_indices]
                    E_sh[:, branch_idx] = apply_random_weights(
                        branch_inputs,
                        weight_norm_order=exc_weight_norm_order,
                        gamma=exc_gamma,
                        rng=rng,
                    )

        # For I: Apply random weights on TopK-masked inhibitory inputs
        if raw_inh is not None and module.branch_inhibition is not None:
            raw_inh_np = raw_inh.detach().cpu().numpy()
            inh_weight_norm_order = getattr(
                module.branch_inhibition, "weight_norm_order", None
            )
            inh_gamma = getattr(module.branch_inhibition, "gamma", 1.0)
            inh_mask = module.branch_inhibition.weight_mask().detach().cpu().numpy()

            for branch_idx in range(n_branches):
                branch_mask = inh_mask[branch_idx, :]
                active_indices = np.where(branch_mask > 0)[0]
                if len(active_indices) > 0:
                    branch_inputs = raw_inh_np[:, active_indices]
                    I_sh[:, branch_idx] = apply_random_weights(
                        branch_inputs,
                        weight_norm_order=inh_weight_norm_order,
                        gamma=inh_gamma,
                        rng=rng,
                    )

        # For Vb: Apply random weights on branch inputs using BlockLinear structure
        if raw_branch is not None and module.input_branches:
            raw_branch_np = raw_branch.detach().cpu().numpy()
            vb_weight_norm_order = getattr(
                module.branches_to_output, "weight_norm_order", None
            )
            vb_gamma = getattr(module.branches_to_output, "gamma", 1.0)

            if hasattr(module.branches_to_output, "block_size"):
                block_size = module.branches_to_output.block_size
                for branch_idx in range(n_branches):
                    start_idx = branch_idx * block_size
                    end_idx = min((branch_idx + 1) * block_size, raw_branch_np.shape[1])
                    if start_idx < raw_branch_np.shape[1]:
                        branch_inputs = raw_branch_np[:, start_idx:end_idx]
                        if branch_inputs.shape[1] > 0:
                            Vb_sh[:, branch_idx] = apply_random_weights(
                                branch_inputs,
                                weight_norm_order=vb_weight_norm_order,
                                gamma=vb_gamma,
                                rng=rng,
                            )

        Vout_sh = compute_projected_branch_output(module, E_sh, I_sh, Vb_sh)

        return E_sh, I_sh, Vb_sh, Vout_sh


__all__ = ["InformationRandomSignalMixin"]

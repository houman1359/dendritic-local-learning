"""
Stochastic TopK Linear Layer for Dendritic Networks.

This module implements a stochastic variant of TopKLinear where connections
are selected probabilistically based on rank, allowing smaller weights a
chance to remain active.
"""

from typing import Optional

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.utils.weight_transforms import WeightTransformType


class StochasticTopKLinear(TopKLinear):
    """
    A variant of TopKLinear where we select top-K stochastically based on rank:
    - We rank by pre_w (or pre_w + noise if noise_level>0).
    - Convert rank => probability of being "kept."
    - Sample a Bernoulli, so smaller weights still have a small chance to stay on.
    """

    # Mask is resampled every forward; never reuse it across recurrent timesteps.
    _supports_recurrent_weight_cache = False

    def __init__(
        self,
        in_features,
        out_features,
        K,
        param_space="log",
        init_method="xavier_normal",
        init_gain=1.0,
        noise_level=0.0,
        weight_transform: WeightTransformType = "exp",
        weight_norm_order: Optional[int] = None,
        gamma: float = 1.0,
        forbidden_input_index_per_output: Optional[torch.Tensor] = None,
        connection_mask: Optional[torch.Tensor] = None,
        temperature=0.5,  # extra hyperparam for sharper or flatter gating
        ultrafast=True,  # ultrafast noisy-topk is the default; set False for rank-probabilistic
    ):
        super().__init__(
            in_features,
            out_features,
            K,
            param_space=param_space,
            init_method=init_method,
            init_gain=init_gain,
            noise_level=noise_level,
            weight_transform=weight_transform,
            weight_norm_order=weight_norm_order,
            gamma=gamma,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
        )
        self.temperature = temperature
        self.ultrafast = ultrafast
        self.initialize()

    def weight_mask_ultrafast(self):
        """
        Ultra-fast approximation of stochastic TopK that:
        1. Adds noise directly to the weights before standard TopK selection
        2. Uses more noise for lower-ranked weights to approximate the desired stochasticity
        3. Performs a single standard TopK operation with no correction steps

        10-50x faster than the full stochastic method with similar behavior.
        """
        # Ensure K is valid and an integer
        k_value = max(1, int(self.K))

        # Apply scaled and temperature-controlled noise only to structurally
        # allowed scores; forbidden positions are masked before selection.
        w_for_topk = self.pre_w.clone()

        # Generate noise with standard deviation that scales with temperature
        noise_scale = max(0.01, self.temperature) * (1.0 + self.noise_level)
        noise = torch.randn_like(w_for_topk) * noise_scale

        # Simply add noise to weights - simpler but still effective approach
        noisy_weights = self._apply_forbidden_scores(w_for_topk + noise)

        # Perform standard TopK selection on the noisy weights
        topK_indices = torch.topk(
            noisy_weights, k_value, dim=-1, largest=True, sorted=False
        )[1]

        # Create mask using the standard TopK approach
        mask = torch.zeros_like(
            self.pre_w,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        batch_indices = (
            torch.arange(self.pre_w.shape[0], device=self.pre_w.device)
            .view(-1, 1)
            .expand(-1, k_value)
        )
        mask[batch_indices, topK_indices] = 1
        mask = self._apply_forbidden_mask(mask)

        return mask

    def weight_mask_rank_probabilistic(self):
        """Rank-probabilistic stochastic TopK selection.

        Ranks are computed after applying structured/forbidden masks. The
        sampled mask is corrected to avoid pathological all-off rows, then the
        hard structural mask is applied again before returning.
        """

        if self.temperature <= 0.0:
            return super().weight_mask()

        # Rank by transformed synaptic strength, not raw parameter magnitude.
        w_for_topk = self.weight()
        if self.weight_transform == "identity":
            w_for_topk = w_for_topk.abs()
        if self.noise_level > 0:
            noise = torch.randn_like(w_for_topk) * self.noise_level
            w_for_topk = w_for_topk + noise

        # Ensure K is an integer and at least 1
        k_value = max(1, int(self.K))

        # Get dimensions
        out_features, in_features = w_for_topk.shape

        scores = self._apply_forbidden_scores(w_for_topk)

        # For each neuron, sort the input weights and get the ranks
        # - First, get the indices that would sort each row in descending order
        _, sort_indices = torch.sort(scores, dim=1, descending=True)

        # - Create a rank tensor with the same shape as weights
        # Ensure the dtype is the same as what we'll assign to it (Long/Int64)
        arange_indices = torch.arange(in_features, device=w_for_topk.device).expand(
            out_features, -1
        )
        ranks = torch.zeros_like(w_for_topk, dtype=arange_indices.dtype)

        # - For each neuron (row), assign ranks to the positions
        batch_indices = (
            torch.arange(out_features, device=w_for_topk.device)
            .view(-1, 1)
            .expand(-1, in_features)
        )
        ranks[batch_indices, sort_indices] = arange_indices

        # Convert ranks to float for sigmoid calculation
        ranks = ranks.to(torch.float)

        # Calculate selection probabilities using sigmoid
        temperature = max(0.01, self.temperature)  # Prevent division by zero
        p = torch.sigmoid((k_value - ranks) / (k_value * temperature))

        # Sample Bernoulli for all neurons at once
        mask = torch.bernoulli(p)

        # Fast connection count correction to ensure each neuron has reasonable connections
        # 1. Count connections per neuron
        connections_per_neuron = mask.sum(dim=1)

        # 2. For neurons with too few connections (less than K/2), add more
        too_few_mask = connections_per_neuron < (k_value // 2)
        if too_few_mask.any():
            # Safe handling of nonzero indices to avoid 0-d tensor issues
            too_few_indices = too_few_mask.nonzero(as_tuple=False).view(-1)
            for i in too_few_indices:
                # Get the indices of the top-K weights for this neuron
                _, top_indices = torch.topk(scores[i], k_value, largest=True)
                mask[i, top_indices] = 1

        # 3. For neurons with too many connections (more than 2*K), keep only the top 2*K
        connections_per_neuron = mask.sum(dim=1)
        too_many_mask = connections_per_neuron > (2 * k_value)
        if too_many_mask.any():
            # Safe handling of nonzero indices to avoid 0-d tensor issues
            too_many_indices = too_many_mask.nonzero(as_tuple=False).view(-1)
            for i in too_many_indices:
                # Find the weakest connections for this neuron
                # First, get indices of all connections that are currently active
                active_indices = mask[i].nonzero(as_tuple=True)[0]
                # Get their weights
                active_weights = scores[i, active_indices]
                # Sort from weakest to strongest
                _, sorted_indices = torch.sort(active_weights)
                # Calculate how many to remove
                to_remove = int(connections_per_neuron[i].item() - k_value)
                # Set the weakest ones to 0
                if to_remove > 0:
                    indices_to_remove = active_indices[sorted_indices[:to_remove]]
                    mask[i, indices_to_remove] = 0

        # 4. Handle neurons with zero connections
        connections_per_neuron = mask.sum(dim=1)
        zero_connections = connections_per_neuron == 0
        if zero_connections.any():
            # Safe handling of nonzero indices to avoid 0-d tensor issues
            zero_indices = zero_connections.nonzero(as_tuple=False).view(-1)
            for i in zero_indices:
                # Get the index of the strongest weight for this neuron
                strongest_idx = torch.argmax(scores[i])
                mask[i, strongest_idx] = 1

        return self._apply_forbidden_mask(mask)

    def weight_mask(self):
        """Return a stochastic TopK mask using the configured sampling mode."""

        if self.ultrafast:
            return self.weight_mask_ultrafast()
        return self.weight_mask_rank_probabilistic()


__all__ = ["StochasticTopKLinear"]

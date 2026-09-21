"""
Point-neuron parameter-matched MLP baseline.

This module implements PointMatchedParamMLP which creates a shallow (one hidden layer)
MLP with point neurons whose parameter count matches an equivalent EINet configuration.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork,
)


class PointMatchedParamMLP(nn.Module):
    """
    Create a point-neuron MLP with (approximately) matched parameter count to an EINet.

    We match the *effective* (post-TopK) parameter count by choosing the smallest hidden width H
    for a 2-layer MLP:

        Linear(input_dim -> H) -> act -> Linear(H -> output_dim)

    such that n_params >= effective_params(EINet(config)).
    """

    def __init__(
        self,
        input_dim: int,
        excitatory_layer_sizes: list[int],
        inhibitory_layer_sizes,
        excitatory_branch_factors,
        inhibitory_branch_factors,
        ee_synapses_per_branch_per_layer,
        ei_synapses_per_branch_per_layer,
        ie_synapses_per_branch_per_layer,
        ii_synapses_per_branch_per_layer=None,
        reactivate=False,
        somatic_synapses=True,
        activation: str = "relu",
        **kwargs,  # Accept but ignore other EINet parameters
    ):
        super().__init__()

        if ii_synapses_per_branch_per_layer is None:
            ii_synapses_per_branch_per_layer = []

        # Create default transfer_params if not provided in kwargs (to build EINet)
        if "transfer_params" not in kwargs:
            kwargs["transfer_params"] = {
                "input_mode": 0,
                "independent_pathways": False,
                "excitatory_dim": [],
                "inhibitory_dim": [],
                "output_activation": "none",
            }

        # Build the reference EINet to match parameter count.
        einet = ExcitationInhibitionNetwork(
            input_dim=input_dim,
            excitatory_layer_sizes=excitatory_layer_sizes,
            inhibitory_layer_sizes=inhibitory_layer_sizes,
            excitatory_branch_factors=excitatory_branch_factors,
            inhibitory_branch_factors=inhibitory_branch_factors,
            ee_synapses_per_branch_per_layer=ee_synapses_per_branch_per_layer,
            ei_synapses_per_branch_per_layer=ei_synapses_per_branch_per_layer,
            ie_synapses_per_branch_per_layer=ie_synapses_per_branch_per_layer,
            ii_synapses_per_branch_per_layer=ii_synapses_per_branch_per_layer,
            reactivate=reactivate,
            somatic_synapses=somatic_synapses,
            **kwargs,
        )

        # Match *effective* params (post-TopK), not the dense trainable parameter tensors.
        # This aligns the baseline with the number of active synapses/blocks the dendritic
        # network actually uses during computation.
        target_params = int(einet.get_effective_params())

        if not excitatory_layer_sizes:
            raise ValueError("excitatory_layer_sizes cannot be empty")
        output_dim = int(excitatory_layer_sizes[-1])

        # Param count for 2-layer MLP:
        # (input_dim*H + H) + (H*output_dim + output_dim)
        denom = int(input_dim) + int(output_dim) + 1
        hidden_width = max(1, math.ceil(max(0, target_params - output_dim) / denom))

        activation_factory = ActivationFactory()

        fc1 = nn.Linear(input_dim, hidden_width)
        fc2 = nn.Linear(hidden_width, output_dim)

        # Initialize weights similarly to other MLP baselines
        if activation == "relu":
            nn.init.kaiming_normal_(fc1.weight)
            nn.init.zeros_(fc1.bias)
            nn.init.kaiming_normal_(fc2.weight)
            nn.init.zeros_(fc2.bias)
        else:
            nn.init.xavier_normal_(fc1.weight)
            nn.init.zeros_(fc1.bias)
            nn.init.xavier_normal_(fc2.weight)
            nn.init.zeros_(fc2.bias)

        self.hidden = nn.Sequential(
            fc1,
            activation_factory.create(act_type=activation, output_dim=hidden_width),
            fc2,
        )
        transfer_params = kwargs.get("transfer_params", {}) or {}
        output_activation = transfer_params.get("output_activation") or "none"
        self.output_activation = activation_factory.create(
            act_type=output_activation,
            output_dim=output_dim,
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_activation(self.hidden(x))


__all__ = ["PointMatchedParamMLP"]

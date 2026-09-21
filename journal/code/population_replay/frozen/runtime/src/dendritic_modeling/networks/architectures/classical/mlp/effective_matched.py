"""
Effective Matched Parameter MLP.

This module implements EffectiveMatchedParamMLP which creates an MLP with
at least as many parameters as the *effective* (post-topk) params of an equivalent EINet.
"""

import torch
from torch import nn

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork,
)


class EffectiveMatchedParamMLP(nn.Module):
    """
    Creates an MLP with at least as many parameters as the effective (post-topk) params
    of an equivalent EINet configuration. Matches the number of layers to DendriticBranchLayers.
    """

    def __init__(
        self,
        input_dim,
        excitatory_layer_sizes,
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
        if ii_synapses_per_branch_per_layer is None:
            ii_synapses_per_branch_per_layer = []
        super().__init__()

        # Create default transfer_params if not provided in kwargs
        if "transfer_params" not in kwargs:
            kwargs["transfer_params"] = {
                "input_mode": 0,
                "independent_pathways": False,
                "excitatory_dim": [],
                "inhibitory_dim": [],
                "output_activation": "none",
            }

        # Create EINet to get effective parameter count and number of layers
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

        target_params = einet.get_effective_params()  # Use effective (sparse) count

        # Get number of branch layers
        n_branch_layers = einet.n_branch_layers
        if not somatic_synapses:
            n_branch_layers -= 2 * len(einet.layers)  # Account for somatic layer

        # Calculate required hidden layer widths
        def get_param_count(hidden_width):
            """Calculate total parameters for given hidden width."""
            params = input_dim * hidden_width  # First layer
            params += (
                hidden_width * hidden_width * (n_branch_layers - 1)
            )  # Hidden layers
            params += hidden_width * excitatory_layer_sizes[-1]  # Output layer
            return params

        # Binary search for minimum hidden width that exceeds target params
        left, right = 1, 10000
        min_width = right
        while left <= right:
            mid = (left + right) // 2
            params = get_param_count(mid)
            if params >= target_params:
                min_width = mid
                right = mid - 1
            else:
                left = mid + 1

        hidden_width = min_width

        # Build MLP layers
        layers = []
        activation_factory = ActivationFactory()

        # Input layer
        fc_linear = nn.Linear(input_dim, hidden_width)
        if activation == "relu":
            nn.init.kaiming_normal_(fc_linear.weight)
            nn.init.zeros_(fc_linear.bias)
        else:
            nn.init.xavier_normal_(fc_linear.weight)
            nn.init.zeros_(fc_linear.bias)
        layers.extend(
            [
                fc_linear,
                activation_factory.create(act_type=activation, output_dim=hidden_width),
            ]
        )

        # Hidden layers
        for _ in range(n_branch_layers - 1):
            fc_linear = nn.Linear(hidden_width, hidden_width)
            if activation == "relu":
                nn.init.kaiming_normal_(fc_linear.weight)
                nn.init.zeros_(fc_linear.bias)
            else:
                nn.init.xavier_normal_(fc_linear.weight)
                nn.init.zeros_(fc_linear.bias)
            layers.extend(
                [
                    fc_linear,
                    activation_factory.create(
                        act_type=activation, output_dim=hidden_width
                    ),
                ]
            )

        # Output layer
        layers.append(nn.Linear(hidden_width, excitatory_layer_sizes[-1]))
        self.hidden = nn.Sequential(*layers)
        # Match the interface expected by training code (decoder input sizing)
        self.output_dim = int(excitatory_layer_sizes[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


__all__ = ["EffectiveMatchedParamMLP"]

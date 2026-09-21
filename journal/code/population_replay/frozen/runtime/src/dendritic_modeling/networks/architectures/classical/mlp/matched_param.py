"""
Matched Parameter MLP.

This module implements MatchedParamMLP which creates an MLP with
at least as many parameters as an equivalent EINet configuration.
"""

from typing import Optional, Union

import torch
from omegaconf import DictConfig
from torch import nn

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    BlockLinear,
    ConfigurableEINetwork,
    DendriNet,
    DendriticBranchLayer,
    TopKLinear,
)
from dendritic_modeling.networks.base import BaseNetwork


class MatchedTotalParamMLP(BaseNetwork):
    """
    Creates an MLP with at least as many parameters as an equivalent EINet configuration.
    Matches the number of layers to DendriticBranchLayers and minimizes parameters while
    exceeding EINet parameter count.
    """

    def __init__(self, config: Union[dict, DictConfig], input_dim):
        super().__init__()
        # Create EINet to get parameter count and number of layers
        einet = ConfigurableEINetwork(config, input_dim)

        self.output_dim = einet.output_dim

        # summary_ = torchinfo.summary(einet, verbose=0)  # Summary for debugging
        target_params = sum(p.numel() for p in einet.parameters() if p.requires_grad)

        # Get number of branch layers
        n_branch_layers = einet.n_branch_layers

        # Calculate required hidden layer widths
        # For n hidden layers (n+1 total layers), we need:
        # input_dim * h1 + h1 * h2 + h2 * h3 + ... + hn * output_dim ≥ target_params
        # Minimize total parameters while satisfying this constraint
        # We'll use equal width hidden layers for simplicity
        # Then: input_dim * h + h * h * (n-1) + h * output_dim ≥ target_params
        # where n is number of hidden layers

        def get_param_count(hidden_width):
            """Calculate total parameters for given hidden width"""
            params = input_dim * hidden_width  # First layer
            params += (
                hidden_width * hidden_width * (n_branch_layers - 2)
            )  # Hidden layers
            params += hidden_width * einet.output_dim  # Output layer
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
        # activation_factory = ActivationFactory()

        # Input layer
        fc_linear = nn.Linear(input_dim, hidden_width)
        # if activation == "relu":
        #     nn.init.kaiming_normal_(fc_linear.weight)
        #     nn.init.zeros_(fc_linear.bias)
        # else:
        #     nn.init.xavier_normal_(fc_linear.weight)
        #     nn.init.zeros_(fc_linear.bias)
        # layers.extend(
        #     [
        #         fc_linear,
        #         activation_factory.create(act_type=activation, output_dim=hidden_width),
        #     ]
        # )
        nn.init.xavier_normal_(fc_linear.weight)
        nn.init.zeros_(fc_linear.bias)
        layers.extend([fc_linear, nn.Sigmoid()])

        # Hidden layers
        for _ in range(n_branch_layers - 2):
            fc_linear = nn.Linear(hidden_width, hidden_width)
            # if activation == "relu":
            #     nn.init.kaiming_normal_(fc_linear.weight)
            #     nn.init.zeros_(fc_linear.bias)
            # else:
            #     nn.init.xavier_normal_(fc_linear.weight)
            #     nn.init.zeros_(fc_linear.bias)
            # layers.extend(
            #     [
            #         fc_linear,
            #         activation_factory.create(
            #             act_type=activation, output_dim=hidden_width
            #         ),
            #     ]
            # )
            nn.init.xavier_normal_(fc_linear.weight)
            nn.init.zeros_(fc_linear.bias)
            layers.extend([fc_linear, nn.Sigmoid()])

        # Output layer
        layers.extend([nn.Linear(hidden_width, einet.output_dim), nn.Sigmoid()])
        self.hidden = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


class MatchedActiveParamMLP(BaseNetwork):
    """
    Creates an MLP with at least as many parameters as an equivalent EINet configuration.
    Matches the number of layers to DendriticBranchLayers and minimizes parameters while
    exceeding EINet parameter count.
    """

    def __init__(self, config: Union[dict, DictConfig], input_dim):
        super().__init__()
        # Create EINet to get parameter count and number of layers
        einet = ConfigurableEINetwork(config, input_dim)

        self.output_dim = einet.output_dim

        target_params = 0
        with torch.no_grad():
            for layer in einet.layers:
                exc_dendrinet = layer.excitatory_cells
                for branch_layer in exc_dendrinet.branch_layers:
                    branch_layer: DendriticBranchLayer
                    if branch_layer.branch_excitation is not None:
                        target_params += (
                            branch_layer.branch_excitation.weight_mask().sum().item()
                        )
                    if branch_layer.branch_inhibition is not None:
                        target_params += (
                            branch_layer.branch_inhibition.weight_mask().sum().item()
                        )
                    if hasattr(branch_layer, "branches_to_output"):
                        target_params += (
                            branch_layer.branches_to_output.log_weight.numel()
                        )
                    target_params += sum(
                        p.numel()
                        for p in branch_layer.reactivation.parameters()
                        if p.requires_grad
                    )

                if layer.inhibitory_cells is not None:
                    if isinstance(layer.inhibitory_cells, DendriNet):
                        inh_dendrinet = layer.inhibitory_cells
                        for branch_layer in inh_dendrinet.branch_layers:
                            branch_layer: DendriticBranchLayer
                            if branch_layer.branch_excitation is not None:
                                target_params += (
                                    branch_layer.branch_excitation.weight_mask()
                                    .sum()
                                    .item()
                                )
                            if branch_layer.branch_inhibition is not None:
                                target_params += (
                                    branch_layer.branch_inhibition.weight_mask()
                                    .sum()
                                    .item()
                                )
                            if hasattr(branch_layer, "branches_to_output"):
                                target_params += (
                                    branch_layer.branches_to_output.log_weight.numel()
                                )
                            target_params += sum(
                                p.numel()
                                for p in branch_layer.reactivation.parameters()
                                if p.requires_grad
                            )
                    else:
                        target_params += sum(
                            p.numel()
                            for p in layer.inhibitory_cells.parameters()
                            if p.requires_grad
                        )

        # Get number of branch layers
        n_branch_layers = einet.n_branch_layers

        # Calculate required hidden layer widths
        # For n hidden layers (n+1 total layers), we need:
        # input_dim * h1 + h1 * h2 + h2 * h3 + ... + hn * output_dim ≥ target_params
        # Minimize total parameters while satisfying this constraint
        # We'll use equal width hidden layers for simplicity
        # Then: input_dim * h + h * h * (n-1) + h * output_dim ≥ target_params
        # where n is number of hidden layers

        def get_param_count(hidden_width):
            """Calculate total parameters for given hidden width"""
            params = input_dim * hidden_width  # First layer
            params += (
                hidden_width * hidden_width * (n_branch_layers - 2)
            )  # Hidden layers
            params += hidden_width * einet.output_dim  # Output layer
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
        # activation_factory = ActivationFactory()

        # Input layer
        fc_linear = nn.Linear(input_dim, hidden_width)
        # if activation == "relu":
        #     nn.init.kaiming_normal_(fc_linear.weight)
        #     nn.init.zeros_(fc_linear.bias)
        # else:
        #     nn.init.xavier_normal_(fc_linear.weight)
        #     nn.init.zeros_(fc_linear.bias)
        # layers.extend(
        #     [
        #         fc_linear,
        #         activation_factory.create(act_type=activation, output_dim=hidden_width),
        #     ]
        # )
        nn.init.xavier_normal_(fc_linear.weight)
        nn.init.zeros_(fc_linear.bias)
        layers.extend([fc_linear, nn.Sigmoid()])

        # Hidden layers
        for _ in range(n_branch_layers - 2):
            fc_linear = nn.Linear(hidden_width, hidden_width)
            # if activation == "relu":
            #     nn.init.kaiming_normal_(fc_linear.weight)
            #     nn.init.zeros_(fc_linear.bias)
            # else:
            #     nn.init.xavier_normal_(fc_linear.weight)
            #     nn.init.zeros_(fc_linear.bias)
            # layers.extend(
            #     [
            #         fc_linear,
            #         activation_factory.create(
            #             act_type=activation, output_dim=hidden_width
            #         ),
            #     ]
            # )
            nn.init.xavier_normal_(fc_linear.weight)
            nn.init.zeros_(fc_linear.bias)
            layers.extend([fc_linear, nn.Sigmoid()])

        # Output layer
        layers.extend([nn.Linear(hidden_width, einet.output_dim), nn.Sigmoid()])
        self.hidden = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


class SparseStructuredLayer(nn.Module):
    def __init__(
        self,
        output_dim: int,
        synaptic_input_dim: Optional[int] = None,
        synaptic_K: Optional[int] = None,
        input_branch_factor: Optional[int] = None,
        reactivate: bool = False,
    ):
        super().__init__()
        self.synaptic_inputs = synaptic_input_dim is not None
        if self.synaptic_inputs:
            self.synapse_layer = TopKLinear(
                in_features=synaptic_input_dim,
                out_features=output_dim,
                K=synaptic_K,
                weight_transform="identity",
            )
            nn.init.xavier_normal_(self.synapse_layer.pre_w)
        else:
            self.synapse_layer = None

        self.input_branches = input_branch_factor is not None
        if self.input_branches:
            self.branches_to_output = BlockLinear(
                output_dim * input_branch_factor,
                output_dim,
                weight_transform="identity",
            )
            nn.init.xavier_normal_(self.branches_to_output.log_weight)

        if reactivate:
            self.reactivation = ActivationFactory.create(
                act_type="param_tanh", output_dim=output_dim, init_m=1.0, init_b=0.0
            )
        else:
            self.reactivation = nn.Identity()
        self.reactivate = reactivate

    def decay_weights(self, weight_decay=0.1, weight_boosting=False):
        if self.synapse_layer is not None:
            self.synapse_layer.decay_weights(weight_decay, weight_boosting)

    def forward(
        self, x: torch.Tensor, branch_input: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        activation = 0

        if self.synaptic_inputs:
            activation = activation + self.synapse_layer(x)

        if self.input_branches and branch_input is not None:
            activation = activation + self.branches_to_output(branch_input)

        output = self.reactivation(activation)
        return output


class SparseStructuredMLP(BaseNetwork):
    def __init__(
        self, config: Union[dict, DictConfig], input_dim: int, flatten_dendrites=False
    ):
        super().__init__()

        einet = ConfigurableEINetwork(
            config, input_dim, flatten_dendrites=flatten_dendrites
        )
        self.output_dim = einet.output_dim

        inh_ss_layers = []
        exc_ss_layers = []

        n_params = 0

        with torch.no_grad():
            final_layer = einet.layers[-1]
            n_inh_cells = 0

            if final_layer.inhibitory_cells is not None:
                inh_dendrinet = final_layer.inhibitory_cells
                n_inh_cells = inh_dendrinet.n_soma

                for i, branch_layer in enumerate(inh_dendrinet.branch_layers):
                    branch_layer: DendriticBranchLayer

                    if i != len(inh_dendrinet.branch_layers) - 1:
                        K = branch_layer.branch_excitation.K
                        synaptic_input_dim = input_dim

                        n_params += K * branch_layer.n_branches  # synapse params
                    else:
                        K = None
                        synaptic_input_dim = None

                    if i != 0:
                        n_params += (
                            branch_layer.n_branches * branch_layer.input_branch_factor
                        )  # blocklinear params

                    n_params += branch_layer.n_branches * 2  # reactivation params

                    inh_ss_layers.append(
                        SparseStructuredLayer(
                            output_dim=branch_layer.n_branches,
                            synaptic_input_dim=synaptic_input_dim,
                            synaptic_K=K,
                            input_branch_factor=branch_layer.input_branch_factor,
                            reactivate=True,
                        )
                    )

            exc_dendrinet = final_layer.excitatory_cells

            for i, branch_layer in enumerate(exc_dendrinet.branch_layers):
                branch_layer: DendriticBranchLayer

                if i != len(exc_dendrinet.branch_layers) - 1:
                    K = branch_layer.branch_excitation.K
                    K += branch_layer.branch_inhibition.K
                    synaptic_input_dim = input_dim + n_inh_cells

                    n_params += K * branch_layer.n_branches  # synapse params
                else:
                    K = None
                    synaptic_input_dim = None

                if i != 0:
                    n_params += (
                        branch_layer.n_branches * branch_layer.input_branch_factor
                    )  # blocklinear params

                n_params += branch_layer.n_branches * 2  # reactivation params

                exc_ss_layers.append(
                    SparseStructuredLayer(
                        output_dim=branch_layer.n_branches,
                        synaptic_input_dim=synaptic_input_dim,
                        synaptic_K=K,
                        input_branch_factor=branch_layer.input_branch_factor,
                        reactivate=True,
                    )
                )

        if inh_ss_layers:
            self.inh_layers = nn.ModuleList(inh_ss_layers)
        else:
            self.inh_layers = None

        self.exc_layers = nn.ModuleList(exc_ss_layers)

        target_params = 0
        with torch.no_grad():
            for layer in einet.layers:
                exc_dendrinet = layer.excitatory_cells
                for branch_layer in exc_dendrinet.branch_layers:
                    branch_layer: DendriticBranchLayer
                    if branch_layer.branch_excitation is not None:
                        target_params += (
                            branch_layer.branch_excitation.weight_mask().sum().item()
                        )
                    if branch_layer.branch_inhibition is not None:
                        target_params += (
                            branch_layer.branch_inhibition.weight_mask().sum().item()
                        )
                    if hasattr(branch_layer, "branches_to_output"):
                        target_params += (
                            branch_layer.branches_to_output.log_weight.numel()
                        )
                    target_params += sum(
                        p.numel()
                        for p in branch_layer.reactivation.parameters()
                        if p.requires_grad
                    )

                if layer.inhibitory_cells is not None:
                    if isinstance(layer.inhibitory_cells, DendriNet):
                        inh_dendrinet = layer.inhibitory_cells
                        for branch_layer in inh_dendrinet.branch_layers:
                            branch_layer: DendriticBranchLayer
                            if branch_layer.branch_excitation is not None:
                                target_params += (
                                    branch_layer.branch_excitation.weight_mask()
                                    .sum()
                                    .item()
                                )
                            if branch_layer.branch_inhibition is not None:
                                target_params += (
                                    branch_layer.branch_inhibition.weight_mask()
                                    .sum()
                                    .item()
                                )
                            if hasattr(branch_layer, "branches_to_output"):
                                target_params += (
                                    branch_layer.branches_to_output.log_weight.numel()
                                )
                            target_params += sum(
                                p.numel()
                                for p in branch_layer.reactivation.parameters()
                                if p.requires_grad
                            )
                    else:
                        target_params += sum(
                            p.numel()
                            for p in layer.inhibitory_cells.parameters()
                            if p.requires_grad
                        )

    def cell_forward(self, mod_list: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        output = None
        for module in mod_list:
            output = module(x, output)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inh_layers is not None:
            inh_input = self.cell_forward(self.inh_layers, x)
            x = torch.cat([x, inh_input], dim=-1)

        output = self.cell_forward(self.exc_layers, x)
        return output

    def decay_weights(self, weight_decay=0.1, weight_boosting=False):
        if self.inh_layers is not None:
            for branch_layer in self.inh_layers:
                branch_layer.decay_weights(weight_decay, weight_boosting)

        for branch_layer in self.exc_layers:
            branch_layer.decay_weights(weight_decay, weight_boosting)


__all__ = ["MatchedActiveParamMLP", "MatchedTotalParamMLP", "SparseStructuredMLP"]

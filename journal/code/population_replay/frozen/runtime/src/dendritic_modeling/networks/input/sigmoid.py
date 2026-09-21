"""
Sigmoid-based Input Transform for neural networks.

This module implements SigmoidInputNetTransform which applies
a sigmoid activation to ensure values are in [0, 1] range.
"""

import torch

from dendritic_modeling.networks.base import BaseNetwork


class SigmoidOutputNetwork(BaseNetwork):
    """
    Extension of InputNetTransform that applies a sigmoid activation to the output,
    ensuring values are in the range [0, 1] before entering the dendrinet.
    """

    def __init__(self, net: BaseNetwork):
        super().__init__()
        self.net = net
        self.output_dim = net.output_dim

    def forward(self, x):
        x = self.net(x)
        return torch.sigmoid(x)


__all__ = ["SigmoidOutputNetwork"]

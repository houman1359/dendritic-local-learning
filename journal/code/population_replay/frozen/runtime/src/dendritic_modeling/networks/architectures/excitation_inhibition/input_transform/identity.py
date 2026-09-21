"""
Identity Input Transform for neural networks.

This module implements IdentityInputTransform which returns the input as is.
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class IdentityInputTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x


__all__ = ["IdentityInputTransform"]

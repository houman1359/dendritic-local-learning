"""
Index-based Input Transform for neural networks.

This module implements IndexInputTransform which randomly splits
input dimensions into excitatory and inhibitory channels.
"""

from typing import Optional

import torch
import torch.nn as nn


class IndexInputTransform(nn.Module):
    def __init__(
        self,
        input_dim: int,
        excitatory_dim: Optional[int] = None,
        inhibitory_dim: Optional[int] = None,
        split_seed: Optional[int] = None,
        split_strategy: str = "random",
    ):
        super().__init__()

        if input_dim is None or int(input_dim) <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        input_dim = int(input_dim)
        self.input_dim = input_dim

        if excitatory_dim is None and inhibitory_dim is None:
            excitatory_dim = int(input_dim / 2)
            inhibitory_dim = int(input_dim - excitatory_dim)
        if excitatory_dim is None:
            excitatory_dim = int(input_dim - inhibitory_dim)
        if inhibitory_dim is None:
            inhibitory_dim = int(input_dim - excitatory_dim)

        excitatory_dim = int(excitatory_dim)
        inhibitory_dim = int(inhibitory_dim)
        if excitatory_dim <= 0 or inhibitory_dim <= 0:
            raise ValueError(
                "excitatory_dim and inhibitory_dim must be positive, "
                f"got {excitatory_dim} and {inhibitory_dim}"
            )
        if excitatory_dim + inhibitory_dim != input_dim:
            raise ValueError(
                "excitatory_dim + inhibitory_dim must equal input_dim, "
                f"got {excitatory_dim} + {inhibitory_dim} != {input_dim}"
            )

        split_strategy = str(split_strategy or "random").lower()
        if split_strategy in {"ordered", "contiguous", "first_half"}:
            dims = torch.arange(input_dim)
        elif split_strategy == "random":
            generator = None
            if split_seed is not None:
                generator = torch.Generator()
                generator.manual_seed(int(split_seed) % ((1 << 63) - 1))
            dims = torch.randperm(input_dim, generator=generator)
        else:
            raise ValueError(
                "split_strategy must be 'random' or 'ordered', "
                f"got {split_strategy!r}"
            )
        exc_dims = dims[:excitatory_dim]
        inh_dims = dims[excitatory_dim:]

        self.exc_dims: torch.Tensor
        self.register_buffer("exc_dims", exc_dims)
        self.inh_dims: torch.Tensor
        self.register_buffer("inh_dims", inh_dims)

        self.excitatory_dim = excitatory_dim
        self.inhibitory_dim = inhibitory_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x[None, :]
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "IndexInputTransform expected last input dimension "
                f"{self.input_dim}, got {x.shape[-1]}"
            )

        excitatory_input = x[..., self.exc_dims]
        inhibitory_input = x[..., self.inh_dims]
        return excitatory_input, inhibitory_input


__all__ = ["IndexInputTransform"]

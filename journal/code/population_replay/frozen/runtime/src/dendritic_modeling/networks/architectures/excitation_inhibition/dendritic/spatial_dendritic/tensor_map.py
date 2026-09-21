"""HierarchicalDendriticTensorMap — flatten-based dendritic core for backbone replacement.

This is the baseline spatial core: it flattens the encoder's spatial output,
runs it through a standard :class:`ConfigurableEINetwork` hierarchy (the same
DendriNet used everywhere else), then projects to the suffix's expected input
dimension via a linear layer.

Pipeline::

    encoder output (B, C, H, W)
        → Flatten → (B, C*H*W)
        → ConfigurableEINetwork → (B, E)   # E = excitatory_layer_sizes[-1]
        → Linear(E, suffix_input_dim)       # only if E ≠ suffix_input_dim
        → suffix input (B, suffix_input_dim)

This wrapper reuses the *entire* existing EINet code without any modifications.
It exists so the factory can distinguish "core that accepts spatial input from a
pretrained encoder" from a plain ``einet`` that expects pre-flattened input.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
)
from dendritic_modeling.networks.base import BaseNetwork

logger = logging.getLogger(__name__)


class HierarchicalDendriticTensorMap(BaseNetwork):
    """Dendritic core that accepts spatial ``(B, C, H, W)`` or flat ``(B, D)`` input.

    Internally flattens to ``(B, D)`` and delegates to a standard
    :class:`ConfigurableEINetwork`. An optional output projection maps the
    EINet's excitatory output to the suffix's expected input dimension.

    Args:
        config: Structured EINet config (architecture, connectivity, etc.).
        input_dim: Flattened input dimension (C*H*W or D).
        suffix_input_dim: Expected input dim of the downstream suffix/decoder.
            If different from the EINet's output dim, a linear projection is
            added. If ``None``, no projection is added.
        synapse_mode: Override for the EINet synapse mode.
        use_shunting: Override for shunting inhibition.
        weight_transform: Override for weight transform.
    """

    def __init__(
        self,
        config: dict | DictConfig,
        input_dim: int,
        suffix_input_dim: int | None = None,
        synapse_mode: str | None = None,
        use_shunting: bool | None = None,
        weight_transform: str | None = None,
    ):
        super().__init__()

        self.flatten = nn.Flatten(start_dim=1)

        self.einet = ConfigurableEINetwork(
            config=config,
            input_dim=input_dim,
            synapse_mode=synapse_mode,
            use_shunting=use_shunting,
            weight_transform=weight_transform,
        )

        einet_output_dim = self.einet.output_dim

        # Add projection if EINet output doesn't match suffix expectation
        if suffix_input_dim is not None and suffix_input_dim != einet_output_dim:
            self.projection = nn.Linear(einet_output_dim, suffix_input_dim)
            self.output_dim = suffix_input_dim
            logger.info(
                "Added output projection: %d → %d",
                einet_output_dim,
                suffix_input_dim,
            )
        else:
            self.projection = None
            self.output_dim = einet_output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten spatial dims if present: (B, C, H, W) → (B, C*H*W)
        if x.dim() > 2:
            x = self.flatten(x)

        x = self.einet(x)

        if self.projection is not None:
            x = self.projection(x)

        return x

    def get_output_dim(self) -> int:
        return self.output_dim

    # ------------------------------------------------------------------
    # Training hook delegation — these MUST be forwarded so that custom
    # weight decay (DEEPST) and synaptic rewiring work through the model's
    # base class dispatch (see models/base.py).
    # ------------------------------------------------------------------
    def decay_weights(self, weight_decay, weight_boosting=False):
        """Forward to the inner EINet."""
        self.einet.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self):
        """Forward to the inner EINet."""
        self.einet.apply_rewiring()

    # ------------------------------------------------------------------
    # Introspection delegation — mirrors the EINet public surface so that
    # analysis tools, parameter-matched MLP baselines, and plotting code
    # treat this wrapper identically to a plain ConfigurableEINetwork.
    # ------------------------------------------------------------------
    @property
    def branch_layers(self):
        """Delegate to the inner EINet for analysis/inspection tools."""
        return self.einet.branch_layers

    @property
    def n_branch_layers(self):
        return self.einet.n_branch_layers

    @property
    def layers(self):
        """E-I layers (ExcitationInhibitionLayer list) from the inner EINet."""
        return self.einet.layers

    @property
    def somatic_synapses(self):
        return getattr(self.einet, "somatic_synapses", True)

    @property
    def use_shunting(self):
        return getattr(self.einet, "use_shunting", True)

    def get_effective_params(self) -> int:
        """Effective (post-sparsity) parameter count from the inner EINet.

        Does **not** include the output projection, which is dense and small.
        """
        proj_params = (
            sum(p.numel() for p in self.projection.parameters())
            if self.projection is not None
            else 0
        )
        return self.einet.get_effective_params() + proj_params

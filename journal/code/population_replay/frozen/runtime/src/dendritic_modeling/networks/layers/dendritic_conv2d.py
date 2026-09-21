"""
Dendritic Convolution Layer - CNN-like feature extraction with biological constraints.

This module implements a dendritic convolution layer that mimics cortical simple cells
with push-pull excitation/inhibition and divisive normalization.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class DendriteConv2d(nn.Module):
    """
    Dendritic Convolution Layer with E/I balance and divisive normalization.

    This layer implements the transfer function:
        v_out = E / (E + I + ε)

    where E and I are excitatory and inhibitory convolutions respectively.
    Each output channel can aggregate multiple excitatory and inhibitory
    dendrites (branches), with optional fixed synapse-count masks.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels (dendritic units)
    kernel_size : int or tuple
        Size of the convolution kernel
    stride : int or tuple, optional
        Stride of the convolution (default: 1)
    padding : int or tuple, optional
        Padding added to input (default: 0)
    dilation : int or tuple, optional
        Spacing between kernel elements (default: 1)
    groups : int, optional
        Number of blocked connections from input to output (default: 1)
    bias : bool, optional
        If True, adds a learnable bias to the output (default: True)
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-4)
    init_method : str, optional
        Weight initialization method (default: 'softplus_normal')
    n_e_dendrites : int, optional
        Number of excitatory dendrites per output neuron/channel.
    n_i_dendrites : int, optional
        Number of inhibitory dendrites per output neuron/channel.
    e_synapses_per_dendrite : int, optional
        Number of active excitatory synapses per dendrite. If None, uses full
        receptive field.
    i_synapses_per_dendrite : int, optional
        Number of active inhibitory synapses per dendrite. If None, uses full
        receptive field.
    branch_aggregation : str, optional
        Branch aggregation mode: "sum", "mean", or "max".
    morphology_seed : int, optional
        Seed for deterministic synapse-mask construction.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        epsilon: float = 1e-4,
        init_method: str = "softplus_normal",
        v_in_channels: Optional[int] = None,
        n_e_dendrites: int = 1,
        n_i_dendrites: int = 1,
        e_synapses_per_dendrite: Optional[int] = None,
        i_synapses_per_dendrite: Optional[int] = None,
        branch_aggregation: str = "sum",
        morphology_seed: Optional[int] = None,
    ):
        super().__init__()

        if groups <= 0:
            raise ValueError("groups must be positive")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if n_e_dendrites < 1 or n_i_dendrites < 1:
            raise ValueError("n_e_dendrites and n_i_dendrites must be >= 1")
        if branch_aggregation not in {"sum", "mean", "max"}:
            raise ValueError("branch_aggregation must be one of: sum, mean, max")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        if len(self.kernel_size) != 2:
            raise ValueError("kernel_size must be an int or a tuple of length 2")
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.dilation = (
            dilation if isinstance(dilation, tuple) else (dilation, dilation)
        )
        self.groups = groups
        self.epsilon = epsilon
        self.init_method = init_method
        self.branch_aggregation = branch_aggregation
        self.n_e_dendrites = n_e_dendrites
        self.n_i_dendrites = n_i_dendrites
        self.morphology_seed = morphology_seed
        self.in_channels_per_group = in_channels // groups

        total_synapses_per_dendrite = (
            self.in_channels_per_group * self.kernel_size[0] * self.kernel_size[1]
        )
        self.e_synapses_per_dendrite = self._validate_synapse_count(
            e_synapses_per_dendrite,
            total_synapses_per_dendrite,
            "e_synapses_per_dendrite",
        )
        self.i_synapses_per_dendrite = self._validate_synapse_count(
            i_synapses_per_dendrite,
            total_synapses_per_dendrite,
            "i_synapses_per_dendrite",
        )

        # V_in channels (for apical/contextual input)
        self.v_in_channels = v_in_channels or out_channels

        # Excitatory branch kernels (positive via softplus, then masked)
        self.weight_E_raw = nn.Parameter(
            torch.empty(
                out_channels,
                n_e_dendrites,
                self.in_channels_per_group,
                *self.kernel_size,
            )
        )

        # Inhibitory branch kernels (positive via softplus, then masked)
        self.weight_I_raw = nn.Parameter(
            torch.empty(
                out_channels,
                n_i_dendrites,
                self.in_channels_per_group,
                *self.kernel_size,
            )
        )

        # Optional bias terms
        if bias:
            self.bias_E = nn.Parameter(torch.empty(out_channels))
            self.bias_I = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias_E", None)
            self.register_parameter("bias_I", None)

        # Apical modulation weights (for v_in integration)
        self.apical_E = nn.Parameter(torch.empty(out_channels, self.v_in_channels))
        self.apical_I = nn.Parameter(torch.empty(out_channels, self.v_in_channels))

        # Fixed morphology masks controlling active synapses per dendrite.
        self.register_buffer(
            "weight_E_mask",
            self._make_synapse_mask(
                out_channels=out_channels,
                n_dendrites=n_e_dendrites,
                synapses_per_dendrite=self.e_synapses_per_dendrite,
                base_seed=morphology_seed,
            ),
        )
        self.register_buffer(
            "weight_I_mask",
            self._make_synapse_mask(
                out_channels=out_channels,
                n_dendrites=n_i_dendrites,
                synapses_per_dendrite=self.i_synapses_per_dendrite,
                base_seed=None if morphology_seed is None else morphology_seed + 1,
            ),
        )

        self.reset_parameters()

    def _validate_synapse_count(
        self,
        synapses_per_dendrite: Optional[int],
        max_synapses: int,
        field_name: str,
    ) -> int:
        if synapses_per_dendrite is None:
            return max_synapses
        if synapses_per_dendrite < 1:
            raise ValueError(f"{field_name} must be >= 1 when provided")
        if synapses_per_dendrite > max_synapses:
            raise ValueError(
                f"{field_name} ({synapses_per_dendrite}) cannot exceed receptive-field "
                f"synapse count ({max_synapses})"
            )
        return int(synapses_per_dendrite)

    def _make_synapse_mask(
        self,
        out_channels: int,
        n_dendrites: int,
        synapses_per_dendrite: int,
        base_seed: Optional[int],
    ) -> torch.Tensor:
        """
        Build a fixed binary mask with exactly `synapses_per_dendrite` active
        synapses per (output_channel, dendrite).
        """
        rf_synapses = (
            self.in_channels_per_group * self.kernel_size[0] * self.kernel_size[1]
        )
        if synapses_per_dendrite >= rf_synapses:
            return torch.ones(
                out_channels,
                n_dendrites,
                self.in_channels_per_group,
                self.kernel_size[0],
                self.kernel_size[1],
            )

        mask_flat = torch.zeros(out_channels, n_dendrites, rf_synapses)
        generator = torch.Generator(device="cpu")
        # base_seed=None must follow the global RNG (set_seed/experiment.seed),
        # not torch's fixed default seed — otherwise the mask is pinned identically
        # across experiment seeds. Mirrors synapse/indexed._connection_generator.
        if base_seed is None:
            base_seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator.manual_seed(int(base_seed) % (2**63 - 1))

        for out_idx in range(out_channels):
            for dend_idx in range(n_dendrites):
                active_idx = torch.randperm(rf_synapses, generator=generator)[
                    :synapses_per_dendrite
                ]
                mask_flat[out_idx, dend_idx, active_idx] = 1.0

        return mask_flat.view(
            out_channels,
            n_dendrites,
            self.in_channels_per_group,
            self.kernel_size[0],
            self.kernel_size[1],
        )

    def reset_parameters(self):
        """Initialize parameters."""
        if self.init_method == "softplus_normal":
            # Initialize in pre-softplus space
            fan_in_e = max(1, self.e_synapses_per_dendrite)
            fan_in_i = max(1, self.i_synapses_per_dendrite)
            std_e = 0.01 / np.sqrt(fan_in_e)
            std_i = 0.01 / np.sqrt(fan_in_i)

            # Softplus inverse: log(exp(x) - 1)
            mean_pre = np.log(np.exp(0.1) - 1)  # Small positive mean

            nn.init.normal_(self.weight_E_raw, mean=mean_pre, std=std_e)
            nn.init.normal_(self.weight_I_raw, mean=mean_pre, std=std_i)

            # Apical weights
            nn.init.normal_(self.apical_E, mean=0, std=0.01)
            nn.init.normal_(self.apical_I, mean=0, std=0.01)

        elif self.init_method == "kaiming":
            nn.init.kaiming_normal_(
                self.weight_E_raw, mode="fan_out", nonlinearity="relu"
            )
            nn.init.kaiming_normal_(
                self.weight_I_raw, mode="fan_out", nonlinearity="relu"
            )
            nn.init.kaiming_normal_(self.apical_E, mode="fan_out", nonlinearity="relu")
            nn.init.kaiming_normal_(self.apical_I, mode="fan_out", nonlinearity="relu")

        if self.bias_E is not None:
            nn.init.zeros_(self.bias_E)
            nn.init.zeros_(self.bias_I)

    @property
    def weight_e(self):
        """Effective excitatory kernel after branch aggregation."""
        return self._aggregate_kernel_bank(self.weight_e_branches)

    @property
    def weight_i(self):
        """Effective inhibitory kernel after branch aggregation."""
        return self._aggregate_kernel_bank(self.weight_i_branches)

    @property
    def weight_e_branches(self):
        """Excitatory branch kernels with positivity + morphology mask applied."""
        return functional.softplus(self.weight_E_raw) * self.weight_E_mask

    @property
    def weight_i_branches(self):
        """Inhibitory branch kernels with positivity + morphology mask applied."""
        return functional.softplus(self.weight_I_raw) * self.weight_I_mask

    def _aggregate_kernel_bank(self, weights: torch.Tensor) -> torch.Tensor:
        if self.branch_aggregation == "sum":
            return weights.sum(dim=1)
        elif self.branch_aggregation == "mean":
            return weights.mean(dim=1)
        elif self.branch_aggregation == "max":
            return weights.max(dim=1).values
        raise ValueError(f"Unknown branch_aggregation: {self.branch_aggregation}")

    def _aggregate_branch_responses(
        self, branch_response: torch.Tensor
    ) -> torch.Tensor:
        # branch_response shape: (batch, out_channels, n_dendrites, h, w)
        if self.branch_aggregation == "sum":
            return branch_response.sum(dim=2)
        elif self.branch_aggregation == "mean":
            return branch_response.mean(dim=2)
        elif self.branch_aggregation == "max":
            return branch_response.max(dim=2).values
        raise ValueError(f"Unknown branch_aggregation: {self.branch_aggregation}")

    def _branch_conv2d(
        self, x: torch.Tensor, branch_weights: torch.Tensor, n_dendrites: int
    ) -> torch.Tensor:
        """
        Compute per-branch conv responses then aggregate across branch dimension.
        """
        weight = branch_weights.reshape(
            self.out_channels * n_dendrites,
            self.in_channels_per_group,
            self.kernel_size[0],
            self.kernel_size[1],
        )
        response = functional.conv2d(
            x,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        response = response.view(
            x.shape[0],
            self.out_channels,
            n_dendrites,
            response.shape[2],
            response.shape[3],
        )
        return self._aggregate_branch_responses(response)

    def forward(
        self, x: torch.Tensor, v_in: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through dendritic convolution.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, in_channels, height, width)
        v_in : torch.Tensor, optional
            Apical/contextual input of shape (batch, v_in_channels) or
            (batch, v_in_channels, height, width)

        Returns
        -------
        v_out : torch.Tensor
            Output tensor of shape (batch, out_channels, height_out, width_out)
        """
        # Excitatory and inhibitory branch computations.
        E = self._branch_conv2d(x, self.weight_e_branches, self.n_e_dendrites)
        I_var = self._branch_conv2d(x, self.weight_i_branches, self.n_i_dendrites)

        if self.bias_E is not None:
            E = E + self.bias_E[None, :, None, None]
            I_var = I_var + self.bias_I[None, :, None, None]

        # Apply apical modulation if v_in is provided
        if v_in is not None:
            # If v_in is global (batch, channels), expand to spatial
            if v_in.dim() == 2:
                v_in = v_in.unsqueeze(-1).unsqueeze(-1)
                v_in = v_in.expand(-1, -1, E.shape[2], E.shape[3])
            elif v_in.dim() != 4:
                raise ValueError("v_in must be shape (B, C) or (B, C, H, W)")

            if v_in.shape[1] != self.v_in_channels:
                raise ValueError(
                    f"v_in channels ({v_in.shape[1]}) must match v_in_channels ({self.v_in_channels})"
                )

            # Modulate E and I with apical input
            # Shape: (batch, out_channels, H, W) = (batch, v_in_channels, H, W) @ (v_in_channels, out_channels)
            apical_mod_E = (
                functional.conv2d(
                    v_in.permute(0, 2, 3, 1).reshape(-1, self.v_in_channels, 1, 1),
                    self.apical_E.t().reshape(
                        self.out_channels, self.v_in_channels, 1, 1
                    ),
                )
                .reshape(x.shape[0], E.shape[2], E.shape[3], self.out_channels)
                .permute(0, 3, 1, 2)
            )

            apical_mod_I = (
                functional.conv2d(
                    v_in.permute(0, 2, 3, 1).reshape(-1, self.v_in_channels, 1, 1),
                    self.apical_I.t().reshape(
                        self.out_channels, self.v_in_channels, 1, 1
                    ),
                )
                .reshape(x.shape[0], I_var.shape[2], I_var.shape[3], self.out_channels)
                .permute(0, 3, 1, 2)
            )

            E = E * torch.sigmoid(apical_mod_E)
            I_var = I_var * torch.sigmoid(apical_mod_I)

        # Divisive normalization
        v_out = E / (E + I_var + self.epsilon)

        return v_out

    def extra_repr(self) -> str:
        """Extra representation for print."""
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
            ", n_e_dendrites={n_e_dendrites}, n_i_dendrites={n_i_dendrites}"
            ", e_synapses_per_dendrite={e_synapses_per_dendrite}"
            ", i_synapses_per_dendrite={i_synapses_per_dendrite}"
            ", branch_aggregation={branch_aggregation}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias_E is None:
            s += ", bias=False"
        s += f", epsilon={self.epsilon}"
        return s.format(**self.__dict__)


class DendriteAttention(nn.Module):
    """
    Dendritic Attention mechanism with E/I balance.

    Replaces softmax attention with divisive normalization:
        a_ij = E_ij / (Σ_k E_ik + I_i + ε)

    Parameters
    ----------
    embed_dim : int
        Total dimension of the model
    num_heads : int
        Number of parallel attention heads
    dropout : float, optional
        Dropout probability (default: 0.0)
    epsilon : float, optional
        Small constant for numerical stability (default: 1e-4)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        epsilon: float = 1e-4,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.epsilon = epsilon

        assert (
            self.head_dim * num_heads == embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Inhibitory gating (per query)
        self.i_proj = nn.Linear(embed_dim, num_heads)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # Initialize inhibition to be slightly weaker
        nn.init.normal_(self.i_proj.weight, std=0.01)

        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)
            nn.init.zeros_(self.i_proj.bias)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        v_in: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through dendritic attention.

        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (batch, seq_len, embed_dim)
        key : torch.Tensor
            Key tensor of shape (batch, seq_len, embed_dim)
        value : torch.Tensor
            Value tensor of shape (batch, seq_len, embed_dim)
        attn_mask : torch.Tensor, optional
            Attention mask of shape (batch, seq_len, seq_len)
        v_in : torch.Tensor, optional
            Apical input for modulation

        Returns
        -------
        output : torch.Tensor
            Output tensor of shape (batch, seq_len, embed_dim)
        attn_weights : torch.Tensor
            Attention weights of shape (batch, num_heads, seq_len, seq_len)
        """
        batch_size, seq_len, _ = query.shape

        # Project and reshape
        Q = (
            self.q_proj(query)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(key)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(value)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores (excitatory)
        E = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if attn_mask is not None:
            E = E.masked_fill(attn_mask == 0, -1e9)

        # Compute inhibition (per query, per head)
        I_var = functional.softplus(self.i_proj(query))  # (batch, seq_len, num_heads)
        I_var = I_var.transpose(1, 2).unsqueeze(-1)  # (batch, num_heads, seq_len, 1)

        # Apply v_in modulation if provided
        if v_in is not None:
            # v_in modulates the inhibition strength
            v_in_mod = torch.sigmoid(v_in).mean(
                dim=-1, keepdim=True
            )  # Global modulation
            I_var = I_var * v_in_mod.unsqueeze(1)

        # Divisive normalization attention
        E_positive = functional.softplus(E)
        E_sum = E_positive.sum(dim=-1, keepdim=True)
        attn_weights = E_positive / (E_sum + I_var + self.epsilon)

        # Apply attention to values
        attn_output = torch.matmul(self.dropout(attn_weights), V)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.embed_dim)
        )
        output = self.out_proj(attn_output)

        return output, attn_weights

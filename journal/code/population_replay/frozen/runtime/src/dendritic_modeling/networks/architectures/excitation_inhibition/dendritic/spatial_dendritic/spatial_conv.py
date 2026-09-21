"""HierarchicalDendriticConv — spatially local dendritic core for conv layer replacement.

Replaces a Conv2d layer (or Conv2d + activation block) with dendritic E-I
computation over local patches.  A single shared :class:`ConfigurableEINetwork`
is evaluated on all unfolded spatial positions in a batched forward pass.

Pipeline::

    input (B, C_in, H, W)
        → F.unfold(kernel_size, stride, padding)          → (B, patch_dim, L)
        → permute + reshape                                → (B*L, patch_dim)
        → ConfigurableEINetwork                            → (B*L, C_out)
        → reshape + permute                                → (B, C_out, H_out, W_out)

where ``patch_dim = C_in x k_h x k_w`` and ``L = H_out x W_out``.

Every dendritic depth receives the feedforward patch as input (not just leaves),
preserving the same signal flow as the standard DendriNet.  Weight sharing
across spatial positions is automatic — the same EINet parameters are applied
to every unfolded patch.
"""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
)
from dendritic_modeling.networks.architectures.replacement import (
    SpatialNonNegativeInputAdapter,
    make_nonmixing_output_adapter,
)
from dendritic_modeling.networks.base import BaseNetwork

logger = logging.getLogger(__name__)


def _pair(x):
    """Convert scalar to 2-tuple, pass tuples/lists through."""
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


class HierarchicalDendriticConv(BaseNetwork):
    """Spatially local hierarchical dendritic core.

    Replaces a Conv2d layer (or a multi-layer conv block) with dendritic E-I
    computation over local patches.  A single shared
    :class:`ConfigurableEINetwork` is evaluated on all unfolded spatial
    positions in a batched forward pass.

    When replacing a multi-layer conv range (e.g. Conv2d + ReLU + Conv2d),
    the dendritic conv acts as a learned surrogate with user-chosen local
    geometry (kernel_size, stride, padding), not a faithful 1:1 replacement
    of each original layer.

    The spatial contract:
      - Input must be 4D: ``(B, C_in, H, W)``
      - Output is 4D: ``(B, C_out, H_out, W_out)``
      - ``C_out = excitatory_layer_sizes[-1]`` from the EINet config

    Args:
        config: Structured EINet config (architecture, connectivity, etc.).
        in_channels: Number of input channels (C_in).
        kernel_size: Spatial extent of the local receptive field.
        stride: Stride for the unfold operation.
        padding: Padding applied before unfolding.
        input_transform: Non-negative input contract for the patch
            (``identity``, ``relu``, or ``signed_split``).
        input_scale: Fixed positive drive calibration applied to the patch
            after ``input_transform``. Pretrained conv feature maps exceed the
            [0, 1] activity range the dendritic conductance initialization
            assumes; a boundary-specific scale keeps the total branch drive
            (and hence the shunting denominator) near its analytical design
            operating point. Defaults to 1.0, which skips the multiply and
            reproduces the historical forward exactly. A declared constant,
            not a parameter: no state-dict entries, so pre-existing
            checkpoints load strictly and resource accounting is unchanged.
            Consumed only by this core; the sparse point control ignores it.
        output_adapter_mode: Channel-preserving positive output stage (see
            ``make_nonmixing_output_adapter``); threshold-ReLU modes supply an
            exact-zero sparse output contract.
        synapse_mode: Override for the EINet synapse mode.
        use_shunting: Override for shunting inhibition.
        weight_transform: Override for weight transform.
    """

    def __init__(
        self,
        config: dict | DictConfig,
        in_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        input_transform: str = "identity",
        input_scale: float = 1.0,
        output_adapter_mode: str = "identity",
        output_initial_scale: float = 1.0,
        output_initial_threshold: float = 0.25,
        synapse_mode: str | None = None,
        use_shunting: bool | None = None,
        weight_transform: str | None = None,
    ):
        super().__init__()

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.in_channels = in_channels
        input_scale = float(input_scale)
        if not input_scale > 0.0:
            raise ValueError(f"input_scale must be positive, got {input_scale}")
        # Declared fixed drive calibration: pretrained conv feature maps carry
        # activations far above the [0, 1] range the dendritic conductance
        # initialization assumes, so the boundary config may rescale the patch
        # before it drives the shared E/I network.
        self.input_scale = input_scale
        self.input_adapter = SpatialNonNegativeInputAdapter(
            in_channels,
            transform=input_transform,
        )
        self.adapted_in_channels = self.input_adapter.output_channels

        patch_dim = self.adapted_in_channels * self.kernel_size[0] * self.kernel_size[1]

        self.einet = ConfigurableEINetwork(
            config=config,
            input_dim=patch_dim,
            synapse_mode=synapse_mode,
            use_shunting=use_shunting,
            weight_transform=weight_transform,
        )

        self.out_channels = self.einet.output_dim
        self.output_dim = self.out_channels  # for introspection
        self.output_adapter_mode = str(output_adapter_mode).strip().lower()
        self.output_adapter = make_nonmixing_output_adapter(
            num_features=self.out_channels,
            feature_axis=1,
            mode=self.output_adapter_mode,
            initial_scale=output_initial_scale,
            initial_threshold=output_initial_threshold,
        )

        logger.info(
            "HierarchicalDendriticConv: in_channels=%d, kernel=%s, stride=%s, "
            "padding=%s, patch_dim=%d, out_channels=%d",
            in_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            patch_dim,
            self.out_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, C_in, H, W)``.

        Returns:
            Output tensor of shape ``(B, C_out, H_out, W_out)``.
        """
        if x.dim() != 4:
            raise ValueError(
                f"HierarchicalDendriticConv requires 4D input (B, C, H, W), "
                f"got {x.dim()}D tensor with shape {tuple(x.shape)}. "
                f"This core type is for replacing spatial (conv) layers. "
                f"For flat input, use tensor_map or einet instead."
            )

        B, _C, H, W = x.shape
        x = self.input_adapter(x)
        if self.input_scale != 1.0:
            x = x * self.input_scale

        # 1. Unfold into local patches: (B, patch_dim, L) where L = H_out * W_out
        patches = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        L = patches.shape[2]

        # 2. Reshape for batched DendriNet: (B*L, patch_dim)
        patches = patches.permute(0, 2, 1).reshape(B * L, -1)

        # 3. Run shared ConfigurableEINetwork
        out = self.einet(patches)  # (B*L, C_out)

        # 4. Reshape back to spatial: (B, C_out, H_out, W_out)
        H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        out = out.reshape(B, H_out, W_out, self.out_channels).permute(0, 3, 1, 2)
        out = self.output_adapter(out)

        return out.contiguous()

    def get_output_dim(self) -> int:
        return self.output_dim

    def compute_output_shape(self, H: int, W: int) -> tuple[int, int]:
        """Compute (H_out, W_out) for a given input spatial size."""
        H_out = (H + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (W + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        return (H_out, W_out)

    def spatial_output_contract(self) -> dict[str, int | float | str | bool]:
        """Describe how local dendritic outputs form the next feature map.

        One shared E/I network is evaluated at every output position. Its
        excitatory soma index is the output-channel index, so channel identity
        is consistent across the full image. The local inhibitory population
        contributes inside that position's E/I computation but is not emitted
        as a second feature stream.
        """

        return {
            "input_layout": "BCHW",
            "patch_layout": "channel_y_x_flat",
            "output_layout": "BCHW",
            "spatial_weight_sharing": True,
            "input_transform": self.input_adapter.transform,
            "input_scale": self.input_scale,
            "adapted_input_channels": self.adapted_in_channels,
            "output_channels": self.out_channels,
            "excitatory_somas_per_location": self.out_channels,
            "channel_mapping": "one_excitatory_soma_per_output_channel",
            "propagated_population": "excitatory",
            "inhibitory_population_scope": "local_per_output_position",
            "output_adapter": self.output_adapter_mode,
        }

    # ------------------------------------------------------------------
    # Training hook delegation
    # ------------------------------------------------------------------
    def decay_weights(self, weight_decay, weight_boosting=False):
        """Forward to the inner EINet."""
        self.einet.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self):
        """Forward to the inner EINet."""
        self.einet.apply_rewiring()

    # ------------------------------------------------------------------
    # Introspection delegation
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

    def get_effective_params(self) -> int:
        """Count active core parameters and explicit output-adapter scalars."""
        adapter_params = sum(
            parameter.numel() for parameter in self.output_adapter.parameters()
        )
        return self.einet.get_effective_params() + adapter_params

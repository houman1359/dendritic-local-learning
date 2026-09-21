"""
Block Linear Layer for Dendritic Networks.

This module implements a custom linear layer that aggregates information
from converging branches in a previous branch layer.
"""

import logging
from math import floor, log

import torch
import torch.nn as nn

from dendritic_modeling.networks.utils.weight_transforms import (
    WeightTransformType,
    apply_weight_transform,
    inverse_softplus,
    inverse_weight_transform,
)

logger = logging.getLogger(__name__)


class BlockLinear(nn.Module):
    """
    A custom linear layer that aggregates information from converging branches
    in a previous branch layer.

    Parameters
    ----------

    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    requires_grad : bool, optional
        If `True`, the weights are trainable. Defaults to `False`.

    Attributes
    ----------
    weight : torch.nn.Parameter
        The weights for the linear transformation.

    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    block_size : int
        The number of branches converging onto a single branch.

    Methods
    -------
    forward(x)
        Performs a forward pass through the layer.

    sum_conductances()
        Returns the sum of conductances for each output neuron.

    Notes
    -----

    - Block_size equals number of branches converging onto single branch/soma
      in next layer.

    """

    def __init__(
        self, in_features, out_features, weight_transform: WeightTransformType = "exp"
    ):
        super().__init__()

        block_size = floor(in_features / out_features)
        assert (
            in_features == out_features * block_size
        ), "in_features must be divisible by out_features."

        self.log_weight = nn.Parameter(
            torch.empty((out_features, block_size)), requires_grad=True
        )
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.weight_transform = weight_transform
        self.input_scale_vec = None  # may be used for gradient scaling
        self.param_scale_vec = None  # may be used for gradient scaling
        self._use_forward_dynamic_grad_scaling = False
        self._last_forward_input_tensor = None
        self._last_forward_param_tensor = None
        # Inference-frozen folds of the transformed block weights and their
        # per-branch conductance sums. Non-persistent: checkpoints keep only
        # the raw ``log_weight`` parameterization.
        self.register_buffer("_inference_folded_weight", None, persistent=False)
        self.register_buffer("_inference_folded_conductance", None, persistent=False)

    def freeze_for_inference(self) -> None:
        """Fold transformed block weights and conductance sums for inference.

        ``weight()`` and ``sum_conductances()`` are input-independent but are
        recomputed on every forward during training. Folding computes both once
        so the frozen forward performs no weight transform and the shunting
        denominator reuses a constant conductance tensor. Dropped automatically
        by ``train(True)`` or by loading new weights; never serialized.
        """
        self.log_weight.requires_grad_(False)
        with torch.no_grad():
            folded = apply_weight_transform(
                self.log_weight, self.weight_transform
            ).detach()
            self._inference_folded_weight = folded.contiguous()
            self._inference_folded_conductance = folded.sum(dim=1).contiguous()
        self.eval()

    def unfreeze_inference_fold(self) -> None:
        """Drop the folded inference tensors (weights may change again)."""
        self._inference_folded_weight = None
        self._inference_folded_conductance = None

    def train(self, mode: bool = True):
        if mode:
            self.unfreeze_inference_fold()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self.unfreeze_inference_fold()
        return super()._load_from_state_dict(*args, **kwargs)

    def initialize(self, g_branch: float = 1.0):
        """
        Initialize the block weights to a specific conductance value.

        Args:
            g_branch: The desired conductance value (positive) after weight transformation.
                     The method handles conversion to the appropriate raw parameter space
                     based on self.weight_transform.
        """
        # Convert the desired conductance to the appropriate raw parameter value
        raw_value = inverse_weight_transform(
            torch.tensor(
                g_branch, dtype=self.log_weight.dtype, device=self.log_weight.device
            ),
            self.weight_transform,
        ).item()

        # Initialize the raw parameters
        nn.init.constant_(self.log_weight, raw_value)

    def weight(self):
        """Return transformed per-block weights with shape (out_features, block_size)."""
        if not self.training and self._inference_folded_weight is not None:
            return self._inference_folded_weight
        return apply_weight_transform(self.log_weight, self.weight_transform)

    def _block_indices(self):
        """Row/col index tensors that scatter per-block weights into the dense (out, in) matrix."""
        row_ix = torch.arange(self.out_features, device=self.log_weight.device)[:, None]
        col_ix = torch.arange(self.in_features, device=self.log_weight.device).view(
            self.out_features, self.block_size
        )
        return row_ix, col_ix

    def block(self):
        block = torch.zeros(
            (self.out_features, self.in_features),
            device=self.log_weight.device,
            dtype=self.log_weight.dtype,
        )
        row_ix, col_ix = self._block_indices()
        weights = self.weight()
        block[row_ix, col_ix] = weights
        return block

    def log_block(self):
        log_block = (
            torch.ones(
                (self.out_features, self.in_features),
                device=self.log_weight.device,
                dtype=self.log_weight.dtype,
            )
            * -10
        )
        row_ix, col_ix = self._block_indices()
        log_block[row_ix, col_ix] = self.log_weight
        return log_block

    def grad_block(self):
        grad_block = torch.zeros(
            (self.out_features, self.in_features),
            device=self.log_weight.device,
            dtype=self.log_weight.dtype,
        )
        row_ix, col_ix = self._block_indices()
        if self.log_weight.grad is not None:
            grad_block[row_ix, col_ix] = self.log_weight.grad
        return grad_block

    def decay_weights(self, weight_decay, weight_boosting=False):
        with torch.no_grad():
            if self.weight_transform in ["identity", "relu"]:
                self.log_weight.data *= 1 - weight_decay
            elif self.weight_transform == "exp":
                decay = log(1 - weight_decay)
                self.log_weight.data += decay
            elif self.weight_transform == "softplus":
                w = nn.functional.softplus(self.log_weight.data)
                decayed_w = w * (1 - weight_decay)
                p_prime = inverse_softplus(decayed_w)
                self.log_weight.data = p_prime

            if weight_boosting:
                if self.weight_transform in ["identity", "relu"]:
                    self.log_weight.data *= 1 + weight_decay
                elif self.weight_transform == "exp":
                    boost = log(1 + weight_decay)
                    self.log_weight.data += boost
                elif self.weight_transform == "softplus":
                    w = nn.functional.softplus(self.log_weight.data)
                    boosted_w = w * (1 + weight_decay)
                    p_prime = inverse_softplus(boosted_w)
                    self.log_weight.data = p_prime

    def forward(self, x):
        # Ensure x is a tensor
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")

        original_shape = x.shape

        # Handle different input dimensions
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            # Handle higher dimensional inputs by flattening batch dimensions
            # This is common during analysis when data has extra batch dimensions
            x = x.reshape(-1, x.shape[-1])

        # x should now be 2D: [batch_size, features]
        if x.dim() != 2:
            raise ValueError(
                f"BlockLinear expects 2D input tensor after reshaping, got shape {x.shape} with {x.dim()} dimensions"
            )

        # Validate input dimension
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"Input feature dimension mismatch: got {x.shape[1]}, expected {self.in_features}"
            )

        self._last_forward_input_tensor = x

        block = torch.zeros(
            (self.out_features, self.in_features),
            device=self.log_weight.device,
            dtype=self.log_weight.dtype,
        )
        row_ix, col_ix = self._block_indices()
        weights = self.weight() * 1.0
        object.__setattr__(self, "_last_forward_param_tensor", weights)
        block[row_ix, col_ix] = weights
        output = torch.matmul(x, block.t())

        # Reshape output to match input shape structure if needed
        if len(original_shape) > 2:
            # Restore the original batch dimensions
            output_shape = [*original_shape[:-1], self.out_features]
            output = output.reshape(output_shape)

        return output

    def register_forward_gradient_scales(
        self,
        *,
        param_scale: torch.Tensor | None = None,
        input_scale: torch.Tensor | None = None,
    ) -> None:
        """Scale input/parameter gradients for the most recent forward call only."""
        self._use_forward_dynamic_grad_scaling = True
        if param_scale is not None:
            target = self._last_forward_param_tensor
            if target is not None and target.requires_grad:
                scale = param_scale.detach().to(
                    device=target.device, dtype=target.dtype
                )
                target.register_hook(lambda grad, scale=scale: grad * scale)
        if input_scale is not None:
            target = self._last_forward_input_tensor
            if target is not None and target.requires_grad:
                scale = input_scale.detach().to(
                    device=target.device, dtype=target.dtype
                )
                target.register_hook(lambda grad, scale=scale: grad * scale)

    def sum_conductances(self):
        if not self.training and self._inference_folded_conductance is not None:
            return self._inference_folded_conductance
        return self.weight().sum(dim=1)


# ---------------------------------------------------------------------------
# Memory-efficient implementation that avoids constructing the block-diagonal
# matrix. It is API-compatible with BlockLinear but allocates only the log_weight
# parameter tensor and no large temporaries.
# ---------------------------------------------------------------------------


class EfficientBlockLinear(BlockLinear):
    """BlockLinear variant with *O(out_features x block_size)* memory.

    Forward pass reshapes the input into (B, out, block_size) and performs a
    weighted sum along the last dimension, eliminating the need to materialise
    the big block-diagonal matrix created in :py:meth:`BlockLinear.block`.
    """

    def forward(self, x):  # type: ignore[override]
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected tensor, got {type(x)}")

        original_shape = x.shape

        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() > 2:
            # Handle higher dimensional inputs by flattening batch dimensions
            # This is common during analysis when data has extra batch dimensions
            feature_dim = x.shape[-1]
            x = x.reshape(-1, feature_dim)

        # x should now be 2D: [B, features]
        B = x.shape[0]
        expected_dim = self.out_features * self.block_size

        # Validate input dimension
        if x.shape[1] != expected_dim:
            details = (
                f"EfficientBlockLinear expected {expected_dim} features "
                f"(out_features={self.out_features}, block_size={self.block_size}), "
                f"but got original shape {tuple(original_shape)} reshaped to {tuple(x.shape)}."
            )
            # Check if this might be a transposed or misshaped tensor.
            if x.shape[1] % self.block_size == 0:
                implied_out = x.shape[1] // self.block_size
                details += f" The provided width implies out_features={implied_out}."
            raise ValueError(details)

        x_reshaped = x.view(B, self.out_features, self.block_size)
        weights = (
            apply_weight_transform(self.log_weight, self.weight_transform) * 1.0
        )  # (out_features, block_size)
        self._last_forward_input_tensor = x
        object.__setattr__(self, "_last_forward_param_tensor", weights)
        output = (x_reshaped * weights).sum(dim=-1)  # (B, out_features)

        # Reshape output to match input shape structure if needed
        if len(original_shape) > 2:
            # Restore the original batch dimensions
            output_shape = [*original_shape[:-1], self.out_features]
            output = output.reshape(output_shape)

        return output


__all__ = ["BlockLinear", "EfficientBlockLinear"]

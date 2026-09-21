"""
TopK Linear Layer for Dendritic Networks.

This module implements the basic TopKLinear layer that retains only the top K
strongest synaptic weights for each output neuron, simulating selective
synaptic connectivity in dendritic computation.
"""

from collections.abc import Callable, Iterator
from math import log
from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    validate_connection_mask,
)
from dendritic_modeling.networks.utils.weight_transforms import (
    WeightTransformType,
    apply_weight_transform,
    inverse_softplus,
)
from dendritic_modeling.utils.hooks import iter_modules_matching

TOPK_INIT_METHODS = {
    "xavier_normal": nn.init.xavier_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "orthogonal": nn.init.orthogonal_,
    "normal": nn.init.normal_,
    "uniform": nn.init.uniform_,
    "eye": nn.init.eye_,
}


def _iter_recurrent_weight_cache_setters(
    module: nn.Module,
) -> Iterator[Callable[[bool], None]]:
    """Yield recurrent-cache setter callables in module traversal order."""
    for sub in iter_modules_matching(
        module,
        lambda candidate: callable(
            getattr(candidate, "set_recurrent_weight_cache", None)
        ),
    ):
        setter = getattr(sub, "set_recurrent_weight_cache", None)
        yield setter


def apply_recurrent_weight_cache(module: nn.Module, enabled: bool) -> None:
    """Toggle per-forward recurrent TopK caching on every TopK layer in a tree.

    Call with ``enabled=True`` immediately before a recurrent unroll's timestep
    loop and ``enabled=False`` (in a ``finally``) immediately after, so the
    deterministic TopK selection is computed once and reused across timesteps,
    then released. During autograd this caches the mask only; during no-grad
    inference it caches the full pruned weight. Layers that cannot safely cache
    (stochastic / variance / annealed, or any layer with ``noise_level>0`` or
    dynamic grad scaling) ignore the request internally, so this is always safe
    to call on a whole network.
    """
    for setter in _iter_recurrent_weight_cache_setters(module):
        setter(enabled)


class TopKLinear(nn.Module):
    """
    A linear layer that retains only the top K strongest synaptic weights for
    in_features.

    This module implements a linear transformation with weights constrained to
    be positive. For each output neuron, only the top K weights are kept
    , and the rest are set to zero. This simulates a neuron receiving
    inputs only from its strongest synaptic connections.

    Parameters
    ----------
    in_features : int
        The number of input features.
    out_features : int
        The number of output features.
    K : int
        The number of strongest synapses to keep per dendritic branch.
    param_space : str, optional
        The parameter space for the weights. Options are 'log' and 'presigmoid'.
        Defaults to 'log'.

    Notes
    -----
    - All weights are constrained to be positive.
    - The pruning is done dynamically during the forward pass.
    - The masked matrix is still materialized densely. For very sparse large
      layers, use the indexed sparse layers instead of expecting this module
      to reduce matrix-multiply memory/compute.
    """

    # Whether this class's TopK selection is deterministic within a single
    # forward and therefore safe to reuse across timesteps of a recurrent
    # unroll (see ``set_recurrent_weight_cache``). Subclasses whose mask can
    # change within/between forward calls (stochastic, variance-tracking,
    # annealed dense->sparse) override this to ``False``.
    _supports_recurrent_weight_cache: bool = True

    def __init__(
        self,
        in_features,
        out_features,
        K,
        param_space="log",
        init_method="xavier_normal",
        init_gain=1.0,
        noise_level=0.0,
        weight_transform: WeightTransformType = "exp",
        weight_norm_order: Optional[int] = None,
        gamma: float = 1.0,
        forbidden_input_index_per_output: Optional[torch.Tensor] = None,
        connection_mask: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method
        self.init_gain = init_gain
        self.noise_level = noise_level
        self.weight_transform = weight_transform
        self.weight_norm_order = weight_norm_order
        self.gamma = gamma

        # zeros (not empty): pre_w is overwritten by initialize() in production
        # (branch_layer calls it), but a read before initialize() must be
        # well-defined — torch.empty leaves uninitialized memory that is
        # occasionally NaN/inf, making any pre-initialize read undefined.
        self.pre_w = nn.Parameter(
            torch.zeros((out_features, in_features)), requires_grad=True
        )

        if not isinstance(K, int):
            K = int(K)  # Ensure K is an integer

        if K < 1:
            raise ValueError("K must be >= 1")

        if K > in_features:
            raise ValueError(
                f"K must be <= number of input features. (K = {K}, in_features = {in_features})"
            )

        self.K = K
        self.param_space = param_space
        self.input_scale_vec = None  # may be used for gradient scaling
        self.param_scale_vec = None  # may be used for gradient scaling
        self._use_forward_dynamic_grad_scaling = False
        self._last_forward_param_tensor: Optional[torch.Tensor] = None
        self.cache_mask = False
        self._last_forward_weight_mask: Optional[torch.Tensor] = None
        # Per-forward TopK reuse for recurrent unrolls (off by default).
        # During autograd we cache only the non-differentiable mask, preserving
        # the original per-timestep weight-transform graph exactly. Under
        # no_grad we can also cache the full pruned weight.
        self._recurrent_weight_cache_enabled = False
        self._recurrent_cached_weight_mask: Optional[torch.Tensor] = None
        self._recurrent_cached_pruned_weight: Optional[torch.Tensor] = None

        if forbidden_input_index_per_output is None:
            forbidden = torch.empty(0, dtype=torch.long)
        else:
            forbidden = torch.as_tensor(
                forbidden_input_index_per_output, dtype=torch.long
            ).view(-1)
            if forbidden.numel() != out_features:
                raise ValueError(
                    "forbidden_input_index_per_output must have one index per "
                    f"output feature ({out_features}), got {forbidden.numel()}"
                )
            valid = (forbidden >= -1) & (forbidden < in_features)
            if not bool(valid.all()):
                raise ValueError(
                    "forbidden_input_index_per_output entries must be -1 or in "
                    f"[0, {in_features - 1}]"
                )
        self.register_buffer(
            "_forbidden_input_index_per_output", forbidden, persistent=False
        )
        allowed = validate_connection_mask(
            connection_mask,
            out_features,
            in_features,
        )
        if allowed is None:
            allowed = torch.empty(0, dtype=torch.bool)
            persist_allowed = False
        else:
            persist_allowed = True
        self.register_buffer(
            "connection_mask",
            allowed,
            persistent=persist_allowed,
        )

    def initialize(self):
        """Initialize the layer weights using the specified method."""
        if self.init_method in TOPK_INIT_METHODS:
            init_func = TOPK_INIT_METHODS[self.init_method]
            init_func(self.pre_w)
            self.pre_w.data.mul_(self.init_gain)  # scale
        else:
            raise ValueError(
                f"Invalid initialization method: {self.init_method}. "
                f"Choose from {list(TOPK_INIT_METHODS.keys())}"
            )

    def decay_weights(self, weight_decay=0.1, weight_boosting=False):
        """Apply weight decay to active synapses and optionally boost inactive ones."""
        with torch.no_grad():
            weight_mask = self.weight_mask()
            if self.weight_transform == "exp":
                decay = log(1 - weight_decay) * weight_mask
                self.pre_w.data += decay
            elif self.weight_transform == "softplus":
                w = nn.functional.softplus(self.pre_w.data)
                decayed_w = w * (1 - weight_decay)
                p_prime = inverse_softplus(decayed_w)
                p_prime_masked = torch.where(weight_mask > 0, p_prime, self.pre_w.data)
                self.pre_w.data = p_prime_masked
            elif self.weight_transform in ["identity", "relu"]:
                # Signed weights: decay active weights multiplicatively, preserve sign
                decay = torch.where(weight_mask > 0, 1 - weight_decay, 1.0)
                self.pre_w.data *= decay

            if weight_boosting:
                allowed_mask = self._connection_mask_like(weight_mask)
                inv_weight_mask = ((-1 * weight_mask) + 1) * allowed_mask
                if self.weight_transform == "exp":
                    boost = log(1 + weight_decay) * inv_weight_mask
                    self.pre_w.data += boost
                elif self.weight_transform == "softplus":
                    w = nn.functional.softplus(self.pre_w.data)
                    boosted_w = w * (1 + weight_decay)
                    p_prime = inverse_softplus(boosted_w)
                    p_prime_masked = torch.where(
                        inv_weight_mask > 0, p_prime, self.pre_w.data
                    )
                    self.pre_w.data = p_prime_masked
                elif self.weight_transform in ["identity", "relu"]:
                    boost = torch.where(inv_weight_mask > 0, 1 + weight_decay, 1.0)
                    self.pre_w.data *= boost

    def _has_forbidden_inputs(self) -> bool:
        return self._forbidden_input_index_per_output.numel() > 0

    def _has_connection_mask(self) -> bool:
        return self.connection_mask.numel() > 0

    def _connection_mask_like(self, reference: torch.Tensor) -> torch.Tensor:
        if not self._has_connection_mask():
            return torch.ones_like(reference)
        return self.connection_mask.to(device=reference.device, dtype=reference.dtype)

    def _apply_forbidden_scores(
        self, scores: torch.Tensor, fill_value: float = float("-inf")
    ) -> torch.Tensor:
        if not self._has_forbidden_inputs() and not self._has_connection_mask():
            return scores
        adjusted = scores.clone()
        if self._has_connection_mask():
            allowed = self.connection_mask.to(device=adjusted.device)
            adjusted = adjusted.masked_fill(~allowed, fill_value)
        row_idx = torch.arange(
            adjusted.shape[0], device=adjusted.device, dtype=torch.long
        )
        if self._has_forbidden_inputs():
            col_idx = self._forbidden_input_index_per_output.to(adjusted.device)
            valid = col_idx >= 0
            if bool(valid.any()):
                adjusted[row_idx[valid], col_idx[valid]] = fill_value
        return adjusted

    def _apply_forbidden_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if not self._has_forbidden_inputs() and not self._has_connection_mask():
            return mask
        adjusted = mask.clone()
        if self._has_connection_mask():
            allowed = self.connection_mask.to(device=adjusted.device)
            adjusted = adjusted * allowed.to(dtype=adjusted.dtype)
        row_idx = torch.arange(
            adjusted.shape[0], device=adjusted.device, dtype=torch.long
        )
        if self._has_forbidden_inputs():
            col_idx = self._forbidden_input_index_per_output.to(adjusted.device)
            valid = col_idx >= 0
            if bool(valid.any()):
                adjusted[row_idx[valid], col_idx[valid]] = 0
        return adjusted

    def _apply_weight_norm(self, pruned_weight: torch.Tensor) -> torch.Tensor:
        if self.weight_norm_order is None:
            return pruned_weight

        w_norm = torch.norm(pruned_weight, p=self.weight_norm_order, dim=-1)
        safe_norm = w_norm.clamp_min(torch.finfo(pruned_weight.dtype).eps)
        normalized = (pruned_weight / safe_norm[:, None]) * self.gamma
        return torch.where(w_norm[:, None] > 0, normalized, pruned_weight)

    def _pruned_weight_from_mask(self, mask: torch.Tensor) -> torch.Tensor:
        return self._apply_weight_norm(mask * self.weight())

    def set_recurrent_weight_cache(self, enabled: bool) -> None:
        """Enable/disable a per-forward pruned-weight cache for recurrent unrolls.

        When enabled, deterministic TopK selection is computed once per unroll
        and reused on the remaining timesteps, then cleared. During training /
        autograd we cache only the non-differentiable mask and recompute the
        transformed weights each timestep, which preserves the original gradient
        graph exactly. During no-grad inference we cache the full pruned weight.

        The cache is automatically bypassed (and never populated) when it would
        change numerics: ``noise_level > 0`` (stochastic mask), dynamic forward
        gradient scaling is active, or for subclasses that set
        ``_supports_recurrent_weight_cache = False``. Always call with
        ``enabled=False`` after the unroll to release the cached tensor.
        """
        self._recurrent_weight_cache_enabled = (
            bool(enabled) and self._supports_recurrent_weight_cache
        )
        # Always drop any cached tensor on a state change so a stale weight can
        # never leak into a later forward.
        self._recurrent_cached_weight_mask = None
        self._recurrent_cached_pruned_weight = None

    def _recurrent_weight_cache_active(self) -> bool:
        return (
            self._recurrent_weight_cache_enabled
            and self._supports_recurrent_weight_cache
            and self.noise_level == 0
            and not self._use_forward_dynamic_grad_scaling
        )

    def forward(self, x):
        """Forward pass through the TopK linear layer."""
        # Ensure x is a 2D tensor for matrix multiplication
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")

        # If x is a 1D tensor, unsqueeze to make it 2D [1, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        use_cache = self._recurrent_weight_cache_active()
        if use_cache and not torch.is_grad_enabled():
            pruned_weight = self._recurrent_cached_pruned_weight
            if pruned_weight is None:
                mask = self.weight_mask()
                if self.cache_mask:
                    self._last_forward_weight_mask = mask.detach()
                pruned_weight = self._pruned_weight_from_mask(mask)
                self._recurrent_cached_pruned_weight = pruned_weight
        else:
            mask = self._recurrent_cached_weight_mask if use_cache else None
            if mask is None:
                mask = self.weight_mask()
                if use_cache:
                    self._recurrent_cached_weight_mask = mask
            if self.cache_mask:
                self._last_forward_weight_mask = mask.detach()
            pruned_weight = self._pruned_weight_from_mask(mask)

        object.__setattr__(
            self,
            "_last_forward_param_tensor",
            pruned_weight if self._use_forward_dynamic_grad_scaling else None,
        )

        return torch.matmul(x, pruned_weight.t())

    def register_forward_param_gradient_scale(self, param_scale: torch.Tensor) -> None:
        """Scale parameter gradients for the most recent forward call only."""
        self._use_forward_dynamic_grad_scaling = True
        target = self._last_forward_param_tensor
        if target is None or not target.requires_grad:
            return
        scale = param_scale.detach().to(device=target.device, dtype=target.dtype)
        target.register_hook(lambda grad, scale=scale: grad * scale)

    def weight(self):
        """Return the transformed synaptic weights."""
        return apply_weight_transform(self.pre_w, self.weight_transform)

    def log_weight(self):
        """Return the log of the transformed weights."""
        w = self.weight()
        if self.weight_transform == "identity":
            # Avoid log of negative values; use magnitude as a stable proxy
            return (w.abs() + 1e-8).log()
        return w.log()

    def weight_mask(self):
        """Return a mask indicating the top K synaptic connections per output neuron."""
        # Determine which weights to use for topk selection
        weights_for_topk = self.pre_w
        if self.noise_level > 0:
            # Generate Gaussian noise with mean=0 and std=noise_level
            noise = torch.randn_like(self.pre_w) * self.noise_level
            # Add noise to the raw weights
            weights_for_topk = self.pre_w + noise

        if self.weight_transform == "identity":
            weights_for_topk = weights_for_topk.abs()

        weights_for_topk = self._apply_forbidden_scores(weights_for_topk)

        topK_indices = torch.topk(
            weights_for_topk, self.K, dim=-1, largest=True, sorted=False
        )[1]

        # Create the mask based on the selected indices
        mask = torch.zeros_like(
            self.pre_w,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        mask[torch.arange(self.pre_w.shape[0])[:, None], topK_indices] = 1
        return self._apply_forbidden_mask(mask)

    def pruned_weight(self):
        """Return the pruned synaptic weights after applying the mask."""
        return self._pruned_weight_from_mask(self.weight_mask())

    def log_pruned_weight(self):
        """Return the log of the pruned weights."""
        return ((self.weight_mask() - 1) * 10) + self.log_weight()

    def weighted_synapses(self, cell_weights, prune=False):
        """Return weighted synapses for a given set of cell weights."""
        if prune:
            synapse_weights = self.pruned_weight()
        else:
            synapse_weights = self.weight()

        weighted_synapses = cell_weights[:, None] * synapse_weights
        return weighted_synapses.sum(dim=0)


__all__ = ["TOPK_INIT_METHODS", "TopKLinear"]

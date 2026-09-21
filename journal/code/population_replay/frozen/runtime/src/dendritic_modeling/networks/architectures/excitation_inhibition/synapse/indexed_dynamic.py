"""Candidate-pool indexed dynamic TopK synapses."""

from __future__ import annotations

import logging
from math import log

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _connection_generator,
    _dispatch_indexed_sparse_projection,
    _resolve_index_dtype,
    _sample_indices_from_mask,
    _scale_for_sparse_target,
    _validate_forbidden_indices,
    _workspace_limited_chunk_size,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    normalize_indexed_projection_options,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    validate_connection_mask,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TOPK_INIT_METHODS,
)
from dendritic_modeling.networks.utils.weight_transforms import (
    WeightTransformType,
    apply_weight_transform,
    inverse_softplus,
)

logger = logging.getLogger(__name__)


class IndexedDynamicTopKLinear(nn.Module):
    """
    Candidate-pool TopK layer with low-parameter dynamic synapse selection.

    ``IndexedSparseLinear`` fixes exactly ``K`` synapses per output.  This
    variant fixes a larger candidate pool of size ``candidate_size`` and
    dynamically selects ``K`` active candidates on each forward pass.  It keeps
    the old TopK idea that inactive candidate synapses can later become active,
    while avoiding a full dense ``out_features x in_features`` parameter matrix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int,
        *,
        candidate_size: int | None = None,
        selection: str = "standard",
        param_space: str = "log",
        init_method: str = "xavier_normal",
        init_gain: float = 1.0,
        noise_level: float = 0.0,
        temperature: float = 0.5,
        ultrafast: bool = False,
        weight_transform: WeightTransformType = "exp",
        weight_norm_order: int | None = None,
        gamma: float = 1.0,
        connection_indices: torch.Tensor | None = None,
        connection_mask: torch.Tensor | None = None,
        forbidden_input_index_per_output: torch.Tensor | None = None,
        seed: int | None = None,
        output_chunk_size: int = 2048,
        index_dtype: str | torch.dtype = "int64",
        workspace_mb: float | None = None,
        cache_transformed_weights: bool = False,
        recompute_backward: bool = False,
        projection_backend: str = "eager",
        persistent_indices: bool = True,
        init_mode: str = "per_rank",
        **_: object,
    ):
        super().__init__()

        if not isinstance(K, int):
            K = int(K)
        if K < 1:
            raise ValueError("K must be >= 1")
        if K > in_features:
            raise ValueError(
                f"K must be <= number of input features. "
                f"(K={K}, in_features={in_features})"
            )

        allowed_connection_mask = validate_connection_mask(
            connection_mask,
            int(out_features),
            int(in_features),
        )
        forbidden = _validate_forbidden_indices(
            forbidden_input_index_per_output,
            out_features=int(out_features),
            in_features=int(in_features),
        )

        if candidate_size is None:
            candidate_size = min(in_features, max(K, 4 * K))
            if allowed_connection_mask is not None:
                allowed_count = allowed_connection_mask.sum(dim=1)
                if forbidden is not None:
                    rows = torch.arange(int(out_features))
                    active = forbidden >= 0
                    allowed_count = allowed_count - (
                        active & allowed_connection_mask[rows, forbidden.clamp_min(0)]
                    ).to(allowed_count.dtype)
                candidate_size = min(candidate_size, int(allowed_count.min().item()))
            elif forbidden is not None and bool((forbidden >= 0).any()):
                candidate_size = min(candidate_size, in_features - 1)
        candidate_size = int(candidate_size)
        if candidate_size < K:
            raise ValueError("candidate_size must be >= K")
        if candidate_size > in_features:
            raise ValueError("candidate_size must be <= in_features")
        if output_chunk_size < 1:
            raise ValueError("output_chunk_size must be >= 1")
        if workspace_mb is not None and float(workspace_mb) <= 0:
            raise ValueError("workspace_mb must be > 0 when provided")
        init_mode = str(init_mode).strip().lower()
        if init_mode not in {"per_rank", "rank0_broadcast"}:
            raise ValueError("init_mode must be 'per_rank' or 'rank0_broadcast'")
        if not persistent_indices and connection_indices is None and seed is None:
            raise ValueError(
                "persistent_indices=False requires an explicit seed so the "
                "topology can be regenerated when loading a checkpoint"
            )

        selection = str(selection).lower()
        if selection not in {"standard", "stochastic", "rank_probabilistic"}:
            raise ValueError(
                "selection must be one of: standard, stochastic, rank_probabilistic"
            )

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.K = K
        self.candidate_size = candidate_size
        self.selection = (
            "stochastic" if selection == "rank_probabilistic" else selection
        )
        self.param_space = param_space
        self.init_method = init_method
        self.init_gain = init_gain
        self.noise_level = float(noise_level)
        self.temperature = float(temperature)
        self.ultrafast = bool(ultrafast)
        self.weight_transform = weight_transform
        self.weight_norm_order = weight_norm_order
        self.gamma = float(gamma)
        self.output_chunk_size = int(output_chunk_size)
        self.workspace_mb = None if workspace_mb is None else float(workspace_mb)
        self.index_dtype = _resolve_index_dtype(
            index_dtype,
            in_features=self.in_features,
        )
        self.cache_transformed_weights = bool(cache_transformed_weights)
        normalized_backend = normalize_indexed_projection_options(
            projection_backend,
            recompute_backward=bool(recompute_backward),
        )
        self.projection_backend = normalized_backend
        self.recompute_backward = normalized_backend == "recompute"
        self._last_resolved_projection_backend: str | None = None
        self._last_projection_device: torch.device | None = None
        self.persistent_indices = bool(persistent_indices)
        self.init_mode = init_mode
        self.input_scale_vec = None
        self.param_scale_vec = None
        self._use_forward_dynamic_grad_scaling = False
        self.cache_mask = False
        self._last_forward_weight_mask: torch.Tensor | None = None
        self._last_forward_active_sparse_mask: torch.Tensor | None = None
        self._last_forward_param_tensor: torch.Tensor | None = None
        self._recurrent_weight_cache_enabled = False
        self._recurrent_cached_weight: torch.Tensor | None = None

        topology_initialized = True
        if connection_indices is None:
            rank0_initializes = (
                self.init_mode == "rank0_broadcast"
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            topology_initialized = (
                not rank0_initializes or torch.distributed.get_rank() == 0
            )
            if topology_initialized:
                generator = _connection_generator(seed)
                connection_indices = _sample_indices_from_mask(
                    out_features=self.out_features,
                    in_features=self.in_features,
                    K=self.candidate_size,
                    connection_mask=allowed_connection_mask,
                    forbidden_input_index_per_output=forbidden,
                    generator=generator,
                    row_chunk_size=self.output_chunk_size,
                    index_dtype=self.index_dtype,
                )
            else:
                connection_indices = torch.zeros(
                    self.out_features,
                    self.candidate_size,
                    dtype=self.index_dtype,
                )
        else:
            connection_indices = torch.as_tensor(
                connection_indices, dtype=self.index_dtype
            )
            if tuple(connection_indices.shape) != (
                self.out_features,
                self.candidate_size,
            ):
                raise ValueError(
                    "connection_indices must have shape "
                    f"({self.out_features}, {self.candidate_size}), got "
                    f"{tuple(connection_indices.shape)}"
                )
            if bool((connection_indices < 0).any()) or bool(
                (connection_indices >= self.in_features).any()
            ):
                raise ValueError(
                    f"connection_indices entries must be in [0, {self.in_features - 1}]"
                )
            sorted_indices = connection_indices.sort(dim=1).values
            if bool((sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()):
                raise ValueError("connection_indices must be unique within each row")
            if allowed_connection_mask is not None:
                row = torch.arange(self.out_features)[:, None]
                if not bool(
                    allowed_connection_mask[
                        row, connection_indices.to(torch.long)
                    ].all()
                ):
                    raise ValueError(
                        "connection_indices contains entries disallowed by "
                        "connection_mask"
                    )
            if forbidden is not None:
                violates_forbidden = (forbidden[:, None] >= 0) & (
                    connection_indices.to(torch.long) == forbidden[:, None]
                )
                if bool(violates_forbidden.any()):
                    raise ValueError(
                        "connection_indices contains entries disallowed by "
                        "forbidden_input_index_per_output"
                    )

        if forbidden is None:
            forbidden = torch.empty(0, dtype=torch.long)
        self.register_buffer(
            "_forbidden_input_index_per_output",
            forbidden.contiguous(),
            persistent=False,
        )
        self.register_buffer(
            "connection_indices",
            connection_indices.contiguous(),
            persistent=self.persistent_indices,
        )
        self.register_buffer(
            "_topology_initialized",
            torch.tensor(int(topology_initialized), dtype=torch.uint8),
            persistent=False,
        )
        self._topology_ready = topology_initialized
        self.pre_w = nn.Parameter(torch.empty(self.out_features, self.candidate_size))
        self.initialize()

    def initialize(self) -> None:
        """Initialize all candidate synaptic weights."""
        if self.init_method not in TOPK_INIT_METHODS:
            raise ValueError(
                f"Invalid initialization method: {self.init_method}. "
                f"Choose from {list(TOPK_INIT_METHODS.keys())}"
            )
        TOPK_INIT_METHODS[self.init_method](self.pre_w)
        self.pre_w.data.mul_(self.init_gain)

    def sparse_weight(self) -> torch.Tensor:
        """Return candidate weights with shape ``[out_features, candidate_size]``."""
        return apply_weight_transform(self.pre_w, self.weight_transform)

    def _scores_for_selection(self, *, transformed: bool = False) -> torch.Tensor:
        scores = self.sparse_weight() if transformed else self.pre_w
        if self.weight_transform == "identity":
            scores = scores.abs()
        return scores

    def _standard_candidate_mask(self, scores: torch.Tensor) -> torch.Tensor:
        topk_indices = torch.topk(scores, self.K, dim=-1, largest=True, sorted=False)[1]
        mask = torch.zeros_like(self.pre_w)
        row = torch.arange(self.out_features, device=self.pre_w.device)[:, None]
        mask[row, topk_indices] = 1
        return mask

    def _stochastic_candidate_mask(self) -> torch.Tensor:
        if self.temperature <= 0.0:
            return self._standard_candidate_mask(self._scores_for_selection())

        if self.ultrafast:
            scores = self._scores_for_selection()
            noise_scale = max(0.01, self.temperature) * (1.0 + self.noise_level)
            scores = scores + torch.randn_like(scores) * noise_scale
            return self._standard_candidate_mask(scores)

        scores = self._scores_for_selection(transformed=True)
        if self.noise_level > 0:
            scores = scores + torch.randn_like(scores) * self.noise_level

        _, sort_indices = torch.sort(scores, dim=1, descending=True)
        arange_indices = torch.arange(self.candidate_size, device=scores.device).expand(
            self.out_features, -1
        )
        ranks = torch.zeros_like(scores, dtype=arange_indices.dtype)
        row = torch.arange(self.out_features, device=scores.device)[:, None]
        ranks[row, sort_indices] = arange_indices
        ranks = ranks.to(torch.float)

        temperature = max(0.01, self.temperature)
        probs = torch.sigmoid((self.K - ranks) / (self.K * temperature))
        mask = torch.bernoulli(probs)

        counts = mask.sum(dim=1)
        too_few = counts < (self.K // 2)
        if bool(too_few.any()):
            for row_idx in too_few.nonzero(as_tuple=False).view(-1):
                top_indices = torch.topk(scores[row_idx], self.K, largest=True)[1]
                mask[row_idx, top_indices] = 1

        counts = mask.sum(dim=1)
        too_many = counts > (2 * self.K)
        if bool(too_many.any()):
            for row_idx in too_many.nonzero(as_tuple=False).view(-1):
                active = mask[row_idx].nonzero(as_tuple=True)[0]
                active_scores = scores[row_idx, active]
                _, order = torch.sort(active_scores)
                to_remove = int(counts[row_idx].item() - self.K)
                if to_remove > 0:
                    mask[row_idx, active[order[:to_remove]]] = 0

        counts = mask.sum(dim=1)
        zero = counts == 0
        if bool(zero.any()):
            for row_idx in zero.nonzero(as_tuple=False).view(-1):
                mask[row_idx, torch.argmax(scores[row_idx])] = 1
        return mask

    def active_sparse_mask(self) -> torch.Tensor:
        """Return active candidate mask with shape ``[out_features, candidate_size]``."""
        if self.selection == "stochastic":
            return self._stochastic_candidate_mask()
        scores = self._scores_for_selection()
        if self.noise_level > 0:
            scores = scores + torch.randn_like(scores) * self.noise_level
        return self._standard_candidate_mask(scores)

    def _normalized_sparse_weight(self) -> torch.Tensor:
        active_mask = self.active_sparse_mask()
        self._last_forward_active_sparse_mask = (
            active_mask.detach() if self.cache_mask else None
        )
        active_weight = self.sparse_weight() * active_mask
        if self.weight_norm_order is None:
            return active_weight
        norm = torch.norm(active_weight, p=self.weight_norm_order, dim=-1, keepdim=True)
        safe_norm = norm.clamp_min(torch.finfo(active_weight.dtype).eps)
        normalized = (active_weight / safe_norm) * self.gamma
        return torch.where(norm > 0, normalized, active_weight)

    def set_recurrent_weight_cache(self, enabled: bool) -> None:
        """Cache deterministic candidate selection once per recurrent unroll."""
        self._recurrent_weight_cache_enabled = (
            bool(enabled)
            and self.cache_transformed_weights
            and self.selection == "standard"
            and self.noise_level == 0
            and not self._use_forward_dynamic_grad_scaling
        )
        self._recurrent_cached_weight = None

    def _forward_sparse_weight(self) -> torch.Tensor:
        if not self._recurrent_weight_cache_enabled:
            return self._normalized_sparse_weight()
        if self._recurrent_cached_weight is None:
            self._recurrent_cached_weight = self._normalized_sparse_weight()
        return self._recurrent_cached_weight

    def _forward_output_chunk_size(self, flat_x: torch.Tensor) -> int:
        return _workspace_limited_chunk_size(
            output_chunk_size=self.output_chunk_size,
            workspace_mb=self.workspace_mb,
            out_features=self.out_features,
            flattened_batch_size=flat_x.shape[0],
            synapses_per_output=self.candidate_size,
            element_size=flat_x.element_size(),
        )

    def candidate_mask(self) -> torch.Tensor:
        """Return dense mask of all stored candidate synapses."""
        mask = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        values = torch.ones_like(self.pre_w, dtype=mask.dtype)
        mask.scatter_add_(1, self.connection_indices.to(mask.device), values)
        return mask.clamp_max(1)

    def weight(self) -> torch.Tensor:
        """Return dense candidate weights; non-candidates are zero."""
        dense = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        dense.scatter_add_(
            1, self.connection_indices.to(dense.device), self.sparse_weight()
        )
        return dense

    def log_weight(self) -> torch.Tensor:
        """Return log candidate weights in dense compatibility form."""
        dense = self.weight()
        if self.weight_transform == "identity":
            return (dense.abs() + 1e-8).log()
        return (dense + 1e-8).log()

    def weight_mask(self) -> torch.Tensor:
        """Return dense active TopK mask within the fixed candidate pool."""
        sparse_mask = self.active_sparse_mask()
        dense = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        dense.scatter_add_(1, self.connection_indices.to(dense.device), sparse_mask)
        return dense.clamp_max(1)

    def pruned_weight(self) -> torch.Tensor:
        """Return dense active TopK weights within the fixed candidate pool."""
        dense = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        active_weight = self.sparse_weight() * self.active_sparse_mask()
        dense.scatter_add_(1, self.connection_indices.to(dense.device), active_weight)
        if self.weight_norm_order is not None:
            dense_norm = torch.norm(
                dense, p=self.weight_norm_order, dim=-1, keepdim=True
            )
            safe_norm = dense_norm.clamp_min(torch.finfo(dense.dtype).eps)
            dense = torch.where(dense_norm > 0, (dense / safe_norm) * self.gamma, dense)
        return dense

    def log_pruned_weight(self) -> torch.Tensor:
        """Return log active weights with inactive entries shifted down."""
        return ((self.weight_mask() - 1) * 10) + self.log_weight()

    def decay_weights(self, weight_decay: float = 0.1, weight_boosting: bool = False):
        """Decay active candidates and optionally boost inactive candidates."""
        with torch.no_grad():
            active = self.active_sparse_mask()
            inactive = 1 - active
            if self.weight_transform == "exp":
                self.pre_w.data += log(1 - weight_decay) * active
                if weight_boosting:
                    self.pre_w.data += log(1 + weight_decay) * inactive
            elif self.weight_transform == "softplus":
                weight = nn.functional.softplus(self.pre_w.data)
                decayed = torch.where(active > 0, weight * (1 - weight_decay), weight)
                if weight_boosting:
                    decayed = torch.where(
                        inactive > 0, decayed * (1 + weight_decay), decayed
                    )
                update_mask = active > 0
                if weight_boosting:
                    update_mask = update_mask | (inactive > 0)
                updated_pre_w = self.pre_w.data.clone()
                updated_pre_w[update_mask] = inverse_softplus(decayed[update_mask])
                self.pre_w.data = updated_pre_w
            elif self.weight_transform in {"identity", "relu"}:
                scale = torch.where(active > 0, 1 - weight_decay, 1.0)
                if weight_boosting:
                    scale = torch.where(inactive > 0, scale * (1 + weight_decay), scale)
                self.pre_w.data *= scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dynamic TopK over the fixed candidate pool."""
        if not self._topology_ready:
            if not bool(self._topology_initialized.item()):
                raise RuntimeError(
                    "Indexed topology is awaiting rank-0 broadcast. Construct "
                    "this module under FSDP with sync_module_states=True before "
                    "forward."
                )
            self._topology_ready = True
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")
        squeezed = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeezed = True
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Input feature dimension mismatch: got {x.shape[-1]}, "
                f"expected {self.in_features}"
            )

        original_shape = x.shape
        flat_x = x.reshape(-1, self.in_features)
        active_weight = self._forward_sparse_weight()
        # Frozen weights cannot receive the optional per-forward gradient
        # hook. Releasing their transformed tensor is essential for successive
        # contact-heavy FSDP modules to reuse the same CUDA memory.
        object.__setattr__(
            self,
            "_last_forward_param_tensor",
            active_weight if active_weight.requires_grad else None,
        )
        indices = self.connection_indices.to(flat_x.device)

        output_chunk_size = self._forward_output_chunk_size(flat_x)
        requested_backend = self.projection_backend
        backend_is_resolved = (
            self._last_resolved_projection_backend is not None
            and self._last_projection_device == flat_x.device
        )
        if backend_is_resolved:
            requested_backend = self._last_resolved_projection_backend
        out, resolved_backend = _dispatch_indexed_sparse_projection(
            flat_x=flat_x,
            weight=active_weight,
            connection_indices=indices,
            output_chunk_size=output_chunk_size,
            projection_backend=requested_backend,
            backend_is_resolved=backend_is_resolved,
        )
        if resolved_backend != self._last_resolved_projection_backend:
            logger.info(
                "IndexedDynamicTopKLinear projection backend resolved: "
                "requested=%s, resolved=%s, device=%s",
                self.projection_backend,
                resolved_backend,
                flat_x.device,
            )
            self._last_resolved_projection_backend = resolved_backend
        self._last_projection_device = flat_x.device

        out = out.reshape(*original_shape[:-1], self.out_features)
        if self.cache_mask:
            sparse_mask = self._last_forward_active_sparse_mask
            assert sparse_mask is not None
            dense_mask = torch.zeros(
                self.out_features,
                self.in_features,
                device=sparse_mask.device,
                dtype=sparse_mask.dtype,
            )
            dense_mask.scatter_add_(
                1, self.connection_indices.to(dense_mask.device), sparse_mask
            )
            self._last_forward_weight_mask = dense_mask.clamp_max(1)
        if squeezed:
            return out.squeeze(0)
        return out

    def register_forward_param_gradient_scale(self, param_scale: torch.Tensor) -> None:
        """Scale active sparse-candidate gradients for the most recent forward call."""
        self._use_forward_dynamic_grad_scaling = True
        target = self._last_forward_param_tensor
        if target is None or not target.requires_grad:
            return
        scale = _scale_for_sparse_target(
            param_scale,
            target=target,
            connection_indices=self.connection_indices,
            in_features=self.in_features,
        )
        target.register_hook(lambda grad, scale=scale: grad * scale)


__all__ = ["IndexedDynamicTopKLinear"]

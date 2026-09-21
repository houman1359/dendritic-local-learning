"""Fixed-index sparse linear synapses."""

from __future__ import annotations

import logging
import os
from math import log
from numbers import Integral

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
    sample_structured_indices,
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

# Wide, frozen replacements can otherwise gather tens of GiB during teacher
# calibration. This is a per-tile budget, not a bound on total device memory:
# selected activations, their product, and transformed weights can coexist.
_DEFAULT_FROZEN_WORKSPACE_MB = 256.0


def _workspace_limited_row_chunk_size(
    *,
    flattened_batch_size: int,
    output_chunk_size: int,
    workspace_mb: float | None,
    synapses_per_output: int,
    element_size: int,
) -> int:
    """Cap gathered batch rows so the workspace fits the same byte budget.

    ``_workspace_limited_chunk_size`` shrinks the output chunk assuming the
    full flattened batch is gathered per chunk, but its ``max(1, ...)`` floor
    means that once a single output column exceeds the budget the workspace is
    still ``flat_rows * K * element_size`` bytes — unbounded in the batch.
    This companion cap bounds the other axis: with the output chunk fixed, it
    limits how many batch rows are gathered per tile so that
    ``row_chunk * output_chunk * K * element_size`` respects the budget.
    When no budget is set, or the whole flattened batch already fits it, this
    returns the full batch (a single row chunk).
    """
    rows = max(1, int(flattened_batch_size))
    if workspace_mb is None or flattened_batch_size <= 0:
        return rows
    bytes_per_row = (
        int(output_chunk_size) * int(synapses_per_output) * int(element_size)
    )
    budget_bytes = int(float(workspace_mb) * 1024 * 1024)
    return min(rows, max(1, budget_bytes // max(1, bytes_per_row)))


class IndexedSparseLinear(nn.Module):
    """
    Sparse linear layer with fixed presynaptic indices and trainable weights.

    The forward pass computes

    .. math::

       y_j = \\sum_{k=1}^{K} w_{j,k} x_{i_{j,k}},

    where ``i[j, k]`` is a fixed integer index.  Unlike dynamic TopK, there is
    no dense trainable matrix.  The morphology/topology is fixed at
    construction time; only the synaptic weights are learned.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int,
        *,
        param_space: str = "log",
        init_method: str = "xavier_normal",
        init_gain: float = 1.0,
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
        support_group_rows: int = 1,
        support_col_block: int = 1,
        **_: object,
    ):
        super().__init__()
        # Hardware-structured support (row-group sharing / column blocks).
        # Defaults of 1/1 preserve the historical unstructured sampler and
        # its exact generator stream.
        self.support_group_rows = int(support_group_rows)
        self.support_col_block = int(support_col_block)

        if isinstance(K, bool) or not isinstance(K, Integral):
            raise TypeError("K must be an integer")
        K = int(K)
        if K < 1:
            raise ValueError("K must be >= 1")
        if K > in_features:
            raise ValueError(
                f"K must be <= number of input features. "
                f"(K={K}, in_features={in_features})"
            )
        if isinstance(output_chunk_size, bool) or not isinstance(
            output_chunk_size, Integral
        ):
            raise TypeError("output_chunk_size must be an integer")
        output_chunk_size = int(output_chunk_size)
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

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.K = K
        self.param_space = param_space
        self.init_method = init_method
        self.init_gain = init_gain
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
        # Retained as a compatibility attribute for code that predates the
        # single backend selector.
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
        self._last_forward_param_tensor: torch.Tensor | None = None
        self._recurrent_weight_cache_enabled = False
        self._recurrent_cached_weight: torch.Tensor | None = None
        # Inference-frozen fold of the transformed weights. Non-persistent so
        # checkpoints keep only the raw ``pre_w`` parameterization; rebuilt on
        # demand by ``freeze_for_inference`` and dropped whenever the module
        # returns to training or loads new weights.
        self.register_buffer("_inference_folded_weight", None, persistent=False)

        allowed_connection_mask = validate_connection_mask(
            connection_mask,
            self.out_features,
            self.in_features,
        )
        forbidden = _validate_forbidden_indices(
            forbidden_input_index_per_output,
            out_features=self.out_features,
            in_features=self.in_features,
        )

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
                structured = self.support_group_rows > 1 or self.support_col_block > 1
                if structured and (
                    allowed_connection_mask is not None or forbidden is not None
                ):
                    raise ValueError(
                        "Structured support (support_group_rows/"
                        "support_col_block > 1) does not compose with "
                        "connection_mask or forbidden indices yet."
                    )
                if structured:
                    connection_indices = sample_structured_indices(
                        out_features=self.out_features,
                        in_features=self.in_features,
                        K=self.K,
                        generator=generator,
                        index_dtype=self.index_dtype,
                        support_group_rows=self.support_group_rows,
                        support_col_block=self.support_col_block,
                    )
                else:
                    connection_indices = _sample_indices_from_mask(
                        out_features=self.out_features,
                        in_features=self.in_features,
                        K=self.K,
                        connection_mask=allowed_connection_mask,
                        forbidden_input_index_per_output=forbidden,
                        generator=generator,
                        row_chunk_size=self.output_chunk_size,
                        index_dtype=self.index_dtype,
                    )
            else:
                # FSDP ``sync_module_states=True`` broadcasts this buffer and
                # the initialization marker from rank 0 before first use.
                connection_indices = torch.zeros(
                    self.out_features,
                    self.K,
                    dtype=self.index_dtype,
                )
        else:
            connection_indices = torch.as_tensor(
                connection_indices, dtype=self.index_dtype
            )
            if tuple(connection_indices.shape) != (self.out_features, self.K):
                raise ValueError(
                    "connection_indices must have shape "
                    f"({self.out_features}, {self.K}), got "
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

        if allowed_connection_mask is None:
            self.register_buffer(
                "_allowed_connection_mask",
                torch.empty(0, dtype=torch.bool),
                persistent=False,
            )
        else:
            self.register_buffer(
                "_allowed_connection_mask",
                allowed_connection_mask.contiguous(),
                persistent=False,
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
        self.pre_w = nn.Parameter(torch.empty(self.out_features, self.K))
        self.initialize()

    def initialize(self) -> None:
        """Initialize the sparse synaptic weights."""
        if self.init_method not in TOPK_INIT_METHODS:
            raise ValueError(
                f"Invalid initialization method: {self.init_method}. "
                f"Choose from {list(TOPK_INIT_METHODS.keys())}"
            )
        TOPK_INIT_METHODS[self.init_method](self.pre_w)
        self.pre_w.data.mul_(self.init_gain)

    def sparse_weight(self) -> torch.Tensor:
        """Return the trainable sparse weights with shape ``[out_features, K]``."""
        return apply_weight_transform(self.pre_w, self.weight_transform)

    def connectivity_resource_counts(self) -> dict[str, int | float | str]:
        """Return exact connectivity counts without materializing a dense matrix.

        Resource accounting is called after training, including for layers whose
        dense compatibility view can be hundreds of GiB.  Fixed indexed layers
        already store every realized contact explicitly, so their counts should
        be obtained from the compact indices.
        """

        allowed = self._allowed_connection_mask
        forbidden = self._forbidden_input_index_per_output
        if allowed.numel() > 0:
            candidate_slots = int(allowed.count_nonzero().item())
            if forbidden.numel() > 0:
                valid = forbidden >= 0
                if bool(valid.any()):
                    rows = torch.arange(
                        self.out_features,
                        device=forbidden.device,
                    )
                    candidate_slots -= int(
                        allowed[
                            rows[valid],
                            forbidden[valid].to(torch.long),
                        ]
                        .count_nonzero()
                        .item()
                    )
        else:
            candidate_slots = self.out_features * self.in_features
            if forbidden.numel() > 0:
                candidate_slots -= int((forbidden >= 0).count_nonzero().item())

        active_synapses = int(self.connection_indices.numel())
        return {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "candidate_slots": candidate_slots,
            "active_synapses": active_synapses,
            "realized_k_min": self.K,
            "realized_k_max": self.K,
            "realized_k_mean": float(self.K),
            "selection_policy": "indexed_fixed",
            "mask_source": "fixed_indices",
        }

    def local_weight_gradient(
        self,
        factor: torch.Tensor,
        x_pre: torch.Tensor,
        *,
        normalize_by_batch: bool,
    ) -> torch.Tensor:
        """Compute a compact local conductance gradient on the fixed support.

        A dense local rule forms ``factor.T @ x_pre``, whose shape is
        ``[out_features, in_features]``.  For fixed indexed connectivity only
        the ``K`` stored contacts per output can be updated.  This method
        evaluates exactly those entries in bounded output chunks and returns a
        tensor matching ``pre_w``.
        """

        if factor.ndim != 2 or x_pre.ndim != 2:
            raise ValueError(
                "Indexed local gradients require two-dimensional factor and "
                f"input tensors, got {tuple(factor.shape)} and {tuple(x_pre.shape)}"
            )
        if factor.shape[0] != x_pre.shape[0]:
            raise ValueError(
                "Indexed local-gradient batch dimensions differ: "
                f"{factor.shape[0]} vs {x_pre.shape[0]}"
            )
        if factor.shape[1] != self.out_features:
            raise ValueError(
                "Indexed local-gradient factor width does not match out_features: "
                f"{factor.shape[1]} vs {self.out_features}"
            )
        if x_pre.shape[1] != self.in_features:
            raise ValueError(
                "Indexed local-gradient input width does not match in_features: "
                f"{x_pre.shape[1]} vs {self.in_features}"
            )
        if factor.device != self.pre_w.device or x_pre.device != self.pre_w.device:
            raise ValueError(
                "Indexed local-gradient tensors must share the layer device"
            )

        factor = factor.detach()
        x_pre = x_pre.detach()
        gradient = torch.empty_like(self.pre_w)
        output_chunk_size = self._forward_output_chunk_size(x_pre)
        batch_scale = 1.0 / factor.shape[0] if normalize_by_batch else 1.0

        with torch.no_grad():
            for start in range(0, self.out_features, output_chunk_size):
                end = min(start + output_chunk_size, self.out_features)
                indices = self.connection_indices[start:end].to(torch.long)
                selected = x_pre.index_select(1, indices.reshape(-1)).reshape(
                    x_pre.shape[0],
                    end - start,
                    self.K,
                )
                factor_chunk = factor[:, start:end]
                if selected.dtype != factor_chunk.dtype:
                    selected = selected.to(factor_chunk.dtype)
                chunk_gradient = torch.einsum(
                    "bo,bok->ok",
                    factor_chunk,
                    selected,
                )
                if normalize_by_batch:
                    chunk_gradient.mul_(batch_scale)
                gradient[start:end].copy_(
                    chunk_gradient.to(device=gradient.device, dtype=gradient.dtype)
                )
        return gradient

    def _normalized_sparse_weight(self) -> torch.Tensor:
        weight = self.sparse_weight()
        if self.weight_norm_order is None:
            return weight
        norm = torch.norm(weight, p=self.weight_norm_order, dim=-1, keepdim=True)
        safe_norm = norm.clamp_min(torch.finfo(weight.dtype).eps)
        normalized = (weight / safe_norm) * self.gamma
        return torch.where(norm > 0, normalized, weight)

    def set_recurrent_weight_cache(self, enabled: bool) -> None:
        """Cache the transformed fixed weights once per recurrent unroll."""
        self._recurrent_weight_cache_enabled = (
            bool(enabled)
            and self.cache_transformed_weights
            and not self._use_forward_dynamic_grad_scaling
        )
        self._recurrent_cached_weight = None

    def _forward_sparse_weight(self) -> torch.Tensor:
        if not self._recurrent_weight_cache_enabled:
            return self._normalized_sparse_weight()
        if self._recurrent_cached_weight is None:
            self._recurrent_cached_weight = self._normalized_sparse_weight()
        return self._recurrent_cached_weight

    def freeze_for_inference(self) -> None:
        """Fold the transformed weights into a reusable inference buffer.

        The softplus/exp weight transform (plus any weight normalization and
        gamma scaling) is otherwise recomputed on every forward even though the
        weights are constant at inference. Folding computes it once; forward
        then feeds the folded tensor straight into the standard projection
        dispatch, so the tuned CUDA/Triton kernels remain in use. The fold is
        dropped automatically by ``train(True)`` or by loading new weights, and
        it is never serialized (the checkpoint representation stays raw).
        """
        self.pre_w.requires_grad_(False)
        with torch.no_grad():
            self._inference_folded_weight = (
                self._normalized_sparse_weight().detach().contiguous()
            )
        self.eval()

    def unfreeze_inference_fold(self) -> None:
        """Drop the folded inference weights (weights may change again)."""
        self._inference_folded_weight = None

    def train(self, mode: bool = True):
        if mode:
            self.unfreeze_inference_fold()
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self.unfreeze_inference_fold()
        return super()._load_from_state_dict(*args, **kwargs)

    def _forward_output_chunk_size(self, flat_x: torch.Tensor) -> int:
        return _workspace_limited_chunk_size(
            output_chunk_size=self.output_chunk_size,
            workspace_mb=self.workspace_mb,
            out_features=self.out_features,
            flattened_batch_size=flat_x.shape[0],
            synapses_per_output=self.K,
            element_size=flat_x.element_size(),
        )

    def _can_use_frozen_chunked_transform(self, flat_x: torch.Tensor) -> bool:
        """Return whether forward can transform each projection chunk in place."""
        if (
            os.environ.get("DENDRITIC_FROZEN_TRITON", "0") == "1"
            and self.weight_transform == "identity"
            and flat_x.is_cuda
        ):
            # Opt-in fast path (2026-08-20): for identity-transform frozen
            # cells the transformed weight IS pre_w (nothing to materialize),
            # so the chunked in-place path only costs speed. Fall through to
            # the normal dispatch, which resolves the Triton kernels — the
            # dominant step cost of late-span composed forwards drops with
            # it. Values differ from the chunked path only at fp32
            # reassociation level (kernels verified <= 3.1e-7 vs reference).
            return False
        return (
            (
                not self.pre_w.requires_grad
                or bool(getattr(self, "_fsdp_frozen_core", False))
            )
            and not flat_x.requires_grad
            and self.weight_norm_order is None
            and not self._recurrent_weight_cache_enabled
        )

    def _forward_frozen_chunked_transform(
        self,
        flat_x: torch.Tensor,
        connection_indices: torch.Tensor,
        output_chunk_size: int,
    ) -> torch.Tensor:
        """Project frozen weights without materializing one full transformed copy.

        Both axes of the gather workspace are bounded. Frozen inference uses
        a conservative default when ``workspace_mb`` is unset. The output and
        row chunks use the same budget, so each gathered tile is at most
        ``row_chunk x output_chunk x K`` elements even when a single output
        column over the full batch would exceed the budget (the regime where
        the output cap alone bottoms out at one column but still gathers
        every row). Tiling is bitwise inert: every ``out[i, j]`` is produced
        by exactly one tile as the same K-length reduction over the same
        operand values in the same order, and when the whole flattened batch
        fits the budget the row loop degenerates to a
        single full-batch chunk — the previous behavior, kernel for kernel.
        """
        rows = flat_x.shape[0]
        out = flat_x.new_empty(rows, self.out_features)
        workspace_mb = (
            self.workspace_mb
            if self.workspace_mb is not None
            else _DEFAULT_FROZEN_WORKSPACE_MB
        )
        # Multiplication may promote a lower-precision activation to the
        # weight dtype, so bound the larger of the gather and product tiles.
        element_size = max(flat_x.element_size(), self.pre_w.element_size())
        output_chunk_size = _workspace_limited_chunk_size(
            output_chunk_size=output_chunk_size,
            workspace_mb=workspace_mb,
            out_features=self.out_features,
            flattened_batch_size=rows,
            synapses_per_output=self.K,
            element_size=element_size,
        )
        row_chunk_size = _workspace_limited_row_chunk_size(
            flattened_batch_size=rows,
            output_chunk_size=min(int(output_chunk_size), self.out_features),
            workspace_mb=workspace_mb,
            synapses_per_output=self.K,
            element_size=element_size,
        )
        for row_start in range(0, rows, row_chunk_size):
            row_end = min(row_start + row_chunk_size, rows)
            x_rows = flat_x[row_start:row_end]
            for start in range(0, self.out_features, output_chunk_size):
                end = min(start + output_chunk_size, self.out_features)
                idx_chunk = connection_indices[start:end].to(torch.long)
                selected = x_rows[:, idx_chunk.reshape(-1)].reshape(
                    row_end - row_start,
                    end - start,
                    self.K,
                )
                weight_chunk = apply_weight_transform(
                    self.pre_w[start:end],
                    self.weight_transform,
                )
                out[row_start:row_end, start:end] = (
                    selected * weight_chunk.unsqueeze(0)
                ).sum(dim=-1)
        return out

    def weight(self) -> torch.Tensor:
        """Return a dense compatibility view of the effective weights."""
        return self.dense_weight()

    def dense_weight(self) -> torch.Tensor:
        """Materialize a dense ``[out_features, in_features]`` weight matrix."""
        dense = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        dense.scatter_add_(
            1, self.connection_indices.to(dense.device), self.sparse_weight()
        )
        if self.weight_norm_order is not None:
            dense_norm = torch.norm(
                dense, p=self.weight_norm_order, dim=-1, keepdim=True
            )
            safe_norm = dense_norm.clamp_min(torch.finfo(dense.dtype).eps)
            dense = torch.where(dense_norm > 0, (dense / safe_norm) * self.gamma, dense)
        return dense

    def log_weight(self) -> torch.Tensor:
        """Return a dense log-weight matrix for compatibility with analyzers."""
        dense = self.dense_weight()
        if self.weight_transform == "identity":
            return (dense.abs() + 1e-8).log()
        return (dense + 1e-8).log()

    def weight_mask(self) -> torch.Tensor:
        """Return a dense binary mask of active fixed synapses."""
        mask = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        values = torch.ones_like(self.pre_w, dtype=mask.dtype)
        mask.scatter_add_(1, self.connection_indices.to(mask.device), values)
        return mask.clamp_max(1)

    def pruned_weight(self) -> torch.Tensor:
        """Return the dense effective sparse matrix."""
        return self.dense_weight()

    def log_pruned_weight(self) -> torch.Tensor:
        """Return log effective weights with inactive entries shifted down."""
        return ((self.weight_mask() - 1) * 10) + self.log_weight()

    def decay_weights(self, weight_decay: float = 0.1, weight_boosting: bool = False):
        """Decay active sparse weights in-place."""
        with torch.no_grad():
            if self.weight_transform == "exp":
                self.pre_w.data += log(1 - weight_decay)
                if weight_boosting:
                    self.pre_w.data += log(1 + weight_decay)
            elif self.weight_transform == "softplus":
                weight = nn.functional.softplus(self.pre_w.data)
                scale = 1 - weight_decay
                if weight_boosting:
                    scale *= 1 + weight_decay
                updated = weight * scale
                self.pre_w.data = inverse_softplus(updated)
            elif self.weight_transform in {"identity", "relu"}:
                scale = 1 - weight_decay
                if weight_boosting:
                    scale *= 1 + weight_decay
                self.pre_w.data *= scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the fixed sparse projection along the last input dimension."""
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
        indices = self.connection_indices.to(flat_x.device)
        output_chunk_size = self._forward_output_chunk_size(flat_x)

        folded_weight = None if self.training else self._inference_folded_weight
        if folded_weight is None and self._can_use_frozen_chunked_transform(flat_x):
            out = self._forward_frozen_chunked_transform(
                flat_x,
                indices,
                output_chunk_size,
            )
            object.__setattr__(self, "_last_forward_param_tensor", None)
            out = out.reshape(*original_shape[:-1], self.out_features)
            if self.cache_mask:
                self._last_forward_weight_mask = self.weight_mask().detach()
            if squeezed:
                return out.squeeze(0)
            return out

        if folded_weight is not None:
            # Inference fold: transform already applied once by
            # freeze_for_inference; reuse the tensor and keep the fast
            # projection dispatch below.
            weight = folded_weight
            object.__setattr__(self, "_last_forward_param_tensor", None)
        else:
            weight = self._forward_sparse_weight()
            # ``weight`` can be the Parameter itself for the identity transform.
            # Bypass ``nn.Module.__setattr__`` so this diagnostic cache is not
            # registered as a duplicate parameter/state-dict key. Frozen weights
            # cannot receive the optional gradient hook, so do not retain their
            # potentially multi-gigabyte transformed tensors after this forward.
            object.__setattr__(
                self,
                "_last_forward_param_tensor",
                weight if weight.requires_grad else None,
            )
        requested_backend = self.projection_backend
        backend_is_resolved = (
            self._last_resolved_projection_backend is not None
            and self._last_projection_device == flat_x.device
        )
        if backend_is_resolved:
            requested_backend = self._last_resolved_projection_backend
        out, resolved_backend = _dispatch_indexed_sparse_projection(
            flat_x=flat_x,
            weight=weight,
            connection_indices=indices,
            output_chunk_size=output_chunk_size,
            projection_backend=requested_backend,
            backend_is_resolved=backend_is_resolved,
        )
        if resolved_backend != self._last_resolved_projection_backend:
            logger.info(
                "IndexedSparseLinear projection backend resolved: requested=%s, "
                "resolved=%s, device=%s",
                self.projection_backend,
                resolved_backend,
                flat_x.device,
            )
            self._last_resolved_projection_backend = resolved_backend
        self._last_projection_device = flat_x.device

        out = out.reshape(*original_shape[:-1], self.out_features)
        if self.cache_mask:
            self._last_forward_weight_mask = self.weight_mask().detach()
        if squeezed:
            return out.squeeze(0)
        return out

    def register_forward_param_gradient_scale(self, param_scale: torch.Tensor) -> None:
        """Scale sparse parameter gradients for the most recent forward call."""
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


__all__ = ["IndexedSparseLinear"]

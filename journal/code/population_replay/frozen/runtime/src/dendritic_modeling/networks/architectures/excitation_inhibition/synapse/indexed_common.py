"""Shared utilities for indexed sparse synapse layers."""

from __future__ import annotations

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    validate_connection_mask,
)


def _resolve_index_dtype(
    index_dtype: str | torch.dtype,
    *,
    in_features: int,
) -> torch.dtype:
    """Resolve a compact, valid dtype for stored presynaptic indices."""
    if isinstance(index_dtype, torch.dtype):
        resolved = index_dtype
    else:
        normalized = str(index_dtype).strip().lower()
        aliases = {
            "auto": "auto",
            "int32": torch.int32,
            "torch.int32": torch.int32,
            "int64": torch.int64,
            "long": torch.int64,
            "torch.int64": torch.int64,
        }
        if normalized not in aliases:
            raise ValueError("index_dtype must be one of: auto, int32, int64")
        value = aliases[normalized]
        if value == "auto":
            resolved = (
                torch.int32
                if in_features <= torch.iinfo(torch.int32).max
                else torch.int64
            )
        else:
            resolved = value

    if resolved not in {torch.int32, torch.int64}:
        raise ValueError("index_dtype must resolve to torch.int32 or torch.int64")
    if resolved == torch.int32 and in_features > torch.iinfo(torch.int32).max:
        raise ValueError(
            "int32 connection indices require in_features <= "
            f"{torch.iinfo(torch.int32).max}, got {in_features}"
        )
    return resolved


def _validate_forbidden_indices(
    forbidden_input_index_per_output: torch.Tensor | None,
    *,
    out_features: int,
    in_features: int,
) -> torch.Tensor | None:
    """Validate and normalize one optional forbidden input index per row."""
    if forbidden_input_index_per_output is None:
        return None

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
    return forbidden


def _sample_indices_from_mask(
    *,
    out_features: int,
    in_features: int,
    K: int,
    connection_mask: torch.Tensor | None,
    generator: torch.Generator | None,
    row_chunk_size: int = 2048,
    forbidden_input_index_per_output: torch.Tensor | None = None,
    index_dtype: str | torch.dtype = torch.int64,
) -> torch.Tensor:
    """Sample exactly ``K`` unique allowed input indices for each output."""
    allowed = validate_connection_mask(connection_mask, out_features, in_features)
    forbidden = _validate_forbidden_indices(
        forbidden_input_index_per_output,
        out_features=out_features,
        in_features=in_features,
    )
    resolved_dtype = _resolve_index_dtype(index_dtype, in_features=in_features)
    indices = torch.empty(out_features, K, dtype=resolved_dtype)

    # The common unmasked sparse case must not construct one full randperm of
    # all inputs for every output row. At large neuronal scales that makes
    # initialization proportional to out_features * in_features even though
    # only out_features * K indices are retained. Rejection sampling is exact
    # and efficient while K is sparse relative to in_features.
    if allowed is None and K * 4 <= in_features:
        if forbidden is not None and bool(
            (in_features - (forbidden >= 0).to(torch.long) < K).any()
        ):
            raise ValueError(
                "Indexed sparse sampling requires at least K allowed inputs "
                "per output row after applying forbidden indices."
            )
        max_chunk_entries = 8_000_000
        rows_per_chunk = max(
            1,
            min(
                int(row_chunk_size),
                max_chunk_entries // max(K, 1),
            ),
        )
        for start in range(0, out_features, rows_per_chunk):
            end = min(start + rows_per_chunk, out_features)
            values = torch.randint(
                in_features,
                (end - start, K),
                dtype=resolved_dtype,
                generator=generator,
            )
            for _ in range(128):
                sorted_values, order = values.sort(dim=1)
                duplicate_sorted = torch.zeros_like(values, dtype=torch.bool)
                duplicate_sorted[:, 1:] = sorted_values[:, 1:] == sorted_values[:, :-1]
                invalid = torch.zeros_like(duplicate_sorted)
                invalid.scatter_(1, order, duplicate_sorted)
                if forbidden is not None:
                    chunk_forbidden = forbidden[start:end].to(dtype=resolved_dtype)
                    invalid |= (chunk_forbidden[:, None] >= 0) & (
                        values == chunk_forbidden[:, None]
                    )
                if not bool(invalid.any()):
                    break
                invalid_count = int(invalid.sum().item())
                values[invalid] = torch.randint(
                    in_features,
                    (invalid_count,),
                    dtype=resolved_dtype,
                    generator=generator,
                )
            else:
                raise RuntimeError(
                    "Unable to sample unique indexed connections after 128 "
                    "rejection rounds."
                )
            indices[start:end] = values
        return indices

    full_arange = (
        torch.arange(in_features, dtype=torch.long) if allowed is None else None
    )

    # Masked sampling intentionally keeps this seeded per-row randperm contract.
    # A rejection sampler is also mathematically valid, but consumes the RNG
    # stream differently and therefore changes the graph generated by an
    # existing seed. That would break topology-regenerated checkpoints whose
    # indices are non-persistent. A future sampling algorithm must use an
    # explicit topology version rather than silently changing this path.
    for out_idx in range(out_features):
        if allowed is None:
            row_allowed = full_arange
        else:
            row_allowed = torch.nonzero(allowed[out_idx], as_tuple=False).flatten()
        if forbidden is not None and forbidden[out_idx] >= 0:
            row_allowed = row_allowed[row_allowed != forbidden[out_idx]]

        if row_allowed.numel() < K:
            raise ValueError(
                "IndexedSparseLinear requires at least K allowed inputs per "
                f"output row; row {out_idx} has {row_allowed.numel()} allowed "
                f"inputs but K={K}."
            )

        perm = torch.randperm(row_allowed.numel(), generator=generator)[:K]
        indices[out_idx] = row_allowed[perm].to(dtype=resolved_dtype)

    return indices


def sample_structured_indices(
    *,
    out_features: int,
    in_features: int,
    K: int,
    generator: torch.Generator,
    index_dtype: str | torch.dtype = "int64",
    support_group_rows: int = 1,
    support_col_block: int = 1,
) -> torch.Tensor:
    """Sample hardware-structured fixed support (row groups x column blocks).

    Two orthogonal constraints make the resulting sparse topology tile-shaped
    for block-sparse execution (SCALING_PLATFORM_ROADMAP.md, WS3):

    - ``support_group_rows``: consecutive output rows in groups of this size
      share one sampled column support (their weights stay independent). Each
      (group_rows x K) tile is then dense, and index storage shrinks by the
      same factor.
    - ``support_col_block``: sampled columns arrive as contiguous runs of this
      length, so gathers are coalesced and tiles align to fixed column blocks.

    ``support_group_rows=1, support_col_block=1`` reproduces unstructured
    sampling semantics but through this dedicated path; callers keep the
    historical sampler when no structure is requested. Fails closed when K or
    ``in_features`` is not divisible by ``support_col_block``.
    """

    group_rows = int(support_group_rows)
    col_block = int(support_col_block)
    if group_rows < 1 or col_block < 1:
        raise ValueError(
            "support_group_rows and support_col_block must be >= 1, got "
            f"{support_group_rows} and {support_col_block}"
        )
    if K % col_block:
        raise ValueError(f"K={K} must be divisible by support_col_block={col_block}")
    if in_features % col_block:
        raise ValueError(
            f"in_features={in_features} must be divisible by "
            f"support_col_block={col_block}"
        )
    blocks_total = in_features // col_block
    blocks_per_row = K // col_block
    if blocks_per_row > blocks_total:
        raise ValueError(
            f"K={K} with support_col_block={col_block} needs "
            f"{blocks_per_row} column blocks but only {blocks_total} exist"
        )

    resolved_dtype = _resolve_index_dtype(index_dtype, in_features=in_features)
    n_groups = (out_features + group_rows - 1) // group_rows
    offsets = torch.arange(col_block, dtype=torch.long)
    indices = torch.empty(out_features, K, dtype=resolved_dtype)
    for group in range(n_groups):
        chosen_blocks = torch.randperm(blocks_total, generator=generator)[
            :blocks_per_row
        ]
        columns = (chosen_blocks[:, None] * col_block + offsets[None, :]).reshape(-1)
        row_start = group * group_rows
        row_end = min(row_start + group_rows, out_features)
        indices[row_start:row_end] = columns.to(resolved_dtype)
    return indices


def _workspace_limited_chunk_size(
    *,
    output_chunk_size: int,
    workspace_mb: float | None,
    out_features: int,
    flattened_batch_size: int,
    synapses_per_output: int,
    element_size: int,
) -> int:
    """Cap output rows so the gathered-input workspace fits a byte budget."""
    chunk_size = min(int(output_chunk_size), int(out_features))
    if workspace_mb is None or flattened_batch_size == 0:
        return chunk_size
    bytes_per_output = (
        int(flattened_batch_size) * int(synapses_per_output) * int(element_size)
    )
    budget_bytes = int(float(workspace_mb) * 1024 * 1024)
    return min(chunk_size, max(1, budget_bytes // max(1, bytes_per_output)))


class _IndexedSparseProjectionFunction(torch.autograd.Function):
    """Indexed projection whose backward re-gathers instead of saving gathers."""

    @staticmethod
    def forward(
        ctx,
        flat_x: torch.Tensor,
        weight: torch.Tensor,
        connection_indices: torch.Tensor,
        output_chunk_size: int,
    ) -> torch.Tensor:
        ctx.output_chunk_size = int(output_chunk_size)
        ctx.save_for_backward(flat_x, weight, connection_indices)
        out_features, synapses_per_output = weight.shape
        out = flat_x.new_empty(flat_x.shape[0], out_features)

        for start in range(0, out_features, ctx.output_chunk_size):
            end = min(start + ctx.output_chunk_size, out_features)
            idx_chunk = connection_indices[start:end]
            selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
                flat_x.shape[0],
                end - start,
                synapses_per_output,
            )
            out[:, start:end] = (selected * weight[start:end].unsqueeze(0)).sum(dim=-1)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        flat_x, weight, connection_indices = ctx.saved_tensors
        out_features, synapses_per_output = weight.shape
        accumulation_dtype = (
            torch.float32
            if flat_x.dtype in {torch.bfloat16, torch.float16}
            else flat_x.dtype
        )
        grad_x = (
            torch.zeros_like(flat_x, dtype=accumulation_dtype)
            if ctx.needs_input_grad[0]
            else None
        )
        grad_weight = torch.empty_like(weight) if ctx.needs_input_grad[1] else None

        for start in range(0, out_features, ctx.output_chunk_size):
            end = min(start + ctx.output_chunk_size, out_features)
            idx_chunk = connection_indices[start:end]
            grad_chunk = grad_output[:, start:end]

            if grad_weight is not None:
                selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
                    flat_x.shape[0],
                    end - start,
                    synapses_per_output,
                )
                grad_weight[start:end] = (selected * grad_chunk.unsqueeze(-1)).sum(
                    dim=0
                )

            if grad_x is not None:
                contributions = grad_chunk.to(accumulation_dtype).unsqueeze(
                    -1
                ) * weight[start:end].to(accumulation_dtype).unsqueeze(0)
                grad_x.index_add_(
                    1,
                    idx_chunk.reshape(-1),
                    contributions.reshape(flat_x.shape[0], -1),
                )

        if grad_x is not None and grad_x.dtype != flat_x.dtype:
            grad_x = grad_x.to(flat_x.dtype)
        return grad_x, grad_weight, None, None


def _indexed_sparse_projection(
    *,
    flat_x: torch.Tensor,
    weight: torch.Tensor,
    connection_indices: torch.Tensor,
    output_chunk_size: int,
    recompute_backward: bool,
) -> torch.Tensor:
    """Apply an indexed projection with optional gather recomputation."""
    if recompute_backward and torch.is_grad_enabled():
        return _IndexedSparseProjectionFunction.apply(
            flat_x,
            weight,
            connection_indices,
            output_chunk_size,
        )

    out_features, synapses_per_output = weight.shape
    out = flat_x.new_empty(flat_x.shape[0], out_features)
    for start in range(0, out_features, output_chunk_size):
        end = min(start + output_chunk_size, out_features)
        idx_chunk = connection_indices[start:end]
        selected = flat_x[:, idx_chunk.reshape(-1)].reshape(
            flat_x.shape[0],
            end - start,
            synapses_per_output,
        )
        out[:, start:end] = (selected * weight[start:end].unsqueeze(0)).sum(dim=-1)
    return out


def _dispatch_indexed_sparse_projection(
    *,
    flat_x: torch.Tensor,
    weight: torch.Tensor,
    connection_indices: torch.Tensor,
    output_chunk_size: int,
    projection_backend: str,
    backend_is_resolved: bool = False,
) -> tuple[torch.Tensor, str]:
    """Apply an indexed projection through one validated execution backend."""
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
        normalize_indexed_projection_backend,
        resolve_indexed_projection_backend,
    )

    if backend_is_resolved:
        resolved = normalize_indexed_projection_backend(projection_backend)
        if resolved == "auto":
            raise ValueError("a resolved indexed projection backend cannot be 'auto'")
    else:
        resolved = resolve_indexed_projection_backend(
            projection_backend,
            device=flat_x.device,
            in_features=flat_x.shape[1],
        )
    if resolved == "triton_transposed":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather_transposed import (
            triton_sparse_gather,
        )

        output = triton_sparse_gather(
            flat_x,
            connection_indices,
            weight,
            chunk_size=output_chunk_size,
        )
    elif resolved == "triton_fused":
        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_fused_chunked import (
            triton_fused_sparse_gather,
        )

        # Fused chunked kernel (kernel-search round 1, 2026-08-18): one
        # launch per pass sharing the idx/w reads and the x gather across
        # forward / grad_w / grad_x. auto resolves here for giant input
        # widths, where it beats chunked recompute 3.4x (H100) / 8.5x
        # (Blackwell).
        output = triton_fused_sparse_gather(
            flat_x,
            connection_indices,
            weight,
            chunk_size=output_chunk_size,
        )
    elif resolved == "triton_ell":
        from dendritic_modeling.kernels.triton_ell import ell_gather_gemm_train

        # Fused ELL gather-GEMM with the feature-major activation layout
        # (coalesced gathers; measured 7-9x over eager at bench shapes).
        # ``weight`` is already transformed by the caller, so the kernel
        # runs with the identity transform; gradients flow through the
        # transformed tensor into pre_w as with every other backend. The
        # per-call transpose is one small copy of [M, D] and is amortized
        # against the eliminated [M, O, K] traffic; sharing the transposed
        # activations across E/I banks is a follow-up optimization.
        flat_x_fm = flat_x.t().contiguous().t()
        output = ell_gather_gemm_train(
            flat_x_fm,
            connection_indices,
            weight,
            transform="identity",
        ).to(flat_x.dtype)
    else:
        output = _indexed_sparse_projection(
            flat_x=flat_x,
            weight=weight,
            connection_indices=connection_indices,
            output_chunk_size=output_chunk_size,
            recompute_backward=resolved == "recompute",
        )
    return output, resolved


def _connection_generator(seed: int | None) -> torch.Generator:
    """Create a CPU generator; seed=None follows the process/global RNG state."""
    if seed is None:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    return generator


def _scale_for_sparse_target(
    param_scale: torch.Tensor,
    *,
    target: torch.Tensor,
    connection_indices: torch.Tensor,
    in_features: int,
) -> torch.Tensor:
    """Broadcast or gather a gradient scale onto a sparse [out, K] tensor."""
    scale = param_scale.detach().to(device=target.device, dtype=target.dtype)
    if scale.ndim == 1:
        scale = scale[:, None]
    if scale.shape == target.shape:
        return scale
    if scale.ndim == 2 and scale.shape[1] == in_features:
        row = torch.arange(target.shape[0], device=target.device)[:, None]
        return scale[row, connection_indices.to(target.device)]
    return scale


def _allowed_mask_with_forbidden(
    *,
    connection_mask: torch.Tensor | None,
    forbidden_input_index_per_output: torch.Tensor | None,
    out_features: int,
    in_features: int,
) -> torch.Tensor | None:
    """Merge hard connectivity masks with per-row forbidden input indices."""
    allowed = validate_connection_mask(connection_mask, out_features, in_features)
    forbidden = _validate_forbidden_indices(
        forbidden_input_index_per_output,
        out_features=out_features,
        in_features=in_features,
    )
    if forbidden is None:
        return allowed

    if allowed is None:
        allowed = torch.ones(out_features, in_features, dtype=torch.bool)
    else:
        allowed = allowed.clone()
    rows = torch.arange(out_features, dtype=torch.long)
    active = forbidden >= 0
    if bool(active.any()):
        allowed[rows[active], forbidden[active]] = False
    if allowed.numel() > 0 and bool((allowed.sum(dim=1) == 0).any()):
        raise ValueError(
            "connection_mask and forbidden_input_index_per_output leave at "
            "least one output row with no allowed inputs"
        )
    return allowed


__all__ = [
    "_allowed_mask_with_forbidden",
    "_connection_generator",
    "_dispatch_indexed_sparse_projection",
    "_indexed_sparse_projection",
    "_resolve_index_dtype",
    "_sample_indices_from_mask",
    "_scale_for_sparse_target",
    "_validate_forbidden_indices",
    "_workspace_limited_chunk_size",
]

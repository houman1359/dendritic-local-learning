"""Exact BSR inference export for block-structured dendritic projections.

``IndexedSparseLinear`` is already physically compact: it stores only K
weights and K source indices per output.  When its topology additionally forms
square dense blocks, this module can fold the weight transform and express the
same operator in PyTorch's block-sparse-row (BSR) format.  That creates a
standard-library deployment control for the custom indexed kernels and follows
the mask-folding/BSR inference method used by PyTorch SuperBlock.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)


@dataclass(frozen=True)
class BSRConversionRecord:
    """Auditable record for one indexed-to-BSR inference conversion."""

    path: str
    source_type: str
    target_type: str
    in_features: int
    out_features: int
    block_size: int
    nonzero_blocks: int
    active_weights: int
    dense_weights: int
    sparsity: float
    value_bytes: int
    topology_bytes: int


class BSRInferenceLinear(nn.Module):
    """Parameter-free linear operator backed by physically compact BSR buffers.

    The three persistent buffers are the complete deployed representation:
    dense nonzero blocks, compressed row pointers, and block-column indices.
    A sparse tensor view is cached only at runtime and is never serialized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        block_size: int,
        crow_indices: torch.Tensor,
        col_indices: torch.Tensor,
        values: torch.Tensor,
    ) -> None:
        super().__init__()
        block_size = int(block_size)
        if block_size < 1:
            raise ValueError("block_size must be positive")
        if in_features % block_size or out_features % block_size:
            raise ValueError("BSR dimensions must be divisible by block_size")
        expected_values = (int(col_indices.numel()), block_size, block_size)
        if tuple(values.shape) != expected_values:
            raise ValueError(
                f"values must have shape {expected_values}, got {tuple(values.shape)}"
            )
        expected_crow = out_features // block_size + 1
        if tuple(crow_indices.shape) != (expected_crow,):
            raise ValueError(
                f"crow_indices must have shape ({expected_crow},), "
                f"got {tuple(crow_indices.shape)}"
            )
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.block_size = block_size
        self.register_buffer("crow_indices", crow_indices.to(dtype=torch.int32))
        self.register_buffer("col_indices", col_indices.to(dtype=torch.int32))
        self.register_buffer("values", values)
        self.register_buffer("_bsr_weight", None, persistent=False)

    @property
    def nonzero_blocks(self) -> int:
        return int(self.col_indices.numel())

    @property
    def active_weights(self) -> int:
        return int(self.values.numel())

    def _weight(self) -> torch.Tensor:
        cached = self._bsr_weight
        if cached is None:
            cached = torch.sparse_bsr_tensor(
                self.crow_indices,
                self.col_indices,
                self.values,
                size=(self.out_features, self.in_features),
                device=self.values.device,
                dtype=self.values.dtype,
            )
            self._bsr_weight = cached
        return cached

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.linear(inputs, self._weight())

    def dense_weight(self) -> torch.Tensor:
        """Materialize the dense compatibility view for tests and diagnostics."""

        return self._weight().to_dense()

    def connectivity_resource_counts(self) -> dict[str, int | float | str]:
        return {
            "out_features": self.out_features,
            "in_features": self.in_features,
            "candidate_slots": self.out_features * self.in_features,
            "active_synapses": self.active_weights,
            "realized_k_min": self.active_weights // self.out_features,
            "realized_k_max": self.active_weights // self.out_features,
            "realized_k_mean": float(self.active_weights / self.out_features),
            "selection_policy": "fixed_bsr",
            "mask_source": "compressed_block_indices",
        }

    def parameter_estimate(self) -> dict[str, int]:
        dense = self.in_features * self.out_features
        return {
            "stored_total": self.active_weights,
            "active_total": self.active_weights,
            "dense_control": dense,
        }

    def _apply(self, fn):
        self._bsr_weight = None
        return super()._apply(fn)

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        self._bsr_weight = None
        super()._load_from_state_dict(*args, **kwargs)


def _indexed_to_bsr_values(
    layer: IndexedSparseLinear,
    *,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build BSR component tensors without materializing a dense weight."""

    block_size = int(block_size)
    indices = layer.connection_indices.to(dtype=torch.long)
    with torch.no_grad():
        weights = layer._normalized_sparse_weight().detach()
    crow = [0]
    block_columns: list[torch.Tensor] = []
    block_values: list[torch.Tensor] = []
    for row_start in range(0, layer.out_features, block_size):
        row_indices = indices[row_start : row_start + block_size]
        reference = row_indices[0].sort().values
        if not bool((row_indices.sort(dim=1).values == reference).all()):
            raise ValueError(
                "Rows within each BSR block row must share identical supports"
            )
        if reference.numel() % block_size:
            raise ValueError("K must be divisible by the BSR block size")
        chunks = reference.reshape(-1, block_size)
        starts = chunks[:, 0]
        expected = starts[:, None] + torch.arange(block_size, device=starts.device)
        if bool((starts.remainder(block_size) != 0).any()) or not bool(
            (chunks == expected).all()
        ):
            raise ValueError(
                "Indexed supports must contain complete aligned column blocks"
            )
        sorted_indices, order = row_indices.sort(dim=1)
        del sorted_indices
        sorted_weights = weights[row_start : row_start + block_size].gather(
            1, order.to(weights.device)
        )
        for block_index, start in enumerate(starts):
            block_columns.append((start // block_size).to(dtype=torch.int32))
            offset = block_index * block_size
            block_values.append(sorted_weights[:, offset : offset + block_size])
        crow.append(crow[-1] + int(starts.numel()))
    return (
        torch.tensor(crow, dtype=torch.int32, device=indices.device),
        torch.stack(block_columns).to(device=indices.device, dtype=torch.int32),
        torch.stack(block_values).to(device=weights.device, dtype=weights.dtype),
    )


def indexed_sparse_to_bsr(
    layer: IndexedSparseLinear,
    *,
    allowed_block_sizes: Sequence[int] = (16, 32, 64),
) -> BSRInferenceLinear:
    """Losslessly fold one eligible structured indexed layer into BSR buffers."""

    if layer.training:
        raise ValueError("BSR export is inference-only; call eval() before conversion")
    row_group = int(getattr(layer, "support_group_rows", 1))
    col_block = int(getattr(layer, "support_col_block", 1))
    allowed = {int(value) for value in allowed_block_sizes}
    if row_group != col_block or row_group not in allowed:
        raise ValueError(
            "BSR export requires equal row/column support blocks in "
            f"{sorted(allowed)}, got ({row_group}, {col_block})"
        )
    if layer.out_features % row_group or layer.in_features % row_group:
        raise ValueError(
            "BSR export requires input/output dimensions divisible by block size"
        )
    crow, columns, values = _indexed_to_bsr_values(layer, block_size=row_group)
    return BSRInferenceLinear(
        layer.in_features,
        layer.out_features,
        block_size=row_group,
        crow_indices=crow,
        col_indices=columns,
        values=values,
    )


def convert_structured_indexed_to_bsr_(
    model: nn.Module,
    *,
    allowed_block_sizes: Sequence[int] = (16, 32, 64),
    minimum_sparsity: float = 0.7,
    include_patterns: Sequence[str] = (),
    exclude_patterns: Sequence[str] = (),
    verify: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    strict: bool = False,
) -> dict[str, Any]:
    """Convert eligible indexed modules to physically compact BSR inference.

    Conversion is prepared and numerically verified before any module is
    swapped. Ineligible modules remain on the indexed kernel path and are
    listed with a reason; ``strict=True`` instead rejects the whole operation.
    """

    if not 0.0 <= float(minimum_sparsity) <= 1.0:
        raise ValueError("minimum_sparsity must be in [0, 1]")
    includes = tuple(str(item) for item in include_patterns)
    excludes = tuple(str(item) for item in exclude_patterns)

    def selected(path: str) -> bool:
        included = not includes or any(
            fnmatch.fnmatchcase(path, item) for item in includes
        )
        excluded = any(fnmatch.fnmatchcase(path, item) for item in excludes)
        return included and not excluded

    prepared: list[
        tuple[nn.Module, str, str, IndexedSparseLinear, BSRInferenceLinear]
    ] = []
    skipped: list[dict[str, str]] = []
    for parent_path, parent in model.named_modules():
        for child_name, child in parent.named_children():
            path = f"{parent_path}.{child_name}" if parent_path else child_name
            if type(child) is not IndexedSparseLinear or not selected(path):
                continue
            sparsity = 1.0 - float(child.K / child.in_features)
            if sparsity < float(minimum_sparsity):
                skipped.append(
                    {
                        "path": path,
                        "reason": f"sparsity {sparsity:.6f} below {minimum_sparsity:.6f}",
                    }
                )
                continue
            try:
                replacement = indexed_sparse_to_bsr(
                    child, allowed_block_sizes=allowed_block_sizes
                )
                if verify:
                    generator = torch.Generator(device=child.pre_w.device).manual_seed(
                        0
                    )
                    sample = torch.randn(
                        2,
                        child.in_features,
                        generator=generator,
                        device=child.pre_w.device,
                        dtype=child.pre_w.dtype,
                    )
                    with torch.no_grad():
                        expected = child(sample)
                        observed = replacement(sample)
                    if not torch.allclose(expected, observed, atol=atol, rtol=rtol):
                        error = float((expected - observed).abs().max())
                        raise RuntimeError(
                            f"BSR conversion changed outputs (max error={error:.6g})"
                        )
            except (RuntimeError, ValueError) as exc:
                skipped.append({"path": path, "reason": str(exc)})
                continue
            prepared.append((parent, child_name, path, child, replacement))
    if strict and skipped:
        details = "; ".join(f"{item['path']}: {item['reason']}" for item in skipped)
        raise ValueError(
            f"Strict BSR conversion rejected ineligible modules: {details}"
        )

    records: list[BSRConversionRecord] = []
    for parent, child_name, path, source, replacement in prepared:
        parent._modules[child_name] = replacement
        value_bytes = replacement.values.numel() * replacement.values.element_size()
        topology_bytes = (
            replacement.crow_indices.numel() * replacement.crow_indices.element_size()
            + replacement.col_indices.numel() * replacement.col_indices.element_size()
        )
        dense = source.in_features * source.out_features
        records.append(
            BSRConversionRecord(
                path=path,
                source_type=type(source).__name__,
                target_type=type(replacement).__name__,
                in_features=source.in_features,
                out_features=source.out_features,
                block_size=replacement.block_size,
                nonzero_blocks=replacement.nonzero_blocks,
                active_weights=replacement.active_weights,
                dense_weights=dense,
                sparsity=1.0 - replacement.active_weights / dense,
                value_bytes=int(value_bytes),
                topology_bytes=int(topology_bytes),
            )
        )
    return {
        "schema": "dendritic_bsr_conversion/v1",
        "converted": [asdict(record) for record in records],
        "skipped": skipped,
        "converted_modules": len(records),
        "active_weights": sum(record.active_weights for record in records),
        "value_bytes": sum(record.value_bytes for record in records),
        "topology_bytes": sum(record.topology_bytes for record in records),
    }


__all__ = [
    "BSRConversionRecord",
    "BSRInferenceLinear",
    "convert_structured_indexed_to_bsr_",
    "indexed_sparse_to_bsr",
]

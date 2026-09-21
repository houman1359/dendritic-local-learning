"""
Sparse Operations Utilities Module.

This module provides utility functions for sparse operations in neural networks,
including TopK selection, sparsity calculations, and parallel processing.
"""

import multiprocessing as mp
from typing import Optional

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
    DeepstLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.dense_to_sparse import (
    DenseToSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory import (
    SPARSE_LAYER_REGISTRY,
    get_available_sparse_types,
    get_sparse_layer,
    normalize_sparse_layer_type,
    register_sparse_layer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed import (
    IndexedDynamicTopKLinear,
    IndexedRewireLinear,
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import (
    StochasticTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)


def calculate_effective_sparsity(weights: torch.Tensor, K: int) -> float:
    """
    Calculate the effective sparsity of a weight tensor with TopK selection.

    Args:
        weights: Weight tensor
        K: Number of active connections per output

    Returns:
        Sparsity as a fraction (0.0 = dense, 1.0 = completely sparse)
    """
    total_weights = weights.numel()
    active_weights = weights.shape[0] * K
    sparsity = 1.0 - (active_weights / total_weights)
    return sparsity


def create_topk_mask(
    weights: torch.Tensor, K: int, dim: int = -1, largest: bool = True
) -> torch.Tensor:
    """
    Create a binary mask for TopK selection.

    Args:
        weights: Weight tensor
        K: Number of elements to select
        dim: Dimension along which to select TopK
        largest: Whether to select largest or smallest values

    Returns:
        Binary mask with 1s for selected elements, 0s elsewhere
    """
    # Get TopK indices
    _, indices = torch.topk(weights, K, dim=dim, largest=largest, sorted=False)

    # Create mask
    mask = torch.zeros_like(weights, dtype=torch.bool)

    if dim == -1:
        # Handle last dimension case
        batch_indices = torch.arange(weights.shape[0], device=weights.device)[:, None]
        mask[batch_indices, indices] = True
    else:
        # More general case - would need more implementation for arbitrary dims
        raise NotImplementedError("Only dim=-1 currently supported")

    return mask.float()


def apply_sparse_mask(weights: torch.Tensor, K: int, dim: int = -1) -> torch.Tensor:
    """
    Apply TopK sparsity to a weight tensor.

    Args:
        weights: Weight tensor to sparsify
        K: Number of elements to keep per row/column
        dim: Dimension along which to apply sparsity

    Returns:
        Sparsified weight tensor
    """
    mask = create_topk_mask(weights, K, dim)
    return weights * mask


def count_active_parameters(weights: torch.Tensor, K: int) -> int:
    """
    Count the number of active (non-zero) parameters after TopK selection.

    Args:
        weights: Weight tensor
        K: Number of active connections per output

    Returns:
        Number of active parameters
    """
    return weights.shape[0] * K


def estimate_memory_usage(
    in_features: int, out_features: int, K: int, dtype: torch.dtype = torch.float32
) -> dict[str, float]:
    """
    Estimate memory usage for a sparse layer.

    Args:
        in_features: Number of input features
        out_features: Number of output features
        K: Number of active connections per output
        dtype: Data type for weights

    Returns:
        Dictionary with memory estimates in MB
    """
    bytes_per_element = torch.tensor(0, dtype=dtype).element_size()

    # Full weight matrix memory
    full_weights_mb = (in_features * out_features * bytes_per_element) / 1e6

    # Active weights memory
    active_weights_mb = (out_features * K * bytes_per_element) / 1e6

    # Memory savings
    savings_mb = full_weights_mb - active_weights_mb
    savings_percent = (savings_mb / full_weights_mb) * 100

    return {
        "full_weights_mb": full_weights_mb,
        "active_weights_mb": active_weights_mb,
        "savings_mb": savings_mb,
        "savings_percent": savings_percent,
        "sparsity": calculate_effective_sparsity(
            torch.randn(out_features, in_features), K
        ),
    }


def parallel_topk_selection(
    weights: torch.Tensor,
    K: int,
    num_workers: Optional[int] = None,
    threshold_size: int = 1000000,
    dim: int = -1,
) -> torch.Tensor:
    """
    Parallel TopK selection for large tensors using multiprocessing.

    For large tensors (>threshold_size elements), this function splits
    the work across multiple CPU cores for faster processing.

    Args:
        weights: Weight tensor for TopK selection
        K: Number of elements to select per row
        num_workers: Number of worker processes (None = auto-detect)
        threshold_size: Minimum tensor size to use parallel processing
        dim: Dimension along which to select TopK

    Returns:
        Binary mask with 1s for selected elements, 0s elsewhere
    """
    total_elements = weights.numel()

    # Use standard TopK for small tensors
    if total_elements < threshold_size:
        return create_topk_mask(weights, K, dim)

    # Auto-detect number of workers
    if num_workers is None:
        num_workers = min(4, mp.cpu_count())  # Cap at 4 for memory efficiency

    # For very large tensors, use parallel processing
    if dim == -1:
        return _parallel_topk_lastdim(weights, K, num_workers)
    else:
        # Fallback to standard implementation for other dimensions
        return create_topk_mask(weights, K, dim)


def _worker_topk(args):
    """Worker function for parallel TopK computation."""
    weights_chunk, K = args
    indices = torch.topk(weights_chunk, K, dim=-1, largest=True, sorted=False)[1]
    mask_chunk = torch.zeros_like(weights_chunk)

    # Create batch indices for the chunk
    batch_indices = torch.arange(weights_chunk.shape[0])[:, None]
    mask_chunk[batch_indices, indices] = 1

    return mask_chunk


def _parallel_topk_lastdim(
    weights: torch.Tensor, K: int, num_workers: int
) -> torch.Tensor:
    """Parallel TopK selection along the last dimension."""
    num_rows = weights.shape[0]

    # Split rows across workers
    chunk_size = max(1, num_rows // num_workers)
    chunks = []

    for i in range(0, num_rows, chunk_size):
        end_idx = min(i + chunk_size, num_rows)
        chunk = weights[i:end_idx]
        chunks.append((chunk, K))

    # Process chunks in parallel
    try:
        with mp.Pool(num_workers) as pool:
            mask_chunks = pool.map(_worker_topk, chunks)

        # Concatenate results
        return torch.cat(mask_chunks, dim=0)

    except Exception:
        # Fallback to standard implementation on any error
        return create_topk_mask(weights, K, -1)


__all__ = [
    "SPARSE_LAYER_REGISTRY",
    "DeepstLinear",
    "DenseToSparseLinear",
    "IndexedDynamicTopKLinear",
    "IndexedRewireLinear",
    "IndexedSparseLinear",
    "StochasticTopKLinear",
    "TopKLinear",
    "VarianceTopKLinear",
    "apply_sparse_mask",
    "calculate_effective_sparsity",
    "count_active_parameters",
    "create_topk_mask",
    "estimate_memory_usage",
    "get_available_sparse_types",
    "get_sparse_layer",
    "normalize_sparse_layer_type",
    "parallel_topk_selection",
    "register_sparse_layer",
]

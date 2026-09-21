"""Optional TorchAO inference transforms for conventional dense layers."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.config.compression import DenseRuntimeCompressionConfig


def _selected_linear_filter(
    config: DenseRuntimeCompressionConfig,
) -> Callable[[nn.Module, str], bool]:
    includes = [re.compile(pattern) for pattern in config.include_patterns]
    excludes = [re.compile(pattern) for pattern in config.exclude_patterns]

    def selected(module: nn.Module, fqn: str) -> bool:
        if type(module) is not nn.Linear:
            return False
        if module.in_features < config.min_in_features:
            return False
        if module.out_features < config.min_out_features:
            return False
        if includes and not any(pattern.search(fqn) for pattern in includes):
            return False
        return not any(pattern.search(fqn) for pattern in excludes)

    return selected


def _block_zero_fraction(weight: torch.Tensor, block_size: int) -> float:
    if weight.ndim != 2:
        raise ValueError("Block sparsity requires a two-dimensional weight")
    rows, columns = weight.shape
    if rows % block_size or columns % block_size:
        raise ValueError(
            f"Weight shape {tuple(weight.shape)} is not divisible by "
            f"block_size={block_size}"
        )
    blocks = weight.detach().reshape(
        rows // block_size,
        block_size,
        columns // block_size,
        block_size,
    )
    zero_blocks = blocks.eq(0).all(dim=1).all(dim=-1)
    return float(zero_blocks.float().mean().item())


def optimize_dense_runtime_(
    model: nn.Module,
    config: DenseRuntimeCompressionConfig,
) -> dict[str, Any]:
    """Apply one explicitly selected TorchAO inference transform in-place.

    Block-sparse conversion never prunes weights. It accepts only layers that
    already satisfy the configured block-zero fraction, preventing accidental
    post-training accuracy loss.
    """

    config.validate()
    if config.backend == "none":
        return {"backend": "none", "transformed_modules": []}
    if model.training:
        raise ValueError("TorchAO runtime compression requires model.eval()")
    try:
        from torchao.quantization import (
            Int4WeightOnlyConfig,
            Int8WeightOnlyConfig,
            quantize_,
        )
        from torchao.sparsity import block_sparse_weight, sparsify_
    except ImportError as error:
        raise ImportError(
            "TorchAO compression is optional. Install "
            "dendritic-modeling[compression] before selecting a torchao backend."
        ) from error

    selected = _selected_linear_filter(config)
    names = [name for name, module in model.named_modules() if selected(module, name)]
    if not names:
        raise ValueError("No dense nn.Linear modules match the compression policy")

    if config.backend == "torchao_block_sparse":
        for name, module in model.named_modules():
            if not selected(module, name):
                continue
            observed = _block_zero_fraction(module.weight, config.block_size)
            if observed < config.minimum_block_sparsity:
                raise ValueError(
                    f"Dense layer {name!r} has block sparsity {observed:.3f}, below "
                    f"the required {config.minimum_block_sparsity:.3f}. Train or "
                    "prune an explicit block mask before BSR conversion."
                )
        sparsify_(
            model,
            block_sparse_weight(blocksize=config.block_size),
            filter_fn=selected,
        )
    elif config.backend == "torchao_int8_weight_only":
        quantize_(model, Int8WeightOnlyConfig(), filter_fn=selected)
    elif config.backend == "torchao_int4_weight_only":
        quantize_(
            model,
            Int4WeightOnlyConfig(group_size=config.int4_group_size),
            filter_fn=selected,
        )
    else:  # validated above; defensive against future config drift
        raise AssertionError(f"Unhandled compression backend {config.backend!r}")
    return {"backend": config.backend, "transformed_modules": names}


__all__ = ["optimize_dense_runtime_"]

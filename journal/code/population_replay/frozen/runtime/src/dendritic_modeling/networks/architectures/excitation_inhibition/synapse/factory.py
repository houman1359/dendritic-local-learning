"""Sparse synapse layer registry and construction helpers."""

from __future__ import annotations

import torch.nn as nn

from dendritic_modeling.config.conversion import normalize_sparsity_type
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.credit_deepst import (
    CreditGatedDeepstLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
    DeepstLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.dense_to_sparse import (
    DenseToSparseLinear,
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

SPARSE_LAYER_REGISTRY: dict[str, type[nn.Module]] = {
    "standard": TopKLinear,
    "stochastic": StochasticTopKLinear,
    "variance": VarianceTopKLinear,
    "deepst": DeepstLinear,
    "credit_deepst": CreditGatedDeepstLinear,
    "dense_to_sparse": DenseToSparseLinear,
    "annealed_topk": DenseToSparseLinear,
    "indexed": IndexedSparseLinear,
    "indexed_dynamic": IndexedDynamicTopKLinear,
    "indexed_rewire": IndexedRewireLinear,
}


def normalize_sparse_layer_type(layer_type: object) -> str:
    """Normalize public sparse-layer aliases to registry keys."""
    return normalize_sparsity_type(layer_type)


def get_sparse_layer(
    layer_type: str,
    in_features: int,
    out_features: int,
    K: int,
    **kwargs,
) -> nn.Module:
    """Create a sparse synapse layer by registered type."""
    layer_type = normalize_sparse_layer_type(layer_type)
    if layer_type not in SPARSE_LAYER_REGISTRY:
        available_types = list(SPARSE_LAYER_REGISTRY.keys())
        raise ValueError(
            f"Unknown sparse layer type '{layer_type}'. Available types: {available_types}"
        )

    layer_class = SPARSE_LAYER_REGISTRY[layer_type]
    if layer_type in {"deepst", "credit_deepst"}:
        return layer_class(in_features=in_features, out_features=out_features, **kwargs)
    return layer_class(
        in_features=in_features,
        out_features=out_features,
        K=K,
        **kwargs,
    )


def register_sparse_layer(name: str, layer_class: type[nn.Module]) -> None:
    """Register a sparse synapse implementation."""
    SPARSE_LAYER_REGISTRY[name] = layer_class


def get_available_sparse_types() -> list[str]:
    """Return registered sparse synapse type names."""
    return list(SPARSE_LAYER_REGISTRY.keys())


__all__ = [
    "SPARSE_LAYER_REGISTRY",
    "get_available_sparse_types",
    "get_sparse_layer",
    "normalize_sparse_layer_type",
    "register_sparse_layer",
]

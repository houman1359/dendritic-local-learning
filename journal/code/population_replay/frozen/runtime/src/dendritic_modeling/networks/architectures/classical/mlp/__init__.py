"""
MLP architectures.

This module contains various MLP-based neural network architectures.
"""

from dendritic_modeling.networks.architectures.classical.mlp.direct_active_matched import (
    DirectActiveMatchedPointBottleneck,
    matched_direct_bottleneck_width,
)
from dendritic_modeling.networks.architectures.classical.mlp.effective_matched import (
    EffectiveMatchedParamMLP,
)
from dendritic_modeling.networks.architectures.classical.mlp.matched_param import (
    MatchedActiveParamMLP,
    MatchedTotalParamMLP,
    SparseStructuredMLP,
)
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
from dendritic_modeling.networks.architectures.classical.mlp.point_mlp import (
    PointMatchedParamMLP,
)
from dendritic_modeling.networks.architectures.classical.mlp.sparse_active_matched import (
    SparseActiveMatchedPointAffine,
)

__all__ = [
    "MLP",
    "DirectActiveMatchedPointBottleneck",
    "EffectiveMatchedParamMLP",
    "MatchedActiveParamMLP",
    "MatchedTotalParamMLP",
    "PointMatchedParamMLP",
    "SparseActiveMatchedPointAffine",
    "SparseStructuredMLP",
    "matched_direct_bottleneck_width",
]

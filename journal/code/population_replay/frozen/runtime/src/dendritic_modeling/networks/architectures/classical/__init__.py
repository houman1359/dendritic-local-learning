from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNNDownsample,
    CNNUpsample,
)
from dendritic_modeling.networks.architectures.classical.identity import Identity
from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    DirectActiveMatchedPointBottleneck,
    MatchedActiveParamMLP,
    MatchedTotalParamMLP,
    SparseActiveMatchedPointAffine,
    SparseStructuredMLP,
)
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)

__all__ = [
    "MLP",
    "AlexNet",
    "CNNDownsample",
    "CNNUpsample",
    "DirectActiveMatchedPointBottleneck",
    "Identity",
    "MatchedActiveParamMLP",
    "MatchedTotalParamMLP",
    "PathwayRouter",
    "SparseActiveMatchedPointAffine",
    "SparseStructuredMLP",
]

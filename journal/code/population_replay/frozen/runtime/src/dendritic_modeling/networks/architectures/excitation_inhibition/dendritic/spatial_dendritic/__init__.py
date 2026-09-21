"""Spatial dendritic core networks for pretrained backbone replacement.

Two variants:
- **Tensor Map** (baseline/control): flatten spatial input → DendriNet → project.
- **Spatial Conv** (target): F.unfold local patches → shared DendriNet → spatial output.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic.spatial_conv import (
    HierarchicalDendriticConv,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic.tensor_map import (
    HierarchicalDendriticTensorMap,
)

__all__ = ["HierarchicalDendriticConv", "HierarchicalDendriticTensorMap"]

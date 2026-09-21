"""Canonical dendritic-depth semantics shared by analysis modules."""

from __future__ import annotations

from numbers import Integral
from typing import Any

SOMA_RELATIVE_DEPTH_REFERENCE = "soma_relative"


def soma_relative_dendritic_depth(module: Any) -> int:
    """Return a branch layer's canonical distance from the soma.

    ``DendriticBranchLayer.layer_idx`` is assigned by the network builders as
    ``0`` for the soma, ``1`` for the proximal dendritic level, and increasing
    values toward distal levels.  Analysis code must use that value directly;
    list position and morphology depth are not interchangeable with it.
    """
    depth = getattr(module, "layer_idx", None)
    if isinstance(depth, bool) or not isinstance(depth, Integral):
        raise ValueError(
            "Dendritic analysis requires an integer layer_idx using the "
            "soma-relative convention (soma=0), "
            f"got {depth!r} from {type(module).__name__}."
        )
    depth = int(depth)
    if depth < 0:
        raise ValueError(
            "Dendritic layer_idx must be non-negative under the soma-relative "
            f"convention, got {depth}."
        )
    return depth


__all__ = [
    "SOMA_RELATIVE_DEPTH_REFERENCE",
    "soma_relative_dendritic_depth",
]

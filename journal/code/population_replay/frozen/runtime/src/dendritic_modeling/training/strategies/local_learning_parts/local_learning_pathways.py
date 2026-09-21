"""Shared pathway descriptors for local-learning synaptic updates."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TopKGradientPath:
    """Recorded TopK pathway keys used by local-learning dispatch."""

    pathway: str
    module_attr: str
    module_key: str
    input_key: str
    output_key: str
    mask_key: str


EXCITATORY_TOPK_PATHS = (
    TopKGradientPath(
        "exc",
        "branch_excitation",
        "exc_module",
        "x_exc",
        "exc_out",
        "exc_mask",
    ),
    TopKGradientPath(
        "rec_exc",
        "branch_recurrent",
        "rec_exc_module",
        "x_rec_exc",
        "rec_exc_out",
        "rec_exc_mask",
    ),
)

INHIBITORY_TOPK_PATHS = (
    TopKGradientPath(
        "inh",
        "branch_inhibition",
        "inh_module",
        "x_inh",
        "inh_out",
        "inh_mask",
    ),
    TopKGradientPath(
        "rec_inh",
        "branch_rec_inhibition",
        "rec_inh_module",
        "x_rec_inh",
        "rec_inh_out",
        "rec_inh_mask",
    ),
)

STDP_TOPK_PATHS = (
    EXCITATORY_TOPK_PATHS[0],
    INHIBITORY_TOPK_PATHS[0],
    EXCITATORY_TOPK_PATHS[1],
    INHIBITORY_TOPK_PATHS[1],
)

RECORDER_TOPK_PATHS = STDP_TOPK_PATHS


def iter_topk_path_modules(
    module: Any,
    paths: Iterable[TopKGradientPath] = STDP_TOPK_PATHS,
) -> Iterator[tuple[TopKGradientPath, Any]]:
    """Yield present TopK modules for the requested pathway descriptors."""
    for path in paths:
        topk_module = getattr(module, path.module_attr, None)
        if topk_module is not None:
            yield path, topk_module


__all__ = [
    "EXCITATORY_TOPK_PATHS",
    "INHIBITORY_TOPK_PATHS",
    "RECORDER_TOPK_PATHS",
    "STDP_TOPK_PATHS",
    "TopKGradientPath",
    "iter_topk_path_modules",
]

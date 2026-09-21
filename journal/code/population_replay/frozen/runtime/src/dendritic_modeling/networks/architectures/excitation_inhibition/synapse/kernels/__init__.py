"""Optional accelerated kernels for indexed sparse synapses."""

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.backend import (
    INDEXED_PROJECTION_BACKENDS,
    indexed_projection_backend_available,
    normalize_indexed_projection_backend,
    normalize_indexed_projection_options,
    resolve_indexed_projection_backend,
)

__all__ = [
    "INDEXED_PROJECTION_BACKENDS",
    "indexed_projection_backend_available",
    "normalize_indexed_projection_backend",
    "normalize_indexed_projection_options",
    "resolve_indexed_projection_backend",
]

"""Public indexed sparse synapse exports.

The implementation is split by layer behavior while this module preserves the
historical import path.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _allowed_mask_with_forbidden as _allowed_mask_with_forbidden,
    _connection_generator as _connection_generator,
    _sample_indices_from_mask as _sample_indices_from_mask,
    _scale_for_sparse_target as _scale_for_sparse_target,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_dynamic import (
    IndexedDynamicTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_rewire import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)

IndexedDynamicTopKLinear.__module__ = __name__
IndexedRewireLinear.__module__ = __name__
IndexedSparseLinear.__module__ = __name__

__all__ = [
    "IndexedDynamicTopKLinear",
    "IndexedRewireLinear",
    "IndexedSparseLinear",
]

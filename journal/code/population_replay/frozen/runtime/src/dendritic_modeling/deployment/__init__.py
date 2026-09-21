"""Deployment and post-training model-compaction utilities."""

from dendritic_modeling.deployment.block_sparse import (
    BSRConversionRecord,
    BSRInferenceLinear,
    convert_structured_indexed_to_bsr_,
    indexed_sparse_to_bsr,
)
from dendritic_modeling.deployment.compression import (
    COMPACT_CHECKPOINT_FORMAT,
    COMPACT_CHECKPOINT_VERSION,
    compact_model_checkpoint,
    freeze_sparse_topology_,
    load_compact_model_checkpoint,
    prepare_model_for_compact_state_,
)
from dendritic_modeling.deployment.csr_sparse import (
    CSRInferenceLinear,
    masked_weight_to_csr,
)
from dendritic_modeling.deployment.packed_population import (
    PackedPopulationLinear,
    PackedValueMatrix,
    pack_population_cell_,
    packed_population_fingerprint,
    packed_population_ledger,
    population_cell_payload,
    repack_population_topology_,
    restore_packed_population_cell_,
    set_packed_population_backend_,
)
from dendritic_modeling.deployment.pruning import (
    IndexedPruningTarget,
    apply_indexed_pruning_targets_,
    prune_indexed_layer,
    prune_model_indexed_,
    pruning_targets_manifest,
    resolve_indexed_pruning_targets,
)
from dendritic_modeling.deployment.torchao import optimize_dense_runtime_

__all__ = [
    "COMPACT_CHECKPOINT_FORMAT",
    "COMPACT_CHECKPOINT_VERSION",
    "BSRConversionRecord",
    "BSRInferenceLinear",
    "CSRInferenceLinear",
    "IndexedPruningTarget",
    "PackedPopulationLinear",
    "PackedValueMatrix",
    "apply_indexed_pruning_targets_",
    "compact_model_checkpoint",
    "convert_structured_indexed_to_bsr_",
    "freeze_sparse_topology_",
    "indexed_sparse_to_bsr",
    "load_compact_model_checkpoint",
    "masked_weight_to_csr",
    "optimize_dense_runtime_",
    "pack_population_cell_",
    "packed_population_fingerprint",
    "packed_population_ledger",
    "population_cell_payload",
    "prepare_model_for_compact_state_",
    "prune_indexed_layer",
    "prune_model_indexed_",
    "pruning_targets_manifest",
    "repack_population_topology_",
    "resolve_indexed_pruning_targets",
    "restore_packed_population_cell_",
    "set_packed_population_backend_",
]

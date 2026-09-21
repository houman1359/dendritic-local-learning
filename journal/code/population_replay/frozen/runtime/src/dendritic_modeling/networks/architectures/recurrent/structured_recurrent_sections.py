"""Section parsing helpers for structured recurrent E/I configs."""

from __future__ import annotations

import logging
from typing import Any

from dendritic_modeling.config.conversion import (
    normalize_sparsity_type,
    to_plain_dict as _to_plain_mapping,
)
from dendritic_modeling.config.legacy import normalize_transfer_config
from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_types import (
    _StructuredRecurrentSections,
)

logger = logging.getLogger(__name__)


def _normalize_sparsity_type(sparsity_type: Any) -> str:
    """Normalize public sparsity aliases to the internal TopK strategy names."""
    return normalize_sparsity_type(sparsity_type)


def _parse_structured_recurrent_sections(
    raw_params: dict[str, Any],
) -> _StructuredRecurrentSections:
    recurrent_cfg = _to_plain_mapping(raw_params.get("recurrent_ei", {}))
    architecture = _to_plain_mapping(raw_params.get("architecture", {}))
    connectivity = _to_plain_mapping(raw_params.get("connectivity", {}))
    structured_alias_present = "structured_connectivity" in connectivity
    structured_alias = connectivity.pop("structured_connectivity", {})
    if structured_alias_present:
        logger.warning(
            "model.core.connectivity.structured_connectivity is deprecated; "
            "use model.core.connectivity.structured instead."
        )
    structured_connectivity = connectivity.pop("structured", structured_alias)
    transfer = normalize_transfer_config(raw_params.get("transfer", {}))
    morphology = _to_plain_mapping(raw_params.get("morphology", {}))
    sparsity = _to_plain_mapping(raw_params.get("sparsity", {}))
    reactivation = _to_plain_mapping(raw_params.get("reactivation", {}))
    blocklinear = _to_plain_mapping(raw_params.get("blocklinear", {}))
    implementation = _to_plain_mapping(raw_params.get("implementation", {}))
    synapse_types = _to_plain_mapping(raw_params.get("synapse_types", {}))
    dynamics = _to_plain_mapping(
        raw_params.get("dynamics", raw_params.get("spiking", {}))
    )
    dendritic_spikes = _to_plain_mapping(
        raw_params.get("dendritic_spikes", dynamics.get("dendritic_spikes", {}))
    )
    soma_feedback = _to_plain_mapping(
        raw_params.get(
            "soma_feedback",
            recurrent_cfg.get("soma_feedback", dynamics.get("soma_feedback", {})),
        )
    )
    deepst = _to_plain_mapping(sparsity.get("deepst", {}))
    if "annealed_topk" in sparsity and "dense_to_sparse" not in sparsity:
        logger.warning(
            "model.core.sparsity.annealed_topk is deprecated; use "
            "model.core.sparsity.dense_to_sparse instead."
        )
    dense_to_sparse = _to_plain_mapping(
        sparsity.get("dense_to_sparse", sparsity.get("annealed_topk", {}))
    )
    indexed = _to_plain_mapping(sparsity.get("indexed", {}))
    return _StructuredRecurrentSections(
        initialization_seed=raw_params.get("initialization_seed"),
        recurrent_cfg=recurrent_cfg,
        architecture=architecture,
        connectivity=connectivity,
        structured_connectivity=structured_connectivity,
        transfer=transfer,
        morphology=morphology,
        sparsity=sparsity,
        reactivation=reactivation,
        blocklinear=blocklinear,
        implementation=implementation,
        synapse_types=synapse_types,
        dynamics=dynamics,
        dendritic_spikes=dendritic_spikes,
        soma_feedback=soma_feedback,
        deepst=deepst,
        dense_to_sparse=dense_to_sparse,
        indexed=indexed,
    )


__all__ = [
    "_normalize_sparsity_type",
    "_parse_structured_recurrent_sections",
    "_to_plain_mapping",
]

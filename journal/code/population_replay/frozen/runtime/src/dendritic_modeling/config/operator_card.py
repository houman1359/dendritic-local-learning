"""Operator card: the resolved dendritic-operator axes of one configuration.

The scaling platform describes every dendritic module by seven orthogonal
axes (SCALING_PLATFORM_ROADMAP.md, WS1): input domain, weight domain,
integration rule, gate nonlinearity, morphology, topology, and output
contract. This module renders a core configuration into that vocabulary so
each trained run carries a compact, machine-readable statement of what
operator it actually used — the config analogue of ``model_resources.json``.

Reading is best-effort and side-effect free: unknown or missing sections
simply appear as their defaults, so the card writer can never break training.
"""

from __future__ import annotations

from typing import Any

OPERATOR_CARD_SCHEMA_VERSION = "dendritic_operator_card_v1"


def _get(section: Any, key: str, default: Any = None) -> Any:
    if section is None:
        return default
    if isinstance(section, dict):
        return section.get(key, default)
    return getattr(section, key, default)


def build_operator_card(core_config: Any) -> dict[str, Any]:
    """Summarize one core configuration along the seven operator axes."""

    spatial = _get(core_config, "spatial", {})
    architecture = _get(core_config, "architecture", {})
    connectivity = _get(core_config, "connectivity", {})
    morphology = _get(core_config, "morphology", {})
    reactivation = _get(core_config, "reactivation", {})
    sparsity = _get(core_config, "sparsity", {})
    transfer = _get(core_config, "transfer", {})

    use_shunting = bool(_get(morphology, "use_shunting", True))
    integration_rule = (
        "shunting"
        if use_shunting
        else f"additive_{_get(morphology, 'additive_mode', 'raw')}"
    )
    weight_transform = str(_get(morphology, "weight_transform", "exp"))

    from dendritic_modeling.config.operator_validation import (
        collect_biological_violations,
    )

    return {
        "schema_version": OPERATOR_CARD_SCHEMA_VERSION,
        "core_type": _get(core_config, "type", "einet"),
        "biological_neuron": {
            "declared": _get(core_config, "biological_neuron", None),
            "violations": collect_biological_violations(core_config),
        },
        "input_domain": {
            "input_transform": _get(spatial, "input_transform", "identity"),
            "input_scale": _get(spatial, "input_scale", 1.0),
            "transfer_output_activation": _get(transfer, "output_activation", None),
            "independent_pathways": _get(transfer, "independent_pathways", False),
        },
        "weight_domain": {
            "weight_transform": weight_transform,
            "signed": weight_transform.lower() == "identity",
        },
        "integration_rule": integration_rule,
        "gates": {
            "enabled": _get(reactivation, "enabled", True),
            "type": _get(reactivation, "type", "param_tanh"),
            "init_policy": _get(reactivation, "init_policy", "analytical"),
            "soma_type": _get(reactivation, "soma_type", None),
        },
        "morphology": {
            "excitatory_layer_sizes": _get(
                architecture, "excitatory_layer_sizes", None
            ),
            "inhibitory_layer_sizes": _get(
                architecture, "inhibitory_layer_sizes", None
            ),
            "excitatory_branch_factors": _get(
                architecture, "excitatory_branch_factors", None
            ),
            "somatic_synapses": _get(morphology, "somatic_synapses", True),
        },
        "topology": {
            "sparsity_type": _get(sparsity, "type", "standard"),
            "ee_synapses_per_branch": _get(
                connectivity, "ee_synapses_per_branch_per_layer", None
            ),
            "ie_synapses_per_branch": _get(
                connectivity, "ie_synapses_per_branch_per_layer", None
            ),
        },
        "output_contract": {
            "adapter_mode": _get(spatial, "output_adapter_mode", "identity"),
            "adapter_initial_scale": _get(spatial, "output_initial_scale", 1.0),
            "adapter_initial_threshold": _get(
                spatial, "output_initial_threshold", 0.25
            ),
        },
    }

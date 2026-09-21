"""Backward-compatible FMI imports for the general replacement compiler.

The compiler is not FMI-specific: teacher-side FMI, a morphology sweep, and a
manual architecture choice all target the same public replacement API. New
code should import from :mod:`dendritic_modeling.networks.architectures.replacement`.
"""

from dendritic_modeling.networks.architectures.replacement.compiler import (
    CANONICAL_FAMILIES,
    INTEGRATION_RULES,
    LEGACY_FAMILY_ALIASES,
    TOPOLOGY_MODES,
    CompiledReplacementPlan,
    apply_compiled_plan_to_config,
    compile_replacement_candidate,
    parse_candidate,
)

__all__ = [
    "CANONICAL_FAMILIES",
    "INTEGRATION_RULES",
    "LEGACY_FAMILY_ALIASES",
    "TOPOLOGY_MODES",
    "CompiledReplacementPlan",
    "apply_compiled_plan_to_config",
    "compile_replacement_candidate",
    "parse_candidate",
]

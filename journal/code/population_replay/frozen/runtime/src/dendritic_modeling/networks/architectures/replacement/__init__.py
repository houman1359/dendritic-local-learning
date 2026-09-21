"""Shared replacement-cell components.

Replacement cells define the reusable computation that sits between a source
model and a target contract:

``input_adapter -> dendritic core -> output_adapter``.

Domain-specific code is still responsible for placement.  Vision models split
backbones into prefix/core/suffix segments; transformer models patch modules
inside decoder blocks.
"""

from dendritic_modeling.networks.architectures.replacement.adapters import (
    FlatAffineOutputAdapter,
    LinearOutputAdapter,
    NonNegativeInputAdapter,
    PopulationToFeatureReduction,
    PositiveFeatureScale,
    PositiveThresholdReLU,
    SpatialNonNegativeInputAdapter,
    ZeroPadOutputAdapter,
    make_nonmixing_output_adapter,
    transformed_feature_dim,
    validate_input_transform,
)
from dendritic_modeling.networks.architectures.replacement.cells import (
    RUNTIME_TENSOR_CONTRACT_SCHEMA,
    LayerwiseTokenReplacementStack,
    TokenReplacementCell,
    make_linear_output_projection,
    preserve_runtime_tensor_contract,
    require_runtime_tensor_contract,
)
from dendritic_modeling.networks.architectures.replacement.compiler import (
    CANONICAL_FAMILIES,
    INTEGRATION_RULES,
    LEGACY_FAMILY_ALIASES,
    TOPOLOGY_MODES,
    CompiledReplacementPlan,
    apply_compiled_plan_to_config,
    apply_compiled_plans_by_layer_to_config,
    compile_replacement_candidate,
    compiled_replacement_plan_from_mapping,
    parse_candidate,
)
from dendritic_modeling.networks.architectures.replacement.module_patching import (
    ChannelMapReplacement,
    SelectedModuleReplacementRecord,
    replace_modules_with_selected_population_networks,
)
from dendritic_modeling.networks.architectures.replacement.projections import (
    OUTPUT_TOPOLOGY_MODES,
    PositiveEICoreOutputAdapter,
    PositiveEIOutputProjection,
    TransformedLinear,
    make_output_projection,
    make_positive_ei_output_projection,
    output_projection_parameter_counts,
)
from dendritic_modeling.networks.architectures.replacement.selection import (
    ALL_SELECTION_AXES,
    FMI_SELECTABLE_AXES,
    MANUAL_ONLY_AXES,
    ResolvedReplacementSelection,
    load_fmi_fingerprint,
    resolve_replacement_selection,
)
from dendritic_modeling.networks.architectures.replacement.teacher_init import (
    copy_topk_weight_,
)
from dendritic_modeling.networks.architectures.replacement.teacher_topology import (
    DEFAULT_TOPOLOGY_ROW_CHUNK_SIZE,
    LEGACY_GLOBAL_TEACHER_TOPOLOGY_METRICS,
    ROW_CONDITIONED_TEACHER_TOPOLOGY_METRICS,
    TEACHER_TOPOLOGY_METRICS,
    initialize_population_topology_from_targets_,
    initialize_population_topology_from_teacher_,
    supports_teacher_topology_initialization,
)

__all__ = [
    "ALL_SELECTION_AXES",
    "CANONICAL_FAMILIES",
    "DEFAULT_TOPOLOGY_ROW_CHUNK_SIZE",
    "FMI_SELECTABLE_AXES",
    "INTEGRATION_RULES",
    "LEGACY_FAMILY_ALIASES",
    "LEGACY_GLOBAL_TEACHER_TOPOLOGY_METRICS",
    "MANUAL_ONLY_AXES",
    "OUTPUT_TOPOLOGY_MODES",
    "ROW_CONDITIONED_TEACHER_TOPOLOGY_METRICS",
    "RUNTIME_TENSOR_CONTRACT_SCHEMA",
    "TEACHER_TOPOLOGY_METRICS",
    "TOPOLOGY_MODES",
    "ChannelMapReplacement",
    "CompiledReplacementPlan",
    "FlatAffineOutputAdapter",
    "LayerwiseTokenReplacementStack",
    "LinearOutputAdapter",
    "NonNegativeInputAdapter",
    "PopulationToFeatureReduction",
    "PositiveEICoreOutputAdapter",
    "PositiveEIOutputProjection",
    "PositiveFeatureScale",
    "PositiveThresholdReLU",
    "ResolvedReplacementSelection",
    "SelectedModuleReplacementRecord",
    "SpatialNonNegativeInputAdapter",
    "TokenReplacementCell",
    "TransformedLinear",
    "ZeroPadOutputAdapter",
    "apply_compiled_plan_to_config",
    "apply_compiled_plans_by_layer_to_config",
    "compile_replacement_candidate",
    "compiled_replacement_plan_from_mapping",
    "copy_topk_weight_",
    "initialize_population_topology_from_targets_",
    "initialize_population_topology_from_teacher_",
    "load_fmi_fingerprint",
    "make_linear_output_projection",
    "make_nonmixing_output_adapter",
    "make_output_projection",
    "make_positive_ei_output_projection",
    "output_projection_parameter_counts",
    "parse_candidate",
    "preserve_runtime_tensor_contract",
    "replace_modules_with_selected_population_networks",
    "require_runtime_tensor_contract",
    "resolve_replacement_selection",
    "supports_teacher_topology_initialization",
    "transformed_feature_dim",
    "validate_input_transform",
]

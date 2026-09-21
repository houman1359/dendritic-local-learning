"""Reusable analysis tools for weights, routing, dynamics, and E/I motifs."""

from .branch_local_sensitivity import BranchLocalSensitivityAnalyzer
from .causal_intervention import SameCheckpointCausalInterventionAnalyzer
from .class_accessibility import (
    ClassAccessibilityProbeAnalyzer,
    LinearClassAccessibilityAnalyzer,
    cross_validated_linear_decode,
    validation_to_test_linear_decode,
    validation_to_test_probe_decode,
)
from .class_information import (
    MatchedClassInformationAnalyzer,
    matched_support_class_information,
)
from .correlation_analysis import CorrelationAnalyzer
from .dendritic_timetraces import DendriticTimetracesAnalyzer
from .ei_connectivity_motifs import (
    LOCAL_RECURRENT_ROUTE_KEYS,
    ROUTE_SPECS,
    TARGETING_CLASS_ORDER,
    analyze_ei_connectivity_motifs_from_state_dict,
    extract_compartment_weights_from_state_dict,
    pair_stats_by_class,
    summarize_compartment_targeting_from_compartment_weights,
)
from .inhibitory_routing_comparison import (
    InhibitoryRoutingComparison,
    fixed_budget_inhibitory_routing_comparison,
)
from .inhibitory_specialization import InhibitorySpecializationAnalyzer
from .input_region_intervention import (
    ImageRegionSpec,
    InputRegionInterventionAnalyzer,
    InputRegionInterventionSettings,
)
from .jacobian_spectrum import JacobianSpectrumAnalyzer
from .layer_population_intervention import LocalInhibitoryPopulationInterventionAnalyzer
from .local_rule_components import LocalRuleComponentAnalyzer
from .path_matched_information import (
    PathMatchedInformationAnalyzer,
    estimate_path_matched_class_information,
)
from .path_matched_intervention import (
    PathMatchedInterventionAnalyzer,
    build_nested_path_dose_supports,
    build_nested_path_supports,
)
from .poisson_exposure_transfer import (
    POISSON_EXPOSURE_DRAW_SEED_RULE,
    POSTHOC_INTEGRATION_NOTE,
    PoissonExposureTransferAnalyzer,
    evaluate_poisson_exposure_accuracy,
    poisson_exposure_draw_seed,
    poisson_exposure_duration_key,
    validate_poisson_exposure_durations,
)
from .reactivation_dynamics import ReactivationDynamicsAnalyzer
from .representation_accessibility import (
    RepresentationAccessibilityAnalyzer,
    apply_balanced_countsketch,
    balanced_countsketch_projection,
    build_layer_representations,
    collect_layer_population_features,
    population_geometry,
    population_layer_inventory,
)
from .source_tuning import SourceTuningOptions, SourceTuningSupportAnalyzer
from .spike_trains import SpikeTrainAnalyzer
from .state_transition_jacobian import StateTransitionJacobianAnalyzer
from .synaptic_activation import SynapticActivationAnalyzer
from .synaptic_pruning import (
    SynapticPruningAnalyzer,
    integrate_pruning_with_turnover_analysis,
)
from .synaptic_turnover import SynapticTurnoverAnalyzer
from .vision_boundary_diagnostics import VisionBoundaryDiagnosticsAnalyzer
from .weight import WeightAnalyzer

__all__ = [
    "LOCAL_RECURRENT_ROUTE_KEYS",
    "POISSON_EXPOSURE_DRAW_SEED_RULE",
    "POSTHOC_INTEGRATION_NOTE",
    "ROUTE_SPECS",
    "TARGETING_CLASS_ORDER",
    "BranchLocalSensitivityAnalyzer",
    "ClassAccessibilityProbeAnalyzer",
    "CorrelationAnalyzer",
    "DendriticTimetracesAnalyzer",
    "ImageRegionSpec",
    "InhibitoryRoutingComparison",
    "InhibitorySpecializationAnalyzer",
    "InputRegionInterventionAnalyzer",
    "InputRegionInterventionSettings",
    "JacobianSpectrumAnalyzer",
    "LinearClassAccessibilityAnalyzer",
    "LocalInhibitoryPopulationInterventionAnalyzer",
    "LocalRuleComponentAnalyzer",
    "MatchedClassInformationAnalyzer",
    "PathMatchedInformationAnalyzer",
    "PathMatchedInterventionAnalyzer",
    "PoissonExposureTransferAnalyzer",
    "ReactivationDynamicsAnalyzer",
    "RepresentationAccessibilityAnalyzer",
    "SameCheckpointCausalInterventionAnalyzer",
    "SourceTuningOptions",
    "SourceTuningSupportAnalyzer",
    "SpikeTrainAnalyzer",
    "StateTransitionJacobianAnalyzer",
    "SynapticActivationAnalyzer",
    "SynapticPruningAnalyzer",
    "SynapticTurnoverAnalyzer",
    "VisionBoundaryDiagnosticsAnalyzer",
    "WeightAnalyzer",  # Now includes all weight analysis functionality
    "analyze_ei_connectivity_motifs_from_state_dict",
    "apply_balanced_countsketch",
    "balanced_countsketch_projection",
    "build_layer_representations",
    "build_nested_path_dose_supports",
    "build_nested_path_supports",
    "collect_layer_population_features",
    "cross_validated_linear_decode",
    "estimate_path_matched_class_information",
    "evaluate_poisson_exposure_accuracy",
    "extract_compartment_weights_from_state_dict",
    "fixed_budget_inhibitory_routing_comparison",
    "integrate_pruning_with_turnover_analysis",
    "matched_support_class_information",
    "pair_stats_by_class",
    "poisson_exposure_draw_seed",
    "poisson_exposure_duration_key",
    "population_geometry",
    "population_layer_inventory",
    "summarize_compartment_targeting_from_compartment_weights",
    "validate_poisson_exposure_durations",
    "validation_to_test_linear_decode",
    "validation_to_test_probe_decode",
]

"""Functional Morphology Inference (FMI) profiler estimators.

Implements the teacher-layer fingerprint estimators of the
dendritic_replacement paper: task-weighted output spectrum (soma count),
gradient support and intrinsic active rank (synapse count and basis),
support stability (sparsity class), sign consistency (E/I allocation), and
feature-interaction structure (branches and depth). Each estimator operates
on calibration activations and scalar latent-target closures; none requires
training the replacement.
"""

from dendritic_modeling.analysis.fmi.allocation import (
    AllocationState,
    enumerate_allocation_frontier,
    log_ratio_excess,
    select_quality_budget,
    select_resource_budget,
    selected_plans,
)
from dendritic_modeling.analysis.fmi.gain_load import (
    challenge_score_distribution,
    challenge_scores,
    gain_load_score,
    gain_sensitivity,
    paired_shunting_win_probability,
)
from dendritic_modeling.analysis.fmi.interactions import (
    infer_blocks,
    infer_tree,
    interaction_matrix,
    partition_modularity,
)
from dendritic_modeling.analysis.fmi.mechanism import (
    MECHANISMS,
    delta_shunt,
    fit_error_curve,
    fit_probe,
    mechanism_scores,
    multiplicative_advantage,
)
from dendritic_modeling.analysis.fmi.rank_validation import (
    DEFAULT_DIAGNOSTIC_ENERGY_TARGETS,
    nested_disjoint_split_indices,
    nested_group_disjoint_split_indices,
    rank_capacity_diagnostics,
    subspace_convergence_diagnostics,
    validate_rank_estimate,
)
from dendritic_modeling.analysis.fmi.shortlist_safety import (
    ADVISORY_SHORTLIST,
    AGGREGATE_EXPLORATORY,
    DECISION_LABELS,
    PAIRED_EVIDENCE,
    SAFETY_VETO_FALLBACK,
    LocalFitEvidence,
    SafetyVetoPolicy,
    apply_shortlist_safety_veto,
    build_local_fit_evidence,
    build_safety_veto_policy,
)
from dendritic_modeling.analysis.fmi.spectrum import (
    SPECTRUM_ESTIMATORS,
    SpectrumFit,
    fit_robust_task_weighted_spectrum,
    latent_target,
    n_min,
    robust_task_weighted_spectrum,
    spectrum_slope_beta,
    task_weighted_spectrum,
)
from dendritic_modeling.analysis.fmi.support import (
    attribution_energy,
    gradient_participation_ratio,
    gradient_second_moment,
    participation_ratio,
    sign_consistency,
    support_jaccard,
    support_size,
)

__all__ = [
    "ADVISORY_SHORTLIST",
    "AGGREGATE_EXPLORATORY",
    "DECISION_LABELS",
    "DEFAULT_DIAGNOSTIC_ENERGY_TARGETS",
    "MECHANISMS",
    "PAIRED_EVIDENCE",
    "SAFETY_VETO_FALLBACK",
    "SPECTRUM_ESTIMATORS",
    "AllocationState",
    "LocalFitEvidence",
    "SafetyVetoPolicy",
    "SpectrumFit",
    "apply_shortlist_safety_veto",
    "attribution_energy",
    "build_local_fit_evidence",
    "build_safety_veto_policy",
    "challenge_score_distribution",
    "challenge_scores",
    "delta_shunt",
    "enumerate_allocation_frontier",
    "fit_error_curve",
    "fit_probe",
    "fit_robust_task_weighted_spectrum",
    "gain_load_score",
    "gain_sensitivity",
    "gradient_participation_ratio",
    "gradient_second_moment",
    "infer_blocks",
    "infer_tree",
    "interaction_matrix",
    "latent_target",
    "log_ratio_excess",
    "mechanism_scores",
    "multiplicative_advantage",
    "n_min",
    "nested_disjoint_split_indices",
    "nested_group_disjoint_split_indices",
    "paired_shunting_win_probability",
    "participation_ratio",
    "partition_modularity",
    "rank_capacity_diagnostics",
    "robust_task_weighted_spectrum",
    "select_quality_budget",
    "select_resource_budget",
    "selected_plans",
    "sign_consistency",
    "spectrum_slope_beta",
    "subspace_convergence_diagnostics",
    "support_jaccard",
    "support_size",
    "task_weighted_spectrum",
    "validate_rank_estimate",
]

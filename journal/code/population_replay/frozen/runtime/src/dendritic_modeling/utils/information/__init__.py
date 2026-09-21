"""
Information analysis subpackage.

This subpackage provides unified information-theoretic analysis for dendritic networks,
including mutual information, conditional mutual information, and partial information decomposition.
Uses multivariate methods from information.py as the core computation engine.
"""

# Core computation functions (multivariate-aware)
from dendritic_modeling.utils.information.nearest_neighbors import (
    build_sr_matrix,
    compute_cmi_ccc,
    compute_ei_mi,
    compute_mi_cc,
    compute_mi_cd,
    compute_pairwise_sr,
    compute_pairwise_sr_with_mi,
    compute_synaptic_information,
)

# PID utilities
from dendritic_modeling.utils.information.pid import (
    check_pid_available,
    pid_synergy_redundancy,
)

__all__ = [
    "build_sr_matrix",
    "check_pid_available",
    "compute_cmi_ccc",
    "compute_ei_mi",
    "compute_mi_cc",
    "compute_mi_cd",
    "compute_pairwise_sr",
    "compute_pairwise_sr_with_mi",
    "compute_synaptic_information",
    "pid_synergy_redundancy",
]

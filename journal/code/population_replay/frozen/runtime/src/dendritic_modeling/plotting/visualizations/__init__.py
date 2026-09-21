"""Visualization plotting functions for analysis, networks, and correlations."""

# Analysis visualizations (updated for modular structure)
# Data extraction utilities
from ..core.utils import dendritic_activations_todict, einet_activations_todict
from .correlation_plots import (
    plot_correlation_comparisons,
    plot_correlation_matrices,
    plot_hierarchical_correlations,
    plot_synaptic_correlations,
    plot_tuning_curves,
)
from .general_analysis_plots import (  # Training and sweep functions
    plot_ablation_results,
    plot_active_synapses_per_branch,
    plot_active_synapses_summary_by_layer,
    plot_branch_sweep_results,
    plot_ei_sweep_results,
    plot_layer_contributions,
    plot_loss_curves,
    plot_noise_sweep_branch_results,
    plot_noise_sweep_ei_results,
    plot_per_layer_ablation,
    plot_performance_evolution,
    plot_performance_metrics,
    plot_pruning_performance_comparison,
    plot_synapse_turnover,
    plot_weight_distributions,
)
from .information_plots import (
    plot_fully_aggregated_analysis,
    plot_info_vs_ei_depth_per_dendritic_layer,
    plot_layer_wise_aggregated_analysis,
    plot_lda_vs_trained_performance,
    plot_network_wise_aggregated_analysis,
    plot_per_layer_information,
    plot_structured_ei_network_analysis,
)

# Network visualizations module exists but all functions have been removed as unused

__all__ = [
    # Data utilities
    "dendritic_activations_todict",
    "einet_activations_todict",
    # General analysis
    "plot_ablation_results",
    "plot_active_synapses_per_branch",
    "plot_active_synapses_summary_by_layer",
    "plot_branch_sweep_results",
    "plot_correlation_comparisons",
    "plot_correlation_matrices",
    "plot_ei_sweep_results",
    "plot_fully_aggregated_analysis",
    # Correlation analysis
    "plot_hierarchical_correlations",
    "plot_info_vs_ei_depth_per_dendritic_layer",
    # Information analysis (now modular)
    "plot_layer_contributions",
    "plot_layer_wise_aggregated_analysis",
    "plot_lda_vs_trained_performance",
    # Training/sweep functions
    "plot_loss_curves",
    "plot_network_wise_aggregated_analysis",
    "plot_noise_sweep_branch_results",
    "plot_noise_sweep_ei_results",
    "plot_per_layer_ablation",
    "plot_per_layer_information",
    "plot_performance_evolution",
    "plot_performance_metrics",
    "plot_pruning_performance_comparison",
    "plot_structured_ei_network_analysis",
    "plot_synapse_turnover",
    "plot_synaptic_correlations",
    "plot_tuning_curves",
    "plot_weight_distributions",
]

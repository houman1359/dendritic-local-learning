"""
Plotting package for dendritic modeling.

This package contains visualization and plotting utilities organized into logical modules:
- core: Core utilities and helper functions
- visualizations: Specialized plotting functions for analysis, networks, training, and correlations
- managers: Centralized plotting orchestration and management

The reorganized structure provides better organization and maintainability.
"""

# Publication-grade style and deterministic vector export
from .publication import (
    ADD_STYLE,
    COLORS,
    E_STYLE,
    ERRBAR,
    FIGURE_EXPORT_CREATOR,
    FIGURE_SVG_HASHSALT,
    I_STYLE,
    NET_STYLE,
    SH_STYLE,
    SHADE_ALPHA,
    PaperColors,
    clean_axis,
    inline_label,
    missing_data_panel,
    panel_label,
    plot_with_band,
    save_figure,
    setup_paper_style,
    slim_colorbar,
    vector_colorbar,
    vector_heatmap,
)

# Import from visualizations
from .visualizations import (  # Analysis functions; Training functions
    dendritic_activations_todict,
    einet_activations_todict,
    plot_ablation_results,
    plot_active_synapses_per_branch,
    plot_branch_sweep_results,
    plot_ei_sweep_results,
    plot_layer_contributions,
    plot_loss_curves,
    plot_noise_sweep_branch_results,
    plot_noise_sweep_ei_results,
    plot_per_layer_information,
    plot_performance_evolution,
    plot_performance_metrics,
    plot_synapse_turnover,
    plot_weight_distributions,
)

__all__ = [
    "ADD_STYLE",
    "COLORS",
    "ERRBAR",
    "E_STYLE",
    "FIGURE_EXPORT_CREATOR",
    "FIGURE_SVG_HASHSALT",
    "I_STYLE",
    "NET_STYLE",
    "SHADE_ALPHA",
    "SH_STYLE",
    "PaperColors",
    # Analysis functions
    "clean_axis",
    "dendritic_activations_todict",
    "einet_activations_todict",
    "inline_label",
    "missing_data_panel",
    "panel_label",
    "plot_ablation_results",
    "plot_active_synapses_per_branch",
    # Training functions
    "plot_branch_sweep_results",
    "plot_ei_sweep_results",
    "plot_layer_contributions",
    "plot_loss_curves",
    "plot_noise_sweep_branch_results",
    "plot_noise_sweep_ei_results",
    "plot_per_layer_information",
    "plot_performance_evolution",
    "plot_performance_metrics",
    "plot_synapse_turnover",
    "plot_weight_distributions",
    "plot_with_band",
    "save_figure",
    "setup_paper_style",
    "slim_colorbar",
    "vector_colorbar",
    "vector_heatmap",
]

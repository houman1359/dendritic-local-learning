"""
Synaptic Weight Analysis Plotting Functions.

This module provides plotting functions for synaptic weight expectation analysis,
following the same structure and style as information analysis plots.
"""

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from dendritic_modeling.plotting.visualizations.plotting_utils import (
    convert_layer_names,
    create_figure,
    get_color_scheme,
    setup_basic_plot,
)

logger = logging.getLogger(__name__)


def plot_synaptic_weight_analysis(
    layer_statistics: dict[str, Any],
    save_path: str,
    somatic_synapses: bool = True,
    logger: Optional[logging.Logger] = None,
):
    """
    Plot synaptic weight expectation analysis results.

    Creates plots showing mean and variance of excitatory and inhibitory
    synaptic weights across dendritic layers (soma to distal).

    Args:
        layer_statistics: Dictionary with layer-wise weight statistics
        save_path: Directory to save plots
        somatic_synapses: Whether somatic synapses are enabled
        logger: Optional logger for debugging
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Generating synaptic weight expectation plots...")

    try:
        # Convert layer names to uniform format (soma to distal)
        converted_results = convert_layer_names(
            layer_statistics, somatic_synapses=somatic_synapses
        )
        if not converted_results:
            logger.warning("No layer results after conversion")
            return

        layer_names = list(converted_results.keys())
        logger.info(f"Plotting synaptic weight analysis for layers: {layer_names}")

        # Define weight metrics to plot (currently unused but kept for future reference)
        # weight_metrics = [
        #     ("exc_weight_mean", "Excitatory Weight Mean", "Weight Mean"),
        #     ("exc_weight_var", "Excitatory Weight Variance", "Weight Variance"),
        #     ("inh_weight_mean", "Inhibitory Weight Mean", "Weight Mean"),
        #     ("inh_weight_var", "Inhibitory Weight Variance", "Weight Variance"),
        # ]

        # Create plots for total synaptic strength (active_synapses x expected_weight)
        _plot_total_synaptic_strength(converted_results, layer_names, save_path, logger)

        # Create comparison plots (exc vs inh)
        _plot_weight_comparisons(converted_results, layer_names, save_path, logger)

        # Create active synapse count plots
        _plot_active_synapse_counts(converted_results, layer_names, save_path, logger)

        logger.info("Synaptic weight analysis plots completed successfully!")

    except Exception as e:
        logger.error(f"Error generating synaptic weight plots: {e}")
        import traceback

        traceback.print_exc()


def _plot_total_synaptic_strength(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
    logger: logging.Logger,
):
    """Plot total synaptic strength (active_synapse_count x expected_weight) across layers."""
    colors = get_color_scheme()

    _, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Collect data for both excitatory and inhibitory
    depths = []
    exc_strengths = []
    inh_strengths = []
    exc_errors = []
    inh_errors = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        # Calculate excitatory total strength = n_active_synapses x weight_mean
        exc_count = layer_data.get("exc_n_active_synapses_mean", 0)
        exc_weight = layer_data.get("exc_weight_mean_mean", 0)
        exc_strength = exc_count * exc_weight

        # Calculate inhibitory total strength = n_active_synapses x weight_mean
        inh_count = layer_data.get("inh_n_active_synapses_mean", 0)
        inh_weight = layer_data.get("inh_weight_mean_mean", 0)
        inh_strength = inh_count * inh_weight

        depths.append(layer_name)
        exc_strengths.append(exc_strength)
        inh_strengths.append(inh_strength)

        # Estimate errors using error propagation
        # sigma(n*w) ≈ sqrt[(sigma_n*w)^2 + (n*sigma_w)^2]
        exc_count_std = layer_data.get("exc_n_active_synapses_std", 0)
        exc_weight_std = layer_data.get("exc_weight_mean_std", 0)
        exc_strength_std = np.sqrt(
            (exc_count_std * exc_weight) ** 2 + (exc_count * exc_weight_std) ** 2
        )
        exc_errors.append(exc_strength_std)

        inh_count_std = layer_data.get("inh_n_active_synapses_std", 0)
        inh_weight_std = layer_data.get("inh_weight_mean_std", 0)
        inh_strength_std = np.sqrt(
            (inh_count_std * inh_weight) ** 2 + (inh_count * inh_weight_std) ** 2
        )
        inh_errors.append(inh_strength_std)

    if depths and (exc_strengths or inh_strengths):
        x_pos = np.arange(len(depths))
        width = 0.35  # Width of bars for side-by-side comparison

        # Plot excitatory and inhibitory bars side by side (no error bars)
        ax.bar(
            x_pos - width / 2,
            exc_strengths,
            width,
            label="Excitatory",
            color=colors["excitatory"],
            alpha=0.7,
        )
        ax.bar(
            x_pos + width / 2,
            inh_strengths,
            width,
            label="Inhibitory",
            color=colors["inhibitory"],
            alpha=0.7,
        )

        setup_basic_plot(
            ax,
            "Total Synaptic Strength (Active Count x Expected Weight)",
            "Layer (Soma to Distal)",
            "Total Strength",
            grid=True,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)
        ax.legend()

    plt.tight_layout()

    # Save plot
    filename = os.path.join(save_path, "total_synaptic_strength.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved total synaptic strength plot: {filename}")


def _plot_weight_comparisons(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
    logger: logging.Logger,
):
    """Plot simplified excitatory vs inhibitory weight comparisons: mean and variance only."""
    colors = get_color_scheme()

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Collect data for simplified 2-panel plot
    depths = []
    exc_means = []
    inh_means = []
    exc_vars = []
    inh_vars = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        # Include all layers, using 0 as default if metrics don't exist (like soma)
        depths.append(layer_name)
        exc_means.append(layer_data.get("exc_weight_mean_mean", 0.0))
        inh_means.append(layer_data.get("inh_weight_mean_mean", 0.0))
        exc_vars.append(layer_data.get("exc_weight_var_mean", 0.0))
        inh_vars.append(layer_data.get("inh_weight_var_mean", 0.0))

    if depths:
        x_pos = np.arange(len(depths))
        width = 0.35

        # Panel 1: Mean weights (mean over branches, then mean over all branches)
        ax1 = axes[0]
        ax1.bar(
            x_pos - width / 2,
            exc_means,
            width,
            label="Excitatory",
            color=colors["excitatory"],
            alpha=0.7,
        )
        ax1.bar(
            x_pos + width / 2,
            inh_means,
            width,
            label="Inhibitory",
            color=colors["inhibitory"],
            alpha=0.7,
        )

        setup_basic_plot(
            ax1,
            "Mean Synaptic Weights",
            "Layer (Soma to Distal)",
            "Mean Weight",
            grid=True,
        )
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(depths, rotation=45)
        ax1.legend()

        # Panel 2: Variance (variance over branches, then mean over all variances)
        ax2 = axes[1]
        ax2.bar(
            x_pos - width / 2,
            exc_vars,
            width,
            label="Excitatory",
            color=colors["excitatory"],
            alpha=0.7,
        )
        ax2.bar(
            x_pos + width / 2,
            inh_vars,
            width,
            label="Inhibitory",
            color=colors["inhibitory"],
            alpha=0.7,
        )

        setup_basic_plot(
            ax2,
            "Synaptic Weight Variance",
            "Layer (Soma to Distal)",
            "Weight Variance",
            grid=True,
        )
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(depths, rotation=45)
        ax2.legend()

    plt.suptitle("Synaptic Weight Analysis: Mean and Variance", fontsize=16)
    plt.tight_layout()

    # Save plot
    filename = os.path.join(save_path, "synaptic_weight_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved synaptic weight comparison plot: {filename}")


def _plot_active_synapse_counts(
    converted_results: dict[str, Any],
    layer_names: list[str],
    save_path: str,
    logger: logging.Logger,
):
    """Plot active synapse counts across layers."""
    colors = get_color_scheme()

    _, ax = create_figure(figsize=(12, 6))

    depths = []
    exc_counts = []
    inh_counts = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        # Include all layers, using 0 as default if metrics don't exist (like soma)
        depths.append(layer_name)
        exc_counts.append(layer_data.get("exc_n_active_synapses_mean", 0))
        inh_counts.append(layer_data.get("inh_n_active_synapses_mean", 0))

    if depths:
        x_pos = np.arange(len(depths))
        width = 0.35

        ax.bar(
            x_pos - width / 2,
            exc_counts,
            width,
            label="Excitatory",
            color=colors["excitatory"],
            alpha=0.7,
        )
        ax.bar(
            x_pos + width / 2,
            inh_counts,
            width,
            label="Inhibitory",
            color=colors["inhibitory"],
            alpha=0.7,
        )

        setup_basic_plot(
            ax,
            "Active Synapse Counts by Layer",
            "Layer (Soma to Distal)",
            "Number of Active Synapses",
            grid=True,
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)
        ax.legend()

    # Save plot
    filename = os.path.join(save_path, "active_synapse_counts.png")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved active synapse counts plot: {filename}")


def plot_detailed_weight_distributions(
    layer_statistics: dict[str, Any],
    save_path: str,
    somatic_synapses: bool = True,
    logger: Optional[logging.Logger] = None,
):
    """
    Plot detailed weight distribution analysis with multiple visualizations.

    Similar to the comprehensive information analysis plots but for synaptic weights.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Generating detailed synaptic weight distribution plots...")

    try:
        # Convert layer names
        converted_results = convert_layer_names(
            layer_statistics, somatic_synapses=somatic_synapses
        )
        layer_names = list(converted_results.keys())

        # Create comprehensive weight analysis plots
        _, axes = plt.subplots(2, 3, figsize=(18, 12))

        # Plot 1: Excitatory weight means
        _plot_single_metric(
            axes[0, 0],
            converted_results,
            layer_names,
            "exc_weight_mean",
            "Excitatory Weight Means",
            "excitatory",
        )

        # Plot 2: Inhibitory weight means
        _plot_single_metric(
            axes[0, 1],
            converted_results,
            layer_names,
            "inh_weight_mean",
            "Inhibitory Weight Means",
            "inhibitory",
        )

        # Plot 3: Excitatory total synaptic strength
        _plot_total_synaptic_strength_single(
            axes[0, 2], converted_results, layer_names, "excitatory"
        )

        # Plot 4: Inhibitory total synaptic strength
        _plot_total_synaptic_strength_single(
            axes[1, 0], converted_results, layer_names, "inhibitory"
        )

        # Plot 5: Weight ratio (exc/inh)
        _plot_weight_ratios(axes[1, 1], converted_results, layer_names)

        # Plot 6: Combined weight magnitude
        _plot_combined_weights(axes[1, 2], converted_results, layer_names)

        plt.suptitle(
            "Comprehensive Synaptic Weight Analysis (Soma to Distal)", fontsize=16
        )
        plt.tight_layout()

        # Save comprehensive plot
        filename = os.path.join(save_path, "comprehensive_synaptic_weight_analysis.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved comprehensive synaptic weight analysis plot: {filename}")

    except Exception as e:
        logger.error(f"Error in detailed weight distribution plotting: {e}")


def _plot_total_synaptic_strength_single(
    ax, converted_results, layer_names, synapse_type
):
    """Plot total synaptic strength for a single synapse type in a subplot."""
    colors = get_color_scheme()

    depths = []
    strengths = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        if synapse_type == "excitatory":
            count = layer_data.get("exc_n_active_synapses_mean", 0)
            weight = layer_data.get("exc_weight_mean_mean", 0)
            color = colors["excitatory"]
            title = "Excitatory Total Strength"
        else:  # inhibitory
            count = layer_data.get("inh_n_active_synapses_mean", 0)
            weight = layer_data.get("inh_weight_mean_mean", 0)
            color = colors["inhibitory"]
            title = "Inhibitory Total Strength"

        strength = count * weight
        depths.append(layer_name)
        strengths.append(strength)

    if depths and strengths:
        x_pos = np.arange(len(depths))
        ax.bar(x_pos, strengths, color=color, alpha=0.7)

        setup_basic_plot(ax, title, "Layer", "Total Strength", grid=True)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)


def _plot_single_metric(
    ax, converted_results, layer_names, metric_key, title, color_key
):
    """Plot a single weight metric across layers."""
    colors = get_color_scheme()

    depths = []
    values = []
    errors = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]
        mean_key = f"{metric_key}_mean"
        std_key = f"{metric_key}_std"

        if mean_key in layer_data:
            depths.append(layer_name)
            values.append(layer_data[mean_key])
            errors.append(layer_data.get(std_key, 0.0))

    if depths and values:
        x_pos = np.arange(len(depths))

        ax.errorbar(
            x_pos,
            values,
            yerr=errors,
            marker="o",
            linewidth=2,
            markersize=6,
            color=colors[color_key],
            capsize=3,
        )

        ax.fill_between(
            x_pos,
            [v - e for v, e in zip(values, errors)],
            [v + e for v, e in zip(values, errors)],
            alpha=0.2,
            color=colors[color_key],
        )

        setup_basic_plot(ax, title, "Layer", "Weight Value", grid=True)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)


def _plot_weight_ratios(ax, converted_results, layer_names):
    """Plot excitatory/inhibitory weight ratios."""
    colors = get_color_scheme()

    depths = []
    ratios = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        exc_mean = layer_data.get("exc_weight_mean_mean", 0)
        inh_mean = layer_data.get("inh_weight_mean_mean", 0)

        if exc_mean > 0 and inh_mean > 0:
            depths.append(layer_name)
            ratios.append(exc_mean / inh_mean)
        elif exc_mean > 0:  # Only excitatory weights
            depths.append(layer_name)
            ratios.append(exc_mean)  # Show excitatory value when no inhibitory

    if depths and ratios:
        x_pos = np.arange(len(depths))

        ax.plot(
            x_pos,
            ratios,
            marker="o",
            linewidth=2,
            markersize=6,
            color=colors["combined"],
        )

        setup_basic_plot(
            ax, "Excitatory/Inhibitory Weight Ratio", "Layer", "Ratio", grid=True
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)
        ax.axhline(
            y=1.0, color="gray", linestyle="--", alpha=0.5, label="Equal weights"
        )
        ax.legend()


def _plot_combined_weights(ax, converted_results, layer_names):
    """Plot combined weight magnitudes."""
    colors = get_color_scheme()

    depths = []
    combined_weights = []

    for layer_name in layer_names:
        layer_data = converted_results[layer_name]

        exc_mean = layer_data.get("exc_weight_mean_mean", 0)
        inh_mean = layer_data.get("inh_weight_mean_mean", 0)

        depths.append(layer_name)
        combined_weights.append(exc_mean + inh_mean)

    if depths and combined_weights:
        x_pos = np.arange(len(depths))

        ax.plot(
            x_pos,
            combined_weights,
            marker="o",
            linewidth=2,
            markersize=6,
            color=colors["primary"],
        )

        setup_basic_plot(
            ax, "Combined Weight Magnitude", "Layer", "Total Weight", grid=True
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(depths, rotation=45)


__all__ = ["plot_detailed_weight_distributions", "plot_synaptic_weight_analysis"]

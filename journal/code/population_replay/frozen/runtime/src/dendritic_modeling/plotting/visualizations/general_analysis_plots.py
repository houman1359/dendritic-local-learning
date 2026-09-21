"""Plotting functions for analysis results and performance metrics."""

import glob
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plotting_utils import (
    convert_layer_names,
    create_figure,
    create_subplots,
    handle_plot_error,
    save_plot,
    setup_basic_plot,
)

logger = logging.getLogger(__name__)


def plot_layer_contributions(results: dict[str, Any], **kwargs):
    """Plot layer contributions showing how each layer contributes to model performance."""
    logger.info("Plotting layer contributions")
    try:
        fig, ax = create_figure(figsize=(10, 6))

        if results and isinstance(results, dict):
            # Extract layer names and their contribution values
            layers = []
            contributions = []

            for layer_name, metrics in results.items():
                layers.append(layer_name)
                # Try to extract a meaningful contribution metric
                if isinstance(metrics, dict):
                    # Look for common contribution metrics
                    if "contribution" in metrics:
                        contributions.append(metrics["contribution"])
                    elif "accuracy_drop" in metrics:
                        contributions.append(metrics["accuracy_drop"])
                    elif "performance_drop" in metrics:
                        contributions.append(metrics["performance_drop"])
                    else:
                        # Use first numeric value found
                        for _key, value in metrics.items():
                            if isinstance(value, (int, float)):
                                contributions.append(value)
                                break
                        else:
                            contributions.append(0.0)
                elif isinstance(metrics, (int, float)):
                    contributions.append(metrics)
                else:
                    contributions.append(0.0)

            if layers and contributions:
                # Create bar plot
                bars = ax.bar(range(len(layers)), contributions)
                ax.set_xticks(range(len(layers)))
                ax.set_xticklabels(layers, rotation=45, ha="right")
                ax.set_ylabel("Contribution")
                ax.set_title("Layer Contributions to Model Performance")

                # Add value labels on bars
                for bar, value in zip(bars, contributions):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{value:.3f}",
                        ha="center",
                        va="bottom",
                    )

                plt.tight_layout()
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No layer contribution data available",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Layer Contributions")
        else:
            ax.text(
                0.5,
                0.5,
                "No data provided",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Layer Contributions")

        save_path = kwargs.get("save_path")
        filename = kwargs.get("filename", "layer_contributions")
        if save_path:
            save_plot(fig, save_path, filename)

        return fig, ax
    except Exception as e:
        return handle_plot_error("layer contribution plotting", e)


def plot_ablation_results(results: dict, **kwargs):
    """Plot ablation study results showing performance impact of removing components."""
    logger.info("Plotting ablation analysis results")
    try:
        fig, ax = create_figure(figsize=(10, 6))
        if results:
            # Handle nested structure from ablation analysis
            if isinstance(next(iter(results.values())), dict) and "depth" not in next(
                iter(results.values())
            ):
                # This is the nested structure: ablation_type -> layer -> metrics
                # Flatten to show average performance drop per ablation type
                keys = []
                values = []
                for ablation_type, layer_data in results.items():
                    keys.append(ablation_type.replace("_", " ").title())
                    # Calculate average accuracy drop across layers
                    acc_drops = []
                    for _layer_name, metrics in layer_data.items():
                        if "accuracy_drop" in metrics:
                            drop_val = metrics["accuracy_drop"]
                            # Ensure we have a scalar value
                            if isinstance(drop_val, (list, np.ndarray)):
                                drop_val = np.mean(drop_val) if len(drop_val) > 0 else 0
                            elif hasattr(drop_val, "item"):  # torch tensor
                                drop_val = drop_val.item()
                            acc_drops.append(float(drop_val))
                    avg_drop = np.mean(acc_drops) if acc_drops else 0
                    values.append(avg_drop)
            else:
                # Simple structure: key -> value
                keys = list(results.keys())
                values = [
                    (
                        next(iter(result.values()))
                        if isinstance(result, dict) and result
                        else result if isinstance(result, (int, float)) else 0
                    )
                    for result in results.values()
                ]

            ax.bar(keys, values)
            setup_basic_plot(
                ax,
                "Ablation Study Results",
                "Ablation Type",
                "Average Performance Drop",
            )
            ax.tick_params(axis="x", rotation=45)
        else:
            ax.text(
                0.5,
                0.5,
                "Ablation Results\n(No data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Ablation Results")

        save_path = kwargs.get("save_path")
        filename = kwargs.get("filename", "ablation_results")
        if save_path:
            save_plot(fig, save_path, filename)

        return fig, ax
    except Exception as e:
        return handle_plot_error("ablation plotting", e)


def plot_layer_ablation(
    layer_results: dict,
    baselines: dict,
    save_path: Optional[str] = None,
    filename: str = "final",
    somatic_synapses: bool = True,
) -> Optional[list[plt.Figure]]:
    """
    Plot layer ablation results showing accuracy/log-likelihood decrease vs depth.

    Args:
        layer_results: Dictionary containing ablation results by type and layer
        save_path: Path to save plots. If None, returns list of figures
        filename: Base filename for saved plots

    Returns:
        List of figures if save_path is None, otherwise None
    """
    if save_path is None:
        fig_list = []

    metric_mapping = {
        "accuracy": "Accuracy",
        "auc": "AUC",
        "categorical_loglikelihood": "Log Likelihood",
        "mse": "MSE",
        "cosine_similarity": "Cosine Similarity",
        "pred_label_mi_bits": "I(C; Ĉ) [bits]",
    }

    def has_metric_nested(dictionary, metric):
        """
        Check if key exists at any level in a nested dictionary
        """
        if isinstance(dictionary, dict):
            key = f"{metric}_drop"
            if key in dictionary:
                return True
            return any(
                has_metric_nested(value, metric) for value in dictionary.values()
            )
        elif isinstance(dictionary, (list, tuple)):
            return any(has_metric_nested(item, metric) for item in dictionary)
        return False

    metrics = []
    for metric in metric_mapping.keys():
        if has_metric_nested(layer_results, metric):
            metrics.append(metric)

    base_targets = ["excitation", "inhibition", "all_synapses", "upstream"]
    method_to_targets = {
        method: targets
        for method, targets in layer_results.items()
        if isinstance(targets, dict)
    }

    if not method_to_targets:
        logger.warning("No recognizable ablation keys found for plotting")
        return fig_list if save_path is None else None

    def _method_sort_key(m: str) -> tuple[int, str]:
        return (0 if m == "lesion" else 1, m)

    for method in sorted(method_to_targets.keys(), key=_method_sort_key):
        target_results = method_to_targets[method]
        bases_present = [b for b in base_targets if b in target_results]
        if not bases_present:
            continue

        # Use consistent depth order across ablation types for this method.
        depths_set = set()
        for base in bases_present:
            type_dict: dict = target_results.get(base, {})
            for layer_dict in type_dict.values():
                if "depth" in layer_dict:
                    depths_set.add(layer_dict["depth"])
        depths = sorted(depths_set)
        if not depths:
            continue

        # Analyzer results use soma-relative depths: zero is the soma and
        # positive values are dendritic stages moving distally.  Do not infer
        # the soma from the minimum *observed* depth because synapse ablations
        # legitimately omit depth zero when somatic synapses are disabled.
        depth_labels = ["Soma" if depth == 0 else f"Distal {depth}" for depth in depths]

        width = 0.8 / max(len(bases_present), 1)

        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 8))

            for i, base in enumerate(bases_present):
                type_dict: dict = target_results.get(base, {})

                # depth -> list[values] (handle duplicates safely)
                depth_to_vals: dict[int, list[float]] = {}
                for layer_dict in type_dict.values():
                    d = layer_dict.get("depth")
                    if d is None:
                        continue
                    v = layer_dict.get(f"{metric}_drop", 0.0)
                    if isinstance(v, list):
                        v = float(np.mean(v)) if v else 0.0
                    depth_to_vals.setdefault(int(d), []).append(float(v))

                decreases = [
                    float(np.mean(depth_to_vals.get(d, [0.0]))) for d in depths
                ]

                x = np.array(depths) + width * (i - (len(bases_present) - 1) / 2)
                ax.bar(x, decreases, width=width, label=base.replace("_", " ").title())

            ax.set_xticks(depths)
            ax.set_xticklabels(depth_labels, rotation=45)
            ax.set_xlabel("Branch Depth (Soma to Distal)")
            ax.set_ylabel(f"{metric_mapping[metric]} Drop")

            title = f"{metric_mapping[metric]} Drop vs. Branch Depth"
            title += f"\nMethod: {method.replace('_', ' ').title()}"
            title += f"\n(Baseline: {float(baselines.get(metric, 0.0)):.3f})"
            ax.set_title(title)
            ax.legend()
            ax.grid(True, axis="y", alpha=0.3)

            if save_path is not None:
                plot_path = os.path.join(save_path, f"{filename}_{method}_{metric}.png")
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()
            else:
                fig_list.append(fig)

    if save_path is None:
        return fig_list


def plot_per_layer_ablation(layer_results: dict[str, dict[str, dict]], **kwargs):
    """Plot comprehensive per-layer ablation results similar to information analysis."""
    logger.info("Per-layer ablation plotting - creating comprehensive visualizations")
    try:
        save_path = kwargs.get("save_path")
        filename_prefix = kwargs.get("filename_prefix", "ablation_per_layer")

        if not layer_results:
            logger.warning("No layer ablation results provided")
            return None

        # Convert layer names and organize by depth
        layer_data_by_depth = {}

        # Use convert_layer_names for consistency with information analysis
        # First restructure the data to work with convert_layer_names
        sample_ablation = next(iter(layer_results.values()))
        temp_layer_results = {}
        for layer_name, layer_data in sample_ablation.items():
            temp_layer_results[layer_name] = layer_data

        # Convert layer names using the standard function
        convert_layer_names(
            temp_layer_results, somatic_synapses=kwargs.get("somatic_synapses", True)
        )

        # Create mapping from original technical names to display names
        all_depths = set()
        for _ablation_type, ablation_data in layer_results.items():
            if ablation_data:
                for _layer_name, layer_metrics in ablation_data.items():
                    if "depth" in layer_metrics:
                        all_depths.add(layer_metrics["depth"])

        all_depths = sorted(all_depths)  # CORRECT: Lower depth = soma, higher = distal

        # Create consistent naming
        for ablation_type, ablation_data in layer_results.items():
            if not ablation_data:
                continue

            # Create a consistent structure for all depths
            layers_info = []
            for depth in all_depths:
                # Depth values are soma-relative, so a missing zero must not
                # cause the first dendritic stage to be displayed as soma.
                display_name = "Soma" if depth == 0 else f"Distal Layer {depth}"

                # Find the layer data for this depth
                layer_metrics = None
                for _layer_name, metrics in ablation_data.items():
                    if metrics.get("depth") == depth:
                        layer_metrics = metrics
                        break

                # If no data for this depth, create empty metrics
                if layer_metrics is None:
                    layer_metrics = {"depth": depth}

                layers_info.append(
                    {"depth": depth, "name": display_name, "metrics": layer_metrics}
                )

            layer_data_by_depth[ablation_type] = layers_info

        plots = []

        # Create separate plots for each metric type
        metric_types = ["accuracy_drop", "auc_drop", "categorical_loglikelihood_drop"]

        for metric_type in metric_types:
            if any(
                any(metric_type in layer["metrics"] for layer in layers)
                for layers in layer_data_by_depth.values()
            ):
                fig, ax = create_figure(figsize=(14, 8))

                # Plot data for each ablation type
                ablation_types = list(layer_data_by_depth.keys())
                n_layers = len(layer_data_by_depth[ablation_types[0]])
                x_pos = np.arange(n_layers) * 1.5  # Add spacing between layer groups
                width = 0.3  # Slightly wider bars for better visibility

                colors = ["red", "blue", "green", "orange", "purple"]

                for i, ablation_type in enumerate(ablation_types):
                    layers = layer_data_by_depth[ablation_type]
                    layer_names = [layer["name"] for layer in layers]
                    values = [layer["metrics"].get(metric_type, 0) for layer in layers]

                    bars = ax.bar(
                        x_pos + i * width,
                        values,
                        width,
                        label=ablation_type.replace("_", " ").title(),
                        color=colors[i % len(colors)],
                        alpha=0.8,
                    )

                    # Add value labels on bars
                    for bar, val in zip(bars, values):
                        if val > 0:
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                bar.get_height() + 0.001,
                                f"{val:.3f}",
                                ha="center",
                                va="bottom",
                                fontsize=9,
                            )

                # Customize plot
                ax.set_xlabel("Layer (Soma to Distal)", fontsize=12)
                ax.set_ylabel(f'{metric_type.replace("_", " ").title()}', fontsize=12)
                ax.set_title(
                    f'Ablation Effects: {metric_type.replace("_", " ").title()} by Layer',
                    fontsize=14,
                )
                ax.set_xticks(x_pos + width * (len(ablation_types) - 1) / 2)
                ax.set_xticklabels(layer_names)
                ax.legend()
                ax.grid(True, alpha=0.3)

                plots.append(fig)

                # Save individual plot
                if save_path:
                    clean_metric = metric_type.replace(" ", "_").replace(" drop", "")
                    save_plot(fig, save_path, f"{filename_prefix}_{clean_metric}")

        logger.info(f"Generated {len(plots)} ablation plots")
        return plots

    except Exception as e:
        logger.error(f"Error in per-layer ablation plotting: {e}")
        return handle_plot_error("per-layer ablation plotting", e)


def plot_pruning_performance_comparison(
    performance_data: dict,
    save_path: Optional[str] = None,
    filename: str = "pruning_performance_comparison",
) -> plt.Figure:
    """
    Create a comprehensive bar plot comparing performance before/after pruning.

    Args:
        performance_data: Dict containing performance metrics for different stages:
            {
                'before_pruning': {...},
                'best_performance': {...},
                'last_epoch': {...},
                'after_pruning': {...}
            }
        save_path: Directory to save the plot
        filename: Base filename for the plot

    Returns:
        matplotlib Figure object
    """
    try:
        logger.info("Creating pruning performance comparison plot")

        if not performance_data:
            logger.warning("No performance data provided")
            return None

        # Extract available metrics from the data
        all_metrics = set()
        stages = ["before_pruning", "best_performance", "last_epoch", "after_pruning"]
        available_stages = []

        for stage in stages:
            if performance_data.get(stage):
                available_stages.append(stage)
                all_metrics.update(performance_data[stage].keys())

        if not available_stages or not all_metrics:
            logger.warning("No valid performance data found")
            return None

        # Remove any non-numeric metrics
        numeric_metrics = []
        for metric in all_metrics:
            try:
                # Check if all values for this metric are numeric
                values = []
                for stage in available_stages:
                    if metric in performance_data[stage]:
                        val = performance_data[stage][metric]
                        if isinstance(val, (int, float)) and not np.isnan(val):
                            values.append(val)
                if values:  # Only include if we have at least one valid value
                    numeric_metrics.append(metric)
            except (KeyError, TypeError, ValueError):
                continue

        if not numeric_metrics:
            logger.warning("No numeric metrics found")
            return None

        # Create the plot
        n_metrics = len(numeric_metrics)
        n_stages = len(available_stages)

        fig, axes = create_subplots(1, n_metrics, figsize=(4 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]

        # Color scheme for different stages
        colors = {
            "before_pruning": "#e74c3c",  # Red
            "best_performance": "#2ecc71",  # Green
            "last_epoch": "#f39c12",  # Orange
            "after_pruning": "#3498db",  # Blue
        }

        # Stage labels for display
        stage_labels = {
            "before_pruning": "Before\nPruning",
            "best_performance": "Best\nPerformance",
            "last_epoch": "Last\nEpoch",
            "after_pruning": "After\nPruning",
        }

        np.arange(n_stages)
        width = 0.6

        for i, metric in enumerate(numeric_metrics):
            ax = axes[i]

            # Collect values for this metric across stages
            values = []
            stage_names = []
            bar_colors = []

            for stage in available_stages:
                if metric in performance_data[stage]:
                    val = performance_data[stage][metric]
                    if isinstance(val, (int, float)) and not np.isnan(val):
                        values.append(val)
                        stage_names.append(stage_labels[stage])
                        bar_colors.append(colors[stage])
                    else:
                        values.append(0)
                        stage_names.append(stage_labels[stage])
                        bar_colors.append("#95a5a6")  # Gray for missing data
                else:
                    values.append(0)
                    stage_names.append(stage_labels[stage])
                    bar_colors.append("#95a5a6")  # Gray for missing data

            # Create the bar plot
            bars = ax.bar(
                range(len(values)), values, width, color=bar_colors, alpha=0.8
            )

            # Add value labels on top of bars
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + max(values) * 0.01,
                        f"{value:.4f}",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

            # Formatting
            ax.set_xlabel("Training Stage")
            ax.set_ylabel(metric.replace("_", " ").title())
            ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
            ax.set_xticks(range(len(stage_names)))
            ax.set_xticklabels(stage_names, rotation=0, ha="center")
            ax.grid(True, alpha=0.3, axis="y")

            # Set y-axis limits for better visualization
            if values and max(values) > 0:
                ax.set_ylim(0, max(values) * 1.1)

        plt.suptitle("Performance Comparison: Training vs Pruning Effects", fontsize=16)
        plt.tight_layout()

        # Save the plot
        if save_path:
            save_plot(fig, save_path, filename)

        return fig

    except Exception as e:
        logger.error(f"Error in plot_pruning_performance_comparison: {e}")
        return handle_plot_error("pruning performance comparison plotting", e)


def plot_weight_distributions(weight_data: dict[str, list[np.ndarray]], **kwargs):
    """Plot weight distributions for different layer types."""
    logger.info("Plotting weight distributions")
    try:
        fig, ax = create_figure(figsize=(10, 6))

        if weight_data:
            for name, weights in weight_data.items():
                if weights:
                    # Flatten all weight arrays and plot histogram
                    flat_weights = np.concatenate([w.flatten() for w in weights])
                    ax.hist(flat_weights, bins=50, alpha=0.7, label=name)
            setup_basic_plot(ax, "Weight Distributions", "Weight Value", "Frequency")
            ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "Weight Distributions\n(No data)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Weight Distributions")

        save_path = kwargs.get("save_path")
        filename = kwargs.get("filename", "weight_distributions")
        if save_path:
            save_plot(fig, save_path, filename)

        return fig, ax
    except Exception as e:
        return handle_plot_error("weight distribution plotting", e)


def plot_synapse_turnover(
    turnover_results: dict,
    save_path: Optional[str] = None,
    filename: str = "synapse_turnover",
):
    """Generate plots for synaptic turnover analysis results."""
    try:
        # Validate input data
        if not turnover_results:
            logger.warning("Empty turnover results provided")
            return None

        logger.info(
            f"Plotting turnover results with keys: {list(turnover_results.keys())}"
        )

        # Check if we have the new format with incremental and cumulative data
        if "incremental" in turnover_results and "cumulative" in turnover_results:
            # New format with both incremental and cumulative data
            figures = []

            # Log data availability
            inc_data = turnover_results["incremental"]
            cum_data = turnover_results["cumulative"]
            n_snapshots = turnover_results.get("n_snapshots", 0)

            logger.info(f"Found {n_snapshots} snapshots for turnover analysis")
            logger.info(
                f"Incremental data keys: {list(inc_data.keys()) if inc_data else 'None'}"
            )
            logger.info(
                f"Cumulative data keys: {list(cum_data.keys()) if cum_data else 'None'}"
            )

            # Create incremental turnover plot (t vs t-1)
            fig1 = _plot_turnover_data(
                turnover_results["incremental"],
                "Incremental Synaptic Turnover (t vs t-1)",
                "incremental",
            )
            figures.append(fig1)

            # Create cumulative turnover plot (t vs t-0)
            fig2 = _plot_turnover_data(
                turnover_results["cumulative"],
                "Cumulative Synaptic Turnover (t vs t-0)",
                "cumulative",
            )
            figures.append(fig2)

            # Save both plots
            if save_path:
                if fig1:
                    save_plot(fig1, save_path, f"{filename}_incremental")
                if fig2:
                    save_plot(fig2, save_path, f"{filename}_cumulative")

            return figures

        else:
            # Original format - assume it's the old structure
            logger.warning("Using original turnover format - may not display correctly")
            return _plot_turnover_original_format(turnover_results, save_path, filename)

    except Exception as e:
        logger.error(f"Error in synapse turnover plotting: {e}")
        return handle_plot_error(
            "synapse turnover plotting", e, create_placeholder=False
        )


def _plot_turnover_data(turnover_data: dict, title: str, plot_type: str) -> plt.Figure:
    """Helper function to plot turnover data."""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot for excitatory connections
        if "exc" in turnover_data:
            _plot_single_turnover(
                axes[0, 0], turnover_data["exc"], f"Excitatory {title}"
            )

        # Plot for inhibitory connections
        if "inh" in turnover_data:
            _plot_single_turnover(
                axes[0, 1], turnover_data["inh"], f"Inhibitory {title}"
            )

        # Plot for both types combined
        if "both" in turnover_data:
            _plot_single_turnover(
                axes[1, 0], turnover_data["both"], f"Combined {title}"
            )

        # Summary statistics plot
        _plot_turnover_summary(axes[1, 1], turnover_data, plot_type)

        plt.tight_layout()
        return fig

    except Exception as e:
        logger.warning(f"Error plotting turnover data: {e}")
        return None


def _plot_single_turnover(ax: plt.Axes, data: dict, title: str):
    """Plot turnover data for a single type."""
    try:
        # Check if data has required keys and non-empty values
        required_keys = ["active", "stable", "transient"]
        if not all(key in data for key in required_keys):
            ax.text(
                0.5,
                0.5,
                f"Missing turnover data\nRequired: {required_keys}\nFound: {list(data.keys())}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Check if data arrays are not empty
        if not any(len(data[key]) > 0 for key in required_keys):
            ax.text(
                0.5,
                0.5,
                "Empty turnover data\nNo snapshots to compare",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Plot if we have data
        max_len = max(len(data[key]) for key in required_keys if len(data[key]) > 0)
        if max_len > 0:
            snapshots = list(range(max_len))

            # Define colors and styles for better visibility
            plot_styles = [
                ("active", "#2E8B57", "o", "Active", "-"),  # Sea green, solid line
                ("stable", "#4169E1", "s", "Stable", "--"),  # Royal blue, dashed line
                ("transient", "#DC143C", "^", "Transient", ":"),  # Crimson, dotted line
            ]

            plotted_any = False
            for key, color, marker, label, linestyle in plot_styles:
                if key in data and len(data[key]) > 0:
                    values = data[key][:max_len]  # Ensure same length
                    ax.plot(
                        snapshots[: len(values)],
                        values,
                        color=color,
                        marker=marker,
                        label=label,
                        linestyle=linestyle,
                        linewidth=2,
                        markersize=6,
                        alpha=0.8,
                    )
                    plotted_any = True

                    # Add text annotation for flat lines
                    if len(set(values)) == 1:  # All values are the same (flat line)
                        mid_x = len(values) // 2
                        ax.annotate(
                            f"{label}: {values[0]}",
                            xy=(mid_x, values[0]),
                            xytext=(10, 10),
                            textcoords="offset points",
                            fontsize=9,
                            alpha=0.7,
                            bbox={
                                "boxstyle": "round,pad=0.3",
                                "facecolor": color,
                                "alpha": 0.2,
                            },
                        )

            if plotted_any:
                ax.set_xlabel("Snapshot Number")
                ax.set_ylabel("Synapse Count")
                ax.set_title(title, fontweight="bold")
                ax.legend(loc="best", fontsize=9)
                ax.grid(True, alpha=0.3)

                # Improve axis formatting
                ax.tick_params(axis="both", which="major", labelsize=9)

                # Set y-axis to start from 0 if all values are positive
                y_min = min(
                    min(data[key]) for key in required_keys if len(data[key]) > 0
                )
                if y_min >= 0:
                    ax.set_ylim(bottom=0)

    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Turnover data\n(Error: {str(e)[:50]})",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


def _plot_turnover_summary(ax: plt.Axes, turnover_data: dict, plot_type: str):
    """Plot summary statistics for turnover."""
    try:
        summary_metrics = []
        labels = []
        colors = []

        color_map = {"exc": "green", "inh": "red", "both": "blue"}

        for conn_type in ["exc", "inh", "both"]:
            if conn_type in turnover_data:
                data = turnover_data[conn_type]

                # Use the actual data keys: active, stable, transient
                for metric_name in ["active", "stable", "transient"]:
                    if metric_name in data and len(data[metric_name]) > 0:
                        # Calculate final value and change from start
                        values = data[metric_name]
                        if len(values) > 1:
                            final_value = values[-1]
                            initial_value = values[0]
                            change = final_value - initial_value

                            summary_metrics.append(change)
                            labels.append(
                                f"{conn_type.upper()}\n{metric_name.capitalize()}\nChange"
                            )
                            colors.append(color_map[conn_type])

        if summary_metrics:
            bars = ax.bar(range(len(labels)), summary_metrics, color=colors, alpha=0.7)

            # Add value labels on bars
            for bar, value in zip(bars, summary_metrics):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height
                    + (abs(height) * 0.02 if height >= 0 else -abs(height) * 0.02),
                    f"{value:+.0f}",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=9,
                )

            ax.set_title(f"{plot_type.capitalize()} Net Change Summary")
            ax.set_ylabel("Net Change (Final - Initial)")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        else:
            ax.text(
                0.5,
                0.5,
                "No summary data available\n(No turnover data found)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

    except Exception as e:
        ax.text(
            0.5,
            0.5,
            f"Summary Error:\n{str(e)[:50]}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )


def _plot_turnover_original_format(
    turnover_results: dict, save_path: str, filename: str
):
    """Handle original turnover format."""
    try:
        figures = []

        for connection_type in ["exc", "inh", "both"]:
            if connection_type in turnover_results:
                fig = _plot_turnover_data(
                    {connection_type: turnover_results[connection_type]},
                    f"Synaptic Turnover - {connection_type.capitalize()}",
                    "original",
                )
                if fig:
                    figures.append(fig)
                    if save_path:
                        save_plot(fig, save_path, f"{filename}_{connection_type}")

        return figures

    except Exception as e:
        logger.warning(f"Error plotting original turnover format: {e}")
        return []


def plot_layer_synapse_statistics(
    layer_stats: dict,
    save_path: Optional[str] = None,
    filename: str = "layer_synapse_statistics",
):
    """Plot per-layer synapse type statistics (ee, ei, ie, ii)."""
    try:
        logger.info("Generating layer synapse statistics plots...")

        # Create figure with subplots for incremental and cumulative
        fig, (ax1, ax2) = create_subplots(1, 2, figsize=(16, 6))

        synapse_types = ["ee", "ei", "ie", "ii"]
        colors = ["blue", "red", "green", "orange"]

        # Plot incremental statistics
        if (
            "incremental" in layer_stats
            and "per_layer_avg" in layer_stats["incremental"]
        ):
            incremental_data = layer_stats["incremental"]["per_layer_avg"]
            x_positions = range(len(synapse_types))

            # Calculate averages for each synapse type
            avg_values = []
            for syn_type in synapse_types:
                if incremental_data.get(syn_type):
                    # Average across all time points
                    avg_values.append(np.mean(incremental_data[syn_type]))
                else:
                    avg_values.append(0)

            bars1 = ax1.bar(x_positions, avg_values, color=colors, alpha=0.7)
            ax1.set_title("Average Per-Layer Synapse Counts (Incremental)")
            ax1.set_xlabel("Synapse Type")
            ax1.set_ylabel("Average Count per Layer")
            ax1.set_xticks(x_positions)
            ax1.set_xticklabels([s.upper() for s in synapse_types])

            # Add value labels on bars
            for bar, value in zip(bars1, avg_values):
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                )

        # Plot cumulative statistics
        if "cumulative" in layer_stats and "per_layer_avg" in layer_stats["cumulative"]:
            cumulative_data = layer_stats["cumulative"]["per_layer_avg"]

            # Calculate averages for each synapse type
            avg_values_cum = []
            for syn_type in synapse_types:
                if cumulative_data.get(syn_type):
                    # Average across all time points
                    avg_values_cum.append(np.mean(cumulative_data[syn_type]))
                else:
                    avg_values_cum.append(0)

            bars2 = ax2.bar(x_positions, avg_values_cum, color=colors, alpha=0.7)
            ax2.set_title("Average Per-Layer Synapse Counts (Cumulative)")
            ax2.set_xlabel("Synapse Type")
            ax2.set_ylabel("Average Count per Layer")
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels([s.upper() for s in synapse_types])

            # Add value labels on bars
            for bar, value in zip(bars2, avg_values_cum):
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()

        if save_path:
            save_plot(fig, save_path, filename)

        return fig

    except Exception as e:
        return handle_plot_error("layer synapse statistics plotting", e)


def plot_performance_metrics(performance_data: dict, **kwargs):
    """Plot performance metrics as a bar chart."""
    try:
        fig, ax = create_figure(figsize=(10, 6))

        if performance_data:
            # Handle nested dictionary structure from PerformanceAnalyzer
            flattened_metrics = {}
            for metric_name, metric_data in performance_data.items():
                if isinstance(metric_data, dict):
                    # Handle nested structure (metric -> dataset -> value)
                    for dataset, value in metric_data.items():
                        if isinstance(value, (int, float)):
                            flattened_metrics[f"{metric_name}_{dataset}"] = value
                elif isinstance(metric_data, (int, float)):
                    # Handle direct values
                    flattened_metrics[metric_name] = metric_data

            if flattened_metrics:
                metrics = list(flattened_metrics.keys())
                values = list(flattened_metrics.values())
                ax.bar(metrics, values)
                setup_basic_plot(ax, "Performance Metrics", "Metric", "Value")
                ax.tick_params(axis="x", rotation=45)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No valid performance data found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Performance Metrics")
        else:
            ax.text(
                0.5,
                0.5,
                "No performance data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Performance Metrics")

        save_path = kwargs.get("save_path")
        filename = kwargs.get("filename", "performance_metrics")
        if save_path:
            save_plot(fig, save_path, filename)

        return fig, ax

    except Exception as e:
        return handle_plot_error("performance metrics plotting", e)


def plot_performance_evolution(
    performance_dir: str,
    save_path: Optional[str] = None,
    filename: str = "performance_evolution",
) -> dict[str, tuple[plt.Figure, plt.Axes]]:
    """Generate performance evolution plots from per-epoch performance files."""
    try:
        plots = {}

        # Look for performance files in the directory
        if not os.path.exists(performance_dir):
            logger.warning(f"Performance directory does not exist: {performance_dir}")
            return {}

        # Find all performance files (assuming JSON format with epoch data)
        perf_files = glob.glob(os.path.join(performance_dir, "*performance*.json"))

        if not perf_files:
            logger.warning(f"No performance files found in {performance_dir}")
            # Create placeholder plot
            fig, ax = create_figure(figsize=(12, 8))
            ax.text(
                0.5,
                0.5,
                "No performance data found in directory",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Performance Evolution Over Time")
            plots["evolution"] = (fig, ax)
            return plots

        # Load and parse performance data
        all_data = {}
        epochs = []

        for file_path in sorted(perf_files):
            try:
                with open(file_path) as f:
                    data = json.load(f)

                # Extract epoch number from filename or data
                if "epoch" in data:
                    epoch = data["epoch"]
                else:
                    # Try to extract from filename
                    match = re.search(r"epoch_(\d+)", file_path)
                    if match:
                        epoch = int(match.group(1))
                    else:
                        continue

                epochs.append(epoch)

                # Store metrics
                for metric, value in data.items():
                    if metric != "epoch" and isinstance(value, (int, float)):
                        if metric not in all_data:
                            all_data[metric] = []
                        all_data[metric].append(value)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Could not parse performance file {file_path}: {e}")
                continue

        if not all_data or not epochs:
            logger.warning("No valid performance data found")
            fig, ax = create_figure(figsize=(12, 8))
            ax.text(
                0.5,
                0.5,
                "No valid performance data found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Performance Evolution Over Time")
            plots["evolution"] = (fig, ax)
            return plots

        # Sort by epoch
        sorted_indices = sorted(range(len(epochs)), key=lambda i: epochs[i])
        epochs = [epochs[i] for i in sorted_indices]
        for metric in all_data:
            all_data[metric] = [all_data[metric][i] for i in sorted_indices]

        # Create evolution plot
        fig, ax = create_figure(figsize=(12, 8))

        # Plot each metric
        for metric, values in all_data.items():
            if len(values) == len(epochs):
                ax.plot(epochs, values, marker="o", label=metric, linewidth=2)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Performance Metric Value")
        ax.set_title("Performance Evolution Over Training")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plots["evolution"] = (fig, ax)

        if save_path:
            save_plot(fig, save_path, filename)

        return plots

    except Exception as e:
        logger.warning(f"Error in performance evolution plotting: {e}")
        return {}


# Training and Sweep Plotting Functions
def plot_loss_curves(
    train_losses: list[float],
    valid_losses: list[float],
    loss_name: str,
    save_dir: Optional[str] = None,
    filename_prefix: str = "",
    train_losses_base: Optional[list[float]] = None,
    train_losses_reg: Optional[list[float]] = None,
):
    """Plot training and validation loss curves with optional regularization breakdown."""
    try:
        fig, ax = create_figure(figsize=(12, 8))

        epochs = range(1, len(train_losses) + 1)

        # Plot main losses
        ax.plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss (Total)")

        if valid_losses:
            epochs_val = range(1, len(valid_losses) + 1)
            ax.plot(
                epochs_val, valid_losses, "r-", linewidth=2, label="Validation Loss"
            )

        # Plot separate components if available
        if train_losses_base is not None:
            ax.plot(
                epochs,
                train_losses_base,
                "b--",
                alpha=0.7,
                linewidth=1.5,
                label="Training Loss (Base)",
            )

        if train_losses_reg is not None and any(reg > 1e-6 for reg in train_losses_reg):
            ax.plot(
                epochs,
                train_losses_reg,
                "g:",
                alpha=0.8,
                linewidth=1.5,
                label="Regularization Loss",
            )

        setup_basic_plot(ax, f"{loss_name} Loss Curves", "Epoch", "Loss")
        ax.legend()

        # Add text box with final values if regularization is present
        if train_losses_base is not None and train_losses_reg is not None:
            final_total = train_losses[-1]
            final_base = train_losses_base[-1]
            final_reg = train_losses_reg[-1]
            final_valid = valid_losses[-1] if valid_losses else 0

            textstr = f"Final Values:\nTrain (Total): {final_total:.4f}\nTrain (Base): {final_base:.4f}\nReg: {final_reg:.4f}"
            if valid_losses:
                textstr += f"\nValid: {final_valid:.4f}"

            props = {"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8}
            ax.text(
                0.02,
                0.98,
                textstr,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=props,
            )

        if save_dir:
            save_plot(fig, save_dir, f"{filename_prefix}loss_curves")

        return fig, ax

    except Exception as e:
        return handle_plot_error("loss curves plotting", e)


def plot_branch_sweep_results(results_csv_path, split="test"):
    """Plot branch sweep results."""
    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific split
    split_df = results_df[results_df["split"] == split]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Branch Sweep Results ({split} split)", fontsize=16)

    # Plot accuracy vs number of branches
    axes[0, 0].plot(split_df["n_branches"], split_df["accuracy"], "o-", color="blue")
    setup_basic_plot(
        axes[0, 0], "Accuracy vs Number of Branches", "Number of Branches", "Accuracy"
    )

    # Plot loss vs number of branches
    axes[0, 1].plot(split_df["n_branches"], split_df["loss"], "o-", color="red")
    setup_basic_plot(
        axes[0, 1], "Loss vs Number of Branches", "Number of Branches", "Loss"
    )

    # Plot precision vs number of branches
    axes[1, 0].plot(split_df["n_branches"], split_df["precision"], "o-", color="green")
    setup_basic_plot(
        axes[1, 0], "Precision vs Number of Branches", "Number of Branches", "Precision"
    )

    # Plot recall vs number of branches
    axes[1, 1].plot(split_df["n_branches"], split_df["recall"], "o-", color="orange")
    setup_basic_plot(
        axes[1, 1], "Recall vs Number of Branches", "Number of Branches", "Recall"
    )

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(
        os.path.dirname(results_csv_path), f"branch_sweep_results_{split}.png"
    )
    save_plot(fig, os.path.dirname(results_csv_path), f"branch_sweep_results_{split}")

    logger.info("Plot saved to: %s", save_path_final)
    return fig, axes


def plot_ei_sweep_results(results_csv_path, ei_ratio_range=None, split="test"):
    """Plot E/I ratio sweep results."""

    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific split
    split_df = results_df[results_df["split"] == split]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"E/I Ratio Sweep Results ({split} split)", fontsize=16)

    # Plot accuracy vs E/I ratio
    axes[0, 0].plot(split_df["ei_ratio"], split_df["accuracy"], "o-", color="blue")
    setup_basic_plot(axes[0, 0], "Accuracy vs E/I Ratio", "E/I Ratio", "Accuracy")

    # Plot loss vs E/I ratio
    axes[0, 1].plot(split_df["ei_ratio"], split_df["loss"], "o-", color="red")
    setup_basic_plot(axes[0, 1], "Loss vs E/I Ratio", "E/I Ratio", "Loss")

    # Plot precision vs E/I ratio
    axes[1, 0].plot(split_df["ei_ratio"], split_df["precision"], "o-", color="green")
    setup_basic_plot(axes[1, 0], "Precision vs E/I Ratio", "E/I Ratio", "Precision")

    # Plot recall vs E/I ratio
    axes[1, 1].plot(split_df["ei_ratio"], split_df["recall"], "o-", color="orange")
    setup_basic_plot(axes[1, 1], "Recall vs E/I Ratio", "E/I Ratio", "Recall")

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(
        os.path.dirname(results_csv_path), f"ei_sweep_results_{split}.png"
    )
    save_plot(fig, os.path.dirname(results_csv_path), f"ei_sweep_results_{split}")

    logger.info("Plot saved to: %s", save_path_final)
    return fig, axes


def plot_noise_sweep_branch_results(results_csv_path, noise_type="gaussian"):
    """Plot noise sweep results for branch networks."""

    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific noise type
    noise_df = results_df[results_df["noise_type"] == noise_type]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Noise Sweep Results - {noise_type.title()} Noise (Branch Networks)",
        fontsize=16,
    )

    # Plot accuracy vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[0, 0].plot(
            branch_data["noise_level"],
            branch_data["accuracy"],
            "o-",
            label=f"Branches={n_branches}",
        )
    setup_basic_plot(axes[0, 0], "Accuracy vs Noise Level", "Noise Level", "Accuracy")
    axes[0, 0].legend()

    # Plot loss vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[0, 1].plot(
            branch_data["noise_level"],
            branch_data["loss"],
            "o-",
            label=f"Branches={n_branches}",
        )
    setup_basic_plot(axes[0, 1], "Loss vs Noise Level", "Noise Level", "Loss")
    axes[0, 1].legend()

    # Plot precision vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[1, 0].plot(
            branch_data["noise_level"],
            branch_data["precision"],
            "o-",
            label=f"Branches={n_branches}",
        )
    setup_basic_plot(axes[1, 0], "Precision vs Noise Level", "Noise Level", "Precision")
    axes[1, 0].legend()

    # Plot recall vs noise level
    for n_branches in noise_df["n_branches"].unique():
        branch_data = noise_df[noise_df["n_branches"] == n_branches]
        axes[1, 1].plot(
            branch_data["noise_level"],
            branch_data["recall"],
            "o-",
            label=f"Branches={n_branches}",
        )
    setup_basic_plot(axes[1, 1], "Recall vs Noise Level", "Noise Level", "Recall")
    axes[1, 1].legend()

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(
        os.path.dirname(results_csv_path),
        f"noise_sweep_branch_results_{noise_type}.png",
    )
    save_plot(
        fig,
        os.path.dirname(results_csv_path),
        f"noise_sweep_branch_results_{noise_type}",
    )

    logger.info("Plot saved to: %s", save_path_final)
    return fig, axes


def plot_noise_sweep_ei_results(results_csv_path, noise_type="gaussian"):
    """Plot noise sweep results for E/I networks."""

    # Read the results CSV
    results_df = pd.read_csv(results_csv_path)

    # Filter for the specific noise type
    noise_df = results_df[results_df["noise_type"] == noise_type]

    # Create figure and subplots
    fig, axes = create_subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        f"Noise Sweep Results - {noise_type.title()} Noise (E/I Networks)", fontsize=16
    )

    # Plot accuracy vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[0, 0].plot(
            ei_data["noise_level"],
            ei_data["accuracy"],
            "o-",
            label=f"E/I_var = {ei_ratio}",
        )
    setup_basic_plot(axes[0, 0], "Accuracy vs Noise Level", "Noise Level", "Accuracy")
    axes[0, 0].legend()

    # Plot loss vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[0, 1].plot(
            ei_data["noise_level"], ei_data["loss"], "o-", label=f"E/I_var = {ei_ratio}"
        )
    setup_basic_plot(axes[0, 1], "Loss vs Noise Level", "Noise Level", "Loss")
    axes[0, 1].legend()

    # Plot precision vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[1, 0].plot(
            ei_data["noise_level"],
            ei_data["precision"],
            "o-",
            label=f"E/I_var = {ei_ratio}",
        )
    setup_basic_plot(axes[1, 0], "Precision vs Noise Level", "Noise Level", "Precision")
    axes[1, 0].legend()

    # Plot recall vs noise level
    for ei_ratio in noise_df["ei_ratio"].unique():
        ei_data = noise_df[noise_df["ei_ratio"] == ei_ratio]
        axes[1, 1].plot(
            ei_data["noise_level"],
            ei_data["recall"],
            "o-",
            label=f"E/I_var = {ei_ratio}",
        )
    setup_basic_plot(axes[1, 1], "Recall vs Noise Level", "Noise Level", "Recall")
    axes[1, 1].legend()

    plt.tight_layout()

    # Save the plot
    save_path_final = os.path.join(
        os.path.dirname(results_csv_path), f"noise_sweep_ei_results_{noise_type}.png"
    )
    save_plot(
        fig, os.path.dirname(results_csv_path), f"noise_sweep_ei_results_{noise_type}"
    )

    logger.info("Plot saved to: %s", save_path_final)
    return fig, axes


def plot_active_synapses_per_branch(
    connectivity_stats: dict,
    save_path: Optional[str] = None,
    filename: str = "active_synapses_per_branch",
) -> plt.Figure:
    """
    Plot active E and I synapses per branch per layer.

    Args:
        connectivity_stats: Dictionary containing layer and branch statistics
        save_path: Directory to save the plot
        filename: Base filename for the plot

    Returns:
        matplotlib Figure object
    """
    try:
        # Extract data from connectivity stats
        layer_stats = connectivity_stats.get("layer_stats", {})
        branch_stats = connectivity_stats.get("branch_stats", {})

        if not layer_stats and not branch_stats:
            logger.warning(
                "No layer or branch statistics found - generating default empty plot"
            )
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(
                0.5,
                0.5,
                "No pruning statistics available\n(Small model or no synapses pruned)",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title("Active Synapses per Branch", fontsize=14, fontweight="bold")
            ax.set_xlabel("Layer")
            ax.set_ylabel("Number of Active Synapses")
            if save_path and filename:
                save_plot(fig, save_path, filename)
            return fig

        # Organize data by layer and branch
        layers_data = defaultdict(list)

        # Parse branch names to extract layer and branch indices
        for branch_name, stats in branch_stats.items():
            # Expected format: "layer_X_branch_Y"
            try:
                parts = branch_name.split("_")
                if len(parts) >= 4 and parts[0] == "layer" and parts[2] == "branch":
                    layer_idx = int(parts[1])
                    branch_idx = int(parts[3])

                    exc_active = stats.get("excitatory", {}).get("active", 0)
                    inh_active = stats.get("inhibitory", {}).get("active", 0)

                    layers_data[layer_idx].append(
                        {
                            "branch_idx": branch_idx,
                            "excitatory": exc_active,
                            "inhibitory": inh_active,
                            "total": exc_active + inh_active,
                        }
                    )
            except (ValueError, IndexError):
                logger.warning(f"Could not parse branch name: {branch_name}")
                continue

        # Sort data by layer and branch indices
        sorted_layers = sorted(layers_data.keys())
        for layer_idx in sorted_layers:
            layers_data[layer_idx].sort(key=lambda x: x["branch_idx"])

        # Create the plot
        n_layers = len(sorted_layers)
        if n_layers == 0:
            logger.warning("No valid layer data found for plotting")
            return None

        # Calculate layout
        n_cols = min(3, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Handle different subplot configurations
        if n_layers == 1:
            axes = [axes]  # Single subplot
        elif n_rows == 1 and n_cols > 1:
            axes = list(axes)  # Single row, multiple columns
        elif n_cols == 1 and n_rows > 1:
            axes = list(axes)  # Single column, multiple rows
        else:
            axes = axes.flatten()  # Multiple rows and columns

        # Plot data for each layer
        for plot_idx, layer_idx in enumerate(sorted_layers):
            ax = axes[plot_idx] if plot_idx < len(axes) else None
            if ax is None:
                continue

            layer_data = layers_data[layer_idx]

            # Extract data for plotting
            branch_indices = [d["branch_idx"] for d in layer_data]
            exc_counts = [d["excitatory"] for d in layer_data]
            inh_counts = [d["inhibitory"] for d in layer_data]

            # Create bar plot
            x_pos = np.arange(len(branch_indices))
            width = 0.35

            bars1 = ax.bar(
                x_pos - width / 2,
                exc_counts,
                width,
                label="Excitatory",
                color="red",
                alpha=0.7,
            )
            bars2 = ax.bar(
                x_pos + width / 2,
                inh_counts,
                width,
                label="Inhibitory",
                color="blue",
                alpha=0.7,
            )

            # Customize the plot
            ax.set_xlabel("Branch Index")
            ax.set_ylabel("Number of Active Synapses")
            ax.set_title(f"Layer {layer_idx}: Active Synapses per Branch")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(branch_indices)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar in bars1:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            for bar in bars2:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        # Hide unused subplots
        for plot_idx in range(n_layers, len(axes)):
            axes[plot_idx].set_visible(False)

        plt.tight_layout()

        # Add overall title
        fig.suptitle(
            "Active Excitatory and Inhibitory Synapses per Branch per Layer",
            fontsize=14,
            y=0.98,
        )

        # Save the plot
        if save_path:
            save_plot(fig, save_path, filename)
            logger.info(
                f"Active synapses per branch plot saved to {save_path}/{filename}"
            )

        return fig

    except Exception as e:
        logger.error(f"Error in plot_active_synapses_per_branch: {e}")
        return handle_plot_error("active synapses per branch plotting", e)


def plot_active_synapses_summary_by_layer(
    connectivity_stats: dict,
    save_path: Optional[str] = None,
    filename: str = "active_synapses_summary_by_layer",
) -> plt.Figure:
    """
    Plot mean and std of active synapses per layer for each synapse type.

    This function creates a plot showing:
    - X-axis: Layer number (L)
    - Y-axis: Number of active synapses
    - For each layer L: mean and std of active synapses across all branches in that layer
    - Separate plots for excitatory (EE), inhibitory (IE), and total synapses

    Args:
        connectivity_stats: Dictionary containing layer and branch statistics
        save_path: Directory to save the plot
        filename: Base filename for the plot

    Returns:
        matplotlib Figure object
    """
    try:
        # Extract data from connectivity stats
        layer_stats = connectivity_stats.get("layer_stats", {})

        logger.debug("Connectivity stats keys: %s", list(connectivity_stats.keys()))
        logger.debug("Layer stats keys: %s", list(layer_stats.keys()))

        if not layer_stats:
            logger.warning("No layer statistics found - generating default empty plot")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.text(
                0.5,
                0.5,
                "No layer statistics available\n(Small model or no synapses pruned)",
                ha="center",
                va="center",
                fontsize=12,
            )
            ax.set_title(
                "Active Synapses Summary by Layer", fontsize=14, fontweight="bold"
            )
            ax.set_xlabel("Layer")
            ax.set_ylabel("Number of Active Synapses")
            if save_path and filename:
                save_plot(fig, save_path, filename)
            return fig

        # Organize data by layer
        layers_summary = {}

        # Extract detailed branch data from layer_stats
        for layer_name, layer_data in layer_stats.items():
            # Extract layer index from name like "layer_0"
            try:
                layer_idx = int(layer_name.split("_")[1])
            except (ValueError, IndexError):
                logger.warning(f"Could not parse layer name: {layer_name}")
                continue

            branch_stats = layer_data.get("branch_stats", {})
            if not branch_stats:
                continue

            # Collect synapse counts for all branches in this layer
            exc_counts = []
            inh_counts = []
            total_counts = []

            for _branch_name, branch_data in branch_stats.items():
                exc_active = branch_data.get("excitatory", {}).get("active", 0)
                inh_active = branch_data.get("inhibitory", {}).get("active", 0)
                total_active = exc_active + inh_active

                exc_counts.append(exc_active)
                inh_counts.append(inh_active)
                total_counts.append(total_active)

            # Calculate statistics for this layer
            if exc_counts or inh_counts:
                layers_summary[layer_idx] = {
                    "excitatory": {
                        "mean": np.mean(exc_counts) if exc_counts else 0,
                        "std": np.std(exc_counts) if len(exc_counts) > 1 else 0,
                        "count": len(exc_counts),
                    },
                    "inhibitory": {
                        "mean": np.mean(inh_counts) if inh_counts else 0,
                        "std": np.std(inh_counts) if len(inh_counts) > 1 else 0,
                        "count": len(inh_counts),
                    },
                    "total": {
                        "mean": np.mean(total_counts) if total_counts else 0,
                        "std": np.std(total_counts) if len(total_counts) > 1 else 0,
                        "count": len(total_counts),
                    },
                }

        if not layers_summary:
            logger.warning("No valid layer data found for summary plotting")
            return None

        # Prepare data for plotting
        sorted_layers = sorted(layers_summary.keys())

        # Extract data arrays
        exc_means = [
            layers_summary[layer]["excitatory"]["mean"] for layer in sorted_layers
        ]
        exc_stds = [
            layers_summary[layer]["excitatory"]["std"] for layer in sorted_layers
        ]

        inh_means = [
            layers_summary[layer]["inhibitory"]["mean"] for layer in sorted_layers
        ]
        inh_stds = [
            layers_summary[layer]["inhibitory"]["std"] for layer in sorted_layers
        ]

        total_means = [
            layers_summary[layer]["total"]["mean"] for layer in sorted_layers
        ]
        total_stds = [layers_summary[layer]["total"]["std"] for layer in sorted_layers]

        # Create the plot with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        x_pos = np.array(sorted_layers)

        # Plot 1: Excitatory synapses
        ax1.errorbar(
            x_pos,
            exc_means,
            yerr=exc_stds,
            marker="o",
            capsize=5,
            capthick=2,
            color="red",
            ecolor="red",
            alpha=0.7,
            label="Excitatory (EE)",
        )
        ax1.set_xlabel("Layer Number")
        ax1.set_ylabel("Active Synapses (Mean ± Std)")
        ax1.set_title("Excitatory Synapses per Branch")
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(sorted_layers)

        # Add branch count annotations
        for i, layer_idx in enumerate(sorted_layers):
            branch_count = layers_summary[layer_idx]["excitatory"]["count"]
            ax1.annotate(
                f"n={branch_count}",
                xy=(layer_idx, exc_means[i]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Plot 2: Inhibitory synapses
        ax2.errorbar(
            x_pos,
            inh_means,
            yerr=inh_stds,
            marker="s",
            capsize=5,
            capthick=2,
            color="blue",
            ecolor="blue",
            alpha=0.7,
            label="Inhibitory (IE)",
        )
        ax2.set_xlabel("Layer Number")
        ax2.set_ylabel("Active Synapses (Mean ± Std)")
        ax2.set_title("Inhibitory Synapses per Branch")
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(sorted_layers)

        # Add branch count annotations
        for i, layer_idx in enumerate(sorted_layers):
            branch_count = layers_summary[layer_idx]["inhibitory"]["count"]
            ax2.annotate(
                f"n={branch_count}",
                xy=(layer_idx, inh_means[i]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        # Plot 3: Total synapses
        ax3.errorbar(
            x_pos,
            total_means,
            yerr=total_stds,
            marker="^",
            capsize=5,
            capthick=2,
            color="green",
            ecolor="green",
            alpha=0.7,
            label="Total (EE + IE)",
        )
        ax3.set_xlabel("Layer Number")
        ax3.set_ylabel("Active Synapses (Mean ± Std)")
        ax3.set_title("Total Synapses per Branch")
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks(sorted_layers)

        # Add branch count annotations
        for i, layer_idx in enumerate(sorted_layers):
            branch_count = layers_summary[layer_idx]["total"]["count"]
            ax3.annotate(
                f"n={branch_count}",
                xy=(layer_idx, total_means[i]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()

        # Add overall title
        fig.suptitle(
            "Active Synapses per Branch: Mean ± Std by Layer and Synapse Type",
            fontsize=14,
            y=1.02,
        )

        # Save the plot
        if save_path:
            save_plot(fig, save_path, filename)
            logger.info(f"Active synapses summary plot saved to {save_path}/{filename}")

        return fig

    except Exception as e:
        logger.error(f"Error in plot_active_synapses_summary_by_layer: {e}")
        return handle_plot_error("active synapses summary by layer plotting", e)

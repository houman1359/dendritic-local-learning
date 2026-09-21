"""
Clean modular information analysis plotting for dendritic modeling.

This module provides specialized plotting classes for information-theoretic analysis.
"""

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

# Import utilities
from .plotting_utils import (
    add_error_bars,
    annotate_bars,
    convert_layer_names,
    create_figure,
    create_heatmap,
    create_subplots,
    get_color_scheme,
    handle_plot_error,
    save_plot,
    setup_basic_plot,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================


class PlotConfig:
    """Configuration for plot appearance."""

    def __init__(self):
        self.figsize = (10, 6)
        self.dpi = 300
        self.fontsize = {"title": 16, "label": 14, "tick": 12, "legend": 12}
        self.colors = get_color_scheme()


# ============================================================================
# Base Plotter Class
# ============================================================================


class BasePlotter:
    """Base class for all plotters with shared functionality."""

    def __init__(self, config: PlotConfig = None):
        self.config = config or PlotConfig()

    def create_subplot(
        self, figsize: Optional[tuple[int, int]] = None
    ) -> tuple[plt.Figure, plt.Axes]:
        return create_figure(figsize or self.config.figsize)

    def create_subplots(
        self, rows: int, cols: int, figsize: Optional[tuple[int, int]] = None
    ) -> tuple[plt.Figure, np.ndarray]:
        return create_subplots(rows, cols, figsize or (15, 10))

    def save(self, fig: plt.Figure, save_path: str, filename: str):
        save_plot(fig, save_path, filename, dpi=self.config.dpi)

    def get_color(self, color_type: str) -> str:
        return self.config.colors.get(color_type, self.config.colors["primary"])


# ============================================================================
# Specialized Plotter Classes
# ============================================================================


class BasicMIPlotter(BasePlotter):
    """Handles basic MI plotting."""

    def plot(self, basic_mi: dict, ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            _, ax = self.create_subplot()

        metrics = {
            k: v
            for k, v in basic_mi.items()
            if not k.endswith("_normalized") and not k.endswith("_std")
        }

        if metrics:
            labels = list(metrics.keys())
            values = list(metrics.values())
            colors = [
                (
                    self.get_color("excitatory")
                    if "E" in label and "I" not in label
                    else (
                        self.get_color("inhibitory")
                        if "I" in label and "E" not in label
                        else self.get_color("combined")
                    )
                )
                for label in labels
            ]

            bars = ax.bar(labels, values, color=colors, alpha=0.8)
            setup_basic_plot(
                ax,
                "Basic Mutual Information with Class Labels",
                "MI Type",
                "MI (bits)",
                self.config.fontsize,
            )
            ax.tick_params(axis="x", rotation=45)
            annotate_bars(ax, bars, values)
        else:
            ax.text(
                0.5,
                0.5,
                "No basic MI metrics found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Basic Mutual Information")

        return ax


class PairwiseMIPlotter(BasePlotter):
    """Handles pairwise MI plotting."""

    def plot_heatmap(self, pairwise_mi: dict, ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            _, ax = self.create_subplot()

        # Start with base variables and add optional ones if present
        variables = ["E", "I"]
        if any("Vb" in key for key in pairwise_mi.keys()):
            variables.append("Vb")
        variables.append("Vout")
        if any("Vinf" in key for key in pairwise_mi.keys()):
            variables.append("Vinf")

        # Add LDA versions if present
        if any("E_lin" in key for key in pairwise_mi.keys()):
            variables.append("E_lin")
        if any("I_lin" in key for key in pairwise_mi.keys()):
            variables.append("I_lin")
        if any("Vb_lin" in key for key in pairwise_mi.keys()):
            variables.append("Vb_lin")

        n_vars = len(variables)
        mi_matrix = np.zeros((n_vars, n_vars))

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    for key in [f"I({var1};{var2})", f"I({var2};{var1})"]:
                        if key in pairwise_mi:
                            mi_matrix[i, j] = pairwise_mi[key]
                            break

        create_heatmap(
            ax,
            mi_matrix,
            variables,
            variables,
            "Pairwise Mutual Information Matrix",
            cmap="YlOrRd",
        )
        return ax


class PerLayerPlotter(BasePlotter):
    """Handles per-layer information plotting."""

    def plot_basic_mi_layers(
        self, layer_results: dict, layer_names: list, ax: plt.Axes
    ):
        # Base metrics - check what's available in the data
        possible_metrics = [
            "I(E;C)_mean",
            "I(I;C)_mean",
            "I(Vb;C)_mean",
            "I(Vout;C)_mean",
            "I(E_lin;C)_mean",
            "I(I_lin;C)_mean",
            "I(Vb_lin;C)_mean",
        ]
        possible_labels = [
            "I(E;C)",
            "I(I;C)",
            "I(Vb;C)",
            "I(Vout;C)",
            "I(E_lin;C)",
            "I(I_lin;C)",
            "I(Vb_lin;C)",
        ]

        # Filter to only metrics present in data
        metrics_to_plot = []
        metric_labels = []
        for metric, label in zip(possible_metrics, possible_labels):
            # Check if any layer has this metric
            if any(metric in layer_results.get(ln, {}) for ln in layer_names):
                metrics_to_plot.append(metric)
                metric_labels.append(label)

        # Fallback to original if none found
        if not metrics_to_plot:
            metrics_to_plot = ["I(E;C)_mean", "I(I;C)_mean", "I(Vout;C)_mean"]
            metric_labels = ["I(E;C)", "I(I;C)", "I(Vout;C)"]

        # Define colors matching the number of metrics
        color_map = {
            "I(E;C)": self.get_color("excitatory"),
            "I(I;C)": self.get_color("inhibitory"),
            "I(Vb;C)": "#FFB347",  # Orange for branch input
            "I(Vout;C)": self.get_color("combined"),
            "I(E_lin;C)": "#90EE90",  # Light green for E LDA
            "I(I_lin;C)": "#FFB6C1",  # Light pink for I LDA
            "I(Vb_lin;C)": "#FFD700",  # Gold for Vb LDA
        }
        colors = [color_map.get(label, "#808080") for label in metric_labels]

        x = np.arange(len(layer_names))
        width = 0.8 / len(metrics_to_plot)  # Dynamic width based on number of metrics

        for i, (metric, label, color) in enumerate(
            zip(metrics_to_plot, metric_labels, colors)
        ):
            values = []
            errors = []
            for layer_name in layer_names:
                values.append(layer_results[layer_name].get(metric, 0))
                std_key = metric.replace("_mean", "_std")
                errors.append(layer_results[layer_name].get(std_key, 0))

            bars = ax.bar(
                x + i * width,
                values,
                width,
                yerr=errors,
                label=label,
                color=color,
                alpha=0.8,
                capsize=4,
                error_kw={"linewidth": 1.5, "capthick": 1.5},
            )

            for bar, val, err in zip(bars, values, errors):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + err + 0.01,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        setup_basic_plot(
            ax,
            "Basic MI Across Layers (Mean ± Std)",
            "Layer (Soma to Distal)",
            "MI (bits)",
            self.config.fontsize,
        )
        ax.set_xticks(x + width)
        ax.set_xticklabels(layer_names, rotation=45)
        ax.legend(loc="upper right")

    def plot_info_flow_layers(
        self, layer_results: dict, layer_names: list, ax: plt.Axes
    ):
        layer_indices = np.arange(len(layer_names))
        vout_values = np.array(
            [layer_results[name].get("I(Vout;C)_mean", 0) for name in layer_names]
        )
        vout_errors = np.array(
            [layer_results[name].get("I(Vout;C)_std", 0) for name in layer_names]
        )

        add_error_bars(
            ax,
            layer_indices,
            vout_values,
            vout_errors,
            self.get_color("combined"),
            "I(Vout;C)",
            capsize=6,
        )

        ax.fill_between(
            layer_indices,
            vout_values - vout_errors,
            vout_values + vout_errors,
            alpha=0.2,
            color=self.get_color("combined"),
        )

        for i, (val, err) in enumerate(zip(vout_values, vout_errors)):
            if val > 0:
                ax.annotate(
                    f"{val:.3f}±{err:.3f}",
                    (layer_indices[i], val + err + 0.02),
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    bbox={
                        "boxstyle": "round,pad=0.3",
                        "facecolor": "white",
                        "alpha": 0.8,
                    },
                )

        setup_basic_plot(
            ax,
            "Information Flow Across Layers",
            "Layer Position (Soma to Distal)",
            "I(Vout;C) (bits)",
            self.config.fontsize,
        )
        ax.set_xticks(layer_indices)
        ax.set_xticklabels(layer_names, rotation=45)
        ax.legend()


# ============================================================================
# Main Information Analysis Plotter
# ============================================================================


class InformationAnalysisPlotter(BasePlotter):
    """Main plotter orchestrating information analysis plots."""

    def __init__(self, config: PlotConfig = None):
        super().__init__(config)
        self.basic_plotter = BasicMIPlotter(self.config)
        self.pairwise_plotter = PairwiseMIPlotter(self.config)
        self.layer_plotter = PerLayerPlotter(self.config)

    def plot_information_metrics(
        self,
        results: dict[str, Any],
        save_path: Optional[str] = None,
        filename_prefix: str = "information",
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        """Create comprehensive plots using specialized plotters."""
        plots = {}

        try:
            # Basic MI plot
            if "basic_mi" in results:
                fig, ax = self.create_subplot()
                self.basic_plotter.plot(results["basic_mi"], ax)
                plots["basic_mi"] = (fig, ax)

            # Pairwise MI heatmap
            if "pairwise_mi" in results:
                fig, ax = self.create_subplot()
                self.pairwise_plotter.plot_heatmap(results["pairwise_mi"], ax)
                plots["pairwise_mi"] = (fig, ax)

            # Comprehensive summary
            if any(key in results for key in ["basic_mi", "pairwise_mi"]):
                fig, axes = self._plot_comprehensive_summary(results)
                plots["summary"] = (fig, axes)

            return plots

        except Exception as e:
            return handle_plot_error("information analysis plotting", e)

    def plot_per_layer_information(
        self, layer_results: dict[str, dict[str, Any]], **kwargs
    ):
        """Plot detailed per-layer information with comprehensive analysis."""
        try:
            save_path = kwargs.get("save_path")
            filename_prefix = kwargs.get("filename_prefix", "per_layer_info")

            if not layer_results:
                logger.warning("No layer results provided")
                return None

            # Try to get somatic_synapses from kwargs
            somatic_synapses = kwargs.get("somatic_synapses", True)
            converted_results = convert_layer_names(
                layer_results, somatic_synapses=somatic_synapses
            )
            if not converted_results:
                converted_results = layer_results

            layer_names = list(converted_results.keys())
            logger.info(f"Plotting per-layer information for layers: {layer_names}")

            # Create comprehensive plots for all information types
            plots = []

            # 1. Basic MI vs Layer Depth (Mean + Variance)
            fig1 = self._plot_layer_depth_analysis(
                converted_results,
                layer_names,
                "basic_mi",
                "Basic Mutual Information vs Layer Depth",
            )
            plots.append(fig1)
            if save_path and fig1:
                self.save(fig1, save_path, f"{filename_prefix}_basic_mi_depth")

            # 2. Pairwise MI vs Layer Depth
            fig2 = self._plot_layer_depth_analysis(
                converted_results,
                layer_names,
                "pairwise_mi",
                "Pairwise Mutual Information vs Layer Depth",
            )
            plots.append(fig2)
            if save_path and fig2:
                self.save(fig2, save_path, f"{filename_prefix}_pairwise_mi_depth")

            # 3. Conditional MI vs Layer Depth
            fig3 = self._plot_layer_depth_analysis(
                converted_results,
                layer_names,
                "conditional_mi",
                "Conditional Mutual Information vs Layer Depth",
            )
            plots.append(fig3)
            if save_path and fig3:
                self.save(fig3, save_path, f"{filename_prefix}_conditional_mi_depth")

            # 4. Branch Variance Analysis
            fig4 = self._plot_branch_variance_analysis(converted_results, layer_names)
            plots.append(fig4)

            # 5. Original detailed plot for compatibility
            fig5, axes = self.create_subplots(2, 2, figsize=(16, 12))
            self.layer_plotter.plot_basic_mi_layers(
                converted_results, layer_names, axes[0, 0]
            )
            self.layer_plotter.plot_info_flow_layers(
                converted_results, layer_names, axes[1, 0]
            )
            self._plot_layer_heatmap(axes[1, 1], converted_results, layer_names)
            plt.suptitle("Per-Layer Information Analysis (Soma to Distal)", fontsize=16)
            plt.tight_layout()
            plots.append(fig5)

            return plots

        except Exception as e:
            logger.error(f"Error in per-layer information plotting: {e}")
            return handle_plot_error("per-layer information plotting", e)

    def _plot_comprehensive_summary(
        self, results: dict[str, Any]
    ) -> tuple[plt.Figure, np.ndarray]:
        """Create comprehensive summary plot."""
        fig, axes = self.create_subplots(2, 2, figsize=(15, 10))

        # Basic MI
        if "basic_mi" in results:
            self.basic_plotter.plot(results["basic_mi"], axes[0, 0])

        # Pairwise MI heatmap
        if "pairwise_mi" in results:
            self.pairwise_plotter.plot_heatmap(results["pairwise_mi"], axes[0, 1])

        # Summary text
        self._add_summary_text(axes[1, 0], results)

        plt.suptitle("Information Analysis Summary", fontsize=16)
        return fig, axes

    def _add_summary_text(self, ax: plt.Axes, results: dict[str, Any]):
        """Add summary statistics text."""
        summary_lines = []

        if "basic_mi" in results:
            basic_mi = results["basic_mi"]
            # Include all available basic MI metrics (E, I, Vb, Vout, and combinations)
            for key in ["I(E;C)", "I(I;C)", "I(Vb;C)", "I(Vout;C)", "I(E,I;C)"]:
                if key in basic_mi:
                    summary_lines.append(f"{key}: {basic_mi[key]:.4f} bits")

        if summary_lines:
            summary_text = "\n".join(summary_lines)
            ax.text(
                0.05,
                0.95,
                summary_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontfamily="monospace",
                fontsize=10,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Summary Statistics")

    def _plot_layer_heatmap(self, ax: plt.Axes, layer_results: dict, layer_names: list):
        """Plot layer information as heatmap."""
        # Check which metrics are available
        possible_metrics = [
            "I(E;C)_mean",
            "I(I;C)_mean",
            "I(Vb;C)_mean",
            "I(Vout;C)_mean",
            "I(E;I)_mean",
            "I(E_lin;C)_mean",
            "I(I_lin;C)_mean",
            "I(Vb_lin;C)_mean",
        ]
        possible_labels = [
            "I(E;C)",
            "I(I;C)",
            "I(Vb;C)",
            "I(Vout;C)",
            "I(E;I)",
            "I(E_lin;C)",
            "I(I_lin;C)",
            "I(Vb_lin;C)",
        ]

        metrics = []
        metric_labels = []
        for metric, label in zip(possible_metrics, possible_labels):
            if any(metric in layer_results.get(ln, {}) for ln in layer_names):
                metrics.append(metric)
                metric_labels.append(label)

        # Fallback if none found
        if not metrics:
            metrics = ["I(E;C)_mean", "I(I;C)_mean", "I(Vout;C)_mean", "I(E;I)_mean"]
            metric_labels = ["I(E;C)", "I(I;C)", "I(Vout;C)", "I(E;I)"]

        data_matrix = []
        for metric in metrics:
            row = [layer_results[name].get(metric, 0) for name in layer_names]
            data_matrix.append(row)

        data_matrix = np.array(data_matrix)
        create_heatmap(
            ax,
            data_matrix,
            layer_names,
            metric_labels,
            "Information Metrics Heatmap",
            annotate=True,
        )

    def _plot_layer_depth_analysis(
        self, layer_results: dict, layer_names: list, metric_type: str, title: str
    ) -> plt.Figure:
        """Plot information metrics vs layer depth with mean and variance."""
        try:
            # Define metric patterns for each type
            if metric_type == "basic_mi":
                metric_patterns = [
                    "I(E;C)",
                    "I(I;C)",
                    "I(Vb;C)",
                    "I(Vout;C)",
                    "I(E,I;C)",
                    "I(E_lin;C)",
                    "I(I_lin;C)",
                    "I(Vb_lin;C)",
                ]
            elif metric_type == "pairwise_mi":
                metric_patterns = [
                    "I(E;I)",
                    "I(E;Vout)",
                    "I(I;Vout)",
                    "I(E;Vb)",
                    "I(I;Vb)",
                    "I(Vb;Vout)",
                    "I(E_lin;I_lin)",
                    "I(E_lin;Vb_lin)",
                    "I(I_lin;Vb_lin)",
                ]
            elif metric_type == "conditional_mi":
                metric_patterns = [
                    "I(E;I|C)",
                    "I(E;Vout|C)",
                    "I(I;Vout|C)",
                    "I(E;Vb|C)",
                    "I(I;Vb|C)",
                    "I(Vb;Vout|C)",
                    "I(E_lin;I_lin|C)",
                    "I(E_lin;Vb_lin|C)",
                    "I(I_lin;Vb_lin|C)",
                ]
            else:
                metric_patterns = []

            # Get all metrics of this type by checking for pattern matches
            metrics = set()
            for layer_data in layer_results.values():
                for key in layer_data.keys():
                    if "_mean" in key:
                        # Extract base metric name by removing "_mean"
                        base_metric = key.replace("_mean", "")
                        # Check if this base metric matches any of our patterns
                        if any(pattern == base_metric for pattern in metric_patterns):
                            metrics.add(base_metric)

            metrics = sorted(metrics)
            logger.info(f"Found {metric_type} metrics: {metrics}")

            # Determine grid size based on number of metrics
            n_metrics = len(metrics)
            if n_metrics == 0:
                logger.warning(f"No {metric_type} metrics found")
                return None

            # Create appropriate grid: 2x2 for 3-4 metrics, 2x3 for 5-6, 3x3 for 7-9, etc.
            ncols = min(3, (n_metrics + 1) // 2 + (n_metrics % 2))
            nrows = (n_metrics + ncols - 1) // ncols

            fig, axes = self.create_subplots(
                nrows, ncols, figsize=(7 * ncols, 5 * nrows)
            )
            if nrows == 1 and ncols == 1:
                axes = [axes]
            else:
                axes = axes.flatten()

            # Plot each metric
            for idx, metric in enumerate(metrics):  # Plot all metrics, not just first 4
                ax = axes.flat[idx]

                means = []
                stds = []
                valid_layers = []

                for layer_name in layer_names:
                    if layer_name in layer_results:
                        mean_key = f"{metric}_mean"
                        std_key = f"{metric}_std"

                        if mean_key in layer_results[layer_name]:
                            means.append(layer_results[layer_name][mean_key])
                            stds.append(layer_results[layer_name].get(std_key, 0))
                            valid_layers.append(layer_name)

                if means:
                    x_pos = range(len(valid_layers))

                    # Plot means with error bars
                    ax.errorbar(
                        x_pos,
                        means,
                        yerr=stds,
                        marker="o",
                        capsize=5,
                        linewidth=2,
                        markersize=6,
                        label=f"{metric} (mean ± std)",
                    )

                    # Plot variance as filled area
                    ax.fill_between(
                        x_pos,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        alpha=0.3,
                    )

                    ax.set_xlabel("Layer (Soma to Distal)")
                    ax.set_ylabel(f"{metric} (bits)")
                    ax.set_title(f"{metric}")
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(valid_layers, rotation=45)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {metric}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

            # Hide unused subplots
            for idx in range(len(metrics), len(axes)):
                axes[idx].axis("off")

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting {metric_type} vs depth: {e}")
            return None

    def _plot_branch_variance_analysis(
        self, layer_results: dict, layer_names: list
    ) -> plt.Figure:
        """Plot analysis of variance between branches in each layer."""
        try:
            fig, axes = self.create_subplots(2, 2, figsize=(15, 10))

            # Get all std metrics (variance indicators)
            std_metrics = []
            for layer_data in layer_results.values():
                for key in layer_data.keys():
                    if "_std" in key and key not in std_metrics:
                        std_metrics.append(key)

            logger.info(f"Found variance metrics: {std_metrics}")

            # Plot variance for different metric types
            for idx, std_metric in enumerate(std_metrics[:4]):  # Limit to 4 subplots
                ax = axes.flat[idx]

                variances = []
                valid_layers = []

                for layer_name in layer_names:
                    if (
                        layer_name in layer_results
                        and std_metric in layer_results[layer_name]
                    ):
                        variances.append(layer_results[layer_name][std_metric])
                        valid_layers.append(layer_name)

                if variances:
                    x_pos = range(len(valid_layers))
                    bars = ax.bar(
                        x_pos,
                        variances,
                        alpha=0.7,
                        color=plt.cm.viridis(idx / max(1, len(std_metrics) - 1)),
                    )

                    # Add value labels on bars
                    for bar, val in zip(bars, variances):
                        height = bar.get_height()
                        ax.text(
                            bar.get_x() + bar.get_width() / 2.0,
                            height + height * 0.01,
                            f"{val:.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                    ax.set_xlabel("Layer (Soma to Distal)")
                    ax.set_ylabel("Standard Deviation (bits)")
                    ax.set_title(f'Branch Variance: {std_metric.replace("_std", "")}')
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(valid_layers, rotation=45)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No variance data for {std_metric}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

            plt.suptitle(
                "Branch Variance Analysis (Variance Between Branches per Layer)",
                fontsize=16,
            )
            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error(f"Error plotting branch variance: {e}")
            return None


# ============================================================================
# Convenience Functions
# ============================================================================


def plot_per_layer_information(layer_results: dict[str, dict[str, Any]], **kwargs):
    """Convenience function for per-layer information plotting."""
    plotter = InformationAnalysisPlotter()
    return plotter.plot_per_layer_information(layer_results, **kwargs)


# ============================================================================
# Dendritic Depth Analysis Functions
# ============================================================================


def plot_structured_ei_network_analysis(
    layer_statistics: dict,
    save_path: str,
    logger: Optional[logging.Logger] = None,
    somatic_synapses: bool = True,
):
    """Generate structured plots comparing different layers with color-coded lines.

    Creates 2 plots:
    - Excitatory Networks: Layer 0 vs Layer 1 as different colored lines
    - Inhibitory Networks: Layer 0 vs Layer 1 as different colored lines
    X-axis: [Soma, Layer 1, Layer 2] (dendritic depth)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Generating structured EI network plots with layer comparison...")

    # Group layers by major layer and network type
    layer_groups = {}
    for layer_name in layer_statistics.keys():
        parts = layer_name.split(".")
        if len(parts) >= 6:
            major_layer = parts[2]  # layers.0 or layers.1
            network_type = parts[3]  # inhibitory_cells or excitatory_cells
            branch_layer = parts[5]  # branch_layers.0, branch_layers.1, branch_layers.2

            group_key = f"{major_layer}_{network_type}"
            if group_key not in layer_groups:
                layer_groups[group_key] = {}

            layer_groups[group_key][branch_layer] = layer_statistics[layer_name]

    logger.info(
        f"Found {len(layer_groups)} major layer groups: {list(layer_groups.keys())}"
    )

    # Reorganize by network type instead of individual layers
    network_comparison_data = {
        "excitatory": {},  # Will contain Layer 0 and Layer 1 data
        "inhibitory": {},  # Will contain Layer 0 and Layer 1 data
    }

    # Process each group and organize by network type
    for group_key, group_data in layer_groups.items():
        major_layer, network_type = group_key.split("_", 1)

        # Create ordered layer data for this group (only include existing branches)
        ordered_data = {}
        branch_to_label = {"0": "Soma", "1": "Layer 1", "2": "Layer 2"}

        # Only process branches that actually exist in the data
        for branch_idx in sorted(group_data.keys()):  # Use available branches only
            if branch_idx in branch_to_label:
                label = branch_to_label[branch_idx]
                ordered_data[label] = group_data[branch_idx]

        if ordered_data:
            # Get clean network type name
            net_type_clean = network_type.split("_")[0]  # "excitatory" or "inhibitory"
            layer_label = f"Layer {major_layer[-1]}"  # "Layer 0" or "Layer 1"

            # Store data organized for comparison plotting
            if net_type_clean not in network_comparison_data:
                network_comparison_data[net_type_clean] = {}

            network_comparison_data[net_type_clean][layer_label] = ordered_data

    # Generate comparison plots
    _generate_layer_comparison_plots(network_comparison_data, save_path, logger)


def _generate_layer_comparison_plots(
    network_comparison_data: dict, save_path: str, logger: logging.Logger
):
    """Generate plots comparing Layer 0 vs Layer 1 for each network type."""
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    # Define colors for different layers
    layer_colors = {
        "Layer 0": "#1f77b4",  # Blue
        "Layer 1": "#ff7f0e",  # Orange
    }

    # Information metrics to plot
    metric_types = [
        (
            "basic_mi",
            [
                "I(E;C)",
                "I(I;C)",
                "I(Vb;C)",
                "I(Vout;C)",
                "I(E,I;C)",
                "I(E_lin;C)",
                "I(I_lin;C)",
                "I(Vb_lin;C)",
            ],
        ),
        (
            "pairwise_mi",
            [
                "I(E;I)",
                "I(E;Vout)",
                "I(I;Vout)",
                "I(E;Vb)",
                "I(I;Vb)",
                "I(Vb;Vout)",
                "I(E_lin;I_lin)",
                "I(E_lin;Vb_lin)",
                "I(I_lin;Vb_lin)",
            ],
        ),
        (
            "conditional_mi",
            [
                "I(E;I|C)",
                "I(E;Vout|C)",
                "I(I;Vout|C)",
                "I(E;Vb|C)",
                "I(I;Vb|C)",
                "I(Vb;Vout|C)",
                "I(E_lin;I_lin|C)",
                "I(E_lin;Vb_lin|C)",
                "I(I_lin;Vb_lin|C)",
            ],
        ),
    ]

    for network_type, layer_data in network_comparison_data.items():
        logger.info(f"Generating comparison plots for {network_type} networks")

        for metric_category, metrics in metric_types:
            # Create figure with subplots for each metric - dynamic sizing
            n_metrics = len(metrics)
            n_cols = 3  # Use 3 columns
            n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1 or n_cols == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()

            for idx, metric in enumerate(metrics):
                if idx >= len(axes):
                    break
                ax = axes[idx]

                # First, determine all available depths from all layers
                all_available_depths = set()
                for _layer_name, depth_data in layer_data.items():
                    all_available_depths.update(depth_data.keys())

                # Sort depths in logical order (soma to distal)
                # Create dynamic depth ordering for any number of distal layers
                def depth_sort_key(depth_name):
                    if depth_name == "Soma":
                        return 0
                    # Backward-compatible: some plots use "Layer N" labels
                    elif depth_name.startswith("Layer"):
                        try:
                            return int(depth_name.split()[-1])
                        except ValueError:
                            return 99
                    elif depth_name.startswith("Distal Layer"):
                        try:
                            return int(depth_name.split()[-1])
                        except ValueError:
                            return 99
                    else:
                        return 99

                sorted_depths = sorted(all_available_depths, key=depth_sort_key)

                # Plot each layer as a different colored line
                for layer_name, depth_data in layer_data.items():
                    if not depth_data:
                        continue

                    # Extract dendritic depths and metric values
                    x_positions = []
                    values = []
                    errors = []

                    # Use the sorted depths that were determined for all layers
                    for depth_idx, depth_label in enumerate(sorted_depths):
                        if depth_label in depth_data:
                            depth_stats = depth_data[depth_label]
                            if f"{metric}_mean" in depth_stats:
                                x_positions.append(depth_idx)
                                values.append(depth_stats[f"{metric}_mean"])
                                errors.append(depth_stats.get(f"{metric}_std", 0))

                    if x_positions and values:
                        x_pos = np.asarray(x_positions)
                        color = layer_colors.get(layer_name, "#808080")

                        # Plot line with error bars
                        ax.errorbar(
                            x_pos,
                            values,
                            yerr=errors,
                            marker="o",
                            linewidth=2,
                            markersize=6,
                            label=layer_name,
                            color=color,
                            capsize=3,
                        )

                        # Fill between for variance
                        ax.fill_between(
                            x_pos,
                            [v - e for v, e in zip(values, errors)],
                            [v + e for v, e in zip(values, errors)],
                            alpha=0.2,
                            color=color,
                        )

                # Customize subplot using the actual depths found in the data
                if sorted_depths:  # Only customize if we have data
                    ax.set_xlabel("Dendritic Depth")
                    ax.set_ylabel(f"{metric} (bits)")
                    ax.set_title(f"{metric}")
                    ax.set_xticks(range(len(sorted_depths)))
                    ax.set_xticklabels(sorted_depths, rotation=0)
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                else:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for {metric}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )

            # Remove empty subplots
            for idx in range(len(metrics), len(axes)):
                fig.delaxes(axes[idx])

            # Set overall title and save
            fig.suptitle(
                f"{network_type.title()} Networks: {metric_category.replace('_', ' ').title()} vs Dendritic Depth",
                fontsize=16,
            )
            plt.tight_layout()

            filename = f"layer_comparison_{network_type}_{metric_category}_depth.png"
            filepath = os.path.join(save_path, filename)
            plt.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close()

            logger.info(f"Saved comparison plot: {filename}")


def plot_layer_wise_aggregated_analysis(
    layer_statistics: dict,
    save_path: str,
    logger: Optional[logging.Logger] = None,
    somatic_synapses: bool = True,
):
    """Generate layer-wise aggregated plots when per_layer is enabled but per_einet is disabled.

    This aggregates E/I networks within each layer but keeps layers separate.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Generating layer-wise aggregated plots...")
    # Goal: keep EI layers separate, but aggregate E/I networks within each layer.
    # We implement this by generating a fully-aggregated dendritic-depth plot *per EI layer*.
    import os

    # Collect EI layer indices present in layer_statistics
    layer_indices: set[int] = set()
    for layer_name in layer_statistics.keys():
        parts = str(layer_name).split(".")
        if "layers" in parts:
            try:
                idx = parts.index("layers")
                layer_indices.add(int(parts[idx + 1]))
            except (ValueError, IndexError):
                continue

    if not layer_indices:
        logger.warning(
            "No EI layers found in layer_statistics; skipping layer-wise plots."
        )
        return

    out_root = os.path.join(save_path, "layer_wise_aggregated")
    os.makedirs(out_root, exist_ok=True)

    for layer_idx in sorted(layer_indices):
        subset: dict[str, Any] = {}
        for layer_name, stats in layer_statistics.items():
            if layer_name == "synthetic_soma":
                # Include synthetic soma in all subsets (safe no-op if absent).
                subset[layer_name] = stats
                continue

            parts = str(layer_name).split(".")
            if "layers" not in parts:
                continue
            try:
                idx = parts.index("layers")
                ei_layer_idx = int(parts[idx + 1])
            except (ValueError, IndexError):
                continue
            if ei_layer_idx == layer_idx:
                subset[layer_name] = stats

        if not subset:
            continue

        layer_dir = os.path.join(out_root, f"ei_layer_{layer_idx}")
        os.makedirs(layer_dir, exist_ok=True)

        # Re-use the robust aggregation/plotting logic.
        plot_fully_aggregated_analysis(
            subset,
            save_path=layer_dir,
            logger=logger,
            somatic_synapses=somatic_synapses,
        )

    logger.info(f"Saved layer-wise aggregated plots under: {out_root}")


def plot_network_wise_aggregated_analysis(
    layer_statistics: dict,
    save_path: str,
    logger: Optional[logging.Logger] = None,
    somatic_synapses: bool = True,
):
    """Generate network-wise aggregated plots when per_einet is enabled but per_layer is disabled.

    This aggregates layers but keeps E/I networks separate.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Generating network-wise aggregated plots...")
    # Goal: aggregate across EI layers, but keep E/I networks separate.
    # We implement this by generating a fully-aggregated dendritic-depth plot *per network type*.
    import os

    # Partition layer_statistics by cell type (excitatory_cells vs inhibitory_cells)
    groups: dict[str, dict[str, Any]] = {"excitatory_cells": {}, "inhibitory_cells": {}}
    for layer_name, stats in layer_statistics.items():
        if layer_name == "synthetic_soma":
            # Include synthetic soma in both networks (safe no-op if absent).
            groups["excitatory_cells"][layer_name] = stats
            groups["inhibitory_cells"][layer_name] = stats
            continue

        parsed = _parse_information_layer_path(str(layer_name))
        if parsed is not None:
            _, cell_type, _ = parsed
            groups[cell_type][layer_name] = stats

    out_root = os.path.join(save_path, "network_wise_aggregated")
    os.makedirs(out_root, exist_ok=True)

    for cell_type, subset in groups.items():
        if not subset:
            continue
        net_dir = os.path.join(out_root, cell_type.replace("_cells", ""))
        os.makedirs(net_dir, exist_ok=True)

        plot_fully_aggregated_analysis(
            subset,
            save_path=net_dir,
            logger=logger,
            somatic_synapses=somatic_synapses,
        )

    logger.info(f"Saved network-wise aggregated plots under: {out_root}")


def _parse_information_layer_path(
    layer_name: str,
) -> tuple[int, str, int | None] | None:
    """Parse current and legacy information-statistic module paths."""
    parts = layer_name.split(".")
    if "layers" not in parts:
        return None

    try:
        layer_position = parts.index("layers")
        network_layer = int(parts[layer_position + 1])
    except (ValueError, IndexError):
        return None

    population_name = parts[layer_position + 2]
    if population_name == "populations":
        try:
            population_name = parts[layer_position + 3]
        except IndexError:
            return None

    normalized = population_name.lower()
    if normalized == "excitatory_cells" or normalized.startswith(("e", "exc")):
        cell_type = "excitatory_cells"
    elif normalized == "inhibitory_cells" or normalized.startswith(("i", "inh")):
        cell_type = "inhibitory_cells"
    else:
        return None

    branch_layer = None
    if "branch_layers" in parts:
        try:
            branch_position = parts.index("branch_layers")
            branch_layer = int(parts[branch_position + 1])
        except (ValueError, IndexError):
            return None

    return network_layer, cell_type, branch_layer


def plot_info_vs_ei_depth_per_dendritic_layer(
    layer_statistics: dict[str, Any],
    save_path: str,
    logger: Optional[logging.Logger] = None,
    somatic_synapses: bool = True,
    layer_soma_relative_depths: Optional[dict[str, int]] = None,
) -> None:
    """Plot information vs EI layer depth for each dendritic branch layer.

    This creates the complement to the existing plots - instead of showing
    info vs dendritic depth for each EI layer, this shows info vs EI depth
    for each dendritic branch layer.

    Args:
        layer_statistics: Layer-wise information statistics
        save_path: Directory to save plots
        logger: Optional logger
    """
    try:
        if logger:
            logger.info("Starting plot_info_vs_ei_depth_per_dendritic_layer...")
            logger.info(f"Received {len(layer_statistics)} layer statistics")

        # Organize data by dendritic branch layer instead of EI layer
        dendritic_layers_data = {}

        # Determine dynamic max branch index for correct depth mapping
        branch_indices = []
        for lname in layer_statistics.keys():
            if "branch_layers." in lname:
                try:
                    branch_indices.append(
                        int(lname.split("branch_layers.")[1].split(".")[0])
                    )
                except (IndexError, ValueError):
                    continue
        max_branch_idx = max(branch_indices) if branch_indices else -1

        # Extract data from layer_statistics
        for layer_name, stats in layer_statistics.items():
            if logger:
                logger.debug(f"Processing layer: {layer_name}")

            try:
                parsed = _parse_information_layer_path(str(layer_name))
                if parsed is None:
                    if logger:
                        logger.debug(f"Skipping unparseable layer: {layer_name}")
                    continue
                ei_layer_idx, cell_type, branch_layer_idx = parsed

                if layer_soma_relative_depths is not None:
                    dendritic_depth = layer_soma_relative_depths.get(layer_name)
                else:
                    dendritic_depth = None
                if dendritic_depth is None:
                    if branch_layer_idx is None:
                        dendritic_depth = 0
                    else:
                        dendritic_depth = max_branch_idx - branch_layer_idx

                # Create key for dendritic layer
                dendritic_key = f"dendritic_depth_{dendritic_depth}_{cell_type}"

                if dendritic_key not in dendritic_layers_data:
                    dendritic_layers_data[dendritic_key] = {}

                dendritic_layers_data[dendritic_key][ei_layer_idx] = stats

            except (ValueError, IndexError) as e:
                if logger:
                    logger.debug(f"Error parsing layer {layer_name}: {e}")
                continue

        if not dendritic_layers_data:
            if logger:
                logger.warning("No valid dendritic layer data found for EI depth plots")
            return

        # Create plots for each metric type (basic, pairwise, conditional)
        metric_types = [
            (
                "basic_mi",
                [
                    "I(E;C)",
                    "I(I;C)",
                    "I(Vb;C)",
                    "I(Vout;C)",
                    "I(E,I;C)",
                    "I(E_lin;C)",
                    "I(I_lin;C)",
                    "I(Vb_lin;C)",
                    "I(E_lin,I_lin;C)",
                ],
            ),
            (
                "pairwise_mi",
                [
                    "I(E;I)",
                    "I(E;Vout)",
                    "I(I;Vout)",
                    "I(E;Vb)",
                    "I(I;Vb)",
                    "I(Vb;Vout)",
                    "I(E_lin;I_lin)",
                    "I(E_lin;Vout)",
                    "I(I_lin;Vout)",
                    "I(E_lin;Vb_lin)",
                    "I(I_lin;Vb_lin)",
                ],
            ),
            (
                "conditional_mi",
                [
                    "I(E;I|C)",
                    "I(E;Vout|C)",
                    "I(I;Vout|C)",
                    "I(E;Vb|C)",
                    "I(I;Vb|C)",
                    "I(Vb;Vout|C)",
                    "I(E_lin;I_lin|C)",
                    "I(E_lin;Vout|C)",
                    "I(I_lin;Vout|C)",
                ],
            ),
        ]

        dendritic_depths = sorted(
            {
                int(key.split("_")[2])
                for key in dendritic_layers_data
                if key.startswith("dendritic_depth_")
            }
        )
        color_values = plt.cm.viridis(
            np.linspace(0.15, 0.85, max(len(dendritic_depths), 1))
        )
        branch_colors = dict(zip(dendritic_depths, color_values))

        if logger:
            logger.info(
                f"Dendritic layers data keys: {list(dendritic_layers_data.keys())}"
            )
            logger.info(f"Going to generate plots for metric types: {metric_types}")

        for metric_category, metric_names in metric_types:
            # Create separate plots for excitatory and inhibitory
            for cell_type in ["excitatory_cells", "inhibitory_cells"]:
                # Dynamically size grid based on number of metrics
                n_metrics = len(metric_names)
                n_cols = 3
                n_rows = (n_metrics + n_cols - 1) // n_cols

                fig, axes = plt.subplots(
                    n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows)
                )
                if n_rows == 1 and n_cols == 1:
                    axes = np.array([axes])
                elif n_rows == 1 or n_cols == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()

                # Track if any plots were created
                plots_created = []

                # Plot all metrics
                for i, metric_name in enumerate(metric_names):
                    if i >= len(axes):
                        break

                    ax = axes[i]
                    data_plotted = False

                    # Plot each dendritic branch layer as a different colored line
                    for dendritic_depth in dendritic_depths:
                        dendritic_key = f"dendritic_depth_{dendritic_depth}_{cell_type}"

                        if dendritic_key not in dendritic_layers_data:
                            continue

                        ei_data = dendritic_layers_data[dendritic_key]

                        # Sort by EI layer index
                        ei_layers = sorted(ei_data.keys())

                        if len(ei_layers) < 2:  # Need at least 2 points
                            continue

                        # Collect values across EI layers using flattened format
                        values = []
                        errors = []

                        for ei_idx in ei_layers:
                            if ei_idx in ei_data:
                                layer_stats = ei_data[ei_idx]
                                # Use flattened format: metric_name + "_mean" and "_std"
                                mean_key = f"{metric_name}_mean"
                                std_key = f"{metric_name}_std"

                                if mean_key in layer_stats:
                                    values.append(layer_stats[mean_key])
                                    errors.append(layer_stats.get(std_key, 0))
                                else:
                                    values.append(0)
                                    errors.append(0)
                            else:
                                values.append(0)
                                errors.append(0)

                        # Only plot if we have non-zero values
                        if any(v != 0 for v in values):
                            color = branch_colors[dendritic_depth]
                            if dendritic_depth == 0:
                                branch_label = "Soma"
                            else:
                                branch_label = f"Dendritic depth {dendritic_depth}"

                            # Plot line with error bars
                            ax.errorbar(
                                ei_layers,
                                values,
                                yerr=errors,
                                marker="o",
                                linewidth=2,
                                markersize=6,
                                label=branch_label,
                                color=color,
                                capsize=3,
                            )

                            # Fill between for variance (like layer_comparison)
                            ax.fill_between(
                                ei_layers,
                                [v - e for v, e in zip(values, errors)],
                                [v + e for v, e in zip(values, errors)],
                                alpha=0.2,
                                color=color,
                            )

                            data_plotted = True

                    # Customize subplot if data was plotted
                    if data_plotted:
                        ax.set_xlabel("Network position")
                        ax.set_ylabel(f"{metric_name} (bits)")
                        ax.set_title(f"{metric_name}")
                        ax.set_xticks(ei_layers)

                        layer_labels = [
                            f"Layer {layer_idx + 1}" for layer_idx in ei_layers
                        ]

                        ax.set_xticklabels(layer_labels, rotation=0)
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc="best")
                        plots_created.append(i)
                    else:
                        ax.set_visible(False)

                # Remove empty subplots and finalize the figure
                if plots_created:
                    # Remove metrics absent from this population instead of
                    # publishing large placeholder panels.
                    for axis_index, axis in enumerate(axes):
                        if axis_index not in plots_created:
                            fig.delaxes(axis)

                    cell_type_name = cell_type.replace("_cells", "").title()
                    fig.suptitle(
                        f'{cell_type_name} Networks: {metric_category.replace("_", " ").title()} vs Network Position',
                        fontsize=16,
                    )
                    plt.tight_layout()

                    # Save plot
                    filename = f"ei_depth_comparison_{cell_type}_{metric_category}.png"
                    plot_path = os.path.join(save_path, filename)
                    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                    plt.close()

                    if logger:
                        logger.info(f"Saved EI depth comparison plot to {plot_path}")
                else:
                    plt.close(fig)

    except Exception as e:
        if logger:
            logger.error(f"Error generating EI depth comparison plots: {e}")
        else:
            print(f"Error generating EI depth comparison plots: {e}")


def plot_fully_aggregated_analysis(
    layer_statistics: dict,
    save_path: str,
    logger: Optional[logging.Logger] = None,
    somatic_synapses: bool = True,
    plot_variance: bool = False,
):
    """Generate fully aggregated plots when both per_layer and per_einet analysis are disabled.

    This creates plots that:
    - Aggregate across all layers (Layer 0 + Layer 1)
    - Aggregate across E/I networks (Excitatory + Inhibitory)
    - Show dendritic depth (Soma, Layer 1, Layer 2) on x-axis
    - Display one line representing the overall network

    Args:
        plot_variance: If True, plot variance (std^2) instead of mean values
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    plot_type = "variance" if plot_variance else "mean"
    logger.info(f"Generating fully aggregated dendritic depth plots ({plot_type})...")
    logger.info(f"Input layer_statistics keys: {list(layer_statistics.keys())}")

    # Debug: Show sample data structure
    if layer_statistics:
        sample_key = next(iter(layer_statistics.keys()))
        sample_data = layer_statistics[sample_key]
        logger.info(
            f"Sample layer data keys for '{sample_key}': {list(sample_data.keys()) if isinstance(sample_data, dict) else 'Not a dict'}"
        )

    # Aggregate data across all layers and E/I networks
    aggregated_data = {}

    # Determine dynamic mapping for dendritic depth (Soma -> proximal -> distal)
    # Collect all branch indices to compute max index
    branch_indices = []
    for lname in layer_statistics.keys():
        if "branch_layers." in lname:
            try:
                branch_indices.append(
                    int(lname.split("branch_layers.")[1].split(".")[0])
                )
            except (IndexError, ValueError):
                continue
    max_branch_idx = max(branch_indices) if branch_indices else -1

    # First, collect all depth labels that exist across all entries
    all_depths = set()
    for layer_name, layer_data in layer_statistics.items():
        if not isinstance(layer_data, dict):
            continue

        # Helper to register a metric entry under a given depth label
        def register_metric(
            depth_label: str, metric_name: str, value: float, std_val: float
        ):
            all_depths.add(depth_label)
            if depth_label not in aggregated_data:
                aggregated_data[depth_label] = {}
            if metric_name not in aggregated_data[depth_label]:
                aggregated_data[depth_label][metric_name] = {"values": [], "stds": []}

            # Store the value directly (either mean or variance depending on plot_variance)
            aggregated_data[depth_label][metric_name]["values"].append(float(value))
            aggregated_data[depth_label][metric_name]["stds"].append(float(std_val))

        def derive_proxy_std(metric_name: str, layer_data: dict) -> float:
            """Best-effort std for proxy aggregations.

            Older runs stored proxy metrics like `<base>_sum` / `<base>_union` with
            std=0. To keep proxy plots informative, derive an approximate std from
            the base metric's per-branch std and branch count when available.
            """
            if not isinstance(metric_name, str) or not metric_name.startswith("I("):
                return 0.0

            base_metric = None
            scale = 1.0

            if metric_name.endswith("_sum_clipped"):
                base_metric = metric_name[: -len("_sum_clipped")]
            elif metric_name.endswith("_sum"):
                # Handle _topK_sum separately below
                if "_top" not in metric_name:
                    base_metric = metric_name[: -len("_sum")]
            elif metric_name.endswith("_union"):
                base_metric = metric_name[: -len("_union")]
                scale = 1.0
            elif metric_name.endswith("_max"):
                base_metric = metric_name[: -len("_max")]
                scale = 1.0

            # _topK_sum pattern: "<base>_top{K}_sum"
            if (
                base_metric is None
                and "_top" in metric_name
                and metric_name.endswith("_sum")
            ):
                prefix, rest = metric_name.rsplit("_top", 1)
                if rest.endswith("_sum"):
                    k_str = rest[: -len("_sum")]
                    if k_str.isdigit():
                        base_metric = prefix
                        k = int(k_str)
                        scale = float(max(1, k))

            if base_metric is None:
                return 0.0

            base_std = float(layer_data.get(f"{base_metric}_std", 0.0))
            if base_std <= 0:
                return 0.0

            base_n = float(layer_data.get(f"{base_metric}_n", 0.0))
            if base_n > 0 and metric_name.endswith(("_sum", "_sum_clipped")):
                return float(np.sqrt(base_n) * base_std)
            if base_n > 0 and "_top" in metric_name and metric_name.endswith("_sum"):
                eff_k = min(scale, base_n)
                return float(np.sqrt(eff_k) * base_std)

            # Fallback for union/max: use base metric spread.
            return base_std

        # Case 1: synthetic soma (added when somatic_synapses=False)
        if layer_name == "synthetic_soma":
            if plot_variance:
                # For variance plots, look for _variance keys
                for depth_key, _val in layer_data.items():
                    if depth_key.endswith("_variance"):
                        metric_name = depth_key.replace("_variance", "")
                        var_val = layer_data.get(f"{metric_name}_variance", 0)
                        register_metric(
                            "Soma", metric_name, var_val, 0
                        )  # No std for variance plots
            else:
                # For mean plots, look for _mean keys
                for depth_key, _val in layer_data.items():
                    if depth_key.endswith("_mean"):
                        metric_name = depth_key.replace("_mean", "")
                        mean_val = layer_data.get(f"{metric_name}_mean", 0)
                        std_val = layer_data.get(f"{metric_name}_std", 0)
                        if not std_val:
                            std_val = derive_proxy_std(metric_name, layer_data)
                        register_metric("Soma", metric_name, mean_val, std_val)
            continue

        # Case 2: actual dendritic branch layers
        if "branch_layers." in layer_name:
            try:
                branch_idx = int(layer_name.split("branch_layers.")[1].split(".")[0])
            except (IndexError, ValueError):
                continue

            # Check if this layer is the soma (no synapses)
            # When somatic_synapses=False, the highest branch_idx layer has has_synapses=False
            has_synapses = layer_data.get("has_synapses", True)
            if not has_synapses:
                # This is the soma layer
                depth_label = "Soma"
            elif max_branch_idx >= 0:
                # Map branch index to distal depth dynamically.
                # For layers with synapses, compute distal number relative to non-soma layers
                # Distal Layer numbering starts at 1 (closest to soma)
                distal_number = max_branch_idx - branch_idx
                if distal_number == 0:
                    # This would be the soma, but has_synapses=True, so label as Distal Layer 1
                    depth_label = "Distal Layer 1"
                else:
                    depth_label = f"Distal Layer {distal_number}"
            else:
                # Fallback if we couldn't determine max index
                depth_label = f"Distal Layer {branch_idx + 1}"

            if plot_variance:
                # For variance plots, look for _variance keys
                for depth_key, _val in layer_data.items():
                    if depth_key.endswith("_variance"):
                        metric_name = depth_key.replace("_variance", "")
                        var_val = layer_data.get(f"{metric_name}_variance", 0)
                        register_metric(
                            depth_label, metric_name, var_val, 0
                        )  # No std for variance plots
            else:
                # For mean plots, look for _mean keys
                for depth_key, _val in layer_data.items():
                    if depth_key.endswith("_mean"):
                        metric_name = depth_key.replace("_mean", "")
                        mean_val = layer_data.get(f"{metric_name}_mean", 0)
                        std_val = layer_data.get(f"{metric_name}_std", 0)
                        if not std_val:
                            std_val = derive_proxy_std(metric_name, layer_data)
                        register_metric(depth_label, metric_name, mean_val, std_val)

    # Compute final aggregated statistics
    final_aggregated = {}
    for depth_label, depth_data in aggregated_data.items():
        final_aggregated[depth_label] = {}
        for metric_name, metric_data in depth_data.items():
            # Average across all layers and networks
            final_aggregated[depth_label][f"{metric_name}_mean"] = np.mean(
                metric_data["values"]
            )
            # Combine standard deviations (root mean square)
            final_aggregated[depth_label][f"{metric_name}_std"] = np.sqrt(
                np.mean(np.array(metric_data["stds"]) ** 2)
            )

    logger.info(f"Final aggregated data depths: {list(final_aggregated.keys())}")
    if final_aggregated:
        sample_depth = next(iter(final_aggregated.keys()))
        logger.info(
            f"Sample depth '{sample_depth}' metrics: {list(final_aggregated[sample_depth].keys())}"
        )

    # Sort depths in logical order: Soma first, then Distal Layer 1..N
    def depth_sort_key(label: str) -> int:
        if label == "Soma":
            return 0
        if label.startswith("Distal Layer"):
            try:
                return int(label.split()[-1])
            except ValueError:
                return 999
        return 999

    sorted_depths = sorted(all_depths, key=depth_sort_key)

    # Define metric categories with paired comparisons.
    # IMPORTANT: Titles should not hard-code LDA metric names, because LDA metrics may
    # be absent when compute_lda_weights=False. Metric names belong in the legend.
    metric_pairs = [
        (
            "basic_mi",
            [
                (
                    "I(E;C)",
                    [
                        ("I(E;C)", "#1f77b4", "DendriNet"),
                        ("I(E_lin;C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh;C)", "#7f7f7f", "Random"),  # Gray for random baseline
                        ("I(E;C)_null", "#000000", "Label-shuffle"),
                    ],
                ),
                (
                    "I(I;C)",
                    [
                        ("I(I;C)", "#1f77b4", "DendriNet"),
                        ("I(I_lin;C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(I_sh;C)", "#7f7f7f", "Random"),
                        ("I(I;C)_null", "#000000", "Label-shuffle"),
                    ],
                ),
                (
                    "I(Vb;C)",
                    [
                        ("I(Vb;C)", "#1f77b4", "DendriNet"),
                        ("I(Vb_lin;C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vb_sh;C)", "#7f7f7f", "Random"),
                        ("I(Vb;C)_null", "#000000", "Label-shuffle"),
                    ],
                ),
                (
                    "I(Vout;C)",
                    [
                        ("I(Vout;C)", "#1f77b4", "DendriNet"),
                        ("I(Vout_lin;C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vout_sh;C)", "#7f7f7f", "Random"),
                        ("I(Vout;C)_null", "#000000", "Label-shuffle"),
                    ],
                ),
                (
                    "I(E,I;C)",
                    [
                        ("I(E,I;C)", "#1f77b4", "DendriNet"),
                        ("I(E_lin,I_lin;C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh,I_sh;C)", "#7f7f7f", "Random"),
                        ("I(E,I;C)_null", "#000000", "Label-shuffle"),
                    ],
                ),
            ],
        ),
        (
            "pairwise_mi",
            [
                (
                    "I(E;I)",
                    [
                        ("I(E;I)", "#1f77b4", "DendriNet"),
                        ("I(E_lin;I_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh;I_sh)", "#7f7f7f", "Random"),
                        ("I(E;I)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(E;Vout)",
                    [
                        ("I(E;Vout)", "#1f77b4", "DendriNet"),
                        ("I(E_lin;Vout_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh;Vout_sh)", "#7f7f7f", "Random"),
                        ("I(E;Vout)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(I;Vout)",
                    [
                        ("I(I;Vout)", "#1f77b4", "DendriNet"),
                        ("I(I_lin;Vout_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(I_sh;Vout_sh)", "#7f7f7f", "Random"),
                        ("I(I;Vout)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(E;Vb)",
                    [
                        ("I(E;Vb)", "#1f77b4", "DendriNet"),
                        ("I(E_lin;Vb_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh;Vb_sh)", "#7f7f7f", "Random"),
                        ("I(E;Vb)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(I;Vb)",
                    [
                        ("I(I;Vb)", "#1f77b4", "DendriNet"),
                        ("I(I_lin;Vb_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(I_sh;Vb_sh)", "#7f7f7f", "Random"),
                        ("I(I;Vb)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(Vb;Vout)",
                    [
                        ("I(Vb;Vout)", "#1f77b4", "DendriNet"),
                        ("I(Vb_lin;Vout_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vb_sh;Vout_sh)", "#7f7f7f", "Random"),
                        ("I(Vb;Vout)_null", "#000000", "Shuffle-null"),
                    ],
                ),
            ],
        ),
        (
            "conditional_mi",
            [
                (
                    "I(E;I|C)",
                    [
                        ("I(E;I|C)", "#1f77b4", "DendriNet"),
                        ("I(E_lin;I_lin|C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh;I_sh|C)", "#7f7f7f", "Random"),
                        ("I(E;I|C)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(E;Vout|C)",
                    [
                        ("I(E;Vout|C)", "#1f77b4", "DendriNet"),
                        ("I(E_lin;Vout_lin|C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh;Vout_sh|C)", "#7f7f7f", "Random"),
                        ("I(E;Vout|C)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(I;Vout|C)",
                    [
                        ("I(I;Vout|C)", "#1f77b4", "DendriNet"),
                        ("I(I_lin;Vout_lin|C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(I_sh;Vout_sh|C)", "#7f7f7f", "Random"),
                        ("I(I;Vout|C)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(E;Vb|C)",
                    [
                        ("I(E;Vb|C)", "#1f77b4", "DendriNet"),
                        ("I(E_lin;Vb_lin|C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh;Vb_sh|C)", "#7f7f7f", "Random"),
                        ("I(E;Vb|C)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(I;Vb|C)",
                    [
                        ("I(I;Vb|C)", "#1f77b4", "DendriNet"),
                        ("I(I_lin;Vb_lin|C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(I_sh;Vb_sh|C)", "#7f7f7f", "Random"),
                        ("I(I;Vb|C)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(Vb;Vout|C)",
                    [
                        ("I(Vb;Vout|C)", "#1f77b4", "DendriNet"),
                        ("I(Vb_lin;Vout_lin|C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vb_sh;Vout_sh|C)", "#7f7f7f", "Random"),
                        ("I(Vb;Vout|C)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(Vout;C|Vb)",
                    [
                        ("I(Vout;C|Vb)", "#1f77b4", "DendriNet"),
                        ("I(Vout_lin;C|Vb_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vout_sh;C|Vb_sh)", "#7f7f7f", "Random"),
                        ("I(Vout;C|Vb)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(Vb;C|Vout)",
                    [
                        ("I(Vb;C|Vout)", "#1f77b4", "DendriNet"),
                        ("I(Vb_lin;C|Vout_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vb_sh;C|Vout_sh)", "#7f7f7f", "Random"),
                        ("I(Vb;C|Vout)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(Vout;C|E,I)",
                    [
                        ("I(Vout;C|E,I)", "#1f77b4", "DendriNet"),
                        ("I(Vout_lin;C|E_lin,I_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vout_sh;C|E_sh,I_sh)", "#7f7f7f", "Random"),
                        ("I(Vout;C|E,I)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(E;C|Vb)",
                    [
                        ("I(E;C|Vb)", "#1f77b4", "DendriNet"),
                        ("I(E_lin;C|Vb_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh;C|Vb_sh)", "#7f7f7f", "Random"),
                        ("I(E;C|Vb)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(I;C|Vb)",
                    [
                        ("I(I;C|Vb)", "#1f77b4", "DendriNet"),
                        ("I(I_lin;C|Vb_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(I_sh;C|Vb_sh)", "#7f7f7f", "Random"),
                        ("I(I;C|Vb)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(E,I;C|Vb)",
                    [
                        ("I(E,I;C|Vb)", "#1f77b4", "DendriNet"),
                        ("I(E_lin,I_lin;C|Vb_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh,I_sh;C|Vb_sh)", "#7f7f7f", "Random"),
                        ("I(E,I;C|Vb)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(E;C|I,Vb)",
                    [
                        ("I(E;C|I,Vb)", "#1f77b4", "DendriNet"),
                        ("I(E_lin;C|I_lin,Vb_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(E_sh;C|I_sh,Vb_sh)", "#7f7f7f", "Random"),
                        ("I(E;C|I,Vb)_null", "#000000", "Shuffle-null"),
                    ],
                ),
                (
                    "I(I;C|E,Vb)",
                    [
                        ("I(I;C|E,Vb)", "#1f77b4", "DendriNet"),
                        ("I(I_lin;C|E_lin,Vb_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(I_sh;C|E_sh,Vb_sh)", "#7f7f7f", "Random"),
                        ("I(I;C|E,Vb)_null", "#000000", "Shuffle-null"),
                    ],
                ),
            ],
        ),
        (
            "dendritic_processing",
            [
                (
                    "I(E;Vout|I,Vb)",
                    [
                        (
                            ("I(E;Vout|I,Vb)", "I(E;Vout|I)"),
                            "#1f77b4",
                            "DendriNet",
                        ),
                        (
                            (
                                "I(E_lin;Vout_lin|I_lin,Vb_lin)",
                                "I(E_lin;Vout_lin|I_lin)",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            (
                                "I(E_sh;Vout_sh|I_sh,Vb_sh)",
                                "I(E_sh;Vout_sh|I_sh)",
                            ),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(I;Vout|E,Vb)",
                    [
                        (
                            ("I(I;Vout|E,Vb)", "I(I;Vout|E)"),
                            "#1f77b4",
                            "DendriNet",
                        ),
                        (
                            (
                                "I(I_lin;Vout_lin|E_lin,Vb_lin)",
                                "I(I_lin;Vout_lin|E_lin)",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            (
                                "I(I_sh;Vout_sh|E_sh,Vb_sh)",
                                "I(I_sh;Vout_sh|E_sh)",
                            ),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(Vb;Vout|E,I)",
                    [
                        ("I(Vb;Vout|E,I)", "#1f77b4", "DendriNet"),
                        ("I(Vb_lin;Vout_lin|E_lin,I_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vb_sh;Vout_sh|E_sh,I_sh)", "#7f7f7f", "Random"),
                    ],
                ),
            ],
        ),
        (
            "ablation_proxy_branch_mean",
            [
                (
                    "I(E;C|I,Vb)",
                    [
                        (("I(E;C|I,Vb)", "I(E;C|I)"), "#1f77b4", "DendriNet"),
                        (
                            (
                                "I(E_lin;C|I_lin,Vb_lin)",
                                "I(E_lin;C|I_lin)",
                                # Backward-compat: older runs didn't rename I/Vb in conditioning
                                "I(E_lin;C|I,Vb)",
                                "I(E_lin;C|I)",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(E_sh;C|I_sh,Vb_sh)", "I(E_sh;C|I_sh)"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(I;C|E,Vb)",
                    [
                        (("I(I;C|E,Vb)", "I(I;C|E)"), "#1f77b4", "DendriNet"),
                        (
                            (
                                "I(I_lin;C|E_lin,Vb_lin)",
                                "I(I_lin;C|E_lin)",
                                # Backward-compat: older runs didn't rename E/Vb in conditioning
                                "I(I_lin;C|E,Vb)",
                                "I(I_lin;C|E)",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(I_sh;C|E_sh,Vb_sh)", "I(I_sh;C|E_sh)"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(E,I;C|Vb)",
                    [
                        (("I(E,I;C|Vb)", "I(E,I;C)"), "#1f77b4", "DendriNet"),
                        (
                            (
                                "I(E_lin,I_lin;C|Vb_lin)",
                                "I(E_lin,I_lin;C)",
                                # Backward-compat: older runs didn't rename Vb in conditioning
                                "I(E_lin,I_lin;C|Vb)",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(E_sh,I_sh;C|Vb_sh)", "I(E_sh,I_sh;C)"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(Vb;C)",
                    [
                        ("I(Vb;C)", "#1f77b4", "DendriNet"),
                        ("I(Vb_lin;C)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vb_sh;C)", "#7f7f7f", "Random"),
                    ],
                ),
                (
                    "I(Vb;C|Vout)",
                    [
                        ("I(Vb;C|Vout)", "#1f77b4", "DendriNet"),
                        ("I(Vb_lin;C|Vout_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vb_sh;C|Vout_sh)", "#7f7f7f", "Random"),
                    ],
                ),
                (
                    "I(Vout;C|Vb)",
                    [
                        (("I(Vout;C|Vb)", "I(Vout;C)"), "#1f77b4", "DendriNet"),
                        (
                            (
                                "I(Vout_lin;C|Vb_lin)",
                                "I(Vout_lin;C)",
                                # Backward-compat: older runs didn't rename Vb in conditioning
                                "I(Vout_lin;C|Vb)",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(Vout_sh;C|Vb_sh)", "I(Vout_sh;C)"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(Vout;C|E,I)",
                    [
                        ("I(Vout;C|E,I)", "#1f77b4", "DendriNet"),
                        ("I(Vout_lin;C|E_lin,I_lin)", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vout_sh;C|E_sh,I_sh)", "#7f7f7f", "Random"),
                    ],
                ),
            ],
        ),
        # ------------------------------------------------------------------
        # Layer-total (branch-aggregated) proxies for ablation effects.
        #
        # These are computed from single-branch MI by aggregating across branches
        # within each layer (e.g. sum/union). They are not "true" multivariate MI,
        # but they often track whole-layer ablation trends better than per-branch means.
        # ------------------------------------------------------------------
        (
            "ablation_proxy_sum",
            [
                (
                    "I(E;C|I,Vb)_sum",
                    [
                        (("I(E;C|I,Vb)_sum", "I(E;C|I)_sum"), "#1f77b4", "DendriNet"),
                        (
                            (
                                "I(E_lin;C|I_lin,Vb_lin)_sum",
                                "I(E_lin;C|I_lin)_sum",
                                # Backward-compat: older runs didn't rename I/Vb in conditioning
                                "I(E_lin;C|I,Vb)_sum",
                                "I(E_lin;C|I)_sum",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(E_sh;C|I_sh,Vb_sh)_sum", "I(E_sh;C|I_sh)_sum"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(I;C|E,Vb)_sum",
                    [
                        (("I(I;C|E,Vb)_sum", "I(I;C|E)_sum"), "#1f77b4", "DendriNet"),
                        (
                            (
                                "I(I_lin;C|E_lin,Vb_lin)_sum",
                                "I(I_lin;C|E_lin)_sum",
                                # Backward-compat: older runs didn't rename E/Vb in conditioning
                                "I(I_lin;C|E,Vb)_sum",
                                "I(I_lin;C|E)_sum",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(I_sh;C|E_sh,Vb_sh)_sum", "I(I_sh;C|E_sh)_sum"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(E,I;C|Vb)_sum",
                    [
                        (("I(E,I;C|Vb)_sum", "I(E,I;C)_sum"), "#1f77b4", "DendriNet"),
                        (
                            (
                                "I(E_lin,I_lin;C|Vb_lin)_sum",
                                "I(E_lin,I_lin;C)_sum",
                                # Backward-compat: older runs didn't rename Vb in conditioning
                                "I(E_lin,I_lin;C|Vb)_sum",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(E_sh,I_sh;C|Vb_sh)_sum", "I(E_sh,I_sh;C)_sum"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(Vb;C)_sum",
                    [
                        ("I(Vb;C)_sum", "#1f77b4", "DendriNet"),
                        ("I(Vb_lin;C)_sum", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vb_sh;C)_sum", "#7f7f7f", "Random"),
                    ],
                ),
                (
                    "I(Vb;C|Vout)_sum",
                    [
                        ("I(Vb;C|Vout)_sum", "#1f77b4", "DendriNet"),
                        ("I(Vb_lin;C|Vout_lin)_sum", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vb_sh;C|Vout_sh)_sum", "#7f7f7f", "Random"),
                    ],
                ),
                (
                    "I(Vout;C|Vb)_sum",
                    [
                        (
                            ("I(Vout;C|Vb)_sum", "I(Vout;C)_sum"),
                            "#1f77b4",
                            "DendriNet",
                        ),
                        (
                            (
                                "I(Vout_lin;C|Vb_lin)_sum",
                                "I(Vout_lin;C)_sum",
                                # Backward-compat: older runs didn't rename Vb in conditioning
                                "I(Vout_lin;C|Vb)_sum",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(Vout_sh;C|Vb_sh)_sum", "I(Vout_sh;C)_sum"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(Vout;C|E,I)_sum",
                    [
                        ("I(Vout;C|E,I)_sum", "#1f77b4", "DendriNet"),
                        (
                            "I(Vout_lin;C|E_lin,I_lin)_sum",
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        ("I(Vout_sh;C|E_sh,I_sh)_sum", "#7f7f7f", "Random"),
                    ],
                ),
            ],
        ),
        (
            "ablation_proxy_union",
            [
                (
                    "I(E;C|I,Vb)_union",
                    [
                        (
                            ("I(E;C|I,Vb)_union", "I(E;C|I)_union"),
                            "#1f77b4",
                            "DendriNet",
                        ),
                        (
                            (
                                "I(E_lin;C|I_lin,Vb_lin)_union",
                                "I(E_lin;C|I_lin)_union",
                                # Backward-compat
                                "I(E_lin;C|I,Vb)_union",
                                "I(E_lin;C|I)_union",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(E_sh;C|I_sh,Vb_sh)_union", "I(E_sh;C|I_sh)_union"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(I;C|E,Vb)_union",
                    [
                        (
                            ("I(I;C|E,Vb)_union", "I(I;C|E)_union"),
                            "#1f77b4",
                            "DendriNet",
                        ),
                        (
                            (
                                "I(I_lin;C|E_lin,Vb_lin)_union",
                                "I(I_lin;C|E_lin)_union",
                                # Backward-compat
                                "I(I_lin;C|E,Vb)_union",
                                "I(I_lin;C|E)_union",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(I_sh;C|E_sh,Vb_sh)_union", "I(I_sh;C|E_sh)_union"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(E,I;C|Vb)_union",
                    [
                        (
                            ("I(E,I;C|Vb)_union", "I(E,I;C)_union"),
                            "#1f77b4",
                            "DendriNet",
                        ),
                        (
                            (
                                "I(E_lin,I_lin;C|Vb_lin)_union",
                                "I(E_lin,I_lin;C)_union",
                                # Backward-compat
                                "I(E_lin,I_lin;C|Vb)_union",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(E_sh,I_sh;C|Vb_sh)_union", "I(E_sh,I_sh;C)_union"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(Vb;C)_union",
                    [
                        ("I(Vb;C)_union", "#1f77b4", "DendriNet"),
                        ("I(Vb_lin;C)_union", "#ff7f0e", r"LDA$_{+}$"),
                        ("I(Vb_sh;C)_union", "#7f7f7f", "Random"),
                    ],
                ),
                (
                    "I(Vout;C|Vb)_union",
                    [
                        (
                            ("I(Vout;C|Vb)_union", "I(Vout;C)_union"),
                            "#1f77b4",
                            "DendriNet",
                        ),
                        (
                            (
                                "I(Vout_lin;C|Vb_lin)_union",
                                "I(Vout_lin;C)_union",
                                # Backward-compat
                                "I(Vout_lin;C|Vb)_union",
                            ),
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            ("I(Vout_sh;C|Vb_sh)_union", "I(Vout_sh;C)_union"),
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                ),
                (
                    "I(Vout;C|E,I)_union",
                    [
                        ("I(Vout;C|E,I)_union", "#1f77b4", "DendriNet"),
                        (
                            "I(Vout_lin;C|E_lin,I_lin)_union",
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        ("I(Vout_sh;C|E_sh,I_sh)_union", "#7f7f7f", "Random"),
                    ],
                ),
            ],
        ),
    ]

    # Optionally plot upstream-unique class info when it was computed.
    # This avoids empty panels when compute_upstream_unique_cmi=false (default).
    def _has_metric(metric_name: str) -> bool:
        return any(
            depth_label in final_aggregated
            and f"{metric_name}_mean" in final_aggregated[depth_label]
            for depth_label in sorted_depths
        )

    def _append_group(category: str, group: tuple[str, list]) -> None:
        for _i, (cat, groups) in enumerate(metric_pairs):
            if cat == category:
                groups.append(group)
                break

    if _has_metric("I(Vb;C|E,I)"):
        _append_group(
            "ablation_proxy_branch_mean",
            (
                "I(Vb;C|E,I)",
                [
                    ("I(Vb;C|E,I)", "#1f77b4", "DendriNet"),
                    ("I(Vb_lin;C|E_lin,I_lin)", "#ff7f0e", r"LDA$_{+}$"),
                    ("I(Vb_sh;C|E_sh,I_sh)", "#7f7f7f", "Random"),
                ],
            ),
        )
        _append_group(
            "ablation_proxy_sum",
            (
                "I(Vb;C|E,I)_sum",
                [
                    ("I(Vb;C|E,I)_sum", "#1f77b4", "DendriNet"),
                    ("I(Vb_lin;C|E_lin,I_lin)_sum", "#ff7f0e", r"LDA$_{+}$"),
                    ("I(Vb_sh;C|E_sh,I_sh)_sum", "#7f7f7f", "Random"),
                ],
            ),
        )
        _append_group(
            "ablation_proxy_union",
            (
                "I(Vb;C|E,I)_union",
                [
                    ("I(Vb;C|E,I)_union", "#1f77b4", "DendriNet"),
                    ("I(Vb_lin;C|E_lin,I_lin)_union", "#ff7f0e", r"LDA$_{+}$"),
                    ("I(Vb_sh;C|E_sh,I_sh)_union", "#7f7f7f", "Random"),
                ],
            ),
        )

    # If layer_total_topk proxies were computed, also plot the *_topK_sum variants.
    topk_k = None
    topk_values = set()
    for depth_stats in final_aggregated.values():
        for key in depth_stats.keys():
            if not key.endswith("_mean"):
                continue
            idx = key.find("_top")
            if idx == -1:
                continue
            idx += len("_top")
            j = key.find("_sum_mean", idx)
            if j == -1:
                continue
            num = key[idx:j]
            if num.isdigit():
                topk_values.add(int(num))

    if topk_values:
        topk_k = max(topk_values)
        suffix_topk = f"_top{topk_k}_sum"

        def _suffix_metrics(metrics, suffix):
            return tuple(f"{m}{suffix}" for m in metrics)

        metric_pairs.append(
            (
                "ablation_proxy_topk_sum",
                [
                    (
                        f"I(E;C|I,Vb){suffix_topk}",
                        [
                            (
                                _suffix_metrics(
                                    ("I(E;C|I,Vb)", "I(E;C|I)"), suffix_topk
                                ),
                                "#1f77b4",
                                "DendriNet",
                            ),
                            (
                                _suffix_metrics(
                                    (
                                        "I(E_lin;C|I_lin,Vb_lin)",
                                        "I(E_lin;C|I_lin)",
                                        # Backward-compat
                                        "I(E_lin;C|I,Vb)",
                                        "I(E_lin;C|I)",
                                    ),
                                    suffix_topk,
                                ),
                                "#ff7f0e",
                                r"LDA$_{+}$",
                            ),
                            (
                                _suffix_metrics(
                                    ("I(E_sh;C|I_sh,Vb_sh)", "I(E_sh;C|I_sh)"),
                                    suffix_topk,
                                ),
                                "#7f7f7f",
                                "Random",
                            ),
                        ],
                    ),
                    (
                        f"I(I;C|E,Vb){suffix_topk}",
                        [
                            (
                                _suffix_metrics(
                                    ("I(I;C|E,Vb)", "I(I;C|E)"), suffix_topk
                                ),
                                "#1f77b4",
                                "DendriNet",
                            ),
                            (
                                _suffix_metrics(
                                    (
                                        "I(I_lin;C|E_lin,Vb_lin)",
                                        "I(I_lin;C|E_lin)",
                                        # Backward-compat
                                        "I(I_lin;C|E,Vb)",
                                        "I(I_lin;C|E)",
                                    ),
                                    suffix_topk,
                                ),
                                "#ff7f0e",
                                r"LDA$_{+}$",
                            ),
                            (
                                _suffix_metrics(
                                    ("I(I_sh;C|E_sh,Vb_sh)", "I(I_sh;C|E_sh)"),
                                    suffix_topk,
                                ),
                                "#7f7f7f",
                                "Random",
                            ),
                        ],
                    ),
                    (
                        f"I(E,I;C|Vb){suffix_topk}",
                        [
                            (
                                _suffix_metrics(
                                    ("I(E,I;C|Vb)", "I(E,I;C)"), suffix_topk
                                ),
                                "#1f77b4",
                                "DendriNet",
                            ),
                            (
                                _suffix_metrics(
                                    (
                                        "I(E_lin,I_lin;C|Vb_lin)",
                                        "I(E_lin,I_lin;C)",
                                        # Backward-compat
                                        "I(E_lin,I_lin;C|Vb)",
                                    ),
                                    suffix_topk,
                                ),
                                "#ff7f0e",
                                r"LDA$_{+}$",
                            ),
                            (
                                _suffix_metrics(
                                    ("I(E_sh,I_sh;C|Vb_sh)", "I(E_sh,I_sh;C)"),
                                    suffix_topk,
                                ),
                                "#7f7f7f",
                                "Random",
                            ),
                        ],
                    ),
                    (
                        f"I(Vb;C){suffix_topk}",
                        [
                            (f"I(Vb;C){suffix_topk}", "#1f77b4", "DendriNet"),
                            (f"I(Vb_lin;C){suffix_topk}", "#ff7f0e", r"LDA$_{+}$"),
                            (f"I(Vb_sh;C){suffix_topk}", "#7f7f7f", "Random"),
                        ],
                    ),
                    (
                        f"I(Vb;C|Vout){suffix_topk}",
                        [
                            (f"I(Vb;C|Vout){suffix_topk}", "#1f77b4", "DendriNet"),
                            (
                                f"I(Vb_lin;C|Vout_lin){suffix_topk}",
                                "#ff7f0e",
                                r"LDA$_{+}$",
                            ),
                            (
                                f"I(Vb_sh;C|Vout_sh){suffix_topk}",
                                "#7f7f7f",
                                "Random",
                            ),
                        ],
                    ),
                    (
                        f"I(Vout;C|Vb){suffix_topk}",
                        [
                            (
                                _suffix_metrics(
                                    ("I(Vout;C|Vb)", "I(Vout;C)"), suffix_topk
                                ),
                                "#1f77b4",
                                "DendriNet",
                            ),
                            (
                                _suffix_metrics(
                                    (
                                        "I(Vout_lin;C|Vb_lin)",
                                        "I(Vout_lin;C)",
                                        # Backward-compat
                                        "I(Vout_lin;C|Vb)",
                                    ),
                                    suffix_topk,
                                ),
                                "#ff7f0e",
                                r"LDA$_{+}$",
                            ),
                            (
                                _suffix_metrics(
                                    ("I(Vout_sh;C|Vb_sh)", "I(Vout_sh;C)"),
                                    suffix_topk,
                                ),
                                "#7f7f7f",
                                "Random",
                            ),
                        ],
                    ),
                    (
                        f"I(Vout;C|E,I){suffix_topk}",
                        [
                            (f"I(Vout;C|E,I){suffix_topk}", "#1f77b4", "DendriNet"),
                            (
                                f"I(Vout_lin;C|E_lin,I_lin){suffix_topk}",
                                "#ff7f0e",
                                r"LDA$_{+}$",
                            ),
                            (
                                f"I(Vout_sh;C|E_sh,I_sh){suffix_topk}",
                                "#7f7f7f",
                                "Random",
                            ),
                        ],
                    ),
                ],
            )
        )

        # If upstream-unique CMI was computed, also plot its top-K sum proxy.
        if _has_metric(f"I(Vb;C|E,I){suffix_topk}"):
            metric_pairs[-1][1].append(
                (
                    f"I(Vb;C|E,I){suffix_topk}",
                    [
                        (f"I(Vb;C|E,I){suffix_topk}", "#1f77b4", "DendriNet"),
                        (
                            f"I(Vb_lin;C|E_lin,I_lin){suffix_topk}",
                            "#ff7f0e",
                            r"LDA$_{+}$",
                        ),
                        (
                            f"I(Vb_sh;C|E_sh,I_sh){suffix_topk}",
                            "#7f7f7f",
                            "Random",
                        ),
                    ],
                )
            )

    has_any_lda = any(
        depth_label in final_aggregated
        and any("lin" in k for k in final_aggregated[depth_label].keys())
        for depth_label in sorted_depths
    )

    def _metric_candidates(metric_spec):
        if isinstance(metric_spec, (list, tuple)):
            return list(metric_spec)
        return [metric_spec]

    def _render_and_save_metric_groups(
        metric_category: str,
        metric_groups: list,
        *,
        part_idx: Optional[int] = None,
        n_parts: Optional[int] = None,
    ) -> None:
        def _group_has_available_data(metrics_list: list) -> bool:
            for metric, _color, _label in metrics_list:
                metric_list = _metric_candidates(metric)
                if any(
                    (
                        depth_label in final_aggregated
                        and any(
                            f"{m}_mean" in final_aggregated[depth_label]
                            for m in metric_list
                        )
                    )
                    for depth_label in sorted_depths
                ):
                    return True
            return False

        metric_groups = [
            (plot_title, metrics_list)
            for plot_title, metrics_list in metric_groups
            if _group_has_available_data(metrics_list)
        ]
        if not metric_groups:
            logger.info(
                "Skipping %s%s: no available metrics in aggregated results",
                metric_category,
                (
                    f" part {part_idx}/{n_parts}"
                    if part_idx is not None and n_parts is not None
                    else ""
                ),
            )
            return

        n_plots = len(metric_groups)
        n_cols = 3  # Use 3 columns for better layout
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        for idx, (plot_title, metrics_list) in enumerate(metric_groups):
            if idx >= len(axes):
                break
            ax = axes[idx]
            has_data = False

            # Plot each metric in this comparison group
            for metric, color, label in metrics_list:
                metric_list = _metric_candidates(metric)

                # Only show lines for metrics that exist in the aggregated data.
                # This prevents LDA labels from appearing when compute_lda_weights=False.
                if not any(
                    (
                        depth_label in final_aggregated
                        and any(
                            f"{m}_mean" in final_aggregated[depth_label]
                            for m in metric_list
                        )
                    )
                    for depth_label in sorted_depths
                ):
                    continue

                values = []
                errors = []
                x_positions = []

                for depth_idx, depth_label in enumerate(sorted_depths):
                    if depth_label in final_aggregated:
                        depth_stats = final_aggregated[depth_label]
                        chosen = next(
                            (m for m in metric_list if f"{m}_mean" in depth_stats),
                            None,
                        )
                        if chosen is not None:
                            x_positions.append(depth_idx)
                            values.append(depth_stats[f"{chosen}_mean"])
                            errors.append(depth_stats.get(f"{chosen}_std", 0))

                if x_positions and values:
                    valid_points = [
                        (x, v, e)
                        for x, v, e in zip(x_positions, values, errors)
                        if np.isfinite(v) and np.isfinite(e)
                    ]
                    if not valid_points:
                        continue

                    has_data = True
                    x_pos = np.asarray([x for x, _, _ in valid_points], dtype=float)
                    values = np.asarray([v for _, v, _ in valid_points], dtype=float)
                    errors = np.asarray([e for _, _, e in valid_points], dtype=float)

                    ax.errorbar(
                        x_pos,
                        values,
                        yerr=errors,
                        marker="o",
                        linewidth=2,
                        markersize=6,
                        label=label,
                        color=color,
                        capsize=3,
                    )

                    ax.fill_between(
                        x_pos,
                        values - errors,
                        values + errors,
                        alpha=0.15,
                        color=color,
                    )

            if has_data:
                ax.set_xlabel("Dendritic Depth")
                if plot_variance:
                    ax.set_ylabel("MI variance (bits^2)")
                    ax.set_title(f"{plot_title} (variance)")
                else:
                    ax.set_ylabel("MI (bits)")
                    ax.set_title(plot_title)

                # Add placeholder markers for depths where *no* plotted series has data.
                available_metrics = []
                for m, _c, _l in metrics_list:
                    candidates = _metric_candidates(m)
                    if any(
                        (
                            depth_label in final_aggregated
                            and any(
                                f"{cand}_mean" in final_aggregated[depth_label]
                                for cand in candidates
                            )
                        )
                        for depth_label in sorted_depths
                    ):
                        available_metrics.append(candidates)
                if available_metrics:
                    missing_x = []
                    for depth_idx, depth_label in enumerate(sorted_depths):
                        if depth_label not in final_aggregated:
                            missing_x.append(depth_idx)
                            continue
                        depth_stats = final_aggregated[depth_label]
                        if not any(
                            f"{cand}_mean" in depth_stats
                            for cand_list in available_metrics
                            for cand in cand_list
                        ):
                            missing_x.append(depth_idx)

                    if missing_x:
                        from matplotlib.transforms import blended_transform_factory

                        trans = blended_transform_factory(ax.transData, ax.transAxes)
                        ax.scatter(
                            missing_x,
                            [0.05] * len(missing_x),
                            marker="x",
                            s=35,
                            color="#666666",
                            alpha=0.6,
                            transform=trans,
                            clip_on=False,
                            label="_nolegend_",
                        )
                ax.set_xticks(range(len(sorted_depths)))
                ax.set_xticklabels(sorted_depths, rotation=0)
                ax.grid(True, alpha=0.3)
                ax.legend(loc="best")
            else:
                data_type = "variance" if plot_variance else "data"
                ax.text(
                    0.5,
                    0.5,
                    f"No {data_type} for {plot_title}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        # Remove empty subplots
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        plot_type = "Variance" if plot_variance else "Analysis"
        comparison_text = r"DendriNet vs LDA$_{+}$" if has_any_lda else "DendriNet"

        title = f"Information {plot_type}: {metric_category.replace('_', ' ').title()} ({comparison_text})"
        if part_idx is not None and n_parts is not None:
            title += f" [part {part_idx}/{n_parts}]"

        # Add a short note describing proxy aggregation in the title.
        if metric_category == "ablation_proxy_sum":
            title += "\n(proxy sum: sum over branches)"
        elif metric_category == "ablation_proxy_branch_mean":
            title += "\n(per-branch mean: average across branches; compare to *_sum for branch-count scaling)"
        elif metric_category == "ablation_proxy_union":
            title += "\n(proxy union: H(C) * (1 - prod_b(1 - I_b/H(C))))"
        elif metric_category == "ablation_proxy_topk_sum":
            if topk_k is not None:
                title += f"\n(proxy top-{topk_k} sum: sum of top-{topk_k} branches)"
            else:
                title += "\n(proxy top-K sum: sum of top-K branches)"
        if metric_category in {
            "ablation_proxy_sum",
            "ablation_proxy_branch_mean",
            "ablation_proxy_union",
            "ablation_proxy_topk_sum",
        }:
            title += "\n(note: most distal has no Vb; '|Vb' metrics fall back to no-Vb)"

        fig.suptitle(title, fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.94])

        suffix = "_variance" if plot_variance else ""
        part_suffix = f"_part{part_idx}" if part_idx is not None else ""
        filename = (
            f"aggregated_{metric_category}_comparison{suffix}_depth{part_suffix}.png"
        )
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved aggregated comparison plot: {filename}")

    # Generate plots for each metric category
    for metric_category, metric_groups in metric_pairs:
        # Conditional MI panels are dense; save as two 6-panel figures (top 6 + second 6).
        if metric_category == "conditional_mi" and len(metric_groups) > 6:
            chunks = [metric_groups[i : i + 6] for i in range(0, len(metric_groups), 6)]
            for part_idx, chunk in enumerate(chunks, start=1):
                _render_and_save_metric_groups(
                    metric_category,
                    chunk,
                    part_idx=part_idx,
                    n_parts=len(chunks),
                )
        else:
            _render_and_save_metric_groups(metric_category, metric_groups)


def plot_information_variance_analysis(
    variance_input: dict,
    save_path: str,
    somatic_synapses: bool = True,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Simple wrapper that calls plot_fully_aggregated_analysis with variance=True.

    This function exists for backward compatibility with the analysis pipeline.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Extract layer_variances if it's wrapped in variance_analysis dict
    if isinstance(variance_input, dict) and "layer_variances" in variance_input:
        layer_statistics = variance_input.get("layer_variances", {})
    else:
        layer_statistics = variance_input

    # Call the enhanced aggregated analysis function with variance=True
    plot_fully_aggregated_analysis(
        layer_statistics=layer_statistics,
        save_path=save_path,
        logger=logger,
        somatic_synapses=somatic_synapses,
        plot_variance=True,
    )


def plot_lda_vs_trained_performance(
    comparison_results: dict,
    save_path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot comparison of network performance with trained weights vs LDA-optimal weights.

    Args:
        comparison_results: Dictionary from InformationAnalyzer.compare_lda_vs_trained_performance
            containing 'trained_accuracy', 'lda_accuracy', 'accuracy_difference'
        save_path: Directory path where the plot will be saved
        logger: Optional logger for output messages
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    trained_acc = comparison_results.get("trained_accuracy", 0.0)
    lda_acc = comparison_results.get("lda_accuracy", 0.0)
    diff = comparison_results.get("accuracy_difference", 0.0)

    if lda_acc is None:
        logger.warning(
            "LDA accuracy not available, skipping performance comparison plot"
        )
        return

    _fig, ax = plt.subplots(figsize=(6, 5))

    # Bar chart comparing accuracies
    x_pos = [0, 1]
    accuracies = [trained_acc * 100, lda_acc * 100]
    colors = ["#1f77b4", "#ff7f0e"]
    labels = ["Trained Weights", r"LDA$_{+}$ Weights"]

    bars = ax.bar(x_pos, accuracies, color=colors, width=0.6, edgecolor="black")

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(
            f"{acc:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title(r"DendriNet Performance: Trained vs LDA$_{+}$ Weights", fontsize=14)

    # Add difference annotation
    ax.annotate(
        f"Δ = {diff * 100:+.2f}%",
        xy=(0.5, max(accuracies) + 3),
        xytext=(0.5, max(accuracies) + 5),
        ha="center",
        fontsize=11,
        color="green" if diff >= 0 else "red",
    )

    # Set y-axis limits
    ax.set_ylim(0, max(accuracies) + 10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, "lda_vs_trained_performance.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved LDA vs trained performance plot: {filepath}")
    logger.info(f"Trained accuracy: {trained_acc:.4f}, LDA accuracy: {lda_acc:.4f}")

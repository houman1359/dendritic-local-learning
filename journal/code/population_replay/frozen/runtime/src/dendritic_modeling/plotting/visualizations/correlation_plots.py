"""
Correlation analysis plotting functions.

This module provides visualization functions for hierarchical correlation analysis results,
similar to the information analysis plots.
"""

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .plotting_utils import get_color_scheme, handle_plot_error, save_plot


def plot_hierarchical_correlations(
    results: dict[str, Any], save_path: str, filename_prefix: str = "correlation"
) -> None:
    """Plot hierarchical correlation analysis results.

    Args:
        results: Correlation analysis results with hierarchical structure
        save_path: Directory to save plots
        filename_prefix: Prefix for saved files
    """
    try:
        # Create output directory
        os.makedirs(save_path, exist_ok=True)

        # Plot network-level summary
        if "network_level" in results:
            plot_network_summary(results["network_level"], save_path=save_path)

        # Plot per-layer results if available
        if "per_layer_analysis" in results and results.get(
            "per_layer_analysis_enabled", False
        ):
            plot_per_layer_correlations(
                results["per_layer_analysis"], save_path=save_path
            )

        # Plot per-EI network results if available
        if "per_einet_analysis" in results and results.get(
            "per_einet_analysis_enabled", False
        ):
            plot_per_ei_correlations(results["per_einet_analysis"], save_path=save_path)

        # Plot per-neuron/unit summary results if available
        if "per_neuron_analysis" in results and results.get(
            "per_neuron_analysis_enabled", False
        ):
            plot_per_neuron_correlations(
                results["per_neuron_analysis"], save_path=save_path
            )

    except Exception as e:
        handle_plot_error("hierarchical correlations", e)


def plot_network_summary(network_results: dict[str, Any], save_path: str) -> None:
    """Plot comprehensive correlation analysis with clear comparisons.

    Args:
        network_results: Network-level correlation results
        save_path: Path to save the plot
    """
    try:
        # Create a comprehensive comparison plot
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 5, figure=fig, hspace=0.4, wspace=0.3)

        # Get summary statistics
        summary_stats = network_results.get("summary_stats", {})

        # Section 1: Correlation Type Comparison (Top row)
        correlation_types = ["EE", "II", "EI", "E_Vout", "I_Vout"]
        colors = ["#e74c3c", "#3498db", "#9b59b6", "#f39c12", "#2ecc71"]

        ax_types = fig.add_subplot(gs[0, :3])
        if summary_stats:
            type_values = []
            type_labels = []
            type_colors = []

            for i, corr_type in enumerate(correlation_types):
                if f"{corr_type}_total_mean" in summary_stats:
                    type_values.append(summary_stats[f"{corr_type}_total_mean"])
                    type_labels.append(corr_type.replace("_", "-"))
                    type_colors.append(colors[i])

            if type_values:
                bars = ax_types.bar(
                    type_labels, type_values, color=type_colors, alpha=0.7
                )
                ax_types.set_title(
                    "Correlation Strengths by Type", fontsize=14, fontweight="bold"
                )
                ax_types.set_ylabel("Mean Correlation")
                ax_types.grid(True, alpha=0.3, axis="y")

                # Add value labels
                for bar, val in zip(bars, type_values):
                    ax_types.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{val:.3f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        # Section 2: Total vs Noise vs Signal Comparison (Top right)
        ax_comparison = fig.add_subplot(gs[0, 3:])
        plot_total_vs_noise_correlations(summary_stats, ax_comparison)
        ax_comparison.set_title(
            "Total vs Noise vs Signal", fontsize=14, fontweight="bold"
        )

        # Section 3: EI Layer Comparison (Second row, left)
        ax_ei_layers = fig.add_subplot(gs[1, :2])
        ax_ei_layers.set_title(
            "Comparison Across EI Layers", fontsize=14, fontweight="bold"
        )
        ax_ei_layers.text(
            0.5,
            0.5,
            "EI Layer 0 vs 1 vs 2\n(Network Depth)",
            ha="center",
            va="center",
            transform=ax_ei_layers.transAxes,
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightblue"},
        )

        # Section 4: E vs I Network Comparison (Second row, middle)
        ax_e_vs_i = fig.add_subplot(gs[1, 2])
        plot_ei_balance(summary_stats, ax_e_vs_i)
        ax_e_vs_i.set_title("E vs I Networks", fontsize=14, fontweight="bold")

        # Section 5: Dendritic Branch Layer Comparison (Second row, right)
        ax_dendrite = fig.add_subplot(gs[1, 3:])
        ax_dendrite.set_title("Dendritic Branch Layers", fontsize=14, fontweight="bold")
        ax_dendrite.text(
            0.5,
            0.5,
            "Branch Layer 0 vs 1 vs 2\n(Dendritic Depth)",
            ha="center",
            va="center",
            transform=ax_dendrite.transAxes,
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgreen"},
        )

        # Section 6: Input-Output Correlations (Third row)
        ax_io = fig.add_subplot(gs[2, :2])
        plot_signal_correlations(summary_stats, ax_io)
        ax_io.set_title("Input-Output Correlations", fontsize=14, fontweight="bold")

        # Section 7: Correlation Matrix Heatmap (Third row, right)
        ax_matrix = fig.add_subplot(gs[2, 2:])
        # Create a simplified correlation matrix visualization
        if summary_stats:
            matrix_data = []
            labels = []
            for corr_type in correlation_types:
                if f"{corr_type}_total_mean" in summary_stats:
                    matrix_data.append(summary_stats[f"{corr_type}_total_mean"])
                    labels.append(corr_type.replace("_", "-"))

            if matrix_data:
                # Create a simple correlation strength visualization
                matrix = np.array(matrix_data).reshape(1, -1)
                im = ax_matrix.imshow(
                    matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1
                )
                ax_matrix.set_xticks(range(len(labels)))
                ax_matrix.set_xticklabels(labels, rotation=45)
                ax_matrix.set_yticks([])
                ax_matrix.set_title(
                    "Correlation Strength Overview", fontsize=14, fontweight="bold"
                )
                plt.colorbar(im, ax=ax_matrix, shrink=0.6)

        # Section 8: Tuning Summary (Bottom row)
        if "tuning" in network_results:
            ax_tuning = fig.add_subplot(gs[3, :])
            plot_tuning_summary(network_results["tuning"], ax_tuning)
            ax_tuning.set_title(
                "Selectivity & Tuning Analysis", fontsize=14, fontweight="bold"
            )

        # Add explanatory text
        fig.text(
            0.02,
            0.95,
            "Correlation Analysis Structure:",
            fontsize=12,
            fontweight="bold",
        )
        fig.text(0.02, 0.92, "• EI Layers: Network depth (0 to 1 to 2)", fontsize=10)
        fig.text(
            0.02, 0.90, "• E/I Networks: Excitatory vs Inhibitory pathways", fontsize=10
        )
        fig.text(
            0.02, 0.88, "• Dendritic Layers: Branch depth (0 to 1 to 2)", fontsize=10
        )

        plt.suptitle(
            "Comprehensive Correlation Analysis\nComparisons: EI Layers | E/I Networks | Dendritic Branch Layers",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )
        save_plot(fig, save_path, "comprehensive_correlation_analysis")

    except Exception as e:
        handle_plot_error("network summary", e)


def plot_correlation_comparisons(results: dict[str, Any], save_path: str) -> None:
    """Create dedicated comparison plots for EI layers, E/I networks, and dendritic layers.

    Args:
        results: Full correlation analysis results
        save_path: Directory to save plots
    """
    try:
        # Create three separate comparison plots

        # 1. EI Layer Comparison
        if "per_layer_analysis" in results:
            fig1, axes1 = plt.subplots(2, 3, figsize=(18, 12))
            fig1.suptitle(
                "EI Layer Comparison\n(How correlations change with network depth)",
                fontsize=16,
                fontweight="bold",
            )

            layer_data = results["per_layer_analysis"]
            correlation_types = ["EE", "II", "EI", "E_Vout", "I_Vout", "Vout_Vout"]

            def _parse_ei_layer_idx(layer_name: str):
                """Best-effort parse EI layer index from module names.

                Supports:
                - "core_network.layers.0.excitatory_cells.branch_layers.1"
                - "layer_0" / "layers_0"
                """
                s = str(layer_name)
                if "layers." in s:
                    try:
                        after = s.split("layers.", 1)[1]
                        return int(after.split(".", 1)[0])
                    except Exception:
                        pass
                for token in ("layer_", "layers_"):
                    if token in s:
                        try:
                            tail = s.split(token, 1)[1]
                            return int(tail.split(".", 1)[0].split("_", 1)[0])
                        except Exception:
                            pass
                try:
                    return int(s.split("_")[-1])
                except Exception:
                    return None

            for i, corr_type in enumerate(correlation_types[:6]):
                ax = axes1.flatten()[i]

                # Collect data across EI layers
                layers = []
                values = []

                for layer_name, layer_stats in layer_data.items():
                    if corr_type in layer_stats:
                        try:
                            layer_idx = _parse_ei_layer_idx(layer_name)
                            if layer_idx is None:
                                continue
                            layers.append(layer_idx)

                            corr_data = layer_stats[corr_type]
                            if isinstance(corr_data, dict) and "total" in corr_data:
                                if hasattr(corr_data["total"], "mean"):
                                    values.append(float(corr_data["total"].mean()))
                                else:
                                    values.append(float(corr_data["total"]))
                            else:
                                values.append(float(corr_data))
                        except (ValueError, AttributeError):
                            continue

                if layers and values:
                    # Sort by layer index
                    sorted_data = sorted(zip(layers, values))
                    layers, values = zip(*sorted_data)

                    ax.plot(layers, values, "o-", linewidth=2, markersize=8)
                    ax.set_title(
                        f"{corr_type} Correlation", fontsize=12, fontweight="bold"
                    )
                    ax.set_xlabel("EI Layer")
                    ax.set_ylabel("Correlation Strength")
                    ax.grid(True, alpha=0.3)

                    # Add value labels
                    for x, y in zip(layers, values):
                        ax.annotate(
                            f"{y:.3f}",
                            (x, y),
                            textcoords="offset points",
                            xytext=(0, 10),
                            ha="center",
                        )

            plt.tight_layout()
            save_plot(fig1, save_path, "ei_layer_comparison")

        # 2. E vs I Network Comparison
        if "per_einet_analysis" in results:
            fig2, axes2 = plt.subplots(2, 2, figsize=(15, 12))
            fig2.suptitle(
                "Excitatory vs Inhibitory Network Comparison\n(How E and I pathways differ)",
                fontsize=16,
                fontweight="bold",
            )

            einet_data = results["per_einet_analysis"]

            # Compare excitatory vs inhibitory results
            if (
                "excitatory_results" in einet_data
                and "inhibitory_results" in einet_data
            ):
                exc_data = einet_data["excitatory_results"]
                inh_data = einet_data["inhibitory_results"]

                comparison_metrics = ["EE", "II", "E_Vout", "I_Vout"]

                for i, metric in enumerate(comparison_metrics):
                    if i < 4:
                        ax = axes2.flatten()[i]

                        exc_val = 0
                        inh_val = 0

                        # Extract values from excitatory and inhibitory results
                        if (
                            "correlations" in exc_data
                            and metric in exc_data["correlations"]
                        ):
                            exc_corr = exc_data["correlations"][metric]
                            if isinstance(exc_corr, dict) and "total" in exc_corr:
                                exc_val = float(
                                    exc_corr["total"].mean()
                                    if hasattr(exc_corr["total"], "mean")
                                    else exc_corr["total"]
                                )

                        if (
                            "correlations" in inh_data
                            and metric in inh_data["correlations"]
                        ):
                            inh_corr = inh_data["correlations"][metric]
                            if isinstance(inh_corr, dict) and "total" in inh_corr:
                                inh_val = float(
                                    inh_corr["total"].mean()
                                    if hasattr(inh_corr["total"], "mean")
                                    else inh_corr["total"]
                                )

                        # Create comparison bar plot
                        networks = ["Excitatory", "Inhibitory"]
                        values = [exc_val, inh_val]
                        colors = ["#e74c3c", "#3498db"]

                        bars = ax.bar(networks, values, color=colors, alpha=0.7)
                        ax.set_title(
                            f"{metric} Correlation", fontsize=12, fontweight="bold"
                        )
                        ax.set_ylabel("Correlation Strength")
                        ax.grid(True, alpha=0.3, axis="y")

                        # Add value labels
                        for bar, val in zip(bars, values):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.01,
                                f"{val:.3f}",
                                ha="center",
                                va="bottom",
                                fontweight="bold",
                            )

            plt.tight_layout()
            save_plot(fig2, save_path, "e_vs_i_network_comparison")

        # 3. Dendritic Branch Layer Comparison
        fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
        fig3.suptitle(
            "Dendritic Branch Layer Comparison\n(How correlations change with dendritic depth)",
            fontsize=16,
            fontweight="bold",
        )

        # Extract data for dendritic layers if available
        dendritic_data = {}
        if "per_layer_correlations" in results:
            for layer_key, layer_data in results["per_layer_correlations"].items():
                # Try to extract dendritic layer info from the key
                if "branch_layer" in layer_key or "layer" in layer_key:
                    dendritic_data[layer_key] = layer_data

        # If we have network-wide data, use that as fallback
        if not dendritic_data and "network_correlations" in results:
            dendritic_data["Network-wide"] = results["network_correlations"]

        # Plot different correlation types across dendritic layers
        plot_idx = 0
        corr_types = [
            ("E-E Total", "EE_corr_total", "Blues"),
            ("I-I Total", "II_corr_total", "Reds"),
            ("E-I Total", "EI_corr_total", "Greens"),
            ("E-E Noise", "EE_corr_noise", "Blues"),
            ("I-I Noise", "II_corr_noise", "Reds"),
            ("E-I Signal", "EI_corr_signal", "Greens"),
        ]

        for title, key, cmap in corr_types:
            if plot_idx >= 6:
                break
            ax = axes3.flatten()[plot_idx]

            # Collect data across layers
            layer_names = []
            corr_values = []

            for layer_name, layer_data in sorted(dendritic_data.items()):
                if key in layer_data and layer_data[key] is not None:
                    layer_names.append(
                        layer_name.split(".")[-1] if "." in layer_name else layer_name
                    )
                    corr_values.append(layer_data[key])

            if corr_values:
                # Create bar plot
                x = np.arange(len(layer_names))
                bars = ax.bar(x, corr_values, color=plt.cm.get_cmap(cmap)(0.6))
                ax.set_xticks(x)
                ax.set_xticklabels(layer_names, rotation=45, ha="right")
                ax.set_ylabel("Correlation")
                ax.set_title(title, fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-1, 1)

                # Add value labels on bars
                for bar, val in zip(bars, corr_values):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.02 * np.sign(height),
                        f"{val:.3f}",
                        ha="center",
                        va="bottom" if height > 0 else "top",
                        fontsize=8,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"{title}\n(No data available)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                    bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray"},
                )
                ax.set_title(title, fontsize=12)

            plot_idx += 1

        plt.tight_layout()
        save_plot(fig3, save_path, "dendritic_layer_comparison")

    except Exception as e:
        handle_plot_error("correlation comparisons", e)


def plot_correlation_strengths(stats: dict[str, float], ax: plt.Axes) -> None:
    """Plot mean correlation strengths for different correlation types."""
    correlation_types = ["EE", "II", "EI", "E_Vout", "I_Vout"]
    conditions = ["total", "noise"]

    data = []
    labels = []
    colors = []
    color_scheme = get_color_scheme()

    for corr_type in correlation_types:
        for cond in conditions:
            key = f"{corr_type}_{cond}_mean"
            if key in stats:
                data.append(stats[key])
                labels.append(f"{corr_type}\n{cond}")
                colors.append(
                    color_scheme["excitatory"]
                    if "E" in corr_type
                    else color_scheme["inhibitory"]
                )

    if data:
        bars = ax.bar(range(len(data)), data, color=colors, alpha=0.7)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Mean |Correlation|")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, val in zip(bars, data):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )


def plot_total_vs_noise_correlations(stats: dict[str, float], ax: plt.Axes) -> None:
    """Plot comparison of total vs noise correlations."""
    correlation_types = ["EE", "II", "EI"]

    total_vals = []
    noise_vals = []

    for corr_type in correlation_types:
        total_key = f"{corr_type}_total_mean"
        noise_key = f"{corr_type}_noise_mean"

        if total_key in stats and noise_key in stats:
            total_vals.append(stats[total_key])
            noise_vals.append(stats[noise_key])

    if total_vals and noise_vals:
        x = np.arange(len(correlation_types))
        width = 0.35

        _ = ax.bar(x - width / 2, total_vals, width, label="Total", alpha=0.7)
        _ = ax.bar(x + width / 2, noise_vals, width, label="Noise", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(correlation_types)
        ax.set_ylabel("Mean |Correlation|")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")


def plot_signal_correlations(stats: dict[str, float], ax: plt.Axes) -> None:
    """Plot signal correlations (between-class differences)."""
    correlation_types = ["EE", "II", "EI", "E_Vout", "I_Vout"]

    signal_vals = []
    labels = []

    for corr_type in correlation_types:
        signal_key = f"{corr_type}_signal_mean"
        if signal_key in stats:
            signal_vals.append(stats[signal_key])
            labels.append(corr_type)

    if signal_vals:
        _ = ax.bar(range(len(signal_vals)), signal_vals, alpha=0.7)
        ax.set_xticks(range(len(signal_vals)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Mean Signal Correlation")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")


def plot_ei_balance(stats: dict[str, float], ax: plt.Axes) -> None:
    """Plot E-I correlation balance metrics."""
    metrics = {}

    # E/I ratio for different conditions
    if "EE_total_mean" in stats and "II_total_mean" in stats:
        metrics["E/I Total"] = stats["EE_total_mean"] / max(
            stats["II_total_mean"], 1e-6
        )

    if "EE_noise_mean" in stats and "II_noise_mean" in stats:
        metrics["E/I Noise"] = stats["EE_noise_mean"] / max(
            stats["II_noise_mean"], 1e-6
        )

    # E-I cross-correlation strength
    if "EI_total_mean" in stats:
        metrics["E-I Cross"] = stats["EI_total_mean"]

    if metrics:
        _ = ax.bar(range(len(metrics)), list(metrics.values()), alpha=0.7)
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels(list(metrics.keys()), rotation=45, ha="right")
        ax.set_ylabel("Value")
        ax.axhline(y=1, color="k", linestyle="--", alpha=0.5, label="Balance")
        ax.grid(True, alpha=0.3, axis="y")


def plot_tuning_summary(tuning: dict[str, Any], ax: plt.Axes) -> None:
    """Plot summary of tuning curves."""
    # Check if we have binary or multi-class tuning
    if "E_tuning" in tuning and "I_tuning" in tuning:
        # Binary classification - violin plots of tuning differences
        data = [tuning["E_tuning"].flatten(), tuning["I_tuning"].flatten()]

        parts = ax.violinplot(data, positions=[0, 1], showmeans=True, showmedians=True)

        # Customize colors
        colors = get_color_scheme()
        parts["bodies"][0].set_facecolor(colors["excitatory"])
        parts["bodies"][1].set_facecolor(colors["inhibitory"])

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Excitatory", "Inhibitory"])
        ax.set_ylabel("Tuning Strength")
        ax.axhline(y=0, color="k", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3, axis="y")
    elif "E_selectivity" in tuning:
        # Multi-class - bar plot of mean selectivity metrics
        metrics = []
        labels = []
        colors_list = []

        colors = get_color_scheme()
        for var_type, color in [
            ("E", colors["excitatory"]),
            ("I", colors["inhibitory"]),
            ("Vout", "green"),
        ]:
            if f"{var_type}_selectivity" in tuning:
                sel_data = tuning[f"{var_type}_selectivity"]
                metrics.extend(
                    [sel_data["mean_selectivity"], sel_data["mean_sparseness"]]
                )
                labels.extend([f"{var_type} Selectivity", f"{var_type} Sparseness"])
                colors_list.extend([color, color])

        if metrics:
            x = np.arange(len(metrics))
            bars = ax.bar(x, metrics, color=colors_list, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_ylabel("Mean Value")
            ax.set_title("Selectivity Metrics Summary")
            ax.grid(True, alpha=0.3, axis="y")

            # Add values on bars
            for bar, val in zip(bars, metrics):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )


def plot_per_layer_correlations(
    per_layer_results: dict[str, dict[str, Any]], save_path: str
) -> None:
    """Plot per-layer correlation results.

    Args:
        per_layer_results: Dictionary with layer names as keys
        save_path: Path to save the plot
    """
    try:
        n_layers = len(per_layer_results)
        fig, axes = plt.subplots(n_layers, 3, figsize=(15, 5 * n_layers), squeeze=False)

        for idx, (layer_name, layer_data) in enumerate(per_layer_results.items()):
            # Plot correlation matrix heatmap
            correlations = layer_data.get("correlations")
            if correlations is None:
                # Current analyzer format: top-level keys are correlation types.
                corr_types = ["EE", "II", "EI", "E_Vout", "I_Vout", "Vout_Vout"]
                correlations = {k: layer_data[k] for k in corr_types if k in layer_data}
            if correlations:
                plot_layer_correlation_matrix(
                    correlations, axes[idx, 0], f"Layer: {layer_name}"
                )

            # Plot summary statistics
            if "summary_stats" in layer_data:
                plot_layer_summary_stats(layer_data["summary_stats"], axes[idx, 1])

            # Plot tuning if available
            if "tuning" in layer_data:
                plot_layer_tuning(layer_data["tuning"], axes[idx, 2])

        plt.suptitle(
            "Per-Layer Correlation Analysis\nComparison Across EI Layers (Rows) and Analysis Types (Columns)",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        save_plot(fig, save_path, "per_layer_correlation_analysis")

    except Exception as e:
        handle_plot_error("per-layer correlations", e)


def plot_per_neuron_correlations(
    per_neuron_results: dict[str, Any], save_path: str
) -> None:
    """Plot per-unit correlation-strength distributions.

    Args:
        per_neuron_results: Output of CorrelationAnalyzer per-neuron summary
        save_path: Directory to save plots
    """
    try:
        by_type = per_neuron_results.get("by_type", {})
        if not isinstance(by_type, dict) or not by_type:
            return

        # Prefer "total" condition; fall back to any available condition.
        def pick_cond(d: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            if "total" in d and isinstance(d["total"], dict):
                return "total", d["total"]
            for k, v in d.items():
                if isinstance(v, dict):
                    return k, v
            return "total", {}

        corr_types = list(by_type.keys())
        n_rows = len(corr_types)
        fig, axes = plt.subplots(
            n_rows, 2, figsize=(14, max(3, 2.6 * n_rows)), squeeze=False
        )

        for r, corr_type in enumerate(corr_types):
            conds = by_type.get(corr_type, {})
            if not isinstance(conds, dict):
                continue
            cond, data = pick_cond(conds)

            ax_l = axes[r, 0]
            ax_r = axes[r, 1]

            if "per_unit_mean_abs" in data:
                vals = np.asarray(data["per_unit_mean_abs"], dtype=float)
                ax_l.hist(vals, bins=30, alpha=0.7)
                ax_l.set_title(f"{corr_type} ({cond}) per-unit |corr| mean")
                ax_l.set_xlabel("mean |corr|")
                ax_l.set_ylabel("count")
                ax_l.grid(True, alpha=0.3)
                ax_r.axis("off")
            elif "per_x_mean_abs" in data or "per_y_mean_abs" in data:
                x_vals = np.asarray(data.get("per_x_mean_abs", []), dtype=float)
                y_vals = np.asarray(data.get("per_y_mean_abs", []), dtype=float)
                ax_l.hist(x_vals, bins=30, alpha=0.7, color="#e74c3c")
                ax_l.set_title(f"{corr_type} ({cond}) x-side per-unit |corr| mean")
                ax_l.set_xlabel("mean |corr|")
                ax_l.set_ylabel("count")
                ax_l.grid(True, alpha=0.3)

                ax_r.hist(y_vals, bins=30, alpha=0.7, color="#3498db")
                ax_r.set_title(f"{corr_type} ({cond}) y-side per-unit |corr| mean")
                ax_r.set_xlabel("mean |corr|")
                ax_r.set_ylabel("count")
                ax_r.grid(True, alpha=0.3)
            else:
                ax_l.text(
                    0.5,
                    0.5,
                    "No per-unit summaries available",
                    ha="center",
                    va="center",
                    transform=ax_l.transAxes,
                )
                ax_l.axis("off")
                ax_r.axis("off")

        plt.suptitle(
            "Per-unit Correlation Strength Distributions",
            fontsize=16,
            fontweight="bold",
        )
        plt.tight_layout()
        save_plot(fig, save_path, "per_neuron_correlation_summary")

    except Exception as e:
        handle_plot_error("per-neuron correlations", e)


def plot_correlation_matrices(
    correlations: dict[str, dict[str, np.ndarray]], save_path: str, max_size: int = 50
) -> None:
    """Plot correlation matrices for different types and conditions.

    Args:
        correlations: Dictionary with correlation matrices
        save_path: Path to save the plot
        max_size: Maximum size to display (for large matrices)
    """
    try:
        # Count available correlation types
        corr_types = ["EE", "II", "EI"]
        conditions = ["total", "noise"]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        for row, condition in enumerate(conditions):
            for col, corr_type in enumerate(corr_types):
                ax = axes[row, col]

                if corr_type in correlations and condition in correlations[corr_type]:
                    matrix = correlations[corr_type][condition]

                    # Limit size for visualization
                    if matrix.shape[0] > max_size:
                        matrix = matrix[:max_size, :max_size]

                    # Plot heatmap
                    im = ax.imshow(
                        matrix,
                        cmap="coolwarm",
                        aspect="auto",
                        vmin=-1,
                        vmax=1,
                        interpolation="nearest",
                    )
                    ax.set_title(f"{corr_type} - {condition}")
                    ax.set_xlabel("Neuron Index")
                    ax.set_ylabel("Neuron Index")

                    # Add colorbar
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "Not Available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=12,
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])

        plt.suptitle("Correlation Matrices", fontsize=16, fontweight="bold")
        plt.tight_layout()
        save_plot(fig, save_path, "correlation_matrices_all")

    except Exception as e:
        handle_plot_error("correlation matrices", e)


def plot_tuning_curves(tuning: dict[str, Any], save_path: str) -> None:
    """Plot tuning curves and selectivity metrics for multi-class scenarios.

    Args:
        tuning: Dictionary with tuning data (can be binary or multi-class)
        save_path: Path to save the plot
    """
    try:
        n_classes = tuning.get("n_classes", 2)

        if n_classes == 2 and "E_tuning" in tuning:
            # Binary classification - use traditional tuning curve plots
            _plot_binary_tuning_curves(tuning, save_path)
        else:
            # Multi-class - use selectivity analysis plots
            _plot_multiclass_tuning_curves(tuning, save_path)

    except Exception as e:
        handle_plot_error("tuning curves", e)


def _plot_binary_tuning_curves(tuning: dict[str, Any], save_path: str) -> None:
    """Plot tuning curves for binary classification."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot histograms
    if "E_tuning" in tuning:
        axes[0, 0].hist(tuning["E_tuning"], bins=50, alpha=0.7, color="red")
        axes[0, 0].set_xlabel("Tuning Strength (Class 1 - Class 0)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Excitatory Tuning Distribution")
        axes[0, 0].axvline(0, color="k", linestyle="--", alpha=0.5)

    if "I_tuning" in tuning:
        axes[0, 1].hist(tuning["I_tuning"], bins=50, alpha=0.7, color="blue")
        axes[0, 1].set_xlabel("Tuning Strength (Class 1 - Class 0)")
        axes[0, 1].set_ylabel("Count")
        axes[0, 1].set_title("Inhibitory Tuning Distribution")
        axes[0, 1].axvline(0, color="k", linestyle="--", alpha=0.5)

    # Plot E vs I tuning scatter
    if "E_tuning" in tuning and "I_tuning" in tuning:
        # Limit points for visualization
        n_points = min(1000, len(tuning["E_tuning"]))
        indices = np.random.choice(len(tuning["E_tuning"]), n_points, replace=False)

        axes[1, 0].scatter(
            tuning["E_tuning"][indices], tuning["I_tuning"][indices], alpha=0.5, s=20
        )
        axes[1, 0].set_xlabel("Excitatory Tuning")
        axes[1, 0].set_ylabel("Inhibitory Tuning")
        axes[1, 0].set_title("E vs I Tuning Correlation")
        axes[1, 0].axhline(0, color="k", linestyle="--", alpha=0.3)
        axes[1, 0].axvline(0, color="k", linestyle="--", alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(tuning["E_tuning"], tuning["I_tuning"])[0, 1]
        axes[1, 0].text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=axes[1, 0].transAxes,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )

    # Plot Vout tuning if available
    if "Vout_tuning" in tuning:
        axes[1, 1].hist(tuning["Vout_tuning"], bins=50, alpha=0.7, color="green")
        axes[1, 1].set_xlabel("Tuning Strength (Class 1 - Class 0)")
        axes[1, 1].set_ylabel("Count")
        axes[1, 1].set_title("Output (Vout) Tuning Distribution")
        axes[1, 1].axvline(0, color="k", linestyle="--", alpha=0.5)

    plt.suptitle("Binary Classification Tuning Curves", fontsize=16, fontweight="bold")
    plt.tight_layout()
    save_plot(fig, save_path, "binary_tuning_curves")


def _plot_multiclass_tuning_curves(tuning: dict[str, Any], save_path: str) -> None:
    """Plot selectivity metrics for multi-class scenarios."""
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    n_classes = tuning.get("n_classes", 0)
    class_labels = tuning.get("class_labels", list(range(n_classes)))

    # Plot 1: Class preference distribution
    ax1 = fig.add_subplot(gs[0, :])
    _plot_class_preference_distribution(tuning, ax1, class_labels)

    # Plot 2: Selectivity index distributions
    ax2 = fig.add_subplot(gs[1, 0])
    _plot_selectivity_distributions(tuning, ax2, "selectivity_index")

    # Plot 3: Sparseness distributions
    ax3 = fig.add_subplot(gs[1, 1])
    _plot_selectivity_distributions(tuning, ax3, "sparseness")

    # Plot 4: Mean responses per class (heatmap)
    ax4 = fig.add_subplot(gs[1, 2])
    _plot_class_response_heatmap(tuning, ax4, "E")

    # Plot 5: E vs I selectivity comparison
    ax5 = fig.add_subplot(gs[2, 0])
    _plot_ei_selectivity_comparison(tuning, ax5)

    # Plot 6: Summary statistics table
    ax6 = fig.add_subplot(gs[2, 1:])
    _plot_tuning_summary_table(tuning, ax6)

    plt.suptitle(
        f"Multi-class Tuning Analysis ({n_classes} classes)",
        fontsize=16,
        fontweight="bold",
    )
    save_plot(fig, save_path, "multiclass_tuning_curves")


def _plot_class_preference_distribution(
    tuning: dict[str, Any], ax: plt.Axes, class_labels: list
) -> None:
    """Plot distribution of preferred classes across units."""
    n_classes = len(class_labels)

    # Count preferred classes for each type
    for var_type, color in [("E", "red"), ("I", "blue"), ("Vout", "green")]:
        if f"{var_type}_selectivity" in tuning:
            pref_classes = tuning[f"{var_type}_selectivity"]["preferred_class"]
            counts = np.bincount(pref_classes, minlength=n_classes)

            x = np.arange(n_classes) + {"E": -0.25, "I": 0, "Vout": 0.25}[var_type]
            ax.bar(x, counts, width=0.25, label=var_type, color=color, alpha=0.7)

    ax.set_xticks(range(n_classes))
    ax.set_xticklabels([f"Class {i}" for i in class_labels])
    ax.set_xlabel("Preferred Class")
    ax.set_ylabel("Number of Units")
    ax.set_title("Distribution of Preferred Classes")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


def _plot_selectivity_distributions(
    tuning: dict[str, Any], ax: plt.Axes, metric: str
) -> None:
    """Plot distributions of selectivity metrics."""
    colors = {"E": "red", "I": "blue", "Vout": "green"}

    for var_type in ["E", "I", "Vout"]:
        if (
            f"{var_type}_selectivity" in tuning
            and metric in tuning[f"{var_type}_selectivity"]
        ):
            values = tuning[f"{var_type}_selectivity"][metric]
            ax.hist(values, bins=30, alpha=0.5, label=var_type, color=colors[var_type])

    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Count")
    ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")


def _plot_class_response_heatmap(
    tuning: dict[str, Any], ax: plt.Axes, var_type: str
) -> None:
    """Plot heatmap of mean responses per class."""
    if f"{var_type}_class_means" in tuning:
        class_means = tuning[f"{var_type}_class_means"]

        # Normalize for visualization
        class_means_norm = class_means / (
            np.max(class_means, axis=0, keepdims=True) + 1e-8
        )

        # Limit number of units shown
        max_units = 50
        if class_means_norm.shape[1] > max_units:
            # Select most selective units
            if f"{var_type}_selectivity" in tuning:
                selectivity = tuning[f"{var_type}_selectivity"]["selectivity_index"]
                top_units = np.argsort(selectivity)[-max_units:]
                class_means_norm = class_means_norm[:, top_units]

        im = ax.imshow(
            class_means_norm, aspect="auto", cmap="hot", interpolation="nearest"
        )
        ax.set_xlabel("Unit Index")
        ax.set_ylabel("Class")
        ax.set_title(f"{var_type} Class Response Patterns")
        plt.colorbar(im, ax=ax, label="Normalized Response")


def _plot_ei_selectivity_comparison(tuning: dict[str, Any], ax: plt.Axes) -> None:
    """Plot E vs I selectivity comparison."""
    if "E_selectivity" in tuning and "I_selectivity" in tuning:
        E_sel = tuning["E_selectivity"]["selectivity_index"]
        I_sel = tuning["I_selectivity"]["selectivity_index"]

        # Sample for visualization
        n_points = min(1000, len(E_sel))
        indices = np.random.choice(len(E_sel), n_points, replace=False)

        ax.scatter(E_sel[indices], I_sel[indices], alpha=0.5, s=20)
        ax.set_xlabel("E Selectivity Index")
        ax.set_ylabel("I Selectivity Index")
        ax.set_title("E vs I Selectivity")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)

        # Add correlation
        corr = np.corrcoef(E_sel, I_sel)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
        )


def _plot_tuning_summary_table(tuning: dict[str, Any], ax: plt.Axes) -> None:
    """Plot summary statistics table."""
    summary_data = []

    for var_type in ["E", "I", "Vout"]:
        if f"{var_type}_selectivity" in tuning:
            sel_data = tuning[f"{var_type}_selectivity"]
            summary_data.append(
                [
                    var_type,
                    f"{sel_data['mean_selectivity']:.3f}",
                    f"{sel_data['mean_sparseness']:.3f}",
                    f"{np.mean(sel_data['class_selectivity']):.3f}",
                ]
            )

    # Create table
    table = ax.table(
        cellText=summary_data,
        colLabels=["Type", "Mean Selectivity", "Mean Sparseness", "Mean Class Sel."],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    ax.axis("off")
    ax.set_title("Tuning Summary Statistics", pad=20)


def plot_layer_correlation_matrix(
    correlations: dict[str, dict[str, np.ndarray]], ax: plt.Axes, title: str
) -> None:
    """Plot a single layer's correlation matrix."""
    # For visualization, we'll show the E-I correlation matrix
    if "EI" in correlations and "total" in correlations["EI"]:
        matrix = correlations["EI"]["total"]

        # Limit size for visualization
        max_size = 30
        if matrix.shape[0] > max_size or matrix.shape[1] > max_size:
            matrix = matrix[:max_size, :max_size]

        im = ax.imshow(
            matrix,
            cmap="coolwarm",
            aspect="auto",
            vmin=-0.5,
            vmax=0.5,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xlabel("Inhibitory")
        ax.set_ylabel("Excitatory")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(
            0.5,
            0.5,
            "No E-I correlations",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])


def plot_layer_summary_stats(stats: dict[str, float], ax: plt.Axes) -> None:
    """Plot summary statistics for a single layer."""
    # Select key statistics to display
    key_stats = [
        "EE_total_mean",
        "II_total_mean",
        "EI_total_mean",
        "E_Vout_total_mean",
        "I_Vout_total_mean",
    ]

    values = []
    labels = []

    for stat in key_stats:
        if stat in stats:
            values.append(stats[stat])
            labels.append(stat.replace("_total_mean", "").replace("_", "-"))

    if values:
        _ = ax.bar(range(len(values)), values, alpha=0.7)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_ylabel("Mean |Correlation|")
        ax.set_title("Correlation Summary")
        ax.grid(True, alpha=0.3, axis="y")


def plot_layer_tuning(tuning: dict[str, np.ndarray], ax: plt.Axes) -> None:
    """Plot tuning summary for a single layer."""
    if "E_tuning" in tuning and "I_tuning" in tuning:
        mean_e = np.mean(np.abs(tuning["E_tuning"]))
        mean_i = np.mean(np.abs(tuning["I_tuning"]))

        bars = ax.bar([0, 1], [mean_e, mean_i], color=["red", "blue"], alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["E", "I"])
        ax.set_ylabel("Mean |Tuning|")
        ax.set_title("Tuning Strength")
        ax.grid(True, alpha=0.3, axis="y")

        # Add values on bars
        for bar, val in zip(bars, [mean_e, mean_i]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
            )


def plot_per_ei_correlations(
    ei_results: dict[str, dict[str, Any]], save_path: str
) -> None:
    """Plot per-EI network correlation results.

    Args:
        ei_results: Dictionary with E/I specific results
        save_path: Path to save the plot
    """
    try:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Plot excitatory network results
        if "excitatory_results" in ei_results:
            exc_data = ei_results["excitatory_results"]
            if "summary_stats" in exc_data:
                plot_ei_network_stats(
                    exc_data["summary_stats"], axes[0, 0], "Excitatory Network"
                )

        # Plot inhibitory network results
        if "inhibitory_results" in ei_results:
            inh_data = ei_results["inhibitory_results"]
            if "summary_stats" in inh_data:
                plot_ei_network_stats(
                    inh_data["summary_stats"], axes[0, 1], "Inhibitory Network"
                )

        # Plot combined results
        if "combined_results" in ei_results:
            comb_data = ei_results["combined_results"]
            if "correlations" in comb_data and "EI" in comb_data["correlations"]:
                plot_ei_cross_correlation(comb_data["correlations"]["EI"], axes[0, 2])

        # Plot comparison across networks
        plot_ei_comparison(ei_results, axes[1, :])

        plt.suptitle(
            "Per-EI Network Correlation Analysis", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        save_plot(fig, save_path, "per_ei_correlation_analysis")

    except Exception as e:
        handle_plot_error("per-EI correlations", e)


def plot_ei_network_stats(stats: dict[str, float], ax: plt.Axes, title: str) -> None:
    """Plot statistics for a single E or I network."""
    # Extract relevant stats
    if title == "Excitatory Network":
        stat_keys = ["EE_total_mean", "EE_noise_mean", "E_Vout_total_mean"]
        labels = ["E-E Total", "E-E Noise", "E-Vout"]
        color = "red"
    else:
        stat_keys = ["II_total_mean", "II_noise_mean", "I_Vout_total_mean"]
        labels = ["I-I Total", "I-I Noise", "I-Vout"]
        color = "blue"

    values = [stats.get(key, 0) for key in stat_keys]

    _ = ax.bar(range(len(values)), values, color=color, alpha=0.7)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel("Mean |Correlation|")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")


def plot_ei_cross_correlation(ei_corr: dict[str, np.ndarray], ax: plt.Axes) -> None:
    """Plot E-I cross-correlation summary."""
    conditions = ["total", "noise", "signal"]
    values = []

    for cond in conditions:
        if cond in ei_corr:
            values.append(np.mean(np.abs(ei_corr[cond])))
        else:
            values.append(0)

    _ = ax.bar(range(len(conditions)), values, color="purple", alpha=0.7)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Mean |E-I Correlation|")
    ax.set_title("E-I Cross-Correlation")
    ax.grid(True, alpha=0.3, axis="y")


def plot_ei_comparison(ei_results: dict[str, Any], axes: np.ndarray) -> None:
    """Plot comparison across E and I networks."""
    # Aggregate comparison plot
    ax = axes[0]

    exc_stats = ei_results.get("excitatory_results", {}).get("summary_stats", {})
    inh_stats = ei_results.get("inhibitory_results", {}).get("summary_stats", {})

    # Compare self-correlations
    exc_self = exc_stats.get("EE_total_mean", 0)
    inh_self = inh_stats.get("II_total_mean", 0)

    _ = ax.bar([0, 1], [exc_self, inh_self], color=["red", "blue"], alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["E-E", "I-I"])
    ax.set_ylabel("Mean Self-Correlation")
    ax.set_title("E vs I Self-Correlation Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Hide unused subplots
    for ax in axes[1:]:
        ax.axis("off")


def plot_synaptic_correlations(
    synaptic_results: dict[str, Any], save_path: str
) -> None:
    """Plot synaptic-level correlation results.

    Args:
        synaptic_results: Results from synaptic correlation analysis
        save_path: Path to save the plot
    """
    try:
        # Count number of layers analyzed
        layer_names = [
            k
            for k in synaptic_results.keys()
            if k not in ["method", "n_samples", "n_sample_pairs", "sampling_strategy"]
        ]
        n_layers = len(layer_names)

        if n_layers == 0:
            return

        fig, axes = plt.subplots(n_layers, 3, figsize=(15, 5 * n_layers), squeeze=False)

        for idx, layer_name in enumerate(layer_names):
            layer_data = synaptic_results[layer_name]

            # Plot 1: Within-branch correlations
            ax1 = axes[idx, 0]
            plot_synaptic_comparison(layer_data, ax1, "within_branch")
            ax1.set_title(f"{layer_name} - Within Branch")

            # Plot 2: Within-layer correlations
            ax2 = axes[idx, 1]
            plot_synaptic_comparison(layer_data, ax2, "within_layer")
            ax2.set_title(f"{layer_name} - Within Layer")

            # Plot 3: Summary statistics
            ax3 = axes[idx, 2]
            plot_synaptic_summary(layer_data, ax3)
            ax3.set_title(f"{layer_name} - Summary")

        plt.suptitle(
            "Synaptic-Level Correlation Analysis", fontsize=16, fontweight="bold"
        )
        plt.tight_layout()
        save_plot(fig, save_path, "synaptic_correlation_analysis")

    except Exception as e:
        handle_plot_error("synaptic correlations", e)


def plot_synaptic_comparison(
    layer_data: dict[str, Any], ax: plt.Axes, category: str
) -> None:
    """Plot comparison of synaptic correlations for a specific category."""
    # Extract relevant statistics
    ee_stats = layer_data.get(f"EE_{category}", {})
    ii_stats = layer_data.get(f"II_{category}", {})
    ei_stats = layer_data.get("EI_cross", {}) if category == "within_layer" else {}

    # Prepare data for plotting
    labels = []
    means = []
    stds = []
    colors = []

    if ee_stats:
        labels.extend(["E-E Total", "E-E Noise"])
        means.extend([ee_stats.get("total_mean", 0), ee_stats.get("noise_mean", 0)])
        stds.extend([ee_stats.get("std", 0), ee_stats.get("noise_std", 0)])
        colors.extend(["red", "lightcoral"])

    if ii_stats:
        labels.extend(["I-I Total", "I-I Noise"])
        means.extend([ii_stats.get("total_mean", 0), ii_stats.get("noise_mean", 0)])
        stds.extend([ii_stats.get("std", 0), ii_stats.get("noise_std", 0)])
        colors.extend(["blue", "lightblue"])

    if ei_stats:
        labels.extend(["E-I Total", "E-I Noise"])
        means.extend([ei_stats.get("total_mean", 0), ei_stats.get("noise_mean", 0)])
        stds.extend([ei_stats.get("std", 0), ei_stats.get("noise_std", 0)])
        colors.extend(["purple", "plum"])

    # Create bar plot with error bars
    x = np.arange(len(labels))
    _ = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Mean Correlation")
    ax.grid(True, alpha=0.3, axis="y")

    # Add sample size info
    for i, (label, stats_dict) in enumerate(
        [
            (label, d)
            for label, d in [("E-E", ee_stats), ("I-I", ii_stats), ("E-I", ei_stats)]
            if d
        ]
    ):
        n_pairs = stats_dict.get("n_pairs", 0)
        if n_pairs > 0:
            ax.text(
                0.02,
                0.98 - i * 0.05,
                f"{label}: {n_pairs} pairs",
                transform=ax.transAxes,
                fontsize=9,
                va="top",
            )


def plot_synaptic_summary(layer_data: dict[str, Any], ax: plt.Axes) -> None:
    """Plot summary statistics for synaptic correlations."""
    # Create a text summary
    summary_text = []

    for corr_type in [
        "EE_within_branch",
        "EE_within_layer",
        "II_within_branch",
        "II_within_layer",
        "EI_cross",
    ]:
        if corr_type in layer_data:
            stats = layer_data[corr_type]
            summary_text.append(f"{corr_type}:")
            summary_text.append(
                f"  Total: {stats.get('total_mean', 0):.3f} ± {stats.get('std', 0):.3f}"
            )
            summary_text.append(
                f"  Noise: {stats.get('noise_mean', 0):.3f} ± {stats.get('noise_std', 0):.3f}"
            )
            if "signal_mean" in stats:
                summary_text.append(f"  Signal: {stats.get('signal_mean', 0):.3f}")
            summary_text.append("")

    ax.text(
        0.05,
        0.95,
        "\n".join(summary_text),
        transform=ax.transAxes,
        fontsize=10,
        va="top",
        fontfamily="monospace",
    )
    ax.axis("off")

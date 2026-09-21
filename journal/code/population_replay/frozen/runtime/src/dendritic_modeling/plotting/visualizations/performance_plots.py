"""
Performance and training visualization plotting.

This module provides plotting for training curves, performance metrics,
and evaluation results.
"""

import json
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

from dendritic_modeling.utils.epoch_files import epoch_file_paths_by_number

from .base import TimeSeriesPlotter


class PerformancePlotter(TimeSeriesPlotter):
    """Plotter for performance metrics and training curves."""

    def plot_loss_curves(
        self,
        train_losses: list[float],
        valid_losses: list[float],
        loss_name: str,
        save_dir: Optional[str] = None,
        filename_prefix: str = "",
    ) -> tuple[plt.Figure, plt.Axes]:
        """Plot training and validation loss curves."""
        if not self._validate_data(train_losses, "training losses"):
            return self._create_error_plot("Loss Curves", "No training data")

        fig, ax = self._create_figure()

        epochs = range(1, len(train_losses) + 1)

        # Plot raw data
        ax.plot(epochs, train_losses, "b-", alpha=0.6, label="Train (raw)")
        if valid_losses:
            ax.plot(
                epochs[: len(valid_losses)],
                valid_losses,
                "r-",
                alpha=0.6,
                label="Valid (raw)",
            )

        # Add smoothed curves
        if len(train_losses) > 10:
            smooth_train = self._smooth_data(np.array(train_losses), window_size=5)
            ax.plot(
                epochs[2:-2], smooth_train, "b-", linewidth=2, label="Train (smooth)"
            )

            if valid_losses and len(valid_losses) > 10:
                smooth_valid = self._smooth_data(np.array(valid_losses), window_size=5)
                ax.plot(
                    epochs[2 : len(smooth_valid) + 2],
                    smooth_valid,
                    "r-",
                    linewidth=2,
                    label="Valid (smooth)",
                )

        # Apply styling
        self._apply_styling(
            ax,
            title=(
                f"{filename_prefix} {loss_name} Loss"
                if filename_prefix
                else f"{loss_name} Loss"
            ),
            xlabel="Epoch",
            ylabel="Loss",
            show_legend=True,
        )

        # Add final values annotation
        final_train = train_losses[-1]
        ax.annotate(
            f"Final: {final_train:.4f}",
            xy=(len(train_losses), final_train),
            xytext=(10, 10),
            textcoords="offset points",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "yellow", "alpha": 0.7},
        )

        if save_dir:
            filename = (
                f"{filename_prefix}_loss_curves" if filename_prefix else "loss_curves"
            )
            self._save_plot(fig, save_dir, filename)

        return fig, ax

    def plot_performance_metrics(
        self,
        results_dict: dict[str, Any],
        save_path: Optional[str] = None,
        filename: str = "performance_summary",
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        """Generate comprehensive performance summary plots."""
        plots = {}

        # Extract metrics
        metrics = {}
        for key, value in results_dict.items():
            if isinstance(value, (int, float)) and not key.startswith("_"):
                metrics[key] = value

        if not metrics:
            fig, ax = self._create_error_plot("Performance Metrics", "No metrics found")
            plots["error"] = (fig, ax)
            return plots

        # Create bar plot of metrics
        fig, ax = self._create_figure(figsize=(12, 6))

        keys = list(metrics.keys())
        values = list(metrics.values())

        # Color based on metric type
        colors = []
        for key in keys:
            if "accuracy" in key.lower():
                colors.append(self.config.colors["success"])
            elif "loss" in key.lower():
                colors.append(self.config.colors["danger"])
            elif "time" in key.lower():
                colors.append(self.config.colors["info"])
            else:
                colors.append(self.config.colors["primary"])

        bars = ax.bar(keys, values, color=colors, alpha=self.config.alpha["main"])
        self._add_value_labels(ax, bars, values, format_str="{:.4f}")

        self._apply_styling(
            ax, title="Performance Metrics Summary", ylabel="Value", rotate_xticks=True
        )

        plots["summary"] = (fig, ax)

        if save_path:
            self._save_plot(fig, save_path, filename)

        return plots

    def plot_performance_evolution(
        self,
        performance_dir: str,
        save_path: Optional[str] = None,
        filename: str = "performance_evolution",
        pruning_performance: Optional[dict[str, dict[str, float]]] = None,
        final_epoch: Optional[int] = None,
    ) -> dict[str, tuple[plt.Figure, plt.Axes]]:
        """Plot performance metrics evolution over epochs from saved files."""
        plots = {}

        perf_files = list(
            epoch_file_paths_by_number(performance_dir, require_json=False)
        )

        if not perf_files:
            fig, ax = self._create_error_plot(
                "Performance Evolution",
                f"No performance files found in {performance_dir}",
            )
            plots["error"] = (fig, ax)
            return plots

        # Sort by epoch
        perf_files.sort(key=lambda x: x[0])

        # Load data
        epochs = []
        metrics_data = {}

        for epoch, filepath in perf_files:
            try:
                # Check if file exists and is readable
                if not os.path.exists(filepath):
                    self.logger.warning(f"Performance file does not exist: {filepath}")
                    continue

                with open(filepath) as f:
                    data = json.load(f)
                    if not data:
                        continue

                    epochs.append(epoch)

                    for metric_name, metric_data in data.items():
                        if metric_name.startswith("_"):
                            continue

                        if isinstance(metric_data, dict):
                            # For each dataset (train, valid, test)
                            for dataset, value in metric_data.items():
                                if isinstance(value, (int, float)) and not (
                                    np.isnan(value) or np.isinf(value)
                                ):
                                    key = f"{metric_name}_{dataset}"
                                    if key not in metrics_data:
                                        metrics_data[key] = []
                                    metrics_data[key].append(value)
                        elif isinstance(metric_data, (int, float)) and not (
                            np.isnan(metric_data) or np.isinf(metric_data)
                        ):
                            # Direct value
                            if metric_name not in metrics_data:
                                metrics_data[metric_name] = []
                            metrics_data[metric_name].append(metric_data)

            except (json.JSONDecodeError, FileNotFoundError, PermissionError) as e:
                self.logger.warning(f"Error loading {filepath}: {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error loading {filepath}: {e}")

        if not metrics_data:
            fig, ax = self._create_error_plot(
                "Performance Evolution", "No valid metrics found in files"
            )
            plots["error"] = (fig, ax)
            return plots

        metric_groups = {}
        for key in metrics_data.keys():
            base_metric = key.split("_")[0]
            if base_metric not in metric_groups:
                metric_groups[base_metric] = []
            metric_groups[base_metric].append(key)

        # Create separate plots for each metric type
        for metric_type, metric_keys in metric_groups.items():
            fig, ax = self._create_figure(figsize=(12, 8))

            colors = ["blue", "red", "green", "orange", "purple"]
            for idx, key in enumerate(metric_keys):
                values = metrics_data[key]
                color = colors[idx % len(colors)]

                # Plot with markers and lines
                ax.plot(
                    epochs[: len(values)],
                    values,
                    "o-",
                    color=color,
                    alpha=0.7,
                    label=key.replace("_", " ").title(),
                    linewidth=2,
                    markersize=4,
                )

                # Add pruning performance markers if available
                if pruning_performance and final_epoch is not None:
                    self._add_pruning_markers(
                        ax, metric_type, key, pruning_performance, final_epoch
                    )

            self._apply_styling(
                ax,
                title=f'{metric_type.replace("_", " ").title()} Evolution Over Training',
                xlabel="Epoch",
                ylabel=metric_type.replace("_", " ").title(),
                show_legend=True,
            )

            plots[metric_type] = (fig, ax)

            # Save individual plots
            if save_path:
                individual_filename = f"{filename}_{metric_type}"
                self._save_plot(fig, save_path, individual_filename)

        return plots

    def _add_pruning_markers(
        self,
        ax: plt.Axes,
        metric_type: str,
        key: str,
        pruning_performance: dict[str, dict[str, float]],
        final_epoch: int,
    ):
        """Add pruning performance markers to the plot."""
        # Check if this key matches any pruning performance data
        for stage in ["pre_pruning_performance", "post_pruning_performance"]:
            if stage in pruning_performance:
                stage_data = pruning_performance[stage]
                # Find matching metric
                for perf_key, value in stage_data.items():
                    # Match metric type (e.g., 'accuracy' in key and 'accuracy' in perf_key)
                    if (
                        metric_type.lower() in perf_key.lower()
                        or perf_key.lower() in metric_type.lower()
                    ):
                        marker_epoch = (
                            final_epoch
                            if stage == "pre_pruning_performance"
                            else final_epoch + 1
                        )
                        marker_style = (
                            "s" if stage == "pre_pruning_performance" else "o"
                        )
                        marker_color = (
                            "red" if stage == "pre_pruning_performance" else "darkred"
                        )
                        marker_size = 12
                        marker_label = f"{'Pre' if stage == 'pre_pruning_performance' else 'Post'} Pruning"

                        ax.scatter(
                            [marker_epoch],
                            [value],
                            marker=marker_style,
                            s=marker_size**2,
                            color=marker_color,
                            alpha=0.9,
                            edgecolor="black",
                            linewidth=2,
                            label=marker_label,
                            zorder=10,
                        )


class SweepResultsPlotter(TimeSeriesPlotter):
    """Plotter for hyperparameter sweep results."""

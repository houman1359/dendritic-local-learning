"""Noise perturbation sweep analyzer."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..plotters import PerformancePlotter
from .base_analyzer import BaseSweepAnalyzer

logger = logging.getLogger(__name__)


class NoiseSweepAnalyzer(BaseSweepAnalyzer):
    """Analyzer for noise perturbation sweeps."""

    def __init__(self):
        """Initialize noise sweep analyzer."""
        super().__init__()
        self.performance_plotter = PerformancePlotter()

    def can_handle(self, results_dir: Path) -> bool:
        """
        Detect if this is a noise perturbation sweep.

        Args:
            results_dir: Path to results directory

        Returns:
            True if this analyzer can handle the directory
        """
        from ..collectors import DataCollector

        # Quick check: look for noise-related columns in first few results
        collector = DataCollector(results_dir)
        config_dirs = collector._get_config_dirs()

        if not config_dirs:
            return False

        # Check first config for noise data
        first_config = config_dirs[0]
        noise_file = first_config / "noise_perturbation_analysis" / "final"

        return noise_file.exists() or noise_file.with_suffix(".json").exists()

    def get_groupby_columns(self) -> list[str]:
        """Get columns to group by for noise sweeps."""
        return [
            "ee_value",
            "ie_value",
            "use_shunting",
            "network_category",
            "noise_type",
            "noise_level",
        ]

    def get_plot_types(self) -> list[str]:
        """Get plot types specific to noise sweeps."""
        return [
            "noise_robustness_curves",
            "noise_degradation_heatmaps",
            "noise_type_comparison",
        ]

    def _generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all noise-specific plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        noise_dir = output_dir / "plots" / "noise_analysis"
        noise_dir.mkdir(parents=True, exist_ok=True)

        # Check for noise data
        noise_cols = [col for col in data.columns if "noise_" in col]
        if not noise_cols:
            logger.info("No noise perturbation data found")
            return plot_paths

        # Generate noise robustness curves
        logger.info("Generating noise robustness plots...")
        plot_paths.extend(self._plot_noise_robustness(data, noise_dir))

        # Generate noise type comparison
        plot_paths.extend(self._plot_noise_type_comparison(data, noise_dir))

        # Also generate standard performance plots
        plot_paths.extend(self.performance_plotter.generate_all(data, output_dir))

        return plot_paths

    def _plot_noise_robustness(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Plot performance degradation under noise."""
        plot_paths = []

        # Find noise columns with pattern: noise_{type}_{metric}_{level}
        # Extract unique noise types and levels
        noise_metrics = {}
        for col in data.columns:
            if col.startswith("noise_") and "_mean" in col:
                parts = col.split("_")
                if len(parts) >= 3:
                    noise_type = parts[1]  # uniform or gaussian
                    metric = parts[2]  # accuracy, auc, etc.

                    if metric not in noise_metrics:
                        noise_metrics[metric] = set()
                    noise_metrics[metric].add(noise_type)

        if not noise_metrics:
            return plot_paths

        # Plot robustness curve for each metric
        for metric, noise_types in noise_metrics.items():
            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                for noise_type in noise_types:
                    # Find all noise levels for this type
                    pattern = f"noise_{noise_type}_{metric}_"
                    level_cols = [
                        col
                        for col in data.columns
                        if col.startswith(pattern) and col.endswith("_mean")
                    ]

                    if not level_cols:
                        continue

                    # Extract levels and values
                    levels = []
                    values = []
                    for col in level_cols:
                        try:
                            level_str = col.replace(pattern, "").replace("_mean", "")
                            level = float(level_str)
                            levels.append(level)
                            values.append(data[col].mean())
                        except ValueError:
                            continue

                    if levels:
                        # Sort by level
                        sorted_pairs = sorted(zip(levels, values))
                        levels, values = zip(*sorted_pairs)

                        ax.plot(
                            levels,
                            values,
                            marker="o",
                            label=f"{noise_type.title()} Noise",
                            linewidth=2,
                            markersize=6,
                        )

                self.performance_plotter.setup_axes_style(
                    ax=ax,
                    xlabel="Noise Level",
                    ylabel=metric.title(),
                    title=f"{metric.title()} Robustness to Noise",
                    grid=True,
                )
                ax.legend(fontsize=10)

                fig.tight_layout()
                plot_path = output_dir / f"{metric}_noise_robustness.png"
                plot_paths.append(self.performance_plotter.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating noise plot for {metric}: {e}")

        logger.info(f"Generated {len(plot_paths)} noise robustness plots")
        return plot_paths

    def _plot_noise_type_comparison(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Compare different noise types (uniform vs gaussian)."""
        plot_paths = []

        try:
            # Create comparison plot for test accuracy
            noise_types = ["uniform", "gaussian"]

            # Check if we have both types
            has_uniform = any("noise_uniform" in col for col in data.columns)
            has_gaussian = any("noise_gaussian" in col for col in data.columns)

            if not (has_uniform and has_gaussian):
                logger.info("Not all noise types present for comparison")
                return plot_paths

            fig, axes = plt.subplots(1, 2, figsize=(16, 6))

            for idx, metric in enumerate(["accuracy", "auc"]):
                if idx >= len(axes):
                    break

                ax = axes[idx]

                for noise_type in noise_types:
                    # Find columns for this noise type and metric
                    pattern = f"noise_{noise_type}_{metric}_"
                    level_cols = [
                        col
                        for col in data.columns
                        if col.startswith(pattern) and col.endswith("_mean")
                    ]

                    if not level_cols:
                        continue

                    # Extract and plot
                    levels = []
                    values = []
                    for col in level_cols:
                        try:
                            level_str = col.replace(pattern, "").replace("_mean", "")
                            level = float(level_str)
                            levels.append(level)
                            values.append(data[col].mean())
                        except ValueError:
                            continue

                    if levels:
                        sorted_pairs = sorted(zip(levels, values))
                        levels, values = zip(*sorted_pairs)

                        ax.plot(
                            levels,
                            values,
                            marker="o",
                            label=noise_type.title(),
                            linewidth=2,
                            markersize=6,
                        )

                self.performance_plotter.setup_axes_style(
                    ax=ax,
                    xlabel="Noise Level",
                    ylabel=metric.title(),
                    title=f"{metric.title()} Under Different Noise Types",
                    grid=True,
                )
                ax.legend(fontsize=10)

            fig.suptitle("Noise Type Comparison", fontsize=14)
            fig.tight_layout()

            plot_path = output_dir / "noise_type_comparison.png"
            plot_paths.append(self.performance_plotter.save_figure(fig, plot_path))

        except Exception as e:
            logger.error(f"Error generating noise type comparison: {e}")

        return plot_paths

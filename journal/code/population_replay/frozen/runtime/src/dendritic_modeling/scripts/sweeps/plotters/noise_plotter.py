"""Noise perturbation plotters."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)


class NoisePlotter(BasePlotter):
    """Generates noise perturbation plots."""

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all noise perturbation plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        noise_dir = output_dir / "noise"
        noise_dir.mkdir(parents=True, exist_ok=True)

        # Check for noise data
        noise_cols = [col for col in data.columns if "noise_" in col]
        if not noise_cols:
            logger.info("No noise data found, skipping noise plots")
            return plot_paths

        # Generate noise robustness curves
        plot_paths.extend(self._plot_noise_robustness(data, noise_dir))

        return plot_paths

    def _plot_noise_robustness(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Plot performance degradation under noise."""
        plot_paths = []

        # Find noise columns
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
                fig, ax = plt.subplots(figsize=self.default_figsize)

                for noise_type in noise_types:
                    pattern = f"noise_{noise_type}_{metric}_"
                    level_cols = [
                        col
                        for col in data.columns
                        if col.startswith(pattern) and col.endswith("_mean")
                    ]

                    if not level_cols:
                        continue

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
                            label=f"{noise_type.title()} Noise",
                            linewidth=2,
                            markersize=6,
                        )

                self.setup_axes_style(
                    ax=ax,
                    xlabel="Noise Level",
                    ylabel=metric.title(),
                    title=f"{metric.title()} Robustness to Noise",
                    grid=True,
                )
                ax.legend(fontsize=10)

                fig.tight_layout()
                plot_path = output_dir / f"{metric}_noise_robustness.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating noise plot for {metric}: {e}")

        logger.info(f"Generated {len(plot_paths)} noise robustness plots")
        return plot_paths

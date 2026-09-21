"""General parameter sweep analyzer."""

import logging
from pathlib import Path

import pandas as pd

from ..plotters import (
    AblationPlotter,
    InformationPlotter,
    NoisePlotter,
    PerformancePlotter,
    WeightPlotter,
)
from .base_analyzer import BaseSweepAnalyzer

logger = logging.getLogger(__name__)


class GeneralSweepAnalyzer(BaseSweepAnalyzer):
    """Analyzer for general parameter sweeps (catchall)."""

    def __init__(self):
        """Initialize general sweep analyzer."""
        super().__init__()
        self.performance_plotter = PerformancePlotter()
        self.information_plotter = InformationPlotter()
        self.weight_plotter = WeightPlotter()
        self.ablation_plotter = AblationPlotter()
        self.noise_plotter = NoisePlotter()

    def can_handle(self, results_dir: Path) -> bool:
        """
        General analyzer handles anything (catchall).

        Args:
            results_dir: Path to results directory

        Returns:
            Always True (this is the fallback analyzer)
        """
        # Check that the directory structure exists
        configs_dir = results_dir / "configs"
        results_subdir = results_dir / "results"

        return configs_dir.exists() and results_subdir.exists()

    def get_groupby_columns(self) -> list[str]:
        """Get columns to group by for general sweeps."""
        # Use a broad set of potential grouping columns
        return [
            "ee_value",
            "ie_value",
            "ei_ratio",
            "use_shunting",
            "network_category",
            "strategy",
            "network_layer_count",
            "blocklinear_strategy",
            "topk_strategy",
            "reactivation_strategy",
            "reactivation_init_policy",
            "optimizer_name",
            "optimizer_weight_decay",
            "sparse_weight_decay_rate",
            "sparse_weight_boosting",
            "dataset",
        ]

    def get_plot_types(self) -> list[str]:
        """Get plot types for general sweeps."""
        return [
            "performance",
            "information",
            "weights",
            "parameter_scaling",
        ]

    def _generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all general plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Generate all standard plots using available plotters
        logger.info("Generating performance plots...")
        plot_paths.extend(self.performance_plotter.generate_all(data, output_dir))

        logger.info("Generating information plots...")
        plot_paths.extend(self.information_plotter.generate_all(data, output_dir))

        logger.info("Generating weight plots...")
        plot_paths.extend(self.weight_plotter.generate_all(data, output_dir))

        # Generate ablation and noise plots if data available
        logger.info("Generating ablation plots...")
        plot_paths.extend(self.ablation_plotter.generate_all(data, output_dir))

        logger.info("Generating noise plots...")
        plot_paths.extend(self.noise_plotter.generate_all(data, output_dir))

        # If no plots generated yet, create summary plots
        if len(plot_paths) == 0:
            logger.info("No sweep-variable plots generated, creating summary plots...")
            plot_paths.extend(self._generate_summary_plots(data, output_dir))

        return plot_paths

    def _generate_summary_plots(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Generate summary plots when no sweep variable is available."""
        import matplotlib.pyplot as plt
        import numpy as np

        plot_paths = []
        summary_dir = output_dir / "summary"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Find performance metrics
        perf_metrics = [
            col
            for col in data.columns
            if any(m in col for m in ["accuracy", "auc", "categorical"])
        ]

        if perf_metrics:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))

                metrics_to_plot = perf_metrics[:10]  # Limit to 10 metrics
                x_pos = np.arange(len(metrics_to_plot))
                values = [data[m].mean() for m in metrics_to_plot]

                ax.bar(x_pos, values, alpha=0.7, color="#45B7D1")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(
                    [m.replace("_", " ").title()[:30] for m in metrics_to_plot],
                    rotation=45,
                    ha="right",
                )
                ax.set_ylabel("Mean Value", fontsize=12)
                ax.set_title("Performance Metrics Summary", fontsize=14)
                ax.grid(True, alpha=0.3, axis="y")

                fig.tight_layout()
                plot_path = summary_dir / "performance_summary.png"
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                plot_paths.append(plot_path)

            except Exception as e:
                logger.error(f"Error generating performance summary: {e}")

        # Find information metrics
        mi_metrics = [
            col
            for col in data.columns
            if col.startswith("mi_") and not col.startswith("mi_layer")
        ]

        if mi_metrics:
            try:
                fig, ax = plt.subplots(figsize=(12, 6))

                metrics_to_plot = mi_metrics[:10]
                x_pos = np.arange(len(metrics_to_plot))
                values = [data[m].mean() for m in metrics_to_plot]

                ax.bar(x_pos, values, alpha=0.7, color="#FF6B6B")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(metrics_to_plot, rotation=45, ha="right")
                ax.set_ylabel("Mutual Information (bits)", fontsize=12)
                ax.set_title("Information Metrics Summary", fontsize=14)
                ax.grid(True, alpha=0.3, axis="y")

                fig.tight_layout()
                plot_path = summary_dir / "information_summary.png"
                fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close(fig)
                plot_paths.append(plot_path)

            except Exception as e:
                logger.error(f"Error generating information summary: {e}")

        logger.info(f"Generated {len(plot_paths)} summary plots")
        return plot_paths

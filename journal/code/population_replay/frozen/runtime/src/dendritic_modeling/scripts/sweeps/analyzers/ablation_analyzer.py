"""Ablation analysis sweep analyzer."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..plotters import PerformancePlotter
from .base_analyzer import BaseSweepAnalyzer

logger = logging.getLogger(__name__)


class AblationSweepAnalyzer(BaseSweepAnalyzer):
    """Analyzer for ablation analysis sweeps."""

    def __init__(self):
        """Initialize ablation sweep analyzer."""
        super().__init__()
        self.performance_plotter = PerformancePlotter()

    def can_handle(self, results_dir: Path) -> bool:
        """
        Detect if this is an ablation analysis sweep.

        Args:
            results_dir: Path to results directory

        Returns:
            True if this analyzer can handle the directory
        """
        from ..collectors import DataCollector

        # Quick check: look for ablation-related data in first few results
        collector = DataCollector(results_dir)
        config_dirs = collector._get_config_dirs()

        if not config_dirs:
            return False

        # Check first config for ablation data
        first_config = config_dirs[0]
        ablation_file = first_config / "ablation_analysis" / "final"

        return ablation_file.exists() or ablation_file.with_suffix(".json").exists()

    def get_groupby_columns(self) -> list[str]:
        """Get columns to group by for ablation sweeps."""
        return ["ee_value", "ie_value", "use_shunting", "network_category"]

    def get_plot_types(self) -> list[str]:
        """Get plot types specific to ablation sweeps."""
        return [
            "layer_contribution",
            "ablation_impact",
            "excitation_vs_inhibition_contribution",
        ]

    def _generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all ablation-specific plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        ablation_dir = output_dir / "plots" / "ablation_analysis"
        ablation_dir.mkdir(parents=True, exist_ok=True)

        # Check for ablation data
        ablation_cols = [col for col in data.columns if "ablation_" in col]
        if not ablation_cols:
            logger.info("No ablation data found")
            return plot_paths

        # Generate layer contribution plots
        logger.info("Generating layer contribution plots...")
        plot_paths.extend(self._plot_layer_contributions(data, ablation_dir))

        # Generate excitation vs inhibition comparison
        plot_paths.extend(self._plot_exc_inh_contributions(data, ablation_dir))

        # Also generate standard performance plots
        plot_paths.extend(self.performance_plotter.generate_all(data, output_dir))

        return plot_paths

    def _plot_layer_contributions(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Plot contribution of each layer when ablated."""
        plot_paths = []

        # Find layer ablation columns using the canonical parser
        from dendritic_modeling.scripts.sweeps.utils.ablation_columns import (
            discover_ablation_dimensions,
            find_ablation_columns,
        )

        parsed_cols = find_ablation_columns(list(data.columns), stat="mean")
        if not parsed_cols:
            return plot_paths

        _methods, _targets, _metrics, depths = discover_ablation_dimensions(
            list(data.columns)
        )

        if not depths:
            return plot_paths

        accuracy_cols = [p for p in parsed_cols if p.metric == "accuracy_drop"]
        if not accuracy_cols:
            return plot_paths

        ablation_types = sorted({(p.method, p.target) for p in accuracy_cols})

        # Plot accuracy drop per layer for each ablation type
        try:
            fig, ax = plt.subplots(figsize=(12, 6))

            for method, target in ablation_types:
                layer_accuracies = []

                for depth in depths:
                    matching_cols = [
                        p.full_col
                        for p in accuracy_cols
                        if p.method == method
                        and p.target == target
                        and p.depth == depth
                    ]

                    if matching_cols:
                        col = matching_cols[0]
                        acc = data[col].mean()
                        layer_accuracies.append(acc)
                    else:
                        layer_accuracies.append(None)

                if any(acc is not None for acc in layer_accuracies):
                    ax.plot(
                        depths,
                        layer_accuracies,
                        marker="o",
                        label=f"{method.replace('_', ' ').title()} {target.replace('_', ' ').title()}",
                        linewidth=2,
                        markersize=6,
                    )

            self.performance_plotter.setup_axes_style(
                ax=ax,
                xlabel="Layer Index",
                ylabel="Accuracy After Ablation",
                title="Layer Contribution Analysis (Ablation Impact)",
                grid=True,
            )
            ax.legend(fontsize=10)

            fig.tight_layout()
            plot_path = output_dir / "layer_contribution_by_ablation.png"
            plot_paths.append(self.performance_plotter.save_figure(fig, plot_path))

        except Exception as e:
            logger.error(f"Error generating layer contribution plot: {e}")

        logger.info(f"Generated {len(plot_paths)} layer contribution plots")
        return plot_paths

    def _plot_exc_inh_contributions(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Plot excitatory vs inhibitory contributions."""
        plot_paths = []

        from dendritic_modeling.scripts.sweeps.utils.ablation_columns import (
            find_ablation_columns,
        )

        parsed_cols = find_ablation_columns(list(data.columns), stat="mean")
        if not parsed_cols:
            return plot_paths

        methods = sorted(
            {p.method for p in parsed_cols if p.target in {"excitation", "inhibition"}}
        )
        metrics = sorted(
            {p.metric for p in parsed_cols if p.target in {"excitation", "inhibition"}}
        )

        for method in methods:
            for metric in metrics:
                exc_cols = {
                    p.depth: p.full_col
                    for p in parsed_cols
                    if p.method == method
                    and p.target == "excitation"
                    and p.metric == metric
                }
                inh_cols = {
                    p.depth: p.full_col
                    for p in parsed_cols
                    if p.method == method
                    and p.target == "inhibition"
                    and p.metric == metric
                }
                common_depths = sorted(set(exc_cols) & set(inh_cols))
                if not common_depths:
                    continue

                try:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    exc_values = [
                        float(data[exc_cols[d]].mean()) for d in common_depths
                    ]
                    inh_values = [
                        float(data[inh_cols[d]].mean()) for d in common_depths
                    ]
                    width = 0.35
                    x_pos = list(range(len(common_depths)))

                    ax.bar(
                        [x - width / 2 for x in x_pos],
                        exc_values,
                        width,
                        label="Excitation Ablated",
                        alpha=0.8,
                    )
                    ax.bar(
                        [x + width / 2 for x in x_pos],
                        inh_values,
                        width,
                        label="Inhibition Ablated",
                        alpha=0.8,
                    )

                    self.performance_plotter.setup_axes_style(
                        ax=ax,
                        xlabel="Layer Index",
                        ylabel=metric.replace("_", " ").title(),
                        title=(
                            "Excitatory vs Inhibitory Contribution\n"
                            f"Method: {method.replace('_', ' ').title()}"
                        ),
                        grid=True,
                    )
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(common_depths)
                    ax.legend(fontsize=10)

                    fig.tight_layout()
                    plot_path = output_dir / (
                        "excitation_vs_inhibition_" f"{method}_{metric}.png"
                    )
                    plot_paths.append(
                        self.performance_plotter.save_figure(fig, plot_path)
                    )
                except Exception as e:
                    logger.error(
                        "Error generating E/I contribution comparison for %s/%s: %s",
                        method,
                        metric,
                        e,
                    )

        return plot_paths

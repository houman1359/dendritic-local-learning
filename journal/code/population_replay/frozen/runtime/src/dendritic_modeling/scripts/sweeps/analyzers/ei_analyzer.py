"""E/I ratio sweep analyzer."""

import logging
from pathlib import Path

import pandas as pd

from ..plotters import (
    AblationPlotter,
    InformationPlotter,
    NoisePlotter,
    PerformancePlotter,
    SNRPlotter,
    WeightPlotter,
)
from .base_analyzer import BaseSweepAnalyzer

logger = logging.getLogger(__name__)


class EISweepAnalyzer(BaseSweepAnalyzer):
    """Analyzer for E/I ratio sweeps."""

    def __init__(self):
        """Initialize E/I sweep analyzer."""
        super().__init__()
        self.performance_plotter = PerformancePlotter()
        self.information_plotter = InformationPlotter()
        self.weight_plotter = WeightPlotter()
        self.ablation_plotter = AblationPlotter()
        self.snr_plotter = SNRPlotter()
        self.noise_plotter = NoisePlotter()

    def can_handle(self, results_dir: Path) -> bool:
        """
        Detect if this is an E/I sweep.

        Checks for:
        - Multiple ee_value and ie_value combinations in configs
        - Presence of E/I-related parameters in sweep configs

        Args:
            results_dir: Path to results directory

        Returns:
            True if this analyzer can handle the directory
        """
        from ..collectors import ConfigExtractor

        configs_dir = results_dir / "configs"
        if not configs_dir.exists():
            return False

        # Load a few config files and check for EE/IE parameters
        config_files = sorted(configs_dir.glob("*.yaml"))[:5]  # Check first 5

        extractor = ConfigExtractor()
        ee_values = set()
        ie_values = set()

        for config_file in config_files:
            params = extractor.extract_from_file(config_file)
            if "ee_value" in params:
                ee_values.add(params["ee_value"])
            if "ie_value" in params:
                ie_values.add(params["ie_value"])

        # This is an E/I sweep if we have multiple E or I values
        return len(ee_values) > 1 or len(ie_values) > 1

    def get_groupby_columns(self) -> list[str]:
        """Get columns to group by for E/I sweeps."""
        return [
            "ee_value",
            "ie_value",
            "ei_ratio",
            "use_shunting",
            "reactivation_init_policy",
            "network_category",
            "nparams",
        ]

    def get_plot_types(self) -> list[str]:
        """Get plot types specific to E/I sweeps."""
        return [
            "performance_vs_ei_ratio",
            "performance_heatmaps",
            "information_vs_ei_ratio",
            "information_panels",
            "weight_vs_ei_ratio",
            "ablation_analysis",
            "noise_robustness",
            "shunting_comparison",
        ]

    def _generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all E/I-specific plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Validate we have E/I data
        if "ei_ratio" not in data.columns:
            logger.warning("No ei_ratio column found - cannot generate E/I plots")
            return plot_paths

        # Generate performance plots
        logger.info("Generating performance plots...")
        plot_paths.extend(self.performance_plotter.generate_all(data, output_dir))

        # Generate information plots
        logger.info("Generating information plots...")
        plot_paths.extend(self.information_plotter.generate_all(data, output_dir))

        # Generate weight plots
        logger.info("Generating weight plots...")
        plot_paths.extend(self.weight_plotter.generate_all(data, output_dir))

        # Generate ablation plots (if data available)
        logger.info("Generating ablation plots...")
        plot_paths.extend(self.ablation_plotter.generate_all(data, output_dir))

        # Generate SNR plots (if data available)
        logger.info("Generating SNR plots...")
        plot_paths.extend(self.snr_plotter.generate_all(data, output_dir))

        # Generate noise plots (if data available)
        logger.info("Generating noise plots...")
        plot_paths.extend(self.noise_plotter.generate_all(data, output_dir))

        # Generate E/I-specific plots
        logger.info("Generating E/I-specific plots...")
        plot_paths.extend(self._generate_shunting_comparison(data, output_dir))

        return plot_paths

    def _generate_shunting_comparison(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """
        Generate shunting vs non-shunting comparison plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        if "network_category" not in data.columns:
            return plot_paths

        comparison_dir = output_dir / "shunting_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # Check we have the baseline pair we want to compare.
        categories = set(data["network_category"].unique())
        shunt_cat = "dendritic_shunting" if "dendritic_shunting" in categories else None
        add_cat = "dendritic_additive" if "dendritic_additive" in categories else None
        if shunt_cat is None or add_cat is None:
            logger.info(
                "Not all network categories present, skipping shunting comparison"
            )
            return plot_paths

        try:
            # Create comprehensive comparison figure
            fig, axes = self.performance_plotter.create_multi_panel_figure(
                nrows=2,
                ncols=3,
                figsize=(18, 12),
                title="Shunting vs Non-Shunting Comparison",
            )
            axes = axes.flatten()

            # Plot different metrics
            metrics = [
                "test_accuracy",
                "test_categorical_loglikelihood",
                "mi_E_C",
                "mi_I_C",
                "mi_E_I",
                "mi_V_C",
            ]

            for idx, metric in enumerate(metrics):
                if idx >= len(axes):
                    break

                mean_col = f"{metric}_mean"
                if mean_col not in data.columns:
                    axes[idx].text(
                        0.5,
                        0.5,
                        f"No {metric} data",
                        ha="center",
                        va="center",
                        transform=axes[idx].transAxes,
                    )
                    axes[idx].set_title(metric.replace("_", " ").title())
                    continue

                # Plot both categories
                for category in [shunt_cat, add_cat]:
                    cat_data = data[data["network_category"] == category].sort_values(
                        "ei_ratio"
                    )
                    if not cat_data.empty:
                        color = self.performance_plotter.category_colors.get(category)
                        linestyle = "-" if category == shunt_cat else "--"
                        axes[idx].plot(
                            cat_data["ei_ratio"],
                            cat_data[mean_col],
                            label=category,
                            color=color,
                            linestyle=linestyle,
                            marker="o",
                            linewidth=2,
                        )

                self.performance_plotter.setup_axes_style(
                    ax=axes[idx],
                    xlabel="E/I Ratio",
                    ylabel=metric.replace("_", " ").title(),
                    title=metric.replace("_", " ").title(),
                    use_log_x=True,
                    grid=True,
                )
                axes[idx].legend(fontsize=9)

            fig.tight_layout()
            plot_path = comparison_dir / "shunting_comparison_panels.png"
            plot_paths.append(self.performance_plotter.save_figure(fig, plot_path))

        except Exception as e:
            logger.error(f"Error generating shunting comparison: {e}")

        logger.info(f"Generated {len(plot_paths)} shunting comparison plots")
        return plot_paths

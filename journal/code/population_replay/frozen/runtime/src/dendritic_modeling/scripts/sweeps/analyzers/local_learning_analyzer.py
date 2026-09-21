"""Local learning parameter sweep analyzer."""

import logging
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import pandas as pd

from ..plotters import (
    AblationPlotter,
    InformationPlotter,
    NoisePlotter,
    PerformancePlotter,
    WeightPlotter,
)
from ..utils.metrics import get_metric_label
from .base_analyzer import BaseSweepAnalyzer

logger = logging.getLogger(__name__)


class LocalLearningSweepAnalyzer(BaseSweepAnalyzer):
    """Analyzer for local learning parameter sweeps."""

    LOCAL_LEARNING_PARAMS: ClassVar[list[str]] = [
        "rule_variant",
        "error_broadcast_mode",
        "decoder_update_mode",
        "rho_mode",
        "rho_estimator",
        "phi_mode",
        "phi_estimator",
        "use_dendritic_normalization",
        "use_path_propagation",
        "morphology_modulator_mode",
        "hsic_enabled",
        "update_inactive_weights",
    ]

    def __init__(self):
        """Initialize local learning sweep analyzer."""
        super().__init__()
        self.performance_plotter = PerformancePlotter()
        self.information_plotter = InformationPlotter()
        self.weight_plotter = WeightPlotter()
        self.ablation_plotter = AblationPlotter()
        self.noise_plotter = NoisePlotter()

    def can_handle(self, results_dir: Path) -> bool:
        """
        Detect if this is a local learning parameter sweep.

        Args:
            results_dir: Path to results directory

        Returns:
            True if this analyzer can handle the directory
        """
        from ..collectors import ConfigExtractor

        configs_dir = results_dir / "configs"
        if not configs_dir.exists():
            return False

        # Load a few configs and check for local learning parameters
        config_files = sorted(configs_dir.glob("*.yaml"))[:5]

        extractor = ConfigExtractor()
        local_param_values = {param: set() for param in self.LOCAL_LEARNING_PARAMS}

        for config_file in config_files:
            params = extractor.extract_from_file(config_file)
            for param in self.LOCAL_LEARNING_PARAMS:
                if param in params and params[param] is not None:
                    local_param_values[param].add(str(params[param]))

        # This is a local learning sweep if any local learning parameter varies
        return any(len(values) > 1 for values in local_param_values.values())

    def get_groupby_columns(self) -> list[str]:
        """Get columns to group by for local learning sweeps."""
        base_cols = ["use_shunting", "network_category"]

        # Add E/I ratio if present
        ei_cols = ["ee_value", "ie_value", "ei_ratio"]

        # Add all local learning params
        return base_cols + ei_cols + self.LOCAL_LEARNING_PARAMS

    def get_plot_types(self) -> list[str]:
        """Get plot types specific to local learning sweeps."""
        return [
            "performance_by_rule_variant",
            "information_by_rule_variant",
            "parameter_combination_comparison",
            "morphology_effect_analysis",
        ]

    def _generate_all_plots(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all local learning-specific plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Generate standard plots
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

        # Generate parameter combination plots
        logger.info("Generating parameter combination plots...")
        plot_paths.extend(self._generate_parameter_combination_plots(data, output_dir))

        # Generate rule variant comparison
        if "rule_variant" in data.columns:
            logger.info("Generating rule variant comparison...")
            plot_paths.extend(self._generate_rule_variant_comparison(data, output_dir))

        return plot_paths

    def _generate_parameter_combination_plots(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Generate plots comparing different parameter combinations."""
        plot_paths = []
        param_dir = output_dir / "parameter_combinations"
        param_dir.mkdir(parents=True, exist_ok=True)

        # Find which local learning parameters vary in this sweep
        varying_params = []
        for param in self.LOCAL_LEARNING_PARAMS:
            if param in data.columns:
                unique_vals = data[param].dropna().unique()
                if len(unique_vals) > 1:
                    varying_params.append(param)

        if not varying_params:
            logger.info("No varying local learning parameters found")
            return plot_paths

        logger.info(f"Found varying parameters: {varying_params}")

        # Create parameter combination legend
        legend_path = self._create_parameter_legend(data, param_dir, varying_params)
        if legend_path:
            plot_paths.append(legend_path)

        # Generate comparison plots for key metrics
        key_metrics = ["test_accuracy", "mi_E_C", "mi_I_C", "mi_V_C"]

        for metric in key_metrics:
            mean_col = f"{metric}_mean"
            if mean_col not in data.columns:
                continue

            try:
                plot_path = self._plot_metric_by_param_combinations(
                    data, metric, varying_params, param_dir
                )
                if plot_path:
                    plot_paths.append(plot_path)
            except Exception as e:
                logger.error(f"Error plotting {metric} by param combinations: {e}")

        return plot_paths

    def _create_parameter_legend(
        self,
        data: pd.DataFrame,
        output_dir: Path,
        varying_params: list[str],
    ) -> Path:
        """Create a legend mapping combination IDs to parameter values."""
        try:
            # Get unique combinations
            combo_df = (
                data[varying_params]
                .drop_duplicates()
                .sort_values(by=varying_params)
                .reset_index(drop=True)
            )
            combo_df.insert(0, "ID", range(1, len(combo_df) + 1))

            # Save as CSV
            csv_path = output_dir / "parameter_combinations_legend.csv"
            combo_df.to_csv(csv_path, index=False)

            # Create visualization
            fig, ax = plt.subplots(
                figsize=(min(12, 2 + len(varying_params)), min(20, 0.5 * len(combo_df)))
            )
            ax.axis("off")

            table = ax.table(
                cellText=combo_df.values,
                colLabels=combo_df.columns,
                loc="center",
                cellLoc="left",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1, 1.5)

            # Color header
            for (i, _j), cell in table.get_celld().items():
                if i == 0:
                    cell.set_facecolor("#4CAF50")
                    cell.set_text_props(weight="bold", color="white")

            fig.tight_layout()
            png_path = output_dir / "parameter_combinations_legend.png"
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            logger.info(f"Created parameter legend: {csv_path}")
            return csv_path

        except Exception as e:
            logger.error(f"Error creating parameter legend: {e}")
            return None

    def _plot_metric_by_param_combinations(
        self,
        data: pd.DataFrame,
        metric: str,
        varying_params: list[str],
        output_dir: Path,
    ) -> Path:
        """Plot a metric across all parameter combinations."""
        try:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            # Get unique combinations and assign IDs
            combo_df = (
                data[varying_params]
                .drop_duplicates()
                .sort_values(by=varying_params)
                .reset_index(drop=True)
            )
            combo_df["combo_id"] = range(1, len(combo_df) + 1)

            # Merge to get combo IDs
            data_with_ids = data.merge(combo_df, on=varying_params, how="left")

            # Group by combo_id
            grouped = (
                data_with_ids.groupby("combo_id")
                .agg(
                    {
                        mean_col: "mean",
                        std_col: (
                            "mean" if std_col in data_with_ids.columns else lambda x: 0
                        ),
                    }
                )
                .sort_index()
            )

            # Plot
            fig, ax = plt.subplots(figsize=(max(10, len(combo_df) * 0.5), 6))

            x_pos = grouped.index
            y_values = grouped[mean_col]
            yerr = grouped[std_col] if std_col in grouped.columns else None

            ax.bar(x_pos, y_values, yerr=yerr, capsize=3, alpha=0.7, color="#45B7D1")

            ax.set_xlabel("Parameter Combination ID (see legend)", fontsize=12)
            ax.set_ylabel(get_metric_label(metric), fontsize=12)
            ax.set_title(
                f"{get_metric_label(metric)} by Parameter Combination", fontsize=14
            )
            ax.set_xticks(x_pos)
            ax.set_xticklabels([str(int(x)) for x in x_pos], rotation=0)
            ax.grid(True, alpha=0.3, axis="y")

            fig.tight_layout()
            plot_path = output_dir / f"{metric}_by_combinations.png"
            return self.performance_plotter.save_figure(fig, plot_path)

        except Exception as e:
            logger.error(f"Error plotting {metric} by combinations: {e}")
            return None

    def _generate_rule_variant_comparison(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """Generate plots comparing different rule variants (3f, 4f, 5f)."""
        plot_paths = []
        rule_dir = output_dir / "rule_variant_comparison"
        rule_dir.mkdir(parents=True, exist_ok=True)

        rule_variants = data["rule_variant"].dropna().unique()
        if len(rule_variants) <= 1:
            logger.info("Only one rule variant found, skipping comparison")
            return plot_paths

        logger.info(f"Comparing rule variants: {rule_variants}")

        # Plot key metrics by rule variant
        metrics = ["test_accuracy", "mi_E_C", "mi_I_C", "mi_V_C"]

        for metric in metrics:
            mean_col = f"{metric}_mean"
            if mean_col not in data.columns:
                continue

            try:
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot each rule variant
                colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
                for idx, variant in enumerate(sorted(rule_variants)):
                    variant_data = data[data["rule_variant"] == variant]

                    if "ei_ratio" in variant_data.columns:
                        grouped = (
                            variant_data.groupby("ei_ratio")[mean_col]
                            .mean()
                            .sort_index()
                        )
                        ax.plot(
                            grouped.index,
                            grouped.values,
                            marker="o",
                            color=colors[idx % len(colors)],
                            label=f"{variant.upper()} rule",
                            linewidth=2,
                            markersize=6,
                        )

                self.performance_plotter.setup_axes_style(
                    ax=ax,
                    xlabel="E/I Ratio",
                    ylabel=get_metric_label(metric),
                    title=f"{get_metric_label(metric)} by Rule Variant",
                    use_log_x=True,
                    grid=True,
                )
                ax.legend(fontsize=10)

                fig.tight_layout()
                plot_path = rule_dir / f"{metric}_by_rule_variant.png"
                plot_paths.append(self.performance_plotter.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error plotting {metric} by rule variant: {e}")

        logger.info(f"Generated {len(plot_paths)} rule variant comparison plots")
        return plot_paths

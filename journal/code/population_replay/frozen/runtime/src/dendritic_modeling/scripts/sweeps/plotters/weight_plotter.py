"""Weight analysis plotters."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)


class WeightPlotter(BasePlotter):
    """Generates weight analysis plots."""

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all weight analysis plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        weight_dir = output_dir / "weights"
        weight_dir.mkdir(parents=True, exist_ok=True)

        # Check if we have weight data
        weight_cols = [col for col in data.columns if "weight_" in col.lower()]
        if not weight_cols:
            logger.info("No weight metrics found, skipping weight plots")
            return plot_paths

        # Generate weight vs E/I ratio plots
        if "ei_ratio" in data.columns:
            plot_paths.extend(self.plot_weights_vs_ei_ratio(data, weight_dir))

        # Generate E/I weight ratio plots
        plot_paths.extend(self.plot_exc_inh_ratio(data, weight_dir))

        return plot_paths

    def plot_weights_vs_ei_ratio(
        self, data: pd.DataFrame, output_dir: Path
    ) -> list[Path]:
        """
        Plot weight metrics vs E/I ratio.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Key weight metrics to plot
        weight_metrics = [
            "weight_exc_weight_mean_mean",
            "weight_inh_weight_mean_mean",
            "weight_exc_weight_var_mean",
            "weight_inh_weight_var_mean",
        ]

        for metric_base in weight_metrics:
            # Find columns matching this pattern (may have layer suffixes)
            matching_cols = [
                col
                for col in data.columns
                if col.startswith(metric_base) and col.endswith("_mean")
            ]

            if not matching_cols:
                continue

            try:
                fig, ax = plt.subplots(figsize=self.default_figsize)

                # Plot each layer/variant
                for col in matching_cols[:5]:  # Limit to first 5 to avoid overcrowding
                    # Extract layer info from column name
                    layer_suffix = col.replace(metric_base, "").replace("_mean", "")
                    label = f"{metric_base.replace('weight_', '').replace('_', ' ').title()}{layer_suffix}"

                    if "network_category" in data.columns:
                        # Plot by category
                        for category in data["network_category"].unique():
                            cat_data = data[
                                data["network_category"] == category
                            ].sort_values("ei_ratio")
                            if not cat_data.empty:
                                color = self.category_colors.get(category, "black")
                                linestyle = self.category_linestyles.get(category, "-")
                                ax.plot(
                                    cat_data["ei_ratio"],
                                    cat_data[col],
                                    label=f"{label} ({category})",
                                    color=color,
                                    linestyle=linestyle,
                                    marker="o",
                                    markersize=4,
                                )
                    else:
                        grouped = data.groupby("ei_ratio")[col].mean().sort_index()
                        ax.plot(
                            grouped.index,
                            grouped.values,
                            marker="o",
                            label=label,
                            linewidth=2,
                        )

                self.setup_axes_style(
                    ax=ax,
                    xlabel="E/I Ratio",
                    ylabel=metric_base.replace("weight_", "").replace("_", " ").title(),
                    title=f"Weight {metric_base.replace('weight_', '').replace('_', ' ').title()} vs E/I Ratio",
                    use_log_x=True,
                    grid=True,
                )

                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
                fig.tight_layout()

                plot_path = output_dir / f"{metric_base}_vs_ei_ratio.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating weight plot for {metric_base}: {e}")

        logger.info(f"Generated {len(plot_paths)} weight vs E/I plots")
        return plot_paths

    def plot_exc_inh_ratio(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Plot excitatory/inhibitory weight ratio.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Find excitatory and inhibitory weight columns
        exc_cols = [
            col
            for col in data.columns
            if "weight_exc_weight_mean_mean" in col and col.endswith("_mean")
        ]
        inh_cols = [
            col
            for col in data.columns
            if "weight_inh_weight_mean_mean" in col and col.endswith("_mean")
        ]

        if not exc_cols or not inh_cols or "ei_ratio" not in data.columns:
            return plot_paths

        try:
            fig, ax = plt.subplots(figsize=self.default_figsize)

            # Match exc and inh columns by layer suffix
            for exc_col in exc_cols:
                # Find corresponding inh column
                suffix = exc_col.replace("weight_exc_weight_mean_mean", "")
                inh_col = f"weight_inh_weight_mean_mean{suffix}"

                if inh_col in inh_cols:
                    # Compute ratio
                    ratio_data = data.copy()
                    ratio_data["weight_ratio"] = ratio_data[exc_col] / (
                        ratio_data[inh_col] + 1e-8
                    )

                    # Filter extreme outliers
                    ratio_data = ratio_data[ratio_data["weight_ratio"] < 50]

                    layer_label = suffix.replace("_", " ").strip() or "global"

                    # Plot by category
                    if "network_category" in ratio_data.columns:
                        for category in ratio_data["network_category"].unique():
                            cat_data = ratio_data[
                                ratio_data["network_category"] == category
                            ].sort_values("ei_ratio")
                            if not cat_data.empty:
                                color = self.category_colors.get(category, "black")
                                linestyle = self.category_linestyles.get(category, "-")
                                ax.plot(
                                    cat_data["ei_ratio"],
                                    cat_data["weight_ratio"],
                                    label=f"{layer_label} ({category})",
                                    color=color,
                                    linestyle=linestyle,
                                    marker="o",
                                    markersize=4,
                                )

            ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5, label="E/I = 1")

            self.setup_axes_style(
                ax=ax,
                xlabel="E/I Ratio",
                ylabel="E/I Weight Ratio",
                title="Excitatory/Inhibitory Weight Ratio vs E/I Synapse Ratio",
                use_log_x=True,
                use_log_y=True,
                grid=True,
            )

            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            fig.tight_layout()

            plot_path = output_dir / "exc_inh_weight_ratio_vs_ei.png"
            plot_paths.append(self.save_figure(fig, plot_path))

        except Exception as e:
            logger.error(f"Error generating E/I weight ratio plot: {e}")

        return plot_paths

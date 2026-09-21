"""Information theory plotters (inspired by comprehensive_analysis.py)."""

import logging
import traceback
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils.metrics import get_metric_label
from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)


class InformationPlotter(BasePlotter):
    """Generates information theory plots."""

    # Metric groups for organized plotting
    METRIC_GROUPS: ClassVar[dict[str, dict]] = {
        "class_information": {
            "metrics": ["mi_E_C", "mi_I_C", "mi_V_C"],
            "labels": ["I(E;C)", "I(I;C)", "I(V;C)"],
            "colors": ["#FF6B6B", "#4ECDC4", "#45B7D1"],  # Red, Teal, Blue
            "title": "Information with Class Labels",
            "ylabel": "Mutual Information (bits)",
        },
        "component_information": {
            "metrics": ["mi_E_I", "mi_E_V", "mi_I_V"],
            "labels": ["I(E;I)", "I(E;V)", "I(I;V)"],
            "colors": ["#8B5A96", "#FFA726", "#9C27B0"],  # Purple, Orange, Purple
            "title": "Component Information",
            "ylabel": "Mutual Information (bits)",
        },
        "conditional_information": {
            "metrics": ["mi_E_I_given_C", "mi_E_V_given_C", "mi_I_V_given_C"],
            "labels": ["I(E;I|C)", "I(E;V|C)", "I(I;V|C)"],
            "colors": [
                "#2E7D32",
                "#558B2F",
                "#827717",
            ],  # Dark green, Forest green, Olive
            "title": "Conditional Information",
            "ylabel": "Mutual Information (bits)",
        },
    }

    @staticmethod
    def _add_legend_if_handles(ax: plt.Axes, **kwargs) -> None:
        """Add a legend only when the axis has labeled artists."""
        handles, _labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(**kwargs)

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all information theory plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        info_dir = output_dir / "information"
        info_dir.mkdir(parents=True, exist_ok=True)

        # Check if we have any information metrics
        has_mi = any(col for col in data.columns if "mi_" in col)
        if not has_mi:
            logger.info("No information metrics found, skipping information plots")
            return plot_paths

        sweep_types = self.detect_sweep_types(data)

        if "ei" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_information_plots_1d_sweep(
                        data=data,
                        sweep_var="ei_ratio",
                        var_name="E/I Ratio",
                        logscale=True,
                        output_dir=info_dir,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating E/I information plots: {e}")
                traceback.print_exc()

        if "stim_duration" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_information_plots_1d_sweep(
                        data=data,
                        sweep_var="stimulus_duration",
                        var_name="Stimulus Duration",
                        logscale=True,
                        output_dir=info_dir,
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error generating stimulus duration information plots: {e}"
                )
                traceback.print_exc()

        if "max_gf" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_information_plots_1d_sweep(
                        data=data,
                        sweep_var="max_gain_factor",
                        var_name="Maximum Gain",
                        logscale=True,
                        output_dir=info_dir,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating max gain factor information plots: {e}")
                traceback.print_exc()

        if "fixed_gf" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_information_plots_1d_sweep(
                        data=data,
                        sweep_var="fixed_gain_factor",
                        var_name="Fixed Gain Factor",
                        logscale=True,
                        output_dir=info_dir,
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error generating fixed gain factor information plots: {e}"
                )
                traceback.print_exc()

        if "nparams" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_information_plots_1d_sweep(
                        data=data,
                        sweep_var="nparams",
                        var_name="Number of Parameters",
                        logscale=True,
                        output_dir=info_dir,
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error generating number of parameters information plots: {e}"
                )
                traceback.print_exc()

        if "stoch1" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_information_heatmaps(
                        data, info_dir, "stimulus_duration", "max_gain_factor"
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error generating stochastic dataset information plots: {e}"
                )
                traceback.print_exc()

        if "stoch2" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_information_heatmaps(
                        data, info_dir, "stimulus_duration", "max_gain_tau_ratio"
                    )
                )
            except Exception as e:
                logger.error(
                    f"Error generating stochastic GV dataset information plots: {e}"
                )
                traceback.print_exc()

        return plot_paths

    def _create_information_plots_1d_sweep(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        output_dir: Path,
    ) -> list[Path]:
        plot_paths = []
        # Generate comprehensive multi-panel plot
        panel_path = self.plot_information_panels_1d_sweep(
            data=data,
            sweep_var=sweep_var,
            var_name=var_name,
            logscale=logscale,
            output_dir=output_dir / "aggregate_plots",
        )
        if panel_path:
            plot_paths.append(panel_path)

        # Generate individual metric group plots
        plot_paths.extend(
            self.plot_metric_groups_1d_sweep(
                data=data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                output_dir=output_dir / "aggregate_plots",
            )
        )

        # Generate layer-wise information plots
        plot_paths.extend(
            self.plot_layerwise_information_1d_sweep(
                data=data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                output_dir=output_dir,
            )
        )

        # Legacy information plots (disabled by default; set include_legacy_plots=True to enable)
        if getattr(self, "include_legacy_plots", False):
            plot_paths.extend(
                self._generate_information_plots_1d_sweep(
                    valid_data=data,
                    sweep_var=sweep_var,
                    var_name=var_name,
                    logscale=logscale,
                    plots_dir=output_dir,
                )
            )

        # Generate new style information plots
        plot_paths.extend(
            self._generate_new_style_information_plots_1d_sweep(
                valid_data=data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                plots_dir=output_dir,
            )
        )

        return plot_paths

    def plot_information_panels_1d_sweep(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        output_dir: Path,
    ) -> Path:
        """
        Generate comprehensive multi-panel information plot.

        This creates a 2x3 grid showing:
        - Row 1: Class information, Component information, Conditional information
        - Row 2: Layer-wise I(V;C), Shunting advantage, Information variance

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            Path to saved plot
        """
        try:
            fig, axes = self.create_multi_panel_figure(
                nrows=2,
                ncols=3,
                figsize=(18, 12),
                title="Comprehensive Information Analysis",
            )
            axes = axes.flatten()

            # Panel 1-3: Metric groups
            for idx, (_group_name, group_info) in enumerate(self.METRIC_GROUPS.items()):
                if idx < 3:
                    self._plot_metric_group_panel(
                        data=data,
                        sweep_var=sweep_var,
                        var_name=var_name,
                        logscale=logscale,
                        ax=axes[idx],
                        group_info=group_info,
                    )

            # Panel 4: Layer-wise I(V;C)
            self._plot_layerwise_ivc_panel(
                data=data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                ax=axes[3],
            )

            # Panel 5: Shunting advantage
            self._plot_shunting_advantage_panel(
                data=data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                ax=axes[4],
            )

            # Panel 6: Information summary statistics
            self._plot_information_summary_panel(data, axes[5])

            fig.tight_layout()
            plot_path = output_dir / "information_panels.png"
            return self.save_figure(fig, plot_path)

        except Exception as e:
            logger.error(f"Error generating information panels: {e}")
            return None

    def _plot_metric_group_panel(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        ax: plt.Axes,
        group_info: dict,
    ):
        """Plot a single metric group panel."""
        metrics = group_info["metrics"]
        labels = group_info["labels"]
        colors = group_info["colors"]

        # Check which metrics are available
        available = [
            (m, lbl, c)
            for m, lbl, c in zip(metrics, labels, colors)
            if f"{m}_mean" in data.columns
        ]

        if not available:
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(group_info["title"])
            return

        # Plot each metric
        for metric, lbl, color in available:
            label = lbl
            mean_col = (
                f"{metric}_mean" if f"{metric}_mean" in data.columns else "metric"
            )

            # Group by ei_ratio and plot
            if sweep_var in data.columns:
                grouped = data.groupby(sweep_var)[mean_col].mean().sort_index()
                ax.plot(
                    grouped.index,
                    grouped.values,
                    marker="o",
                    color=color,
                    label=label,
                    linewidth=2,
                )

        self.setup_axes_style(
            ax=ax,
            xlabel=var_name,
            ylabel=group_info["ylabel"],
            title=group_info["title"],
            use_log_x=logscale,
            grid=True,
        )
        ax.legend(fontsize=9)

    def _plot_layerwise_ivc_panel(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        ax: plt.Axes,
    ):
        """Plot layer-wise I(V;C) across depths."""
        # Find layer MI columns
        layer_cols = [
            col
            for col in data.columns
            if "layer_mi_V_C_depth" in col and col.endswith("_mean")
        ]

        if not layer_cols:
            ax.text(
                0.5,
                0.5,
                "No layer data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Layer-wise I(V;C)")
            return

        # Extract depths
        depths = []
        for col in layer_cols:
            try:
                depth = int(col.split("_depth")[1].split("_mean")[0])
                depths.append(depth)
            except (ValueError, IndexError):
                pass

        depths = sorted(set(depths))

        if not depths:
            ax.text(
                0.5,
                0.5,
                "No depth data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Plot I(V;C) for each depth
        for depth in depths:
            col = f"layer_mi_V_C_depth{depth}_mean"
            if col in data.columns:
                if sweep_var in data.columns:
                    grouped = data.groupby(sweep_var)[col].mean().sort_index()
                    ax.plot(
                        grouped.index,
                        grouped.values,
                        marker="o",
                        label=f"Layer {depth}",
                        linewidth=2,
                    )

        self.setup_axes_style(
            ax=ax,
            xlabel=var_name,
            ylabel="I(V;C) (bits)",
            title="Layer-wise Information with Class",
            use_log_x=logscale,
            grid=True,
        )
        ax.legend(fontsize=8, ncol=2)

    def _plot_shunting_advantage_panel(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        ax: plt.Axes,
    ):
        """Plot shunting advantage (difference in metrics)."""
        metric = "test_accuracy"
        mean_col = f"{metric}_mean" if f"{metric}_mean" in data.columns else "metric"

        if mean_col not in data.columns or "network_category" not in data.columns:
            ax.text(
                0.5,
                0.5,
                "No shunting data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Shunting Advantage")
            return

        # Calculate difference
        shunting_data = data[data["network_category"] == "dendritic_shunting"]
        no_shunting_data = data[data["network_category"] == "dendritic_additive"]

        if shunting_data.empty or no_shunting_data.empty:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Merge on ei_ratio
        merged = pd.merge(
            shunting_data[[sweep_var, mean_col]],
            no_shunting_data[[sweep_var, mean_col]],
            on=sweep_var,
            suffixes=("_shunt", "_no_shunt"),
        )

        merged["advantage"] = (
            merged[f"{mean_col}_shunt"] - merged[f"{mean_col}_no_shunt"]
        )

        ax.plot(
            merged[sweep_var],
            merged["advantage"],
            marker="o",
            color="#2E7D32",
            linewidth=2,
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        self.setup_axes_style(
            ax=ax,
            xlabel=var_name,
            ylabel="Accuracy Difference (Shunting - No Shunting)",
            title="Shunting Advantage",
            use_log_x=logscale,
            grid=True,
        )

    def _plot_information_summary_panel(self, data: pd.DataFrame, ax: plt.Axes):
        """Plot summary statistics of information metrics."""
        # Collect all MI metrics
        mi_metrics = [
            col.replace("_mean", "")
            for col in data.columns
            if "mi_" in col and col.endswith("_mean")
        ]

        if not mi_metrics:
            ax.text(
                0.5,
                0.5,
                "No MI metrics",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title("Information Summary")
            return

        # Compute mean and std across all ei_ratios for each metric
        means = []
        stds = []
        labels = []

        for metric in mi_metrics[:10]:  # Limit to avoid overcrowding
            mean_col = (
                f"{metric}_mean" if f"{metric}_mean" in data.columns else "metric"
            )
            if mean_col in data.columns:
                values = data[mean_col].dropna()
                if len(values) > 0:
                    means.append(values.mean())
                    stds.append(values.std())
                    labels.append(get_metric_label(metric))

        if means:
            x_pos = np.arange(len(labels))
            ax.bar(x_pos, means, yerr=stds, capsize=3, alpha=0.7, color="#45B7D1")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Mean MI (bits)", fontsize=10)
            ax.set_title("Information Metric Summary", fontsize=12)
            ax.grid(True, alpha=0.3, axis="y")

    def plot_metric_groups_1d_sweep(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        output_dir: Path,
    ) -> list[Path]:
        """
        Generate individual plots for each metric group.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []

        if sweep_var not in data.columns or "network_category" not in data.columns:
            return plot_paths

        for group_name, group_info in self.METRIC_GROUPS.items():
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # Plot for shunting and no_shunting separately
                for use_shunting, ax, title_suffix in [
                    (True, ax1, "Shunting"),
                    (False, ax2, "No Shunting"),
                ]:
                    category = (
                        "dendritic_shunting" if use_shunting else "dendritic_additive"
                    )
                    cat_data = data[data["network_category"] == category]

                    if cat_data.empty:
                        ax.text(
                            0.5,
                            0.5,
                            f"No {category} data",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"{group_info['title']} - {title_suffix}")
                        continue

                    # Plot each metric
                    for metric, label, color in zip(
                        group_info["metrics"],
                        group_info["labels"],
                        group_info["colors"],
                    ):
                        mean_col = (
                            f"{metric}_mean"
                            if f"{metric}_mean" in data.columns
                            else "metric"
                        )
                        if mean_col in cat_data.columns:
                            grouped = (
                                cat_data.groupby(sweep_var)[mean_col]
                                .mean()
                                .sort_index()
                            )
                            ax.plot(
                                grouped.index,
                                grouped.values,
                                marker="o",
                                color=color,
                                label=label,
                                linewidth=2,
                                markersize=6,
                            )

                    self.setup_axes_style(
                        ax=ax,
                        xlabel=var_name,
                        ylabel=group_info["ylabel"],
                        title=f"{group_info['title']} - {title_suffix}",
                        use_log_x=logscale,
                        grid=True,
                    )
                    self._add_legend_if_handles(ax, fontsize=10)

                fig.tight_layout()
                plot_path = output_dir / f"{group_name}.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating {group_name} plot: {e}")

        logger.info(f"Generated {len(plot_paths)} metric group plots")
        return plot_paths

    def plot_layerwise_information_1d_sweep(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        output_dir: Path,
    ) -> list[Path]:
        """
        Generate layer-wise information plots and heatmaps.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        layer_dir = output_dir / "layer_information"
        layer_dir.mkdir(parents=True, exist_ok=True)

        # Generate I(V;C) heatmap by layer
        heatmap_path = self._plot_ivc_heatmap_by_layer(
            data=data, sweep_var=sweep_var, var_name=var_name, output_dir=layer_dir
        )
        if heatmap_path:
            plot_paths.append(heatmap_path)

        # Generate first vs last layer comparison
        comparison_paths = self._plot_first_last_layer_comparison(
            data=data,
            sweep_var=sweep_var,
            var_name=var_name,
            logscale=logscale,
            output_dir=layer_dir,
        )
        plot_paths.extend(comparison_paths)

        logger.info(f"Generated {len(plot_paths)} layer-wise information plots")
        return plot_paths

    def _plot_ivc_heatmap_by_layer(
        self, data: pd.DataFrame, sweep_var: str, var_name: str, output_dir: Path
    ) -> Path:
        """Generate I(V;C) heatmap across layers and sweep variable."""
        # Find layer MI columns for I(V;C)
        ivc_cols = [
            col
            for col in data.columns
            if "layer_mi_V_C_depth" in col and col.endswith("_mean")
        ]

        if not ivc_cols:
            logger.info("No layer I(V;C) data for heatmap")
            return None

        try:
            # Extract layer data
            layer_data = []
            for col in ivc_cols:
                depth = int(col.split("_depth")[1].split("_mean")[0])
                for _, row in data.iterrows():
                    if not pd.isna(row[col]) and sweep_var in row:
                        layer_data.append(
                            {
                                "layer": depth,
                                sweep_var: row[sweep_var],
                                "use_shunting": row.get("network_category")
                                == "dendritic_shunting",
                                "ivc_value": row[col],
                            }
                        )

            if not layer_data:
                return None

            layer_df = pd.DataFrame(layer_data)

            # Create dual heatmaps
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

            # Calculate global vmin/vmax for consistent scaling
            vmin, vmax = layer_df["ivc_value"].min(), layer_df["ivc_value"].max()

            for use_shunting, ax, title in [
                (True, ax1, "Shunting"),
                (False, ax2, "Non-Shunting"),
            ]:
                subset = layer_df[layer_df["use_shunting"] == use_shunting]

                if subset.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {title} data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"I(V;C) Heatmap - {title}")
                    continue

                pivot = subset.pivot_table(
                    values="ivc_value",
                    index="layer",
                    columns=sweep_var,
                    aggfunc="mean",
                )

                sns.heatmap(
                    pivot,
                    cmap="viridis",
                    annot=True,
                    fmt=".2f",
                    cbar_kws={"label": "I(V;C) (bits)"},
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                )
                # Invert y-axis to place origin at lower left
                ax.invert_yaxis()

                ax.set_xlabel(var_name, fontsize=12)
                ax.set_ylabel("Layer Index", fontsize=12)
                ax.set_title(f"I(V;C) Heatmap - {title}", fontsize=14)

            fig.suptitle(f"Information with Class by Layer and {var_name}", fontsize=16)
            fig.tight_layout()

            plot_path = output_dir / "ivc_heatmap_by_layer.png"
            return self.save_figure(fig, plot_path)

        except Exception as e:
            logger.error(f"Error generating I(V;C) heatmap: {e}")
            return None

    def _plot_first_last_layer_comparison(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        output_dir: Path,
    ) -> list[Path]:
        """Generate first vs last layer comparison plots."""
        plot_paths = []

        # Find all layer MI columns
        layer_cols = [
            col
            for col in data.columns
            if "layer_mi_" in col and "_depth" in col and col.endswith("_mean")
        ]

        if not layer_cols:
            return plot_paths

        # Extract depths
        depths = set()
        for col in layer_cols:
            try:
                depth = int(col.split("_depth")[1].split("_mean")[0])
                depths.add(depth)
            except (ValueError, IndexError):
                pass

        depths = sorted(depths)
        if len(depths) < 2:
            logger.info(f"Not enough layers ({len(depths)}) for first/last comparison")
            return plot_paths

        first_layer, last_layer = depths[0], depths[-1]
        logger.info(
            f"Comparing layer {first_layer} (first) vs layer {last_layer} (last)"
        )

        # Generate comparison plots for each metric group
        for group_name, group_info in self.METRIC_GROUPS.items():
            try:
                plot_path = self._plot_metric_group_first_last(
                    data=data,
                    sweep_var=sweep_var,
                    var_name=var_name,
                    logscale=logscale,
                    first_layer=first_layer,
                    last_layer=last_layer,
                    group_name=group_name,
                    group_info=group_info,
                    output_dir=output_dir,
                )
                if plot_path:
                    plot_paths.append(plot_path)
            except Exception as e:
                logger.error(
                    f"Error generating first/last comparison for {group_name}: {e}"
                )

        return plot_paths

    def _plot_metric_group_first_last(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        first_layer: int,
        last_layer: int,
        group_name: str,
        group_info: dict,
        output_dir: Path,
    ) -> Path:
        """Plot first vs last layer comparison for a metric group."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for layer_idx, ax, title in [
            (first_layer, ax1, f"Layer {first_layer} (First)"),
            (last_layer, ax2, f"Layer {last_layer} (Last)"),
        ]:
            # Plot each metric for both shunting conditions
            for metric, label, color in zip(
                group_info["metrics"],
                group_info["labels"],
                group_info["colors"],
            ):
                # Get base metric name (e.g., "E_C" from "mi_E_C")
                base_metric = metric.replace("mi_", "")
                layer_col = f"layer_mi_{base_metric}_depth{layer_idx}_mean"

                if layer_col not in data.columns:
                    continue

                # Plot for both shunting conditions
                for use_shunting, linestyle, alpha in [
                    (True, "-", 0.9),
                    (False, "--", 0.6),
                ]:
                    category = (
                        "dendritic_shunting" if use_shunting else "dendritic_additive"
                    )
                    cat_data = data[data["network_category"] == category]

                    if cat_data.empty or sweep_var not in cat_data.columns:
                        continue

                    grouped = cat_data.groupby(sweep_var)[layer_col].mean().sort_index()
                    ax.plot(
                        grouped.index,
                        grouped.values,
                        color=color,
                        linestyle=linestyle,
                        label=f"{label} ({'S' if use_shunting else 'NS'})",
                        linewidth=2,
                        marker="o" if use_shunting else "s",
                        markersize=5,
                        alpha=alpha,
                    )

            self.setup_axes_style(
                ax=ax,
                xlabel=var_name,
                ylabel=group_info["ylabel"],
                title=title,
                use_log_x=logscale,
                grid=True,
            )
            self._add_legend_if_handles(
                ax,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
                fontsize=8,
            )

        fig.suptitle(f"{group_info['title']} - First vs Last Layer", fontsize=16)
        fig.tight_layout()

        plot_path = output_dir / f"{group_name}_first_last_layer.png"
        return self.save_figure(fig, plot_path)

    def _plot_shunting_advantage_panel(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        ax: plt.Axes,
    ):
        """Plot shunting advantage across sweep variable."""
        # Calculate shunting advantage for test accuracy
        self._plot_shunting_advantage_for_metric(
            data=data,
            sweep_var=sweep_var,
            var_name=var_name,
            logscale=logscale,
            ax=ax,
            metric="test_accuracy",
        )

    def _plot_shunting_advantage_for_metric(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        ax: plt.Axes,
        metric: str,
    ):
        """Plot shunting advantage for a specific metric."""
        mean_col = f"{metric}_mean" if f"{metric}_mean" in data.columns else "metric"

        if mean_col not in data.columns or "network_category" not in data.columns:
            ax.text(
                0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes
            )
            ax.set_title("Shunting Advantage")
            return

        # Separate shunting and no_shunting data
        shunting = data[data["network_category"] == "dendritic_shunting"]
        no_shunting = data[data["network_category"] == "dendritic_additive"]

        if shunting.empty or no_shunting.empty or sweep_var not in data.columns:
            ax.text(
                0.5,
                0.5,
                "Insufficient data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return

        # Merge on ei_ratio
        merged = pd.merge(
            shunting[[sweep_var, mean_col]],
            no_shunting[[sweep_var, mean_col]],
            on=sweep_var,
            suffixes=("_shunt", "_no_shunt"),
        )

        merged["advantage"] = (
            merged[f"{mean_col}_shunt"] - merged[f"{mean_col}_no_shunt"]
        )
        merged = merged.sort_values(sweep_var)

        ax.plot(
            merged[sweep_var],
            merged["advantage"],
            marker="o",
            color="#2E7D32",
            linewidth=2,
            markersize=6,
        )
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.fill_between(
            merged[sweep_var], 0, merged["advantage"], alpha=0.3, color="#2E7D32"
        )

        self.setup_axes_style(
            ax=ax,
            xlabel=var_name,
            ylabel=f"{get_metric_label(metric)} Difference",
            title="Shunting Advantage (Shunting - Non-Shunting)",
            use_log_x=logscale,
            grid=True,
        )

    def _generate_information_plots_1d_sweep(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        plots_dir: Path,
    ) -> list[str]:
        """Generate information theory plots matching old analysis format"""
        plot_paths = []

        if sweep_var not in valid_data.columns:
            return plot_paths

        info_plot_dir = plots_dir / "information_theory_plots"

        # Collect global and layer-specific MI metrics
        global_mi_metrics = []
        layer_mi_metrics = []

        for col in valid_data.columns:
            if any(
                mi_pattern in col.lower()
                for mi_pattern in ["mi_", "mutual_information"]
            ):
                # Extract the metric name
                if "_mean" in col:
                    metric_name = col.replace("_mean", "")
                    if "layer_mi_" in metric_name:
                        # This is a layer-specific metric
                        base_metric = metric_name.replace("layer_mi_", "").split(
                            "_depth"
                        )[0]
                        if base_metric not in layer_mi_metrics:
                            layer_mi_metrics.append(base_metric)
                    else:
                        # This is a global metric
                        if metric_name not in global_mi_metrics:
                            global_mi_metrics.append(metric_name)

        if not global_mi_metrics and not layer_mi_metrics:
            print("No mutual information metrics found")
            return plot_paths

        if (
            not valid_data["network_category"]
            .isin(["dendritic_shunting", "dendritic_additive"])
            .any()
        ):
            print(
                "No dendritic_shunting/dendritic_additive data for information analysis"
            )
            return plot_paths

        # Define metric groups matching old analysis exactly
        metric_groups = {
            "class_information": {
                "metrics": ["mi_E_C", "mi_I_C", "mi_V_C"],
                "labels": ["I(E;C)", "I(I;C)", "I(V;C)"],
                "colors": [
                    self.info_colors["excitatory"],
                    self.info_colors["inhibitory"],
                    self.info_colors["combined"],
                ],
                "title": "Information with Class Labels",
                "ylabel": "Mutual Information (bits)",
            },
            "component_information": {
                "metrics": ["mi_E_I", "mi_E_V", "mi_I_V"],
                "labels": ["I(E;I)", "I(E;V)", "I(I;V)"],
                "colors": ["purple", "orange", "brown"],
                "title": "Component Information",
                "ylabel": "Mutual Information (bits)",
            },
            "conditional_information": {
                "metrics": ["mi_E_I_given_C", "mi_E_V_given_C", "mi_I_V_given_C"],
                "labels": ["I(E;I|C)", "I(E;V|C)", "I(I;V|C)"],
                "colors": ["darkgreen", "forestgreen", "olive"],
                "title": "Conditional Information",
                "ylabel": "Mutual Information (bits)",
            },
        }

        # Generate global information plots grouped by metric type (matching old format)
        for group_name, group_info in metric_groups.items():
            metrics = group_info["metrics"]
            labels = group_info["labels"]
            metric_colors = group_info["colors"]
            group_title = group_info["title"]
            ylabel = group_info["ylabel"]

            # Check which metrics are available
            available_metrics = []
            available_labels = []
            available_colors = []

            for metric, label, color in zip(metrics, labels, metric_colors):
                mean_col = f"{metric}_mean"
                if mean_col in valid_data.columns:
                    available_metrics.append(metric)
                    available_labels.append(label)
                    available_colors.append(color)

            if not available_metrics:
                print(f"No {group_name} metrics found")
                continue

            try:
                # Create separate plots for shunting vs no shunting (matching old analysis)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

                # First pass: collect all y-values to determine global y-axis range
                all_y_values = []
                plot_data = {}

                for use_shunting in [True, False]:
                    category = (
                        "dendritic_shunting" if use_shunting else "dendritic_additive"
                    )
                    shunt_data = valid_data[valid_data["network_category"] == category]
                    plot_data[use_shunting] = {}

                    if not shunt_data.empty:
                        for metric in available_metrics:
                            mean_col = f"{metric}_mean"
                            # Group by E/I ratio and calculate statistics
                            grouped = shunt_data.groupby(sweep_var)[mean_col].mean()
                            plot_data[use_shunting][metric] = grouped
                            all_y_values.extend(grouped.values)

                # Calculate global y-axis range
                if all_y_values:
                    global_y_min = min(all_y_values)
                    global_y_max = max(all_y_values)
                    y_margin = (global_y_max - global_y_min) * 0.1
                    final_y_min = global_y_min - y_margin
                    final_y_max = global_y_max + y_margin
                else:
                    final_y_min, final_y_max = 0, 1

                # Second pass: create the actual plots
                for use_shunting, ax, title_suffix in [
                    (True, ax1, "Shunting"),
                    (False, ax2, "No Shunting"),
                ]:
                    if use_shunting not in plot_data or not plot_data[use_shunting]:
                        ax.text(
                            0.5,
                            0.5,
                            f"No data for\\n{title_suffix}",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                        )
                        ax.set_title(f"{group_title}\\n{title_suffix} Networks")
                        ax.set_ylim(final_y_min, final_y_max)
                        continue

                    # Plot each metric
                    for metric, label, color in zip(
                        available_metrics, available_labels, available_colors
                    ):
                        if metric in plot_data[use_shunting]:
                            grouped = plot_data[use_shunting][metric]
                            ax.plot(
                                grouped.index,
                                grouped.values,
                                marker="o",
                                color=color,
                                label=label,
                                linewidth=2,
                                markersize=6,
                            )

                    ax.legend(fontsize=10)
                    ax.set_ylim(final_y_min, final_y_max)

                    self.setup_axes_style(
                        ax=ax,
                        xlabel=var_name,
                        ylabel=ylabel,
                        title=f"{group_title}\\n{title_suffix} Networks",
                        use_log_x=logscale,
                        grid=True,
                    )

                plt.suptitle(f"Branch Information Analysis: {group_title}", fontsize=16)
                plt.tight_layout()

                plot_path = info_plot_dir / f"{group_name}_branch_info.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                print(f"Error generating {group_name} plot: {e}")

        # # Generate layer-specific information plots and heatmaps
        # layer_plot_dir = os.path.join(plots_dir, "layer_information_plots")
        # os.makedirs(layer_plot_dir, exist_ok=True)

        # # Create I(V;C) heatmap by layer (matching old analysis)
        # plot_paths.extend(
        #     self._create_ivc_heatmap_by_layer(valid_data, layer_plot_dir, base_name)
        # )

        # # Create first vs last layer comparison plots
        # plot_paths.extend(
        #     self._create_first_last_layer_plots(valid_data, layer_plot_dir, base_name)
        # )

        print(
            f"Created {len(plot_paths)} information theory plots (global + layer-specific + heatmaps)"
        )
        return plot_paths

    def _generate_new_style_information_plots_1d_sweep(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        plots_dir: Path,
    ) -> list[str]:
        plot_paths = []

        info_plot_dir = plots_dir / "new_style_info_plots"

        plot_paths.extend(
            self._plot_new_style_class_info_by_layer(
                valid_data=valid_data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                plots_dir=info_plot_dir / "class_information",
            )
        )

        plot_paths.extend(
            self._plot_new_style_component_info_by_layer(
                valid_data=valid_data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                plots_dir=info_plot_dir / "component_information",
            )
        )

        plot_paths.extend(
            self._plot_new_style_conditional_info_by_layer(
                valid_data=valid_data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                plots_dir=info_plot_dir / "conditional_information",
            )
        )

        plot_paths.extend(
            self._plot_new_style_coinfo_by_layer(
                valid_data=valid_data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                plots_dir=info_plot_dir / "interaction_information",
            )
        )

        plot_paths.extend(
            self._plot_new_style_total_info_panels(
                valid_data=valid_data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                plots_dir=info_plot_dir / "total_information",
            )
        )

        plot_paths.extend(
            self._plot_new_style_soma_info(
                valid_data=valid_data,
                sweep_var=sweep_var,
                var_name=var_name,
                logscale=logscale,
                plots_dir=info_plot_dir,
            )
        )

        return plot_paths

    def _plot_new_style_total_info_panels(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        plots_dir: Path,
    ) -> list[Path]:
        """
        Two 2x3 figures: one subplot per ``network_category`` (dendritic/flat x shunting/additive/signed).
        Uses aggregate ``mi_*_mean`` columns, matching ``plot_metric_groups_1d_sweep``
        (groupby sweep, mean). Interaction terms: I(X;Y;C) = I(X;Y) - I(X;Y|C).
        """
        plot_paths: list[Path] = []
        plots_dir.mkdir(parents=True, exist_ok=True)

        if (
            sweep_var not in valid_data.columns
            or "network_category" not in valid_data.columns
        ):
            return plot_paths

        layout: list[tuple[str, int, int]] = [
            ("dendritic_shunting", 0, 0),
            ("flat_shunting", 1, 0),
            ("dendritic_additive", 0, 1),
            ("flat_additive", 1, 1),
            ("dendritic_signed", 0, 2),
            ("flat_signed", 1, 2),
        ]

        def grouped_metric(cat_df: pd.DataFrame, metric: str) -> pd.Series | None:
            col = f"{metric}_mean"
            if col not in cat_df.columns:
                return None
            s = cat_df.groupby(sweep_var)[col].mean().sort_index()
            return s if not s.empty else None

        def coinfo_series(
            cat_df: pd.DataFrame, mi_xy: str, mi_xy_given_c: str
        ) -> pd.Series | None:
            s_xy = grouped_metric(cat_df, mi_xy)
            s_c = grouped_metric(cat_df, mi_xy_given_c)
            if s_xy is None or s_c is None:
                return None
            idx = s_xy.index.intersection(s_c.index)
            if len(idx) == 0:
                return None
            return (s_xy.loc[idx] - s_c.loc[idx]).sort_index()

        def plot_series_on_ax(
            ax: plt.Axes,
            series: pd.Series | None,
            label: str,
            color: str,
        ) -> None:
            if series is None or series.empty:
                return
            ax.plot(
                series.index,
                series.values,
                marker="o",
                color=color,
                label=label,
                linewidth=2,
                markersize=5,
            )

        def fill_figure(
            fig: plt.Figure,
            axes: np.ndarray,
            series_specs: list[tuple[str, str, str | None, str]],
            *,
            suptitle: str,
            ylabel: str,
            show_zero_line: bool,
        ) -> None:
            # (legend_label, mi_xy, mi_xy_given_c or None, color)
            for category, row, col in layout:
                ax = axes[row, col]
                cat_df = valid_data[valid_data["network_category"] == category]

                if cat_df.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {category} data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                else:
                    for lbl, m_xy, m_given, color in series_specs:
                        if m_given is not None:
                            s = coinfo_series(cat_df, m_xy, m_given)
                        else:
                            s = grouped_metric(cat_df, m_xy)
                        plot_series_on_ax(ax, s, lbl, color)

                if show_zero_line:
                    ax.axhline(
                        y=0, color="gray", linestyle="--", alpha=0.45, linewidth=1
                    )

                self.setup_axes_style(
                    ax=ax,
                    xlabel=var_name if row == 1 else None,
                    ylabel=ylabel,
                    title=category.replace("_", " ").title(),
                    use_log_x=logscale,
                    grid=True,
                )
                ax.legend(fontsize=8, loc="best")

            fig.suptitle(suptitle, fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.96])

        comp_colors = self.METRIC_GROUPS["component_information"]["colors"]
        # E-V: cyan/teal (light = pairwise MI, dark = interaction); contrasts with I-V magenta/purple.
        ev_pair_light = "#5DCEC4"
        ev_pair_dark = "#004D52"
        # E-I: amber/rust (light/dark), distinct from cyan (E-V) and purple (I-V).
        ei_pair_light = "#FFCA70"
        ei_pair_dark = "#A82A00"

        # Figure 1: I(E;V), I(E;V;C), I(I;V), I(I;V;C)
        need_ev = "mi_E_V_mean" in valid_data.columns
        need_iv = "mi_I_V_mean" in valid_data.columns
        if need_ev or need_iv:
            fig1, axes1 = plt.subplots(2, 3, figsize=(21, 9))
            specs_f1: list[tuple[str, str, str | None, str]] = [
                ("I(E;V)", "mi_E_V", None, ev_pair_light),
                ("I(E;V;C)", "mi_E_V", "mi_E_V_given_C", ev_pair_dark),
                ("I(I;V)", "mi_I_V", None, comp_colors[2]),
                ("I(I;V;C)", "mi_I_V", "mi_I_V_given_C", "#4A148C"),
            ]
            fill_figure(
                fig1,
                axes1,
                specs_f1,
                suptitle=(f"E-V and I-V Information vs {var_name}"),
                ylabel="Information (bits)",
                show_zero_line=True,
            )
            p1 = plots_dir / f"total_EV_IV_panels_{sweep_var}.png"
            plot_paths.append(self.save_figure(fig1, p1))

        # Figure 2: I(E;I), I(E;I;C)
        if "mi_E_I_mean" in valid_data.columns:
            fig2, axes2 = plt.subplots(2, 3, figsize=(21, 9))
            specs_f2: list[tuple[str, str, str | None, str]] = [
                ("I(E;I)", "mi_E_I", None, ei_pair_light),
                ("I(E;I;C)", "mi_E_I", "mi_E_I_given_C", ei_pair_dark),
            ]
            fill_figure(
                fig2,
                axes2,
                specs_f2,
                suptitle=(f"E-I Information vs {var_name}"),
                ylabel="Information (bits)",
                show_zero_line=True,
            )
            p2 = plots_dir / f"total_EI_panels_{sweep_var}.png"
            plot_paths.append(self.save_figure(fig2, p2))

        return plot_paths

    def _plot_new_style_soma_info(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        plots_dir: Path,
    ) -> list[Path]:
        """
        Analogous to SNR ``_create_soma_snr_plots_1d_sweep``: one panel, all categories,
        error bars; y = I(V;C) at soma / layer 0 only (``layer_mi_V_C_depth0``).
        """
        plot_paths: list[Path] = []
        y_col = "layer_mi_V_C_depth0"

        if f"{y_col}_mean" not in valid_data.columns:
            return plot_paths
        if (
            sweep_var not in valid_data.columns
            or "network_category" not in valid_data.columns
        ):
            return plot_paths

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(
            f"Soma Information: I(V;C) vs {var_name}",
            fontsize=16,
        )
        self.plot_by_category(
            ax=ax,
            data=valid_data,
            x_col=sweep_var,
            y_col=y_col,
            category_col="network_category",
            show_error=True,
        )
        self.setup_axes_style(
            ax=ax,
            xlabel=var_name,
            ylabel="I(V;C) (bits)",
            use_log_x=logscale,
            grid=True,
        )
        ax.legend(fontsize=10)
        fig.tight_layout()

        plot_path = plots_dir / f"soma_mi_V_C_vs_{sweep_var}.png"
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _plot_new_style_class_info_by_layer(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        plots_dir: Path,
    ) -> list[Path]:
        """
        Ablation-style figures (1x2 shunting vs additive): one PNG per class metric,
        each line is a layer depth (layer_mi_* columns).
        """
        plot_paths: list[Path] = []
        plots_dir.mkdir(parents=True, exist_ok=True)
        group_info = self.METRIC_GROUPS["class_information"]

        if (
            sweep_var not in valid_data.columns
            or "network_category" not in valid_data.columns
        ):
            return plot_paths

        if (
            not valid_data["network_category"]
            .isin(["dendritic_shunting", "dendritic_additive", "dendritic_signed"])
            .any()
        ):
            return plot_paths

        categories = ["dendritic_shunting", "dendritic_additive", "dendritic_signed"]

        for metric, label in zip(group_info["metrics"], group_info["labels"]):
            base_metric = metric.replace("mi_", "")
            prefix = f"layer_mi_{base_metric}_depth"
            depths: list[int] = []
            for col in valid_data.columns:
                if col.startswith(prefix) and col.endswith("_mean"):
                    try:
                        depth_str = col[len(prefix) : -len("_mean")]
                        depths.append(int(depth_str))
                    except ValueError:
                        continue
            depths = sorted(set(depths))
            if not depths:
                continue

            cmap = plt.get_cmap("tab10")
            depth_colors = {d: cmap(d / 9.0) for d in depths}

            fig, axes = plt.subplots(1, 3, figsize=(21, 6))
            fig.suptitle(
                f"{group_info['title']}: {label} vs {var_name} by Layer",
                fontsize=16,
            )

            depths.reverse()
            for idx, category in enumerate(categories):
                ax = axes[idx]
                category_data = valid_data[valid_data["network_category"] == category]

                if category_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {category} data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(category.replace("_", " ").title())
                    continue

                for depth in depths:
                    depth_col = f"layer_mi_{base_metric}_depth{depth}_mean"
                    if depth_col not in category_data.columns:
                        continue

                    sorted_data = category_data.sort_values(sweep_var)
                    plot_data = sorted_data[[depth_col, sweep_var]].dropna()
                    if plot_data.empty:
                        continue

                    try:
                        y_vals = pd.to_numeric(
                            plot_data[depth_col], errors="coerce"
                        ).dropna()
                        x_vals = plot_data.loc[y_vals.index, sweep_var]
                        if len(y_vals) > 0:
                            ax.plot(
                                x_vals,
                                y_vals,
                                marker="o",
                                label=f"Layer {depth}",
                                color=depth_colors[depth],
                                alpha=0.8,
                                linewidth=2,
                            )
                    except Exception as e:
                        logger.warning(
                            "Could not plot layer %s for %s: %s", depth, category, e
                        )
                        continue

                self.setup_axes_style(
                    ax=ax,
                    xlabel=var_name,
                    ylabel=group_info["ylabel"],
                    title=category.replace("_", " ").title(),
                    use_log_x=logscale,
                    grid=True,
                )
                ax.legend()

            fig.tight_layout()
            plot_path = plots_dir / f"class_info_layer_{base_metric}_{sweep_var}.png"
            plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _plot_new_style_component_info_by_layer(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        plots_dir: Path,
    ) -> list[Path]:
        """
        Ablation-style figures (1x2 shunting vs additive): one PNG per component metric,
        each line is a layer depth.
        """
        plot_paths: list[Path] = []
        plots_dir.mkdir(parents=True, exist_ok=True)
        group_info = self.METRIC_GROUPS["component_information"]

        if (
            sweep_var not in valid_data.columns
            or "network_category" not in valid_data.columns
        ):
            return plot_paths

        if (
            not valid_data["network_category"]
            .isin(["dendritic_shunting", "dendritic_additive", "dendritic_signed"])
            .any()
        ):
            return plot_paths

        categories = ["dendritic_shunting", "dendritic_additive", "dendritic_signed"]

        for metric, label in zip(group_info["metrics"], group_info["labels"]):
            base_metric = metric.replace("mi_", "")
            prefix = f"layer_mi_{base_metric}_depth"
            depths: list[int] = []
            for col in valid_data.columns:
                if col.startswith(prefix) and col.endswith("_mean"):
                    try:
                        depth_str = col[len(prefix) : -len("_mean")]
                        depths.append(int(depth_str))
                    except ValueError:
                        continue
            depths = sorted(set(depths))
            if not depths:
                continue

            cmap = plt.get_cmap("tab10")
            depth_colors = {d: cmap(d / 9.0) for d in depths}

            fig, axes = plt.subplots(1, 3, figsize=(21, 6))
            fig.suptitle(
                f"{group_info['title']}: {label} vs {var_name} by Layer",
                fontsize=16,
            )

            depths.reverse()
            for idx, category in enumerate(categories):
                ax = axes[idx]
                category_data = valid_data[valid_data["network_category"] == category]

                if category_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {category} data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(category.replace("_", " ").title())
                    continue

                for depth in depths:
                    depth_col = f"layer_mi_{base_metric}_depth{depth}_mean"
                    if depth_col not in category_data.columns:
                        continue

                    sorted_data = category_data.sort_values(sweep_var)
                    plot_data = sorted_data[[depth_col, sweep_var]].dropna()
                    if plot_data.empty:
                        continue

                    try:
                        y_vals = pd.to_numeric(
                            plot_data[depth_col], errors="coerce"
                        ).dropna()
                        x_vals = plot_data.loc[y_vals.index, sweep_var]
                        if len(y_vals) > 0:
                            ax.plot(
                                x_vals,
                                y_vals,
                                marker="o",
                                label=f"Layer {depth}",
                                color=depth_colors[depth],
                                alpha=0.8,
                                linewidth=2,
                            )
                    except Exception as e:
                        logger.warning(
                            "Could not plot layer %s for %s: %s", depth, category, e
                        )
                        continue

                self.setup_axes_style(
                    ax=ax,
                    xlabel=var_name,
                    ylabel=group_info["ylabel"],
                    title=category.replace("_", " ").title(),
                    use_log_x=logscale,
                    grid=True,
                )
                ax.legend()

            fig.tight_layout()
            plot_path = (
                plots_dir / f"component_info_layer_{base_metric}_{sweep_var}.png"
            )
            plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _plot_new_style_conditional_info_by_layer(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        plots_dir: Path,
    ) -> list[Path]:
        """
        Ablation-style figures (1x2 shunting vs additive): one PNG per conditional metric,
        each line is a layer depth.
        """
        plot_paths: list[Path] = []
        plots_dir.mkdir(parents=True, exist_ok=True)
        group_info = self.METRIC_GROUPS["conditional_information"]

        if (
            sweep_var not in valid_data.columns
            or "network_category" not in valid_data.columns
        ):
            return plot_paths

        if (
            not valid_data["network_category"]
            .isin(["dendritic_shunting", "dendritic_additive", "dendritic_signed"])
            .any()
        ):
            return plot_paths

        categories = ["dendritic_shunting", "dendritic_additive", "dendritic_signed"]

        for metric, label in zip(group_info["metrics"], group_info["labels"]):
            base_metric = metric.replace("mi_", "")
            prefix = f"layer_mi_{base_metric}_depth"
            depths: list[int] = []
            for col in valid_data.columns:
                if col.startswith(prefix) and col.endswith("_mean"):
                    try:
                        depth_str = col[len(prefix) : -len("_mean")]
                        depths.append(int(depth_str))
                    except ValueError:
                        continue
            depths = sorted(set(depths))
            if not depths:
                continue

            cmap = plt.get_cmap("tab10")
            depth_colors = {d: cmap(d / 9.0) for d in depths}

            fig, axes = plt.subplots(1, 3, figsize=(21, 6))
            fig.suptitle(
                f"{group_info['title']}: {label} vs {var_name} by Layer",
                fontsize=16,
            )

            depths.reverse()
            for idx, category in enumerate(categories):
                ax: plt.Axes = axes[idx]
                category_data = valid_data[valid_data["network_category"] == category]

                if category_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {category} data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(category.replace("_", " ").title())
                    continue

                for depth in depths:
                    depth_col = f"layer_mi_{base_metric}_depth{depth}_mean"
                    if depth_col not in category_data.columns:
                        continue

                    sorted_data = category_data.sort_values(sweep_var)
                    plot_data = sorted_data[[depth_col, sweep_var]].dropna()
                    if plot_data.empty:
                        continue

                    try:
                        y_vals = pd.to_numeric(
                            plot_data[depth_col], errors="coerce"
                        ).dropna()
                        x_vals = plot_data.loc[y_vals.index, sweep_var]
                        if len(y_vals) > 0:
                            ax.plot(
                                x_vals,
                                y_vals,
                                marker="o",
                                label=f"Layer {depth}",
                                color=depth_colors[depth],
                                alpha=0.8,
                                linewidth=2,
                            )
                    except Exception as e:
                        logger.warning(
                            "Could not plot layer %s for %s: %s", depth, category, e
                        )
                        continue

                self.setup_axes_style(
                    ax=ax,
                    xlabel=var_name,
                    ylabel=group_info["ylabel"],
                    title=category.replace("_", " ").title(),
                    use_log_x=logscale,
                    grid=True,
                )
                ax.legend()

            fig.tight_layout()
            plot_path = (
                plots_dir / f"conditional_info_layer_{base_metric}_{sweep_var}.png"
            )
            plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _plot_new_style_coinfo_by_layer(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        plots_dir: Path,
    ) -> list[Path]:
        """
        Ablation-style figures (1x2 shunting vs additive): one PNG per pair (E,I), (E,V), (I,V).

        Coinformation / interaction information with class C:
            I(X;Y;C) = I(X;Y) - I(X;Y|C)
        computed per layer from layer_mi component minus layer_mi conditional columns.
        """
        plot_paths: list[Path] = []
        plots_dir.mkdir(parents=True, exist_ok=True)

        comp_group = self.METRIC_GROUPS["component_information"]
        cond_group = self.METRIC_GROUPS["conditional_information"]

        if (
            sweep_var not in valid_data.columns
            or "network_category" not in valid_data.columns
        ):
            return plot_paths

        if (
            not valid_data["network_category"]
            .isin(["dendritic_shunting", "dendritic_additive", "dendritic_signed"])
            .any()
        ):
            return plot_paths

        categories = ["dendritic_shunting", "dendritic_additive", "dendritic_signed"]
        ylabel = "Interaction information (bits)"

        triples = list(
            zip(
                comp_group["metrics"],
                cond_group["metrics"],
                ["I(E;I;C)", "I(E;V;C)", "I(I;V;C)"],
                comp_group["labels"],
                cond_group["labels"],
            )
        )

        for mi_comp, mi_cond, coinfo_label, _lab_comp, _lab_cond in triples:
            base_comp = mi_comp.replace("mi_", "")
            base_cond = mi_cond.replace("mi_", "")

            prefix_comp = f"layer_mi_{base_comp}_depth"
            depths: list[int] = []
            for col in valid_data.columns:
                if col.startswith(prefix_comp) and col.endswith("_mean"):
                    try:
                        depth_str = col[len(prefix_comp) : -len("_mean")]
                        d = int(depth_str)
                        cond_col = f"layer_mi_{base_cond}_depth{d}_mean"
                        if cond_col in valid_data.columns:
                            depths.append(d)
                    except ValueError:
                        continue
            depths = sorted(set(depths))
            if not depths:
                continue

            cmap = plt.get_cmap("tab10")
            depth_colors = {d: cmap(d / 9.0) for d in depths}

            fig, axes = plt.subplots(1, 3, figsize=(21, 6))
            fig.suptitle(
                f"Interaction Information: {coinfo_label} vs " f"{var_name} by Layer",
                fontsize=16,
            )

            depths.reverse()
            for idx, category in enumerate(categories):
                ax: plt.Axes = axes[idx]
                category_data = valid_data[valid_data["network_category"] == category]

                if category_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No {category} data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(category.replace("_", " ").title())
                    continue

                for depth in depths:
                    comp_col = f"layer_mi_{base_comp}_depth{depth}_mean"
                    cond_col = f"layer_mi_{base_cond}_depth{depth}_mean"
                    if comp_col not in category_data.columns:
                        continue
                    if cond_col not in category_data.columns:
                        continue

                    sorted_data = category_data.sort_values(sweep_var)
                    plot_data = sorted_data[[comp_col, cond_col, sweep_var]].dropna()
                    if plot_data.empty:
                        continue

                    try:
                        y_comp = pd.to_numeric(plot_data[comp_col], errors="coerce")
                        y_cond = pd.to_numeric(plot_data[cond_col], errors="coerce")
                        valid_idx = y_comp.notna() & y_cond.notna()
                        if not valid_idx.any():
                            continue
                        y_vals = y_comp[valid_idx] - y_cond[valid_idx]
                        x_vals = plot_data.loc[y_vals.index, sweep_var]
                        if len(y_vals) > 0:
                            ax.plot(
                                x_vals,
                                y_vals,
                                marker="o",
                                label=f"Layer {depth}",
                                color=depth_colors[depth],
                                alpha=0.8,
                                linewidth=2,
                            )
                    except Exception as e:
                        logger.warning(
                            "Could not plot coinfo layer %s for %s: %s",
                            depth,
                            category,
                            e,
                        )
                        continue

                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
                self.setup_axes_style(
                    ax=ax,
                    xlabel=var_name,
                    ylabel=ylabel,
                    title=category.replace("_", " ").title(),
                    use_log_x=logscale,
                    grid=True,
                )
                ax.legend()

            fig.tight_layout()
            safe_pair = f"{base_comp}_C"
            plot_path = plots_dir / f"coinfo_layer_{safe_pair}_{sweep_var}.png"
            plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _create_information_heatmaps(
        self, data: pd.DataFrame, output_dir: Path, colx: str, coly: str
    ) -> list[Path]:
        plot_paths = []

        # Generate aggregate heatmaps for each metric group
        plot_paths.extend(
            self.plot_metric_groups_aggregate_heatmaps(data, output_dir, colx, coly)
        )

        # Generate layerwise heatmaps for each metric
        plot_paths.extend(
            self.plot_metric_layerwise_heatmaps(data, output_dir, colx, coly)
        )

        return plot_paths

    def plot_metric_groups_aggregate_heatmaps(
        self, data: pd.DataFrame, output_dir: Path, colx: str, coly: str
    ) -> list[Path]:
        """
        Generate heatmap plots for each metric group.

        Creates one figure per metric group with:
        - 2 columns (shunting vs non-shunting)
        - N rows where N = number of metrics in the group

        Each subplot shows a heatmap with colx on x-axis and coly on y-axis.

        Args:
            data: Aggregated data
            output_dir: Output directory
            colx: Column name for x-axis
            coly: Column name for y-axis

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Validate required columns
        if colx not in data.columns or coly not in data.columns:
            logger.warning(f"Missing required columns: {colx} or {coly}")
            return plot_paths

        if "network_category" not in data.columns:
            logger.warning("Missing 'network_category' column")
            return plot_paths

        # Generate one figure per metric group
        for group_name, group_info in self.METRIC_GROUPS.items():
            try:
                metrics = group_info["metrics"]
                labels = group_info["labels"]

                # Find available metrics
                available_metrics = []
                available_labels = []

                for metric, label in zip(metrics, labels):
                    mean_col = f"{metric}_mean"
                    if mean_col in data.columns:
                        available_metrics.append(metric)
                        available_labels.append(label)

                if not available_metrics:
                    logger.info(f"No available metrics for {group_name}, skipping")
                    continue

                # Create figure with nrows = number of metrics, ncols = 2
                nrows = len(available_metrics)
                fig, axes = plt.subplots(
                    nrows, 2, figsize=(18, 6 * nrows), sharex=True, sharey=True
                )

                # Handle single row case
                if nrows == 1:
                    axes = axes.reshape(1, -1)

                # Collect all values for consistent color scaling
                all_values = []
                plot_data = {}

                for use_shunting in [True, False]:
                    category = (
                        "dendritic_shunting" if use_shunting else "dendritic_additive"
                    )
                    cat_data = data[data["network_category"] == category]
                    plot_data[use_shunting] = {}

                    if not cat_data.empty:
                        for metric in available_metrics:
                            mean_col = f"{metric}_mean"
                            values = cat_data[mean_col].dropna()
                            all_values.extend(values.tolist())

                # Calculate global vmin/vmax for consistent scaling
                if all_values:
                    vmin, vmax = min(all_values), max(all_values)
                else:
                    vmin, vmax = 0, 1

                # Create heatmaps for each metric and shunting condition
                for metric_idx, (metric, label) in enumerate(
                    zip(available_metrics, available_labels)
                ):
                    mean_col = f"{metric}_mean"

                    for col_idx, (use_shunting, title_suffix) in enumerate(
                        [
                            (True, "Shunting"),
                            (False, "Non-Shunting"),
                        ]
                    ):
                        ax = axes[metric_idx, col_idx]
                        category = (
                            "dendritic_shunting"
                            if use_shunting
                            else "dendritic_additive"
                        )
                        cat_data = data[data["network_category"] == category]

                        if cat_data.empty or mean_col not in cat_data.columns:
                            ax.text(
                                0.5,
                                0.5,
                                f"No {category} data",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(f"{label} - {title_suffix}")
                            continue

                        # Create pivot table for heatmap
                        try:
                            pivot = cat_data.pivot_table(
                                values=mean_col,
                                index=coly,
                                columns=colx,
                                aggfunc="mean",
                            )

                            if pivot.empty:
                                ax.text(
                                    0.5,
                                    0.5,
                                    "No data to pivot",
                                    ha="center",
                                    va="center",
                                    transform=ax.transAxes,
                                )
                                ax.set_title(f"{label} - {title_suffix}")
                                continue

                            # Create heatmap
                            sns.heatmap(
                                pivot,
                                cmap="viridis",
                                annot=True,
                                fmt=".3f",
                                cbar_kws={"label": group_info["ylabel"]},
                                ax=ax,
                                vmin=vmin,
                                vmax=vmax,
                            )
                            # Invert y-axis to place origin at lower left
                            ax.invert_yaxis()

                            ax.set_xlabel(colx, fontsize=12)
                            if col_idx == 0:  # Only label y-axis on left column
                                ax.set_ylabel(coly, fontsize=12)
                            else:
                                ax.set_ylabel("")
                            ax.set_title(f"{label} - {title_suffix}", fontsize=12)

                        except Exception as e:
                            logger.error(
                                f"Error creating heatmap for {metric} ({title_suffix}): {e}"
                            )
                            ax.text(
                                0.5,
                                0.5,
                                f"Error: {str(e)[:50]}",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(f"{label} - {title_suffix}")

                fig.suptitle(f"{group_info['title']} Heatmaps", fontsize=16, y=0.995)
                fig.tight_layout(rect=[0, 0, 1, 0.98])

                plot_path = output_dir / f"{group_name}_heatmaps.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating {group_name} heatmaps: {e}")
                traceback.print_exc()

        logger.info(f"Generated {len(plot_paths)} metric group heatmap figures")
        return plot_paths

    def plot_metric_layerwise_heatmaps(
        self, data: pd.DataFrame, output_dir: Path, colx: str, coly: str
    ) -> list[Path]:
        """
        Generate layerwise heatmap plots for each metric.

        Creates one figure per metric with:
        - 2 columns (shunting vs non-shunting)
        - N rows where N = number of depths/layers found in the data

        Each subplot shows a heatmap with colx on x-axis and coly on y-axis.

        Args:
            data: Aggregated data
            output_dir: Output directory
            colx: Column name for x-axis
            coly: Column name for y-axis

        Returns:
            List of plot paths
        """
        plot_paths = []

        # Validate required columns
        if colx not in data.columns or coly not in data.columns:
            logger.warning(f"Missing required columns: {colx} or {coly}")
            return plot_paths

        if "network_category" not in data.columns:
            logger.warning("Missing 'network_category' column")
            return plot_paths

        # Find all layer MI columns
        layer_cols = [
            col
            for col in data.columns
            if "layer_mi_" in col and "_depth" in col and col.endswith("_mean")
        ]

        if not layer_cols:
            logger.info("No layer MI columns found for heatmaps")
            return plot_paths

        # Extract all unique depths
        depths = set()
        for col in layer_cols:
            try:
                depth = int(col.split("_depth")[1].split("_mean")[0])
                depths.add(depth)
            except (ValueError, IndexError):
                pass

        depths = sorted(depths)

        if not depths:
            logger.info("No valid depths found in layer columns")
            return plot_paths

        logger.info(f"Found {len(depths)} depths: {depths}")

        # Collect all metrics from all metric groups
        all_metrics = []
        metric_to_group_info = {}

        for _group_name, group_info in self.METRIC_GROUPS.items():
            for metric, label in zip(group_info["metrics"], group_info["labels"]):
                # Get base metric name (e.g., "E_C" from "mi_E_C")
                base_metric = metric.replace("mi_", "")

                # Check if we have layer data for this metric
                has_layer_data = any(
                    f"layer_mi_{base_metric}_depth{depth}_mean" in data.columns
                    for depth in depths
                )

                if has_layer_data:
                    if metric not in all_metrics:
                        all_metrics.append(metric)
                        metric_to_group_info[metric] = {
                            "base_metric": base_metric,
                            "label": label,
                            "group_title": group_info["title"],
                            "ylabel": group_info["ylabel"],
                        }

        if not all_metrics:
            logger.info("No metrics with layer data found")
            return plot_paths

        # Generate one figure per metric
        for metric in all_metrics:
            try:
                metric_info = metric_to_group_info[metric]
                base_metric = metric_info["base_metric"]
                label = metric_info["label"]
                group_title = metric_info["group_title"]
                ylabel = metric_info["ylabel"]

                # Find which depths have data for this metric
                available_depths = []
                for depth in depths:
                    layer_col = f"layer_mi_{base_metric}_depth{depth}_mean"
                    if layer_col in data.columns:
                        available_depths.append(depth)

                if not available_depths:
                    logger.info(f"No layer data available for metric {metric}")
                    continue

                nrows = len(available_depths)
                fig, axes = plt.subplots(
                    nrows, 2, figsize=(18, 6 * nrows), sharex=True, sharey=True
                )

                # Handle single row case
                if nrows == 1:
                    axes = axes.reshape(1, -1)

                # Collect all values for consistent color scaling across all depths
                all_values = []

                for use_shunting in [True, False]:
                    category = (
                        "dendritic_shunting" if use_shunting else "dendritic_additive"
                    )
                    cat_data = data[data["network_category"] == category]

                    if not cat_data.empty:
                        for depth in available_depths:
                            layer_col = f"layer_mi_{base_metric}_depth{depth}_mean"
                            if layer_col in cat_data.columns:
                                values = cat_data[layer_col].dropna()
                                all_values.extend(values.tolist())

                # Calculate global vmin/vmax for consistent scaling
                if all_values:
                    vmin, vmax = min(all_values), max(all_values)
                else:
                    vmin, vmax = 0, 1

                # Create heatmaps for each depth and shunting condition
                for row_idx, depth in enumerate(available_depths):
                    layer_col = f"layer_mi_{base_metric}_depth{depth}_mean"

                    for col_idx, (use_shunting, title_suffix) in enumerate(
                        [
                            (True, "Shunting"),
                            (False, "Non-Shunting"),
                        ]
                    ):
                        ax = axes[row_idx, col_idx]
                        category = (
                            "dendritic_shunting"
                            if use_shunting
                            else "dendritic_additive"
                        )
                        cat_data = data[data["network_category"] == category]

                        if cat_data.empty or layer_col not in cat_data.columns:
                            ax.text(
                                0.5,
                                0.5,
                                f"No {category} data",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(f"Layer {depth} - {title_suffix}")
                            continue

                        # Create pivot table for heatmap
                        try:
                            pivot = cat_data.pivot_table(
                                values=layer_col,
                                index=coly,
                                columns=colx,
                                aggfunc="mean",
                            )

                            if pivot.empty:
                                ax.text(
                                    0.5,
                                    0.5,
                                    "No data to pivot",
                                    ha="center",
                                    va="center",
                                    transform=ax.transAxes,
                                )
                                ax.set_title(f"Layer {depth} - {title_suffix}")
                                continue

                            # Create heatmap
                            sns.heatmap(
                                pivot,
                                cmap="viridis",
                                annot=True,
                                fmt=".3f",
                                cbar_kws={"label": ylabel},
                                ax=ax,
                                vmin=vmin,
                                vmax=vmax,
                            )
                            # Invert y-axis to place origin at lower left
                            ax.invert_yaxis()

                            ax.set_xlabel(colx, fontsize=12)
                            if col_idx == 0:  # Only label y-axis on left column
                                ax.set_ylabel(f"{coly}\n(Layer {depth})", fontsize=12)
                            else:
                                ax.set_ylabel("")
                            ax.set_title(f"Layer {depth} - {title_suffix}", fontsize=12)

                        except Exception as e:
                            logger.error(
                                f"Error creating heatmap for {metric} layer {depth} ({title_suffix}): {e}"
                            )
                            ax.text(
                                0.5,
                                0.5,
                                f"Error: {str(e)[:50]}",
                                ha="center",
                                va="center",
                                transform=ax.transAxes,
                            )
                            ax.set_title(f"Layer {depth} - {title_suffix}")

                fig.suptitle(
                    f"{label} ({group_title}) - Layerwise Heatmaps",
                    fontsize=16,
                    y=0.995,
                )
                fig.tight_layout(rect=[0, 0, 1, 0.98])

                # Create safe filename from metric name
                safe_metric_name = metric.replace("mi_", "").replace("_", "-")
                plot_path = output_dir / f"{safe_metric_name}_layerwise_heatmaps.png"
                plot_paths.append(self.save_figure(fig, plot_path))

            except Exception as e:
                logger.error(f"Error generating layerwise heatmaps for {metric}: {e}")
                traceback.print_exc()

        logger.info(f"Generated {len(plot_paths)} layerwise heatmap figures")
        return plot_paths

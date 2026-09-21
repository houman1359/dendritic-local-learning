"""Multiplicative gain analysis plotters."""

import logging
import traceback
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)

METRIC_MAPPING = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "categorical_loglikelihood": "Log Likelihood",
    "mse": "Mean Squared Error",
    "cosine_similarity": "Cosine Similarity",
}

SCORE_MAPPING = {
    "id": "ID",
    "ood_lower": "OOD Below Train",
    "ood_upper": "OOD Above Train",
}

CATEGORY_MAP = {
    "dendritic_shunting": "Dendritic Shunting",
    "dendritic_additive": "Dendritic Additive",
    "dendritic_normalized_additive": "Dendritic Normalized Additive",
    "flat_shunting": "Flat Shunting",
    "flat_additive": "Flat Additive",
    "flat_normalized_additive": "Flat Normalized Additive",
    "dendritic_signed": "Dendritic Signed",
    "flat_signed": "Flat Signed",
    "dendritic_mlp": "Dendritic MLP",
    "flat_mlp": "Flat MLP",
    "ss_mlp": "Sparse Structured MLP",
    "ss_mlp_flat": "Sparse Structured MLP Flat",
    "active_param_mlp": "Matched Active Param MLP",
    "total_param_mlp": "Matched Total Param MLP",
}


class MultiplicativeGainPlotter(BasePlotter):
    """Generates multiplicative gain analysis plots."""

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all multiplicative gain analysis plots.
        """

        plot_paths = []
        gain_dir = output_dir / "multiplicative_gain"
        gain_dir.mkdir(parents=True, exist_ok=True)

        # Check for multiplicative gain data
        gain_cols = [col for col in data.columns if "mult_gain_" in col]
        if not gain_cols:
            logger.info(
                "No multiplicative gain data found, skipping multiplicative gain plots"
            )
            return plot_paths

        # Generate multiplicative gain analysis plots
        plot_paths.extend(self._generate_gain_plots(data, gain_dir))

        return plot_paths

    def _generate_gain_plots(self, data: pd.DataFrame, plots_dir: Path) -> list[Path]:
        plot_paths = []

        sweep_types = self.detect_sweep_types(data)

        if "ei" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_1d_sweep(
                        data=data,
                        sweep_var="ei_ratio",
                        var_name="E/I Ratio",
                        plots_dir=plots_dir,
                        logscale=True,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating E/I gain plot: {e}")
                traceback.print_exc()

        if "stim_duration" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_1d_sweep(
                        data=data,
                        sweep_var="stimulus_duration",
                        var_name="Stimulus Duration",
                        plots_dir=plots_dir,
                        logscale=True,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating stimulus duration gain plot: {e}")
                traceback.print_exc()

        if "max_gf" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_1d_sweep(
                        data=data,
                        sweep_var="max_gain_factor",
                        var_name="Maximum Gain",
                        plots_dir=plots_dir,
                        logscale=True,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating max gain factor gain plot: {e}")
                traceback.print_exc()

        if "fixed_gf" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_1d_sweep(
                        data=data,
                        sweep_var="fixed_gain_factor",
                        var_name="Fixed Gain Factor",
                        plots_dir=plots_dir,
                        logscale=True,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating fixed gain factor gain plot: {e}")
                traceback.print_exc()

        if "nparams" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_1d_sweep(
                        data=data,
                        sweep_var="nparams",
                        var_name="Number of Parameters",
                        plots_dir=plots_dir,
                        logscale=True,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating nparams gain plot: {e}")
                traceback.print_exc()

        if "stoch1" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_2d_sweep(
                        data,
                        "stimulus_duration",
                        "max_gain_factor",
                        plots_dir,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating stochastic gain plot: {e}")
                traceback.print_exc()

        if "stoch2" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_2d_sweep(
                        data,
                        "stimulus_duration",
                        "max_gain_tau_ratio",
                        plots_dir,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating stochastic GV gain plot: {e}")
                traceback.print_exc()

        if "network_type_comp" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_gain_plots_network_type_comparison(data, plots_dir)
                )
            except Exception as e:
                logger.error("Error generating network-type gain plot: %s", e)
                traceback.print_exc()

        logger.info(f"Created {len(plot_paths)} multiplicative gain analysis plots")
        return plot_paths

    def _extract_metrics_and_gain_factors(
        self,
        gain_cols: list[str],
    ) -> tuple[list[str], list[float]]:
        metrics: list[str] = []
        gain_factors: list[float] = []

        for col in gain_cols:
            if "_mean" not in col or "_gf" not in col:
                continue

            parts = col.replace("mult_gain_", "").replace("_mean", "").split("_gf")
            metric = parts[0]
            try:
                gain_factor = float(parts[-1])
            except ValueError:
                continue

            if metric not in metrics:
                metrics.append(metric)
            if gain_factor not in gain_factors:
                gain_factors.append(gain_factor)

        gain_factors.sort()
        return metrics, gain_factors

    def _detect_generalization_score_types(self, gain_cols: list[str]) -> list[str]:
        detected = []
        for score_type in ["id", "ood_lower", "ood_upper"]:
            suffix = f"_{score_type}"
            if any(col.replace("_mean", "").endswith(suffix) for col in gain_cols):
                detected.append(score_type)
        return detected

    def _create_gain_plots_1d_sweep(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        plots_dir: Path,
        logscale: bool,
    ) -> list[Path]:
        """Create multiplicative gain vs sweep_var plots"""
        plot_paths = []

        gain_cols: list[str] = [col for col in data.columns if "mult_gain_" in col]

        metrics, gain_factors = self._extract_metrics_and_gain_factors(gain_cols)
        score_types = self._detect_generalization_score_types(gain_cols)

        plot_paths.extend(
            self._plot_gain_1d_sweep_raw_scores_heatmap(
                data=data,
                sweep_var=sweep_var,
                var_name=var_name,
                plots_dir=plots_dir,
                metrics=metrics,
                gain_factors=gain_factors,
            )
        )
        plot_paths.extend(
            self._plot_gain_generalization_scores_1d_sweep(
                data=data,
                sweep_var=sweep_var,
                var_name=var_name,
                plots_dir=plots_dir,
                metrics=metrics,
                score_types=score_types,
                logscale=logscale,
            )
        )

        return plot_paths

    def _plot_gain_generalization_scores_1d_sweep(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        plots_dir: Path,
        metrics: list[str],
        score_types: list[str],
        logscale: bool,
    ) -> list[Path]:
        """Create ID/OOD gain-generalization score plots."""
        plot_paths = []
        if not score_types:
            return plot_paths

        for metric in metrics:
            available_scores = [
                score_type
                for score_type in score_types
                if (
                    f"mult_gain_{metric}_{score_type}_mean" in data.columns
                    or f"mult_gain_{metric}_{score_type}" in data.columns
                )
            ]
            if not available_scores:
                continue

            fig, axes = plt.subplots(
                1,
                len(available_scores),
                figsize=(8 * len(available_scores), 6),
                squeeze=False,
            )

            for ax, score_type in zip(axes[0], available_scores):
                self.plot_by_category(
                    ax=ax,
                    data=data,
                    x_col=sweep_var,
                    y_col=f"mult_gain_{metric}_{score_type}",
                    category_col="network_category",
                    show_error=True,
                )
                self.setup_axes_style(
                    ax=ax,
                    xlabel=var_name,
                    ylabel=f"{METRIC_MAPPING[metric]} Expectation",
                    title=SCORE_MAPPING[score_type],
                    use_log_x=logscale,
                    grid=True,
                )
                ax.legend(fontsize=10)

            fig.tight_layout()
            plot_path = (
                plots_dir / f"gain_{metric}_vs_{sweep_var}_generalization_scores.png"
            )
            plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _plot_gain_1d_sweep_raw_scores_heatmap(
        self,
        data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        plots_dir: Path,
        metrics: list[str],
        gain_factors: list[float],
    ) -> list[Path]:
        """Create multiplicative gain performance vs sweep_var plots"""
        plot_paths = []

        if not gain_factors:
            return plot_paths

        # Get unique values of sweep_var and sort them
        sweep_values = sorted(data[sweep_var].unique())

        possible_categories = [
            "dendritic_shunting",
            "flat_shunting",
            "dendritic_additive",
            "flat_additive",
            "dendritic_signed",
            "flat_signed",
            "dendritic_mlp",
            "flat_mlp",
            "point_mlp",
            "ss_mlp",
            "ss_mlp_flat",
            "active_param_mlp",
            "total_param_mlp",
        ]
        categories = [
            cat
            for cat in possible_categories
            if cat in data["network_category"].unique()
        ]

        for metric in metrics:
            heatmap_dict = {}

            for category in categories:
                cat_data = data[data["network_category"] == category]

                # Create a 2D array for the heatmap: rows = gain_factors, columns = sweep_values
                heatmap_data = np.zeros((len(gain_factors), len(sweep_values)))

                # Fill the heatmap data
                for i, gf in enumerate(gain_factors):
                    for j, sweep_val in enumerate(sweep_values):
                        # Filter data for this sweep value
                        filtered_data = cat_data[cat_data[sweep_var] == sweep_val]

                        # Extract the mean value for this gain factor
                        if len(filtered_data) > 0:
                            mean_val = filtered_data[
                                f"mult_gain_{metric}_gf{gf}_mean"
                            ].iloc[0]
                            heatmap_data[i, j] = mean_val
                        else:
                            heatmap_data[i, j] = np.nan

                heatmap_dict[category] = heatmap_data

            # Create the heatmap
            fig, axes = plt.subplots(
                len(categories) - 1,
                3,
                figsize=(24, 6 * (len(categories) - 1) + 4),
                squeeze=False,
            )

            def create_heatmap(
                ax: plt.Axes,
                data: np.ndarray,
                cmap: str,
                cbar_label: str,
                ax_title: str,
                vmin: float,
                vmax: float,
                center: Optional[float] = None,
            ):
                sns.heatmap(
                    data,
                    xticklabels=sweep_values,
                    yticklabels=[f"{gf:.3f}" for gf in gain_factors],
                    cmap=cmap,
                    annot=True,
                    fmt=".3f",
                    cbar_kws={"label": cbar_label},
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    center=center,
                )
                ax.invert_yaxis()
                ax.set_xlabel(var_name, fontsize=12)
                ax.set_ylabel("Gain Factor", fontsize=12)
                ax.set_title(ax_title, fontsize=14)

            for i in range(len(categories) - 1):
                vmin = min(
                    np.min(heatmap_dict["dendritic_shunting"]),
                    np.min(heatmap_dict[categories[i + 1]]),
                )
                vmax = max(
                    np.max(heatmap_dict["dendritic_shunting"]),
                    np.max(heatmap_dict[categories[i + 1]]),
                )
                create_heatmap(
                    ax=axes[i, 0],
                    data=heatmap_dict["dendritic_shunting"],
                    cmap="viridis",
                    cbar_label=METRIC_MAPPING[metric],
                    ax_title="Shunting",
                    vmin=vmin,
                    vmax=vmax,
                )
                create_heatmap(
                    ax=axes[i, 2],
                    data=heatmap_dict[categories[i + 1]],
                    cmap="viridis",
                    cbar_label=METRIC_MAPPING[metric],
                    ax_title=CATEGORY_MAP[categories[i + 1]],
                    vmin=vmin,
                    vmax=vmax,
                )

                diff_data = (
                    heatmap_dict["dendritic_shunting"] - heatmap_dict[categories[i + 1]]
                )
                max_abs = np.max(np.abs(diff_data))
                create_heatmap(
                    ax=axes[i, 1],
                    data=diff_data,
                    cmap="seismic_r",
                    cbar_label=f"{METRIC_MAPPING[metric]} Difference",
                    ax_title=f"Difference (Shunting - {CATEGORY_MAP[categories[i+1]]})",
                    vmin=-max_abs,
                    vmax=max_abs,
                )

            fig.tight_layout()

            plot_path = plots_dir / f"gain_{metric}_vs_{sweep_var}_heatmap.png"
            plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _create_gain_plots_2d_sweep(
        self,
        data: pd.DataFrame,
        sweep_var1: str,
        sweep_var2: str,
        plots_dir: Path,
    ) -> list[Path]:
        """Create multiplicative gain vs 2D sweep plots"""
        plot_paths = []

        return plot_paths

    def _create_gain_plots_network_type_comparison(
        self,
        data: pd.DataFrame,
        plots_dir: Path,
    ) -> list[Path]:
        """Create multiplicative gain shunting vs non-shunting comparison plots"""
        plot_paths = []

        gain_cols: list[str] = [col for col in data.columns if "mult_gain_" in col]

        metrics, gain_factors = self._extract_metrics_and_gain_factors(gain_cols)
        for metric in metrics:
            plot_paths.extend(
                self._plot_gain_performance_network_type_comparison(
                    data=data,
                    plots_dir=plots_dir,
                    metric=metric,
                    gain_factors=gain_factors,
                )
            )

        return plot_paths

    def _plot_gain_performance_network_type_comparison(
        self,
        data: pd.DataFrame,
        plots_dir: Path,
        metric: str,
        gain_factors: list[float],
    ) -> list[Path]:
        """Create multiplicative gain performance network type comparison plots"""
        plot_paths = []

        possible_categories = [
            "dendritic_shunting",
            "flat_shunting",
            "dendritic_additive",
            "flat_additive",
            "dendritic_signed",
            "flat_signed",
            "dendritic_mlp",
            "flat_mlp",
            "point_mlp",
            "ss_mlp",
            "ss_mlp_flat",
            "active_param_mlp",
            "total_param_mlp",
        ]
        categories = [
            cat
            for cat in possible_categories
            if cat in data["network_category"].unique()
        ]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle(
            f"Multiplicative Gain: {METRIC_MAPPING[metric]} vs Gain Factor",
            fontsize=16,
        )

        for category in categories:
            cat_data = data[data["network_category"] == category]

            mean_data = []
            std_data = []

            for gf in gain_factors:
                mean_data.append(cat_data[f"mult_gain_{metric}_gf{gf}_mean"].iloc[0])
                std_data.append(cat_data[f"mult_gain_{metric}_gf{gf}_std"].iloc[0])
            mean_data = np.array(mean_data)
            std_data = np.array(std_data)

            color = self.category_colors.get(category, "black")
            marker = self.category_markers.get(category, "o")
            linestyle = self.category_linestyles.get(category, "-")

            ax.plot(
                gain_factors,
                mean_data,
                color=color,
                marker=marker,
                label=category,
                linestyle=linestyle,
            )
            ax.fill_between(
                gain_factors,
                mean_data - std_data,
                mean_data + std_data,
                color=color,
                alpha=0.2,
            )

        ax.set_xscale("log")
        self.setup_axes_style(
            ax=ax, xlabel="Gain Factor", ylabel=METRIC_MAPPING[metric], grid=True
        )

        ax.legend(fontsize=10)
        fig.tight_layout()

        plot_path = plots_dir / f"gain_{metric}_vs_network_type.png"
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

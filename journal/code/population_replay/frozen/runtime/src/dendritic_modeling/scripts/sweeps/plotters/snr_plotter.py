"""SNR analysis plotters."""

import itertools
import logging
import traceback
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .base_plotter import BasePlotter

logger = logging.getLogger(__name__)


SNR_PREFIX_MAP = {
    "exc_inputs": "Excitatory Inputs",
    "inh_inputs": "Inhibitory Inputs",
    "upstream_inputs": "Upstream Inputs",
    "network_exc_inputs": "Network Excitatory Inputs",
    "network_inh_outputs": "Network Inhibitory Outputs",
    "exc": "Excitation",
    "inh": "Inhibition",
    "upstream": "Upstream",
    "vout": "Vout",
}
METRIC_MAPPING = {
    "d_max": "Max Discriminability",
    "d_total": "Total Discriminability",
    "n_sig_dirs": "Number of Significant Directions",
    "ratio_sig_dirs": "Ratio of Significant Directions",
}


class SNRPlotter(BasePlotter):
    """Generates SNR analysis plots."""

    def generate_all(self, data: pd.DataFrame, output_dir: Path) -> list[Path]:
        """
        Generate all SNR analysis plots.

        Args:
            data: Aggregated data
            output_dir: Output directory

        Returns:
            List of plot paths
        """
        plot_paths = []
        snr_dir = output_dir / "snr"
        snr_dir.mkdir(parents=True, exist_ok=True)

        # Check for SNR data
        snr_cols = [col for col in data.columns if "snr_" in col]
        if not snr_cols:
            logger.info("No SNR data found, skipping SNR plots")
            return plot_paths

        # Generate SNR analysis plots
        plot_paths.extend(self._generate_snr_plots(data, snr_dir))

        return plot_paths

    def _generate_snr_plots(
        self, valid_data: pd.DataFrame, plots_dir: str
    ) -> list[str]:
        """Generate SNR analysis plots"""
        plot_paths = []

        # Find SNR columns (look for both column name patterns)
        input_snr_cols: list[str] = [
            col
            for col in valid_data.columns
            if col.startswith("snr_") and ("inputs" in col or "outputs" in col)
        ]

        act_snr_cols: list[str] = [
            col for col in valid_data.columns if "snr_" in col and "depth" in col
        ]

        if not input_snr_cols and not act_snr_cols:
            print("No SNR columns found")
            return plot_paths

        input_prefixes = []
        input_metrics = []

        for col in input_snr_cols:
            if "std" in col:
                continue

            stem = col.replace("snr_", "").replace("_mean", "")
            for input_metric in METRIC_MAPPING:
                suffix = f"_{input_metric}"
                if not stem.endswith(suffix):
                    continue

                input_prefix = stem[: -len(suffix)]
                if input_prefix not in input_prefixes:
                    input_prefixes.append(input_prefix)
                if input_metric not in input_metrics:
                    input_metrics.append(input_metric)
                break

        act_prefixes = []
        act_metrics = []
        depths = []

        for col in act_snr_cols:
            if "std" not in col:
                parts = (
                    col.replace("snr_", "")
                    .replace("depth", "")
                    .replace("_mean", "")
                    .split("_")
                )
                act_prefix = parts[0]
                act_metric = "_".join(parts[1:-1])
                depth = int(parts[-1])

                if act_prefix not in act_prefixes:
                    act_prefixes.append(act_prefix)
                if act_metric not in act_metrics:
                    act_metrics.append(act_metric)
                if depth not in depths:
                    depths.append(depth)

        depths.sort()

        if (
            not valid_data["network_category"]
            .isin(["dendritic_shunting", "dendritic_additive", "dendritic_signed"])
            .any()
        ):
            print("No supported dendritic data for SNR analysis")
            return plot_paths

        sweep_types = self.detect_sweep_types(valid_data)
        # Generate E/I ratio plots by depth (current approach)
        if "ei" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_snr_plots_1d_sweep(
                        valid_data=valid_data,
                        sweep_var="ei_ratio",
                        var_name="E/I Ratio",
                        logscale=True,
                        output_dir=plots_dir,
                        input_prefixes=input_prefixes,
                        input_metrics=input_metrics,
                        act_prefixes=act_prefixes,
                        act_metrics=act_metrics,
                        depths=depths,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating E/I SNR plot: {e}")
                traceback.print_exc()

        if "stim_duration" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_snr_plots_1d_sweep(
                        valid_data=valid_data,
                        sweep_var="stimulus_duration",
                        var_name="Stimulus Duration",
                        logscale=True,
                        output_dir=plots_dir,
                        input_prefixes=input_prefixes,
                        input_metrics=input_metrics,
                        act_prefixes=act_prefixes,
                        act_metrics=act_metrics,
                        depths=depths,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating stimulus duration SNR plot: {e}")
                traceback.print_exc()

        if "max_gf" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_snr_plots_1d_sweep(
                        valid_data=valid_data,
                        sweep_var="max_gain_factor",
                        var_name="Maximum Gain",
                        logscale=True,
                        output_dir=plots_dir,
                        input_prefixes=input_prefixes,
                        input_metrics=input_metrics,
                        act_prefixes=act_prefixes,
                        act_metrics=act_metrics,
                        depths=depths,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating max gain factor SNR plot: {e}")
                traceback.print_exc()

        if "fixed_gf" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_snr_plots_1d_sweep(
                        valid_data=valid_data,
                        sweep_var="fixed_gain_factor",
                        var_name="Fixed Gain Factor",
                        logscale=True,
                        output_dir=plots_dir,
                        input_prefixes=input_prefixes,
                        input_metrics=input_metrics,
                        act_prefixes=act_prefixes,
                        act_metrics=act_metrics,
                        depths=depths,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating fixed gain factor SNR plot: {e}")
                traceback.print_exc()

        if "nparams" in sweep_types:
            try:
                plot_paths.extend(
                    self._create_snr_plots_1d_sweep(
                        valid_data=valid_data,
                        sweep_var="nparams",
                        var_name="Number of Parameters",
                        logscale=True,
                        output_dir=plots_dir,
                        input_prefixes=input_prefixes,
                        input_metrics=input_metrics,
                        act_prefixes=act_prefixes,
                        act_metrics=act_metrics,
                        depths=depths,
                    )
                )
            except Exception as e:
                logger.error(f"Error generating nparams SNR plot: {e}")
                traceback.print_exc()

        if "stoch1" in sweep_types:
            for act_prefix in act_prefixes:
                try:
                    plot_paths.extend(
                        self._create_snr_heatmaps(
                            valid_data,
                            plots_dir,
                            act_prefix,
                            depths,
                            "stimulus_duration",
                            "max_gain_factor",
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error generating stochastic SNR plot for {act_prefix}: {e}"
                    )
                    traceback.print_exc()

        if "stoch2" in sweep_types:
            for act_prefix in act_prefixes:
                try:
                    plot_paths.extend(
                        self._create_snr_heatmaps(
                            valid_data,
                            plots_dir,
                            act_prefix,
                            depths,
                            "stimulus_duration",
                            "max_gain_tau_ratio",
                        )
                    )
                except Exception as e:
                    logger.error(
                        f"Error generating stochastic GV SNR plot for {act_prefix}: {e}"
                    )
                    traceback.print_exc()

        logger.info(f"Created {len(plot_paths)} SNR analysis plots")
        return plot_paths

    def _create_snr_plots_1d_sweep(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        output_dir: str,
        input_prefixes: list[str],
        input_metrics: list[str],
        act_prefixes: list[str],
        act_metrics: list[str],
        depths: list[int],
    ) -> list[str]:
        """Create SNR vs sweep_var plots"""
        plot_paths = []

        for input_metric in input_metrics:
            plot_paths.extend(
                self._create_input_snr_plots_1d_sweep(
                    valid_data=valid_data,
                    sweep_var=sweep_var,
                    var_name=var_name,
                    logscale=logscale,
                    output_dir=output_dir,
                    input_prefixes=input_prefixes,
                    input_metric=input_metric,
                )
            )

        for act_prefix, act_metric in itertools.product(act_prefixes, act_metrics):
            plot_paths.extend(
                self._create_activation_snr_plots_1d_sweep(
                    valid_data=valid_data,
                    sweep_var=sweep_var,
                    var_name=var_name,
                    logscale=logscale,
                    output_dir=output_dir,
                    act_prefix=act_prefix,
                    act_metric=act_metric,
                    depths=depths,
                )
            )

        for act_metric in act_metrics:
            plot_paths.extend(
                self._create_soma_snr_plots_1d_sweep(
                    valid_data=valid_data,
                    sweep_var=sweep_var,
                    var_name=var_name,
                    logscale=logscale,
                    output_dir=output_dir,
                    act_metric=act_metric,
                )
            )

        return plot_paths

    def _create_input_snr_plots_1d_sweep(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        output_dir: str,
        input_prefixes: list[str],
        input_metric: str,
    ) -> list[str]:
        """Create input SNR vs sweep_var plots"""
        plot_paths = []

        # Create separate plots for each network category
        categories = [
            "dendritic_shunting",
            "flat_shunting",
            "dendritic_additive",
            "flat_additive",
            "dendritic_signed",
            "flat_signed",
            "dendritic_mlp",
            "flat_mlp",
        ]

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(
            "Input Disciminability: " + f"{METRIC_MAPPING[input_metric]} vs {var_name}",
            fontsize=16,
        )

        def _plot_data(
            category_data: pd.DataFrame,
            category: str,
            input_prefix: str,
            input_metric: str,
            label: str,
            color: str,
            linestyle: str,
            marker: str,
        ):
            field_name = f"{input_prefix}_{input_metric}"
            mean_col = f"snr_{field_name}_mean"
            std_col = f"snr_{field_name}_std"
            if mean_col not in category_data.columns:
                return

            sorted_data = category_data.sort_values(sweep_var)

            # Filter out non-numeric or missing values
            plot_cols = [mean_col, sweep_var]
            if std_col in sorted_data.columns:
                plot_cols.append(std_col)
            plot_data = sorted_data[plot_cols].dropna()
            if plot_data.empty:
                return

            try:
                # Convert to numeric if needed
                mean_vals = pd.to_numeric(plot_data[mean_col], errors="coerce").dropna()
                x_vals = plot_data.loc[mean_vals.index, sweep_var]
                if std_col in plot_data.columns:
                    std_vals = pd.to_numeric(
                        plot_data.loc[mean_vals.index, std_col], errors="coerce"
                    ).fillna(0.0)
                else:
                    std_vals = None

                if len(mean_vals) > 0:
                    ax.errorbar(
                        x_vals,
                        mean_vals,
                        yerr=std_vals,
                        marker=marker,
                        label=label,
                        color=color,
                        linestyle=linestyle,
                        alpha=0.8,
                        linewidth=2,
                        capsize=3,
                    )
            except Exception as e:
                print(
                    f"Warning: Could not plot {input_prefix} for {category}: {e}",
                    flush=True,
                )
                traceback.print_exc()

        for category in categories:
            category_data = valid_data[valid_data["network_category"] == category]
            for input_prefix in input_prefixes:
                prefix_label = SNR_PREFIX_MAP.get(
                    input_prefix, input_prefix.replace("_", " ").title()
                )
                label = f"{category.replace('_', ' ').title()} {prefix_label}"
                color = self.category_colors.get(category, "black")
                linestyle = self.category_linestyles.get(category, "-")
                marker = self.category_markers.get(category, "o")

                if input_prefix in {"exc_inputs", "network_exc_inputs"}:
                    linestyle = "-"
                    marker = "o"
                elif input_prefix in {"inh_inputs", "network_inh_outputs"}:
                    linestyle = "--"
                    marker = "s"

                _plot_data(
                    category_data=category_data,
                    category=category,
                    input_prefix=input_prefix,
                    input_metric=input_metric,
                    label=label,
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                )

        self.setup_axes_style(
            ax=ax,
            xlabel=var_name,
            ylabel=f"{METRIC_MAPPING[input_metric]}",
            use_log_x=logscale,
            grid=True,
        )
        ax.legend()

        fig.tight_layout()

        plot_path = output_dir / "input" / f"input_{input_metric}_vs_{sweep_var}.png"
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _create_activation_snr_plots_1d_sweep(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        output_dir: str,
        act_prefix: str,
        act_metric: str,
        depths: list[int],
    ) -> list[str]:
        """Create SNR vs sweep_var plots by depth"""
        plot_paths = []

        # Create separate plots for each network category
        categories = ["dendritic_shunting", "dendritic_additive", "dendritic_signed"]

        # Create color mapping for depths using a colormap
        cmap = plt.get_cmap("tab10")
        depth_colors = {depth: cmap(depth / 9.0) for depth in depths}

        fig, axes = plt.subplots(1, len(categories), figsize=(7.5 * len(categories), 6))
        fig.suptitle(
            f"{SNR_PREFIX_MAP[act_prefix]} Disciminability: "
            + f"{METRIC_MAPPING[act_metric]} vs {var_name} by Depth",
            fontsize=16,
        )
        depths = deepcopy(depths)
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
                ax.set_title(f"{category.title()}")
                continue

            # Plot lines for each depth
            for depth in depths:
                depth_col = f"snr_{act_prefix}_{act_metric}_depth{depth}_mean"
                if depth_col not in category_data.columns:
                    continue

                # Sort by E/I ratio for proper line plotting
                sorted_data = category_data.sort_values(sweep_var)

                # Filter out non-numeric or missing values
                plot_data = sorted_data[[depth_col, sweep_var]].dropna()
                if plot_data.empty:
                    continue

                try:
                    # Convert to numeric if needed
                    y_vals = pd.to_numeric(
                        plot_data[depth_col], errors="coerce"
                    ).dropna()
                    x_vals = plot_data.loc[y_vals.index, sweep_var]

                    if len(y_vals) > 0:
                        ax.plot(
                            x_vals,
                            y_vals,
                            marker="o",
                            label=f"Depth {depth}",
                            color=depth_colors[depth],
                            alpha=0.8,
                            linewidth=2,
                        )
                except Exception as e:
                    print(f"Warning: Could not plot depth {depth} for {category}: {e}")
                    continue

            self.setup_axes_style(
                ax=ax,
                xlabel=var_name,
                ylabel=f"{SNR_PREFIX_MAP[act_prefix]} {METRIC_MAPPING[act_metric]}",
                title=f"{category.title()}",
                use_log_x=logscale,
                grid=True,
            )
            ax.legend()

        fig.tight_layout()

        plot_path = (
            output_dir
            / f"{act_prefix}_act"
            / f"{act_prefix}_act_{act_metric}_vs_{sweep_var}.png"
        )
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _create_soma_snr_plots_1d_sweep(
        self,
        valid_data: pd.DataFrame,
        sweep_var: str,
        var_name: str,
        logscale: bool,
        output_dir: str,
        act_metric: str,
    ):
        """Create soma SNR vs sweep_var plots"""
        plot_paths = []

        y_col = f"snr_vout_{act_metric}_depth0"

        if f"{y_col}_mean" not in valid_data.columns:
            return plot_paths

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle(
            f"Soma Disciminability: {METRIC_MAPPING[act_metric]} vs {var_name}",
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
            ylabel=f"{METRIC_MAPPING[act_metric]}",
            use_log_x=logscale,
            grid=True,
        )
        ax.legend(fontsize=10)
        fig.tight_layout()

        plot_path = (
            output_dir / "soma_act" / f"soma_act_{act_metric}_vs_{sweep_var}.png"
        )
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

    def _create_snr_heatmaps(
        self,
        valid_data: pd.DataFrame,
        output_dir: str,
        prefix: str,
        depths: list[int],
        colx: str,
        coly: str,
    ) -> list[str]:
        """
        Create SNR heatmaps

        Args:
            valid_data: DataFrame with SNR data
            output_dir: Output directory for plots
            prefix: Prefix of SNR column (e.g., 'exc', 'inh', 'upstream', 'vout')
            depths: List of depths to create heatmaps for
            colx: Column name for x-axis of heatmap
            coly: Column name for y-axis of heatmap
        """
        plot_paths = []

        # Find snr columns for this state variable
        snr_cols_filtered = [
            col
            for col in valid_data.columns
            if f"snr_{prefix}_" in col and "std" not in col
        ]

        if not snr_cols_filtered:
            return plot_paths

        # Validate that colx and coly exist in the data
        if colx not in valid_data.columns:
            logger.warning(f"Column {colx} not found in data")
            return plot_paths
        if coly not in valid_data.columns:
            logger.warning(f"Column {coly} not found in data")
            return plot_paths

        # Create separate plots for each network category
        categories = ["dendritic_shunting", "dendritic_additive", "dendritic_signed"]

        heatmap_metric = "d_total"
        d_prime_sq_str = r"$d_{total}$"

        # Create subplot layout: len(depths) rows x one column per category
        fig, axes = plt.subplots(
            len(depths),
            len(categories),
            figsize=(7.5 * len(categories), 3 + 6 * len(depths)),
        )
        fig.suptitle(
            f"{SNR_PREFIX_MAP[prefix]} Discriminability: {d_prime_sq_str} Heatmap ({colx} vs {coly})",
            fontsize=16,
        )

        # Handle case where there's only one depth (axes becomes 1D)
        if len(depths) == 1:
            axes = axes.reshape(1, -1)

        # Collect all values for consistent color scaling
        all_values = []
        for depth in depths:
            depth_col = f"snr_{prefix}_{heatmap_metric}_depth{depth}_mean"
            if depth_col in valid_data.columns:
                for category in categories:
                    category_data = valid_data[
                        valid_data["network_category"] == category
                    ]
                    if not category_data.empty:
                        values = pd.to_numeric(
                            category_data[depth_col], errors="coerce"
                        ).dropna()
                        all_values.extend(values.tolist())

        # Calculate global vmin/vmax for consistent scaling
        if all_values:
            vmin, vmax = min(all_values), max(all_values)
        else:
            vmin, vmax = 0, 1

        for depth_idx, depth in enumerate(depths):
            depth_col = f"snr_{prefix}_{heatmap_metric}_depth{depth}_mean"

            if depth_col not in valid_data.columns:
                # Fill empty subplots if depth column doesn't exist
                for cat_idx in range(2):
                    ax = axes[depth_idx, cat_idx]
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for depth {depth}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"Depth {depth} - {categories[cat_idx].title()}")
                continue

            for cat_idx, category in enumerate(categories):
                ax = axes[depth_idx, cat_idx]
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
                    ax.set_title(f"Depth {depth} - {category.title()}")
                    continue

                # Prepare data for heatmap
                plot_data = category_data[[depth_col, colx, coly]].dropna()
                if plot_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data for depth {depth}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"Depth {depth} - {category.title()}")
                    continue

                try:
                    # Convert metric values to numeric
                    plot_data[depth_col] = pd.to_numeric(
                        plot_data[depth_col], errors="coerce"
                    )
                    plot_data = plot_data.dropna()

                    if plot_data.empty:
                        ax.text(
                            0.5,
                            0.5,
                            f"No valid data for depth {depth}",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"Depth {depth} - {category.title()}")
                        continue

                    # Create pivot table for heatmap
                    pivot_data = plot_data.pivot_table(
                        values=depth_col,
                        index=coly,
                        columns=colx,
                        aggfunc="mean",
                    )

                    if pivot_data.empty:
                        ax.text(
                            0.5,
                            0.5,
                            f"No data to pivot for depth {depth}",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                        ax.set_title(f"Depth {depth} - {category.title()}")
                        continue

                    # Create heatmap
                    sns.heatmap(
                        pivot_data,
                        cmap="viridis",
                        annot=True,
                        fmt=".3f",
                        cbar_kws={
                            "label": f"{SNR_PREFIX_MAP[prefix]} {d_prime_sq_str}"
                        },
                        ax=ax,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    # Invert y-axis to place origin at lower left
                    ax.invert_yaxis()

                    ax.set_xlabel(colx, fontsize=12)
                    ax.set_ylabel(coly, fontsize=12)
                    ax.set_title(f"Depth {depth} - {category.title()}", fontsize=12)

                except Exception as e:
                    logger.warning(
                        f"Could not create heatmap for depth {depth}, category {category}: {e}"
                    )
                    ax.text(
                        0.5,
                        0.5,
                        f"Error plotting depth {depth}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_title(f"Depth {depth} - {category.title()}")
                    continue

        fig.tight_layout()

        plot_path = output_dir / f"{prefix}_snr_heatmap_{colx}_vs_{coly}.png"
        plot_paths.append(self.save_figure(fig, plot_path))

        return plot_paths

"""Base plotter class with common utilities."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter

from ..utils.metrics import (
    INFO_COLORS,
    NETWORK_CATEGORY_COLORS,
    NETWORK_CATEGORY_LINESTYLES,
    NETWORK_CATEGORY_MARKERS,
)

logger = logging.getLogger(__name__)


class BasePlotter:
    """Base class for all plotters with common utilities."""

    def __init__(self):
        """Initialize the plotter."""
        self.category_colors = NETWORK_CATEGORY_COLORS
        self.category_markers = NETWORK_CATEGORY_MARKERS
        self.category_linestyles = NETWORK_CATEGORY_LINESTYLES
        self.info_colors = INFO_COLORS
        self.default_dpi = 300
        self.default_figsize = (10, 6)

    def detect_sweep_types(self, data: pd.DataFrame) -> list[str]:
        """
        Detect the sweep type from the data.
        """
        sweep_types = []

        if "ei_ratio" in data.columns and data["ei_ratio"].unique().size > 1:
            sweep_types.append("ei")

        if "stimulus_duration" in data.columns and "max_gain_factor" in data.columns:
            if (
                data["stimulus_duration"].unique().size > 1
                and data["max_gain_factor"].unique().size > 1
            ):
                sweep_types.append("stoch1")

            elif data["stimulus_duration"].unique().size > 1:
                sweep_types.append("stim_duration")

            elif data["max_gain_factor"].unique().size > 1:
                sweep_types.append("max_gf")

        if "stimulus_duration" in data.columns and "fixed_gain_factor" in data.columns:
            if (
                data["stimulus_duration"].unique().size > 1
                and data["fixed_gain_factor"].unique().size > 1
            ):
                sweep_types.append("stoch3")

            elif (
                data["stimulus_duration"].unique().size > 1
                and "stim_duration" not in sweep_types
            ):
                sweep_types.append("stim_duration")

            elif data["fixed_gain_factor"].unique().size > 1:
                sweep_types.append("fixed_gf")

        if "stimulus_duration" in data.columns and "max_gain_tau_ratio" in data.columns:
            if (
                data["stimulus_duration"].unique().size > 1
                and data["max_gain_tau_ratio"].unique().size > 1
            ):
                sweep_types.append("stoch2")

        if (
            "nparams" in data.columns
            and "network_category" in data.columns
            and data[
                data["network_category"].isin(
                    [
                        "dendritic_shunting",
                        "dendritic_additive",
                        "dendritic_signed",
                    ]
                )
            ]["nparams"]
            .dropna()
            .unique()
            .size
            > 1
        ):
            sweep_types.append("nparams")

        if not sweep_types:
            sweep_types.append("network_type_comp")

        return sweep_types

    def save_figure(
        self,
        fig: plt.Figure,
        output_path: Path,
        dpi: Optional[int] = None,
        close: bool = True,
    ) -> Path:
        """
        Save figure to file.

        Args:
            fig: Matplotlib figure
            output_path: Output file path
            dpi: DPI for saving (defaults to self.default_dpi)
            close: Whether to close figure after saving

        Returns:
            Path to saved file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi or self.default_dpi, bbox_inches="tight")
        if close:
            plt.close(fig)
        return output_path

    def setup_axes_style(
        self,
        ax: plt.Axes,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        use_log_x: bool = False,
        use_log_y: bool = False,
        grid: bool = True,
    ):
        """
        Apply consistent styling to axes.

        Args:
            ax: Matplotlib axes
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            use_log_x: Use log scale for x-axis
            use_log_y: Use log scale for y-axis
            grid: Show grid
        """
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
        if title:
            ax.set_title(title, fontsize=14)

        if use_log_x:
            ax.set_xscale("log")
            formatter = ScalarFormatter()
            formatter.set_scientific(True)
            ax.xaxis.set_major_formatter(formatter)

        if use_log_y:
            ax.set_yscale("log")
            formatter = ScalarFormatter()
            formatter.set_scientific(False)
            ax.yaxis.set_major_formatter(formatter)

        if grid:
            ax.grid(True, alpha=0.3)

        ax.tick_params(axis="both", labelsize=10)

    def plot_by_category(
        self,
        ax: plt.Axes,
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        category_col: str = "network_category",
        show_error: bool = True,
        **plot_kwargs,
    ):
        """
        Plot data grouped by category with consistent styling.

        Args:
            ax: Matplotlib axes
            data: DataFrame to plot
            x_col: X-axis column name
            y_col: Y-axis column name (without _mean suffix)
            category_col: Column to group by
            show_error: Whether to show error bars
            **plot_kwargs: Additional plot arguments
        """
        categories: list[str] = data[category_col].unique()

        # Deterministic ordering by canonical category order
        _canonical_order = [
            "dendritic_shunting",
            "flat_shunting",
            "dendritic_additive",
            "dendritic_normalized_additive",
            "flat_additive",
            "flat_normalized_additive",
            "dendritic_signed",
            "flat_signed",
            "dendritic_mlp",
            "flat_mlp",
            "point_mlp",
            "ss_mlp",
            "ss_mlp_flat",
            "total_param_mlp",
            "active_param_mlp",
            # Old names (for sorting pre-migration data)
            "shunting",
            "no_shunting",
            "unknown",
            "default",
        ]
        sorted_categories = sorted(
            categories,
            key=lambda x: (
                _canonical_order.index(x)
                if x in _canonical_order
                else len(_canonical_order)
            ),
        )

        for category in sorted_categories:
            cat_data = data[data[category_col] == category].sort_values(x_col)

            if cat_data.empty:
                continue

            color = self.category_colors.get(category, "black")
            marker = self.category_markers.get(category, "o")
            linestyle = self.category_linestyles.get(category, "-")

            # Check for _mean and _std columns
            mean_col = f"{y_col}_mean" if f"{y_col}_mean" in cat_data.columns else y_col
            std_col = f"{y_col}_std" if f"{y_col}_std" in cat_data.columns else None

            y_values = cat_data[mean_col]
            yerr = (
                cat_data[std_col]
                if show_error and std_col and std_col in cat_data.columns
                else None
            )

            ax.errorbar(
                cat_data[x_col],
                y_values,
                yerr=yerr,
                label=category,
                color=color,
                marker=marker,
                linestyle=linestyle,
                alpha=0.8,
                capsize=3,
                **plot_kwargs,
            )

    def create_multi_panel_figure(
        self,
        nrows: int,
        ncols: int,
        figsize: Optional[tuple] = None,
        title: Optional[str] = None,
    ) -> tuple:
        """
        Create a multi-panel figure with consistent styling.

        Args:
            nrows: Number of rows
            ncols: Number of columns
            figsize: Figure size (auto-calculated if None)
            title: Overall figure title

        Returns:
            (fig, axes) tuple
        """
        if figsize is None:
            figsize = (ncols * 6, nrows * 5)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        if title:
            fig.suptitle(title, fontsize=16, y=0.995)

        return fig, axes

    def filter_valid_data(
        self,
        data: pd.DataFrame,
        required_cols: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Filter to rows with valid (non-NaN) data in required columns.

        Args:
            data: Input DataFrame
            required_cols: Columns that must have valid data

        Returns:
            Filtered DataFrame
        """
        if data.empty:
            return data

        if required_cols:
            return data.dropna(subset=required_cols)

        return data

"""
Base classes for plotting functionality.

This module provides base classes and common utilities for all plotting operations.
"""

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class PlotConfig:
    """Configuration for plot appearance and behavior."""

    def __init__(self, **kwargs):
        """Initialize plot configuration with defaults that can be overridden."""
        self.figsize = kwargs.get("figsize", (10, 6))
        self.dpi = kwargs.get("dpi", 300)
        self.fontsize = kwargs.get(
            "fontsize",
            {
                "title": 16,
                "label": 14,
                "tick": 12,
                "legend": 12,
                "annotation": 10,
            },
        )
        self.colors = kwargs.get(
            "colors",
            {
                "excitatory": "#e74c3c",
                "inhibitory": "#3498db",
                "combined": "#9b59b6",
                "primary": "#2c3e50",
                "secondary": "#34495e",
                "success": "#27ae60",
                "warning": "#f39c12",
                "danger": "#c0392b",
                "info": "#2980b9",
            },
        )
        self.alpha = kwargs.get(
            "alpha",
            {
                "main": 0.8,
                "fill": 0.3,
                "annotation": 0.9,
            },
        )
        self.grid = kwargs.get(
            "grid",
            {
                "enabled": True,
                "alpha": 0.3,
                "style": "-",
            },
        )
        self.save_format = kwargs.get("save_format", "png")
        self.bbox_inches = kwargs.get("bbox_inches", "tight")


class BasePlotter:
    """Base class for all plotters with common functionality."""

    def __init__(self, config: Optional[PlotConfig] = None):
        """Initialize base plotter with configuration."""
        self.config = config or PlotConfig()
        self.logger = logger

    def _create_figure(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[tuple[int, int]] = None,
        **kwargs,
    ) -> tuple[plt.Figure, Any]:
        """Create a figure with axes."""
        figsize = figsize or self.config.figsize
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
        return fig, axes

    def _save_plot(
        self,
        fig: plt.Figure,
        save_path: str,
        filename: str,
        close_after_save: bool = True,
    ):
        """Save plot to file with error handling."""
        try:
            os.makedirs(save_path, exist_ok=True)
            filepath = os.path.join(save_path, f"{filename}.{self.config.save_format}")
            fig.savefig(
                filepath, dpi=self.config.dpi, bbox_inches=self.config.bbox_inches
            )
            self.logger.info(f"Plot saved to {filepath}")
            if close_after_save:
                plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Error saving plot: {e}")

    def _apply_styling(self, ax: plt.Axes, **kwargs):
        """Apply common styling to axes."""
        # Title
        if "title" in kwargs:
            ax.set_title(kwargs["title"], fontsize=self.config.fontsize["title"])

        # Labels
        if "xlabel" in kwargs:
            ax.set_xlabel(kwargs["xlabel"], fontsize=self.config.fontsize["label"])
        if "ylabel" in kwargs:
            ax.set_ylabel(kwargs["ylabel"], fontsize=self.config.fontsize["label"])

        # Grid
        if self.config.grid["enabled"]:
            ax.grid(
                True,
                alpha=self.config.grid["alpha"],
                linestyle=self.config.grid["style"],
            )

        # Tick parameters
        ax.tick_params(labelsize=self.config.fontsize["tick"])
        if kwargs.get("rotate_xticks", False):
            ax.tick_params(axis="x", rotation=45)

        # Legend
        if kwargs.get("show_legend", False):
            ax.legend(fontsize=self.config.fontsize["legend"])

    def _add_value_labels(
        self,
        ax: plt.Axes,
        bars,
        values: list[float],
        errors: Optional[list[float]] = None,
        format_str: str = "{:.3f}",
        rotation: int = 0,
    ):
        """Add value labels on bars."""
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            y_offset = errors[i] if errors else 0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + y_offset + 0.01 * ax.get_ylim()[1],
                format_str.format(val),
                ha="center",
                va="bottom",
                fontsize=self.config.fontsize["annotation"],
                rotation=rotation,
            )

    def _create_error_plot(
        self, title: str = "Error", message: str = "An error occurred"
    ) -> tuple[plt.Figure, plt.Axes]:
        """Create a simple error plot when data is missing or invalid."""
        fig, ax = self._create_figure()
        ax.text(
            0.5,
            0.5,
            f"{title}\n{message}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=self.config.fontsize["title"],
        )
        ax.set_title(title)
        ax.axis("off")
        return fig, ax

    def _validate_data(self, data: Any, data_name: str = "data") -> bool:
        """Validate that data is not None or empty."""
        if data is None:
            self.logger.warning(f"No {data_name} provided")
            return False
        if isinstance(data, (list, dict, np.ndarray)) and len(data) == 0:
            self.logger.warning(f"Empty {data_name} provided")
            return False
        return True

    def plot(self, *args, **kwargs):
        """Main plotting method to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement plot method")


class ComparisonPlotter(BasePlotter):
    """Base class for plotters that compare multiple datasets."""

    def _normalize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        """Normalize data for comparison."""
        # To be implemented based on specific needs
        return data

    def _align_data(self, datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Align multiple datasets for comparison."""
        # To be implemented based on specific needs
        return datasets


class TimeSeriesPlotter(BasePlotter):
    """Base class for time series plotting."""

    def _smooth_data(
        self, data: np.ndarray, window_size: int = 5, method: str = "mean"
    ) -> np.ndarray:
        """Smooth time series data."""
        if method == "mean":
            kernel = np.ones(window_size) / window_size
            return np.convolve(data, kernel, mode="valid")
        elif method == "ewm":
            # Exponentially weighted mean
            alpha = 2 / (window_size + 1)
            result = np.zeros_like(data)
            result[0] = data[0]
            for i in range(1, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i - 1]
            return result
        else:
            return data

    def _add_trend_line(
        self, ax: plt.Axes, x: np.ndarray, y: np.ndarray, order: int = 1, **kwargs
    ):
        """Add a trend line to the plot."""
        z = np.polyfit(x, y, order)
        p = np.poly1d(z)
        x_trend = np.linspace(x.min(), x.max(), 100)
        ax.plot(
            x_trend,
            p(x_trend),
            "--",
            color=kwargs.get("color", "red"),
            alpha=kwargs.get("alpha", 0.8),
            linewidth=kwargs.get("linewidth", 2),
            label=kwargs.get("label", f"Trend (order={order})"),
        )


class DistributionPlotter(BasePlotter):
    """Base class for distribution plotting."""

    def _calculate_bins(self, data: np.ndarray, method: str = "auto") -> int:
        """Calculate optimal number of bins for histogram."""
        if method == "auto":
            # Use Freedman-Diaconis rule
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr / (len(data) ** (1 / 3))
            n_bins = int((data.max() - data.min()) / bin_width)
            return max(10, min(n_bins, 100))
        elif method == "sqrt":
            return int(np.sqrt(len(data)))
        elif method == "sturges":
            return int(np.log2(len(data)) + 1)
        else:
            return 50

    def _add_statistics_text(
        self, ax: plt.Axes, data: np.ndarray, position: str = "upper right"
    ):
        """Add statistical summary to plot."""
        stats_text = (
            f"Mean: {np.mean(data):.3f}\n"
            f"Std: {np.std(data):.3f}\n"
            f"Min: {np.min(data):.3f}\n"
            f"Max: {np.max(data):.3f}"
        )

        # Position mapping
        positions = {
            "upper right": (0.95, 0.95),
            "upper left": (0.05, 0.95),
            "lower right": (0.95, 0.05),
            "lower left": (0.05, 0.05),
        }

        x, y = positions.get(position, (0.95, 0.95))
        ha = "right" if "right" in position else "left"
        va = "top" if "upper" in position else "bottom"

        ax.text(
            x,
            y,
            stats_text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
            fontsize=self.config.fontsize["annotation"],
        )

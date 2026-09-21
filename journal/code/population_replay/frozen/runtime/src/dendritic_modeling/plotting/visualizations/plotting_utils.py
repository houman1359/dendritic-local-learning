"""Shared utilities for plotting functions."""

import logging
import os
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def save_plot(fig: plt.Figure, save_path: str, filename: str, dpi: int = 300) -> None:
    """Save plot to file with error handling."""
    try:
        os.makedirs(save_path, exist_ok=True)
        filepath = os.path.join(save_path, f"{filename}.png")
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight")
        plt.close(fig)  # Close the figure to free memory
        logger.info(f"Plot saved to {filepath}")

        # Close all figures if too many are open to prevent memory issues
        if len(plt.get_fignums()) > 10:
            plt.close("all")
    except Exception as e:
        logger.warning(f"Error saving plot: {e}")
        plt.close(fig)  # Close figure even if save failed


def create_figure(figsize: tuple[int, int] = (10, 6)) -> tuple[plt.Figure, plt.Axes]:
    """Create a figure and axis with standard settings."""
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def create_subplots(
    rows: int, cols: int, figsize: tuple[int, int] = (15, 10)
) -> tuple[plt.Figure, np.ndarray]:
    """Create subplots with standard settings."""
    # Close some figures if too many are open
    if len(plt.get_fignums()) > 15:
        plt.close("all")
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    return fig, axes


def add_error_bars(
    ax: plt.Axes,
    x,
    y,
    yerr,
    color: str,
    label: str,
    capsize: int = 4,
    capthick: float = 1.5,
):
    """Add error bars to a plot with standard styling."""
    return ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker="o",
        linewidth=2,
        color=color,
        label=label,
        capsize=capsize,
        capthick=capthick,
    )


def annotate_bars(
    ax: plt.Axes, bars, values, errors=None, offset: float = 0.01, fontsize: int = 8
):
    """Annotate bars with values and optional errors."""
    for i, (bar, val) in enumerate(zip(bars, values)):
        if errors is not None and len(errors) > i:
            err = errors[i]
            text = f"{val:.3f}±{err:.3f}"
        else:
            text = f"{val:.3f}"

        height = bar.get_height() if hasattr(bar, "get_height") else val
        x_pos = bar.get_x() + bar.get_width() / 2.0 if hasattr(bar, "get_x") else i

        ax.text(
            x_pos, height + offset, text, ha="center", va="bottom", fontsize=fontsize
        )


def setup_basic_plot(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylabel: str,
    fontsize_dict: Optional[dict[str, int]] = None,
    grid: bool = True,
):
    """Set up basic plot formatting."""
    if fontsize_dict is None:
        fontsize_dict = {"title": 16, "label": 14}

    ax.set_title(title, fontsize=fontsize_dict.get("title", 16))
    ax.set_xlabel(xlabel, fontsize=fontsize_dict.get("label", 14))
    ax.set_ylabel(ylabel, fontsize=fontsize_dict.get("label", 14))

    if grid:
        ax.grid(True, alpha=0.3, axis="y")


def convert_layer_names(
    layer_results: dict[str, dict[str, Any]],
    somatic_synapses: bool = True,
) -> dict[str, dict[str, Any]]:
    """Convert technical layer names to readable format (soma to distal).

    Parameters
    ----------
    layer_results : dict
        Dictionary with layer results keyed by technical layer names
    somatic_synapses : bool, optional
        Whether the model was configured with somatic_synapses=True.
        Default is True for backward compatibility.

    Returns
    -------
    dict
        Dictionary with converted layer names
    """
    if not layer_results:
        return {}

    layer_names = list(layer_results.keys())
    indexed_layers = []

    # First pass: collect all branch indices to determine the mapping dynamically
    branch_indices = []
    for name in layer_names:
        if "branch_layers." in name:
            parts = name.split("branch_layers.")
            if len(parts) > 1:
                try:
                    branch_idx = int(parts[1])
                    branch_indices.append(branch_idx)
                except ValueError:
                    continue

    # Branch indices collected for potential future use

    for name in layer_names:
        try:
            if name == "synthetic_soma":
                # Synthetic soma always gets depth 0 (for information analysis)
                indexed_layers.append((0, name))
            elif "branch_layers." in name:
                parts = name.split("branch_layers.")
                if len(parts) > 1:
                    branch_idx = int(parts[1].split(".")[0])
                    # Use actual layer_idx (branch_idx) so ordering matches ablation (0=soma to distal)
                    indexed_layers.append((branch_idx, name))
                else:
                    indexed_layers.append((999, name))
            elif "layers." in name:
                # Handle ablation analysis layer names (original logic)
                parts = name.split("layers.")
                if len(parts) > 1:
                    layer_part = parts[1].split(".")[0]
                    if layer_part.isdigit():
                        layer_idx = int(layer_part)
                        indexed_layers.append((layer_idx, name))
                    else:
                        indexed_layers.append((999, name))
                else:
                    indexed_layers.append((999, name))
            else:
                indexed_layers.append((999, name))
        except (ValueError, IndexError, AttributeError):
            indexed_layers.append((999, name))

    # Sort by index (CORRECT ORDER: lowest index = most proximal, highest = most distal)
    # In dendritic networks: layer_idx=0 is soma (if present), higher layer_idx is more distal
    indexed_layers.sort(key=lambda x: x[0])

    # Create new dictionary with readable names
    converted_results = {}
    for _i, (original_idx, original_name) in enumerate(indexed_layers):
        # Treat layer_idx==0 as Soma in all cases (synthetic or real)
        is_soma_layer = original_idx == 0

        if is_soma_layer:
            new_name = "Soma"
        else:
            # Match ablation labeling: distal_layer = original_idx - min_depth (but since min_depth=0 for soma, it's original_idx)
            distal_layer_num = original_idx
            new_name = f"Distal Layer {distal_layer_num}"

        converted_results[new_name] = layer_results[original_name]

        # Add metadata about whether this is actually a functional soma
        if is_soma_layer and "has_somatic_synapses" not in converted_results[new_name]:
            converted_results[new_name]["has_somatic_synapses"] = somatic_synapses

    # Ensure Soma appears on the x-axis even when somatic_synapses is False
    if not somatic_synapses and "Soma" not in converted_results:
        ordered = {"Soma": {"has_somatic_synapses": False}}
        ordered.update(converted_results)
        return ordered

    return converted_results


def get_color_scheme():
    """Get standard color scheme for consistency."""
    return {
        "excitatory": "#e74c3c",
        "inhibitory": "#3498db",
        "combined": "#9b59b6",
        "primary": "#2c3e50",
        "baseline": "#2c3e50",
    }


def create_heatmap(
    ax: plt.Axes,
    data_matrix: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    title: str,
    cmap: str = "viridis",
    annotate: bool = True,
    fmt: str = ".3f",
):
    """Create a standardized heatmap."""

    im = ax.imshow(data_matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_yticklabels(ylabels)
    ax.set_title(title)

    # Add colorbar
    plt.colorbar(im, ax=ax, label="Value")

    # Add text annotations
    if annotate:
        for i in range(len(ylabels)):
            for j in range(len(xlabels)):
                val = data_matrix[i, j]
                if val > 0:
                    text_color = "white" if val > np.max(data_matrix) * 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:{fmt}}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                        weight="bold",
                    )

    return im


def handle_plot_error(
    func_name: str, error: Exception, create_placeholder: bool = True
):
    """Standard error handling for plotting functions."""
    logger.warning(f"Error in {func_name}: {error}")

    if create_placeholder:
        try:
            fig, ax = create_figure()
            ax.text(
                0.5,
                0.5,
                f"{func_name}\n(Error: {str(error)[:50]}...)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(f"{func_name} - Error")
            return fig, ax
        except Exception:  # Broad exception for matplotlib/plotting errors
            return None
    return None

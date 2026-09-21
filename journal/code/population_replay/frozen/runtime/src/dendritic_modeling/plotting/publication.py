"""Publication-grade plotting style and deterministic vector export helpers.

This module is the package-owned plotting surface for paper figures.  It keeps
the visual vocabulary, Illustrator-editable heatmaps, and atomic PDF/SVG/PNG
export behavior independent of any individual manuscript directory.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import QuadMesh
from matplotlib.colorbar import Colorbar
from matplotlib.transforms import Bbox


@dataclass(frozen=True)
class PaperColors:
    """Color-blind-conscious semantic colors shared by publication figures."""

    shunting: str = "#d62728"
    additive: str = "#1f77b4"
    delta: str = "#2a9d55"
    lateral: str = "#7b3294"
    feedforward: str = "#008837"
    excitatory: str = "#e67e22"
    inhibitory: str = "#16a085"
    mlp: str = "#7f7f7f"
    point_mlp: str = "#bcbd22"
    cond_mi: str = "#9467bd"
    cond_mi_e: str = "#d4a017"
    voltage: str = "#2ca02c"
    branch_voltage: str = "#ff7f0e"
    noise_uniform: str = "#ff7f0e"
    noise_gaussian: str = "#4daf4a"
    black: str = "#111111"
    gray: str = "#7f7f7f"
    light_gray: str = "#cccccc"


COLORS = PaperColors()

# Model comparisons use one stable visual grammar across papers.
SH_STYLE: dict[str, Any] = {
    "color": COLORS.shunting,
    "ls": "-",
    "lw": 2.0,
    "marker": "s",
    "ms": 3.5,
    "markeredgecolor": "white",
    "markeredgewidth": 0.3,
    "alpha": 1.0,
    "label": "Shunting",
}
ADD_STYLE: dict[str, Any] = {
    "color": COLORS.additive,
    "ls": "-",
    "lw": 1.8,
    "marker": "o",
    "ms": 3.5,
    "markeredgecolor": "white",
    "markeredgewidth": 0.3,
    "alpha": 0.9,
    "label": "Additive",
}
E_STYLE: dict[str, Any] = {
    "color": COLORS.excitatory,
    "ls": "-",
    "lw": 1.8,
    "marker": "^",
    "ms": 4.0,
    "markeredgecolor": "white",
    "markeredgewidth": 0.3,
    "alpha": 1.0,
}
I_STYLE: dict[str, Any] = {
    "color": "#5ab4ac",
    "ls": "-",
    "lw": 1.5,
    "marker": "D",
    "ms": 3.0,
    "markeredgecolor": "white",
    "markeredgewidth": 0.3,
    "alpha": 0.85,
}
ERRBAR: dict[str, Any] = {"capsize": 2, "capthick": 0.7}
SHADE_ALPHA = 0.18
NET_STYLE: dict[str, tuple[str, str]] = {
    "dendritic_shunting": ("Shunting", COLORS.shunting),
    "dendritic_additive": ("Additive", COLORS.additive),
    "dendritic_mlp": ("Dendritic MLP", COLORS.mlp),
    "point_mlp": ("Point MLP", COLORS.point_mlp),
    "shunting": ("Shunting", COLORS.shunting),
    "no_shunting": ("Additive", COLORS.additive),
}

# Stable values used by every paper-facing vector export.  Live SVG text is
# required for Illustrator editing; the fixed salt makes generated IDs stable.
FIGURE_EXPORT_CREATOR = "dendritic-modeling"
FIGURE_SVG_HASHSALT = "dendritic-modeling-editable-v1"
_DETERMINISTIC_EXPORT_RCPARAMS: dict[str, Any] = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "svg.hashsalt": FIGURE_SVG_HASHSALT,
}


def setup_paper_style(*, font_scale: float = 1.0) -> None:
    """Set deterministic journal-quality Matplotlib defaults."""

    base_font = 8.0 * font_scale
    mpl.rcParams.update(
        {
            "font.size": base_font,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.titlesize": 8.5 * font_scale,
            "axes.titleweight": "medium",
            "axes.labelsize": 7.5 * font_scale,
            "axes.labelweight": "regular",
            "xtick.labelsize": 6.8 * font_scale,
            "ytick.labelsize": 6.8 * font_scale,
            "legend.fontsize": 6.5 * font_scale,
            "figure.titlesize": 10.0 * font_scale,
            "lines.linewidth": 1.8,
            "lines.markersize": 3.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.linewidth": 0.7,
            "axes.facecolor": "white",
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#111111",
            "axes.labelpad": 4.0,
            "axes.titlepad": 8.0,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.35,
            "grid.color": "#cccccc",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.minor.visible": False,
            "ytick.minor.visible": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.35,
            "ytick.minor.width": 0.35,
            "xtick.major.pad": 3.5,
            "ytick.major.pad": 3.5,
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "legend.frameon": True,
            "legend.framealpha": 0.85,
            "legend.edgecolor": "#dddddd",
            "legend.fancybox": False,
            "legend.borderpad": 0.4,
            "legend.handlelength": 1.6,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.0,
            "legend.labelspacing": 0.35,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": FIGURE_SVG_HASHSALT,
        }
    )


def vector_heatmap(
    ax: plt.Axes,
    values: Any,
    *,
    origin: str = "upper",
    aspect: str | float = "equal",
    cmap: Any = None,
    norm: Any = None,
    vmin: float | None = None,
    vmax: float | None = None,
    gid: str | None = None,
) -> QuadMesh:
    """Draw an Illustrator-editable matrix without an embedded bitmap."""

    if origin not in {"upper", "lower"}:
        raise ValueError("origin must be 'upper' or 'lower'")
    if norm is not None and (vmin is not None or vmax is not None):
        raise ValueError("vmin/vmax cannot be combined with an explicit norm")

    matrix = np.ma.masked_invalid(np.ma.asarray(values, dtype=float))
    if matrix.ndim != 2:
        raise ValueError("vector_heatmap values must be a two-dimensional matrix")
    n_rows, n_columns = matrix.shape
    if n_rows == 0 or n_columns == 0:
        raise ValueError("vector_heatmap values cannot be empty")

    x_edges = np.arange(n_columns + 1, dtype=float) - 0.5
    y_edges = np.arange(n_rows + 1, dtype=float) - 0.5
    mesh = ax.pcolormesh(
        x_edges,
        y_edges,
        matrix,
        shading="flat",
        cmap=cmap,
        norm=norm,
        vmin=vmin if norm is None else None,
        vmax=vmax if norm is None else None,
        rasterized=False,
        antialiased=False,
        edgecolors="none",
        linewidth=0.0,
        snap=True,
    )
    mesh.set_rasterized(False)
    if gid is not None:
        mesh.set_gid(gid)

    ax.set_xlim(-0.5, n_columns - 0.5)
    if origin == "upper":
        ax.set_ylim(n_rows - 0.5, -0.5)
    else:
        ax.set_ylim(-0.5, n_rows - 0.5)
    ax.set_aspect(aspect)
    return mesh


def vector_colorbar(
    figure: plt.Figure,
    mappable: Any,
    **kwargs: Any,
) -> Colorbar:
    """Create a colorbar whose gradient remains vector geometry."""

    colorbar = figure.colorbar(mappable, **kwargs)
    if colorbar.solids is not None:
        colorbar.solids.set_rasterized(False)
        colorbar.solids.set_antialiased(False)
        colorbar.solids.set_edgecolor("face")
        colorbar.solids.set_linewidth(0.0)
    return colorbar


def _frozen_save_bbox(fig: plt.Figure) -> Bbox | None:
    """Resolve one tight bounding box for every output backend."""

    configured_bbox = mpl.rcParams["savefig.bbox"]
    if configured_bbox != "tight":
        return None
    pad_inches = mpl.rcParams["savefig.pad_inches"]
    if not isinstance(pad_inches, (int, float)):
        raise ValueError("savefig.pad_inches must be numeric for deterministic export")
    renderer = fig.canvas.get_renderer()
    return fig.get_tightbbox(renderer).padded(float(pad_inches))


def save_figure(
    fig: plt.Figure,
    *,
    out_path: str | Path,
    also_png: bool = True,
    also_svg: bool = True,
    close: bool = True,
) -> None:
    """Atomically save deterministic PDF/SVG/PNG publication artifacts."""

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    outputs = {"pdf": out_path.with_suffix(".pdf")}
    if also_svg:
        outputs["svg"] = out_path.with_suffix(".svg")
    if also_png:
        outputs["png"] = out_path.with_suffix(".png")

    try:
        with mpl.rc_context(_DETERMINISTIC_EXPORT_RCPARAMS):
            # Resolve layout/crop once on the Agg canvas so every backend has
            # exactly the same physical artboard.
            fig.canvas.draw()
            if hasattr(fig, "set_layout_engine"):
                fig.set_layout_engine("none")
            frozen_bbox = _frozen_save_bbox(fig)
            save_options = {
                "bbox_inches": frozen_bbox,
                "pad_inches": 0.0 if frozen_bbox is not None else None,
                "facecolor": "white",
                "edgecolor": "none",
                "transparent": False,
            }
            title = out_path.stem

            with tempfile.TemporaryDirectory(
                prefix=f".{out_path.stem}-export-",
                dir=out_path.parent,
            ) as temporary_directory:
                temporary = Path(temporary_directory)
                staged: dict[str, Path] = {}

                staged["pdf"] = temporary / outputs["pdf"].name
                fig.savefig(
                    staged["pdf"],
                    metadata={
                        "Title": title,
                        "Creator": FIGURE_EXPORT_CREATOR,
                        "CreationDate": None,
                        "ModDate": None,
                    },
                    **save_options,
                )
                if also_svg:
                    staged["svg"] = temporary / outputs["svg"].name
                    fig.savefig(
                        staged["svg"],
                        metadata={
                            "Title": title,
                            "Creator": FIGURE_EXPORT_CREATOR,
                            "Date": None,
                        },
                        **save_options,
                    )
                if also_png:
                    staged["png"] = temporary / outputs["png"].name
                    fig.savefig(
                        staged["png"],
                        dpi=300,
                        metadata={
                            "Title": title,
                            "Software": FIGURE_EXPORT_CREATOR,
                        },
                        **save_options,
                    )

                # Stage every artifact before atomically replacing any output.
                for output_format, destination in outputs.items():
                    staged[output_format].replace(destination)
    finally:
        if close:
            plt.close(fig)


def panel_label(
    ax: plt.Axes,
    letter: str,
    *,
    x: float = -0.12,
    y: float = 1.08,
) -> None:
    """Add a standard bold panel letter just outside an axis."""

    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.0,
        fontweight="bold",
        color="black",
        fontfamily="sans-serif",
    )


def missing_data_panel(ax: plt.Axes, title: str, message: str) -> None:
    """Draw a visibly unavailable-data placeholder on an axis."""

    ax.set_axis_off()
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color="#888888",
    )


def plot_with_band(
    ax: plt.Axes,
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    yerr: np.ndarray | pd.Series | float | None = None,
    *,
    color: str = "#333333",
    ls: str = "-",
    lw: float = 1.5,
    alpha: float = 1.0,
    label: str | None = None,
    shade_alpha: float = SHADE_ALPHA,
    marker: str | None = None,
    ms: float = 0,
    markevery: int | None = None,
) -> None:
    """Plot a line with a symmetric uncertainty band."""

    x_values = np.asarray(x, float)
    y_values = np.asarray(y, float)
    kwargs: dict[str, Any] = {
        "color": color,
        "ls": ls,
        "lw": lw,
        "alpha": alpha,
    }
    if marker:
        kwargs["marker"] = marker
        kwargs["ms"] = ms
        kwargs["markeredgecolor"] = color
        kwargs["markeredgewidth"] = 0
    if markevery is not None:
        kwargs["markevery"] = markevery
    if label:
        kwargs["label"] = label
    ax.plot(x_values, y_values, **kwargs)
    if yerr is not None:
        error = np.asarray(yerr, float)
        ax.fill_between(
            x_values,
            y_values - error,
            y_values + error,
            color=color,
            alpha=shade_alpha,
            linewidth=0,
        )


def slim_colorbar(
    mappable: Any,
    ax: plt.Axes,
    label: str = "",
    *,
    fraction: float = 0.025,
    pad: float = 0.02,
    aspect: int = 30,
) -> Colorbar:
    """Add a thin editable colorbar with publication-scale typography."""

    colorbar = vector_colorbar(
        ax.figure,
        mappable,
        ax=ax,
        fraction=fraction,
        pad=pad,
        aspect=aspect,
    )
    if label:
        colorbar.set_label(label, fontsize=6)
    colorbar.ax.tick_params(labelsize=6, length=2, width=0.4)
    colorbar.outline.set_linewidth(0.4)
    return colorbar


def inline_label(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    color: str,
    *,
    fontsize: float = 6,
    ha: str = "left",
    va: str = "center",
    offset: tuple[float, float] = (3, 0),
) -> None:
    """Place a label directly on or near a curve."""

    ax.annotate(
        text,
        (x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        ha=ha,
        va=va,
        fontweight="medium",
    )


def clean_axis(ax: plt.Axes, *, title: str | None = None) -> None:
    """Apply minimal publication-axis formatting."""

    ax.grid(False)
    ax.tick_params(which="both", direction="out")
    if title:
        ax.set_title(title, fontsize=8.5, fontweight="medium", pad=8)


__all__ = [
    "ADD_STYLE",
    "COLORS",
    "ERRBAR",
    "E_STYLE",
    "FIGURE_EXPORT_CREATOR",
    "FIGURE_SVG_HASHSALT",
    "I_STYLE",
    "NET_STYLE",
    "SHADE_ALPHA",
    "SH_STYLE",
    "PaperColors",
    "clean_axis",
    "inline_label",
    "missing_data_panel",
    "panel_label",
    "plot_with_band",
    "save_figure",
    "setup_paper_style",
    "slim_colorbar",
    "vector_colorbar",
    "vector_heatmap",
]

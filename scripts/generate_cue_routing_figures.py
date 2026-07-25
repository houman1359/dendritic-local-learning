#!/usr/bin/env python3
"""Generate publication-quality figures for the routed cue-integration benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import (  # noqa: E402
    # noqa: E402,
    COLORS as NEURIPS_COLORS,
    FIG_W,
    LW_EDGE,
    LW_ERR,
    LW_HAIR,
    PT_ANNOT,
    PT_LABEL,
    PT_LEGEND,
    PT_SMALL,
    REF_LW,
    apply_neurips_style,
    clean_legend,
    grid_figure,
    panel_label,
    panel_title,
    style_axis,
)
apply_neurips_style()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_DIR = SCRIPT_DIR.parent
FIGURES_DIR = DRAFT_DIR / "figures"
ANALYSIS_DIR = DRAFT_DIR / "analysis"
SUMMARY_CSV = ANALYSIS_DIR / "cue_routing_summary.csv"

COLOR_SHUNTING = NEURIPS_COLORS["shunting"]
COLOR_ADDITIVE = NEURIPS_COLORS["additive"]
COLOR_PATHWAY = NEURIPS_COLORS["pathway"]
COLOR_CONTROL = NEURIPS_COLORS["low_rank"]
COLOR_BP = "#4A4A4A"
COLOR_LIGHT = "#F5F2EA"
DOUBLE_COL_W = 11.0
DPI = 300


def _setup_style() -> None:
    # Unified style already applied at import time; this is a no-op kept for call-site compat.
    pass


def _panel(ax: plt.Axes, label: str, x: float = -0.16, y: float = 1.08) -> None:
    panel_label(ax, label, x=x, y=y)


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=DPI)
    print(f"Saved {name}.{{pdf,png}}")


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    return pd.read_csv(path)


def _load_router_probabilities(run_dir: str) -> np.ndarray:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    model_path = run_dir / "final_model.pt"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing config/model for run: {run_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    temperature = float(
        config.get("model", {})
        .get("encoder", {})
        .get("params", {})
        .get("learned_router_temperature", 1.0)
        or 1.0
    )
    state = torch.load(model_path, map_location="cpu")
    logits_key = next(key for key in state.keys() if key.endswith("assignment_logits"))
    logits = state[logits_key].float()
    probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
    return probs.detach().cpu().numpy()


def _find_record(
    df: pd.DataFrame, strategy: str, network_type: str, router_mode: str, variant: str
) -> dict[str, Any]:
    row = df[
        (df["strategy"] == strategy)
        & (df["network_type"] == network_type)
        & (df["router_mode"] == router_mode)
        & (df["variant"] == variant)
    ]
    if len(row) == 0:
        raise RuntimeError(
            "Missing cue-routing summary row for "
            f"{strategy=} {network_type=} {router_mode=} {variant=}"
        )
    return row.iloc[0].to_dict()


def _summary_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    entries = [
        ("local_ca", "dendritic_additive", "fixed", "baseline", "Add.\nfixed"),
        ("local_ca", "dendritic_additive", "learned", "baseline", "Add.\nlearned"),
        ("local_ca", "dendritic_shunting", "learned", "baseline", "Shunt.\nMW"),
        ("local_ca", "dendritic_shunting", "learned", "baseline_tuned", "Shunt.\nMW tuned"),
        ("local_ca", "dendritic_shunting", "learned", "low_rank_k1", "Random\n$K{=}1$"),
        ("local_ca", "dendritic_shunting", "learned", "low_rank_k2", "Random\n$K{=}2$"),
        ("local_ca", "dendritic_shunting", "learned", "pathway_vector_tuned", "PV-LocalCA"),
        ("standard", "dendritic_shunting", "learned", "baseline", "Shunt. BP\nlearned"),
    ]

    records: list[dict[str, Any]] = []
    for strategy, network_type, router_mode, variant, label in entries:
        record = _find_record(df, strategy, network_type, router_mode, variant)
        record["short_label"] = label
        records.append(record)
    return records


def _bar_style(record: dict[str, Any]) -> tuple[str, str]:
    variant = str(record["variant"])
    strategy = str(record["strategy"])
    network_type = str(record["network_type"])
    if "pathway_vector" in variant:
        return COLOR_PATHWAY, "#40204F"
    if "low_rank" in variant:
        return COLOR_CONTROL, "#7A3D10"
    if strategy == "standard":
        return COLOR_BP, "#2D2D2D"
    if network_type == "dendritic_additive":
        return COLOR_ADDITIVE, "white"
    if variant in {"temp03", "freeze20"}:
        return COLOR_CONTROL, "#7A3D10"
    return COLOR_SHUNTING, "white"


def _draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    wh: tuple[float, float],
    text: str,
    facecolor: str,
    edgecolor: str = "#3A3A3A",
    fontsize: float = 6.7,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=LW_EDGE,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=fontsize)
    return patch


def _draw_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    text: str | None = None,
    text_xy: tuple[float, float] | None = None,
    linestyle: str = "-",
    linewidth: float = 1.1,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=10,
        linewidth=linewidth,
        color=color,
        linestyle=linestyle,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)
    if text and text_xy is not None:
        ax.text(
            text_xy[0],
            text_xy[1],
            text,
            color=color,
            fontsize=PT_SMALL,
            ha="center",
            va="center",
        )


def _plot_task_schematic(ax: plt.Axes) -> None:
    _panel(ax, "A")
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    _draw_box(ax, (0.04, 0.73), (0.18, 0.13), "Context\nreliability cue", "#F3E7C8")
    _draw_box(ax, (0.06, 0.49), (0.14, 0.10), "Cue A\nreliable", "#DCE7F8")
    _draw_box(ax, (0.06, 0.25), (0.14, 0.10), "Cue B\ncan conflict", "#DCE7F8")

    _draw_box(ax, (0.26, 0.52), (0.14, 0.12), "Path 1", "#EAF4EC", edgecolor=COLOR_SHUNTING)
    _draw_box(ax, (0.26, 0.27), (0.14, 0.12), "Path 2", "#EAF4EC", edgecolor=COLOR_SHUNTING)
    _draw_box(ax, (0.50, 0.41), (0.14, 0.12), "Soma /\ndecoder", COLOR_LIGHT)

    _draw_arrow(ax, (0.20, 0.54), (0.26, 0.58), COLOR_ADDITIVE, linewidth=LW_ERR)
    _draw_arrow(ax, (0.20, 0.30), (0.26, 0.33), COLOR_ADDITIVE, linewidth=LW_ERR)
    _draw_arrow(ax, (0.40, 0.58), (0.50, 0.47), COLOR_SHUNTING, linewidth=LW_ERR)
    _draw_arrow(ax, (0.40, 0.33), (0.50, 0.47), COLOR_SHUNTING, linewidth=LW_ERR)
    _draw_arrow(ax, (0.22, 0.79), (0.33, 0.64), "#A07015", text="route", text_xy=(0.30, 0.72), linewidth=LW_ERR)
    _draw_arrow(ax, (0.22, 0.79), (0.33, 0.39), "#A07015", linewidth=LW_ERR)

    _draw_box(ax, (0.68, 0.66), (0.22, 0.085), "MW / scalar fallback", "#F4F4F4")
    _draw_box(ax, (0.68, 0.47), (0.22, 0.095), "Random low-rank\n$e_n = \\Gamma_K(\\delta_0)$", "#F8EBDD", edgecolor=COLOR_CONTROL)
    _draw_box(ax, (0.68, 0.25), (0.22, 0.115), "Structured pathways\n$e_n = \\sum_k q_{n,k} c_k$", "#EFE4F8", edgecolor=COLOR_PATHWAY)
    _draw_arrow(ax, (0.64, 0.47), (0.68, 0.70), "#777777", linestyle="--", text="shared signal", text_xy=(0.80, 0.79), linewidth=LW_ERR)
    _draw_arrow(ax, (0.64, 0.47), (0.68, 0.515), COLOR_CONTROL, text="$K$ unstructured channels", text_xy=(0.81, 0.60), linewidth=LW_ERR)
    _draw_arrow(ax, (0.64, 0.47), (0.68, 0.305), COLOR_PATHWAY, text="pathway-aware channels", text_xy=(0.81, 0.18), linewidth=LW_ERR)

    ax.text(
        0.04,
        0.06,
        "Branch identity matters: scalar feedback collapses routed credit.",
        fontsize=PT_SMALL,
        ha="left",
        va="bottom",
        color="#333333",
    )
    ax.set_title("Cue-routing task", fontsize=PT_LABEL)


def _plot_accuracy_panel(ax: plt.Axes, records: list[dict[str, Any]]) -> None:
    _panel(ax, "B", x=-0.13)
    style_axis(ax, grid="x")
    labels = [str(record["short_label"]) for record in records]
    values = [100.0 * float(record["test_accuracy"]) for record in records]
    errors = [100.0 * float(record.get("test_accuracy_std", 0.0) or 0.0) for record in records]
    y = np.arange(len(records))

    for ypos, value, error, record in zip(y, values, errors, records):
        color, edge = _bar_style(record)
        ax.barh(
            ypos,
            value,
            color=color,
            edgecolor=edge,
            linewidth=LW_HAIR,
            height=0.72,
            alpha=0.96,
            zorder=3,
        )
        if error > 0:
            ax.errorbar(
                value,
                ypos,
                xerr=error,
                fmt="none",
                ecolor="#333333",
                elinewidth=0.6,
                capsize=2,
                capthick=0.6,
                zorder=5,
            )
        ax.text(
            min(value + 0.5, 99.6),
            ypos,
            f"{value:.1f}",
            va="center",
            ha="left",
            fontsize=PT_ANNOT,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85},
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=PT_SMALL, linespacing=0.92)
    ax.invert_yaxis()
    ax.set_xlim(70, 100.2)
    ax.set_xlabel("Test accuracy (%)")
    ax.set_title("Higher-rank feedback helps", fontsize=PT_LABEL, pad=7)
    ax.axvline(95, color="#999999", linewidth=LW_HAIR, linestyle=":")
    ax.text(
        0.73,
        0.96,
        "high-accuracy regime",
        transform=ax.transAxes,
        fontsize=PT_SMALL,
        color="#666666",
        ha="left",
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5, "alpha": 0.85},
    )


def _plot_specialization_panel(ax: plt.Axes, df: pd.DataFrame) -> None:
    _panel(ax, "C")
    style_axis(ax, grid="both")
    learned = df[df["router_mode"] == "learned"].copy()
    learned = learned.dropna(subset=["router_mean_max_assignment", "test_accuracy"]).reset_index(drop=True)
    if learned.empty:
        ax.text(0.5, 0.5, "No learned-router runs found", ha="center", va="center")
        return

    for _, row in learned.iterrows():
        variant = str(row["variant"])
        strategy = str(row["strategy"])
        network_type = str(row["network_type"])
        if "pathway_vector" in variant:
            color = COLOR_PATHWAY
            marker = "*"
            size = 140
            edge = "#40204F"
        elif "low_rank" in variant:
            color = COLOR_CONTROL
            marker = "D"
            size = 62
            edge = "#7A3D10"
        elif strategy == "standard":
            color = COLOR_BP
            marker = "s"
            size = 42
            edge = "white"
        elif network_type == "dendritic_additive":
            color = COLOR_ADDITIVE
            marker = "o"
            size = 46
            edge = "white"
        elif variant in {"temp03", "freeze20"}:
            color = COLOR_CONTROL
            marker = "o"
            size = 52
            edge = "#7A3D10"
        else:
            color = COLOR_SHUNTING
            marker = "o"
            size = 48
            edge = "white"

        ax.scatter(
            float(row["router_mean_max_assignment"]),
            100.0 * float(row["test_accuracy"]),
            s=size,
            c=color,
            marker=marker,
            edgecolors=edge,
            linewidths=0.7,
            alpha=0.98,
            zorder=4,
        )

    annotations = {
        "baseline": "MW",
        "low_rank_k2": "$K{=}2$",
        "pathway_vector_tuned": "PV",
    }
    targets = learned[
        (learned["strategy"] == "local_ca")
        & (learned["network_type"] == "dendritic_shunting")
        & (learned["variant"].isin(annotations))
    ]
    for _, row in targets.iterrows():
        is_pathway = "pathway_vector" in str(row["variant"])
        ax.annotate(
            annotations[str(row["variant"])],
            xy=(float(row["router_mean_max_assignment"]), 100.0 * float(row["test_accuracy"])),
            xytext=(-18 if is_pathway else 8, 12 if is_pathway else -16),
            textcoords="offset points",
            fontsize=PT_SMALL,
            color="#333333",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85},
        )

    ax.text(
        0.03,
        0.05,
        "o LocalCA | square BP | diamond low-rank | star pathway",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PT_SMALL,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CCCCCC", "linewidth": 0.5},
    )
    ax.set_xlabel("Mean max router assignment")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Specialization is not enough", fontsize=PT_LABEL)
    ax.set_xlim(0.955, 1.0015)
    ax.set_ylim(78, 100.5)


def _plot_assignment_panel(ax: plt.Axes, summary_row: dict[str, Any]) -> None:
    _panel(ax, "D", x=-0.14)
    probs = _load_router_probabilities(str(summary_row["run_dir"]))
    image = ax.imshow(probs, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)

    n_features, n_pathways = probs.shape
    ax.set_xticks(np.arange(n_pathways))
    ax.set_xticklabels([f"Path {idx + 1}" for idx in range(n_pathways)])
    ax.set_yticks(np.arange(n_features))
    row_labels = [f"A{idx + 1}" for idx in range(n_features // 2)] + [f"B{idx + 1}" for idx in range(n_features // 2)]
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Latent pathway")
    ax.set_ylabel("Cue feature")
    ax.set_title("Recovered pathway map", fontsize=PT_LABEL)
    if n_features % 2 == 0:
        midpoint = n_features // 2 - 0.5
        ax.axhline(midpoint, color="white", linewidth=LW_ERR, alpha=0.9)
        ax.text(-0.95, midpoint / 2.0, "Cue A", fontsize=PT_SMALL, rotation=90, va="center")
        ax.text(-0.95, midpoint + 1 + midpoint / 2.0, "Cue B", fontsize=PT_SMALL, rotation=90, va="center")

    ax.text(
        0.02,
        0.02,
        (
            f"test {100.0 * float(summary_row['test_accuracy']):.1f}%"
            f" | max assignment {float(summary_row['router_mean_max_assignment']):.3f}"
        ),
        transform=ax.transAxes,
        fontsize=PT_SMALL,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )
    cbar = plt.colorbar(image, ax=ax, fraction=0.09, pad=0.05)
    cbar.set_label("Assignment prob.", fontsize=PT_LEGEND)
    cbar.ax.tick_params(labelsize=7.5)


def build_figure(summary_csv: Path) -> None:
    _setup_style()
    summary = _load_summary(summary_csv)
    records = _summary_records(summary)
    pv_row = _find_record(summary, "local_ca", "dendritic_shunting", "learned", "pathway_vector_tuned")

    fig, axes = grid_figure(2, 2, panel_h=2.15, gap_w=0.95, gap_h=0.72,
                            margin_l=0.70, margin_b=0.55)
    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    _plot_task_schematic(ax_a)
    _plot_accuracy_panel(ax_b, records)
    _plot_specialization_panel(ax_c, summary)
    _plot_assignment_panel(ax_d, pv_row)

    _save(fig, "fig_s_cue_routing")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    args = parser.parse_args()
    build_figure(args.summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

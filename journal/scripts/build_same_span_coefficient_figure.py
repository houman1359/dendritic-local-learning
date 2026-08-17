#!/usr/bin/env python3
"""Render the same-span coefficient-learning bias--variance experiment."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from journal_style import (
    COLORS,
    FIG_W,
    LW_DATA,
    LW_REF,
    PT_ANNOT,
    PT_LEGEND,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_legend,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "same_span_coefficient_learning"
CONFIG = ROOT / "configs" / "credit_phase_theory" / "same_span_learning_confirmatory.json"
FIGURES = ROOT / "figures" / "generated"


def load_runner():
    path = ROOT / "scripts" / "run_same_span_coefficient_learning.py"
    spec = importlib.util.spec_from_file_location("same_span_figure_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNNER = load_runner()
STYLES = {
    "tree_haar": ("orthonormal Haar", COLORS["shunting"], "o"),
    "raw_nested_indicators": ("raw nested", COLORS["per_soma"], "s"),
    "static_gain_scaled_nested": ("scaled nested", COLORS["additive"], "D"),
}


def schematic(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    panel_title(ax, "A", "Same address span")
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.56), 0.34, 0.20, boxstyle="round,pad=0.02",
            facecolor="white", edgecolor=COLORS["dend"], lw=LW_DATA,
        )
    )
    ax.text(0.19, 0.66, "one rank-8\nroute span", ha="center", va="center",
            fontsize=PT_ANNOT, color=COLORS["dend"])
    for y, label, color in [
        (0.76, "Haar", COLORS["shunting"]),
        (0.56, "nested", COLORS["per_soma"]),
        (0.36, "scaled", COLORS["additive"]),
    ]:
        ax.add_patch(
            FancyBboxPatch(
                (0.63, y - 0.07), 0.28, 0.14, boxstyle="round,pad=0.015",
                facecolor="white", edgecolor=color, lw=LW_DATA,
            )
        )
        ax.text(0.77, y, label, ha="center", va="center", fontsize=PT_ANNOT,
                color=color)
        ax.add_patch(FancyArrowPatch((0.36, 0.66), (0.62, y), arrowstyle="-|>",
                                     mutation_scale=7, lw=LW_REF, color=COLORS["mute"]))
    ax.text(0.50, 0.16, r"same $P_\Phi$; different $\Phi^\top\Phi$",
            ha="center", va="center", fontsize=PT_ANNOT, color=COLORS["ink"])


def trajectory(ax: plt.Axes, summary: pd.DataFrame, sample_size: int, letter: str) -> None:
    part = summary[
        summary.effective_sample_size.eq(sample_size)
        & summary.optimizer.eq("vanilla_local")
    ]
    for parameterization, (label, color, marker) in STYLES.items():
        curve = part[part.parameterization.eq(parameterization)].sort_values("checkpoint")
        ax.plot(curve.checkpoint, curve.mean_population_loss, color=color,
                marker=marker, ms=3.2, lw=LW_DATA, label=label)
        ax.fill_between(curve.checkpoint, curve.ci95_low_population_loss,
                        curve.ci95_high_population_loss, color=color, alpha=0.09,
                        linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks([1, 5, 20, 80], ["1", "5", "20", "80"])
    ax.set_xlabel("coefficient updates")
    ax.set_ylabel("population loss")
    regime = "Low data" if sample_size == 4 else "High data"
    panel_title(ax, letter, f"{regime} ($n={sample_size}$)")
    style_axis(ax)


def main() -> None:
    apply_neurips_style()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    summary = pd.read_csv(SOURCE / "condition_summary.csv")
    contrasts = pd.read_csv(SOURCE / "paired_contrasts.csv")
    predictions = pd.read_csv(SOURCE / "theory_predictions.csv")
    dictionaries = RUNNER.dictionaries(cfg)

    fig, axes = plt.subplots(
        2, 3, figsize=(FIG_W, 4.95),
        gridspec_kw={"left": 0.10, "right": 0.985, "bottom": 0.15, "top": 0.91,
                     "wspace": 0.69, "hspace": 0.79},
    )
    ax_a, ax_b, ax_c, ax_d, ax_e, ax_f = axes.ravel()
    schematic(ax_a)

    for parameterization, dictionary in dictionaries.items():
        eigenvalues = np.linalg.eigvalsh(dictionary @ dictionary.T)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        eigenvalues = np.sort(eigenvalues / eigenvalues.max())[::-1]
        label, color, marker = STYLES[parameterization]
        ax_b.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, color=color,
                  marker=marker, ms=3.2, lw=LW_DATA, label=label)
    ax_b.set_yscale("log")
    ax_b.set_xlabel("positive Gram mode")
    ax_b.set_ylabel(r"normalized eigenvalue $\lambda_j/\lambda_1$")
    panel_title(ax_b, "B", "Different conditioning")
    style_axis(ax_b)

    trajectory(ax_c, summary, 4, "C")
    trajectory(ax_d, summary, 256, "D")

    final = summary[
        summary.checkpoint.eq(int(cfg["iterations"]))
        & summary.optimizer.eq("vanilla_local")
    ]
    for parameterization, (label, color, marker) in STYLES.items():
        part = final[final.parameterization.eq(parameterization)].sort_values(
            "effective_sample_size"
        )
        ax_e.plot(part.effective_sample_size, part.mean_population_loss,
                  color=color, marker=marker, ms=3.2, lw=LW_DATA, label=label)
        ax_e.fill_between(part.effective_sample_size, part.ci95_low_population_loss,
                          part.ci95_high_population_loss, color=color, alpha=0.09,
                          linewidth=0)
    ax_e.set_xscale("log", base=2)
    ax_e.set_xticks(cfg["effective_sample_sizes"], [str(v) for v in cfg["effective_sample_sizes"]])
    ax_e.set_xlabel("effective sample size")
    ax_e.set_ylabel("loss after 80 updates")
    panel_title(ax_e, "E", "Final-loss crossover")
    style_axis(ax_e)

    pair_styles = [
        ("raw_nested", "raw $-$ Haar", COLORS["per_soma"], "s"),
        ("static_gain_scaled_nested", "scaled $-$ Haar", COLORS["additive"], "D"),
    ]
    prediction_final = predictions[
        predictions.checkpoint.eq(int(cfg["iterations"]))
        & predictions.optimizer.eq("vanilla_local")
    ]
    pred_means = prediction_final.groupby(
        ["effective_sample_size", "parameterization"]
    ).total_expected_loss.mean().unstack()
    for prefix, label, color, marker in pair_styles:
        part = contrasts[
            contrasts.left_minus_right.str.startswith(prefix)
            & contrasts.left_minus_right.str.endswith("tree_haar / vanilla_local")
        ].sort_values("effective_sample_size")
        ax_f.errorbar(
            part.effective_sample_size,
            part.mean_loss_difference,
            yerr=np.vstack([
                part.mean_loss_difference - part.ci95_low,
                part.ci95_high - part.mean_loss_difference,
            ]),
            color=color, marker=marker, ms=3.5, lw=LW_DATA, capsize=2,
            label=label,
        )
        parameterization = (
            "raw_nested_indicators" if prefix == "raw_nested"
            else "static_gain_scaled_nested"
        )
        predicted_difference = (
            pred_means[parameterization] - pred_means["tree_haar"]
        )
        ax_f.plot(predicted_difference.index, predicted_difference.values,
                  color=color, ls="--", lw=LW_REF)
    ax_f.axhline(0, color=COLORS["mute"], ls=":", lw=LW_REF)
    ax_f.set_xscale("log", base=2)
    ax_f.set_xticks(cfg["effective_sample_sizes"], [str(v) for v in cfg["effective_sample_sizes"]])
    ax_f.set_xlabel("effective sample size")
    ax_f.set_ylabel("nested loss $-$ Haar loss")
    panel_title(ax_f, "F", "Risk reversal")
    style_axis(ax_f)
    contrast_handles, contrast_labels = ax_f.get_legend_handles_labels()

    handles, labels = ax_c.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.51, 0.505), ncol=3,
               frameon=False, fontsize=PT_LEGEND - 0.4, handlelength=1.4,
               columnspacing=1.0)
    fig.legend(contrast_handles, contrast_labels, loc="lower center",
               bbox_to_anchor=(0.79, 0.012), ncol=2, frameon=False,
               fontsize=PT_LEGEND - 0.4, handlelength=1.3, columnspacing=0.9)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    audit_layout(fig, "fig_same_span_coefficient_learning")
    audit_text_over_data(fig, "fig_same_span_coefficient_learning")
    fig.savefig(FIGURES / "fig_same_span_coefficient_learning.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIGURES / "fig_same_span_coefficient_learning.png", dpi=600)
    plt.close(fig)


if __name__ == "__main__":
    main()

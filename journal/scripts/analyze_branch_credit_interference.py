#!/usr/bin/env python3
"""Derive and visualize a static branch-credit interference prediction.

This is not a temporal neuron model or a new causal experiment.  It evaluates
the exact one-step loss change for two quadratic task objectives at a fixed
operating point.  The calculation isolates how overlap between task routes and
leakage outside an addressed route produce forgetting of a previously learned
task.  It provides a quantitative bridge to published branch-specific motor
learning experiments while keeping that evidence explicitly external.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from neurips_style import (  # noqa: E402
    COLORS,
    LW_DATA,
    LW_EDGE,
    LW_REF,
    PT_ANNOT,
    PT_SMALL,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    clean_schematic_axis,
    grid_figure,
    panel_title,
    style_axis,
)


apply_neurips_style()
DEFAULT_OUTDIR = PROJECT / "source_data" / "branch_interference"
DEFAULT_STEM = PROJECT / "figures" / "generated" / "fig_branch_interference_prediction"


def exact_forgetting(overlap: float, leakage: float, eta: float) -> float:
    """Task-A loss increase after one approximate Task-B step.

    Both task routes occupy the same number of coordinates.  At the optimum of
    Task A, Task B requests an opposing update on their overlap.  Its routed
    field has unit gain on the Task-B route and leakage ``leakage`` onto the
    Task-A-only coordinates.  With loss normalized per Task-A coordinate,
    exact quadratic expansion gives the expression below.
    """
    overlap = float(overlap)
    leakage = float(leakage)
    eta = float(eta)
    return 0.5 * eta**2 * (4.0 * overlap + leakage**2 * (1.0 - overlap))


def numerical_forgetting(
    overlap: float, leakage: float, eta: float, n_route: int = 1000
) -> float:
    n_overlap = int(round(float(overlap) * n_route))
    route_a = np.arange(n_route)
    route_b = np.concatenate(
        [np.arange(n_overlap), np.arange(n_route, 2 * n_route - n_overlap)]
    )
    dimension = 2 * n_route
    w = np.zeros(dimension)
    w[route_a] = 1.0
    target_b = np.zeros(dimension)
    target_b[route_b] = -1.0
    gradient = np.zeros(dimension)
    gradient[route_b] = w[route_b] - target_b[route_b]
    a_only = np.setdiff1d(route_a, route_b, assume_unique=True)
    gradient[a_only] = float(leakage)
    before = 0.5 * np.mean((w[route_a] - 1.0) ** 2)
    after_w = w - float(eta) * gradient
    after = 0.5 * np.mean((after_w[route_a] - 1.0) ** 2)
    return float(after - before)


def _draw_routes(ax) -> None:
    clean_schematic_axis(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.plot([5, 5, 2.1, 5, 7.9], [0.8, 2.2, 5.4, 2.2, 5.4], color=COLORS["dend"], lw=3.2)
    ax.add_patch(Circle((5, 0.75), 0.42, fc=COLORS["soma"], ec=COLORS["edge"], lw=LW_EDGE))
    ax.plot([2.15, 4.15], [5.25, 3.15], color=COLORS["additive"], lw=5.2, alpha=0.72)
    ax.plot([7.85, 5.85], [5.25, 3.15], color=COLORS["inh"], lw=5.2, alpha=0.72)
    ax.text(1.85, 5.95, "Task A route", ha="center", color=COLORS["additive"], fontsize=PT_ANNOT, fontweight="bold")
    ax.text(8.15, 5.95, "Task B route", ha="center", color=COLORS["inh"], fontsize=PT_ANNOT, fontweight="bold")
    ax.add_patch(FancyArrowPatch((7.25, 4.55), (3.45, 3.95), arrowstyle="-|>", mutation_scale=8, linestyle="--", color=COLORS["highlight"], lw=1.2))
    ax.text(5.3, 4.65, "leakage", ha="center", color=COLORS["highlight"], fontsize=PT_ANNOT)
    ax.add_patch(FancyBboxPatch((2.1, 0.18), 5.8, 1.18, boxstyle="round,pad=0.12", fc="#F7F8FA", ec=COLORS["edge"], lw=LW_EDGE))
    box_style = dict(facecolor="#F7F8FA", edgecolor="none", pad=0.12)
    ax.text(5, 0.91, "forgetting depends on route overlap", ha="center", fontsize=PT_ANNOT, fontweight="bold", bbox=box_style)
    ax.text(5, 0.46, "and off-route credit leakage", ha="center", fontsize=PT_SMALL, bbox=box_style)


def _draw_animal_bridge(ax) -> None:
    clean_schematic_axis(ax)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    boxes = [
        (0.15, 4.45, 2.75, 1.45, "task-specific\nbranch events", COLORS["additive"]),
        (3.63, 4.45, 2.75, 1.45, "branch-local\nspine change", COLORS["shunting"]),
        (7.10, 4.45, 2.75, 1.45, "less cross-task\ninterference", COLORS["oracle"]),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12", fc="white", ec=color, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=PT_SMALL, fontweight="bold", color=color, bbox=dict(facecolor="white", edgecolor="none", pad=0.08))
    for left in [2.92, 6.40]:
        ax.add_patch(FancyArrowPatch((left, 5.18), (left + 0.75, 5.18), arrowstyle="-|>", mutation_scale=8, color=COLORS["mute"], lw=1.1))
    ax.add_patch(FancyBboxPatch((0.75, 1.05), 8.50, 1.72, boxstyle="round,pad=0.14", fc="#FAF4F7", ec=COLORS["highlight"], lw=LW_EDGE))
    prediction_box = dict(facecolor="#FAF4F7", edgecolor="none", pad=0.08)
    ax.text(5.0, 2.28, "prediction", ha="center", fontsize=PT_ANNOT, fontweight="bold", color=COLORS["highlight"], bbox=prediction_box)
    ax.text(5.0, 1.75, "loss of branch-selective inhibition increases", ha="center", fontsize=PT_SMALL, bbox=prediction_box)
    ax.text(5.0, 1.35, "effective leakage and cross-task interference", ha="center", fontsize=PT_SMALL, bbox=prediction_box)
    ax.text(5.0, 0.35, "published observations motivate the test;\nthey do not identify this mechanism uniquely", ha="center", va="center", fontsize=PT_SMALL, color=COLORS["mute"])


def make_figure(frame: pd.DataFrame, stem: Path, eta: float) -> None:
    fig, axes = grid_figure(2, 2, panel_h=2.0, gap_h=0.72, gap_w=1.02, margin_t=0.52)
    ax = axes.ravel()
    _draw_routes(ax[0])
    panel_title(ax[0], "A", "Route overlap creates interference")

    pivot = frame.pivot(index="leakage", columns="overlap", values="exact_forgetting")
    im = ax[1].imshow(
        pivot.to_numpy(), origin="lower", aspect="auto", cmap="magma",
        extent=[pivot.columns.min(), pivot.columns.max(), pivot.index.min(), pivot.index.max()],
    )
    ax[1].set_xlabel("route overlap")
    ax[1].set_ylabel("off-route leakage")
    cbar = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
    cbar.set_label("Task A loss increase")
    panel_title(ax[1], "B", "Exact one-step forgetting")
    style_axis(ax[1])

    for leakage, color in zip([0.0, 0.25, 0.5, 1.0], [COLORS["shunting"], COLORS["rule_3f"], COLORS["highlight"], COLORS["inh"]]):
        part = frame[np.isclose(frame.leakage, leakage)]
        ax[2].plot(part.overlap, part.exact_forgetting, color=color, lw=LW_DATA, label=f"leak={leakage:g}")
    ax[2].axhline(0, color=COLORS["mute"], lw=LW_REF, ls="--")
    ax[2].set_xlabel("route overlap")
    ax[2].set_ylabel("Task A loss increase")
    ax[2].legend(loc="upper left", frameon=False, ncol=2)
    panel_title(ax[2], "C", f"Interference at step size {eta:g}")
    style_axis(ax[2], grid="both")

    _draw_animal_bridge(ax[3])
    panel_title(ax[3], "D", "Link to branch-specific motor learning")

    audit_layout(fig, stem.name)
    audit_text_over_data(fig, stem.name)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(stem.with_suffix(".png"), dpi=350)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_STEM)
    parser.add_argument("--eta", type=float, default=0.2)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    overlaps = np.linspace(0.0, 1.0, 101)
    leakages = np.linspace(0.0, 1.0, 101)
    rows = []
    for leakage in leakages:
        for overlap in overlaps:
            exact = exact_forgetting(overlap, leakage, args.eta)
            numerical = numerical_forgetting(overlap, leakage, args.eta)
            rows.append({
                "overlap": overlap,
                "leakage": leakage,
                "step_size": args.eta,
                "exact_forgetting": exact,
                "numerical_forgetting": numerical,
                "absolute_verification_error": abs(exact - numerical),
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.outdir / "interference_surface.csv", index=False)
    summary = {
        "analysis": "static two-task quadratic interference",
        "evidence_level": "exact mathematical consequence and numerical verification",
        "step_size": args.eta,
        "maximum_absolute_verification_error": float(frame.absolute_verification_error.max()),
        "interpretation": "route overlap and off-route leakage are sufficient to produce forgetting; published animal observations are convergent but not mechanism-specific",
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_figure(frame, args.figure_stem, args.eta)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

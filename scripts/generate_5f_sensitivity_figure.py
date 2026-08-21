#!/usr/bin/env python3
"""Generate supplementary figure for 5F sensitivity analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from neurips_style import (  # noqa: E402
    # noqa: E402,
    COLORS,
    FIG_W,
    LW_HAIR,
    LW_REF,
    PT_SMALL,
    PT_TITLE,
    apply_neurips_style,
    grid_figure,
    panel_title,
)

DRAFT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = DRAFT_DIR / "figures"


def _tracked_or_analysis(tracked_name: str, *analysis_parts: str) -> Path:
    """Prefer the git-tracked figures/data/ copy; fall back to local analysis/."""
    tracked = DRAFT_DIR / "figures" / "data" / tracked_name
    return tracked if tracked.exists() else DRAFT_DIR.joinpath("analysis", *analysis_parts)


SUMMARY_CSV = _tracked_or_analysis(
    "five_factor_sensitivity_summary.csv",
    "five_factor_sensitivity", "five_factor_sensitivity_summary.csv",
)
COLOR_BASE = COLORS["shunting"]
COLOR_ALT = COLORS["pathway"]
COLOR_ACCENT = COLORS["local"]
W = 5.5
DPI = 300


def _setup_style() -> None:
    apply_neurips_style()


def _panel(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=PT_TITLE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=DPI)
    print(f"Saved {name}.{{pdf,png}}")


def _load_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {path}")
    return pd.read_csv(path)


def build_figure(summary_csv: Path = SUMMARY_CSV) -> plt.Figure:
    _setup_style()
    df = _load_summary(summary_csv)

    clamp = df[df["group"].isin(["baseline", "clamp_tight", "clamp_wide"])].copy()
    clamp = clamp.sort_values("plot_order")
    ema = df[df["group"].isin(["ema_alpha_005", "baseline", "ema_alpha_020"])].copy()
    ema["ema_label"] = ema["ema_alpha"].map(lambda v: f"{float(v):.2f}")
    ema = ema.sort_values("ema_alpha")

    fig, axes = grid_figure(2)

    ax = axes[0]
    _panel(ax, "A")
    x = np.arange(len(clamp))
    vals = 100.0 * clamp["test_accuracy_mean"].to_numpy()
    errs = 100.0 * clamp["test_accuracy_std"].fillna(0.0).to_numpy()
    colors = [COLOR_BASE if g == "baseline" else COLOR_ALT for g in clamp["group"]]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=LW_HAIR, width=0.62, zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=2, zorder=5)
    for xpos, val in zip(x, vals):
        ax.text(xpos, val + 0.35, f"{val:.1f}", ha="center", va="bottom", fontsize=PT_SMALL)
    ax.set_xticks(x)
    ax.set_xticklabels(clamp["display_label"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Clamp-bound sensitivity")
    ax.set_ylim(max(85, vals.min() - 2.5), min(94, vals.max() + 1.8))
    ax.grid(axis="y", alpha=0.22, linewidth=LW_HAIR, zorder=0)

    ax = axes[1]
    _panel(ax, "B")
    x = ema["ema_alpha"].to_numpy(dtype=float)
    vals = 100.0 * ema["test_accuracy_mean"].to_numpy()
    errs = 100.0 * ema["test_accuracy_std"].fillna(0.0).to_numpy()
    ax.plot(x, vals, color=COLOR_ACCENT, marker="o", markersize=4.5, linewidth=LW_REF, zorder=3)
    ax.fill_between(x, vals - errs, vals + errs, color=COLOR_ACCENT, alpha=0.16, zorder=2)
    baseline = ema[ema["group"] == "baseline"]
    if not baseline.empty:
        base_x = float(baseline["ema_alpha"].iloc[0])
        base_y = 100.0 * float(baseline["test_accuracy_mean"].iloc[0])
        ax.scatter([base_x], [base_y], s=34, color=COLOR_BASE, edgecolor="white", linewidth=LW_HAIR, zorder=4)
        ax.annotate("default", xy=(base_x, base_y), xytext=(6, 8), textcoords="offset points", fontsize=PT_SMALL)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in x])
    ax.set_xlabel(r"4F/5F EMA smoothing $\alpha$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("EMA-rate sensitivity")
    ax.set_ylim(max(85, vals.min() - 2.5), min(94, vals.max() + 1.8))
    ax.grid(alpha=0.22, linewidth=LW_HAIR)

    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.22, top=0.88, wspace=0.45)
    return fig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", type=Path, default=SUMMARY_CSV)
    args = parser.parse_args()
    fig = build_figure(args.summary_csv)
    _save(fig, "fig_s3_5f_sensitivity")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

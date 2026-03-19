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


DRAFT_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = DRAFT_DIR / "figures"
SUMMARY_CSV = (
    DRAFT_DIR / "analysis" / "five_factor_sensitivity" / "five_factor_sensitivity_summary.csv"
)
COLOR_BASE = "#18864B"
COLOR_ALT = "#7A3E9D"
COLOR_ACCENT = "#C65D1E"
W = 5.5
DPI = 300


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.titlepad": 6,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )


def _panel(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=11,
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

    fig, axes = plt.subplots(1, 2, figsize=(W, 2.7), gridspec_kw={"wspace": 0.45})

    ax = axes[0]
    _panel(ax, "A")
    x = np.arange(len(clamp))
    vals = 100.0 * clamp["test_accuracy_mean"].to_numpy()
    errs = 100.0 * clamp["test_accuracy_std"].fillna(0.0).to_numpy()
    colors = [COLOR_BASE if g == "baseline" else COLOR_ALT for g in clamp["group"]]
    ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.4, width=0.62, zorder=3)
    ax.errorbar(x, vals, yerr=errs, fmt="none", ecolor="#333333", elinewidth=0.6, capsize=2, zorder=5)
    for xpos, val in zip(x, vals):
        ax.text(xpos, val + 0.35, f"{val:.1f}", ha="center", va="bottom", fontsize=5.7)
    ax.set_xticks(x)
    ax.set_xticklabels(clamp["display_label"])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("5F is not overly sensitive to clamp bounds")
    ax.set_ylim(max(85, vals.min() - 2.5), min(94, vals.max() + 1.8))
    ax.grid(axis="y", alpha=0.22, linewidth=0.4, zorder=0)

    ax = axes[1]
    _panel(ax, "B")
    x = ema["ema_alpha"].to_numpy(dtype=float)
    vals = 100.0 * ema["test_accuracy_mean"].to_numpy()
    errs = 100.0 * ema["test_accuracy_std"].fillna(0.0).to_numpy()
    ax.plot(x, vals, color=COLOR_ACCENT, marker="o", markersize=4.5, linewidth=1.4, zorder=3)
    ax.fill_between(x, vals - errs, vals + errs, color=COLOR_ACCENT, alpha=0.16, zorder=2)
    baseline = ema[ema["group"] == "baseline"]
    if not baseline.empty:
        base_x = float(baseline["ema_alpha"].iloc[0])
        base_y = 100.0 * float(baseline["test_accuracy_mean"].iloc[0])
        ax.scatter([base_x], [base_y], s=34, color=COLOR_BASE, edgecolor="white", linewidth=0.5, zorder=4)
        ax.annotate("default", xy=(base_x, base_y), xytext=(6, 8), textcoords="offset points", fontsize=5.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.2f}" for v in x])
    ax.set_xlabel(r"4F/5F EMA smoothing $\alpha$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("5F is moderately robust to EMA rate")
    ax.set_ylim(max(85, vals.min() - 2.5), min(94, vals.max() + 1.8))
    ax.grid(alpha=0.22, linewidth=0.4)

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

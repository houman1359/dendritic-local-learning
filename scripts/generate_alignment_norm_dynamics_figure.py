#!/usr/bin/env python3
"""Generate alignment-over-training diagnostic with gradient norm checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from neurips_style import (  # noqa: E402
    # noqa: E402,
    COLORS,
    FIG_W,
    LW_DATA,
    LW_HAIR,
    PT_LEGEND,
    PT_TICK,
    REF_LW,
    apply_neurips_style,
    clean_legend,
    grid_figure,
    panel_title,
    style_axis,
)

apply_neurips_style()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "analysis" / "gradient_fidelity"
FIG_DIR = ROOT / "figures"


def _load() -> pd.DataFrame:
    frames = []
    for csv_path in sorted(DATA_DIR.glob("config_*/gradient_fidelity_trajectory.csv")):
        cfg_idx = int(csv_path.parent.name.split("_")[1])
        frame = pd.read_csv(csv_path)
        frame["network_type"] = "shunting" if cfg_idx < 3 else "additive"
        frame["seed"] = cfg_idx % 3
        frame["config_idx"] = cfg_idx
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No trajectory CSVs found under {DATA_DIR}")
    data = pd.concat(frames, ignore_index=True)
    return data[data["rule_variant"] == "5f"].copy()


def _weighted_trajectory(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (network_type, seed, epoch), group in data.groupby(["network_type", "seed", "epoch"]):
        weights = group["numel"].to_numpy(dtype=float)
        rows.append({
            "network_type": network_type,
            "seed": seed,
            "epoch": int(epoch),
            "weighted_cosine": np.average(group["cosine_similarity"], weights=weights),
            "local_grad_norm": np.average(group["local_grad_norm"], weights=weights),
            "backprop_grad_norm": np.average(group["backprop_grad_norm"], weights=weights),
            "norm_ratio": np.average(group["norm_ratio"], weights=weights),
        })
    return pd.DataFrame(rows)


def _summary(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    return (
        data.groupby(["network_type", "epoch"])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def _line_panel(ax, data: pd.DataFrame, metric: str, ylabel: str, title: str, *, letter: str = "", log_y: bool = False) -> None:
    summary = _summary(data, metric)
    for network_type, label, color in [
        ("additive", "Additive", COLORS["additive"]),
        ("shunting", "Shunting", COLORS["shunting"]),
    ]:
        sub = summary[summary["network_type"] == network_type].sort_values("epoch")
        x = sub["epoch"].to_numpy(dtype=float)
        y = sub["mean"].to_numpy(dtype=float)
        sd = sub["std"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, label=label, color=color, linewidth=LW_DATA)
        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.16, linewidth=0)
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel, fontsize=PT_LEGEND)
    panel_title(ax, letter, title)
    style_axis(ax, grid="y")


def main() -> None:
    data = _weighted_trajectory(_load())
    fig, axes_grid = grid_figure(2, 2)
    axes = axes_grid.ravel()

    _line_panel(axes[0], data, "weighted_cosine", "Weighted cosine", "Alignment", letter="A")
    axes[0].axhline(0.0, color=COLORS["edge"], linewidth=LW_HAIR, linestyle=":")
    axes[0].set_ylim(-0.18, 0.36)
    _line_panel(axes[1], data, "local_grad_norm", "Local grad norm", "Local updates", letter="B", log_y=True)
    _line_panel(axes[2], data, "backprop_grad_norm", "BP grad norm", "Backprop signal", letter="C", log_y=True)
    _line_panel(axes[3], data, "norm_ratio", "Local / BP norm", "Scale mismatch", letter="D", log_y=True)

    clean_legend(axes[0], loc="upper right", fontsize=PT_TICK)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "fig_s_alignment_norm_dynamics"
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=350)
    print(f"Saved {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()

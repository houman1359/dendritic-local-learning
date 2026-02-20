#!/usr/bin/env python3
"""Extract per-epoch learning curves from sweep results and produce publication figures.

For each sweep directory, reads per-epoch JSON files under
``results/config_*/performance/epochs/epoch*.json`` and the config to identify
conditions (strategy, rule_variant, use_shunting, etc.).

Outputs:
- ``learning_curves.csv``: tidy table (config_id, epoch, split, accuracy, condition_cols...)
- ``learning_curves_figure.pdf/png``: multi-panel figure grouped by condition
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_SWEEP_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)


def _dig(data: dict, keys: list[str], default=None):
    cur = data
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _safe_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_condition(cfg: dict) -> dict[str, Any]:
    """Pull out the experimental condition columns from a config."""
    strategy = _dig(cfg, ["training", "main", "strategy"], "unknown")
    local = _dig(cfg, ["training", "main", "learning_strategy_config"], {}) or {}
    arch = _dig(cfg, ["model", "core", "architecture"], {}) or {}
    return {
        "strategy": strategy,
        "core_type": _dig(cfg, ["model", "core", "type"], ""),
        "use_shunting": arch.get("use_shunting", None),
        "rule_variant": local.get("rule_variant", None),
        "error_broadcast_mode": local.get("error_broadcast_mode", None),
        "dataset": _dig(cfg, ["data", "dataset_name"], ""),
        "seed": _dig(cfg, ["experiment", "seed"], None),
    }


def extract_curves(sweep_dir: Path) -> pd.DataFrame:
    """Extract per-epoch accuracy curves for all configs in a sweep."""
    results_dir = sweep_dir / "results"
    if not results_dir.exists():
        return pd.DataFrame()

    rows = []
    for config_dir in sorted(results_dir.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
            continue

        cfg_blob = _safe_json(config_dir / "config.json")
        if cfg_blob is None:
            continue
        condition = extract_condition(cfg_blob)

        epochs_dir = config_dir / "performance" / "epochs"
        if not epochs_dir.exists():
            continue

        for ep_file in sorted(epochs_dir.glob("epoch*.json")):
            m = re.search(r"epoch(\d+)", ep_file.name)
            if not m:
                continue
            epoch = int(m.group(1))
            ep_data = _safe_json(ep_file)
            if ep_data is None:
                continue

            acc = ep_data.get("accuracy", {})
            for split in ("train", "valid", "test"):
                if split in acc:
                    row = {
                        "sweep": sweep_dir.name,
                        "config_id": config_dir.name,
                        "epoch": epoch,
                        "split": split,
                        "accuracy": acc[split],
                    }
                    row.update(condition)
                    rows.append(row)

    return pd.DataFrame(rows)


def plot_learning_curves(
    df: pd.DataFrame,
    group_col: str = "strategy",
    hue_col: str = "rule_variant",
    split: str = "test",
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot learning curves with mean +/- SEM across seeds."""
    sub = df[df["split"] == split].copy()
    if sub.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    # Build a composite label
    sub["condition"] = sub[group_col].astype(str)
    if hue_col and hue_col in sub.columns:
        mask = sub[hue_col].notna()
        sub.loc[mask, "condition"] = (
            sub.loc[mask, group_col].astype(str) + "/" + sub.loc[mask, hue_col].astype(str)
        )

    conditions = sorted(sub["condition"].unique())
    n_cond = len(conditions)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = plt.cm.tab10

    for i, cond in enumerate(conditions):
        cond_data = sub[sub["condition"] == cond]
        grouped = cond_data.groupby("epoch")["accuracy"]
        mean = grouped.mean()
        sem = grouped.sem().fillna(0)
        color = cmap(i % 10)
        ax.plot(mean.index, mean.values, label=cond, color=color, linewidth=1.5)
        ax.fill_between(
            mean.index,
            (mean - sem).values,
            (mean + sem).values,
            alpha=0.2,
            color=color,
        )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel(f"{split.capitalize()} Accuracy", fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        # Also save PDF
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=DEFAULT_SWEEP_ROOT,
        help="Root directory containing sweep_* folders",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default=None,
        help="Specific sweep folder name (if not set, process all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for CSV and figures",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="strategy",
        help="Column to group conditions by",
    )
    parser.add_argument(
        "--hue-col",
        type=str,
        default="rule_variant",
        help="Column for hue within groups",
    )
    args = parser.parse_args()

    sweep_root = args.sweep_root
    output_dir = args.output_dir or sweep_root.parent / "analysis" / "learning_curves"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect sweeps
    if args.sweep_name:
        sweep_dirs = [sweep_root / args.sweep_name]
    else:
        sweep_dirs = sorted(
            d for d in sweep_root.iterdir() if d.is_dir() and d.name.startswith("sweep_")
        )

    all_dfs = []
    for sd in sweep_dirs:
        print(f"Processing {sd.name}...")
        df = extract_curves(sd)
        if not df.empty:
            all_dfs.append(df)
            print(f"  -> {len(df)} rows, {df['config_id'].nunique()} configs")

    if not all_dfs:
        print("No data found.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    csv_path = output_dir / "learning_curves.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved {len(combined)} rows to {csv_path}")

    # Plot per-sweep
    for sweep_name in combined["sweep"].unique():
        sweep_df = combined[combined["sweep"] == sweep_name]
        fig_path = output_dir / f"learning_curves_{sweep_name}.png"
        plot_learning_curves(
            sweep_df,
            group_col=args.group_col,
            hue_col=args.hue_col,
            output_path=fig_path,
        )
        plt.close("all")
        print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()

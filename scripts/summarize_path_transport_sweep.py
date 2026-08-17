#!/usr/bin/env python3
"""Summarize the path-transport oracle sweep into run- and condition-level CSVs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_runs(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(results_dir.glob("config_*")):
        final_path = run_dir / "performance" / "final.json"
        config_path = run_dir / "config.json"
        if not final_path.exists() or not config_path.exists():
            continue
        final = _read_json(final_path)
        config = _read_json(config_path)
        acc = final.get("accuracy", {})
        rows.append(
            {
                "run_dir": str(run_dir),
                "dataset": config["data"]["dataset_name"],
                "network_type": config["model"]["core"]["type"],
                "ie_value": config["model"]["core"]["connectivity"][
                    "ie_synapses_per_branch_per_layer"
                ][0],
                "seed": config["experiment"]["seed"],
                "error_broadcast_mode": config["training"]["main"][
                    "learning_strategy_config"
                ]["error_broadcast_mode"],
                "rule_variant": config["training"]["main"]["learning_strategy_config"][
                    "rule_variant"
                ],
                "train_accuracy": acc.get("train"),
                "valid_accuracy": acc.get("valid"),
                "test_accuracy": acc.get("test"),
            }
        )
    return pd.DataFrame(rows)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["train_accuracy", "valid_accuracy", "test_accuracy"]
    out = (
        df.groupby(["dataset", "network_type", "ie_value", "error_broadcast_mode"])
        .agg(
            n_runs=("run_dir", "count"),
            **{
                f"{metric}_{stat}": (metric, stat)
                for metric in metrics
                for stat in ("mean", "std")
            },
        )
        .reset_index()
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    results_dir = args.sweep_dir / "results"
    runs = _collect_runs(results_dir)
    summary = _aggregate(runs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / "path_transport_upper_bound_runs.csv", index=False)
    summary.to_csv(args.output_dir / "path_transport_upper_bound_summary.csv", index=False)
    print(f"Saved summaries to {args.output_dir}")


if __name__ == "__main__":
    main()

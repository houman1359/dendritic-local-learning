#!/usr/bin/env python3
"""Summarize activation-audit sweeps from run directories.

Produces one CSV with per-run metadata and one grouped summary that conditions on:
dataset, training strategy, dendritic core, inhibition setting, and reactivation type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _get_nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _extract_first(value: Any, default: Any = None) -> Any:
    if isinstance(value, list) and value:
        return value[0]
    return value if value is not None else default


def _iter_run_dirs(sweep_dir: Path):
    results = sweep_dir / "results"
    if not results.exists():
        raise FileNotFoundError(f"Missing results dir under {sweep_dir}")
    for path in sorted(results.iterdir()):
        if path.is_dir() and path.name.startswith("config_"):
            yield path


def _row_from_run(run_dir: Path) -> dict[str, Any]:
    cfg = _read_json(run_dir / "config.json")
    perf_path = run_dir / "performance" / "final.json"
    if not perf_path.exists():
        raise FileNotFoundError(f"Incomplete run: missing {perf_path}")
    perf = _read_json(perf_path)
    acc = perf.get("accuracy", {}) if isinstance(perf, dict) else {}
    react = _get_nested(cfg, "model", "core", "reactivation", default={}) or {}
    return {
        "run_dir": str(run_dir),
        "seed": _get_nested(cfg, "experiment", "seed"),
        "dataset": _get_nested(cfg, "data", "dataset_name"),
        "strategy": _get_nested(cfg, "training", "main", "strategy"),
        "network_type": _get_nested(cfg, "model", "core", "type"),
        "ie_value": _extract_first(
            _get_nested(
                cfg,
                "model",
                "core",
                "connectivity",
                "ie_synapses_per_branch_per_layer",
                default=[],
            ),
            default=0,
        ),
        "reactivation_enabled": bool(react.get("enabled", True)),
        "reactivation_type": str(react.get("type", "none")),
        "reactivation_init_m": react.get("init_m"),
        "reactivation_init_b": react.get("init_b"),
        "reactivation_match_additive_init_to_shunting": bool(
            react.get("match_additive_init_to_shunting", False)
        ),
        "train_accuracy": acc.get("train"),
        "valid_accuracy": acc.get("valid"),
        "test_accuracy": acc.get("test"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    skipped = 0
    for run_dir in _iter_run_dirs(args.sweep_dir):
        try:
            rows.append(_row_from_run(run_dir))
        except FileNotFoundError:
            skipped += 1
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(
            f"No completed runs found under {args.sweep_dir}. "
            f"Skipped {skipped} incomplete run directories."
        )
    group_cols = ["dataset", "strategy", "network_type", "ie_value", "reactivation_type"]
    grouped = (
        frame.groupby(group_cols)
        .agg(
            n_runs=("run_dir", "count"),
            train_accuracy_mean=("train_accuracy", "mean"),
            train_accuracy_std=("train_accuracy", "std"),
            valid_accuracy_mean=("valid_accuracy", "mean"),
            valid_accuracy_std=("valid_accuracy", "std"),
            test_accuracy_mean=("test_accuracy", "mean"),
            test_accuracy_std=("test_accuracy", "std"),
        )
        .reset_index()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "activation_audit_runs.csv", index=False)
    grouped.to_csv(args.output_dir / "activation_audit_summary.csv", index=False)
    print(
        f"Saved activation audit summary to {args.output_dir} "
        f"({len(frame)} completed runs, {skipped} incomplete skipped)"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize the focused 5F sensitivity sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


SWEEP_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)
DRAFT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = DRAFT_DIR / "analysis" / "five_factor_sensitivity"


def _dig(blob: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = blob
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _latest_sweep(prefix: str) -> Path:
    matches = sorted(SWEEP_ROOT.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No sweep root found for prefix: {prefix}")
    return matches[-1]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_runs(sweep_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, Any]] = []
    for run_dir in sorted((sweep_dir / "results").glob("config_*")):
        config_path = run_dir / "config.json"
        perf_path = run_dir / "performance" / "final.json"
        if not config_path.exists() or not perf_path.exists():
            continue

        config = _load_json(config_path)
        perf = _load_json(perf_path)
        accuracy = perf.get("accuracy", {}) if isinstance(perf, dict) else {}
        phi_min = float(
            _dig(
                config,
                [
                    "training",
                    "main",
                    "learning_strategy_config",
                    "five_factor",
                    "phi_clamp_min",
                ],
                0.25,
            )
        )
        phi_max = float(
            _dig(
                config,
                [
                    "training",
                    "main",
                    "learning_strategy_config",
                    "five_factor",
                    "phi_clamp_max",
                ],
                4.0,
            )
        )
        ema_alpha = float(
            _dig(
                config,
                [
                    "training",
                    "main",
                    "learning_strategy_config",
                    "four_factor",
                    "ema_alpha",
                ],
                0.1,
            )
        )
        raw_group = config.get("_seed_repeat_group")
        group = str(raw_group) if raw_group not in {None, ""} else "unknown"
        if group == "unknown":
            if abs(phi_min - 0.5) < 1e-8 and abs(phi_max - 2.0) < 1e-8:
                group = "clamp_tight"
            elif abs(phi_min - 0.1) < 1e-8 and abs(phi_max - 8.0) < 1e-8:
                group = "clamp_wide"
            elif abs(ema_alpha - 0.05) < 1e-8:
                group = "ema_alpha_005"
            elif abs(ema_alpha - 0.2) < 1e-8:
                group = "ema_alpha_020"
            else:
                group = "baseline"

        records.append(
            {
                "group": group,
                "seed": _dig(config, ["experiment", "seed"]),
                "phi_clamp_min": phi_min,
                "phi_clamp_max": phi_max,
                "ema_alpha": ema_alpha,
                "train_accuracy": accuracy.get("train"),
                "valid_accuracy": accuracy.get("valid"),
                "test_accuracy": accuracy.get("test"),
                "run_dir": str(run_dir),
            }
        )

    runs = pd.DataFrame.from_records(records)
    if runs.empty:
        raise RuntimeError(f"No completed runs found under {sweep_dir}")

    summary = (
        runs.groupby(["group", "phi_clamp_min", "phi_clamp_max", "ema_alpha"], dropna=False)[
            ["train_accuracy", "valid_accuracy", "test_accuracy"]
        ]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        for col in summary.columns.to_flat_index()
    ]

    display_map = {
        "baseline": "default clamp\n[0.25, 4.0]",
        "clamp_tight": "tight clamp\n[0.5, 2.0]",
        "clamp_wide": "wide clamp\n[0.1, 8.0]",
        "ema_alpha_005": "EMA alpha 0.05",
        "ema_alpha_020": "EMA alpha 0.20",
    }
    summary["display_label"] = summary["group"].map(display_map).fillna(summary["group"])
    order_map = {
        "baseline": 0,
        "clamp_tight": 1,
        "clamp_wide": 2,
        "ema_alpha_005": 3,
        "ema_alpha_020": 4,
    }
    summary["plot_order"] = summary["group"].map(order_map).fillna(99)
    summary = summary.sort_values("plot_order").reset_index(drop=True)
    runs["display_label"] = runs["group"].map(display_map).fillna(runs["group"])
    return runs, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=None,
        help="Completed sweep directory. Defaults to latest sweep_neurips_5f_sensitivity_*",
    )
    args = parser.parse_args()

    sweep_dir = args.sweep_dir or _latest_sweep("sweep_neurips_5f_sensitivity")
    runs, summary = summarize_runs(sweep_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(OUTPUT_DIR / "five_factor_sensitivity_runs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "five_factor_sensitivity_summary.csv", index=False)
    print(f"Sweep dir: {sweep_dir}")
    print(f"Wrote: {OUTPUT_DIR / 'five_factor_sensitivity_runs.csv'}")
    print(f"Wrote: {OUTPUT_DIR / 'five_factor_sensitivity_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

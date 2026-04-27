#!/usr/bin/env python3
"""Summarize the one-layer inhibitory input-mode probe sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


DRAFT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_GLOB = "input_mode*_onelayer*_202*"


def _nested_get(obj: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def _load_final_accuracy(run_dir: Path) -> tuple[float | None, str]:
    final_json = run_dir / "performance" / "final.json"
    if not final_json.exists():
        if run_dir.exists():
            return None, "running_or_incomplete"
        return None, "pending"
    with final_json.open() as handle:
        payload = json.load(handle)
    accuracy = payload.get("accuracy", {})
    test_acc = accuracy.get("test")
    if test_acc is None:
        return None, "missing_test_accuracy"
    return float(test_acc), "complete"


def _condition_from_config(config: dict[str, Any]) -> dict[str, Any]:
    input_mode = int(_nested_get(config, "model.core.transfer.input_mode", -1))
    inhibitory_sizes = _nested_get(
        config, "model.core.architecture.inhibitory_layer_sizes", []
    )
    has_explicit_i = bool(inhibitory_sizes)
    strategy = _nested_get(config, "training.main.strategy", "")
    local_cfg = _nested_get(config, "training.main.learning_strategy_config", {}) or {}
    broadcast = local_cfg.get("error_broadcast_mode", "bp")
    explicit_i_mode = local_cfg.get(
        "explicit_inhibitory_update_mode",
        local_cfg.get("inhibitory_cell_update_mode"),
    )
    legacy_update_i = local_cfg.get("update_explicit_inhibitory_cells")
    if explicit_i_mode is None:
        if legacy_update_i is None:
            explicit_i_mode = "local_ca"
        else:
            explicit_i_mode = "local_ca" if bool(legacy_update_i) else "freeze"
    explicit_i_mode = str(explicit_i_mode).strip().lower().replace("-", "_")
    if explicit_i_mode in {"same_as_excitatory", "same_local_rule", "update", "train"}:
        explicit_i_mode = "local_ca"
    update_i = explicit_i_mode == "local_ca"

    if input_mode == 1:
        source = "direct_i_stream"
    elif has_explicit_i:
        source = "explicit_i_cells"
    else:
        source = "no_inhibition_source"

    if strategy == "standard":
        condition = f"{source}__standard_bp"
    elif source == "explicit_i_cells":
        condition = f"{source}__localca_{broadcast}__i_updates_{bool(update_i)}"
    else:
        condition = f"{source}__localca_{broadcast}"

    return {
        "condition": condition,
        "source": source,
        "strategy": strategy,
        "broadcast": broadcast,
        "explicit_inhibitory_update_mode": explicit_i_mode,
        "update_explicit_inhibitory_cells": update_i,
        "input_mode": input_mode,
        "excitatory_layers": json.dumps(
            _nested_get(config, "model.core.architecture.excitatory_layer_sizes", [])
        ),
        "inhibitory_layers": json.dumps(inhibitory_sizes),
        "excitatory_morphology": json.dumps(
            _nested_get(config, "model.core.architecture.excitatory_branch_factors", [])
        ),
        "inhibitory_morphology": json.dumps(
            _nested_get(config, "model.core.architecture.inhibitory_branch_factors", [])
        ),
    }


def _summarize_sweep(sweep_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_path in sorted((sweep_dir / "configs").glob("unified_config_*.yaml")):
        config = _load_yaml(config_path)
        config_id = config_path.stem.replace("unified_", "")
        result_dir = Path(_nested_get(config, "outputs.results_dir", ""))
        test_acc, status = _load_final_accuracy(result_dir)
        row = {
            "sweep_dir": str(sweep_dir),
            "config_id": config_id,
            "seed": _nested_get(config, "experiment.seed", ""),
            "status": status,
            "test_accuracy": "" if test_acc is None else f"{test_acc:.10f}",
        }
        row.update(_condition_from_config(config))
        rows.append(row)
    return rows


def _group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row["condition"]
        meta.setdefault(
            key,
            {
                "condition": row["condition"],
                "source": row["source"],
                "strategy": row["strategy"],
                "broadcast": row["broadcast"],
                "explicit_inhibitory_update_mode": row[
                    "explicit_inhibitory_update_mode"
                ],
                "update_explicit_inhibitory_cells": row[
                    "update_explicit_inhibitory_cells"
                ],
                "input_mode": row["input_mode"],
                "excitatory_layers": row["excitatory_layers"],
                "inhibitory_layers": row["inhibitory_layers"],
                "excitatory_morphology": row["excitatory_morphology"],
                "inhibitory_morphology": row["inhibitory_morphology"],
            },
        )
        if row["status"] == "complete" and row["test_accuracy"]:
            grouped[key].append(float(row["test_accuracy"]))

    summary = []
    for key in sorted(meta):
        values = grouped.get(key, [])
        mean = statistics.fmean(values) if values else math.nan
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        entry = {
            **meta[key],
            "n_complete": len(values),
            "test_accuracy_mean": "" if math.isnan(mean) else f"{mean:.10f}",
            "test_accuracy_std": "" if math.isnan(mean) else f"{std:.10f}",
        }
        summary.append(entry)
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _latest_probe_dirs() -> list[Path]:
    run_root = DRAFT_DIR / "local_sweep_runs"
    candidates = sorted(
        path
        for path in run_root.glob(DEFAULT_SWEEP_GLOB)
        if path.is_dir() and (path / "configs").exists()
    )
    latest: dict[str, Path] = {}
    for path in candidates:
        prefix = path.name.rsplit("_", 1)[0]
        latest[prefix] = path
    return sorted(latest.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, action="append")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DRAFT_DIR
        / "analysis"
        / f"input_mode_onelayer_probe_summary_{datetime.now():%Y%m%d}",
    )
    args = parser.parse_args()

    sweep_dirs = args.sweep_dir or _latest_probe_dirs()
    if not sweep_dirs:
        raise SystemExit("No one-layer input-mode probe sweep directories found.")

    rows: list[dict[str, Any]] = []
    for sweep_dir in sweep_dirs:
        rows.extend(_summarize_sweep(sweep_dir.resolve()))

    grouped = _group_rows(rows)
    _write_csv(args.out_dir / "input_mode_onelayer_detailed.csv", rows)
    _write_csv(args.out_dir / "input_mode_onelayer_grouped.csv", grouped)

    print(f"Wrote {len(rows)} detailed rows and {len(grouped)} grouped rows")
    print(args.out_dir)
    for row in grouped:
        mean = row["test_accuracy_mean"] or "pending"
        std = row["test_accuracy_std"] or ""
        print(
            f"{row['condition']}: n={row['n_complete']} "
            f"test={mean}{' +/- ' + std if std else ''}"
        )


if __name__ == "__main__":
    main()

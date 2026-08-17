#!/usr/bin/env python
"""Summarize generated repeated-config sweep results.

The repeated sweep launcher writes one trained run per results/config_<idx>.
This script reads the saved config.json and performance/final.json files,
then emits a detailed table plus grouped means by condition. It is intentionally
generic so new paper-facing sweeps can be summarized without another custom
parser.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DRAFT_ROOT = REPO_ROOT / "drafts" / "dendritic-local-learning"
LOCAL_RUNS = DRAFT_ROOT / "local_sweep_runs"
ANALYSIS = DRAFT_ROOT / "analysis"


def _latest_sweep(prefix: str) -> Path:
    matches = sorted(LOCAL_RUNS.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No sweep root matching {prefix!r} under {LOCAL_RUNS}")
    return matches[-1]


def _strip_seed(run_name: str) -> tuple[str, int | None]:
    match = re.match(r"(.+)_s(\d+)$", run_name)
    if match:
        return match.group(1), int(match.group(2))
    return run_name, None


def _get_nested(data: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open() as handle:
        return json.load(handle)


def _condition_metadata(cfg: dict[str, Any], condition: str) -> dict[str, Any]:
    learning_rule = _get_nested(
        cfg, ("training", "learning_rule"), _get_nested(cfg, ("training", "main", "strategy"))
    )
    local_ca_cfg = _get_nested(cfg, ("training", "local_credit_assignment"))
    if local_ca_cfg is None:
        local_ca_cfg = _get_nested(cfg, ("training", "main", "learning_strategy_config"), {})
    broadcast_mode = _get_nested(local_ca_cfg, ("error_broadcast_mode",))
    rule_variant = _get_nested(local_ca_cfg, ("rule_variant",))
    decoder_update_mode = _get_nested(local_ca_cfg, ("decoder_update_mode",))
    hsic_cfg = _get_nested(local_ca_cfg, ("hsic",), {})
    hsic_enabled = _get_nested(hsic_cfg, ("enabled",))
    hsic_weight = _get_nested(hsic_cfg, ("weight",))
    dataset = _get_nested(cfg, ("data", "dataset_name"))
    normalize = _get_nested(cfg, ("data", "processing", "normalize"))
    use_shunting = _get_nested(cfg, ("model", "core", "use_shunting"))
    core_type = _get_nested(cfg, ("model", "core", "type"))
    reactivation_enabled = _get_nested(
        cfg, ("model", "core", "reactivation", "enabled")
    )
    ie_synapses = _get_nested(
        cfg, ("model", "core", "connectivity", "ie_synapses_per_branch_per_layer"), []
    )
    ii_synapses = _get_nested(
        cfg, ("model", "core", "connectivity", "ii_synapses_per_branch_per_layer"), []
    )
    additive_gain_mode = _get_nested(local_ca_cfg, ("three_factor", "additive_gain_mode"))
    dendritic_norm = _get_nested(
        local_ca_cfg, ("morphology_aware", "use_dendritic_normalization")
    )
    broadcast_rank = _get_nested(local_ca_cfg, ("broadcast_rank",))

    has_i_to_e = bool(ie_synapses) and any(float(v) > 0 for v in ie_synapses)
    has_i_to_i = bool(ii_synapses) and any(float(v) > 0 for v in ii_synapses)
    if use_shunting is not None:
        core = "shunting" if use_shunting else "additive"
    elif core_type == "dendritic_shunting":
        core = "shunting"
    elif core_type == "dendritic_additive":
        core = "additive"
    elif core_type == "dendritic_normalized_additive":
        core = "normalized_additive"
    else:
        core = core_type

    return {
        "condition": condition,
        "dataset": dataset,
        "learning_rule": learning_rule,
        "rule_variant": rule_variant,
        "broadcast_mode": broadcast_mode,
        "broadcast_rank": broadcast_rank,
        "decoder_update_mode": decoder_update_mode,
        "hsic_enabled": hsic_enabled,
        "hsic_weight": hsic_weight,
        "core": core,
        "core_type": core_type,
        "normalize_inputs": normalize,
        "has_i_to_e": has_i_to_e,
        "has_i_to_i": has_i_to_i,
        "additive_gain_mode": additive_gain_mode,
        "dendritic_normalization": bool(dendritic_norm),
        "reactivation_enabled": reactivation_enabled,
    }


def summarize_sweep(sweep_root: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_dir = sweep_root / "results"
    config_dirs = sorted(
        results_dir.glob("config_*"),
        key=lambda p: int(p.name.split("_", maxsplit=1)[1]) if p.name.split("_")[-1].isdigit() else p.name,
    )

    rows: list[dict[str, Any]] = []
    for config_dir in config_dirs:
        index_match = re.match(r"config_(\d+)$", config_dir.name)
        if not index_match:
            continue
        config_index = int(index_match.group(1))
        cfg = _load_json(config_dir / "config.json")
        if cfg is None:
            rows.append(
                {
                    "config_index": config_index,
                    "status": "missing_config",
                    "run_name": config_dir.name,
                    "seed": None,
                    "condition": config_dir.name,
                    "result_dir": str(config_dir),
                }
            )
            continue

        run_name = _get_nested(cfg, ("outputs", "run_name"), config_dir.name)
        condition, parsed_seed = _strip_seed(str(run_name))
        seed = _get_nested(cfg, ("experiment", "seed"), parsed_seed)
        final = _load_json(config_dir / "performance" / "final.json")
        status = "complete" if final is not None else "incomplete"

        row: dict[str, Any] = {
            "config_index": config_index,
            "status": status,
            "run_name": run_name,
            "seed": seed,
            "result_dir": str(config_dir),
        }
        row.update(_condition_metadata(cfg, condition))

        if final is not None:
            accuracy = final.get("accuracy", {})
            loss = final.get("loss", {})
            row.update(
                {
                    "train_accuracy": accuracy.get("train"),
                    "valid_accuracy": accuracy.get("valid"),
                    "test_accuracy": accuracy.get("test"),
                    "train_loss": loss.get("train"),
                    "valid_loss": loss.get("valid"),
                    "test_loss": loss.get("test"),
                }
            )
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        detailed = pd.DataFrame(rows).sort_values(["condition", "seed", "config_index"])
    else:
        detailed = pd.DataFrame(
            columns=[
                "config_index",
                "status",
                "run_name",
                "seed",
                "condition",
                "dataset",
                "core",
                "learning_rule",
                "rule_variant",
                "broadcast_mode",
                "broadcast_rank",
                "test_accuracy",
                "result_dir",
            ]
        )
    detailed.to_csv(out_dir / "detailed_results.csv", index=False)

    complete = detailed[detailed["status"] == "complete"].copy()
    if complete.empty:
        grouped = pd.DataFrame()
    else:
        grouped = (
            complete.groupby(
                [
                    "condition",
                    "dataset",
                    "core",
                    "learning_rule",
                    "rule_variant",
                    "broadcast_mode",
                    "broadcast_rank",
                    "decoder_update_mode",
                    "hsic_enabled",
                    "hsic_weight",
                    "core_type",
                    "normalize_inputs",
                    "has_i_to_e",
                    "additive_gain_mode",
                    "dendritic_normalization",
                    "reactivation_enabled",
                ],
                dropna=False,
            )["test_accuracy"]
            .agg(["mean", "std", "min", "max", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": "test_acc_mean",
                    "std": "test_acc_std",
                    "min": "test_acc_min",
                    "max": "test_acc_max",
                    "count": "n_seeds",
                }
            )
            .sort_values(["dataset", "condition"])
        )
    grouped.to_csv(out_dir / "grouped_summary.csv", index=False)
    return detailed, grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root_arg = parser.add_mutually_exclusive_group(required=True)
    root_arg.add_argument("--sweep-root", type=Path, help="Path to generated sweep directory.")
    root_arg.add_argument("--latest-prefix", help="Use latest local_sweep_runs/<prefix>_* directory.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Output directory. Defaults to analysis/<sweep-name>_summary_<date>.",
    )
    args = parser.parse_args()

    sweep_root = args.sweep_root if args.sweep_root is not None else _latest_sweep(args.latest_prefix)
    sweep_root = sweep_root.resolve()
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = ANALYSIS / f"{sweep_root.name}_summary_{date.today().strftime('%Y%m%d')}"

    detailed, grouped = summarize_sweep(sweep_root, out_dir)
    n_complete = int((detailed["status"] == "complete").sum()) if not detailed.empty else 0
    print(f"Sweep: {sweep_root}")
    print(f"Detailed rows: {len(detailed)} ({n_complete} complete)")
    print(f"Grouped rows: {len(grouped)}")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()

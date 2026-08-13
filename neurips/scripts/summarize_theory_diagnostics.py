#!/usr/bin/env python3
"""Aggregate theory-diagnostic runs into condition-level summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
DRAFT_DIR = REPO_ROOT / "drafts" / "dendritic-local-learning"
DEFAULT_COMBINED = DRAFT_DIR / "analysis" / "combined_results.csv"


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


def _resolve_run_dir(run_dir: Path) -> Path:
    if run_dir.exists():
        return run_dir
    if run_dir.is_absolute():
        return run_dir

    candidates = [
        Path.cwd() / run_dir,
        REPO_ROOT / run_dir,
        DRAFT_DIR / run_dir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return run_dir


def _metadata_from_run(run_dir: Path) -> dict[str, Any]:
    config = _read_json(run_dir / "config.json")
    final = _read_json(run_dir / "performance" / "final.json")

    core_type = _get_nested(config, "model", "core", "type", default="unknown")
    ie_value = _extract_first(
        _get_nested(
            config,
            "model",
            "core",
            "connectivity",
            "ie_synapses_per_branch_per_layer",
            default=None,
        ),
        default=0,
    )
    ee_value = _extract_first(
        _get_nested(
            config,
            "model",
            "core",
            "connectivity",
            "ee_synapses_per_branch_per_layer",
            default=None,
        ),
        default=None,
    )
    local_cfg = _get_nested(
        config,
        "training",
        "main",
        "learning_strategy_config",
        default={},
    )
    reactivation_cfg = _get_nested(
        config,
        "model",
        "core",
        "reactivation",
        default={},
    )

    accuracy = final.get("accuracy", {}) if isinstance(final, dict) else {}
    return {
        "run_dir": str(run_dir),
        "dataset": _get_nested(config, "data", "dataset_name", default="unknown"),
        "network_type": str(core_type).lower(),
        "ie_value": ie_value,
        "ee_value": ee_value,
        "reactivation_type": reactivation_cfg.get("type", "none"),
        "seed": _get_nested(config, "experiment", "seed", default=None),
        "rule_variant": local_cfg.get("rule_variant"),
        "error_broadcast_mode": local_cfg.get("error_broadcast_mode"),
        "train_accuracy": accuracy.get("train"),
        "valid_accuracy": accuracy.get("valid"),
        "test_accuracy": accuracy.get("test"),
    }


def _collect_run_metadata(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir_str in summary["run_dir"].astype(str).tolist():
        run_dir = Path(run_dir_str)
        resolved_run_dir = _resolve_run_dir(run_dir)
        try:
            metadata = _metadata_from_run(resolved_run_dir)
            metadata["run_dir"] = run_dir_str
            metadata["resolved_run_dir"] = str(resolved_run_dir)
            rows.append(metadata)
        except Exception as exc:
            rows.append(
                {
                    "run_dir": run_dir_str,
                    "metadata_error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def _aggregate(frame: pd.DataFrame, metric_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    aggregations = {"run_dir": "count"}
    for col in metric_cols:
        aggregations[col] = ["mean", "std"]
    out = frame.groupby(group_cols).agg(aggregations)
    out.columns = [
        "n_runs" if col == "run_dir" and stat == "count" else f"{col}_{stat}"
        for col, stat in out.columns
    ]
    return out.reset_index()


def _select_group_cols(frame: pd.DataFrame) -> list[str]:
    base_cols = ["dataset", "network_type", "ie_value"]
    optional_cols = ["ee_value", "reactivation_type", "rule_variant", "error_broadcast_mode"]
    group_cols = list(base_cols)
    for col in optional_cols:
        if col not in frame.columns:
            continue
        non_null = frame[col].dropna()
        if non_null.empty:
            continue
        if non_null.nunique() > 1:
            group_cols.append(col)
    return group_cols


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diag-dir", type=Path, required=True)
    parser.add_argument("--combined-results", type=Path, default=DEFAULT_COMBINED)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = pd.read_csv(args.diag_dir / "run_summary.csv")
    metadata = _collect_run_metadata(summary)
    merged = summary.merge(metadata, on="run_dir", how="left")

    merged["path_transport_cosine_gain"] = (
        merged["path_transport_weighted_cosine"] - merged["per_soma_weighted_cosine"]
    )
    merged["path_factor_scale_gain"] = (
        merged["per_soma_weighted_scale_mismatch"]
        - merged["path_factor_scalar_weighted_scale_mismatch"]
    )
    merged["per_soma_over_scalar_cosine_gain"] = (
        merged["per_soma_weighted_cosine"] - merged["scalar_weighted_cosine"]
    )

    metric_cols = [
        "factorization_weighted_cosine",
        "factorization_weighted_scale_mismatch",
        "path_gain_cv_mean",
        "scalar_weighted_cosine",
        "scalar_weighted_scale_mismatch",
        "per_soma_weighted_cosine",
        "per_soma_weighted_scale_mismatch",
        "path_factor_scalar_weighted_cosine",
        "path_factor_scalar_weighted_scale_mismatch",
        "path_transport_weighted_cosine",
        "path_transport_weighted_scale_mismatch",
        "path_transport_cosine_gain",
        "path_factor_scale_gain",
        "per_soma_over_scalar_cosine_gain",
        "test_accuracy",
        "valid_accuracy",
        "train_accuracy",
    ]
    group_cols = _select_group_cols(merged)
    by_condition = _aggregate(merged, metric_cols, group_cols)

    corr_rows: list[dict[str, float | str]] = []
    corr_specs = [
        ("path_gain_cv_mean", "per_soma_weighted_cosine"),
        ("path_gain_cv_mean", "path_transport_weighted_cosine"),
        ("per_soma_weighted_cosine", "test_accuracy"),
        ("path_transport_weighted_cosine", "test_accuracy"),
        ("path_factor_scalar_weighted_scale_mismatch", "test_accuracy"),
    ]
    for (dataset, network_type), group in merged.groupby(["dataset", "network_type"]):
        for x_col, y_col in corr_specs:
            valid = group[[x_col, y_col]].dropna()
            if len(valid) < 2:
                continue
            corr_rows.append(
                {
                    "dataset": dataset,
                    "network_type": network_type,
                    "x_metric": x_col,
                    "y_metric": y_col,
                    "pearson_r": float(valid[x_col].corr(valid[y_col])),
                    "n_runs": int(len(valid)),
                }
            )
    correlations = pd.DataFrame(corr_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output_dir / "theory_diag_merged_runs.csv", index=False)
    by_condition.to_csv(args.output_dir / "theory_diag_by_condition.csv", index=False)
    correlations.to_csv(args.output_dir / "theory_diag_correlations.csv", index=False)
    print(f"Saved summaries to {args.output_dir}")


if __name__ == "__main__":
    main()

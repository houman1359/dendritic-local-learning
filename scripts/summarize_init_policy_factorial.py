#!/usr/bin/env python3
"""Summarize the matched 3F architecture-by-initialization-policy factorial."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


SHUNTING = "dendritic_shunting"
ADDITIVE = "dendritic_additive"
ANALYTICAL = "analytical"
OCCUPANCY = "occupancy_quantile"


def _read_runs(root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((root / "results").glob("config_*")):
        config_path = run_dir / "config.json"
        result_path = run_dir / "performance" / "final.json"
        if not config_path.exists() or not result_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        calibration_path = run_dir / "post_calibration_gate_stats.json"
        calibration = (
            json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration_path.exists()
            else {}
        )
        core = config["model"]["core"]
        local_cfg = config["training"]["main"]["learning_strategy_config"]
        rows.append(
            {
                "run_dir": str(run_dir),
                "seed": int(config["experiment"]["seed"]),
                "network_type": str(core["type"]),
                "init_policy": str(core["reactivation"]["init_policy"]),
                "rule_variant": str(local_cfg["rule_variant"]),
                "broadcast_mode": str(local_cfg["error_broadcast_mode"]),
                "decoder_update_mode": str(local_cfg["decoder_update_mode"]),
                "post_calibration_present": calibration_path.exists(),
                "post_calibration_warning_count": len(
                    calibration.get("warnings", [])
                ),
                "test_accuracy": float(result["accuracy"]["test"]),
            }
        )
    return pd.DataFrame(rows)


def _paired_test(
    left: np.ndarray,
    right: np.ndarray,
    *,
    comparison: str,
    left_label: str,
    right_label: str,
) -> dict[str, Any]:
    paired_t = ttest_rel(left, right)
    try:
        signed_rank_p = float(wilcoxon(left, right).pvalue)
    except ValueError:
        signed_rank_p = float("nan")
    difference = left - right
    return {
        "comparison": comparison,
        "left": left_label,
        "right": right_label,
        "n_pairs": int(len(difference)),
        "left_mean": float(np.mean(left)),
        "left_std": float(np.std(left, ddof=1)),
        "right_mean": float(np.mean(right)),
        "right_std": float(np.std(right, ddof=1)),
        "paired_difference_mean": float(np.mean(difference)),
        "paired_difference_std": float(np.std(difference, ddof=1)),
        "left_wins": int(np.sum(difference > 0)),
        "ties": int(np.sum(difference == 0)),
        "paired_t_pvalue": float(paired_t.pvalue),
        "wilcoxon_pvalue": signed_rank_p,
    }


def _paired_rows(details: pd.DataFrame) -> pd.DataFrame:
    pivot = details.pivot_table(
        index="seed",
        columns=["network_type", "init_policy"],
        values="test_accuracy",
        aggfunc="mean",
    ).dropna()
    required = {
        (SHUNTING, ANALYTICAL),
        (SHUNTING, OCCUPANCY),
        (ADDITIVE, ANALYTICAL),
        (ADDITIVE, OCCUPANCY),
    }
    if not required.issubset(set(pivot.columns)):
        missing = sorted(required - set(pivot.columns))
        raise ValueError(f"Factorial is incomplete; missing conditions: {missing}")

    rows = [
        _paired_test(
            pivot[(SHUNTING, ANALYTICAL)].to_numpy(dtype=float),
            pivot[(ADDITIVE, ANALYTICAL)].to_numpy(dtype=float),
            comparison="architecture_within_analytical",
            left_label="shunting_analytical",
            right_label="additive_analytical",
        ),
        _paired_test(
            pivot[(SHUNTING, OCCUPANCY)].to_numpy(dtype=float),
            pivot[(ADDITIVE, OCCUPANCY)].to_numpy(dtype=float),
            comparison="architecture_within_occupancy_quantile",
            left_label="shunting_occupancy_quantile",
            right_label="additive_occupancy_quantile",
        ),
        _paired_test(
            pivot[(SHUNTING, OCCUPANCY)].to_numpy(dtype=float),
            pivot[(SHUNTING, ANALYTICAL)].to_numpy(dtype=float),
            comparison="policy_within_shunting",
            left_label="shunting_occupancy_quantile",
            right_label="shunting_analytical",
        ),
        _paired_test(
            pivot[(ADDITIVE, OCCUPANCY)].to_numpy(dtype=float),
            pivot[(ADDITIVE, ANALYTICAL)].to_numpy(dtype=float),
            comparison="policy_within_additive",
            left_label="additive_occupancy_quantile",
            right_label="additive_analytical",
        ),
    ]

    shunting_gap = (
        pivot[(SHUNTING, OCCUPANCY)] - pivot[(SHUNTING, ANALYTICAL)]
    ).to_numpy(dtype=float)
    additive_gap = (
        pivot[(ADDITIVE, OCCUPANCY)] - pivot[(ADDITIVE, ANALYTICAL)]
    ).to_numpy(dtype=float)
    rows.append(
        _paired_test(
            shunting_gap,
            additive_gap,
            comparison="architecture_by_policy_interaction",
            left_label="shunting_policy_effect",
            right_label="additive_policy_effect",
        )
    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-seeds",
        type=int,
        help="Require this many unique seeds in every factorial cell.",
    )
    args = parser.parse_args()

    details = _read_runs(args.sweep_root)
    if details.empty:
        raise RuntimeError(f"No completed runs found under {args.sweep_root}")
    duplicate_keys = ["seed", "network_type", "init_policy"]
    if details.duplicated(duplicate_keys).any():
        duplicates = details.loc[
            details.duplicated(duplicate_keys, keep=False), duplicate_keys
        ]
        raise ValueError(f"Duplicate factorial cells:\n{duplicates}")
    if args.expected_seeds is not None:
        counts = details.groupby(["network_type", "init_policy"])["seed"].nunique()
        bad_counts = counts[counts != args.expected_seeds]
        if not bad_counts.empty:
            raise ValueError(
                "Incomplete factorial cells; expected "
                f"{args.expected_seeds} seeds:\n{bad_counts}"
            )

    expected = {SHUNTING, ADDITIVE}
    if set(details["network_type"]) != expected:
        raise ValueError(
            f"Expected architectures {sorted(expected)}, got "
            f"{sorted(details['network_type'].unique())}"
        )
    expected_policies = {ANALYTICAL, OCCUPANCY}
    if set(details["init_policy"]) != expected_policies:
        raise ValueError(
            f"Expected policies {sorted(expected_policies)}, got "
            f"{sorted(details['init_policy'].unique())}"
        )

    summary = (
        details.groupby(["network_type", "init_policy"], as_index=False)[
            "test_accuracy"
        ]
        .agg(["count", "mean", "std", "sem"])
    )
    paired = _paired_rows(details)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details.sort_values(duplicate_keys).to_csv(
        args.output_dir / "seed_level.csv", index=False
    )
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    paired.to_csv(args.output_dir / "paired_tests.csv", index=False)

    print(summary.to_string(index=False))
    print()
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize a matched scalar-fallback versus ancestry-shared feedback test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def _read_runs(root: Path, feedback_label: str | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted((root / "results").glob("config_*")):
        config_path = run_dir / "config.json"
        result_path = run_dir / "performance" / "final.json"
        if not config_path.exists() or not result_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        result = json.loads(result_path.read_text(encoding="utf-8"))
        local_cfg = config["training"]["main"]["learning_strategy_config"]
        broadcast_mode = str(local_cfg["error_broadcast_mode"])
        resolved_feedback = feedback_label
        if resolved_feedback is None:
            if broadcast_mode == "per_soma":
                resolved_feedback = "scalar_fallback"
            elif broadcast_mode in {"per_soma_shared", "per_soma_tree"}:
                resolved_feedback = "ancestry_shared"
            else:
                continue
        rows.append(
            {
                "feedback": resolved_feedback,
                "run_dir": str(run_dir),
                "seed": int(config["experiment"]["seed"]),
                "network_type": str(config["model"]["core"]["type"]),
                "rule_variant": str(local_cfg["rule_variant"]),
                "broadcast_mode": broadcast_mode,
                "decoder_update_mode": str(local_cfg["decoder_update_mode"]),
                "test_accuracy": float(result["accuracy"]["test"]),
            }
        )
    return pd.DataFrame(rows)


def _paired_row(
    frame: pd.DataFrame,
    *,
    comparison: str,
    column: str,
    left: str,
    right: str,
) -> dict[str, Any] | None:
    pivot = frame.pivot_table(
        index="seed",
        columns=column,
        values="test_accuracy",
        aggfunc="mean",
    ).dropna()
    if left not in pivot or right not in pivot or pivot.empty:
        return None
    left_values = pivot[left].to_numpy(dtype=float)
    right_values = pivot[right].to_numpy(dtype=float)
    paired_t = ttest_rel(left_values, right_values)
    try:
        signed_rank_p = float(wilcoxon(left_values, right_values).pvalue)
    except ValueError:
        signed_rank_p = float("nan")
    return {
        "comparison": comparison,
        "left": left,
        "right": right,
        "n_pairs": int(len(pivot)),
        "left_mean": float(np.mean(left_values)),
        "right_mean": float(np.mean(right_values)),
        "paired_difference_mean": float(np.mean(left_values - right_values)),
        "paired_t_pvalue": float(paired_t.pvalue),
        "wilcoxon_pvalue": signed_rank_p,
    }


def _interaction_row(details: pd.DataFrame) -> dict[str, Any] | None:
    pivot = details.pivot_table(
        index="seed",
        columns=["feedback", "network_type"],
        values="test_accuracy",
        aggfunc="mean",
    ).dropna()
    needed = {
        ("ancestry_shared", "dendritic_shunting"),
        ("ancestry_shared", "dendritic_additive"),
        ("scalar_fallback", "dendritic_shunting"),
        ("scalar_fallback", "dendritic_additive"),
    }
    if not needed.issubset(set(pivot.columns)) or pivot.empty:
        return None
    ancestry_gap = (
        pivot[("ancestry_shared", "dendritic_shunting")]
        - pivot[("ancestry_shared", "dendritic_additive")]
    ).to_numpy(dtype=float)
    scalar_gap = (
        pivot[("scalar_fallback", "dendritic_shunting")]
        - pivot[("scalar_fallback", "dendritic_additive")]
    ).to_numpy(dtype=float)
    test = ttest_rel(ancestry_gap, scalar_gap)
    try:
        signed_rank_p = float(wilcoxon(ancestry_gap, scalar_gap).pvalue)
    except ValueError:
        signed_rank_p = float("nan")
    return {
        "comparison": "feedback_by_architecture_interaction",
        "left": "ancestry_shunting_minus_additive",
        "right": "scalar_fallback_shunting_minus_additive",
        "n_pairs": int(len(pivot)),
        "left_mean": float(np.mean(ancestry_gap)),
        "right_mean": float(np.mean(scalar_gap)),
        "paired_difference_mean": float(np.mean(ancestry_gap - scalar_gap)),
        "paired_t_pvalue": float(test.pvalue),
        "wilcoxon_pvalue": signed_rank_p,
    }


def _summarize_gradient_diagnostics(
    paths: list[Path],
    *,
    extension_first_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize scalar-fallback vs neuron-wise fields at fixed checkpoints."""
    frames = [pd.read_csv(path) for path in paths]
    diagnostics = pd.concat(frames, ignore_index=True)
    conditions = {
        "approx_direct_code_per_soma": "scalar_fallback",
        "approx_direct_blockwise_per_soma": "neuron_wise",
    }
    diagnostics = diagnostics[
        diagnostics["condition"].isin(conditions)
    ].copy()
    diagnostics["diagnostic_feedback"] = diagnostics["condition"].map(conditions)
    metrics = [
        "branch_numel_weighted_cosine",
        "branch_macro_cosine",
        "branch_concatenated_cosine",
        "branch_local_exact_norm_ratio",
    ]
    keys = [
        "seed",
        "network_type",
        "trained_broadcast_mode",
        "diagnostic_feedback",
    ]
    seed_level = (
        diagnostics.groupby(keys, as_index=False)[metrics]
        .mean()
        .drop_duplicates(keys, keep="last")
    )
    seed_level["cohort"] = np.where(
        seed_level["seed"] >= extension_first_seed,
        "extension",
        "original",
    )
    combined = seed_level.copy()
    combined["cohort"] = "combined"
    with_cohorts = pd.concat([seed_level, combined], ignore_index=True)

    summary = (
        with_cohorts.groupby(
            [
                "cohort",
                "network_type",
                "trained_broadcast_mode",
                "diagnostic_feedback",
            ],
            as_index=False,
        )[metrics]
        .agg(["count", "mean", "std", "sem"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    paired_rows: list[dict[str, Any]] = []
    for (cohort, network_type, trained_mode), frame in with_cohorts.groupby(
        ["cohort", "network_type", "trained_broadcast_mode"]
    ):
        for metric in metrics:
            pivot = frame.pivot_table(
                index="seed",
                columns="diagnostic_feedback",
                values=metric,
                aggfunc="mean",
            ).dropna()
            needed = {"neuron_wise", "scalar_fallback"}
            if not needed.issubset(pivot.columns) or pivot.empty:
                continue
            neuron_wise = pivot["neuron_wise"].to_numpy(dtype=float)
            scalar = pivot["scalar_fallback"].to_numpy(dtype=float)
            test = ttest_rel(neuron_wise, scalar)
            try:
                signed_rank_p = float(wilcoxon(neuron_wise, scalar).pvalue)
            except ValueError:
                signed_rank_p = float("nan")
            paired_rows.append(
                {
                    "cohort": cohort,
                    "network_type": network_type,
                    "trained_broadcast_mode": trained_mode,
                    "metric": metric,
                    "n_pairs": int(len(pivot)),
                    "neuron_wise_mean": float(np.mean(neuron_wise)),
                    "scalar_fallback_mean": float(np.mean(scalar)),
                    "paired_difference_mean": float(np.mean(neuron_wise - scalar)),
                    "neuron_wise_wins": int(np.sum(neuron_wise > scalar)),
                    "paired_t_pvalue": float(test.pvalue),
                    "wilcoxon_pvalue": signed_rank_p,
                }
            )
    return seed_level, summary, pd.DataFrame(paired_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scalar-root", type=Path, nargs="+")
    parser.add_argument("--ancestry-root", type=Path, nargs="+")
    parser.add_argument(
        "--factorial-root",
        type=Path,
        nargs="+",
        help=(
            "Sweep roots containing both per_soma and per_soma_shared modes; "
            "labels are inferred from each saved configuration."
        ),
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        nargs="+",
        help="Previously exported seed-level detail tables to append.",
    )
    parser.add_argument(
        "--diagnostic-csv",
        type=Path,
        nargs="+",
        help=(
            "Fixed-checkpoint branch-gradient summaries containing the "
            "operational scalar-fallback and neuron-wise diagnostic fields."
        ),
    )
    parser.add_argument("--extension-first-seed", type=int, default=47)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for root in args.scalar_root or []:
        frames.append(_read_runs(root, "scalar_fallback"))
    for root in args.ancestry_root or []:
        frames.append(_read_runs(root, "ancestry_shared"))
    for root in args.factorial_root or []:
        frames.append(_read_runs(root, None))
    for path in args.details_csv or []:
        frames.append(pd.read_csv(path))
    if not frames:
        raise ValueError(
            "Provide scalar/ancestry roots, one or more factorial roots, or both."
        )
    details = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["feedback", "seed", "network_type"],
        keep="last",
    )
    if details.empty:
        raise FileNotFoundError("No completed runs found in either sweep.")

    group_cols = ["feedback", "network_type", "rule_variant", "decoder_update_mode"]
    grouped = (
        details.groupby(group_cols)["test_accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "test_accuracy_mean",
                "std": "test_accuracy_std",
                "min": "test_accuracy_min",
                "max": "test_accuracy_max",
                "count": "n_seeds",
            }
        )
    )

    paired_rows: list[dict[str, Any]] = []
    for feedback, frame in details.groupby("feedback"):
        row = _paired_row(
            frame,
            comparison=f"shunting_vs_additive_with_{feedback}",
            column="network_type",
            left="dendritic_shunting",
            right="dendritic_additive",
        )
        if row is not None:
            paired_rows.append(row)
    for network_type, frame in details.groupby("network_type"):
        row = _paired_row(
            frame,
            comparison=f"ancestry_vs_scalar_with_{network_type}",
            column="feedback",
            left="ancestry_shared",
            right="scalar_fallback",
        )
        if row is not None:
            paired_rows.append(row)
    interaction = _interaction_row(details)
    if interaction is not None:
        paired_rows.append(interaction)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_dir / "feedback_definition_details.csv", index=False)
    grouped.to_csv(args.output_dir / "feedback_definition_grouped.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(
        args.output_dir / "feedback_definition_paired_tests.csv",
        index=False,
    )
    if args.diagnostic_csv:
        seed_level, diagnostic_summary, diagnostic_tests = (
            _summarize_gradient_diagnostics(
                args.diagnostic_csv,
                extension_first_seed=args.extension_first_seed,
            )
        )
        seed_level.to_csv(
            args.output_dir / "feedback_gradient_seed.csv",
            index=False,
        )
        diagnostic_summary.to_csv(
            args.output_dir / "feedback_gradient_summary.csv",
            index=False,
        )
        diagnostic_tests.to_csv(
            args.output_dir / "feedback_gradient_paired.csv",
            index=False,
        )
    print(f"Saved feedback-definition summary to {args.output_dir}")


if __name__ == "__main__":
    main()

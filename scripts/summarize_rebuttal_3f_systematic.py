#!/usr/bin/env python3
"""Summarize matched 3F gradient diagnostics without aggregation artifacts.

The submitted additive and shunting models allocate backpropagation energy very
differently across parameter blocks. A single concatenated cosine can therefore
be dominated by the final soma-coupling block, whose local error is exact in
both architectures. This script reports that diagnostic for completeness, but
also computes the comparisons relevant to dendritic credit assignment:

* synaptic parameters only;
* all non-somatic branch parameters;
* a parameter-count-weighted mean of blockwise cosines;
* an equal-block macro mean;
* the fraction of backpropagation energy in the trivially exact soma block.

Dataset splits are averaged within checkpoint before seeds are treated as
replicates. Paired tests match additive and shunting checkpoints by seed.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


CONDITION_LABELS = {
    "approx_direct_code_per_soma": "legacy_hybrid",
    "approx_direct_blockwise_per_soma": "true_per_soma_shared",
    "approx_direct_path_transport": "approx_soma_path_transport",
    "exact_soma_path_transport": "exact_soma_path_transport",
}


def _checkpoint_seed(name: str) -> int:
    match = re.search(r"(\d+)$", str(name))
    if match is None:
        raise ValueError(f"Cannot infer seed from checkpoint label: {name}")
    return int(match.group(1))


def _concat_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    local_norm = frame["local_grad_norm"].to_numpy(dtype=float)
    bp_norm = frame["backprop_grad_norm"].to_numpy(dtype=float)
    cosine = frame["gradient_cosine"].to_numpy(dtype=float)
    dot = np.nansum(cosine * local_norm * bp_norm)
    local_total = float(np.sqrt(np.nansum(np.square(local_norm))))
    bp_total = float(np.sqrt(np.nansum(np.square(bp_norm))))
    denom = local_total * bp_total
    return (
        float(dot / denom) if denom > 0 else float("nan"),
        float(local_total / bp_total) if bp_total > 0 else float("nan"),
    )


def _weighted_mean(
    frame: pd.DataFrame,
    value: str,
    weight: str,
) -> float:
    values = frame[value].to_numpy(dtype=float)
    weights = frame[weight].to_numpy(dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return float("nan")
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))


def _scope_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    concat_cosine, norm_ratio = _concat_metrics(frame)
    return {
        f"{prefix}_concat_cosine": concat_cosine,
        f"{prefix}_norm_ratio": norm_ratio,
        f"{prefix}_numel_weighted_cosine": _weighted_mean(
            frame,
            "gradient_cosine",
            "numel",
        ),
        f"{prefix}_energy_weighted_cosine": _weighted_mean(
            frame,
            "gradient_cosine",
            "backprop_grad_energy",
        ),
        f"{prefix}_macro_cosine": float(frame["gradient_cosine"].mean()),
    }


def _summarize_one(frame: pd.DataFrame) -> dict[str, float]:
    synaptic = frame[
        frame["component"].isin(["excitatory_synapse", "inhibitory_synapse"])
    ]
    soma_exact = (
        (frame["branch_layer_index"] == frame["branch_layer_index"].max())
        & (frame["component"] == "dendritic_conductance")
    )
    branch = frame[~soma_exact]
    total_energy = float(frame["backprop_grad_energy"].sum())
    soma_energy = float(frame.loc[soma_exact, "backprop_grad_energy"].sum())
    return {
        **_scope_metrics(frame, "all"),
        **_scope_metrics(branch, "branch"),
        **_scope_metrics(synaptic, "synaptic"),
        "soma_exact_bp_energy_fraction": (
            soma_energy / total_energy if total_energy > 0 else float("nan")
        ),
    }


def _discover_details(
    train_dir: Path,
    split_dir: Path,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(train_dir.glob("*/layer_soma_factorial_details.csv")):
        frame = pd.read_csv(path)
        frame["checkpoint"] = path.parent.name
        frame["split"] = "train"
        frames.append(frame)
    for path in sorted(split_dir.glob("*/*/layer_soma_factorial_details.csv")):
        frame = pd.read_csv(path)
        frame["checkpoint"] = path.parent.name
        frame["split"] = path.parents[1].name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No layer-soma factorial detail files were found.")
    out = pd.concat(frames, ignore_index=True)
    out["seed"] = out["checkpoint"].map(_checkpoint_seed)
    return out


def _paired_tests(checkpoint_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    id_cols = {"condition", "checkpoint", "seed", "network_type"}
    metrics = [col for col in checkpoint_summary.columns if col not in id_cols]
    for condition, condition_frame in checkpoint_summary.groupby("condition"):
        for metric in metrics:
            pivot = condition_frame.pivot(
                index="seed",
                columns="network_type",
                values=metric,
            ).dropna()
            required = {"dendritic_shunting", "dendritic_additive"}
            if not required.issubset(pivot.columns) or pivot.empty:
                continue
            shunting = pivot["dendritic_shunting"].to_numpy(dtype=float)
            additive = pivot["dendritic_additive"].to_numpy(dtype=float)
            paired_t = ttest_rel(shunting, additive)
            try:
                paired_w = wilcoxon(shunting, additive)
                wilcoxon_p = float(paired_w.pvalue)
            except ValueError:
                wilcoxon_p = float("nan")
            rows.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "n_pairs": int(len(pivot)),
                    "shunting_mean": float(np.mean(shunting)),
                    "additive_mean": float(np.mean(additive)),
                    "paired_difference_mean": float(np.mean(shunting - additive)),
                    "paired_t_statistic": float(paired_t.statistic),
                    "paired_t_pvalue": float(paired_t.pvalue),
                    "wilcoxon_pvalue": wilcoxon_p,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    details = _discover_details(args.train_dir, args.split_dir)
    details = details[details["condition"].isin(CONDITION_LABELS)].copy()
    details["condition"] = details["condition"].map(CONDITION_LABELS)

    batch_rows = []
    group_cols = ["condition", "checkpoint", "seed", "split", "network_type"]
    for keys, frame in details.groupby(group_cols, dropna=False):
        batch_rows.append(
            {
                **dict(zip(group_cols, keys)),
                **_summarize_one(frame),
            }
        )
    batch_summary = pd.DataFrame(batch_rows)

    metric_cols = [
        col
        for col in batch_summary.columns
        if col not in {"condition", "checkpoint", "seed", "split", "network_type"}
    ]
    checkpoint_summary = (
        batch_summary.groupby(
            ["condition", "checkpoint", "seed", "network_type"],
            as_index=False,
        )[metric_cols]
        .mean()
    )
    architecture_summary = (
        checkpoint_summary.groupby(["condition", "network_type"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    architecture_summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in architecture_summary.columns
    ]

    block_summary = (
        details.groupby(
            [
                "condition",
                "checkpoint",
                "seed",
                "network_type",
                "branch_layer_index",
                "component",
            ],
            as_index=False,
        )["gradient_cosine"]
        .mean()
        .groupby(
            [
                "condition",
                "network_type",
                "branch_layer_index",
                "component",
            ]
        )["gradient_cosine"]
        .agg(["mean", "std"])
        .reset_index()
    )
    tests = _paired_tests(checkpoint_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_dir / "details_combined.csv", index=False)
    batch_summary.to_csv(args.output_dir / "batch_summary.csv", index=False)
    checkpoint_summary.to_csv(
        args.output_dir / "checkpoint_summary.csv",
        index=False,
    )
    architecture_summary.to_csv(
        args.output_dir / "architecture_summary.csv",
        index=False,
    )
    block_summary.to_csv(args.output_dir / "block_summary.csv", index=False)
    tests.to_csv(args.output_dir / "paired_tests.csv", index=False)
    print(f"Saved systematic 3F summary to {args.output_dir}")


if __name__ == "__main__":
    main()

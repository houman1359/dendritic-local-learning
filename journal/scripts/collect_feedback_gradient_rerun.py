#!/usr/bin/env python3
"""Validate and stage clean fixed-checkpoint feedback-gradient diagnostics.

This collector is deliberately non-destructive. It accepts one complete
diagnostic output per architecture, verifies the frozen seed-by-training-mode
design, retains only the scalar-fallback and neuron-indexed fields used in
Figure 2c, and writes staged source data plus an audit summary. Promotion into
``source_data/`` is a separate, explicit step after inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SHUNTING = (
    ROOT / "analysis" / "feedback_rerun_validation" / "shunting_gradient_a100"
)
DEFAULT_ADDITIVE = (
    ROOT / "analysis" / "feedback_rerun_validation" / "additive_gradient_a100"
)
DEFAULT_ACCURACY = ROOT / "source_data" / "figure2" / "feedback_accuracy_runs.csv"
DEFAULT_OUTDIR = ROOT / "analysis" / "feedback_rerun_validation"
EXPECTED_SEEDS = tuple(range(42, 57))
EXPECTED_TRAINED_MODES = ("per_soma", "per_soma_shared")
CONDITION_TO_FEEDBACK = {
    "approx_direct_code_per_soma": "scalar_fallback",
    "approx_direct_blockwise_per_soma": "neuron_wise",
}
METRICS = (
    "branch_numel_weighted_cosine",
    "branch_macro_cosine",
    "branch_concatenated_cosine",
    "branch_local_exact_norm_ratio",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_one(directory: Path, expected_network: str) -> tuple[pd.DataFrame, dict[str, str]]:
    source = directory / "branch_gradient_checkpoint_summary.csv"
    if not source.exists():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    required = {
        "run_dir",
        "run_name",
        "seed",
        "condition",
        "network_type",
        "trained_broadcast_mode",
        *METRICS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")
    if set(frame["network_type"].astype(str)) != {expected_network}:
        raise ValueError(f"{source}: unexpected network types")

    selected = frame.loc[
        frame["condition"].isin(CONDITION_TO_FEEDBACK)
    ].copy()
    selected["diagnostic_feedback"] = selected["condition"].map(
        CONDITION_TO_FEEDBACK
    )
    selected["seed"] = selected["seed"].astype(int)
    for metric in METRICS:
        selected[metric] = pd.to_numeric(selected[metric], errors="raise")
        if not np.isfinite(selected[metric]).all():
            raise ValueError(f"{source}: non-finite {metric}")

    keys = [
        "seed",
        "network_type",
        "trained_broadcast_mode",
        "diagnostic_feedback",
    ]
    selected = selected.groupby(keys, as_index=False)[list(METRICS)].mean()
    expected = {
        (seed, expected_network, trained_mode, feedback)
        for seed in EXPECTED_SEEDS
        for trained_mode in EXPECTED_TRAINED_MODES
        for feedback in CONDITION_TO_FEEDBACK.values()
    }
    observed = set(selected[keys].itertuples(index=False, name=None))
    if observed != expected:
        missing_rows = sorted(expected.difference(observed))
        extra_rows = sorted(observed.difference(expected))
        raise ValueError(
            f"{source}: incomplete frozen design; missing={missing_rows}, extra={extra_rows}"
        )
    return selected, {
        "path": str(source.relative_to(ROOT)),
        "sha256": sha256(source),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shunting", type=Path, default=DEFAULT_SHUNTING)
    parser.add_argument("--additive", type=Path, default=DEFAULT_ADDITIVE)
    parser.add_argument("--accuracy", type=Path, default=DEFAULT_ACCURACY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    shunting, shunting_source = load_one(args.shunting, "dendritic_shunting")
    additive, additive_source = load_one(args.additive, "dendritic_additive")
    frame = pd.concat([shunting, additive], ignore_index=True)

    accuracy = pd.read_csv(args.accuracy)
    checkpoint_columns = [
        "seed",
        "network_type",
        "broadcast_mode",
        "run_dir",
        "checkpoint_sha256",
    ]
    missing_accuracy = sorted(set(checkpoint_columns).difference(accuracy.columns))
    if missing_accuracy:
        raise ValueError(f"{args.accuracy}: missing columns {missing_accuracy}")
    checkpoints = accuracy[checkpoint_columns].rename(
        columns={"broadcast_mode": "trained_broadcast_mode"}
    )
    frame = frame.merge(
        checkpoints,
        on=["seed", "network_type", "trained_broadcast_mode"],
        how="left",
        validate="many_to_one",
    )
    if frame[["run_dir", "checkpoint_sha256"]].isna().any().any():
        raise ValueError("Diagnostic rows could not all be matched to clean checkpoints")
    frame["cohort"] = "clean_current_code"
    frame = frame[
        [
            "seed",
            "network_type",
            "trained_broadcast_mode",
            "diagnostic_feedback",
            *METRICS,
            "cohort",
            "run_dir",
            "checkpoint_sha256",
        ]
    ].sort_values(
        ["network_type", "trained_broadcast_mode", "seed", "diagnostic_feedback"]
    )
    if len(frame) != 120 or frame.duplicated(
        ["seed", "network_type", "trained_broadcast_mode", "diagnostic_feedback"]
    ).any():
        raise ValueError("Expected 120 unique clean diagnostic rows")

    summary = (
        frame.groupby(
            ["network_type", "trained_broadcast_mode", "diagnostic_feedback"],
            as_index=False,
        )[list(METRICS)]
        .agg(["count", "mean", "std"])
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    payload = {
        "status": "complete_and_validated",
        "n_rows": int(len(frame)),
        "expected_seeds": list(EXPECTED_SEEDS),
        "source_summaries": {
            "shunting": shunting_source,
            "additive": additive_source,
        },
        "accuracy_source": {
            "path": str(args.accuracy.relative_to(ROOT)),
            "sha256": sha256(args.accuracy),
        },
        "figure2c_scope": (
            "The figure uses rows trained with per_soma_shared and compares the "
            "scalar-fallback and neuron-indexed diagnostic fields at each same checkpoint."
        ),
        "replacement_policy": (
            "This clean current-code diagnostic replaces the mixed archive regardless "
            "of the direction or magnitude of the resulting effect."
        ),
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.outdir / "feedback_gradient_runs_clean.csv", index=False)
    summary.to_csv(args.outdir / "feedback_gradient_summary_clean.csv", index=False)
    (args.outdir / "gradient_validation_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

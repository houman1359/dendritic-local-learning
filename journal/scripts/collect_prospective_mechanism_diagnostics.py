#!/usr/bin/env python3
"""Validate and collect prospective fixed-checkpoint mechanism diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parents[2]
DEFAULT_INPUT = ROOT / "prospective_runs" / "prospective_mechanism_20260802"
DEFAULT_OUTPUT = ROOT / "source_data" / "prospective_learning"
REQUIRED_METRICS = [
    "field_scaled_capture",
    "gradient_scaled_capture",
    "gradient_cosine",
    "norm_matched_fraction_of_exact",
]
PRIMARY_STEP = 1e-5
UNIQUE_FEEDBACK_FAMILIES = [
    "global_scalar_available",
    "ancestry_available",
    "exact_transport",
]


def _depth(run_dir: Path) -> tuple[int, str]:
    if not run_dir.is_absolute():
        run_dir = WORKSPACE_ROOT / run_dir
    with (run_dir / "config.json").open() as handle:
        config = json.load(handle)
    factors = config["model"]["core"]["architecture"]["excitatory_branch_factors"]
    return len(factors), "x".join(str(value) for value in factors)


def collect(input_dir: Path, expected_items: int, expected_seeds: int) -> pd.DataFrame:
    paths = sorted(input_dir.glob("item_*_runs.csv"))
    if len(paths) != expected_items:
        raise SystemExit(
            f"Expected {expected_items} diagnostic outputs, found {len(paths)}"
        )
    frames = []
    for path in paths:
        item = int(path.stem.removeprefix("item_").removesuffix("_runs"))
        part = pd.read_csv(path)
        if len(part) != 20:
            raise SystemExit(f"{path} has {len(part)} rows; expected 20")
        run_dirs = part.run_dir.unique()
        if len(run_dirs) != 1:
            raise SystemExit(f"{path} contains multiple checkpoint directories")
        depth, factors = _depth(Path(run_dirs[0]))
        part["diagnostic_item"] = item
        part["depth"] = depth
        part["branch_factors"] = factors
        frames.append(part)

    frame = pd.concat(frames, ignore_index=True)
    if not np.isfinite(frame[REQUIRED_METRICS].to_numpy(dtype=float)).all():
        raise SystemExit("Non-finite prospective mechanism metric")
    key = [
        "dataset",
        "network_type",
        "depth",
        "seed",
        "split",
        "feedback_family",
        "relative_step",
    ]
    if frame.duplicated(key).any():
        raise SystemExit("Duplicate prospective mechanism condition")
    expected_families = {
        "global_scalar_available",
        "submitted_mw",
        "ancestry_available",
        "ancestry_exact_soma",
        "exact_transport",
    }
    if set(frame.feedback_family) != expected_families:
        raise SystemExit("Unexpected feedback-family set")
    counts = frame.groupby(
        ["dataset", "network_type", "depth", "feedback_family", "relative_step"]
    ).seed.nunique()
    if not (counts == expected_seeds).all():
        raise SystemExit(
            f"Unbalanced diagnostic seeds:\n{counts[counts != expected_seeds]}"
        )
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    group = [
        "dataset",
        "network_type",
        "depth",
        "branch_factors",
        "split",
        "feedback_family",
        "relative_step",
    ]
    return (
        frame.groupby(group, as_index=False)[REQUIRED_METRICS]
        .agg(["mean", "std", "count"])
        .reset_index()
        .pipe(
            lambda table: table.set_axis(
                [
                    (
                        "_".join(str(value) for value in column if str(value))
                        if isinstance(column, tuple)
                        else str(column)
                    )
                    for column in table.columns
                ],
                axis=1,
            )
        )
    )


def paired_feedback_contrasts(frame: pd.DataFrame) -> pd.DataFrame:
    index = ["dataset", "network_type", "depth", "seed", "relative_step"]
    rows = []
    for metric in REQUIRED_METRICS:
        wide = frame.pivot(index=index, columns="feedback_family", values=metric)
        for left, right, label in (
            ("ancestry_available", "global_scalar_available", "ancestry - scalar"),
            ("exact_transport", "ancestry_available", "exact - ancestry"),
        ):
            values = (wide[left] - wide[right]).rename("difference").reset_index()
            values["metric"] = metric
            values["contrast"] = label
            rows.append(values)
    return pd.concat(rows, ignore_index=True)


def _bootstrap_spearman(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    seed: int,
    n_boot: int = 5_000,
) -> tuple[float, float, float]:
    """Checkpoint-clustered bootstrap interval for a pooled rank correlation."""
    checkpoint_columns = ["dataset", "network_type", "depth", "seed"]
    groups = [
        part.sort_values("feedback_family")
        for _, part in frame.groupby(checkpoint_columns, sort=True)
    ]
    group_sizes = {len(part) for part in groups}
    if group_sizes != {2}:
        raise ValueError(f"Expected two feedback rows per checkpoint, found {group_sizes}")
    x_groups = np.stack(
        [part[x_column].to_numpy(dtype=float) for part in groups], axis=0
    )
    y_groups = np.stack(
        [part[y_column].to_numpy(dtype=float) for part in groups], axis=0
    )
    rho = float(spearmanr(frame[x_column], frame[y_column]).statistic)
    rng = np.random.default_rng(seed)
    sampled = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        draw = rng.integers(0, len(groups), size=len(groups))
        sampled[index] = float(
            spearmanr(
                x_groups[draw].reshape(-1),
                y_groups[draw].reshape(-1),
            ).statistic
        )
    low, high = np.quantile(sampled[np.isfinite(sampled)], [0.025, 0.975])
    return rho, float(low), float(high)


def association_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for step_index, step in enumerate(sorted(frame.relative_step.unique())):
        selected = frame[
            np.isclose(frame.relative_step, step)
            & frame.feedback_family.isin(
                ["global_scalar_available", "ancestry_available"]
            )
        ].copy()
        for metric_index, metric in enumerate(
            ["gradient_cosine", "gradient_scaled_capture", "field_scaled_capture"]
        ):
            rho, low, high = _bootstrap_spearman(
                selected,
                metric,
                "norm_matched_fraction_of_exact",
                seed=100 * step_index + metric_index,
            )
            rows.append(
                {
                    "relative_step": float(step),
                    "geometry_metric": metric,
                    "feedback_families": "scalar_and_ancestry",
                    "n_checkpoints": int(
                        selected[
                            ["dataset", "network_type", "depth", "seed"]
                        ].drop_duplicates().shape[0]
                    ),
                    "n_rows": len(selected),
                    "spearman_rho": rho,
                    "checkpoint_bootstrap_ci95_low": low,
                    "checkpoint_bootstrap_ci95_high": high,
                }
            )
    return pd.DataFrame(rows)


def feedback_summary(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame[
        np.isclose(frame.relative_step, PRIMARY_STEP)
        & frame.feedback_family.isin(UNIQUE_FEEDBACK_FAMILIES)
    ].copy()
    rows: list[dict[str, object]] = []
    for family, part in selected.groupby("feedback_family", sort=False):
        rows.append(
            {
                "feedback_family": family,
                "relative_step": PRIMARY_STEP,
                "n_checkpoints": len(part),
                "gradient_cosine_mean": float(part.gradient_cosine.mean()),
                "gradient_scaled_capture_mean": float(
                    part.gradient_scaled_capture.mean()
                ),
                "one_step_progress_mean": float(
                    part.norm_matched_fraction_of_exact.mean()
                ),
                "one_step_progress_median": float(
                    part.norm_matched_fraction_of_exact.median()
                ),
                "descent_count": int(part.norm_matched_is_descent.astype(bool).sum()),
            }
        )
    return pd.DataFrame(rows)


def report(
    frame: pd.DataFrame,
    contrasts: pd.DataFrame,
    associations: pd.DataFrame,
    feedback: pd.DataFrame,
) -> str:
    step = frame[np.isclose(frame.relative_step, PRIMARY_STEP)].copy()
    primary_association = associations[
        np.isclose(associations.relative_step, PRIMARY_STEP)
    ].set_index("geometry_metric")
    cosine = primary_association.loc["gradient_cosine"]
    capture = primary_association.loc["gradient_scaled_capture"]
    ancestry = contrasts[
        (contrasts.metric == "norm_matched_fraction_of_exact")
        & (contrasts.contrast == "ancestry - scalar")
        & np.isclose(contrasts.relative_step, PRIMARY_STEP)
    ]
    available = feedback.set_index("feedback_family")
    scalar = available.loc["global_scalar_available"]
    ancestry_feedback = available.loc["ancestry_available"]
    lines = [
        "# Prospective checkpoint-mechanism audit",
        "",
        f"Collected {len(frame)} rows from 160 backpropagation checkpoints. All expected conditions contain ten paired seeds.",
        "",
        "## Geometry-to-learning link",
        "",
        f"At relative step $10^{{-5}}$, gradient cosine correlated with retained norm-matched one-step progress across scalar and neuron-indexed fields at Spearman rho={cosine.spearman_rho:.3f} (checkpoint-clustered 95% bootstrap interval {cosine.checkpoint_bootstrap_ci95_low:.3f} to {cosine.checkpoint_bootstrap_ci95_high:.3f}).",
        f"Eligibility-weighted gradient capture gave rho={capture.spearman_rho:.3f} ({capture.checkpoint_bootstrap_ci95_low:.3f} to {capture.checkpoint_bootstrap_ci95_high:.3f}).",
        f"Scalar feedback was a descent direction in {int(scalar.descent_count)}/160 checkpoints, compared with {int(ancestry_feedback.descent_count)}/160 for neuron-indexed feedback.",
        "",
        "## Ancestry value",
        "",
    ]
    grouped = ancestry.groupby(["dataset", "network_type", "depth"]).difference
    for (dataset, network, depth), values in grouped:
        lines.append(
            f"- {dataset}, {network}, depth {depth}: neuron-indexed feedback retained "
            f"{values.mean():.3f} more of the exact norm-matched one-step progress than scalar feedback "
            f"({int((values > 0).sum())}/{len(values)} paired seeds)."
        )
    lines += [
        "",
        "These are fixed-checkpoint diagnostics on matched backpropagation representations. They test how much of an exact update each feedback family can express; they do not substitute for the prospective trained-learning comparisons.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-items", type=int, default=160)
    parser.add_argument("--expected-seeds", type=int, default=10)
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT / "analysis" / "prospective_mechanism_results.md",
    )
    args = parser.parse_args()

    frame = collect(args.input_dir, args.expected_items, args.expected_seeds)
    summary = summarize(frame)
    contrasts = paired_feedback_contrasts(frame)
    associations = association_summary(frame)
    feedback = feedback_summary(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "mechanism_checkpoint_rows.csv", index=False)
    summary.to_csv(args.output_dir / "mechanism_checkpoint_summary.csv", index=False)
    contrasts.to_csv(args.output_dir / "mechanism_paired_contrasts.csv", index=False)
    associations.to_csv(
        args.output_dir / "mechanism_association_summary.csv", index=False
    )
    feedback.to_csv(args.output_dir / "mechanism_feedback_summary.csv", index=False)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        report(frame, contrasts, associations, feedback)
    )
    print(f"Collected {len(frame)} rows from {args.expected_items} checkpoints")


if __name__ == "__main__":
    main()

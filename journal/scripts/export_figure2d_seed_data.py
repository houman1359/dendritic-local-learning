#!/usr/bin/env python3
"""Export and verify the independent-seed values underlying Figure 2d.

The source runs are the archived five-seed exact-transport factorial and the
same-architecture five-seed backpropagation reference.  The exporter reads
each resolved ``config.json`` and ``performance/final.json`` directly, writes
one publication-facing row per run, and verifies that grouping those rows
reproduces every frozen aggregate statistic used by Figure 2d.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


JOURNAL = Path(__file__).resolve().parents[1]
REPOSITORY = JOURNAL.parents[2]

FACTORIAL_RUN_NAME = "revision_exact_transport_factorial_mnist_5seed_20260624152116"
BACKPROP_RUN_NAME = "revision_exact_transport_bp_mnist_5seed_20260624152116"

DEFAULT_FACTORIAL_RESULTS = (
    REPOSITORY
    / "drafts"
    / "dendritic-local-learning"
    / "neurips"
    / "local_sweep_runs"
    / FACTORIAL_RUN_NAME
    / "results"
)
DEFAULT_BACKPROP_RESULTS = (
    REPOSITORY
    / "drafts"
    / "dendritic-local-learning"
    / "neurips"
    / "local_sweep_runs"
    / BACKPROP_RUN_NAME
    / "results"
)
DEFAULT_FACTORIAL_SUMMARY = JOURNAL / "source_data" / "figure2" / "exact_transport_factorial_summary.csv"
DEFAULT_BACKPROP_SUMMARY = JOURNAL / "source_data" / "figure2" / "backprop_summary.csv"
DEFAULT_OUTPUT = JOURNAL / "source_data" / "figure2" / "exact_transport_and_backprop_runs.csv"

OUTPUT_COLUMNS = (
    "run_id",
    "config_id",
    "seed",
    "dataset",
    "core",
    "learning_method",
    "rule",
    "decoder_mode",
    "feedback_mode",
    "test_accuracy",
)
EXPECTED_SEEDS = {42, 43, 44, 45, 46}
EXPECTED_FACTORIAL_CONDITIONS = {
    ("3f", "backprop"),
    ("3f", "local"),
    ("5f", "backprop"),
    ("5f", "local"),
}
STAT_TOLERANCE = 1e-15


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _config_directories(results: Path, expected_count: int) -> list[Path]:
    if not results.is_dir():
        raise FileNotFoundError(results)
    directories = [
        path
        for path in results.iterdir()
        if path.is_dir() and path.name.startswith("config_") and path.name[7:].isdigit()
    ]
    directories.sort(key=lambda path: int(path.name[7:]))
    expected_ids = list(range(expected_count))
    observed_ids = [int(path.name[7:]) for path in directories]
    if observed_ids != expected_ids:
        raise ValueError(
            f"Expected config IDs {expected_ids} in {results}, observed {observed_ids}"
        )
    return directories


def _common_run_fields(config: dict[str, Any]) -> tuple[int, str, str]:
    seed = int(config["experiment"]["seed"])
    dataset = str(config["data"]["dataset_name"])
    raw_core = str(config["model"]["core"]["type"])
    if dataset != "mnist" or raw_core != "dendritic_shunting":
        raise ValueError(f"Unexpected Figure 2d architecture: dataset={dataset}, core={raw_core}")
    return seed, dataset, "shunting"


def extract_rows(factorial_results: Path, backprop_results: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for run_dir in _config_directories(factorial_results, expected_count=20):
        config = _read_json(run_dir / "config.json")
        performance = _read_json(run_dir / "performance" / "final.json")
        seed, dataset, core = _common_run_fields(config)
        training = config["training"]["main"]
        if training["strategy"] != "local_ca":
            raise ValueError(f"{run_dir} is not a local_ca run")
        local_config = training["learning_strategy_config"]
        rule = str(local_config["rule_variant"])
        decoder_mode = str(local_config["decoder_update_mode"])
        feedback_mode = str(local_config["error_broadcast_mode"])
        if (rule, decoder_mode) not in EXPECTED_FACTORIAL_CONDITIONS:
            raise ValueError(f"Unexpected factorial condition in {run_dir}: {(rule, decoder_mode)}")
        if feedback_mode != "path_transport":
            raise ValueError(f"Unexpected feedback mode in {run_dir}: {feedback_mode}")
        rows.append(
            {
                "run_id": f"{FACTORIAL_RUN_NAME}/{run_dir.name}",
                "config_id": run_dir.name,
                "seed": seed,
                "dataset": dataset,
                "core": core,
                "learning_method": "local_ca",
                "rule": rule,
                "decoder_mode": decoder_mode,
                "feedback_mode": feedback_mode,
                "test_accuracy": float(performance["accuracy"]["test"]),
            }
        )

    for run_dir in _config_directories(backprop_results, expected_count=5):
        config = _read_json(run_dir / "config.json")
        performance = _read_json(run_dir / "performance" / "final.json")
        seed, dataset, core = _common_run_fields(config)
        training = config["training"]["main"]
        if training["strategy"] != "standard" or training["learning_strategy_config"] is not None:
            raise ValueError(f"{run_dir} is not a standard backpropagation run")
        rows.append(
            {
                "run_id": f"{BACKPROP_RUN_NAME}/{run_dir.name}",
                "config_id": run_dir.name,
                "seed": seed,
                "dataset": dataset,
                "core": core,
                "learning_method": "standard",
                "rule": "backpropagation",
                "decoder_mode": "backprop",
                "feedback_mode": "backpropagation",
                "test_accuracy": float(performance["accuracy"]["test"]),
            }
        )

    run_ids = [str(row["run_id"]) for row in rows]
    if len(rows) != 25 or len(set(run_ids)) != 25:
        raise ValueError(f"Expected 25 unique Figure 2d runs, observed {len(rows)} rows")
    return rows


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    values = list(values)
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _summary_stats(row: dict[str, str]) -> dict[str, float | int]:
    return {
        "mean": float(row["test_acc_mean"]),
        "std": float(row["test_acc_std"]),
        "min": float(row["test_acc_min"]),
        "max": float(row["test_acc_max"]),
        "n": int(row["n_seeds"]),
    }


def _assert_matching_stats(
    label: str,
    observed: dict[str, float | int],
    expected: dict[str, float | int],
) -> float:
    maximum_difference = 0.0
    for key in ("mean", "std", "min", "max"):
        difference = abs(float(observed[key]) - float(expected[key]))
        maximum_difference = max(maximum_difference, difference)
        if not math.isclose(
            float(observed[key]),
            float(expected[key]),
            rel_tol=0.0,
            abs_tol=STAT_TOLERANCE,
        ):
            raise ValueError(
                f"{label} {key} differs: observed {observed[key]!r}, expected {expected[key]!r}"
            )
    if int(observed["n"]) != int(expected["n"]):
        raise ValueError(
            f"{label} n differs: observed {observed['n']}, expected {expected['n']}"
        )
    return maximum_difference


def verify_rows(
    rows: list[dict[str, Any]],
    factorial_summary: Path,
    backprop_summary: Path,
) -> tuple[list[dict[str, Any]], float]:
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["learning_method"] == "local_ca":
            grouped[(str(row["rule"]), str(row["decoder_mode"]))].append(row)

    if set(grouped) != EXPECTED_FACTORIAL_CONDITIONS:
        raise ValueError(f"Unexpected exported factorial conditions: {sorted(grouped)}")
    for condition, condition_rows in grouped.items():
        if {int(row["seed"]) for row in condition_rows} != EXPECTED_SEEDS:
            raise ValueError(f"Factorial condition {condition} does not contain seeds 42--46")

    with factorial_summary.open(newline="", encoding="utf-8") as handle:
        factorial_expected = list(csv.DictReader(handle))
    with backprop_summary.open(newline="", encoding="utf-8") as handle:
        backprop_expected = list(csv.DictReader(handle))
    if len(factorial_expected) != 4 or len(backprop_expected) != 1:
        raise ValueError("Frozen Figure 2d summaries have an unexpected number of rows")

    verification: list[dict[str, Any]] = []
    maximum_difference = 0.0
    for expected_row in factorial_expected:
        condition = (expected_row["rule_variant"], expected_row["decoder_update_mode"])
        observed = _stats(float(row["test_accuracy"]) for row in grouped[condition])
        expected = _summary_stats(expected_row)
        maximum_difference = max(
            maximum_difference,
            _assert_matching_stats(f"{condition[0]}/{condition[1]}", observed, expected),
        )
        verification.append(
            {
                "condition": f"{condition[0]}/{condition[1]}/path_transport",
                **observed,
            }
        )

    backprop_rows = [row for row in rows if row["learning_method"] == "standard"]
    if len(backprop_rows) != 5 or {int(row["seed"]) for row in backprop_rows} != EXPECTED_SEEDS:
        raise ValueError("Backpropagation reference does not contain exactly seeds 42--46")
    observed = _stats(float(row["test_accuracy"]) for row in backprop_rows)
    expected = _summary_stats(backprop_expected[0])
    maximum_difference = max(
        maximum_difference,
        _assert_matching_stats("backpropagation", observed, expected),
    )
    verification.append({"condition": "backpropagation", **observed})
    return verification, maximum_difference


def write_rows(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
            raise ValueError(
                f"Unexpected Figure 2d source-data columns in {path}: {reader.fieldnames}"
            )
        rows: list[dict[str, Any]] = []
        for row in reader:
            row["seed"] = int(row["seed"])
            row["test_accuracy"] = float(row["test_accuracy"])
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial-results", type=Path, default=DEFAULT_FACTORIAL_RESULTS)
    parser.add_argument("--backprop-results", type=Path, default=DEFAULT_BACKPROP_RESULTS)
    parser.add_argument("--factorial-summary", type=Path, default=DEFAULT_FACTORIAL_SUMMARY)
    parser.add_argument("--backprop-summary", type=Path, default=DEFAULT_BACKPROP_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = extract_rows(args.factorial_results.resolve(), args.backprop_results.resolve())
    verification, maximum_difference = verify_rows(
        rows,
        args.factorial_summary.resolve(),
        args.backprop_summary.resolve(),
    )
    write_rows(rows, args.output.resolve())

    print(f"Wrote {args.output.resolve()} ({len(rows)} independent-seed rows)")
    for result in verification:
        print(
            f"{result['condition']}: n={result['n']}, mean={result['mean']:.16g}, "
            f"sample_sd={result['std']:.16g}, min={result['min']:.16g}, "
            f"max={result['max']:.16g}"
        )
    print(f"Maximum absolute difference from frozen summaries: {maximum_difference:.3e}")


if __name__ == "__main__":
    main()

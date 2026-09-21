"""Paired radial exposure control with frozen baseline optimizer choices."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run(config_path: Path, output: Path) -> dict:
    specification = json.loads(config_path.read_text())
    baseline = Path(specification["baseline_dir"])
    frozen_source = Path(specification["frozen_source_path"])
    for path, expected in (
        (frozen_source, specification["frozen_source_sha256"]),
        (
            baseline / "prespecification.json",
            specification["baseline_prespecification_sha256"],
        ),
        (
            baseline / "frozen_selection.json",
            specification["baseline_selection_sha256"],
        ),
    ):
        if sha(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    if not (baseline / "receipt.json").is_file():
        raise ValueError("Baseline must be complete before the paired extension.")
    baseline_spec = json.loads((baseline / "prespecification.json").read_text())
    baseline_selection = json.loads((baseline / "frozen_selection.json").read_text())
    if (
        specification["selected_learning_rates"]
        != baseline_selection["selected_learning_rates"]
    ):
        raise ValueError("Exposure control must reuse every frozen optimizer choice.")
    if any(
        specification["config"][key] != value
        for key, value in baseline_spec["config"].items()
        if key != "steps"
    ):
        raise ValueError("Only training exposure may differ from the baseline config.")
    if specification["config"]["steps"] <= baseline_spec["config"]["steps"]:
        raise ValueError("Exposure control must increase the terminal step.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite exposure evidence.")
    output.mkdir(parents=True, exist_ok=True)
    module_spec = importlib.util.spec_from_file_location("frozen_radial", frozen_source)
    radial = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = radial
    module_spec.loader.exec_module(radial)
    config = radial.RadialConfig(**specification["config"])
    config.validate()
    torch.set_num_threads(1)
    grids = radial.make_grids(config)
    started = time.perf_counter()
    worker = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config_sha256": sha(config_path),
        "source_sha256": sha(Path(__file__)),
        "baseline_source_sha256": sha(frozen_source),
        "baseline_receipt_sha256": sha(baseline / "receipt.json"),
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
        "status": "running",
    }
    write_json(output / "worker.json", worker)
    write_json(output / "prespecification.json", specification)
    records = []
    prefix_checks = []
    for split, seeds in (
        ("development", config.development_seeds),
        ("confirmation", config.confirmation_seeds),
    ):
        for family in radial.FAMILIES:
            for budget in config.budgets:
                rate = specification["selected_learning_rates"][family][str(budget)]
                for seed in seeds:
                    model, trace, metrics = radial.fit_model(
                        family, budget, seed, rate, config, grids
                    )
                    key = f"p{budget}_seed{seed}_lr{rate:g}"
                    run_dir = output / split / family / key
                    run_dir.mkdir(parents=True)
                    baseline_dir = baseline / split / family / key
                    with (baseline_dir / "trace.csv").open() as handle:
                        old_trace = list(csv.DictReader(handle))
                    max_loss_difference = max(
                        abs(float(old[field]) - new[field])
                        for old, new in zip(old_trace, trace)
                        for field in ("train_mse", "validation_mse")
                    )
                    same_evaluations = all(
                        int(old["closure_calls"]) == new["closure_calls"]
                        for old, new in zip(old_trace, trace)
                    )
                    exact_prefix = all(
                        float(old[field]) == new[field]
                        for old, new in zip(old_trace, trace)
                        for field in old
                    )
                    prefix_checks.append(
                        {
                            "split": split,
                            "family": family,
                            "budget": budget,
                            "seed": seed,
                            "baseline_rows": len(old_trace),
                            "bitwise_numeric_prefix": exact_prefix,
                            "same_closure_counts": same_evaluations,
                            "max_loss_difference": max_loss_difference,
                        }
                    )
                    metrics["split"] = split
                    metrics["baseline_terminal_train_mse"] = float(
                        old_trace[-1]["train_mse"]
                    )
                    metrics["baseline_terminal_validation_mse"] = float(
                        old_trace[-1]["validation_mse"]
                    )
                    if split == "confirmation":
                        metrics.update(
                            {
                                f"test_{key}": value
                                for key, value in radial._metrics(
                                    model, grids["test"]
                                ).items()
                            }
                        )
                    radial._write_csv(run_dir / "trace.csv", trace)
                    torch.save(
                        {
                            "state_dict": model.state_dict(),
                            "family": family,
                            "budget": budget,
                            "seed": seed,
                        },
                        run_dir / "terminal.pt",
                    )
                    write_json(run_dir / "metrics.json", metrics)
                    records.append(metrics)
                print(
                    f"radial exposure: {split} {family} P={budget} finished", flush=True
                )
    for split in ("development", "confirmation"):
        radial._write_csv(
            output / f"{split}_summary.csv",
            [row for row in records if row["split"] == split],
        )
    equivalence = []
    for split, seeds in (
        ("development", config.development_seeds),
        ("confirmation", config.confirmation_seeds),
    ):
        for budget in config.budgets:
            for seed in seeds:
                paths = []
                for family in ("positive_shunt", "divisive_control"):
                    rate = specification["selected_learning_rates"][family][str(budget)]
                    paths.append(
                        output
                        / split
                        / family
                        / f"p{budget}_seed{seed}_lr{rate:g}"
                        / "trace.csv"
                    )
                equivalence.append(
                    {
                        "split": split,
                        "budget": budget,
                        "seed": seed,
                        "bitwise_identical_trace": paths[0].read_bytes()
                        == paths[1].read_bytes(),
                    }
                )
    worker.update(
        {
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    write_json(output / "worker.json", worker)
    receipt = {
        "worker": worker,
        "runs": len(records),
        "prefix_checks": prefix_checks,
        "all_prefixes_bitwise_numeric_identical": all(
            row["bitwise_numeric_prefix"] for row in prefix_checks
        ),
        "max_prefix_loss_difference": max(
            row["max_loss_difference"] for row in prefix_checks
        ),
        "generic_control_equivalence": equivalence,
        "all_counts_match": all(
            row["budget"] == row["actual_parameters"] for row in records
        ),
        "all_models_optimized": all(row["parameter_change_l2"] > 0 for row in records),
        "source_unchanged": worker["source_sha256"] == sha(Path(__file__)),
        "baseline_source_unchanged": worker["baseline_source_sha256"]
        == sha(frozen_source),
        "baseline_receipt_unchanged": worker["baseline_receipt_sha256"]
        == sha(baseline / "receipt.json"),
        "artifact_sha256": {
            str(path.relative_to(output)): sha(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    write_json(output / "receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.config, args.output_dir)


if __name__ == "__main__":
    main()

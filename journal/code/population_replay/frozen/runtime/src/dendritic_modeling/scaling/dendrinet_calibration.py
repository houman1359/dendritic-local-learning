"""Audit every cell of a two-budget DendriNet learning-rate calibration.

Only existing validation receipts and TRAIN diagnostics are read. This module
does not select architectures, open datasets, fit exponents, or release a sweep.
Descriptive LR comparisons wait until the entire manifest is terminal.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from .analyze import _identity, collect_campaign


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(directory: Path, relative: str) -> Path:
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError(f"Campaign path escapes its directory: {relative}")
    return path


def _jsonl(path: Path, *, required: bool) -> tuple[list[dict], str | None]:
    if not path.exists() and not required:
        return [], None
    raw = path.read_bytes()
    lines = raw.splitlines()
    # A live writer can be observed between writes. Never treat a partial last
    # line as a terminal observation or discard malformed completed artifacts.
    if raw and not raw.endswith(b"\n") and not required:
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()], hashlib.sha256(
        raw
    ).hexdigest()


def _finite(value, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {name}")
    return value


def _diagnostic_summary(records: list[dict], parameters: int) -> list[dict]:
    result = []
    for record in records:
        groups = record.get("parameter_groups", {})
        if (
            record.get("status") == "backward_completed"
            and sum(group["parameters"] for group in groups.values()) != parameters
        ):
            raise ValueError("Diagnostic parameter groups do not reconcile")
        branches = {}
        for name, observations in record.get("activations", {}).items():
            selected = []
            for observation in observations:
                if observation["module_type"] != "DendriticBranchLayer":
                    continue
                output = observation["output"]
                gradient = observation.get("output_gradient", {})
                selected.append(
                    {
                        "output_rms": output["rms"],
                        "output_max_abs": output["max_abs"],
                        "all_zero_units_fraction": output.get(
                            "all_zero_units_fraction"
                        ),
                        "zero_output_fraction": output["zero_fraction_of_finite"],
                        "nonfinite_output_elements": output["elements"]
                        - output["finite_elements"],
                        "output_gradient_rms": gradient.get("rms"),
                    }
                )
            if selected:
                branches[name] = selected
        result.append(
            {
                "step": record["step"],
                "status": record.get("status"),
                "gradient_stage": record.get("gradient_stage"),
                "branches": branches,
                "parameter_groups": {
                    name: {
                        key: group.get(key)
                        for key in (
                            "parameters",
                            "parameters_missing_gradients",
                            "zero_gradient_elements",
                            "nonfinite_parameter_elements",
                            "nonfinite_gradient_elements",
                            "parameter_norm",
                            "gradient_norm",
                            "gradient_to_parameter_norm",
                        )
                    }
                    for name, group in groups.items()
                },
            }
        )
    return result


def _trajectory(records: list[dict], receipt: dict, *, completed: bool) -> dict:
    trajectory = [
        {
            "step": int(row["step"]),
            "validation_loss": _finite(row["validation_loss"], "validation loss"),
            "validation_accuracy": _finite(
                row["validation_accuracy"], "validation accuracy"
            ),
            "training_loss": row.get("training_loss"),
            "gradient_norm": row.get("gradient_norm"),
        }
        for row in records
    ]
    steps = [row["step"] for row in trajectory]
    if steps != sorted(set(steps)):
        raise ValueError("Validation trajectory has duplicate or unordered steps")
    if completed:
        if not trajectory or trajectory[-1]["step"] != receipt["training"]["steps"]:
            raise ValueError("Validation trajectory lacks the terminal training step")
        if trajectory[-1]["validation_loss"] != receipt["metrics"]["validation_loss"]:
            raise ValueError("Terminal validation log and receipt disagree")
        if (
            trajectory[0]["step"] != 0
            or trajectory[0]["validation_loss"]
            != receipt["metrics"]["initial_validation_loss"]
        ):
            raise ValueError("Initial validation log and receipt disagree")
    trained = [row for row in trajectory if row["step"] > 0]
    earlier = trained[:-1]
    best = min(earlier, key=lambda row: row["validation_loss"]) if earlier else None
    terminal = trained[-1] if completed and trained else None
    return {
        "logged_validation": trajectory,
        "best_earlier_logged_step": best["step"] if best else None,
        "best_earlier_logged_validation_loss": best["validation_loss"]
        if best
        else None,
        "terminal_minus_best_earlier_validation_loss": (
            terminal["validation_loss"] - best["validation_loss"]
            if terminal and best
            else None
        ),
        "last_interval_validation_improvement": (
            earlier[-1]["validation_loss"] - terminal["validation_loss"]
            if terminal and earlier
            else None
        ),
        "note": "Intermediate logged values diagnose horizon sensitivity; no best-step checkpoint or early-stopping result is claimed.",
    }


def _lr_tables(rows: list[dict], *, all_terminal: bool) -> dict:
    if not all_terminal:
        return {"status": "withheld_until_all_planned_cells_terminal", "recipes": []}
    by_base = defaultdict(list)
    for row in rows:
        by_base[row["base_architecture_id"]].append(row)
    recipes = []
    for base, records in sorted(by_base.items()):
        budgets = sorted({row["target_parameters"] for row in records})
        rates = sorted({row["learning_rate"] for row in records})
        tables, optima = [], []
        for budget in budgets:
            candidates = []
            for rate in rates:
                selected = [
                    row
                    for row in records
                    if row["target_parameters"] == budget
                    and row["learning_rate"] == rate
                ]
                eligible = all(row["status"] == "completed" for row in selected)
                losses = [
                    row["loss"] for row in selected if row["status"] == "completed"
                ]
                candidates.append(
                    {
                        "learning_rate": rate,
                        "eligible": eligible,
                        "actual_parameter_counts": sorted(
                            {row["actual_parameters"] for row in selected}
                        ),
                        "seed_observations": [
                            {
                                "seed": row["seed"],
                                "status": row["status"],
                                "validation_loss": row.get("loss"),
                            }
                            for row in selected
                        ],
                        "validation_loss_mean": statistics.mean(losses)
                        if eligible
                        else None,
                        "validation_loss_sd": statistics.stdev(losses)
                        if eligible and len(losses) > 1
                        else None,
                    }
                )
            available = [row for row in candidates if row["eligible"]]
            best = (
                min(
                    available,
                    key=lambda row: (row["validation_loss_mean"], row["learning_rate"]),
                )
                if available
                else None
            )
            optimum = best["learning_rate"] if best else None
            optima.append(optimum)
            tables.append(
                {
                    "target_parameters": budget,
                    "candidates": candidates,
                    "descriptive_best_lr": optimum,
                }
            )
        common = []
        for rate in rates:
            selected = [row for row in records if row["learning_rate"] == rate]
            eligible = all(row["status"] == "completed" for row in selected)
            # The planned Cartesian grid has equal seed counts at each budget.
            common.append(
                {
                    "learning_rate": rate,
                    "eligible": eligible,
                    "equally_weighted_budget_seed_mean": statistics.mean(
                        row["loss"] for row in selected
                    )
                    if eligible
                    else None,
                }
            )
        disagreement = len({rate for rate in optima if rate is not None}) > 1
        recipes.append(
            {
                "base_architecture_id": base,
                "budget_specific": tables,
                "common_lr_sensitivity": common,
                "budget_specific_lr_optima_disagree": disagreement,
                "additional_calibration_required": disagreement
                or any(rate is None for rate in optima),
            }
        )
    return {"status": "descriptive_only_no_sweep_release", "recipes": recipes}


def audit_calibration(campaign_dir: str | Path) -> dict:
    """Read and validate an immutable campaign plus its possibly live outputs."""
    directory = Path(campaign_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    spec_path = directory / "spec.json"
    if _hash(spec_path) != manifest["spec_file_sha256"]:
        raise ValueError("Frozen calibration specification hash mismatch")
    spec = json.loads(spec_path.read_text())
    if spec.get("phase") != "calibration" or manifest.get("phase") != "calibration":
        raise ValueError("An explicit calibration campaign is required")
    if spec.get("evaluate_test") is not False:
        raise ValueError("Calibration must not request test evaluation")
    if len(spec["parameter_budgets"]) != 2 or len(spec["data_budgets"]) != 1:
        raise ValueError(
            "This audit requires two parameter budgets and one data budget"
        )
    for axis in ("parameter_budgets", "data_budgets", "seeds"):
        if not spec[axis] or len(spec[axis]) != len(set(spec[axis])):
            raise ValueError(f"Invalid or duplicate {axis}")
    if not manifest.get("source_snapshot"):
        raise ValueError("Frozen production sources are required")
    source_hashes = {}
    for entry in manifest["source_files"]:
        if entry["path"] in source_hashes:
            raise ValueError("Duplicate frozen source entry")
        if _hash(_inside(directory, entry["path"])) != entry["sha256"]:
            raise ValueError(f"Frozen source hash mismatch: {entry['path']}")
        source_hashes[entry["path"]] = entry["sha256"]
    for name in ("train.py", "models.py", "data.py", "diagnostics.py"):
        if f"source/src/dendritic_modeling/scaling/{name}" not in source_hashes:
            raise ValueError(f"Missing frozen production source: {name}")
    architectures = {row["name"]: row for row in spec["architectures"]}
    if len(architectures) != len(spec["architectures"]):
        raise ValueError("Duplicate calibration candidate")
    axes = spec["study_design"]["architecture_axes"]
    candidate_coordinates = [
        (axes[name]["base_architecture_id"], row["training_overrides"]["lr"])
        for name, row in architectures.items()
    ]
    if len(candidate_coordinates) != len(set(candidate_coordinates)):
        raise ValueError("A base recipe declares a duplicate learning-rate candidate")
    expected = set(
        itertools.product(
            architectures,
            spec["parameter_budgets"],
            spec["data_budgets"],
            spec["seeds"],
        )
    )
    observed, configs = set(), {}
    recipes, allocations = {}, {}
    rates_by_base = defaultdict(set)
    for task in manifest["tasks"]:
        config_path = _inside(directory, task["config"])
        _inside(directory, task["output"])
        if _hash(config_path) != task["config_file_sha256"]:
            raise ValueError(f"Frozen config hash mismatch: {task['id']}")
        config = json.loads(config_path.read_text())
        candidate = config["architecture_id"]
        cell = (
            candidate,
            config["target_parameters"],
            config["data"]["train_size"],
            config["seed"],
        )
        if cell not in expected or cell in observed:
            raise ValueError("Unexpected or duplicate calibration grid cell")
        if config["id"] != task["id"] or config.get("evaluate_test") is not False:
            raise ValueError("Config ID mismatch or test evaluation requested")
        if (
            config["model"]["seed"] != config["seed"]
            or config["model"]["topology_seed"] != config["seed"]
        ):
            raise ValueError(
                "Model seed differs from the declared coupled initialization/topology seed"
            )
        declared_model = {**spec.get("model_defaults", {}), **architectures[candidate]}
        for key, value in declared_model.items():
            if (
                key
                not in {"name", "training_overrides", "width", "seed", "topology_seed"}
                and config["model"].get(key) != value
            ):
                raise ValueError(
                    f"Frozen config changes the declared model recipe: {key}"
                )
        declared_training = {
            **spec.get("training", {}),
            **architectures[candidate].get("training_overrides", {}),
        }
        if any(
            config["training"].get(key) != value
            for key, value in declared_training.items()
        ):
            raise ValueError("Frozen config changes the declared training protocol")
        tolerance = float(config.get("budget_tolerance", 0.02))
        if (
            not 0 <= tolerance <= 0.02
            or abs(task["total_parameters"] - config["target_parameters"])
            > tolerance * config["target_parameters"]
        ):
            raise ValueError(
                "Planned whole-model parameters miss the declared budget tolerance"
            )
        observed.add(cell)
        configs[task["id"]] = config
        base = axes[candidate]["base_architecture_id"]
        rate = _finite(config["training"]["lr"], "learning rate")
        if rate <= 0 or rate != architectures[candidate]["training_overrides"]["lr"]:
            raise ValueError("Learning rate disagrees with declared candidate")
        recipe = {
            key: value
            for key, value in config["model"].items()
            if key not in {"width", "seed", "topology_seed"}
        }
        training = {
            key: value for key, value in config["training"].items() if key != "lr"
        }
        identity = _identity(
            {"model": recipe, "training": training, "data": config["data"]}
        )
        if base in recipes and recipes[base] != identity:
            raise ValueError(
                "A base recipe changes more than learning rate, seed, or width"
            )
        recipes[base] = identity
        allocation_key = (base, config["target_parameters"])
        allocation = (config["model"]["width"], task["total_parameters"])
        if allocation_key in allocations and allocations[allocation_key] != allocation:
            raise ValueError(
                "Learning-rate or seed candidates change the allocated width or parameter count"
            )
        allocations[allocation_key] = allocation
        rates_by_base[base].add(rate)
    if observed != expected:
        raise ValueError("Manifest omits planned calibration grid cells")
    if len({tuple(sorted(rates)) for rates in rates_by_base.values()}) != 1:
        raise ValueError("Recipes do not have equal learning-rate search coverage")

    rows = collect_campaign(directory, metric_split="validation")
    tasks = {task["id"]: task for task in manifest["tasks"]}
    for row in rows:
        config = configs[row["id"]]
        candidate = config["architecture_id"]
        row.update(
            base_architecture_id=axes[candidate]["base_architecture_id"],
            learning_rate=config["training"]["lr"],
            actual_parameters=tasks[row["id"]]["total_parameters"],
            width=config["model"]["width"],
        )
        run = _inside(directory, tasks[row["id"]]["output"])
        receipt_path = run / "receipt.json"
        if not receipt_path.exists():
            continue
        try:
            receipt_raw = receipt_path.read_bytes()
            receipt = json.loads(receipt_raw)
            completed = row["status"] == "completed"
            if receipt.get("config") != config or receipt.get(
                "config_sha256"
            ) != _identity(config):
                raise ValueError("Receipt does not bind its frozen configuration")
            required_source_names = {
                "models.py",
                "train.py",
                "data.py",
                "diagnostics.py",
            }
            if not required_source_names.issubset(receipt.get("source_sha256", {})):
                raise ValueError("Receipt omits a required source identity")
            for name, digest in receipt["source_sha256"].items():
                if (
                    source_hashes.get(f"source/src/dendritic_modeling/scaling/{name}")
                    != digest
                ):
                    raise ValueError(f"Receipt source hash mismatch: {name}")
            if receipt.get("data", {}).get("identity", {}).get("test_materialized"):
                raise ValueError("Receipt indicates test data were materialized")
            if completed:
                if receipt["data"]["identity"].get("test_materialized") is not False:
                    raise ValueError(
                        "Completed receipt lacks an explicit unopened-test declaration"
                    )
                if any(key.startswith("test_") for key in receipt["metrics"]):
                    raise ValueError("Calibration receipt contains test metrics")
                if receipt["training"].get("lr") != config["training"]["lr"]:
                    raise ValueError(
                        "Completed receipt reports a different learning rate"
                    )
                initial, terminal = (
                    receipt["model_report"],
                    receipt["terminal_model_report"],
                )
                for report in (initial, terminal):
                    count = sum(
                        math.prod(shape)
                        for shape in report["parameter_shapes"].values()
                    )
                    if (
                        count != row["actual_parameters"]
                        or count != report["total_parameters"]
                    ):
                        raise ValueError("Observed parameter shapes do not reconcile")
                    dendritic = config["model"]["family"].startswith("dendritic_")
                    inventory = report.get("dendrinet_inventory")
                    if bool(inventory) != dendritic:
                        raise ValueError(
                            "Production DendriNet inventory missing or mislabeled"
                        )
                    if dendritic and (
                        sum(inventory["parameter_partition"].values()) != count
                        or inventory["population_count"]
                        != config["model"]["network_depth"]
                    ):
                        raise ValueError(
                            "Production DendriNet inventory does not reconcile"
                        )
                if initial["topology_sha256"] != terminal["topology_sha256"]:
                    raise ValueError("Static topology changed during calibration")
            metrics, metrics_hash = _jsonl(run / "metrics.jsonl", required=completed)
            diagnostics, diagnostics_hash = _jsonl(
                run / "diagnostics.jsonl", required=completed
            )
            if completed:
                horizon, every = (
                    config["training"]["steps"],
                    config["training"]["diagnostics_every"],
                )
                expected_steps = sorted({1, horizon, *range(every, horizon + 1, every)})
                if [record["step"] for record in diagnostics] != expected_steps or any(
                    record.get("status") != "backward_completed"
                    for record in diagnostics
                ):
                    raise ValueError(
                        "Completed diagnostics have a missing/failed training step"
                    )
            row.update(_trajectory(metrics, receipt, completed=completed))
            row["training_diagnostics"] = _diagnostic_summary(
                diagnostics, row["actual_parameters"]
            )
            row["receipt_sha256"] = hashlib.sha256(receipt_raw).hexdigest()
            row["metrics_sha256"] = metrics_hash
            row["diagnostics_sha256"] = diagnostics_hash
            row["receipt_status_at_artifact_read"] = receipt["status"]
        except (KeyError, ValueError, TypeError, OSError) as error:
            row.update(status="invalid", reason=str(error))
    identities = {
        (row.get("task_identity"), row.get("train_identity"))
        for row in rows
        if row["status"] == "completed"
    }
    if len(identities) > 1:
        raise ValueError("Calibration cells use different targets or sample splits")
    counts = dict(Counter(row["status"] for row in rows))
    all_terminal = all(row["status"] in {"completed", "failed"} for row in rows)
    return {
        "schema_version": "real_dendrinet_two_budget_calibration_audit_v1",
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "campaign": str(directory),
        "manifest_sha256": _hash(directory / "manifest.json"),
        "spec_sha256": _hash(spec_path),
        "verified_source_files": len(source_hashes),
        "analyzer_sources": {
            name: _hash(Path(__file__).with_name(name))
            for name in ("dendrinet_calibration.py", "analyze.py")
        },
        "expected_tasks": len(expected),
        "task_status_counts": counts,
        "all_planned_cells_terminal": all_terminal,
        "test_data_opened": False,
        "main_sweep_release_allowed": False,
        "release_note": "Audit only. All planned cells must be terminal before any LR comparison; horizon, conditioning, and budget-specific LR disagreement still require review. The legacy single-budget scaling.select interface is incompatible with this campaign.",
        "lr_response": _lr_tables(rows, all_terminal=all_terminal),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    report = audit_calibration(args.campaign_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "expected_tasks",
                    "task_status_counts",
                    "all_planned_cells_terminal",
                    "main_sweep_release_allowed",
                )
            }
        )
    )


if __name__ == "__main__":
    main()

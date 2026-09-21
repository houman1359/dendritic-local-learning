"""Read-only receipt and protocol audit for the bounded DendriNet interventions.

No checkpoint is loaded, model retrained, or TEST data opened. This audits
recorded evidence and protocols, not independent forward/backward replay.
Full readout contrasts are withheld while any declared task is nonterminal.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import inspect
import itertools
import json
import math
from pathlib import Path
import statistics

from .analyze import collect_campaign
from .dendrinet_interventions import validate_intervention_spec


def _identity(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _readout_receipt(
    receipt: dict, expected: dict, readout_parameters: int | None = None
) -> None:
    if receipt.get("mode") != expected["mode"]:
        raise ValueError("Initializer treatment mismatch")
    if receipt.get("parameters_added") != 0:
        raise ValueError("Readout initialization changed parameter inventory")
    if expected["mode"] == "preserve":
        if receipt.get("training_examples_exposed") != 0:
            raise ValueError("Preserve initialization exposed training examples")
        return
    requirements = {
        "target_rms": 0.1,
        "training_examples_exposed": 128,
        "initialization_forward_passes": 2,
        "initialization_examples_processed": 256,
        "prefix_within_existing_unique_training_budget": True,
        "labels_used": False,
        "validation_used": False,
    }
    if any(receipt.get(key) != value for key, value in requirements.items()):
        raise ValueError("RMS initializer exposure/target/label contract differs")
    if not math.isclose(receipt["achieved_rms"], 0.1, rel_tol=2e-5, abs_tol=1e-8):
        raise ValueError("RMS initializer did not achieve its declared target")
    for key in ("initial_rms", "scale"):
        if not math.isfinite(receipt[key]) or receipt[key] <= 0:
            raise ValueError("Invalid RMS initializer scale")
    if not math.isclose(
        receipt["scale"] * receipt["initial_rms"], 0.1, rel_tol=2e-12, abs_tol=1e-12
    ):
        raise ValueError("RMS scale and initial RMS are internally inconsistent")
    if (
        readout_parameters is not None
        and receipt.get("parameters_scaled") != readout_parameters
    ):
        raise ValueError(
            "RMS initializer scaled-parameter count differs from the readout inventory"
        )
    before, after = (
        receipt.get("hidden_state_sha256_before"),
        receipt.get("hidden_state_sha256_after"),
    )
    if (
        not isinstance(before, str)
        or len(before) != 64
        or any(c not in "0123456789abcdef" for c in before)
        or before != after
    ):
        raise ValueError("Initializer hidden-state hashes do not match")


def _config_contract(config: dict, spec: dict) -> None:
    architecture = next(
        (
            row
            for row in spec["architectures"]
            if row["name"] == config["architecture_id"]
        ),
        None,
    )
    if architecture is None:
        raise ValueError("Configuration refers to an undeclared architecture")
    expected_training = {
        **spec["training"],
        **architecture.get("training_overrides", {}),
    }
    if config["training"] != expected_training:
        raise ValueError(
            "Resolved training protocol differs from the declared architecture override"
        )
    if config["data"] != {**spec["data"], "train_size": config["data"]["train_size"]}:
        raise ValueError("Resolved data recipe differs from the declared specification")
    expected_model = {
        **spec["model_defaults"],
        **{
            k: v
            for k, v in architecture.items()
            if k not in {"name", "training_overrides"}
        },
        "seed": config["seed"],
        "topology_seed": config["seed"],
    }
    if any(config["model"].get(key) != value for key, value in expected_model.items()):
        raise ValueError("Resolved model recipe differs from the declared architecture")
    if (
        config["phase"] != spec["phase"]
        or config["budget_tolerance"] != spec["budget_tolerance"]
    ):
        raise ValueError("Resolved phase/budget contract differs")


def _terminal_inventory(receipt: dict, config: dict, planned_parameters: int) -> None:
    if (
        abs(planned_parameters - config["target_parameters"])
        / config["target_parameters"]
        > config["budget_tolerance"]
    ):
        raise ValueError("Whole-model count misses the declared parameter budget")
    first, last = receipt["model_report"], receipt["terminal_model_report"]
    required = {
        "total_parameters",
        "trainable_parameters",
        "parameter_shapes",
        "topology_sha256",
        "parameter_bytes",
        "buffer_bytes",
        "hidden_parameters",
        "readout_parameters",
        "resolved_spec",
        "indexed_projection_backends",
    }
    for report in (first, last):
        if not required <= set(report):
            raise ValueError("Model inventory is missing required schema fields")
        if (
            report["total_parameters"] != planned_parameters
            or report["trainable_parameters"] != planned_parameters
        ):
            raise ValueError(
                "Registered/trainable parameter inventory differs from the manifest"
            )
        if report["resolved_spec"] != config["model"]:
            raise ValueError(
                "Recorded resolved model differs from the frozen configuration"
            )
        if (
            report["hidden_parameters"] + report["readout_parameters"]
            != planned_parameters
        ):
            raise ValueError(
                "Hidden/readout parameter partition does not add to whole P"
            )
        if (
            sum(math.prod(shape) for shape in report["parameter_shapes"].values())
            != planned_parameters
        ):
            raise ValueError("Registered parameter shapes do not add to whole P")
    for key in (
        "parameter_shapes",
        "topology_sha256",
        "parameter_bytes",
        "buffer_bytes",
        "hidden_parameters",
        "readout_parameters",
        "dendrinet_inventory",
    ):
        if first.get(key) != last.get(key):
            raise ValueError(f"Terminal architecture/topology changed: {key}")
    dendritic = config["model"]["family"].startswith("dendritic_")
    activation = config["model"].get("activation", "relu")
    gate_shapes = {
        name: shape
        for name, shape in last["parameter_shapes"].items()
        if name.endswith((".log_m", ".b"))
    }
    if activation == "relu" and gate_shapes:
        raise ValueError(
            "Ordinary ReLU model unexpectedly contains learned gate parameters"
        )
    if not dendritic and activation in {"param_relu", "param_tanh"}:
        expected_names = {
            f"hidden.{layer}.1.{parameter}"
            for layer in range(config["model"]["network_depth"])
            for parameter in ("log_m", "b")
        }
        if set(gate_shapes) != expected_names or any(
            shape != [config["model"]["width"]] for shape in gate_shapes.values()
        ):
            raise ValueError(
                "Ordinary control learned-gate parameter shapes/counts differ from the declared model"
            )
    if (
        dendritic
        and last.get("dendrinet_inventory", {}).get("population_count")
        != config["model"]["network_depth"]
    ):
        raise ValueError("Missing or inconsistent production DendriNet inventory")
    if dendritic:
        inventory = last["dendrinet_inventory"]
        if len(inventory["populations"]) != config["model"]["network_depth"]:
            raise ValueError("Observed population list differs from network depth")
        expected_gate = {
            "relu": "ReLU",
            "param_relu": "ParametricReLU",
            "param_tanh": "ParametricTanh",
        }[activation]
        expected_gate_count = (
            0 if activation == "relu" else 2 * inventory["total_compartments"]
        )
        if (
            sum(math.prod(shape) for shape in gate_shapes.values())
            != expected_gate_count
        ):
            raise ValueError(
                "DendriNet learned-gate shapes disagree with the observed compartment inventory"
            )
        for population in inventory["populations"]:
            if (
                population["n_soma"] != config["model"]["width"]
                or population["branch_factors"] != config["model"]["branch_factors"]
            ):
                raise ValueError(
                    "Observed soma width/morphology differs from the declared model"
                )
            if len(population["depths"]) != len(config["model"]["branch_factors"]) + 1:
                raise ValueError("Observed branch-depth inventory is incomplete")
            for depth in population["depths"]:
                gate_parameters = (
                    0 if activation == "relu" else 2 * depth["compartment_count"]
                )
                if (
                    depth["gate_enabled"] is not True
                    or depth["gate_type"] != expected_gate
                    or depth["gate_parameters"] != gate_parameters
                ):
                    raise ValueError(
                        "Observed gate type/count differs from the declared model"
                    )
    if set(first["indexed_projection_backends"]) != set(
        last["indexed_projection_backends"]
    ):
        raise ValueError("Indexed projection inventory changed")
    for backend in last["indexed_projection_backends"].values():
        if backend["requested"] != "eager" or backend["resolved"] != "eager":
            raise ValueError("Observed backend differs from the declared eager path")
        if config["training"].get("device") == "cuda" and not str(
            backend["device"]
        ).startswith("cuda"):
            raise ValueError(
                "Indexed projection did not execute on the declared CUDA device"
            )


def _finite_tree(value: object) -> None:
    if isinstance(value, dict):
        if "finite_elements" in value and value["finite_elements"] != value["elements"]:
            raise ValueError("Diagnostic contains nonfinite tensor elements")
        for key in (
            "nonfinite_gradient_elements",
            "nonfinite_parameter_elements",
            "parameters_missing_gradients",
        ):
            if key in value and value[key] != 0:
                raise ValueError(f"Successful diagnostic has nonzero {key}")
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Diagnostic contains nonfinite scalars")


def _no_test(receipt: dict) -> None:
    if receipt.get("data", {}).get("identity", {}).get("test_materialized", False):
        raise ValueError("Receipt materialized TEST data")
    if any(key.startswith("test_") for key in receipt.get("metrics", {})):
        raise ValueError("Development receipt contains TEST metrics")


def _source_receipt(receipt: dict, manifest_sources: dict, read, inside) -> None:
    names = {
        "train.py",
        "data.py",
        "models.py",
        "diagnostics.py",
        "readout_initialization.py",
    }
    recorded = receipt.get("source_sha256")
    if not isinstance(recorded, dict) or set(recorded) != names:
        raise ValueError("Receipt must bind all five executed training source files")
    for name in names:
        path = f"source/src/dendritic_modeling/scaling/{name}"
        if path not in manifest_sources or manifest_sources[path] != recorded[name]:
            raise ValueError(
                "Receipt source hash differs from the manifest snapshot binding"
            )
        if hashlib.sha256(read(inside(path))).hexdigest() != recorded[name]:
            raise ValueError("Receipt source identity mismatch")


def _successful_streams(
    metrics: list[dict], diagnostics: list[dict], receipt: dict, training: dict
) -> None:
    steps, every, diagnostic_every = (
        training["steps"],
        training["eval_every"],
        training["diagnostics_every"],
    )
    expected_training_receipt = {
        "lr": training["lr"],
        "schedule": training.get("schedule", "constant"),
        "weight_decay": training.get("weight_decay", 0.0),
        "batch_size": training["batch_size"],
        "steps": steps,
        "horizon_steps": steps,
        "processed_examples": steps * training["batch_size"],
    }
    if any(
        receipt["training"].get(key) != value
        for key, value in expected_training_receipt.items()
    ):
        raise ValueError(
            "Training receipt optimizer/horizon/exposure protocol differs from configuration"
        )
    expected_metrics = sorted({0, steps} | set(range(every, steps + 1, every)))
    expected_diagnostics = sorted(
        {1, steps} | set(range(diagnostic_every, steps + 1, diagnostic_every))
    )
    if [row["step"] for row in metrics] != expected_metrics:
        raise ValueError(
            "Raw metric cadence/order/uniqueness differs from the declared protocol"
        )
    if [row["step"] for row in diagnostics] != expected_diagnostics:
        raise ValueError(
            "Training diagnostic cadence/order/uniqueness differs from the declared protocol"
        )
    if (
        metrics[-1]["validation_loss"] != receipt["metrics"]["validation_loss"]
        or metrics[0]["validation_loss"]
        != receipt["metrics"]["initial_validation_loss"]
    ):
        raise ValueError("Raw metric endpoints and receipt disagree")
    if metrics[-1]["validation_accuracy"] != receipt["metrics"]["validation_accuracy"]:
        raise ValueError("Terminal raw accuracy and receipt disagree")
    for metric in metrics:
        _finite_tree(metric)
        if (
            metric.get("learning_rate") != training["lr"]
            or metric.get("schedule") != training.get("schedule", "constant")
            or metric.get("horizon_steps") != steps
            or metric.get("processed_examples")
            != metric["step"] * training["batch_size"]
            or not math.isclose(
                metric.get("horizon_fraction", -1),
                metric["step"] / steps,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "Raw optimizer/horizon/exposure protocol differs from the declared training"
            )
        if any(key.startswith("test_") for key in metric):
            raise ValueError("Raw metric stream contains TEST evaluation")
    for diagnostic in diagnostics:
        if (
            diagnostic.get("status") != "backward_completed"
            or diagnostic.get("split") != "train"
            or diagnostic.get("gradient_stage") != "pre_clipping"
            or diagnostic.get("parameter_stage") != "pre_update"
        ):
            raise ValueError(
                "Successful diagnostic lacks backward/TRAIN/stage certification"
            )
        _finite_tree(diagnostic)
        groups = diagnostic.get("parameter_groups", {})
        if (
            not groups
            or sum(group["parameters"] for group in groups.values())
            != receipt["model_report"]["total_parameters"]
        ):
            raise ValueError(
                "Diagnostic parameter groups do not cover the registered model"
            )
        for group in groups.values():
            if any(
                group.get(key) != 0
                for key in (
                    "nonfinite_gradient_elements",
                    "nonfinite_parameter_elements",
                    "parameters_missing_gradients",
                )
            ):
                raise ValueError(
                    "Diagnostic group lacks finite/complete gradient certification"
                )
        if not diagnostic.get("activations"):
            raise ValueError("Successful diagnostic has no activation observations")
        for observations in diagnostic["activations"].values():
            if not observations or any(
                observation.get("output") is None
                or observation.get("output_gradient") is None
                for observation in observations
            ):
                raise ValueError(
                    "Successful activation diagnostic lacks output/gradient observations"
                )


def _paired_comparisons(
    rows: list[dict], expected_seeds: list[int]
) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(dict)
    for row in rows:
        key = (
            row["base_architecture_id"],
            row["target_parameters"],
            row["seed"],
            row["protocol"],
        )
        if row["intervention"] in groups[key]:
            raise ValueError("Duplicate paired treatment")
        groups[key][row["intervention"]] = row
    paired = []
    for (base, budget, seed, protocol), treatments in sorted(groups.items()):
        if set(treatments) != {"preserve", "train_batch_rms"}:
            raise ValueError("Readout comparison is missing a paired arm")
        left, right = treatments["preserve"], treatments["train_batch_rms"]
        for key in ("model_identity", "training_protocol_identity", "steps", "lr"):
            if left[key] != right[key]:
                raise ValueError(f"Paired runs differ outside initialization: {key}")
        available = left["status"] == right["status"] == "completed"
        if available:
            for key in (
                "parameters",
                "topology_sha256",
                "train_identity",
                "task_identity",
            ):
                if left[key] != right[key]:
                    raise ValueError(f"Paired data/model inventory mismatch: {key}")
        record = {
            "base_architecture_id": base,
            "target_parameters": budget,
            "seed": seed,
            "protocol": protocol,
            "steps": left["steps"],
            "lr": left["lr"],
            "preserve_status": left["status"],
            "rms_status": right["status"],
            "preserve_id": left["id"],
            "rms_id": right["id"],
            "preserve_validation_ce": left.get("validation_ce"),
            "rms_validation_ce": right.get("validation_ce"),
            "difference_rms_minus_preserve": (
                right["validation_ce"] - left["validation_ce"] if available else None
            ),
        }
        paired.append(record)
    by_cell = defaultdict(list)
    for row in paired:
        by_cell[
            (row["base_architecture_id"], row["target_parameters"], row["protocol"])
        ].append(row)
    means = []
    for (base, budget, protocol), records in sorted(by_cell.items()):
        if sorted(row["seed"] for row in records) != sorted(expected_seeds):
            raise ValueError(
                "Pairwise seed aggregation differs from the declared seed set"
            )
        differences = [row["difference_rms_minus_preserve"] for row in records]
        available = all(value is not None for value in differences)
        means.append(
            {
                "base_architecture_id": base,
                "target_parameters": budget,
                "protocol": protocol,
                "steps": records[0]["steps"],
                "lr": records[0]["lr"],
                "declared_seeds": expected_seeds,
                "paired_seeds_available": sum(v is not None for v in differences),
                "preserve_validation_ce_mean": statistics.mean(
                    r["preserve_validation_ce"] for r in records
                )
                if available
                else None,
                "rms_validation_ce_mean": statistics.mean(
                    r["rms_validation_ce"] for r in records
                )
                if available
                else None,
                "mean_difference_rms_minus_preserve": statistics.mean(differences)
                if available
                else None,
            }
        )
    return paired, means


def audit_campaign(campaign: str | Path) -> dict:
    directory = Path(campaign).resolve()
    bindings = {}

    def read(path: Path) -> bytes:
        path = path.resolve()
        raw = path.read_bytes()
        bindings[str(path)] = hashlib.sha256(raw).hexdigest()
        return raw

    def inside(relative: str) -> Path:
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Campaign path escapes its frozen directory")
        return path

    result = {
        "schema_version": "real_dendrinet_intervention_audit_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "campaign": str(directory),
        "test_data_opened": False,
        "checkpoint_replay_performed": False,
        "errors": [],
        "rows": [],
        "paired_comparisons": [],
        "paired_seed_means": [],
    }
    for tool in (audit_campaign, collect_campaign, validate_intervention_spec):
        read(Path(inspect.getsourcefile(tool)))
    manifest = json.loads(read(directory / "manifest.json"))
    if manifest.get("source_snapshot") is not True or not manifest.get("source_files"):
        raise ValueError(
            "Intervention audit requires a nonempty frozen source snapshot"
        )
    spec_raw = read(directory / "spec.json")
    if hashlib.sha256(spec_raw).hexdigest() != manifest["spec_file_sha256"]:
        raise ValueError("Frozen specification hash mismatch")
    spec = json.loads(spec_raw)
    validate_intervention_spec(spec)
    kind = spec["study_design"]["stage"]
    result["intervention_kind"] = kind
    expected_count = 72 if kind == "readout_pair" else 36
    if (
        len(manifest["tasks"]) != expected_count
        or len({t["id"] for t in manifest["tasks"]}) != expected_count
    ):
        raise ValueError(
            "Manifest task count/uniqueness differs from declared intervention"
        )
    for source in manifest["source_files"]:
        if hashlib.sha256(read(inside(source["path"]))).hexdigest() != source["sha256"]:
            raise ValueError(f"Frozen source hash mismatch: {source['path']}")
    manifest_sources = {
        source["path"]: source["sha256"] for source in manifest["source_files"]
    }
    collected = {row["id"]: row for row in collect_campaign(directory, "validation")}
    observed_grid = set()
    for task in manifest["tasks"]:
        row = {"id": task["id"], "status": collected[task["id"]]["status"]}
        result["rows"].append(row)
        try:
            raw = read(inside(task["config"]))
            if hashlib.sha256(raw).hexdigest() != task["config_file_sha256"]:
                raise ValueError("Configuration hash mismatch")
            config = json.loads(raw)
            if config["id"] != task["id"] or config.get("evaluate_test", False):
                raise ValueError("Task identity or no-TEST contract differs")
            _config_contract(config, spec)
            axes = spec["study_design"]["architecture_axes"][config["architecture_id"]]
            observed_grid.add(
                (
                    config["architecture_id"],
                    config["target_parameters"],
                    config["data"]["train_size"],
                    config["seed"],
                )
            )
            training = {
                k: v
                for k, v in config["training"].items()
                if k != "readout_initialization"
            }
            row.update(
                {
                    "architecture_id": config["architecture_id"],
                    "base_architecture_id": axes["base_architecture_id"],
                    "intervention": axes["intervention"],
                    "protocol": axes["protocol"],
                    "target_parameters": config["target_parameters"],
                    "seed": config["seed"],
                    "steps": training["steps"],
                    "lr": training["lr"],
                    "model_identity": _identity(config["model"]),
                    "training_protocol_identity": _identity(training),
                }
            )
            path = inside(task["output"] + "/receipt.json")
            if not path.exists():
                continue
            receipt = json.loads(read(path))
            row["status"] = receipt.get("status", "invalid")
            if row["status"] not in {"running", "completed", "failed"}:
                raise ValueError("Unknown receipt status")
            if receipt.get("config") != config or receipt.get(
                "config_sha256"
            ) != _identity(config):
                raise ValueError("Receipt configuration identity mismatch")
            if any(receipt.get(key) != config[key] for key in ("id", "phase", "seed")):
                raise ValueError("Receipt task labels differ")
            _no_test(receipt)
            _source_receipt(receipt, manifest_sources, read, inside)
            raw_streams = {}
            for name in ("metrics.jsonl", "diagnostics.jsonl"):
                path = inside(task["output"] + "/" + name)
                if path.exists():
                    raw_streams[name] = read(path)
            if "readout_initialization" in receipt:
                readout_parameters = receipt.get(
                    "model_report", config.get("planned_model_report", {})
                ).get("readout_parameters")
                if (
                    config["training"]["readout_initialization"]["mode"]
                    == "train_batch_rms"
                    and readout_parameters is None
                ):
                    raise ValueError(
                        "RMS initializer lacks a bound readout parameter inventory"
                    )
                _readout_receipt(
                    receipt["readout_initialization"],
                    config["training"]["readout_initialization"],
                    readout_parameters,
                )
            elif row["status"] == "completed":
                raise ValueError("Completed run has no initializer receipt")
            if row["status"] != "completed":
                row["failure"] = receipt.get("failure", receipt.get("error"))
                continue
            run_config = json.loads(read(inside(task["output"] + "/config.json")))
            if run_config != config:
                raise ValueError(
                    "Run configuration artifact differs from the frozen configuration"
                )
            # Recollect if this run completed after the initial live scan.
            evidence = collected[row["id"]]
            if evidence["status"] != "completed":
                evidence = next(
                    r
                    for r in collect_campaign(directory, "validation")
                    if r["id"] == row["id"]
                )
            if evidence["status"] != "completed":
                raise ValueError(
                    f"Completed receipt fails collector checks: {evidence.get('reason')}"
                )
            if evidence["loss"] != receipt["metrics"]["validation_loss"]:
                raise ValueError("Collector loss differs from the hash-bound receipt")
            _terminal_inventory(receipt, config, task["total_parameters"])
            metrics = [
                json.loads(line)
                for line in raw_streams["metrics.jsonl"].splitlines()
                if line.strip()
            ]
            diagnostics = [
                json.loads(line)
                for line in raw_streams["diagnostics.jsonl"].splitlines()
                if line.strip()
            ]
            _successful_streams(metrics, diagnostics, receipt, training)
            if receipt["training"]["parameters_missing_gradients_max"] != 0:
                raise ValueError("Completed run recorded missing parameter gradients")
            if not math.isfinite(receipt["training"]["gradient_norm_max"]):
                raise ValueError("Completed run recorded nonfinite gradients")
            row.update(
                {
                    "parameters": task["total_parameters"],
                    "topology_sha256": receipt["terminal_model_report"][
                        "topology_sha256"
                    ],
                    "train_identity": evidence["train_identity"],
                    "task_identity": evidence["task_identity"],
                    "diagnostic_records": len(diagnostics),
                    "engineering_checks": "passed",
                }
            )
            if kind == "readout_pair":
                row["validation_ce"] = evidence["loss"]
        except (OSError, KeyError, ValueError, TypeError) as error:
            row.update(status="invalid", error=str(error))
            result["errors"].append(f"{row['id']}: {error}")
    expected_grid = set(
        itertools.product(
            [r["name"] for r in spec["architectures"]],
            spec["parameter_budgets"],
            spec["data_budgets"],
            spec["seeds"],
        )
    )
    if observed_grid != expected_grid:
        result["errors"].append(
            "Observed configuration axes differ from the declared grid"
        )
    result["status_counts"] = dict(
        sorted(Counter(r["status"] for r in result["rows"]).items())
    )
    terminal = all(r["status"] in {"completed", "failed"} for r in result["rows"])
    result["all_declared_tasks_terminal"] = terminal
    result["comparisons_released"] = (
        kind == "readout_pair" and terminal and not result["errors"]
    )
    if result["comparisons_released"]:
        try:
            result["paired_comparisons"], result["paired_seed_means"] = (
                _paired_comparisons(result["rows"], spec["seeds"])
            )
        except (KeyError, ValueError, TypeError) as error:
            result["errors"].append(str(error))
            result["comparisons_released"] = False
    result["status"] = (
        "invalid" if result["errors"] else ("terminal" if terminal else "incomplete")
    )
    result["scope"] = (
        "All declared paired terminal validation-CE differences, separated by protocol; failed pairs remain unavailable and suppress full-seed means. No exponent/frontier inference."
        if kind == "readout_pair"
        else "Gate profile engineering checks only; no quality ranking."
    )
    result["limitations"] = (
        "Receipt/raw-stream consistency, not checkpoint replay. RMS receipts bind hidden state before/after initialization; paired initial tensor equality was implementation-qualified, not reconstructed by this auditor. Live nonterminal streams may change after the recorded hashes."
    )
    result["file_sha256"] = dict(sorted(bindings.items()))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    result = audit_campaign(args.campaign)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "status_counts": result["status_counts"],
                "comparisons_released": result["comparisons_released"],
                "output": str(args.output),
            }
        )
    )
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

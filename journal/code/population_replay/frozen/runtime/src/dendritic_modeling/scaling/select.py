"""Freeze validation-mean learning-rate choices without selecting model recipes.

All planned calibration seeds must have terminal receipts. A candidate with any
failed seed is ineligible, and its failure remains in the selection audit.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import os
import re
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path

from .analyze import _identity, collect_campaign
from .models import _normalize_spec

_CANDIDATE = re.compile(
    r"(?P<base>[A-Za-z0-9_-]+)_lr(?P<rate>(?:0|[1-9][0-9]*)p[0-9]+)"
)
_NON_CLASSIFICATION_SOURCES = {
    "source/src/dendritic_modeling/scaling/select.py",
    "source/src/dendritic_modeling/scaling/analyze.py",
    "source/src/dendritic_modeling/scaling/language.py",
}


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seeds(values: list, name: str) -> set[int]:
    if (
        not values
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values
        )
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{name} must contain distinct nonnegative integer seeds")
    return set(values)


def _recipes(spec: dict) -> dict[str, dict]:
    entries = spec["architectures"]
    result = {entry["name"]: entry for entry in entries}
    if not entries or len(result) != len(entries):
        raise ValueError("Architecture names must be nonempty and unique")
    return result


def _model_for(spec: dict, architecture: dict, width: int) -> dict:
    model = copy.deepcopy(spec.get("model_defaults", {}))
    model.update(
        {
            key: value
            for key, value in architecture.items()
            if key not in {"name", "training_overrides"}
        }
    )
    model.update(width=width, seed=0, topology_seed=0)
    return _normalize_spec(model)


def _training_for(spec: dict, architecture: dict) -> dict:
    training = copy.deepcopy(spec.get("training", {}))
    training.update(copy.deepcopy(architecture.get("training_overrides", {})))
    return training


def _load_campaign(directory: Path) -> dict:
    manifest_path, spec_path = directory / "manifest.json", directory / "spec.json"
    manifest = json.loads(manifest_path.read_text())
    calibration = json.loads(spec_path.read_text())
    if _file_hash(spec_path) != manifest["spec_file_sha256"]:
        raise ValueError("Calibration spec differs from the frozen manifest")
    if (
        manifest.get("phase") != "calibration"
        or calibration.get("phase") != "calibration"
    ):
        raise ValueError("Selection requires an explicit calibration campaign")
    source_files = {}
    if manifest.get("source_snapshot", False):
        for entry in manifest["source_files"]:
            path = directory / entry["path"]
            if entry["path"] in source_files or not path.resolve().is_relative_to(
                directory
            ):
                raise ValueError(
                    "Frozen source manifest contains a duplicate or external path"
                )
            if _file_hash(path) != entry["sha256"]:
                raise ValueError(f"Frozen source changed: {entry['path']}")
            source_files[entry["path"]] = entry["sha256"]
        for name in ("train.py", "data.py", "models.py"):
            if f"source/src/dendritic_modeling/scaling/{name}" not in source_files:
                raise ValueError(
                    "Frozen calibration lacks required classification source identities"
                )
    classification_sources = {
        key: value
        for key, value in source_files.items()
        if key not in _NON_CLASSIFICATION_SOURCES
    }
    provenance = {
        "campaign": str(directory),
        "calibration_manifest_sha256": _file_hash(manifest_path),
        "calibration_spec_sha256": _file_hash(spec_path),
        "source_snapshot": bool(manifest.get("source_snapshot", False)),
        "verified_source_files": len(source_files),
        "classification_source_identity": (
            _identity(classification_sources) if source_files else None
        ),
        "excluded_nonclassification_sources": {
            key: source_files[key]
            for key in sorted(_NON_CLASSIFICATION_SOURCES)
            if key in source_files
        },
    }
    return {
        "directory": directory,
        "manifest": manifest,
        "spec": calibration,
        "classification_sources": classification_sources,
        "source_files": source_files,
        "provenance": provenance,
    }


def select_calibration(
    campaign_dir: str | Path | list[str | Path], pilot_spec: dict
) -> tuple[dict, dict]:
    """Combine compatible LR extensions, retaining every frozen candidate/seed."""
    inputs = (
        [campaign_dir] if isinstance(campaign_dir, (str, Path)) else list(campaign_dir)
    )
    directories = [Path(path).resolve() for path in inputs]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("Calibration campaigns must be nonempty and distinct")
    sources = [_load_campaign(directory) for directory in directories]
    calibration = sources[0]["spec"]
    if len(sources) > 1 and any(
        not source["provenance"]["source_snapshot"] for source in sources
    ):
        raise ValueError(
            "Combining campaigns requires individually verified frozen source snapshots"
        )
    for source in sources[1:]:
        if source["classification_sources"] != sources[0]["classification_sources"]:
            raise ValueError(
                "Calibration campaigns have different classification or production source identities"
            )
        for key in ("data", "parameter_budgets", "data_budgets"):
            if source["spec"].get(key) != calibration.get(key):
                raise ValueError(f"Calibration campaigns must have identical {key}")
        if _seeds(source["spec"]["seeds"], "Calibration seeds") != _seeds(
            calibration["seeds"], "Calibration seeds"
        ):
            raise ValueError(
                "Calibration campaigns must have identical calibration seeds"
            )
    if pilot_spec.get("phase", "pilot") != "pilot":
        raise ValueError("The destination specification must be a pilot")
    if any(
        source["spec"].get("evaluate_test", False) for source in sources
    ) or pilot_spec.get("evaluate_test", False):
        raise ValueError(
            "Test access is forbidden during calibration and pilot selection"
        )
    calibration_seeds = _seeds(calibration["seeds"], "Calibration seeds")
    pilot_seeds = _seeds(pilot_spec["seeds"], "Pilot seeds")
    if calibration_seeds & pilot_seeds:
        raise ValueError("Calibration and pilot seeds overlap")
    if len(calibration_seeds) < 2:
        raise ValueError(
            "Calibration requires at least two independent seeds per candidate"
        )
    if (
        len(calibration["parameter_budgets"]) != 1
        or len(calibration["data_budgets"]) != 1
    ):
        raise ValueError("This selector requires one common calibration P and U")
    base_recipes, candidates = _recipes(pilot_spec), {}
    for source in sources:
        local_candidates = _recipes(source["spec"])
        if set(candidates) & set(local_candidates):
            raise ValueError(
                "Duplicate calibration candidates across campaigns would repeat candidate/seed points"
            )
        candidates.update(local_candidates)
    if calibration.get("data", {}) != pilot_spec.get("data", {}):
        raise ValueError(
            "Calibration and pilot must declare the same data/task and evaluation protocol"
        )
    parsed = {}
    for name in candidates:
        match = _CANDIDATE.fullmatch(name)
        if match is None or match["base"] not in base_recipes:
            raise ValueError(
                f"Candidate does not identify a retained pilot recipe: {name}"
            )
        rate = float(match["rate"].replace("p", "."))
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError(f"Candidate has an invalid learning rate: {name}")
        parsed[name] = (match["base"], rate)
    if {base for base, _ in parsed.values()} != set(base_recipes):
        raise ValueError("Every pilot recipe needs calibration candidates")
    expected = set(
        itertools.product(
            candidates,
            calibration_seeds,
            calibration["parameter_budgets"],
            calibration["data_budgets"],
        )
    )
    rows = [
        row
        for source in sources
        for row in collect_campaign(source["directory"], metric_split="validation")
    ]
    unfinished = [row for row in rows if row["status"] not in {"completed", "failed"}]
    if unfinished:
        raise ValueError(
            "All planned tasks need verified completed or failed receipts: "
            + ", ".join(f"{row['id']}={row['status']}" for row in unfinished)
        )
    by_id = {(row["campaign"], row["id"]): row for row in rows}
    observed = set()
    evidence = defaultdict(list)
    base_invariants = {}
    data_identities = set()
    for source, task in (
        (context, planned_task)
        for context in sources
        for planned_task in context["manifest"]["tasks"]
    ):
        directory, calibration = source["directory"], source["spec"]
        config_path = directory / task["config"]
        config = json.loads(config_path.read_text())
        receipt_path = directory / task["output"] / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        # collect_campaign validates completed receipts. Failures also need an
        # exact frozen identity before they can make a candidate ineligible.
        if receipt.get("config") != config or receipt.get("config_sha256") != _identity(
            config
        ):
            raise ValueError(
                "Calibration receipt identity does not match its frozen configuration"
            )
        if any(receipt.get(key) != config[key] for key in ("id", "phase", "seed")):
            raise ValueError(
                "Calibration receipt id/phase/seed does not match its configuration"
            )
        if source["source_files"]:
            for name in ("train.py", "data.py", "models.py"):
                expected_source = source["source_files"][
                    f"source/src/dendritic_modeling/scaling/{name}"
                ]
                if receipt.get("source_sha256", {}).get(name) != expected_source:
                    raise ValueError(
                        "Calibration receipt source identity differs from its frozen snapshot"
                    )
        if config["phase"] != "calibration" or config.get("evaluate_test", False):
            raise ValueError("Every calibration task must prohibit test evaluation")
        metrics = receipt.get("metrics", {})
        data_identity = receipt.get("data", {}).get("identity", {})
        if (
            any(key.startswith("test_") for key in metrics)
            or data_identity.get("test_materialized", False)
            or "test" in data_identity.get("splits", {})
        ):
            raise ValueError("Calibration receipt contains test access or test metrics")
        name = config["architecture_id"]
        if name not in _recipes(calibration):
            raise ValueError("Manifest task is not a declared calibration candidate")
        base, rate = parsed[name]
        key = (
            name,
            config["seed"],
            config["target_parameters"],
            config["data"]["train_size"],
        )
        if key in observed or key not in expected:
            raise ValueError(
                "Calibration manifest contains a duplicate or unexpected candidate/seed/budget"
            )
        observed.add(key)
        width = config["model"]["width"]
        effective = _normalize_spec({**config["model"], "seed": 0, "topology_seed": 0})
        if effective != _model_for(pilot_spec, base_recipes[base], width):
            raise ValueError(
                f"Calibration model recipe differs from pilot recipe {base}"
            )
        if effective != _model_for(calibration, candidates[name], width):
            raise ValueError(
                "Calibration model differs from its frozen candidate specification"
            )
        planned_training = _training_for(calibration, candidates[name])
        if config["training"] != planned_training or planned_training.get("lr") != rate:
            raise ValueError(
                "Candidate learning rate/training does not match the frozen specification"
            )
        expected_data = {**calibration.get("data", {}), "train_size": key[3]}
        if config["data"] != expected_data:
            raise ValueError(
                "Calibration task data differs from its frozen specification"
            )
        invariant = {
            "model": effective,
            "target_parameters": key[2],
            "unique_data": key[3],
            "actual_parameters": task["total_parameters"],
            "training_except_lr": {
                key: value for key, value in planned_training.items() if key != "lr"
            },
        }
        if base in base_invariants and invariant != base_invariants[base]:
            raise ValueError(
                f"Candidates for {base} differ in more than learning rate or seed"
            )
        base_invariants[base] = invariant
        row = by_id[(str(directory), task["id"])]
        if row["status"] == "completed":
            measured_spec = receipt["model_report"].get(
                "resolved_spec", receipt.get("effective_model_spec")
            )
            if (
                measured_spec is None
                or _normalize_spec({**measured_spec, "seed": 0, "topology_seed": 0})
                != effective
            ):
                raise ValueError(
                    "Completed receipt does not verify the actual normalized model recipe"
                )
            data_identities.add((row["task_identity"], row["train_identity"]))
            if receipt["training"].get("lr") != rate:
                raise ValueError(
                    "Recorded training learning rate differs from its planned candidate"
                )
        evidence[name].append(
            {
                "id": task["id"],
                "campaign": str(directory),
                "seed": config["seed"],
                "status": row["status"],
                "validation_loss": row.get("loss"),
                "failure": receipt.get("failure"),
                "config_file_sha256": _file_hash(config_path),
                "config_sha256": _identity(config),
                "receipt_sha256": _file_hash(receipt_path),
            }
        )
    if observed != expected:
        raise ValueError(
            "Calibration manifest does not account for every declared candidate and seed"
        )
    if len(data_identities) > 1:
        raise ValueError(
            "Completed calibration runs do not share the same task, training data, and validation set"
        )
    grouped = defaultdict(list)
    for name, records in sorted(evidence.items()):
        base, rate = parsed[name]
        eligible = all(record["status"] == "completed" for record in records)
        grouped[base].append(
            {
                "candidate": name,
                "lr": rate,
                "eligible": eligible,
                "mean_validation_loss": (
                    statistics.mean(record["validation_loss"] for record in records)
                    if eligible
                    else None
                ),
                "ineligibility_reason": (
                    None if eligible else "At least one planned seed failed"
                ),
                "seeds": sorted(records, key=lambda record: record["seed"]),
            }
        )
    grids = {
        tuple(sorted(candidate["lr"] for candidate in group))
        for group in grouped.values()
    }
    if len(grids) != 1:
        raise ValueError(
            "All retained recipes must have the same planned learning-rate search grid"
        )
    if any(
        len(group) != len({candidate["lr"] for candidate in group})
        for group in grouped.values()
    ):
        raise ValueError(
            "Numeric learning rates cannot have duplicate candidate aliases"
        )
    selected = copy.deepcopy(pilot_spec)
    selected.update(
        name=pilot_spec["name"] + "_calibrated", phase="pilot", evaluate_test=False
    )
    decisions = []
    for architecture in selected["architectures"]:
        base = architecture["name"]
        eligible = [candidate for candidate in grouped[base] if candidate["eligible"]]
        if not eligible:
            raise ValueError(
                f"No candidate completed every calibration seed for recipe {base}"
            )
        winner = min(
            eligible,
            key=lambda candidate: (
                candidate["mean_validation_loss"],
                candidate["lr"],
                candidate["candidate"],
            ),
        )
        overrides = architecture.setdefault("training_overrides", {})
        overrides["lr"] = winner["lr"]
        decisions.append(
            {
                "architecture_id": base,
                "selected_candidate": winner["candidate"],
                "selected_lr": winner["lr"],
                "candidates": grouped[base],
            }
        )
    receipt = {
        "schema_version": "dendritic_scaling_lr_selection_v1",
        "method": "validation mean",
        "metric_split": "validation",
        "status": "completed",
        "campaigns": [source["provenance"] for source in sources],
        "source_comparison_policy": "Every frozen source file is verified against its own manifest. Combined campaigns require identical classification/production code; only select.py, analyze.py, and the unrelated language.py may differ, with their hashes recorded.",
        "input_pilot_spec_sha256": _identity(pilot_spec),
        "selected_pilot_spec_sha256": _identity(selected),
        "calibration_seeds": sorted(calibration_seeds),
        "pilot_seeds": sorted(pilot_seeds),
        "tie_break": "Lowest mean validation loss, then lowest numeric learning rate, then candidate name",
        "failure_policy": "A candidate with any failed planned seed is ineligible; failures remain visible",
        "scope": "All model recipes are retained. Only learning rates are selected; these calibration scores are not independent pilot or test results.",
        "task_status_counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("completed", "failed")
        },
        "decisions": decisions,
    }
    if len(sources) == 1:
        receipt.update(
            {
                key: sources[0]["provenance"][key]
                for key in (
                    "campaign",
                    "calibration_manifest_sha256",
                    "calibration_spec_sha256",
                )
            }
        )
    return selected, receipt


def _atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        action="append",
        required=True,
        help="Repeat for compatible calibration extensions with additional learning rates",
    )
    parser.add_argument("--pilot-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite selection directory {args.output_dir}"
        )
    selected, receipt = select_calibration(
        args.campaign, json.loads(args.pilot_spec.read_text())
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(args.output_dir / "spec.json", selected)
    # Written last: this file is the completion marker for the two-file output.
    _atomic_json(args.output_dir / "selection.json", receipt)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "selected_recipes": len(receipt["decisions"]),
            }
        )
    )


if __name__ == "__main__":
    main()

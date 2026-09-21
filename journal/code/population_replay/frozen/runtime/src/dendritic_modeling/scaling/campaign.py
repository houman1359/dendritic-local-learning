"""Prepare immutable, parameter-accounted experiment grids and execute one task.

Preparation never submits jobs. A campaign contains a frozen source snapshot,
resolved model configurations, source identities, and explicit data/seed axes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _source_root() -> Path:
    return Path(__file__).resolve().parents[2]


def prepare_campaign(
    spec: dict, output_dir: str | Path, *, snapshot: bool = True
) -> dict:
    """Resolve every budget before publishing a runnable campaign manifest.

    A failure leaves an incomplete directory without manifest.json. It must be
    inspected and removed explicitly before retrying, preventing mixed campaigns.
    """
    import torch

    from .models import match_parameter_budget

    torch.set_num_threads(1)
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    configurations = destination / "configs"
    configurations.mkdir()
    required = ("name", "architectures", "parameter_budgets", "data_budgets", "seeds")
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"Missing campaign fields: {missing}")
    architectures = spec["architectures"]
    names = [entry["name"] for entry in architectures]
    if len(names) != len(set(names)):
        raise ValueError("Architecture names must be unique")
    for axis in ("parameter_budgets", "data_budgets", "seeds"):
        if not spec[axis] or len(spec[axis]) != len(set(spec[axis])):
            raise ValueError(f"{axis} must be nonempty and contain no duplicates")
    if any(
        int(x) <= 0 for key in ("parameter_budgets", "data_budgets") for x in spec[key]
    ):
        raise ValueError("Parameter and data budgets must be positive")
    phase = spec.get("phase", "pilot")
    if spec.get("evaluate_test", False) and phase not in {"confirmation", "held_out"}:
        raise ValueError(
            "Test evaluation is reserved for explicit confirmation/held_out campaigns"
        )
    resolved: dict[tuple[str, int], tuple[dict, dict]] = {}
    counted_recipes: dict[tuple[str, int], tuple[dict, dict]] = {}
    for architecture, budget in itertools.product(
        architectures, spec["parameter_budgets"]
    ):
        model = copy.deepcopy(spec.get("model_defaults", {}))
        model.update(
            {
                key: value
                for key, value in architecture.items()
                if key not in {"name", "training_overrides"}
            }
        )
        model.setdefault("seed", 0)
        identity = (hashlib.sha256(_json_bytes(model)).hexdigest(), int(budget))
        if identity not in counted_recipes:
            counted_recipes[identity] = match_parameter_budget(
                model, int(budget), tolerance=float(spec.get("budget_tolerance", 0.02))
            )
        resolved[(architecture["name"], budget)] = copy.deepcopy(
            counted_recipes[identity]
        )
    tasks = []
    for architecture, budget, data_budget, seed in itertools.product(
        architectures, spec["parameter_budgets"], spec["data_budgets"], spec["seeds"]
    ):
        model, accounting = copy.deepcopy(resolved[(architecture["name"], budget)])
        model["seed"] = int(seed)
        # Deliberately couple routing across mechanism arms with the same seed.
        model["topology_seed"] = int(seed)
        data = copy.deepcopy(spec.get("data", {}))
        data["train_size"] = int(data_budget)
        task_id = f"{architecture['name']}_p{budget}_u{data_budget}_s{seed}"
        if not all(c.isalnum() or c in "_-" for c in task_id):
            raise ValueError(
                "Architecture identifiers must use letters, numbers, '_' or '-'"
            )
        training = copy.deepcopy(spec.get("training", {}))
        training.update(copy.deepcopy(architecture.get("training_overrides", {})))
        config = {
            "id": task_id,
            "phase": phase,
            "seed": int(seed),
            "architecture_id": architecture["name"],
            "target_parameters": int(budget),
            "budget_tolerance": float(spec.get("budget_tolerance", 0.02)),
            "evaluate_test": bool(spec.get("evaluate_test", False)),
            "model": model,
            "data": data,
            "training": training,
            "planned_model_report": accounting,
        }
        path = configurations / f"{task_id}.json"
        _write_json(path, config)
        tasks.append(
            {
                "index": len(tasks),
                "id": task_id,
                "config": str(path.relative_to(destination)),
                "config_file_sha256": file_sha256(path),
                "output": f"runs/{task_id}",
                "total_parameters": accounting["total_parameters"],
            }
        )
    source = _source_root()
    source_files = []
    if snapshot:
        frozen = destination / "source" / "src" / "dendritic_modeling"
        shutil.copytree(
            source / "dendritic_modeling",
            frozen,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        for path in sorted(frozen.rglob("*")):
            if path.is_file():
                source_files.append(
                    {
                        "path": str(path.relative_to(destination)),
                        "sha256": file_sha256(path),
                    }
                )
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = None
    _write_json(destination / "spec.json", spec)
    manifest = {
        "schema_version": "dendritic_parameter_scaling_campaign_v1",
        "name": spec["name"],
        "phase": phase,
        "source_commit": commit,
        "source_snapshot": snapshot,
        "source_files": source_files,
        "spec_file_sha256": file_sha256(destination / "spec.json"),
        "tasks": tasks,
        "budget_note": "Actual learned whole-model counts; tolerance is explicit. Search and confirmation are separate.",
    }
    _write_json(destination / "manifest.json", manifest)
    return manifest


def run_task(campaign_dir: str | Path, index: int) -> dict:
    """Verify frozen inputs before running one independent training replicate."""
    from .train import run_experiment

    directory = Path(campaign_dir).resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    if not 0 <= index < len(manifest["tasks"]):
        raise IndexError(f"Task {index} is outside this campaign")
    task = manifest["tasks"][index]
    path = directory / task["config"]
    if file_sha256(path) != task["config_file_sha256"]:
        raise ValueError("Configuration changed after campaign preparation")
    if manifest["source_snapshot"]:
        expected = (
            directory
            / "source"
            / "src"
            / "dendritic_modeling"
            / "scaling"
            / "campaign.py"
        )
        if Path(__file__).resolve() != expected.resolve():
            raise RuntimeError(
                "Run this campaign with PYTHONPATH set to its frozen source/src directory"
            )
        for item in manifest["source_files"]:
            if file_sha256(directory / item["path"]) != item["sha256"]:
                raise ValueError(f"Frozen source changed: {item['path']}")
    result = run_experiment(json.loads(path.read_text()), directory / task["output"])
    if int(result["model_report"]["total_parameters"]) != task["total_parameters"]:
        raise RuntimeError(
            "Training model count differs from instantiated campaign count"
        )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    prepare = actions.add_parser("prepare")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    run = actions.add_parser("run-task")
    run.add_argument("--campaign", type=Path, required=True)
    run.add_argument(
        "--index", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare_campaign(json.loads(args.spec.read_text()), args.output_dir)
        print(
            json.dumps(
                {
                    "campaign": str(args.output_dir.resolve()),
                    "tasks": len(result["tasks"]),
                }
            )
        )
    else:
        result = run_task(args.campaign, args.index)
        print(json.dumps({"id": result["id"], "status": result["status"]}))


if __name__ == "__main__":
    main()

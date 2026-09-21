"""Development-only ReLU scaling/convergence pilot with sealed source and resume.

This is a new follow-up, not a replacement for any previous radial campaign.
The profile, grid, convergence and restart phases are invoked independently.
Fresh-history optimization blocks provide durable resume boundaries. All held-
out endpoints are validation data; this module does not release confirmation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import multiprocessing
import platform
import resource
import shutil
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import radial_width_learning as width
from . import radial_width_tasks as tasks
from . import rank_learning as base
from . import rank_orthogonal_v2 as optimizer

DEPENDENCIES = (
    "radial_followup_campaign.py",
    "radial_width_learning.py",
    "radial_width_tasks.py",
    "radial_width_constructive.py",
    "rank_learning.py",
    "rank_orthogonal_v2.py",
)
PHASES = ("profile", "grid", "convergence", "restarts")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def configuration():
    return {
        "schema": "radial_followup_development_v1",
        "task": "radial_m2",
        "teacher": 400,
        "observation": 0,
        "teacher_seed_base": 2026147001,
        "observation_seed_base": 2026148001,
        "model_seed_base": 2026149001,
        "blocks": 4,
        "radius": 0.5,
        "train_sizes": [8192, 32768],
        "validation_n": 32768,
        "ceilings": [1925, 3845, 7685],
        "geometries": ["width2", "width3", "rank2", "full"],
        "coverage_lines": 8,
        "grid_steps": 300,
        "convergence_steps": 1200,
        "restart_every": 100,
        "profile_steps": 12,
        "diagnostic_shapes": [[1925, 32768], [7685, 8192]],
        "endpoint": "validation",
        "scope": "Fresh development teacher only. Equal four recipes at every geometry, P and N; no selection or confirmation claim. Full projections have a learned direction per branch and hence growing directional width. Rank2 is also a shared linear bottleneck plus ReLU layer. Fixed recipes are paired across N and iteration budgets. Profile measurements gate a separate broad run. Earlier source/results are unchanged.",
    }


def choices():
    return [
        {
            "id": f"{method}_ridge{ridge:.0e}",
            "method": method,
            "fit": asdict(
                base.FitConfig(
                    ridge=ridge,
                    steps=100,
                    learning_rate=0.5,
                    objective_scale=1e4,
                    tolerance_grad=1e-10,
                    tolerance_change=1e-14,
                    history_size=30,
                )
            ),
        }
        for method, ridge in itertools.product(("stein", "random"), (1e-8, 1e-5))
    ]


def validate_config(config):
    if config["endpoint"] != "validation" or config["task"] != "radial_m2":
        raise ValueError("This pilot is quartic development validation only")
    if config["teacher"] in (80, 81, *range(100, 108)):
        raise ValueError("Fresh teacher ID required")
    if config["blocks"] != 4 or config["radius"] <= 0:
        raise ValueError("Four supplied positive-radius balls required")
    if config["geometries"] != ["width2", "width3", "rank2", "full"]:
        raise ValueError("All declared adequate and restricted controls required")
    for key in ("train_sizes", "ceilings"):
        values = config[key]
        if (
            not values
            or values != sorted(set(values))
            or any(not isinstance(v, int) or v < 4 for v in values)
        ):
            raise ValueError(f"Positive increasing integer {key} required")
    if not 1 <= config["profile_steps"] <= config["restart_every"]:
        raise ValueError("Profile must fit within a single restart block")
    if config["grid_steps"] < 1 or config["convergence_steps"] <= config["grid_steps"]:
        raise ValueError("Convergence budget must extend the grid budget")
    if any(
        config[k] % config["restart_every"] for k in ("grid_steps", "convergence_steps")
    ):
        raise ValueError("Grid and extended runs need complete restart blocks")
    for p, n in config["diagnostic_shapes"]:
        if p not in config["ceilings"] or n not in config["train_sizes"]:
            raise ValueError("Diagnostic shape must have its matched grid fit")


def case_key(case):
    return "/".join(
        str(case[k])
        for k in (
            "phase",
            "geometry",
            "ceiling",
            "train_n",
            "choice",
            "restart",
            "steps",
        )
    )


def case_grid(config):
    validate_config(config)
    result = []

    def add(phase, geometry, ceiling, train_n, choice, restart, steps):
        branches = width.branches_under_ceiling(ceiling, geometry)
        inventory = width.parameter_inventory(branches, geometry)
        result.append(
            {
                "phase": phase,
                "task": config["task"],
                "teacher": config["teacher"],
                "observation": config["observation"],
                "family": "relu",
                "geometry": geometry,
                "ceiling": ceiling,
                "train_n": train_n,
                "choice": choice,
                "restart": restart,
                "steps": steps,
                "inventory": inventory,
                "dense_design_MiB": 8 * train_n * (branches + 1) / 2**20,
            }
        )

    for geometry, n in itertools.product(config["geometries"], config["train_sizes"]):
        add(
            "profile",
            geometry,
            max(config["ceilings"]),
            n,
            "stein_ridge1e-08",
            0,
            config["profile_steps"],
        )
    for geometry, p, n, choice in itertools.product(
        config["geometries"], config["ceilings"], config["train_sizes"], choices()
    ):
        add("grid", geometry, p, n, choice["id"], 0, config["grid_steps"])
    for geometry, (p, n) in itertools.product(
        config["geometries"], config["diagnostic_shapes"]
    ):
        add(
            "convergence",
            geometry,
            p,
            n,
            "stein_ridge1e-08",
            0,
            config["convergence_steps"],
        )
        add("restarts", geometry, p, n, "random_ridge1e-08", 1, config["grid_steps"])
    if len({case_key(c) for c in result}) != len(result):
        raise ValueError("Duplicate cases")
    return result


def verify(root):
    root = Path(root)
    receipt = read(root / "initialized.json")
    for name, expected in receipt["bindings"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    return receipt


def create_dataset(root, config):
    path = root / "data" / "development.npz"
    receipt_path = path.with_suffix(".json")
    if receipt_path.exists():
        if sha(path) != read(receipt_path)["sha256"]:
            raise ValueError("Dataset changed")
        return
    if path.exists():
        raise ValueError("Uncommitted dataset exists; preserve and inspect it")
    teacher = tasks.RadialTeacher(
        2,
        config["teacher_seed_base"] + config["teacher"],
        blocks=config["blocks"],
        radius=config["radius"],
    )
    specification = teacher.specification()
    private = root / "private_teacher.json"
    if private.exists():
        if read(private) != specification:
            raise ValueError("Private teacher changed")
    else:
        write(private, specification)
    seed = (
        config["observation_seed_base"]
        + 100 * config["teacher"]
        + 2 * config["observation"]
    )
    x = tasks.sample_inputs(
        seed, max(config["train_sizes"]), config["blocks"], config["radius"]
    )
    z = tasks.sample_inputs(
        seed + 1, config["validation_n"], config["blocks"], config["radius"]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(
            handle,
            x_train=x,
            y_train=teacher.evaluate(x),
            x_endpoint=z,
            y_endpoint=teacher.evaluate(z),
        )
    write(
        receipt_path,
        {
            "utc": now(),
            "sha256": sha(path),
            "endpoint": "validation",
            "seed": seed,
            "teacher": config["teacher"],
            "observation": config["observation"],
            "nested_train_sizes": config["train_sizes"],
            "private_teacher_sha256": sha(private),
            "initialized_sha256": sha(root / "initialized.json"),
        },
    )


def initialize(root, config=None):
    root = Path(root).resolve()
    config = configuration() if config is None else config
    validate_config(config)
    if (root / "initialized.json").exists():
        verify(root)
        if read(root / "config.json") != config:
            raise ValueError("Existing configuration differs")
        create_dataset(root, config)
        return
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "source"
    snapshot.mkdir(exist_ok=True)
    for name in DEPENDENCIES:
        origin, target = Path(__file__).with_name(name), snapshot / name
        if target.exists():
            if sha(target) != sha(origin):
                raise ValueError(f"Existing snapshot differs: {name}")
        else:
            shutil.copyfile(origin, target)
    init = snapshot / "__init__.py"
    if not init.exists():
        init.write_text("")
    manifest = case_grid(config)
    write(root / "config.json", config)
    write(root / "choices.json", choices())
    write(root / "manifest.json", manifest)
    write(
        root / "environment.json",
        {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "scope": "Recorded software versions; launcher fixes single-threaded BLAS and CPU FP64.",
        },
    )
    write(
        root / "resource_plan.json",
        {
            "phase_counts": {
                phase: sum(c["phase"] == phase for c in manifest) for phase in PHASES
            },
            "maximum_dense_design_MiB": max(c["dense_design_MiB"] for c in manifest),
            "scope": "Dense design size is only one array, not peak memory. Measure profile process peak RSS and elapsed time before choosing parallelism or expanding. CPU FP64; no GPU allocation assumed. 300/1200 runs restart every 100 iterations, so the first 300 iterations share the schedule. Profile uses 12 iterations and cannot certify convergence.",
        },
    )
    paths = sorted(snapshot.glob("*.py")) + [
        root / n
        for n in (
            "config.json",
            "choices.json",
            "manifest.json",
            "resource_plan.json",
            "environment.json",
        )
    ]
    paths += [p for p in (root / "protocol.md", root / "run.slurm") if p.exists()]
    write(
        root / "initialized.json",
        {
            "utc": now(),
            "bindings": {str(p.relative_to(root)): sha(p) for p in paths},
            "source_execution": "Run python -m source.radial_followup_campaign with this root on PYTHONPATH.",
        },
    )
    create_dataset(root, config)


def verify_result(root, receipt, case):
    if receipt["status"] != "complete" or receipt["case"] != case:
        raise ValueError("Completed case binding differs")
    if receipt["initialized_sha256"] != sha(root / "initialized.json"):
        raise ValueError("Initialized source binding differs")
    for name, expected in receipt["artifacts"].items():
        if sha(root / name) != expected:
            raise ValueError(f"Result artifact changed: {name}")
    if receipt["dataset_sha256"] != sha(root / "data" / "development.npz"):
        raise ValueError("Case dataset changed")
    if receipt["dataset_receipt_sha256"] != sha(root / "data" / "development.json"):
        raise ValueError("Case dataset receipt changed")


def _resume_stage(root, stage_dir, case):
    marker = stage_dir / "complete.json"
    if not marker.exists():
        return None
    receipt = read(marker)
    if receipt["case"] != case or receipt["initialized_sha256"] != sha(
        root / "initialized.json"
    ):
        raise ValueError("Stage binding changed")
    for name, expected in receipt["artifacts"].items():
        if sha(root / name) != expected:
            raise ValueError("Stage artifact changed")
    return receipt


def run_case(root_string, case):
    root = Path(root_string).resolve()
    torch.set_num_threads(1)
    verify(root)
    config = read(root / "config.json")
    if case not in read(root / "manifest.json"):
        raise ValueError("Case outside frozen manifest")
    directory = root / "cases" / case_key(case)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = directory / "result.json"
        if result.exists():
            receipt = read(result)
            verify_result(root, receipt, case)
            return receipt
        attempt = directory / f"attempt_{len(list(directory.glob('attempt_*'))):03d}"
        attempt.mkdir(exist_ok=False)
        write(attempt / "case.json", case)
        started = time.monotonic()
        try:
            return _fit_case(root, directory, attempt, case, config, started)
        except Exception:
            failure = {
                "utc": now(),
                "status": "failed",
                "case": case,
                "elapsed_seconds": time.monotonic() - started,
                "traceback": traceback.format_exc(),
            }
            write(attempt / "failure.json", failure)
            return failure


def _fit_case(root, directory, attempt, case, config, started):
    data_path = root / "data" / "development.npz"
    dataset_receipt = read(data_path.with_suffix(".json"))
    if dataset_receipt["sha256"] != sha(data_path):
        raise ValueError("Dataset hash differs")
    if dataset_receipt["initialized_sha256"] != sha(root / "initialized.json"):
        raise ValueError("Dataset source binding differs")
    with np.load(data_path, allow_pickle=False) as data:
        x, y = (
            data["x_train"][: case["train_n"]].copy(),
            data["y_train"][: case["train_n"]].copy(),
        )
    if len(x) != case["train_n"]:
        raise ValueError("TRAIN prefix exceeds archive")
    choice = next(c for c in read(root / "choices.json") if c["id"] == case["choice"])
    seed = config["model_seed_base"] + 100 * case["teacher"] + 100000 * case["restart"]
    bank, init_receipt = width.estimate_bank(
        choice["method"], (x, y), seed=seed, radius=config["radius"]
    )
    model = width.model_under_ceiling(
        case["ceiling"],
        "relu",
        case["geometry"],
        bank=bank,
        train_data=(x, y),
        seed=seed,
        allocation_seed=case["teacher"] + 1000 * case["restart"],
        coverage_lines=config["coverage_lines"],
        initializer_receipt=init_receipt,
        base_module=base,
    )
    counted = model.counted_metadata()
    if counted["stored_parameters"] != case["inventory"]["stored_parameters"]:
        raise ValueError("Manifest parameter inventory differs from actual model")
    write(attempt / "initializer.json", init_receipt)
    artifacts = {
        str((attempt / "initializer.json").relative_to(root)): sha(
            attempt / "initializer.json"
        )
    }
    remaining, stage, fits, stages_resumed = case["steps"], 0, [], 0
    while remaining:
        block_steps = min(remaining, config["restart_every"])
        stage_dir = directory / f"stage_{stage:03d}"
        stage_dir.mkdir(exist_ok=True)
        completed = _resume_stage(root, stage_dir, case)
        predecessor_sha256 = (
            None
            if stage == 0
            else sha(directory / f"stage_{stage - 1:03d}" / "complete.json")
        )
        if completed is None:
            stage_attempt = (
                stage_dir / f"attempt_{len(list(stage_dir.glob('attempt_*'))):03d}"
            )
            stage_attempt.mkdir(exist_ok=False)
            fit_config = dict(choice["fit"], steps=block_steps)
            fit = optimizer.fit_restarted(
                model.model,
                model.expanded_inputs(x),
                y,
                base.FitConfig(**fit_config),
                stages=1,
                rebalance=True,
                output_dir=stage_attempt / "fit",
            )
            files = sorted((stage_attempt / "fit").rglob("*"))
            completed = {
                "utc": now(),
                "case": case,
                "stage": stage,
                "steps": block_steps,
                "initialized_sha256": sha(root / "initialized.json"),
                "dataset_sha256": sha(data_path),
                "dataset_receipt_sha256": sha(data_path.with_suffix(".json")),
                "predecessor_sha256": predecessor_sha256,
                "state_path": str(
                    (stage_attempt / "fit" / "final.npz").relative_to(root)
                ),
                "fit_path": str((stage_attempt / "fit" / "fit.json").relative_to(root)),
                "artifacts": {
                    str(p.relative_to(root)): sha(p) for p in files if p.is_file()
                },
            }
            write(stage_dir / "complete.json", completed)
        else:
            if completed["steps"] != block_steps or completed["dataset_sha256"] != sha(
                data_path
            ):
                raise ValueError(
                    "Resumed stage TRAIN binding or iteration schedule differs"
                )
            if (
                completed["stage"] != stage
                or completed["predecessor_sha256"] != predecessor_sha256
            ):
                raise ValueError("Resumed stage predecessor or index differs")
            if completed["dataset_receipt_sha256"] != sha(
                data_path.with_suffix(".json")
            ):
                raise ValueError("Resumed stage dataset receipt differs")
            model = width.load_state(root / completed["state_path"], base_module=base)
            fit = read(root / completed["fit_path"])
            stages_resumed += 1
        fits.append(fit)
        artifacts.update(completed["artifacts"])
        artifacts[str((stage_dir / "complete.json").relative_to(root))] = sha(
            stage_dir / "complete.json"
        )
        remaining -= block_steps
        stage += 1
    # Held-out data is materialized only after every optimization stage terminates.
    with np.load(data_path, allow_pickle=False) as data:
        z, target = data["x_endpoint"].copy(), data["y_endpoint"].copy()
    with torch.no_grad():
        train_prediction = model.model(model.expanded_inputs(x)).cpu().numpy()
        endpoint_prediction = model.model(model.expanded_inputs(z)).cpu().numpy()
    if (
        not np.isfinite(train_prediction).all()
        or not np.isfinite(endpoint_prediction).all()
    ):
        raise FloatingPointError("Nonfinite prediction")
    predictions = attempt / "predictions.npz"
    with predictions.open("xb") as handle:
        np.savez_compressed(
            handle, train=train_prediction, endpoint=endpoint_prediction
        )
    artifacts[str(predictions.relative_to(root))] = sha(predictions)
    receipt = {
        "utc": now(),
        "status": "complete",
        "case": case,
        "endpoint": "validation",
        "initialized_sha256": sha(root / "initialized.json"),
        "dataset_sha256": sha(data_path),
        "dataset_receipt_sha256": sha(data_path.with_suffix(".json")),
        "artifacts": artifacts,
        "counted_inventory": counted,
        "state_path": completed["state_path"],
        "predictions_path": str(predictions.relative_to(root)),
        "metrics": {
            "train_mse": float(np.mean((train_prediction - y) ** 2)),
            "validation_mse": float(np.mean((endpoint_prediction - target) ** 2)),
        },
        "train_rows": len(y),
        "validation_rows": len(target),
        "terminal_objective": fits[-1]["terminal_objective"],
        "terminal_unscaled_gradient_l2": fits[-1]["terminal_unscaled_gradient_l2"],
        "iterations": sum(f["iterations"] for f in fits),
        "closure_calls": sum(f["closure_calls"] for f in fits),
        "fit_elapsed_seconds": sum(f["elapsed_seconds"] for f in fits),
        "elapsed_seconds_this_attempt": time.monotonic() - started,
        "process_peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "rss_scope": "Linux process lifetime peak RSS, not incremental per-case memory.",
        "resumed_complete_stages": stages_resumed,
        "completed_restart_blocks": len(fits),
        "scope": "Development observation only; no independent numerical audit or confirmation claim implied by status complete.",
    }
    write(directory / "result.json", receipt)
    return receipt


def execute(root, phase, workers=1, limit=None):
    root = Path(root).resolve()
    verify(root)
    if Path(__file__).resolve() != root / "source" / "radial_followup_campaign.py":
        raise ValueError("Execute the immutable source snapshot")
    if workers < 1 or phase not in PHASES or (limit is not None and limit < 1):
        raise ValueError(
            "Valid phase, worker count and positive optional limit required"
        )
    if phase != "profile":
        marker = root / "profile_complete.json"
        if not marker.exists():
            raise ValueError("Complete and inspect the profile before broader phases")
        verify_phase(root, "profile")
    cases = [c for c in read(root / "manifest.json") if c["phase"] == phase]
    cases.sort(key=lambda c: (-c["train_n"], -c["ceiling"], case_key(c)))
    chosen = cases if limit is None else cases[:limit]
    # One case per process makes the OS peak RSS attributable to that fit.
    with multiprocessing.get_context("spawn").Pool(
        processes=workers, maxtasksperchild=1
    ) as pool:
        for index, row in enumerate(
            pool.imap_unordered(_worker, [(str(root), c) for c in chosen], chunksize=1)
        ):
            print(
                json.dumps(
                    {
                        "done": index + 1,
                        "total": len(chosen),
                        "case": case_key(row["case"]),
                        "status": row["status"],
                        "fit_seconds": row.get("fit_elapsed_seconds"),
                        "peak_rss_MiB": row.get("process_peak_rss_MiB"),
                    }
                ),
                flush=True,
            )
    paths = [root / "cases" / case_key(c) / "result.json" for c in cases]
    if all(p.exists() for p in paths):
        rows = [read(p) for p in paths]
        for row, case in zip(rows, cases, strict=True):
            verify_result(root, row, case)
        marker = root / f"{phase}_complete.json"
        value = {
            "utc": now(),
            "phase": phase,
            "status": "complete",
            "fits": len(rows),
            "results": {str(p.relative_to(root)): sha(p) for p in paths},
            "sum_fit_seconds": sum(r["fit_elapsed_seconds"] for r in rows),
            "maximum_process_peak_rss_MiB": max(
                r["process_peak_rss_MiB"] for r in rows
            ),
        }
        if marker.exists():
            verify_phase(root, phase)
        else:
            write(marker, value)
    elif limit is None:
        raise RuntimeError(
            "Incomplete phase; preserved failed attempts require inspection"
        )


def _worker(arguments):
    return run_case(*arguments)


def verify_phase(root, phase):
    receipt = read(root / f"{phase}_complete.json")
    cases = [c for c in read(root / "manifest.json") if c["phase"] == phase]
    expected = {
        str((root / "cases" / case_key(c) / "result.json").relative_to(root))
        for c in cases
    }
    if set(receipt["results"]) != expected or receipt["fits"] != len(cases):
        raise ValueError("Phase inventory differs")
    for name, expected_sha in receipt["results"].items():
        if sha(root / name) != expected_sha:
            raise ValueError("Phase result changed")
    for case in cases:
        verify_result(root, read(root / "cases" / case_key(case) / "result.json"), case)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "run", "status"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, default="profile")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "initialize":
        initialize(args.root)
    elif args.action == "run":
        execute(args.root, args.phase, args.workers, args.limit)
    else:
        verify(args.root)
        manifest = read(args.root / "manifest.json")
        print(
            json.dumps(
                {
                    phase: {
                        "expected": sum(c["phase"] == phase for c in manifest),
                        "completed": sum(
                            (args.root / "cases" / case_key(c) / "result.json").exists()
                            for c in manifest
                            if c["phase"] == phase
                        ),
                        "barrier": (args.root / f"{phase}_complete.json").exists(),
                    }
                    for phase in PHASES
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

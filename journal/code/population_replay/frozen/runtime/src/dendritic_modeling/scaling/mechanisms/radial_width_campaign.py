"""Prospective radial response study with nested sample sizes and sealed releases.

The lifecycle follows the completed rank/width campaign, whose source and
outputs remain unchanged. Only the coordinator evaluates private teachers.
Fitting workers receive supplied input blocks and scalar TRAIN prefixes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from . import (
    radial_width_learning as width,
    radial_width_tasks as tasks,
    rank_learning as base,
    rank_orthogonal_v2 as optimizer,
)

FORECAST_PAIRS = (
    ("width2", "rank2"),
    ("width4", "rank2"),
    ("width8", "rank2"),
    ("rank2", "full"),
    ("width8", "full"),
)


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


def with_counts(config):
    """Compute declared counts; grid construction independently checks them."""
    arms = (
        len(config["tasks"])
        * len(config["geometries"])
        * len(config["families"])
        * len(config["train_sizes"])
    )
    config["fit_counts"] = {
        "development": arms
        * len(config["development_teachers"])
        * len(config["calibration_p"])
        * 4,
        "bridge": arms * len(config["development_teachers"]),
        "confirmation": arms
        * len(config["confirmation_teachers"])
        * 2
        * len(config["primary_p"]),
    }
    config["expected_fits"] = sum(config["fit_counts"].values())
    return config


def configuration(confirmation_teachers=8, steps=300):
    if confirmation_teachers not in (2, 4, 8) or int(steps) != steps or steps < 3:
        raise ValueError(
            "Declared teacher count and at least three total iterations required"
        )
    return with_counts(
        {
            "schema": "radial_width_v1",
            "tasks": ["radial_m2", "radial_m4"],
            "geometries": list(width.GEOMETRIES),
            "families": ["shunt", "relu", "tanh"],
            "development_teachers": [80, 81],
            "confirmation_teachers": list(range(100, 100 + confirmation_teachers)),
            "teacher_seed_base": 2026141001,
            "observation_seed_base": 2026142001,
            "model_seed_base": 2026143001,
            "blocks": 4,
            "radius": 0.5,
            "max_train_n": 32768,
            "train_sizes": [2048, 8192],
            "validation_n": 8192,
            "test_n": 32768,
            "calibration_p": [245, 965],
            "bridge_p": 485,
            "withheld_p": 1925,
            "primary_p": [245, 485, 965, 1925],
            "coverage_lines": 8,
            "stages": 3,
            "steps": int(steps),
            "log_risk_threshold": 1e-18,
            "bootstrap_draws": 10000,
            "bootstrap_seed": 2026146001,
            "constructive_n": [512, 2048, 8192, 32768],
            "constructive_intervals": [4, 8, 16, 32, 64],
            "scope": "Noiseless radial quartic and degree-eight responses on supplied independent balls. "
            "Teacher draws change orientation only; two paired observation draws per teacher. "
            "Nested TRAIN prefixes and common endpoint inputs across all arms and powers. "
            "Generic fitting knows the block law and rank prior but not the radial profile. "
            "The known-profile constructive learner is a separate procedure. All outcomes retained.",
        }
    )


def choices(config):
    return [
        {
            "id": f"{method}_ridge{ridge:.0e}",
            "method": method,
            "fit": asdict(
                base.FitConfig(
                    ridge=ridge,
                    steps=config["steps"],
                    learning_rate=0.5,
                    objective_scale=1e4,
                    tolerance_grad=1e-10,
                    tolerance_change=1e-14,
                    history_size=30,
                )
            ),
        }
        for method, ridge in itertools.product(["stein", "random"], [1e-8, 1e-5])
    ]


def recipe_key(task, geometry, family, train_n):
    return f"{task}/{geometry}/{family}/{train_n}"


def verify(root):
    receipt = read(root / "initialized.json")
    for name, value in receipt["bindings"].items():
        if sha(root / name) != value:
            raise ValueError(f"Frozen file changed: {name}")
    return receipt


def verify_selection(root):
    verify(root)
    receipt = read(root / "selection_complete.json")
    if sha(root / "selected_recipes.json") != receipt["selected_sha256"]:
        raise ValueError("Selected recipe changed")
    if sha(root / "development_complete.json") != receipt["development_barrier_sha256"]:
        raise ValueError("Selection development barrier changed")


def verify_forecast(root):
    verify_selection(root)
    receipt = read(root / "forecast_complete.json")
    if sha(root / "frozen_forecasts.json") != receipt["forecast_sha256"]:
        raise ValueError("Forecast changed")
    if sha(root / "bridge_complete.json") != receipt["bridge_barrier_sha256"]:
        raise ValueError("Forecast bridge barrier changed")
    if (
        sha(root / "selected_recipes.json")
        != read(root / "frozen_forecasts.json")["selected_sha256"]
    ):
        raise ValueError("Forecast recipe binding changed")


def observation_seed(config, teacher, observation):
    return config["observation_seed_base"] + 100 * teacher + 2 * observation


def dataset_path(root, stage, task, teacher, observation):
    release = "confirmation" if stage == "confirmation" else "development"
    return root / "data" / release / f"{task}_t{teacher}_o{observation}.npz"


def create_datasets(root, config, release):
    if release not in ("development", "confirmation"):
        raise ValueError("Unknown data release")
    if release == "confirmation":
        verify_forecast(root)
    endpoint = "test" if release == "confirmation" else "validation"
    for index in config[release + "_teachers"]:
        for task in config["tasks"]:
            m = int(task.removeprefix("radial_m"))
            teacher = tasks.RadialTeacher(
                m,
                config["teacher_seed_base"] + index,
                blocks=config["blocks"],
                radius=config["radius"],
            )
            private_path = root / "private_teachers" / f"{release}_{task}_t{index}.json"
            specification = teacher.specification()
            if private_path.exists():
                if read(private_path) != specification:
                    raise ValueError("Existing private teacher changed")
            else:
                write(private_path, specification)
            for observation in range(2 if release == "confirmation" else 1):
                path = dataset_path(root, release, task, index, observation)
                if path.exists() and path.with_suffix(".json").exists():
                    if sha(path) != read(path.with_suffix(".json"))["sha256"]:
                        raise ValueError("Existing dataset changed")
                    continue
                seed = observation_seed(config, index, observation)
                x = tasks.sample_inputs(
                    seed, config["max_train_n"], config["blocks"], config["radius"]
                )
                z = tasks.sample_inputs(
                    seed + 1,
                    config[endpoint + "_n"],
                    config["blocks"],
                    config["radius"],
                )
                arrays = {
                    "x_train": x,
                    "y_train": teacher.evaluate(x),
                    "x_endpoint": z,
                    "y_endpoint": teacher.evaluate(z),
                }
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("xb") as handle:
                    np.savez_compressed(handle, **arrays)
                write(
                    path.with_suffix(".json"),
                    {
                        "utc": now(),
                        "release": release,
                        "endpoint": endpoint,
                        "task": task,
                        "teacher": index,
                        "observation": observation,
                        "seed": seed,
                        "sha256": sha(path),
                        "max_train_n": config["max_train_n"],
                        "nested_train_sizes": config["train_sizes"],
                        "private_specification_sha256": sha(private_path),
                        "forecast_sha256": (
                            sha(root / "frozen_forecasts.json")
                            if release == "confirmation"
                            else None
                        ),
                    },
                )


def initialize(root, config):
    if set(config["train_sizes"]) - set(config["constructive_n"]):
        raise ValueError(
            "Generic sample sizes must be present in the declared shared prefix grid"
        )
    if max(config["constructive_n"]) > config["max_train_n"]:
        raise ValueError("Archive too small for declared TRAIN prefixes")
    if set(config["development_teachers"]) & set(config["confirmation_teachers"]):
        raise ValueError("Development and confirmation teachers overlap")
    if config != with_counts(dict(config)):
        raise ValueError("Declared campaign counts are inconsistent")
    if (root / "initialized.json").exists():
        verify(root)
        if read(root / "config.json") != config:
            raise ValueError("Existing configuration differs")
    else:
        timing = read(root / "timing_decision.json")
        if timing["confirmation_teacher_count"] != len(config["confirmation_teachers"]):
            raise ValueError("Teacher count differs from resource decision")
        write(root / "config.json", config)
        write(root / "choices.json", choices(config))
        write(
            root / "parameter_inventory.json",
            width.matched_ceiling_inventory(config["primary_p"]),
        )
        paths = [
            root / name
            for name in (
                "config.json",
                "choices.json",
                "protocol.md",
                "timing_decision.json",
                "parameter_inventory.json",
            )
        ]
        paths += sorted((root / "source").glob("*.py"))
        if Path(__file__).resolve() not in [p.resolve() for p in paths]:
            raise ValueError("Run the source snapshot inside the study root")
        write(
            root / "initialized.json",
            {
                "utc": now(),
                "bindings": {str(p.relative_to(root)): sha(p) for p in paths},
            },
        )
    create_datasets(root, config, "development")


def case_key(case):
    return "/".join(
        str(case[k])
        for k in (
            "stage",
            "task",
            "geometry",
            "family",
            "teacher",
            "observation",
            "train_n",
            "choice",
            "coverage",
            "rebalance",
            "ceiling",
        )
    )


def case_grid(config, stage, recipes=None):
    if stage not in ("development", "bridge", "confirmation"):
        raise ValueError("Unknown stage")
    indices = (
        config["confirmation_teachers"]
        if stage == "confirmation"
        else config["development_teachers"]
    )
    result = []
    for task, geometry, family, teacher, observation, train_n in itertools.product(
        config["tasks"],
        config["geometries"],
        config["families"],
        indices,
        range(2 if stage == "confirmation" else 1),
        config["train_sizes"],
    ):
        selected = (
            [c["id"] for c in choices(config)]
            if stage == "development"
            else [recipes[recipe_key(task, geometry, family, train_n)]["choice"]]
        )
        budgets = (
            config["calibration_p"]
            if stage == "development"
            else [config["bridge_p"]] if stage == "bridge" else config["primary_p"]
        )
        for choice, ceiling in itertools.product(selected, budgets):
            result.append(
                {
                    "stage": stage,
                    "task": task,
                    "geometry": geometry,
                    "family": family,
                    "teacher": teacher,
                    "observation": observation,
                    "train_n": train_n,
                    "choice": choice,
                    "coverage": "plane8",
                    "rebalance": True,
                    "ceiling": ceiling,
                }
            )
    if len(result) != config["fit_counts"][stage] or len(
        {case_key(c) for c in result}
    ) != len(result):
        raise ValueError("Case count or uniqueness mismatch")
    return sorted(result, key=lambda c: (-c["ceiling"], -c["train_n"], case_key(c)))


@lru_cache(maxsize=64)
def load_train(path_string):
    path = Path(path_string)
    if sha(path) != read(path.with_suffix(".json"))["sha256"]:
        raise ValueError("Dataset hash differs")
    with np.load(path, allow_pickle=False) as data:
        return data["x_train"].copy(), data["y_train"].copy()


def verify_result(root, receipt, case):
    if receipt["case"] != case or receipt["initialized_sha256"] != sha(
        root / "initialized.json"
    ):
        raise ValueError("Completed case/source binding mismatch")
    for stem in ("data", "fit", "state", "predictions"):
        if stem != "data" and receipt["status"] != "complete":
            continue
        if sha(receipt[stem + "_path"]) != receipt[stem + "_sha256"]:
            raise ValueError(f"Completed {stem} artifact changed")
    expected = dataset_path(
        root, case["stage"], case["task"], case["teacher"], case["observation"]
    )
    if Path(receipt["data_path"]).resolve() != expected.resolve():
        raise ValueError("Case data path differs from declared split")


def run_case(root_string, case):
    root = Path(root_string)
    torch.set_num_threads(1)
    verify(root)
    config = read(root / "config.json")
    if case["stage"] == "confirmation":
        verify_forecast(root)
    elif case["stage"] == "bridge":
        verify_selection(root)
    recipes = (
        read(root / "selected_recipes.json") if case["stage"] != "development" else None
    )
    if case not in case_grid(config, case["stage"], recipes):
        raise ValueError("Worker case outside frozen grid")
    case_dir = root / "cases" / case_key(case)
    result_path = case_dir / "result.json"
    if result_path.exists():
        receipt = read(result_path)
        verify_result(root, receipt, case)
        return receipt
    case_dir.mkdir(parents=True, exist_ok=True)
    attempt_number = len(list(case_dir.glob("attempt_*")))
    attempt = case_dir / f"attempt_{attempt_number:03d}"
    attempt.mkdir(exist_ok=False)
    write(attempt / "case.json", case)
    started = time.monotonic()
    data_path = dataset_path(
        root, case["stage"], case["task"], case["teacher"], case["observation"]
    )
    receipt = {
        "case": case,
        "utc": now(),
        "data_path": str(data_path),
        "data_sha256": sha(data_path),
        "attempt_path": str(attempt),
        "previous_incomplete_attempts": attempt_number,
        "initialized_sha256": sha(root / "initialized.json"),
    }
    try:
        full_x, full_y = load_train(str(data_path))
        n = case["train_n"]
        if not 1 <= n <= len(full_x):
            raise ValueError("Invalid TRAIN prefix")
        x, y = full_x[:n].copy(), full_y[:n].copy()
        choice = next(
            c for c in read(root / "choices.json") if c["id"] == case["choice"]
        )
        seed = (
            config["model_seed_base"] + 100 * case["teacher"] + 2 * case["observation"]
        )
        bank, init_receipt = width.estimate_bank(
            choice["method"], (x, y), seed=seed, radius=config["radius"]
        )
        model = width.model_under_ceiling(
            case["ceiling"],
            case["family"],
            case["geometry"],
            bank=bank,
            train_data=(x, y),
            seed=seed,
            allocation_seed=case["teacher"],
            coverage_lines=config["coverage_lines"],
            initializer_receipt=init_receipt,
            base_module=base,
        )
        write(attempt / "initializer.json", init_receipt)
        fit = optimizer.fit_restarted(
            model.model,
            model.expanded_inputs(x),
            y,
            base.FitConfig(**choice["fit"]),
            stages=config["stages"],
            rebalance=True,
            output_dir=attempt / "fit",
        )
        # The optimizer has terminated before any held-out array is materialized.
        with np.load(data_path, allow_pickle=False) as data:
            endpoint_x, endpoint_y = (
                data["x_endpoint"].copy(),
                data["y_endpoint"].copy(),
            )
        with torch.no_grad():
            train_prediction = model.model(model.expanded_inputs(x)).cpu().numpy()
            endpoint_prediction = (
                model.model(model.expanded_inputs(endpoint_x)).cpu().numpy()
            )
        if (
            not np.isfinite(train_prediction).all()
            or not np.isfinite(endpoint_prediction).all()
        ):
            raise FloatingPointError("Nonfinite endpoint")
        with (attempt / "predictions.npz").open("xb") as handle:
            np.savez_compressed(
                handle, train=train_prediction, endpoint=endpoint_prediction
            )
        receipt.update(
            status="complete",
            metrics={
                "train_mse": float(np.mean((train_prediction - y) ** 2)),
                "endpoint_mse": float(np.mean((endpoint_prediction - endpoint_y) ** 2)),
            },
            ridge=choice["fit"]["ridge"],
            train_rows=n,
            endpoint_rows=len(endpoint_y),
            fit_path=str(attempt / "fit/fit.json"),
            fit_sha256=sha(attempt / "fit/fit.json"),
            state_path=str(attempt / "fit/final.npz"),
            state_sha256=sha(attempt / "fit/final.npz"),
            predictions_path=str(attempt / "predictions.npz"),
            predictions_sha256=sha(attempt / "predictions.npz"),
            counted_inventory=model.counted_metadata(),
            objective=fit["terminal_objective"],
            closure_calls=fit["closure_calls"],
            iterations=fit["iterations"],
            fit_elapsed_seconds=fit["elapsed_seconds"],
        )
    except Exception:
        receipt.update(status="failed", traceback=traceback.format_exc())
    receipt["elapsed_seconds"] = time.monotonic() - started
    write(result_path, receipt)
    return receipt


def stage_results(root, stage, config):
    recipes = read(root / "selected_recipes.json") if stage != "development" else None
    cases = case_grid(config, stage, recipes)
    paths = [root / "cases" / case_key(c) / "result.json" for c in cases]
    marker = root / f"{stage}_complete.json"
    if marker.exists():
        record = read(marker)
        if record["fits"] != len(cases) or record["results"] != {
            str(p.relative_to(root)): sha(p) for p in paths
        }:
            raise ValueError("Stage results differ from completion barrier")
    rows = [read(p) for p in paths]
    for row, case in zip(rows, cases, strict=True):
        if row["status"] != "complete" or row["case"] != case:
            raise ValueError(
                "All correctly bound complete outcomes required; failures cannot be dropped"
            )
        verify_result(root, row, case)
    return rows


def execute_stage(root, stage, workers):
    verify(root)
    config = read(root / "config.json")
    if stage == "confirmation":
        verify_forecast(root)
    elif stage == "bridge":
        verify_selection(root)
    if stage != "development":
        stage_results(root, "development", config)
    if stage == "confirmation":
        stage_results(root, "bridge", config)
        create_datasets(root, config, "confirmation")
    recipes = read(root / "selected_recipes.json") if stage != "development" else None
    cases = case_grid(config, stage, recipes)
    marker = root / f"{stage}_complete.json"
    if marker.exists():
        stage_results(root, stage, config)
        return
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(run_case, str(root), c) for c in cases]
        for index, future in enumerate(as_completed(futures)):
            row = future.result()
            print(
                json.dumps(
                    {
                        "stage": stage,
                        "done": index + 1,
                        "total": len(cases),
                        "case": case_key(row["case"]),
                        "status": row["status"],
                    }
                ),
                flush=True,
            )
    rows = stage_results(root, stage, config)
    write(
        marker,
        {
            "utc": now(),
            "status": "complete",
            "fits": len(rows),
            "workers": workers,
            "elapsed_seconds": time.monotonic() - started,
            "results": {
                str(
                    (root / "cases" / case_key(r["case"]) / "result.json").relative_to(
                        root
                    )
                ): sha(root / "cases" / case_key(r["case"]) / "result.json")
                for r in rows
            },
        },
    )


def select(root):
    verify(root)
    if (root / "selection_complete.json").exists():
        verify_selection(root)
        return
    config = read(root / "config.json")
    rows = stage_results(root, "development", config)
    recipes = {}
    for task, geometry, family, train_n in itertools.product(
        config["tasks"],
        config["geometries"],
        config["families"],
        config["train_sizes"],
    ):
        key = recipe_key(task, geometry, family, train_n)
        subset = [
            r
            for r in rows
            if recipe_key(
                *(r["case"][k] for k in ("task", "geometry", "family", "train_n"))
            )
            == key
        ]
        scores = {}
        for choice in choices(config):
            selected = [r for r in subset if r["case"]["choice"] == choice["id"]]
            if len(selected) != len(config["development_teachers"]) * len(
                config["calibration_p"]
            ):
                raise ValueError("Unbalanced development outcomes")
            scores[choice["id"]] = float(
                np.mean(
                    np.log(
                        [
                            max(
                                config["log_risk_threshold"],
                                r["metrics"]["endpoint_mse"],
                            )
                            for r in selected
                        ]
                    )
                )
            )
        recipes[key] = {
            "choice": min(scores, key=lambda k: (scores[k], k)),
            "mean_log_validation_scores": scores,
        }
    write(root / "selected_recipes.json", recipes)
    write(
        root / "selection_complete.json",
        {
            "utc": now(),
            "selected_sha256": sha(root / "selected_recipes.json"),
            "development_barrier_sha256": sha(root / "development_complete.json"),
            "rule": "Four equally available choices per task/geometry/primitive/TRAIN size. Minimum mean log validation MSE across two development teachers and two calibration budgets; lexical exact ties. Timing canaries excluded.",
        },
    )


def forecasts_from_rows(config, recipes, rows):
    curves = {}
    for key, selected in recipes.items():
        task, geometry, family, n = key.split("/")
        subset = [
            r
            for r in rows
            if (
                r["case"]["task"],
                r["case"]["geometry"],
                r["case"]["family"],
                r["case"]["train_n"],
                r["case"]["choice"],
            )
            == (task, geometry, family, int(n), selected["choice"])
        ]
        ps, log_risks = [], []
        for p in sorted(config["calibration_p"] + [config["bridge_p"]]):
            values = [r for r in subset if r["case"]["ceiling"] == p]
            if len(values) != len(config["development_teachers"]):
                raise ValueError(
                    "Forecast requires every selected development teacher at each disclosed budget"
                )
            ps.append(values[0]["counted_inventory"]["stored_parameters"])
            log_risks.append(
                float(
                    np.mean(
                        [
                            np.log(
                                max(
                                    config["log_risk_threshold"],
                                    r["metrics"]["endpoint_mse"],
                                )
                            )
                            for r in values
                        ]
                    )
                )
            )
        pnew = width.parameter_inventory(
            width.branches_under_ceiling(config["withheld_p"], geometry), geometry
        )["stored_parameters"]
        power, exponential = (
            np.polyfit(np.log(ps), log_risks, 1),
            np.polyfit(ps, log_risks, 1),
        )
        curves[key] = {
            "actual_parameters": ps,
            "mean_log_validation_risks": log_risks,
            "withheld_actual_parameters": pnew,
            "constant_log_risk": log_risks[-1],
            "power_log_risk": float(np.polyval(power, np.log(pnew))),
            "exponential_log_risk": float(np.polyval(exponential, pnew)),
            "power_slope": float(power[0]),
            "exponential_slope": float(exponential[0]),
        }
    comparisons = []
    for task, family, train_n, (numerator, denominator) in itertools.product(
        config["tasks"],
        config["families"],
        config["train_sizes"],
        FORECAST_PAIRS,
    ):
        delta = (
            curves[recipe_key(task, numerator, family, train_n)][
                "mean_log_validation_risks"
            ][-1]
            - curves[recipe_key(task, denominator, family, train_n)][
                "mean_log_validation_risks"
            ][-1]
        )
        comparisons.append(
            {
                "task": task,
                "family": family,
                "train_n": train_n,
                "numerator": numerator,
                "denominator": denominator,
                "development_log_ratio": float(delta),
                "predicted_numerator_better": bool(delta < 0),
            }
        )
    return {"curves": curves, "architecture_comparisons": comparisons}


def freeze_forecasts(root):
    verify_selection(root)
    if (root / "forecast_complete.json").exists():
        verify_forecast(root)
        return
    config = read(root / "config.json")
    rows = stage_results(root, "development", config) + stage_results(
        root, "bridge", config
    )
    forecast = forecasts_from_rows(config, read(root / "selected_recipes.json"), rows)
    forecast.update(
        utc=now(),
        selected_sha256=sha(root / "selected_recipes.json"),
        scope="All numerical and architecture predictions frozen before fresh-data generation. Slopes are finite-range forecasts, not learned exponents.",
    )
    write(root / "frozen_forecasts.json", forecast)
    write(
        root / "forecast_complete.json",
        {
            "utc": now(),
            "forecast_sha256": sha(root / "frozen_forecasts.json"),
            "bridge_barrier_sha256": sha(root / "bridge_complete.json"),
        },
    )


def finish(root):
    verify_forecast(root)
    config = read(root / "config.json")
    rows = [
        r for stage in config["fit_counts"] for r in stage_results(root, stage, config)
    ]
    if len(rows) != config["expected_fits"]:
        raise ValueError("Final fit count differs")
    if not (root / "complete.json").exists():
        write(
            root / "complete.json",
            {
                "utc": now(),
                "status": "complete",
                "fits": len(rows),
                "expected_fits": config["expected_fits"],
                "failed_fits": 0,
                "fit_counts": config["fit_counts"],
                "summed_fit_seconds": sum(r["fit_elapsed_seconds"] for r in rows),
                "scope": "Generic fitting complete. Constructive references, independent numerical replay and scientific interpretation are separate.",
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "initialize",
            "development",
            "select",
            "bridge",
            "forecast",
            "confirmation",
            "finish",
        ],
        default="all",
    )
    parser.add_argument("--teachers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage in ("all", "initialize"):
        initialize(root, configuration(args.teachers, args.steps))
    if args.stage in ("all", "development"):
        execute_stage(root, "development", args.workers)
    if args.stage in ("all", "select"):
        select(root)
    if args.stage in ("all", "bridge"):
        execute_stage(root, "bridge", args.workers)
    if args.stage in ("all", "forecast"):
        freeze_forecasts(root)
    if args.stage in ("all", "confirmation"):
        execute_stage(root, "confirmation", args.workers)
    if args.stage in ("all", "finish"):
        finish(root)


if __name__ == "__main__":
    main()

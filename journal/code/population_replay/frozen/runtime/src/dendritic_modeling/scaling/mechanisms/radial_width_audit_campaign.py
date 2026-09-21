"""Post-completion independent audit of every radial rank/width case and checkpoint.

No production teacher, model, optimizer or campaign imports. All failures remain
in a new output tree. Numerical tolerances are fixed arguments, never widened.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Must precede NumPy initialization in native and spawned workers.
for _thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_thread_variable] = "1"


def _load_helper():
    path = Path(__file__).with_name("radial_width_audit.py")
    if (
        __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        != "6b00bccf34a40b21f5bd998767a96141e91f8de47dd8a5806253fff4dc400fc8"
    ):
        raise AssertionError("Frozen independent radial helper changed")
    spec = importlib.util.spec_from_file_location(
        "radial_width_frozen_audit_helper", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load_helper()
CASE_KEYS = (
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
STAGES = ("development", "bridge", "confirmation")


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _case_id(case):
    if set(case) != set(CASE_KEYS):
        raise AssertionError("Unexpected/missing declared case coordinates")
    return tuple(case[key] for key in CASE_KEYS)


def expected_cases(config, choices, selected, stage):
    """Independently enumerate the frozen grid, including nonproduction smokes."""
    if stage not in STAGES:
        raise ValueError("Unknown declared stage")
    teachers = config[
        "confirmation_teachers" if stage == "confirmation" else "development_teachers"
    ]
    cases = set()
    for task, geometry, family, teacher, observation, train_n in itertools.product(
        config["tasks"],
        config["geometries"],
        config["families"],
        teachers,
        range(2 if stage == "confirmation" else 1),
        config["train_sizes"],
    ):
        key = f"{task}/{geometry}/{family}/{train_n}"
        recipes = (
            [row["id"] for row in choices]
            if stage == "development"
            else [selected[key]["choice"]]
        )
        budgets = (
            config["calibration_p"]
            if stage == "development"
            else ([config["bridge_p"]] if stage == "bridge" else config["primary_p"])
        )
        cases.update(
            (
                stage,
                task,
                geometry,
                family,
                teacher,
                observation,
                train_n,
                choice,
                "plane8",
                True,
                budget,
            )
            for choice, budget in itertools.product(recipes, budgets)
        )
    return cases


def verify_complete(root):
    """Read only seals and case receipts until every completed stage is bound."""
    root = Path(root).resolve()
    complete = AUDIT.read_json(root / "complete.json")
    initialized = AUDIT.read_json(root / "initialized.json")
    bindings = []
    if complete.get("status") != "complete" or complete.get("failed_fits", 0) != 0:
        raise AssertionError("Campaign must be complete before audit or references")
    for relative, digest in initialized["bindings"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise AssertionError("Initialization source binding escapes root")
        bindings.append(AUDIT._bound(path, digest))
    config = AUDIT.read_json(root / "config.json")
    choices = AUDIT.read_json(root / "choices.json")
    selected = AUDIT.read_json(root / "selected_recipes.json")
    selection = AUDIT.read_json(root / "selection_complete.json")
    forecast = AUDIT.read_json(root / "forecast_complete.json")
    seals = {
        "selected_recipes.json": selection["selected_sha256"],
        "development_complete.json": selection["development_barrier_sha256"],
        "frozen_forecasts.json": forecast["forecast_sha256"],
        "bridge_complete.json": forecast["bridge_barrier_sha256"],
    }
    for relative, digest in seals.items():
        bindings.append(AUDIT._bound(root / relative, digest))
    if (
        AUDIT.read_json(root / "frozen_forecasts.json")["selected_sha256"]
        != selection["selected_sha256"]
    ):
        raise AssertionError("Forecast and selected recipe bindings disagree")
    if (
        set(config["fit_counts"]) != set(STAGES)
        or complete["fit_counts"] != config["fit_counts"]
    ):
        raise AssertionError("Completion stage inventory differs from frozen config")
    if complete["fits"] != config["expected_fits"] or complete["fits"] != sum(
        config["fit_counts"].values()
    ):
        raise AssertionError("Completion total differs from declared grid")
    if set(config["development_teachers"]) & set(config["confirmation_teachers"]):
        raise AssertionError("Development and confirmation teachers overlap")
    if config["withheld_p"] in config["calibration_p"] + [config["bridge_p"]]:
        raise AssertionError("Withheld budget entered development selection")
    if len({choice["id"] for choice in choices}) != 4:
        raise AssertionError("Exactly four equally available frozen recipes required")
    recipe_lookup = {choice["id"]: choice for choice in choices}
    rows, seen_paths, datasets = [], set(), set()
    for stage in STAGES:
        marker_path = root / f"{stage}_complete.json"
        marker = AUDIT.read_json(marker_path)
        expected = expected_cases(config, choices, selected, stage)
        if (
            marker.get("status") != "complete"
            or marker["fits"] != config["fit_counts"][stage]
            or len(expected) != marker["fits"]
            or len(marker["results"]) != marker["fits"]
        ):
            raise AssertionError(f"Incomplete or miscounted stage: {stage}")
        actual = set()
        for relative, digest in marker["results"].items():
            path = (root / relative).resolve()
            if not path.is_relative_to(root) or str(path) in seen_paths:
                raise AssertionError("Result path escapes root or appears twice")
            seen_paths.add(str(path))
            AUDIT._bound(path, digest)
            row = AUDIT.read_json(path)
            identity = _case_id(row["case"])
            if (
                row.get("status") != "complete"
                or identity in actual
                or identity not in expected
            ):
                raise AssertionError(
                    f"Missing/duplicate/out-of-grid completed case: {relative}"
                )
            if row["initialized_sha256"] != AUDIT.sha256(root / "initialized.json"):
                raise AssertionError("Case source/config initialization seal differs")
            release = "confirmation" if stage == "confirmation" else "development"
            case = row["case"]
            expected_data = (
                root
                / "data"
                / release
                / f"{case['task']}_t{case['teacher']}_o{case['observation']}.npz"
            )
            if Path(row["data_path"]).resolve() != expected_data.resolve():
                raise AssertionError("Case points to an undeclared dataset")
            actual.add(identity)
            datasets.add(str(expected_data.resolve()))
            for stem in ("data", "fit", "state", "predictions"):
                artifact = Path(row[stem + "_path"]).resolve()
                if not artifact.is_relative_to(root):
                    raise AssertionError("Completed case artifact escapes root")
                AUDIT._bound(artifact, row[stem + "_sha256"])
            if row["train_rows"] != case["train_n"]:
                raise AssertionError(
                    "Result TRAIN row count differs from declared prefix"
                )
            expected_endpoint = config[
                "test_n" if release == "confirmation" else "validation_n"
            ]
            if row["endpoint_rows"] != expected_endpoint:
                raise AssertionError(
                    "Result endpoint row count differs from declared split"
                )
            rows.append(
                {
                    "path": str(path),
                    "sha256": digest,
                    "case": case,
                    "fit_config": recipe_lookup[case["choice"]]["fit"],
                }
            )
        if actual != expected:
            raise AssertionError(f"Case grid differs from frozen factorial: {stage}")
        bindings.append(AUDIT._bound(marker_path))
    expected_datasets = (
        len(config["development_teachers"]) + 2 * len(config["confirmation_teachers"])
    ) * len(config["tasks"])
    if len(rows) != config["expected_fits"] or len(datasets) != expected_datasets:
        raise AssertionError("Unique cases or datasets differ from frozen count")
    for name in (
        "complete.json",
        "initialized.json",
        "selection_complete.json",
        "forecast_complete.json",
    ):
        bindings.append(AUDIT._bound(root / name))
    return {
        "config": config,
        "cases": sorted(rows, key=lambda row: row["path"]),
        "datasets": sorted(datasets),
        "bindings": bindings,
        "completion_sha256": AUDIT.sha256(root / "complete.json"),
        "release_checks": verify_data_releases(
            root, config, sorted(datasets), forecast["forecast_sha256"]
        ),
    }


def verify_data_releases(root, config, dataset_paths, forecast_sha256):
    """Bind public dataset receipts and pairing without opening private files."""
    seen = {}
    receipts = []
    for text in dataset_paths:
        path = Path(text)
        sidecar = AUDIT.read_json(path.with_suffix(".json"))
        AUDIT._bound(path, sidecar["sha256"])
        expected_seed = (
            config["observation_seed_base"]
            + 100 * sidecar["teacher"]
            + 2 * sidecar["observation"]
        )
        if (
            sidecar["seed"] != expected_seed
            or sidecar["max_train_n"] != config["max_train_n"]
            or sidecar["nested_train_sizes"] != config["train_sizes"]
        ):
            raise AssertionError("Dataset prefix/seed declaration differs from config")
        expected_endpoint = (
            "test" if sidecar["release"] == "confirmation" else "validation"
        )
        if sidecar["endpoint"] != expected_endpoint:
            raise AssertionError("Dataset endpoint role differs")
        if sidecar["forecast_sha256"] != (
            forecast_sha256 if expected_endpoint == "test" else None
        ):
            raise AssertionError("Confirmation dataset lacks the frozen forecast seal")
        data = AUDIT._dataset(path)
        if (
            len(data["x_train"]) != config["max_train_n"]
            or len(data["x_endpoint"]) != config[expected_endpoint + "_n"]
        ):
            raise AssertionError(
                "Dataset array row counts differ from frozen declaration"
            )
        hashes = {
            split: AUDIT.train_fingerprint(data[f"x_{split}"])
            for split in ("train", "endpoint")
        }
        key = (sidecar["release"], sidecar["teacher"], sidecar["observation"])
        if key in seen and hashes != seen[key]:
            raise AssertionError("Raw inputs differ across paired response powers")
        seen[key] = hashes
        if hashes["train"] == hashes["endpoint"]:
            raise AssertionError("Identical TRAIN and endpoint arrays")
        receipts.append(
            {
                "path": str(path),
                "data_sha256": sidecar["sha256"],
                "sidecar": AUDIT._bound(path.with_suffix(".json")),
                "input_hashes": hashes,
            }
        )
    return receipts


def _failure(error):
    return {
        "status": "failed",
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }


def checkpoint_audit(result_path, *, rtol=3e-6, atol=5e-15, expected_fit_config=None):
    """Check every unique checkpoint and every available stage TRAIN objective."""
    result = AUDIT.read_json(result_path)
    fit_path = Path(result["fit_path"])
    fit = AUDIT.read_json(fit_path)
    AUDIT._bound(fit_path, result["fit_sha256"])
    if expected_fit_config is not None and fit["config"] != expected_fit_config:
        raise AssertionError(
            "Fitted optimizer settings differ from the frozen selected recipe"
        )
    if result["ridge"] != fit["config"]["ridge"]:
        raise AssertionError("Case and fit ridge settings differ")
    stage_fits = fit["stage_fits"]
    boundaries = fit["boundary_records"]
    if len(stage_fits) != len(boundaries) or len(stage_fits) != len(
        fit["restart_schedule"]
    ):
        raise AssertionError("Stage/boundary/restart counts disagree")
    if [row["config"]["steps"] for row in stage_fits] != fit["restart_schedule"] or sum(
        fit["restart_schedule"]
    ) != fit["config"]["steps"]:
        raise AssertionError("Restart schedules do not preserve total iteration budget")
    if (
        fit["iterations"] != sum(row["iterations"] for row in stage_fits)
        or fit["iterations"] > fit["config"]["steps"]
    ):
        raise AssertionError("Stage iteration counts differ from total allowance")
    if fit["closure_calls"] != sum(row["closure_calls"] for row in stage_fits) or fit[
        "closure_calls"
    ] != len(fit["closure_history"]):
        raise AssertionError("Stage objective evaluation counts disagree")
    records = {}

    def add(record, role, expectations=None):
        path = str(Path(record["path"]).resolve())
        if path not in records:
            records[path] = {"record": record, "roles": [], "expectations": []}
        elif records[path]["record"] != record:
            raise AssertionError("Aliased checkpoint receipts disagree")
        records[path]["roles"].append(role)
        if expectations:
            records[path]["expectations"].append({"role": role, **expectations})

    add(fit["initial_state"], "top_initial")
    add(fit["warm_start_state"], "top_warm", fit["stages"][0])
    add(fit["final_state"], "top_final", fit["stages"][-1])
    for index, (stage, boundary) in enumerate(zip(stage_fits, boundaries, strict=True)):
        if (
            stage["status"] != "complete"
            or boundary["stage"] != index
            or stage["config"]["ridge"] != fit["config"]["ridge"]
        ):
            raise AssertionError("Checkpoint stage identity or ridge differs")
        for kind in ("initial", "warm_start", "final"):
            expectations = (
                None
                if kind == "initial"
                else stage["stages"][0 if kind == "warm_start" else -1]
            )
            add(stage[f"{kind}_state"], f"stage{index}_{kind}", expectations)
        add(
            boundary["before_state"],
            f"boundary{index}_before",
            {"objective": boundary["before_objective"]},
        )
        add(
            boundary["after_state"],
            f"boundary{index}_after",
            {"objective": boundary["after_objective"]},
        )
    expected_paths = {str(path.resolve()) for path in fit_path.parent.rglob("*.npz")}
    if set(records) != expected_paths or len(records) != 4 * len(stage_fits) + 2:
        raise AssertionError("Referenced checkpoint set differs from every saved NPZ")
    AUDIT._bound(result["data_path"], result["data_sha256"])
    data = AUDIT.train_prefix(
        AUDIT._dataset(result["data_path"]), result["case"]["train_n"]
    )
    checks = []
    for path, entry in sorted(records.items()):
        checked = {
            "path": path,
            "roles": entry["roles"],
            "recorded_sha256": entry["record"]["sha256"],
        }
        try:
            AUDIT._bound(path, entry["record"]["sha256"])
            loaded = AUDIT.load_width_state(path)
            if (
                loaded["inventory"]["stored_parameters"]
                != entry["record"]["stored_parameters"]
                or loaded["inventory"]["stored_parameters"]
                != result["counted_inventory"]["stored_parameters"]
            ):
                raise AssertionError("Checkpoint slot count differs from case")
            if (
                loaded["width_metadata"] != result["counted_inventory"]
                or loaded["state"]["metadata"]["family"] != result["case"]["family"]
            ):
                raise AssertionError("Checkpoint geometry/map/source differs from case")
            checked.update(
                status="passed", inventory=loaded["inventory"], numeric_checks=[]
            )
            if entry["expectations"]:
                numeric = AUDIT.REPLAY.train_metrics(
                    loaded["state"],
                    AUDIT.expanded_inputs(data["x_train"], loaded),
                    data["y_train"],
                    float(result["ridge"]),
                )
                checked["train_metrics"] = {
                    key: numeric[key] for key in ("mse", "penalty", "objective")
                }
                for expected in entry["expectations"]:
                    for metric in ("mse", "penalty", "objective"):
                        if metric not in expected:
                            continue
                        numeric_check = {
                            "role": expected["role"],
                            "metric": metric,
                            "actual": numeric[metric],
                            "recorded": expected[metric],
                        }
                        try:
                            numeric_check["absolute_difference"] = AUDIT._close(
                                numeric[metric],
                                expected[metric],
                                expected["role"] + ":" + metric,
                                rtol,
                                atol,
                            )
                            numeric_check["status"] = "passed"
                        except Exception as error:
                            numeric_check.update(_failure(error))
                            checked["status"] = "failed"
                        checked["numeric_checks"].append(numeric_check)
        except Exception as error:
            checked.update(_failure(error))
        checks.append(checked)
    return {
        "status": (
            "passed" if all(row["status"] == "passed" for row in checks) else "failed"
        ),
        "unique_checkpoint_files": len(checks),
        "checkpoint_roles": sum(len(row["roles"]) for row in checks),
        "stages": len(stage_fits),
        "checks": checks,
        "scope": "Every saved NPZ hash/map/source/count checked. Available warm, terminal and boundary TRAIN objectives replayed. Initial checkpoints without recorded metrics receive inventory checks only. No checkpoint is refitted.",
    }


def _case_worker(payload):
    index, case, tolerances, output = payload
    started = time.monotonic()
    result = {"index": index, "result_path": case["path"], "case": case["case"]}
    try:
        result["endpoint_audit"] = AUDIT.audit_result(
            case["path"], expected_hashes={case["path"]: case["sha256"]}, **tolerances
        )
    except Exception as error:
        result["endpoint_audit"] = _failure(error)
    try:
        result["checkpoint_audit"] = checkpoint_audit(
            case["path"],
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
            expected_fit_config=case["fit_config"],
        )
    except Exception as error:
        result["checkpoint_audit"] = _failure(error)
    result["status"] = (
        "passed"
        if all(
            result[key]["status"] == "passed"
            for key in ("endpoint_audit", "checkpoint_audit")
        )
        else "failed"
    )
    result["elapsed_seconds"] = time.monotonic() - started
    path = Path(output) / "cases" / f"{index:05d}.json"
    _write(path, result)
    return {
        "index": index,
        "status": result["status"],
        "path": str(path),
        "sha256": AUDIT.sha256(path),
        "result_path": case["path"],
        "checkpoint_files": result["checkpoint_audit"].get(
            "unique_checkpoint_files", 0
        ),
        "elapsed_seconds": result["elapsed_seconds"],
    }


def _dataset_worker(payload):
    (
        index,
        data_path,
        root,
        output,
        config,
        completion_sha256,
        label_tolerances,
        reference_sizes,
    ) = payload
    result = {"index": index, "data_path": data_path, "polynomial_references": []}
    metadata = AUDIT.read_json(Path(data_path).with_suffix(".json"))
    private_path = (
        Path(root)
        / "private_teachers"
        / f"{metadata['release']}_{metadata['task']}_t{metadata['teacher']}.json"
    )
    try:
        result["label_audit"] = AUDIT.audit_radial_labels(
            data_path,
            private_path,
            data_sha256=metadata["sha256"],
            specification_sha256=metadata["private_specification_sha256"],
            completion_path=Path(root) / "complete.json",
            completion_sha256=completion_sha256,
            expected_fit_count=config["expected_fits"],
            **label_tolerances,
        )
        specification = AUDIT.read_json(private_path)
        if (
            specification["power"] != int(metadata["task"].removeprefix("radial_m"))
            or specification["seed"]
            != config["teacher_seed_base"] + metadata["teacher"]
            or specification["radius"] != config["radius"]
            or specification["blocks"] != config["blocks"]
        ):
            raise AssertionError(
                "Private teacher identity differs from the frozen public task"
            )
    except Exception as error:
        result["label_audit"] = _failure(error)
    for train_n in reference_sizes:
        record = {
            "train_n": train_n,
            "comparison_scope": (
                "generic-prefix baseline"
                if train_n in config["train_sizes"]
                else "additional known-profile-prefix baseline"
            ),
        }
        try:
            record["reference"] = AUDIT.save_homogeneous_reference(
                data_path,
                Path(output) / "references" / f"{index:03d}" / f"n{train_n}",
                int(metadata["task"].removeprefix("radial_m")),
                train_n=train_n,
                rcond=1e-12,
            )
            record["status"] = "passed"
        except Exception as error:
            record.update(_failure(error))
        result["polynomial_references"].append(record)
    result["status"] = (
        "passed"
        if result["label_audit"]["status"] == "passed"
        and all(row["status"] == "passed" for row in result["polynomial_references"])
        else "failed"
    )
    path = Path(output) / "datasets" / f"{index:03d}.json"
    _write(path, result)
    return {
        "index": index,
        "data_path": data_path,
        "status": result["status"],
        "path": str(path),
        "sha256": AUDIT.sha256(path),
        "polynomial_references": len(reference_sizes),
    }


def run_campaign_audit(
    root,
    output,
    *,
    workers=1,
    rtol=3e-6,
    atol=5e-15,
    forward_rtol=3e-6,
    forward_atol=5e-12,
    label_rtol=1e-11,
    label_atol=1e-12,
    include_constructive_prefixes=True,
):
    if isinstance(workers, bool) or int(workers) != workers or not 1 <= workers <= 64:
        raise ValueError("One to 64 explicitly allocated audit workers required")
    tolerances = {
        "rtol": rtol,
        "atol": atol,
        "forward_rtol": forward_rtol,
        "forward_atol": forward_atol,
    }
    label_tolerances = {"rtol": label_rtol, "atol": label_atol}
    if not all(
        math.isfinite(value) and value >= 0
        for value in [*tolerances.values(), *label_tolerances.values()]
    ):
        raise ValueError("Finite nonnegative prespecified tolerances required")
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    try:
        verified = verify_complete(root)
    except Exception as error:
        _write(
            output / "barrier_failure.json",
            {
                **_failure(error),
                "root": str(root),
                "source_sha256": AUDIT.sha256(__file__),
            },
        )
        raise
    reference_sizes = sorted(
        set(verified["config"]["train_sizes"])
        | (
            set(verified["config"]["constructive_n"])
            if include_constructive_prefixes
            else set()
        )
    )
    if any(
        isinstance(n, bool)
        or not isinstance(n, int)
        or not 1 <= n <= verified["config"]["max_train_n"]
        for n in reference_sizes
    ):
        raise AssertionError("Invalid reference TRAIN prefix")
    for directory in ("cases", "datasets", "references"):
        (output / directory).mkdir()
    _write(
        output / "protocol_receipt.json",
        {
            "root": str(root),
            "completion_gate": verified["bindings"],
            "workers": workers,
            "source_sha256": AUDIT.sha256(__file__),
            "helper_sha256": AUDIT.sha256(AUDIT.__file__),
            "replay_sha256": AUDIT.sha256(AUDIT.REPLAY.__file__),
            "expected_cases": len(verified["cases"]),
            "expected_datasets": len(verified["datasets"]),
            "tolerances": tolerances,
            "label_tolerances": label_tolerances,
            "reference_rcond": 1e-12,
            "reference_train_sizes": reference_sizes,
            "public_data_release_checks": verified["release_checks"],
            "scope": "All declared completed cells and checkpoints retained. No tolerance widening, case exclusion or refitting. Fixed homogeneous polynomial references (61/181 coefficients) and private labels are post-completion diagnostics only. All declared TRAIN prefixes remain distinct; no reference is chosen by endpoint risk.",
        },
    )
    case_payloads = [
        (index, row, tolerances, str(output))
        for index, row in enumerate(verified["cases"])
    ]
    data_payloads = [
        (
            index,
            path,
            str(root),
            str(output),
            verified["config"],
            verified["completion_sha256"],
            label_tolerances,
            reference_sizes,
        )
        for index, path in enumerate(verified["datasets"])
    ]
    case_results, data_results = [], []
    if workers == 1:
        for payload in case_payloads:
            try:
                result = _case_worker(payload)
            except Exception as error:
                result = {"index": payload[0], **_failure(error)}
                path = output / f"worker_failure_case_{payload[0]:05d}.json"
                _write(path, result)
                result.update(path=str(path), sha256=AUDIT.sha256(path))
            case_results.append(result)
        for payload in data_payloads:
            try:
                result = _dataset_worker(payload)
            except Exception as error:
                result = {"index": payload[0], **_failure(error)}
                path = output / f"worker_failure_dataset_{payload[0]:05d}.json"
                _write(path, result)
                result.update(path=str(path), sha256=AUDIT.sha256(path))
            data_results.append(result)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            tasks = {
                executor.submit(_case_worker, payload): ("case", payload[0])
                for payload in case_payloads
            }
            tasks.update(
                {
                    executor.submit(_dataset_worker, payload): ("dataset", payload[0])
                    for payload in data_payloads
                }
            )
            for future in as_completed(tasks):
                kind, index = tasks[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {"index": index, **_failure(error)}
                    path = output / f"worker_failure_{kind}_{index:05d}.json"
                    _write(path, result)
                    result.update(path=str(path), sha256=AUDIT.sha256(path))
                (case_results if kind == "case" else data_results).append(result)
                if (len(case_results) + len(data_results)) % 100 == 0:
                    print(
                        json.dumps(
                            {
                                "completed_cases": len(case_results),
                                "completed_datasets": len(data_results),
                            }
                        ),
                        flush=True,
                    )
    summary = {
        "status": (
            "passed"
            if all(row["status"] == "passed" for row in case_results + data_results)
            else "failed"
        ),
        "root": str(root),
        "cases": len(case_results),
        "passed_cases": sum(row["status"] == "passed" for row in case_results),
        "datasets": len(data_results),
        "passed_datasets": sum(row["status"] == "passed" for row in data_results),
        "polynomial_references": sum(
            row.get("polynomial_references", 0) for row in data_results
        ),
        "checkpoint_files_checked": sum(
            row.get("checkpoint_files", 0) for row in case_results
        ),
        "case_receipts": sorted(case_results, key=lambda row: row["index"]),
        "dataset_receipts": sorted(data_results, key=lambda row: row["index"]),
        "elapsed_seconds": time.monotonic() - start,
        "workers": workers,
        "source_sha256": AUDIT.sha256(__file__),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Strict original result: failed receipts remain unchanged. Any later numerical reconciliation must be a separately reviewed artifact, not a tolerance revision here.",
    }
    if len(case_results) != len(case_payloads) or len(data_results) != len(
        data_payloads
    ):
        summary["status"] = "failed"
        summary["incomplete_worker_results"] = True
    _write(output / "audit.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--generic-prefixes-only", action="store_true")
    args = parser.parse_args()
    result = run_campaign_audit(
        args.root,
        args.output,
        workers=args.workers,
        include_constructive_prefixes=not args.generic_prefixes_only,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "cases",
                    "passed_cases",
                    "datasets",
                    "checkpoint_files_checked",
                )
            }
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

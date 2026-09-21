"""Post-completion independent audit of every rank/width case and checkpoint.

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

import numpy as np  # noqa: E402


def _load_helper():
    path = Path(__file__).with_name("rank_width_audit.py")
    spec = importlib.util.spec_from_file_location(
        "rank_width_frozen_audit_helper", path
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
    "choice",
    "coverage",
    "rebalance",
    "ceiling",
)
STAGES = ("diagnostic", "development", "bridge", "confirmation")


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _case_id(case):
    if set(case) != set(CASE_KEYS):
        raise AssertionError("Unexpected/missing declared case coordinates")
    return tuple(case[key] for key in CASE_KEYS)


def expected_cases(config, choices, selected, stage):
    """Independently enumerate declared factorial coordinates, without driver code."""
    teachers = (
        config["confirmation_teachers"]
        if stage == "confirmation"
        else config["development_teachers"]
    )
    cases = set()
    for task, geometry, family, teacher, observation in itertools.product(
        config["tasks"],
        config["geometries"],
        config["families"],
        teachers,
        range(2 if stage == "confirmation" else 1),
    ):
        prefix = (stage, task, geometry, family, teacher, observation)
        if stage == "diagnostic":
            if task not in ("mixture_q2", "quadratic_q2") or family not in (
                "shunt",
                "relu",
            ):
                continue
            if geometry not in ("rank2", "full", "width2"):
                continue
            starts = [("ols", "axial"), ("stein", "axial")]
            if geometry == "rank2":
                starts.append(("ols", "legacy"))
            for method, coverage in starts:
                for rebalance in (False, True) if geometry == "rank2" else (True,):
                    cases.add(
                        (
                            *prefix,
                            f"{method}_ridge1e-05",
                            coverage,
                            rebalance,
                            config["bridge_p"],
                        )
                    )
        else:
            recipes = (
                [row["id"] for row in choices]
                if stage == "development"
                else [selected[f"{task}/{geometry}/{family}"]["choice"]]
            )
            budgets = (
                config["calibration_p"]
                if stage == "development"
                else (
                    [config["bridge_p"]] if stage == "bridge" else config["primary_p"]
                )
            )
            cases.update(
                (*prefix, choice, "axial", True, budget)
                for choice, budget in itertools.product(recipes, budgets)
            )
    return cases


def verify_complete(root):
    """Read only seals and case receipts until every completed stage is bound."""
    root = Path(root).resolve()
    complete = AUDIT.read_json(root / "complete.json")
    initialized = AUDIT.read_json(root / "initialized.json")
    bindings = []
    if complete.get("status") != "complete":
        raise AssertionError("Campaign must be complete before audit or references")
    for relative, digest in initialized["bindings"].items():
        bindings.append(AUDIT._bound(root / relative, digest))
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
            rows.append({"path": str(path), "sha256": digest, "case": case})
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
    }


def _failure(error):
    return {
        "status": "failed",
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }


def checkpoint_audit(result_path, *, rtol=3e-6, atol=5e-15):
    """Check every unique checkpoint and every available stage TRAIN objective."""
    result = AUDIT.read_json(result_path)
    fit_path = Path(result["fit_path"])
    fit = AUDIT.read_json(fit_path)
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
    data = AUDIT._dataset(result["data_path"])
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
            case["path"], rtol=tolerances["rtol"], atol=tolerances["atol"]
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


def _mixture_response(t, interval, tilt, order):
    a, b = interval
    if not (0 < a < b and abs(tilt) < 1):
        raise AssertionError("Private mixture density is not positive/admitted")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    scale = (math.log(b) - math.log(a)) / 2
    conductance = np.exp((math.log(a) + math.log(b)) / 2 + scale * nodes)
    masses = (
        weights
        * scale
        * conductance
        * (1 + tilt * (2 * conductance - a - b) / (b - a))
        / (b - a)
    )
    prediction = np.empty_like(t)
    for start in range(0, len(t), 2048):
        values = t[start : start + 2048, None]
        prediction[start : start + 2048] = (values / (values + conductance)) @ masses
    return prediction


def audit_labels(root, data_path, *, rtol=2e-10, atol=2e-11):
    """Post-run private verification only; no production teacher code invoked."""
    root, data_path = Path(root), Path(data_path)
    metadata = AUDIT.read_json(data_path.with_suffix(".json"))
    AUDIT._bound(data_path, metadata["sha256"])
    task, qtext = metadata["task"].rsplit("_q", 1)
    q = int(qtext)
    if task not in ("quadratic", "mixture") or q not in (1, 2):
        raise AssertionError("Unsupported independently audited task")
    private = (
        root
        / "private_teachers"
        / f"{metadata['release']}_{task}_t{metadata['teacher']}.json"
    )
    binding = AUDIT._bound(private, metadata["private_specification_sha256"])
    specification = AUDIT.read_json(private)
    if (
        specification["index"] != metadata["teacher"]
        or specification["response"] != task
    ):
        raise AssertionError("Private specification identity differs")
    rotations = np.asarray(specification["rotations"], dtype=np.float64)
    if rotations.shape != (4, 3, 3):
        raise AssertionError("Invalid private rotation inventory")
    AUDIT._close(
        rotations.transpose(0, 2, 1) @ rotations,
        np.broadcast_to(np.eye(3), (4, 3, 3)),
        "private orthogonality",
        1e-12,
        1e-12,
    )
    data = AUDIT._dataset(data_path)
    checks = []
    if task == "quadratic":
        for dimension in (1, 2):
            expected_mean = dimension / 20
            expected_variance = (dimension - dimension**2 / 5) / 280
            for key, expected in (
                ("mean", expected_mean),
                ("variance", expected_variance),
            ):
                AUDIT._close(
                    specification["population_block_moments"][str(dimension)][key],
                    expected,
                    "quadratic " + key,
                    1e-12,
                    1e-15,
                )
        operator = rotations[:, :, :q] @ rotations[:, :, :q].transpose(0, 2, 1)
    for split in ("train", "endpoint"):
        x = data[f"x_{split}"]
        if x.shape[1:] != (4, 3) or np.max(np.linalg.norm(x, axis=2)) > 0.5 + 1e-12:
            raise AssertionError("Raw data lie outside the common ball law")
        if task == "quadratic":
            energy = np.einsum("nbi,bij,nbj->nb", x, operator, x)
            predicted = ((energy - q / 20) / math.sqrt((q - q**2 / 5) / 70)).sum(1)
            quadrature_difference = None
        else:
            intervals = np.asarray(specification["intervals"])
            tilts = np.asarray(specification["density_tilt"])
            means = np.asarray(specification["component_means"])
            variances = np.asarray(specification["component_variances_q1_q2"])[q - 1]
            if (
                intervals.shape != (4, 2)
                or tilts.shape != (4,)
                or means.shape != (4,)
                or variances.shape != (4,)
                or not np.all(variances > 0)
            ):
                raise AssertionError("Invalid source-bound mixture normalization")
            predictions = []
            for order in (96, 128):
                component = np.zeros((len(x), 4))
                for block, axis in itertools.product(range(4), range(q)):
                    t = 0.5 + x[:, block] @ rotations[block, :, axis]
                    component[:, block] += _mixture_response(
                        t, intervals[block], tilts[block], order
                    )
                predictions.append(
                    ((component - q * means) / np.sqrt(4 * variances)).sum(1)
                )
            quadrature_difference = AUDIT._close(
                predictions[0],
                predictions[1],
                "mixture quadrature agreement",
                rtol,
                atol,
            )
            predicted = predictions[-1]
        error = AUDIT._close(
            predicted, data[f"y_{split}"], split + " independent labels", rtol, atol
        )
        checks.append(
            {
                "split": split,
                "rows": len(x),
                "maximum_label_difference": error,
                "quadrature_order_difference": quadrature_difference,
            }
        )
    return {
        "status": "passed",
        "data_path": str(data_path),
        "private_specification": binding,
        "checks": checks,
        "tolerances": {"rtol": rtol, "atol": atol},
        "scope": "Teacher data used only after root completion for independent label verification. Quadratic moments and projector algebra independently derived. Mixtures use96/128-point positive log-conductance quadrature and source-bound stored moments; mixture population normalization is not newly audited.",
    }


def _dataset_worker(payload):
    index, data_path, root, output, label_tolerances = payload
    result = {"index": index, "data_path": data_path}
    try:
        result["label_audit"] = audit_labels(root, data_path, **label_tolerances)
    except Exception as error:
        result["label_audit"] = _failure(error)
    try:
        result["quadratic_reference"] = AUDIT.save_quadratic_reference(
            data_path, Path(output) / "references" / f"{index:03d}"
        )
    except Exception as error:
        result["quadratic_reference"] = _failure(error)
    result["status"] = (
        "passed"
        if result["label_audit"]["status"] == "passed"
        and result["quadratic_reference"]["status"] == "complete"
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
    label_rtol=2e-10,
    label_atol=2e-11,
):
    if isinstance(workers, bool) or int(workers) != workers or not 1 <= workers <= 8:
        raise ValueError("One to eight audit workers required")
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
            "scope": "All declared completed cells and checkpoints retained. No tolerance widening, case exclusion or refitting. Fixed quadratic reference and private labels are post-completion diagnostics only.",
        },
    )
    case_payloads = [
        (index, row, tolerances, str(output))
        for index, row in enumerate(verified["cases"])
    ]
    data_payloads = [
        (index, path, str(root), str(output), label_tolerances)
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
    args = parser.parse_args()
    result = run_campaign_audit(args.root, args.output, workers=args.workers)
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

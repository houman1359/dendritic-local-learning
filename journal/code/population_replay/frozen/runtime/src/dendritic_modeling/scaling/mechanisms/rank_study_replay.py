"""Independent NumPy replay of every stored rank-study endpoint.

No Torch or production model/task imports. The scalar shunt, factored matrices,
TRAIN feature normalization and effective-readout ridge are reconstructed from
saved arrays. This audits records; it never modifies or improves a predictor.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

RMS_EPSILON = 1e-8


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(actual, expected, label, *, rtol=3e-6, atol=5e-15):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{label}: independently replayed {actual!r}; recorded {expected!r}"
        )
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def load_state(path):
    with np.load(path, allow_pickle=False) as archive:
        state = {
            name: archive[name].copy()
            for name in archive.files
            if name != "metadata_json"
        }
        metadata = json.loads(str(archive["metadata_json"]))
    caps = np.asarray(metadata["capacities"], dtype=int)
    if len(caps) < 1 or np.any(caps < 1):
        raise AssertionError("Every supplied block must have positive capacity")
    k, m, d = len(caps), int(caps.sum()), metadata["inputs_per_block"]
    architecture = metadata["architecture"]
    if architecture not in ("full", "rank1", "rank2") or metadata["family"] not in (
        "shunt",
        "relu",
        "tanh",
    ):
        raise AssertionError("Unknown saved architecture or family")
    expected_shapes = {
        "threshold": (m,),
        "internal_readout": (m,),
        "soma_readout": (k,),
        "bias": (),
    }
    if architecture == "full":
        expected_shapes["projection"] = (m, d)
        expected_count = (d + 2) * m + k + 1
    else:
        rank = int(architecture[-1])
        expected_shapes.update(basis=(k, d, rank), branch_coefficients=(m, rank))
        expected_count = (rank + 2) * m + d * rank * k + k + 1
    if set(state) != set(expected_shapes) | {"group"}:
        raise AssertionError("Saved tensor inventory has missing or unexpected slots")
    for name, shape in expected_shapes.items():
        value = state[name]
        if (
            value.shape != shape
            or value.dtype != np.float64
            or not np.isfinite(value).all()
        ):
            raise AssertionError(f"Invalid stored shape/dtype/finiteness for {name}")
    count = sum(state[name].size for name in expected_shapes)
    if count != expected_count:
        raise AssertionError(
            "Stored parameter count disagrees with architecture formula"
        )
    np.testing.assert_array_equal(state["group"], np.repeat(np.arange(k), caps))
    np.testing.assert_array_equal(state["soma_readout"], np.ones(k))
    return {"arrays": state, "metadata": metadata, "parameter_count": count}


def effective_projection(state):
    arrays = state["arrays"]
    if state["metadata"]["architecture"] == "full":
        return arrays["projection"]
    return np.einsum(
        "mdr,mr->md", arrays["basis"][arrays["group"]], arrays["branch_coefficients"]
    )


def features(state, x):
    arrays = state["arrays"]
    pre = (
        np.sum(
            np.asarray(x)[:, arrays["group"], :] * effective_projection(state), axis=2
        )
        + arrays["threshold"]
    )
    family = state["metadata"]["family"]
    if family == "tanh":
        return np.tanh(pre)
    excitation = np.maximum(pre, 0.0)
    return excitation if family == "relu" else excitation / (1.0 + excitation)


def effective_readout(state):
    arrays = state["arrays"]
    return (
        arrays["internal_readout"]
        * arrays["soma_readout"][arrays["group"]]
        / math.sqrt(len(state["metadata"]["capacities"]))
    )


def predict(state, x):
    return features(state, x) @ effective_readout(state) + state["arrays"]["bias"]


def train_metrics(state, x, y, ridge):
    phi = features(state, x)
    mean = phi.mean(axis=0)
    centered = phi - mean
    scale = np.sqrt(np.mean(centered**2, axis=0) + RMS_EPSILON**2)
    raw_beta = effective_readout(state)
    beta = raw_beta * scale
    prediction = phi @ raw_beta + state["arrays"]["bias"]
    residual = prediction - y
    mse = float(np.mean(residual**2))
    penalty = float(ridge * np.sum(beta**2))
    stationarity = (centered / scale).T @ residual / len(y) + ridge * beta
    return {
        "mse": mse,
        "penalty": penalty,
        "objective": mse + penalty,
        "readout_stationarity_l2": float(np.linalg.norm(stationarity)),
        "residual_mean": float(residual.mean()),
        "standardized_coefficient_l2": float(np.linalg.norm(beta)),
        "raw_coefficient_l2": float(np.linalg.norm(raw_beta)),
        "minimum_training_rms": float(scale.min()),
        "training_feature_means": mean,
        "training_feature_scales": scale,
    }


def _serial_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }


def _bound_state(record):
    path = Path(record["path"])
    if sha(path) != record["sha256"]:
        raise AssertionError(f"State archive hash mismatch: {path}")
    state = load_state(path)
    if state["parameter_count"] != record["stored_parameters"]:
        raise AssertionError("State receipt count mismatch")
    return state


def audit_evaluation(evaluation_path, dataset_cache=None):
    """Replay initial/warm/terminal states and one reported endpoint."""
    dataset_cache = {} if dataset_cache is None else dataset_cache
    row = read(evaluation_path)
    fit = read(row["fit_file"])
    if fit["status"] != "complete":
        raise AssertionError("Evaluation points to an incomplete fit")
    dataset_path = str(row["dataset"])
    if dataset_path not in dataset_cache:
        path = Path(dataset_path)
        metadata = read(path.with_suffix(".json"))
        if sha(path) != metadata["sha256"]:
            raise AssertionError("Dataset archive hash mismatch")
        with np.load(path, allow_pickle=False) as archive:
            dataset = {name: archive[name].copy() for name in archive.files}
        dataset_cache[dataset_path] = (dataset, metadata)
    data, dataset_metadata = dataset_cache[dataset_path]
    if (row["teacher"], row["intrinsic_rank"], row["seed"]) != (
        dataset_metadata["teacher"],
        dataset_metadata["intrinsic_rank"],
        dataset_metadata["seed"],
    ):
        raise AssertionError("Dataset identity differs from evaluation identity")
    endpoint = "validation" if "validation" in row else "test"
    n, sigma = row["train_n"], row["sigma"]
    x = data["train_x"][:n]
    y = data["train_y_clean"][:n] + sigma * data["train_epsilon"][:n]
    if len(x) != n or y.shape != (n,):
        raise AssertionError("Reported training row count differs from actual prefix")
    if (
        "train_labels" in dataset_metadata
        and len(data["train_y_clean"]) != dataset_metadata["train_labels"]
    ):
        raise AssertionError("Stored training label count differs from dataset receipt")
    if (
        "endpoint_labels" in dataset_metadata
        and len(data[endpoint + "_y_clean"]) != dataset_metadata["endpoint_labels"]
    ):
        raise AssertionError("Stored endpoint label count differs from dataset receipt")
    initial = _bound_state(fit["initial_state"])
    warm = _bound_state(fit["warm_start_state"])
    final = _bound_state(fit["final_state"])
    if row["state"] != fit["final_state"]:
        raise AssertionError("Evaluation and fit name different endpoint states")
    for state in (initial, warm, final):
        metadata = state["metadata"]
        if (
            metadata["architecture"],
            metadata["family"],
            list(metadata["capacities"]),
        ) != (row["architecture"], row["family"], row["capacities"]):
            raise AssertionError("Model identity differs from reported identity")
        if (
            state["parameter_count"] != row["parameters"]
            or state["parameter_count"] != fit["stored_parameters"]
        ):
            raise AssertionError("Reported parameter count mismatch")
        if sum(metadata["capacities"]) != row["branches"]:
            raise AssertionError("Reported branch inventory mismatch")
        if "model_seed" in row:
            initialization = metadata["initialization_receipt"]
            if (
                initialization["seed"] != row["model_seed"]
                or initialization["initializer"] != row["initialization"]
            ):
                raise AssertionError(
                    "Initializer seed/provenance differs from saved model"
                )
    for name in initial["arrays"]:
        if name not in ("internal_readout", "bias"):
            np.testing.assert_array_equal(
                initial["arrays"][name],
                warm["arrays"][name],
                err_msg="Readout-only warm start changed body",
            )
    if fit["gradient_parameters"] != row["parameters"] - len(row["capacities"]):
        raise AssertionError("Fixed soma gauge missing from gradient-active inventory")
    ridge = fit["config"]["ridge"]
    initial_metrics, warm_metrics, final_metrics = [
        train_metrics(state, x, y, ridge) for state in (initial, warm, final)
    ]
    errors = []
    for metrics, recorded in [
        (warm_metrics, fit["stages"][0]),
        (final_metrics, fit["stages"][-1]),
    ]:
        for field in ("mse", "penalty", "objective"):
            errors.append(close(metrics[field], recorded[field], f"TRAIN {field}"))
        for field in ("training_feature_means", "training_feature_scales"):
            if field in recorded:
                close(metrics[field], recorded[field], field, rtol=1e-10, atol=1e-12)
    close(warm_metrics["objective"], fit["warm_start_objective"], "warm objective")
    close(final_metrics["objective"], fit["terminal_objective"], "terminal objective")
    if (
        warm_metrics["readout_stationarity_l2"] > 1e-7
        or abs(warm_metrics["residual_mean"]) > 1e-7
    ):
        raise AssertionError(
            "Warm-start coefficients are not an accurate standardized ridge optimum"
        )
    if fit["optimizer"] == "reduced" and (
        final_metrics["readout_stationarity_l2"] > 1e-7
        or abs(final_metrics["residual_mean"]) > 1e-7
    ):
        raise AssertionError(
            "Reduced endpoint coefficients are not an accurate ridge optimum"
        )
    predicted = predict(final, data[endpoint + "_x"])
    clean = data[endpoint + "_y_clean"]
    observed = clean + sigma * data[endpoint + "_epsilon"]
    endpoint_metrics = {
        "observed_mse": float(np.mean((predicted - observed) ** 2)),
        "clean_teacher_mse": float(np.mean((predicted - clean) ** 2)),
        "max_observed_error": float(np.max(abs(predicted - observed))),
    }
    endpoint_metrics["expected_observed_mse_from_clean"] = (
        endpoint_metrics["clean_teacher_mse"] + sigma**2
    )
    for field, value in endpoint_metrics.items():
        errors.append(close(value, row[endpoint][field], endpoint + " " + field))
    for name, value in fit["parameter_movement_l2"].items():
        movement = float(
            np.linalg.norm(final["arrays"][name] - initial["arrays"][name])
        )
        close(movement, value, "parameter movement " + name, rtol=1e-10, atol=1e-12)
    close(
        np.linalg.norm(effective_projection(final) - effective_projection(initial)),
        fit["effective_projection_movement_l2"],
        "effective projection movement",
        rtol=1e-10,
        atol=1e-12,
    )
    history = fit["closure_history"]
    if fit["closure_calls"] != len(history):
        raise AssertionError("Closure count differs from preserved history")
    for index, record in enumerate(history):
        if record["evaluation"] != index or not all(
            np.isfinite(record[key])
            for key in ("objective", "mse", "penalty", "unscaled_gradient_l2")
        ):
            raise AssertionError("Invalid closure history")
        close(
            record["objective"],
            record["mse"] + record["penalty"],
            "closure complete objective",
            rtol=1e-12,
        )
    return {
        "evaluation": str(evaluation_path),
        "stage": row["stage"],
        "variant": row["variant"],
        "teacher": row["teacher"],
        "intrinsic_rank": row["intrinsic_rank"],
        "seed": row["seed"],
        "family": row["family"],
        "architecture": row["architecture"],
        "parameters": row["parameters"],
        "branches": row["branches"],
        "optimizer": fit["optimizer"],
        "ridge": ridge,
        "train_n": n,
        "sigma": sigma,
        "endpoint": endpoint,
        "initial": _serial_metrics(initial_metrics),
        "warm_start": _serial_metrics(warm_metrics),
        "terminal": _serial_metrics(final_metrics),
        "endpoint_metrics": endpoint_metrics,
        "closure_calls": len(history),
        "iterations": fit["iterations"],
        "maximum_absolute_replay_discrepancy": max(errors),
        "status": "passed",
    }


def audit_bindings(root):
    initialized = read(root / "initialized.json")
    checked = []
    for filename, field in [
        ("config.json", "config_sha256"),
        ("choices.json", "choices_sha256"),
        ("protocol.md", "protocol_sha256"),
    ]:
        if field in initialized:
            if sha(root / filename) != initialized[field]:
                raise AssertionError("Initialization binding changed: " + filename)
            checked.append(str(root / filename))
    for source in initialized.get("sources", []):
        if sha(source["path"]) != source["sha256"]:
            raise AssertionError("Frozen source hash changed")
        checked.append(source["path"])
    for filename, receipt_name, field in [
        ("selected_recipes.json", "selection_complete.json", "selected_sha256"),
        ("frozen_forecasts.json", "forecast_complete.json", "forecast_sha256"),
    ]:
        if (root / receipt_name).exists():
            if sha(root / filename) != read(root / receipt_name)[field]:
                raise AssertionError("Frozen selection/forecast binding changed")
            checked.append(str(root / filename))
    if (root / "frozen_forecasts.json").exists() and sha(
        root / "selected_recipes.json"
    ) != read(root / "frozen_forecasts.json")["selected_sha256"]:
        raise AssertionError("Forecast points to changed selected recipes")
    return checked


def run_audit(campaign_root, output_dir, require_complete=True):
    root, output = Path(campaign_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    rows, failures, bindings, datasets = [], [], [], {}
    try:
        bindings = audit_bindings(root)
        if require_complete and not (root / "campaign_complete.json").exists():
            raise AssertionError("Campaign completion marker absent")
    except Exception as error:
        failures.append({"scope": "campaign bindings/completion", "error": repr(error)})
    for path in sorted((root / "fits").rglob("evaluation.json")):
        try:
            rows.append(audit_evaluation(path, datasets))
        except Exception as error:
            failures.append({"scope": str(path), "error": repr(error)})
    fit_files = list((root / "fits").rglob("fit.json"))
    terminal_states = list((root / "fits").rglob("final.npz"))
    evaluation_count = len(rows) + sum(
        failure["scope"].endswith("evaluation.json") for failure in failures
    )
    if len(fit_files) != evaluation_count or len(terminal_states) != evaluation_count:
        failures.append(
            {
                "scope": "inventory",
                "error": "Fit/final-state/evaluation counts differ",
                "fit_files": len(fit_files),
                "terminal_states": len(terminal_states),
                "evaluations": evaluation_count,
            }
        )
    if require_complete:
        expected = read(root / "config.json")["expected_fits"]
        if len(rows) != expected:
            failures.append(
                {
                    "scope": "inventory",
                    "error": f"Passed {len(rows)} endpoints; expected {expected}",
                }
            )
    preserved_failures = [
        str(path)
        for name in ("failure.json", "campaign_failure.json")
        for path in (root / "fits").rglob(name)
    ]
    if preserved_failures:
        failures.append(
            {"scope": "preserved training failures", "paths": preserved_failures}
        )
    result = {
        "status": "passed" if not failures else "failed",
        "utc": datetime.now(timezone.utc).isoformat(),
        "campaign_root": str(root),
        "auditor_source_sha256": sha(__file__),
        "endpoint_count": len(rows),
        "state_replays": 3 * len(rows),
        "dataset_count": len(datasets),
        "bindings": bindings,
        "failures": failures,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - start,
        "tolerances": {
            "metric_relative": 3e-6,
            "metric_absolute": 5e-15,
            "ridge_stationarity_l2": 1e-7,
        },
        "scope": "Independent NumPy replay of raw saved predictions, complete standardized ridge objectives, all stored parameter slots, source/data/state hashes and fit receipts. No Torch/production imports, retraining, selection or outcome changes. Diagnostic rank thresholds never enter this replay.",
    }
    with (output / "audit.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    columns = [
        "teacher",
        "intrinsic_rank",
        "seed",
        "stage",
        "variant",
        "family",
        "architecture",
        "parameters",
        "branches",
        "optimizer",
        "ridge",
        "train_n",
        "sigma",
        "endpoint",
        "train_mse",
        "penalty",
        "objective",
        "endpoint_mse",
        "clean_endpoint_mse",
        "iterations",
        "closure_calls",
        "maximum_absolute_replay_discrepancy",
    ]
    with (output / "replayed_endpoints.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {name: row[name] for name in columns if name in row}
            flat.update(
                train_mse=row["terminal"]["mse"],
                penalty=row["terminal"]["penalty"],
                objective=row["terminal"]["objective"],
                endpoint_mse=row["endpoint_metrics"]["observed_mse"],
                clean_endpoint_mse=row["endpoint_metrics"]["clean_teacher_mse"],
            )
            writer.writerow(flat)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    result = run_audit(args.root, args.output_dir, not args.allow_incomplete)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "endpoint_count",
                    "state_replays",
                    "dataset_count",
                    "elapsed_seconds",
                    "failures",
                )
            },
            indent=2,
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

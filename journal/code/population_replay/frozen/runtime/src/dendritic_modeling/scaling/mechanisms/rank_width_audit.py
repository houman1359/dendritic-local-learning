"""Independent NumPy replay and a TRAIN-only fixed quadratic reference.

No production model, fitter, task or teacher imports. A native source snapshot
needs the adjacent rank_study_replay.py, rank_learning.py and
rank_width_learning.py files; the latter two are hashed, never executed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def _array_sha(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(
        str(value.shape).encode() + str(value.dtype).encode() + value.tobytes()
    ).hexdigest()


def _load_replay():
    path = Path(__file__).with_name("rank_study_replay.py")
    spec = importlib.util.spec_from_file_location("rank_width_numpy_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = _load_replay()
GEOMETRIES = {
    "full": (0, 1),
    "rank2": (2, 1),
    "width1": (1, 1),
    "width2": (1, 2),
    "width4": (1, 4),
}


def _integer(value, label, minimum=1):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise AssertionError(f"Invalid integer {label}: {value!r}")
    return int(value)


def _close(actual, expected, label, rtol, atol):
    if not all(math.isfinite(value) and value >= 0 for value in (rtol, atol)):
        raise ValueError("Finite nonnegative numeric tolerances required")
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise AssertionError(f"Nonfinite or shape-mismatched comparison: {label}")
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{label}: maximum absolute difference {float(np.max(np.abs(actual - expected)))}; "
            f"rtol={rtol}, atol={atol}"
        )
    return float(np.max(np.abs(actual - expected)))


def _bound(path, expected=None):
    path = Path(path).resolve()
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise AssertionError(f"Hash mismatch: {path}")
    return {
        "path": str(path),
        "sha256": actual,
        "checked_against_prior_hash": expected is not None,
    }


def load_width_state(path, *, base_source_path=None, wrapper_source_path=None):
    """Validate native arrays plus the fixed duplication map without Torch."""
    state = REPLAY.load_state(path)
    native = state["metadata"]
    width = native.get("initialization_receipt", {}).get("_rank_width")
    if not isinstance(width, dict) or width.get("geometry") not in GEOMETRIES:
        raise AssertionError("Native state lacks a supported _rank_width map")
    rank, copies = GEOMETRIES[width["geometry"]]
    raw_blocks = _integer(width.get("raw_blocks"), "raw_blocks")
    dimension = _integer(width.get("inputs_per_block"), "inputs_per_block")
    nodes = raw_blocks * copies
    capacities = [_integer(value, "capacity") for value in native["capacities"]]
    if len(capacities) != nodes or native["inputs_per_block"] != dimension:
        raise AssertionError("Native dimensions disagree with expanded nodes")
    architecture = "full" if rank == 0 else f"rank{rank}"
    if native["architecture"] != architecture:
        raise AssertionError("Width geometry and native architecture disagree")
    branches = sum(capacities)
    branch_cost = dimension + 2 if rank == 0 else rank + 2
    overhead = nodes + 1 + dimension * rank * nodes
    count = branches * branch_cost + overhead
    node_map = width.get("node_to_raw")
    expected_map = np.repeat(np.arange(raw_blocks), copies).tolist()
    if not isinstance(node_map, list):
        raise AssertionError("Missing fixed integer node map")
    for value in node_map:
        _integer(value, "node_to_raw", minimum=0)
    if node_map != expected_map:
        raise AssertionError("Fixed map differs from block-major duplication")
    if state["arrays"]["group"].dtype.kind not in "iu":
        raise AssertionError("Fixed branch group must have integer dtype")
    inventory = {
        "geometry": width["geometry"],
        "base_architecture": architecture,
        "raw_blocks": raw_blocks,
        "inputs_per_block": dimension,
        "nodes_per_raw_block": copies,
        "nodes": nodes,
        "branches": branches,
        "branch_parameter_cost": branch_cost,
        "fixed_parameter_overhead": overhead,
        "stored_parameters": count,
        "fixed_soma_parameter_slots": nodes,
        "gradient_parameter_slots": count - nodes,
        "fixed_node_map_integer_entries": nodes,
        "fixed_branch_group_integer_entries": branches,
        "capacities": capacities,
        "node_to_raw": expected_map,
    }
    for key, expected in inventory.items():
        if width.get(key) != expected:
            raise AssertionError(f"Width metadata inventory mismatch: {key}")
    if state["parameter_count"] != count:
        raise AssertionError("Counted native arrays disagree with width formula")
    sources = []
    for name, override in (
        ("rank_learning.py", base_source_path),
        ("rank_width_learning.py", wrapper_source_path),
    ):
        source = (
            Path(override) if override is not None else Path(__file__).with_name(name)
        )
        key = (
            "base_source_sha256"
            if name == "rank_learning.py"
            else "wrapper_source_sha256"
        )
        if not isinstance(width.get(key), str):
            raise AssertionError(f"Missing frozen source binding: {key}")
        sources.append(_bound(source, width[key]))
    return {
        "state": state,
        "inventory": inventory,
        "width_metadata": width,
        "sources": sources,
    }


def expanded_inputs(raw_x, loaded):
    inventory = loaded["inventory"]
    raw_x = np.asarray(raw_x, dtype=np.float64)
    if raw_x.ndim != 3 or raw_x.shape[1:] != (
        inventory["raw_blocks"],
        inventory["inputs_per_block"],
    ):
        raise AssertionError("Raw inputs disagree with the counted block shape")
    if not np.isfinite(raw_x).all():
        raise AssertionError("Nonfinite raw inputs")
    return raw_x[:, np.asarray(inventory["node_to_raw"], dtype=int), :]


def _dataset(path, cache=None):
    key = str(Path(path).resolve())
    if cache is not None and key in cache:
        return cache[key]
    with np.load(path, allow_pickle=False) as archive:
        required = ("x_train", "y_train", "x_endpoint", "y_endpoint")
        if not set(required).issubset(archive.files):
            raise AssertionError("Missing scalar train/endpoint arrays")
        arrays = {name: archive[name].copy() for name in required}
    for split in ("train", "endpoint"):
        x, y = arrays[f"x_{split}"], arrays[f"y_{split}"]
        if (
            x.ndim != 3
            or len(x) == 0
            or y.shape != (len(x),)
            or x.dtype != np.float64
            or y.dtype != np.float64
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise AssertionError(f"Invalid float64 scalar dataset: {split}")
    if arrays["x_train"].shape[1:] != arrays["x_endpoint"].shape[1:]:
        raise AssertionError("Train and endpoint block shapes disagree")
    if cache is not None:
        cache[key] = arrays
    return arrays


def audit_result(
    result_path,
    *,
    rtol=3e-6,
    atol=5e-15,
    forward_rtol=3e-6,
    forward_atol=5e-12,
    expected_hashes=None,
    dataset_cache=None,
    enforce_maximal_ceiling=True,
):
    """Audit one immutable final endpoint; failed checks raise without mutation.

    Results may bind data_sha256, fit_sha256 and predictions_path/sha256. The
    receipt distinguishes checked prior hashes from newly computed hashes.
    Campaign prediction archives use train/endpoint; the longer
    prediction_train/prediction_endpoint spelling is also accepted.
    """
    result_path = Path(result_path).resolve()
    bindings = dict(expected_hashes or {})
    sources = [_bound(result_path, bindings.get(str(result_path)))]
    result = read_json(result_path)
    if result.get("status") != "complete":
        raise AssertionError("Result is not a complete endpoint")
    case = result["case"]
    fit_path = Path(result["fit_path"])
    if fit_path.is_dir():
        fit_path /= "fit.json"
    data_path = Path(result["data_path"])
    sources.extend(
        (
            _bound(
                fit_path,
                result.get("fit_sha256", bindings.get(str(fit_path.resolve()))),
            ),
            _bound(
                data_path,
                result.get("data_sha256", bindings.get(str(data_path.resolve()))),
            ),
        )
    )
    dataset_identity_check = "No public dataset sidecar supplied."
    sidecar = data_path.with_suffix(".json")
    if sidecar.exists():
        metadata = read_json(sidecar)
        sources.append(_bound(sidecar, bindings.get(str(sidecar.resolve()))))
        if metadata["sha256"] != sha256(data_path):
            raise AssertionError("Public dataset sidecar hash differs from archive")
        for key in ("task", "teacher", "observation"):
            if metadata[key] != case[key]:
                raise AssertionError(f"Case and public dataset identity differ: {key}")
        if "stage" in case:
            expected_release = (
                "confirmation" if case["stage"] == "confirmation" else "development"
            )
            if metadata["release"] != expected_release:
                raise AssertionError("Case and public dataset release differ")
        dataset_identity_check = "Public task/teacher/observation/archive binding checked; private teacher specifications were not opened."
    fit = read_json(fit_path)
    if fit.get("status") != "complete":
        raise AssertionError("Endpoint refers to an incomplete fit")
    state_record = fit["final_state"]
    state_path = Path(state_record["path"])
    sources.append(_bound(state_path, state_record["sha256"]))
    if (
        "state_path" in result
        and Path(result["state_path"]).resolve() != state_path.resolve()
    ):
        raise AssertionError("Result and fit point to different final states")
    if "state_sha256" in result and result["state_sha256"] != state_record["sha256"]:
        raise AssertionError("Result and fit bind different final-state hashes")
    loaded = load_width_state(state_path)
    sources.extend(loaded["sources"])
    state, inventory = loaded["state"], loaded["inventory"]
    if (
        case["geometry"] != inventory["geometry"]
        or case["family"] != state["metadata"]["family"]
    ):
        raise AssertionError("Case identity differs from saved geometry/family")
    for key in (
        "stored_parameters",
        "nodes",
        "branches",
        "raw_blocks",
        "inputs_per_block",
    ):
        if result["counted_inventory"].get(key) != inventory[key]:
            raise AssertionError(f"Result inventory mismatch: {key}")
    for key, value in result["counted_inventory"].items():
        if (
            key not in loaded["width_metadata"]
            or value != loaded["width_metadata"][key]
        ):
            raise AssertionError(f"Result inventory mismatch: {key}")
    for key in ("stored_parameters",):
        if state_record[key] != inventory[key] or fit[key] != inventory[key]:
            raise AssertionError(f"Fit/state inventory mismatch: {key}")
    ceiling = _integer(case["ceiling"], "ceiling")
    if inventory["stored_parameters"] > ceiling:
        raise AssertionError("Actual parameters exceed declared ceiling")
    if (
        enforce_maximal_ceiling
        and inventory["stored_parameters"] + inventory["branch_parameter_cost"]
        <= ceiling
    ):
        raise AssertionError(
            "Saved branch capacity is not maximal under the declared ceiling"
        )
    ridge = float(result["ridge"])
    if not math.isfinite(ridge) or ridge <= 0 or ridge != float(fit["config"]["ridge"]):
        raise AssertionError("Invalid or inconsistent positive ridge")
    if "rebalance" in case and bool(case["rebalance"]) != bool(fit.get("rebalance")):
        raise AssertionError("Case and fit disagree on periodic rebalancing")
    if "coverage" in case:
        saved_coverage = (
            state["metadata"]["initialization_receipt"]
            .get("direction_bank", {})
            .get("coverage")
        )
        if saved_coverage != case["coverage"]:
            raise AssertionError("Case and saved initialization disagree on coverage")
    data = _dataset(data_path, dataset_cache)
    estimator = (
        state["metadata"]["initialization_receipt"]
        .get("direction_bank", {})
        .get("estimator")
    )
    initializer_train_hash_check = False
    if estimator is not None:
        for source_key, array_key in (
            ("train_x_sha256", "x_train"),
            ("train_y_sha256", "y_train"),
        ):
            actual = hashlib.sha256(
                np.ascontiguousarray(data[array_key]).tobytes()
            ).hexdigest()
            if estimator[source_key] != actual:
                raise AssertionError(
                    f"Saved initializer used different TRAIN data: {source_key}"
                )
        if estimator["rows"] != len(data["x_train"]):
            raise AssertionError("Saved initializer TRAIN row count differs")
        initializer_train_hash_check = True
    train = expanded_inputs(data["x_train"], loaded)
    endpoint = expanded_inputs(data["x_endpoint"], loaded)
    metrics = REPLAY.train_metrics(state, train, data["y_train"], ridge)
    prediction_train = REPLAY.predict(state, train)
    prediction_endpoint = REPLAY.predict(state, endpoint)
    endpoint_mse = float(np.mean((prediction_endpoint - data["y_endpoint"]) ** 2))
    differences = {
        "train_mse": _close(
            metrics["mse"], result["metrics"]["train_mse"], "train_mse", rtol, atol
        ),
        "endpoint_mse": _close(
            endpoint_mse, result["metrics"]["endpoint_mse"], "endpoint_mse", rtol, atol
        ),
        "terminal_objective": _close(
            metrics["objective"],
            fit["terminal_objective"],
            "terminal_objective",
            rtol,
            atol,
        ),
    }
    terminal = fit["stages"][-1]
    if terminal["stage"] != "terminal":
        raise AssertionError("Final fit stage is not the terminal evaluation")
    for key in ("mse", "penalty", "objective"):
        differences[f"fit_terminal_{key}"] = _close(
            metrics[key], terminal[key], f"fit_terminal_{key}", rtol, atol
        )
    if "objective" in result:
        differences["result_objective"] = _close(
            metrics["objective"], result["objective"], "result_objective", rtol, atol
        )
    prediction_comparison = "No original pointwise predictions supplied; independent predictions checked through recorded MSE/objective only."
    if "predictions_path" in result:
        prediction_path = Path(result["predictions_path"])
        sources.append(
            _bound(
                prediction_path,
                result.get(
                    "predictions_sha256", bindings.get(str(prediction_path.resolve()))
                ),
            )
        )
        with np.load(prediction_path, allow_pickle=False) as archive:
            for split, actual in (
                ("train", prediction_train),
                ("endpoint", prediction_endpoint),
            ):
                key = split if split in archive.files else f"prediction_{split}"
                differences[f"prediction_{split}"] = _close(
                    actual,
                    archive[key],
                    f"prediction_{split}",
                    forward_rtol,
                    forward_atol,
                )
        prediction_comparison = (
            "All saved train and endpoint pointwise predictions independently compared."
        )
    return {
        "status": "passed",
        "result_path": str(result_path),
        "case": case,
        "sources": sources,
        "inventory": inventory,
        "ridge": ridge,
        "metrics": {
            "train_mse": metrics["mse"],
            "endpoint_mse": endpoint_mse,
            "train_penalty": metrics["penalty"],
            "train_objective": metrics["objective"],
            "raw_readout_l2": metrics["raw_coefficient_l2"],
            "standardized_readout_l2": metrics["standardized_coefficient_l2"],
        },
        "absolute_differences": differences,
        "tolerances": {
            "rtol": rtol,
            "atol": atol,
            "forward_rtol": forward_rtol,
            "forward_atol": forward_atol,
        },
        "train_rows": len(train),
        "endpoint_rows": len(endpoint),
        "prediction_check": prediction_comparison,
        "dataset_identity_check": dataset_identity_check,
        "initializer_train_hash_check": initializer_train_hash_check,
        "implementation": "Independent original NumPy replay with validated fixed raw-to-node duplication. No Torch model calls, fitting, teacher inputs, or state changes.",
        "audit_source_sha256": sha256(__file__),
        "replay_source_sha256": sha256(REPLAY.__file__),
    }


def quadratic_features(x):
    """One fixed intercept and nine monomials per supplied 3D block: 37 total."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (4, 3) or not np.isfinite(x).all():
        raise ValueError(
            "Finite [N,4,3] raw inputs required for the 37-column reference"
        )
    columns, names = [np.ones(len(x))], ["constant"]
    for block in range(4):
        for axis in range(3):
            columns.append(x[:, block, axis])
            names.append(f"block{block}:x{axis}")
        for first in range(3):
            for second in range(first, 3):
                columns.append(x[:, block, first] * x[:, block, second])
                names.append(f"block{block}:x{first}*x{second}")
    return np.column_stack(columns), names


def fit_quadratic_reference(x_train, y_train, *, rcond=1e-12):
    """Fit only supplied TRAIN scalar labels; no task or endpoint argument."""
    if not math.isfinite(rcond) or rcond <= 0:
        raise ValueError("Positive finite prespecified least-squares rcond required")
    design, names = quadratic_features(x_train)
    y_train = np.asarray(y_train, dtype=np.float64)
    if (
        len(design) == 0
        or y_train.shape != (len(design),)
        or not np.isfinite(y_train).all()
    ):
        raise ValueError("Nonempty TRAIN scalar labels required")
    coefficients, _, rank, singular = np.linalg.lstsq(design, y_train, rcond=rcond)
    if not np.isfinite(coefficients).all() or not np.isfinite(singular).all():
        raise FloatingPointError("Nonfinite quadratic least-squares solution")
    return {
        "coefficients": coefficients,
        "feature_names": names,
        "stored_coefficients": 37,
        "fit_rank": int(rank),
        "singular_values": singular,
        "rcond": float(rcond),
        "train_rows": len(design),
        "train_x_sha256": _array_sha(x_train),
        "train_y_sha256": _array_sha(y_train),
        "scope": "TRAIN-only ordinary least squares on a supplied fixed additive quadratic dictionary. All 37 coefficient slots count, including any numerically unidentifiable slots. This reference is not a learned-feature architecture.",
    }


def predict_quadratic_reference(model, x):
    design, names = quadratic_features(x)
    if names != model["feature_names"] or np.asarray(model["coefficients"]).shape != (
        37,
    ):
        raise AssertionError(
            "Quadratic reference dictionary or coefficient inventory changed"
        )
    return design @ model["coefficients"]


def save_quadratic_reference(data_path, output_dir, *, rcond=1e-12):
    """Fit TRAIN first, then save unchanged train/endpoint predictions and risk."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = _dataset(data_path)
    model = fit_quadratic_reference(data["x_train"], data["y_train"], rcond=rcond)
    prediction_train = predict_quadratic_reference(model, data["x_train"])
    prediction_endpoint = predict_quadratic_reference(model, data["x_endpoint"])
    metadata = {
        key: value for key, value in model.items() if not isinstance(value, np.ndarray)
    }
    state_path, predictions_path = output / "reference.npz", output / "predictions.npz"
    with state_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            coefficients=model["coefficients"],
            singular_values=model["singular_values"],
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
    with predictions_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            prediction_train=prediction_train,
            prediction_endpoint=prediction_endpoint,
        )
    receipt = {
        "status": "complete",
        "model": metadata,
        "metrics": {
            "train_mse": float(np.mean((prediction_train - data["y_train"]) ** 2)),
            "endpoint_mse": float(
                np.mean((prediction_endpoint - data["y_endpoint"]) ** 2)
            ),
        },
        "sources": [
            _bound(data_path),
            _bound(__file__),
            _bound(state_path),
            _bound(predictions_path),
        ],
        "scope": "One fit per supplied dataset; TRAIN coefficients frozen before endpoint evaluation. Endpoint outcomes do not select or change this reference.",
    }
    with (output / "result.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--result", required=True)
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--rtol", type=float, default=3e-6)
    audit.add_argument("--atol", type=float, default=5e-15)
    audit.add_argument("--forward-rtol", type=float, default=3e-6)
    audit.add_argument("--forward-atol", type=float, default=5e-12)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--data-path", required=True)
    reference.add_argument("--output-dir", required=True)
    reference.add_argument("--rcond", type=float, default=1e-12)
    args = parser.parse_args()
    if args.command == "reference":
        result = save_quadratic_reference(
            args.data_path, args.output_dir, rcond=args.rcond
        )
    else:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=False)
        try:
            result = audit_result(
                args.result,
                rtol=args.rtol,
                atol=args.atol,
                forward_rtol=args.forward_rtol,
                forward_atol=args.forward_atol,
            )
        except Exception as error:
            failure = {
                "status": "failed",
                "error": repr(error),
                "result_path": str(Path(args.result).resolve()),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "audit_source_sha256": sha256(__file__),
                "tolerances": {
                    "rtol": args.rtol,
                    "atol": args.atol,
                    "forward_rtol": args.forward_rtol,
                    "forward_atol": args.forward_atol,
                },
            }
            with (output / "failure.json").open("x") as handle:
                json.dump(failure, handle, indent=2, allow_nan=False)
            raise
        with (output / "audit.json").open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output_dir": str(Path(args.output_dir).resolve()),
            }
        )
    )


if __name__ == "__main__":
    main()

"""Independent radial-task replay, fixed homogeneous references and label checks.

No production model, fitter, task or teacher imports. A native source snapshot
needs the adjacent rank_study_replay.py, rank_learning.py and
radial_width_learning.py files; the latter two are hashed, never executed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from fractions import Fraction
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


def train_fingerprint(value, encoding="raw_bytes"):
    value = np.ascontiguousarray(value)
    if encoding == "raw_bytes":
        payload = value.tobytes()
    elif encoding == "dtype_shape_bytes":
        payload = (
            str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()
        )
    else:
        raise AssertionError("Unknown declared TRAIN array hash encoding")
    return hashlib.sha256(payload).hexdigest()


def _load_replay():
    path = Path(__file__).with_name("rank_study_replay.py")
    if (
        sha256(path)
        != "cfc3bd106a7e87fcf26001b528575114900e649cd1344720cfd957840c3dc7a5"
    ):
        raise AssertionError("Independent replay source fingerprint changed")
    spec = importlib.util.spec_from_file_location("radial_width_numpy_replay", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPLAY = _load_replay()
GEOMETRIES = {
    "full": (0, 1),
    "rank2": (2, 1),
    "width1": (1, 1),
    "width2": (1, 2),
    "width3": (1, 3),
    "width4": (1, 4),
    "width5": (1, 5),
    "width8": (1, 8),
}


def train_prefix(data, train_n=None):
    """Use the case's TRAIN prefix; endpoints and cached full arrays stay fixed."""
    n = len(data["x_train"]) if train_n is None else _integer(train_n, "train_n")
    if n > len(data["x_train"]) or len(data["y_train"]) != len(data["x_train"]):
        raise AssertionError("TRAIN prefix is unavailable or labels differ in length")
    return {**data, "x_train": data["x_train"][:n], "y_train": data["y_train"][:n]}


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
    width = native.get("initialization_receipt", {}).get("_radial_width")
    if not isinstance(width, dict) or width.get("geometry") not in GEOMETRIES:
        raise AssertionError("Native state lacks a supported _radial_width map")
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
        ("radial_width_learning.py", wrapper_source_path),
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
        if case["coverage"] == "plane8":
            saved_lines = state["metadata"]["initialization_receipt"][
                "direction_bank"
            ].get("coverage_lines")
            if (
                saved_coverage != "uniform_plane_lines"
                or isinstance(saved_lines, bool)
                or saved_lines != 8
            ):
                raise AssertionError(
                    "Declared plane8 coverage requires eight uniform plane lines"
                )
        elif saved_coverage != case["coverage"]:
            raise AssertionError("Case and saved initialization disagree on coverage")
    data = train_prefix(_dataset(data_path, dataset_cache), case.get("train_n"))
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
            actual = train_fingerprint(
                data[array_key], estimator.get("train_hash_encoding", "raw_bytes")
            )
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
        "inherited_audit_source_sha256": "1b26d719e9345918920410457066a330b0c4e86234a329c66a9d662b64687ff8",
        "replay_source_sha256": sha256(REPLAY.__file__),
    }


def homogeneous_powers(m):
    """Public degree prior: all degree-2m monomials in three raw coordinates."""
    if isinstance(m, bool) or m not in (2, 4):
        raise ValueError("Supported public radial degrees are m=2 and m=4")
    degree = 2 * int(m)
    return [
        (a, b, degree - a - b) for a in range(degree + 1) for b in range(degree - a + 1)
    ]


def homogeneous_features(x, m):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (4, 3) or not np.isfinite(x).all():
        raise ValueError("Finite raw [N,4,3] inputs required")
    powers = homogeneous_powers(m)
    columns, names = [np.ones(len(x))], ["constant"]
    for block in range(4):
        for exponent in powers:
            columns.append(np.prod(x[:, block, :] ** np.asarray(exponent), axis=1))
            names.append(f"block{block}:powers{exponent}")
    matrix = np.column_stack(columns)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("Nonfinite fixed polynomial features")
    return matrix, names


def fit_homogeneous_reference(x_train, y_train, m, *, rcond=1e-12):
    """TRAIN-only ordinary least squares; no private axes or endpoint argument."""
    if not math.isfinite(rcond) or rcond <= 0:
        raise ValueError("Positive finite predeclared numerical-rank cutoff required")
    matrix, names = homogeneous_features(x_train, m)
    y = np.asarray(y_train, dtype=np.float64)
    if len(matrix) == 0 or y.shape != (len(matrix),) or not np.isfinite(y).all():
        raise ValueError("Nonempty matching finite scalar TRAIN labels required")
    coefficients, _, rank, singular = np.linalg.lstsq(matrix, y, rcond=rcond)
    if not np.isfinite(coefficients).all() or not np.isfinite(singular).all():
        raise FloatingPointError("Nonfinite ordinary least-squares result")
    slots = 1 + 4 * len(homogeneous_powers(m))
    assert slots == (61 if m == 2 else 181)
    return {
        "m": int(m),
        "coefficients": coefficients,
        "feature_names": names,
        "stored_coefficients": slots,
        "fit_rank": int(rank),
        "rcond": float(rcond),
        "singular_values": singular,
        "train_rows": len(matrix),
        "train_x_sha256": _array_sha(x_train),
        "train_y_sha256": _array_sha(y),
        "scope": "TRAIN-only ordinary least squares on supplied fixed homogeneous raw-coordinate monomials. All coefficient slots count, including the intercept and numerically unidentifiable slots. Fixed feature arithmetic and the supplied degree prior are disclosed; no plane/profile coefficients are supplied.",
    }


def predict_homogeneous_reference(model, x):
    matrix, names = homogeneous_features(x, model["m"])
    if names != model["feature_names"] or np.asarray(model["coefficients"]).shape != (
        matrix.shape[1],
    ):
        raise AssertionError(
            "Fixed dictionary or complete coefficient inventory changed"
        )
    return matrix @ model["coefficients"]


def save_homogeneous_reference(data_path, output_dir, m, *, train_n=None, rcond=1e-12):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    data = train_prefix(_dataset(data_path), train_n)
    model = fit_homogeneous_reference(data["x_train"], data["y_train"], m, rcond=rcond)
    # No endpoint array enters the fitting API. Coefficients remain frozen here.
    predictions = {
        split: predict_homogeneous_reference(model, data[f"x_{split}"])
        for split in ("train", "endpoint")
    }
    metadata = {
        key: value for key, value in model.items() if not isinstance(value, np.ndarray)
    }
    state_path, prediction_path = output / "reference.npz", output / "predictions.npz"
    with state_path.open("xb") as handle:
        np.savez_compressed(
            handle,
            coefficients=model["coefficients"],
            singular_values=model["singular_values"],
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
    with prediction_path.open("xb") as handle:
        np.savez_compressed(handle, **predictions)
    receipt = {
        "status": "complete",
        "model": metadata,
        "metrics": {
            f"{split}_mse": float(
                np.mean((predictions[split] - data[f"y_{split}"]) ** 2)
            )
            for split in ("train", "endpoint")
        },
        "sources": [
            _bound(path) for path in (data_path, __file__, state_path, prediction_path)
        ],
        "scope": "Fixed homogeneous reference fitted only on the declared TRAIN prefix, then scored unchanged. No selection across degree, cutoff or endpoint risk.",
    }
    with (output / "result.json").open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return receipt


def radial_moments(m, radius):
    """Independent exact-rational public moments, then radius rescaling."""
    homogeneous_powers(m)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Positive finite public radius required")

    def moment(j):
        return Fraction(
            3 * 4**j * math.factorial(j) ** 2, (2 * j + 3) * math.factorial(2 * j + 1)
        )

    unit_mean = moment(m)
    unit_variance = moment(2 * m) - unit_mean**2
    return float(unit_mean) * radius ** (2 * m), float(unit_variance) * radius ** (
        4 * m
    )


def post_fit_barrier(completion_path, completion_sha256, expected_fit_count):
    """Verify the explicit completion receipt before any private file is read."""
    _integer(expected_fit_count, "expected_fit_count")
    binding = _bound(completion_path, completion_sha256)
    completion = read_json(completion_path)
    if completion.get("status") not in ("complete", "fits", "fits_complete"):
        raise AssertionError("Private audit requires the completed-fit barrier")
    count = completion.get("completed_fits", completion.get("fits"))
    if isinstance(count, bool) or count != expected_fit_count:
        raise AssertionError(
            "Completed fit count differs from the frozen expected count"
        )
    if completion.get("failed_fits", 0) != 0 or completion.get("failures", 0) not in (
        0,
        [],
    ):
        raise AssertionError("Completion receipt contains unresolved fit failures")
    return binding


def audit_radial_labels(
    data_path,
    specification_path,
    *,
    data_sha256,
    specification_sha256,
    completion_path,
    completion_sha256,
    expected_fit_count,
    rtol=1e-11,
    atol=1e-12,
):
    """Private-projector verification only after the hash-bound fit barrier."""
    barrier = post_fit_barrier(completion_path, completion_sha256, expected_fit_count)
    sources = [
        barrier,
        _bound(data_path, data_sha256),
        _bound(specification_path, specification_sha256),
    ]
    specification = read_json(specification_path)
    m = specification.get("power", specification.get("m"))
    radius = float(specification.get("radius", specification.get("R")))
    if specification.get("blocks", specification.get("K")) != 4:
        raise AssertionError(
            "Private specification differs from supplied four-block law"
        )
    mean, variance = radial_moments(m, radius)
    projectors = np.asarray(specification["projectors"], dtype=np.float64)
    if projectors.shape != (4, 3, 3) or not np.isfinite(projectors).all():
        raise AssertionError("Invalid private projector dimensions/finiteness")
    if "projectors_sha256" in specification and specification[
        "projectors_sha256"
    ] != train_fingerprint(projectors, "dtype_shape_bytes"):
        raise AssertionError(
            "Private projector array hash differs from its specification"
        )
    _close(projectors, projectors.transpose(0, 2, 1), "projector_symmetry", 0, 2e-12)
    _close(projectors @ projectors, projectors, "projector_idempotence", 0, 2e-12)
    _close(
        np.trace(projectors, axis1=1, axis2=2),
        np.full(4, 2.0),
        "projector_rank_trace",
        0,
        2e-12,
    )
    data = _dataset(data_path)
    differences = {}
    for split in ("train", "endpoint"):
        x = data[f"x_{split}"]
        if np.max(np.sum(x * x, axis=2)) > radius**2 + 2e-13:
            raise AssertionError("An observed raw block lies outside the declared ball")
        squared_radius = np.einsum("nki,kij,nkj->nk", x, projectors, x)
        labels = np.sum(squared_radius**m - mean, axis=1) / math.sqrt(4 * variance)
        differences[split] = _close(
            labels, data[f"y_{split}"], f"{split}_labels", rtol, atol
        )
    return {
        "status": "passed",
        "sources": sources,
        "m": int(m),
        "radius": radius,
        "mean": mean,
        "variance": variance,
        "rows": {split: len(data[f"x_{split}"]) for split in ("train", "endpoint")},
        "maximum_label_differences": differences,
        "tolerances": {"rtol": rtol, "atol": atol},
        "scope": "Post-fit private verification of full saved label arrays by independent projector and exact-moment algebra. No teacher code imports, label changes, training, or selection.",
    }


def audit_constructed_state(
    state_path,
    data_path,
    *,
    train_n,
    m,
    radius,
    intervals,
    state_sha256,
    data_sha256,
    predictions_path=None,
    predictions_sha256=None,
    learner_source_path=None,
    task_source_path=None,
    rtol=3e-6,
    atol=5e-12,
):
    """Replay a known-profile construction; no optimizer/objective is invented."""
    sources = [_bound(state_path, state_sha256), _bound(data_path, data_sha256)]
    state = REPLAY.load_state(state_path)
    initialization = state["metadata"]["initialization_receipt"]
    inventory = initialization["_radial_width"]
    _integer(intervals, "intervals", minimum=2)
    homogeneous_powers(m)
    mean, variance = radial_moments(m, radius)
    expected_capacity = (m + 1) * intervals
    if (
        inventory["geometry"] != "rank2"
        or state["metadata"]["family"] != "relu"
        or inventory["capacities"] != [expected_capacity] * 4
    ):
        raise AssertionError(
            "Constructed profile differs from the counted m+1-line spline model"
        )
    if inventory["stored_parameters"] != 16 * expected_capacity + 29:
        raise AssertionError("Constructed parameter count differs from 4K(m+1)S+7K+1")
    if state["parameter_count"] != inventory["stored_parameters"]:
        raise AssertionError(
            "Constructive stored arrays differ from its scalar inventory"
        )
    expected_inventory = {
        "raw_blocks": 4,
        "inputs_per_block": 3,
        "nodes_per_raw_block": 1,
        "nodes": 4,
        "branches": 4 * expected_capacity,
        "branch_parameter_cost": 4,
        "fixed_parameter_overhead": 29,
        "fixed_soma_parameter_slots": 4,
        "fixed_node_map_integer_entries": 0,
        "fixed_branch_group_integer_entries": 4 * expected_capacity,
    }
    for key, expected in expected_inventory.items():
        if inventory[key] != expected:
            raise AssertionError(f"Constructive inventory differs: {key}")
    sources.extend(
        (
            _bound(
                learner_source_path
                or Path(__file__).with_name("radial_width_constructive.py"),
                initialization["learner_source_sha256"],
            ),
            _bound(
                task_source_path or Path(__file__).with_name("radial_width_tasks.py"),
                initialization["task_source_sha256"],
            ),
        )
    )
    profile = initialization["profile"]
    if (
        profile["power"] != m
        or profile["radius"] != radius
        or initialization["intervals_per_line"] != intervals
    ):
        raise AssertionError(
            "Declared constructive public profile differs from the case"
        )
    _close(profile["mean"], mean, "constructive_mean", 2e-12, 0)
    _close(profile["variance"], variance, "constructive_variance", 2e-12, 0)
    data = train_prefix(_dataset(data_path), train_n)
    estimator = initialization["direction_bank"]["estimator"]
    for key, value in (
        ("train_x_sha256", data["x_train"]),
        ("train_y_sha256", data["y_train"]),
    ):
        if estimator[key] != train_fingerprint(
            value, estimator.get("train_hash_encoding", "dtype_shape_bytes")
        ):
            raise AssertionError(
                "Constructed plane estimate used a different TRAIN prefix"
            )
    if estimator["rows"] != len(data["x_train"]):
        raise AssertionError("Constructed plane-estimator row count differs")
    x, y = data["x_train"], data["y_train"]
    # Independently accumulate the raw-label Stein matrix without allocating
    # the production [N,K,3,3] kernel and without calling its estimator.
    moment = 8 * np.einsum("n,nki,nkj->kij", y, x, x)
    diagonal = -4 * np.sum(y[:, None] * (radius**2 - np.sum(x * x, axis=2)), axis=0)
    moment += diagonal[:, None, None] * np.eye(3)
    moment /= len(x) * (8 * radius**4 / 35)
    moment_difference = _close(
        moment, estimator["moment"], "raw_TRAIN_Stein_matrix", 2e-11, 2e-11
    )
    _, eigenvectors = np.linalg.eigh(moment)
    expected_projectors = eigenvectors[:, :, -2:] @ eigenvectors[:, :, -2:].transpose(
        0, 2, 1
    )
    bank = state["arrays"]["basis"]
    _close(
        bank.transpose(0, 2, 1) @ bank,
        np.tile(np.eye(2), (4, 1, 1)),
        "constructive_basis_orthonormality",
        0,
        2e-12,
    )
    plane_difference = _close(
        bank @ bank.transpose(0, 2, 1),
        expected_projectors,
        "TRAIN_top_algebraic_plane",
        2e-10,
        2e-10,
    )
    # Verify every analytic spline coefficient from the public profile. Two
    # Gauss points per segment integrate affine spline times quadratic density.
    knots = np.linspace(-radius, radius, intervals + 1)
    slopes = np.diff(knots ** (2 * m)) / np.diff(knots)
    slope_changes = np.concatenate((slopes[:1], np.diff(slopes)))
    angles = np.arange(m + 1) * np.pi / (m + 1)
    directions = np.repeat(
        np.column_stack((np.cos(angles), np.sin(angles))), intervals, axis=0
    )
    cm = math.comb(2 * m, m) / 4**m
    nodes = np.array([-1 / math.sqrt(3), 1 / math.sqrt(3)])
    segment_mid = (knots[1:] + knots[:-1]) / 2
    segment_half = np.diff(knots) / 2
    abscissas = segment_mid[:, None] + segment_half[:, None] * nodes
    values = knots[:-1, None] ** (2 * m) + slopes[:, None] * (
        abscissas - knots[:-1, None]
    )
    spline_mean = float(
        np.sum(
            segment_half[:, None]
            * values
            * (3 / (4 * radius))
            * (1 - (abscissas / radius) ** 2)
        )
    )
    expected_arrays = {
        "branch_coefficients": np.tile(directions, (4, 1)),
        "threshold": np.tile(-knots[:-1], 4 * (m + 1)),
        "internal_readout": np.tile(
            slope_changes / ((m + 1) * cm * math.sqrt(variance)), 4 * (m + 1)
        ),
        "bias": np.array(
            2 * (radius ** (2 * m) - spline_mean) / (cm * math.sqrt(variance))
        ),
    }
    analytic_differences = {
        key: _close(state["arrays"][key], value, f"analytic_{key}", 2e-12, 2e-12)
        for key, value in expected_arrays.items()
    }
    predictions = {
        split: REPLAY.predict(state, data[f"x_{split}"])
        for split in ("train", "endpoint")
    }
    differences = {}
    if predictions_path is not None:
        sources.append(_bound(predictions_path, predictions_sha256))
        with np.load(predictions_path, allow_pickle=False) as saved:
            differences = {
                split: _close(
                    predictions[split], saved[split], f"constructed_{split}", rtol, atol
                )
                for split in ("train", "endpoint")
            }
    return {
        "status": "passed",
        "inventory": inventory,
        "sources": sources,
        "metrics": {
            f"{split}_mse": float(
                np.mean((predictions[split] - data[f"y_{split}"]) ** 2)
            )
            for split in ("train", "endpoint")
        },
        "prediction_differences": differences,
        "train_rows": len(data["x_train"]),
        "independent_raw_Stein_matrix_max_difference": moment_difference,
        "independent_top_algebraic_projector_max_difference": plane_difference,
        "analytic_array_max_differences": analytic_differences,
        "scope": "Counted known-profile ReLU construction using a TRAIN-estimated plane. Analytic coefficients are fully stored and counted; this is not a generic optimizer fit or evidence that all stored slots were learned freely.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--result", required=True)
    audit.add_argument("--output-dir", required=True)
    reference = subparsers.add_parser("reference")
    reference.add_argument("--data-path", required=True)
    reference.add_argument("--output-dir", required=True)
    reference.add_argument("--m", type=int, choices=(2, 4), required=True)
    reference.add_argument("--train-n", type=int, required=True)
    reference.add_argument("--rcond", type=float, default=1e-12)
    args = parser.parse_args()
    if args.command == "reference":
        result = save_homogeneous_reference(
            args.data_path,
            args.output_dir,
            args.m,
            train_n=args.train_n,
            rcond=args.rcond,
        )
    else:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=False)
        try:
            result = audit_result(args.result)
        except Exception as error:
            with (output / "failure.json").open("x") as handle:
                json.dump(
                    {
                        "status": "failed",
                        "error": repr(error),
                        "result_path": str(Path(args.result).resolve()),
                        "audit_source_sha256": sha256(__file__),
                    },
                    handle,
                    indent=2,
                )
            raise
        with (output / "audit.json").open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
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

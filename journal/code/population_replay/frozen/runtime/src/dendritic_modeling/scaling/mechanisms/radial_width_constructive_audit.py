"""Independent post-completion audit of learned radial constructions and oracles.

No production task, model, constructor, estimator or campaign imports. Original
strict checks remain immutable; every failure is saved without tolerance changes.
"""

from __future__ import annotations

import argparse
import hashlib
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

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
import numpy as np  # noqa: E402


def _load(name, expected):
    path = Path(__file__).with_name(name + ".py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise AssertionError("Independent audit dependency changed: " + name)
    spec = importlib.util.spec_from_file_location(
        "radial_constructive_independent_" + name, path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AUDIT = _load(
    "radial_width_audit",
    "6b00bccf34a40b21f5bd998767a96141e91f8de47dd8a5806253fff4dc400fc8",
)
GENERIC = _load(
    "radial_width_audit_campaign",
    "63278632d520ac3b0d2b72a7e60273bd82fb92dac148c6b0a19b1689f1f6ec38",
)
REPLAY = AUDIT.REPLAY
read_json = AUDIT.read_json
_bound = AUDIT._bound
_close = AUDIT._close
_integer = AUDIT._integer
radial_moments = AUDIT.radial_moments
homogeneous_powers = AUDIT.homogeneous_powers
train_prefix = AUDIT.train_prefix
train_fingerprint = AUDIT.train_fingerprint
_dataset = AUDIT._dataset


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _failure(error):
    return {
        "status": "failed",
        "error": repr(error),
        "traceback": traceback.format_exc(),
    }


def oracle_barriers(
    main_gate_path,
    main_gate_sha256,
    learned_path,
    learned_sha256,
    expected_learned_states,
):
    """Both barriers checked before a private specification may be opened."""
    sources = [
        _bound(main_gate_path, main_gate_sha256),
        _bound(learned_path, learned_sha256),
    ]
    learned = read_json(learned_path)
    if (
        learned.get("status") != "complete"
        or learned.get("states") != expected_learned_states
        or learned.get("failed_states") != 0
    ):
        raise AssertionError("Oracle audit requires all learned states complete")
    gate = read_json(main_gate_path)
    if gate.get("status") != "passed" or gate.get("failed_results") != 0:
        raise AssertionError("Oracle audit requires generic completion gate")
    sources.append(
        AUDIT.post_fit_barrier(
            Path(gate["main_root"]) / "complete.json",
            gate["barriers"]["complete.json"],
            gate["verified_results"],
        )
    )
    for path, digest in learned["packets"].items():
        sources.append(_bound(path, digest))
    return sources


def audit_oracle_state(
    state_path,
    data_path,
    *,
    train_n,
    specification_path,
    specification_sha256,
    main_gate_path,
    main_gate_sha256,
    learned_completion_path,
    learned_completion_sha256,
    expected_learned_states,
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
    barrier_sources = oracle_barriers(
        main_gate_path,
        main_gate_sha256,
        learned_completion_path,
        learned_completion_sha256,
        expected_learned_states,
    )
    sources = [
        *barrier_sources,
        _bound(state_path, state_sha256),
        _bound(data_path, data_sha256),
        _bound(specification_path, specification_sha256),
    ]
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
    if estimator is not None:
        raise AssertionError("Privileged reference must not claim a TRAIN estimator")
    privileged = initialization["privileged_reference"]
    if (
        Path(privileged["private_specification_path"]).resolve()
        != Path(specification_path).resolve()
        or privileged["private_specification_sha256"] != specification_sha256
        or privileged["main_completion_gate_sha256"] != main_gate_sha256
        or privileged["learned_complete_sha256"] != learned_completion_sha256
        or privileged["known_profile"] is not True
        or privileged["learner_observations"] != 0
    ):
        raise AssertionError(
            "Privileged state provenance differs from both completion gates"
        )
    specification = read_json(specification_path)
    if (
        specification["power"] != m
        or specification["radius"] != radius
        or specification["blocks"] != 4
    ):
        raise AssertionError("Privileged reference task identity differs")
    expected_projectors = np.asarray(specification["projectors"], dtype=np.float64)
    if (
        train_fingerprint(expected_projectors, "dtype_shape_bytes")
        != specification["projectors_sha256"]
    ):
        raise AssertionError("Private projector array hash differs")
    _close(
        expected_projectors,
        expected_projectors.transpose(0, 2, 1),
        "private_symmetry",
        0,
        2e-12,
    )
    _close(
        expected_projectors @ expected_projectors,
        expected_projectors,
        "private_idempotence",
        0,
        2e-12,
    )
    _close(
        np.trace(expected_projectors, axis1=1, axis2=2),
        np.full(4, 2.0),
        "private_rank2",
        0,
        2e-12,
    )
    bank = state["arrays"]["basis"]
    if train_fingerprint(bank, "dtype_shape_bytes") != initialization["bank_sha256"]:
        raise AssertionError(
            "Oracle native bank hash differs from construction receipt"
        )
    _close(
        bank.transpose(0, 2, 1) @ bank,
        np.tile(np.eye(2), (4, 1, 1)),
        "oracle_basis_orthonormality",
        0,
        2e-12,
    )
    plane_difference = _close(
        bank @ bank.transpose(0, 2, 1),
        expected_projectors,
        "privileged_true_plane",
        0,
        2e-12,
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
        "estimator_present": False,
        "independent_private_projector_max_difference": plane_difference,
        "analytic_array_max_differences": analytic_differences,
        "scope": "Privileged true-plane and known-profile ReLU construction, never a learned result. All analytic coefficients stored and counted; no optimizer or learned-rate claim.",
    }


def _inside(root, path):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise AssertionError("Constructive artifact escapes root")
    return path


def verify_complete(root):
    """Verify both complete inventories before any privileged numerical audit."""
    root = Path(root).resolve()
    complete = read_json(root / "complete.json")
    if complete.get("status") != "complete" or complete.get("failed_states") != 0:
        raise AssertionError("Constructive campaign must be completely sealed")
    bindings = [
        _bound(root / "complete.json"),
        _bound(root / "initialized.json", complete["initialized_sha256"]),
    ]
    initialized = read_json(root / "initialized.json")
    for name, digest in initialized["bindings"].items():
        bindings.append(_bound(_inside(root, root / name), digest))
    main = Path(initialized["main_root"]).resolve()
    generic = GENERIC.verify_complete(main)
    gate = read_json(root / "main_completion_gate.json")
    if (
        gate["status"] != "passed"
        or gate["verified_results"] != len(generic["cases"])
        or gate["failed_results"] != 0
    ):
        raise AssertionError("Constructive generic gate count differs")
    for name, digest in gate["barriers"].items():
        bindings.append(_bound(main / name, digest))
    config = read_json(root / "config.json")
    if (
        config["teachers"] != generic["config"]["confirmation_teachers"]
        or config["tasks"] != generic["config"]["tasks"]
        or config["train_sizes"] != generic["config"]["constructive_n"]
        or config["intervals"] != generic["config"]["constructive_intervals"]
    ):
        raise AssertionError(
            "Constructive grid differs from frozen generic declaration"
        )
    expected_learned = set(
        itertools.product(
            config["tasks"],
            config["teachers"],
            config["observations"],
            config["train_sizes"],
            config["intervals"],
        )
    )
    expected_oracle = (
        set(itertools.product(config["tasks"], config["teachers"], config["intervals"]))
        if initialized["include_oracle"]
        else set()
    )
    if (
        len(expected_learned) != config["expected_learned_states"]
        or complete["learned_states"] != len(expected_learned)
        or complete["oracle_states"] != len(expected_oracle)
    ):
        raise AssertionError("Constructive frozen state count differs")
    public = read_json(root / "public_data_inventory.json")
    expected_keys = {
        f"{t}_t{i}_o{o}"
        for t, i, o in itertools.product(
            config["tasks"], config["teachers"], config["observations"]
        )
    }
    if set(public) != expected_keys:
        raise AssertionError("Public constructive observation inventory differs")
    for entry in public.values():
        bindings.extend(
            [
                _bound(entry["path"], entry["sha256"]),
                _bound(entry["sidecar_path"], entry["sidecar_sha256"]),
            ]
        )
    learned_seal = read_json(root / "learned_complete.json")
    bindings.append(
        _bound(root / "learned_complete.json", complete["learned_complete_sha256"])
    )
    if (
        learned_seal["status"] != "complete"
        or learned_seal["failed_states"] != 0
        or learned_seal["states"] != len(expected_learned)
        or learned_seal["initialized_sha256"] != complete["initialized_sha256"]
    ):
        raise AssertionError("Learned completion seal differs")
    learned = []
    actual = set()
    for path, digest in learned_seal["packets"].items():
        bindings.append(_bound(_inside(root, path), digest))
        packet = read_json(path)
        if (
            packet["status"] != "complete"
            or packet["phase"] != "learned"
            or packet["initialized_sha256"] != complete["initialized_sha256"]
        ):
            raise AssertionError("Learned packet not complete/source-bound")
        barrier_path = Path(path).parent / "construction_complete.json"
        barrier = read_json(barrier_path)
        if barrier["status"] != "complete" or len(packet["states"]) != len(
            config["intervals"]
        ):
            raise AssertionError(
                "All interval states must be frozen before endpoint evaluation"
            )
        states = {}
        for item in packet["states"]:
            p = _inside(root, item["path"])
            bindings.append(_bound(p, item["sha256"]))
            row = read_json(p)
            case = row["case"]
            identity = tuple(
                case[k]
                for k in ("task", "teacher", "observation", "train_n", "intervals")
            )
            if (
                set(case) != {"task", "teacher", "observation", "train_n", "intervals"}
                or identity not in expected_learned
                or identity in actual
                or row["status"] != "complete"
                or row["phase"] != "learned"
                or any(case[k] != v for k, v in packet["packet"].items())
            ):
                raise AssertionError("Missing/duplicate/out-of-grid learned state")
            if row["initialized_sha256"] != complete["initialized_sha256"]:
                raise AssertionError("Learned state initialization differs")
            bindings.append(_bound(barrier_path, row["construction_barrier_sha256"]))
            for stem in ("state", "predictions", "estimator"):
                bindings.append(
                    _bound(_inside(root, row[stem + "_path"]), row[stem + "_sha256"])
                )
            key = f"{case['task']}_t{case['teacher']}_o{case['observation']}"
            if (
                row["data_path"] != public[key]["path"]
                or row["data_sha256"] != public[key]["sha256"]
            ):
                raise AssertionError("Learned state uses wrong observations")
            if (
                row["train_rows"] != case["train_n"]
                or row["endpoint_rows"] != config["endpoint_rows"]
            ):
                raise AssertionError("Learned TRAIN prefix/endpoint count differs")
            states[row["state_path"]] = row["state_sha256"]
            actual.add(identity)
            learned.append({"path": str(p), "sha256": item["sha256"], "case": case})
        if barrier["states"] != states:
            raise AssertionError("Pre-endpoint construction state seal differs")
    if actual != expected_learned:
        raise AssertionError("Learned grid incomplete")
    oracles = []
    diagnostics = []
    actual = set()
    if initialized["include_oracle"]:
        bindings.append(
            _bound(root / "oracle_complete.json", complete["oracle_complete_sha256"])
        )
        seal = read_json(root / "oracle_complete.json")
        if (
            seal["status"] != "complete"
            or seal["failed_states"] != 0
            or seal["states"] != len(expected_oracle)
            or seal["evaluations"] != len(expected_oracle) * len(config["observations"])
            or seal["learned_complete_sha256"] != complete["learned_complete_sha256"]
        ):
            raise AssertionError("Oracle completion/grid differs")
        for path, digest in seal["packets"].items():
            bindings.append(_bound(_inside(root, path), digest))
            packet = read_json(path)
            if (
                packet["status"] != "complete"
                or packet["phase"] != "privileged_oracle"
                or packet["learned_complete_sha256"]
                != complete["learned_complete_sha256"]
            ):
                raise AssertionError("Oracle packet lacks learned-completion lineage")
            bindings.append(
                _bound(
                    packet["plane_diagnostics_path"], packet["plane_diagnostics_sha256"]
                )
            )
            diagnostics.extend(read_json(packet["plane_diagnostics_path"]))
            for item in packet["states"]:
                p = _inside(root, item["path"])
                bindings.append(_bound(p, item["sha256"]))
                row = read_json(p)
                case = row["case"]
                identity = tuple(case[k] for k in ("task", "teacher", "intervals"))
                if (
                    set(case) != {"task", "teacher", "intervals"}
                    or identity not in expected_oracle
                    or identity in actual
                    or row["phase"] != "privileged_oracle"
                    or row["status"] != "complete"
                    or row["learned_complete_sha256"]
                    != complete["learned_complete_sha256"]
                ):
                    raise AssertionError("Missing/duplicate/out-of-grid oracle state")
                bindings.append(
                    _bound(_inside(root, row["state_path"]), row["state_sha256"])
                )
                private = (
                    main
                    / "private_teachers"
                    / f"confirmation_{case['task']}_t{case['teacher']}.json"
                )
                if Path(row["private_specification_path"]).resolve() != private:
                    raise AssertionError("Oracle private specification path differs")
                if [e["observation"] for e in row["evaluations"]] != config[
                    "observations"
                ]:
                    raise AssertionError("Oracle paired evaluations differ")
                for e in row["evaluations"]:
                    key = f"{case['task']}_t{case['teacher']}_o{e['observation']}"
                    if (
                        e["data_path"] != public[key]["path"]
                        or e["data_sha256"] != public[key]["sha256"]
                    ):
                        raise AssertionError("Oracle uses wrong public observations")
                    if (
                        e["train_rows"] != max(config["train_sizes"])
                        or e["endpoint_rows"] != config["endpoint_rows"]
                    ):
                        raise AssertionError("Oracle evaluation row count differs")
                    sidecar = read_json(public[key]["sidecar_path"])
                    if (
                        row["private_specification_sha256"]
                        != sidecar["private_specification_sha256"]
                    ):
                        raise AssertionError(
                            "Oracle private source differs from both public datasets"
                        )
                    bindings.append(
                        _bound(
                            _inside(root, e["predictions_path"]),
                            e["predictions_sha256"],
                        )
                    )
                actual.add(identity)
                oracles.append({"path": str(p), "sha256": item["sha256"], "case": case})
        if actual != expected_oracle:
            raise AssertionError("Oracle grid incomplete")
        bindings.append(
            _bound(
                root / "private_plane_diagnostics.json",
                seal["private_plane_diagnostics_sha256"],
            )
        )
        bindings.append(
            _bound(
                root / "private_plane_diagnostics.csv",
                seal["private_plane_diagnostics_csv_sha256"],
            )
        )
        if read_json(root / "private_plane_diagnostics.json") != diagnostics:
            raise AssertionError("Aggregate private diagnostics differ from packets")
    return {
        "root": str(root),
        "main_root": str(main),
        "config": config,
        "initialized": initialized,
        "complete": complete,
        "bindings": bindings,
        "learned": learned,
        "oracle": oracles,
        "diagnostics": diagnostics,
        "constructive_complete_sha256": AUDIT.sha256(root / "complete.json"),
        "main_gate_sha256": AUDIT.sha256(root / "main_completion_gate.json"),
    }


def _learned_worker(payload):
    index, item, verified, output = payload
    result = {"case": item["case"], "result_path": item["path"], "phase": "learned"}
    try:
        _bound(item["path"], item["sha256"])
        row = read_json(item["path"])
        case = row["case"]
        root = Path(verified["root"])
        result["replay"] = AUDIT.audit_constructed_state(
            row["state_path"],
            row["data_path"],
            train_n=case["train_n"],
            m=int(case["task"].removeprefix("radial_m")),
            radius=verified["config"]["radius"],
            intervals=case["intervals"],
            state_sha256=row["state_sha256"],
            data_sha256=row["data_sha256"],
            predictions_path=row["predictions_path"],
            predictions_sha256=row["predictions_sha256"],
            learner_source_path=root / "source/radial_width_constructive.py",
            task_source_path=root / "source/radial_width_tasks.py",
        )
        state = REPLAY.load_state(row["state_path"])
        init = state["metadata"]["initialization_receipt"]
        bank_sha = train_fingerprint(state["arrays"]["basis"], "dtype_shape_bytes")
        if (
            bank_sha != init["bank_sha256"]
            or bank_sha != init["direction_bank"]["estimator"]["bank_sha256"]
        ):
            raise AssertionError(
                "Learned native bank hash differs from its TRAIN estimator"
            )
        if (
            init["direction_bank"]["estimator"] != read_json(row["estimator_path"])
            or init["_radial_width"]["source_bindings"]
            != verified["initialized"]["source_bindings"]
        ):
            raise AssertionError(
                "Learned estimator/source receipt differs from native state"
            )
        if (
            result["replay"]["inventory"] != row["counted_inventory"]
            or row["stored_parameters"] != state["parameter_count"]
        ):
            raise AssertionError("Learned result/native parameter inventories differ")
        result["metric_differences"] = {
            k: _close(result["replay"]["metrics"][k], v, "learned_" + k, 3e-6, 5e-15)
            for k, v in row["metrics"].items()
        }
        result["status"] = "passed"
    except Exception as error:
        result.update(_failure(error))
    path = Path(output) / "learned" / f"{index:04d}.json"
    _write(path, result)
    return {
        "index": index,
        "status": result["status"],
        "path": str(path),
        "sha256": AUDIT.sha256(path),
        "case": item["case"],
    }


def _oracle_worker(payload):
    index, item, verified, output = payload
    result = {
        "case": item["case"],
        "result_path": item["path"],
        "phase": "privileged_oracle",
        "evaluations": [],
    }
    try:
        _bound(item["path"], item["sha256"])
        row = read_json(item["path"])
        case = row["case"]
        root = Path(verified["root"])
        for e in row["evaluations"]:
            evaluation = {"observation": e["observation"]}
            try:
                evaluation["replay"] = audit_oracle_state(
                    row["state_path"],
                    e["data_path"],
                    train_n=e["train_rows"],
                    m=int(case["task"].removeprefix("radial_m")),
                    radius=verified["config"]["radius"],
                    intervals=case["intervals"],
                    state_sha256=row["state_sha256"],
                    data_sha256=e["data_sha256"],
                    predictions_path=e["predictions_path"],
                    predictions_sha256=e["predictions_sha256"],
                    learner_source_path=root / "source/radial_width_constructive.py",
                    task_source_path=root / "source/radial_width_tasks.py",
                    specification_path=row["private_specification_path"],
                    specification_sha256=row["private_specification_sha256"],
                    main_gate_path=root / "main_completion_gate.json",
                    main_gate_sha256=verified["main_gate_sha256"],
                    learned_completion_path=root / "learned_complete.json",
                    learned_completion_sha256=verified["complete"][
                        "learned_complete_sha256"
                    ],
                    expected_learned_states=verified["config"][
                        "expected_learned_states"
                    ],
                )
                state = REPLAY.load_state(row["state_path"])
                if (
                    evaluation["replay"]["inventory"] != row["counted_inventory"]
                    or state["parameter_count"] != row["stored_parameters"]
                    or state["metadata"]["initialization_receipt"]["_radial_width"][
                        "source_bindings"
                    ]
                    != verified["initialized"]["source_bindings"]
                ):
                    raise AssertionError("Oracle native count/source inventory differs")
                evaluation["metric_differences"] = {
                    k: _close(
                        evaluation["replay"]["metrics"][k],
                        v,
                        "oracle_" + k,
                        3e-6,
                        5e-15,
                    )
                    for k, v in e["metrics"].items()
                }
                evaluation["status"] = "passed"
            except Exception as error:
                evaluation.update(_failure(error))
            result["evaluations"].append(evaluation)
        result["status"] = (
            "passed"
            if all(e["status"] == "passed" for e in result["evaluations"])
            else "failed"
        )
    except Exception as error:
        result.update(_failure(error))
    path = Path(output) / "oracle" / f"{index:04d}.json"
    _write(path, result)
    return {
        "index": index,
        "status": result["status"],
        "path": str(path),
        "sha256": AUDIT.sha256(path),
        "case": item["case"],
        "evaluations": len(result["evaluations"]),
        "passed_evaluations": sum(
            e["status"] == "passed" for e in result["evaluations"]
        ),
    }


def audit_plane_diagnostics(verified):
    config = verified["config"]
    root = Path(verified["root"])
    if not verified["initialized"]["include_oracle"]:
        return []
    oracle_barriers(
        root / "main_completion_gate.json",
        verified["main_gate_sha256"],
        root / "learned_complete.json",
        verified["complete"]["learned_complete_sha256"],
        config["expected_learned_states"],
    )
    expected = set(
        itertools.product(
            config["tasks"],
            config["teachers"],
            config["observations"],
            config["train_sizes"],
        )
    )
    actual = {
        tuple(row[k] for k in ("task", "teacher", "observation", "train_n"))
        for row in verified["diagnostics"]
    }
    if actual != expected or len(actual) != len(verified["diagnostics"]):
        raise AssertionError("Private plane-diagnostic grid differs")
    learned = {
        tuple(
            item["case"][k] for k in ("task", "teacher", "observation", "train_n")
        ): item
        for item in verified["learned"]
        if item["case"]["intervals"] == config["intervals"][0]
    }
    results = []
    for row in verified["diagnostics"]:
        record = {k: row[k] for k in ("task", "teacher", "observation", "train_n")}
        try:
            key = tuple(row[k] for k in ("task", "teacher", "observation", "train_n"))
            learned_row = read_json(learned[key]["path"])
            if (
                row["learned_state_path"] != learned_row["state_path"]
                or row["learned_state_sha256"] != learned_row["state_sha256"]
            ):
                raise AssertionError(
                    "Plane diagnostic did not use the declared first-S learned state"
                )
            _bound(row["learned_state_path"], row["learned_state_sha256"])
            private = (
                Path(verified["main_root"])
                / "private_teachers"
                / f"confirmation_{row['task']}_t{row['teacher']}.json"
            )
            _bound(private, row["private_specification_sha256"])
            bank = REPLAY.load_state(row["learned_state_path"])["arrays"]["basis"]
            p = np.asarray(read_json(private)["projectors"])
            errors = np.sum((bank @ bank.transpose(0, 2, 1) - p) ** 2, axis=(1, 2))
            record["per_block_squared_projector_error"] = errors.tolist()
            record["block_difference"] = _close(
                errors,
                row["per_block_squared_projector_frobenius_error"],
                "private_plane_errors",
                2e-12,
                2e-12,
            )
            record["mean_difference"] = _close(
                float(errors.mean()),
                row["mean_squared_projector_frobenius_error"],
                "private_plane_mean",
                2e-12,
                2e-12,
            )
            record["status"] = "passed"
        except Exception as error:
            record.update(_failure(error))
        results.append(record)
    return results


def run(root, output, *, workers=1):
    if (
        isinstance(workers, bool)
        or not isinstance(workers, int)
        or not 1 <= workers <= 64
    ):
        raise ValueError("One to64 explicitly allocated workers required")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        verified = verify_complete(root)
    except Exception as error:
        _write(output / "barrier_failure.json", _failure(error))
        raise
    _write(
        output / "protocol_receipt.json",
        {
            "source_bindings": [
                _bound(p)
                for p in (__file__, AUDIT.__file__, GENERIC.__file__, REPLAY.__file__)
            ],
            "completion_bindings": verified["bindings"],
            "workers": workers,
            "primary_outcomes_selected": False,
            "scalar_tolerances": {"rtol": 3e-6, "atol": 5e-15},
            "forward_tolerances": {"rtol": 3e-6, "atol": 5e-12},
        },
    )
    slim = {
        k: verified[k]
        for k in ("root", "config", "initialized", "complete", "main_gate_sha256")
    }
    learned = []
    oracle = []
    payloads = [
        ("learned", (i, item, slim, str(output)))
        for i, item in enumerate(verified["learned"])
    ] + [
        ("oracle", (i, item, slim, str(output)))
        for i, item in enumerate(verified["oracle"])
    ]

    def retain(kind, index, error):
        path = output / f"worker_failure_{kind}_{index:04d}.json"
        _write(path, _failure(error))
        return {
            "index": index,
            "status": "failed",
            "path": str(path),
            "sha256": AUDIT.sha256(path),
        }

    if workers == 1:
        for kind, payload in payloads:
            try:
                result = (_learned_worker if kind == "learned" else _oracle_worker)(
                    payload
                )
            except Exception as error:
                result = retain(kind, payload[0], error)
            (learned if kind == "learned" else oracle).append(result)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(
                    _learned_worker if kind == "learned" else _oracle_worker, payload
                ): (kind, payload[0])
                for kind, payload in payloads
            }
            for future in as_completed(futures):
                kind, index = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = retain(kind, index, error)
                (learned if kind == "learned" else oracle).append(result)
    try:
        planes = audit_plane_diagnostics(verified)
    except Exception as error:
        planes = [_failure(error)]
    _write(output / "plane_diagnostics.json", planes)
    config = verified["config"]
    summary = {
        "status": (
            "passed"
            if all(r["status"] == "passed" for r in learned + oracle + planes)
            else "failed"
        ),
        "constructive_root": str(Path(root).resolve()),
        "constructive_complete_sha256": verified["constructive_complete_sha256"],
        "expected_learned_states": config["expected_learned_states"],
        "expected_oracle_states": len(verified["oracle"]),
        "expected_oracle_evaluations": len(verified["oracle"])
        * len(config["observations"]),
        "learned_states": len(learned),
        "passed_learned_states": sum(r["status"] == "passed" for r in learned),
        "oracle_states": len(oracle),
        "passed_oracle_states": sum(r["status"] == "passed" for r in oracle),
        "oracle_evaluations": sum(r.get("evaluations", 0) for r in oracle),
        "passed_oracle_evaluations": sum(
            r.get("passed_evaluations", 0) for r in oracle
        ),
        "plane_diagnostics": len(planes),
        "passed_plane_diagnostics": sum(r["status"] == "passed" for r in planes),
        "learned_receipts": sorted(learned, key=lambda r: r["index"]),
        "oracle_receipts": sorted(oracle, key=lambda r: r["index"]),
        "source_bindings": [
            _bound(p)
            for p in (__file__, AUDIT.__file__, GENERIC.__file__, REPLAY.__file__)
        ],
        "elapsed_seconds": time.monotonic() - started,
        "workers": workers,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Every learned and privileged state retained separately. No production constructor/model/teacher imports, fitting, model selection or tolerance revisions. Privileged plane errors are post-completion diagnostics only.",
    }
    _write(output / "audit.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    result = run(args.root, args.output, workers=args.workers)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "learned_states",
                    "passed_learned_states",
                    "oracle_states",
                    "passed_oracle_states",
                    "oracle_evaluations",
                    "passed_oracle_evaluations",
                )
            }
        )
    )
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

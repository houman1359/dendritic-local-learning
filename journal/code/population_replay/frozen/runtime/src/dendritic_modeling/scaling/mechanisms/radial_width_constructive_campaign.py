"""Post-completion, source-bound known-profile radial learning campaign.

The learned phase never opens a private teacher specification. A separate oracle
phase may run only after all 640 learned states are sealed. No optimizer, model
selection, or hyperparameter search occurs here. Existing outcomes are immutable;
a failed invocation is preserved and requires a fresh output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import multiprocessing
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SOURCE_NAMES = (
    "radial_width_constructive_campaign.py",
    "radial_width_constructive.py",
    "radial_width_tasks.py",
)
MAIN_STAGES = ("development", "bridge", "confirmation")
CASE_FIELDS = (
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


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def load_construction():
    path = Path(__file__).with_name("radial_width_constructive.py").resolve()
    name = "radial_constructive_" + sha(path)[:20]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _inside(root, path):
    path = Path(path)
    path = path if path.is_absolute() else root / path
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Artifact escapes declared main root")
    return path


def _main_cases(config, stage, choices, recipes):
    indices = config[
        "confirmation_teachers" if stage == "confirmation" else "development_teachers"
    ]
    budgets = (
        config["calibration_p"]
        if stage == "development"
        else [config["bridge_p"]] if stage == "bridge" else config["primary_p"]
    )
    cases = []
    for task, geometry, family, teacher, observation, n in itertools.product(
        config["tasks"],
        config["geometries"],
        config["families"],
        indices,
        range(2 if stage == "confirmation" else 1),
        config["train_sizes"],
    ):
        key = f"{task}/{geometry}/{family}/{n}"
        selected = (
            [c["id"] for c in choices]
            if stage == "development"
            else [recipes[key]["choice"]]
        )
        for choice, ceiling in itertools.product(selected, budgets):
            cases.append(
                dict(
                    zip(
                        CASE_FIELDS,
                        (
                            stage,
                            task,
                            geometry,
                            family,
                            teacher,
                            observation,
                            n,
                            choice,
                            "plane8",
                            True,
                            ceiling,
                        ),
                        strict=True,
                    )
                )
            )
    return cases


def _verify_completed_inventory(main_root, config):
    """Independent generic inventory replay; no numerical outcome is selected."""
    root = Path(main_root).resolve()
    complete = read(root / "complete.json")
    required = {
        "status": "complete",
        "fits": config["expected_fits"],
        "expected_fits": config["expected_fits"],
        "failed_fits": 0,
        "fit_counts": config["fit_counts"],
    }
    if any(complete.get(k) != v for k, v in required.items()):
        raise ValueError("Main completion/count/failure barrier has not passed")
    initialized = read(root / "initialized.json")
    checked = {}

    def check(path, expected):
        path = _inside(root, path)
        key = str(path.resolve())
        if key not in checked:
            checked[key] = sha(path)
        if checked[key] != expected:
            raise ValueError(f"Main artifact hash changed: {path}")
        return path

    for name, value in initialized["bindings"].items():
        check(name, value)
    if "config.json" not in initialized["bindings"]:
        raise ValueError("Main configuration is not source-bound")
    selection = read(root / "selection_complete.json")
    check("selected_recipes.json", selection["selected_sha256"])
    check("development_complete.json", selection["development_barrier_sha256"])
    forecast = read(root / "forecast_complete.json")
    check("frozen_forecasts.json", forecast["forecast_sha256"])
    check("bridge_complete.json", forecast["bridge_barrier_sha256"])
    check(
        "selected_recipes.json", read(root / "frozen_forecasts.json")["selected_sha256"]
    )
    choices, recipes = read(root / "choices.json"), read(root / "selected_recipes.json")
    initialized_sha = sha(root / "initialized.json")
    result_count = 0
    for stage in MAIN_STAGES:
        barrier = read(root / f"{stage}_complete.json")
        cases = _main_cases(config, stage, choices, recipes)
        expected_paths = {
            "cases/" + "/".join(str(c[k]) for k in CASE_FIELDS) + "/result.json": c
            for c in cases
        }
        if (
            len(cases) != config["fit_counts"][stage]
            or len(expected_paths) != len(cases)
            or barrier.get("status") != "complete"
            or barrier.get("fits") != len(cases)
            or set(barrier["results"]) != set(expected_paths)
        ):
            raise ValueError("Main stage inventory differs from frozen grid")
        for name, case in expected_paths.items():
            row = read(check(name, barrier["results"][name]))
            if (
                row.get("status") != "complete"
                or row["case"] != case
                or row["initialized_sha256"] != initialized_sha
            ):
                raise ValueError("Main contains failed or misbound outcome")
            release = "confirmation" if stage == "confirmation" else "development"
            expected_data = (
                root
                / "data"
                / release
                / f"{case['task']}_t{case['teacher']}_o{case['observation']}.npz"
            )
            if _inside(root, row["data_path"]).resolve() != expected_data.resolve():
                raise ValueError("Main result uses the wrong dataset")
            for stem in ("data", "fit", "state", "predictions"):
                check(row[stem + "_path"], row[stem + "_sha256"])
            result_count += 1
    if result_count != config["expected_fits"]:
        raise ValueError("Main aggregate inventory differs")
    return {
        "status": "passed",
        "utc": now(),
        "verified_results": result_count,
        "failed_results": 0,
        "verified_artifacts": len(checked),
        "main_root": str(root),
        "barriers": {
            name: sha(root / name)
            for name in (
                "complete.json",
                "initialized.json",
                "config.json",
                "selected_recipes.json",
                "frozen_forecasts.json",
                "selection_complete.json",
                "forecast_complete.json",
                *(f"{stage}_complete.json" for stage in MAIN_STAGES),
            )
        },
        "scope": "Hash and status inventory; main losses are not used by this learner.",
    }


def verify_main_completion(main_root, *, smoke=False):
    """Production gate is deliberately fixed to the approved 7,872-fit study."""
    root = Path(main_root).resolve()
    # Read the completion marker before any source, dataset or private artifact.
    complete = read(root / "complete.json")
    if (
        complete.get("status"),
        complete.get("expected_fits"),
        complete.get("failed_fits"),
    ) != ("complete", 544 if smoke else 7872, 0):
        raise ValueError("The approved main study is not complete")
    config = read(root / "config.json")
    expected = {
        "schema": "radial_width_v1",
        "expected_fits": 7872,
        "tasks": ["radial_m2", "radial_m4"],
        "blocks": 4,
        "radius": 0.5,
        "confirmation_teachers": list(range(100, 108)),
        "constructive_n": [512, 2048, 8192, 32768],
        "constructive_intervals": [4, 8, 16, 32, 64],
        "max_train_n": 32768,
        "test_n": 32768,
    }
    if smoke:
        expected.update(
            expected_fits=544,
            tasks=["radial_m2"],
            confirmation_teachers=[910, 911],
            constructive_n=[32, 64],
            max_train_n=64,
            test_n=64,
        )
    if any(config.get(k) != v for k, v in expected.items()):
        raise ValueError(
            "Main study differs from the prospective constructive protocol"
        )
    receipt = _verify_completed_inventory(root, config)
    receipt["mode"] = "SMOKE_ONLY" if smoke else "production"
    return receipt


def configuration(*, smoke=False):
    config = {
        "schema": "radial_width_constructive_campaign_v1",
        "tasks": ["radial_m2", "radial_m4"],
        "teachers": list(range(100, 108)),
        "observations": [0, 1],
        "train_sizes": [512, 2048, 8192, 32768],
        "intervals": [4, 8, 16, 32, 64],
        "blocks": 4,
        "radius": 0.5,
        "expected_learned_states": 640,
        "expected_oracle_states": 80,
        "expected_oracle_evaluations": 160,
        "learner": "Raw scalar TRAIN-label top-algebraic Stein plane; known radial profile ReLU interpolation",
        "label_queries_added": 0,
        "search_candidates": 0,
        "normalization": "Exact known uniform-ball moments; no fitted or held-out centering",
        "evaluation": "Raw scalar MSE; no clipping; TEST never chooses N, S, plane or state",
        "teacher_replication_scope": "Eight independently rotated planes per fixed power; shapes fixed within power",
        "oracle_scope": "Privileged true-plane plus known-profile approximation reference; never a learned result",
        "count_formula": "P=4*K*(m+1)*S+7*K+1; all raw stored floating slots counted",
        "mode": "production",
        "endpoint_rows": 32768,
    }
    if smoke:
        config.update(
            tasks=["radial_m2"],
            teachers=[910, 911],
            train_sizes=[32, 64],
            expected_learned_states=40,
            expected_oracle_states=10,
            expected_oracle_evaluations=20,
            mode="SMOKE_ONLY",
            endpoint_rows=64,
        )
    return config


def learned_packets(config):
    return [
        {"task": t, "teacher": i, "observation": o, "train_n": n}
        for t, i, o, n in itertools.product(
            config["tasks"],
            config["teachers"],
            config["observations"],
            config["train_sizes"],
        )
    ]


def oracle_packets(config):
    return [
        {"task": t, "teacher": i}
        for t, i in itertools.product(config["tasks"], config["teachers"])
    ]


def _data_key(task, teacher, observation):
    return f"{task}_t{teacher}_o{observation}"


def _public_inventory(main_root, config):
    result = {}
    main = read(main_root / "config.json")
    for task, teacher, observation in itertools.product(
        config["tasks"], config["teachers"], config["observations"]
    ):
        key = _data_key(task, teacher, observation)
        path = main_root / "data/confirmation" / f"{key}.npz"
        sidecar = path.with_suffix(".json")
        record = read(sidecar)
        identity = {
            "release": "confirmation",
            "endpoint": "test",
            "task": task,
            "teacher": teacher,
            "observation": observation,
            "max_train_n": main["max_train_n"],
            "forecast_sha256": sha(main_root / "frozen_forecasts.json"),
        }
        if (
            any(record.get(k) != v for k, v in identity.items())
            or sha(path) != record["sha256"]
        ):
            raise ValueError("Public confirmation dataset identity/hash differs")
        result[key] = {
            "path": str(path),
            "sha256": record["sha256"],
            "sidecar_path": str(sidecar),
            "sidecar_sha256": sha(sidecar),
        }
    return result


def initialize(main_root, output_dir, *, include_oracle=True, smoke=False):
    main_root, output_dir = Path(main_root).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            "Constructive output root already exists; preserve it and use a fresh attempt"
        )
    barrier = verify_main_completion(main_root, smoke=smoke)
    config = configuration(smoke=smoke)
    sources = {name: sha(Path(__file__).with_name(name)) for name in SOURCE_NAMES}
    main_bindings = read(main_root / "initialized.json")["bindings"]
    for name, value in sources.items():
        # The pre-existing smoke generic snapshot predates this runner. Its
        # mathematical dependencies must match; this new runner is frozen below.
        if smoke and name == "radial_width_constructive_campaign.py":
            continue
        if main_bindings.get("source/" + name) != value:
            raise ValueError(
                f"Constructive source is not in the frozen main snapshot: {name}"
            )
    public = _public_inventory(
        main_root, config
    )  # Hash bytes only; no array labels loaded.
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "source").mkdir()
    for name in sources:
        (output_dir / "source" / name).write_bytes(
            Path(__file__).with_name(name).read_bytes()
        )
    write(output_dir / "config.json", config)
    write(output_dir / "main_completion_gate.json", barrier)
    write(output_dir / "public_data_inventory.json", public)
    write(
        output_dir / "initialized.json",
        {
            "utc": now(),
            "main_root": str(main_root),
            "include_oracle": bool(include_oracle),
            "mode": config["mode"],
            "source_bindings": sources,
            "bindings": {
                name: sha(output_dir / name)
                for name in (
                    "config.json",
                    "main_completion_gate.json",
                    "public_data_inventory.json",
                    *("source/" + n for n in sources),
                )
            },
            "scope": "Frozen before loading constructive TRAIN or TEST arrays; private specifications unopened",
        },
    )
    return output_dir


def verify_frozen(output_dir):
    root = Path(output_dir).resolve()
    receipt = read(root / "initialized.json")
    for name, value in receipt["bindings"].items():
        if sha(root / name) != value:
            raise ValueError("Constructive frozen artifact changed")
    for name, value in receipt["source_bindings"].items():
        if sha(Path(__file__).with_name(name)) != value:
            raise ValueError("Executing source differs from frozen constructive source")
    main = Path(receipt["main_root"])
    gate = read(root / "main_completion_gate.json")
    for name, value in gate["barriers"].items():
        if sha(main / name) != value:
            raise ValueError("Main completion lineage changed")
    return receipt


def _data_entry(root, task, teacher, observation):
    entry = read(root / "public_data_inventory.json")[
        _data_key(task, teacher, observation)
    ]
    if (
        sha(entry["path"]) != entry["sha256"]
        or sha(entry["sidecar_path"]) != entry["sidecar_sha256"]
    ):
        raise ValueError("Public dataset changed after constructive freeze")
    return entry


def _train_arrays(entry, n, config):
    with np.load(entry["path"], allow_pickle=False) as archive:
        x, y = archive["x_train"][:n].copy(), archive["y_train"][:n].copy()
    if (
        x.shape != (n, config["blocks"], 3)
        or y.shape != (n,)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Invalid finite TRAIN prefix")
    return x, y


def _endpoint_arrays(entry, config):
    with np.load(entry["path"], allow_pickle=False) as archive:
        x, y = archive["x_endpoint"].copy(), archive["y_endpoint"].copy()
    if (
        x.shape != (config["endpoint_rows"], config["blocks"], 3)
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("Invalid finite TEST arrays")
    return x, y


def _evaluate(model, x, y, endpoint, endpoint_y, path):
    started = time.monotonic()
    train, test = model.predict(x), model.predict(endpoint)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise FloatingPointError("Nonfinite construction predictions")
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, train=train, endpoint=test)
    return {
        "metrics": {
            "train_mse": float(np.mean((train - y) ** 2)),
            "endpoint_mse": float(np.mean((test - endpoint_y) ** 2)),
        },
        "train_rows": len(y),
        "endpoint_rows": len(endpoint_y),
        "predictions_path": str(path),
        "predictions_sha256": sha(path),
        "evaluation_seconds": time.monotonic() - started,
    }


def run_learned_packet(output_dir, packet):
    root = Path(output_dir).resolve()
    frozen = verify_frozen(root)
    config = read(root / "config.json")
    if packet not in learned_packets(config):
        raise ValueError("Learned packet outside frozen grid")
    key = _data_key(packet["task"], packet["teacher"], packet["observation"])
    directory = root / "learned" / key / f"N{packet['train_n']}"
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "packet.json", packet)
    started = time.monotonic()
    record = {
        "packet": packet,
        "phase": "learned",
        "utc": now(),
        "states": [],
        "initialized_sha256": sha(root / "initialized.json"),
    }
    try:
        entry = _data_entry(
            root, packet["task"], packet["teacher"], packet["observation"]
        )
        x, y = _train_arrays(entry, packet["train_n"], config)
        construction = load_construction()
        estimation_started = time.monotonic()
        bank, estimator = construction.estimate_planes(x, y, radius=config["radius"])
        record["estimation_seconds"] = time.monotonic() - estimation_started
        write(directory / "estimator.json", estimator)
        m = int(packet["task"].removeprefix("radial_m"))
        # All states fixed before any TEST arrays are materialized.
        models = []
        for intervals in config["intervals"]:
            state_dir = directory / f"S{intervals}"
            state_dir.mkdir()
            stamp = time.monotonic()
            model = construction.construct_model(
                bank,
                m,
                radius=config["radius"],
                intervals=intervals,
                estimator_receipt=estimator,
                source_bindings=frozen["source_bindings"],
            )
            state_path = state_dir / "state.npz"
            construction.save_state(model, state_path)
            row = {
                "case": dict(packet, intervals=intervals),
                "phase": "learned",
                "scope": "Label-only estimated plane; known public radial profile; no optimization",
                "state_path": str(state_path),
                "state_sha256": sha(state_path),
                "data_path": entry["path"],
                "data_sha256": entry["sha256"],
                "estimator_path": str(directory / "estimator.json"),
                "estimator_sha256": sha(directory / "estimator.json"),
                "stored_parameters": model.parameter_count,
                "counted_inventory": model.metadata["initialization_receipt"][
                    "_radial_width"
                ],
                "construction_seconds": time.monotonic() - stamp,
            }
            models.append((model, state_dir, row))
        write(
            directory / "construction_complete.json",
            {
                "utc": now(),
                "status": "complete",
                "states": {r["state_path"]: r["state_sha256"] for _, _, r in models},
                "scope": "All S states sealed before TEST array access",
            },
        )
        endpoint, endpoint_y = _endpoint_arrays(entry, config)
        for model, state_dir, row in models:
            row.update(
                _evaluate(
                    model, x, y, endpoint, endpoint_y, state_dir / "predictions.npz"
                )
            )
            row.update(
                status="complete",
                initialized_sha256=record["initialized_sha256"],
                construction_barrier_sha256=sha(
                    directory / "construction_complete.json"
                ),
            )
            write(state_dir / "result.json", row)
            record["states"].append(
                {
                    "path": str(state_dir / "result.json"),
                    "sha256": sha(state_dir / "result.json"),
                }
            )
        record["status"] = "complete"
    except Exception:
        record.update(status="failed", traceback=traceback.format_exc())
    record["elapsed_seconds"] = time.monotonic() - started
    write(directory / "packet_result.json", record)
    return record


def _learned_rows(root, config):
    rows, packets = [], {}
    for packet in learned_packets(config):
        key = _data_key(packet["task"], packet["teacher"], packet["observation"])
        path = root / "learned" / key / f"N{packet['train_n']}" / "packet_result.json"
        record = read(path)
        if (
            record["status"] != "complete"
            or record["packet"] != packet
            or record["initialized_sha256"] != sha(root / "initialized.json")
        ):
            raise ValueError("All frozen learned packets must succeed")
        if len(record["states"]) != len(config["intervals"]):
            raise ValueError("Learned state count differs")
        for item, intervals in zip(record["states"], config["intervals"], strict=True):
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Learned result changed")
            row = read(item["path"])
            if row["status"] != "complete" or row["case"] != dict(
                packet, intervals=intervals
            ):
                raise ValueError("Learned case differs")
            for stem in ("state", "predictions", "data", "estimator"):
                if sha(row[stem + "_path"]) != row[stem + "_sha256"]:
                    raise ValueError("Learned artifact changed")
            rows.append(row)
        packets[str(path)] = sha(path)
    if len(rows) != config["expected_learned_states"]:
        raise ValueError("Learned total differs")
    return rows, packets


def seal_learned(root):
    root = Path(root).resolve()
    verify_frozen(root)
    config = read(root / "config.json")
    rows, packets = _learned_rows(root, config)
    receipt = {
        "utc": now(),
        "status": "complete",
        "states": len(rows),
        "failed_states": 0,
        "packets": packets,
        "initialized_sha256": sha(root / "initialized.json"),
        "scope": "All learned states, predictions and scores sealed before private teacher access",
    }
    write(root / "learned_complete.json", receipt)
    return receipt


def verify_learned_completion(root):
    root = Path(root).resolve()
    frozen = verify_frozen(root)
    receipt = read(root / "learned_complete.json")
    config = read(root / "config.json")
    if (
        receipt.get("status"),
        receipt.get("states"),
        receipt.get("failed_states"),
        receipt.get("initialized_sha256"),
    ) != (
        "complete",
        config["expected_learned_states"],
        0,
        sha(root / "initialized.json"),
    ):
        raise ValueError("Learned completion barrier has not passed")
    for name, value in receipt["packets"].items():
        if sha(name) != value:
            raise ValueError("Sealed learned packet changed")
    return frozen


def _oracle_bank(spec, m, config):
    if (
        spec.get("power") != m
        or spec.get("blocks") != config["blocks"]
        or spec.get("radius") != config["radius"]
    ):
        raise ValueError("Private oracle task identity differs")
    projector = np.asarray(spec["projectors"], dtype=np.float64)
    if projector.shape != (config["blocks"], 3, 3) or not np.isfinite(projector).all():
        raise ValueError("Invalid private projector shape")
    if (
        not np.allclose(projector, projector.swapaxes(-1, -2), atol=2e-12, rtol=0)
        or not np.allclose(projector @ projector, projector, atol=2e-12, rtol=0)
        or not np.allclose(np.trace(projector, axis1=1, axis2=2), 2, atol=2e-12, rtol=0)
    ):
        raise ValueError("Private projector is not an orthogonal rank-two projector")
    digest = hashlib.sha256()
    digest.update(str(projector.dtype).encode())
    digest.update(str(projector.shape).encode())
    digest.update(np.ascontiguousarray(projector).tobytes())
    if digest.hexdigest() != spec["projectors_sha256"]:
        raise ValueError("Private projector array hash differs")
    _, vectors = np.linalg.eigh(projector)
    bank = vectors[..., -2:][:, :, ::-1].copy()
    for block in range(len(bank)):
        for j in range(2):
            if bank[block, np.argmax(np.abs(bank[block, :, j])), j] < 0:
                bank[block, :, j] *= -1
    return bank


def run_oracle_packet(output_dir, packet):
    root = Path(output_dir).resolve()
    frozen = verify_learned_completion(root)
    if not frozen["include_oracle"]:
        raise ValueError("Oracle phase was not prospectively enabled")
    config = read(root / "config.json")
    if packet not in oracle_packets(config):
        raise ValueError("Oracle packet outside frozen grid")
    directory = root / "oracle" / f"{packet['task']}_t{packet['teacher']}"
    directory.mkdir(parents=True, exist_ok=False)
    write(directory / "packet.json", packet)
    record = {
        "packet": packet,
        "phase": "privileged_oracle",
        "utc": now(),
        "states": [],
        "initialized_sha256": sha(root / "initialized.json"),
        "learned_complete_sha256": sha(root / "learned_complete.json"),
    }
    started = time.monotonic()
    try:
        entries = [
            _data_entry(root, packet["task"], packet["teacher"], o)
            for o in config["observations"]
        ]
        private = (
            Path(frozen["main_root"])
            / "private_teachers"
            / f"confirmation_{packet['task']}_t{packet['teacher']}.json"
        )
        private_sha = sha(private)
        if any(
            read(e["sidecar_path"])["private_specification_sha256"] != private_sha
            for e in entries
        ):
            raise ValueError(
                "Oracle specification is not bound to both public observations"
            )
        # First private read occurs behind both completion barriers above.
        spec = read(private)
        m = int(packet["task"].removeprefix("radial_m"))
        bank = _oracle_bank(spec, m, config)
        construction = load_construction()
        plane_diagnostics = []
        true_projector = np.asarray(spec["projectors"], dtype=np.float64)
        for observation, n in itertools.product(
            config["observations"], config["train_sizes"]
        ):
            key = _data_key(packet["task"], packet["teacher"], observation)
            learned_dir = (
                root / "learned" / key / f"N{n}" / f"S{config['intervals'][0]}"
            )
            learned_row = read(learned_dir / "result.json")
            if sha(learned_row["state_path"]) != learned_row["state_sha256"]:
                raise ValueError("Learned diagnostic state changed")
            with np.load(learned_row["state_path"], allow_pickle=False) as archive:
                estimated_bank = archive["basis"].copy()
            estimated_projector = estimated_bank @ estimated_bank.swapaxes(-1, -2)
            squared_errors = np.sum(
                (estimated_projector - true_projector) ** 2, axis=(1, 2)
            )
            plane_diagnostics.append(
                {
                    "task": packet["task"],
                    "teacher": packet["teacher"],
                    "observation": observation,
                    "train_n": n,
                    "per_block_squared_projector_frobenius_error": squared_errors.tolist(),
                    "mean_squared_projector_frobenius_error": float(
                        np.mean(squared_errors)
                    ),
                    "learned_state_path": learned_row["state_path"],
                    "learned_state_sha256": learned_row["state_sha256"],
                    "private_specification_sha256": private_sha,
                    "scope": "Private diagnostic after all learned states sealed; never selection input",
                }
            )
        write(directory / "plane_diagnostics.json", plane_diagnostics)
        record.update(
            plane_diagnostics_path=str(directory / "plane_diagnostics.json"),
            plane_diagnostics_sha256=sha(directory / "plane_diagnostics.json"),
        )
        for intervals in config["intervals"]:
            state_dir = directory / f"S{intervals}"
            state_dir.mkdir()
            stamp = time.monotonic()
            model = construction.construct_model(
                bank,
                m,
                radius=config["radius"],
                intervals=intervals,
                source_bindings=frozen["source_bindings"],
            )
            model.metadata["initialization_receipt"].update(
                {
                    "bank_provenance": "PRIVILEGED true private projector; never a learned plane",
                    "privileged_reference": {
                        "private_specification_path": str(private),
                        "private_specification_sha256": private_sha,
                        "main_completion_gate_sha256": sha(
                            root / "main_completion_gate.json"
                        ),
                        "learned_complete_sha256": record["learned_complete_sha256"],
                        "known_profile": True,
                        "learner_observations": 0,
                    },
                }
            )
            state_path = state_dir / "state.npz"
            construction.save_state(model, state_path)
            row = {
                "case": dict(packet, intervals=intervals),
                "phase": "privileged_oracle",
                "scope": "Private true-plane known-profile approximation reference; not learning",
                "state_path": str(state_path),
                "state_sha256": sha(state_path),
                "private_specification_path": str(private),
                "private_specification_sha256": private_sha,
                "stored_parameters": model.parameter_count,
                "counted_inventory": model.metadata["initialization_receipt"][
                    "_radial_width"
                ],
                "construction_seconds": time.monotonic() - stamp,
                "evaluations": [],
            }
            for observation, entry in zip(config["observations"], entries, strict=True):
                x, y = _train_arrays(entry, max(config["train_sizes"]), config)
                endpoint, endpoint_y = _endpoint_arrays(entry, config)
                evaluation = _evaluate(
                    model,
                    x,
                    y,
                    endpoint,
                    endpoint_y,
                    state_dir / f"predictions_o{observation}.npz",
                )
                evaluation.update(
                    observation=observation,
                    data_path=entry["path"],
                    data_sha256=entry["sha256"],
                )
                row["evaluations"].append(evaluation)
            row.update(
                status="complete",
                initialized_sha256=record["initialized_sha256"],
                learned_complete_sha256=record["learned_complete_sha256"],
            )
            write(state_dir / "result.json", row)
            record["states"].append(
                {
                    "path": str(state_dir / "result.json"),
                    "sha256": sha(state_dir / "result.json"),
                }
            )
        record["status"] = "complete"
    except Exception:
        record.update(status="failed", traceback=traceback.format_exc())
    record["elapsed_seconds"] = time.monotonic() - started
    write(directory / "packet_result.json", record)
    return record


def _execute(function, root, packets, workers):
    if workers == 1:
        records = [function(str(root), packet) for packet in packets]
    else:
        records = []
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(function, str(root), p) for p in packets]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                print(
                    json.dumps(
                        {
                            "phase": record["phase"],
                            "completed_packets": len(records),
                            "total_packets": len(packets),
                            "status": record["status"],
                        }
                    ),
                    flush=True,
                )
    if any(record["status"] != "complete" for record in records):
        raise ValueError("Constructive failures preserved; campaign cannot be sealed")
    return records


def seal_oracle(root):
    root = Path(root).resolve()
    verify_learned_completion(root)
    config = read(root / "config.json")
    packets, count, evaluations, diagnostics = {}, 0, 0, []
    for packet in oracle_packets(config):
        path = (
            root
            / "oracle"
            / f"{packet['task']}_t{packet['teacher']}"
            / "packet_result.json"
        )
        record = read(path)
        if record["status"] != "complete" or record["packet"] != packet:
            raise ValueError("All oracle packets must succeed")
        if sha(record["plane_diagnostics_path"]) != record["plane_diagnostics_sha256"]:
            raise ValueError("Private plane diagnostic changed")
        packet_diagnostics = read(record["plane_diagnostics_path"])
        expected_diagnostic_cases = list(
            itertools.product(config["observations"], config["train_sizes"])
        )
        if [
            (r["observation"], r["train_n"]) for r in packet_diagnostics
        ] != expected_diagnostic_cases:
            raise ValueError("Private plane diagnostic inventory differs")
        diagnostics.extend(packet_diagnostics)
        if len(record["states"]) != len(config["intervals"]):
            raise ValueError("Oracle packet count differs")
        for item, intervals in zip(record["states"], config["intervals"], strict=True):
            if sha(item["path"]) != item["sha256"]:
                raise ValueError("Oracle result changed")
            row = read(item["path"])
            if row["status"] != "complete" or row["case"] != dict(
                packet, intervals=intervals
            ):
                raise ValueError("Oracle state identity differs")
            for stem in ("state", "private_specification"):
                if sha(row[stem + "_path"]) != row[stem + "_sha256"]:
                    raise ValueError("Oracle source/state changed")
            if [e["observation"] for e in row["evaluations"]] != config["observations"]:
                raise ValueError("Oracle observation count differs")
            for evaluation in row["evaluations"]:
                for stem in ("predictions", "data"):
                    if sha(evaluation[stem + "_path"]) != evaluation[stem + "_sha256"]:
                        raise ValueError("Oracle evaluation changed")
                evaluations += 1
            count += 1
        packets[str(path)] = sha(path)
    if (
        count != config["expected_oracle_states"]
        or evaluations != config["expected_oracle_evaluations"]
    ):
        raise ValueError("Oracle aggregate inventory differs")
    write(root / "private_plane_diagnostics.json", diagnostics)
    with (root / "private_plane_diagnostics.csv").open("x", newline="") as handle:
        fields = [
            "task",
            "teacher",
            "observation",
            "train_n",
            "mean_squared_projector_frobenius_error",
        ]
        fields += [
            f"block_{i}_squared_projector_frobenius_error"
            for i in range(config["blocks"])
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in diagnostics:
            values = {key: row[key] for key in fields[:5]}
            values.update(
                {
                    fields[5 + i]: error
                    for i, error in enumerate(
                        row["per_block_squared_projector_frobenius_error"]
                    )
                }
            )
            writer.writerow(values)
    result = {
        "utc": now(),
        "status": "complete",
        "states": count,
        "evaluations": evaluations,
        "failed_states": 0,
        "private_plane_diagnostics": len(diagnostics),
        "private_plane_diagnostics_sha256": sha(
            root / "private_plane_diagnostics.json"
        ),
        "private_plane_diagnostics_csv_sha256": sha(
            root / "private_plane_diagnostics.csv"
        ),
        "packets": packets,
        "learned_complete_sha256": sha(root / "learned_complete.json"),
        "scope": "Privileged approximation reference; no training or inference claim",
    }
    write(root / "oracle_complete.json", result)
    return result


def run_campaign(main_root, output_dir, *, workers=1, include_oracle=True, smoke=False):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("Positive integer workers required")
    root = initialize(main_root, output_dir, include_oracle=include_oracle, smoke=smoke)
    started = time.monotonic()
    config = read(root / "config.json")
    try:
        _execute(run_learned_packet, root, learned_packets(config), workers)
        learned = seal_learned(root)
        oracle = None
        if include_oracle:
            _execute(run_oracle_packet, root, oracle_packets(config), workers)
            oracle = seal_oracle(root)
        result = {
            "utc": now(),
            "status": "complete",
            "mode": config["mode"],
            "learned_states": learned["states"],
            "oracle_states": oracle["states"] if oracle else 0,
            "failed_states": 0,
            "workers": workers,
            "elapsed_seconds": time.monotonic() - started,
            "learned_complete_sha256": sha(root / "learned_complete.json"),
            "oracle_complete_sha256": (
                sha(root / "oracle_complete.json") if oracle else None
            ),
            "initialized_sha256": sha(root / "initialized.json"),
        }
        write(root / "complete.json", result)
        return result
    except Exception:
        write(
            root / "failed.json",
            {
                "utc": now(),
                "status": "failed",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only the separate completed 544-fit pipeline smoke; never production",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_campaign(
                args.main_root,
                args.output_dir,
                workers=args.workers,
                include_oracle=not args.no_oracle,
                smoke=args.smoke,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

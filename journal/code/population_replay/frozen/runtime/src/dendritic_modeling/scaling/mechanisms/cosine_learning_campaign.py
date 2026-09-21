"""Frozen three-cosine learning bridge with a cost-gated development campaign.

The centered TRAIN moment is the existing signed-centered recipe. Raw Hermite,
random, and explicitly private-plane initializations are diagnostic rank2 arms.
Independent observation evaluation is released only after every declared study
model is complete; its teachers remain development teachers, not confirmation
response functions. All parameter counts, initializations and outcomes remain.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import re
import resource
import shutil
import socket
import stat
import tempfile
import time
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from multiprocessing.connection import wait as wait_for_processes
from pathlib import Path

import numpy as np
import torch

from . import cosine_family_tasks as family_tasks
from . import cosine_learning_tasks as tasks
from . import radial_width_learning as width
from . import rank_learning as base
from . import rank_orthogonal_v2 as optimizer

SOURCE_NAMES = (
    "cosine_learning_campaign.py",
    "cosine_learning_tasks.py",
    "cosine_family_tasks.py",
    "radial_width_learning.py",
    "rank_learning.py",
    "rank_orthogonal_v2.py",
)
QUALIFIED_SOURCE_HASHES = {
    "radial_width_learning.py": "949349d18b43143e77a6176455a33674ef00dd7aa93abf6f4c3806a85076f885",
    "rank_learning.py": "9333999f69e656ddb4b61cb28ca71c52905cfe6a63fa0c73fa21858620109187",
    "rank_orthogonal_v2.py": "33208881aec0b19e89de2366026ce2cee6552dc8c4e470eece999344a7e3d57f",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def _temporary(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name("." + path.name + ".pending_" + uuid.uuid4().hex)


def write(path, value):
    """Publish complete JSON exclusively; interrupted temporary files survive."""
    path = Path(path)
    temporary = _temporary(path)
    with temporary.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


def _save_arrays(path, arrays):
    path = Path(path)
    temporary = _temporary(path)
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.link(temporary, path)
    temporary.unlink()


@contextmanager
def exclusive_lease(path):
    """Acquire an NFS-compatible exclusive directory lease without advisory locks.

    Atomic mkdir either succeeds or rejects an existing lease immediately. Hard
    kills can leave stale leases; operators must verify the recorded host/PID and
    Slurm owner before manual removal. There is no automatic stale-lease cleanup.
    Python unwind releases only a lease whose unique owner token still matches.
    """
    path = Path(path)
    try:
        path.mkdir()
    except FileExistsError as error:
        raise RuntimeError(
            f"Existing lease at {path}. Verify its owner.json host/PID and Slurm "
            "job before manual cleanup; no automatic stale-lease deletion."
        ) from error
    owner = {
        "schema": "cosine_campaign_directory_lease_v1",
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "utc": now(),
        "token": uuid.uuid4().hex,
    }
    owner_path = path / "owner.json"
    # A failure before this atomic owner publication leaves the directory in
    # place: without a bound owner token, automatic removal is not permitted.
    write(owner_path, owner)
    try:
        yield owner
    finally:
        try:
            current = read(owner_path)
        except (OSError, ValueError) as error:
            raise RuntimeError(
                f"Lease owner receipt unavailable at {path}; lease retained "
                "for manual owner verification."
            ) from error
        if current.get("token") != owner["token"]:
            raise RuntimeError(
                f"Lease ownership changed at {path}; another owner's lease "
                "will not be removed."
            )
        if {entry.name for entry in path.iterdir()} != {"owner.json"}:
            raise RuntimeError(
                f"Unexpected files in lease at {path}; lease retained for "
                "manual owner verification."
            )
        owner_path.unlink()
        path.rmdir()


def _integer(value, name, minimum=1):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        valid = int(value) == value and value >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def configuration():
    return {
        "schema": "three_cosine_learning_development_v1",
        "dimensions": [3, 32],
        "teachers": [901, 902],
        "conditions": [
            {"geometry": geometry, "method": "centered_second_moment"}
            for geometry in ("width2", "width3", "rank2", "full")
        ]
        + [
            {"geometry": "rank2", "method": method}
            for method in ("signed_second_hermite", "random", "oracle_private_plane")
        ],
        "teacher_seed_base": 2026161001,
        "observation_seed_base": 2026162001,
        "model_seed_base": 2026163001,
        "ceilings": [274, 546, 1090],
        "train_n": 8192,
        "validation_n": 4096,
        "evaluation_n": 65536,
        "steps": 600,
        "restart_every": 50,
        "checkpoints": [150, 300, 600],
        "coverage_lines": 8,
        "ridge": 1e-3,
        "workers": 8,
        "memory_gib": 64,
        "maximum_parameter_gap": 0.01,
        "profile": {"teacher": 900, "ceiling": 1090, "steps": 50},
        "profile_gate": {
            "maximum_case_seconds": 1200,
            "maximum_case_rss_gib": 5,
            "maximum_projected_case_hours": 32,
            "runtime_margin": 1.5,
            "memory_margin": 1.4,
            "coordinator_memory_gib": 4,
        },
        "audit": {"forward_rtol": 3e-6, "forward_atol": 5e-12},
        "scope": (
            "Development teacher orientations only, fixed three-cosine response. "
            "Primary centered-moment controls include width2, width3, rank2 and full. "
            "Raw-moment, random and private true-plane initializations are separate "
            "rank2 diagnostics. Every fit uses raw inputs and scalar TRAIN labels; "
            "only the oracle diagnostic receives private subspace information. "
            "All final study models receive independent-observation evaluation only "
            "after complete source/model binding, with no validation or TEST selection. "
            "This does not release fresh response confirmation or establish an exponent."
        ),
    }


def _validate_config(config):
    family_tasks.validate_campaign(config)
    tau = config.get("projection_rms_floor", 0.0)
    if (
        isinstance(tau, bool)
        or not isinstance(tau, (int, float))
        or not math.isfinite(tau)
        or tau < 0
    ):
        raise ValueError("Nonnegative finite projection_rms_floor required")
    if "bank_frame_rotation" in config:
        rotation = config["bank_frame_rotation"]
        if not isinstance(rotation, dict) or set(rotation) != {"seed_base"}:
            raise ValueError("bank_frame_rotation requires only a seed_base")
        tasks._integer(rotation["seed_base"], "Bank frame rotation seed base")
    for key in (
        "train_n",
        "validation_n",
        "evaluation_n",
        "steps",
        "restart_every",
        "coverage_lines",
        "workers",
    ):
        _integer(config[key], key, 4 if key == "train_n" else 1)
    if config["steps"] % config["restart_every"]:
        raise ValueError("Study steps must consist of whole declared restart blocks")
    if not math.isfinite(config["ridge"]) or config["ridge"] <= 0:
        raise ValueError("Finite positive ridge required")
    if not math.isfinite(config["memory_gib"]) or config["memory_gib"] <= 0:
        raise ValueError("Finite positive memory allocation required")
    if config["evaluation_n"] < 2:
        raise ValueError("Independent evaluation requires at least two rows for MCSE")
    if (
        not math.isfinite(config["maximum_parameter_gap"])
        or config["maximum_parameter_gap"] <= 0
    ):
        raise ValueError("Finite positive parameter matching tolerance required")
    for key in ("dimensions", "teachers", "ceilings", "conditions"):
        if not config[key]:
            raise ValueError(f"Nonempty {key} required")
    for dimension in config["dimensions"]:
        _integer(dimension, "dimension", 2)
    for teacher in config["teachers"]:
        _integer(teacher, "teacher", 0)
    for key in ("teacher_seed_base", "observation_seed_base", "model_seed_base"):
        _integer(config[key], key, 0)
    for ceiling in config["ceilings"]:
        _integer(ceiling, "ceiling")
    methods = {
        "centered_second_moment",
        "signed_second_hermite",
        "random",
        "oracle_private_plane",
    }
    for condition in config["conditions"]:
        if condition["geometry"] not in {"width2", "width3", "rank2", "full"}:
            raise ValueError("Unknown geometry")
        if condition["method"] not in methods:
            raise ValueError("Unknown initializer")
        if (
            condition["method"] != "centered_second_moment"
            and condition["geometry"] != "rank2"
        ):
            raise ValueError("Additional initializers are declared rank2 diagnostics")
    for checkpoint in config["checkpoints"]:
        _integer(checkpoint, "checkpoint")
        if checkpoint > config["steps"] or checkpoint % config["restart_every"]:
            raise ValueError("Checkpoints must lie on declared restart boundaries")
    if config["steps"] not in config["checkpoints"]:
        raise ValueError("Final study checkpoint must be retained")
    for key in ("teacher", "ceiling", "steps"):
        _integer(config["profile"][key], "profile " + key, 0 if key == "teacher" else 1)
    if config["profile"]["teacher"] in config["teachers"]:
        raise ValueError("Profile teacher must be separate from study teachers")
    if config["profile"]["steps"] != config["restart_every"]:
        raise ValueError("Profile must measure one full declared restart block")
    if config["profile"]["ceiling"] < max(config["ceilings"]):
        raise ValueError("Profile ceiling must cover every study ceiling")
    for value in config["profile_gate"].values():
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Finite positive profile limits required")
    if config["audit"] != {"forward_rtol": 3e-6, "forward_atol": 5e-12}:
        raise ValueError("Independent replay tolerances are unchanged")


def grid(config, stage):
    _validate_config(config)
    if stage not in ("profile", "study"):
        raise ValueError("Only profile and study have fitting grids")
    profile = config["profile"]
    teachers = [profile["teacher"]] if stage == "profile" else config["teachers"]
    ceilings = [profile["ceiling"]] if stage == "profile" else config["ceilings"]
    steps = profile["steps"] if stage == "profile" else config["steps"]
    rows = []
    for dimension, teacher, ceiling, condition in itertools.product(
        config["dimensions"], teachers, ceilings, config["conditions"]
    ):
        geometry, method = condition["geometry"], condition["method"]
        branches = width.branches_under_ceiling(
            ceiling, geometry, raw_blocks=1, inputs_per_block=dimension
        )
        inventory = width.parameter_inventory(
            branches, geometry, raw_blocks=1, inputs_per_block=dimension
        )
        rows.append(
            {
                "id": f"d{dimension}_t{teacher}_{geometry}_p{ceiling}_{method}",
                "stage": stage,
                "dimension": dimension,
                "teacher": teacher,
                "geometry": geometry,
                "method": method,
                "ceiling": ceiling,
                "train_n": config["train_n"],
                "steps": steps,
                "inventory": inventory,
            }
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate cases in declared grid")
    for dimension, ceiling in itertools.product(config["dimensions"], ceilings):
        inventories = [
            r["inventory"]
            for r in rows
            if r["dimension"] == dimension and r["ceiling"] == ceiling
        ]
        for field in ("stored_parameters", "gradient_parameter_slots"):
            counts = [item[field] for item in inventories]
            if (max(counts) - min(counts)) / min(counts) > config[
                "maximum_parameter_gap"
            ]:
                raise ValueError("Declared matched parameter gap exceeded")
    return rows


def initialize(root, config=None):
    root = Path(root).resolve()
    config = configuration() if config is None else config
    _validate_config(config)
    for name, expected in QUALIFIED_SOURCE_HASHES.items():
        if sha(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Qualified learner source changed: {name}")
    if (root / "initialized.json").exists():
        raise FileExistsError(
            "Initialized campaign is immutable; verify or use a new root"
        )
    root.mkdir(parents=True, exist_ok=True)
    source = root / "source"
    source.mkdir(exist_ok=False)
    (source / "__init__.py").write_text('"""Immutable campaign snapshot."""\n')
    for name in SOURCE_NAMES:
        shutil.copyfile(Path(__file__).with_name(name), source / name)
    write(root / "config.json", config)
    write(
        root / "manifest.json",
        {stage: grid(config, stage) for stage in ("profile", "study")},
    )
    bound = [root / "config.json", root / "manifest.json", *sorted(source.iterdir())]
    bound += [
        root / name
        for name in ("protocol.md", "preflight_checks.json", "environment.json")
        if (root / name).exists()
    ]
    write(
        root / "initialized.json",
        {
            "utc": now(),
            "bindings": {str(path.relative_to(root)): sha(path) for path in bound},
            "qualified_learner_sources": QUALIFIED_SOURCE_HASHES,
            "scope": config["scope"],
            "evaluation_generated": False,
            "source_execution": "Use python -m source.cosine_learning_campaign with this root on PYTHONPATH.",
        },
    )
    return root


def verify(root):
    root = Path(root).resolve()
    receipt = read(root / "initialized.json")
    for name, digest in receipt["bindings"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    loaded = {
        "cosine_learning_campaign.py": __file__,
        "cosine_learning_tasks.py": tasks.__file__,
        "cosine_family_tasks.py": family_tasks.__file__,
        "radial_width_learning.py": width.__file__,
        "rank_learning.py": base.__file__,
        "rank_orthogonal_v2.py": optimizer.__file__,
    }
    for name, source in loaded.items():
        if sha(source) != receipt["bindings"]["source/" + name]:
            raise ValueError(f"Executing source differs from frozen campaign: {name}")
    for name, expected in QUALIFIED_SOURCE_HASHES.items():
        if receipt["bindings"]["source/" + name] != expected:
            raise ValueError("Qualified learner source binding differs")
    config = read(root / "config.json")
    if read(root / "manifest.json") != {
        s: grid(config, s) for s in ("profile", "study")
    }:
        raise ValueError("Frozen manifest does not match configuration")
    return receipt


def _validate_case(root, case):
    if (
        case.get("stage") not in ("profile", "study")
        or case not in read(Path(root) / "manifest.json")[case["stage"]]
    ):
        raise ValueError("Case outside frozen stage manifest")


def dataset_key(case):
    return f"d{case['dimension']}_t{case['teacher']}_n{case['train_n']}"


def _data_paths(root, case):
    key = dataset_key(case)
    return (
        root / "data" / (key + ".npz"),
        root / "data" / (key + ".json"),
        root / "private_teachers" / (key + ".json"),
        root / "oracle_banks" / (key + ".npz"),
    )


def prepare_data(root, cases):
    """Coordinator-only generation; no private file is opened by ordinary fits."""
    root = Path(root).resolve()
    verify(root)
    config = read(root / "config.json")
    seen = set()
    for case in cases:
        _validate_case(root, case)
        key = dataset_key(case)
        if key in seen:
            continue
        seen.add(key)
        data, sidecar, private, oracle = _data_paths(root, case)
        present = [p.exists() for p in (data, sidecar, private, oracle)]
        if any(present) and not all(present):
            raise ValueError(
                "Incomplete dataset release preserved; use a reviewed new root"
            )
        if all(present):
            saved = read(sidecar)
            expected = [
                saved["sha256"],
                saved["private_teacher_sha256"],
                saved["oracle_bank_sha256"],
            ]
            if [sha(p) for p in (data, private, oracle)] != expected:
                raise ValueError(
                    "Archived data or private generation provenance changed"
                )
            if saved["initialized_sha256"] != sha(root / "initialized.json"):
                raise ValueError("Dataset initialization binding changed")
            continue
        teacher = family_tasks.make_teacher(config, case)
        arrays, receipt = tasks.generate_dataset(
            teacher,
            config["observation_seed_base"] + case["teacher"],
            train_n=case["train_n"],
            validation_n=config["validation_n"],
            test_n=0,
        )
        if any("test" in name for name in arrays) or receipt["test_generated"]:
            raise ValueError("Fitting release must not generate TEST")
        bank, oracle_receipt = tasks.oracle_bank(
            teacher, seed=config["model_seed_base"] + case["teacher"]
        )
        write(private, teacher.specification())
        _save_arrays(
            oracle,
            {"bank": bank, "receipt_json": np.asarray(json.dumps(oracle_receipt))},
        )
        _save_arrays(data, arrays)
        write(
            sidecar,
            {
                "utc": now(),
                "sha256": sha(data),
                "receipt": receipt,
                "private_teacher_sha256": sha(private),
                "oracle_bank_sha256": sha(oracle),
                "initialized_sha256": sha(root / "initialized.json"),
                "scope": "Public TRAIN/validation scalar observations; separate private archive and rotated-plane diagnostic. No TEST.",
            },
        )


def load_training_inputs(root, case):
    """Open only public TRAIN observations and their public integrity receipt."""
    root = Path(root).resolve()
    _validate_case(root, case)
    data, sidecar, _, _ = _data_paths(root, case)
    receipt = read(sidecar)
    if receipt["initialized_sha256"] != sha(root / "initialized.json"):
        raise ValueError("TRAIN release belongs to another source/configuration")
    if sha(data) != receipt["sha256"]:
        raise ValueError("Public dataset changed")
    with np.load(data, allow_pickle=False) as archive:
        result = {
            name: archive[name].copy() for name in ("x_train", "y_train", "y_train_raw")
        }
    for name, value in result.items():
        if tasks.array_sha256(value) != receipt["receipt"]["array_hashes"][name]:
            raise ValueError("Public TRAIN array hash changed")
    if result["x_train"].shape != (case["train_n"], 1, case["dimension"]):
        raise ValueError("TRAIN shape differs from frozen case")
    if any(
        result[name].shape != (case["train_n"],) for name in ("y_train", "y_train_raw")
    ):
        raise ValueError("Scalar TRAIN label count differs")
    return result, receipt


def independent_prediction(state_path, raw_x):
    """NumPy serialized ordinary MLP/bottleneck replay; no production forward."""
    with np.load(state_path, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    metadata = json.loads(str(arrays.pop("metadata_json")))
    mapping = metadata["initialization_receipt"][width.WIDTH_METADATA_KEY][
        "node_to_raw"
    ]
    capacities = metadata["capacities"]
    group = np.repeat(np.arange(len(capacities)), capacities)
    if not np.array_equal(arrays["group"], group):
        raise ValueError("Serialized groups differ from capacities")
    x = np.asarray(raw_x, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != 1 or not np.isfinite(x).all():
        raise ValueError("Finite raw [N,1,d] inputs required")
    result = np.full(len(x), float(arrays["bias"]), dtype=np.float64)
    offset = 0
    for node, capacity in enumerate(capacities):
        subset = slice(offset, offset + capacity)
        if metadata["architecture"] == "full":
            pre = x[:, mapping[node]] @ arrays["projection"][subset].T
        else:
            projected = x[:, mapping[node]] @ arrays["basis"][node]
            pre = projected @ arrays["branch_coefficients"][subset].T
        pre += arrays["threshold"][subset]
        if metadata["family"] != "relu":
            raise ValueError("This campaign qualifies ReLU only")
        weights = arrays["internal_readout"][subset] * arrays["soma_readout"][node]
        result += np.maximum(pre, 0) @ weights / np.sqrt(len(capacities))
        offset += capacity
    return result


def _case_root(root, case):
    return root / "cases" / case["stage"] / case["id"]


def _transient_artifact_reason(path):
    """Recognize only Linux NFS aliases and this writer's UUID temporary names.

    These exact filename conventions identify incidental names in new output
    inventories. They never relax an already-published artifact binding.
    """
    name = Path(path).name
    if re.fullmatch(r"\.nfs[0-9a-f]{24}", name):
        return "linux_nfs_silly_rename_name"
    if re.fullmatch(r"\..+\.pending_[0-9a-f]{32}", name):
        return "coordinator_atomic_publication_pending_name"
    return None


def _inventory_diagnostics(excluded):
    return {
        "schema": "cosine_durable_artifact_inventory_v1",
        "excluded_transients": excluded,
        "scope": (
            "Exact .nfs plus 24 lowercase hexadecimal characters and this "
            "coordinator's .<destination>.pending_<32 lowercase hexadecimal "
            "characters> names are recorded as observations, not durable "
            "artifacts. Their hashes and continued existence are not required. "
            "All durable artifact bindings remain strict; this rule does not "
            "repair or relax previously published inventories."
        ),
    }


def artifact_inventory(root, directory):
    """Return durable hashes and an explicit inventory of transient exclusions.

    Classification precedes hashing so a transient may disappear without
    invalidating the inventory. Missing or changed persistent files still fail.
    Similar-looking names outside the exact recognized grammars remain bound.
    A directory with a matching name is not itself excluded; durable descendants
    continue to be inventoried normally.
    """
    root, directory = Path(root), Path(directory)
    artifacts, excluded = {}, []
    for path in sorted(directory.rglob("*")):
        relative = str(path.relative_to(root))
        reason = _transient_artifact_reason(path)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            if reason is None:
                raise
            # Its exact transient name was observed during directory traversal;
            # expiration before metadata inspection is expected and recorded.
            excluded.append({"path": relative, "reason": reason})
            continue
        if reason is not None and stat.S_ISREG(metadata.st_mode):
            excluded.append({"path": relative, "reason": reason})
            continue
        if stat.S_ISREG(metadata.st_mode) or (
            stat.S_ISLNK(metadata.st_mode) and stat.S_ISREG(path.stat().st_mode)
        ):
            artifacts[relative] = sha(path)
    return {
        "artifacts": artifacts,
        "diagnostics": _inventory_diagnostics(excluded),
    }


def _hash_files(root, directory):
    """Compatibility helper; publication sites also retain inventory diagnostics."""
    return artifact_inventory(root, directory)["artifacts"]


def _check_bindings(root, bindings):
    for name, expected in bindings.items():
        if sha(root / name) != expected:
            raise ValueError(f"Bound artifact changed: {name}")


def _count_model(model, case):
    expected = case["inventory"]
    stored = sum(parameter.numel() for parameter in model.parameters())
    gradient = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if (
        stored != expected["stored_parameters"]
        or gradient != expected["gradient_parameter_slots"]
        or stored != model.parameter_count
        or gradient != model.gradient_parameter_count
    ):
        raise ValueError(
            "Actual stored or gradient parameter count differs from manifest"
        )
    return model.counted_metadata()


def verify_case(root, case):
    """Read-only verification, including all committed stage and prediction files."""
    root = Path(root).resolve()
    _validate_case(root, case)
    receipt = read(_case_root(root, case) / "complete.json")
    if receipt["status"] != "complete" or receipt["case"] != case:
        raise ValueError("Completed case differs from frozen manifest")
    if receipt["initialized_sha256"] != sha(root / "initialized.json"):
        raise ValueError("Completed case source/configuration binding differs")
    data, sidecar, _, oracle = _data_paths(root, case)
    if receipt["dataset_sha256"] != sha(data) or receipt[
        "dataset_receipt_sha256"
    ] != sha(sidecar):
        raise ValueError("Completed case public observations changed")
    if case["method"] == "oracle_private_plane" and receipt[
        "oracle_bank_sha256"
    ] != sha(oracle):
        raise ValueError("Completed oracle case private plane archive changed")
    _check_bindings(root, receipt["artifacts"])
    expected_checkpoints = _checkpoints(read(root / "config.json"), case)
    if set(receipt["checkpoints"]) != {str(n) for n in expected_checkpoints}:
        raise ValueError("Completed checkpoint inventory differs")
    return receipt


def _checkpoints(config, case):
    return (
        [case["steps"]]
        if case["stage"] == "profile"
        else sorted(set(config["checkpoints"] + [case["steps"]]))
    )


def _native_prediction(model, inputs, batch=512):
    with torch.no_grad():
        prediction = np.concatenate(
            [
                model(inputs[start : start + batch]).detach().cpu().numpy()
                for start in range(0, len(inputs), batch)
            ]
        )
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite native prediction")
    return prediction


def _replay_prediction(model, state_path, inputs, config):
    native = _native_prediction(model, inputs)
    replay = np.concatenate(
        [
            independent_prediction(state_path, inputs[start : start + 512])
            for start in range(0, len(inputs), 512)
        ]
    )
    if not np.isfinite(replay).all():
        raise FloatingPointError("Nonfinite independent prediction")
    np.testing.assert_allclose(
        replay,
        native,
        rtol=config["audit"]["forward_rtol"],
        atol=config["audit"]["forward_atol"],
    )
    return native, float(np.max(np.abs(replay - native)))


def _record_checkpoint(
    root,
    case,
    model,
    state_path,
    destination,
    budget_steps,
    iterations,
    closure_calls,
    config,
    training,
    sidecar,
):
    data_path, _, _, _ = _data_paths(root, case)
    with np.load(data_path, allow_pickle=False) as archive:
        validation_x = archive["x_validation"].copy()
        validation_y = archive["y_validation"].copy()
    normalization = sidecar["receipt"]["normalization"]
    predictions, metrics = {}, {}
    for split, inputs, labels in (
        ("train", training["x_train"], training["y_train"]),
        ("validation", validation_x, validation_y),
    ):
        prediction, error = _replay_prediction(model, state_path, inputs, config)
        predictions[split] = prediction
        metrics[split + "_mse"] = float(np.mean((prediction - labels) ** 2))
        metrics[split + "_raw_mse"] = (
            metrics[split + "_mse"] * normalization["scale"] ** 2
        )
        metrics[split + "_replay_max_error"] = error
    path = destination / "predictions.npz"
    _save_arrays(path, predictions)
    result = {
        "budget_steps": budget_steps,
        "iterations": iterations,
        "closure_calls": closure_calls,
        "metrics": metrics,
        "state_path": str(state_path.relative_to(root)),
        "predictions_path": str(path.relative_to(root)),
        "scope": "Fixed optimization milestone; TRAIN/validation are descriptive diagnostics, with no MCSE or model selection.",
    }
    write(destination / "checkpoint.json", result)
    return result


def _verify_restart(
    root, completed, case, stage, steps, predecessor, data_hash, sidecar_hash
):
    if (
        completed["case"] != case
        or completed["stage"] != stage
        or completed["steps"] != steps
        or completed["initialized_sha256"] != sha(root / "initialized.json")
        or completed["dataset_sha256"] != data_hash
        or completed["dataset_receipt_sha256"] != sidecar_hash
        or completed["predecessor_sha256"] != predecessor
    ):
        raise ValueError("Committed restart state binding differs")
    _check_bindings(root, completed["artifacts"])


def run_case(root, case):
    root = Path(root).resolve()
    verify(root)
    _validate_case(root, case)
    directory = _case_root(root, case)
    if (directory / "complete.json").exists():
        return verify_case(root, case)
    torch.set_num_threads(1)
    config = read(root / "config.json")
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_lease(directory / "worker.lease"):
        if (directory / "complete.json").exists():
            return verify_case(root, case)
        if list(directory.rglob("failure.json")):
            raise RuntimeError(
                "Scientific/numerical failure retained; no automatic retry"
            )
        previous_attempts = sorted(directory.glob("attempt_*"))
        attempt = directory / f"attempt_{len(previous_attempts):03d}"
        attempt.mkdir()
        write(
            attempt / "invocation.json",
            {
                "utc": now(),
                "case": case,
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "interrupted_attempts_preserved": [
                    path.name for path in previous_attempts
                ],
            },
        )
        started = time.monotonic()
        try:
            training, sidecar = load_training_inputs(root, case)
            data, sidecar_path, _, oracle = _data_paths(root, case)
            data_hash, sidecar_hash = sha(data), sha(sidecar_path)
            seed = config["model_seed_base"] + case["teacher"]
            oracle_hash = None
            if case["method"] == "oracle_private_plane":
                oracle_hash = sha(oracle)
                if oracle_hash != sidecar["oracle_bank_sha256"]:
                    raise ValueError("Private plane diagnostic archive changed")
                with np.load(oracle, allow_pickle=False) as archive:
                    bank = archive["bank"].copy()
                    bank_receipt = json.loads(str(archive["receipt_json"]))
                if tasks.array_sha256(bank) != bank_receipt["bank_sha256"]:
                    raise ValueError("Private plane diagnostic array changed")
            else:
                bank, bank_receipt = tasks.estimate_bank(
                    training["x_train"],
                    training["y_train_raw"],
                    method=case["method"],
                    seed=seed,
                )
            if "bank_frame_rotation" in config:
                bank, bank_receipt = tasks.rotate_bank_frame(
                    bank,
                    bank_receipt,
                    seed_base=config["bank_frame_rotation"]["seed_base"],
                    teacher=case["teacher"],
                    ambient_dim=case["dimension"],
                )
            model = width.model_under_ceiling(
                case["ceiling"],
                "relu",
                case["geometry"],
                bank=bank,
                train_data=(training["x_train"], training["y_train"]),
                seed=seed,
                allocation_seed=seed,
                coverage_lines=config["coverage_lines"],
                initializer_receipt=bank_receipt,
                base_module=base,
            )
            counted = _count_model(model, case)
            write(attempt / "initializer.json", bank_receipt)
            checkpoints, artifacts = {}, {}
            excluded_transients = []
            iterations = closure_calls = resumed = budget = 0
            summed_stage_seconds = 0.0
            fit_records = []
            stage_count = math.ceil(case["steps"] / config["restart_every"])
            for index in range(stage_count):
                stage_dir = directory / f"stage_{index:03d}"
                stage_dir.mkdir(exist_ok=True)
                steps = min(config["restart_every"], case["steps"] - budget)
                predecessor = (
                    None
                    if index == 0
                    else sha(directory / f"stage_{index - 1:03d}" / "complete.json")
                )
                marker = stage_dir / "complete.json"
                if marker.exists():
                    committed = read(marker)
                    _verify_restart(
                        root,
                        committed,
                        case,
                        index,
                        steps,
                        predecessor,
                        data_hash,
                        sidecar_hash,
                    )
                    model = width.load_state(
                        root / committed["state_path"], base_module=base
                    )
                    _count_model(model, case)
                    fit = read(root / committed["fit_path"])
                    resumed += 1
                else:
                    stage_attempt = (
                        stage_dir
                        / f"attempt_{len(list(stage_dir.glob('attempt_*'))):03d}"
                    )
                    stage_attempt.mkdir()
                    stage_started = time.monotonic()
                    fit = optimizer.fit_restarted(
                        model.core,
                        model.expanded_inputs(training["x_train"]),
                        training["y_train"],
                        base.FitConfig(
                            ridge=config["ridge"],
                            projection_rms_floor=config.get(
                                "projection_rms_floor", 0.0
                            ),
                            steps=steps,
                            learning_rate=0.5,
                            objective_scale=1e4,
                            tolerance_grad=1e-10,
                            tolerance_change=1e-14,
                            history_size=30,
                        ),
                        stages=1,
                        rebalance=True,
                        output_dir=stage_attempt / "fit",
                    )
                    state_path = stage_attempt / "model.npz"
                    width.save_state(model, state_path)
                    _count_model(model, case)
                    checkpoint = None
                    if budget + steps in _checkpoints(config, case):
                        checkpoint = _record_checkpoint(
                            root,
                            case,
                            model,
                            state_path,
                            stage_attempt,
                            budget + steps,
                            iterations + fit["iterations"],
                            closure_calls + fit["closure_calls"],
                            config,
                            training,
                            sidecar,
                        )
                    inventory = artifact_inventory(root, stage_attempt)
                    committed = {
                        "utc": now(),
                        "case": case,
                        "stage": index,
                        "steps": steps,
                        "initialized_sha256": sha(root / "initialized.json"),
                        "dataset_sha256": data_hash,
                        "dataset_receipt_sha256": sidecar_hash,
                        "predecessor_sha256": predecessor,
                        "state_path": str(state_path.relative_to(root)),
                        "fit_path": str(
                            (stage_attempt / "fit" / "fit.json").relative_to(root)
                        ),
                        "checkpoint": checkpoint,
                        "seconds": time.monotonic() - stage_started,
                        "artifacts": inventory["artifacts"],
                        "artifact_inventory_diagnostics": inventory["diagnostics"],
                    }
                    if sha(data) != data_hash or sha(sidecar_path) != sidecar_hash:
                        raise ValueError(
                            "Public observations changed during restart block"
                        )
                    write(marker, committed)
                iterations += fit["iterations"]
                closure_calls += fit["closure_calls"]
                budget += steps
                summed_stage_seconds += committed["seconds"]
                fit_records.append(fit)
                if committed["checkpoint"] is not None:
                    checkpoint = committed["checkpoint"]
                    if (
                        checkpoint["iterations"] != iterations
                        or checkpoint["budget_steps"] != budget
                    ):
                        raise ValueError(
                            "Committed checkpoint iteration accounting differs"
                        )
                    checkpoints[str(budget)] = checkpoint
                artifacts.update(committed["artifacts"])
                excluded_transients.extend(
                    committed["artifact_inventory_diagnostics"]["excluded_transients"]
                )
                artifacts[str(marker.relative_to(root))] = sha(marker)
            if set(checkpoints) != {str(n) for n in _checkpoints(config, case)}:
                raise ValueError("Incomplete declared milestone inventory")
            verify(root)
            if sha(data) != data_hash or sha(sidecar_path) != sidecar_hash:
                raise ValueError("Public observations changed before case publication")
            if oracle_hash is not None and sha(oracle) != oracle_hash:
                raise ValueError("Private plane diagnostic changed during fit")
            inventory = artifact_inventory(root, attempt)
            artifacts.update(inventory["artifacts"])
            excluded_transients.extend(inventory["diagnostics"]["excluded_transients"])
            result = {
                "utc": now(),
                "status": "complete",
                "case": case,
                "initialized_sha256": sha(root / "initialized.json"),
                "dataset_sha256": data_hash,
                "dataset_receipt_sha256": sidecar_hash,
                "oracle_bank_sha256": oracle_hash,
                "counted_inventory": counted,
                "initializer": bank_receipt,
                "normalization": sidecar["receipt"]["normalization"],
                "checkpoints": checkpoints,
                "state_path": committed["state_path"],
                "metrics": checkpoints[str(case["steps"])]["metrics"],
                "iterations": iterations,
                "closure_calls": closure_calls,
                "terminal_unscaled_gradient_l2": fit_records[-1][
                    "terminal_unscaled_gradient_l2"
                ],
                "seconds": summed_stage_seconds,
                "elapsed_seconds_this_attempt": time.monotonic() - started,
                "resumed_restart_blocks": resumed,
                "process_peak_rss_gib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 2**20,
                "rss_scope": "Linux worker lifetime maximum; production supervisor isolates one case per process.",
                "artifacts": artifacts,
                "artifact_inventory_diagnostics": _inventory_diagnostics(
                    excluded_transients
                ),
                "audit_scope": "All TRAIN/validation checkpoint predictions independently replayed from serialized NumPy graph with unchanged tolerances. Actual iterations can be less than budget after early termination.",
            }
            write(directory / "complete.json", result)
            return result
        except Exception as error:
            write(
                attempt / "failure.json",
                {
                    "utc": now(),
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "seconds": time.monotonic() - started,
                },
            )
            raise


def completed_stage(root, stage):
    root = Path(root).resolve()
    verify(root)
    if stage not in ("profile", "study"):
        raise ValueError("Unknown fitting stage")
    marker = read(root / f"{stage}_complete.json")
    cases = read(root / "manifest.json")[stage]
    if (
        marker["status"] != "complete"
        or marker["expected_cases"] != len(cases)
        or marker["errors"]
    ):
        raise ValueError("Stage is incomplete or contains retained failures")
    expected = {
        case["id"]: sha(_case_root(root, case) / "complete.json") for case in cases
    }
    if marker["case_receipt_hashes"] != expected:
        raise ValueError("Stage completed-case receipt inventory changed")
    return [verify_case(root, case) for case in cases]


def profile_gate(root):
    root = Path(root).resolve()
    config = read(root / "config.json")
    rows = completed_stage(root, "profile")
    limits = config["profile_gate"]

    def measured_cost(row):
        return max(row["seconds"], row["elapsed_seconds_this_attempt"])

    largest_seconds = max(measured_cost(row) for row in rows)
    largest_rss = max(row["process_peak_rss_gib"] for row in rows)
    if (
        largest_seconds > limits["maximum_case_seconds"]
        or largest_rss > limits["maximum_case_rss_gib"]
    ):
        raise ValueError("Profile exceeds frozen per-case runtime or memory gate")
    by_condition = {
        (row["case"]["dimension"], row["case"]["geometry"], row["case"]["method"]): row
        for row in rows
    }
    study = read(root / "manifest.json")["study"]
    projected_seconds = (
        sum(
            measured_cost(
                by_condition[(case["dimension"], case["geometry"], case["method"])]
            )
            * case["steps"]
            / config["profile"]["steps"]
            for case in study
        )
        * limits["runtime_margin"]
    )
    memory = (
        config["workers"] * largest_rss * limits["memory_margin"]
        + limits["coordinator_memory_gib"]
    )
    if projected_seconds / 3600 > limits["maximum_projected_case_hours"]:
        raise ValueError("Profile exceeds frozen total projected study case-hour gate")
    if memory > config["memory_gib"]:
        raise ValueError("Measured parallel worker memory exceeds allocation")
    return {
        "status": "passed",
        "profile_cases": len(rows),
        "maximum_case_seconds": largest_seconds,
        "maximum_case_rss_gib": largest_rss,
        "projected_study_case_hours_with_margin": projected_seconds / 3600,
        "projected_memory_gib_with_margin": memory,
        "timing_definition": "Per-profile maximum of summed committed restart seconds and current-attempt elapsed seconds, including initialization and replay. Child import and coordinator overhead are excluded.",
        "scope": "Runtime/memory and replay completeness only. No validation value is a gate. Linear iteration proxy uses largest profiled budget for every smaller study budget, with declared margin; not a wall-time guarantee.",
    }


def _worker(arguments):
    root, case = arguments
    try:
        result = run_case(root, case)
        return {"case": case["id"], "status": "complete", "seconds": result["seconds"]}
    except Exception as error:
        return {"case": case["id"], "status": "failed", "error": repr(error)}


_MAX_WORKER_RESULT_BYTES = 1024 * 1024


def _worker_entry(worker, arguments, result_path):
    """One spawned case, with an atomic local-file handoff and no queue feeder."""
    try:
        result = worker(arguments)
    except Exception as error:
        result = {"case": arguments[1]["id"], "status": "failed", "error": repr(error)}
    payload = json.dumps(result, allow_nan=False).encode("utf-8")
    if len(payload) > _MAX_WORKER_RESULT_BYTES:
        raise ValueError("Worker result exceeds bounded handoff size")
    result_path = Path(result_path)
    temporary = result_path.with_suffix(".partial")
    with temporary.open("xb") as handle:
        handle.write(payload)
    os.replace(temporary, result_path)


def _exited_worker_result(entry, exitcode):
    """Never accept a successful message from an abnormally exiting process."""
    case_id = entry["arguments"][1]["id"]
    failure = {
        "case": case_id,
        "status": "failed",
        "worker_pid": entry["process"].pid,
        "worker_exitcode": exitcode,
        "worker_exit_signal": -exitcode if exitcode < 0 else None,
    }
    if exitcode != 0:
        return {
            **failure,
            "failure_kind": "abnormal_worker_exit",
            "error": f"Worker exited with code {exitcode}; no successful handoff accepted",
        }, True
    try:
        # The child has exited. Read at most the declared bound from a local
        # regular file, rather than blocking on a possibly incomplete IPC frame.
        path = entry["result_path"]
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError("Worker result is not a regular file")
        with path.open("rb") as handle:
            payload = handle.read(_MAX_WORKER_RESULT_BYTES + 1)
        if len(payload) > _MAX_WORKER_RESULT_BYTES:
            raise ValueError("Worker result exceeds bounded handoff size")
        result = json.loads(payload)
        if (
            not isinstance(result, dict)
            or result.get("case") != case_id
            or result.get("status") not in ("complete", "failed")
        ):
            raise ValueError("Worker result has invalid case/status")
        if result["status"] == "complete" and (
            not isinstance(result.get("seconds"), (float, int))
            or not math.isfinite(result["seconds"])
            or result["seconds"] < 0
        ):
            raise ValueError("Worker success lacks finite elapsed seconds")
        if result["status"] == "failed" and not isinstance(result.get("error"), str):
            raise ValueError("Worker failure lacks an error description")
    except Exception as error:
        return {
            **failure,
            "failure_kind": "invalid_worker_handoff",
            "error": repr(error),
        }, True
    return result, False


def _supervised_results(
    arguments,
    workers,
    *,
    worker=_worker,
    on_abnormal=None,
    poll_seconds=0.2,
    cleanup_seconds=2.0,
):
    """Bound concurrency and detect lost workers without a Pool result queue.

    Abnormal exit or missing/malformed handoff stops further launches. Running
    peers are allowed to finish; every unlaunched case is explicitly reported.
    The callback runs as soon as an abnormal exit is observed, before waiting
    for peers. It must persist that failure for a later Slurm interruption.
    No timeout is imposed on a healthy numerical worker's declared fit budget.
    """
    arguments = list(arguments)
    workers = _integer(workers, "workers")
    if not 0 < poll_seconds <= 1 or not 0 < cleanup_seconds <= 10:
        raise ValueError("Supervisor polling/cleanup bounds are invalid")
    if len({argument[1]["id"] for argument in arguments}) != len(arguments):
        raise ValueError("Each supervised case must be unique")
    context = multiprocessing.get_context("spawn")
    active, outcomes, next_index, stopped = {}, [], 0, False

    def abnormal(result):
        if on_abnormal is not None:
            on_abnormal(
                {
                    "utc": now(),
                    "failure": result,
                    "running_cases": [
                        item["arguments"][1]["id"] for item in active.values()
                    ],
                    "not_started_cases": [
                        item[1]["id"] for item in arguments[next_index:]
                    ],
                    "observed_outcomes": list(outcomes),
                    "policy": "Stop new launches, retain running peers, account for every case, never retry a lost worker.",
                    "transport": "Bounded atomic JSON files under local /tmp; no blocking queue or pipe receive.",
                }
            )

    # Deliberately ignore TMPDIR, which can point to NFS on cluster nodes.
    with tempfile.TemporaryDirectory(prefix="cosine-workers-", dir="/tmp") as local:
        try:
            while active or (next_index < len(arguments) and not stopped):
                while (
                    not stopped
                    and next_index < len(arguments)
                    and len(active) < workers
                ):
                    argument = arguments[next_index]
                    result_path = Path(local) / f"result_{next_index:06d}.json"
                    next_index += 1
                    process = context.Process(
                        target=_worker_entry, args=(worker, argument, str(result_path))
                    )
                    try:
                        process.start()
                    except Exception as error:
                        result = {
                            "case": argument[1]["id"],
                            "status": "failed",
                            "failure_kind": "worker_start_failure",
                            "error": repr(error),
                            "worker_pid": process.pid,
                            "worker_exitcode": process.exitcode,
                        }
                        process.close()
                        stopped = True
                        outcomes.append(result)
                        abnormal(result)
                        yield result
                        break
                    active[process.pid] = {
                        "process": process,
                        "arguments": argument,
                        "result_path": result_path,
                    }
                if not active:
                    continue
                ready = wait_for_processes(
                    [item["process"].sentinel for item in active.values()],
                    timeout=poll_seconds,
                )
                for pid, entry in list(active.items()):
                    process = entry["process"]
                    if process.sentinel not in ready and process.exitcode is None:
                        continue
                    process.join(timeout=0)
                    exitcode = process.exitcode
                    if exitcode is None:
                        continue
                    del active[pid]
                    result, failed_handoff = _exited_worker_result(entry, exitcode)
                    process.close()
                    outcomes.append(result)
                    if failed_handoff:
                        stopped = True
                        abnormal(result)
                    yield result
            for argument in arguments[next_index:]:
                result = {
                    "case": argument[1]["id"],
                    "status": "not_started",
                    "failure_kind": "stage_stopped_after_worker_failure",
                    "error": "Not launched after an abnormal worker failure; no retry attempted",
                }
                outcomes.append(result)
                yield result
        finally:
            # Cleanup on coordinator exceptions/consumer cancellation is bounded;
            # normal execution has already reaped every launched process.
            for entry in active.values():
                entry["process"].terminate()
            deadline = time.monotonic() + cleanup_seconds
            for entry in active.values():
                entry["process"].join(timeout=max(0, deadline - time.monotonic()))
            for entry in active.values():
                if entry["process"].is_alive():
                    entry["process"].kill()
            deadline = time.monotonic() + cleanup_seconds
            for entry in active.values():
                process = entry["process"]
                process.join(timeout=max(0, deadline - time.monotonic()))
                if not process.is_alive():
                    process.close()


def run_stage(root, stage, workers):
    root = Path(root).resolve()
    verify(root)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Production fitting requires a Slurm allocation")
    workers = _integer(workers, "workers")
    config = read(root / "config.json")
    if workers > config["workers"]:
        raise ValueError("Worker count exceeds frozen allocation")
    if stage not in ("profile", "study"):
        raise ValueError("Unknown fitting stage")
    marker = root / f"{stage}_complete.json"
    if marker.exists():
        return completed_stage(root, stage)

    with exclusive_lease(root / "coordinator.lease"):
        if marker.exists():
            return completed_stage(root, stage)
        worker_failure_marker = root / f"{stage}_worker_failure.json"
        if worker_failure_marker.exists():
            raise RuntimeError(
                "A retained abnormal worker failure blocks this stage; no automatic retry"
            )
        if stage == "study":
            gate = profile_gate(root)
            if not (root / "profile_gate.json").exists():
                write(root / "profile_gate.json", gate)
            elif read(root / "profile_gate.json") != gate:
                raise ValueError("Previously published profile gate changed")
        cases = read(root / "manifest.json")[stage]
        prepare_data(root, cases)
        outcomes = []
        worker_failure_receipts = {}

        def retain_worker_failure(event):
            receipt = {
                "stage": stage,
                "coordinator_pid": os.getpid(),
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "initialized_sha256": sha(root / "initialized.json"),
                "expected_cases": [case["id"] for case in cases],
                **event,
            }
            # Persist the first failure before any peer can delay final stage
            # publication. A later invocation must never restart this stage.
            if not worker_failure_marker.exists():
                write(worker_failure_marker, receipt)
            path = (
                root
                / f"{stage}_worker_failures"
                / f"{len(worker_failure_receipts):03d}.json"
            )
            write(path, receipt)
            worker_failure_receipts[str(path.relative_to(root))] = sha(path)
            print(json.dumps({"worker_failure_detected": receipt}), flush=True)

        for result in _supervised_results(
            [(str(root), case) for case in cases],
            workers,
            on_abnormal=retain_worker_failure,
        ):
            outcomes.append(result)
            print(
                json.dumps({"done": len(outcomes), "total": len(cases), **result}),
                flush=True,
            )
        errors = [row for row in outcomes if row["status"] != "complete"]
        complete_hashes = {}
        for case in cases:
            if (_case_root(root, case) / "complete.json").exists():
                verify_case(root, case)
                complete_hashes[case["id"]] = sha(
                    _case_root(root, case) / "complete.json"
                )
        if len(complete_hashes) != len(cases) and not errors:
            errors.append({"error": "Incomplete result inventory"})
        verify(root)
        write(
            marker,
            {
                "utc": now(),
                "status": "failed" if errors else "complete",
                "expected_cases": len(cases),
                "errors": errors,
                "case_receipt_hashes": complete_hashes,
                "worker_outcomes": outcomes,
                "worker_failure_receipts": worker_failure_receipts,
            },
        )
        if errors:
            raise RuntimeError(
                f"{len(errors)} failed or unstarted cases; no scientific retries"
            )
        return completed_stage(root, stage)


def _evaluation_input_bindings(root, receipts):
    """Bind all frozen models before any independent evaluation observations exist."""
    bindings = {
        "initialized.json": sha(root / "initialized.json"),
        "study_complete.json": sha(root / "study_complete.json"),
    }
    bindings.update(read(root / "initialized.json")["bindings"])
    for receipt in receipts:
        case = receipt["case"]
        data, sidecar, private, oracle = _data_paths(root, case)
        archived = read(sidecar)
        if (
            sha(private) != archived["private_teacher_sha256"]
            or sha(oracle) != archived["oracle_bank_sha256"]
        ):
            raise ValueError("Private generation provenance changed before evaluation")
        for path in (
            data,
            sidecar,
            private,
            oracle,
            _case_root(root, case) / "complete.json",
        ):
            bindings[str(path.relative_to(root))] = sha(path)
        bindings.update(receipt["artifacts"])
    return bindings


def _verify_evaluation(root, receipt):
    verify(root)
    completed_stage(root, "study")
    if receipt["status"] != "complete":
        raise ValueError("Independent evaluation is incomplete")
    _check_bindings(root, receipt["input_bindings"])
    _check_bindings(root, receipt["artifacts"])
    return receipt


def _evaluation_dataset(root, case, config, destination, model_binding_sha):
    data, sidecar, private, _ = _data_paths(root, case)
    public_receipt = read(sidecar)
    if sha(private) != public_receipt["private_teacher_sha256"]:
        raise ValueError("Private teacher archive changed before evaluation draw")
    teacher = family_tasks.restore_teacher(read(private))
    arrays, receipt = tasks.generate_dataset(
        teacher,
        config["observation_seed_base"] + case["teacher"],
        train_n=case["train_n"],
        validation_n=config["validation_n"],
        test_n=config["evaluation_n"],
    )
    with np.load(data, allow_pickle=False) as original:
        if set(original.files) != {key for key in arrays if "test" not in key}:
            raise ValueError("Regenerated public observation inventory differs")
        for key in original.files:
            if not np.array_equal(original[key], arrays[key]):
                raise ValueError("Regenerated TRAIN/validation observations differ")
    if receipt["normalization"] != public_receipt["receipt"]["normalization"]:
        raise ValueError("Regenerated TRAIN normalization differs")
    if not receipt["test_generated"] or len(arrays["x_test"]) != config["evaluation_n"]:
        raise ValueError("Independent evaluation row count differs")
    test_arrays = {key: value for key, value in arrays.items() if "test" in key}
    path = destination / ("test_" + dataset_key(case) + ".npz")
    _save_arrays(path, test_arrays)
    write(
        path.with_suffix(".json"),
        {
            "utc": now(),
            "sha256": sha(path),
            "array_hashes": {
                key: tasks.array_sha256(value) for key, value in test_arrays.items()
            },
            "split_seed": receipt["split_seeds"]["test"],
            "observation_seed": receipt["observation_seed"],
            "normalization": receipt["normalization"],
            "all_study_models_binding_sha256": model_binding_sha,
            "source_dataset_sha256": sha(data),
            "source_dataset_receipt_sha256": sha(sidecar),
            "private_teacher_sha256": sha(private),
            "scope": "Independent observations of fixed DEVELOPMENT teachers, released after all study models completed. Not fresh-response confirmation; no model selection.",
        },
    )
    return test_arrays, receipt["normalization"]


def _row_base(case, normalization, floor=None):
    if floor is None:
        floor = math.exp(-1) * (1 - 1 / math.sqrt(2)) ** 2 / 72
    return {
        "case_id": case["id"],
        "dimension": case["dimension"],
        "teacher": case["teacher"],
        "geometry": case["geometry"],
        "method": case["method"],
        "ceiling": case["ceiling"],
        "stored_parameters": case["inventory"]["stored_parameters"],
        "gradient_parameters": case["inventory"]["gradient_parameter_slots"],
        "two_direction_raw_floor": floor,
        "two_direction_normalized_floor": floor / normalization["scale"] ** 2,
        "oracle_information": case["method"] == "oracle_private_plane",
    }


def _export_evaluation(destination, endpoint_rows, test_rows, config=None):
    rows = endpoint_rows + test_rows
    fields = [
        "case_id",
        "dimension",
        "teacher",
        "geometry",
        "method",
        "ceiling",
        "stored_parameters",
        "gradient_parameters",
        "kind",
        "budget_steps",
        "actual_iterations",
        "train_mse",
        "validation_mse",
        "test_mse",
        "train_raw_mse",
        "validation_raw_mse",
        "test_raw_mse",
        "test_mcse",
        "test_raw_mcse",
        "test_rows",
        "two_direction_raw_floor",
        "two_direction_normalized_floor",
        "empirical_test_below_two_direction_floor",
        "oracle_information",
    ]
    with (destination / "all_endpoints.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    varied = config is not None and config.get("response_family") == family_tasks.FAMILY
    scope = (
        "Every declared final model is evaluated on independent observations after all "
        "study fits complete. These development targets vary directions, frequencies and "
        "amplitudes independently of the ambient plane. They are not reserved confirmation "
        "teachers. No validation/TEST selection, fitted scaling exponents, or confidence "
        "intervals are reported."
        if varied
        else "Every declared final model is evaluated on independent observations after all "
        "study fits complete. Teacher orientations were used in development; these are "
        "not fresh response functions or confirmation teachers. No validation/TEST "
        "selection, fitted scaling exponents, or confidence intervals are reported."
    )
    lines = [
        "# Three-cosine development learning bridge",
        "",
        scope,
        "",
        "The MCSE is the sample standard deviation of pointwise squared errors divided "
        "by the square root of the independent evaluation size, conditional on this "
        "fixed model and teacher. It is a descriptive Monte Carlo standard error, not "
        "a rigorous confidence interval, and does not measure variation across teachers. "
        "No such MCSE is assigned to TRAIN errors.",
        "",
        "The two-direction floor is the fourth-Hermite population-risk lower bound for "
        "this teacher, evaluated numerically without an interval certificate. It applies "
        "to sums of two ridge functions. Empirical evaluation below "
        "that floor is a diagnostic estimate, not a certified population-risk inequality. "
        "TRAIN-derived target normalization changes the normalized floor by scale squared.",
        "",
        "Centered-moment initialization retains the existing signed-centered recipe. "
        "The raw Hermite, random and privately supplied rotated true-plane arms are "
        "separate rank2 diagnostics. Width3 and full-projection conventional controls "
        "remain in the complete outcome table.",
        "",
        f"Retained {len(endpoint_rows)} fixed TRAIN/validation milestones and {len(test_rows)} "
        "final-model independent evaluation entries in all_endpoints.csv.",
        "",
        "| d | Teacher | Geometry | Initializer | Stored P | Raw TEST MSE | Raw MCSE | MSE / floor |",
        "|---:|---:|:---|:---|---:|---:|---:|---:|",
    ]
    for row in test_rows:
        lines.append(
            f"| {row['dimension']} | {row['teacher']} | {row['geometry']} | {row['method']} | "
            f"{row['stored_parameters']} | {row['test_raw_mse']:.8g} | {row['test_raw_mcse']:.5g} | "
            f"{row['test_raw_mse'] / row['two_direction_raw_floor']:.6g} |"
        )
    (destination / "summary.md").write_text("\n".join(lines) + "\n")


def evaluate_complete_study(root):
    """Release TEST only after all study models pass complete immutable binding.

    No candidate selection takes place. The evaluation covers all final models
    and reports descriptive independent-observation MCSEs conditional on models
    and fixed development teachers, never fitted exponents or rigorous intervals.
    """
    root = Path(root).resolve()
    verify(root)
    completed = completed_stage(root, "study")
    marker = root / "evaluation_complete.json"
    if marker.exists():
        return _verify_evaluation(root, read(marker))
    config = read(root / "config.json")
    with exclusive_lease(root / "coordinator.lease"):
        if marker.exists():
            return _verify_evaluation(root, read(marker))
        # This gate precedes even creation of the evaluation directory.
        completed = completed_stage(root, "study")
        input_bindings = _evaluation_input_bindings(root, completed)
        _check_bindings(root, input_bindings)
        directory = root / "evaluation"
        if directory.exists() and list(directory.rglob("failure.json")):
            raise RuntimeError(
                "Failed evaluation retained; no automatic scientific retry"
            )
        directory.mkdir(exist_ok=True)
        attempt = directory / f"attempt_{len(list(directory.glob('attempt_*'))):03d}"
        attempt.mkdir()
        started = time.monotonic()
        try:
            write(attempt / "input_bindings.json", input_bindings)
            binding_sha = sha(attempt / "input_bindings.json")
            write(
                attempt / "invocation.json",
                {
                    "utc": now(),
                    "models": len(completed),
                    "all_study_models_binding_sha256": binding_sha,
                    "evaluation_n": config["evaluation_n"],
                    "scope": "Independent observations of fixed development teachers, all frozen final models.",
                },
            )
            datasets = {}
            for receipt in completed:
                case = receipt["case"]
                key = dataset_key(case)
                if key not in datasets:
                    datasets[key] = _evaluation_dataset(
                        root, case, config, attempt, binding_sha
                    )
            endpoint_rows, test_rows = [], []
            for receipt in completed:
                case = receipt["case"]
                arrays, normalization = datasets[dataset_key(case)]
                _, _, private_path, _ = _data_paths(root, case)
                teacher = family_tasks.restore_teacher(read(private_path))
                row_base = _row_base(
                    case, normalization, family_tasks.raw_floor(teacher)
                )
                for budget, checkpoint in sorted(
                    receipt["checkpoints"].items(), key=lambda item: int(item[0])
                ):
                    metrics = checkpoint["metrics"]
                    endpoint_rows.append(
                        {
                            **row_base,
                            "kind": "train_validation_checkpoint",
                            "budget_steps": int(budget),
                            "actual_iterations": checkpoint["iterations"],
                            **{
                                key: metrics[key]
                                for key in (
                                    "train_mse",
                                    "validation_mse",
                                    "train_raw_mse",
                                    "validation_raw_mse",
                                )
                            },
                        }
                    )
                model_path = root / receipt["state_path"]
                model = width.load_state(model_path, base_module=base)
                _count_model(model, case)
                prediction, error = _replay_prediction(
                    model, model_path, arrays["x_test"], config
                )
                losses = (prediction - arrays["y_test"]) ** 2
                mse = float(np.mean(losses))
                mcse = (
                    float(np.std(losses, ddof=1) / math.sqrt(len(losses)))
                    if len(losses) > 1
                    else None
                )
                if mcse is None:
                    raise ValueError(
                        "Independent evaluation requires at least two rows for MCSE"
                    )
                row = {
                    **row_base,
                    "kind": "independent_observation_evaluation",
                    "budget_steps": case["steps"],
                    "actual_iterations": receipt["iterations"],
                    "test_mse": mse,
                    "test_raw_mse": mse * normalization["scale"] ** 2,
                    "test_mcse": mcse,
                    "test_raw_mcse": mcse * normalization["scale"] ** 2,
                    "test_rows": len(losses),
                    "empirical_test_below_two_direction_floor": mse
                    < row_base["two_direction_normalized_floor"],
                }
                test_rows.append(row)
                path = attempt / ("prediction_" + case["id"] + ".npz")
                _save_arrays(path, {"prediction": prediction})
                write(
                    path.with_suffix(".json"),
                    {
                        "case": case,
                        "metrics": row,
                        "prediction_sha256": sha(path),
                        "state_sha256": sha(model_path),
                        "replay_max_error": error,
                        "all_study_models_binding_sha256": binding_sha,
                        "scope": "Fixed-model independent observation MCSE only; no model selection or rigorous risk certificate.",
                    },
                )
            if len(test_rows) != len(completed):
                raise ValueError("Independent evaluation omitted a declared model")
            _export_evaluation(attempt, endpoint_rows, test_rows, config)
            # Rehash every source, observation, model and milestone before publishing.
            verify(root)
            completed_stage(root, "study")
            _check_bindings(root, input_bindings)
            # Exclusive hard links expose a concrete complete table and archives.
            # The completion marker is published last; interrupted attempts survive.
            published = {}
            inventory = artifact_inventory(root, attempt)
            for relative in inventory["artifacts"]:
                path = root / relative
                if path.name in ("input_bindings.json", "invocation.json"):
                    continue
                if path.parent != attempt:
                    raise ValueError(
                        "Unexpected nested evaluation publication artifact"
                    )
                target = directory / path.name
                os.link(path, target)
                published[str(target.relative_to(root))] = sha(target)
            _check_bindings(root, input_bindings)
            _check_bindings(root, inventory["artifacts"])
            result = {
                "utc": now(),
                "status": "complete",
                "models": len(test_rows),
                "endpoint_rows": len(endpoint_rows),
                "test_rows": len(test_rows),
                "evaluation_n": config["evaluation_n"],
                "seconds": time.monotonic() - started,
                "csv_path": str((directory / "all_endpoints.csv").relative_to(root)),
                "summary_path": str((directory / "summary.md").relative_to(root)),
                "input_bindings": input_bindings,
                "artifacts": {**inventory["artifacts"], **published},
                "artifact_inventory_diagnostics": inventory["diagnostics"],
                "scope": "Independent observations on fixed DEVELOPMENT teachers. All conditions retained; no validation/TEST selection, fitted exponents, rigorous confidence intervals or fresh-response confirmation.",
            }
            write(marker, result)
            return result
        except Exception as error:
            write(
                attempt / "failure.json",
                {
                    "utc": now(),
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                    "seconds": time.monotonic() - started,
                },
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("initialize", "profile", "study", "evaluate"), required=True
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "initialize":
        initialize(args.root)
    elif args.stage == "evaluate":
        print(json.dumps(evaluate_complete_study(args.root), indent=2), flush=True)
    else:
        run_stage(args.root, args.stage, args.workers)


if __name__ == "__main__":
    main()

"""Development-only nonpolynomial sharing study with immutable run receipts.

The pilot exposes TRAIN and validation only. It cannot release confirmation
data or support an exponent claim. Full projections are the ordinary block
MLP with a growing number of directions; rank2 is a shared linear bottleneck.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import itertools
import json
import multiprocessing
import os
import resource
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from . import nonpolynomial_sharing_tasks as tasks
from . import radial_width_learning as width
from . import rank_learning as base
from . import rank_orthogonal_v2 as optimizer

SOURCE_NAMES = (
    "nonpolynomial_sharing_campaign.py",
    "nonpolynomial_sharing_tasks.py",
    "radial_width_learning.py",
    "rank_learning.py",
    "rank_orthogonal_v2.py",
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


def configuration():
    return {
        "schema": "nonpolynomial_sharing_development_v1",
        "families": ["fourier_ridges", "smooth_bumps"],
        "dimensions": [3, 32],
        "geometries": ["full", "rank2", "width2", "width3", "width8"],
        "blocks": 4,
        "intrinsic_rank": 2,
        "directions": 8,
        "teachers": [601, 602],
        "teacher_seed_base": 2026151001,
        "observation_seed_base": 2026152001,
        "model_seed_base": 2026153001,
        "ceilings": [1441, 2881],
        "train_n": 2048,
        "validation_n": 4096,
        "steps": 150,
        "methods": ["random", "absolute_second_moment"],
        "ridge": 1e-6,
        "profile": {
            "teacher": 600,
            "family": "fourier_ridges",
            "geometries": ["full", "rank2", "width2", "width8"],
            "ceiling": 2881,
            "train_n": 8192,
            "steps": 6,
        },
        "profile_gate": {"maximum_case_seconds": 900, "maximum_case_rss_gib": 20},
        "audit": {"forward_rtol": 3e-6, "forward_atol": 5e-12},
        "scope": "Development only, Gaussian inputs, known supplied blocks, unknown "
        "nonpolynomial response profiles. No private planes or response parameters "
        "enter fitting. No TEST release, automatic confirmation, or automatic exponent "
        "claim. Fixed ridge and equal two-initializer opportunity; all cases retained.",
    }


def grid(config, stage):
    if stage not in ("profile", "screen"):
        raise ValueError("Only development profile and screen stages are available")
    if stage == "profile":
        p = config["profile"]
        cells = itertools.product(
            [p["family"]],
            config["dimensions"],
            [p["teacher"]],
            p["geometries"],
            [p["ceiling"]],
            ["absolute_second_moment"],
        )
        n, steps = p["train_n"], p["steps"]
    else:
        cells = itertools.product(
            config["families"],
            config["dimensions"],
            config["teachers"],
            config["geometries"],
            config["ceilings"],
            config["methods"],
        )
        n, steps = config["train_n"], config["steps"]
    result = []
    for family, d, teacher, geometry, ceiling, method in cells:
        branches = width.branches_under_ceiling(
            ceiling, geometry, raw_blocks=config["blocks"], inputs_per_block=d
        )
        inventory = width.parameter_inventory(
            branches, geometry, raw_blocks=config["blocks"], inputs_per_block=d
        )
        result.append(
            {
                "id": f"{family}_d{d}_t{teacher}_{geometry}_p{ceiling}_{method}",
                "stage": stage,
                "family": family,
                "dimension": d,
                "teacher": teacher,
                "geometry": geometry,
                "ceiling": ceiling,
                "method": method,
                "train_n": n,
                "steps": steps,
                "inventory": inventory,
            }
        )
    return result


def initialize(root, config=None):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = config or configuration()
    source = root / "source"
    source.mkdir()
    (source / "__init__.py").write_text(
        '"""Immutable standalone campaign package."""\n'
    )
    for name in SOURCE_NAMES:
        shutil.copyfile(Path(__file__).with_name(name), source / name)
    write(root / "config.json", config)
    write(
        root / "manifest.json",
        {stage: grid(config, stage) for stage in ("profile", "screen")},
    )
    bindings = {
        str(path.relative_to(root)): sha(path)
        for path in [
            root / "config.json",
            root / "manifest.json",
            *sorted(source.iterdir()),
        ]
    }
    write(
        root / "initialized.json",
        {
            "utc": now(),
            "bindings": bindings,
            "scope": config["scope"],
            "confirmation_released": False,
        },
    )
    return root


def verify(root):
    root = Path(root)
    receipt = read(root / "initialized.json")
    for name, digest in receipt["bindings"].items():
        if sha(root / name) != digest:
            raise ValueError(f"Frozen input changed: {name}")
    loaded_sources = {
        "nonpolynomial_sharing_campaign.py": __file__,
        "nonpolynomial_sharing_tasks.py": tasks.__file__,
        "radial_width_learning.py": width.__file__,
        "rank_learning.py": base.__file__,
        "rank_orthogonal_v2.py": optimizer.__file__,
    }
    for name, loaded_path in loaded_sources.items():
        if sha(loaded_path) != receipt["bindings"]["source/" + name]:
            raise ValueError(f"Executing source differs from frozen campaign: {name}")
    return receipt


def estimate_bank(x, y, method, seed):
    """A generic TRAIN-only second-moment heuristic, not a radial oracle."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.ndim != 3 or x.shape[-1] < 2 or y.shape != (len(x),) or len(x) < 4:
        raise ValueError("Expected at least four [N,K,d] TRAIN rows and scalar labels")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite TRAIN arrays")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 719]))
    banks, spectra = [], []
    for block in range(x.shape[1]):
        if method == "random":
            vectors = np.linalg.qr(rng.normal(size=(x.shape[2], 2)))[0]
        elif method == "absolute_second_moment":
            centered_y = y - y.mean()
            matrix = np.einsum(
                "ni,nj,n->ij", x[:, block], x[:, block], centered_y
            ) / len(x)
            eigenvalues, eigenvectors = np.linalg.eigh((matrix + matrix.T) / 2)
            order = np.argsort(-np.abs(eigenvalues), kind="stable")
            vectors = eigenvectors[:, order[:2]]
            spectra.append(eigenvalues.tolist())
        else:
            raise ValueError("Unknown generic initializer")
        signs = np.sign(vectors[np.argmax(np.abs(vectors), axis=0), np.arange(2)])
        banks.append(vectors * np.where(signs == 0, 1, signs))
    receipt = {
        "method": method,
        "seed": int(seed),
        "eigenvalues": spectra,
        **width.train_fingerprints(x, y),
        "scope": "TRAIN-only heuristic. No known response, private directions or covariance "
        "oracle. Absolute eigenvalue order allows either sign; no consistency theorem asserted.",
    }
    return np.asarray(banks), receipt


def independent_prediction(state_path, raw_x):
    """NumPy graph replay from serialized parameters, with no production forward."""
    with np.load(state_path, allow_pickle=False) as a:
        arrays = {k: a[k].copy() for k in a.files}
    metadata = json.loads(str(arrays.pop("metadata_json")))
    mapping = metadata["initialization_receipt"][width.WIDTH_METADATA_KEY][
        "node_to_raw"
    ]
    capacities = metadata["capacities"]
    group = np.repeat(np.arange(len(capacities)), capacities)
    if not np.array_equal(arrays["group"], group):
        raise ValueError("Serialized branch group differs from declared capacities")
    x = np.asarray(raw_x, dtype=np.float64)
    result = np.full(len(x), float(arrays["bias"]), dtype=np.float64)
    offset = 0
    for node, capacity in enumerate(capacities):
        sl = slice(offset, offset + capacity)
        if metadata["architecture"] == "full":
            pre = x[:, mapping[node]] @ arrays["projection"][sl].T
        else:
            # Ordinary linear bottleneck followed by branch affine maps.
            projected = x[:, mapping[node]] @ arrays["basis"][node]
            pre = projected @ arrays["branch_coefficients"][sl].T
        pre += arrays["threshold"][sl]
        if metadata["family"] != "relu":
            raise ValueError("This pilot deliberately qualifies ReLU only")
        weights = arrays["internal_readout"][sl] * arrays["soma_readout"][node]
        result += np.maximum(pre, 0) @ weights / np.sqrt(len(capacities))
        offset += capacity
    return result


def dataset_key(case):
    return (
        f"{case['family']}_d{case['dimension']}_t{case['teacher']}_n{case['train_n']}"
    )


def prepare_data(root, config, cases):
    """Coordinator-only teacher evaluation. Fitting receives archived x/y arrays."""
    seen = set()
    for case in cases:
        key = dataset_key(case)
        if key in seen:
            continue
        seen.add(key)
        path = root / "data" / (key + ".npz")
        sidecar = path.with_suffix(".json")
        private_path = root / "private_teachers" / (key + ".json")
        present = [p.exists() for p in (path, sidecar, private_path)]
        if any(present) and not all(present):
            raise ValueError(
                "Incomplete dataset release preserved; use a new reviewed output root"
            )
        if all(present):
            saved = read(sidecar)
            if (
                sha(path) != saved["sha256"]
                or sha(private_path) != saved["private_teacher_sha256"]
            ):
                raise ValueError("Archived dataset changed")
            continue
        task_config = tasks.SharingTaskConfig(
            family=case["family"],
            ambient_dim=case["dimension"],
            intrinsic_rank=config["intrinsic_rank"],
            blocks=config["blocks"],
            directions=config["directions"],
            support="known",
        )
        teacher_seed = config["teacher_seed_base"] + case["teacher"]
        teacher = tasks.NonpolynomialSharingTeacher(task_config, teacher_seed)
        observation_seed = config["observation_seed_base"] + case["teacher"]
        arrays, receipt = tasks.generate_dataset(
            teacher,
            observation_seed=observation_seed,
            train_n=case["train_n"],
            validation_n=config["validation_n"],
            test_n=0,
        )
        if any("test" in key and len(value) for key, value in arrays.items()):
            raise ValueError("Development campaign must not release TEST observations")
        arrays = {key: value for key, value in arrays.items() if "test" not in key}
        write(private_path, teacher.specification())
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        write(
            path.with_suffix(".json"),
            {
                "sha256": sha(path),
                "utc": now(),
                "receipt": receipt,
                "teacher_seed": teacher_seed,
                "private_teacher_sha256": sha(private_path),
                "scope": "Development TRAIN/validation; no held-out confirmation data.",
            },
        )


def verify_case(root, case):
    case_root = root / case["stage"] / case["id"]
    receipt = read(case_root / "complete.json")
    if receipt["case"] != case:
        raise ValueError("Completed case configuration changed")
    for name, digest in receipt["outputs"].items():
        if sha(case_root / name) != digest:
            raise ValueError(f"Completed case output changed: {name}")
    data_path = root / "data" / (dataset_key(case) + ".npz")
    if sha(data_path) != receipt["dataset_sha256"]:
        raise ValueError("Completed case dataset changed")
    if sha(data_path.with_suffix(".json")) != receipt["dataset_receipt_sha256"]:
        raise ValueError("Completed case dataset receipt changed")
    private_path = root / "private_teachers" / (dataset_key(case) + ".json")
    if (
        sha(private_path)
        != read(data_path.with_suffix(".json"))["private_teacher_sha256"]
    ):
        raise ValueError("Completed case private provenance changed")
    return receipt


def run_case(root_string, case):
    root = Path(root_string)
    verify(root)
    torch.set_num_threads(1)
    config = read(root / "config.json")
    case_root = root / case["stage"] / case["id"]
    case_root.mkdir(parents=True, exist_ok=True)
    with (case_root / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (case_root / "complete.json").exists():
            return verify_case(root, case)
        attempts = sorted(case_root.glob("attempt_*"))
        if any((p / "failure.json").exists() for p in attempts):
            raise RuntimeError(
                "Scientific/numerical failure retained; no automatic retry"
            )
        attempt = case_root / f"attempt_{len(attempts):03d}"
        attempt.mkdir()
        write(
            attempt / "invocation.json",
            {
                "utc": now(),
                "case": case,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "pid": os.getpid(),
                "interrupted_attempts_preserved": [p.name for p in attempts],
            },
        )
        started = time.monotonic()
        try:
            data_path = root / "data" / (dataset_key(case) + ".npz")
            dataset_receipt = read(data_path.with_suffix(".json"))
            if sha(data_path) != dataset_receipt["sha256"]:
                raise ValueError("Dataset changed before fit")
            normalization = dataset_receipt["receipt"]["normalization"]
            with np.load(data_path, allow_pickle=False) as arrays:
                x, y = arrays["x_train"].copy(), arrays["y_train"].copy()
                vx, vy = arrays["x_validation"].copy(), arrays["y_validation"].copy()
            seed = config["model_seed_base"] + case["teacher"]
            bank, bank_receipt = estimate_bank(x, y, case["method"], seed)
            model = width.model_under_ceiling(
                case["ceiling"],
                "relu",
                case["geometry"],
                bank=bank,
                train_data=(x, y),
                seed=seed,
                allocation_seed=seed,
                initializer_receipt=bank_receipt,
                base_module=base,
            )
            if model.parameter_count != case["inventory"]["stored_parameters"]:
                raise ValueError("Manifest count differs from allocated model")
            fit_config = base.FitConfig(
                ridge=config["ridge"],
                steps=case["steps"],
                learning_rate=0.5,
                objective_scale=1e4,
                tolerance_grad=1e-10,
                tolerance_change=1e-14,
                history_size=30,
            )
            fitted = optimizer.fit_restarted(
                model.core,
                model.expanded_inputs(x),
                y,
                fit_config,
                stages=3,
                rebalance=True,
                output_dir=attempt / "fit",
            )
            state_path = attempt / "model.npz"
            width.save_state(model, state_path)
            predictions = {}
            metrics = {}
            with torch.no_grad():
                for name, inputs, labels in [("train", x, y), ("validation", vx, vy)]:
                    prediction = np.concatenate(
                        [
                            model(inputs[start : start + 512]).numpy()
                            for start in range(0, len(inputs), 512)
                        ]
                    )
                    replay = independent_prediction(state_path, inputs)
                    if (
                        not np.isfinite(prediction).all()
                        or not np.isfinite(replay).all()
                    ):
                        raise FloatingPointError("Nonfinite model prediction")
                    np.testing.assert_allclose(
                        replay,
                        prediction,
                        rtol=config["audit"]["forward_rtol"],
                        atol=config["audit"]["forward_atol"],
                    )
                    predictions[name] = prediction
                    metrics[name + "_mse"] = float(np.mean((prediction - labels) ** 2))
                    metrics[name + "_raw_mse"] = (
                        metrics[name + "_mse"] * normalization["scale"] ** 2
                    )
                    metrics[name + "_replay_max_error"] = float(
                        np.max(np.abs(replay - prediction))
                    )
            prediction_path = attempt / "predictions.npz"
            with prediction_path.open("xb") as handle:
                np.savez_compressed(handle, **predictions)
            result = {
                "case": case,
                "utc": now(),
                "status": "complete",
                "metrics": metrics,
                "seconds": time.monotonic() - started,
                "process_peak_rss_gib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 2**20,
                "rss_scope": "Worker lifetime maximum; may include a prior case in this process.",
                "iterations": fitted["iterations"],
                "closure_calls": fitted["closure_calls"],
                "terminal_unscaled_gradient_l2": fitted[
                    "terminal_unscaled_gradient_l2"
                ],
                "dataset_sha256": sha(data_path),
                "initializer": bank_receipt,
                "dataset_receipt_sha256": sha(data_path.with_suffix(".json")),
                "normalization": normalization,
                "audit_scope": "TRAIN and validation serialized-state forward replay through "
                "an ordinary NumPy block MLP/bottleneck graph; not a full optimizer audit.",
                "outputs": {
                    str(p.relative_to(case_root)): sha(p)
                    for p in sorted(attempt.rglob("*"))
                    if p.is_file()
                },
            }
            write(case_root / "complete.json", result)
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


def profile_gate(root, config):
    marker = root / "profile_complete.json"
    if not marker.exists() or read(marker)["status"] != "complete":
        raise ValueError("A complete passing profile is required before the screen")
    rows = [verify_case(root, c) for c in read(root / "manifest.json")["profile"]]
    limits = config["profile_gate"]
    if any(
        r["seconds"] > limits["maximum_case_seconds"]
        or r["process_peak_rss_gib"] > limits["maximum_case_rss_gib"]
        for r in rows
    ):
        raise ValueError(
            "Profile exceeds frozen runtime/memory gate; create a reviewed new plan"
        )


def run_stage(root, stage, workers):
    root = Path(root).resolve()
    verify(root)
    config = read(root / "config.json")
    if stage == "screen":
        profile_gate(root, config)
    marker = root / f"{stage}_complete.json"
    cases = read(root / "manifest.json")[stage]
    if marker.exists():
        if read(marker)["status"] != "complete":
            raise RuntimeError("Previous failed stage retained")
        return [verify_case(root, c) for c in cases]
    prepare_data(root, config, cases)
    errors = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(run_case, str(root), c): c for c in cases}
        for future in as_completed(futures):
            case = futures[future]
            try:
                result = future.result()
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "status": "complete",
                            "seconds": result["seconds"],
                            "metrics": result["metrics"],
                        }
                    ),
                    flush=True,
                )
            except Exception as error:
                errors.append({"case": case["id"], "error": repr(error)})
                print(json.dumps(errors[-1]), flush=True)
    write(
        marker,
        {
            "utc": now(),
            "status": "failed" if errors else "complete",
            "expected_cases": len(cases),
            "errors": errors,
            "case_receipt_hashes": {
                case["id"]: sha(root / stage / case["id"] / "complete.json")
                for case in cases
                if (root / stage / case["id"] / "complete.json").exists()
            },
            "scope": "Development only. Confirmation not generated or evaluated.",
        },
    )
    if errors:
        raise RuntimeError(f"{len(errors)} retained case failures")
    return [verify_case(root, c) for c in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["initialize", "profile", "screen"], required=True
    )
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.stage == "initialize":
        initialize(args.root)
    else:
        run_stage(args.root, args.stage, args.workers)


if __name__ == "__main__":
    main()

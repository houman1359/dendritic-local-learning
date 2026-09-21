"""Bounded CUDA restart and accumulation checks for the language trainer.

These are within-precision numerical checks on a synthetic token fixture, not
evidence that FP32 and BF16 training are equivalent or that a model learns text.
Run against a frozen source package; existing output directories are refused.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from . import language_train
from .token_data import file_hash
from .train import _atomic_json

TOLERANCES = {
    "restart": {"rtol": 0.0, "atol": 0.0},
    "fp32_accumulation": {"rtol": 5e-5, "atol": 5e-7},
    "bf16_accumulation": {"rtol": 1e-3, "atol": 1e-4},
}


def _fixture(destination: Path) -> Path:
    destination.mkdir()
    tokenizer = destination / "tokenizer.json"
    tokenizer.write_text(json.dumps({"fixture_only": True, "vocab_size": 32}) + "\n")
    splits = {}
    for offset, split in enumerate(("train", "validation", "test")):
        path = destination / f"{split}.tokens.bin"
        ((np.arange(257) + offset) % 32).astype("<u2").tofile(path)
        document_ids = destination / f"{split}.document_ids.txt"
        document_ids.write_text(f"synthetic-{split}\n")
        splits[split] = {
            "path": path.name,
            "dtype": "<u2",
            "num_tokens": 257,
            "sha256": file_hash(path),
            "document_ids_path": document_ids.name,
            "document_ids_sha256": file_hash(document_ids),
        }
    manifest = destination / "manifest.json"
    _atomic_json(
        manifest,
        {
            "schema": "dendritic_scaling_packed_tokens_v1",
            "vocab_size": 32,
            "tokenizer_path": tokenizer.name,
            "tokenizer_sha256": file_hash(tokenizer),
            "splits": splits,
            "provenance": {
                "kind": "synthetic_integer_fixture",
                "natural_language_data": False,
                "tokenizer_is_placeholder": True,
                "purpose": "Systems parity only; split content is not a generalization test",
            },
        },
    )
    return manifest


def _configuration(
    manifest: Path, precision: str, family: str, projection_backend: str = "eager"
) -> dict:
    model = {
        "vocab_size": 32,
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 32,
        "max_position_embeddings": 16,
    }
    if family == "dendritic_shunting":
        model.update(
            ffn_model={
                "family": family,
                "width": 8,
                "branch_factors": [2],
                "contacts_e": 4,
                "contacts_i": 2,
                "projection_backend": projection_backend,
            },
            token_chunk_size=3,
            checkpoint_ffn=True,
        )
    return {
        "id": f"language-parity-{precision}-{family}",
        "phase": "systems_qualification",
        "seed": 19,
        "model": model,
        "data": {
            "manifest_path": str(manifest),
            "sequence_length": 4,
            "unique_train_tokens": 32,
            "validation_tokens": 16,
        },
        "training": {
            "steps": 4,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "lr": 0.001,
            "weight_decay": 0.01,
            "warmup_steps": 2,
            "schedule": "cosine",
            "eval_every": 2,
            "checkpoint_every": 2,
            "device": "cuda",
            "precision": precision,
            "num_threads": 1,
        },
    }


@contextmanager
def _observe_checkpoint_devices(observations: list):
    """Inspect state immediately before the unchanged runner serializes it."""
    original = language_train._atomic_checkpoint

    def observed(path, state):
        states = list(state["optimizer"]["state"].values())
        steps = [value["step"] for value in states]
        moments = [
            value[name] for value in states for name in ("exp_avg", "exp_avg_sq")
        ]
        observations.append(
            {
                "run": path.parent.name,
                "step": state["step"],
                "parameter_states": len(states),
                "step_devices": sorted({str(value.device) for value in steps}),
                "moment_devices": sorted({str(value.device) for value in moments}),
                "moment_dtypes": sorted({str(value.dtype) for value in moments}),
                "passed": all(value.device.type == "cpu" for value in steps)
                and all(
                    value.device.type == "cuda" and value.dtype == torch.float32
                    for value in moments
                ),
            }
        )
        return original(path, state)

    language_train._atomic_checkpoint = observed
    try:
        yield
    finally:
        language_train._atomic_checkpoint = original


def _compare(reference, actual, *, rtol: float, atol: float) -> dict:
    """Compare nested checkpoint values on CPU and retain numerical maxima."""
    result = {
        "passed": True,
        "exact": True,
        "rtol": rtol,
        "atol": atol,
        "numeric_leaves": 0,
        "numeric_values": 0,
        "max_absolute_error": 0.0,
        "max_relative_error": 0.0,
        "relative_error_denominator_floor": 1e-30,
        "max_tolerance_ratio": 0.0,
        "worst_absolute_error_path": None,
        "failure_count": 0,
        "failures": [],
    }

    def fail(path, reason):
        result["passed"] = False
        result["exact"] = False
        result["failure_count"] += 1
        if len(result["failures"]) < 12:
            result["failures"].append({"path": path, "reason": reason})

    def numeric(left, right, path, exact=False):
        left, right = left.detach().cpu(), right.detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            fail(
                path,
                f"shape/dtype mismatch: {left.shape}/{left.dtype} vs {right.shape}/{right.dtype}",
            )
            return
        result["numeric_leaves"] += 1
        result["numeric_values"] += left.numel()
        if not left.numel():
            return
        if not (torch.isfinite(left).all() and torch.isfinite(right).all()):
            fail(path, "nonfinite values")
            return
        if not torch.equal(left, right):
            result["exact"] = False
        expected, observed = left.double(), right.double()
        difference = (observed - expected).abs()
        absolute = difference.max().item()
        relative = (difference / expected.abs().clamp_min(1e-30)).max().item()
        threshold = (
            torch.zeros_like(expected) if exact else atol + rtol * expected.abs()
        )
        ratio = (difference / threshold.clamp_min(1e-30)).max().item()
        if absolute > result["max_absolute_error"]:
            result["max_absolute_error"] = absolute
            result["worst_absolute_error_path"] = path
        result["max_relative_error"] = max(result["max_relative_error"], relative)
        result["max_tolerance_ratio"] = max(result["max_tolerance_ratio"], ratio)
        if not torch.all(difference <= threshold):
            fail(path, f"numeric tolerance exceeded; max absolute error {absolute:.9g}")

    def visit(left, right, path):
        if isinstance(left, torch.Tensor):
            if not isinstance(right, torch.Tensor):
                fail(path, "tensor type mismatch")
            else:
                numeric(left, right, path, exact=not left.is_floating_point())
        elif isinstance(left, np.ndarray):
            if not isinstance(right, np.ndarray):
                fail(path, "array type mismatch")
            else:
                numeric(
                    torch.from_numpy(left),
                    torch.from_numpy(right),
                    path,
                    exact=not np.issubdtype(left.dtype, np.floating),
                )
        elif isinstance(left, dict):
            if not isinstance(right, dict) or left.keys() != right.keys():
                fail(path, "dictionary keys differ")
            else:
                for key in left:
                    visit(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (tuple, list)):
            if type(left) is not type(right) or len(left) != len(right):
                fail(path, "sequence type/length mismatch")
            else:
                for index, (a, b) in enumerate(zip(left, right)):
                    visit(a, b, f"{path}[{index}]")
        elif isinstance(left, float):
            if (
                not isinstance(right, float)
                or not math.isfinite(left)
                or not math.isfinite(right)
            ):
                fail(path, "nonfinite scalar or scalar type mismatch")
            else:
                numeric(
                    torch.tensor(left, dtype=torch.float64),
                    torch.tensor(right, dtype=torch.float64),
                    path,
                )
        elif type(left) is not type(right) or left != right:
            fail(path, "exact metadata mismatch")

    visit(reference, actual, "root")
    return result


def _load_run(directory: Path):
    state = torch.load(
        directory / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    metrics = [
        json.loads(line)
        for line in (directory / "metrics.jsonl").read_text().splitlines()
    ]
    return state, metrics


def _comparison(reference, actual, tolerance) -> dict:
    a, metrics_a = reference
    b, metrics_b = actual
    components = {
        name: _compare(a[name], b[name], **tolerance) for name in ("model", "optimizer")
    }
    components["metrics"] = _compare(metrics_a, metrics_b, **tolerance)
    for name in (
        "sampler",
        "torch_rng",
        "numpy_rng",
        "python_rng",
        "cuda_rng",
        "step",
        "processed_targets",
        "data",
        "source_sha256",
        "runtime_versions",
        "ffn_initialization",
    ):
        components[name] = _compare(a[name], b[name], **TOLERANCES["restart"])
    return {
        "status": (
            "passed"
            if all(value["passed"] for value in components.values())
            else "failed"
        ),
        "exact": all(value["exact"] for value in components.values()),
        "components": components,
    }


def run_checks(
    output_dir: str | Path,
    *,
    precisions=("fp32", "bf16"),
    projection_backend="eager",
    ffn_init_target_rms=None,
) -> dict:
    if (
        not precisions
        or len(set(precisions)) != len(precisions)
        or set(precisions) - {"fp32", "bf16"}
    ):
        raise ValueError("Choose distinct supported precisions")
    if projection_backend not in {"eager", "recompute", "triton_transposed"}:
        raise ValueError("Unsupported qualification backend")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "dendritic_scaling_language_cuda_checks_v1",
        "status": "running",
        "source_sha256": file_hash(__file__),
        "tolerances": copy.deepcopy(TOLERANCES),
        "precisions": list(precisions),
        "projection_backend": projection_backend,
        "ffn_init_target_rms": ffn_init_target_rms,
        "scope": "Within-precision CUDA parity on synthetic token IDs; no FP32-versus-BF16 equivalence claim",
        "checkpoint_observation": "Read-only observer before runner serialization; comparisons load all tensors on CPU",
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda},
        "cells": [],
    }
    report_path = output / "report.json"
    _atomic_json(report_path, report)
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            report.update(status="unsupported", reason="CUDA is unavailable")
            return report
        report["runtime"].update(
            device_name=torch.cuda.get_device_name(),
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
        manifest = _fixture(output / "tokens")
        for precision in precisions:
            for family in ("native_swiglu", "dendritic_shunting"):
                cell = {
                    "precision": precision,
                    "family": family,
                    "status": "running",
                    "optimizer_device_observations": [],
                }
                report["cells"].append(cell)
                _atomic_json(report_path, report)
                if precision == "bf16" and not report["runtime"]["bf16_supported"]:
                    cell.update(
                        status="unsupported", reason="CUDA device does not support BF16"
                    )
                    continue
                cell_dir = output / f"{precision}_{family}"
                cell_dir.mkdir()
                config = _configuration(manifest, precision, family, projection_backend)
                if ffn_init_target_rms is not None:
                    config["ffn_initialization"] = {
                        "mode": "train_batch_rms",
                        "target_rms": ffn_init_target_rms,
                        "training_windows": 2,
                    }
                cell["config"] = config
                continuous = cell_dir / "continuous"
                restarted = cell_dir / "restarted"
                full_batch = cell_dir / "full_batch"
                try:
                    with _observe_checkpoint_devices(
                        cell["optimizer_device_observations"]
                    ):
                        language_train.run_experiment(config, continuous)
                        interruption = language_train.run_experiment(
                            config, restarted, stop_after_steps=3
                        )
                        if interruption["status"] != "interrupted":
                            raise RuntimeError(
                                "Expected an interrupted run after step three"
                            )
                        language_train.run_experiment(config, restarted, resume=True)
                        full_config = copy.deepcopy(config)
                        full_config["training"].update(
                            batch_size=4, gradient_accumulation_steps=1
                        )
                        language_train.run_experiment(full_config, full_batch)
                    reference = _load_run(continuous)
                    cell["restart"] = _comparison(
                        reference, _load_run(restarted), TOLERANCES["restart"]
                    )
                    cell["accumulation"] = _comparison(
                        reference,
                        _load_run(full_batch),
                        TOLERANCES[f"{precision}_accumulation"],
                    )
                    observations = cell["optimizer_device_observations"]
                    observed_runs = {
                        entry["run"]
                        for entry in observations
                        if entry["parameter_states"]
                    }
                    devices_passed = observed_runs == {
                        "continuous",
                        "restarted",
                        "full_batch",
                    } and all(entry["passed"] for entry in observations)
                    cell["optimizer_devices_status"] = (
                        "passed" if devices_passed else "failed"
                    )
                    cell["status"] = (
                        "passed"
                        if devices_passed
                        and all(
                            cell[name]["status"] == "passed"
                            for name in ("restart", "accumulation")
                        )
                        else "failed"
                    )
                    del reference
                except Exception as error:
                    cell.update(
                        status="failed",
                        failure={"type": type(error).__name__, "message": str(error)},
                    )
                _atomic_json(report_path, report)
        report["status"] = (
            "passed"
            if all(cell["status"] == "passed" for cell in report["cells"])
            else "failed"
        )
    except Exception as error:
        report.update(
            status="failed",
            failure={"type": type(error).__name__, "message": str(error)},
        )
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "bf16"), action="append")
    parser.add_argument(
        "--projection-backend",
        choices=("eager", "recompute", "triton_transposed"),
        default="eager",
    )
    parser.add_argument("--ffn-init-target-rms", type=float)
    args = parser.parse_args(argv)
    report = run_checks(
        args.output_dir,
        precisions=args.precision or ("fp32", "bf16"),
        projection_backend=args.projection_backend,
        ffn_init_target_rms=args.ffn_init_target_rms,
    )
    print(
        json.dumps(
            {"status": report["status"], "report": str(args.output_dir / "report.json")}
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Common fixed-horizon training loop for parameter-matched scaling studies.

Run ``python -m dendritic_modeling.scaling.train --config run.json --output-dir out``.
The runner never selects checkpoints on test data and never silently resumes an
incomplete run. A completed identical configuration is returned idempotently.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset

from .data import build_datasets, canonical_hash, derived_seed


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _evaluate(
    model: nn.Module, dataset: TensorDataset, batch_size: int, device: torch.device
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    loss_sum, correct = 0.0, 0
    try:
        with torch.no_grad():
            for start in range(0, len(dataset), batch_size):
                x, y = (
                    tensor[start : start + batch_size].to(device)
                    for tensor in dataset.tensors
                )
                logits = model(x)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("Nonfinite evaluation logits")
                loss = F.cross_entropy(logits, y, reduction="sum")
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite evaluation loss")
                loss_sum += float(loss.item())
                correct += int((logits.argmax(dim=1) == y).sum().item())
    finally:
        model.train(was_training)
    return {"loss": loss_sum / len(dataset), "accuracy": correct / len(dataset)}


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def run_experiment(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Train one model, preserving parameter/data accounting and atomic receipts."""
    config = copy.deepcopy(config)
    config_sha256 = canonical_hash(config)
    output_dir = Path(output_dir)
    receipt_path = output_dir / "receipt.json"
    if output_dir.exists():
        if receipt_path.is_file():
            existing = json.loads(receipt_path.read_text())
            if (
                existing.get("config_sha256") == config_sha256
                and existing.get("status") == "completed"
            ):
                return existing
        raise FileExistsError(
            f"Refusing to overwrite {output_dir}: it is incomplete or has another configuration. "
            "Choose a new run directory."
        )

    training = config.get("training", {})
    steps = _positive_int(training.get("steps", 100), "training.steps")
    batch_size = _positive_int(training.get("batch_size", 64), "training.batch_size")
    eval_every = _positive_int(training.get("eval_every", steps), "training.eval_every")
    num_threads = _positive_int(training.get("num_threads", 1), "training.num_threads")
    diagnostics_every = training.get("diagnostics_every")
    if diagnostics_every is not None:
        diagnostics_every = _positive_int(diagnostics_every, "training.diagnostics_every")
    lr, weight_decay = float(training.get("lr", 1e-3)), float(
        training.get("weight_decay", 0.0)
    )
    if (
        not math.isfinite(lr)
        or lr <= 0
        or not math.isfinite(weight_decay)
        or weight_decay < 0
    ):
        raise ValueError(
            "Learning rate must be positive and weight decay nonnegative; both must be finite"
        )
    schedule = str(training.get("schedule", "constant"))
    if schedule not in {"constant", "cosine"}:
        raise ValueError("training.schedule must be constant or cosine")
    min_lr_fraction = float(training.get("min_lr_fraction", 0.0))
    if not 0 <= min_lr_fraction <= 1:
        raise ValueError("training.min_lr_fraction must be in [0,1]")
    clip_norm = training.get("max_grad_norm")
    if clip_norm is not None and (
        not math.isfinite(float(clip_norm)) or float(clip_norm) <= 0
    ):
        raise ValueError("training.max_grad_norm must be positive and finite")

    seed = int(config.get("seed", 0))
    model_spec = copy.deepcopy(config["model"])
    model_spec.setdefault("seed", seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(output_dir / "config.json", config)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "id": config.get("id", output_dir.name),
        "phase": config.get("phase", "unspecified"),
        "status": "running",
        "seed": seed,
        "config_sha256": config_sha256,
        "config": config,
        "effective_model_spec": model_spec,
        "git_revision": _git_revision(),
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("train.py", "data.py", "models.py") + (
                ("diagnostics.py",) if diagnostics_every is not None else ()
            ) + (
                ("readout_initialization.py",) if "readout_initialization" in training else ()
            )
            if Path(__file__).with_name(name).is_file()
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
    }
    _atomic_json(receipt_path, receipt)
    started = time.perf_counter()
    step = 0
    diagnostics = None
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(num_threads)
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        deterministic = bool(training.get("deterministic", True))
        torch.use_deterministic_algorithms(deterministic)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = deterministic
        device_name = str(training.get("device", "cpu"))
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        receipt["runtime"].update(
            {
                "device": str(device),
                "num_threads": num_threads,
                "deterministic_algorithms": deterministic,
                "device_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else platform.processor()
                ),
                "cuda_version": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        data_started = time.perf_counter()
        bundle = build_datasets(
            config.get("data", {}),
            model_spec,
            include_test=bool(config.get("evaluate_test", False)),
        )
        data_loading_seconds = time.perf_counter() - data_started
        # Import lazily so data-generation utilities do not import model code.
        from .models import build_model, model_report

        model_started = time.perf_counter()
        model = build_model(model_spec).to(device)
        model_build_seconds = time.perf_counter() - model_started
        if "readout_initialization" in training:
            from .readout_initialization import initialize_readout

            initialization_started = time.perf_counter()
            receipt["readout_initialization"] = initialize_readout(
                model, bundle.train.tensors[0], training["readout_initialization"]
            )
            receipt["readout_initialization"]["elapsed_seconds"] = time.perf_counter() - initialization_started
        report = dict(model_report(model))
        total_parameters = sum(parameter.numel() for parameter in model.parameters())
        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        if report.get("total_parameters", total_parameters) != total_parameters:
            raise ValueError(
                "model_report total_parameters disagrees with the instantiated whole model"
            )
        planned = config.get("planned_model_report", {}).get("total_parameters")
        if planned is not None and int(planned) != total_parameters:
            raise ValueError(
                f"Planned total_parameters={planned} differs from actual {total_parameters}"
            )
        if "target_parameters" in config:
            target = _positive_int(config["target_parameters"], "target_parameters")
            tolerance = float(config.get("budget_tolerance", 0.02))
            if not math.isfinite(tolerance) or tolerance < 0:
                raise ValueError("budget_tolerance must be finite and nonnegative")
            mismatch = abs(total_parameters - target) / target
            if mismatch > tolerance:
                raise ValueError(
                    f"Actual whole-model parameters {total_parameters} miss target {target} "
                    f"by {mismatch:.3%}, exceeding tolerance {tolerance:.3%}"
                )
        report.update(
            {
                "total_parameters": total_parameters,
                "trainable_parameters": trainable_parameters,
            }
        )
        receipt["model_report"] = report
        if diagnostics_every is not None:
            from .diagnostics import TrainingDiagnostics

            diagnostics = TrainingDiagnostics(
                model, output_dir / "diagnostics.jsonl",
                every=diagnostics_every, steps=steps,
            )
            receipt["diagnostics"] = {
                "path": "diagnostics.jsonl", "every": diagnostics_every,
                "split": "train", "includes_first_and_last": True,
                "gradient_stage": "pre_clipping", "parameter_stage": "pre_update",
                "scope": "Existing branch/hidden/readout outputs and gradients; no extra forward pass. Timing includes diagnostics.",
            }
        receipt["data"] = {
            "dataset": bundle.identity["dataset"],
            "train_size": len(bundle.train),
            "unique_train_examples": len(bundle.train),
            "validation_size": len(bundle.validation),
            "identity": bundle.identity,
            "identity_sha256": bundle.identity["sha256"],
        }
        _atomic_json(receipt_path, receipt)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        sampler = torch.Generator().manual_seed(derived_seed(seed, "training-sampler"))
        order = torch.randperm(len(bundle.train), generator=sampler)
        cursor = 0
        seen = torch.zeros(len(bundle.train), dtype=torch.bool)
        processed_examples = 0
        grad_norm_sum = grad_norm_max = loss_sum = 0.0
        final_grad_norm = 0.0
        missing_gradients_max = missing_gradients_final = 0
        validation_seconds = 0.0
        training_started = time.perf_counter()
        metrics_file = output_dir / "metrics.jsonl"
        with metrics_file.open("x", buffering=1) as metrics_handle:
            evaluation_started = time.perf_counter()
            initial_validation = _evaluate(model, bundle.validation, batch_size, device)
            validation_seconds += time.perf_counter() - evaluation_started
            metrics_handle.write(
                json.dumps(
                    {
                        "step": 0,
                        "horizon_steps": steps,
                        "horizon_fraction": 0.0,
                        "processed_examples": 0,
                        "unique_examples_seen": 0,
                        "validation_loss": initial_validation["loss"],
                        "validation_accuracy": initial_validation["accuracy"],
                        "learning_rate": lr,
                        "schedule": schedule,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            model.train()
            for step in range(1, steps + 1):
                pieces, remaining = [], batch_size
                while remaining:
                    take = min(remaining, len(order) - cursor)
                    pieces.append(order[cursor : cursor + take])
                    cursor += take
                    remaining -= take
                    if cursor == len(order):
                        order = torch.randperm(len(bundle.train), generator=sampler)
                        cursor = 0
                index = torch.cat(pieces)
                seen[index] = True
                x, y = (tensor[index].to(device) for tensor in bundle.train.tensors)
                if schedule == "cosine":
                    progress = (step - 1) / max(1, steps - 1)
                    fraction = min_lr_fraction + (1 - min_lr_fraction) * 0.5 * (
                        1 + math.cos(math.pi * progress)
                    )
                    for group in optimizer.param_groups:
                        group["lr"] = lr * fraction
                optimizer.zero_grad(set_to_none=True)
                if diagnostics is not None:
                    diagnostics.begin_step(step)
                logits = model(x)
                if (
                    logits.shape != (len(y), model_spec["output_dim"])
                    or not torch.isfinite(logits).all()
                ):
                    raise FloatingPointError(
                        f"Invalid or nonfinite logits at step {step}"
                    )
                loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                loss.backward()
                gradient_norms = [
                    p.grad.detach().float().norm()
                    for p in model.parameters()
                    if p.grad is not None
                ]
                missing_gradients_final = sum(
                    p.numel()
                    for p in model.parameters()
                    if p.requires_grad and p.grad is None
                )
                missing_gradients_max = max(
                    missing_gradients_max, missing_gradients_final
                )
                if not gradient_norms:
                    raise RuntimeError("No parameter gradients were produced")
                final_grad_norm = float(torch.stack(gradient_norms).norm().item())
                if not math.isfinite(final_grad_norm):
                    raise FloatingPointError(f"Nonfinite gradients at step {step}")
                if diagnostics is not None:
                    diagnostics.finish_step()
                if clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(clip_norm), error_if_nonfinite=True
                    )
                optimizer.step()
                processed_examples += len(y)
                loss_sum += float(loss.item())
                grad_norm_sum += final_grad_norm
                grad_norm_max = max(grad_norm_max, final_grad_norm)
                if step % eval_every == 0 or step == steps:
                    evaluation_started = time.perf_counter()
                    validation = _evaluate(model, bundle.validation, batch_size, device)
                    validation_seconds += time.perf_counter() - evaluation_started
                    metrics_handle.write(
                        json.dumps(
                            {
                                "step": step,
                                "horizon_steps": steps,
                                "horizon_fraction": step / steps,
                                "processed_examples": processed_examples,
                                "unique_examples_seen": int(seen.sum()),
                                "training_loss": float(loss.item()),
                                "gradient_norm": final_grad_norm,
                                "parameters_missing_gradients": missing_gradients_final,
                                "validation_loss": validation["loss"],
                                "validation_accuracy": validation["accuracy"],
                                "learning_rate": optimizer.param_groups[0]["lr"],
                                "schedule": schedule,
                                "elapsed_seconds": time.perf_counter()
                                - training_started,
                            },
                            allow_nan=False,
                        )
                        + "\n"
                    )
        training_loop_seconds = time.perf_counter() - training_started
        if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
            raise FloatingPointError("Nonfinite terminal model parameters")
        if (
            sum(parameter.numel() for parameter in model.parameters())
            != total_parameters
        ):
            raise ValueError("Whole-model parameter count changed during training")
        terminal_report = dict(model_report(model))
        if report.get("topology_sha256") != terminal_report.get("topology_sha256"):
            raise ValueError("Fixed topology changed during training")
        receipt["terminal_model_report"] = terminal_report
        metrics = {
            "initial_validation_loss": initial_validation["loss"],
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "mean_training_loss": loss_sum / steps,
        }
        if bundle.test is not None:
            test = _evaluate(model, bundle.test, batch_size, device)
            metrics.update(
                {"test_loss": test["loss"], "test_accuracy": test["accuracy"]}
            )
            receipt["data"]["test_size"] = len(bundle.test)
        checkpoint_path = output_dir / "checkpoint.pt"
        temporary_checkpoint = output_dir / ".checkpoint.pt.tmp"
        try:
            torch.save(
                {
                    "schema_version": 1,
                    "config_sha256": config_sha256,
                    "config": config,
                    "effective_model_spec": model_spec,
                    "step": steps,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                },
                temporary_checkpoint,
            )
            os.replace(temporary_checkpoint, checkpoint_path)
        finally:
            temporary_checkpoint.unlink(missing_ok=True)
        receipt.update(
            {
                "status": "completed",
                "metrics": metrics,
                "training": {
                    "steps": steps,
                    "batch_size": batch_size,
                    "processed_examples": processed_examples,
                    "unique_examples_seen": int(seen.sum()),
                    "epochs_equivalent": processed_examples / len(bundle.train),
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "schedule": schedule,
                    "horizon_steps": steps,
                    "min_lr_fraction": min_lr_fraction,
                    "gradient_norm_mean": grad_norm_sum / steps,
                    "gradient_norm_max": grad_norm_max,
                    "gradient_norm_final": final_grad_norm,
                    "max_grad_norm": clip_norm,
                    "parameters_missing_gradients_max": missing_gradients_max,
                    "parameters_missing_gradients_final": missing_gradients_final,
                    "elapsed_seconds": time.perf_counter() - started,
                    "training_elapsed_seconds": training_loop_seconds,
                    "validation_elapsed_seconds": validation_seconds,
                    "optimization_and_batch_transfer_seconds": training_loop_seconds
                    - validation_seconds,
                    "data_loading_seconds": data_loading_seconds,
                    "model_build_seconds": model_build_seconds,
                    "gpu_count": 1 if device.type == "cuda" else 0,
                    "cuda_max_memory_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0
                    ),
                    "cuda_max_memory_reserved_bytes": (
                        torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda"
                        else 0
                    ),
                },
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "metrics": metrics_file.name,
                    "config": "config.json",
                },
            }
        )
        _atomic_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        if diagnostics is not None:
            try:
                diagnostics.record_failure(error, step)
            except Exception as diagnostic_error:
                receipt["diagnostic_failure"] = {
                    "type": type(diagnostic_error).__name__,
                    "message": str(diagnostic_error),
                }
        receipt.update(
            {
                "status": "failed",
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "step": step,
                },
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        _atomic_json(receipt_path, receipt)
        raise
    finally:
        if diagnostics is not None:
            diagnostics.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_experiment(json.loads(args.config.read_text()), args.output_dir)
    print(
        json.dumps(
            {
                "id": receipt["id"],
                "status": receipt["status"],
                "metrics": receipt["metrics"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

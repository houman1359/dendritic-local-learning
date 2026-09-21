"""Single-process causal-LM training with token accounting and exact restart.

Labels from PackedTokenStore are already shifted. Native and custom FFNs use
the same objective, optimizer, sampler, evaluation, and checkpoint protocol.
Distributed execution is explicitly unsupported in this first trainer.
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import canonical_hash
from .language import build_language_model, language_model_report
from .token_data import PackedTokenStore, file_hash
from .train import _atomic_json


def token_loss_sum(logits, labels):
    """Return summed CE and valid target count, without any additional shift."""
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Already-shifted labels must match every logit position")
    count = int((labels != -100).sum())
    if count < 1:
        raise ValueError("A language batch needs at least one valid target")
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss, count


class WindowSampler:
    def __init__(self, size, seed):
        self.size = size
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(size, generator=self.generator)
        self.cursor = 0
        self.seen = torch.zeros(size, dtype=torch.bool)

    def draw(self, count):
        pieces = []
        while count:
            if self.cursor == self.size:
                self.order = torch.randperm(self.size, generator=self.generator)
                self.cursor = 0
            take = min(count, self.size - self.cursor)
            indices = self.order[self.cursor : self.cursor + take]
            self.seen[indices] = True
            pieces.append(indices)
            self.cursor += take
            count -= take
        return torch.cat(pieces)

    def state_dict(self):
        return {
            "generator": self.generator.get_state(),
            "order": self.order,
            "cursor": self.cursor,
            "seen": self.seen,
        }

    def load_state_dict(self, state):
        self.generator.set_state(state["generator"].cpu())
        self.order = state["order"].cpu()
        self.cursor = state["cursor"]
        self.seen = state["seen"].cpu()


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _atomic_checkpoint(path, state):
    descriptor, temporary = tempfile.mkstemp(prefix=".checkpoint.", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def run_experiment(config, output_dir, *, resume=False, stop_after_steps=None):
    """Train to the frozen horizon; explicit invocation controls permit restart.

    stop_after_steps is an absolute completed-step boundary used for deliberate
    interruption checks. It does not change the optimizer schedule or config.
    """
    config = copy.deepcopy(config)
    identity = canonical_hash(config)
    output = Path(output_dir)
    receipt_path, checkpoint_path = output / "receipt.json", output / "checkpoint.pt"
    package = Path(__file__).resolve().parents[1]
    source_hashes = {
        str(path.relative_to(package)): file_hash(path)
        for path in sorted(package.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    versions = {
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    try:
        versions["triton"] = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        versions["triton"] = None
    versions["indexed_backend_environment"] = {
        name: os.environ.get(name)
        for name in (
            "DENDRITIC_ATOMIC_GRADX",
            "DENDRITIC_GX_SETUP_CACHE_MAX",
            "DENDRITIC_GX_SETUP_CACHE_MAX_BYTES",
        )
    }
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
        normalize_indexed_projection_backend,
    )

    ffn_spec = config["model"].get("ffn_model") or {}
    projection_backends = {
        normalize_indexed_projection_backend(
            ffn_spec.get("projection_backend", "eager")
        )
    }
    readout = ffn_spec.get("readout")
    if isinstance(readout, dict) and readout.get("kind") == "indexed":
        projection_backends.add(
            normalize_indexed_projection_backend(
                readout.get("projection_backend", "eager")
            )
        )
    if not projection_backends <= {"eager", "recompute", "triton_transposed"}:
        raise ValueError(
            "Qualification requires an explicit eager, recompute, or triton_transposed backend"
        )
    if (
        "triton_transposed" in projection_backends
        and os.environ.get("DENDRITIC_ATOMIC_GRADX") != "0"
    ):
        raise ValueError(
            "Qualified triton_transposed training requires DENDRITIC_ATOMIC_GRADX=0"
        )
    initialization = config.get("ffn_initialization", {"mode": "preserve"})
    if not isinstance(initialization, dict) or initialization.get("mode") not in {
        "preserve",
        "train_batch_rms",
    }:
        raise ValueError("Unknown FFN initialization recipe")
    allowed = (
        {"mode"}
        if initialization["mode"] == "preserve"
        else {"mode", "target_rms", "training_windows"}
    )
    if set(initialization) - allowed:
        raise ValueError("Unknown FFN initialization options")
    if initialization["mode"] == "train_batch_rms":
        target_rms = float(initialization["target_rms"])
        if not math.isfinite(target_rms) or target_rms <= 0:
            raise ValueError("FFN target RMS must be finite and positive")
        initialization_windows = _positive(
            initialization.get("training_windows", 4), "initialization training_windows"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This language trainer supports one process only")
    if config.get("evaluate_test", False):
        raise ValueError("This qualification trainer does not evaluate final test data")
    if output.exists():
        if not receipt_path.exists():
            raise FileExistsError("Existing output lacks an auditable receipt")
        previous = json.loads(receipt_path.read_text())
        if (
            previous.get("config_sha256") != identity
            or previous.get("source_sha256") != source_hashes
            or previous.get("runtime_versions") != versions
        ):
            raise ValueError(
                "Configuration or training source changed; use a new run directory"
            )
        if previous["status"] == "completed":
            return previous
        if not resume or not checkpoint_path.exists():
            raise FileExistsError(
                "Incomplete run requires explicit resume and a saved checkpoint"
            )
    elif resume:
        raise FileNotFoundError("Cannot resume a nonexistent run")
    training, data = config["training"], config["data"]
    steps = _positive(training["steps"], "steps")
    batch_size = _positive(training["batch_size"], "batch_size")
    accumulation = _positive(
        training.get("gradient_accumulation_steps", 1), "gradient_accumulation_steps"
    )
    eval_every = _positive(training.get("eval_every", steps), "eval_every")
    checkpoint_every = _positive(
        training.get("checkpoint_every", eval_every), "checkpoint_every"
    )
    warmup = training.get("warmup_steps", 0)
    if (
        isinstance(warmup, bool)
        or not isinstance(warmup, int)
        or not 0 <= warmup < steps
    ):
        raise ValueError("warmup_steps must be an integer in [0,steps)")
    lr, decay = float(training["lr"]), float(training.get("weight_decay", 0.01))
    clip = float(training.get("max_grad_norm", 1.0))
    if (
        not all(math.isfinite(value) for value in (lr, decay, clip))
        or lr <= 0
        or decay < 0
        or clip <= 0
    ):
        raise ValueError(
            "Invalid optimizer learning rate, weight decay, or gradient clipping"
        )
    schedule = training.get("schedule", "cosine")
    if schedule not in {"constant", "cosine"}:
        raise ValueError("Unknown learning-rate schedule")
    limit = (
        steps
        if stop_after_steps is None
        else _positive(stop_after_steps, "stop_after_steps")
    )
    limit = min(limit, steps)
    device = torch.device(training.get("device", "cpu"))
    precision = training.get("precision", "fp32")
    if precision not in {"fp32", "bf16"} or (
        precision == "bf16" and device.type != "cuda"
    ):
        raise ValueError("Use fp32, or bf16 autocast on CUDA with FP32 parameters")
    seed = int(config.get("seed", 0))
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "config.json", config)
    receipt = {
        "schema": "dendritic_scaling_language_training_v1",
        "status": "running",
        "id": config.get("id", output.name),
        "phase": config.get("phase", "qualification"),
        "config": config,
        "config_sha256": identity,
        "source_sha256": source_hashes,
        "seed": seed,
        "single_process": True,
        "test_evaluated": False,
        "runtime_versions": versions,
    }
    _atomic_json(receipt_path, receipt)
    started = time.perf_counter()
    step = processed = 0
    try:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.set_num_threads(_positive(training.get("num_threads", 1), "num_threads"))
        random.seed(seed)
        np.random.seed(seed % 2**32)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but unavailable")
            torch.cuda.reset_peak_memory_stats(device)
        train = PackedTokenStore(
            data["manifest_path"],
            "train",
            data["sequence_length"],
            data.get("unique_train_tokens"),
        )
        validation = PackedTokenStore(
            data["manifest_path"],
            "validation",
            data["sequence_length"],
            data.get("validation_tokens"),
        )
        if config["model"]["vocab_size"] != train.vocab_size:
            raise ValueError("Model vocabulary differs from the frozen tokenizer")
        if data["sequence_length"] > config["model"].get(
            "max_position_embeddings", 512
        ):
            raise ValueError("Sequence exceeds the declared context length")
        effective_model_spec = copy.deepcopy(config["model"])
        effective_model_spec.setdefault("seed", seed)
        model = build_language_model(effective_model_spec).to(device)
        report = language_model_report(model)
        if "target_parameters" in config:
            target = _positive(config["target_parameters"], "target_parameters")
            if abs(report["total_parameters"] - target) / target > config.get(
                "budget_tolerance", 0.02
            ):
                raise ValueError("Actual whole-LM parameters miss the declared budget")
        initialization_audit = {"mode": initialization["mode"]}
        if not resume and initialization["mode"] == "train_batch_rms":
            from .language import condition_ffn_output_rms

            if initialization_windows > len(train):
                raise ValueError(
                    "FFN initialization batch exceeds the declared training prefix"
                )
            initialization_indices = list(range(initialization_windows))
            initialization_inputs = train.get_batch(initialization_indices)[
                "input_ids"
            ].to(device)
            initialization_audit.update(
                condition_ffn_output_rms(
                    model, initialization_inputs, target_rms=target_rms
                )
            )
            initialization_audit.update(
                training_window_indices=initialization_indices,
                training_data_identity=train.identity,
                training_input_token_occurrences=initialization_inputs.numel(),
                exposure_note="Training inputs used for initialization only; labels and validation/test examples are not used. Report separately from processed optimization targets.",
            )
            del initialization_inputs
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
        sampler = WindowSampler(len(train), seed + 31337)
        metrics_path = output / "metrics.jsonl"
        elapsed_previous = optimization_seconds = evaluation_seconds = 0.0
        step_seconds = []

        def autocast():
            return torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            )

        def evaluate():
            model.eval()
            total_loss = 0.0
            total_count = 0
            with torch.no_grad(), autocast():
                for offset in range(0, len(validation), batch_size):
                    batch = {
                        key: value.to(device)
                        for key, value in validation.get_batch(
                            list(
                                range(offset, min(offset + batch_size, len(validation)))
                            )
                        ).items()
                    }
                    loss, count = token_loss_sum(
                        model(input_ids=batch["input_ids"]).logits, batch["labels"]
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite validation loss")
                    total_loss += loss.item()
                    total_count += count
            model.train()
            return total_loss / total_count

        def backend_runtime():
            name = "dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels.triton_indexed_gather_transposed"
            module = sys.modules.get(name)
            if module is None:
                return {}
            state = {
                "atomic_grad_input": module.ATOMIC_GRADX,
                "deterministic_fallback_observed": module._det_gx_warned_fallback,
                "grad_input_setup_cache": module.gx_setup_cache_stats(),
            }
            if "triton_transposed" in projection_backends and (
                state["atomic_grad_input"] or state["deterministic_fallback_observed"]
            ):
                raise RuntimeError(
                    "Transposed projection used an unqualified atomic input-gradient path"
                )
            return state

        def log(record):
            with metrics_path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")

        def save_checkpoint():
            _atomic_checkpoint(
                checkpoint_path,
                {
                    "config_sha256": identity,
                    "source_sha256": source_hashes,
                    "runtime_versions": versions,
                    "ffn_initialization": initialization_audit,
                    "data": {
                        "train": train.identity,
                        "validation": validation.identity,
                    },
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "processed_targets": processed,
                    "sampler": sampler.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": np.random.get_state(),
                    "python_rng": random.getstate(),
                    "cuda_rng": (
                        torch.cuda.get_rng_state_all()
                        if device.type == "cuda"
                        else None
                    ),
                    "metrics_bytes": metrics_path.stat().st_size,
                    "validation_loss": validation_loss,
                    "optimization_seconds": optimization_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "per_step_optimization_seconds": step_seconds,
                    "elapsed_seconds": elapsed_previous + time.perf_counter() - started,
                },
            )

        if resume:
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (
                saved["config_sha256"] != identity
                or saved["source_sha256"] != source_hashes
                or saved["runtime_versions"] != versions
                or saved["data"]
                != {"train": train.identity, "validation": validation.identity}
            ):
                raise ValueError(
                    "Checkpoint configuration, source, or data identity changed"
                )
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            sampler.load_state_dict(saved["sampler"])
            step, processed = saved["step"], saved["processed_targets"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            np.random.set_state(saved["numpy_rng"])
            random.setstate(saved["python_rng"])
            if saved["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(
                    [state.cpu() for state in saved["cuda_rng"]]
                )
            with metrics_path.open("r+b") as stream:
                stream.truncate(saved["metrics_bytes"])
            validation_loss = saved["validation_loss"]
            initialization_audit = saved["ffn_initialization"]
            elapsed_previous = saved["elapsed_seconds"]
            optimization_seconds, evaluation_seconds = (
                saved["optimization_seconds"],
                saved["evaluation_seconds"],
            )
            step_seconds = saved["per_step_optimization_seconds"]
            del saved
        else:
            before = time.perf_counter()
            validation_loss = evaluate()
            evaluation_seconds += time.perf_counter() - before
            log({"step": 0, "validation_loss": validation_loss})
            save_checkpoint()

        while step < limit:
            before = time.perf_counter()
            factor = (step + 1) / warmup if step < warmup else 1.0
            if step >= warmup and schedule == "cosine":
                factor = 0.5 * (
                    1 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup - 1))
                )
            for group in optimizer.param_groups:
                group["lr"] = lr * factor
            optimizer.zero_grad(set_to_none=True)
            indices = sampler.draw(batch_size * accumulation)
            batches = [train.get_batch(piece) for piece in indices.split(batch_size)]
            target_count = sum(
                int((batch["labels"] != -100).sum()) for batch in batches
            )
            training_loss = 0.0
            for batch in batches:
                batch = {key: value.to(device) for key, value in batch.items()}
                with autocast():
                    loss_sum, _ = token_loss_sum(
                        model(input_ids=batch["input_ids"]).logits, batch["labels"]
                    )
                    loss = loss_sum / target_count
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                loss.backward()
                training_loss += loss_sum.item() / target_count
            if any(parameter.grad is None for parameter in model.parameters()):
                raise RuntimeError(
                    "A learned language-model parameter is missing its gradient"
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), clip, error_if_nonfinite=True
            )
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_seconds.append(time.perf_counter() - before)
            optimization_seconds += step_seconds[-1]
            step += 1
            processed += target_count
            record = {
                "step": step,
                "training_loss": training_loss,
                "lr": lr * factor,
                "gradient_norm": float(gradient_norm),
                "processed_targets": processed,
            }
            if step % eval_every == 0 or step == steps:
                before = time.perf_counter()
                validation_loss = evaluate()
                evaluation_seconds += time.perf_counter() - before
                record["validation_loss"] = validation_loss
            log(record)
            if step % checkpoint_every == 0 or step == limit:
                save_checkpoint()
        report = language_model_report(model)
        backend_state = backend_runtime()
        receipt.update(
            {
                "status": "completed" if step == steps else "interrupted",
                "ffn_initialization": initialization_audit,
                "model_report": report,
                "data": {"train": train.identity, "validation": validation.identity},
                "metrics": {"validation_loss": validation_loss},
                "training": {
                    "steps_completed": step,
                    "horizon_steps": steps,
                    "processed_targets": processed,
                    "unique_target_positions_seen": int(sampler.seen.sum())
                    * train.sequence_length,
                    "unique_target_positions_available": train.unique_tokens,
                    "global_batch_targets": batch_size
                    * accumulation
                    * train.sequence_length,
                    "optimization_seconds": optimization_seconds,
                    "per_step_optimization_seconds": step_seconds,
                    "evaluation_seconds": evaluation_seconds,
                    "elapsed_seconds": elapsed_previous + time.perf_counter() - started,
                    "gpu_count": int(device.type == "cuda"),
                    "cuda_peak_allocated_bytes": (
                        torch.cuda.max_memory_allocated(device)
                        if device.type == "cuda"
                        else 0
                    ),
                    "cuda_peak_reserved_bytes": (
                        torch.cuda.max_memory_reserved(device)
                        if device.type == "cuda"
                        else 0
                    ),
                },
                "runtime": {
                    "torch": str(torch.__version__),
                    "device": str(device),
                    "precision": precision,
                    "parameter_dtype": "float32",
                    "cuda": torch.version.cuda,
                    "indexed_backend": backend_state,
                    "device_name": (
                        torch.cuda.get_device_name(device)
                        if device.type == "cuda"
                        else "cpu"
                    ),
                },
                "artifacts": {
                    "checkpoint": checkpoint_path.name,
                    "metrics": metrics_path.name,
                },
            }
        )
        _atomic_json(receipt_path, receipt)
        return receipt
    except BaseException as error:
        receipt.update(
            {
                "status": "failed",
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "step": step,
                },
            }
        )
        _atomic_json(receipt_path, receipt)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-steps", type=int)
    args = parser.parse_args(argv)
    receipt = run_experiment(
        json.loads(args.config.read_text()),
        args.output_dir,
        resume=args.resume,
        stop_after_steps=args.stop_after_steps,
    )
    print(
        json.dumps(
            {
                "id": receipt["id"],
                "status": receipt["status"],
                "metrics": receipt.get("metrics"),
            }
        )
    )


if __name__ == "__main__":
    main()

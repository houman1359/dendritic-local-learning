#!/usr/bin/env python
"""Measure end-to-end wall time and CUDA peak memory for one training run."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from dendritic_modeling.scripts.training import train_experiments


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--learning-strategy",
        choices=("local_ca", "standard"),
        required=True,
    )
    parser.add_argument(
        "--profile-scope",
        choices=("training", "end_to_end"),
        default="training",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the resource profile")

    training_phase: dict[str, float | int] = {}
    original_training_phase = train_experiments._run_main_training_phase

    def profiled_training_phase(**kwargs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        phase_started = time.perf_counter()
        result = original_training_phase(**kwargs)
        torch.cuda.synchronize()
        training_phase.update(
            {
                "wall_time_seconds": time.perf_counter() - phase_started,
                "peak_cuda_allocated_bytes": int(
                    torch.cuda.max_memory_allocated()
                ),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
        return result

    if args.profile_scope == "training":
        train_experiments._run_main_training_phase = profiled_training_phase
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    train_args = SimpleNamespace(
        output_dir=str(args.output_dir),
        run_name=None,
        learning_strategy=args.learning_strategy,
        seed=args.seed,
        validate_only=False,
    )
    started = time.perf_counter()
    train_experiments.main(config_path=str(args.config), args=train_args)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    if args.profile_scope == "training":
        profiled_time = float(training_phase["wall_time_seconds"])
        allocated = int(training_phase["peak_cuda_allocated_bytes"])
        reserved = int(training_phase["peak_cuda_reserved_bytes"])
        scope_description = (
            "CUDA peak is reset immediately before the main training phase "
            "and read immediately after it."
        )
    else:
        profiled_time = elapsed
        allocated = int(torch.cuda.max_memory_allocated())
        reserved = int(torch.cuda.max_memory_reserved())
        scope_description = (
            "CUDA peak covers the end-to-end process from prepared config "
            "loading through final evaluation."
        )
    profile = {
        "config": str(args.config.resolve()),
        "learning_strategy": args.learning_strategy,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "profile_scope": args.profile_scope,
        "end_to_end_wall_time_seconds": elapsed,
        "profiled_wall_time_seconds": profiled_time,
        "peak_cuda_allocated_bytes": allocated,
        "peak_cuda_reserved_bytes": reserved,
        "peak_cuda_allocated_mib": allocated / (1024**2),
        "peak_cuda_reserved_mib": reserved / (1024**2),
        "scope": (
            f"{scope_description} End-to-end wall time includes setup, "
            "training, and final evaluation. Analysis and hook settings are "
            "identical across strategies."
        ),
    }
    with (args.output_dir / "resource_profile.json").open("w") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(profile, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

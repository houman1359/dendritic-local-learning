"""Actually optimize equal-capacity models and fixed-metric readout controls.

The modal objective is an exact population objective in a supplied orthonormal
dictionary, not sampled language data. No validation/test selection is involved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.special import zeta
from torch import nn


class ModalModel(nn.Module):
    """Exactly m learned scalars with fixed, invertible coordinate gains."""

    def __init__(self, modes: int, exponent: float):
        super().__init__()
        if modes < 1 or not math.isfinite(exponent) or exponent < 0:
            raise ValueError(
                "Require positive mode count and finite nonnegative exponent"
            )
        self.weight = nn.Parameter(torch.zeros(modes, dtype=torch.float64))
        indices = torch.arange(1, modes + 1, dtype=torch.float64)
        self.register_buffer("gain", indices.pow(-exponent / 2))

    def forward(self):
        return self.weight * self.gain


def target_coefficients(modes: int) -> torch.Tensor:
    index = torch.arange(1, modes + 1, dtype=torch.float64)
    # One fixed target for every budget; signs do not depend on model size.
    return torch.where(index.remainder(2) == 0, -1.0, 1.0) / index


def train_modal(modes: int, exponent: float, steps: int, lr: float, precondition: bool):
    if steps < 1 or not 0 < lr < 1:
        raise ValueError(
            "Require positive steps and learning rate strictly between zero and one"
        )
    model = ModalModel(modes, exponent)
    target = target_coefficients(modes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    tail = float(zeta(2, modes + 1))
    trajectory = []
    started = time.perf_counter()
    checkpoints = set(np.unique(np.geomspace(1, steps, min(33, steps)).astype(int))) | {
        0,
        steps,
    }

    def record(step):
        with torch.no_grad():
            excess = float(((model() - target) ** 2).sum())
        trajectory.append(
            {
                "step": step,
                "accessible_squared_error": excess,
                "inaccessible_squared_error": tail,
                "population_squared_error": tail + excess,
            }
        )

    record(0)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = 0.5 * ((model() - target) ** 2).sum()
        loss.backward()
        if precondition:
            model.weight.grad.div_(model.gain.square())
        optimizer.step()
        if step in checkpoints:
            record(step)
    eigenvalues = torch.ones_like(model.gain) if precondition else model.gain.square()
    expected = target * (1 - (1 - lr * eigenvalues).pow(steps))
    residual = float((model().detach() - expected).abs().max())
    final = trajectory[-1]
    return (
        model,
        trajectory,
        {
            "modes": modes,
            "whole_learned_parameters": sum(p.numel() for p in model.parameters()),
            "gain_exponent": exponent,
            "preconditioned": precondition,
            "steps": steps,
            "lr": lr,
            "elapsed_seconds": time.perf_counter() - started,
            "population_squared_error": final["population_squared_error"],
            "approximation_floor": tail,
            "exact_recurrence_max_abs_residual": residual,
            "predicted_power_for_steps_proportional_to_P": (
                1.0 if precondition or exponent == 0 else min(1.0, 1 / exponent)
            ),
            "scope": "Trained SGD on an exact fixed-dictionary population modal objective; no finite-data generalization claim",
        },
    )


class RankReadout(nn.Module):
    """Linear latent/readout factorization with every affine slot counted."""

    def __init__(self, dimension: int, rank: int, seed: int):
        super().__init__()
        if not 1 <= rank <= dimension:
            raise ValueError("Require 1 <= rank <= dimension")
        generator = torch.Generator().manual_seed(seed)
        self.latent = nn.Parameter(
            torch.randn(rank, dimension, generator=generator, dtype=torch.float64)
            * 0.15
        )
        self.readout = nn.Parameter(
            torch.randn(dimension, rank, generator=generator, dtype=torch.float64)
            * 0.15
        )
        self.bias = nn.Parameter(torch.zeros(dimension, dtype=torch.float64))

    def forward(self, x):
        return (x @ self.latent.T) @ self.readout.T + self.bias


def readout_problem(dimension: int):
    # This is the entire finite population, with mean zero and covariance I.
    eye = torch.eye(dimension, dtype=torch.float64) * math.sqrt(dimension)
    inputs = torch.cat((eye, -eye))
    scale = 1 / torch.arange(1, dimension + 1, dtype=torch.float64)
    return inputs, inputs * scale, scale.square()


def train_readout(dimension: int, rank: int, seed: int, max_iter: int):
    model = RankReadout(dimension, rank, seed)
    inputs, targets, eigenvalues = readout_problem(dimension)
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        max_iter=max_iter,
        tolerance_grad=1e-11,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )
    closures = []
    initial = float(((model(inputs) - targets).detach().square().sum(1)).mean())

    def closure():
        optimizer.zero_grad(set_to_none=True)
        error = (model(inputs) - targets).square().sum(1).mean()
        (error / 2).backward()
        closures.append(float(error.detach()))
        return error / 2

    started = time.perf_counter()
    optimizer.step(closure)
    final = float((model(inputs) - targets).detach().square().sum(1).mean())
    bound = float(eigenvalues[rank:].sum())
    return model, {
        "dimension": dimension,
        "rank": rank,
        "seed": seed,
        "whole_learned_parameters": sum(p.numel() for p in model.parameters()),
        "initial_population_mse": initial,
        "final_population_mse": final,
        "exact_rank_floor": bound,
        "excess_above_floor": final - bound,
        "closure_evaluations": len(closures),
        "closure_mse": closures,
        "elapsed_seconds": time.perf_counter() - started,
        "scope": "Population optimization over all finite support points; fixed identity output metric",
    }


DEFAULT_CONFIG = {
    "modes": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "gain_exponents": [0.0, 2.0, 4.0],
    "steps_per_parameter": 4,
    "learning_rate": 0.5,
    "preconditioning": [False, True],
    "readout_dimension": 8,
    "readout_ranks": [1, 2, 4, 8],
    "readout_seeds": [7103, 7109, 7121],
    "readout_max_iter": 180,
}


def quick_config():
    return {
        **DEFAULT_CONFIG,
        "modes": [16, 32],
        "readout_ranks": [1, 4],
        "readout_seeds": [7103],
        "readout_max_iter": 60,
    }


def run_campaign(output_dir, config=None):
    config = {**DEFAULT_CONFIG, **(config or {})}
    if set(config) != set(DEFAULT_CONFIG):
        raise ValueError("Unknown configuration keys")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite an existing experiment")
    output.mkdir(parents=True, exist_ok=True)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    started = time.perf_counter()
    source = Path(__file__)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "task": "One infinite orthogonal target theta_j=(-1)^(j+1)/j; fixed finite rank-control population",
        "prediction": "Ordinary coordinate scaling changes finite-training slopes; exact preconditioning restores equal field dynamics. Readout rank gives covariance-tail floors.",
        "selection": "No hyperparameter or checkpoint selection; all predeclared terminal outcomes retained",
        "evaluation": "Population objectives, not a sampled-data or held-out confirmation study",
        "device": "cpu",
        "dtype": "float64",
        "torch_version": torch.__version__,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    records, traces, readout = [], [], []
    try:
        for m in config["modes"]:
            for exponent in config["gain_exponents"]:
                for precondition in config["preconditioning"]:
                    model, trace, result = train_modal(
                        m,
                        exponent,
                        config["steps_per_parameter"] * m,
                        config["learning_rate"],
                        precondition,
                    )
                    key = f"m{m}_a{exponent:g}_pre{int(precondition)}"
                    records.append({"case": key, **result})
                    traces.extend({"case": key, **row} for row in trace)
                    torch.save(model.state_dict(), output / (key + ".pt"))
        for rank in config["readout_ranks"]:
            for seed in config["readout_seeds"]:
                model, result = train_readout(
                    config["readout_dimension"], rank, seed, config["readout_max_iter"]
                )
                readout.append(result)
                torch.save(model.state_dict(), output / f"readout_r{rank}_s{seed}.pt")
    finally:
        torch.set_num_threads(previous)
    for name, data in [("conditioning.csv", records), ("trajectories.csv", traces)]:
        with (output / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (output / "readout.json").write_text(json.dumps(readout, indent=2) + "\n")
    summary = {
        "schema": "mechanism_conditioning_population_v1",
        "status": "completed",
        "modal_fits": len(records),
        "readout_fits": len(readout),
        "elapsed_seconds": time.perf_counter() - started,
        "max_modal_recurrence_residual": max(
            r["exact_recurrence_max_abs_residual"] for r in records
        ),
        "max_readout_excess_above_floor": max(r["excess_above_floor"] for r in readout),
        "scope": protocol["evaluation"],
        "artifacts_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config")
    args = parser.parse_args()
    if args.quick and args.config:
        parser.error("Choose either --quick or --config")
    config = (
        json.loads(Path(args.config).read_text())
        if args.config
        else quick_config() if args.quick else None
    )
    print(json.dumps(run_campaign(args.output_dir, config), indent=2))


if __name__ == "__main__":
    main()

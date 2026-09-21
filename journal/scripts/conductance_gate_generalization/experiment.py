"""Exploratory extensions of Figure 5; never overwrite the published cohort.

All rules train the same 24-conductance neuron and affine scalar readout.
Exact paths are a comparator, not an input to either local gate.
The default run is a small pilot, not a confirmatory result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "conductance_local_gate/model.py"
spec = importlib.util.spec_from_file_location("released_local_gate", MODEL_PATH)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

RULES = ("exact", "unit_broadcast", "proportional_gate", "resistance_gate", "swapped_gate")
TASKS = ("graded_teacher", "sensory_selection", "graded_sensory_mixture")
RATES = (0.01, 0.03, 0.1)
OFFSETS = {"train": 11, "validation": 23, "test": 37, "calibration": 41}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset(seed, split, size, task):
    """Independent features per subtree; target definition precedes training.

    External targets average two bounded sensory features in each stream.
    Stream 2 has opposed tuning. Context selects or mixes the two streams.
    The target is not generated from a student's or teacher's conductances.
    """
    rng = np.random.default_rng(np.random.SeedSequence([seed, OFFSETS[split]]))
    z = rng.uniform(-2, 2, (size, 4))
    c = rng.uniform(0, 1, size)
    if task == "sensory_selection":
        c = (c >= 0.5).astype(float)
    x = np.column_stack((np.exp(z), np.exp(-z), 10 * c, 10 * (1 - c)))
    if task == "graded_teacher":
        y = model.forward(model.teacher(seed, opposed=True)[None], x)["output"][0]
    elif task in {"sensory_selection", "graded_sensory_mixture"}:
        features = np.tanh(z)
        first = features[:, :2].mean(axis=1)
        second = -features[:, 2:].mean(axis=1)
        y = (1 - c) * first + c * second
    else:
        raise ValueError(task)
    return x, y, c


def delivery(state, x, rules):
    q = np.ones_like(state["voltage"])
    for i, rule in enumerate(rules):
        if rule == "exact":
            q[i] = model.exact_path(model.subset(state, i))[0]
        elif rule == "unit_broadcast":
            continue
        elif rule in {"proportional_gate", "resistance_gate", "swapped_gate"}:
            g = state["conductance"][i]
            drive = x[:, 8:10]
            if rule == "swapped_gate":
                drive = drive[:, ::-1]
            base = (1 + g[18:22].reshape(2, 2).sum(axis=1)
                    if rule == "resistance_gate" else np.ones(2))
            gate = base / (base + g[16:18] * drive)
            q[i, :, :4] = np.repeat(gate, 2, axis=1)
        else:
            raise ValueError(rule)
    return q


def gradients(theta, readout, x, y, variance, rules):
    state = model.forward(theta, x)
    output = state["output"]
    prediction = readout[:, 0, None] * output + readout[:, 1, None]
    error = (prediction - y[None]) / variance
    local = model.eligibility(state, x)
    q = delivery(state, x, rules)
    gradient = np.mean(error[:, :, None] * readout[:, 0, None, None]
                       * local * q[:, :, model.PARAM_UNIT], axis=1)
    decoder = np.column_stack(((error * output).mean(axis=1), error.mean(axis=1)))
    return np.column_stack((gradient, decoder))


def evaluate(parameters, x, y, variance):
    output = model.forward(parameters[:, :24], x)["output"]
    prediction = parameters[:, 24, None] * output + parameters[:, 25, None]
    return np.mean((prediction - y[None]) ** 2, axis=1) / variance


def run_task(seed, task, steps, size):
    started = time.perf_counter()
    records = [{"rule": rule, "rate": rate} for rate in RATES for rule in RULES]
    rules = [r["rule"] for r in records]
    rates = np.array([r["rate"] for r in records])
    x, y, _ = dataset(seed, "train", size, task)
    vx, vy, _ = dataset(seed, "validation", size // 2, task)
    tx, ty, _ = dataset(seed, "test", 2 * size, task)
    variance = float(np.var(y))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 45678]))
    theta = np.log(model.NOMINAL) + rng.normal(0, 0.4, 24)
    # Same label-free initial slope in all arms; training labels center only
    # the common readout bias. No exact field calibrates a local gate.
    initial_output = model.forward(theta[None], x)["output"][0]
    slope = 1.0 if task == "graded_teacher" else 4.0
    initial = np.r_[theta, slope, y.mean() - slope * initial_output.mean()]
    parameters = np.tile(initial, (len(records), 1))
    m = np.zeros_like(parameters)
    v = np.zeros_like(parameters)
    best = np.full(len(records), np.inf)
    selected = parameters.copy()
    selected_step = np.zeros(len(records), int)
    bound_steps = np.zeros(len(records), int)
    clipped_steps = np.zeros(len(records), int)
    stream = np.random.default_rng(np.random.SeedSequence([seed, 56789]))
    checkpoints = sorted({0, steps} | set(range(64, steps + 1, 64)))
    curves = []
    for step in range(steps + 1):
        if step in checkpoints:
            val = evaluate(parameters, vx, vy, np.var(vy))
            change = val < best
            best[change], selected[change], selected_step[change] = val[change], parameters[change], step
            # Test labels never affect state or hyperparameter selection.
            test = evaluate(parameters, tx, ty, np.var(ty))
            curves.extend(dict(seed=seed, task=task, step=step, **r,
                               validation_nmse=float(val[i]), test_nmse=float(test[i]))
                          for i, r in enumerate(records))
        if step == steps:
            break
        ix = stream.integers(len(x), size=128)
        grad = gradients(parameters[:, :24], parameters[:, 24:], x[ix], y[ix], variance, rules)
        norm = np.linalg.norm(grad, axis=1)
        clipped_steps += norm > 10
        grad *= np.minimum(1, 10 / np.maximum(norm, 1e-30))[:, None]
        m = 0.9 * m + 0.1 * grad
        v = 0.999 * v + 0.001 * grad**2
        parameters -= rates[:, None] * (m / (1 - 0.9**(step + 1))) / (np.sqrt(v / (1 - 0.999**(step + 1))) + 1e-8)
        bound_steps += np.any(np.abs(parameters[:, :24]) > 7, axis=1)
        parameters[:, :24] = np.clip(parameters[:, :24], -7, 7)
        if not np.isfinite(parameters).all():
            raise FloatingPointError((seed, task, step))
    test = evaluate(selected, tx, ty, np.var(ty))
    endpoints = [dict(seed=seed, task=task, **r, selected_step=int(selected_step[i]),
                      validation_nmse=float(best[i]), test_nmse=float(test[i]),
                      bound_steps=int(bound_steps[i]), clipped_steps=int(clipped_steps[i]))
                 for i, r in enumerate(records)]
    selected_rates = []
    for rule in RULES:
        # Per-seed validation selection is explicitly exploratory, not a
        # substitute for development-selected fixed confirmatory rates.
        candidates = [r for r in endpoints if r["rule"] == rule]
        selected_rates.append(min(candidates, key=lambda r: r["validation_nmse"]))
    return dict(seed=seed, task=task, exploratory=True, steps=steps,
                train_size=size, endpoints=endpoints, validation_selected_rates=selected_rates,
                curves=curves, elapsed_seconds=time.perf_counter() - started), selected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=4096)
    p.add_argument("--size", type=int, default=2048)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provenance = {"source_sha256": {str(path): digest(path) for path in (Path(__file__), MODEL_PATH)},
                  "numpy": np.__version__, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "status": "exploratory pilot; not incorporated into manuscript evidence"}
    for task in args.tasks:
        destination = args.output / f"seed_{args.seed}_{task}.json"
        if destination.exists():
            raise FileExistsError(destination)
        result, parameters = run_task(args.seed, task, args.steps, args.size)
        with destination.open("x") as handle:
            json.dump(dict(provenance=provenance, **result), handle, indent=2, allow_nan=False)
        np.savez_compressed(destination.with_suffix(".npz"), selected_parameters=parameters)
        print(task, result["elapsed_seconds"], result["validation_selected_rates"], flush=True)


if __name__ == "__main__":
    main()

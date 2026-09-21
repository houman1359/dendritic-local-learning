"""Controlled Boolean interaction coverage and passive-depth experiments.

This is an explicitly new model family, not a result about existing transformers.
All registered model scalars are learned; supports are fixed structural choices.
Aligned supports use disclosed target structure. No fitting result is an oracle
approximation result, and no finite Boolean task establishes an asymptotic law.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


def boolean_domain(d: int, *, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """Enumerate the uniform domain; row order is fixed and contains no sampling."""
    if not 1 <= d <= 20:
        raise ValueError("Enumeration requires 1 <= d <= 20")
    rows = torch.arange(2**d, dtype=torch.int64)
    return (2 * ((rows[:, None] >> torch.arange(d)) & 1) - 1).to(dtype)


def parity_target(x: torch.Tensor, interactions, coefficients) -> torch.Tensor:
    if len(interactions) != len(coefficients):
        raise ValueError("Each interaction needs one coefficient")
    return sum(
        c * x[:, tuple(a)].prod(dim=1) for a, c in zip(interactions, coefficients)
    )


def validate_supports(supports, d: int) -> torch.Tensor:
    out = torch.as_tensor(supports, dtype=torch.int64)
    if out.ndim != 2 or not out.shape[0] or not out.shape[1]:
        raise ValueError("Supports must be a nonempty S by k matrix")
    if out.min() < 0 or out.max() >= d:
        raise ValueError("Support coordinate outside input domain")
    if any(len(set(row.tolist())) != out.shape[1] for row in out):
        raise ValueError("This experiment requires distinct contacts within each unit")
    return out


def interaction_coverage(supports, interactions) -> list[bool]:
    sets = [set(map(int, row)) for row in supports]
    return [not a or any(set(a) <= r for r in sets) for a in interactions]


def oracle_mse(supports, interactions, coefficients) -> float:
    """Exact uniform-domain residual of arbitrary local functions + affine readout."""
    keys = [tuple(sorted(a)) for a in interactions]
    if len(set(keys)) != len(keys):
        raise ValueError("Combine duplicate Walsh coefficients before computing energy")
    return float(
        sum(
            c * c
            for c, hit in zip(
                coefficients, interaction_coverage(supports, interactions)
            )
            if not hit
        )
    )


def random_coverage_probability(d: int, k: int, q: int) -> float:
    if not 0 <= k <= d or not 0 <= q <= d:
        raise ValueError("Require 0 <= k,q <= d")
    return math.comb(d - q, k - q) / math.comb(d, k) if k >= q else 0.0


def make_supports(d: int, k: int, units: int, seed: int, mode: str, interactions):
    rng = np.random.default_rng(seed)
    out = []
    for s in range(units):
        if mode == "random":
            row = rng.choice(d, size=k, replace=False).tolist()
        elif mode == "aligned":
            required = list(interactions[s % len(interactions)])
            if len(required) > k:
                raise ValueError(
                    "Aligned support cannot contain the target interaction"
                )
            available = [i for i in range(d) if i not in required]
            row = (
                required
                + rng.choice(available, size=k - len(required), replace=False).tolist()
            )
        else:
            raise ValueError(f"Unknown support mode {mode}")
        out.append(sorted(row))
    return validate_supports(out, d)


class LocalReLUPopulation(nn.Module):
    """S biased local one-hidden-layer MLPs and a learned affine soma readout.

    Exactly P = S * (H * (k + 2) + 2) + 1 registered learned scalars.
    Both local output biases and the global bias count despite redundancy.
    A conventional block-sparse factorized ReLU MLP realizes the same class.
    """

    def __init__(self, d: int, supports, hidden: int, seed: int = 0):
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden must be positive")
        supports = validate_supports(supports, d)
        self.register_buffer("supports", supports.clone())
        self.d = d
        s, k = supports.shape
        generator = torch.Generator().manual_seed(seed)
        self.input_weight = nn.Parameter(
            torch.randn(s, hidden, k, generator=generator) / math.sqrt(k)
        )
        self.hidden_bias = nn.Parameter(
            torch.rand(s, hidden, generator=generator) * 2 - 1
        )
        self.local_readout = nn.Parameter(
            torch.randn(s, hidden, generator=generator) / math.sqrt(hidden)
        )
        self.local_bias = nn.Parameter(torch.zeros(s))
        self.soma_readout = nn.Parameter(
            torch.randn(s, generator=generator) / math.sqrt(s)
        )
        self.output_bias = nn.Parameter(torch.zeros(()))

    def hidden_features(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(
            torch.einsum("nsk,shk->nsh", x[:, self.supports], self.input_weight)
            + self.hidden_bias
        )

    def soma_features(self, x: torch.Tensor) -> torch.Tensor:
        return (self.hidden_features(x) * self.local_readout).sum(-1) + self.local_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.soma_features(x) @ self.soma_readout + self.output_bias

    def conventional_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Dense arithmetic realization of identical structural/factorized MLP.

        Zeros outside supports are fixed masks, not additional learned scalars.
        Every learned scalar is shared with the population realization.
        """
        s, h, k = self.input_weight.shape
        dense = self.input_weight.new_zeros(s, h, self.d)
        indices = self.supports[:, None, :].expand(s, h, k)
        dense = dense.scatter(2, indices, self.input_weight)
        features = torch.relu(
            x @ dense.reshape(s * h, self.d).T + self.hidden_bias.reshape(-1)
        )
        somas = (features.reshape(-1, s, h) * self.local_readout).sum(
            -1
        ) + self.local_bias
        return somas @ self.soma_readout + self.output_bias

    def inventory(self) -> dict:
        entries = {
            name: {
                "shape": list(value.shape),
                "count": value.numel(),
                "learned": value.requires_grad,
            }
            for name, value in self.named_parameters()
        }
        total = sum(value.numel() for value in self.parameters())
        s, h, k = self.input_weight.shape
        expected = s * (h * (k + 2) + 2) + 1
        if total != expected:
            raise AssertionError((total, expected))
        return {
            "total": total,
            "formula": "S*(H*(k+2)+2)+1",
            "tensors": entries,
            "fixed_support_indices": self.supports.numel(),
            "conventional_realization_total": total,
        }


def paired_permutation(model: LocalReLUPopulation, x: torch.Tensor, permutation):
    permutation = torch.as_tensor(permutation, dtype=torch.int64)
    if sorted(permutation.tolist()) != list(range(model.d)):
        raise ValueError("Need a coordinate permutation")
    inverse = torch.argsort(permutation)
    paired = copy.deepcopy(model)
    paired.supports.copy_(inverse[model.supports])
    return paired, x[:, permutation]


def construct_walsh_projection(model: LocalReLUPopulation, interactions, coefficients):
    """A disclosed target-dependent witness, never an optimization initialization.

    Each Boolean vertex v has indicator ReLU(v dot x - k + 1). With H >= 2**k,
    these counted, learnable affine weights realize arbitrary local truth tables.
    Assign each covered character to its first covering unit. The returned model
    therefore attains the relaxed Walsh projection using the same finite inventory.
    """
    result = copy.deepcopy(model)
    _, hidden, k = result.input_weight.shape
    if hidden < 2**k:
        raise ValueError("Exact local truth-table witness requires H >= 2**k")
    vertices = boolean_domain(k, dtype=result.input_weight.dtype)
    assigned = set()
    with torch.no_grad():
        result.input_weight.zero_()
        result.hidden_bias.fill_(-1)
        result.local_readout.zero_()
        result.local_bias.zero_()
        result.soma_readout.fill_(1)
        result.output_bias.zero_()
        for unit, support in enumerate(result.supports.tolist()):
            result.input_weight[unit, : 2**k].copy_(vertices)
            result.hidden_bias[unit, : 2**k].fill_(1 - k)
            for index, (interaction, coefficient) in enumerate(
                zip(interactions, coefficients)
            ):
                if index in assigned:
                    continue
                if not interaction:
                    result.output_bias.add_(coefficient)
                    assigned.add(index)
                elif set(interaction) <= set(support):
                    positions = [support.index(i) for i in interaction]
                    result.local_readout[unit, : 2**k].add_(
                        coefficient * vertices[:, positions].prod(dim=1)
                    )
                    assigned.add(index)
    return result


def passive_tree(x: torch.Tensor, raw_edges: torch.Tensor, raw_leaks: torch.Tensor):
    """Four nonnegative leaf voltages, positive passive conductances, no internal drive."""
    g = torch.nn.functional.softplus(raw_edges)
    leak = torch.nn.functional.softplus(raw_leaks)
    left = (g[0] * x[:, 0] + g[1] * x[:, 1]) / (leak[0] + g[0] + g[1])
    right = (g[2] * x[:, 2] + g[3] * x[:, 3]) / (leak[1] + g[2] + g[3])
    return (g[4] * left + g[5] * right) / (leak[2] + g[4] + g[5])


def flattened_passive_tree(
    x: torch.Tensor, raw_edges: torch.Tensor, raw_leaks: torch.Tensor
):
    """Same nine learned scalars, flattened arithmetic with derived path weights."""
    g = torch.nn.functional.softplus(raw_edges)
    leak = torch.nn.functional.softplus(raw_leaks)
    left_den = leak[0] + g[0] + g[1]
    right_den = leak[1] + g[2] + g[3]
    root_den = leak[2] + g[4] + g[5]
    coefficients = (
        torch.stack(
            [
                g[4] * g[0] / left_den,
                g[4] * g[1] / left_den,
                g[5] * g[2] / right_den,
                g[5] * g[3] / right_den,
            ]
        )
        / root_den
    )
    return x @ coefficients


def protocol() -> dict:
    return {
        "version": 1,
        "d": 10,
        "k": 4,
        "hidden": 16,
        "units": [1, 2, 4, 8, 16, 32],
        "interactions": [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 4, 9]],
        "coefficients": [0.5, 0.5, 0.5, 0.5],
        "support_modes": ["random", "aligned"],
        "development_seeds": [1201],
        "confirmation_seeds": [2201, 2203, 2207],
        "train_vertices": 512,
        "validation_vertices": 256,
        "test_vertices": 256,
        "steps": 1000,
        "learning_rate": 0.01,
        "optimizer": "Adam",
        "batch": "full training split",
        "dtype": "float64",
        "checkpoint_selection": "terminal only; validation is recorded, never selects steps or parameters",
        "structure_information": "aligned supports use disclosed target interaction identities; random supports do not",
        "claim_boundary": "fixed finite Boolean-domain population risk; no asymptotic empirical exponent and no unique dendritic advantage",
        "predictions": [
            "Every population's enumerated risk is at least its exact missing-Walsh-interaction floor.",
            "Aligned oracle risks are 0.75, 0.5, 0, 0, 0, 0 as S increases.",
            "Random-support expected oracle risk is (1 - 1/30)^S.",
            "Matched conventional realization and paired coordinate/support permutation preserve outputs and parameter gradients.",
            "Passive internal depth preserves a flattened weighted-leaf function and gradients when no internal drives are present.",
            "Whether end-to-end fitting approaches the support floor is an empirical outcome, not an assumption.",
        ],
    }


def _mse(a, b):
    return float(torch.mean((a - b) ** 2).detach())


def run_cell(config: dict, units: int, seed: int, mode: str, output: Path) -> dict:
    start = time.perf_counter()
    d, k, hidden = config["d"], config["k"], config["hidden"]
    interactions, coefficients = config["interactions"], config["coefficients"]
    supports = make_supports(d, k, units, seed + 10000, mode, interactions)
    model = LocalReLUPopulation(d, supports, hidden, seed=seed + units * 100).double()
    x = boolean_domain(d)
    y = parity_target(x, interactions, coefficients)
    permutation = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    ntrain, nval = config["train_vertices"], config["validation_vertices"]
    ids = {
        "train": permutation[:ntrain],
        "validation": permutation[ntrain : ntrain + nval],
        "test": permutation[ntrain + nval :],
    }
    if sorted(torch.cat(list(ids.values())).tolist()) != list(range(len(x))):
        raise AssertionError("Splits do not partition the domain")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    trace = []
    for step in range(config["steps"] + 1):
        if step % 100 == 0 or step == config["steps"]:
            with torch.no_grad():
                trace.append(
                    {
                        "step": step,
                        "train_mse": _mse(model(x[ids["train"]]), y[ids["train"]]),
                        "validation_mse": _mse(
                            model(x[ids["validation"]]), y[ids["validation"]]
                        ),
                    }
                )
        if step == config["steps"]:
            break
        optimizer.zero_grad(set_to_none=True)
        loss = torch.mean((model(x[ids["train"]]) - y[ids["train"]]) ** 2)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(x)
        somas = model.soma_features(x)
        centered = somas - somas.mean(0)
        singular = torch.linalg.svdvals(centered)
        tol = max(centered.shape) * torch.finfo(centered.dtype).eps * singular.max()
        soma_rank = int((singular > tol).sum())
        design = torch.cat([somas, torch.ones(len(somas), 1, dtype=somas.dtype)], dim=1)
        refit = torch.linalg.lstsq(
            design[ids["train"]], y[ids["train"]], driver="gelsd"
        ).solution
        refit_pred = design @ refit
        oracle = oracle_mse(supports.tolist(), interactions, coefficients)
        covered = interaction_coverage(supports.tolist(), interactions)
        oracle_pred = (
            parity_target(
                x,
                [a for a, hit in zip(interactions, covered) if hit],
                [c for c, hit in zip(coefficients, covered) if hit],
            )
            if any(covered)
            else torch.zeros_like(y)
        )
        measured_oracle = _mse(oracle_pred, y)
        constructed = construct_walsh_projection(model, interactions, coefficients)
        construction_residual = float((constructed(x) - oracle_pred).abs().max())
        if construction_residual > 1e-12:
            raise AssertionError(
                "Counted finite neural construction did not attain the oracle projection"
            )
        if not math.isclose(measured_oracle, oracle, abs_tol=1e-12):
            raise AssertionError(
                "Walsh projection formula does not match full-domain enumeration"
            )
        population_mse = _mse(pred, y)
        if population_mse < oracle - 1e-10:
            raise AssertionError(
                "Finite local model violated exact support obstruction"
            )
    probe_x = x[:47].clone().requires_grad_()
    params = tuple(model.parameters())
    ordinary = model(probe_x)
    control = model.conventional_forward(probe_x)
    weights = torch.linspace(-1, 1, len(probe_x), dtype=probe_x.dtype)
    grads_a = torch.autograd.grad(
        (ordinary * weights).sum(), (*params, probe_x), retain_graph=True
    )
    grads_b = torch.autograd.grad((control * weights).sum(), (*params, probe_x))
    conventional_gradient_max = max(
        float((a - b).abs().max()) for a, b in zip(grads_a, grads_b)
    )
    coordinate_order = torch.randperm(d, generator=torch.Generator().manual_seed(999))
    paired, permuted_x = paired_permutation(model, x[:47], coordinate_order)
    paired_delta = float((paired(permuted_x) - model(x[:47])).abs().max().detach())
    paired_grads = torch.autograd.grad(
        (paired(permuted_x) * weights).sum(), tuple(paired.parameters())
    )
    original_grads = torch.autograd.grad((model(x[:47]) * weights).sum(), params)
    paired_gradient_max = max(
        float((a - b).abs().max()) for a, b in zip(paired_grads, original_grads)
    )
    row = {
        "units": units,
        "hidden": hidden,
        "fan_in": k,
        "seed": seed,
        "mode": mode,
        "parameters": model.inventory()["total"],
        "oracle_mse": oracle,
        "constructed_projection_forward_max_abs": construction_residual,
        "constructed_projection_parameters": constructed.inventory()["total"],
        "expected_random_oracle_mse": sum(
            c * c * (1 - random_coverage_probability(d, k, len(a))) ** units
            for a, c in zip(interactions, coefficients)
        ),
        "population_mse": population_mse,
        "optimization_and_estimation_excess": population_mse - oracle,
        "soma_rank_centered": soma_rank,
        "readout_refit_population_mse": _mse(refit_pred, y),
        "population_target_mean": float(y.mean()),
        "population_target_variance": float(y.square().mean()),
        "population_prediction_variance": float(((pred - pred.mean()) ** 2).mean()),
        "population_centered_signal_capture": 1
        - _mse(pred - pred.mean(), y - y.mean()) / float(((y - y.mean()) ** 2).mean()),
        "zero_predictor_population_mse": float(y.square().mean()),
        "constant_fit_population_mse": _mse(y[ids["train"]].mean().expand_as(y), y),
        "conventional_forward_max_abs": float(
            (ordinary - control).abs().max().detach()
        ),
        "conventional_gradient_max_abs": conventional_gradient_max,
        "paired_permutation_max_abs": paired_delta,
        "paired_permutation_gradient_max_abs": paired_gradient_max,
        "cpu_seconds": time.perf_counter() - start,
    }
    for split, indices in ids.items():
        row[f"{split}_mse"] = _mse(pred[indices], y[indices])
        row[f"readout_refit_{split}_mse"] = _mse(refit_pred[indices], y[indices])
    destination = output / f"{mode}_S{units:02d}_seed{seed}"
    destination.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "model": model.state_dict(),
        "split_indices": ids,
        "config": config,
        "row": row,
    }
    torch.save(checkpoint, destination / "state.pt")
    torch.save(
        {
            "model": constructed.state_dict(),
            "interpretation": "Target-dependent constructive witness, not trained and never used as optimizer initialization",
            "inventory": constructed.inventory(),
        },
        destination / "constructed_projection.pt",
    )
    details = {
        "metrics": row,
        "inventory": model.inventory(),
        "supports": supports.tolist(),
        "covered_interactions": covered,
        "trace": trace,
        "split_sha256": {
            name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for name, value in ids.items()
        },
    }
    (destination / "result.json").write_text(json.dumps(details, indent=2) + "\n")
    return row


def run_passive_checks(output: Path):
    gen = torch.Generator().manual_seed(717)
    x = torch.rand(61, 4, generator=gen, dtype=torch.float64).requires_grad_()
    edges = torch.randn(6, generator=gen, dtype=torch.float64).requires_grad_()
    leaks = torch.randn(3, generator=gen, dtype=torch.float64).requires_grad_()
    weights = torch.randn(61, generator=gen, dtype=torch.float64)
    tree = passive_tree(x, edges, leaks)
    flat = flattened_passive_tree(x, edges, leaks)
    ga = torch.autograd.grad(
        (tree * weights).sum(), (x, edges, leaks), retain_graph=True
    )
    gb = torch.autograd.grad((flat * weights).sum(), (x, edges, leaks))
    result = {
        "tree_learned_parameters": 9,
        "derived_flat_control_learned_parameters": 9,
        "free_flat_leaf_weight_count_if_reparameterized": 4,
        "forward_max_abs": float((tree - flat).abs().max().detach()),
        "gradient_max_abs": max(float((a - b).abs().max()) for a, b in zip(ga, gb)),
        "assumptions": [
            "nonnegative leaf voltage",
            "positive conductances and leaks",
            "no internal input drive",
            "no active nonlinear threshold",
            "fixed binary topology",
        ],
        "interpretation": "Passive depth adds a reparameterization but no new function of these leaf features. This does not cover active, internally driven dendrites.",
    }
    (output / "passive_depth_control.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    return result


def run_oracle_sweep(config: dict, output: Path):
    rows = []
    repetitions = config.get("oracle_support_draws", 4096)
    for units in config["units"]:
        residuals = []
        for seed in range(40000, 40000 + repetitions):
            supports = make_supports(
                config["d"], config["k"], units, seed, "random", config["interactions"]
            )
            residuals.append(
                oracle_mse(
                    supports.tolist(), config["interactions"], config["coefficients"]
                )
            )
        expected = sum(
            c
            * c
            * (1 - random_coverage_probability(config["d"], config["k"], len(a)))
            ** units
            for a, c in zip(config["interactions"], config["coefficients"])
        )
        rows.append(
            {
                "units": units,
                "parameters": units * (config["hidden"] * (config["k"] + 2) + 2) + 1,
                "random_exact_expected_mse": expected,
                "random_monte_carlo_mean": float(np.mean(residuals)),
                "random_monte_carlo_standard_error": float(
                    np.std(residuals, ddof=1) / math.sqrt(repetitions)
                ),
                "support_draws": repetitions,
                "aligned_oracle_mse": oracle_mse(
                    make_supports(
                        config["d"],
                        config["k"],
                        units,
                        1701,
                        "aligned",
                        config["interactions"],
                    ),
                    config["interactions"],
                    config["coefficients"],
                ),
            }
        )
    (output / "oracle_sweep.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def run_campaign(output_dir, config=None) -> dict:
    """Run a predeclared campaign; test/population metrics never select anything."""
    output = Path(output_dir)
    config = protocol() if config is None else copy.deepcopy(config)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if set(config["development_seeds"]) & set(config["confirmation_seeds"]):
        raise ValueError("Development and confirmation seeds must be disjoint")
    if (
        sum(config[f"{name}_vertices"] for name in ("train", "validation", "test"))
        != 2 ** config["d"]
    ):
        raise ValueError(
            "Declared splits must exhaustively partition the Boolean domain"
        )
    output.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(config, indent=2) + "\n"
    (output / "protocol.json").write_text(encoded)
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "torch_threads": torch.get_num_threads(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    controls = run_passive_checks(output)
    oracle_rows = run_oracle_sweep(config, output)
    all_rows = []
    for phase in ("development", "confirmation"):
        phase_directory = output / phase
        phase_directory.mkdir(exist_ok=False)
        rows = []
        for seed, units, mode in itertools.product(
            config[f"{phase}_seeds"], config["units"], config["support_modes"]
        ):
            row = run_cell(config, units, seed, mode, phase_directory)
            row["phase"] = phase
            rows.append(row)
            all_rows.append(row)
            print(json.dumps(row), flush=True)
            (phase_directory / "results.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
            with (phase_directory / "results.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    summary = {
        "protocol_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "trained_cells": len(all_rows),
        "cpu_training_seconds": sum(row["cpu_seconds"] for row in all_rows),
        "passive_control": controls,
        "oracle_sweep": oracle_rows,
        "results": all_rows,
        "claim_boundary": config["claim_boundary"],
        "selection": config["checkpoint_selection"],
        "resource": "one CPU thread; no GPU",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", "--output", dest="output", type=Path, required=True
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Timing canary only: two development cells, 25 steps, 32 support draws",
    )
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    config = json.loads(args.config.read_text()) if args.config else protocol()
    if args.quick:
        config.update(
            {
                "units": [4],
                "steps": 25,
                "confirmation_seeds": [],
                "oracle_support_draws": 32,
                "timing_canary_only": True,
            }
        )
    print(json.dumps(run_campaign(args.output, config)), flush=True)


if __name__ == "__main__":
    main()

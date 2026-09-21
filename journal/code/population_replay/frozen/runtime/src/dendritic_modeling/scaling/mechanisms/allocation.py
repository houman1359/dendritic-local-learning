"""Finite fixed-feature allocation experiments with fully counted learned models.

This is a mechanism/control experiment, not a learned dendritic feature model.
The common Fourier dictionary is supplied and its evaluations are reported as
a separate resource. Least squares learns effective coefficients from sampled
observations; balanced node factorization never uses target coefficients.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy import linalg
from scipy.special import zeta
from torch import nn


@dataclass(frozen=True)
class AllocationConfig:
    max_frequency: int = 1024
    p: float = 1.0
    q: float = 1.0
    development_budgets: tuple[int, ...] = (17, 33, 65, 129, 257, 513)
    confirmation_budget: int = 1025
    sample_seeds: tuple[int, ...] = (2026091701, 2026091702, 2026091703)
    target_seed: int = 2026091700
    allocation_seed: int = 2026091704
    train_samples: int = 8192
    validation_samples: int = 2048
    test_samples: int = 8192

    def __post_init__(self):
        if self.max_frequency < 2 or min(self.p, self.q) <= 0:
            raise ValueError(
                "Positive tail exponents and at least two frequencies required"
            )
        if len(self.development_budgets) < 2 or min(self.development_budgets) < 3:
            raise ValueError("At least two development budgets >= 3 required")
        if list(self.development_budgets) != sorted(set(self.development_budgets)):
            raise ValueError("Development budgets must be strictly increasing")
        if self.confirmation_budget <= max(self.development_budgets):
            raise ValueError("Confirmation budget must exceed all development budgets")
        if self.train_samples <= self.confirmation_budget:
            raise ValueError(
                "Training sample count must exceed largest parameter budget"
            )
        if min(self.validation_samples, self.test_samples) < 1 or not self.sample_seeds:
            raise ValueError("Nonempty sample splits and seeds required")


@dataclass(frozen=True)
class Support:
    family: str
    budget: int
    modes: tuple[tuple[int, int], ...]
    factorized: bool

    @property
    def groups(self):
        return tuple(sorted({i for i, _ in self.modes}))

    @property
    def parameter_count(self):
        return len(self.modes) + (len(self.groups) if self.factorized else 0) + 1


def ranked_modes(max_frequency: int, p: float, q: float, count: int):
    """Descending energy, with a deterministic predecessor-respecting tie order."""
    heap, seen, result = [(-1.0, 1, 1)], {(1, 1)}, []
    for _ in range(min(count, max_frequency**2)):
        _, i, j = heapq.heappop(heap)
        result.append((i, j))
        for u, v in ((i + 1, j), (i, j + 1)):
            if max(u, v) <= max_frequency and (u, v) not in seen:
                seen.add((u, v))
                heapq.heappush(heap, (-(u ** (-p - 1)) * v ** (-q - 1), u, v))
    return tuple(result)


def make_supports(config: AllocationConfig, budget: int):
    k = config.max_frequency
    hp = np.r_[0.0, np.cumsum(np.arange(1, k + 1, dtype=float) ** (-config.p - 1))]
    hq = np.r_[0.0, np.cumsum(np.arange(1, k + 1, dtype=float) ** (-config.q - 1))]
    options = []
    for s in range(1, min(k, (budget - 1) // 2) + 1):
        c = min(k, (budget - 1) // s - 1)
        options.append((hp[s] * hq[c], -s, s, c))
    _, _, s, c = max(options)
    uniform = Support(
        "uniform",
        budget,
        tuple((i, j) for i in range(1, s + 1) for j in range(1, c + 1)),
        True,
    )
    ranked = ranked_modes(k, config.p, config.q, budget - 1)
    selected, groups = [], set()
    for mode in ranked:
        prospective = len(selected) + 1 + len(groups | {mode[0]}) + 1
        if prospective > budget:
            break
        selected.append(mode)
        groups.add(mode[0])
    heterogeneous = Support("heterogeneous", budget, tuple(selected), True)
    # One permanent random mapping of group indices is shared across budgets.
    # Capacities, node count, and learned count remain exactly matched to het.
    permutation = np.random.default_rng(config.allocation_seed).permutation(k) + 1
    random = Support(
        "random_groups",
        budget,
        tuple((int(permutation[i - 1]), j) for i, j in selected),
        True,
    )
    sparse = Support("unrestricted_sparse_linear", budget, ranked, False)
    return (uniform, heterogeneous, random, sparse)


class FixedFourierTask:
    """Budget-independent finite target under independent Uniform([0,1]) inputs."""

    def __init__(self, config: AllocationConfig):
        self.config = config
        axis = np.arange(1, config.max_frequency + 1, dtype=float)
        rng = np.random.default_rng(config.target_seed)
        signs = 2 * rng.integers(0, 2, size=(len(axis), len(axis))) - 1
        self.coefficients = signs * np.outer(
            axis ** (-(config.p + 1) / 2), axis ** (-(config.q + 1) / 2)
        )
        self.total_energy = float(np.square(self.coefficients).sum())
        self.infinite_extension_tail = float(
            zeta(config.p + 1, 1) * zeta(config.q + 1, 1) - self.total_energy
        )

    def evaluate(self, x: np.ndarray, batch_size=1024):
        axis = np.arange(1, self.config.max_frequency + 1, dtype=float)
        result = np.empty(len(x))
        for start in range(0, len(x), batch_size):
            chunk = x[start : start + batch_size]
            sx = np.sin(2 * np.pi * chunk[:, :1] * axis)
            sy = np.sin(2 * np.pi * chunk[:, 1:] * axis)
            result[start : start + len(chunk)] = 2 * np.sum(
                (sx @ self.coefficients) * sy, axis=1
            )
        return result

    def selected_coefficients(self, support):
        indices = np.asarray(support.modes, dtype=int) - 1
        return self.coefficients[indices[:, 0], indices[:, 1]]

    def oracle_mse(self, support):
        return float(
            max(
                0.0,
                self.total_energy
                - np.square(self.selected_coefficients(support)).sum(),
            )
        )

    def population_mse(self, support, fitted_coefficients, bias):
        return (
            self.oracle_mse(support)
            + float(
                np.square(
                    fitted_coefficients - self.selected_coefficients(support)
                ).sum()
            )
            + float(bias**2)
        )


def features(x: np.ndarray, modes):
    indices = np.asarray(modes, dtype=float)
    return (
        2
        * np.sin(2 * np.pi * x[:, :1] * indices[:, 0])
        * np.sin(2 * np.pi * x[:, 1:] * indices[:, 1])
    )


class FourierAllocationModel(nn.Module):
    """Learned internal coefficients plus counted node readouts and output bias."""

    def __init__(self, support: Support, *, seed=0, dtype=torch.float64):
        super().__init__()
        self.support = support
        self.register_buffer("frequencies", torch.tensor(support.modes, dtype=dtype))
        group_map = {group: i for i, group in enumerate(support.groups)}
        self.register_buffer(
            "group_index",
            torch.tensor([group_map[i] for i, _ in support.modes], dtype=torch.long),
        )
        generator = torch.Generator().manual_seed(seed)
        self.internal = nn.Parameter(
            torch.randn(len(support.modes), generator=generator, dtype=dtype) * 0.01
        )
        self.readout = (
            nn.Parameter(torch.ones(len(support.groups), dtype=dtype))
            if support.factorized
            else None
        )
        self.bias = nn.Parameter(torch.zeros((), dtype=dtype))

    def effective_coefficients(self):
        if self.readout is None:
            return self.internal
        return self.internal * self.readout[self.group_index]

    def forward(self, x):
        phi = (
            2
            * torch.sin(2 * torch.pi * x[:, :1] * self.frequencies[:, 0])
            * torch.sin(2 * torch.pi * x[:, 1:] * self.frequencies[:, 1])
        )
        return phi @ self.effective_coefficients() + self.bias

    def set_fitted_coefficients(self, coefficient, bias):
        """Balance an observation-fitted linear solution; no target access."""
        fitted = torch.as_tensor(
            coefficient, dtype=self.internal.dtype, device=self.internal.device
        )
        with torch.no_grad():
            if self.readout is None:
                self.internal.copy_(fitted)
            else:
                norms = torch.zeros_like(self.readout).scatter_add_(
                    0, self.group_index, fitted.square()
                )
                scale = norms.sqrt().sqrt().clamp_min(torch.finfo(fitted.dtype).eps)
                self.readout.copy_(scale)
                self.internal.copy_(fitted / scale[self.group_index])
            self.bias.copy_(
                torch.as_tensor(bias, dtype=self.bias.dtype, device=self.bias.device)
            )


def fit_observations(design, target):
    """Unregularized overdetermined QR least squares; no validation/test access."""
    augmented = np.column_stack((design, np.ones(len(design))))
    solution, _, rank, _ = linalg.lstsq(
        augmented, target, lapack_driver="gelsy", check_finite=True
    )
    residual = augmented @ solution - target
    normal_residual = np.linalg.norm(augmented.T @ residual) / (
        np.linalg.norm(augmented) * max(np.linalg.norm(residual), 1e-30)
    )
    return solution[:-1], float(solution[-1]), int(rank), float(normal_residual)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_campaign(output_dir, config=None):
    """Freeze predictions, fit sampled observations, evaluate a held-out budget."""
    config = config or AllocationConfig()
    if isinstance(config, dict):
        config = AllocationConfig(**config)
    torch.set_num_threads(1)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frozen_plan.json").exists():
        raise FileExistsError(
            "Use a fresh output directory; frozen campaigns are immutable"
        )
    started = time.perf_counter()
    task = FixedFourierTask(config)
    all_budgets = (*config.development_budgets, config.confirmation_budget)
    supports = {budget: make_supports(config, budget) for budget in all_budgets}
    plan = {
        "schema": "fixed_fourier_allocation_campaign_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "source_sha256": _digest(__file__),
        "target_coefficients_sha256": hashlib.sha256(
            task.coefficients.tobytes()
        ).hexdigest(),
        "target_energy": task.total_energy,
        "omitted_infinite_extension_energy": task.infinite_extension_tail,
        "scope": "Finite fixed Fourier feature mechanism; sampled noiseless observations; supplied dictionary, no learned compartment claim.",
        "selection": "Target-energy-specified support rules frozen before observations; QR least squares uses training only; no tuning/selection on validation or test.",
        "random_control": "One budget-independent permutation maps aligned group indices to random groups; exact same capacities and count as heterogeneous.",
        "primary_contrast": "Paired population MSE heterogeneous / uniform at the predeclared confirmation budget; all seeds reported.",
        "secondary_contrast": "Finite-range slopes are descriptive, not asymptotic exponents; compare to exact finite oracle and sparse linear control.",
        "dictionary_cost": "M retained 2D product Fourier feature evaluations per example, requiring 2M scalar sine evaluations without reuse; frequencies supplied structural constants. Target evaluator uses K^2 modes and is not student cost.",
        "fitting": "QR least squares in effective coordinates, then balanced factorization into counted learned internal/readout weights. This is algorithm-assisted fitting, not SGD dynamics.",
        "supports": [
            {
                "budget": budget,
                "family": s.family,
                "factorized": s.factorized,
                "actual_parameters": s.parameter_count,
                "retained_modes": len(s.modes),
                "active_nodes": len(s.groups),
                "modes": s.modes,
                "finite_oracle_mse": task.oracle_mse(s),
            }
            for budget in all_budgets
            for s in supports[budget]
        ],
    }
    _write_json(output / "frozen_plan.json", plan)
    plan_hash = _digest(output / "frozen_plan.json")
    rows, sample_receipts, datasets = [], [], {}
    for seed in config.sample_seeds:
        spawned = np.random.SeedSequence(seed).spawn(3)
        splits = {}
        for name, n, child in zip(
            ("train", "validation", "test"),
            (config.train_samples, config.validation_samples, config.test_samples),
            spawned,
        ):
            x = np.random.default_rng(child).random((n, 2))
            y = task.evaluate(x)
            splits[name] = (x, y)
            sample_receipts.append(
                {
                    "sample_seed": seed,
                    "split": name,
                    "count": n,
                    "input_sha256": hashlib.sha256(x.tobytes()).hexdigest(),
                    "target_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
                }
            )
        datasets[seed] = splits
    _write_json(output / "sample_receipt.json", sample_receipts)

    def evaluate_budget(budget, phase):
        for seed in config.sample_seeds:
            splits = datasets[seed]
            for support in supports[budget]:
                tick = time.perf_counter()
                x, target = splits["train"]
                design = features(x, support.modes)
                model = FourierAllocationModel(support, seed=seed)
                actual = sum(p.numel() for p in model.parameters())
                if actual != support.parameter_count or actual > budget:
                    raise AssertionError("Whole learned count contract failed")
                initial = model.effective_coefficients().detach().numpy().copy()
                initial_loss = float(np.mean(np.square(design @ initial - target)))
                fitted, bias, rank, normal = fit_observations(design, target)
                model.set_fitted_coefficients(fitted, bias)
                reconstructed = model.effective_coefficients().detach().numpy()
                factorization_error = float(np.max(np.abs(reconstructed - fitted)))
                torch_x = torch.from_numpy(x[:64])
                torch_error = float(
                    np.max(
                        np.abs(
                            model(torch_x).detach().numpy()
                            - (design[:64] @ fitted + bias)
                        )
                    )
                )
                row = {
                    "phase": phase,
                    "sample_seed": seed,
                    "family": support.family,
                    "budget_ceiling": budget,
                    "actual_parameters": actual,
                    "active_nodes": len(support.groups),
                    "retained_modes": len(support.modes),
                    "sine_evaluations_per_example_no_reuse": 2 * len(support.modes),
                    "flat_same_support_parameters": len(support.modes) + 1,
                    "train_samples": config.train_samples,
                    "initial_train_mse": initial_loss,
                    "train_mse": float(
                        np.mean(np.square(design @ fitted + bias - target))
                    ),
                    "finite_oracle_mse": task.oracle_mse(support),
                    "population_mse": task.population_mse(support, fitted, bias),
                    "estimation_excess_mse": float(
                        np.square(fitted - task.selected_coefficients(support)).sum()
                        + bias**2
                    ),
                    "qr_rank": rank,
                    "qr_relative_normal_residual": normal,
                    "factorization_max_error": factorization_error,
                    "torch_numpy_forward_max_error": torch_error,
                }
                for name in ("validation", "test"):
                    xx, yy = splits[name]
                    prediction = features(xx, support.modes) @ fitted + bias
                    row[name + "_mse"] = float(np.mean(np.square(prediction - yy)))
                row["fit_and_evaluation_seconds"] = time.perf_counter() - tick
                rows.append(row)
                np.savez(
                    output / f"fit_{support.family}_p{budget}_s{seed}.npz",
                    coefficients=fitted,
                    bias=bias,
                    modes=np.array(support.modes),
                    internal=model.internal.detach().numpy(),
                    readout=(
                        np.array([])
                        if model.readout is None
                        else model.readout.detach().numpy()
                    ),
                )
                with (output / "results.jsonl").open("a") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "finished_budget": budget,
                    "phase": phase,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )

    for budget in config.development_budgets:
        evaluate_budget(budget, "development")
    predictions = []
    for family in [s.family for s in supports[config.confirmation_budget]]:
        family_rows = [r for r in rows if r["family"] == family]
        budgets = list(config.development_budgets)
        means = [
            np.mean(
                [r["population_mse"] for r in family_rows if r["budget_ceiling"] == b]
            )
            for b in budgets
        ]
        actual = [
            next(
                r["actual_parameters"] for r in family_rows if r["budget_ceiling"] == b
            )
            for b in budgets
        ]
        slope, intercept = np.polyfit(np.log(actual), np.log(means), 1)
        confirmation = next(
            s for s in supports[config.confirmation_budget] if s.family == family
        )
        predictions.append(
            {
                "family": family,
                "development_descriptive_power": float(-slope),
                "power_fit_predicted_confirmation_mse": float(
                    np.exp(intercept) * confirmation.parameter_count**slope
                ),
                "predeclared_finite_oracle_confirmation_mse": task.oracle_mse(
                    confirmation
                ),
                "warning": "Descriptive finite-range fit; target truncation and random design estimation affect observed powers.",
            }
        )
    _write_json(
        output / "predictions_before_confirmation.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "frozen_plan_sha256": plan_hash,
            "evaluated_rows_so_far": len(rows),
            "confirmation_rows_evaluated": 0,
            "predictions": predictions,
        },
    )
    evaluate_budget(config.confirmation_budget, "confirmation")
    with (output / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    confirmation = [r for r in rows if r["phase"] == "confirmation"]
    pairs = []
    for seed in config.sample_seeds:
        matched = {r["family"]: r for r in confirmation if r["sample_seed"] == seed}
        pairs.append(
            {
                "sample_seed": seed,
                "heterogeneous_uniform_population_ratio": matched["heterogeneous"][
                    "population_mse"
                ]
                / matched["uniform"]["population_mse"],
                "heterogeneous_uniform_test_ratio": matched["heterogeneous"]["test_mse"]
                / matched["uniform"]["test_mse"],
            }
        )
    summary = {
        "schema": "fixed_fourier_allocation_results_v1",
        "status": "completed",
        "config": asdict(config),
        "rows": len(rows),
        "cpu_only": True,
        "gpu_jobs": 0,
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_plan_sha256": plan_hash,
        "predictions_sha256": _digest(output / "predictions_before_confirmation.json"),
        "paired_confirmation": pairs,
        "development_predictions": predictions,
        "confirmation_family_means": {
            family: {
                metric: float(
                    np.mean([r[metric] for r in confirmation if r["family"] == family])
                )
                for metric in (
                    "population_mse",
                    "finite_oracle_mse",
                    "test_mse",
                    "actual_parameters",
                )
            }
            for family in [s.family for s in supports[config.confirmation_budget]]
        },
        "max_qr_normal_residual": max(r["qr_relative_normal_residual"] for r in rows),
        "max_forward_residual": max(r["torch_numpy_forward_max_error"] for r in rows),
        "min_qr_rank_margin": min(
            r["qr_rank"] - (r["retained_modes"] + 1) for r in rows
        ),
        "infinite_extension_tail": task.infinite_extension_tail,
        "scope": plan["scope"],
        "fitting": plan["fitting"],
        "dictionary_cost": plan["dictionary_cost"],
    }
    if _digest(output / "frozen_plan.json") != plan_hash:
        raise AssertionError("Frozen plan changed")
    _write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", help="JSON path containing AllocationConfig fields")
    args = parser.parse_args()
    if args.quick and args.config:
        parser.error("--quick and --config are mutually exclusive")
    config = (
        AllocationConfig(
            max_frequency=32,
            development_budgets=(9, 17, 33, 65, 129),
            confirmation_budget=257,
            sample_seeds=(2026091799,),
            train_samples=1024,
            validation_samples=256,
            test_samples=512,
        )
        if args.quick
        else AllocationConfig()
    )
    if args.config:
        config = AllocationConfig(**json.loads(Path(args.config).read_text()))
    print(json.dumps(run_campaign(args.output_dir, config), indent=2))


if __name__ == "__main__":
    main()

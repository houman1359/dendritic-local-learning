"""Prospective training-only support discovery for the fixed Boolean mechanism task.

Selection knows interaction order three but no target interaction identities. It
uses only training inputs and labels. Dictionary estimation and topology search
are explicitly additional training work, not deployed real-valued parameters.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _core():
    """Prefer frozen sibling when running a frozen experiment bundle."""
    path = Path(__file__).with_name("support.py")
    specification = importlib.util.spec_from_file_location(
        "support_discovery_frozen_core", path
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def estimate_walsh_scores(x_train, y_train, order=3):
    """No target definition, validation, test, population or model enters this API."""
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    if x.ndim != 2 or y.shape != (len(x),) or not 1 <= order <= x.shape[1]:
        raise ValueError(
            "Need a training matrix, matching labels and valid interaction order"
        )
    if not np.all(np.isin(x, [-1, 1])) or not np.all(np.isfinite(y)):
        raise ValueError("Need Boolean inputs and finite training labels")
    dictionary = list(itertools.combinations(range(x.shape[1]), order))
    scores = np.asarray(
        [np.mean(y * np.prod(x[:, interaction], axis=1)) for interaction in dictionary]
    )
    return dictionary, scores


def greedy_supports(dictionary, scores, d, k, units):
    """Greedy uncovered squared-score gain, lexicographic deterministic ties.

    After all positive scores are covered, repeat the first selected support.
    This API has no labels or ground-truth target definition.
    """
    dictionary = [tuple(a) for a in dictionary]
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(dictionary),) or not np.all(np.isfinite(scores)):
        raise ValueError("One finite score is required for each dictionary interaction")
    if not 1 <= k <= d or units < 1:
        raise ValueError("Need 1 <= k <= d and positive units")
    candidates = list(itertools.combinations(range(d), k))
    incidence = np.asarray(
        [[set(a) <= set(candidate) for a in dictionary] for candidate in candidates],
        dtype=bool,
    )
    uncovered = np.ones(len(dictionary), dtype=bool)
    selected = []
    trace = []
    for step in range(units):
        gains = incidence @ (scores**2 * uncovered)
        best = int(np.argmax(gains))
        if gains[best] <= 0:
            choice = selected[0] if selected else candidates[0]
            selected.append(choice)
            trace.append(
                {
                    "unit": step,
                    "support": list(choice),
                    "new_squared_score": 0.0,
                    "action": "deterministic_repeat",
                }
            )
        else:
            selected.append(candidates[best])
            trace.append(
                {
                    "unit": step,
                    "support": list(candidates[best]),
                    "new_squared_score": float(gains[best]),
                    "action": "greedy_gain",
                }
            )
            uncovered &= ~incidence[best]
    return np.asarray(selected, dtype=np.int64), trace


def protocol():
    config = _core().protocol()
    config.update(
        {
            "version": "support_discovery_v1",
            "development_seeds": [3101],
            "confirmation_seeds": [3201, 3203, 3207],
            "support_modes": ["random", "aligned", "learned", "permuted_scores"],
            "score_order": 3,
            "score_permutation_seed": 99017,
            "selector": "Estimate all order-3 Walsh coefficients on training vertices only; greedily maximize squared uncovered score over all k-subsets; lexicographic ties; repeat first selected support after zero gain.",
            "selection_information": "Order three is supplied as a task prior. Learned and score-permuted selectors receive no target identities or true coefficients.",
            "checkpoint_selection": "terminal only; no checkpoint, algorithm or setting selected using validation, test or full-population outcomes",
            "claim_boundary": "Prospective exploratory test of training-only structure discovery on the same fixed finite Boolean task; not an asymptotic exponent or uniquely dendritic gain.",
            "predictions": [
                "Training-only discovered supports will approach the known-aligned finite-budget transition on fresh seeds.",
                "Fixed permutation of training scores should weaken useful support selection; all outcomes will be retained.",
                "Exact oracle projection remains attainable at the same counted neural P for every realized support.",
                "Selection adds 120 training-only coefficient estimates and a 210-support search; these are reported separately from deployed learned real scalars and discrete topology.",
            ],
        }
    )
    return config


def train_cell(
    core, config, x, y, ids, supports, units, seed, mode, destination, selection_hash
):
    start = time.perf_counter()
    model = core.LocalReLUPopulation(
        config["d"], supports, config["hidden"], seed=seed + units * 100
    ).double()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    trace = []
    for step in range(config["steps"] + 1):
        if step % 100 == 0 or step == config["steps"]:
            with torch.no_grad():
                trace.append(
                    {
                        "step": step,
                        "train_mse": core._mse(model(x[ids["train"]]), y[ids["train"]]),
                        "validation_mse": core._mse(
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
    with torch.no_grad():
        pred = model(x)
        somas = model.soma_features(x)
        centered = somas - somas.mean(0)
        singular = torch.linalg.svdvals(centered)
        tol = max(centered.shape) * torch.finfo(centered.dtype).eps * singular.max()
        design = torch.cat([somas, torch.ones(len(somas), 1, dtype=somas.dtype)], dim=1)
        refit = torch.linalg.lstsq(
            design[ids["train"]], y[ids["train"]], driver="gelsd"
        ).solution
        refit_pred = design @ refit
        oracle = core.oracle_mse(
            supports, config["interactions"], config["coefficients"]
        )
        constructed = core.construct_walsh_projection(
            model, config["interactions"], config["coefficients"]
        )
        construction_mse = core._mse(constructed(x), y)
        assert math.isclose(construction_mse, oracle, abs_tol=1e-12)
        covered = core.interaction_coverage(supports, config["interactions"])
        projected = sum(
            (
                coefficient * x[:, interaction].prod(dim=1)
                if hit
                else torch.zeros_like(y)
            )
            for interaction, coefficient, hit in zip(
                config["interactions"], config["coefficients"], covered
            )
        )
        construction_forward_error = float((constructed(x) - projected).abs().max())
        assert construction_forward_error < 1e-12
        population_mse = core._mse(pred, y)
        assert population_mse >= oracle - 1e-10
    probe = x[:47].clone().requires_grad_()
    weights = torch.linspace(-1, 1, len(probe), dtype=probe.dtype)
    ordinary, conventional = model(probe), model.conventional_forward(probe)
    params = tuple(model.parameters())
    ga = torch.autograd.grad(
        (ordinary * weights).sum(), (*params, probe), retain_graph=True
    )
    gb = torch.autograd.grad((conventional * weights).sum(), (*params, probe))
    gradient_delta = max(float((a - b).abs().max()) for a, b in zip(ga, gb))
    paired, permuted = core.paired_permutation(
        model,
        x[:47],
        torch.randperm(config["d"], generator=torch.Generator().manual_seed(999)),
    )
    paired_pred = paired(permuted)
    gc = torch.autograd.grad((paired_pred * weights).sum(), tuple(paired.parameters()))
    gd = torch.autograd.grad((model(x[:47]) * weights).sum(), params)
    paired_gradient_delta = max(float((a - b).abs().max()) for a, b in zip(gc, gd))
    assert gradient_delta < 1e-9 and paired_gradient_delta < 1e-9
    row = {
        "units": units,
        "hidden": config["hidden"],
        "fan_in": config["k"],
        "seed": seed,
        "mode": mode,
        "parameters": model.inventory()["total"],
        "oracle_mse": oracle,
        "population_mse": population_mse,
        "optimization_and_estimation_excess": population_mse - oracle,
        "constructed_projection_forward_max_abs": construction_forward_error,
        "constructed_projection_parameters": constructed.inventory()["total"],
        "soma_rank_centered": int((singular > tol).sum()),
        "readout_refit_population_mse": core._mse(refit_pred, y),
        "population_target_mean": float(y.mean()),
        "population_target_variance": float(y.square().mean()),
        "population_prediction_variance": float(((pred - pred.mean()) ** 2).mean()),
        "population_centered_signal_capture": 1
        - core._mse(pred - pred.mean(), y - y.mean())
        / float(((y - y.mean()) ** 2).mean()),
        "zero_predictor_population_mse": float(y.square().mean()),
        "constant_fit_population_mse": core._mse(
            y[ids["train"]].mean().expand_as(y), y
        ),
        "conventional_forward_max_abs": float(
            (ordinary - conventional).abs().max().detach()
        ),
        "conventional_gradient_max_abs": gradient_delta,
        "paired_permutation_max_abs": float(
            (paired_pred - model(x[:47])).abs().max().detach()
        ),
        "paired_permutation_gradient_max_abs": paired_gradient_delta,
        "selection_record_sha256": selection_hash,
        "deployed_support_index_count": int(np.asarray(supports).size),
        "training_only_score_estimates": (
            math.comb(config["d"], config["score_order"])
            if mode in ("learned", "permuted_scores")
            else 0
        ),
        "cpu_seconds": time.perf_counter() - start,
    }
    for split, indices in ids.items():
        row[f"{split}_mse"] = core._mse(pred[indices], y[indices])
        row[f"readout_refit_{split}_mse"] = core._mse(refit_pred[indices], y[indices])
    destination.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "model": model.state_dict(),
            "split_indices": ids,
            "config": config,
            "row": row,
        },
        destination / "state.pt",
    )
    torch.save(
        {
            "model": constructed.state_dict(),
            "interpretation": "Target-dependent construction separate from support discovery and optimizer initialization",
            "inventory": constructed.inventory(),
        },
        destination / "constructed_projection.pt",
    )
    details = {
        "metrics": row,
        "inventory": model.inventory(),
        "supports": np.asarray(supports).tolist(),
        "covered_interactions": core.interaction_coverage(
            supports, config["interactions"]
        ),
        "trace": trace,
        "split_sha256": {
            name: hashlib.sha256(value.numpy().tobytes()).hexdigest()
            for name, value in ids.items()
        },
    }
    (destination / "result.json").write_text(json.dumps(details, indent=2) + "\n")
    return row


def run_campaign(output_dir, config=None):
    output = Path(output_dir)
    config = protocol() if config is None else config
    if set(config["development_seeds"]) & set(config["confirmation_seeds"]):
        raise ValueError("Development and confirmation seeds must differ")
    if (
        sum(config[f"{split}_vertices"] for split in ("train", "validation", "test"))
        != 2 ** config["d"]
    ):
        raise ValueError("Splits must partition the domain")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    core = _core()
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
        "core_sha256": hashlib.sha256(
            Path(__file__).with_name("support.py").read_bytes()
        ).hexdigest(),
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    x = core.boolean_domain(config["d"])
    y = core.parity_target(x, config["interactions"], config["coefficients"])
    rows, selection_records = [], []
    for phase in ("development", "confirmation"):
        phase_dir = output / phase
        phase_dir.mkdir()
        phase_rows = []
        for seed in config[f"{phase}_seeds"]:
            permutation = torch.randperm(
                len(x), generator=torch.Generator().manual_seed(seed)
            )
            ntrain, nval = config["train_vertices"], config["validation_vertices"]
            ids = {
                "train": permutation[:ntrain],
                "validation": permutation[ntrain : ntrain + nval],
                "test": permutation[ntrain + nval :],
            }
            begin = time.perf_counter()
            dictionary, scores = estimate_walsh_scores(
                x[ids["train"]].numpy(), y[ids["train"]].numpy(), config["score_order"]
            )
            score_seconds = time.perf_counter() - begin
            score_permutation = np.random.default_rng(
                config["score_permutation_seed"]
            ).permutation(len(scores))
            begin = time.perf_counter()
            selected, selected_trace = greedy_supports(
                dictionary, scores, config["d"], config["k"], max(config["units"])
            )
            search_seconds = time.perf_counter() - begin
            begin = time.perf_counter()
            permuted_selected, permuted_trace = greedy_supports(
                dictionary,
                scores[score_permutation],
                config["d"],
                config["k"],
                max(config["units"]),
            )
            permutation_search_seconds = time.perf_counter() - begin
            selection = {
                "phase": phase,
                "seed": seed,
                "dictionary": dictionary,
                "estimated_training_coefficients": scores.tolist(),
                "score_permutation": score_permutation.tolist(),
                "selected_supports": selected.tolist(),
                "permuted_score_supports": permuted_selected.tolist(),
                "selected_trace": selected_trace,
                "permuted_trace": permuted_trace,
                "dictionary_size": len(dictionary),
                "candidate_support_count": math.comb(config["d"], config["k"]),
                "score_estimation_seconds": score_seconds,
                "learned_search_seconds": search_seconds,
                "permuted_search_seconds": permutation_search_seconds,
                "training_input_sha256": hashlib.sha256(
                    x[ids["train"]].numpy().tobytes()
                ).hexdigest(),
                "training_label_sha256": hashlib.sha256(
                    y[ids["train"]].numpy().tobytes()
                ).hexdigest(),
                "training_indices_sha256": hashlib.sha256(
                    ids["train"].numpy().tobytes()
                ).hexdigest(),
                "score_sha256": hashlib.sha256(scores.tobytes()).hexdigest(),
                "selector_information": config["selection_information"],
            }
            selection_text = json.dumps(selection, indent=2) + "\n"
            (phase_dir / f"selection_seed{seed}.json").write_text(selection_text)
            selection_hash = hashlib.sha256(selection_text.encode()).hexdigest()
            selection_records.append(
                {
                    "phase": phase,
                    "seed": seed,
                    "selection_sha256": selection_hash,
                    "score_estimation_seconds": score_seconds,
                    "search_seconds": search_seconds,
                    "permuted_search_seconds": permutation_search_seconds,
                }
            )
            for units, mode in itertools.product(
                config["units"], config["support_modes"]
            ):
                if mode == "learned":
                    supports = selected[:units]
                elif mode == "permuted_scores":
                    supports = permuted_selected[:units]
                else:
                    supports = core.make_supports(
                        config["d"],
                        config["k"],
                        units,
                        seed + 10000,
                        mode,
                        config["interactions"],
                    ).numpy()
                destination = phase_dir / f"{mode}_S{units:02d}_seed{seed}"
                row = train_cell(
                    core,
                    config,
                    x,
                    y,
                    ids,
                    supports,
                    units,
                    seed,
                    mode,
                    destination,
                    selection_hash,
                )
                row["phase"] = phase
                rows.append(row)
                phase_rows.append(row)
                (phase_dir / "results.json").write_text(
                    json.dumps(phase_rows, indent=2) + "\n"
                )
                with (phase_dir / "results.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(phase_rows)
                print(json.dumps(row), flush=True)
    summary = {
        "protocol_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "trained_cells": len(rows),
        "cpu_training_seconds": sum(row["cpu_seconds"] for row in rows),
        "selection_records": selection_records,
        "results": rows,
        "claim_boundary": config["claim_boundary"],
        "selection": config["checkpoint_selection"],
        "resource": "one CPU thread; no GPU",
        "extra_training_resources": "120 estimated order-3 coefficients, 210 candidate k4 supports, training-only search; deployed real scalar P excludes discarded search scores and separately reports discrete support indices",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else protocol()
    if args.quick:
        config.update(
            {
                "units": [4],
                "steps": 10,
                "confirmation_seeds": [],
                "timing_canary_only": True,
            }
        )
    summary = run_campaign(args.output_dir, config)
    print(
        json.dumps(
            {
                "trained_cells": summary["trained_cells"],
                "cpu_training_seconds": summary["cpu_training_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

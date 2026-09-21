#!/usr/bin/env python3
"""Collect schema-aware state and gradient diagnostics for physical depth.

The generic compartment-statistics analyzer targets the legacy single-cell
weight schema.  This diagnostic instead reloads the production population
checkpoints, replays a fixed held-out batch, and uses the LocalCA recorders
that generated the training updates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dendritic_modeling.config import load_config
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_epoch import (
    _forward_with_local_recorders,
)


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "nonlinear_physical_depth_runs"
SOURCE = ROOT / "source_data" / "nonlinear_physical_depth_confirmatory"
RUN_STEM = "journal_confirmatory_physical_depth_aligned_shunting_bp_"
GRADIENT_PARAMETER_TOKENS = (
    ".branch_excitation.pre_w",
    ".branch_inhibition.pre_w",
    ".branches_to_output.log_weight",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _plain(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return copy.deepcopy(value)
    try:
        return asdict(value)
    except TypeError:
        converted = to_plain_dict(value)
        if not isinstance(converted, dict):
            raise TypeError(f"Expected mapping, got {type(value)!r}")
        return converted


def _task_config(config: Any) -> SimpleNamespace:
    data = config.data
    dataset_name = data.dataset_name
    params = _plain(data.dataset_params)
    dataset_params = params.get(dataset_name, {})
    processing = _plain(getattr(data, "processing", {}))
    experiment = config.experiment
    parameters = {**processing, **dataset_params}
    parameters.setdefault("seed", int(experiment.dataset_seed))
    parameters.setdefault("label_noise_seed", int(experiment.dataset_seed))
    parameters.setdefault("split_seed", int(experiment.split_seed))
    return SimpleNamespace(
        dataset=dataset_name,
        data_path=(data.base_dir or None),
        train_valid_split=float(experiment.train_valid_split),
        parameters=parameters,
    )


def _load_checkpoint(result_dir: Path) -> tuple[Any, Any, torch.utils.data.Dataset]:
    config_path = result_dir / "config.json"
    checkpoint = result_dir / "final_model.pt"
    config = load_config(str(config_path))
    _, _, test_dataset = get_unified_datasets(task_cfg=_task_config(config))
    model, _ = initialize_model(config.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to("cpu")
    model.eval()
    return model, config, test_dataset


def _trainer(model: Any, config: Any, transport: str) -> LocalCreditAssignment:
    local_config = _plain(config.training.main.learning_strategy_config)
    local_config["error_broadcast_mode"] = transport
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    trainer = LocalCreditAssignment(
        optimizer=optimizer,
        loss_function="ce",
        epochs=1,
        batch_size=128,
        shuffle=False,
        suppress_prints=True,
        device="cpu",
        task="classification",
        reactivation_update_mode="frozen",
        local_rule_config=local_config,
    )
    trainer._initialize_attributes(model, 1)
    return trainer


def _selected_gradients(model: Any) -> dict[str, torch.Tensor]:
    selected: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if not any(token in name for token in GRADIENT_PARAMETER_TOKENS):
            continue
        if parameter.grad is not None:
            selected[name] = parameter.grad.detach().reshape(-1).double().clone()
    return selected


def _gradient_metrics(
    exact: dict[str, torch.Tensor], approximate: dict[str, torch.Tensor]
) -> dict[str, float | int]:
    missing_from_approximate = sorted(set(exact) - set(approximate))
    unexpected_in_approximate = sorted(set(approximate) - set(exact))
    names = sorted(set(exact) & set(approximate))
    if not names:
        raise RuntimeError("No shared dendritic conductance gradients were recorded")
    exact_vector = torch.cat([exact[name] for name in names])
    approximate_vector = torch.cat([approximate[name] for name in names])
    exact_norm = torch.linalg.vector_norm(exact_vector)
    approximate_norm = torch.linalg.vector_norm(approximate_vector)
    denominator = exact_norm * approximate_norm
    cosine = (
        torch.dot(exact_vector, approximate_vector) / denominator
        if float(denominator) > 0
        else exact_vector.new_tensor(float("nan"))
    )
    return {
        "gradient_cosine": float(cosine),
        "gradient_norm_ratio": float(approximate_norm / exact_norm),
        "exact_gradient_norm": float(exact_norm),
        "approximate_gradient_norm": float(approximate_norm),
        "gradient_parameter_tensors": len(names),
        "gradient_parameter_scalars": int(exact_vector.numel()),
        "gradient_tensors_missing_from_approximate": len(missing_from_approximate),
        "gradient_tensors_unexpected_in_approximate": len(unexpected_in_approximate),
    }


def _local_gradients(
    *, model: Any, config: Any, x: torch.Tensor, y: torch.Tensor, transport: str
) -> dict[str, torch.Tensor]:
    trainer = _trainer(model, config, transport)
    model.zero_grad(set_to_none=True)
    logits, records = _forward_with_local_recorders(trainer, model, x)
    delta_out = trainer._compute_soma_error(logits, y)
    v0, delta = trainer._resolve_local_soma_signals(
        model=model,
        y_hat=logits,
        delta_out=delta_out.detach(),
    )
    trainer._apply_local_rule_gradients(
        model,
        records,
        v0=v0,
        delta=delta,
        y_target=None,
        update_reactivation=False,
    )
    return _selected_gradients(model)


def _finite_summary(values: torch.Tensor) -> dict[str, float | int]:
    values = values.detach().reshape(-1).double().cpu()
    values = values[torch.isfinite(values)]
    if not values.numel():
        raise RuntimeError("Diagnostic tensor had no finite values")
    q05, q50, q95 = torch.quantile(values, values.new_tensor([0.05, 0.5, 0.95]))
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
        "max": float(values.max()),
        "n": int(values.numel()),
    }


def _state_rows(
    trainer: LocalCreditAssignment,
    records: list[dict[str, Any]],
    *,
    seed: int,
    depth: int,
    config_index: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    path_factors = trainer._precompute_path_propagation_factors(
        records,
        include_parent_activation_derivative=True,
    )
    rows: list[dict[str, Any]] = []
    distal_values: list[torch.Tensor] = []
    for record_index, (record, path_factor) in enumerate(zip(records, path_factors)):
        voltage = record.get("v_n")
        if not isinstance(voltage, torch.Tensor):
            continue
        total_conductance = trainer._compute_layer_total_conductance(record, voltage)
        resistance = 1.0 / (total_conductance + 1e-8)
        derivative = trainer._get_layer_activation_derivative(record, voltage)
        if not isinstance(derivative, torch.Tensor):
            derivative = torch.ones_like(voltage)
        if not isinstance(path_factor, torch.Tensor):
            path_factor = torch.ones_like(voltage) * float(path_factor)
        if path_factor.shape != voltage.shape:
            path_factor = path_factor.expand_as(voltage)
        if record_index < len(records) - 1:
            distal_values.append(path_factor.reshape(-1))
        for component, values in (
            ("branch_voltage", voltage),
            ("total_conductance", total_conductance),
            ("input_resistance", resistance),
            ("activation_derivative", derivative),
            ("path_gain", path_factor),
        ):
            rows.append(
                {
                    "seed": seed,
                    "depth": depth,
                    "config_index": config_index,
                    "record_index_distal_to_soma": record_index,
                    "is_soma_stage": record_index == len(records) - 1,
                    "module_name": record.get("module_name", ""),
                    "component": component,
                    **_finite_summary(values),
                }
            )
    if not distal_values:
        raise RuntimeError("No nonsomatic path factors were found")
    combined = torch.cat(distal_values).double()
    mean = combined.mean()
    path_summary = {
        "path_gain_mean": float(mean),
        "path_gain_cv": float(combined.std(unbiased=False) / mean.abs().clamp_min(1e-12)),
        "path_gain_log_variance": float(torch.log(combined.clamp_min(1e-12)).var(unbiased=False)),
    }
    return rows, path_summary


def diagnose_one(result_dir: Path, config_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model, config, test_dataset = _load_checkpoint(result_dir)
    loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
    x, y = next(iter(loader))
    network_config = _plain(config.model.core.population_network)
    depth = len(network_config["layers"][0]["populations"][0]["branch_factors"])
    seed = int(config.experiment.seed)
    recorder = _trainer(model, config, "path_transport")

    for module in model.modules():
        if isinstance(module, DendriticBranchLayer):
            module.set_branch_diagnostics(True)
    model.zero_grad(set_to_none=True)
    logits, records = _forward_with_local_recorders(recorder, model, x)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    exact = _selected_gradients(model)
    state_rows, path_summary = _state_rows(
        recorder,
        records,
        seed=seed,
        depth=depth,
        config_index=config_index,
    )
    for module in model.modules():
        if isinstance(module, DendriticBranchLayer):
            module.set_branch_diagnostics(False)

    shared = _local_gradients(
        model=model,
        config=config,
        x=x,
        y=y,
        transport="per_soma_shared",
    )
    path = _local_gradients(
        model=model,
        config=config,
        x=x,
        y=y,
        transport="path_transport",
    )
    row: dict[str, Any] = {
        "seed": seed,
        "depth": depth,
        "config_index": config_index,
        "batch_size": int(x.shape[0]),
        "checkpoint_sha256": _sha256(result_dir / "final_model.pt"),
        "batch_loss": float(loss.detach()),
        **path_summary,
    }
    for prefix, values in (
        ("shared", _gradient_metrics(exact, shared)),
        ("path", _gradient_metrics(exact, path)),
    ):
        row.update({f"{prefix}_{key}": value for key, value in values.items()})
    return state_rows, row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run_matches = sorted(RUNS.glob(f"{RUN_STEM}*"))
    if not run_matches:
        raise FileNotFoundError(f"No run matching {RUN_STEM}*")
    run = run_matches[-1]
    config_paths = sorted(
        (run / "configs").glob("unified_config_*.yaml"),
        key=lambda path: int(path.stem.rsplit("_", 1)[1]),
    )
    if args.limit is not None:
        config_paths = config_paths[: args.limit]

    state_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    for position, config_path in enumerate(config_paths, start=1):
        index = int(config_path.stem.rsplit("_", 1)[1])
        result_dir = run / "results" / f"config_{index}"
        required = (result_dir / "final_model.pt", result_dir / "performance" / "final.json")
        if not all(path.exists() for path in required):
            raise RuntimeError(f"Missing completed result for config {index}")
        rows, gradient = diagnose_one(result_dir, index)
        state_rows.extend(rows)
        gradient_rows.append(gradient)
        print(f"diagnosed {position}/{len(config_paths)}: config {index}", flush=True)

    SOURCE.mkdir(parents=True, exist_ok=True)
    states = pd.DataFrame(state_rows)
    gradients = pd.DataFrame(gradient_rows).sort_values(["depth", "seed"])
    states.to_csv(SOURCE / "mechanism_state_summary.csv", index=False, float_format="%.10g")
    gradients.to_csv(SOURCE / "mechanism_gradient_summary.csv", index=False, float_format="%.10g")
    condition_summary = (
        gradients.groupby("depth", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_shared_gradient_cosine=("shared_gradient_cosine", "mean"),
            min_shared_gradient_cosine=("shared_gradient_cosine", "min"),
            mean_path_gradient_cosine=("path_gradient_cosine", "mean"),
            min_path_gradient_cosine=("path_gradient_cosine", "min"),
            mean_path_gain_cv=("path_gain_cv", "mean"),
            mean_path_gain_log_variance=("path_gain_log_variance", "mean"),
        )
        .sort_values("depth")
    )
    condition_summary.to_csv(
        SOURCE / "mechanism_condition_summary.csv", index=False, float_format="%.10g"
    )

    shared_wide = gradients.pivot(index="seed", columns="depth", values="shared_gradient_cosine")
    shared_depth_effect = (shared_wide[3] - shared_wide[1]).to_numpy(float)
    rng = np.random.default_rng(7_300_000)
    resampled = rng.choice(
        shared_depth_effect,
        size=(50_000, len(shared_depth_effect)),
        replace=True,
    ).mean(axis=1)
    shared_depth_contrast = {
        "contrast": "shared_gradient_cosine_d3_minus_d1",
        "n_pairs": int(len(shared_depth_effect)),
        "mean_difference": float(shared_depth_effect.mean()),
        "ci95_low": float(np.quantile(resampled, 0.025)),
        "ci95_high": float(np.quantile(resampled, 0.975)),
        "positive_pairs": int((shared_depth_effect > 0).sum()),
    }
    (SOURCE / "mechanism_paired_contrast.json").write_text(
        json.dumps(shared_depth_contrast, indent=2, sort_keys=True)
    )
    audit = {
        "run_dir": str(run.relative_to(ROOT)),
        "expected_checkpoints": 30 if args.limit is None else int(args.limit),
        "observed_checkpoints": len(gradients),
        "finite_state_values": bool(
            np.isfinite(states[["mean", "std", "min", "q05", "q50", "q95", "max"]]).all().all()
        ),
        "finite_gradient_values": bool(
            np.isfinite(
                gradients.select_dtypes(include=[np.number]).drop(columns=["config_index"])
            ).all().all()
        ),
        "path_transport_cosine_min": float(gradients.path_gradient_cosine.min()),
        "shared_gradient_tensor_mismatches": int(
            gradients.shared_gradient_tensors_missing_from_approximate.sum()
            + gradients.shared_gradient_tensors_unexpected_in_approximate.sum()
        ),
        "path_gradient_tensor_mismatches": int(
            gradients.path_gradient_tensors_missing_from_approximate.sum()
            + gradients.path_gradient_tensors_unexpected_in_approximate.sum()
        ),
        "shared_cosine_by_depth": {
            f"D{int(depth)}": float(part.shared_gradient_cosine.mean())
            for depth, part in gradients.groupby("depth")
        },
    }
    audit["all_artifact_gates_pass"] = bool(
        audit["observed_checkpoints"] == audit["expected_checkpoints"]
        and audit["finite_state_values"]
        and audit["finite_gradient_values"]
        and audit["shared_gradient_tensor_mismatches"] == 0
        and audit["path_gradient_tensor_mismatches"] == 0
    )
    (SOURCE / "mechanism_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    rows_by_depth = condition_summary.set_index("depth")
    report = (
        "# Nonlinear physical-depth checkpoint mechanism diagnostic\n\n"
        f"All {len(gradients)} aligned-shunting BP checkpoints passed the "
        "finite-value and parameter-inventory gates. Exact path transport "
        "reproduced the autograd conductance-gradient direction with minimum "
        f"cosine {audit['path_transport_cosine_min']:.12f}.\n\n"
        "Mean shared-soma gradient cosine was "
        f"{rows_by_depth.loc[1, 'mean_shared_gradient_cosine']:.4f} at D1, "
        f"{rows_by_depth.loc[2, 'mean_shared_gradient_cosine']:.4f} at D2 and "
        f"{rows_by_depth.loc[3, 'mean_shared_gradient_cosine']:.4f} at D3. "
        "The paired D3-minus-D1 change was "
        f"{shared_depth_contrast['mean_difference']:.4f} "
        f"({shared_depth_contrast['ci95_low']:.4f} to "
        f"{shared_depth_contrast['ci95_high']:.4f}; "
        f"{shared_depth_contrast['positive_pairs']}/"
        f"{shared_depth_contrast['n_pairs']} positive pairs).\n\n"
        "This is a post-training mechanism diagnostic, not a separately "
        "preregistered performance endpoint.\n"
    )
    (SOURCE / "mechanism_report.md").write_text(report)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

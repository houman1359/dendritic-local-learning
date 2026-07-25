#!/usr/bin/env python3
"""Measure finite-step loss change after matching LocalCA and exact gradient norms.

The diagnostic perturbs only non-somatic dendritic parameters. For each saved
checkpoint and batch, it compares:

1. the exact branch gradient;
2. the submitted LocalCA branch gradient after one global rescaling to the
   exact branch-gradient norm; and
3. the unscaled LocalCA branch gradient at the same learning rate.

Step sizes are specified as fractions of the branch-parameter norm, making the
finite perturbation comparable across architectures and checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from measure_layer_soma_factorial import (  # noqa: E402
    _apply_local_grads_from_errors,
    _branch_layer_index,
    _capture_forward_backward,
    _component_from_name,
    _condition_errors,
    _discover_run_dirs,
    _exact_soma_seeds,
    _get_batch,
    _get_value,
    _group_records_by_core_layer,
    _load_config_from_run,
    _load_model,
    _make_helper,
    _set_encoder_input_dim,
    _to_plain_dict,
)
from measure_theory_diagnostics import _compute_loss  # noqa: E402


def _branch_parameter_names(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    exact_grads: dict[str, torch.Tensor],
    local_grads: dict[str, torch.Tensor],
) -> list[str]:
    """Return the branch set used by the matched 3F gradient diagnostic."""
    max_stage_by_core: dict[int, int] = {}
    for rec in records:
        core_idx = int(rec.get("core_layer_index", -1))
        stage_idx = int(rec.get("branch_layer_index", -1))
        max_stage_by_core[core_idx] = max(max_stage_by_core.get(core_idx, -1), stage_idx)

    names: list[str] = []
    model_names = {name for name, _ in model.named_parameters()}
    for name in sorted(model_names & set(exact_grads) & set(local_grads)):
        component = _component_from_name(name)
        if component not in {
            "excitatory_synapse",
            "inhibitory_synapse",
            "dendritic_conductance",
        }:
            continue
        core_idx = -1
        for rec in records:
            if name.startswith(str(rec["layer_name"])):
                core_idx = int(rec.get("core_layer_index", -1))
                break
        is_soma_coupling = (
            component == "dendritic_conductance"
            and _branch_layer_index(name) == max_stage_by_core.get(core_idx, -2)
        )
        if not is_soma_coupling:
            names.append(name)
    return names


def _global_norm(grads: dict[str, torch.Tensor], names: list[str]) -> float:
    return float(
        torch.sqrt(
            sum(grads[name].detach().float().square().sum() for name in names)
        ).item()
    )


def _global_parameter_norm(
    parameters: dict[str, torch.nn.Parameter], names: list[str]
) -> float:
    return float(
        torch.sqrt(
            sum(parameters[name].detach().float().square().sum() for name in names)
        ).item()
    )


def _global_cosine(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    names: list[str],
) -> float:
    dot = sum(
        (left[name].detach().float() * right[name].detach().float()).sum()
        for name in names
    )
    left_norm = torch.sqrt(
        sum(left[name].detach().float().square().sum() for name in names)
    )
    right_norm = torch.sqrt(
        sum(right[name].detach().float().square().sum() for name in names)
    )
    denom = left_norm * right_norm
    return float((dot / denom).item()) if float(denom.item()) > 0.0 else float("nan")


def _evaluate_loss(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    loss_name: str,
) -> float:
    with torch.no_grad():
        return float(_compute_loss(loss_name, model(x_batch), y_batch).item())


def analyze_run(
    run_dir: Path,
    *,
    batch_size: int,
    split: str,
    device: torch.device,
    relative_steps: list[float],
    condition: str,
) -> pd.DataFrame:
    config = _load_config_from_run(run_dir)
    x_batch, y_batch = _get_batch(config, split=split, batch_size=batch_size)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    _set_encoder_input_dim(config, x_batch)

    main_cfg = _get_value(config.training, "main", {})
    common_cfg = _to_plain_dict(_get_value(main_cfg, "common", {}))
    loss_name = str(common_cfg.get("loss_function", "cat_nll"))
    helper = _make_helper(config, rule_variant="3f", loss_name=loss_name)
    model = _load_model(config, run_dir, device)

    records, exact_grads, v0_direct, delta_direct, captured_loss = (
        _capture_forward_backward(
            model,
            x_batch,
            y_batch,
            helper=helper,
            loss_name=loss_name,
        )
    )
    exact_seeds = _exact_soma_seeds(records, batch_size=int(x_batch.size(0)))
    grouped = _group_records_by_core_layer(records)
    conditions = _condition_errors(helper, grouped, exact_seeds, delta_direct)
    if condition not in conditions:
        raise KeyError(f"Condition {condition!r} unavailable for {run_dir}")
    local_grads = _apply_local_grads_from_errors(
        model,
        helper,
        records,
        conditions[condition],
        v0=v0_direct,
    )

    parameters = dict(model.named_parameters())
    names = _branch_parameter_names(model, records, exact_grads, local_grads)
    if not names:
        raise RuntimeError(f"No shared non-somatic branch parameters in {run_dir}")
    originals = {name: parameters[name].detach().clone() for name in names}

    exact_norm = _global_norm(exact_grads, names)
    local_norm = _global_norm(local_grads, names)
    theta_norm = _global_parameter_norm(parameters, names)
    local_scale = exact_norm / max(local_norm, 1e-30)
    cosine = _global_cosine(local_grads, exact_grads, names)
    baseline_loss = _evaluate_loss(model, x_batch, y_batch, loss_name)
    if not np.isclose(baseline_loss, captured_loss, rtol=1e-5, atol=1e-7):
        raise RuntimeError(
            f"Baseline loss changed after gradient capture: {captured_loss} vs "
            f"{baseline_loss} for {run_dir}"
        )

    directions: dict[str, tuple[dict[str, torch.Tensor], float]] = {
        "exact": (exact_grads, 1.0),
        "local_norm_matched": (local_grads, local_scale),
        "local_raw": (local_grads, 1.0),
    }
    rows: list[dict[str, Any]] = []
    try:
        for relative_step in relative_steps:
            eta = (
                float(relative_step) * theta_norm / max(exact_norm, 1e-30)
            )
            for direction_name, (direction, multiplier) in directions.items():
                with torch.no_grad():
                    for name in names:
                        parameters[name].copy_(originals[name])
                        parameters[name].add_(
                            direction[name],
                            alpha=-eta * float(multiplier),
                        )
                loss_after = _evaluate_loss(model, x_batch, y_batch, loss_name)
                applied_norm = eta * float(multiplier) * (
                    exact_norm if direction_name == "exact" else local_norm
                )
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "run_name": run_dir.name,
                        "seed": int(
                            _get_value(
                                _get_value(config, "experiment", {}),
                                "seed",
                                -1,
                            )
                        ),
                        "dataset": str(config.data.dataset_name),
                        "network_type": str(config.model.core.type),
                        "split": split,
                        "condition": condition,
                        "direction": direction_name,
                        "relative_step": float(relative_step),
                        "eta": eta,
                        "loss_before": baseline_loss,
                        "loss_after": loss_after,
                        "loss_decrease": baseline_loss - loss_after,
                        "relative_loss_decrease": (
                            baseline_loss - loss_after
                        )
                        / max(abs(baseline_loss), 1e-30),
                        "step_norm": applied_norm,
                        "step_to_parameter_norm": applied_norm
                        / max(theta_norm, 1e-30),
                        "branch_parameter_norm": theta_norm,
                        "exact_branch_gradient_norm": exact_norm,
                        "raw_local_branch_gradient_norm": local_norm,
                        "local_to_exact_norm_ratio": local_norm
                        / max(exact_norm, 1e-30),
                        "local_norm_match_scale": local_scale,
                        "branch_concatenated_cosine": cosine,
                        "n_branch_parameters": int(
                            sum(parameters[name].numel() for name in names)
                        ),
                    }
                )
    finally:
        with torch.no_grad():
            for name in names:
                parameters[name].copy_(originals[name])

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--condition",
        default="approx_direct_code_per_soma",
    )
    parser.add_argument(
        "--relative-steps",
        type=float,
        nargs="+",
        default=[1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
    )
    args = parser.parse_args()

    if bool(args.run_dir) == bool(args.sweep_dir):
        raise ValueError("Specify exactly one of --run-dir or --sweep-dir.")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    root = args.run_dir or args.sweep_dir
    run_dirs = _discover_run_dirs(root)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]

    frames: list[pd.DataFrame] = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir}", flush=True)
        frames.append(
            analyze_run(
                run_dir,
                batch_size=args.batch_size,
                split=args.split,
                device=device,
                relative_steps=args.relative_steps,
                condition=args.condition,
            )
        )
    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False)
    manifest = {
        "root": str(root),
        "n_runs": len(run_dirs),
        "batch_size": args.batch_size,
        "split": args.split,
        "condition": args.condition,
        "relative_steps": args.relative_steps,
        "device": str(device),
    }
    args.output_csv.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Saved {len(output)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()

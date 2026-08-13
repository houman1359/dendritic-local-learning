#!/usr/bin/env python3
"""Connect fixed-state feedback-field diagnostics to finite-step progress.

For the same saved checkpoint and batch, this script compares five feedback
fields at four levels:

1. optimally rescaled capture of the exact *dendritic* compartment-error field;
2. optimally rescaled capture of the resulting non-somatic branch gradient;
3. concatenated branch-gradient cosine; and
4. loss decrease after matching the candidate update to the exact branch
   gradient norm.

The forward state, exact soma errors, eligibility factors, parameters, and
batch are shared across conditions.  The diagnostic is checkpoint-only and
does not imply that oracle feedback fields are biologically available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from measure_layer_soma_factorial import (  # noqa: E402
    _apply_local_grads_from_errors,
    _capture_forward_backward,
    _condition_errors,
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
from measure_norm_matched_one_step import (  # noqa: E402
    _branch_parameter_names,
    _evaluate_loss,
    _global_norm,
    _global_parameter_norm,
)


def _default_run_dirs() -> list[Path]:
    root = (
        DRAFT_ROOT
        / "local_sweep_runs"
        / "gradient_fidelity_vs_ie_nonnegativeinput_fix_20260409164042"
        / "results"
    )
    return (
        [root / f"config_{index}" for index in range(10, 15)]
        + [root / f"config_{index}" for index in range(60, 65)]
    )


def _global_scalar_errors(
    helper: Any,
    records: list[dict[str, Any]],
    direct_seed: torch.Tensor,
) -> dict[str, torch.Tensor]:
    scalar = helper._reduce_error_to_scalar(direct_seed)
    errors: dict[str, torch.Tensor] = {}
    for record in records:
        voltage = record["v_n"]
        errors[record["layer_name"]] = scalar.to(
            device=voltage.device,
            dtype=voltage.dtype,
        ).expand_as(voltage)
    return errors


def _dendritic_layer_names(
    records: list[dict[str, Any]],
) -> list[str]:
    grouped = _group_records_by_core_layer(records)
    return [
        str(record["layer_name"])
        for group in grouped.values()
        for record in group[:-1]
    ]


def _vector_metrics(
    candidate: torch.Tensor,
    exact: torch.Tensor,
) -> dict[str, float]:
    candidate = candidate.detach().float().reshape(-1)
    exact = exact.detach().float().reshape(-1)
    candidate_norm = float(candidate.norm().item())
    exact_norm = float(exact.norm().item())
    dot = float(torch.dot(candidate, exact).item())
    denom = candidate_norm * exact_norm
    cosine = dot / denom if denom > 0.0 else float("nan")
    scale = dot / max(candidate_norm**2, 1e-30)
    residual = float((scale * candidate - exact).norm().item()) / max(
        exact_norm,
        1e-30,
    )
    return {
        "cosine": cosine,
        "optimal_scale": scale,
        "scaled_residual": residual,
        "scaled_capture": 1.0 - residual**2,
        "candidate_to_exact_norm_ratio": candidate_norm
        / max(exact_norm, 1e-30),
    }


def _field_metrics(
    candidate: dict[str, torch.Tensor],
    exact: dict[str, torch.Tensor],
    layer_names: list[str],
) -> dict[str, float]:
    candidate_vector = torch.cat(
        [candidate[name].detach().float().reshape(-1) for name in layer_names]
    )
    exact_vector = torch.cat(
        [exact[name].detach().float().reshape(-1) for name in layer_names]
    )
    return _vector_metrics(candidate_vector, exact_vector)


def _gradient_metrics(
    candidate: dict[str, torch.Tensor],
    exact: dict[str, torch.Tensor],
    parameter_names: list[str],
) -> dict[str, float]:
    candidate_vector = torch.cat(
        [candidate[name].detach().float().reshape(-1) for name in parameter_names]
    )
    exact_vector = torch.cat(
        [exact[name].detach().float().reshape(-1) for name in parameter_names]
    )
    return _vector_metrics(candidate_vector, exact_vector)


def analyze_run(
    run_dir: Path,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
    relative_steps: list[float],
) -> list[dict[str, Any]]:
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
    family_errors = {
        "global_scalar_available": _global_scalar_errors(
            helper,
            records,
            delta_direct,
        ),
        "submitted_mw": conditions["approx_direct_code_per_soma"],
        "ancestry_available": conditions["approx_direct_blockwise_per_soma"],
        "ancestry_exact_soma": conditions["exact_soma_blockwise_per_soma"],
        "exact_transport": conditions["exact_soma_path_transport"],
    }
    family_grads = {
        family: _apply_local_grads_from_errors(
            model,
            helper,
            records,
            errors,
            v0=v0_direct,
        )
        for family, errors in family_errors.items()
    }

    exact_error = conditions["exact_soma_path_transport"]
    layer_names = _dendritic_layer_names(records)
    parameters = dict(model.named_parameters())
    representative = next(iter(family_grads.values()))
    parameter_names = _branch_parameter_names(
        model,
        records,
        exact_grads,
        representative,
    )
    originals = {
        name: parameters[name].detach().clone() for name in parameter_names
    }
    exact_norm = _global_norm(exact_grads, parameter_names)
    theta_norm = _global_parameter_norm(parameters, parameter_names)
    baseline_loss = _evaluate_loss(model, x_batch, y_batch, loss_name)
    if not np.isclose(baseline_loss, captured_loss, rtol=1e-5, atol=1e-7):
        raise RuntimeError(
            f"Captured and replayed losses differ in {run_dir}: "
            f"{captured_loss} vs {baseline_loss}"
        )

    exact_loss_after: dict[float, float] = {}
    try:
        for relative_step in relative_steps:
            eta = float(relative_step) * theta_norm / max(exact_norm, 1e-30)
            with torch.no_grad():
                for name in parameter_names:
                    parameters[name].copy_(originals[name])
                    parameters[name].add_(exact_grads[name], alpha=-eta)
            exact_loss_after[relative_step] = _evaluate_loss(
                model,
                x_batch,
                y_batch,
                loss_name,
            )

        rows: list[dict[str, Any]] = []
        for family, gradients in family_grads.items():
            field = _field_metrics(
                family_errors[family],
                exact_error,
                layer_names,
            )
            gradient = _gradient_metrics(
                gradients,
                exact_grads,
                parameter_names,
            )
            local_norm = _global_norm(gradients, parameter_names)
            norm_scale = exact_norm / max(local_norm, 1e-30)
            for relative_step in relative_steps:
                eta = (
                    float(relative_step)
                    * theta_norm
                    / max(exact_norm, 1e-30)
                )
                with torch.no_grad():
                    for name in parameter_names:
                        parameters[name].copy_(originals[name])
                        parameters[name].add_(
                            gradients[name],
                            alpha=-eta * norm_scale,
                        )
                loss_after = _evaluate_loss(
                    model,
                    x_batch,
                    y_batch,
                    loss_name,
                )
                exact_decrease = (
                    baseline_loss - exact_loss_after[relative_step]
                )
                local_decrease = baseline_loss - loss_after
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
                        "feedback_family": family,
                        "relative_step": float(relative_step),
                        "loss_before": baseline_loss,
                        "exact_loss_decrease": exact_decrease,
                        "norm_matched_loss_decrease": local_decrease,
                        "norm_matched_fraction_of_exact": local_decrease
                        / max(abs(exact_decrease), 1e-30),
                        "norm_matched_is_descent": local_decrease > 0.0,
                        "field_cosine": field["cosine"],
                        "field_scaled_capture": field["scaled_capture"],
                        "field_scaled_residual": field["scaled_residual"],
                        "field_to_exact_norm_ratio": field[
                            "candidate_to_exact_norm_ratio"
                        ],
                        "gradient_cosine": gradient["cosine"],
                        "gradient_scaled_capture": gradient[
                            "scaled_capture"
                        ],
                        "gradient_scaled_residual": gradient[
                            "scaled_residual"
                        ],
                        "gradient_to_exact_norm_ratio": gradient[
                            "candidate_to_exact_norm_ratio"
                        ],
                        "gradient_norm_match_scale": norm_scale,
                    }
                )
        return rows
    finally:
        with torch.no_grad():
            for name in parameter_names:
                parameters[name].copy_(originals[name])


def _summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "field_cosine",
        "field_scaled_capture",
        "field_scaled_residual",
        "field_to_exact_norm_ratio",
        "gradient_cosine",
        "gradient_scaled_capture",
        "gradient_scaled_residual",
        "gradient_to_exact_norm_ratio",
        "norm_matched_loss_decrease",
        "norm_matched_fraction_of_exact",
        "norm_matched_is_descent",
    ]
    return (
        frame.groupby(
            [
                "dataset",
                "network_type",
                "split",
                "feedback_family",
                "relative_step",
            ],
            dropna=False,
        )[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
        .pipe(
            lambda table: table.set_axis(
                [
                    "_".join(str(part) for part in column if str(part))
                    if isinstance(column, tuple)
                    else str(column)
                    for column in table.columns
                ],
                axis=1,
            )
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        nargs="+",
        default=["test"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--relative-step",
        type=float,
        nargs="+",
        default=[1e-6, 1e-5, 1e-4, 1e-3],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=(
            DRAFT_ROOT
            / "figures"
            / "data"
            / "feedback_learning_relevance"
        ),
    )
    args = parser.parse_args()

    run_dirs = args.run_dir or _default_run_dirs()
    missing = [run_dir for run_dir in run_dirs if not run_dir.exists()]
    if missing:
        raise FileNotFoundError(f"Missing run directories: {missing}")
    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    rows: list[dict[str, Any]] = []
    total = len(run_dirs) * len(args.split)
    completed = 0
    for run_dir in run_dirs:
        for split in args.split:
            completed += 1
            print(f"[{completed}/{total}] {run_dir} ({split})", flush=True)
            rows.extend(
                analyze_run(
                    run_dir,
                    split=split,
                    batch_size=args.batch_size,
                    device=device,
                    relative_steps=args.relative_step,
                )
            )
    frame = pd.DataFrame(rows)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        args.output_prefix.with_name(
            args.output_prefix.name + "_runs.csv"
        ),
        index=False,
    )
    _summarize(frame).to_csv(
        args.output_prefix.with_name(
            args.output_prefix.name + "_summary.csv"
        ),
        index=False,
    )
    print(f"Wrote outputs with prefix {args.output_prefix}")


if __name__ == "__main__":
    main()

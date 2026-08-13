#!/usr/bin/env python3
"""Calibrate backward-only inhibition removal at fixed forward states.

Recorded voltages, activation derivatives, couplings, soma errors, and local
eligibility are held fixed.  Only inhibitory conductance in parent resistance
factors is scaled by a removal fraction. Fractions above one are explicitly
non-physical sensitivity extensions; the script records how often their
counterfactual total conductance reaches a numerical floor.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
DRAFT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from measure_layer_soma_factorial import (  # noqa: E402
    _activation_stats,
    _alignment_rows,
    _apply_local_grads_from_errors,
    _branch_level_summary,
    _capture_forward_backward,
    _condition_errors,
    _exact_soma_seeds,
    _expand_soma_to_tree,
    _get_batch,
    _get_value,
    _group_records_by_core_layer,
    _load_config_from_run,
    _load_model,
    _make_helper,
    _set_encoder_input_dim,
    _to_plain_dict,
)


def _default_run_dirs() -> list[Path]:
    mnist = (
        DRAFT_ROOT
        / "local_sweep_runs"
        / "gradient_fidelity_vs_ie_nonnegativeinput_fix_20260409164042"
        / "results"
    )
    noise = (
        DRAFT_ROOT
        / "local_sweep_runs"
        / "noise_resilience_rank_bridge_nonnegativeinput_fix_20260409164042"
        / "results"
    )
    return (
        [mnist / f"config_{index}" for index in range(10, 15)]
        + [noise / f"config_{index}" for index in range(0, 5)]
    )


def _scaled_path_factors(
    helper: Any,
    group: list[dict[str, Any]],
    *,
    removal_fraction: float,
    minimum_total_fraction: float,
) -> tuple[list[torch.Tensor], dict[str, float]]:
    factors: list[torch.Tensor | None] = [None] * len(group)
    actual_factors: list[torch.Tensor | None] = [None] * len(group)
    soma_v = group[-1]["v_n"]
    factors[-1] = torch.ones_like(soma_v)
    actual_factors[-1] = torch.ones_like(soma_v)
    clipped = 0
    total_values = 0
    ratios: list[torch.Tensor] = []

    for index in range(len(group) - 2, -1, -1):
        child = group[index]
        parent = group[index + 1]
        child_v = child["v_n"]
        parent_v = parent["v_n"]
        block = parent["blk_module"]
        actual_total = helper._compute_layer_total_conductance(
            parent,
            parent_v,
        )
        inhibitory = parent.get("inh_out")
        inhibitory = (
            F.relu(inhibitory.detach())
            if isinstance(inhibitory, torch.Tensor)
            else torch.zeros_like(actual_total)
        )
        proposed_total = actual_total - float(removal_fraction) * inhibitory
        floor = actual_total * float(minimum_total_fraction)
        clipped_mask = proposed_total < floor
        counterfactual_total = torch.maximum(proposed_total, floor)
        clipped += int(clipped_mask.sum().item())
        total_values += int(clipped_mask.numel())

        parent_derivative = helper._get_layer_activation_derivative(
            parent,
            v_n=parent_v,
            v_out=parent.get("v_out"),
        )
        if not isinstance(parent_derivative, torch.Tensor):
            parent_derivative = torch.ones_like(parent_v)
        parent_factor = factors[index + 1]
        if not isinstance(parent_factor, torch.Tensor):
            parent_factor = torch.ones_like(parent_v)
        actual_parent_factor = actual_factors[index + 1]
        if not isinstance(actual_parent_factor, torch.Tensor):
            actual_parent_factor = torch.ones_like(parent_v)
        block_size = int(getattr(block, "block_size", 1))
        edge = block.weight().detach().to(
            device=child_v.device,
            dtype=child_v.dtype,
        )
        expanded_parent = helper._expand_parent_signal_to_children(
            parent_factor,
            block_size,
            child_v.size(1),
        )
        expanded_actual_parent = helper._expand_parent_signal_to_children(
            actual_parent_factor,
            block_size,
            child_v.size(1),
        )
        expanded_derivative = helper._expand_parent_signal_to_children(
            parent_derivative,
            block_size,
            child_v.size(1),
        )
        expanded_resistance = helper._expand_parent_signal_to_children(
            1.0 / counterfactual_total.clamp_min(1e-12),
            block_size,
            child_v.size(1),
        )
        factors[index] = (
            expanded_parent
            * expanded_derivative
            * expanded_resistance
            * edge.reshape(1, -1)
        )
        actual_factors[index] = (
            expanded_actual_parent
            * expanded_derivative
            * helper._expand_parent_signal_to_children(
                1.0 / actual_total.clamp_min(1e-12),
                block_size,
                child_v.size(1),
            )
            * edge.reshape(1, -1)
        )
        ratios.append(
            (actual_total / counterfactual_total.clamp_min(1e-12))
            .detach()
            .float()
            .reshape(-1)
        )

    ratio = torch.cat(ratios) if ratios else torch.ones(1)
    distal_factor = factors[0]
    actual_distal_factor = actual_factors[0]
    if isinstance(distal_factor, torch.Tensor) and isinstance(
        actual_distal_factor,
        torch.Tensor,
    ):
        actual_abs = actual_distal_factor.abs()
        valid = actual_abs > 1e-20
        distal_ratio = (
            distal_factor.abs()[valid] / actual_abs[valid]
        ).detach().float().reshape(-1)
        if distal_ratio.numel() == 0:
            distal_ratio = torch.ones(1)
    else:
        distal_ratio = torch.ones(1)
    return (
        [factor for factor in factors if isinstance(factor, torch.Tensor)],
        {
            "parent_gain_ratio_mean": float(ratio.mean().item()),
            "parent_gain_ratio_q50": float(torch.quantile(ratio, 0.50).item()),
            "parent_gain_ratio_q95": float(torch.quantile(ratio, 0.95).item()),
            "distal_path_gain_ratio_mean": float(distal_ratio.mean().item()),
            "distal_path_gain_ratio_q50": float(
                torch.quantile(distal_ratio, 0.50).item()
            ),
            "distal_path_gain_ratio_q95": float(
                torch.quantile(distal_ratio, 0.95).item()
            ),
            "parent_total_clipped_fraction": (
                clipped / total_values if total_values else 0.0
            ),
        },
    )


def _scaled_transport_errors(
    helper: Any,
    grouped: Any,
    exact_seeds: dict[int, torch.Tensor],
    *,
    removal_fraction: float,
    minimum_total_fraction: float,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, float],
    dict[int, dict[str, float]],
]:
    errors: dict[str, torch.Tensor] = {}
    diagnostics: list[dict[str, float]] = []
    diagnostics_by_core: dict[int, dict[str, float]] = {}
    for core_index, group in grouped.items():
        shared = _expand_soma_to_tree(helper, group, exact_seeds[core_index])
        factors, diagnostic = _scaled_path_factors(
            helper,
            group,
            removal_fraction=removal_fraction,
            minimum_total_fraction=minimum_total_fraction,
        )
        diagnostics.append(diagnostic)
        diagnostics_by_core[int(core_index)] = diagnostic
        for record, shared_error, factor in zip(group, shared, factors):
            errors[record["layer_name"]] = (
                shared_error.to(device=factor.device, dtype=factor.dtype) * factor
            ).detach()
    return (
        errors,
        {
            key: float(sum(item[key] for item in diagnostics) / len(diagnostics))
            for key in diagnostics[0]
        },
        diagnostics_by_core,
    )


def analyze_run(
    run_dir: Path,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
    fractions: list[float],
    minimum_total_fraction: float,
) -> list[dict[str, Any]]:
    config = _load_config_from_run(run_dir)
    x_batch, y_batch = _get_batch(config, split=split, batch_size=batch_size)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    _set_encoder_input_dim(config, x_batch)
    main_cfg = _get_value(config.training, "main", {})
    common_cfg = _to_plain_dict(_get_value(main_cfg, "common", {}))
    local_cfg = _to_plain_dict(
        _get_value(main_cfg, "learning_strategy_config", {})
    )
    loss_name = str(common_cfg.get("loss_function", "cat_nll"))
    helper = _make_helper(config, rule_variant="3f", loss_name=loss_name)
    model = _load_model(config, run_dir, device)
    records, _, v0_direct, delta_direct, loss_value = _capture_forward_backward(
        model,
        x_batch,
        y_batch,
        helper=helper,
        loss_name=loss_name,
    )
    exact_seeds = _exact_soma_seeds(records, batch_size=int(x_batch.size(0)))
    grouped = _group_records_by_core_layer(records)
    base_conditions = _condition_errors(
        helper,
        grouped,
        exact_seeds,
        delta_direct,
    )
    source_names = (
        "approx_direct_code_per_soma",
        "exact_soma_blockwise_per_soma",
    )
    source_gradients = {
        name: _apply_local_grads_from_errors(
            model,
            helper,
            records,
            base_conditions[name],
            v0=v0_direct,
        )
        for name in source_names
    }
    activation = _activation_stats(helper, records)
    seed = int(_get_value(_get_value(config, "experiment", {}), "seed", -1))
    rows: list[dict[str, Any]] = []

    for fraction in fractions:
        target_errors, diagnostics, diagnostics_by_core = _scaled_transport_errors(
            helper,
            grouped,
            exact_seeds,
            removal_fraction=fraction,
            minimum_total_fraction=minimum_total_fraction,
        )
        target_gradients = _apply_local_grads_from_errors(
            model,
            helper,
            records,
            target_errors,
            v0=v0_direct,
        )
        details: list[dict[str, Any]] = []
        for source, gradients in source_gradients.items():
            meta = {
                "run_dir": str(run_dir),
                "run_name": run_dir.name,
                "seed": seed,
                "dataset": config.data.dataset_name,
                "network_type": config.model.core.type,
                "trained_broadcast_mode": local_cfg.get("error_broadcast_mode"),
                "diagnostic_rule_variant": "3f",
                "split": split,
                "loss_value": loss_value,
                "removal_fraction": fraction,
                "source_feedback": source,
                **diagnostics,
            }
            details.extend(
                _alignment_rows(
                    gradients,
                    target_gradients,
                    meta=meta,
                    condition=f"{source}_vs_scaled_backward_target",
                    activation_stats=activation,
                )
            )
        details_frame = pd.DataFrame(details)
        scopes: list[
            tuple[str, pd.DataFrame, dict[str, float]]
        ] = [("all_populations", details_frame, diagnostics)]
        if len(grouped) > 1:
            for core_index in sorted(grouped):
                scopes.append(
                    (
                        f"population_{core_index}",
                        details_frame.loc[
                            details_frame["core_layer_index"] == int(core_index)
                        ].copy(),
                        diagnostics_by_core[int(core_index)],
                    )
                )
        for analysis_scope, scoped_details, scoped_diagnostics in scopes:
            checkpoint, _ = _branch_level_summary(scoped_details)
            if checkpoint.empty:
                continue
            for record in checkpoint.to_dict("records"):
                source = str(record["condition"]).removesuffix(
                    "_vs_scaled_backward_target"
                )
                rows.append(
                    {
                        "run_dir": str(run_dir),
                        "run_name": run_dir.name,
                        "seed": seed,
                        "dataset": config.data.dataset_name,
                        "network_type": config.model.core.type,
                        "split": split,
                        "analysis_scope": analysis_scope,
                        "source_feedback": source,
                        "removal_fraction": fraction,
                        **scoped_diagnostics,
                        "branch_numel_weighted_cosine": record[
                            "branch_numel_weighted_cosine"
                        ],
                        "branch_macro_cosine": record["branch_macro_cosine"],
                        "branch_concatenated_cosine": record[
                            "branch_concatenated_cosine"
                        ],
                        "branch_local_target_norm_ratio": record[
                            "branch_local_exact_norm_ratio"
                        ],
                    }
                )
    return rows


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "parent_gain_ratio_mean",
        "parent_gain_ratio_q50",
        "parent_gain_ratio_q95",
        "distal_path_gain_ratio_mean",
        "distal_path_gain_ratio_q50",
        "distal_path_gain_ratio_q95",
        "parent_total_clipped_fraction",
        "branch_numel_weighted_cosine",
        "branch_macro_cosine",
        "branch_concatenated_cosine",
        "branch_local_target_norm_ratio",
    ]
    return (
        frame.groupby(
            [
                "dataset",
                "network_type",
                "split",
                "analysis_scope",
                "source_feedback",
                "removal_fraction",
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
    parser.add_argument("--run-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--dataset",
        choices=["all", "mnist", "noise_resilience"],
        default="all",
    )
    parser.add_argument(
        "--split",
        choices=["train", "valid", "test"],
        nargs="+",
        default=["test"],
    )
    parser.add_argument(
        "--removal-fraction",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
    )
    parser.add_argument("--minimum-total-fraction", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=(
            DRAFT_ROOT
            / "figures"
            / "data"
            / "backward_inhibition_dose_response"
        ),
    )
    args = parser.parse_args()

    run_dirs = args.run_dir or _default_run_dirs()
    if args.dataset != "all":
        token = (
            "gradient_fidelity"
            if args.dataset == "mnist"
            else "noise_resilience"
        )
        run_dirs = [path for path in run_dirs if token in str(path)]
    missing = [path for path in run_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing run directories: {missing}")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
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
                    fractions=args.removal_fraction,
                    minimum_total_fraction=args.minimum_total_fraction,
                )
            )
    frame = pd.DataFrame(rows)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(prefix.with_name(prefix.name + "_runs.csv"), index=False)
    _summary(frame).to_csv(
        prefix.with_name(prefix.name + "_summary.csv"),
        index=False,
    )
    print(f"Wrote outputs with prefix {prefix}")


if __name__ == "__main__":
    main()

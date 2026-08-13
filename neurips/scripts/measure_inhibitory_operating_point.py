#!/usr/bin/env python3
"""Measure trained inhibitory operating points and calibrate Proposition 2.

The operating-point table reports learned inhibitory conductance, total
conductance, input resistance, and the fraction of local conductance supplied
by inhibition.  The calibration table performs a conductance-stage
finite-difference check on one proximal compartment per soma in the one-layer
MNIST trees.  It is an implementation verification and first-order-range
calibration, not an empirical discovery.
"""

from __future__ import annotations

import argparse
import re
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

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (  # noqa: E402
    TopKLinear,
)
from dendritic_modeling.scripts.script_utils.setup_utils import (  # noqa: E402
    initialize_model,
)
from measure_theory_diagnostics import (  # noqa: E402
    _get_batch,
    _hook_branch_layers,
    _load_config_from_run,
    _locate_model_path,
    _make_helper,
)


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_STAGE_RE = re.compile(r"(?:^|\.)branch_layers\.(\d+)(?:\.|$)")
_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


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
        [mnist / f"config_{idx}" for idx in range(10, 15)]
        + [noise / f"config_{idx}" for idx in range(0, 5)]
    )


def _population_and_stage(name: str, fallback_stage: int) -> tuple[int, int]:
    population_match = _LAYER_RE.search(name)
    stage_match = _STAGE_RE.search(name)
    population = int(population_match.group(1)) if population_match else 0
    stage = int(stage_match.group(1)) if stage_match else fallback_stage
    return population, stage


def _collect_forward_state(
    run_dir: Path,
    *,
    batch_size: int,
    split: str,
    device: torch.device,
) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    config = _load_config_from_run(run_dir)
    if hasattr(config.model.core, "implementation"):
        config.model.core.implementation.compile_forward = False
    x_batch, _ = _get_batch(config, split=split, batch_size=batch_size)
    x_batch = x_batch.to(device)
    input_dim = int(x_batch[0].numel())
    encoder_params = getattr(config.model.encoder, "params", None)
    if encoder_params is None:
        config.model.encoder.params = {"input_dim": input_dim}
    elif isinstance(encoder_params, dict):
        encoder_params["input_dim"] = input_dim
    else:
        encoder_params.input_dim = input_dim

    model, _ = initialize_model(config.model)
    state = torch.load(
        _locate_model_path(run_dir),
        map_location=device,
        weights_only=False,
    )
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()

    records, handles = _hook_branch_layers(model)
    topk_modules = [
        module for module in model.modules() if isinstance(module, TopKLinear)
    ]
    for module in topk_modules:
        module.cache_mask = True
    try:
        model(x_batch)
    finally:
        for handle in handles:
            handle.remove()
        for module in topk_modules:
            module.cache_mask = False

    for fallback_stage, record in enumerate(records):
        population, stage = _population_and_stage(
            str(record.get("layer_name", "")),
            fallback_stage,
        )
        record["population"] = population
        record["stage"] = stage

    meta = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "seed": int(config.experiment.seed),
        "dataset": str(config.data.dataset_name),
        "network_type": str(config.model.core.type),
        "split": split,
        "batch_size": batch_size,
    }
    return meta, _make_helper(), records


def _summary_stats(prefix: str, values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            **{f"{prefix}_q{int(q * 100):02d}": float("nan") for q in _QUANTILES},
        }
    result = {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_std": float(flat.std(unbiased=False).item()),
    }
    quantiles = torch.quantile(
        flat,
        torch.tensor(_QUANTILES, device=flat.device, dtype=flat.dtype),
    )
    for q, value in zip(_QUANTILES, quantiles):
        result[f"{prefix}_q{int(q * 100):02d}"] = float(value.item())
    return result


def _stage_role(stage: int, max_stage: int) -> str:
    if stage == max_stage:
        return "soma"
    if max_stage - stage == 1:
        return "proximal"
    if max_stage - stage == 2:
        return "distal"
    return f"distal_{max_stage - stage}"


def _operating_rows(
    meta: dict[str, Any],
    helper: Any,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    populations = sorted({int(record["population"]) for record in records})
    for population in populations:
        pop_records = [
            record
            for record in records
            if int(record["population"]) == population
        ]
        max_stage = max(int(record["stage"]) for record in pop_records)
        for record in pop_records:
            v_n = record.get("v_n")
            if not isinstance(v_n, torch.Tensor):
                continue
            inhibitory = record.get("inh_out")
            inhibitory = (
                F.relu(inhibitory.detach())
                if isinstance(inhibitory, torch.Tensor)
                else torch.zeros_like(v_n)
            )
            total = helper._compute_layer_total_conductance(record, v_n.detach())
            resistance = 1.0 / total.clamp_min(1e-12)
            fraction = inhibitory / total.clamp_min(1e-12)
            non_inhibitory = (total - inhibitory).clamp_min(1e-12)
            removal_log_gain = torch.log(total / non_inhibitory)
            removal_gain_ratio = total / non_inhibitory
            coupling_module = record.get("blk_module")
            coupling = (
                coupling_module.weight().detach()
                if coupling_module is not None
                and hasattr(coupling_module, "weight")
                else torch.empty(0, device=v_n.device, dtype=v_n.dtype)
            )
            rows.append(
                {
                    **meta,
                    "population": population,
                    "stage": int(record["stage"]),
                    "stage_role": _stage_role(int(record["stage"]), max_stage),
                    "layer_name": str(record.get("layer_name", "")),
                    "n_values": int(total.numel()),
                    "inhibition_active_fraction": float(
                        (inhibitory > 0).float().mean().item()
                    ),
                    **_summary_stats("g_inhibitory", inhibitory),
                    **_summary_stats("g_total", total),
                    **_summary_stats("input_resistance", resistance),
                    **_summary_stats("inhibitory_fraction", fraction),
                    **_summary_stats("removal_log_gain", removal_log_gain),
                    **_summary_stats("removal_gain_ratio", removal_gain_ratio),
                    **_summary_stats("delta_g_half", total),
                    **_summary_stats("child_coupling", coupling),
                }
            )
    return rows


def _path_gains(
    records: list[dict[str, Any]],
    helper: Any,
    *,
    conductance_overrides: dict[int, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    ordered = sorted(records, key=lambda record: int(record["stage"]))
    gains: dict[int, torch.Tensor] = {}
    max_stage = int(ordered[-1]["stage"])
    soma_v = ordered[-1]["v_n"]
    gains[max_stage] = torch.ones_like(soma_v)
    by_stage = {int(record["stage"]): record for record in ordered}
    for child_stage in range(max_stage - 1, -1, -1):
        child = by_stage[child_stage]
        parent = by_stage[child_stage + 1]
        child_v = child["v_n"]
        parent_v = parent["v_n"]
        block = parent.get("blk_module")
        if block is None or not hasattr(block, "weight"):
            raise RuntimeError(
                f"Missing parent coupling for stage {child_stage + 1}"
            )
        parent_total = (
            conductance_overrides[child_stage + 1]
            if conductance_overrides
            and child_stage + 1 in conductance_overrides
            else helper._compute_layer_total_conductance(parent, parent_v.detach())
        )
        parent_resistance = 1.0 / parent_total.clamp_min(1e-12)
        block_size = int(getattr(block, "block_size", 1))
        edge = block.weight().detach().to(
            device=child_v.device,
            dtype=child_v.dtype,
        )
        if edge.numel() != child_v.size(1):
            raise RuntimeError(
                f"Coupling width {edge.numel()} != child width {child_v.size(1)}"
            )
        expanded_gain = helper._expand_parent_signal_to_children(
            gains[child_stage + 1],
            block_size,
            child_v.size(1),
        )
        expanded_resistance = helper._expand_parent_signal_to_children(
            parent_resistance,
            block_size,
            child_v.size(1),
        )
        gains[child_stage] = expanded_gain * expanded_resistance * edge.reshape(1, -1)
    return gains


def _relative_error(measured: torch.Tensor, predicted: torch.Tensor) -> float:
    denominator = predicted.norm().clamp_min(1e-30)
    return float(((measured - predicted).norm() / denominator).item())


def _correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.detach().float().reshape(-1)
    y = y.detach().float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    if denominator <= 0:
        return float("nan")
    return float((torch.dot(x, y) / denominator).item())


def _calibration_rows(
    meta: dict[str, Any],
    helper: Any,
    records: list[dict[str, Any]],
    delta_values: list[float],
) -> list[dict[str, Any]]:
    """Calibrate one proximal-site intervention in one-population trees."""
    populations = sorted({int(record["population"]) for record in records})
    if len(populations) != 1:
        return []
    ordered = sorted(records, key=lambda record: int(record["stage"]))
    max_stage = int(ordered[-1]["stage"])
    if max_stage < 2:
        return []
    by_stage = {int(record["stage"]): record for record in ordered}
    proximal_stage = max_stage - 1
    distal_stage = proximal_stage - 1
    proximal = by_stage[proximal_stage]
    distal = by_stage[distal_stage]
    proximal_v = proximal["v_n"]
    distal_v = distal["v_n"]
    n_soma = int(by_stage[max_stage]["v_n"].size(1))
    if proximal_v.size(1) % n_soma != 0:
        raise RuntimeError("Proximal width is not divisible by soma count")
    proximal_per_soma = proximal_v.size(1) // n_soma
    selected = torch.zeros_like(proximal_v)
    selected_indices = torch.arange(
        0,
        proximal_v.size(1),
        proximal_per_soma,
        device=proximal_v.device,
    )
    selected[:, selected_indices] = 1.0

    proximal_total = helper._compute_layer_total_conductance(
        proximal,
        proximal_v.detach(),
    )
    proximal_resistance = 1.0 / proximal_total.clamp_min(1e-12)
    block = proximal.get("blk_module")
    if block is None:
        raise RuntimeError("Proximal stage lacks child coupling")
    block_size = int(getattr(block, "block_size", 1))
    descendant_mask = helper._expand_parent_signal_to_children(
        selected,
        block_size,
        distal_v.size(1),
    ).bool()
    sister_mask = ~descendant_mask
    baseline = _path_gains(ordered, helper)
    rows: list[dict[str, Any]] = []
    for delta_g in delta_values:
        perturbed_total = proximal_total + float(delta_g) * selected
        perturbed = _path_gains(
            ordered,
            helper,
            conductance_overrides={proximal_stage: perturbed_total},
        )
        measured = torch.log(
            perturbed[distal_stage].abs().clamp_min(1e-30)
            / baseline[distal_stage].abs().clamp_min(1e-30)
        )
        exact_parent = torch.log(
            proximal_total
            / (proximal_total + float(delta_g) * selected)
        )
        exact = helper._expand_parent_signal_to_children(
            exact_parent,
            block_size,
            distal_v.size(1),
        )
        first_parent = -proximal_resistance * float(delta_g) * selected
        first = helper._expand_parent_signal_to_children(
            first_parent,
            block_size,
            distal_v.size(1),
        )
        measured_desc = measured[descendant_mask]
        exact_desc = exact[descendant_mask]
        first_desc = first[descendant_mask]
        self_change = torch.log(
            perturbed[proximal_stage].abs().clamp_min(1e-30)
            / baseline[proximal_stage].abs().clamp_min(1e-30)
        )
        selected_total = proximal_total[selected.bool()]
        selected_resistance = proximal_resistance[selected.bool()]
        selected_inhibitory = (
            F.relu(proximal["inh_out"].detach())[selected.bool()]
            if isinstance(proximal.get("inh_out"), torch.Tensor)
            else torch.zeros_like(selected_total)
        )
        rows.append(
            {
                **meta,
                "population": populations[0],
                "perturbed_stage": proximal_stage,
                "descendant_stage": distal_stage,
                "delta_g": float(delta_g),
                "n_descendants": int(descendant_mask.sum().item()),
                "n_sisters": int(sister_mask.sum().item()),
                "measured_delta_log_gain_mean": float(
                    measured_desc.mean().item()
                ),
                "exact_delta_log_gain_mean": float(exact_desc.mean().item()),
                "first_order_delta_log_gain_mean": float(
                    first_desc.mean().item()
                ),
                "relative_error_exact": _relative_error(
                    measured_desc,
                    exact_desc,
                ),
                "relative_error_first_order": _relative_error(
                    measured_desc,
                    first_desc,
                ),
                "correlation_exact": _correlation(measured_desc, exact_desc),
                "max_abs_sister_change": float(
                    measured[sister_mask].abs().max().item()
                ),
                "max_abs_inhibited_compartment_change": float(
                    self_change[selected.bool()].abs().max().item()
                ),
                "selected_g_total_mean": float(selected_total.mean().item()),
                "selected_input_resistance_mean": float(
                    selected_resistance.mean().item()
                ),
                "selected_g_inhibitory_mean": float(
                    selected_inhibitory.mean().item()
                ),
                "selected_inhibitory_fraction_mean": float(
                    (selected_inhibitory / selected_total).mean().item()
                ),
            }
        )
    return rows


def _aggregate(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.select_dtypes(include="number").columns
        if column not in group_cols and column not in {"seed", "batch_size"}
    ]
    return (
        frame.groupby(group_cols, dropna=False)[numeric]
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--delta-g",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0],
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DRAFT_ROOT / "figures" / "data" / "inhibitory_operating_point",
    )
    args = parser.parse_args()

    run_dirs = args.run_dir or _default_run_dirs()
    missing = [path for path in run_dirs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing run directories: {missing}")
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    operating_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir}")
        meta, helper, records = _collect_forward_state(
            run_dir,
            batch_size=args.batch_size,
            split=args.split,
            device=device,
        )
        operating_rows.extend(_operating_rows(meta, helper, records))
        if meta["dataset"] == "mnist":
            calibration_rows.extend(
                _calibration_rows(
                    meta,
                    helper,
                    records,
                    args.delta_g,
                )
            )

    operating = pd.DataFrame(operating_rows)
    calibration = pd.DataFrame(calibration_rows)
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    operating_path = prefix.with_name(prefix.name + "_runs.csv")
    operating_summary_path = prefix.with_name(prefix.name + "_summary.csv")
    calibration_path = prefix.with_name(prefix.name + "_calibration_runs.csv")
    calibration_summary_path = prefix.with_name(
        prefix.name + "_calibration_summary.csv"
    )
    operating.to_csv(operating_path, index=False)
    _aggregate(
        operating,
        ["dataset", "network_type", "split", "population", "stage", "stage_role"],
    ).to_csv(operating_summary_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    if not calibration.empty:
        _aggregate(
            calibration,
            ["dataset", "network_type", "split", "delta_g"],
        ).to_csv(calibration_summary_path, index=False)
    print(f"Wrote {operating_path}")
    print(f"Wrote {operating_summary_path}")
    print(f"Wrote {calibration_path}")
    if not calibration.empty:
        print(f"Wrote {calibration_summary_path}")


if __name__ == "__main__":
    main()

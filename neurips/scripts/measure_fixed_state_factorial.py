#!/usr/bin/env python3
"""Decompose shunting credit factors while holding the recorded forward state fixed.

For each checkpoint and batch, the script records one forward/backward pass and
then recomputes branch gradients after selectively changing:

- local input resistance in the eligibility term;
- input resistance in backward path transport;
- parent transfer derivatives in path transport; and
- the synaptic driving-force factor.

Presynaptic inputs, voltages, soma errors, couplings, activation derivatives
unless explicitly removed, masks, and all other local variables remain fixed.
The resulting hybrids are diagnostic counterfactuals, not physical forward
models or training conditions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from measure_layer_soma_factorial import (  # noqa: E402
    _activation_stats,
    _alignment_rows,
    _apply_local_grads_from_errors,
    _branch_level_summary,
    _capture_forward_backward,
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


FACTORIAL_SPECS: dict[str, dict[str, str]] = {
    "exact_full": {
        "error_condition": "exact_soma_path_transport",
        "input_resistance_mode": "actual",
        "driving_force_mode": "actual",
    },
    "eligibility_R_no_inhibition": {
        "error_condition": "exact_soma_path_transport",
        "input_resistance_mode": "no_inhibition",
        "driving_force_mode": "actual",
    },
    "eligibility_R_one": {
        "error_condition": "exact_soma_path_transport",
        "input_resistance_mode": "one",
        "driving_force_mode": "actual",
    },
    "transport_R_no_inhibition": {
        "error_condition": "exact_soma_path_transport_no_inhibition",
        "input_resistance_mode": "actual",
        "driving_force_mode": "actual",
    },
    "eligibility_and_transport_R_no_inhibition": {
        "error_condition": "exact_soma_path_transport_no_inhibition",
        "input_resistance_mode": "no_inhibition",
        "driving_force_mode": "actual",
    },
    "transport_no_parent_derivative": {
        "error_condition": "exact_soma_path_transport_no_parent_activation",
        "input_resistance_mode": "actual",
        "driving_force_mode": "actual",
    },
    "transport_no_R_or_parent_derivative": {
        "error_condition": (
            "exact_soma_path_transport_no_inhibition_no_parent_activation"
        ),
        "input_resistance_mode": "actual",
        "driving_force_mode": "actual",
    },
    "synaptic_voltage_proxy": {
        "error_condition": "exact_soma_path_transport",
        "input_resistance_mode": "actual",
        "driving_force_mode": "voltage_proxy",
    },
    "submitted_full": {
        "error_condition": "approx_direct_code_per_soma",
        "input_resistance_mode": "actual",
        "driving_force_mode": "actual",
    },
    "submitted_eligibility_R_no_inhibition": {
        "error_condition": "approx_direct_code_per_soma",
        "input_resistance_mode": "no_inhibition",
        "driving_force_mode": "actual",
    },
}


def analyze_run(
    run_dir: Path,
    *,
    batch_size: int,
    split: str,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _load_config_from_run(run_dir)
    if str(config.model.core.type) != "dendritic_shunting":
        return pd.DataFrame(), pd.DataFrame()
    x_batch, y_batch = _get_batch(config, split=split, batch_size=batch_size)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    _set_encoder_input_dim(config, x_batch)

    main_cfg = _get_value(config.training, "main", {})
    common_cfg = _to_plain_dict(_get_value(main_cfg, "common", {}))
    train_local_cfg = _to_plain_dict(
        _get_value(main_cfg, "learning_strategy_config", {})
    )
    loss_name = str(common_cfg.get("loss_function", "cat_nll"))
    helper = _make_helper(config, rule_variant="3f", loss_name=loss_name)
    model = _load_model(config, run_dir, device)
    records, exact_grads, v0_direct, delta_direct, loss_value = (
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
    activation_stats = _activation_stats(helper, records)

    meta = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "seed": int(
            _get_value(_get_value(config, "experiment", {}), "seed", -1)
        ),
        "dataset": str(config.data.dataset_name),
        "network_type": str(config.model.core.type),
        "trained_broadcast_mode": train_local_cfg.get("error_broadcast_mode"),
        "diagnostic_rule_variant": "3f",
        "split": split,
        "loss_name": loss_name,
        "loss_value": loss_value,
    }

    rows: list[dict[str, Any]] = []
    grads_by_variant: dict[str, dict[str, torch.Tensor]] = {}
    for variant, spec in FACTORIAL_SPECS.items():
        error_condition = spec["error_condition"]
        if error_condition not in conditions:
            continue
        grads = _apply_local_grads_from_errors(
            model,
            helper,
            records,
            conditions[error_condition],
            v0=v0_direct,
            input_resistance_mode=spec["input_resistance_mode"],
            driving_force_mode=spec["driving_force_mode"],
        )
        grads_by_variant[variant] = grads
        variant_meta = {
            **meta,
            "error_condition": error_condition,
            "input_resistance_mode": spec["input_resistance_mode"],
            "driving_force_mode": spec["driving_force_mode"],
        }
        rows.extend(
            _alignment_rows(
                grads,
                exact_grads,
                meta=variant_meta,
                condition=variant,
                activation_stats=activation_stats,
            )
        )

    full = grads_by_variant.get("exact_full")
    if full is None:
        raise RuntimeError(f"Exact full condition missing for {run_dir}")
    for variant, grads in grads_by_variant.items():
        if variant == "exact_full":
            continue
        spec = FACTORIAL_SPECS[variant]
        rows.extend(
            _alignment_rows(
                grads,
                full,
                meta={
                    **meta,
                    "error_condition": spec["error_condition"],
                    "input_resistance_mode": spec["input_resistance_mode"],
                    "driving_force_mode": spec["driving_force_mode"],
                },
                condition=f"{variant}_vs_reconstructed_exact",
                activation_stats=activation_stats,
            )
        )

    details = pd.DataFrame(rows)
    checkpoint, _ = _branch_level_summary(details)
    return details, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="auto")
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

    detail_frames: list[pd.DataFrame] = []
    checkpoint_frames: list[pd.DataFrame] = []
    for index, run_dir in enumerate(run_dirs, start=1):
        print(f"[{index}/{len(run_dirs)}] {run_dir}", flush=True)
        details, checkpoint = analyze_run(
            run_dir,
            batch_size=args.batch_size,
            split=args.split,
            device=device,
        )
        if not details.empty:
            detail_frames.append(details)
        if not checkpoint.empty:
            checkpoint_frames.append(checkpoint)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_all = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame()
    )
    checkpoint_all = (
        pd.concat(checkpoint_frames, ignore_index=True)
        if checkpoint_frames
        else pd.DataFrame()
    )
    details_all.to_csv(args.output_dir / "fixed_state_factorial_details.csv", index=False)
    checkpoint_all.to_csv(
        args.output_dir / "fixed_state_factorial_checkpoints.csv",
        index=False,
    )
    manifest = {
        "root": str(root),
        "n_runs_discovered": len(run_dirs),
        "n_shunting_runs": int(checkpoint_all["run_name"].nunique())
        if not checkpoint_all.empty
        else 0,
        "batch_size": args.batch_size,
        "split": args.split,
        "device": str(device),
        "variants": FACTORIAL_SPECS,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(
        f"Saved {len(details_all)} detail rows and "
        f"{len(checkpoint_all)} checkpoint rows to {args.output_dir}"
    )


if __name__ == "__main__":
    main()

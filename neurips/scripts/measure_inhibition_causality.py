#!/usr/bin/env python3
"""Post-hoc causal interventions on inhibitory conductance.

This checkpoint-only diagnostic asks whether the learned inhibitory conductance
field is doing useful causal work after training. It evaluates each model under:

- original: no intervention,
- zero_i: remove inhibitory conductance,
- shuffle_i: shuffle inhibitory conductance across samples,
- mean_clamp_i: replace each branch's inhibitory conductance by its batch mean,
- uniform_matched_i: replace inhibitory conductance by one matched scalar.

For each condition it reports accuracy on a small held-out subset plus
path-gain and broadcast-fidelity diagnostics on one batch.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

from dendritic_modeling.datasets import get_unified_datasets  # noqa: E402
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (  # noqa: E402
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (  # noqa: E402
    TopKLinear,
)
from dendritic_modeling.scripts.script_utils.setup_utils import (  # noqa: E402
    initialize_model,
)
from measure_theory_diagnostics import (  # noqa: E402
    _capture_decoder_input,
    _compute_loss,
    _discover_run_dirs,
    _error_rows,
    _get_batch,
    _get_value,
    _hook_branch_layers,
    _load_config_from_run,
    _locate_model_path,
    _make_helper,
    _to_plain_dict,
    _weighted_mean,
)


INTERVENTIONS = (
    "original",
    "zero_i",
    "shuffle_i",
    "mean_clamp_i",
    "uniform_matched_i",
)


def _dataset_for_split(config: Any, split: str):
    base_dir = config.data.base_dir or ""
    dataset_specific = {}
    if hasattr(config.data.dataset_params, config.data.dataset_name):
        dataset_specific = _to_plain_dict(
            getattr(config.data.dataset_params, config.data.dataset_name)
        )
    task_cfg = type(
        "TaskConfig",
        (),
        {
            "dataset": config.data.dataset_name,
            "data_path": (
                str(Path(base_dir) / config.data.dataset_name) if base_dir else None
            ),
            "train_valid_split": config.experiment.train_valid_split,
            "parameters": {
                **_to_plain_dict(config.data.processing),
                **dataset_specific,
            },
        },
    )()
    train_ds, valid_ds, test_ds = get_unified_datasets(task_cfg=task_cfg)
    if split == "train":
        return train_ds
    if split == "valid":
        return valid_ds
    return test_ds


def _load_model_and_config(run_dir: Path, device: torch.device):
    config = _load_config_from_run(run_dir)
    if hasattr(config.model.core, "implementation"):
        config.model.core.implementation.compile_forward = False
    batch = _get_batch(config, split="train", batch_size=1)
    input_dim = int(batch[0][0].numel())
    encoder_params = getattr(config.model.encoder, "params", None)
    if encoder_params is None:
        config.model.encoder.params = {"input_dim": input_dim}
    elif isinstance(encoder_params, dict):
        encoder_params["input_dim"] = input_dim
    else:
        encoder_params.input_dim = input_dim

    model, _ = initialize_model(config.model)
    state = torch.load(_locate_model_path(run_dir), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    return config, model


def _make_inhibition_hooks(model: nn.Module, intervention: str, seed: int) -> list[Any]:
    if intervention == "original":
        return []

    handles: list[Any] = []
    counter = itertools.count()

    def _hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            return output
        if intervention == "zero_i":
            return torch.zeros_like(output)
        if intervention == "mean_clamp_i":
            return output.mean(dim=0, keepdim=True).expand_as(output)
        if intervention == "uniform_matched_i":
            return output.mean().expand_as(output)
        if intervention == "shuffle_i":
            call_idx = next(counter)
            gen = torch.Generator(device=output.device) if output.is_cuda else torch.Generator()
            gen.manual_seed(seed + call_idx)
            perm = torch.randperm(output.shape[0], device=output.device, generator=gen)
            return output[perm]
        raise ValueError(f"Unknown intervention: {intervention}")

    for module in model.modules():
        if (
            isinstance(module, DendriticBranchLayer)
            and getattr(module, "branch_inhibition", None) is not None
        ):
            handles.append(module.branch_inhibition.register_forward_hook(_hook))
    return handles


def _accuracy(
    model: nn.Module,
    dataset,
    *,
    device: torch.device,
    batch_size: int,
    max_samples: int,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, *_rest in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            if total >= max_samples:
                break
    return correct / max(total, 1)


def _diagnostics(
    config: Any,
    model: nn.Module,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    x_batch, y_batch = _get_batch(config, split=split, batch_size=batch_size)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    helper = _make_helper()
    decoder_cache: dict[str, Any] = {}
    dec_handle = _capture_decoder_input(model, decoder_cache)
    layer_records, layer_handles = _hook_branch_layers(model)
    topk_modules = [module for module in model.modules() if isinstance(module, TopKLinear)]
    for module in topk_modules:
        module.cache_mask = True

    try:
        y_hat = model(x_batch)
    finally:
        if dec_handle is not None:
            dec_handle.remove()
        for handle in layer_handles:
            handle.remove()
        for module in topk_modules:
            module.cache_mask = False

    common_cfg = _get_value(_get_value(config.training, "main"), "common", {})
    loss_name = _get_value(common_cfg, "loss_function", "cat_nll")
    loss = _compute_loss(loss_name, y_hat, y_batch)

    helper.loss_function = type("Loss", (), {"_loss_name": str(loss_name)})()
    helper._decoder_cache = decoder_cache
    delta_out = helper._compute_soma_error(y_hat.detach(), y_batch.detach())
    _, delta_local = helper._resolve_local_soma_signals(
        model=model,
        y_hat=y_hat,
        delta_out=delta_out.detach(),
    )

    model.zero_grad(set_to_none=True)
    loss.backward()

    error_rows, path_rows = _error_rows(helper, layer_records, delta_local.detach())
    error_df = pd.DataFrame(error_rows)
    path_df = pd.DataFrame(path_rows)

    out: dict[str, float] = {}
    if not error_df.empty:
        for mode in ["scalar", "per_soma", "path_factor_scalar", "path_transport"]:
            sub = error_df[error_df["broadcast_mode"] == mode]
            out[f"{mode}_weighted_cosine"] = _weighted_mean(sub, "cosine") if not sub.empty else float("nan")
            out[f"{mode}_scale_mismatch"] = _weighted_mean(sub, "scale_mismatch") if not sub.empty else float("nan")
    if not path_df.empty:
        out["path_gain_cv_mean"] = _weighted_mean(path_df, "path_gain_cv")
        out["path_gain_no_i_cv_mean"] = _weighted_mean(path_df, "path_gain_no_inhibition_cv")
        out["inhibitory_conductance_mean"] = _weighted_mean(path_df, "inhibitory_conductance_mean")
        out["inhibitory_conductance_fraction"] = _weighted_mean(path_df, "inhibitory_conductance_fraction")
    return out


def analyze_run(
    run_dir: Path,
    *,
    split: str,
    eval_batch_size: int,
    diag_batch_size: int,
    max_samples: int,
    device: torch.device,
    seed: int,
) -> list[dict[str, Any]]:
    config, model = _load_model_and_config(run_dir, device)
    dataset = _dataset_for_split(config, split)
    rows: list[dict[str, Any]] = []
    connectivity = getattr(config.model.core, "connectivity", None)
    ie_values = _get_value(connectivity, "ie_synapses_per_branch_per_layer", [])
    ie_value = ie_values[0] if isinstance(ie_values, list) and ie_values else None
    meta = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "dataset": config.data.dataset_name,
        "network_type": config.model.core.type,
        "ie_value": ie_value,
        "strategy": _get_value(_get_value(config.training, "main"), "strategy"),
        "rule_variant": _get_value(
            _get_value(_get_value(config.training, "main"), "learning_strategy_config"),
            "rule_variant",
        ),
        "error_broadcast_mode": _get_value(
            _get_value(_get_value(config.training, "main"), "learning_strategy_config"),
            "error_broadcast_mode",
        ),
    }

    for intervention in INTERVENTIONS:
        handles = _make_inhibition_hooks(model, intervention, seed=seed)
        try:
            acc = _accuracy(
                model,
                dataset,
                device=device,
                batch_size=eval_batch_size,
                max_samples=max_samples,
            )
            diag = _diagnostics(
                config,
                model,
                split=split,
                batch_size=diag_batch_size,
                device=device,
            )
        finally:
            for handle in handles:
                handle.remove()
        rows.append({**meta, "intervention": intervention, "accuracy": acc, **diag})
    return rows


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    metric_cols = [
        col
        for col in rows.columns
        if col
        not in {
            "run_dir",
        "run_name",
        "dataset",
        "network_type",
        "ie_value",
        "strategy",
            "rule_variant",
            "error_broadcast_mode",
            "intervention",
            "error",
        }
        and pd.api.types.is_numeric_dtype(rows[col])
    ]
    group_cols = [
        "dataset",
        "network_type",
        "ie_value",
        "strategy",
        "rule_variant",
        "error_broadcast_mode",
        "intervention",
    ]
    summary = rows.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--run-dirs", type=Path, nargs="+")
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--diag-batch-size", type=int, default=256)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    supplied = sum(bool(x) for x in (args.run_dir, args.run_dirs, args.sweep_dir))
    if supplied != 1:
        raise ValueError("Specify exactly one of --run-dir, --run-dirs, or --sweep-dir.")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    if args.run_dirs:
        run_dirs = list(args.run_dirs)
    else:
        root = args.run_dir or args.sweep_dir
        run_dirs = _discover_run_dirs(root)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]

    all_rows: list[dict[str, Any]] = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"[{idx}/{len(run_dirs)}] {run_dir}")
        try:
            all_rows.extend(
                analyze_run(
                    run_dir,
                    split=args.split,
                    eval_batch_size=args.eval_batch_size,
                    diag_batch_size=args.diag_batch_size,
                    max_samples=args.max_samples,
                    device=device,
                    seed=args.seed,
                )
            )
        except Exception as exc:
            print(f"WARNING: failed {run_dir}: {exc}")
            all_rows.append({"run_dir": str(run_dir), "error": str(exc)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(all_rows)
    rows.to_csv(args.output_dir / "inhibition_causality_runs.csv", index=False)
    clean = rows[rows.get("error").isna()] if "error" in rows else rows
    _summarize(clean).to_csv(args.output_dir / "inhibition_causality_summary.csv", index=False)
    print(f"Saved diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()

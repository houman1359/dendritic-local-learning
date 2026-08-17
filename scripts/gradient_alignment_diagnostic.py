#!/usr/bin/env python
"""Run one-batch soma-error mapping diagnostic for local learning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from dendritic_modeling.config import load_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _compute_loss(loss_name: str, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    name = (loss_name or "").lower()
    if name in {"cat_nll", "categorical_negative_log_likelihood"}:
        return (-Categorical(logits=y_hat).log_prob(y)).mean()
    if name in {"ce", "cross_entropy"}:
        return F.cross_entropy(y_hat, y)
    if name in {"mse", "mean_squared_error"}:
        return F.mse_loss(y_hat, y)
    if name in {"bce", "binary_cross_entropy"}:
        return F.binary_cross_entropy(y_hat, y)
    # Local learning defaults to mse-like fallback for unknown; keep script permissive.
    return F.mse_loss(y_hat, y)


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to experiment config YAML")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where diagnostic JSON/summary will be written",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            "data_path": os.path.join(base_dir, config.data.dataset_name)
            if base_dir
            else None,
            "train_valid_split": config.experiment.train_valid_split,
            "parameters": {
                **_to_plain_dict(config.data.processing),
                **dataset_specific,
            },
        },
    )()

    train_ds, _, _ = get_unified_datasets(task_cfg=task_cfg)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    x_batch, y_batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    model, _ = initialize_model(config.model)
    model = model.to(device)
    model.train()

    decoder_linears = [
        module
        for module in model.decoder_network.modules()
        if isinstance(module, nn.Linear)
    ]

    mapping_possible = len(decoder_linears) == 1
    hook_context: dict[str, torch.Tensor] = {}
    handle = None
    if mapping_possible:
        decoder_linear = decoder_linears[0]

        def _capture_decoder_input(mod: nn.Module, inputs: tuple, outputs: torch.Tensor):
            hook_context["h_in"] = inputs[0]
            hook_context["h_in"].retain_grad()

        handle = decoder_linear.register_forward_hook(_capture_decoder_input)
    else:
        decoder_linear = None

    y_hat = model(x_batch)
    if handle is not None:
        handle.remove()

    main_cfg = _get_value(config.training, "main")
    common_cfg = _get_value(main_cfg, "common", {})
    loss_name = _get_value(common_cfg, "loss_function", "cat_nll")
    loss = _compute_loss(loss_name, y_hat, y_batch)

    helper = LocalCreditAssignment.__new__(LocalCreditAssignment)
    helper.local_cfg = type("LocalCfg", (), {"error_mode": "auto"})()
    helper.loss_function = type("Loss", (), {"_loss_name": str(loss_name)})()
    delta_out = helper._compute_soma_error(y_hat.detach(), y_batch.detach())

    if mapping_possible and decoder_linear is not None and "h_in" in hook_context:
        delta_local = delta_out @ decoder_linear.weight.detach()
        v0_local = hook_context["h_in"].detach()
    else:
        delta_local = delta_out
        v0_local = y_hat.detach()

    model.zero_grad(set_to_none=True)
    loss.backward()

    if "h_in" in hook_context and hook_context["h_in"].grad is not None:
        grad_ref = hook_context["h_in"].grad.detach()
        grad_ref_source = "decoder_input_grad"
    else:
        grad_ref = delta_out.detach()
        grad_ref_source = "output_error_fallback"

    flat_local = delta_local.reshape(-1)
    flat_ref = grad_ref.reshape(-1)
    cosine = float(F.cosine_similarity(flat_local, flat_ref, dim=0).item())
    rel_l2 = float((flat_local - flat_ref).norm().item() / (flat_ref.norm().item() + 1e-12))
    max_abs = float((flat_local - flat_ref).abs().max().item())
    scale_num = torch.dot(flat_local, flat_ref)
    scale_den = torch.dot(flat_local, flat_local) + 1e-12
    best_scale = float((scale_num / scale_den).item())
    scaled_rel_l2 = float(
        ((best_scale * flat_local - flat_ref).norm().item())
        / (flat_ref.norm().item() + 1e-12)
    )

    summary = {
        "config": str(args.config),
        "device": str(device),
        "mapping_possible": bool(mapping_possible),
        "decoder_linear_layers": len(decoder_linears),
        "loss_name": str(loss_name),
        "loss_value": float(loss.detach().item()),
        "reference": grad_ref_source,
        "delta_local_shape": list(delta_local.shape),
        "grad_ref_shape": list(grad_ref.shape),
        "v0_local_shape": list(v0_local.shape),
        "cosine_similarity": cosine,
        "relative_l2_error": rel_l2,
        "best_scale_to_reference": best_scale,
        "scaled_relative_l2_error": scaled_rel_l2,
        "max_abs_error": max_abs,
    }

    out_json = output_dir / "gradient_alignment_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved diagnostic summary to {out_json}")


if __name__ == "__main__":
    main()

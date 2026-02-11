#!/usr/bin/env python
"""Probe local-CA gradient magnitudes per optimizer parameter group on one batch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from dendritic_modeling.config import load_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.scripts.script_utils.config_utils import (
    normalize_param_groups_config,
)
from dendritic_modeling.scripts.script_utils.config_utils import prepare_trainer_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training import CustomWeightDecayOptimizer, get_trainer
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _build_task_cfg(config) -> Any:
    base_dir = config.data.base_dir or ""
    dataset_specific = {}
    if hasattr(config.data.dataset_params, config.data.dataset_name):
        dataset_specific = _to_plain_dict(
            getattr(config.data.dataset_params, config.data.dataset_name)
        )

    return type(
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


def _apply_decoder_backprop_if_needed(
    trainer: LocalCreditAssignment,
    model: torch.nn.Module,
    loss: torch.Tensor,
) -> None:
    if not hasattr(model, "decoder_network"):
        return
    if trainer.local_cfg.decoder_update_mode != "backprop":
        return

    dec_params = [p for p in model.decoder_network.parameters() if p.requires_grad]
    if not dec_params:
        return

    grads = torch.autograd.grad(loss, dec_params, retain_graph=True, allow_unused=True)
    for p, g in zip(dec_params, grads):
        if g is None:
            continue
        if p.grad is None:
            p.grad = g.detach().clone()
        else:
            p.grad = p.grad + g.detach()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Path to YAML config")
    parser.add_argument("--output-json", required=True, help="Path to save probe JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Optional override for lr/topk_lr/decoder_lr",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = load_config(args.config)

    main_cfg = config.training.main
    if isinstance(main_cfg, dict):
        common_cfg = main_cfg.get("common", {})
    else:
        common_cfg = main_cfg.common

    if args.lr is not None:
        if isinstance(common_cfg, dict):
            pg = common_cfg.setdefault("param_groups", {})
            pg["lr"] = args.lr
            pg["topk_lr"] = args.lr
            pg["decoder_lr"] = args.lr
        else:
            common_cfg.param_groups.lr = args.lr
            common_cfg.param_groups.topk_lr = args.lr
            common_cfg.param_groups.decoder_lr = args.lr

    task_cfg = _build_task_cfg(config)
    train_ds, _, _ = get_unified_datasets(task_cfg=task_cfg)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    x_batch, y_batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    model, _ = initialize_model(config.model)
    model = model.to(device)
    model.train()

    if isinstance(common_cfg, dict):
        param_groups_cfg = common_cfg.get("param_groups", {})
        weight_decay_rate = common_cfg.get("weight_decay_rate", 0.0)
        weight_boosting = common_cfg.get("weight_boosting", False)
    else:
        param_groups_cfg = common_cfg.param_groups
        weight_decay_rate = common_cfg.weight_decay_rate
        weight_boosting = getattr(common_cfg, "weight_boosting", False)

    param_groups_cfg = normalize_param_groups_config(param_groups_cfg)

    raw_optimizer = torch.optim.Adam(model.get_param_groups(param_groups_cfg))
    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=raw_optimizer,
        weight_decay=weight_decay_rate,
        weight_boosting=weight_boosting,
    )

    trainer_cfg = prepare_trainer_config(
        main_train_config=config.training.main,
        experiment_config=config.experiment,
        save_path=str(Path(args.output_json).resolve().parent / "probe_tmp"),
    )

    trainer = get_trainer(
        strategy=config.training.main.strategy,
        optimizer=optimizer,
        trainer_config_dict=trainer_cfg,
        analysis_manager=None,
    )
    if not isinstance(trainer, LocalCreditAssignment):
        raise TypeError("Expected LocalCreditAssignment trainer")

    layer_records, handles = trainer._attach_local_recorders(model)
    try:
        y_hat = model(x_batch)
    finally:
        for handle in handles:
            handle.remove()

    loss = trainer._compute_logging_loss(model, x_batch, y_batch)
    delta_out = trainer._compute_soma_error(y_hat.detach(), y_batch.detach())
    v0_local, delta_local = trainer._resolve_local_soma_signals(
        model=model,
        y_hat=y_hat.detach(),
        delta_out=delta_out.detach(),
    )

    optimizer.zero_grad()
    trainer._apply_local_rule_gradients(
        model,
        layer_records,
        v0=v0_local,
        delta=delta_local,
        y_target=None,
        update_reactivation=trainer.local_cfg.update_reactivation,
    )
    _apply_decoder_backprop_if_needed(trainer, model, loss)

    if trainer.local_cfg.clip_grad_value and trainer.local_cfg.clip_grad_value > 0:
        torch.nn.utils.clip_grad_value_(model.parameters(), trainer.local_cfg.clip_grad_value)

    id_to_name = {id(param): name for name, param in model.named_parameters()}
    group_metrics: list[dict[str, Any]] = []
    for idx, group in enumerate(optimizer.param_groups):
        params = list(group.get("params", []))
        nonnull = 0
        nonzero = 0
        total_l2 = 0.0
        max_abs = 0.0
        sample_names: list[str] = []
        for param in params:
            if param.grad is None:
                continue
            nonnull += 1
            grad = param.grad.detach()
            norm = float(grad.norm().item())
            total_l2 += norm
            max_abs = max(max_abs, float(grad.abs().max().item()))
            if norm > 0.0:
                nonzero += 1
            if len(sample_names) < 6:
                sample_names.append(id_to_name.get(id(param), "<unnamed>"))

        group_metrics.append(
            {
                "group_index": idx,
                "lr": float(group.get("lr", 0.0)),
                "num_params_in_group": len(params),
                "num_params_with_grad": nonnull,
                "num_params_with_nonzero_grad": nonzero,
                "sum_param_grad_l2": total_l2,
                "max_abs_grad": max_abs,
                "sample_param_names": sample_names,
            }
        )

    payload = {
        "config": args.config,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr_override": args.lr,
        "param_groups_config_type": type(param_groups_cfg).__name__,
        "optimizer_group_lrs": [float(g.get("lr", 0.0)) for g in optimizer.param_groups],
        "loss": float(loss.detach().item()),
        "local_delta_l2": float(delta_local.norm().item()),
        "group_metrics": group_metrics,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

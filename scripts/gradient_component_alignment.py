#!/usr/bin/env python3
"""Compare LocalCA and backprop gradients by component on a single batch.

This diagnostic computes parameter-level gradient alignment between:
- local learning gradients produced by LocalCreditAssignment rules, and
- exact backprop gradients from autograd on the same batch and initial weights.

Outputs:
- per_parameter_alignment.csv
- component_alignment_summary.csv
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from dendritic_modeling.config import load_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "asdict") and callable(obj.asdict):
        return obj.asdict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_core_type(config_obj: Any, core_type: str) -> None:
    model = _get_value(config_obj, "model")
    if model is None:
        return
    core = _get_value(model, "core")
    if core is None:
        return
    if isinstance(core, dict):
        core["type"] = core_type
    else:
        setattr(core, "type", core_type)


def _split_csv(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint_state(path: str) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state_dict" in obj:
        state = obj["model_state_dict"]
    else:
        state = obj
    if not isinstance(state, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {path}")
    return state


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
    return F.mse_loss(y_hat, y)


def _build_task_config(config: Any) -> Any:
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
    return task_cfg


def _component_from_name(name: str) -> str:
    if "branch_excitation.pre_w" in name:
        return "excitatory_synapse"
    if "branch_inhibition.pre_w" in name:
        return "inhibitory_synapse"
    if "branches_to_output.log_weight" in name:
        return "dendritic_conductance"
    if ".reactivation." in name:
        return "reactivation"
    if name.startswith("decoder_network."):
        return "decoder"
    return "other"


def _layer_id_from_name(name: str) -> str:
    match = re.search(r"layers\.(\d+).*branch_layers\.(\d+)", name)
    if not match:
        return "n/a"
    return f"L{match.group(1)}B{match.group(2)}"


def _new_local_helper(
    local_cfg_dict: dict[str, Any],
    *,
    loss_name: str,
    epoch_counter: int = 1,
) -> LocalCreditAssignment:
    helper = LocalCreditAssignment.__new__(LocalCreditAssignment)
    helper.local_cfg = helper._build_local_rule_config(local_cfg_dict)
    helper._layer_stats = {}
    helper._decoder_cache = {}
    helper._warned_decoder_soma_fallback = False
    helper.epoch_counter = epoch_counter
    helper.loss_function = type("Loss", (), {"_loss_name": str(loss_name)})()
    return helper


def _compute_local_grads(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    *,
    helper: LocalCreditAssignment,
) -> dict[str, torch.Tensor]:
    model.train()
    for p in model.parameters():
        p.grad = None

    layer_records, handles = helper._attach_local_recorders(model)
    topk_modules = [m for m in model.modules() if isinstance(m, TopKLinear)]
    for module in topk_modules:
        module.cache_mask = True

    try:
        y_hat = model(x_batch)
    finally:
        for handle in handles:
            handle.remove()
        for module in topk_modules:
            module.cache_mask = False
            module._last_forward_weight_mask = None

    delta_out = helper._compute_soma_error(y_hat.detach(), y_batch)
    v0_local, delta_local = helper._resolve_local_soma_signals(
        model=model,
        y_hat=y_hat.detach(),
        delta_out=delta_out.detach(),
    )

    y_target = None
    if helper.local_cfg.hsic.enabled:
        if helper.local_cfg.hsic.target_source == "labels":
            if y_batch.dim() == 1 or (y_batch.dim() == 2 and y_batch.size(1) == 1):
                num_classes = y_hat.size(-1)
                y_target = torch.nn.functional.one_hot(
                    y_batch.view(-1), num_classes=num_classes
                ).to(y_hat.dtype)
            else:
                y_target = y_batch.to(y_hat.dtype)
        else:
            y_target = y_hat.detach()

    helper._apply_local_rule_gradients(
        model=model,
        layer_records=layer_records,
        v0=v0_local,
        delta=delta_local,
        y_target=y_target,
        update_reactivation=bool(helper.local_cfg.update_reactivation),
    )

    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def _compute_backprop_grads(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    *,
    loss_name: str,
) -> dict[str, torch.Tensor]:
    model.train()
    for p in model.parameters():
        p.grad = None
    y_hat = model(x_batch)
    loss = _compute_loss(loss_name, y_hat, y_batch)
    loss.backward()

    grads: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def _alignment_rows(
    local_grads: dict[str, torch.Tensor],
    bp_grads: dict[str, torch.Tensor],
    *,
    condition: dict[str, Any],
    include_decoder: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    common_names = sorted(set(local_grads.keys()) & set(bp_grads.keys()))

    for name in common_names:
        component = _component_from_name(name)
        if component == "decoder" and not include_decoder:
            continue

        g_local = local_grads[name].reshape(-1).float()
        g_bp = bp_grads[name].reshape(-1).float()

        local_norm = float(g_local.norm().item())
        bp_norm = float(g_bp.norm().item())
        denom = local_norm * bp_norm + 1e-12
        cosine = float(torch.dot(g_local, g_bp).item() / denom) if denom > 0 else np.nan
        norm_ratio = float(local_norm / (bp_norm + 1e-12))
        l2_rel = float((g_local - g_bp).norm().item() / (bp_norm + 1e-12))

        row = {
            **condition,
            "parameter_name": name,
            "layer_id": _layer_id_from_name(name),
            "component": component,
            "numel": int(g_local.numel()),
            "cosine_similarity": cosine,
            "local_grad_norm": local_norm,
            "backprop_grad_norm": bp_norm,
            "norm_ratio_local_over_backprop": norm_ratio,
            "relative_l2_error": l2_rel,
        }
        rows.append(row)

    return rows


def _component_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    def weighted_mean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
        values = frame[value_col].to_numpy(dtype=float)
        weights = frame[weight_col].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            return float("nan")
        return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))

    group_cols = [
        "core_type",
        "rule_variant",
        "error_broadcast_mode",
        "component",
    ]

    rows: list[dict[str, Any]] = []
    for keys, frame in df.groupby(group_cols, dropna=False):
        rows.append(
            {
                "core_type": keys[0],
                "rule_variant": keys[1],
                "error_broadcast_mode": keys[2],
                "component": keys[3],
                "n_parameters": int(len(frame)),
                "total_numel": int(frame["numel"].sum()),
                "mean_cosine": float(frame["cosine_similarity"].mean()),
                "weighted_mean_cosine": weighted_mean(
                    frame, "cosine_similarity", "numel"
                ),
                "mean_norm_ratio": float(
                    frame["norm_ratio_local_over_backprop"].mean()
                ),
                "mean_relative_l2_error": float(frame["relative_l2_error"].mean()),
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["core_type", "rule_variant", "error_broadcast_mode", "component"]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment config YAML")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for CSV and JSON summaries",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rule-variants",
        type=str,
        default="3f,4f,5f",
        help="Comma-separated LocalCA rule variants",
    )
    parser.add_argument(
        "--error-broadcast-modes",
        type=str,
        default="scalar,per_soma,local_mismatch",
        help="Comma-separated broadcast modes",
    )
    parser.add_argument(
        "--core-types",
        type=str,
        default="",
        help="Optional comma-separated core types (default: use config core type only)",
    )
    parser.add_argument(
        "--include-decoder",
        action="store_true",
        help="Include decoder parameters in alignment tables",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Optional path to model checkpoint/state_dict used as initial weights",
    )
    args = parser.parse_args()

    _set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    model_cfg = _get_value(config, "model")
    core_cfg = _get_value(model_cfg, "core", {})
    base_core_type = _get_value(core_cfg, "type")
    core_types = _split_csv(args.core_types) or [str(base_core_type)]
    rule_variants = _split_csv(args.rule_variants)
    broadcast_modes = _split_csv(args.error_broadcast_modes)

    task_cfg = _build_task_config(config)
    train_ds, _, _ = get_unified_datasets(task_cfg=task_cfg)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    x_batch, y_batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    training_cfg = _get_value(config, "training")
    main_training_cfg = _get_value(training_cfg, "main", {})
    common_cfg = _to_plain_dict(_get_value(main_training_cfg, "common", {}))
    loss_name = str(common_cfg.get("loss_function", "cat_nll"))
    local_cfg_base = _to_plain_dict(
        _get_value(main_training_cfg, "learning_strategy_config", {})
    )

    all_rows: list[dict[str, Any]] = []

    for core_type in core_types:
        cfg_core = copy.deepcopy(config)
        _set_core_type(cfg_core, core_type)
        template_model, _ = initialize_model(_get_value(cfg_core, "model"))
        template_model = template_model.to(device)
        if args.checkpoint:
            state = _load_checkpoint_state(args.checkpoint)
            template_model.load_state_dict(state, strict=True)
        template_state = copy.deepcopy(template_model.state_dict())

        for rule_variant in rule_variants:
            for broadcast_mode in broadcast_modes:
                print(
                    f"[align] core={core_type} rule={rule_variant} broadcast={broadcast_mode}",
                    flush=True,
                )
                condition = {
                    "core_type": core_type,
                    "rule_variant": rule_variant,
                    "error_broadcast_mode": broadcast_mode,
                }

                model_local, _ = initialize_model(_get_value(cfg_core, "model"))
                model_local = model_local.to(device)
                model_local.load_state_dict(template_state)

                model_bp, _ = initialize_model(_get_value(cfg_core, "model"))
                model_bp = model_bp.to(device)
                model_bp.load_state_dict(template_state)

                local_cfg = copy.deepcopy(local_cfg_base)
                local_cfg["rule_variant"] = rule_variant
                local_cfg["error_broadcast_mode"] = broadcast_mode
                local_cfg["decoder_update_mode"] = "none"
                # Keep alignment focused on local dendritic rules.
                hsic_cfg = _to_plain_dict(local_cfg.get("hsic"))
                hsic_cfg["enabled"] = False
                hsic_cfg["weight"] = 0.0
                hsic_cfg["self_weight"] = 0.0
                hsic_cfg["target_weight"] = 0.0
                local_cfg["hsic"] = hsic_cfg

                helper = _new_local_helper(local_cfg, loss_name=loss_name)
                local_grads = _compute_local_grads(
                    model_local,
                    x_batch,
                    y_batch,
                    helper=helper,
                )
                print(
                    f"[align] local grads computed ({len(local_grads)} tensors)",
                    flush=True,
                )
                bp_grads = _compute_backprop_grads(
                    model_bp,
                    x_batch,
                    y_batch,
                    loss_name=loss_name,
                )
                print(
                    f"[align] backprop grads computed ({len(bp_grads)} tensors)",
                    flush=True,
                )

                all_rows.extend(
                    _alignment_rows(
                        local_grads,
                        bp_grads,
                        condition=condition,
                        include_decoder=bool(args.include_decoder),
                    )
                )

    per_param = pd.DataFrame(all_rows)
    summary = _component_summary(per_param)

    per_param_path = output_dir / "per_parameter_alignment.csv"
    summary_path = output_dir / "component_alignment_summary.csv"
    meta_path = output_dir / "alignment_meta.json"

    per_param.to_csv(per_param_path, index=False)
    summary.to_csv(summary_path, index=False)

    meta = {
        "config": str(args.config),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "device": str(device),
        "core_types": core_types,
        "rule_variants": rule_variants,
        "error_broadcast_modes": broadcast_modes,
        "include_decoder": bool(args.include_decoder),
        "n_per_parameter_rows": int(len(per_param)),
        "n_summary_rows": int(len(summary)),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved per-parameter alignment: {per_param_path}")
    print(f"Saved component summary: {summary_path}")
    print(f"Saved metadata: {meta_path}")

    if not summary.empty:
        print("\nTop component alignments (weighted cosine):")
        top = summary.sort_values("weighted_mean_cosine", ascending=False).head(20)
        print(top.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

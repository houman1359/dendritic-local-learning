#!/usr/bin/env python3
"""Measure exact factorization and compartment-error fidelity diagnostics.

This script operates on existing run directories or sweep result directories.
For each run it:
1. loads the trained checkpoint,
2. measures exact compartment errors dL/dV_n on one batch,
3. compares broadcast approximations to those exact errors,
4. reconstructs raw parameter gradients from the theorem and compares them to
   autograd gradients.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.distributions import Categorical
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dendritic_modeling.config import load_config  # noqa: E402
from dendritic_modeling.config.config import Config  # noqa: E402
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
from dendritic_modeling.training.strategies.local_learning import (  # noqa: E402
    LocalCreditAssignment,
)


def _to_plain_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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


def _locate_model_path(run_dir: Path) -> Path:
    candidates = [
        run_dir / "main_network" / "local_learning_best_model.pt",
        run_dir / "final_model.pt",
        run_dir / "main_network" / "best_model.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found under {run_dir}")


def _resolve_dataclass_type(type_hint):
    if is_dataclass(type_hint):
        return type_hint
    origin = get_origin(type_hint)
    if origin is None:
        return None
    for arg in get_args(type_hint):
        resolved = _resolve_dataclass_type(arg)
        if resolved is not None:
            return resolved
    return None


def _sanitize_with_schema(data: Any, schema_type: Any) -> Any:
    dc_type = _resolve_dataclass_type(schema_type)
    if dc_type is None or not isinstance(data, dict):
        return data

    sanitized: dict[str, Any] = {}
    for field in fields(dc_type):
        if field.name not in data:
            continue
        value = data[field.name]
        nested_dc = _resolve_dataclass_type(field.type)
        origin = get_origin(field.type)
        if nested_dc is not None and isinstance(value, dict):
            sanitized[field.name] = _sanitize_with_schema(value, nested_dc)
        elif origin is list and isinstance(value, list):
            args = get_args(field.type)
            if args:
                item_dc = _resolve_dataclass_type(args[0])
                if item_dc is not None:
                    sanitized[field.name] = [
                        (
                            _sanitize_with_schema(item, item_dc)
                            if isinstance(item, dict)
                            else item
                        )
                        for item in value
                    ]
                    continue
            sanitized[field.name] = value
        else:
            sanitized[field.name] = value
    return sanitized


def _load_config_from_run(run_dir: Path):
    cfg_json = run_dir / "config.json"
    cfg_yaml = run_dir / "config.yaml"
    if cfg_yaml.exists():
        return load_config(str(cfg_yaml))
    if not cfg_json.exists():
        raise FileNotFoundError(f"Missing config.json under {run_dir}")
    with cfg_json.open("r", encoding="utf-8") as f:
        cfg_dict = _sanitize_with_schema(json.load(f), Config)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(cfg_dict, tmp, default_flow_style=False)
        tmp_path = tmp.name
    try:
        load_config.cache_clear()
        return load_config(tmp_path)
    finally:
        os.unlink(tmp_path)


def _get_batch(config, *, split: str, batch_size: int):
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
                os.path.join(base_dir, config.data.dataset_name) if base_dir else None
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
        dataset = train_ds
    elif split == "valid":
        dataset = valid_ds
    else:
        dataset = test_ds
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def _make_helper() -> LocalCreditAssignment:
    helper = LocalCreditAssignment.__new__(LocalCreditAssignment)
    helper.local_cfg = helper._build_local_rule_config(
        {
            "error_mode": "auto",
            "error_broadcast_mode": "path_transport",
            "morphology_aware": {
                "use_path_propagation": True,
                "path_factor_mode": "per_branch",
            },
        }
    )
    helper._layer_stats = {}
    helper._decoder_cache = {}
    helper._broadcast_cache = {}
    helper._warned_decoder_soma_fallback = False
    return helper


def _hook_branch_layers(model: nn.Module) -> tuple[list[dict[str, Any]], list[Any]]:
    records: list[dict[str, Any]] = []
    handles: list[Any] = []

    for name, module in model.named_modules():
        if not isinstance(module, DendriticBranchLayer):
            continue
        rec: dict[str, Any] = {"layer_name": name, "layer": module}

        def _react_hook(mod, inputs, outputs, rec=rec):
            pre_v = inputs[0]
            pre_v.retain_grad()
            rec["v_n"] = pre_v
            rec["v_out"] = outputs

        handles.append(module.reactivation.register_forward_hook(_react_hook))

        if getattr(module, "branch_excitation", None) is not None:
            exc_layer = module.branch_excitation

            def _exc_hook(mod, inputs, outputs, rec=rec, layer=exc_layer):
                rec["x_exc"] = inputs[0]
                rec["exc_out"] = outputs
                rec["exc_module"] = layer
                rec["exc_mask"] = getattr(layer, "_last_forward_weight_mask", None)

            handles.append(exc_layer.register_forward_hook(_exc_hook))

        if getattr(module, "branch_inhibition", None) is not None:
            inh_layer = module.branch_inhibition

            def _inh_hook(mod, inputs, outputs, rec=rec, layer=inh_layer):
                rec["x_inh"] = inputs[0]
                rec["inh_out"] = outputs
                rec["inh_module"] = layer
                rec["inh_mask"] = getattr(layer, "_last_forward_weight_mask", None)

            handles.append(inh_layer.register_forward_hook(_inh_hook))

        if getattr(module, "branches_to_output", None) is not None:
            blk_layer = module.branches_to_output

            def _blk_hook(mod, inputs, outputs, rec=rec, layer=blk_layer):
                rec["x_blk_raw"] = inputs[0]
                rec["blk_out"] = outputs
                rec["blk_module"] = layer

            handles.append(blk_layer.register_forward_hook(_blk_hook))

        records.append(rec)

    return records, handles


def _capture_decoder_input(model: nn.Module, cache: dict[str, Any]) -> Any | None:
    linears = [
        module
        for module in model.decoder_network.modules()
        if isinstance(module, nn.Linear)
    ]
    if len(linears) != 1:
        return None
    linear = linears[0]

    def _hook(mod, inputs, outputs):
        cache["module"] = mod
        cache["input"] = inputs[0]
        cache["input"].retain_grad()

    return linear.register_forward_hook(_hook)


def _vec_metrics(approx: torch.Tensor, exact: torch.Tensor) -> dict[str, float]:
    approx = approx.detach()
    exact = exact.detach()
    flat_approx = approx.reshape(-1)
    flat_exact = exact.reshape(-1)
    approx_norm = flat_approx.norm().item()
    exact_norm = flat_exact.norm().item()
    denom = max(exact_norm, 1e-12)
    cosine = float("nan")
    if approx_norm > 0 and exact_norm > 0:
        cosine = float(F.cosine_similarity(flat_approx, flat_exact, dim=0).item())
    sign_agreement = float(
        (torch.sign(approx) == torch.sign(exact)).float().mean().item()
    )
    rel_l2 = float((flat_approx - flat_exact).norm().item() / denom)
    norm_ratio = float(approx_norm / denom)
    best_scale = float("nan")
    scaled_rel_l2 = float("nan")
    scale_mismatch = float("nan")
    if approx_norm > 0:
        best_scale = float(
            (
                torch.dot(flat_approx, flat_exact)
                / (torch.dot(flat_approx, flat_approx) + 1e-12)
            ).item()
        )
        scaled_rel_l2 = float(
            ((best_scale * flat_approx - flat_exact).norm().item()) / denom
        )
        scale_mismatch = float(abs(math.log10(max(norm_ratio, 1e-12))))
    var_exact = float(flat_exact.var(unbiased=False).item())
    if var_exact > 1e-12:
        mse = float(torch.mean((flat_approx - flat_exact) ** 2).item())
        r2 = float(1.0 - (mse / var_exact))
    else:
        r2 = float("nan")
    return {
        "cosine": cosine,
        "sign_agreement": sign_agreement,
        "relative_l2": rel_l2,
        "norm_ratio": norm_ratio,
        "best_scale": best_scale,
        "scaled_relative_l2": scaled_rel_l2,
        "scale_mismatch": scale_mismatch,
        "r2": r2,
        "numel": int(flat_exact.numel()),
    }


def _weighted_mean(
    frame: pd.DataFrame, value_col: str, weight_col: str = "numel"
) -> float:
    valid = frame[[value_col, weight_col]].dropna()
    if valid.empty:
        return float("nan")
    weights = valid[weight_col].to_numpy(dtype=float)
    values = valid[value_col].to_numpy(dtype=float)
    return float((weights * values).sum() / max(weights.sum(), 1e-12))


def _compute_layer_total_conductance_no_inhibition(
    rec: dict[str, Any],
    v_n: torch.Tensor,
) -> torch.Tensor:
    """Counterfactual total conductance with the inhibitory synapse bank removed."""
    g_tot = torch.ones_like(v_n)

    exc_out = rec.get("exc_out")
    if exc_out is not None:
        g_tot = g_tot + F.relu(exc_out)

    blk_module = rec.get("blk_module")
    if blk_module is not None and hasattr(blk_module, "sum_conductances"):
        g_blk = blk_module.sum_conductances().detach()[None, :].expand_as(v_n)
        g_tot = g_tot + F.relu(g_blk)

    return g_tot


def _precompute_path_propagation_factors_counterfactual(
    helper: LocalCreditAssignment,
    layer_records: list[dict[str, Any]],
    *,
    no_inhibition: bool,
    include_parent_activation_derivative: bool = False,
) -> list[torch.Tensor | float]:
    """Mirror LocalCA path factors while optionally dropping inhibitory conductance."""
    if not layer_records:
        return []

    mode = str(
        getattr(helper.local_cfg.morphology_aware, "path_factor_mode", "per_branch")
    ).lower()
    path_factors: list[torch.Tensor | float] = [1.0] * len(layer_records)

    soma_v = layer_records[-1].get("v_n")
    if isinstance(soma_v, torch.Tensor):
        if mode == "scalar_mean":
            path_factors[-1] = torch.ones(
                soma_v.size(0), 1, device=soma_v.device, dtype=soma_v.dtype
            )
        else:
            path_factors[-1] = torch.ones_like(soma_v)

    for layer_idx in range(len(layer_records) - 2, -1, -1):
        child_rec = layer_records[layer_idx]
        parent_rec = layer_records[layer_idx + 1]

        child_v = child_rec.get("v_n")
        parent_v = parent_rec.get("v_n")
        blk_module = parent_rec.get("blk_module")
        if not (
            isinstance(child_v, torch.Tensor)
            and isinstance(parent_v, torch.Tensor)
            and blk_module is not None
            and hasattr(blk_module, "weight")
        ):
            if isinstance(child_v, torch.Tensor) and mode == "scalar_mean":
                path_factors[layer_idx] = torch.ones(
                    child_v.size(0), 1, device=child_v.device, dtype=child_v.dtype
                )
            elif isinstance(child_v, torch.Tensor):
                path_factors[layer_idx] = torch.ones_like(child_v)
            else:
                path_factors[layer_idx] = 1.0
            continue

        parent_path = path_factors[layer_idx + 1]
        if not isinstance(parent_path, torch.Tensor):
            parent_path = torch.ones_like(parent_v)
        elif parent_path.size(1) == 1 and parent_v.size(1) > 1:
            parent_path = parent_path.expand(-1, parent_v.size(1))

        parent_act_deriv = torch.ones_like(parent_v)
        if include_parent_activation_derivative:
            parent_act_deriv_candidate = helper._get_layer_activation_derivative(
                parent_rec,
                parent_v,
            )
            if isinstance(parent_act_deriv_candidate, torch.Tensor):
                parent_act_deriv = parent_act_deriv_candidate.to(
                    device=parent_v.device,
                    dtype=parent_v.dtype,
                )

        if no_inhibition:
            parent_g_tot = _compute_layer_total_conductance_no_inhibition(
                parent_rec,
                parent_v,
            )
        else:
            parent_g_tot = helper._compute_layer_total_conductance(parent_rec, parent_v)
        parent_r_tot = 1.0 / (parent_g_tot + 1e-8)

        block_size = int(getattr(blk_module, "block_size", 1))
        edge_weights = (
            blk_module.weight().detach().to(device=child_v.device, dtype=child_v.dtype)
        )
        expected_child_out = edge_weights.numel()
        if child_v.size(1) != expected_child_out:
            scalar_factor = (parent_path * parent_act_deriv * parent_r_tot).mean(
                dim=1,
                keepdim=True,
            ) * edge_weights.mean()
            path_factors[layer_idx] = scalar_factor
            continue

        expanded_parent_path = helper._expand_parent_signal_to_children(
            parent_path,
            block_size,
            child_v.size(1),
        )
        expanded_parent_act = helper._expand_parent_signal_to_children(
            parent_act_deriv,
            block_size,
            child_v.size(1),
        )
        expanded_parent_r_tot = helper._expand_parent_signal_to_children(
            parent_r_tot,
            block_size,
            child_v.size(1),
        )
        edge_gain = edge_weights.reshape(1, -1)
        branch_factor = (
            expanded_parent_path
            * expanded_parent_act
            * expanded_parent_r_tot
            * edge_gain
        )
        if mode == "scalar_mean":
            branch_factor = branch_factor.mean(dim=1, keepdim=True)
        path_factors[layer_idx] = branch_factor

    return path_factors


def _factor_stats(
    factor: torch.Tensor | float, reference: torch.Tensor
) -> tuple[float, float, float]:
    if isinstance(factor, torch.Tensor):
        values = (
            factor.detach()
            .to(device=reference.device, dtype=reference.dtype)
            .reshape(-1)
        )
    else:
        values = torch.full(
            (reference.numel(),),
            float(factor),
            device=reference.device,
            dtype=reference.dtype,
        )
    mean = float(values.mean().item())
    std = float(values.std(unbiased=False).item())
    cv = float(std / max(abs(mean), 1e-12))
    return mean, std, cv


def _factor_tensor(
    factor: torch.Tensor | float, reference: torch.Tensor
) -> torch.Tensor:
    if isinstance(factor, torch.Tensor):
        values = factor.detach().to(device=reference.device, dtype=reference.dtype)
        if values.shape == reference.shape:
            return values
        if values.dim() == 2 and values.size(1) == 1:
            return values.expand_as(reference)
        return values.reshape_as(reference)
    return torch.full_like(reference, float(factor))


def _inhibitory_conductance_stats(
    helper: LocalCreditAssignment,
    rec: dict[str, Any],
    v_n: torch.Tensor,
) -> tuple[float, float]:
    inh_out = rec.get("inh_out")
    if not isinstance(inh_out, torch.Tensor):
        return 0.0, 0.0
    inh_g = F.relu(inh_out.detach()).to(device=v_n.device, dtype=v_n.dtype)
    total_g = helper._compute_layer_total_conductance(rec, v_n.detach())
    fraction = inh_g / (total_g.detach() + 1e-8)
    return float(inh_g.mean().item()), float(fraction.mean().item())


def _discover_run_dirs(root: Path) -> list[Path]:
    if (root / "config.json").exists() or (root / "config.yaml").exists():
        return [root]
    results_dir = root / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"No run/config layout found under {root}")
    run_dirs = sorted(
        path
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith("config_")
    )
    if not run_dirs:
        raise FileNotFoundError(f"No result runs found under {results_dir}")
    return run_dirs


def _factorization_rows(
    helper: LocalCreditAssignment,
    rec: dict[str, Any],
    layer_idx: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    layer = rec["layer"]
    v_n = rec["v_n"]
    exact_error = v_n.grad.detach()
    if layer.use_shunting:
        R_tot = 1.0 / (
            helper._compute_layer_total_conductance(rec, v_n.detach()) + 1e-8
        )
    else:
        R_tot = torch.ones_like(v_n.detach())

    exc_layer = rec.get("exc_module")
    x_exc = rec.get("x_exc")
    if (
        exc_layer is not None
        and isinstance(x_exc, torch.Tensor)
        and exc_layer.pre_w.grad is not None
    ):
        mask = rec.get("exc_mask")
        if not isinstance(mask, torch.Tensor):
            mask = exc_layer.weight_mask().detach()
        if layer.use_shunting:
            exc_current = rec.get("exc_out")
            if isinstance(exc_current, torch.Tensor):
                conductance_gate = (exc_current.detach() > 0).to(v_n.dtype)
            else:
                conductance_gate = torch.ones_like(v_n.detach())
            factor = exact_error * R_tot * (1.0 - v_n.detach() * conductance_gate)
        else:
            factor = exact_error
        recon = torch.einsum("bo,bi->oi", factor, x_exc.detach())
        recon = (
            recon
            * mask.to(recon.dtype)
            * helper._weight_transform_derivative(
                exc_layer.pre_w.detach(), exc_layer.weight_transform
            )
        )
        rows.append(
            {
                "layer_index": layer_idx,
                "layer_name": rec["layer_name"],
                "parameter_group": "exc_syn",
                **_vec_metrics(recon, exc_layer.pre_w.grad.detach()),
            }
        )

    inh_layer = rec.get("inh_module")
    x_inh = rec.get("x_inh")
    if (
        inh_layer is not None
        and isinstance(x_inh, torch.Tensor)
        and inh_layer.pre_w.grad is not None
    ):
        mask = rec.get("inh_mask")
        if not isinstance(mask, torch.Tensor):
            mask = inh_layer.weight_mask().detach()
        if layer.use_shunting:
            inh_current = rec.get("inh_out")
            if isinstance(inh_current, torch.Tensor):
                conductance_gate = (inh_current.detach() > 0).to(v_n.dtype)
            else:
                conductance_gate = torch.ones_like(v_n.detach())
            factor = exact_error * R_tot * (0.0 - v_n.detach() * conductance_gate)
        else:
            factor = -exact_error
        recon = torch.einsum("bo,bi->oi", factor, x_inh.detach())
        recon = (
            recon
            * mask.to(recon.dtype)
            * helper._weight_transform_derivative(
                inh_layer.pre_w.detach(), inh_layer.weight_transform
            )
        )
        rows.append(
            {
                "layer_index": layer_idx,
                "layer_name": rec["layer_name"],
                "parameter_group": "inh_syn",
                **_vec_metrics(recon, inh_layer.pre_w.grad.detach()),
            }
        )

    blk = rec.get("blk_module")
    x_blk_raw = rec.get("x_blk_raw")
    if (
        blk is not None
        and isinstance(x_blk_raw, torch.Tensor)
        and blk.log_weight.grad is not None
    ):
        x_blk = x_blk_raw.detach().view(
            x_blk_raw.size(0), blk.out_features, blk.block_size
        )
        if layer.use_shunting:
            recon = (
                exact_error.unsqueeze(-1)
                * R_tot.unsqueeze(-1)
                * (x_blk - v_n.detach().unsqueeze(-1))
            ).sum(dim=0)
        else:
            recon = (exact_error.unsqueeze(-1) * x_blk).sum(dim=0)
        recon = recon * helper._weight_transform_derivative(
            blk.log_weight.detach(), blk.weight_transform
        )
        rows.append(
            {
                "layer_index": layer_idx,
                "layer_name": rec["layer_name"],
                "parameter_group": "dendritic_cond",
                **_vec_metrics(recon, blk.log_weight.grad.detach()),
            }
        )

    return rows


def _error_rows(
    helper: LocalCreditAssignment,
    layer_records: list[dict[str, Any]],
    delta_local: torch.Tensor,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    delta_scalar = helper._reduce_error_to_scalar(delta_local)
    conductance_path_factors = helper._precompute_path_propagation_factors(
        layer_records,
        include_parent_activation_derivative=False,
    )
    conductance_path_factors_no_inhibition = (
        _precompute_path_propagation_factors_counterfactual(
            helper,
            layer_records,
            no_inhibition=True,
            include_parent_activation_derivative=False,
        )
    )
    effective_path_factors = helper._precompute_path_propagation_factors(
        layer_records,
        include_parent_activation_derivative=True,
    )
    effective_path_factors_no_inhibition = (
        _precompute_path_propagation_factors_counterfactual(
            helper,
            layer_records,
            no_inhibition=True,
            include_parent_activation_derivative=True,
        )
    )
    transported = helper._precompute_path_transport_errors(
        layer_records=layer_records,
        delta=delta_local,
        delta_scalar=delta_scalar,
    )

    rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    for layer_idx, rec in enumerate(layer_records):
        exact_error = rec["v_n"].grad.detach()
        out_features = exact_error.size(1)
        scalar = delta_scalar.expand(-1, out_features)
        if delta_local.dim() == 2 and delta_local.size(1) == out_features:
            per_soma = delta_local
        else:
            per_soma = scalar

        activation_derivative = helper._get_layer_activation_derivative(
            rec,
            v_n=rec.get("v_n"),
            v_out=rec.get("v_out"),
        )
        if not isinstance(activation_derivative, torch.Tensor):
            activation_derivative = torch.ones_like(exact_error)
        activation_derivative = activation_derivative.to(
            device=exact_error.device, dtype=exact_error.dtype
        )

        conductance_path_factor = conductance_path_factors[layer_idx]
        conductance_path_factor_no_inhibition = conductance_path_factors_no_inhibition[
            layer_idx
        ]
        effective_path_factor = effective_path_factors[layer_idx]
        effective_path_factor_no_inhibition = effective_path_factors_no_inhibition[
            layer_idx
        ]
        if isinstance(effective_path_factor, torch.Tensor):
            path_scaled_scalar = scalar * effective_path_factor.to(
                device=scalar.device, dtype=scalar.dtype
            )
        else:
            path_scaled_scalar = scalar

        if isinstance(effective_path_factor_no_inhibition, torch.Tensor):
            path_scaled_scalar_no_inhibition = (
                scalar
                * effective_path_factor_no_inhibition.to(
                    device=scalar.device,
                    dtype=scalar.dtype,
                )
            )
        else:
            path_scaled_scalar_no_inhibition = scalar

        mean_pf, std_pf, cv_pf = _factor_stats(conductance_path_factor, exact_error)
        mean_pf_no_i, std_pf_no_i, cv_pf_no_i = _factor_stats(
            conductance_path_factor_no_inhibition,
            exact_error,
        )
        pf = _factor_tensor(conductance_path_factor, exact_error).clamp_min(1e-12)
        pf_no_i = _factor_tensor(
            conductance_path_factor_no_inhibition,
            exact_error,
        ).clamp_min(1e-12)
        path_gain_suppression = pf / pf_no_i
        path_gain_log_suppression = torch.log(pf_no_i) - torch.log(pf)
        inhibitory_conductance_mean, inhibitory_conductance_fraction = (
            _inhibitory_conductance_stats(helper, rec, exact_error)
        )

        path_transport = transported[layer_idx]
        if not isinstance(path_transport, torch.Tensor):
            path_transport = scalar

        for mode, approx in [
            ("scalar", scalar * activation_derivative),
            ("per_soma", per_soma * activation_derivative),
            ("path_factor_scalar", path_scaled_scalar * activation_derivative),
            (
                "path_factor_scalar_no_inhibition",
                path_scaled_scalar_no_inhibition * activation_derivative,
            ),
            ("path_transport", path_transport * activation_derivative),
        ]:
            rows.append(
                {
                    "layer_index": layer_idx,
                    "layer_name": rec["layer_name"],
                    "broadcast_mode": mode,
                    **_vec_metrics(approx, exact_error),
                }
            )
        path_rows.append(
            {
                "layer_index": layer_idx,
                "layer_name": rec["layer_name"],
                "path_gain_mean": mean_pf,
                "path_gain_std": std_pf,
                "path_gain_cv": cv_pf,
                "path_gain_no_inhibition_mean": mean_pf_no_i,
                "path_gain_no_inhibition_std": std_pf_no_i,
                "path_gain_no_inhibition_cv": cv_pf_no_i,
                "path_gain_suppression_mean": float(
                    path_gain_suppression.mean().item()
                ),
                "path_gain_log_suppression_mean": float(
                    path_gain_log_suppression.mean().item()
                ),
                "inhibitory_conductance_mean": inhibitory_conductance_mean,
                "inhibitory_conductance_fraction": inhibitory_conductance_fraction,
                "numel": int(exact_error.numel()),
            }
        )
    return rows, path_rows


def analyze_run(
    run_dir: Path,
    *,
    batch_size: int,
    split: str,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = _load_config_from_run(run_dir)
    x_batch, y_batch = _get_batch(config, split=split, batch_size=batch_size)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    input_dim = int(x_batch[0].numel())
    encoder_params = getattr(config.model.encoder, "params", None)
    if encoder_params is None:
        config.model.encoder.params = {"input_dim": input_dim}
    elif isinstance(encoder_params, dict):
        encoder_params["input_dim"] = input_dim
    else:
        encoder_params.input_dim = input_dim

    model, _ = initialize_model(config.model)
    model.load_state_dict(
        torch.load(_locate_model_path(run_dir), map_location=device, weights_only=False)
    )
    model = model.to(device)
    model.eval()

    helper = _make_helper()
    decoder_cache: dict[str, Any] = {}
    dec_handle = _capture_decoder_input(model, decoder_cache)
    layer_records, layer_handles = _hook_branch_layers(model)
    topk_modules = [
        module for module in model.modules() if isinstance(module, TopKLinear)
    ]
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
    # Keep y_hat attached here so decoder-aware soma mapping can use the
    # decoder Jacobian when the decoder is nonlinear.
    _, delta_local = helper._resolve_local_soma_signals(
        model=model,
        y_hat=y_hat,
        delta_out=delta_out.detach(),
    )

    model.zero_grad(set_to_none=True)
    loss.backward()

    factor_rows: list[dict[str, Any]] = []
    for layer_idx, rec in enumerate(layer_records):
        factor_rows.extend(_factorization_rows(helper, rec, layer_idx))

    error_rows, path_rows = _error_rows(helper, layer_records, delta_local.detach())

    run_meta = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "dataset": config.data.dataset_name,
        "network_type": config.model.core.type,
        "strategy": _get_value(_get_value(config.training, "main"), "strategy"),
        "rule_variant": _get_value(
            _get_value(_get_value(config.training, "main"), "learning_strategy_config"),
            "rule_variant",
        ),
        "error_broadcast_mode": _get_value(
            _get_value(_get_value(config.training, "main"), "learning_strategy_config"),
            "error_broadcast_mode",
        ),
        "input_mode": (
            _get_value(config.model.core.transfer, "input_mode")
            if hasattr(config.model.core, "transfer")
            else None
        ),
        "inhibitory_layer_sizes": (
            json.dumps(
                _get_value(
                    config.model.core.architecture,
                    "inhibitory_layer_sizes",
                    [],
                )
            )
            if hasattr(config.model.core, "architecture")
            else "[]"
        ),
        "ei_value": (
            _get_value(
                config.model.core.connectivity,
                "ei_synapses_per_branch_per_layer",
                [None],
            )[0]
            if hasattr(config.model.core, "connectivity")
            else None
        ),
        "ie_value": (
            _get_value(
                config.model.core.connectivity,
                "ie_synapses_per_branch_per_layer",
                [None],
            )[0]
            if hasattr(config.model.core, "connectivity")
            else None
        ),
        "loss_name": str(loss_name),
        "loss_value": float(loss.detach().item()),
    }

    factor_df = pd.DataFrame([{**run_meta, **row} for row in factor_rows])
    error_df = pd.DataFrame([{**run_meta, **row} for row in error_rows])
    path_df = pd.DataFrame([{**run_meta, **row} for row in path_rows])
    return factor_df, error_df, path_df


def summarize_run(
    factor_df: pd.DataFrame, error_df: pd.DataFrame, path_df: pd.DataFrame
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "run_dir": (
            factor_df["run_dir"].iloc[0]
            if not factor_df.empty
            else error_df["run_dir"].iloc[0]
        ),
    }
    if not factor_df.empty:
        summary["factorization_weighted_cosine"] = _weighted_mean(factor_df, "cosine")
        summary["factorization_weighted_relative_l2"] = _weighted_mean(
            factor_df, "relative_l2"
        )
        summary["factorization_weighted_scaled_relative_l2"] = _weighted_mean(
            factor_df, "scaled_relative_l2"
        )
        summary["factorization_weighted_scale_mismatch"] = _weighted_mean(
            factor_df, "scale_mismatch"
        )
        summary["factorization_weighted_r2"] = _weighted_mean(factor_df, "r2")
    if not error_df.empty:
        for mode, group in error_df.groupby("broadcast_mode"):
            summary[f"{mode}_weighted_cosine"] = _weighted_mean(group, "cosine")
            summary[f"{mode}_weighted_relative_l2"] = _weighted_mean(
                group, "relative_l2"
            )
            summary[f"{mode}_weighted_scaled_relative_l2"] = _weighted_mean(
                group, "scaled_relative_l2"
            )
            summary[f"{mode}_weighted_scale_mismatch"] = _weighted_mean(
                group, "scale_mismatch"
            )
            summary[f"{mode}_weighted_r2"] = _weighted_mean(group, "r2")
    if not path_df.empty:
        summary["path_gain_cv_mean"] = _weighted_mean(path_df, "path_gain_cv")
        summary["path_gain_mean"] = _weighted_mean(path_df, "path_gain_mean")
        summary["path_gain_no_inhibition_cv_mean"] = _weighted_mean(
            path_df,
            "path_gain_no_inhibition_cv",
        )
        summary["path_gain_no_inhibition_mean"] = _weighted_mean(
            path_df,
            "path_gain_no_inhibition_mean",
        )
        summary["path_gain_suppression_mean"] = _weighted_mean(
            path_df,
            "path_gain_suppression_mean",
        )
        summary["path_gain_log_suppression_mean"] = _weighted_mean(
            path_df,
            "path_gain_log_suppression_mean",
        )
        summary["inhibitory_conductance_mean"] = _weighted_mean(
            path_df,
            "inhibitory_conductance_mean",
        )
        summary["inhibitory_conductance_fraction"] = _weighted_mean(
            path_df,
            "inhibitory_conductance_fraction",
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, help="Single run directory with config/checkpoint"
    )
    parser.add_argument(
        "--sweep-dir", type=Path, help="Sweep directory with results/config_* subdirs"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Where CSV summaries are written"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if bool(args.run_dir) == bool(args.sweep_dir):
        raise ValueError("Specify exactly one of --run-dir or --sweep-dir.")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )

    root = args.run_dir or args.sweep_dir
    run_dirs = _discover_run_dirs(root)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    factor_frames: list[pd.DataFrame] = []
    error_frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"[{idx}/{len(run_dirs)}] {run_dir}")
        factor_df, error_df, path_df = analyze_run(
            run_dir=run_dir,
            batch_size=args.batch_size,
            split=args.split,
            device=device,
        )
        factor_frames.append(factor_df)
        error_frames.append(error_df)
        path_frames.append(path_df)
        summaries.append(summarize_run(factor_df, error_df, path_df))

    factor_all = (
        pd.concat(factor_frames, ignore_index=True) if factor_frames else pd.DataFrame()
    )
    error_all = (
        pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    )
    path_all = (
        pd.concat(path_frames, ignore_index=True) if path_frames else pd.DataFrame()
    )
    summary_all = pd.DataFrame(summaries)

    factor_all.to_csv(args.output_dir / "factorization_details.csv", index=False)
    error_all.to_csv(args.output_dir / "compartment_error_fidelity.csv", index=False)
    path_all.to_csv(args.output_dir / "path_gain_stats.csv", index=False)
    summary_all.to_csv(args.output_dir / "run_summary.csv", index=False)

    print(f"Saved diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()

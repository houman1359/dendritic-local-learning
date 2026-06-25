#!/usr/bin/env python3
"""Factor layer-soma feedback from within-tree transport.

This diagnostic answers the reviewer-facing question:

1. exact layer-soma error + per-soma shared branch feedback;
2. approximate/direct layer error + exact within-tree path transport;
3. exact layer-soma error + exact path transport;
4. practical direct layer error + current code per-soma feedback.

It operates on saved run directories and reports gradient alignment against
autograd separately for each equal-width core layer and parameter family.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from measure_theory_diagnostics import (  # noqa: E402
    _capture_decoder_input,
    _compute_loss,
    _get_batch,
    _get_value,
    _load_config_from_run,
    _locate_model_path,
    _to_plain_dict,
    _vec_metrics,
)

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
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_epoch import (  # noqa: E402
    _apply_path_propagation_factor,
    _resolve_stdp_error_signals,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (  # noqa: E402
    _BroadcastState,
)


_LAYER_RE = re.compile(r"core_network\.layers\.(\d+).*branch_layers\.(\d+)")


def _discover_run_dirs(root: Path) -> list[Path]:
    if (root / "config.json").exists() or (root / "config.yaml").exists():
        return [root]
    results_dir = root / "results"
    if not results_dir.exists():
        raise FileNotFoundError(f"No run/config layout found under {root}")
    runs = [
        path
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.startswith("config_")
    ]
    return sorted(runs, key=lambda p: int(p.name.split("_")[-1]))


def _core_layer_index(layer_name: str) -> int:
    match = _LAYER_RE.search(layer_name)
    return int(match.group(1)) if match else -1


def _branch_layer_index(layer_name: str) -> int:
    match = _LAYER_RE.search(layer_name)
    return int(match.group(2)) if match else -1


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


def _hook_branch_layers(model: torch.nn.Module) -> tuple[list[dict[str, Any]], list[Any]]:
    records: list[dict[str, Any]] = []
    handles: list[Any] = []

    for name, module in model.named_modules():
        if not isinstance(module, DendriticBranchLayer):
            continue
        rec: dict[str, Any] = {
            "layer_name": name,
            "layer": module,
            "core_layer_index": _core_layer_index(name),
            "branch_layer_index": _branch_layer_index(name),
        }

        def _react_hook(_mod, inputs, outputs, rec=rec):
            pre_v = inputs[0]
            if isinstance(pre_v, torch.Tensor) and pre_v.requires_grad:
                pre_v.retain_grad()
            if isinstance(outputs, torch.Tensor) and outputs.requires_grad:
                outputs.retain_grad()
            rec["v_n"] = pre_v
            rec["v_out"] = outputs

        handles.append(module.reactivation.register_forward_hook(_react_hook))

        if getattr(module, "branch_excitation", None) is not None:
            exc_layer = module.branch_excitation

            def _exc_hook(_mod, inputs, outputs, rec=rec, layer=exc_layer):
                rec["x_exc"] = inputs[0]
                rec["exc_out"] = outputs
                rec["exc_module"] = layer
                rec["exc_mask"] = getattr(layer, "_last_forward_weight_mask", None)

            handles.append(exc_layer.register_forward_hook(_exc_hook))

        if getattr(module, "branch_inhibition", None) is not None:
            inh_layer = module.branch_inhibition

            def _inh_hook(_mod, inputs, outputs, rec=rec, layer=inh_layer):
                rec["x_inh"] = inputs[0]
                rec["inh_out"] = outputs
                rec["inh_module"] = layer
                rec["inh_mask"] = getattr(layer, "_last_forward_weight_mask", None)

            handles.append(inh_layer.register_forward_hook(_inh_hook))

        if getattr(module, "branches_to_output", None) is not None:
            blk_layer = module.branches_to_output

            def _blk_hook(_mod, inputs, outputs, rec=rec, layer=blk_layer):
                rec["x_blk_raw"] = inputs[0]
                rec["blk_out"] = outputs
                rec["blk_module"] = layer

            handles.append(blk_layer.register_forward_hook(_blk_hook))

        records.append(rec)

    return records, handles


def _detach_record(rec: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in rec.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach()
        else:
            out[key] = value
    return out


def _group_records_by_core_layer(
    records: list[dict[str, Any]],
) -> "OrderedDict[int, list[dict[str, Any]]]":
    grouped: "OrderedDict[int, list[dict[str, Any]]]" = OrderedDict()
    for rec in records:
        idx = int(rec.get("core_layer_index", -1))
        grouped.setdefault(idx, []).append(rec)
    for idx, group in grouped.items():
        grouped[idx] = sorted(group, key=lambda r: int(r.get("branch_layer_index", 0)))
    return grouped


def _make_helper(config: Any, *, rule_variant: str, loss_name: str) -> LocalCreditAssignment:
    main_cfg = _get_value(config.training, "main", {})
    local_cfg = copy.deepcopy(
        _to_plain_dict(_get_value(main_cfg, "learning_strategy_config", {}))
    )
    local_cfg["rule_variant"] = rule_variant
    local_cfg["decoder_update_mode"] = "none"
    local_cfg["update_reactivation"] = False
    hsic_cfg = _to_plain_dict(local_cfg.get("hsic", {}))
    hsic_cfg.update({"enabled": False, "weight": 0.0, "self_weight": 0.0, "target_weight": 0.0})
    local_cfg["hsic"] = hsic_cfg

    helper = LocalCreditAssignment.__new__(LocalCreditAssignment)
    helper.local_cfg = helper._build_local_rule_config(local_cfg)
    helper._layer_stats = {}
    helper._decoder_cache = {}
    helper._additive_gain_cache = {}
    helper._additive_running_var = {}
    helper._broadcast_cache = {}
    helper._stdp_traces = {}
    helper._warned_decoder_soma_fallback = False
    helper.epoch_counter = 1
    helper.loss_function = type("Loss", (), {"_loss_name": str(loss_name)})()
    return helper


def _set_encoder_input_dim(config: Any, x_batch: torch.Tensor) -> None:
    input_dim = int(x_batch[0].numel())
    encoder_params = getattr(config.model.encoder, "params", None)
    if encoder_params is None:
        config.model.encoder.params = {"input_dim": input_dim}
    elif isinstance(encoder_params, dict):
        encoder_params["input_dim"] = input_dim
    else:
        encoder_params.input_dim = input_dim


def _load_model(config: Any, run_dir: Path, device: torch.device) -> torch.nn.Module:
    model, _ = initialize_model(config.model)
    state = torch.load(_locate_model_path(run_dir), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _capture_forward_backward(
    model: torch.nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    *,
    helper: LocalCreditAssignment,
    loss_name: str,
) -> tuple[
    list[dict[str, Any]],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    float,
]:
    decoder_cache: dict[str, Any] = {}
    dec_handle = _capture_decoder_input(model, decoder_cache)
    records, handles = _hook_branch_layers(model)
    topk_modules = [module for module in model.modules() if isinstance(module, TopKLinear)]
    for module in topk_modules:
        module.cache_mask = True

    try:
        y_hat = model(x_batch)
    finally:
        if dec_handle is not None:
            dec_handle.remove()
        for handle in handles:
            handle.remove()
        for module in topk_modules:
            module.cache_mask = False

    loss = _compute_loss(loss_name, y_hat, y_batch)
    delta_out = helper._compute_soma_error(y_hat.detach(), y_batch.detach())
    helper._decoder_cache = decoder_cache
    v0_direct, delta_direct = helper._resolve_local_soma_signals(
        model=model,
        y_hat=y_hat,
        delta_out=delta_out.detach(),
    )

    model.zero_grad(set_to_none=True)
    loss.backward()

    for rec in records:
        v_n = rec.get("v_n")
        v_out = rec.get("v_out")
        if isinstance(v_n, torch.Tensor) and v_n.grad is not None:
            rec["v_n_exact_grad"] = v_n.grad.detach().clone()
        if isinstance(v_out, torch.Tensor) and v_out.grad is not None:
            rec["v_out_exact_grad"] = v_out.grad.detach().clone()

    bp_grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    detached_records = [_detach_record(rec) for rec in records]
    return detached_records, bp_grads, v0_direct.detach(), delta_direct.detach(), float(loss.item())


def _exact_soma_seeds(
    original_records: list[dict[str, Any]],
    *,
    batch_size: int,
) -> dict[int, torch.Tensor]:
    grouped = _group_records_by_core_layer(original_records)
    seeds: dict[int, torch.Tensor] = {}
    for core_idx, group in grouped.items():
        soma_rec = group[-1]
        soma_grad = soma_rec.get("v_out_exact_grad")
        if not isinstance(soma_grad, torch.Tensor):
            raise RuntimeError(
                f"Missing exact soma activation error for core layer {core_idx}."
            )
        # Autograd sees the mean loss, while LocalCA expects per-example errors
        # and applies the batch mean inside the gradient rule.
        seeds[core_idx] = soma_grad.detach().clone() * float(batch_size)
    return seeds


def _expand_soma_to_tree(
    helper: LocalCreditAssignment,
    group: list[dict[str, Any]],
    seed: torch.Tensor,
) -> list[torch.Tensor]:
    expanded: list[torch.Tensor | None] = [None] * len(group)
    expanded[-1] = seed
    for idx in range(len(group) - 2, -1, -1):
        child_rec = group[idx]
        parent_rec = group[idx + 1]
        child_v = child_rec.get("v_n")
        parent_error = expanded[idx + 1]
        blk_module = parent_rec.get("blk_module")
        if not (
            isinstance(child_v, torch.Tensor)
            and isinstance(parent_error, torch.Tensor)
            and blk_module is not None
            and hasattr(blk_module, "block_size")
        ):
            scalar = helper._reduce_error_to_scalar(seed).to(
                device=seed.device, dtype=seed.dtype
            )
            expanded[idx] = scalar.expand(-1, child_v.size(1))
            continue
        expanded[idx] = helper._expand_parent_signal_to_children(
            parent_error.to(device=child_v.device, dtype=child_v.dtype),
            int(getattr(blk_module, "block_size", 1)),
            child_v.size(1),
        )
    return [item for item in expanded if isinstance(item, torch.Tensor)]


def _path_transport_tree(
    helper: LocalCreditAssignment,
    group: list[dict[str, Any]],
    seed: torch.Tensor,
) -> list[torch.Tensor]:
    delta_scalar = helper._reduce_error_to_scalar(seed)
    transported = helper._precompute_path_transport_errors(
        layer_records=group,
        delta=seed,
        delta_scalar=delta_scalar,
    )
    out: list[torch.Tensor] = []
    for rec, value in zip(group, transported):
        if isinstance(value, torch.Tensor):
            out.append(value)
        else:
            v_n = rec["v_n"]
            out.append(delta_scalar.to(device=v_n.device, dtype=v_n.dtype).expand(-1, v_n.size(1)))
    return out


def _code_per_soma_tree(
    helper: LocalCreditAssignment,
    group: list[dict[str, Any]],
    seed: torch.Tensor,
) -> list[torch.Tensor]:
    scalar = helper._reduce_error_to_scalar(seed)
    out: list[torch.Tensor] = []
    for rec in group:
        v_n = rec["v_n"]
        if seed.dim() == 2 and seed.size(1) == v_n.size(1):
            out.append(seed.to(device=v_n.device, dtype=v_n.dtype))
        else:
            out.append(scalar.to(device=v_n.device, dtype=v_n.dtype).expand(-1, v_n.size(1)))
    return out


def _condition_errors(
    helper: LocalCreditAssignment,
    grouped: "OrderedDict[int, list[dict[str, Any]]]",
    exact_seeds: dict[int, torch.Tensor],
    direct_seed: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    conditions: dict[str, dict[str, torch.Tensor]] = {
        "exact_soma_blockwise_per_soma": {},
        "approx_direct_path_transport": {},
        "exact_soma_path_transport": {},
        "approx_direct_code_per_soma": {},
        "approx_direct_blockwise_per_soma": {},
    }
    for core_idx, group in grouped.items():
        exact_seed = exact_seeds[core_idx]
        if direct_seed.dim() == 2 and direct_seed.size(1) == exact_seed.size(1):
            approx_seed = direct_seed.to(device=exact_seed.device, dtype=exact_seed.dtype)
        else:
            approx_seed = helper._reduce_error_to_scalar(direct_seed).to(
                device=exact_seed.device, dtype=exact_seed.dtype
            ).expand_as(exact_seed)

        per_condition = {
            "exact_soma_blockwise_per_soma": _expand_soma_to_tree(helper, group, exact_seed),
            "approx_direct_path_transport": _path_transport_tree(helper, group, approx_seed),
            "exact_soma_path_transport": _path_transport_tree(helper, group, exact_seed),
            "approx_direct_code_per_soma": _code_per_soma_tree(helper, group, approx_seed),
            "approx_direct_blockwise_per_soma": _expand_soma_to_tree(helper, group, approx_seed),
        }
        for condition, values in per_condition.items():
            for rec, e_n in zip(group, values):
                conditions[condition][rec["layer_name"]] = e_n.detach()
    return conditions


def _apply_local_grads_from_errors(
    model: torch.nn.Module,
    helper: LocalCreditAssignment,
    records: list[dict[str, Any]],
    e_by_layer_name: dict[str, torch.Tensor],
    *,
    v0: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    helper._layer_stats = {}
    helper._additive_gain_cache = {}
    helper._additive_running_var = {}
    helper._stdp_traces = {}
    batch_size = int(v0.size(0))
    num_layers = len(records)
    broadcast_state = _BroadcastState(mode="custom", transported_errors=[], feedback_seeds=[])

    for layer_idx, rec in enumerate(records):
        v_n = rec.get("v_n")
        if not isinstance(v_n, torch.Tensor):
            continue
        e_n = e_by_layer_name[rec["layer_name"]].to(device=v_n.device, dtype=v_n.dtype)
        e_n, stdp_error_signal = _resolve_stdp_error_signals(helper.local_cfg, e_n)
        layer_dynamics_mode = helper._resolve_layer_dynamics_mode(rec)
        r_tot = helper._compute_local_input_resistance(
            rec=rec,
            v_n=v_n,
            layer_dynamics_mode=layer_dynamics_mode,
        )
        e_n = _apply_path_propagation_factor(
            helper.local_cfg,
            broadcast_state,
            rec,
            e_n,
        )
        modulators = helper._compute_local_layer_modulators(
            rec=rec,
            v0=v0,
            layer_depth=layer_idx + 1,
        )
        post_factors = helper._compute_local_post_factors(
            rec=rec,
            e_n=e_n,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            layer_idx=layer_idx,
            modulators=modulators,
        )
        helper._apply_layer_local_gradients(
            rec=rec,
            layer_idx=layer_idx,
            num_layers=num_layers,
            y_target=None,
            batch_size=batch_size,
            e_n=e_n,
            stdp_error_signal=stdp_error_signal,
            v_n=v_n,
            r_tot=r_tot,
            layer_dynamics_mode=layer_dynamics_mode,
            modulators=modulators,
            post_factors=post_factors,
            update_reactivation=False,
        )

    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


def _alignment_rows(
    local_grads: dict[str, torch.Tensor],
    bp_grads: dict[str, torch.Tensor],
    *,
    meta: dict[str, Any],
    condition: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in sorted(set(local_grads) & set(bp_grads)):
        component = _component_from_name(name)
        if component in {"decoder", "other", "reactivation"}:
            continue
        metrics = _vec_metrics(local_grads[name].float(), bp_grads[name].float())
        rows.append(
            {
                **meta,
                "condition": condition,
                "parameter_name": name,
                "core_layer_index": _core_layer_index(name),
                "branch_layer_index": _branch_layer_index(name),
                "component": component,
                "local_grad_norm": float(local_grads[name].float().norm().item()),
                "backprop_grad_norm": float(bp_grads[name].float().norm().item()),
                "backprop_grad_energy": float(
                    bp_grads[name].float().square().sum().item()
                ),
                **{
                    f"gradient_{key}" if key not in {"numel"} else key: value
                    for key, value in metrics.items()
                },
            }
        )
    return rows


def _summarize(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    def wmean(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
        values = frame[value_col].to_numpy(dtype=float)
        weights = frame[weight_col].to_numpy(dtype=float)
        valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
        if not valid.any():
            return float("nan")
        return float((values[valid] * weights[valid]).sum() / weights[valid].sum())

    group_cols = [
        "condition",
        "core_layer_index",
        "component",
        "network_type",
        "trained_broadcast_mode",
        "diagnostic_rule_variant",
    ]
    summary_rows = []
    for keys, frame in rows.groupby(group_cols, dropna=False):
        summary_rows.append(
            {
                "condition": keys[0],
                "core_layer_index": int(keys[1]),
                "component": keys[2],
                "network_type": keys[3],
                "trained_broadcast_mode": keys[4],
                "diagnostic_rule_variant": keys[5],
                "n_parameters": int(len(frame)),
                "total_numel": int(frame["numel"].sum()),
                "total_backprop_grad_energy": float(
                    frame["backprop_grad_energy"].sum()
                ),
                "numel_weighted_gradient_cosine": wmean(
                    frame, "gradient_cosine", "numel"
                ),
                "energy_weighted_gradient_cosine": wmean(
                    frame, "gradient_cosine", "backprop_grad_energy"
                ),
                "numel_weighted_scaled_relative_l2": wmean(
                    frame, "gradient_scaled_relative_l2", "numel"
                ),
                "energy_weighted_scaled_relative_l2": wmean(
                    frame, "gradient_scaled_relative_l2", "backprop_grad_energy"
                ),
                "numel_weighted_relative_l2": wmean(
                    frame, "gradient_relative_l2", "numel"
                ),
                "energy_weighted_relative_l2": wmean(
                    frame, "gradient_relative_l2", "backprop_grad_energy"
                ),
                "numel_weighted_norm_ratio": wmean(
                    frame, "gradient_norm_ratio", "numel"
                ),
                "energy_weighted_norm_ratio": wmean(
                    frame, "gradient_norm_ratio", "backprop_grad_energy"
                ),
            }
        )
    return pd.DataFrame(summary_rows).sort_values(group_cols)


def analyze_run(
    run_dir: Path,
    *,
    batch_size: int,
    split: str,
    device: torch.device,
    rule_variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = _load_config_from_run(run_dir)
    x_batch, y_batch = _get_batch(config, split=split, batch_size=batch_size)
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    _set_encoder_input_dim(config, x_batch)

    main_cfg = _get_value(config.training, "main", {})
    common_cfg = _to_plain_dict(_get_value(main_cfg, "common", {}))
    loss_name = str(common_cfg.get("loss_function", "cat_nll"))
    train_local_cfg = _to_plain_dict(_get_value(main_cfg, "learning_strategy_config", {}))

    helper = _make_helper(config, rule_variant=rule_variant, loss_name=loss_name)
    model = _load_model(config, run_dir, device)
    records, bp_grads, v0_direct, delta_direct, loss_value = _capture_forward_backward(
        model,
        x_batch,
        y_batch,
        helper=helper,
        loss_name=loss_name,
    )
    exact_seeds = _exact_soma_seeds(records, batch_size=int(x_batch.size(0)))
    grouped = _group_records_by_core_layer(records)
    conditions = _condition_errors(helper, grouped, exact_seeds, delta_direct)

    meta = {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "dataset": config.data.dataset_name,
        "network_type": config.model.core.type,
        "trained_rule_variant": train_local_cfg.get("rule_variant"),
        "trained_broadcast_mode": train_local_cfg.get("error_broadcast_mode"),
        "diagnostic_rule_variant": rule_variant,
        "loss_name": loss_name,
        "loss_value": loss_value,
    }

    all_rows: list[dict[str, Any]] = []
    for condition, e_by_layer_name in conditions.items():
        local_grads = _apply_local_grads_from_errors(
            model,
            helper,
            records,
            e_by_layer_name,
            v0=v0_direct,
        )
        all_rows.extend(
            _alignment_rows(
                local_grads,
                bp_grads,
                meta=meta,
                condition=condition,
            )
        )

    details = pd.DataFrame(all_rows)
    return details, _summarize(details)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--sweep-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="train")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rule-variant", default="3f")
    args = parser.parse_args()

    if bool(args.run_dir) == bool(args.sweep_dir):
        raise ValueError("Specify exactly one of --run-dir or --sweep-dir.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    root = args.run_dir or args.sweep_dir
    run_dirs = _discover_run_dirs(root)
    if args.max_runs is not None:
        run_dirs = run_dirs[: args.max_runs]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    detail_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"[{idx}/{len(run_dirs)}] {run_dir}", flush=True)
        details, summary = analyze_run(
            run_dir,
            batch_size=args.batch_size,
            split=args.split,
            device=device,
            rule_variant=args.rule_variant,
        )
        detail_frames.append(details)
        summary_frames.append(summary)

    details_all = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    details_all.to_csv(args.output_dir / "layer_soma_factorial_details.csv", index=False)
    summary_all.to_csv(args.output_dir / "layer_soma_factorial_summary.csv", index=False)

    grouped_cols = [
        "condition",
        "core_layer_index",
        "component",
        "network_type",
        "trained_broadcast_mode",
        "diagnostic_rule_variant",
    ]
    if not summary_all.empty:
        aggregate = (
            summary_all.groupby(grouped_cols, dropna=False)
            .agg(
                n_runs=("run_dir", "count") if "run_dir" in summary_all else ("condition", "count"),
                mean_numel_weighted_gradient_cosine=(
                    "numel_weighted_gradient_cosine",
                    "mean",
                ),
                std_numel_weighted_gradient_cosine=(
                    "numel_weighted_gradient_cosine",
                    "std",
                ),
                mean_energy_weighted_gradient_cosine=(
                    "energy_weighted_gradient_cosine",
                    "mean",
                ),
                std_energy_weighted_gradient_cosine=(
                    "energy_weighted_gradient_cosine",
                    "std",
                ),
                mean_energy_weighted_scaled_relative_l2=(
                    "energy_weighted_scaled_relative_l2",
                    "mean",
                ),
                mean_energy_weighted_relative_l2=(
                    "energy_weighted_relative_l2",
                    "mean",
                ),
                mean_energy_weighted_norm_ratio=(
                    "energy_weighted_norm_ratio",
                    "mean",
                ),
            )
            .reset_index()
        )
    else:
        aggregate = pd.DataFrame()
    aggregate.to_csv(args.output_dir / "layer_soma_factorial_aggregate.csv", index=False)

    manifest = {
        "root": str(root),
        "n_runs": len(run_dirs),
        "batch_size": args.batch_size,
        "split": args.split,
        "device": str(device),
        "diagnostic_rule_variant": args.rule_variant,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved layer-soma factorial diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()

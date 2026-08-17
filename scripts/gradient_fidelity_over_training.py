#!/usr/bin/env python3
"""Measure gradient fidelity (local vs backprop alignment) over training epochs.

Loads a series of checkpoints saved during training and runs the
gradient_component_alignment diagnostic at each checkpoint. Produces a
trajectory of cosine similarity over training epochs.

Outputs:
- ``gradient_fidelity_trajectory.csv``: per-component, per-epoch alignment
- ``gradient_fidelity_figure.pdf/png``: trajectory plot

Usage:
    python gradient_fidelity_over_training.py \\
        --config experiment.yaml \\
        --checkpoint-dir <path>/checkpoints/ \\
        --output-dir <path>/gradient_fidelity/
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dendritic_modeling.config import load_config
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


def _get_value(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_plain_dict(obj):
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(obj):
            return asdict(obj)
    except ImportError:
        pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {}


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_task_config(config):
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


def _compute_loss(loss_name: str, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    import torch.nn.functional as F
    from torch.distributions import Categorical

    name = (loss_name or "").lower()
    if name in {"cat_nll", "categorical_negative_log_likelihood"}:
        return (-Categorical(logits=y_hat).log_prob(y)).mean()
    if name in {"ce", "cross_entropy"}:
        return F.cross_entropy(y_hat, y)
    return F.mse_loss(y_hat, y)


def _new_local_helper(local_cfg_dict, *, loss_name, epoch_counter=1):
    helper = LocalCreditAssignment.__new__(LocalCreditAssignment)
    helper.local_cfg = helper._build_local_rule_config(local_cfg_dict)
    helper._layer_stats = {}
    helper._decoder_cache = {}
    helper._warned_decoder_soma_fallback = False
    helper.epoch_counter = epoch_counter
    helper.loss_function = type("Loss", (), {"_loss_name": str(loss_name)})()
    return helper


def compute_local_grads(model, x_batch, y_batch, helper):
    model.train()
    for p in model.parameters():
        p.grad = None

    layer_records, handles = helper._attach_local_recorders(model)
    topk_modules = [m for m in model.modules() if isinstance(m, TopKLinear)]
    for m in topk_modules:
        m.cache_mask = True
    try:
        y_hat = model(x_batch)
    finally:
        for h in handles:
            h.remove()
        for m in topk_modules:
            m.cache_mask = False
            m._last_forward_weight_mask = None

    delta_out = helper._compute_soma_error(y_hat.detach(), y_batch)
    v0_local, delta_local = helper._resolve_local_soma_signals(
        model=model, y_hat=y_hat.detach(), delta_out=delta_out.detach()
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

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def compute_backprop_grads(model, x_batch, y_batch, loss_name):
    model.train()
    for p in model.parameters():
        p.grad = None
    y_hat = model(x_batch)
    loss = _compute_loss(loss_name, y_hat, y_batch)
    loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.detach().clone()
    return grads


def compute_alignment_at_checkpoint(
    config, checkpoint_path, x_batch, y_batch, device,
    rule_variant, broadcast_mode, loss_name, local_cfg_base, epoch,
):
    """Compute alignment for a single checkpoint."""
    model, _ = initialize_model(_get_value(config, "model"))
    model = model.to(device)

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)

    # Backprop grads
    model_bp, _ = initialize_model(_get_value(config, "model"))
    model_bp = model_bp.to(device)
    model_bp.load_state_dict(model.state_dict())
    bp_grads = compute_backprop_grads(model_bp, x_batch, y_batch, loss_name)

    # Local grads
    local_cfg = copy.deepcopy(local_cfg_base)
    local_cfg["rule_variant"] = rule_variant
    local_cfg["error_broadcast_mode"] = broadcast_mode
    local_cfg["decoder_update_mode"] = "none"
    hsic_cfg = _to_plain_dict(local_cfg.get("hsic", {}))
    hsic_cfg["enabled"] = False
    hsic_cfg["weight"] = 0.0
    hsic_cfg["self_weight"] = 0.0
    hsic_cfg["target_weight"] = 0.0
    local_cfg["hsic"] = hsic_cfg

    helper = _new_local_helper(local_cfg, loss_name=loss_name, epoch_counter=epoch)
    local_grads = compute_local_grads(model, x_batch, y_batch, helper)

    # Compute per-component alignment
    rows = []
    common_names = sorted(set(local_grads.keys()) & set(bp_grads.keys()))
    for name in common_names:
        component = _component_from_name(name)
        if component == "decoder":
            continue

        g_local = local_grads[name].reshape(-1).float()
        g_bp = bp_grads[name].reshape(-1).float()

        local_norm = float(g_local.norm().item())
        bp_norm = float(g_bp.norm().item())
        denom = local_norm * bp_norm + 1e-12
        cosine = float(torch.dot(g_local, g_bp).item() / denom) if denom > 0 else np.nan

        rows.append({
            "epoch": epoch,
            "rule_variant": rule_variant,
            "error_broadcast_mode": broadcast_mode,
            "parameter_name": name,
            "component": component,
            "numel": int(g_local.numel()),
            "cosine_similarity": cosine,
            "local_grad_norm": local_norm,
            "backprop_grad_norm": bp_norm,
            "norm_ratio": float(local_norm / (bp_norm + 1e-12)),
        })

    return rows


def plot_fidelity_trajectory(
    df: pd.DataFrame, output_path: Path | None = None
) -> plt.Figure:
    """Plot gradient fidelity (cosine similarity) over training epochs."""
    # Aggregate by epoch, rule_variant, component
    agg = df.groupby(["epoch", "rule_variant", "component"]).agg(
        mean_cosine=("cosine_similarity", "mean"),
        weighted_cosine=("cosine_similarity", lambda x: np.average(
            x, weights=df.loc[x.index, "numel"]
        )),
    ).reset_index()

    components = sorted(agg["component"].unique())
    n_comp = len(components)
    fig, axes = plt.subplots(1, max(n_comp, 1), figsize=(5 * n_comp, 4), squeeze=False)

    cmap = plt.cm.tab10
    rules = sorted(agg["rule_variant"].unique())

    for j, comp in enumerate(components):
        ax = axes[0, j]
        comp_data = agg[agg["component"] == comp]
        for i, rule in enumerate(rules):
            sub = comp_data[comp_data["rule_variant"] == rule].sort_values("epoch")
            ax.plot(sub["epoch"], sub["weighted_cosine"], label=rule, color=cmap(i % 10), linewidth=1.5)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Cosine Similarity", fontsize=10)
        ax.set_title(comp, fontsize=11)
        ax.legend(fontsize=8)
        ax.set_ylim(-0.1, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def _load_config_from_json(json_path: str):
    """Load a Config object from a sweep-generated config.json file."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    # OmegaConf can create a DictConfig from a plain dict, then load_config
    # expects a YAML file. Instead, write a temp YAML and load it.
    import tempfile
    import yaml
    # Strip keys that aren't part of the standard config schema
    keep_keys = {"experiment", "data", "model", "training", "analysis", "wandb", "outputs", "fsdp"}
    filtered = {k: v for k, v in data.items() if k in keep_keys}
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.dump(filtered, tmp, default_flow_style=False)
        tmp_path = tmp.name
    try:
        return load_config(tmp_path)
    finally:
        os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Experiment config YAML")
    config_group.add_argument("--config-json", help="Experiment config JSON (from sweep results)")
    parser.add_argument("--checkpoint-dir", required=True, help="Directory of epoch_*.pt files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rule-variants", type=str, default="3f,4f,5f",
        help="Comma-separated rule variants"
    )
    parser.add_argument(
        "--broadcast-mode", type=str, default="per_soma",
        help="Error broadcast mode"
    )
    parser.add_argument(
        "--max-checkpoints", type=int, default=20,
        help="Max checkpoints to sample (evenly spaced)"
    )
    args = parser.parse_args()

    _set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.config_json:
        config = _load_config_from_json(args.config_json)
    else:
        config = load_config(args.config)

    # Load data
    task_cfg = _build_task_config(config)
    train_ds, _, _ = get_unified_datasets(task_cfg=task_cfg)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    x_batch, y_batch = next(iter(loader))
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    # Training config
    training_cfg = _get_value(config, "training")
    main_training_cfg = _get_value(training_cfg, "main", {})
    common_cfg = _to_plain_dict(_get_value(main_training_cfg, "common", {}))
    loss_name = str(common_cfg.get("loss_function", "cat_nll"))
    local_cfg_base = _to_plain_dict(
        _get_value(main_training_cfg, "learning_strategy_config", {})
    )

    # Find checkpoints
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"))
    if not ckpt_files:
        print(f"No checkpoints found in {ckpt_dir}")
        return

    # Extract epochs and subsample
    ckpt_epochs = []
    for f in ckpt_files:
        m = re.search(r"epoch_(\d+)", f.name)
        if m:
            ckpt_epochs.append((int(m.group(1)), f))
    ckpt_epochs.sort(key=lambda x: x[0])

    if len(ckpt_epochs) > args.max_checkpoints:
        indices = np.linspace(0, len(ckpt_epochs) - 1, args.max_checkpoints, dtype=int)
        ckpt_epochs = [ckpt_epochs[i] for i in indices]

    print(f"Processing {len(ckpt_epochs)} checkpoints")

    rule_variants = [r.strip() for r in args.rule_variants.split(",")]
    all_rows = []

    for epoch, ckpt_path in ckpt_epochs:
        for rule_variant in rule_variants:
            print(f"  Epoch {epoch}, rule={rule_variant}...", flush=True)
            rows = compute_alignment_at_checkpoint(
                config, ckpt_path, x_batch, y_batch, device,
                rule_variant, args.broadcast_mode, loss_name,
                local_cfg_base, epoch,
            )
            all_rows.extend(rows)

    if not all_rows:
        print("No alignment data computed.")
        return

    df = pd.DataFrame(all_rows)
    csv_path = output_dir / "gradient_fidelity_trajectory.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows to {csv_path}")

    fig_path = output_dir / "gradient_fidelity_trajectory.png"
    plot_fidelity_trajectory(df, fig_path)
    plt.close("all")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()

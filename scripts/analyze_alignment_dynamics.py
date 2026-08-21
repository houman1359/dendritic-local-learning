#!/usr/bin/env python3
"""Tier-1 mechanistic analyses: alignment dynamics, layer-by-layer gradient
quality, and CKA representation similarity.

Produces three figures:
  1. fig_alignment_dynamics.pdf   – Per-layer cosine over training (Lillicrap-style)
  2. fig_layer_gradient_decay.pdf – Gradient fidelity by layer depth
  3. fig_cka_similarity.pdf       – CKA heatmaps (local vs backprop representations)

Usage:
    # Analyses 1 & 2 (from existing gradient fidelity CSVs):
    python analyze_alignment_dynamics.py --mode trajectories \
        --data-dir analysis/gradient_fidelity --output-dir analysis/mechanistic

    # Analysis 3 (CKA – needs model checkpoints + data):
    python analyze_alignment_dynamics.py --mode cka \
        --sweep-root <path>/sweep_runs \
        --output-dir analysis/mechanistic
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ────────────────────────────────────────────────────────────────
# NeurIPS figure styling
# ────────────────────────────────────────────────────────────────
TEXTWIDTH = 5.5  # NeurIPS single-column
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SHUNTING_COLOR = "#2ca02c"
ADDITIVE_COLOR = "#1f77b4"


def _extract_layer_index(param_name: str) -> int:
    """Extract the branch_layer index from parameter name."""
    # e.g. core_network.layers.0.excitatory_cells.branch_layers.1.branch_excitation.pre_w
    parts = param_name.split(".")
    for i, p in enumerate(parts):
        if p == "branch_layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return -1


def load_trajectory_data(data_dir: Path) -> pd.DataFrame:
    """Load all gradient fidelity trajectory CSVs and label shunting/additive."""
    frames = []
    for cfg_dir in sorted(data_dir.iterdir()):
        if not cfg_dir.is_dir() or not cfg_dir.name.startswith("config_"):
            continue
        csv_path = cfg_dir / "gradient_fidelity_trajectory.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        cfg_idx = int(cfg_dir.name.split("_")[1])
        # configs 0-2 are shunting, 3-5 are additive (verified from sweep)
        df["core_type"] = "shunting" if cfg_idx < 3 else "additive"
        df["seed"] = cfg_idx % 3
        df["config_idx"] = cfg_idx
        df["layer_idx"] = df["parameter_name"].apply(_extract_layer_index)
        df["epoch"] = df["epoch"].astype(int)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No gradient fidelity CSVs found in {data_dir}")
    return pd.concat(frames, ignore_index=True)


# ════════════════════════════════════════════════════════════════
# Analysis 1: Alignment dynamics over training (Lillicrap-style)
# ════════════════════════════════════════════════════════════════
def plot_alignment_dynamics(df: pd.DataFrame, output_dir: Path):
    """Per-layer weighted cosine similarity over training epochs."""
    # Compute parameter-count-weighted cosine per (config, epoch, layer)
    df_valid = df[df["cosine_similarity"].notna()].copy()
    df_valid["weighted_cos"] = df_valid["cosine_similarity"] * df_valid["numel"]

    grouped = (
        df_valid.groupby(["core_type", "seed", "epoch", "layer_idx"])
        .agg(sum_wcos=("weighted_cos", "sum"), sum_numel=("numel", "sum"))
        .reset_index()
    )
    grouped["w_cosine"] = grouped["sum_wcos"] / grouped["sum_numel"]

    # Average over seeds, compute std
    agg = (
        grouped.groupby(["core_type", "epoch", "layer_idx"])["w_cosine"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    layers = sorted(agg["layer_idx"].unique())
    n_layers = len(layers)

    fig, axes = plt.subplots(1, n_layers, figsize=(TEXTWIDTH, 2.0), sharey=True)
    if n_layers == 1:
        axes = [axes]

    layer_names = {0: "Layer 0\n(distal)", 1: "Layer 1\n(mid)", 2: "Layer 2\n(proximal)"}

    for ax, layer in zip(axes, layers):
        for core_type, color, label in [
            ("shunting", SHUNTING_COLOR, "Shunting"),
            ("additive", ADDITIVE_COLOR, "Additive"),
        ]:
            sub = agg[(agg["core_type"] == core_type) & (agg["layer_idx"] == layer)]
            sub = sub.sort_values("epoch")
            ax.plot(sub["epoch"], sub["mean"], color=color, label=label, linewidth=1.5)
            ax.fill_between(
                sub["epoch"],
                sub["mean"] - sub["se"],
                sub["mean"] + sub["se"],
                alpha=0.2,
                color=color,
            )
        ax.set_xlabel("Epoch")
        ax.set_title(layer_names.get(layer, f"Layer {layer}"))
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    axes[0].set_ylabel("Weighted cosine\n(local vs backprop)")
    axes[-1].legend(frameon=False, loc="lower right")
    fig.suptitle("Gradient alignment dynamics over training", fontsize=10, y=1.02)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_alignment_dynamics.{ext}")
    plt.close(fig)
    print(f"  Saved fig_alignment_dynamics.pdf/png")

    # Also make aggregate (all layers combined) version
    agg_all = (
        grouped.groupby(["core_type", "seed", "epoch"])
        .apply(lambda g: np.average(g["w_cosine"], weights=g["sum_numel"]))
        .reset_index(name="w_cosine")
    )
    agg_all2 = (
        agg_all.groupby(["core_type", "epoch"])["w_cosine"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg_all2["se"] = agg_all2["std"] / np.sqrt(agg_all2["count"])

    fig2, ax2 = plt.subplots(figsize=(TEXTWIDTH * 0.5, 2.2))
    for core_type, color, label in [
        ("shunting", SHUNTING_COLOR, "Shunting"),
        ("additive", ADDITIVE_COLOR, "Additive"),
    ]:
        sub = agg_all2[agg_all2["core_type"] == core_type].sort_values("epoch")
        ax2.plot(sub["epoch"], sub["mean"], color=color, label=label, linewidth=1.5)
        ax2.fill_between(
            sub["epoch"],
            sub["mean"] - sub["se"],
            sub["mean"] + sub["se"],
            alpha=0.2,
            color=color,
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Weighted cosine\n(local vs backprop)")
    ax2.set_title("Aggregate alignment dynamics")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.legend(frameon=False)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(output_dir / f"fig_alignment_dynamics_aggregate.{ext}")
    plt.close(fig2)
    print(f"  Saved fig_alignment_dynamics_aggregate.pdf/png")

    # Print key statistics
    for core in ["shunting", "additive"]:
        epoch0 = agg_all2[(agg_all2["core_type"] == core) & (agg_all2["epoch"] == 0)]
        epoch_max = agg_all2[agg_all2["core_type"] == core].sort_values("epoch").iloc[-1]
        if len(epoch0) > 0:
            print(f"  {core}: epoch 0 cosine = {epoch0['mean'].values[0]:.4f}, "
                  f"final cosine = {epoch_max['mean']:.4f}")


# ════════════════════════════════════════════════════════════════
# Analysis 2: Layer-by-layer gradient quality decay
# ════════════════════════════════════════════════════════════════
def plot_layer_gradient_decay(df: pd.DataFrame, output_dir: Path):
    """Gradient fidelity (cosine + scale) as function of layer depth."""
    # Use epoch 0 (initialization) and final epoch
    df_valid = df[df["cosine_similarity"].notna()].copy()
    epochs = sorted(df_valid["epoch"].unique())
    epoch_init = epochs[0]
    epoch_final = epochs[-1]

    df_valid["weighted_cos"] = df_valid["cosine_similarity"] * df_valid["numel"]
    df_valid["log_ratio"] = np.abs(np.log10(df_valid["norm_ratio"].clip(1e-10)))
    df_valid["weighted_scale"] = df_valid["log_ratio"] * df_valid["numel"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2))

    for epoch_label, epoch_val, ls in [
        ("Init", epoch_init, "--"),
        ("Final", epoch_final, "-"),
    ]:
        grouped = (
            df_valid[df_valid["epoch"] == epoch_val]
            .groupby(["core_type", "seed", "layer_idx"])
            .agg(
                sum_wcos=("weighted_cos", "sum"),
                sum_wscale=("weighted_scale", "sum"),
                sum_numel=("numel", "sum"),
            )
            .reset_index()
        )
        grouped["w_cosine"] = grouped["sum_wcos"] / grouped["sum_numel"]
        grouped["w_scale"] = grouped["sum_wscale"] / grouped["sum_numel"]

        agg = (
            grouped.groupby(["core_type", "layer_idx"])
            .agg(
                cos_mean=("w_cosine", "mean"),
                cos_se=("w_cosine", "sem"),
                scale_mean=("w_scale", "mean"),
                scale_se=("w_scale", "sem"),
            )
            .reset_index()
        )

        for core_type, color in [
            ("shunting", SHUNTING_COLOR),
            ("additive", ADDITIVE_COLOR),
        ]:
            sub = agg[agg["core_type"] == core_type].sort_values("layer_idx")
            label = f"{core_type.capitalize()} ({epoch_label})"
            ax1.errorbar(
                sub["layer_idx"],
                sub["cos_mean"],
                yerr=sub["cos_se"],
                color=color,
                linestyle=ls,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=label,
                capsize=2,
            )
            ax2.errorbar(
                sub["layer_idx"],
                sub["scale_mean"],
                yerr=sub["scale_se"],
                color=color,
                linestyle=ls,
                marker="o",
                markersize=4,
                linewidth=1.5,
                label=label,
                capsize=2,
            )

    ax1.set_xlabel("Layer depth (0=distal, 2=proximal)")
    ax1.set_ylabel("Weighted cosine")
    ax1.set_title("Directional alignment by layer")
    ax1.legend(frameon=False, fontsize=6)
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ax2.set_xlabel("Layer depth (0=distal, 2=proximal)")
    ax2.set_ylabel("|log₁₀(norm ratio)|")
    ax2.set_title("Scale mismatch by layer")
    ax2.legend(frameon=False, fontsize=6)

    fig.suptitle("Layer-by-layer gradient quality", fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_layer_gradient_decay.{ext}")
    plt.close(fig)
    print(f"  Saved fig_layer_gradient_decay.pdf/png")

    # Print key statistics
    final = df_valid[df_valid["epoch"] == epoch_final]
    for core in ["shunting", "additive"]:
        for layer in sorted(df_valid["layer_idx"].unique()):
            sub = final[(final["core_type"] == core) & (final["layer_idx"] == layer)]
            if len(sub) > 0:
                wcos = (sub["cosine_similarity"] * sub["numel"]).sum() / sub["numel"].sum()
                print(f"  {core} layer {layer}: weighted cosine = {wcos:.4f}")


# ════════════════════════════════════════════════════════════════
# Analysis 3: Component-wise breakdown (E vs I vs dendritic)
# ════════════════════════════════════════════════════════════════
def plot_component_dynamics(df: pd.DataFrame, output_dir: Path):
    """Track how different component types (E synapses, I synapses,
    dendritic conductances) evolve their gradient fidelity over training."""
    df_valid = df[df["cosine_similarity"].notna()].copy()
    df_valid["weighted_cos"] = df_valid["cosine_similarity"] * df_valid["numel"]

    # Aggregate by component type
    grouped = (
        df_valid.groupby(["core_type", "seed", "epoch", "component"])
        .agg(sum_wcos=("weighted_cos", "sum"), sum_numel=("numel", "sum"))
        .reset_index()
    )
    grouped["w_cosine"] = grouped["sum_wcos"] / grouped["sum_numel"]

    agg = (
        grouped.groupby(["core_type", "epoch", "component"])["w_cosine"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    components = sorted(agg["component"].unique())
    comp_colors = {
        "excitatory_synapse": "#2196F3",
        "inhibitory_synapse": "#F44336",
        "dendritic_conductance": "#4CAF50",
        "reactivation": "#FF9800",
    }
    comp_labels = {
        "excitatory_synapse": "E synapses",
        "inhibitory_synapse": "I synapses",
        "dendritic_conductance": "Dendritic cond.",
        "reactivation": "Reactivation",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2), sharey=True)

    for ax, core_type, title in [
        (ax1, "shunting", "Shunting"),
        (ax2, "additive", "Additive"),
    ]:
        for comp in components:
            sub = agg[(agg["core_type"] == core_type) & (agg["component"] == comp)]
            sub = sub.sort_values("epoch")
            color = comp_colors.get(comp, "gray")
            label = comp_labels.get(comp, comp)
            ax.plot(sub["epoch"], sub["mean"], color=color, label=label, linewidth=1.5)
            ax.fill_between(
                sub["epoch"],
                sub["mean"] - sub["se"],
                sub["mean"] + sub["se"],
                alpha=0.15,
                color=color,
            )
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ax1.set_ylabel("Weighted cosine\n(local vs backprop)")
    ax2.legend(frameon=False, loc="lower right", fontsize=6)
    fig.suptitle("Component-wise gradient alignment dynamics", fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_component_dynamics.{ext}")
    plt.close(fig)
    print(f"  Saved fig_component_dynamics.pdf/png")


# ════════════════════════════════════════════════════════════════
# Analysis 4: CKA representation similarity
# ════════════════════════════════════════════════════════════════
def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two activation matrices.

    X: (n_samples, d_x), Y: (n_samples, d_y)
    Returns scalar CKA similarity in [0, 1].
    """
    n = X.shape[0]
    # Center
    H = np.eye(n) - np.ones((n, n)) / n
    # Linear kernel + centering
    KX = X @ X.T
    KY = Y @ Y.T
    HKX = H @ KX @ H
    HKY = H @ KY @ H
    # HSIC
    hsic_xy = np.trace(HKX @ HKY) / (n - 1) ** 2
    hsic_xx = np.trace(HKX @ HKX) / (n - 1) ** 2
    hsic_yy = np.trace(HKY @ HKY) / (n - 1) ** 2
    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-12:
        return 0.0
    return float(hsic_xy / denom)


def collect_layer_activations(model, dataloader, device, max_batches=10):
    """Run forward pass and collect per-layer activations via hooks."""
    import torch

    activations = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            if name not in activations:
                activations[name] = []
            activations[name].append(output.detach().cpu())
        return hook_fn

    # Hook into each branch_layer's output
    for name, module in model.named_modules():
        if "branch_layers" in name and name.count(".") == 5:
            # e.g. core_network.layers.0.excitatory_cells.branch_layers.0
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Also hook the decoder output
    for name, module in model.named_modules():
        if "decoder" in name and hasattr(module, "weight"):
            hooks.append(module.register_forward_hook(make_hook("decoder")))

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            if isinstance(batch, (list, tuple)):
                x = batch[0].to(device)
            else:
                x = batch.to(device)
            model(x)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Concatenate
    result = {}
    for name, tensors in activations.items():
        cat = torch.cat(tensors, dim=0).numpy()
        if cat.ndim > 2:
            cat = cat.reshape(cat.shape[0], -1)
        result[name] = cat

    return result


def plot_cka_analysis(sweep_root: Path, output_dir: Path):
    """Compute CKA between local-learning and backprop-trained networks."""
    try:
        import torch
        import yaml
    except ImportError as e:
        print(f"  CKA analysis requires torch and yaml: {e}")
        return

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    from dendritic_modeling.config import load_config
    from dendritic_modeling.datasets import get_unified_datasets
    from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

    # We need pairs of (backprop-trained, local-learning-trained) on same architecture
    # Use gradient fidelity sweep (backprop) + phase2b (local learning)
    gf_sweep = sweep_root / "sweep_neurips_gradient_fidelity_checkpoints_20260219163546"
    p2b_sweep = sweep_root / "sweep_neurips_phase2b_gap_closing_pilot_20260218161300"

    if not gf_sweep.exists() or not p2b_sweep.exists():
        print(f"  CKA: sweep directories not found, skipping")
        return

    device = torch.device("cpu")

    # Load backprop model (shunting, config_0)
    bp_config_path = gf_sweep / "configs" / "unified_config_0.yaml"
    bp_result_dir = gf_sweep / "results" / "config_0"
    bp_model_path = bp_result_dir / "main_network" / "standard_best_model.pt"

    if not bp_model_path.exists():
        print(f"  CKA: backprop model not found at {bp_model_path}")
        return

    # Load local learning model (shunting) - find config with shunting + 5f
    local_model_path = None
    local_config_path = None
    for cfg_name in sorted(os.listdir(p2b_sweep / "results")):
        cfg_dir = p2b_sweep / "results" / cfg_name
        cfg_json = cfg_dir / "config.json"
        if not cfg_json.exists():
            continue
        with open(cfg_json) as f:
            c = json.load(f)
        core_type = c.get("model", {}).get("core", {}).get("type", "")
        rule = (
            c.get("training", {}).get("main", {})
            .get("learning_strategy_config", {})
            .get("rule_variant", "")
        )
        model_pt = cfg_dir / "main_network" / "local_learning_best_model.pt"
        if not model_pt.exists():
            model_pt = cfg_dir / "final_model.pt"
        if core_type == "dendritic_shunting" and rule == "5f" and model_pt.exists():
            local_model_path = model_pt
            local_config_path = cfg_json
            break

    if local_model_path is None:
        print("  CKA: no matching local learning model found, trying alternative...")
        # Try fashion MNIST sweep
        fm_sweep = sweep_root / "sweep_neurips_fashion_mnist_20260219163051"
        if fm_sweep.exists():
            for cfg_name in sorted(os.listdir(fm_sweep / "results")):
                cfg_dir = fm_sweep / "results" / cfg_name
                cfg_json = cfg_dir / "config.json"
                if not cfg_json.exists():
                    continue
                with open(cfg_json) as f:
                    c = json.load(f)
                core_type = c.get("model", {}).get("core", {}).get("type", "")
                model_pt = cfg_dir / "main_network" / "local_learning_best_model.pt"
                if not model_pt.exists():
                    model_pt = cfg_dir / "final_model.pt"
                if core_type == "dendritic_shunting" and model_pt.exists():
                    local_model_path = model_pt
                    local_config_path = cfg_json
                    break

    if local_model_path is None:
        print("  CKA: no local learning model found, skipping CKA analysis")
        return

    print(f"  CKA: backprop model = {bp_model_path}")
    print(f"  CKA: local model = {local_model_path}")

    # Load configs and models
    bp_config = load_config(str(bp_config_path))
    bp_model = initialize_model(bp_config, device)
    bp_state = torch.load(bp_model_path, map_location=device)
    if "model_state_dict" in bp_state:
        bp_model.load_state_dict(bp_state["model_state_dict"])
    else:
        bp_model.load_state_dict(bp_state)

    local_config = load_config(str(local_config_path))
    local_model = initialize_model(local_config, device)
    local_state = torch.load(local_model_path, map_location=device)
    if "model_state_dict" in local_state:
        local_model.load_state_dict(local_state["model_state_dict"])
    else:
        local_model.load_state_dict(local_state)

    # Build dataloader
    task_cfg = {
        "base_dir": bp_config.data.base_dir or "",
        "dataset_name": bp_config.data.dataset_name,
        "processing": bp_config.data.processing,
    }
    datasets = get_unified_datasets(**task_cfg)
    test_loader = torch.utils.data.DataLoader(
        datasets["test"], batch_size=256, shuffle=False
    )

    # Collect activations
    print("  CKA: collecting backprop model activations...")
    bp_acts = collect_layer_activations(bp_model, test_loader, device)
    print("  CKA: collecting local model activations...")
    local_acts = collect_layer_activations(local_model, test_loader, device)

    # Compute CKA matrix
    bp_layers = sorted(bp_acts.keys())
    local_layers = sorted(local_acts.keys())

    if not bp_layers or not local_layers:
        print("  CKA: no layer activations captured, skipping")
        return

    cka_matrix = np.zeros((len(bp_layers), len(local_layers)))
    for i, bp_layer in enumerate(bp_layers):
        for j, local_layer in enumerate(local_layers):
            cka_matrix[i, j] = linear_cka(bp_acts[bp_layer], local_acts[local_layer])

    # Plot
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.55, TEXTWIDTH * 0.5))
    im = ax.imshow(cka_matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(local_layers)))
    ax.set_xticklabels(
        [f"L{i}" for i in range(len(local_layers))], rotation=45, ha="right"
    )
    ax.set_yticks(range(len(bp_layers)))
    ax.set_yticklabels([f"L{i}" for i in range(len(bp_layers))])
    ax.set_xlabel("Local learning layers")
    ax.set_ylabel("Backprop layers")
    ax.set_title("CKA similarity\n(Shunting: local vs backprop)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Annotate diagonal
    for i in range(min(len(bp_layers), len(local_layers))):
        ax.text(
            i, i, f"{cka_matrix[i, i]:.2f}",
            ha="center", va="center", fontsize=7, color="white"
        )

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_cka_similarity.{ext}")
    plt.close(fig)
    print(f"  Saved fig_cka_similarity.pdf/png")

    # Print diagonal CKA values
    for i in range(min(len(bp_layers), len(local_layers))):
        print(f"  CKA diagonal L{i}: {cka_matrix[i, i]:.4f}")


# ════════════════════════════════════════════════════════════════
# Analysis 5: Combined Refinetti-style align-then-memorize plot
# ════════════════════════════════════════════════════════════════
def plot_align_then_memorize(df: pd.DataFrame, output_dir: Path):
    """Two-panel: (left) alignment over epochs, (right) training loss over
    epochs, to test whether there's a distinct alignment-then-memorize phase."""
    # We don't have training loss in the gradient fidelity CSV, but we can
    # show the relationship between alignment improvement rate and epoch
    df_valid = df[df["cosine_similarity"].notna()].copy()
    df_valid["weighted_cos"] = df_valid["cosine_similarity"] * df_valid["numel"]

    # Aggregate all layers per (config, epoch)
    grouped = (
        df_valid.groupby(["core_type", "seed", "epoch"])
        .agg(sum_wcos=("weighted_cos", "sum"), sum_numel=("numel", "sum"))
        .reset_index()
    )
    grouped["w_cosine"] = grouped["sum_wcos"] / grouped["sum_numel"]

    agg = (
        grouped.groupby(["core_type", "epoch"])["w_cosine"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg["se"] = agg["std"] / np.sqrt(agg["count"])

    # Compute alignment improvement rate (delta cosine / delta epoch)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.2))

    for core_type, color, label in [
        ("shunting", SHUNTING_COLOR, "Shunting"),
        ("additive", ADDITIVE_COLOR, "Additive"),
    ]:
        sub = agg[agg["core_type"] == core_type].sort_values("epoch")
        ax1.plot(sub["epoch"], sub["mean"], color=color, label=label, linewidth=1.5)
        ax1.fill_between(
            sub["epoch"],
            sub["mean"] - sub["se"],
            sub["mean"] + sub["se"],
            alpha=0.2,
            color=color,
        )

        # Compute rate of change
        epochs = sub["epoch"].values
        cosines = sub["mean"].values
        if len(epochs) > 1:
            dcosine = np.diff(cosines)
            depoch = np.diff(epochs)
            rate = dcosine / depoch
            mid_epochs = (epochs[:-1] + epochs[1:]) / 2
            ax2.plot(mid_epochs, rate, color=color, label=label, linewidth=1.5)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Weighted cosine")
    ax1.set_title("Alignment trajectory")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.legend(frameon=False)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Δcosine / Δepoch")
    ax2.set_title("Alignment improvement rate")
    ax2.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax2.legend(frameon=False)

    fig.suptitle("Align-then-memorize analysis (Refinetti et al.)", fontsize=10, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"fig_align_memorize.{ext}")
    plt.close(fig)
    print(f"  Saved fig_align_memorize.pdf/png")


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Tier-1 mechanistic analyses")
    parser.add_argument("--mode", choices=["trajectories", "cka", "all"], default="all")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("analysis/gradient_fidelity"),
        help="Directory with gradient fidelity trajectory CSVs",
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path(
            "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/mechanistic"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("trajectories", "all"):
        print("Loading gradient fidelity trajectory data...")
        df = load_trajectory_data(args.data_dir)
        print(f"  Loaded {len(df)} rows from {df['config_idx'].nunique()} configs")
        print(f"  Epochs: {sorted(df['epoch'].unique())}")
        print(f"  Core types: {df['core_type'].value_counts().to_dict()}")

        print("\n=== Analysis 1: Alignment dynamics ===")
        plot_alignment_dynamics(df, args.output_dir)

        print("\n=== Analysis 2: Layer-by-layer gradient decay ===")
        plot_layer_gradient_decay(df, args.output_dir)

        print("\n=== Analysis 3: Component-wise dynamics ===")
        plot_component_dynamics(df, args.output_dir)

        print("\n=== Analysis 4: Align-then-memorize ===")
        plot_align_then_memorize(df, args.output_dir)

    if args.mode in ("cka", "all"):
        print("\n=== Analysis 5: CKA representation similarity ===")
        plot_cka_analysis(args.sweep_root, args.output_dir)

    print("\nDone! All figures saved to", args.output_dir)


if __name__ == "__main__":
    main()

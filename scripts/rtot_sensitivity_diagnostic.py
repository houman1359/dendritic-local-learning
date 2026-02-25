#!/usr/bin/env python3
"""
R_tot Distribution Diagnostic for Dendritic Networks
=====================================================

Extracts per-compartment R_tot (= 1/g_tot) and sensitivity magnitudes
from trained model checkpoints, comparing shunting vs additive across
IE synapse counts.

Usage:
    cd /n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling
    PYTHONPATH=src:$PYTHONPATH python drafts/dendritic-local-learning/scripts/rtot_sensitivity_diagnostic.py

Outputs:
    drafts/dendritic-local-learning/data/rtot_distributions.csv
    drafts/dendritic-local-learning/figures/fig_rtot_distributions.pdf
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

# Ensure src is on path
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "src"))

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

# ---- configuration ----
SWEEP_BASE = (
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "sweep_runs/sweep_neurips_claim3_shunting_regime_robust_20260211133151"
)
OUTPUT_CSV = str(
    repo_root / "drafts" / "dendritic-local-learning" / "data" / "rtot_distributions.csv"
)
OUTPUT_FIG = str(
    repo_root
    / "drafts"
    / "dendritic-local-learning"
    / "figures"
    / "fig_rtot_distributions.pdf"
)
BATCH_SIZE = 256
DEVICE = "cpu"  # CPU is fine for diagnostics on small models


def find_completed_configs(sweep_dir):
    """Find all completed configs with checkpoints."""
    results_dir = os.path.join(sweep_dir, "results")
    configs = []
    for d in sorted(os.listdir(results_dir)):
        if not d.startswith("config_"):
            continue
        cfg_path = os.path.join(results_dir, d, "config.json")
        # Try multiple checkpoint locations
        model_path = None
        for candidate in [
            os.path.join(results_dir, d, "main_network", "local_learning_best_model.pt"),
            os.path.join(results_dir, d, "final_model.pt"),
            os.path.join(results_dir, d, "main_network", "best_model.pt"),
        ]:
            if os.path.exists(candidate):
                model_path = candidate
                break

        if os.path.exists(cfg_path) and model_path is not None:
            with open(cfg_path) as f:
                cfg = json.load(f)
            configs.append(
                {
                    "config_dir": os.path.join(results_dir, d),
                    "config": cfg,
                    "model_path": model_path,
                    "config_id": d,
                }
            )
    return configs


def load_model_from_config(cfg_dict, model_path):
    """Load a model from config dict and checkpoint."""
    from omegaconf import OmegaConf

    # Convert dict to OmegaConf for load_config compatibility
    cfg_omega = OmegaConf.create(cfg_dict)

    # Build model config
    from dendritic_modeling.config.config import Config

    config = OmegaConf.structured(Config)
    config = OmegaConf.merge(config, cfg_omega)
    config = OmegaConf.to_object(config)

    # Need to convert back to the dataclass format
    from dendritic_modeling.config import load_config as _lc
    from dendritic_modeling.config.config import (
        Config as ConfigDC,
        ModelConfig,
    )
    from dataclasses import fields

    # Simpler approach: write temp yaml and load it
    import tempfile
    import yaml

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)
        tmp_path = f.name

    try:
        _lc.cache_clear()  # Clear LRU cache
        loaded_config = _lc(tmp_path)
    finally:
        os.unlink(tmp_path)

    # We need to set the input_dim for the encoder
    dataset_name = cfg_dict["data"]["dataset_name"]
    if dataset_name == "mnist":
        input_dim = 784
    elif dataset_name in ("cifar10", "cifar-10"):
        input_dim = 3072
    elif dataset_name == "fashion_mnist":
        input_dim = 784
    else:
        input_dim = 784  # fallback

    if hasattr(loaded_config.model.encoder, "params"):
        if loaded_config.model.encoder.params is not None:
            loaded_config.model.encoder.params.input_dim = input_dim
        else:
            loaded_config.model.encoder.params = {"input_dim": input_dim}
    else:
        loaded_config.model.encoder.params = {"input_dim": input_dim}

    model, _ = initialize_model(loaded_config.model)

    # Load checkpoint
    state_dict = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model, loaded_config


def get_branch_layers(model):
    """Find all DendriticBranchLayer modules in the model."""
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
        DendriticBranchLayer,
    )

    layers = []
    for name, module in model.named_modules():
        if isinstance(module, DendriticBranchLayer):
            layers.append((name, module))
    return layers


def extract_rtot_from_model(model, x_batch):
    """
    Run forward pass and extract R_tot, voltage, and sensitivity
    for each dendritic branch layer.

    Uses the _store_diagnostics flag on DendriticBranchLayer to capture
    g_tot (denominator) directly from the forward pass, avoiding
    issues with recomputing from hook inputs.

    Returns dict of per-layer statistics.
    """
    branch_layers = get_branch_layers(model)
    if not branch_layers:
        return {}

    # Enable diagnostic storage on all branch layers
    for _, layer in branch_layers:
        layer._store_diagnostics = True

    # Storage for hook data (capture voltage from output)
    hook_data = {}

    def make_hook(layer_name, layer_module):
        def hook_fn(module, inputs, output):
            voltage = output.detach().cpu()
            is_shunting = layer_module.use_shunting

            if is_shunting and hasattr(layer_module, "_diag_g_tot"):
                g_tot = layer_module._diag_g_tot.cpu()
                R_tot = 1.0 / (g_tot + 1e-8)
            else:
                # Additive or no diagnostics available
                g_tot = torch.ones_like(voltage)
                R_tot = torch.ones_like(voltage)

            hook_data[layer_name] = {
                "R_tot": R_tot,
                "g_tot": g_tot,
                "voltage": voltage,
                "is_shunting": is_shunting,
            }

        return hook_fn

    # Register hooks
    handles = []
    for name, layer in branch_layers:
        h = layer.register_forward_hook(make_hook(name, layer))
        handles.append(h)

    # Forward pass
    with torch.no_grad():
        x_batch = x_batch.to(DEVICE)
        _ = model(x_batch)

    # Remove hooks and disable diagnostic storage
    for h in handles:
        h.remove()
    for _, layer in branch_layers:
        layer._store_diagnostics = False
        if hasattr(layer, "_diag_g_tot"):
            del layer._diag_g_tot
        if hasattr(layer, "_diag_numerator"):
            del layer._diag_numerator

    return hook_data


# Dataset cache to avoid re-downloading/re-loading for each config
_dataset_cache = {}


def get_test_batch(dataset_name, batch_size=BATCH_SIZE):
    """Load a batch of test data (cached across configs)."""
    if dataset_name in _dataset_cache:
        return _dataset_cache[dataset_name]

    from dendritic_modeling.datasets import get_unified_datasets

    task_cfg = type(
        "TaskConfig",
        (),
        {
            "dataset": dataset_name,
            "data_path": None,
            "train_valid_split": 0.1,
            "parameters": {
                "flatten": True,
                "normalize": dataset_name != "mnist",
            },
        },
    )()

    _, _, test_ds = get_unified_datasets(task_cfg=task_cfg)
    # Take a batch
    indices = list(range(min(batch_size, len(test_ds))))
    x_batch = torch.stack([test_ds[i][0] for i in indices])
    _dataset_cache[dataset_name] = x_batch
    return x_batch


def process_single_config(cfg_entry):
    """Process a single config: load model, extract R_tot stats."""
    cfg = cfg_entry["config"]
    model_type = cfg["model"]["core"]["type"]
    ie_synapses = cfg["model"]["core"]["connectivity"][
        "ie_synapses_per_branch_per_layer"
    ]
    dataset_name = cfg["data"]["dataset_name"]
    seed = cfg["experiment"]["seed"]
    strategy = cfg["training"]["main"]["strategy"]
    broadcast_mode = (
        cfg.get("training", {})
        .get("main", {})
        .get("learning_strategy_config", {})
        .get("error_broadcast_mode", "scalar")
    )

    print(
        f"  Processing {cfg_entry['config_id']}: {model_type}, "
        f"IE={ie_synapses}, {dataset_name}, seed={seed}"
    )

    try:
        model, loaded_config = load_model_from_config(cfg, cfg_entry["model_path"])
    except Exception as e:
        print(f"    ERROR loading model: {e}")
        return None

    # Get test data
    x_batch = get_test_batch(dataset_name)

    # Extract R_tot
    hook_data = extract_rtot_from_model(model, x_batch)

    if not hook_data:
        print("    WARNING: No branch layers found")
        return None

    # Aggregate statistics across all layers
    rows = []
    for layer_name, data in hook_data.items():
        R_tot = data["R_tot"]
        g_tot = data["g_tot"]
        voltage = data["voltage"]

        # Per-compartment statistics (flatten batch and neuron dims)
        R_flat = R_tot.flatten().numpy()
        g_flat = g_tot.flatten().numpy()
        v_flat = voltage.flatten().numpy()

        # Sensitivity = |x * R_tot * (E - V)| approximated as R_tot * |E - V|
        # Using E_exc = 1.0, E_inh = 0.0 as defaults
        sensitivity_exc = (R_tot * torch.abs(1.0 - voltage)).flatten().numpy()
        sensitivity_inh = (R_tot * torch.abs(0.0 - voltage)).flatten().numpy()

        row = {
            "config_id": cfg_entry["config_id"],
            "model_type": model_type,
            "ie_synapses": ie_synapses[0] if isinstance(ie_synapses, list) else ie_synapses,
            "dataset": dataset_name,
            "seed": seed,
            "strategy": strategy,
            "broadcast_mode": broadcast_mode,
            "layer_name": layer_name,
            "is_shunting": data["is_shunting"],
            # R_tot statistics
            "R_tot_mean": float(np.mean(R_flat)),
            "R_tot_std": float(np.std(R_flat)),
            "R_tot_median": float(np.median(R_flat)),
            "R_tot_q25": float(np.percentile(R_flat, 25)),
            "R_tot_q75": float(np.percentile(R_flat, 75)),
            "R_tot_max": float(np.max(R_flat)),
            "R_tot_min": float(np.min(R_flat)),
            # g_tot statistics
            "g_tot_mean": float(np.mean(g_flat)),
            "g_tot_std": float(np.std(g_flat)),
            # Voltage statistics
            "voltage_mean": float(np.mean(v_flat)),
            "voltage_std": float(np.std(v_flat)),
            # Sensitivity statistics
            "sensitivity_exc_mean": float(np.mean(sensitivity_exc)),
            "sensitivity_exc_std": float(np.std(sensitivity_exc)),
            "sensitivity_inh_mean": float(np.mean(sensitivity_inh)),
            "sensitivity_inh_std": float(np.std(sensitivity_inh)),
        }
        rows.append(row)

    # Clean up
    del model
    return rows


def generate_figure(df, output_path):
    """Generate R_tot distribution comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Filter to MNIST, per_soma broadcast for cleaner plot
    mask = (df["dataset"] == "mnist") & (df["broadcast_mode"] == "per_soma")
    if mask.sum() == 0:
        # Fall back to whatever is available
        mask = df["dataset"] == "mnist"
    if mask.sum() == 0:
        mask = pd.Series([True] * len(df))
    plot_df = df[mask].copy()

    COLOR_SHUNTING = "#18864B"
    COLOR_ADDITIVE = "#2D5DA8"

    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0))

    # --- Panel A: R_tot mean vs IE synapses ---
    ax = axes[0]
    for model_type, color, label in [
        ("dendritic_shunting", COLOR_SHUNTING, "Shunting"),
        ("dendritic_additive", COLOR_ADDITIVE, "Additive"),
    ]:
        sub = plot_df[plot_df["model_type"] == model_type]
        if len(sub) == 0:
            continue
        grouped = sub.groupby("ie_synapses").agg(
            R_mean=("R_tot_mean", "mean"),
            R_sem=("R_tot_mean", "sem"),
        )
        ax.errorbar(
            grouped.index,
            grouped["R_mean"],
            yerr=grouped["R_sem"],
            marker="o",
            markersize=4,
            color=color,
            label=label,
            capsize=2,
            linewidth=1.2,
        )
    ax.set_xlabel("IE synapses", fontsize=8)
    ax.set_ylabel(r"$R_{\mathrm{tot}}$ (mean)", fontsize=8)
    ax.set_title("A", fontsize=9, fontweight="bold", loc="left")
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, frameon=False)

    # --- Panel B: R_tot CV vs IE synapses ---
    ax = axes[1]
    for model_type, color, label in [
        ("dendritic_shunting", COLOR_SHUNTING, "Shunting"),
        ("dendritic_additive", COLOR_ADDITIVE, "Additive"),
    ]:
        sub = plot_df[plot_df["model_type"] == model_type]
        if len(sub) == 0:
            continue
        # CV = std / mean
        sub = sub.copy()
        sub["R_cv"] = sub["R_tot_std"] / (sub["R_tot_mean"] + 1e-8)
        grouped = sub.groupby("ie_synapses").agg(
            cv_mean=("R_cv", "mean"),
            cv_sem=("R_cv", "sem"),
        )
        ax.errorbar(
            grouped.index,
            grouped["cv_mean"],
            yerr=grouped["cv_sem"],
            marker="s",
            markersize=4,
            color=color,
            label=label,
            capsize=2,
            linewidth=1.2,
        )
    ax.set_xlabel("IE synapses", fontsize=8)
    ax.set_ylabel(r"$R_{\mathrm{tot}}$ CV", fontsize=8)
    ax.set_title("B", fontsize=9, fontweight="bold", loc="left")
    ax.tick_params(labelsize=7)

    # --- Panel C: Sensitivity magnitude vs IE synapses ---
    ax = axes[2]
    for model_type, color, label in [
        ("dendritic_shunting", COLOR_SHUNTING, "Shunting"),
        ("dendritic_additive", COLOR_ADDITIVE, "Additive"),
    ]:
        sub = plot_df[plot_df["model_type"] == model_type]
        if len(sub) == 0:
            continue
        grouped = sub.groupby("ie_synapses").agg(
            sens_mean=("sensitivity_exc_mean", "mean"),
            sens_sem=("sensitivity_exc_mean", "sem"),
        )
        ax.errorbar(
            grouped.index,
            grouped["sens_mean"],
            yerr=grouped["sens_sem"],
            marker="^",
            markersize=4,
            color=color,
            label=label,
            capsize=2,
            linewidth=1.2,
        )
    ax.set_xlabel("IE synapses", fontsize=8)
    ax.set_ylabel(r"$|x \cdot R_{\mathrm{tot}} \cdot (E-V)|$", fontsize=8)
    ax.set_title("C", fontsize=9, fontweight="bold", loc="left")
    ax.tick_params(labelsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to {output_path}")
    plt.close(fig)


def main():
    import gc

    print("R_tot Distribution Diagnostic")
    print("=" * 50)

    # Find completed configs
    configs = find_completed_configs(SWEEP_BASE)
    print(f"Found {len(configs)} completed configs in sweep")

    if not configs:
        print("ERROR: No completed configs found!")
        sys.exit(1)

    # Filter to MNIST only (CIFAR-10 adds memory pressure and isn't needed for diagnostic)
    configs = [
        c for c in configs if c["config"]["data"]["dataset_name"] == "mnist"
    ]
    print(f"Filtered to {len(configs)} MNIST configs")

    # Process all configs with explicit memory cleanup
    all_rows = []
    for i, cfg_entry in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}]")
        rows = process_single_config(cfg_entry)
        if rows:
            all_rows.extend(rows)
        # Explicit garbage collection to prevent OOM
        gc.collect()

    if not all_rows:
        print("ERROR: No data extracted!")
        sys.exit(1)

    # Save to CSV
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")

    # Print summary
    print("\n--- Summary ---")
    summary = df.groupby(["model_type", "ie_synapses"]).agg(
        R_tot_mean=("R_tot_mean", "mean"),
        R_tot_std=("R_tot_std", "mean"),
        voltage_std=("voltage_std", "mean"),
        sens_exc=("sensitivity_exc_mean", "mean"),
        n_configs=("config_id", "nunique"),
    )
    print(summary.to_string())

    # Generate figure
    try:
        generate_figure(df, OUTPUT_FIG)
    except Exception as e:
        print(f"WARNING: Figure generation failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

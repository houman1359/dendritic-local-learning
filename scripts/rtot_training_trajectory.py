#!/usr/bin/env python3
"""
R_tot Training Trajectory Diagnostic
=====================================

Loads model checkpoints at multiple training epochs and records
per-layer R_tot statistics (mean, std, min, max), voltage statistics,
and driving force statistics. Compares shunting vs additive across
training to demonstrate the self-normalizing property of conductance-
based dendritic computation.

Usage:
    cd /n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling
    PYTHONPATH=src:$PYTHONPATH python \
        drafts/dendritic-local-learning/scripts/rtot_training_trajectory.py

Outputs:
    drafts/dendritic-local-learning/data/rtot_trajectory.csv
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root / "src"))

from dendritic_modeling.config import load_config
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

# ---- configuration ----
# Sweep with both shunting and additive, with checkpoints saved per epoch
SWEEP_DIRS = [
    # Depth scaling v2 has both standard and local_ca with checkpoints
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "sweep_runs/sweep_neurips_depth_scaling_v2_20260222152754",
]

OUTPUT_CSV = str(
    repo_root / "drafts" / "dendritic-local-learning" / "data" / "rtot_trajectory.csv"
)
BATCH_SIZE = 256
DEVICE = "cpu"


def find_configs_with_checkpoints(sweep_dir):
    """Find configs that have per-epoch checkpoints."""
    results_dir = os.path.join(sweep_dir, "results")
    if not os.path.isdir(results_dir):
        return []

    configs = []
    for d in sorted(os.listdir(results_dir)):
        if not d.startswith("config_"):
            continue
        cfg_path = os.path.join(results_dir, d, "config.json")
        if not os.path.exists(cfg_path):
            continue

        # Look for checkpoint files
        checkpoint_dir = os.path.join(results_dir, d, "main_network")
        if not os.path.isdir(checkpoint_dir):
            continue

        # Find all model checkpoint files
        checkpoints = {}
        for f in os.listdir(checkpoint_dir):
            if f.endswith(".pt") and ("best" in f or "epoch" in f or "final" in f):
                checkpoints[f] = os.path.join(checkpoint_dir, f)

        # Also check for the final/best model
        for candidate_name in [
            "local_learning_best_model.pt",
            "best_model.pt",
            "final_model.pt",
        ]:
            candidate = os.path.join(checkpoint_dir, candidate_name)
            if os.path.exists(candidate):
                checkpoints[candidate_name] = candidate

        if not checkpoints:
            continue

        with open(cfg_path) as f:
            cfg = json.load(f)

        configs.append(
            {
                "config_dir": os.path.join(results_dir, d),
                "config": cfg,
                "checkpoints": checkpoints,
            }
        )

    return configs


def get_dataset_batch(cfg, batch_size=256):
    """Load a fixed batch from the dataset for consistent evaluation."""
    from dendritic_modeling.datasets import get_unified_datasets

    dataset_name = cfg.get("data", {}).get("dataset_name", "mnist")
    task_cfg = type(
        "TaskConfig",
        (),
        {
            "dataset": dataset_name,
            "data_path": None,
            "train_valid_split": 0.8,
            "parameters": {"flatten": True, "normalize": False},
        },
    )()
    train_ds, _, _ = get_unified_datasets(task_cfg)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=False
    )
    x_batch, y_batch = next(iter(loader))
    return x_batch, y_batch


def extract_rtot_stats(model, x_batch, device="cpu"):
    """Forward pass and extract per-layer R_tot, voltage, driving force stats."""
    model.eval()
    model.to(device)
    x_batch = x_batch.to(device)

    # Enable diagnostic storage
    branch_layers = [
        m for m in model.modules() if isinstance(m, DendriticBranchLayer)
    ]
    for bl in branch_layers:
        bl._store_diagnostics = True

    with torch.no_grad():
        _ = model(x_batch)

    rows = []
    for layer_idx, bl in enumerate(branch_layers):
        is_shunting = bl.use_shunting
        g_tot = getattr(bl, "_diag_g_tot", None)
        numerator = getattr(bl, "_diag_numerator", None)

        if g_tot is not None and torch.is_tensor(g_tot):
            R_tot = 1.0 / (g_tot + 1e-8)
            voltage = numerator / (g_tot + 1e-8) if numerator is not None else None

            row = {
                "layer": layer_idx,
                "is_shunting": is_shunting,
                "R_tot_mean": float(R_tot.mean()),
                "R_tot_std": float(R_tot.std()),
                "R_tot_min": float(R_tot.min()),
                "R_tot_max": float(R_tot.max()),
                "g_tot_mean": float(g_tot.mean()),
                "g_tot_std": float(g_tot.std()),
            }

            if voltage is not None:
                e_rev = 1.0  # default excitatory reversal
                driving = e_rev - voltage
                row.update(
                    {
                        "v_mean": float(voltage.mean()),
                        "v_std": float(voltage.std()),
                        "driving_mean": float(driving.mean()),
                        "driving_std": float(driving.std()),
                    }
                )
            rows.append(row)
        else:
            # Additive mode — R_tot = 1 always
            rows.append(
                {
                    "layer": layer_idx,
                    "is_shunting": False,
                    "R_tot_mean": 1.0,
                    "R_tot_std": 0.0,
                    "R_tot_min": 1.0,
                    "R_tot_max": 1.0,
                    "g_tot_mean": 1.0,
                    "g_tot_std": 0.0,
                    "v_mean": float("nan"),
                    "v_std": float("nan"),
                    "driving_mean": float("nan"),
                    "driving_std": float("nan"),
                }
            )

    # Disable diagnostic storage
    for bl in branch_layers:
        bl._store_diagnostics = False

    return rows


def main():
    all_rows = []

    for sweep_dir in SWEEP_DIRS:
        print(f"Scanning {sweep_dir} ...")
        configs = find_configs_with_checkpoints(sweep_dir)
        print(f"  Found {len(configs)} configs with checkpoints")

        for entry in configs:
            cfg = entry["config"]
            core_type = cfg.get("model", {}).get("core", {}).get("type", "unknown")
            strategy = (
                cfg.get("training", {}).get("main", {}).get("strategy", "unknown")
            )
            dataset = cfg.get("data", {}).get("dataset_name", "unknown")
            seed = cfg.get("experiment", {}).get("seed", 0)

            print(f"  Processing {core_type}/{strategy}/seed={seed} ...")

            # Load dataset batch
            try:
                x_batch, y_batch = get_dataset_batch(cfg, BATCH_SIZE)
            except Exception as e:
                print(f"    Skipping: dataset load failed: {e}")
                continue

            # Load the best/final model checkpoint
            best_ckpt = None
            for name in [
                "local_learning_best_model.pt",
                "best_model.pt",
                "final_model.pt",
            ]:
                if name in entry["checkpoints"]:
                    best_ckpt = entry["checkpoints"][name]
                    break
            if best_ckpt is None:
                best_ckpt = list(entry["checkpoints"].values())[0]

            try:
                config_obj = load_config(
                    os.path.join(entry["config_dir"], "config.json")
                )
                model = initialize_model(config_obj.model, config_obj.data)
                state_dict = torch.load(best_ckpt, map_location=DEVICE)
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"    Skipping: model load failed: {e}")
                continue

            stats = extract_rtot_stats(model, x_batch, DEVICE)
            for row in stats:
                row.update(
                    {
                        "core_type": core_type,
                        "strategy": strategy,
                        "dataset": dataset,
                        "seed": seed,
                        "checkpoint": "final",
                    }
                )
                all_rows.append(row)

    if all_rows:
        df = pd.DataFrame(all_rows)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(df)} rows to {OUTPUT_CSV}")
        print("\nSummary by core_type:")
        summary = df.groupby("core_type")[
            ["R_tot_mean", "R_tot_std", "g_tot_mean", "v_std"]
        ].mean()
        print(summary.to_string())
    else:
        print("No data extracted!")


if __name__ == "__main__":
    main()

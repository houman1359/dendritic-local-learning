#!/usr/bin/env python3
"""Aggregate all Phase B experiment results into unified CSVs.

Produces:
  fa_dfa_results.csv          — FA/DFA baseline comparison
  cifar10_results.csv         — CIFAR-10 local learning
  low_bandwidth_results.csv   — Low-bandwidth broadcast modes
  additive_norm_results.csv   — Additive + normalization control
  unified_revision_results.csv — All results merged

Usage:
    python drafts/dendritic-local-learning/data/aggregate_all_revision.py
"""

import json
import os
import re

import pandas as pd
import yaml

SWEEP_BASE = (
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_last_epoch_acc(results_dir):
    """Extract test/train/valid accuracy from the last epoch JSON."""
    perf_dir = os.path.join(results_dir, "performance", "epochs")
    if not os.path.isdir(perf_dir):
        return None
    epochs = []
    for f in os.listdir(perf_dir):
        m = re.match(r"epoch(\d+)\.json", f)
        if m:
            epochs.append(int(m.group(1)))
    if not epochs:
        return None
    last = max(epochs)
    path = os.path.join(perf_dir, f"epoch{last}.json")
    with open(path) as fh:
        data = json.load(fh)
    return {
        "last_epoch": last,
        "test_accuracy": data.get("accuracy", {}).get("test"),
        "train_accuracy": data.get("accuracy", {}).get("train"),
        "valid_accuracy": data.get("accuracy", {}).get("valid"),
    }


def aggregate_sweep(sweep_name, extract_fn):
    """Generic sweep aggregator. extract_fn(config_yaml) -> dict of extra fields."""
    sweep_dir = os.path.join(SWEEP_BASE, sweep_name)
    configs_dir = os.path.join(sweep_dir, "configs")
    results_dir = os.path.join(sweep_dir, "results")
    rows = []
    for cfg_file in sorted(os.listdir(configs_dir)):
        if not cfg_file.startswith("unified_config_") or not cfg_file.endswith(".yaml"):
            continue
        cfg_num = cfg_file.replace("unified_config_", "").replace(".yaml", "")
        cfg_path = os.path.join(configs_dir, cfg_file)
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        extra = extract_fn(cfg)
        res_dir = os.path.join(results_dir, f"config_{cfg_num}")
        acc = get_last_epoch_acc(res_dir)
        if acc is None:
            continue
        row = {"config_id": f"config_{cfg_num}", "sweep": sweep_name}
        row.update(extra)
        row.update(acc)
        rows.append(row)
    return pd.DataFrame(rows)


def _identify_model(cfg):
    """Identify model type from config using core.type field."""
    core = cfg.get("model", {}).get("core", {})
    core_type = core.get("type", "unknown")
    if core_type in ("dendritic_shunting", "dendritic_additive", "point_mlp"):
        return core_type
    # Fallback for older configs
    morph = core.get("morphology", {})
    wt = morph.get("weight_transform", "none")
    if core_type == "point_mlp":
        return "point_mlp"
    elif wt == "softplus":
        return "dendritic_shunting"
    else:
        return "dendritic_additive"


def extract_fa_dfa(cfg):
    return {
        "model": _identify_model(cfg),
        "strategy": cfg["training"]["main"]["strategy"],
        "seed": cfg["experiment"]["seed"],
        "dataset": cfg["data"]["dataset_name"],
    }


def extract_cifar10(cfg):
    return {
        "model": _identify_model(cfg),
        "strategy": cfg["training"]["main"]["strategy"],
        "seed": cfg["experiment"]["seed"],
        "dataset": cfg["data"]["dataset_name"],
    }


def extract_low_bandwidth(cfg):
    lsc = cfg.get("training", {}).get("main", {}).get("learning_strategy_config", {})
    return {
        "model": "dendritic_shunting",
        "strategy": cfg["training"]["main"]["strategy"],
        "seed": cfg["experiment"]["seed"],
        "dataset": cfg["data"]["dataset_name"],
        "broadcast_bandwidth": lsc.get("broadcast_bandwidth", "full"),
        "broadcast_bits": lsc.get("broadcast_bits", 8),
        "broadcast_topk_fraction": lsc.get("broadcast_topk_fraction", 0.3),
    }


def extract_additive_norm(cfg):
    core = cfg.get("model", {}).get("core", {})
    morph = core.get("morphology", {})
    norm = morph.get("use_additive_normalization", False)
    return {
        "model": "dendritic_additive",
        "strategy": cfg["training"]["main"]["strategy"],
        "seed": cfg["experiment"]["seed"],
        "dataset": cfg["data"]["dataset_name"],
        "use_additive_normalization": norm,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_dfs = []

    # FA/DFA v2
    print("Aggregating FA/DFA v2...")
    fa_dfa = aggregate_sweep(
        "sweep_neurips_fa_dfa_v2_20260225132701", extract_fa_dfa
    )
    fa_dfa.to_csv(os.path.join(OUT_DIR, "fa_dfa_results.csv"), index=False)
    print(f"  {len(fa_dfa)} rows")
    all_dfs.append(fa_dfa)

    # CIFAR-10
    print("Aggregating CIFAR-10...")
    cifar = aggregate_sweep(
        "sweep_neurips_cifar10_local_20260225132739", extract_cifar10
    )
    cifar.to_csv(os.path.join(OUT_DIR, "cifar10_results.csv"), index=False)
    print(f"  {len(cifar)} rows")
    all_dfs.append(cifar)

    # Low-bandwidth part 1
    print("Aggregating low-bandwidth part 1...")
    lbw1 = aggregate_sweep(
        "sweep_neurips_low_bandwidth_20260225141656", extract_low_bandwidth
    )
    # Low-bandwidth part 2 (bits)
    print("Aggregating low-bandwidth part 2 (bits)...")
    lbw2 = aggregate_sweep(
        "sweep_neurips_low_bandwidth_bits_20260225141713", extract_low_bandwidth
    )
    lbw = pd.concat([lbw1, lbw2], ignore_index=True)
    lbw.to_csv(os.path.join(OUT_DIR, "low_bandwidth_results.csv"), index=False)
    print(f"  {len(lbw)} rows")
    all_dfs.append(lbw)

    # Additive normalization
    print("Aggregating additive normalization...")
    anorm = aggregate_sweep(
        "sweep_neurips_additive_norm_20260225142220", extract_additive_norm
    )
    anorm.to_csv(os.path.join(OUT_DIR, "additive_norm_results.csv"), index=False)
    print(f"  {len(anorm)} rows")
    all_dfs.append(anorm)

    # Unified
    unified = pd.concat(all_dfs, ignore_index=True)
    unified.to_csv(os.path.join(OUT_DIR, "unified_revision_results.csv"), index=False)
    print(f"\nUnified: {len(unified)} total rows")

    # Print summary tables
    print("\n=== FA/DFA SUMMARY ===")
    for (model, strat), g in fa_dfa.groupby(["model", "strategy"]):
        m, s = g["test_accuracy"].mean() * 100, g["test_accuracy"].std() * 100
        print(f"  {model:25s} {strat:10s}: {m:.1f}% +/- {s:.1f}% (n={len(g)})")

    print("\n=== CIFAR-10 SUMMARY ===")
    for (model, strat), g in cifar.groupby(["model", "strategy"]):
        m, s = g["test_accuracy"].mean() * 100, g["test_accuracy"].std() * 100
        print(f"  {model:25s} {strat:10s}: {m:.1f}% +/- {s:.1f}% (n={len(g)})")

    print("\n=== LOW-BANDWIDTH SUMMARY ===")
    for (bw, bits), g in lbw.groupby(["broadcast_bandwidth", "broadcast_bits"]):
        m, s = g["test_accuracy"].mean() * 100, g["test_accuracy"].std() * 100
        label = f"{bw}" if bw != "quantized" else f"{bw}_{bits}bit"
        print(f"  {label:25s}: {m:.1f}% +/- {s:.1f}% (n={len(g)})")

    print("\n=== ADDITIVE NORMALIZATION SUMMARY ===")
    for (norm, strat), g in anorm.groupby(["use_additive_normalization", "strategy"]):
        m, s = g["test_accuracy"].mean() * 100, g["test_accuracy"].std() * 100
        label = f"norm={norm} {strat}"
        print(f"  {label:30s}: {m:.1f}% +/- {s:.1f}% (n={len(g)})")


if __name__ == "__main__":
    main()

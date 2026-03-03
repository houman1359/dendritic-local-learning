"""
Extract weight distribution sigma (lognormal) from depth_scaling_v2 sweep.

Sweep varies branch depth: [9], [3,3], [3,3,3], [3,3,3,3]
across {standard, local_ca} x {dendritic_shunting, dendritic_additive}.

For each model:
  - Load config to get strategy, core_type, branch_factors, connectivity
  - Load final_model.pt checkpoint
  - Extract LAYER 0 excitatory weights (branch_excitation.pre_w in branch_layers.0)
  - Apply softplus: w = log(1 + exp(pre_w))
  - Apply top-K mask (K from ee_synapses_per_branch_per_layer[0])
  - Fit lognormal sigma = std(log(active_positive_weights))
  - Same for inhibitory weights (branch_inhibition, K from ie_synapses)

Aggregate across seeds, print table, save JSON.
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch

SWEEP_DIR = (
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "sweep_runs/sweep_neurips_depth_scaling_v2_20260225184411"
)
RESULTS_DIR = os.path.join(SWEEP_DIR, "results")
OUTPUT_PATH = (
    "/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/"
    "drafts/dendritic-local-learning/data/weight_distributions/sigma_vs_depth.json"
)

NUM_CONFIGS = 80


def softplus(x):
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def extract_topk_sigma(pre_w_tensor, k_per_branch):
    """
    Apply softplus, then top-K mask per row (branch), then compute
    lognormal sigma = std(log(active positive weights)).
    """
    pre_w = pre_w_tensor.detach().cpu().numpy()
    w = softplus(pre_w)  # shape: (n_branches, n_inputs)

    n_branches, n_inputs = w.shape
    k = min(k_per_branch, n_inputs)

    active_weights = []
    for i in range(n_branches):
        row = w[i]
        topk_idx = np.argpartition(row, -k)[-k:]
        active_weights.append(row[topk_idx])

    active_weights = np.concatenate(active_weights)

    mask = active_weights > 1e-10
    active_positive = active_weights[mask]

    if len(active_positive) < 2:
        return float("nan"), 0

    log_w = np.log(active_positive)
    sigma = float(np.std(log_w))
    return sigma, len(active_positive)


def process_config(config_idx):
    """Process a single config directory, return extracted info or None."""
    config_dir = os.path.join(RESULTS_DIR, f"config_{config_idx}")
    config_path = os.path.join(config_dir, "config.json")
    model_path = os.path.join(config_dir, "final_model.pt")

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"  config_{config_idx}: missing config.json or final_model.pt, skipping")
        return None

    with open(config_path) as f:
        config = json.load(f)

    strategy = config["training"]["main"]["strategy"]
    core_type = config["model"]["core"]["type"]
    branch_factors = tuple(
        config["model"]["core"]["architecture"]["excitatory_branch_factors"]
    )
    ee_synapses = config["model"]["core"]["connectivity"][
        "ee_synapses_per_branch_per_layer"
    ]
    ie_synapses = config["model"]["core"]["connectivity"][
        "ie_synapses_per_branch_per_layer"
    ]
    seed = config["experiment"]["seed"]

    k_exc = ee_synapses[0]
    k_inh = ie_synapses[0]

    ckpt = torch.load(model_path, map_location="cpu")

    exc_key = None
    inh_key = None
    for key in ckpt.keys():
        if "branch_layers.0.branch_excitation.pre_w" in key and "layers.0." in key:
            exc_key = key
        if "branch_layers.0.branch_inhibition.pre_w" in key and "layers.0." in key:
            inh_key = key

    result = {
        "config_idx": config_idx,
        "strategy": strategy,
        "core_type": core_type,
        "branch_factors": branch_factors,
        "seed": seed,
    }

    if exc_key is not None:
        sigma_exc, n_exc = extract_topk_sigma(ckpt[exc_key], k_exc)
        result["sigma_exc"] = sigma_exc
        result["n_active_exc"] = n_exc
    else:
        print(f"  config_{config_idx}: no excitatory key found")
        result["sigma_exc"] = float("nan")
        result["n_active_exc"] = 0

    if inh_key is not None and k_inh > 0:
        sigma_inh, n_inh = extract_topk_sigma(ckpt[inh_key], k_inh)
        result["sigma_inh"] = sigma_inh
        result["n_active_inh"] = n_inh
    else:
        result["sigma_inh"] = float("nan")
        result["n_active_inh"] = 0

    return result


def main():
    print(f"Processing {NUM_CONFIGS} configs from: {RESULTS_DIR}")
    print()

    all_results = []
    for ci in range(NUM_CONFIGS):
        res = process_config(ci)
        if res is not None:
            all_results.append(res)
            if (ci + 1) % 20 == 0:
                print(f"  Processed {ci + 1}/{NUM_CONFIGS} configs...")

    print(f"
Total valid results: {len(all_results)}")

    groups = defaultdict(list)
    for r in all_results:
        key = (r["strategy"], r["core_type"], r["branch_factors"])
        groups[key].append(r)

    branch_factor_options = [(9,), (3, 3), (3, 3, 3), (3, 3, 3, 3)]
    strategy_core_options = [
        ("standard", "dendritic_shunting"),
        ("standard", "dendritic_additive"),
        ("local_ca", "dendritic_shunting"),
        ("local_ca", "dendritic_additive"),
    ]

    print("
" + "=" * 100)
    print("EXCITATORY weight sigma (lognormal std of log(w))")
    print("=" * 100)

    col_headers = [f"{s}/{c.replace(chr(39)+chr(39), chr(39)).replace('dendritic_', '')}" for s, c in strategy_core_options]
    header = f"{'branch_factors':<18}" + "".join(f"{h:>22}" for h in col_headers)
    print(header)
    print("-" * 100)

    summary = {}
    for bf in branch_factor_options:
        bf_str = str(list(bf))
        row = f"{bf_str:<18}"
        for strat, ct in strategy_core_options:
            key = (strat, ct, bf)
            entries = groups.get(key, [])
            sigmas = [e["sigma_exc"] for e in entries if not np.isnan(e["sigma_exc"])]
            if sigmas:
                mean_s = np.mean(sigmas)
                std_s = np.std(sigmas)
                row += f"{mean_s:>10.4f} +/- {std_s:<7.4f}"
                summary_key = f"{strat}_{ct}_{bf_str}"
                summary[summary_key] = {
                    "strategy": strat,
                    "core_type": ct,
                    "branch_factors": list(bf),
                    "sigma_exc_mean": float(mean_s),
                    "sigma_exc_std": float(std_s),
                    "sigma_exc_values": [float(s) for s in sigmas],
                    "n_seeds": len(sigmas),
                }
            else:
                row += f"{'N/A':>22}"
        print(row)

    print("
" + "=" * 100)
    print("INHIBITORY weight sigma (lognormal std of log(w))")
    print("=" * 100)

    header = f"{'branch_factors':<18}" + "".join(f"{h:>22}" for h in col_headers)
    print(header)
    print("-" * 100)

    for bf in branch_factor_options:
        bf_str = str(list(bf))
        row = f"{bf_str:<18}"
        for strat, ct in strategy_core_options:
            key = (strat, ct, bf)
            entries = groups.get(key, [])
            sigmas = [e["sigma_inh"] for e in entries if not np.isnan(e["sigma_inh"])]
            if sigmas:
                mean_s = np.mean(sigmas)
                std_s = np.std(sigmas)
                row += f"{mean_s:>10.4f} +/- {std_s:<7.4f}"
                summary_key = f"{strat}_{ct}_{bf_str}"
                if summary_key in summary:
                    summary[summary_key]["sigma_inh_mean"] = float(mean_s)
                    summary[summary_key]["sigma_inh_std"] = float(std_s)
                    summary[summary_key]["sigma_inh_values"] = [float(s) for s in sigmas]
                else:
                    summary[summary_key] = {
                        "strategy": strat,
                        "core_type": ct,
                        "branch_factors": list(bf),
                        "sigma_inh_mean": float(mean_s),
                        "sigma_inh_std": float(std_s),
                        "sigma_inh_values": [float(s) for s in sigmas],
                        "n_seeds": len(sigmas),
                    }
            else:
                row += f"{'N/A':>22}"
        print(row)

    output = {
        "sweep_dir": SWEEP_DIR,
        "description": (
            "Lognormal sigma (std of log(w)) for layer-0 weights after "
            "softplus + top-K masking, from depth_scaling_v2 sweep"
        ),
        "branch_factor_options": [list(bf) for bf in branch_factor_options],
        "strategy_options": ["standard", "local_ca"],
        "core_type_options": ["dendritic_shunting", "dendritic_additive"],
        "conditions": summary,
        "per_model_results": [
            {
                "config_idx": r["config_idx"],
                "strategy": r["strategy"],
                "core_type": r["core_type"],
                "branch_factors": list(r["branch_factors"]),
                "seed": r["seed"],
                "sigma_exc": r["sigma_exc"] if not np.isnan(r["sigma_exc"]) else None,
                "sigma_inh": r["sigma_inh"] if not np.isnan(r["sigma_inh"]) else None,
            }
            for r in all_results
        ],
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"
Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

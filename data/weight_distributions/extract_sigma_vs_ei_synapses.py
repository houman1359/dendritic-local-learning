"""
Extract weight distribution sigma (lognormal width) from the ei_grid_pilot sweep.

Varies: ee_synapses [20, 40, 80] x ie_synapses [0, 2, 5, 10, 20, 30, 40]
        x core_type {dendritic_additive, dendritic_shunting}

For each model:
  - Load config.json and final_model.pt
  - Extract excitatory weights (branch_excitation.pre_w)
  - Apply softplus: w = log(1 + exp(pre_w))
  - Apply top-K mask per row (K = ee_synapses), keeping K largest pre_w values
  - Fit lognormal to positive active weights -> sigma
  - Same for inhibitory (branch_inhibition.pre_w, K = ie_synapses)
"""

import json
import os
from collections import defaultdict

import numpy as np
import torch


SWEEP_DIRS = [
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "sweep_runs/sweep_neurips_claimA_ei_grid_pilot_20260220125929",
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/"
    "sweep_runs/sweep_neurips_claimA_ei_grid_pilot_20260223180734",
]

OUTPUT_PATH = (
    "/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/"
    "drafts/dendritic-local-learning/data/weight_distributions/"
    "sigma_vs_ei_synapses.json"
)


def softplus(x):
    """Numerically stable softplus."""
    return np.where(x > 20, x, np.log1p(np.exp(x)))


def extract_sigma(state_dict, key_pattern, K):
    """Extract lognormal sigma for weights matching key_pattern with top-K mask."""
    all_w = []
    for key, tensor in state_dict.items():
        if key_pattern in key and "pre_w" in key:
            pre_w = tensor.numpy()
            if K <= 0 or K > pre_w.shape[1]:
                continue
            if K < pre_w.shape[1]:
                topk_idx = np.argpartition(pre_w, -K, axis=1)[:, -K:]
                mask = np.zeros_like(pre_w, dtype=bool)
                mask[np.arange(pre_w.shape[0])[:, None], topk_idx] = True
            else:
                mask = np.ones_like(pre_w, dtype=bool)
            w = softplus(pre_w)
            active = w[mask]
            active = active[active > 1e-8]
            all_w.append(active)
    if not all_w:
        return float("nan")
    all_w = np.concatenate(all_w)
    if len(all_w) < 10:
        return float("nan")
    return float(np.std(np.log(all_w)))


def process_sweep(sweep_dir):
    """Process all models in a sweep directory."""
    results_dir = os.path.join(sweep_dir, "results")
    records = []
    for cn in sorted(os.listdir(results_dir)):
        cp = os.path.join(results_dir, cn, "config.json")
        mp = os.path.join(results_dir, cn, "final_model.pt")
        if not os.path.exists(cp) or not os.path.exists(mp):
            continue
        with open(cp) as f:
            cfg = json.load(f)
        ct = cfg["model"]["core"]["type"]
        ee = cfg["model"]["core"]["connectivity"]["ee_synapses_per_branch_per_layer"][0]
        ie = cfg["model"]["core"]["connectivity"]["ie_synapses_per_branch_per_layer"][0]
        strat = cfg["training"]["main"]["strategy"]
        sd = torch.load(mp, map_location="cpu")
        se = extract_sigma(sd, "branch_excitation", ee)
        si = extract_sigma(sd, "branch_inhibition", ie) if ie > 0 else float("nan")
        records.append(dict(
            sweep=os.path.basename(sweep_dir), config=cn, core_type=ct,
            ee_synapses=ee, ie_synapses=ie, strategy=strat,
            sigma_exc=se, sigma_inh=si,
        ))
        inh_s = "{:.4f}".format(si) if not np.isnan(si) else "NaN"
        print("  {}: {} ee={} ie={} sigma_exc={:.4f} sigma_inh={}".format(
            cn, ct, ee, ie, se, inh_s))
    return records


def main():
    all_records = []
    for sweep_dir in SWEEP_DIRS:
        print("\nProcessing: {}".format(os.path.basename(sweep_dir)))
        recs = process_sweep(sweep_dir)
        all_records.extend(recs)
        print("  -> {} models".format(len(recs)))

    print("\nTotal: {}".format(len(all_records)))

    # Aggregate
    g_exc = defaultdict(list)
    g_inh = defaultdict(list)
    for r in all_records:
        k = (r["core_type"], r["ee_synapses"], r["ie_synapses"])
        if not np.isnan(r["sigma_exc"]):
            g_exc[k].append(r["sigma_exc"])
        if not np.isnan(r["sigma_inh"]):
            g_inh[k].append(r["sigma_inh"])

    cts = sorted(set(k[0] for k in g_exc))
    ees = sorted(set(k[1] for k in g_exc))
    ies = sorted(set(k[2] for k in g_exc))

    # Table 1: Excitatory
    print("\n" + "=" * 120)
    print("TABLE 1: Excitatory Weight sigma -- mean +/- std")
    print("=" * 120)
    for ct in cts:
        print("\n  Core: {}".format(ct))
        hdr = "  {:>8s}".format("ee/ie")
        for ie in ies:
            hdr += "  {:>14d}".format(ie)
        print(hdr)
        print("  " + "-" * (10 + 16 * len(ies)))
        for ee in ees:
            row = "  {:>8d}".format(ee)
            for ie in ies:
                vals = g_exc.get((ct, ee, ie), [])
                if vals:
                    row += "  {:.4f}+/-{:.4f}".format(np.mean(vals), np.std(vals))
                else:
                    row += "  {:>14s}".format("N/A")
            print(row)

    # Table 2: Inhibitory
    print("\n" + "=" * 120)
    print("TABLE 2: Inhibitory Weight sigma -- mean +/- std")
    print("=" * 120)
    ies_nz = [v for v in ies if v > 0]
    for ct in cts:
        print("\n  Core: {}".format(ct))
        hdr = "  {:>8s}".format("ee/ie")
        for ie in ies_nz:
            hdr += "  {:>14d}".format(ie)
        print(hdr)
        print("  " + "-" * (10 + 16 * len(ies_nz)))
        for ee in ees:
            row = "  {:>8d}".format(ee)
            for ie in ies_nz:
                vals = g_inh.get((ct, ee, ie), [])
                if vals:
                    row += "  {:.4f}+/-{:.4f}".format(np.mean(vals), np.std(vals))
                else:
                    row += "  {:>14s}".format("N/A")
            print(row)

    # Save JSON
    output = {
        "description": "Lognormal sigma of exc/inh weight distributions from ei_grid_pilot sweep",
        "sweeps": [os.path.basename(d) for d in SWEEP_DIRS],
        "dimensions": {"core_types": cts, "ee_synapses": ees, "ie_synapses": ies},
        "excitatory_sigma": {},
        "inhibitory_sigma": {},
        "individual_records": [],
    }
    for k, vals in sorted(g_exc.items()):
        ct, ee, ie = k
        output["excitatory_sigma"]["{}_ee{}_ie{}".format(ct, ee, ie)] = dict(
            core_type=ct, ee_synapses=ee, ie_synapses=ie, values=vals,
            mean=float(np.mean(vals)), std=float(np.std(vals)), n_seeds=len(vals))
    for k, vals in sorted(g_inh.items()):
        ct, ee, ie = k
        output["inhibitory_sigma"]["{}_ee{}_ie{}".format(ct, ee, ie)] = dict(
            core_type=ct, ee_synapses=ee, ie_synapses=ie, values=vals,
            mean=float(np.mean(vals)), std=float(np.std(vals)), n_seeds=len(vals))
    for r in all_records:
        output["individual_records"].append(
            {kk: (None if isinstance(vv, float) and np.isnan(vv) else vv)
             for kk, vv in r.items()})

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print("\nSaved to: {}".format(OUTPUT_PATH))


if __name__ == "__main__":
    main()

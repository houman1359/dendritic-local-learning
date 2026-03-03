"""
Extract excitatory and inhibitory synapse size distributions from the
MICrONS mm3 condensed connectome (Zenodo v1181 HDF5 file).

Produces:
  - Fitted lognormal parameters for E→E, I→E, E→I, I→I quadrants
  - Synapse count per connection distributions
  - Reference data file for comparison with model weight distributions

Requires: h5py, numpy, scipy, matplotlib
Data source: https://zenodo.org/records/13849415
"""

import argparse
import json
import logging
import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_microns_connectome(h5_path: str) -> dict:
    """Load the condensed connectome from the MICrONS HDF5 file.

    Returns dict with arrays for pre/post neuron IDs, synapse counts,
    mean/sum sizes, and cell-type information.
    """
    logger.info(f"Loading {h5_path}...")

    with h5py.File(h5_path, "r") as f:
        logger.info(f"Top-level keys: {list(f.keys())}")

        # Explore the structure
        def _print_structure(name, obj):
            if isinstance(obj, h5py.Dataset):
                logger.info(f"  Dataset: {name}, shape={obj.shape}, dtype={obj.dtype}")

        f.visititems(_print_structure)

    # Now load the relevant data
    data = {}
    with h5py.File(h5_path, "r") as f:
        # Try different possible structures
        # The condensed connectome should have edges with pre/post IDs and synapse properties

        # Check for 'condensed' group
        for group_name in ["condensed", "full", "edges", "connectivity"]:
            if group_name in f:
                g = f[group_name]
                logger.info(f"Found group '{group_name}' with keys: {list(g.keys())}")
                for k in g.keys():
                    if isinstance(g[k], h5py.Dataset):
                        data[f"{group_name}/{k}"] = g[k][:]
                        logger.info(f"  Loaded {group_name}/{k}: shape={g[k].shape}")

        # Check for node (neuron) data
        for group_name in ["nodes", "neurons", "cells"]:
            if group_name in f:
                g = f[group_name]
                logger.info(f"Found group '{group_name}' with keys: {list(g.keys())}")
                for k in g.keys():
                    if isinstance(g[k], h5py.Dataset):
                        data[f"{group_name}/{k}"] = g[k][:]

        # If flat structure, load all datasets
        if not data:
            for k in f.keys():
                if isinstance(f[k], h5py.Dataset):
                    data[k] = f[k][:]
                    logger.info(f"  Loaded {k}: shape={f[k].shape}")

    return data


def extract_ei_distributions(h5_path: str) -> dict:
    """Extract synapse size distributions split by E/I cell types.

    Returns dict with:
      - 'ee_sizes': E→E synapse sizes
      - 'ie_sizes': I→E synapse sizes
      - 'ei_sizes': E→I synapse sizes
      - 'ii_sizes': I→I synapse sizes
      - 'ee_counts': E→E synapse counts per connection
      - etc.
      - 'lognormal_fits': fitted parameters for each quadrant
    """
    data = load_microns_connectome(h5_path)

    # Print all available keys to understand structure
    logger.info(f"Available data keys: {sorted(data.keys())}")

    # The structure depends on the HDF5 organization
    # Try to find: pre/post neuron IDs, synapse sizes/counts, cell types
    results = {}

    # Look for edge data (condensed connectome)
    # Expected: pre_pt_root_id, post_pt_root_id, count, mean_size, sum_size
    # And node data: root_id, cell_type (excitatory/inhibitory)

    # We'll handle multiple possible layouts
    edge_keys = {}
    node_keys = {}

    for k, v in data.items():
        kl = k.lower()
        if "pre" in kl and ("root" in kl or "id" in kl or "source" in kl):
            edge_keys["pre_id"] = k
        elif "post" in kl and ("root" in kl or "id" in kl or "target" in kl):
            edge_keys["post_id"] = k
        elif "count" in kl and "syn" not in kl.replace("count", ""):
            edge_keys["count"] = k
        elif "mean_size" in kl or "avg_size" in kl:
            edge_keys["mean_size"] = k
        elif "sum_size" in kl or "total_size" in kl:
            edge_keys["sum_size"] = k
        elif "size" in kl and "mean" not in kl and "sum" not in kl:
            edge_keys.setdefault("size", k)

        # Node/cell type data
        if "cell_type" in kl or "mtype" in kl or "etype" in kl or "class" in kl:
            node_keys["cell_type"] = k
        if "root_id" in kl and "pre" not in kl and "post" not in kl:
            node_keys["root_id"] = k

    logger.info(f"Edge keys found: {edge_keys}")
    logger.info(f"Node keys found: {node_keys}")

    return data, edge_keys, node_keys


def analyze_microns_data(h5_path: str, output_dir: str):
    """Main analysis: extract E/I synapse distributions and fit lognormals.

    HDF5 layout (v1181):
      connectivity/condensed/edge_indices/block0_values  [N_edges, 2]  (row, col = pre, post node index)
      connectivity/condensed/edges/block0_values         [N_edges, 2]  (count, total_size)
      connectivity/condensed/edges/block1_values         [N_edges, 1]  (mean_size)
      connectivity/condensed/vertex_properties/table     [N_nodes]     structured array with cell type
    """
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        # ── Load edge data (condensed: one row per connected neuron pair) ──
        edge_idx = f["connectivity/condensed/edge_indices/block0_values"][:]  # [N, 2]
        edge_props = f["connectivity/condensed/edges/block0_values"][:]       # [N, 2]
        edge_mean = f["connectivity/condensed/edges/block1_values"][:]        # [N, 1]

        pre_idx = edge_idx[:, 0]    # node index of presynaptic neuron
        post_idx = edge_idx[:, 1]   # node index of postsynaptic neuron
        syn_count = edge_props[:, 0]
        total_size = edge_props[:, 1]
        mean_size = edge_mean[:, 0]

        logger.info(f"Loaded {len(pre_idx):,} connections")
        logger.info(f"Synapse count range: {syn_count.min()}-{syn_count.max()}, "
                     f"median={np.median(syn_count):.1f}")
        logger.info(f"Mean size range: {mean_size.min():.0f}-{mean_size.max():.0f}, "
                     f"median={np.median(mean_size):.0f}")

        # ── Load neuron properties ──
        vtx_table = f["connectivity/condensed/vertex_properties/table"][:]

    # Parse cell types from structured array
    # Column layout: ('index', int), ('values_block_0', int[7]), ('values_block_1', float[1]), ('values_block_2', str[7])
    # values_block_2[:, 1] = cell class ('excitatory_neuron' or inhibitory subtype)
    n_nodes = len(vtx_table)
    logger.info(f"Loaded {n_nodes:,} neurons")

    # Build E/I lookup by node index (0-based)
    cell_class = np.empty(n_nodes, dtype="U1")
    layer_labels = np.empty(n_nodes, dtype="U10")
    for i, row in enumerate(vtx_table):
        str_block = row["values_block_2"]  # array of 7 byte-strings
        class_str = str_block[1].decode() if isinstance(str_block[1], bytes) else str(str_block[1])
        layer_str = str_block[2].decode() if isinstance(str_block[2], bytes) else str(str_block[2])

        if "excitatory" in class_str.lower() or "pyr" in class_str.lower():
            cell_class[i] = "E"
        elif "inhibitory" in class_str.lower() or class_str.lower() in ("", "empty"):
            # Check if it's truly inhibitory or just unclassified
            if "inhibitory" in class_str.lower():
                cell_class[i] = "I"
            else:
                cell_class[i] = "?"
        else:
            cell_class[i] = "?"

        layer_labels[i] = layer_str

    n_exc = np.sum(cell_class == "E")
    n_inh = np.sum(cell_class == "I")
    n_unk = np.sum(cell_class == "?")
    logger.info(f"Cell types: {n_exc:,} E, {n_inh:,} I, {n_unk:,} unknown")
    logger.info(f"E/(E+I) ratio: {n_exc/(n_exc+n_inh):.1%}")

    # Unique cell classes for debugging
    unique_classes = set()
    for row in vtx_table[:100]:
        unique_classes.add(row["values_block_2"][1])
    logger.info(f"Sample cell classes: {unique_classes}")

    # ── Split connections by E/I quadrant (vectorized) ──
    # Map pre/post indices to E/I type, filtering out-of-range indices
    valid_mask = (pre_idx < n_nodes) & (post_idx < n_nodes)
    pre_idx_v = pre_idx[valid_mask].astype(int)
    post_idx_v = post_idx[valid_mask].astype(int)
    mean_size_v = mean_size[valid_mask]
    syn_count_v = syn_count[valid_mask]
    total_size_v = total_size[valid_mask]

    pre_types = cell_class[pre_idx_v]
    post_types = cell_class[post_idx_v]

    logger.info(f"Valid connections (both nodes in range): {len(pre_idx_v):,}")

    quadrant_sizes = {}
    quadrant_counts = {}
    quadrant_totals = {}
    for quad, pre_t, post_t in [("EE", "E", "E"), ("IE", "I", "E"),
                                 ("EI", "E", "I"), ("II", "I", "I")]:
        mask = (pre_types == pre_t) & (post_types == post_t)
        quadrant_sizes[quad] = mean_size_v[mask]
        quadrant_counts[quad] = syn_count_v[mask]
        quadrant_totals[quad] = total_size_v[mask]
        logger.info(f"  {quad}: {mask.sum():,} connections")

    # ── Fit lognormal to each quadrant ──
    results = {}
    for quad in ["EE", "IE", "EI", "II"]:
        sizes = quadrant_sizes[quad].astype(float)
        counts = quadrant_counts[quad].astype(float)
        totals = quadrant_totals[quad].astype(float)

        sizes = sizes[sizes > 0]
        counts = counts[counts > 0]
        totals = totals[totals > 0]

        if len(sizes) < 100:
            logger.info(f"  {quad}: only {len(sizes)} connections, skipping")
            continue

        log_sizes = np.log(sizes)
        mu_s = np.mean(log_sizes)
        sigma_s = np.std(log_sizes)
        ks_s, ksp_s = stats.kstest(
            (log_sizes - mu_s) / sigma_s, "norm"
        )

        log_totals = np.log(totals)
        mu_t = np.mean(log_totals)
        sigma_t = np.std(log_totals)

        results[quad] = {
            "n_connections": int(len(sizes)),
            # Mean synapse size per connection
            "mean_size_lognormal_mu": float(mu_s),
            "mean_size_lognormal_sigma": float(sigma_s),
            "mean_size_median": float(np.median(sizes)),
            "mean_size_mean": float(np.mean(sizes)),
            "mean_size_cv": float(np.std(sizes) / np.mean(sizes)),
            "mean_size_ks_stat": float(ks_s),
            "mean_size_skewness_log": float(stats.skew(log_sizes)),
            # Total synaptic weight per connection
            "total_size_lognormal_mu": float(mu_t),
            "total_size_lognormal_sigma": float(sigma_t),
            "total_size_median": float(np.median(totals)),
            "total_size_cv": float(np.std(totals) / np.mean(totals)),
            # Synapse count per connection
            "count_mean": float(np.mean(counts)),
            "count_median": float(np.median(counts)),
            "count_max": float(np.max(counts)),
            "frac_multi_synapse": float(np.mean(counts > 1)),
        }

        logger.info(
            f"  {quad}: n={len(sizes):>8,d}  "
            f"mean_size σ={sigma_s:.3f}  total_size σ={sigma_t:.3f}  "
            f"count_median={np.median(counts):.0f}  multi_syn={np.mean(counts > 1):.1%}"
        )

    # Save results
    results_path = os.path.join(output_dir, "microns_ei_distributions.json")
    with open(results_path, "w") as fp:
        json.dump(results, fp, indent=2)
    logger.info(f"Saved E/I distributions to {results_path}")

    # Also save raw distributions for later comparison
    for quad in ["EE", "IE", "EI", "II"]:
        if len(quadrant_sizes[quad]) > 0:
            arr = quadrant_sizes[quad].astype(np.float32)
            np.save(os.path.join(output_dir, f"microns_{quad}_mean_sizes.npy"), arr[arr > 0])
            arr_t = quadrant_totals[quad].astype(np.float32)
            np.save(os.path.join(output_dir, f"microns_{quad}_total_sizes.npy"), arr_t[arr_t > 0])

    # Generate figures
    _plot_ei_distributions(quadrant_sizes, results, output_dir)

    # Print summary
    print("\n" + "=" * 80)
    print("MICrONS SYNAPSE SIZE DISTRIBUTIONS BY E/I QUADRANT")
    print("=" * 80)
    print(f"{'Quadrant':>10s} {'N':>10s} {'LN μ':>10s} {'LN σ':>10s} "
          f"{'CV':>8s} {'Median':>10s} {'Count':>8s} {'Multi%':>8s}")
    print("-" * 80)
    for quad in ["EE", "IE", "EI", "II"]:
        if quad in results:
            r = results[quad]
            print(
                f"{quad:>10s} {r['n_connections']:>10,d} "
                f"{r['mean_size_lognormal_mu']:>10.3f} {r['mean_size_lognormal_sigma']:>10.3f} "
                f"{r['mean_size_cv']:>8.3f} {r['mean_size_median']:>10.0f} "
                f"{r['count_median']:>8.0f} {r['frac_multi_synapse']:>7.1%}"
            )
    print("-" * 80)
    print(f"{'Song 2005':>10s} {'931':>10s} {'-0.702':>10s} {'0.936':>10s}")
    print(f"{'':>10s} (rat V1 L5, EPSP amplitude, excitatory only)")
    print("=" * 80)


    # No separate _extract_and_fit needed — logic is above in analyze_microns_data


def _plot_ei_distributions(quadrants, results, output_dir):
    """Plot synapse size distributions for each E/I quadrant."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
    quad_labels = {"EE": "E → E", "IE": "I → E", "EI": "E → I", "II": "I → I"}
    quad_colors = {"EE": "#EE6677", "IE": "#4477AA", "EI": "#CCBB44", "II": "#66CCEE"}

    for idx, (quad, label) in enumerate(quad_labels.items()):
        ax = axes[idx // 2, idx % 2]
        if quad not in quadrants or len(quadrants[quad]) < 100:
            ax.set_visible(False)
            continue

        arr = np.asarray(quadrants[quad], dtype=float)
        arr = arr[arr > 0]
        log_arr = np.log(arr)

        ax.hist(log_arr, bins=100, density=True, alpha=0.6,
                color=quad_colors[quad], label=f"MICrONS {label}")

        # Overlay fitted lognormal
        if quad in results:
            mu = results[quad]["mean_size_lognormal_mu"]
            sigma = results[quad]["mean_size_lognormal_sigma"]
            x = np.linspace(log_arr.min(), log_arr.max(), 200)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), "k-", lw=2,
                    label=f"LN fit (σ={sigma:.2f})")

        ax.set_title(f"{label} (n={len(arr):,})", fontsize=12)
        ax.set_xlabel("log(synapse size)")
        ax.legend(fontsize=9)

    fig.suptitle("MICrONS: Synapse Size Distributions by E/I Type",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_path = os.path.join(output_dir, "microns_ei_synapse_distributions.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved figure to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Extract E/I synapse distributions from MICrONS connectome"
    )
    parser.add_argument(
        "--h5-path",
        default="/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/microns_data/microns_mm3_connectome_v1181.h5",
        help="Path to MICrONS HDF5 file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/microns_distributions",
        help="Output directory for results",
    )
    args = parser.parse_args()
    analyze_microns_data(args.h5_path, args.output_dir)


if __name__ == "__main__":
    main()

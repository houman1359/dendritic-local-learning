"""
Weight distribution analysis: compare learned synaptic weights across
training strategies and architectures against biological reference data.

Extracts excitatory and inhibitory weights from trained model checkpoints,
fits log-normal and normal distributions, and compares to EM connectomics
data (Song et al. 2005, Buzsaki & Mizuseki 2014, MICrONS).

Key prediction: shunting + local learning produces multiplicative weight
dynamics (updates scale with R_tot = 1/g_tot) → log-normal distributions,
matching biology. Standard BP produces non-log-normal distributions.

Reference:
  Saxena et al. (2024, ICLR) showed that multiplicative update rules
  (exponentiated gradient) produce log-normal weights, while Euclidean
  gradient descent does not.
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Biological reference distributions ──────────────────────────────────────
# Song et al. (2005) PLoS Biology — rat V1, L5 pyramidal neurons
# Log-normal parameters for EPSP amplitudes (mV)
SONG_2005_MU = -0.702        # mean of ln(EPSP)
SONG_2005_SIGMA = 0.9355     # std of ln(EPSP)

# Loewenstein et al. (2011) J Neurosci — mouse auditory cortex spine sizes
LOEWENSTEIN_2011_MU = 1.74
LOEWENSTEIN_2011_SIGMA = 0.316  # sqrt(0.10)


def extract_weights_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    weight_transform: str = "softplus",
) -> dict[str, np.ndarray]:
    """Extract and transform synaptic weights from a model checkpoint.

    Returns dict with keys like 'exc_layer0', 'inh_layer0', etc.,
    each mapping to a 1D array of the ACTIVE (top-K masked) synaptic weights.
    Also stores '_all' variants with all weights for comparison.
    """
    sd = torch.load(checkpoint_path, map_location="cpu")

    # Load config to get K values
    with open(config_path) as f:
        config = json.load(f)
    connectivity = config["model"]["core"].get("connectivity", {})
    ee_k_list = connectivity.get("ee_synapses_per_branch_per_layer", [40])
    ie_k_list = connectivity.get("ie_synapses_per_branch_per_layer", [20])

    weights = {}

    for key, tensor in sd.items():
        if "pre_w" not in key:
            continue

        # Determine type and K
        if "branch_excitation" in key:
            wtype = "exc"
            k_list = ee_k_list
        elif "branch_inhibition" in key:
            wtype = "inh"
            k_list = ie_k_list
        else:
            continue

        # Extract branch layer index
        parts = key.split(".")
        try:
            layer_idx = int(parts[parts.index("branch_layers") + 1])
        except (ValueError, IndexError):
            layer_idx = 0

        K = k_list[min(layer_idx, len(k_list) - 1)]

        # Transform from parameter space to weight space
        if weight_transform == "softplus":
            w = F.softplus(tensor)
        elif weight_transform == "exp":
            w = torch.exp(tensor)
        elif weight_transform in ("identity", "relu"):
            w = tensor.abs() if weight_transform == "relu" else tensor
        else:
            w = F.softplus(tensor)

        w_np = w.numpy()
        out_features, in_features = w_np.shape

        # Store all weights (for reference)
        weights[f"{wtype}_layer{layer_idx}_all"] = w_np.flatten()

        # Apply top-K mask: for each branch, keep only the K largest
        # This mirrors the TopKLinear.weight_mask() logic
        if K < in_features:
            topk_indices = torch.topk(tensor, K, dim=-1, largest=True)[1]
            mask = torch.zeros_like(tensor)
            mask.scatter_(1, topk_indices, 1.0)
            active_w = (w * mask).numpy()
            # Extract only the non-zero (active) values
            active_flat = active_w[active_w > 0]
        else:
            active_flat = w_np[w_np > 0].flatten()

        weights[f"{wtype}_layer{layer_idx}"] = active_flat

    return weights


def extract_weights_from_sweep(
    sweep_dir: str,
    filter_fn=None,
    max_seeds: int = 5,
) -> dict[str, list[dict[str, np.ndarray]]]:
    """Extract weights from all matching configs in a sweep directory.

    Args:
        sweep_dir: Path to sweep results directory.
        filter_fn: Function(config_dict) -> bool to select configs.
        max_seeds: Maximum seeds to load per condition.

    Returns:
        Dict mapping condition_key -> list of weight dicts (one per seed).
    """
    results_dir = os.path.join(sweep_dir, "results")
    if not os.path.isdir(results_dir):
        logger.warning(f"No results directory: {results_dir}")
        return {}

    conditions = defaultdict(list)

    for config_name in sorted(os.listdir(results_dir)):
        config_dir = os.path.join(results_dir, config_name)
        config_path = os.path.join(config_dir, "config.json")
        model_path = os.path.join(config_dir, "final_model.pt")

        if not os.path.isfile(config_path) or not os.path.isfile(model_path):
            continue

        with open(config_path) as f:
            config = json.load(f)

        if filter_fn and not filter_fn(config):
            continue

        strategy = config["training"]["main"]["strategy"]
        core_type = config["model"]["core"]["type"]
        weight_transform = config["model"]["core"].get("morphology", {}).get(
            "weight_transform", "softplus"
        )

        condition_key = f"{strategy}_{core_type}"

        if len(conditions[condition_key]) >= max_seeds:
            continue

        logger.info(f"Loading {config_name}: {condition_key}")
        w = extract_weights_from_checkpoint(model_path, config_path, weight_transform)
        w["_config"] = config
        w["_config_name"] = config_name
        conditions[condition_key] = conditions.get(condition_key, [])
        conditions[condition_key].append(w)

    return dict(conditions)


def fit_lognormal(data: np.ndarray) -> dict:
    """Fit log-normal distribution to positive data.

    Returns dict with mu, sigma (of log), KS statistic and p-value.
    """
    data = data[data > 0]
    if len(data) < 10:
        return {"mu": np.nan, "sigma": np.nan, "ks_stat": np.nan, "ks_p": np.nan}

    log_data = np.log(data)
    mu = np.mean(log_data)
    sigma = np.std(log_data)

    # KS test against fitted log-normal
    ks_stat, ks_p = stats.kstest(log_data, "norm", args=(mu, sigma))

    return {
        "mu": mu,
        "sigma": sigma,
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "median": np.median(data),
        "mean": np.mean(data),
        "cv": np.std(data) / np.mean(data) if np.mean(data) > 0 else np.nan,
        "skewness": stats.skew(log_data),
        "kurtosis": stats.kurtosis(log_data),
        "n": len(data),
    }


def fit_normal(data: np.ndarray) -> dict:
    """Fit normal distribution."""
    if len(data) < 10:
        return {"mu": np.nan, "sigma": np.nan, "ks_stat": np.nan, "ks_p": np.nan}

    mu = np.mean(data)
    sigma = np.std(data)
    ks_stat, ks_p = stats.kstest(data, "norm", args=(mu, sigma))

    return {"mu": mu, "sigma": sigma, "ks_stat": ks_stat, "ks_p": ks_p}


def compute_distribution_stats(
    conditions: dict[str, list[dict[str, np.ndarray]]],
    weight_key: str = "exc_layer0",
) -> dict:
    """Compute distribution statistics for each condition.

    Returns dict mapping condition -> aggregated stats across seeds.
    """
    results = {}

    for condition, seed_weights_list in conditions.items():
        stats_list = []
        for w_dict in seed_weights_list:
            if weight_key not in w_dict:
                continue
            data = w_dict[weight_key]
            ln_fit = fit_lognormal(data)
            n_fit = fit_normal(data)
            stats_list.append({
                "lognormal": ln_fit,
                "normal": n_fit,
            })

        if not stats_list:
            continue

        # Aggregate across seeds
        ln_mus = [s["lognormal"]["mu"] for s in stats_list if not np.isnan(s["lognormal"]["mu"])]
        ln_sigmas = [s["lognormal"]["sigma"] for s in stats_list if not np.isnan(s["lognormal"]["sigma"])]
        ln_ks = [s["lognormal"]["ks_stat"] for s in stats_list if not np.isnan(s["lognormal"]["ks_stat"])]
        ln_cvs = [s["lognormal"]["cv"] for s in stats_list if not np.isnan(s["lognormal"]["cv"])]
        ln_skew = [s["lognormal"]["skewness"] for s in stats_list if not np.isnan(s["lognormal"]["skewness"])]

        results[condition] = {
            "lognormal_mu": (np.mean(ln_mus), np.std(ln_mus)),
            "lognormal_sigma": (np.mean(ln_sigmas), np.std(ln_sigmas)),
            "lognormal_ks": (np.mean(ln_ks), np.std(ln_ks)),
            "cv": (np.mean(ln_cvs), np.std(ln_cvs)),
            "skewness": (np.mean(ln_skew), np.std(ln_skew)),
            "n_seeds": len(stats_list),
        }

    return results


# ── Plotting ────────────────────────────────────────────────────────────────

CONDITION_LABELS = {
    "standard_dendritic_shunting": "BP + Shunting",
    "standard_dendritic_additive": "BP + Additive",
    "local_ca_dendritic_shunting": "Local + Shunting",
    "local_ca_dendritic_additive": "Local + Additive",
}

CONDITION_COLORS = {
    "standard_dendritic_shunting": "#4477AA",
    "standard_dendritic_additive": "#66CCEE",
    "local_ca_dendritic_shunting": "#EE6677",
    "local_ca_dendritic_additive": "#CCBB44",
}

CONDITION_ORDER = [
    "standard_dendritic_shunting",
    "standard_dendritic_additive",
    "local_ca_dendritic_shunting",
    "local_ca_dendritic_additive",
]


def plot_weight_distributions(
    conditions: dict[str, list[dict[str, np.ndarray]]],
    weight_key: str = "exc_layer0",
    output_path: str = "weight_distributions.png",
    title_suffix: str = "",
):
    """Main figure: weight distribution comparison across 4 conditions.

    Panel layout:
      (A) Overlaid log-scale histograms (all 4 conditions)
      (B) QQ plots against log-normal (2x2 subpanels)
      (C) Summary bar chart of log-normal sigma and KS statistic
      (D) Biological reference overlay
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # ── Panel A: Overlaid log-weight histograms ──
    ax_a = fig.add_subplot(gs[0, 0])

    for cond in CONDITION_ORDER:
        if cond not in conditions:
            continue
        # Pool weights from all seeds
        all_w = []
        for w_dict in conditions[cond]:
            if weight_key in w_dict:
                all_w.append(w_dict[weight_key])
        if not all_w:
            continue
        pooled = np.concatenate(all_w)
        pooled = pooled[pooled > 0]

        log_w = np.log(pooled)
        label = CONDITION_LABELS.get(cond, cond)
        color = CONDITION_COLORS.get(cond, None)
        ax_a.hist(
            log_w, bins=100, density=True, alpha=0.5,
            label=label, color=color, histtype="stepfilled",
        )

    # Add biological reference (Song et al. 2005)
    x_ref = np.linspace(-5, 3, 200)
    bio_pdf = stats.norm.pdf(x_ref, SONG_2005_MU, SONG_2005_SIGMA)
    ax_a.plot(x_ref, bio_pdf, "k--", lw=2, label="Song et al. 2005 (V1)")

    ax_a.set_xlabel("log(weight)")
    ax_a.set_ylabel("Density")
    ax_a.set_title(f"A. Log-weight distributions{title_suffix}")
    ax_a.legend(fontsize=8, loc="upper left")

    # ── Panel B: QQ plots (2x2) ──
    gs_b = gs[0, 1].subgridspec(2, 2, hspace=0.4, wspace=0.4)

    for i, cond in enumerate(CONDITION_ORDER):
        ax = fig.add_subplot(gs_b[i // 2, i % 2])
        if cond not in conditions:
            ax.set_visible(False)
            continue

        all_w = []
        for w_dict in conditions[cond]:
            if weight_key in w_dict:
                all_w.append(w_dict[weight_key])
        if not all_w:
            continue

        pooled = np.concatenate(all_w)
        pooled = pooled[pooled > 0]
        log_w = np.log(pooled)

        # Subsample for QQ plot (too many points is slow)
        if len(log_w) > 10000:
            rng = np.random.default_rng(42)
            log_w = rng.choice(log_w, 10000, replace=False)

        (osm, osr), (slope, intercept, _) = stats.probplot(log_w, dist="norm")
        color = CONDITION_COLORS.get(cond, "gray")
        ax.scatter(osm, osr, s=1, alpha=0.3, color=color)
        ax.plot(osm, slope * osm + intercept, "k-", lw=1)

        label = CONDITION_LABELS.get(cond, cond)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("")

    # Add shared labels
    fig.text(0.75, 0.52, "B. QQ plots (normal reference for log-weights)",
             ha="center", fontsize=11, fontweight="bold")

    # ── Panel C: Summary statistics bar chart ──
    ax_c = fig.add_subplot(gs[1, 0])

    cond_stats = compute_distribution_stats(conditions, weight_key)
    conds_present = [c for c in CONDITION_ORDER if c in cond_stats]

    if conds_present:
        x_pos = np.arange(len(conds_present))
        sigmas = [cond_stats[c]["lognormal_sigma"][0] for c in conds_present]
        sigma_errs = [cond_stats[c]["lognormal_sigma"][1] for c in conds_present]
        colors = [CONDITION_COLORS.get(c, "gray") for c in conds_present]
        labels = [CONDITION_LABELS.get(c, c) for c in conds_present]

        bars = ax_c.bar(x_pos, sigmas, yerr=sigma_errs, color=colors,
                        capsize=4, edgecolor="black", linewidth=0.5)

        # Biological reference line
        ax_c.axhline(SONG_2005_SIGMA, color="black", ls="--", lw=1.5,
                      label=f"Song et al. 2005 (σ={SONG_2005_SIGMA:.2f})")

        ax_c.set_xticks(x_pos)
        ax_c.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax_c.set_ylabel("Log-normal σ (log-weight dispersion)")
        ax_c.set_title("C. Log-normal width parameter")
        ax_c.legend(fontsize=8)

    # ── Panel D: Coefficient of variation + skewness ──
    ax_d = fig.add_subplot(gs[1, 1])

    if conds_present:
        x_pos = np.arange(len(conds_present))
        cvs = [cond_stats[c]["cv"][0] for c in conds_present]
        cv_errs = [cond_stats[c]["cv"][1] for c in conds_present]
        skews = [cond_stats[c]["skewness"][0] for c in conds_present]
        skew_errs = [cond_stats[c]["skewness"][1] for c in conds_present]

        width = 0.35
        ax_d.bar(x_pos - width / 2, cvs, width, yerr=cv_errs,
                 label="CV", color=[CONDITION_COLORS.get(c, "gray") for c in conds_present],
                 alpha=0.7, capsize=3, edgecolor="black", linewidth=0.5)
        ax_d2 = ax_d.twinx()
        ax_d2.bar(x_pos + width / 2, skews, width, yerr=skew_errs,
                  label="Skewness (log-w)", color=[CONDITION_COLORS.get(c, "gray") for c in conds_present],
                  alpha=0.3, capsize=3, edgecolor="black", linewidth=0.5, hatch="//")

        ax_d.set_xticks(x_pos)
        ax_d.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax_d.set_ylabel("Coefficient of Variation")
        ax_d2.set_ylabel("Skewness of log(weight)")
        ax_d.set_title("D. Weight heterogeneity measures")

        # Combined legend
        lines_a, labels_a = ax_d.get_legend_handles_labels()
        lines_b, labels_b = ax_d2.get_legend_handles_labels()
        ax_d.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper left")

    fig.suptitle(
        "Synaptic Weight Distributions: Local Learning × Shunting Architecture",
        fontsize=14, fontweight="bold", y=0.98,
    )

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved figure to {output_path}")
    plt.close()


def plot_exc_vs_inh_distributions(
    conditions: dict[str, list[dict[str, np.ndarray]]],
    output_path: str = "weight_distributions_ei.png",
):
    """Compare excitatory vs inhibitory weight distributions for each condition."""
    n_cond = len([c for c in CONDITION_ORDER if c in conditions])
    if n_cond == 0:
        return

    fig, axes = plt.subplots(1, n_cond, figsize=(4 * n_cond, 4), sharey=True)
    if n_cond == 1:
        axes = [axes]

    for idx, cond in enumerate([c for c in CONDITION_ORDER if c in conditions]):
        ax = axes[idx]

        for wtype, color, ls in [("exc_layer0", "tab:red", "-"), ("inh_layer0", "tab:blue", "--")]:
            all_w = []
            for w_dict in conditions[cond]:
                if wtype in w_dict:
                    all_w.append(w_dict[wtype])
            if not all_w:
                continue
            pooled = np.concatenate(all_w)
            pooled = pooled[pooled > 0]
            log_w = np.log(pooled)

            label_prefix = "Excitatory" if "exc" in wtype else "Inhibitory"
            ax.hist(log_w, bins=80, density=True, alpha=0.4, color=color,
                    histtype="stepfilled", label=label_prefix)

        ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=10)
        ax.set_xlabel("log(weight)")
        if idx == 0:
            ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    fig.suptitle("Excitatory vs Inhibitory Weight Distributions", fontsize=12, fontweight="bold")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved E/I figure to {output_path}")
    plt.close()


def save_statistics_table(
    conditions: dict[str, list[dict[str, np.ndarray]]],
    output_path: str = "weight_distribution_stats.json",
):
    """Save comprehensive statistics table as JSON."""
    all_stats = {}

    for weight_key in ["exc_layer0", "inh_layer0", "exc_layer1", "inh_layer1"]:
        all_stats[weight_key] = {}
        for cond in CONDITION_ORDER:
            if cond not in conditions:
                continue
            seeds_stats = []
            for w_dict in conditions[cond]:
                if weight_key not in w_dict:
                    continue
                data = w_dict[weight_key]
                ln = fit_lognormal(data)
                seeds_stats.append(ln)

            if seeds_stats:
                # Aggregate
                agg = {}
                for k in seeds_stats[0]:
                    vals = [s[k] for s in seeds_stats if not (isinstance(s[k], float) and np.isnan(s[k]))]
                    if vals and isinstance(vals[0], (int, float)):
                        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
                    else:
                        agg[k] = vals[0] if vals else None
                all_stats[weight_key][cond] = agg

    with open(output_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    logger.info(f"Saved statistics to {output_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze synaptic weight distributions across training strategies"
    )
    parser.add_argument(
        "--sweep-dirs", nargs="+", required=True,
        help="Paths to sweep result directories",
    )
    parser.add_argument(
        "--output-dir", default="data/weight_distributions",
        help="Directory for output figures and data",
    )
    parser.add_argument(
        "--filter-noise-rate", type=float, default=None,
        help="Only include configs with this label_noise_rate (e.g. 0.0 for clean)",
    )
    parser.add_argument(
        "--filter-dataset", type=str, default=None,
        help="Only include configs with this dataset name (e.g. mnist)",
    )
    parser.add_argument(
        "--max-seeds", type=int, default=5,
        help="Maximum seeds per condition to load",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build filter function
    def filter_fn(config):
        if args.filter_noise_rate is not None:
            noise = config.get("data", {}).get("processing", {}).get("label_noise_rate", 0.0)
            if abs(noise - args.filter_noise_rate) > 1e-6:
                return False
        if args.filter_dataset is not None:
            ds = config.get("data", {}).get("dataset_name", "")
            if ds != args.filter_dataset:
                return False
        return True

    # Extract weights from all sweeps
    all_conditions = {}
    for sweep_dir in args.sweep_dirs:
        logger.info(f"Processing sweep: {sweep_dir}")
        conds = extract_weights_from_sweep(
            sweep_dir, filter_fn=filter_fn, max_seeds=args.max_seeds
        )
        for k, v in conds.items():
            if k not in all_conditions:
                all_conditions[k] = []
            all_conditions[k].extend(v)

    if not all_conditions:
        logger.error("No matching conditions found!")
        sys.exit(1)

    logger.info(f"Loaded conditions: {list(all_conditions.keys())}")
    for k, v in all_conditions.items():
        logger.info(f"  {k}: {len(v)} seeds")

    # Generate all outputs
    plot_weight_distributions(
        all_conditions,
        weight_key="exc_layer0",
        output_path=os.path.join(args.output_dir, "weight_distributions_exc.png"),
        title_suffix=" (excitatory, layer 0)",
    )

    plot_weight_distributions(
        all_conditions,
        weight_key="inh_layer0",
        output_path=os.path.join(args.output_dir, "weight_distributions_inh.png"),
        title_suffix=" (inhibitory, layer 0)",
    )

    plot_exc_vs_inh_distributions(
        all_conditions,
        output_path=os.path.join(args.output_dir, "weight_distributions_ei.png"),
    )

    save_statistics_table(
        all_conditions,
        output_path=os.path.join(args.output_dir, "weight_distribution_stats.json"),
    )

    # Print summary table
    print("\n" + "=" * 80)
    print("WEIGHT DISTRIBUTION SUMMARY (excitatory, layer 0)")
    print("=" * 80)
    print(f"{'Condition':35s} {'LN μ':>10s} {'LN σ':>10s} {'CV':>10s} {'Skew':>10s} {'Seeds':>6s}")
    print("-" * 80)

    cond_stats = compute_distribution_stats(all_conditions, "exc_layer0")
    for cond in CONDITION_ORDER:
        if cond not in cond_stats:
            continue
        s = cond_stats[cond]
        label = CONDITION_LABELS.get(cond, cond)
        print(
            f"{label:35s} "
            f"{s['lognormal_mu'][0]:>8.3f}±{s['lognormal_mu'][1]:.3f} "
            f"{s['lognormal_sigma'][0]:>8.3f}±{s['lognormal_sigma'][1]:.3f} "
            f"{s['cv'][0]:>8.3f}±{s['cv'][1]:.3f} "
            f"{s['skewness'][0]:>8.3f}±{s['skewness'][1]:.3f} "
            f"{s['n_seeds']:>6d}"
        )

    print("-" * 80)
    print(f"{'Song et al. 2005 (V1 biology)':35s} {SONG_2005_MU:>10.3f} {SONG_2005_SIGMA:>10.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

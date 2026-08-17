"""
Generate publication figure: model weight distributions vs biological data.

Overlays learned synaptic weight distributions from 4 conditions
(BP/Local × Shunting/Additive) with MICrONS connectome data and
Song et al. (2005) electrophysiology.

Key comparison:
  - Model excitatory weights ↔ MICrONS E→E synapse sizes
  - Model inhibitory weights ↔ MICrONS I→E synapse sizes
  - σ parameter as summary statistic for distribution width

Usage:
  cd dendritic-modeling
  python drafts/dendritic-local-learning/scripts/figure_weight_vs_biology.py \
    --sweep-dirs /path/to/sweep1 /path/to/sweep2 \
    --microns-dir drafts/dendritic-local-learning/data/microns_distributions \
    --output-dir drafts/dendritic-local-learning/data/weight_distributions
"""

import argparse
import json
import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# Import from sibling script
import sys
sys.path.insert(0, os.path.dirname(__file__))
from weight_distribution_analysis import (
    extract_weights_from_sweep,
    fit_lognormal,
    compute_distribution_stats,
    CONDITION_LABELS,
    CONDITION_COLORS,
    CONDITION_ORDER,
    SONG_2005_MU,
    SONG_2005_SIGMA,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# MICrONS reference colors
MICRONS_COLORS = {
    "EE": "#888888",  # dark gray
    "IE": "#555555",  # medium gray
}


def load_microns_data(microns_dir: str) -> dict:
    """Load MICrONS E/I distribution data."""
    json_path = os.path.join(microns_dir, "microns_ei_distributions.json")
    if not os.path.isfile(json_path):
        logger.warning(f"MICrONS data not found at {json_path}")
        return {}

    with open(json_path) as f:
        microns_stats = json.load(f)

    # Also load raw distributions if available
    result = {"stats": microns_stats}
    for quad in ["EE", "IE", "EI", "II"]:
        npy_path = os.path.join(microns_dir, f"microns_{quad}_mean_sizes.npy")
        if os.path.isfile(npy_path):
            result[f"{quad}_sizes"] = np.load(npy_path)
            logger.info(f"Loaded MICrONS {quad}: {len(result[f'{quad}_sizes']):,} synapses")

    return result


def plot_model_vs_biology(
    conditions: dict,
    microns_data: dict,
    output_dir: str,
    dataset_label: str = "MNIST",
):
    """Generate 3-panel publication figure.

    Panel A: Overlaid log-weight histograms (4 model conditions) + MICrONS E→E overlay
    Panel B: σ bar chart comparing all conditions + biological references
    Panel C: Excitatory vs inhibitory comparison with MICrONS E→E and I→E
    """
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.35, left=0.06, right=0.97, top=0.88, bottom=0.15)

    microns_stats = microns_data.get("stats", {})

    # ── Panel A: Overlaid distributions ──
    ax_a = fig.add_subplot(gs[0, 0])

    for cond in CONDITION_ORDER:
        if cond not in conditions:
            continue
        all_w = []
        for w_dict in conditions[cond]:
            if "exc_layer0" in w_dict:
                all_w.append(w_dict["exc_layer0"])
        if not all_w:
            continue

        pooled = np.concatenate(all_w)
        pooled = pooled[pooled > 0]
        log_w = np.log(pooled)

        # Normalize to zero-mean for shape comparison (since model and bio are on different scales)
        log_w_centered = log_w - np.mean(log_w)

        label = CONDITION_LABELS.get(cond, cond)
        color = CONDITION_COLORS.get(cond, None)
        ax_a.hist(
            log_w_centered, bins=80, density=True, alpha=0.45,
            label=label, color=color, histtype="stepfilled",
        )

    # Overlay MICrONS E→E (centered)
    if "EE_sizes" in microns_data:
        ee_log = np.log(microns_data["EE_sizes"])
        ee_centered = ee_log - np.mean(ee_log)
        # Subsample for plotting (millions of points)
        rng = np.random.default_rng(42)
        if len(ee_centered) > 500000:
            ee_sub = rng.choice(ee_centered, 500000, replace=False)
        else:
            ee_sub = ee_centered
        ax_a.hist(
            ee_sub, bins=120, density=True, alpha=0.3,
            color="#888888", histtype="stepfilled",
            label=f"MICrONS E→E (σ={microns_stats.get('EE', {}).get('mean_size_lognormal_sigma', 0):.2f})",
        )

    # Song et al. 2005 reference curve (centered at 0)
    x_ref = np.linspace(-4, 4, 300)
    ax_a.plot(x_ref, stats.norm.pdf(x_ref, 0, SONG_2005_SIGMA), "k--", lw=2,
              label=f"Song 2005 (σ={SONG_2005_SIGMA:.2f})")

    ax_a.set_xlabel("Centered log(weight)", fontsize=11)
    ax_a.set_ylabel("Density", fontsize=11)
    ax_a.set_title("A. Excitatory weight distributions", fontsize=12, fontweight="bold")
    ax_a.legend(fontsize=7, loc="upper left")
    ax_a.set_xlim(-4.5, 4.5)

    # ── Panel B: σ bar chart ──
    ax_b = fig.add_subplot(gs[0, 1])

    cond_stats = compute_distribution_stats(conditions, "exc_layer0")
    conds_present = [c for c in CONDITION_ORDER if c in cond_stats]

    if conds_present:
        # Model conditions
        n_model = len(conds_present)
        x_model = np.arange(n_model)
        sigmas = [cond_stats[c]["lognormal_sigma"][0] for c in conds_present]
        sigma_errs = [cond_stats[c]["lognormal_sigma"][1] for c in conds_present]
        colors = [CONDITION_COLORS.get(c, "gray") for c in conds_present]
        labels = [CONDITION_LABELS.get(c, c) for c in conds_present]

        bars = ax_b.bar(x_model, sigmas, yerr=sigma_errs, color=colors,
                        capsize=4, edgecolor="black", linewidth=0.5, width=0.7)

        # Biological reference bars
        bio_x = []
        bio_sigmas = []
        bio_colors = []
        bio_labels = []

        if "EE" in microns_stats:
            bio_x.append(n_model + 0.5)
            bio_sigmas.append(microns_stats["EE"]["mean_size_lognormal_sigma"])
            bio_colors.append("#888888")
            bio_labels.append("MICrONS\nE→E")

        if "IE" in microns_stats:
            bio_x.append(n_model + 1.5)
            bio_sigmas.append(microns_stats["IE"]["mean_size_lognormal_sigma"])
            bio_colors.append("#555555")
            bio_labels.append("MICrONS\nI→E")

        bio_x.append(n_model + 2.5)
        bio_sigmas.append(SONG_2005_SIGMA)
        bio_colors.append("black")
        bio_labels.append("Song\n2005")

        ax_b.bar(bio_x, bio_sigmas, color=bio_colors, width=0.7,
                 edgecolor="black", linewidth=0.5, alpha=0.6)

        # Labels
        all_x = list(x_model) + bio_x
        all_labels = labels + bio_labels
        ax_b.set_xticks(all_x)
        ax_b.set_xticklabels(all_labels, rotation=40, ha="right", fontsize=8)

        # Separator line between model and biology
        ax_b.axvline(n_model - 0.15, color="gray", ls=":", lw=0.8)
        ax_b.text(n_model / 2 - 0.5, ax_b.get_ylim()[1] * 0.95, "Model",
                  ha="center", fontsize=9, fontstyle="italic", color="gray")
        ax_b.text(n_model + 1.5, ax_b.get_ylim()[1] * 0.95, "Biology",
                  ha="center", fontsize=9, fontstyle="italic", color="gray")

    ax_b.set_ylabel("Log-normal σ", fontsize=11)
    ax_b.set_title("B. Distribution width (σ)", fontsize=12, fontweight="bold")

    # ── Panel C: E vs I comparison with biology ──
    ax_c = fig.add_subplot(gs[0, 2])

    # For each condition, plot exc σ and inh σ side by side
    inh_stats = compute_distribution_stats(conditions, "inh_layer0")
    conds_both = [c for c in CONDITION_ORDER if c in cond_stats and c in inh_stats]

    if conds_both:
        n_c = len(conds_both)
        x_c = np.arange(n_c)
        width = 0.35

        exc_sigmas = [cond_stats[c]["lognormal_sigma"][0] for c in conds_both]
        exc_errs = [cond_stats[c]["lognormal_sigma"][1] for c in conds_both]
        inh_sigmas = [inh_stats[c]["lognormal_sigma"][0] for c in conds_both]
        inh_errs = [inh_stats[c]["lognormal_sigma"][1] for c in conds_both]

        ax_c.bar(x_c - width / 2, exc_sigmas, width, yerr=exc_errs,
                 color="#EE6677", alpha=0.7, capsize=3, edgecolor="black",
                 linewidth=0.5, label="Excitatory")
        ax_c.bar(x_c + width / 2, inh_sigmas, width, yerr=inh_errs,
                 color="#4477AA", alpha=0.7, capsize=3, edgecolor="black",
                 linewidth=0.5, label="Inhibitory")

        # MICrONS reference lines
        if "EE" in microns_stats:
            ax_c.axhline(microns_stats["EE"]["mean_size_lognormal_sigma"],
                         color="#EE6677", ls="--", lw=1.5, alpha=0.6,
                         label=f"MICrONS E→E (σ={microns_stats['EE']['mean_size_lognormal_sigma']:.2f})")
        if "IE" in microns_stats:
            ax_c.axhline(microns_stats["IE"]["mean_size_lognormal_sigma"],
                         color="#4477AA", ls="--", lw=1.5, alpha=0.6,
                         label=f"MICrONS I→E (σ={microns_stats['IE']['mean_size_lognormal_sigma']:.2f})")

        labels_c = [CONDITION_LABELS.get(c, c) for c in conds_both]
        ax_c.set_xticks(x_c)
        ax_c.set_xticklabels(labels_c, rotation=40, ha="right", fontsize=8)
        ax_c.legend(fontsize=7, loc="upper right")

    ax_c.set_ylabel("Log-normal σ", fontsize=11)
    ax_c.set_title("C. Exc vs Inh σ (model ↔ biology)", fontsize=12, fontweight="bold")

    fig.suptitle(
        f"Synaptic Weight Distributions: Model vs Biology ({dataset_label})",
        fontsize=14, fontweight="bold",
    )

    out_path = os.path.join(output_dir, f"figure_weight_vs_biology_{dataset_label.lower()}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved figure to {out_path}")
    plt.close()

    return out_path


def plot_synapse_count_comparison(
    conditions: dict,
    microns_data: dict,
    output_dir: str,
):
    """Supplementary figure: synapse count per connection distribution.

    Shows MICrONS multi-synapse fractions by E/I type.
    """
    microns_stats = microns_data.get("stats", {})
    if not microns_stats:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: Multi-synapse fractions by quadrant
    quads = ["EE", "IE", "EI", "II"]
    quad_labels = ["E→E", "I→E", "E→I", "I→I"]
    quad_colors = ["#EE6677", "#4477AA", "#CCBB44", "#66CCEE"]

    frac_multi = []
    count_means = []
    for q in quads:
        if q in microns_stats:
            frac_multi.append(microns_stats[q]["frac_multi_synapse"])
            count_means.append(microns_stats[q]["count_mean"])
        else:
            frac_multi.append(0)
            count_means.append(0)

    ax1.bar(range(4), frac_multi, color=quad_colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(quad_labels)
    ax1.set_ylabel("Fraction multi-synaptic")
    ax1.set_title("A. Multi-synapse connections (MICrONS)")

    for i, (f, m) in enumerate(zip(frac_multi, count_means)):
        ax1.text(i, f + 0.01, f"mean={m:.1f}", ha="center", fontsize=9)

    # Panel 2: σ comparison across quadrants + model
    cond_stats = compute_distribution_stats(conditions, "exc_layer0")
    inh_stats = compute_distribution_stats(conditions, "inh_layer0")

    # Collect all σ values
    bar_labels = []
    bar_sigmas = []
    bar_colors = []

    # Model conditions
    for cond in CONDITION_ORDER:
        if cond in cond_stats:
            bar_labels.append(CONDITION_LABELS.get(cond, cond) + "\n(exc)")
            bar_sigmas.append(cond_stats[cond]["lognormal_sigma"][0])
            bar_colors.append(CONDITION_COLORS.get(cond, "gray"))
        if cond in inh_stats:
            bar_labels.append(CONDITION_LABELS.get(cond, cond) + "\n(inh)")
            bar_sigmas.append(inh_stats[cond]["lognormal_sigma"][0])
            bar_colors.append(CONDITION_COLORS.get(cond, "gray"))

    # Biology
    for q, ql in zip(quads, quad_labels):
        if q in microns_stats:
            bar_labels.append(f"MICrONS\n{ql}")
            bar_sigmas.append(microns_stats[q]["mean_size_lognormal_sigma"])
            bar_colors.append("#888888")

    bar_labels.append("Song\n2005")
    bar_sigmas.append(SONG_2005_SIGMA)
    bar_colors.append("black")

    x = range(len(bar_labels))
    ax2.bar(x, bar_sigmas, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=7)
    ax2.set_ylabel("Log-normal σ")
    ax2.set_title("B. Comprehensive σ comparison")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure_synapse_count_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info(f"Saved supplementary figure to {out_path}")
    plt.close()


def print_summary_table(conditions: dict, microns_data: dict):
    """Print comprehensive comparison table."""
    microns_stats = microns_data.get("stats", {})

    print("\n" + "=" * 90)
    print("MODEL vs BIOLOGY: SYNAPTIC WEIGHT DISTRIBUTION COMPARISON")
    print("=" * 90)
    print(f"{'Source':35s} {'Type':>8s} {'LN σ':>10s} {'CV':>8s} {'N':>12s}")
    print("-" * 90)

    # Model results
    exc_stats = compute_distribution_stats(conditions, "exc_layer0")
    inh_stats = compute_distribution_stats(conditions, "inh_layer0")

    for cond in CONDITION_ORDER:
        label = CONDITION_LABELS.get(cond, cond)
        if cond in exc_stats:
            s = exc_stats[cond]
            print(f"{label:35s} {'exc':>8s} "
                  f"{s['lognormal_sigma'][0]:>8.3f}±{s['lognormal_sigma'][1]:.3f} "
                  f"{s['cv'][0]:>6.3f}±{s['cv'][1]:.3f} "
                  f"{'~46K':>12s}")
        if cond in inh_stats:
            s = inh_stats[cond]
            print(f"{'':35s} {'inh':>8s} "
                  f"{s['lognormal_sigma'][0]:>8.3f}±{s['lognormal_sigma'][1]:.3f} "
                  f"{s['cv'][0]:>6.3f}±{s['cv'][1]:.3f} "
                  f"{'~8K':>12s}")

    print("-" * 90)

    # Biology
    for quad, name in [("EE", "MICrONS E→E"), ("IE", "MICrONS I→E"),
                       ("EI", "MICrONS E→I"), ("II", "MICrONS I→I")]:
        if quad in microns_stats:
            s = microns_stats[quad]
            print(f"{name:35s} {'syn':>8s} "
                  f"{s['mean_size_lognormal_sigma']:>10.3f} "
                  f"{s['mean_size_cv']:>8.3f} "
                  f"{s['n_connections']:>12,d}")

    print(f"{'Song et al. 2005 (rat V1 L5)':35s} {'EPSP':>8s} "
          f"{SONG_2005_SIGMA:>10.3f} "
          f"{'':>8s} "
          f"{'931':>12s}")

    print("=" * 90)

    # Key comparisons
    print("\nKey comparisons:")
    if "standard_dendritic_shunting" in exc_stats and "EE" in microns_stats:
        bp_s = exc_stats["standard_dendritic_shunting"]["lognormal_sigma"][0]
        ee_s = microns_stats["EE"]["mean_size_lognormal_sigma"]
        print(f"  BP+Shunting σ = {bp_s:.3f}  vs  MICrONS E→E σ = {ee_s:.3f}  "
              f"(Δ = {abs(bp_s - ee_s):.3f})")

    if "local_ca_dendritic_shunting" in exc_stats and "EE" in microns_stats:
        loc_s = exc_stats["local_ca_dendritic_shunting"]["lognormal_sigma"][0]
        ee_s = microns_stats["EE"]["mean_size_lognormal_sigma"]
        print(f"  Local+Shunting σ = {loc_s:.3f}  vs  MICrONS E→E σ = {ee_s:.3f}  "
              f"(Δ = {abs(loc_s - ee_s):.3f})")

    if "standard_dendritic_shunting" in exc_stats:
        bp_s = exc_stats["standard_dendritic_shunting"]["lognormal_sigma"][0]
        print(f"  BP+Shunting σ = {bp_s:.3f}  vs  Song 2005 σ = {SONG_2005_SIGMA:.3f}  "
              f"(Δ = {abs(bp_s - SONG_2005_SIGMA):.3f})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate model vs biology weight distribution comparison figure"
    )
    parser.add_argument(
        "--sweep-dirs", nargs="+", required=True,
        help="Paths to sweep result directories with trained models",
    )
    parser.add_argument(
        "--microns-dir",
        default="drafts/dendritic-local-learning/data/microns_distributions",
        help="Directory with MICrONS analysis results",
    )
    parser.add_argument(
        "--output-dir",
        default="drafts/dendritic-local-learning/data/weight_distributions",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--filter-dataset", type=str, default=None,
        help="Filter by dataset name (e.g. 'mnist')",
    )
    parser.add_argument(
        "--filter-noise-rate", type=float, default=None,
        help="Filter by label noise rate (e.g. 0.0)",
    )
    parser.add_argument(
        "--max-seeds", type=int, default=5,
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load MICrONS data
    microns_data = load_microns_data(args.microns_dir)

    # Build filter
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

    # Extract model weights
    all_conditions = {}
    for sweep_dir in args.sweep_dirs:
        logger.info(f"Processing sweep: {sweep_dir}")
        conds = extract_weights_from_sweep(sweep_dir, filter_fn=filter_fn, max_seeds=args.max_seeds)
        for k, v in conds.items():
            if k not in all_conditions:
                all_conditions[k] = []
            all_conditions[k].extend(v)

    if not all_conditions:
        logger.error("No matching conditions found!")
        return

    logger.info(f"Loaded conditions: {list(all_conditions.keys())}")

    # Dataset label
    ds_label = args.filter_dataset.upper() if args.filter_dataset else "All"

    # Generate figures
    plot_model_vs_biology(all_conditions, microns_data, args.output_dir, ds_label)
    plot_synapse_count_comparison(all_conditions, microns_data, args.output_dir)

    # Print summary
    print_summary_table(all_conditions, microns_data)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate publication-quality NeurIPS figures for the local credit assignment paper.

Creates:
  - fig_neurips_combined.pdf: 4-panel main figure (bar + curves + grad fidelity + fashion)
  - fig_main_results.pdf: standalone bar chart
  - fig_learning_curves.pdf: multi-panel learning curves
  - fig_gradient_fidelity_trajectory.pdf: gradient alignment over training
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# NeurIPS style
# ---------------------------------------------------------------------------
NEURIPS_FULL = 5.5  # inches (NeurIPS text width)
DPI = 300

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "lines.linewidth": 1.2,
    "lines.markersize": 4,
    "patch.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Consistent color palette
C_BP = "#2C3E50"     # dark navy - backprop
C_5F = "#27AE60"     # green - 5F local
C_4F = "#8E44AD"     # purple - 4F local
C_3F = "#E67E22"     # orange - 3F local
C_FA = "#E74C3C"     # red - FA
C_DFA = "#3498DB"    # blue - DFA
C_SHUNT = "#27AE60"  # green - shunting
C_ADD = "#E74C3C"    # red - additive
C_MLP = "#3498DB"    # blue - MLP

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SWEEP_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)
ANALYSIS_DIR = Path(
    "/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/"
    "drafts/dendritic-local-learning/analysis"
)
OUT_DIR = Path(
    "/n/holylabs/LABS/kempner_dev/Users/hsafaai/Code/dendritic-modeling/"
    "drafts/dendritic-local-learning/figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def _dig(d: dict, keys: list[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _get_rule_variant(cfg: dict) -> str:
    """Extract rule_variant from config, checking multiple locations."""
    # Primary: learning_strategy_config.rule_variant
    rv = _dig(cfg, ["training", "main", "learning_strategy_config", "rule_variant"], "")
    if rv:
        return rv
    # Fallback: common.local_learning.rule_variant
    rv = _dig(cfg, ["training", "main", "common", "local_learning", "rule_variant"], "")
    return rv


def _get_broadcast_mode(cfg: dict) -> str:
    """Extract error_broadcast_mode from config."""
    return _dig(cfg, ["training", "main", "learning_strategy_config",
                       "error_broadcast_mode"], "")


def extract_final_accuracies(sweep_name: str) -> pd.DataFrame:
    """Read final test accuracy from each config in a sweep."""
    results_dir = SWEEP_ROOT / sweep_name / "results"
    if not results_dir.exists():
        return pd.DataFrame()

    rows = []
    for config_dir in sorted(results_dir.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
            continue
        perf_file = config_dir / "performance" / "final.json"
        cfg_file = config_dir / "config.json"
        if not perf_file.exists() or not cfg_file.exists():
            continue
        try:
            perf = json.loads(perf_file.read_text())
            cfg = json.loads(cfg_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        test_acc = _dig(perf, ["accuracy", "test"])
        if test_acc is None:
            continue
        rows.append({
            "sweep": sweep_name,
            "config_id": config_dir.name,
            "core_type": _dig(cfg, ["model", "core", "type"], "unknown"),
            "strategy": _dig(cfg, ["training", "main", "strategy"], "unknown"),
            "rule_variant": _get_rule_variant(cfg),
            "broadcast": _get_broadcast_mode(cfg),
            "dataset": _dig(cfg, ["data", "dataset_name"], "mnist"),
            "seed": _dig(cfg, ["experiment", "seed"], 0),
            "test_acc": test_acc,
            "train_acc": _dig(perf, ["accuracy", "train"]),
            "valid_acc": _dig(perf, ["accuracy", "valid"]),
        })
    return pd.DataFrame(rows)


def extract_learning_curves(sweep_name: str) -> pd.DataFrame:
    """Read per-epoch accuracy from each config in a sweep."""
    results_dir = SWEEP_ROOT / sweep_name / "results"
    if not results_dir.exists():
        return pd.DataFrame()

    rows = []
    for config_dir in sorted(results_dir.iterdir()):
        if not config_dir.is_dir() or not config_dir.name.startswith("config_"):
            continue
        epochs_dir = config_dir / "performance" / "epochs"
        cfg_file = config_dir / "config.json"
        if not epochs_dir.exists() or not cfg_file.exists():
            continue
        try:
            cfg = json.loads(cfg_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue

        meta = {
            "sweep": sweep_name,
            "config_id": config_dir.name,
            "core_type": _dig(cfg, ["model", "core", "type"], "unknown"),
            "strategy": _dig(cfg, ["training", "main", "strategy"], "unknown"),
            "rule_variant": _get_rule_variant(cfg),
            "broadcast": _get_broadcast_mode(cfg),
            "dataset": _dig(cfg, ["data", "dataset_name"], "mnist"),
            "seed": _dig(cfg, ["experiment", "seed"], 0),
        }

        for epoch_file in sorted(epochs_dir.glob("epoch*.json")):
            match = re.search(r"epoch(\d+)", epoch_file.stem)
            if not match:
                continue
            epoch = int(match.group(1))
            try:
                ep = json.loads(epoch_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            for split in ("train", "test"):
                acc = _dig(ep, ["accuracy", split])
                if acc is not None:
                    rows.append({**meta, "epoch": epoch, "split": split,
                                 "accuracy": acc})
    return pd.DataFrame(rows)


def _label(row):
    """Create a human-readable method label."""
    if row["strategy"] == "standard":
        return "Backprop"
    elif row["strategy"] == "fa":
        return "FA"
    elif row["strategy"] == "dfa":
        return "DFA"
    elif row["strategy"] == "local_ca":
        rv = row.get("rule_variant", "")
        return f"Local {rv.upper()}" if rv else "Local"
    return row["strategy"]


def panel_label(ax, text, x=-0.14, y=1.06):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="top", ha="left")


def save(fig, name):
    for fmt in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{name}.{fmt}")
    plt.close(fig)
    print(f"  -> {OUT_DIR / name}.pdf")


# ===================================================================
# FIGURE: Main results bar chart
# ===================================================================
def figure_main_results():
    """Bar chart: backprop vs local 5F/3F vs DFA across architectures on MNIST."""
    print("\n=== Main Results Bar Chart ===")

    # Load data from all relevant sweeps
    df_fadfa = extract_final_accuracies("sweep_neurips_fa_dfa_baselines_20260220000705")
    df_local = extract_final_accuracies("sweep_neurips_mlp_local_learning_20260219163529")

    dfs = [d for d in [df_fadfa, df_local] if not d.empty]
    if not dfs:
        print("  No data!"); return
    df = pd.concat(dfs, ignore_index=True)
    df["method"] = df.apply(_label, axis=1)

    # For local_ca with dendritic models, filter to per_soma broadcast (best mode)
    mask_local_dend = (df["strategy"] == "local_ca") & (df["core_type"] != "point_mlp")
    mask_per_soma = df["broadcast"] == "per_soma"
    df = df[~mask_local_dend | mask_per_soma]

    arch_map = {"dendritic_shunting": "Shunting", "dendritic_additive": "Additive",
                "point_mlp": "MLP"}
    df["arch"] = df["core_type"].map(arch_map)

    summary = df.groupby(["method", "arch"])["test_acc"].agg(
        ["mean", "std", "count"]).reset_index()
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])

    method_order = ["Backprop", "Local 5F", "Local 3F", "DFA"]
    colors = {"Backprop": C_BP, "Local 5F": C_5F, "Local 3F": C_3F,
              "DFA": C_DFA, "FA": C_FA}
    arch_order = ["Shunting", "Additive", "MLP"]

    exist_m = [m for m in method_order if m in summary["method"].values]
    exist_a = [a for a in arch_order if a in summary["arch"].values]
    n_m, n_a = len(exist_m), len(exist_a)

    fig, ax = plt.subplots(figsize=(NEURIPS_FULL, 2.6))
    bw = 0.8 / n_m
    x = np.arange(n_a)

    for i, meth in enumerate(exist_m):
        vals, errs = [], []
        for arch in exist_a:
            row = summary[(summary["method"] == meth) & (summary["arch"] == arch)]
            if not row.empty:
                vals.append(row["mean"].iloc[0])
                errs.append(row["ci95"].iloc[0])
            else:
                vals.append(0); errs.append(0)
        off = (i - n_m / 2 + 0.5) * bw
        ax.bar(x + off, vals, bw * 0.85, yerr=errs, label=meth,
               color=colors.get(meth, "#999"), edgecolor="white", linewidth=0.3,
               capsize=2, error_kw={"linewidth": 0.7})
        for j, (v, e) in enumerate(zip(vals, errs)):
            if v > 0.03:
                ax.text(x[j] + off, v + e + 0.012, f"{v:.2f}", ha="center",
                        va="bottom", fontsize=5.5)

    ax.set_xticks(x); ax.set_xticklabels(exist_a, fontsize=8)
    ax.set_ylabel("Test Accuracy"); ax.set_ylim(0, 1.02)
    ax.set_title("MNIST: Training Strategy Comparison (Matched Architecture)",
                 fontsize=9, fontweight="bold", pad=8)
    ax.legend(framealpha=0.9, edgecolor="none", ncol=2, loc="upper right")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.grid(axis="y", alpha=0.25, linewidth=0.3); ax.set_axisbelow(True)

    save(fig, "fig_main_results")


# ===================================================================
# FIGURE: Learning Curves
# ===================================================================
def figure_learning_curves():
    """Multi-panel learning curves."""
    print("\n=== Learning Curves ===")

    sweeps = [
        ("sweep_neurips_fa_dfa_baselines_20260220000705", "MNIST (Backprop/DFA)"),
        ("sweep_neurips_mlp_local_learning_20260219163529", "MNIST (Local Rules)"),
        ("sweep_neurips_fashion_mnist_20260219163051", "Fashion-MNIST (Local Rules)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(NEURIPS_FULL, 2.3))

    for pidx, (sweep_name, title) in enumerate(sweeps):
        ax = axes[pidx]
        df = extract_learning_curves(sweep_name)
        if df.empty:
            ax.set_title(title, fontsize=7, fontweight="bold")
            panel_label(ax, chr(65 + pidx))
            continue

        df_t = df[df["split"] == "test"].copy()
        df_t["method"] = df_t.apply(_label, axis=1)
        arch_map = {"dendritic_shunting": "Shunting", "dendritic_additive": "Additive",
                    "point_mlp": "MLP"}
        df_t["arch"] = df_t["core_type"].map(arch_map)

        # For local_ca, filter to per_soma broadcast for dendritic architectures
        if "local_ca" in df_t["strategy"].values:
            mask_local = (df_t["strategy"] == "local_ca") & (df_t["arch"] != "MLP")
            mask_ps = df_t["broadcast"] == "per_soma"
            df_t = df_t[~mask_local | mask_ps]

        # Group and plot
        conditions = df_t.groupby(["method", "arch"]).size().reset_index()
        colors = {"Backprop": C_BP, "Local 5F": C_5F, "Local 3F": C_3F,
                  "Local 4F": C_4F, "DFA": C_DFA, "FA": C_FA}
        styles = {"Backprop": "-", "Local 5F": "-", "Local 3F": "--",
                  "Local 4F": ":", "DFA": "-.", "FA": ":"}
        markers = {"Shunting": "o", "Additive": "s", "MLP": "^"}

        for _, cond in conditions.iterrows():
            meth, arch = cond["method"], cond["arch"]
            mask = (df_t["method"] == meth) & (df_t["arch"] == arch)
            grp = df_t[mask].groupby("epoch")["accuracy"].agg(
                ["mean", "std"]).reset_index().sort_values("epoch")
            if grp.empty:
                continue
            c = colors.get(meth, "#999")
            lbl = f"{meth} ({arch[:3]})" if arch != "MLP" else f"{meth} (MLP)"
            ax.plot(grp["epoch"], grp["mean"], color=c,
                    linestyle=styles.get(meth, "-"),
                    marker=markers.get(arch, "o"), markersize=2,
                    markevery=max(1, len(grp) // 5), label=lbl)
            if grp["std"].max() > 0:
                ax.fill_between(grp["epoch"],
                                grp["mean"] - grp["std"],
                                grp["mean"] + grp["std"],
                                alpha=0.1, color=c)

        ax.set_title(title, fontsize=7, fontweight="bold")
        ax.set_xlabel("Epoch")
        if pidx == 0:
            ax.set_ylabel("Test Accuracy")
        ax.legend(fontsize=4.5, loc="lower right", framealpha=0.9, edgecolor="none",
                  ncol=1)
        ax.grid(alpha=0.2, linewidth=0.3); ax.set_axisbelow(True)
        panel_label(ax, chr(65 + pidx))

    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_learning_curves")


# ===================================================================
# FIGURE: Gradient Fidelity Over Training
# ===================================================================
def figure_gradient_fidelity():
    """Gradient fidelity trajectory: cosine similarity + scale mismatch."""
    print("\n=== Gradient Fidelity Trajectory ===")

    grad_dir = ANALYSIS_DIR / "gradient_fidelity"
    if not grad_dir.exists():
        print("  Not found!"); return

    core_map = {
        "config_0": "Shunting", "config_1": "Shunting", "config_2": "Shunting",
        "config_3": "Additive", "config_4": "Additive", "config_5": "Additive",
    }

    dfs = []
    for cname, ctype in core_map.items():
        p = grad_dir / cname / "gradient_fidelity_trajectory.csv"
        if p.exists():
            d = pd.read_csv(p)
            d["config"] = cname
            d["core_type"] = ctype
            dfs.append(d)
    if not dfs:
        print("  No data!"); return

    df = pd.concat(dfs, ignore_index=True)

    # 3-panel: A) weighted cosine by rule, B) per-component bars, C) scale mismatch
    fig, axes = plt.subplots(1, 3, figsize=(NEURIPS_FULL, 2.2))

    # --- Panel A: Weighted cosine sim over epochs for each rule × core type ---
    ax = axes[0]
    rule_colors = {"5f": C_5F, "3f": C_3F, "4f": C_4F}
    rule_styles = {"5f": "-", "3f": "--", "4f": ":"}

    for rule in ["5f", "3f", "4f"]:
        for ctype, ccolor, mk in [("Shunting", C_SHUNT, "o"),
                                    ("Additive", C_ADD, "s")]:
            sub = df[(df["rule_variant"] == rule) & (df["core_type"] == ctype)]
            if sub.empty:
                continue
            epoch_vals = []
            for epoch in sorted(sub["epoch"].unique()):
                de = sub[sub["epoch"] == epoch]
                per_seed = []
                for cfg in de["config"].unique():
                    dc = de[de["config"] == cfg]
                    wcos = (dc["cosine_similarity"] * dc["numel"]).sum() / dc["numel"].sum()
                    per_seed.append(wcos)
                epoch_vals.append({"epoch": epoch,
                                   "mean": np.mean(per_seed),
                                   "std": np.std(per_seed) if len(per_seed) > 1 else 0})
            ev = pd.DataFrame(epoch_vals)
            alpha = 1.0 if ctype == "Shunting" else 0.4
            color = rule_colors[rule]
            ax.plot(ev["epoch"], ev["mean"], color=color, alpha=alpha,
                    linestyle=rule_styles[rule], marker=mk, markersize=2,
                    markevery=max(1, len(ev) // 6),
                    label=f"{rule.upper()} ({ctype[:3]})")
            if ev["std"].max() > 0:
                ax.fill_between(ev["epoch"], ev["mean"] - ev["std"],
                                ev["mean"] + ev["std"], alpha=0.07, color=color)

    ax.set_xlabel("Epoch"); ax.set_ylabel("Weighted Cosine Sim.")
    ax.set_title("Gradient Alignment", fontsize=8, fontweight="bold")
    ax.legend(fontsize=4.5, loc="upper right", framealpha=0.9, edgecolor="none",
              ncol=2)
    ax.grid(alpha=0.2, linewidth=0.3); ax.set_axisbelow(True)
    ax.axhline(y=0, color="gray", linewidth=0.4, linestyle="-")
    panel_label(ax, "A")

    # --- Panel B: Per-component cosine at epoch 0 (shunting, 5F) ---
    ax = axes[1]
    comp_order = ["excitatory_synapse", "inhibitory_synapse",
                  "dendritic_conductance", "reactivation"]
    comp_labels = ["Excitatory", "Inhibitory", "Dendritic", "Reactivation"]

    sub_5f = df[(df["rule_variant"] == "5f")]
    epochs_avail = sorted(sub_5f["epoch"].unique())
    if epochs_avail:
        ep0, epf = epochs_avail[0], epochs_avail[-1]
        for idx_ct, (ctype, ccolor, hatch) in enumerate([
                ("Shunting", C_SHUNT, None), ("Additive", C_ADD, "//")]):
            vals = []
            for comp in comp_order:
                dc = sub_5f[(sub_5f["core_type"] == ctype) &
                            (sub_5f["component"] == comp) &
                            (sub_5f["epoch"] == ep0)]
                if dc.empty:
                    vals.append(0)
                else:
                    wcos = (dc["cosine_similarity"] * dc["numel"]).sum() / dc["numel"].sum()
                    vals.append(wcos)

            x_pos = np.arange(len(comp_order))
            w = 0.35
            off = (idx_ct - 0.5) * w
            ax.bar(x_pos + off, vals, w, color=ccolor, alpha=0.7, hatch=hatch,
                   edgecolor="white" if hatch is None else ccolor, linewidth=0.3,
                   label=ctype)

        ax.set_xticks(np.arange(len(comp_order)))
        ax.set_xticklabels(comp_labels, fontsize=5.5, rotation=25, ha="right")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Per-Component (5F, Epoch 0)", fontsize=8, fontweight="bold")
        ax.legend(fontsize=6, loc="upper right", framealpha=0.9, edgecolor="none")
        ax.grid(axis="y", alpha=0.2, linewidth=0.3); ax.set_axisbelow(True)
    panel_label(ax, "B")

    # --- Panel C: Scale mismatch over epochs ---
    ax = axes[2]
    for rule in ["5f", "3f"]:
        for ctype, mk in [("Shunting", "o"), ("Additive", "s")]:
            sub = df[(df["rule_variant"] == rule) & (df["core_type"] == ctype)]
            if sub.empty:
                continue
            sub = sub.copy()
            sub["abs_log_ratio"] = np.abs(np.log10(sub["norm_ratio"].clip(1e-10)))
            sub["w_lr"] = sub["abs_log_ratio"] * sub["numel"]

            epoch_vals = []
            for epoch in sorted(sub["epoch"].unique()):
                de = sub[sub["epoch"] == epoch]
                per_seed = []
                for cfg in de["config"].unique():
                    dc = de[de["config"] == cfg]
                    wlr = dc["w_lr"].sum() / dc["numel"].sum()
                    per_seed.append(wlr)
                epoch_vals.append({"epoch": epoch, "mean": np.mean(per_seed),
                                   "std": np.std(per_seed) if len(per_seed) > 1 else 0})
            ev = pd.DataFrame(epoch_vals)
            color = rule_colors[rule]
            alpha = 1.0 if ctype == "Shunting" else 0.4
            ax.plot(ev["epoch"], ev["mean"], color=color, alpha=alpha,
                    linestyle=rule_styles[rule], marker=mk, markersize=2,
                    markevery=max(1, len(ev) // 6),
                    label=f"{rule.upper()} ({ctype[:3]})")

    ax.set_xlabel("Epoch"); ax.set_ylabel(r"|log$_{10}$(norm ratio)|")
    ax.set_title("Scale Mismatch", fontsize=8, fontweight="bold")
    ax.legend(fontsize=4.5, loc="upper right", framealpha=0.9, edgecolor="none",
              ncol=2)
    ax.grid(alpha=0.2, linewidth=0.3); ax.set_axisbelow(True)
    panel_label(ax, "C")

    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_gradient_fidelity_trajectory")


# ===================================================================
# FIGURE: Fashion-MNIST
# ===================================================================
def figure_fashion_mnist():
    """Fashion-MNIST local learning results + MNIST comparison."""
    print("\n=== Fashion-MNIST ===")

    df_fm = extract_final_accuracies("sweep_neurips_fashion_mnist_20260219163051")
    df_mn = extract_final_accuracies("sweep_neurips_mlp_local_learning_20260219163529")

    if df_fm.empty:
        print("  No Fashion-MNIST data!"); return

    df_fm["method"] = df_fm.apply(_label, axis=1)
    arch_map = {"dendritic_shunting": "Shunting", "dendritic_additive": "Additive",
                "point_mlp": "MLP"}
    df_fm["arch"] = df_fm["core_type"].map(arch_map)

    fig, axes = plt.subplots(1, 2, figsize=(NEURIPS_FULL, 2.5))

    # --- Panel A: Fashion-MNIST bar chart by method × arch (per_soma only for local) ---
    ax = axes[0]
    # Filter to per_soma broadcast for local rules
    df_ps = df_fm[(df_fm["broadcast"] == "per_soma") | (df_fm["strategy"] != "local_ca")]
    summary = df_ps.groupby(["method", "arch"])["test_acc"].agg(
        ["mean", "std", "count"]).reset_index()
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])

    method_order = ["Local 5F", "Local 4F", "Local 3F"]
    colors = {"Local 5F": C_5F, "Local 4F": C_4F, "Local 3F": C_3F}
    arch_order = ["Shunting", "Additive"]
    exist_m = [m for m in method_order if m in summary["method"].values]
    exist_a = [a for a in arch_order if a in summary["arch"].values]

    n_m, n_a = len(exist_m), len(exist_a)
    bw = 0.75 / max(n_m, 1)
    x = np.arange(n_a)

    for i, meth in enumerate(exist_m):
        vals, errs = [], []
        for arch in exist_a:
            row = summary[(summary["method"] == meth) & (summary["arch"] == arch)]
            if not row.empty:
                vals.append(row["mean"].iloc[0])
                errs.append(row["ci95"].iloc[0])
            else:
                vals.append(0); errs.append(0)
        off = (i - n_m / 2 + 0.5) * bw
        ax.bar(x + off, vals, bw * 0.85, yerr=errs, label=meth,
               color=colors.get(meth, "#999"), edgecolor="white", linewidth=0.3,
               capsize=2, error_kw={"linewidth": 0.7})
        for j, (v, e) in enumerate(zip(vals, errs)):
            if v > 0.03:
                ax.text(x[j] + off, v + e + 0.008, f"{v:.3f}", ha="center",
                        va="bottom", fontsize=5)

    ax.set_xticks(x); ax.set_xticklabels(exist_a, fontsize=7)
    ax.set_ylabel("Test Accuracy"); ax.set_ylim(0, 0.55)
    ax.set_title("Fashion-MNIST: Local Rules\n(per-soma broadcast)", fontsize=8,
                 fontweight="bold")
    ax.legend(fontsize=6, framealpha=0.9, edgecolor="none")
    ax.grid(axis="y", alpha=0.2, linewidth=0.3); ax.set_axisbelow(True)
    panel_label(ax, "A")

    # --- Panel B: MNIST vs Fashion-MNIST (shunting, per_soma) ---
    ax = axes[1]
    if not df_mn.empty:
        df_mn["method"] = df_mn.apply(_label, axis=1)
        df_mn["arch"] = df_mn["core_type"].map(arch_map)

        # Shunting + per_soma only
        df_mn_s = df_mn[(df_mn["arch"] == "Shunting") & (df_mn["broadcast"] == "per_soma")]
        df_fm_s = df_fm[(df_fm["arch"] == "Shunting") & (df_fm["broadcast"] == "per_soma")]

        methods_compare = sorted(set(df_mn_s["method"].unique()) &
                                 set(df_fm_s["method"].unique()))
        if methods_compare:
            x_pos = np.arange(len(methods_compare))
            w = 0.35
            for i, meth in enumerate(methods_compare):
                mn_vals = df_mn_s[df_mn_s["method"] == meth]["test_acc"]
                fm_vals = df_fm_s[df_fm_s["method"] == meth]["test_acc"]
                ax.bar(i - w / 2, mn_vals.mean(), w,
                       yerr=1.96 * mn_vals.std() / np.sqrt(len(mn_vals)) if len(mn_vals) > 1 else 0,
                       color=C_BP, alpha=0.7, edgecolor="white", linewidth=0.3,
                       capsize=2, error_kw={"linewidth": 0.7},
                       label="MNIST" if i == 0 else "")
                ax.bar(i + w / 2, fm_vals.mean(), w,
                       yerr=1.96 * fm_vals.std() / np.sqrt(len(fm_vals)) if len(fm_vals) > 1 else 0,
                       color=C_DFA, alpha=0.7, edgecolor="white", linewidth=0.3,
                       capsize=2, error_kw={"linewidth": 0.7},
                       label="Fashion-MNIST" if i == 0 else "")
            ax.set_xticks(x_pos); ax.set_xticklabels(methods_compare, fontsize=7)
            ax.legend(fontsize=6, framealpha=0.9, edgecolor="none")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("MNIST vs Fashion-MNIST\n(Shunting, per-soma)", fontsize=8,
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linewidth=0.3); ax.set_axisbelow(True)
    panel_label(ax, "B")

    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_fashion_mnist_results")


# ===================================================================
# COMBINED 4-panel figure
# ===================================================================
def figure_combined():
    """Main 4-panel figure for the paper."""
    print("\n=== Combined 4-Panel Figure ===")

    # Load all data
    df_fadfa = extract_final_accuracies("sweep_neurips_fa_dfa_baselines_20260220000705")
    df_local = extract_final_accuracies("sweep_neurips_mlp_local_learning_20260219163529")
    df_fmnist = extract_final_accuracies("sweep_neurips_fashion_mnist_20260219163051")

    # Merge MNIST sweeps
    dfs = [d for d in [df_fadfa, df_local] if not d.empty]
    if not dfs:
        print("  No data!"); return
    df_mnist = pd.concat(dfs, ignore_index=True)
    df_mnist["method"] = df_mnist.apply(_label, axis=1)
    arch_map = {"dendritic_shunting": "Shunting", "dendritic_additive": "Additive",
                "point_mlp": "MLP"}
    df_mnist["arch"] = df_mnist["core_type"].map(arch_map)

    # Filter local_ca to per_soma broadcast for dendritic models
    mask_local = (df_mnist["strategy"] == "local_ca") & (df_mnist["arch"] != "MLP")
    mask_ps = df_mnist["broadcast"] == "per_soma"
    df_mnist = df_mnist[~mask_local | mask_ps]

    fig = plt.figure(figsize=(NEURIPS_FULL, 5.0))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.38,
                          left=0.10, right=0.97, top=0.94, bottom=0.07)

    # ---- Panel A: Bar chart ----
    ax_a = fig.add_subplot(gs[0, 0])
    summary = df_mnist.groupby(["method", "arch"])["test_acc"].agg(
        ["mean", "std", "count"]).reset_index()
    summary["ci95"] = 1.96 * summary["std"] / np.sqrt(summary["count"])

    method_order = ["Backprop", "Local 5F", "Local 3F", "DFA"]
    colors = {"Backprop": C_BP, "Local 5F": C_5F, "Local 3F": C_3F, "DFA": C_DFA}
    arch_order = ["Shunting", "Additive", "MLP"]
    exist_m = [m for m in method_order if m in summary["method"].values]
    exist_a = [a for a in arch_order if a in summary["arch"].values]
    n_m, n_a = len(exist_m), len(exist_a)
    bw = 0.8 / max(n_m, 1)
    x = np.arange(n_a)

    for i, meth in enumerate(exist_m):
        vals, errs = [], []
        for arch in exist_a:
            row = summary[(summary["method"] == meth) & (summary["arch"] == arch)]
            if not row.empty:
                vals.append(row["mean"].iloc[0]); errs.append(row["ci95"].iloc[0])
            else:
                vals.append(0); errs.append(0)
        off = (i - n_m / 2 + 0.5) * bw
        ax_a.bar(x + off, vals, bw * 0.85, yerr=errs, label=meth,
                 color=colors.get(meth, "#999"), edgecolor="white", linewidth=0.3,
                 capsize=1.5, error_kw={"linewidth": 0.6})
        for j, (v, e) in enumerate(zip(vals, errs)):
            if v > 0.05:
                ax_a.text(x[j] + off, v + e + 0.01, f"{v:.2f}", ha="center",
                          va="bottom", fontsize=4.5)

    ax_a.set_xticks(x); ax_a.set_xticklabels(exist_a, fontsize=7)
    ax_a.set_ylabel("Test Accuracy"); ax_a.set_ylim(0, 1.0)
    ax_a.set_title("MNIST: Strategy Comparison", fontsize=8, fontweight="bold")
    ax_a.legend(fontsize=5, framealpha=0.9, edgecolor="none", ncol=2,
                loc="upper right")
    ax_a.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax_a.grid(axis="y", alpha=0.2, linewidth=0.3); ax_a.set_axisbelow(True)
    panel_label(ax_a, "A", x=-0.17)

    # ---- Panel B: Learning curves ----
    ax_b = fig.add_subplot(gs[0, 1])
    df_curves = extract_learning_curves("sweep_neurips_fa_dfa_baselines_20260220000705")
    df_curves2 = extract_learning_curves("sweep_neurips_mlp_local_learning_20260219163529")
    curve_dfs = [d for d in [df_curves, df_curves2] if not d.empty]
    if curve_dfs:
        dfc = pd.concat(curve_dfs, ignore_index=True)
        dfc = dfc[dfc["split"] == "test"].copy()
        dfc["method"] = dfc.apply(_label, axis=1)
        dfc["arch"] = dfc["core_type"].map(arch_map)

        # Filter per_soma for local
        mask_l = (dfc["strategy"] == "local_ca") & (dfc["arch"] != "MLP")
        mask_p = dfc["broadcast"] == "per_soma"
        dfc = dfc[~mask_l | mask_p]

        conds = [
            ("Backprop", "Shunting", C_BP, "-", "o"),
            ("Local 5F", "Shunting", C_5F, "-", "D"),
            ("Local 3F", "Shunting", C_3F, "--", "s"),
            ("DFA", "Shunting", C_DFA, "-.", "^"),
            ("Backprop", "MLP", "#7f8c8d", ":", "v"),
        ]
        for meth, arch, c, ls, mk in conds:
            sub = dfc[(dfc["method"] == meth) & (dfc["arch"] == arch)]
            if sub.empty:
                continue
            grp = sub.groupby("epoch")["accuracy"].agg(
                ["mean", "std"]).reset_index().sort_values("epoch")
            short = f"{meth}" if arch == "Shunting" else f"{meth} ({arch})"
            ax_b.plot(grp["epoch"], grp["mean"], color=c, linestyle=ls,
                      marker=mk, markersize=2.5,
                      markevery=max(1, len(grp) // 5), label=short)
            if grp["std"].max() > 0:
                ax_b.fill_between(grp["epoch"],
                                  grp["mean"] - grp["std"],
                                  grp["mean"] + grp["std"],
                                  alpha=0.08, color=c)

        ax_b.set_xlabel("Epoch"); ax_b.set_ylabel("Test Accuracy")
        ax_b.set_title("MNIST: Learning Dynamics", fontsize=8, fontweight="bold")
        ax_b.legend(fontsize=5, loc="lower right", framealpha=0.9, edgecolor="none")
        ax_b.grid(alpha=0.2, linewidth=0.3); ax_b.set_axisbelow(True)
    panel_label(ax_b, "B", x=-0.17)

    # ---- Panel C: Gradient fidelity ----
    ax_c = fig.add_subplot(gs[1, 0])
    grad_dir = ANALYSIS_DIR / "gradient_fidelity"
    core_map = {
        "config_0": "Shunting", "config_1": "Shunting", "config_2": "Shunting",
        "config_3": "Additive", "config_4": "Additive", "config_5": "Additive",
    }
    traj_dfs = []
    for cn, ct in core_map.items():
        p = grad_dir / cn / "gradient_fidelity_trajectory.csv"
        if p.exists():
            d = pd.read_csv(p); d["config"] = cn; d["core_type"] = ct
            traj_dfs.append(d)

    if traj_dfs:
        dft = pd.concat(traj_dfs, ignore_index=True)
        rule_colors = {"5f": C_5F, "3f": C_3F, "4f": C_4F}
        rule_styles = {"5f": "-", "3f": "--", "4f": ":"}

        for rule in ["5f", "3f"]:
            for ctype, alpha_v in [("Shunting", 1.0), ("Additive", 0.5)]:
                sub = dft[(dft["rule_variant"] == rule) & (dft["core_type"] == ctype)]
                if sub.empty:
                    continue
                epoch_vals = []
                for epoch in sorted(sub["epoch"].unique()):
                    de = sub[sub["epoch"] == epoch]
                    sv = []
                    for cfg in de["config"].unique():
                        dc = de[de["config"] == cfg]
                        sv.append((dc["cosine_similarity"] * dc["numel"]).sum() /
                                  dc["numel"].sum())
                    epoch_vals.append({"epoch": epoch, "mean": np.mean(sv),
                                       "std": np.std(sv) if len(sv) > 1 else 0})
                ev = pd.DataFrame(epoch_vals)
                mk = "o" if ctype == "Shunting" else "s"
                ax_c.plot(ev["epoch"], ev["mean"], color=rule_colors[rule],
                          alpha=alpha_v, linestyle=rule_styles[rule],
                          marker=mk, markersize=2,
                          markevery=max(1, len(ev) // 6),
                          label=f"{rule.upper()} ({ctype[:3]})")
                if ev["std"].max() > 0:
                    ax_c.fill_between(ev["epoch"], ev["mean"] - ev["std"],
                                      ev["mean"] + ev["std"],
                                      alpha=0.06, color=rule_colors[rule])

        ax_c.axhline(y=0, color="gray", linewidth=0.4)
        ax_c.set_xlabel("Epoch"); ax_c.set_ylabel("Weighted Cosine Sim.")
        ax_c.set_title("Gradient Fidelity Over Training", fontsize=8,
                       fontweight="bold")
        ax_c.legend(fontsize=5, loc="upper right", framealpha=0.9, edgecolor="none",
                    ncol=2)
        ax_c.grid(alpha=0.2, linewidth=0.3); ax_c.set_axisbelow(True)
    panel_label(ax_c, "C", x=-0.17)

    # ---- Panel D: Fashion-MNIST ----
    ax_d = fig.add_subplot(gs[1, 1])
    if not df_fmnist.empty:
        df_fmnist["method"] = df_fmnist.apply(_label, axis=1)
        df_fmnist["arch"] = df_fmnist["core_type"].map(arch_map)

        # per_soma broadcast only
        df_ps = df_fmnist[df_fmnist["broadcast"] == "per_soma"]
        fm_summary = df_ps.groupby(["method", "arch"])["test_acc"].agg(
            ["mean", "std", "count"]).reset_index()
        fm_summary["ci95"] = 1.96 * fm_summary["std"] / np.sqrt(fm_summary["count"])

        method_order_fm = ["Local 5F", "Local 4F", "Local 3F"]
        colors_fm = {"Local 5F": C_5F, "Local 4F": C_4F, "Local 3F": C_3F}
        exist_fm = [m for m in method_order_fm if m in fm_summary["method"].values]
        exist_fa = [a for a in ["Shunting", "Additive"] if a in fm_summary["arch"].values]
        n_fm, n_fa = len(exist_fm), len(exist_fa)

        if n_fm > 0 and n_fa > 0:
            bw_d = 0.75 / max(n_fm, 1)
            x_d = np.arange(n_fa)
            for i, meth in enumerate(exist_fm):
                vals, errs = [], []
                for arch in exist_fa:
                    row = fm_summary[(fm_summary["method"] == meth) &
                                     (fm_summary["arch"] == arch)]
                    if not row.empty:
                        vals.append(row["mean"].iloc[0])
                        errs.append(row["ci95"].iloc[0])
                    else:
                        vals.append(0); errs.append(0)
                off = (i - n_fm / 2 + 0.5) * bw_d
                ax_d.bar(x_d + off, vals, bw_d * 0.85, yerr=errs, label=meth,
                         color=colors_fm.get(meth, "#999"),
                         edgecolor="white", linewidth=0.3,
                         capsize=1.5, error_kw={"linewidth": 0.6})
                for j, (v, e) in enumerate(zip(vals, errs)):
                    if v > 0.03:
                        ax_d.text(x_d[j] + off, v + e + 0.006, f"{v:.3f}",
                                  ha="center", va="bottom", fontsize=4.5)

            ax_d.set_xticks(x_d)
            ax_d.set_xticklabels(exist_fa, fontsize=7)
            ax_d.legend(fontsize=5, framealpha=0.9, edgecolor="none")

    ax_d.set_ylabel("Test Accuracy")
    ax_d.set_title("Fashion-MNIST: Local Rules\n(per-soma broadcast)", fontsize=8,
                   fontweight="bold")
    ax_d.grid(axis="y", alpha=0.2, linewidth=0.3); ax_d.set_axisbelow(True)
    panel_label(ax_d, "D", x=-0.17)

    save(fig, "fig_neurips_combined")


# ===================================================================
if __name__ == "__main__":
    print("Generating NeurIPS paper figures...")
    print(f"Output: {OUT_DIR}\n")
    figure_main_results()
    figure_learning_curves()
    figure_gradient_fidelity()
    figure_fashion_mnist()
    figure_combined()
    print("\nDone!")

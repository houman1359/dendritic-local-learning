#!/usr/bin/env python
"""Summarize the four new NeurIPS sweeps:
  1. component_ablation
  2. weight_dist_by_depth
  3. cue_routing_soma
  4. cifar10_depth4_localca_soma

Each summarizer reads `final.json` (test/valid accuracy) from each completed
config and groups results by sweep parameters. Writes summary CSVs into
analysis/<sweep>_summary_<date>/ for downstream figure generation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import date
import yaml

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DRAFT = REPO_ROOT / "drafts" / "dendritic-local-learning"
LOCAL_RUNS = DRAFT / "local_sweep_runs"
ANALYSIS = DRAFT / "analysis"
TODAY = date.today().strftime("%Y%m%d")


def _latest(prefix: str) -> Path | None:
    matches = sorted(LOCAL_RUNS.glob(f"{prefix}_*"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def _load_final(config_dir: Path) -> dict | None:
    final_path = config_dir / "performance" / "final.json"
    if not final_path.exists():
        return None
    with final_path.open() as f:
        return json.load(f)


def _walk_sweep(sweep_root: Path) -> list[dict]:
    """Yield per-config rows. Reads config.json inside each results/config_X to
    recover the true run_name (index order from the generator; alphabetical
    listing of YAML files doesn't match)."""
    results = []
    results_dir = sweep_root / "results"
    if not results_dir.exists():
        return results
    for cfg_subdir in sorted(results_dir.iterdir()):
        if not cfg_subdir.is_dir():
            continue
        idx_match = re.match(r"config_(\d+)", cfg_subdir.name)
        if not idx_match:
            continue
        idx = int(idx_match.group(1))
        # Read config.json that was saved during training (authoritative)
        cfg_json_path = cfg_subdir / "config.json"
        if not cfg_json_path.exists():
            continue
        with cfg_json_path.open() as f:
            cfg = json.load(f)
        run_name = cfg.get("outputs", {}).get("run_name", cfg_subdir.name)
        seed = cfg.get("experiment", {}).get("seed")
        final = _load_final(cfg_subdir)
        if final is None:
            continue
        acc = final.get("accuracy", {})
        results.append({
            "config_index": idx,
            "run_name": run_name,
            "seed": seed,
            "train_accuracy": acc.get("train"),
            "valid_accuracy": acc.get("valid"),
            "test_accuracy": acc.get("test"),
            "config_path": str(cfg_json_path),
        })
    return results


def _extract_seed_suffix(run_name: str) -> tuple[str, int | None]:
    """Strip _s{seed} from run_name and return (base_name, seed)."""
    m = re.match(r"(.+)_s(\d+)$", run_name)
    if m:
        return m.group(1), int(m.group(2))
    return run_name, None


# ============================================================================
# Ablation sweep summarizer
# ============================================================================
def summarize_ablation():
    sweep_root = _latest("sweep_neurips_component_ablation")
    if sweep_root is None:
        print("No component_ablation sweep found.")
        return
    print(f"Summarizing ablation: {sweep_root.name}")
    rows = _walk_sweep(sweep_root)
    if not rows:
        print("  No completed configs yet.")
        return

    df = pd.DataFrame(rows)
    df["base_name"], df["parsed_seed"] = zip(*df["run_name"].map(_extract_seed_suffix))
    df["core"] = df["base_name"].apply(
        lambda n: "shunting" if "_shunting_" in n else "additive"
    )
    df["condition"] = df["base_name"].apply(
        lambda n: re.sub(r"^ablation_(shunting|additive)_", "", n)
    )

    out_dir = ANALYSIS / f"component_ablation_summary_{TODAY}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_dir / "ablation_detailed_results.csv", index=False)

    grouped = (
        df.groupby(["core", "condition"])["test_accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "test_acc_mean",
            "std": "test_acc_std",
            "min": "test_acc_min",
            "max": "test_acc_max",
            "count": "n_seeds",
        })
    )
    grouped.to_csv(out_dir / "ablation_grouped_summary.csv", index=False)
    print(f"  Wrote {len(df)} rows to {out_dir.name}/ "
          f"({grouped['n_seeds'].sum()} seeds, {len(grouped)} groups)")
    return grouped


# ============================================================================
# Weight distribution summarizer
# ============================================================================
def summarize_weight_dist():
    sweep_root = _latest("sweep_neurips_weight_dist_by_depth")
    if sweep_root is None:
        print("No weight_dist_by_depth sweep found.")
        return
    print(f"Summarizing weight distribution: {sweep_root.name}")
    rows = []
    results_dir = sweep_root / "results"
    if not results_dir.exists():
        print("  No results yet.")
        return

    for cfg_subdir in sorted(results_dir.iterdir()):
        idx_match = re.match(r"config_(\d+)", cfg_subdir.name)
        if not idx_match:
            continue
        idx = int(idx_match.group(1))
        cfg_json_path = cfg_subdir / "config.json"
        if not cfg_json_path.exists():
            continue
        with cfg_json_path.open() as f:
            cfg = json.load(f)
        run_name = cfg.get("outputs", {}).get("run_name", cfg_subdir.name)
        seed = cfg.get("experiment", {}).get("seed")

        # Final accuracy
        final = _load_final(cfg_subdir)
        if final is None:
            continue
        acc = final.get("accuracy", {})

        # Weight statistics from weight_analysis/final
        weight_path = cfg_subdir / "weight_analysis" / "final"
        weight_stats = {}
        if weight_path.exists():
            with weight_path.open() as f:
                wa = json.load(f)
            global_stats = wa.get("global_statistics", {})
            for w_type in ("excitatory_weights", "inhibitory_weights", "branch_weights"):
                gs = global_stats.get(w_type, {})
                for k in ("mean", "variance", "std", "min", "max", "median", "q1", "q3"):
                    weight_stats[f"{w_type}_{k}"] = gs.get(k)
            # Per-layer mean / std for excitatory weights
            layer_stats = wa.get("layer_statistics", {})
            for layer_name, ls in layer_stats.items():
                weight_stats[f"layer_{layer_name}_exc_mean"] = ls.get("exc_weight_mean_mean")
                weight_stats[f"layer_{layer_name}_exc_std"] = ls.get("exc_weight_mean_std")

        base_name, _ = _extract_seed_suffix(run_name)
        # Parse run_name: wd_<core>_<strategy>_<depth>
        m = re.match(r"wd_(shunting|additive)_(bp|localca)_(d\d+)", base_name)
        core, strategy, depth = (m.group(1), m.group(2), m.group(3)) if m else ("?", "?", "?")

        rows.append({
            "config_index": idx,
            "run_name": run_name,
            "core": core,
            "strategy": strategy,
            "depth": depth,
            "seed": seed,
            "train_accuracy": acc.get("train"),
            "valid_accuracy": acc.get("valid"),
            "test_accuracy": acc.get("test"),
            **weight_stats,
        })
    if not rows:
        print("  No completed configs yet.")
        return

    df = pd.DataFrame(rows)
    out_dir = ANALYSIS / f"weight_dist_by_depth_summary_{TODAY}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "weight_dist_detailed_results.csv", index=False)

    # Grouped by (core, strategy, depth)
    grouped_cols = [c for c in df.columns
                    if c not in ("config_index", "run_name", "seed", "config_path")]
    grouped = df.groupby(["core", "strategy", "depth"])[
        [c for c in grouped_cols if c not in ("core", "strategy", "depth")
         and pd.api.types.is_numeric_dtype(df[c])]
    ].mean().reset_index()
    grouped.to_csv(out_dir / "weight_dist_grouped_summary.csv", index=False)
    print(f"  Wrote {len(df)} rows to {out_dir.name}/ "
          f"({len(grouped)} groups)")
    return grouped


# ============================================================================
# Cue routing soma summarizer
# ============================================================================
def summarize_cue_soma():
    sweep_root = _latest("cue_routing_soma_5seed")
    if sweep_root is None:
        print("No cue_routing_soma sweep found.")
        return
    print(f"Summarizing cue routing soma: {sweep_root.name}")
    rows = _walk_sweep(sweep_root)
    if not rows:
        print("  No completed configs yet.")
        return

    df = pd.DataFrame(rows)
    df["base_name"], df["parsed_seed"] = zip(*df["run_name"].map(_extract_seed_suffix))

    out_dir = ANALYSIS / f"cue_routing_soma_summary_{TODAY}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cue_routing_soma_detailed.csv", index=False)

    grouped = (
        df.groupby("base_name")["test_accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "test_acc_mean",
            "std": "test_acc_std",
            "min": "test_acc_min",
            "max": "test_acc_max",
            "count": "n_seeds",
        })
        .sort_values("test_acc_mean", ascending=False)
    )
    grouped.to_csv(out_dir / "cue_routing_soma_grouped.csv", index=False)
    print(f"  Wrote {len(df)} rows to {out_dir.name}/ ({len(grouped)} conditions)")
    return grouped


# ============================================================================
# CIFAR-10 depth-4 soma summarizer (merges additive + shunting rerun)
# ============================================================================
def _strip_v2(name: str) -> str:
    """Drop the _v2 suffix we added for the shunting rerun so conditions align."""
    return name[:-3] if name.endswith("_v2") else name


def summarize_cifar_depth4_soma():
    # The original sweep at .../sweep_neurips_cifar10_depth4_localca_soma_*
    # actually ran the shunting half cleanly but failed on the additive half
    # due to a legacy base-config field. We now merge three sources:
    #   1. shunting rerun (cifar10_shunting_depth4_soma_rerun_*)
    #   2. additive rerun (cifar10_additive_depth4_soma_rerun_*)
    #   3. original shunting half from the first sweep (same run_names)
    sources = []
    for prefix in (
        "cifar10_shunting_depth4_soma_rerun",
        "cifar10_additive_depth4_soma_rerun",
        "sweep_neurips_cifar10_depth4_localca_soma",
    ):
        root = _latest(prefix)
        if root is not None:
            sources.append(root)

    rows = []
    seen_run_names = set()
    for root in sources:
        print(f"Summarizing CIFAR-10 depth4 soma: {root.name}")
        for row in _walk_sweep(root):
            # Deduplicate: prefer the rerun result (which we walk first) over
            # the original sweep, which only has shunting completed.
            if row["run_name"] in seen_run_names:
                continue
            # Also strip _v2 from run_name so reruns match originals.
            base = _extract_seed_suffix(row["run_name"])[0]
            base = _strip_v2(base)
            dedup_key = (base, row.get("seed"))
            if dedup_key in seen_run_names:
                continue
            seen_run_names.add(dedup_key)
            rows.append(row)

    if not rows:
        print("  No completed configs yet.")
        return

    df = pd.DataFrame(rows)
    df["base_name"], df["parsed_seed"] = zip(*df["run_name"].map(_extract_seed_suffix))
    df["base_name"] = df["base_name"].map(_strip_v2)
    # Keep only the soma-on conditions (not the original broken shunting runs in the additive sweep)
    df = df[df["base_name"].str.contains("_soma_")].reset_index(drop=True)

    out_dir = ANALYSIS / f"cifar10_depth4_soma_summary_{TODAY}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "cifar10_depth4_soma_detailed.csv", index=False)

    grouped = (
        df.groupby("base_name")["test_accuracy"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(columns={
            "mean": "test_acc_mean",
            "std": "test_acc_std",
            "min": "test_acc_min",
            "max": "test_acc_max",
            "count": "n_seeds",
        })
        .sort_values("test_acc_mean", ascending=False)
    )
    grouped.to_csv(out_dir / "cifar10_depth4_soma_grouped.csv", index=False)
    print(f"  Wrote {len(df)} rows to {out_dir.name}/ ({len(grouped)} conditions)")
    return grouped


# ============================================================================
# Soma extension summarizer (replaces hard-coded numbers in fig_s_soma_extension)
# ============================================================================
PHASE_FAMILY_SWEEPS = {
    "phase1_capacity":       ("phase1_capacity_calibration", "Phase 1\n(Capacity)"),
    "claimA_shunting_regime":("phase3_claimA_shunting_regime_strong", "Claim A\n(Shunting\nRegime)"),
    "claimB_morphology":     ("phase3_claimB_morphology_scaling", "Claim B\n(Morphology\nScaling)"),
    "claimC_error_shaping":  ("phase3_claimC_error_shaping", "Claim C\n(Error\nShaping)"),
}
LFS06_SWEEP_ROOT = Path(
    "/n/holylfs06/LABS/kempner_project_b/Lab/dendritic/HS/LOCAL_LEARNING/sweep_runs"
)


def _latest_lfs06(prefix: str) -> Path | None:
    """Return the most recent sweep directory under LFS06_SWEEP_ROOT matching prefix."""
    if not LFS06_SWEEP_ROOT.exists():
        return None
    matches = sorted(LFS06_SWEEP_ROOT.glob(f"{prefix}_*"))
    return matches[-1] if matches else None


def _walk_final_accuracies(sweep_root: Path) -> list[float]:
    """Return list of test accuracies from all completed configs in a sweep."""
    results = []
    results_dir = sweep_root / "results"
    if not results_dir.exists():
        return results
    for cfg_subdir in sorted(results_dir.iterdir()):
        if not cfg_subdir.is_dir():
            continue
        final = _load_final(cfg_subdir)
        if final is None:
            continue
        acc = final.get("accuracy", {}).get("test")
        if acc is not None:
            results.append(float(acc))
    return results


def summarize_soma_extension():
    """Walk the four paper-facing LocalCA phase-family sweeps (soma-off safe
    reruns and soma-on extensions) and produce a structured CSV with mean /
    best test accuracy per family x soma setting. This replaces the hard-coded
    numbers currently embedded in `fig_s_soma_extension`.
    """
    import pandas as pd  # already imported globally; here for clarity

    rows = []
    for family_key, (path_stub, display_label) in PHASE_FAMILY_SWEEPS.items():
        for soma_tag, soma_on in (("safe", False), ("safe_soma", True)):
            # The sweep prefix is one of:
            #   sweep_neurips_{path_stub}_localca_occq_{soma_tag}_rerun_*
            prefix = f"sweep_neurips_{path_stub}_localca_occq_{soma_tag}_rerun"
            sweep = _latest_lfs06(prefix)
            if sweep is None:
                print(f"  soma-extension: no sweep for {family_key} soma={soma_on}")
                continue
            accs = _walk_final_accuracies(sweep)
            if not accs:
                print(f"  soma-extension: no results yet for {family_key} soma={soma_on}")
                continue
            rows.append({
                "family": family_key,
                "display_label": display_label,
                "soma": "on" if soma_on else "off",
                "sweep_name": sweep.name,
                "n_configs_completed": len(accs),
                "mean_test_accuracy": float(sum(accs) / len(accs)),
                "best_test_accuracy": float(max(accs)),
            })

    if not rows:
        print("No soma-extension sweeps found on disk.")
        return

    df = pd.DataFrame(rows)
    out_dir = ANALYSIS / f"soma_extension_summary_{TODAY}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "soma_extension_summary.csv", index=False)
    print(f"Summarized soma extension: {len(rows)} rows -> {out_dir.name}/soma_extension_summary.csv")
    return df


# ============================================================================
# Weight-dist under soma-on (closes the soma <-> weight-stats loop)
# ============================================================================
def summarize_weight_dist_soma():
    sweep_root = _latest("sweep_neurips_weight_dist_soma")
    if sweep_root is None:
        print("No weight-dist-soma sweep found.")
        return
    print(f"Summarizing weight-dist-soma: {sweep_root.name}")
    rows = []
    results_dir = sweep_root / "results"
    if not results_dir.exists():
        print("  No results yet.")
        return

    for cfg_subdir in sorted(results_dir.iterdir()):
        idx_match = re.match(r"config_(\d+)", cfg_subdir.name)
        if not idx_match:
            continue
        cfg_json_path = cfg_subdir / "config.json"
        if not cfg_json_path.exists():
            continue
        with cfg_json_path.open() as f:
            cfg = json.load(f)
        run_name = cfg.get("outputs", {}).get("run_name", cfg_subdir.name)
        seed = cfg.get("experiment", {}).get("seed")

        final = _load_final(cfg_subdir)
        if final is None:
            continue
        acc = final.get("accuracy", {})

        # Weight statistics
        weight_path = cfg_subdir / "weight_analysis" / "final"
        ws = {}
        if weight_path.exists():
            with weight_path.open() as f:
                wa = json.load(f)
            gs = wa.get("global_statistics", {}).get("excitatory_weights", {})
            for k in ("mean", "variance", "std", "min", "max", "median", "q1", "q3"):
                ws[f"excitatory_weights_{k}"] = gs.get(k)

        # Parse: wdsoma_<core>_<strategy>_s<seed>
        base, _ = _extract_seed_suffix(run_name)
        m = re.match(r"wdsoma_(shunting|additive)_(bp|localca)$", base)
        if not m:
            continue
        core, strategy = m.group(1), m.group(2)

        rows.append({
            "config_index": int(idx_match.group(1)),
            "run_name": run_name,
            "core": core,
            "strategy": strategy,
            "seed": seed,
            "train_accuracy": acc.get("train"),
            "valid_accuracy": acc.get("valid"),
            "test_accuracy": acc.get("test"),
            **ws,
        })

    if not rows:
        print("  No completed configs yet.")
        return

    df = pd.DataFrame(rows)
    out_dir = ANALYSIS / f"weight_dist_soma_summary_{TODAY}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "weight_dist_soma_detailed.csv", index=False)

    numeric_cols = [c for c in df.columns
                    if c not in ("config_index", "run_name", "seed")
                    and pd.api.types.is_numeric_dtype(df[c])]
    grouped = df.groupby(["core", "strategy"])[numeric_cols].mean().reset_index()
    grouped.to_csv(out_dir / "weight_dist_soma_grouped_summary.csv", index=False)
    print(f"  Wrote {len(df)} rows to {out_dir.name}/ ({len(grouped)} groups)")
    return grouped


def main():
    print("=" * 60)
    print("Summarizing new NeurIPS sweeps")
    print("=" * 60)
    summarize_ablation()
    summarize_weight_dist()
    summarize_cue_soma()
    summarize_cifar_depth4_soma()
    summarize_soma_extension()
    summarize_weight_dist_soma()
    print("Done.")


if __name__ == "__main__":
    main()

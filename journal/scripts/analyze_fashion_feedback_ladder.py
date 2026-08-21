#!/usr/bin/env python3
"""Audit and analyze the frozen Fashion-MNIST feedback ladder."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from journal_style import (  # noqa: E402
    COLORS,
    ERR_CAPSIZE,
    FIG_W,
    LW_DATA,
    LW_ERR,
    LW_HAIR,
    LW_REF,
    MARKER_MS,
    PT_LEGEND,
    PT_SMALL,
    SEED_ALPHA,
    SEED_MS,
    apply_neurips_style,
    audit_layout,
    audit_text_over_data,
    panel_title,
    style_axis,
)


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "fashion_feedback_runs"
OUTPUT = ROOT / "source_data" / "fashion_feedback_ladder"
FIGURES = ROOT / "figures" / "generated"
SEEDS = tuple(range(10200, 10210))
MODES = {
    "per_soma": "scalar fallback",
    "per_soma_shared": "neuron indexed",
    "path_transport": "exact path",
}
CORE_LABEL = {
    "dendritic_shunting": "shunting",
    "dendritic_additive": "additive",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean(values: np.ndarray, seed: int, draws: int = 50_000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(values.mean())
    null = np.asarray([
        np.mean(values * np.asarray(signs, dtype=float))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    return float(np.mean(np.abs(null) >= observed - 1e-15))


def _config_index(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def collect(run_dirs: list[Path], allow_incomplete: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    duplicates: list[str] = []
    warnings: list[str] = []
    seen: set[tuple[str, int, str]] = set()
    run_records: list[dict[str, Any]] = []

    for run in run_dirs:
        if not run.is_dir():
            raise FileNotFoundError(run)
        manifest = json.loads((run / "frozen_sweep_manifest.json").read_text())
        run_records.append({
            "run_dir": str(run.relative_to(ROOT)),
            "manifest_sha256": sha256(run / "frozen_sweep_manifest.json"),
            "source_identity": manifest.get("source_identity", {}),
        })
        for config_path in sorted((run / "configs").glob("unified_config_*.yaml"), key=_config_index):
            index = _config_index(config_path)
            config = yaml.safe_load(config_path.read_text())
            core = str(config["model"]["core"]["type"])
            seed = int(config["experiment"]["seed"])
            mode = str(config["training"]["main"]["learning_strategy_config"]["error_broadcast_mode"])
            if core not in CORE_LABEL or mode not in MODES:
                raise RuntimeError(f"unexpected core/mode in {config_path}: {core}/{mode}")
            if config["data"]["dataset_name"] != "fashion_mnist" or seed not in SEEDS:
                raise RuntimeError(f"dataset/seed contract failure in {config_path}")
            result_dir = run / "results" / f"config_{index}"
            final_path = result_dir / "performance" / "final.json"
            checkpoint = result_dir / "final_model.pt"
            if not final_path.is_file() or not checkpoint.is_file():
                continue
            key = (core, seed, mode)
            if key in seen:
                duplicates.append("/".join(map(str, key)))
                continue
            seen.add(key)
            final = json.loads(final_path.read_text())
            accuracy = float(final["accuracy"]["test"])
            if not np.isfinite(accuracy) or not 0 <= accuracy <= 1:
                raise RuntimeError(f"invalid accuracy {accuracy} in {final_path}")
            log_text = "\n".join(
                path.read_text(errors="replace")
                for path in (result_dir / "train.log", result_dir / "dendritic_modeling.log")
                if path.is_file()
            ).lower()
            if "fallback" in log_text and "scalar" not in log_text:
                warnings.append(f"fallback marker in {result_dir}")
            rows.append({
                "architecture": CORE_LABEL[core],
                "core": core,
                "seed": seed,
                "feedback": MODES[mode],
                "broadcast_mode": mode,
                "test_accuracy": accuracy,
                "run_dir": str(run.relative_to(ROOT)),
                "config_index": index,
                "config_sha256": sha256(config_path),
                "result_sha256": sha256(final_path),
                "checkpoint_sha256": sha256(checkpoint),
            })

    frame = pd.DataFrame(rows)
    expected = {(core, seed, mode) for core in CORE_LABEL for seed in SEEDS for mode in MODES}
    missing = sorted(expected - seen)
    audit = {
        "status": "incomplete" if missing else "complete_and_validated",
        "n_expected": 60,
        "n_complete": len(frame),
        "missing": ["/".join(map(str, item)) for item in missing],
        "duplicates_ignored": duplicates,
        "warnings": warnings,
        "seeds": list(SEEDS),
        "runs": run_records,
    }
    if (missing or warnings) and not allow_incomplete:
        raise RuntimeError(json.dumps(audit, indent=2))
    return frame.sort_values(["architecture", "seed", "feedback"]), audit


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    primary_passes: list[bool] = []
    order = ["scalar fallback", "neuron indexed", "exact path"]
    for arch_index, architecture in enumerate(("shunting", "additive")):
        part = frame[frame.architecture.eq(architecture)]
        wide = part.pivot(index="seed", columns="feedback", values="test_accuracy").loc[list(SEEDS), order]
        for mode_index, feedback in enumerate(order):
            mean, low, high = bootstrap_mean(wide[feedback].to_numpy(float), 180_000 + 10 * arch_index + mode_index)
            summary_rows.append({
                "architecture": architecture,
                "feedback": feedback,
                "n_seeds": len(wide),
                "mean_accuracy": mean,
                "sd_accuracy": float(wide[feedback].std(ddof=1)),
                "ci95_low": low,
                "ci95_high": high,
            })
        for contrast_index, (high_name, low_name, label) in enumerate((
            ("neuron indexed", "scalar fallback", "neuron indexed - scalar fallback"),
            ("exact path", "neuron indexed", "exact path - neuron indexed"),
        )):
            diff = (wide[high_name] - wide[low_name]).to_numpy(float)
            mean, low, high = bootstrap_mean(diff, 181_000 + 10 * arch_index + contrast_index)
            signs = int(np.sum(diff > 0))
            contrast_rows.append({
                "architecture": architecture,
                "contrast": label,
                "n_seeds": len(diff),
                "mean_difference": mean,
                "ci95_low": low,
                "ci95_high": high,
                "positive_seeds": signs,
                "negative_seeds": int(np.sum(diff < 0)),
                "exact_two_sided_sign_flip_p": exact_sign_flip_p(diff),
            })
            if contrast_index == 0:
                primary_passes.append(signs >= 8 and low > 0)
    decision = {
        "primary_identity_replication_pass": bool(all(primary_passes)),
        "criterion": "both architectures: at least 8/10 positive seeds and paired-bootstrap CI above zero",
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(contrast_rows), decision


def plot(frame: pd.DataFrame, summary: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    from matplotlib.ticker import MultipleLocator

    apply_neurips_style()
    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, 2.55), gridspec_kw={
        "left": 0.09, "right": 0.985, "bottom": 0.15, "top": 0.845, "wspace": 0.30,
    })
    order = ["scalar fallback", "neuron indexed", "exact path"]
    labels = ["scalar", "neuron\nindexed", "exact\npath"]
    arch_colors = {"shunting": COLORS["shunting"], "additive": COLORS["additive"]}
    offsets = {"shunting": -0.06, "additive": 0.06}
    for architecture in ("shunting", "additive"):
        wide = frame[frame.architecture.eq(architecture)].pivot(index="seed", columns="feedback", values="test_accuracy").loc[list(SEEDS), order]
        x = np.arange(3, dtype=float) + offsets[architecture]
        for values in wide.to_numpy(float):
            axes[0].plot(x, values, color=arch_colors[architecture], alpha=0.15, lw=LW_HAIR)
        part = summary[summary.architecture.eq(architecture)].set_index("feedback").loc[order]
        y = part.mean_accuracy.to_numpy(float)
        axes[0].errorbar(x, y, yerr=np.vstack([y - part.ci95_low, part.ci95_high - y]),
                         color=arch_colors[architecture], marker="o" if architecture == "shunting" else "s",
                         ms=MARKER_MS, lw=LW_DATA, elinewidth=LW_ERR, capsize=ERR_CAPSIZE)
    axes[0].set_xticks(range(3), labels)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.02))
    # Percent ticks match the MNIST accuracy panels (fig. 2 B/D/F): one
    # unit format for the same quantity across the multi-part figure.
    axes[0].yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    axes[0].set_ylabel("Fashion-MNIST test accuracy")
    panel_title(axes[0], "P", "Second-dataset ladder")
    style_axis(axes[0])
    # Two series only: stacked direct colour labels in the empty lower-right
    # corner replace the boxed legend (this is the figure's colour key; panel Q
    # cross-references it).
    axes[0].text(0.97, 0.16, "shunting", transform=axes[0].transAxes,
                 color=arch_colors["shunting"], fontsize=PT_LEGEND,
                 ha="right", va="bottom")
    axes[0].text(0.97, 0.05, "additive", transform=axes[0].transAxes,
                 color=arch_colors["additive"], fontsize=PT_LEGEND,
                 ha="right", va="bottom")

    contrast_order = ["neuron indexed - scalar fallback", "exact path - neuron indexed"]
    cursor = 0
    for contrast in contrast_order:
        for architecture in ("shunting", "additive"):
            row = contrasts[(contrasts.contrast.eq(contrast)) & (contrasts.architecture.eq(architecture))].iloc[0]
            values_wide = frame[frame.architecture.eq(architecture)].pivot(index="seed", columns="feedback", values="test_accuracy").loc[list(SEEDS)]
            high_name, low_name = ("neuron indexed", "scalar fallback") if contrast.startswith("neuron") else ("exact path", "neuron indexed")
            values = 100 * (values_wide[high_name] - values_wide[low_name]).to_numpy(float)
            axes[1].scatter(cursor + np.linspace(-0.055, 0.055, len(values)), values,
                            s=SEED_MS**2, color=arch_colors[architecture], alpha=SEED_ALPHA, edgecolors="none")
            axes[1].errorbar(cursor, 100 * row.mean_difference,
                             yerr=[[100 * (row.mean_difference - row.ci95_low)], [100 * (row.ci95_high - row.mean_difference)]],
                             color=arch_colors[architecture], marker="D", markerfacecolor="white",
                             ms=MARKER_MS + 1.2, lw=LW_ERR, capsize=ERR_CAPSIZE)
            cursor += 1
    axes[1].axhline(0, color=COLORS["mute"], ls="--", lw=LW_REF)
    axes[1].axvline(1.5, color=COLORS["grid"], lw=LW_HAIR)
    # Architecture is already carried by the colour key in P, so the x axis
    # names only the two paired contrasts (group centres, tick marks hidden).
    axes[1].set_xlim(-0.5, 3.5)
    axes[1].set_xticks([0.5, 2.5], ["identity", "within-tree\ntransport"])
    axes[1].set_yticks([0, 2, 4, 6])
    axes[1].set_ylabel("paired accuracy gain (pp)")
    # This block is published as Fig. 2 G-H, where the ladder panel carrying
    # the colour key is lettered G.
    axes[1].text(0.97, 0.97, "colors as in G", transform=axes[1].transAxes,
                 color=COLORS["mute"], fontsize=PT_SMALL, ha="right", va="top")
    panel_title(axes[1], "Q", "Replicated bottleneck")
    style_axis(axes[1])
    # After style_axis (which resets tick geometry): group labels sit at the
    # cluster centres, so their tick marks would point at empty space.
    axes[1].tick_params(axis="x", length=0)

    fig.canvas.draw()
    audit_layout(fig, "fig_fashion_feedback_ladder")
    audit_text_over_data(fig, "fig_fashion_feedback_ladder")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "fig_fashion_feedback_ladder.pdf", metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(FIGURES / "fig_fashion_feedback_ladder.png", dpi=600)
    plt.close(fig)


def replot_from_source_data() -> None:
    """Regenerate the figure from the frozen source-data tables.

    Used when the raw run directories have been archived off the checkout:
    the audited seed/summary/contrast CSVs under ``source_data`` are the
    quantitative record, so restyling passes re-read them verbatim instead of
    re-collecting (and never rewrite them).
    """
    frame = pd.read_csv(OUTPUT / "seed_outcomes.csv")
    summary = pd.read_csv(OUTPUT / "condition_summary.csv")
    contrasts = pd.read_csv(OUTPUT / "paired_contrasts.csv")
    if len(frame) != 60:
        raise RuntimeError("frozen seed_outcomes.csv is incomplete; refusing to plot")
    plot(frame, summary, contrasts)
    print(json.dumps({"status": "replotted_from_frozen_source_data", "n_seed_rows": len(frame)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    run_dirs = args.run or sorted(RUN_ROOT.glob("journal_fashion_feedback_ladder_*"))
    if not run_dirs and (OUTPUT / "seed_outcomes.csv").is_file():
        replot_from_source_data()
        return
    frame, audit = collect(run_dirs, args.allow_incomplete)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "seed_outcomes.csv", index=False)
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    if len(frame) == 60:
        summary, contrasts, decision = summarize(frame)
        summary.to_csv(OUTPUT / "condition_summary.csv", index=False)
        contrasts.to_csv(OUTPUT / "paired_contrasts.csv", index=False)
        payload = {"audit": audit, "decision": decision}
        (OUTPUT / "report.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        plot(frame, summary, contrasts)
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

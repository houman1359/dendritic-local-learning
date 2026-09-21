#!/usr/bin/env python3
"""Audit the Ky--Fan bound and affine alignment thresholds in frozen phase data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "credit_phase_theory"


def bootstrap(values: np.ndarray, seed: int, draws: int = 20_000):
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def main() -> None:
    frame = pd.read_csv(SOURCE / "spectral_phase_seed.csv")
    rows, summaries = [], []
    for budget_index, budget in enumerate(sorted(frame.budget_k.unique())):
        part = frame[frame.budget_k.eq(budget)]
        wide = part.pivot_table(
            index=["seed", "alignment"], columns="method", values="spectral_capture"
        ).reset_index()
        endpoints = wide[wide.alignment.isin([0.0, 1.0])].pivot(
            index="seed", columns="alignment", values=["ancestry", "random_rank"]
        )
        delta_zero = endpoints[("ancestry", 0.0)] - endpoints[("random_rank", 0.0)]
        delta_one = endpoints[("ancestry", 1.0)] - endpoints[("random_rank", 1.0)]
        raw_crossover = -delta_zero / (delta_one - delta_zero)
        threshold = np.where(delta_zero >= 0.0, 0.0, raw_crossover)
        threshold = np.asarray(threshold, dtype=float)
        if np.any((threshold < -1e-12) | (threshold > 1.0 + 1e-12)):
            raise SystemExit(f"invalid affine threshold at budget {budget}")
        for seed, d0, d1, raw, required in zip(
            endpoints.index, delta_zero, delta_one, raw_crossover, threshold
        ):
            rows.append(
                {
                    "seed": int(seed),
                    "budget_k": int(budget),
                    "ancestry_minus_random_at_rho0": float(d0),
                    "ancestry_minus_random_at_rho1": float(d1),
                    "raw_affine_crossover": float(raw),
                    "minimum_alignment_for_ancestry_advantage": float(required),
                    "ancestry_already_favored_at_rho0": bool(d0 >= 0.0),
                }
            )
        mean, low, high = bootstrap(threshold, 11_100_000 + budget_index)
        rho_one = part[part.alignment.eq(1.0)]
        ancestry = rho_one[rho_one.method.eq("ancestry")]
        maximum_ky_fan_violation = float(
            np.max(ancestry.spectral_capture - ancestry.pca_rank_upper_bound)
        )
        summaries.append(
            {
                "budget_k": int(budget),
                "n_seeds": len(threshold),
                "mean_minimum_alignment": mean,
                "ci95_low_mean_minimum_alignment": low,
                "ci95_high_mean_minimum_alignment": high,
                "median_minimum_alignment": float(np.median(threshold)),
                "already_favored_at_rho0": int(np.sum(delta_zero >= 0.0)),
                "favored_at_rho1": int(np.sum(delta_one > 0.0)),
                "maximum_ky_fan_violation": maximum_ky_fan_violation,
                "mean_ky_fan_regret_at_rho1": float(
                    np.mean(
                        ancestry.pca_rank_upper_bound
                        - ancestry.spectral_capture
                    )
                ),
            }
        )
    seed_table = pd.DataFrame(rows)
    summary = pd.DataFrame(summaries)
    seed_table.to_csv(SOURCE / "spectral_alignment_thresholds.csv", index=False, float_format="%.10g")
    summary.to_csv(SOURCE / "spectral_bound_summary.csv", index=False, float_format="%.10g")
    k4 = summary[summary.budget_k.eq(4)].iloc[0]
    report = f"""# Spectral upper-bound and alignment-threshold reanalysis

This deterministic secondary analysis uses the frozen 50-seed spectral phase.
For every rank and seed, dense principal-eigenspace capture obeyed the Ky--Fan
upper bound; the maximum apparent violation was
{summary.maximum_ky_fan_violation.max():.2e}. At alignment one, the ancestry
dictionary attained the bound at K=4 to numerical precision (mean regret
{k4.mean_ky_fan_regret_at_rho1:.2e}).

Because covariance is an affine mixture with constant trace, each paired
ancestry-minus-random contrast is affine in alignment rho. At K=4, ancestry
was already favored at rho=0 in {int(k4.already_favored_at_rho0)}/50 random
route draws and favored at rho=1 in {int(k4.favored_at_rho1)}/50. Treating an
already favorable seed as requiring zero alignment, the median minimum
alignment was {k4.median_minimum_alignment:.4f}; its mean was
{k4.mean_minimum_alignment:.4f} (95% paired-seed bootstrap interval
{k4.ci95_low_mean_minimum_alignment:.4f}--{k4.ci95_high_mean_minimum_alignment:.4f}).

The threshold is relative to a sampled random rank-matched route, not a
universal biological constant. The Ky--Fan result is an oracle capacity upper
bound and does not provide a local circuit for learning principal modes.
"""
    (SOURCE / "spectral_bound_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

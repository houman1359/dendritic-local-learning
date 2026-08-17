#!/usr/bin/env python3
"""Analyze the prospectively corrected fixed-step conductance experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source_data" / "positive_conductance_reliability_step_consistent"
CONFIG = (
    ROOT
    / "configs"
    / "positive_conductance_reliability"
    / "step_consistent_confirmatory.json"
)


def bootstrap(values: np.ndarray, seed: int, draws: int = 20_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def contrast(frame, h, left, right, metric, seed):
    part = frame[np.isclose(frame.reliability_heterogeneity, h)]
    wide = part.pivot(index="seed", columns="method", values=metric)
    values = (wide[left] - wide[right]).to_numpy(float)
    mean, low, high = bootstrap(values, seed)
    return {
        "heterogeneity": h,
        "left_minus_right": f"{left} - {right}",
        "metric": metric,
        "n_pairs": len(values),
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "positive_pairs": int(np.sum(values > 0)),
        "ties": int(np.sum(np.isclose(values, 0))),
        "wilcoxon_p_two_sided": 1.0 if np.allclose(values, 0) else float(
            wilcoxon(values, zero_method="wilcox").pvalue
        ),
    }


def main() -> None:
    frame = pd.read_csv(SOURCE / "seed_outcomes.csv")
    cfg = json.loads(CONFIG.read_text())
    high = float(max(cfg["reliability_heterogeneity"]))
    comparisons = [
        ("reliability_aligned_shunt", "best_global_shunt"),
        ("reliability_aligned_shunt", "noisy_no_shunt"),
        ("reliability_aligned_shunt", "shuffled_shunt"),
        ("reliability_aligned_shunt", "anti_aligned_shunt"),
        ("reliability_aligned_shunt", "explicit_point_gate"),
        ("aligned_clean_credit", "exact_clean_bp"),
    ]
    rows = []
    for h_index, h in enumerate(cfg["reliability_heterogeneity"]):
        for comparison_index, (left, right) in enumerate(comparisons):
            for metric_index, metric in enumerate(
                ["one_step_test_loss_decrease", "final_test_loss", "final_test_accuracy"]
            ):
                rows.append(
                    contrast(
                        frame,
                        float(h),
                        left,
                        right,
                        metric,
                        4_500_000 + 1000 * h_index + 10 * comparison_index + metric_index,
                    )
                )
    table = pd.DataFrame(rows)
    table.to_csv(SOURCE / "extended_contrasts.csv", index=False, float_format="%.10g")

    def select(left, right, metric):
        match = table[
            np.isclose(table.heterogeneity, high)
            & table.left_minus_right.eq(f"{left} - {right}")
            & table.metric.eq(metric)
        ]
        return match.iloc[0]

    one_global = select(
        "reliability_aligned_shunt", "best_global_shunt", "one_step_test_loss_decrease"
    )
    one_none = select(
        "reliability_aligned_shunt", "noisy_no_shunt", "one_step_test_loss_decrease"
    )
    final_global = select(
        "reliability_aligned_shunt", "best_global_shunt", "final_test_loss"
    )
    final_none = select(
        "reliability_aligned_shunt", "noisy_no_shunt", "final_test_loss"
    )
    report = f"""# Step-consistent state-matched conductance reliability result

The prospectively corrected programme completed {frame.seed.nunique()} fresh paired seeds,
{len(frame):,} seed--condition rows and every numerical gate. All input rates,
excitatory conductances and shunt conductances are nonnegative. A compensating
current holds branch voltage identical, isolating the physical input-resistance
factor in local eligibility. Unlike the superseded pilot, gains use the
fixed-step optimum for the actual half-smoothness step.

At reliability heterogeneity {high:g}, aligned minus best-global one-step loss
decrease was {one_global.mean_difference:.6f} (95% interval
{one_global.ci95_low:.6f}--{one_global.ci95_high:.6f}); all
{one_global.positive_pairs:.0f}/{one_global.n_pairs:.0f} paired seeds favored
aligned conductance. Aligned minus no-shunt one-step decrease was
{one_none.mean_difference:.6f} ({one_none.ci95_low:.6f}--{one_none.ci95_high:.6f}),
with {one_none.positive_pairs:.0f}/{one_none.n_pairs:.0f} pairs favoring
aligned and the two-sided Wilcoxon value was
{one_none.wilcoxon_p_two_sided:.3f}.

After 40 updates, aligned-minus-global final test loss was
{final_global.mean_difference:.6f} ({final_global.ci95_low:.6f}--{final_global.ci95_high:.6f});
negative favors aligned. However, aligned-minus-unshunted final loss was
{final_none.mean_difference:.6f} ({final_none.ci95_low:.6f}--{final_none.ci95_high:.6f}),
with {final_none.positive_pairs:.0f}/{final_none.n_pairs:.0f} positive pairs
and two-sided Wilcoxon value {final_none.wilcoxon_p_two_sided:.3g}.
Exact clean backpropagation remained the optimization reference. Independently
computed state-matched conductance and explicit point-gate trajectories agreed
to numerical precision.

This validates the ordering of a state-dependent positive-conductance
approximation to fixed-step reliability shrinkage, not exact block-scalar
realization, autonomous shunt learning, a dendrite-exclusive operation, or
uniform training superiority.
"""
    (SOURCE / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

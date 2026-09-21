"""Sensitivity analyses for the depth-association result (eighth review).

Two checks on the parent artifact produced by ``depth_association.py``:
the association recomputed on delta-NLL (trained minus teacher LM loss)
instead of the exponentiated perplexity ratio, and a quadratic depth term
F-tested against the linear model, with the fitted vertex reported so the
convexity can be described accurately (degradation is not "accelerating
across the entire network" — it has a fitted minimum, after which it
increases increasingly rapidly).

This module is the committed chain of custody for
``olmo2_32b_depth_association_sensitivity_v1.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compute_sensitivities(artifact: dict) -> dict:
    import numpy as np
    from scipy import stats

    teacher = float(artifact["teacher_lm_loss"])
    per_layer: dict[int, float] = {}
    for arm in artifact["arm_records"]:
        layer = int(arm["layer"])
        loss = float(arm["trained_lm_loss"])
        per_layer[layer] = min(per_layer.get(layer, float("inf")), loss)
    layers = np.array(sorted(per_layer), dtype=float)
    delta_nll = np.array([per_layer[int(layer)] - teacher for layer in layers])

    pearson = stats.pearsonr(layers, delta_nll)
    spearman = stats.spearmanr(layers, delta_nll)

    design_linear = np.column_stack([np.ones_like(layers), layers])
    design_quadratic = np.column_stack([np.ones_like(layers), layers, layers**2])
    beta_linear, *_ = np.linalg.lstsq(design_linear, delta_nll, rcond=None)
    beta_quadratic, *_ = np.linalg.lstsq(design_quadratic, delta_nll, rcond=None)
    ss_linear = float(np.sum((delta_nll - design_linear @ beta_linear) ** 2))
    ss_quadratic = float(np.sum((delta_nll - design_quadratic @ beta_quadratic) ** 2))
    n = len(layers)
    f_statistic = (ss_linear - ss_quadratic) / (ss_quadratic / (n - 3))
    p_quadratic = float(1.0 - stats.f.cdf(f_statistic, 1, n - 3))
    vertex_layer = float(-beta_quadratic[1] / (2.0 * beta_quadratic[2]))

    return {
        "schema": "dendritic_depth_association_sensitivity/v2",
        "note": (
            "Eighth-review sensitivities with committed provenance: "
            "delta-NLL metric instead of exponentiated perplexity ratio; "
            "quadratic depth term F-tested against the linear model. "
            "Best-arm-per-layer convention and quarantine exclusions are "
            "inherited from the parent artifact."
        ),
        "metric": "delta_nll",
        "n_layers": int(n),
        "pearson_r": round(float(pearson[0]), 6),
        "pearson_p": float(pearson[1]),
        "spearman_rho": round(float(spearman[0]), 6),
        "spearman_p": float(spearman[1]),
        "quadratic_term_beta": float(beta_quadratic[2]),
        "quadratic_vs_linear_F": round(float(f_statistic), 4),
        "quadratic_vs_linear_p": round(p_quadratic, 6),
        "fitted_vertex_layer": round(vertex_layer, 2),
        "shape_statement": (
            "Within this 32B single-layer screen, degradation showed a "
            "convex relationship with depth: the fitted minimum occurred "
            f"near layer {vertex_layer:.0f}, after which degradation "
            "increased increasingly rapidly in later layers."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text())
    result = compute_sensitivities(artifact)
    result["parent_artifact"] = str(args.artifact.resolve())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=1))
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "pearson_r",
                    "spearman_rho",
                    "quadratic_vs_linear_F",
                    "quadratic_vs_linear_p",
                    "fitted_vertex_layer",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()

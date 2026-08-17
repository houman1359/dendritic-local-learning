#!/usr/bin/env python3
"""Run the frozen hierarchical signal--noise credit phase experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "credit_phase_theory" / "confirmatory.json"
CANARY = ROOT / "analysis" / "credit_phase_theory_canary.json"
OUTPUT = ROOT / "source_data" / "credit_phase_theory"


def load_phase_module():
    path = ROOT / "code" / "theory" / "credit_phase.py"
    spec = importlib.util.spec_from_file_location("credit_phase_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PHASE = load_phase_module()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def projector(basis: np.ndarray) -> np.ndarray:
    if basis.size == 0:
        return np.zeros((basis.shape[0], basis.shape[0]), dtype=float)
    q, _ = np.linalg.qr(np.asarray(basis, dtype=float), mode="reduced")
    return q @ q.T


def random_orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    return q * signs[None, :]


def level_variances(levels: np.ndarray, energy_by_level: list[float]) -> np.ndarray:
    result = np.zeros(len(levels), dtype=float)
    for level, energy in enumerate(energy_by_level):
        count = int(np.sum(levels == level))
        if count:
            result[levels == level] = float(energy) / count
    return result


def spectral_rows(seed: int, cfg: dict, basis: np.ndarray, levels: np.ndarray) -> list[dict]:
    phase = cfg["spectral_phase"]
    lambdas = level_variances(levels, phase["level_energy"])
    aligned = basis @ np.diag(lambdas) @ basis.T
    rng = np.random.default_rng(seed + 2_100_000)
    rotation = random_orthogonal(rng, len(levels))
    misaligned = rotation @ np.diag(lambdas) @ rotation.T
    rows = []
    for rho in phase["alignment"]:
        covariance = float(rho) * aligned + (1.0 - float(rho)) * misaligned
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, order]
        total = float(np.trace(covariance))
        for budget in phase["feedback_budgets"]:
            k = int(budget)
            ancestry = projector(basis[:, :k])
            random_basis = random_orthogonal(
                np.random.default_rng(seed + 2_200_000 + 101 * k), len(levels)
            )[:, :k]
            route_projectors = {
                "ancestry": ancestry,
                "random_rank": projector(random_basis),
                "dense_pca_upper_bound": projector(eigenvectors[:, :k]),
            }
            pca_capture = float(np.sum(eigenvalues[order][:k]) / total)
            for method, route_projector in route_projectors.items():
                capture = PHASE.spectral_capture(route_projector, covariance)
                rows.append(
                    {
                        "seed": seed,
                        "alignment": float(rho),
                        "budget_k": k,
                        "method": method,
                        "spectral_capture": capture,
                        "pca_rank_upper_bound": pca_capture,
                        "morphology_regret": pca_capture - capture,
                    }
                )
    return rows


def task_target(
    rng: np.random.Generator,
    basis: np.ndarray,
    levels: np.ndarray,
    hierarchy_depth: int,
) -> np.ndarray:
    active = levels <= int(hierarchy_depth)
    coefficients = np.zeros(len(levels), dtype=float)
    for level in range(int(hierarchy_depth) + 1):
        mask = levels == level
        coefficients[mask] = rng.normal(size=int(np.sum(mask))) / np.sqrt(
            max(int(np.sum(mask)), 1)
        )
    target = basis @ coefficients
    return target / max(np.linalg.norm(target), 1e-30)


def depth_rows(seed: int, cfg: dict, basis: np.ndarray, levels: np.ndarray) -> list[dict]:
    phase = cfg["depth_training"]
    rng = np.random.default_rng(seed + 2_300_000)
    permutation = rng.permutation(len(levels))
    permutation_matrix = np.eye(len(levels))[permutation]
    rows = []
    for task_depth in phase["task_depths"]:
        target = task_target(rng, basis, levels, int(task_depth))
        base_sd = float(phase["gradient_noise_sd"])
        multiplier = float(phase["fine_mode_noise_multiplier"])
        mode_sd = base_sd * (
            1.0 + multiplier * np.maximum(levels - int(task_depth), 0)
        )
        covariance = basis @ np.diag(mode_sd * mode_sd) @ basis.T
        common_noises = rng.multivariate_normal(
            np.zeros(len(levels)), covariance, size=int(phase["iterations"])
        )
        for model_depth in phase["model_depths"]:
            aligned = projector(basis[:, levels <= int(model_depth)])
            rewired = permutation_matrix.T @ aligned @ permutation_matrix
            for method, route_projector in (
                ("aligned_tree", aligned),
                ("rewired_tree", rewired),
            ):
                weights = np.zeros(len(levels), dtype=float)
                for noise in common_noises:
                    gradient = weights - target + noise
                    weights -= float(phase["step_size"]) * (route_projector @ gradient)
                initial_gradient = -target
                _, predicted = PHASE.optimal_credit_step(
                    initial_gradient,
                    route_projector,
                    covariance,
                    1.0,
                )
                rows.append(
                    {
                        "seed": seed,
                        "task_depth": int(task_depth),
                        "model_depth": int(model_depth),
                        "method": method,
                        "route_rank": int(round(np.trace(route_projector))),
                        "final_population_loss": float(
                            0.5 * np.sum((weights - target) ** 2)
                        ),
                        "target_signal_capture": float(
                            target @ route_projector @ target
                        ),
                        "admitted_noise": float(
                            np.trace(route_projector @ covariance @ route_projector.T)
                        ),
                        "initial_maximum_guaranteed_decrease": predicted,
                    }
                )
        # Full stochastic BP is the depth-four/full-rank reference.
        weights = np.zeros(len(levels), dtype=float)
        for noise in common_noises:
            weights -= float(phase["step_size"]) * (weights - target + noise)
        rows.append(
            {
                "seed": seed,
                "task_depth": int(task_depth),
                "model_depth": int(levels.max()),
                "method": "full_stochastic_bp",
                "route_rank": len(levels),
                "final_population_loss": float(0.5 * np.sum((weights - target) ** 2)),
                "target_signal_capture": 1.0,
                "admitted_noise": float(np.trace(covariance)),
                "initial_maximum_guaranteed_decrease": PHASE.optimal_credit_step(
                    -target, np.eye(len(levels)), covariance, 1.0
                )[1],
            }
        )
    return rows


def projection_rows(seed: int, cfg: dict) -> list[dict]:
    phase = cfg["projection_phase"]
    n = int(cfg["n_leaves"])
    k = n // 2
    route = np.diag(np.r_[np.ones(k), np.zeros(n - k)])
    rng = np.random.default_rng(seed + 2_400_000)
    rows = []
    signal_total = float(phase["global_gradient_signal"])
    noise_total = float(phase["global_gradient_noise"])
    eta = float(phase["step_size"])
    for r in phase["signal_retention"]:
        for noise_retention in phase["noise_retention"]:
            inside = rng.normal(size=k)
            inside /= np.linalg.norm(inside)
            outside = rng.normal(size=n - k)
            outside /= np.linalg.norm(outside)
            gradient = np.r_[
                np.sqrt(float(r) * signal_total) * inside,
                np.sqrt((1.0 - float(r)) * signal_total) * outside,
            ]
            variances = np.r_[
                np.full(k, float(noise_retention) * noise_total / k),
                np.full(
                    n - k,
                    (1.0 - float(noise_retention)) * noise_total / (n - k),
                ),
            ]
            noise = rng.normal(scale=np.sqrt(variances))
            local_variance = (
                float(phase["local_coefficient_noise_fraction"]) * noise_total / k
            )
            local_noise = np.r_[
                rng.normal(scale=np.sqrt(local_variance), size=k),
                np.zeros(n - k),
            ]
            methods = {
                "full_stochastic_bp": gradient + noise,
                "bp_plus_route_projection": route @ (gradient + noise),
                "routed_localca": route @ (gradient + noise) + local_noise,
            }
            optimum = -gradient
            for method, update_gradient in methods.items():
                weights = -eta * update_gradient
                empirical_loss = float(0.5 * np.sum((weights - optimum) ** 2))
                if method == "full_stochastic_bp":
                    expected_loss = 0.5 * noise_total
                elif method == "bp_plus_route_projection":
                    expected_loss = 0.5 * (
                        (1.0 - float(r)) * signal_total
                        + float(noise_retention) * noise_total
                    )
                else:
                    expected_loss = 0.5 * (
                        (1.0 - float(r)) * signal_total
                        + float(noise_retention) * noise_total
                        + float(phase["local_coefficient_noise_fraction"]) * noise_total
                    )
                rows.append(
                    {
                        "seed": seed,
                        "signal_retention": float(r),
                        "noise_retention": float(noise_retention),
                        "method": method,
                        "population_loss_after_one_step": empirical_loss,
                        "expected_population_loss": expected_loss,
                        "projected_bp_predicted_better_than_full": bool(
                            (1.0 - float(noise_retention)) * noise_total
                            > (1.0 - float(r)) * signal_total
                        ),
                    }
                )
    return rows


def reliability_rows(seed: int, cfg: dict) -> list[dict]:
    phase = cfg["reliability_phase"]
    branches = int(phase["branches"])
    axis = np.linspace(-1.0, 1.0, branches)
    rng = np.random.default_rng(seed + 2_500_000)
    rows = []
    for heterogeneity in phase["heterogeneity"]:
        h = float(heterogeneity)
        signal = np.exp(h * axis)
        noise_energy = np.exp(-h * axis)
        signal *= branches / signal.sum()
        noise_energy *= branches / noise_energy.sum()
        optimal = PHASE.reliability_shrinkage(signal, noise_energy)
        global_gain = float(signal.sum() / np.sum(signal + noise_energy))
        shuffled = optimal[rng.permutation(branches)]
        gains = {
            "no_shunting": np.ones(branches),
            "best_global_gain": np.full(branches, global_gain),
            "reliability_aligned": optimal,
            "shuffled_shunting": shuffled,
            "anti_aligned_shunting": optimal[::-1],
            "explicit_point_gate": optimal,
        }
        gradient = np.sqrt(signal)
        stochastic = gradient + rng.normal(scale=np.sqrt(noise_energy))
        initial_loss = 0.5 * float(signal.sum())
        for method, attenuation in gains.items():
            weights = -attenuation * stochastic
            optimum = -gradient
            final_loss = 0.5 * float(np.sum((weights - optimum) ** 2))
            guaranteed = float(
                np.sum(
                    attenuation * signal
                    - 0.5 * attenuation * attenuation * (signal + noise_energy)
                )
            )
            rows.append(
                {
                    "seed": seed,
                    "reliability_heterogeneity": h,
                    "method": method,
                    "population_loss_after_one_step": final_loss,
                    "population_loss_decrease": initial_loss - final_loss,
                    "expected_guaranteed_decrease": guaranteed,
                    "mean_attenuation": float(np.mean(attenuation)),
                    "attenuation_reliability_correlation": float(
                        np.corrcoef(
                            attenuation,
                            signal / (signal + noise_energy),
                        )[0, 1]
                    )
                    if np.std(attenuation) > 1e-14
                    else 0.0,
                }
            )
    return rows


def raw_nested_dictionary(n: int, max_depth: int) -> np.ndarray:
    columns = []
    for level in range(max_depth + 1):
        blocks = 2**level
        width = n // blocks
        for block in range(blocks):
            column = np.zeros(n, dtype=float)
            column[block * width : (block + 1) * width] = 1.0
            columns.append(column)
    return np.column_stack(columns)


def same_span_rows(cfg: dict, basis: np.ndarray, levels: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(2_600_000)
    covariance = random_orthogonal(rng, len(levels)) @ np.diag(
        np.linspace(2.0, 0.2, len(levels))
    ) @ random_orthogonal(np.random.default_rng(2_600_001), len(levels)).T
    covariance = covariance @ covariance.T
    rows = []
    for depth in range(1, int(levels.max()) + 1):
        raw = raw_nested_dictionary(len(levels), depth)
        haar = basis[:, levels <= depth]
        beta = np.exp(rng.normal(scale=1.0, size=raw.shape[1]))
        scaled = raw @ np.diag(beta)
        p_raw = PHASE.weighted_projector(raw, np.ones(len(levels)))
        p_haar = PHASE.weighted_projector(haar, np.ones(len(levels)))
        p_scaled = PHASE.weighted_projector(scaled, np.ones(len(levels)))
        for name, dictionary, route_projector in (
            ("raw_nested_indicators", raw, p_raw),
            ("tree_haar", haar, p_haar),
            ("static_gain_scaled_nested", scaled, p_scaled),
        ):
            diagnostics = PHASE.route_gram_diagnostics(
                dictionary, np.ones(len(levels))
            )
            rows.append(
                {
                    "depth": depth,
                    "basis": name,
                    "rank": diagnostics["rank"],
                    "condition_number": diagnostics["condition_number"],
                    "coherence": diagnostics["coherence"],
                    "spectral_capture": PHASE.spectral_capture(
                        route_projector, covariance
                    ),
                    "projector_difference_from_haar": float(
                        np.max(np.abs(route_projector - p_haar))
                    ),
                }
            )
    return rows


def bootstrap(values: np.ndarray, seed: int, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(int(draws), len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize_table(
    frame: pd.DataFrame, groups: list[str], metrics: list[str], draws: int, seed: int
) -> pd.DataFrame:
    rows = []
    for group_index, (keys, part) in enumerate(frame.groupby(groups, sort=True)):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(groups, keys))
        row["n_seeds"] = int(part.seed.nunique()) if "seed" in part else 0
        for metric_index, metric in enumerate(metrics):
            mean, low, high = bootstrap(
                part[metric].to_numpy(float),
                seed + 100 * group_index + metric_index,
                draws,
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_contrast(
    frame: pd.DataFrame,
    index: list[str],
    column: str,
    left: str,
    right: str,
    metric: str,
    draws: int,
    seed: int,
) -> dict:
    wide = frame.pivot_table(index=index, columns=column, values=metric)
    values = (wide[left] - wide[right]).dropna().to_numpy(float)
    mean, low, high = bootstrap(values, seed, draws)
    pvalue = 1.0 if np.allclose(values, 0) else float(
        wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue
    )
    return {
        "left_minus_right": f"{left} - {right}",
        "metric": metric,
        "n_pairs": int(len(values)),
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "positive_pairs": int(np.sum(values > 0)),
        "ties": int(np.sum(np.isclose(values, 0))),
        "wilcoxon_p_two_sided": pvalue,
    }


def numerical_gates(cfg: dict, basis: np.ndarray, same_span: pd.DataFrame, reliability: pd.DataFrame) -> dict:
    identity = np.eye(len(basis))
    ancestry = projector(basis[:, :8])
    aligned = reliability[reliability.method.eq("reliability_aligned")].sort_values(
        ["seed", "reliability_heterogeneity"]
    )
    point = reliability[reliability.method.eq("explicit_point_gate")].sort_values(
        ["seed", "reliability_heterogeneity"]
    )
    payload = {
        "maximum_orthogonality_error": float(np.max(np.abs(basis.T @ basis - identity))),
        "maximum_projector_idempotence_error": float(
            np.max(np.abs(ancestry @ ancestry - ancestry))
        ),
        "maximum_static_gain_span_difference": float(
            same_span[
                same_span.basis.eq("static_gain_scaled_nested")
            ].projector_difference_from_haar.max()
        ),
        "maximum_explicit_gate_equivalence_error": float(
            np.max(
                np.abs(
                    aligned.population_loss_after_one_step.to_numpy()
                    - point.population_loss_after_one_step.to_numpy()
                )
            )
        ),
    }
    gates = cfg["gates"]
    payload["passed"] = bool(
        payload["maximum_orthogonality_error"]
        <= float(gates["maximum_orthogonality_error"])
        and payload["maximum_projector_idempotence_error"]
        <= float(gates["maximum_projector_idempotence_error"])
        and payload["maximum_static_gain_span_difference"]
        <= float(gates["maximum_static_gain_span_difference"])
        and payload["maximum_explicit_gate_equivalence_error"]
        <= float(gates["maximum_explicit_gate_equivalence_error"])
    )
    return payload


def run(seeds: list[int], cfg: dict):
    basis, levels = PHASE.tree_haar_basis(int(cfg["n_leaves"]))
    spectral, depth, projection, reliability = [], [], [], []
    for seed in seeds:
        spectral.extend(spectral_rows(seed, cfg, basis, levels))
        depth.extend(depth_rows(seed, cfg, basis, levels))
        projection.extend(projection_rows(seed, cfg))
        reliability.extend(reliability_rows(seed, cfg))
    return (
        pd.DataFrame(spectral),
        pd.DataFrame(depth),
        pd.DataFrame(projection),
        pd.DataFrame(reliability),
        pd.DataFrame(same_span_rows(cfg, basis, levels)),
        basis,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["canary", "confirmatory"], required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    seeds = [int(value) for value in cfg[f"{args.phase}_seeds"]]
    spectral, depth, projection, reliability, same_span, basis = run(seeds, cfg)
    if not all(
        np.isfinite(frame.select_dtypes(include=[np.number])).all().all()
        for frame in (spectral, depth, projection, reliability, same_span)
    ):
        raise SystemExit("non-finite phase-theory outcome")
    gates = numerical_gates(cfg, basis, same_span, reliability)
    if args.phase == "canary":
        payload = {
            "phase": "artifact_only_canary",
            "config_sha256": digest(CONFIG),
            "script_sha256": digest(Path(__file__).resolve()),
            "reference_sha256": digest(ROOT / "code" / "theory" / "credit_phase.py"),
            "numerical_gates": gates,
            "outcomes_are_not_confirmatory": True,
        }
        CANARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        if not gates["passed"]:
            raise SystemExit("canary failed")
        return

    if not CANARY.is_file():
        raise SystemExit("passing canary required")
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    if not canary["numerical_gates"]["passed"]:
        raise SystemExit("canary did not pass")
    if (
        canary["config_sha256"] != digest(CONFIG)
        or canary["script_sha256"] != digest(Path(__file__).resolve())
        or canary["reference_sha256"]
        != digest(ROOT / "code" / "theory" / "credit_phase.py")
    ):
        raise SystemExit("code, reference, or configuration changed after canary")
    expected = {
        "spectral": len(seeds)
        * len(cfg["spectral_phase"]["alignment"])
        * len(cfg["spectral_phase"]["feedback_budgets"])
        * 3,
        "depth": len(seeds)
        * len(cfg["depth_training"]["task_depths"])
        * (2 * len(cfg["depth_training"]["model_depths"]) + 1),
        "projection": len(seeds)
        * len(cfg["projection_phase"]["signal_retention"])
        * len(cfg["projection_phase"]["noise_retention"])
        * 3,
        "reliability": len(seeds)
        * len(cfg["reliability_phase"]["heterogeneity"])
        * 6,
    }
    observed = {
        "spectral": len(spectral),
        "depth": len(depth),
        "projection": len(projection),
        "reliability": len(reliability),
    }
    if observed != expected:
        raise SystemExit(f"incomplete phase factorial: {observed} != {expected}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    spectral.to_csv(OUTPUT / "spectral_phase_seed.csv", index=False, float_format="%.10g")
    depth.to_csv(OUTPUT / "depth_training_seed.csv", index=False, float_format="%.10g")
    projection.to_csv(OUTPUT / "projection_phase_seed.csv", index=False, float_format="%.10g")
    reliability.to_csv(OUTPUT / "reliability_phase_seed.csv", index=False, float_format="%.10g")
    same_span.to_csv(OUTPUT / "same_span_diagnostics.csv", index=False, float_format="%.10g")
    draws = int(cfg["analysis"]["bootstrap_draws"])
    summarize_table(
        spectral,
        ["alignment", "budget_k", "method"],
        ["spectral_capture", "morphology_regret"],
        draws,
        2_700_000,
    ).to_csv(OUTPUT / "spectral_phase_summary.csv", index=False, float_format="%.10g")
    summarize_table(
        depth,
        ["task_depth", "model_depth", "method"],
        ["final_population_loss", "target_signal_capture", "admitted_noise"],
        draws,
        2_710_000,
    ).to_csv(OUTPUT / "depth_training_summary.csv", index=False, float_format="%.10g")
    summarize_table(
        projection,
        ["signal_retention", "noise_retention", "method"],
        ["population_loss_after_one_step", "expected_population_loss"],
        draws,
        2_720_000,
    ).to_csv(OUTPUT / "projection_phase_summary.csv", index=False, float_format="%.10g")
    summarize_table(
        reliability,
        ["reliability_heterogeneity", "method"],
        ["population_loss_after_one_step", "population_loss_decrease", "expected_guaranteed_decrease"],
        draws,
        2_730_000,
    ).to_csv(OUTPUT / "reliability_phase_summary.csv", index=False, float_format="%.10g")

    contrasts = []
    # Positive values favor ancestry capture.
    contrasts.append(
        {
            "analysis": "spectral_alignment_rho1_k4",
            **paired_contrast(
                spectral[
                    spectral.alignment.eq(1.0) & spectral.budget_k.eq(4)
                ],
                ["seed", "alignment", "budget_k"],
                "method",
                "ancestry",
                "random_rank",
                "spectral_capture",
                draws,
                2_740_001,
            ),
        }
    )
    # Negative values favor aligned tree because the endpoint is loss.
    matched = depth[
        depth.method.eq("aligned_tree")
        & depth.task_depth.eq(depth.model_depth)
    ]
    mismatch = depth[
        depth.method.eq("aligned_tree")
        & ~depth.task_depth.eq(depth.model_depth)
    ]
    best_mismatch = (
        mismatch.groupby(["seed", "task_depth"], as_index=False)
        .final_population_loss.min()
        .rename(columns={"final_population_loss": "best_mismatched"})
    )
    matched_values = matched[["seed", "task_depth", "final_population_loss"]].merge(
        best_mismatch, on=["seed", "task_depth"], validate="one_to_one"
    )
    depth_difference = (
        matched_values.final_population_loss - matched_values.best_mismatched
    ).to_numpy(float)
    mean, low, high = bootstrap(depth_difference, 2_740_002, draws)
    contrasts.append(
        {
            "analysis": "matched_depth_minus_best_mismatched_depth",
            "left_minus_right": "D=H - best D!=H",
            "metric": "final_population_loss",
            "n_pairs": len(depth_difference),
            "mean_difference": mean,
            "ci95_low": low,
            "ci95_high": high,
            "positive_pairs": int(np.sum(depth_difference > 0)),
            "ties": int(np.sum(np.isclose(depth_difference, 0))),
            "wilcoxon_p_two_sided": float(
                wilcoxon(depth_difference, zero_method="wilcox").pvalue
            ),
        }
    )
    high_heterogeneity = reliability[
        reliability.reliability_heterogeneity.eq(
            max(cfg["reliability_phase"]["heterogeneity"])
        )
    ]
    contrasts.append(
        {
            "analysis": "reliability_alignment_high_heterogeneity",
            **paired_contrast(
                high_heterogeneity,
                ["seed", "reliability_heterogeneity"],
                "method",
                "reliability_aligned",
                "best_global_gain",
                "population_loss_decrease",
                draws,
                2_740_003,
            ),
        }
    )
    pd.DataFrame(contrasts).to_csv(
        OUTPUT / "primary_contrasts.csv", index=False, float_format="%.10g"
    )

    ancestry_full = spectral[
        spectral.method.eq("ancestry")
        & spectral.budget_k.eq(int(cfg["n_leaves"]))
    ].spectral_capture
    pca_full = spectral[
        spectral.method.eq("dense_pca_upper_bound")
        & spectral.budget_k.eq(int(cfg["n_leaves"]))
    ].spectral_capture
    gate_equality = float(
        np.max(
            np.abs(
                reliability[reliability.method.eq("reliability_aligned")]
                .sort_values(["seed", "reliability_heterogeneity"])
                .population_loss_after_one_step.to_numpy()
                - reliability[reliability.method.eq("explicit_point_gate")]
                .sort_values(["seed", "reliability_heterogeneity"])
                .population_loss_after_one_step.to_numpy()
            )
        )
    )
    metadata = {
        "study": cfg["study"],
        "status": "complete_confirmatory",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "config_sha256": digest(CONFIG),
        "script_sha256": digest(Path(__file__).resolve()),
        "reference_sha256": digest(ROOT / "code" / "theory" / "credit_phase.py"),
        "n_seeds": len(seeds),
        "row_counts": observed,
        "maximum_full_rank_capture_difference": float(
            np.max(np.abs(ancestry_full.to_numpy() - pca_full.to_numpy()))
        ),
        "maximum_explicit_point_gate_difference": gate_equality,
        "scope_boundary": cfg["scope_boundary"],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = f"""# Hierarchical credit signal--noise phase experiment

The frozen confirmatory programme contains {len(seeds)} independent seeds and
{sum(observed.values()):,} seed-level condition rows. It tests spectral
task--tree alignment, stochastic hierarchy-depth matching, projection denoising,
and branch-reliability shrinkage in one fixed-state quadratic family.

The experiment preserves two important equivalences: all full-rank route
representations have identical capture to numerical precision
({metadata['maximum_full_rank_capture_difference']:.3g}), and an explicit point
gate supplied with the same reliability factors matches the routed attenuation
exactly ({gate_equality:.3g}). The latter locates any benefit in the structured
gain prior rather than in an impossible-to-emulate dendritic operation.

This is a phase-theory validation, not a positive-conductance forward-network
experiment and not a claim of uniform superiority to exact full-batch BP.
"""
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

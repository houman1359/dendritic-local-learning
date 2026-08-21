#!/usr/bin/env python3
"""Run the frozen adaptive local conductance-reliability experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parents[1]
BASE_PATH = ROOT / "scripts" / "run_positive_conductance_reliability.py"
CONFIG = ROOT / "configs" / "positive_conductance_reliability" / "adaptive_local_confirmatory.json"
CANARY = ROOT / "analysis" / "adaptive_conductance_reliability_canary.json"
OUTPUT = ROOT / "source_data" / "adaptive_conductance_reliability"


def _load_base():
    spec = importlib.util.spec_from_file_location("positive_reliability_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def observed_moments(
    q: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    center: float,
    cfg: dict,
    noise_a: np.ndarray,
    noise_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    gradient_a = BASE.loss_and_gradient(
        q, x, labels, center, cfg, coefficient_noise=noise_a,
        update_mode="unshunted",
    )["gradient"]
    gradient_b = BASE.loss_and_gradient(
        q, x, labels, center, cfg, coefficient_noise=noise_b,
        update_mode="unshunted",
    )["gradient"]
    signal = np.sum(gradient_a * gradient_b, axis=1)
    noise = 0.5 * np.sum((gradient_a - gradient_b) ** 2, axis=1)
    return signal, noise


def gain_from_moments(
    signal: np.ndarray,
    noise: np.ndarray,
    cfg: dict,
    *,
    global_gain: bool = False,
) -> np.ndarray:
    floor = float(cfg["estimator"]["minimum_moment"])
    signal = np.maximum(np.asarray(signal, dtype=float), floor)
    noise = np.maximum(np.asarray(noise, dtype=float), floor)
    if global_gain:
        signal = np.full_like(signal, signal.sum())
        noise = np.full_like(noise, noise.sum())
    gain = BASE.fixed_step_reliability_gain(
        signal,
        noise,
        float(cfg["training"]["smoothness_safety_factor"]),
    )
    return np.clip(
        gain,
        float(cfg["training"]["minimum_reliability_gain"]),
        1.0,
    )


def initialize_estimator(
    q: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    center: float,
    cfg: dict,
    probe_pairs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    observations = [
        observed_moments(q, x, labels, center, cfg, left, right)
        for left, right in probe_pairs
    ]
    signal = np.mean([item[0] for item in observations], axis=0)
    noise = np.mean([item[1] for item in observations], axis=0)
    return signal, noise


def update_estimator(
    old_signal: np.ndarray,
    old_noise: np.ndarray,
    observed_signal: np.ndarray,
    observed_noise: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray]:
    decay = float(cfg["estimator"]["ema_decay"])
    return (
        decay * old_signal + (1.0 - decay) * observed_signal,
        decay * old_noise + (1.0 - decay) * observed_noise,
    )


def _noise_arrays(
    rng: np.random.Generator,
    count: int,
    examples: int,
    coefficient_sd: np.ndarray,
) -> list[np.ndarray]:
    return [
        rng.normal(size=(examples, len(coefficient_sd))) * coefficient_sd[None, :]
        for _ in range(int(count))
    ]


def run_seed(seed: int, cfg: dict) -> tuple[list[dict], list[dict], dict]:
    rng = np.random.default_rng(seed)
    branches = int(cfg["task"]["branches"])
    features = int(cfg["task"]["features_per_branch"])
    teachers = rng.uniform(0.6, 1.4, size=(branches, features))
    teachers /= teachers.mean(axis=1, keepdims=True)
    x_train, y_train = BASE.make_data(
        rng, int(cfg["task"]["train_examples"]), cfg, teachers
    )
    x_test, y_test = BASE.make_data(
        rng, int(cfg["task"]["test_examples"]), cfg, teachers
    )
    q_initial = np.full(
        (branches, features),
        BASE.inverse_softplus(float(cfg["task"]["initial_excitatory_conductance"])),
    )
    center = float(BASE.base_state(q_initial, x_train, cfg)[3].mean())
    smoothness, finite_difference_error = BASE.local_smoothness(
        q_initial, x_train, y_train, center, cfg
    )
    step_size = float(cfg["training"]["smoothness_safety_factor"]) / smoothness
    iterations = int(cfg["training"]["iterations"])
    calibration_pairs = int(cfg["estimator"]["calibration_pairs"])

    rows: list[dict] = []
    estimator_rows: list[dict] = []
    maximum_state_error = 0.0
    maximum_point_error = 0.0
    maximum_physical_fd_error = 0.0
    minimum_gain = 1.0
    maximum_gain = 0.0

    for heterogeneity in cfg["reliability_heterogeneity"]:
        h = float(heterogeneity)
        (
            _,
            _,
            coefficient_sd,
            _,
            oracle_gain,
            _,
            _,
            initial_state,
        ) = BASE.initial_reliability(q_initial, x_train, y_train, center, h, cfg)
        mean_total_initial = initial_state["total"].mean(axis=0)
        oracle_shunt = BASE.shunt_for_gain(oracle_gain, mean_total_initial, cfg)
        maximum_physical_fd_error = max(
            maximum_physical_fd_error,
            BASE.physical_shunt_finite_difference_error(
                q_initial, x_train, y_train, center, cfg, oracle_shunt
            ),
        )

        noise_rng = np.random.default_rng(seed + 5_100_000 + int(100 * h))
        main_noise = _noise_arrays(
            noise_rng, iterations, len(x_train), coefficient_sd
        )
        probe_noise = _noise_arrays(
            noise_rng,
            2 * (calibration_pairs + iterations),
            len(x_train),
            coefficient_sd,
        )
        calibration = [
            (probe_noise[2 * index], probe_noise[2 * index + 1])
            for index in range(calibration_pairs)
        ]
        offset = 2 * calibration_pairs
        probes = [
            (probe_noise[offset + 2 * index], probe_noise[offset + 2 * index + 1])
            for index in range(iterations)
        ]
        permutation = np.random.default_rng(
            seed + 5_200_000 + int(100 * h)
        ).permutation(branches)

        trajectories: dict[str, np.ndarray] = {}
        final_gains: dict[str, np.ndarray] = {}
        for method in cfg["methods"]:
            q = q_initial.copy()
            estimated_signal, estimated_noise = initialize_estimator(
                q, x_train, y_train, center, cfg, calibration
            )
            initial_test = BASE.loss_and_gradient(q, x_test, y_test, center, cfg)
            one_step = None
            used_gain = np.ones(branches, dtype=float)
            for iteration in range(iterations):
                if method.startswith("adaptive_"):
                    observed_signal, observed_noise = observed_moments(
                        q,
                        x_train,
                        y_train,
                        center,
                        cfg,
                        probes[iteration][0],
                        probes[iteration][1],
                    )
                    estimated_signal, estimated_noise = update_estimator(
                        estimated_signal,
                        estimated_noise,
                        observed_signal,
                        observed_noise,
                        cfg,
                    )
                    used_gain = gain_from_moments(
                        estimated_signal,
                        estimated_noise,
                        cfg,
                        global_gain=method == "adaptive_global_shunt",
                    )
                    if method == "adaptive_shuffled_shunt":
                        used_gain = used_gain[permutation]
                    mean_total = BASE.base_state(q, x_train, cfg)[2].mean(axis=0)
                    shunt = BASE.shunt_for_gain(used_gain, mean_total, cfg)
                    update_mode = (
                        "point_gate" if method == "adaptive_point_gate" else "physical_shunt"
                    )
                    used_noise = main_noise[iteration]
                elif method == "initial_oracle_shunt":
                    used_gain = oracle_gain
                    shunt = oracle_shunt
                    update_mode = "physical_shunt"
                    used_noise = main_noise[iteration]
                elif method == "noisy_no_shunt":
                    shunt = np.zeros(branches)
                    update_mode = "unshunted"
                    used_noise = main_noise[iteration]
                elif method == "exact_clean_bp":
                    shunt = np.zeros(branches)
                    update_mode = "unshunted"
                    used_noise = None
                else:
                    raise ValueError(method)

                minimum_gain = min(minimum_gain, float(np.min(used_gain)))
                maximum_gain = max(maximum_gain, float(np.max(used_gain)))
                state = BASE.loss_and_gradient(
                    q,
                    x_train,
                    y_train,
                    center,
                    cfg,
                    shunt=shunt,
                    coefficient_noise=used_noise,
                    update_mode=update_mode,
                )
                maximum_state_error = max(
                    maximum_state_error,
                    float(np.max(np.abs(state["matched_voltage"] - state["voltage"]))),
                )
                q -= step_size * state["gradient"]
                if iteration == 0:
                    one_step = BASE.loss_and_gradient(q, x_test, y_test, center, cfg)

            final_train = BASE.loss_and_gradient(q, x_train, y_train, center, cfg)
            final_test = BASE.loss_and_gradient(q, x_test, y_test, center, cfg)
            assert one_step is not None
            rows.append(
                {
                    "seed": seed,
                    "reliability_heterogeneity": h,
                    "method": method,
                    "step_size": step_size,
                    "initial_test_loss": initial_test["loss"],
                    "test_loss_after_one_step": one_step["loss"],
                    "one_step_test_loss_decrease": initial_test["loss"] - one_step["loss"],
                    "final_train_loss": final_train["loss"],
                    "final_test_loss": final_test["loss"],
                    "final_test_accuracy": final_test["accuracy"],
                    "mean_final_gain": float(np.mean(used_gain)),
                    "minimum_input_rate": float(min(x_train.min(), x_test.min())),
                }
            )
            trajectories[method] = q.copy()
            final_gains[method] = used_gain.copy()

        maximum_point_error = max(
            maximum_point_error,
            float(
                np.max(
                    np.abs(
                        trajectories["adaptive_local_shunt"]
                        - trajectories["adaptive_point_gate"]
                    )
                )
            ),
        )
        for branch in range(branches):
            estimator_rows.append(
                {
                    "seed": seed,
                    "reliability_heterogeneity": h,
                    "branch": branch,
                    "oracle_initial_gain": float(oracle_gain[branch]),
                    "adaptive_final_gain": float(final_gains["adaptive_local_shunt"][branch]),
                }
            )

    gates = {
        "minimum_input_rate": float(min(x_train.min(), x_test.min())),
        "maximum_state_match_error": maximum_state_error,
        "maximum_adaptive_point_gate_trajectory_error": maximum_point_error,
        "maximum_relative_finite_difference_error": finite_difference_error,
        "maximum_physical_shunt_finite_difference_error": maximum_physical_fd_error,
        "minimum_estimated_gain": minimum_gain,
        "maximum_estimated_gain": maximum_gain,
    }
    return rows, estimator_rows, gates


def bootstrap(values: np.ndarray, seed: int, draws: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def paired(
    frame: pd.DataFrame,
    heterogeneity: float,
    left: str,
    right: str,
    metric: str,
    cfg: dict,
    seed: int,
) -> dict:
    part = frame[np.isclose(frame.reliability_heterogeneity, heterogeneity)]
    wide = part.pivot(index="seed", columns="method", values=metric)
    values = (wide[left] - wide[right]).to_numpy(float)
    mean, low, high = bootstrap(values, seed, int(cfg["analysis"]["bootstrap_draws"]))
    return {
        "heterogeneity": heterogeneity,
        "left_minus_right": f"{left} - {right}",
        "metric": metric,
        "n_pairs": len(values),
        "mean_difference": mean,
        "ci95_low": low,
        "ci95_high": high,
        "positive_pairs": int(np.sum(values > 0)),
        "wilcoxon_p_two_sided": 1.0 if np.allclose(values, 0) else float(
            wilcoxon(values, alternative="two-sided").pvalue
        ),
    }


def summarize(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows = []
    metrics = ["one_step_test_loss_decrease", "final_test_loss", "final_test_accuracy"]
    for group_index, ((h, method), part) in enumerate(
        frame.groupby(["reliability_heterogeneity", "method"], sort=True)
    ):
        row = {
            "reliability_heterogeneity": h,
            "method": method,
            "n_seeds": int(part.seed.nunique()),
        }
        for metric_index, metric in enumerate(metrics):
            mean, low, high = bootstrap(
                part[metric].to_numpy(float),
                5_300_000 + 100 * group_index + metric_index,
                int(cfg["analysis"]["bootstrap_draws"]),
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_gates(gates: list[dict], cfg: dict) -> dict:
    combined = {
        "minimum_input_rate": min(item["minimum_input_rate"] for item in gates),
        "maximum_state_match_error": max(item["maximum_state_match_error"] for item in gates),
        "maximum_adaptive_point_gate_trajectory_error": max(
            item["maximum_adaptive_point_gate_trajectory_error"] for item in gates
        ),
        "maximum_relative_finite_difference_error": max(
            item["maximum_relative_finite_difference_error"] for item in gates
        ),
        "maximum_physical_shunt_finite_difference_error": max(
            item["maximum_physical_shunt_finite_difference_error"] for item in gates
        ),
        "minimum_estimated_gain": min(item["minimum_estimated_gain"] for item in gates),
        "maximum_estimated_gain": max(item["maximum_estimated_gain"] for item in gates),
    }
    limits = cfg["gates"]
    combined["passed"] = bool(
        combined["minimum_input_rate"] >= float(limits["minimum_input"])
        and combined["maximum_state_match_error"] <= float(limits["maximum_state_match_error"])
        and combined["maximum_adaptive_point_gate_trajectory_error"]
        <= float(limits["maximum_adaptive_point_gate_trajectory_error"])
        and combined["maximum_relative_finite_difference_error"]
        <= float(limits["maximum_relative_finite_difference_error"])
        and combined["maximum_physical_shunt_finite_difference_error"]
        <= float(limits["maximum_physical_shunt_finite_difference_error"])
        and combined["minimum_estimated_gain"] >= float(limits["minimum_estimated_gain"])
        and combined["maximum_estimated_gain"] <= float(limits["maximum_estimated_gain"])
    )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["canary", "confirmatory"], required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    hashes = {
        "config_sha256": digest(CONFIG),
        "script_sha256": digest(Path(__file__).resolve()),
        "base_script_sha256": digest(BASE_PATH),
    }
    all_rows: list[dict] = []
    all_estimators: list[dict] = []
    all_gates: list[dict] = []
    for seed in cfg[f"{args.phase}_seeds"]:
        rows, estimators, gates = run_seed(int(seed), cfg)
        all_rows.extend(rows)
        all_estimators.extend(estimators)
        all_gates.append(gates)
    frame = pd.DataFrame(all_rows)
    estimator = pd.DataFrame(all_estimators)
    gates = aggregate_gates(all_gates, cfg)

    if args.phase == "canary":
        payload = {
            "phase": "artifact_only_canary",
            **hashes,
            "numerical_gates": gates,
            "outcomes_are_not_confirmatory": True,
        }
        CANARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        if not gates["passed"]:
            raise SystemExit("adaptive reliability canary failed")
        return

    if not CANARY.is_file():
        raise SystemExit("passing canary required")
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    if not canary["numerical_gates"]["passed"] or any(
        canary[key] != value for key, value in hashes.items()
    ):
        raise SystemExit("canary failed or code/configuration changed")
    expected = (
        len(cfg["confirmatory_seeds"])
        * len(cfg["reliability_heterogeneity"])
        * len(cfg["methods"])
    )
    if len(frame) != expected or frame.duplicated(
        ["seed", "reliability_heterogeneity", "method"]
    ).any():
        raise SystemExit(f"incomplete adaptive factorial: {len(frame)} != {expected}")
    if not np.isfinite(frame.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("non-finite adaptive outcome")
    if not gates["passed"]:
        raise SystemExit(f"confirmatory numerical gates failed: {gates}")

    high = float(cfg["analysis"]["primary_heterogeneity"])
    comparisons = [
        ("adaptive_local_shunt", "adaptive_global_shunt"),
        ("adaptive_local_shunt", "adaptive_shuffled_shunt"),
        ("adaptive_local_shunt", "noisy_no_shunt"),
        ("adaptive_local_shunt", "initial_oracle_shunt"),
        ("adaptive_local_shunt", "adaptive_point_gate"),
    ]
    contrasts = []
    for index, (left, right) in enumerate(comparisons):
        for metric in ("one_step_test_loss_decrease", "final_test_loss"):
            contrasts.append(
                paired(
                    frame,
                    high,
                    left,
                    right,
                    metric,
                    cfg,
                    5_400_000 + 10 * index + len(contrasts),
                )
            )
    high_estimator = estimator[np.isclose(estimator.reliability_heterogeneity, high)]
    estimator_rhos = []
    for seed, part in high_estimator.groupby("seed", sort=True):
        estimator_rhos.append(
            {
                "seed": int(seed),
                "spearman_rho_estimated_vs_initial_oracle_gain": float(
                    spearmanr(part.oracle_initial_gain, part.adaptive_final_gain).statistic
                ),
                "mean_absolute_gain_error": float(
                    np.mean(np.abs(part.oracle_initial_gain - part.adaptive_final_gain))
                ),
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "seed_outcomes.csv", index=False, float_format="%.10g")
    estimator.to_csv(OUTPUT / "branch_estimates.csv", index=False, float_format="%.10g")
    summarize(frame, cfg).to_csv(
        OUTPUT / "condition_summary.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(contrasts).to_csv(
        OUTPUT / "paired_contrasts.csv", index=False, float_format="%.10g"
    )
    pd.DataFrame(estimator_rhos).to_csv(
        OUTPUT / "estimator_seed_summary.csv", index=False, float_format="%.10g"
    )
    summary = {
        "study": cfg["study"],
        "status": "complete_confirmatory",
        "n_seeds": len(cfg["confirmatory_seeds"]),
        "row_count": len(frame),
        "python": platform.python_version(),
        "numpy": np.__version__,
        **hashes,
        "numerical_gates": gates,
        "scope_boundary": cfg["scope_boundary"],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

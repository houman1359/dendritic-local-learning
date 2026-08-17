#!/usr/bin/env python3
"""Run the step-consistent state-matched conductance reliability experiment.

The August 11 pilot used the reliability optimum derived for ``eta=1/L``
while taking a half-smoothness step.  This prospective correction uses the
general fixed-step optimum and computes the physical-shunt, point-gate and
state-clamped unattenuated controls through separate update paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "theory"))
from credit_phase import fixed_step_reliability_gain  # noqa: E402

CONFIG = (
    ROOT
    / "configs"
    / "positive_conductance_reliability"
    / "step_consistent_confirmatory.json"
)
CANARY = ROOT / "analysis" / "positive_conductance_reliability_step_consistent_canary.json"
OUTPUT = ROOT / "source_data" / "positive_conductance_reliability_step_consistent"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exponential = np.exp(x[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))


def inverse_softplus(value: float) -> float:
    return float(np.log(np.expm1(float(value))))


def make_data(rng: np.random.Generator, n: int, cfg: dict, teachers: np.ndarray):
    task = cfg["task"]
    labels = np.tile(np.array([0.0, 1.0]), int(np.ceil(n / 2)))[:n]
    rng.shuffle(labels)
    sign = 2.0 * labels - 1.0
    noise = rng.normal(
        scale=float(task["input_noise_sd"]),
        size=(n, int(task["branches"]), int(task["features_per_branch"])),
    )
    rates = (
        float(task["input_baseline"])
        + sign[:, None, None] * float(task["label_signal"]) * teachers[None, :, :]
        + noise
    )
    # The positive-conductance model never receives a signed rate.
    return np.maximum(rates, 0.0), labels


def base_state(q: np.ndarray, x: np.ndarray, cfg: dict):
    leak = float(cfg["task"]["leak_conductance"])
    conductance = softplus(q)
    excitatory = np.einsum("nbd,bd->nb", x, conductance)
    total = leak + excitatory
    voltage = excitatory / total
    return conductance, excitatory, total, voltage


def loss_and_gradient(
    q: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    center: float,
    cfg: dict,
    shunt: np.ndarray | None = None,
    coefficient_noise: np.ndarray | None = None,
    update_mode: str = "unshunted",
):
    branches = int(cfg["task"]["branches"])
    output_gain = float(cfg["task"]["output_gain"])
    _, excitatory, total, voltage = base_state(q, x, cfg)
    if shunt is None:
        shunt = np.zeros(branches, dtype=float)
    shunt = np.asarray(shunt, dtype=float)
    if np.any(shunt < 0):
        raise ValueError("positive-conductance shunts must be nonnegative")

    # The clamp current is computed at the pre-intervention state and held
    # fixed while differentiating the local step.  Its value makes the
    # shunted forward state identical at the current parameter vector.
    clamp_current = shunt[None, :] * voltage
    matched_voltage = (excitatory + clamp_current) / (total + shunt[None, :])
    if update_mode in {"physical_shunt", "point_gate", "state_clamped_unattenuated"}:
        forward_voltage = matched_voltage
    elif update_mode == "unshunted":
        forward_voltage = voltage
    else:
        raise ValueError(f"unknown update mode: {update_mode}")

    logits = output_gain * (forward_voltage.mean(axis=1) - float(center))
    probability = sigmoid(logits)
    tiny = 1e-12
    loss = -float(
        np.mean(labels * np.log(probability + tiny) + (1.0 - labels) * np.log(1.0 - probability + tiny))
    )
    coefficient = (probability - labels) * output_gain / branches
    if coefficient_noise is not None:
        coefficient = coefficient[:, None] + coefficient_noise
    else:
        coefficient = np.repeat(coefficient[:, None], branches, axis=1)
    raw_jacobian = sigmoid(q)
    base_eligibility = (
        x
        * (1.0 - voltage)[:, :, None]
        / total[:, :, None]
        * raw_jacobian[None, :, :]
    )
    attenuation = total / (total + shunt[None, :])
    if update_mode == "physical_shunt":
        eligibility = (
            x
            * (1.0 - matched_voltage)[:, :, None]
            / (total + shunt[None, :])[:, :, None]
            * raw_jacobian[None, :, :]
        )
    elif update_mode == "point_gate":
        # Independent point implementation: first construct the unshunted
        # eligibility, then multiply by the sample-dependent gate realized by
        # the physical shunt.  Equality is checked on the resulting updates.
        eligibility = base_eligibility * attenuation[:, :, None]
    elif update_mode == "state_clamped_unattenuated":
        # Forward state is constructed through the shunt-plus-current clamp,
        # but the local update deliberately retains unshunted input
        # resistance.  This is the independently evaluated current control.
        eligibility = (
            x
            * (1.0 - matched_voltage)[:, :, None]
            / total[:, :, None]
            * raw_jacobian[None, :, :]
        )
    else:
        eligibility = base_eligibility
    gradient = np.mean(coefficient[:, :, None] * eligibility, axis=0)
    accuracy = float(np.mean((probability >= 0.5) == labels))
    return {
        "loss": loss,
        "accuracy": accuracy,
        "gradient": gradient,
        "eligibility": eligibility,
        "attenuation": attenuation,
        "voltage": voltage,
        "matched_voltage": matched_voltage,
        "forward_voltage": forward_voltage,
        "clamp_current": clamp_current,
        "total": total,
    }


def frozen_clamp_loss(
    q: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    center: float,
    cfg: dict,
    shunt: np.ndarray,
    clamp_current: np.ndarray,
) -> float:
    """Loss after a parameter perturbation with the clamp current held fixed."""

    branches = int(cfg["task"]["branches"])
    output_gain = float(cfg["task"]["output_gain"])
    _, excitatory, total, _ = base_state(q, x, cfg)
    voltage = (excitatory + np.asarray(clamp_current, dtype=float)) / (
        total + np.asarray(shunt, dtype=float)[None, :]
    )
    probability = sigmoid(output_gain * (voltage.mean(axis=1) - float(center)))
    tiny = 1e-12
    return -float(
        np.mean(
            labels * np.log(probability + tiny)
            + (1.0 - labels) * np.log(1.0 - probability + tiny)
        )
    )


def local_smoothness(
    q: np.ndarray, x: np.ndarray, labels: np.ndarray, center: float, cfg: dict
) -> tuple[float, float]:
    epsilon = float(cfg["training"]["finite_difference_step"])
    flat = q.ravel()
    hessian = np.empty((len(flat), len(flat)), dtype=float)
    for column in range(len(flat)):
        plus = flat.copy()
        minus = flat.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        g_plus = loss_and_gradient(
            plus.reshape(q.shape), x, labels, center, cfg
        )["gradient"].ravel()
        g_minus = loss_and_gradient(
            minus.reshape(q.shape), x, labels, center, cfg
        )["gradient"].ravel()
        hessian[:, column] = (g_plus - g_minus) / (2.0 * epsilon)
    hessian = 0.5 * (hessian + hessian.T)
    smoothness = float(np.max(np.abs(np.linalg.eigvalsh(hessian))))
    # One coordinate finite-difference check against the analytic gradient.
    direction = np.zeros_like(flat)
    direction[0] = 1.0
    plus_loss = loss_and_gradient(
        (flat + epsilon * direction).reshape(q.shape), x, labels, center, cfg
    )["loss"]
    minus_loss = loss_and_gradient(
        (flat - epsilon * direction).reshape(q.shape), x, labels, center, cfg
    )["loss"]
    numerical = (plus_loss - minus_loss) / (2.0 * epsilon)
    analytic = loss_and_gradient(q, x, labels, center, cfg)["gradient"].ravel()[0]
    relative_error = abs(numerical - analytic) / max(abs(numerical), abs(analytic), 1e-12)
    return max(smoothness, 1e-8), float(relative_error)


def initial_reliability(
    q: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    center: float,
    heterogeneity: float,
    cfg: dict,
):
    state = loss_and_gradient(q, x, labels, center, cfg)
    signal = np.sum(state["gradient"] ** 2, axis=1)
    axis = np.linspace(-1.0, 1.0, len(signal))
    target_noise = signal * np.exp(-2.0 * float(heterogeneity) * axis)
    eligibility_energy = np.sum(state["eligibility"] ** 2, axis=(0, 2))
    n = len(x)
    coefficient_sd = np.sqrt(
        np.divide(
            target_noise * n * n,
            eligibility_energy,
            out=np.zeros_like(target_noise),
            where=eligibility_energy > 0,
        )
    )
    statistical_reliability = np.divide(
        signal,
        signal + target_noise,
        out=np.ones_like(signal),
        where=(signal + target_noise) > 0,
    )
    statistical_reliability = np.maximum(
        statistical_reliability,
        float(cfg["training"]["minimum_reliability_gain"]),
    )
    step_fraction = float(cfg["training"]["smoothness_safety_factor"])
    if not 0.0 < step_fraction <= 1.0:
        raise ValueError("smoothness_safety_factor must lie in (0, 1]")
    # For eta=c/L, the constrained fixed-step optimum is
    # min(1, S/[c(S+N)]), not S/(S+N) unless c=1.
    step_optimal_gain = fixed_step_reliability_gain(
        signal, target_noise, step_fraction
    )
    global_reliability = float(signal.sum() / np.sum(signal + target_noise))
    step_optimal_global_gain = float(
        fixed_step_reliability_gain(
            np.array([signal.sum()]),
            np.array([target_noise.sum()]),
            step_fraction,
        )[0]
    )
    return (
        signal,
        target_noise,
        coefficient_sd,
        statistical_reliability,
        step_optimal_gain,
        global_reliability,
        step_optimal_global_gain,
        state,
    )


def shunt_for_gain(
    desired_gain: np.ndarray, mean_total: np.ndarray, cfg: dict
) -> np.ndarray:
    gain = np.maximum(
        np.asarray(desired_gain, dtype=float),
        float(cfg["training"]["minimum_reliability_gain"]),
    )
    return mean_total * (1.0 / gain - 1.0)


def physical_shunt_finite_difference_error(
    q: np.ndarray,
    x: np.ndarray,
    labels: np.ndarray,
    center: float,
    cfg: dict,
    shunt: np.ndarray,
) -> float:
    """Check the shunted update against a frozen-current directional derivative."""

    epsilon = float(cfg["training"]["finite_difference_step"])
    state = loss_and_gradient(
        q,
        x,
        labels,
        center,
        cfg,
        shunt=shunt,
        update_mode="physical_shunt",
    )
    direction = np.zeros_like(q)
    direction.flat[0] = 1.0
    plus = frozen_clamp_loss(
        q + epsilon * direction,
        x,
        labels,
        center,
        cfg,
        shunt,
        state["clamp_current"],
    )
    minus = frozen_clamp_loss(
        q - epsilon * direction,
        x,
        labels,
        center,
        cfg,
        shunt,
        state["clamp_current"],
    )
    numerical = (plus - minus) / (2.0 * epsilon)
    analytic = float(state["gradient"].flat[0])
    return float(
        abs(numerical - analytic)
        / max(abs(numerical), abs(analytic), 1e-12)
    )


def method_shunts(
    seed: int,
    heterogeneity: float,
    step_optimal_gain: np.ndarray,
    step_optimal_global_gain: float,
    mean_total: np.ndarray,
    cfg: dict,
):
    rng = np.random.default_rng(seed + 4_100_000 + int(round(100 * heterogeneity)))
    shuffled = step_optimal_gain[rng.permutation(len(step_optimal_gain))]
    anti = step_optimal_gain[::-1]
    zero = np.zeros_like(step_optimal_gain)
    aligned_shunt = shunt_for_gain(step_optimal_gain, mean_total, cfg)
    return {
        "exact_clean_bp": (zero, False, "unshunted"),
        "noisy_no_shunt": (zero, True, "unshunted"),
        "state_matched_additive": (aligned_shunt, True, "state_clamped_unattenuated"),
        "best_global_shunt": (
            shunt_for_gain(
                np.full_like(step_optimal_gain, step_optimal_global_gain),
                mean_total,
                cfg,
            ),
            True,
            "physical_shunt",
        ),
        "reliability_aligned_shunt": (aligned_shunt, True, "physical_shunt"),
        "shuffled_shunt": (
            shunt_for_gain(shuffled, mean_total, cfg), True, "physical_shunt"
        ),
        "anti_aligned_shunt": (
            shunt_for_gain(anti, mean_total, cfg), True, "physical_shunt"
        ),
        "explicit_point_gate": (aligned_shunt, True, "point_gate"),
        "aligned_clean_credit": (aligned_shunt, False, "physical_shunt"),
    }


def run_seed(seed: int, cfg: dict):
    rng = np.random.default_rng(seed)
    branches = int(cfg["task"]["branches"])
    features = int(cfg["task"]["features_per_branch"])
    teachers = rng.uniform(0.6, 1.4, size=(branches, features))
    teachers /= teachers.mean(axis=1, keepdims=True)
    x_train, y_train = make_data(
        rng, int(cfg["task"]["train_examples"]), cfg, teachers
    )
    x_test, y_test = make_data(
        rng, int(cfg["task"]["test_examples"]), cfg, teachers
    )
    q_initial = np.full(
        (branches, features),
        inverse_softplus(float(cfg["task"]["initial_excitatory_conductance"])),
    )
    initial_voltage = base_state(q_initial, x_train, cfg)[3]
    center = float(initial_voltage.mean())
    smoothness, finite_difference_error = local_smoothness(
        q_initial, x_train, y_train, center, cfg
    )
    step_size = float(cfg["training"]["smoothness_safety_factor"]) / smoothness
    rows, branch_rows = [], []
    maximum_state_error = 0.0
    maximum_point_error = 0.0
    maximum_additive_difference = 0.0
    maximum_physical_fd_error = 0.0
    for heterogeneity in cfg["reliability_heterogeneity"]:
        h = float(heterogeneity)
        (
            signal,
            noise_energy,
            coefficient_sd,
            statistical_reliability,
            step_optimal_gain,
            global_reliability,
            step_optimal_global_gain,
            initial,
        ) = initial_reliability(q_initial, x_train, y_train, center, h, cfg)
        mean_total = initial["total"].mean(axis=0)
        shunts = method_shunts(
            seed,
            h,
            step_optimal_gain,
            step_optimal_global_gain,
            mean_total,
            cfg,
        )
        physical_fd_error = physical_shunt_finite_difference_error(
            q_initial,
            x_train,
            y_train,
            center,
            cfg,
            shunts["reliability_aligned_shunt"][0],
        )
        maximum_physical_fd_error = max(
            maximum_physical_fd_error, physical_fd_error
        )
        noise_rng = np.random.default_rng(seed + 4_200_000 + int(round(100 * h)))
        coefficient_noises = [
            noise_rng.normal(size=(len(x_train), branches)) * coefficient_sd[None, :]
            for _ in range(int(cfg["training"]["iterations"]))
        ]
        for branch in range(branches):
            branch_rows.append(
                {
                    "seed": seed,
                    "reliability_heterogeneity": h,
                    "branch": branch,
                    "signal_energy": float(signal[branch]),
                    "noise_energy": float(noise_energy[branch]),
                    "signal_to_noise": float(signal[branch] / max(noise_energy[branch], 1e-30)),
                    "statistical_reliability": float(statistical_reliability[branch]),
                    "step_optimal_attenuation": float(step_optimal_gain[branch]),
                    # Kept for plotting compatibility; now explicitly the
                    # fixed-step optimum rather than the eta=1/L reliability.
                    "optimal_reliability_gain": float(step_optimal_gain[branch]),
                    "global_statistical_reliability": global_reliability,
                    "step_optimal_global_attenuation": step_optimal_global_gain,
                    "coefficient_noise_sd": float(coefficient_sd[branch]),
                }
            )
        method_trajectories = {}
        for method in cfg["methods"]:
            shunt, noisy, update_mode = shunts[method]
            q = q_initial.copy()
            initial_test = loss_and_gradient(q, x_test, y_test, center, cfg)
            one_step = None
            for iteration, noise in enumerate(coefficient_noises):
                used_noise = noise if noisy else None
                state = loss_and_gradient(
                    q, x_train, y_train, center, cfg,
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
                    one_step = loss_and_gradient(q, x_test, y_test, center, cfg)
            final_train = loss_and_gradient(q, x_train, y_train, center, cfg)
            final_test = loss_and_gradient(q, x_test, y_test, center, cfg)
            final_attenuation = loss_and_gradient(
                q,
                x_train,
                y_train,
                center,
                cfg,
                shunt=shunt,
                update_mode=update_mode,
            )["attenuation"]
            assert one_step is not None
            rows.append(
                {
                    "seed": seed,
                    "reliability_heterogeneity": h,
                    "method": method,
                    "step_size": step_size,
                    "objective_smoothness": smoothness,
                    "initial_test_loss": initial_test["loss"],
                    "test_loss_after_one_step": one_step["loss"],
                    "one_step_test_loss_decrease": initial_test["loss"] - one_step["loss"],
                    "final_train_loss": final_train["loss"],
                    "final_test_loss": final_test["loss"],
                    "final_test_accuracy": final_test["accuracy"],
                    "mean_initial_statistical_reliability": float(
                        np.mean(statistical_reliability)
                    ),
                    "mean_step_optimal_attenuation": float(
                        np.mean(step_optimal_gain)
                    ),
                    "step_optimal_global_attenuation": step_optimal_global_gain,
                    "mean_shunt_conductance": float(np.mean(shunt)),
                    "mean_final_physical_attenuation": float(np.mean(final_attenuation)),
                    "minimum_input_rate": float(min(x_train.min(), x_test.min())),
                }
            )
            method_trajectories[method] = q.copy()
        maximum_point_error = max(
            maximum_point_error,
            float(
                np.max(
                    np.abs(
                        method_trajectories["reliability_aligned_shunt"]
                        - method_trajectories["explicit_point_gate"]
                    )
                )
            ),
        )
        maximum_additive_difference = max(
            maximum_additive_difference,
            float(
                np.max(
                    np.abs(
                        method_trajectories["noisy_no_shunt"]
                        - method_trajectories["state_matched_additive"]
                    )
                )
            ),
        )
    gates = {
        "minimum_input_rate": float(min(x_train.min(), x_test.min())),
        "maximum_state_match_error": maximum_state_error,
        "maximum_point_gate_update_error": maximum_point_error,
        "maximum_additive_control_difference": maximum_additive_difference,
        "maximum_relative_finite_difference_error": finite_difference_error,
        "maximum_physical_shunt_finite_difference_error": maximum_physical_fd_error,
    }
    return rows, branch_rows, gates


def bootstrap(values: np.ndarray, seed: int, draws: int):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(int(draws), len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def summarize(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    metrics = [
        "test_loss_after_one_step",
        "one_step_test_loss_decrease",
        "final_test_loss",
        "final_test_accuracy",
        "mean_final_physical_attenuation",
    ]
    rows = []
    draws = int(cfg["analysis"]["bootstrap_draws"])
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
                4_300_000 + 100 * group_index + metric_index,
                draws,
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired(frame: pd.DataFrame, heterogeneity: float, left: str, right: str, metric: str, cfg: dict, seed: int):
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
        "ties": int(np.sum(np.isclose(values, 0))),
        "wilcoxon_p_two_sided": 1.0 if np.allclose(values, 0) else float(
            wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["canary", "confirmatory"], required=True)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    seeds = [int(value) for value in cfg[f"{args.phase}_seeds"]]
    all_rows, all_branches, all_gates = [], [], []
    for seed in seeds:
        rows, branches, gates = run_seed(seed, cfg)
        all_rows.extend(rows)
        all_branches.extend(branches)
        all_gates.append(gates)
    frame = pd.DataFrame(all_rows)
    branches = pd.DataFrame(all_branches)
    gate_limits = cfg["gates"]
    aggregate_gates = {
        "minimum_input_rate": float(min(g["minimum_input_rate"] for g in all_gates)),
        "maximum_state_match_error": float(max(g["maximum_state_match_error"] for g in all_gates)),
        "maximum_point_gate_update_error": float(max(g["maximum_point_gate_update_error"] for g in all_gates)),
        "maximum_additive_control_difference": float(max(g["maximum_additive_control_difference"] for g in all_gates)),
        "maximum_relative_finite_difference_error": float(max(g["maximum_relative_finite_difference_error"] for g in all_gates)),
        "maximum_physical_shunt_finite_difference_error": float(max(g["maximum_physical_shunt_finite_difference_error"] for g in all_gates)),
    }
    aggregate_gates["passed"] = bool(
        aggregate_gates["minimum_input_rate"] >= float(gate_limits["minimum_input"])
        and aggregate_gates["maximum_state_match_error"] <= float(gate_limits["maximum_state_match_error"])
        and aggregate_gates["maximum_point_gate_update_error"] <= float(gate_limits["maximum_point_gate_update_error"])
        and aggregate_gates["maximum_additive_control_difference"] <= float(gate_limits["maximum_additive_control_difference"])
        and aggregate_gates["maximum_relative_finite_difference_error"] <= float(gate_limits["maximum_relative_finite_difference_error"])
        and aggregate_gates["maximum_physical_shunt_finite_difference_error"] <= float(gate_limits["maximum_physical_shunt_finite_difference_error"])
    )
    hashes = {
        "config_sha256": digest(CONFIG),
        "script_sha256": digest(Path(__file__).resolve()),
    }
    if args.phase == "canary":
        payload = {
            "phase": "artifact_only_canary",
            **hashes,
            "numerical_gates": aggregate_gates,
            "outcomes_are_not_confirmatory": True,
        }
        CANARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        if not aggregate_gates["passed"]:
            raise SystemExit("canary failed")
        return
    if not CANARY.is_file():
        raise SystemExit("passing canary required")
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    if not canary["numerical_gates"]["passed"] or any(
        canary[key] != value for key, value in hashes.items()
    ):
        raise SystemExit("canary failed or code/configuration changed")
    expected = (
        len(seeds)
        * len(cfg["reliability_heterogeneity"])
        * len(cfg["methods"])
    )
    if len(frame) != expected or frame.duplicated(
        ["seed", "reliability_heterogeneity", "method"]
    ).any():
        raise SystemExit(f"incomplete reliability factorial: {len(frame)} != {expected}")
    if not np.isfinite(frame.select_dtypes(include=[np.number])).all().all():
        raise SystemExit("non-finite outcome")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "seed_outcomes.csv", index=False, float_format="%.10g")
    branches.to_csv(OUTPUT / "branch_reliability.csv", index=False, float_format="%.10g")
    summarize(frame, cfg).to_csv(
        OUTPUT / "condition_summary.csv", index=False, float_format="%.10g"
    )
    high = float(max(cfg["reliability_heterogeneity"]))
    contrasts = [
        paired(frame, high, "reliability_aligned_shunt", "best_global_shunt", "one_step_test_loss_decrease", cfg, 4_400_001),
        paired(frame, high, "reliability_aligned_shunt", "best_global_shunt", "final_test_loss", cfg, 4_400_002),
        paired(frame, high, "reliability_aligned_shunt", "shuffled_shunt", "final_test_loss", cfg, 4_400_003),
        paired(frame, high, "reliability_aligned_shunt", "anti_aligned_shunt", "final_test_loss", cfg, 4_400_004),
        paired(frame, 0.0, "reliability_aligned_shunt", "best_global_shunt", "final_test_loss", cfg, 4_400_005),
        paired(frame, high, "reliability_aligned_shunt", "explicit_point_gate", "final_test_loss", cfg, 4_400_006),
    ]
    pd.DataFrame(contrasts).to_csv(
        OUTPUT / "paired_contrasts.csv", index=False, float_format="%.10g"
    )
    summary = {
        "study": cfg["study"],
        "status": "complete_confirmatory",
        "n_seeds": len(seeds),
        "row_count": len(frame),
        "python": platform.python_version(),
        "numpy": np.__version__,
        **hashes,
        "numerical_gates": aggregate_gates,
        "scope_boundary": cfg["scope_boundary"],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

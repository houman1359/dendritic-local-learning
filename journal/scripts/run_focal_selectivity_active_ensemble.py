#!/usr/bin/env python3
"""Run the frozen active steady-state focal-credit sensitivity ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
RECON = ROOT / "code" / "reconstructed_tree"
if str(RECON) not in sys.path:
    sys.path.insert(0, str(RECON))

from analyze_physical_cable_sensitivity import physical_conductance_system  # noqa: E402
from run_focal_shunting_credit_perturbation import choose_focal_sites, stable_rng  # noqa: E402
from run_focal_selectivity_phase1 import relation_metrics  # noqa: E402


CONFIG = ROOT / "configs" / "focal_selectivity" / "active_linearization_ensemble.json"
SEGMENTS = ROOT / "source_data" / "figure3" / "segment_metrics.csv"
SOURCE = ROOT / "source_data" / "focal_selectivity_active_ensemble"
CANARY = ROOT / "analysis" / "focal_selectivity_active_ensemble_canary.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def gate(values: np.ndarray, v_half: float, slope: float, power: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip((values - v_half) / slope, -60.0, 60.0)
    activation = 1.0 / (1.0 + np.exp(-z))
    derivative = activation * (1.0 - activation) / slope
    powered = activation**power
    powered_derivative = power * activation ** (power - 1) * derivative
    return powered, powered_derivative


def active_current_jacobian(
    voltage: np.ndarray, conductances: dict[str, np.ndarray], channels: dict
) -> tuple[np.ndarray, np.ndarray]:
    current = np.zeros_like(voltage)
    diagonal = np.zeros_like(voltage)
    for name, specification in channels.items():
        activation, derivative = gate(
            voltage,
            float(specification["v_half"]),
            float(specification["slope"]),
            int(specification["power"]),
        )
        reversal = float(specification["reversal"])
        gbar = conductances[name]
        current += gbar * activation * (voltage - reversal)
        diagonal += gbar * (activation + derivative * (voltage - reversal))
    return current, diagonal


def solve_active_state(
    passive: np.ndarray,
    rhs: np.ndarray,
    conductances: dict[str, np.ndarray],
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, float, int] | None:
    voltage = np.linalg.solve(passive, rhs)
    solver = cfg["solver"]
    for iteration in range(int(solver["maximum_iterations"])):
        current, active_diagonal = active_current_jacobian(
            voltage, conductances, cfg["channels"]
        )
        residual = passive @ voltage - rhs + current
        residual_norm = float(np.max(np.abs(residual)))
        jacobian = passive + np.diag(active_diagonal)
        if residual_norm <= float(solver["residual_tolerance"]):
            minimum = float(np.linalg.eigvalsh(jacobian)[0])
            if (
                minimum > float(solver["minimum_jacobian_eigenvalue"])
                and voltage.min() >= float(solver["minimum_voltage"])
                and voltage.max() <= float(solver["maximum_voltage"])
            ):
                return voltage, jacobian, residual_norm, iteration
            return None
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            return None
        baseline = float(np.linalg.norm(residual))
        accepted = False
        fraction = 1.0
        for _ in range(20):
            candidate = voltage + fraction * step
            candidate_current, _ = active_current_jacobian(
                candidate, conductances, cfg["channels"]
            )
            candidate_residual = passive @ candidate - rhs + candidate_current
            if float(np.linalg.norm(candidate_residual)) < baseline:
                voltage = candidate
                accepted = True
                break
            fraction *= 0.5
        if not accepted:
            return None
    return None


def sample_conductances(
    rng: np.random.Generator, electrical: pd.DataFrame, cfg: dict
) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    leak = electrical.g_leak.to_numpy(float)
    excitatory = electrical.g_e.to_numpy(float)
    positive_e = excitatory[excitatory > 0]
    e_floor = float(np.median(positive_e)) if len(positive_e) else float(np.median(leak))
    excitatory_reference = np.maximum(excitatory, 0.05 * e_floor)
    for name, specification in cfg["channels"].items():
        reference = leak if specification["reference"] == "leak" else excitatory_reference
        global_scale = np.exp(float(cfg["channel_global_log_sd"]) * rng.normal())
        spatial_scale = np.exp(
            float(cfg["channel_spatial_log_sd"]) * rng.normal(size=len(electrical))
        )
        values[name] = (
            float(specification["median_scale"])
            * global_scale
            * spatial_scale
            * reference
        )
    return values


def intervention(
    inverse: np.ndarray,
    voltage: np.ndarray,
    soma_index: int,
    focal_index: int,
    eta: float,
    reversal: float,
    perturbation: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    column = inverse[:, focal_index]
    soma_column = inverse[:, soma_index]
    drive = eta * (reversal - voltage[focal_index])
    if perturbation == "focal shunt":
        denominator = 1.0 + eta * inverse[focal_index, focal_index]
        changed_voltage = voltage + drive * column / denominator
        soma_response = soma_column - eta * column * column[soma_index] / denominator
        adjoint = soma_response
    elif perturbation == "matched additive":
        changed_voltage = voltage + drive * column
        soma_response = soma_column
        adjoint = soma_column
    else:
        raise ValueError(perturbation)
    compensating_current = (voltage[soma_index] - changed_voltage[soma_index]) / soma_response[soma_index]
    changed_voltage = changed_voltage + compensating_current * soma_response
    residual = abs(changed_voltage[soma_index] - voltage[soma_index])
    return changed_voltage, adjoint, float(residual)


def bootstrap(values: np.ndarray, seed: int, draws: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def run_cell(root_id: int, segments: pd.DataFrame, cfg: dict, *, canary: bool) -> tuple[list[dict], dict]:
    electrical, passive, rhs, soma_index, parents, children = physical_conductance_system(
        segments,
        float(cfg["e_scale"]),
        float(cfg["i_scale"]),
        float(cfg["excitatory_reversal"]),
        float(cfg["inhibitory_reversal"]),
        axial_resistivity_ohm_cm=float(cfg["axial_resistivity_ohm_cm"]),
        membrane_resistance_ohm_cm2=float(cfg["membrane_resistance_ohm_cm2"]),
    )
    background = float(cfg["background_leak_multiplier"]) * electrical.g_leak.to_numpy(float)
    passive[np.diag_indices(len(passive))] += background
    rhs += background * float(cfg["background_reversal"])
    identifiers = electrical.segment_id.astype(int).tolist()
    index = {segment: position for position, segment in enumerate(identifiers)}
    e_segments = electrical.loc[electrical.E_size > 0, "segment_id"].astype(int).tolist()
    e_indices = np.asarray([index[segment] for segment in e_segments], dtype=int)
    focal_sites, relations = choose_focal_sites(
        electrical,
        e_segments,
        parents,
        children,
        stable_rng(int(cfg["seed"]), root_id, stream=47),
        1 if canary else int(cfg["max_focal_sites_per_cell"]),
        int(cfg["minimum_descendant_and_comparison_sites"]),
    )
    target_draws = 1 if canary else int(cfg["accepted_draws_per_cell"])
    rng = stable_rng(int(cfg["seed"]), root_id, stream=53)
    rows: list[dict] = []
    accepted = 0
    attempts = 0
    while accepted < target_draws and attempts < int(cfg["maximum_attempts_per_cell"]):
        conductances = sample_conductances(rng, electrical, cfg)
        solved = solve_active_state(passive, rhs, conductances, cfg)
        attempts += 1
        if solved is None:
            continue
        voltage, jacobian, state_residual, iterations = solved
        inverse = np.linalg.inv(jacobian)
        baseline_adjoint = inverse[:, soma_index]
        baseline_gradient = baseline_adjoint[e_indices] * (
            float(cfg["excitatory_reversal"]) - voltage[e_indices]
        )
        for focal in focal_sites:
            focal_index = index[int(focal)]
            local_input_conductance = 1.0 / inverse[focal_index, focal_index]
            descendant_indices = relations[focal]["descendant"]
            comparison_indices = relations[focal]["depth-matched unrelated"]
            cable = np.abs(inverse[e_indices, focal_index])
            selectivity = float(
                np.median(cable[descendant_indices])
                / max(float(np.median(cable[comparison_indices])), 1e-30)
            )
            for dose in cfg["dose_relative_to_local_input_conductance"]:
                eta = float(dose) * local_input_conductance
                for perturbation in ("matched additive", "focal shunt"):
                    changed_voltage, adjoint, soma_residual = intervention(
                        inverse,
                        voltage,
                        soma_index,
                        focal_index,
                        eta,
                        float(cfg["inhibitory_reversal"]),
                        perturbation,
                    )
                    changed_gradient = adjoint[e_indices] * (
                        float(cfg["excitatory_reversal"]) - changed_voltage[e_indices]
                    )
                    descendant = relation_metrics(
                        baseline_gradient, changed_gradient, descendant_indices
                    )
                    comparison = relation_metrics(
                        baseline_gradient, changed_gradient, comparison_indices
                    )
                    row = {
                        "root_id": root_id,
                        "draw": accepted,
                        "attempt": attempts - 1,
                        "focal_segment_id": int(focal),
                        "dose_relative_to_local_input_conductance": float(dose),
                        "delta_conductance_ns": eta,
                        "perturbation": perturbation,
                        "transport_selectivity": selectivity,
                        "state_residual": state_residual,
                        "soma_state_residual": soma_residual,
                        "jacobian_minimum_eigenvalue": float(np.linalg.eigvalsh(jacobian)[0]),
                        "solver_iterations": iterations,
                        "baseline_voltage_minimum": float(voltage.min()),
                        "baseline_voltage_maximum": float(voltage.max()),
                    }
                    for name, value in descendant.items():
                        row[f"descendant_{name}"] = value
                    for name, value in comparison.items():
                        row[f"matched_{name}"] = value
                    row["localization_index"] = (
                        descendant["median_abs_log_change"]
                        - comparison["median_abs_log_change"]
                    )
                    row["signed_localization"] = (
                        descendant["median_signed_log_change"]
                        - comparison["median_signed_log_change"]
                    )
                    rows.append(row)
        accepted += 1
    return rows, {
        "root_id": root_id,
        "attempts": attempts,
        "accepted_draws": accepted,
        "acceptance_fraction": accepted / max(attempts, 1),
        "n_focal_sites": len(focal_sites),
    }


def summarize(
    frame: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = [
        "localization_index",
        "signed_localization",
        "descendant_attenuated_fraction",
        "descendant_enhanced_fraction",
        "descendant_sign_flip_fraction",
        "descendant_gradient_energy_ratio",
        "transport_selectivity",
    ]
    draw = frame.groupby(
        ["root_id", "draw", "dose_relative_to_local_input_conductance", "perturbation"],
        as_index=False,
    )[metrics].mean()
    cell = draw.groupby(
        ["root_id", "dose_relative_to_local_input_conductance", "perturbation"],
        as_index=False,
    )[metrics].mean()
    summary_rows: list[dict] = []
    contrasts: list[dict] = []
    for condition_index, (keys, part) in enumerate(
        cell.groupby(["dose_relative_to_local_input_conductance", "perturbation"], sort=True)
    ):
        dose, perturbation = keys
        row = {"dose_relative_to_local_input_conductance": dose, "perturbation": perturbation, "n_cells": int(part.root_id.nunique())}
        for metric_index, metric in enumerate(metrics):
            mean, low, high = bootstrap(
                part[metric].to_numpy(float),
                210_000 + 20 * condition_index + metric_index,
                int(cfg["analysis"]["bootstrap_draws"]),
            )
            row[f"mean_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
        summary_rows.append(row)
    for dose_index, dose in enumerate(cfg["dose_relative_to_local_input_conductance"]):
        part = cell[np.isclose(cell.dose_relative_to_local_input_conductance, dose)]
        for metric_index, metric in enumerate(metrics[:-1]):
            wide = part.pivot(index="root_id", columns="perturbation", values=metric)
            values = (wide["focal shunt"] - wide["matched additive"]).to_numpy(float)
            mean, low, high = bootstrap(values, 220_000 + 20 * dose_index + metric_index, int(cfg["analysis"]["bootstrap_draws"]))
            pvalue = 1.0 if np.allclose(values, 0) else float(wilcoxon(values, alternative="two-sided").pvalue)
            contrasts.append({
                "dose_relative_to_local_input_conductance": float(dose),
                "metric": metric,
                "n_cells": len(values),
                "mean_shunt_minus_additive": mean,
                "ci95_low": low,
                "ci95_high": high,
                "cells_positive": int(np.sum(values > 0)),
                "wilcoxon_p_two_sided": pvalue,
            })
    return draw, cell, pd.DataFrame(summary_rows), pd.DataFrame(contrasts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["canary", "confirmatory"], required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    segments = pd.read_csv(SEGMENTS)
    roots = [int(value) for value in sorted(segments.root_id.unique())]
    if args.phase == "canary":
        roots = roots[:1]
    requests = [
        (root_id, segments[segments.root_id.eq(root_id)].copy())
        for root_id in roots
    ]
    if args.workers > 1 and len(requests) > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(requests))) as pool:
            futures = [
                pool.submit(run_cell, root_id, cell, cfg, canary=False)
                for root_id, cell in requests
            ]
            results = [future.result() for future in futures]
    else:
        results = [
            run_cell(
                root_id,
                cell,
                cfg,
                canary=args.phase == "canary",
            )
            for root_id, cell in requests
        ]
    rows = [row for cell_rows, _ in results for row in cell_rows]
    acceptance = [cell_acceptance for _, cell_acceptance in results]
    frame = pd.DataFrame(rows)
    accepted = pd.DataFrame(acceptance)
    gates = cfg["gates"]
    gate = {
        "rows": int(len(frame)),
        "all_finite": bool(not frame.empty and np.isfinite(frame.select_dtypes(include=[np.number])).all().all()),
        "maximum_state_residual": float(frame.state_residual.max()),
        "maximum_soma_state_residual": float(frame.soma_state_residual.max()),
        "minimum_jacobian_eigenvalue": float(frame.jacobian_minimum_eigenvalue.min()),
        "minimum_acceptance_fraction": float(accepted.acceptance_fraction.min()),
        "minimum_accepted_draws_per_cell": int(accepted.accepted_draws.min()),
    }
    expected_draws = 1 if args.phase == "canary" else int(gates["minimum_accepted_draws_per_cell"])
    gate["passed"] = bool(
        gate["all_finite"]
        and gate["maximum_state_residual"] <= float(gates["maximum_state_residual"])
        and gate["maximum_soma_state_residual"] <= float(gates["maximum_soma_state_residual"])
        and gate["minimum_jacobian_eigenvalue"] > float(cfg["solver"]["minimum_jacobian_eigenvalue"])
        and gate["minimum_acceptance_fraction"] >= float(gates["minimum_acceptance_fraction"])
        and gate["minimum_accepted_draws_per_cell"] >= expected_draws
    )
    if args.phase == "canary":
        payload = {
            "phase": "canary",
            "config_sha256": digest(CONFIG),
            "script_sha256": digest(Path(__file__).resolve()),
            "artifact_and_numerical_gates": gate,
            "outcomes_are_not_confirmatory": True,
        }
        CANARY.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2))
        if not gate["passed"]:
            raise SystemExit("canary failed")
        return
    if not CANARY.is_file():
        raise SystemExit("passing canary required")
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    if not canary["artifact_and_numerical_gates"]["passed"]:
        raise SystemExit("canary did not pass")
    if canary["config_sha256"] != digest(CONFIG) or canary["script_sha256"] != digest(Path(__file__).resolve()):
        raise SystemExit("code or configuration changed after canary")
    if not gate["passed"]:
        raise SystemExit("confirmatory numerical or completeness gate failed")
    SOURCE.mkdir(parents=True, exist_ok=True)
    draw, cell, summary, contrasts = summarize(frame, cfg)
    frame.to_csv(SOURCE / "site_draw_outcomes.csv.gz", index=False, compression="gzip", float_format="%.10g")
    draw.to_csv(SOURCE / "draw_condition_metrics.csv", index=False, float_format="%.10g")
    cell.to_csv(SOURCE / "cell_condition_metrics.csv", index=False, float_format="%.10g")
    summary.to_csv(SOURCE / "condition_summary.csv", index=False, float_format="%.10g")
    contrasts.to_csv(SOURCE / "paired_contrasts.csv", index=False, float_format="%.10g")
    accepted.to_csv(SOURCE / "acceptance_ledger.csv", index=False)
    metadata = {
        "study": cfg["study"],
        "status": "complete_confirmatory",
        "scope_boundary": cfg["scope_boundary"],
        "config_sha256": digest(CONFIG),
        "script_sha256": digest(Path(__file__).resolve()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_cells": int(frame.root_id.nunique()),
        "accepted_draws_per_cell": int(cfg["accepted_draws_per_cell"]),
        "n_accepted_cell_draws": int(frame[["root_id", "draw"]].drop_duplicates().shape[0]),
        "n_focal_sites": int(frame[["root_id", "focal_segment_id"]].drop_duplicates().shape[0]),
        "n_site_draw_condition_rows": int(len(frame)),
        "numerical_gates": gate,
    }
    (SOURCE / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

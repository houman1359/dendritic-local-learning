#!/usr/bin/env python3
"""Run the frozen conservative adaptive-shunting dose pilot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ADAPTIVE_PATH = ROOT / "scripts" / "run_adaptive_conductance_reliability.py"
PILOT_CONFIG = (
    ROOT
    / "configs"
    / "positive_conductance_reliability"
    / "conservative_adaptive_pilot.json"
)
CANARY = ROOT / "analysis" / "conservative_adaptive_shunting_pilot_canary.json"
OUTPUT = ROOT / "source_data" / "conservative_adaptive_shunting_pilot"


def _load_adaptive():
    spec = importlib.util.spec_from_file_location(
        "adaptive_reliability_for_conservative_pilot", ADAPTIVE_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {ADAPTIVE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ADAPTIVE = _load_adaptive()
ORIGINAL_GAIN_FROM_MOMENTS = ADAPTIVE.gain_from_moments


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def conservative_gain_from_moments(
    signal: np.ndarray,
    noise: np.ndarray,
    cfg: dict,
    *,
    global_gain: bool = False,
) -> np.ndarray:
    """Shrink the plug-in attenuation toward the unshunted gain of one."""
    raw = ORIGINAL_GAIN_FROM_MOMENTS(
        signal,
        noise,
        cfg,
        global_gain=global_gain,
    )
    dose = float(cfg["estimator"]["attenuation_dose"])
    if not 0.0 <= dose <= 1.0:
        raise ValueError(f"attenuation dose outside [0, 1]: {dose}")
    return 1.0 - dose * (1.0 - raw)


def _bootstrap(values: np.ndarray, *, seed: int, draws: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def _run_phase(phase: str) -> tuple[pd.DataFrame, list[dict]]:
    pilot = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    base = json.loads(ADAPTIVE.CONFIG.read_text(encoding="utf-8"))
    seeds = pilot[f"{phase}_seeds"]
    rows: list[dict] = []
    gates: list[dict] = []
    ADAPTIVE.gain_from_moments = conservative_gain_from_moments
    try:
        for dose in pilot["attenuation_doses"]:
            cfg = json.loads(json.dumps(base))
            cfg["estimator"]["attenuation_dose"] = float(dose)
            for seed in seeds:
                seed_rows, _estimators, seed_gates = ADAPTIVE.run_seed(int(seed), cfg)
                for row in seed_rows:
                    row["attenuation_dose"] = float(dose)
                    rows.append(row)
                gates.append({"attenuation_dose": float(dose), **seed_gates})
    finally:
        ADAPTIVE.gain_from_moments = ORIGINAL_GAIN_FROM_MOMENTS
    return pd.DataFrame(rows), gates


def _aggregate_gates(gates: list[dict], base: dict) -> dict:
    combined = ADAPTIVE.aggregate_gates(gates, base)
    combined["tested_doses"] = sorted(
        {float(item["attenuation_dose"]) for item in gates}
    )
    return combined


def _audit_repeated_controls(frame: pd.DataFrame) -> dict:
    controls = ("exact_clean_bp", "noisy_no_shunt", "initial_oracle_shunt")
    maximum = 0.0
    for method in controls:
        part = frame[frame.method.eq(method)]
        for metric in (
            "one_step_test_loss_decrease",
            "final_test_loss",
            "final_test_accuracy",
        ):
            wide = part.pivot_table(
                index=["seed", "reliability_heterogeneity"],
                columns="attenuation_dose",
                values=metric,
                aggfunc="first",
            )
            maximum = max(
                maximum,
                float(np.max(np.abs(wide.to_numpy() - wide.iloc[:, [0]].to_numpy()))),
            )
    return {
        "maximum_repeated_control_difference": maximum,
        "passed": bool(maximum <= 1e-14),
    }


def _analyze(frame: pd.DataFrame, pilot: dict) -> tuple[pd.DataFrame, dict]:
    high = float(pilot["primary_heterogeneity"])
    part = frame[np.isclose(frame.reliability_heterogeneity, high)]
    local = part[part.method.eq("adaptive_local_shunt")].pivot(
        index="seed", columns="attenuation_dose", values="final_test_loss"
    )
    no_shunt = (
        part[part.method.eq("noisy_no_shunt")]
        .drop_duplicates(["seed"])
        .set_index("seed")["final_test_loss"]
    )
    rows = []
    for index, dose in enumerate(sorted(local.columns)):
        difference = local[dose] - no_shunt
        low, high_ci = _bootstrap(
            difference.to_numpy(),
            seed=8_600_000 + index,
            draws=int(pilot["bootstrap_draws"]),
        )
        rows.append(
            {
                "attenuation_dose": float(dose),
                "n_seeds": int(len(difference)),
                "mean_final_loss": float(local[dose].mean()),
                "mean_loss_minus_no_shunt": float(difference.mean()),
                "ci95_low_loss_minus_no_shunt": low,
                "ci95_high_loss_minus_no_shunt": high_ci,
                "seeds_improved_over_no_shunt": int(np.sum(difference < 0)),
            }
        )
    summary = pd.DataFrame(rows)
    positive = summary[summary.attenuation_dose.gt(0)].sort_values(
        "mean_loss_minus_no_shunt"
    )
    best = positive.iloc[0]
    eligible = bool(
        best.mean_loss_minus_no_shunt
        <= -float(pilot["minimum_mean_loss_improvement"])
        and int(best.seeds_improved_over_no_shunt)
        >= int(pilot["minimum_improved_pairs"])
    )
    decision = {
        "best_positive_dose": float(best.attenuation_dose),
        "best_mean_loss_minus_no_shunt": float(best.mean_loss_minus_no_shunt),
        "best_seeds_improved_over_no_shunt": int(best.seeds_improved_over_no_shunt),
        "eligible_for_fresh_seed_confirmation": eligible,
        "decision_rule": {
            "minimum_mean_loss_improvement": float(
                pilot["minimum_mean_loss_improvement"]
            ),
            "minimum_improved_pairs": int(pilot["minimum_improved_pairs"]),
        },
    }
    return summary, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("canary", "pilot"), required=True)
    args = parser.parse_args()
    pilot = json.loads(PILOT_CONFIG.read_text(encoding="utf-8"))
    base = json.loads(ADAPTIVE.CONFIG.read_text(encoding="utf-8"))
    hashes = {
        "pilot_config_sha256": digest(PILOT_CONFIG),
        "pilot_script_sha256": digest(Path(__file__).resolve()),
        "adaptive_script_sha256": digest(ADAPTIVE_PATH),
        "adaptive_config_sha256": digest(ADAPTIVE.CONFIG),
        "positive_conductance_script_sha256": digest(ADAPTIVE.BASE_PATH),
    }
    frame, seed_gates = _run_phase(args.phase)
    gates = _aggregate_gates(seed_gates, base)
    repeated = _audit_repeated_controls(frame)
    if not gates["passed"] or not repeated["passed"]:
        raise SystemExit(f"pilot numerical gate failed: {gates}; {repeated}")

    if args.phase == "canary":
        payload = {
            "phase": "artifact_only_canary",
            **hashes,
            "numerical_gates": gates,
            "repeated_control_audit": repeated,
            "outcomes_are_not_pilot_or_confirmatory": True,
        }
        CANARY.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    if not CANARY.is_file():
        raise SystemExit("passing canary required before the pilot")
    canary = json.loads(CANARY.read_text(encoding="utf-8"))
    if any(canary.get(key) != value for key, value in hashes.items()):
        raise SystemExit("pilot files changed after canary")
    expected = (
        len(pilot["pilot_seeds"])
        * len(pilot["attenuation_doses"])
        * len(base["reliability_heterogeneity"])
        * len(base["methods"])
    )
    if len(frame) != expected:
        raise SystemExit(f"incomplete pilot: {len(frame)} != {expected}")
    summary, decision = _analyze(frame, pilot)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "seed_outcomes.csv", index=False, float_format="%.10g")
    summary.to_csv(OUTPUT / "dose_summary.csv", index=False, float_format="%.10g")
    payload = {
        "study": pilot["study"],
        "status": "complete_exploratory_pilot",
        "row_count": int(len(frame)),
        "n_pilot_seeds": int(len(pilot["pilot_seeds"])),
        **hashes,
        "numerical_gates": gates,
        "repeated_control_audit": repeated,
        "selection_decision": decision,
        "scope_boundary": pilot["scope_boundary"],
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

